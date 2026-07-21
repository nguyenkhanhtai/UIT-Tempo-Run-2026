"""Stage 3 — text->keyframe retrieval -> submission.json.

Loads all per-video shards into one keyframe index, embeds each task description
with CLIP text encoder, finds the most similar keyframes (chunked top-k on GPU),
dedups to distinct videos, and emits up to `max_predictions` (video_id, frame_ms) predictions
per task.

Added features (Merged):
- User's Temporal Smoothing, Dual Softmax, Causal Bonus.
- OOP model loading.
- OCR Hard Filtering.
- KMeans Clustering for Diversity.
- LLM Router for intelligent Modality Fusion.

Refactored to use modular architecture under `pipeline/retrieval/`.
"""
import argparse
import glob
import os
import json
import shutil
import collections
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from retrieval.loader import load_index
from retrieval.temporal import smooth_features
from retrieval.parser import parse_queries
from retrieval.scorer import compute_similarity, aggregate_scores
from retrieval.postprocess import apply_reranking
from models.factory import get_embedding_model

def route_asr_queries(tasks, engine_name="qwen"):
    """Attach LLM ASR routes to tasks and return (router, routes)."""
    try:
        from pipeline.retrieval.routing.lm_router import LMRouter
        lm_router = LMRouter(engine_name=engine_name)
        routes = lm_router.route_batch([task["description"] for task in tasks])
    except Exception as e:
        print(f"[retrieve] Warning: Could not load LMRouter: {e}")
        lm_router = None
        routes = [{"Use_asr": False, "asr_query": None} for _ in tasks]

    for task, route in zip(tasks, routes):
        task["route"] = route

    return lm_router, routes

def build_asr_queries(tasks, routes):
    audio_queries = []
    audio_task_mapping = []

    for ti, (task, route) in enumerate(zip(tasks, routes)):
        asr_query = route.get("asr_query")
        if route.get("Use_asr", False) and isinstance(asr_query, str) and asr_query.strip():
            audio_queries.append(asr_query.strip())
            audio_task_mapping.append((ti, 0, 0))
        elif route.get("Use_asr", False):
            task["route"] = {"Use_asr": False, "asr_query": None}

    return audio_queries, audio_task_mapping


def process_visual_modality(args, tasks, task_mapping, all_queries, dev):
    all_models_Q = []
    all_models_idx = []
    
    first_vids, first_ts, first_metadata, first_emb = None, None, None, None
    first_clip_model_name = None
    
    # Flatten expanded queries
    expanded_queries_flat = []
    for t in tasks:
        expanded_queries_flat.extend(t.get("expanded_queries", [t["description"]] * 5))
    expanded_Q_embs = None

    for idx_vlm, vlm_str in enumerate(args.vlms):
        model_name, pretrained = vlm_str.split(",")
        model_safe = model_name.replace("/", "_")
        pre_safe = pretrained.replace("/", "_") if pretrained else "none"
        
        shard_dir = Path(args.visual_shards) / f"{model_safe}_{pre_safe}"
        
        # 1. Load Data
        print(f"[retrieve] Loading index for {vlm_str} from {shard_dir}...")
        emb, vids, ts, metadata = load_index(str(shard_dir), args.metadata if first_metadata is None else None)
        
        if first_vids is None:
            first_vids, first_ts, first_metadata, first_emb = vids, ts, metadata, emb
        else:
            if not np.array_equal(first_vids, vids) or not np.array_equal(first_ts, ts):
                raise ValueError(f"Shards for {vlm_str} do not match the keyframe alignment of the first model!")
                
        # 2. Temporal Smoothing
        emb_smoothed = smooth_features(emb, vids, smoothing_window=args.smoothing_window, sigma=args.smoothing_sigma)
        all_models_idx.append(emb_smoothed)
        
        # 3. Encode Queries
        clip = get_embedding_model(model_name, pretrained, device=args.device, precision=args.precision)
        Q = clip.encode_texts(all_queries)      # [T_all, D] fp32
        all_models_Q.append(torch.from_numpy(Q).to(dev).float())
        
        # Encode Expanded Queries ONLY for the first model
        if idx_vlm == 0 and expanded_queries_flat:
            print(f"[retrieve] Encoding expanded queries with {vlm_str}...")
            Q_exp = clip.encode_texts(expanded_queries_flat)
            expanded_Q_embs = torch.from_numpy(Q_exp).to(dev).float()
            
        # Free up RAM/VRAM
        del clip
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 4. Compute GPU Similarity
    T_all, N = all_models_Q[0].shape[0], all_models_idx[0].shape[0]
    K = min(args.cand_keyframes, N)
    if args.use_sequential:
        K = N
    
    top_idx, top_val = compute_similarity(all_models_Q, all_models_idx, T_all, N, K, dev)

    # 5. Aggregate Scores
    all_candidates_visual = aggregate_scores(
        task_mapping, top_idx, top_val, tasks, first_vids, first_ts, first_emb, first_metadata, args.use_sequential
    )
    
    return all_candidates_visual, first_vids, first_ts, first_metadata, first_emb, T_all, N, K, expanded_Q_embs

def process_audio_modality(args, tasks, audio_task_mapping, audio_queries, first_vids, first_ts, first_metadata, dev, N, K):
    if not audio_queries:
        print("[retrieve] No valid ASR queries from LMRouter; skipping Audio Modality.")
        return [(task, []) for task in tasks]

    print(f"[retrieve] Loading Audio Modality for {len(audio_queries)} ASR queries...")
    audio_model_safe = args.audio_model.replace("/", "_")
    audio_shard_dir = os.path.join(args.audio_shards, audio_model_safe)
    
    if not os.path.exists(audio_shard_dir):
        raise SystemExit(f"[retrieve] Lỗi: Không tìm thấy thư mục {audio_shard_dir}. Hãy chạy scripts/extract_audio_features.sh trước!")
        
    audio_emb, audio_vids, audio_ts, _ = load_index(audio_shard_dir, meta_dir=None)
    
    # Verify alignment
    if not np.array_equal(first_vids, audio_vids) or not np.array_equal(first_ts, audio_ts):
        raise ValueError(f"[retrieve] Lỗi: Audio features không đồng bộ với Visual features! Hãy chạy lại extract_audio_features.sh")
        
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("[retrieve] Please run: uv pip install sentence-transformers")
        
    print(f"[retrieve] Loading Audio Embedding Model: {args.audio_model}")
    audio_model = SentenceTransformer(args.audio_model, device=dev)
    
    print("[retrieve] Encoding ASR Queries for Audio...")
    Q_audio = audio_model.encode(audio_queries, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True).to(dev).float()

    print("[retrieve] Computing GPU Similarity for Audio...")
    top_idx_a, top_val_a = compute_similarity([Q_audio], [audio_emb], len(audio_queries), N, K, dev)

    print("[retrieve] Aggregating Scores for Audio...")
    subset_tasks = [tasks[ti] for ti, _, _ in audio_task_mapping]
    subset_mapping = [(i, sent_idx, seg_idx) for i, (_, sent_idx, seg_idx) in enumerate(audio_task_mapping)]
    subset_candidates = aggregate_scores(
        subset_mapping, top_idx_a, top_val_a, subset_tasks, first_vids, first_ts, audio_emb, first_metadata, False
    )

    all_candidates_audio = [(task, []) for task in tasks]
    for (ti, _, _), (_, candidates) in zip(audio_task_mapping, subset_candidates):
        all_candidates_audio[ti] = (tasks[ti], candidates)

    return all_candidates_audio



def save_submission(args, preds):
    """Save submission.json and config.json; return the submission directory (or None)."""
    # Clean up auxiliary fields before saving to avoid format issues
    clean_preds = []
    for pred in preds:
        clean_results = []
        for c in pred.get("results", []):
            # Keep rank, video_id and frame_ms for the final submission
            clean_c = {"rank": c.get("rank"), "video_id": c.get("video_id"), "frame_ms": c.get("frame_ms")}
            clean_results.append(clean_c)
        clean_preds.append({"task_id": pred["task_id"], "results": clean_results})
        
    sub = {"predictions": clean_preds}
    out_path = args.out
    out_dir = None
    
    if out_path == "submission.json":
        os.makedirs("submission", exist_ok=True)
        sub_dirs = glob.glob("submission/*")
        valid_dirs = [d for d in sub_dirs if os.path.basename(d).isdigit()]
        if not valid_dirs:
            next_id = 1
        else:
            valid_dirs.sort(key=lambda x: int(os.path.basename(x)))
            next_id = int(os.path.basename(valid_dirs[-1])) + 1
            
        out_dir = os.path.join("submission", f"{next_id:03d}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "submission.json")
    else:
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir, exist_ok=True)

    json.dump(sub, open(out_path, "w"))
    
    # Save parameters
    config_data = {
        "args": vars(args),
        "env_params": {
            "MAIN_QUERY_WEIGHT": os.environ.get("MAIN_QUERY_WEIGHT"),
            "SPLIT_QUERY": os.environ.get("SPLIT_QUERY"),
            "MAX_SEQ_GAP_MS": os.environ.get("MAX_SEQ_GAP_MS"),
            "DISCOUNT_FACTOR": os.environ.get("DISCOUNT_FACTOR"),
            "AGG_MODE": os.environ.get("AGG_MODE"),
            "DP_MODE": os.environ.get("DP_MODE"),
            "POSITION_MODE": os.environ.get("POSITION_MODE"),
            "MAX_PREDS_PER_VIDEO": os.environ.get("MAX_PREDS_PER_VIDEO"),
            "OVERLAP_THRESHOLD": os.environ.get("OVERLAP_THRESHOLD"),
            "USE_OD_RERANKING": os.environ.get("USE_OD_RERANKING"),
            "OD_RERANKING_WEIGHT": os.environ.get("OD_RERANKING_WEIGHT")
        }
    }
    
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
            
    if out_path != args.out:
        shutil.copy(out_path, args.out)
        
    import zipfile
    zip_path = os.path.join(out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        zf.write(out_path, arcname="submission.json")
    print(f"[done] wrote {out_path}, {config_path}, {zip_path} ({len(preds)} tasks)", flush=True)
    
    return out_dir

def generate_figures(out_dir, preds, tasks):
    import os
    import numpy as np
    import glob
    from PIL import Image, ImageDraw, ImageFont
    try:
        from pipeline.utils.visualize import wrap_text, get_text_size
    except ImportError:
        def get_text_size(text, font, draw):
            if hasattr(draw, "textbbox"):
                bbox = draw.textbbox((0, 0), text, font=font)
                return bbox[2] - bbox[0], bbox[3] - bbox[1]
            elif hasattr(font, "getsize"):
                return font.getsize(text)
            else:
                return len(text) * 6, 12

        def wrap_text(text, font, max_width, draw):
            words = text.split()
            lines = []
            curr_line = []
            for word in words:
                curr_line.append(word)
                line_str = " ".join(curr_line)
                w, _ = get_text_size(line_str, font, draw)
                if w > max_width and len(curr_line) > 1:
                    curr_line.pop()
                    lines.append(" ".join(curr_line))
                    curr_line = [word]
            if curr_line:
                lines.append(" ".join(curr_line))
            return lines

    
    kf_dir = os.environ.get("KF_DIR", "keyframes/fps/1")
    fig_dir = os.path.join(out_dir, "figures", "task")
    os.makedirs(fig_dir, exist_ok=True)
    
    def get_keyframe_path(video_id, frame_ms):
        vdir = os.path.join(kf_dir, str(video_id))
        ts_file = os.path.join(vdir, "ts_ms.npy")
        if not os.path.exists(ts_file):
            return None
        try:
            ts = np.load(ts_file)
            idx = np.argmin(np.abs(ts - frame_ms))
            files = sorted(glob.glob(os.path.join(vdir, "k_*.jpg")))
            if idx < len(files):
                return files[idx]
        except Exception:
            pass
        return None

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
            
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    _, line_height = get_text_size("A", font, temp_draw)
    line_height = max(line_height, 20)
    
    task_map = {str(t.get("task_id")): t.get("description", "") for t in tasks}

    def overlay_text(img, full_text):
        padding = 20
        max_text_width = img.width - 2 * padding
        lines = []
        for text_line in full_text.split('\n'):
            if text_line.strip() == "":
                lines.append("")
            else:
                lines.extend(wrap_text(text_line, font, max_text_width, temp_draw))
        text_height = len(lines) * int(line_height * 1.5) + 2 * padding
        new_img = Image.new("RGB", (img.width, img.height + text_height), "white")
        draw = ImageDraw.Draw(new_img)
        y = padding
        for line in lines:
            draw.text((padding, y), line, font=font, fill="black")
            y += int(line_height * 1.5)
        new_img.paste(img, (0, text_height))
        return new_img

    for pred in preds:
        task_id = pred.get("task_id")
        results = pred.get("results", [])
        if not results:
            continue
            
        for rank_idx, res in enumerate(results[:10]):
            rank = rank_idx + 1
            vid = res.get("video_id")
            first_ms = res.get("start_ms", res.get("frame_ms"))
            chosen_ms = res.get("frame_ms")
            
            task_dir = os.path.join(fig_dir, str(task_id))
            os.makedirs(task_dir, exist_ok=True)
            
            desc = task_map.get(str(task_id), "")
            
            p1 = get_keyframe_path(vid, first_ms)
            if p1:
                try:
                    img1 = Image.open(p1).convert("RGB")
                    text1 = f"Rank: {rank} | Type: FIRST FRAME\nVideo: {vid} | Time: {first_ms}ms\n\nInstruction: {desc}"
                    img1 = overlay_text(img1, text1)
                    img1.save(os.path.join(task_dir, f"rank{rank}_first_frame.jpg"))
                except Exception:
                    pass
                    
            p2 = get_keyframe_path(vid, chosen_ms)
            if p2:
                try:
                    img2 = Image.open(p2).convert("RGB")
                    text2 = f"Rank: {rank} | Type: CHOSEN FRAME\nVideo: {vid} | Time: {chosen_ms}ms\n\nInstruction: {desc}"
                    img2 = overlay_text(img2, text2)
                    img2.save(os.path.join(task_dir, f"rank{rank}_chosen_frame.jpg"))
                except Exception:
                    pass

def save_debug_info(out_dir, tasks, task_segments, all_candidates_final):
    """Write segments.txt and routing.txt into the submission folder."""
    os.makedirs(out_dir, exist_ok=True)
    
    # --- segments.txt ---
    with open(os.path.join(out_dir, "segments.txt"), "w", encoding="utf-8") as f:
        for ti, task in enumerate(tasks):
            f.write(f"Task ID: {task.get('task_id', ti)}\n")
            f.write(f"Full Query: {task.get('description', '')}\n")
            segs = task_segments.get(ti, {})
            if segs:
                for sent_idx in sorted(segs):
                    items = segs[sent_idx]
                    # First item (seg_idx==0) is always the scene
                    scene_text = items[0]["text"] if items else ""
                    f.write(f"  Scene {sent_idx + 1}: {scene_text}\n")
                    for item in items[1:]:
                        f.write(f"    - Object: {item['text']}\n")
            else:
                f.write(f"  (No segmentation — full query used)\n")
            f.write("\n")
    
    # --- routing.txt ---
    with open(os.path.join(out_dir, "routing.txt"), "w", encoding="utf-8") as f:
        for task, _ in all_candidates_final:
            route = task.get("route", {})
            f.write(f"Task ID: {task.get('task_id', '?')}\n")
            f.write(f"Full Query: {task.get('description', '')}\n")
            if "expanded_queries" in task:
                f.write(f"Expanded Queries:\n")
                for eq in task["expanded_queries"]:
                    f.write(f"  - {eq}\n")
            if route:
                use_asr = route.get("Use_asr", False) and bool(route.get("asr_query"))
                use_ocr = route.get("Use_ocr", False) and bool(route.get("ocr_query"))
                f.write(f"  Visual: ON (always)\n")
                f.write(f"  ASR:    {'ON' if use_asr else 'OFF'}\n")
                f.write(f"  ASR query: {route.get('asr_query') or 'NULL'}\n")
                f.write(f"  OCR:    {'ON' if use_ocr else 'OFF'}\n")
                f.write(f"  OCR query: {route.get('ocr_query') or 'NULL'}\n")
            else:
                f.write("  Visual: ON (always)\n")
                f.write("  ASR:    OFF (no router - ASR not used)\n")
                f.write("  ASR query: NULL\n")
                f.write("  OCR:    OFF (no router - OCR not used)\n")
                f.write("  OCR query: NULL\n")
            f.write("\n")
    
    print(f"[debug] wrote segments.txt and routing.txt to {out_dir}", flush=True)

def run_retrieval(args):
    tasks = [json.loads(l) for l in open(args.tasks)]
    if args.n is not None:
        tasks = tasks[:args.n]
    dev = args.device
    
    print(f"[tasks] {len(tasks)}", flush=True)
    all_queries, task_mapping, task_segments = parse_queries(tasks, args.use_sequential, args.scene_segmenter, args.object_segmenter)
    print(f"[queries] extracted {len(all_queries)} object queries from {len(tasks)} tasks", flush=True)
        
    if args.num_expansions > 0:
        print("[retrieve] Generating Expanded Queries for Reranking...")
        try:
            from pipeline.retrieval.routing.qe_expander import QueryExpander
            qe = QueryExpander(engine_name="qwen", num_expansions=args.num_expansions)
            # We expand the original task description
            tasks_desc = [t["description"] for t in tasks]
            expanded_queries_list = qe.expand_batch(tasks_desc)
            qe.cleanup()
        except Exception as e:
            print(f"[retrieve] Warning: Could not run QueryExpander: {e}")
            expanded_queries_list = [[t["description"]] * args.num_expansions for t in tasks]
    else:
        expanded_queries_list = [[] for _ in tasks]
        
    for ti, task in enumerate(tasks):
        task["expanded_queries"] = expanded_queries_list[ti]
        task["overlap_threshold"] = args.overlap_threshold
    # 1. Process Visual Modality
    visual_res = process_visual_modality(args, tasks, task_mapping, all_queries, dev)
    all_candidates_visual, first_vids, first_ts, first_metadata, first_emb, T_all, N, K, expanded_Q_embs = visual_res
    
    # 2. Route ASR needs, then process Audio Modality (Optional)
    all_candidates_audio = None
    lm_router = None
    if args.use_audio:
        lm_router, routes = route_asr_queries(tasks, engine_name="qwen")
        audio_queries, audio_task_mapping = build_asr_queries(tasks, routes)
        all_candidates_audio = process_audio_modality(
            args, tasks, audio_task_mapping, audio_queries,
            first_vids, first_ts, first_metadata, dev, N, K
        )
        if lm_router is not None and hasattr(lm_router, "cleanup"):
            lm_router.cleanup()

    # 3. Postprocess Pipeline
    from pipeline.retrieval.postprocess import postprocess_pipeline
    preds, all_candidates_final = postprocess_pipeline(
        args, tasks, all_candidates_visual, all_candidates_audio,
        first_vids, first_ts, first_emb, expanded_Q_embs
    )
    
    # 4. Save results
    out_dir = save_submission(args, preds)
    if out_dir:
        save_debug_info(out_dir, tasks, task_segments, all_candidates_final)
        print("[debug] Generating task figures (first_frame, chosen_frame)...", flush=True)
        generate_figures(out_dir, preds, tasks)

    # Return data for decorators/analysis if needed
    return tasks, task_mapping, first_vids, first_ts, first_metadata, all_queries, all_candidates_final

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visual-shards", required=True)
    p.add_argument("--metadata", default=None, help="dir containing .jsonl metadata")
    p.add_argument("--tasks", required=True, help="a round's task file, e.g. public_round_tasks.jsonl")
    p.add_argument("--out", required=True, help="submission.json path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n", type=int, default=None, help="Limit number of tasks")
    p.add_argument("--vlms", nargs="+", required=True, help="List of model,pretrained pairs e.g. ViT-B-32,laion2b_s34b_b79k")
    p.add_argument("--precision", default=None)
    p.add_argument("--top-videos", type=int, default=10)
    p.add_argument("--cand-keyframes", type=int, default=2000)
    
    # Audio args
    p.add_argument("--use-audio", action="store_true", help="Enable audio retrieval and fuse with visual")
    p.add_argument("--audio-shards", default=None, help="Path to audio index shards")
    p.add_argument("--audio-model", default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace text embedding model for audio")
    
    # Toggles for metadata scoring and clustering
    p.add_argument("--use-sequential", action="store_true", help="Enable Sequential DP Matching with SaT")
    p.add_argument("--scene-segmenter", type=str, default="regex", help="Segmenter engine for scenes (regex, bert_srl)")
    p.add_argument("--object-segmenter", type=str, default="regex", help="Segmenter engine for objects (regex, spacy, sat, scenegraph, none)")
    p.add_argument("--smoothing-window", type=int, default=3, help="Window size for Gaussian temporal smoothing")
    p.add_argument("--smoothing-sigma", type=float, default=1.0, help="Sigma for Gaussian temporal smoothing")
    p.add_argument("--num-expansions", type=int, default=2, help="Number of queries to expand for reranking")
    p.add_argument("--overlap-threshold", type=int, default=5000, help="Overlap threshold in ms for postprocessing")
    
    args = p.parse_args()
    run_retrieval(args)

if __name__ == "__main__":
    main()
