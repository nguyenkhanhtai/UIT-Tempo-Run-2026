import os
import time
import json
import glob
import math
import argparse
import subprocess
import concurrent.futures
import multiprocessing as mp
from tqdm import tqdm

def extract_audio(video_path, audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    # Extract audio at 16kHz, mono, PCM format suitable for Whisper
    cmd = [
        "ffmpeg", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        audio_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except subprocess.CalledProcessError:
        return False

def _gpu_worker(audio_batch):
    """
    Worker function executed in a separate process to avoid CUDA context sharing issues.
    Initializes a WhisperModel and transcribes a batch of audio files.
    """
    import torch
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        model = WhisperModel("base", device=device, compute_type=compute_type)
        pipeline = BatchedInferencePipeline(model)
    except Exception as e:
        print(f"[Worker] Failed to load model: {e}")
        return {}
    
    results = {}
    for vid_id, audio in audio_batch:
        try:
            segments, info = pipeline.transcribe(audio, batch_size=32, beam_size=5)
            transcript = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
            results[vid_id] = transcript
        except Exception as e:
            print(f"[Worker] Transcription error for {vid_id}: {e}")
            
        if os.path.exists(audio):
            try:
                os.remove(audio)
            except:
                pass
                
    return results

def main(args):
    print(f"--- Bắt đầu trích xuất Audio (Whisper ASR) ---")
    print(f"Video Dir : {args.video_dir}")
    print(f"Output    : {args.out}")
    print(f"Workers   : {args.gpu_workers} GPU / {args.cpu_workers} CPU")
    print(f"Chunk Size: {args.chunk_size}")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    
    # Hỗ trợ cả V3C1 và V3C2
    # Cấu trúc: dataset/Video_V3C/V3C1/videos/00001/00001.mp4
    # Nếu truyền dataset/Video_V3C thì sẽ lấy tất cả
    search_path = args.video_dir
    if not search_path.endswith("*.mp4"):
        if "videos" not in search_path:
            search_path = os.path.join(search_path, "*", "videos", "*", "*.mp4")
        else:
            search_path = os.path.join(search_path, "*", "*.mp4")
            
    all_videos = sorted(glob.glob(search_path))
    if not all_videos:
        print(f"Không tìm thấy video nào trong {args.video_dir}")
        return
        
    mine = [v for i, v in enumerate(all_videos) if i % args.shard_count == args.shard_index]
    print(f"[Shard {args.shard_index}/{args.shard_count}] Tìm thấy tổng cộng {len(mine)}/{len(all_videos)} videos.")
    
    # Tạo tên file output riêng cho từng shard để tránh xung đột
    out_file = args.out
    if args.shard_count > 1:
        base, ext = os.path.splitext(out_file)
        out_file = f"{base}_shard{args.shard_index}{ext}"
    
    results_dict = {}
    if os.path.exists(out_file):
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                results_dict = json.load(f)
            print(f"Đã load {len(results_dict)} transcripts (Checkpoint).")
        except Exception as e:
            print(f"Không thể đọc file checkpoint {out_file}: {e}")
            
    # Lọc bỏ những video đã xử lý
    pending_videos = []
    for v in mine:
        vid_id = os.path.basename(v).split('.')[0]
        if vid_id not in results_dict:
            pending_videos.append(v)
            
    print(f"Còn lại {len(pending_videos)} videos cần xử lý.")
    if not pending_videos:
        print("Đã hoàn thành toàn bộ dataset!")
        return

    def extract_job(path):
        vid_id = os.path.basename(path).split('.')[0]
        tmp_audio = os.path.join(args.tmp_dir, f"audio_{vid_id}.wav")
        if extract_audio(path, tmp_audio):
            return (vid_id, tmp_audio)
        return None
        
    for chunk_start in range(0, len(pending_videos), args.chunk_size):
        chunk_paths = pending_videos[chunk_start:chunk_start+args.chunk_size]
        chunk_idx = chunk_start // args.chunk_size + 1
        total_chunks = math.ceil(len(pending_videos) / args.chunk_size)
        
        print(f"\n[Chunk {chunk_idx}/{total_chunks}] Trích xuất audio {len(chunk_paths)} videos (CPU)...")
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.cpu_workers) as executor:
            import sys
            _stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            results = list(executor.map(extract_job, chunk_paths))
            sys.stdout.close()
            sys.stdout = _stdout
            
        audio_jobs = [r for r in results if r is not None]
        print(f" -> Hoàn tất trong {time.time()-start_time:.2f}s")
        
        if not audio_jobs:
            continue
            
        print(f"Bắt đầu dịch {len(audio_jobs)} file (GPU)...")
        gpu_start = time.time()
        
        batch_size = math.ceil(len(audio_jobs) / args.gpu_workers)
        gpu_batches = [audio_jobs[i:i + batch_size] for i in range(0, len(audio_jobs), batch_size)]
        
        # Dùng ProcessPoolExecutor với mode 'spawn'
        mp_context = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.gpu_workers, mp_context=mp_context) as executor:
            gpu_results = list(tqdm(executor.map(_gpu_worker, gpu_batches), total=len(gpu_batches), desc="GPU Workers"))
            
        for res in gpu_results:
            results_dict.update(res)
                
        # Ghi checkpoint
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
        print(f" -> Dịch hoàn tất trong {time.time()-gpu_start:.2f}s.")
            
    print(f"\n✅ [Shard {args.shard_index}] HOÀN THÀNH TOÀN BỘ PIPELINE ASR! Kết quả lưu tại: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Whisper ASR transcripts from videos")
    parser.add_argument("--video-dir", type=str, default="dataset/Video_V3C", help="Thư mục gốc chứa các dataset (V3C1, V3C2)")
    parser.add_argument("--out", type=str, default="artifacts/metadata/whisper_transcripts.json", help="Đường dẫn file kết quả")
    parser.add_argument("--tmp-dir", type=str, default="testing/tmp_audios", help="Thư mục tạm chứa file audio")
    parser.add_argument("--chunk-size", type=int, default=200, help="Số lượng video mỗi chunk")
    parser.add_argument("--gpu-workers", type=int, default=4, help="Số GPU process song song")
    parser.add_argument("--cpu-workers", type=int, default=16, help="Số CPU thread cắt audio")
    parser.add_argument("--shard-index", type=int, default=0, help="Chỉ số shard hiện tại (0-based)")
    parser.add_argument("--shard-count", type=int, default=1, help="Tổng số shard cần chia")
    
    args = parser.parse_args()
    main(args)
