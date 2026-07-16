import os, glob, subprocess, shutil, sys, time, argparse
from pathlib import Path
import numpy as np
import re

FFMPEG = "ffmpeg"
PTS_RE = re.compile(r"pts_time:([0-9\.]+)")

def extract_1fps(mp4: str, out_dir: str):
    """Extract 1 fps frames into out_dir and return sorted files + ts_ms array."""
    os.makedirs(out_dir, exist_ok=True)
    pat = os.path.join(out_dir, "new_%05d.jpg")
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "info", "-i", mp4, "-vf", "fps=1,showinfo", "-q:v", "3", pat]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
    pts = [float(x) for x in PTS_RE.findall(proc.stderr.decode("utf-8", "ignore"))]
    files = sorted(glob.glob(os.path.join(out_dir, "new_*.jpg")))
    
    # If showinfo failed to capture, fallback
    if not pts:
        pts = list(range(len(files)))
        
    n = min(len(files), len(pts))
    for extra in files[n:]:
        os.remove(extra)
        
    ts = [int(round(pts[i] * 1000)) for i in range(n)]
    return files[:n], ts

def fill_video(vdir, mp4_path, embed_npz_path):
    # print(f"Processing {vdir.name}...")
    # Load existing ts_ms
    old_ts_path = os.path.join(vdir, "ts_ms.npy")
    if not os.path.exists(old_ts_path):
        return
        
    old_ts = np.load(old_ts_path).tolist()
    old_files = sorted(glob.glob(os.path.join(vdir, "k_*.jpg")))
    if len(old_ts) != len(old_files):
        print(f"[WARN] Length mismatch in {vdir.name}")
        return
        
    # Load existing embeddings if any
    old_emb = None
    if embed_npz_path and os.path.exists(embed_npz_path):
        data = np.load(embed_npz_path)
        if len(data["ts_ms"]) == len(old_ts):
            old_emb = data["emb"]
            
    # Extract new frames
    tmp_dir = os.path.join(vdir, "tmp_1fps")
    new_files, new_ts = extract_1fps(mp4_path, tmp_dir)
    
    # Filter new frames (keep if gap > 500ms from ALL existing frames)
    surviving_new = []
    for f, t in zip(new_files, new_ts):
        # find closest old_ts
        min_dist = min([abs(t - ot) for ot in old_ts]) if old_ts else 999999
        if min_dist > 500:
            surviving_new.append((f, t))
        else:
            os.remove(f) # trash it
            
    if not surviving_new:
        shutil.rmtree(tmp_dir)
        print(f"  [{vdir.name}] Already dense enough, skipping.")
        return # nothing to add
        
    print(f"  [{vdir.name}] Adding {len(surviving_new)} new frames to {len(old_ts)} existing frames.")
    
    # Merge and sort
    merged = []
    for i, (f, t) in enumerate(zip(old_files, old_ts)):
        merged.append({ "old_file": f, "ts": t, "is_new": False, "old_idx": i })
        
    for f, t in surviving_new:
        merged.append({ "old_file": f, "ts": t, "is_new": True, "old_idx": -1 })
        
    merged.sort(key=lambda x: x["ts"])
    
    # Rename and rebuild
    new_ts_array = []
    new_emb_list = []
    
    dim = old_emb.shape[1] if old_emb is not None else 512
    
    # We rename to k_temp_*.jpg first to avoid collision
    for i, m in enumerate(merged):
        temp_name = os.path.join(vdir, f"k_temp_{i:05d}.jpg")
        shutil.move(m["old_file"], temp_name)
        m["temp_file"] = temp_name
        
    for i, m in enumerate(merged):
        final_name = os.path.join(vdir, f"k_{i:05d}.jpg")
        shutil.move(m["temp_file"], final_name)
        new_ts_array.append(m["ts"])
        
        if old_emb is not None:
            if m["is_new"]:
                new_emb_list.append(np.zeros(dim, dtype=np.float16))
            else:
                new_emb_list.append(old_emb[m["old_idx"]])
                
    # Save
    np.save(old_ts_path, np.array(new_ts_array, dtype=np.int32))
    
    if old_emb is not None:
        new_emb_array = np.stack(new_emb_list)
        np.savez(embed_npz_path, emb=new_emb_array.astype(np.float16), ts_ms=np.array(new_ts_array, dtype=np.int32))
        
    shutil.rmtree(tmp_dir)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", action="append", default=[])
    p.add_argument("--keyframes", required=True)
    p.add_argument("--embed-dir", default=None)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()
    
    # find all mp4s
    mp4s = {}
    for root in args.dataset_root:
        coll = Path(root).name.lower()
        for mp4 in glob.glob(os.path.join(root, "videos", "*", "*.mp4")):
            vid = f"{coll}_{Path(mp4).stem}"
            mp4s[vid] = mp4
            
    vdirs = sorted([Path(p).parent for p in glob.glob(os.path.join(args.keyframes, "*", "ts_ms.npy"))])
    mine = [d for i, d in enumerate(vdirs) if i % args.shard_count == args.shard_index]
    print(f"[shard {args.shard_index}/{args.shard_count}] Checking {len(mine)} videos using 16 threads...")
    
    import concurrent.futures
    
    def process_vdir(vdir):
        vid = vdir.name
        if vid not in mp4s:
            return
        npz = None
        if args.embed_dir:
            npz = os.path.join(args.embed_dir, f"{vid}.npz")
        try:
            fill_video(vdir, mp4s[vid], npz)
        except Exception as e:
            print(f"Error processing {vid}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_vdir, mine)

if __name__ == "__main__":
    main()
