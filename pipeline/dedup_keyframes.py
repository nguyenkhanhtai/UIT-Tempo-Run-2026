import os
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import shutil

def image_diff(img1_path, img2_path):
    try:
        i1 = np.array(Image.open(img1_path).convert('L').resize((32, 32)), dtype=np.float32)
        i2 = np.array(Image.open(img2_path).convert('L').resize((32, 32)), dtype=np.float32)
        return np.mean(np.abs(i1 - i2))
    except Exception as e:
        print(f"Error comparing {img1_path} and {img2_path}: {e}")
        return 999.0 # Force keep on error

def dedup_video(vdir, embed_npz_path, threshold=5.0):
    old_ts_path = os.path.join(vdir, "ts_ms.npy")
    if not os.path.exists(old_ts_path):
        return
        
    ts = np.load(old_ts_path)
    files = sorted(glob.glob(os.path.join(vdir, "k_*.jpg")))
    
    if len(ts) != len(files):
        print(f"[WARN] Length mismatch in {vdir.name}")
        return
        
    # Load embeddings
    emb = None
    if embed_npz_path and os.path.exists(embed_npz_path):
        data = np.load(embed_npz_path)
        if len(data["ts_ms"]) == len(ts):
            emb = data["emb"]
            
    if emb is None:
        print(f"[WARN] No embeddings found for {vdir.name}, skipping dedup.")
        return

    keep_indices = []
    last_kept_idx = -1
    
    for i in range(len(ts)):
        # Check if it's a NEW frame (embedding is all zeros)
        # We only consider dropping new frames to protect old metadata!
        is_new = np.all(emb[i] == 0.0)
        
        if not is_new:
            # Always keep old frames
            keep_indices.append(i)
            last_kept_idx = i
        else:
            if last_kept_idx == -1:
                # First frame, must keep
                keep_indices.append(i)
                last_kept_idx = i
            else:
                # Compare this new frame to the last kept frame
                diff = image_diff(files[i], files[last_kept_idx])
                if diff > threshold:
                    keep_indices.append(i)
                    last_kept_idx = i
                else:
                    # Drop it (delete file later)
                    pass

    dropped = len(ts) - len(keep_indices)
    if dropped == 0:
        return
        
    print(f"[{vdir.name}] Dropped {dropped} similar new frames. Kept {len(keep_indices)}.")
    
    # Process deletions and renumbering
    # First, mark files to keep
    new_files = [files[i] for i in keep_indices]
    new_ts = [ts[i] for i in keep_indices]
    new_emb = np.stack([emb[i] for i in keep_indices])
    
    # Delete dropped files
    for i in range(len(files)):
        if i not in keep_indices:
            os.remove(files[i])
            
    # Renumber the kept files to avoid gaps
    # Rename to temp first
    for i, f in enumerate(new_files):
        temp_name = os.path.join(vdir, f"k_temp_{i:05d}.jpg")
        shutil.move(f, temp_name)
        
    # Rename back to final
    for i in range(len(new_files)):
        temp_name = os.path.join(vdir, f"k_temp_{i:05d}.jpg")
        final_name = os.path.join(vdir, f"k_{i:05d}.jpg")
        shutil.move(temp_name, final_name)
        
    # Save updated arrays
    np.save(old_ts_path, np.array(new_ts, dtype=np.int32))
    np.savez(embed_npz_path, emb=new_emb.astype(np.float16), ts_ms=np.array(new_ts, dtype=np.int32))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keyframes", required=True)
    p.add_argument("--embed-dir", required=True)
    p.add_argument("--threshold", type=float, default=5.0, help="Pixel diff threshold (0-255)")
    args = p.parse_args()
    
    vdirs = sorted([Path(p).parent for p in glob.glob(os.path.join(args.keyframes, "*", "ts_ms.npy"))])
    print(f"Checking {len(vdirs)} videos for deduplication...")
    
    for vdir in vdirs:
        vid = vdir.name
        npz = os.path.join(args.embed_dir, f"{vid}.npz")
        dedup_video(vdir, npz, threshold=args.threshold)

if __name__ == "__main__":
    main()
