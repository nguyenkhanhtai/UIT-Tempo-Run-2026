"""
Purpose: Applies moving average temporal smoothing to video frame embeddings
to reduce noise and provide contextual features.
"""
import numpy as np

def smooth_features(emb, vids):
    print("[retrieve] Smoothing features temporally...", flush=True)
    emb_prev = np.roll(emb, 1, axis=0)
    emb_next = np.roll(emb, -1, axis=0)
    
    same_prev = (vids == np.roll(vids, 1))
    same_next = (vids == np.roll(vids, -1))
    
    smoothed = emb.astype(np.float32).copy()
    emb_prev = emb_prev.astype(np.float32)
    emb_next = emb_next.astype(np.float32)
    
    mask_mid = same_prev & same_next
    smoothed[mask_mid] = 0.6 * smoothed[mask_mid] + 0.2 * emb_prev[mask_mid] + 0.2 * emb_next[mask_mid]
    
    mask_left = same_next & ~same_prev
    smoothed[mask_left] = 0.8 * smoothed[mask_left] + 0.2 * emb_next[mask_left]
    
    mask_right = same_prev & ~same_next
    smoothed[mask_right] = 0.8 * smoothed[mask_right] + 0.2 * emb_prev[mask_right]
    
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    emb_smoothed = np.where(norms > 1e-9, smoothed / norms, smoothed).astype(np.float16)
    
    return emb_smoothed
