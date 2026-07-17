"""
Purpose: Applies moving average temporal smoothing to video frame embeddings
to reduce noise and provide contextual features.
"""
import numpy as np

def smooth_features(emb, vids, smoothing_window=3, sigma=1.0):
    print(f"[retrieve] Smoothing features temporally (Gaussian, window={smoothing_window}, sigma={sigma})...", flush=True)
    
    if smoothing_window < 1 or smoothing_window % 2 == 0:
        raise ValueError("smoothing_window must be a positive odd integer")
        
    radius = smoothing_window // 2
    
    # Tính trọng số Gaussian
    x = np.arange(-radius, radius + 1)
    weights = np.exp(-(x**2) / (2 * sigma**2))
    # Không cần thiết phải chia np.sum(weights) ở đây vì bên dưới sẽ normalize lại theo weight_sum
    
    smoothed = np.zeros_like(emb, dtype=np.float32)
    weight_sum = np.zeros((emb.shape[0], 1), dtype=np.float32)
    emb_f32 = emb.astype(np.float32)
    
    # Duyệt qua từng vị trí trong cửa sổ
    for i, offset in enumerate(range(-radius, radius + 1)):
        weight = weights[i]
        
        # Dịch chuyển mảng
        # Nếu offset < 0 (khung hình trước), roll với số dương để đẩy mảng xuống
        emb_rolled = np.roll(emb_f32, -offset, axis=0)
        vids_rolled = np.roll(vids, -offset)
        
        # Chỉ áp dụng nếu khung hình lân cận thuộc cùng một video
        mask = (vids == vids_rolled)
        
        # Cộng dồn giá trị và trọng số
        smoothed[mask] += emb_rolled[mask] * weight
        weight_sum[mask, 0] += weight
        
    # Chia cho tổng trọng số hợp lệ (nhằm xử lý các frame ở rìa video)
    smoothed = smoothed / np.where(weight_sum > 0, weight_sum, 1.0)
    
    # Chuẩn hóa L2 (Unit length) cho tính toán Cosine Similarity
    norms = np.linalg.norm(smoothed, axis=1, keepdims=True)
    emb_smoothed = np.where(norms > 1e-9, smoothed / norms, smoothed).astype(np.float16)
    
    return emb_smoothed

