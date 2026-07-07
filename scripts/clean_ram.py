import gc
import torch

def clean_memory():
    """
    Hàm tiện ích giúp dọn dẹp bộ nhớ RAM của Python và VRAM của GPU (PyTorch).
    Rất hữu ích khi cần giải phóng bộ nhớ tồn đọng trước khi tải một model mới
    vào bộ nhớ hoặc sau khi xử lý xong một khối lượng dữ liệu lớn.
    """
    # 1. Chạy Garbage Collector của Python để xóa các biến không còn sử dụng trên RAM
    gc.collect()
    
    # 2. Xóa bộ nhớ cache trên GPU (nếu dùng PyTorch và CUDA)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[Memory] Đã dọn dẹp RAM và GPU VRAM thành công.")
    else:
        print("[Memory] Đã dọn dẹp RAM thành công (Không tìm thấy GPU CUDA).")

if __name__ == "__main__":
    clean_memory()
