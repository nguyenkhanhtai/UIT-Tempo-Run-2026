import os
import cv2
import time
import numpy as np

def test_easyocr(image_path):
    print(f"\n--- Testing EasyOCR (CNN-RNN based) on {image_path} ---")
    try:
        import easyocr
        # Khởi tạo mô hình: 'en' cho tiếng Anh. Có thể thêm 'vi' nếu cần tiếng Việt.
        # gpu=True để sử dụng GPU nếu có.
        reader = easyocr.Reader(['en'], gpu=True)
        
        start_time = time.time()
        # Chạy pipeline: Text Detection (CRAFT) -> Text Recognition (CRNN)
        results = reader.readtext(image_path)
        end_time = time.time()
        
        print(f"[EasyOCR] Tổng thời gian Detection + Recognition: {end_time - start_time:.4f}s")
        print(f"[EasyOCR] Tìm thấy {len(results)} vùng chứa chữ.")
        
        img = cv2.imread(image_path)
        if img is not None:
            for (bbox, text, prob) in results:
                print(f"  - '{text}' (Độ tự tin: {prob:.4f})")
                
                # Vẽ khung chữ nhật (Bounding Box)
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                br = (int(br[0]), int(br[1]))
                cv2.rectangle(img, tl, br, (0, 255, 0), 2)
                cv2.putText(img, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Lưu ảnh visualize
            out_dir = "test_images/output"
            os.makedirs(out_dir, exist_ok=True)
            basename = os.path.basename(image_path).replace(".png", "_easyocr.png")
            out_path = os.path.join(out_dir, basename)
            cv2.imwrite(out_path, img)
            print(f"[EasyOCR] Đã lưu ảnh kết quả tại: {out_path}")
            
    except ImportError:
        print("Cảnh báo: easyocr chưa được cài đặt. Thử: pip install easyocr")


def test_rapidocr(image_path):
    print(f"\n--- Testing RapidOCR (Dựa trên PaddleOCR, siêu nhẹ) on {image_path} ---")
    try:
        from rapidocr_onnxruntime import RapidOCR
        
        # Khởi tạo Pipeline (DBNet cho Detection, CRNN/SVTR cho Recognition)
        engine = RapidOCR()
        
        start_time = time.time()
        result, elapse = engine(image_path)
        end_time = time.time()
        
        print(f"[RapidOCR] Tổng thời gian Detection + Recognition: {end_time - start_time:.4f}s")
        if result is None:
            print("[RapidOCR] Không tìm thấy chữ nào.")
            return
            
        print(f"[RapidOCR] Tìm thấy {len(result)} vùng chứa chữ.")
        
        img = cv2.imread(image_path)
        if img is not None:
            for res in result:
                # Mỗi result trả về: [tọa_độ_box, text, độ_tự_tin]
                box, text, score = res
                print(f"  - '{text}' (Độ tự tin: {score:.4f})")
                
                box = np.array(box).astype(np.int32)
                # Vẽ box đa giác (cho các chữ in nghiêng/chéo)
                cv2.polylines(img, [box], isClosed=True, color=(255, 0, 0), thickness=2)
                cv2.putText(img, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
            out_dir = "test_images/output"
            os.makedirs(out_dir, exist_ok=True)
            basename = os.path.basename(image_path).replace(".png", "_rapidocr.png")
            out_path = os.path.join(out_dir, basename)
            cv2.imwrite(out_path, img)
            print(f"[RapidOCR] Đã lưu ảnh kết quả tại: {out_path}")

    except ImportError:
        print("Cảnh báo: rapidocr-onnxruntime chưa được cài đặt. Thử: pip install rapidocr-onnxruntime")

import concurrent.futures
import multiprocessing as mp

_rapid_engine = None
def init_worker_rapid():
    global _rapid_engine
    from rapidocr_onnxruntime import RapidOCR
    _rapid_engine = RapidOCR()

def process_rapid(image_path):
    global _rapid_engine
    if _rapid_engine is None:
        init_worker_rapid()
    _rapid_engine(image_path)
    return image_path

_easy_engine = None
def init_worker_easy():
    """Khởi tạo EasyOCR worker"""
    global _easy_engine
    import easyocr
    import torch
    import warnings
    # Suppress UserWarning about RNN module weights
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.modules.rnn')
    _easy_engine = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    # Flatten parameters to fix RNN memory contiguous warning
    for module in _easy_engine.recognizer.modules():
        if isinstance(module, torch.nn.RNNBase):
            module.flatten_parameters()

def process_easy(image_path):
    global _easy_engine
    if _easy_engine is None:
        init_worker_easy()
    _easy_engine.readtext(image_path)
    return image_path

def extract_frames(video_path, out_dir="test_images/frames_tmp", max_frames=30):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25
    
    frame_count = 0
    saved_frames = []
    
    # Lấy 1 frame mỗi 1 giây (để test)
    while cap.isOpened() and len(saved_frames) < max_frames:
        frame_id = int(round(cap.get(1)))
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_id % int(fps) == 0:
            out_path = os.path.join(out_dir, f"frame_{len(saved_frames):04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_frames.append(out_path)
            
        frame_count += 1
        
    cap.release()
    print(f"Đã trích xuất {len(saved_frames)} frames từ video.")
    return saved_frames

def test_parallel(image_paths):
    """Test tốc độ chạy song song và batching"""
    import multiprocessing as mp
    import concurrent.futures
    import time
    
    # Validate
    valid_images = [p for p in image_paths if os.path.exists(p)]
    if not valid_images:
        print("Không có ảnh hợp lệ để test!")
        return
        
    print(f"\n--- Tốc độ xử lý SONG SONG trên {len(valid_images)} ảnh ---")
    
    # # 1. Test RapidOCR (CPU)
    # print("\n[RapidOCR] Đang test Tuần tự (Sequential)...")
    # start = time.time()
    # for img in valid_images:
    #     process_rapid(img)
    # seq_rapid_time = time.time() - start
    # print(f"[RapidOCR] Thời gian Tuần tự: {seq_rapid_time:.2f}s")
    
    # print("\n[RapidOCR] Đang test Song song (8 Process Workers)...")
    # # Giới hạn số luồng của OpenBLAS/OMP để tránh lỗi tràn luồng (Resource temporarily unavailable)
    # os.environ['OMP_NUM_THREADS'] = '1'
    # os.environ['OPENBLAS_NUM_THREADS'] = '1'
    # os.environ['MKL_NUM_THREADS'] = '1'
    
    # start = time.time()
    # with concurrent.futures.ProcessPoolExecutor(max_workers=8, initializer=init_worker_rapid) as executor:
    #     list(executor.map(process_rapid, valid_images))
    # par_rapid_time = time.time() - start
    # print(f"[RapidOCR] Thời gian Song song: {par_rapid_time:.2f}s")
    # print(f"[RapidOCR] Speedup: {seq_rapid_time/par_rapid_time:.2f}x nhanh hơn!")
    
    # 2. Test EasyOCR (GPU)
    # Với GPU, dùng multiprocessing cần cẩn thận vì CUDA context. Thường dùng ThreadPool sẽ an toàn hơn
    # hoặc ProcessPool với 'spawn'. Ở đây dùng ThreadPoolExecutor vì EasyOCR có nhả GIL khi chạy C++/CUDA.
    print("\n[EasyOCR] Đang test Tuần tự (Sequential)...")
    start = time.time()
    for img in valid_images:
        process_easy(img)
    seq_easy_time = time.time() - start
    print(f"[EasyOCR] Thời gian Tuần tự: {seq_easy_time:.2f}s")
    
    print(f"\n[EasyOCR] Đang test Song song ({min(16, os.cpu_count() or 4)} Thread Workers)...")
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        list(executor.map(process_easy, valid_images))
    par_easy_time = time.time() - start
    print(f"[EasyOCR] Thời gian Song song: {par_easy_time:.2f}s")
    print(f"[EasyOCR] Speedup (Parallel): {seq_easy_time/par_easy_time:.2f}x nhanh hơn!")
    
    # Xoá cache GPU trước khi test batch
    import torch
    torch.cuda.empty_cache()
    
    print("\n[EasyOCR] Đang test Batched Inference (Batch Size = 8)...")
    start = time.time()
    # Batching natively via EasyOCR
    init_worker_easy() # init engine in main thread
    # load images to memory to avoid I/O delay in testing compute time
    import cv2
    img_data = [cv2.imread(img) for img in valid_images]
    _easy_engine.readtext_batched(img_data, batch_size=1)
    batched_easy_time = time.time() - start
    print(f"[EasyOCR] Thời gian Batched: {batched_easy_time:.2f}s")
    print(f"[EasyOCR] Speedup (Batched): {seq_easy_time/batched_easy_time:.2f}x nhanh hơn!")


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    video_path = "dataset/Video_V3C/V3C1/videos/00001/00001.mp4"
    if not os.path.exists(video_path):
        print(f"Không tìm thấy video: {video_path}")
    else:
        print(f"\nBắt đầu trích xuất frames từ video: {video_path}...")
        test_images = extract_frames(video_path, max_frames=180)
        
        print("\nBắt đầu test xử lý song song trên video frames...")
        test_parallel(test_images)
