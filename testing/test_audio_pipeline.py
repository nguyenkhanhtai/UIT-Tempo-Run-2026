import os
import time
import subprocess
from faster_whisper import WhisperModel

def extract_audio(video_path, audio_path):
    print(f"Trích xuất audio từ {video_path}...")
    start = time.time()
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
        
    # Chạy ffmpeg để tách audio chuẩn (16kHz, mono, PCM) cho Whisper
    cmd = [
        "ffmpeg", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        audio_path
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"Đã trích xuất xong audio ra {audio_path} trong {time.time()-start:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy ffmpeg: {e}")
        return False

def test_whisper(audio_path, batched=False):
    print("\nKhởi tạo model Faster-Whisper (base model)...")
    # Có thể thử các model: tiny, base, small, medium, large-v3
    model_size = "base"
    
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print(f"Bắt đầu trích xuất transcript (Device: {device}, Compute: {compute_type})...")
    start = time.time()
    
    if batched:
        from faster_whisper import BatchedInferencePipeline
        pipeline = BatchedInferencePipeline(model)
        segments, info = pipeline.transcribe(audio_path, batch_size=8, beam_size=5)
    else:
        # Chạy inference
        segments, info = model.transcribe(audio_path, beam_size=5)
    
    # Ép Generator thực thi để đo thời gian chính xác
    segments = list(segments)
    
    print(f"\nPhát hiện ngôn ngữ: {info.language} với độ tự tin {info.language_probability:.2f}")
    print("-" * 50)
    print("CÁC ĐOẠN HỘI THOẠI (TRANSCRIPT):")
    
    full_text = ""
    for segment in segments:
        text = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}"
        print(text)
        full_text += segment.text + " "
        
    print("-" * 50)
    print(f"Tổng thời gian Whisper xử lý: {time.time()-start:.2f}s")
    return full_text

def process_single_video(video_path, audio_tmp):
    if extract_audio(video_path, audio_tmp):
        test_whisper(audio_tmp, batched=True)
        if os.path.exists(audio_tmp):
            os.remove(audio_tmp)

    def _gpu_worker(audio_batch):
        # Nạp model riêng biệt trên mỗi tiến trình GPU (chiếm khoảng ~1GB VRAM mỗi tiến trình)
        import torch
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        import os
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = WhisperModel("base", device=device, compute_type=compute_type)
        pipeline = BatchedInferencePipeline(model)
        
        results = {}
        for vid_id, audio in audio_batch:
            try:
                segments, info = pipeline.transcribe(audio, batch_size=32, beam_size=5)
                transcript = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
                results[vid_id] = transcript
            except Exception as e:
                pass
                
            if os.path.exists(audio):
                os.remove(audio)
                
        return results

def test_multiple_videos(video_paths):
    print(f"\n--- Bắt đầu Xử lý {len(video_paths)} video ---")
    import concurrent.futures
    import multiprocessing as mp
    import json
    from tqdm import tqdm
    
    output_file = "testing/all_transcripts.json"
    results_dict = {}
    
    # Nếu file đã tồn tại thì load lên để resume
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            results_dict = json.load(f)
            print(f"Đã load {len(results_dict)} transcripts từ trước.")
    
    CHUNK_SIZE = 200
    NUM_GPU_WORKERS = 4 # Tận dụng 16GB VRAM bằng cách khởi tạo 4 mô hình song song
    
    def extract_job(path):
        vid_id = os.path.basename(path).split('.')[0]
        if vid_id in results_dict:
            return None # Bỏ qua nếu đã dịch
        tmp_audio = f"testing/tmp_audio_{vid_id}.wav"
        if extract_audio(path, tmp_audio):
            return (vid_id, tmp_audio)
        return None
        
    for chunk_start in range(0, len(video_paths), CHUNK_SIZE):
        chunk_paths = video_paths[chunk_start:chunk_start+CHUNK_SIZE]
        print(f"\n[Chunk {chunk_start//CHUNK_SIZE + 1}] Trích xuất audio {len(chunk_paths)} videos...")
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            import sys
            _stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            results = list(executor.map(extract_job, chunk_paths))
            sys.stdout.close()
            sys.stdout = _stdout
            
        audio_jobs = [r for r in results if r is not None]
        print(f"Trích xuất audio hoàn tất trong {time.time()-start_time:.2f}s")
        
        if not audio_jobs:
            continue
            
        gpu_start = time.time()
        print(f"Bắt đầu dịch {len(audio_jobs)} file bằng {NUM_GPU_WORKERS} GPU Workers song song...")
        
        # Chia batch data cho 4 tiến trình
        import math
        batch_size = math.ceil(len(audio_jobs) / NUM_GPU_WORKERS)
        gpu_batches = [audio_jobs[i:i + batch_size] for i in range(0, len(audio_jobs), batch_size)]
        
        # Dùng ProcessPoolExecutor với mode 'spawn' để tránh lỗi CUDA in Multiprocessing
        mp_context = mp.get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_GPU_WORKERS, mp_context=mp_context) as executor:
            gpu_results = list(tqdm(executor.map(_gpu_worker, gpu_batches), total=len(gpu_batches)))
            
        for res in gpu_results:
            results_dict.update(res)
                
        # Lưu kết quả sau mỗi chunk
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)
            
        print(f"Chunk hoàn tất trong {time.time()-gpu_start:.2f}s. Đã lưu tiến độ.")
            
    print(f"\nĐã hoàn thành toàn bộ dataset! Kết quả lưu tại {output_file}")

if __name__ == "__main__":
    import glob
    all_videos = glob.glob("dataset/Video_V3C/V3C1/videos/*/*.mp4")
    
    if not all_videos:
        print("Không tìm thấy video test!")
        exit()
        
    test_multiple_videos(all_videos)
