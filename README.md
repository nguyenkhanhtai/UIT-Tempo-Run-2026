# Video Retrieval Pipeline

## 1. Giới thiệu ngắn gọn về phương pháp
Repository này chứa toàn bộ pipeline từ đầu đến cuối cho hệ thống Video Retrieval. Phương pháp của chúng tôi kết hợp trích xuất đặc trưng hình ảnh (Visual Embeddings) bằng các mô hình VLM tiên tiến (như OpenCLIP, SigLIP, EVA-CLIP, X-CLIP, PE-Core) và sử dụng Mô hình Ngôn ngữ Lớn (LLM - Qwen 2.5) để phân tích và chia nhỏ các truy vấn (query segmentation). Quá trình truy xuất được thực hiện theo từng chunk để đảm bảo tính chịu lỗi (fault-tolerant) cao, giúp phục hồi dễ dàng khi có sự cố.

## 2. Mô tả cấu trúc repository
Cấu trúc tổng quan của dự án được tổ chức như sau:
```text
UIT-Tempo-Run-2026/
├── dataset/                     # Chứa dữ liệu đầu vào của hệ thống
│   ├── V3C1/                    # Dữ liệu video V3C1
│   │   └── videos/              # Chứa các file mp4/webm...
│   ├── V3C2/                    # Dữ liệu video V3C2
│   │   └── videos/              # Chứa các file mp4/webm...
│   ├── public_round_tasks.jsonl # File truy vấn vòng public
│   └── private_round_tasks.jsonl# File truy vấn vòng private
├── scripts/                     # Các shell script thực thi tự động pipeline
│   ├── run_pipeline.sh          # Master script chạy tự động toàn bộ quy trình từ A-Z
│   ├── extract_keyframes.sh     # Bước 1: Trích xuất frame (hình ảnh) từ video gốc
│   ├── extract_embeddings.sh    # Bước 2: Dùng CLIP chuyển đổi hình ảnh thành vector đặc trưng
│   ├── extract_metadata.sh      # Bước 3: Trích xuất thêm metadata phụ trợ (như OCR, Object)
│   └── retrieval.sh             # Bước 4: So khớp truy vấn văn bản, tìm video và tạo output
├── pipeline/                    # Mã nguồn chính chứa logic xử lý thuật toán (Python)
├── models/                      # Mã nguồn khởi tạo và định nghĩa các mô hình VLM, LLM
├── visualizer/                  # Công cụ Web UI dùng để trực quan hoá kết quả truy xuất
│   └── app.py                   # Script khởi chạy web app
├── submission/                  # Thư mục chứa kết quả nộp bài cuối cùng (submission file)
├── keyframes/                   # Dữ liệu trung gian lưu trữ các ảnh được trích xuất (Stage 1)
├── artifacts/                   # Dữ liệu vector siêu dữ liệu nén (.npz) và file index (Stage 2)
├── pyproject.toml               # File quản lý phiên bản thư viện Python (dùng uv)
└── README.md                    # Tài liệu hướng dẫn sử dụng (File này)
```

## 3. Yêu cầu phần cứng và phần mềm
**Phần cứng:**
- GPU có dung lượng VRAM lớn (yêu cầu/khuyến nghị >= 24GB) để có thể chạy mô hình VLM và LLM (Qwen 7B) mượt mà mà không bị lỗi Out of Memory.
- Ổ cứng có dung lượng lớn để lưu trữ dataset video gốc, keyframes và các feature vector.

**Phần mềm:**
- Môi trường chạy: Ubuntu (khuyến nghị) hoặc các bản phân phối Linux khác (hỗ trợ bash script).
- Python `>= 3.11`
- CUDA 12.4 (được cấu hình trong `pyproject.toml` cho PyTorch)
- `ffmpeg` (yêu cầu cài đặt sẵn để trích xuất keyframe từ video gốc)

## 4. Hướng dẫn cài đặt môi trường
Dự án sử dụng `uv` để quản lý môi trường và dependency một cách siêu tốc và nhất quán.

1. **Cài đặt `uv`** (nếu chưa có):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Cài đặt dependencies**:
   Đầu tiên, hãy di chuyển vào thư mục dự án. Sau đó chạy lệnh để tạo virtual environment và cài đặt tất cả các gói từ `uv.lock`:
   ```bash
   cd UIT-Tempo-Run-2026
   uv sync
   ```
3. **Kích hoạt môi trường ảo** (quan trọng trước khi chạy các lệnh/script tiếp theo, **luôn đảm bảo bạn đang ở trong thư mục `UIT-Tempo-Run-2026`**):
   ```bash
   source .venv/bin/activate
   ```

## 5. Hướng dẫn tải checkpoint hoặc tài nguyên bổ sung
- Các mô hình từ HuggingFace (như Qwen 2.5 7B, OpenCLIP, v.v.) sẽ tự động được tải xuống và cache lại trong lần chạy đầu tiên.
- Các tài nguyên khác (nếu có yêu cầu từ script) có thể được tải thông qua thư viện `gdown` đã được cài đặt sẵn trong dự án.

## 6. Mô tả dữ liệu đầu vào

> [!IMPORTANT]
> **Vị trí đặt Dataset:** Bạn có thể đặt thư mục chứa dataset ở bất kỳ vị trí nào trên máy tính. Chỉ cần truyền đúng đường dẫn thư mục đó vào biến `--video-root` bên trong file `scripts/run_pipeline.sh` để cấu hình.

Chương trình cần nhận đường dẫn dữ liệu thông qua tham số dòng lệnh hoặc tệp cấu hình.
Cấu trúc dữ liệu video dự kiến:
```text
dataset/
├── V3C1/
│   └── videos/
│       └── ...
└── V3C2/
    └── videos/
        └── ...
```

File task của từng vòng:
- `public_round_tasks.jsonl`
- `private_round_tasks.jsonl`

Mỗi dòng trong file task là một JSON object:

```json
{
  "task_id": "T0001",
  "description": "a man in a red jacket walks across a snowy street at night",
  "submission_type": "temporal_video_retrieval",
  "max_predictions": 10
}
```

## 7. Mô tả kết quả đầu ra
Sau khi pipeline chạy kết thúc, quá trình tự động dọn dẹp các file tạm sẽ diễn ra và kết quả được tạo trong thư mục đánh số tự động (ví dụ: `submission/001/`).
Các file được tạo bao gồm:
- `submission.json`: File kết quả theo đúng định dạng yêu cầu của ban tổ chức.
- `submission.zip`: File nén của bản nộp (submission).
- `detailed_submission.json`: Chứa các metadata siêu dữ liệu chi tiết và thông tin debug cho từng query.

## 8. Chi tiết chức năng của từng file Script
Trong dự án có rất nhiều file script nhằm phục vụ các mục đích khác nhau. Dưới đây là giải thích chi tiết cho từng file:

### 8.1 Các Script Thu thập và Chuẩn bị Dữ liệu
- **`download_dataset.py`**: Hỗ trợ tải dataset tự động. **Lưu ý:** Nếu bạn đã có sẵn dataset ở trong máy theo đúng cấu trúc yêu cầu thì không cần chạy script này để tải lại nữa.
- **`extract_audio_features.sh`**: Dùng để trích xuất đặc trưng âm thanh. **(0 use - Hiện tại không sử dụng trong pipeline)**
- **`extract_transcript.sh`**: Dùng để trích xuất văn bản từ video (speech-to-text). **(0 use - Hiện tại không sử dụng trong pipeline)**
- **`extract_metadata.sh`**: Dùng để trích xuất metadata phụ trợ như OCR/Object Detection. **(0 use - Hiện tại không sử dụng trong pipeline)**

### 8.2 Các Script Xử lý Cốt lõi (Core Pipeline)
1. **`extract_keyframes.sh`**: 
   Dùng FFmpeg để cắt video thành các frame hình ảnh rời rạc (keyframes) dựa trên chỉ số FPS được cấu hình.
2. **`extract_embeddings.sh`**: 
   Chạy các mô hình VLM (như CLIP) để chuyển đổi các frame hình ảnh đã cắt ở bước trên thành các vector đặc trưng (embeddings) số thực.
3. **`retrieval.sh`**: 
   Script chính để chạy thuật toán truy xuất. Nó sẽ nạp các câu query, phân tách ngữ nghĩa, tính toán độ tương đồng với video và cho ra thứ hạng kết quả.
4. **`run_pipeline.sh`**: 
   Master script tự động gọi lần lượt các script trên theo đúng thứ tự. Nơi chứa mọi tham số cấu hình chung.
5. **`run_chunked.py`**:
   Script Python lõi được `retrieval.sh` gọi ở bên dưới. **Cơ chế hoạt động:** Thay vì nạp toàn bộ danh sách truy vấn và video vào RAM (rất dễ gây quá tải - OOM), script này sẽ "cắt nhỏ" danh sách truy vấn thành nhiều đoạn (chunks). Việc xử lý và lưu file theo từng chunk sẽ giúp hệ thống chịu lỗi cực kỳ tốt. Nếu tiến trình bị crash giữa chừng, lần chạy sau nó sẽ tự động bỏ qua các chunk đã xử lý xong và chạy tiếp từ đoạn bị đứt (resumable).
6. **`clean_ram.py`**:
   Tiện ích nhỏ được gọi sau mỗi lần xử lý xong một khối lượng lớn dữ liệu để giải phóng RAM/VRAM rác cho hệ thống.

### 8.3 Các Script Phụ trợ khác
- **`run_analysis.sh`**: Dùng để chạy đánh giá và phân tích kết quả sau khi tạo submission.
- **`visualize.sh`** & **`viewer.py`**: Khởi chạy ứng dụng Web UI để người dùng có thể xem kết quả trực quan trên trình duyệt.
- **`upload_to_drive.py`**: Script hỗ trợ upload file submission lên Google Drive sau khi chạy xong.

## 9. Lệnh chạy toàn bộ pipeline
Để chạy toàn bộ quy trình từ đầu đến cuối một cách tự động, bạn chỉ cần gọi master script. Các tham số chạy cho toàn pipeline đã được thiết lập sẵn trong file này thành từng dòng rất rõ ràng và dễ tùy biến.

> [!WARNING]
> **Lưu ý cực kỳ quan trọng:** Bạn phải gọi lệnh thực thi pipeline khi **đang ở bên trong thư mục `UIT-Tempo-Run-2026`** (tức là đã chạy `cd UIT-Tempo-Run-2026`) để tránh lỗi đường dẫn script.

Lệnh để chạy toàn pipeline:
```bash
./scripts/run_pipeline.sh \
    --video-root "đường_dẫn_chứa_video" \          # đường dẫn tới thư mục <dataset> ở mục 6
    --task-file "đường_dẫn_tới_file_task.jsonl" \  # đường dẫn tới file task jsonl
    --keyframe-shards 6 \                          # số lượng tiến trình trích xuất hình ảnh chạy song song
    --embedding-shards 1 \                         # số lượng tiến trình VLM chạy song song (Lưu ý: tăng để chạy nhanh hơn, nếu bị lỗi OOM - Out of Memory, hãy giảm số này xuống)
    --batch-size 128                               # số lượng ảnh xử lý cùng lúc (tăng để chạy nhanh hơn, giảm nếu tràn VRAM)
```

## 10. Các tham số mặc định
Các tham số cấu hình chính được định nghĩa trong `scripts/retrieval.sh`:
- `FPS=1`: Trích xuất 1 frame mỗi giây.
- `VLMS=("PE-Core-bigG-14-448,meta")`: Mô hình Visual Language Model mặc định.
- `SCENE_SEGMENTER="qwen7b"`: Mô hình LLM dùng để phân tách ngữ nghĩa truy vấn.
- `NUM_TASKS=700`: Số lượng task xử lý tối đa mặc định.
- `USE_SEQUENTIAL="true"`
- `POSITION_MODE="second"`

## 11. Các lỗi hoặc giới hạn đã biết
- **Lỗi FFmpeg không tìm thấy**: Pipeline ở bước trích xuất keyframe sẽ báo lỗi nếu `ffmpeg` chưa được cài đặt và chưa có trong biến môi trường hệ thống (PATH).
- **Out of Memory (OOM) khi dùng PE-Core**: Việc chạy đồng thời LLM nặng (`qwen7b`) và các VLM kích thước lớn có thể gây lỗi tràn VRAM. Đặc biệt ở bước trích xuất embedding, mô hình `PE-Core` rất nặng. **Hướng giải quyết:** Khuyến nghị chạy hệ thống trên nhiều GPU (multi-GPU) hoặc dùng GPU có nhiều VRAM (>= 24GB). Nếu vẫn gặp lỗi, bạn nên giảm `batch-size` và `embedding-shards` xuống.
- **Tiến trình gián đoạn**: Quá trình Retrieval lưu tạm (checkpoint) theo từng chunk. Nếu bị gián đoạn, bạn có thể chạy lại `retrieval.sh`, hệ thống sẽ tự động resume (tiếp tục) từ các chunk đã chạy xong mà không cần tính toán lại từ đầu.

---
**Công cụ Trực quan hóa kết quả (Visualizer)**
Sau khi chạy xong, bạn có thể xem trực quan kết quả truy xuất bằng web app tích hợp sẵn:
```bash
uv run python visualizer/app.py
```
Mở trình duyệt và truy cập vào `http://localhost:5050` để sử dụng UI.
