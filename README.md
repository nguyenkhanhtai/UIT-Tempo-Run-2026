# Video Retrieval Pipeline - SymbCoT

## 1. Giới thiệu ngắn gọn về phương pháp
Repository này chứa toàn bộ pipeline từ đầu đến cuối cho hệ thống Video Retrieval. Phương pháp của chúng tôi kết hợp trích xuất đặc trưng hình ảnh (Visual Embeddings) bằng các mô hình VLM tiên tiến (như OpenCLIP, SigLIP, EVA-CLIP, X-CLIP, PE-Core) và sử dụng Mô hình Ngôn ngữ Lớn (LLM - Qwen 2.5) để phân tích và chia nhỏ các truy vấn (query segmentation). Quá trình truy xuất được thực hiện theo từng chunk để đảm bảo tính chịu lỗi (fault-tolerant) cao, giúp phục hồi dễ dàng khi có sự cố.

## 2. Mô tả cấu trúc repository
- `dataset/`: Thư mục chứa dữ liệu video gốc (VD: V3C) và file truy vấn (tasks).
- `scripts/`: Chứa các shell script để chạy từng bước trong pipeline (trích xuất keyframe, embedding, metadata, và truy xuất).
- `visualizer/`: Chứa mã nguồn cho ứng dụng web trực quan hóa kết quả truy xuất.
- `submission/`: Thư mục lưu kết quả cuối cùng được tạo tự động sau khi chạy pipeline.
- `keyframes/` & `artifacts/`: Nơi lưu trữ các frame được trích xuất và các vector đặc trưng (embeddings) sau khi xử lý.

## 3. Yêu cầu phần cứng và phần mềm
**Phần cứng:**
- GPU có dung lượng VRAM lớn (khuyến nghị >= 16GB) để chạy các mô hình VLM và LLM (như Qwen 7B).
- Ổ cứng có dung lượng lớn để lưu trữ dataset video gốc, keyframes và các feature vector.

**Phần mềm:**
- Hệ điều hành: Linux/Windows (hỗ trợ bash script)
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
   Chạy lệnh sau ở thư mục gốc để tạo virtual environment và cài đặt tất cả các gói từ `uv.lock`:
   ```bash
   uv sync
   ```

## 5. Hướng dẫn tải checkpoint hoặc tài nguyên bổ sung
- Các mô hình từ HuggingFace (như Qwen 2.5 7B, OpenCLIP, v.v.) sẽ tự động được tải xuống và cache lại trong lần chạy đầu tiên.
- Các tài nguyên khác (nếu có yêu cầu từ script) có thể được tải thông qua thư viện `gdown` đã được cài đặt sẵn trong dự án.

## 6. Mô tả dữ liệu đầu vào
Cần đảm bảo dữ liệu được đặt đúng định dạng và cấu trúc thư mục trước khi chạy pipeline:
1. **Video Dataset**: Đặt các thư mục chứa video gốc (ví dụ: `Video_V3C`) vào đường dẫn `dataset/Video_V3C/`.
2. **Tasks File**: Đặt file JSONL chứa các câu truy vấn/task đánh giá tại `dataset/private_round_tasks.jsonl`.

## 7. Mô tả kết quả đầu ra
Sau khi pipeline chạy kết thúc, quá trình tự động dọn dẹp các file tạm sẽ diễn ra và kết quả được tạo trong thư mục đánh số tự động (ví dụ: `submission/001/`).
Các file được tạo bao gồm:
- `submission.json`: File kết quả theo đúng định dạng yêu cầu của ban tổ chức.
- `submission.zip`: File nén của bản nộp (submission).
- `detailed_submission.json`: Chứa các metadata siêu dữ liệu chi tiết và thông tin debug cho từng query.

## 8. Hướng dẫn chạy từng script
Bạn có thể linh hoạt chạy riêng lẻ từng bước bằng các script trong thư mục `scripts/`:
1. **`./scripts/extract_keyframes.sh`**: Lấy mẫu (sample) các frame từ video gốc bằng công cụ FFmpeg.
2. **`./scripts/extract_embeddings.sh`**: Tính toán embedding (đặc trưng hình ảnh) bằng các mô hình VLM.
3. **`./scripts/extract_metadata.sh`**: Trích xuất metadata đa phương thức bổ sung (ví dụ: OCR/Object Detection) nếu được bật.
4. **`./scripts/retrieval.sh`**: Chạy logic truy xuất chính. Đọc file task, chia nhỏ truy vấn bằng LLM, tính toán độ tương đồng giữa text-video và hợp nhất các chunk kết quả.

## 9. Lệnh chạy toàn bộ pipeline
Để chạy toàn bộ quy trình từ đầu đến cuối một cách tự động, bạn chỉ cần chạy duy nhất master script sau:
```bash
./scripts/run_pipeline.sh
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
- **Out of Memory (OOM)**: Việc chạy đồng thời LLM nặng (`qwen7b`) và các VLM kích thước lớn có thể gây lỗi tràn VRAM trên các GPU phổ thông. Hướng giải quyết: cấu hình dùng VLM nhẹ hơn trong `retrieval.sh` hoặc giảm kích thước batch.
- **Tiến trình gián đoạn**: Quá trình Retrieval lưu tạm (checkpoint) theo từng chunk. Nếu bị gián đoạn, bạn có thể chạy lại `retrieval.sh`, hệ thống sẽ tự động resume (tiếp tục) từ các chunk đã chạy xong mà không cần tính toán lại từ đầu.

---
**Công cụ Trực quan hóa kết quả (Visualizer)**
Sau khi chạy xong, bạn có thể xem trực quan kết quả truy xuất bằng web app tích hợp sẵn:
```bash
uv run python visualizer/app.py
```
Mở trình duyệt và truy cập vào `http://localhost:5000` để sử dụng UI.
