import os
import gdown
import zipfile

# Định nghĩa đường dẫn lưu dataset
dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

# Tạo thư mục nếu chưa tồn tại
os.makedirs(dataset_dir, exist_ok=True)
print(f"Thư mục lưu trữ: {dataset_dir}")

# 1. Tải Public round tasks.jsonl
print("\n" + "="*40)
print("Đang tải file Public round tasks.jsonl...")
jsonl_url = "https://drive.google.com/file/d/1sjslvmbv_jP9O83rKqB4TzJ3YWSMPzdM/view?usp=sharing"
jsonl_output = os.path.join(dataset_dir, "Public_round_tasks.jsonl")
gdown.download(url=jsonl_url, output=jsonl_output, quiet=False)

import time

# 2. Tải Video V3C.zip
print("\n" + "="*40)
print("Đang tải file Video V3C.zip (tự động khôi phục nếu bị ngắt kết nối sau 1 tiếng)...")
zip_id = "1dX1bCthvy_9Q_qRS5Qf_17zlsaziYs61"
zip_output = os.path.join(dataset_dir, "Video_V3C.zip")

max_retries = 20
for attempt in range(1, max_retries + 1):
    try:
        # resume=True giúp tải tiếp phần còn dở, không tải lại từ đầu
        result = gdown.download(id=zip_id, output=zip_output, quiet=False, resume=True)
        if result:
            print("Đã tải thành công 100% Video V3C.zip!")
            break
    except Exception as e:
        print(f"Bị ngắt kết nối hoặc gặp lỗi ở lần thử thứ {attempt}: {e}")
        print("Đang đợi 5 giây để thử kết nối và tải tiếp...")
        time.sleep(5)
else:
    print("Vượt quá số lần tự động tải lại. Vui lòng kiểm tra lại cấu hình cookie hoặc kết nối mạng.")

# 3. Giải nén file zip
print("\n" + "="*40)
if os.path.exists(zip_output):
    print("Đang giải nén Video V3C.zip...")
    extract_dir = os.path.join(dataset_dir, "Video_V3C")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_output, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Giải nén hoàn tất!")
else:
    print("Không tìm thấy file Video V3C.zip để giải nén. Quá trình tải có thể đã bị lỗi.")

print("\nHoàn thành script!")
