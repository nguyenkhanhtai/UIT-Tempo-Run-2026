from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import os

def upload_file(file_path):
    # 1. Xác thực tài khoản
    gauth = GoogleAuth()
    
    # Tạo file credentials cục bộ để không phải đăng nhập lại mỗi lần chạy
    # Nó sẽ tự động tải file client_secrets.json trong cùng thư mục
    gauth.LoadCredentialsFile("mycreds.txt")
    if gauth.credentials is None:
        # Lần đầu tiên chạy, mở trình duyệt để xác thực
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh token nếu hết hạn
        gauth.Refresh()
    else:
        # Nếu đã có token thì authenticate trực tiếp
        gauth.Authorize()
        
    # Lưu credentials cho lần sau
    gauth.SaveCredentialsFile("mycreds.txt")

    drive = GoogleDrive(gauth)

    # 2. Upload file
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return

    print(f"Đang upload file {file_path}...")
    file_name = os.path.basename(file_path)
    
    # CreateFile tạo một đối tượng GoogleDriveFile
    file_drive = drive.CreateFile({'title': file_name}) 
    file_drive.SetContentFile(file_path)
    file_drive.Upload()

    print(f"Upload thành công! Tên file trên Drive: {file_name}")
    print(f"ID của file là: {file_drive['id']}")

if __name__ == '__main__':
    # Đổi tên file ở đây thành file bạn muốn upload
    target_file = 'V3C.zip' 
    upload_file(target_file)
