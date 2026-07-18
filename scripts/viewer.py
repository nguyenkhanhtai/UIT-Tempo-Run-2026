import gradio as gr
import os
import subprocess

def load_video(collection_id, video_id, timestamp_ms):
    # Clean inputs
    collection_id = collection_id.strip().upper()
    video_id = video_id.strip()
    timestamp_ms = timestamp_ms.strip()
    
    video_path = os.path.join("dataset", "Video_V3C", collection_id, "videos", video_id, f"{video_id}.mp4")
    
    if not os.path.exists(video_path):
        return f"<p style='color:red; font-size: 16px;'>❌ Không tìm thấy file video tại: <b>{video_path}</b></p>"
    
    # Convert ms to seconds for HTML5 Video
    try:
        t_sec = float(timestamp_ms) / 1000.0
    except ValueError:
        t_sec = 0.0
        
    abs_path = os.path.abspath(video_path)
    
    html_code = f"""
    <div style="display: flex; justify-content: center; margin-top: 20px;">
        <video width="800" controls autoplay>
            <source src="/file={abs_path}#t={t_sec}" type="video/mp4">
            Trình duyệt của bạn không hỗ trợ thẻ video.
        </video>
    </div>
    <p style="text-align: center; margin-top: 10px; font-size: 16px;">
        Đang phát: <b>{video_id}.mp4</b> từ <b>{t_sec}s</b>
    </p>
    """
    return html_code

def generate_snippet(collection_id, video_id, timestamp_ms):
    collection_id = collection_id.strip().upper()
    video_id = video_id.strip()
    timestamp_ms = timestamp_ms.strip()
    
    video_path = os.path.join("dataset", "Video_V3C", collection_id, "videos", video_id, f"{video_id}.mp4")
    
    if not os.path.exists(video_path):
        return None, f"❌ Không tìm thấy file: {video_path}"
    
    try:
        t_sec = float(timestamp_ms) / 1000.0
    except ValueError:
        t_sec = 0.0
        
    out_path = f"/tmp/{video_id}_{int(t_sec)}.mp4"
    start_time = max(0, t_sec - 2) # Bắt đầu sớm 2s để dễ xem
    
    cmd = [
        "ffmpeg", "-y", 
        "-ss", str(start_time),
        "-i", video_path, 
        "-t", "10", 
        "-c:v", "libx264", 
        "-preset", "ultrafast", 
        "-c:a", "aac", 
        out_path
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path, f"✅ Đã tạo đoạn cắt 10s (từ {start_time:.1f}s)"
    except Exception as e:
        return None, f"❌ Lỗi ffmpeg: {str(e)}"

with gr.Blocks(title="V3C Video Viewer") as demo:
    gr.Markdown("<h1 style='text-align: center;'>🎬 V3C Quick Video Viewer</h1>")
    gr.Markdown("<p style='text-align: center;'>Video V3C được mã hóa bằng <b>HEVC (H.265)</b> nên trình duyệt Chrome/Edge thường không xem được trực tiếp.<br>Khuyên dùng nút <b>Cắt & Đổi đuôi 10s (H.264)</b> để xem mượt mà nhất!</p>")
    
    with gr.Row():
        col_id = gr.Textbox(label="Collection ID", placeholder="VD: V3C1", value="V3C1")
        vid_id = gr.Textbox(label="Video ID", placeholder="VD: 00001", value="00001")
        ts_ms = gr.Textbox(label="Timestamp (milliseconds)", placeholder="VD: 15200", value="15200")
        
    with gr.Row():
        btn_snippet = gr.Button("✂️ Cắt & Đổi đuôi 10s (Khuyên dùng)", variant="primary")
        btn = gr.Button("▶️ Mở HTML5 Trực tiếp (Chỉ hỗ trợ Safari)", variant="secondary")
    
    with gr.Tab("Trình phát Snippet (H.264)"):
        status_text = gr.Markdown()
        vid_player = gr.Video(label="Video Snippet (10 giây)", width=800)
        
    with gr.Tab("Trình phát HTML5 (Bản gốc)"):
        output_html = gr.HTML()
    
    btn.click(fn=load_video, inputs=[col_id, vid_id, ts_ms], outputs=output_html)
    btn_snippet.click(fn=generate_snippet, inputs=[col_id, vid_id, ts_ms], outputs=[vid_player, status_text])
    


if __name__ == "__main__":
    # allowed_paths is required for Gradio to serve local files via /file=
    demo.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/workspace"])
