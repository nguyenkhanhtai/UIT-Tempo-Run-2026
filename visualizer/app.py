import os
import json
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

PORT = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "figures", "task"))
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "annotations.json")

def load_annotations():
    if os.path.exists(ANNOTATIONS_FILE):
        try:
            with open(ANNOTATIONS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_annotations(data):
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)

class AnnotationHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Prevent caching
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
        
    def do_GET(self):
        if self.path == '/':
            self.path = '/templates/index.html'
            
        if self.path == '/api/tasks':
            self.handle_get_tasks()
            return
            
        if self.path.startswith('/api/tasks/'):
            task_id = self.path.split('/')[-1]
            self.handle_get_task_images(task_id)
            return
            
        if self.path.startswith('/images/'):
            # Route /images/T0002/file.jpg to figures/task/T0002/file.jpg
            parts = self.path.split('/')
            task_id = parts[2]
            filename = urllib.parse.unquote(parts[3])
            
            filepath = os.path.join(FIGURES_DIR, task_id, filename)
            if os.path.exists(filepath):
                self.send_response(200)
                if filename.endswith(".png"):
                    self.send_header("Content-type", "image/png")
                else:
                    self.send_header("Content-type", "image/jpeg")
                self.end_headers()
                with open(filepath, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "File not found")
            return
            
        if self.path.startswith('/video/'):
            # Route /video/v3c1_04089 to dataset/Video_V3C/V3C1/videos/04089/04089.mp4
            video_id = self.path.split('/')[-1] # v3c1_04089
            try:
                parts = video_id.split('_')
                if len(parts) >= 2:
                    collection = parts[0].upper() # V3C1 or V3C2
                    vid = parts[1] # 04089
                    vid_path = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset", "Video_V3C", collection, "videos", vid, f"{vid}.mp4"))
                    
                    if os.path.exists(vid_path):
                        file_size = os.path.getsize(vid_path)
                        range_header = self.headers.get('Range', None)
                        
                        if range_header:
                            byte_range = range_header.strip().split('=')[-1]
                            parts = byte_range.split('-')
                            start = int(parts[0]) if parts[0] else 0
                            end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
                            if start >= file_size:
                                self.send_error(416, 'Requested Range Not Satisfiable')
                                return
                            chunk_size = end - start + 1
                            
                            self.send_response(206)
                            self.send_header('Content-type', 'video/mp4')
                            self.send_header('Accept-Ranges', 'bytes')
                            self.send_header('Content-Range', f'bytes {start}-{end}/{file_size}')
                            self.send_header('Content-Length', str(chunk_size))
                            self.end_headers()
                            
                            with open(vid_path, 'rb') as f:
                                f.seek(start)
                                remaining = chunk_size
                                while remaining > 0:
                                    chunk = f.read(min(8192, remaining))
                                    if not chunk:
                                        break
                                    self.wfile.write(chunk)
                                    remaining -= len(chunk)
                        else:
                            self.send_response(200)
                            self.send_header("Content-type", "video/mp4")
                            self.send_header('Accept-Ranges', 'bytes')
                            self.send_header('Content-Length', str(file_size))
                            self.end_headers()
                            with open(vid_path, 'rb') as f:
                                while True:
                                    chunk = f.read(8192)
                                    if not chunk:
                                        break
                                    self.wfile.write(chunk)
                        return
            except Exception as e:
                print(e)
            self.send_error(404, "Video not found")
            return
            
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/annotate':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            task_id = data.get("task_id")
            annotation = data.get("annotation")
            
            if not task_id:
                self.send_error(400, "Missing task_id")
                return
                
            annotations = load_annotations()
            
            # The user requested to only save the video_id
            if annotation and isinstance(annotation, dict) and "video_id" in annotation:
                annotations[task_id] = annotation["video_id"]
            else:
                annotations[task_id] = None # skipped
                
            save_annotations(annotations)
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
            return

    def handle_get_tasks(self):
        if not os.path.exists(FIGURES_DIR):
            tasks = []
        else:
            tasks = []
            for t_dir in sorted(os.listdir(FIGURES_DIR)):
                if t_dir.startswith("T") and os.path.isdir(os.path.join(FIGURES_DIR, t_dir)):
                    tasks.append(t_dir)
                    
        annotations = load_annotations()
        
        task_list = []
        for t in tasks:
            status = "pending"
            if t in annotations:
                if annotations[t] is None:
                    status = "skipped"
                else:
                    status = "annotated"
            task_list.append({"id": t, "status": status})
            
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(task_list).encode('utf-8'))

    def handle_get_task_images(self, task_id):
        task_dir = os.path.join(FIGURES_DIR, task_id)
        if not os.path.exists(task_dir):
            self.send_error(404, "Task not found")
            return
            
        images = []
        for f in sorted(os.listdir(task_dir)):
            if f.endswith(".jpg") or f.endswith(".png"):
                parts = f.replace(".jpg", "").replace(".png", "").split("_")
                rank = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 999
                # video_id is everything after rank until the end
                video_id = "_".join(parts[2:]) if len(parts) > 2 else f
                images.append({
                    "filename": f,
                    "rank": rank,
                    "video_id": video_id,
                    "url": f"/images/{task_id}/{f}"
                })
                
        images.sort(key=lambda x: x["rank"])
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"task_id": task_id, "images": images}).encode('utf-8'))

if __name__ == "__main__":
    os.chdir(BASE_DIR)
    print(f"Starting server on http://localhost:{PORT}")
    server = ThreadingHTTPServer(('', PORT), AnnotationHandler)
    server.serve_forever()
