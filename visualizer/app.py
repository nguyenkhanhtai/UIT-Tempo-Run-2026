import os
import json
import urllib.parse
import glob
import numpy as np
from pathlib import Path
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
            
        if self.path == '/api/submissions':
            self.handle_get_submissions()
            return
            
        if self.path.startswith('/api/submissions/'):
            sub_id = self.path.split('/')[-1]
            self.handle_get_submission_detail(sub_id)
            return
            
        if self.path.startswith('/api/frame/'):
            parts = self.path.split('/')
            if len(parts) >= 5:
                video_id = parts[3]
                frame_ms = parts[4]
                self.handle_get_frame(video_id, frame_ms)
            else:
                self.send_error(400, "Bad Request")
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
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(task_list).encode())

    def handle_get_submission_detail(self, sub_id):
        sub_file = os.path.abspath(os.path.join(BASE_DIR, "..", "submission", sub_id, "submission.json"))
        if not os.path.exists(sub_file):
            self.send_error(404, "Submission not found")
            return
        try:
            with open(sub_file, "r") as f:
                data = json.load(f)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        except Exception as e:
            self.send_error(500, f"Error reading submission: {e}")

    def handle_get_frame(self, vid, frame_ms):
        kf_dir = os.path.abspath(os.path.join(BASE_DIR, "..", "keyframes", "fps", "1"))
        vdir = Path(kf_dir) / vid
        ts_path = vdir / "ts_ms.npy"
        
        if not ts_path.exists():
            self.send_error(404, "Frame index not found")
            return
            
        try:
            ts = np.load(ts_path)
            idx = np.argmin(np.abs(ts - float(frame_ms)))
            k_filename = f"k_{idx + 1:05d}.jpg"
            img_path = vdir / k_filename
            
            if img_path.exists():
                self.send_response(200)
                self.send_header("Content-type", "image/jpeg")
                self.end_headers()
                with open(img_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Image file not found")
        except Exception as e:
            print(e)
            self.send_error(500, "Internal Server Error")

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
        annotations = load_annotations()
        gt_video = annotations.get(task_id, None)
        
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "task_id": task_id, 
            "gt_video": gt_video,
            "images": images
        }).encode('utf-8'))

    def handle_get_submissions(self):
        sub_dirs = glob.glob(os.path.abspath(os.path.join(BASE_DIR, "..", "submission", "*")))
        valid_dirs = [d for d in sub_dirs if os.path.basename(d).isdigit()]
        valid_dirs.sort(key=lambda x: int(os.path.basename(x)), reverse=True)
        
        annotations = load_annotations()
        
        subs_data = []
        for d in valid_dirs:
            sub_id = os.path.basename(d)
            sub_file = os.path.join(d, "submission.json")
            conf_file = os.path.join(d, "config.json")
            
            if not os.path.exists(sub_file):
                continue
                
            try:
                with open(sub_file, "r") as f:
                    data = json.load(f)
            except:
                continue
                
            preds = {item["task_id"]: item["results"] for item in data.get("predictions", [])}
            
            total_mrr = 0.0
            hits = 0
            top_1_hits = 0
            evaluated_count = 0
            
            for task_id, gt_video in annotations.items():
                if not gt_video:
                    continue
                evaluated_count += 1
                task_results = preds.get(task_id, [])
                seen = set()
                video_list = []
                for r in task_results:
                    vid = r.get("video_id")
                    if vid and vid not in seen:
                        seen.add(vid)
                        video_list.append(vid)
                rank = -1
                for i, vid in enumerate(video_list):
                    if vid == gt_video:
                        rank = i + 1
                        break
                if rank != -1:
                    hits += 1
                    total_mrr += 1.0 / rank
                    if rank == 1:
                        top_1_hits += 1
            mrr = total_mrr / len(preds) if len(preds) > 0 else 0
            

            cfg_params = {}
            bash_script = "# No config found"
            if os.path.exists(conf_file):
                try:
                    with open(conf_file, "r") as fc:
                        cfg = json.load(fc)
                        args_dict = cfg.get("args", {})
                        env_params = cfg.get("env_params", {})
                        
                        cfg_params = {
                            "USE_SEQUENTIAL": args_dict.get('use_sequential', False),
                            "USE_AUDIO": args_dict.get('use_audio', False),
                            "SCENE_SEGMENTER": args_dict.get('scene_segmenter', 'none'),
                            "OBJECT_SEGMENTER": args_dict.get('object_segmenter', 'none'),
                            "NUM_EXPANSIONS": args_dict.get('num_expansions', 0),
                            "SMOOTHING_WINDOW": args_dict.get('smoothing_window', 0),
                            "OVERLAP_THRESH": args_dict.get('overlap_threshold', 0),
                            "AGG_MODE": env_params.get('AGG_MODE', 'mean')
                        }
                        
                        script_lines = [
                            f"export USE_SEQUENTIAL=\"{'true' if cfg_params['USE_SEQUENTIAL'] else 'false'}\"",
                            f"export USE_AUDIO=\"{'true' if cfg_params['USE_AUDIO'] else 'false'}\"",
                            f"export SCENE_SEGMENTER=\"{cfg_params['SCENE_SEGMENTER']}\"",
                            f"export OBJECT_SEGMENTER=\"{cfg_params['OBJECT_SEGMENTER']}\"",
                            f"export NUM_EXPANSIONS={cfg_params['NUM_EXPANSIONS']}",
                            f"export SMOOTHING_WINDOW={cfg_params['SMOOTHING_WINDOW']}",
                            f"export OVERLAP_THRESHOLD={cfg_params['OVERLAP_THRESH']}",
                            f"export AGG_MODE=\"{cfg_params['AGG_MODE']}\"",
                            "./scripts/retrieval.sh"
                        ]
                        bash_script = "\\n".join(script_lines)
                except:
                    pass
            
            subs_data.append({
                "id": sub_id,
                "total_tasks": len(preds),
                "evaluated_count": evaluated_count,
                "hits": hits,
                "top1_hits": top_1_hits,
                "mrr": round(mrr, 4),
                "config": cfg_params,
                "script": bash_script
            })
            
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(subs_data).encode('utf-8'))

if __name__ == "__main__":
    os.chdir(BASE_DIR)
    print(f"Starting server on http://localhost:{PORT}")
    server = ThreadingHTTPServer(('', PORT), AnnotationHandler)
    server.serve_forever()
