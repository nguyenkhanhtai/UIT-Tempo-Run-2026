import json
import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_keyframe_path(video_id, frame_ms, kf_dir="keyframes"):
    vdir = os.path.join(kf_dir, video_id)
    ts_file = os.path.join(vdir, "ts_ms.npy")
    if not os.path.exists(ts_file):
        return None
    
    ts = np.load(ts_file)
    # Find closest timestamp index
    idx = np.argmin(np.abs(ts - frame_ms))
    
    # Get sorted image files
    files = sorted(glob.glob(os.path.join(vdir, "k_*.jpg")))
    if idx < len(files):
        return files[idx]
    return None

def get_text_size(text, font, draw):
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    elif hasattr(font, "getsize"):
        return font.getsize(text)
    else:
        return len(text) * 6, 12

def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    curr_line = []
    for word in words:
        curr_line.append(word)
        line_str = " ".join(curr_line)
        w, _ = get_text_size(line_str, font, draw)
            
        if w > max_width and len(curr_line) > 1:
            curr_line.pop()
            lines.append(" ".join(curr_line))
            curr_line = [word]
    if curr_line:
        lines.append(" ".join(curr_line))
    return lines

def main():
    import sys
    tasks_file = "dataset/Public_round_tasks.jsonl"
    
    if len(sys.argv) > 1:
        submission_file = sys.argv[1]
    else:
        sub_dirs = glob.glob("submission/*")
        valid_dirs = [d for d in sub_dirs if os.path.basename(d).isdigit()]
        if not valid_dirs:
            print("No submissions found in submission/ directory.")
            return
        valid_dirs.sort(key=lambda x: int(os.path.basename(x)))
        submission_file = os.path.join(valid_dirs[-1], "submission.json")
        print(f"Auto-selected latest submission: {submission_file}")
        
    out_dir = "figures/task"
    kf_dir = "keyframes"
    
    if not os.path.exists(submission_file):
        print(f"File not found: {submission_file}. Please generate it first.")
        return
        
    # Read tasks
    tasks = {}
    with open(tasks_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            t = json.loads(line)
            tasks[t["task_id"]] = t["description"]
            
    # Read submission predictions
    with open(submission_file, 'r', encoding='utf-8') as f:
        sub = json.load(f)
        
    # Attempt to load a better font, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except IOError:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except IOError:
            font = ImageFont.load_default()
            
    temp_img = Image.new("RGB", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)
    _, line_height = get_text_size("A", font, temp_draw)
    line_height = max(line_height, 20)
        
    for pred in sub.get("predictions", []):
        task_id = pred["task_id"]
        if task_id not in tasks:
            continue
            
        desc = tasks[task_id]
        
        task_out_dir = os.path.join(out_dir, task_id)
        os.makedirs(task_out_dir, exist_ok=True)
        
        print(f"Processing Task: {task_id}")
        
        for res in pred["results"]:
            rank = res["rank"]
            if rank > 10:
                continue
                
            video_id = res["video_id"]
            frame_ms = res["frame_ms"]
            
            img_path = get_keyframe_path(video_id, frame_ms, kf_dir=kf_dir)
            if not img_path:
                print(f"  Warning: Image not found for {video_id} {frame_ms}")
                continue
                
            try:
                frame = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
                continue
                
            padding = 20
            max_text_width = frame.width - 2 * padding
            
            full_text = f"Rank: {rank} | Video: {video_id} | Time: {frame_ms}ms\n\nInstruction: {desc}"
            
            lines = []
            for text_line in full_text.split('\n'):
                if text_line.strip() == "":
                    lines.append("")
                else:
                    lines.extend(wrap_text(text_line, font, max_text_width, temp_draw))
            
            text_height = len(lines) * int(line_height * 1.5) + 2 * padding
            
            # Create new image with white background
            new_img = Image.new("RGB", (frame.width, frame.height + text_height), "white")
            draw = ImageDraw.Draw(new_img)
            
            y = padding
            for line in lines:
                draw.text((padding, y), line, font=font, fill="black")
                y += int(line_height * 1.5)
                
            # Paste the keyframe below the text
            new_img.paste(frame, (0, text_height))
            
            out_file = os.path.join(task_out_dir, f"rank_{rank:02d}_{video_id}.jpg")
            new_img.save(out_file)
            print(f"  -> Saved {out_file}")

if __name__ == '__main__':
    main()
