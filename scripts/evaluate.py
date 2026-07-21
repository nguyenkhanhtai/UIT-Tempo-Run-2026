import os
import glob
import json
import argparse

def evaluate(submission_file):
    annotations_file = "visualizer/annotations.json"
    if not os.path.exists(annotations_file):
        print(f"Lỗi: Không tìm thấy file {annotations_file}")
        return
        
    if not submission_file:
        # Tìm submission mới nhất
        sub_dirs = glob.glob("submission/*")
        valid_dirs = [d for d in sub_dirs if os.path.basename(d).isdigit()]
        if not valid_dirs:
            print("Không tìm thấy thư mục submission nào!")
            return
        valid_dirs.sort(key=lambda x: int(os.path.basename(x)))
        submission_file = os.path.join(valid_dirs[-1], "submission.json")
        
    if not os.path.exists(submission_file):
        print(f"Lỗi: Không tìm thấy file {submission_file}")
        return
        
    print(f"Đang đánh giá file: {submission_file}")
    
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
        
    with open(submission_file, "r") as f:
        data = json.load(f)
        
    preds = {item["task_id"]: item["results"] for item in data.get("predictions", [])}
    
    total_mrr = 0.0
    hits = 0
    top_1_hits = 0
    evaluated_count = 0
    
    for task_id, gt_video in annotations.items():
        if not gt_video or task_id not in preds:
            continue
            
        evaluated_count += 1
        
        task_results = preds.get(task_id, [])
        # Chỉ quan tâm đến video_id, lọc trùng nếu 1 video xuất hiện nhiều lần (chắc chắn logic)
        seen = set()
        video_list = []
        for r in task_results:
            vid = r.get("video_id")
            if vid and vid not in seen:
                seen.add(vid)
                video_list.append(vid)
                
        # Tìm rank của gt_video
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
    if len(preds) == 0:
        print("Không có task nào trong submission.")
        return
        
    total_tasks = len(preds)
    mrr = total_mrr / total_tasks
    
    print("================ KẾT QUẢ ==================")
    print(f"Tổng số task trong submission: {total_tasks}")
    print(f"Số lượng task có GT để đối chiếu: {evaluated_count}")
    print(f"Số task tìm đúng (Hits):     {hits} (trên {total_tasks} task: {hits/total_tasks*100:.2f}%)")
    print(f"Số task đạt Top 1:           {top_1_hits} (trên {total_tasks} task: {top_1_hits/total_tasks*100:.2f}%)")
    print(f"MRR (trên TOÀN BỘ {total_tasks} task): {mrr:.4f}")
    
    config_file = os.path.join(os.path.dirname(submission_file), "config.json")
    if os.path.exists(config_file):
        print("-------------------------------------------")
        print("Thông số đã dùng (Config):")
        try:
            with open(config_file, "r") as fc:
                cfg = json.load(fc)
                args_dict = cfg.get("args", {})
                env_params = cfg.get("env_params", {})
                print(f"  - USE_SEQUENTIAL:   {args_dict.get('use_sequential', False)}")
                print(f"  - USE_AUDIO:        {args_dict.get('use_audio', False)}")
                print(f"  - SCENE_SEGMENTER:  {args_dict.get('scene_segmenter', 'none')}")
                print(f"  - OBJECT_SEGMENTER: {args_dict.get('object_segmenter', 'none')}")
                print(f"  - NUM_EXPANSIONS:   {args_dict.get('num_expansions', 0)}")
                print(f"  - SMOOTHING_WINDOW: {args_dict.get('smoothing_window', 0)}")
                print(f"  - OVERLAP_THRESH:   {args_dict.get('overlap_threshold', 0)}")
                print(f"  - AGG_MODE:         {env_params.get('AGG_MODE', 'mean')}")
        except Exception as e:
            print("  (Lỗi khi đọc config.json)")
    print("===========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", default=None, help="Đường dẫn đến file submission.json. Mặc định: lấy cái mới nhất.")
    args = parser.parse_args()
    
    evaluate(args.submission)
