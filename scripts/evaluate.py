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
        if not gt_video:
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
    if evaluated_count == 0:
        print("Không có task nào có ground truth trong annotations.json.")
        return
        
    mrr = total_mrr / evaluated_count
    
    print("================ KẾT QUẢ ==================")
    print(f"Tổng số task trong submission: {len(preds)}")
    print(f"Số lượng task có GT để đánh giá: {evaluated_count}")
    print(f"Số task tìm đúng (Hits):     {hits} ({hits/evaluated_count*100:.2f}%)")
    print(f"Số task đạt Top 1:           {top_1_hits} ({top_1_hits/evaluated_count*100:.2f}%)")
    print(f"MRR (Mean Reciprocal Rank):  {mrr:.4f}")
    print("===========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", nargs="?", default=None, help="Đường dẫn đến file submission.json. Mặc định: lấy cái mới nhất.")
    args = parser.parse_args()
    
    evaluate(args.submission)
