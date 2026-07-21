import os
import glob
import json
import csv
import argparse

def get_latest_submission():
    sub_dirs = glob.glob("submission/*/")
    if not sub_dirs:
        return None
    # Sort by the integer folder name if possible
    try:
        sub_dirs.sort(key=lambda x: int(os.path.basename(os.path.normpath(x))))
    except ValueError:
        sub_dirs.sort(key=os.path.getmtime)
    
    latest_dir = sub_dirs[-1]
    return os.path.join(latest_dir, "submission.json")

def evaluate_submission(submission_path, labels_path="dataset/synthetic_eval_labels.csv"):
    if not os.path.exists(submission_path):
        print(f"Error: Submission file not found: {submission_path}")
        return
        
    if not os.path.exists(labels_path):
        print(f"Error: Labels file not found: {labels_path}")
        return
        
    print(f"Evaluating {submission_path} against {labels_path}...")
    
    # 1. Load Ground Truth Labels (CSV)
    gt = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row['task_id']
            gt[task_id] = {
                'video_id': row['answer_video_id'],
                'start_ms': int(row['answer_start_ms']),
                'end_ms': int(row['answer_end_ms'])
            }
            
    # 2. Load Submission (JSON dict with 'predictions' list)
    with open(submission_path, 'r', encoding='utf-8') as f:
        sub_data = json.load(f)
        
    # Convert submission to a dictionary keyed by task_id
    preds = {}
    for task in sub_data.get('predictions', []):
        preds[task['task_id']] = task['results']
        
    # 3. Calculate Metrics
    evaluated_tasks = 0
    mrr_sum = 0.0
    recall_1 = 0
    recall_5 = 0
    recall_10 = 0
    
    # Strict metrics
    strict_mrr_sum = 0.0
    strict_recall_1 = 0
    strict_recall_5 = 0
    strict_recall_10 = 0
    
    for task_id, gt_info in gt.items():
        if task_id not in preds:
            print(f"Warning: Task {task_id} is in GT but missing from submission.")
            continue
            
        evaluated_tasks += 1
        ans_vid = gt_info['video_id']
        start_ms = gt_info['start_ms']
        end_ms = gt_info['end_ms']
        
        results = preds[task_id]
        
        # Standard Evaluation (Video Match Only)
        hit_rank = -1
        for res in results:
            if res['video_id'] == ans_vid:
                hit_rank = res['rank']
                break
                
        if hit_rank > 0:
            mrr_sum += 1.0 / hit_rank
            if hit_rank <= 1: recall_1 += 1
            if hit_rank <= 5: recall_5 += 1
            if hit_rank <= 10: recall_10 += 1
            
        # Strict Evaluation (Video Match + Temporal Match)
        strict_hit_rank = -1
        for res in results:
            if res['video_id'] == ans_vid and start_ms <= res['frame_ms'] <= end_ms:
                strict_hit_rank = res['rank']
                break
                
        if strict_hit_rank > 0:
            strict_mrr_sum += 1.0 / strict_hit_rank
            if strict_hit_rank <= 1: strict_recall_1 += 1
            if strict_hit_rank <= 5: strict_recall_5 += 1
            if strict_hit_rank <= 10: strict_recall_10 += 1
            
        # Detailed logging for each task
        print(f"Task {task_id}: ", end="")
        if hit_rank > 0:
            strict_str = f"Rank {strict_hit_rank}" if strict_hit_rank > 0 else "MISS"
            print(f"Standard Rank = {hit_rank:<2} | Strict Rank = {strict_str}")
        else:
            print(f"MISS (Not found in Top 10)")

    # 4. Print Results
    total_tasks = len(tasks_data)
    if total_tasks == 0:
        print("Không có task nào trong submission.")
        return

    print(f"\nEvaluation Results (trên TOÀN BỘ {total_tasks} queries trong submission):")
    print(f"Số lượng query có Ground Truth để đối chiếu: {evaluated_tasks}")
    print(f"----------------------------------------")
    print(f" Theo luật Loose (Overlap):")
    print(f" Recall@1 : {recall_1 / total_tasks * 100:.2f}%")
    print(f" Recall@5 : {recall_5 / total_tasks * 100:.2f}%")
    print(f" Recall@10: {recall_10 / total_tasks * 100:.2f}%")
    print(f" MRR      : {mrr_sum / total_tasks:.4f}")
    
    print(f"\n Theo luật Strict (Chính xác Keyframe ID):")
    print(f" Strict R@1 : {strict_recall_1 / total_tasks * 100:.2f}%")
    print(f" Strict R@5 : {strict_recall_5 / total_tasks * 100:.2f}%")
    print(f" Strict R@10: {strict_recall_10 / total_tasks * 100:.2f}%")
    print(f" Strict MRR : {strict_mrr_sum / total_tasks:.4f}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate submission against synthetic labels.")
    parser.add_argument("--sub", type=str, default=None, help="Path to submission.json. Defaults to latest.")
    parser.add_argument("--gt", type=str, default="dataset/synthetic_eval_labels.csv", help="Path to GT CSV.")
    
    args = parser.parse_args()
    
    sub_path = args.sub if args.sub else get_latest_submission()
    
    if sub_path:
        evaluate_submission(sub_path, args.gt)
    else:
        print("Could not automatically find a submission.json file.")
