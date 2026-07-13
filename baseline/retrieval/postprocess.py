"""
Purpose: Applies post-processing algorithms like KMeans clustering on raw image
features to diversify the final Top-K retrieval results.
"""
import numpy as np

def apply_clustering(task, candidates, default_top_videos):
    max_preds = task.get("max_predictions", default_top_videos)
    
    num_clusters = min(max_preds, len(candidates))
    if num_clusters > 1:
        from sklearn.cluster import KMeans
        X = np.array([c["feat"] for c in candidates])
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit(X)
        labels = kmeans.labels_
        
        clusters = {i: [] for i in range(num_clusters)}
        for i, c in enumerate(candidates):
            clusters[labels[i]].append(c)
            
        final_results = []
        for i in range(num_clusters):
            if clusters[i]:
                best_in_cluster = sorted(clusters[i], key=lambda x: x["sim"], reverse=True)[0]
                final_results.append(best_in_cluster)
                
        final_results = sorted(final_results, key=lambda x: x["sim"], reverse=True)
    else:
        final_results = candidates[:max_preds]
        
    results = []
    for rank, res in enumerate(final_results[:max_preds], 1):
        results.append({
            "rank": rank, "video_id": res["video_id"],
            "frame_ms": res["frame_ms"],
        })
    return {"task_id": task["task_id"], "results": results}
