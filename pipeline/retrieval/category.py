import torch
import torch.nn.functional as F
import numpy as np

V3C_CATEGORIES = [
    "Art and Design, painting, drawing, sculpture, performance, museum, architecture, creativity",
    "Camera Techniques and Drones, slow motion, macro, drone footage, time lapse, panning, photography",
    "Comedy, funny, humor, stand-up, prank, sketch, laughing",
    "Documentary, real life footage, historical events, nature wildlife, culture, people and society",
    "Fashion, clothing, makeup, runway, models, style, apparel, beauty",
    "Food, cooking, recipes, dining, eating, restaurant, meals, ingredients",
    "Instructionals and How-to, tutorial, DIY, guide, making of, step by step, educational",
    "News and Journalism, report, broadcast, interview, press, politics, current events",
    "Music, concert, singing, playing instruments, band, performance, music video",
    "Narrative and Fiction, short film, movie, acting, story, cinematic, drama",
    "Personal Vlog, daily life, talking to camera, lifestyle, blog, personal update",
    "Products and Reviews, unboxing, testing, gadget review, technology review, showcase",
    "Sports, competition, athletes, game, training, exercise, action, fitness",
    "Technology, computers, gadgets, software, hardware, robots, programming, future",
    "Travel, tourism, vacation, exploring, landmarks, cityscapes, outdoors, road trip"
]

def predict_query_categories(queries, st_model, device="cuda", top_k=5):
    """
    Predicts the top-K category indices for a list of string queries.
    Uses SentenceTransformers model for high-quality text-to-text similarity.
    Returns a tensor of shape [len(queries), top_k] containing category indices.
    """
    cat_embeddings = st_model.encode(V3C_CATEGORIES, convert_to_tensor=True, device=device)
    q_embeddings = st_model.encode(queries, convert_to_tensor=True, device=device)
    
    # Cosine similarity
    cat_embeddings = F.normalize(cat_embeddings, p=2, dim=1)
    q_embeddings = F.normalize(q_embeddings, p=2, dim=1)
    
    sims = q_embeddings @ cat_embeddings.T
    topk_indices = sims.topk(top_k, dim=1).indices
    return topk_indices

def predict_frame_categories(clip_model, idx_chunk_np, vids_chunk_np, device="cuda"):
    """
    Predicts the smoothed category index for a chunk of frames.
    Uses CLIP zero-shot text encoder to embed category names.
    Applies a 5-frame sliding window majority vote per video.
    Returns a 1D tensor of length len(idx_chunk_np) containing the smoothed category index.
    """
    # Create prompts for CLIP
    prompts = [f"A photo about {cat}" for cat in V3C_CATEGORIES]
    cat_embeddings = clip_model.encode_texts(prompts) # returns numpy array
    cat_embeddings = torch.from_numpy(cat_embeddings).to(device).float()
    
    idx_chunk = torch.from_numpy(idx_chunk_np).to(device).float()
    
    # Cosine similarity
    cat_embeddings = F.normalize(cat_embeddings, p=2, dim=1)
    idx_chunk = F.normalize(idx_chunk, p=2, dim=1)
    
    # cat_embeddings: [15, D], idx_chunk: [B, D] -> sims: [15, B]
    sims = cat_embeddings @ idx_chunk.T
    raw_cats = sims.argmax(dim=0).cpu().numpy() # [B]
    
    # Smooth with 5-frame majority vote within each video
    smoothed_cats = np.zeros_like(raw_cats)
    n = len(raw_cats)
    
    # Simple sliding window of size 5
    # For a frame i, we look at [i-2, i-1, i, i+1, i+2]
    for i in range(n):
        vid = vids_chunk_np[i]
        start = max(0, i - 2)
        end = min(n, i + 3)
        
        # Collect categories of adjacent frames that belong to the SAME video
        votes = []
        for j in range(start, end):
            if vids_chunk_np[j] == vid:
                votes.append(raw_cats[j])
        
        # Majority vote (Mode)
        if votes:
            smoothed_cats[i] = max(set(votes), key=votes.count)
        else:
            smoothed_cats[i] = raw_cats[i]
            
    return torch.from_numpy(smoothed_cats).to(device).long()
