import sys
import os

# Add the project root to sys.path so we can import the pipeline modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from pipeline.retrieval.segmentation.model import get_segmenters

def test_scene_graph_segmenter():
    print("=== Testing Scene Graph Segmenter ===")
    
    # engine="scenegraph" to load SceneGraphSegmenter
    scene_seg, obj_seg = get_segmenters(scene_engine="regex", object_engine="scenegraph")
    
    test_queries = [
        "A woman in a red dress is playing the piano while a dog sleeps on the sofa.",
        "A red car driving on a dirt road next to a green field.",
        "Two people walking their dog in a crowded park."
    ]
    
    for query in test_queries:
        print(f"\n[Input Query]: {query}")
        
        # Test Scene Segmentation (Regex)
        scenes = scene_seg.segment(query)
        print(f"[Scenes extracted]: {scenes}")
        
        # Test Object Segmentation (SceneGraph)
        for i, scene in enumerate(scenes):
            objects = obj_seg.segment(scene)
            print(f"  -> Scene {i+1} Objects: {objects}")

def test_bert_srl_segmenter():
    print("\n=== Testing BERT SRL Segmenter ===")
    
    # scene_engine="bert_srl" loads BertSRLSegmenter for scenes
    scene_seg, obj_seg = get_segmenters(scene_engine="bert_srl", object_engine="none")
    
    test_queries = [
        "A woman in a red dress is playing the piano. The scene shifts to a dog sleeping on the sofa.",
        "A red car driving on a dirt road next to a green field.",
        "Two people walking their dog in a crowded park."
    ]
    
    for query in test_queries:
        print(f"\n[Input Query]: {query}")
        scenes = scene_seg.segment(query)
        print(f"[Scenes extracted by BERT_SRL]: {scenes}")

def test_spacy_scene_segmenter():
    print("\n=== Testing Spacy Scene Segmenter ===")
    
    # scene_engine="spacy" loads SpacySceneSegmenter for scenes
    scene_seg, obj_seg = get_segmenters(scene_engine="spacy", object_engine="scenegraph")
    
    # Read a few queries from dataset/synthetic_tasks_v0.jsonl
    import json
    dataset_path = os.path.join(project_root, "dataset", "synthetic_tasks_v0.jsonl")
    test_queries = []
    
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for _ in range(5):
                line = f.readline()
                if not line: break
                task = json.loads(line)
                test_queries.append(task.get("description", ""))
    except Exception as e:
        print(f"Error reading dataset: {e}")
        # Fallback if file not found
        test_queries = [
            "A woman in a red dress is playing the piano while a dog sleeps on the sofa."
        ]
    
    for i, query in enumerate(test_queries):
        print(f"\n[Input Query {i+1}]: {query}")
        scenes = scene_seg.segment(query)
        print(f"[Scenes extracted by Spacy]:")
        for j, scene in enumerate(scenes):
            print(f"  -> Scene {j+1}: '{scene}'")
            objects = obj_seg.segment(scene)
            print(f"     Objects (SceneGraph): {objects}")

if __name__ == "__main__":
    # test_scene_graph_segmenter()
    test_bert_srl_segmenter()
    test_spacy_scene_segmenter()
