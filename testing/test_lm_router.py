"""
Test LMRouter on the first 5 tasks of the dataset.
Run from project root:
    PYTHONPATH=. LM_CACHE=false uv run python testing/test_lm_router.py
"""
import json
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load first 5 tasks
TASKS_FILE = "dataset/private_round_tasks.jsonl"
N = 10

with open(TASKS_FILE) as f:
    tasks = [json.loads(line) for _, line in zip(range(N), f)]

queries = [t["description"] for t in tasks]

print("=" * 60)
print(f"Testing LMRouter on first {N} tasks of: {TASKS_FILE}")
print("=" * 60)
for i, q in enumerate(queries, 1):
    print(f"  [{i}] {q}")
print()

# Load router
from pipeline.retrieval.routing.lm_router import LMRouter

router = LMRouter(engine_name="qwen")

print("=" * 60)
print("Running route_batch()...")
print("=" * 60)
routes = router.route_batch(queries)

print()
print("=" * 60)
print("Results:")
print("=" * 60)
for task, route in zip(tasks, routes):
    tid = task.get("task_id", "?")
    desc = task.get("description", "")
    use_asr = route.get("Use_asr", False) and bool(route.get("asr_query"))
    use_ocr = route.get("Use_ocr", False) and bool(route.get("ocr_query"))
    print(f"\nTask {tid}")
    print(f"  Query    : {desc}")
    print(f"  Visual   : ON  (always)")
    print(f"  ASR      : {'ON' if use_asr else 'OFF'}")
    print(f"  ASR query: {route.get('asr_query') or 'NULL'}")
    print(f"  OCR      : {'ON' if use_ocr else 'OFF'}")
    print(f"  OCR query: {route.get('ocr_query') or 'NULL'}")

print()
print("Cleaning up...")
router.cleanup()
print("Done.")
