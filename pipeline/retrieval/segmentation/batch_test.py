"""Quick test for the new LMRouter batch routing."""
from pipeline.retrieval.routing.lm_router import LMRouter

router = LMRouter(engine_name="qwen")

texts = ["find the cat", "someone says hello", "a man walking in the park"]
routes = router.route_batch(texts)

for text, route in zip(texts, routes):
    print(f"Query: {text!r:50s} -> {route}")

router.cleanup()
