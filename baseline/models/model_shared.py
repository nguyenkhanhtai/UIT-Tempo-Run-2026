import torch
import gc

_CACHE = {}

def get_shared_model(model_key, load_fn):
    """
    Generic function to cache and share models across tasks/shards.
    model_key: Unique identifier, e.g. ('florence2', 'microsoft/Florence-2-large', 'cuda:0')
    load_fn: Callable that returns the model/processor to be cached.
    """
    if model_key not in _CACHE:
        print(f"[init] Loading Shared Model {model_key}...", flush=True)
        _CACHE[model_key] = load_fn()
    else:
        print(f"[init] Using cached Model {model_key}", flush=True)
        
    return _CACHE[model_key]
