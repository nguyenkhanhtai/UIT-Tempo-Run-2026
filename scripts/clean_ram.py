import gc
import torch

def clean_memory():
    """
    Utility function to clean up Python's RAM and GPU VRAM (PyTorch).
    Very useful for freeing up residual memory before loading a new model
    into memory or after processing a large amount of data.
    """
    # 1. Run Python's Garbage Collector to delete unused variables in RAM
    gc.collect()
    
    # 2. Clear GPU cache (if using PyTorch and CUDA)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("[Memory] Successfully cleaned up RAM and GPU VRAM.")
    else:
        print("[Memory] Successfully cleaned up RAM (CUDA GPU not found).")

if __name__ == "__main__":
    clean_memory()
