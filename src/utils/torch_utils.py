import torch

def get_torch_device() -> str:
    """ Returns the best available torch device: cuda, mps, or cpu. """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def get_device_map() -> str:
    """ Returns a device map string for transformers pipelines. """
    device = get_torch_device()
    if device == "cuda":
        return "cuda:0"
    elif device == "mps":
        return "mps"
    else:
        return "cpu"
