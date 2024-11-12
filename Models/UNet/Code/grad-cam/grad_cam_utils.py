import torch
import numpy as np
import random
import os
import re

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_built():
        return "mps"
    else:
        return "cpu"

def parse_model_filename(model_path):
    """Parse the model filename to extract parameters."""
    filename = os.path.splitext(os.path.basename(model_path))[0]
    pattern = r'^(.*?)_([^_]+(?:-[^_]+)*)_(\d+)_len([\d\.]+)S$'
    match = re.match(pattern, filename)
    if match:
        model_type = match.group(1)
        feature_type = match.group(2)
        n_mfcc = int(match.group(3))
        length_in_seconds = float(match.group(4))
        return model_type, feature_type, n_mfcc, length_in_seconds
    else:
        raise ValueError(f"Model filename '{filename}' does not match the expected pattern.")