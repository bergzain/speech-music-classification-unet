# data_utils.py
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
from grad_cam_utils import set_random_seeds
import random

def find_samples_with_seed(model: torch.nn.Module, loader: DataLoader, 
                           device: str, samples_per_class: int = 2, 
                           seed: int = 15) -> Dict[int, List[Tuple[torch.Tensor, int]]]:
    """Find samples with fixed random seed for reproducibility."""
    set_random_seeds(seed)
    class_samples = {0: [], 1: []}
    print("Collecting candidate samples...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            for sample_idx in range(len(labels)):
                label = labels[sample_idx].item()
                if preds[sample_idx] == label:
                    class_samples[label].append((
                        inputs[sample_idx].cpu(),
                        label,
                        batch_idx,
                        sample_idx
                    ))
    selected_samples = {0: [], 1: []}
    for class_idx in class_samples:
        if len(class_samples[class_idx]) >= samples_per_class:
            selected_indices = random.sample(range(len(class_samples[class_idx])), samples_per_class)
            selected_samples[class_idx] = [
                (class_samples[class_idx][i][0], class_samples[class_idx][i][1])
                for i in selected_indices
            ]
            print(f"Selected samples for class {class_idx} from batches: "
                  f"{[class_samples[class_idx][i][2] for i in selected_indices]}, "
                  f"indices: {[class_samples[class_idx][i][3] for i in selected_indices]}")
        else:
            print(f"Warning: Not enough samples for class {class_idx}. "
                  f"Found {len(class_samples[class_idx])}, needed {samples_per_class}")
    return selected_samples