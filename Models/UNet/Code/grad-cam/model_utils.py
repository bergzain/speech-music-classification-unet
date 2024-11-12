# model_utils.py

import sys
import os

# Adjust sys.path to include parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.insert(0, parent_dir)

import torch

# Import models from parent directory
from models import U_Net, R2U_Net, R2AttU_Net, AttentionUNet
#%%
def load_model(model_path: str, model_type: str, device: str) -> torch.nn.Module:
    """Load model with proper device handling."""
    model_classes = {
        'U_Net': lambda: U_Net(img_ch=1, output_ch=2),
        'R2U_Net': lambda: R2U_Net(img_ch=1, output_ch=2),
        'R2AttU_Net': lambda: R2AttU_Net(img_ch=1, output_ch=2),
        'AttentionUNet': lambda: AttentionUNet(mfcc_dim=32, output_ch=2)
    }
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    model = model_classes[model_type]()
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    # Initialize fc layer
    dummy_input = torch.randn(1, 1, 32, 112).to(device)
    _ = model(dummy_input)
    if 'fc.weight' in state_dict:
        model.fc.weight.data = state_dict['fc.weight'].to(device)
        model.fc.bias.data = state_dict['fc.bias'].to(device)
    return model.eval()

def get_target_layer(model: torch.nn.Module, model_type: str) -> torch.nn.Module:
    """Get the target layer for CAM visualization."""
    target_layers = {
        'U_Net': lambda m: m.Up_conv2,
        'AttentionUNet': lambda m: m.Up_conv2,
        'R2U_Net': lambda m: m.Up_RRCNN2,
        'R2AttU_Net': lambda m: m.Up_RRCNN2
    }
    return target_layers[model_type](model)