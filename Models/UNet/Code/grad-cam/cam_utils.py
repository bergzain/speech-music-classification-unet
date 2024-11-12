# cam_utils.py
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

class BaseCAM:
    """Base class for all CAM techniques."""
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        def save_activation(module, input, output):
            self.activations = output.detach()

        def save_gradient(grad):
            self.gradients = grad.detach()

        target_layer.register_forward_hook(save_activation)
        target_layer.register_backward_hook(lambda m, grad_in, grad_out: save_gradient(grad_out[0]))

    def forward_backward(self, input_tensor: torch.Tensor, class_idx: Optional[int]) -> Tuple[torch.Tensor, int]:
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        return output, class_idx

class VanillaGradCAM(BaseCAM):
    """Original Grad-CAM implementation."""
    def get_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output, class_idx = self.forward_backward(input_tensor, class_idx)
        with torch.no_grad():
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
            heatmap = F.relu(heatmap)
            heatmap = F.interpolate(heatmap, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap, output

class GradCAMPlusPlus(BaseCAM):
    """Grad-CAM++ implementation."""
    def get_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output, class_idx = self.forward_backward(input_tensor, class_idx)
        with torch.no_grad():
            alpha_num = self.gradients.pow(2)
            alpha_denom = alpha_num.mul(2) + self.activations.mul(self.gradients.pow(3)).sum(dim=[2, 3], keepdim=True)
            alpha = alpha_num.div(alpha_denom + 1e-7)
            weights = (alpha * F.relu(self.gradients)).sum(dim=[2, 3], keepdim=True)
            heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)
            heatmap = F.interpolate(heatmap, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
            heatmap = F.relu(heatmap)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap, output

class ScoreCAM(BaseCAM):
    """Score-CAM implementation."""
    def get_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        with torch.no_grad():
            b, c, h, w = self.activations.shape
            acts = self.activations
            acts = F.interpolate(acts, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
            scores = torch.zeros((b, c), device=input_tensor.device)
            for i in range(c):
                act_mask = acts[:, i:i+1, :, :]
                act_mask = (act_mask - act_mask.min()) / (act_mask.max() - act_mask.min() + 1e-8)
                masked_input = input_tensor * act_mask
                score = self.model(masked_input)
                scores[:, i] = score[:, class_idx]
            scores = F.softmax(scores, dim=1)
            cam = torch.sum(scores.view(b, c, 1, 1) * acts, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output

class LayerCAM(BaseCAM):
    """LayerCAM implementation."""
    def get_heatmap(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        output, class_idx = self.forward_backward(input_tensor, class_idx)
        with torch.no_grad():
            positive_gradients = F.relu(self.gradients)
            weighted_activations = self.activations * positive_gradients
            weights = torch.mean(positive_gradients, dim=(2, 3), keepdim=True)
            heatmap = torch.sum(weighted_activations * weights, dim=1, keepdim=True)
            heatmap = F.relu(heatmap)
            heatmap = F.interpolate(heatmap, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap, output