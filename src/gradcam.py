"""
Native PyTorch Explainable AI (XAI) Engine:
- Grad-CAM: Class Activation Mapping via Gradient-Weighted Feature Maps
- Grad-CAM++: Generalized CAM addressing multiple instances and finer localization
- Deletion Test: Quantitative faithfulness validation via attention-guided pixel masking
- Visualization and Colormap Overlay Utilities
"""
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradCAM:
    """
    Grad-CAM & Grad-CAM++ implementation with hook management.
    Compatible with ResNet, EfficientNet, MobileNet, and Scratch CNN.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # grad_output[0] shape: (B, C, H, W)
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = "gradcam"  # 'gradcam' or 'gradcam++'
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute CAM heatmap for an input tensor (1, C, H, W).
        Returns: (heatmap [H, W], predicted_class, confidence)
        """
        self.model.eval()
        self.model.zero_grad()

        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        score = logits[0, target_class]
        confidence = float(probs[0, target_class].item())

        score.backward(retain_graph=True)

        activations = self.activations[0]  # (C, H, W)
        gradients = self.gradients[0]      # (C, H, W)

        if method.lower() == "gradcam++":
            # Grad-CAM++ formulation:
            # alpha_k = sum_{i,j} (grad^2 / (2*grad^2 + sum_{a,b}(act * grad^3))) * relu(grad)
            grad_2 = gradients.pow(2)
            grad_3 = gradients.pow(3)
            sum_activations = activations.sum(dim=(1, 2), keepdim=True)
            eps = 1e-8
            denominator = 2.0 * grad_2 + sum_activations * grad_3
            denominator = torch.where(denominator != 0.0, denominator, torch.ones_like(denominator) * eps)
            alpha = grad_2 / denominator
            alpha = alpha * F.relu(gradients)
            weights = alpha.sum(dim=(1, 2))  # (C,)
        else:
            # Standard Grad-CAM: Global Average Pooling of gradients
            weights = gradients.mean(dim=(1, 2))  # (C,)

        # Linear combination of feature maps weighted by alpha
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Pass through ReLU to keep features that have positive influence
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        heatmap = cam.cpu().numpy()

        # Resize heatmap to match input dimensions
        _, _, H, W = input_tensor.shape
        heatmap_resized = cv2.resize(heatmap, (W, H))

        return heatmap_resized, target_class, confidence

    def remove_hooks(self) -> None:
        """Clean up hooks when done."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

def overlay_cam_on_image(
    rgb_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay 2D normalized heatmap [0, 1] onto RGB uint8 image [0, 255].
    Returns RGB uint8 blended image.
    """
    if rgb_img.dtype != np.uint8:
        rgb_img = (rgb_img * 255).astype(np.uint8)

    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = np.float32(heatmap_colored) * alpha + np.float32(rgb_img) * (1.0 - alpha)
    blended = np.uint8(np.clip(blended, 0, 255))
    return blended

def run_deletion_test(
    model: nn.Module,
    cam_engine: GradCAM,
    input_tensor: torch.Tensor,
    mask_fractions: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    device: str = "cpu"
) -> Dict[str, any]:
    """
    Quantitative Faithfulness Sanity Check:
    Mask highest-attention pixels according to the CAM and observe the drop in prediction confidence.
    Compares against a random pixel masking baseline.
    """
    model.eval()
    heatmap, pred_class, base_conf = cam_engine.generate_cam(input_tensor)

    H, W = heatmap.shape
    total_pixels = H * W
    # Sort pixel indices by saliency (highest to lowest)
    flat_heatmap = heatmap.flatten()
    sorted_salient_indices = np.argsort(flat_heatmap)[::-1].copy()
    
    # Random indices for baseline control
    np.random.seed(42)
    random_indices = np.random.permutation(total_pixels).copy()

    salient_conf_drops = []
    random_conf_drops = []

    with torch.no_grad():
        for frac in mask_fractions:
            num_mask = int(total_pixels * frac)

            # 1. Mask highest salient pixels
            mask_idx = sorted_salient_indices[:num_mask]
            masked_tensor = input_tensor.clone()
            flat_img = masked_tensor[0].reshape(3, -1)
            # Mask to 0.0 (or mean value)
            flat_img[:, mask_idx] = 0.0
            masked_tensor = flat_img.reshape(1, 3, H, W).to(device)

            out_sal = model(masked_tensor)
            prob_sal = F.softmax(out_sal, dim=1)[0, pred_class].item()
            salient_conf_drops.append(base_conf - prob_sal)

            # 2. Mask random pixels (control)
            rand_idx = random_indices[:num_mask]
            rand_tensor = input_tensor.clone()
            flat_rand = rand_tensor[0].reshape(3, -1)
            flat_rand[:, rand_idx] = 0.0
            rand_tensor = flat_rand.reshape(1, 3, H, W).to(device)

            out_rand = model(rand_tensor)
            prob_rand = F.softmax(out_rand, dim=1)[0, pred_class].item()
            random_conf_drops.append(base_conf - prob_rand)

    return {
        "base_confidence": base_conf,
        "mask_fractions": mask_fractions,
        "salient_drops": salient_conf_drops,
        "random_drops": random_conf_drops,
        "is_faithful": bool(np.mean(salient_conf_drops) > np.mean(random_conf_drops))
    }
