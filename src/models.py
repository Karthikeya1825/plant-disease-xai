"""
Model Architectures for Plant Disease Classification:
- BaselineScratchCNN: 4-stage Convolutional Neural Network from scratch
- ResNet-50: Transfer learning with Residual connections
- EfficientNet-B0: Transfer learning with compound scaling
- MobileNetV3-Large: Transfer learning optimized for edge/mobile inference
Includes backbone freezing/unfreezing helpers and Grad-CAM target layer hooks.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models

class BaselineScratchCNN(nn.Module):
    """
    Lightweight 4-stage CNN trained from scratch as an empirical baseline.
    Input: (B, 3, 224, 224) -> Output: (B, num_classes)
    """
    def __init__(self, num_classes: int = 38, dropout_rate: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Stage 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.gap(feat)
        flat = torch.flatten(pooled, 1)
        logits = self.classifier(flat)
        return logits

def get_model(
    model_name: str,
    num_classes: int = 38,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function to instantiate models with appropriate classification heads.
    """
    name_lower = model_name.lower().replace("-", "").replace("_", "")

    if name_lower in ["baseline", "scratch", "baselinescratchcnn"]:
        model = BaselineScratchCNN(num_classes=num_classes)

    elif "resnet50" in name_lower:
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    elif "efficientnetb0" in name_lower or "efficientnet" in name_lower:
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    elif "mobilenetv3" in name_lower or "mobilenet" in name_lower:
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return model

def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze all convolutional feature extraction layers; leave classification head trainable."""
    name_lower = model_name.lower().replace("-", "").replace("_", "")
    
    if "baseline" in name_lower:
        # Baseline has no pre-trained weights; train end-to-end
        return

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the classification head
    if "resnet" in name_lower:
        for param in model.fc.parameters():
            param.requires_grad = True
    elif "efficientnet" in name_lower:
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif "mobilenet" in name_lower:
        for param in model.classifier.parameters():
            param.requires_grad = True

def unfreeze_backbone(model: nn.Module, model_name: str, top_only: bool = True) -> None:
    """
    Unfreeze backbone for stage 2 fine-tuning.
    If top_only=True, unfreezes only the topmost convolutional block + classifier,
    which is standard practice in transfer learning to retain low-level edge features
    and dramatically accelerate training on CPU/GPU.
    """
    name_lower = model_name.lower().replace("-", "").replace("_", "")
    
    if not top_only or "baseline" in name_lower:
        for param in model.parameters():
            param.requires_grad = True
        return

    # Keep early layers frozen, unfreeze top block
    if "resnet" in name_lower:
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    elif "efficientnet" in name_lower:
        for block in model.features[-3:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif "mobilenet" in name_lower:
        for block in model.features[-3:]:
            for param in block.parameters():
                param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

def get_target_layer_for_cam(model: nn.Module, model_name: str) -> nn.Module:
    """Return the designated target convolutional layer for Grad-CAM / Grad-CAM++."""
    name_lower = model_name.lower().replace("-", "").replace("_", "")

    if "baseline" in name_lower:
        return model.features[-2]  # ReLU of stage 4
    elif "resnet" in name_lower:
        return model.layer4[-1].conv3  # or model.layer4[-1]
    elif "efficientnet" in name_lower:
        return model.features[-1]  # Top convolutional block
    elif "mobilenet" in name_lower:
        return model.features[-1]  # Final pointwise block
    else:
        raise ValueError(f"Unknown target layer for {model_name}")
