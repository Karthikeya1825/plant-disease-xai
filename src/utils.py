"""
Utility functions for reproducibility, metrics, plotting, and checkpointing.
"""
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

def set_seed(seed: int = 42) -> None:
    """Set random seeds across python, numpy, and torch for strict reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a PyTorch module."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_mb(filepath: Path) -> float:
    """Return model file size in Megabytes."""
    if not os.path.exists(filepath):
        return 0.0
    return os.path.getsize(filepath) / (1024 * 1024)

def benchmark_latency(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = "cpu",
    runs: int = 50,
    warmup: int = 10
) -> float:
    """Benchmark inference latency (ms per image) on the specified device."""
    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)
    
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
            
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0) # convert to ms
            
    return float(np.mean(times))

def compute_metrics(y_true: List[int], y_pred: List[int], class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Calculate multi-class classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    prec_per_class, rec_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred)
    
    per_class_dict = {}
    if class_names:
        for idx, name in enumerate(class_names):
            if idx < len(f1_per_class):
                per_class_dict[name] = {
                    "precision": float(prec_per_class[idx]),
                    "recall": float(rec_per_class[idx]),
                    "f1": float(f1_per_class[idx]),
                    "support": int(support[idx])
                }

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (16, 14)
) -> None:
    """Plot and save a styled confusion matrix heatmap."""
    plt.figure(figsize=figsize)
    # Normalize by row (true class)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Proportion"}
    )
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_training_curves(history: Dict[str, List[float]], save_path: Path, model_name: str = "Model") -> None:
    """Plot train/val loss and train/val accuracy across epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1B5E20", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#D32F2F", linewidth=2, linestyle="--")
    ax1.set_title(f"{model_name} - Loss Trajectory", fontsize=13)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")
    
    # Accuracy / F1 plot
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="#2E7D32", linewidth=2)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="#388E3C", linewidth=2, linestyle="--")
    if "val_f1" in history:
        ax2.plot(epochs, history["val_f1"], label="Val Macro F1", color="#E65100", linewidth=2, linestyle="-.")
    ax2.set_title(f"{model_name} - Accuracy & F1 Trajectory", fontsize=13)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
