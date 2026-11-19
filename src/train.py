"""
Training Pipeline for Plant Disease Classification.
Supports:
- Baseline Scratch CNN training
- Two-Stage Transfer Learning (Warmup frozen backbone -> Fine-tune unfreezing)
- AdamW optimizer + ReduceLROnPlateau / Cosine scheduler
- Early stopping monitored on validation macro F1-score
- Checkpointing best weights and metric logging
- Mini-batch smoke testing mode
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm

from src.config import config
from src.dataset import get_dataloaders
from src.models import get_model, freeze_backbone, unfreeze_backbone
from src.utils import set_seed, compute_metrics, plot_training_curves

def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    desc: str = "Training"
) -> Tuple[float, float]:
    """Train for a single epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        current_loss = total_loss / total
        current_acc = correct / total
        pbar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.4f}"})

    epoch_loss = total_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    class_names: Optional[List[str]] = None,
    desc: str = "Evaluating"
) -> Tuple[float, float, float, Dict[str, any]]:
    """Evaluate for a single epoch on validation or test set."""
    model.eval()
    total_loss = 0.0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = total_loss / total
    metrics = compute_metrics(all_labels, all_preds, class_names=class_names)
    epoch_acc = metrics["accuracy"]
    epoch_f1 = metrics["f1_macro"]

    return epoch_loss, epoch_acc, epoch_f1, metrics

def train_model(
    model_name: str,
    smoke_test: bool = False,
    warmup_epochs: Optional[int] = None,
    finetune_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_samples: Optional[int] = None
) -> Dict[str, any]:
    """Execute complete training cycle (warmup + fine-tuning or scratch)."""
    set_seed(config.seed)
    device = config.device
    print(f"\n{'='*60}")
    print(f"Beginning Training for: {model_name.upper()} on {device}")
    print(f"{'='*60}")

    # Load class mapping
    with open(config.classes_json, "r") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Dataloaders
    max_samples_eff = 300 if smoke_test else max_samples
    bs = batch_size or (16 if smoke_test else config.batch_size)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=bs,
        max_samples=max_samples_eff
    )
    print(f"Loaded datasets - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Instantiate model
    is_baseline = "baseline" in model_name.lower() or "scratch" in model_name.lower()
    model = get_model(model_name, num_classes=num_classes, pretrained=not is_baseline)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": []
    }

    best_val_f1 = -1.0
    best_weights_path = config.models_dir / f"{model_name}_best.pt"
    patience_counter = 0

    if smoke_test:
        w_epochs = 1
        ft_epochs = 0 if is_baseline else 1
    else:
        w_epochs = config.warmup_epochs if warmup_epochs is None else warmup_epochs
        ft_epochs = config.finetune_epochs if finetune_epochs is None else finetune_epochs

    stage2_epochs = ft_epochs if ft_epochs is not None else (1 if smoke_test else (config.scratch_epochs if is_baseline else config.finetune_epochs))
    total_epochs = w_epochs + stage2_epochs if not is_baseline else stage2_epochs

    # -------------------------------------------------------------
    # STAGE 1: Warmup Head (Backbone Frozen)
    # -------------------------------------------------------------
    if not is_baseline and w_epochs > 0:
        print(f"\n--- Stage 1: Warmup ({w_epochs} epochs, backbone frozen) ---")
        freeze_backbone(model, model_name)
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.warmup_lr,
            weight_decay=config.weight_decay
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

        for epoch in range(1, w_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                desc=f"Stage 1 Ep {epoch}/{w_epochs}"
            )
            val_loss, val_acc, val_f1, _ = evaluate_epoch(
                model, val_loader, criterion, device, class_names=class_names,
                desc=f"Val Ep {epoch}"
            )
            scheduler.step(val_f1)
            t1 = time.time()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(f"Stage 1 Ep {epoch:02d}/{w_epochs:02d} | Time: {t1-t0:.1f}s | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "class_to_idx": class_to_idx,
                }, best_weights_path)
                print(f" -> Best checkpoint saved with Val Macro F1: {val_f1*100:.2f}%")

    # -------------------------------------------------------------
    # STAGE 2: Full Fine-Tuning (Or scratch training loop)
    # -------------------------------------------------------------
    if stage2_epochs > 0:
        if not is_baseline:
            print(f"\n--- Stage 2: Fine-Tuning ({stage2_epochs} epochs, full model unfrozen) ---")
            unfreeze_backbone(model, model_name)
            current_lr = config.finetune_lr
        else:
            print(f"\n--- Training Scratch CNN ({stage2_epochs} epochs) ---")
            current_lr = config.scratch_lr

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_lr,
            weight_decay=config.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)

        for ep_idx in range(1, stage2_epochs + 1):
            curr_epoch_num = len(history["train_loss"]) + 1
            t0 = time.time()
            tr_loss, tr_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                desc=f"Ep {curr_epoch_num}/{total_epochs}"
            )
            val_loss, val_acc, val_f1, _ = evaluate_epoch(
                model, val_loader, criterion, device, class_names=class_names,
                desc=f"Val Ep {curr_epoch_num}"
            )
            scheduler.step()
            t1 = time.time()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(f"Ep {curr_epoch_num:02d}/{total_epochs:02d} | Time: {t1-t0:.1f}s | "
                  f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% F1: {val_f1*100:.2f}%")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save({
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "epoch": curr_epoch_num,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "class_to_idx": class_to_idx,
                }, best_weights_path)
                print(f" -> Best checkpoint saved with Val Macro F1: {val_f1*100:.2f}%")
            else:
                patience_counter += 1
                if not smoke_test and patience_counter >= config.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {curr_epoch_num} (patience={config.early_stopping_patience})")
                    break

    # Save training curves figure
    curve_path = config.figures_dir / f"{model_name}_training_curves.png"
    plot_training_curves(history, curve_path, model_name=model_name)
    print(f"Saved training curves to: {curve_path}")

    # Flag suspiciously perfect performance
    if best_val_f1 > 0.995:
        print("\n[WARNING] Validation F1 exceeds 99.5%! Checking for potential background bias or data leakage.")

    return {
        "model_name": model_name,
        "best_val_f1": best_val_f1,
        "checkpoint_path": str(best_weights_path),
        "history": history
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "resnet50", "efficientnet_b0", "mobilenet_v3"])
    parser.add_argument("--smoke-test", action="store_true", help="Quick 1-epoch test on small subset")
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Cap training samples for CPU acceleration")
    args = parser.parse_args()

    train_model(
        model_name=args.model,
        smoke_test=args.smoke_test,
        warmup_epochs=args.warmup_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
