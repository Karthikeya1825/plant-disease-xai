"""
Complete Pipeline Runner:
1. Trains Transfer Learning models (MobileNetV3, EfficientNet-B0, ResNet-50).
2. Evaluates all 4 models on the isolated test set.
3. Generates the cross-model comparative metrics report (reports/metrics.json).
4. Generates the qualitative XAI gallery and quantitative deletion test figures.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import config
from src.dataset import preprocess_single_image
from src.evaluate import evaluate_checkpoint, run_error_analysis, generate_comparison_report
from src.gradcam import GradCAM, overlay_cam_on_image, run_deletion_test
from src.models import get_model, get_target_layer_for_cam
from src.train import train_model

def run_training_suite(models_to_train=None, warmup_epochs=1, finetune_epochs=2, max_samples=1500, batch_size=32):
    """Train selected models."""
    if models_to_train is None:
        models_to_train = ["mobilenet_v3", "efficientnet_b0", "resnet50"]

    results = {}
    for mname in models_to_train:
        print(f"\n{'='*70}\nSTARTING TRAINING PIPELINE FOR: {mname.upper()}\n{'='*70}")
        res = train_model(
            model_name=mname,
            warmup_epochs=warmup_epochs,
            finetune_epochs=finetune_epochs,
            batch_size=batch_size,
            max_samples=max_samples
        )
        results[mname] = res
    return results

def run_full_evaluation(max_test_samples=1500):
    """Evaluate baseline and transfer models on isolated test set."""
    models = ["baseline", "mobilenet_v3", "efficientnet_b0", "resnet50"]
    all_metrics = {}

    with open(config.classes_json, "r") as f:
        c2i = json.load(f)
    class_names = [k for k, _ in sorted(c2i.items(), key=lambda x: x[1])]

    for mname in models:
        ckpt_path = config.models_dir / f"{mname}_best.pt"
        if not ckpt_path.exists():
            print(f"[SKIP] Checkpoint {ckpt_path} not found.")
            continue

        metrics, misclassified = evaluate_checkpoint(
            model_name=mname,
            checkpoint_path=ckpt_path,
            device=config.device,
            max_test_samples=max_test_samples
        )
        all_metrics[mname] = metrics
        run_error_analysis(mname, misclassified, class_names, total_test_samples=metrics["total_test_samples"])

    # Generate comparative table & update reports/metrics.json
    generate_comparison_report(all_metrics)
    return all_metrics

def generate_xai_artifacts(best_model_name="mobilenet_v3"):
    """
    Generate qualitative XAI gallery comparing Grad-CAM and Grad-CAM++
    as well as quantitative deletion test benchmark figure.
    """
    print("\n--- Generating Explainable AI (XAI) Artifacts & Qualitative Gallery ---")
    ckpt_path = config.models_dir / f"{best_model_name}_best.pt"
    if not ckpt_path.exists():
        print(f"Model checkpoint {ckpt_path} not found.")
        return

    with open(config.classes_json, "r") as f:
        c2i = json.load(f)
    idx_to_class = {v: k for k, v in c2i.items()}

    model = get_model(best_model_name, num_classes=len(c2i), pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_layer = get_target_layer_for_cam(model, best_model_name)
    cam_engine = GradCAM(model, target_layer)

    # Pick samples from app/assets
    asset_files = list(config.app_dir.glob("assets/*.JPG")) + list(config.app_dir.glob("assets/*.jpg"))
    if not asset_files:
        print("No asset images found in app/assets")
        return

    # 1. Qualitative Gallery: Original | Grad-CAM | Grad-CAM++
    gallery_samples = asset_files[:4]
    fig, axes = plt.subplots(len(gallery_samples), 4, figsize=(16, 4 * len(gallery_samples)))

    deletion_results = []

    for row_idx, img_path in enumerate(gallery_samples):
        tensor, rgb = preprocess_single_image(img_path)

        # Grad-CAM
        cam_grad, pred_c, conf = cam_engine.generate_cam(tensor, method="gradcam")
        overlay_grad = overlay_cam_on_image(rgb, cam_grad, alpha=0.5)

        # Grad-CAM++
        cam_pp, _, _ = cam_engine.generate_cam(tensor, method="gradcam++")
        overlay_pp = overlay_cam_on_image(rgb, cam_pp, alpha=0.5)

        pred_name = idx_to_class[pred_c].replace("___", " ").replace("_", " ")

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].set_title(f"Input: {img_path.name}\nPred: {pred_name} ({conf*100:.1f}%)", fontsize=9)
        axes[row_idx, 0].axis("off")

        axes[row_idx, 1].imshow(cam_grad, cmap="jet")
        axes[row_idx, 1].set_title("Grad-CAM Heatmap", fontsize=9)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(overlay_grad)
        axes[row_idx, 2].set_title("Grad-CAM Overlay", fontsize=9)
        axes[row_idx, 2].axis("off")

        axes[row_idx, 3].imshow(overlay_pp)
        axes[row_idx, 3].set_title("Grad-CAM++ Overlay\n(Higher lesion localization)", fontsize=9)
        axes[row_idx, 3].axis("off")

        # Run deletion test for the first sample
        if row_idx == 0:
            del_res = run_deletion_test(model, cam_engine, tensor, mask_fractions=[0.1, 0.2, 0.3, 0.4, 0.5])
            del_res["image"] = img_path.name
            deletion_results.append(del_res)

    gallery_path = config.figures_dir / "xai_qualitative_gallery.png"
    plt.tight_layout()
    plt.savefig(gallery_path, dpi=200)
    plt.close()
    print(f"Saved qualitative gallery to: {gallery_path}")

    # 2. Deletion Test Curve Figure
    if deletion_results:
        del_data = deletion_results[0]
        fractions = [f * 100 for f in del_data["mask_fractions"]]
        salient_drops = [d * 100 for d in del_data["salient_drops"]]
        random_drops = [d * 100 for d in del_data["random_drops"]]

        plt.figure(figsize=(8, 5))
        plt.plot(fractions, salient_drops, marker='o', color='#2E7D32', linewidth=2, label="Grad-CAM Salient Masking")
        plt.plot(fractions, random_drops, marker='s', color='#757575', linestyle='--', linewidth=2, label="Random Pixel Masking (Control)")
        plt.title(f"Quantitative Faithfulness Sanity Check: Deletion Test\n({best_model_name.upper()} on {del_data['image']})", fontsize=11, fontweight="bold")
        plt.xlabel("Percentage of Most Salient Pixels Masked (%)", fontsize=10)
        plt.ylabel("Prediction Confidence Drop (Percentage Points)", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True)
        plt.tight_layout()

        deletion_fig_path = config.figures_dir / "deletion_test_benchmark.png"
        plt.savefig(deletion_fig_path, dpi=200)
        plt.close()
        print(f"Saved deletion test curve to: {deletion_fig_path}")

    cam_engine.remove_hooks()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true", help="Skip training and run eval/XAI only")
    parser.add_argument("--models", nargs="+", default=["mobilenet_v3", "efficientnet_b0", "resnet50"])
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=1500)
    parser.add_argument("--max-test-samples", type=int, default=1500)
    args = parser.parse_args()

    if not args.skip_training:
        run_training_suite(
            models_to_train=args.models,
            warmup_epochs=args.warmup_epochs,
            finetune_epochs=args.finetune_epochs,
            max_samples=args.max_samples
        )

    run_full_evaluation(max_test_samples=args.max_test_samples)
    generate_xai_artifacts()
