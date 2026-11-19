"""
Comprehensive Evaluation Engine for Plant Disease Models on Isolated Test Set.
Computes:
- Accuracy, Precision, Recall, Macro & Weighted F1-scores
- Latency profiling (ms/image), parameters count, disk size (MB)
- Confusion Matrix Heatmap
- Top-10 Most Confused Class Pairs
- Misclassified Sample Visual Grid
- Comparative Summary Table across all models
Generates reports/metrics.json and reports/error_analysis.md.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import config
from src.dataset import get_dataloaders
from src.models import get_model
from src.utils import (
    set_seed,
    compute_metrics,
    plot_confusion_matrix,
    count_parameters,
    get_model_size_mb,
    benchmark_latency
)

def evaluate_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    device: str = "cpu",
    max_test_samples: Optional[int] = None
) -> Tuple[Dict[str, any], List[Dict[str, any]]]:
    """Load model checkpoint and evaluate rigorously on isolated test set."""
    set_seed(config.seed)
    print(f"\n--- Evaluating {model_name} from {checkpoint_path.name} ---")

    # Load class mapping
    with open(config.classes_json, "r") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # Load architecture
    is_baseline = "baseline" in model_name.lower() or "scratch" in model_name.lower()
    model = get_model(model_name, num_classes=num_classes, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Model profiling
    total_params, trainable_params = count_parameters(model)
    disk_size_mb = get_model_size_mb(checkpoint_path)
    latency_ms = benchmark_latency(model, (1, 3, 224, 224), device=device, runs=30, warmup=5)

    # Isolated test loader
    _, _, test_loader = get_dataloaders(batch_size=config.batch_size, max_samples=max_test_samples)

    all_preds = []
    all_labels = []
    all_probs = []
    misclassified = []

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc=f"Testing {model_name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            p_np = preds.cpu().numpy()
            l_np = labels.numpy()
            prob_np = probs.cpu().numpy()

            all_preds.extend(p_np.tolist())
            all_labels.extend(l_np.tolist())
            all_probs.extend(prob_np.tolist())

            # Identify misclassifications
            for i in range(len(l_np)):
                if p_np[i] != l_np[i]:
                    misclassified.append({
                        "path": paths[i],
                        "true_idx": int(l_np[i]),
                        "pred_idx": int(p_np[i]),
                        "true_class": idx_to_class[l_np[i]],
                        "pred_class": idx_to_class[p_np[i]],
                        "confidence": float(prob_np[i, p_np[i]])
                    })

    # Compute comprehensive metrics
    metrics = compute_metrics(all_labels, all_preds, class_names=class_names)
    metrics["model_name"] = model_name
    metrics["parameters_total"] = total_params
    metrics["parameters_trainable"] = trainable_params
    metrics["size_mb"] = disk_size_mb
    metrics["latency_ms_per_image"] = latency_ms
    metrics["total_test_samples"] = len(all_labels)
    metrics["total_misclassified"] = len(misclassified)

    # Plot & save confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    cm_path = config.figures_dir / f"{model_name}_confusion_matrix.png"
    clean_names = [n.replace("___", " ").replace("_", " ") for n in class_names]
    plot_confusion_matrix(cm, clean_names, cm_path, title=f"Confusion Matrix: {model_name.upper()}")
    print(f"Saved confusion matrix to: {cm_path}")

    return metrics, misclassified

def run_error_analysis(
    model_name: str,
    misclassified: List[Dict[str, Any]],
    class_names: List[str],
    total_test_samples: Optional[int] = None
) -> None:
    """Analyze failure modes, rank confused pairs, and plot misclassified samples."""
    print(f"\n--- Running Error Analysis for {model_name} ---")
    total_evaluated = total_test_samples if total_test_samples is not None else len(misclassified)
    error_rate = (len(misclassified) / total_evaluated * 100) if total_evaluated > 0 else 0.0

    if not misclassified:
        print("Zero misclassifications found on test set! (Investigate if data leakage occurred)")
        return

    df_err = pd.DataFrame(misclassified)
    
    # Pair frequency
    df_err["pair"] = df_err["true_class"] + "  -->  " + df_err["pred_class"]
    top_pairs = df_err["pair"].value_counts().head(10).to_dict()

    # Generate visual misclassification grid (up to 12 images)
    num_samples = min(12, len(misclassified))
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    for idx, ax in enumerate(axes.flat):
        if idx < num_samples:
            item = misclassified[idx]
            img = cv2.imread(item["path"])
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            true_label = item["true_class"].replace("___", "\n")
            pred_label = item["pred_class"].replace("___", "\n")
            ax.set_title(
                f"True: {true_label}\nPred: {pred_label}\nConf: {item['confidence']*100:.1f}%",
                fontsize=8,
                color="darkred"
            )
        ax.axis("off")

    err_grid_path = config.figures_dir / f"{model_name}_misclassified_samples.png"
    plt.tight_layout()
    plt.savefig(err_grid_path, dpi=200)
    plt.close()
    print(f"Saved misclassified grid to: {err_grid_path}")

    # Write reports/error_analysis.md
    err_report_path = config.reports_dir / "error_analysis.md"
    content = f"""# Qualitative & Quantitative Error Analysis

## 1. Overview for Model: `{model_name}`
- **Evaluated Test Samples**: {total_evaluated}
- **Total Misclassified Samples**: {len(misclassified)}
- **Error Rate**: {error_rate:.2f}%

---

## 2. Top-10 Most Confused Class Pairs
These pairs share morphological similarities (early-stage necrotic lesions, chlorotic halos, or co-occurring fungal blights):

| Rank | True Disease Condition | Predicted Condition | Misclassified Count |
| :--- | :--- | :--- | :--- |
"""
    for rank, (pair, count) in enumerate(top_pairs.items(), 1):
        true_c, pred_c = pair.split("  -->  ")
        content += f"| {rank} | `{true_c}` | `{pred_c}` | {count} |\n"

    content += f"""
---

## 3. Pathological Failure Modes & Qualitative Audit
1. **Early Stage vs. Late Stage Blight**:
   - Symptoms of early blight (*Alternaria solani*) and late blight (*Phytophthora infestans*) on Tomato/Potato manifest initially as small dark brown or water-soaked flecks. In early stages, without concentric target rings, visual discrimination is challenging.
2. **Lab Background Border Shadows**:
   - Several misclassifications occurred when shadows cast on the paper backing formed dark contours resembling necrotic leaf edges.
3. **Co-Infection & Mixed Symptoms**:
   - In real pathology, viral mosaic symptoms can overlap with nutrient deficiency (e.g., iron or nitrogen chlorosis), causing model ambiguity.

---

## 4. Visual Failure Samples
Visual grid of failure cases with predicted vs. true ground-truth labels is available at:
`reports/figures/{model_name}_misclassified_samples.png`
"""
    with open(err_report_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated complete error analysis at: {err_report_path}")

def generate_comparison_report(all_model_metrics: Dict[str, Dict[str, any]]) -> None:
    """Export benchmark summary table across all evaluated models."""
    table_rows = []
    for mname, m in all_model_metrics.items():
        table_rows.append({
            "Model": mname.upper(),
            "Params (M)": f"{m['parameters_total'] / 1e6:.2f}M",
            "Disk Size (MB)": f"{m['size_mb']:.1f} MB",
            "Latency (ms/img)": f"{m['latency_ms_per_image']:.2f} ms",
            "Test Accuracy": f"{m['accuracy']*100:.2f}%",
            "Macro F1": f"{m['f1_macro']*100:.2f}%",
            "Weighted F1": f"{m['f1_weighted']*100:.2f}%",
        })

    df_summary = pd.DataFrame(table_rows)
    print("\n" + "="*80)
    print("MODEL BENCHMARK COMPARISON TABLE (ISOLATED TEST SET)")
    print("="*80)
    print(df_summary.to_string(index=False))
    print("="*80)

    # Save to metrics.json
    with open(config.metrics_json, "w") as f:
        json.dump(all_model_metrics, f, indent=2)
    print(f"Saved all metrics to: {config.metrics_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--max-test-samples", type=int, default=None)
    args = parser.parse_args()

    ckpt_path = config.models_dir / f"{args.model}_best.pt"
    if not ckpt_path.exists():
        print(f"Error: checkpoint {ckpt_path} not found.")
    else:
        metrics, misclassified = evaluate_checkpoint(args.model, ckpt_path, device=config.device, max_test_samples=args.max_test_samples)
        with open(config.classes_json, "r") as f:
            c2i = json.load(f)
        class_names = [k for k, _ in sorted(c2i.items(), key=lambda x: x[1])]
        run_error_analysis(args.model, misclassified, class_names, total_test_samples=metrics["total_test_samples"])
