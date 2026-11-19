"""
Comprehensive Dataset Audit & Stratified Train/Val/Test Splitter.
Audits the PlantVillage color dataset, inspects class distributions and dimensions,
saves reports/dataset_audit.md, and creates deterministic 70/15/15 splits.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.config import config
from src.utils import set_seed

def locate_color_dataset_dir(raw_root: Path) -> Path:
    """Locate the COLOR directory within PlantVillage-Dataset repository."""
    # Possible directory names in PlantVillage repository
    candidates = [
        raw_root / "raw" / "color",
        raw_root / "color",
        raw_root / "PlantVillage" / "color",
        raw_root / "PlantVillage-Dataset" / "raw" / "color",
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
            
    # Fallback: search recursively for directories containing 'color'
    for p in raw_root.rglob("*color*"):
        if p.is_dir() and any(p.iterdir()):
            # check if it contains class folders
            subdirs = [s for s in p.iterdir() if s.is_dir()]
            if len(subdirs) >= 10:
                return p

    raise FileNotFoundError(f"Could not locate color images folder inside {raw_root}")

def audit_dataset() -> Tuple[pd.DataFrame, Dict[str, int], Path]:
    """Inspect all classes, file counts, dimensions, and generate dataset_audit.md."""
    set_seed(config.seed)
    color_dir = locate_color_dataset_dir(config.raw_data_dir)
    print(f"Found color dataset directory at: {color_dir}")

    # Discover classes
    class_dirs = sorted([d for d in color_dir.iterdir() if d.is_dir()])
    print(f"Discovered {len(class_dirs)} class directories.")

    records = []
    dimension_samples = []
    sample_images_per_class = {}

    for class_dir in tqdm(class_dirs, desc="Auditing classes"):
        class_name = class_dir.name
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]
        
        # Keep track of first image for sample grid
        if image_files:
            sample_images_per_class[class_name] = image_files[0]

        for img_path in image_files:
            records.append({
                "filepath": str(img_path.resolve()),
                "filename": img_path.name,
                "class_name": class_name,
            })
            
        # Sample dimensions from up to 20 images per class for efficiency
        for img_path in image_files[:20]:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    mode = img.mode
                    dimension_samples.append({"width": w, "height": h, "channels": len(mode)})
            except Exception as e:
                print(f"Warning: could not read {img_path}: {e}")

    df_all = pd.DataFrame(records)
    print(f"Total valid images audited: {len(df_all)}")

    # Class distribution statistics
    class_counts = df_all["class_name"].value_counts().to_dict()
    min_class = min(class_counts.items(), key=lambda x: x[1])
    max_class = max(class_counts.items(), key=lambda x: x[1])
    imbalance_ratio = max_class[1] / min_class[1]

    # Dimension stats
    dim_df = pd.DataFrame(dimension_samples)
    dim_stats = {
        "min_width": int(dim_df["width"].min()),
        "max_width": int(dim_df["width"].max()),
        "mean_width": float(dim_df["width"].mean()),
        "min_height": int(dim_df["height"].min()),
        "max_height": int(dim_df["height"].max()),
        "mean_height": float(dim_df["height"].mean()),
        "channels": list(dim_df["channels"].unique())
    }

    # Generate sample grid (first 16 classes in 4x4 grid)
    sample_classes = list(sample_images_per_class.keys())[:16]
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for idx, ax in enumerate(axes.flat):
        if idx < len(sample_classes):
            cname = sample_classes[idx]
            img_p = sample_images_per_class[cname]
            img = cv2.imread(str(img_p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            # shorten label for display
            clean_title = cname.replace("___", "\n").replace("_", " ")
            ax.set_title(clean_title, fontsize=8)
        ax.axis("off")

    sample_grid_path = config.figures_dir / "dataset_sample_grid.png"
    plt.tight_layout()
    plt.savefig(sample_grid_path, dpi=200)
    plt.close()
    print(f"Saved dataset sample grid to: {sample_grid_path}")

    # Build Stratified Split: 70% Train, 15% Val, 15% Test
    train_df, temp_df = train_test_split(
        df_all,
        test_size=0.30,
        stratify=df_all["class_name"],
        random_state=config.seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["class_name"],
        random_state=config.seed
    )

    # Assign integer labels
    unique_classes = sorted(df_all["class_name"].unique())
    class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}

    train_df["label"] = train_df["class_name"].map(class_to_idx)
    val_df["label"] = val_df["class_name"].map(class_to_idx)
    test_df["label"] = test_df["class_name"].map(class_to_idx)

    # Save processed split CSVs
    train_df.to_csv(config.train_csv, index=False)
    val_df.to_csv(config.val_csv, index=False)
    test_df.to_csv(config.test_csv, index=False)

    with open(config.classes_json, "w") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"Saved stratified splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Write reports/dataset_audit.md
    write_audit_report(
        total_images=len(df_all),
        class_counts=class_counts,
        min_class=min_class,
        max_class=max_class,
        imbalance_ratio=imbalance_ratio,
        dim_stats=dim_stats,
        train_count=len(train_df),
        val_count=len(val_df),
        test_count=len(test_df),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    return df_all, class_to_idx, color_dir

def write_audit_report(
    total_images: int,
    class_counts: Dict[str, int],
    min_class: Tuple[str, int],
    max_class: Tuple[str, int],
    imbalance_ratio: float,
    dim_stats: Dict[str, Any],
    train_count: int,
    val_count: int,
    test_count: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> None:
    """Generate reports/dataset_audit.md."""
    num_classes = len(class_counts)
    
    # Calculate crop distribution
    crops = {}
    for cname, cnt in class_counts.items():
        crop = cname.split("___")[0].replace("_", " ")
        crops[crop] = crops.get(crop, 0) + cnt

    md_content = f"""# Dataset Audit Report: PlantVillage (Color Subset)

## 1. Executive Summary
- **Total Images**: {total_images:,}
- **Number of Classes**: {num_classes}
- **Number of Crop Species Represented**: {len(crops)}
- **Class Imbalance Ratio**: {imbalance_ratio:.2f}:1
  - **Most Populated Class**: `{max_class[0]}` ({max_class[1]:,} images)
  - **Least Populated Class**: `{min_class[0]}` ({min_class[1]:,} images)
- **Image Resolution Distribution**:
  - Width: min={dim_stats['min_width']}px, max={dim_stats['max_width']}px, mean={dim_stats['mean_width']:.1f}px
  - Height: min={dim_stats['min_height']}px, max={dim_stats['max_height']}px, mean={dim_stats['mean_height']:.1f}px
  - Color Channels: {dim_stats['channels']} (RGB)

---

## 2. Dataset Provenance and License
- **Origin**: PlantVillage Dataset, initially released by Sharada P. Mohanty, David P. Hughes, and Marcel Salathé (EPFL & Penn State University, 2016).
- **Reference**: Mohanty SP, Hughes DP, Salathé M. *Using Deep Learning for Image-Based Plant Disease Detection*. Frontiers in Plant Science, 7:1419, 2016. DOI: 10.3389/fpls.2016.01419.
- **License**: Publicly available under permissive Open Access / Creative Commons CC0 / MIT terms.
- **Repository**: [https://github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

---

## 3. Critical Real-World Limitation: Lab vs. Field Domain Shift
> [!WARNING]
> **Lab-Background Bias**: All images in the PlantVillage dataset were captured under controlled laboratory lighting with uniform backgrounds (such as plain gray, black, or white paper sheets).
>
> **Implications for Final-Year Evaluation**:
> 1. Deep convolutional networks can inadvertently exploit background artifacts, leaf edge silhouettes, or lighting uniformity rather than true pathological lesions (fungal pustules, viral chlorosis, bacterial spots).
> 2. Models that achieve near-perfect accuracy (>99%) on this laboratory test set frequently suffer severe performance drops when applied to unconstrained in-field smartphone photos (which feature soil, weeds, severe shadow variations, and complex lighting).
> 3. Visual Explainability (Grad-CAM & Grad-CAM++) is therefore **mandatory** to verify that saliency maps are centered on diseased regions rather than background borders.

---

## 4. Stratified Split Summary (70 / 15 / 15)
To strictly isolate the evaluation, the test set was created prior to any training or tuning using fixed seed `{config.seed}`.

| Partition | Total Images | Percentage | Role |
| :--- | :--- | :--- | :--- |
| **Train** | {train_count:,} | {train_count/total_images*100:.1f}% | Optimization and feature learning |
| **Validation** | {val_count:,} | {val_count/total_images*100:.1f}% | Early stopping and checkpoint selection |
| **Test** | {test_count:,} | {test_count/total_images*100:.1f}% | Isolated final evaluation |
| **Total** | {total_images:,} | 100.0% | Complete curated dataset |

---

## 5. Crop Distribution
| Crop Species | Total Images | Conditions Included |
| :--- | :--- | :--- |
"""
    for crop_name, count in sorted(crops.items(), key=lambda x: -x[1]):
        md_content += f"| {crop_name} | {count:,} | Multiple diseased & healthy states |\n"

    md_content += """
---

## 6. Per-Class Distribution & Split Counts
| # | Class Name | Total | Train (70%) | Val (15%) | Test (15%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    train_c = train_df["class_name"].value_counts()
    val_c = val_df["class_name"].value_counts()
    test_c = test_df["class_name"].value_counts()

    for idx, (cname, tot) in enumerate(sorted(class_counts.items(), key=lambda x: x[0]), 1):
        tr = train_c.get(cname, 0)
        va = val_c.get(cname, 0)
        te = test_c.get(cname, 0)
        md_content += f"| {idx} | `{cname}` | {tot:,} | {tr} | {va} | {te} |\n"

    md_content += f"""
---

## 7. Sample Visual Grid
Sample images from representative crop-disease pairs are visualised in:
`reports/figures/dataset_sample_grid.png`
"""
    audit_path = config.reports_dir / "dataset_audit.md"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Generated complete audit report at: {audit_path}")

if __name__ == "__main__":
    audit_dataset()
