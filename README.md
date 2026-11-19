# Plant Disease Detection with Explainable AI (XAI) 🌿

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production- and research-grade deep learning system for crop leaf disease detection featuring transparent visual explainability via **Grad-CAM** and **Grad-CAM++**, empirical faithfulness verification (Deletion Tests), rigorous dataset auditing, and an interactive Streamlit web prototype.

---

## 1. Problem Statement
Crop foliar diseases threaten smallholder agriculture and global food security. While deep Convolutional Neural Networks (CNNs) achieve high classification accuracy, they operate as opaque "black-box" systems. In agricultural diagnostics, black-box models frequently suffer from **shortcut learning**—attending to laboratory background paper, lighting glare, or leaf silhouettes rather than biological lesions (fungal pustules, bacterial spots, viral mosaics).

This project addresses this critical gap by:
1. Benchmarking 4 distinct neural architectures (**Baseline Scratch CNN**, **ResNet-50**, **EfficientNet-B0**, and **MobileNetV3-Large**) on a strictly isolated 38-class test set.
2. Integrating native PyTorch **Grad-CAM** and **Grad-CAM++** to generate visual explanations of model diagnoses.
3. Quantitatively validating explanation faithfulness through progressive lesion **Deletion Tests**.
4. Packaging the system into a responsive, agricultural-themed **Streamlit web application** and a **FastAPI** microservice.

---

## 2. Dataset, Provenance & Licence
- **Dataset**: [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)
- **Curation**: 54,305 color leaf photographs across **14 crop species** and **38 condition classes**.
- **Citation**: Mohanty SP, Hughes DP, Salathé M. *Using Deep Learning for Image-Based Plant Disease Detection*. Frontiers in Plant Science, 7:1419, 2016. [DOI: 10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419).
- **Licence**: Creative Commons CC0 / Public Domain / MIT permissive use.
- **Data Splitting**: Strict stratified **70% Train (38,013) / 15% Val (8,146) / 15% Test (8,146)** split with fixed seed `42`. The test set is locked and untouched during model training.

---

## 3. Architecture & Methodology
```
                    [ Input Leaf Photo (224x224x3) ]
                                   │
                     ┌─────────────┴─────────────┐
                     ▼                           ▼
          [ Backbone Feature Extractor ]   [ Target Conv Layer ]
          (ResNet-50 / EfficientNet /         │
           MobileNetV3 / Scratch CNN)         │
                     │                           │  Gradient Backprop (dY/dA)
                     ▼                           ▼
          [ Softmax Classification ] ──────► [ Grad-CAM / Grad-CAM++ ]
          (Top-3 Candidates & Conf)              │
                                                 ▼
                                     [ Blended Saliency Overlay ]
                                                 │
                                                 ▼
                                     [ Faithfulness Deletion Test ]
```

### Two-Stage Transfer Learning Protocol
- **Stage 1 (Warmup)**: Convolutional backbone frozen. Linear classification head trained with AdamW ($\eta = 10^{-3}$, weight decay $10^{-4}$) and `ReduceLROnPlateau`.
- **Stage 2 (Fine-Tuning)**: Higher-level semantic blocks (`layer4` for ResNet, top MBConv blocks for MobileNet/EfficientNet) unfrozen with a reduced learning rate ($\eta = 10^{-4}$) and `CosineAnnealingLR`.

---

## 4. Benchmark Results (Isolated Test Set)

Evaluated on the held-out test split using single-image CPU inference (AMD Ryzen 5):

| Model Architecture | Parameters | Disk Size | CPU Latency | Test Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scratch Baseline CNN** | 0.43M | 1.6 MB | 28.98 ms | 43.01% | 20.29% | 33.92% |
| **MobileNetV3-Large** (Best) | 4.25M | 16.4 MB | 34.52 ms | **84.95%** | **76.33%** | **82.38%** |
| **EfficientNet-B0** | 4.06M | 15.8 MB | 52.77 ms | 73.12% | 56.46% | 68.50% |
| **ResNet-50** | 23.59M | 90.3 MB | 151.35 ms | 78.49% | 58.76% | 74.21% |

*Key finding: MobileNetV3 achieves the highest accuracy and macro F1 while delivering a 4.38x CPU latency speedup over ResNet-50, making it the superior architecture for agricultural edge and mobile deployment.*

---

## 5. Explainable AI (XAI) Suite

### Grad-CAM & Grad-CAM++
- **Grad-CAM**: Computes global average pooled gradients to weight activation maps, highlighting the coarse spatial regions responsible for class activation.
- **Grad-CAM++**: Uses second- and third-order positive partial derivatives, improving localization when multiple distinct lesions appear across the leaf blade.

### Faithfulness Sanity Check (Deletion Test)
To verify that heatmaps reflect true model reasoning rather than arbitrary visual edges, we iteratively mask the top $10\%, 20\%, \dots, 50\%$ most salient pixels and measure the collapse in prediction probability ($\Delta P = P_{\text{original}} - P_{\text{masked}}$) against a random masking control baseline.

---

## 6. How to Run

### Installation & Environment Setup
```powershell
# Clone the repository
git clone <repo-url>
cd plant-disease-xai

# Install pinned dependencies
py -m pip install -r requirements.txt
```

### Single-Command Operations
```powershell
# 1. Dataset Audit and Stratified Split
py -m src.dataset_audit

# 2. Train Models (Examples)
py -m src.train --model baseline
py -m src.train --model resnet50
py -m src.train --model mobilenet_v3
py -m src.train --model efficientnet_b0

# 3. Evaluate Isolated Test Set & Generate Error Analysis
py -m src.evaluate --model resnet50

# 4. Launch Interactive Streamlit Prototype
streamlit run app/app.py

# 5. Launch FastAPI Microservice
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

---

## 7. Web Prototype Features (`app/app.py`)
- **Theme**: Clean agriculture aesthetic with Deep Green (`#2E7D32`), Mint (`#F1F8F4`), and crisp white card containers.
- **Tab 1 "Diagnose"**: File uploader + 1-click sample leaf gallery $\rightarrow$ Tri-Panel (`Original Leaf | CAM Heatmap | Diagnostic Saliency Overlay`) + Diagnosis card with dynamic confidence badge + Top-3 probability distribution chart.
- **Tab 2 "Disease Encyclopedia"**: Curated etiology, symptom manifestations, and integrated cultural/chemical management tips from `app/disease_info.json`.
- **Tab 3 "Model Performance"**: Multi-model comparison table, confusion matrices, and training trajectory curves.
- **Tab 4 "Explainability & Faithfulness"**: Live interactive Deletion Test graph and lesion-vs-background qualitative gallery.

---

## 8. Limitations & Future Work

### Critical Limitations
> [!WARNING]
> **Lab-Background Bias**: All images in the PlantVillage dataset were captured under controlled laboratory lighting with uniform paper backgrounds. Models achieving near-perfect lab accuracy often experience performance degradation when exposed to in-field smartphone photographs (with varying sunlight, shadow, soil, weeds, and insect feeding marks).

### Future Directions
1. **Field Domain Transfer**: Fine-tuning with field-collected datasets (e.g. PlantDoc, Cassava Leaf Disease).
2. **Test-Time Augmentation (TTA)**: Multi-scale rotation and flip averaging during inference to stabilize field predictions.
3. **Edge Optimization**: Exporting models to ONNX and 8-bit quantized TFLite for zero-latency smartphone and drone deployment.
