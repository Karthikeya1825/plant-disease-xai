# Dataset Audit Report: PlantVillage (Color Subset)

## 1. Executive Summary
- **Total Images**: 54,305
- **Number of Classes**: 38
- **Number of Crop Species Represented**: 14
- **Class Imbalance Ratio**: 36.23:1
  - **Most Populated Class**: `Orange___Haunglongbing_(Citrus_greening)` (5,507 images)
  - **Least Populated Class**: `Potato___healthy` (152 images)
- **Image Resolution Distribution**:
  - Width: min=256px, max=256px, mean=256.0px
  - Height: min=256px, max=256px, mean=256.0px
  - Color Channels: [np.int64(3)] (RGB)

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
To strictly isolate the evaluation, the test set was created prior to any training or tuning using fixed seed `42`.

| Partition | Total Images | Percentage | Role |
| :--- | :--- | :--- | :--- |
| **Train** | 38,013 | 70.0% | Optimization and feature learning |
| **Validation** | 8,146 | 15.0% | Early stopping and checkpoint selection |
| **Test** | 8,146 | 15.0% | Isolated final evaluation |
| **Total** | 54,305 | 100.0% | Complete curated dataset |

---

## 5. Crop Distribution
| Crop Species | Total Images | Conditions Included |
| :--- | :--- | :--- |
| Tomato | 18,160 | Multiple diseased & healthy states |
| Orange | 5,507 | Multiple diseased & healthy states |
| Soybean | 5,090 | Multiple diseased & healthy states |
| Grape | 4,062 | Multiple diseased & healthy states |
| Corn (maize) | 3,852 | Multiple diseased & healthy states |
| Apple | 3,171 | Multiple diseased & healthy states |
| Peach | 2,657 | Multiple diseased & healthy states |
| Pepper, bell | 2,475 | Multiple diseased & healthy states |
| Potato | 2,152 | Multiple diseased & healthy states |
| Cherry (including sour) | 1,906 | Multiple diseased & healthy states |
| Squash | 1,835 | Multiple diseased & healthy states |
| Strawberry | 1,565 | Multiple diseased & healthy states |
| Blueberry | 1,502 | Multiple diseased & healthy states |
| Raspberry | 371 | Multiple diseased & healthy states |

---

## 6. Per-Class Distribution & Split Counts
| # | Class Name | Total | Train (70%) | Val (15%) | Test (15%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Apple___Apple_scab` | 630 | 441 | 95 | 94 |
| 2 | `Apple___Black_rot` | 621 | 435 | 93 | 93 |
| 3 | `Apple___Cedar_apple_rust` | 275 | 193 | 41 | 41 |
| 4 | `Apple___healthy` | 1,645 | 1152 | 247 | 246 |
| 5 | `Blueberry___healthy` | 1,502 | 1051 | 225 | 226 |
| 6 | `Cherry_(including_sour)___Powdery_mildew` | 1,052 | 736 | 158 | 158 |
| 7 | `Cherry_(including_sour)___healthy` | 854 | 598 | 128 | 128 |
| 8 | `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot` | 513 | 359 | 77 | 77 |
| 9 | `Corn_(maize)___Common_rust_` | 1,192 | 834 | 179 | 179 |
| 10 | `Corn_(maize)___Northern_Leaf_Blight` | 985 | 690 | 148 | 147 |
| 11 | `Corn_(maize)___healthy` | 1,162 | 813 | 175 | 174 |
| 12 | `Grape___Black_rot` | 1,180 | 826 | 177 | 177 |
| 13 | `Grape___Esca_(Black_Measles)` | 1,383 | 968 | 208 | 207 |
| 14 | `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)` | 1,076 | 753 | 161 | 162 |
| 15 | `Grape___healthy` | 423 | 296 | 63 | 64 |
| 16 | `Orange___Haunglongbing_(Citrus_greening)` | 5,507 | 3855 | 826 | 826 |
| 17 | `Peach___Bacterial_spot` | 2,297 | 1608 | 345 | 344 |
| 18 | `Peach___healthy` | 360 | 252 | 54 | 54 |
| 19 | `Pepper,_bell___Bacterial_spot` | 997 | 698 | 149 | 150 |
| 20 | `Pepper,_bell___healthy` | 1,478 | 1035 | 221 | 222 |
| 21 | `Potato___Early_blight` | 1,000 | 700 | 150 | 150 |
| 22 | `Potato___Late_blight` | 1,000 | 700 | 150 | 150 |
| 23 | `Potato___healthy` | 152 | 106 | 23 | 23 |
| 24 | `Raspberry___healthy` | 371 | 260 | 56 | 55 |
| 25 | `Soybean___healthy` | 5,090 | 3563 | 763 | 764 |
| 26 | `Squash___Powdery_mildew` | 1,835 | 1284 | 276 | 275 |
| 27 | `Strawberry___Leaf_scorch` | 1,109 | 776 | 166 | 167 |
| 28 | `Strawberry___healthy` | 456 | 319 | 69 | 68 |
| 29 | `Tomato___Bacterial_spot` | 2,127 | 1489 | 319 | 319 |
| 30 | `Tomato___Early_blight` | 1,000 | 700 | 150 | 150 |
| 31 | `Tomato___Late_blight` | 1,909 | 1336 | 286 | 287 |
| 32 | `Tomato___Leaf_Mold` | 952 | 666 | 143 | 143 |
| 33 | `Tomato___Septoria_leaf_spot` | 1,771 | 1240 | 266 | 265 |
| 34 | `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 | 1173 | 251 | 252 |
| 35 | `Tomato___Target_Spot` | 1,404 | 983 | 210 | 211 |
| 36 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 | 3750 | 804 | 803 |
| 37 | `Tomato___Tomato_mosaic_virus` | 373 | 261 | 56 | 56 |
| 38 | `Tomato___healthy` | 1,591 | 1114 | 238 | 239 |

---

## 7. Sample Visual Grid
Sample images from representative crop-disease pairs are visualised in:
`reports/figures/dataset_sample_grid.png`
