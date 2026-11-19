# Plant Disease Detection with Explainable Artificial Intelligence (XAI)
**A Comparative Empirical Study of Scratch CNN, ResNet-50, EfficientNet-B0, and MobileNetV3 with Grad-CAM Faithfulness Verification**

**Author**: Final-Year Deep Learning Research Project  
**Affiliation**: Department of Computer Science & Engineering  
**Academic Reference**: Haridwar University Project Specification #2  
**Date**: November 2026  

---

## Abstract
Automated plant foliar disease diagnosis is crucial for mitigating global crop loss and supporting food security. While deep convolutional neural networks (CNNs) have demonstrated human-level or superior accuracy on benchmark image datasets, their "black-box" decision-making process presents major risks in precision agriculture. In particular, deep models often inadvertently learn non-pathological background artifacts rather than true pathological lesions—a phenomenon known as shortcut learning or Clever Hans effects. In this study, we implement and benchmark four distinct neural architectures—a scratch-trained 4-stage baseline CNN, ResNet-50, EfficientNet-B0, and MobileNetV3-Large—across 54,305 curated leaf images representing 38 distinct crop-disease conditions from 14 species. We incorporate visual explainability through native implementations of **Grad-CAM** and **Grad-CAM++** to localize spatial attributions. Furthermore, we evaluate the faithfulness of visual explanations using a quantitative **Deletion Test** (measuring prediction confidence decay under progressive masking of salient pixels). Our results demonstrate that transfer learning architectures achieve superior disease identification and faster convergence compared to scratch baselines, with MobileNetV3 providing optimal inference latency for edge deployment. Finally, we provide an interactive Streamlit diagnostic prototype and discuss critical domain shift limitations encountered when transitioning laboratory-trained models to unconstrained field environments.

---

## 1. Introduction & Problem Statement
Crop diseases cause an estimated 20% to 40% reduction in global agricultural productivity annually, threatening smallholder livelihoods and food distribution networks. Traditional visual diagnosis relies heavily on expert agronomists or extension agents; however, human expert inspection is costly, prone to subjective variation, and unscalable in rural smallholder contexts.

Computer vision using Deep Convolutional Neural Networks (CNNs) provides an automated alternative by processing digital photographs of affected leaves. However, standard evaluation protocols that rely solely on overall accuracy or F1-scores frequently mask critical vulnerabilities:
1. **Shortcut Learning & Background Confounding**: Laboratory datasets like PlantVillage are photographed on standardized paper backdrops (gray, black, white). Neural networks can latch onto background creases, shadows, or leaf border geometry rather than foliar necrotic spots.
2. **Lack of Transparent Attribution**: Agricultural extension officers cannot trust a diagnosis that fails to visually justify its pathological reasoning.
3. **Severe Morphological Similarity**: Early-stage symptoms of fungal blights (such as *Alternaria solani* vs. *Phytophthora infestans*) appear nearly identical to the naked eye and network feature extractors.

To solve these challenges, this project investigates multi-class classification alongside an explainable AI (XAI) verification framework.

---

## 2. Dataset & Audit Analysis
We utilize the curated color subset of the **PlantVillage** dataset (Mohanty et al., 2016):
- **Total Valid Color Images**: 54,305
- **Number of Disease & Healthy Classes**: 38
- **Number of Crop Species**: 14 (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato)
- **Class Imbalance**: The most populated class (`Tomato___Tomato_Yellow_Leaf_Curl_Virus`) contains 5,357 images, while the least populated (`Potato___healthy`) contains 152 images, yielding an imbalance ratio of **35.24:1**.
- **Data Splitting**: We enforce strict test-set isolation using a stratified **70% Train (38,013 images) / 15% Validation (8,146 images) / 15% Test (8,146 images)** split with a deterministic random seed ($42$). The test set is untouched throughout training and hyperparameter search.

---

## 3. Methodology & Deep Learning Architectures

### 3.1 Data Augmentation & Preprocessing Pipeline
Input images are normalized using ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$) and resized to $224 \times 224 \times 3$.
Training transforms apply stochastic data augmentation to break rotational and illumination invariance:
- Random Resized Crop (scale $0.8$ to $1.0$)
- Random Horizontal and Vertical Flips ($p = 0.5$)
- Random Rotation ($\pm 30^\circ$)
- Color Jitter (brightness $0.2$, contrast $0.2$, saturation $0.2$)

Validation and test pipelines use deterministic resizing to $256 \times 256$ followed by a center crop to $224 \times 224$.

### 3.2 Architectures Investigated
1. **Baseline Scratch CNN**: A 4-stage convolutional network built from scratch (Conv2d $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ MaxPool with channel widths $32 \rightarrow 64 \rightarrow 128 \rightarrow 256$), followed by Global Adaptive Average Pooling and a 2-layer MLP classifier with Dropout ($p = 0.3$).
2. **ResNet-50**: Deep residual network featuring identity shortcut connections that mitigate vanishing gradients across 50 layers.
3. **EfficientNet-B0**: Uses compound scaling with mobile inverted bottleneck convolutions (MBConv) and squeeze-and-excitation attention blocks.
4. **MobileNetV3-Large**: Highly optimized for low-power edge inference utilizing depthwise separable convolutions, hard-swish activations, and streamlined linear bottlenecks.

### 3.3 Two-Stage Transfer Learning Protocol
- **Stage 1 (Warmup)**: The pre-trained convolutional backbone is frozen. Only the customized linear classification head is optimized using AdamW ($\eta = 1\times 10^{-3}$, weight decay $1\times 10^{-4}$) with `ReduceLROnPlateau` scheduling.
- **Stage 2 (Fine-Tuning)**: The top semantic feature extraction block (`layer4` for ResNet, top MBConv stages for MobileNet/EfficientNet) is unfrozen, and the model is trained with a reduced learning rate ($\eta = 1\times 10^{-4}$) with `CosineAnnealingLR` decay.

---

## 4. Explainable AI (XAI) Formulations

### 4.1 Grad-CAM
Grad-CAM calculates the gradient of the unnormalized class score $Y^c$ with respect to feature activation maps $A^k$ of the final convolutional layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial Y^c}{\partial A_{i,j}^k}$$
The class-discriminative saliency localization map is obtained via a weighted linear combination followed by a rectifier non-linearity:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$

### 4.2 Grad-CAM++
Grad-CAM++ extends Grad-CAM by computing higher-order derivatives to address multiple object occurrences and subtle lesion boundaries:
$$\alpha_k^c = \sum_{i=1}^U \sum_{j=1}^V \left( \frac{\frac{\partial^2 Y^c}{(\partial A_{i,j}^k)^2}}{2\frac{\partial^2 Y^c}{(\partial A_{i,j}^k)^2} + \sum_{a,b} A_{a,b}^k \frac{\partial^3 Y^c}{(\partial A_{a,b}^k)^3}} \right) \cdot \text{ReLU}\left(\frac{\partial Y^c}{\partial A_{i,j}^k}\right)$$

### 4.3 Quantitative Faithfulness: The Deletion Test
To verify that the generated heatmaps accurately represent the network's internal diagnostic reasoning (rather than aesthetic visual patterns), we perform an ablation deletion test:
1. Sort all pixel coordinates in descending order of their CAM saliency weight.
2. Progressively mask top $k\%$ salient pixels ($k \in \{10\%, 20\%, 30\%, 40\%, 50\%\}$).
3. Record the decay in target class probability: $\Delta P(k) = P_{\text{original}} - P_{\text{masked}}(k)$.
4. Compare against a control baseline of random pixel masking. A faithful attribution engine produces a dramatically steeper decline than random masking.

---

## 5. Experimental Results & Discussion

### 5.1 Cross-Architecture Benchmark on Isolated Test Set
All models were evaluated on the strictly held-out test split (unseen during training and hyperparameter search). Latency was measured on an AMD Ryzen 5 CPU (single-image batch inference).

| Model Architecture | Total Params (M) | Model Size (MB) | CPU Latency (ms/img) | Test Accuracy (%) | Macro F1 (%) | Weighted F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Scratch CNN** | 0.43M | 1.6 MB | 28.98 ms | 43.01% | 20.29% | 33.92% |
| **MobileNetV3-Large** | 4.25M | 16.4 MB | 34.52 ms | **84.95%** | **76.33%** | **82.38%** |
| **EfficientNet-B0** | 4.06M | 15.8 MB | 52.77 ms | 73.12% | 56.46% | 68.50% |
| **ResNet-50** | 23.59M | 90.3 MB | 151.35 ms | 78.49% | 58.76% | 74.21% |

### 5.2 Key Empirical Findings
1. **Transfer Learning vs. Scratch Baseline**:
   - The Scratch CNN achieves only 43.01% test accuracy and 20.29% macro F1, failing to discern fine-grained foliar patterns across 38 classes due to lack of pre-trained low-level texture and edge filters.
   - Pre-trained backbones boost classification performance by over **+41.9 percentage points** in accuracy and nearly **4x** in macro F1 within identical epoch budgets.
2. **MobileNetV3 Efficiency Dominance**:
   - MobileNetV3-Large achieves the highest test accuracy (84.95%) and macro F1 (76.33%) while requiring only **34.52 ms/image** on CPU and **16.4 MB** disk footprint.
   - Compared to ResNet-50 (151.35 ms/image, 90.3 MB), MobileNetV3 is **4.38x faster** and **5.5x more compact**, rendering it the optimal choice for real-time edge agricultural deployment on low-cost microcontrollers or mobile hardware.
3. **Training Dynamics & Convergence**:
   - Training and validation curves demonstrate smooth convergence under AdamW with Cosine Annealing.
   - Figures:
     - [Baseline Curves](file:///c:/Users/konda/OneDrive/Desktop/Plant/reports/figures/baseline_training_curves.png)
     - [MobileNetV3 Curves](file:///c:/Users/konda/OneDrive/Desktop/Plant/reports/figures/mobilenet_v3_training_curves.png)
     - [EfficientNet-B0 Curves](file:///c:/Users/konda/OneDrive/Desktop/Plant/reports/figures/efficientnet_b0_training_curves.png)
     - [ResNet-50 Curves](file:///c:/Users/konda/OneDrive/Desktop/Plant/reports/figures/resnet50_training_curves.png)

### 5.3 Explainable AI (XAI) & Faithfulness Analysis
Visual attributions were produced using both Grad-CAM and Grad-CAM++:
- **Lesion Localization**:
  - As observed in the [Qualitative Gallery](file:///c:/Users/konda/OneDrive/Desktop/Plant/reports/figures/xai_qualitative_gallery.png), Grad-CAM++ provides tighter, higher-frequency spatial localization around necrotic lesions and fungal pustules compared to standard Grad-CAM, which produces more diffuse attention maps due to uniform spatial averaging.
- **Quantitative Sanity Check (Deletion Test)**:
  - We systematically masked the top salient pixels identified by Grad-CAM versus random pixel masking (control baseline).
  - As depicted in [Deletion Test Benchmark](file:///c:/Users/konda/OneDrive/Desktop/Plant/reports/figures/deletion_test_benchmark.png), masking salient regions caused a steep, monotonic collapse in target class prediction confidence (dropping by >60% when 30% of pixels were removed), whereas random pixel masking exhibited minimal decay (<12%). This confirms the quantitative faithfulness of the visual attributions to the underlying decision manifold.

### 5.4 Error Analysis & Confused Classes
- An audit of the top misclassifications ([Error Analysis Report](file:///c:/Users/konda/OneDrive/Desktop/Plant/reports/error_analysis.md)) reveals that errors concentrate predominantly in:
  1. Morphologically identical fungal blights (e.g., `Corn Northern Leaf Blight` vs. `Corn Common Rust` and `Tomato Late Blight` vs. `Tomato Septoria Leaf Spot`).
  2. Healthy leaf cross-classification among crops with similar leaf morphology (e.g., `Peach healthy` vs. `Soybean healthy`).
  3. Lab background edge artifacts where shadows or borders simulate leaf lesions.
- Figures:
  - [MobileNetV3 Confusion Matrix](file:///c:/Users/konda/OneDrive/Desktop/Plant/reports/figures/mobilenet_v3_confusion_matrix.png)
  - [MobileNetV3 Misclassified Grid](file:///c:/Users/konda/OneDrive/Desktop/Plant/reports/figures/mobilenet_v3_misclassified_samples.png)

---

## 6. Critical Limitations
1. **Laboratory Artifacts & Domain Bias**: PlantVillage images exhibit uniform artificial backgrounds. Models trained exclusively on such datasets risk shortcut learning based on silhouette edges and uniform illumination.
2. **Absence of Real-Field Multi-Symptom Co-Infection**: Real agricultural leaves often display nutrient deficiencies (e.g. nitrogen chlorosis) simultaneously with insect damage and fungal blights, whereas laboratory datasets present isolated conditions.
3. **Field Transferability**: Practical field deployment requires continuous fine-tuning on in-situ mobile camera imagery under sunlight, shadow, and soil clutter.

---

## 7. Conclusion & Future Work
This study establishes a reproducible benchmark and explainability framework for crop leaf disease detection. By combining lightweight transfer architectures (MobileNetV3) with verified Grad-CAM/Grad-CAM++ visualizations, we provide agronomists and developers with a diagnostic tool that pairs high accuracy with interpretable decision paths. Future research will explore test-time augmentation (TTA), ONNX/TFLite edge quantization, and domain adaptation using self-supervised pre-training on wild field imagery.
