# Qualitative & Quantitative Error Analysis

## 1. Overview for Model: `resnet50`
- **Evaluated Test Samples**: 93
- **Total Misclassified Samples**: 20
- **Error Rate**: 21.51%

---

## 2. Top-10 Most Confused Class Pairs
These pairs share morphological similarities (early-stage necrotic lesions, chlorotic halos, or co-occurring fungal blights):

| Rank | True Disease Condition | Predicted Condition | Misclassified Count |
| :--- | :--- | :--- | :--- |
| 1 | `Corn_(maize)___Northern_Leaf_Blight` | `Corn_(maize)___Common_rust_` | 2 |
| 2 | `Tomato___Late_blight` | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 1 |
| 3 | `Tomato___Septoria_leaf_spot` | `Tomato___Late_blight` | 1 |
| 4 | `Peach___healthy` | `Soybean___healthy` | 1 |
| 5 | `Tomato___healthy` | `Soybean___healthy` | 1 |
| 6 | `Apple___healthy` | `Blueberry___healthy` | 1 |
| 7 | `Potato___Late_blight` | `Orange___Haunglongbing_(Citrus_greening)` | 1 |
| 8 | `Tomato___Leaf_Mold` | `Tomato___Late_blight` | 1 |
| 9 | `Apple___Apple_scab` | `Tomato___healthy` | 1 |
| 10 | `Pepper,_bell___healthy` | `Soybean___healthy` | 1 |

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
`reports/figures/resnet50_misclassified_samples.png`
