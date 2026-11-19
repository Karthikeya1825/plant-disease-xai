"""
Interactive Streamlit Application: Plant Disease Detection with Explainable AI (XAI).
Theme: Deep green (#2E7D32), mint background (#F1F8F4), soft white cards, responsive layout.
Tabs:
1. Diagnose: Upload or sample leaf -> Tri-panel (Original | CAM Heatmap | Overlay) + Diagnosis + Top-3 Chart
2. Disease Info: Pathology details, symptoms, cultural/chemical management tips
3. Model Performance: Benchmark comparison, confusion matrices, training curves
4. Explainability & Faithfulness: Qualitative gallery & Quantitative deletion test
"""
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.config import config
from src.gradcam import GradCAM, overlay_cam_on_image, run_deletion_test
from src.models import get_model, get_target_layer_for_cam

# -------------------------------------------------------------
# STREAMLIT PAGE CONFIG & THEME
# -------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection with Explainable AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Agricultural Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide the Streamlit Cloud Deploy Button & Toolbar */
    [data-testid="stDeployButton"], [data-testid="stToolbarActions"] {
        display: none !important;
    }
    
    .stApp {
        background-color: #F1F8F4;
        color: #212121;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1B5E20, #2E7D32, #43A047);
        color: white;
        padding: 24px 32px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.2);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF !important;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 1.05rem;
        opacity: 0.95;
        color: #E8F5E9 !important;
    }
    
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border: 1px solid #E0E0E0;
        margin-bottom: 18px;
    }
    
    .badge-green {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #A5D6A7;
    }
    .badge-amber {
        background-color: #FFF8E1;
        color: #F57F17;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #FFE082;
    }
    .badge-red {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 12px;
        border-radius: 16px;
        font-weight: 600;
        display: inline-block;
        border: 1px solid #FFCDD2;
    }
    
    .footer-text {
        text-align: center;
        padding: 24px 0 10px 0;
        color: #555555;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CACHED RESOURCE LOADERS
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading deep learning model...")
def load_cached_model(model_name: str, device: str = "cpu"):
    """Load model architecture and weights with Streamlit resource caching."""
    with open(config.classes_json, "r") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    ckpt_path = config.models_dir / f"{model_name}_best.pt"
    model = get_model(model_name, num_classes=num_classes, pretrained=False)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # If best checkpoint not trained yet, use pretrained weights for demonstration
        model = get_model(model_name, num_classes=num_classes, pretrained=True)

    model = model.to(device)
    model.eval()
    return model, class_to_idx, idx_to_class

@st.cache_data
def load_disease_info() -> Dict[str, any]:
    """Load curated agronomic disease knowledgebase."""
    info_path = config.app_dir / "disease_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_metrics_report() -> Dict[str, any]:
    """Load model benchmarks and metrics."""
    if config.metrics_json.exists():
        with open(config.metrics_json, "r") as f:
            return json.load(f)
    return {}

def preprocess_image(pil_img: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    """Convert PIL image to preprocessed PyTorch tensor and clean uint8 RGB array."""
    pil_rgb = pil_img.convert("RGB")
    rgb_resized = pil_rgb.resize((config.image_size, config.image_size))
    rgb_np = np.array(rgb_resized)

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ])
    tensor = transform(pil_rgb).unsqueeze(0)
    return tensor, rgb_np

# -------------------------------------------------------------
# APPLICATION HEADER
# -------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🌿 Plant Disease Detection with Explainable AI</h1>
    <p>Research-grade deep transfer learning with transparent visual saliency (Grad-CAM & Grad-CAM++)</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_choice = st.selectbox(
        "Select Deep Learning Model",
        options=["resnet50", "efficientnet_b0", "mobilenet_v3", "baseline"],
        format_func=lambda x: {
            "resnet50": "ResNet-50 (Deep Residual)",
            "efficientnet_b0": "EfficientNet-B0 (Compound Scaled)",
            "mobilenet_v3": "MobileNetV3-Large (Edge Optimized)",
            "baseline": "Scratch 4-Stage CNN (Baseline)"
        }[x]
    )

    cam_method = st.radio(
        "Explainability (XAI) Algorithm",
        options=["Grad-CAM", "Grad-CAM++"],
        help="Grad-CAM++ provides higher-order gradient formulation for fine-grained multi-lesion localization."
    )

    alpha_val = st.slider(
        "Heatmap Overlay Opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.55,
        step=0.05
    )

    st.markdown("---")
    with st.expander("ℹ️ About the Project"):
        st.write("""
        **Final-Year Deep Learning Project**
        - **Dataset**: PlantVillage (Mohanty et al., 2016)
        - **Core**: 38 crop condition classes across 14 plant species.
        - **Extension**: Grad-CAM/Grad-CAM++ saliency visualization & faithfulness deletion tests.
        - **Architecture**: Transfer learning with AdamW & Cosine/Plateau scheduling.
        """)

# Load resources
disease_kb = load_disease_info()
metrics_data = load_metrics_report()
model, class_to_idx, idx_to_class = load_cached_model(model_choice, device=config.device)

# -------------------------------------------------------------
# MAIN APPLICATION TABS
# -------------------------------------------------------------
tab_diagnose, tab_info, tab_perf, tab_xai = st.tabs([
    "🔍 Diagnose Leaf",
    "📖 Disease Encyclopedia",
    "📊 Model Performance",
    "🔬 Explainability & Faithfulness"
])

# -------------------------------------------------------------
# TAB 1: DIAGNOSE LEAF
# -------------------------------------------------------------
with tab_diagnose:
    st.subheader("1. Select or Upload Crop Leaf Image")
    
    col_upload, col_samples = st.columns([1.5, 1])
    
    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload leaf photograph (PNG, JPG, JPEG)",
            type=["png", "jpg", "jpeg"]
        )

    # Discover sample images from asset directory or test set
    sample_assets_dir = config.app_dir / "assets"
    sample_files = list(sample_assets_dir.glob("*.jpg")) + list(sample_assets_dir.glob("*.png"))

    selected_sample_path = None
    with col_samples:
        st.markdown("**Or load a quick test sample:**")
        if sample_files:
            sample_names = [f.stem.replace("___", " - ").replace("_", " ") for f in sample_files]
            sample_sel = st.selectbox("Choose sample image:", ["None"] + sample_names)
            if sample_sel != "None":
                selected_sample_path = sample_files[sample_names.index(sample_sel)]
        else:
            st.info("Sample assets will populate after dataset audit.")

    active_image = None
    if uploaded_file is not None:
        try:
            active_image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Invalid image file: {e}")
    elif selected_sample_path is not None:
        active_image = Image.open(selected_sample_path)

    if active_image is not None:
        st.markdown("---")
        with st.spinner("Analyzing foliar pathology & computing Grad-CAM gradients..."):
            tensor_input, rgb_np = preprocess_image(active_image)
            tensor_input = tensor_input.to(config.device)

            # Grad-CAM computation
            target_layer = get_target_layer_for_cam(model, model_choice)
            cam_engine = GradCAM(model, target_layer)
            heatmap, pred_idx, confidence = cam_engine.generate_cam(
                tensor_input,
                method=cam_method.lower()
            )
            cam_engine.remove_hooks()

            overlay_img = overlay_cam_on_image(rgb_np, heatmap, alpha=alpha_val)

            # Top-3 predictions
            with torch.no_grad():
                logits = model(tensor_input)
                all_probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
            top3_indices = np.argsort(all_probs)[::-1][:3]
            pred_class_name = idx_to_class[pred_idx]

        # Display Tri-Panel
        st.subheader("2. Visual Explainability Tri-Panel")
        col_orig, col_cam, col_overlay = st.columns(3)

        with col_orig:
            st.markdown("<div class='card'><b>Input Leaf Image</b>", unsafe_allow_html=True)
            st.image(rgb_np, use_container_width=True)
            st.caption(f"Input Resolution: {rgb_np.shape[0]}x{rgb_np.shape[1]} px")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_cam:
            st.markdown(f"<div class='card'><b>{cam_method} Heatmap</b>", unsafe_allow_html=True)
            heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            st.image(heatmap_colored, use_container_width=True)
            st.caption("Normalized saliency gradient intensity (0.0 to 1.0)")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_overlay:
            st.markdown("<div class='card'><b>Diagnostic Saliency Overlay</b>", unsafe_allow_html=True)
            st.image(overlay_img, use_container_width=True)
            st.caption(f"Blended with opacity α = {alpha_val:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Diagnostic Report Card
        st.subheader("3. Diagnostic Prediction")
        col_diag, col_chart = st.columns([1.2, 1])

        clean_crop = pred_class_name.split("___")[0].replace("_", " ")
        clean_disease = pred_class_name.split("___")[1].replace("_", " ") if "___" in pred_class_name else pred_class_name

        with col_diag:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### 🌱 Crop: **{clean_crop}**")
            st.markdown(f"### 🩺 Diagnosis: **{clean_disease}**")

            # Colored badge based on confidence threshold
            conf_pct = confidence * 100
            if conf_pct >= 85.0:
                badge_html = f"<span class='badge-green'>High Confidence: {conf_pct:.2f}%</span>"
            elif conf_pct >= 60.0:
                badge_html = f"<span class='badge-amber'>Moderate Confidence: {conf_pct:.2f}%</span>"
            else:
                badge_html = f"<span class='badge-red'>Low Confidence: {conf_pct:.2f}%</span>"

            st.markdown(badge_html, unsafe_allow_html=True)

            if conf_pct < 60.0:
                st.warning("⚠️ **Low Confidence Diagnosis**: The model is uncertain. Visual symptoms may be ambiguous, co-infected, or outside the training distribution.")

            st.markdown("</div>", unsafe_allow_html=True)

        with col_chart:
            st.markdown("<div class='card'><b>Top-3 Candidate Predictions</b>", unsafe_allow_html=True)
            top3_labels = [idx_to_class[i].replace("___", " - ").replace("_", " ") for i in top3_indices]
            top3_vals = [all_probs[i] * 100 for i in top3_indices]

            fig_bar, ax_bar = plt.subplots(figsize=(6, 2.5))
            y_pos = np.arange(len(top3_labels))
            ax_bar.barh(y_pos, top3_vals, color=["#2E7D32", "#81C784", "#C8E6C9"])
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(top3_labels, fontsize=8)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel("Probability (%)", fontsize=8)
            ax_bar.set_xlim(0, 100)
            for i, v in enumerate(top3_vals):
                ax_bar.text(v + 1.5, i + 0.1, f"{v:.1f}%", fontsize=8, color="#1B5E20", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close()
            st.markdown("</div>", unsafe_allow_html=True)

        # Store predicted class in session state for Tab 2
        st.session_state["last_prediction"] = pred_class_name
    else:
        st.info("👆 Please upload an image or select a sample leaf above to run diagnosis.")

# -------------------------------------------------------------
# TAB 2: DISEASE ENCYCLOPEDIA & MANAGEMENT
# -------------------------------------------------------------
with tab_info:
    st.subheader("Agronomic Pathology Knowledgebase")
    st.caption("Detailed etiology, symptom manifestation, and integrated pest management (IPM) guidelines.")

    last_pred = st.session_state.get("last_prediction", "Apple___Apple_scab")
    
    available_classes = sorted(list(disease_kb.keys())) if disease_kb else []
    if available_classes:
        selected_info_class = st.selectbox(
            "Select disease profile:",
            available_classes,
            index=available_classes.index(last_pred) if last_pred in available_classes else 0,
            format_func=lambda x: x.replace("___", " — ").replace("_", " ")
        )
        info = disease_kb.get(selected_info_class, {})

        col_desc, col_mgmt = st.columns(2)
        with col_desc:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"### 🌿 {info.get('crop', 'Crop')}: **{info.get('disease', 'Disease')}**")
            st.markdown(f"**Causal Agent**: `{info.get('pathogen', 'N/A')}`")
            st.markdown("#### Description")
            st.write(info.get("description", "No description available."))
            st.markdown("#### Symptoms")
            st.write(info.get("symptoms", "No symptoms available."))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_mgmt:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🛡️ Recommended Management Tips")
            st.write(info.get("management", "No management guidance available."))
            st.markdown("---")
            st.warning("⚠️ **Agronomic Disclaimer**: *This diagnosis is informational only and generated by an AI research model. Always consult a certified agricultural extension officer or professional agronomist before applying chemical treatments.*")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Loading disease knowledgebase...")

# -------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & BENCHMARKS
# -------------------------------------------------------------
with tab_perf:
    st.subheader("Isolated Test Set Benchmark Evaluation")

    if metrics_data:
        # Comparison Table
        table_rows = []
        for mname, m in metrics_data.items():
            table_rows.append({
                "Model Architecture": mname.upper(),
                "Parameters (M)": f"{m.get('parameters_total', 0) / 1e6:.2f}M",
                "Disk Size": f"{m.get('size_mb', 0):.1f} MB",
                "Latency (CPU)": f"{m.get('latency_ms_per_image', 0):.2f} ms",
                "Accuracy": f"{m.get('accuracy', 0)*100:.2f}%",
                "Macro F1": f"{m.get('f1_macro', 0)*100:.2f}%",
                "Weighted F1": f"{m.get('f1_weighted', 0)*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

        col_cm, col_curves = st.columns(2)
        with col_cm:
            st.markdown(f"**Confusion Matrix: {model_choice.upper()}**")
            cm_img_path = config.figures_dir / f"{model_choice}_confusion_matrix.png"
            if cm_img_path.exists():
                st.image(str(cm_img_path), use_container_width=True)
            else:
                st.info("Run evaluation script to generate high-resolution confusion matrix.")

        with col_curves:
            st.markdown(f"**Training & Validation Trajectory: {model_choice.upper()}**")
            curves_img_path = config.figures_dir / f"{model_choice}_training_curves.png"
            if curves_img_path.exists():
                st.image(str(curves_img_path), use_container_width=True)
            else:
                st.info("Curves will display after model training completes.")
    else:
        st.info("Metrics benchmarks will appear once training and evaluation stages complete.")

# -------------------------------------------------------------
# TAB 4: EXPLAINABILITY & FAITHFULNESS
# -------------------------------------------------------------
with tab_xai:
    st.subheader("Quantitative Faithfulness: Deletion Test Sanity Check")
    st.write("""
    To verify whether Grad-CAM explanations are **faithful** (i.e. truly highlighting pixels crucial to the network's prediction rather than arbitrary edges),
    we execute a **Deletion Test**:
    - We iteratively mask the top-salient pixels identified by CAM (10%, 20%, ..., 50%).
    - We compare the resulting confidence collapse against random pixel ablation.
    """)

    if active_image is not None and st.button("🧪 Run Deletion Test on Current Image"):
        with st.spinner("Running ablation steps across masking thresholds..."):
            tensor_input, _ = preprocess_image(active_image)
            target_layer = get_target_layer_for_cam(model, model_choice)
            cam_engine = GradCAM(model, target_layer)
            del_results = run_deletion_test(model, cam_engine, tensor_input, device=config.device)
            cam_engine.remove_hooks()

        fig_del, ax_del = plt.subplots(figsize=(8, 4))
        fractions_pct = [f * 100 for f in del_results["mask_fractions"]]
        ax_del.plot(fractions_pct, del_results["salient_drops"], marker="o", color="#D32F2F", linewidth=2.5, label="CAM Salient Masking (Hypothesis)")
        ax_del.plot(fractions_pct, del_results["random_drops"], marker="s", color="#757575", linewidth=2.0, linestyle="--", label="Random Masking (Baseline Control)")
        ax_del.set_xlabel("Percentage of Pixels Masked (%)", fontsize=11)
        ax_del.set_ylabel("Drop in Prediction Confidence (ΔP)", fontsize=11)
        ax_del.set_title("Faithfulness Verification: Prediction Confidence Collapse", fontsize=12)
        ax_del.grid(True, linestyle=":", alpha=0.6)
        ax_del.legend(loc="upper left")
        st.pyplot(fig_del)
        plt.close()

        if del_results["is_faithful"]:
            st.success("✅ **Sanity Check Passed**: Masking CAM-salient regions caused a significantly steeper drop in network confidence than random pixel ablation, verifying attribution faithfulness.")
        else:
            st.warning("⚠️ Saliency drop did not exceed random baseline on this sample.")

    st.markdown("---")
    st.subheader("Qualitative Lesion vs. Background Audit")
    misclass_img_path = config.figures_dir / f"{model_choice}_misclassified_samples.png"
    if misclass_img_path.exists():
        st.image(str(misclass_img_path), caption="Inspection of Failure Cases & Ambiguous Boundaries", use_container_width=True)

# -------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer-text">
    <b>Plant Disease Detection with Explainable AI</b> | Final-Year Deep Learning Project<br>
    Dataset: PlantVillage (Mohanty et al., 2016) | Tested on PyTorch 2.x & Streamlit<br>
    <i>Limitation Notice: PlantVillage consists of lab-acquired photos on uniform backgrounds; models require in-field validation before practical agricultural deployment.</i>
</div>
""", unsafe_allow_html=True)
