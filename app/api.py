"""
FastAPI REST API for Plant Disease Diagnosis with Explainable AI.
Endpoints:
- GET /health: Health check, device, loaded models
- POST /predict: Multi-class prediction with top-k candidates and confidence scores
- POST /explain: Grad-CAM / Grad-CAM++ explanation returning base64 overlay image
"""
import base64
import io
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import config
from src.models import get_model, get_target_layer_for_cam
from src.gradcam import GradCAM, overlay_cam_on_image

class ModelName(str, Enum):
    resnet50 = "resnet50"
    efficientnet_b0 = "efficientnet_b0"
    mobilenet_v3 = "mobilenet_v3"
    baseline = "baseline"

class CAMMethod(str, Enum):
    gradcam = "gradcam"
    gradcam_pp = "gradcam++"

app = FastAPI(
    title="Plant Disease Detection & XAI API",
    description="Deep learning inference engine with Grad-CAM visual explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_REGISTRY = {}
CLASS_TO_IDX = {}
IDX_TO_CLASS = {}

@app.on_event("startup")
def startup_event():
    global CLASS_TO_IDX, IDX_TO_CLASS
    if config.classes_json.exists():
        with open(config.classes_json, "r") as f:
            CLASS_TO_IDX = json.load(f)
        IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
    else:
        CLASS_TO_IDX = {}
        IDX_TO_CLASS = {}

def get_or_load_model(model_name: str = "resnet50"):
    device = config.device
    if model_name not in MODEL_REGISTRY:
        num_classes = len(CLASS_TO_IDX) if CLASS_TO_IDX else 38
        model = get_model(model_name, num_classes=num_classes, pretrained=False)
        ckpt_path = config.models_dir / f"{model_name}_best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model = get_model(model_name, num_classes=num_classes, pretrained=True)
        model = model.to(device)
        model.eval()
        MODEL_REGISTRY[model_name] = model
    return MODEL_REGISTRY[model_name]

def process_upload_image(file_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray]:
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    rgb_resized = pil_img.resize((config.image_size, config.image_size))
    rgb_np = np.array(rgb_resized)

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.mean, std=config.std)
    ])
    tensor = transform(pil_img).unsqueeze(0).to(config.device)
    return tensor, rgb_np

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": config.device,
        "classes_registered": len(CLASS_TO_IDX),
        "cached_models": list(MODEL_REGISTRY.keys())
    }

@app.post("/predict")
async def predict_disease(
    file: UploadFile = File(...),
    model_name: ModelName = Query(ModelName.mobilenet_v3, description="Model architecture"),
    top_k: int = Query(3, ge=1, le=10)
):
    content = await file.read()
    tensor_img, _ = process_upload_image(content)
    m_name = model_name.value if isinstance(model_name, ModelName) else str(model_name)
    model = get_or_load_model(m_name)

    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    predictions = []
    for idx in top_indices:
        cname = IDX_TO_CLASS.get(int(idx), f"Class_{idx}")
        predictions.append({
            "class_id": int(idx),
            "class_name": cname,
            "crop": cname.split("___")[0].replace("_", " "),
            "condition": cname.split("___")[1].replace("_", " ") if "___" in cname else cname,
            "probability": float(probs[idx])
        })

    return {
        "model": m_name,
        "top_prediction": predictions[0],
        "top_candidates": predictions
    }

@app.post("/explain")
async def explain_prediction(
    file: UploadFile = File(...),
    model_name: ModelName = Query(ModelName.mobilenet_v3, description="Model architecture"),
    method: CAMMethod = Query(CAMMethod.gradcam, description="CAM method (gradcam or gradcam++)"),
    alpha: float = Query(0.55, ge=0.1, le=0.9)
):
    content = await file.read()
    tensor_img, rgb_np = process_upload_image(content)
    m_name = model_name.value if isinstance(model_name, ModelName) else str(model_name)
    cam_m = method.value if isinstance(method, CAMMethod) else str(method)
    model = get_or_load_model(m_name)

    target_layer = get_target_layer_for_cam(model, m_name)
    cam_engine = GradCAM(model, target_layer)
    heatmap, pred_idx, confidence = cam_engine.generate_cam(tensor_img, method=cam_m)
    cam_engine.remove_hooks()

    overlay = overlay_cam_on_image(rgb_np, heatmap, alpha=alpha)

    # Encode overlay image to base64 JPEG
    success, buffer = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode overlay image.")
    b64_str = base64.b64encode(buffer).decode("utf-8")

    cname = IDX_TO_CLASS.get(pred_idx, f"Class_{pred_idx}")

    return {
        "model": model_name,
        "method": method,
        "predicted_class": cname,
        "confidence": confidence,
        "overlay_base64_jpeg": f"data:image/jpeg;base64,{b64_str}"
    }
