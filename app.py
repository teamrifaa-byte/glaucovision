"""
Glaucoma Detection API — FastAPI Backend
Multimodal Fundus + OCT inference using ResNet50 + Weighted Average Fusion (alpha=0.49)
"""

import os
import io
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
BEST_ALPHA    = 0.49          # from notebook 05 grid search
THRESHOLD     = 0.5
MODELS_DIR    = Path("models")
DEVICE        = torch.device("cpu")  # HF Spaces free tier → CPU


# ── Model Architecture (identical to notebooks 03 & 04) ───────────────────
def build_resnet50_binary() -> nn.Module:
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, 1),
    )
    return model


# ── Transforms ─────────────────────────────────────────────────────────────
def get_val_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


VAL_TRANSFORM = get_val_transform()


# ── Model loading ──────────────────────────────────────────────────────────
fundus_model: Optional[nn.Module] = None
oct_model:    Optional[nn.Module] = None
models_loaded = False
load_error: Optional[str] = None


def load_models():
    global fundus_model, oct_model, models_loaded, load_error

    fundus_path = MODELS_DIR / "fundus_model_best.pth"
    oct_path    = MODELS_DIR / "oct_model_best.pth"

    missing = []
    if not fundus_path.exists():
        missing.append(str(fundus_path))
    if not oct_path.exists():
        missing.append(str(oct_path))

    if missing:
        load_error = f"Model files not found: {missing}. Please upload fundus_model_best.pth and oct_model_best.pth to the models/ directory."
        logger.warning(load_error)
        return

    try:
        # Fundus model
        fundus_model = build_resnet50_binary().to(DEVICE)
        ckpt = torch.load(fundus_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        fundus_model.load_state_dict(state)
        fundus_model.eval()
        logger.info("✅ Fundus model loaded")

        # OCT model
        oct_model = build_resnet50_binary().to(DEVICE)
        ckpt = torch.load(oct_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        oct_model.load_state_dict(state)
        oct_model.eval()
        logger.info("✅ OCT model loaded")

        models_loaded = True

    except Exception as e:
        load_error = f"Error loading models: {str(e)}"
        logger.error(load_error)


# ── Inference helpers ──────────────────────────────────────────────────────
def preprocess_fundus(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    tensor = VAL_TRANSFORM(image=img_np)["image"]
    return tensor.unsqueeze(0).to(DEVICE)


def preprocess_oct(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("L").convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    tensor = VAL_TRANSFORM(image=img_np)["image"]
    return tensor.unsqueeze(0).to(DEVICE)


def predict_single(model: nn.Module, tensor: torch.Tensor) -> float:
    with torch.no_grad():
        logit = model(tensor).squeeze()
        prob  = torch.sigmoid(logit).item()
    return prob


def interpret_risk(prob: float) -> dict:
    if prob >= 0.75:
        return {"level": "HIGH",   "label": "Glaucoma Suspected",   "color": "#e74c3c", "emoji": "🔴"}
    elif prob >= 0.5:
        return {"level": "MEDIUM", "label": "Borderline — Monitor", "color": "#f39c12", "emoji": "🟡"}
    elif prob >= 0.30:
        return {"level": "LOW",    "label": "Low Risk",              "color": "#2ecc71", "emoji": "🟢"}
    else:
        return {"level": "NORMAL", "label": "Normal",                "color": "#27ae60", "emoji": "🟢"}


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="GlaucoVision API",
    description="Multimodal glaucoma detection from Fundus + OCT images",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting GlaucoVision API...")
    load_models()
    if models_loaded:
        logger.info("✅ Models ready — API is live")
    else:
        logger.warning(f"⚠️  Demo mode — {load_error}")


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path("static/index.html")
    if not html_path.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "device": str(DEVICE),
        "demo_mode": not models_loaded,
        "error": load_error,
    }


@app.post("/predict")
async def predict(
    fundus: UploadFile = File(..., description="Fundus (color) image — JPG/PNG"),
    oct:    UploadFile = File(..., description="OCT B-scan image — JPG/PNG"),
):
    # Validate file types
    for f in [fundus, oct]:
        if f.content_type not in ("image/jpeg", "image/png", "image/jpg"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid file type '{f.content_type}'. Only JPEG/PNG accepted."
            )

    fundus_bytes = await fundus.read()
    oct_bytes    = await oct.read()

    # Demo mode — return simulated result if models not loaded
    if not models_loaded:
        import random
        random.seed(len(fundus_bytes) % 100)
        p_f = round(random.uniform(0.25, 0.75), 4)
        p_o = round(random.uniform(0.25, 0.75), 4)
        p_fusion = round(BEST_ALPHA * p_f + (1 - BEST_ALPHA) * p_o, 4)
        risk = interpret_risk(p_fusion)
        return JSONResponse({
            "demo_mode":            True,
            "fundus_probability":   p_f,
            "oct_probability":      p_o,
            "fusion_probability":   p_fusion,
            "prediction":           "Glaucoma" if p_fusion >= THRESHOLD else "Normal",
            "glaucoma_detected":    p_fusion >= THRESHOLD,
            "risk":                 risk,
            "alpha":                BEST_ALPHA,
            "threshold":            THRESHOLD,
            "note":                 "DEMO MODE — Upload model weights to models/ for real inference",
        })

    try:
        # Real inference
        fundus_tensor = preprocess_fundus(fundus_bytes)
        oct_tensor    = preprocess_oct(oct_bytes)

        p_fundus = predict_single(fundus_model, fundus_tensor)
        p_oct    = predict_single(oct_model,    oct_tensor)

        # Weighted average fusion (best method from notebook 05)
        p_fusion = BEST_ALPHA * p_fundus + (1 - BEST_ALPHA) * p_oct
        risk     = interpret_risk(p_fusion)

        return JSONResponse({
            "demo_mode":            False,
            "fundus_probability":   round(p_fundus, 4),
            "oct_probability":      round(p_oct,    4),
            "fusion_probability":   round(p_fusion, 4),
            "prediction":           "Glaucoma" if p_fusion >= THRESHOLD else "Normal",
            "glaucoma_detected":    p_fusion >= THRESHOLD,
            "risk":                 risk,
            "alpha":                BEST_ALPHA,
            "threshold":            THRESHOLD,
        })

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/info")
async def model_info():
    return {
        "architecture":     "ResNet50 (ImageNet pretrained)",
        "fusion_method":    "Weighted average",
        "alpha":            BEST_ALPHA,
        "threshold":        THRESHOLD,
        "input_size":       f"{IMG_SIZE}x{IMG_SIZE}",
        "fundus_augment":   "CLAHE, flip, affine, brightness, HueSaturation",
        "oct_augment":      "CLAHE (stronger), flip, affine (±10°), GaussNoise",
        "training":         "StratifiedKFold K=5, BCEWithLogitsLoss, AdamW",
        "dataset":          "48 Fundus+OCT pairs, 20 glaucoma / 28 normal",
        "fundus_auc_mean":  0.6950,
        "oct_auc_mean":     0.8783,
        "fusion_auc":       0.8018,
    }
