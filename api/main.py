"""
FastAPI application for food recognition.

Endpoints:
  POST /predict   — Upload an image, get top-5 food predictions + nutrition
  POST /feedback  — Submit a correction (predicted vs correct class + optional image)
  GET  /health    — Liveness/readiness check
  GET  /classes   — List all 101 supported food classes

Environment variables (set in Railway or .env):
  FATSECRET_CLIENT_ID      — FatSecret OAuth client ID
  FATSECRET_CLIENT_SECRET  — FatSecret OAuth client secret
  PORT                     — Railway injects this automatically (default 8000)
"""

import io
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api.schemas import (
    PredictionResponse, FoodCandidate, NutritionInfo,
    HealthResponse, ClassesResponse,
)
from api.predictor import get_predictor
from api.ood_detector import detect_ood
from api.usda_client import get_usda_client
from api.feedback_store import save_feedback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 10
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the model on startup so the first request isn't slow
    logger.info("Loading predictor model...")
    predictor = get_predictor()
    logger.info(f"Model ready (ONNX={predictor.is_onnx})")
    yield


app = FastAPI(
    title="Food Recognition API",
    description="ResNet-50 trained on Food-101 — Phase 2, Stage C",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _display_name(class_name: str) -> str:
    """'apple_pie' → 'Apple Pie'"""
    return class_name.replace("_", " ").title()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG, PNG, or WebP.",
        )

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE_MB * 1_000_000:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_FILE_SIZE_MB} MB).",
        )

    try:
        image = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read image file.")

    predictor = get_predictor()
    results = predictor.predict(image)  # list of (class_name, confidence)

    top_class, top_conf = results[0]
    is_ood, ood_msg = detect_ood(top_class, top_conf)

    usda = get_usda_client()

    # Fetch nutrition for top prediction only (avoid 5 API calls per request)
    nutrition = await usda.get_nutrition_for_class(top_class)

    top = FoodCandidate(
        class_name=top_class,
        display_name=_display_name(top_class),
        confidence=top_conf,
        nutrition=nutrition,
    )

    alternatives = [
        FoodCandidate(
            class_name=cls,
            display_name=_display_name(cls),
            confidence=conf,
        )
        for cls, conf in results[1:]
    ]

    return PredictionResponse(
        top_prediction=top,
        alternatives=alternatives,
        ood_detected=is_ood,
        ood_message=ood_msg,
    )


@app.post("/feedback", status_code=204)
async def feedback(
    predicted_class: str = Form(...),
    correct_class: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    image_bytes = await file.read() if file else None
    save_feedback(predicted_class, correct_class, image_bytes)


@app.get("/health", response_model=HealthResponse)
async def health():
    predictor = get_predictor()
    return HealthResponse(
        status="ok",
        model_loaded=True,
        onnx_available=predictor.is_onnx,
        device=predictor.device if not predictor.is_onnx else "cpu (onnx)",
    )


@app.get("/classes", response_model=ClassesResponse)
async def classes():
    predictor = get_predictor()
    names = predictor.class_names
    return ClassesResponse(total_classes=len(names), classes=names)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)
