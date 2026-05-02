"""
Pydantic schemas for the food recognition API.
"""

from pydantic import BaseModel, Field


class NutritionInfo(BaseModel):
    calories: float | None = None
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    fiber_g: float | None = None
    serving_description: str | None = None


class FoodCandidate(BaseModel):
    class_name: str
    display_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    nutrition: NutritionInfo | None = None


class PredictionResponse(BaseModel):
    top_prediction: FoodCandidate
    alternatives: list[FoodCandidate]
    ood_detected: bool
    ood_message: str | None = None
    model_version: str = "resnet50-stagec-phase2"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    onnx_available: bool
    device: str


class ClassesResponse(BaseModel):
    total_classes: int
    classes: list[str]
