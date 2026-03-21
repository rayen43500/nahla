"""
FastAPI REST API - IoT Network Intrusion Detection System

Phase 12:
- POST /predict - single flow prediction
- POST /predict_batch - batch prediction
- GET /health - health check
- GET /model_info - model metadata
- SSE /stream - streaming detection
- Structured logging, error handling
"""

import logging
import time
import pathlib
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import torch
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import create_model

# ============================================================================
# CONFIG
# ============================================================================

MODEL_DIR = pathlib.Path("models")
DATA_DIR = pathlib.Path("data")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("IoT_IDS_API")

# Global model state
_state: Dict[str, Any] = {}


# ============================================================================
# SCHEMAS
# ============================================================================

class FlowFeatures(BaseModel):
    """Single network flow feature vector."""
    features: List[float] = Field(..., description="Feature vector for a single network flow")


class BatchFlowFeatures(BaseModel):
    """Batch of network flow feature vectors."""
    flows: List[List[float]] = Field(..., description="List of feature vectors")


class PredictionResponse(BaseModel):
    """Prediction result for a single flow."""
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Prediction results for a batch of flows."""
    predictions: List[PredictionResponse]
    total_inference_time_ms: float
    num_flows: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str]
    device: str


class ModelInfoResponse(BaseModel):
    """Model information."""
    model_type: str
    input_dim: int
    num_classes: int
    classes: List[str]
    device: str


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path: pathlib.Path = None) -> None:
    """Load the best available model."""
    if model_path is None:
        # Auto-detect best model
        candidates = ["mlp_best.pt", "lstm_best.pt", "cnn_best.pt", "hybrid_best.pt"]
        for name in candidates:
            p = MODEL_DIR / name
            if p.exists() and p.stat().st_size > 10:
                model_path = p
                break

    if model_path is None or not model_path.exists():
        logger.warning("No trained model found. API will return errors until a model is loaded.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model_type = checkpoint.get("model_type", "mlp")
        input_dim = checkpoint["input_dim"]
        num_classes = checkpoint["num_classes"]
        classes = checkpoint["classes"]
        model_kwargs = checkpoint.get("model_kwargs", {})

        model = create_model(model_type, input_dim, num_classes, **model_kwargs).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        _state["model"] = model
        _state["model_type"] = model_type
        _state["input_dim"] = input_dim
        _state["num_classes"] = num_classes
        _state["classes"] = [str(c) for c in classes]
        _state["device"] = device

        # Load preprocessor if available
        preprocessor_path = DATA_DIR / "preprocessor.joblib"
        if preprocessor_path.exists():
            _state["preprocessor"] = joblib.load(preprocessor_path)

        logger.info(f"Model loaded: {model_type} from {model_path} (device={device})")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


# ============================================================================
# APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield
    _state.clear()


app = FastAPI(
    title="IoT Network Intrusion Detection API",
    description="Predicts network intrusion attacks using Deep Learning models",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_model():
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _state["model"]


def _predict_single(features: List[float]) -> PredictionResponse:
    """Run prediction on a single feature vector."""
    model = _get_model()
    device = _state["device"]
    classes = _state["classes"]
    input_dim = _state["input_dim"]

    if len(features) != input_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {input_dim} features, got {len(features)}"
        )

    x = torch.tensor([features], dtype=torch.float32).to(device)

    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    inference_ms = (time.time() - t0) * 1000

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    probabilities = {cls: float(p) for cls, p in zip(classes, probs)}

    return PredictionResponse(
        predicted_class=pred_class,
        predicted_index=pred_idx,
        confidence=confidence,
        probabilities=probabilities,
        inference_time_ms=inference_ms,
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded="model" in _state,
        model_type=_state.get("model_type"),
        device=str(_state.get("device", "cpu")),
    )


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
    """Get model metadata."""
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_type=_state["model_type"],
        input_dim=_state["input_dim"],
        num_classes=_state["num_classes"],
        classes=_state["classes"],
        device=str(_state["device"]),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(flow: FlowFeatures):
    """Predict intrusion for a single network flow."""
    logger.info(f"Prediction request: {len(flow.features)} features")
    result = _predict_single(flow.features)
    logger.info(f"Predicted: {result.predicted_class} (conf={result.confidence:.3f})")
    return result


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchFlowFeatures):
    """Predict intrusion for a batch of network flows."""
    model = _get_model()
    device = _state["device"]
    classes = _state["classes"]
    input_dim = _state["input_dim"]

    if not batch.flows:
        raise HTTPException(status_code=422, detail="Empty batch")

    for i, flow in enumerate(batch.flows):
        if len(flow) != input_dim:
            raise HTTPException(
                status_code=422,
                detail=f"Flow {i}: expected {input_dim} features, got {len(flow)}"
            )

    X = torch.tensor(batch.flows, dtype=torch.float32).to(device)

    t0 = time.time()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    total_ms = (time.time() - t0) * 1000

    predictions = []
    per_sample_ms = total_ms / len(batch.flows)
    for i in range(len(batch.flows)):
        pred_idx = int(np.argmax(probs[i]))
        predictions.append(PredictionResponse(
            predicted_class=classes[pred_idx],
            predicted_index=pred_idx,
            confidence=float(probs[i][pred_idx]),
            probabilities={cls: float(p) for cls, p in zip(classes, probs[i])},
            inference_time_ms=per_sample_ms,
        ))

    return BatchPredictionResponse(
        predictions=predictions,
        total_inference_time_ms=total_ms,
        num_flows=len(batch.flows),
    )


@app.get("/stream")
async def stream_predictions():
    """
    Server-Sent Events endpoint for real-time streaming detection.
    Connect via EventSource for continuous predictions.
    """
    import asyncio

    async def event_generator():
        yield "data: {\"status\": \"connected\", \"message\": \"Send flows to /predict for real-time detection\"}\n\n"
        # Keep connection alive
        while True:
            await asyncio.sleep(30)
            yield f"data: {{\"status\": \"heartbeat\", \"timestamp\": {time.time()}}}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
