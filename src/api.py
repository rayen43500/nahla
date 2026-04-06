"""
FastAPI REST API - IoT Network Intrusion Detection System
(Version corrigée avec preprocessing)
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
from fastapi.responses import StreamingResponse
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("IoT_IDS_API")

_state: Dict[str, Any] = {}

# ============================================================================
# SCHEMAS
# ============================================================================

class FlowFeatures(BaseModel):
    features: List[float]


class BatchFlowFeatures(BaseModel):
    flows: List[List[float]]


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_inference_time_ms: float
    num_flows: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: Optional[str]
    device: str


class ModelInfoResponse(BaseModel):
    model_type: str
    input_dim: int
    num_classes: int
    classes: List[str]
    device: str


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(model_path: pathlib.Path = None) -> None:
    if model_path is None:
        candidates = ["mlp_best.pt", "lstm_best.pt", "cnn_best.pt", "hybrid_best.pt"]
        for name in candidates:
            p = MODEL_DIR / name
            if p.exists() and p.stat().st_size > 10:
                model_path = p
                break

    if model_path is None or not model_path.exists():
        logger.warning("No trained model found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model = create_model(
            checkpoint["model_type"],
            checkpoint["input_dim"],
            checkpoint["num_classes"],
            **checkpoint.get("model_kwargs", {})
        ).to(device)

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        _state.update({
            "model": model,
            "model_type": checkpoint["model_type"],
            "input_dim": checkpoint["input_dim"],
            "num_classes": checkpoint["num_classes"],
            "classes": [str(c) for c in checkpoint["classes"]],
            "device": device,
        })

        # Try the common preprocessing artifact locations.
        preprocessor_candidates = [
            DATA_DIR / "preprocessed" / "preprocessor.joblib",
            DATA_DIR / "preprocessor.joblib",
        ]
        loaded_preprocessor = None
        for preprocessor_path in preprocessor_candidates:
            if preprocessor_path.exists():
                loaded_preprocessor = joblib.load(preprocessor_path)
                _state["preprocessor"] = loaded_preprocessor
                logger.info(f"Preprocessor loaded from {preprocessor_path} ✔️")
                break

        if loaded_preprocessor is None:
            logger.warning("No preprocessor found ⚠️")

        logger.info(f"Model loaded: {_state['model_type']} (device={device})")

    except Exception as e:
        logger.error(f"Model loading failed: {e}")


# ============================================================================
# APP
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    _state.clear()


app = FastAPI(
    title="IoT IDS API",
    version="1.0",
    lifespan=lifespan,
)


def _get_model():
    if "model" not in _state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _state["model"]


# ============================================================================
# CORE PREDICTION (FIXED)
# ============================================================================

def _preprocess(features):
    """Apply preprocessing if available"""
    if "preprocessor" in _state:
        return _state["preprocessor"].transform([features])[0]
    else:
        logger.warning("No preprocessing applied ⚠️")
        return features


def _predict_single(features: List[float]) -> PredictionResponse:
    model = _get_model()
    device = _state["device"]
    classes = _state["classes"]
    input_dim = _state["input_dim"]

    if len(features) != input_dim:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {input_dim} features, got {len(features)}"
        )

    # 🔥 APPLY PREPROCESSING
    features = _preprocess(features)

    x = torch.tensor([features], dtype=torch.float32).to(device)

    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    inference_ms = (time.time() - t0) * 1000

    pred_idx = int(np.argmax(probs))

    return PredictionResponse(
        predicted_class=classes[pred_idx],
        predicted_index=pred_idx,
        confidence=float(probs[pred_idx]),
        probabilities={cls: float(p) for cls, p in zip(classes, probs)},
        inference_time_ms=inference_ms,
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        model_loaded="model" in _state,
        model_type=_state.get("model_type"),
        device=str(_state.get("device", "cpu")),
    )


@app.get("/model_info", response_model=ModelInfoResponse)
async def model_info():
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
    return _predict_single(flow.features)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(batch: BatchFlowFeatures):

    if not batch.flows:
        raise HTTPException(status_code=422, detail="Empty batch")

    flows = batch.flows
    input_dim = _state["input_dim"]

    for i, f in enumerate(flows):
        if len(f) != input_dim:
            raise HTTPException(
                status_code=422,
                detail=f"Flow {i}: expected {input_dim}, got {len(f)}"
            )

    # 🔥 APPLY PREPROCESSING
    if "preprocessor" in _state:
        flows = _state["preprocessor"].transform(flows)

    X = torch.tensor(flows, dtype=torch.float32).to(_state["device"])

    t0 = time.time()
    with torch.no_grad():
        logits = _state["model"](X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    total_ms = (time.time() - t0) * 1000

    results = []
    for i in range(len(flows)):
        idx = int(np.argmax(probs[i]))
        results.append(PredictionResponse(
            predicted_class=_state["classes"][idx],
            predicted_index=idx,
            confidence=float(probs[i][idx]),
            probabilities={cls: float(p) for cls, p in zip(_state["classes"], probs[i])},
            inference_time_ms=total_ms / len(flows),
        ))

    return BatchPredictionResponse(
        predictions=results,
        total_inference_time_ms=total_ms,
        num_flows=len(flows),
    )


@app.get("/stream")
async def stream():
    import asyncio

    async def generator():
        while True:
            await asyncio.sleep(10)
            yield f"data: {{\"status\": \"alive\", \"time\": {time.time()}}}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)