# API FastAPI pour prédiction

from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

# @app.post("/predict")
# def predict(...):
#     ...

# @app.post("/predict_batch")
# def predict_batch(...):
#     ...
