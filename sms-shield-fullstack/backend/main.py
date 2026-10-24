from pathlib import Path
import os, re
from typing import List
import joblib
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
# Put model files in backend/artifacts/... or set MODEL_BASE_DIR to your project root.
MODEL_BASE_DIR = Path(os.getenv("MODEL_BASE_DIR", str(ROOT)))
MODEL_PATHS = {
    "normalized_baseline": MODEL_BASE_DIR / "artifacts" / "obfuscation" / "word_char_normalized_model.joblib",
    "paraphrase_augmented": MODEL_BASE_DIR / "artifacts" / "paraphrase_augmentation" / "word_char_normalized_paraphrase_augmented_model.joblib",
}

app = FastAPI(title="SMS Shield API", version="1.0.0", description="SMS spam classification research demo")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
_models = {}
_errors = {}

class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    models: List[str] = Field(default_factory=lambda: list(MODEL_PATHS.keys()))

class SingleRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)


def normalize_obfuscated_text(text: str) -> str:
    s = str(text).lower()
    pattern = re.compile(r"(?<!\S)(?:[a-z0-9]\s+){3,}[a-z0-9](?!\S)", re.I)
    return pattern.sub(lambda m: re.sub(r"\s+", "", m.group(0)), s)


def get_model(name: str):
    if name not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail=f"Unknown model: {name}")
    if name not in _models and name not in _errors:
        path = MODEL_PATHS[name]
        try:
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            bundle = joblib.load(path)
            required = {"word_vectorizer", "char_vectorizer", "classifier"}
            if not isinstance(bundle, dict) or not required.issubset(bundle.keys()):
                raise ValueError("Unexpected model bundle format")
            _models[name] = bundle
        except Exception as exc:
            _errors[name] = str(exc)
    if name in _errors:
        raise HTTPException(status_code=503, detail=f"Model '{name}' unavailable: {_errors[name]}")
    return _models[name]


def predict(bundle, text: str):
    processed = normalize_obfuscated_text(text) if bundle.get("normalization") else text
    word = bundle["word_vectorizer"].transform([processed])
    char = bundle["char_vectorizer"].transform([processed])
    features = hstack([word, char], format="csr")
    label = str(bundle["classifier"].predict(features)[0]).lower()
    score = None
    try:
        score = float(bundle["classifier"].decision_function(features)[0])
    except Exception:
        pass
    return {"label": label, "decision_score": score, "score_note": "SVM decision margin, not a probability." if score is not None else None}

@app.get("/")
def root():
    return {"name": "SMS Shield API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    status = {}
    for name, path in MODEL_PATHS.items():
        status[name] = {"available": path.exists(), "path": str(path)}
    return {"status": "ok", "models": status}

@app.get("/api/models")
def models():
    return {"models": [
        {"id": "normalized_baseline", "name": "Normalized baseline", "description": "Word + character TF-IDF with obfuscation normalization."},
        {"id": "paraphrase_augmented", "name": "Paraphrase-augmented", "description": "Normalized word + character model trained with reviewed paraphrases."},
    ]}

@app.post("/api/predict")
def predict_many(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter an SMS message.")
    results = []
    for name in request.models:
        bundle = get_model(name)
        results.append({"model": name, **predict(bundle, text)})
    return {"text": text, "normalized_text": normalize_obfuscated_text(text), "results": results}

@app.post("/api/robustness")
def robustness(request: SingleRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter an SMS message.")
    results = []
    for name in MODEL_PATHS:
        bundle = get_model(name)
        results.append({"model": name, **predict(bundle, text)})
    return {"text": text, "normalized_text": normalize_obfuscated_text(text), "results": results}
