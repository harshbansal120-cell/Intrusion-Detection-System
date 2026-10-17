"""
AE-XLSTM-CNN Intrusion Detection — inference backend
-----------------------------------------------------
Serves:
  GET  /                -> the demo interface (static/ids_demo.html)
  POST /predict          -> runs the model on one of the 6 built-in demo samples
  POST /predict_csv      -> runs the model on a REAL uploaded flow-data CSV
  GET  /health           -> tells you exactly what is / isn't wired up yet

HOW TO USE THIS FILE
--------------------
Everything works right now in SIMULATION mode — nothing is broken while the
model is still training. Search this file for the banners that say
">>> STEP" — those are the only places you need to touch once your friend's
weights are ready. Nothing else needs to change.
"""

import io
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "model.pt"
SCALER_PATH = BASE_DIR / "artifacts" / "scaler.pkl"
CLASSES_PATH = BASE_DIR / "artifacts" / "label_classes.json"
FEATURES_PATH = BASE_DIR / "artifacts" / "features_list.json"
FIXTURES_DIR = BASE_DIR / "fixtures"

SEQ_LEN = 10   # rows per window — must match what the model was trained on
N_FEATURES = 39   # confirmed from Merged58.csv (39 feature columns + Label) —
                  # the report's "47" was a rounded/approximate figure

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_REGISTRY = {
    # true_label values are the EXACT strings found in Merged58.csv's Label
    # column — must match whatever your friend's LabelEncoder produces.
    "benign": {"fixture": "benign.json",  "true_label": "BENIGN"},
    "ddos":   {"fixture": "ddos.json",    "true_label": "DDOS-UDP_FLOOD"},
    "dos":    {"fixture": "dos.json",     "true_label": "DOS-SYN_FLOOD"},
    "mirai":  {"fixture": "mirai.json",   "true_label": "MIRAI-GREIP_FLOOD"},
    "recon":  {"fixture": "recon.json",   "true_label": "RECON-PORTSCAN"},
    "mitm":   {"fixture": "mitm.json",    "true_label": "MITM-ARPSPOOFING"},
}

# =============================================================================
# >>> STEP 1 — PASTE THE REAL MODEL ARCHITECTURE HERE <<<
# Ask your friend for the exact nn.Module class from the Colab notebook and
# replace the body below with it (keep the class name AEXLSTMCNN, or rename
# consistently in the two spots that reference it further down).
# =============================================================================
class AEXLSTMCNN(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, seq_len: int = SEQ_LEN, n_classes: int = 34):
        super().__init__()
        # PLACEHOLDER layers — replace with the real encoder / xlstm / cnn / head
        self.encoder = nn.Sequential(nn.Linear(n_features, 128), nn.ReLU(), nn.Linear(128, 64))
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.cnn = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.head = nn.Linear(32, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        z, _ = self.lstm(z)
        z = self.cnn(z.transpose(1, 2))
        z = z.mean(dim=-1)
        return self.head(z)


# =============================================================================
# >>> STEP 2 — DROP THESE 4 FILES IN, THEN NOTHING ELSE CHANGES <<<
#   backend/model/model.pt                 <- trained weights from Colab
#   backend/artifacts/scaler.pkl           <- fitted StandardScaler (joblib)
#   backend/artifacts/label_classes.json   <- ["BENIGN", "DDoS-...", ...] in
#                                              model output order
#   backend/artifacts/features_list.json   <- ["feat_col_1", "feat_col_2", ...]
#                                              the 47 column names, IN ORDER,
#                                              that the model was trained on
# See HANDOFF_CHECKLIST.md for exactly what to ask your friend for.
# =============================================================================
LOAD_MODE = "state_dict"  # "state_dict" if saved via model.state_dict(), or
                           # "full" if saved via torch.save(model, ...) directly


def load_model():
    if not MODEL_PATH.exists():
        return None
    if LOAD_MODE == "full":
        model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    else:
        model = AEXLSTMCNN()
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def load_scaler():
    return joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None


def load_json_list(path):
    return json.loads(path.read_text()) if path.exists() else None


model = load_model()
scaler = load_scaler()
class_names = load_json_list(CLASSES_PATH)
feature_names = load_json_list(FEATURES_PATH)

app = FastAPI(title="AE-XLSTM-CNN IDS backend")

# Frontend (Vercel) and backend (Hugging Face) are on different domains now,
# so the browser needs explicit permission to call across origins.
# For the hackathon, "*" is simplest. Once you know your Vercel URL, you can
# tighten this to allow_origins=["https://your-project.vercel.app"].
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_inference(window: np.ndarray):
    """window: (SEQ_LEN, N_FEATURES) raw feature values -> prediction dict."""
    if scaler is not None:
        window = scaler.transform(window)
    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    top_idx = int(np.argmax(probs))
    predicted_label = class_names[top_idx] if class_names else f"class_{top_idx}"
    return predicted_label, float(probs[top_idx]), latency_ms, probs


class PredictRequest(BaseModel):
    sample: str


@app.get("/health")
def health():
    return {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "classes_loaded": class_names is not None,
        "features_loaded": feature_names is not None,
        "mode": "LIVE MODEL" if model is not None else "SIMULATION (backend has no weights yet)",
        "device": str(DEVICE),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    """Runs the model on one of the 6 built-in demo samples."""
    if req.sample not in SAMPLE_REGISTRY:
        raise HTTPException(400, f"Unknown sample '{req.sample}'")
    if model is None:
        raise HTTPException(503, "Model not loaded yet — drop model.pt into backend/model/")

    entry = SAMPLE_REGISTRY[req.sample]
    fixture_path = FIXTURES_DIR / entry["fixture"]
    if not fixture_path.exists():
        raise HTTPException(503, f"Missing fixture — run export_helpers.py to build {entry['fixture']}")

    window = np.array(json.loads(fixture_path.read_text()), dtype=np.float32)
    predicted_label, confidence, latency_ms, probs = run_inference(window)

    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "true_label": entry["true_label"],
        "is_correct": predicted_label == entry["true_label"],
        "latency_ms": latency_ms,
        "top5": sorted(
            [{"label": (class_names[i] if class_names else f"class_{i}"), "prob": float(p)} for i, p in enumerate(probs)],
            key=lambda d: -d["prob"],
        )[:5],
    }


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), start_row: int = 0):
    """
    Runs the model on a REAL uploaded flow-data CSV — this is the
    "attach real network traffic" path. The CSV must have the same column
    names the model was trained on (e.g. exported straight from
    CICFlowMeter, or a slice of the original CIC-IoT2023 files).
    It takes SEQ_LEN consecutive rows starting at start_row as one window.
    """
    if model is None:
        raise HTTPException(503, "Model not loaded yet — drop model.pt into backend/model/")
    if feature_names is None:
        raise HTTPException(503, "features_list.json missing — see HANDOFF_CHECKLIST.md")

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Uploaded CSV is missing expected columns: {missing[:5]}...")

    if start_row + SEQ_LEN > len(df):
        raise HTTPException(400, f"CSV needs at least {start_row + SEQ_LEN} rows from start_row={start_row}, only has {len(df)}")

    window_df = df.iloc[start_row:start_row + SEQ_LEN][feature_names]
    window = window_df.to_numpy(dtype=np.float32)

    predicted_label, confidence, latency_ms, probs = run_inference(window)

    return {
        "predicted_class": predicted_label,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "rows_used": f"{start_row}-{start_row + SEQ_LEN - 1}",
        "total_rows_in_file": len(df),
        "top5": sorted(
            [{"label": (class_names[i] if class_names else f"class_{i}"), "prob": float(p)} for i, p in enumerate(probs)],
            key=lambda d: -d["prob"],
        )[:5],
    }


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def root():
    return FileResponse(BASE_DIR / "static" / "ids_demo.html")
