# AE-XLSTM-CNN - interface + backend

## Folder layout
```
backend/
  app.py                  -> FastAPI server (API + serves the frontend).
                             Search for ">>> STEP" to find the only 2 spots
                             you need to edit once training is done.
  export_helpers.py       -> one needs to run this in Colab to produce
                             the 5 handoff files
  requirements.txt
  model/model.pt                <- (empty until handoff)
  artifacts/scaler.pkl           <- (empty until handoff)
  artifacts/label_classes.json   <- (empty until handoff)
  artifacts/features_list.json   <- (empty until handoff)
  fixtures/*.json                <- (empty until handoff)
  static/ids_demo.html    -> the interface itself
```

## Run it right now (works in simulation mode, nothing needed yet)
```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Open `http://localhost:8000` - the 6 demo buttons work end-to-end
using the calibrated simulation. It'll silently switch to real predictions the moment the model files land.

## Two ways to run detection
1. **Built-in samples** (`/predict`) - six one-click buttons (Benign, DDoS,
   DoS, Mirai, Recon, MITM). Always available once the 5 handoff files are
   in place.
2. **Upload real traffic** (`/predict_csv`) - a file-upload panel where
   anyone can drop in a flow-data CSV (same column layout as CIC-IoT2023,
   e.g. a CICFlowMeter export) and pick a starting row; the model runs on
   those 10 real consecutive rows. This is the honest version of "attach a
   network" for a one-night build - true live packet-sniffing + feature
   extraction is a much larger project and isn't safe to wire up before
   judging (see note in HANDOFF_CHECKLIST.md).

## Getting a link judges can open
- **Fastest:** `ngrok http 8000` (or a Cloudflare Tunnel) from whichever
  laptop is running the server - gives a public URL instantly, no redeploy.
- **More permanent:** push `backend/` to a repo and deploy on Render,
  Railway, or a Hugging Face Space (Docker SDK), start command
  `uvicorn app:app --host 0.0.0.0 --port $PORT`.
Either way judges only need one URL - frontend and API share one process,
so there's no CORS setup.
