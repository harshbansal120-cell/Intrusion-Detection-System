"""
features_list.json, label_classes.json, and the 6 fixtures/*.json are
already built from your local Merged58.csv — see HANDOFF_CHECKLIST.md.

Only 2 things are still needed from your friend's Colab notebook. Have
them run this (with real variable names swapped in) at the end of training,
or send you the notebook so you can run it.
"""

from pathlib import Path

import joblib

OUT = Path("backend")

# ---------------------------------------------------------------------------
# 1) Save the trained model weights
# ---------------------------------------------------------------------------
# torch.save(model.state_dict(), OUT / "model" / "model.pt")
# (or torch.save(model, OUT / "model" / "model.pt") for the whole object —
#  if you do this, set LOAD_MODE = "full" in app.py instead)

# ---------------------------------------------------------------------------
# 2) Save the fitted scaler used at train time
# ---------------------------------------------------------------------------
# joblib.dump(scaler, OUT / "artifacts" / "scaler.pkl")

# ---------------------------------------------------------------------------
# Also worth doing while you're in the notebook: print the label order and
# compare it to backend/artifacts/label_classes.json (see the note in
# HANDOFF_CHECKLIST.md about why this matters).
# ---------------------------------------------------------------------------
# print(list(label_encoder.classes_))

print("Only 2 files needed now — see the two commented blocks above.")
