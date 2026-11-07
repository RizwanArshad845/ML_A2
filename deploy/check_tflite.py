"""Task 7: prove the Vercel pipeline (Pillow/NumPy + TFLite) matches training/serving
(TensorFlow preprocessing + Keras) on every validation and test image.

    python deploy/check_tflite.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.lite import LitePredictor  # noqa: E402
from app.inference import Predictor  # noqa: E402

MODEL_DIR = ROOT / "models" / "champion"


def main() -> None:
    keras_p = Predictor(MODEL_DIR / "model.keras")
    lite_p = LitePredictor(MODEL_DIR)
    rows = []
    for split in ("val", "test"):
        df = pd.read_csv(ROOT / "data" / "splits" / f"{split}.csv")
        for path in df["path"]:
            raw = (ROOT / "data" / "raw" / "archive" / path).read_bytes()
            rows.append((split, keras_p.predict(raw).prob_ai, lite_p.prob_ai(raw)))
    res = pd.DataFrame(rows, columns=["split", "keras", "tflite"])
    diff = (res.keras - res.tflite).abs()
    thr = keras_p.threshold
    agree = ((res.keras >= thr) == (res.tflite >= thr)).mean()
    print(f"images compared      : {len(res)} (val + test)")
    print(f"mean |P(AI) diff|    : {diff.mean():.5f}")
    print(f"95th pct |diff|      : {np.percentile(diff, 95):.5f}")
    print(f"max |P(AI) diff|     : {diff.max():.5f}")
    print(f"same verdict (thr {thr:.2f}): {agree:.1%}")


if __name__ == "__main__":
    main()
