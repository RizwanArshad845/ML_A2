"""Task 4.1: inference module.

The Keras model is loaded ONCE when the Predictor is created (at app start-up) and
kept in memory. Every request then only runs preprocessing plus one forward pass.

Preprocessing comes from src/preprocess.py, the same function the training pipeline
uses, and the image size / resize mode are read from the config.json that train.py
saved next to the model, so serving always matches training.

Console check for two known images:
    python -m app.inference path/to/real.jpg path/to/ai.png
"""
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

AI_LABEL = "AI-generated"
REAL_LABEL = "Real"


@dataclass
class Prediction:
    label: str          # AI_LABEL or REAL_LABEL
    prob_ai: float      # model output: probability the image is AI-generated
    confidence: float   # probability of the label that was chosen
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float


class Predictor:
    def __init__(self, model_path, threshold=None, model_name="visionguard",
                 model_version=""):
        model_path = Path(model_path)
        if not model_path.is_file():
            # Fail at start-up with a clear message instead of on the first request.
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Set MODEL_PATH to a trained .keras file.")

        import tensorflow as tf  # imported here so tests with a fake predictor stay fast
        from tensorflow import keras

        from src.preprocess import decode_and_resize_bytes

        self._tf = tf
        self._preprocess = decode_and_resize_bytes

        cfg_file = model_path.with_name("config.json")
        cfg = json.loads(cfg_file.read_text()) if cfg_file.exists() else {}
        self.img_size = int(cfg.get("img_size", 224))
        self.resize_mode = cfg.get("resize_mode", "pad")

        if threshold is None:
            thr_file = model_path.with_name("threshold.json")
            threshold = json.loads(thr_file.read_text())["threshold"] if thr_file.exists() else 0.5
        self.threshold = float(threshold)
        self.model_name = model_name
        # model_info.json is written by src/load_champion.py --export (registry version).
        info_file = model_path.with_name("model_info.json")
        info = json.loads(info_file.read_text()) if info_file.exists() else {}
        self.model_version = model_version or str(info.get("version", "local"))

        start = time.perf_counter()
        self.model = keras.models.load_model(model_path)
        # One dummy pass so the first real request is not slowed by graph tracing.
        self.model(tf.zeros((1, self.img_size, self.img_size, 3)), training=False)
        log.info("Loaded model %s in %.1fs (img_size=%d, resize=%s, threshold=%.2f)",
                 model_path, time.perf_counter() - start, self.img_size, self.resize_mode,
                 self.threshold)

    def predict(self, raw: bytes) -> Prediction:
        start = time.perf_counter()
        try:
            img = self._preprocess(self._tf.constant(raw), self.img_size, self.resize_mode)
        except self._tf.errors.InvalidArgumentError as e:
            raise ValueError("image could not be decoded") from e
        prob_ai = float(self.model(img[None, ...], training=False)[0, 0])
        is_ai = prob_ai >= self.threshold
        return Prediction(
            label=AI_LABEL if is_ai else REAL_LABEL,
            prob_ai=prob_ai,
            confidence=prob_ai if is_ai else 1.0 - prob_ai,
            threshold=self.threshold,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=(time.perf_counter() - start) * 1000,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from app.config import Settings

    s = Settings.from_env()
    predictor = Predictor(s.model_path, s.threshold, s.model_name, s.model_version)
    for path in sys.argv[1:]:
        p = predictor.predict(Path(path).read_bytes())
        print(f"{Path(path).name}: {p.label} (confidence {p.confidence:.1%}, "
              f"P(AI)={p.prob_ai:.3f}, threshold {p.threshold:.2f}, {p.latency_ms:.0f} ms)")


if __name__ == "__main__":
    main()
