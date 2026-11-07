"""Task 7: TensorFlow-free inference for the Vercel model API.

Vercel cannot fit TensorFlow, so the cloud API runs the champion model as TensorFlow
Lite and preprocesses with Pillow + NumPy. The preprocessing reproduces
src/preprocess.py (tf.image.resize_with_pad with antialias=True) step by step:

  1. decode to RGB (alpha dropped, greyscale/palette expanded, like channels=3)
  2. scale so the longer side fits img_size, keeping the aspect ratio
     (same floor rounding as TensorFlow), antialiased bilinear resize in float32
  3. pad with black to img_size x img_size, image centred (same offsets as TensorFlow)
  4. values stay in 0..255; the model rescales internally

deploy/check_tflite.py compares this against the Keras + TensorFlow pipeline on every
validation and test image before deploying.
"""
import json
import math
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from app.inference import AI_LABEL, REAL_LABEL, Prediction  # no TensorFlow import at module level


def _resize_weights(in_size: int, out_size: int) -> np.ndarray:
    """(out_size, in_size) weight matrix of TensorFlow's antialiased bilinear resize.

    Mirrors ScaleAndTranslate with the triangle kernel (what tf.image.resize uses for
    bilinear + antialias=True): each output pixel is a weighted average of the input
    pixels under a triangle that is widened by the shrink factor, with half-pixel centres.
    """
    inv_scale = in_size / out_size
    kernel_scale = max(inv_scale, 1.0)  # antialias: widen the kernel when shrinking
    weights = np.zeros((out_size, in_size), dtype=np.float64)
    for x in range(out_size):
        centre = (x + 0.5) * inv_scale
        start = max(0, math.ceil(centre - kernel_scale - 0.5))
        end = min(in_size - 1, math.floor(centre + kernel_scale - 0.5))
        src = np.arange(start, end + 1)
        w = np.maximum(0.0, 1.0 - np.abs((src + 0.5 - centre) / kernel_scale))
        if w.sum() > 0:
            weights[x, src] = w / w.sum()
    return weights


def preprocess(raw: bytes, img_size: int) -> np.ndarray:
    with Image.open(BytesIO(raw)) as im:
        im.load()
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)  # first frame only, like expand_animations=False
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64)
    h, w = rgb.shape[:2]
    ratio = max(w / img_size, h / img_size)
    new_w, new_h = max(1, math.floor(w / ratio)), max(1, math.floor(h / ratio))
    # Separable resize: rows then columns, in floating point (no rounding to integers).
    wh, ww = _resize_weights(h, new_h), _resize_weights(w, new_w)
    resized = np.einsum("oh,hwc->owc", wh, rgb)
    resized = np.einsum("pw,owc->opc", ww, resized).astype(np.float32)
    out = np.zeros((img_size, img_size, 3), dtype=np.float32)
    # TensorFlow centres using the unrounded size: int((160 - 106.63) / 2) = 26, not 27.
    top, left = max(0, int((img_size - h / ratio) / 2)), max(0, int((img_size - w / ratio) / 2))
    out[top:top + new_h, left:left + new_w] = resized
    return np.clip(out, 0.0, 255.0)


def _interpreter_class():
    try:  # LiteRT: the small runtime used in the cloud (Linux wheels)
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # local Windows check: TensorFlow ships the same interpreter
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    return Interpreter


class LitePredictor:
    """Loads model.tflite once (cold start) and predicts from raw image bytes."""

    def __init__(self, model_dir, threshold: float | None = None, model_name: str = "visionguard",
                 model_version: str = ""):
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text())
        self.img_size = int(cfg.get("img_size", 224))
        if threshold is None:
            threshold = json.loads((model_dir / "threshold.json").read_text())["threshold"]
        self.threshold = float(threshold)
        info_file = model_dir / "model_info.json"
        info = json.loads(info_file.read_text()) if info_file.exists() else {}
        self.model_name = model_name
        self.model_version = model_version or str(info.get("version", "local"))

        self.interpreter = _interpreter_class()(model_path=str(model_dir / "model.tflite"))
        self.interpreter.allocate_tensors()
        self._in = self.interpreter.get_input_details()[0]["index"]
        self._out = self.interpreter.get_output_details()[0]["index"]

    def prob_ai(self, raw: bytes) -> float:
        x = preprocess(raw, self.img_size)[None, ...]
        self.interpreter.set_tensor(self._in, x)
        self.interpreter.invoke()
        return float(self.interpreter.get_tensor(self._out)[0, 0])

    def predict(self, raw: bytes) -> Prediction:
        start = time.perf_counter()
        try:
            p = self.prob_ai(raw)
        except OSError as e:
            raise ValueError("image could not be decoded") from e
        is_ai = p >= self.threshold
        return Prediction(label=AI_LABEL if is_ai else REAL_LABEL, prob_ai=p,
                          confidence=p if is_ai else 1.0 - p, threshold=self.threshold,
                          model_name=self.model_name, model_version=self.model_version,
                          latency_ms=(time.perf_counter() - start) * 1000)
