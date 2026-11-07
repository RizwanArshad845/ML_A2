"""Task 7: convert the champion Keras model to TensorFlow Lite for Vercel.

Vercel Python functions have a bundle size limit, and full TensorFlow (~1 GB) does not fit.
TFLite keeps the same network and weights in a compact file that the small LiteRT
runtime (ai-edge-litert, ~11 MB) can run without TensorFlow.

    python deploy/export_tflite.py          # writes models/champion/model.tflite
"""
from pathlib import Path

import tensorflow as tf

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "champion"


def main() -> None:
    model = tf.keras.models.load_model(MODEL_DIR / "model.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Float32 weights, no quantization: keeps predictions identical to the Keras model.
    tflite = converter.convert()
    out = MODEL_DIR / "model.tflite"
    out.write_bytes(tflite)
    keras_mb = (MODEL_DIR / "model.keras").stat().st_size / 1e6
    print(f"model.keras {keras_mb:.1f} MB -> model.tflite {len(tflite) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
