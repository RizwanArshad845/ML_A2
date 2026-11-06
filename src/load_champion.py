"""Task 3.6: load the model by its registry alias and classify one image.

    python src/load_champion.py path/to/image.jpg
    python src/load_champion.py path/to/image.jpg --export models/champion

--export also writes the model and its preprocessing settings to a local folder.
That is how the champion reaches the server (Task 6): it is downloaded ONCE at
deploy time, and the Flask app loads the local file. The app never calls DagsHub
per request.
"""
import argparse
import json
import shutil
from pathlib import Path

import mlflow
import tensorflow as tf

import tracking
from preprocess import decode_and_resize_bytes

MODEL_NAME = "visionguard"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--name", default=MODEL_NAME)
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--export", help="folder to save model.keras, config.json, threshold.json")
    args = ap.parse_args()

    tracking.setup()
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(args.name, args.alias)
    uri = f"models:/{args.name}@{args.alias}"
    print(f"Loading {uri} -> version {mv.version}, run {mv.run_id}")
    model = mlflow.keras.load_model(uri)

    # Preprocessing settings and the validation-chosen threshold were saved with the run.
    files = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="run_files"))
    cfg = json.loads((files / "config.json").read_text())
    threshold = json.loads((files / "threshold.json").read_text())["threshold"]

    x = decode_and_resize_bytes(tf.constant(Path(args.image).read_bytes()),
                                cfg["img_size"], cfg["resize_mode"])[None, ...]
    prob_ai = float(model(x, training=False)[0, 0])
    label = "AI-generated" if prob_ai >= threshold else "Real"
    conf = prob_ai if label == "AI-generated" else 1 - prob_ai
    print(f"{Path(args.image).name}: {label} (confidence {conf:.1%}, P(AI)={prob_ai:.3f}, "
          f"threshold {threshold:.2f})")

    if args.export:
        out = Path(args.export)
        out.mkdir(parents=True, exist_ok=True)
        model.save(out / "model.keras")
        shutil.copy(files / "config.json", out / "config.json")
        shutil.copy(files / "threshold.json", out / "threshold.json")
        (out / "model_info.json").write_text(json.dumps(
            {"name": args.name, "alias": args.alias, "version": mv.version,
             "run_id": mv.run_id}, indent=2))
        print(f"Exported champion to {out}")


if __name__ == "__main__":
    main()
