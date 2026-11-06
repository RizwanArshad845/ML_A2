"""Bonus: Grad-CAM heat maps for 4 correct and 4 wrong test predictions.

Run after evaluate.py (it needs test_predictions.csv):
    python src/gradcam.py --run_name mobilenet_v1

The heat map shows which image regions pushed the model towards the class it
predicted. Red means strong influence.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import ARTIFACT_DIR, DATA_ROOT
from data import load_image_array
from models import get_base


def conv_layer_name(model) -> str:
    base = get_base(model)
    return base.name if base is not None else "relu4"  # last conv block of the baseline


def forward(model, x, target):
    """Run the model layer by layer so we can grab the feature map named `target`.
    (A nested pretrained model hides its inner tensors, so a split model will not do.)"""
    conv_out = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        x = layer(x)
        if layer.name == target:
            conv_out = x
    return conv_out, x


def gradcam(model, target, x, pred_class):
    with tf.GradientTape() as tape:
        conv_out, prob = forward(model, x, target)
        score = prob[:, 0] if pred_class == 1 else 1.0 - prob[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * conv_out, axis=-1))[0].numpy()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, float(prob[0, 0])


def overlay(img_uint8, cam):
    h, w = img_uint8.shape[:2]
    cam_img = tf.image.resize(cam[..., None], (h, w)).numpy()[..., 0]
    heat = plt.get_cmap("jet")(cam_img)[..., :3] * 255
    return (0.55 * img_uint8 + 0.45 * heat).astype("uint8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    run = ARTIFACT_DIR / args.run_name
    cfg = json.loads((run / "config.json").read_text())
    model = keras.models.load_model(run / "model.keras")
    layer = conv_layer_name(model)

    preds = pd.read_csv(run / "test_predictions.csv")
    preds["conf"] = np.where(preds["pred"] == 1, preds["prob_ai"], 1 - preds["prob_ai"])
    picks = [("correct", preds[preds["correct"]].nlargest(args.n, "conf")),
             ("wrong", preds[~preds["correct"]].nlargest(args.n, "conf"))]

    fig, axes = plt.subplots(2, args.n, figsize=(args.n * 3.2, 7), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for row, (kind, df) in enumerate(picks):
        for col, (_, r) in enumerate(df.iterrows()):
            x = load_image_array(DATA_ROOT / r["path"], cfg["img_size"], cfg["resize_mode"])
            cam, p = gradcam(model, layer, x, int(r["pred"]))
            axes[row, col].imshow(overlay(x[0].numpy().astype("uint8"), cam))
            truth = "AI" if r["label"] == 1 else "Real"
            said = "AI" if r["pred"] == 1 else "Real"
            axes[row, col].set_title(f"{kind}: true {truth}, said {said}\nP(AI)={p:.2f}", fontsize=9)
    fig.suptitle(f"Grad-CAM, {args.run_name} (top row correct, bottom row wrong)")
    fig.tight_layout()
    out = run / "gradcam.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
