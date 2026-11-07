"""Tasks 2.2, 2.3 and 2.4: train the baseline CNN or the transfer model.

Examples:
    python src/train.py --model baseline --run_name baseline_v1
    python src/train.py --model transfer --run_name mobilenet_v1

The transfer model trains in two stages: (1) pretrained base frozen, only the new
head learns; (2) last block of the base unfrozen, trained again with a much smaller
learning rate. Everything is written to artifacts/<run_name>/.
"""
import argparse
import json
import os
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from config import ARTIFACT_DIR, SEED, SPLIT_DIR
from data import build_dataset
from models import (
    build_baseline,
    build_transfer,
    compile_model,
    get_base,
    layer_table,
    trainable_report,
    unfreeze_last_block,
)


def set_seeds(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)  # python, numpy and tensorflow seeds together
    if deterministic:
        tf.config.experimental.enable_op_determinism()


def make_callbacks(ckpt_path, patience, lr_patience):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss",
                                        save_best_only=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=lr_patience, min_lr=1e-7, verbose=1),
    ]


def plot_curves(hist: pd.DataFrame, best_epoch: int, title: str, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in zip(axes, ["loss", "accuracy"], ["Loss", "Accuracy"]):
        ax.plot(hist["epoch"], hist[metric], label=f"train {metric}")
        ax.plot(hist["epoch"], hist[f"val_{metric}"], label=f"validation {metric}")
        ax.axvline(best_epoch, color="red", linestyle="--", label=f"best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_stage(model, name, epochs, lr, initial_epoch, train_ds, val_ds, outdir, args,
              class_weight):
    compile_model(model, lr)  # recompile: required after changing trainable flags
    (outdir / f"trainable_layers_{name}.txt").write_text(trainable_report(model))
    ckpt = outdir / f"best_{name}.keras"
    callbacks = make_callbacks(ckpt, args.patience, args.lr_patience)
    if args.tracking == "manual":
        from tracking import MlflowEpochLogger
        callbacks.append(MlflowEpochLogger(name))
    h = model.fit(train_ds, validation_data=val_ds, epochs=initial_epoch + epochs,
                  initial_epoch=initial_epoch, class_weight=class_weight, shuffle=False,
                  callbacks=callbacks, verbose=2)
    hist = pd.DataFrame(h.history)
    hist["epoch"] = np.arange(initial_epoch + 1, initial_epoch + 1 + len(hist))
    hist["stage"] = name
    best_idx = int(hist["val_loss"].idxmin())
    best_epoch = int(hist.loc[best_idx, "epoch"])
    plot_curves(hist, best_epoch, f"{args.run_name}: {name} (lr={lr})",
                outdir / f"curves_{name}.png")
    return hist, best_epoch, float(hist.loc[best_idx, "val_loss"]), ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "transfer"], required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--resize_mode", choices=["pad", "stretch"], default="pad")
    ap.add_argument("--augment", choices=["none", "basic", "strong"], default="basic")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30, help="stage 1 maximum epochs")
    ap.add_argument("--epochs_ft", type=int, default=20, help="stage 2 maximum epochs")
    ap.add_argument("--lr", type=float, default=1e-3, help="stage 1 learning rate")
    ap.add_argument("--lr_ft", type=float, default=1e-5, help="stage 2 learning rate")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr_patience", type=int, default=2)
    ap.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (tests only)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--deterministic", action="store_true")
    # Task 3: MLflow tracking
    ap.add_argument("--tracking", choices=["none", "manual", "autolog"], default="none",
                    help="none, manual (own Keras callback) or autolog (mlflow.tensorflow.autolog)")
    ap.add_argument("--change", default="", help="what differs from the reference run (saved as a tag)")
    ap.add_argument("--experiment", default="visionguard")
    ap.add_argument("--criterion", choices=["f1", "fpr_cap"], default="fpr_cap")
    ap.add_argument("--max_fpr", type=float, default=0.15)
    args = ap.parse_args()

    set_seeds(args.seed, args.deterministic)
    outdir = ARTIFACT_DIR / args.run_name
    outdir.mkdir(parents=True, exist_ok=True)

    if args.tracking == "none":
        train(args, outdir)
        return

    import mlflow

    import tracking
    from evaluate import evaluate_model

    tracking.setup(args.experiment)
    if args.tracking == "autolog":
        # Logs fit() params, optimizer settings, per-epoch metrics and the model by itself.
        mlflow.tensorflow.autolog(log_models=True, log_datasets=False, silent=True)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.set_tags(tracking.run_tags(args))
        if args.tracking == "manual":
            mlflow.log_params(vars(args))  # every hyperparameter, including our own

        final, summary = train(args, outdir)
        result = evaluate_model(final, args.img_size, args.resize_mode, outdir, args.run_name,
                                args.criterion, args.max_fpr)

        mlflow.log_metrics({"best_val_loss": summary["best_val_loss"],
                            "best_epoch": summary["best_epoch"], **result["val"]})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in result["test"].items()
                            if k != "threshold"})
        tracking.log_run_artifacts(outdir)
        if args.tracking == "manual":
            mlflow.keras.log_model(final, name="model")
        t = result["test"]
        print(f"\nMLflow run {run.info.run_id}: val_roc_auc {result['val']['val_roc_auc']:.4f}, "
              f"test acc {t['accuracy']:.4f}, f1 {t['f1']:.4f}, auc {t['roc_auc']:.4f}")


def train(args, outdir):
    """Build, train (one or two stages) and save the model. Returns (final model, summary)."""
    common = dict(img_size=args.img_size, batch_size=args.batch_size,
                  resize_mode=args.resize_mode, augment=args.augment, seed=args.seed)
    train_ds, train_df = build_dataset(SPLIT_DIR / "train.csv", training=True, **common)
    val_ds, _ = build_dataset(SPLIT_DIR / "val.csv", training=False, **common)

    class_weight = None
    if args.class_weight == "balanced":
        n, pos = len(train_df), int(train_df["label"].sum())
        class_weight = {0: n / (2 * (n - pos)), 1: n / (2 * pos)}
        print("class weights:", class_weight)

    if args.model == "baseline":
        model = build_baseline(args.img_size, args.dropout)
    else:
        weights = None if args.weights.lower() == "none" else args.weights
        model = build_transfer(args.img_size, args.dropout, weights)
    layer_table(model).to_csv(outdir / "layer_table.csv", index=False)
    with open(outdir / "model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda s, **kw: f.write(s + "\n"))
    model.summary()

    stages = []
    hist1, best1, vloss1, ckpt1 = run_stage(model, "stage1", args.epochs, args.lr, 0,
                                            train_ds, val_ds, outdir, args, class_weight)
    stages.append(("stage1", hist1, best1, vloss1, ckpt1))

    if args.model == "transfer" and args.epochs_ft > 0:  # --epochs_ft 0 keeps the base frozen
        unfreeze_last_block(get_base(model))
        hist2, best2, vloss2, ckpt2 = run_stage(model, "stage2", args.epochs_ft, args.lr_ft,
                                                len(hist1), train_ds, val_ds, outdir, args,
                                                class_weight)
        stages.append(("stage2", hist2, best2, vloss2, ckpt2))

    full = pd.concat([s[1] for s in stages], ignore_index=True)
    full.to_csv(outdir / "history.csv", index=False)

    # The final model is the checkpoint with the lowest validation loss across stages.
    best_stage = min(stages, key=lambda s: s[3])
    final = keras.models.load_model(best_stage[4])
    final.save(outdir / "model.keras")

    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "chosen_stage": best_stage[0],
        "best_epoch": best_stage[2],
        "best_val_loss": best_stage[3],
        "stages": {s[0]: {"best_epoch": s[2], "best_val_loss": s[3]} for s in stages},
        "args": vars(args),
        "versions": {"tensorflow": tf.__version__, "keras": keras.__version__,
                     "python": platform.python_version()},
    }
    (outdir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    # evaluate.py and the Flask app read this file so preprocessing always matches.
    (outdir / "config.json").write_text(json.dumps({
        "img_size": args.img_size, "resize_mode": args.resize_mode,
        "model": args.model, "seed": args.seed}, indent=2))
    print(f"\nBest epoch: {best_stage[2]} ({best_stage[0]}), val_loss {best_stage[3]:.4f}")
    print(f"Final model saved to {outdir / 'model.keras'}")
    return final, summary


if __name__ == "__main__":
    main()
