"""Tasks 2.5 and 2.6: evaluation and error analysis.

Order of operations matters:
  1. predict on the VALIDATION set and choose the decision threshold there
  2. predict on the TEST set once and report metrics at that threshold

The script refuses to touch the test set a second time for the same run unless
you pass --force, so the test score stays an honest estimate.

Usage:
    python src/evaluate.py --run_name mobilenet_v1
    python src/evaluate.py --run_name mobilenet_v1 --criterion fpr_cap --max_fpr 0.05
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from tensorflow import keras

from config import ARTIFACT_DIR, DATA_ROOT, REPORT_DIR, SPLIT_DIR
from data import build_dataset


def threshold_table(y, p) -> pd.DataFrame:
    rows = []
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        pred = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": t,
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "false_positive_rate": fp / max(fp + tn, 1),   # real photos called fake
            "false_negative_rate": fn / max(fn + tp, 1),   # fakes passed as real
        })
    return pd.DataFrame(rows)


def choose_threshold(table: pd.DataFrame, criterion: str, max_fpr: float) -> float:
    if criterion == "f1":
        best = table["f1"].max()
        cands = table[table["f1"] >= best - 1e-12]
        # among ties, prefer the one closest to 0.5
        return float(cands.iloc[(cands["threshold"] - 0.5).abs().argsort().iloc[0]]["threshold"])
    # fpr_cap: protect real photos. Highest recall while keeping false positives low.
    ok = table[table["false_positive_rate"] <= max_fpr]
    if ok.empty:
        return float(table["threshold"].max())
    return float(ok.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]["threshold"])


def metrics_at(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": thr,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion(y, p, thr, title, path):
    cm = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    names = ["Real", "AI"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y, p, thr, title, path):
    fpr, tpr, _ = roc_curve(y, p)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.scatter([fp / max(fp + tn, 1)], [tp / max(tp + fn, 1)], color="red", zorder=5,
               label=f"chosen threshold {thr:.2f}")
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def error_grid(pred_df: pd.DataFrame, path, n=10):
    wrong = pred_df[~pred_df["correct"]].copy()
    wrong["conf_in_wrong"] = np.where(wrong["pred"] == 1, wrong["prob_ai"], 1 - wrong["prob_ai"])
    wrong = wrong.sort_values("conf_in_wrong", ascending=False).head(n)
    if wrong.empty:
        print("No misclassified test images.")
        return wrong
    cols = 5
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (ax, (_, r)) in enumerate(zip(axes.ravel(), wrong.iterrows()), start=1):
        with Image.open(DATA_ROOT / r["path"]) as im:
            ax.imshow(im.convert("RGB"))
        truth = "AI" if r["label"] == 1 else "Real"
        said = "AI" if r["pred"] == 1 else "Real"
        ax.set_title(f"#{k} true {truth}, said {said}\nP(AI)={r['prob_ai']:.2f}", fontsize=9)
    fig.suptitle("Most confident mistakes on the test set")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return wrong


def evaluate_model(model, img_size, resize_mode, outdir, title, criterion, max_fpr) -> dict:
    """Pick the threshold on validation, then score the test set at that threshold.
    Writes threshold.json, test_metrics.json, the confusion matrix and the ROC curve to outdir.
    Shared by this script and train.py (which logs the results to MLflow)."""
    common = dict(img_size=img_size, resize_mode=resize_mode, batch_size=32, training=False)

    # 1. validation set: choose the threshold
    val_ds, val_df = build_dataset(SPLIT_DIR / "val.csv", **common)
    p_val = model.predict(val_ds, verbose=0).ravel()
    y_val = val_df["label"].values
    table = threshold_table(y_val, p_val)
    table.to_csv(outdir / "val_threshold_table.csv", index=False)
    thr = choose_threshold(table, criterion, max_fpr)
    print(f"Threshold chosen on validation ({criterion}): {thr:.2f}")
    (outdir / "threshold.json").write_text(json.dumps(
        {"threshold": thr, "criterion": criterion, "max_fpr": max_fpr}, indent=2))
    val_row = table[table["threshold"] == thr].iloc[0]
    # Named "...at_thr" so they do not clash with Keras' per-epoch val_* metrics in MLflow.
    val = {"val_roc_auc": float(roc_auc_score(y_val, p_val)), "threshold": thr,
           "val_precision_at_thr": float(val_row["precision"]),
           "val_recall_at_thr": float(val_row["recall"]),
           "val_f1_at_thr": float(val_row["f1"]),
           "val_fpr_at_thr": float(val_row["false_positive_rate"])}

    # 2. test set: scored at the threshold fixed above
    test_ds, test_df = build_dataset(SPLIT_DIR / "test.csv", **common)
    p_test = model.predict(test_ds, verbose=0).ravel()
    y_test = test_df["label"].values
    m = metrics_at(y_test, p_test, thr)
    m_default = metrics_at(y_test, p_test, 0.5)
    (outdir / "test_metrics.json").write_text(json.dumps({"chosen": m, "at_0.5": m_default}, indent=2))

    plot_confusion(y_test, p_test, thr, f"{title} (threshold {thr:.2f})",
                   outdir / "confusion_matrix.png")
    plot_roc(y_test, p_test, thr, f"ROC: {title}", outdir / "roc_curve.png")

    preds = test_df.copy()
    preds["prob_ai"] = p_test
    preds["pred"] = (p_test >= thr).astype(int)
    preds["correct"] = preds["pred"] == preds["label"]
    preds.to_csv(outdir / "test_predictions.csv", index=False)
    return {"threshold": thr, "val": val, "test": m, "predictions": preds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--criterion", choices=["f1", "fpr_cap"], default="f1")
    ap.add_argument("--max_fpr", type=float, default=0.05)
    ap.add_argument("--force", action="store_true", help="allow a second look at the test set")
    args = ap.parse_args()

    run = ARTIFACT_DIR / args.run_name
    metrics_file = run / "test_metrics.json"
    if metrics_file.exists() and not args.force:
        raise SystemExit(f"{metrics_file} exists: the test set was already used for this run. "
                         "Use --force only if you accept that this is no longer a clean test.")

    cfg = json.loads((run / "config.json").read_text())
    model = keras.models.load_model(run / "model.keras")
    result = evaluate_model(model, cfg["img_size"], cfg["resize_mode"], run, args.run_name,
                            args.criterion, args.max_fpr)
    m = result["test"]
    error_grid(result["predictions"], run / "misclassified_top10.png")

    REPORT_DIR.mkdir(exist_ok=True)
    row = {"run_name": args.run_name, **{k: round(v, 4) if isinstance(v, float) else v
                                         for k, v in m.items()}}
    comp = REPORT_DIR / "model_comparison.csv"
    old = pd.read_csv(comp) if comp.exists() else pd.DataFrame()
    old = old[old["run_name"] != args.run_name] if len(old) else old
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(comp, index=False)

    print("\nTest metrics at the chosen threshold:")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {k:10s} {m[k]:.4f}")
    print(f"  real photos called fake (false positives): {m['fp']}")
    print(f"  fake images passed as real (false negatives): {m['fn']}")
    print(f"\nArtifacts written to {run}")


if __name__ == "__main__":
    main()
