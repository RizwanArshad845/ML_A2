"""Task 3.3 / 3.4: MLflow helpers for train.py.

Two tracking modes:
  manual   our own Keras callback logs every metric per epoch, we log params ourselves
  autolog  mlflow.tensorflow.autolog() captures what it can automatically
Both modes add the same tags, test metrics and artifacts, so the runs stay comparable.

Credentials come from .env (MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD), never from code.
"""
import os
import subprocess
import sys
from pathlib import Path

import mlflow
import yaml
from tensorflow import keras

from config import ROOT, SPLIT_DIR

EXPERIMENT = "visionguard"


def setup(experiment: str = EXPERIMENT) -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")  # does not override variables already set in the shell
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)
    # MLflow 3 UI otherwise opens experiments in the GenAI view, hiding the runs table.
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    print("MLflow tracking to", mlflow.get_tracking_uri())


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def dvc_data_version() -> str:
    """The md5 of the dataset folder recorded in data/raw.dvc (changes whenever the data changes)."""
    try:
        info = yaml.safe_load((ROOT / "data" / "raw.dvc").read_text())
        return info["outs"][0]["md5"]
    except Exception:
        return "unknown"


def run_tags(args) -> dict:
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
        "dvc_data_version": dvc_data_version(),
        "seed": str(args.seed),
        "model_type": args.model,
        "tracking_mode": args.tracking,
        "change": args.change or "reference",
    }


class MlflowEpochLogger(keras.callbacks.Callback):
    """Manual mode: log every training and validation metric at the end of each epoch.
    `epoch` is already global because stage 2 continues with initial_epoch."""

    def __init__(self, stage: str):
        super().__init__()
        self.stage = stage

    def on_epoch_end(self, epoch, logs=None):
        metrics = {k: float(v) for k, v in (logs or {}).items()}
        metrics["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        mlflow.log_metrics(metrics, step=epoch + 1)


def log_environment(outdir: Path) -> None:
    """Exact package list of this environment, so the run can be rebuilt later."""
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True).stdout
    path = outdir / "requirements-freeze.txt"
    path.write_text(freeze)
    mlflow.log_artifact(str(path), artifact_path="environment")


def log_run_artifacts(outdir: Path) -> None:
    """Task 3.4: figures, split lists, package list and the files serving needs."""
    for name in ["confusion_matrix.png", "roc_curve.png", "curves_stage1.png", "curves_stage2.png"]:
        if (outdir / name).exists():
            mlflow.log_artifact(str(outdir / name), artifact_path="figures")
    for name in ["train.csv", "val.csv", "test.csv", "split_summary.json"]:
        mlflow.log_artifact(str(SPLIT_DIR / name), artifact_path="splits")
    for name in ["config.json", "threshold.json", "train_summary.json", "history.csv",
                 "layer_table.csv", "model_summary.txt", "trainable_layers_stage1.txt",
                 "trainable_layers_stage2.txt", "val_threshold_table.csv", "test_metrics.json"]:
        if (outdir / name).exists():
            mlflow.log_artifact(str(outdir / name), artifact_path="run_files")
    log_environment(outdir)
