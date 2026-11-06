"""Tasks 3.1 and 3.2: MLflow tool lab with a tiny Keras model on Fashion-MNIST.

Local (3.1), runs are stored in ./mlflow.db:
    python src/mlflow_toy.py --lr 0.001
    python src/mlflow_toy.py --lr 0.01
    mlflow ui --backend-store-uri sqlite:///mlflow.db      -> http://127.0.0.1:5000

DagsHub (3.2): put MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME and
MLFLOW_TRACKING_PASSWORD in .env (git-ignored) and run the same command with --remote.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from tensorflow import keras


class MlflowEpochLogger(keras.callbacks.Callback):
    """Send every Keras metric to MLflow at the end of each epoch."""

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--units", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--remote", action="store_true", help="log to DagsHub using .env settings")
    args = ap.parse_args()

    if args.remote:
        from dotenv import load_dotenv
        load_dotenv()  # reads the token from .env, never from code
        if not os.getenv("MLFLOW_TRACKING_URI"):
            raise SystemExit("MLFLOW_TRACKING_URI is not set. Add it to .env first.")
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    keras.utils.set_random_seed(42)
    (x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
    x, xt = x[:10000] / 255.0, xt[:2000] / 255.0
    y, yt = y[:10000], yt[:2000]

    model = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(args.units, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(keras.optimizers.Adam(args.lr), "sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    mlflow.set_experiment("toy-fashion-mnist")
    # MLflow 3 UI otherwise opens experiments in its GenAI view, which hides the runs table.
    mlflow.set_experiment_tag("mlflow.experimentKind", "custom_model_development")
    with mlflow.start_run(run_name=f"toy_lr{args.lr}") as run:
        mlflow.log_params({"lr": args.lr, "units": args.units, "epochs": args.epochs,
                           "batch_size": 32, "train_samples": len(x)})
        mlflow.set_tags({"task": "3.1" if not args.remote else "3.2",
                         "dataset": "fashion_mnist", "seed": "42"})

        h = model.fit(x, y, validation_split=0.2, epochs=args.epochs, verbose=2,
                      callbacks=[MlflowEpochLogger()])
        test_loss, test_acc = model.evaluate(xt, yt, verbose=0)
        mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_acc})

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(h.history["loss"], label="train loss")
        ax.plot(h.history["val_loss"], label="validation loss")
        ax.set_xlabel("epoch")
        ax.set_title(f"Toy model, lr={args.lr}")
        ax.legend()
        mlflow.log_figure(fig, "loss_curve.png")
        plt.close(fig)

        mlflow.keras.log_model(model, name="model")
        print(f"run {run.info.run_id}: test accuracy {test_acc:.4f}")


if __name__ == "__main__":
    main()
