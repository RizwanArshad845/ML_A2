"""Task 2.7: Keras Tuner.

Toy part (Fashion-MNIST, tiny network):
    python src/tune.py toy

Apply part (VisionGuard transfer model, at least 8 trials):
    python src/tune.py apply --trials 8 --epochs 6

The search sees only the training and validation lists. The test set is never used.
"""
import argparse

import keras_tuner as kt
import numpy as np
import pandas as pd
from tensorflow import keras

from config import REPORT_DIR, SEED, SPLIT_DIR
from data import build_dataset
from models import build_transfer, compile_model, get_base, unfreeze_last_n


def results_table(tuner) -> pd.DataFrame:
    rows = []
    for t in tuner.oracle.trials.values():
        rows.append({"trial": t.trial_id, **t.hyperparameters.values, "score": t.score})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def run_toy(args) -> None:
    (x, y), _ = keras.datasets.fashion_mnist.load_data()
    x = (x[:10000] / 255.0).astype("float32")
    y = y[:10000]

    def build(hp):
        model = keras.Sequential([
            keras.Input((28, 28)),
            keras.layers.Flatten(),
            keras.layers.Dense(hp.Int("units", 32, 256, step=32), activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ])
        lr = hp.Float("lr", 1e-4, 1e-2, sampling="log")
        model.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    tuner = kt.RandomSearch(build, objective="val_accuracy", max_trials=args.trials,
                            seed=SEED, directory="tuner_runs", project_name="toy_fashion",
                            overwrite=True)
    tuner.search(x, y, epochs=args.epochs, validation_split=0.2, verbose=2)
    table = results_table(tuner)
    REPORT_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORT_DIR / "tuner_toy_results.csv", index=False)
    print("\nToy results (best first):")
    print(table.round(4).to_string(index=False))
    print("\nBest hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)


def run_apply(args) -> None:
    common = dict(img_size=args.img_size, batch_size=32, resize_mode="pad", seed=SEED)
    train_ds, _ = build_dataset(SPLIT_DIR / "train.csv", augment="basic", training=True, **common)
    val_ds, _ = build_dataset(SPLIT_DIR / "val.csv", training=False, **common)
    weights = None if args.weights.lower() == "none" else args.weights

    def build(hp):
        lr = hp.Float("lr", 1e-5, 1e-3, sampling="log")
        dropout = hp.Float("dropout", 0.1, 0.5, step=0.1)
        n_unfrozen = hp.Choice("unfrozen_layers", [0, 10, 20, 40])
        model = build_transfer(args.img_size, dropout, weights)
        unfreeze_last_n(get_base(model), n_unfrozen)
        return compile_model(model, lr)

    tuner = kt.RandomSearch(build, objective=kt.Objective("val_auc", "max"),
                            max_trials=args.trials, seed=SEED, directory="tuner_runs",
                            project_name="visionguard", overwrite=True)
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    tuner.search(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[stop], verbose=2)

    table = results_table(tuner)
    REPORT_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORT_DIR / "tuner_visionguard_results.csv", index=False)
    print("\nVisionGuard tuner results (best validation AUC first):")
    print(table.round(5).to_string(index=False))

    # Rough sensitivity check: rank correlation between each setting and the score.
    corr = table[["lr", "dropout", "unfrozen_layers", "score"]].corr(method="spearman")["score"]
    print("\nSpearman correlation with validation AUC (closer to +1 or -1 = matters more):")
    print(corr.drop("score").round(3).to_string())
    print("\nBest hyperparameters:", tuner.get_best_hyperparameters(1)[0].values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["toy", "apply"])
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--weights", default="imagenet")
    args = ap.parse_args()
    if args.part == "toy":
        args.trials = args.trials or 6
        args.epochs = args.epochs or 3
        run_toy(args)
    else:
        args.trials = args.trials or 8
        args.epochs = args.epochs or 6
        run_apply(args)


if __name__ == "__main__":
    main()
