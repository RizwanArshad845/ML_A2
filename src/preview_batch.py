"""Task 2.1 evidence: save a figure of one augmented training batch.

Usage:
    python src/preview_batch.py --augment basic --out reports/figures/augmented_batch.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SEED, SPLIT_DIR
from data import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--resize_mode", choices=["pad", "stretch"], default="pad")
    ap.add_argument("--augment", choices=["basic", "strong"], default="basic")
    ap.add_argument("--out", default="reports/figures/augmented_batch.png")
    args = ap.parse_args()

    ds, _ = build_dataset(SPLIT_DIR / "train.csv", img_size=args.img_size, batch_size=16,
                          resize_mode=args.resize_mode, augment=args.augment,
                          training=True, seed=SEED)
    images, labels = next(iter(ds))
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, img, y in zip(axes.ravel(), images.numpy(), labels.numpy()):
        ax.imshow(img.astype("uint8"))
        ax.set_title("AI" if y == 1 else "Real", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"One augmented training batch (augment={args.augment}, "
                 f"{args.img_size}px, resize={args.resize_mode})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out, "pixel range:", float(images.numpy().min()), float(images.numpy().max()))


if __name__ == "__main__":
    main()
