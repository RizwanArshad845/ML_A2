"""Task 1.3: visual EDA.

Reads the CSV from audit_dataset.py and makes:
  1. a sample grid per class
  2. histograms of width, height and file size per class

Usage:
    python src/eda_figures.py --root data/raw --audit reports/audit.csv --outdir reports/figures
"""
import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def sample_grid(df, root: Path, outdir: Path, n: int = 12, seed: int = 42) -> None:
    rnd = random.Random(seed)
    for cls, g in df[df["status"] == "ok"].groupby("class"):
        paths = rnd.sample(list(g["path"]), min(n, len(g)))
        cols = 4
        rows = (len(paths) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, p in zip(axes.ravel(), paths):
            with Image.open(root / p) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(Path(p).name[:22], fontsize=7)
        fig.suptitle(f"Random samples: {cls}", fontsize=14)
        fig.tight_layout()
        fig.savefig(outdir / f"samples_{cls}.png", dpi=150)
        plt.close(fig)


def histograms(df, outdir: Path) -> None:
    ok = df[df["status"] == "ok"]
    for col, label in [("width", "Width (px)"), ("height", "Height (px)"), ("file_size", "File size (KB)")]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for cls, g in ok.groupby("class"):
            vals = g[col] / 1024 if col == "file_size" else g[col]
            ax.hist(vals, bins=40, alpha=0.55, label=cls)
        ax.set_xlabel(label)
        ax.set_ylabel("Number of images")
        ax.set_title(f"{label} by class")
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"hist_{col}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--audit", required=True, type=Path)
    ap.add_argument("--outdir", default=Path("reports/figures"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.audit)
    sample_grid(df, args.root, args.outdir)
    histograms(df, args.outdir)
    print("Figures written to", args.outdir)


if __name__ == "__main__":
    main()
