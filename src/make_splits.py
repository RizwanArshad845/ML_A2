"""Task 1 follow up: build the train, validation and test lists.

Reads the audit table, drops corrupt files and extra copies of duplicates, then
makes a stratified 70/15/15 split with a fixed seed. Paths are stored with forward
slashes so the same lists work on Windows, Linux and inside Docker.

Usage:
    python src/make_splits.py --audit reports/audit.csv
"""
import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import POSITIVE_CLASS_FOLDER, SEED, SPLIT_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default="reports/audit.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.audit)
    n_all = len(df)
    df = df[(df["status"] == "ok") & (~df["is_extra_copy"].astype(bool))].copy()
    print(f"{n_all} files in audit, {len(df)} kept after dropping corrupt files and extra copies")

    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    df["label"] = (df["class"] == POSITIVE_CLASS_FOLDER).astype(int)
    df = df[["path", "label", "class", "sha256"]]

    held = args.val_frac + args.test_frac
    train, rest = train_test_split(df, test_size=held, stratify=df["label"], random_state=args.seed)
    val, test = train_test_split(
        rest, test_size=args.test_frac / held, stratify=rest["label"], random_state=args.seed
    )

    # Leakage check: no byte-identical image may appear in two splits.
    sets = {"train": set(train.sha256), "val": set(val.sha256), "test": set(test.sha256)}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not (sets[a] & sets[b]), f"hash overlap between {a} and {b}"

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.sample(frac=1, random_state=args.seed).to_csv(SPLIT_DIR / f"{name}.csv", index=False)
        summary[name] = {
            "total": int(len(part)),
            "ai": int(part["label"].sum()),
            "real": int((part["label"] == 0).sum()),
        }
    summary["seed"] = args.seed
    (SPLIT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))

    print(pd.DataFrame({k: v for k, v in summary.items() if k != "seed"}).T.to_string())
    print(f"\nWritten to {SPLIT_DIR}")


if __name__ == "__main__":
    main()
