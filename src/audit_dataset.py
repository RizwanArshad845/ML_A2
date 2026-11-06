"""Task 1.2: dataset audit.

Walks a dataset folder where each sub folder is a class, and writes one row per
file with: class, size on disk, width, height, colour mode, format, SHA-256 and a
status (ok or corrupt). Then prints the numbers the report needs.

Usage:
    python src/audit_dataset.py --root data/raw --out reports/audit.csv
"""
import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError



def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Hash the raw bytes of the file. Same bytes means same hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def inspect(path: Path, label: str, root: Path) -> dict:
    row = {
        "path": str(path.relative_to(root)),
        "class": label,
        "file_size": path.stat().st_size,
        "width": None,
        "height": None,
        "mode": None,
        "format": None,
        "sha256": sha256_of(path),
        "status": "ok",
        "error": "",
    }
    try:
        # verify() checks the file structure but leaves the object unusable,
        # so we open the file a second time to read the real dimensions.
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()  # forces a full decode, catches truncated files
            row["width"], row["height"] = im.size
            row["mode"] = im.mode
            row["format"] = im.format
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        row["status"] = "corrupt"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("reports/audit.csv"), type=Path)
    args = ap.parse_args()

    rows = []
    for class_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for f in sorted(class_dir.rglob("*")):
            if f.is_file():
                # Every file is inspected, even one with a non image extension,
                # because a stray text file is exactly what an audit should catch.
                rows.append(inspect(f, class_dir.name, args.root))

    df = pd.DataFrame(rows)
    # Duplicates: every file whose hash appears more than once.
    df["dup_group_size"] = df.groupby("sha256")["sha256"].transform("size")
    df["is_duplicate"] = df["dup_group_size"] > 1
    # Within a group of identical files, keep the first and call the rest extras.
    df["is_extra_copy"] = df.duplicated("sha256", keep="first")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    ok = df[df["status"] == "ok"]
    print("\n== Counts per class (all files) ==")
    print(df["class"].value_counts().to_string())

    print("\n== min / median / max per class (readable files only) ==")
    stats = ok.groupby("class")[["width", "height", "file_size"]].agg(["min", "median", "max"])
    print(stats.to_string())

    print("\n== Colour modes and formats ==")
    print(ok.groupby(["class", "mode"]).size().to_string())
    print(ok.groupby(["class", "format"]).size().to_string())

    print("\n== Integrity ==")
    print("corrupt files:", int((df["status"] == "corrupt").sum()))
    print("files that belong to a duplicate group:", int(df["is_duplicate"].sum()))
    print("extra copies that could be removed:", int(df["is_extra_copy"].sum()))
    cross = df[df["is_duplicate"]].groupby("sha256")["class"].nunique()
    print("duplicate groups spanning both classes (label conflicts):", int((cross > 1).sum()))
    print(f"\nFull table written to {args.out}")


if __name__ == "__main__":
    main()
