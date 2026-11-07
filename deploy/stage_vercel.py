"""Task 7: stage the Vercel model API in build/visionguard-api/ (then `vercel deploy`).

    python deploy/export_tflite.py      # once: models/champion/model.tflite
    python deploy/stage_vercel.py
    cd build/visionguard-api && npx vercel deploy --prod

Only what the API needs is copied: no TensorFlow code paths, no .keras file, no secrets.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "build" / "visionguard-api"
MODEL_DIR = ROOT / "models" / "champion"
COPY = {
    "deploy/vercel/index.py": "index.py",
    "deploy/vercel/vercel.json": "vercel.json",
    "deploy/vercel/requirements.txt": "requirements.txt",
    "deploy/vercel/.vercelignore": ".vercelignore",
    "api/__init__.py": "api/__init__.py",
    "api/lite.py": "api/lite.py",
    "app/__init__.py": "app/__init__.py",
    "app/config.py": "app/config.py",
    "app/db.py": "app/db.py",
    "app/inference.py": "app/inference.py",
    "app/validation.py": "app/validation.py",
    "app/routes.py": "app/routes.py",
}
MODEL_FILES = ["model.tflite", "config.json", "threshold.json", "model_info.json"]


def main() -> None:
    keep = STAGE / ".vercel"  # project link created by the first `vercel deploy`
    saved = None
    if keep.exists():
        saved = STAGE.parent / ".vercel-link"
        shutil.rmtree(saved, ignore_errors=True)
        shutil.move(str(keep), str(saved))
    shutil.rmtree(STAGE, ignore_errors=True)
    for src, dst in COPY.items():
        (STAGE / dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / src, STAGE / dst)
    (STAGE / "model").mkdir()
    for name in MODEL_FILES:
        if not (MODEL_DIR / name).exists():
            raise SystemExit(f"missing {MODEL_DIR / name} (run deploy/export_tflite.py first)")
        shutil.copy(MODEL_DIR / name, STAGE / "model" / name)
    if saved:
        shutil.move(str(saved), str(keep))
    size = sum(p.stat().st_size for p in STAGE.rglob("*") if p.is_file()) / 1e6
    print(f"Staged {STAGE} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
