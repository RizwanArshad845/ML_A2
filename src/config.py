"""Shared settings. Paths can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("DATA_ROOT", ROOT / "data" / "raw" / "archive"))
SPLIT_DIR = Path(os.getenv("SPLIT_DIR", ROOT / "data" / "splits"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
REPORT_DIR = ROOT / "reports"

# Label convention used everywhere: 1 = AI generated (the positive class), 0 = real.
POSITIVE_CLASS_FOLDER = "AiArtData"
SEED = int(os.getenv("SEED", 42))
