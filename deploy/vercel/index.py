"""Vercel entry point for the VisionGuard model API (Task 7).

Vercel's Python runtime imports this file and serves the WSGI object named `app`.
The model is loaded once per cold start (module import), then reused for every request
that the same function instance handles.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MODEL_BACKEND", "tflite")
os.environ.setdefault("MODEL_PATH", str(HERE / "model" / "model.tflite"))
os.environ.setdefault("MAX_UPLOAD_MB", "4")  # Vercel rejects request bodies over 4.5 MB

from api import create_api  # noqa: E402

app = create_api()  # API_KEY comes from the Vercel project's environment variables
