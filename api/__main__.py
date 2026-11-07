"""Run the model API with the Waitress production server (debug off).

    API_KEY=... python -m api            # listens on PORT (default 7860, Hugging Face's port)
"""
import logging
import os

from waitress import serve

from api import create_api

try:  # local .env for testing; in the cloud API_KEY is a Space secret
    from dotenv import load_dotenv
    load_dotenv()
    os.environ.setdefault("API_KEY", os.getenv("VG_API_KEY", ""))
except ImportError:
    pass


def main() -> None:
    api = create_api()
    host, port = os.getenv("HOST", "0.0.0.0"), int(os.getenv("PORT", "7860"))
    logging.getLogger(__name__).info("VisionGuard model API on http://%s:%d (debug off)", host, port)
    serve(api, host=host, port=port, threads=4)


if __name__ == "__main__":
    main()
