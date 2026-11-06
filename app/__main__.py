"""Task 4.4: run VisionGuard with a production WSGI server, debug mode off.

    python -m app

Flask's built-in server (app.run) is for development only: it handles requests
poorly under load and, in debug mode, exposes an interactive debugger that lets
anyone run Python code on the server. Waitress is a production server that works
on Windows and Linux. Debug mode is never switched on.
"""
import logging

from waitress import serve

from app import create_app

try:  # load a local .env file if python-dotenv is installed (never needed in Docker)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    app = create_app()
    s = app.config["VG_SETTINGS"]
    logging.getLogger(__name__).info("VisionGuard serving on http://%s:%d (debug off)", s.host, s.port)
    serve(app, host=s.host, port=s.port, threads=4)


if __name__ == "__main__":
    main()
