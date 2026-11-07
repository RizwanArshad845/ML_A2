# VisionGuard model API image for Hugging Face Spaces (Task 7).
# Same ideas as the web Dockerfile (Task 6.2): slim base, deps before code, CPU-only
# TensorFlow, non-root user. Differences for the cloud:
#   * the champion model is BUILT IN (a Space has no folder to mount, see report 6.3)
#   * Spaces run containers as uid 1000 and route traffic to port 7860
#   * API_KEY is NOT here: it is a Space secret, injected as an environment variable
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /srv

COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

RUN useradd --create-home --uid 1000 appuser

COPY model/ model/
COPY app/ app/
COPY src/ src/
COPY api/ api/

ENV HOST=0.0.0.0 \
    PORT=7860 \
    MODEL_PATH=/srv/model/model.keras

USER appuser
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "api"]
