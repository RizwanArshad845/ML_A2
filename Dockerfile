# VisionGuard web app (Task 6.2)
# Slim Debian base: has glibc (needed by TensorFlow wheels) but no compilers or docs.
# Python 3.13 matches the environment the model was trained in.
FROM python:3.13-slim

# No .pyc files, unbuffered logs (so `docker logs` shows lines immediately), no pip cache.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /srv

# 1) Dependencies first. This layer is rebuilt only when requirements-app.txt changes,
#    so editing the app code reuses the cached (large) TensorFlow layer.
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# 2) Unprivileged user. If the app is ever compromised, the attacker is not root.
RUN useradd --create-home --uid 10001 appuser

# 3) Source code last: it changes most often.
COPY app/ app/
COPY src/__init__.py src/preprocess.py src/

# Settings come from the environment at run time (Task 4.4). No secrets are baked in:
# passwords arrive from docker-compose / .env, never from this file.
ENV HOST=0.0.0.0 \
    PORT=5000 \
    MODEL_PATH=/models/champion/model.keras

USER appuser
EXPOSE 5000

# Docker marks the container unhealthy if /health stops answering.
# start-period gives TensorFlow time to load the model before checks count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "app"]
