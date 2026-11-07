# VisionGuard

[![CI](https://github.com/RizwanArshad845/ML_A2/actions/workflows/ci.yml/badge.svg)](https://github.com/RizwanArshad845/ML_A2/actions/workflows/ci.yml)

A web service that tells whether an uploaded image is **AI-generated or real**.
A MobileNetV2 model (TensorFlow/Keras) is trained on the Kaggle *AI Generated Images vs Real Images*
dataset, tracked with MLflow on DagsHub, served by a Flask app that logs every prediction to a
database, and packaged with Docker Compose.

- DagsHub (code, DVC data, MLflow runs): https://dagshub.com/RizwanArshad845/ML_A2
- Experiments: https://dagshub.com/RizwanArshad845/ML_A2.mlflow

## Project layout

| Path | What it is |
|---|---|
| `src/` | data audit, splits, `tf.data` pipeline, models, training, evaluation, tuning, MLflow tracking, registry |
| `app/` | Flask app: upload, result, history and health pages, validation, database trail |
| `tests/` | app tests (run in CI) |
| `data/raw.dvc` | DVC pointer to the dataset stored on DagsHub |
| `Dockerfile`, `docker-compose.yml` | web app image and the web + Postgres stack |
| `.github/workflows/ci.yml` | lint, test and Docker build on every push |

## Quick start

```bash
# 1. environment
python -m venv .venv && .venv/Scripts/activate        # Windows (source .venv/bin/activate on Linux)
pip install -r requirements.txt

# 2. data (needs DagsHub credentials, see .env.example)
dvc pull

# 3. train and evaluate
python src/make_splits.py --audit reports/audit.csv
python src/train.py --model transfer --run_name mobilenet_v1 --tracking manual

# 4. get the champion model from the registry, then run the stack
python src/load_champion.py some_image.jpg --export models/champion
cp .env.example .env   # then fill in the passwords
docker compose up -d --build      # open http://localhost:5000
```

## Secrets

Credentials (DagsHub token, database password) live only in `.env`, which is git-ignored.
`.env.example` lists the variables with placeholder values.
