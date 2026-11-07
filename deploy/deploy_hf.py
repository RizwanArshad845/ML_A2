"""Task 7: deploy the VisionGuard model API to a Hugging Face Space (Docker).

    python deploy/deploy_hf.py                 # creates/updates <hf-user>/visionguard-api

Steps it performs:
  1. reads HF_TOKEN (a Hugging Face *write* token) from .env
  2. creates VG_API_KEY in .env if it is not there yet (random, 43 characters)
  3. stages Dockerfile + code + the exported champion model in build/hf_space/
  4. creates the Space (Docker SDK, public) if needed and stores API_KEY as a Space *secret*
  5. uploads the staged folder; Hugging Face then builds and starts the container

Neither the token nor the API key is printed or uploaded as a file.
"""
import argparse
import secrets
import shutil
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"
STAGE = ROOT / "build" / "hf_space"
MODEL_DIR = ROOT / "models" / "champion"
MODEL_FILES = ["model.keras", "config.json", "threshold.json", "model_info.json"]
APP_FILES = ["__init__.py", "config.py", "db.py", "inference.py", "validation.py", "routes.py"]


def ensure_api_key() -> str:
    env = dotenv_values(ENV_FILE)
    if env.get("VG_API_KEY"):
        return env["VG_API_KEY"]
    key = secrets.token_urlsafe(32)
    with ENV_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n# Task 7: secret key for the cloud model API (X-API-Key header)\nVG_API_KEY={key}\n")
    print("Created VG_API_KEY and saved it to .env")
    return key


def stage() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "app").mkdir(parents=True)
    (STAGE / "src").mkdir()
    (STAGE / "api").mkdir()
    (STAGE / "model").mkdir()
    for name in ["Dockerfile", "requirements-api.txt", "README.md"]:
        shutil.copy(ROOT / "deploy" / "hf_space" / name, STAGE / name)
    for name in APP_FILES:
        shutil.copy(ROOT / "app" / name, STAGE / "app" / name)
    for name in ["__init__.py", "preprocess.py"]:
        shutil.copy(ROOT / "src" / name, STAGE / "src" / name)
    for name in ["__init__.py", "__main__.py"]:
        shutil.copy(ROOT / "api" / name, STAGE / "api" / name)
    for name in MODEL_FILES:
        src = MODEL_DIR / name
        if not src.exists():
            raise SystemExit(f"{src} missing. Run src/load_champion.py <img> --export models/champion")
        shutil.copy(src, STAGE / "model" / name)
    print(f"Staged {sum(1 for p in STAGE.rglob('*') if p.is_file())} files in {STAGE}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="visionguard-api", help="Space name")
    args = ap.parse_args()

    token = dotenv_values(ENV_FILE).get("HF_TOKEN")
    if not token:
        raise SystemExit("Add HF_TOKEN=<your Hugging Face write token> to .env first.")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{args.space}"

    key = ensure_api_key()
    stage()

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True, private=False)
    api.add_space_secret(repo_id, "API_KEY", key, description="VisionGuard X-API-Key")
    api.upload_folder(repo_id=repo_id, repo_type="space", folder_path=STAGE,
                      commit_message="Deploy VisionGuard model API")

    host = f"{user}-{args.space}".lower().replace("_", "-").replace(".", "-")
    print(f"\nSpace page : https://huggingface.co/spaces/{repo_id}")
    print(f"API URL    : https://{host}.hf.space")
    print("The first build takes about 5-10 minutes (Logs tab on the Space page).")


if __name__ == "__main__":
    main()
