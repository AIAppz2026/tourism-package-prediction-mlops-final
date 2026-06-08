from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

SPACE_REPO_ID = "AIApps2026/tourism-package-predictor-final"
HF_REPOS_PRIVATE = False

DEPLOYMENT_DIR = Path(__file__).resolve().parent

hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN was not found in the environment. Please add it as a GitHub Actions repository secret.")

api = HfApi(token=hf_token)

create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    private=HF_REPOS_PRIVATE,
    exist_ok=True,
    token=hf_token
)

api.upload_folder(
    folder_path=str(DEPLOYMENT_DIR),
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    token=hf_token,
    commit_message="Deploy Streamlit app to Hugging Face Space"
)

print("Deployment files pushed successfully to Hugging Face Space.")
print("Space repository:", SPACE_REPO_ID)