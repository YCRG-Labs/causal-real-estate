"""Upload the release/ bundle to Hugging Face Hub.

Creates a *private* dataset repo on first run; flip to public via the web UI or
`HfApi().update_repo_visibility(...)` once the paper preprints.
"""
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID = "jcrainic2/causal-real-estate"
REPO_TYPE = "dataset"
PRIVATE = False

ROOT = Path(__file__).resolve().parents[1]  # the release/ folder

IGNORE = ["scripts/push_to_hf.py", "scripts/__pycache__", "*.pyc"]


def main() -> None:
    api = HfApi()
    user = api.whoami()
    print(f"logged in as {user['name']}; pushing to {REPO_ID} (private={PRIVATE})")

    create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=PRIVATE,
        exist_ok=True,
    )
    print("repo ready, uploading folder...")

    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="initial release",
        ignore_patterns=IGNORE,
    )
    print(f"done. https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
