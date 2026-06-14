"""Upload the release/ bundle to the Hugging Face dataset repo.

Uploads the current twelve-market payload (data_12/ + the docs + the pipeline
scripts) and prunes the stale three-city data/ and splits/ that the current
dataset card no longer references, so the Hub matches the paper.

Dry run by default — prints exactly what would upload and what would be pruned.
Add --execute to actually publish.

  python release/scripts/push_to_hf.py            # dry run (shows the plan)
  python release/scripts/push_to_hf.py --execute  # publish
"""
import argparse
import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "jcrainic2/causal-real-estate"
REPO_TYPE = "dataset"
PRIVATE = False
ROOT = Path(__file__).resolve().parents[1]

# Local paths under release/ to NOT upload (old three-city data + stale splits).
IGNORE = ["scripts/push_to_hf.py", "scripts/__pycache__/*", "*.pyc",
          "data/*", "data/**", "splits/*", "splits/**"]
# Stale repo paths to delete so the Hub matches the current card.
PRUNE = ["data/*", "data/**", "splits/*", "splits/**"]


def _local_upload_set() -> list[str]:
    files = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in IGNORE):
            continue
        files.append(rel)
    return sorted(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="actually publish (default: dry run)")
    args = ap.parse_args()

    api = HfApi()
    user = api.whoami()["name"]
    local = _local_upload_set()

    try:
        existing = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    except Exception:
        existing = []
    to_prune = sorted(f for f in existing
                      if any(fnmatch.fnmatch(f, pat) for pat in PRUNE))

    import collections
    by_top = collections.Counter(f.split("/")[0] for f in local)
    print(f"user: {user}   repo: {REPO_ID} (private={PRIVATE})")
    print(f"\nWILL UPLOAD {len(local)} files:")
    for k, v in sorted(by_top.items()):
        print(f"  {k}: {v}")
    print(f"\nWILL PRUNE {len(to_prune)} stale repo files:")
    for f in to_prune:
        print(f"  - {f}")

    if not args.execute:
        print("\n[dry run] re-run with --execute to publish.")
        return

    create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, private=PRIVATE, exist_ok=True)
    print("\nrepo ready, uploading...")
    api.upload_folder(
        folder_path=str(ROOT),
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message="update to twelve-market release; prune stale three-city data",
        ignore_patterns=IGNORE,
        delete_patterns=PRUNE,
    )
    print(f"done -> https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
