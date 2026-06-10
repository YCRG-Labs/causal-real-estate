#!/usr/bin/env bash
# Read-only sync of the full-panel embeddings from the Brev box into
# data/processed/. Run this yourself; it only pulls files, never launches compute.
#
#   bash pull_embeddings.sh                 # uses defaults below
#   BREV_HOST=cong bash pull_embeddings.sh  # override instance name
#
# If you don't have the brev ssh alias set up, run `brev ls` to see your
# instances and `brev ssh-config` (or `brev shell <name>`) once so the host
# resolves, then re-run this.
set -euo pipefail

BREV_HOST="${BREV_HOST:-cong}"
REMOTE_DIR="${REMOTE_DIR:-causal-real-estate/data/processed}"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/data/processed"

mkdir -p "$LOCAL_DIR"

echo "==> listing remote embeddings on '$BREV_HOST:$REMOTE_DIR'"
ssh "$BREV_HOST" "ls -lh $REMOTE_DIR/*_embeddings*.parquet" || {
  echo "!! could not list remote files. Check: brev ls   /   ssh $BREV_HOST 'echo ok'"
  exit 1
}

echo "==> syncing *_embeddings.parquet (mpnet) + *_embeddings_*MiniLM*.parquet -> $LOCAL_DIR"
rsync -avz --progress \
  "$BREV_HOST:$REMOTE_DIR/"'*_embeddings.parquet' \
  "$BREV_HOST:$REMOTE_DIR/"'*_embeddings_*MiniLM*.parquet' \
  "$LOCAL_DIR/" || {
  echo "!! rsync failed. If brev uses a custom port/proxy, try:"
  echo "   rsync -avz -e 'brev ssh' $BREV_HOST:$REMOTE_DIR/'*_embeddings.parquet' $LOCAL_DIR/"
  exit 1
}

echo "==> done. now verify with:"
echo "   python3 data/scripts/check_embeddings.py --all_12"
