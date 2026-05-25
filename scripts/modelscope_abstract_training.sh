#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/workspace/PHYSMOL}"
OUT_DIR="${OUT_DIR:-$PROJECT_DIR/checkpoints/abstract_training}"
DATASET_ID="${DATASET_ID:-}"
SUBSET_NAME="${SUBSET_NAME:-}"
SPLIT="${SPLIT:-train}"
LIMIT="${LIMIT:-1000}"

cd "$PROJECT_DIR"
python -m pip install -e ".[modelscope]"

ARGS=(--out-dir "$OUT_DIR")
if [[ -n "$DATASET_ID" ]]; then
  ARGS+=(--modelscope-dataset "$DATASET_ID" --split "$SPLIT" --limit "$LIMIT")
  if [[ -n "$SUBSET_NAME" ]]; then
    ARGS+=(--subset-name "$SUBSET_NAME")
  fi
else
  ARGS+=(--use-builtin)
fi

python -m physmol.abstract_train "${ARGS[@]}"
