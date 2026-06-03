#!/usr/bin/env bash
# Union-mount scene tars per split via ratarmount and set KITSCENES_ROOT.
#
# Input: dataset root containing data/<split>/*.tar (same path used for hf download).
# Mount layout:
#   /tmp/$USER/ratarmount/kitscenes/data/<split>/<scene_uuid>/...
#
# Source this script so export takes effect:
#
#   export KITSCENES_ROOT=/data/kitscenes
#   source scripts/mount_kitscenes_tars.sh
#   python -c "from kitscenes import KITScenesDataset; print(KITScenesDataset(split='val'))"
#
# Unmount:
#   ratarmount -u /tmp/$USER/ratarmount/kitscenes/data/val

set -euo pipefail

if ! command -v ratarmount >/dev/null 2>&1; then
  echo "ratarmount not found on PATH (pip install -e \".[mount]\")." >&2
  return 2 2>/dev/null || exit 1
fi

TAR_ROOT="${1:-${KITSCENES_ROOT:-}}"
if [[ -z "$TAR_ROOT" ]]; then
  echo "Usage: source $0 [TAR_ROOT]" >&2
  echo "  e.g. source $0 /data/kitscenes" >&2
  echo "  or set KITSCENES_ROOT to the dataset root with data/<split>/*.tar" >&2
  return 2 2>/dev/null || exit 1
fi

DATA_DIR="$TAR_ROOT/data"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data directory not found: $DATA_DIR" >&2
  return 1 2>/dev/null || exit 1
fi

KITSCENES_ROOT="/tmp/${USER:?}/ratarmount/kitscenes"
mkdir -p "$KITSCENES_ROOT/data"

mounted=()
shopt -s nullglob
for split_dir in "$DATA_DIR"/*/; do
  [[ -d "$split_dir" ]] || continue
  split="$(basename "$split_dir")"
  tars=("$split_dir"*.tar)
  [[ ${#tars[@]} -eq 0 ]] && continue

  mountpoint="$KITSCENES_ROOT/data/$split"
  mkdir -p "$mountpoint"

  if mountpoint -q "$mountpoint"; then
    echo "Already mounted: $mountpoint"
    mounted+=("$split")
    continue
  fi

  echo "ratarmount ${#tars[@]} tar(s) -> $mountpoint"
  if ! ratarmount "${tars[@]}" "$mountpoint"; then
    echo "Failed to mount split $split" >&2
    for prev in "${mounted[@]}"; do
      ratarmount -u "$KITSCENES_ROOT/data/$prev" 2>/dev/null || \
        fusermount -u "$KITSCENES_ROOT/data/$prev" 2>/dev/null || true
    done
    return 1 2>/dev/null || exit 1
  fi
  mounted+=("$split")
done
shopt -u nullglob

if [[ ${#mounted[@]} -eq 0 ]]; then
  echo "No .tar files found under $DATA_DIR/<split>/" >&2
  return 1 2>/dev/null || exit 1
fi

export KITSCENES_ROOT

echo ""
echo "Mounted ${#mounted[@]} split(s) under $KITSCENES_ROOT/data/<split>/"
echo "KITSCENES_ROOT=$KITSCENES_ROOT"
