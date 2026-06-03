#!/usr/bin/env bash
# Mount per-split SquashFS images to an ephemeral tree and set KITSCENES_ROOT.
#
# Mount layout (always):
#   /tmp/$USER/sqfs_mnt/kitscenes/data/<split>/<scene_id>/...
#
# Uses squashfuse (FUSE, no root). Source this script so export takes effect:
#
#   source scripts/mount_kitscenes_sqfs.sh /data/kitscenes_sqfs
#   python -c "from kitscenes import KITScenesDataset; print(KITScenesDataset())"
#
# Unmount:
#   fusermount -u /tmp/$USER/sqfs_mnt/kitscenes/data/train

set -euo pipefail

if ! command -v squashfuse >/dev/null 2>&1; then
  echo "squashfuse not found on PATH (install squashfuse package)." >&2
  return 2 2>/dev/null || exit 1
fi

SQFS_DIR="${1:-${SQFS_DIR:-}}"
if [[ -z "$SQFS_DIR" ]]; then
  echo "Usage: source $0 <SQFS_DIR>" >&2
  echo "  e.g. source $0 /data/kitscenes_sqfs" >&2
  echo "  or set SQFS_DIR" >&2
  return 2 2>/dev/null || exit 1
fi

if [[ ! -d "$SQFS_DIR" ]]; then
  echo "SquashFS directory not found: $SQFS_DIR" >&2
  return 1 2>/dev/null || exit 1
fi

KITSCENES_ROOT="/tmp/${USER:?}/sqfs_mnt/kitscenes"
mkdir -p "$KITSCENES_ROOT/data"

shopt -s nullglob
sqfs_files=("$SQFS_DIR"/*.sqfs)
shopt -u nullglob

if [[ ${#sqfs_files[@]} -eq 0 ]]; then
  echo "No .sqfs files in $SQFS_DIR" >&2
  return 1 2>/dev/null || exit 1
fi

mounted=()
for sqfs in "${sqfs_files[@]}"; do
  split="$(basename "$sqfs" .sqfs)"
  mountpoint="$KITSCENES_ROOT/data/$split"
  mkdir -p "$mountpoint"

  if mountpoint -q "$mountpoint"; then
    echo "Already mounted: $mountpoint"
    mounted+=("$split")
    continue
  fi

  echo "squashfuse $sqfs -> $mountpoint"
  if ! squashfuse "$sqfs" "$mountpoint"; then
    echo "Failed to mount $sqfs" >&2
    for prev in "${mounted[@]}"; do
      fusermount -u "$KITSCENES_ROOT/data/$prev" 2>/dev/null || true
    done
    return 1 2>/dev/null || exit 1
  fi
  mounted+=("$split")
done

export KITSCENES_ROOT

echo ""
echo "Mounted ${#mounted[@]} split(s) under $KITSCENES_ROOT/data/<split>/"
echo "KITSCENES_ROOT=$KITSCENES_ROOT"
