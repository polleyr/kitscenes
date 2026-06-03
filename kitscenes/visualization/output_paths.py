"""Resolve and validate visualization output paths."""

from __future__ import annotations

import os
from pathlib import Path

from kitscenes.constants import KITSCENES_ROOT_ENV_VAR, KITSCENES_VIZ_OUTPUT_ENV_VAR


def resolve_viz_output_dir(output_dir: Path | str | None = None) -> Path:
    """Resolve projection output directory from *output_dir* or ``$KITSCENES_VIZ_OUTPUT``.

    Raises:
        ValueError: If neither is set, or if the path lies inside ``$KITSCENES_ROOT``.
    """
    if output_dir is not None:
        path = Path(output_dir)
    else:
        env_value = os.environ.get(KITSCENES_VIZ_OUTPUT_ENV_VAR)
        if not env_value:
            raise ValueError(
                f"Projection output directory is required. Pass output_dir= or set "
                f"${KITSCENES_VIZ_OUTPUT_ENV_VAR}."
            )
        path = Path(env_value)
    return guard_output_path(path, kind="directory")


def resolve_viz_output_file(
    output_path: Path | str | None = None,
    *,
    default_name: str = "map_projection.mp4",
) -> Path:
    """Resolve video output file from *output_path* or ``$KITSCENES_VIZ_OUTPUT``."""
    if output_path is not None:
        path = Path(output_path)
    else:
        env_value = os.environ.get(KITSCENES_VIZ_OUTPUT_ENV_VAR)
        if not env_value:
            raise ValueError(
                f"Video output path is required. Pass output_path= or set "
                f"${KITSCENES_VIZ_OUTPUT_ENV_VAR}."
            )
        env_path = Path(env_value)
        path = env_path if env_path.suffix else env_path / default_name
    return guard_output_path(path, kind="file")


def guard_output_path(path: Path, *, kind: str) -> Path:
    """Ensure *path* is resolved and not under ``$KITSCENES_ROOT``."""
    resolved = path.expanduser().resolve()
    dataset_root = os.environ.get(KITSCENES_ROOT_ENV_VAR)
    if dataset_root:
        root = Path(dataset_root).expanduser().resolve()
        if resolved == root or root in resolved.parents:
            raise ValueError(
                f"Refusing to write visualization {kind} inside the dataset root "
                f"({root}). Set ${KITSCENES_VIZ_OUTPUT_ENV_VAR} or pass an external "
                f"output path."
            )
    return resolved
