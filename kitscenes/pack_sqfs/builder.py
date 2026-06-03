"""Per-split SquashFS builder using ratarmount + mksquashfs.

Each split (``train``, ``val``, …) becomes its own ``<split>.sqfs`` with scene
UUID folders at the image root.  Mount with ``scripts/mount_kitscenes_sqfs.sh``
to ``/tmp/$USER/sqfs_mnt/kitscenes/data/<split>/`` (sets ``$KITSCENES_ROOT``).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

_DATA_SUBDIR = "data"
_DEFAULT_OUTPUT_DIRNAME = "kitscenes_sqfs"


class SquashFsBuilder:
    """Build ``<split>.sqfs`` images next to the dataset root."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        output_dir: str | Path | None = None,
        compression: str = "lz4",
        processors: int = 1,
    ) -> None:
        self.dataset_root = Path(dataset_root).resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        if output_dir is None:
            output_dir = self.dataset_root.parent / _DEFAULT_OUTPUT_DIRNAME
        self.output_dir = Path(output_dir).resolve()
        self.compression = compression
        self.processors = max(1, processors)

        self._ratarmount = _require_tool("ratarmount")
        _require_tool("mksquashfs")

    @property
    def tar_paths(self) -> list[Path]:
        """Scene archives under ``data/<split>/*.tar``, sorted."""
        data_dir = self.dataset_root / _DATA_SUBDIR
        if not data_dir.is_dir():
            return []
        return sorted(data_dir.rglob("*.tar"))

    def tar_paths_by_split(self) -> dict[str, list[Path]]:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for tar_path in self.tar_paths:
            split = _split_from_tar(tar_path, self.dataset_root)
            grouped[split].append(tar_path)
        return dict(sorted(grouped.items()))

    def output_path_for_split(self, split: str) -> Path:
        return self.output_dir / f"{split}.sqfs"

    def build(
        self,
        *,
        splits: list[str] | None = None,
        delete_tars: bool = True,
        force: bool = False,
        dry_run: bool = False,
    ) -> None:
        by_split = self.tar_paths_by_split()
        if splits is not None:
            unknown = [s for s in splits if s not in by_split]
            if unknown:
                raise ValueError(
                    f"No tars for split(s) {unknown}. "
                    f"Available: {sorted(by_split)}"
                )
            by_split = {s: by_split[s] for s in splits}

        if not by_split:
            print(f"No .tar files found under {self.dataset_root / _DATA_SUBDIR}")
            return

        total_tars = sum(len(v) for v in by_split.values())
        total_gb = sum(
            p.stat().st_size for paths in by_split.values() for p in paths
        ) / 1000 ** 3

        print(f"Dataset root : {self.dataset_root}")
        print(f"Output dir   : {self.output_dir}")
        print(f"Scene tars   : {total_tars}  ({total_gb:.2f} GB)")
        print(f"Splits       : {', '.join(by_split)}")
        print(f"Compression  : {self.compression}")
        print(f"Processors   : {self.processors}")

        if dry_run:
            print("\n[dry-run] Would build:")
            for split, paths in by_split.items():
                print(f"  {self.output_path_for_split(split)}  ({len(paths)} tars)")
                for tar_path in paths:
                    print(f"    {tar_path.relative_to(self.dataset_root)}")
            return

        if force and not dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for split in by_split:
                sqfs = self.output_path_for_split(split)
                if sqfs.exists():
                    sqfs.unlink()

        failed: list[tuple[Path, Exception]] = []
        batch_t0 = time.perf_counter()
        completed_bytes = 0

        for split, paths in by_split.items():
            print(f"\n=== split: {split} -> {self.output_path_for_split(split).name} ===")
            split_t0 = time.perf_counter()
            split_bytes = 0

            for i, tar_path in enumerate(paths, 1):
                rel = tar_path.relative_to(self.dataset_root)
                size_bytes = tar_path.stat().st_size
                print(f"[{i}/{len(paths)}] {rel}  ({size_bytes / 1000 ** 3:.2f} GB)")
                try:
                    elapsed = self._append_tar(tar_path, self.output_path_for_split(split))
                    split_bytes += size_bytes
                    completed_bytes += size_bytes
                    split_elapsed = time.perf_counter() - split_t0
                    avg_gbps = split_bytes / split_elapsed / 1000 ** 3 if split_elapsed > 0 else 0.0
                    scene_gbps = size_bytes / elapsed / 1000 ** 3 if elapsed > 0 else 0.0
                    print(
                        f"  ✔ appended  ({elapsed:.1f}s, {scene_gbps:.2f} GB/s, "
                        f"split avg {avg_gbps:.2f} GB/s)"
                    )
                    if delete_tars:
                        tar_path.unlink()
                except Exception as exc:
                    failed.append((tar_path, exc))
                    print(f"  ✗ {rel}: {exc}")

            split_elapsed = time.perf_counter() - split_t0
            if split_bytes:
                print(
                    f"Split {split} done: {self.output_path_for_split(split)} "
                    f"({split_elapsed:.1f}s, "
                    f"{split_bytes / split_elapsed / 1000 ** 3:.2f} GB/s)"
                )

        print()
        if failed:
            print(f"Completed with {len(failed)} error(s).")
        else:
            elapsed = time.perf_counter() - batch_t0
            overall = completed_bytes / elapsed / 1000 ** 3 if elapsed > 0 else 0.0
            print(
                f"Done. Images in {self.output_dir}  "
                f"({elapsed:.1f}s, {overall:.2f} GB/s overall)"
            )
            print(f"Mount: source scripts/mount_kitscenes_sqfs.sh {self.output_dir}")

    def _append_tar(self, tar_path: Path, output_path: Path) -> float:
        _split_from_tar(tar_path, self.dataset_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="kitscenes_sqfs_stage_") as tmp:
            mount_point = Path(tmp)
            self._run_ratarmount(tar_path, mount_point)
            try:
                print(f"  ↗ Appending {tar_path.stem} ...")
                self._run_mksquashfs(mount_point, output_path)
            finally:
                _unmount(mount_point)

        return time.perf_counter() - t0

    def _run_ratarmount(self, tar_path: Path, mount_point: Path) -> None:
        subprocess.run(
            [self._ratarmount, str(tar_path), str(mount_point)],
            check=True,
            capture_output=True,
            text=True,
        )

    def _run_mksquashfs(self, source_dir: Path, output_path: Path) -> None:
        cmd = [
            "mksquashfs",
            str(source_dir),
            str(output_path),
            "-processors",
            str(self.processors),
            "-no-progress",
        ]
        if self.compression in ("none", "store"):
            cmd.append("-no-compression")
        else:
            cmd.extend(["-comp", self.compression])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"mksquashfs failed (exit {result.returncode}): {err}")


def _split_from_tar(tar_path: Path, dataset_root: Path) -> str:
    rel = tar_path.relative_to(dataset_root)
    if len(rel.parts) < 3 or rel.parts[0] != _DATA_SUBDIR:
        raise ValueError(f"Unexpected tar location (want data/<split>/…): {tar_path}")
    return rel.parts[1]


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"Required tool {name!r} not found on PATH. "
            f"Install it before running pack_sqfs."
        )
    return path


def _unmount(mount_point: Path) -> None:
    for cmd in (
        ["fusermount", "-u", str(mount_point)],
        ["ratarmount", "-u", str(mount_point)],
    ):
        if shutil.which(cmd[0]) is None:
            continue
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return
    raise RuntimeError(f"Could not unmount {mount_point}")
