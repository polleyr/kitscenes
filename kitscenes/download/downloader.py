"""KITScenes dataset downloader.

Downloads scene tar files from HuggingFace Hub, verifies SHA-256 checksums,
extracts them in place, and removes the tars automatically.

The manifest is read from ``sequence_archives.csv`` in the HuggingFace repo root
and cached locally by ``huggingface_hub`` — no bundled manifest file is needed.

Typical usage::

    from kitscenes.download import KITScenesDownloader

    dl = KITScenesDownloader(output_dir="/data/kitscenes")
    scenes = dl.select_scenes(split="train", max_gb=50.0)
    dl.download(scenes)

Or via the CLI::

    python -m kitscenes.download /data/kitscenes --split train --max-gb 50
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import tqdm

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "data/sequence_archives.csv"
_DATA_SUBDIR = "data"
_DEFAULT_REPO_ID = "KIT-MRT/KITScenes-Multimodal"

_print_lock = threading.Lock()


class KITScenesDownloader:
    """Downloads KITScenes scenes from HuggingFace Hub.

    For each requested scene the downloader:

    1. Downloads the scene's ``.tar`` file to a temporary directory.
    2. Verifies the SHA-256 checksum against ``sequence_archives.csv``.
    3. Extracts the tar into ``output_dir/data/<split>/`` (mirrors HuggingFace layout).
    4. Deletes the tar (the temporary directory is cleaned up automatically).

    The manifest (``sequence_archives.csv``) is fetched from the HuggingFace
    repo on first use and cached locally by ``huggingface_hub``.

    Args:
        output_dir: Root directory where extracted scene folders will be placed.
            Created automatically if it does not exist.
        repo_id: HuggingFace Hub repository ID for the dataset.
        token: HuggingFace user access token.  ``None`` uses the token saved by
            ``huggingface-cli login``; ``False`` disables authentication.
        staging_dir: Directory used for temporary tar downloads before extraction.
            Defaults to ``<output_dir>/.kitscenes_staging``.  Can also be set via
            the ``KITSCENES_DOWNLOAD_STAGING`` environment variable.  Use this when
            ``/tmp`` is too small for multi-GB scene archives.
    """

    def __init__(
        self,
        output_dir: str | Path,
        repo_id: str = _DEFAULT_REPO_ID,
        token: str | bool | None = None,
        staging_dir: str | Path | None = None,
    ) -> None:
        import huggingface_hub

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir = _resolve_staging_dir(self.output_dir, staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.repo_id = repo_id
        self.token = token
        self._api = huggingface_hub.HfApi(token=token)
        self._manifest: Dict[str, dict] = self._load_manifest(huggingface_hub)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def available_splits(self) -> list[str]:
        """All split names present in the manifest, sorted."""
        return sorted({info["split"] for info in self._manifest.values()})

    @property
    def all_scene_ids(self) -> list[str]:
        """All scene IDs listed in the manifest, sorted."""
        return sorted(self._manifest.keys())

    def scene_ids_for_split(self, split: str) -> list[str]:
        """Return all scene IDs belonging to *split*, sorted."""
        if split not in self.available_splits:
            raise ValueError(
                f"Unknown split {split!r}. "
                f"Available splits: {self.available_splits}"
            )
        return sorted(
            sid for sid, info in self._manifest.items()
            if info["split"] == split
        )

    def total_size_gb(self, scene_ids: list[str]) -> float:
        """Return the cumulative download size in GB for *scene_ids*."""
        return (
            sum(self._manifest[sid]["size_bytes"] for sid in scene_ids)
            / 1000 ** 3
        )

    def select_scenes(
        self,
        *,
        split: Optional[str] = None,
        scene_ids: Optional[list[str]] = None,
        max_gb: Optional[float] = None,
        skip_existing: bool = True,
    ) -> list[str]:
        """Build a filtered list of scene IDs ready for :meth:`download`.

        Args:
            split: If given, start from all scenes in this split.
            scene_ids: Explicit list of scene IDs (takes precedence over *split*).
            max_gb: Stop adding scenes once the cumulative size would exceed this
                limit.  Scenes are taken in sorted order.
            skip_existing: Skip scenes whose output directory already exists
                (i.e. scenes that have already been extracted).

        Returns:
            Ordered list of scene IDs to pass to :meth:`download`.
        """
        if scene_ids is not None:
            unknown = [s for s in scene_ids if s not in self._manifest]
            if unknown:
                raise ValueError(
                    f"Scene ID(s) not found in manifest: {unknown}"
                )
            selected = list(scene_ids)
        elif split is not None:
            selected = self.scene_ids_for_split(split)
        else:
            selected = self.all_scene_ids

        if skip_existing:
            before = len(selected)
            selected = [
                s for s in selected
                if not self._scene_dir(s).is_dir()
            ]
            skipped = before - len(selected)
            if skipped:
                print(f"Skipping {skipped} already-extracted scene(s).")

        if max_gb is not None:
            kept, cumulative_gb = [], 0.0
            for sid in selected:
                scene_gb = self._manifest[sid]["size_bytes"] / 1000 ** 3
                if cumulative_gb + scene_gb > max_gb:
                    break
                kept.append(sid)
                cumulative_gb += scene_gb
            dropped = len(selected) - len(kept)
            if dropped:
                print(
                    f"max_gb={max_gb:.1f}: keeping {len(kept)} scene(s) "
                    f"({cumulative_gb:.2f} GB), dropping {dropped}."
                )
            selected = kept

        return selected

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(
        self,
        scene_ids: list[str],
        *,
        dry_run: bool = False,
    ) -> None:
        """Download, verify, extract, and clean up tars for *scene_ids*.

        Args:
            scene_ids: List returned by :meth:`select_scenes`.
            dry_run: Print what would be downloaded without actually downloading.
        """
        if not scene_ids:
            print("Nothing to download.")
            return

        total_gb = self.total_size_gb(scene_ids)

        print(f"Scenes selected : {len(scene_ids)}")
        print(f"Total size      : {total_gb:.2f} GB")
        print(f"Output directory: {self.output_dir}")

        if dry_run:
            print("\n[dry-run] The following tars would be downloaded and extracted:")
            for sid in scene_ids:
                info = self._manifest[sid]
                print(
                    f"  {sid}  "
                    f"({info['size_bytes'] / 1000 ** 3:.2f} GB)  "
                    f"[{info['split']}]"
                )
            return

        print()
        failed: list[tuple[str, Exception]] = []
        for i, scene_id in enumerate(scene_ids, 1):
            info = self._manifest[scene_id]
            print(
                f"[{i}/{len(scene_ids)}] {scene_id}  "
                f"({info['size_bytes'] / 1000 ** 3:.2f} GB)"
            )
            try:
                self._process_scene(scene_id, info)
            except Exception as exc:
                logger.error("Failed to download %s: %s", scene_id, exc)
                failed.append((scene_id, exc))

        print()
        if failed:
            print(f"Completed with {len(failed)} error(s):")
            for sid, exc in failed:
                print(f"  FAILED  {sid}: {exc}")
        else:
            print(f"Done. {len(scene_ids)} scene(s) extracted to {self.output_dir}")

    def extract_local(
        self,
        scene_ids: list[str],
        *,
        jobs: int = 1,
        verify: bool = True,
        delete_tars: bool = True,
        dry_run: bool = False,
    ) -> None:
        """Verify and extract scene tars already present under *output_dir*.

        Use after bulk-fetching archives with ``hf download``, git-xet, etc.
        Expects the HuggingFace layout: ``data/<split>/*.tar`` plus
        ``data/sequence_archives.csv``.
        """
        if not scene_ids:
            print("Nothing to extract.")
            return

        ready, missing_tars = [], []
        for sid in scene_ids:
            if (self.output_dir / self._manifest[sid]["filename"]).is_file():
                ready.append(sid)
            else:
                missing_tars.append(sid)

        if missing_tars:
            print(f"Skipping {len(missing_tars)} scene(s) with no local tar.")

        if not ready:
            print("No local tars to extract.")
            return

        total_gb = self.total_size_gb(ready)
        print(f"Scenes to extract: {len(ready)}")
        print(f"Tar size          : {total_gb:.2f} GB")
        print(f"Output directory  : {self.output_dir}")
        print(f"Parallel jobs     : {jobs}")

        if dry_run:
            print("\n[dry-run] Would extract:")
            for sid in ready:
                info = self._manifest[sid]
                print(
                    f"  {sid}  "
                    f"({info['size_bytes'] / 1000 ** 3:.2f} GB)  "
                    f"[{info['split']}]"
                )
            return

        failed: list[tuple[str, Exception]] = []
        progress_lock = threading.Lock()
        completed_bytes = 0
        batch_t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            futures = {
                pool.submit(
                    self._extract_local_scene,
                    sid,
                    verify=verify,
                    delete_tar=delete_tars,
                ): sid
                for sid in ready
            }
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    timing = future.result()
                    with progress_lock:
                        completed_bytes += timing.size_bytes
                        avg_gbps = _gb_per_s(
                            completed_bytes,
                            time.perf_counter() - batch_t0,
                        )
                    print(
                        f"  ✔ {sid}  "
                        f"({_format_scene_timing(timing)}, avg {avg_gbps:.2f} GB/s)"
                    )
                except Exception as exc:
                    logger.error("Failed to extract %s: %s", sid, exc)
                    failed.append((sid, exc))
                    print(f"  ✗ {sid}: {exc}")

        print()
        if failed:
            print(f"Completed with {len(failed)} error(s).")
        else:
            elapsed = time.perf_counter() - batch_t0
            overall_gbps = _gb_per_s(completed_bytes, elapsed)
            print(
                f"Done. {len(ready)} scene(s) extracted to {self.output_dir} "
                f"({elapsed:.1f}s, {overall_gbps:.2f} GB/s overall)"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self, huggingface_hub) -> Dict[str, dict]:
        local = self.output_dir / _MANIFEST_FILENAME
        if local.is_file():
            logger.info("Using local manifest %s", local)
            return _parse_manifest_csv(str(local))
        return self._fetch_manifest(huggingface_hub)

    def _fetch_manifest(self, huggingface_hub) -> Dict[str, dict]:
        """Download (and cache) sequence_archives.csv, return parsed dict."""
        logger.info("Fetching manifest from %s ...", self.repo_id)
        try:
            csv_path = huggingface_hub.hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=_MANIFEST_FILENAME,
                local_dir=self.output_dir,
                token=self.token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not fetch {_MANIFEST_FILENAME!r} from {self.repo_id!r}.\n"
                f"  {exc}\n"
                "Make sure you are authenticated (huggingface-cli login) and that "
                "the repository exists."
            ) from exc

        return _parse_manifest_csv(csv_path)

    def _process_scene(self, scene_id: str, info: dict) -> None:
        """Download → verify SHA-256 → extract → delete tar."""
        import huggingface_hub

        filename: str = info["filename"]
        expected_sha256: str = info["sha256"]
        size_bytes: int = info["size_bytes"]

        with tempfile.TemporaryDirectory(
            prefix="kitscenes_dl_",
            dir=self.staging_dir,
        ) as tmp_dir:
            # --- Download ---
            print(f"  ↓ Downloading ...")
            huggingface_hub.hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=tmp_dir,
                token=self.token,
            )
            # hf_hub_download mirrors the repo path under local_dir
            tar_path = Path(tmp_dir) / filename
            if not tar_path.exists():
                # Fallback: find any .tar written under the temp tree
                candidates = list(Path(tmp_dir).rglob("*.tar"))
                if not candidates:
                    raise FileNotFoundError(
                        f"Downloaded tar not found under {tmp_dir}"
                    )
                tar_path = candidates[0]

            # --- Verify SHA-256 + extract ---
            print("  ✓ Verifying SHA-256 ...")
            split_dir = self.output_dir / _DATA_SUBDIR / info["split"]
            timing = _extract_tar_file(
                tar_path,
                split_dir,
                expected_sha256=expected_sha256,
                size_bytes=size_bytes,
                scene_id=scene_id,
            )

            # tar_path lives inside tmp_dir → deleted automatically on context exit.

        print(f"  ✔ Done  ({_format_scene_timing(timing)})")

    def _extract_local_scene(
        self,
        scene_id: str,
        *,
        verify: bool,
        delete_tar: bool,
    ) -> "_ExtractTiming":
        info = self._manifest[scene_id]
        tar_path = self.output_dir / info["filename"]
        if not tar_path.is_file():
            raise FileNotFoundError(f"Tar not found: {tar_path}")

        split_dir = self.output_dir / _DATA_SUBDIR / info["split"]
        timing = _extract_tar_file(
            tar_path,
            split_dir,
            expected_sha256=info["sha256"] if verify else None,
            size_bytes=info["size_bytes"],
            scene_id=scene_id,
        )
        if delete_tar:
            tar_path.unlink()
        return timing

    def _scene_dir(self, scene_id: str) -> Path:
        info = self._manifest[scene_id]
        return self.output_dir / _DATA_SUBDIR / info["split"] / scene_id


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _resolve_staging_dir(
    output_dir: Path,
    staging_dir: str | Path | None,
) -> Path:
    if staging_dir is not None:
        return Path(staging_dir)
    env = os.environ.get("KITSCENES_DOWNLOAD_STAGING")
    if env:
        return Path(env)
    return output_dir / ".kitscenes_staging"


def _parse_manifest_csv(csv_path: str) -> Dict[str, dict]:
    """Parse ``sequence_archives.csv`` into the internal manifest dict.

    Expected columns (order-independent):
        sequence_id, split, archive_path, archive_sha256, archive_size_bytes
    """
    manifest: Dict[str, dict] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            manifest[row["sequence_id"]] = {
                "split": row["split"],
                "filename": row["archive_path"],
                "sha256": row["archive_sha256"],
                "size_bytes": int(row["archive_size_bytes"]),
            }
    return manifest


@dataclass(frozen=True)
class _ExtractTiming:
    size_bytes: int
    elapsed_s: float


def _gb_per_s(size_bytes: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return size_bytes / elapsed_s / 1000 ** 3


def _format_scene_timing(timing: _ExtractTiming) -> str:
    return (
        f"{timing.elapsed_s:.1f}s, "
        f"{_gb_per_s(timing.size_bytes, timing.elapsed_s):.2f} GB/s"
    )


def _log_extract_start(scene_id: str, label: str) -> None:
    name = scene_id or label
    with _print_lock:
        print(f"  ↗ Extracting {name} ...")


def _scene_label(scene_id: str, tar_path: Path) -> str:
    label = scene_id or tar_path.stem
    if len(label) > 36:
        label = label[:8] + "…"
    return label


def _extract_tar_file(
    tar_path: Path,
    split_dir: Path,
    *,
    expected_sha256: str | None = None,
    size_bytes: int | None = None,
    scene_id: str = "",
) -> _ExtractTiming:
    label = _scene_label(scene_id, tar_path)
    tar_size = size_bytes or tar_path.stat().st_size
    if expected_sha256 is not None:
        actual_sha256 = _sha256_file(
            tar_path,
            total_bytes=tar_size,
            desc=f"sha256 {label}",
        )
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {label}:\n"
                f"  expected : {expected_sha256}\n"
                f"  computed : {actual_sha256}"
            )

    split_dir.mkdir(parents=True, exist_ok=True)
    _log_extract_start(scene_id, label)
    t0 = time.perf_counter()
    if shutil.which("tar") is not None:
        _extract_tar_subprocess(tar_path, split_dir)
    else:
        logger.warning("tar(1) not found — falling back to Python tarfile.")
        _extract_tar_python(tar_path, split_dir)
    return _ExtractTiming(size_bytes=tar_size, elapsed_s=time.perf_counter() - t0)


def _extract_tar_subprocess(tar_path: Path, split_dir: Path) -> None:
    """Extract via GNU tar (silent; timing printed on completion)."""
    tar_bin = shutil.which("tar")
    assert tar_bin is not None

    result = subprocess.run(
        [tar_bin, "xf", str(tar_path.resolve()), "-C", str(split_dir.resolve())],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        err = result.stderr.strip()
        raise RuntimeError(
            f"tar extract failed (exit {result.returncode}): {err}"
        )


def _extract_tar_python(tar_path: Path, split_dir: Path) -> None:
    with tarfile.open(tar_path) as tf:
        tf.extractall(split_dir)


def _sha256_file(path: Path, total_bytes: int, *, desc: str = "sha256") -> str:
    """Compute the SHA-256 hex digest of *path* with a tqdm progress bar."""
    digest = hashlib.sha256()
    with tqdm.tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"    {desc}",
        leave=False,
    ) as bar:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                bar.update(len(chunk))
    return digest.hexdigest()
