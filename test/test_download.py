"""Tests for kitscenes.download — local tar extraction."""

from __future__ import annotations

import csv
import shutil
import tarfile
from pathlib import Path

from kitscenes.download.downloader import KITScenesDownloader


def _write_manifest(root: Path, rows: list[dict[str, str]]) -> None:
    manifest = root / "data" / "sequence_archives.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sequence_id",
                "split",
                "archive_path",
                "archive_sha256",
                "archive_size_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_scene_tar(tar_path: Path, scene_id: str) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    payload = tar_path.parent / "_payload" / scene_id
    payload.mkdir(parents=True)
    (payload / "timestamp.reference.txt").write_text("1\n")
    with tarfile.open(tar_path, "w") as tf:
        tf.add(payload, arcname=scene_id)
    shutil.rmtree(payload.parent)


def test_extract_local(tmp_path: Path) -> None:
    root = tmp_path / "kitscenes"
    scene_id = "aaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    archive_path = f"data/val/{scene_id}.tar"
    tar_path = root / archive_path

    _write_scene_tar(tar_path, scene_id)
    _write_manifest(
        root,
        [
            {
                "sequence_id": scene_id,
                "split": "val",
                "archive_path": archive_path,
                "archive_sha256": "unused",
                "archive_size_bytes": str(tar_path.stat().st_size),
            }
        ],
    )

    dl = KITScenesDownloader(output_dir=root, token=False)
    dl.extract_local([scene_id], verify=False, delete_tars=True, jobs=2)

    scene_dir = root / "data" / "val" / scene_id
    assert scene_dir.is_dir()
    assert (scene_dir / "timestamp.reference.txt").is_file()
    assert not tar_path.is_file()


def test_extract_python_fallback(tmp_path: Path, monkeypatch) -> None:
    real_which = shutil.which

    def fake_which(name: str) -> str | None:
        if name == "tar":
            return None
        return real_which(name)

    monkeypatch.setattr(
        "kitscenes.download.downloader.shutil.which",
        fake_which,
    )

    root = tmp_path / "kitscenes"
    scene_id = "aaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    archive_path = f"data/val/{scene_id}.tar"
    tar_path = root / archive_path

    _write_scene_tar(tar_path, scene_id)
    _write_manifest(
        root,
        [
            {
                "sequence_id": scene_id,
                "split": "val",
                "archive_path": archive_path,
                "archive_sha256": "unused",
                "archive_size_bytes": str(tar_path.stat().st_size),
            }
        ],
    )

    dl = KITScenesDownloader(output_dir=root, token=False)
    dl.extract_local([scene_id], verify=False, delete_tars=True)

    assert (root / "data" / "val" / scene_id / "timestamp.reference.txt").is_file()
