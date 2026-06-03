"""Tests for kitscenes.pack_sqfs."""

from __future__ import annotations

from pathlib import Path

import pytest

from kitscenes.pack_sqfs.builder import SquashFsBuilder, _split_from_tar


def test_output_dir_and_split_images(tmp_path: Path) -> None:
    root = tmp_path / "kitscenes"
    (root / "data" / "val").mkdir(parents=True)
    builder = SquashFsBuilder(root)
    assert builder.output_dir == tmp_path / "kitscenes_sqfs"
    assert builder.output_path_for_split("train") == tmp_path / "kitscenes_sqfs" / "train.sqfs"


def test_tar_paths_by_split(tmp_path: Path) -> None:
    root = tmp_path / "kitscenes"
    (root / "data" / "val").mkdir(parents=True)
    (root / "data" / "train").mkdir(parents=True)
    (root / "data" / "val" / "a.tar").write_bytes(b"x")
    (root / "data" / "train" / "b.tar").write_bytes(b"y")

    builder = SquashFsBuilder(root)
    by_split = builder.tar_paths_by_split()
    assert set(by_split) == {"train", "val"}
    assert len(by_split["train"]) == 1
    assert len(by_split["val"]) == 1


def test_split_from_tar(tmp_path: Path) -> None:
    root = tmp_path / "kitscenes"
    tar = root / "data" / "test_e2e" / "scene.tar"
    tar.parent.mkdir(parents=True)
    tar.write_bytes(b"x")
    assert _split_from_tar(tar, root) == "test_e2e"


def test_build_requires_ratarmount(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "kitscenes"
    (root / "data" / "val").mkdir(parents=True)
    (root / "data" / "val" / "scene.tar").write_bytes(b"x")

    monkeypatch.setattr(
        "kitscenes.pack_sqfs.builder.shutil.which",
        lambda name: "/usr/bin/mksquashfs" if name == "mksquashfs" else None,
    )
    with pytest.raises(RuntimeError, match="ratarmount"):
        SquashFsBuilder(root)
