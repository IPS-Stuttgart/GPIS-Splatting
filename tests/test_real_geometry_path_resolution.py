from __future__ import annotations

from pathlib import Path

import pytest

from gpis_splatting.real_geometry import resolve_optional_scene_file, resolve_scene_file


def test_resolve_scene_file_prefers_scene_root_over_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_root = tmp_path / "scene"
    cwd = tmp_path / "cwd"
    scene_root.mkdir()
    cwd.mkdir()
    (scene_root / "real_splats.npz").write_bytes(b"scene")
    (cwd / "real_splats.npz").write_bytes(b"stale-cwd")

    monkeypatch.chdir(cwd)

    assert resolve_scene_file(scene_root, None, "real_splats.npz") == scene_root / "real_splats.npz"
    assert resolve_scene_file(scene_root, "real_splats.npz", "unused.npz") == scene_root / "real_splats.npz"


def test_resolve_optional_scene_file_prefers_scene_root_over_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scene_root = tmp_path / "scene"
    cwd = tmp_path / "cwd"
    scene_root.mkdir()
    cwd.mkdir()
    (scene_root / "gt.ply").write_text("scene\n", encoding="utf-8")
    (cwd / "gt.ply").write_text("stale-cwd\n", encoding="utf-8")

    monkeypatch.chdir(cwd)

    assert resolve_optional_scene_file(scene_root, None, "gt.ply") == scene_root / "gt.ply"
    assert resolve_optional_scene_file(scene_root, "gt.ply", "fallback.ply") == scene_root / "gt.ply"


def test_resolve_scene_files_preserve_absolute_paths(tmp_path: Path) -> None:
    absolute = tmp_path / "external" / "real_splats.npz"

    assert resolve_scene_file(tmp_path / "scene", absolute, "unused.npz") == absolute
    assert resolve_optional_scene_file(tmp_path / "scene", absolute, None) == absolute
    assert resolve_optional_scene_file(tmp_path / "scene", None, None) is None
