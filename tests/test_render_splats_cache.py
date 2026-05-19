from __future__ import annotations

import pytest

from gpis_splatting.cli.render_splats import _expected_splat_cache_metadata, _load_cached_splats
from gpis_splatting.serialization import write_json
from gpis_splatting.splats import make_candidate_splats, save_splats


def _write_splat_cache(tmp_path, *, shape: str = "sphere", num_splats: int = 16, seed: int = 4):
    splat_path = tmp_path / "splats.npz"
    metadata_path = tmp_path / "splats_metadata.json"
    splats = make_candidate_splats(shape, num_splats=num_splats, seed=seed)
    save_splats(str(splat_path), splats)
    return splat_path, metadata_path


def test_cached_splats_reject_missing_metadata(tmp_path):
    splat_path, metadata_path = _write_splat_cache(tmp_path)

    with pytest.raises(ValueError, match="metadata"):
        _load_cached_splats(splat_path, metadata_path, _expected_splat_cache_metadata("sphere", 16, 4))


def test_cached_splats_reject_changed_seed(tmp_path):
    splat_path, metadata_path = _write_splat_cache(tmp_path)
    write_json(metadata_path, _expected_splat_cache_metadata("sphere", 16, 4))

    with pytest.raises(ValueError, match="seed"):
        _load_cached_splats(splat_path, metadata_path, _expected_splat_cache_metadata("sphere", 16, 5))


def test_cached_splats_accept_matching_metadata(tmp_path):
    splat_path, metadata_path = _write_splat_cache(tmp_path)
    write_json(metadata_path, _expected_splat_cache_metadata("sphere", 16, 4))

    splats = _load_cached_splats(splat_path, metadata_path, _expected_splat_cache_metadata("sphere", 16, 4))

    assert splats.centers.shape[0] == 16
