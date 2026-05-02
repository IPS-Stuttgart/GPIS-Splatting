from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gpis_splatting.real_splat_filtering import build_filter_variants, validate_filtering_config
from gpis_splatting.splats import SplatCloud


def test_random_same_retention_variants_match_gate_threshold_retention(tmp_path: Path) -> None:
    splats = SplatCloud(
        centers=torch.arange(18, dtype=torch.float64).reshape(6, 3),
        colors=torch.ones((6, 3), dtype=torch.float64),
        tau=torch.ones((6,), dtype=torch.float64),
        sigma=torch.ones((6,), dtype=torch.float64),
        is_surface=torch.ones((6,), dtype=torch.bool),
    )
    gates = np.asarray([0.05, 0.2, 0.3, 0.4, 0.7, 0.9], dtype=np.float64)

    variants = build_filter_variants(
        splats=splats,
        gates=gates,
        splats_path=tmp_path / "source_splats.npz",
        gate_path=tmp_path / "source_gate.npz",
        out_dir=tmp_path,
        method_name="test_filter",
        gate_thresholds=(0.25, 0.75),
        include_baseline=False,
        write_scaled=False,
        write_filtered=True,
        include_random_baselines=True,
        random_baseline_seeds=(0, 1),
        tau_scale_floor=0.0,
    )

    by_name = {variant.name: variant for variant in variants}
    assert by_name["gate_ge_0p25"].retained_count == 4
    assert by_name["gate_ge_0p75"].retained_count == 1

    for seed in (0, 1):
        random_0p25 = by_name[f"random_same_retention_0p25_seed{seed}"]
        random_0p75 = by_name[f"random_same_retention_0p75_seed{seed}"]
        assert random_0p25.kind == "random_same_retention"
        assert random_0p25.retained_count == by_name["gate_ge_0p25"].retained_count
        assert random_0p25.retention_fraction == by_name["gate_ge_0p25"].retention_fraction
        assert random_0p25.random_seed == seed
        assert random_0p75.retained_count == by_name["gate_ge_0p75"].retained_count
        assert random_0p75.retention_fraction == by_name["gate_ge_0p75"].retention_fraction
        assert random_0p75.random_seed == seed
        assert random_0p25.splats_path.exists()
        assert random_0p25.gate_path is not None and random_0p25.gate_path.exists()


def test_random_same_retention_requires_filtered_variants() -> None:
    try:
        validate_filtering_config(
            gate_thresholds=(0.25,),
            include_baseline=True,
            write_scaled=False,
            write_filtered=False,
            include_random_baselines=True,
            random_baseline_seeds=(0,),
            tau_scale_floor=0.0,
            render_max_frames=0,
        )
    except ValueError as exc:
        assert "Random same-retention baselines" in str(exc)
    else:
        raise AssertionError("Expected random baselines without filtered variants to fail.")
