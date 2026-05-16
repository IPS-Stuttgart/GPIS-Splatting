from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from gpis_splatting.render_metadata import validate_prediction_render_metadata as _validate_prediction_render_metadata


_DIAGNOSTIC_CPU_PROXY_TESTS = {
    "test_real_gpis_fit_render_and_evaluate_loop",
    "test_run_tanks_temples_hard_negative_calibration_cli",
}


def _is_within_pytest_tmp(path: str | Path) -> bool:
    return ".pytest_tmp" in Path(path).resolve(strict=False).parts


@pytest.fixture(autouse=True)
def allow_diagnostic_proxy_metrics_for_explicit_proxy_render_tests(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Allow diagnostic CPU-render metrics only in tests that intentionally exercise them.

    Production evaluation keeps the default guard: CPU proxy renders are rejected unless
    callers opt in explicitly. The real-data smoke tests create tiny synthetic proxy
    renders under pytest's temporary directory and only assert that diagnostic metrics
    can be produced, so they opt in here without weakening runtime defaults.
    """

    if request.node.name not in _DIAGNOSTIC_CPU_PROXY_TESTS:
        return

    def validate_prediction_render_metadata(*, predictions_dir: str | Path, allow_diagnostic_proxy: bool) -> dict[str, Any]:
        if not allow_diagnostic_proxy and _is_within_pytest_tmp(predictions_dir):
            allow_diagnostic_proxy = True
        return _validate_prediction_render_metadata(predictions_dir=predictions_dir, allow_diagnostic_proxy=allow_diagnostic_proxy)

    import gpis_splatting.real_benchmark as real_benchmark

    monkeypatch.setattr(real_benchmark, "validate_prediction_render_metadata", validate_prediction_render_metadata)
