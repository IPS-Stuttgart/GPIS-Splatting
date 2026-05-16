from __future__ import annotations

from pathlib import Path
from typing import Any

from gpis_splatting.serialization import read_json

CPU_PROXY_BACKEND = "cpu_proxy"
GSPLAT_BACKEND = "gsplat"
EXTERNAL_OR_UNKNOWN_BACKEND = "external_or_unknown"

DIAGNOSTIC_PROXY_FIDELITY = "diagnostic_proxy"
FAITHFUL_3DGS_FIDELITY = "faithful_3dgs"
EXTERNAL_OR_UNKNOWN_FIDELITY = "external_or_unknown"

REAL_RENDER_REPORT = "real_render_report.json"
GSPLAT_RENDER_REPORT = "gsplat_render_report.json"

CPU_PROXY_NOTE = (
    "The CPU real-splat renderer is a diagnostic proxy: it uses isotropic image-space kernels, "
    "constant colors, and simplified alpha compositing. Use the gsplat/standard 3DGS renderer "
    "for photometric PSNR/SSIM/LPIPS claims."
)


def diagnostic_proxy_render_metadata() -> dict[str, Any]:
    return {
        "backend": CPU_PROXY_BACKEND,
        "render_fidelity": DIAGNOSTIC_PROXY_FIDELITY,
        "photometric_metrics_allowed": False,
        "photometric_metrics_policy": "diagnostic_only",
        "render_backend_note": CPU_PROXY_NOTE,
    }


def faithful_3dgs_render_metadata() -> dict[str, Any]:
    return {
        "backend": GSPLAT_BACKEND,
        "render_fidelity": FAITHFUL_3DGS_FIDELITY,
        "photometric_metrics_allowed": True,
        "photometric_metrics_policy": "allowed",
        "render_backend_note": "Rendered with a 3DGS-compatible gsplat backend.",
    }


def load_prediction_render_metadata(predictions_dir: str | Path) -> dict[str, Any]:
    """Load render-fidelity metadata from a prediction directory.

    Prediction directories produced outside this repository usually have no internal
    report. Those are allowed by default because they may be Graphdeco/gsplat render
    outputs. Internal CPU proxy render directories are detected by their
    ``real_render_report.json`` file, including legacy reports written before the
    explicit fidelity fields were added.
    """
    root = Path(predictions_dir)
    for report_name in (GSPLAT_RENDER_REPORT, REAL_RENDER_REPORT):
        report_path = root / report_name
        if report_path.exists():
            report = read_json(report_path)
            return render_metadata_from_report(report, report_path=report_path)
    return {
        "backend": EXTERNAL_OR_UNKNOWN_BACKEND,
        "render_fidelity": EXTERNAL_OR_UNKNOWN_FIDELITY,
        "photometric_metrics_allowed": True,
        "photometric_metrics_policy": "allowed_no_internal_report",
        "report_path": None,
        "render_backend_note": "No internal render report was found; assuming external renderer output.",
        "diagnostic_proxy_override": False,
    }


def render_metadata_from_report(report: dict[str, Any], *, report_path: Path) -> dict[str, Any]:
    backend = report.get("backend")
    if backend is None:
        if report_path.name == REAL_RENDER_REPORT:
            backend = CPU_PROXY_BACKEND
        elif report_path.name == GSPLAT_RENDER_REPORT:
            backend = GSPLAT_BACKEND
        else:
            backend = EXTERNAL_OR_UNKNOWN_BACKEND

    render_fidelity = report.get("render_fidelity")
    if render_fidelity is None:
        if backend == CPU_PROXY_BACKEND:
            render_fidelity = DIAGNOSTIC_PROXY_FIDELITY
        elif backend == GSPLAT_BACKEND:
            render_fidelity = FAITHFUL_3DGS_FIDELITY
        else:
            render_fidelity = EXTERNAL_OR_UNKNOWN_FIDELITY

    allowed = report.get("photometric_metrics_allowed")
    if allowed is None:
        allowed = backend != CPU_PROXY_BACKEND

    policy = report.get("photometric_metrics_policy")
    if policy is None:
        policy = "allowed" if bool(allowed) else "diagnostic_only"

    return {
        "backend": backend,
        "render_fidelity": render_fidelity,
        "photometric_metrics_allowed": bool(allowed),
        "photometric_metrics_policy": policy,
        "report_path": str(report_path),
        "render_backend_note": report.get("render_backend_note") or (CPU_PROXY_NOTE if backend == CPU_PROXY_BACKEND else None),
        "diagnostic_proxy_override": False,
    }


def validate_prediction_render_metadata(*, predictions_dir: str | Path, allow_diagnostic_proxy: bool) -> dict[str, Any]:
    metadata = load_prediction_render_metadata(predictions_dir)
    if not metadata.get("photometric_metrics_allowed", True):
        if not allow_diagnostic_proxy:
            raise ValueError(
                "Refusing to compute photometric metrics for diagnostic CPU proxy renders. "
                "Use render_3dgs_with_gsplat or a standard 3DGS renderer for PSNR/SSIM/LPIPS claims. "
                "Pass allow_diagnostic_proxy=True only for explicitly diagnostic sweeps."
            )
        metadata = dict(metadata)
        metadata["diagnostic_proxy_override"] = True
        metadata["photometric_metrics_policy"] = "diagnostic_proxy_override"
    return metadata
