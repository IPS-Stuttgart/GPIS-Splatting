from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pandas as pd
except Exception:  # pragma: no cover - fallback for minimal report environments.
    pd = None  # type: ignore[assignment]

DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = ("*.json", "*.csv", "*.md", "*.png", "*.npz", "*.ply")
STATUS_PATTERNS: tuple[str, ...] = ("*status.json", "*report.json", "evaluation_config.json", "real_gpis_config.json")
CSV_PATTERNS: tuple[str, ...] = ("*metrics.csv", "*checks.csv", "*summary.csv", "*comparison.csv", "*winners.csv", "*manifest.csv")
HASH_SIZE_LIMIT_BYTES = 25 * 1024 * 1024


@dataclass(frozen=True)
class ArtifactRecord:
    """Compact artifact-manifest entry for a reproducibility report."""

    path: str
    size_bytes: int
    sha256_12: str | None = None


def load_reproducibility_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON reproducibility config with light schema validation."""

    config_path = Path(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Reproducibility config must be a JSON object: {config_path}")
    config.setdefault("name", config_path.stem)
    if "description" not in config:
        raise ValueError(f"Reproducibility config is missing 'description': {config_path}")
    if "ablation" in config and "targets" not in config:
        raise ValueError(f"Evaluation preset configs with 'ablation' must also define 'targets': {config_path}")
    if "commands" in config and not isinstance(config["commands"], list):
        raise ValueError(f"Reproducibility config field 'commands' must be a list: {config_path}")
    return config


def collect_artifact_manifest(root: str | Path, *, include_patterns: Iterable[str] | None = None, max_files: int = 200) -> list[ArtifactRecord]:
    """Collect a deterministic, size-bounded manifest of experiment artifacts."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {root_path}")
    patterns = tuple(include_patterns or DEFAULT_INCLUDE_PATTERNS)
    records: list[ArtifactRecord] = []
    for path in sorted(candidate for candidate in root_path.rglob("*") if candidate.is_file()):
        rel = path.relative_to(root_path).as_posix()
        if not any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            continue
        size = path.stat().st_size
        digest = _sha256_12(path) if size <= HASH_SIZE_LIMIT_BYTES else None
        records.append(ArtifactRecord(path=rel, size_bytes=size, sha256_12=digest))
        if len(records) >= max_files:
            break
    return records


def build_reproducibility_report(
    experiment_root: str | Path,
    *,
    config_path: str | Path | None = None,
    command_lines: Iterable[str] | None = None,
    commit: str | None = None,
    include_patterns: Iterable[str] | None = None,
    max_files: int = 200,
) -> str:
    """Build a Markdown report recording how an experiment was produced and what it wrote."""

    root = Path(experiment_root)
    if not root.exists():
        raise FileNotFoundError(f"Experiment root does not exist: {root}")

    config = load_reproducibility_config(config_path) if config_path else None
    resolved_commit = commit or _git_commit(root) or _git_commit(Path.cwd()) or "unknown"
    statuses = _read_json_artifacts(root, STATUS_PATTERNS)
    csv_summaries = _summarize_csv_artifacts(root, CSV_PATTERNS)
    manifest = collect_artifact_manifest(root, include_patterns=include_patterns, max_files=max_files)

    lines = [
        "# Reproducibility Report",
        "",
        "## Run identity",
        "",
        f"- Experiment root: `{root}`",
        f"- Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Git commit: `{resolved_commit}`",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Platform: `{platform.platform()}`",
    ]
    if config_path:
        config_hash = _sha256_12(Path(config_path)) if Path(config_path).exists() else "missing"
        lines.extend([f"- Config: `{config_path}`", f"- Config SHA-256 prefix: `{config_hash}`"])
    if config:
        lines.extend(["", "## Config summary", "", f"- Name: `{config.get('name', '')}`", f"- Description: {config.get('description', '')}"])
        if "ablation" in config:
            lines.append(f"- Ablation keys: `{', '.join(sorted(config['ablation']))}`")
        if "targets" in config:
            lines.append(f"- Target checks: `{', '.join(sorted(config['targets']))}`")
        if "commands" in config:
            lines.append(f"- Configured command steps: `{len(config['commands'])}`")

    commands = [str(command) for command in (command_lines or [])]
    if commands:
        lines.extend(["", "## Commands", ""])
        for command in commands:
            lines.extend(["```powershell", command, "```"])

    lines.extend(["", "## Status JSON", "", _markdown_table(statuses) if statuses else "No status JSON files matched the configured patterns."])
    lines.extend(["", "## CSV summaries", "", _markdown_table(csv_summaries) if csv_summaries else "No CSV summary files matched the configured patterns."])
    lines.extend(["", "## Artifact manifest", ""])
    if manifest:
        lines.append(_markdown_table([{"path": record.path, "size_bytes": record.size_bytes, "sha256_12": record.sha256_12 or "large-file-skipped"} for record in manifest]))
    else:
        lines.append("No artifacts matched the configured include patterns.")

    lines.extend(
        [
            "",
            "## Re-run checklist",
            "",
            "1. Check out the recorded commit.",
            "2. Install the package with `python -m pip install -e .`.",
            "3. Run the command(s) above, or the command sequence in the referenced config.",
            "4. Compare status JSON, CSV row counts, and artifact hashes against this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reproducibility_report(experiment_root: str | Path, output_path: str | Path | None = None, **kwargs: Any) -> Path:
    """Write a reproducibility report and return its path."""

    root = Path(experiment_root)
    destination = Path(output_path) if output_path else root / "reproducibility_report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_reproducibility_report(root, **kwargs), encoding="utf-8")
    return destination


def _read_json_artifacts(root: Path, patterns: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _matching_files(root, patterns):
        rel = path.relative_to(root).as_posix()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            rows.append({"path": rel, "status": "unreadable", "passed": "", "keys": "", "details": str(exc)})
            continue
        rows.append(
            {
                "path": rel,
                "status": "ok",
                "passed": data.get("passed", data.get("success", "")) if isinstance(data, dict) else "",
                "keys": ", ".join(sorted(data)[:12]) if isinstance(data, dict) else type(data).__name__,
                "details": data.get("preset", data.get("method", data.get("scene", ""))) if isinstance(data, dict) else "",
            }
        )
    return rows


def _summarize_csv_artifacts(root: Path, patterns: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _matching_files(root, patterns):
        rel = path.relative_to(root).as_posix()
        try:
            if pd is not None:
                frame = pd.read_csv(path)
                rows.append({"path": rel, "rows": int(frame.shape[0]), "columns": int(frame.shape[1]), "column_names": ", ".join(str(column) for column in frame.columns[:10])})
            else:
                row_count, column_names = _csv_shape_without_pandas(path)
                rows.append({"path": rel, "rows": row_count, "columns": len(column_names), "column_names": ", ".join(column_names[:10])})
        except Exception as exc:  # pragma: no cover - defensive path for malformed files.
            rows.append({"path": rel, "rows": "", "columns": "", "column_names": f"unreadable: {exc}"})
    return rows


def _matching_files(root: Path, patterns: Iterable[str]) -> list[Path]:
    matches: list[Path] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
            matches.append(path)
    return matches


def _csv_shape_without_pandas(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return rows, header


def _sha256_12(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _git_commit(path: Path) -> str | None:
    try:
        completed = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No rows."
    columns = list(dict.fromkeys(column for row in rows for column in row))
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "\\|")
