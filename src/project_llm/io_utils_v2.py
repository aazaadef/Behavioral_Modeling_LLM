"""I/O utilities for v2 experiments.

Provides safe file-writing functions (all refuse to overwrite existing
files), serialization helpers, and environment-capture for experiment
reproducibility.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def iso_timestamp() -> str:
    # Normalize timestamps so run metadata is easy to compare across machines.
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_new_output_dir(path: Path) -> None:
    # Guard against overwriting prior experiments or paper artifacts.
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {path}. "
            "Choose a new versioned output path."
        )
    path.mkdir(parents=True, exist_ok=False)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def fail_if_path_exists(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifact: {path}. "
            "Choose a new versioned path or remove it manually if that is intentional."
        )


def _normalize_row(row: Any) -> dict[str, Any]:
    # Accept dataclasses, plain dicts, or project objects with a to_dict method.
    if is_dataclass(row):
        return asdict(row)
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "to_dict"):
        payload = row.to_dict()
        if isinstance(payload, dict):
            return payload
    raise TypeError(f"Unsupported row type for serialization: {type(row)!r}")


def write_json(path: Path, payload: Any) -> None:
    fail_if_path_exists(path)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def write_text(path: Path, text: str) -> None:
    fail_if_path_exists(path)
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def write_jsonl(path: Path, rows: list[Any]) -> None:
    fail_if_path_exists(path)
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_normalize_row(row), ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    fail_if_path_exists(path)
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def sanitize_model_spec(spec: str) -> str:
    return spec.replace("/", "_").replace(":", "__")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def build_reproducibility_notes(
    command: str,
    dataset_root: Path,
    selected_split: str | None,
    execution_mode: str,
    model_specs: list[str],
) -> str:
    # Capture lightweight environment facts that matter for later paper reproduction.
    package_lines = []
    for name in ["openai", "transformers", "pandas", "sklearn", "torch"]:
        try:
            module = importlib.import_module(name)
            package_lines.append(f"{name}_version: {getattr(module, '__version__', 'unknown')}")
        except Exception:
            package_lines.append(f"{name}_version: unavailable")
    lines = [
        "ChildPlay extended experiment reproducibility notes",
        f"timestamp_utc: {iso_timestamp()}",
        f"command: {command}",
        f"cwd: {Path.cwd()}",
        f"python_executable: {sys.executable}",
        f"python_version: {platform.python_version()}",
        f"platform: {platform.platform()}",
        f"dataset_root: {dataset_root}",
        f"selected_split: {selected_split}",
        f"execution_mode: {execution_mode}",
        f"model_specs: {', '.join(model_specs)}",
        f"OPENAI_API_KEY_present: {'yes' if bool(os.getenv('OPENAI_API_KEY')) else 'no'}",
        f"DEEPSEEK_API_KEY_present: {'yes' if bool(os.getenv('DEEPSEEK_API_KEY')) else 'no'}",
        f"HF_HOME: {os.getenv('HF_HOME', '')}",
        f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE', '')}",
        *package_lines,
    ]
    return "\n".join(lines) + "\n"
