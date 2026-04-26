"""Unit tests for the I/O utility helpers in io_utils_v2.

These helpers are foundational — every v2 experiment script uses them —
so we test the safety guards (refuse-to-overwrite) and the basic
serialization shapes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from project_llm.io_utils_v2 import (
    ensure_new_output_dir,
    fail_if_path_exists,
    iso_timestamp,
    read_jsonl,
    sanitize_model_spec,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)


@dataclass(frozen=True)
class _SampleRow:
    name: str
    score: float


class TestSafetyGuards:
    def test_ensure_new_output_dir_creates_path(self, tmp_path: Path) -> None:
        target = tmp_path / "fresh_output"
        ensure_new_output_dir(target)
        assert target.is_dir()

    def test_ensure_new_output_dir_refuses_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "existing"
        target.mkdir()
        with pytest.raises(FileExistsError):
            ensure_new_output_dir(target)

    def test_fail_if_path_exists_passes_for_missing_path(self, tmp_path: Path) -> None:
        # No exception expected.
        fail_if_path_exists(tmp_path / "does_not_exist.txt")

    def test_fail_if_path_exists_raises_for_present_file(self, tmp_path: Path) -> None:
        existing = tmp_path / "already.txt"
        existing.write_text("hi")
        with pytest.raises(FileExistsError):
            fail_if_path_exists(existing)


class TestWriteHelpers:
    def test_write_json_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "payload.json"
        write_json(path, {"a": 1, "b": [2, 3]})
        assert json.loads(path.read_text()) == {"a": 1, "b": [2, 3]}

    def test_write_text_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "nested" / "note.txt"
        write_text(path, "hello")
        assert path.read_text() == "hello"

    def test_write_jsonl_serializes_dataclasses(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        rows = [_SampleRow(name="a", score=0.5), _SampleRow(name="b", score=0.7)]
        write_jsonl(path, rows)
        loaded = read_jsonl(path)
        assert loaded == [{"name": "a", "score": 0.5}, {"name": "b", "score": 0.7}]

    def test_write_csv_writes_only_listed_fieldnames(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.csv"
        rows = [{"name": "a", "score": 0.5, "extra": "drop_me"}, {"name": "b", "score": 0.7}]
        write_csv(path, rows, fieldnames=["name", "score"])
        text = path.read_text()
        assert "name,score" in text.splitlines()[0]
        assert "drop_me" not in text


class TestMisc:
    def test_iso_timestamp_is_parseable(self) -> None:
        ts = iso_timestamp()
        assert ts.endswith("+00:00")
        # Should be a valid ISO datetime string.
        from datetime import datetime

        datetime.fromisoformat(ts)

    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("Qwen/Qwen2.5-7B-Instruct", "Qwen_Qwen2.5-7B-Instruct"),
            ("plain", "plain"),
            ("ns:name/sub", "ns__name_sub"),
        ],
    )
    def test_sanitize_model_spec(self, spec: str, expected: str) -> None:
        assert sanitize_model_spec(spec) == expected
