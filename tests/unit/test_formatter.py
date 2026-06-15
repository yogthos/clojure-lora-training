"""Tests for LLaMA-Factory JSONL formatter."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.assembly.formatter import (
    format_record,
    format_jsonl,
    format_jsonl_file,
    DEFAULT_SYSTEM_PROMPT,
)


class TestFormatRecord:
    def test_adds_system_if_missing(self):
        rec = {"instruction": "fix bug", "output": "diff"}
        result = format_record(rec)
        assert result["system"] == DEFAULT_SYSTEM_PROMPT

    def test_preserves_existing_system(self):
        rec = {"instruction": "x", "output": "y", "system": "custom"}
        result = format_record(rec)
        assert result["system"] == "custom"

    def test_adds_empty_history(self):
        rec = {"instruction": "x", "output": "y"}
        result = format_record(rec)
        assert result["history"] == []

    def test_preserves_existing_history(self):
        rec = {
            "instruction": "x",
            "output": "y",
            "history": [["q1", "a1"]],
        }
        result = format_record(rec)
        assert result["history"] == [["q1", "a1"]]

    def test_ensures_string_values(self):
        rec = {"instruction": 123, "output": 456, "system": None, "input": 0}
        result = format_record(rec)
        assert result["instruction"] == "123"
        assert result["output"] == "456"
        assert result["input"] == "0"
        assert isinstance(result["system"], str)

    def test_removes_unrecognized_fields(self):
        rec = {
            "instruction": "x",
            "output": "y",
            "source": "git",
            "_id": "abc",
            "extra_field": True,
        }
        result = format_record(rec)
        assert "source" not in result
        assert "_id" not in result
        assert "extra_field" not in result
        assert result["instruction"] == "x"
        assert result["output"] == "y"

    def test_standard_field_order(self):
        rec = {"output": "d", "input": "cc", "instruction": "a", "system": "b"}
        result = format_record(rec)
        keys = list(result.keys())
        assert keys == ["instruction", "input", "output", "system", "history"]


class TestFormatJSONL:
    def test_formats_all_records(self):
        recs = [
            {"instruction": "fix", "output": "d1"},
            {"instruction": "add", "output": "d2", "system": "custom"},
        ]
        result = format_jsonl(recs)
        assert len(result) == 2
        assert all(r["history"] == [] for r in result)
        assert result[0]["system"] == DEFAULT_SYSTEM_PROMPT
        assert result[1]["system"] == "custom"

    def test_handles_empty_input(self):
        assert format_jsonl([]) == []


class TestFormatJSONLFile:
    def test_reads_input_and_writes_output(self):
        with TemporaryDirectory() as d:
            dpath = Path(d)
            in_path = dpath / "in.jsonl"
            out_path = dpath / "out.jsonl"

            # Write raw records
            recs = [
                {"instruction": "fix bug", "output": "diff", "source": "git"},
                {"instruction": "add feat", "output": "diff2", "_id": "abc"},
            ]
            with open(in_path, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")

            count = format_jsonl_file(in_path, out_path)

            assert count == 2
            assert out_path.exists()

            with open(out_path) as f:
                formatted = [json.loads(line) for line in f if line.strip()]

            assert len(formatted) == 2
            for r in formatted:
                assert set(r.keys()) == {"instruction", "input", "output", "system", "history"}
                assert isinstance(r["history"], list)

    def test_handles_empty_input_file(self):
        with TemporaryDirectory() as d:
            dpath = Path(d)
            in_path = dpath / "in.jsonl"
            out_path = dpath / "out.jsonl"
            in_path.write_text("")

            count = format_jsonl_file(in_path, out_path)
            assert count == 0
