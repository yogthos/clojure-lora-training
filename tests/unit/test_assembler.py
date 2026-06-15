"""Tests for dataset assembler — merge, deduplicate, balance."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.assembly.assembler import (
    load_jsonl,
    compute_dedup_key,
    classify_example,
    deduplicate,
    balance_by_type,
    assemble_dataset,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestLoadJSONL:
    def test_loads_all_records(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            records = [
                {"instruction": "fix bug", "output": "diff1"},
                {"instruction": "add feature", "output": "diff2"},
            ]
            _write_jsonl(p, records)
            result = load_jsonl(p)
            assert len(result) == 2
            assert result[0]["instruction"] == "fix bug"

    def test_loads_empty_file(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "empty.jsonl"
            p.write_text("")
            result = load_jsonl(p)
            assert result == []

    def test_loads_directory(self):
        with TemporaryDirectory() as d:
            dir_path = Path(d)
            _write_jsonl(dir_path / "a.jsonl", [{"a": 1}])
            _write_jsonl(dir_path / "b.jsonl", [{"b": 2}])
            result = load_jsonl(dir_path)
            assert len(result) == 2

    def test_skips_malformed_lines(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            p.write_text('{"valid": 1}\nnot json\n{"also": 2}\n')
            result = load_jsonl(p)
            assert len(result) == 2


class TestComputeDedupKey:
    def test_same_instruction_output_same_key(self):
        ex1 = {"instruction": "fix bug", "output": "diff here"}
        ex2 = {"instruction": "fix bug", "output": "diff here"}
        assert compute_dedup_key(ex1) == compute_dedup_key(ex2)

    def test_different_output_different_key(self):
        ex1 = {"instruction": "fix bug", "output": "diff1"}
        ex2 = {"instruction": "fix bug", "output": "diff2"}
        assert compute_dedup_key(ex1) != compute_dedup_key(ex2)

    def test_ignores_extra_fields(self):
        ex1 = {"instruction": "fix", "output": "diff", "system": "A"}
        ex2 = {"instruction": "fix", "output": "diff", "system": "B", "history": []}
        assert compute_dedup_key(ex1) == compute_dedup_key(ex2)


class TestClassifyExample:
    def test_detects_bug_fix(self):
        ex = {"instruction": "fix the null pointer bug in the handler"}
        assert classify_example(ex) == "bug-fix"

    def test_detects_refactor(self):
        ex = {"instruction": "refactor the middleware chain to use comp"}
        assert classify_example(ex) == "refactor"

    def test_detects_add_feature(self):
        ex = {"instruction": "add endpoint for user preferences"}
        assert classify_example(ex) == "add-feature"

    def test_detects_optimize(self):
        ex = {"instruction": "optimize the transducer pipeline with lazy-seq"}
        assert classify_example(ex) == "optimize"

    def test_uses_output_fallback(self):
        ex = {"instruction": "update things", "output": ";; fix: handle edge case"}
        assert classify_example(ex) == "bug-fix"

    def test_unknown_defaults_to_refactor(self):
        ex = {"instruction": "do something vague"}
        assert classify_example(ex) == "refactor"


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        recs = [
            {"instruction": "fix bug", "output": "diff"},
            {"instruction": "fix bug", "output": "diff"},
            {"instruction": "add", "output": "other"},
        ]
        result = deduplicate(recs)
        assert len(result) == 2

    def test_keeps_first_occurrence(self):
        recs = [
            {"instruction": "fix bug", "output": "diff", "source": "git"},
            {"instruction": "fix bug", "output": "diff", "source": "synth"},
        ]
        result = deduplicate(recs)
        assert result[0]["source"] == "git"


class TestBalanceByType:
    def test_caps_dominant_types(self):
        recs = []
        for i in range(100):
            recs.append({"instruction": f"fix bug {i}", "output": "diff"})
        for i in range(10):
            recs.append({"instruction": f"refactor {i}", "output": "diff2"})
        result = balance_by_type(recs, max_per_type=50)
        bug_fixes = [r for r in result if classify_example(r) == "bug-fix"]
        assert len(bug_fixes) <= 50

    def test_leaves_small_types_untouched(self):
        recs = [{"instruction": f"refactor {i}", "output": "diff"} for i in range(3)]
        result = balance_by_type(recs, max_per_type=100)
        assert len(result) == 3

    def test_uses_default_max_when_none(self):
        recs = [{"instruction": f"fix {i}", "output": "d"} for i in range(100)]
        result = balance_by_type(recs)
        assert len(result) == len(recs)  # default max computed from smallest type


class TestAssembleDataset:
    def test_merges_and_deduplicates(self):
        with TemporaryDirectory() as d:
            dir_path = Path(d)
            git_dir = dir_path / "git"
            synth_dir = dir_path / "synth"
            git_dir.mkdir()
            synth_dir.mkdir()

            _write_jsonl(git_dir / "out.jsonl", [
                {"instruction": "fix null pointer", "output": "diff1", "source": "git"},
                {"instruction": "fix null pointer", "output": "diff1", "source": "git"},
            ])
            _write_jsonl(synth_dir / "out.jsonl", [
                {"instruction": "add protocol", "output": "diff2", "source": "synth"},
            ])

            output = dir_path / "merged.jsonl"
            result = assemble_dataset(
                git_paths=[git_dir],
                synth_paths=[synth_dir],
                output_path=output,
            )
            assert len(result) == 2

    def test_handles_empty_inputs(self):
        with TemporaryDirectory() as d:
            dir_path = Path(d)
            (dir_path / "git").mkdir()
            (dir_path / "synth").mkdir()
            _write_jsonl(dir_path / "git" / "out.jsonl", [])
            _write_jsonl(dir_path / "synth" / "out.jsonl", [])

            output = dir_path / "merged.jsonl"
            result = assemble_dataset(
                git_paths=[dir_path / "git"],
                synth_paths=[dir_path / "synth"],
                output_path=output,
            )
            assert len(result) == 0
