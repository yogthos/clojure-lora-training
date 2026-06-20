"""Tests for dataset assembler — merge, deduplicate, balance."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.codeflow.assembly.assembler import (
    load_jsonl,
    compute_dedup_key,
    classify_example,
    deduplicate,
    balance_by_type,
    filter_by_length,
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


class TestFilterByLength:
    def test_drops_oversize_records(self):
        recs = [
            {"instruction": "small", "input": "x" * 10, "output": "y" * 10},
            {"instruction": "huge", "input": "x" * 5000, "output": "y" * 5000},
        ]
        result = filter_by_length(recs, max_chars=1000)
        assert len(result) == 1
        assert result[0]["instruction"] == "small"

    def test_none_keeps_all(self):
        recs = [
            {"instruction": "a", "input": "x" * 100000, "output": "y"},
            {"instruction": "b", "input": "z", "output": "w"},
        ]
        assert len(filter_by_length(recs, max_chars=None)) == 2

    def test_measures_combined_fields(self):
        # input alone is under the cap, but input+output together exceed it
        rec = {"instruction": "i", "input": "x" * 600, "output": "y" * 600}
        assert filter_by_length([rec], max_chars=1000) == []
        assert filter_by_length([rec], max_chars=2000) == [rec]

    def test_boundary_is_inclusive(self):
        # exactly at the cap is kept
        rec = {"instruction": "", "input": "x" * 500, "output": "y" * 500}
        assert filter_by_length([rec], max_chars=1000) == [rec]

    def test_empty_input(self):
        assert filter_by_length([], max_chars=1000) == []


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

    def test_drops_oversize_before_balancing(self):
        with TemporaryDirectory() as d:
            dir_path = Path(d)
            git_dir = dir_path / "git"
            synth_dir = dir_path / "synth"
            git_dir.mkdir()
            synth_dir.mkdir()

            _write_jsonl(git_dir / "out.jsonl", [
                {"instruction": "fix small bug", "input": "x", "output": "diff1"},
                {"instruction": "fix huge bug", "input": "x" * 9000, "output": "diff2"},
            ])
            _write_jsonl(synth_dir / "out.jsonl", [])

            output = dir_path / "merged.jsonl"
            result = assemble_dataset(
                git_paths=[git_dir],
                synth_paths=[synth_dir],
                output_path=output,
                max_chars=1000,
            )
            assert len(result) == 1
            assert result[0]["instruction"] == "fix small bug"

    def test_tags_record_source(self):
        with TemporaryDirectory() as d:
            dir_path = Path(d)
            git_dir = dir_path / "git"
            synth_dir = dir_path / "synth"
            git_dir.mkdir()
            synth_dir.mkdir()
            _write_jsonl(git_dir / "g.jsonl", [
                {"instruction": "fix null bug", "output": "diff-g"},
            ])
            _write_jsonl(synth_dir / "s.jsonl", [
                {"instruction": "add feature endpoint", "output": "diff-s"},
            ])
            result = assemble_dataset(
                git_paths=[git_dir], synth_paths=[synth_dir],
                output_path=dir_path / "out.jsonl",
            )
            by_instr = {r["instruction"]: r["source"] for r in result}
            assert by_instr["fix null bug"] == "git"
            assert by_instr["add feature endpoint"] == "synthetic"

    def test_dup_across_sources_keeps_git(self):
        with TemporaryDirectory() as d:
            dir_path = Path(d)
            git_dir = dir_path / "git"
            synth_dir = dir_path / "synth"
            git_dir.mkdir()
            synth_dir.mkdir()
            dup = {"instruction": "fix null bug", "output": "same-diff"}
            _write_jsonl(git_dir / "g.jsonl", [dict(dup)])
            _write_jsonl(synth_dir / "s.jsonl", [dict(dup)])
            result = assemble_dataset(
                git_paths=[git_dir], synth_paths=[synth_dir],
                output_path=dir_path / "out.jsonl",
            )
            assert len(result) == 1
            assert result[0]["source"] == "git"  # git loaded first, wins dedup

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

    def test_synthetic_exempt_from_type_cap(self):
        # The per-type cap exists to tame the abundant git pool; the curated,
        # verified synthetic workflow set should survive in full, not get
        # diluted by random sampling inside saturated change-type buckets.
        with TemporaryDirectory() as d:
            dir_path = Path(d)
            git_dir = dir_path / "git"
            synth_dir = dir_path / "synth"
            git_dir.mkdir()
            synth_dir.mkdir()

            # 100 git bug-fix records (one dominant change type).
            _write_jsonl(git_dir / "g.jsonl", [
                {"instruction": f"fix bug {i}", "output": "diff"}
                for i in range(100)
            ])
            # 30 synthetic records that also classify as bug-fix.
            _write_jsonl(synth_dir / "s.jsonl", [
                {"instruction": f"resolve crash {i}", "output": "diff"}
                for i in range(30)
            ])

            result = assemble_dataset(
                git_paths=[git_dir], synth_paths=[synth_dir],
                output_path=dir_path / "out.jsonl",
                max_per_type=10,
            )
            git_kept = [r for r in result if r["source"] == "git"]
            synth_kept = [r for r in result if r["source"] == "synthetic"]
            assert len(git_kept) == 10        # git capped at max_per_type
            assert len(synth_kept) == 30      # all synthetic survive
