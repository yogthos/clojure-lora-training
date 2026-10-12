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
    filter_trivial_diffs,
    count_changed_lines,
    assemble_dataset,
)


_DIFF_HEAD = "diff --git a/x.clj b/x.clj\n--- a/x.clj\n+++ b/x.clj\n@@ -1,2 +1,2 @@\n"


def _diff(added: int, removed: int) -> str:
    body = "".join(f"+new line {i}\n" for i in range(added))
    body += "".join(f"-old line {i}\n" for i in range(removed))
    return _DIFF_HEAD + " context\n" + body


class TestCountChangedLines:
    def test_counts_add_and_remove_excluding_headers(self):
        # +++/--- file headers must not count as changes.
        assert count_changed_lines(_diff(3, 2)) == 5

    def test_zero_for_no_changes(self):
        assert count_changed_lines("just text\n context only\n") == 0


class TestFilterTrivialDiffs:
    def test_drops_git_transitions_below_floor(self):
        recs = [
            {"instruction": "big", "output": _diff(5, 5), "source": "git"},
            {"instruction": "tiny", "output": _diff(1, 0), "source": "git"},
        ]
        kept = filter_trivial_diffs(recs, min_changed_lines=4)
        assert [r["instruction"] for r in kept] == ["big"]

    def test_synthetic_records_are_exempt(self):
        # Workflow output is not a diff; never filtered on change-count.
        recs = [{"instruction": "wf", "output": ";; Goal: x\n;; eval: (+ 1 2)",
                 "source": "synthetic"}]
        assert filter_trivial_diffs(recs, min_changed_lines=4) == recs

    def test_non_diff_git_record_is_kept(self):
        recs = [{"instruction": "x", "output": "no diff here", "source": "git"}]
        assert filter_trivial_diffs(recs, min_changed_lines=4) == recs


def _live(instruction, output):
    """A synthetic multi-turn trace record (only those survive assembly)."""
    return {"instruction": instruction, "output": output, "tools": "[]",
            "conversations": [{"from": "human", "value": instruction},
                              {"from": "gpt", "value": output}]}


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

    def test_malformed_line_is_an_error(self):
        # A corrupt record in the middle is data loss, not something to skip.
        import pytest
        with TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            p.write_text('{"valid": 1}\nnot json\n{"also": 2}\n')
            with pytest.raises(ValueError, match="test.jsonl:2"):
                load_jsonl(p)

    def test_torn_final_line_is_skipped(self):
        # An interrupted append leaves an unterminated last line.
        with TemporaryDirectory() as d:
            p = Path(d) / "test.jsonl"
            p.write_text('{"valid": 1}\n{"also": 2}\n{"to')
            assert load_jsonl(p) == [{"valid": 1}, {"also": 2}]


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

    def test_ignores_code_in_output_and_input(self):
        # Code is full of words like "add", "test" and "error"; the change type
        # is what was asked for.
        ex = {"instruction": "rename the config loader",
              "input": "(defn add-error-handler [])", "output": "+(deftest add-test)"}
        assert classify_example(ex) == "refactor"

    @pytest.mark.parametrize("instruction", [
        "update the prefix of the address field",   # fix in prefix, add in address
        "handle the latest dependency",              # test in latest
    ])
    def test_keywords_match_whole_words(self, instruction):
        assert classify_example({"instruction": instruction}) == "refactor"

    @pytest.mark.parametrize("instruction,kind", [
        ("fixes the retry loop", "bug-fix"),
        ("added streaming support", "add-feature"),
        ("adding tests for the parser", "test"),
    ])
    def test_inflections_match(self, instruction, kind):
        assert classify_example({"instruction": instruction}) == kind

    def test_unknown_defaults_to_refactor(self):
        ex = {"instruction": "do something vague"}
        assert classify_example(ex) == "refactor"

    def test_docs_folds_into_refactor(self):
        # docs is too sparse to balance as its own bucket; fold it into refactor
        # (both are non-behavioral cleanup) so there is no singleton category.
        ex = {"instruction": "update the readme and fix docstring typos in the comment"}
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

    def test_none_means_no_cap(self):
        # Capping every type at the smallest one by default silently threw away
        # most of the data.
        recs = ([{"instruction": f"fix bug {i}", "output": "d"} for i in range(100)]
                + [{"instruction": f"refactor {i}", "output": "d"} for i in range(2)])
        assert len(balance_by_type(recs)) == 102


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
                _live("add feature endpoint", "diff-s"),
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
                _live(f"resolve crash {i}", "diff")
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

    def test_assigns_diverse_system_prompts_by_objective(self):
        from src.shared import TRANSITION_SYSTEM_PROMPTS, WORKFLOW_SYSTEM_PROMPTS
        with TemporaryDirectory() as d:
            dir_path = Path(d)
            git_dir = dir_path / "git"
            synth_dir = dir_path / "synth"
            git_dir.mkdir()
            synth_dir.mkdir()
            _write_jsonl(git_dir / "g.jsonl", [
                {"instruction": f"add feature {i}", "output": "diff"}
                for i in range(60)
            ])
            _write_jsonl(synth_dir / "s.jsonl", [
                _live(f"build thing {i}", ";; Goal: x")
                for i in range(60)
            ])
            result = assemble_dataset(
                git_paths=[git_dir], synth_paths=[synth_dir],
                output_path=dir_path / "out.jsonl",
            )
            git = [r for r in result if r["source"] == "git"]
            synth = [r for r in result if r["source"] == "synthetic"]
            # Each objective draws only from its own paraphrase pool...
            assert all(r["system"] in TRANSITION_SYSTEM_PROMPTS for r in git)
            assert all(r["system"] in WORKFLOW_SYSTEM_PROMPTS for r in synth)
            # ...and more than one variant is actually used (no attention collapse).
            assert len({r["system"] for r in git}) > 1
            assert len({r["system"] for r in synth}) > 1


_BAD_COUNTS = (";; eval: (+ 1 2)\n;; result: 3\n;; apply:\n"
               "diff --git a/f.clj b/f.clj\n--- a/f.clj\n+++ b/f.clj\n"
               "@@ -1,5 +1,9 @@\n (ns f)\n-(def a 1)\n+(def a 2)\n")


def test_multi_message_ties_prefer_bug_fix():
    ex = {"instruction": "Remove unneeded expiry test; Fix admin logic; Refactor for readability"}
    assert classify_example(ex) == "bug-fix"


def test_one_shot_synthetic_traces_are_dropped(tmp_path):
    # Their ;; result: lines sit in the loss as text the model would learn to
    # write; only live multi-turn traces are kept.
    git, synth = tmp_path / "git", tmp_path / "synth"
    git.mkdir(); synth.mkdir()
    conv = [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b"}]
    _write_jsonl(synth / "s.jsonl", [
        {"instruction": "one shot", "output": _BAD_COUNTS},
        {"instruction": "live", "output": "b", "conversations": conv, "tools": "[]"},
    ])
    _write_jsonl(git / "g.jsonl", [])
    out = assemble_dataset([git], [synth], tmp_path / "o.jsonl", min_changed_lines=0)
    assert [r["instruction"] for r in out] == ["live"]


def test_length_prefilter_counts_every_turn():
    conv = [{"from": "human", "value": "a"}, {"from": "observation", "value": "x" * 2000},
            {"from": "gpt", "value": "b"}]
    rec = {"instruction": "a", "output": "b", "conversations": conv}
    assert filter_by_length([rec], max_chars=1000) == []
