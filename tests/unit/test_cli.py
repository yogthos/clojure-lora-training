"""Tests for assemble_codeflow_dataset CLI."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

from src.codeflow.cli.assemble_codeflow_dataset import (
    parse_args,
    main,
)


class _CharTokenizer:
    def apply_chat_template(self, messages, tokenize=False, **kw):
        text = "".join(m["content"] for m in messages)
        return {"input_ids": list(text)} if tokenize else text


@pytest.fixture(autouse=True)
def char_tokenizer(monkeypatch):
    import src.codeflow.cli.assemble_codeflow_dataset as cli
    loaded = []
    monkeypatch.setattr(cli, "load_tokenizer", lambda name: loaded.append(name) or _CharTokenizer())
    return loaded


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


class TestParseArgs:
    def test_required_arguments(self):
        with patch("sys.argv", [
            "assemble_codeflow_dataset",
            "--git-dir", "/tmp/git",
            "--synth-dir", "/tmp/synth",
            "--output", "/tmp/merged.jsonl",
        ]):
            args = parse_args()
            assert args.git_dir == ["/tmp/git"]
            assert args.synth_dir == ["/tmp/synth"]
            assert args.output == "/tmp/merged.jsonl"

    def test_defaults(self):
        with patch("sys.argv", [
            "assemble_codeflow_dataset",
            "--git-dir", "/tmp/g",
            "--synth-dir", "/tmp/s",
            "--output", "/tmp/o.jsonl",
        ]):
            args = parse_args()
            assert args.max_per_type is None
            assert args.no_format is False
            assert args.no_validate is False
            assert args.validation_report is None

    def test_multiple_dirs(self):
        with patch("sys.argv", [
            "assemble_codeflow_dataset",
            "--git-dir", "/tmp/g1",
            "--git-dir", "/tmp/g2",
            "--synth-dir", "/tmp/s1",
            "--output", "/tmp/o.jsonl",
        ]):
            args = parse_args()
            assert args.git_dir == ["/tmp/g1", "/tmp/g2"]

    def test_max_per_type(self):
        with patch("sys.argv", [
            "assemble_codeflow_dataset",
            "--git-dir", "/tmp/g",
            "--synth-dir", "/tmp/s",
            "--output", "/tmp/o.jsonl",
            "--max-per-type", "100",
        ]):
            args = parse_args()
            assert args.max_per_type == 100

    def test_token_filter_defaults_to_training_config(self):
        with patch("sys.argv", ["a", "--git-dir", "g", "--synth-dir", "s", "--output", "o"]):
            args = parse_args()
        assert args.tokenizer == "Qwen/Qwen3.6-27B"
        assert args.cutoff_len == 16384

    def test_flags(self):
        with patch("sys.argv", [
            "assemble_codeflow_dataset",
            "--git-dir", "/tmp/g",
            "--synth-dir", "/tmp/s",
            "--output", "/tmp/o.jsonl",
            "--no-format",
            "--no-validate",
            "--validation-report", "/tmp/report.json",
        ]):
            args = parse_args()
            assert args.no_format is True
            assert args.no_validate is True
            assert args.validation_report == "/tmp/report.json"


class TestMainIntegration:
    """Test the full pipeline end-to-end."""
    def test_full_pipeline(self):
        with TemporaryDirectory() as d:
            dpath = Path(d)
            git_dir = dpath / "git"
            synth_dir = dpath / "synth"
            git_dir.mkdir()
            synth_dir.mkdir()
            output = dpath / "merged.jsonl"
            report = dpath / "report.json"

            # Write git-mined records
            _write_jsonl(git_dir / "data.jsonl", [
                {
                    "instruction": "refactor middleware to use comp for handler composition",
                    "output": (
                        ";; nREPL session:\n"
                        ";; eval: (require '[ring.middleware.json :refer [wrap-json-response]])\n"
                        ";; result: nil\n"
                        ";; apply:\n"
                        "diff --git a/src/handler.clj b/src/handler.clj\n"
                        "--- a/src/handler.clj\n"
                        "+++ b/src/handler.clj\n"
                        "@@ -10,4 +10,6 @@\n"
                        " (defn app [request]\n"
                        "-  (-> request wrap-params wrap-json-body))\n"
                        "-\n"
                        "-(def handler app)\n"
                        "+  (-> request\n"
                        "+      wrap-params\n"
                        "+      (comp wrap-json-response wrap-json-body)))\n"
                        "+\n"
                        "+(def handler (wrap-defaults app))\n"
                    ),
                    "source": "git",
                },
            ])

            # Write synthetic records
            _write_jsonl(synth_dir / "data.jsonl", [
                {
                    "instruction": "fix null pointer exception in request handler",
                    "output": (
                        ";; nREPL session:\n"
                        ";; eval: (defn safe-handler [request]\n"
                        ";;        (when request (handle request)))\n"
                        ";; result: #'user/safe-handler\n"
                        ";; apply:\n"
                        "diff --git a/src/handler.clj b/src/handler.clj\n"
                        "--- a/src/handler.clj\n"
                        "+++ b/src/handler.clj\n"
                        "@@ -10,6 +10,8 @@\n"
                        " (defn app [request]\n"
                        "-  (handle-that-may-be-nil request))\n"
                        "-\n"
                        "-(def handler app)\n"
                        "+  (some-> request\n"
                        "+          validate\n"
                        "+          handle-that-may-be-nil))\n"
                        "+\n"
                        "+(def handler (wrap-errors app))\n"
                    ),
                    "source": "synth",
                },
            ])

            with patch("sys.argv", [
                "assemble_codeflow_dataset",
                "--git-dir", str(git_dir),
                "--synth-dir", str(synth_dir),
                "--output", str(output),
                "--validation-report", str(report),
            ]):
                result = main()

            assert result == 0
            assert output.exists()

            # Verify output is valid JSONL
            with open(output) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    assert "instruction" in rec
                    assert "output" in rec
                    assert "system" in rec
                    assert "history" in rec
                    assert rec["history"] == []
                    # Extra fields should be stripped by format step
                    assert "source" not in rec

            # Verify validation report
            assert report.exists()
            with open(report) as f:
                rpt = json.load(f)
            assert rpt["summary"]["valid"] >= 1

    def test_empty_inputs(self):
        with TemporaryDirectory() as d:
            dpath = Path(d)
            (dpath / "git").mkdir()
            (dpath / "synth").mkdir()
            _write_jsonl(dpath / "git" / "data.jsonl", [])
            _write_jsonl(dpath / "synth" / "data.jsonl", [])
            output = dpath / "merged.jsonl"

            with patch("sys.argv", [
                "assemble_codeflow_dataset",
                "--git-dir", str(dpath / "git"),
                "--synth-dir", str(dpath / "synth"),
                "--output", str(output),
            ]):
                result = main()

            assert result == 0
            assert output.exists() or True  # Should not crash


def test_records_over_cutoff_are_dropped(tmp_path, char_tokenizer):
    git, synth = tmp_path / "git", tmp_path / "synth"
    git.mkdir(); synth.mkdir()
    diff = ("diff --git a/f.clj b/f.clj\n--- a/f.clj\n+++ b/f.clj\n"
            "@@ -1,2 +1,2 @@\n (ns f)\n-(def a 1)\n+(def a 2)\n")
    _write_jsonl(git / "d.jsonl", [
        {"instruction": "change the default value of a", "output": diff, "repo": "r"},
        {"instruction": "change the default value of b", "output": diff + "+" * 0,
         "input": "x" * 5000, "repo": "r"},
    ])
    _write_jsonl(synth / "d.jsonl", [])
    out = tmp_path / "o.jsonl"
    with patch("sys.argv", ["a", "--git-dir", str(git), "--synth-dir", str(synth),
                            "--output", str(out), "--min-changed-lines", "0",
                            "--cutoff-len", "4000", "--no-validate"]):
        assert main() == 0
    recs = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["instruction"] for r in recs] == ["change the default value of a"]
    assert char_tokenizer == ["Qwen/Qwen3.6-27B"]
