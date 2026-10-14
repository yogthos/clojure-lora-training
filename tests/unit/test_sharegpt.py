"""Final LLaMA-Factory records and the dataset_info that reads them."""

import json
from pathlib import Path

from src.codeflow.assembly.sharegpt import to_sharegpt


def test_transition_becomes_two_turns():
    rec = {"system": "S", "instruction": "fix x", "input": "### a.clj\n(ns a)",
           "output": "<think>\nr\n</think>\n\ndiff", "repo": "o__r"}
    assert to_sharegpt(rec) == {
        "system": "S",
        "conversations": [{"from": "human", "value": "fix x\n### a.clj\n(ns a)"},
                          {"from": "gpt", "value": "<think>\nr\n</think>\n\ndiff"}],
        "tools": "",
    }


def test_multi_turn_keeps_its_tools_column():
    # LLaMA-Factory's qwen3_8 template renders the tools block, the system
    # prompt and the reasoning-effort line exactly as Qwen's template does
    # (tests/unit/test_training_parity.py), so tools pass through as-is.
    conv = [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b\n"}]
    rec = {"system": "S", "instruction": "a", "output": "b", "conversations": conv,
           "tools": '[{"type": "function"}]'}
    out = to_sharegpt(rec)
    assert out["system"] == "S"
    assert out["tools"] == '[{"type": "function"}]'
    assert out["conversations"][-1]["value"] == "b"     # trimmed like the template


def test_final_turn_is_trimmed():
    rec = {"system": "S", "instruction": "x", "output": "<think>\nr\n</think>\n\ndiff\n"}
    assert to_sharegpt(rec)["conversations"][-1]["value"] == "<think>\nr\n</think>\n\ndiff"


def test_dataset_info_reads_these_records():
    info = json.loads(Path("configs/dataset_info.codeflow.json").read_text())
    for name in ("codeflow_train", "codeflow_val"):
        entry = info[name]
        assert entry["formatting"] == "sharegpt"
        assert entry["columns"] == {"messages": "conversations", "system": "system", "tools": "tools"}
        assert entry["tags"]["function_tag"] == "function_call"
        assert entry["tags"]["observation_tag"] == "observation"


def test_split_cli_writes_sharegpt(tmp_path, monkeypatch):
    import sys
    from src.codeflow.cli import split_codeflow_dataset as cli
    from src.shared import write_jsonl
    recs = [{"system": "S", "instruction": f"fix bug number {i}", "input": "",
             "output": f"<think>\nr\n</think>\n\nd{i}", "repo": f"r{i}"} for i in range(40)]
    write_jsonl(recs, tmp_path / "all.jsonl")
    monkeypatch.setattr(sys, "argv", ["split", "-i", str(tmp_path / "all.jsonl"),
                                      "--train-out", str(tmp_path / "t.jsonl"),
                                      "--val-out", str(tmp_path / "v.jsonl"), "--val-frac", "0.1"])
    assert cli.main() == 0
    rows = [json.loads(l) for p in ("t.jsonl", "v.jsonl")
            for l in (tmp_path / p).read_text().splitlines()]
    assert len(rows) == 40
    assert all(set(r) == {"system", "conversations", "tools"} for r in rows)
