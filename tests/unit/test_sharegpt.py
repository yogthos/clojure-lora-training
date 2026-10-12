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


def test_split_requires_tokenizer_only_for_tools():
    rec = {"system": "S", "instruction": "x", "output": "y", "tools": "[]",
           "conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": "y"}]}
    import pytest
    with pytest.raises(ValueError):
        to_sharegpt(rec)


class _Tok:
    """Renders a system turn the way Qwen's template does: tools first."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, **kw):
        sys_ = messages[0]["content"]
        body = (f"# Tools {tools}\n</IMPORTANT>\n\n{sys_}" if tools else sys_)
        return f"<|im_start|>system\n{body}<|im_end|>\n<|im_start|>user\nx<|im_end|>\n"


def test_multi_turn_system_carries_the_official_tools_block():
    # LLaMA-Factory's qwen3_5/qwen3_6 templates put the system prompt before
    # the tools block and unwrap the tool schema; Qwen's own template does
    # neither. Rendering the block into the system field keeps training and
    # inference identical.
    conv = [{"from": "human", "value": "a"}, {"from": "gpt", "value": "b\n"}]
    rec = {"system": "S", "instruction": "a", "output": "b", "conversations": conv,
           "tools": '[{"type": "function"}]'}
    out = to_sharegpt(rec, _Tok())
    assert out["system"] == "# Tools [{'type': 'function'}]\n</IMPORTANT>\n\nS"
    assert out["tools"] == ""
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
    monkeypatch.setattr(cli, "load_tokenizer", lambda name: _Tok())
    monkeypatch.setattr(sys, "argv", ["split", "-i", str(tmp_path / "all.jsonl"),
                                      "--train-out", str(tmp_path / "t.jsonl"),
                                      "--val-out", str(tmp_path / "v.jsonl"), "--val-frac", "0.1"])
    assert cli.main() == 0
    rows = [json.loads(l) for p in ("t.jsonl", "v.jsonl")
            for l in (tmp_path / p).read_text().splitlines()]
    assert len(rows) == 40
    assert all(set(r) == {"system", "conversations", "tools"} for r in rows)
