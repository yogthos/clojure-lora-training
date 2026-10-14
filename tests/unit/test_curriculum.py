"""Training data ordered by difficulty, in phases (IQuest's SFT curriculum)."""

import json

from src.codeflow.assembly.curriculum import curriculum_order, phase


def _diff(n):
    return "diff --git a/a b/a\n--- a/a\n+++ b/a\n@@ -1 +1 @@\n" + "+x\n" * n


def _transition(n, i=0):
    return {"instruction": f"t{i}", "output": f"<think>\nr\n</think>\n\n{_diff(n)}"}


def _chain(i=0):
    conv = [{"from": "human", "value": "a"}, {"from": "gpt", "value": _diff(3)},
            {"from": "human", "value": "b"}, {"from": "gpt", "value": _diff(3)}]
    return {"instruction": f"c{i}", "output": _diff(3), "conversations": conv}


def _trace(i=0):
    conv = [{"from": "human", "value": "a"},
            {"from": "function_call", "value": '{"name": "eval", "arguments": {"code": "(+ 1 2)"}}'},
            {"from": "observation", "value": "=> 3"}, {"from": "gpt", "value": _diff(3)}]
    return {"instruction": f"w{i}", "output": "", "conversations": conv, "tools": "[]"}


def test_phases():
    assert phase(_transition(5)) == 0
    assert phase(_transition(80)) == 1
    assert phase(_chain()) == 1
    assert phase(_trace()) == 2


def test_order_is_easy_to_hard_and_keeps_every_record():
    recs = ([_trace(i) for i in range(5)] + [_chain(i) for i in range(5)]
            + [_transition(5, i) for i in range(5)] + [_transition(80, 100 + i) for i in range(5)])
    out = curriculum_order(recs, seed=1)
    assert [phase(r) for r in out] == sorted(phase(r) for r in recs)
    assert sorted(json.dumps(r, sort_keys=True) for r in out) == \
        sorted(json.dumps(r, sort_keys=True) for r in recs)
    assert out == curriculum_order(recs, seed=1)
    assert out != curriculum_order(recs, seed=2)      # shuffled within phases


def test_split_cli_orders_train(tmp_path, monkeypatch):
    import sys
    from src.codeflow.cli import split_codeflow_dataset as cli
    from src.shared import write_jsonl
    recs = []
    for i in range(30):
        r = [_trace(i), _chain(i), _transition(5, i)][i % 3]
        recs.append(dict(r, repo=f"r{i}", system="S", input="",
                         instruction=f"{r['instruction']} number {i}"))
    write_jsonl(recs, tmp_path / "all.jsonl")
    monkeypatch.setattr(sys, "argv", ["split", "-i", str(tmp_path / "all.jsonl"),
                                      "--train-out", str(tmp_path / "t.jsonl"),
                                      "--val-out", str(tmp_path / "v.jsonl"), "--val-frac", "0.1"])
    assert cli.main() == 0
    rows = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    kinds = [2 if r["tools"] else 1 if len(r["conversations"]) > 2 else 0 for r in rows]
    assert kinds == sorted(kinds)
