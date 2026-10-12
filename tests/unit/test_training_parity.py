"""What LLaMA-Factory trains on must be what Qwen's chat template serves.

Runs only where llamafactory is installed (it isn't a project dependency):
    uv run --with "llamafactory @ git+https://github.com/hiyouga/LlamaFactory" \
        --with "transformers==5.8.0" --with "peft==0.18.1" \
        pytest tests/unit/test_training_parity.py
(the qwen3_8 template is on main; 0.9.5 doesn't have it)
"""

import json

import pytest

template_mod = pytest.importorskip("llamafactory.data.template")

from src.codeflow.assembly.sharegpt import to_sharegpt  # noqa: E402
from src.codeflow.assembly.token_length import record_messages  # noqa: E402
from src.codeflow.synthetic.jolt_eval import EvalResult  # noqa: E402
from src.codeflow.synthetic.live_trace import LiveTrace, Step  # noqa: E402

_ROLE = {"human": "user", "gpt": "assistant", "function_call": "function",
         "observation": "observation"}


@pytest.fixture(scope="module")
def tok_and_template():
    import yaml
    from llamafactory.hparams import DataArguments
    from transformers import AutoTokenizer
    cfg = yaml.safe_load(open("configs/codeflow_lora_qwen.yaml"))
    tok = AutoTokenizer.from_pretrained(cfg["model_name_or_path"])
    template = template_mod.get_template_and_fix_tokenizer(
        tok, DataArguments(template=cfg["template"], enable_thinking=cfg["enable_thinking"]))
    return tok, template


def _render(tok, template, record):
    sg = to_sharegpt(record)
    msgs = [{"role": _ROLE[t["from"]], "content": t["value"]} for t in sg["conversations"]]
    pairs = template.encode_multiturn(tok, msgs, sg["system"], sg["tools"])
    trained = [tok.decode(r) for _, r in pairs]
    return tok.decode([i for p, r in pairs for i in p + r]), trained


def test_live_trace_matches_chat_template(tok_and_template):
    tok, template = tok_and_template
    trace = LiveTrace(
        request="add a square fn", context="a math lib",
        steps=[Step("Plan: define square.", "(defn square [x]\n  (* x x))",
                    EvalResult(form="", value="#'user/square")),
               Step("Check 4.", "(square 4)", EvalResult(form="", stdout="hi\n", value="16"))],
        final_thinking="Works.",
        diff="diff --git a/s.clj b/s.clj\n--- /dev/null\n+++ b/s.clj\n@@ -0,0 +1 @@\n+(ns s)\n")
    rec = dict(trace.to_record(), system="SYS")
    text, trained = _render(tok, template, rec)
    assert text == tok.apply_chat_template(record_messages(rec), tools=json.loads(rec["tools"]),
                                           tokenize=False)
    # Loss only on the model's own turns; observations are prompt.
    assert len(trained) == 3
    assert all(t.startswith("<think>\n") for t in trained)
    assert not any("=> 16" in t for t in trained)


def test_transition_matches_chat_template(tok_and_template):
    tok, template = tok_and_template
    rec = {"system": "SYS", "instruction": "fix x", "input": "### a.clj\n(ns a)",
           "output": "<think>\nr\n</think>\n\ndiff --git a/a b/a\n+x\n"}
    text, trained = _render(tok, template, rec)
    assert text == tok.apply_chat_template(record_messages(rec), tokenize=False)
    assert trained == ["<think>\nr\n</think>\n\ndiff --git a/a b/a\n+x<|im_end|>\n"]


def test_history_chain_matches_chat_template(tok_and_template):
    # Every turn's reasoning is trained (qwen3_8 preserves past thinking), and
    # each turn's request and code are prompt.
    tok, template = tok_and_template
    conv = [{"from": "human", "value": "fix x\n### a.clj\n(ns a)"},
            {"from": "gpt", "value": "<think>\nfirst\n</think>\n\ndiff --git a/a b/a\n+x\n"},
            {"from": "human", "value": "fix y\n### a.clj\n(ns a)\nx"},
            {"from": "gpt", "value": "<think>\nsecond\n</think>\n\ndiff --git a/a b/a\n+y\n"}]
    rec = {"system": "SYS", "instruction": "fix x", "input": "", "output": conv[1]["value"],
           "conversations": conv}
    text, trained = _render(tok, template, rec)
    assert text == tok.apply_chat_template(record_messages(rec), tokenize=False)
    assert trained == ["<think>\nfirst\n</think>\n\ndiff --git a/a b/a\n+x<|im_end|>\n",
                       "<think>\nsecond\n</think>\n\ndiff --git a/a b/a\n+y<|im_end|>\n"]
