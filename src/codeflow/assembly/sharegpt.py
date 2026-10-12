"""The final training record: LLaMA-Factory sharegpt.

Assembly works on flat records (instruction / input / output, plus the full
``conversations`` for multi-turn traces); this is the last step, when the
train/val files are written. configs/dataset_info.codeflow.json maps these
columns.

The text LLaMA-Factory trains on has to match what the model is served with
(Qwen's own chat template). Two things don't line up by default, so they are
fixed here:

- LLaMA-Factory's qwen3_5/qwen3_6 templates put the system prompt before the
  tools block and unwrap the tool schema to its inner function object; Qwen's
  template renders the full schema first and the system prompt after it. For
  records with tools, the system field becomes exactly what Qwen's template
  renders and the tools column is left empty, which LLaMA-Factory copies
  verbatim.
- Qwen's template trims assistant content; a diff's final newline would
  otherwise sit in the loss before <|im_end|>.
"""

import json
from functools import lru_cache
from typing import Optional

_SYSTEM_OPEN = "<|im_start|>system\n"
_TURN_END = "<|im_end|>"


def official_system(tokenizer, system: str, tools_json: str) -> str:
    """The system turn's content as the model's chat template renders it."""
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": "x"}],
        tools=json.loads(tools_json), tokenize=False)
    if not text.startswith(_SYSTEM_OPEN):
        raise ValueError("chat template doesn't open with a system turn")
    return text[len(_SYSTEM_OPEN):text.index(_TURN_END)]


def to_sharegpt(record: dict, tokenizer=None) -> dict:
    """LLaMA-Factory sharegpt record. ``tokenizer`` (the trained model's) is
    needed for records with tools."""
    conv = record.get("conversations")
    if not conv:
        human = record.get("instruction", "")
        if record.get("input"):
            human = f"{human}\n{record['input']}" if human else record["input"]
        conv = [{"from": "human", "value": human},
                {"from": "gpt", "value": record.get("output", "")}]
    conv = [dict(t, value=t["value"].strip()) if t["from"] == "gpt" else t for t in conv]

    system = record.get("system", "")
    tools = record.get("tools", "")
    if tools:
        if tokenizer is None:
            raise ValueError("a tokenizer is needed to render the tools block")
        system = _cached_system(tokenizer, system, tools)
    return {"system": system, "conversations": conv, "tools": ""}


@lru_cache(maxsize=None)
def _cached_system(tokenizer, system: str, tools: str) -> str:
    return official_system(tokenizer, system, tools)
