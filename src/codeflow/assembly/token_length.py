"""Drop examples that don't fit the training context, measured in real tokens.

Character counts are a poor proxy: code averages ~3.5 chars/token but REPL
results full of numbers get close to 1. LLaMA-Factory truncates anything past
``cutoff_len``, which cuts a diff mid-hunk, so such records are dropped here
instead. Length is measured on the rendered chat template, system prompt and
markup included, with the target model's tokenizer.
"""

import json
from typing import List, Tuple


def _assistant(value: str) -> dict:
    """An assistant turn with its <think> block as reasoning_content, the way
    the Qwen chat template expects it."""
    from .validator import split_think

    parts = split_think(value)
    if parts is None:
        return {"role": "assistant", "content": value}
    return {"role": "assistant", "reasoning_content": parts[0], "content": parts[1]}


def record_messages(record: dict) -> List[dict]:
    """Chat messages for a record, as the model will see them.

    Alpaca records become system / user / assistant, with the user turn
    ``instruction + "\n" + input`` as LLaMA-Factory builds it. Multi-turn
    records map function_call turns to assistant tool calls and observation
    turns to tool messages.
    """
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    conv = record.get("conversations")
    if not conv:
        user = record.get("instruction", "")
        if record.get("input"):
            user = f"{user}\n{record['input']}" if user else record["input"]
        messages.append({"role": "user", "content": user})
        messages.append(_assistant(record.get("output", "")))
        return messages
    for turn in conv:
        role, value = turn["from"], turn["value"]
        if role == "human":
            messages.append({"role": "user", "content": value})
        elif role == "observation":
            messages.append({"role": "tool", "content": value})
        elif role == "function_call":
            msg = _assistant(value)
            call = json.loads(msg.pop("content"))
            msg["content"] = ""
            msg["tool_calls"] = [{"type": "function", "function": call}]
            messages.append(msg)
        else:
            messages.append(_assistant(value))
    return messages


def count_tokens(record: dict, tokenizer) -> int:
    tools = json.loads(record["tools"]) if record.get("tools") else None
    ids = tokenizer.apply_chat_template(record_messages(record), tools=tools, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return len(ids)


def filter_by_tokens(records: List[dict], tokenizer,
                     cutoff_len: int) -> Tuple[List[dict], List[int]]:
    """Records that fit in ``cutoff_len`` tokens, and the lengths of those that don't."""
    kept, dropped = [], []
    for r in records:
        n = count_tokens(r, tokenizer)
        if n <= cutoff_len:
            kept.append(r)
        else:
            dropped.append(n)
    return kept, dropped


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)
