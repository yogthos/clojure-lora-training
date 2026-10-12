"""Drop examples that don't fit the training context, measured in real tokens.

Character counts are a poor proxy: code averages ~3.5 chars/token but REPL
results full of numbers get close to 1. LLaMA-Factory truncates anything past
``cutoff_len``, which cuts a diff mid-hunk, so such records are dropped here
instead. Length is measured on the rendered chat template, system prompt and
markup included, with the target model's tokenizer.
"""

from typing import List, Tuple


def record_messages(record: dict) -> List[dict]:
    """Chat messages for a record, as LLaMA-Factory builds them.

    Records with ``messages`` are used as-is. Alpaca records become system /
    user / assistant, with the user turn ``instruction + "\\n" + input``.
    """
    if "messages" in record:
        return record["messages"]
    user = record.get("instruction", "")
    if record.get("input"):
        user = f"{user}\n{record['input']}" if user else record["input"]
    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": record.get("output", "")})
    return messages


def count_tokens(record: dict, tokenizer) -> int:
    ids = tokenizer.apply_chat_template(record_messages(record), tokenize=True)
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
