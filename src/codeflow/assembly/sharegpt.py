"""The final training record: LLaMA-Factory sharegpt.

Assembly works on flat records (instruction / input / output, plus the full
``conversations`` for multi-turn traces); this is the last step, when the
train/val files are written. configs/dataset_info.codeflow.json maps these
columns.

The text LLaMA-Factory trains on has to match what the model is served with
(Qwen's own chat template). LLaMA-Factory's qwen3_8 template renders the
system prompt, reasoning-effort line and tools block as Qwen's does
(tests/unit/test_training_parity.py checks it), so system and tools pass
through. One thing is fixed here: Qwen's template trims assistant content,
and a diff's final newline would otherwise sit in the loss before
<|im_end|>.
"""


def to_sharegpt(record: dict) -> dict:
    """LLaMA-Factory sharegpt record."""
    conv = record.get("conversations")
    if not conv:
        human = record.get("instruction", "")
        if record.get("input"):
            human = f"{human}\n{record['input']}" if human else record["input"]
        conv = [{"from": "human", "value": human},
                {"from": "gpt", "value": record.get("output", "")}]
    conv = [dict(t, value=t["value"].strip()) if t["from"] == "gpt" else t for t in conv]
    return {"system": record.get("system", ""), "conversations": conv,
            "tools": record.get("tools", "")}
