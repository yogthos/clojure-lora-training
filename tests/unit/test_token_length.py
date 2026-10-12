"""Token-length filtering on the rendered chat template."""

from src.codeflow.assembly.token_length import filter_by_tokens, record_messages


class _CharTokenizer:
    """One token per character of the rendered template."""

    def apply_chat_template(self, messages, tokenize=False, **kw):
        text = "".join(f"<{m['role']}>{m['content']}</>" for m in messages)
        return {"input_ids": list(text)} if tokenize else text


def test_messages_follow_llamafactory_alpaca():
    rec = {"system": "S", "instruction": "do x", "input": "ctx", "output": "done"}
    assert record_messages(rec) == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "do x\nctx"},
        {"role": "assistant", "content": "done"},
    ]
    assert record_messages({"instruction": "i", "output": "o"})[0] == {"role": "user", "content": "i"}


def test_counts_the_whole_rendered_example():
    # system prompt and template markup count against the context too
    rec = {"system": "x" * 50, "instruction": "i", "input": "", "output": "o"}
    n = len(_CharTokenizer().apply_chat_template(record_messages(rec)))
    kept, dropped = filter_by_tokens([rec], _CharTokenizer(), cutoff_len=n)
    assert kept == [rec] and dropped == []
    kept, dropped = filter_by_tokens([rec], _CharTokenizer(), cutoff_len=n - 1)
    assert kept == [] and dropped == [n]
