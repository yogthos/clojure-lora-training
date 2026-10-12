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


def test_multi_turn_record_becomes_tool_messages():
    from src.codeflow.synthetic.jolt_eval import EvalResult
    from src.codeflow.synthetic.live_trace import EVAL_TOOL, LiveTrace, Step
    rec = LiveTrace(request="sq", context="", diff="d",
                    steps=[Step("why", "(sq 2)", EvalResult(form="", value="4"))],
                    final_thinking="done").to_record()
    rec["system"] = "S"
    msgs = record_messages(rec)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant"]
    assert msgs[2]["reasoning_content"] == "why"
    assert msgs[2]["tool_calls"] == [{"type": "function", "function": {
        "name": "eval", "arguments": {"code": "(sq 2)"}}}]
    assert msgs[3]["content"] == "=> 4"
    assert msgs[4] == {"role": "assistant", "reasoning_content": "done", "content": ";; apply:\nd"}


def test_tools_reach_the_template():
    from src.codeflow.assembly.token_length import count_tokens
    seen = {}

    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, tools=None, **kw):
            seen["tools"] = tools
            return {"input_ids": [1, 2, 3]}

    count_tokens({"instruction": "i", "output": "o", "tools": '[{"type": "function"}]',
                  "conversations": [{"from": "human", "value": "i"}, {"from": "gpt", "value": "o"}]},
                 _Tok())
    assert seen["tools"] == [{"type": "function"}]
