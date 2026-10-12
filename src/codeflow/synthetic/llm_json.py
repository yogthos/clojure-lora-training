"""Parse list-shaped replies from require_json LLM calls.

require_json sends response_format json_object, which can only return a JSON
object, never a top-level array. Prompts therefore ask for the items under a
named key ({"tasks": [...]}); this unwraps that, and still accepts a bare list
or a lone item from models that ignore the requested shape.
"""

import json
import re
from typing import Any, List

_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def parse_json_list(reply: Any, key: str) -> List[dict]:
    """Return the list of item dicts in an LLM JSON reply.

    Accepts, in order: the object's ``key`` list, its only list of objects
    under any other key, a bare list, or a single object as one item. Non-dict
    items are dropped.

    Raises:
        ValueError: The reply isn't JSON, or isn't an object or list.
    """
    data = reply
    if isinstance(reply, str):
        try:
            data = json.loads(reply)
        except json.JSONDecodeError:
            match = _FENCE.search(reply)
            if not match:
                raise
            data = json.loads(match.group(1))

    if isinstance(data, dict):
        if isinstance(data.get(key), list):
            items = data[key]
        elif len(data) == 1 and isinstance(next(iter(data.values())), list):
            items = next(iter(data.values()))
        else:
            # A wrapper under some other key holds a list of objects; an item
            # may carry lists too, but of strings (e.g. context_files).
            lists = [
                v for v in data.values()
                if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)
            ]
            items = lists[0] if len(lists) == 1 else [data]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"expected a JSON object or array, got {type(data).__name__}")

    return [item for item in items if isinstance(item, dict)]
