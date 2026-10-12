"""Tests for synthetic data generation: llm_json.py"""

import json

import pytest

from src.codeflow.synthetic.llm_json import parse_json_list


class TestParseJsonList:
    """require_json replies are JSON objects (json_object mode can't return a
    top-level array), so the item list sits under a key."""

    def test_named_key(self):
        reply = json.dumps({"tasks": [{"instruction": "a"}, {"instruction": "b"}]})
        assert parse_json_list(reply, "tasks") == [{"instruction": "a"}, {"instruction": "b"}]

    def test_single_list_value_under_other_key(self):
        reply = json.dumps({"items": [{"name": "x"}]})
        assert parse_json_list(reply, "features") == [{"name": "x"}]

    def test_empty_wrapper_under_other_key(self):
        assert parse_json_list('{"items": []}', "features") == []

    def test_bare_list(self):
        assert parse_json_list('[{"name": "x"}]', "features") == [{"name": "x"}]

    def test_single_object_is_one_item(self):
        # An item that happens to carry a list of strings isn't a wrapper.
        reply = json.dumps({"instruction": "a", "context_files": ["x.clj"]})
        assert parse_json_list(reply, "tasks") == [
            {"instruction": "a", "context_files": ["x.clj"]}
        ]

    def test_markdown_fence(self):
        reply = '```json\n{"features": [{"name": "x"}]}\n```'
        assert parse_json_list(reply, "features") == [{"name": "x"}]

    def test_non_dict_items_dropped(self):
        reply = json.dumps({"features": [{"name": "x"}, "junk", 3]})
        assert parse_json_list(reply, "features") == [{"name": "x"}]

    def test_invalid_json_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_json_list("not json", "features")

    def test_scalar_raises_value_error(self):
        with pytest.raises(ValueError):
            parse_json_list("42", "features")
