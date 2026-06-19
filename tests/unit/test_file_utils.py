"""Tests for synthetic data generation: file_utils.py"""

import json
import pytest
from pathlib import Path
from src.codeflow.synthetic.file_utils import (
    sort_jsonl,
    merge_jsonl,
    shuffle_jsonl,
    deduplicate_jsonl,
    count_records,
    split_jsonl,
    write_jsonl,
    read_jsonl,
)


def _write_test_jsonl(path: str, records: list) -> None:
    write_jsonl(records, path)


def _read_test_jsonl(path: str) -> list:
    return read_jsonl(path)


class TestWriteRead:
    def test_roundtrip(self, tmp_path):
        records = [
            {"instruction": "task 1", "input": "code 1", "output": "diff 1"},
            {"instruction": "task 2", "input": "code 2", "output": "diff 2"},
        ]
        path = str(tmp_path / "test.jsonl")
        write_jsonl(records, path)

        read = read_jsonl(path)
        assert len(read) == 2
        assert read[0]["instruction"] == "task 1"
        assert read[1]["instruction"] == "task 2"


class TestCountRecords:
    def test_counts_correctly(self, tmp_path):
        path = str(tmp_path / "count.jsonl")
        write_jsonl([{"a": 1}, {"a": 2}, {"a": 3}], path)
        assert count_records(path) == 3

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.jsonl")
        path = Path(path)
        path.write_text("")
        assert count_records(str(path)) == 0


class TestSortJSONL:
    def test_sorts_by_instruction(self, tmp_path):
        path = str(tmp_path / "sort.jsonl")
        records = [
            {"instruction": "zebra task"},
            {"instruction": "apple task"},
            {"instruction": "mango task"},
        ]
        write_jsonl(records, path)

        sorted_path = sort_jsonl(path)
        sorted_records = read_jsonl(sorted_path)
        instructions = [r["instruction"] for r in sorted_records]
        assert instructions == ["apple task", "mango task", "zebra task"]


class TestMergeJSONL:
    def test_merges_multiple_files(self, tmp_path):
        p1 = str(tmp_path / "a.jsonl")
        p2 = str(tmp_path / "b.jsonl")
        write_jsonl([{"instruction": "task a"}], p1)
        write_jsonl([{"instruction": "task b"}], p2)

        out = str(tmp_path / "merged.jsonl")
        merge_jsonl([p1, p2], out, deduplicate=False)
        assert count_records(out) == 2

    def test_deduplicates(self, tmp_path):
        p1 = str(tmp_path / "a.jsonl")
        p2 = str(tmp_path / "b.jsonl")
        write_jsonl([{"instruction": "same task"}], p1)
        write_jsonl([{"instruction": "same task"}], p2)

        out = str(tmp_path / "merged.jsonl")
        merge_jsonl([p1, p2], out, deduplicate=True)
        assert count_records(out) == 1


class TestShuffleJSONL:
    def test_same_count(self, tmp_path):
        path = str(tmp_path / "shuf.jsonl")
        records = [{"instruction": f"task {i}"} for i in range(20)]
        write_jsonl(records, path)

        shuffled_path = shuffle_jsonl(path, seed=42)
        assert count_records(shuffled_path) == 20

    def test_deterministic(self, tmp_path):
        path = str(tmp_path / "det.jsonl")
        records = [{"instruction": f"task {i}"} for i in range(10)]
        write_jsonl(records, path)

        shuf1 = read_jsonl(shuffle_jsonl(path, seed=42))
        shuf2 = read_jsonl(shuffle_jsonl(path, seed=42))
        assert shuf1 == shuf2


class TestDeduplicate:
    def test_removes_dupes(self, tmp_path):
        path = str(tmp_path / "dedup.jsonl")
        records = [
            {"instruction": "task 1"},
            {"instruction": "task 1"},
            {"instruction": "task 2"},
        ]
        write_jsonl(records, path)

        deduped_path = deduplicate_jsonl(path)
        assert count_records(deduped_path) == 2


class TestSplitJSONL:
    def test_splits_by_ratio(self, tmp_path):
        path = str(tmp_path / "split.jsonl")
        records = [{"instruction": f"task {i}"} for i in range(100)]
        write_jsonl(records, path)

        out_dir = str(tmp_path / "splits")
        train_path, val_path = split_jsonl(path, out_dir, train_ratio=0.8, seed=42)

        train_count = count_records(train_path)
        val_count = count_records(val_path)
        assert train_count + val_count == 100
        assert train_count == 80
        assert val_count == 20

    def test_no_val_split(self, tmp_path):
        path = str(tmp_path / "all.jsonl")
        write_jsonl([{"instruction": "task"}], path)

        out_dir = str(tmp_path / "splits")
        train_path, val_path = split_jsonl(path, out_dir, train_ratio=1.0, seed=42)

        assert count_records(train_path) == 1
        assert count_records(val_path) == 0
