"""Tests for src/codeflow/cli/generate_synthetic_data.py output handling."""

import argparse
import importlib.util
import pathlib

from src.codeflow.synthetic.gen_code import CodeGenResult


def _script():
    return importlib.import_module("src.codeflow.cli.generate_synthetic_data")


def _args(**kw):
    base = dict(target=3, seed=42, no_coreset=False)
    base.update(kw)
    return argparse.Namespace(**base)


class TestSelectCoreset:
    def test_mined_data_does_not_wipe_output(self):
        mod = _script()
        synthetic = [
            CodeGenResult(instruction=f"s{i}", analysis={}, solution=f"o{i}",
                          feature_used="macros" if i % 2 else "atoms")
            for i in range(4)
        ]
        mined = [{"instruction": f"m{i}", "input": "", "output": "x"} for i in range(4)]
        selected = mod.step_select_coreset(_args(target=5), synthetic, mined)
        assert len(selected) == 5
        # Records stay plain alpaca examples.
        assert all(set(r) <= {"system", "instruction", "input", "output"} for r in selected)


class TestSplitDir:
    def test_splits_are_per_output(self, tmp_path):
        mod = _script()
        a = mod.split_dir_for(str(tmp_path / "synth_a.jsonl"))
        b = mod.split_dir_for(str(tmp_path / "synth_b.jsonl"))
        assert a != b
        assert pathlib.Path(a).parent == tmp_path
        assert "synth_a" in pathlib.Path(a).name
