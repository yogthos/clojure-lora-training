"""Invariants of the LLaMA-Factory training configs.

Each check is a way a config silently trains the wrong thing on Qwen3.6
(qwen3_5 architecture) with LLaMA-Factory.
"""

from pathlib import Path

import pytest
import yaml

_CONFIGS = sorted(Path("configs").glob("codeflow_*_qwen.yaml"))


@pytest.fixture(params=_CONFIGS, ids=lambda p: p.name)
def config(request):
    return yaml.safe_load(request.param.read_text())


def test_configs_found():
    assert len(_CONFIGS) == 2


def test_no_remote_code(config):
    # qwen3_5 is native in transformers >= 5.2; remote code would run instead.
    assert "trust_remote_code" not in config


def test_packing_cannot_leak_linear_attention_state(config):
    # LLaMA-Factory only patches GatedDeltaNet to reset at packed-sequence
    # boundaries when flash_attn is fa2 on CUDA; otherwise state leaks
    # between packed examples.
    if config.get("neat_packing") or config.get("packing"):
        assert config.get("flash_attn") == "fa2"


def test_thinking_mode_is_explicit(config):
    # With thinking on and no <think> in a response, LLaMA-Factory trains an
    # empty think block into every answer.
    assert config.get("enable_thinking") is True
    assert config["template"] == "qwen3_6"


def test_early_stopping_is_a_real_argument(config):
    assert isinstance(config.get("early_stopping_steps"), int)
    assert config["save_steps"] % config["eval_steps"] == 0
    assert config.get("load_best_model_at_end") is True


def test_assembler_measures_with_the_trained_model(config):
    import sys
    from unittest.mock import patch
    from src.codeflow.cli.assemble_codeflow_dataset import parse_args
    with patch.object(sys, "argv", ["a", "--git-dir", "g", "--synth-dir", "s", "--output", "o"]):
        args = parse_args()
    assert args.tokenizer == config["model_name_or_path"]
    assert args.cutoff_len == config["cutoff_len"]


def test_keys_are_llamafactory_arguments(config):
    hparams = pytest.importorskip("llamafactory.hparams")
    from transformers import HfArgumentParser
    parser = HfArgumentParser([hparams.ModelArguments, hparams.DataArguments,
                               hparams.TrainingArguments, hparams.FinetuningArguments,
                               hparams.GeneratingArguments])
    known = {a.dest for a in parser._actions}
    assert set(config) - known == set()
