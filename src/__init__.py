"""Clojure Code Flow LoRA Trainer.

Train a Qwen-based model on Clojure code evolution using the Code Flow paradigm.
Mines git repositories for commit transitions, generates synthetic training
data, and assembles JSONL datasets for LLaMA-Factory fine-tuning.
"""

from .shared import _SYSTEM_PROMPT, compute_dedup_key, count_records, load_jsonl, write_jsonl

__version__ = "0.2.0"