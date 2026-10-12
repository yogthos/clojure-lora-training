#!/usr/bin/env python3
"""Convert an F16/BF16 GGUF to HuggingFace safetensors (lossless for full-precision sources).

For Shadow0482/mythos_fast the source is VibeThinker-3B.BF16.gguf, which is bfloat16
full precision (config.json torch_dtype == bfloat16), so the round-trip is lossless.

Usage:
    uv run --with "transformers[torch]" --with gguf --with safetensors --with accelerate \
        python scripts/convert_gguf_to_safetensors.py \
        --gguf models/mythos_fast-gguf/VibeThinker-3B.BF16.gguf \
        --out models/mythos_fast-hf
"""
import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True, help="path to a full-precision .gguf file")
    ap.add_argument("--out", required=True, help="output HF-format directory")
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="output dtype (match the GGUF's native precision)",
    )
    args = ap.parse_args()
    args.gguf = os.path.abspath(args.gguf)
    args.out = os.path.abspath(args.out)

    dtype = getattr(torch, args.dtype)

    print(f"[1/4] Loading GGUF model: {args.gguf}", flush=True)
    # transformers 5.x GGUF API: pass the containing dir as the model id and the
    # filename via gguf_file=. Works for local files (resolved_archive_file path).
    gguf_dir, gguf_name = os.path.dirname(args.gguf), os.path.basename(args.gguf)
    model = AutoModelForCausalLM.from_pretrained(
        gguf_dir, gguf_file=gguf_name, low_cpu_mem_usage=True
    )
    model = model.to(dtype)

    print(f"[2/4] Saving safetensors to {args.out} (dtype={args.dtype})", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)

    print("[3/4] Loading & saving tokenizer from GGUF", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(gguf_dir, gguf_file=gguf_name)
    tokenizer.save_pretrained(args.out)

    print("[4/4] Verifying saved artifacts...", flush=True)
    reloaded = AutoModelForCausalLM.from_pretrained(args.out, low_cpu_mem_usage=True)
    n_params = sum(p.numel() for p in reloaded.parameters())
    print(f"  architecture : {reloaded.config.architectures}")
    print(f"  model_type   : {reloaded.config.model_type}")
    print(f"  layers       : {reloaded.config.num_hidden_layers}")
    print(f"  hidden_size  : {reloaded.config.hidden_size}")
    print(f"  params       : {n_params:,} (~{n_params / 1e9:.2f}B)")
    print(f"  dtype        : {next(reloaded.parameters()).dtype}")

    rt = AutoTokenizer.from_pretrained(args.out)
    enc = rt("Hello, world", return_tensors="pt").input_ids
    print(f"  tokenizer round-trip: {rt.decode(enc[0])!r}")
    for tok in ("<|im_start|>", "<|im_end|>"):
        print(f"  has {tok}: {tok in rt.get_vocab()}")
    template = getattr(rt, "chat_template", None)
    print(f"  chat_template present: {bool(template)}"
          + ("" if template else " — set template:qwen2 in LLaMA-Factory"))

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
