#!/usr/bin/env python3
"""Fuse a PEFT LoRA checkpoint into a base model and AWQ-quantize to 4-bit.

Output is a HuggingFace-format AWQ model servable on RunPod via vLLM:

    vllm serve /path/to/output --quantization awq --dtype float16

This script must run on a CUDA GPU (RunPod, Colab, local NVIDIA box). It
will NOT run on Apple Silicon — use scripts/fuse_model.py --mlx for that.

Typical usage (run on RunPod after rsyncing the base model + checkpoint):

    python scripts/fuse_and_quantize_awq.py \\
        --model models/Qwen2.5-32B \\
        --checkpoint checkpoints/checkpoint-7000 \\
        --output models/Qwen2.5-32B-lovecraft-7000-awq \\
        --calibration data/corpus/curated/lovecraft.txt

Memory: 32B merge needs ~70GB RAM (bf16 weights in CPU). AWQ calibration
needs a single GPU with ~24GB VRAM (A10G / L4 / 3090 / 4090 / A100 all work).
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
from pathlib import Path


def merge_lora(base_model: str, checkpoint: Path, merged_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"[merge] attaching PEFT adapter: {checkpoint}")
    model = PeftModel.from_pretrained(model, str(checkpoint))

    print("[merge] merge_and_unload()...")
    model = model.merge_and_unload()

    print(f"[merge] saving merged model to {merged_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(merged_dir), safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(str(merged_dir))

    del model
    gc.collect()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_calibration(path: Path | None, num_samples: int, max_chars: int) -> list[str]:
    if path is None:
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Split on blank lines, keep paragraphs long enough to be useful calibration.
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 200]
    if not paras:
        # Fall back to fixed-size windows.
        paras = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
    return [p[:max_chars] for p in paras[:num_samples]]


def quantize_awq(
    merged_dir: Path,
    output_dir: Path,
    calibration: Path | None,
    num_samples: int,
    max_chars: int,
    group_size: int,
    zero_point: bool,
) -> None:
    # autoawq installs as `awq`
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print(f"[awq] loading merged model: {merged_dir}")
    model = AutoAWQForCausalLM.from_pretrained(
        str(merged_dir),
        safetensors=True,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": 4,
        "version": "GEMM",
    }
    print(f"[awq] quant config: {quant_config}")

    calib_kwargs: dict = {}
    samples = load_calibration(calibration, num_samples, max_chars)
    if samples:
        print(f"[awq] using {len(samples)} calibration samples from {calibration}")
        calib_kwargs["calib_data"] = samples
    else:
        print("[awq] using autoawq default calibration (pileval)")

    print("[awq] quantizing (this will use one GPU and take a while)...")
    model.quantize(tokenizer, quant_config=quant_config, **calib_kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[awq] saving quantized model to {output_dir}")
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


def write_metadata(
    output_dir: Path,
    base_model: str,
    checkpoint: Path,
    adapter_cfg: dict,
    group_size: int,
    zero_point: bool,
) -> None:
    metadata = {
        "base_model": base_model,
        "checkpoint": str(checkpoint),
        "lora_rank": adapter_cfg.get("r"),
        "lora_alpha": adapter_cfg.get("lora_alpha"),
        "fusion_method": "peft_merge_and_unload",
        "quantization": "awq",
        "w_bit": 4,
        "q_group_size": group_size,
        "zero_point": zero_point,
        "runtime": "vllm --quantization awq",
    }
    with open(output_dir / "fuse_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA into a base model and AWQ-quantize to 4-bit."
    )
    parser.add_argument("--model", "-m", required=True, help="Base model path or HF id")
    parser.add_argument(
        "--checkpoint", "-c", required=True, help="PEFT checkpoint directory"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for the AWQ model"
    )
    parser.add_argument(
        "--merged-dir",
        default=None,
        help="Where to store the intermediate bf16-merged model "
        "(default: <output>-merged, deleted after quantization unless --keep-merged).",
    )
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Keep the intermediate merged fp16/bf16 model after quantizing.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Assume --merged-dir already contains a merged model; only quantize.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Plain-text file for AWQ calibration (e.g. a corpus file). "
        "Omit to use autoawq's default pileval calibration.",
    )
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--max-chars", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--no-zero-point",
        action="store_true",
        help="Disable zero-point quantization (default: enabled, higher quality).",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"Error: checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)
    adapter_cfg_path = checkpoint / "adapter_config.json"
    if not adapter_cfg_path.exists():
        print(f"Error: {adapter_cfg_path} not found", file=sys.stderr)
        sys.exit(1)
    with open(adapter_cfg_path) as f:
        adapter_cfg = json.load(f)

    output_dir = Path(args.output)
    tmp_merged: tempfile.TemporaryDirectory | None = None
    if args.merged_dir:
        merged_dir = Path(args.merged_dir)
    elif args.keep_merged:
        merged_dir = output_dir.parent / (output_dir.name + "-merged")
    else:
        tmp_merged = tempfile.TemporaryDirectory(prefix="fused_merge_")
        merged_dir = Path(tmp_merged.name)

    try:
        if not args.skip_merge:
            merge_lora(args.model, checkpoint, merged_dir)
        elif not merged_dir.exists():
            print(
                f"Error: --skip-merge set but {merged_dir} does not exist",
                file=sys.stderr,
            )
            sys.exit(1)

        quantize_awq(
            merged_dir=merged_dir,
            output_dir=output_dir,
            calibration=args.calibration,
            num_samples=args.num_samples,
            max_chars=args.max_chars,
            group_size=args.group_size,
            zero_point=not args.no_zero_point,
        )

        write_metadata(
            output_dir=output_dir,
            base_model=args.model,
            checkpoint=checkpoint,
            adapter_cfg=adapter_cfg,
            group_size=args.group_size,
            zero_point=not args.no_zero_point,
        )
    finally:
        if tmp_merged is not None:
            tmp_merged.cleanup()
        elif not args.keep_merged and not args.merged_dir and merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)

    print(f"\nDone. AWQ model saved to {output_dir}")
    print("\nServe on RunPod:")
    print(
        f"  vllm serve {output_dir} --quantization awq --dtype float16 "
        "--max-model-len 8192"
    )


if __name__ == "__main__":
    main()
