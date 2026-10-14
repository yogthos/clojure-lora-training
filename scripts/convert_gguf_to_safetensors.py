#!/usr/bin/env python3
"""Convert an F32/F16/BF16 GGUF to HuggingFace safetensors (lossless for full-precision sources).

Quantized GGUFs can't be trained on and are refused. The output dtype defaults
to the GGUF's own precision.

transformers rebuilds the tokenizer from GGUF metadata but registers only the
first few control tokens, so tokens like <think> or <tool_call> would split into
pieces. They are re-added from the GGUF's token types (CONTROL = special,
USER_DEFINED = added, not special), and generation_config stops on the chat
template's end-of-turn token as well as the end-of-text token. Token ids match
the original exactly; special flags can't always: llama.cpp writes FIM tokens
as CONTROL, so they come back special (only skip_special_tokens decoding sees
the difference).

Usage:
    uv run --with "transformers[torch]" --with gguf --with safetensors --with accelerate \
        python scripts/convert_gguf_to_safetensors.py \
        --gguf models/mythos_fast-gguf/VibeThinker-3B.BF16.gguf \
        --out models/mythos_fast-hf
"""
import argparse
import os
import re
from typing import Dict, List

# general.file_type values (llama_ftype) that hold full-precision weights.
_NATIVE_DTYPES = {0: "float32", 1: "float16", 32: "bfloat16"}

# Placeholder rows llama.cpp adds to pad the vocab to the embedding size.
_PLACEHOLDER = re.compile(r"^\[PAD\d+\]$")

# End-of-turn markers a chat template may close assistant turns with.
_END_OF_TURN = ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>")
# End-of-document tokens. A GGUF records one EOS, which for a chat model may be
# the end-of-turn marker; generation should stop on these too.
_END_OF_TEXT = ("<|endoftext|>", "<|end_of_text|>", "</s>", "<eos>")


def native_dtype(file_type: int) -> str:
    """torch dtype name for a GGUF general.file_type; quantized types raise."""
    if file_type not in _NATIVE_DTYPES:
        raise ValueError(f"GGUF file_type {file_type} is quantized; only F32/F16/BF16 "
                         "GGUFs convert losslessly and can be trained")
    return _NATIVE_DTYPES[file_type]


def read_gguf_metadata(path) -> Dict:
    """The tokenizer and precision fields the conversion needs."""
    from gguf import GGUFReader

    fields = GGUFReader(str(path)).fields

    def get(key, default=None):
        return fields[key].contents() if key in fields else default

    return {
        "file_type": get("general.file_type"),
        "tokens": get("tokenizer.ggml.tokens", []),
        "token_types": get("tokenizer.ggml.token_type", []),
        "eos_token_id": get("tokenizer.ggml.eos_token_id"),
        "eot_token_id": get("tokenizer.ggml.eot_token_id"),
        "bos_token_id": get("tokenizer.ggml.bos_token_id"),
        "pad_token_id": get("tokenizer.ggml.padding_token_id"),
        "chat_template": get("tokenizer.chat_template"),
    }


def load_tokenizer(path, meta: Dict):
    """The GGUF's tokenizer with every CONTROL/USER_DEFINED token registered."""
    from gguf import TokenType
    from tokenizers import AddedToken
    from transformers import AutoTokenizer

    path = os.path.abspath(path)
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(path),
                                              gguf_file=os.path.basename(path))
    vocab = tokenizer.get_vocab()
    added = []
    for i, (tok, typ) in enumerate(zip(meta["tokens"], meta["token_types"])):
        if typ not in (TokenType.CONTROL, TokenType.USER_DEFINED) or _PLACEHOLDER.match(tok):
            continue
        if vocab.get(tok, i) != i:
            raise ValueError(f"token {tok!r} is id {i} in the GGUF but {vocab[tok]} in the tokenizer")
        added.append(AddedToken(tok, special=typ == TokenType.CONTROL, normalized=False))
    tokenizer.add_tokens(added)
    for tok in added:
        if tokenizer.convert_tokens_to_ids(tok.content) != meta["tokens"].index(tok.content):
            raise ValueError(f"re-adding {tok.content!r} changed its id")

    if meta["chat_template"] and not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = meta["chat_template"]
    if meta["eos_token_id"] is not None:
        tokenizer.eos_token = meta["tokens"][meta["eos_token_id"]]
    pad = meta["pad_token_id"]
    tokenizer.pad_token = meta["tokens"][pad if pad is not None else meta["eos_token_id"]]
    return tokenizer


def generation_settings(meta: Dict, tokenizer) -> Dict:
    """eos/pad/bos for generation_config.json.

    Chat models end a turn with a marker (``<|im_end|>``) that is often not the
    GGUF's EOS (``<|endoftext|>``); generation has to stop on both.
    """
    eos: List[int] = []
    template = meta["chat_template"] or ""
    for marker in _END_OF_TURN:
        if marker in template and marker in tokenizer.get_vocab():
            eos.append(tokenizer.convert_tokens_to_ids(marker))
    for key in ("eot_token_id", "eos_token_id"):
        if meta[key] is not None and meta[key] not in eos:
            eos.append(meta[key])
    from gguf import TokenType
    control = {tok for tok, typ in zip(meta["tokens"], meta["token_types"])
               if typ == TokenType.CONTROL}
    for marker in _END_OF_TEXT:
        if marker in control and tokenizer.convert_tokens_to_ids(marker) not in eos:
            eos.append(tokenizer.convert_tokens_to_ids(marker))
    settings = {"eos_token_id": eos, "pad_token_id": tokenizer.pad_token_id}
    if meta["bos_token_id"] is not None:
        settings["bos_token_id"] = meta["bos_token_id"]
    return settings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True, help="path to a full-precision .gguf file")
    ap.add_argument("--out", required=True, help="output HF-format directory")
    ap.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="output dtype (default: the GGUF's own precision)",
    )
    args = ap.parse_args()
    args.gguf = os.path.abspath(args.gguf)
    args.out = os.path.abspath(args.out)

    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    meta = read_gguf_metadata(args.gguf)
    source_dtype = native_dtype(meta["file_type"])
    out_dtype = source_dtype if args.dtype == "auto" else args.dtype
    if out_dtype != source_dtype:
        print(f"  note: GGUF is {source_dtype}; writing {out_dtype} is not lossless", flush=True)

    print(f"[1/4] Loading GGUF model: {args.gguf}", flush=True)
    # transformers 5.x GGUF API: pass the containing dir as the model id and the
    # filename via gguf_file=. Works for local files (resolved_archive_file path).
    gguf_dir, gguf_name = os.path.dirname(args.gguf), os.path.basename(args.gguf)
    model = AutoModelForCausalLM.from_pretrained(
        gguf_dir, gguf_file=gguf_name, low_cpu_mem_usage=True
    )
    model = model.to(getattr(torch, out_dtype))

    print("[2/4] Building tokenizer from GGUF metadata", flush=True)
    tokenizer = load_tokenizer(args.gguf, meta)
    gen = generation_settings(meta, tokenizer)
    model.config.pad_token_id = gen["pad_token_id"]
    model.generation_config = GenerationConfig(**gen)

    print(f"[3/4] Saving safetensors to {args.out} (dtype={out_dtype})", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)
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
    print(f"  eos_token_id : {reloaded.generation_config.eos_token_id}")
    print(f"  pad_token_id : {reloaded.generation_config.pad_token_id}")

    from transformers import AutoTokenizer
    rt = AutoTokenizer.from_pretrained(args.out)
    enc = rt("Hello, world", return_tensors="pt").input_ids
    print(f"  tokenizer round-trip: {rt.decode(enc[0])!r}")
    split = [t for t in rt.added_tokens_encoder
             if len(rt.encode(t, add_special_tokens=False)) != 1]
    print(f"  added tokens : {len(rt.added_tokens_encoder)}"
          + (f", SPLIT: {split}" if split else ", all single ids"))
    template = getattr(rt, "chat_template", None)
    print(f"  chat_template present: {bool(template)}"
          + ("" if template else " — set the template in LLaMA-Factory"))

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
