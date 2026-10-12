"""Tokenizer and generation settings from scripts/convert_gguf_to_safetensors.py."""

import importlib.util
import pathlib

import pytest

gguf = pytest.importorskip("gguf")

_VOCAB = pathlib.Path("~/src/llama.cpp/models/ggml-vocab-qwen35.gguf").expanduser()

# Qwen3's added tokens as llama.cpp's convert_hf_to_gguf types them: CONTROL
# for special tokens, USER_DEFINED for added tokens that aren't special. It
# also promotes the FIM tokens to CONTROL.
_C, _U = gguf.TokenType.CONTROL, gguf.TokenType.USER_DEFINED
_QWEN3_ADDED = {
    151643: ("<|endoftext|>", _C), 151644: ("<|im_start|>", _C),
    151645: ("<|im_end|>", _C), 151646: ("<|object_ref_start|>", _C),
    151647: ("<|object_ref_end|>", _C), 151648: ("<|box_start|>", _C),
    151649: ("<|box_end|>", _C), 151650: ("<|quad_start|>", _C),
    151651: ("<|quad_end|>", _C), 151652: ("<|vision_start|>", _C),
    151653: ("<|vision_end|>", _C), 151654: ("<|vision_pad|>", _C),
    151655: ("<|image_pad|>", _C), 151656: ("<|video_pad|>", _C),
    151657: ("<tool_call>", _U), 151658: ("</tool_call>", _U),
    151659: ("<|fim_prefix|>", _C), 151660: ("<|fim_middle|>", _C),
    151661: ("<|fim_suffix|>", _C), 151662: ("<|fim_pad|>", _C),
    151663: ("<|repo_name|>", _C), 151664: ("<|file_sep|>", _C),
    151665: ("<tool_response>", _U), 151666: ("</tool_response>", _U),
    151667: ("<think>", _U), 151668: ("</think>", _U),
}


def _script():
    spec = importlib.util.spec_from_file_location(
        "convert_gguf", pathlib.Path("scripts/convert_gguf_to_safetensors.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_vocab(path, eos_id=151643, pad_id=151643, file_type=1):
    """A vocab-only GGUF shaped like a Qwen3 chat model's."""
    src = gguf.GGUFReader(_VOCAB)
    tokens = list(src.fields["tokenizer.ggml.tokens"].contents())
    types = list(src.fields["tokenizer.ggml.token_type"].contents())
    for i, (tok, typ) in _QWEN3_ADDED.items():
        tokens[i], types[i] = tok, int(typ)
    w = gguf.GGUFWriter(str(path), "qwen2")
    for key, field in src.fields.items():
        if key.startswith("GGUF.") or key in ("general.architecture", "general.file_type",
                                              "tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
                                              "tokenizer.ggml.eos_token_id",
                                              "tokenizer.ggml.padding_token_id"):
            continue
        sub = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(key, field.contents(), field.types[0], sub_type=sub)
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_eos_token_id(eos_id)
    w.add_pad_token_id(pad_id)
    w.add_file_type(file_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


pytestmark = pytest.mark.skipif(not _VOCAB.exists(), reason="llama.cpp vocab GGUF not present")


@pytest.fixture(scope="module")
def tokenizer_and_meta(tmp_path_factory):
    mod = _script()
    path = _write_vocab(tmp_path_factory.mktemp("gguf") / "vocab.gguf")
    meta = mod.read_gguf_metadata(path)
    return mod, mod.load_tokenizer(path, meta), meta


@pytest.mark.parametrize("tok", ["<think>", "</think>", "<tool_call>", "<|fim_prefix|>",
                                 "<|im_end|>", "<|vision_pad|>"])
def test_added_tokens_encode_to_one_id(tokenizer_and_meta, tok):
    _, tokenizer, _ = tokenizer_and_meta
    ids = tokenizer.encode(tok, add_special_tokens=False)
    assert ids == [next(i for i, (t, _) in _QWEN3_ADDED.items() if t == tok)]


def test_special_flags_follow_token_types(tokenizer_and_meta):
    # <think> must survive skip_special_tokens decoding like it does upstream.
    _, tokenizer, _ = tokenizer_and_meta
    added = {t.content: t.special for t in tokenizer.added_tokens_decoder.values()}
    assert added["<|im_end|>"] is True
    assert added["<think>"] is False
    assert "[PAD151700]" not in added


@pytest.mark.parametrize("gguf_eos", [151643, 151645])
def test_generation_stops_on_im_end_and_endoftext(tmp_path, gguf_eos):
    # llama.cpp writes the HF config's eos (<|im_end|> for Qwen3 chat models)
    # as the GGUF's EOS; other converters write <|endoftext|>. Upstream's
    # generation_config stops on both either way.
    mod = _script()
    path = _write_vocab(tmp_path / "vocab.gguf", eos_id=gguf_eos)
    meta = mod.read_gguf_metadata(path)
    tokenizer = mod.load_tokenizer(path, meta)
    gen = mod.generation_settings(meta, tokenizer)
    assert gen["eos_token_id"] == [151645, 151643]
    assert gen["pad_token_id"] == 151643


def test_native_dtype_from_file_type():
    mod = _script()
    assert mod.native_dtype(0) == "float32"
    assert mod.native_dtype(1) == "float16"
    assert mod.native_dtype(32) == "bfloat16"
    with pytest.raises(ValueError, match="quantized"):
        mod.native_dtype(15)  # Q4_K_M


def test_saved_tokenizer_keeps_added_tokens(tokenizer_and_meta, tmp_path):
    from transformers import AutoTokenizer
    _, tokenizer, _ = tokenizer_and_meta
    tokenizer.save_pretrained(tmp_path)
    rt = AutoTokenizer.from_pretrained(tmp_path)
    assert rt.encode("<think>x</think>", add_special_tokens=False)[0] == 151667
    assert rt.eos_token == "<|endoftext|>" and rt.pad_token == "<|endoftext|>"
    assert rt.chat_template
