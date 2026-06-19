# Clojure LoRA Trainer

Train Qwen-based models on Clojure code evolution using the **Code Flow** paradigm. Mines git repositories for commit transitions, generates synthetic training data from Clojure feature taxonomies, and assembles JSONL datasets for LLaMA-Factory fine-tuning.

The trained model learns to develop Clojure code interactively via nREPL — evaluating forms, inspecting results, and applying unified diffs — rather than producing static code snapshots.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   git_mining/    │    │   synthetic/      │    │   assembly/      │
│                     │    │                     │    │                     │
│ • commit diffs    │    │ • feature trees    │    │ • merge & dedup  │
│ • session groups  │    │ • code generation  │    │ • balance types  │
│ • before/after    │    │ • question gen     │    │ • LLaMA-Factory  │
│   multi-file      │    │ • feature evo      │    │   formatting     │
└────────┬────────┘    └────────┬─────────┘    └────────┬────────┘
         │                      │                        │
         ▼                      ▼                        ▼
    git-mined              synthetic                  training
    JSONL pairs            JSONL pairs               dataset
                                                        │
                                              ┌─────────▼─────────┐
                                              │  LLaMA-Factory     │
                                              │  LoRA fine-tune    │
                                              │  Qwen3.6-Coder-27B │
                                              └───────────────────┘
```

### Key modules

- **`src/shared.py`** — Single source of truth: system prompt, JSONL I/O, dedup keys
- **`src/codeflow/git_mining/`** — Mines Clojure repos for commit transitions with multi-file diffs
- **`src/codeflow/synthetic/`** — Generates instruction/response pairs from Clojure feature taxonomies
- **`src/codeflow/assembly/`** — Merges, deduplicates, balances, and formats the final dataset
- **`src/codeflow/synthetic/prompts/`** — Prompt templates as standalone `.txt` files (editable without touching Python)

## Quick Start (uv)

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and task running.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync dependencies
git clone <repo-url> && cd clojure-lora-trainer
uv sync

# Set up your config (edit values to match your setup)
cp config.json.sample config.json

# Set required environment variables
export DEEPSEEK_API_KEY="your-api-key"

# Mining: extract commit histories from Clojure repos
uv run python3 scripts/mine_clojure_repos.py \
    --repo /path/to/clojure-project \
    --output data/git-mining/output

# Synthetic: generate training data from Clojure feature taxonomy
uv run python3 scripts/generate_synthetic_data.py \
    --features-file data/features/clojure_features.json \
    --output data/synthetic/output \
    --target 500

# Assembly: merge, deduplicate, balance, format
uv run python3 scripts/assemble_codeflow_dataset.py \
    --git-dir data/git-mining/output \
    --synth-dir data/synthetic/output \
    --output data/training/codeflow.jsonl

# Run tests
uv run python3 -m pytest

# Run the style transfer REPL
uv run python3 restyle.py
```

### Running without uv

If you prefer pip + venv:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .
# Then use python3 and pytest directly (no uv prefix needed)
```

## Configuration

Copy `config.json.sample` to `config.json` and edit it. The sample file documents all available options.

Environment variables in config values use `${VAR}` syntax (e.g. `"api_key": "${DEEPSEEK_API_KEY}"`).

## Training Data Format

Each record is a JSON object compatible with LLaMA-Factory:

```json
{
  "instruction": "Fix the race condition in the core.async pipeline...",
  "input": "",
  "output": ";; nREPL session:\n;; eval: (require '[clojure.core.async :refer [chan go <!]])\n;; result: nil\n;; ...\n;; apply:\ndiff --git a/src/pipeline.clj b/src/pipeline.clj\n...",
  "system": "You are a Clojure coding agent using nREPL-driven development..."
}
```

The output format interleaves REPL evaluation blocks with final unified diffs, teaching the model to develop interactively.

## Training Configuration

Target: **Qwen3.6-Coder-27B** (BF16, ~54GB) on RunPod A100 80GB.

| Parameter | Value |
|-----------|-------|
| LoRA rank | 64 |
| LoRA alpha | 128 |
| Target modules | All linear layers |
| Learning rate | 3e-4 cosine schedule |
| Batch size | 2–4 |
| Gradient accumulation | → effective batch 16–32 |
| Epochs | 3 |
| Max context | 8K (expandable with gradient checkpointing) |
| Training duration | 6–12 hours on A100 |

Inference runs locally on MLX (Apple Silicon).

## Requirements

- Python 3.10+
- PyTorch 2.0+, Transformers 4.36+, PEFT 0.18+
- For cloud training: A100 80GB or H100 80GB (RunPod)
- For local inference: Apple Silicon Mac with MLX

## License

MIT
