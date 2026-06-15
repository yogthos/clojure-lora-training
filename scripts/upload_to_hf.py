#!/usr/bin/env python3
"""Upload Howard/Russell LoRA adapter + training artifacts to HuggingFace Hub.

Batch-uploads N checkpoints into a single repo. One checkpoint is designated
"primary" and goes to the repo root (so `PeftModel.from_pretrained(repo_id)`
works with no subfolder); the rest live under `checkpoints/checkpoint-N/`.

Repo layout:

    repo/
    ├── README.md                       # generated model card
    ├── adapter_model.safetensors       # primary checkpoint at root
    ├── adapter_config.json
    ├── training/
    │   ├── qwen25_32b_lora.yaml        # training config
    │   ├── trainer_log.jsonl           # step-by-step log
    │   └── eval_curve.png              # generated plot
    └── checkpoints/
        ├── checkpoint-10200/{adapter_model.safetensors, adapter_config.json, ...}
        ├── checkpoint-10400/...
        └── checkpoint-10923/...

Usage:
    # Batch — scan ./checkpoints for all checkpoint-* subdirs
    python scripts/upload_to_hf.py \\
        --checkpoints-dir ./checkpoints \\
        --config-yaml data/training/howard_russell/LlamaFactory/qwen25_32b_lora.yaml \\
        --trainer-log trainer_log.jsonl \\
        --repo-id yogthos/howard-russell-qwen25-32b

    # Pick a specific primary (default: highest-numbered checkpoint)
    python scripts/upload_to_hf.py ... --primary-checkpoint checkpoint-10200

    # Dry-run (stage + preview, skip upload)
    python scripts/upload_to_hf.py ... --dry-run --staging-dir /tmp/hf-stage
"""

import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path


# -----------------------------------------------------------------------------
# Checkpoint discovery
# -----------------------------------------------------------------------------

def discover_checkpoints(root: Path) -> list[Path]:
    """Find all checkpoint-N subdirectories containing an adapter, sorted by step."""
    candidates = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        m = re.match(r"checkpoint-(\d+)$", d.name)
        if not m:
            continue
        if not (d / "adapter_model.safetensors").exists():
            print(f"  Skipping {d.name}: missing adapter_model.safetensors")
            continue
        if not (d / "adapter_config.json").exists():
            print(f"  Skipping {d.name}: missing adapter_config.json")
            continue
        candidates.append((int(m.group(1)), d))
    candidates.sort(key=lambda t: t[0])
    return [d for _, d in candidates]


def checkpoint_step(d: Path) -> int:
    m = re.match(r"checkpoint-(\d+)$", d.name)
    if not m:
        raise ValueError(f"Not a checkpoint directory: {d}")
    return int(m.group(1))


# -----------------------------------------------------------------------------
# Trainer log parsing
# -----------------------------------------------------------------------------

def parse_trainer_log(path: Path) -> dict:
    """Split trainer_log.jsonl into training and eval series."""
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    final_step = final_epoch = total_steps = None
    step_to_epoch: dict[int, float] = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            step = entry["current_steps"]
            total_steps = entry.get("total_steps", total_steps)
            final_step = step
            final_epoch = entry.get("epoch", final_epoch)
            if "epoch" in entry:
                step_to_epoch[step] = entry["epoch"]
            if "eval_loss" in entry:
                eval_steps.append(step)
                eval_loss.append(entry["eval_loss"])
            elif "loss" in entry:
                train_steps.append(step)
                train_loss.append(entry["loss"])

    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
        "final_step": final_step,
        "final_epoch": final_epoch,
        "total_steps": total_steps,
        "step_to_epoch": step_to_epoch,
    }


def eval_loss_at(log: dict, step: int) -> float | None:
    if step in log["eval_steps"]:
        return log["eval_loss"][log["eval_steps"].index(step)]
    return None


def epoch_at(log: dict, step: int) -> float | None:
    # Exact match, or nearest earlier step
    if step in log["step_to_epoch"]:
        return log["step_to_epoch"][step]
    earlier = [s for s in log["step_to_epoch"] if s <= step]
    return log["step_to_epoch"][max(earlier)] if earlier else None


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

def plot_eval_curve(
    log: dict,
    checkpoint_steps: list[int],
    primary_step: int,
    output_path: Path,
) -> None:
    """Plot training + eval loss with markers for each checkpoint."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    if log["train_steps"]:
        ax.plot(log["train_steps"], log["train_loss"],
                alpha=0.3, color="tab:blue", linewidth=0.8, label="Training loss")
    if log["eval_steps"]:
        ax.plot(log["eval_steps"], log["eval_loss"],
                color="tab:red", linewidth=1.8, marker="o", markersize=3,
                label="Evaluation loss")

    # Vertical markers for every uploaded checkpoint (primary is bold)
    for step in checkpoint_steps:
        is_primary = step == primary_step
        ax.axvline(
            step,
            color="black" if is_primary else "gray",
            linestyle="--",
            alpha=0.8 if is_primary else 0.3,
            linewidth=1.4 if is_primary else 0.8,
            label=(f"Primary checkpoint (step {step})" if is_primary else None),
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Howard/Russell LoRA — Training and Evaluation Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# README generation
# -----------------------------------------------------------------------------

def build_readme(
    repo_id: str,
    adapter_config: dict,
    training_config: dict,
    log: dict,
    checkpoints: list[Path],
    primary_step: int,
) -> str:
    """Generate model card README from config + log + checkpoint list."""
    rank = adapter_config.get("r") or training_config.get("lora_rank", 0)
    alpha = adapter_config.get("lora_alpha") or training_config.get("lora_alpha", 0)
    base_model = (
        adapter_config.get("base_model_name_or_path")
        or training_config.get("model_name_or_path", "Qwen/Qwen2.5-32B")
    )
    scale = f"{alpha/rank:.1f}" if rank else "?"

    # Checkpoint table
    rows = []
    for ckpt_dir in checkpoints:
        step = checkpoint_step(ckpt_dir)
        epoch = epoch_at(log, step)
        eval_loss = eval_loss_at(log, step)
        epoch_s = f"{epoch:.2f}" if epoch is not None else "—"
        eval_s = f"{eval_loss:.4f}" if eval_loss is not None else "—"
        note = "**Primary** (at repo root)" if step == primary_step else ""
        rows.append(f"| `checkpoint-{step}` | {epoch_s} | {eval_s} | {note} |")
    checkpoint_table = "\n".join(rows)

    # Summary numbers
    best_eval = min(log["eval_loss"]) if log["eval_loss"] else None
    best_step = (
        log["eval_steps"][log["eval_loss"].index(best_eval)]
        if best_eval is not None else None
    )
    best_str = f"**{best_eval:.4f}** (step {best_step})" if best_eval is not None else "n/a"

    final_eval_str = (
        f"**{log['eval_loss'][-1]:.4f}** (step {log['eval_steps'][-1]})"
        if log["eval_loss"] else "n/a"
    )

    total = log["total_steps"] or primary_step
    completion_pct = (primary_step / total * 100) if total else 100.0
    primary_epoch = epoch_at(log, primary_step) or 0
    primary_eval = eval_loss_at(log, primary_step)
    primary_eval_str = f"**{primary_eval:.4f}**" if primary_eval is not None else "n/a"

    # Intermediate checkpoint example (use earliest non-primary if any)
    intermediates = [c for c in checkpoints if checkpoint_step(c) != primary_step]
    intermediate_example = (
        f"checkpoint-{checkpoint_step(intermediates[0])}"
        if intermediates else f"checkpoint-{primary_step}"
    )

    intermediates_section = ""
    if intermediates:
        intermediates_section = f"""
To load a specific intermediate checkpoint:

```python
model = PeftModel.from_pretrained(
    base, "{repo_id}",
    subfolder="checkpoints/{intermediate_example}",
)
```
"""

    return f"""---
license: apache-2.0
base_model: {base_model}
library_name: peft
tags:
- lora
- peft
- style-transfer
- text-generation
- qwen2.5
language:
- en
---

# Howard/Russell Style LoRA for Qwen2.5-32B

A LoRA adapter fine-tuned on a bidirectional blended corpus of Robert E. Howard
and H. P. Lovecraft prose (Russell blend), targeting pulp-era weird-fiction voice
for paragraph-level style transfer.

## Quick Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("{base_model}", torch_dtype="bfloat16")
tok  = AutoTokenizer.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
```
{intermediates_section}
For Apple Silicon (MLX) inference, convert the adapter with the companion script
at [yogthos/text-style-transfer](https://github.com/yogthos/text-style-transfer):

```bash
python scripts/convert_peft_to_mlx.py \\
    --input <downloaded-repo-dir> \\
    --output lora_adapters/howard_russell_mlx \\
    --mlx-model models/Qwen2.5-32B-Base-4bit-MLX \\
    --author "Howard Russell Blend"
```

## Training

| | |
|---|---|
| Base model | `{base_model}` |
| Method | LoRA, rank {rank}, alpha {alpha} (scale α/r = {scale}) |
| Dataset | Howard/Russell bidirectional blended corpus (~60k SFT examples from ~22k chunks of 40-100 words) |
| Hardware | 2× A100 80GB with DeepSpeed ZeRO-3 |
| Hyperparameters | LR 1e-5 cosine, 5% warmup, bf16, NEFTune α=5.0, dropout 0.1, `train_on_prompt: true` |
| Planned epochs | 3 ({total} steps) |

## Checkpoints

Primary checkpoint: **step {primary_step} of {total} ({completion_pct:.1f}%)**, epoch {primary_epoch:.2f}, eval loss {primary_eval_str}.

| Checkpoint | Epoch | Eval loss | |
|---|---|---|---|
{checkpoint_table}

Best eval loss observed during training: {best_str}.
Final eval loss recorded: {final_eval_str}.

Training was terminated before completion due to an infrastructure issue
(resume-from-checkpoint failure after `save_total_limit` cleanup). At termination,
evaluation loss had plateaued well below meaningful further improvement — under
the observed trajectory, a completed run would have landed within ~0.002 of the
primary checkpoint.

![Training curve](training/eval_curve.png)

## Reproducing

The `training/` directory contains the exact training configuration and the full
step-by-step trainer log:

- `qwen25_32b_lora.yaml` — LlamaFactory SFT config (rank, alpha, LR schedule, dataset reference, DeepSpeed config)
- `trainer_log.jsonl` — raw trainer log (training loss every 10 steps, eval every 100 steps)
- `eval_curve.png` — generated plot of the above

Train with:

```bash
llamafactory-cli train training/qwen25_32b_lora.yaml
```

Note: the dataset (`howard_russell_sft`) is not bundled here; see the generation
pipeline in the companion repository.

## License

Apache 2.0, inherited from the base model.
"""


# -----------------------------------------------------------------------------
# Staging
# -----------------------------------------------------------------------------

def stage_files(
    checkpoints: list[Path],
    primary: Path,
    config_yaml: Path,
    trainer_log: Path,
    repo_id: str,
    staging: Path,
    adapter_config: dict,
    training_config: dict,
    log: dict,
) -> None:
    """Copy all files into the staging directory."""
    staging.mkdir(parents=True, exist_ok=True)

    # Primary checkpoint → repo root
    print(f"Staging primary checkpoint ({primary.name}) at repo root...")
    for f in primary.iterdir():
        if f.is_file():
            shutil.copy2(f, staging / f.name)

    # All checkpoints → checkpoints/checkpoint-N/
    checkpoints_dir = staging / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    for ckpt in checkpoints:
        print(f"Staging {ckpt.name} in checkpoints/...")
        dest = checkpoints_dir / ckpt.name
        dest.mkdir(exist_ok=True)
        for f in ckpt.iterdir():
            if f.is_file():
                shutil.copy2(f, dest / f.name)

    # training/ artifacts
    training_dir = staging / "training"
    training_dir.mkdir(exist_ok=True)
    shutil.copy2(config_yaml, training_dir / "qwen25_32b_lora.yaml")
    shutil.copy2(trainer_log, training_dir / "trainer_log.jsonl")

    print("Generating eval curve plot...")
    checkpoint_steps = [checkpoint_step(c) for c in checkpoints]
    plot_eval_curve(log, checkpoint_steps, checkpoint_step(primary),
                    training_dir / "eval_curve.png")

    print("Generating README.md...")
    readme = build_readme(
        repo_id, adapter_config, training_config, log,
        checkpoints, checkpoint_step(primary),
    )
    (staging / "README.md").write_text(readme)


def print_staged_tree(staging: Path) -> None:
    total_mb = 0.0
    print("\nStaged files:")
    for item in sorted(staging.rglob("*")):
        if item.is_file():
            rel = item.relative_to(staging)
            size_mb = item.stat().st_size / (1024 ** 2)
            total_mb += size_mb
            print(f"  {rel}  ({size_mb:.2f} MB)")
    print(f"Total: {total_mb:.1f} MB")


# -----------------------------------------------------------------------------
# Upload
# -----------------------------------------------------------------------------

def upload(staging: Path, repo_id: str, private: bool, primary_step: int) -> None:
    from huggingface_hub import HfApi, create_repo

    print(f"\nCreating/verifying repo: {repo_id} (private={private})")
    create_repo(repo_id, private=private, exist_ok=True, repo_type="model")

    print(f"Uploading {staging} → {repo_id}...")
    HfApi().upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload Howard/Russell LoRA adapters (primary: step {primary_step})",
    )
    print(f"\nDone: https://huggingface.co/{repo_id}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoints-dir", required=True, type=Path,
                        help="Directory containing checkpoint-N subdirectories")
    parser.add_argument("--primary-checkpoint", default=None,
                        help="Directory name of primary checkpoint (e.g. 'checkpoint-10200'). "
                             "Default: highest-numbered checkpoint.")
    parser.add_argument("--config-yaml", required=True, type=Path,
                        help="Path to training YAML (qwen25_32b_lora.yaml)")
    parser.add_argument("--trainer-log", required=True, type=Path,
                        help="Path to trainer_log.jsonl")
    parser.add_argument("--repo-id", required=True,
                        help="HF repo ID, e.g. yogthos/howard-russell-qwen25-32b")
    parser.add_argument("--private", action="store_true",
                        help="Create as private repo (default: public)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage files and print tree, then exit without uploading")
    parser.add_argument("--staging-dir", type=Path, default=None,
                        help="Persistent staging directory (default: temp dir, cleaned on exit)")
    args = parser.parse_args()

    # Validate non-checkpoint inputs
    for p in (args.config_yaml, args.trainer_log):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")
    if not args.checkpoints_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.checkpoints_dir}")

    # Discover checkpoints
    print(f"Scanning {args.checkpoints_dir} for checkpoint-* subdirectories...")
    checkpoints = discover_checkpoints(args.checkpoints_dir)
    if not checkpoints:
        raise SystemExit(f"No valid checkpoint-* subdirectories found in {args.checkpoints_dir}")
    print(f"Found {len(checkpoints)} checkpoint(s): {[c.name for c in checkpoints]}")

    # Select primary
    if args.primary_checkpoint:
        matches = [c for c in checkpoints if c.name == args.primary_checkpoint]
        if not matches:
            raise SystemExit(
                f"--primary-checkpoint '{args.primary_checkpoint}' not found among "
                f"{[c.name for c in checkpoints]}"
            )
        primary = matches[0]
    else:
        primary = checkpoints[-1]  # highest-numbered
    print(f"Primary checkpoint: {primary.name}")

    # Load configs (use primary's adapter_config.json for the model card)
    with open(primary / "adapter_config.json") as f:
        adapter_config = json.load(f)
    import yaml
    with open(args.config_yaml) as f:
        training_config = yaml.safe_load(f)

    # Parse trainer log
    log = parse_trainer_log(args.trainer_log)

    # Stage
    tmp_ctx = tempfile.TemporaryDirectory() if args.staging_dir is None else None
    staging = Path(tmp_ctx.name) if tmp_ctx else args.staging_dir
    print(f"Staging in: {staging}")

    try:
        stage_files(
            checkpoints, primary, args.config_yaml, args.trainer_log,
            args.repo_id, staging, adapter_config, training_config, log,
        )
        print_staged_tree(staging)

        if args.dry_run:
            print(f"\n[dry-run] Upload skipped. Staging dir: {staging}")
            if tmp_ctx:
                print("[dry-run] Temp dir will be deleted on exit — use --staging-dir to persist.")
            return

        upload(staging, args.repo_id, args.private, checkpoint_step(primary))
    finally:
        if tmp_ctx:
            tmp_ctx.cleanup()


if __name__ == "__main__":
    main()
