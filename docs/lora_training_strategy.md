# LoRA Training Strategy for Style Transfer

This document summarizes our findings on training LoRA adapters for style transfer and outlines strategies for improving fact preservation.

## Critical Insight: Training/Inference Distribution Matching

**The most important factor for quality output is matching inference conditions to training conditions.**

The LoRA learns specific patterns from training data. If inference uses different:
- Input format (clean vs perturbed)
- Persona frames
- Content classification
- Constraint patterns

...the model produces mechanical, formulaic output because it's operating outside its training distribution.

### What Must Match Between Training and Inference

| Factor | Training | Inference | Implementation |
|--------|----------|-----------|----------------|
| **Input perturbation** | 8% noise (typos, word drops) | Same 8% noise | `apply_input_perturbation: true` |
| **Persona frames** | From `PERSONA_FRAMES` dict | From `prompts/{author}_worldview.txt` | Must be identical text |
| **Content classifier** | `classify_content_type()` | Same function | `src/utils/content_classifier.py` |
| **Constraints** | Probabilistic (70%, 40%, 30%) | Same probabilities | `prompt_builder.py` |
| **Scale** | 2.0 during training | 2.0 in config | `config.json` |
| **Word count ratio** | ~1.21x average | Cannot force expansion | Use `expand_for_texture` |

### Why Input Perturbation Matters

Training data uses 8% perturbation:
- Typos (swapped characters)
- Word drops (articles, filler words)
- Synonym swaps
- Adjective drops (30% chance)

The model learned: **"Take broken/degraded input → produce rich prose"**

Without perturbation at inference, the model receives clean RTT text and produces mechanical output because it has no "room" to exercise creative reconstruction.

**Solution**: Enable `apply_input_perturbation: true` in config.json

### Why LoRA Cannot Expand Text

The LoRA was trained with (neutralized_text, styled_text) pairs of SIMILAR lengths (~1.21x expansion ratio). This means:
- Word count instructions are ignored when they differ significantly from input
- `target_expansion_ratio` alone does NOT work
- No amount of scale adjustment will make the LoRA expand

**Solution**: Enable `expand_for_texture: true` to pre-expand content via critic model BEFORE RTT/LoRA.

## Problem Statement

Standard LoRA fine-tuning on style corpora teaches the model to generate text *in the style of* an author, but introduces two critical failure modes:

1. **Fact Hallucination**: Model adds information not present in input
2. **Fact Dropping**: Model omits key entities, dates, or relationships from the input

These failures occur because standard training only teaches *what the output should look like*, not *what invariants must be preserved*.

## Key Findings

### Inference Settings (UPDATED)

| Setting | Optimal Value | Notes |
|---------|---------------|-------|
| Scale | **2.0** | Must match training scale |
| Temperature | 0.8 | Higher allows more creativity |
| Input Perturbation | **Enabled** | Critical for non-mechanical output |
| expand_for_texture | **Enabled** | Required for text expansion |
| RTT Neutralization | Enabled | Matches training pipeline |

### Prompt Complexity Paradox

Counter-intuitively, **complex prompts cause more hallucinations**:

```
# BAD - causes more hallucinations
"Rephrase in X's style. Keep all facts. Add nothing new. Preserve names exactly."

# GOOD - fewer hallucinations
"Rephrase in X's prose style."
```

The model appears to interpret detailed instructions as permission to be "creative" about facts.

## Training Strategy 1: Convergent Training

### Concept

Instead of the standard one-to-one mapping:
```
input_A → output_A
input_B → output_B
```

We use many-to-one (convergent) mapping:
```
input_A1 → output_A
input_A2 → output_A  (same output!)
input_A3 → output_A  (same output!)
```

Multiple neutral phrasings of the same content all map to the **same** author text.

### Why It Works

1. **Entity Invariance**: If "Karl Marx" appears in ALL inputs AND the output, the model learns it's essential and must be preserved.

2. **Reduces Hallucination**: The model learns to map TO a fixed target, not generate freely. It can't add "(1818-1883)" because the target doesn't have that.

3. **Style Focus**: With content held constant across variations, the only learnable pattern is the STYLE transformation.

4. **Robustness**: Model sees many ways to express the same idea, learns what's essential vs. surface-level.

### Implementation

```python
# For each author paragraph:
original = "In the shadowed corridors of philosophical inquiry..."
entities = extract_entities(original)  # ["Karl Marx", "Dialectical Materialism"]

# Generate neutral variations
variations = [
    "Karl Marx developed Dialectical Materialism...",
    "Dialectical Materialism was developed by Karl Marx...",
    "The methodology called Dialectical Materialism, created by Karl Marx...",
]

# Validate each variation preserves entities
for var in variations:
    if not all(entity in var for entity in entities):
        reject(var)  # Missing critical entity
    if has_added_entities(var, original):
        reject(var)  # Added facts not in original

# All validated variations → SAME output
training_examples = [
    {"input": var, "output": original}
    for var in validated_variations
]
```

### Validation Rules

Variations are rejected if:
- Missing critical entities (names, terms > 4 chars, numbers)
- Added more than 3 new entities not in original
- Too short (< 20 chars)

## Training Strategy 2: Repair Training (Proposed)

### Concept

Train the LoRA to not only *generate* styled text, but also *repair* outputs that violate fact preservation. This uses graph-based proposition diffing with no LLM calls.

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│  Original Text  │     │  LoRA Output    │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Proposition     │     │ Proposition     │
│ Graph (cached)  │     │ Graph (compute) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
            ┌─────────────────┐
            │   Graph Diff    │
            │   (no LLM)      │
            └────────┬────────┘
                     │
                     ▼
            ┌─────────────────┐
            │ Repair Training │
            │    Examples     │
            └─────────────────┘
```

### Proposition Graph Structure

Each text is decomposed into propositions (subject-verb-object triples):

```python
text = "Karl Marx developed Dialectical Materialism in the 19th century."

propositions = [
    ("Karl Marx", "developed", "Dialectical Materialism"),
    ("Dialectical Materialism", "developed_in", "19th century"),
]
```

### Graph Diff Categories

| Diff Type | Meaning | Repair Action |
|-----------|---------|---------------|
| Missing Proposition | Fact in original, not in output | Add back |
| Added Proposition | Fact in output, not in original | Remove |
| Modified Proposition | Same subject, different predicate/object | Correct |

### Repair Training Data Generation

```python
def generate_repair_examples(original_text, lora_output):
    """Generate repair training examples from LoRA errors."""

    # Build proposition graphs (no LLM, just spaCy/dependency parsing)
    original_graph = extract_propositions(original_text)
    output_graph = extract_propositions(lora_output)

    # Compute diff
    missing = original_graph - output_graph
    added = output_graph - original_graph

    if not missing and not added:
        return None  # Output is faithful, no repair needed

    # Create repair instruction
    repair_instruction = format_repair_instruction(
        output=lora_output,
        missing=missing,
        added=added,
    )

    return {
        "instruction": "Repair the following text to preserve all original facts.",
        "input": repair_instruction,
        "output": original_text,  # Target is the faithful original
    }

def format_repair_instruction(output, missing, added):
    """Format the repair task for the model."""
    parts = [f"Text: {output}"]

    if missing:
        props = "; ".join(f"{s} {p} {o}" for s, p, o in missing)
        parts.append(f"Missing facts: {props}")

    if added:
        props = "; ".join(f"{s} {p} {o}" for s, p, o in added)
        parts.append(f"Remove these hallucinations: {props}")

    return "\n".join(parts)
```

### Synthetic Corruption for Repair Training

To generate more repair training data without running LoRA:

```python
def generate_synthetic_corruption(original_text):
    """Create corrupted versions for repair training."""

    propositions = extract_propositions(original_text)
    corruptions = []

    # Type 1: Drop a proposition
    for prop in propositions:
        corrupted = remove_proposition(original_text, prop)
        corruptions.append({
            "corrupted": corrupted,
            "missing": [prop],
            "added": [],
        })

    # Type 2: Add hallucinated proposition
    fake_props = generate_plausible_hallucinations(original_text)
    for fake in fake_props:
        corrupted = insert_proposition(original_text, fake)
        corruptions.append({
            "corrupted": corrupted,
            "missing": [],
            "added": [fake],
        })

    return corruptions
```

### Combined Training Objective

The final LoRA training combines both strategies:

```
Training Data = Convergent Examples + Repair Examples

Convergent: Teaches "different inputs → same styled output"
Repair: Teaches "detect and fix fact violations"
```

### Expected Benefits

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Named Entity Preservation | ~60% | 90%+ |
| Hallucination Rate | ~30% | <5% |
| Repair Success (when prompted) | N/A | 80%+ |

### Observed Results (Convergent Training)

With strict validation + automatic repair:
- **100% paragraph success rate** (all paragraphs produce valid variations)
- **3 variations per paragraph** on average
- **~8 seconds per paragraph** with 2 workers
- **Zero fact deviation** in output training data

## Implementation Plan

### Phase 1: Convergent Training (Complete)

- [x] Generate neutral variations via LLM
- [x] Validate entity preservation before including in training
- [x] **STRICT validation**: Any missing/added entities = failure (no relaxed mode)
- [x] **Automatic repair**: Failed variations are repaired via LLM and re-validated
- [x] **Smart quote handling**: Unicode U+2019 (apostrophe), U+201C/U+201D (quotes)
- [x] **Parallel workers**: Configurable `--workers` for faster generation
- [x] Format for MLX LoRA training
- [x] Scripts: `generate_convergent_training_llm.py`, `prepare_lora_training.py`

**Key principle**: Fact deviation in training data = training for hallucinations. This is unacceptable. All variations must preserve 100% of facts from the original.

### Phase 2: Proposition Graph Infrastructure (Complete)

Already implemented in:
- `src/ingestion/proposition_extractor.py` - spaCy-based SVO extraction with:
  - Epistemic stance detection (factual, appearance, hypothetical)
  - Logical relation detection (contrast, cause, condition, example)
  - Content anchor detection (entities, quotes, statistics)
  - RST role classification (nucleus vs satellite)

- `src/validation/semantic_graph.py` - Graph comparison with:
  - `SemanticGraphBuilder.build_from_text()` - builds graph (no LLM)
  - `SemanticGraphComparator.compare()` - computes diff (no LLM)
  - `GraphDiff.to_repair_instructions()` - generates repair text
  - Entity role comparison for conflation/role swap detection

### Phase 3: Repair Training Data

- [x] Use existing graph infrastructure (no new code needed)
- [ ] Generate corrupted examples via `generate_repair_training.py`
- [ ] Cache source graphs for efficiency
- [ ] Combine with convergent training data
- [ ] Train combined LoRA

### Phase 4: Inference Integration

- [ ] After LoRA generation, compute proposition diff
- [ ] If diff detected, prompt LoRA with repair instruction
- [ ] Iterate until diff is empty or max attempts reached

## File Locations

```
scripts/
├── generate_convergent_training_llm.py  # Convergent training generation (LLM)
├── generate_convergent_training.py      # Rule-based variations (no LLM)
├── prepare_lora_training.py             # Combine and format for MLX
└── generate_repair_training.py          # Repair training from corruptions

src/
├── ingestion/
│   └── proposition_extractor.py         # spaCy-based SVO extraction
├── validation/
│   └── semantic_graph.py                # Graph structure, diff, comparison
└── generation/
    └── lora_generator.py                # LoRA inference with repair loop
```

## Training Data Format

### Messages Format (Required for Prompt Masking)

The training data must use the **messages format** to enable prompt masking:

```json
{
  "messages": [
    {"role": "system", "content": "Rephrase in Carl Sagan's prose style."},
    {"role": "user", "content": "The universe is very old and large..."},
    {"role": "assistant", "content": "The Cosmos is all that is or was..."}
  ]
}
```

**Why messages format?**
- Enables `--mask-prompt` which focuses learning on the assistant response only
- Without masking, the model wastes training signal learning to predict instruction tokens
- The `prepare_lora_training.py` script outputs this format automatically

### Repair Training Format

For repair examples, the system prompt indicates the task:

```json
{
  "messages": [
    {"role": "system", "content": "Fix the errors and rephrase in Carl Sagan's prose style."},
    {"role": "user", "content": "Text with errors:\n...\n\nErrors to fix:\n- Remove hallucination: '(1818-1883)' is not in the source\n\nOutput the corrected text:"},
    {"role": "assistant", "content": "The corrected text without hallucinations..."}
  ]
}
```

## MLX LoRA Configuration

### Recommended Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `model` | `mlx-community/Qwen3-8B-Base-bf16` | Base model, not instruct |
| `rank` | 32 | High capacity for complex multi-author patterns |
| `alpha` | 64 | 2x rank (standard heuristic) |
| `scale` | **1.0** | **Critical**: >1.0 increases hallucinations |
| `dropout` | 0.05 | Light regularization |
| `batch_size` | 2 | With grad_accum=8 → effective 16 |
| `grad_accumulation_steps` | 8 | Smooth gradients |
| `iters` | ~2000 | ~2 epochs for 15k examples |
| `learning_rate` | 1e-4 | Conservative, stable |
| `mask_prompt` | true | Focus on output style |
| `num_layers` | -1 | All layers |
| `max_seq_length` | 1024 | Covers all examples |
| `optimizer` | adamw | Weight decay prevents overfitting |

### Configuration File

See `data/lora_training/mixed_combined/lora_config.yaml` for a complete example:

```bash
mlx_lm.lora --config data/lora_training/mixed_combined/lora_config.yaml
```

### Calculating Iterations

```
steps_per_epoch = training_examples / effective_batch_size
effective_batch_size = batch_size × grad_accumulation_steps

Example: 15,435 examples / 16 = 964 steps/epoch
  - 1 epoch: ~1000 iters
  - 2 epochs: ~2000 iters (recommended)
  - 3 epochs: ~3000 iters (risk of overfitting)
```

## Quick Start: Generate Combined Training Data

```bash
# 1. Generate convergent training (multiple inputs → same output)
python scripts/generate_convergent_training_llm.py \
  --corpus data/corpus/sagan.txt \
  --author "Carl Sagan" \
  --output /tmp/sagan_convergent.jsonl \
  --variations 3 \
  --workers 30 \
  --max-paragraphs 50

# 2. Generate repair training (corrupted → original)
python scripts/generate_repair_training.py \
  --corpus data/corpus/sagan.txt \
  --author "Carl Sagan" \
  --output /tmp/sagan_repair.jsonl \
  --cache /tmp/sagan_graph_cache.json \
  --corruptions 3 \
  --workers 30 \
  --max-paragraphs 50

# 3. Combine into MLX format (outputs messages format)
python scripts/prepare_lora_training.py \
  --inputs /tmp/sagan_convergent.jsonl /tmp/sagan_repair.jsonl \
  --output-dir data/lora_training/sagan_combined

# 4. Train LoRA with config
mlx_lm.lora --config data/lora_training/sagan_combined/lora_config.yaml

# Or manually:
mlx_lm.lora \
    --model mlx-community/Qwen3-8B-Base-bf16 \
    --train-file data/training/lovecraft_flat.jsonl \
    --batch-size 1 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --lora-dropout 0.05 \
    --num-layers -1 \
    --mask-prompt \
    --save-every 2000 \
    --target-modules q_proj v_proj k_proj o_proj gate_proj down_proj up_proj \
    --learning-rate 1e-5 \
    --iters 7859 \
    --adapter-path lora_adapters/lovecraft
```


## Shared Modules (Critical for Consistency)

These modules are shared between training and inference. **DO NOT DUPLICATE** logic:

| Module | Purpose | Used By |
|--------|---------|---------|
| `src/utils/content_classifier.py` | Narrative vs conceptual detection | Training + Inference |
| `src/utils/perturbation.py` | Input noise (8% typos, drops) | Training + Inference |
| `prompts/{author}_worldview.txt` | Persona frames | Inference (must match training) |

### Content Classification

Both training and inference use the same spaCy-based classifier:
- Named entity detection (PERSON, GPE, LOC, FAC)
- Past tense verb counting
- Temporal marker detection
- Abstract vocabulary matching

**If classification differs, the model gets wrong persona frames and produces inconsistent output.**

## Troubleshooting Output Quality

### Problem: Output is mechanical/robotic/formulaic
- **Cause**: Inference distribution doesn't match training
- **Fix 1**: Enable `apply_input_perturbation: true`
- **Fix 2**: Verify persona frames match training exactly
- **Fix 3**: Check content classifier is working correctly

### Problem: Output doesn't expand (stays same length)
- **Cause**: LoRA trained on similar-length pairs (~1.21x)
- **Fix**: Enable `expand_for_texture: true`
- **Note**: `target_expansion_ratio` alone will NOT work

### Problem: Style is weak
- **Cause**: Scale too low
- **Fix**: Set `scale: 2.0` to match training
- **Note**: Scale > 2.5 may cause incoherence

### Problem: Output has repetitive phrases
- **Cause**: repetition_penalty too low
- **Fix**: Set `repetition_penalty: 1.08-1.15`
- **Note**: Too high (>1.2) kills characteristic author repetition

### Problem: Wrong persona frame type selected
- **Cause**: Content misclassified as narrative when conceptual
- **Fix**: Both use `src/utils/content_classifier.py`
- **Debug**: Check spaCy NER and POS output

## Quick Reference: config.json Settings

```json
{
  "generation": {
    "expand_for_texture": true,        // Pre-expand via critic model
    "apply_input_perturbation": true,  // 8% noise to match training
    "skip_neutralization": false,      // RTT must run
    "use_structural_rag": true,        // Rhythm patterns from corpus
    "use_structural_grafting": true,   // Argument skeletons
    "use_persona": true,               // Persona frames
    "target_expansion_ratio": 1.5,     // Word count target multiplier
    "rag_sample_size": 300             // Patterns to sample
  },
  "lora_adapters": {
    "lora_adapters/author": {
      "scale": 2.0,                    // MUST match training
      "temperature": 0.8,
      "top_p": 0.96,
      "min_p": 0.03,
      "repetition_penalty": 1.08,
      "worldview": "author_worldview.txt"
    }
  }
}
```

## References

- Convergent training concept inspired by contrastive learning
- Proposition extraction based on Open Information Extraction (OpenIE)
- Graph-based semantic comparison avoids LLM calls for validation
- Training/inference distribution matching based on empirical testing
