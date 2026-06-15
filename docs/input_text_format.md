# Input Text Format for LoRA Training

This document describes the ideal format for input text used in LoRA training and style transfer.

## Corpus Format

Author corpus files are plain text stored in `data/corpus/`. Each file contains prose from a single author.

### Structure

```
Paragraph one of author's writing. This is a complete thought
that may span multiple sentences. It flows naturally and
demonstrates the author's voice.

Paragraph two begins here. Double newlines separate paragraphs.
This is how we delineate discrete units of text for training.

Paragraph three continues the pattern.
```

**Key rules:**
- Paragraphs separated by double newlines (`\n\n`)
- No headers, titles, or chapter markers
- No metadata or annotations
- Pure prose only

### Paragraph Length

Optimal paragraph characteristics:
- **Minimum**: 30 words (shorter = insufficient style signal)
- **Maximum**: 200 words (longer = harder to learn consistent patterns)
- **Ideal**: 50-150 words

Paragraphs outside this range are automatically filtered during training data generation.

## Training Data Format (JSONL)

### Intermediate Format (from generation scripts)

The convergent training script produces JSONL with this structure:

```json
{
  "instruction": "Rephrase in Carl Sagan's prose style.",
  "input": "The universe is very old and very large...",
  "output": "The Cosmos is all that is or was or ever will be...",
  "group_id": "sagan_0042",
  "variation_num": 1
}
```

| Field | Description |
|-------|-------------|
| `instruction` | Fixed prompt for the LoRA task |
| `input` | Neutral/plain variation (what user provides) |
| `output` | Original author text (target style) |
| `group_id` | Groups variations of same paragraph |
| `variation_num` | Which variation (1, 2, 3...) |

### Final Format (for MLX LoRA training)

The `prepare_lora_training.py` script converts to **messages format**:

```json
{
  "messages": [
    {"role": "system", "content": "Rephrase in Carl Sagan's prose style."},
    {"role": "user", "content": "The universe is very old and very large..."},
    {"role": "assistant", "content": "The Cosmos is all that is or was or ever will be..."}
  ]
}
```

**Why messages format?**
- Required for `--mask-prompt` option in MLX LoRA
- Prompt masking focuses learning on the assistant response only
- Without masking, training signal is diluted by learning to predict instruction tokens

### Convergent Training Principle

Multiple neutral inputs map to the **same** styled output:

```
input_variation_1 → author_paragraph_A
input_variation_2 → author_paragraph_A  (same!)
input_variation_3 → author_paragraph_A  (same!)
```

This teaches the model to find **invariants** (facts that must be preserved) while learning style transformation.

## What Makes Good Corpus Text

### Ideal Characteristics

1. **Consistent voice**: Text from essays, books, or articles where the author's style is prominent
2. **Prose, not dialogue**: Narrative or expository text, not conversations
3. **Complete thoughts**: Each paragraph is self-contained
4. **Rich vocabulary**: Demonstrates the author's word choices and preferences
5. **Varied sentence structure**: Shows the author's rhythmic patterns

### Good Example (Hofstadter)

```
I slice up and devour tomatoes without the slightest sense of guilt.
I do not go to bed uneasily after having consumed a fresh tomato. It
does not occur to me to ask myself which tomato I ate, or whether by
eating it I have snuffed an inner light, nor do I believe it is
meaningful to try to imagine how the tomato felt as it was sitting on
my plate being sliced apart. To me, a tomato is a desireless, soulless,
nonconscious entity, and I have no qualms about doing with its "body"
as I like.
```

Why it works:
- Personal voice ("I slice", "I do not")
- Philosophical musing with concrete example
- Characteristic parenthetical asides
- Distinctive vocabulary ("snuffed an inner light")
- Clear paragraph boundary

### Good Example (Sagan)

```
The surface of the Earth is the shore of the cosmic ocean. On this
shore, we've learned most of what we know. Recently, we've waded a
little way out, maybe ankle-deep, and the water seems inviting. Some
part of our being knows this is where we came from. We long to return,
and we can, because the cosmos is also within us. We're made of
star-stuff. We are a way for the cosmos to know itself.
```

Why it works:
- Poetic imagery ("shore of the cosmic ocean")
- Builds from concrete to abstract
- Characteristic phrases ("star-stuff")
- Wonder and humility combined

### What to Avoid

| Problem | Example | Why Bad |
|---------|---------|---------|
| Headers/titles | `Chapter 3: The Mind` | Not prose, breaks paragraph detection |
| Dialogue heavy | `"Hello," she said. "Hi," he replied.` | Style is in narration, not dialogue |
| Lists/bullets | `• First point • Second point` | Not natural prose |
| Citations inline | `Smith (2019) argues that...` | Academic scaffolding, not voice |
| Too short | `It was raining.` | Insufficient style signal |
| Too long | (500+ word paragraph) | Dilutes style patterns |

## Preparing Corpus Text

### From Books/Articles

1. **Extract prose sections only** - skip tables of contents, indices, footnotes
2. **Remove chapter headers** - or convert to paragraph separators
3. **Strip page numbers and running headers**
4. **Normalize quotes** - convert smart quotes to straight quotes (or vice versa, but be consistent)
5. **Fix OCR errors** - if sourced from scanned text

### Cleaning Script Example

```bash
# Basic cleanup pipeline
cat raw_text.txt | \
  sed 's/^Chapter [0-9]*.*$/\n/g' | \  # Remove chapter headers
  sed 's/^[0-9]*$/\n/g' | \             # Remove page numbers
  tr -s '\n' | \                         # Collapse multiple newlines
  sed 's/  */ /g' > clean_corpus.txt    # Normalize spaces
```

### Validation

Check your corpus before training:

```bash
# Count paragraphs
grep -c '^$' data/corpus/author.txt

# Show paragraph length distribution
python3 -c "
text = open('data/corpus/author.txt').read()
paras = [p.strip() for p in text.split('\n\n') if p.strip()]
lengths = [len(p.split()) for p in paras]
print(f'Paragraphs: {len(paras)}')
print(f'Avg words: {sum(lengths)/len(lengths):.0f}')
print(f'Range: {min(lengths)}-{max(lengths)} words')
valid = [l for l in lengths if 30 < l < 200]
print(f'Valid (30-200 words): {len(valid)} ({100*len(valid)/len(lengths):.0f}%)')
"
```

## Fact Preservation Requirements

**Critical**: Training data must have zero fact deviation between input and output.

### What Must Match

| Element | Example | Must Preserve? |
|---------|---------|----------------|
| Named entities | "Karl Marx", "Stanford University" | **Yes** |
| Dates/years | "1991", "nineteenth century" | **Yes** |
| Quoted terms | "dialectical materialism" | **Yes** |
| Numbers | "fifteen years", "two children" | **Yes** |
| Core relationships | "X developed Y" | **Yes** |
| Adjectives/adverbs | "gloomy day" vs "dark day" | No (style varies) |
| Sentence structure | Order, length | No (style varies) |

### Validation Process

The training script (`generate_convergent_training_llm.py`) automatically:

1. Extracts entities from original text
2. Generates neutral variations
3. **Strictly validates** each variation (any missing/added entity = failure)
4. **Repairs** failed variations via LLM
5. **Re-validates** after repair
6. Only accepts variations with 100% fact preservation

This ensures the LoRA learns style transformation, not fact hallucination.

## Quick Reference

### File Locations

```
data/corpus/
├── sagan.txt           # Carl Sagan prose
├── hofstadter.txt      # Douglas Hofstadter prose
├── hitchens.txt        # Christopher Hitchens prose
└── ...                 # Other authors
```

### Generation Commands

```bash
# Generate convergent training data
python scripts/generate_convergent_training_llm.py \
  --corpus data/corpus/sagan.txt \
  --author "Carl Sagan" \
  --output /tmp/sagan_training.jsonl \
  --variations 3 \
  --max-paragraphs 100 \
  --workers 4

# Check output quality
head -3 /tmp/sagan_training.jsonl | python -m json.tool
```

### Corpus Size Guidelines

| Corpus Size | Training Examples | Expected Quality |
|-------------|-------------------|------------------|
| < 50 paragraphs | < 150 examples | Poor (insufficient data) |
| 50-200 paragraphs | 150-600 examples | Moderate |
| 200-500 paragraphs | 600-1500 examples | Good |
| 500+ paragraphs | 1500+ examples | Excellent |

More data with consistent style > less data with mixed sources.
