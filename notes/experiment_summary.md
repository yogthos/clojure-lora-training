# LoRA Style Transfer Experiment Summary

## Best Settings Found

| Setting | Value | Notes |
|---------|-------|-------|
| Temperature | 0.2 | Lower = more consistent, fewer hallucinations |
| LoRA Scale | 1.0 | Default works best |
| Prompt | "Rephrase in {author}'s prose style." | Simpler = better |
| skip_neutralization | true | Direct input preserves more facts |

## Key Findings

### What Works
1. **Simple prompts**: "Rephrase in X's style" works better than complex instructions
2. **Lower temperature (0.2)**: Reduces hallucinations significantly  
3. **Skip neutralization**: Passing original text directly preserves facts better
4. **LoRA scale 1.0**: Higher scales (1.2+) cause more hallucinations

### What Doesn't Work
1. **Complex prompts**: "Keep all facts", "Add nothing new" paradoxically cause MORE hallucinations
2. **High temperature (0.5+)**: Causes random topic drift
3. **Neutralization step**: DeepSeek summarization loses key facts
4. **Excessive instructions**: Model ignores or misinterprets them

### Fact Preservation Results

| Config | Facts Preserved | Style Change | Hallucinations |
|--------|-----------------|--------------|----------------|
| minimal + 0.2 + 1.0 | 100% | 46% | No |
| preserve + 0.2 + 1.2 | 100% | 86% | Yes (but correct) |
| strict + 0.2 + 1.0 | 100% | 22% | No |
| transform + 0.5 + 1.2 | 100% | 80% | No |

## Recommendations for Next LoRA Training

### Training Data Format
1. **Include explicit names in training examples**: 
   - Bad: "The philosopher developed the theory"
   - Good: "Karl Marx developed Dialectical Materialism"

2. **Train on fact-preserving transformations**:
   - Input: "Marx said X. Lenin added Y."
   - Output: "In the teachings of Marx we find X. Lenin, building upon this, proposed Y."

3. **Avoid training on content that adds explanatory notes**:
   - Don't include: "(better known as...)", "(1818-1893)", etc.
   - These patterns leak into inference

### Training Prompt Format
Use simple, consistent prompts:
```
Rephrase in [Author]'s prose style:

[Input text]

[Output text]
```

### Key Metrics to Track During Training
1. Named entity preservation rate
2. Factual accuracy (no added dates/places)
3. Style distinctiveness (word overlap ratio)
4. No parenthetical additions

## Current System Settings (Optimal)

```json
{
  "mlx": {
    "temperature": 0.2,
    "max_tokens": 256
  },
  "generation": {
    "lora_scale": 1.0,
    "skip_neutralization": true
  }
}
```

Prompt: `Rephrase in {author}'s prose style.`
