For minimal grammar correction that respects a specific authorial style (like Lovecraft) without sanitizing it into "corporate English," you should avoid standard generative models (like ChatGPT or standard T5) which tend to rewrite entire sentences.

Instead, use **Tagging** or **Rule-Based** libraries. These identify specific errors and propose isolated patches rather than regenerating the text from scratch.

Here are the best Python libraries for this specific "minimalist" goal:

### 1. `language_tool_python` (Best for Control)

This is a wrapper for LanguageTool. It is ideal for your use case because it categorize errors. You can explicitly tell it to fix **Grammar** and **Typos** but ignore **Style**, **Complexity**, or **Non-Standard** categories (which would flag Lovecraftian prose as "bad").

* **Why it fits:** It returns *matches* with offsets, not a rewritten string. You choose which matches to apply.
* **Installation:** `pip install language-tool-python`

**How to use it for Minimal Edits (Style-Safe Mode):**

```python
import language_tool_python

def style_safe_correction(text):
    # 'en-US' or 'en-GB' depending on your LoRA's base
    tool = language_tool_python.LanguageTool('en-US')
    
    matches = tool.check(text)
    
    # Must build based ChromaDB 
    # Filter out "Style" rules that hate author style
    # e.g., Passive Voice, Sentence Length, Archaic phrasing
    filtered_matches = [
        m for m in matches 
        if m.category not in ['STYLE', 'CASING', 'MISC'] 
        and m.ruleId not in ['PASSIVE_VOICE', 'TOO_LONG_SENTENCE']
    ]
    
    # Apply only the allowed fixes
    return language_tool_python.utils.correct(text, filtered_matches)

# Input: "The eldritch horror were waiting in the shadows." (Grammar error)
# Output: "The eldritch horror was waiting in the shadows." (Fixed)
# Input: "It was a dark and lugubrious night..." (Style "error")
# Output: "It was a dark and lugubrious night..." (Preserved)

```

### 2. `GECToR` (Best ML-Based Minimalist)

GECToR (Grammatical Error Correction: Tag, Not Rewrite) is a Google/Grammarly research project. Unlike T5 or GPT, it does not generate text. It looks at the existing words and applies tiny tags like `$KEEP`, `$DELETE`, or `$APPEND`.

* **Why it fits:** Because it is structurally incapable of "hallucinating" a new sentence. It can only edit what is there, preserving the rhythm significantly better than seq2seq models.
* **Trade-off:** Harder to install (requires AllenNLP/PyTorch).
* **Library:** [GitHub - grammarly/gector](https://github.com/grammarly/gector)

### 3. `HappyTransformer` (Easiest ML Option)

If you need the power of a neural network (to catch deep syntax errors that rules miss) but want to minimize rewriting, use `HappyTransformer` with a T5 model, but force **Greedy Search** to prevent it from getting creative.

* **Installation:** `pip install happytransformer`

```python
from happytransformer import HappyTextToText, TTSettings

def minimal_neural_fix(text):
    # Load a specialized GEC model (not a generic summarizer)
    happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
    
    # Settings to force conservatism
    # num_beams=1 (Greedy) ensures it picks the most obvious fix only
    # min_length=1 prevents truncation
    args = TTSettings(num_beams=1, min_length=1)
    
    # Prefix 'grammar: ' is specific to this model
    result = happy_tt.generate_text(f"grammar: {text}", args=args)
    return result.text

```

### Recommendation

Start with **`language_tool_python`**. It is the only option that lets you explicitly "whitelist" author style while fixing the objective typos that occur during LoRA inference.
