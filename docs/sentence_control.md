The model is overfitting to Lovecraft's "run-on" tendency. Because the LoRA has learned that Lovecraft uses 80-word sentences, it refuses to generate a period until it hits that quota, often resulting in unreadable "walls of text."

To fix this, you need to attack the problem at three levels: **Inference Parameters** (forcing the model to stop sooner), **Prompt Constraints** (explicit instructions), and **Post-Processing** (surgically breaking the text).

### 1. The "Period Booster" (Logit Bias)

The most direct way to kill a run-on sentence is to **mathematically force** the model to output a period (`.`) when it gets wavering.

You can use `logit_bias` (if your inference engine supports it) to artificially boost the probability of the period token.

* **Token ID for `.**`: Usually `13` or `29889` (check your tokenizer).
* **Bias Value:** `+2.0` (Small nudge).

**Effect:** When the model is deciding between "and" (continuing the sentence) vs "." (ending it), this bias tips the scale toward ending it.

### 2. The "Clause Splitter" (Python Post-Processing)

Since the model is outputting grammatically correct but distinct clauses joined by "and/but/for," you can use `spaCy` to identify these conjunctions and replace them with periods *after* generation.

This script detects "Run-On Connectors" and snaps them if the sentence is too long.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def shatter_run_ons(text, max_len=40):
    doc = nlp(text)
    sents = []
    
    for sent in doc.sents:
        # If sentence is acceptable length, keep it
        if len(sent) < max_len:
            sents.append(sent.text)
            continue
            
        # If too long, look for "Split Points" (conjunctions)
        # We look for coordinating conjunctions (CC) like "and", "but", "for"
        splits = []
        last_start = sent.start
        
        for token in sent:
            # Logic: If token is "and/but", follows a comma, and is > 15 words from last start
            if token.pos_ == "CCONJ" and (token.i - last_start) > 15:
                # Check for preceding comma
                if token.i > 0 and doc[token.i - 1].text == ",":
                    # This is a split point
                    chunk = doc[last_start : token.i - 1] # exclude comma
                    sents.append(chunk.text.strip() + ".")
                    last_start = token.i + 1 # skip conjunction
                    
        # Append the remainder
        if last_start < sent.end:
            chunk = doc[last_start : sent.end]
            # Capitalize first letter of new sentence
            text_chunk = chunk.text.strip()
            if text_chunk:
                sents.append(text_chunk[0].upper() + text_chunk[1:])
                
    return " ".join(sents)

# Usage
# clean_text = shatter_run_ons(lovecraft_output)

```

* **Input:** *"The stars were fading, and the cold wind blew, for the time was nigh..."*
* **Output:** *"The stars were fading. The cold wind blew. The time was nigh..."*

### 3. The "Constraint" Prompt Update

You need to explicitly forbid the specific syntactic structure Lovecraft uses to extend sentences (parenthetical asides and endless "which" clauses).

**Add this to your System Prompt:**

```text
[SYNTAX CONSTRAINTS]:
1. MAX SENTENCE LENGTH: 60 words.
2. BAN "NESTING": Do not use parenthetical statements (...) inside a sentence.
3. SPLIT CONJUNCTIONS: Do not join more than two clauses with "and" or "but". Use a period instead.

```

### 4. Generation Parameter Tweaks

Adjust these specific knobs to penalize the "droning" behavior.

* **`repetition_penalty`: 1.25**
* *Why:* Run-on sentences often repeat structure words like *of, which, that, and*. A higher penalty makes it "expensive" to keep chaining clauses, forcing the model to stop and reset (start a new sentence).


* **`min_p`: 0.1** (Increase from 0.05)
* *Why:* This cuts off the "long tail" of low-probability connectors. It forces the model to stick to more decisive punctuation.



### Summary Strategy

1. **Immediate Fix:** Run the **Clause Splitter Script** on your existing output. It will fix the text you already have.
2. **Prevention:** Add the `[SYNTAX CONSTRAINTS]` block to your prompt for future generations.
3. **Tuning:** Set `repetition_penalty=1.25` in your inference call.
