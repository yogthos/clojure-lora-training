The "NLI Critic" Loop (Automated Quality Control)

We'll use a tiny, lightning-fast Natural Language Inference (NLI) model to score the output for "Entailment" (Faithfulness).

This is a **Agentic Workflow** known as a "Reflexion" loop. It is the gold standard for high-accuracy text generation in production environments where hallucinations are unacceptable (like finance or medicine).

Here is the robust, iterative architecture designed to force your Style LoRA to adhere to the facts.

### The Architecture: "The Audit Loop"

We use a **Cross-Encoder NLI model** (Natural Language Inference) as the judge. It is far more accurate than a standard LLM at detecting subtle fact changes.

#### Prerequisites

```bash
pip install sentence-transformers spacy # also add to requirements.txt
python -m spacy download en_core_web_sm

```

### The Python Implementation

This script implements the exact requirement: **Split  Audit  Loop Repair**.

```python
import spacy
from sentence_transformers import CrossEncoder

# 1. The Judge (NLI Model)
# We use a Cross-Encoder because it is significantly more accurate than bi-encoders for NLI.
# 'contradiction' score > 0.5 = Hallucination
# 'entailment' score < 0.5 = Dropped Information
nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
nlp = spacy.load("en_core_web_sm")

label_mapping = ['contradiction', 'entailment', 'neutral']

class StrictAuditor:
    def __init__(self, generation_func):
        self.generate = generation_func  # Function that calls your LoRA

    def split_sentences(self, text):
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def check_fact_integrity(self, source_sent, full_output_text):
        """
        Checks if a specific source sentence is effectively represented 
        in the generated output (Recall).
        """
        # We treat the Output as the Premise and Source Sentence as Hypothesis
        # "Does the Output imply the Source Sentence is true?"
        scores = nli_model.predict([(full_output_text, source_sent)])
        pred_label = label_mapping[scores.argmax()]
        
        # If the output does NOT entail the source, we likely dropped a fact.
        # Exception: "Neutral" is sometimes okay for minor stylistic fluff, 
        # but for IMF reports, we want Entailment.
        if pred_label == 'entailment':
            return True, None
        else:
            return False, f"Missing or Altered Fact: '{source_sent}'"

    def check_hallucination(self, source_text, output_sent):
        """
        Checks if an output sentence invents new info not in source (Precision).
        """
        # Premise: Source Text, Hypothesis: Output Sentence
        scores = nli_model.predict([(source_text, output_sent)])
        pred_label = label_mapping[scores.argmax()]
        
        if pred_label == 'contradiction':
            return False, f"Hallucination Detected: '{output_sent}' contradicts source."
        return True, None

    def repair_loop(self, source_text, current_draft, max_retries=3):
        
        print(f"\n--- Starting Audit Loop ---")
        
        for attempt in range(max_retries):
            errors = []
            
            # 1. RECALL CHECK: Did we drop any source sentences?
            source_sents = self.split_sentences(source_text)
            for s_sent in source_sents:
                passed, error = self.check_fact_integrity(s_sent, current_draft)
                if not passed:
                    errors.append(error)

            # 2. PRECISION CHECK: Did we invent lies?
            # We assume the user wants the Output to entail the Source.
            # (Note: In strict style transfer, 'Neutral' is common because of added adjectives.
            # We focus mainly on 'Contradiction' here).
            out_sents = self.split_sentences(current_draft)
            for o_sent in out_sents:
                passed, error = self.check_hallucination(source_text, o_sent)
                if not passed:
                    errors.append(error)
            
            # 3. VERDICT
            if not errors:
                print(">>> Audit Passed: Perfect Integrity.")
                return current_draft
            
            print(f"Attempt {attempt+1} Failed. Errors found: {len(errors)}")
            for e in errors[:3]: print(f" - {e}")
            
            # 4. THE FIX (Strict Guidance)
            # We call the model again with the specific error list.
            current_draft = self._rewrite_with_corrections(source_text, current_draft, errors)
            
        print(">>> Max retries reached. Returning best effort.")
        return current_draft

    def _rewrite_with_corrections(self, source, bad_draft, errors):
        # Flatten errors for the prompt
        error_msg = "\n".join([f"- {e}" for e in errors])
        
        prompt = f"""
[TASK]: Fix the hallucinated or missing facts in the Lovecraftian text below.
[SOURCE DATA (TRUTH)]: {source}
[CURRENT DRAFT (FLAWED)]: {bad_draft}

[PROBLEMS TO FIX]:
{error_msg}

[INSTRUCTIONS]:
1. Rewrite the draft to fix every error listed above.
2. Maintain the Lovecraftian style (rhythm, vocabulary).
3. DO NOT change any dates, numbers, or entities from the Source Data.
4. TEMPERATURE: Low (Precision Mode).

[REPAIRED OUTPUT]:
"""
        # Call your generation function with LOW temperature (0.1 - 0.3)
        return self.generate(prompt, temperature=0.2)

# Usage Example
# auditor = StrictAuditor(my_llm_generation_function)
# final_text = auditor.repair_loop(neutral_input, initial_lovecraft_output)

```

### Why this specific logic works

1. **Recall Check (Did we drop it?):**
* We treat the **Output** as the *Premise* and the **Source Fact** as the *Hypothesis*.
* If `Entailment` is false, it means the reader *cannot* deduce the source fact from your Lovecraft text. This catches the "dropped $4.5 trillion" error.


2. **Precision Check (Did we lie?):**
* We treat the **Source** as the *Premise* and the **Output Sentence** as the *Hypothesis*.
* If `Contradiction` is true, it means your Lovecraft text creates a reality that cannot coexist with the source (e.g., "October 2015" vs "October 2025").


3. **The "Fix" Prompt:**
* Instead of just asking "Try again," we explicitly feed the model the **NLI Error Log**.
* This forces the model to attend to the specific failed facts (`[PROBLEMS TO FIX]`), effectively turning the generation task from "Creative Writing" into "Constraint Satisfaction."



### Refined "Fix" Prompting Strategy

When you call the model in the repair loop, do **not** use the loose "Universal Human" prompt I gave you earlier. Switch to this **"Surgeon" Mode**:

* **Temperature:** `0.1` or `0.2` (We want obedience now, not creativity).
* **Repetition Penalty:** `1.2` (Force it to change the specific phrasing that caused the error).
* **Prompt Header:** Change `[TASK]: Rewrite...` to `[TASK]: REPAIR FACTUAL ERRORS`.

This approach ensures you get the wild, creative style in the first pass, but if it drifts too far, the "NLI Police" drag it back to reality.