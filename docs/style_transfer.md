This LoRA training plan synthesizes findings from the five uploaded papers, specifically adapting the **"Instruction Back-Translation"** method (from *Readers Prefer Outputs...*) and the **"Inverse Paraphrasing"** strategy (from *Reformulating Unsupervised Style Transfer...*) to the `mlx-community/Qwen3-8B-Base-bf16` model.

The core principle derived from these papers is that to prevent content memorization while capturing style, you must train the model on the **transformation function** (Neutral Content  Stylized Content) rather than raw text modeling.

### **Phase 1: Data Construction (The "Inverse Paraphrase" Pipeline)**

You cannot train on raw books; doing so causes the model to memorize the plot (content). You must create a parallel dataset where the *input* is style-neutral and the *output* is the author's original text.

**1. Segmentation (Paragraph Coherence)**

* **Source:** The target author's corpus (ePub/PDF converted to text).
* 
**Chunking:** As per the *Long Text Style Transfer* paper, style resides in "inter-sentential relationships." Do not split by sentence.


* 
**Action:** Segment text into **paragraphs or chunks of 250–600 words**. Ensure chunks end on sentence boundaries.



**2. Synthetic "Neutral" Input Generation**

* **Method:** Use a strong LLM (like GPT-4o or a large Qwen model) to strip the style from the chunks, creating the "Input" for your training pairs.
* 
**The Prompt:** Based on the *Readers Prefer Outputs* paper, use a prompt that explicitly demands "neutral" or "flat" descriptive text to force the LoRA to learn the "delta" (the difference between bland and stylized).


* *Prompt:* "Rewrite the following text to be completely neutral and flat. Remove all specific vocabulary, unique sentence structures, and authorial voice. Preserve all facts, characters, and events exactly, but write it like a dry technical summary or a Wikipedia entry."
* *Result:* You now have pairs of `(Neutral Summary, Original Author Text)`.



**3. Data Formatting (JSONL)**

* Format the data for instruction tuning.
* **System Prompt:** "You are a creative writing assistant. Rewrite the user's input in the style of [Author Name], ensuring faithful adherence to their sentence structure, vocabulary, and rhythm."
* **User:** [The Neutral Summary]
* **Assistant:** [The Original Author Text]

---

### **Phase 2: LoRA Configuration (MLX-Specific)**

Style transfer requires modifying how the model processes relationships between words (syntax) and selects words (lexicon).

**1. Target Modules**

* **Finding:** Style is deep and syntactic. Restricting LoRA to just `q_proj` or `v_proj` (Attention) often results in correct vocabulary but wrong sentence structure.
* **Setting:** Target **all linear layers**.
* `keys`: `['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'down_proj', 'up_proj']`



**2. Rank and Alpha**

* **Finding:** Syntactic reordering is a complex task requiring higher capacity than simple fact injection.
* **Rank (r):** **64** or **128**. (Lower ranks like 8 or 16 are insufficient for capturing complex authorial voice).
* **Alpha:** **128** or **256** (Set Alpha to  to ensure the style signal is strong enough to override the base model's neutral RLHF alignment).

**3. Dropout**

* **Setting:** `0.05` or `0.1`. (Helps prevent overfitting to the specific *content* of the training chunks, forcing the model to generalize the *style* patterns).

---

### **Phase 3: Training Hyperparameters**

**1. Batch Size**

* **Recommendation:** **1**.
* 
**Reasoning:** The *Readers Prefer Outputs* paper specifically used a batch size of 1. This forces the model to update weights based on the specific stylistic nuances of *each* individual example, rather than averaging gradients across a batch (which can "smooth out" the unique style).



**2. Learning Rate**

* **Recommendation:** `1e-5` to `2e-5`.
* **Note:** If using the high rank (128), lean towards the lower end (`1e-5`) to maintain stability.

**3. Epochs**

* **Recommendation:** **1 to 3 Epochs**.
* **Constraint:** Do *not* overtrain. As noted in the *Capturing Classic Authorial Style* paper, style is learned quickly; overtraining leads to the model ignoring the input content and just regurgitating the training data (hallucination). Start with 1 epoch and evaluate.



---

### **Phase 4: Advanced Refinement (GRPO)**

If the standard LoRA (SFT) does not yield high enough fidelity, you can apply the **Group Relative Policy Optimization (GRPO)** method detailed in the *Capturing Classic Authorial Style* paper. This is a "Post-training" step.

**1. The Reward Model**

* You need a way to score "Style Similarity" automatically.
* 
**Method:** Use a sentence transformer (like `all-mpnet-base-v2`) fine-tuned on the author's work to act as a discriminator.


* **Calculation:** Calculate the cosine similarity between the model's output and a "Gold Standard" chunk of the author's writing.

**2. Training Logic**

* Generate multiple outputs for the same input.
* Reward the outputs that have the highest embedding similarity to the author's style *and* maintain the content facts.
* Update the LoRA adapter to favor these high-reward paths.

### **Summary of the "No Memorization" Algorithm**

To satisfy your requirement that the model "MUST NOT learn the content," the training objective is fundamentally shifted by the data structure:

1. **Input:** "A man walked down the street." (Neutral)
2. **Target:** "He stalked the cobbles, a shadow among shadows..." (Stylized)
3. **Loss Calculation:** The model is penalized if it fails to predict "stalked" given "walked". It learns that `walked` + `[Style Token]`  `stalked`. It does *not* learn that "A man was on the street" because that information was provided in the prompt.

**Command Line Implementation Plan (MLX Example):**

```bash
# Data generation (Python psuedocode)
# source_text = "Original author paragraph..."
# neutral_input = call_llm("Rewrite neutrally: " + source_text)
# save_to_jsonl(neutral_input, source_text)

# Training command (Conceptual)
python -m mlx.lora.train \
    --model mlx-community/Qwen3-8B-Base-bf16 \
    --train-file ./author_style_pairs.jsonl \
    --batch-size 1 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --target-modules q_proj v_proj k_proj o_proj gate_proj down_proj up_proj \
    --learning-rate 1e-5 \
    --iters 600  # Adjust based on dataset size (approx 1 epoch)

```