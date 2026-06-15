Here is the consolidated, step-by-step master plan for training a High-Fidelity Style Transfer LoRA using `mlx-community/Qwen3-8B-Base`.

### Phase 1: Data Construction (The "Triad" Strategy)

**Goal:** Create a dataset of ~2 Million tokens (approx. 45k–60k examples) that forces the model to learn *transformation* rather than memorization.

**1. The "HSK 5" Scrubbing Pipeline**
Do not use an LLM to "rewrite neutrally." Use Round-Trip Translation to chemically dissolve style.

* **Tool:** `Qwen2.5-3B-Instruct` (Native bilingual capabilities).
* **Step A:** Translate Author Text  Mandarin (Constraint: **HSK 5 Vocabulary only**).
* **Step B:** Translate Mandarin  English (Constraint: **Simple, Staccato SVO sentences**).
* **Result:** A purely functional, monotone input that preserves logic but destroys rhythm.

**2. The Expansion Ratio (1:3)**
For every 1 paragraph of original text, generate 3 training entries to prevent content overfitting:

* **Entry A (Anchor):** Neutral Input  Original Author Text.
* **Entry B (Snowflake):** Neutral Input (Mundane Topic)  Synthetic Author Style. (e.g., describing a toaster in Lovecraft’s voice).
* **Entry C (Noise):** Input from Entry A with **10% typos/dropped words**  Original Author Text. (Simulates NEFTune for robustness).

**3. The "Lossless" Entity Check (Critical)**

* **Rule:** The Proper Nouns in the Input **must match** the Output.
* **Action:** If the Author text mentions "Cthulhu," the Neutral Input must say "Cthulhu," not "The Monster."
* **Why:** Prevents the model from learning to hallucinate random names during inference.

**4. Structural Tagging**
Prepend tags to the `prompt` field to teach the model *control* over sentence complexity.

* Format: `[STYLE: Long & Flowing | Complex Syntax]` or `[STYLE: Short | Punchy]`.

---

### Phase 2: LoRA Configuration (The "Base Model" Settings)

**Goal:** Overwrite the model's default "Helpful Assistant" probability distribution with a volatile, stylized distribution.

**1. Model Selection**

* **Model:** `mlx-community/Qwen3-8B-Base-bf16`.
* **Reason:** Instruct models have RLHF safety filters that resist stylistic overwriting. Base models are pure pattern completers.

**2. JSONL Format**

* **Keys:** `{"text": "..."}` (Concatenated format).
* **Structure:** `[STYLE TAG]\n[NEUTRAL]: {input}\n\n[AUTHOR]: {output}`.
* **Note:** Do not use `messages` (Chat Template).

**3. Hyperparameters**

* **Batch Size:** **1** (Effective Batch Size ~2-4 with accumulation).
* *Reason:* High batch sizes "average out" style. Low batch sizes preserve the specific syntactic quirks of each example.


* **Rank:** **64**.
* **Alpha:** **128** (Scale 2.0).
* *Reason:* High alpha provides a "Megaphone" signal to override pre-trained weights.


* **Dropout:** **0.20**.
* *Reason:* High dropout forces the model to learn robust features (syntax/rhythm) rather than memorizing specific keywords.


* **Learning Rate:** **1e-5**.
* **Target Modules:** All linear layers (`q, k, v, o, gate, up, down`).

---

### Phase 3: Inference Strategy (The "Unshackled" Mode)

**Goal:** Force the model to take risks and generate "bursty" text.

**1. Inference Parameters**

* **Adapter Scale:** Hardcode to **2.0**. (Default is often 1.0, which dampens the style by 50%).
* **Temperature:** **1.1 – 1.3**. (Standard is 0.7; Style requires accessing rare, "tail-end" vocabulary).
* **Min-P:** **0.05**. (Better than Top-P for maintaining coherence at high temperatures).
* **Repetition Penalty:** **1.15**.

**2. The "One-Shot" Prompt**
Do not just ask for the style. Provide an "Anchor Example" in the prompt to calibrate the model's expectations.

```text
Rewrite the input in the style of {Author}.
[NEUTRAL]: The dog barked at the moon.
[AUTHOR]: A guttural cacophony erupted from the canine's throat, a blasphemous baying directed toward the gibbous and indifferent orb hanging in the firmament.

[NEUTRAL]: {Your Actual Input}
[AUTHOR]:

```

**3. Post-Processing**

* **Disable Verification:** Turn off any "Critic" or "Entailment" agents. They will flag stylistic adjectives as "hallucinations."
* **Negative Constraints:** Use Logit Bias to **ban** corporate words (`market, system, report, global, interconnected`). This forces the model to invent metaphors.