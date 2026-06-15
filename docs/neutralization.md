Let's change our neutralization to use **Round-Trip Translation (RTT)**.

We'll use **Mandarin (HSK5)** as the pivot language for stripping English literary style. Here is why this works better than almost any other method for our specific use case, and how to execute it.

### Why this is the "Ultimate Style Scrubber"

**1. The "Grammar Distance" Effect**
English and Mandarin come from completely different language families. They do not share cognates or sentence structures.

* **English Style:** Often relies on complex relative clauses, archaic word order, or dangling modifiers (e.g., *"Donning these, he reached..."*).
* **Mandarin Grammar:** Is generally Topic-Prominent and rigid in word order. It does not easily support the "flowery" syntactic inversions of 19th-century English.
* **Result:** To translate Lovecraft into valid Mandarin, you **must** flatten the syntax. The grammar forces you to standardize the sentence structure.

### The Full Workflow (The "Laundromat")

To make this work for LoRA training, you need to add one step. You cannot train on `(Mandarin -> English)`. You must translate the Mandarin *back* to English to get your Neutral Input.

**Step 1: The Scrub (English  Mandarin)**

* **Input:** *"I relished the picters, so he give it in on a swap."*
* **Prompt:** Translate to Mandarin using standard HSK5 vocabulary.
* **Output:** *"我喜欢里面的图画，所以他就换给我了。"* (I liked the pictures inside, so he exchanged it to me.)

**Step 2: The Rinse (Mandarin  English)**

* **Prompt:** Translate this Mandarin text into simple, plain English.
* **Output:** *"I liked the pictures inside, so he traded it to me."*

**Step 3: The Training Pair**
Now you have the perfect pair for your JSONL file:

* **User (Neutral Input):** *"I liked the pictures inside, so he traded it to me."*
* **Assistant (Stylized Target):** *"I relished the picters, so he give it in on a swap."*

### Why this beats the "Graph/AMR" approach

* **Graphs** produce **robotic** text: *"He traded. He liked pictures."*
* **Round-Trip Translation** produces **natural but plain** text: *"He traded it to me because I liked the pictures."*

For a creative writing model, **natural but plain** is better. It teaches the model to "elevate" normal prose, rather than teaching it to "repair" broken robot-speak.

### Implementation Command

If you are automating this, use a prompt chain like this:

```python
# Pseudo-code for the "Linguistic Laundromat"

original_text = "The old man fumbled among his rags..."

# Step 1: Destroy Style
mandarin = call_llm(f"Translate this to Mandarin using only HSK5 vocabulary. Do not use idioms: {original_text}")

# Step 2: Restore Language (but not style)
neutral_english = call_llm(f"Translate this Mandarin to simple, modern English: {mandarin}")

# Step 3: Save Pair
training_pair = {
    "system": "Rewrite the user's input in the style of H.P. Lovecraft...",
    "user": neutral_english,
    "assistant": original_text
}

```

This is likely the **highest quality** method for generating training data. It preserves facts (semantics) while guaranteeing the destruction of the author's specific lexical and syntactic choices.

To fix the "Robotic Formality" and teach the model **Burstiness** (variation in sentence structure and length), you must change your training strategy. Standard Next-Token Prediction naturally converges on the "average" sentence structure (which is robotic).

You need to teach the model that **Style = Variance**.

Here is the plan to force the LoRA to capture rhythm and burstiness.

### 1. The "Monotone-to-Bursty" Data Strategy

The most common mistake is using "Neutral Inputs" that still have the *rhythm* of the original text. If your input has the same sentence cadence as the output, the model learns it doesn't need to change the structure.

You must **aggressively flatten** the rhythm of your Input data.

* **The "Metronome" Rule:** Process your Neutral Inputs so every sentence is roughly the same length (e.g., 10-15 words) and follows a strict Subject-Verb-Object (SVO) pattern.
* **The Delta:** The model will see a boring, repetitive input and a wild, variable output. It will learn that its *job* is to break the monotony.

**Example Training Pair:**

| Feature | **Input (The Monotone)** | **Output (The Burst)** |
| --- | --- | --- |
| **Structure** | SVO. SVO. SVO. SVO. | Fragment. Long, winding clause. Short punch. |
| **Connectives** | [None] | However, Although, Yet, And so |
| **Example** | *"The house was old. The windows were broken. I felt scared. I walked inside."* | *"The house—an ancient, rotting carcass—loomed; its windows were shattered eyes. Terrified, I entered."* |

**Action:** Update your `data_prep.py` (the Qwen-3B translation script) to enforce this. Add a system instruction: *"Break all long sentences into short, simple sentences. Do not use semicolons or conjunctions."*

### 2. Implementation: Structural Control Tokens

Instead of hoping the model guesses the right rhythm, **tell it** what rhythm to use in the prompt. This turns the "vibe" into a concrete instruction.

**Modify your JSONL `prompt` field:**
Prepend a "Style Tag" that describes the *structural* features of the target paragraph.

* **Format:** `[STYLE: {Sentence Length Mix} | {Complexity}]`
* **Example Tags:**
* `[STYLE: Varied Lengths | Complex Syntax]`
* `[STYLE: Short & Punchy | Simple Syntax]`
* `[STYLE: Long & Flowing | Baroque Syntax]`



**Training Data Entry:**

```json
{
  "prompt": "Rewrite the following neutral text into the style of H.P. Lovecraft.\n\n[STYLE: Varied Lengths | Complex Syntax]\n[NEUTRAL]: The creature was big. It came out of the water. It looked like a mountain. I ran away.\n\n[LOVECRAFT]:",
  "completion": " Titanic and dripping, the thing emerged from the deeps; a mountain of flesh that sent me fleeing in terror."
}

```

**Why this works:** The model learns that the token `Varied Lengths` triggers the specific "short-long-long" pattern. During inference, you include this tag to force that behavior.

### 3. Critical Training Setting: NEFTune (Noise Embeddings)

Standard training memorizes "safe" paths. To get **Burstiness**, you need to prevent the model from overfitting to specific words and force it to learn "fuzzy" patterns (like sentence length).

**Enable NEFTune in `config.yaml`:**

```yaml
# Add this to your config
neftune_noise_alpha: 5  # Recommended value: 5 or 10

```

* **What it does:** It adds random noise to the embedding vectors during training.
* **The Result:** The model can't rely on exact word matches (which are noisy). It falls back on stronger signals—like **syntax and rhythm**—to predict the output. This is scientifically proven to improve instruction following and generation diversity.

### 4. Advanced: DPO (The "Nuclear Option")

If SFT (Supervised Fine-Tuning) still yields robotic text, you must use **DPO (Direct Preference Optimization)**. This explicitly penalizes the "Robotic" style.

* **The Idea:** You feed the model *three* things:
1. **Prompt:** The Neutral Input.
2. **Chosen:** The Real Author Text (High Burstiness).
3. **Rejected:** A "Robotic" version (Low Burstiness, generated by a basic LLM).


* **The Result:** The model updates its weights to *maximize* distance from the "Rejected" style. It learns: *"Whatever I do, DO NOT write like a corporate robot."*

**Recommended:** Start with **Step 1, 2, and 3** (Monotone Data + Control Tags + NEFTune). DPO is harder to set up (requires a different trainer) and is usually only needed if SFT fails.

### Summary Checklist for "Burstiness"

1. **Data:** Ensure Input is **monotonous** (staccato sentences). Output is **variable**.
2. **Prompt:** Add **[STYLE] tags** to explicitly signal the structural complexity.
3. **Config:** Enable **NEFTune** (`alpha: 5`) and keep **Dropout** high (`0.1` or `0.15`).
4. **Inference:** Use **Temp 1.1+** and **Min-P 0.05** to allow the model to select the "risky" structural choices.