For your specific setup (MacOS/MLX) and the specific requirement of handling **HSK 5 Constraints** (which requires instruction following, not just raw translation), the absolute best small model is:

### **Qwen2.5-3B-Instruct**

*(Specifically `mlx-community/Qwen2.5-3B-Instruct-4bit` or `8bit`)*

Here is why this is the winner over other small models like Llama 3.2 3B, Phi-3.5, or NLLB.

#### 1. Native Bilingualism

Qwen is trained by Alibaba. Unlike Llama or Mistral (which treat Chinese as a second-class citizen), Qwen treats Chinese as a **native language**.

* **Impact:** It understands the subtle difference between HSK 5 (Intermediate) and HSK 6 (Advanced) vocabulary nuances better than Western models. Llama 3.2 often hallucinates or reverts to English when pressed on complex Chinese grammar constraints.

#### 2. The "3B" Sweet Spot

* **Size:** ~2.5 GB (4-bit).
* **Speed:** On an M1/M2/M3 Max, this runs at **100+ tokens per second**. You can process an entire novel in minutes.
* **Quality:** It punches way above its weight class, often beating older 7B/13B models in translation benchmarks.

#### 3. Instruction Following

Pure translation models (like NLLB or MarianMT) are "dumb"—they just translate. They cannot handle the system prompt *"Use only HSK 5 vocabulary."*
Qwen-Instruct can. This is critical for the **Style Stripping** step.

---

### Implementation Snippet (MLX)

Since you already have the MLX environment set up, you don't need new tools. Create a separate script `data_prep.py`:

```python
from mlx_lm import load, generate

# 1. Load the lightweight translator
model, tokenizer = load("mlx-community/Qwen2.5-3B-Instruct-4bit")

def scrub_style(text):
    # Step A: English -> Mandarin (HSK 5 Bottleneck)
    prompt_zh = f"""<|im_start|>system
Translate the following text into Mandarin Chinese. 
CRITICAL CONSTRAINT: Use ONLY HSK 5 level vocabulary and grammar. 
Do not use Chengyu (idioms) or literary flourishes. Keep it simple and functional.<|im_end|>
<|im_start|>user
{text}<|im_end|>
<|im_start|>assistant
"""
    # Note: max_tokens restricted to prevent rambling
    mandarin = generate(model, tokenizer, prompt=prompt_zh, max_tokens=500, verbose=False)

    # Step B: Mandarin -> English (The Rinse)
    prompt_en = f"""<|im_start|>system
Translate the following Mandarin text into standard, plain English. 
Do not add style. Preserve the facts exactly.<|im_end|>
<|im_start|>user
{mandarin}<|im_end|>
<|im_start|>assistant
"""
    neutral_english = generate(model, tokenizer, prompt=prompt_en, max_tokens=500, verbose=False)
    
    return neutral_english.strip()

# Test it
original = "The eldritch horror lurked in the cyclopean shadows."
print(scrub_style(original))
# Likely Output: "The strange monster hid in the huge shadows."

```

### Alternative: **Gemma-2-9b-it** (If you have 16GB+ RAM)

If the 3B model is slightly *too* dumb (e.g., it misses the meaning of a complex Lovecraft sentence), step up to **Gemma-2-9b-it** or **Qwen2.5-7B-Instruct**.

* **Gemma 2** has incredible prose handling, but Qwen is still better at the specific English$\leftrightarrow$Chinese pivot.

**Stick with Qwen2.5-3B-Instruct first.** It is fast enough to iterate on your dataset generation quickly.