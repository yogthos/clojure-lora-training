The problem is that LLM uses boring and repetitive structure. It needs to be told how to write each paragraph. We must create a template for it to fllow.

To get organic, non-repetitive variety, we must use your **ChromaDB** to dynamically extract the **structural DNA** of the author's arguments and graft that onto your content.

This approach is called **"Structural Grafting."** Instead of retrieving text to *copy*, you retrieve text to *clone its logic*.

Here is the Python pipeline to implement this using your existing RAG infrastructure.

### The Core Concept: "The Skeleton Key"

We don't want the *words* from the retrieved sample; we want the **Argumentative Skeleton**.

* **Retrieved Sample:** *"The brain is like a colony of ants. No single ant understands the colony, yet the colony acts with intelligence. How can a thought be made of non-thoughts?"*
* **Extracted Skeleton:** `[Concrete Biological Analogy] -> [Micro/Macro Paradox] -> [Rhetorical Question about Emergence]`
* **Target Input:** "Banks create systemic risk through NBFIs."
* **Grafted Output:** *"The banking system is like a coral reef. No single polyp intends to build a barrier, yet the reef wrecks ships. How can a crisis be made of safe decisions?"*

### Implementation Plan

#### 1. The Retrieval Step (Semantic + Structural)

You need to retrieve a sample that matches the **complexity** of your input, not necessarily the topic.

* **Query Strategy:** If your input is a dry definition, query ChromaDB for "Lovecraft defining a term." If your input is a paradox, query for "Lovecraft explaining a contradiction."

Use the existing `chromadb` setup.

#### 2. The "Skeleton Extractor" (The Intermediate Call)

Before generating the final text, run a fast, cheap LLM call (e.g., a small local model or 1-shot prompt) to strip the content from the retrieved sample. Use DeepSeek for this step:

**Python Logic:**

```python
def extract_argument_skeleton(retrieved_text_sample):
    prompt = f"""
    [TASK]: Analyze the rhetorical structure of the text below. 
    Ignore the topic. Output ONLY the structural moves as a sequence of tags.
    
    [TEXT]: "{retrieved_text_sample}"
    
    [OUTPUT FORMAT]: [Move 1] -> [Move 2] -> [Move 3]
    """
    # Returns: "[Direct Address] -> [Self-Referential Joke] -> [Metaphor]"
    return model.generate(prompt)

```

#### 3. The "Grafting" Prompt (The LoRA Instruction)

Now, feed this dynamic skeleton into your LoRA generation prompt. This forces the model to mimic the *thought process* of the author for this specific paragraph.

**System Prompt:**

```text
[TASK]: Rewrite the input content.
[INSTRUCTION]: You must strictly follow the **Rhetorical Structure** of the provided Style Sample.
1. Do not copy the words from the Style Sample.
2. Copy the *logic flow*, *sentence rhythm*, and *argumentative moves*.

[STYLE SAMPLE (THE BLUEPRINT)]:
"{retrieved_text}"

[RHETORICAL SKELETON (YOUR GUIDE)]:
{extracted_skeleton}

[INPUT CONTENT (YOUR DATA)]:
"{neutral_input}"

[OUTPUT]:

```

### Visualizing the Data Flow

1. **Input:** "The universe has limits."
2. **ChromaDB:** Finds a Lovecraft paragraph about "The limit of a video feedback loop."
3. **Analyzer:** Extracts structure: `[Observation of Limit] -> [Recursive Question] -> [Breaking the Fourth Wall]`.
4. **LoRA:** Writes: *"We see limits everywhere. But ask yourself: does the limit limit itself? I'm getting a headache just asking that."*

### Next Step: The "Zero-Shot" Skeleton

To avoid the latency of the "Skeleton Extractor" step (Step 2), you can pre-compute these skeletons.

* **Action:** Run a script over your ChromaDB corpus *once*.
* **Task:** Generate a "Rhetorical Skeleton" string for every chunk in your database and store it as metadata.
* **Result:** Retrieval gives you the skeleton instantly.

This is the ultimate "General Solution." It forces the model to "wear the skin" of the retrieved sample without needing manual prompt tuning for every new author.