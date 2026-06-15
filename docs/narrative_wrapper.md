To implement a general solution that defeats AI detection ("Mechanical Precision," "Formulaic Flow") while preserving author style, you need to shift the pipeline from **Style Translation** to **Persona Simulation**.

The core issue is that your current pipeline asks the model to "rewrite text," which triggers its training to be helpful, clear, and objective (aka "Robotic"). You need to trick the model into being **subjective, opinionated, and messy**.

Here are the instructions for Claude to implement the **"Persona-Based RAG Pipeline."**

---

### **Prompt for Claude**

**Context:**
We are building a local Style Transfer pipeline (using MLX/LoRA) that rewrites neutral text into specific author styles (e.g., Lovecraft). The current output is stylistically correct but gets flagged by GPTZero as "AI Generated" because it is too logically structured, impersonal, and formulaic. We need a generalizable fix that works for *any* author LoRA, not just Lovecraft.

**Task:**
Implement a **"Persona Wrapper"** system. Instead of asking the model to "rewrite the text," we will ask it to "roleplay a specific character archetype encountering this information." This forces the model to abandon "Essay Mode" (Topic Sentences, Transitions) and adopt "Narrative Mode" (Subjectivity, Sensory Detail).

**Implementation Plan:**

#### **1. The "Persona" Configuration System**

Create a dictionary or JSON config that maps each Author LoRA to a specific **Persona Archetype**.

* **Lovecraft:** "Frantic Scholar writing a journal entry by candlelight."
* **Hemingway:** "Weary War Correspondent cabling a report from the front."
* **Austen:** "Witty Socialite writing a scandalous letter to a confidant."
* **Generic/Default:** "Opinionated Diarist recording their raw thoughts."

#### **2. The "Anti-Robot" System Prompt**

Replace the current system prompt with this **Subjectivity Engine**. This is the key to breaking the "Formulaic Flow" flag.

```text
[SYSTEM ROLE]:
You are not an AI assistant. You are {persona_archetype}.
You are encountering the following information: "{topic_summary}".

[PRIME DIRECTIVE - DEFEAT THE ROBOT]:
1. **NO TOPIC SENTENCES**: Never start a paragraph with a summary. Start *in media res* with a sensory detail, an emotion, or a fragment.
2. **SUBJECTIVITY OVER CLARITY**: Do not explain the concepts clearly. Filter them through your {persona_emotion} (e.g., fear, cynicism, joy). If the data is complex, complain about it or fear it; do not simplify it.
3. **KILL TRANSITIONS**: BANNED words: "Moreover," "Therefore," "However," "In conclusion," "It is important to note." Use dashes (—) or jump abruptly to the next thought.
4. **BURSTINESS**: You must include at least one sentence under 5 words in every paragraph. You must include one sentence over 50 words that is messy and winding.

[INPUT DATA (THE TRUTH YOU MUST ADAPT)]:
{content}

```

#### **3. The "Lexical Injector" Update (RAG)**

Update the `LexicalInjector` to support **Metaphor Mapping**. The model acts robotic because it uses "Data Words" (e.g., "financial crisis"). It needs "Style Words" (e.g., "ruinous collapse").

* **Action:** When retrieving from the Author Corpus, extract **Adjective-Noun pairs** (e.g., "abyssal void," "cyclopean masonry").
* **Prompt Injection:** Add a `[VOCABULARY PALETTE]` section to the prompt containing 5-10 specific metaphors to swap.

#### **4. Post-Processing: The "De-Linter"**

Update the cleaning script to aggressively target "Essayisms" that slip through.

* **Transition Stripper:** Regex remove sentence starters like *"It serves to," "It is worth noting," "Fundamentally."*
* **Fragment Enforcer:** If a paragraph has NO sentences under 8 words, break the longest sentence at a conjunction (and/but) and replace it with a period to create a fragment artificially.

#### **5. Inference Parameters (The Chaos Settings)**

Hardcode these settings for all "Style Transfer" tasks to ensure the model takes risks:

* `temperature`: **1.2** (High variance).
* `min_p`: **0.05** (Filters out absolute gibberish).
* `repetition_penalty`: **1.1** (Prevents loops without killing style).

**Goal:**
The output must read like a **human experiencing a fact**, not an **AI explaining a fact**.

---

### **Example Output You Expect**

* **Input:** "The stock market crashed due to bad loans."
* **Old Output (AI Flagged):** "The financial system, which had been unstable, collapsed because of poor lending practices." (Mechanical).
* **New Output (Persona - Lovecraft):** "Ruin! It has come at last—the great and terrible collapse of the ledger! Those accursed loans, rotting from within like dead flesh, have finally brought the temple of coin crashing down upon our heads." (Human/Subjective).

The specific scientific paper you linked—*The Cognitive Neuroscience of Constructive Memory* by Schacter & Addis—is an excellent benchmark. It passes GPTZero not because it is "dramatic" or "emotional," but because it is **syntactically dense**.

AI models are trained to be **clarifiers**. They take complex ideas and straighten them out.
Humans (especially academics and Lovecraft) are **complicators**. They take simple ideas and wrap them in layers of context, citations, and caveats.

Here is why that dry paper passes, and how to replicate that "Academic Density" for your Lovecraft pipeline without making it hysterical.

### The "Academic Density" Formula (Why the PMC Paper Passes)

1. **The "Interruptive" Clause:** The authors constantly interrupt their own main point with context.
* *Paper:* "One of the least controversial**—but most important—**observations is that..."
* *AI Version:* "One important and uncontroversial observation is that..."
* *Lesson:* Force the model to use em-dashes and commas to insert thoughts *inside* other thoughts.


2. **The "Nominalization" of Verbs:** The paper turns actions into nouns.
* *Paper:* "...reflect the **operation** of specific components..."
* *AI Version:* "...show how specific components operate..."
* *Lesson:* Use heavy nouns ("The accumulation of...", "The manifestation of...") instead of active verbs. This creates that "heavy," "dry" feel that AI avoids because it's "bad writing" (but very human).


3. **The "Hedged" Authority:** The paper rarely says "X is true." It says "It has been proposed that..." or "Evidence suggests..."
* *Lesson:* Absolute statements look robotic. Qualified statements look human.



---

### The New Instructions for Claude: "The Obsessive Antiquarian"

To achieve this natural, human-passing style without the melodrama, we need to instruct Claude to treat the output not as a "Story" but as a **"Dense Monograph."**

Here are the specific instructions to generate the prompt.

**Target Persona:** "A Victorian Encyclopedist or Antiquarian writing a definitive, dense report."

#### 1. The Syntax Instructions (The "Anti-Clarity" Engine)

Tell Claude to implement these syntactic constraints. This mimics the "bad" habits of academic humans that detectors interpret as "Real."

* **Constraint A: The "Garden Path" Sentence.**
* *Instruction:* "Do not put the subject and verb close together. Separate them with long modifying clauses. (e.g., 'The report, which had been gathering dust for centuries in the archives of the unremembered, **revealed**...')."


* **Constraint B: Nominalization.**
* *Instruction:* "Prefer abstract nouns over active verbs. Instead of 'The market crashed,' say 'The market suffered a catastrophic dissolution of value.'"


* **Constraint C: The 'Which' Hunt.**
* *Instruction:* "Use relative clauses starting with 'which', 'whereof', or 'wherein' to extend sentences past the point of comfort. Avoid starting new sentences. Link them."



#### 2. The "Dry Dread" Tone

We want the *feeling* of Lovecraft without the *screaming*.

* **Instruction:** "Adopt a tone of **Clinical Detachment**. You are describing horror, but you are describing it with the dryness of a coroner writing an autopsy report. Do not use exclamation points. Do not say 'It was scary.' Describe the *structure* of the fear."

#### 3. The "Human Imperfection" Features

* **Instruction:** "Use **Parenthetical Asides**. (e.g., '...a fact which I initially doubted, though later events proved me wrong...')."
* **Instruction:** "Use **Reference/Citation Style** interruptions. Mention specific dates, names, or 'figures' in the middle of the flow to break the rhythm."

---

### The Implementation Plan for Claude

Here is the exact block to give Claude to build your **"General Fix"**:

```text
[TASK]: Create a System Prompt for a Style Transfer Model.

[GOAL]: Rewrite input text into the style of H.P. Lovecraft, but aiming for his "Academic/Scientific" voice (e.g., 'At the Mountains of Madness'), not his "Pulp/Hysterical" voice.

[CRITICAL CONSTRAINT]: The output must pass AI detection by mimicking "Human Academic Density."

[SYSTEM PROMPT RULES]:
1. **DENSITY OVER CLARITY**: The AI default is to simplify. You must complicate. Take simple actions and turn them into complex processes.
   - *Bad (AI):* "The bank failed."
   - *Good (Human/Lovecraft):* "The institution, having long teetered upon a foundation of doubtful solvency, underwent a sudden and ruinous dissolution."

2. **SYNTACTIC INTERRUPTION**: Imitate human academic writing by interrupting the main clause.
   - Use paired em-dashes (—) to insert context mid-sentence.
   - Use "which" clauses to drag the sentence out.
   - *Example:* "The report—though initially dismissed by those of lesser intellect as mere conjecture—contains truths that..."

3. **PASSIVE/NOMINALIZED VOICE**: Use the passive voice to sound authoritative and detached.
   - *Bad:* "We found the error."
   - *Good:* "The error was discovered to lie within the fundamental architecture..."

4. **VOCABULARY MAPPING**:
   - Swap "Modern Corporate" terms for "Archaic Academic" terms.
   - *Report* -> *Treatise / Monograph*
   - *Trend* -> *manifestation / phenomenon*
   - *Risk* -> *peril / hazard*

5. **NO MELODRAMA**: Do not use words like "spooky," "scary," or "creepy." The horror comes from the *implication* of the facts, not the adjectives.

[INPUT DATA]:
{content}

```

### Why this works

This prompt tells the model to emulate **Bad Academic Writing** (dense, passive, complex). Paradoxically, "Bad" academic writing is the hardest thing for an AI to fake, because AI is trained to be "Good" (clear, active, simple). By forcing it to be dense and passive, you bypass the "Robotic Clarity" flag.