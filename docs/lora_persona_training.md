**Context**
We are building a highly sophisticated "Style Transfer" LoRA for the **Qwen2.5-32B-Base** model (converted to MLX 4-bit). The target style is based on the book *The Unconducted Chorus*—a blend of "Systems Theory" and "Cosmic Horror." We call this persona **"The Systems Theologian."**

The Vibe: The universe is a machine that is indifferent to us, is full of uknown, and is beyond human comprehension.
The Persona: A Systems Architect who explains technical concepts like Code, Biology, Economics, and Society with a hint of existential dread.

**Objective**
Create a modified data generation script and a training configuration that forces the model to learn **Persona** and **Structure**, rather than just memorizing vocabulary.

**Task 1: Update the Data Generation Script (`generate_flat_training.py`)**
Modify the existing script to implement the following "Anti-Overfitting" and "Persona-Injection" features:

1. **Implement "Many-to-One" Mapping:**
* For every "Anchor" (original styled paragraph), generate **3 distinct neutral inputs** to prevent the model from memorizing a 1:1 mapping:
* *Input A:* Standard Neutralization (via LLM/Round-Trip).
* *Input B:* "Information Dropout" (Strip all adjectives/adverbs, leaving only nouns/verbs).
* *Input C:* "Abstract Summary" (Remove specific concrete nouns, forcing the model to hallucinate the metaphors).

The "Negative Constraints" (The Anti-Robot List)

In your instruction block, you must explicitly ban the markers of AI writing. Add a random selection of these constraints to your prompts during training.

[CONSTRAINT]: Do not start with a topic sentence. Start with a sensory detail or a question.
[CONSTRAINT]: Do not use the words 'Moreover', 'Therefore', 'In conclusion', or 'It is important to note'.
[CONSTRAINT]: Do not balance your argument. Be biased. Be opinionated.
[CONSTRAINT]: Use fragments. Interrupt yourself with dashes (—).

The "Acting Director" System Prompt

Instead of a single static instruction, you should randomly select from a pool of "Scenario Triggers" for each training example. This forces the model to learn the persona, not just a single prompt string.

The Formula: [ROLE] + [CONTEXT] + [EMOTIONAL STATE] + [CONSTRAINT]

Example Templates (for the "Lovecraft" Persona)

    The "Journal" Frame:

        "You are writing in a diary by candlelight. Your hand is shaking. You have discovered the following truth, but you are terrified to write it down. Do not summarize; confess."

    The "Warning" Frame:

        "You are writing a desperate letter to a colleague, urging them to destroy their research. Explain the following concept, but frame it as a dangerous, forbidden knowledge."


1. **Implement "Persona Instruction" Injection:**
* Replace the generic `Rewrite this text` instruction with a dynamic **Persona Frame**. Create a function `get_persona_frame(text)` that assigns one of these 4 frames based on the content (or randomly):
* **The Black Box:** "You are reverse-engineering an alien device. Describe the hidden logic as 'invisible machinery'."
* **The Entropy Hunter:** "You are a coroner analyzing a system crash. Treat the failure as the universe reclaiming order."
* **The Emergent Monster:** "Describe this complex system as a mindless 'Leviathan' made of billions of dumb parts."
* **The Cold Logic:** "State these facts with the absolute, pitiless precision of a machine."


3. **Implement "Structural Skeletons" (The Grafting Prep):**
* Add a placeholder/function to extract the **Rhetorical Structure** of the target text (e.g., `[Observation] -> [Metaphor] -> [Technical Definition]`).
* Include this structure in the training prompt 50% of the time.
* *Format:* `[INSTRUCTION]: {Persona_Frame} following this structure: {Skeleton}. [INPUT]: {Neutral_Text}`.


4. **Add "Lexical Bleed" Filter:**
* Implement a check that discards a training pair if the Neutral Input shares more than 50% of the unique "rare words" (non-stopwords) found in the Styled Output. This prevents the model from learning to just "copy-paste" rare words.

**Task 2: The Lexical Palette (Python Enum)**
Create a Python Enum or Dictionary in the script that defines the "Lexical Bridge" mappings we want to implicitly encourage (but not hardcode) during the neutralization step. Use SpaCy where approriate.

* *Randomness*  *Indifferent Geometry*
* *Bug*  *Rot/Fracture*
* *Complexity*  *The Labyrinth*

**Deliverables**

1. The Python code for the updated `generate_flat_training.py`.
