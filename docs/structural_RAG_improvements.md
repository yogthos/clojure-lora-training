# Structural RAG Improvements Plan

## Problem Statement

The current Structural RAG provides **abstract guidance** (e.g., "LONG → SHORT → MEDIUM") that doesn't give the model concrete patterns to follow. This results in:

1. **Mechanical Precision**: Uniform sentence structures, lacks author's specific syntactic choices
2. **Impersonal Tone**: Output lacks emotional engagement, feels detached and neutral
3. **Sophisticated Clarity**: Output may lack coherent argument flow and natural transitions

The key insight from our training research: **"Style = Variance"**. The model needs concrete, varied patterns—not abstract categories.

---

## Root Cause Analysis

### Why Current Approach Falls Short

| Issue | Current Approach | Problem |
|-------|------------------|---------|
| **Mechanical Precision** | Length categories (SHORT, MEDIUM, LONG) | Too abstract; doesn't specify HOW to make sentences long/short |
| **Impersonal Tone** | Punctuation detection (dashes, semicolons) | Missing emotional vocabulary and stance markers |
| **Sophisticated Clarity** | No transition guidance | Missing cohesion devices and argument structure |

### What's Missing

1. **Concrete syntactic templates**: Actual POS patterns, not just length categories
2. **Vocabulary clusters**: Author-specific word banks by function
3. **Transition inventory**: Specific connectives the author uses
4. **Emotional markers**: Stance and evaluative vocabulary
5. **Rhetorical patterns**: Questions, exclamations, direct address

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Enhanced Structural RAG Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐    │
│  │ Author      │    │ Multi-Channel    │    │ Concrete Guidance       │    │
│  │ Corpus      │───▶│ Analyzer         │───▶│ Generator               │    │
│  │ (ChromaDB)  │    │                  │    │                         │    │
│  └─────────────┘    └──────────────────┘    └─────────────────────────┘    │
│                              │                          │                   │
│                              ▼                          ▼                   │
│                     ┌────────────────┐         ┌───────────────────┐       │
│                     │ 5 Analyzers:   │         │ Prompt Injection: │       │
│                     │ 1. Syntax      │         │ • POS templates   │       │
│                     │ 2. Vocabulary  │         │ • Word banks      │       │
│                     │ 3. Transition  │         │ • Transitions     │       │
│                     │ 4. Emotion     │         │ • Stance markers  │       │
│                     │ 5. Rhetoric    │         │ • Opening patterns│       │
│                     └────────────────┘         └───────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Syntactic Template Extraction (Addresses Mechanical Precision)

**Goal**: Extract concrete POS patterns the model can follow as skeletons.

#### New Data Structure

```python
@dataclass
class SyntacticTemplate:
    """Concrete syntactic template from author's prose."""
    pos_pattern: str          # "DET ADJ NOUN — ADV VERB PREP DET NOUN"
    clause_structure: str     # "main + subordinate + fragment"
    length_category: str      # "LONG" (derived, not primary)
    example_skeleton: str     # "The [ADJ] [NOUN] — [ADV] [VERB] upon the [NOUN]"
    frequency: float          # How often this pattern appears
```

#### Extraction Method

```python
def extract_syntactic_templates(self, text: str) -> List[SyntacticTemplate]:
    """Extract POS templates with clause structure."""
    doc = self.nlp(text)
    templates = []

    for sent in doc.sents:
        # Extract POS sequence with punctuation preserved
        pos_tokens = []
        for token in sent:
            if token.is_punct:
                pos_tokens.append(token.text)  # Keep actual punctuation
            elif not token.is_space:
                pos_tokens.append(token.pos_)

        # Identify clause boundaries (by subordinating conjunctions, commas)
        clause_structure = self._analyze_clause_structure(sent)

        # Create fillable skeleton
        skeleton = self._create_skeleton(sent)

        templates.append(SyntacticTemplate(
            pos_pattern=" ".join(pos_tokens),
            clause_structure=clause_structure,
            length_category=self.categorize_length(len([t for t in sent if not t.is_punct])),
            example_skeleton=skeleton,
            frequency=1.0
        ))

    return templates
```

#### Prompt Injection Format

```
SYNTACTIC TEMPLATES (follow these patterns):
• "DET ADJ NOUN — DET ADJ ADJ NOUN — VERB PREP DET NOUN"
  Skeleton: "The [ADJ] [NOUN] — the [ADJ] [ADJ] [NOUN] — [VERB] upon the [NOUN]"
• "ADV , DET NOUN VERB ."
  Skeleton: "[ADV], the [NOUN] [VERB]."
• "FRAGMENT ."
  Skeleton: "A [NOUN]." or "Pure [NOUN]."
```

---

### Phase 2: Vocabulary Cluster Extraction (Addresses Impersonal Tone)

**Goal**: Extract author-specific word banks grouped by function.

#### New Data Structure

```python
@dataclass
class VocabularyCluster:
    """Vocabulary grouped by stylistic function."""
    intensifiers: List[str]      # "utterly", "tremendously", "unspeakably"
    evaluatives: List[str]       # "blasphemous", "eldritch", "magnificent"
    emotional: List[str]         # "dread", "horror", "wonder", "fascination"
    sensory: List[str]           # "fetid", "writhing", "luminous"
    archaic: List[str]           # "whereupon", "whilst", "heretofore"
    stance_certain: List[str]    # "clearly", "obviously", "undeniably"
    stance_hedge: List[str]      # "perhaps", "possibly", "one might argue"
```

#### Extraction Method

```python
def extract_vocabulary_clusters(self, texts: List[str]) -> VocabularyCluster:
    """Extract distinctive vocabulary by function."""

    # Use POS tagging and semantic categories
    intensifiers = set()
    evaluatives = set()
    emotional = set()

    for text in texts:
        doc = self.nlp(text)
        for token in doc:
            # Intensifiers: adverbs modifying adjectives
            if token.pos_ == "ADV" and token.head.pos_ == "ADJ":
                if token.text.lower() not in COMMON_ADVERBS:
                    intensifiers.add(token.text.lower())

            # Evaluatives: adjectives with strong sentiment
            if token.pos_ == "ADJ":
                sentiment = self._get_sentiment(token.text)
                if abs(sentiment) > 0.5:
                    evaluatives.add(token.text.lower())

            # Emotional: nouns in emotional semantic category
            if token.pos_ == "NOUN" and self._is_emotional_noun(token):
                emotional.add(token.text.lower())

    return VocabularyCluster(
        intensifiers=list(intensifiers)[:20],
        evaluatives=list(evaluatives)[:30],
        emotional=list(emotional)[:20],
        # ... other categories
    )
```

#### Prompt Injection Format

```
VOCABULARY GUIDANCE (use these author-characteristic words):
• Intensifiers: utterly, tremendously, unspeakably, wholly, profoundly
• Evaluatives: blasphemous, eldritch, squamous, rugose, cyclopean
• Emotional nouns: dread, horror, fascination, revulsion, wonder
• Stance markers: clearly, one must admit, it cannot be denied
```

---

### Phase 3: Transition Inventory (Addresses Sophisticated Clarity)

**Goal**: Extract the specific connectives and transition phrases the author uses.

#### New Data Structure

```python
@dataclass
class TransitionInventory:
    """Author's transition vocabulary."""
    additive: List[str]        # "and", "moreover", "furthermore", "also"
    adversative: List[str]     # "yet", "but", "however", "nevertheless"
    causal: List[str]          # "thus", "therefore", "hence", "wherefore"
    temporal: List[str]        # "then", "whereupon", "thereafter", "meanwhile"
    exemplifying: List[str]    # "indeed", "in fact", "specifically"

    # Sentence-initial patterns
    opening_connectives: List[str]  # Words that start sentences
```

#### Extraction Method

```python
def extract_transitions(self, texts: List[str]) -> TransitionInventory:
    """Extract transition words and their frequencies."""

    transitions = {
        'additive': Counter(),
        'adversative': Counter(),
        'causal': Counter(),
        'temporal': Counter(),
    }

    # Define transition categories
    ADDITIVE = {'and', 'moreover', 'furthermore', 'also', 'besides', 'additionally'}
    ADVERSATIVE = {'but', 'yet', 'however', 'nevertheless', 'nonetheless', 'still'}
    CAUSAL = {'thus', 'therefore', 'hence', 'consequently', 'wherefore', 'so'}
    TEMPORAL = {'then', 'whereupon', 'thereafter', 'meanwhile', 'subsequently'}

    for text in texts:
        doc = self.nlp(text)
        for sent in doc.sents:
            first_word = None
            for token in sent:
                if not token.is_space and not token.is_punct:
                    first_word = token.text.lower()
                    break

            if first_word:
                if first_word in ADDITIVE:
                    transitions['additive'][first_word] += 1
                elif first_word in ADVERSATIVE:
                    transitions['adversative'][first_word] += 1
                # ... etc

    return TransitionInventory(
        additive=[w for w, _ in transitions['additive'].most_common(5)],
        adversative=[w for w, _ in transitions['adversative'].most_common(5)],
        causal=[w for w, _ in transitions['causal'].most_common(5)],
        temporal=[w for w, _ in transitions['temporal'].most_common(5)],
    )
```

#### Prompt Injection Format

```
TRANSITIONS (use the author's connectives):
• Adversative: yet, but, however, nevertheless (prefer "yet" over "however")
• Causal: thus, therefore, hence, wherefore
• Temporal: whereupon, thereafter, then
• Avoid: "Furthermore", "Additionally", "Moreover" (LLM-speak)
```

---

### Phase 4: Emotional Stance Markers (Addresses Impersonal Tone)

**Goal**: Capture how the author expresses attitude and engagement.

#### New Data Structure

```python
@dataclass
class StanceProfile:
    """Author's stance and emotional engagement patterns."""
    certainty_markers: List[str]     # "clearly", "obviously", "undeniably"
    hedging_markers: List[str]       # "perhaps", "possibly", "it seems"
    evaluative_stance: str           # "negative", "positive", "mixed"
    engagement_level: float          # 0-1 scale of emotional engagement

    # Rhetorical patterns
    rhetorical_question_freq: float  # Questions per paragraph
    exclamation_freq: float          # Exclamations per paragraph
    direct_address_freq: float       # "you", "we", "one" usage
    parenthetical_freq: float        # Asides and interruptions
```

#### Extraction Method

```python
def extract_stance_profile(self, texts: List[str]) -> StanceProfile:
    """Analyze author's emotional engagement and stance."""

    certainty = Counter()
    hedges = Counter()
    rhetorical_qs = 0
    exclamations = 0
    direct_address = 0
    total_sents = 0

    CERTAINTY = {'clearly', 'obviously', 'undeniably', 'surely', 'certainly'}
    HEDGES = {'perhaps', 'possibly', 'maybe', 'seemingly', 'apparently'}

    for text in texts:
        doc = self.nlp(text)
        for sent in doc.sents:
            total_sents += 1

            # Check for rhetorical questions
            if sent.text.strip().endswith('?'):
                rhetorical_qs += 1

            # Check for exclamations
            if '!' in sent.text:
                exclamations += 1

            # Check for stance markers
            for token in sent:
                word = token.text.lower()
                if word in CERTAINTY:
                    certainty[word] += 1
                elif word in HEDGES:
                    hedges[word] += 1

                # Direct address
                if word in {'you', 'your', 'we', 'our', 'one'}:
                    direct_address += 1

    return StanceProfile(
        certainty_markers=[w for w, _ in certainty.most_common(5)],
        hedging_markers=[w for w, _ in hedges.most_common(5)],
        rhetorical_question_freq=rhetorical_qs / max(1, total_sents / 5),
        exclamation_freq=exclamations / max(1, total_sents),
        direct_address_freq=direct_address / max(1, total_sents),
    )
```

#### Prompt Injection Format

```
EMOTIONAL ENGAGEMENT:
• Use certainty markers: "clearly", "obviously", "one cannot deny"
• Rhetorical questions: ~1 per paragraph (to engage reader)
• Exclamations: Occasional (for emphasis, not excess)
• Direct address: Use "one" for formal, "we" for inclusive
• Parenthetical asides: Use em-dashes for interruptions and elaborations
```

---

### Phase 5: Opening Pattern Templates (Addresses Sophisticated Clarity)

**Goal**: Capture how the author begins sentences for natural variety.

#### New Data Structure

```python
@dataclass
class OpeningPatterns:
    """Sentence opening patterns with frequencies."""
    patterns: Dict[str, float]  # POS pattern -> frequency
    # e.g., {"DET ADJ NOUN": 0.25, "ADV ,": 0.15, "CONJ": 0.10}

    avoid_patterns: List[str]   # Patterns to avoid (LLM-speak)
    # e.g., ["Furthermore ,", "Additionally ,", "It is important"]
```

#### Extraction Method

```python
def extract_opening_patterns(self, texts: List[str]) -> OpeningPatterns:
    """Extract sentence-initial POS patterns."""

    patterns = Counter()
    total = 0

    for text in texts:
        doc = self.nlp(text)
        for sent in doc.sents:
            # Get first 3 POS tags
            pos_tags = []
            for token in sent:
                if token.is_space:
                    continue
                if len(pos_tags) >= 3:
                    break
                pos_tags.append(token.pos_ if not token.is_punct else token.text)

            if pos_tags:
                pattern = " ".join(pos_tags)
                patterns[pattern] += 1
                total += 1

    # Convert to frequencies
    pattern_freq = {p: c/total for p, c in patterns.most_common(15)}

    return OpeningPatterns(
        patterns=pattern_freq,
        avoid_patterns=["Furthermore ,", "Additionally ,", "Moreover ,", "However ,"]
    )
```

#### Prompt Injection Format

```
SENTENCE OPENINGS (vary your starts):
• 30% - "DET ADJ NOUN..." (The ancient edifice...)
• 20% - "ADV, ..." (Slowly, the...)
• 15% - Start with VERB (Consider the...)
• 10% - FRAGMENT (A horror. Pure dread.)
• NEVER start with: "Furthermore", "Additionally", "It is important to note"
```

---

## Combined Prompt Injection

The enhanced Structural RAG will produce guidance like:

```
STRUCTURAL GUIDANCE:

RHYTHM: [LONG → SHORT → MEDIUM → FRAGMENT → LONG]
  Concrete: 28 words → 8 words → 15 words → 4 words → 32 words

SYNTACTIC TEMPLATES:
• "DET ADJ NOUN — DET NOUN of NOUN — VERB ADV"
• "ADV , DET NOUN VERB ."
• "FRAGMENT ."

VOCABULARY BANKS:
• Intensifiers: utterly, tremendously, unspeakably
• Evaluatives: blasphemous, eldritch, cyclopean, rugose
• Emotional: dread, horror, fascination, revulsion

TRANSITIONS (author-specific):
• Adversative: yet, but, nevertheless (avoid "However")
• Causal: thus, hence, wherefore
• Temporal: whereupon, thereafter

EMOTIONAL ENGAGEMENT:
• Use ~1 rhetorical question per long passage
• Occasional exclamations for horror/wonder
• Parenthetical asides with em-dashes

OPENINGS (vary these):
• 30%: "The [ADJ] [NOUN]..."
• 20%: "[ADV], the..."
• 15%: Start with verb
• 10%: Fragment
• AVOID: "Furthermore", "Additionally"
```

---

## Implementation Steps

### Step 1: Create Enhanced Analyzer Module

Create `src/rag/enhanced_analyzer.py`:

```python
class EnhancedStructuralAnalyzer:
    """Multi-channel analyzer for concrete style patterns."""

    def __init__(self):
        self.nlp = get_nlp()

    def analyze(self, texts: List[str]) -> EnhancedStyleProfile:
        return EnhancedStyleProfile(
            syntactic_templates=self.extract_syntactic_templates(texts),
            vocabulary_clusters=self.extract_vocabulary_clusters(texts),
            transitions=self.extract_transitions(texts),
            stance=self.extract_stance_profile(texts),
            openings=self.extract_opening_patterns(texts),
        )
```

### Step 2: Update StructuralRAG

Modify `src/rag/structural_rag.py`:

```python
class StructuralRAG:
    def __init__(self, author: str):
        self.analyzer = get_structural_analyzer()
        self.enhanced_analyzer = EnhancedStructuralAnalyzer()
        self._enhanced_profile: Optional[EnhancedStyleProfile] = None

    def load_enhanced_patterns(self, sample_size: int = 100):
        chunks = self.indexer.get_random_chunks(self.author, n=sample_size)
        self._enhanced_profile = self.enhanced_analyzer.analyze(chunks)

    def get_guidance(self, input_text: str) -> StructuralGuidance:
        # Include enhanced guidance
        return StructuralGuidance(
            rhythm_pattern=self.get_rhythm_pattern(...),
            syntactic_templates=self._enhanced_profile.syntactic_templates[:3],
            vocabulary_banks=self._enhanced_profile.vocabulary_clusters,
            transitions=self._enhanced_profile.transitions,
            stance_guidance=self._format_stance_guidance(),
            opening_patterns=self._enhanced_profile.openings,
        )
```

### Step 3: Update Prompt Formatter

Modify `format_for_prompt()` to include all new guidance types.

### Step 4: Add Tests

Create comprehensive tests for each analyzer component.

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Sentence length variance | Low | High (match author's corpus) |
| Transition word overlap | ~20% | ~60% (author-specific) |
| Opening pattern diversity | Low | High (match corpus distribution) |
| Emotional vocabulary usage | Generic | Author-specific |
| Rhetorical device usage | Rare | Match author frequency |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Guidance too long → context bloat | Limit to top 3 templates, top 10 words per category |
| Over-specific → rigid output | Include "vary these" instructions |
| Vocabulary mismatch → hallucination | Verify words exist in author corpus |
| Complex extraction → slow inference | Cache extracted profiles per author |

---

## Timeline

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1: Syntactic Templates | 1 day | High |
| Phase 2: Vocabulary Clusters | 1 day | High |
| Phase 3: Transition Inventory | 0.5 day | Medium |
| Phase 4: Stance Markers | 0.5 day | Medium |
| Phase 5: Opening Patterns | 0.5 day | Low |
| Integration & Testing | 1 day | High |

Total: ~4.5 days
