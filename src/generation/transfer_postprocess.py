"""Post-processing helpers for StyleTransfer output.

Content-overlap checks and punctuation/sentence-ending cleanup applied after
the LoRA generates styled text. Mixed into StyleTransfer; ``_check_content_overlap``
relies on ``self.services.nlp`` (spaCy).
"""

import re

from ..utils.nlp import split_into_sentences
from ..utils.logging import get_logger

logger = get_logger(__name__)


class _TransferPostProcess:
    """Output post-processing steps, mixed into StyleTransfer."""

    def _check_content_overlap(self, input_text: str, output_text: str) -> float:
        """Check content word overlap between input and output.

        Returns ratio of input content words found in output.
        Low overlap suggests memorized/hallucinated output.
        """
        nlp = self.services.nlp

        def get_content_words(text: str) -> set:
            """Extract lemmatized content words using spaCy."""
            doc = nlp(text)
            words = set()
            for token in doc:
                # Skip stopwords, punctuation, and short words
                if not token.is_stop and not token.is_punct and len(token.lemma_) >= 4:
                    words.add(token.lemma_.lower())
            return words

        input_words = get_content_words(input_text)
        output_words = get_content_words(output_text)

        if not input_words:
            return 1.0  # No content words to check

        overlap = len(input_words & output_words)
        return overlap / len(input_words)

    def _clean_punctuation_artifacts(self, text: str) -> str:
        """Clean up punctuation artifacts from LoRA output and post-processing.

        Fixes common issues like:
        - "—," or ",—" (em-dash combined with comma)
        - ".—" or "—." (em-dash combined with period)
        - Double punctuation
        """
        # Fix em-dash + punctuation combinations
        text = re.sub(r"—\s*,", ",", text)  # "—," -> ","
        text = re.sub(r",\s*—", ",", text)  # ",—" -> ","
        text = re.sub(r"—\s*\.", ".", text)  # "—." -> "."
        text = re.sub(r"\.\s*—", ".", text)  # ".—" -> "."
        text = re.sub(r"—\s*;", ";", text)  # "—;" -> ";"
        text = re.sub(r";\s*—", ";", text)  # ";—" -> ";"
        text = re.sub(r"—\s*:", ":", text)  # "—:" -> ":"
        text = re.sub(r":\s*—", ":", text)  # ":—" -> ":"

        # Fix double punctuation
        text = re.sub(r",\s*,", ",", text)
        text = re.sub(r"\.\s*\.", ".", text)
        text = re.sub(r";\s*;", ";", text)
        text = re.sub(r":\s*:", ":", text)

        # Fix spacing around punctuation
        text = re.sub(r"\s+([.,;:!?])", r"\1", text)  # No space before
        # Space after punctuation, but not between single uppercase letters (abbreviations like U.S.)
        text = re.sub(r"([.,;:!?])(?!(?<=[A-Z]\.)[A-Z])([A-Za-z])", r"\1 \2", text)

        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _ensure_complete_ending(self, text: str) -> str:
        """Ensure text ends with a complete sentence.

        If text ends mid-sentence, remove the incomplete part.
        """
        # First clean punctuation artifacts
        text = self._clean_punctuation_artifacts(text)

        text = text.strip()
        if not text:
            return text

        # If already ends with sentence terminator, we're good
        if text[-1] in ".!?":
            return text

        # Find the last complete sentence
        sentences = split_into_sentences(text)
        if not sentences:
            return text

        # Check if last sentence is complete (ends with punctuation)
        complete_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if sent and sent[-1] in ".!?":
                complete_sentences.append(sent)
            elif sent and len(sent) > 20:
                # Long fragment - try to salvage by adding period
                # Only if it looks like a complete thought
                words = sent.split()
                if len(words) >= 5:
                    complete_sentences.append(sent + ".")
                    logger.warning(
                        f"Added period to incomplete sentence: ...{sent[-30:]}"
                    )

        if complete_sentences:
            return " ".join(complete_sentences)

        # Fallback: add period to entire text
        return text + "."
