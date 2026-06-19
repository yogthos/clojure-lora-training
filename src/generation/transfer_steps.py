"""Pipeline transformation steps for StyleTransfer.

These are the LLM-driven stages applied to each paragraph before style
application: RTT neutralization, texture expansion, narrativization, and
perspective conversion. They are mixed into StyleTransfer and rely on the
host providing ``self.critic_provider``, ``self._rtt_neutralizer`` and the
module-level ``logger``.
"""

from typing import Optional

from ..utils.prompts import load_prompt
from ..utils.logging import get_logger

logger = get_logger(__name__)


class _TransferSteps:
    """LLM-driven transformation steps, mixed into StyleTransfer.

    Relies on host attributes ``self.critic_provider`` and
    ``self._rtt_neutralizer`` (both set up by StyleTransfer). The implicit
    ``self`` contract is intentional; this mixin is never instantiated alone.
    """

    def _rtt_neutralize(self, text: str, max_retries: int = 2) -> Optional[str]:
        """Round-Trip Translation neutralization via Mandarin pivot.

        This matches the training data generation process:
        Step 1 (Scrub): English → Mandarin (HSK3 vocabulary)
        Step 2 (Rinse): Mandarin → Plain English

        Uses provider from config.json under llm.provider.rtt.
        Options: 'mlx' (local), 'deepseek' (API).

        Args:
            text: Input text to neutralize.
            max_retries: Number of retry attempts.

        Returns:
            Neutralized text, or None if failed.
        """
        # Lazy-load the RTT neutralizer using factory function
        if self._rtt_neutralizer is None:
            try:
                from ..llm.rtt_neutralizer import create_rtt_neutralizer

                self._rtt_neutralizer = create_rtt_neutralizer()
                logger.debug(f"RTT neutralizer: {type(self._rtt_neutralizer).__name__}")
            except Exception as e:
                logger.error(f"Failed to initialize RTT neutralizer: {e}")
                return None

        return self._rtt_neutralizer.neutralize(text, max_retries=max_retries)

    def _expand_with_texture(self, text: str) -> str:
        """Expand text with texture using the critic model.

        Adds asides, observations, parenthetical thoughts, and sensory details
        to enrich flat prose before style transfer.

        Args:
            text: Input text to expand.

        Returns:
            Expanded text with added texture, or original text if expansion fails.
        """
        try:
            system_prompt = load_prompt("expand_texture")
            user_prompt = text

            response = self.critic_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,  # Some creativity for texture
                max_tokens=len(text.split()) * 3,  # Allow ~2x expansion headroom
            )

            input_words = len(text.split())
            output_words = len(response.split()) if response else 0
            logger.info(
                f"TEXTURE EXPANSION result: {input_words} → {output_words} words"
            )

            if response and output_words > input_words:
                expansion = output_words / input_words
                logger.info(f"TEXTURE EXPANSION: expanded by {expansion:.0%}")
                return response.strip()
            else:
                logger.warning(
                    f"Texture expansion returned shorter/equal text ({output_words} vs {input_words}), using original"
                )
                return text

        except Exception as e:
            logger.warning(f"Texture expansion failed: {e}")
            return text

    def _narrativize(self, text: str) -> str:
        """Convert impersonal exposition to first-person narrative.

        CRITICAL FOR LORA QUALITY:
        The LoRA was trained on first-person narrative inputs ("I saw", "I found",
        "I discovered"). But RTT neutralization produces impersonal exposition
        ("We trace", "One observes", "It is known that").

        This step bridges that gap by converting input to match training format:
        - "We now trace the forces..." → "I have traced the forces..."
        - "One must understand..." → "I came to understand..."
        - "It is observed that..." → "I observed..."

        Args:
            text: Impersonal exposition text.

        Returns:
            First-person narrative version, or original text if conversion fails.
        """
        try:
            system_prompt = load_prompt("narrativize")
            user_prompt = text

            response = self.critic_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.5,  # Some variation but controlled
                max_tokens=len(text.split()) * 2,  # Allow for slight expansion
            )

            if response and response.strip():
                input_words = len(text.split())
                output_words = len(response.split())
                logger.info(
                    f"NARRATIVIZE: {input_words} → {output_words} words (converted to first-person)"
                )
                return response.strip()
            else:
                logger.warning("Narrativization returned empty, using original")
                return text

        except Exception as e:
            logger.warning(f"Narrativization failed: {e}")
            return text

    def _convert_to_perspective(self, text: str, target_perspective: str) -> str:
        """Convert text to target perspective BEFORE RTT neutralization.

        CRITICAL: This must happen BEFORE RTT because the LoRA was trained on
        perspective-varied text that went through RTT. The training pairs are:
            neutral(third_person) → styled(third_person)

        So the perspective is embedded in the text BEFORE RTT, and the LoRA
        preserves it during styling.

        Args:
            text: Input text in any perspective.
            target_perspective: Target perspective from config.

        Returns:
            Text converted to target perspective.
        """
        # "preserve" means don't convert - keep original perspective
        if target_perspective == "preserve":
            return text

        # "first_person_singular" uses the existing narrativize prompt
        if target_perspective == "first_person_singular":
            return self._narrativize(text)

        try:
            # Build the perspective description
            perspective_descriptions = {
                "first_person_plural": "first_person_plural (use: we, us, our, ours)",
                "third_person": "third_person (use: the observer, they, one)",
                "author_voice_third_person": "author_voice_third_person (impersonal exposition: one observes, it is known, passive voice)",
            }
            perspective_desc = perspective_descriptions.get(
                target_perspective, target_perspective
            )

            system_prompt = load_prompt("convert_perspective").format(
                target_perspective=perspective_desc
            )
            user_prompt = text

            response = self.critic_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,  # Low temperature for precise conversion
                max_tokens=len(text.split()) * 2,
            )

            if response and response.strip():
                input_words = len(text.split())
                output_words = len(response.split())
                logger.info(
                    f"PERSPECTIVE CONVERSION: {input_words} → {output_words} words "
                    f"(converted to {target_perspective})"
                )
                return response.strip()
            else:
                logger.warning("Perspective conversion returned empty, using original")
                return text

        except Exception as e:
            logger.warning(f"Perspective conversion failed: {e}")
            return text
