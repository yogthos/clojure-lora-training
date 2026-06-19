"""MLX-based LLM provider for text generation.

Uses the same Qwen model as the LoRA pipeline for consistency.
This allows the entire pipeline to be self-contained without external services.
"""

import json
from typing import Optional
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)


def _load_mlx_config() -> dict:
    """Load MLX config from config.json."""
    config_path = Path(__file__).parent.parent.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            return config.get("llm", {}).get("providers", {}).get("mlx", {})
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    return {}

# Check MLX availability
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. Install with: pip install mlx mlx-lm")


class MLXGenerator:
    """MLX-based text generator using Qwen model.

    Can be used for:
    - Neutralizing author text (converting to plain English)
    - Any other text generation tasks in the pipeline

    Configuration is loaded from config.json under llm.providers.mlx:
        {
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "max_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.9
        }

    Example:
        generator = MLXGenerator()
        neutral = generator.generate(
            prompt="Convert to plain English: ...",
            max_tokens=200,
        )
    """

    DEFAULT_MODEL = "mlx-community/Qwen3-8B-4bit"

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Initialize the MLX generator.

        Args:
            model_name: Model to use (from config or default).
            temperature: Generation temperature (from config or 0.3).
            top_p: Top-p sampling parameter (from config or 0.9).
            max_tokens: Default max tokens (from config or 512).
        """
        if not MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is not available. Install with: pip install mlx mlx-lm\n"
                "Note: MLX only works on Apple Silicon Macs."
            )

        # Load config defaults
        config = _load_mlx_config()

        self.model_name = model_name or config.get("model", self.DEFAULT_MODEL)
        self.temperature = temperature if temperature is not None else config.get("temperature", 0.3)
        self.top_p = top_p if top_p is not None else config.get("top_p", 0.9)
        self.default_max_tokens = max_tokens if max_tokens is not None else config.get("max_tokens", 512)

        # Detect if this is a base model (no chat template)
        self.is_base_model = "instruct" not in self.model_name.lower() and "chat" not in self.model_name.lower()

        logger.info(f"MLX config: model={self.model_name}, base_model={self.is_base_model}, temp={self.temperature}")

        # Lazy load model
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._model is not None:
            return

        logger.debug(f"Loading MLX model: {self.model_name}")
        self._model, self._tokenizer = load(self.model_name)
        logger.debug("Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The user prompt.
            max_tokens: Maximum tokens to generate (from config if not specified).
            system_prompt: Optional system prompt.
            temperature: Override temperature for this call.

        Returns:
            Generated text.
        """
        self._ensure_loaded()
        max_tokens = max_tokens or self.default_max_tokens

        # For base models, use raw text completion
        if self.is_base_model:
            # Build a simple prompt format for base models
            if system_prompt:
                formatted_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                formatted_prompt = prompt
        else:
            # Build messages for instruct models
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            # Apply chat template
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Create sampler
        temp = temperature if temperature is not None else self.temperature
        sampler = make_sampler(temp=temp, top_p=self.top_p)

        # Generate
        response = generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )

        return response.strip()

    def unload(self):
        """Unload model to free memory."""
        self._model = None
        self._tokenizer = None
        logger.info("Model unloaded")


