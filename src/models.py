"""Shared data models for the LLM and generation subsystems."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class MessageRole(str, Enum):
    """Role of a message in an LLM conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in an LLM conversation."""
    role: MessageRole
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dict format for API calls."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
