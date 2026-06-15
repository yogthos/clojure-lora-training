"""Shared data models for the LLM and generation subsystems."""

from dataclasses import dataclass
from enum import Enum


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


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
