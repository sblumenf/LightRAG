"""
LLM module for LightRAG.

This module provides functionality for generating responses using LLMs,
including Chain-of-Thought reasoning, enhanced citation handling,
and diagram/formula description integration.
"""

from lightrag.llm.llm_generator import (
    LLMGenerator,
    LLMGenerationError,
    extract_reasoning_and_answer,
    process_citations
)
from lightrag.llm.placeholder_resolver import (
    PlaceholderResolver,
    resolve_placeholders_in_context
)

__all__ = [
    'LLMGenerator',
    'LLMGenerationError',
    'extract_reasoning_and_answer',
    'process_citations',
    'PlaceholderResolver',
    'resolve_placeholders_in_context'
]