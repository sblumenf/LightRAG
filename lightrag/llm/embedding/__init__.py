"""Embedding functionality for LightRAG.

This package provides embedding generation functionality, including:
- Basic embeddings with standard LLM providers
- Enhanced embeddings with improved error handling, batch processing, and retries
- Cached embeddings for improved performance and reduced API costs
"""

from lightrag.llm.embedding_generator import get_embedding_func
from lightrag.llm.enhanced_embedding import create_openai_enhanced_embedding, create_google_enhanced_embedding
from .cached_embedding import create_cached_embedding, CachedEmbedding

__all__ = [
    'get_embedding_func',
    'create_openai_enhanced_embedding',
    'create_google_enhanced_embedding',
    'create_cached_embedding',
    'CachedEmbedding',
]
