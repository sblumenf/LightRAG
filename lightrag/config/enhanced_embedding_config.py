"""
Configuration module for enhanced embedding settings.

This module provides configuration options for the enhanced embedding system,
including environment variable support and default values.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Default values
DEFAULT_ENABLED = False
DEFAULT_PROVIDER = "openai"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_GOOGLE_MODEL = "models/embedding-001"


@dataclass
class EnhancedEmbeddingConfig:
    """Configuration for enhanced embedding system."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("ENHANCED_EMBEDDING_ENABLED", str(DEFAULT_ENABLED)).lower() in ("true", "1", "yes")
    )
    """Whether to use the enhanced embedding system."""

    provider: str = field(
        default_factory=lambda: os.getenv("ENHANCED_EMBEDDING_PROVIDER", DEFAULT_PROVIDER)
    )
    """The embedding provider to use ('openai' or 'google')."""

    batch_size: int = field(
        default_factory=lambda: int(os.getenv("ENHANCED_EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    )
    """Number of texts to process in a single API call."""

    max_retries: int = field(
        default_factory=lambda: int(os.getenv("ENHANCED_EMBEDDING_MAX_RETRIES", DEFAULT_MAX_RETRIES))
    )
    """Maximum number of retries for API calls."""

    retry_delay: float = field(
        default_factory=lambda: float(os.getenv("ENHANCED_EMBEDDING_RETRY_DELAY", DEFAULT_RETRY_DELAY))
    )
    """Delay between retries in seconds."""

    openai_model: str = field(
        default_factory=lambda: os.getenv("ENHANCED_EMBEDDING_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    )
    """The OpenAI embedding model to use."""

    google_model: str = field(
        default_factory=lambda: os.getenv("ENHANCED_EMBEDDING_GOOGLE_MODEL", DEFAULT_GOOGLE_MODEL)
    )
    """The Google embedding model to use."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "enabled": self.enabled,
            "provider": self.provider,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "openai_model": self.openai_model,
            "google_model": self.google_model,
        }


def get_enhanced_embedding_config() -> EnhancedEmbeddingConfig:
    """Get the enhanced embedding configuration."""
    return EnhancedEmbeddingConfig()
