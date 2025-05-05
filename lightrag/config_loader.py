"""
Configuration loader for LightRAG enhanced features.
This module loads configuration settings for schema, diagram/formula analysis,
Chain-of-Thought, and entity resolution from environment variables.
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Set up logger
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)


def get_env_value(env_key: str, default: Any, value_type: type = str) -> Any:
    """
    Get value from environment variable with type conversion

    Args:
        env_key (str): Environment variable key
        default (any): Default value if env variable is not set
        value_type (type): Type to convert the value to

    Returns:
        any: Converted value from environment or default
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    if value_type is bool:
        return value.lower() in ("true", "1", "yes", "t", "on")
    try:
        return value_type(value)
    except ValueError:
        logger.warning(f"Could not convert {env_key}={value} to {value_type}, using default {default}")
        return default


class EnhancedConfig:
    """Configuration for enhanced LightRAG features."""

    def __init__(self):
        # Schema configuration
        self.schema_file_path = get_env_value("SCHEMA_FILE_PATH", "./docs/schema.json")
        
        # Feature flags
        self.enable_diagram_analysis = get_env_value("ENABLE_DIAGRAM_ANALYSIS", True, bool)
        self.enable_formula_analysis = get_env_value("ENABLE_FORMULA_ANALYSIS", True, bool)
        self.enable_cot = get_env_value("ENABLE_COT", True, bool)
        
        # Entity resolution thresholds
        self.entity_resolution_name_threshold = get_env_value("ENTITY_RESOLUTION_NAME_THRESHOLD", 0.8, float)
        self.entity_resolution_embedding_threshold = get_env_value("ENTITY_RESOLUTION_EMBEDDING_THRESHOLD", 0.85, float)
        self.entity_resolution_context_threshold = get_env_value("ENTITY_RESOLUTION_CONTEXT_THRESHOLD", 0.7, float)
        self.entity_resolution_merge_weight_property = get_env_value("ENTITY_RESOLUTION_MERGE_WEIGHT_PROPERTY", 0.2, float)
        
        # Diagram settings
        self.diagram_detection_threshold = get_env_value("DIAGRAM_DETECTION_THRESHOLD", 0.6, float)


# Create a singleton instance
enhanced_config = EnhancedConfig()


def get_enhanced_config() -> EnhancedConfig:
    """
    Get the enhanced configuration singleton.
    
    Returns:
        EnhancedConfig: The enhanced configuration instance
    """
    return enhanced_config
