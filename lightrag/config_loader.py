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
        self.schema_match_threshold = get_env_value("SCHEMA_MATCH_CONFIDENCE_THRESHOLD", 0.75, float)
        self.new_type_threshold = get_env_value("NEW_TYPE_CONFIDENCE_THRESHOLD", 0.85, float)
        self.default_entity_type = get_env_value("DEFAULT_ENTITY_TYPE", "UNKNOWN")
        self.default_relationship_type = get_env_value("DEFAULT_RELATIONSHIP_TYPE", "RELATED_TO")
        self.enable_schema_classification = get_env_value("ENABLE_SCHEMA_CLASSIFICATION", True, bool)

        # Feature flags
        self.enable_diagram_analysis = get_env_value("ENABLE_DIAGRAM_ANALYSIS", True, bool)
        self.enable_formula_analysis = get_env_value("ENABLE_FORMULA_ANALYSIS", True, bool)

        # Chain-of-Thought settings
        self.enable_cot = get_env_value("ENABLE_COT", True, bool)
        self.max_cot_refinement_attempts = get_env_value("MAX_COT_REFINEMENT_ATTEMPTS", 2, int)
        self.enable_enhanced_citations = get_env_value("ENABLE_ENHANCED_CITATIONS", True, bool)

        # Diagram and Formula integration settings
        self.enable_diagram_formula_integration = get_env_value("ENABLE_DIAGRAM_FORMULA_INTEGRATION", True, bool)
        self.resolve_placeholders_in_context = get_env_value("RESOLVE_PLACEHOLDERS_IN_CONTEXT", True, bool)
        self.diagram_citation_format = get_env_value("DIAGRAM_CITATION_FORMAT", "[Diagram ID: {id}]")
        self.formula_citation_format = get_env_value("FORMULA_CITATION_FORMAT", "[Formula ID: {id}]")
        self.placeholder_output_format = get_env_value("PLACEHOLDER_OUTPUT_FORMAT", "detailed")

        # Entity resolution settings
        self.entity_resolution_name_threshold = get_env_value("ENTITY_RESOLUTION_NAME_THRESHOLD", 0.8, float)
        self.entity_resolution_embedding_threshold = get_env_value("ENTITY_RESOLUTION_EMBEDDING_THRESHOLD", 0.85, float)
        self.entity_resolution_context_threshold = get_env_value("ENTITY_RESOLUTION_CONTEXT_THRESHOLD", 0.7, float)
        self.entity_resolution_final_threshold = get_env_value("ENTITY_RESOLUTION_FINAL_THRESHOLD", 0.85, float)
        self.entity_resolution_merge_weight_property = get_env_value("ENTITY_RESOLUTION_MERGE_WEIGHT_PROPERTY", 0.2, float)

        # Entity resolution weights
        self.entity_resolution_weight_name = get_env_value("ENTITY_RESOLUTION_WEIGHT_NAME", 0.3, float)
        self.entity_resolution_weight_alias = get_env_value("ENTITY_RESOLUTION_WEIGHT_ALIAS", 0.15, float)
        self.entity_resolution_weight_embedding = get_env_value("ENTITY_RESOLUTION_WEIGHT_EMBEDDING", 0.35, float)
        self.entity_resolution_weight_context = get_env_value("ENTITY_RESOLUTION_WEIGHT_CONTEXT", 0.2, float)

        # Entity resolution batch settings
        self.entity_resolution_batch_size = get_env_value("ENTITY_RESOLUTION_BATCH_SIZE", 100, int)
        self.entity_resolution_candidate_limit = get_env_value("ENTITY_RESOLUTION_CANDIDATE_LIMIT", 10, int)
        self.entity_resolution_string_similarity_method = get_env_value("ENTITY_RESOLUTION_STRING_SIMILARITY_METHOD", "fuzzy_ratio")

        # Diagram settings
        self.diagram_detection_threshold = get_env_value("DIAGRAM_DETECTION_THRESHOLD", 0.6, float)

        # Text chunking settings
        self.chunking_strategy = get_env_value("CHUNKING_STRATEGY", "token")
        self.chunk_size = get_env_value("CHUNK_SIZE", 1200, int)
        self.chunk_overlap_size = get_env_value("CHUNK_OVERLAP_SIZE", 100, int)

        # Advanced chunking settings
        self.use_semantic_boundaries = get_env_value("USE_SEMANTIC_BOUNDARIES", True, bool)
        self.use_hierarchical_chunking = get_env_value("USE_HIERARCHICAL_CHUNKING", True, bool)
        self.adaptive_chunking = get_env_value("ADAPTIVE_CHUNKING", True, bool)
        self.preserve_entities = get_env_value("PRESERVE_ENTITIES", True, bool)
        self.track_cross_references = get_env_value("TRACK_CROSS_REFERENCES", True, bool)
        self.enable_multi_resolution = get_env_value("ENABLE_MULTI_RESOLUTION", True, bool)
        self.content_type_aware = get_env_value("CONTENT_TYPE_AWARE", True, bool)
        self.nlp_model = get_env_value("NLP_MODEL", "en_core_web_sm")
        self.min_shared_entities = get_env_value("MIN_SHARED_ENTITIES", 2, int)

        # Schema entity types for entity-aware chunking
        entity_types_str = get_env_value("SCHEMA_ENTITY_TYPES", "")
        self.schema_entity_types = [t.strip() for t in entity_types_str.split(",")] if entity_types_str else []

        # Query processing settings
        self.enable_intelligent_retrieval = get_env_value("ENABLE_INTELLIGENT_RETRIEVAL", True, bool)
        self.query_analysis_confidence_threshold = get_env_value("QUERY_ANALYSIS_CONFIDENCE_THRESHOLD", 0.7, float)
        self.auto_strategy_selection = get_env_value("AUTO_STRATEGY_SELECTION", True, bool)
        self.default_retrieval_strategy = get_env_value("DEFAULT_RETRIEVAL_STRATEGY", "hybrid")
        self.retrieval_similarity_threshold = get_env_value("RETRIEVAL_SIMILARITY_THRESHOLD", 0.6, float)
        self.retrieval_max_related_depth = get_env_value("RETRIEVAL_MAX_RELATED_DEPTH", 2, int)
        self.retrieval_limit = get_env_value("RETRIEVAL_LIMIT", 60, int)

        # Query intent indicators
        graph_indicators_str = get_env_value("GRAPH_INTENT_INDICATORS",
            "related,connected,relationship,connection,link,between,compare,difference,similar,versus,vs,contrast,cause,effect,impact,influence")
        self.graph_intent_indicators = [t.strip() for t in graph_indicators_str.split(",")] if graph_indicators_str else []

        vector_indicators_str = get_env_value("VECTOR_INTENT_INDICATORS",
            "like,similar to,example of,such as,about,concept of,definition,meaning,explain,describe,summarize")
        self.vector_intent_indicators = [t.strip() for t in vector_indicators_str.split(",")] if vector_indicators_str else []


# Create a singleton instance
enhanced_config = EnhancedConfig()


def get_enhanced_config() -> EnhancedConfig:
    """
    Get the enhanced configuration singleton.

    Returns:
        EnhancedConfig: The enhanced configuration instance
    """
    return enhanced_config
