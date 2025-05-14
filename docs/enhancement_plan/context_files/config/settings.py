"""
Secure configuration management for the GraphRAG tutor application.

This module uses Pydantic for structured configuration management with type validation,
environment variable loading, and nested configuration models.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jSettings(BaseModel):
    """Neo4j database connection settings."""
    uri: str = Field("bolt://localhost:7687", description="Neo4j connection URI")
    username: str = Field("neo4j", description="Neo4j username")
    password: str = Field("password", description="Neo4j password")
    database: str = Field("neo4j", description="Neo4j database name")
    vector_index_name: str = Field("text_embeddings", description="Name of the vector index in Neo4j")
    vector_dimensions: int = Field(768, description="Dimensions of the embedding vectors")


class LLMSettings(BaseModel):
    """LLM provider and model settings."""
    provider: str = Field("google", json_schema_extra={'env': 'LLM_PROVIDER'})
    model_name: str = Field("gemini-1.5-pro", json_schema_extra={'env': 'LLM_MODEL_NAME'})
    temperature: float = Field(0.7, json_schema_extra={'env': 'LLM_TEMPERATURE'})
    max_tokens: int = Field(1000, json_schema_extra={'env': 'LLM_MAX_TOKENS'})
    max_context_length: int = Field(8000, json_schema_extra={'env': 'LLM_MAX_CONTEXT_LENGTH'})
    max_refinement_attempts: int = Field(2, json_schema_extra={'env': 'LLM_MAX_REFINEMENT_ATTEMPTS'})

    # OpenAI specific settings
    openai_api_key: Optional[str] = Field(None, json_schema_extra={'env': 'OPENAI_API_KEY'})
    openai_model: str = Field("gpt-4o-mini", json_schema_extra={'env': 'DEFAULT_LLM_MODEL'})

    # Google AI specific settings
    google_api_key: Optional[str] = Field(None, json_schema_extra={'env': 'GOOGLE_API_KEY'})
    google_model: str = Field("gemini-1.5-pro", json_schema_extra={'env': 'DEFAULT_GOOGLE_LLM_MODEL'})


class EmbeddingSettings(BaseModel):
    """Embedding generation settings."""
    provider: str = Field("google", json_schema_extra={'env': 'EMBEDDING_PROVIDER'})
    vector_dimensions: int = Field(768, json_schema_extra={'env': 'VECTOR_DIMENSIONS'})
    batch_size: int = Field(10, json_schema_extra={'env': 'EMBEDDING_BATCH_SIZE'})
    max_retries: int = Field(3, json_schema_extra={'env': 'EMBEDDING_MAX_RETRIES'})
    retry_delay: float = Field(1.0, json_schema_extra={'env': 'EMBEDDING_RETRY_DELAY'})

    # OpenAI specific settings
    openai_model: str = Field("text-embedding-3-small", json_schema_extra={'env': 'DEFAULT_OPENAI_EMBEDDING_MODEL'})

    # Google AI specific settings
    google_model: str = Field("models/embedding-004", json_schema_extra={'env': 'DEFAULT_EMBEDDING_MODEL'})


class ChunkerSettings(BaseModel):
    """Text chunking settings."""
    default_chunk_size: int = Field(500, json_schema_extra={'env': 'TEXT_CHUNKER_DEFAULT_CHUNK_SIZE'})
    default_overlap: int = Field(50, json_schema_extra={'env': 'TEXT_CHUNKER_DEFAULT_OVERLAP'})
    use_semantic_boundaries: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_USE_SEMANTIC'})
    use_hierarchical_chunking: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_USE_HIERARCHICAL'})
    adaptive_chunking: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_ADAPTIVE'})
    preserve_entities: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_PRESERVE_ENTITIES'})
    track_cross_references: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_TRACK_CROSS_REFS'})
    enable_multi_resolution: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_MULTI_RESOLUTION'})
    content_type_aware: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_CONTENT_AWARE'})
    nlp_model: str = Field("en_core_web_sm", json_schema_extra={'env': 'TEXT_CHUNKER_NLP_MODEL'})
    min_shared_entities: int = Field(2, json_schema_extra={'env': 'TEXT_CHUNKER_MIN_SHARED_ENTITIES'})


class RetrievalSettings(BaseModel):
    """Retrieval system settings."""
    default_limit: int = Field(5, json_schema_extra={'env': 'DEFAULT_RETRIEVAL_LIMIT'})
    similarity_threshold: float = Field(0.6, json_schema_extra={'env': 'RETRIEVAL_SIMILARITY_THRESHOLD'})
    max_related_depth: int = Field(2, json_schema_extra={'env': 'RETRIEVAL_MAX_RELATED_DEPTH'})


class SchemaSettings(BaseModel):
    """Schema and entity resolution settings."""
    file_path: str = Field("./docs/schema.json", json_schema_extra={'env': 'SCHEMA_FILE_PATH'})
    match_confidence_threshold: float = Field(0.75, json_schema_extra={'env': 'SCHEMA_MATCH_CONFIDENCE_THRESHOLD'})
    new_type_confidence_threshold: float = Field(0.85, json_schema_extra={'env': 'NEW_TYPE_CONFIDENCE_THRESHOLD'})
    default_entity_type: str = Field("Chunk", json_schema_extra={'env': 'DEFAULT_ENTITY_TYPE'})
    default_relationship_type: str = Field("RELATED_TO", json_schema_extra={'env': 'DEFAULT_RELATIONSHIP_TYPE'})
    tentative_entity_prefix: str = Field("Tentative", json_schema_extra={'env': 'TENTATIVE_ENTITY_PREFIX'})
    tentative_relationship_prefix: str = Field("TENTATIVE_", json_schema_extra={'env': 'TENTATIVE_RELATIONSHIP_PREFIX'})


class EntityResolverSettings(BaseModel):
    """Entity resolution settings."""
    name_threshold: int = Field(85, json_schema_extra={'env': 'ENTITY_RESOLUTION_NAME_THRESHOLD'})
    alias_threshold: int = Field(80, json_schema_extra={'env': 'ENTITY_RESOLUTION_ALIAS_THRESHOLD'})
    embedding_threshold: float = Field(0.90, json_schema_extra={'env': 'ENTITY_RESOLUTION_EMBEDDING_THRESHOLD'})
    property_threshold: float = Field(0.75, json_schema_extra={'env': 'ENTITY_RESOLUTION_PROPERTY_THRESHOLD'})
    final_threshold: float = Field(0.85, json_schema_extra={'env': 'ENTITY_RESOLUTION_FINAL_THRESHOLD'})
    candidate_limit: int = Field(10, json_schema_extra={'env': 'ENTITY_RESOLUTION_CANDIDATE_LIMIT'})
    ignore_properties: List[str] = Field(
        ["id", "created_at", "updated_at"],
        json_schema_extra={'env': 'ENTITY_RESOLUTION_IGNORE_PROPERTIES'}
    )
    string_similarity_method: str = Field(
        "jaro_winkler",
        json_schema_extra={'env': 'ENTITY_RESOLUTION_STRING_SIMILARITY_METHOD'}
    )
    merge_weight_name: float = Field(0.3, json_schema_extra={'env': 'ENTITY_RESOLUTION_MERGE_WEIGHT_NAME'})
    merge_weight_alias: float = Field(0.15, json_schema_extra={'env': 'ENTITY_RESOLUTION_MERGE_WEIGHT_ALIAS'})
    merge_weight_embedding: float = Field(0.55, json_schema_extra={'env': 'ENTITY_RESOLUTION_MERGE_WEIGHT_EMBEDDING'})
    merge_weight_property: float = Field(0.2, json_schema_extra={'env': 'ENTITY_RESOLUTION_MERGE_WEIGHT_PROPERTY'})
    batch_size: int = Field(100, json_schema_extra={'env': 'ENTITY_RESOLUTION_BATCH_SIZE'})
    default_conflict_strategy: str = Field("keep_latest", json_schema_extra={'env': 'ENTITY_DEFAULT_CONFLICT_STRATEGY'})


class DiagramSettings(BaseModel):
    """Diagram analysis settings."""
    detection_threshold: float = Field(0.6, json_schema_extra={'env': 'DIAGRAM_DETECTION_THRESHOLD'})
    default_prompts: Dict[str, str] = Field(
        {
            'general': "Analyze this diagram and provide a detailed textual description. Focus on identifying the diagram type, key components, and the relationships or processes it illustrates. Be concise and informative.",
            'flowchart': "Analyze this flowchart and describe the process and steps involved. Focus on the sequence of actions, decision points, and overall workflow. Be detailed and clear.",
            'bar_chart': "Analyze this bar chart and describe the data being presented, including comparisons and trends. Focus on categories, values, axes, and any significant patterns. Be analytical and precise.",
            'financial': "Analyze this financial diagram and explain the financial concepts, trends, and metrics displayed. Focus on financial elements, indicators, and their implications. Be specific and use financial terminology.",
            'scientific': "Analyze this scientific diagram and explain the scientific principles, processes, and structures illustrated. Focus on scientific components, labels, and their functions. Be thorough and scientifically accurate.",
            'network_diagram': "Analyze this network diagram and describe the network topology, components, and connections. Focus on nodes, links, network structure, and data flow. Be comprehensive and technically detailed.",
            'organizational_chart': "Analyze this organizational chart and describe the hierarchical structure, roles, and relationships within the organization. Focus on reporting lines, departments, and key personnel. Be structured and focus on organizational aspects."
        },
        json_schema_extra={'env': 'DEFAULT_DIAGRAM_PROMPTS'}
    )


class APISettings(BaseModel):
    """API server settings."""
    host: str = Field("0.0.0.0", json_schema_extra={'env': 'HOST'})
    port: int = Field(8000, json_schema_extra={'env': 'PORT'})
    log_level: str = Field("INFO", json_schema_extra={'env': 'LOG_LEVEL'})
    allowed_origins: str = Field(
        "http://localhost,http://localhost:3000,http://localhost:8080",
        json_schema_extra={'env': 'ALLOWED_ORIGINS'}
    )
    api_key: Optional[str] = Field(None, json_schema_extra={'env': 'API_KEY'})


class AppSettings(BaseSettings):
    """Main application settings."""
    # Neo4j settings
    neo4j_uri: str = Field("bolt://localhost:7687", json_schema_extra={'env': 'NEO4J_URI'})
    neo4j_username: str = Field("neo4j", json_schema_extra={'env': 'NEO4J_USERNAME'})
    neo4j_password: str = Field("password", json_schema_extra={'env': 'NEO4J_PASSWORD'})
    neo4j_database: str = Field("neo4j", json_schema_extra={'env': 'NEO4J_DATABASE'})
    vector_index_name: str = Field("text_embeddings", json_schema_extra={'env': 'VECTOR_INDEX_NAME'})
    vector_dimensions: int = Field(768, json_schema_extra={'env': 'VECTOR_DIMENSIONS'})

    # LLM settings
    llm_provider: str = Field("google", json_schema_extra={'env': 'LLM_PROVIDER'})
    llm_model_name: str = Field("gemini-1.5-pro", json_schema_extra={'env': 'LLM_MODEL_NAME'})
    llm_temperature: float = Field(0.7, json_schema_extra={'env': 'LLM_TEMPERATURE'})
    llm_max_tokens: int = Field(1000, json_schema_extra={'env': 'LLM_MAX_TOKENS'})
    llm_max_context_length: int = Field(8000, json_schema_extra={'env': 'LLM_MAX_CONTEXT_LENGTH'})
    llm_max_refinement_attempts: int = Field(2, json_schema_extra={'env': 'LLM_MAX_REFINEMENT_ATTEMPTS'})
    openai_api_key: Optional[str] = Field(None, json_schema_extra={'env': 'OPENAI_API_KEY'})
    openai_model: str = Field("gpt-4o-mini", json_schema_extra={'env': 'DEFAULT_LLM_MODEL'})
    google_api_key: Optional[str] = Field(None, json_schema_extra={'env': 'GOOGLE_API_KEY'})
    google_model: str = Field("gemini-1.5-pro", json_schema_extra={'env': 'DEFAULT_GOOGLE_LLM_MODEL'})

    # Embedding settings
    embedding_provider: str = Field("google", json_schema_extra={'env': 'EMBEDDING_PROVIDER'})
    embedding_batch_size: int = Field(10, json_schema_extra={'env': 'EMBEDDING_BATCH_SIZE'})
    embedding_max_retries: int = Field(3, json_schema_extra={'env': 'EMBEDDING_MAX_RETRIES'})
    embedding_retry_delay: float = Field(1.0, json_schema_extra={'env': 'EMBEDDING_RETRY_DELAY'})
    openai_embedding_model: str = Field("text-embedding-3-small", json_schema_extra={'env': 'DEFAULT_OPENAI_EMBEDDING_MODEL'})
    google_embedding_model: str = Field("models/embedding-004", json_schema_extra={'env': 'DEFAULT_EMBEDDING_MODEL'})

    # Chunker settings
    text_chunker_default_chunk_size: int = Field(500, json_schema_extra={'env': 'TEXT_CHUNKER_DEFAULT_CHUNK_SIZE'})
    text_chunker_default_overlap: int = Field(50, json_schema_extra={'env': 'TEXT_CHUNKER_DEFAULT_OVERLAP'})
    text_chunker_use_semantic: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_USE_SEMANTIC'})
    text_chunker_use_hierarchical: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_USE_HIERARCHICAL'})
    text_chunker_adaptive: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_ADAPTIVE'})
    text_chunker_preserve_entities: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_PRESERVE_ENTITIES'})
    text_chunker_track_cross_refs: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_TRACK_CROSS_REFS'})
    text_chunker_multi_resolution: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_MULTI_RESOLUTION'})
    text_chunker_content_aware: bool = Field(True, json_schema_extra={'env': 'TEXT_CHUNKER_CONTENT_AWARE'})
    text_chunker_nlp_model: str = Field("en_core_web_sm", json_schema_extra={'env': 'TEXT_CHUNKER_NLP_MODEL'})
    text_chunker_min_shared_entities: int = Field(2, json_schema_extra={'env': 'TEXT_CHUNKER_MIN_SHARED_ENTITIES'})

    # Retrieval settings
    default_retrieval_limit: int = Field(5, json_schema_extra={'env': 'DEFAULT_RETRIEVAL_LIMIT'})
    retrieval_similarity_threshold: float = Field(0.6, json_schema_extra={'env': 'RETRIEVAL_SIMILARITY_THRESHOLD'})
    retrieval_max_related_depth: int = Field(2, json_schema_extra={'env': 'RETRIEVAL_MAX_RELATED_DEPTH'})

    # Schema settings
    schema_file_path: str = Field("./docs/schema.json", json_schema_extra={'env': 'SCHEMA_FILE_PATH'})
    schema_match_confidence_threshold: float = Field(0.75, json_schema_extra={'env': 'SCHEMA_MATCH_CONFIDENCE_THRESHOLD'})
    new_type_confidence_threshold: float = Field(0.85, json_schema_extra={'env': 'NEW_TYPE_CONFIDENCE_THRESHOLD'})
    default_entity_type: str = Field("Chunk", json_schema_extra={'env': 'DEFAULT_ENTITY_TYPE'})
    default_relationship_type: str = Field("RELATED_TO", json_schema_extra={'env': 'DEFAULT_RELATIONSHIP_TYPE'})
    tentative_entity_prefix: str = Field("Tentative", json_schema_extra={'env': 'TENTATIVE_ENTITY_PREFIX'})
    tentative_relationship_prefix: str = Field("TENTATIVE_", json_schema_extra={'env': 'TENTATIVE_RELATIONSHIP_PREFIX'})

    # Entity resolver settings
    entity_name_similarity_threshold: int = Field(85, json_schema_extra={'env': 'ENTITY_NAME_SIMILARITY_THRESHOLD'})
    entity_embedding_similarity_threshold: float = Field(0.92, json_schema_extra={'env': 'ENTITY_EMBEDDING_SIMILARITY_THRESHOLD'})
    entity_id_match_confidence: float = Field(1.0, json_schema_extra={'env': 'ENTITY_ID_MATCH_CONFIDENCE'})
    entity_merge_threshold: float = Field(0.9, json_schema_extra={'env': 'ENTITY_MERGE_THRESHOLD'})
    entity_batch_size: int = Field(100, json_schema_extra={'env': 'ENTITY_BATCH_SIZE'})
    entity_default_conflict_strategy: str = Field("keep_latest", json_schema_extra={'env': 'ENTITY_DEFAULT_CONFLICT_STRATEGY'})
    entity_string_similarity_method: str = Field("jaro_winkler", json_schema_extra={'env': 'ENTITY_STRING_SIMILARITY_METHOD'})
    entity_context_match_min_shared: int = Field(2, json_schema_extra={'env': 'ENTITY_CONTEXT_MATCH_MIN_SHARED'})
    entity_resolution_alias_threshold: int = Field(80, json_schema_extra={'env': 'ENTITY_RESOLUTION_ALIAS_THRESHOLD'})
    entity_resolution_embedding_threshold: float = Field(0.90, json_schema_extra={'env': 'ENTITY_RESOLUTION_EMBEDDING_THRESHOLD'})
    entity_resolution_property_threshold: float = Field(0.75, json_schema_extra={'env': 'ENTITY_RESOLUTION_PROPERTY_THRESHOLD'})
    entity_resolution_final_threshold: float = Field(0.85, json_schema_extra={'env': 'ENTITY_RESOLUTION_FINAL_THRESHOLD'})
    entity_resolution_candidate_limit: int = Field(10, json_schema_extra={'env': 'ENTITY_RESOLUTION_CANDIDATE_LIMIT'})
    entity_string_similarity_threshold: float = Field(0.8, json_schema_extra={'env': 'ENTITY_STRING_SIMILARITY_THRESHOLD'})
    min_shared_entities_for_relationship: int = Field(3, json_schema_extra={'env': 'MIN_SHARED_ENTITIES_FOR_RELATIONSHIP'})
    entity_resolution_merge_weight_name: float = Field(0.3, json_schema_extra={'env': 'ENTITY_RESOLUTION_MERGE_WEIGHT_NAME'})
    entity_resolution_merge_weight_alias: float = Field(0.15, json_schema_extra={'env': 'ENTITY_RESOLUTION_MERGE_WEIGHT_ALIAS'})
    entity_resolution_merge_weight_embedding: float = Field(0.55, json_schema_extra={'env': 'ENTITY_RESOLUTION_MERGE_WEIGHT_EMBEDDING'})
    entity_resolution_merge_weight_property: float = Field(0.2, json_schema_extra={'env': 'ENTITY_RESOLUTION_MERGE_WEIGHT_PROPERTY'})

    # Diagram settings
    diagram_detection_threshold: float = Field(0.6, json_schema_extra={'env': 'DIAGRAM_DETECTION_THRESHOLD'})

    # API settings
    host: str = Field("0.0.0.0", json_schema_extra={'env': 'HOST'})
    port: int = Field(8000, json_schema_extra={'env': 'PORT'})
    log_level: str = Field("INFO", json_schema_extra={'env': 'LOG_LEVEL'})
    allowed_origins: str = Field(
        "http://localhost,http://localhost:3000,http://localhost:8080",
        json_schema_extra={'env': 'ALLOWED_ORIGINS'}
    )
    api_key: Optional[str] = Field(None, json_schema_extra={'env': 'API_KEY'})

    # Constants
    KEEP_LATEST: str = "keep_latest"
    CONCATENATE: str = "concatenate"
    MERGE_ARRAYS: str = "merge_arrays"

    # Configure environment variable loading
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Create nested models for backward compatibility
    @property
    def neo4j(self) -> Neo4jSettings:
        return Neo4jSettings(
            uri=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            database=self.neo4j_database,
            vector_index_name=self.vector_index_name,
            vector_dimensions=self.vector_dimensions
        )

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings(
            provider=self.llm_provider,
            model_name=self.llm_model_name,
            temperature=self.llm_temperature,
            max_tokens=self.llm_max_tokens,
            max_context_length=self.llm_max_context_length,
            max_refinement_attempts=self.llm_max_refinement_attempts,
            openai_api_key=self.openai_api_key,
            openai_model=self.openai_model,
            google_api_key=self.google_api_key,
            google_model=self.google_model
        )

    @property
    def embedding(self) -> EmbeddingSettings:
        return EmbeddingSettings(
            provider=self.embedding_provider,
            vector_dimensions=self.vector_dimensions,
            batch_size=self.embedding_batch_size,
            max_retries=self.embedding_max_retries,
            retry_delay=self.embedding_retry_delay,
            openai_model=self.openai_embedding_model,
            google_model=self.google_embedding_model
        )

    @property
    def chunker(self) -> ChunkerSettings:
        return ChunkerSettings(
            default_chunk_size=self.text_chunker_default_chunk_size,
            default_overlap=self.text_chunker_default_overlap,
            use_semantic_boundaries=self.text_chunker_use_semantic,
            use_hierarchical_chunking=self.text_chunker_use_hierarchical,
            adaptive_chunking=self.text_chunker_adaptive,
            preserve_entities=self.text_chunker_preserve_entities,
            track_cross_references=self.text_chunker_track_cross_refs,
            enable_multi_resolution=self.text_chunker_multi_resolution,
            content_type_aware=self.text_chunker_content_aware,
            nlp_model=self.text_chunker_nlp_model,
            min_shared_entities=self.text_chunker_min_shared_entities
        )

    @property
    def retrieval(self) -> RetrievalSettings:
        return RetrievalSettings(
            default_limit=self.default_retrieval_limit,
            similarity_threshold=self.retrieval_similarity_threshold,
            max_related_depth=self.retrieval_max_related_depth
        )

    @property
    def schema(self) -> SchemaSettings:
        return SchemaSettings(
            file_path=self.schema_file_path,
            match_confidence_threshold=self.schema_match_confidence_threshold,
            new_type_confidence_threshold=self.new_type_confidence_threshold,
            default_entity_type=self.default_entity_type,
            default_relationship_type=self.default_relationship_type,
            tentative_entity_prefix=self.tentative_entity_prefix,
            tentative_relationship_prefix=self.tentative_relationship_prefix
        )

    @property
    def entity_resolver(self) -> EntityResolverSettings:
        return EntityResolverSettings(
            name_threshold=self.entity_name_similarity_threshold,
            alias_threshold=self.entity_resolution_alias_threshold,
            embedding_threshold=self.entity_resolution_embedding_threshold,
            property_threshold=self.entity_resolution_property_threshold,
            final_threshold=self.entity_resolution_final_threshold,
            candidate_limit=self.entity_resolution_candidate_limit,
            ignore_properties=["id", "created_at", "updated_at"],
            string_similarity_method=self.entity_string_similarity_method,
            merge_weight_name=self.entity_resolution_merge_weight_name,
            merge_weight_alias=self.entity_resolution_merge_weight_alias,
            merge_weight_embedding=self.entity_resolution_merge_weight_embedding,
            merge_weight_property=self.entity_resolution_merge_weight_property,
            batch_size=self.entity_batch_size,
            default_conflict_strategy=self.entity_default_conflict_strategy
        )

    @property
    def diagram(self) -> DiagramSettings:
        return DiagramSettings(
            detection_threshold=self.diagram_detection_threshold,
            default_prompts={
                'general': "Analyze this diagram and provide a detailed textual description. Focus on identifying the diagram type, key components, and the relationships or processes it illustrates. Be concise and informative.",
                'flowchart': "Analyze this flowchart and describe the process and steps involved. Focus on the sequence of actions, decision points, and overall workflow. Be detailed and clear.",
                'bar_chart': "Analyze this bar chart and describe the data being presented, including comparisons and trends. Focus on categories, values, axes, and any significant patterns. Be analytical and precise.",
                'financial': "Analyze this financial diagram and explain the financial concepts, trends, and metrics displayed. Focus on financial elements, indicators, and their implications. Be specific and use financial terminology.",
                'scientific': "Analyze this scientific diagram and explain the scientific principles, processes, and structures illustrated. Focus on scientific components, labels, and their functions. Be thorough and scientifically accurate.",
                'network_diagram': "Analyze this network diagram and describe the network topology, components, and connections. Focus on nodes, links, network structure, and data flow. Be comprehensive and technically detailed.",
                'organizational_chart': "Analyze this organizational chart and describe the hierarchical structure, roles, and relationships within the organization. Focus on reporting lines, departments, and key personnel. Be structured and focus on organizational aspects."
            }
        )

    @property
    def api(self) -> APISettings:
        return APISettings(
            host=self.host,
            port=self.port,
            log_level=self.log_level,
            allowed_origins=self.allowed_origins,
            api_key=self.api_key
        )


# Create the settings instance
settings = AppSettings()

# For backward compatibility with code that uses the old style settings
# These will be deprecated in future versions
NEO4J_URI = settings.neo4j_uri
NEO4J_USERNAME = settings.neo4j_username
NEO4J_PASSWORD = settings.neo4j_password
NEO4J_DATABASE = settings.neo4j_database
VECTOR_INDEX_NAME = settings.vector_index_name
VECTOR_DIMENSIONS = settings.vector_dimensions

OPENAI_API_KEY = settings.openai_api_key
DEFAULT_LLM_MODEL = settings.openai_model
DEFAULT_OPENAI_EMBEDDING_MODEL = settings.openai_embedding_model

GOOGLE_API_KEY = settings.google_api_key
DEFAULT_GOOGLE_LLM_MODEL = settings.google_model
DEFAULT_EMBEDDING_MODEL = settings.google_embedding_model
EMBEDDING_PROVIDER = settings.embedding_provider

CHUNK_SIZE = settings.text_chunker_default_chunk_size
CHUNK_OVERLAP = settings.text_chunker_default_overlap

DEFAULT_RETRIEVAL_LIMIT = settings.default_retrieval_limit
RETRIEVAL_SIMILARITY_THRESHOLD = settings.retrieval_similarity_threshold
RETRIEVAL_MAX_RELATED_DEPTH = settings.retrieval_max_related_depth

SCHEMA_FILE_PATH = settings.schema_file_path
SCHEMA_MATCH_CONFIDENCE_THRESHOLD = settings.schema_match_confidence_threshold
NEW_TYPE_CONFIDENCE_THRESHOLD = settings.new_type_confidence_threshold
DEFAULT_ENTITY_TYPE = settings.default_entity_type
DEFAULT_RELATIONSHIP_TYPE = settings.default_relationship_type
TENTATIVE_ENTITY_PREFIX = settings.tentative_entity_prefix
TENTATIVE_RELATIONSHIP_PREFIX = settings.tentative_relationship_prefix

TEXT_CHUNKER_DEFAULT_CHUNK_SIZE = settings.text_chunker_default_chunk_size
TEXT_CHUNKER_DEFAULT_OVERLAP = settings.text_chunker_default_overlap
TEXT_CHUNKER_USE_SEMANTIC = settings.text_chunker_use_semantic
TEXT_CHUNKER_USE_HIERARCHICAL = settings.text_chunker_use_hierarchical
TEXT_CHUNKER_ADAPTIVE = settings.text_chunker_adaptive
TEXT_CHUNKER_PRESERVE_ENTITIES = settings.text_chunker_preserve_entities
TEXT_CHUNKER_TRACK_CROSS_REFS = settings.text_chunker_track_cross_refs
TEXT_CHUNKER_MULTI_RESOLUTION = settings.text_chunker_multi_resolution
TEXT_CHUNKER_CONTENT_AWARE = settings.text_chunker_content_aware
TEXT_CHUNKER_NLP_MODEL = settings.text_chunker_nlp_model
TEXT_CHUNKER_MIN_SHARED_ENTITIES = settings.text_chunker_min_shared_entities

EMBEDDING_BATCH_SIZE = settings.embedding_batch_size
EMBEDDING_MAX_RETRIES = settings.embedding_max_retries
EMBEDDING_RETRY_DELAY = settings.embedding_retry_delay

LLM_MAX_CONTEXT_LENGTH = settings.llm_max_context_length
LLM_TEMPERATURE = settings.llm_temperature
LLM_MAX_TOKENS = settings.llm_max_tokens
LLM_PROVIDER = settings.llm_provider
LLM_MODEL_NAME = settings.llm_model_name
LLM_MAX_REFINEMENT_ATTEMPTS = settings.llm_max_refinement_attempts

ENTITY_NAME_SIMILARITY_THRESHOLD = settings.entity_name_similarity_threshold
ENTITY_EMBEDDING_SIMILARITY_THRESHOLD = settings.entity_embedding_similarity_threshold
ENTITY_ID_MATCH_CONFIDENCE = settings.entity_id_match_confidence
ENTITY_MERGE_THRESHOLD = settings.entity_merge_threshold
ENTITY_BATCH_SIZE = settings.entity_batch_size
ENTITY_DEFAULT_CONFLICT_STRATEGY = settings.entity_default_conflict_strategy
ENTITY_STRING_SIMILARITY_METHOD = settings.entity_string_similarity_method
ENTITY_CONTEXT_MATCH_MIN_SHARED = settings.entity_context_match_min_shared
ENTITY_RESOLUTION_ALIAS_THRESHOLD = settings.entity_resolution_alias_threshold
ENTITY_RESOLUTION_EMBEDDING_THRESHOLD = settings.entity_resolution_embedding_threshold
ENTITY_RESOLUTION_PROPERTY_THRESHOLD = settings.entity_resolution_property_threshold
ENTITY_RESOLUTION_FINAL_THRESHOLD = settings.entity_resolution_final_threshold
ENTITY_RESOLUTION_CANDIDATE_LIMIT = settings.entity_resolution_candidate_limit
ENTITY_RESOLUTION_IGNORE_PROPERTIES = ["id", "created_at", "updated_at"]  # Hardcoded for backward compatibility
ENTITY_RESOLUTION_STRING_SIMILARITY_METHOD = settings.entity_string_similarity_method
ENTITY_RESOLUTION_MERGE_WEIGHT_NAME = settings.entity_resolution_merge_weight_name
ENTITY_RESOLUTION_MERGE_WEIGHT_ALIAS = settings.entity_resolution_merge_weight_alias
ENTITY_RESOLUTION_MERGE_WEIGHT_EMBEDDING = settings.entity_resolution_merge_weight_embedding
ENTITY_RESOLUTION_MERGE_WEIGHT_PROPERTY = settings.entity_resolution_merge_weight_property
ENTITY_STRING_SIMILARITY_THRESHOLD = settings.entity_string_similarity_threshold
MIN_SHARED_ENTITIES_FOR_RELATIONSHIP = settings.min_shared_entities_for_relationship

KEEP_LATEST = settings.KEEP_LATEST
CONCATENATE = settings.CONCATENATE
MERGE_ARRAYS = settings.MERGE_ARRAYS

DIAGRAM_DETECTION_THRESHOLD = settings.diagram_detection_threshold
DEFAULT_DIAGRAM_PROMPTS = {
    'general': "Analyze this diagram and provide a detailed textual description. Focus on identifying the diagram type, key components, and the relationships or processes it illustrates. Be concise and informative.",
    'flowchart': "Analyze this flowchart and describe the process and steps involved. Focus on the sequence of actions, decision points, and overall workflow. Be detailed and clear.",
    'bar_chart': "Analyze this bar chart and describe the data being presented, including comparisons and trends. Focus on categories, values, axes, and any significant patterns. Be analytical and precise.",
    'financial': "Analyze this financial diagram and explain the financial concepts, trends, and metrics displayed. Focus on financial elements, indicators, and their implications. Be specific and use financial terminology.",
    'scientific': "Analyze this scientific diagram and explain the scientific principles, processes, and structures illustrated. Focus on scientific components, labels, and their functions. Be thorough and scientifically accurate.",
    'network_diagram': "Analyze this network diagram and describe the network topology, components, and connections. Focus on nodes, links, network structure, and data flow. Be comprehensive and technically detailed.",
    'organizational_chart': "Analyze this organizational chart and describe the hierarchical structure, roles, and relationships within the organization. Focus on reporting lines, departments, and key personnel. Be structured and focus on organizational aspects."
}

# spaCy Entity Type List (if used)
