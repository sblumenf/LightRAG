"""
Configs for the LightRAG API.
"""

import os
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Configure logging with both file and console output
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a timestamped log file
log_file = log_dir / f"lightrag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Root logger configuration
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger for this module
logger = logging.getLogger(__name__)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)
logger.info("Configuration loading started")


# Model server information removed


class DefaultRAGStorageConfig:
    KV_STORAGE = "JsonKVStorage"
    VECTOR_STORAGE = "NanoVectorDBStorage"
    GRAPH_STORAGE = "NetworkXStorage"
    DOC_STATUS_STORAGE = "JsonDocStatusStorage"


def get_default_host(binding_type: str) -> str:
    default_hosts = {
        "lollms": os.getenv("LLM_BINDING_HOST", "http://localhost:9600"),
        "azure_openai": os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.openai.com/v1"),
        "openai": os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
    }
    return default_hosts.get(
        binding_type, os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1")
    )  # fallback to OpenAI if unknown


def get_env_value(env_key: str, default: any, value_type: type = str) -> any:
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
        return default


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

    Args:
        is_uvicorn_mode: Whether running under uvicorn mode

    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="LightRAG FastAPI Server with separate working and input directories"
    )

    # Server configuration
    parser.add_argument(
        "--host",
        default=get_env_value("HOST", "0.0.0.0"),
        help="Server host (default: from env or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_env_value("PORT", 9621, int),
        help="Server port (default: from env or 9621)",
    )

    # Directory configuration
    parser.add_argument(
        "--working-dir",
        default=get_env_value("WORKING_DIR", "./rag_storage"),
        help="Working directory for RAG storage (default: from env or ./rag_storage)",
    )
    parser.add_argument(
        "--input-dir",
        default=get_env_value("INPUT_DIR", "./inputs"),
        help="Directory containing input documents (default: from env or ./inputs)",
    )

    def timeout_type(value):
        if value is None:
            return 150
        if value is None or value == "None":
            return None
        return int(value)

    parser.add_argument(
        "--timeout",
        default=get_env_value("TIMEOUT", None, timeout_type),
        type=timeout_type,
        help="Timeout in seconds (useful when using slow AI). Use None for infinite timeout",
    )

    # RAG configuration
    parser.add_argument(
        "--max-async",
        type=int,
        default=get_env_value("MAX_ASYNC", 4, int),
        help="Maximum async operations (default: from env or 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_env_value("MAX_TOKENS", 32768, int),
        help="Maximum token size (default: from env or 32768)",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        default=get_env_value("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: from env or INFO)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=get_env_value("VERBOSE", False, bool),
        help="Enable verbose debug output(only valid for DEBUG log-level)",
    )

    parser.add_argument(
        "--key",
        type=str,
        default=get_env_value("LIGHTRAG_API_KEY", None),
        help="API key for authentication. This protects lightrag server against unauthorized access",
    )
    parser.add_argument(
        "--rate-limit-requests",
        type=int,
        default=get_env_value("RATE_LIMIT_REQUESTS", 60, int),
        help="Maximum number of requests allowed per rate limit window",
    )
    parser.add_argument(
        "--rate-limit-window", 
        type=int,
        default=get_env_value("RATE_LIMIT_WINDOW", 60, int),
        help="Rate limit window in seconds",
    )
    parser.add_argument(
        "--trusted-networks",
        type=str,
        default=get_env_value("TRUSTED_NETWORKS", ""),
        help="Comma-separated list of trusted IP networks in CIDR notation (e.g., '10.0.0.0/8,192.168.1.0/24')",
    )

    # Optional https parameters
    parser.add_argument(
        "--ssl",
        action="store_true",
        default=get_env_value("SSL", False, bool),
        help="Enable HTTPS (default: from env or False)",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=get_env_value("SSL_CERTFILE", None),
        help="Path to SSL certificate file (required if --ssl is enabled)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=get_env_value("SSL_KEYFILE", None),
        help="Path to SSL private key file (required if --ssl is enabled)",
    )

    parser.add_argument(
        "--history-turns",
        type=int,
        default=get_env_value("HISTORY_TURNS", 3, int),
        help="Number of conversation history turns to include (default: from env or 3)",
    )

    # Search parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=get_env_value("TOP_K", 60, int),
        help="Number of most similar results to return (default: from env or 60)",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=get_env_value("COSINE_THRESHOLD", 0.2, float),
        help="Cosine similarity threshold (default: from env or 0.4)",
    )

    # Model name for simulation
    parser.add_argument(
        "--simulated-model-name",
        type=str,
        default=get_env_value(
            "SIMULATED_MODEL_NAME", "mistral-7b"
        ),
        help="Model name for simulation purposes",
    )

    # Namespace
    parser.add_argument(
        "--namespace-prefix",
        type=str,
        default=get_env_value("NAMESPACE_PREFIX", ""),
        help="Prefix of the namespace",
    )

    parser.add_argument(
        "--auto-scan-at-startup",
        action="store_true",
        default=False,
        help="Enable automatic scanning when the program starts",
    )

    # Server workers configuration
    parser.add_argument(
        "--workers",
        type=int,
        default=get_env_value("WORKERS", 1, int),
        help="Number of worker processes (default: from env or 1)",
    )

    # LLM and embedding bindings
    parser.add_argument(
        "--llm-binding",
        type=str,
        default=get_env_value("LLM_BINDING", "openai"),
        choices=["lollms", "openai", "azure_openai", "anthropic", "bedrock", "gemini", "hf", "jina", "lmdeploy", "nvidia_openai", "siliconcloud", "zhipu"],
        help="LLM binding type (default: from env or openai)",
    )
    parser.add_argument(
        "--embedding-binding",
        type=str,
        default=get_env_value("EMBEDDING_BINDING", "openai"),
        choices=["lollms", "openai", "azure_openai", "anthropic", "bedrock", "gemini", "hf", "jina", "lmdeploy", "nvidia_openai", "siliconcloud", "zhipu"],
        help="Embedding binding type (default: from env or openai)",
    )

    args = parser.parse_args()

    # convert relative path to absolute path
    args.working_dir = os.path.abspath(args.working_dir)
    args.input_dir = os.path.abspath(args.input_dir)

    # Inject storage configuration from environment variables
    args.kv_storage = get_env_value(
        "LIGHTRAG_KV_STORAGE", DefaultRAGStorageConfig.KV_STORAGE
    )
    args.doc_status_storage = get_env_value(
        "LIGHTRAG_DOC_STATUS_STORAGE", DefaultRAGStorageConfig.DOC_STATUS_STORAGE
    )
    args.graph_storage = get_env_value(
        "LIGHTRAG_GRAPH_STORAGE", DefaultRAGStorageConfig.GRAPH_STORAGE
    )
    args.vector_storage = get_env_value(
        "LIGHTRAG_VECTOR_STORAGE", DefaultRAGStorageConfig.VECTOR_STORAGE
    )

    # Get MAX_PARALLEL_INSERT from environment
    args.max_parallel_insert = get_env_value("MAX_PARALLEL_INSERT", 2, int)

    # Special case handling removed

    args.llm_binding_host = get_env_value(
        "LLM_BINDING_HOST", get_default_host(args.llm_binding)
    )
    args.embedding_binding_host = get_env_value(
        "EMBEDDING_BINDING_HOST", get_default_host(args.embedding_binding)
    )
    args.llm_binding_api_key = get_env_value("LLM_BINDING_API_KEY", None)
    args.embedding_binding_api_key = get_env_value("EMBEDDING_BINDING_API_KEY", "")

    # Inject model configuration
    args.llm_model = get_env_value("LLM_MODEL", "mistral-nemo:latest")
    args.embedding_model = get_env_value("EMBEDDING_MODEL", "bge-m3:latest")
    args.embedding_dim = get_env_value("EMBEDDING_DIM", 1024, int)
    args.max_embed_tokens = get_env_value("MAX_EMBED_TOKENS", 8192, int)

    # Enhanced embedding configuration
    args.enhanced_embedding_enabled = get_env_value("ENHANCED_EMBEDDING_ENABLED", False, bool)
    args.enhanced_embedding_provider = get_env_value("ENHANCED_EMBEDDING_PROVIDER", "openai")
    args.enhanced_embedding_batch_size = get_env_value("ENHANCED_EMBEDDING_BATCH_SIZE", 32, int)
    args.enhanced_embedding_max_retries = get_env_value("ENHANCED_EMBEDDING_MAX_RETRIES", 3, int)
    args.enhanced_embedding_retry_delay = get_env_value("ENHANCED_EMBEDDING_RETRY_DELAY", 1.0, float)
    args.enhanced_embedding_openai_model = get_env_value("ENHANCED_EMBEDDING_OPENAI_MODEL", "text-embedding-3-small")
    args.enhanced_embedding_google_model = get_env_value("ENHANCED_EMBEDDING_GOOGLE_MODEL", "models/embedding-001")

    # Inject chunk configuration
    args.chunk_size = get_env_value("CHUNK_SIZE", 1200, int)
    args.chunk_overlap_size = get_env_value("CHUNK_OVERLAP_SIZE", 100, int)

    # Inject LLM cache configuration
    args.enable_llm_cache_for_extract = get_env_value(
        "ENABLE_LLM_CACHE_FOR_EXTRACT", True, bool
    )
    args.enable_llm_cache = get_env_value("ENABLE_LLM_CACHE", True, bool)

    # Inject LLM temperature configuration
    args.temperature = get_env_value("TEMPERATURE", 0.5, float)

    # Select Document loading tool (DOCLING, DEFAULT)
    args.document_loading_engine = get_env_value("DOCUMENT_LOADING_ENGINE", "DEFAULT")

    # Add environment variables that were previously read directly
    args.cors_origins = get_env_value("CORS_ORIGINS", "*")
    args.summary_language = get_env_value("SUMMARY_LANGUAGE", "en")
    args.whitelist_paths = get_env_value("WHITELIST_PATHS", "/health,/api/*")

    # For JWT Auth
    args.auth_accounts = get_env_value("AUTH_ACCOUNTS", "")
    args.token_secret = get_env_value("TOKEN_SECRET", "lightrag-jwt-default-secret")
    args.token_expire_hours = get_env_value("TOKEN_EXPIRE_HOURS", 48, int)
    args.guest_token_expire_hours = get_env_value("GUEST_TOKEN_EXPIRE_HOURS", 24, int)
    args.jwt_algorithm = get_env_value("JWT_ALGORITHM", "HS256")

    # Removed Ollama model reference

    return args


def update_uvicorn_mode_config():
    # If in uvicorn mode and workers > 1, force it to 1 and log warning
    if global_args.workers > 1:
        original_workers = global_args.workers
        global_args.workers = 1
        # Log warning directly here
        logging.warning(
            f"In uvicorn mode, workers parameter was set to {original_workers}. Forcing workers=1"
        )


global_args = parse_args()
