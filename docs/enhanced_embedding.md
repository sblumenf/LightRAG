# Enhanced Embedding System

The Enhanced Embedding System is a robust implementation for generating embeddings in LightRAG. It provides advanced error handling, batching, and multi-provider support.

## Features

- **Robust Error Handling**: Handles API errors, rate limits, and network issues with configurable retries and exponential backoff.
- **Batching**: Processes texts in batches to optimize API usage and performance.
- **Multi-Provider Support**: Supports both OpenAI and Google embedding providers through a consistent interface.
- **Dimension Verification**: Verifies that the embedding dimension matches what's expected.
- **Zero Vector Handling**: Returns zero vectors for empty or whitespace-only texts.

## Configuration

The Enhanced Embedding System can be configured through environment variables or command-line arguments:

| Environment Variable | Command-Line Argument | Description | Default |
|---------------------|------------------------|-------------|---------|
| `ENHANCED_EMBEDDING_ENABLED` | `--enhanced-embedding-enabled` | Enable the enhanced embedding system | `False` |
| `ENHANCED_EMBEDDING_PROVIDER` | `--enhanced-embedding-provider` | The embedding provider to use (`openai` or `google`) | `openai` |
| `ENHANCED_EMBEDDING_BATCH_SIZE` | `--enhanced-embedding-batch-size` | Number of texts to process in a single API call | `32` |
| `ENHANCED_EMBEDDING_MAX_RETRIES` | `--enhanced-embedding-max-retries` | Maximum number of retries for API calls | `3` |
| `ENHANCED_EMBEDDING_RETRY_DELAY` | `--enhanced-embedding-retry-delay` | Delay between retries in seconds | `1.0` |
| `ENHANCED_EMBEDDING_OPENAI_MODEL` | `--enhanced-embedding-openai-model` | The OpenAI embedding model to use | `text-embedding-3-small` |
| `ENHANCED_EMBEDDING_GOOGLE_MODEL` | `--enhanced-embedding-google-model` | The Google embedding model to use | `models/embedding-001` |

## Usage

### Using Environment Variables

```bash
# Enable the enhanced embedding system with OpenAI
export ENHANCED_EMBEDDING_ENABLED=true
export ENHANCED_EMBEDDING_PROVIDER=openai
export ENHANCED_EMBEDDING_OPENAI_MODEL=text-embedding-3-small
export OPENAI_API_KEY=your-api-key

# Start LightRAG
python -m lightrag.api.lightrag_server
```

### Using Command-Line Arguments

```bash
# Enable the enhanced embedding system with Google
python -m lightrag.api.lightrag_server \
  --enhanced-embedding-enabled \
  --enhanced-embedding-provider google \
  --enhanced-embedding-google-model models/embedding-001
```

## API Usage

If you're using LightRAG as a library, you can use the enhanced embedding system directly:

```python
from lightrag.llm.enhanced_embedding import create_openai_enhanced_embedding
from lightrag import LightRAG

# Create an enhanced embedding function for OpenAI
embedding_func = create_openai_enhanced_embedding(
    model_name="text-embedding-3-small",
    embedding_dim=1536,
    max_token_size=8192,
    batch_size=32,
    max_retries=3,
    retry_delay=1.0,
    api_key="your-api-key",  # Optional, defaults to OPENAI_API_KEY env var
)

# Create a LightRAG instance with the enhanced embedding function
rag = LightRAG(
    working_dir="./rag_storage",
    embedding_func=embedding_func,
    # ... other arguments
)
```

## Provider-Specific Factory Functions

The enhanced embedding system provides factory functions for common providers:

### OpenAI

```python
from lightrag.llm.enhanced_embedding import create_openai_enhanced_embedding

embedding_func = create_openai_enhanced_embedding(
    model_name="text-embedding-3-small",  # Default
    embedding_dim=1536,  # Default
    max_token_size=8192,  # Default
    batch_size=32,  # Default
    max_retries=3,  # Default
    retry_delay=1.0,  # Default
    api_key=None,  # Optional, defaults to OPENAI_API_KEY env var
)
```

### Google

```python
from lightrag.llm.enhanced_embedding import create_google_enhanced_embedding

embedding_func = create_google_enhanced_embedding(
    model_name="models/embedding-001",  # Default
    embedding_dim=768,  # Default
    max_token_size=8192,  # Default
    batch_size=32,  # Default
    max_retries=3,  # Default
    retry_delay=1.0,  # Default
    api_key=None,  # Optional, defaults to GOOGLE_API_KEY env var
)
```

## Advanced Configuration

For advanced configuration, you can use the `create_enhanced_embedding_func` function:

```python
from lightrag.llm.enhanced_embedding import create_enhanced_embedding_func

embedding_func = create_enhanced_embedding_func(
    embedding_dim=1536,
    max_token_size=8192,
    provider="openai",
    model_name="text-embedding-3-small",
    batch_size=32,
    max_retries=3,
    retry_delay=1.0,
    config={
        "llm": {
            "openai_api_key": "your-api-key",
        },
        # Other configuration options
    },
)
```

## Error Handling

The enhanced embedding system handles various error scenarios:

- **Rate Limits**: Retries with exponential backoff when rate limited.
- **Network Issues**: Retries when network issues occur.
- **API Errors**: Handles API errors gracefully.
- **Empty Texts**: Returns zero vectors for empty or whitespace-only texts.

## Performance Considerations

- **Batch Size**: Adjust the batch size based on your API limits and performance needs.
- **Retries**: Adjust the maximum retries and retry delay based on your API stability.
- **Dimension Verification**: The system verifies that the embedding dimension matches what's expected, which can help catch configuration issues.
