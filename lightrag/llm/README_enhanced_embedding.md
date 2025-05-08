# Enhanced Embedding System for LightRAG

This module provides an adapter for the robust EmbeddingGenerator implementation to work with LightRAG's EmbeddingFunc interface.

## Overview

The Enhanced Embedding System is designed to provide:

1. **Robust Error Handling**: Handles API errors, rate limits, and network issues with configurable retries and exponential backoff.
2. **Batching**: Processes texts in batches to optimize API usage and performance.
3. **Multi-Provider Support**: Supports both OpenAI and Google embedding providers through a consistent interface.
4. **Dimension Verification**: Verifies that the embedding dimension matches what's expected.
5. **Zero Vector Handling**: Returns zero vectors for empty or whitespace-only texts.

## Architecture

The system uses an adapter pattern to integrate the robust EmbeddingGenerator implementation with LightRAG's EmbeddingFunc interface:

```
┌───────────────────┐     ┌───────────────────────┐     ┌───────────────────┐
│                   │     │                       │     │                   │
│    LightRAG       │     │  EnhancedEmbedding    │     │  EmbeddingGenerator│
│    (Client)       │────▶│  Adapter             │────▶│  (Implementation)  │
│                   │     │                       │     │                   │
└───────────────────┘     └───────────────────────┘     └───────────────────┘
```

## Components

### EnhancedEmbeddingAdapter

The `EnhancedEmbeddingAdapter` class implements LightRAG's EmbeddingFunc interface and internally uses the EmbeddingGenerator class from the existing implementation.

### Factory Functions

The module provides factory functions for easy creation of the enhanced embedding adapter:

- `create_enhanced_embedding_func`: Generic factory function for creating an enhanced embedding adapter.
- `create_openai_enhanced_embedding`: Factory function for creating an OpenAI-specific enhanced embedding adapter.
- `create_google_enhanced_embedding`: Factory function for creating a Google-specific enhanced embedding adapter.

## Usage

### Basic Usage

```python
from lightrag.llm.enhanced_embedding import create_openai_enhanced_embedding
from lightrag import LightRAG

# Create an enhanced embedding function for OpenAI
embedding_func = create_openai_enhanced_embedding(
    model_name="text-embedding-3-small",
    embedding_dim=1536,
    max_token_size=8192,
)

# Create a LightRAG instance with the enhanced embedding function
rag = LightRAG(
    working_dir="./rag_storage",
    embedding_func=embedding_func,
    # ... other arguments
)
```

### Advanced Usage

```python
from lightrag.llm.enhanced_embedding import create_enhanced_embedding_func

# Create a custom enhanced embedding function
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
    },
)
```

## Configuration

The Enhanced Embedding System can be configured through environment variables or command-line arguments:

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `ENHANCED_EMBEDDING_ENABLED` | Enable the enhanced embedding system | `False` |
| `ENHANCED_EMBEDDING_PROVIDER` | The embedding provider to use (`openai` or `google`) | `openai` |
| `ENHANCED_EMBEDDING_BATCH_SIZE` | Number of texts to process in a single API call | `32` |
| `ENHANCED_EMBEDDING_MAX_RETRIES` | Maximum number of retries for API calls | `3` |
| `ENHANCED_EMBEDDING_RETRY_DELAY` | Delay between retries in seconds | `1.0` |
| `ENHANCED_EMBEDDING_OPENAI_MODEL` | The OpenAI embedding model to use | `text-embedding-3-small` |
| `ENHANCED_EMBEDDING_GOOGLE_MODEL` | The Google embedding model to use | `models/embedding-001` |

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
