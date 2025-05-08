# Text Chunking Module for LightRAG

This module provides enhanced text chunking functionality for LightRAG with support for multiple chunking strategies.

## Features

- **Multiple chunking strategies**: Choose between token-based or paragraph-based chunking
- **Metadata preservation**: Automatically includes document metadata in chunks
- **Extracted elements support**: Passes through placeholders for diagrams, formulas, etc.
- **Configurable via environment variables**: Easy to configure through `.env` file

## Configuration

The text chunking behavior can be configured through the following environment variables:

```
# Text chunking settings
CHUNKING_STRATEGY=token  # Options: 'token' or 'paragraph'
CHUNK_SIZE=1200          # Maximum tokens per chunk
CHUNK_OVERLAP_SIZE=100   # Overlap between chunks
```

## Chunking Strategies

### Token-based Chunking

The default strategy splits text based on token count, ensuring that each chunk has at most `CHUNK_SIZE` tokens with an overlap of `CHUNK_OVERLAP_SIZE` tokens between consecutive chunks.

This strategy is optimal for:
- Dense technical content
- Content where semantic boundaries are less important
- Maximum information density per chunk

### Paragraph-based Chunking

This strategy splits text by paragraphs (defined by blank lines), preserving the natural structure of the document. If a paragraph exceeds the maximum token size, it will be further split using token-based chunking.

This strategy is optimal for:
- Narrative content with clear paragraph structure
- Content where preserving semantic units is important
- Documents with natural breaks that should be preserved

## Usage

The chunking function is automatically used by LightRAG during document processing. You can also use it directly:

```python
from lightrag.text_chunker import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer

tokenizer = TiktokenTokenizer()
chunks = chunking_by_token_size(
    tokenizer=tokenizer,
    content="Your document text here...",
    chunking_strategy="paragraph",  # or "token"
    max_token_size=1200,
    overlap_token_size=100,
    file_path="document.txt",
    full_doc_id="doc-123"
)
```

## Output Format

The chunking function returns a list of dictionaries, where each dictionary contains:

- `tokens`: Number of tokens in the chunk
- `content`: Text content of the chunk
- `chunk_order_index`: Index of the chunk in the sequence
- `full_doc_id`: ID of the full document (if provided)
- `file_path`: Path to the source file (if provided)

## Integration with Document Processing

The chunking function is integrated into LightRAG's document processing pipeline and is called during document ingestion. The resulting chunks are stored in the vector database and used for retrieval during queries.
