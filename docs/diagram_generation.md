# Diagram Description Generation

## Overview

LightRAG includes the capability to automatically extract and analyze diagrams from PDF documents, generating rich textual descriptions that can be indexed and used for retrieval. This document provides an overview of the diagram description generation system.

## Key Components

### 1. DiagramAnalyzer

The `DiagramAnalyzer` class in `document_processing/diagram_analyzer.py` is the core component responsible for:

- Extracting diagrams from PDF documents
- Determining which images are likely diagrams (versus photos or decorative elements)
- Generating textual descriptions using vision-capable LLMs
- Caching descriptions to improve performance

### 2. Vision Adapters

The vision adapter system in `lightrag/llm/vision_adapter.py` provides:

- A consistent interface for different vision-capable LLM providers (OpenAI, Anthropic)
- Automatic fallback between providers if one is unavailable
- Provider-specific parameter handling
- Unified description generation

### 3. Placeholder Resolver

The diagram descriptions are integrated into the main document text through the `PlaceholderResolver` class, which:

- Replaces diagram placeholders with detailed descriptions
- Formats descriptions appropriately for context
- Preserves metadata about diagram sources

## Features

### Diagram Detection

The system uses several heuristics to identify diagrams:

- **Image dimensions and aspect ratio**: Wide, rectangular images are more likely to be diagrams
- **Color diversity**: Diagrams typically use fewer colors than photographs
- **Edge density**: Diagrams have higher density of clean edges
- **Shape detection**: When OpenCV is available, shapes are detected and counted

### Context-Aware Description Generation

Descriptions are enhanced by contextual information:

- **Surrounding text**: Text near the diagram provides context for the LLM
- **Captions**: Associated figure captions are included in the prompt
- **Page information**: Page numbers help with document references

### Specialized Prompts by Diagram Type

The system includes specialized prompts for different diagram types, such as:

- Flowcharts and process diagrams
- Data visualizations (bar charts, line charts, pie charts, scatter plots)
- Technical diagrams (network, architecture, UML, ER diagrams)
- Organizational charts and concept maps

### Caching System

For performance and cost efficiency, the system includes:

- Persistent disk-based caching of generated descriptions
- Configurable expiration periods
- Cache cleanup for expired entries

### Fallback Mechanisms

Multiple fallback mechanisms ensure robustness:

- If vision APIs are unavailable, falls back to text-based LLM description
- If multiple vision APIs are configured, tries each in sequence
- If all LLM options fail, provides basic metadata description

## Configuration

The diagram analyzer can be configured through the LightRAG configuration system:

- `diagram_detection_threshold`: Sensitivity for diagram detection (default: 0.6)
- `vision_provider`: Preferred vision API provider ("auto", "openai", "anthropic", etc.)
- `vision_model`: Specific model to use for vision tasks
- `vision_api_key`: API key for vision service (falls back to main LLM API key if not provided)
- `vision_base_url`: Alternative base URL for vision API
- `enable_diagram_description_cache`: Whether to cache descriptions (default: true)
- `diagram_description_cache_expiry`: Time in seconds before cache entries expire
- `diagram_description_cache_dir`: Directory for cache storage

## Custom Prompts

You can customize the prompts used for different diagram types:

```python
from document_processing.diagram_analyzer import DiagramAnalyzer

analyzer = DiagramAnalyzer()

# Add a custom prompt for a specific diagram type
analyzer.add_description_template(
    "chemical_reaction", 
    """
    This chemical reaction diagram appears in educational material. Provide a detailed explanation that:
    
    1. Identifies the reactants and products
    2. Describes the reaction conditions
    3. Explains the reaction mechanism
    4. Identifies functional groups involved
    5. Describes any catalysts or intermediates
    6. Notes reaction type (e.g., substitution, addition)
    7. Explains the significance of this reaction
    
    Your description should be educational, technical, and suitable for chemistry students.
    """
)
```

## Testing

The diagram description system has extensive test coverage (over 70%) including:

- Unit tests for diagram detection and extraction
- Tests for caching functionality
- Tests for prompt handling
- Vision integration tests with mocked adapters

## Performance Considerations

- Diagram extraction is CPU-intensive, especially for large documents
- Vision API calls can add cost and latency
- The caching system helps mitigate these costs for repeated processing
- The system is designed for batch processing rather than real-time use

## Future Enhancements

Potential areas for future development:

- Support for additional vision-capable LLM providers
- Enhanced diagram classification system
- Interactive diagram exploration interface
- Support for vector-based diagram formats (SVG)
- Expanded diagram type recognition