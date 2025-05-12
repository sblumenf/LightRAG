# Diagram and Formula Integration in LightRAG

This document explains how LightRAG integrates diagrams and formulas into the retrieval and generation process.

## Overview

LightRAG can extract diagrams and formulas from documents during the document processing phase. These non-text elements are stored as placeholders in the text content, with their descriptions and metadata preserved. During the generation phase, these placeholders can be resolved and included in the context provided to the LLM, allowing for more comprehensive responses that reference diagrams and formulas.

## Features

- **Placeholder Resolution**: Automatically replaces diagram and formula placeholders with their descriptions
- **Enhanced Citation Handling**: Supports citing diagrams and formulas in LLM responses
- **Integrated with Chain-of-Thought**: Works seamlessly with CoT reasoning for better explanations
- **Configurable Output Formats**: Supports both detailed and concise formats for placeholder resolution

## Configuration

The diagram and formula integration can be configured through environment variables or directly in the LightRAG instance:

```python
from lightrag import LightRAG

# Configure through environment variables
# ENABLE_DIAGRAM_FORMULA_INTEGRATION=True
# RESOLVE_PLACEHOLDERS_IN_CONTEXT=True
# DIAGRAM_CITATION_FORMAT="[Diagram ID: {id}]"
# FORMULA_CITATION_FORMAT="[Formula ID: {id}]"
# PLACEHOLDER_OUTPUT_FORMAT="detailed"  # or "concise"

# Or configure directly in the LightRAG instance
lightrag = LightRAG(
    enable_diagram_formula_integration=True,
    resolve_placeholders_in_context=True,
    placeholder_output_format="detailed"  # or "concise"
)
```

## How It Works

### 1. Document Processing

During document processing, diagrams and formulas are extracted and stored with unique IDs:

```
Original text:
This is a diagram showing the system architecture.
[DIAGRAM-diagram-1]
The energy-mass equivalence is given by:
[FORMULA-formula-1]
```

### 2. Placeholder Resolution

When generating a response, the placeholders are resolved to include descriptions:

#### Detailed Format

```
Resolved text (detailed format):
This is a diagram showing the system architecture.
[DIAGRAM: diagram-1]
Caption: System Architecture Diagram
Description: A diagram showing the system architecture with components and connections
Type: architecture
Reference: [Diagram ID: diagram-1]

The energy-mass equivalence is given by:
[FORMULA: formula-1]
Formula: E = mc^2
Textual representation: Energy equals mass times the speed of light squared
Description: Einstein's mass-energy equivalence formula
LaTeX: E = mc^2
Reference: [Formula ID: formula-1]
```

#### Concise Format

```
Resolved text (concise format):
This is a diagram showing the system architecture.
[DIAGRAM: A diagram showing the system architecture with components and connections] [Diagram ID: diagram-1]

The energy-mass equivalence is given by:
[FORMULA: E = mc^2] [Formula ID: formula-1]
```

### 3. LLM Generation

The LLM is instructed to cite diagrams and formulas in its reasoning:

```
<reasoning>
Based on the system architecture shown in [Diagram ID: diagram-1],
we can see that the components are connected in a specific way.
The formula [Formula ID: formula-1] establishes the equivalence between energy and mass.
</reasoning>
<answer>
The system architecture has interconnected components, and the energy-mass equivalence
formula (E = mc^2) is fundamental to understanding the system's behavior.
</answer>
```

### 4. Citation Processing

Citations are processed to include numbered references and sources:

```
## Reasoning

Based on the system architecture shown in [1],
we can see that the components are connected in a specific way.
The formula [2] establishes the equivalence between energy and mass.

## Answer

The system architecture has interconnected components, and the energy-mass equivalence
formula (E = mc^2) is fundamental to understanding the system's behavior.

Sources:
1. Diagram: A diagram showing the system architecture with components and connections - System Architecture Diagram
2. Formula: E = mc^2 - Einstein's mass-energy equivalence formula
```

## API Reference

### PlaceholderResolver

The `PlaceholderResolver` class handles the resolution of placeholders in text content.

```python
from lightrag.llm.placeholder_resolver import PlaceholderResolver

resolver = PlaceholderResolver(output_format="detailed")  # or "concise"
resolved_text = resolver.resolve_placeholders(text, extracted_elements)
```

### resolve_placeholders_in_context

The `resolve_placeholders_in_context` function processes a list of context items, resolving any placeholders in their content.

```python
from lightrag.llm.placeholder_resolver import resolve_placeholders_in_context

resolved_items = resolve_placeholders_in_context(
    context_items,
    output_format="detailed"  # or "concise"
)
```

### process_citations

The `process_citations` function processes citations in a response, replacing them with numbered references and adding sources.

```python
from lightrag.llm.llm_generator import process_citations

processed_response = process_citations(response, context_items)
```

## Best Practices

1. **Enable Both Features**: For best results, enable both `enable_diagram_formula_integration` and `resolve_placeholders_in_context`.

2. **Choose the Right Format**: Use "detailed" format for comprehensive information or "concise" format for shorter context.

3. **Use Chain-of-Thought**: The diagram and formula integration works best with Chain-of-Thought reasoning enabled.

4. **Provide Clear Instructions**: When querying about diagrams or formulas, be specific in your query to help the LLM focus on the relevant elements.

5. **Check Extraction Quality**: Verify that diagrams and formulas are being correctly extracted during document processing.

## Troubleshooting

- **Placeholders Not Resolved**: Check that `resolve_placeholders_in_context` is enabled and that the extracted elements contain the correct diagram/formula IDs.

- **Citations Not Processed**: Ensure that the citation format matches the expected format (`[Diagram ID: X]` or `[Formula ID: X]`).

- **Missing Descriptions**: If diagram or formula descriptions are missing, they may not have been properly extracted during document processing.

## Example

```python
from lightrag import LightRAG

# Create a LightRAG instance with diagram/formula integration enabled
lightrag = LightRAG(
    enable_diagram_formula_integration=True,
    resolve_placeholders_in_context=True,
    placeholder_output_format="detailed",  # or "concise"
    enable_cot=True
)

# Add documents with diagrams and formulas
lightrag.add_documents(["document_with_diagrams_and_formulas.pdf"])

# Query about a diagram or formula
response = lightrag.query("Explain the system architecture diagram and how it relates to the energy-mass equivalence formula.")

print(response)
```

## Limitations

- The quality of diagram and formula integration depends on the quality of the extraction process.
- Very complex diagrams or formulas may not be fully captured in the textual descriptions.
- The LLM may not always correctly cite diagrams or formulas, especially if they are not directly relevant to the query.
