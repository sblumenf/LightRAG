# Diagram and Formula Integration Examples

This document provides examples of how to use the diagram and formula integration feature in LightRAG.

## Basic Usage

### Configuration

First, configure LightRAG to enable diagram and formula integration:

```python
from lightrag import LightRAG

# Create a LightRAG instance with diagram/formula integration enabled
lightrag = LightRAG(
    enable_diagram_formula_integration=True,
    resolve_placeholders_in_context=True,
    placeholder_output_format="detailed",  # or "concise"
    enable_cot=True  # Chain-of-Thought works best with diagrams and formulas
)
```

### Adding Documents with Diagrams and Formulas

Add documents that contain diagrams and formulas:

```python
# Add a PDF document that contains diagrams and formulas
lightrag.add_document("path/to/document_with_diagrams_and_formulas.pdf")

# Add multiple documents
lightrag.add_documents([
    "path/to/document1.pdf",
    "path/to/document2.pdf"
])
```

### Querying About Diagrams and Formulas

Query LightRAG about diagrams and formulas:

```python
# Query about a diagram
response = lightrag.query("Explain the system architecture diagram in the document.")

# Query about a formula
response = lightrag.query("What is the energy-mass equivalence formula and how is it used?")

# Query about both
response = lightrag.query("Explain how the system architecture relates to the energy-mass equivalence formula.")
```

## Advanced Usage

### Customizing Placeholder Resolution Format

You can choose between detailed and concise formats for placeholder resolution:

```python
# Detailed format (default)
lightrag = LightRAG(
    enable_diagram_formula_integration=True,
    resolve_placeholders_in_context=True,
    placeholder_output_format="detailed"
)

# Concise format
lightrag = LightRAG(
    enable_diagram_formula_integration=True,
    resolve_placeholders_in_context=True,
    placeholder_output_format="concise"
)
```

#### Detailed Format Example

The detailed format provides comprehensive information about diagrams and formulas:

```
[DIAGRAM: diagram-1]
Caption: System Architecture Diagram
Description: A diagram showing the system architecture with components and connections
Type: architecture
Reference: [Diagram ID: diagram-1]

[FORMULA: formula-1]
Formula: E = mc^2
Textual representation: Energy equals mass times the speed of light squared
Description: Einstein's mass-energy equivalence formula
LaTeX: E = mc^2
Reference: [Formula ID: formula-1]
```

#### Concise Format Example

The concise format provides minimal information about diagrams and formulas:

```
[DIAGRAM: A diagram showing the system architecture with components and connections] [Diagram ID: diagram-1]

[FORMULA: E = mc^2] [Formula ID: formula-1]
```

### Customizing Citation Formats

You can customize the citation formats for diagrams and formulas:

```python
# Through environment variables
# DIAGRAM_CITATION_FORMAT="[Diagram: {id}]"
# FORMULA_CITATION_FORMAT="[Formula: {id}]"

# Or directly in the LightRAG instance
lightrag = LightRAG(
    enable_diagram_formula_integration=True,
    resolve_placeholders_in_context=True,
    diagram_citation_format="[Diagram: {id}]",
    formula_citation_format="[Formula: {id}]"
)
```

## Integration with Chain-of-Thought

The diagram and formula integration works best with Chain-of-Thought reasoning enabled:

```python
lightrag = LightRAG(
    enable_diagram_formula_integration=True,
    resolve_placeholders_in_context=True,
    enable_cot=True
)

response = lightrag.query("Analyze the relationship between the system architecture and the energy-mass equivalence formula.")
```

Example response:

```
## Reasoning

Based on the system architecture diagram [1], I can see that the system has several interconnected components. The central hub connects to various modules, including a data processing unit and an energy conversion module.

The energy-mass equivalence formula [2] is fundamental to understanding how the energy conversion module works. According to E = mc², energy and mass are interchangeable, which explains the conversion process implemented in this module.

The diagram shows that energy flows from the conversion module to other components, demonstrating the practical application of the formula in the system design.

## Answer

The system architecture implements the energy-mass equivalence principle through its energy conversion module. This module, shown in the diagram, applies Einstein's E = mc² formula to convert between energy and mass states, enabling efficient energy distribution throughout the interconnected components.

Sources:
1. Diagram: A diagram showing the system architecture with components and connections - System Architecture Diagram
2. Formula: E = mc² - Einstein's mass-energy equivalence formula
```

## Programmatic Access to Placeholders

You can also access the placeholder resolution functionality programmatically:

```python
from lightrag.llm.placeholder_resolver import PlaceholderResolver, resolve_placeholders_in_context

# Create a resolver with the desired output format
resolver = PlaceholderResolver(output_format="detailed")  # or "concise"

# Resolve placeholders in text
text_with_placeholders = "This is a diagram: [DIAGRAM-diagram-1]"
extracted_elements = {
    "diagrams": [
        {
            "diagram_id": "diagram-1",
            "caption": "System Architecture",
            "description": "A diagram showing the system architecture",
            "page": "1",
            "diagram_type": "architecture"
        }
    ]
}
resolved_text = resolver.resolve_placeholders(text_with_placeholders, extracted_elements)

# Resolve placeholders in context items
context_items = [
    {
        "content": "This is a diagram: [DIAGRAM-diagram-1]",
        "extracted_elements": {
            "diagrams": [
                {
                    "diagram_id": "diagram-1",
                    "caption": "System Architecture",
                    "description": "A diagram showing the system architecture",
                    "page": "1",
                    "diagram_type": "architecture"
                }
            ]
        }
    }
]
resolved_items = resolve_placeholders_in_context(context_items, output_format="detailed")
```

## Best Practices

1. **Enable Both Features**: For best results, enable both `enable_diagram_formula_integration` and `resolve_placeholders_in_context`.

2. **Choose the Right Format**: Use "detailed" format for comprehensive information or "concise" format for shorter context.

3. **Use Chain-of-Thought**: The diagram and formula integration works best with Chain-of-Thought reasoning enabled.

4. **Be Specific in Queries**: When querying about diagrams or formulas, be specific to help the LLM focus on the relevant elements.

5. **Check Extraction Quality**: Verify that diagrams and formulas are being correctly extracted during document processing.

## Troubleshooting

- **Placeholders Not Resolved**: Check that `resolve_placeholders_in_context` is enabled and that the extracted elements contain the correct diagram/formula IDs.

- **Citations Not Processed**: Ensure that the citation format matches the expected format (`[Diagram ID: X]` or `[Formula ID: X]`).

- **Missing Descriptions**: If diagram or formula descriptions are missing, they may not have been properly extracted during document processing.

- **Poor Quality Descriptions**: If diagram descriptions are not helpful, consider adjusting the diagram detection threshold or using a different LLM provider for diagram description generation.
