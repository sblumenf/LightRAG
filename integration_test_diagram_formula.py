"""
Simple integration test for diagram and formula integration.

This script demonstrates the placeholder resolver functionality for diagrams and formulas.

Usage:
    python integration_test_diagram_formula.py
"""

import asyncio
from lightrag.llm.placeholder_resolver import PlaceholderResolver, resolve_placeholders_in_context


async def test_placeholder_resolver():
    """Test the placeholder resolver functionality."""
    print("\n=== Testing PlaceholderResolver ===")

    # Test the placeholder resolver directly
    print("\n=== Testing with detailed format ===")
    resolver = PlaceholderResolver(output_format="detailed")
    text = "This is a diagram: [DIAGRAM-diagram-1] and a formula: [FORMULA-formula-1]"
    extracted_elements = {
        "diagrams": [
            {
                "diagram_id": "diagram-1",
                "caption": "System Architecture",
                "description": "A diagram showing the system architecture with components and connections",
                "page": "1",
                "diagram_type": "architecture"
            }
        ],
        "formulas": [
            {
                "formula_id": "formula-1",
                "formula": "E = mc^2",
                "textual_representation": "Energy equals mass times the speed of light squared",
                "description": "Einstein's mass-energy equivalence formula",
                "latex": "E = mc^2"
            }
        ]
    }

    # Resolve placeholders with detailed format
    resolved_text = resolver.resolve_placeholders(text, extracted_elements)
    print("\nOriginal text:")
    print(text)
    print("\nResolved text (detailed format):")
    print(resolved_text)

    # Test with concise format
    print("\n=== Testing with concise format ===")
    resolver = PlaceholderResolver(output_format="concise")
    resolved_text = resolver.resolve_placeholders(text, extracted_elements)
    print("\nOriginal text:")
    print(text)
    print("\nResolved text (concise format):")
    print(resolved_text)

    # Test resolve_placeholders_in_context
    print("\n=== Testing resolve_placeholders_in_context ===")
    context_items = [
        {
            "content": "This is a diagram: [DIAGRAM-diagram-1]",
            "extracted_elements": {
                "diagrams": [
                    {
                        "diagram_id": "diagram-1",
                        "caption": "System Architecture",
                        "description": "A diagram showing the system architecture with components and connections",
                        "page": "1",
                        "diagram_type": "architecture"
                    }
                ]
            }
        },
        {
            "content": "This is a formula: [FORMULA-formula-1]",
            "extracted_elements": {
                "formulas": [
                    {
                        "formula_id": "formula-1",
                        "formula": "E = mc^2",
                        "textual_representation": "Energy equals mass times the speed of light squared",
                        "description": "Einstein's mass-energy equivalence formula",
                        "latex": "E = mc^2"
                    }
                ]
            }
        }
    ]

    # Resolve placeholders in context items with detailed format
    resolved_items = resolve_placeholders_in_context(context_items, output_format="detailed")
    print("\nResolved context items (detailed format):")
    for i, item in enumerate(resolved_items):
        print(f"\nItem {i+1}:")
        print(item["content"])

    # Resolve placeholders in context items with concise format
    resolved_items = resolve_placeholders_in_context(context_items, output_format="concise")
    print("\nResolved context items (concise format):")
    for i, item in enumerate(resolved_items):
        print(f"\nItem {i+1}:")
        print(item["content"])



    print("\n=== Test completed successfully ===")


if __name__ == "__main__":
    asyncio.run(test_placeholder_resolver())
