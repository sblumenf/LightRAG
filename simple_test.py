"""
Simple test for the placeholder resolver.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the placeholder resolver
from lightrag.llm.placeholder_resolver import PlaceholderResolver

def main():
    """Run a simple test of the placeholder resolver."""
    print("Testing PlaceholderResolver...")
    
    # Create a resolver
    resolver = PlaceholderResolver()
    
    # Sample diagram data
    sample_diagrams = [
        {
            "diagram_id": "diagram-1",
            "page": 1,
            "caption": "System Architecture",
            "description": "A diagram showing the system architecture with components and connections",
            "diagram_type": "architecture"
        }
    ]
    
    # Sample formula data
    sample_formulas = [
        {
            "formula_id": "formula-1",
            "formula": "E = mc^2",
            "description": "Einstein's mass-energy equivalence formula",
            "latex": "E = mc^2"
        }
    ]
    
    # Sample extracted elements
    extracted_elements = {
        "diagrams": sample_diagrams,
        "formulas": sample_formulas
    }
    
    # Text with placeholders
    text = "This is a test with a diagram [DIAGRAM-diagram-1] and a formula [FORMULA-formula-1]."
    
    # Resolve placeholders
    result = resolver.resolve_placeholders(text, extracted_elements)
    
    # Print the result
    print("Original text:")
    print(text)
    print("\nResolved text:")
    print(result)
    
    # Check that placeholders were replaced
    if "[DIAGRAM-diagram-1]" not in result:
        print("✅ Diagram placeholder was replaced")
    else:
        print("❌ Diagram placeholder was not replaced")
    
    if "[FORMULA-formula-1]" not in result:
        print("✅ Formula placeholder was replaced")
    else:
        print("❌ Formula placeholder was not replaced")
    
    # Check that descriptions were added
    if "System Architecture" in result:
        print("✅ Diagram caption was added")
    else:
        print("❌ Diagram caption was not added")
    
    if "E = mc^2" in result:
        print("✅ Formula text was added")
    else:
        print("❌ Formula text was not added")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
