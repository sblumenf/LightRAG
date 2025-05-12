"""
Simple test for citation handling.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the citation processor
from lightrag.llm.llm_generator import process_citations

def main():
    """Run a simple test of the citation processor."""
    print("Testing process_citations...")
    
    # Sample context items
    context_items = [
        {
            "id": "node1",
            "content": "This is content from node1."
        },
        {
            "id": "node2",
            "content": "This is content from node2 with a diagram.",
            "extracted_elements": {
                "diagrams": [
                    {
                        "diagram_id": "diagram-1",
                        "description": "System architecture diagram",
                        "caption": "Figure 1: System Architecture"
                    }
                ]
            }
        },
        {
            "id": "node3",
            "content": "This is content from node3 with a formula.",
            "extracted_elements": {
                "formulas": [
                    {
                        "formula_id": "formula-1",
                        "formula": "E = mc^2",
                        "description": "Einstein's mass-energy equivalence"
                    }
                ]
            }
        }
    ]
    
    # Response with citations
    response = """
    <reasoning>
    According to [Entity ID: node1], this is important. 
    [Diagram ID: diagram-1] illustrates the architecture.
    [Formula ID: formula-1] is used for calculations.
    </reasoning>
    <answer>
    Based on the information provided, the system architecture is shown in [Diagram ID: diagram-1]
    and calculations use [Formula ID: formula-1].
    </answer>
    """
    
    # Process citations
    processed_response = process_citations(response, context_items)
    
    # Print the results
    print("Original response:")
    print(response)
    print("\nProcessed response:")
    print(processed_response)
    
    # Check that citations were replaced
    if "[Entity ID: node1]" not in processed_response:
        print("✅ Entity citation was replaced")
    else:
        print("❌ Entity citation was not replaced")
    
    if "[Diagram ID: diagram-1]" not in processed_response:
        print("✅ Diagram citation was replaced")
    else:
        print("❌ Diagram citation was not replaced")
    
    if "[Formula ID: formula-1]" not in processed_response:
        print("✅ Formula citation was replaced")
    else:
        print("❌ Formula citation was not replaced")
    
    # Check that numbered references were added
    if "[1]" in processed_response:
        print("✅ Numbered reference [1] was added")
    else:
        print("❌ Numbered reference [1] was not added")
    
    if "[2]" in processed_response:
        print("✅ Numbered reference [2] was added")
    else:
        print("❌ Numbered reference [2] was not added")
    
    if "[3]" in processed_response:
        print("✅ Numbered reference [3] was added")
    else:
        print("❌ Numbered reference [3] was not added")
    
    # Check that sources were added
    if "Sources:" in processed_response:
        print("✅ Sources section was added")
    else:
        print("❌ Sources section was not added")
    
    if "System architecture diagram" in processed_response:
        print("✅ Diagram description was added to sources")
    else:
        print("❌ Diagram description was not added to sources")
    
    if "E = mc^2" in processed_response:
        print("✅ Formula text was added to sources")
    else:
        print("❌ Formula text was not added to sources")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
