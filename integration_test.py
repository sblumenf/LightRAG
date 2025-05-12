"""
Integration test for diagram and formula integration.

This script simulates the entire pipeline from document processing to response generation.
"""

import os
import sys
import json
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the necessary modules
from lightrag.llm.placeholder_resolver import PlaceholderResolver, resolve_placeholders_in_context
from lightrag.llm.llm_generator import process_citations
from lightrag.config_loader import get_enhanced_config, EnhancedConfig

# Mock the advanced generation manager
class MockAdvancedGenerationManager:
    """Mock implementation of the AdvancedGenerationManager."""
    
    def __init__(self):
        """Initialize the mock manager."""
        self.config = get_enhanced_config()
        self.resolver = PlaceholderResolver()
    
    def format_context(self, context_items: List[Dict[str, Any]]) -> str:
        """Format context items for the LLM."""
        formatted_items = []
        
        for i, item in enumerate(context_items, 1):
            # Vector store item
            content = item.get('content', 'No content')
            source = item.get('source', 'unknown')
            file_path = item.get('file_path', '')
            
            # Check for extracted elements
            has_diagrams = False
            has_formulas = False
            if 'extracted_elements' in item:
                if 'diagrams' in item['extracted_elements'] and item['extracted_elements']['diagrams']:
                    has_diagrams = True
                if 'formulas' in item['extracted_elements'] and item['extracted_elements']['formulas']:
                    has_formulas = True
            
            # Build the formatted item
            formatted_item = f"[{i}] Document ID: {item.get('id', 'unknown')}\n"
            
            if file_path:
                formatted_item += f"File: {file_path}\n"
            
            formatted_item += f"Source: {source}\n"
            
            # Add diagram/formula information if present
            if has_diagrams or has_formulas:
                elements_info = []
                if has_diagrams:
                    diagram_count = len(item['extracted_elements']['diagrams'])
                    elements_info.append(f"{diagram_count} diagram(s)")
                if has_formulas:
                    formula_count = len(item['extracted_elements']['formulas'])
                    elements_info.append(f"{formula_count} formula(s)")
                
                if elements_info:
                    formatted_item += f"Contains: {', '.join(elements_info)}\n"
            
            formatted_item += f"Content: {content}\n"
            formatted_items.append(formatted_item)
        
        return "\n".join(formatted_items)
    
    def generate_response(self, query: str, context_items: List[Dict[str, Any]]) -> str:
        """Generate a response using the context items."""
        # Resolve placeholders if enabled
        if self.config.enable_diagram_formula_integration and self.config.resolve_placeholders_in_context:
            print("Resolving placeholders in context...")
            resolved_items = resolve_placeholders_in_context(context_items)
        else:
            resolved_items = context_items
        
        # Format the context
        formatted_context = self.format_context(resolved_items)
        
        # Generate a mock response with citations
        response = f"""
        <reasoning>
        Based on the provided context, I can see that the document discusses system architecture and mathematical models.
        
        The system architecture is illustrated in [Diagram ID: diagram-1], which shows the components and connections of the system.
        
        The document also mentions Einstein's famous equation [Formula ID: formula-1], which establishes the equivalence between energy and mass.
        
        This formula, E = mc^2, is described as "fundamental to modern physics" [Entity ID: chunk-1].
        </reasoning>
        <answer>
        The document contains a system architecture diagram showing components and connections, and Einstein's energy-mass equivalence formula (E = mc^2), which is described as fundamental to modern physics.
        </answer>
        """
        
        # Process citations
        processed_response = process_citations(response, resolved_items)
        
        return processed_response

def main():
    """Run the integration test."""
    print("Running integration test for diagram and formula integration...")
    
    # Create a mock document with diagrams and formulas
    document_content = """
    # Test Document with Diagrams and Formulas
    
    This is a test document that contains diagrams and formulas.
    
    ## System Architecture
    
    [DIAGRAM-diagram-1]
    
    The system architecture consists of several components.
    
    ## Mathematical Model
    
    The energy-mass equivalence is given by:
    
    [FORMULA-formula-1]
    
    This formula is fundamental to modern physics.
    """
    
    # Create extracted elements
    extracted_elements = {
        "diagrams": [
            {
                "diagram_id": "diagram-1",
                "page": 1,
                "caption": "System Architecture Diagram",
                "description": "A diagram showing the system architecture with components and connections",
                "diagram_type": "architecture"
            }
        ],
        "formulas": [
            {
                "formula_id": "formula-1",
                "formula": "E = mc^2",
                "description": "Einstein's mass-energy equivalence formula",
                "latex": "E = mc^2"
            }
        ]
    }
    
    # Create context items
    context_items = [
        {
            "id": "chunk-1",
            "content": document_content,
            "extracted_elements": extracted_elements,
            "file_path": "test_document.txt",
            "source": "test"
        }
    ]
    
    # Create a mock advanced generation manager
    manager = MockAdvancedGenerationManager()
    
    # Generate a response
    query = "What does the document contain?"
    response = manager.generate_response(query, context_items)
    
    # Print the results
    print("\nQuery:")
    print(query)
    print("\nResponse:")
    print(response)
    
    # Check that the response contains the expected content
    if "[1]" in response and "[2]" in response and "[3]" in response:
        print("\n✅ Response contains numbered references")
    else:
        print("\n❌ Response does not contain numbered references")
    
    if "Sources:" in response:
        print("✅ Response contains sources section")
    else:
        print("❌ Response does not contain sources section")
    
    if "System Architecture Diagram" in response:
        print("✅ Response contains diagram information")
    else:
        print("❌ Response does not contain diagram information")
    
    if "E = mc^2" in response:
        print("✅ Response contains formula information")
    else:
        print("❌ Response does not contain formula information")
    
    print("\nIntegration test completed!")

if __name__ == "__main__":
    main()
