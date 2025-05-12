"""
Example script for demonstrating diagram extraction and description generation.

This script extracts diagrams from a PDF document and generates descriptions
using the DiagramAnalyzer class with LLM vision capabilities.

Usage:
    python diagram_description_example.py [pdf_path]
"""

import os
import sys
import asyncio
import json
from pathlib import Path

# Add parent directory to path to allow importing from parent modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_processing.diagram_analyzer import DiagramAnalyzer
from lightrag.config_loader import get_enhanced_config


async def main():
    """Run the example."""
    # Get the PDF path from command line arguments or use a default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Try to find a test PDF in the fixtures directory
        fixtures_dir = Path(__file__).parent.parent / 'tests' / 'fixtures' / 'pdfs'
        if fixtures_dir.exists():
            pdfs = list(fixtures_dir.glob('*.pdf'))
            if pdfs:
                pdf_path = str(pdfs[0])
            else:
                print("No PDF files found in fixtures directory.")
                print("Please provide a PDF path as argument:")
                print("python diagram_description_example.py path/to/document.pdf")
                return
        else:
            print("Fixtures directory not found.")
            print("Please provide a PDF path as argument:")
            print("python diagram_description_example.py path/to/document.pdf")
            return
    
    # Check if the PDF exists
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print(f"Analyzing PDF: {pdf_path}")
    
    # Create a configuration
    config = {
        'diagram_detection_threshold': 0.5,
        'enable_diagram_description_cache': True,
        'vision_provider': 'auto',  # Try to find the best available provider
    }
    
    # Try to get enhanced config from LightRAG
    try:
        enhanced_config = get_enhanced_config()
        if enhanced_config:
            # Copy relevant settings
            if hasattr(enhanced_config, 'diagram_detection_threshold'):
                config['diagram_detection_threshold'] = enhanced_config.diagram_detection_threshold
                
            # Get API keys from enhanced config if available
            if hasattr(enhanced_config, 'openai_api_key'):
                config['vision_api_key'] = enhanced_config.openai_api_key
                config['vision_provider'] = 'openai'
                
            elif hasattr(enhanced_config, 'anthropic_api_key'):
                config['vision_api_key'] = enhanced_config.anthropic_api_key
                config['vision_provider'] = 'anthropic'
    except Exception as e:
        print(f"Warning: Unable to load LightRAG config: {e}")
        print("Continuing with default configuration.")

    # Create the diagram analyzer
    analyzer = DiagramAnalyzer(config=config)
    
    # Initialize the vision adapter
    adapter_initialized = await analyzer.initialize_vision_adapter()
    if adapter_initialized:
        print(f"Using vision provider: {analyzer.vision_adapter.provider_name}")
    else:
        print("Warning: No vision adapter available. Descriptions will be limited.")
    
    # Extract diagrams from the PDF
    print("Extracting diagrams from PDF...")
    diagrams = analyzer.extract_diagrams_from_pdf(pdf_path)
    
    if not diagrams:
        print("No diagrams found in the PDF.")
        return
    
    print(f"Found {len(diagrams)} diagram(s).")
    
    # Process each diagram
    results = []
    for i, diagram in enumerate(diagrams):
        print(f"\nProcessing diagram {i+1}/{len(diagrams)} (ID: {diagram['diagram_id']})...")
        
        # Determine the diagram type
        diagram_type = 'general'
        if diagram.get('caption'):
            caption = diagram['caption'].lower()
            if 'flow' in caption or 'process' in caption:
                diagram_type = 'flowchart'
            elif 'architecture' in caption:
                diagram_type = 'architecture_diagram'
            elif 'network' in caption:
                diagram_type = 'network_diagram'
            elif 'class' in caption or 'uml' in caption:
                diagram_type = 'uml_diagram'
            elif 'entity' in caption or 'er ' in caption:
                diagram_type = 'er_diagram'
            elif 'org' in caption or 'chart' in caption:
                diagram_type = 'organizational_chart'
        
        # Generate a description
        print(f"Generating description (type: {diagram_type})...")
        description = await analyzer.generate_diagram_description(diagram, diagram_type=diagram_type)
        
        # Update the diagram with the description
        diagram['description'] = description
        diagram['base64_data'] = diagram['base64_data'][:50] + '...'  # Truncate for display
        
        # Remove the full base64 data for the results (too large)
        result_diagram = {k: v for k, v in diagram.items() if k != '_full_base64'}
        results.append(result_diagram)
        
        print(f"Description ({len(description)} chars): {description[:100]}...")
    
    # Save the results to a JSON file
    output_path = 'diagram_extraction_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Clean up
    if analyzer.enable_caching:
        print(f"Description cache saved to {analyzer.cache_dir}")


if __name__ == "__main__":
    asyncio.run(main())