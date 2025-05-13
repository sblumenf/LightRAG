"""
Example script demonstrating diagram entity extraction functionality.

This example shows how to:
1. Extract diagrams from a PDF document
2. Generate textual descriptions for the diagrams
3. Extract entities and relationships from the diagrams
4. Add the extracted entities and relationships to a knowledge graph
"""

import os
import asyncio
import argparse
import logging
import json
from pathlib import Path

from document_processing.diagram_analyzer import DiagramAnalyzer
from lightrag.schema.schema_validator import SchemaValidator
from lightrag.kg.networkx_impl import NetworkxKnowledgeGraph

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory containing the example script
SCRIPT_DIR = Path(__file__).parent

# Default schema path
DEFAULT_SCHEMA_PATH = SCRIPT_DIR.parent / 'tests' / 'docs' / 'schema.json'


async def process_diagram(diagram, analyzer, schema_validator, llm_service):
    """Process a single diagram to extract entities and relationships."""
    logger.info(f"Processing diagram {diagram['diagram_id']} from page {diagram['page']}")
    
    # Generate description if needed
    if not diagram.get('description'):
        description = await analyzer.generate_diagram_description(diagram)
        diagram['description'] = description
        logger.info(f"Generated description: {description[:150]}...")
    else:
        logger.info(f"Using existing description: {diagram['description'][:150]}...")
    
    # Extract entities and relationships
    entities, relationships = await analyzer.extract_entities_and_relationships(
        diagram, schema_validator, llm_service
    )
    
    logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
    
    return entities, relationships


async def main(pdf_path, schema_path, output_dir=None):
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize schema validator
    schema_validator = SchemaValidator(schema_path)
    logger.info(f"Loaded schema with {len(schema_validator.get_entity_types())} entity types "
                f"and {len(schema_validator.get_relationship_types())} relationship types")
    
    # Simple async LLM function using OpenAI (replace with your preferred LLM provider)
    async def llm_service(prompt):
        try:
            import openai
            # Set your API key here or use environment variable
            # openai.api_key = "your_api_key_here"
            
            # Make API call
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts structured information from diagram descriptions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            # Extract text response
            return response.choices[0].message.content
            
        except ImportError:
            logger.warning("OpenAI package not available. Using mock response for demo purposes.")
            # Return mock response for demo purposes
            if "Extract named entities" in prompt:
                return """
                {
                  "entities": [
                    {
                      "entity_name": "UserInterface",
                      "entity_type": "Component",
                      "properties": {
                        "name": "User Interface",
                        "technology": "React"
                      },
                      "description": "Frontend user interface component"
                    },
                    {
                      "entity_name": "APIGateway",
                      "entity_type": "Service",
                      "properties": {
                        "name": "API Gateway",
                        "endpoint": "/api"
                      },
                      "description": "API Gateway for routing requests"
                    },
                    {
                      "entity_name": "Database",
                      "entity_type": "DataStore",
                      "properties": {
                        "name": "Database",
                        "type": "SQL"
                      },
                      "description": "SQL database for persistent storage"
                    }
                  ]
                }
                """
            elif "Extract relationships" in prompt:
                return """
                {
                  "relationships": [
                    {
                      "source": "UserInterface",
                      "target": "APIGateway",
                      "type": "CALLS",
                      "description": "UI makes HTTP requests to the API Gateway",
                      "properties": {
                        "protocol": "HTTP"
                      }
                    },
                    {
                      "source": "APIGateway",
                      "target": "Database",
                      "type": "QUERIES",
                      "description": "API Gateway queries the database",
                      "properties": {
                        "protocol": "SQL"
                      }
                    }
                  ]
                }
                """
            return "{}"
            
    # Initialize diagram analyzer
    analyzer = DiagramAnalyzer()
    
    # Extract diagrams from PDF
    diagrams = analyzer.extract_diagrams_from_pdf(pdf_path)
    logger.info(f"Extracted {len(diagrams)} diagrams from {pdf_path}")
    
    if not diagrams:
        logger.warning("No diagrams found in the PDF")
        return
    
    # Process each diagram
    all_entities = []
    all_relationships = []
    
    for diagram in diagrams:
        entities, relationships = await process_diagram(diagram, analyzer, schema_validator, llm_service)
        all_entities.extend(entities)
        all_relationships.extend(relationships)
    
    # Initialize a simple knowledge graph to store the entities and relationships
    kg = NetworkxKnowledgeGraph()
    
    # Add entities to the knowledge graph
    entity_ids = {}
    for entity in all_entities:
        entity_id = kg.add_entity(
            entity['entity_type'],
            entity['entity_name'],
            entity.get('properties', {}),
            entity.get('description', '')
        )
        entity_ids[entity['entity_name']] = entity_id
    
    # Add relationships to the knowledge graph
    for rel in all_relationships:
        source_name = rel['source']
        target_name = rel['target']
        
        if source_name in entity_ids and target_name in entity_ids:
            source_id = entity_ids[source_name]
            target_id = entity_ids[target_name]
            
            kg.add_relationship(
                source_id,
                target_id,
                rel['type'],
                rel.get('properties', {}),
                rel.get('description', '')
            )
    
    # Save results if output directory specified
    if output_dir:
        # Save diagrams with descriptions
        with open(os.path.join(output_dir, 'diagrams.json'), 'w') as f:
            # Create a serializable version of diagrams (remove base64 data)
            serializable_diagrams = []
            for diagram in diagrams:
                d = diagram.copy()
                if '_full_base64' in d:
                    del d['_full_base64']
                if 'base64_data' in d:
                    d['base64_data'] = d['base64_data'][:20] + '...'  # Truncate
                serializable_diagrams.append(d)
            json.dump(serializable_diagrams, f, indent=2)
            
        # Save extracted entities and relationships
        with open(os.path.join(output_dir, 'entities.json'), 'w') as f:
            json.dump(all_entities, f, indent=2)
            
        with open(os.path.join(output_dir, 'relationships.json'), 'w') as f:
            json.dump(all_relationships, f, indent=2)
        
        # Save knowledge graph visualization
        kg.visualize(os.path.join(output_dir, 'diagram_kg.html'))
        logger.info(f"Results saved to {output_dir}")
    
    # Print summary
    logger.info(f"Processed {len(diagrams)} diagrams")
    logger.info(f"Extracted {len(all_entities)} entities and {len(all_relationships)} relationships")
    logger.info(f"Added {kg.get_entity_count()} entities and {kg.get_relationship_count()} relationships to the knowledge graph")
    
    return diagrams, all_entities, all_relationships, kg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entities and relationships from diagrams in a PDF")
    parser.add_argument("pdf_path", help="Path to the PDF file containing diagrams")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Path to the schema JSON file")
    parser.add_argument("--output", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run the async main function
    asyncio.run(main(args.pdf_path, args.schema, args.output))