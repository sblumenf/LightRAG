"""
Example script demonstrating the use of the schema-aware Neo4j implementation.

This script shows how to:
1. Initialize a schema-aware Neo4j storage
2. Create schema-aware nodes and edges
3. Validate entities and relationships against a schema
4. Handle tentative entities and relationships
5. Get schema statistics and violations
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any

# Add the parent directory to the path so we can import lightrag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.kg.schema_aware_neo4j import SchemaAwareNeo4JStorage


async def main():
    """Run the example."""
    # Initialize schema-aware Neo4j storage
    # Replace these with your Neo4j connection details
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    
    # Path to the schema file
    schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "lightrag", "kg", "sample_schema.json")
    
    print(f"Using schema file: {schema_path}")
    
    # Create schema-aware Neo4j storage
    storage = SchemaAwareNeo4JStorage(
        uri=uri,
        username=username,
        password=password,
        database=database,
        schema_path=schema_path
    )
    
    # Print loaded schema types
    print(f"Loaded entity types: {storage._entity_types_cache}")
    print(f"Loaded relationship types: {storage._relationship_types_cache}")
    
    # Create some valid entities
    print("\nCreating valid entities...")
    await storage.upsert_node(
        "person1",
        {
            "entity_type": "Person",
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "occupation": "Software Engineer"
        }
    )
    
    await storage.upsert_node(
        "org1",
        {
            "entity_type": "Organization",
            "name": "Acme Inc.",
            "industry": "Technology",
            "founded": 2000,
            "size": "medium"
        }
    )
    
    await storage.upsert_node(
        "doc1",
        {
            "entity_type": "Document",
            "title": "Annual Report",
            "content": "This is the annual report for Acme Inc.",
            "date": "2023-01-01",
            "author": "John Doe"
        }
    )
    
    # Create a valid relationship
    print("Creating valid relationship...")
    await storage.upsert_edge(
        "person1",
        "org1",
        {
            "type": "WORKS_FOR",
            "role": "Developer",
            "since": 2020
        }
    )
    
    # Create an invalid entity (missing required property)
    print("\nCreating invalid entity (missing required property)...")
    await storage.upsert_node(
        "person2",
        {
            "entity_type": "Person",
            "age": 25,
            "email": "invalid@example.com"
            # Missing required 'name' property
        }
    )
    
    # Create an invalid relationship (wrong direction)
    print("Creating invalid relationship (wrong direction)...")
    await storage.upsert_edge(
        "doc1",  # Should be Person -> Document, not Document -> Person
        "person1",
        {
            "type": "AUTHORED",
            "date": "2023-01-01"
        }
    )
    
    # Create a new entity type not in the schema
    print("\nCreating new entity type not in schema...")
    await storage.upsert_node(
        "project1",
        {
            "entity_type": "Project",  # Not in schema
            "name": "New Project",
            "description": "A new project",
            "status": "active"
        }
    )
    
    # Get schema statistics
    print("\nGetting schema statistics...")
    stats = await storage.get_schema_statistics()
    print(json.dumps(stats, indent=2))
    
    # Get schema violations
    print("\nGetting schema violations...")
    violations = await storage.get_schema_violations()
    print(json.dumps(violations, indent=2))
    
    # Fix schema violations
    print("\nFixing schema violations...")
    fix_result = await storage.fix_schema_violations(auto_fix=True)
    print(json.dumps(fix_result, indent=2))
    
    # Get tentative entities
    print("\nGetting tentative entities...")
    tentative_entities = await storage.get_tentative_entities()
    print(json.dumps(tentative_entities, indent=2))
    
    # Get tentative relationships
    print("\nGetting tentative relationships...")
    tentative_relationships = await storage.get_tentative_relationships()
    print(json.dumps(tentative_relationships, indent=2))
    
    # Promote a tentative entity
    if tentative_entities:
        print(f"\nPromoting tentative entity: {tentative_entities[0]['entity_id']}...")
        result = await storage.promote_tentative_entity(tentative_entities[0]["entity_id"])
        print(f"Promotion result: {result}")
    
    # Promote a tentative relationship
    if tentative_relationships:
        source_id = tentative_relationships[0]["source_id"]
        target_id = tentative_relationships[0]["target_id"]
        print(f"\nPromoting tentative relationship: {source_id} -> {target_id}...")
        result = await storage.promote_tentative_relationship(source_id, target_id)
        print(f"Promotion result: {result}")
    
    # Create schema-aware graph from classified chunks
    print("\nCreating schema-aware graph from classified chunks...")
    chunks = [
        {
            "chunk_id": "chunk1",
            "content": "John Doe works for Acme Inc.",
            "metadata": {
                "schema_classification": {
                    "entity_type": "Document",
                    "confidence": 0.9,
                    "properties": {
                        "title": "Employee Record",
                        "content": "John Doe works for Acme Inc."
                    },
                    "extracted_entities": [
                        {
                            "entity_id": "person3",
                            "entity_type": "Person",
                            "confidence": 0.85,
                            "text": "John Smith",
                            "properties": {
                                "name": "John Smith"
                            }
                        },
                        {
                            "entity_id": "org2",
                            "entity_type": "Organization",
                            "confidence": 0.8,
                            "text": "Tech Corp",
                            "properties": {
                                "name": "Tech Corp"
                            }
                        }
                    ],
                    "extracted_relationships": [
                        {
                            "source_id": "person3",
                            "target_id": "org2",
                            "relationship_type": "WORKS_FOR",
                            "confidence": 0.75,
                            "properties": {
                                "role": "Employee"
                            }
                        }
                    ]
                }
            }
        }
    ]
    
    result = await storage.create_schema_aware_graph(chunks)
    print(json.dumps(result, indent=2))
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
