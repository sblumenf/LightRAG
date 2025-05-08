# Schema-Aware Knowledge Graph for LightRAG

This module provides a schema-aware implementation of the knowledge graph for LightRAG. It extends the existing Neo4J implementation with schema validation capabilities.

## Features

- **Schema Validation**: Validate entities and relationships against a schema before adding them to the graph
- **Tentative Entities and Relationships**: Handle entities and relationships that don't conform to the schema
- **Schema Statistics**: Get statistics about the schema and graph
- **Schema Violations**: Identify and fix schema violations
- **Schema-Aware Graph Creation**: Create a knowledge graph from classified chunks with schema awareness

## Schema Format

The schema is defined in a JSON file with the following structure:

```json
{
  "entities": [
    {
      "name": "Person",
      "properties": [
        {
          "name": "name",
          "type": "string",
          "required": true,
          "description": "The person's full name"
        },
        {
          "name": "age",
          "type": "integer",
          "description": "The person's age"
        }
      ]
    }
  ],
  "relationships": [
    {
      "name": "WORKS_FOR",
      "source": "Person",
      "target": "Organization",
      "properties": [
        {
          "name": "role",
          "type": "string",
          "description": "The person's role in the organization"
        }
      ]
    }
  ]
}
```

## Usage

### Initialization

```python
from lightrag.kg.schema_aware_neo4j import SchemaAwareNeo4JStorage

# Initialize schema-aware Neo4j storage
storage = SchemaAwareNeo4JStorage(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j",
    schema_path="path/to/schema.json"
)
```

### Creating Entities and Relationships

```python
# Create a valid entity
await storage.upsert_node(
    "person1",
    {
        "entity_type": "Person",
        "name": "John Doe",
        "age": 30
    }
)

# Create a valid relationship
await storage.upsert_edge(
    "person1",
    "org1",
    {
        "type": "WORKS_FOR",
        "role": "Developer"
    }
)
```

### Schema-Aware Graph Creation

```python
# Create schema-aware graph from classified chunks
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
                        "entity_id": "person1",
                        "entity_type": "Person",
                        "confidence": 0.85,
                        "text": "John Doe",
                        "properties": {
                            "name": "John Doe"
                        }
                    }
                ],
                "extracted_relationships": [
                    {
                        "source_id": "person1",
                        "target_id": "org1",
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
```

### Schema Statistics and Violations

```python
# Get schema statistics
stats = await storage.get_schema_statistics()

# Get schema violations
violations = await storage.get_schema_violations()

# Fix schema violations
fix_result = await storage.fix_schema_violations(auto_fix=True)
```

### Tentative Entities and Relationships

```python
# Get tentative entities
tentative_entities = await storage.get_tentative_entities()

# Get tentative relationships
tentative_relationships = await storage.get_tentative_relationships()

# Promote a tentative entity
await storage.promote_tentative_entity("entity_id")

# Promote a tentative relationship
await storage.promote_tentative_relationship("source_id", "target_id")
```

## Example

See the `examples/schema_aware_neo4j_example.py` file for a complete example of using the schema-aware Neo4j implementation.

## Testing

Run the tests with pytest:

```bash
pytest tests/kg/test_schema_aware_neo4j.py
```
