# Schema Design Best Practices for Diagram Entity Extraction

This guide provides recommendations for designing schemas that work effectively with LightRAG's diagram entity extraction capability. A well-designed schema is essential for accurately extracting entities and relationships from diagrams.

## Introduction

The diagram entity extraction feature in LightRAG enables the system to transform visual information in diagrams into structured entities and relationships for the knowledge graph. An appropriate schema design ensures:

1. Accurate classification of diagram components
2. Proper relationship mapping
3. Consistent entity properties
4. Seamless integration with text-based entities

## Schema Structure Recommendations

### Entity Types

When designing entity types for diagrams, consider these recommendations:

#### 1. Include Diagram-Specific Entity Types

Include entity types that correspond to common diagram components:

```json
{
  "entities": [
    {
      "name": "Component",
      "description": "A functional component in an architecture or system diagram",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "functionality", "type": "string", "required": false},
        {"name": "technology", "type": "string", "required": false}
      ]
    },
    {
      "name": "Service",
      "description": "A service or API in a system architecture",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "endpoint", "type": "string", "required": false},
        {"name": "protocol", "type": "string", "required": false}
      ]
    },
    {
      "name": "DataStore",
      "description": "A data storage element in a system architecture",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "type", "type": "string", "required": false},
        {"name": "persistence", "type": "boolean", "required": false}
      ]
    },
    {
      "name": "Process",
      "description": "A process or action in a flowchart",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "description", "type": "string", "required": false},
        {"name": "duration", "type": "float", "required": false}
      ]
    },
    {
      "name": "Decision",
      "description": "A decision point in a flowchart",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "condition", "type": "string", "required": false}
      ]
    },
    {
      "name": "Class",
      "description": "A class in a UML diagram",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "attributes", "type": "string", "required": false},
        {"name": "methods", "type": "string", "required": false}
      ]
    }
  ]
}
```

#### 2. Align with Domain-Specific Entities

Ensure diagram entity types align with your domain-specific entity types:

- For software architecture: `Backend`, `Frontend`, `Database`, `API`
- For business processes: `Department`, `Role`, `Task`, `Milestone`
- For infrastructure: `Server`, `Network`, `Device`, `Cloud`

#### 3. Keep Property Structures Simple

For diagram entities, focus on essential properties:

- Prioritize string properties which are easier to extract
- Include required properties like `name` for all entity types
- Add domain-specific properties that can be extracted from context

### Relationship Types

When designing relationship types for diagrams, consider these recommendations:

#### 1. Include Diagram-Specific Relationship Types

```json
{
  "relationships": [
    {
      "name": "CONNECTS_TO",
      "source": "Component",
      "target": "Component",
      "description": "Indicates a connection between components",
      "properties": [
        {"name": "protocol", "type": "string", "required": false},
        {"name": "direction", "type": "string", "required": false}
      ]
    },
    {
      "name": "USES",
      "source": "Component",
      "target": "DataStore",
      "description": "Indicates a component uses a data store",
      "properties": []
    },
    {
      "name": "CALLS",
      "source": "Component",
      "target": "Service",
      "description": "Indicates a component calls a service",
      "properties": [
        {"name": "method", "type": "string", "required": false}
      ]
    },
    {
      "name": "NEXT",
      "source": "Process",
      "target": "Process",
      "description": "Indicates sequence in a flowchart",
      "properties": []
    },
    {
      "name": "NEXT_IF",
      "source": "Decision",
      "target": "Process",
      "description": "Indicates conditional flow in a flowchart",
      "properties": [
        {"name": "condition", "type": "string", "required": false}
      ]
    },
    {
      "name": "INHERITS_FROM",
      "source": "Class",
      "target": "Class",
      "description": "Indicates inheritance in a UML diagram",
      "properties": []
    }
  ]
}
```

#### 2. Support Bidirectional Relationships

Include relationship types that allow for bidirectional connections:

```json
{
  "relationships": [
    {
      "name": "INTERACTS_WITH",
      "source": "Component",
      "target": "Component",
      "description": "Indicates two-way interaction",
      "properties": [
        {"name": "type", "type": "string", "required": false}
      ]
    }
  ]
}
```

#### 3. Define Relationship Properties Sparingly

For diagram-derived relationships, keep properties minimal and focused on information typically found in diagrams:

- Connection types (e.g., "REST", "JDBC", "HTTP")
- Directions (e.g., "bidirectional", "one-way")
- Conditions (for flowcharts)

## Schema Design Strategies

### Strategy 1: Diagram-Type Specific Schemas

Create schema sections tailored to specific diagram types:

```json
{
  "entities": [
    // Architecture diagram entities
    {"name": "ApiGateway", "properties": [...]},
    {"name": "Microservice", "properties": [...]},
    
    // Flowchart entities
    {"name": "StartNode", "properties": [...]},
    {"name": "EndNode", "properties": [...]},
    
    // UML diagram entities
    {"name": "Interface", "properties": [...]},
    {"name": "Class", "properties": [...]}
  ]
}
```

### Strategy 2: Generic Schemas with Type Properties

Use generic entity types with type-specific properties:

```json
{
  "entities": [
    {
      "name": "DiagramNode",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "node_type", "type": "string", "required": true}, // e.g., "service", "database", "component"
        {"name": "description", "type": "string", "required": false},
        {"name": "technical_details", "type": "string", "required": false}
      ]
    }
  ]
}
```

### Strategy 3: Hierarchical Schemas

Use a hierarchical approach where specific types inherit from more general types:

```json
{
  "entities": [
    {
      "name": "SystemComponent",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "description", "type": "string", "required": false}
      ]
    },
    {
      "name": "Database",
      "parent": "SystemComponent",
      "properties": [
        {"name": "database_type", "type": "string", "required": false},
        {"name": "schema", "type": "string", "required": false}
      ]
    }
  ]
}
```

## Best Practices

### 1. Start Simple and Iterate

Begin with a basic schema covering the most common diagram entities and relationships, then refine based on extraction results.

### 2. Align with Text Extraction Schema

Ensure diagram entity types align with text-extracted entity types to enable seamless integration in the knowledge graph.

### 3. Use Clear, Descriptive Names

Choose entity and relationship type names that clearly communicate their purpose and are easy to map from diagram elements.

### 4. Focus on Structural Information

Diagrams excel at conveying structural information. Prioritize entity types and relationships that capture this structural nature.

### 5. Include Visual Properties when Relevant

Consider properties that capture visual aspects when they convey meaningful information:

```json
{
  "properties": [
    {"name": "position", "type": "string", "required": false}, // "top", "bottom", "center"
    {"name": "color", "type": "string", "required": false}, // May indicate status or importance
    {"name": "size", "type": "string", "required": false} // May indicate relative importance
  ]
}
```

### 6. Test with Representative Diagrams

Test your schema with a variety of diagram types typical in your domain to verify that it appropriately captures the relevant entities and relationships.

## Example: Complete Schema for Software Architecture Diagrams

Here's a complete example schema well-suited for software architecture diagrams:

```json
{
  "entities": [
    {
      "name": "Service",
      "description": "A service in a software architecture",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "responsibility", "type": "string", "required": false},
        {"name": "technology", "type": "string", "required": false},
        {"name": "version", "type": "string", "required": false},
        {"name": "status", "type": "string", "required": false}
      ]
    },
    {
      "name": "Database",
      "description": "A database in a software architecture",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "type", "type": "string", "required": false},
        {"name": "engine", "type": "string", "required": false},
        {"name": "schema_name", "type": "string", "required": false},
        {"name": "persistence", "type": "string", "required": false}
      ]
    },
    {
      "name": "UI",
      "description": "A user interface component",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "platform", "type": "string", "required": false},
        {"name": "technology", "type": "string", "required": false},
        {"name": "target_users", "type": "string", "required": false}
      ]
    },
    {
      "name": "ExternalSystem",
      "description": "An external system that interacts with the architecture",
      "properties": [
        {"name": "name", "type": "string", "required": true},
        {"name": "provider", "type": "string", "required": false},
        {"name": "interface_type", "type": "string", "required": false}
      ]
    }
  ],
  "relationships": [
    {
      "name": "CALLS",
      "source": "Service",
      "target": "Service",
      "description": "Indicates one service calls another",
      "properties": [
        {"name": "protocol", "type": "string", "required": false},
        {"name": "frequency", "type": "string", "required": false},
        {"name": "synchronous", "type": "boolean", "required": false}
      ]
    },
    {
      "name": "USES",
      "source": "Service",
      "target": "Database",
      "description": "Indicates a service uses a database",
      "properties": [
        {"name": "access_type", "type": "string", "required": false},
        {"name": "orm", "type": "string", "required": false}
      ]
    },
    {
      "name": "PRESENTS",
      "source": "UI",
      "target": "Service",
      "description": "Indicates a UI presents a service",
      "properties": [
        {"name": "view_type", "type": "string", "required": false}
      ]
    },
    {
      "name": "INTEGRATES_WITH",
      "source": "Service",
      "target": "ExternalSystem",
      "description": "Indicates a service integrates with an external system",
      "properties": [
        {"name": "integration_type", "type": "string", "required": false},
        {"name": "direction", "type": "string", "required": false}
      ]
    }
  ]
}
```

## Conclusion

A well-designed schema is crucial for effective diagram entity extraction. By following these best practices, you can ensure that LightRAG accurately extracts and integrates diagram entities into your knowledge graph, providing a more comprehensive representation of your documents' information.