# Product Requirements Document: Extracting Entities and Relationships from Diagrams for Knowledge Graph Integration

## 1. Overview

This PRD outlines the enhancement of LightRAG to extract entities and relationships from diagrams for seamless integration into the knowledge graph. This feature will enable LightRAG to create a more comprehensive knowledge graph by incorporating structured information from diagrams alongside text-based content.

## 2. Background

LightRAG currently has strong capabilities for extracting diagrams from documents and generating textual descriptions using vision models. It also has robust entity and relationship extraction from text. However, it lacks the ability to extract structured entities and relationships directly from diagrams for knowledge graph integration.

## 3. Business Requirements

- Enable extraction of entities and relationships from diagrams found in documents
- Integrate diagram-extracted entities into the existing knowledge graph
- Use the existing schema for entity and relationship classification
- Focus on extracting major components rather than fine details
- Maintain consistency with text-based entity extraction

## 4. Technical Specifications

### 4.1 Enhanced Diagram Analyzer

#### Current Capabilities:
- Extract diagrams from PDFs
- Classify diagram types
- Generate textual descriptions
- Cache processed results

#### New Capabilities:
- Extract structured entities from diagram descriptions
- Identify relationships between diagram entities
- Map extracted information to schema entity types
- Link diagram entities to the knowledge graph

### 4.2 Diagram Entity Extraction Module

```python
class DiagramEntityExtractor:
    """
    Extract entities and relationships from diagram descriptions for knowledge graph integration.
    """
    
    def __init__(self, schema_validator, llm_func, config=None):
        """
        Initialize the diagram entity extractor.
        
        Args:
            schema_validator: SchemaValidator instance with loaded schema
            llm_func: Async function to call the LLM
            config: Optional configuration dictionary
        """
        self.schema_validator = schema_validator
        self.llm_func = llm_func
        self.config = config or {}
        
    async def extract_entities_from_diagram(self, diagram_data):
        """
        Extract entities from a diagram description.
        
        Args:
            diagram_data: Dictionary containing diagram information
                including description, caption, and metadata
                
        Returns:
            List of extracted entities with schema types
        """
        # Implementation logic will go here
        
    async def extract_relationships_from_diagram(self, diagram_data, entities):
        """
        Extract relationships between entities in a diagram.
        
        Args:
            diagram_data: Dictionary containing diagram information
            entities: List of extracted entities
            
        Returns:
            List of extracted relationships between entities
        """
        # Implementation logic will go here
```

### 4.3 Integration with Existing Schema

The diagram entity extraction will use the existing schema validation system:

1. The `SchemaValidator` will be used to validate diagram-extracted entities and relationships
2. The system will map diagram components to schema entity types
3. Entity properties will be populated based on available information
4. Relationships will be validated against schema relationship types

### 4.4 Prompt Template for Entity Extraction

```
You are an expert in extracting structured entities from diagram descriptions.

TASK:
Extract named entities from the following diagram description according to the schema provided.

DIAGRAM DESCRIPTION:
```
{diagram_description}
```

DIAGRAM CAPTION:
{caption}

DIAGRAM TYPE:
{diagram_type}

SCHEMA ENTITY TYPES:
{entity_types}

INSTRUCTIONS:
1. Identify distinct entities mentioned in the diagram description
2. Classify each entity according to the schema entity types
3. Extract relevant properties for each entity
4. For each entity, provide:
   - entity_name: A unique name for the entity
   - entity_type: The schema type that best matches
   - properties: Key-value pairs of relevant properties
   - description: A brief description of the entity

FORMAT YOUR RESPONSE AS JSON:
{
  "entities": [
    {
      "entity_name": "EntityName1",
      "entity_type": "SchemaType1",
      "properties": {
        "property1": "value1",
        "property2": "value2"
      },
      "description": "Brief description of entity"
    },
    ...
  ]
}

IMPORTANT:
- Focus on major components only
- Use consistent entity naming
- Only use entity types from the provided schema
- If you're uncertain about an entity's type, use the most probable one
- Return ONLY the JSON object
```

### 4.5 Prompt Template for Relationship Extraction

```
You are an expert in extracting structured relationships from diagram descriptions.

TASK:
Extract relationships between entities from the following diagram description according to the schema provided.

DIAGRAM DESCRIPTION:
```
{diagram_description}
```

DIAGRAM CAPTION:
{caption}

DIAGRAM TYPE:
{diagram_type}

ENTITIES:
{entities_json}

RELATIONSHIP TYPES:
{relationship_types}

RELATIONSHIP DEFINITIONS:
{relationship_definitions}

INSTRUCTIONS:
1. Identify relationships between the extracted entities
2. Classify each relationship according to the schema relationship types
3. For each relationship, provide:
   - source: The name of the source entity
   - target: The name of the target entity
   - type: The schema relationship type
   - description: A brief description of the relationship
   - properties: Any relevant properties for the relationship

FORMAT YOUR RESPONSE AS JSON:
{
  "relationships": [
    {
      "source": "EntityName1",
      "target": "EntityName2",
      "type": "RELATIONSHIP_TYPE",
      "description": "Brief description of relationship",
      "properties": {
        "property1": "value1"
      }
    },
    ...
  ]
}

IMPORTANT:
- Only create relationships between entities in the provided entity list
- Only use relationship types from the provided schema
- Focus on semantic relationships, not just visual connections
- Return ONLY the JSON object
```

### 4.6 Integration with DiagramAnalyzer

The current `DiagramAnalyzer` class will be enhanced to integrate the entity and relationship extraction:

```python
class DiagramAnalyzer:
    # Existing methods...
    
    async def extract_entities_and_relationships(self, diagram_data, schema_validator, llm_func):
        """
        Extract entities and relationships from a diagram for knowledge graph integration.
        
        Args:
            diagram_data: Dictionary containing diagram information
            schema_validator: SchemaValidator instance
            llm_func: LLM function for entity/relationship extraction
            
        Returns:
            Tuple of (entities, relationships)
        """
        # Get or generate diagram description
        description = diagram_data.get('description')
        if not description:
            description = await self.generate_diagram_description(diagram_data)
            diagram_data['description'] = description
            
        # Create diagram entity extractor
        extractor = DiagramEntityExtractor(schema_validator, llm_func)
        
        # Extract entities
        entities = await extractor.extract_entities_from_diagram(diagram_data)
        
        # Extract relationships
        relationships = await extractor.extract_relationships_from_diagram(
            diagram_data, entities
        )
        
        return entities, relationships
```

### 4.7 Pipeline Integration

The diagram entity extraction will be integrated into the existing document processing pipeline:

1. The document processing module extracts diagrams from documents
2. The diagram analyzer generates descriptions for each diagram
3. The diagram entity extractor identifies entities and relationships
4. The extracted entities and relationships are validated against the schema
5. The validated entities and relationships are added to the knowledge graph

### 4.8 Caching Strategy

To avoid redundant processing:

1. Cache diagram descriptions (already implemented)
2. Cache extracted entities and relationships by diagram ID
3. Invalidate cache when the diagram or schema changes
4. Use LLM response caching for improved performance

## 5. Implementation Plan

The following steps outline the implementation process for this enhancement:

### Phase 1: Core Implementation

- [x] Create the `DiagramEntityExtractor` class
- [x] Implement entity extraction from diagram descriptions
- [x] Implement relationship extraction from diagram descriptions
- [x] Add unit tests for the extractor

### Phase 2: Integration

- [x] Enhance the `DiagramAnalyzer` class to use the entity extractor
- [x] Implement caching for extracted entities and relationships
- [ ] Integrate with the document processing pipeline
- [ ] Add integration tests for the complete pipeline

### Phase 3: Refinement

- [x] Optimize prompts for better entity and relationship extraction
- [x] Implement confidence scoring for extracted entities and relationships
- [x] Add filtering for low-confidence extractions
- [ ] Create evaluation metrics to measure extraction quality

### Phase 4: Documentation and Examples

- [x] Update API documentation
- [x] Create examples demonstrating diagram entity extraction
- [ ] Add usage instructions to the README
- [ ] Document best practices for schema design

## 6. Test Cases

### 6.1 Unit Tests

1. **Entity Extraction Tests**
   - Test extracting entities from different diagram types
   - Test handling of empty or invalid descriptions
   - Test mapping to different schema entity types
   - Test extraction with various diagram complexities

2. **Relationship Extraction Tests**
   - Test extracting relationships between entities
   - Test validation against schema relationship types
   - Test handling of invalid relationships
   - Test relationship property extraction

3. **Integration Tests**
   - Test the full pipeline from diagram extraction to knowledge graph integration
   - Test with sample PDFs containing different diagram types
   - Test caching behavior with repeated extractions
   - Test error handling and fallback mechanisms

### 6.2 Test Fixtures

1. **Sample Diagrams**
   - Flowcharts with process steps
   - Architecture diagrams with components
   - UML diagrams with classes and relationships
   - Network diagrams with connected nodes

2. **Expected Results**
   - For each sample diagram, define expected:
     - Number of entities
     - Entity types
     - Key relationships
     - Essential properties

## 7. Success Metrics

1. **Extraction Quality**
   - Precision: Percentage of correctly extracted entities and relationships
   - Recall: Percentage of diagram entities successfully extracted
   - F1 Score: Harmonic mean of precision and recall

2. **Performance Metrics**
   - Processing time per diagram
   - Caching effectiveness
   - Token usage for LLM calls

3. **Integration Quality**
   - Knowledge graph coherence with diagram-extracted entities
   - Query effectiveness with diagram information

## 8. Future Enhancements

These items are out of scope for the current implementation but may be considered for future updates:

1. Visual positioning data for entities (for spatial querying)
2. Diagram versioning and change tracking
3. Cross-diagram entity resolution
4. Interactive diagram visualization with knowledge graph overlay
5. User feedback mechanism for extraction corrections

## 9. API Reference

### 9.1 DiagramEntityExtractor

```python
# Initialize extractor
extractor = DiagramEntityExtractor(schema_validator, llm_func, config)

# Extract entities from a diagram
entities = await extractor.extract_entities_from_diagram(diagram_data)

# Extract relationships between entities
relationships = await extractor.extract_relationships_from_diagram(diagram_data, entities)
```

### 9.2 Enhanced DiagramAnalyzer

```python
# Initialize analyzer
analyzer = DiagramAnalyzer(config)

# Extract diagrams from PDF
diagrams = analyzer.extract_diagrams_from_pdf(pdf_path)

# Generate description for a diagram
description = await analyzer.generate_diagram_description(diagram_data)

# Extract entities and relationships
entities, relationships = await analyzer.extract_entities_and_relationships(
    diagram_data, schema_validator, llm_func
)
```

## 10. Implementation Checklist

- [x] Create DiagramEntityExtractor class
- [x] Implement entity extraction from diagram descriptions
- [x] Implement relationship extraction from diagram descriptions
- [x] Enhance DiagramAnalyzer with entity and relationship extraction
- [x] Implement caching for extracted entities and relationships
- [ ] Integrate with document processing pipeline
- [x] Create unit tests for entity extraction
- [x] Create unit tests for relationship extraction
- [ ] Create integration tests for full pipeline
- [ ] Implement evaluation metrics
- [x] Update documentation and examples
- [ ] Create sample diagrams for testing
- [x] Optimize prompts for better extraction quality
- [x] Add error handling and fallback mechanisms
- [x] Implement confidence scoring for extractions
- [x] Add filtering for low-confidence extractions

## 11. Technical Approach

### 11.1 DiagramEntityExtractor Class

```python
class DiagramEntityExtractor:
    def __init__(self, schema_validator, llm_func, config=None):
        self.schema_validator = schema_validator
        self.llm_func = llm_func
        self.config = config or {}
        
        # Configuration settings with defaults
        self.confidence_threshold = self.config.get('diagram_entity_confidence', 0.7)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.enable_caching = self.config.get('enable_diagram_entity_cache', True)
        self.cache_dir = self.config.get('diagram_entity_cache_dir', '~/.lightrag/diagram_entity_cache')
        
        # Initialize cache if enabled
        if self.enable_caching:
            self._init_cache()
```

### 11.2 Entity Extraction Implementation

```python
async def extract_entities_from_diagram(self, diagram_data):
    """
    Extract entities from a diagram description.
    
    Args:
        diagram_data: Dictionary containing diagram information
            including description, caption, and metadata
            
    Returns:
        List of extracted entities with schema types
    """
    diagram_id = diagram_data.get('diagram_id', 'unknown')
    
    # Check cache for entities
    if self.enable_caching and diagram_id != 'unknown':
        cached_entities = self.get_entities_from_cache(diagram_id)
        if cached_entities:
            return cached_entities
    
    # Build prompt for entity extraction
    prompt = self._build_entity_extraction_prompt(diagram_data)
    
    # Call LLM to extract entities
    try:
        response_text = await self._call_llm_with_retry(prompt)
        
        # Parse the LLM response
        extracted_entities = self._parse_entity_response(response_text)
        
        # Validate entities against schema
        validated_entities = []
        for entity in extracted_entities:
            entity_type = entity.get('entity_type')
            properties = entity.get('properties', {})
            
            # Validate entity against schema
            is_valid, error_msg = self.schema_validator.validate_entity(
                entity_type, properties
            )
            
            if is_valid:
                # Add source information to the entity
                entity['source_id'] = diagram_id
                entity['extraction_method'] = 'diagram'
                validated_entities.append(entity)
            else:
                logger.warning(f"Invalid entity from diagram {diagram_id}: {error_msg}")
        
        # Cache validated entities
        if self.enable_caching and diagram_id != 'unknown':
            self.save_entities_to_cache(diagram_id, validated_entities)
        
        return validated_entities
        
    except Exception as e:
        logger.error(f"Error extracting entities from diagram {diagram_id}: {str(e)}")
        return []
```

## 12. Schema Integration

The diagram entity extraction will utilize the existing schema system without requiring schema modifications:

### 12.1 Entity Type Mapping

Entities extracted from diagrams will be mapped to existing schema entity types through:
1. Direct Type Mapping: The LLM will map diagram components to available schema entity types
2. Context-Aware Classification: The system will use surrounding text and captions for better classification
3. Diagram Type Hints: Different diagram types will suggest likely entity types (e.g., UML diagrams suggest "Class" entities)

### 12.2 Relationship Mapping

Relationships between diagram entities will be mapped to schema relationship types through:
1. Connection Analysis: Analyzing described connections in the diagram
2. Schema Validation: Ensuring relationships adhere to schema definitions
3. Visual Relationship Translation: Converting visual relationships to semantic relationships
4. Directional Analysis: Determining source and target roles based on diagram conventions

### 12.3 Schema-Guided Property Extraction

The system will use schema property definitions to guide property extraction from diagrams:

```python
def _extract_properties_for_entity_type(self, entity_type, entity_description):
    """
    Extract properties for an entity based on its type and description.
    
    Args:
        entity_type: The entity type
        entity_description: The description of the entity
        
    Returns:
        Dictionary of extracted properties
    """
    properties = {}
    
    # Get property definitions for this entity type
    property_defs = self.schema_validator.get_entity_properties(entity_type)
    
    # Attempt to extract each property from the description
    for prop_def in property_defs:
        prop_name = prop_def.get('name')
        prop_type = prop_def.get('type', 'string')
        
        # Skip complex property types
        if prop_type not in ['string', 'integer', 'float', 'boolean']:
            continue
            
        # Try to extract property value from description
        prop_value = self._extract_property_value(
            prop_name, 
            prop_type, 
            entity_description
        )
        
        if prop_value is not None:
            properties[prop_name] = prop_value
    
    return properties
```

## 13. Evaluation Metrics

### 13.1 Quantitative Metrics

#### Entity Extraction Metrics

- **Entity Extraction Rate**: The number of entities extracted per diagram
- **Entity Type Coverage**: The percentage of schema entity types represented in the extracted entities
- **Entity Property Density**: The average number of properties per entity

#### Relationship Extraction Metrics

- **Relationship Extraction Rate**: The number of relationships extracted per diagram
- **Relationship to Entity Ratio**: The ratio of relationships to entities
- **Relationship Type Coverage**: The percentage of schema relationship types represented

### 13.2 Qualitative Metrics

- **Entity Name Accuracy**: How accurately entity names are extracted
- **Entity Type Accuracy**: How accurately entity types are assigned
- **Relationship Accuracy**: How accurately relationships are extracted

### 13.3 Evaluation Framework

An evaluation framework will be implemented to regularly measure these metrics:

```python
async def evaluate_diagram_entity_extraction(dataset, analyzer, schema_validator, llm_func):
    """Evaluate diagram entity extraction using the evaluation dataset."""
    results = {
        'entity_metrics': {},
        'relationship_metrics': {},
        'performance_metrics': {}
    }
    
    # Process each diagram in dataset
    for diagram in dataset['diagrams']:
        # Extract entities and relationships
        start_time = time.time()
        entities, relationships = await analyzer.extract_entities_and_relationships(
            diagram, schema_validator, llm_func
        )
        end_time = time.time()
        
        # Get ground truth data
        ground_truth = diagram.get('ground_truth', {})
        
        # Calculate and store metrics
        # ...
    
    return results
```

## 14. Configuration Options

The following configuration settings will be added:

```ini
[diagram_entity_extraction]
# Enable or disable diagram entity extraction
enabled = true

# Confidence threshold for extracted entities (0.0-1.0)
confidence_threshold = 0.7

# Enable or disable caching
enable_diagram_entity_cache = true

# Cache directory path
diagram_entity_cache_dir = ~/.lightrag/diagram_entity_cache

# Cache expiry time in seconds (default: 1 week)
diagram_entity_cache_expiry = 604800

# Maximum number of retries for LLM calls
max_retries = 3

# Delay between retries in seconds
retry_delay = 1.0
```

## 15. Example Usage

```python
# Initialize components
schema_path = "path/to/your/schema.json"
schema_validator = SchemaValidator(schema_path)
llm_func = initialize_llm_function()
analyzer = DiagramAnalyzer()

# Extract diagrams from PDF
pdf_path = "examples/sample_architecture.pdf"
diagrams = analyzer.extract_diagrams_from_pdf(pdf_path)

# Process diagrams for entity/relationship extraction
for diagram in diagrams:
    # Generate description if needed
    if not diagram.get('description'):
        diagram['description'] = await analyzer.generate_diagram_description(diagram)
    
    # Extract entities and relationships
    entities, relationships = await analyzer.extract_entities_and_relationships(
        diagram, schema_validator, llm_func
    )
    
    # Add to knowledge graph
    entity_ids = {}
    for entity in entities:
        entity_id = kg.add_entity(
            entity['entity_type'],
            entity['entity_name'],
            entity.get('properties', {}),
            entity.get('description', '')
        )
        entity_ids[entity['entity_name']] = entity_id
    
    # Add relationships
    for rel in relationships:
        source_id = entity_ids.get(rel['src_id'])
        target_id = entity_ids.get(rel['tgt_id'])
        
        if source_id and target_id:
            kg.add_relationship(
                source_id,
                target_id,
                rel['keywords'],
                rel.get('properties', {}),
                rel.get('description', '')
            )
```

## 16. Conclusion

The diagram entity extraction enhancement will significantly improve LightRAG's knowledge graph construction capabilities by incorporating structured information from diagrams. By extracting entities and relationships from diagram descriptions generated by vision models, this feature enables a more comprehensive understanding of document content, bridging the gap between visual and textual information.