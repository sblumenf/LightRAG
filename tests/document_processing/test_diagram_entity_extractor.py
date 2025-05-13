"""
Unit tests for the DiagramEntityExtractor class.

These tests provide comprehensive coverage for the DiagramEntityExtractor class,
including edge cases, caching behavior, and error handling.
"""
import pytest
import asyncio
import json
import os
import time
import tempfile
import shutil
import pickle
from unittest.mock import patch, MagicMock, AsyncMock, call
from pathlib import Path

from document_processing.diagram_entity_extractor import DiagramEntityExtractor

# Mock schema validator class
class MockSchemaValidator:
    def __init__(self, entity_types=None, relationship_types=None):
        self.entity_types = entity_types or {
            'Component': {
                'name': 'Component',
                'properties': [
                    {'name': 'name', 'type': 'string', 'required': True},
                    {'name': 'function', 'type': 'string', 'required': False},
                ]
            },
            'Service': {
                'name': 'Service',
                'properties': [
                    {'name': 'name', 'type': 'string', 'required': True},
                    {'name': 'endpoint', 'type': 'string', 'required': False},
                ]
            }
        }
        
        self.relationship_types = relationship_types or {
            'CALLS': {
                'name': 'CALLS',
                'source': 'Component',
                'target': 'Service',
                'properties': [
                    {'name': 'protocol', 'type': 'string', 'required': False}
                ]
            },
            'CONTAINS': {
                'name': 'CONTAINS',
                'source': 'Component',
                'target': 'Component',
                'properties': []
            }
        }

    def get_entity_types(self):
        return list(self.entity_types.keys())
    
    def get_relationship_types(self):
        return list(self.relationship_types.keys())
    
    def get_entity_properties(self, entity_type):
        if entity_type in self.entity_types:
            return self.entity_types[entity_type].get('properties', [])
        return []
    
    def get_relationship_properties(self, relationship_type):
        if relationship_type in self.relationship_types:
            return self.relationship_types[relationship_type].get('properties', [])
        return []
    
    def validate_entity(self, entity_type, properties):
        if entity_type not in self.entity_types:
            return False, f"Entity type '{entity_type}' not found in schema"
        return True, ""
    
    def validate_relationship(self, relationship_type, source_type, target_type, properties):
        if relationship_type not in self.relationship_types:
            return False, f"Relationship type '{relationship_type}' not found in schema"
        
        rel_def = self.relationship_types.get(relationship_type, {})
        if rel_def.get('source') != source_type:
            return False, f"Invalid source entity type '{source_type}'"
        
        if rel_def.get('target') != target_type:
            return False, f"Invalid target entity type '{target_type}'"
            
        return True, ""


# Test data
MOCK_DIAGRAM_DATA = {
    'diagram_id': 'test-diagram-123',
    'description': 'This architecture diagram shows a Frontend component that connects to a Backend API service. The Frontend makes HTTP requests to the Backend API which processes the data.',
    'caption': 'System Architecture Diagram',
    'diagram_type': 'architecture_diagram'
}

MOCK_LLM_ENTITY_RESPONSE = """
{
  "entities": [
    {
      "entity_name": "Frontend",
      "entity_type": "Component",
      "properties": {
        "name": "Frontend",
        "function": "User interface"
      },
      "description": "The frontend component that handles user interactions"
    },
    {
      "entity_name": "BackendAPI",
      "entity_type": "Service",
      "properties": {
        "name": "Backend API",
        "endpoint": "/api"
      },
      "description": "The backend API service that processes requests"
    }
  ]
}
"""

MOCK_LLM_RELATIONSHIP_RESPONSE = """
{
  "relationships": [
    {
      "source": "Frontend",
      "target": "BackendAPI",
      "type": "CALLS",
      "description": "Frontend makes HTTP requests to the Backend API",
      "properties": {
        "protocol": "HTTP"
      }
    }
  ]
}
"""

MOCK_ENTITIES = [
    {
        "entity_name": "Frontend",
        "entity_type": "Component",
        "properties": {
            "name": "Frontend",
            "function": "User interface"
        },
        "description": "The frontend component that handles user interactions"
    },
    {
        "entity_name": "BackendAPI",
        "entity_type": "Service",
        "properties": {
            "name": "Backend API",
            "endpoint": "/api"
        },
        "description": "The backend API service that processes requests"
    }
]


@pytest.fixture
def mock_llm_func():
    async def mock_func(prompt):
        if "Extract named entities" in prompt:
            return MOCK_LLM_ENTITY_RESPONSE
        elif "Extract relationships between entities" in prompt:
            return MOCK_LLM_RELATIONSHIP_RESPONSE
        return ""
    return mock_func


@pytest.fixture
def schema_validator():
    return MockSchemaValidator()


@pytest.fixture
def entity_extractor(schema_validator, mock_llm_func):
    config = {
        'enable_diagram_entity_cache': False  # Disable caching for tests
    }
    return DiagramEntityExtractor(schema_validator, mock_llm_func, config)


@pytest.mark.asyncio
async def test_extract_entities_from_diagram(entity_extractor):
    # Test entity extraction
    entities = await entity_extractor.extract_entities_from_diagram(MOCK_DIAGRAM_DATA)
    
    # Verify results
    assert len(entities) == 2
    assert entities[0]['entity_name'] == 'Frontend'
    assert entities[0]['entity_type'] == 'Component'
    assert entities[1]['entity_name'] == 'BackendAPI'
    assert entities[1]['entity_type'] == 'Service'
    
    # Check that source information is added
    assert entities[0]['source_id'] == 'test-diagram-123'
    assert entities[0]['extraction_method'] == 'diagram'


@pytest.mark.asyncio
async def test_extract_relationships_from_diagram(entity_extractor):
    # Test relationship extraction
    relationships = await entity_extractor.extract_relationships_from_diagram(MOCK_DIAGRAM_DATA, MOCK_ENTITIES)
    
    # Verify results
    assert len(relationships) == 1
    assert relationships[0]['source'] == 'Frontend'
    assert relationships[0]['target'] == 'BackendAPI'
    assert relationships[0]['type'] == 'CALLS'
    assert 'protocol' in relationships[0]['properties']
    assert relationships[0]['properties']['protocol'] == 'HTTP'


@pytest.mark.asyncio
async def test_extract_entities_with_invalid_schema(schema_validator, mock_llm_func):
    # Test with invalid entity type in response
    invalid_entity_response = """
    {
      "entities": [
        {
          "entity_name": "InvalidEntity",
          "entity_type": "NonExistentType",
          "properties": {},
          "description": "This entity type doesn't exist in the schema"
        }
      ]
    }
    """
    
    async def mock_invalid_llm(prompt):
        return invalid_entity_response
    
    # Create extractor with the invalid response
    extractor = DiagramEntityExtractor(schema_validator, mock_invalid_llm, {'enable_diagram_entity_cache': False})
    
    # Extract entities
    entities = await extractor.extract_entities_from_diagram(MOCK_DIAGRAM_DATA)
    
    # Should filter out invalid entities
    assert len(entities) == 0


@pytest.mark.asyncio
async def test_caching(schema_validator, mock_llm_func, tmp_path):
    # Create a temp cache directory
    cache_dir = tmp_path / "diagram_entity_cache"
    cache_dir.mkdir()
    
    # Configure with caching enabled
    config = {
        'enable_diagram_entity_cache': True,
        'diagram_entity_cache_dir': str(cache_dir)
    }
    
    # Create a mock LLM that counts calls
    call_count = 0
    async def counting_llm_func(prompt):
        nonlocal call_count
        call_count += 1
        if "Extract named entities" in prompt:
            return MOCK_LLM_ENTITY_RESPONSE
        elif "Extract relationships between entities" in prompt:
            return MOCK_LLM_RELATIONSHIP_RESPONSE
        return ""
    
    # Create extractor
    extractor = DiagramEntityExtractor(schema_validator, counting_llm_func, config)
    
    # First call should hit the LLM
    entities1 = await extractor.extract_entities_from_diagram(MOCK_DIAGRAM_DATA)
    assert call_count == 1
    assert len(entities1) == 2
    
    # Second call should use cache
    entities2 = await extractor.extract_entities_from_diagram(MOCK_DIAGRAM_DATA)
    assert call_count == 1  # Call count should not increase
    assert len(entities2) == 2
    
    # Verify that cache files were created
    assert os.path.exists(cache_dir / "diagram_entities.pkl")


@pytest.mark.asyncio
async def test_llm_retry_logic(schema_validator):
    # Create a mock LLM that fails twice then succeeds
    failure_count = 0
    async def failing_llm_func(prompt):
        nonlocal failure_count
        if failure_count < 2:
            failure_count += 1
            raise Exception(f"Mock LLM failure {failure_count}")
        return MOCK_LLM_ENTITY_RESPONSE
    
    # Configure with retry settings
    config = {
        'enable_diagram_entity_cache': False,
        'max_retries': 3,
        'retry_delay': 0.01
    }
    
    # Create extractor
    extractor = DiagramEntityExtractor(schema_validator, failing_llm_func, config)
    
    # Should succeed after retries
    entities = await extractor.extract_entities_from_diagram(MOCK_DIAGRAM_DATA)
    assert len(entities) == 2
    assert failure_count == 2  # Verify it failed exactly twice


@pytest.mark.asyncio
async def test_empty_description(entity_extractor):
    # Test with empty description
    empty_data = MOCK_DIAGRAM_DATA.copy()
    empty_data['description'] = ""

    # Should return empty list
    entities = await entity_extractor.extract_entities_from_diagram(empty_data)
    assert len(entities) == 0


@pytest.mark.asyncio
async def test_null_description(entity_extractor):
    # Test with null description
    null_data = MOCK_DIAGRAM_DATA.copy()
    del null_data['description']

    # Should return empty list
    entities = await entity_extractor.extract_entities_from_diagram(null_data)
    assert len(entities) == 0


@pytest.mark.asyncio
async def test_build_entity_extraction_prompt(entity_extractor):
    # Test prompt building
    prompt = entity_extractor._build_entity_extraction_prompt(MOCK_DIAGRAM_DATA)

    # Verify prompt content
    assert "DIAGRAM DESCRIPTION" in prompt
    assert MOCK_DIAGRAM_DATA['description'] in prompt
    assert "DIAGRAM CAPTION" in prompt
    assert MOCK_DIAGRAM_DATA['caption'] in prompt
    assert "SCHEMA ENTITY TYPES" in prompt
    assert "Component" in prompt
    assert "Service" in prompt


@pytest.mark.asyncio
async def test_build_relationship_extraction_prompt(entity_extractor):
    # Test relationship prompt building
    prompt = entity_extractor._build_relationship_extraction_prompt(MOCK_DIAGRAM_DATA, MOCK_ENTITIES)

    # Verify prompt content
    assert "DIAGRAM DESCRIPTION" in prompt
    assert MOCK_DIAGRAM_DATA['description'] in prompt
    assert "DIAGRAM CAPTION" in prompt
    assert MOCK_DIAGRAM_DATA['caption'] in prompt
    assert "ENTITIES" in prompt
    assert "Frontend" in prompt
    assert "BackendAPI" in prompt
    assert "RELATIONSHIP TYPES" in prompt
    assert "CALLS" in prompt
    assert "CONTAINS" in prompt


@pytest.mark.asyncio
async def test_parse_entity_response_valid_json(entity_extractor):
    # Test with valid JSON response
    response = """
    {
      "entities": [
        {
          "entity_name": "TestEntity",
          "entity_type": "Component",
          "properties": {
            "name": "Test"
          },
          "description": "Test entity"
        }
      ]
    }
    """

    entities = entity_extractor._parse_entity_response(response)
    assert len(entities) == 1
    assert entities[0]['entity_name'] == "TestEntity"
    assert entities[0]['entity_type'] == "Component"


@pytest.mark.asyncio
async def test_parse_entity_response_invalid_json(entity_extractor):
    # Test with invalid JSON
    response = "This is not JSON"
    entities = entity_extractor._parse_entity_response(response)
    assert len(entities) == 0

    # Test with valid text but no JSON object
    response = "I found the following entities: Entity1, Entity2"
    entities = entity_extractor._parse_entity_response(response)
    assert len(entities) == 0

    # Test with JSON but no entities field
    response = '{"results": [{"name": "Entity1"}]}'
    entities = entity_extractor._parse_entity_response(response)
    assert len(entities) == 0


@pytest.mark.asyncio
async def test_parse_relationship_response(entity_extractor):
    # Test with valid JSON response
    response = """
    {
      "relationships": [
        {
          "source": "Entity1",
          "target": "Entity2",
          "type": "CALLS",
          "description": "Test relationship",
          "properties": {
            "property1": "value1"
          }
        }
      ]
    }
    """

    relationships = entity_extractor._parse_relationship_response(response)
    assert len(relationships) == 1
    assert relationships[0]['source'] == "Entity1"
    assert relationships[0]['target'] == "Entity2"
    assert relationships[0]['type'] == "CALLS"

    # Test with invalid JSON
    response = "This is not JSON"
    relationships = entity_extractor._parse_relationship_response(response)
    assert len(relationships) == 0


@pytest.mark.asyncio
async def test_invalid_relationship(entity_extractor):
    # Create entities with incompatible types for a relationship
    invalid_entities = [
        {
            "entity_name": "Service1",
            "entity_type": "Service",
            "properties": {"name": "Service1"}
        },
        {
            "entity_name": "Service2",
            "entity_type": "Service",
            "properties": {"name": "Service2"}
        }
    ]

    # Mock LLM to return an invalid relationship (Service CALLS Service)
    async def invalid_rel_llm(prompt):
        return """
        {
          "relationships": [
            {
              "source": "Service1",
              "target": "Service2",
              "type": "CALLS",
              "properties": {}
            }
          ]
        }
        """

    extractor = DiagramEntityExtractor(
        MockSchemaValidator(),
        invalid_rel_llm,
        {'enable_diagram_entity_cache': False}
    )

    # Should filter out invalid relationships
    relationships = await extractor.extract_relationships_from_diagram(MOCK_DIAGRAM_DATA, invalid_entities)
    assert len(relationships) == 0


@pytest.mark.asyncio
async def test_nonexistent_entity_relationships(entity_extractor):
    # Mock LLM to return relationships with nonexistent entities
    async def nonexistent_entity_llm(prompt):
        return """
        {
          "relationships": [
            {
              "source": "NonExistentEntity1",
              "target": "BackendAPI",
              "type": "CALLS",
              "properties": {}
            },
            {
              "source": "Frontend",
              "target": "NonExistentEntity2",
              "type": "CALLS",
              "properties": {}
            }
          ]
        }
        """

    extractor = DiagramEntityExtractor(
        MockSchemaValidator(),
        nonexistent_entity_llm,
        {'enable_diagram_entity_cache': False}
    )

    # Should filter out relationships with nonexistent entities
    relationships = await extractor.extract_relationships_from_diagram(MOCK_DIAGRAM_DATA, MOCK_ENTITIES)
    assert len(relationships) == 0

@pytest.mark.asyncio
async def test_circular_relationships(schema_validator):
    # Create custom mock schema validator that allows circular relationships
    circular_schema = MockSchemaValidator(
        entity_types={
            'Component': {
                'name': 'Component',
                'properties': [
                    {'name': 'name', 'type': 'string', 'required': True},
                ]
            }
        },
        relationship_types={
            'CONNECTS_TO': {
                'name': 'CONNECTS_TO',
                'source': 'Component',
                'target': 'Component',
                'properties': [
                    {'name': 'direction', 'type': 'string', 'required': False}
                ]
            }
        }
    )

    # Mock entities with same type for circular relationships
    circular_entities = [
        {
            "entity_name": "ComponentA",
            "entity_type": "Component",
            "properties": {"name": "Component A"}
        },
        {
            "entity_name": "ComponentB",
            "entity_type": "Component",
            "properties": {"name": "Component B"}
        },
        {
            "entity_name": "ComponentC",
            "entity_type": "Component",
            "properties": {"name": "Component C"}
        }
    ]

    # Mock LLM to return circular relationships (A→B→C→A)
    async def circular_llm(prompt):
        return """
        {
          "relationships": [
            {
              "source": "ComponentA",
              "target": "ComponentB",
              "type": "CONNECTS_TO",
              "description": "A connects to B",
              "properties": {"direction": "bidirectional"}
            },
            {
              "source": "ComponentB",
              "target": "ComponentC",
              "type": "CONNECTS_TO",
              "description": "B connects to C",
              "properties": {"direction": "bidirectional"}
            },
            {
              "source": "ComponentC",
              "target": "ComponentA",
              "type": "CONNECTS_TO",
              "description": "C connects back to A (circular)",
              "properties": {"direction": "bidirectional"}
            }
          ]
        }
        """

    # Create extractor with circular schema
    extractor = DiagramEntityExtractor(
        circular_schema,
        circular_llm,
        {'enable_diagram_entity_cache': False}
    )

    # Extract relationships
    diagram_data = {
        'diagram_id': 'circular-diagram',
        'description': 'This is a circular network diagram showing ComponentA connected to ComponentB, ComponentB connected to ComponentC, and ComponentC connected back to ComponentA.',
        'caption': 'Circular Network Diagram',
        'diagram_type': 'network_diagram'
    }

    relationships = await extractor.extract_relationships_from_diagram(diagram_data, circular_entities)

    # Should extract all 3 relationships
    assert len(relationships) == 3

    # Verify circular connection exists
    circular_sources = [r['source'] for r in relationships]
    circular_targets = [r['target'] for r in relationships]

    # Check that ComponentA is both a source and a target
    assert "ComponentA" in circular_sources
    assert "ComponentA" in circular_targets

    # Check that there's a path: A→B→C→A
    assert any(r['source'] == 'ComponentA' and r['target'] == 'ComponentB' for r in relationships)
    assert any(r['source'] == 'ComponentB' and r['target'] == 'ComponentC' for r in relationships)
    assert any(r['source'] == 'ComponentC' and r['target'] == 'ComponentA' for r in relationships)


@pytest.mark.asyncio
async def test_relationship_with_missing_fields(entity_extractor):
    # Mock LLM to return relationships with missing required fields
    async def missing_fields_llm(prompt):
        return """
        {
          "relationships": [
            {
              "source": "Frontend",
              "type": "CALLS",
              "properties": {}
            },
            {
              "source": "Frontend",
              "target": "BackendAPI",
              "properties": {}
            }
          ]
        }
        """

    extractor = DiagramEntityExtractor(
        MockSchemaValidator(),
        missing_fields_llm,
        {'enable_diagram_entity_cache': False}
    )

    # Should filter out relationships with missing fields
    relationships = await extractor.extract_relationships_from_diagram(MOCK_DIAGRAM_DATA, MOCK_ENTITIES)
    assert len(relationships) == 0


@pytest.mark.asyncio
async def test_llm_call_retry_success(schema_validator):
    # Test retry logic - success after failure
    call_count = 0

    async def failing_then_success_llm(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("First attempt failed")
        return MOCK_LLM_ENTITY_RESPONSE

    config = {
        'enable_diagram_entity_cache': False,
        'max_retries': 3,
        'retry_delay': 0.01  # Small delay for tests
    }

    extractor = DiagramEntityExtractor(schema_validator, failing_then_success_llm, config)
    entities = await extractor.extract_entities_from_diagram(MOCK_DIAGRAM_DATA)

    # Verify that LLM was called twice (one failure, one success)
    assert call_count == 2
    assert len(entities) == 2


@pytest.mark.asyncio
async def test_llm_call_retry_exhausted(schema_validator):
    # Test retry logic - all attempts fail
    call_count = 0

    async def always_failing_llm(prompt):
        nonlocal call_count
        call_count += 1
        raise Exception(f"Attempt {call_count} failed")

    config = {
        'enable_diagram_entity_cache': False,
        'max_retries': 3,
        'retry_delay': 0.01  # Small delay for tests
    }

    extractor = DiagramEntityExtractor(schema_validator, always_failing_llm, config)
    entities = await extractor.extract_entities_from_diagram(MOCK_DIAGRAM_DATA)

    # Verify that LLM was called max_retries times and returned empty list
    assert call_count == 3
    assert len(entities) == 0


@pytest.mark.asyncio
async def test_cache_initialization_and_saving(schema_validator, mock_llm_func, tmp_path):
    # Set up a temporary cache directory
    cache_dir = tmp_path / "test_cache"
    entity_cache_file = cache_dir / "diagram_entities.pkl"
    relationship_cache_file = cache_dir / "diagram_relationships.pkl"

    config = {
        'enable_diagram_entity_cache': True,
        'diagram_entity_cache_dir': str(cache_dir)
    }

    # Create extractor with caching enabled
    extractor = DiagramEntityExtractor(schema_validator, mock_llm_func, config)

    # Verify cache directory was created
    assert os.path.exists(cache_dir)

    # Extract entities and relationships to populate cache
    diagram_data = MOCK_DIAGRAM_DATA.copy()
    diagram_data['diagram_id'] = 'test-cache-diagram'

    entities = await extractor.extract_entities_from_diagram(diagram_data)
    relationships = await extractor.extract_relationships_from_diagram(diagram_data, entities)

    # Verify cache files were created
    assert os.path.exists(entity_cache_file)
    assert os.path.exists(relationship_cache_file)

    # Verify cache content
    with open(entity_cache_file, 'rb') as f:
        entity_cache = pickle.load(f)
        assert 'test-cache-diagram' in entity_cache
        assert 'entities' in entity_cache['test-cache-diagram']
        assert len(entity_cache['test-cache-diagram']['entities']) == len(entities)

    with open(relationship_cache_file, 'rb') as f:
        relationship_cache = pickle.load(f)
        assert 'test-cache-diagram' in relationship_cache
        assert 'relationships' in relationship_cache['test-cache-diagram']
        assert len(relationship_cache['test-cache-diagram']['relationships']) == len(relationships)


@pytest.mark.asyncio
async def test_cache_retrieval(schema_validator, mock_llm_func, tmp_path):
    # Set up a temporary cache directory
    cache_dir = tmp_path / "test_cache_retrieval"
    cache_dir.mkdir()

    config = {
        'enable_diagram_entity_cache': True,
        'diagram_entity_cache_dir': str(cache_dir)
    }

    # Create extractor with caching enabled
    extractor = DiagramEntityExtractor(schema_validator, mock_llm_func, config)

    # Extract entities and relationships to populate cache
    diagram_data = MOCK_DIAGRAM_DATA.copy()
    diagram_data['diagram_id'] = 'test-cache-retrieval'

    # Count LLM calls
    call_count = 0
    async def counting_llm_func(prompt):
        nonlocal call_count
        call_count += 1
        if "Extract named entities" in prompt:
            return MOCK_LLM_ENTITY_RESPONSE
        elif "Extract relationships between entities" in prompt:
            return MOCK_LLM_RELATIONSHIP_RESPONSE
        return ""

    extractor.llm_func = counting_llm_func

    # First call should hit the LLM
    entities1 = await extractor.extract_entities_from_diagram(diagram_data)
    assert call_count == 1

    # Second call should use cache
    entities2 = await extractor.extract_entities_from_diagram(diagram_data)
    assert call_count == 1  # Should not increase

    # Verify cached entities match
    assert len(entities1) == len(entities2)
    assert entities1[0]['entity_name'] == entities2[0]['entity_name']


@pytest.mark.asyncio
async def test_cache_clear(schema_validator, mock_llm_func, tmp_path):
    # Set up a temporary cache directory
    cache_dir = tmp_path / "test_cache_clear"
    cache_dir.mkdir()
    entity_cache_file = cache_dir / "diagram_entities.pkl"
    relationship_cache_file = cache_dir / "diagram_relationships.pkl"

    config = {
        'enable_diagram_entity_cache': True,
        'diagram_entity_cache_dir': str(cache_dir)
    }

    # Create extractor with caching enabled
    extractor = DiagramEntityExtractor(schema_validator, mock_llm_func, config)

    # Extract entities and relationships to populate cache
    diagram_data = MOCK_DIAGRAM_DATA.copy()
    diagram_data['diagram_id'] = 'test-cache-clear'

    await extractor.extract_entities_from_diagram(diagram_data)

    # Verify cache files were created
    assert os.path.exists(entity_cache_file)

    # Clear cache
    extractor.clear_cache()

    # Verify cache files were removed
    assert not os.path.exists(entity_cache_file)
    assert not os.path.exists(relationship_cache_file)

    # Verify in-memory cache is empty
    assert len(extractor.entity_cache) == 0
    assert len(extractor.relationship_cache) == 0


@pytest.mark.asyncio
async def test_cache_expiry(schema_validator, mock_llm_func, tmp_path):
    # Set up a temporary cache directory
    cache_dir = tmp_path / "test_cache_expiry"
    cache_dir.mkdir()

    # Configure with short cache expiry
    config = {
        'enable_diagram_entity_cache': True,
        'diagram_entity_cache_dir': str(cache_dir),
        'diagram_entity_cache_expiry': 0.1  # 100ms expiry for testing
    }

    # Create extractor with caching enabled
    extractor = DiagramEntityExtractor(schema_validator, mock_llm_func, config)

    # Extract entities and relationships to populate cache
    diagram_data = MOCK_DIAGRAM_DATA.copy()
    diagram_data['diagram_id'] = 'test-cache-expiry'

    # Count LLM calls
    call_count = 0
    async def counting_llm_func(prompt):
        nonlocal call_count
        call_count += 1
        if "Extract named entities" in prompt:
            return MOCK_LLM_ENTITY_RESPONSE
        elif "Extract relationships between entities" in prompt:
            return MOCK_LLM_RELATIONSHIP_RESPONSE
        return ""

    extractor.llm_func = counting_llm_func

    # First call should hit the LLM
    await extractor.extract_entities_from_diagram(diagram_data)
    assert call_count == 1

    # Wait for cache to expire
    await asyncio.sleep(0.2)

    # Call after expiry should hit the LLM again
    await extractor.extract_entities_from_diagram(diagram_data)
    assert call_count == 2  # Should increase

@pytest.mark.asyncio
async def test_complex_entity_hierarchies(schema_validator):
    # Create schema with hierarchical entity types
    hierarchical_schema = MockSchemaValidator(
        entity_types={
            'System': {
                'name': 'System',
                'properties': [
                    {'name': 'name', 'type': 'string', 'required': True}
                ]
            },
            'Component': {
                'name': 'Component',
                'properties': [
                    {'name': 'name', 'type': 'string', 'required': True}
                ]
            },
            'Subcomponent': {
                'name': 'Subcomponent',
                'properties': [
                    {'name': 'name', 'type': 'string', 'required': True}
                ]
            }
        },
        relationship_types={
            'CONTAINS': {
                'name': 'CONTAINS',
                'source': 'System',
                'target': 'Component',
                'properties': []
            },
            'HAS_PART': {
                'name': 'HAS_PART',
                'source': 'Component',
                'target': 'Subcomponent',
                'properties': []
            },
            'DEPENDS_ON': {
                'name': 'DEPENDS_ON',
                'source': 'Subcomponent',
                'target': 'Subcomponent',
                'properties': []
            }
        }
    )

    # Define hierarchical entities
    hierarchical_entities = [
        {
            "entity_name": "MainSystem",
            "entity_type": "System",
            "properties": {"name": "Main System"},
            "description": "The overall system"
        },
        {
            "entity_name": "ComponentA",
            "entity_type": "Component",
            "properties": {"name": "Component A"},
            "description": "A major component"
        },
        {
            "entity_name": "ComponentB",
            "entity_type": "Component",
            "properties": {"name": "Component B"},
            "description": "Another major component"
        },
        {
            "entity_name": "SubA1",
            "entity_type": "Subcomponent",
            "properties": {"name": "Subcomponent A1"},
            "description": "A subcomponent of Component A"
        },
        {
            "entity_name": "SubA2",
            "entity_type": "Subcomponent",
            "properties": {"name": "Subcomponent A2"},
            "description": "Another subcomponent of Component A"
        },
        {
            "entity_name": "SubB1",
            "entity_type": "Subcomponent",
            "properties": {"name": "Subcomponent B1"},
            "description": "A subcomponent of Component B"
        }
    ]

    # Mock LLM to return hierarchical relationships
    async def hierarchical_llm(prompt):
        return """
        {
          "relationships": [
            {
              "source": "MainSystem",
              "target": "ComponentA",
              "type": "CONTAINS",
              "description": "Main System contains Component A",
              "properties": {}
            },
            {
              "source": "MainSystem",
              "target": "ComponentB",
              "type": "CONTAINS",
              "description": "Main System contains Component B",
              "properties": {}
            },
            {
              "source": "ComponentA",
              "target": "SubA1",
              "type": "HAS_PART",
              "description": "Component A has Subcomponent A1",
              "properties": {}
            },
            {
              "source": "ComponentA",
              "target": "SubA2",
              "type": "HAS_PART",
              "description": "Component A has Subcomponent A2",
              "properties": {}
            },
            {
              "source": "ComponentB",
              "target": "SubB1",
              "type": "HAS_PART",
              "description": "Component B has Subcomponent B1",
              "properties": {}
            },
            {
              "source": "SubA2",
              "target": "SubB1",
              "type": "DEPENDS_ON",
              "description": "Subcomponent A2 depends on Subcomponent B1",
              "properties": {}
            }
          ]
        }
        """

    # Create extractor with hierarchical schema
    extractor = DiagramEntityExtractor(
        hierarchical_schema,
        hierarchical_llm,
        {'enable_diagram_entity_cache': False}
    )

    # Extract relationships
    diagram_data = {
        'diagram_id': 'hierarchical-diagram',
        'description': 'A hierarchical system diagram showing MainSystem containing ComponentA and ComponentB. ComponentA has SubA1 and SubA2. ComponentB has SubB1. SubA2 depends on SubB1.',
        'caption': 'Hierarchical System Diagram',
        'diagram_type': 'system_diagram'
    }

    relationships = await extractor.extract_relationships_from_diagram(diagram_data, hierarchical_entities)

    # Should extract all 6 relationships
    assert len(relationships) == 6

    # Check different relationship types are extracted correctly
    contains_relationships = [r for r in relationships if r['type'] == 'CONTAINS']
    has_part_relationships = [r for r in relationships if r['type'] == 'HAS_PART']
    depends_on_relationships = [r for r in relationships if r['type'] == 'DEPENDS_ON']

    assert len(contains_relationships) == 2
    assert len(has_part_relationships) == 3
    assert len(depends_on_relationships) == 1

    # Verify cross-component dependency
    cross_dependencies = [r for r in relationships
                         if r['source'] == 'SubA2' and r['target'] == 'SubB1']
    assert len(cross_dependencies) == 1

@pytest.mark.asyncio
async def test_relationships_with_special_characters(schema_validator):
    # Create a custom schema that supports the relationships we need
    special_schema = MockSchemaValidator(
        entity_types={
            'Component': {
                'name': 'Component',
                'properties': [
                    {'name': 'name', 'type': 'string', 'required': True}
                ]
            },
            'Service': {
                'name': 'Service',
                'properties': [
                    {'name': 'name', 'type': 'string', 'required': True}
                ]
            }
        },
        relationship_types={
            'CALLS': {
                'name': 'CALLS',
                'source': 'Component',
                'target': 'Service',
                'properties': [
                    {'name': 'protocol', 'type': 'string', 'required': False}
                ]
            },
            'USES': {
                'name': 'USES',
                'source': 'Service',
                'target': 'Component',
                'properties': [
                    {'name': 'protocol', 'type': 'string', 'required': False}
                ]
            },
            'CONNECTS_TO': {
                'name': 'CONNECTS_TO',
                'source': 'Component',
                'target': 'Component',
                'properties': [
                    {'name': 'protocol', 'type': 'string', 'required': False}
                ]
            }
        }
    )

    # Create entities with special characters in names
    special_char_entities = [
        {
            "entity_name": "API-Gateway",
            "entity_type": "Component",
            "properties": {"name": "API-Gateway"}
        },
        {
            "entity_name": "Auth_Service",
            "entity_type": "Service",
            "properties": {"name": "Auth_Service"}
        },
        {
            "entity_name": "Database/SQL",
            "entity_type": "Component",
            "properties": {"name": "Database/SQL"}
        },
        {
            "entity_name": "User Interface (UI)",
            "entity_type": "Component",
            "properties": {"name": "User Interface (UI)"}
        }
    ]

    # Mock LLM to return relationships with special characters
    async def special_char_llm(prompt):
        return """
        {
          "relationships": [
            {
              "source": "API-Gateway",
              "target": "Auth_Service",
              "type": "CALLS",
              "description": "API Gateway calls Auth Service",
              "properties": {"protocol": "REST"}
            },
            {
              "source": "Auth_Service",
              "target": "Database/SQL",
              "type": "USES",
              "description": "Auth Service uses Database/SQL",
              "properties": {"protocol": "JDBC"}
            },
            {
              "source": "User Interface (UI)",
              "target": "API-Gateway",
              "type": "CONNECTS_TO",
              "description": "UI connects to API Gateway",
              "properties": {"protocol": "HTTPS"}
            }
          ]
        }
        """

    # Create extractor
    extractor = DiagramEntityExtractor(
        special_schema,
        special_char_llm,
        {'enable_diagram_entity_cache': False}
    )

    # Extract relationships
    diagram_data = {
        'diagram_id': 'special-chars-diagram',
        'description': 'System with API-Gateway, Auth_Service, Database/SQL, and User Interface (UI). UI calls API-Gateway, which calls Auth_Service, which calls Database/SQL.',
        'caption': 'System Architecture with Special Characters',
        'diagram_type': 'architecture_diagram'
    }

    relationships = await extractor.extract_relationships_from_diagram(diagram_data, special_char_entities)

    # Should extract all 3 relationships
    assert len(relationships) == 3

    # Check that entity names with special characters are handled correctly
    assert any(r['source'] == 'API-Gateway' and r['target'] == 'Auth_Service' and r['type'] == 'CALLS' for r in relationships)
    assert any(r['source'] == 'Auth_Service' and r['target'] == 'Database/SQL' and r['type'] == 'USES' for r in relationships)
    assert any(r['source'] == 'User Interface (UI)' and r['target'] == 'API-Gateway' and r['type'] == 'CONNECTS_TO' for r in relationships)

@pytest.mark.asyncio
async def test_malformed_relationship_response(entity_extractor):
    # Mock LLM to return malformed relationship response
    async def malformed_llm(prompt):
        return """
        {
          "relationships": [
            {
              "source": "Frontend",
              "target": "BackendAPI",
              "type": "CALLS"
              "description": "Frontend calls Backend API",
              "properties": {"protocol": "HTTP"}
            },
            {
              source: "BackendAPI",
              target: "Database",
              type: "USES",
              description: "Backend API uses Database",
              properties: {}
            }
          ]
        }
        """

    extractor = DiagramEntityExtractor(
        MockSchemaValidator(),
        malformed_llm,
        {'enable_diagram_entity_cache': False}
    )

    # Should handle malformed JSON gracefully
    relationships = await extractor.extract_relationships_from_diagram(MOCK_DIAGRAM_DATA, MOCK_ENTITIES)
    assert len(relationships) == 0  # Should return empty list on JSON parse error

@pytest.mark.asyncio
async def test_large_number_of_relationships(schema_validator):
    """Test extraction of a large number of relationships between many entities."""

    # Create a large number of entities (20 components)
    large_entities = []
    for i in range(1, 21):
        large_entities.append({
            "entity_name": f"Component{i}",
            "entity_type": "Component",
            "properties": {"name": f"Component {i}"},
            "description": f"Component {i} description"
        })

    # Generate a large mesh network of relationships (many-to-many)
    relationships_json = {
        "relationships": []
    }

    # Create a mesh pattern where each component connects to all components with higher indices
    for i in range(1, 21):
        for j in range(i+1, 21):
            relationships_json["relationships"].append({
                "source": f"Component{i}",
                "target": f"Component{j}",
                "type": "CONTAINS",
                "description": f"Component{i} connects to Component{j}",
                "properties": {}
            })

    # Turn the relationships into JSON string
    import json
    relationships_str = json.dumps(relationships_json, indent=2)

    # Mock LLM to return the large relationship set
    async def large_relationship_llm(prompt):
        return relationships_str

    # Create schema that allows CONTAINS relationship between Components
    component_schema = MockSchemaValidator(
        entity_types={
            'Component': {
                'name': 'Component',
                'properties': [
                    {'name': 'name', 'type': 'string', 'required': True},
                ]
            }
        },
        relationship_types={
            'CONTAINS': {
                'name': 'CONTAINS',
                'source': 'Component',
                'target': 'Component',
                'properties': []
            }
        }
    )

    # Create extractor
    extractor = DiagramEntityExtractor(
        component_schema,
        large_relationship_llm,
        {'enable_diagram_entity_cache': False}
    )

    # Extract relationships
    diagram_data = {
        'diagram_id': 'large-network-diagram',
        'description': 'A complex diagram showing connections between 20 components in a mesh pattern.',
        'caption': 'Complex Network Diagram',
        'diagram_type': 'network_diagram'
    }

    # Time the extraction operation
    import time
    start_time = time.time()
    relationships = await extractor.extract_relationships_from_diagram(diagram_data, large_entities)
    end_time = time.time()

    # We expect n*(n-1)/2 relationships for a full mesh network of n nodes
    expected_relationships = 20 * 19 // 2  # 190

    # Verify all relationships were extracted correctly
    assert len(relationships) == expected_relationships

    # Log performance info (don't assert specific time as it may vary)
    processing_time = end_time - start_time
    print(f"Processing time for {expected_relationships} relationships: {processing_time:.4f} seconds")

    # Check that some specific relationships exist
    assert any(r['source'] == 'Component1' and r['target'] == 'Component20' for r in relationships)
    assert any(r['source'] == 'Component5' and r['target'] == 'Component15' for r in relationships)
    assert any(r['source'] == 'Component10' and r['target'] == 'Component11' for r in relationships)


@pytest.mark.asyncio
async def test_cache_initialization_error_handling(tmp_path):
    """Test error handling during cache initialization."""
    # Create a directory that can't be accessed for testing
    invalid_cache_dir = tmp_path / "invalid_cache_dir"
    invalid_cache_dir.mkdir()

    # Use MagicMock for schema_validator and llm_func
    schema_validator = MagicMock()
    async def mock_llm(prompt):
        return MOCK_LLM_ENTITY_RESPONSE

    # Create a patch to simulate permission error when accessing the directory
    with patch('pickle.load', side_effect=Exception("Simulated cache load error")):
        # Configure with caching enabled but problematic cache
        config = {
            'enable_diagram_entity_cache': True,
            'diagram_entity_cache_dir': str(invalid_cache_dir)
        }

        # Create extractor - should handle errors gracefully
        extractor = DiagramEntityExtractor(schema_validator, mock_llm, config)

        # Verify that cache is initialized but empty
        assert hasattr(extractor, 'entity_cache')
        assert isinstance(extractor.entity_cache, dict)
        assert len(extractor.entity_cache) == 0

        # Test cache save with error
        with patch('builtins.open', side_effect=Exception("Simulated file write error")):
            # This should not raise an exception
            extractor._save_entity_cache()
            extractor._save_relationship_cache()

            # Test clear cache with error
            with patch('os.remove', side_effect=Exception("Simulated file delete error")):
                extractor.clear_cache()


@pytest.mark.asyncio
async def test_empty_entities_for_relationship_extraction(entity_extractor):
    """Test behavior when extracting relationships with no entities."""
    diagram_data = MOCK_DIAGRAM_DATA.copy()

    # Extract relationships with empty entities list
    relationships = await entity_extractor.extract_relationships_from_diagram(diagram_data, [])

    # Should return empty list
    assert len(relationships) == 0

    # Test with None for entities
    relationships = await entity_extractor.extract_relationships_from_diagram(diagram_data, None)

    # Should return empty list
    assert len(relationships) == 0


@pytest.mark.asyncio
async def test_caching_disabled(schema_validator, mock_llm_func, tmp_path):
    """Test behavior when caching is disabled."""
    # Configure with caching explicitly disabled
    config = {
        'enable_diagram_entity_cache': False
    }

    # Create extractor
    extractor = DiagramEntityExtractor(schema_validator, mock_llm_func, config)

    # Verify cache operations do nothing
    cached_entities = extractor.get_entities_from_cache('test-id')
    assert cached_entities is None

    cached_relationships = extractor.get_relationships_from_cache('test-id')
    assert cached_relationships is None

    # Saving to cache should be a no-op
    extractor.save_entities_to_cache('test-id', [{'name': 'Test'}])
    extractor.save_relationships_to_cache('test-id', [{'name': 'TestRel'}])

    # Clearing cache should be a no-op
    extractor.clear_cache()

    # Extract entities - should always hit LLM
    call_count = 0
    async def counting_llm_func(prompt):
        nonlocal call_count
        call_count += 1
        return MOCK_LLM_ENTITY_RESPONSE

    extractor.llm_func = counting_llm_func

    # First call
    entities1 = await extractor.extract_entities_from_diagram(MOCK_DIAGRAM_DATA)
    assert call_count == 1

    # Second call should also hit LLM (no caching)
    entities2 = await extractor.extract_entities_from_diagram(MOCK_DIAGRAM_DATA)
    assert call_count == 2