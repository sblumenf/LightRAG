"""
Unit tests for the DiagramEntityExtractor class.
"""
import pytest
import asyncio
import json
import os
from unittest.mock import patch, MagicMock, AsyncMock
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