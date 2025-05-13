"""
Integration tests for diagram entity extraction pipeline.
"""
import os
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from document_processing.pdf_parser import process_pdf_document
from document_processing.diagram_analyzer import DiagramAnalyzer
from lightrag.schema.schema_validator import SchemaValidator


# Path to the test PDF file
TEST_PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'fixtures',
    'sample.pdf'
)

# Path to the test schema file
TEST_SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'docs',
    'schema.json'
)


class MockLLM:
    """Mock LLM class for testing."""
    
    async def __call__(self, prompt: str) -> str:
        """Mock LLM call that returns predefined responses based on the prompt content."""
        if "Extract named entities" in prompt:
            return """
            {
              "entities": [
                {
                  "entity_name": "Component1",
                  "entity_type": "Component",
                  "properties": {
                    "name": "Main Component"
                  },
                  "description": "The main component of the system"
                },
                {
                  "entity_name": "Service1",
                  "entity_type": "Service",
                  "properties": {
                    "name": "API Service",
                    "endpoint": "/api"
                  },
                  "description": "API service for external access"
                }
              ]
            }
            """
        elif "Extract relationships between entities" in prompt:
            return """
            {
              "relationships": [
                {
                  "source": "Component1",
                  "target": "Service1",
                  "type": "USES",
                  "description": "Component1 uses Service1 for data access",
                  "properties": {
                    "protocol": "HTTP"
                  }
                }
              ]
            }
            """
        return "{}"


# Fixture for the schema validator
@pytest.fixture
def schema_validator():
    """Create a schema validator for testing."""
    if os.path.exists(TEST_SCHEMA_PATH):
        return SchemaValidator(TEST_SCHEMA_PATH)
    
    # Use a mock if the schema file doesn't exist
    mock_validator = MagicMock()
    mock_validator.get_entity_types.return_value = ["Component", "Service"]
    mock_validator.get_relationship_types.return_value = ["USES", "CONTAINS"]
    mock_validator.validate_entity.return_value = (True, "")
    mock_validator.validate_relationship.return_value = (True, "")
    return mock_validator


# Fixture for the LLM function
@pytest.fixture
def llm_func():
    """Create an async LLM function for testing."""
    return MockLLM()


@pytest.mark.skipif(not os.path.exists(TEST_PDF_PATH), reason="Test PDF file not found")
def test_diagram_entity_extraction_integration(schema_validator, llm_func):
    """Test the integration of diagram entity extraction in the document processing pipeline."""
    # Create context with schema validator and LLM function
    context = {
        'schema_validator': schema_validator,
        'llm_func': llm_func
    }
    
    # Process the PDF document
    result = process_pdf_document(
        pdf_path=TEST_PDF_PATH,
        extract_diagrams=True,
        context=context
    )
    
    # Check that the result contains the expected fields
    assert 'extracted_elements' in result
    assert 'text_content' in result
    assert 'metadata' in result
    
    # Verify diagrams were extracted
    if 'diagrams' in result['extracted_elements']:
        # Expect diagram entities and relationships to be extracted
        if 'diagram_entities' in result['extracted_elements']:
            entities = result['extracted_elements']['diagram_entities']
            # Verify entities
            assert len(entities) > 0
            assert all(isinstance(entity, dict) for entity in entities)
            assert all('entity_name' in entity for entity in entities)
            assert all('entity_type' in entity for entity in entities)
            assert all('properties' in entity for entity in entities)
            
            # Verify entity metadata in PDF metadata
            assert 'diagram_entity_count' in result['metadata']
            assert result['metadata']['diagram_entity_count'] == len(entities)
        
        # Verify relationships
        if 'diagram_relationships' in result['extracted_elements']:
            relationships = result['extracted_elements']['diagram_relationships']
            assert len(relationships) > 0
            assert all(isinstance(rel, dict) for rel in relationships)
            assert all('source' in rel for rel in relationships)
            assert all('target' in rel for rel in relationships)
            assert all('type' in rel for rel in relationships)
            
            # Verify relationship metadata in PDF metadata
            assert 'diagram_relationship_count' in result['metadata']
            assert result['metadata']['diagram_relationship_count'] == len(relationships)


@pytest.mark.skipif(not os.path.exists(TEST_PDF_PATH), reason="Test PDF file not found")
@patch('document_processing.diagram_analyzer.DiagramAnalyzer.extract_diagrams_from_pdf')
def test_pipeline_with_mocked_diagrams(mock_extract_diagrams, schema_validator, llm_func):
    """Test the pipeline with mocked diagram extraction."""
    # Create mock diagrams
    mock_diagrams = [
        {
            'diagram_id': 'test-diagram-1',
            'page': 1,
            'width': 500,
            'height': 300,
            'format': 'png',
            'is_diagram': True,
            'position': [100, 100, 600, 400],
            'surrounding_text': 'This is a diagram of the system architecture.',
            'caption': 'System Architecture Diagram',
            'description': 'This diagram shows a system with Component1 connected to Service1.'
        }
    ]
    
    # Configure the mock to return our predefined diagrams
    mock_extract_diagrams.return_value = mock_diagrams
    
    # Create context with schema validator and LLM function
    context = {
        'schema_validator': schema_validator,
        'llm_func': llm_func
    }
    
    # Process the PDF document
    result = process_pdf_document(
        pdf_path=TEST_PDF_PATH,
        extract_diagrams=True,
        context=context
    )
    
    # Verify diagrams were processed
    assert 'extracted_elements' in result
    assert 'diagrams' in result['extracted_elements']
    assert len(result['extracted_elements']['diagrams']) == len(mock_diagrams)
    
    # Verify entities were extracted
    assert 'diagram_entities' in result['extracted_elements']
    entities = result['extracted_elements']['diagram_entities']
    assert len(entities) > 0
    
    # Verify relationships were extracted
    assert 'diagram_relationships' in result['extracted_elements']
    relationships = result['extracted_elements']['diagram_relationships']
    assert len(relationships) > 0
    
    # Verify entity and relationship metadata
    assert 'metadata' in result
    assert 'diagram_entity_count' in result['metadata']
    assert 'diagram_relationship_count' in result['metadata']


@pytest.mark.skipif(not os.path.exists(TEST_PDF_PATH), reason="Test PDF file not found")
def test_pipeline_with_no_schema_or_llm():
    """Test that the pipeline handles missing schema or LLM gracefully."""
    # Process the PDF document without providing schema or LLM
    result = process_pdf_document(
        pdf_path=TEST_PDF_PATH,
        extract_diagrams=True,
        context={}  # Empty context
    )
    
    # Verify diagrams were processed but no entities were extracted
    if 'diagrams' in result['extracted_elements']:
        assert 'diagram_entities' not in result['extracted_elements']
        assert 'diagram_relationships' not in result['extracted_elements']


@pytest.mark.skipif(not os.path.exists(TEST_PDF_PATH), reason="Test PDF file not found")
def test_pipeline_with_diagram_extraction_disabled():
    """Test that diagram entity extraction is skipped when diagram extraction is disabled."""
    # Create context with schema validator and LLM function
    context = {
        'schema_validator': MagicMock(),
        'llm_func': MagicMock()
    }
    
    # Process the PDF document with diagram extraction disabled
    result = process_pdf_document(
        pdf_path=TEST_PDF_PATH,
        extract_diagrams=False,  # Disable diagram extraction
        context=context
    )
    
    # Verify no diagrams or entities were extracted
    assert 'diagrams' not in result['extracted_elements']
    assert 'diagram_entities' not in result['extracted_elements']
    assert 'diagram_relationships' not in result['extracted_elements']