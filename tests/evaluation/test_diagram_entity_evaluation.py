"""
Unit tests for diagram entity evaluation metrics.
"""
import os
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from lightrag.evaluation.diagram_entity_evaluation import (
    DiagramEntityEvaluator,
    evaluate_diagram_entity_extraction
)


# Path to the test dataset
TEST_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    'test_diagram_dataset.json'
)


class MockDiagramAnalyzer:
    """Mock DiagramAnalyzer for testing."""
    
    async def extract_entities_and_relationships(self, diagram, schema_validator, llm_func):
        """Mock extraction that returns predefined entities and relationships."""
        # For the first architecture diagram
        if diagram.get('diagram_id') == 'test-arch-diagram-1':
            return [
                {
                    "entity_name": "Frontend",
                    "entity_type": "Component",
                    "properties": {
                        "name": "Frontend UI"
                    },
                    "description": "Frontend user interface component"
                },
                {
                    "entity_name": "BackendAPI",
                    "entity_type": "Service",
                    "properties": {
                        "name": "Backend API",
                        "endpoint": "/api"
                    },
                    "description": "Backend API service"
                }
            ], [
                {
                    "source": "Frontend",
                    "target": "BackendAPI",
                    "type": "CALLS",
                    "description": "Frontend calls Backend API",
                    "properties": {
                        "protocol": "REST"
                    }
                }
            ]
        
        # For the second architecture diagram
        elif diagram.get('diagram_id') == 'test-arch-diagram-2':
            return [
                {
                    "entity_name": "AuthService",
                    "entity_type": "Service",
                    "properties": {
                        "name": "Auth Service",
                        "endpoint": "/auth"
                    },
                    "description": "Authentication service"
                },
                {
                    "entity_name": "UserService",
                    "entity_type": "Service",
                    "properties": {
                        "name": "User Service",
                        "endpoint": "/users"
                    },
                    "description": "User management service"
                }
            ], [
                {
                    "source": "UserService",
                    "target": "AuthService",
                    "type": "DEPENDS_ON",
                    "description": "UserService depends on AuthService",
                    "properties": {}
                }
            ]
        
        # For the flowchart diagram
        elif diagram.get('diagram_id') == 'test-flow-diagram-1':
            return [
                {
                    "entity_name": "LoginRequest",
                    "entity_type": "Process",
                    "properties": {
                        "name": "Login Request"
                    },
                    "description": "Process login request step"
                },
                {
                    "entity_name": "ValidateCredentials",
                    "entity_type": "Process",
                    "properties": {
                        "name": "Validate Credentials"
                    },
                    "description": "Validate user credentials step"
                },
                {
                    "entity_name": "GenerateToken",
                    "entity_type": "Process",
                    "properties": {
                        "name": "Generate Token"
                    },
                    "description": "Generate authentication token step"
                }
            ], [
                {
                    "source": "LoginRequest",
                    "target": "ValidateCredentials",
                    "type": "NEXT",
                    "description": "After login request, validate credentials",
                    "properties": {}
                },
                {
                    "source": "ValidateCredentials",
                    "target": "GenerateToken",
                    "type": "NEXT_IF",
                    "description": "If credentials are valid, generate token",
                    "properties": {
                        "condition": "valid"
                    }
                }
            ]
        
        # Default case
        return [], []


@pytest.fixture
def mock_analyzer():
    """Fixture for mock diagram analyzer."""
    return MockDiagramAnalyzer()


@pytest.fixture
def mock_schema_validator():
    """Fixture for mock schema validator."""
    mock = MagicMock()
    mock.get_entity_types.return_value = ['Component', 'Service', 'DataStore', 'Process']
    mock.get_relationship_types.return_value = ['CALLS', 'USES', 'DEPENDS_ON', 'NEXT', 'NEXT_IF']
    return mock


@pytest.fixture
def mock_llm_func():
    """Fixture for mock LLM function."""
    async def mock_func(prompt):
        return "{}"
    return mock_func


@pytest.fixture
def test_datasets():
    """Load test datasets from file."""
    if os.path.exists(TEST_DATASET_PATH):
        with open(TEST_DATASET_PATH, 'r') as f:
            return json.load(f)
    return {}


@pytest.mark.asyncio
async def test_entity_metrics_calculation():
    """Test calculation of entity metrics."""
    evaluator = DiagramEntityEvaluator()
    
    # Test with perfect match
    extracted = [
        {"entity_name": "Entity1", "entity_type": "Type1"},
        {"entity_name": "Entity2", "entity_type": "Type2"}
    ]
    ground_truth = [
        {"entity_name": "Entity1", "entity_type": "Type1"},
        {"entity_name": "Entity2", "entity_type": "Type2"}
    ]
    
    precision, recall, f1 = evaluator._calculate_entity_metrics(extracted, ground_truth)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
    
    # Test with partial match
    extracted = [
        {"entity_name": "Entity1", "entity_type": "Type1"},
        {"entity_name": "Entity3", "entity_type": "Type3"}
    ]
    ground_truth = [
        {"entity_name": "Entity1", "entity_type": "Type1"},
        {"entity_name": "Entity2", "entity_type": "Type2"}
    ]
    
    precision, recall, f1 = evaluator._calculate_entity_metrics(extracted, ground_truth)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5
    
    # Test with no match
    extracted = [
        {"entity_name": "Entity3", "entity_type": "Type3"},
        {"entity_name": "Entity4", "entity_type": "Type4"}
    ]
    ground_truth = [
        {"entity_name": "Entity1", "entity_type": "Type1"},
        {"entity_name": "Entity2", "entity_type": "Type2"}
    ]
    
    precision, recall, f1 = evaluator._calculate_entity_metrics(extracted, ground_truth)
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0
    
    # Test with empty extraction
    extracted = []
    ground_truth = [
        {"entity_name": "Entity1", "entity_type": "Type1"},
        {"entity_name": "Entity2", "entity_type": "Type2"}
    ]
    
    precision, recall, f1 = evaluator._calculate_entity_metrics(extracted, ground_truth)
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0
    
    # Test with empty ground truth
    extracted = [
        {"entity_name": "Entity1", "entity_type": "Type1"},
        {"entity_name": "Entity2", "entity_type": "Type2"}
    ]
    ground_truth = []
    
    precision, recall, f1 = evaluator._calculate_entity_metrics(extracted, ground_truth)
    assert precision == 0.0
    assert recall == 0.0
    assert f1 == 0.0


@pytest.mark.asyncio
async def test_relationship_metrics_calculation():
    """Test calculation of relationship metrics."""
    evaluator = DiagramEntityEvaluator()
    
    # Test with perfect match
    extracted = [
        {"source": "Entity1", "target": "Entity2", "type": "CALLS"},
        {"source": "Entity2", "target": "Entity3", "type": "USES"}
    ]
    ground_truth = [
        {"source": "Entity1", "target": "Entity2", "type": "CALLS"},
        {"source": "Entity2", "target": "Entity3", "type": "USES"}
    ]
    
    precision, recall, f1 = evaluator._calculate_relationship_metrics(extracted, ground_truth)
    assert precision == 1.0
    assert recall == 1.0
    assert f1 == 1.0
    
    # Test with partial match
    extracted = [
        {"source": "Entity1", "target": "Entity2", "type": "CALLS"},
        {"source": "Entity3", "target": "Entity4", "type": "DEPENDS_ON"}
    ]
    ground_truth = [
        {"source": "Entity1", "target": "Entity2", "type": "CALLS"},
        {"source": "Entity2", "target": "Entity3", "type": "USES"}
    ]
    
    precision, recall, f1 = evaluator._calculate_relationship_metrics(extracted, ground_truth)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


@pytest.mark.asyncio
async def test_evaluate_extraction(mock_analyzer, mock_schema_validator, mock_llm_func, test_datasets):
    """Test evaluation of extraction on a test dataset."""
    # Skip if test dataset is not available
    if not test_datasets:
        pytest.skip("Test dataset not available")
        
    evaluator = DiagramEntityEvaluator(mock_schema_validator)
    
    # Run evaluation
    results = await evaluator.evaluate_extraction(mock_analyzer, test_datasets, mock_llm_func)
    
    # Check that results contain expected fields
    assert 'entity_metrics' in results
    assert 'relationship_metrics' in results
    assert 'performance_metrics' in results
    assert 'details' in results
    
    # Check that metrics contain expected fields
    assert 'precision' in results['entity_metrics']
    assert 'recall' in results['entity_metrics']
    assert 'f1_score' in results['entity_metrics']
    assert 'extraction_rate' in results['entity_metrics']
    
    assert 'precision' in results['relationship_metrics']
    assert 'recall' in results['relationship_metrics']
    assert 'f1_score' in results['relationship_metrics']
    assert 'extraction_rate' in results['relationship_metrics']
    
    assert 'avg_processing_time' in results['performance_metrics']
    assert 'total_diagrams' in results['performance_metrics']
    assert 'total_entities_extracted' in results['performance_metrics']
    assert 'total_relationships_extracted' in results['performance_metrics']
    
    # Check that details contain results for each dataset
    for dataset_name in test_datasets.keys():
        assert dataset_name in results['details']
        assert 'diagrams' in results['details'][dataset_name]
        assert 'metrics' in results['details'][dataset_name]


@pytest.mark.asyncio
@patch('lightrag.evaluation.diagram_entity_evaluation.DiagramEntityEvaluator')
async def test_evaluate_diagram_entity_extraction(mock_evaluator_class, 
                                                mock_analyzer, mock_schema_validator, 
                                                mock_llm_func, tmp_path):
    """Test the main evaluation function."""
    # Create a mock evaluator instance
    mock_evaluator = MagicMock()
    mock_evaluator.evaluate_extraction.return_value = {'test': 'results'}
    mock_evaluator.save_evaluation_results = MagicMock()
    mock_evaluator_class.return_value = mock_evaluator
    
    # Create a temporary test dataset file
    dataset_path = tmp_path / "test_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump({"test": {"diagrams": []}}, f)
    
    # Create a temporary output file
    output_path = tmp_path / "test_results.json"
    
    # Run the evaluation function
    results = await evaluate_diagram_entity_extraction(
        dataset_path, mock_analyzer, mock_schema_validator, mock_llm_func, output_path
    )
    
    # Check that the evaluator was created with the correct arguments
    mock_evaluator_class.assert_called_once_with(mock_schema_validator)
    
    # Check that evaluate_extraction was called with the correct arguments
    mock_evaluator.evaluate_extraction.assert_called_once()
    
    # Check that save_evaluation_results was called with the correct arguments
    mock_evaluator.save_evaluation_results.assert_called_once_with({'test': 'results'}, output_path)
    
    # Check that the function returned the expected results
    assert results == {'test': 'results'}