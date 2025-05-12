"""
Pytest configuration and fixtures for LightRAG tests.
"""

import os
import json
import pytest
import asyncio
import tempfile
import shutil
import warnings
from typing import Dict, Any, Callable, Optional

import pytest_asyncio

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*swigvarlink.*")
warnings.filterwarnings("ignore", message=".*SwigPyPacked.*")
warnings.filterwarnings("ignore", message=".*SwigPyObject.*")
warnings.filterwarnings("ignore", message=".*importlib._bootstrap.*")
warnings.filterwarnings("ignore", message=".*NumPy module was reloaded.*")

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.schema_utils import load_schema

# Set up logging for tests
setup_logger("lightrag", level="INFO")

# Default schema path
DEFAULT_SCHEMA_PATH = os.path.abspath(os.getenv("SCHEMA_FILE_PATH", "docs/schema.json"))

# Dummy functions for testing
async def dummy_llm_func(prompt: str, **kwargs) -> str:
    """Dummy LLM function that returns a simple response."""
    return f"Response to: {prompt[:30]}..."

async def dummy_embedding_func(texts: list[str]) -> list[list[float]]:
    """Dummy embedding function that returns fixed-size vectors."""
    # Return a simple deterministic embedding (all zeros with the first element being the hash of the text)
    return [[hash(text) % 100] + [0.0] * 767 for text in texts]


# We don't need to define event_loop fixture as pytest-asyncio provides it


@pytest.fixture
def sample_doc_path() -> str:
    """Return the path to the sample document."""
    # Use a path relative to the project root
    return os.path.abspath("tests/fixtures/sample_doc.txt")


@pytest.fixture
def sample_doc_content(sample_doc_path: str) -> str:
    """Return the content of the sample document."""
    with open(sample_doc_path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def schema_path() -> str:
    """Return the path to the schema file."""
    return DEFAULT_SCHEMA_PATH


@pytest.fixture
def schema(schema_path: str) -> Dict[str, Any]:
    """Load and return the schema."""
    return load_schema(schema_path)


@pytest_asyncio.fixture
async def temp_working_dir():
    """Create a temporary working directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="lightrag_test_")
    yield temp_dir
    # Clean up after the test
    shutil.rmtree(temp_dir)


@pytest_asyncio.fixture
async def lightrag_instance(temp_working_dir: str):
    """Create a LightRAG instance with file-based storage for testing."""
    # Create a LightRAG instance with file-based storage
    rag = LightRAG(
        working_dir=temp_working_dir,
        llm_model_func=dummy_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=dummy_embedding_func,
        ),
        # Use file-based storage implementations
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage"
    )

    # Initialize storages
    await rag.initialize_storages()
    await initialize_pipeline_status()

    yield rag

    # Clean up
    await rag.finalize_storages()


@pytest_asyncio.fixture
async def lightrag_with_sample_doc(lightrag_instance: LightRAG, sample_doc_content: str):
    """Create a LightRAG instance with the sample document loaded."""
    # Add the sample document using ainsert
    await lightrag_instance.ainsert(sample_doc_content)

    # For testing purposes, we don't need the actual document ID
    # We'll just return the instance and a placeholder ID
    return lightrag_instance, "sample_doc_id"


@pytest.fixture
def mock_llm_func():
    """Return a mock LLM function for testing."""
    from unittest.mock import AsyncMock

    mock_func = AsyncMock()
    mock_func.return_value = """
    <reasoning>
    Based on the provided context, I can see that LightRAG is a lightweight Knowledge Graph
    Retrieval-Augmented Generation system [Entity ID: node1]. It supports multiple LLM backends
    and provides efficient document processing [Entity ID: node2].
    </reasoning>
    <answer>
    LightRAG is a lightweight Knowledge Graph RAG system that supports multiple LLM backends
    and provides efficient document processing.
    </answer>
    """

    return mock_func


@pytest.fixture
def sample_context_items():
    """Return sample context items for testing."""
    return [
        {
            "id": "node1",
            "content": "LightRAG is a lightweight Knowledge Graph Retrieval-Augmented Generation system."
        },
        {
            "id": "node2",
            "content": "It supports multiple LLM backends and provides efficient document processing."
        },
        {
            "id": "node3",
            "content": "LightRAG includes diagram and formula handling capabilities.",
            "extracted_elements": {
                "diagrams": [
                    {
                        "diagram_id": "diagram-1",
                        "description": "System architecture diagram",
                        "caption": "Figure 1: System Architecture"
                    }
                ],
                "formulas": [
                    {
                        "formula_id": "formula-1",
                        "formula": "E = mc^2",
                        "description": "Einstein's mass-energy equivalence"
                    }
                ]
            }
        }
    ]


@pytest.fixture
def mock_config():
    """Return a mock configuration for testing."""
    from unittest.mock import MagicMock
    from lightrag.config_loader import EnhancedConfig

    config = MagicMock(spec=EnhancedConfig)
    config.enable_cot = True
    config.enable_enhanced_citations = True
    config.enable_diagram_formula_integration = True
    config.resolve_placeholders_in_context = True
    config.max_cot_refinement_attempts = 2

    return config
