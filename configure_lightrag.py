import os
import asyncio
import numpy as np
from dotenv import load_dotenv
from lightrag import LightRAG
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete
from lightrag.llm.ollama import ollama_embedding, ollama_model_complete as ollama_complete

# Load environment variables from .env file
load_dotenv()

# Set up logging
setup_logger("lightrag", level="INFO")

# Create LightRAG instance with Neo4J and MongoDB
async def create_rag(llm_model_func=None, embedding_func=None):
    """Create a LightRAG instance with Neo4J and MongoDB storage."""

    # Verify Neo4J connection variables are loaded
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_username = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not all([neo4j_uri, neo4j_username, neo4j_password]):
        raise ValueError("Neo4J connection details missing from environment variables")

    # Verify MongoDB connection variables are loaded
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("MongoDB connection URI missing from environment variables")

    # Use default LLM and embedding functions if none provided
    if llm_model_func is None:
        # Try to use OpenAI if API key is set, otherwise use Ollama
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            print("Using OpenAI GPT-4o-mini for LLM")
            llm_model_func = gpt_4o_mini_complete
        else:
            print("Using Ollama for LLM (make sure Ollama is running locally)")
            llm_model_func = lambda prompt, **kwargs: ollama_complete(
                prompt, model="mistral", host="http://localhost:11434", **kwargs
            )

    if embedding_func is None:
        # Try to use OpenAI if API key is set, otherwise use Ollama
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            print("Using OpenAI embeddings")
            embedding_dim = 1536  # OpenAI embedding dimension
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=openai_embed
            )
        else:
            print("Using Ollama embeddings (make sure Ollama is running locally)")
            embedding_dim = 1024  # BGE-M3 embedding dimension
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=lambda texts: ollama_embedding(
                    texts, embed_model="bge-m3", host="http://localhost:11434"
                )
            )

    # Create LightRAG instance
    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        kv_storage=os.getenv("LIGHTRAG_KV_STORAGE", "MongoKVStorage"),
        vector_storage=os.getenv("LIGHTRAG_VECTOR_STORAGE", "MongoVectorDBStorage"),
        graph_storage=os.getenv("LIGHTRAG_GRAPH_STORAGE", "Neo4JStorage"),
        doc_status_storage=os.getenv("LIGHTRAG_DOC_STATUS_STORAGE", "MongoDocStatusStorage")
    )

    # Initialize all storage connections
    await rag.initialize_storages()
    await initialize_pipeline_status()

    print("LightRAG initialized with Neo4J and MongoDB storage")
    return rag

# Example usage with dummy functions for testing
async def test_with_dummy_functions():
    # Define dummy functions for testing
    async def dummy_llm(prompt, **kwargs):
        return "This is a dummy LLM response"

    async def dummy_embedding(texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), 1536)  # Random embedding vectors

    # Create dummy embedding function
    dummy_embed_func = EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=dummy_embedding
    )

    # Test the configuration
    try:
        rag = await create_rag(dummy_llm, dummy_embed_func)
        print("Configuration successful!")

        # Add a test document
        test_text = """
        LightRAG is a lightweight Knowledge Graph Retrieval-Augmented Generation system.
        It supports multiple LLM backends and provides efficient document processing.
        This is a test document to verify the configuration is working correctly.
        """

        doc_id = await rag.add_text(text=test_text, description="Test Document")
        print(f"Successfully added test document with ID: {doc_id}")

        # Clean up
        await rag.finalize_storages()
        return True
    except Exception as e:
        print(f"Configuration failed: {e}")
        return False

# Example usage with real functions
async def test_with_real_functions():
    try:
        # Create LightRAG with default functions (will use OpenAI or Ollama)
        rag = await create_rag()
        print("Configuration successful!")

        # Add a test document
        test_text = """
        LightRAG is a lightweight Knowledge Graph Retrieval-Augmented Generation system.
        It supports multiple LLM backends and provides efficient document processing.
        This is a test document to verify the configuration is working correctly.
        """

        doc_id = await rag.add_text(text=test_text, description="Test Document")
        print(f"Successfully added test document with ID: {doc_id}")

        # Clean up
        await rag.finalize_storages()
        return True
    except Exception as e:
        print(f"Configuration failed: {e}")
        return False

if __name__ == "__main__":
    # Choose which test to run
    use_real_functions = True  # Set to True to use real LLM and embedding functions

    if use_real_functions:
        asyncio.run(test_with_real_functions())
    else:
        asyncio.run(test_with_dummy_functions())
