import os
import asyncio
from lightrag import LightRAG, QueryParam
# Make sure both LLM function and embedding function are imported
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
# Import EmbeddingFunc
from lightrag.utils import EmbeddingFunc
import traceback # Import traceback for detailed error printing

# Defines the directory for cache files etc. (Postgres/Neo4j data goes to databases)
WORKING_DIR = "./local_neo4j_pg_WorkDir"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Keep initialize_rag async
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        # Configure OpenAI embedding function
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=openai_embed
        ),
        # Configure storage backends
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        graph_storage="Neo4JStorage",
    )

    # This will attempt connections to Neo4j & Postgres based on ENV variables
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

# --- main remains an async function ---
async def main():
    # Initialize RAG instance
    print("Initializing LightRAG...")
    rag = await initialize_rag()
    print("LightRAG Initialized.")

    # Make sure book.txt exists
    book_path = "./book.txt"
    if not os.path.exists(book_path):
        print(f"Error: {book_path} not found.")
        print("Please download it using:")
        print("curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt > ./book.txt")
        return # Exit if book.txt is not found

    # Process the document using await with the async insert method
    print("Starting document insertion (this may take a while)...")
    try:
        with open(book_path, "r", encoding="utf-8") as f:
            # --- Use await with ainsert AND pass file_paths ---
            await rag.ainsert(f.read(), file_paths=[book_path]) # <-- Corrected line
        print("Document insertion finished.")
    except AttributeError:
        # Fallback if ainsert truly doesn't exist (unlikely given previous warnings)
        print("Warning: rag.ainsert not found. Falling back to synchronous insert.")
        print("If errors occur later, LightRAG's async handling might need review.")
        try:
             with open(book_path, "r", encoding="utf-8") as f:
                # Add file_paths here too for consistency if falling back
                rag.insert(f.read(), file_paths=[book_path])
             print("Document insertion finished (sync).")
        except Exception as e:
            print(f"An error occurred during synchronous insertion: {e}")
            traceback.print_exc()
            return # Exit if insertion fails
    except Exception as e:
        print(f"An error occurred during async insertion: {e}")
        traceback.print_exc()
        return # Exit if insertion fails


    # Perform queries using await with the async query method
    print("\n--- Running Queries ---")
    query_text = "What are the top themes in this story?"

    try:
        # --- Use await with aquery for all queries ---
        print("\n--- Naive Query ---")
        result_naive = await rag.aquery(query_text, param=QueryParam(mode="naive"))
        print(result_naive)

        print("\n--- Local Query ---")
        result_local = await rag.aquery(query_text, param=QueryParam(mode="local"))
        print(result_local)

        print("\n--- Global Query ---")
        result_global = await rag.aquery(query_text, param=QueryParam(mode="global"))
        print(result_global)

        print("\n--- Hybrid Query ---")
        result_hybrid = await rag.aquery(query_text, param=QueryParam(mode="hybrid"))
        print(result_hybrid)

    except Exception as e:
        print(f"\nAn error occurred during querying: {e}")
        print("Traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function using asyncio.run()
    try:
        asyncio.run(main())
    except RuntimeError as e:
        print(f"Caught RuntimeError at top level: {e}")