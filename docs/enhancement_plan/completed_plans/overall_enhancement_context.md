
**Project Overview Summary (Context for AI Assistant)**

**Project:** Enhance the existing `LightRAG` Python project.

**Goal:** Implement advanced features detailed in `overview.txt` to create a more sophisticated RAG system. Key enhancement areas include:
1.  **Detailed Document Processing:** Improved extraction of text, metadata, tables, and handling of diagrams/formulas from source documents.
2.  **Schema-Driven KG Construction:** Building the Knowledge Graph based on a **pre-existing `schema.json`** for entity/relationship classification and validation.
3.  **Advanced Entity Resolution:** Merging duplicate entities using multiple strategies.
4.  **Intelligent Retrieval:** Processing queries to understand intent and dynamically select the best retrieval strategy (graph, vector, hybrid).
5.  **Sophisticated Generation:** Implementing Chain-of-Thought (CoT) reasoning and enhanced source citation in LLM responses.

**Key Implementation Principles:**
*   **Integration, Not Replacement:** Enhance the *existing* LightRAG framework. Do *not* build from scratch.
*   **Use LightRAG Abstractions:** Utilize LightRAG's existing base classes and interfaces (e.g., `BaseGraphStorage`, `BaseKVStorage`, `BaseVectorStorage`, `LightRAG.embedding_func`, `LightRAG.llm_model_func`).
*   **Backend Agnostic Core:** Core logic implemented in general modules (like `operate.py`, `lightrag.py`) must remain backend-agnostic. Backend-specific code belongs only in dedicated implementation files (e.g., `neo4j_impl.py`).
*   **Use Provided Schema:** A specific `schema.json` **is provided** and *must* be used to guide KG construction and validation tasks.
*   **Adapt Reference Code:** Code snippets from referenced `graphrag_tutor/*` files are for *context and inspiration only*. Adapt the logic to fit LightRAG's structure and interfaces. Do not copy directly if it violates backend agnosticism or existing structure.
*   **Configuration:** Integrate new settings (feature flags, thresholds, schema path) into LightRAG's existing `.env`/config system.
*   **Robustness:** Implement robust error handling, especially for external calls (LLMs, file processing).
*   **Mandatory Testing:** All newly generated or modified code *must* be accompanied by comprehensive unit and integration tests (`pytest`) that **pass cleanly without warnings or skips**.
