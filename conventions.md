
**General Implementation Notes:**

*   **Code Structure:** Adapt features into a modular structure (e.g., document processing, knowledge graph management, retrieval, generation). Leverage base classes and utilities if building upon an existing framework like `LightRAG`.
*   **Error Handling:** Implement robust, specific error handling. Use `try...except` blocks for I/O, API calls, and library interactions. Catch specific exceptions (e.g., `neo4j.exceptions.Neo4jError`, `FileNotFoundError`, `google.api_core.exceptions`, `openai.error`, `JSONDecodeError`, library-specific errors like `fitz.fitz.FitzError`). Define custom exceptions (`PDFReadError`, `KnowledgeGraphError`, `EmbeddingError`, `LLMGenerationError`, `ConfigurationError`, `SchemaError`) for better control flow and reporting. Log errors clearly with context and stack traces where appropriate.
*   **Logging:** Utilize the standard `logging` module extensively.
    *   `DEBUG`: Detailed flow, variable states, intermediate results, API parameters (excluding secrets).
    *   `INFO`: Major steps initiated/completed, progress indicators, counts (nodes created, chunks processed, etc.).
    *   `WARNING`: Handled issues, potential problems (e.g., missing optional data, schema conflicts resolved, API retries, fallback strategies used).
    *   `ERROR`: Critical failures preventing a step/component from completing successfully.
    *   Ensure `logger = logging.getLogger(__name__)` is present in each module.
*   **Configurability:** Default parameters should be sourced from a central configuration mechanism (e.g., `config/settings.py` loaded via `python-dotenv`). All components requiring configuration should accept an optional `config: Dict[str, Any]` dictionary in their `__init__` method to override these defaults using the pattern: `self.param = config.get('config_key', settings.DEFAULT_SETTING)`. Avoid hardcoding values like thresholds, model names, limits, etc. Validate configurations on initialization where practical (e.g., checking if overlap < chunk size).
*   **Asynchronous Operations:** Use `async/await` for I/O-bound operations, especially database interactions (Neo4j async driver), LLM calls, and embedding API calls. Manage concurrency carefully (e.g., using `asyncio.Semaphore` or task groups).
*   **Testing:** Each deliverable should have associated tests.
    *   **Unit Tests:** Isolate components, mock external dependencies (databases, APIs, file system).
    *   **Integration Tests:** Test interactions between components and with real external systems (Neo4j instance, potentially stubbed APIs).
*   **Documentation:** Add clear Python docstrings (`"""Docstring"""`) to all new/modified classes, methods, and functions. Include type hints for parameters and return values. Update README and other relevant documentation (`docs/`) to reflect implemented features and usage.
*   **Style:** Adhere to PEP 8 guidelines. Use Python 3.9+ features and type hints consistently.
