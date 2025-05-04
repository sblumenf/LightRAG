```text
# Enhanced LightRAG Implementation Plan (AI Assistant Guided with Incremental & Mandatory Testing)

## Project Goal:
Enhance the existing LightRAG system by implementing the features detailed in `overview.txt`, utilizing a **pre-existing `schema.json`**, focusing on improved document processing, schema-driven KG construction, advanced entity resolution, intelligent retrieval, sophisticated generation (CoT), and ensuring **full, passing test coverage** at each phase.

## AI Assistant Usage Guidelines:
*   **Provide Full Context:** For each prompt, provide the AI with:
    *   The prompt text itself.
    *   The *content* of any referenced script files (e.g., `graphrag_tutor/document_processing/pdf_parser.py`).
    *   Relevant snippets of existing LightRAG code (e.g., class definitions, function signatures, data structures from `lightrag/base.py`, `lightrag/operate.py`, relevant storage implementations).
    *   The **existing `schema.json`** file.
    *   Clear instructions on the desired output format or interface modifications.
*   **Iterative Development & Testing:** Implement features in small, testable units. Use the AI to generate code *and* corresponding test cases.
*   **Critical Review:** Treat AI-generated code *and* tests as a draft. Review carefully for correctness, efficiency, style, **robust error handling**, and adherence to LightRAG's **multi-backend architecture**. Refactor as needed.
*   **Mandatory Testing:** **All generated unit and integration tests must pass without warnings or skips before moving to the next task or phase.** Ensure comprehensive coverage of functionality and edge cases. Supplement mock LLM tests with targeted tests using real LLM calls during integration/E2E phases.
*   **Adapt, Don't Just Copy:** Guide the AI to adapt logic into LightRAG's framework, using its existing interfaces (`BaseGraphStorage`, etc.). Ensure generated core logic remains **backend-agnostic**.

## Recommendation:
Work on a dedicated feature branch in your forked Git repository (e.g., `git checkout -b feature/enhanced-pipeline`).

---
**Phase 0: Foundation, Configuration & Test Setup**
---

*   **Objective:** Configure the system to use the existing schema, set up configuration management for new features, and establish the testing framework.

*   **Task 0.1: Configuration Management for Schema & New Features**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: LightRAG uses `.env`/`config.ini`. Need to add config options: path to existing `schema.json`, **boolean flags** for enabling/disabling diagram/formula analysis, CoT enablement, **configurable thresholds** for entity resolution similarity scores. Ref script `config/settings.py`.
        Task: 1. Propose method (extend `.env` preferred). 2. Generate Python code for loading existing + new config (env overrides file), including loading the schema path setting, **feature flags**, and **thresholds**.
        Output: 1. Proposal text. 2. Python config loading code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Choose approach, integrate loading code. Update `.env.example` with new **configurable** options (including `SCHEMA_PATH`). Place the existing `schema.json` where specified in the config.
    *   **Testing:** Implement and verify **passing** unit tests for config loading (schema path, flags, thresholds, defaults, overrides). **Ensure no test warnings or skips.**
    *   *AI Guidance (Testing):* "Generate `pytest` unit tests for the configuration loading code. Tests must cover loading `SCHEMA_PATH`, feature flags, thresholds, defaults, and environment variable overrides. Tests must pass cleanly."

*   **Task 0.2: Schema Loading Utility**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Need to load `schema.json` using the path from config.
        Task: Generate Python function `load_schema(schema_path: str) -> dict`.
        Requirements: Read JSON from path, parse, return dict, handle file not found/invalid JSON errors robustly.
        Output: Python function code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Add this utility function (e.g., to `schema_utils.py`).
    *   **Testing:** Implement and verify **passing** unit tests for `load_schema` (valid path, non-existent path, invalid JSON). **Ensure no test warnings or skips.**

*   **Task 0.3: Basic Test Setup**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Need a `pytest` structure for LightRAG.
        Task: Generate basic `pytest` setup.
        Requirements: Include `pytest.ini`, `tests/` structure, `conftest.py` with fixtures for sample text file loading and basic `LightRAG` instance initialization (using file-based storage and **loading the actual `schema.json` via the updated config**).
        Output: File structure and Python code for `pytest.ini`, `tests/conftest.py`, placeholder for `tests/fixtures/sample_doc.txt`.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Set up directories/files. Create sample fixture file. Ensure fixtures load config and schema correctly.
    *   **Testing:** Run `pytest`. Verify setup and fixtures work without errors. **Address any warnings immediately.**

---
**Phase 1: Enhanced Document Processing & Ingestion**
---

*   **Objective:** Improve data quality and structure from source documents.
*   **Integration Strategy:** Functions created here will likely form a new `document_processor` module. Modify LightRAG's ingestion pipeline (`apipeline_...` methods) to call the main function of this module (e.g., `process_document(file_path)`), passing its output dictionary (`text_content`, `metadata`, `extracted_elements`) to the chunking phase.

*   **Task 1.1: Advanced PDF Parsing**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Need robust PDF text extraction preserving order. Ref script: `graphrag_tutor/document_processing/pdf_parser.py`. Use `PyMuPDF` (fitz).
        Task: Generate Python function `extract_structured_text_from_pdf(pdf_path: str) -> str`.
        Requirements: Use `fitz`, extract text preserving structure, return single string, basic error handling.
        Output: Python function code with docstrings.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate into `document_processing` module.
    *   **Testing:** Implement and verify **passing** unit tests using sample PDFs (simple, multi-column, headers/footers). Verify accuracy/order. Test invalid/corrupted PDFs. **Ensure no test warnings or skips.**
    *   *AI Guidance (Testing):* "Generate `pytest` unit tests for `extract_structured_text_from_pdf`. Include simple, multi-column, invalid PDF cases. Tests must pass cleanly."

*   **Task 1.2: Metadata Extraction**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Need PDF and file system metadata. Ref script: `pdf_parser.py`.
        Task: Enhance/wrap into `process_pdf_document(pdf_path)` returning `{'text_content': str, 'metadata': dict}`.
        Requirements: Use `fitz` (PDF meta), `os` (file meta). Normalize dates (ISO 8601).
        Output: Updated Python function code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate. Define final metadata keys.
    *   **Testing:** Implement and verify **passing** unit tests for `process_pdf_document`, checking metadata presence/correctness. **Ensure no test warnings or skips.**

*   **Task 1.3: Content Filtering**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Remove headers, footers, page numbers, TOC/Index. Ref script: `content_filter.py`.
        Task: Generate Python function `filter_extracted_text(text_content: str) -> str`.
        Requirements: Use regex/layout analysis to remove common patterns.
        Output: Python function code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate post-extraction. Refine patterns.
    *   **Testing:** Implement and verify **passing** unit tests for `filter_extracted_text` using samples with headers, footers, page numbers, TOCs. Verify removal. **Ensure no test warnings or skips.**
    *   *AI Guidance (Testing):* "Generate `pytest` unit tests for `filter_extracted_text` covering headers, footers, page numbers, TOCs. Tests must pass cleanly."

*   **Task 1.4: Table Extraction**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Extract tables to Markdown. Ref script: `table_extractor.py`. Use `pdfplumber`.
        Task: Generate Python function `extract_tables_to_markdown(pdf_path: str) -> list[str]`.
        Requirements: Use `pdfplumber`, `page.extract_tables()`, format as Markdown, handle None cells, return list of Markdown strings.
        Output: Python function code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Install `pdfplumber`. Integrate. Define storage strategy (e.g., add to `extracted_elements` dict).
    *   **Testing:** Implement and verify **passing** unit tests for `extract_tables_to_markdown` with various PDFs. Verify count and Markdown validity. **Ensure no test warnings or skips.**
    *   *AI Guidance (Testing):* "Generate `pytest` unit tests for `extract_tables_to_markdown` covering PDFs with single, multiple, and no tables. Tests must pass cleanly."

*   **Task 1.5: Diagram/Formula Placeholders (Initial)**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Detect diagrams/formulas, replace with placeholders. Ref scripts: `diagram_analyzer.py`, `formula_extractor.py`. Use `PyMuPDF` (fitz).
        Task: Modify `process_pdf_document` for placeholder replacement.
        Requirements: Use `page.get_images()`. Basic regex for formulas. Replace with unique placeholders. Return dict with `'text_content'`, `'metadata'`, `'extracted_elements'` (mapping ID to raw data/text/extracted table markdown). **Include extracted tables from Task 1.4.**
        Output: Updated Python function code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate. Test detection. Ensure tables are added to `extracted_elements`.
    *   **Testing:** Implement and verify **passing** unit tests for `process_pdf_document`. Verify placeholders in `text_content`, corresponding entries (images, formulas, tables) in `extracted_elements`. Check counts. **Ensure no test warnings or skips.**

---
**Phase 2: Schema-Driven Knowledge Graph Construction**
---

*   **Objective:** Build a structured KG using the **existing schema**.
*   **Robustness Note:** Implement graceful handling for LLM failures or invalid outputs during classification and relationship extraction.

*   **Task 2.1: Text Chunking Refinement**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Enhance LightRAG's chunking (`operate.py`). Ref script: `text_chunker.py`.
        Task: Refactor `chunking_by_token_size` function.
        Requirements: Add `chunking_strategy` ('token', 'paragraph'). Implement paragraph splitting. Ensure output dict includes 'tokens', 'content', 'chunk_order_index', 'full_doc_id', 'file_path'. **Pass `extracted_elements` placeholders through.**
        Output: Refactored Python function code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Replace existing function. Ensure placeholder handling.
    *   **Testing:** Implement and verify **passing** unit tests for refactored chunking (strategies, metadata, placeholders). **Ensure no test warnings or skips.**

*   **Task 2.2: Embedding Generation (Adaptation)**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Ensure embeddings are generated for new chunks. Ref script: `embedding_generator.py`.
        Task: Review LightRAG pipeline (`operate.py`). Confirm `LightRAG.embedding_func` is called with 'content' from the *new* chunking output.
        Output: Confirmation or description.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Verify flow.
    *   **Testing:** (Covered by integration tests in Task 2.4)

*   **Task 2.3: Schema-Based Classification & Property Extraction**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Classify chunks using **existing `schema.json`** and extract schema-defined properties. Ref script: `schema_classifier.py`. Provide `schema.json`.
        Task: Generate async `classify_chunk_and_extract_properties(chunk_text: str, schema: dict, llm_func: callable) -> dict`.
        Requirements: Prompt LLM: 1. Identify primary entity type from schema. 2. Extract *only* schema-defined properties for that type. Return `{'entity_type': '...', 'properties': {'prop1': 'val1', ...}}`. **Handle LLM errors/invalid responses gracefully (return default `{'entity_type': 'UNKNOWN', 'properties': {}}`)**.
        Output: Python async function code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate into KG pipeline.
    *   **Testing:** Implement and verify **passing** unit tests with mock LLM. Test various chunks (matching, non-matching, missing props). **Verify error handling.** **Ensure no test warnings or skips.**

*   **Task 2.4: Graph Creation with Schema Validation**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Modify graph building (`operate.py`) to use schema. Ref scripts: `neo4j_knowledge_graph.py` (context only), `operate.py`. Provide `schema.json`. Use `BaseGraphStorage`.
        Task: Refactor `extract_entities` and `merge_nodes_and_edges`.
        Requirements: 1. `extract_entities`: Call `classify_chunk...`. Use separate LLM call for *relationship extraction*, validating type against schema. Pass classified entity data and *validated* relations. **Handle LLM/validation failures gracefully (log, skip invalid).** 2. `merge_nodes_and_edges`: Use classified `entity_type`/properties for `upsert_node`. Validate `relationship_type` before `upsert_edge`. Use only `BaseGraphStorage` methods.
        Output: Refactored Python code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate refactored code.
    *   **Testing:** Implement and verify **passing** integration tests for KG building. Input docs, run pipeline (mock LLM ok). Assert nodes have correct schema types/props, edges have valid types. **Assert invalid relationships are NOT created.** **Ensure no test warnings or skips.**

---
**Phase 3: Advanced KG Refinement**
---

*   **Objective:** Improve KG quality via entity resolution and index sync.
*   **Performance Note:** Consider strategies for large graphs (indexing, subsetting).

*   **Task 3.1: Entity Similarity Functions**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Need functions to compare entities. Ref script: `entity_resolver.py`.
        Task: Generate standalone Python functions: `calculate_name_similarity`, `calculate_embedding_similarity`, `match_by_context`.
        Requirements: Use libraries (`fuzzywuzzy`, `numpy`). Basic context matching (Jaccard index).
        Output: Python code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Test functions.
    *   **Testing:** Implement and verify **passing** unit tests for each similarity function with diverse inputs. **Ensure no test warnings or skips.**

*   **Task 3.2: Entity Resolver Class Structure & Merging Logic**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Design `EntityResolver` class. Ref script: `entity_resolver.py`.
        Task: Generate `EntityResolver` structure.
        Requirements: `__init__`, `find_duplicates` (using Task 3.1), `select_primary` (basic scoring), property merging helpers.
        Output: Python class definition with signatures, basic implementation/pseudocode.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Refine scoring. Implement merge helpers.
    *   **Testing:** Implement and verify **passing** unit tests for merge helpers and `select_primary`. **Ensure no test warnings or skips.**

*   **Task 3.3: Entity Merging Implementation**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Implement merging in `EntityResolver`. Ref script: `entity_resolver.py`. Use LightRAG interfaces.
        Task: Implement `merge_entities(self, primary_entity_id, duplicate_entity_ids)`.
        Requirements: Fetch data, merge properties, update primary (`upsert_node`), transfer relationships (`upsert_edge`), delete duplicates (`delete_node`, `entity_vdb.delete`), update primary VDB entry (`entity_vdb.upsert`). **Include detailed logging.**
        Output: Python code for `merge_entities`.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate `EntityResolver`. Decide trigger.
    *   **Testing:** Implement and verify **passing** integration tests for `merge_entities` with duplicates. Assert duplicate removal, property merging, relationship transfer, VDB updates, check logs. **Ensure no test warnings or skips.**

*   **Task 3.4: Index Synchronization**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Graph changes must update VDBs. Ref script: `kg_index_sync.py`.
        Task: Modify graph interaction methods (`operate.py` or storage impls) for VDB consistency.
        Requirements: Add VDB `upsert`/`delete` calls after graph ops. Ensure embeddings generated/updated. Include in `merge_entities` and deletion methods.
        Output: Modified Python code snippets.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate sync calls.
    *   **Testing:** Implement and verify **passing** integration tests extending previous tests (KG building, resolution, deletion) to assert VDB state after graph ops. **Ensure no test warnings or skips.**

---
**Phase 4: Intelligent Retrieval**
---

*   **Objective:** Enhance retrieval via query understanding and dynamic strategy selection.

*   **Task 4.1: Query Processing Module**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Analyze user queries. Ref script: `retriever.py`.
        Task: Generate async `process_query(query_text: str, llm_func: callable) -> dict`.
        Requirements: Use LLM: extract Intent (predefined list), Key Entities, Keywords, optional Expanded Terms. Return dict. **Handle LLM errors gracefully.**
        Output: Python async function code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Define intents. Test. Integrate.
    *   **Testing:** Implement and verify **passing** unit tests with mock LLM, various query types, verify components and error handling. **Ensure no test warnings or skips.**

*   **Task 4.2: Retrieval Strategy Selection**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Select strategy based on processed query. Ref scripts: `retriever.py`, `hybrid_retrieval.py`.
        Task: Generate `select_retrieval_strategy(processed_query_data: dict) -> str`.
        Requirements: Rule-based logic (if/else) on intent, entities, keywords. Return strategy name ('local', 'global', 'hybrid', 'naive', 'mix'). Define default.
        Output: Python function code.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Refine rules. Test.
    *   **Testing:** Implement and verify **passing** unit tests with various simulated inputs, assert correct strategy output. **Ensure no test warnings or skips.**

*   **Task 4.3: Integrate Strategy Selection in Query**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Integrate into `LightRAG.aquery`.
        Task: Modify `aquery` method.
        Requirements: Call `process_query`, then `select_retrieval_strategy`. Use *selected strategy* to route to internal logic. Optionally pass extracted keywords/entities.
        Output: Modified Python code for `aquery`.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate. Modify internal retrieval functions if needed.
    *   **Testing:** Implement and verify **passing** integration tests for `aquery`. Mock `process_query`/`select_retrieval_strategy`. Verify correct internal function calls. **Ensure no test warnings or skips.**

*   **Task 4.4: Refine Ranking/Filtering**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Improve retrieval results.
        Task: Propose mods to retrieval functions (`kg_query`, etc.) for filtering/re-ranking.
        Requirements: 1. Suggest logic for filtering by entity type from `process_query`. 2. Suggest re-ranking strategies (keywords, centrality, recency).
        Output: Description and pseudocode.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Implement chosen logic. Add config.
    *   **Testing:** Implement and verify **passing** unit/integration tests for filtering (assert removal) and re-ranking (assert order change). **Ensure no test warnings or skips.**

---
**Phase 5: Advanced Generation**
---

*   **Objective:** Improve response quality and reasoning using CoT and better citations.
*   **Robustness Note:** CoT refinement loop needs clear exit conditions. Parsing LLM reasoning can be fragile.

*   **Task 5.1: Chain-of-Thought (CoT) Implementation**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Implement CoT reasoning. Ref script: `llm_generator.py`. Prompt: `lightrag/prompt.py`.
        Task: 1. Modify generation prompt for step-by-step reasoning (`<reasoning>...</reasoning><answer>...</answer>`). 2. Update response parsing to extract both. 3. Implement basic refinement loop (**max 2 attempts**) if reasoning is poor. **Handle LLM format errors.**
        Output: 1. Modified prompt string(s). 2. Python snippets for parsing and refinement loop.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate. Add config flag.
    *   **Testing:** Implement and verify **passing** integration tests for CoT queries. Assert reasoning presence based on config. Manually evaluate reasoning/answer. Test refinement loop. **Test LLM responses *not* following CoT format.** **Ensure no test warnings or skips.**

*   **Task 5.2: Enhanced Citation Handling**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Trace citations to context elements. Ref script: `llm_generator.py`.
        Task: 1. Modify CoT prompt: Instruct LLM to include source IDs (e.g., `[Entity ID: node1]`) in reasoning. 2. Update citation logic (`operate.py`) to parse IDs, look up context, format citations. **Handle missing/invalid IDs in reasoning.**
        Output: 1. Modified CoT prompt string. 2. Python snippet for citation parsing/formatting.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Ensure context has usable IDs. Implement/test parsing.
    *   **Testing:** Implement and verify **passing** integration tests extending CoT tests. Pass context with known IDs. Verify correct citations reference IDs used in (mocked) reasoning. **Test reasoning with invalid/missing IDs.** **Ensure no test warnings or skips.**

*   **Task 5.3: Diagram/Formula Description Integration (Optional)**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Integrate stored diagram/formula placeholders/descriptions.
        Task: Describe integration steps.
        Requirements: 1. Explain how retrieval fetches data for placeholders. 2. Explain how generation prompt includes this. 3. (Optional) Outline on-the-fly LLM description point.
        Output: Descriptive text and pseudocode.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Implement chosen strategy.
    *   **Testing:** Implement and verify **passing** integration tests for queries involving diagrams/formulas. Verify descriptions appear correctly. **Ensure no test warnings or skips.**

---
**Phase 6: Final Integration, Testing, Benchmarking & Refinement**
---

*   **Objective:** Ensure system cohesion, performance, and quality.

*   **Task 6.1: Comprehensive Integration & E2E Tests**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Need comprehensive integration and E2E tests.
        Task: Generate `pytest` test case ideas covering the *entire* enhanced pipeline.
        Requirements: Cover full flow, doc types, query types, modes, CoT, entity resolution, error conditions.
        Output: List of descriptive integration/E2E test scenarios.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Implement tests. Use representative data. **Ensure all tests pass cleanly without warnings or skips.** Include some tests using *real* LLM calls.

*   **Task 6.2: Benchmarking Execution**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Need to benchmark key components.
        Task: Generate Python code snippets using `timeit` or `time.time` for benchmarking chunking, `EntityResolver.merge_entities`, end-to-end `aquery`.
        Output: Python benchmarking code snippets.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Integrate. Run benchmarks. Analyze.

*   **Task 6.3: Qualitative Evaluation**
    *   **AI Prompt:**
        ```
        --- AI PROMPT START ---
        Context: Need qualitative evaluation criteria.
        Task: Propose criteria list.
        Requirements: Criteria for KG Quality, Retrieval Relevance, Response Quality (incl. reasoning/citations).
        Output: List of criteria.
        --- AI PROMPT END ---
        ```
    *   **User Actions:** Define evaluation set. Perform evaluation.

*   **User Actions (Final):**
    *   **Run All Tests:** Achieve 100% pass rate with **no warnings or skips**.
    *   **Analyze Benchmarks:** Identify/address bottlenecks.
    *   **Perform Qualitative Evaluation:** Use feedback for final tuning.
    *   **Refine:** Final code adjustments, prompt tuning.
    *   **Update Documentation:** Thoroughly update all project documentation.

---
```