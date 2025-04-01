**Phase 1: Basic Pipeline and Core Functionality Migration**

**Deliverable 1.1: Document Processing Pipeline Functionality**

1.  **PDF Parsing:**
    *   **Objective:** Reliably ingest PDF documents and extract clean text content.
    *   **Micro-processes:**
        *   Implement a `PDFParser` class.
        *   `validate_pdf(file_path)`: Check file existence and basic PDF validity (e.g., using `PyPDF2.PdfReader` or `fitz.open` in a `try...except`).
        *   `extract_text(file_path)`:
            *   Open PDF using `fitz` (PyMuPDF) for robustness with different PDF types. Handle `FileNotFoundError` and `fitz.fitz.FitzError` (or equivalent), raising `PDFReadError`.
            *   Iterate through pages, extract raw text (`page.get_text()`). Handle page-level extraction errors gracefully (log warning, append empty string, continue).
            *   Store raw text per page for subsequent filtering.
            *   Concatenate text from *filtered* pages (see Content Filtering) into a single string.
    *   **Logging:** `DEBUG` Start/end processing `file_path`. `INFO` Number of pages extracted. `ERROR` File not found, unreadable/corrupted PDF. `WARNING` Failure to extract text from a specific page.
    *   **Testing Strategy:**
        *   **Unit:** Test `validate_pdf` with mock `os.path.exists` and mock `fitz.open` raising exceptions. Test `extract_text` with mock `fitz.Document` and `fitz.Page` objects, simulating page text extraction success/failure and PDF read errors. Verify correct text concatenation.
        *   **Integration:** Test with various real PDF files (text-based, image-based requiring OCR - though OCR is out of scope for base parsing, note the limitation, scanned, complex layouts, corrupted). Verify output text quality.

2.  **Content Filtering:**
    *   **Objective:** Identify and optionally exclude non-substantive content (TOC, Index, Headers/Footers, Boilerplate) from extracted page text *before* final concatenation.
    *   **Micro-processes:**
        *   Implement a `ContentFilter` class.
        *   `detect_toc_pages(pages_text: List[str])`: Implement heuristics (keywords like "Table of Contents", "Contents"; page number patterns like "... \d+"; indentation patterns) to identify likely TOC page indices.
        *   `detect_index_pages(pages_text: List[str])`: Implement heuristics (keywords like "Index"; alphabetical listing patterns; page number lists like "term ..., 10, 15-20") to identify likely Index page indices, typically near the end.
        *   `filter_line(line: str)`: Use regex patterns (configurable via `settings.py`) to identify and flag common header/footer lines (e.g., `Page \d+`, repeating document titles/section names) and boilerplate (copyright, disclaimers).
        *   `filter_document_pages(pages_text: List[str])`: Orchestrates the filtering. Takes the list of raw page texts. Calls `detect_toc_pages` and `detect_index_pages`. Iterates through *non-excluded* pages, applying `filter_line` to each line within those pages. Returns a list of cleaned page texts (or empty strings for excluded pages) and a list of excluded page indices.
        *   Integrate into `PDFParser.extract_text`: After initial text extraction per page, call `ContentFilter.filter_document_pages`. Use the returned cleaned page texts for final concatenation. Store excluded page indices in metadata. Make filtering optional via `PDFParser` config.
    *   **Logging:** `INFO` Number of pages identified as TOC/Index. `DEBUG` Specific lines filtered as header/footer/boilerplate. `INFO` Total pages excluded.
    *   **Testing Strategy:**
        *   **Unit:** Test detection methods (`detect_toc_pages`, `detect_index_pages`) with mock page text lists. Test `filter_line` with various example lines. Test `filter_document_pages` with mock page texts, verifying correct filtering and excluded indices list.
        *   **Integration:** Test `filter_document_pages` on text extracted from real PDFs with clear TOCs, indexes, headers, etc. Verify expected pages/lines are removed.

3.  **Text Chunking:**
    *   **Objective:** Segment the filtered document text into contextually relevant chunks for embedding and analysis.
    *   **Micro-processes:**
        *   Implement `TextChunker` class orchestrating different strategies.
        *   Implement `BoundaryDetector`: Use NLTK/spaCy for sentence tokenization (`nltk.sent_tokenize`). Use regex for paragraphs (`\n\s*\n`), headers (`^#+\s`, `^[A-Z][^\n]+\n=+$`), lists (`^\s*[\*\-\+]|\d+\.`). If `preserve_entities=True` and spaCy is available, add entity start/end character offsets from NER results to the boundary list.
        *   Implement `FixedSizeChunker`: Split text based on character/token count (`default_chunk_size`, `default_overlap` from config). If boundaries are provided, attempt to align chunk ends with the nearest boundary within a tolerance, otherwise split mid-word if necessary (or ensure word boundaries).
        *   Implement `SemanticChunker`: Split text using boundaries from `BoundaryDetector`. Merge adjacent small segments if their combined size is below a threshold (e.g., `target_chunk_size * 1.5`) to avoid overly fragmented chunks.
        *   Implement `HierarchicalChunker`: Parse text for structural markers (e.g., Markdown headers). Create nested `TextChunk` objects reflecting the hierarchy (using `parent_id`, `level`). Split content within sections if they exceed size limits, using semantic/fixed chunking for the content blocks.
        *   Implement `ContentTypeAnalyzer`: Simple regex/heuristic checks for code blocks (```), lists (`*`, `-`, `1.`), tables (`| --- |`), quotes (`>`).
        *   `TextChunker.chunk_text`:
            *   Determine primary strategy based on config (`use_hierarchical_chunking`, `use_semantic_boundaries`).
            *   If `adaptive_chunking`, use `ContentTypeAnalyzer` and potentially text complexity metrics (e.g., sentence length via NLTK) to adjust `target_chunk_size` per segment before passing to sub-chunkers.
            *   Apply the chosen primary strategy (Hierarchical > Semantic > Fixed-size fallback).
            *   Generate `TextChunk` objects, populating `text`, `chunk_id`, `source_doc`, `metadata` (including start/end offsets, strategy used), `position` (sequential index), `level`/`parent_id` (if hierarchical).
    *   **Logging:** `INFO` Primary chunking strategy selected. `INFO` Number of chunks generated. `DEBUG` Boundaries detected. `DEBUG` Adaptive chunk size calculated. `WARNING` Fallback strategy used if primary fails.
    *   **Testing Strategy:**
        *   **Unit:** Test each chunker strategy (`FixedSize`, `Semantic`, `Hierarchical`) with diverse texts. Test `BoundaryDetector` with mock NLP/NLTK results. Test `ContentTypeAnalyzer`. Test `TextChunker.chunk_text` orchestration logic with various configurations. Test edge cases (empty/short text).
        *   **Integration:** Test `TextChunker` with text extracted from real documents containing mixed content types and structures. Evaluate chunk quality (semantic coherence, size variance).

4.  **Metadata Extraction & Enrichment:**
    *   **Objective:** Consolidate file system and document metadata.
    *   **Micro-processes:**
        *   Implement `MetadataExtractor` class.
        *   `extract_file_metadata(file_path)`: Use `os.stat` for file system info (size, modified, created). Handle `FileNotFoundError`.
        *   `enrich_metadata(base_metadata, extracted_metadata)`: Combine file system metadata with metadata extracted during PDF parsing (title, author, page count, language, etc.). Normalize date formats (ISO 8601). Add processing timestamp.
        *   Integrate into `DocumentProcessor`: Call `extract_file_metadata` first, then pass its result along with PDF-internal metadata to `enrich_metadata`. Store the final enriched metadata.
    *   **Logging:** `INFO` Metadata fields extracted. `INFO` Detected language. `WARNING` Missing expected metadata fields. `ERROR` File access/stat errors.
    *   **Testing Strategy:**
        *   **Unit:** Test `extract_file_metadata` mocking `os.stat`. Test `enrich_metadata` logic with different combinations of input metadata dicts.
        *   **Integration:** Test the end-to-end metadata flow within `DocumentProcessor` using real files.

**Deliverable 1.2: Knowledge Graph Builder Functionality**

1.  **Entity and Relationship Extraction:**
    *   **Objective:** Extract entities and relationships from text chunks using an LLM, guided by the schema.
    *   **Micro-processes:**
        *   Implement `KnowledgeGraphBuilder` class (or integrate into pipeline).
        *   Load schema using `SchemaLoader`.
        *   For each `TextChunk`:
            *   Format extraction prompt for the LLM (e.g., Gemini). Include:
                *   Chunk text.
                *   Task instructions (extract entities/relationships).
                *   Schema definition (relevant entity types, relationship types, properties, domain/range constraints).
                *   Desired JSON output format specification.
                *   Few-shot examples (optional, configurable).
            *   Call LLM (`LLMGenerator.generate_response` or direct API call) with the prompt. Handle API errors/retries.
            *   Parse JSON response. Handle `JSONDecodeError`.
            *   Validate extracted items against schema (if strict mode enabled). Log/discard invalid items.
            *   Assign confidence scores (from LLM or estimated).
            *   Flag tentative elements based on schema mismatch and confidence thresholds (`NEW_TYPE_CONFIDENCE_THRESHOLD`, `SCHEMA_MATCH_CONFIDENCE_THRESHOLD`). Add prefixes (`TENTATIVE_ENTITY_PREFIX`, `TENTATIVE_RELATIONSHIP_PREFIX`) to types/labels if tentative.
            *   Store extracted entities/relationships (possibly temporarily or in chunk metadata) for graph writing.
    *   **Logging:** `INFO` Starting/ending extraction for chunk/batch. `INFO` Number of entities/relationships extracted per chunk. `DEBUG` LLM Prompt (potentially truncated, exclude API keys). `DEBUG` Raw LLM response (potentially truncated). `WARNING` JSON parsing errors. `WARNING` Schema validation failures/pruned items. `WARNING` Tentative entities/relationships created. `ERROR` LLM API call failures.
    *   **Testing Strategy:**
        *   **Unit:** Mock LLM calls. Test prompt formatting with different schemas/chunks. Test JSON parsing logic for valid/invalid responses. Test schema validation logic. Test tentative flagging logic.
        *   **Integration (Requires LLM):** Test end-to-end extraction on sample chunks with a schema. Evaluate the quality, accuracy, and schema adherence of extracted data. Test handling of complex or ambiguous text.

2.  **Embedding Generation:**
    *   **Objective:** Create vector representations for text chunks.
    *   **Micro-processes:**
        *   Implement `EmbeddingGenerator` class.
        *   Initialize with configured provider (Google/OpenAI), model name, batch size, retries, delay.
        *   `generate_embeddings_batch(texts)`:
            *   Handle empty input list.
            *   Iterate through texts in batches (`EMBEDDING_BATCH_SIZE`).
            *   Handle empty strings within a batch (e.g., return zero vector or skip and reconstruct).
            *   Call appropriate provider API (Google AI `embed_content` or OpenAI `embeddings.create`).
            *   Implement retry logic for transient errors (`ServiceUnavailable`, `RateLimitError`, `APIConnectionError`) using `EMBEDDING_MAX_RETRIES` and `EMBEDDING_RETRY_DELAY`.
            *   Handle non-retryable API errors (`InvalidRequestError`, other `APIError`) by logging and raising `EmbeddingError`.
            *   Handle unexpected exceptions.
            *   Return list of embedding vectors.
        *   `embed_text_chunks(chunks)`: Helper method to extract text from `TextChunk` list, call `generate_embeddings_batch`, and add the resulting vectors back to the `TextChunk.embedding` attribute. Handle potential mismatches in list lengths due to errors.
    *   **Logging:** `INFO` Starting/ending embedding generation for batch/document. `INFO` Provider/model used. `DEBUG` Number of texts in batch. `WARNING` Retrying API call due to transient error (include attempt count). `ERROR` Embedding generation failed for chunk/batch after retries. `ERROR` Non-retryable API error. `ERROR` Unexpected error.
    *   **Testing Strategy:**
        *   **Unit:** Mock embedding API calls. Test batching logic. Test retry mechanism by mocking transient and non-retryable API errors. Test handling of empty texts/batches. Test correct assignment of embeddings back to chunks.
        *   **Integration (Requires Embedding API):** Test with a small batch of sample texts. Verify embedding dimensionality and basic structure. Test error handling if possible (e.g., sending invalid input).

3.  **Neo4j Knowledge Graph Writing:**
    *   **Objective:** Persist chunks, extracted entities, and relationships into a Neo4j graph database.
    *   **Micro-processes:**
        *   Implement `Neo4jKnowledgeGraph` class (or adapt existing `LightRAG` storage).
        *   `__init__`: Establish driver connection using configured credentials. Verify connection. Store index name/dimensions.
        *   `_execute_query`: Private helper for running Cypher queries with parameters, error handling (`Neo4jError`), and transaction management (use `session.execute_write` or `session.execute_read`).
        *   `setup_vector_index()`: Check if index exists. If not, create a vector index on the `embedding` property of `:Chunk` nodes using configured `vector_index_name` and `vector_dimensions`. Create unique constraint on `Chunk.chunk_id`. Wait for index to come online. Handle errors.
        *   `apply_schema_constraints(schema_cypher)`: Execute provided Cypher statements (e.g., from `SchemaLoader`) to create DB constraints/indexes.
        *   `create_chunks_in_graph(chunks: List[TextChunk])`:
            *   Use `UNWIND $chunks AS chunk MERGE (c:Chunk {chunk_id: chunk.chunk_id}) SET c += chunk.properties, c.embedding = chunk.embedding` (adapt properties). Add `created_at` / `updated_at` timestamps.
            *   Handle chunk relationships (e.g., hierarchical `PART_OF`, sequential `NEXT_CHUNK`, cross-references `REFERENCES`) using `MATCH (source), (target) MERGE (source)-[r:TYPE]->(target)`.
        *   `create_schema_aware_graph(classified_chunks: List[TextChunk])`:
            *   Iterate through chunks containing extracted entities/relationships in their metadata.
            *   **Nodes:** `UNWIND $entities AS entity MERGE (e {entity_id: entity.id}) SET e += entity.properties SET e:{entity.label}` (Use `apoc.merge.node` for dynamic labels including `:TentativeEntity`).
            *   **Relationships:** `UNWIND $relationships AS rel MATCH (source {entity_id: rel.source_id}), (target {entity_id: rel.target_id}) MERGE (source)-[r:{rel.type}]->(target) SET r += rel.properties` (Use `apoc.create.relationship` for dynamic types, add `:TENTATIVE_RELATIONSHIP` label if needed).
            *   Create `:CONTAINS_ENTITY` relationships from `:Chunk` to the primary `:Entity` nodes extracted from it.
    *   **Logging:** `INFO` Start/end graph writing. `INFO` Nodes/Relationships created/merged counts. `INFO` Index/Constraint status. `DEBUG` Parameterized Cypher queries (sensitive data masked). `WARNING` Constraint violations handled. `ERROR` DB Connection errors. `ERROR` Cypher execution errors.
    *   **Testing Strategy:**
        *   **Unit:** Mock Neo4j driver/session/transaction. Test Cypher generation for all operations (nodes, relationships, index, constraints). Verify correct parameter binding. Test error handling logic.
        *   **Integration (Requires Neo4j):** Test creating chunks, entities, relationships. Verify data persistence, properties, labels (including tentative), and relationships. Test `MERGE` idempotency. Test vector index creation and querying `SHOW INDEXES`. Test error scenarios (e.g., constraint violation).

**Deliverable 1.3: Basic Retriever Integration Functionality**

1.  **Vector-Based Retrieval:**
    *   **Objective:** Retrieve relevant text chunks using vector similarity search against the Neo4j index.
    *   **Micro-processes:**
        *   Implement `GraphRAGRetriever` class.
        *   `__init__`: Store `Neo4jKnowledgeGraph` instance, embedding generator instance, and retrieval parameters (limit, threshold).
        *   `retrieve(query, strategy="vector")`:
            *   Generate embedding for `query` using `EmbeddingGenerator`.
            *   Call `Neo4jKnowledgeGraph.similarity_search` with query embedding, limit, threshold, and optionally entity type filter.
            *   Format results into a consistent list of dictionaries (e.g., containing `chunk_id`, `text`, `score`, `metadata`).
    *   **Logging:** `INFO` Starting vector retrieval. `INFO` Number of results retrieved. `DEBUG` Query embedding (potentially summarized). `DEBUG` Parameters passed to `similarity_search`. `ERROR` Failure during embedding generation or similarity search.
    *   **Testing Strategy:**
        *   **Unit:** Mock `EmbeddingGenerator` and `Neo4jKnowledgeGraph`. Test `retrieve(strategy="vector")` logic, parameter passing, and result formatting.
        *   **Integration (Requires Neo4j with index, Embedding API):** Test `retrieve` with sample queries. Verify relevance and ranking based on similarity scores. Test limit and threshold parameters.

**Deliverable 1.4: Basic Query Functionality**

1.  **LLM-Based Question Answering:**
    *   **Objective:** Generate a coherent answer based on retrieved context using an LLM.
    *   **Micro-processes:**
        *   Implement `LLMGenerator` class.
        *   `__init__`: Initialize LLM client (Gemini/OpenAI) based on config, store API key, model name, default parameters (temperature, max_tokens).
        *   `_format_context(context_items, max_length)`: Combine text from context items (list of dicts). Prioritize items (e.g., by score). Truncate context to fit `max_length` (estimated token count). Return formatted string.
        *   `_call_gemini`/`_call_openai`: Private methods to handle API calls for the specific provider, including error handling and retries.
        *   `generate_response(query, context_items)`:
            *   Call `_format_context`.
            *   Select and format a basic prompt template (e.g., "Answer the query based on the context...") inserting the query and formatted context.
            *   Call the appropriate LLM (`_call_gemini` or `_call_openai`).
            *   Return the LLM's text response in a structured dictionary (e.g., `{'response': text, 'success': True}`).
    *   **Logging:** `INFO` Starting response generation. `INFO` LLM provider/model used. `DEBUG` Formatted context (potentially truncated). `DEBUG` Final prompt sent to LLM. `DEBUG` Raw LLM response. `ERROR` LLM API call errors after retries.
    *   **Testing Strategy:**
        *   **Unit:** Mock LLM client/API calls. Test `_format_context` logic (prioritization, truncation). Test prompt formatting. Test `generate_response` orchestration. Test error handling wrappers in `_call_...` methods.
        *   **Integration (Requires LLM API):** Test `generate_response` with mock context data. Evaluate response quality and coherence.

---

**Phase 2: Feature Completeness & Enhanced Capabilities**

**Deliverable 2.1: Advanced Hybrid Retriever Functionality**

1.  **Hybrid Retrieval Core:**
    *   **Objective:** Combine vector search seeding with graph traversal for context expansion.
    *   **Micro-processes:**
        *   Enhance `GraphRAGRetriever.retrieve(strategy="hybrid")`:
            *   Perform initial `_vector_retrieval` to get top N seed chunks (N configurable).
            *   For each seed chunk ID, call `_graph_retrieval` (or `Neo4jKnowledgeGraph.get_related_nodes`) to find related nodes (entities, concepts, other chunks) up to `max_related_depth`. Filter by relationship types if specified by query analysis.
            *   Implement `_merge_and_deduplicate_results`: Combine seed results and graph results. Remove duplicate nodes/chunks based on ID. Rank the combined list using a weighted score (e.g., `w1 * vector_similarity + w2 * graph_distance_score + w3 * node_importance`). Weights can be dynamically adjusted based on query intent from `query_analysis`.
    *   **Logging:** `INFO` Executing hybrid retrieval. `INFO` Number of seed nodes used. `INFO` Number of graph nodes retrieved. `INFO` Final count after merge/deduplication. `DEBUG` Seed node IDs. `DEBUG` Weights used for ranking.
    *   **Testing Strategy:**
        *   **Unit:** Mock `_vector_retrieval` and `_graph_retrieval`/`get_related_nodes`. Test `_merge_and_deduplicate_results` logic for combining, deduplicating, and ranking results with different weights and inputs.
        *   **Integration (Requires Neo4j, Embeddings):** Test `retrieve(strategy="hybrid")` on queries where context expansion via graph traversal is expected to yield better results than vector search alone. Compare outputs.

**Deliverable 2.2: Schema Integration into Pipeline Functionality**

1.  **Schema-Guided Extraction:**
    *   **Objective:** Leverage the loaded schema to improve the accuracy and structure of LLM-based entity/relationship extraction.
    *   **Micro-processes:**
        *   `KnowledgeGraphBuilder`/Extractor component: Modify prompt generation to include a formatted representation of relevant schema parts (target entity types, properties, relationship types with domain/range) based on `SchemaLoader`.
        *   **(Optional) Strict Validation:** Add a post-LLM step to validate extracted entities/relationships against the schema. If an entity type doesn't exist, either discard or flag as tentative. If a relationship violates domain/range, discard or flag. If properties don't match, discard/ignore extra properties. Make this behavior configurable.
    *   **Logging:** `DEBUG` Schema snippet included in extraction prompt. `WARNING` Extracted entity/relationship discarded/flagged due to schema violation (if strict mode).
    *   **Testing Strategy:**
        *   **Unit:** Test prompt formatting ensures schema inclusion. Mock LLM responses containing schema-compliant and non-compliant data. Test validation logic flags/discards correctly based on config.
        *   **Integration (Requires LLM):** Compare extracted structures with and without schema guidance. Evaluate if schema guidance improves type accuracy and property adherence.

2.  **Schema-Enhanced Retrieval:**
    *   **Objective:** Use schema information to refine graph traversal and result ranking during retrieval.
    *   **Micro-processes:**
        *   `GraphRAGRetriever`:
            *   Modify `process_query` to identify potential target entity types based on query keywords and schema entity names/descriptions.
            *   Modify `_graph_retrieval`/`_get_related_nodes`: Prioritize traversing relationship types defined in the schema. Use identified target entity types from `process_query` to guide traversal or filter results (e.g., `MATCH (start)-[...]-(end:{target_type})`).
            *   Modify `_merge_and_deduplicate_results`: Potentially boost the rank of results whose entity types match those identified in the query analysis.
    *   **Logging:** `INFO` Using schema types (`{types}`) to guide graph retrieval. `DEBUG` Prioritized relationship types used for traversal. `DEBUG` Rank boost applied based on schema type match.
    *   **Testing Strategy:**
        *   **Unit:** Mock `Neo4jKnowledgeGraph`. Test that `_graph_retrieval` generates Cypher queries that incorporate entity type filters or relationship type priorities based on mock `query_analysis` and schema. Test ranking adjustments in `_merge_and_deduplicate_results`.
        *   **Integration (Requires Neo4j):** Design queries where schema knowledge (e.g., specific relationship paths between expected entity types) is crucial for relevance. Compare results with and without schema-enhanced retrieval logic.

3.  **Prompt Template Schema Context:**
    *   **Objective:** Provide the LLM with relevant schema context during response generation.
    *   **Micro-processes:**
        *   `LLMGenerator._format_schema_context(context_items, schema_loader)`: Identify the main entity types present in the `context_items`. Retrieve their definitions and key properties from the `schema_loader`. Retrieve relevant relationship types connecting these entities, including their domain/range. Format this information concisely (e.g., "Entities relevant here include: Concept (properties: name, definition), Formula (properties: expression, variables). Relationships: Concept IS_CALCULATED_USING Formula.").
        *   `LLMGenerator.generate_response`: Call `_format_schema_context`. Estimate token count. Adjust main context length if needed. Integrate the schema context string into the final prompt, clearly separated from the data context (e.g., using headers). Update template formatting logic to handle potential `schema_context` variable gracefully.
    *   **Logging:** `INFO` Adding schema context to generation prompt. `DEBUG` Formatted schema context string. `INFO` Adjusted main context length due to schema context size.
    *   **Testing Strategy:**
        *   **Unit:** Test `_format_schema_context` with mock context items and schema loader, verify output format and relevance. Test prompt formatting in `generate_response` ensures correct integration and length management.
        *   **Integration (Requires LLM):** Compare responses generated with/without schema context for queries where schema understanding is beneficial (e.g., asking about properties of a concept).

**Deliverable 2.3: Enhanced Response Generation with CoT and Citations**

1.  **Chain-of-Thought (CoT) Prompting:**
    *   **Objective:** Elicit step-by-step reasoning from the LLM for better explainability and potential refinement.
    *   **Micro-processes:**
        *   `LLMGenerator`: Define structured CoT prompt templates (e.g., `qa_cot`, `financial_concept_cot`) instructing the LLM to break down the problem, analyze context, synthesize steps, and conclude.
        *   `generate_response`: Select CoT template if `use_cot=True` is passed.
    *   **Logging:** `INFO` Using CoT prompt template.
    *   **Testing Strategy:**
        *   **Unit:** Verify correct template selection.
        *   **Integration (Requires LLM):** Evaluate CoT responses for explicit reasoning steps.

2.  **Iterative Response Refinement:**
    *   **Objective:** Improve LLM responses based on internal analysis of the CoT output.
    *   **Micro-processes:**
        *   `LLMGenerator._analyze_cot_response`: Implement logic to parse CoT response (e.g., split by step markers). Assess logical consistency, check if steps reference context (simple keyword checks), verify presence of a final conclusion. Return analysis dict (e.g., `{'consistent': bool, 'supported': bool, 'conclusive': bool, 'score': float}`).
        *   `LLMGenerator._is_response_sufficient`: Check if analysis score meets a configurable threshold and if essential elements (e.g., conclusion) are present based on `task_type`.
        *   `LLMGenerator._generate_refinement_prompt`: Create a prompt based on analysis failures (e.g., "Refine reasoning: Step 3 lacks clear context support. Ensure final answer summarizes steps."). Include original prompt and previous insufficient response.
        *   `LLMGenerator.generate_response`: If `use_cot=True`, enter loop (up to `max_refinement_attempts` from config). Call LLM -> Analyze -> If insufficient & attempts left -> Generate refinement prompt -> Call LLM again. Store history of attempts/analysis. Return best/final response.
    *   **Logging:** `INFO` Refinement iteration start/end. `DEBUG` CoT analysis results. `DEBUG` Refinement prompt generated. `WARNING` Max refinement attempts reached.
    *   **Testing Strategy:**
        *   **Unit:** Test `_analyze_cot_response` with various mock CoT outputs. Test `_is_response_sufficient` logic. Test `_generate_refinement_prompt`. Mock LLM calls and test the refinement loop in `generate_response`.
        *   **Integration (Requires LLM):** Test with queries likely to yield improvable CoT responses. Verify if refinement improves the response based on analysis criteria. Check `refinement_history` structure.

3.  **Citation Handling:**
    *   **Objective:** Link statements in the generated response back to the source context items.
    *   **Micro-processes:**
        *   `LLMGenerator._find_citation_evidence`: Implement a more robust check, potentially using simple sentence embedding similarity (cosine distance between generated sentence and source chunk embeddings) or better keyword overlap logic, returning confidence.
        *   `LLMGenerator._format_citations`: Iterate sentences in the LLM response. For each, find the top source(s) from `context_items` using `_find_citation_evidence`. Filter based on confidence threshold. Select best N citations (configurable `max_citations`), potentially prioritizing diversity of sources. Append citation markers (e.g., `[1]`) to sentences. Create a numbered list of cited sources (using `source_doc`, `chunk_id`, or `title` from context item metadata).
        *   `LLMGenerator.generate_response`: If `include_citations=True`, call `_format_citations` on the final LLM text. Return the text with inline citations and the separate list of citation details (dictionaries with source info, cited text snippet, confidence).
    *   **Logging:** `INFO` Generating citations. `INFO` Number of citations generated. `DEBUG` Evidence details for potential citations (sentence, source chunk, score).
    *   **Testing Strategy:**
        *   **Unit:** Test `_find_citation_evidence` with various sentence/source pairs. Test `_format_citations` logic, ensuring correct marker placement and source list generation based on mock evidence. Test `generate_response` returns the expected citation structure.
        *   **Integration (Requires LLM):** Generate responses with citations. Manually review if citations are placed reasonably and link to relevant source context provided in the prompt.

**Deliverable 2.4: Advanced Diagram and Table Analysis Functionality**

1.  **Advanced Diagram Analysis:**
    *   **Objective:** Improve diagram classification and description generation quality.
    *   **Micro-processes:**
        *   `DiagramAnalyzer._is_diagram`: Enhance heuristic scoring (e.g., weight aspect ratio, color ratio, edge density from `PIL.ImageFilter.FIND_EDGES`, text-to-image ratio, simple shape count via OpenCV if available). Compare score against configurable `DIAGRAM_DETECTION_THRESHOLD`.
        *   `DiagramAnalyzer.add_description_template(diagram_type, prompt_template)`: Add method to manage custom prompts.
        *   `DiagramAnalyzer._create_description_prompt(diagram_type)`: Retrieve prompt from internal dictionary using `diagram_type`, fallback to 'general'.
        *   `DiagramAnalyzer.extract_diagrams_from_pdf`: *Optionally* add a step to classify diagram type (heuristics or simple LLM call on image/context) and store in metadata. Pass `diagram_type` to `generate_diagram_description`.
        *   `DiagramAnalyzer.generate_diagram_description`: Accept `diagram_type`. Use it to fetch the specific prompt template for the vision LLM call.
    *   **Logging:** `DEBUG` Diagram detection scores (aspect, color, edge, shape). `INFO` Classified diagram type (if implemented). `INFO` Using specific/general prompt template for description.
    *   **Testing Strategy:**
        *   **Unit:** Test `_is_diagram` scoring with mock image stats. Test prompt selection logic in `_create_description_prompt`. Mock diagram classification. Mock vision LLM call and verify correct prompt usage in `generate_diagram_description`.
        *   **Integration (Requires Vision LLM):** Test with PDFs containing diverse diagrams. Evaluate detection accuracy. Evaluate description quality based on diagram type and specific prompts.

2.  **Enhanced Table Extraction:**
    *   **Objective:** Improve flexibility and robustness of table extraction and conversion.
    *   **Micro-processes:**
        *   `TableExtractor._extract_with_...`: Modify to accept `pages: Optional[List[int]]`. Handle page number conversion (0-based list to 1-based string for Camelot, 0-based list for pdfplumber).
        *   `TableExtractor.extract_tables_from_pdf`: Update signature to accept `pages`.
        *   `TableExtractor.convert_table_to_markdown`: Improve header detection heuristic (check first row content vs second, e.g., type consistency, capitalization, string presence). If no header detected, log `WARNING` and use `df.to_markdown(index=False)` *without* `headers="keys"`. Catch exceptions during the `to_markdown` call itself and log `ERROR`, returning empty string.
    *   **Logging:** `INFO` Table extraction engine used. `INFO` Number of tables extracted. `WARNING` Header detection failed, using fallback markdown conversion. `ERROR` Exception during markdown conversion.
    *   **Testing Strategy:**
        *   **Unit:** Test page number handling. Mock extraction libraries. Test `extract_tables_from_pdf`. Test `convert_table_to_markdown` header detection and fallback logic with various table data structures. Verify correct markdown output.
        *   **Integration:** Test extraction on PDFs with various table formats (bordered, borderless, complex headers). Verify accuracy and markdown quality.

**Deliverable 2.5: Advanced Entity Resolution**

1.  **Context-Based Matching:**
    *   **Objective:** Leverage graph structure to find entities with similar contexts.
    *   **Micro-processes:**
        *   `EntityResolver._match_entities_by_context(entity_type)`: Implement Cypher query: `MATCH (e1:{entity_type})-[r1]->(neighbor)<-[r2]-(e2:{entity_type}) WHERE id(e1) < id(e2) AND type(r1) = type(r2) WITH e1, e2, count(DISTINCT neighbor) as shared_neighbors WHERE shared_neighbors >= $min_shared RETURN id(e1), id(e2), shared_neighbors`. Parameterize `$min_shared` from config (`ENTITY_CONTEXT_MATCH_MIN_SHARED`). Calculate confidence based on `shared_neighbors`. Return list of `(entity1_data, entity2_data, confidence)`.
    *   **Logging:** `INFO` Starting context matching for `entity_type`. `INFO` Found N potential pairs via context. `DEBUG` Cypher query executed.
    *   **Testing Strategy:**
        *   **Unit:** Mock Neo4j query results. Test confidence calculation.
        *   **Integration (Requires Neo4j):** Create test entities with varying shared neighbors. Verify `_match_entities_by_context` identifies correct pairs based on `min_shared` and confidence.

2.  **Advanced Conflict Resolution:**
    *   **Objective:** Provide flexible strategies for merging properties of duplicate entities.
    *   **Micro-processes:**
        *   `EntityResolver._resolve_property_conflicts`: Enhance logic to handle strategies like `KEEP_LATEST` (requires timestamp property), `CONCATENATE` (for strings, avoid duplicates), `MERGE_ARRAYS` (combine lists, unique items), `KEEP_MOST_COMPLETE`, `KEEP_LONGEST`. Use `self.property_conflict_strategy` dict, fallback to `self.default_conflict_strategy`. Handle type mismatches gracefully.
    *   **Logging:** `DEBUG` Resolving property `prop_name` using strategy `strategy_name`. `DEBUG` Conflicting values: `[val1, val2]`, Resolved value: `resolved_val`.
    *   **Testing Strategy:**
        *   **Unit:** Test `_resolve_property_conflicts` with diverse mock entities and conflicting properties (strings, lists, numbers, dates with timestamps). Verify each strategy produces the correct merged value. Test fallback to default strategy.

3.  **Audit Trail:**
    *   **Objective:** Log merge operations for traceability.
    *   **Micro-processes:**
        *   `EntityResolver._create_merge_audit(...)`: Create `:EntityMergeAudit` node in Neo4j. Store: timestamp, primary entity ID, list of merged entity IDs, entity type, final confidence, contributing match methods (e.g., `matched_by_name: true`). Link audit node to primary entity (`[:MERGED_INTO]`).
        *   Call `_create_merge_audit` in `merge_duplicate_entities` if `audit_trail=True` and merge is confirmed and successful.
    *   **Logging:** `INFO` Creating audit record for merge `primary_id <- [duplicate_ids]`. `ERROR` Failed to create audit record.
    *   **Testing Strategy:**
        *   **Unit:** Mock Neo4j query execution. Test that `_create_merge_audit` generates the correct Cypher and parameters.
        *   **Integration (Requires Neo4j):** Perform merges with `audit_trail=True`. Query Neo4j to verify `:EntityMergeAudit` nodes are created with correct properties and relationships.

**Deliverable 2.6: KG Index Synchronization**

1.  **Synchronization Component:**
    *   **Objective:** Keep Neo4j vector index up-to-date with graph changes affecting embeddings.
    *   **Micro-processes:**
        *   Implement `KGIndexSynchronizer` class.
        *   `__init__`: Initialize Neo4j driver, EmbeddingGenerator, config (batch size, index name).
        *   `_get_nodes_needing_updates[_async]`: Query Neo4j for `:Chunk` nodes WHERE `n.embedding IS NULL` OR `n._needs_embedding_update = true`. Return node IDs and text content.
        *   `process_updates[_async]`:
            *   Call `_get_nodes_needing_updates`.
            *   Process node IDs in batches (`batch_size`).
            *   For each batch: call `EmbeddingGenerator.generate_embeddings_batch`.
            *   Write a Cypher query (`UNWIND $updates AS u MATCH (n) WHERE id(n) = u.id SET n.embedding = u.embedding, n._needs_embedding_update = null`). Handle errors within the batch (log failures, continue if possible).
            *   Return stats (`updated`, `failed`, `total`).
        *   `rebuild_index`: Mark *all* relevant nodes (`MATCH (n:Chunk) SET n._needs_embedding_update = true`). Optionally drop/recreate index via `setup_vector_index`. Run `process_updates`. Log stages.
        *   `mark_nodes_for_update(nodes: List[Dict])`: Set `_needs_embedding_update = true` on specified nodes (matched by `chunk_id`).
        *   `check_index_consistency`: Query `COUNT {(n:Chunk) WHERE n.embedding IS NOT NULL}` and compare with approximate index size (e.g., `CALL db.index.vector.queryNodes(...)` status or count). Return stats/percentage.
        *   **(Optional) Automation:** `start_scheduled_sync`: Use `threading.Timer` or `asyncio.create_task` with `asyncio.sleep` to periodically call `process_updates[_async]`. `stop_scheduled_sync`.
    *   **Logging:** `INFO` Sync start/end, batches processed, nodes updated/failed. `INFO` Rebuild start/end. `DEBUG` Nodes identified for update. `WARNING` Index inconsistency detected. `ERROR` Batch processing failure. `ERROR` Index management errors.
    *   **Testing Strategy:**
        *   **Unit:** Mock Neo4j driver and EmbeddingGenerator. Test `_get_nodes_needing_updates` query. Test `process_updates` batching, embedding calls, update query generation, error handling. Test `rebuild_index` logic flow.
        *   **Integration (Requires Neo4j, Embeddings):** Create nodes with/without embeddings. Run `process_updates`, verify embeddings and flags. Test `mark_nodes_for_update`. Test `rebuild_index`. Test `check_index_consistency`.