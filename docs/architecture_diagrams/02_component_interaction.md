# Component Interaction Diagram

This class diagram shows the core classes of LightRAG and how they interact with each other, illustrating the object-oriented design of the system.

## Diagram Description

The diagram represents the following key components and their relationships:

1. **LightRAG**: The main class that users interact with, containing methods for document insertion, querying, and knowledge graph management. It coordinates all other components and implements the core functionality of the system.

2. **Storage Abstractions**: A hierarchy of abstract classes that define interfaces for different storage types:
   - **BaseStorage**: The root abstract class for all storage types
   - **BaseKVStorage**: For key-value storage implementations
   - **BaseVectorStorage**: For vector database implementations
   - **BaseGraphStorage**: For knowledge graph storage implementations
   - **DocStatusStorage**: For document processing status tracking

3. **Storage Implementations**: Various concrete implementations of the storage abstractions, including:
   - JSON file storage options
   - Database integrations (Neo4j, PostgreSQL, MongoDB, etc.)
   - Vector databases (FAISS, Milvus, Chroma, etc.)
   - Graph databases and in-memory options

4. **QueryParam**: A data class that encapsulates all configuration options for queries, including retrieval mode, token limits, and various flags for controlling the query behavior.

5. **Utility Classes**:
   - **Operators**: Functions for chunking, extraction, and query processing
   - **DiagramAnalyzer**: For processing diagrams in documents
   - **FormulaInterpreter**: For handling mathematical formulas
   - **QueryProcessor**: For analyzing and optimizing queries

The diagram illustrates the composition relationships between LightRAG and its storage components, as well as the inheritance hierarchies within the storage abstractions. It also shows how the various utility classes are used by the core LightRAG class.

```mermaid
classDiagram
    class LightRAG {
        +String working_dir
        +String kv_storage
        +String vector_storage
        +String graph_storage
        +String doc_status_storage
        +int chunk_token_size
        +int chunk_overlap_token_size
        +Tokenizer tokenizer
        +EmbeddingFunc embedding_func
        +Callable llm_model_func
        +initialize_storages()
        +finalize_storages()
        +insert(content, ids, file_paths)
        +query(query_text, param, system_prompt)
        +create_entity(entity_name, properties)
        +create_relation(src_id, tgt_id, properties)
        +clear_cache(modes)
    }

    class BaseStorage {
        <<abstract>>
        +String namespace
        +Dict global_config
        +initialize()
        +finalize()
        +index_done_callback()
        +drop()
    }

    class BaseKVStorage {
        <<abstract>>
        +EmbeddingFunc embedding_func
        +get_by_id(id)
        +get_by_ids(ids)
        +filter_keys(keys)
        +upsert(data)
        +delete(ids)
        +drop_cache_by_modes(modes)
    }

    class BaseVectorStorage {
        <<abstract>>
        +EmbeddingFunc embedding_func
        +float cosine_better_than_threshold
        +Set meta_fields
        +query(query, top_k, ids)
        +upsert(data)
        +delete_entity(entity_name)
        +delete_entity_relation(entity_name)
        +get_by_id(id)
        +get_by_ids(ids)
        +delete(ids)
    }

    class BaseGraphStorage {
        <<abstract>>
        +EmbeddingFunc embedding_func
        +has_node(node_id)
        +has_edge(source_node_id, target_node_id)
        +node_degree(node_id)
        +edge_degree(src_id, tgt_id)
        +get_node(node_id)
        +get_edge(source_node_id, target_node_id)
        +get_node_edges(source_node_id)
        +upsert_node(node_id, node_data)
        +upsert_edge(source_node_id, target_node_id, edge_data)
        +delete_node(node_id)
        +remove_nodes(nodes)
        +remove_edges(edges)
        +get_all_labels()
        +get_knowledge_graph(node_label, max_depth, max_nodes)
    }

    class DocStatusStorage {
        <<abstract>>
        +get_status_counts()
        +get_docs_by_status(status)
    }

    class QueryParam {
        +String mode
        +boolean only_need_context
        +boolean only_need_prompt
        +String response_type
        +boolean stream
        +int top_k
        +int max_token_for_text_unit
        +int max_token_for_global_context
        +int max_token_for_local_context
        +List hl_keywords
        +List ll_keywords
        +List conversation_history
        +int history_turns
        +List ids
        +Callable model_func
        +boolean use_intelligent_retrieval
        +Dict query_analysis
        +boolean filter_by_entity_type
        +boolean rerank_results
    }

    class StorageImplementations {
        <<interface>>
        +JsonKVStorage
        +NetworkXStorage
        +NanoVectorDBStorage
        +JsonDocStatusStorage
        +Neo4JStorage
        +PGKVStorage
        +PGVectorStorage
        +PGGraphStorage
        +MilvusVectorDBStorage
        +ChromaVectorDBStorage
        +FaissVectorDBStorage
        +MongoVectorDBStorage
        +QdrantVectorDBStorage
        +RedisKVStorage
        +AGEStorage
    }

    class Operators {
        <<utility>>
        +chunking_by_token_size()
        +extract_entities()
        +merge_nodes_and_edges()
        +kg_query()
        +mix_kg_vector_query()
        +naive_query()
        +query_with_keywords()
    }

    class DiagramAnalyzer {
        +analyze_diagram()
        +extract_entities_from_diagram()
        +extract_relationships_from_diagram()
    }

    class FormulaInterpreter {
        +interpret_formula()
        +extract_formula_meaning()
        +identify_formula_variables()
        +relate_formula_to_context()
    }

    class QueryProcessor {
        +process_query()
        +select_retrieval_strategy()
        +extract_keywords()
        +analyze_query_intent()
    }

    LightRAG "1" *-- "4" BaseStorage : uses
    BaseStorage <|-- BaseKVStorage
    BaseStorage <|-- BaseVectorStorage
    BaseStorage <|-- BaseGraphStorage
    BaseKVStorage <|-- DocStatusStorage
    BaseKVStorage <|.. StorageImplementations : implements
    BaseVectorStorage <|.. StorageImplementations : implements
    BaseGraphStorage <|.. StorageImplementations : implements
    DocStatusStorage <|.. StorageImplementations : implements
    LightRAG ..> QueryParam : uses
    LightRAG ..> Operators : uses
    LightRAG ..> DiagramAnalyzer : uses
    LightRAG ..> FormulaInterpreter : uses
    LightRAG ..> QueryProcessor : uses
```