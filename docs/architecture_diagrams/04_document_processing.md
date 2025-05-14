# Document Processing Pipeline Diagram

This sequence diagram details the step-by-step process of how documents are ingested, processed, and stored in the LightRAG system.

## Diagram Description

The document processing pipeline involves these major steps:

1. **Initial Processing**:
   - The user submits a document to LightRAG via the `insert()` method
   - LightRAG creates a document status entry with PENDING status
   - The document processor begins processing the document
   - Document status is updated to PROCESSING

2. **Document Analysis and Chunking**:
   - For PDF or image documents, text and visual elements are extracted
   - The document is split into chunks based on token size and overlap settings
   - Special content like diagrams and formulas receive additional processing:
     - Diagrams are analyzed using vision capabilities to extract entities and relationships
     - Formulas are interpreted to understand their meaning and context
   - Chunks are stored in the Key-Value storage

3. **Entity and Relationship Extraction**:
   - For each chunk, an LLM is used to extract entities with structured prompting
   - Entity results are parsed and normalized
   - Similarly, relationships between entities are extracted using LLM
   - Relationship results are parsed and normalized

4. **Knowledge Graph Construction**:
   - Detected entities are collected and processed
   - Duplicate entities are merged
   - For each entity:
     - If it already exists in the graph, it's merged with the existing entity
     - If descriptions become too long, an LLM summarizes them
     - The entity node is upserted to the graph and its embedding to the vector database
   - For each relationship:
     - If it already exists, it's merged with the existing relationship
     - If descriptions become too long, an LLM summarizes them
     - The relationship edge is upserted to the graph and its embedding to the vector database

5. **Finalization**:
   - Document status is updated to PROCESSED
   - Storage updates are committed via index_done_callback
   - Success is reported back to the user

The diagram shows the interaction between the User, LightRAG, and various components like the DocumentProcessor, Chunker, EntityExtractor, storage systems, and the LLM. It illustrates both the sequential nature of document processing and the parallel processing of multiple chunks for efficiency.

```mermaid
sequenceDiagram
    participant User
    participant LightRAG
    participant DocumentProcessor
    participant Chunker
    participant DiagramAnalyzer
    participant FormulaInterpreter
    participant EntityExtractor
    participant LLM
    participant KVStore
    participant VectorDB
    participant GraphDB
    participant DocStatus
    
    User->>LightRAG: insert(document)
    
    activate LightRAG
    LightRAG->>DocStatus: Create document status entry (PENDING)
    DocStatus-->>LightRAG: Document ID
    
    LightRAG->>DocumentProcessor: Process document
    activate DocumentProcessor
    
    DocumentProcessor->>DocStatus: Update status (PROCESSING)
    
    alt PDF or Image Document
        DocumentProcessor->>DocumentProcessor: Extract text and visual elements
    end
    
    DocumentProcessor->>Chunker: Chunk document text
    activate Chunker
    
    Chunker->>Chunker: Apply tokenization
    Chunker->>Chunker: Split by chunk_token_size
    Chunker->>Chunker: Apply chunk_overlap_token_size
    
    alt Special Content Detected
        Chunker->>DiagramAnalyzer: Extract and analyze diagrams
        activate DiagramAnalyzer
        DiagramAnalyzer->>LLM: Request diagram description
        LLM-->>DiagramAnalyzer: Diagram description
        DiagramAnalyzer->>Chunker: Return diagram entities and relationships
        deactivate DiagramAnalyzer
        
        Chunker->>FormulaInterpreter: Extract and interpret formulas
        activate FormulaInterpreter
        FormulaInterpreter->>LLM: Request formula interpretation
        LLM-->>FormulaInterpreter: Formula interpretation
        FormulaInterpreter->>Chunker: Return formula entities and relationships
        deactivate FormulaInterpreter
    end
    
    Chunker-->>DocumentProcessor: Return chunks (TextChunkSchema[])
    deactivate Chunker
    
    DocumentProcessor->>KVStore: Store text chunks
    
    loop For each chunk
        DocumentProcessor->>EntityExtractor: Extract entities and relationships
        activate EntityExtractor
        
        EntityExtractor->>LLM: Submit chunk with entity extraction prompt
        LLM-->>EntityExtractor: Raw entity extraction result
        
        EntityExtractor->>EntityExtractor: Parse entity results
        EntityExtractor->>EntityExtractor: Normalize entity names
        
        EntityExtractor->>LLM: Submit chunk with relationship extraction prompt
        LLM-->>EntityExtractor: Raw relationship extraction result
        
        EntityExtractor->>EntityExtractor: Parse relationship results
        EntityExtractor->>EntityExtractor: Normalize relationship data
        
        EntityExtractor-->>DocumentProcessor: Return entities and relationships
        deactivate EntityExtractor
        
        DocumentProcessor->>DocumentProcessor: Collect entities and relationships
    end
    
    DocumentProcessor->>DocumentProcessor: Merge duplicate entities
    
    loop For each entity
        DocumentProcessor->>GraphDB: Check if entity exists
        GraphDB-->>DocumentProcessor: Entity exists/doesn't exist
        
        alt Entity exists
            DocumentProcessor->>DocumentProcessor: Merge with existing entity
            
            alt Description too long
                DocumentProcessor->>LLM: Summarize entity description
                LLM-->>DocumentProcessor: Summarized description
            end
        end
        
        DocumentProcessor->>GraphDB: Upsert entity node
        DocumentProcessor->>VectorDB: Store entity embedding
    end
    
    loop For each relationship
        DocumentProcessor->>GraphDB: Check if edge exists
        GraphDB-->>DocumentProcessor: Edge exists/doesn't exist
        
        alt Edge exists
            DocumentProcessor->>DocumentProcessor: Merge with existing relationship
            
            alt Description too long
                DocumentProcessor->>LLM: Summarize relationship description
                LLM-->>DocumentProcessor: Summarized description
            end
        end
        
        DocumentProcessor->>GraphDB: Upsert relationship edge
        DocumentProcessor->>VectorDB: Store relationship embedding
    end
    
    DocumentProcessor->>DocStatus: Update status (PROCESSED)
    DocumentProcessor-->>LightRAG: Processing complete
    deactivate DocumentProcessor
    
    LightRAG->>GraphDB: Commit graph updates (index_done_callback)
    LightRAG->>VectorDB: Commit vector updates (index_done_callback)
    LightRAG->>KVStore: Commit KV updates (index_done_callback)
    
    LightRAG-->>User: Document processed successfully
    deactivate LightRAG
```