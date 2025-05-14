# Data Flow Diagram

This flowchart illustrates how data moves through the LightRAG system, from input documents and queries to storage and response generation.

## Diagram Description

The diagram tracks two primary data flows:

### Document Processing Flow

1. **Input**: Documents enter the system and begin the processing pipeline
2. **Chunking**: Documents are divided into manageable chunks based on token size
3. **Analysis**: Different types of analysis are performed:
   - Entity Extraction identifies named entities
   - Relationship Extraction identifies connections between entities
   - Diagram Analysis extracts information from visual elements
   - Formula Interpretation handles mathematical content
4. **Storage**: The processed information is stored in different storage systems:
   - Text chunks go to Key-Value Storage
   - Entity and relationship embeddings go to Vector Storage
   - Entity and relationship metadata go to Knowledge Graph Storage
   - Processing status is tracked in Document Status Storage

### Query Processing Flow

1. **Input**: User query enters the system
2. **Analysis**: The query is analyzed to extract:
   - Query intent and type
   - High-level and low-level keywords
3. **Strategy Selection**: Based on analysis, a retrieval strategy is chosen
4. **Retrieval**: Different modes retrieve information differently:
   - Local mode focuses on entities
   - Global mode focuses on relationships
   - Hybrid mode combines local and global approaches
   - Naive mode uses simple vector similarity
   - Mix mode integrates knowledge graph and vector retrieval
5. **Reranking**: Results are reranked based on relevance
6. **LLM Prompting**: Retrieved context is combined with the query in a prompt
7. **Response Generation**: The LLM generates a response based on the prompt

The diagram also shows how caching works, with results being stored for future queries and lookups happening before processing.

```mermaid
flowchart TD
    %% Input/Output nodes
    InputDoc["Input Document"]
    UserQuery["User Query"]
    Response["Generated Response"]
    
    %% Processing nodes
    Chunking["Text Chunking"]
    EntityExtraction["Entity Extraction"]
    RelationExtraction["Relationship Extraction"]
    DiagramAnalysis["Diagram Analysis"]
    FormulaInterpretation["Formula Interpretation"]
    
    %% Storage nodes
    KVstore[(Key-Value Storage)]
    VectorDB[(Vector Storage)]
    GraphDB[(Knowledge Graph)]
    DocStatus[(Document Status)]
    
    %% Query processing nodes
    QueryAnalysis["Query Analysis"]
    StrategySelection["Retrieval Strategy Selection"]
    KeywordExtraction["Keyword Extraction"]
    Retrieval{"Retrieval Mode"}
    LocalRetrieval["Local Retrieval"]
    GlobalRetrieval["Global Retrieval"]
    HybridRetrieval["Hybrid Retrieval"]
    NaiveRetrieval["Naive Retrieval"]
    MixRetrieval["Mix Retrieval"]
    Reranking["Result Reranking"]
    LLMPrompting["LLM Prompting"]
    
    %% Data flow for document processing
    InputDoc --> Chunking
    Chunking --> KVstore
    Chunking --> EntityExtraction
    Chunking --> RelationExtraction
    Chunking --> DiagramAnalysis
    Chunking --> FormulaInterpretation
    
    EntityExtraction --> GraphDB
    RelationExtraction --> GraphDB
    DiagramAnalysis --> EntityExtraction
    DiagramAnalysis --> RelationExtraction
    FormulaInterpretation --> EntityExtraction
    FormulaInterpretation --> RelationExtraction
    
    EntityExtraction --> VectorDB
    RelationExtraction --> VectorDB
    
    Chunking --> DocStatus
    
    %% Data flow for query processing
    UserQuery --> QueryAnalysis
    QueryAnalysis --> StrategySelection
    QueryAnalysis --> KeywordExtraction
    
    StrategySelection --> Retrieval
    KeywordExtraction --> |"High-level keywords"| GlobalRetrieval
    KeywordExtraction --> |"Low-level keywords"| LocalRetrieval
    
    Retrieval --> |"Mode: local"| LocalRetrieval
    Retrieval --> |"Mode: global"| GlobalRetrieval
    Retrieval --> |"Mode: hybrid"| HybridRetrieval
    Retrieval --> |"Mode: naive"| NaiveRetrieval
    Retrieval --> |"Mode: mix"| MixRetrieval
    
    LocalRetrieval --> |"Entity search"| VectorDB
    LocalRetrieval --> |"Entity retrieval"| GraphDB
    LocalRetrieval --> |"Content retrieval"| KVstore
    
    GlobalRetrieval --> |"Relationship search"| VectorDB
    GlobalRetrieval --> |"Relationship retrieval"| GraphDB
    GlobalRetrieval --> |"Content retrieval"| KVstore
    
    HybridRetrieval --> LocalRetrieval
    HybridRetrieval --> GlobalRetrieval
    
    NaiveRetrieval --> |"Vector similarity"| VectorDB
    NaiveRetrieval --> |"Content retrieval"| KVstore
    
    MixRetrieval --> |"KG retrieval"| GraphDB
    MixRetrieval --> |"Vector similarity"| VectorDB
    MixRetrieval --> |"Content retrieval"| KVstore
    
    LocalRetrieval --> Reranking
    GlobalRetrieval --> Reranking
    HybridRetrieval --> Reranking
    NaiveRetrieval --> Reranking
    MixRetrieval --> Reranking
    
    Reranking --> LLMPrompting
    LLMPrompting --> Response
    
    %% Caching flow
    LLMPrompting -.-> |"Cache result"| KVstore
    LLMPrompting -.-> |"Cache lookup"| KVstore

    %% Styling
    classDef input fill:#bbf,stroke:#333,stroke-width:1px
    classDef process fill:#bfb,stroke:#333,stroke-width:1px
    classDef storage fill:#fbb,stroke:#333,stroke-width:1px
    classDef decision fill:#fbf,stroke:#333,stroke-width:1px
    classDef output fill:#bbf,stroke:#333,stroke-width:1px
    
    class InputDoc,UserQuery input
    class Chunking,EntityExtraction,RelationExtraction,DiagramAnalysis,FormulaInterpretation,QueryAnalysis,StrategySelection,KeywordExtraction,LocalRetrieval,GlobalRetrieval,HybridRetrieval,NaiveRetrieval,MixRetrieval,Reranking,LLMPrompting process
    class KVstore,VectorDB,GraphDB,DocStatus storage
    class Retrieval decision
    class Response output
```