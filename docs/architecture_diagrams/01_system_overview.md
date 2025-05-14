# System Overview Diagram

This diagram provides a high-level overview of the entire LightRAG system architecture, showing the major components and their relationships.

## Diagram Description

The system is organized into several layers:

1. **API Layer**: The entry point for users and applications to interact with LightRAG via FastAPI
2. **Core Engine**: The central LightRAG class that orchestrates all operations
3. **Storage Layer**: Various storage implementations for different data types:
   - Key-Value Storage: For document chunks and LLM cache
   - Vector Storage: For embedding vectors
   - Knowledge Graph Storage: For entities and relationships
   - Document Status Storage: For tracking processing status
4. **Processing Layer**: Components for document analysis and information extraction:
   - Document Processing: Handles chunking and initial processing
   - Entity/Relationship Extraction: Uses LLMs to identify entities and relationships
   - Diagram Analysis: Processes visual elements in documents
   - Formula Interpretation: Handles mathematical content
5. **Retrieval Layer**: Different query modes for information retrieval:
   - Local Mode: Entity-based retrieval
   - Global Mode: Relationship-based retrieval
   - Hybrid Mode: Combines local and global approaches
   - Naive Mode: Simple vector similarity search
   - Mix Mode: Integrates knowledge graph and vector search
6. **LLM Layer**: Components for working with external language models:
   - LLM Generation: For response creation
   - Embedding Generation: For creating vector representations
   - Enhanced Embedding: For improved semantic understanding

The diagram also shows connections to external systems like LLM services and databases.

```mermaid
graph TB
    subgraph "LightRAG System"
        API["API Layer<br>(FastAPI)"]
        Core["Core Engine<br>(LightRAG Class)"]
        
        subgraph "Storage Layer"
            KV["Key-Value Storage<br>(Document Chunks, LLM Cache)"]
            Vector["Vector Storage<br>(Embeddings)"]
            Graph["Knowledge Graph Storage<br>(Entities & Relationships)"]
            DocStatus["Document Status Storage<br>(Processing Status)"]
        end
        
        subgraph "Processing Layer"
            DocProc["Document Processing<br>(Chunking, Extraction)"]
            EntityEx["Entity Extraction<br>(LLM-based)"]
            RelEx["Relationship Extraction<br>(LLM-based)"]
            DiagramEx["Diagram Analysis<br>(Vision + LLM)"]
            FormulaEx["Formula Interpretation<br>(Specialized LLM)"]
        end
        
        subgraph "Retrieval Layer"
            Local["Local Mode<br>(Entity-based)"]
            Global["Global Mode<br>(Relationship-based)"]
            Hybrid["Hybrid Mode<br>(Combined)"]
            Naive["Naive Mode<br>(Simple Vector)"]
            Mix["Mix Mode<br>(KG + Vector)"]
        end
        
        subgraph "LLM Layer"
            LLMGen["LLM Generation<br>(Response Creation)"]
            LLMEmb["Embedding Generation<br>(Vector Creation)"]
            EnhEmb["Enhanced Embedding<br>(Semantic Understanding)"]
        end
    end
    
    Users["End Users / Applications"]
    LLMs["External LLM Services<br>(OpenAI, Azure, Gemini, etc.)"]
    DBs["External Databases<br>(Neo4j, Postgres, Milvus, etc.)"]
    
    Users <--> API
    API <--> Core
    Core <--> Processing Layer
    Core <--> Storage Layer
    Core <--> Retrieval Layer
    Core <--> LLM Layer
    LLM Layer <--> LLMs
    Storage Layer <--> DBs
    
    %% Detailed connections within the system
    DocProc --> EntityEx
    DocProc --> RelEx
    DocProc --> DiagramEx
    DocProc --> FormulaEx
    
    EntityEx --> Graph
    RelEx --> Graph
    
    DocProc --> KV
    EntityEx --> Vector
    RelEx --> Vector
    
    Local --> KV
    Local --> Vector
    Local --> Graph
    
    Global --> KV
    Global --> Vector
    Global --> Graph
    
    Hybrid --> Local
    Hybrid --> Global
    
    Naive --> KV
    Naive --> Vector
    
    Mix --> KV
    Mix --> Vector
    Mix --> Graph
    
    LLMGen <--> LLMs
    LLMEmb <--> LLMs
    EnhEmb --> LLMEmb
    
    classDef core fill:#f9f,stroke:#333,stroke-width:2px
    classDef storage fill:#bbf,stroke:#333,stroke-width:1px
    classDef process fill:#bfb,stroke:#333,stroke-width:1px
    classDef retrieval fill:#fbf,stroke:#333,stroke-width:1px
    classDef llm fill:#fbb,stroke:#333,stroke-width:1px
    classDef external fill:#ddd,stroke:#333,stroke-width:1px
    
    class Core core
    class KV,Vector,Graph,DocStatus storage
    class DocProc,EntityEx,RelEx,DiagramEx,FormulaEx process
    class Local,Global,Hybrid,Naive,Mix retrieval
    class LLMGen,LLMEmb,EnhEmb llm
    class Users,LLMs,DBs external
```