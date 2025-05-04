"""
Hybrid Retrieval Pipeline for GraphRAG tutor.

This module provides utility functions for hybrid retrieval combining
semantic search and graph-based retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Union

from ..retrieval.retriever import GraphRAGRetriever
from ..retrieval.domain_retriever import DomainSchemaRetriever
from ..knowledge_graph.neo4j_knowledge_graph import Neo4jKnowledgeGraph
from ..knowledge_graph.schema_loader import SchemaLoader
from config import settings

logger = logging.getLogger(__name__)

def create_retriever(
    neo4j_kg: Neo4jKnowledgeGraph,
    schema_loader: Optional[SchemaLoader] = None,
    use_domain_schema: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Union[GraphRAGRetriever, DomainSchemaRetriever]:
    """
    Create an appropriate retriever based on configuration.

    Args:
        neo4j_kg: Neo4jKnowledgeGraph instance
        schema_loader: Optional SchemaLoader instance
        use_financial_schema: Whether to use financial schema retriever
        config: Optional configuration parameters

    Returns:
        An initialized retriever instance
    """
    config = config or {}

    # Extract configuration parameters with defaults
    retrieval_config = config.get('retrieval', {})
    model_name = retrieval_config.get('model_name', settings.DEFAULT_GOOGLE_LLM_MODEL)
    embedding_model = retrieval_config.get('embedding_model', settings.DEFAULT_EMBEDDING_MODEL)
    retrieval_limit = retrieval_config.get('retrieval_limit', settings.DEFAULT_RETRIEVAL_LIMIT)
    similarity_threshold = retrieval_config.get('similarity_threshold', 0.6)
    max_related_depth = retrieval_config.get('max_related_depth', 2)

    # Create a config dictionary to pass to the retriever
    retriever_config = {
        'model_name': model_name,
        'embedding_model': embedding_model,
        'retrieval_limit': retrieval_limit,
        'similarity_threshold': similarity_threshold,
        'max_related_depth': max_related_depth
    }

    # Create appropriate retriever
    if use_domain_schema and schema_loader:
        logger.info("Creating DomainSchemaRetriever with schema integration")
        return DomainSchemaRetriever(
            knowledge_graph=neo4j_kg,
            schema_loader=schema_loader,
            config=retriever_config
        )
    else:
        logger.info("Creating basic GraphRAGRetriever")
        return GraphRAGRetriever(
            knowledge_graph=neo4j_kg,
            config=retriever_config
        )

def retrieve_for_query(
    query: str,
    retriever: Union[GraphRAGRetriever, DomainSchemaRetriever],
    strategy: str = "auto",
    entity_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve information for a given query using the specified strategy.

    Args:
        query: User query string
        retriever: Initialized retriever instance
        strategy: Retrieval strategy ("vector", "graph", "hybrid", or "auto")
        entity_type: Optional entity type to filter results

    Returns:
        Dict containing retrieval results
    """
    logger.info(f"Retrieving information for query: '{query}' using {strategy} strategy")

    # If entity type is specified, use entity-specific retrieval
    if entity_type:
        return retriever.retrieve_by_entity_type(query, entity_type)

    # For auto strategy, analyze query to determine best strategy
    if strategy == "auto":
        analysis = retriever.process_query(query)
        recommended_strategy = determine_optimal_strategy(query, analysis)
        logger.info(f"Auto strategy selection chose: {recommended_strategy}")
        strategy = recommended_strategy

    # Apply the appropriate retrieval strategy with weights based on query characteristics
    if strategy in ["vector", "graph", "hybrid"]:
        return retriever.retrieve(query, strategy)
    else:
        logger.warning(f"Unknown strategy '{strategy}', falling back to hybrid")
        return retriever.retrieve(query, "hybrid")

def analyze_query(query: str, retriever: Union[GraphRAGRetriever, DomainSchemaRetriever]) -> Dict[str, Any]:
    """
    Analyze a query without performing retrieval.

    Args:
        query: User query string
        retriever: Initialized retriever instance

    Returns:
        Dict containing query analysis, with strategy recommendation
    """
    logger.info(f"Analyzing query: '{query}'")
    analysis = retriever.process_query(query)

    # Enhance analysis with recommended strategy
    strategy_recommendation = determine_optimal_strategy(query, analysis)
    analysis['recommended_strategy'] = strategy_recommendation

    return analysis


class QueryStrategySelector:
    """
    Selects optimal retrieval strategy based on query analysis.
    """

    def __init__(self):
        """Initialize the strategy selector with defined indicator terms."""
        # Terms indicating graph-based retrieval would be beneficial
        self.graph_indicators = [
            'related', 'connected', 'relationship', 'connection', 'link', 'between',
            'compare', 'difference', 'similar', 'versus', 'vs', 'contrast',
            'cause', 'effect', 'impact', 'influence', 'leads to',
            'prerequisite', 'requirement', 'depends on'
        ]

        # Terms indicating vector-based retrieval would be beneficial
        self.vector_indicators = [
            'like', 'similar to', 'example of', 'such as',
            'about', 'concept of', 'definition', 'meaning',
            'explain', 'describe', 'summarize'
        ]

        # Intents that suggest graph traversal
        self.graph_intents = [
            'compare', 'contrast', 'relationship', 'dependency',
            'connection', 'causality', 'workflow', 'process'
        ]

        # Intents that suggest vector search
        self.vector_intents = [
            'define', 'summarize', 'explain', 'describe',
            'overview', 'concept', 'meaning'
        ]

    def determine_strategy(self, query: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine the optimal retrieval strategy based on query analysis.

        Args:
            query: User query string
            analysis: Query analysis from process_query

        Returns:
            Dict containing recommended strategy and confidence score
        """
        intent = analysis.get('intent', '').lower()
        entity_types = analysis.get('entity_types', [])
        keywords = analysis.get('keywords', [])

        # Calculate scores for each strategy
        query_lower = query.lower()

        # 1. Score graph strategy
        graph_score = sum(1 for indicator in self.graph_indicators if indicator in query_lower)

        # 2. Score vector strategy
        vector_score = sum(1 for indicator in self.vector_indicators if indicator in query_lower)

        # 3. Check intent-based indicators
        for graph_intent in self.graph_intents:
            if graph_intent in intent:
                graph_score += 2
                break

        for vector_intent in self.vector_intents:
            if vector_intent in intent:
                vector_score += 2
                break

        # 4. Check for entity types (schema-classified entities benefit from graph connections)
        if len(entity_types) >= 2:
            graph_score += 2  # Multiple entity types suggest relationships between them
        elif len(entity_types) == 1:
            graph_score += 1  # Single entity type still benefits from graph context

        # 5. Check query complexity (longer queries often benefit from hybrid approach)
        query_length = len(query.split())
        hybrid_bias = 0
        if query_length > 10:  # Complex query
            hybrid_bias = 2  # Increase hybrid bias for complex queries

        # 6. Apply entity-specific adjustments
        entity_mentions = set(entity_types).intersection(set(keywords))
        if len(entity_mentions) >= 2:  # Query explicitly mentions multiple entities by name
            graph_score += 2

        # 7. Calculate strategy confidence scores (0-1 scale)
        total_score = graph_score + vector_score + hybrid_bias
        if total_score == 0:
            total_score = 1  # Avoid division by zero

        # Ensure minimum confidence values
        graph_confidence = max(0.6, min(0.95, graph_score / total_score * 0.9))
        vector_confidence = max(0.6, min(0.95, vector_score / total_score * 0.9))
        hybrid_confidence = 0.8 if hybrid_bias > 0 else 0.6  # Higher for complex queries

        # Special case for definition intent - boost vector confidence
        if 'definition' in intent:
            vector_confidence = max(0.75, vector_confidence)

        # 8. Determine final strategy with confidence
        # For complex queries with multiple entity types, prefer hybrid
        if hybrid_bias >= 2 and query_length > 12:
            strategy = "hybrid"
            confidence = hybrid_confidence
        # For queries with impact/influence keywords, prefer hybrid
        elif 'impact' in query_lower or 'influence' in query_lower or 'affect' in query_lower:
            strategy = "hybrid"
            confidence = hybrid_confidence
        # For graph-heavy scores
        elif graph_score > vector_score + 1 and 'definition' not in intent:
            strategy = "graph"
            confidence = graph_confidence
        # For vector-heavy scores or definition queries
        elif vector_score > graph_score + 1 or 'definition' in intent:
            # Prioritize vector strategy for definition queries
            strategy = "vector"
            confidence = vector_confidence
        # Default to hybrid when scores are close
        else:
            strategy = "hybrid"
            confidence = hybrid_confidence

        return {
            "strategy": strategy,
            "confidence": confidence,
            "scores": {
                "graph": graph_score,
                "vector": vector_score,
                "query_complexity": query_length
            }
        }


def determine_optimal_strategy(query: str, analysis: Dict[str, Any]) -> str:
    """
    Determine the optimal retrieval strategy based on query analysis.

    Args:
        query: User query string
        analysis: Query analysis from process_query

    Returns:
        String indicating recommended strategy ("vector", "graph", or "hybrid")
    """
    # Use the QueryStrategySelector for more sophisticated analysis
    selector = QueryStrategySelector()
    result = selector.determine_strategy(query, analysis)

    # Just return the selected strategy string
    return result["strategy"]

def retrieve_related(
    chunk_id: str,
    retriever: Union[GraphRAGRetriever, DomainSchemaRetriever],
    relationship_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Retrieve nodes related to a specific chunk.

    Args:
        chunk_id: ID of the chunk to find relationships for
        retriever: Initialized retriever instance
        relationship_type: Optional relationship type to filter by

    Returns:
        Dict containing related nodes
    """
    logger.info(f"Retrieving nodes related to chunk {chunk_id}")

    # Use domain retriever's specialized method if available
    if isinstance(retriever, DomainSchemaRetriever):
        return retriever.retrieve_related_domain_concepts(chunk_id, relationship_type)

    # Otherwise, use the base retriever's method through a compatible interface
    query_analysis = {"intent": "related"}
    related_nodes = retriever._get_related_nodes(chunk_id, query_analysis)

    return {
        'chunk_id': chunk_id,
        'relationship_type': relationship_type,
        'results': related_nodes,
        'count': len(related_nodes)
    }