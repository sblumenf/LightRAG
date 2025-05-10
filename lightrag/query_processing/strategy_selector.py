"""
Strategy selector module for LightRAG.

This module provides functionality for selecting the optimal retrieval strategy
based on query analysis.
"""

import logging
from typing import Dict, Any, List, Optional

from ..config_loader import get_enhanced_config

# Set up logger
logger = logging.getLogger(__name__)


def select_retrieval_strategy(processed_query_data: Dict[str, Any]) -> str:
    """
    Select the optimal retrieval strategy based on query analysis.

    Args:
        processed_query_data: Query analysis from process_query

    Returns:
        String indicating recommended strategy ("vector", "graph", "hybrid", "naive", or "mix")
    """
    # Use the QueryStrategySelector for more sophisticated analysis
    selector = QueryStrategySelector()
    result = selector.determine_strategy(
        processed_query_data.get('original_query', ''),
        processed_query_data
    )

    # Return the selected strategy string
    return result["strategy"]


class QueryStrategySelector:
    """
    Selects optimal retrieval strategy based on query analysis.
    """

    def __init__(self):
        """Initialize the strategy selector with defined indicator terms."""
        config = get_enhanced_config()
        
        # Terms indicating graph-based retrieval would be beneficial
        self.graph_indicators = config.graph_intent_indicators or [
            'related', 'connected', 'relationship', 'connection', 'link', 'between',
            'compare', 'difference', 'similar', 'versus', 'vs', 'contrast',
            'cause', 'effect', 'impact', 'influence', 'leads to',
            'prerequisite', 'requirement', 'depends on'
        ]

        # Terms indicating vector-based retrieval would be beneficial
        self.vector_indicators = config.vector_intent_indicators or [
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
        elif vector_score > vector_score + 1 or 'definition' in intent:
            # Prioritize vector strategy for definition queries
            strategy = "vector"
            confidence = vector_confidence
        # Default to hybrid when scores are close
        else:
            strategy = "hybrid"
            confidence = hybrid_confidence

        # Map strategies to LightRAG modes
        strategy_mapping = {
            "vector": "naive",
            "graph": "global",
            "hybrid": "hybrid",
        }
        
        lightrag_strategy = strategy_mapping.get(strategy, "hybrid")

        return {
            "strategy": lightrag_strategy,
            "confidence": confidence,
            "scores": {
                "graph": graph_score,
                "vector": vector_score,
                "query_complexity": query_length
            }
        }
