"""
Evaluation package for LightRAG.

This package provides utilities for evaluating the quality of LightRAG's
knowledge graph, retrieval, and response generation.
"""

from .kg_quality import (
    evaluate_kg_quality,
    evaluate_schema_conformance,
    evaluate_entity_resolution,
    evaluate_relationship_quality,
    KGQualityMetrics
)

from .retrieval_relevance import (
    evaluate_retrieval_relevance,
    evaluate_retrieval_diversity,
    evaluate_retrieval_coverage,
    RetrievalRelevanceMetrics
)

from .response_quality import (
    evaluate_response_quality,
    evaluate_reasoning_quality,
    evaluate_citation_quality,
    evaluate_factual_accuracy,
    ResponseQualityMetrics
)

from .diagram_entity_evaluation import (
    evaluate_diagram_entity_extraction,
    DiagramEntityEvaluator
)

__all__ = [
    'evaluate_kg_quality',
    'evaluate_schema_conformance',
    'evaluate_entity_resolution',
    'evaluate_relationship_quality',
    'KGQualityMetrics',
    'evaluate_retrieval_relevance',
    'evaluate_retrieval_diversity',
    'evaluate_retrieval_coverage',
    'RetrievalRelevanceMetrics',
    'evaluate_response_quality',
    'evaluate_reasoning_quality',
    'evaluate_citation_quality',
    'evaluate_factual_accuracy',
    'ResponseQualityMetrics',
    'evaluate_diagram_entity_extraction',
    'DiagramEntityEvaluator'
]
