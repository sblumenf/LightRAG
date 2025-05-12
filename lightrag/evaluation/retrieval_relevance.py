"""
Retrieval relevance evaluation for LightRAG.

This module provides utilities for evaluating the relevance, diversity, and
coverage of LightRAG's retrieval results.
"""

import asyncio
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
import numpy as np

from lightrag import LightRAG, QueryParam


@dataclass
class RetrievalRelevanceMetrics:
    """Class for storing retrieval relevance metrics."""
    relevance_score: float = 0.0  # Average relevance score
    diversity_score: float = 0.0  # Diversity of retrieved results
    coverage_score: float = 0.0  # Coverage of query aspects
    precision_at_k: Dict[int, float] = field(default_factory=dict)  # Precision at different k values
    recall_at_k: Dict[int, float] = field(default_factory=dict)  # Recall at different k values
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)  # NDCG at different k values
    mrr: float = 0.0  # Mean Reciprocal Rank
    avg_result_count: float = 0.0  # Average number of results returned
    query_count: int = 0  # Number of queries evaluated

    def to_dict(self) -> Dict[str, Any]:
        """Convert the metrics to a dictionary."""
        return asdict(self)


async def evaluate_retrieval_relevance(
    rag: LightRAG,
    queries: List[str],
    relevance_judgments: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    modes: List[str] = ["naive", "local", "global", "hybrid", "auto"],
    top_k: int = 10,
    use_intelligent_retrieval: bool = True
) -> Dict[str, RetrievalRelevanceMetrics]:
    """
    Evaluate the relevance of retrieval results for different modes.

    Args:
        rag: The LightRAG instance to evaluate
        queries: List of queries to evaluate
        relevance_judgments: Optional dictionary mapping queries to lists of relevant items with relevance scores
        modes: List of retrieval modes to evaluate
        top_k: Number of results to retrieve
        use_intelligent_retrieval: Whether to use intelligent retrieval

    Returns:
        Dict[str, RetrievalRelevanceMetrics]: Retrieval relevance metrics for each mode
    """
    results = {}
    
    for mode in modes:
        # Initialize metrics for this mode
        metrics = RetrievalRelevanceMetrics()
        metrics.query_count = len(queries)
        
        # Process each query
        relevance_scores = []
        diversity_scores = []
        coverage_scores = []
        result_counts = []
        precision_values = {k: [] for k in [1, 3, 5, 10]}
        recall_values = {k: [] for k in [1, 3, 5, 10]}
        ndcg_values = {k: [] for k in [1, 3, 5, 10]}
        reciprocal_ranks = []
        
        for i, query in enumerate(queries):
            # Set up query parameters
            param = QueryParam(
                mode=mode,
                top_k=top_k,
                use_intelligent_retrieval=use_intelligent_retrieval,
                generate_response=False  # We only want the retrieved items
            )
            
            # Get retrieval results
            retrieval_result = await rag.aquery(query, param=param)
            
            # Extract retrieved items
            if isinstance(retrieval_result, dict) and "context_items" in retrieval_result:
                retrieved_items = retrieval_result["context_items"]
            elif isinstance(retrieval_result, list):
                retrieved_items = retrieval_result
            else:
                retrieved_items = []
            
            # Count results
            result_counts.append(len(retrieved_items))
            
            # Calculate relevance metrics
            if relevance_judgments and query in relevance_judgments:
                # We have ground truth relevance judgments for this query
                relevant_items = relevance_judgments[query]
                
                # Calculate relevance score
                relevance_score = calculate_relevance_score(retrieved_items, relevant_items)
                relevance_scores.append(relevance_score)
                
                # Calculate precision and recall at different k values
                for k in precision_values.keys():
                    if k <= len(retrieved_items):
                        precision = calculate_precision_at_k(retrieved_items[:k], relevant_items)
                        recall = calculate_recall_at_k(retrieved_items[:k], relevant_items)
                        ndcg = calculate_ndcg_at_k(retrieved_items[:k], relevant_items, k)
                        
                        precision_values[k].append(precision)
                        recall_values[k].append(recall)
                        ndcg_values[k].append(ndcg)
                
                # Calculate Mean Reciprocal Rank
                rr = calculate_reciprocal_rank(retrieved_items, relevant_items)
                reciprocal_ranks.append(rr)
            else:
                # No ground truth, use a heuristic approach
                relevance_score = estimate_relevance(query, retrieved_items)
                relevance_scores.append(relevance_score)
            
            # Calculate diversity score
            diversity_score = calculate_diversity(retrieved_items)
            diversity_scores.append(diversity_score)
            
            # Calculate coverage score
            coverage_score = calculate_coverage(query, retrieved_items)
            coverage_scores.append(coverage_score)
        
        # Calculate average metrics
        metrics.relevance_score = np.mean(relevance_scores) if relevance_scores else 0.0
        metrics.diversity_score = np.mean(diversity_scores) if diversity_scores else 0.0
        metrics.coverage_score = np.mean(coverage_scores) if coverage_scores else 0.0
        metrics.avg_result_count = np.mean(result_counts) if result_counts else 0.0
        
        # Calculate precision, recall, and NDCG at different k values
        for k in precision_values.keys():
            metrics.precision_at_k[k] = np.mean(precision_values[k]) if precision_values[k] else 0.0
            metrics.recall_at_k[k] = np.mean(recall_values[k]) if recall_values[k] else 0.0
            metrics.ndcg_at_k[k] = np.mean(ndcg_values[k]) if ndcg_values[k] else 0.0
        
        # Calculate Mean Reciprocal Rank
        metrics.mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        # Store metrics for this mode
        results[mode] = metrics
    
    return results


def calculate_relevance_score(
    retrieved_items: List[Dict[str, Any]],
    relevant_items: List[Dict[str, Any]]
) -> float:
    """
    Calculate the relevance score for retrieved items.

    Args:
        retrieved_items: List of retrieved items
        relevant_items: List of relevant items with relevance scores

    Returns:
        float: Relevance score between 0 and 1
    """
    if not retrieved_items or not relevant_items:
        return 0.0
    
    # Create a mapping of item IDs to relevance scores
    relevance_map = {item["id"]: item.get("relevance", 1.0) for item in relevant_items}
    
    # Calculate the relevance score for each retrieved item
    total_score = 0.0
    for i, item in enumerate(retrieved_items):
        item_id = item.get("id")
        if item_id in relevance_map:
            # Apply a position discount (items at the top are more important)
            position_discount = 1.0 / (i + 1)
            total_score += relevance_map[item_id] * position_discount
    
    # Normalize by the maximum possible score
    max_score = sum(1.0 / (i + 1) for i in range(min(len(retrieved_items), len(relevant_items))))
    
    return total_score / max_score if max_score > 0 else 0.0


def estimate_relevance(query: str, retrieved_items: List[Dict[str, Any]]) -> float:
    """
    Estimate the relevance of retrieved items when no ground truth is available.

    Args:
        query: The query
        retrieved_items: List of retrieved items

    Returns:
        float: Estimated relevance score between 0 and 1
    """
    if not retrieved_items:
        return 0.0
    
    # Simple heuristic: check if query terms appear in the content
    query_terms = set(query.lower().split())
    
    # Calculate the relevance score for each retrieved item
    total_score = 0.0
    for i, item in enumerate(retrieved_items):
        content = item.get("content", "").lower()
        
        # Count how many query terms appear in the content
        matching_terms = sum(1 for term in query_terms if term in content)
        term_ratio = matching_terms / len(query_terms) if query_terms else 0
        
        # Apply a position discount (items at the top are more important)
        position_discount = 1.0 / (i + 1)
        total_score += term_ratio * position_discount
    
    # Normalize by the maximum possible score
    max_score = sum(1.0 / (i + 1) for i in range(len(retrieved_items)))
    
    return total_score / max_score if max_score > 0 else 0.0


def calculate_diversity(retrieved_items: List[Dict[str, Any]]) -> float:
    """
    Calculate the diversity of retrieved items.

    Args:
        retrieved_items: List of retrieved items

    Returns:
        float: Diversity score between 0 and 1
    """
    if not retrieved_items or len(retrieved_items) < 2:
        return 0.0
    
    # Extract content from items
    contents = [item.get("content", "") for item in retrieved_items]
    
    # Calculate pairwise similarity
    total_similarity = 0.0
    pair_count = 0
    
    for i in range(len(contents)):
        for j in range(i + 1, len(contents)):
            similarity = calculate_text_similarity(contents[i], contents[j])
            total_similarity += similarity
            pair_count += 1
    
    # Calculate average similarity
    avg_similarity = total_similarity / pair_count if pair_count > 0 else 0.0
    
    # Diversity is the opposite of similarity
    return 1.0 - avg_similarity


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate the similarity between two texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        float: Similarity score between 0 and 1
    """
    # Simple Jaccard similarity
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def calculate_coverage(query: str, retrieved_items: List[Dict[str, Any]]) -> float:
    """
    Calculate how well the retrieved items cover different aspects of the query.

    Args:
        query: The query
        retrieved_items: List of retrieved items

    Returns:
        float: Coverage score between 0 and 1
    """
    if not retrieved_items:
        return 0.0
    
    # Simple heuristic: extract key terms from the query
    query_terms = set(query.lower().split())
    
    # Count how many query terms are covered by the retrieved items
    covered_terms = set()
    for item in retrieved_items:
        content = item.get("content", "").lower()
        for term in query_terms:
            if term in content:
                covered_terms.add(term)
    
    # Calculate coverage
    return len(covered_terms) / len(query_terms) if query_terms else 0.0


def calculate_precision_at_k(
    retrieved_items: List[Dict[str, Any]],
    relevant_items: List[Dict[str, Any]],
    k: int = None
) -> float:
    """
    Calculate precision at k.

    Args:
        retrieved_items: List of retrieved items
        relevant_items: List of relevant items
        k: Number of items to consider (if None, use all retrieved items)

    Returns:
        float: Precision at k
    """
    if not retrieved_items:
        return 0.0
    
    # Use all retrieved items if k is not specified
    if k is None:
        k = len(retrieved_items)
    else:
        k = min(k, len(retrieved_items))
    
    # Create a set of relevant item IDs
    relevant_ids = {item["id"] for item in relevant_items}
    
    # Count relevant items in the top k
    relevant_count = sum(1 for item in retrieved_items[:k] if item.get("id") in relevant_ids)
    
    return relevant_count / k


def calculate_recall_at_k(
    retrieved_items: List[Dict[str, Any]],
    relevant_items: List[Dict[str, Any]],
    k: int = None
) -> float:
    """
    Calculate recall at k.

    Args:
        retrieved_items: List of retrieved items
        relevant_items: List of relevant items
        k: Number of items to consider (if None, use all retrieved items)

    Returns:
        float: Recall at k
    """
    if not retrieved_items or not relevant_items:
        return 0.0
    
    # Use all retrieved items if k is not specified
    if k is None:
        k = len(retrieved_items)
    else:
        k = min(k, len(retrieved_items))
    
    # Create a set of relevant item IDs
    relevant_ids = {item["id"] for item in relevant_items}
    
    # Count relevant items in the top k
    relevant_count = sum(1 for item in retrieved_items[:k] if item.get("id") in relevant_ids)
    
    return relevant_count / len(relevant_ids)


def calculate_ndcg_at_k(
    retrieved_items: List[Dict[str, Any]],
    relevant_items: List[Dict[str, Any]],
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    Args:
        retrieved_items: List of retrieved items
        relevant_items: List of relevant items with relevance scores
        k: Number of items to consider

    Returns:
        float: NDCG at k
    """
    if not retrieved_items or not relevant_items or k <= 0:
        return 0.0
    
    # Create a mapping of item IDs to relevance scores
    relevance_map = {item["id"]: item.get("relevance", 1.0) for item in relevant_items}
    
    # Calculate DCG
    dcg = 0.0
    for i in range(min(k, len(retrieved_items))):
        item_id = retrieved_items[i].get("id")
        if item_id in relevance_map:
            # Apply log2(i+2) discount
            dcg += relevance_map[item_id] / np.log2(i + 2)
    
    # Calculate ideal DCG
    ideal_order = sorted(relevance_map.values(), reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_order))):
        idcg += ideal_order[i] / np.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_reciprocal_rank(
    retrieved_items: List[Dict[str, Any]],
    relevant_items: List[Dict[str, Any]]
) -> float:
    """
    Calculate Reciprocal Rank.

    Args:
        retrieved_items: List of retrieved items
        relevant_items: List of relevant items

    Returns:
        float: Reciprocal Rank
    """
    if not retrieved_items or not relevant_items:
        return 0.0
    
    # Create a set of relevant item IDs
    relevant_ids = {item["id"] for item in relevant_items}
    
    # Find the first relevant item
    for i, item in enumerate(retrieved_items):
        if item.get("id") in relevant_ids:
            return 1.0 / (i + 1)
    
    return 0.0


async def evaluate_retrieval_diversity(
    rag: LightRAG,
    queries: List[str],
    mode: str = "naive",
    top_k: int = 10,
    use_intelligent_retrieval: bool = True
) -> float:
    """
    Evaluate the diversity of retrieval results.

    Args:
        rag: The LightRAG instance to evaluate
        queries: List of queries to evaluate
        mode: Retrieval mode to use
        top_k: Number of results to retrieve
        use_intelligent_retrieval: Whether to use intelligent retrieval

    Returns:
        float: Average diversity score
    """
    diversity_scores = []
    
    for query in queries:
        # Set up query parameters
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            use_intelligent_retrieval=use_intelligent_retrieval,
            generate_response=False  # We only want the retrieved items
        )
        
        # Get retrieval results
        retrieval_result = await rag.aquery(query, param=param)
        
        # Extract retrieved items
        if isinstance(retrieval_result, dict) and "context_items" in retrieval_result:
            retrieved_items = retrieval_result["context_items"]
        elif isinstance(retrieval_result, list):
            retrieved_items = retrieval_result
        else:
            retrieved_items = []
        
        # Calculate diversity score
        diversity_score = calculate_diversity(retrieved_items)
        diversity_scores.append(diversity_score)
    
    # Calculate average diversity score
    return np.mean(diversity_scores) if diversity_scores else 0.0


async def evaluate_retrieval_coverage(
    rag: LightRAG,
    queries: List[str],
    mode: str = "naive",
    top_k: int = 10,
    use_intelligent_retrieval: bool = True
) -> float:
    """
    Evaluate the coverage of retrieval results.

    Args:
        rag: The LightRAG instance to evaluate
        queries: List of queries to evaluate
        mode: Retrieval mode to use
        top_k: Number of results to retrieve
        use_intelligent_retrieval: Whether to use intelligent retrieval

    Returns:
        float: Average coverage score
    """
    coverage_scores = []
    
    for query in queries:
        # Set up query parameters
        param = QueryParam(
            mode=mode,
            top_k=top_k,
            use_intelligent_retrieval=use_intelligent_retrieval,
            generate_response=False  # We only want the retrieved items
        )
        
        # Get retrieval results
        retrieval_result = await rag.aquery(query, param=param)
        
        # Extract retrieved items
        if isinstance(retrieval_result, dict) and "context_items" in retrieval_result:
            retrieved_items = retrieval_result["context_items"]
        elif isinstance(retrieval_result, list):
            retrieved_items = retrieval_result
        else:
            retrieved_items = []
        
        # Calculate coverage score
        coverage_score = calculate_coverage(query, retrieved_items)
        coverage_scores.append(coverage_score)
    
    # Calculate average coverage score
    return np.mean(coverage_scores) if coverage_scores else 0.0
