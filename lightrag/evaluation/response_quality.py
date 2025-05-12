"""
Response quality evaluation for LightRAG.

This module provides utilities for evaluating the quality of LightRAG's
responses, including reasoning quality, citation quality, and factual accuracy.
"""

import asyncio
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field

from lightrag import LightRAG, QueryParam


@dataclass
class ResponseQualityMetrics:
    """Class for storing response quality metrics."""
    overall_quality_score: float = 0.0  # Overall quality score
    reasoning_quality_score: float = 0.0  # Quality of reasoning
    citation_quality_score: float = 0.0  # Quality of citations
    factual_accuracy_score: float = 0.0  # Factual accuracy
    completeness_score: float = 0.0  # Completeness of the response
    coherence_score: float = 0.0  # Coherence of the response
    relevance_score: float = 0.0  # Relevance to the query
    has_reasoning: float = 0.0  # Percentage of responses with reasoning
    has_citations: float = 0.0  # Percentage of responses with citations
    avg_citation_count: float = 0.0  # Average number of citations per response
    query_count: int = 0  # Number of queries evaluated

    def to_dict(self) -> Dict[str, Any]:
        """Convert the metrics to a dictionary."""
        return asdict(self)


async def evaluate_response_quality(
    rag: LightRAG,
    queries: List[str],
    ground_truth: Optional[Dict[str, str]] = None,
    mode: str = "auto",
    use_cot: bool = True,
    use_enhanced_citations: bool = True,
    use_intelligent_retrieval: bool = True
) -> ResponseQualityMetrics:
    """
    Evaluate the quality of responses.

    Args:
        rag: The LightRAG instance to evaluate
        queries: List of queries to evaluate
        ground_truth: Optional dictionary mapping queries to ground truth answers
        mode: Retrieval mode to use
        use_cot: Whether to use Chain-of-Thought reasoning
        use_enhanced_citations: Whether to use enhanced citations
        use_intelligent_retrieval: Whether to use intelligent retrieval

    Returns:
        ResponseQualityMetrics: Response quality metrics
    """
    # Initialize metrics
    metrics = ResponseQualityMetrics()
    metrics.query_count = len(queries)
    
    # Process each query
    reasoning_scores = []
    citation_scores = []
    factual_scores = []
    completeness_scores = []
    coherence_scores = []
    relevance_scores = []
    has_reasoning_count = 0
    has_citations_count = 0
    citation_counts = []
    
    for query in queries:
        # Set up query parameters
        param = QueryParam(
            mode=mode,
            use_cot=use_cot,
            use_enhanced_citations=use_enhanced_citations,
            use_intelligent_retrieval=use_intelligent_retrieval
        )
        
        # Get response
        response = await rag.aquery(query, param=param)
        
        # Check if response has reasoning
        has_reasoning = "<reasoning>" in response or "Reasoning:" in response
        if has_reasoning:
            has_reasoning_count += 1
        
        # Check if response has citations
        has_citations = bool(re.search(r'\[\d+\]', response))
        if has_citations:
            has_citations_count += 1
            
            # Count citations
            citation_count = len(re.findall(r'\[\d+\]', response))
            citation_counts.append(citation_count)
        
        # Evaluate reasoning quality
        reasoning_score = evaluate_reasoning_quality(query, response)
        reasoning_scores.append(reasoning_score)
        
        # Evaluate citation quality
        citation_score = evaluate_citation_quality(response)
        citation_scores.append(citation_score)
        
        # Evaluate factual accuracy
        factual_score = evaluate_factual_accuracy(
            query, response, ground_truth.get(query) if ground_truth else None
        )
        factual_scores.append(factual_score)
        
        # Evaluate completeness
        completeness_score = evaluate_completeness(query, response)
        completeness_scores.append(completeness_score)
        
        # Evaluate coherence
        coherence_score = evaluate_coherence(response)
        coherence_scores.append(coherence_score)
        
        # Evaluate relevance
        relevance_score = evaluate_relevance(query, response)
        relevance_scores.append(relevance_score)
    
    # Calculate average metrics
    metrics.reasoning_quality_score = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0
    metrics.citation_quality_score = sum(citation_scores) / len(citation_scores) if citation_scores else 0.0
    metrics.factual_accuracy_score = sum(factual_scores) / len(factual_scores) if factual_scores else 0.0
    metrics.completeness_score = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
    metrics.coherence_score = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
    metrics.relevance_score = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    
    metrics.has_reasoning = has_reasoning_count / len(queries) if queries else 0.0
    metrics.has_citations = has_citations_count / len(queries) if queries else 0.0
    metrics.avg_citation_count = sum(citation_counts) / len(citation_counts) if citation_counts else 0.0
    
    # Calculate overall quality score
    metrics.overall_quality_score = (
        metrics.reasoning_quality_score * 0.25 +
        metrics.citation_quality_score * 0.15 +
        metrics.factual_accuracy_score * 0.25 +
        metrics.completeness_score * 0.15 +
        metrics.coherence_score * 0.1 +
        metrics.relevance_score * 0.1
    )
    
    return metrics


def evaluate_reasoning_quality(query: str, response: str) -> float:
    """
    Evaluate the quality of reasoning in the response.

    Args:
        query: The query
        response: The response

    Returns:
        float: Reasoning quality score between 0 and 1
    """
    # Extract reasoning section if present
    reasoning = ""
    if "<reasoning>" in response and "</reasoning>" in response:
        reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
    elif "Reasoning:" in response:
        reasoning_parts = response.split("Reasoning:", 1)
        if len(reasoning_parts) > 1:
            reasoning_end = reasoning_parts[1].find("Answer:") if "Answer:" in reasoning_parts[1] else None
            reasoning = reasoning_parts[1][:reasoning_end].strip() if reasoning_end else reasoning_parts[1].strip()
    
    if not reasoning:
        return 0.0
    
    # Evaluate reasoning quality based on several factors
    
    # 1. Length of reasoning (longer reasoning is generally more detailed)
    length_score = min(len(reasoning) / 500, 1.0)
    
    # 2. Structure (look for logical connectors)
    logical_connectors = ["because", "therefore", "thus", "since", "as a result", "consequently", "first", "second", "third", "finally"]
    connector_count = sum(1 for connector in logical_connectors if connector in reasoning.lower())
    structure_score = min(connector_count / 5, 1.0)
    
    # 3. Relevance to query (simple keyword matching)
    query_terms = set(query.lower().split())
    query_term_count = sum(1 for term in query_terms if term in reasoning.lower())
    relevance_score = query_term_count / len(query_terms) if query_terms else 0.0
    
    # 4. Depth (look for multiple sentences/points)
    sentences = [s.strip() for s in re.split(r'[.!?]', reasoning) if s.strip()]
    depth_score = min(len(sentences) / 5, 1.0)
    
    # Combine scores with weights
    return (
        length_score * 0.2 +
        structure_score * 0.3 +
        relevance_score * 0.3 +
        depth_score * 0.2
    )


def evaluate_citation_quality(response: str) -> float:
    """
    Evaluate the quality of citations in the response.

    Args:
        response: The response

    Returns:
        float: Citation quality score between 0 and 1
    """
    # Check if response has citations
    citations = re.findall(r'\[\d+\]', response)
    if not citations:
        return 0.0
    
    # Check if response has a sources section
    has_sources_section = "Sources:" in response or "References:" in response
    
    # Extract sources section if present
    sources_section = ""
    if "Sources:" in response:
        sources_parts = response.split("Sources:", 1)
        if len(sources_parts) > 1:
            sources_section = sources_parts[1].strip()
    elif "References:" in response:
        sources_parts = response.split("References:", 1)
        if len(sources_parts) > 1:
            sources_section = sources_parts[1].strip()
    
    # Count unique citation numbers
    unique_citations = set(re.findall(r'\[(\d+)\]', response))
    
    # Check if all citations are defined in the sources section
    defined_citations = set()
    if sources_section:
        for citation in unique_citations:
            if f"[{citation}]" in sources_section:
                defined_citations.add(citation)
    
    # Calculate scores
    citation_count_score = min(len(citations) / 5, 1.0)
    unique_citation_score = len(unique_citations) / len(citations) if citations else 0.0
    defined_citation_score = len(defined_citations) / len(unique_citations) if unique_citations else 0.0
    has_sources_score = 1.0 if has_sources_section else 0.0
    
    # Combine scores with weights
    return (
        citation_count_score * 0.2 +
        unique_citation_score * 0.2 +
        defined_citation_score * 0.4 +
        has_sources_score * 0.2
    )


def evaluate_factual_accuracy(
    query: str,
    response: str,
    ground_truth: Optional[str] = None
) -> float:
    """
    Evaluate the factual accuracy of the response.

    Args:
        query: The query
        response: The response
        ground_truth: Optional ground truth answer

    Returns:
        float: Factual accuracy score between 0 and 1
    """
    if ground_truth:
        # If we have ground truth, compare the response to it
        # This is a simple text similarity approach
        return calculate_text_similarity(response, ground_truth)
    else:
        # Without ground truth, we use heuristics
        
        # 1. Check for hedging language (indicates uncertainty)
        hedging_terms = ["might", "may", "could", "possibly", "perhaps", "seems", "appears"]
        hedging_count = sum(1 for term in hedging_terms if term in response.lower())
        hedging_score = max(1.0 - (hedging_count / 5), 0.0)
        
        # 2. Check for specific details (dates, numbers, names)
        has_dates = bool(re.search(r'\b\d{4}\b', response))  # Years
        has_numbers = bool(re.search(r'\b\d+\b', response))  # Any numbers
        has_names = bool(re.search(r'\b[A-Z][a-z]+\b', response))  # Capitalized words
        
        detail_score = (has_dates + has_numbers + has_names) / 3
        
        # 3. Check for citations
        has_citations = bool(re.search(r'\[\d+\]', response))
        citation_score = 1.0 if has_citations else 0.0
        
        # Combine scores with weights
        return (
            hedging_score * 0.3 +
            detail_score * 0.3 +
            citation_score * 0.4
        )


def evaluate_completeness(query: str, response: str) -> float:
    """
    Evaluate the completeness of the response.

    Args:
        query: The query
        response: The response

    Returns:
        float: Completeness score between 0 and 1
    """
    # 1. Length of response (longer responses are generally more complete)
    length_score = min(len(response) / 1000, 1.0)
    
    # 2. Check if response addresses all parts of the query
    query_parts = query.split("?")
    query_parts = [part.strip() + "?" for part in query_parts if part.strip()]
    
    if len(query_parts) <= 1:
        # Single question, check for query terms
        query_terms = set(query.lower().split())
        query_term_count = sum(1 for term in query_terms if term in response.lower())
        query_coverage_score = query_term_count / len(query_terms) if query_terms else 0.0
    else:
        # Multiple questions, check if each is addressed
        addressed_parts = 0
        for part in query_parts:
            part_terms = set(part.lower().split())
            part_term_count = sum(1 for term in part_terms if term in response.lower())
            if part_term_count / len(part_terms) > 0.5:
                addressed_parts += 1
        
        query_coverage_score = addressed_parts / len(query_parts)
    
    # 3. Structure completeness (introduction, body, conclusion)
    has_intro = len(response.split("\n")[0]) > 20 if response else False
    has_conclusion = len(response.split("\n")[-1]) > 20 if response else False
    structure_score = (has_intro + has_conclusion) / 2
    
    # Combine scores with weights
    return (
        length_score * 0.3 +
        query_coverage_score * 0.5 +
        structure_score * 0.2
    )


def evaluate_coherence(response: str) -> float:
    """
    Evaluate the coherence of the response.

    Args:
        response: The response

    Returns:
        float: Coherence score between 0 and 1
    """
    if not response:
        return 0.0
    
    # 1. Check for paragraph structure
    paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
    paragraph_score = min(len(paragraphs) / 3, 1.0)
    
    # 2. Check for coherence markers
    coherence_markers = ["first", "second", "third", "finally", "moreover", "furthermore", "in addition", "however", "nevertheless", "in conclusion"]
    marker_count = sum(1 for marker in coherence_markers if marker in response.lower())
    marker_score = min(marker_count / 5, 1.0)
    
    # 3. Check for sentence length variation (more natural text has varied sentence lengths)
    sentences = [s.strip() for s in re.split(r'[.!?]', response) if s.strip()]
    if len(sentences) < 2:
        variation_score = 0.0
    else:
        sentence_lengths = [len(s) for s in sentences]
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variation = sum(abs(length - avg_length) for length in sentence_lengths) / len(sentence_lengths)
        variation_score = min(variation / 20, 1.0)
    
    # Combine scores with weights
    return (
        paragraph_score * 0.4 +
        marker_score * 0.4 +
        variation_score * 0.2
    )


def evaluate_relevance(query: str, response: str) -> float:
    """
    Evaluate the relevance of the response to the query.

    Args:
        query: The query
        response: The response

    Returns:
        float: Relevance score between 0 and 1
    """
    # 1. Check for query terms in the response
    query_terms = set(query.lower().split())
    query_term_count = sum(1 for term in query_terms if term in response.lower())
    term_score = query_term_count / len(query_terms) if query_terms else 0.0
    
    # 2. Check for semantic similarity (simple approach)
    similarity_score = calculate_text_similarity(query, response)
    
    # 3. Check if response directly addresses the query format
    is_what_query = "what" in query.lower()
    is_how_query = "how" in query.lower()
    is_why_query = "why" in query.lower()
    is_when_query = "when" in query.lower()
    is_where_query = "where" in query.lower()
    
    format_score = 0.0
    if is_what_query and "is" in response.lower():
        format_score = 1.0
    elif is_how_query and any(term in response.lower() for term in ["steps", "process", "way", "method"]):
        format_score = 1.0
    elif is_why_query and any(term in response.lower() for term in ["because", "reason", "due to"]):
        format_score = 1.0
    elif is_when_query and any(term in response.lower() for term in ["time", "date", "period", "century", "year"]):
        format_score = 1.0
    elif is_where_query and any(term in response.lower() for term in ["location", "place", "country", "city"]):
        format_score = 1.0
    
    # Combine scores with weights
    return (
        term_score * 0.4 +
        similarity_score * 0.4 +
        format_score * 0.2
    )


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


async def evaluate_reasoning_quality_batch(
    rag: LightRAG,
    queries: List[str],
    mode: str = "auto",
    use_cot: bool = True
) -> float:
    """
    Evaluate the quality of reasoning in responses for a batch of queries.

    Args:
        rag: The LightRAG instance to evaluate
        queries: List of queries to evaluate
        mode: Retrieval mode to use
        use_cot: Whether to use Chain-of-Thought reasoning

    Returns:
        float: Average reasoning quality score
    """
    reasoning_scores = []
    
    for query in queries:
        # Set up query parameters
        param = QueryParam(
            mode=mode,
            use_cot=use_cot
        )
        
        # Get response
        response = await rag.aquery(query, param=param)
        
        # Evaluate reasoning quality
        reasoning_score = evaluate_reasoning_quality(query, response)
        reasoning_scores.append(reasoning_score)
    
    # Calculate average reasoning quality score
    return sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0


async def evaluate_citation_quality_batch(
    rag: LightRAG,
    queries: List[str],
    mode: str = "auto",
    use_enhanced_citations: bool = True
) -> float:
    """
    Evaluate the quality of citations in responses for a batch of queries.

    Args:
        rag: The LightRAG instance to evaluate
        queries: List of queries to evaluate
        mode: Retrieval mode to use
        use_enhanced_citations: Whether to use enhanced citations

    Returns:
        float: Average citation quality score
    """
    citation_scores = []
    
    for query in queries:
        # Set up query parameters
        param = QueryParam(
            mode=mode,
            use_enhanced_citations=use_enhanced_citations
        )
        
        # Get response
        response = await rag.aquery(query, param=param)
        
        # Evaluate citation quality
        citation_score = evaluate_citation_quality(response)
        citation_scores.append(citation_score)
    
    # Calculate average citation quality score
    return sum(citation_scores) / len(citation_scores) if citation_scores else 0.0


async def evaluate_factual_accuracy_batch(
    rag: LightRAG,
    queries: List[str],
    ground_truth: Optional[Dict[str, str]] = None,
    mode: str = "auto"
) -> float:
    """
    Evaluate the factual accuracy of responses for a batch of queries.

    Args:
        rag: The LightRAG instance to evaluate
        queries: List of queries to evaluate
        ground_truth: Optional dictionary mapping queries to ground truth answers
        mode: Retrieval mode to use

    Returns:
        float: Average factual accuracy score
    """
    factual_scores = []
    
    for query in queries:
        # Set up query parameters
        param = QueryParam(mode=mode)
        
        # Get response
        response = await rag.aquery(query, param=param)
        
        # Evaluate factual accuracy
        factual_score = evaluate_factual_accuracy(
            query, response, ground_truth.get(query) if ground_truth else None
        )
        factual_scores.append(factual_score)
    
    # Calculate average factual accuracy score
    return sum(factual_scores) / len(factual_scores) if factual_scores else 0.0
