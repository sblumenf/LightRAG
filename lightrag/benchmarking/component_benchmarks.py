"""
Component benchmarks for LightRAG.

This module provides benchmarking functions for individual LightRAG components,
including chunking, entity resolution, query processing, retrieval, and generation.
"""

import time
import asyncio
from typing import Dict, List, Any, Callable, Union, Optional, Tuple

from lightrag import LightRAG, QueryParam
from lightrag.text_chunker import TextChunker
from lightrag.kg.entity_resolver import EntityResolver
from lightrag.query_processing.query_analyzer import process_query
from lightrag.query_processing.strategy_selector import select_retrieval_strategy
from lightrag.llm.llm_generator import LLMGenerator

from .benchmark_utils import BenchmarkResult, run_benchmark, run_async_benchmark


async def benchmark_chunking(
    text: str,
    chunking_strategy: str = 'token',
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    iterations: int = 5,
    warmup_iterations: int = 1
) -> BenchmarkResult:
    """
    Benchmark the text chunking component.

    Args:
        text: The text to chunk
        chunking_strategy: The chunking strategy to use
        chunk_size: The chunk size to use
        chunk_overlap: The chunk overlap to use
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run

    Returns:
        BenchmarkResult: The benchmark results
    """
    # Create a TextChunker instance
    chunker = TextChunker(
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Define the function to benchmark
    async def chunk_text():
        return chunker.chunk_text(text)

    # Run the benchmark
    result = await run_async_benchmark(
        chunk_text,
        name=f"chunking_{chunking_strategy}",
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        measure_memory=True,
        calculate_throughput=True,
        throughput_items=len(text) / 1000,  # Throughput in KB
        parameters={
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "text_length": len(text)
        }
    )

    return result


async def benchmark_entity_resolution(
    entity_resolver: EntityResolver,
    num_entities: int = 100,
    similarity_threshold: float = 0.7,
    iterations: int = 3,
    warmup_iterations: int = 1
) -> BenchmarkResult:
    """
    Benchmark the entity resolution component.

    Args:
        entity_resolver: The EntityResolver instance to benchmark
        num_entities: Number of entities to use in the benchmark
        similarity_threshold: Similarity threshold for duplicate detection
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run

    Returns:
        BenchmarkResult: The benchmark results
    """
    # Define the function to benchmark
    async def find_and_merge_duplicates():
        # Find duplicate candidates
        duplicate_candidates = await entity_resolver.find_duplicate_candidates(
            similarity_threshold=similarity_threshold
        )
        
        # Process each set of duplicates
        for primary_id, duplicate_ids in duplicate_candidates.items():
            if duplicate_ids:
                await entity_resolver.merge_entities(primary_id, duplicate_ids)

    # Run the benchmark
    result = await run_async_benchmark(
        find_and_merge_duplicates,
        name="entity_resolution",
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        measure_memory=True,
        calculate_throughput=True,
        throughput_items=num_entities,
        parameters={
            "num_entities": num_entities,
            "similarity_threshold": similarity_threshold
        }
    )

    return result


async def benchmark_query_processing(
    llm_func: Callable,
    queries: List[str],
    iterations: int = 3,
    warmup_iterations: int = 1
) -> BenchmarkResult:
    """
    Benchmark the query processing component.

    Args:
        llm_func: The LLM function to use
        queries: List of queries to process
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run

    Returns:
        BenchmarkResult: The benchmark results
    """
    # Define the function to benchmark
    async def process_queries():
        results = []
        for query in queries:
            result = await process_query(query, llm_func)
            results.append(result)
        return results

    # Run the benchmark
    result = await run_async_benchmark(
        process_queries,
        name="query_processing",
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        measure_memory=True,
        calculate_throughput=True,
        throughput_items=len(queries),
        parameters={
            "num_queries": len(queries),
            "avg_query_length": sum(len(q) for q in queries) / len(queries)
        }
    )

    return result


async def benchmark_retrieval(
    rag: LightRAG,
    queries: List[str],
    mode: str = "naive",
    top_k: int = 5,
    iterations: int = 3,
    warmup_iterations: int = 1
) -> BenchmarkResult:
    """
    Benchmark the retrieval component.

    Args:
        rag: The LightRAG instance to benchmark
        queries: List of queries to retrieve for
        mode: The retrieval mode to use
        top_k: Number of results to retrieve
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run

    Returns:
        BenchmarkResult: The benchmark results
    """
    # Define the function to benchmark
    async def retrieve_for_queries():
        results = []
        for query in queries:
            param = QueryParam(mode=mode, top_k=top_k, generate_response=False)
            result = await rag.aquery(query, param=param)
            results.append(result)
        return results

    # Run the benchmark
    result = await run_async_benchmark(
        retrieve_for_queries,
        name=f"retrieval_{mode}",
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        measure_memory=True,
        calculate_throughput=True,
        throughput_items=len(queries),
        parameters={
            "mode": mode,
            "top_k": top_k,
            "num_queries": len(queries)
        }
    )

    return result


async def benchmark_generation(
    llm_generator: LLMGenerator,
    queries: List[str],
    contexts: List[List[Dict[str, Any]]],
    use_cot: bool = True,
    use_enhanced_citations: bool = True,
    iterations: int = 3,
    warmup_iterations: int = 1
) -> BenchmarkResult:
    """
    Benchmark the generation component.

    Args:
        llm_generator: The LLMGenerator instance to benchmark
        queries: List of queries to generate responses for
        contexts: List of contexts for each query
        use_cot: Whether to use Chain-of-Thought reasoning
        use_enhanced_citations: Whether to use enhanced citations
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run

    Returns:
        BenchmarkResult: The benchmark results
    """
    # Define the function to benchmark
    async def generate_responses():
        results = []
        for i, query in enumerate(queries):
            context = contexts[i] if i < len(contexts) else []
            result = await llm_generator.generate_response(
                query,
                context,
                use_cot=use_cot,
                use_enhanced_citations=use_enhanced_citations
            )
            results.append(result)
        return results

    # Run the benchmark
    result = await run_async_benchmark(
        generate_responses,
        name="generation",
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        measure_memory=True,
        calculate_throughput=True,
        throughput_items=len(queries),
        parameters={
            "use_cot": use_cot,
            "use_enhanced_citations": use_enhanced_citations,
            "num_queries": len(queries),
            "avg_context_items": sum(len(c) for c in contexts) / len(contexts) if contexts else 0
        }
    )

    return result
