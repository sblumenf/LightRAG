"""
End-to-end benchmarks for LightRAG.

This module provides benchmarking functions for end-to-end LightRAG workflows,
including document ingestion, query response, and the full pipeline.
"""

import time
import asyncio
from typing import Dict, List, Any, Callable, Union, Optional, Tuple

from lightrag import LightRAG, QueryParam

from .benchmark_utils import BenchmarkResult, run_benchmark, run_async_benchmark


async def benchmark_document_ingestion(
    rag: LightRAG,
    documents: List[str],
    wait_for_processing: bool = True,
    iterations: int = 1,
    warmup_iterations: int = 0
) -> BenchmarkResult:
    """
    Benchmark document ingestion.

    Args:
        rag: The LightRAG instance to benchmark
        documents: List of documents to ingest
        wait_for_processing: Whether to wait for processing to complete
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run

    Returns:
        BenchmarkResult: The benchmark results
    """
    # Define the function to benchmark
    async def ingest_documents():
        doc_ids = []
        for doc in documents:
            doc_id = await rag.ainsert(doc)
            doc_ids.append(doc_id)
        
        if wait_for_processing:
            # Wait for all documents to be processed
            max_wait_time = 60  # seconds
            wait_interval = 1  # seconds
            total_wait_time = 0
            
            while total_wait_time < max_wait_time:
                all_processed = True
                for doc_id in doc_ids:
                    doc_status = await rag.doc_status_storage.get_by_id(doc_id)
                    if not doc_status or doc_status.get("status") != "PROCESSED":
                        all_processed = False
                        break
                
                if all_processed:
                    break
                
                await asyncio.sleep(wait_interval)
                total_wait_time += wait_interval
        
        return doc_ids

    # Run the benchmark
    result = await run_async_benchmark(
        ingest_documents,
        name="document_ingestion",
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        measure_memory=True,
        calculate_throughput=True,
        throughput_items=len(documents),
        parameters={
            "num_documents": len(documents),
            "avg_document_length": sum(len(d) for d in documents) / len(documents),
            "wait_for_processing": wait_for_processing
        }
    )

    return result


async def benchmark_query_response(
    rag: LightRAG,
    queries: List[str],
    query_params: List[QueryParam] = None,
    iterations: int = 1,
    warmup_iterations: int = 0
) -> BenchmarkResult:
    """
    Benchmark query response.

    Args:
        rag: The LightRAG instance to benchmark
        queries: List of queries to run
        query_params: List of query parameters for each query (or None to use defaults)
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run

    Returns:
        BenchmarkResult: The benchmark results
    """
    # Use default query parameters if none provided
    if query_params is None:
        query_params = [QueryParam() for _ in queries]
    
    # Ensure we have parameters for each query
    if len(query_params) < len(queries):
        query_params.extend([QueryParam() for _ in range(len(queries) - len(query_params))])

    # Define the function to benchmark
    async def run_queries():
        results = []
        for i, query in enumerate(queries):
            param = query_params[i]
            result = await rag.aquery(query, param=param)
            results.append(result)
        return results

    # Extract mode information for metadata
    modes = [param.mode for param in query_params]
    mode_counts = {mode: modes.count(mode) for mode in set(modes)}

    # Run the benchmark
    result = await run_async_benchmark(
        run_queries,
        name="query_response",
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        measure_memory=True,
        calculate_throughput=True,
        throughput_items=len(queries),
        parameters={
            "num_queries": len(queries),
            "avg_query_length": sum(len(q) for q in queries) / len(queries),
            "modes": mode_counts
        }
    )

    return result


async def benchmark_full_pipeline(
    rag: LightRAG,
    documents: List[str],
    queries: List[str],
    query_params: List[QueryParam] = None,
    iterations: int = 1,
    warmup_iterations: int = 0
) -> List[BenchmarkResult]:
    """
    Benchmark the full pipeline from document ingestion to query response.

    Args:
        rag: The LightRAG instance to benchmark
        documents: List of documents to ingest
        queries: List of queries to run
        query_params: List of query parameters for each query (or None to use defaults)
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run

    Returns:
        List[BenchmarkResult]: The benchmark results for each phase
    """
    # Use default query parameters if none provided
    if query_params is None:
        query_params = [QueryParam() for _ in queries]
    
    # Ensure we have parameters for each query
    if len(query_params) < len(queries):
        query_params.extend([QueryParam() for _ in range(len(queries) - len(query_params))])

    # Define the function to benchmark
    async def run_full_pipeline():
        # Phase 1: Document ingestion
        doc_ids = []
        for doc in documents:
            doc_id = await rag.ainsert(doc)
            doc_ids.append(doc_id)
        
        # Wait for all documents to be processed
        max_wait_time = 60  # seconds
        wait_interval = 1  # seconds
        total_wait_time = 0
        
        while total_wait_time < max_wait_time:
            all_processed = True
            for doc_id in doc_ids:
                doc_status = await rag.doc_status_storage.get_by_id(doc_id)
                if not doc_status or doc_status.get("status") != "PROCESSED":
                    all_processed = False
                    break
            
            if all_processed:
                break
            
            await asyncio.sleep(wait_interval)
            total_wait_time += wait_interval
        
        # Phase 2: Query response
        results = []
        for i, query in enumerate(queries):
            param = query_params[i]
            result = await rag.aquery(query, param=param)
            results.append(result)
        
        return {
            "doc_ids": doc_ids,
            "query_results": results
        }

    # Run the benchmark
    full_result = await run_async_benchmark(
        run_full_pipeline,
        name="full_pipeline",
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        measure_memory=True,
        parameters={
            "num_documents": len(documents),
            "avg_document_length": sum(len(d) for d in documents) / len(documents),
            "num_queries": len(queries),
            "avg_query_length": sum(len(q) for q in queries) / len(queries)
        }
    )

    # Also benchmark individual components
    ingestion_result = await benchmark_document_ingestion(
        rag,
        documents,
        wait_for_processing=True,
        iterations=iterations,
        warmup_iterations=warmup_iterations
    )

    query_result = await benchmark_query_response(
        rag,
        queries,
        query_params,
        iterations=iterations,
        warmup_iterations=warmup_iterations
    )

    return [full_result, ingestion_result, query_result]


async def benchmark_document_sizes(
    rag: LightRAG,
    base_document: str,
    sizes: List[int],
    iterations: int = 1,
    warmup_iterations: int = 0
) -> List[BenchmarkResult]:
    """
    Benchmark document ingestion with different document sizes.

    Args:
        rag: The LightRAG instance to benchmark
        base_document: The base document to replicate
        sizes: List of document sizes to test (in KB)
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run

    Returns:
        List[BenchmarkResult]: The benchmark results for each document size
    """
    results = []
    
    for size in sizes:
        # Create a document of the specified size
        target_size = size * 1024  # Convert KB to bytes
        repetitions = max(1, target_size // len(base_document))
        document = base_document * repetitions
        
        # Truncate or pad to get exact size
        if len(document) > target_size:
            document = document[:target_size]
        elif len(document) < target_size:
            document += " " * (target_size - len(document))
        
        # Benchmark ingestion of this document
        result = await benchmark_document_ingestion(
            rag,
            [document],
            wait_for_processing=True,
            iterations=iterations,
            warmup_iterations=warmup_iterations
        )
        
        # Update the name to include the size
        result.name = f"document_ingestion_{size}KB"
        result.parameters["document_size_kb"] = size
        
        results.append(result)
    
    return results
