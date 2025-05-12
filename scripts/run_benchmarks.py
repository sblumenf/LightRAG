#!/usr/bin/env python
"""
Script to run benchmarks for LightRAG.

This script runs a series of benchmarks on LightRAG components and the overall system,
measuring performance metrics like execution time, memory usage, and throughput.
"""

import os
import sys
import asyncio
import argparse
import json
import datetime
from typing import Dict, List, Any, Tuple

# Add the parent directory to the path so we can import lightrag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.text_chunker import TextChunker
from lightrag.kg.entity_resolver import EntityResolver
from lightrag.llm.llm_generator import LLMGenerator

from lightrag.benchmarking import (
    BenchmarkResult,
    run_benchmark,
    run_benchmarks,
    print_benchmark_results,
    save_benchmark_results,
    load_benchmark_results,
    plot_benchmark_results,
    benchmark_chunking,
    benchmark_entity_resolution,
    benchmark_query_processing,
    benchmark_retrieval,
    benchmark_generation,
    benchmark_document_ingestion,
    benchmark_query_response,
    benchmark_full_pipeline,
    benchmark_document_sizes
)


# Set up logging
setup_logger("lightrag", level="INFO")


async def run_component_benchmarks(args):
    """Run benchmarks for individual components."""
    print("\n=== Running Component Benchmarks ===\n")
    
    results = []
    
    # Create a LightRAG instance for benchmarking
    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=gpt_4o_mini_complete if args.use_openai else dummy_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536 if args.use_openai else 768,
            max_token_size=8192,
            func=openai_embed if args.use_openai else dummy_embedding_func,
        ),
        # Use file-based storage implementations for benchmarking
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
        # Enable enhanced features
        use_enhanced_chunking=True,
        use_schema_classification=True,
        use_entity_resolution=True,
        use_intelligent_retrieval=True,
        use_cot=True,
        use_enhanced_citations=True,
    )

    # Initialize storages
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    try:
        # Sample text for chunking benchmark
        sample_text = """
        # LightRAG: A Lightweight Knowledge Graph RAG System

        LightRAG is a lightweight Knowledge Graph Retrieval-Augmented Generation system.
        It supports multiple LLM backends and provides efficient document processing.

        ## Features

        LightRAG includes the following key features:

        1. **Schema-driven Knowledge Graph Construction**: LightRAG uses a schema to guide the construction of the knowledge graph, ensuring that entities and relationships conform to a predefined structure.

        2. **Advanced Entity Resolution**: LightRAG includes sophisticated entity resolution capabilities that can identify and merge duplicate entities based on name similarity, embedding similarity, and context.

        3. **Intelligent Retrieval**: LightRAG analyzes queries to determine the optimal retrieval strategy, including entity type filtering and result reranking.

        4. **Chain-of-Thought Reasoning**: LightRAG implements Chain-of-Thought (CoT) reasoning to improve response quality by encouraging step-by-step reasoning.

        5. **Enhanced Citation Handling**: LightRAG provides enhanced citation handling that traces citations to specific context elements and formats them as numbered references.

        6. **Diagram and Formula Integration**: LightRAG can extract and integrate diagrams and formulas from documents, making them available for retrieval and generation.
        """
        
        # Benchmark chunking with different strategies
        if args.run_chunking:
            for strategy in ['token', 'paragraph', 'semantic']:
                result = await benchmark_chunking(
                    sample_text,
                    chunking_strategy=strategy,
                    chunk_size=1000,
                    chunk_overlap=200,
                    iterations=args.iterations,
                    warmup_iterations=1
                )
                results.append(result)
        
        # Insert some documents for other benchmarks
        if args.run_entity_resolution or args.run_retrieval or args.run_generation:
            doc1 = "LightRAG is a lightweight Knowledge Graph RAG system."
            doc2 = "LightRAG supports multiple LLM backends and provides efficient document processing."
            doc3 = "The system uses a schema-driven approach to build knowledge graphs from text documents."
            
            await rag.ainsert(doc1)
            await rag.ainsert(doc2)
            await rag.ainsert(doc3)
            
            # Wait for processing to complete
            await asyncio.sleep(2)
        
        # Benchmark entity resolution
        if args.run_entity_resolution:
            entity_resolver = EntityResolver(rag.chunk_entity_relation_graph, rag.embedding_func)
            result = await benchmark_entity_resolution(
                entity_resolver,
                num_entities=10,
                similarity_threshold=0.7,
                iterations=args.iterations,
                warmup_iterations=1
            )
            results.append(result)
        
        # Benchmark query processing
        if args.run_query_processing:
            queries = [
                "What is LightRAG?",
                "What features does LightRAG have?",
                "How does LightRAG handle entity resolution?",
                "Explain the architecture of LightRAG."
            ]
            
            result = await benchmark_query_processing(
                rag.llm_model_func,
                queries,
                iterations=args.iterations,
                warmup_iterations=1
            )
            results.append(result)
        
        # Benchmark retrieval
        if args.run_retrieval:
            queries = [
                "What is LightRAG?",
                "What features does LightRAG have?",
                "How does LightRAG handle entity resolution?",
                "Explain the architecture of LightRAG."
            ]
            
            for mode in ['naive', 'local', 'global', 'hybrid']:
                result = await benchmark_retrieval(
                    rag,
                    queries,
                    mode=mode,
                    top_k=5,
                    iterations=args.iterations,
                    warmup_iterations=1
                )
                results.append(result)
        
        # Benchmark generation
        if args.run_generation:
            queries = [
                "What is LightRAG?",
                "What features does LightRAG have?"
            ]
            
            contexts = [
                [
                    {"id": "1", "content": "LightRAG is a lightweight Knowledge Graph RAG system."},
                    {"id": "2", "content": "It supports multiple LLM backends and provides efficient document processing."}
                ],
                [
                    {"id": "3", "content": "LightRAG includes schema-driven knowledge graph construction."},
                    {"id": "4", "content": "It also provides advanced entity resolution capabilities."}
                ]
            ]
            
            llm_generator = LLMGenerator(rag.llm_model_func)
            
            result = await benchmark_generation(
                llm_generator,
                queries,
                contexts,
                use_cot=True,
                use_enhanced_citations=True,
                iterations=args.iterations,
                warmup_iterations=1
            )
            results.append(result)
        
        # Print and save results
        print_benchmark_results(results)
        
        if args.output:
            save_benchmark_results(results, args.output)
            print(f"\nResults saved to {args.output}")
        
        if args.plot:
            plot_path = f"benchmark_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_benchmark_results(results, save_path=plot_path)
            print(f"\nPlot saved to {plot_path}")
    
    finally:
        # Clean up
        await rag.finalize_storages()
    
    return results


async def run_e2e_benchmarks(args):
    """Run end-to-end benchmarks."""
    print("\n=== Running End-to-End Benchmarks ===\n")
    
    results = []
    
    # Create a LightRAG instance for benchmarking
    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=gpt_4o_mini_complete if args.use_openai else dummy_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536 if args.use_openai else 768,
            max_token_size=8192,
            func=openai_embed if args.use_openai else dummy_embedding_func,
        ),
        # Use file-based storage implementations for benchmarking
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        doc_status_storage="JsonDocStatusStorage",
        # Enable enhanced features
        use_enhanced_chunking=True,
        use_schema_classification=True,
        use_entity_resolution=True,
        use_intelligent_retrieval=True,
        use_cot=True,
        use_enhanced_citations=True,
    )

    # Initialize storages
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    try:
        # Sample documents for benchmarking
        documents = [
            "LightRAG is a lightweight Knowledge Graph RAG system.",
            "LightRAG supports multiple LLM backends and provides efficient document processing.",
            "The system uses a schema-driven approach to build knowledge graphs from text documents."
        ]
        
        # Sample queries for benchmarking
        queries = [
            "What is LightRAG?",
            "What features does LightRAG have?"
        ]
        
        # Sample query parameters for benchmarking
        query_params = [
            QueryParam(mode="naive"),
            QueryParam(mode="hybrid", use_cot=True, use_enhanced_citations=True)
        ]
        
        # Benchmark document ingestion
        if args.run_ingestion:
            result = await benchmark_document_ingestion(
                rag,
                documents,
                wait_for_processing=True,
                iterations=args.iterations,
                warmup_iterations=1
            )
            results.append(result)
        
        # Benchmark query response
        if args.run_query:
            # Insert documents first if we didn't run ingestion benchmark
            if not args.run_ingestion:
                for doc in documents:
                    await rag.ainsert(doc)
                await asyncio.sleep(2)  # Wait for processing
            
            result = await benchmark_query_response(
                rag,
                queries,
                query_params,
                iterations=args.iterations,
                warmup_iterations=1
            )
            results.append(result)
        
        # Benchmark full pipeline
        if args.run_full:
            # Clear existing documents
            await rag.finalize_storages()
            await rag.initialize_storages()
            await initialize_pipeline_status()
            
            pipeline_results = await benchmark_full_pipeline(
                rag,
                documents,
                queries,
                query_params,
                iterations=args.iterations,
                warmup_iterations=1
            )
            results.extend(pipeline_results)
        
        # Benchmark document sizes
        if args.run_sizes:
            # Clear existing documents
            await rag.finalize_storages()
            await rag.initialize_storages()
            await initialize_pipeline_status()
            
            base_document = "LightRAG is a lightweight Knowledge Graph RAG system. " * 10
            sizes = [1, 10, 50, 100]  # KB
            
            size_results = await benchmark_document_sizes(
                rag,
                base_document,
                sizes,
                iterations=args.iterations,
                warmup_iterations=1
            )
            results.extend(size_results)
        
        # Print and save results
        print_benchmark_results(results)
        
        if args.output:
            save_benchmark_results(results, args.output)
            print(f"\nResults saved to {args.output}")
        
        if args.plot:
            plot_path = f"benchmark_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plot_benchmark_results(results, save_path=plot_path)
            print(f"\nPlot saved to {plot_path}")
    
    finally:
        # Clean up
        await rag.finalize_storages()
    
    return results


# Dummy functions for testing without OpenAI
async def dummy_llm_func(prompt: str, **kwargs) -> str:
    """Dummy LLM function that returns a simple response."""
    return f"Response to: {prompt[:30]}..."

async def dummy_embedding_func(texts: list[str]) -> list[list[float]]:
    """Dummy embedding function that returns fixed-size vectors."""
    # Return a simple deterministic embedding (all zeros with the first element being the hash of the text)
    return [[hash(text) % 100] + [0.0] * 767 for text in texts]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmarks for LightRAG")
    
    # General options
    parser.add_argument("--working-dir", default="./benchmark_cache", help="Working directory for LightRAG")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations for each benchmark")
    parser.add_argument("--output", help="Output file for benchmark results (JSON)")
    parser.add_argument("--plot", action="store_true", help="Generate plots of benchmark results")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for LLM and embeddings")
    
    # Component benchmark options
    parser.add_argument("--run-chunking", action="store_true", help="Run chunking benchmarks")
    parser.add_argument("--run-entity-resolution", action="store_true", help="Run entity resolution benchmarks")
    parser.add_argument("--run-query-processing", action="store_true", help="Run query processing benchmarks")
    parser.add_argument("--run-retrieval", action="store_true", help="Run retrieval benchmarks")
    parser.add_argument("--run-generation", action="store_true", help="Run generation benchmarks")
    
    # E2E benchmark options
    parser.add_argument("--run-ingestion", action="store_true", help="Run document ingestion benchmarks")
    parser.add_argument("--run-query", action="store_true", help="Run query response benchmarks")
    parser.add_argument("--run-full", action="store_true", help="Run full pipeline benchmarks")
    parser.add_argument("--run-sizes", action="store_true", help="Run document size benchmarks")
    
    # Run all benchmarks if none specified
    args = parser.parse_args()
    if not any([
        args.run_chunking, args.run_entity_resolution, args.run_query_processing,
        args.run_retrieval, args.run_generation, args.run_ingestion, args.run_query,
        args.run_full, args.run_sizes
    ]):
        args.run_chunking = True
        args.run_entity_resolution = True
        args.run_query_processing = True
        args.run_retrieval = True
        args.run_generation = True
        args.run_ingestion = True
        args.run_query = True
        args.run_full = True
        args.run_sizes = True
    
    return args


async def main():
    """Main function."""
    args = parse_args()
    
    # Create working directory if it doesn't exist
    os.makedirs(args.working_dir, exist_ok=True)
    
    # Run component benchmarks
    if any([args.run_chunking, args.run_entity_resolution, args.run_query_processing, args.run_retrieval, args.run_generation]):
        await run_component_benchmarks(args)
    
    # Run E2E benchmarks
    if any([args.run_ingestion, args.run_query, args.run_full, args.run_sizes]):
        await run_e2e_benchmarks(args)


if __name__ == "__main__":
    asyncio.run(main())
