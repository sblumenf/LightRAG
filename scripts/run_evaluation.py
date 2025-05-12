#!/usr/bin/env python
"""
Script to run qualitative evaluation for LightRAG.

This script evaluates the quality of LightRAG's knowledge graph, retrieval, and
response generation, providing metrics and visualizations.
"""

import os
import sys
import asyncio
import argparse
import json
import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import lightrag
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, setup_logger
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

from lightrag.evaluation import (
    evaluate_kg_quality,
    evaluate_schema_conformance,
    evaluate_entity_resolution,
    evaluate_relationship_quality,
    KGQualityMetrics,
    evaluate_retrieval_relevance,
    evaluate_retrieval_diversity,
    evaluate_retrieval_coverage,
    RetrievalRelevanceMetrics,
    evaluate_response_quality,
    evaluate_reasoning_quality,
    evaluate_citation_quality,
    evaluate_factual_accuracy,
    ResponseQualityMetrics
)


# Set up logging
setup_logger("lightrag", level="INFO")


async def run_kg_evaluation(args, rag: LightRAG):
    """Run knowledge graph quality evaluation."""
    print("\n=== Running Knowledge Graph Quality Evaluation ===\n")
    
    # Evaluate knowledge graph quality
    kg_metrics = await evaluate_kg_quality(rag, args.schema_path)
    
    # Print metrics
    print("\nKnowledge Graph Quality Metrics:")
    print(f"Schema Conformance Rate: {kg_metrics.schema_conformance_rate:.2f}")
    print(f"Entity Property Completeness: {kg_metrics.entity_property_completeness:.2f}")
    print(f"Relationship Property Completeness: {kg_metrics.relationship_property_completeness:.2f}")
    print(f"Entity Resolution F1 Score: {kg_metrics.entity_resolution_f1:.2f}")
    print(f"Relationship Quality Score: {kg_metrics.relationship_quality_score:.2f}")
    print(f"Total Entities: {kg_metrics.total_entities}")
    print(f"Total Relationships: {kg_metrics.total_relationships}")
    print(f"Schema Violations: {kg_metrics.schema_violations}")
    print(f"Orphaned Entities: {kg_metrics.orphaned_entities}")
    
    # Print entity type coverage
    print("\nEntity Type Coverage:")
    for entity_type, coverage in kg_metrics.entity_types_coverage.items():
        print(f"  {entity_type}: {coverage:.2f}")
    
    # Print relationship type coverage
    print("\nRelationship Type Coverage:")
    for rel_type, coverage in kg_metrics.relationship_types_coverage.items():
        print(f"  {rel_type}: {coverage:.2f}")
    
    # Save metrics to file
    if args.output:
        output_file = f"{args.output}_kg_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(kg_metrics.to_dict(), f, indent=2)
        print(f"\nKG metrics saved to {output_file}")
    
    # Create visualizations
    if args.plot:
        # Create bar chart of entity type coverage
        plt.figure(figsize=(10, 6))
        entity_types = list(kg_metrics.entity_types_coverage.keys())
        coverages = list(kg_metrics.entity_types_coverage.values())
        plt.bar(entity_types, coverages)
        plt.xlabel('Entity Type')
        plt.ylabel('Coverage')
        plt.title('Entity Type Coverage')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = f"kg_entity_coverage_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file)
        print(f"\nEntity type coverage plot saved to {plot_file}")
        
        # Create summary metrics chart
        plt.figure(figsize=(10, 6))
        metrics = [
            kg_metrics.schema_conformance_rate,
            kg_metrics.entity_property_completeness,
            kg_metrics.relationship_property_completeness,
            kg_metrics.entity_resolution_f1,
            kg_metrics.relationship_quality_score
        ]
        labels = [
            'Schema Conformance',
            'Entity Property Completeness',
            'Relationship Property Completeness',
            'Entity Resolution F1',
            'Relationship Quality'
        ]
        plt.bar(labels, metrics)
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Knowledge Graph Quality Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = f"kg_quality_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file)
        print(f"KG quality metrics plot saved to {plot_file}")
    
    return kg_metrics


async def run_retrieval_evaluation(args, rag: LightRAG):
    """Run retrieval relevance evaluation."""
    print("\n=== Running Retrieval Relevance Evaluation ===\n")
    
    # Load test queries
    queries = load_test_queries(args.queries_file)
    
    # Load relevance judgments if available
    relevance_judgments = None
    if args.relevance_file:
        with open(args.relevance_file, 'r') as f:
            relevance_judgments = json.load(f)
    
    # Evaluate retrieval relevance for different modes
    modes = ["naive", "local", "global", "hybrid", "auto"]
    retrieval_metrics = await evaluate_retrieval_relevance(
        rag,
        queries,
        relevance_judgments,
        modes=modes,
        top_k=args.top_k,
        use_intelligent_retrieval=True
    )
    
    # Print metrics for each mode
    for mode, metrics in retrieval_metrics.items():
        print(f"\nRetrieval Metrics for {mode.upper()} mode:")
        print(f"Relevance Score: {metrics.relevance_score:.2f}")
        print(f"Diversity Score: {metrics.diversity_score:.2f}")
        print(f"Coverage Score: {metrics.coverage_score:.2f}")
        print(f"Average Result Count: {metrics.avg_result_count:.2f}")
        
        print("\nPrecision at k:")
        for k, precision in metrics.precision_at_k.items():
            print(f"  P@{k}: {precision:.2f}")
        
        print("\nRecall at k:")
        for k, recall in metrics.recall_at_k.items():
            print(f"  R@{k}: {recall:.2f}")
        
        print("\nNDCG at k:")
        for k, ndcg in metrics.ndcg_at_k.items():
            print(f"  NDCG@{k}: {ndcg:.2f}")
        
        print(f"\nMean Reciprocal Rank: {metrics.mrr:.2f}")
    
    # Save metrics to file
    if args.output:
        output_file = f"{args.output}_retrieval_metrics.json"
        with open(output_file, 'w') as f:
            json.dump({mode: metrics.to_dict() for mode, metrics in retrieval_metrics.items()}, f, indent=2)
        print(f"\nRetrieval metrics saved to {output_file}")
    
    # Create visualizations
    if args.plot:
        # Create comparison chart of relevance scores
        plt.figure(figsize=(10, 6))
        mode_labels = list(retrieval_metrics.keys())
        relevance_scores = [metrics.relevance_score for metrics in retrieval_metrics.values()]
        diversity_scores = [metrics.diversity_score for metrics in retrieval_metrics.values()]
        coverage_scores = [metrics.coverage_score for metrics in retrieval_metrics.values()]
        
        x = np.arange(len(mode_labels))
        width = 0.25
        
        plt.bar(x - width, relevance_scores, width, label='Relevance')
        plt.bar(x, diversity_scores, width, label='Diversity')
        plt.bar(x + width, coverage_scores, width, label='Coverage')
        
        plt.xlabel('Retrieval Mode')
        plt.ylabel('Score')
        plt.title('Retrieval Quality Metrics by Mode')
        plt.xticks(x, mode_labels)
        plt.legend()
        plt.tight_layout()
        
        plot_file = f"retrieval_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file)
        print(f"\nRetrieval metrics plot saved to {plot_file}")
        
        # Create precision-recall curve
        plt.figure(figsize=(10, 6))
        for mode, metrics in retrieval_metrics.items():
            precision_values = list(metrics.precision_at_k.values())
            recall_values = list(metrics.recall_at_k.values())
            plt.plot(recall_values, precision_values, marker='o', label=mode)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve by Mode')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plot_file = f"precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file)
        print(f"Precision-recall plot saved to {plot_file}")
    
    return retrieval_metrics


async def run_response_evaluation(args, rag: LightRAG):
    """Run response quality evaluation."""
    print("\n=== Running Response Quality Evaluation ===\n")
    
    # Load test queries
    queries = load_test_queries(args.queries_file)
    
    # Load ground truth if available
    ground_truth = None
    if args.ground_truth_file:
        with open(args.ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
    
    # Evaluate response quality
    response_metrics = await evaluate_response_quality(
        rag,
        queries,
        ground_truth,
        mode=args.mode,
        use_cot=args.use_cot,
        use_enhanced_citations=args.use_citations,
        use_intelligent_retrieval=args.use_intelligent_retrieval
    )
    
    # Print metrics
    print("\nResponse Quality Metrics:")
    print(f"Overall Quality Score: {response_metrics.overall_quality_score:.2f}")
    print(f"Reasoning Quality Score: {response_metrics.reasoning_quality_score:.2f}")
    print(f"Citation Quality Score: {response_metrics.citation_quality_score:.2f}")
    print(f"Factual Accuracy Score: {response_metrics.factual_accuracy_score:.2f}")
    print(f"Completeness Score: {response_metrics.completeness_score:.2f}")
    print(f"Coherence Score: {response_metrics.coherence_score:.2f}")
    print(f"Relevance Score: {response_metrics.relevance_score:.2f}")
    print(f"Percentage with Reasoning: {response_metrics.has_reasoning:.2f}")
    print(f"Percentage with Citations: {response_metrics.has_citations:.2f}")
    print(f"Average Citation Count: {response_metrics.avg_citation_count:.2f}")
    print(f"Query Count: {response_metrics.query_count}")
    
    # Save metrics to file
    if args.output:
        output_file = f"{args.output}_response_metrics.json"
        with open(output_file, 'w') as f:
            json.dump(response_metrics.to_dict(), f, indent=2)
        print(f"\nResponse metrics saved to {output_file}")
    
    # Create visualizations
    if args.plot:
        # Create bar chart of response quality metrics
        plt.figure(figsize=(10, 6))
        metrics = [
            response_metrics.reasoning_quality_score,
            response_metrics.citation_quality_score,
            response_metrics.factual_accuracy_score,
            response_metrics.completeness_score,
            response_metrics.coherence_score,
            response_metrics.relevance_score
        ]
        labels = [
            'Reasoning Quality',
            'Citation Quality',
            'Factual Accuracy',
            'Completeness',
            'Coherence',
            'Relevance'
        ]
        plt.bar(labels, metrics)
        plt.xlabel('Metric')
        plt.ylabel('Score')
        plt.title('Response Quality Metrics')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plot_file = f"response_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file)
        print(f"\nResponse metrics plot saved to {plot_file}")
    
    return response_metrics


def load_test_queries(file_path: str) -> List[str]:
    """Load test queries from a file."""
    if not file_path or not os.path.exists(file_path):
        # Use default test queries
        return [
            "What is LightRAG?",
            "How does LightRAG handle entity resolution?",
            "What are the main features of LightRAG?",
            "Explain the architecture of LightRAG.",
            "How does LightRAG's knowledge graph work?"
        ]
    
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


async def main():
    """Main function."""
    args = parse_args()
    
    # Create LightRAG instance
    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=gpt_4o_mini_complete if args.use_openai else dummy_llm_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536 if args.use_openai else 768,
            max_token_size=8192,
            func=openai_embed if args.use_openai else dummy_embedding_func,
        ),
        # Use specified storage implementations
        kv_storage=args.kv_storage,
        vector_storage=args.vector_storage,
        graph_storage=args.graph_storage,
        doc_status_storage=args.doc_status_storage,
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
        # Run evaluations
        if args.evaluate_kg:
            await run_kg_evaluation(args, rag)
        
        if args.evaluate_retrieval:
            await run_retrieval_evaluation(args, rag)
        
        if args.evaluate_response:
            await run_response_evaluation(args, rag)
    
    finally:
        # Clean up
        await rag.finalize_storages()


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
    parser = argparse.ArgumentParser(description="Run qualitative evaluation for LightRAG")
    
    # General options
    parser.add_argument("--working-dir", default="./evaluation_cache", help="Working directory for LightRAG")
    parser.add_argument("--output", help="Base name for output files")
    parser.add_argument("--plot", action="store_true", help="Generate plots of evaluation results")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI for LLM and embeddings")
    
    # Storage options
    parser.add_argument("--kv-storage", default="JsonKVStorage", help="KV storage implementation")
    parser.add_argument("--vector-storage", default="NanoVectorDBStorage", help="Vector storage implementation")
    parser.add_argument("--graph-storage", default="NetworkXStorage", help="Graph storage implementation")
    parser.add_argument("--doc-status-storage", default="JsonDocStatusStorage", help="Document status storage implementation")
    
    # Evaluation options
    parser.add_argument("--evaluate-kg", action="store_true", help="Evaluate knowledge graph quality")
    parser.add_argument("--evaluate-retrieval", action="store_true", help="Evaluate retrieval relevance")
    parser.add_argument("--evaluate-response", action="store_true", help="Evaluate response quality")
    
    # KG evaluation options
    parser.add_argument("--schema-path", help="Path to schema file")
    
    # Retrieval evaluation options
    parser.add_argument("--queries-file", help="Path to file containing test queries")
    parser.add_argument("--relevance-file", help="Path to file containing relevance judgments")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to retrieve")
    
    # Response evaluation options
    parser.add_argument("--ground-truth-file", help="Path to file containing ground truth answers")
    parser.add_argument("--mode", default="auto", help="Retrieval mode to use")
    parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought reasoning")
    parser.add_argument("--use-citations", action="store_true", help="Use enhanced citations")
    parser.add_argument("--use-intelligent-retrieval", action="store_true", help="Use intelligent retrieval")
    
    # Run all evaluations if none specified
    args = parser.parse_args()
    if not any([args.evaluate_kg, args.evaluate_retrieval, args.evaluate_response]):
        args.evaluate_kg = True
        args.evaluate_retrieval = True
        args.evaluate_response = True
    
    return args


if __name__ == "__main__":
    asyncio.run(main())
