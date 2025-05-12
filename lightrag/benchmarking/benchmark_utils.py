"""
Benchmark utilities for LightRAG.

This module provides utilities for running benchmarks, collecting results,
and visualizing performance metrics.
"""

import time
import timeit
import asyncio
import json
import os
import psutil
import gc
from typing import Dict, List, Any, Callable, Union, Optional, Tuple
from dataclasses import dataclass, asdict, field
import matplotlib.pyplot as plt
import numpy as np
from functools import wraps


@dataclass
class BenchmarkResult:
    """Class for storing benchmark results."""
    name: str
    execution_time: float  # in seconds
    memory_usage: float = 0.0  # in MB
    throughput: float = 0.0  # items per second
    iterations: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the benchmark result to a dictionary."""
        return asdict(self)


def measure_memory_usage() -> float:
    """Measure the current memory usage of the process in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB


def run_benchmark(
    func: Callable,
    *args,
    name: str = None,
    iterations: int = 1,
    warmup_iterations: int = 0,
    measure_memory: bool = True,
    calculate_throughput: bool = False,
    throughput_items: int = 1,
    parameters: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
    **kwargs
) -> BenchmarkResult:
    """
    Run a benchmark on a function.

    Args:
        func: The function to benchmark
        *args: Arguments to pass to the function
        name: Name of the benchmark (defaults to function name)
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run (not included in results)
        measure_memory: Whether to measure memory usage
        calculate_throughput: Whether to calculate throughput
        throughput_items: Number of items processed in each iteration (for throughput calculation)
        parameters: Additional parameters to store with the result
        metadata: Additional metadata to store with the result
        **kwargs: Keyword arguments to pass to the function

    Returns:
        BenchmarkResult: The benchmark results
    """
    name = name or func.__name__
    parameters = parameters or {}
    metadata = metadata or {}

    # Run warmup iterations
    for _ in range(warmup_iterations):
        func(*args, **kwargs)

    # Force garbage collection before measuring
    gc.collect()

    # Measure memory before
    memory_before = measure_memory_usage() if measure_memory else 0

    # Measure execution time
    start_time = time.time()
    for _ in range(iterations):
        func(*args, **kwargs)
    end_time = time.time()

    # Measure memory after
    memory_after = measure_memory_usage() if measure_memory else 0

    # Calculate metrics
    execution_time = (end_time - start_time) / iterations
    memory_usage = memory_after - memory_before if measure_memory else 0
    throughput = throughput_items / execution_time if calculate_throughput else 0

    # Create and return result
    return BenchmarkResult(
        name=name,
        execution_time=execution_time,
        memory_usage=memory_usage,
        throughput=throughput,
        iterations=iterations,
        parameters=parameters,
        metadata=metadata
    )


async def run_async_benchmark(
    func: Callable,
    *args,
    name: str = None,
    iterations: int = 1,
    warmup_iterations: int = 0,
    measure_memory: bool = True,
    calculate_throughput: bool = False,
    throughput_items: int = 1,
    parameters: Dict[str, Any] = None,
    metadata: Dict[str, Any] = None,
    **kwargs
) -> BenchmarkResult:
    """
    Run a benchmark on an async function.

    Args:
        func: The async function to benchmark
        *args: Arguments to pass to the function
        name: Name of the benchmark (defaults to function name)
        iterations: Number of iterations to run
        warmup_iterations: Number of warmup iterations to run (not included in results)
        measure_memory: Whether to measure memory usage
        calculate_throughput: Whether to calculate throughput
        throughput_items: Number of items processed in each iteration (for throughput calculation)
        parameters: Additional parameters to store with the result
        metadata: Additional metadata to store with the result
        **kwargs: Keyword arguments to pass to the function

    Returns:
        BenchmarkResult: The benchmark results
    """
    name = name or func.__name__
    parameters = parameters or {}
    metadata = metadata or {}

    # Run warmup iterations
    for _ in range(warmup_iterations):
        await func(*args, **kwargs)

    # Force garbage collection before measuring
    gc.collect()

    # Measure memory before
    memory_before = measure_memory_usage() if measure_memory else 0

    # Measure execution time
    start_time = time.time()
    for _ in range(iterations):
        await func(*args, **kwargs)
    end_time = time.time()

    # Measure memory after
    memory_after = measure_memory_usage() if measure_memory else 0

    # Calculate metrics
    execution_time = (end_time - start_time) / iterations
    memory_usage = memory_after - memory_before if measure_memory else 0
    throughput = throughput_items / execution_time if calculate_throughput else 0

    # Create and return result
    return BenchmarkResult(
        name=name,
        execution_time=execution_time,
        memory_usage=memory_usage,
        throughput=throughput,
        iterations=iterations,
        parameters=parameters,
        metadata=metadata
    )


def run_benchmarks(benchmarks: List[Tuple[Callable, List, Dict]]) -> List[BenchmarkResult]:
    """
    Run multiple benchmarks.

    Args:
        benchmarks: List of (function, args, kwargs) tuples to benchmark

    Returns:
        List[BenchmarkResult]: The benchmark results
    """
    results = []
    for func, args, kwargs in benchmarks:
        result = run_benchmark(func, *args, **kwargs)
        results.append(result)
    return results


async def run_async_benchmarks(benchmarks: List[Tuple[Callable, List, Dict]]) -> List[BenchmarkResult]:
    """
    Run multiple async benchmarks.

    Args:
        benchmarks: List of (async function, args, kwargs) tuples to benchmark

    Returns:
        List[BenchmarkResult]: The benchmark results
    """
    results = []
    for func, args, kwargs in benchmarks:
        result = await run_async_benchmark(func, *args, **kwargs)
        results.append(result)
    return results


def print_benchmark_results(results: List[BenchmarkResult]) -> None:
    """
    Print benchmark results in a formatted table.

    Args:
        results: List of benchmark results to print
    """
    print("\n=== Benchmark Results ===")
    print(f"{'Name':<30} {'Time (s)':<10} {'Memory (MB)':<12} {'Throughput':<10} {'Iterations':<10}")
    print("-" * 80)
    for result in results:
        print(f"{result.name:<30} {result.execution_time:<10.4f} {result.memory_usage:<12.2f} {result.throughput:<10.2f} {result.iterations:<10}")


def save_benchmark_results(results: List[BenchmarkResult], file_path: str) -> None:
    """
    Save benchmark results to a JSON file.

    Args:
        results: List of benchmark results to save
        file_path: Path to save the results to
    """
    with open(file_path, 'w') as f:
        json.dump([result.to_dict() for result in results], f, indent=2)


def load_benchmark_results(file_path: str) -> List[BenchmarkResult]:
    """
    Load benchmark results from a JSON file.

    Args:
        file_path: Path to load the results from

    Returns:
        List[BenchmarkResult]: The loaded benchmark results
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [BenchmarkResult(**result) for result in data]


def plot_benchmark_results(
    results: List[BenchmarkResult],
    metric: str = 'execution_time',
    title: str = None,
    save_path: str = None
) -> None:
    """
    Plot benchmark results.

    Args:
        results: List of benchmark results to plot
        metric: Metric to plot ('execution_time', 'memory_usage', or 'throughput')
        title: Title for the plot
        save_path: Path to save the plot to (if None, the plot is displayed)
    """
    if not results:
        print("No results to plot")
        return

    # Extract data
    names = [result.name for result in results]
    if metric == 'execution_time':
        values = [result.execution_time for result in results]
        ylabel = 'Execution Time (s)'
        title = title or 'Benchmark Execution Time'
    elif metric == 'memory_usage':
        values = [result.memory_usage for result in results]
        ylabel = 'Memory Usage (MB)'
        title = title or 'Benchmark Memory Usage'
    elif metric == 'throughput':
        values = [result.throughput for result in results]
        ylabel = 'Throughput (items/s)'
        title = title or 'Benchmark Throughput'
    else:
        raise ValueError(f"Invalid metric: {metric}")

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.bar(names, values)
    plt.xlabel('Benchmark')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save or display
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
