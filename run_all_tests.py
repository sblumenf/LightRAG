#!/usr/bin/env python
"""
Comprehensive test runner for all LightRAG phases.

This script runs tests for all phases of the LightRAG enhancement plan,
collects coverage data, and generates a coverage report.
"""

import os
import sys
import argparse
import subprocess
from typing import List, Dict, Any
import json
import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run tests for LightRAG phases")
    
    parser.add_argument("--phase", type=int, choices=range(7), help="Specific phase to test (0-6)")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--output", help="Output directory for reports")
    
    return parser.parse_args()

def run_phase_tests(phase: int, coverage: bool = False, verbose: bool = False) -> int:
    """Run tests for a specific phase."""
    print(f"\n=== Running tests for Phase {phase} ===\n")
    
    # Define test patterns for each phase
    phase_patterns = {
        0: ["test_config_loader.py", "test_schema_utils.py"],
        1: ["test_document_processing.py", "document_processing/test_"],
        2: ["test_schema_components.py", "test_text_chunker.py", "schema/test_"],
        3: ["test_entity_resolution.py", "test_kg_index_sync.py"],
        4: ["test_query_processing.py", "test_intelligent_retrieval.py", "test_intelligent_retrieval_e2e.py"],
        5: ["test_advanced_generation.py", "test_cot_integration.py", "test_llm_generator_cot.py", 
            "test_placeholder_resolver.py", "test_diagram_formula_integration.py"],
        6: ["integration/test_", "e2e/test_"]
    }
    
    # Build the pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add patterns for the specified phase
    for pattern in phase_patterns.get(phase, []):
        cmd.extend(["-k", pattern])
    
    # Add coverage options if requested
    if coverage:
        cmd.extend(["--cov=lightrag", "--cov-report=term"])
    
    # Add verbosity if requested
    if verbose:
        cmd.append("-v")
    
    # Run the command
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=not verbose)
    
    # Print output if not verbose (verbose mode shows output in real-time)
    if not verbose:
        print(result.stdout.decode('utf-8'))
        if result.stderr:
            print(f"Errors:\n{result.stderr.decode('utf-8')}")
    
    return result.returncode

def run_all_tests(coverage: bool = False, verbose: bool = False) -> Dict[int, int]:
    """Run tests for all phases."""
    results = {}
    
    for phase in range(7):
        results[phase] = run_phase_tests(phase, coverage, verbose)
    
    return results

def generate_coverage_report(output_dir: str = None):
    """Generate a coverage report."""
    print("\n=== Generating coverage report ===\n")
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, f"coverage_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        report_path = f"coverage_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Run pytest with coverage options
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=lightrag",
        f"--cov-report=html:{report_path}_html",
        f"--cov-report=xml:{report_path}.xml"
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    
    if result.returncode == 0:
        print(f"Coverage report generated at: {report_path}_html")
        print(f"Coverage XML report generated at: {report_path}.xml")
    else:
        print("Failed to generate coverage report")
        print(result.stderr.decode('utf-8'))
    
    return result.returncode

def main():
    """Main function."""
    args = parse_args()
    
    if args.phase is not None:
        # Run tests for a specific phase
        result = run_phase_tests(args.phase, args.coverage, args.verbose)
    else:
        # Run tests for all phases
        results = run_all_tests(args.coverage, args.verbose)
        result = 1 if any(code != 0 for code in results.values()) else 0
    
    # Generate coverage report if requested
    if args.coverage and (args.phase is None or result == 0):
        coverage_result = generate_coverage_report(args.output)
        if coverage_result != 0:
            result = coverage_result
    
    # Return the overall result code
    sys.exit(result)

if __name__ == "__main__":
    main()