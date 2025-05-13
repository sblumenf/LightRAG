#!/usr/bin/env python3
"""
Comprehensive test coverage script for LightRAG.
Runs all tests and generates a detailed coverage report.
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import shutil

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description='Run tests with coverage report')
    parser.add_argument('--output', type=str, default='coverage-reports',
                      help='Directory to store coverage reports')
    parser.add_argument('--threshold', type=int, default=80,
                      help='Minimum coverage percentage required')
    parser.add_argument('--html', action='store_true',
                      help='Generate HTML coverage report')
    parser.add_argument('--xml', action='store_true',
                      help='Generate XML coverage report')
    parser.add_argument('--include', type=str, default='lightrag/*',
                      help='Files to include in coverage (comma-separated glob patterns)')
    parser.add_argument('--python', type=str, default='python3',
                      help='Python executable to use')
    return parser

def run_coverage(args):
    """Run tests with coverage tool."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_prefix = f"{args.output}/coverage_{timestamp}"
    
    # Verify python executable exists
    python_exec = args.python
    if not shutil.which(python_exec):
        print(f"ERROR: Python executable '{python_exec}' not found")
        return 1
    
    # Build coverage command
    cmd = [
        python_exec, '-m', 'pytest',
        '--cov=lightrag',
        f'--cov-report=term-missing',
    ]
    
    # Add HTML report if requested
    if args.html:
        cmd.append(f'--cov-report=html:{report_prefix}_html')
    
    # Add XML report if requested
    if args.xml:
        cmd.append(f'--cov-report=xml:{report_prefix}.xml')
    
    # Add include patterns
    if args.include:
        patterns = args.include.split(',')
        for pattern in patterns:
            cmd.append(f'--cov-config=.coveragerc')
    
    # Add all test directories
    cmd.extend([
        'tests/',
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print output
        print(result.stdout)
        
        if result.stderr:
            print("ERRORS:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        
        # Extract coverage percentage
        coverage_line = None
        for line in result.stdout.splitlines():
            if "TOTAL" in line and "%" in line:
                coverage_line = line
                break
        
        if coverage_line:
            try:
                # Extract the coverage percentage
                total_coverage = int(coverage_line.split('%')[0].split()[-1])
                print(f"Total coverage: {total_coverage}%")
                
                if total_coverage < args.threshold:
                    print(f"ERROR: Coverage {total_coverage}% is below the required threshold of {args.threshold}%")
                    return 1
                else:
                    print(f"SUCCESS: Coverage {total_coverage}% meets the required threshold of {args.threshold}%")
                    return 0
            except (IndexError, ValueError):
                print("Could not parse coverage percentage")
                return 1
        else:
            print("Could not find coverage total in output")
            return 1
    except Exception as e:
        print(f"ERROR: Failed to run coverage: {str(e)}")
        return 1

def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # Create .coveragerc file if it doesn't exist
    if not os.path.exists('.coveragerc'):
        with open('.coveragerc', 'w') as f:
            f.write("""[run]
source = lightrag
omit = 
    */tests/*
    */venv/*
    setup.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if __name__ == .__main__.:
    pass
    raise ImportError
""")
    
    # Run the tests with coverage
    return run_coverage(args)

if __name__ == "__main__":
    sys.exit(main())