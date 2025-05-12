#!/bin/bash
# Comprehensive test runner for LightRAG

# Enable error handling
set -e

# Define help function
function show_help {
    echo "LightRAG Test Runner"
    echo "Usage: ./run_tests.sh [OPTION]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -a, --all                 Run all tests"
    echo "  -p, --phase PHASE         Run tests for specific phase (0-6)"
    echo "  -c, --coverage            Generate coverage report"
    echo "  -i, --integration         Run integration tests"
    echo "  -e, --e2e                 Run end-to-end tests"
    echo "  -d, --diagram-formula     Run diagram and formula integration test"
    echo "  -b, --benchmarks          Run benchmarks"
    echo "  -v, --verbose             Verbose output"
    echo ""
    exit 0
}

# Check if a Python virtual environment exists
if [ -d "venv" ]; then
    # Activate virtual environment
    source venv/bin/activate
    echo "Virtual environment activated"
else
    echo "Warning: No virtual environment found. Using system Python."
fi

# Run specified tests
case "$1" in
    -h|--help)
        show_help
        ;;
    -a|--all)
        echo "Running all tests with coverage..."
        python3 run_all_tests.py --coverage --verbose
        ;;
    -p|--phase)
        PHASE="$2"
        echo "Running tests for Phase ${PHASE}..."
        python3 run_all_tests.py --phase "$PHASE" --verbose
        ;;
    -c|--coverage)
        echo "Generating coverage report..."
        python3 run_all_tests.py --coverage --output coverage_reports
        ;;
    -i|--integration)
        echo "Running integration tests..."
        python3 -m pytest tests/integration/ -v
        ;;
    -e|--e2e)
        echo "Running end-to-end tests..."
        python3 -m pytest tests/e2e/ -v
        ;;
    -d|--diagram-formula)
        echo "Running diagram and formula integration test..."
        python3 integration_test_diagram_formula.py
        python3 citation_test.py
        ;;
    -b|--benchmarks)
        echo "Running benchmarks..."
        python3 scripts/run_benchmarks.py
        ;;
    -v|--verbose)
        echo "Running all tests in verbose mode..."
        python3 -m pytest -v
        ;;
    *)
        echo "Running default test suite..."
        python3 -m pytest
        ;;
esac

# If we activated a virtual environment, deactivate it
if [ -d "venv" ]; then
    deactivate
    echo "Virtual environment deactivated"
fi

echo "Tests completed!"