"""
Simple test runner for LightRAG tests.
"""

import unittest
import sys
import os
import subprocess

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the test modules
from tests.llm.test_placeholder_resolver import TestPlaceholderResolver, TestResolvePlaceholdersInContext
from tests.llm.test_enhanced_citations import TestProcessCitations, TestLLMGenerator
from tests.llm.test_cot_implementation import TestCoTPrompt, TestExtractReasoningAndAnswer, TestLLMGeneratorCoT
from tests.llm.test_enhanced_citation_handling import TestProcessCitationsAdvanced, TestLLMGeneratorCitations
from tests.integration.test_diagram_formula_integration_class import TestDiagramFormulaIntegration
from tests.integration.test_cot_and_citations_simple import TestCoTAndCitationsSimple
from tests.integration.test_cot_and_citations_integration import TestCoTAndCitationsIntegration

if __name__ == "__main__":
    # Create a test suite for unittest tests
    test_suite = unittest.TestSuite()

    # Add the test cases - using loader instead of deprecated makeSuite
    loader = unittest.TestLoader()

    # Placeholder resolver tests
    test_suite.addTest(loader.loadTestsFromTestCase(TestPlaceholderResolver))
    test_suite.addTest(loader.loadTestsFromTestCase(TestResolvePlaceholdersInContext))

    # Enhanced citations tests
    test_suite.addTest(loader.loadTestsFromTestCase(TestProcessCitations))
    test_suite.addTest(loader.loadTestsFromTestCase(TestLLMGenerator))
    test_suite.addTest(loader.loadTestsFromTestCase(TestProcessCitationsAdvanced))
    test_suite.addTest(loader.loadTestsFromTestCase(TestLLMGeneratorCitations))

    # Chain-of-Thought tests
    test_suite.addTest(loader.loadTestsFromTestCase(TestCoTPrompt))
    test_suite.addTest(loader.loadTestsFromTestCase(TestExtractReasoningAndAnswer))
    test_suite.addTest(loader.loadTestsFromTestCase(TestLLMGeneratorCoT))

    # Integration tests
    test_suite.addTest(loader.loadTestsFromTestCase(TestDiagramFormulaIntegration))
    test_suite.addTest(loader.loadTestsFromTestCase(TestCoTAndCitationsSimple))
    test_suite.addTest(loader.loadTestsFromTestCase(TestCoTAndCitationsIntegration))

    # Run the unittest tests
    print("Running unittest tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    unittest_result = runner.run(test_suite)

    # Run pytest tests
    print("\nRunning pytest tests...")
    pytest_files = [
        "tests/llm/test_advanced_generation_pytest.py",
        "tests/llm/test_cot_implementation_pytest.py",
        "tests/llm/test_enhanced_citation_handling_pytest.py",
        "tests/integration/test_cot_and_citations_integration_pytest.py"
    ]
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest"] + pytest_files + ["-v"],
        capture_output=False
    )

    # Exit with appropriate status code
    # If either unittest or pytest fails, exit with non-zero status
    if not unittest_result.wasSuccessful() or pytest_result.returncode != 0:
        sys.exit(1)
    else:
        sys.exit(0)
