"""
Script to verify the implementation of the intelligent retrieval functionality.
"""

import os
import sys
import importlib.util

def check_module_exists(module_path):
    """Check if a module exists at the given path."""
    try:
        spec = importlib.util.spec_from_file_location("module", module_path)
        if spec is None:
            return False
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True
    except Exception as e:
        print(f"Error loading module {module_path}: {str(e)}")
        return False

def main():
    """Main function to verify implementation."""
    print("Verifying implementation of intelligent retrieval functionality...")
    
    # Check if the query processing module exists
    query_analyzer_path = "lightrag/query_processing/query_analyzer.py"
    strategy_selector_path = "lightrag/query_processing/strategy_selector.py"
    init_path = "lightrag/query_processing/__init__.py"
    
    print(f"Checking {query_analyzer_path}...")
    if os.path.exists(query_analyzer_path):
        print(f"✅ {query_analyzer_path} exists")
        if check_module_exists(query_analyzer_path):
            print(f"✅ {query_analyzer_path} can be imported")
        else:
            print(f"❌ {query_analyzer_path} cannot be imported")
    else:
        print(f"❌ {query_analyzer_path} does not exist")
    
    print(f"Checking {strategy_selector_path}...")
    if os.path.exists(strategy_selector_path):
        print(f"✅ {strategy_selector_path} exists")
        if check_module_exists(strategy_selector_path):
            print(f"✅ {strategy_selector_path} can be imported")
        else:
            print(f"❌ {strategy_selector_path} cannot be imported")
    else:
        print(f"❌ {strategy_selector_path} does not exist")
    
    print(f"Checking {init_path}...")
    if os.path.exists(init_path):
        print(f"✅ {init_path} exists")
        if check_module_exists(init_path):
            print(f"✅ {init_path} can be imported")
        else:
            print(f"❌ {init_path} cannot be imported")
    else:
        print(f"❌ {init_path} does not exist")
    
    # Check if the QueryParam class has been updated
    base_path = "lightrag/base.py"
    print(f"Checking {base_path} for QueryParam updates...")
    if os.path.exists(base_path):
        print(f"✅ {base_path} exists")
        with open(base_path, "r") as f:
            content = f.read()
            if "use_intelligent_retrieval" in content:
                print("✅ QueryParam has been updated with use_intelligent_retrieval")
            else:
                print("❌ QueryParam has not been updated with use_intelligent_retrieval")
            
            if "query_analysis" in content:
                print("✅ QueryParam has been updated with query_analysis")
            else:
                print("❌ QueryParam has not been updated with query_analysis")
            
            if "filter_by_entity_type" in content:
                print("✅ QueryParam has been updated with filter_by_entity_type")
            else:
                print("❌ QueryParam has not been updated with filter_by_entity_type")
            
            if "rerank_results" in content:
                print("✅ QueryParam has been updated with rerank_results")
            else:
                print("❌ QueryParam has not been updated with rerank_results")
    else:
        print(f"❌ {base_path} does not exist")
    
    # Check if the LightRAG.aquery method has been updated
    lightrag_path = "lightrag/lightrag.py"
    print(f"Checking {lightrag_path} for aquery updates...")
    if os.path.exists(lightrag_path):
        print(f"✅ {lightrag_path} exists")
        with open(lightrag_path, "r") as f:
            content = f.read()
            if "from .query_processing import process_query" in content:
                print("✅ LightRAG imports process_query")
            else:
                print("❌ LightRAG does not import process_query")
            
            if "query_analysis = await process_query" in content:
                print("✅ LightRAG.aquery calls process_query")
            else:
                print("❌ LightRAG.aquery does not call process_query")
            
            if "select_retrieval_strategy" in content:
                print("✅ LightRAG.aquery calls select_retrieval_strategy")
            else:
                print("❌ LightRAG.aquery does not call select_retrieval_strategy")
    else:
        print(f"❌ {lightrag_path} does not exist")
    
    # Check if the operate.py file has been updated
    operate_path = "lightrag/operate.py"
    print(f"Checking {operate_path} for filtering and reranking updates...")
    if os.path.exists(operate_path):
        print(f"✅ {operate_path} exists")
        with open(operate_path, "r") as f:
            content = f.read()
            if "filter_by_entity_type" in content:
                print("✅ operate.py has entity type filtering")
            else:
                print("❌ operate.py does not have entity type filtering")
            
            if "rerank_results" in content or "Re-ranking chunks" in content:
                print("✅ operate.py has result reranking")
            else:
                print("❌ operate.py does not have result reranking")
    else:
        print(f"❌ {operate_path} does not exist")
    
    # Check if the config_loader.py file has been updated
    config_path = "lightrag/config_loader.py"
    print(f"Checking {config_path} for configuration updates...")
    if os.path.exists(config_path):
        print(f"✅ {config_path} exists")
        with open(config_path, "r") as f:
            content = f.read()
            if "enable_intelligent_retrieval" in content:
                print("✅ config_loader.py has enable_intelligent_retrieval")
            else:
                print("❌ config_loader.py does not have enable_intelligent_retrieval")
            
            if "query_analysis_confidence_threshold" in content:
                print("✅ config_loader.py has query_analysis_confidence_threshold")
            else:
                print("❌ config_loader.py does not have query_analysis_confidence_threshold")
            
            if "auto_strategy_selection" in content:
                print("✅ config_loader.py has auto_strategy_selection")
            else:
                print("❌ config_loader.py does not have auto_strategy_selection")
            
            if "graph_intent_indicators" in content:
                print("✅ config_loader.py has graph_intent_indicators")
            else:
                print("❌ config_loader.py does not have graph_intent_indicators")
    else:
        print(f"❌ {config_path} does not exist")
    
    # Check if the test files exist
    test_query_processing_path = "tests/test_query_processing.py"
    test_intelligent_retrieval_path = "tests/test_intelligent_retrieval.py"
    test_intelligent_retrieval_e2e_path = "tests/test_intelligent_retrieval_e2e.py"
    
    print(f"Checking {test_query_processing_path}...")
    if os.path.exists(test_query_processing_path):
        print(f"✅ {test_query_processing_path} exists")
    else:
        print(f"❌ {test_query_processing_path} does not exist")
    
    print(f"Checking {test_intelligent_retrieval_path}...")
    if os.path.exists(test_intelligent_retrieval_path):
        print(f"✅ {test_intelligent_retrieval_path} exists")
    else:
        print(f"❌ {test_intelligent_retrieval_path} does not exist")
    
    print(f"Checking {test_intelligent_retrieval_e2e_path}...")
    if os.path.exists(test_intelligent_retrieval_e2e_path):
        print(f"✅ {test_intelligent_retrieval_e2e_path} exists")
    else:
        print(f"❌ {test_intelligent_retrieval_e2e_path} does not exist")
    
    print("\nVerification complete!")

if __name__ == "__main__":
    main()
