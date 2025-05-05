#!/bin/bash
# Run pytest excluding Ollama tests and graph storage tests
source venv/bin/activate
python -m pytest -k "not test_lightrag_ollama_chat and not test_graph_storage"
