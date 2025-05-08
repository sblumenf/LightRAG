#!/bin/bash
# Run pytest excluding graph storage tests
source venv/bin/activate
python -m pytest -k "not test_graph_storage"
