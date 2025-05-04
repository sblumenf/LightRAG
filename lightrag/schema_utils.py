"""
Schema utilities for LightRAG.

This module provides utility functions for loading and working with schema files.
"""

import json
import logging
import os
from typing import Dict, Any, Optional

# Get logger
logger = logging.getLogger(__name__)

def load_schema(schema_path: str) -> Dict[str, Any]:
    """
    Load a schema from a JSON file.
    
    Args:
        schema_path (str): Path to the schema JSON file
        
    Returns:
        Dict[str, Any]: The loaded schema as a dictionary
        
    Raises:
        FileNotFoundError: If the schema file does not exist
        json.JSONDecodeError: If the schema file contains invalid JSON
        ValueError: If the schema path is empty or None
    """
    if not schema_path:
        logger.error("Schema path is empty or None")
        raise ValueError("Schema path cannot be empty or None")
    
    # Normalize path
    schema_path = os.path.abspath(os.path.expanduser(schema_path))
    
    # Check if file exists
    if not os.path.isfile(schema_path):
        logger.error(f"Schema file not found: {schema_path}")
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    try:
        # Open and parse JSON file
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
            logger.info(f"Successfully loaded schema from {schema_path}")
            return schema
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in schema file {schema_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading schema from {schema_path}: {str(e)}")
        raise
