"""
Schema-based classification and property extraction functions for LightRAG.

This module provides functions for classifying text chunks according to a schema
and extracting schema-defined properties.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable

# Set up logger
logger = logging.getLogger(__name__)


async def classify_chunk_and_extract_properties(
    chunk_text: str, 
    schema: dict, 
    llm_func: Callable
) -> Dict[str, Any]:
    """
    Classify a text chunk according to the schema and extract schema-defined properties.
    
    Args:
        chunk_text: The text content of the chunk
        schema: The schema dictionary
        llm_func: Async function to call the LLM
        
    Returns:
        Dict containing:
            - entity_type: The determined entity type
            - properties: Dict of extracted properties
    """
    try:
        # Build the prompt for classification
        prompt = _build_classification_prompt(chunk_text, schema)
        
        # Call the LLM
        response = await llm_func(prompt)
        
        # Parse the response
        result = _parse_classification_response(response, schema)
        
        return result
    except Exception as e:
        logger.error(f"Error in classify_chunk_and_extract_properties: {str(e)}")
        # Return default values on error
        return {
            "entity_type": "UNKNOWN",
            "properties": {}
        }


def _build_classification_prompt(chunk_text: str, schema: dict) -> str:
    """
    Build a prompt for chunk classification.
    
    Args:
        chunk_text: The text content of the chunk
        schema: The schema dictionary
        
    Returns:
        Prompt string for the LLM
    """
    # Extract entity types and properties from schema
    entity_types = []
    entity_properties = {}
    
    # Handle different schema formats
    if 'entities' in schema and isinstance(schema['entities'], list):
        # New format with 'entities' list
        for entity in schema['entities']:
            entity_name = entity.get('name')
            if entity_name:
                entity_types.append(entity_name)
                properties = [prop.get('name') for prop in entity.get('properties', []) if prop.get('name')]
                entity_properties[entity_name] = properties
    else:
        # Old format with nested dictionaries
        for container_name, container_content in schema.items():
            if isinstance(container_content, dict):
                for category_name, category_content in container_content.items():
                    if isinstance(category_content, dict):
                        for entity_type, properties in category_content.items():
                            entity_types.append(entity_type)
                            if isinstance(properties, dict):
                                entity_properties[entity_type] = list(properties.keys())
    
    # Construct a list of available entity types with their properties
    entity_type_descriptions = []
    for entity_type in entity_types:
        props = entity_properties.get(entity_type, [])
        props_str = ", ".join(props[:10])  # Limit to first 10 properties to avoid huge prompts
        if len(props) > 10:
            props_str += "..."
        entity_type_descriptions.append(f"- {entity_type}: properties = [{props_str}]")

    entity_types_info = "\n".join(entity_type_descriptions)
    
    # Limit text length to avoid token issues
    chunk_text_limited = chunk_text[:2000] if len(chunk_text) > 2000 else chunk_text
    
    prompt = f"""You are an expert knowledge graph entity classifier.

TASK:
1. Analyze the following text and determine the most appropriate entity type from the schema provided
2. Extract relevant properties for that entity type
3. Provide a confidence score for your classification

TEXT TO ANALYZE:
```
{chunk_text_limited}
```

SCHEMA ENTITY TYPES:
{entity_types_info}

INSTRUCTIONS:
1. Identify the primary entity type from the schema that best matches the text
2. Extract ONLY the properties defined for that entity type in the schema
3. Provide a confidence score (0.0-1.0) for your classification
4. If the text doesn't match any schema entity type well, use "UNKNOWN" as the entity type

FORMAT YOUR RESPONSE AS JSON:
{{
  "entity_type": "SchemaEntityType",
  "properties": {{
    "property1": "extracted value 1",
    "property2": "extracted value 2"
  }},
  "confidence": 0.85,
  "reasoning": "Brief explanation of why this entity type was chosen"
}}

IMPORTANT:
- Return ONLY the JSON object
- Choose the most specific entity type possible
- Only include properties that are defined for the chosen entity type
- If a text segment doesn't contain identifiable entities, use "UNKNOWN" as the entity_type
"""
    return prompt


def _parse_classification_response(response_text: str, schema: dict) -> Dict[str, Any]:
    """
    Parse LLM's classification response.
    
    Args:
        response_text: LLM's response
        schema: The schema dictionary
        
    Returns:
        Parsed classification dict
    """
    try:
        # Extract JSON from the response (LLM sometimes adds markdown formatting)
        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()
        
        # Parse the JSON
        classification = json.loads(json_text)
        
        # Ensure required fields are present
        if "entity_type" not in classification:
            logger.warning("Missing entity_type in classification response, defaulting to 'UNKNOWN'")
            classification["entity_type"] = "UNKNOWN"
        
        if "properties" not in classification:
            classification["properties"] = {}
        
        return {
            "entity_type": classification.get("entity_type", "UNKNOWN"),
            "properties": classification.get("properties", {})
        }
    
    except Exception as e:
        logger.error(f"Error parsing classification response: {str(e)}")
        logger.debug(f"Raw response: {response_text}")
        # Return a default classification
        return {
            "entity_type": "UNKNOWN",
            "properties": {}
        }
