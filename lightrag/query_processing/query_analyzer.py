"""
Query analyzer module for LightRAG.

This module provides functionality for analyzing user queries to extract
intent, key entities, keywords, and expanded terms.
"""

import json
import logging
import time
from typing import Dict, Any, Callable, Optional, List, Tuple, Union, Awaitable

from ..config_loader import get_enhanced_config

# Set up logger
logger = logging.getLogger(__name__)


async def process_query(
    query_text: str, 
    llm_func: Callable[[str, Optional[str], Optional[bool]], Awaitable[str]],
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Process a query and extract important information.

    Args:
        query_text: The user query string
        llm_func: Async function to call LLM with (query, system_prompt, stream) signature
        max_retries: Maximum number of retries for LLM calls
        retry_delay: Delay between retries in seconds

    Returns:
        Dict containing query analysis results:
            - entity_types: Potential entity types mentioned in the query
            - keywords: Key terms extracted from the query
            - intent: Understanding of query intent (e.g., explanation, comparison)
            - expanded_query: Enhanced version of the query for better retrieval
            - original_query: The original query text
    """
    # Build prompt for query analysis
    prompt = _build_query_analysis_prompt(query_text)

    # Call LLM for query analysis with retries
    response_text = None
    attempts = 0

    while attempts < max_retries and response_text is None:
        try:
            response_text = await llm_func(prompt, None, False)
        except Exception as e:
            attempts += 1
            logger.warning(f"LLM API call attempt {attempts} failed: {str(e)}")
            if attempts < max_retries:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to call LLM after {max_retries} attempts")
                # Return default analysis on failure
                return {
                    "entity_types": [],
                    "keywords": [],
                    "intent": "information",
                    "expanded_query": query_text,
                    "original_query": query_text
                }

    # Parse the response
    analysis = _parse_query_analysis_response(response_text)
    
    # Add the original query
    analysis['original_query'] = query_text

    return analysis


def _build_query_analysis_prompt(query: str) -> str:
    """
    Build a prompt for query analysis.

    Args:
        query: The user query

    Returns:
        Prompt string for LLM
    """
    prompt = f"""You are an expert query analyzer for a domain knowledge graph.

TASK:
Analyze the following query and extract key information needed for effective retrieval.

QUERY:
```
{query}
```

INSTRUCTIONS:
1. Identify potential entity types mentioned in the query
2. Extract key terms or concepts
3. Determine the query intent (explanation, comparison, definition, example, etc.)
4. Create an expanded version of the query that might yield better search results

FORMAT YOUR RESPONSE AS JSON:
{{
  "entity_types": ["EntityType1", "EntityType2"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "intent": "explanation",
  "expanded_query": "Enhanced version of the query with synonyms and related terms"
}}

IMPORTANT:
- Return ONLY the JSON object
- The knowledge graph contains domain-specific entity types like Concept, Entity, Process, etc.
- For entity_types, only include types if you're confident they're present
- Include 3-5 most relevant keywords
- Be specific about the intent
"""
    return prompt


def _parse_query_analysis_response(response_text: str) -> Dict[str, Any]:
    """
    Parse LLM's query analysis response.

    Args:
        response_text: LLM's response

    Returns:
        Parsed analysis dict
    """
    try:
        # Extract JSON from the response (LLM sometimes adds markdown formatting)
        json_text = response_text
        if "```json" in response_text:
            json_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_text = response_text.split("```")[1].strip()

        # Parse the JSON
        analysis = json.loads(json_text)

        # Ensure required fields are present
        if "entity_types" not in analysis:
            analysis["entity_types"] = []

        if "keywords" not in analysis:
            analysis["keywords"] = []

        if "intent" not in analysis:
            analysis["intent"] = "information"

        if "expanded_query" not in analysis:
            analysis["expanded_query"] = ""

        return analysis

    except Exception as e:
        logger.error(f"Error parsing query analysis response: {str(e)}")
        logger.debug(f"Raw response: {response_text}")
        # Return a default analysis
        return {
            "entity_types": [],
            "keywords": [],
            "intent": "information",
            "expanded_query": ""
        }
