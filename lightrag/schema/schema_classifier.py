"""
Schema Classifier for LightRAG.

This module provides functionality to classify text chunks according to
the schema, using LLM-based classification.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable

from ..text_chunker import TextChunk
from .schema_loader import SchemaLoader
from ..config_loader import get_enhanced_config

# Set up logger
logger = logging.getLogger(__name__)


class SchemaClassifier:
    """
    Classifies text chunks according to the schema using an LLM.
    """

    def __init__(
        self,
        schema_loader: SchemaLoader,
        llm_func: Callable,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the schema classifier.

        Args:
            schema_loader: SchemaLoader instance with loaded schema
            llm_func: Async function to call the LLM
            config: Optional configuration dictionary for overrides
        """
        self.schema_loader = schema_loader
        self.llm_func = llm_func
        self.config = config if config is not None else {}

        # Set default values
        self.schema_match_threshold = self.config.get('schema_match_confidence_threshold', 0.75)
        self.new_type_threshold = self.config.get('new_type_confidence_threshold', 0.85)
        self.default_entity_type = self.config.get('default_entity_type', 'UNKNOWN')
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)

        # Extract relevant schema information for prompts
        self.entity_types = []
        self.relationship_types = []
        self.entity_properties = {}

        # Load schema information if available
        if self.schema_loader and self.schema_loader.is_schema_loaded():
            self.entity_types = list(self.schema_loader.get_entity_types())
            self.relationship_types = list(self.schema_loader.get_relationship_types())
            for entity_type in self.entity_types:
                self.entity_properties[entity_type] = list(self.schema_loader.get_entity_properties(entity_type))
            logger.info(f"Initialized SchemaClassifier with {len(self.entity_types)} entity types")
        else:
            logger.warning("SchemaClassifier initialized without a loaded schema.")

    async def classify_chunk(self, chunk: TextChunk) -> Dict[str, Any]:
        """
        Classify a text chunk according to the schema.

        Args:
            chunk: The TextChunk to classify

        Returns:
            Dict containing classification results:
                - entity_type: The determined entity type
                - properties: Dict of extracted properties
                - confidence: Confidence score (0-1)
        """
        # Check if schema is loaded
        if not self.schema_loader or not self.schema_loader.is_schema_loaded():
            logger.warning("Schema not loaded in SchemaClassifier. Cannot perform schema-based classification. Returning default entity type.")
            result = {
                "entity_type": self.default_entity_type,
                "properties": {},
                "confidence": 0.0,
                "reasoning": "Schema not loaded",
                "extracted_entities": [],
                "extracted_relationships": []
            }
            # Store classification in chunk metadata
            chunk.metadata["schema_classification"] = result
            return result

        # Prepare prompt for classification
        prompt = self._build_classification_prompt(chunk)

        # Call LLM for classification
        try:
            result = await self._call_llm(prompt)
            # Parse the response
            classification = self._parse_classification_response(result)
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            # Create default classification on error
            classification = {
                "entity_type": self.default_entity_type,
                "properties": {},
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "extracted_entities": [],
                "extracted_relationships": []
            }
            # Store classification in chunk metadata
            chunk.metadata["schema_classification"] = classification
            return classification

        # Apply confidence threshold filtering
        confidence = classification.get("confidence", 0.0)
        entity_type = classification.get("entity_type")

        # If entity type is not the default and confidence is below threshold, use default
        if (entity_type != self.default_entity_type and
            confidence < self.schema_match_threshold):
            logger.info(f"Entity type {entity_type} below confidence threshold ({confidence:.2f}), using default type {self.default_entity_type}")
            classification["entity_type"] = self.default_entity_type

        # If entity type is not in schema and not default and confidence is below new type threshold, use default
        if (entity_type not in self.entity_types and
            entity_type != self.default_entity_type and
            confidence < self.new_type_threshold):
            logger.info(f"New entity type {entity_type} below new type threshold ({confidence:.2f}), using default type {self.default_entity_type}")
            classification["entity_type"] = self.default_entity_type

        # Store classification in chunk metadata
        chunk.metadata["schema_classification"] = classification

        return classification

    async def classify_chunks(self, chunks: List[TextChunk], batch_size: int = 10) -> List[TextChunk]:
        """
        Classify multiple text chunks.

        Args:
            chunks: List of TextChunk objects to classify
            batch_size: Number of chunks to process in a batch

        Returns:
            List of classified TextChunk objects
        """
        logger.info(f"Classifying {len(chunks)} chunks using schema")

        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)")

            # Classify each chunk in the batch
            for chunk in batch:
                try:
                    await self.classify_chunk(chunk)
                except Exception as e:
                    logger.error(f"Error classifying chunk {chunk.chunk_id}: {str(e)}")
                    # Add a default classification to avoid further errors
                    chunk.metadata["schema_classification"] = {
                        "entity_type": self.default_entity_type,
                        "properties": {},
                        "confidence": 0.0
                    }

        return chunks

    def _build_classification_prompt(self, chunk: TextChunk) -> str:
        """
        Build a prompt for chunk classification.

        Args:
            chunk: The chunk to classify

        Returns:
            Prompt string for the LLM
        """
        # Construct a list of available entity types with their properties
        entity_type_descriptions = []
        for entity_type in self.entity_types:
            props = self.entity_properties.get(entity_type, [])
            props_str = ", ".join(props[:10])  # Limit to first 10 properties to avoid huge prompts
            if len(props) > 10:
                props_str += "..."
            entity_type_descriptions.append(f"- {entity_type}: properties = [{props_str}]")

        entity_types_info = "\n".join(entity_type_descriptions)

        # Limit text length to avoid token issues
        chunk_text = chunk.text[:2000] if len(chunk.text) > 2000 else chunk.text

        prompt = f"""You are an expert knowledge graph entity classifier.

TASK:
1. Analyze the following text and determine the most appropriate entity type from the schema provided
2. Extract relevant properties for that entity type
3. Provide a confidence score for your classification

TEXT TO ANALYZE:
```
{chunk_text}
```

SCHEMA ENTITY TYPES:
{entity_types_info}

INSTRUCTIONS:
1. Identify the primary entity type from the schema that best matches the text
2. Extract ONLY the properties defined for that entity type in the schema
3. Provide a confidence score (0.0-1.0) for your classification
4. If the text doesn't match any schema entity type well, use "{self.default_entity_type}" as the entity type

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
- If a text segment doesn't contain identifiable entities, use "{self.default_entity_type}" as the entity_type
"""
        return prompt

    async def _call_llm(self, prompt: str) -> str:
        """
        Call LLM with the given prompt.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            LLM's response text
        """
        response_text = None
        attempts = 0

        while attempts < self.max_retries and response_text is None:
            try:
                response_text = await self.llm_func(prompt)
            except Exception as e:
                attempts += 1
                logger.warning(f"LLM API call attempt {attempts} failed: {str(e)}")
                if attempts < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to call LLM after {self.max_retries} attempts")
                    raise

        return response_text

    def _parse_classification_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM's classification response.

        Args:
            response_text: LLM's response

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
                logger.warning(f"Missing entity_type in classification response, defaulting to '{self.default_entity_type}'")
                classification["entity_type"] = self.default_entity_type

            if "properties" not in classification:
                classification["properties"] = {}

            if "confidence" not in classification:
                classification["confidence"] = 0.5

            if "reasoning" not in classification:
                classification["reasoning"] = "No reasoning provided"

            # Add empty lists for extracted entities and relationships if not present
            if "extracted_entities" not in classification:
                classification["extracted_entities"] = []

            if "extracted_relationships" not in classification:
                classification["extracted_relationships"] = []

            return classification

        except Exception as e:
            logger.error(f"Error parsing classification response: {str(e)}")
            logger.debug(f"Raw response: {response_text}")
            # Return a default classification
            return {
                "entity_type": self.default_entity_type,
                "properties": {},
                "confidence": 0.0,
                "reasoning": f"Error parsing response: {str(e)}",
                "extracted_entities": [],
                "extracted_relationships": []
            }
