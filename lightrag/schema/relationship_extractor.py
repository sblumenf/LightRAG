"""
Relationship extractor module for LightRAG.

This module provides functionality to extract relationships between entities using LLM.
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from functools import partial

from ..utils import (
    logger,
    clean_str,
    normalize_extracted_info,
    is_float_regex,
    use_llm_func_with_cache,
)
from ..prompt import PROMPTS
from ..base import BaseKVStorage
from .schema_validator import SchemaValidator

logger = logging.getLogger(__name__)


class RelationshipExtractor:
    """
    Extracts relationships between entities using LLM with schema validation.
    """

    def __init__(
        self,
        schema_validator: SchemaValidator,
        global_config: Dict[str, Any],
        llm_response_cache: Optional[BaseKVStorage] = None,
    ):
        """
        Initialize the relationship extractor.

        Args:
            schema_validator: Schema validator instance
            global_config: Global configuration dictionary
            llm_response_cache: Optional cache for LLM responses
        """
        self.schema_validator = schema_validator
        self.global_config = global_config
        self.llm_response_cache = llm_response_cache
        self.use_llm_func = global_config["llm_model_func"]
        # Apply higher priority (7) to relationship extraction tasks
        self.use_llm_func = partial(self.use_llm_func, _priority=7)

    async def extract_relationships(
        self, entities: List[Dict[str, Any]], chunk_key: str, file_path: str = "unknown_source"
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities using LLM with schema validation.

        Args:
            entities: List of entities to extract relationships between
            chunk_key: The chunk key for source tracking
            file_path: The file path for citation

        Returns:
            List of extracted relationships
        """
        if not entities or len(entities) < 2:
            return []

        # Prepare entity information for the prompt
        entity_info = []
        for entity in entities:
            entity_info.append(
                {
                    "name": entity["entity_name"],
                    "type": entity["entity_type"],
                    "description": entity["description"],
                }
            )

        # Get relationship types from schema
        relationship_types = self.schema_validator.get_relationship_types()
        if not relationship_types:
            logger.warning("No relationship types defined in schema")
            return []

        # Build relationship extraction prompt
        prompt_template = PROMPTS.get("schema_relationship_extraction", PROMPTS["relationship_extraction"])

        # Add schema information to the prompt
        schema_info = {
            "relationship_types": relationship_types,
            "entity_types": self.schema_validator.get_entity_types(),
        }

        # Add relationship definitions
        relationship_definitions = []
        for rel_type in relationship_types:
            rel_schema = self.schema_validator.relationship_types.get(rel_type, {})
            source_type = rel_schema.get("source", "")
            target_type = rel_schema.get("target", "")
            properties = rel_schema.get("properties", [])

            rel_def = {
                "name": rel_type,
                "source": source_type,
                "target": target_type,
                "properties": properties,
            }
            relationship_definitions.append(rel_def)

        schema_info["relationship_definitions"] = relationship_definitions

        # Format the prompt
        language = self.global_config["addon_params"].get(
            "language", PROMPTS["DEFAULT_LANGUAGE"]
        )

        # Use safe string formatting to avoid issues with JSON
        if "schema_relationship_extraction" in PROMPTS and prompt_template == PROMPTS["schema_relationship_extraction"]:
            prompt = prompt_template.format(
                entities=json.dumps(entity_info),
                schema=json.dumps(schema_info),
                language=language,
            )
        else:
            # For regular relationship extraction without schema
            prompt = prompt_template.format(
                entities=json.dumps(entity_info),
                language=language,
            )

        # Call LLM to extract relationships
        result = await use_llm_func_with_cache(
            prompt,
            self.use_llm_func,
            llm_response_cache=self.llm_response_cache,
            max_tokens=2000,
            cache_type="extract",
        )

        # Parse the LLM response to extract relationships
        relationships = self._parse_relationship_response(result, chunk_key, file_path)

        # Validate relationships against schema
        validated_relationships = []
        for rel in relationships:
            # Get entity types for source and target
            source_entity = next((e for e in entities if e["entity_name"] == rel["src_id"]), None)
            target_entity = next((e for e in entities if e["entity_name"] == rel["tgt_id"]), None)

            if not source_entity or not target_entity:
                logger.warning(f"Source or target entity not found for relationship: {rel}")
                continue

            source_type = source_entity["entity_type"]
            target_type = target_entity["entity_type"]

            # Extract relationship type from keywords
            rel_type = rel.get("keywords", "RELATED_TO")

            # Validate relationship against schema
            properties = {}  # Extract properties from description if needed
            is_valid, error_msg = self.schema_validator.validate_relationship(
                rel_type, source_type, target_type, properties
            )

            if is_valid:
                validated_relationships.append(rel)
            else:
                logger.warning(f"Invalid relationship: {error_msg}")

        return validated_relationships

    def _parse_relationship_response(
        self, response: str, chunk_key: str, file_path: str
    ) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract relationships.

        Args:
            response: LLM response text
            chunk_key: The chunk key for source tracking
            file_path: The file path for citation

        Returns:
            List of extracted relationships
        """
        relationships = []

        # Try to extract JSON from the response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                json_data = json.loads(json_match.group(0))
                if isinstance(json_data, dict) and "relationships" in json_data:
                    for rel in json_data["relationships"]:
                        if "source" in rel and "target" in rel and "type" in rel:
                            source = normalize_extracted_info(rel["source"], is_entity=True)
                            target = normalize_extracted_info(rel["target"], is_entity=True)
                            rel_type = rel["type"]
                            description = rel.get("description", f"{source} {rel_type} {target}")

                            # Extract properties if available
                            properties = rel.get("properties", {})
                            if properties:
                                prop_str = "; ".join([f"{k}: {v}" for k, v in properties.items()])
                                description = f"{description}. Properties: {prop_str}"

                            weight = rel.get("weight", 1.0)
                            if not is_float_regex(str(weight)):
                                weight = 1.0

                            relationships.append({
                                "src_id": source,
                                "tgt_id": target,
                                "weight": float(weight),
                                "description": description,
                                "keywords": rel_type,
                                "source_id": chunk_key,
                                "file_path": file_path,
                                "properties": properties
                            })
                    return relationships
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from LLM response")

        # Fallback to regex parsing if JSON extraction fails
        # Look for relationship patterns in the text
        rel_pattern = r'\(([^)]+)\)\s*-\[([^]]+)\]->\s*\(([^)]+)\)'
        for match in re.finditer(rel_pattern, response):
            try:
                source = normalize_extracted_info(match.group(1), is_entity=True)
                rel_type = match.group(2).strip()
                target = normalize_extracted_info(match.group(3), is_entity=True)

                description = f"{source} {rel_type} {target}"

                relationships.append({
                    "src_id": source,
                    "tgt_id": target,
                    "weight": 1.0,
                    "description": description,
                    "keywords": rel_type,
                    "source_id": chunk_key,
                    "file_path": file_path,
                })
            except Exception as e:
                logger.warning(f"Error parsing relationship: {str(e)}")

        return relationships
