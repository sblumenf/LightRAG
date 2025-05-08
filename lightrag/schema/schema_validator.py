"""
Schema validation module for LightRAG.

This module provides functionality to validate entities and relationships against a schema.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class SchemaValidator:
    """
    Validates entities and relationships against a schema.
    """

    def __init__(self, schema_path: str):
        """
        Initialize the schema validator with a schema file.

        Args:
            schema_path: Path to the schema JSON file
        """
        self.schema_path = schema_path
        self.schema = self._load_schema(schema_path)
        self.entity_types = {entity["name"]: entity for entity in self.schema.get("entities", [])}
        self.relationship_types = {rel["name"]: rel for rel in self.schema.get("relationships", [])}

    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """
        Load the schema from a JSON file.

        Args:
            schema_path: Path to the schema JSON file

        Returns:
            The schema as a dictionary
        """
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            return schema
        except Exception as e:
            logger.error(f"Error loading schema from {schema_path}: {str(e)}")
            # Return a minimal valid schema
            return {"entities": [], "relationships": []}

    def validate_entity(self, entity_type: str, properties: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate an entity against the schema.

        Args:
            entity_type: The type of the entity
            properties: The properties of the entity

        Returns:
            A tuple of (is_valid, error_message)
        """
        # Check if entity type exists in schema
        if entity_type not in self.entity_types:
            return False, f"Entity type '{entity_type}' not found in schema"

        # Get the entity schema
        entity_schema = self.entity_types[entity_type]
        schema_properties = {prop["name"]: prop for prop in entity_schema.get("properties", [])}

        # Check required properties
        for prop_name, prop_schema in schema_properties.items():
            if prop_schema.get("required", False) and prop_name not in properties:
                return False, f"Required property '{prop_name}' missing for entity type '{entity_type}'"

        # Check property types
        for prop_name, prop_value in properties.items():
            if prop_name in schema_properties:
                prop_schema = schema_properties[prop_name]
                prop_type = prop_schema.get("type", "string")
                
                # Basic type validation
                if prop_type == "string" and not isinstance(prop_value, str):
                    return False, f"Property '{prop_name}' should be a string"
                elif prop_type == "integer" and not isinstance(prop_value, int):
                    return False, f"Property '{prop_name}' should be an integer"
                elif prop_type == "float" and not isinstance(prop_value, (int, float)):
                    return False, f"Property '{prop_name}' should be a float"
                elif prop_type == "boolean" and not isinstance(prop_value, bool):
                    return False, f"Property '{prop_name}' should be a boolean"
                elif prop_type == "date" and not isinstance(prop_value, str):
                    # Simple date validation - could be enhanced
                    return False, f"Property '{prop_name}' should be a date string"

        return True, ""

    def validate_relationship(self, relationship_type: str, source_type: str, target_type: str, 
                             properties: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a relationship against the schema.

        Args:
            relationship_type: The type of the relationship
            source_type: The type of the source entity
            target_type: The type of the target entity
            properties: The properties of the relationship

        Returns:
            A tuple of (is_valid, error_message)
        """
        # Check if relationship type exists in schema
        if relationship_type not in self.relationship_types:
            return False, f"Relationship type '{relationship_type}' not found in schema"

        # Get the relationship schema
        rel_schema = self.relationship_types[relationship_type]
        
        # Check source and target entity types
        if rel_schema.get("source") != source_type:
            return False, f"Invalid source entity type '{source_type}' for relationship '{relationship_type}'"
        
        if rel_schema.get("target") != target_type:
            return False, f"Invalid target entity type '{target_type}' for relationship '{relationship_type}'"

        # Check properties
        schema_properties = {prop["name"]: prop for prop in rel_schema.get("properties", [])}
        
        # Check required properties
        for prop_name, prop_schema in schema_properties.items():
            if prop_schema.get("required", False) and prop_name not in properties:
                return False, f"Required property '{prop_name}' missing for relationship type '{relationship_type}'"

        # Check property types
        for prop_name, prop_value in properties.items():
            if prop_name in schema_properties:
                prop_schema = schema_properties[prop_name]
                prop_type = prop_schema.get("type", "string")
                
                # Basic type validation
                if prop_type == "string" and not isinstance(prop_value, str):
                    return False, f"Property '{prop_name}' should be a string"
                elif prop_type == "integer" and not isinstance(prop_value, int):
                    return False, f"Property '{prop_name}' should be an integer"
                elif prop_type == "float" and not isinstance(prop_value, (int, float)):
                    return False, f"Property '{prop_name}' should be a float"
                elif prop_type == "boolean" and not isinstance(prop_value, bool):
                    return False, f"Property '{prop_name}' should be a boolean"
                elif prop_type == "date" and not isinstance(prop_value, str):
                    # Simple date validation - could be enhanced
                    return False, f"Property '{prop_name}' should be a date string"

        return True, ""

    def get_entity_types(self) -> List[str]:
        """
        Get all entity types defined in the schema.

        Returns:
            A list of entity type names
        """
        return list(self.entity_types.keys())

    def get_relationship_types(self) -> List[str]:
        """
        Get all relationship types defined in the schema.

        Returns:
            A list of relationship type names
        """
        return list(self.relationship_types.keys())

    def get_entity_properties(self, entity_type: str) -> List[Dict[str, Any]]:
        """
        Get the properties for a specific entity type.

        Args:
            entity_type: The entity type name

        Returns:
            A list of property definitions
        """
        if entity_type in self.entity_types:
            return self.entity_types[entity_type].get("properties", [])
        return []

    def get_relationship_properties(self, relationship_type: str) -> List[Dict[str, Any]]:
        """
        Get the properties for a specific relationship type.

        Args:
            relationship_type: The relationship type name

        Returns:
            A list of property definitions
        """
        if relationship_type in self.relationship_types:
            return self.relationship_types[relationship_type].get("properties", [])
        return []
