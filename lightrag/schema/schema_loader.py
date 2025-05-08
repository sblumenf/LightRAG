"""
Schema Loader for LightRAG.

This module provides functionality to load and parse schema files for use in
schema-based classification and property extraction.
"""

import json
import logging
import os
from typing import Dict, Any, List, Set, Optional, Tuple, Union

from ..schema_utils import load_schema

# Set up logger
logger = logging.getLogger(__name__)


class SchemaLoader:
    """
    Loads and parses schema files for use in schema-based classification.
    
    This class provides methods to access entity types, relationship types,
    and their properties from a schema file.
    """
    
    def __init__(self, schema_path: str):
        """
        Initialize the schema loader.
        
        Args:
            schema_path: Path to the schema JSON file
        """
        self.schema_path = schema_path
        self.schema = None
        self._entity_types = set()
        self._relationship_types = set()
        self._entity_properties = {}
        self._relationship_properties = {}
        self._relationship_domain_range = {}
        
        # Try to load the schema
        self._load_schema()
    
    def _load_schema(self) -> None:
        """
        Load the schema from the specified path.
        """
        try:
            self.schema = load_schema(self.schema_path)
            self._parse_schema()
            logger.info(f"Successfully loaded schema from {self.schema_path}")
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to load schema: {str(e)}")
            self.schema = None
    
    def _parse_schema(self) -> None:
        """
        Parse the loaded schema to extract entity types, relationship types, and properties.
        """
        if not self.schema:
            logger.warning("Cannot parse schema: No schema loaded")
            return
        
        # Check if schema has the new format with 'entities' and 'relationships' lists
        if 'entities' in self.schema and isinstance(self.schema['entities'], list):
            self._parse_list_schema()
        else:
            # Assume old format with nested dictionaries
            self._parse_nested_schema()
    
    def _parse_list_schema(self) -> None:
        """
        Parse schema in the format with 'entities' and 'relationships' lists.
        """
        # Extract entity types and properties
        for entity in self.schema.get('entities', []):
            entity_name = entity.get('name')
            if entity_name:
                self._entity_types.add(entity_name)
                properties = [prop.get('name') for prop in entity.get('properties', []) if prop.get('name')]
                self._entity_properties[entity_name] = set(properties)
        
        # Extract relationship types, properties, and domain/range
        for rel in self.schema.get('relationships', []):
            rel_name = rel.get('name')
            if rel_name:
                self._relationship_types.add(rel_name)
                properties = [prop.get('name') for prop in rel.get('properties', []) if prop.get('name')]
                self._relationship_properties[rel_name] = set(properties)
                
                # Store domain and range
                source = rel.get('source')
                target = rel.get('target')
                if source and target:
                    if rel_name not in self._relationship_domain_range:
                        self._relationship_domain_range[rel_name] = (set(), set())
                    
                    domain, range_entities = self._relationship_domain_range[rel_name]
                    domain.add(source)
                    range_entities.add(target)
    
    def _parse_nested_schema(self) -> None:
        """
        Parse schema in the nested dictionary format.
        """
        # First level is usually a container (like "Curriculum")
        for container_name, container_content in self.schema.items():
            # Second level might be categories or volumes
            for category_name, category_content in container_content.items():
                # Third level contains entity types
                if isinstance(category_content, dict):
                    for entity_type, properties in category_content.items():
                        self._entity_types.add(entity_type)
                        
                        # Extract properties
                        if isinstance(properties, dict):
                            self._entity_properties[entity_type] = set(properties.keys())
        
        # For this format, we don't have explicit relationship information
        # We'll use a default relationship type
        self._relationship_types.add("RELATED_TO")
        self._relationship_properties["RELATED_TO"] = set(["weight", "description"])
        
        # Set domain/range to allow any entity type to connect to any other
        for rel_type in self._relationship_types:
            self._relationship_domain_range[rel_type] = (self._entity_types.copy(), self._entity_types.copy())
    
    def is_schema_loaded(self) -> bool:
        """
        Check if a schema is loaded.
        
        Returns:
            True if a schema is loaded, False otherwise
        """
        return self.schema is not None
    
    def get_entity_types(self) -> Set[str]:
        """
        Get all entity types defined in the schema.
        
        Returns:
            Set of entity type names
        """
        return self._entity_types
    
    def get_relationship_types(self) -> Set[str]:
        """
        Get all relationship types defined in the schema.
        
        Returns:
            Set of relationship type names
        """
        return self._relationship_types
    
    def get_entity_properties(self, entity_type: str) -> Set[str]:
        """
        Get all properties for a specific entity type.
        
        Args:
            entity_type: The entity type to get properties for
            
        Returns:
            Set of property names
        """
        return self._entity_properties.get(entity_type, set())
    
    def get_relationship_properties(self, relationship_type: str) -> Set[str]:
        """
        Get all properties for a specific relationship type.
        
        Args:
            relationship_type: The relationship type to get properties for
            
        Returns:
            Set of property names
        """
        return self._relationship_properties.get(relationship_type, set())
    
    def get_relationship_domain_range(self, relationship_type: str) -> Tuple[Set[str], Set[str]]:
        """
        Get the domain and range for a specific relationship type.
        
        Args:
            relationship_type: The relationship type to get domain/range for
            
        Returns:
            Tuple of (domain, range) where each is a set of entity type names
        """
        return self._relationship_domain_range.get(relationship_type, (set(), set()))
    
    def get_valid_relationships(self, source_type: str, target_type: str) -> List[str]:
        """
        Get valid relationship types between two entity types.
        
        Args:
            source_type: Source entity type
            target_type: Target entity type
            
        Returns:
            List of valid relationship type names
        """
        valid_relationships = []
        
        for rel_type in self._relationship_types:
            domain, range_entities = self.get_relationship_domain_range(rel_type)
            
            # If domain or range is empty, allow any entity type
            domain_valid = not domain or source_type in domain
            range_valid = not range_entities or target_type in range_entities
            
            if domain_valid and range_valid:
                valid_relationships.append(rel_type)
        
        return valid_relationships
    
    def reload_schema(self) -> bool:
        """
        Reload the schema from the specified path.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            self._entity_types = set()
            self._relationship_types = set()
            self._entity_properties = {}
            self._relationship_properties = {}
            self._relationship_domain_range = {}
            
            self._load_schema()
            return self.is_schema_loaded()
        except Exception as e:
            logger.error(f"Error reloading schema: {str(e)}")
            return False
