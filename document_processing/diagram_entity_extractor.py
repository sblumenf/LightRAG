"""
Diagram entity extraction module for LightRAG.

This module provides functionality to extract entities and relationships from diagram descriptions
for integration into the knowledge graph.
"""
import logging
import os
import json
import time
import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)

class DiagramEntityExtractor:
    """
    Extract entities and relationships from diagram descriptions for knowledge graph integration.
    """
    
    def __init__(self, schema_validator, llm_func, config=None):
        """
        Initialize the diagram entity extractor.
        
        Args:
            schema_validator: SchemaValidator instance with loaded schema
            llm_func: Async function to call the LLM
            config: Optional configuration dictionary
        """
        self.schema_validator = schema_validator
        self.llm_func = llm_func
        self.config = config or {}
        
        # Configuration settings with defaults
        self.confidence_threshold = self.config.get('diagram_entity_confidence', 0.7)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)
        self.enable_caching = self.config.get('enable_diagram_entity_cache', True)
        self.cache_dir = self.config.get('diagram_entity_cache_dir', 
                                         os.path.join(os.path.expanduser('~'), '.lightrag', 'diagram_entity_cache'))
        
        # Initialize cache if enabled
        if self.enable_caching:
            self._init_cache()
            
    def _init_cache(self):
        """Initialize the cache for diagram entities and relationships."""
        try:
            # Create cache directory if it doesn't exist
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

            # Cache file paths for entities and relationships
            self.entity_cache_file = os.path.join(self.cache_dir, 'diagram_entities.pkl')
            self.relationship_cache_file = os.path.join(self.cache_dir, 'diagram_relationships.pkl')
            
            # Initialize cache dictionaries
            self.entity_cache = {}
            self.relationship_cache = {}
            
            # Load existing entity cache if available
            if os.path.exists(self.entity_cache_file):
                try:
                    with open(self.entity_cache_file, 'rb') as f:
                        loaded_cache = pickle.load(f)
                        if isinstance(loaded_cache, dict):
                            self.entity_cache = loaded_cache
                            logger.info(f"Loaded {len(self.entity_cache)} diagram entities from cache")
                        else:
                            logger.warning("Entity cache file exists but contains invalid data. Using empty cache.")
                except Exception as e:
                    logger.warning(f"Error loading entity cache file: {str(e)}. Using empty cache.")
            
            # Load existing relationship cache if available
            if os.path.exists(self.relationship_cache_file):
                try:
                    with open(self.relationship_cache_file, 'rb') as f:
                        loaded_cache = pickle.load(f)
                        if isinstance(loaded_cache, dict):
                            self.relationship_cache = loaded_cache
                            logger.info(f"Loaded {len(self.relationship_cache)} diagram relationships from cache")
                        else:
                            logger.warning("Relationship cache file exists but contains invalid data. Using empty cache.")
                except Exception as e:
                    logger.warning(f"Error loading relationship cache file: {str(e)}. Using empty cache.")
                    
        except Exception as e:
            logger.warning(f"Error initializing cache: {str(e)}. Caching will be disabled.")
            self.enable_caching = False
            self.entity_cache = {}
            self.relationship_cache = {}
            
    def _save_entity_cache(self):
        """Save the entity cache to disk."""
        if not self.enable_caching:
            return
            
        try:
            with open(self.entity_cache_file, 'wb') as f:
                pickle.dump(self.entity_cache, f)
            logger.debug(f"Saved {len(self.entity_cache)} entries to diagram entity cache")
        except Exception as e:
            logger.warning(f"Error saving entity cache: {str(e)}")
            
    def _save_relationship_cache(self):
        """Save the relationship cache to disk."""
        if not self.enable_caching:
            return
            
        try:
            with open(self.relationship_cache_file, 'wb') as f:
                pickle.dump(self.relationship_cache, f)
            logger.debug(f"Saved {len(self.relationship_cache)} entries to diagram relationship cache")
        except Exception as e:
            logger.warning(f"Error saving relationship cache: {str(e)}")
            
    def get_entities_from_cache(self, diagram_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached entities for a diagram if available.
        
        Args:
            diagram_id: The unique ID of the diagram
            
        Returns:
            List of entity dictionaries or None if not cached
        """
        if not self.enable_caching:
            return None
            
        cached_data = self.entity_cache.get(diagram_id)
        if cached_data and isinstance(cached_data, dict):
            # Check if the cache is still valid
            cache_expiry = self.config.get('diagram_entity_cache_expiry', 3600 * 24 * 7)  # Default: 1 week
            if time.time() - cached_data.get('timestamp', 0) < cache_expiry:
                return cached_data.get('entities', [])
        return None
        
    def get_relationships_from_cache(self, diagram_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached relationships for a diagram if available.
        
        Args:
            diagram_id: The unique ID of the diagram
            
        Returns:
            List of relationship dictionaries or None if not cached
        """
        if not self.enable_caching:
            return None
            
        cached_data = self.relationship_cache.get(diagram_id)
        if cached_data and isinstance(cached_data, dict):
            # Check if the cache is still valid
            cache_expiry = self.config.get('diagram_entity_cache_expiry', 3600 * 24 * 7)  # Default: 1 week
            if time.time() - cached_data.get('timestamp', 0) < cache_expiry:
                return cached_data.get('relationships', [])
        return None
        
    def save_entities_to_cache(self, diagram_id: str, entities: List[Dict[str, Any]]):
        """
        Save extracted entities to cache.
        
        Args:
            diagram_id: The unique ID of the diagram
            entities: List of entity dictionaries
        """
        if not self.enable_caching:
            return
            
        self.entity_cache[diagram_id] = {
            'entities': entities,
            'timestamp': time.time()
        }
        self._save_entity_cache()
        
    def save_relationships_to_cache(self, diagram_id: str, relationships: List[Dict[str, Any]]):
        """
        Save extracted relationships to cache.
        
        Args:
            diagram_id: The unique ID of the diagram
            relationships: List of relationship dictionaries
        """
        if not self.enable_caching:
            return
            
        self.relationship_cache[diagram_id] = {
            'relationships': relationships,
            'timestamp': time.time()
        }
        self._save_relationship_cache()
        
    def clear_cache(self):
        """Clear all diagram entity and relationship caches."""
        if self.enable_caching:
            try:
                if os.path.exists(self.entity_cache_file):
                    os.remove(self.entity_cache_file)
                if os.path.exists(self.relationship_cache_file):
                    os.remove(self.relationship_cache_file)
                logger.info("Diagram entity and relationship caches cleared")
                self.entity_cache = {}
                self.relationship_cache = {}
            except Exception as e:
                logger.warning(f"Error clearing cache files: {str(e)}")
                
    async def _call_llm_with_retry(self, prompt: str) -> str:
        """
        Call the LLM function with retry logic.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM response text
            
        Raises:
            Exception: If all retries fail
        """
        import asyncio
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.llm_func(prompt)
                return response
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call attempt {attempt+1}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    
        # If we get here, all retries failed
        raise Exception(f"All {self.max_retries} LLM call attempts failed. Last error: {str(last_error)}")
        
    def _build_entity_extraction_prompt(self, diagram_data: Dict[str, Any]) -> str:
        """
        Build a prompt for extracting entities from a diagram description.
        
        Args:
            diagram_data: Dictionary containing diagram information
            
        Returns:
            Formatted prompt string
        """
        # Get diagram information
        description = diagram_data.get('description', '')
        caption = diagram_data.get('caption', '')
        diagram_type = diagram_data.get('diagram_type', 'general')
        
        # Get entity types from schema
        entity_types = self.schema_validator.get_entity_types()
        entity_type_details = []
        
        # Add details for each entity type
        for entity_type in entity_types:
            properties = self.schema_validator.get_entity_properties(entity_type)
            prop_details = []
            for prop in properties:
                prop_name = prop.get('name', '')
                prop_type = prop.get('type', 'string')
                required = prop.get('required', False)
                prop_details.append(f"- {prop_name} ({prop_type}{', required' if required else ''})")
                
            entity_type_details.append(f"{entity_type}:\n" + "\n".join(prop_details))
        
        # Format the prompt template
        prompt = f"""
You are an expert in extracting structured entities from diagram descriptions.

TASK:
Extract named entities from the following diagram description according to the schema provided.

DIAGRAM DESCRIPTION:
```
{description}
```

DIAGRAM CAPTION:
{caption}

DIAGRAM TYPE:
{diagram_type}

SCHEMA ENTITY TYPES:
{chr(10).join(entity_type_details)}

INSTRUCTIONS:
1. Identify distinct entities mentioned in the diagram description
2. Classify each entity according to the schema entity types
3. Extract relevant properties for each entity
4. For each entity, provide:
   - entity_name: A unique name for the entity
   - entity_type: The schema type that best matches
   - properties: Key-value pairs of relevant properties
   - description: A brief description of the entity

FORMAT YOUR RESPONSE AS JSON:
{{
  "entities": [
    {{
      "entity_name": "EntityName1",
      "entity_type": "SchemaType1",
      "properties": {{
        "property1": "value1",
        "property2": "value2"
      }},
      "description": "Brief description of entity"
    }},
    ...
  ]
}}

IMPORTANT:
- Focus on major components only
- Use consistent entity naming
- Only use entity types from the provided schema
- If you're uncertain about an entity's type, use the most probable one
- Return ONLY the JSON object
"""
        return prompt
        
    def _parse_entity_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response for entity extraction.
        
        Args:
            response_text: The raw response from the LLM
            
        Returns:
            List of extracted entity dictionaries
        """
        try:
            # First, try to find JSON block in response
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                # Parse the JSON
                data = json.loads(json_str)
                entities = data.get('entities', [])
                return entities
            else:
                logger.warning("No JSON object found in LLM response")
                return []
        except Exception as e:
            logger.error(f"Error parsing entity response: {str(e)}")
            logger.debug(f"Failed to parse response: {response_text[:500]}...")
            return []
            
    async def extract_entities_from_diagram(self, diagram_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from a diagram description.
        
        Args:
            diagram_data: Dictionary containing diagram information
                including description, caption, and metadata
                
        Returns:
            List of extracted entities with schema types
        """
        diagram_id = diagram_data.get('diagram_id', 'unknown')
        
        # Check cache for entities
        if self.enable_caching and diagram_id != 'unknown':
            cached_entities = self.get_entities_from_cache(diagram_id)
            if cached_entities:
                return cached_entities
        
        # Check if we have a description to work with
        description = diagram_data.get('description')
        if not description:
            logger.warning(f"No description found for diagram {diagram_id}")
            return []
        
        # Build prompt for entity extraction
        prompt = self._build_entity_extraction_prompt(diagram_data)
        
        # Call LLM to extract entities
        try:
            response_text = await self._call_llm_with_retry(prompt)
            
            # Parse the LLM response
            extracted_entities = self._parse_entity_response(response_text)
            
            # Validate entities against schema
            validated_entities = []
            for entity in extracted_entities:
                entity_type = entity.get('entity_type')
                properties = entity.get('properties', {})
                
                # Validate entity against schema
                is_valid, error_msg = self.schema_validator.validate_entity(
                    entity_type, properties
                )
                
                if is_valid:
                    # Add source information to the entity
                    entity['source_id'] = diagram_id
                    entity['extraction_method'] = 'diagram'
                    validated_entities.append(entity)
                else:
                    logger.warning(f"Invalid entity from diagram {diagram_id}: {error_msg}")
            
            # Cache validated entities
            if self.enable_caching and diagram_id != 'unknown':
                self.save_entities_to_cache(diagram_id, validated_entities)
            
            return validated_entities
            
        except Exception as e:
            logger.error(f"Error extracting entities from diagram {diagram_id}: {str(e)}")
            return []
    
    def _build_relationship_extraction_prompt(self, diagram_data: Dict[str, Any], entities: List[Dict[str, Any]]) -> str:
        """
        Build a prompt for extracting relationships from a diagram description.
        
        Args:
            diagram_data: Dictionary containing diagram information
            entities: List of extracted entities
            
        Returns:
            Formatted prompt string
        """
        # Get diagram information
        description = diagram_data.get('description', '')
        caption = diagram_data.get('caption', '')
        diagram_type = diagram_data.get('diagram_type', 'general')
        
        # Get relationship types from schema
        relationship_types = self.schema_validator.get_relationship_types()
        relationship_definitions = []
        
        # Add details for each relationship type
        for rel_type in relationship_types:
            rel_def = self.schema_validator.relationship_types.get(rel_type, {})
            source_type = rel_def.get('source', '')
            target_type = rel_def.get('target', '')
            properties = self.schema_validator.get_relationship_properties(rel_type)
            
            prop_details = []
            for prop in properties:
                prop_name = prop.get('name', '')
                prop_type = prop.get('type', 'string')
                required = prop.get('required', False)
                prop_details.append(f"- {prop_name} ({prop_type}{', required' if required else ''})")
                
            relationship_definitions.append(
                f"{rel_type}: {source_type} -> {target_type}\n" + 
                "\n".join(prop_details) if prop_details else ""
            )
        
        # Format entities as JSON for the prompt
        entities_json = json.dumps(entities, indent=2)
        
        # Format the prompt template
        prompt = f"""
You are an expert in extracting structured relationships from diagram descriptions.

TASK:
Extract relationships between entities from the following diagram description according to the schema provided.

DIAGRAM DESCRIPTION:
```
{description}
```

DIAGRAM CAPTION:
{caption}

DIAGRAM TYPE:
{diagram_type}

ENTITIES:
{entities_json}

RELATIONSHIP TYPES:
{", ".join(relationship_types)}

RELATIONSHIP DEFINITIONS:
{chr(10).join(relationship_definitions)}

INSTRUCTIONS:
1. Identify relationships between the extracted entities
2. Classify each relationship according to the schema relationship types
3. For each relationship, provide:
   - source: The name of the source entity
   - target: The name of the target entity
   - type: The schema relationship type
   - description: A brief description of the relationship
   - properties: Any relevant properties for the relationship

FORMAT YOUR RESPONSE AS JSON:
{{
  "relationships": [
    {{
      "source": "EntityName1",
      "target": "EntityName2",
      "type": "RELATIONSHIP_TYPE",
      "description": "Brief description of relationship",
      "properties": {{
        "property1": "value1"
      }}
    }},
    ...
  ]
}}

IMPORTANT:
- Only create relationships between entities in the provided entity list
- Only use relationship types from the provided schema
- Focus on semantic relationships, not just visual connections
- Return ONLY the JSON object
"""
        return prompt
        
    def _parse_relationship_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response for relationship extraction.
        
        Args:
            response_text: The raw response from the LLM
            
        Returns:
            List of extracted relationship dictionaries
        """
        try:
            # First, try to find JSON block in response
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                json_str = json_match.group(1)
                # Parse the JSON
                data = json.loads(json_str)
                relationships = data.get('relationships', [])
                return relationships
            else:
                logger.warning("No JSON object found in LLM response")
                return []
        except Exception as e:
            logger.error(f"Error parsing relationship response: {str(e)}")
            logger.debug(f"Failed to parse response: {response_text[:500]}...")
            return []
            
    async def extract_relationships_from_diagram(self, diagram_data: Dict[str, Any], entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities in a diagram.
        
        Args:
            diagram_data: Dictionary containing diagram information
            entities: List of extracted entities
            
        Returns:
            List of extracted relationships between entities
        """
        diagram_id = diagram_data.get('diagram_id', 'unknown')
        
        # Check cache for relationships
        if self.enable_caching and diagram_id != 'unknown':
            cached_relationships = self.get_relationships_from_cache(diagram_id)
            if cached_relationships:
                return cached_relationships
        
        # If we have no entities or no description, return empty list
        if not entities or 'description' not in diagram_data:
            logger.warning(f"No entities or description for relationship extraction in diagram {diagram_id}")
            return []
        
        # Build prompt for relationship extraction
        prompt = self._build_relationship_extraction_prompt(diagram_data, entities)
        
        # Call LLM to extract relationships
        try:
            response_text = await self._call_llm_with_retry(prompt)
            
            # Parse the LLM response
            extracted_relationships = self._parse_relationship_response(response_text)
            
            # Validate relationships against schema
            validated_relationships = []
            entity_names = {entity['entity_name']: entity for entity in entities}
            
            for relationship in extracted_relationships:
                source_name = relationship.get('source')
                target_name = relationship.get('target')
                rel_type = relationship.get('type')
                properties = relationship.get('properties', {})
                
                # Check if source and target entities exist
                if source_name not in entity_names:
                    logger.warning(f"Source entity '{source_name}' not found in extracted entities")
                    continue
                    
                if target_name not in entity_names:
                    logger.warning(f"Target entity '{target_name}' not found in extracted entities")
                    continue
                
                source_entity = entity_names[source_name]
                target_entity = entity_names[target_name]
                source_type = source_entity.get('entity_type')
                target_type = target_entity.get('entity_type')
                
                # Validate relationship against schema
                is_valid, error_msg = self.schema_validator.validate_relationship(
                    rel_type, source_type, target_type, properties
                )
                
                if is_valid:
                    # Add source information to the relationship
                    relationship['source_id'] = diagram_id
                    relationship['extraction_method'] = 'diagram'
                    relationship['source_entity_id'] = source_entity.get('entity_id', source_name)
                    relationship['target_entity_id'] = target_entity.get('entity_id', target_name)
                    validated_relationships.append(relationship)
                else:
                    logger.warning(f"Invalid relationship from diagram {diagram_id}: {error_msg}")
            
            # Cache validated relationships
            if self.enable_caching and diagram_id != 'unknown':
                self.save_relationships_to_cache(diagram_id, validated_relationships)
            
            return validated_relationships
            
        except Exception as e:
            logger.error(f"Error extracting relationships from diagram {diagram_id}: {str(e)}")
            return []