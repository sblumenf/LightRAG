"""
Schema-aware knowledge graph implementation for LightRAG.

This module provides a schema-aware implementation of the BaseGraphStorage interface
that validates entities and relationships against a schema before adding them to the graph.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union, DefaultDict
from collections import defaultdict

from ..base import BaseGraphStorage
from ..schema.schema_validator import SchemaValidator
from ..schema.schema_loader import SchemaLoader
from ..utils import logger

# Set up logger
logger = logging.getLogger(__name__)


class SchemaAwareGraphStorage:
    """
    Schema-aware wrapper for BaseGraphStorage implementations.

    This class wraps any BaseGraphStorage implementation and adds schema validation
    for entities and relationships before they are added to the graph.
    """

    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        schema_path: str,
        new_type_confidence_threshold: float = 0.7,
        schema_match_confidence_threshold: float = 0.8,
    ):
        """
        Initialize the schema-aware graph storage.

        Args:
            graph_storage: The underlying graph storage implementation
            schema_path: Path to the schema JSON file
            new_type_confidence_threshold: Confidence threshold for accepting new entity/relationship types
            schema_match_confidence_threshold: Confidence threshold for schema matching
        """
        self.graph_storage = graph_storage
        self.schema_path = schema_path
        self.new_type_confidence_threshold = new_type_confidence_threshold
        self.schema_match_confidence_threshold = schema_match_confidence_threshold

        # Initialize schema validator and loader
        self.schema_validator = SchemaValidator(schema_path)
        self.schema_loader = SchemaLoader(schema_path)

        # Cache for entity and relationship types
        self._entity_types_cache = set()
        self._relationship_types_cache = set()

        # Load entity and relationship types from schema
        self._load_schema_types()

    def _load_schema_types(self) -> None:
        """
        Load entity and relationship types from the schema.
        """
        if self.schema_loader.is_schema_loaded():
            self._entity_types_cache = self.schema_loader.get_entity_types()
            self._relationship_types_cache = self.schema_loader.get_relationship_types()
            logger.info(f"Loaded {len(self._entity_types_cache)} entity types and {len(self._relationship_types_cache)} relationship types from schema")
        else:
            logger.warning("Failed to load schema types")

    async def create_schema_aware_graph(self, classified_chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Create a schema-aware knowledge graph from classified chunks.

        Args:
            classified_chunks: List of chunks with schema classifications in metadata

        Returns:
            Dict with statistics about the created graph elements
        """
        logger.info(f"Creating schema-aware graph from {len(classified_chunks)} classified chunks")

        if not classified_chunks:
            logger.info("No chunks provided for schema-aware creation")
            return {
                "nodes_created": 0,
                "relationships_created": 0,
                "tentative_entities_created": 0,
                "tentative_relationships_created": 0
            }

        # Statistics counters
        stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "tentative_entities_created": 0,
            "tentative_relationships_created": 0
        }

        # Process chunks in batches
        batch_size = 100  # Adjust as needed
        for i in range(0, len(classified_chunks), batch_size):
            batch = classified_chunks[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            num_batches = (len(classified_chunks) + batch_size - 1) // batch_size
            logger.info(f"Processing schema-aware batch {batch_num}/{num_batches} ({len(batch)} chunks)")

            # Extract entities and relationships from batch
            batch_stats = await self._process_batch(batch)

            # Update statistics
            for key in stats:
                stats[key] += batch_stats.get(key, 0)

        logger.info(f"Completed schema-aware creation. Stats: {stats}")
        return stats

    async def _process_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Process a batch of classified chunks.

        Args:
            batch: List of chunks with schema classifications in metadata

        Returns:
            Dict with statistics about the created graph elements
        """
        # Statistics counters
        stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "tentative_entities_created": 0,
            "tentative_relationships_created": 0
        }

        # Collect entities and relationships from batch
        chunk_nodes = []
        extracted_entities = []
        chunk_relationships = []
        extracted_relationships = []

        for chunk in batch:
            # Get classification from metadata
            classification = chunk.get("metadata", {}).get("schema_classification", {})

            # Process chunk node
            chunk_node = self._prepare_chunk_node(chunk, classification)
            if chunk_node:
                chunk_nodes.append(chunk_node)

            # Process extracted entities
            if "extracted_entities" in classification:
                for entity in classification["extracted_entities"]:
                    if self._validate_entity_confidence(entity):
                        entity["chunk_id"] = chunk.get("chunk_id")
                        extracted_entities.append(entity)

            # Process chunk relationships
            if "relationships" in chunk:
                for rel in chunk["relationships"]:
                    rel_data = {
                        "source_id": chunk.get("chunk_id"),
                        "target_id": rel.get("target"),
                        "type": rel.get("type"),
                        "properties": rel.get("properties", {}),
                        "is_new_type": False,
                        "confidence": 1.0
                    }
                    chunk_relationships.append(rel_data)

            # Process extracted relationships
            if "extracted_relationships" in classification:
                for rel in classification["extracted_relationships"]:
                    if self._validate_relationship_confidence(rel):
                        extracted_relationships.append(rel)

        # Create nodes and relationships
        nodes_result = await self._create_nodes(chunk_nodes, extracted_entities)
        relationships_result = await self._create_relationships(chunk_relationships, extracted_relationships)

        # Update statistics
        stats["nodes_created"] = nodes_result.get("nodes_created", 0)
        stats["tentative_entities_created"] = nodes_result.get("tentative_entities_created", 0)
        stats["relationships_created"] = relationships_result.get("relationships_created", 0)
        stats["tentative_relationships_created"] = relationships_result.get("tentative_relationships_created", 0)

        return stats

    def _prepare_chunk_node(self, chunk: Dict[str, Any], classification: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Prepare a chunk node with schema classification.

        Args:
            chunk: The chunk data
            classification: The schema classification data

        Returns:
            Prepared chunk node data or None if invalid
        """
        if not chunk.get("chunk_id"):
            return None

        # Extract classification info
        entity_type = classification.get("entity_type", "Chunk")
        properties = classification.get("properties", {})
        confidence = classification.get("confidence", 0.0)
        is_new_type = classification.get("is_new_type", False)

        # Validate entity type against schema
        is_valid = True
        is_tentative = False
        if entity_type != "Chunk" and entity_type not in self._entity_types_cache:
            if is_new_type and confidence >= self.new_type_confidence_threshold:
                # Mark as tentative entity type
                is_tentative = True
                entity_type = f"Tentative{entity_type}"
            else:
                # Invalid entity type, fallback to Chunk
                is_valid = False
                entity_type = "Chunk"
        elif entity_type != "Chunk" and confidence < self.schema_match_confidence_threshold:
            # Confidence too low, fallback to Chunk
            is_valid = False
            entity_type = "Chunk"

        # Prepare node data
        node_data = {
            "entity_id": chunk.get("chunk_id"),
            "text": chunk.get("content", ""),
            "source_doc": chunk.get("full_doc_id", ""),
            "parent_id": chunk.get("parent_id"),
            "level": chunk.get("level", 0),
            "position": chunk.get("chunk_order_index", 0),
            "importance": chunk.get("importance", 0.0),
            "embedding": chunk.get("embedding"),
            "entity_type": entity_type,
            "confidence": confidence,
            "is_tentative": is_tentative,
            "properties": properties
        }

        return node_data

    def _validate_entity_confidence(self, entity: Dict[str, Any]) -> bool:
        """
        Validate an entity's confidence against thresholds.

        Args:
            entity: The entity data

        Returns:
            True if the entity meets confidence thresholds, False otherwise
        """
        is_new_type = entity.get("is_new_type", False)
        confidence = entity.get("confidence", 0.0)
        threshold = self.new_type_confidence_threshold if is_new_type else self.schema_match_confidence_threshold

        # Check if entity has required fields and meets confidence threshold
        return (confidence >= threshold and
                entity.get("entity_id") is not None and
                entity.get("entity_type") is not None)

    def _validate_relationship_confidence(self, relationship: Dict[str, Any]) -> bool:
        """
        Validate a relationship's confidence against thresholds.

        Args:
            relationship: The relationship data

        Returns:
            True if the relationship meets confidence thresholds, False otherwise
        """
        is_new_type = relationship.get("is_new_type", False)
        confidence = relationship.get("confidence", 0.0)
        threshold = self.new_type_confidence_threshold if is_new_type else self.schema_match_confidence_threshold

        # Check if relationship has required fields and meets confidence threshold
        return (confidence >= threshold and
                relationship.get("source_id") is not None and
                relationship.get("target_id") is not None and
                (relationship.get("relationship_type") is not None or relationship.get("type") is not None))

    async def _create_nodes(self, chunk_nodes: List[Dict[str, Any]], extracted_entities: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Create nodes in the graph.

        Args:
            chunk_nodes: List of chunk nodes
            extracted_entities: List of extracted entities

        Returns:
            Dict with statistics about the created nodes
        """
        nodes_created = 0
        tentative_entities_created = 0

        # Create chunk nodes
        for node in chunk_nodes:
            entity_id = node.get("entity_id")
            if not entity_id:
                continue

            # Check if node is tentative
            is_tentative = node.get("is_tentative", False)

            # Upsert node
            await self.graph_storage.upsert_node(entity_id, node)
            nodes_created += 1

            if is_tentative:
                tentative_entities_created += 1

        # Create extracted entity nodes
        for entity in extracted_entities:
            entity_id = entity.get("entity_id")
            if not entity_id:
                continue

            # Check if entity is tentative
            is_tentative = entity.get("is_new_type", False)

            # Prepare node data
            node_data = {
                "entity_id": entity_id,
                "entity_type": entity.get("entity_type"),
                "text": entity.get("text", ""),
                "confidence": entity.get("confidence", 0.0),
                "is_tentative": is_tentative,
                "properties": entity.get("properties", {})
            }

            # Upsert node
            await self.graph_storage.upsert_node(entity_id, node_data)
            nodes_created += 1

            # Create relationship to source chunk
            chunk_id = entity.get("chunk_id")
            if chunk_id:
                await self.graph_storage.upsert_edge(
                    chunk_id,
                    entity_id,
                    {
                        "type": "CONTAINS_ENTITY",
                        "created_at": int(time.time())
                    }
                )

            if is_tentative:
                tentative_entities_created += 1

        return {
            "nodes_created": nodes_created,
            "tentative_entities_created": tentative_entities_created
        }

    async def _create_relationships(self, chunk_relationships: List[Dict[str, Any]], extracted_relationships: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Create relationships in the graph.

        Args:
            chunk_relationships: List of chunk relationships
            extracted_relationships: List of extracted relationships

        Returns:
            Dict with statistics about the created relationships
        """
        relationships_created = 0
        tentative_relationships_created = 0

        # Create chunk relationships
        for rel in chunk_relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            if not source_id or not target_id:
                continue

            # Check if relationship is tentative
            is_tentative = rel.get("is_new_type", False)

            # Prepare edge data
            edge_data = {
                "type": rel.get("type", "RELATED_TO"),
                "confidence": rel.get("confidence", 1.0),
                "is_tentative": is_tentative,
                "properties": rel.get("properties", {}),
                "created_at": int(time.time())
            }

            # Upsert edge
            await self.graph_storage.upsert_edge(source_id, target_id, edge_data)
            relationships_created += 1

            if is_tentative:
                tentative_relationships_created += 1

        # Create extracted relationships
        for rel in extracted_relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            if not source_id or not target_id:
                continue

            # Check if relationship is tentative
            is_tentative = rel.get("is_new_type", False)

            # Check for constraint violations
            constraint_violation = None
            if not is_tentative:
                # Validate relationship against schema
                source_type = await self._get_entity_type(source_id)
                target_type = await self._get_entity_type(target_id)
                relationship_type = rel.get("relationship_type")

                if source_type and target_type and relationship_type:
                    # Check if relationship type is valid for these entity types
                    valid_relationships = self.schema_loader.get_valid_relationships(source_type, target_type)
                    if relationship_type not in valid_relationships:
                        constraint_violation = f"Invalid relationship type '{relationship_type}' between '{source_type}' and '{target_type}'"
                        logger.warning(constraint_violation)

            # Prepare edge data
            edge_data = {
                "type": rel.get("relationship_type", "RELATED_TO"),
                "confidence": rel.get("confidence", 0.0),
                "is_tentative": is_tentative,
                "constraint_violation": constraint_violation,
                "properties": rel.get("properties", {}),
                "created_at": int(time.time())
            }

            # Upsert edge
            await self.graph_storage.upsert_edge(source_id, target_id, edge_data)
            relationships_created += 1

            if is_tentative:
                tentative_relationships_created += 1

        return {
            "relationships_created": relationships_created,
            "tentative_relationships_created": tentative_relationships_created
        }

    async def _get_entity_type(self, entity_id: str) -> Optional[str]:
        """
        Get the entity type for an entity.

        Args:
            entity_id: The entity ID

        Returns:
            The entity type or None if not found
        """
        node = await self.graph_storage.get_node(entity_id)
        if node:
            return node.get("entity_type")
        return None

    async def validate_entity(self, entity_type: str, properties: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate an entity against the schema.

        Args:
            entity_type: The entity type
            properties: The entity properties

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.schema_validator.validate_entity(entity_type, properties)

    async def validate_relationship(self, relationship_type: str, source_type: str, target_type: str, properties: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a relationship against the schema.

        Args:
            relationship_type: The relationship type
            source_type: The source entity type
            target_type: The target entity type
            properties: The relationship properties

        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.schema_validator.validate_relationship(relationship_type, source_type, target_type, properties)

    async def get_schema_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the schema and graph.

        Returns:
            Dict with schema statistics
        """
        # Get all nodes and edges
        nodes = await self.graph_storage.get_all_nodes()
        edges = await self.graph_storage.get_all_edges()

        # Count entity types
        entity_type_counts = defaultdict(int)
        tentative_entity_count = 0
        for node in nodes:
            entity_type = node.get("entity_type", "UNKNOWN")
            entity_type_counts[entity_type] += 1
            if node.get("is_tentative", False):
                tentative_entity_count += 1

        # Count relationship types
        relationship_type_counts = defaultdict(int)
        tentative_relationship_count = 0
        for edge in edges:
            rel_type = edge.get("type", "UNKNOWN")
            relationship_type_counts[rel_type] += 1
            if edge.get("is_tentative", False):
                tentative_relationship_count += 1

        # Get schema entity and relationship types
        schema_entity_types = self._entity_types_cache
        schema_relationship_types = self._relationship_types_cache

        # Calculate statistics
        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_type_counts": dict(entity_type_counts),
            "relationship_type_counts": dict(relationship_type_counts),
            "schema_entity_types": list(schema_entity_types),
            "schema_relationship_types": list(schema_relationship_types),
            "tentative_entity_count": tentative_entity_count,
            "tentative_relationship_count": tentative_relationship_count,
            "schema_coverage": {
                "entity_types": sum(1 for et in entity_type_counts if et in schema_entity_types) / len(schema_entity_types) if schema_entity_types else 0,
                "relationship_types": sum(1 for rt in relationship_type_counts if rt in schema_relationship_types) / len(schema_relationship_types) if schema_relationship_types else 0
            }
        }

        return stats

    async def get_schema_violations(self) -> List[Dict[str, Any]]:
        """
        Get all schema violations in the graph.

        Returns:
            List of schema violations
        """
        violations = []

        # Get all nodes and edges
        nodes = await self.graph_storage.get_all_nodes()
        edges = await self.graph_storage.get_all_edges()

        # Check node violations
        for node in nodes:
            entity_id = node.get("entity_id")
            entity_type = node.get("entity_type")
            properties = {k: v for k, v in node.items() if k not in ["entity_id", "entity_type", "is_tentative", "confidence"]}

            # Skip tentative entities
            if node.get("is_tentative", False):
                continue

            # Validate entity
            is_valid, error_msg = self.schema_validator.validate_entity(entity_type, properties)
            if not is_valid:
                violations.append({
                    "type": "entity_violation",
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "error": error_msg
                })

        # Check edge violations
        for edge in edges:
            source_id = edge.get("source_id")
            target_id = edge.get("target_id")
            rel_type = edge.get("type")
            properties = {k: v for k, v in edge.items() if k not in ["source_id", "target_id", "type", "is_tentative", "confidence"]}

            # Skip tentative relationships
            if edge.get("is_tentative", False):
                continue

            # Get source and target types
            source_node = await self.graph_storage.get_node(source_id)
            target_node = await self.graph_storage.get_node(target_id)

            if source_node and target_node:
                source_type = source_node.get("entity_type")
                target_type = target_node.get("entity_type")

                # Validate relationship
                is_valid, error_msg = self.schema_validator.validate_relationship(rel_type, source_type, target_type, properties)
                if not is_valid:
                    violations.append({
                        "type": "relationship_violation",
                        "source_id": source_id,
                        "target_id": target_id,
                        "relationship_type": rel_type,
                        "source_type": source_type,
                        "target_type": target_type,
                        "error": error_msg
                    })

        return violations

    async def fix_schema_violations(self, auto_fix: bool = False) -> Dict[str, int]:
        """
        Fix schema violations in the graph.

        Args:
            auto_fix: Whether to automatically fix violations

        Returns:
            Dict with statistics about fixed violations
        """
        violations = await self.get_schema_violations()

        fixed_count = 0
        unfixed_count = 0

        for violation in violations:
            if violation["type"] == "entity_violation":
                entity_id = violation["entity_id"]
                entity_type = violation["entity_type"]

                if auto_fix:
                    # Get the node
                    node = await self.graph_storage.get_node(entity_id)
                    if node:
                        # Mark as tentative
                        node["is_tentative"] = True
                        node["constraint_violation"] = violation["error"]

                        # Upsert the node
                        await self.graph_storage.upsert_node(entity_id, node)
                        fixed_count += 1
                    else:
                        unfixed_count += 1
                else:
                    unfixed_count += 1

            elif violation["type"] == "relationship_violation":
                source_id = violation["source_id"]
                target_id = violation["target_id"]

                if auto_fix:
                    # Get the edge
                    edge = await self.graph_storage.get_edge(source_id, target_id)
                    if edge:
                        # Mark as tentative
                        edge["is_tentative"] = True
                        edge["constraint_violation"] = violation["error"]

                        # Upsert the edge
                        await self.graph_storage.upsert_edge(source_id, target_id, edge)
                        fixed_count += 1
                    else:
                        unfixed_count += 1
                else:
                    unfixed_count += 1

        return {
            "total_violations": len(violations),
            "fixed_violations": fixed_count,
            "unfixed_violations": unfixed_count
        }

    # Delegate methods to underlying graph storage

    async def get_node(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get a node from the graph."""
        return await self.graph_storage.get_node(entity_id)

    async def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes from the graph."""
        return await self.graph_storage.get_all_nodes()

    async def upsert_node(self, entity_id: str, node_data: Dict[str, Any]) -> None:
        """Create or update a node in the graph with schema validation."""
        # Validate entity against schema if entity_type is provided
        if "entity_type" in node_data and node_data["entity_type"] != "UNKNOWN":
            entity_type = node_data["entity_type"]
            # Extract properties from node_data
            properties = {k: v for k, v in node_data.items() if k not in ["entity_id", "entity_type", "is_tentative", "confidence"]}

            # Validate entity
            is_valid, error_msg = self.schema_validator.validate_entity(entity_type, properties)
            if not is_valid:
                logger.warning(f"Invalid entity: {error_msg}")
                # Mark as tentative if validation fails
                node_data["is_tentative"] = True
                node_data["constraint_violation"] = error_msg

        # Upsert node
        await self.graph_storage.upsert_node(entity_id, node_data)

    async def delete_node(self, entity_id: str) -> None:
        """Delete a node from the graph."""
        await self.graph_storage.delete_node(entity_id)

    async def get_edge(self, src_id: str, tgt_id: str) -> Optional[Dict[str, Any]]:
        """Get an edge between two nodes."""
        return await self.graph_storage.get_edge(src_id, tgt_id)

    async def get_all_edges(self) -> List[Dict[str, Any]]:
        """Get all edges from the graph."""
        return await self.graph_storage.get_all_edges()

    async def upsert_edge(self, src_id: str, tgt_id: str, edge_data: Dict[str, Any]) -> None:
        """Create or update an edge in the graph with schema validation."""
        # Validate relationship against schema if type is provided
        if "type" in edge_data and edge_data["type"] != "UNKNOWN":
            relationship_type = edge_data["type"]

            # Get source and target types
            source_node = await self.graph_storage.get_node(src_id)
            target_node = await self.graph_storage.get_node(tgt_id)

            if source_node and target_node:
                source_type = source_node.get("entity_type")
                target_type = target_node.get("entity_type")

                # Extract properties from edge_data
                properties = {k: v for k, v in edge_data.items() if k not in ["type", "is_tentative", "confidence"]}

                # Validate relationship
                is_valid, error_msg = self.schema_validator.validate_relationship(relationship_type, source_type, target_type, properties)
                if not is_valid:
                    logger.warning(f"Invalid relationship: {error_msg}")
                    # Mark as tentative if validation fails
                    edge_data["is_tentative"] = True
                    edge_data["constraint_violation"] = error_msg

        # Upsert edge
        await self.graph_storage.upsert_edge(src_id, tgt_id, edge_data)

    async def delete_edge(self, src_id: str, tgt_id: str) -> None:
        """Delete an edge from the graph."""
        await self.graph_storage.delete_edge(src_id, tgt_id)

    async def get_neighbors(self, entity_id: str, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get neighbors of a node."""
        return await self.graph_storage.get_neighbors(entity_id, edge_type)

    async def get_incoming_neighbors(self, entity_id: str, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get incoming neighbors of a node."""
        return await self.graph_storage.get_incoming_neighbors(entity_id, edge_type)

    async def get_outgoing_neighbors(self, entity_id: str, edge_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get outgoing neighbors of a node."""
        return await self.graph_storage.get_outgoing_neighbors(entity_id, edge_type)
