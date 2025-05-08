"""
Schema-aware Neo4j implementation for LightRAG.

This module provides a schema-aware implementation of the Neo4JStorage class
that validates entities and relationships against a schema before adding them to the graph.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from collections import defaultdict

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError

from ..base import BaseGraphStorage
from ..schema.schema_validator import SchemaValidator
from ..schema.schema_loader import SchemaLoader
from ..kg.neo4j_impl import Neo4JStorage

logger = logging.getLogger(__name__)


class SchemaAwareNeo4JStorage(Neo4JStorage):
    """
    Schema-aware Neo4j implementation of BaseGraphStorage.

    This class extends Neo4JStorage and adds schema validation for entities and relationships
    before they are added to the graph.
    """

    def __init__(
        self,
        uri: str = None,
        username: str = None,
        password: str = None,
        database: str = None,
        schema_path: str = "schema.json",
        namespace: str = None,
        global_config: Dict[str, Any] = None,
        embedding_func: callable = None,
        new_type_confidence_threshold: float = 0.7,
        schema_match_confidence_threshold: float = 0.8,
    ):
        """
        Initialize the schema-aware Neo4j graph storage.

        Args:
            uri: Neo4j server URI (can be set via NEO4J_URI env var)
            username: Neo4j username (can be set via NEO4J_USERNAME env var)
            password: Neo4j password (can be set via NEO4J_PASSWORD env var)
            database: Neo4j database name (can be set via NEO4J_DATABASE env var)
            schema_path: Path to the schema JSON file
            namespace: Namespace for the graph storage
            global_config: Global configuration
            embedding_func: Function to generate embeddings
            new_type_confidence_threshold: Confidence threshold for accepting new entity/relationship types
            schema_match_confidence_threshold: Confidence threshold for schema matching
        """
        # Initialize parent class
        super().__init__(namespace, global_config, embedding_func)

        # Override connection parameters if provided
        if uri:
            self._URI = uri
        if username:
            self._USERNAME = username
        if password:
            self._PASSWORD = password
        if database:
            self._DATABASE = database

        # Initialize schema validator and loader
        self.schema_path = schema_path
        self.schema_validator = SchemaValidator(schema_path)
        self.schema_loader = SchemaLoader(schema_path)

        # Set confidence thresholds
        self.new_type_confidence_threshold = new_type_confidence_threshold
        self.schema_match_confidence_threshold = schema_match_confidence_threshold

        # Cache for entity and relationship types
        self._entity_types_cache = set(self.schema_validator.get_entity_types())
        self._relationship_types_cache = set(self.schema_validator.get_relationship_types())

        # Add default types if not in schema
        if "UNKNOWN" not in self._entity_types_cache:
            self._entity_types_cache.add("UNKNOWN")
        if "RELATED_TO" not in self._relationship_types_cache:
            self._relationship_types_cache.add("RELATED_TO")

        logger.info(f"Initialized schema-aware Neo4j storage with {len(self._entity_types_cache)} entity types and {len(self._relationship_types_cache)} relationship types")

    async def upsert_node(self, entity_id: str, node_data: Dict[str, Any]) -> None:
        """
        Create or update a node in the graph with schema validation.

        Args:
            entity_id: Entity ID to upsert
            node_data: Node data to upsert
        """
        # Validate entity against schema if entity_type is provided
        if "entity_type" in node_data and node_data["entity_type"] != "UNKNOWN":
            entity_type = node_data["entity_type"]
            # Extract properties from node_data
            properties = {k: v for k, v in node_data.items() if k not in ["entity_id", "entity_type"]}

            # Add properties from the properties field if it exists
            if "properties" in node_data and isinstance(node_data["properties"], dict):
                properties.update(node_data["properties"])

            is_valid, error_msg = self.schema_validator.validate_entity(entity_type, properties)
            if not is_valid:
                logger.warning(f"Invalid entity: {error_msg}")
                # Mark as tentative if validation fails
                node_data["is_tentative"] = True
                node_data["constraint_violation"] = error_msg

                # Fall back to UNKNOWN type if validation fails and no tentative flag
                if not node_data.get("is_tentative", False):
                    node_data["entity_type"] = "UNKNOWN"

        # Ensure entity_id is in the node data
        node_data["entity_id"] = entity_id

        # Create Cypher parameter dict
        params = {"entity_id": entity_id, "data": node_data}

        # Build dynamic labels based on entity_type
        entity_type = node_data.get("entity_type", "UNKNOWN")
        is_tentative = node_data.get("is_tentative", False)

        # Add Tentative prefix for tentative entities
        if is_tentative and not entity_type.startswith("Tentative"):
            entity_type = f"Tentative{entity_type}"

        labels = ["Entity", entity_type]
        labels_str = ":".join(labels)

        # Build dynamic properties
        props = []
        for key, value in node_data.items():
            if key == "properties" and isinstance(value, dict):
                # Handle nested properties
                for prop_key, prop_value in value.items():
                    props.append(f"n.{prop_key} = $data.properties.{prop_key}")
            else:
                props.append(f"n.{key} = $data.{key}")

        props_str = ", ".join(props)

        query = f"""
        MERGE (n:{labels_str} {{entity_id: $entity_id}})
        ON CREATE SET {props_str}
        ON MATCH SET {props_str}
        """

        async with self._driver.session(database=self._DATABASE) as session:
            await session.run(query, **params)

    async def upsert_edge(self, src_id: str, tgt_id: str, edge_data: Dict[str, Any]) -> None:
        """
        Create or update an edge in the graph with schema validation.

        Args:
            src_id: Source entity ID
            tgt_id: Target entity ID
            edge_data: Edge data to upsert
        """
        # Get relationship type
        rel_type = edge_data.get("type", "RELATED_TO")

        # Validate relationship against schema
        if rel_type != "RELATED_TO":
            # Get source and target types
            source_node = await self.get_node(src_id)
            target_node = await self.get_node(tgt_id)

            if source_node and target_node:
                source_type = source_node.get("entity_type", "UNKNOWN")
                target_type = target_node.get("entity_type", "UNKNOWN")

                # Extract properties from edge_data
                properties = {k: v for k, v in edge_data.items() if k not in ["type", "is_tentative", "confidence"]}

                # Validate relationship
                is_valid, error_msg = self.schema_validator.validate_relationship(rel_type, source_type, target_type, properties)
                if not is_valid:
                    logger.warning(f"Invalid relationship: {error_msg}")
                    # Mark as tentative if validation fails
                    edge_data["is_tentative"] = True
                    edge_data["constraint_violation"] = error_msg

                    # Fall back to RELATED_TO type if validation fails and no tentative flag
                    if not edge_data.get("is_tentative", False):
                        rel_type = "RELATED_TO"
                        edge_data["type"] = rel_type

        # Check if relationship is tentative
        is_tentative = edge_data.get("is_tentative", False)

        # Add Tentative prefix for tentative relationships
        if is_tentative and not rel_type.startswith("Tentative"):
            rel_type = f"Tentative{rel_type}"
            edge_data["type"] = rel_type

        # Create Cypher parameter dict
        params = {
            "src_id": src_id,
            "tgt_id": tgt_id,
            "data": edge_data
        }

        # Build dynamic properties
        props = []
        for key, value in edge_data.items():
            if key == "properties" and isinstance(value, dict):
                # Handle nested properties
                for prop_key, prop_value in value.items():
                    props.append(f"r.{prop_key} = $data.properties.{prop_key}")
            else:
                props.append(f"r.{key} = $data.{key}")

        props_str = ", ".join(props)

        query = f"""
        MATCH (src:Entity {{entity_id: $src_id}})
        MATCH (tgt:Entity {{entity_id: $tgt_id}})
        MERGE (src)-[r:{rel_type}]->(tgt)
        ON CREATE SET {props_str}
        ON MATCH SET {props_str}
        """

        async with self._driver.session(database=self._DATABASE) as session:
            await session.run(query, **params)

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
            "entity_type": entity_type,
            "text": chunk.get("content", ""),
            "source_doc": chunk.get("full_doc_id", ""),
            "parent_id": chunk.get("parent_id"),
            "level": chunk.get("level", 0),
            "position": chunk.get("chunk_order_index", 0),
            "importance": chunk.get("importance", 0.0),
            "embedding": chunk.get("embedding"),
            "confidence": confidence,
            "is_tentative": is_tentative,
            "properties": properties,
            "created_at": int(time.time())
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

        return confidence >= threshold and entity.get("entity_id") and entity.get("entity_type")

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

        return (confidence >= threshold and
                relationship.get("source_id") and
                relationship.get("target_id") and
                relationship.get("relationship_type"))

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
            chunk_id = node.get("entity_id")
            if not chunk_id:
                continue

            # Check if node is tentative
            is_tentative = node.get("is_tentative", False)

            # Upsert node
            await self.upsert_node(chunk_id, node)
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
                "properties": entity.get("properties", {}),
                "created_at": int(time.time())
            }

            # Upsert node
            await self.upsert_node(entity_id, node_data)
            nodes_created += 1

            # Create relationship to source chunk
            chunk_id = entity.get("chunk_id")
            if chunk_id:
                await self.upsert_edge(
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
            await self.upsert_edge(source_id, target_id, edge_data)
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
            await self.upsert_edge(source_id, target_id, edge_data)
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
        node = await self.get_node(entity_id)
        if node:
            return node.get("entity_type")
        return None

    async def get_schema_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the schema and graph.

        Returns:
            Dict with schema statistics
        """
        # Get all nodes and edges
        nodes = await self.get_all_nodes()
        edges = await self.get_all_edges()

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
        nodes = await self.get_all_nodes()
        edges = await self.get_all_edges()

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
            source_node = await self.get_node(source_id)
            target_node = await self.get_node(target_id)

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
                    node = await self.get_node(entity_id)
                    if node:
                        # Mark as tentative
                        node["is_tentative"] = True
                        node["constraint_violation"] = violation["error"]

                        # Upsert the node
                        await self.upsert_node(entity_id, node)
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
                    edge = await self.get_edge(source_id, target_id)
                    if edge:
                        # Mark as tentative
                        edge["is_tentative"] = True
                        edge["constraint_violation"] = violation["error"]

                        # Upsert the edge
                        await self.upsert_edge(source_id, target_id, edge)
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

    async def get_tentative_entities(self) -> List[Dict[str, Any]]:
        """
        Get all tentative entities in the graph.

        Returns:
            List of tentative entities
        """
        query = """
        MATCH (n:Entity)
        WHERE n.is_tentative = true
        RETURN n
        """

        tentative_entities = []

        async with self._driver.session(database=self._DATABASE) as session:
            result = await session.run(query)
            records = await result.fetch()

            for record in records:
                node = record["n"]
                tentative_entities.append(dict(node))

        return tentative_entities

    async def get_tentative_relationships(self) -> List[Dict[str, Any]]:
        """
        Get all tentative relationships in the graph.

        Returns:
            List of tentative relationships
        """
        query = """
        MATCH (src:Entity)-[r]->(tgt:Entity)
        WHERE r.is_tentative = true
        RETURN src.entity_id AS source_id, tgt.entity_id AS target_id, type(r) AS type, r
        """

        tentative_relationships = []

        async with self._driver.session(database=self._DATABASE) as session:
            result = await session.run(query)
            records = await result.fetch()

            for record in records:
                rel = {
                    "source_id": record["source_id"],
                    "target_id": record["target_id"],
                    "type": record["type"],
                    **dict(record["r"])
                }
                tentative_relationships.append(rel)

        return tentative_relationships

    async def promote_tentative_entity(self, entity_id: str) -> bool:
        """
        Promote a tentative entity to a regular entity.

        Args:
            entity_id: The entity ID to promote

        Returns:
            True if the entity was promoted, False otherwise
        """
        # Get the node
        node = await self.get_node(entity_id)
        if not node or not node.get("is_tentative", False):
            return False

        # Remove tentative flag
        node["is_tentative"] = False
        if "constraint_violation" in node:
            del node["constraint_violation"]

        # Update entity type if it has a Tentative prefix
        entity_type = node.get("entity_type", "UNKNOWN")
        if entity_type.startswith("Tentative"):
            node["entity_type"] = entity_type[9:]  # Remove "Tentative" prefix

        # Upsert the node
        await self.upsert_node(entity_id, node)

        # Add entity type to schema if it's not already there
        if node["entity_type"] not in self._entity_types_cache:
            self._entity_types_cache.add(node["entity_type"])

        return True

    async def promote_tentative_relationship(self, source_id: str, target_id: str) -> bool:
        """
        Promote a tentative relationship to a regular relationship.

        Args:
            source_id: The source entity ID
            target_id: The target entity ID

        Returns:
            True if the relationship was promoted, False otherwise
        """
        # Get the edge
        edge = await self.get_edge(source_id, target_id)
        if not edge or not edge.get("is_tentative", False):
            return False

        # Remove tentative flag
        edge["is_tentative"] = False
        if "constraint_violation" in edge:
            del edge["constraint_violation"]

        # Update relationship type if it has a Tentative prefix
        rel_type = edge.get("type", "RELATED_TO")
        if rel_type.startswith("Tentative"):
            edge["type"] = rel_type[9:]  # Remove "Tentative" prefix

        # Upsert the edge
        await self.upsert_edge(source_id, target_id, edge)

        # Add relationship type to schema if it's not already there
        if edge["type"] not in self._relationship_types_cache:
            self._relationship_types_cache.add(edge["type"])

        return True