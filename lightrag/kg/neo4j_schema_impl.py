"""
Neo4j implementation with schema validation for LightRAG.

This module provides a Neo4j implementation of the BaseGraphStorage interface with schema validation.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set, Union

from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError

from ..base import BaseGraphStorage
from ..schema.schema_validator import SchemaValidator
from ..kg.neo4j_impl import Neo4JStorage

logger = logging.getLogger(__name__)


class Neo4JSchemaStorage(Neo4JStorage):
    """
    Neo4j implementation of BaseGraphStorage with schema validation.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        schema_path: str = "schema.json",
        max_connection_pool_size: int = 50,
        max_transaction_retry_time: float = 30.0,
    ):
        """
        Initialize the Neo4j graph storage with schema validation.

        Args:
            uri: Neo4j server URI
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name
            schema_path: Path to the schema JSON file
            max_connection_pool_size: Maximum connection pool size
            max_transaction_retry_time: Maximum transaction retry time in seconds
        """
        # Set up environment variables for Neo4JStorage
        import os
        os.environ["NEO4J_URI"] = uri
        os.environ["NEO4J_USERNAME"] = username
        os.environ["NEO4J_PASSWORD"] = password
        os.environ["NEO4J_DATABASE"] = database
        os.environ["NEO4J_MAX_CONNECTION_POOL_SIZE"] = str(max_connection_pool_size)
        os.environ["NEO4J_MAX_TRANSACTION_RETRY_TIME"] = str(max_transaction_retry_time)
        
        # Initialize parent class with a namespace based on the database name
        super().__init__(
            namespace=database,
            global_config={},
            embedding_func=None,
        )
        
        # Initialize schema validator
        self.schema_validator = SchemaValidator(schema_path)

    async def has_node(self, entity_id: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            entity_id: Entity ID to check

        Returns:
            True if the node exists, False otherwise
        """
        query = """
        MATCH (n {entity_id: $entity_id})
        RETURN count(n) > 0 AS exists
        """
        
        async with self._driver.session(database=self._DATABASE) as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()
            return record and record["exists"]

    async def get_node(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node from the graph.

        Args:
            entity_id: Entity ID to get

        Returns:
            Node data or None if not found
        """
        query = """
        MATCH (n {entity_id: $entity_id})
        RETURN n
        """
        
        async with self._driver.session(database=self._DATABASE) as session:
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()
            if record:
                node = record["n"]
                return dict(node)
            return None

    async def upsert_node(self, entity_id: str, node_data: Dict[str, Any]) -> None:
        """
        Create or update a node in the graph.

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
                # Fall back to UNKNOWN type if validation fails
                node_data["entity_type"] = "UNKNOWN"
        
        # Ensure entity_id is in the node data
        node_data["entity_id"] = entity_id
        
        # Create Cypher parameter dict
        params = {"entity_id": entity_id, "data": node_data}
        
        # Build dynamic labels based on entity_type
        entity_type = node_data.get("entity_type", "UNKNOWN")
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

    async def delete_node(self, entity_id: str) -> None:
        """
        Delete a node from the graph.

        Args:
            entity_id: Entity ID to delete
        """
        query = """
        MATCH (n {entity_id: $entity_id})
        DETACH DELETE n
        """
        
        async with self._driver.session(database=self._DATABASE) as session:
            await session.run(query, entity_id=entity_id)

    async def has_edge(self, src_id: str, tgt_id: str) -> bool:
        """
        Check if an edge exists between two nodes.

        Args:
            src_id: Source entity ID
            tgt_id: Target entity ID

        Returns:
            True if the edge exists, False otherwise
        """
        query = """
        MATCH (src {entity_id: $src_id})-[r]->(tgt {entity_id: $tgt_id})
        RETURN count(r) > 0 AS exists
        """
        
        async with self._driver.session(database=self._DATABASE) as session:
            result = await session.run(query, src_id=src_id, tgt_id=tgt_id)
            record = await result.single()
            return record and record["exists"]

    async def get_edge(self, src_id: str, tgt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an edge between two nodes.

        Args:
            src_id: Source entity ID
            tgt_id: Target entity ID

        Returns:
            Edge data or None if not found
        """
        query = """
        MATCH (src {entity_id: $src_id})-[r]->(tgt {entity_id: $tgt_id})
        RETURN r
        """
        
        async with self._driver.session(database=self._DATABASE) as session:
            result = await session.run(query, src_id=src_id, tgt_id=tgt_id)
            record = await result.single()
            if record:
                edge = record["r"]
                return dict(edge)
            return None
