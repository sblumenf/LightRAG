"""
Synchronization-aware Neo4j implementation for LightRAG.

This module provides a Neo4j implementation that automatically synchronizes
graph changes with vector storage to maintain consistency.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union

from ..base import BaseGraphStorage, BaseVectorStorage
from ..kg.neo4j_impl import Neo4JStorage
from ..kg.kg_index_sync import KGIndexSynchronizer

logger = logging.getLogger(__name__)


class SyncAwareNeo4JStorage(Neo4JStorage):
    """
    Neo4j implementation that automatically synchronizes with vector storage.

    This class extends Neo4JStorage and adds automatic synchronization with
    vector storage to ensure that vector indices stay up-to-date with changes
    in the graph.
    """

    def __init__(
        self,
        namespace: str,
        global_config: Dict[str, Any],
        embedding_func: callable,
        entity_vector_storage: Optional[BaseVectorStorage] = None,
        relationship_vector_storage: Optional[BaseVectorStorage] = None,
        auto_sync: bool = True,
        sync_interval: int = 60,
    ):
        """
        Initialize the synchronization-aware Neo4j storage.

        Args:
            namespace: Namespace for the storage
            global_config: Global configuration
            embedding_func: Function to generate embeddings
            entity_vector_storage: Vector storage for entities
            relationship_vector_storage: Vector storage for relationships
            auto_sync: Whether to automatically synchronize with vector storage
            sync_interval: Interval in seconds for automatic synchronization
        """
        super().__init__(namespace, global_config, embedding_func)

        self.entity_vector_storage = entity_vector_storage
        self.relationship_vector_storage = relationship_vector_storage
        self.auto_sync = auto_sync
        self.sync_interval = sync_interval

        # Initialize synchronizer if vector storage is provided
        self.synchronizer = None
        if entity_vector_storage:
            self.synchronizer = KGIndexSynchronizer(
                graph_storage=self,
                entity_vector_storage=entity_vector_storage,
                relationship_vector_storage=relationship_vector_storage,
                batch_size=50,
            )

        # Flag to track if scheduled sync is running
        self._scheduled_sync_running = False

    async def initialize(self):
        """Initialize the storage and start scheduled synchronization if enabled."""
        await super().initialize()

        # Initialize async listener flag
        self._async_listener_running = False

        # Start scheduled synchronization if auto_sync is enabled
        if self.auto_sync and self.synchronizer:
            await self.start_scheduled_sync()

    async def finalize(self):
        """Finalize the storage and stop scheduled synchronization."""
        # Stop scheduled synchronization if running
        if hasattr(self, '_scheduled_sync_running') and self._scheduled_sync_running and self.synchronizer:
            self.synchronizer.stop_scheduled_sync()
            self._scheduled_sync_running = False

        # Stop async listener if running
        if hasattr(self, '_async_listener_running') and self._async_listener_running and self.synchronizer:
            self.synchronizer.stop_async_listener()
            self._async_listener_running = False

        await super().finalize()

    async def start_scheduled_sync(self) -> bool:
        """
        Start scheduled synchronization.

        Returns:
            bool: True if successfully started
        """
        if not self.synchronizer:
            logger.warning("Cannot start scheduled sync: No synchronizer available")
            return False

        if self._scheduled_sync_running:
            logger.warning("Scheduled synchronization is already running")
            return False

        success = await self.synchronizer.start_scheduled_sync(self.sync_interval)
        if success:
            self._scheduled_sync_running = True

        return success

    async def stop_scheduled_sync(self) -> bool:
        """
        Stop scheduled synchronization.

        Returns:
            bool: True if successfully stopped
        """
        if not self.synchronizer:
            logger.warning("Cannot stop scheduled sync: No synchronizer available")
            return False

        if not self._scheduled_sync_running:
            logger.warning("No scheduled synchronization running")
            return False

        success = self.synchronizer.stop_scheduled_sync()
        if success:
            self._scheduled_sync_running = False

        return success

    async def start_async_listener(self, interval_seconds: int = 60) -> bool:
        """
        Start an asynchronous listener for index updates.

        Args:
            interval_seconds: Interval in seconds between checks

        Returns:
            bool: True if successfully started
        """
        if not self.synchronizer:
            logger.warning("Cannot start async listener: No synchronizer available")
            return False

        # Initialize the flag if it doesn't exist
        if not hasattr(self, '_async_listener_running'):
            self._async_listener_running = False

        if self._async_listener_running:
            logger.warning("Async listener is already running")
            return False

        try:
            success = await self.synchronizer.start_async_listener(interval_seconds)
            if success:
                self._async_listener_running = True
            return success
        except Exception as e:
            logger.error(f"Error starting async listener: {e}")
            return False

    async def stop_async_listener(self) -> bool:
        """
        Stop the asynchronous listener.

        Returns:
            bool: True if successfully stopped
        """
        if not self.synchronizer:
            logger.warning("Cannot stop async listener: No synchronizer available")
            return False

        # Initialize the flag if it doesn't exist
        if not hasattr(self, '_async_listener_running'):
            self._async_listener_running = False
            return False

        if not self._async_listener_running:
            logger.warning("No async listener running")
            return False

        success = self.synchronizer.stop_async_listener()
        if success:
            self._async_listener_running = False

        return success

    async def upsert_node(self, node_id: str, node_data: Dict[str, Any]) -> None:
        """
        Insert or update a node and synchronize with vector storage.

        Args:
            node_id: ID of the node to insert or update
            node_data: Node properties
        """
        # Call parent method to update the graph
        await super().upsert_node(node_id, node_data)

        # Mark the node for vector storage update
        if self.synchronizer:
            await self.synchronizer.mark_entity_for_update(node_id)

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: Dict[str, Any]
    ) -> None:
        """
        Insert or update an edge and synchronize with vector storage.

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node
            edge_data: Edge properties
        """
        # Call parent method to update the graph
        await super().upsert_edge(source_node_id, target_node_id, edge_data)

        # Mark the relationship for vector storage update
        if self.synchronizer:
            await self.synchronizer.mark_relationship_for_update(source_node_id, target_node_id)

    async def delete_node(self, node_id: str) -> bool:
        """
        Delete a node and synchronize with vector storage.

        Args:
            node_id: ID of the node to delete

        Returns:
            bool: True if the node was deleted
        """
        # Call parent method to delete the node from the graph
        result = await super().delete_node(node_id)

        # Delete the node from vector storage
        if result and self.entity_vector_storage:
            try:
                await self.entity_vector_storage.delete([node_id])
                logger.debug(f"Deleted node {node_id} from vector storage")
            except Exception as e:
                logger.error(f"Error deleting node {node_id} from vector storage: {e}")

        return result

    async def delete_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Delete an edge and synchronize with vector storage.

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node

        Returns:
            bool: True if the edge was deleted
        """
        # Neo4JStorage doesn't have delete_edge, but it has remove_edges
        # which internally calls a private _do_delete_edge method
        try:
            # Create a helper function to delete the edge
            async def _do_delete_edge(tx):
                query = """
                MATCH (source:base {entity_id: $source_entity_id})-[r]-(target:base {entity_id: $target_entity_id})
                DELETE r
                """
                result = await tx.run(
                    query, source_entity_id=source_node_id, target_entity_id=target_node_id
                )
                await result.consume()  # Ensure result is fully consumed
                return True

            # Execute the delete operation
            async with self._driver.session(database=self._DATABASE) as session:
                result = await session.execute_write(_do_delete_edge)
                logger.debug(f"Deleted edge from '{source_node_id}' to '{target_node_id}'")
        except Exception as e:
            logger.error(f"Error deleting edge {source_node_id}->{target_node_id}: {e}")
            return False

        # Delete the relationship from vector storage
        if result and self.relationship_vector_storage:
            try:
                from ..utils import compute_mdhash_id
                rel_id = compute_mdhash_id(f"{source_node_id}_{target_node_id}", prefix="rel-")
                await self.relationship_vector_storage.delete([rel_id])
                logger.debug(f"Deleted relationship {rel_id} from vector storage")
            except Exception as e:
                logger.error(f"Error deleting relationship {source_node_id}->{target_node_id} from vector storage: {e}")

        return result

    # Alias for delete_edge to support tests that expect delete_relationship
    async def delete_relationship(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Alias for delete_edge to maintain compatibility with tests.

        Args:
            source_node_id: ID of the source node
            target_node_id: ID of the target node

        Returns:
            bool: True if the edge was deleted
        """
        return await self.delete_edge(source_node_id, target_node_id)

    async def check_index_consistency(self) -> Dict[str, Any]:
        """
        Check consistency between graph and vector storage.

        Returns:
            Dict[str, Any]: Statistics about index consistency
        """
        if not self.synchronizer:
            return {"error": "No synchronizer available"}

        return await self.synchronizer.check_index_consistency()

    async def rebuild_indices(self) -> Dict[str, Any]:
        """
        Rebuild all vector indices from graph data.

        Returns:
            Dict[str, Any]: Statistics about the rebuild operation
        """
        if not self.synchronizer:
            return {"error": "No synchronizer available"}

        return await self.synchronizer.rebuild_indices()

    async def process_updates_async(self, batch_size: int = None) -> Dict[str, Any]:
        """
        Process pending updates asynchronously.

        Args:
            batch_size: Optional batch size to override the default

        Returns:
            Dict[str, Any]: Statistics about the updates
        """
        if not self.synchronizer:
            return {"error": "No synchronizer available"}

        return await self.synchronizer.process_updates_async(batch_size)
