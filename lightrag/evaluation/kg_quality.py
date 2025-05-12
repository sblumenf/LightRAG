"""
Knowledge Graph quality evaluation for LightRAG.

This module provides utilities for evaluating the quality of LightRAG's
knowledge graph, including schema conformance, entity resolution, and
relationship quality.
"""

import asyncio
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field

from lightrag import LightRAG
from lightrag.schema.schema_loader import SchemaLoader
from lightrag.schema.schema_validator import SchemaValidator


@dataclass
class KGQualityMetrics:
    """Class for storing knowledge graph quality metrics."""
    schema_conformance_rate: float = 0.0  # Percentage of entities conforming to schema
    entity_types_coverage: Dict[str, float] = field(default_factory=dict)  # Coverage of each entity type
    relationship_types_coverage: Dict[str, float] = field(default_factory=dict)  # Coverage of each relationship type
    entity_property_completeness: float = 0.0  # Average percentage of properties filled
    relationship_property_completeness: float = 0.0  # Average percentage of relationship properties filled
    entity_resolution_precision: float = 0.0  # Precision of entity resolution
    entity_resolution_recall: float = 0.0  # Recall of entity resolution
    entity_resolution_f1: float = 0.0  # F1 score of entity resolution
    relationship_quality_score: float = 0.0  # Overall relationship quality score
    total_entities: int = 0  # Total number of entities
    total_relationships: int = 0  # Total number of relationships
    schema_violations: int = 0  # Number of schema violations
    orphaned_entities: int = 0  # Number of entities with no relationships

    def to_dict(self) -> Dict[str, Any]:
        """Convert the metrics to a dictionary."""
        return asdict(self)


async def evaluate_kg_quality(
    rag: LightRAG,
    schema_path: Optional[str] = None
) -> KGQualityMetrics:
    """
    Evaluate the overall quality of the knowledge graph.

    Args:
        rag: The LightRAG instance to evaluate
        schema_path: Path to the schema file (if None, uses the one from config)

    Returns:
        KGQualityMetrics: The knowledge graph quality metrics
    """
    # Load schema
    schema_loader = SchemaLoader()
    schema = schema_loader.load_schema(schema_path)
    
    # Create schema validator
    schema_validator = SchemaValidator(schema)
    
    # Get all entities and relationships from the graph
    graph_storage = rag.chunk_entity_relation_graph
    
    # Get all entity IDs
    entity_ids = await graph_storage.get_all_node_ids()
    
    # Initialize metrics
    metrics = KGQualityMetrics()
    metrics.total_entities = len(entity_ids)
    
    # Evaluate schema conformance
    schema_conformance = await evaluate_schema_conformance(graph_storage, schema_validator)
    metrics.schema_conformance_rate = schema_conformance["conformance_rate"]
    metrics.entity_types_coverage = schema_conformance["entity_types_coverage"]
    metrics.relationship_types_coverage = schema_conformance["relationship_types_coverage"]
    metrics.entity_property_completeness = schema_conformance["entity_property_completeness"]
    metrics.relationship_property_completeness = schema_conformance["relationship_property_completeness"]
    metrics.schema_violations = schema_conformance["schema_violations"]
    
    # Evaluate entity resolution
    entity_resolution = await evaluate_entity_resolution(graph_storage, rag.embedding_func)
    metrics.entity_resolution_precision = entity_resolution["precision"]
    metrics.entity_resolution_recall = entity_resolution["recall"]
    metrics.entity_resolution_f1 = entity_resolution["f1"]
    
    # Evaluate relationship quality
    relationship_quality = await evaluate_relationship_quality(graph_storage)
    metrics.relationship_quality_score = relationship_quality["quality_score"]
    metrics.total_relationships = relationship_quality["total_relationships"]
    metrics.orphaned_entities = relationship_quality["orphaned_entities"]
    
    return metrics


async def evaluate_schema_conformance(
    graph_storage: Any,
    schema_validator: SchemaValidator
) -> Dict[str, Any]:
    """
    Evaluate how well the knowledge graph conforms to the schema.

    Args:
        graph_storage: The graph storage to evaluate
        schema_validator: The schema validator to use

    Returns:
        Dict[str, Any]: Schema conformance metrics
    """
    # Get all entity IDs
    entity_ids = await graph_storage.get_all_node_ids()
    
    # Initialize counters
    conforming_entities = 0
    entity_type_counts = {}
    entity_property_completeness = []
    schema_violations = 0
    
    # Check each entity
    for entity_id in entity_ids:
        entity = await graph_storage.get_node_by_id(entity_id)
        if not entity:
            continue
        
        # Check entity type
        entity_type = entity.get("entity_type")
        if not entity_type:
            schema_violations += 1
            continue
        
        # Update entity type counts
        entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
        
        # Validate entity against schema
        is_valid, violations = schema_validator.validate_entity(entity)
        if is_valid:
            conforming_entities += 1
        else:
            schema_violations += len(violations)
        
        # Calculate property completeness
        required_properties = schema_validator.get_required_properties(entity_type)
        if required_properties:
            properties = entity.get("properties", {})
            filled_properties = sum(1 for prop in required_properties if prop in properties and properties[prop])
            completeness = filled_properties / len(required_properties)
            entity_property_completeness.append(completeness)
    
    # Get all relationships
    relationship_ids = await graph_storage.get_all_edge_ids()
    
    # Initialize relationship counters
    relationship_type_counts = {}
    relationship_property_completeness = []
    
    # Check each relationship
    for rel_id in relationship_ids:
        relationship = await graph_storage.get_edge_by_id(rel_id)
        if not relationship:
            continue
        
        # Check relationship type
        rel_type = relationship.get("relationship_type")
        if not rel_type:
            schema_violations += 1
            continue
        
        # Update relationship type counts
        relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1
        
        # Validate relationship against schema
        is_valid, violations = schema_validator.validate_relationship(relationship)
        if not is_valid:
            schema_violations += len(violations)
        
        # Calculate property completeness
        required_properties = schema_validator.get_required_relationship_properties(rel_type)
        if required_properties:
            properties = relationship.get("properties", {})
            filled_properties = sum(1 for prop in required_properties if prop in properties and properties[prop])
            completeness = filled_properties / len(required_properties)
            relationship_property_completeness.append(completeness)
    
    # Calculate entity type coverage
    schema_entity_types = schema_validator.get_entity_types()
    entity_types_coverage = {
        entity_type: entity_type_counts.get(entity_type, 0) / len(entity_ids) if entity_ids else 0
        for entity_type in schema_entity_types
    }
    
    # Calculate relationship type coverage
    schema_relationship_types = schema_validator.get_relationship_types()
    relationship_types_coverage = {
        rel_type: relationship_type_counts.get(rel_type, 0) / len(relationship_ids) if relationship_ids else 0
        for rel_type in schema_relationship_types
    }
    
    # Calculate overall metrics
    conformance_rate = conforming_entities / len(entity_ids) if entity_ids else 0
    avg_entity_property_completeness = sum(entity_property_completeness) / len(entity_property_completeness) if entity_property_completeness else 0
    avg_relationship_property_completeness = sum(relationship_property_completeness) / len(relationship_property_completeness) if relationship_property_completeness else 0
    
    return {
        "conformance_rate": conformance_rate,
        "entity_types_coverage": entity_types_coverage,
        "relationship_types_coverage": relationship_types_coverage,
        "entity_property_completeness": avg_entity_property_completeness,
        "relationship_property_completeness": avg_relationship_property_completeness,
        "schema_violations": schema_violations
    }


async def evaluate_entity_resolution(
    graph_storage: Any,
    embedding_func: Any
) -> Dict[str, float]:
    """
    Evaluate the quality of entity resolution.

    Args:
        graph_storage: The graph storage to evaluate
        embedding_func: The embedding function to use

    Returns:
        Dict[str, float]: Entity resolution metrics
    """
    # This is a simplified evaluation that would need ground truth data for a real evaluation
    # For now, we'll use a heuristic approach to estimate precision and recall
    
    # Get all entity IDs
    entity_ids = await graph_storage.get_all_node_ids()
    
    # Initialize counters
    total_entities = len(entity_ids)
    potential_duplicates = 0
    confirmed_duplicates = 0
    
    # Simple heuristic: entities with very similar names or embeddings are potential duplicates
    from lightrag.kg.entity_resolver import calculate_name_similarity, calculate_embedding_similarity
    
    # Check a sample of entity pairs for potential duplicates
    # In a real evaluation, we would use ground truth data
    sample_size = min(100, total_entities)
    sampled_ids = entity_ids[:sample_size]
    
    for i, entity_id1 in enumerate(sampled_ids):
        entity1 = await graph_storage.get_node_by_id(entity_id1)
        if not entity1:
            continue
        
        entity1_name = entity1.get("properties", {}).get("name", "")
        entity1_type = entity1.get("entity_type", "")
        
        for j in range(i + 1, len(sampled_ids)):
            entity_id2 = sampled_ids[j]
            entity2 = await graph_storage.get_node_by_id(entity_id2)
            if not entity2:
                continue
            
            entity2_name = entity2.get("properties", {}).get("name", "")
            entity2_type = entity2.get("entity_type", "")
            
            # Only compare entities of the same type
            if entity1_type != entity2_type:
                continue
            
            # Calculate name similarity
            name_similarity = await calculate_name_similarity(entity1_name, entity2_name)
            
            # If names are similar, they are potential duplicates
            if name_similarity > 0.8:
                potential_duplicates += 1
                
                # If they also have very similar properties, they are confirmed duplicates
                property_similarity = calculate_property_similarity(
                    entity1.get("properties", {}),
                    entity2.get("properties", {})
                )
                
                if property_similarity > 0.7:
                    confirmed_duplicates += 1
    
    # Calculate precision and recall
    # These are estimates based on our heuristic approach
    precision = confirmed_duplicates / potential_duplicates if potential_duplicates > 0 else 1.0
    recall = 0.8  # This is a placeholder; real recall would require ground truth data
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def calculate_property_similarity(props1: Dict[str, Any], props2: Dict[str, Any]) -> float:
    """
    Calculate the similarity between two sets of properties.

    Args:
        props1: First set of properties
        props2: Second set of properties

    Returns:
        float: Similarity score between 0 and 1
    """
    # Get all property keys
    all_keys = set(props1.keys()) | set(props2.keys())
    if not all_keys:
        return 0
    
    # Count matching properties
    matches = 0
    for key in all_keys:
        if key in props1 and key in props2:
            val1 = props1[key]
            val2 = props2[key]
            
            # Compare values
            if isinstance(val1, str) and isinstance(val2, str):
                # For strings, check if they are similar
                if val1.lower() == val2.lower() or val1.lower() in val2.lower() or val2.lower() in val1.lower():
                    matches += 1
            elif val1 == val2:
                matches += 1
    
    # Calculate similarity
    return matches / len(all_keys)


async def evaluate_relationship_quality(graph_storage: Any) -> Dict[str, Any]:
    """
    Evaluate the quality of relationships in the knowledge graph.

    Args:
        graph_storage: The graph storage to evaluate

    Returns:
        Dict[str, Any]: Relationship quality metrics
    """
    # Get all entity IDs
    entity_ids = await graph_storage.get_all_node_ids()
    
    # Get all relationships
    relationship_ids = await graph_storage.get_all_edge_ids()
    
    # Initialize counters
    total_entities = len(entity_ids)
    total_relationships = len(relationship_ids)
    entities_with_relationships = set()
    
    # Check each relationship
    for rel_id in relationship_ids:
        relationship = await graph_storage.get_edge_by_id(rel_id)
        if not relationship:
            continue
        
        # Add source and target entities to the set
        source_id = relationship.get("source_id")
        target_id = relationship.get("target_id")
        
        if source_id:
            entities_with_relationships.add(source_id)
        
        if target_id:
            entities_with_relationships.add(target_id)
    
    # Calculate metrics
    orphaned_entities = total_entities - len(entities_with_relationships)
    relationship_density = total_relationships / total_entities if total_entities > 0 else 0
    
    # Calculate a quality score based on relationship density and orphaned entities
    quality_score = relationship_density * (1 - orphaned_entities / total_entities) if total_entities > 0 else 0
    
    return {
        "quality_score": quality_score,
        "total_relationships": total_relationships,
        "orphaned_entities": orphaned_entities,
        "relationship_density": relationship_density
    }
