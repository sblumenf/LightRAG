"""
Evaluation module for diagram entity extraction quality.

This module provides metrics to evaluate the quality of entities and relationships
extracted from diagrams, including precision, recall, and F1 score.
"""
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class DiagramEntityEvaluator:
    """
    Evaluator for diagram entity extraction quality.
    
    This class provides methods to evaluate the quality of entities and relationships
    extracted from diagrams, including precision, recall, and F1 score.
    """
    
    def __init__(self, schema_validator=None):
        """
        Initialize the evaluator.
        
        Args:
            schema_validator: Optional schema validator for entity type validation
        """
        self.schema_validator = schema_validator
    
    async def evaluate_extraction(self, analyzer, datasets, llm_func) -> Dict[str, Any]:
        """
        Evaluate diagram entity extraction across a dataset.
        
        Args:
            analyzer: DiagramAnalyzer instance
            datasets: Dictionary of test datasets with ground truth
            llm_func: Async function to call the LLM
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {
            'entity_metrics': {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'extraction_rate': 0.0,
                'type_coverage': 0.0,
                'property_density': 0.0
            },
            'relationship_metrics': {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0, 
                'extraction_rate': 0.0,
                'relationship_to_entity_ratio': 0.0,
                'type_coverage': 0.0
            },
            'performance_metrics': {
                'avg_processing_time': 0.0,
                'total_diagrams': 0,
                'total_entities_extracted': 0,
                'total_relationships_extracted': 0
            },
            'details': {}
        }
        
        # Track totals for averaging
        total_entity_precision = 0.0
        total_entity_recall = 0.0
        total_entity_f1 = 0.0
        total_relationship_precision = 0.0
        total_relationship_recall = 0.0
        total_relationship_f1 = 0.0
        total_processing_time = 0.0
        total_diagrams = 0
        total_entities = 0
        total_relationships = 0
        
        # Available schema entity and relationship types
        entity_types = set()
        if self.schema_validator:
            entity_types = set(self.schema_validator.get_entity_types())
        relationship_types = set()
        if self.schema_validator:
            relationship_types = set(self.schema_validator.get_relationship_types())
        
        # Extracted entity and relationship types
        extracted_entity_types = set()
        extracted_relationship_types = set()
        
        # Process each diagram in the dataset
        for dataset_name, dataset in datasets.items():
            results['details'][dataset_name] = {
                'diagrams': []
            }
            
            for i, diagram in enumerate(dataset.get('diagrams', [])):
                diagram_id = diagram.get('diagram_id', f'diagram-{i}')
                diagram_result = {
                    'diagram_id': diagram_id,
                    'metrics': {}
                }
                
                # Extract entities and relationships
                start_time = time.time()
                entities, relationships = await analyzer.extract_entities_and_relationships(
                    diagram, self.schema_validator, llm_func
                )
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Calculate metrics for this diagram
                ground_truth = diagram.get('ground_truth', {})
                gt_entities = ground_truth.get('entities', [])
                gt_relationships = ground_truth.get('relationships', [])
                
                # Entity metrics
                entity_precision, entity_recall, entity_f1 = self._calculate_entity_metrics(
                    entities, gt_entities
                )
                
                # Relationship metrics
                rel_precision, rel_recall, rel_f1 = self._calculate_relationship_metrics(
                    relationships, gt_relationships
                )
                
                # Calculate extraction rates
                entity_extraction_rate = len(entities)
                relationship_extraction_rate = len(relationships)
                
                # Calculate property density
                total_properties = sum(len(entity.get('properties', {})) for entity in entities)
                property_density = total_properties / len(entities) if entities else 0
                
                # Track entity and relationship types
                for entity in entities:
                    entity_type = entity.get('entity_type')
                    if entity_type:
                        extracted_entity_types.add(entity_type)
                
                for rel in relationships:
                    rel_type = rel.get('type')
                    if rel_type:
                        extracted_relationship_types.add(rel_type)
                
                # Calculate relationship to entity ratio
                rel_to_entity_ratio = len(relationships) / len(entities) if entities else 0
                
                # Store metrics for this diagram
                diagram_result['metrics'] = {
                    'entity_precision': entity_precision,
                    'entity_recall': entity_recall,
                    'entity_f1': entity_f1,
                    'relationship_precision': rel_precision,
                    'relationship_recall': rel_recall,
                    'relationship_f1': rel_f1,
                    'entity_extraction_rate': entity_extraction_rate,
                    'relationship_extraction_rate': relationship_extraction_rate,
                    'property_density': property_density,
                    'relationship_to_entity_ratio': rel_to_entity_ratio,
                    'processing_time': processing_time
                }
                
                # Store results for this diagram
                diagram_result['extracted_entities'] = entities
                diagram_result['extracted_relationships'] = relationships
                diagram_result['ground_truth'] = ground_truth
                
                # Add to results
                results['details'][dataset_name]['diagrams'].append(diagram_result)
                
                # Update totals
                total_entity_precision += entity_precision
                total_entity_recall += entity_recall
                total_entity_f1 += entity_f1
                total_relationship_precision += rel_precision
                total_relationship_recall += rel_recall
                total_relationship_f1 += rel_f1
                total_processing_time += processing_time
                total_diagrams += 1
                total_entities += len(entities)
                total_relationships += len(relationships)
            
            # Calculate dataset-level metrics
            dataset_diagrams = results['details'][dataset_name]['diagrams']
            if dataset_diagrams:
                dataset_entity_precision = sum(d['metrics']['entity_precision'] for d in dataset_diagrams) / len(dataset_diagrams)
                dataset_entity_recall = sum(d['metrics']['entity_recall'] for d in dataset_diagrams) / len(dataset_diagrams)
                dataset_entity_f1 = sum(d['metrics']['entity_f1'] for d in dataset_diagrams) / len(dataset_diagrams)
                
                dataset_rel_precision = sum(d['metrics']['relationship_precision'] for d in dataset_diagrams) / len(dataset_diagrams)
                dataset_rel_recall = sum(d['metrics']['relationship_recall'] for d in dataset_diagrams) / len(dataset_diagrams)
                dataset_rel_f1 = sum(d['metrics']['relationship_f1'] for d in dataset_diagrams) / len(dataset_diagrams)
                
                # Store dataset-level metrics
                results['details'][dataset_name]['metrics'] = {
                    'entity_precision': dataset_entity_precision,
                    'entity_recall': dataset_entity_recall,
                    'entity_f1': dataset_entity_f1,
                    'relationship_precision': dataset_rel_precision,
                    'relationship_recall': dataset_rel_recall,
                    'relationship_f1': dataset_rel_f1
                }
        
        # Calculate overall metrics
        if total_diagrams > 0:
            # Entity metrics
            results['entity_metrics']['precision'] = total_entity_precision / total_diagrams
            results['entity_metrics']['recall'] = total_entity_recall / total_diagrams
            results['entity_metrics']['f1_score'] = total_entity_f1 / total_diagrams
            results['entity_metrics']['extraction_rate'] = total_entities / total_diagrams
            results['entity_metrics']['type_coverage'] = len(extracted_entity_types) / len(entity_types) if entity_types else 0
            results['entity_metrics']['property_density'] = sum(
                sum(len(e.get('properties', {})) for e in d['extracted_entities']) / len(d['extracted_entities']) 
                if d['extracted_entities'] else 0
                for dataset in results['details'].values() 
                for d in dataset['diagrams']
            ) / total_diagrams if total_diagrams > 0 else 0
            
            # Relationship metrics
            results['relationship_metrics']['precision'] = total_relationship_precision / total_diagrams
            results['relationship_metrics']['recall'] = total_relationship_recall / total_diagrams
            results['relationship_metrics']['f1_score'] = total_relationship_f1 / total_diagrams
            results['relationship_metrics']['extraction_rate'] = total_relationships / total_diagrams
            results['relationship_metrics']['relationship_to_entity_ratio'] = total_relationships / total_entities if total_entities > 0 else 0
            results['relationship_metrics']['type_coverage'] = len(extracted_relationship_types) / len(relationship_types) if relationship_types else 0
            
            # Performance metrics
            results['performance_metrics']['avg_processing_time'] = total_processing_time / total_diagrams
            results['performance_metrics']['total_diagrams'] = total_diagrams
            results['performance_metrics']['total_entities_extracted'] = total_entities
            results['performance_metrics']['total_relationships_extracted'] = total_relationships
        
        return results
    
    def _calculate_entity_metrics(self, extracted_entities, ground_truth_entities) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score for entity extraction.
        
        Args:
            extracted_entities: List of extracted entities
            ground_truth_entities: List of ground truth entities
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not ground_truth_entities:
            return 0.0, 0.0, 0.0
        
        if not extracted_entities:
            return 0.0, 0.0, 0.0
        
        # Convert entities to comparable format
        extracted_entities_set = {self._entity_to_comparable(entity) for entity in extracted_entities}
        ground_truth_set = {self._entity_to_comparable(entity) for entity in ground_truth_entities}
        
        # Calculate true positives (correctly extracted entities)
        true_positives = extracted_entities_set.intersection(ground_truth_set)
        
        # Calculate precision and recall
        precision = len(true_positives) / len(extracted_entities) if extracted_entities else 0
        recall = len(true_positives) / len(ground_truth_entities) if ground_truth_entities else 0
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        return precision, recall, f1_score
    
    def _calculate_relationship_metrics(self, extracted_relationships, ground_truth_relationships) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score for relationship extraction.
        
        Args:
            extracted_relationships: List of extracted relationships
            ground_truth_relationships: List of ground truth relationships
            
        Returns:
            Tuple of (precision, recall, f1_score)
        """
        if not ground_truth_relationships:
            return 0.0, 0.0, 0.0
        
        if not extracted_relationships:
            return 0.0, 0.0, 0.0
        
        # Convert relationships to comparable format
        extracted_relationships_set = {self._relationship_to_comparable(rel) for rel in extracted_relationships}
        ground_truth_set = {self._relationship_to_comparable(rel) for rel in ground_truth_relationships}
        
        # Calculate true positives (correctly extracted relationships)
        true_positives = extracted_relationships_set.intersection(ground_truth_set)
        
        # Calculate precision and recall
        precision = len(true_positives) / len(extracted_relationships) if extracted_relationships else 0
        recall = len(true_positives) / len(ground_truth_relationships) if ground_truth_relationships else 0
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        return precision, recall, f1_score
    
    def _entity_to_comparable(self, entity) -> str:
        """
        Convert an entity to a comparable string representation.
        
        This normalizes the entity for comparison, focusing on entity type and name.
        
        Args:
            entity: Entity dictionary
            
        Returns:
            String representation for comparison
        """
        # Extract essential fields for comparison
        entity_type = entity.get('entity_type', '')
        entity_name = entity.get('entity_name', '')
        
        # Normalize by removing case sensitivity and whitespace
        entity_type = entity_type.lower().strip()
        entity_name = entity_name.lower().strip()
        
        return f"{entity_type}:{entity_name}"
    
    def _relationship_to_comparable(self, relationship) -> str:
        """
        Convert a relationship to a comparable string representation.
        
        This normalizes the relationship for comparison, focusing on type, source, and target.
        
        Args:
            relationship: Relationship dictionary
            
        Returns:
            String representation for comparison
        """
        # Extract essential fields for comparison
        rel_type = relationship.get('type', '')
        source = relationship.get('source', '')
        target = relationship.get('target', '')
        
        # Normalize by removing case sensitivity and whitespace
        rel_type = rel_type.lower().strip()
        source = source.lower().strip()
        target = target.lower().strip()
        
        return f"{source}:{rel_type}:{target}"
    
    def save_evaluation_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save the results
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Evaluation results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")


async def evaluate_diagram_entity_extraction(dataset_path, analyzer, schema_validator, llm_func, output_path=None):
    """
    Evaluate diagram entity extraction using the evaluation dataset.
    
    Args:
        dataset_path: Path to the evaluation dataset JSON file
        analyzer: DiagramAnalyzer instance
        schema_validator: SchemaValidator instance
        llm_func: Async function to call the LLM
        output_path: Optional path to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Load the evaluation dataset
        with open(dataset_path, 'r') as f:
            datasets = json.load(f)
        
        # Create evaluator
        evaluator = DiagramEntityEvaluator(schema_validator)
        
        # Run evaluation
        results = await evaluator.evaluate_extraction(analyzer, datasets, llm_func)
        
        # Save results if output path is provided
        if output_path:
            evaluator.save_evaluation_results(results, output_path)
        
        return results
    
    except Exception as e:
        logger.error(f"Error evaluating diagram entity extraction: {str(e)}")
        raise