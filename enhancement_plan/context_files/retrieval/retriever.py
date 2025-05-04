"""
Retriever module for GraphRAG tutor.

This module provides functionality to retrieve relevant information from the knowledge graph
using hybrid retrieval approaches combining vector similarity and graph traversal.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple, Set

from ..knowledge_graph.neo4j_knowledge_graph import Neo4jKnowledgeGraph
from config import settings
from typing import Optional, Dict, Any # Ensure these are imported

logger = logging.getLogger(__name__)

# Lazy import for Google AI
_google_ai = None

def _load_google_ai():
    """Lazy load Google AI library."""
    global _google_ai
    if _google_ai is None:
        try:
            import google.generativeai as genai
            _google_ai = genai
            _google_ai.configure(api_key=settings.GOOGLE_API_KEY)
        except ImportError:
            logger.error("Failed to import Google AI library. Make sure it's installed with 'pip install google-generativeai'")
            raise
    return _google_ai


class GraphRAGRetriever:
    """
    Retriever for GraphRAG that implements hybrid retrieval strategies.

    Combines vector similarity search with graph traversal to retrieve
    the most relevant information for a given query.
    """

    # Modify the __init__ signature and body:
    def __init__(
        self,
        knowledge_graph: Neo4jKnowledgeGraph,
        # Add config parameter
        config: Optional[Dict[str, Any]] = None,
        # Remove defaults from other parameters that will be configured
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        retrieval_limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_related_depth: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ):
        """
        Initialize the retriever with the knowledge graph and settings.

        Args:
            knowledge_graph: Neo4jKnowledgeGraph instance
            config: Optional configuration dictionary for overrides.
            model_name: Name of the LLM model to use for query understanding (overridden by config)
            embedding_model: Name of the embedding model for vector similarity (overridden by config)
            retrieval_limit: Maximum number of results to retrieve (overridden by config)
            similarity_threshold: Minimum similarity score for vector search (overridden by config)
            max_related_depth: Maximum relationship depth for graph traversal (overridden by config)
            max_retries: Maximum number of retries for API calls (overridden by config)
            retry_delay: Delay between retries in seconds (overridden by config)
        """
        self.knowledge_graph = knowledge_graph
        # Initialize config dictionary
        self.config = config if config is not None else {}

        # Initialize attributes using config override pattern
        self.model_name = self.config.get('model_name', settings.DEFAULT_GOOGLE_LLM_MODEL)
        self.embedding_model = self.config.get('embedding_model', settings.DEFAULT_EMBEDDING_MODEL)
        self.retrieval_limit = self.config.get('retrieval_limit', settings.DEFAULT_RETRIEVAL_LIMIT) # Use DEFAULT_RETRIEVAL_LIMIT
        self.similarity_threshold = self.config.get('similarity_threshold', settings.RETRIEVAL_SIMILARITY_THRESHOLD) # Use new setting
        self.max_related_depth = self.config.get('max_related_depth', settings.RETRIEVAL_MAX_RELATED_DEPTH) # Use new setting
        # Configure retries/delay if needed by retriever methods (e.g., _call_gemini)
        self.max_retries = self.config.get('max_retries', 3) # Example default if not in settings
        self.retry_delay = self.config.get('retry_delay', 1.0) # Example default if not in settings

        # Initialize Google AI (keep existing logic)
        self.genai = _load_google_ai()

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query and extract important information.

        Args:
            query: The user query string

        Returns:
            Dict containing query analysis results:
                - entity_types: Potential entity types mentioned in the query
                - keywords: Key terms extracted from the query
                - intent: Understanding of query intent (e.g., explanation, comparison)
                - expanded_query: Enhanced version of the query for better retrieval
        """
        # Build prompt for query analysis
        prompt = self._build_query_analysis_prompt(query)

        # Call Gemini for query analysis
        result = self._call_gemini(prompt)

        # Parse the response
        analysis = self._parse_query_analysis_response(result)

        # Add the original query
        analysis['original_query'] = query

        return analysis

    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding vector for the query.

        Args:
            query: The query string

        Returns:
            List[float]: Embedding vector
        """
        genai = self.genai

        try:
            embedding_model = genai.GenerativeModel(self.embedding_model)
            result = embedding_model.generate_content(query, generation_config={"embedding_only": True})
            embedding = result.embedding
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

    def retrieve(self, query: str, strategy: str = "hybrid") -> Dict[str, Any]:
        """
        Retrieve relevant information for a query using the specified strategy.

        Args:
            query: The user query string
            strategy: Retrieval strategy to use:
                      "vector" - vector similarity only
                      "graph" - graph traversal only
                      "hybrid" - combined approach (default)

        Returns:
            Dict containing retrieval results:
                - results: List of retrieved chunks/nodes
                - query_analysis: Analysis of the original query
                - strategy_used: Strategy that was used
                - execution_time: Time taken for retrieval in seconds
                - confidence: Strategy confidence score
        """
        start_time = time.time()
        logger.info(f"Retrieving with strategy: {strategy} for query: '{query[:50]}...'") # Add this line

        # Process and analyze the query
        query_analysis = self.process_query(query)

        # Generate query embedding
        query_embedding = self.generate_query_embedding(query)

        # Initialize results
        all_results = []
        used_strategy = strategy
        strategy_confidence = 0.7  # Default confidence in the strategy

        # Execute the appropriate retrieval strategy
        if strategy == "vector":
            # Pure vector-based retrieval with analysis-based limit adjustment
            max_results = self.retrieval_limit
            # For definition queries, we often need fewer results
            if query_analysis.get('intent', '') in ['define', 'explain']:
                max_results = min(max_results, 5)
            # Use the version with limit parameter
            results = self._vector_retrieval(query_embedding, query_analysis, limit=max_results)
            all_results = results

            # Calculate strategy confidence
            vector_indicators = ['definition', 'meaning', 'concept', 'explain', 'summarize']
            query_lower = query.lower()
            vector_score = sum(1 for indicator in vector_indicators if indicator in query_lower)
            strategy_confidence = min(0.9, 0.6 + (vector_score * 0.1))

        elif strategy == "graph":
            # Pure graph-based retrieval with intent-guided traversal
            intent = query_analysis.get('intent', '').lower()
            depth = self.max_related_depth

            # Adjust traversal depth based on intent
            if intent in ['compare', 'relationship']:
                depth = 3  # Deeper for relationship queries
            elif intent in ['example', 'instance']:
                depth = 2  # Moderate for example queries

            # Execute graph retrieval with intent-specific parameters
            results = self._graph_retrieval(query_analysis, depth)
            all_results = results

            # Calculate strategy confidence
            graph_indicators = ['related', 'compare', 'connection', 'impact']
            query_lower = query.lower()
            graph_score = sum(1 for indicator in graph_indicators if indicator in query_lower)
            strategy_confidence = min(0.9, 0.6 + (graph_score * 0.1))

        elif strategy == "hybrid":
            # First get vector results
            vector_results = self._vector_retrieval(query_embedding, query_analysis)

            # Then get graph-based results from top vector matches
            graph_results = []

            # Determine optimal seed count based on query analysis
            seed_count = 3  # Default number of seeds

            # For relationship/comparison queries, use more seeds
            if 'compare' in query_analysis.get('intent', '').lower() or len(query_analysis.get('entity_types', [])) > 1:
                seed_count = 5

            # Use top N as seeds
            seed_ids = [r.get('chunk_id') for r in vector_results[:seed_count]]

            for chunk_id in seed_ids:
                related = self._get_related_nodes(chunk_id, query_analysis)
                graph_results.extend(related)

            # Combine results with deduplication and dynamic weighting
            all_results = self._merge_and_deduplicate_results(vector_results, graph_results, query_analysis)

            # High confidence in hybrid approach for complex queries
            strategy_confidence = 0.85

        else:
            logger.warning(f"Unknown retrieval strategy: {strategy}. Falling back to hybrid.")
            used_strategy = "hybrid"
            return self.retrieve(query, strategy="hybrid")

        # Calculate execution time
        execution_time = time.time() - start_time

        return {
            'results': all_results,
            'query_analysis': query_analysis,
            'strategy_used': used_strategy,
            'execution_time': execution_time,
            'strategy_confidence': strategy_confidence
        }

    def _vector_retrieval(self,
                          query_embedding: List[float],
                          query_analysis: Dict[str, Any],
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform vector similarity-based retrieval.

        Args:
            query_embedding: Embedding vector of the query
            query_analysis: Analysis of the query
            limit: Optional custom limit (defaults to self.retrieval_limit)

        Returns:
            List[Dict[str, Any]]: Retrieved results
        """
        # Extract potential entity type from query analysis
        entity_type = None
        if query_analysis.get('entity_types') and len(query_analysis['entity_types']) == 1:
            entity_type = query_analysis['entity_types'][0]

        # Use provided limit or default
        result_limit = limit or self.retrieval_limit

        # Perform vector similarity search using configured values
        results = self.knowledge_graph.similarity_search(
            query_text_or_embedding=query_embedding,
            limit=result_limit, # Uses self.retrieval_limit
            similarity_threshold=self.similarity_threshold, # Uses configured value
            entity_type=entity_type
        )

        return results

    def retrieve_by_entity_type(self, query: str, entity_type: str) -> Dict[str, Any]:
        """
        Retrieve information specific to a given entity type.

        Args:
            query: The user query string
            entity_type: Entity type to filter results by

        Returns:
            Dict: Retrieval results filtered by entity type
        """
        logger.info(f"Retrieving by entity type: {entity_type} for query: '{query[:50]}...'") # Add logging
        # Process query and generate embedding
        query_embedding = self.generate_query_embedding(query)
        # Perform vector similarity search with entity type filter
        results = self.knowledge_graph.similarity_search(
            query_embedding,
            limit=self.retrieval_limit,
            similarity_threshold=self.similarity_threshold,
            entity_type=entity_type
        )

        return {
            'results': results,
            'entity_type': entity_type,
            'count': len(results)
        }

    # This method is a duplicate and has been removed

    def _graph_retrieval(self, query_analysis: Dict[str, Any], depth: int = 2) -> List[Dict[str, Any]]:
        """
        Perform graph-based retrieval using Cypher queries.

        Args:
            query_analysis: Analysis of the query
            depth: Maximum path length for relationship traversal

        Returns:
            List[Dict[str, Any]]: Retrieved results
        """
        # Extract relevant query components
        entity_types = query_analysis.get('entity_types', [])
        keywords = query_analysis.get('keywords', [])
        intent = query_analysis.get('intent', '').lower()
        expanded_query = query_analysis.get('expanded_query', '')

        # Execute a schema-aware Cypher query
        try:
            with self.knowledge_graph._driver.session(database=self.knowledge_graph.database) as session:
                # Start building cypher query parts
                match_clauses = []
                where_conditions = []
                return_items = """
                    n.chunk_id AS chunk_id,
                    n.text AS text,
                    n.source_doc AS source_doc,
                    n.entity_type AS entity_type,
                    labels(n) AS labels,
                    n.importance AS importance,
                    n.confidence AS confidence
                """
                order_by = "n.importance DESC, n.confidence DESC"
                limit = self.retrieval_limit # Use configured value

                # 1. Build entity type filters - prioritize schema-defined types
                if entity_types:
                    # Include both exact entity types and tentative types that might be relevant
                    entity_condition_parts = []

                    # Exact entity type matching
                    entity_labels = " OR ".join([f"n:{entity_type}" for entity_type in entity_types])
                    entity_condition_parts.append(f"({entity_labels})")

                    # Also tentative entity types that might be related
                    tentative_conditions = []
                    for entity_type in entity_types:
                        tentative_conditions.append(f"n:TentativeEntity AND n.entity_type CONTAINS '{entity_type}'")

                    if tentative_conditions:
                        entity_condition_parts.append("(" + " OR ".join(tentative_conditions) + ")")

                    # Combine all entity conditions
                    where_conditions.append("(" + " OR ".join(entity_condition_parts) + ")")

                # 2. Build keyword filters with smarter term matching
                if keywords:
                    keyword_parts = []

                    # Basic text search with word boundaries for better precision
                    text_conditions = []
                    for keyword in keywords:
                        if len(keyword) > 3:  # Only use word boundaries for longer terms
                            text_conditions.append(f"n.text =~ '(?i)\\\\b{keyword}\\\\b'")
                        else:
                            text_conditions.append(f"n.text CONTAINS '{keyword}'")

                    if text_conditions:
                        keyword_parts.append("(" + " OR ".join(text_conditions) + ")")

                    # Property-based search for entity properties
                    property_conditions = []
                    for keyword in keywords:
                        property_conditions.append(f"ANY(prop IN keys(n) WHERE n[prop] CONTAINS '{keyword}')")

                    if property_conditions:
                        keyword_parts.append("(" + " OR ".join(property_conditions) + ")")

                    # Combine all keyword conditions
                    where_conditions.append("(" + " OR ".join(keyword_parts) + ")")

                # 3. Intent-based query enhancement - specialized for each intent type
                if 'compare' in intent or 'difference' in intent:
                    # For comparison intents, prioritize entities with comparison relationships
                    match_clauses.append(f"MATCH path = (n)-[:COMPARED_TO|CONTRASTS_WITH|ALTERNATIVE_TO|SIMILAR_TO|DIFFERS_FROM*1..{depth}]-() ")
                    where_conditions.append("n.entity_type IS NOT NULL")

                    # Include path length in return
                    return_items += ", length(path) AS distance"

                    # Adjust order to prioritize comparison-related entities
                    order_by = "size((n)-[:COMPARED_TO|CONTRASTS_WITH|ALTERNATIVE_TO]-()) DESC, " + order_by

                elif 'define' in intent or 'explain' in intent:
                    # For definition intents, prioritize schema-typed entities
                    where_conditions.append("(n.entity_type IS NOT NULL AND n.entity_type <> 'Chunk')")

                    # For definitions, look for definitional relationships
                    match_clauses.append(f"MATCH (n) WHERE EXISTS((n)-[:DEFINES|EXPLAINS|DESCRIBES]-()) ")

                    # Prioritize higher confidence matches for definitions
                    order_by = "n.confidence DESC, " + order_by

                elif 'example' in intent or 'instance' in intent:
                    # For example intents, look for example relationships with custom depth
                    match_clauses.append(f"MATCH path = (n)-[:HAS_EXAMPLE|DEMONSTRATES|INSTANCE_OF*1..{depth}]-() ")

                    # Include path length in return
                    return_items += ", length(path) AS distance"

                    # Prioritize examples with higher importance
                    order_by = "n.importance DESC, size((n)-[:HAS_EXAMPLE|DEMONSTRATES]-()) DESC"

                elif 'process' in intent or 'steps' in intent or 'workflow' in intent:
                    # For process intents, look for sequential relationships with proper ordering
                    match_clauses.append(f"MATCH path = (n)-[:NEXT_STEP|FOLLOWS|PRECEDES|LEADS_TO*1..{depth}]-() ")

                    # Include path length and steps in return
                    return_items += ", length(path) AS distance"

                    # Prioritize by step order and importance
                    order_by = "n.step_order ASC, n.importance DESC"

                elif 'cause' in intent or 'effect' in intent or 'impact' in intent:
                    # For causal relationship queries
                    match_clauses.append(f"MATCH path = (n)-[:CAUSES|RESULTS_IN|IMPACTS|AFFECTS|INFLUENCES*1..{depth}]-() ")

                    # Include path length in return
                    return_items += ", length(path) AS distance"

                    # Prioritize by causal strength
                    order_by = "n.importance DESC, size((n)-[:CAUSES|RESULTS_IN|IMPACTS]-()) DESC"

                # 4. Default match clause if none specified
                if not match_clauses:
                    match_clauses.append("MATCH (n) ")

                # 5. Combine all WHERE conditions if present
                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)

                # 6. Build the final query
                cypher_query = f"""
                {match_clauses[0]}
                {where_clause}
                RETURN DISTINCT {return_items}
                ORDER BY {order_by}
                LIMIT {limit}
                """

                result = session.run(cypher_query)
                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Error in graph retrieval: {str(e)}")
            logger.debug(f"Failed Cypher query: {cypher_query if 'cypher_query' in locals() else 'not generated'}")
            return []

    def _get_related_nodes(self,
                          chunk_id: str,
                          query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get nodes related to a given chunk through relationships.

        Args:
            chunk_id: ID of the chunk to find relationships for
            query_analysis: Analysis of the query to guide relationship types

        Returns:
            List[Dict[str, Any]]: List of related nodes
        """
        # Get the entity type of the source chunk first
        source_entity_type = None
        with self.knowledge_graph._driver.session(database=self.knowledge_graph.database) as session:
            try:
                result = session.run(
                    "MATCH (c {chunk_id: $chunk_id}) RETURN c.entity_type as entity_type",
                    {"chunk_id": chunk_id}
                )
                record = result.single()
                if record and record["entity_type"]:
                    source_entity_type = record["entity_type"]
            except Exception as e:
                logger.warning(f"Error getting source entity type: {str(e)}")

        # Determine if there are specific relationship types to filter by
        # based on the query analysis and potentially source entity type
        relationship_types = None
        intent = query_analysis.get('intent', '').lower()
        entity_types = query_analysis.get('entity_types', [])

        # For certain intents, use specific relationship types
        if 'compare' in intent or 'difference' in intent:
            relationship_types = ['COMPARED_TO', 'CONTRASTS_WITH', 'ALTERNATIVE_TO']
        elif 'example' in intent or 'instance' in intent:
            relationship_types = ['HAS_EXAMPLE', 'INSTANCE_OF', 'DEMONSTRATES']
        elif 'explain' in intent or 'definition' in intent:
            relationship_types = ['EXPLAINS', 'DEFINES', 'ELABORATES']

        # If we have a source entity type, try to get valid relationship types from schema
        schema_relationship_types = None
        if source_entity_type and entity_types and hasattr(self.knowledge_graph, 'schema_loader') and self.knowledge_graph.schema_loader:
            try:
                # For each target entity type mentioned in the query
                valid_relationships = []
                for target_type in entity_types:
                    # Get valid relationships from source to target
                    rel_types = self.knowledge_graph.schema_loader.get_valid_relationships(
                        source_entity_type, target_type
                    )
                    valid_relationships.extend(rel_types)

                if valid_relationships:
                    schema_relationship_types = valid_relationships
                    logger.info(f"Using schema-defined relationships: {schema_relationship_types}")
            except Exception as e:
                logger.warning(f"Error getting schema relationships: {str(e)}")

        # Prioritize schema-defined relationships if available
        if schema_relationship_types:
            relationship_types = schema_relationship_types
        elif relationship_types is None:
            # If no intent-based or schema-based relationships,
            # use a broader set of common relationship types, but avoid default RELATED_TO
            relationship_types = [
                'HAS_PREREQUISITE', 'PART_OF', 'LEADS_TO', 'CAUSES', 'EXPLAINS',
                'CONNECTED_TO', 'CONTAINS', 'SUBSET_OF', 'INFLUENCES', 'DERIVED_FROM'
            ]

        # Include tentative relationships with similar names
        if relationship_types:
            # Add tentative versions of the same relationships
            tentative_types = [f"TENTATIVE_{rel_type}" for rel_type in relationship_types]
            relationship_types.extend(tentative_types)

        # Get related nodes through the knowledge graph using configured value
        return self.knowledge_graph.get_related_nodes(
            chunk_id=chunk_id,
            relationship_types=relationship_types,
            max_distance=self.max_related_depth # Use configured value
        )
    def _merge_and_deduplicate_results(self,
                                     vector_results: List[Dict[str, Any]],
                                     graph_results: List[Dict[str, Any]],
                                     query_analysis: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate results from different retrieval methods with dynamic weighting.

        Args:
            vector_results: Results from vector similarity search
            graph_results: Results from graph traversal
            query_analysis: Optional query analysis to adjust weights

        Returns:
            List[Dict[str, Any]]: Combined and deduplicated results
        """
        # Track seen chunk IDs
        seen_chunk_ids = set()
        merged_results = []

        # Determine dynamic weights based on query analysis
        vector_weight = 0.7  # Default vector weight
        graph_weight = 0.3   # Default graph weight
        importance_weight = 0.3  # Default importance weight

        # Adjust weights based on query analysis if available
        if query_analysis:
            # Get query intent
            intent = query_analysis.get('intent', '').lower()
            entity_types = query_analysis.get('entity_types', [])

            # Adjust weights for relationship-focused queries
            if intent in ['compare', 'relationship', 'connection', 'dependency']:
                vector_weight = 0.4
                graph_weight = 0.6
                importance_weight = 0.5
            # Adjust weights for concept-focused queries
            elif intent in ['define', 'explain', 'describe']:
                vector_weight = 0.8
                graph_weight = 0.2
                importance_weight = 0.4
            # Adjust weights for multi-entity queries
            elif len(entity_types) >= 2:
                vector_weight = 0.5
                graph_weight = 0.5
                importance_weight = 0.4

        # First add vector results with similarity scores
        for result in vector_results:
            chunk_id = result.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                # Mark the source of this result
                result['retrieval_method'] = 'vector'
                # Calculate a confidence score based on similarity
                result['confidence'] = result.get('similarity', 0) * vector_weight
                merged_results.append(result)

        # Then add unique graph results
        for result in graph_results:
            chunk_id = result.get('chunk_id')
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                # Mark the source of this result
                result['retrieval_method'] = 'graph'
                # Add a default similarity score for ranking consistency
                if 'similarity' not in result:
                    result['similarity'] = 0.5

                # Calculate confidence score based on importance and graph relevance
                graph_relevance = min(1.0, result.get('distance', 3) / 3)  # Normalize distance
                result['confidence'] = (result.get('importance', 0.5) * importance_weight +
                                       graph_relevance * graph_weight)
                merged_results.append(result)

        # Sort by confidence score (combination of similarity, importance, and strategy weights)
        return sorted(
            merged_results,
            key=lambda x: x.get('confidence', 0) +
                          (x.get('similarity', 0) * vector_weight +
                           x.get('importance', 0) * importance_weight),
            reverse=True
        )

    def _build_query_analysis_prompt(self, query: str) -> str:
        """
        Build a prompt for query analysis.

        Args:
            query: The user query

        Returns:
            Prompt string for Gemini
        """
        prompt = f"""You are an expert query analyzer for a domain knowledge graph.

TASK:
Analyze the following query and extract key information needed for effective retrieval.

QUERY:
```
{query}
```

INSTRUCTIONS:
1. Identify potential entity types mentioned in the query
2. Extract key terms or concepts
3. Determine the query intent (explanation, comparison, definition, example, etc.)
4. Create an expanded version of the query that might yield better search results

FORMAT YOUR RESPONSE AS JSON:
{{
  "entity_types": ["EntityType1", "EntityType2"],
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "intent": "explanation",
  "expanded_query": "Enhanced version of the query with synonyms and related terms"
}}

IMPORTANT:
- Return ONLY the JSON object
- The knowledge graph contains domain-specific entity types like Concept, Entity, Process, etc.
- For entity_types, only include types if you're confident they're present
- Include 3-5 most relevant keywords
- Be specific about the intent
"""
        return prompt

    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini model with the given prompt.

        Args:
            prompt: The prompt to send to Gemini

        Returns:
            Gemini's response text
        """
        genai = self.genai
        response_text = None
        attempts = 0

        while attempts < self.max_retries and response_text is None:
            try:
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                response_text = response.text
            except Exception as e:
                attempts += 1
                logger.warning(f"Gemini API call attempt {attempts} failed: {str(e)}")
                if attempts < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to call Gemini after {self.max_retries} attempts")
                    raise

        return response_text

    def _parse_query_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini's query analysis response.

        Args:
            response_text: Gemini's response

        Returns:
            Parsed analysis dict
        """
        try:
            # Extract JSON from the response (Gemini sometimes adds markdown formatting)
            import json

            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()

            # Parse the JSON
            analysis = json.loads(json_text)

            # Ensure required fields are present
            if "entity_types" not in analysis:
                analysis["entity_types"] = []

            if "keywords" not in analysis:
                analysis["keywords"] = []

            if "intent" not in analysis:
                analysis["intent"] = "information"

            if "expanded_query" not in analysis:
                analysis["expanded_query"] = ""

            return analysis

        except Exception as e:
            logger.error(f"Error parsing query analysis response: {str(e)}")
            logger.debug(f"Raw response: {response_text}")
            # Return a default analysis
            return {
                "entity_types": [],
                "keywords": [],
                "intent": "information",
                "expanded_query": ""
            }
