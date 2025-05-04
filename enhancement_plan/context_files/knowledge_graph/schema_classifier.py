"""
Schema Classifier for GraphRAG tutor.

This module provides functionality to classify text chunks according to
the schema, using Google's Gemini Pro 1.5 LLM.
"""
import json
import logging
import time
# Add Dict, Any, Optional if missing
from typing import Dict, Any, List, Optional, Set, Tuple, Union

from ..knowledge_graph.text_chunker import TextChunk
from ..knowledge_graph.schema_loader import SchemaLoader
from config import settings

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


class SchemaClassifier:
    """
    Classifies text chunks according to the schema using Gemini Pro 1.5.
    """

    # Modify the __init__ signature:
    def __init__(
        self,
        schema_loader: SchemaLoader,
        model_name: str = settings.DEFAULT_GOOGLE_LLM_MODEL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        config: Optional[Dict[str, Any]] = None # Add config parameter
    ):
        """
        Initialize the schema classifier.

        Args:
            schema_loader: SchemaLoader instance with loaded schema
            model_name: Name of the Gemini model to use
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
            config: Optional configuration dictionary for overrides
        """
        self.schema_loader = schema_loader
        self.config = config if config is not None else {} # Store config

        # Use config overrides or defaults from settings
        self.model_name = self.config.get('model_name', model_name)
        self.max_retries = self.config.get('max_retries', max_retries)
        self.retry_delay = self.config.get('retry_delay', retry_delay)
        # Initialize thresholds using config override pattern
        self.schema_match_threshold = self.config.get('schema_match_confidence_threshold', settings.SCHEMA_MATCH_CONFIDENCE_THRESHOLD)
        self.new_type_threshold = self.config.get('new_type_confidence_threshold', settings.NEW_TYPE_CONFIDENCE_THRESHOLD)
        # Use configured default (existing setting is "Chunk", plan said "Entity", stick to existing "Chunk")
        self.default_entity_type = self.config.get('default_entity_type', settings.DEFAULT_ENTITY_TYPE)

        # Initialize Google AI only if schema is loaded
        self.genai = None
        if self.schema_loader and self.schema_loader.is_schema_loaded():
            self.genai = _load_google_ai()

        # Extract relevant schema information for prompts (keep existing logic)
        self.entity_types = []
        self.relationship_types = []
        self.entity_properties = {}
        # Use the new is_schema_loaded method here
        if self.schema_loader and self.schema_loader.is_schema_loaded():
            self.entity_types = list(self.schema_loader.get_entity_types())
            self.relationship_types = list(self.schema_loader.get_relationship_types())
            for entity_type in self.entity_types:
                self.entity_properties[entity_type] = list(self.schema_loader.get_entity_properties(entity_type))
            logger.info(f"Initialized SchemaClassifier with {len(self.entity_types)} entity types")
        else:
            logger.warning("SchemaClassifier initialized without a loaded schema.")


    def classify_chunk(self, chunk: TextChunk) -> Dict[str, Any]:
        """
        Classify a text chunk according to the schema.

        Args:
            chunk: The TextChunk to classify

        Returns:
            Dict containing classification results:
                - entity_type: The determined entity type
                - properties: Dict of extracted properties
                - confidence: Confidence score (0-1)
        """
        # Add this check at the very beginning of the method:
        # Use the new is_schema_loaded method
        if not self.schema_loader or not self.schema_loader.is_schema_loaded():
            logger.warning("Schema not loaded in SchemaClassifier. Cannot perform schema-based classification. Returning default entity type.")
            # Use the configured default entity type from self
            result = {
                "entity_type": self.default_entity_type,
                "properties": {},
                "confidence": 0.0,
                "reasoning": "Schema not loaded", # Add reasoning field for clarity
                "extracted_entities": [], # Ensure these keys exist for consistency if expected downstream
                "extracted_relationships": []
            }
            # Store classification in chunk metadata
            chunk.metadata["schema_classification"] = result
            return result

        # Prepare prompt for classification
        prompt = self._build_classification_prompt(chunk)

        # Call Gemini for classification
        try:
            result = self._call_gemini(prompt)
            # Parse the response
            classification = self._parse_classification_response(result)
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            # Create default classification on error
            classification = {
                "entity_type": self.default_entity_type,
                "properties": {},
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "extracted_entities": [],
                "extracted_relationships": []
            }
            # Store classification in chunk metadata
            chunk.metadata["schema_classification"] = classification
            return classification

        # Apply confidence threshold filtering
        confidence = classification.get("confidence", 0.0)
        entity_type = classification.get("entity_type")

        # --- In the classify_chunk method, update threshold and default type usage ---
        # Find the existing confidence threshold checks and update them:

        # Replace:
        # if (entity_type != settings.DEFAULT_ENTITY_TYPE and
        #     confidence < settings.SCHEMA_MATCH_CONFIDENCE_THRESHOLD):
        # With:
        if (entity_type != self.default_entity_type and # Use self.default_entity_type
            confidence < self.schema_match_threshold): # Use self.schema_match_threshold
            logger.info(f"Entity type {entity_type} below confidence threshold ({confidence:.2f}), using default type {self.default_entity_type}")
            classification["entity_type"] = self.default_entity_type # Use self.default_entity_type

        # Replace:
        # if (entity_type not in self.entity_types and
        #     entity_type != settings.DEFAULT_ENTITY_TYPE and
        #     confidence < settings.NEW_TYPE_CONFIDENCE_THRESHOLD):
        # With:
        if (entity_type not in self.entity_types and
            entity_type != self.default_entity_type and # Use self.default_entity_type
            confidence < self.new_type_threshold): # Use self.new_type_threshold
            logger.info(f"New entity type {entity_type} below new type threshold ({confidence:.2f}), using default type {self.default_entity_type}")
            classification["entity_type"] = self.default_entity_type # Use self.default_entity_type

        # Store classification in chunk metadata
        chunk.metadata["schema_classification"] = classification

        return classification

    def identify_relationships(self, chunks: List[TextChunk]) -> List[Dict[str, Any]]:
        """
        Identify relationships between classified chunks.

        Args:
            chunks: List of classified TextChunk objects

        Returns:
            List of relationship dictionaries
        """
        relationships = []

        # Group chunks by entity type for more efficient processing
        chunks_by_type = {}
        for chunk in chunks:
            entity_type = chunk.metadata.get("schema_classification", {}).get("entity_type")
            if entity_type:
                if entity_type not in chunks_by_type:
                    chunks_by_type[entity_type] = []
                chunks_by_type[entity_type].append(chunk)

        # Process each entity type pair to find relationships
        processed_pairs = set()
        for source_type, source_chunks in chunks_by_type.items():
            for target_type, target_chunks in chunks_by_type.items():
                # Skip self-relationships for efficiency unless specifically needed
                if source_type == target_type and source_type not in ["CorporateFinanceConcept", "EconomicConcept"]:
                    continue

                # Skip if this pair has been processed already
                pair_key = f"{source_type}_{target_type}"
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                # Get valid relationship types between these entity types
                valid_rel_types = self.schema_loader.get_valid_relationships(source_type, target_type)
                if not valid_rel_types:
                    continue

                # For each source chunk, find related target chunks
                for source_chunk in source_chunks:
                    source_id = source_chunk.chunk_id

                    # Use configurable minimum number of targets for relationship creation
                    max_targets = max(settings.MIN_SHARED_ENTITIES_FOR_RELATIONSHIP, 5)
                    target_sample = target_chunks[:max_targets]
                    target_ids = [chunk.chunk_id for chunk in target_sample]

                    # Build prompt to identify relationships
                    prompt = self._build_relationship_prompt(
                        source_chunk,
                        target_sample,
                        valid_rel_types,
                        source_type=source_type,
                        target_type=target_type
                    )

                    # Call Gemini to identify relationships
                    result = self._call_gemini(prompt)

                    # Parse relationships and add them
                    rel_results = self._parse_relationship_response(result, source_id, target_ids)
                    relationships.extend(rel_results)

                    # Add relationships to source chunk
                    for rel in rel_results:
                        source_chunk.add_relationship(
                            rel["target_id"],
                            rel["relationship_type"],
                            rel.get("properties", {})
                        )

        return relationships

    def classify_chunks(self, chunks: List[TextChunk], batch_size: int = 10) -> List[TextChunk]:
        """
        Classify multiple text chunks.

        Args:
            chunks: List of TextChunk objects to classify
            batch_size: Number of chunks to process in a batch

        Returns:
            List of classified TextChunk objects
        """
        return self.classify_chunks_batch(chunks, batch_size)

    def classify_chunks_batch(self, chunks: List[TextChunk], batch_size: int = 10) -> List[TextChunk]:
        """
        Classify multiple text chunks in batches.

        Args:
            chunks: List of TextChunk objects to classify
            batch_size: Number of chunks to process in a batch

        Returns:
            List of classified TextChunk objects
        """
        logger.info(f"Classifying {len(chunks)} chunks using schema")

        # Process chunks in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)")

            # Classify each chunk in the batch
            for chunk in batch:
                try:
                    self.classify_chunk(chunk)
                except Exception as e:
                    logger.error(f"Error classifying chunk {chunk.chunk_id}: {str(e)}")
                    # Add a default classification to avoid further errors
                    chunk.metadata["schema_classification"] = {
                        "entity_type": self.default_entity_type, # Use self.default_entity_type
                        "properties": {},
                        "confidence": 0.0
                    }

        # Identify relationships between chunks
        self.identify_relationships(chunks)

        return chunks

    def _build_classification_prompt(self, chunk: TextChunk) -> str:
        """
        Build a prompt for chunk classification.

        Args:
            chunk: The chunk to classify

        Returns:
            Prompt string for Gemini
        """
        # Construct a list of available entity types with their properties
        entity_type_descriptions = []
        for entity_type in self.entity_types:
            props = self.entity_properties.get(entity_type, [])
            props_str = ", ".join(props[:10])  # Limit to first 10 properties to avoid huge prompts
            if len(props) > 10:
                props_str += "..."
            entity_type_descriptions.append(f"- {entity_type}: properties = [{props_str}]")

        entity_types_info = "\n".join(entity_type_descriptions)

        # Add relationship types information
        relationship_descriptions = []
        for rel_type in self.relationship_types:
            domain, range_entities = self.schema_loader.get_relationship_domain_range(rel_type)
            domain_str = ", ".join(domain[:5]) if domain else "any"
            range_str = ", ".join(range_entities[:5]) if range_entities else "any"

            if len(domain) > 5:
                domain_str += "..."
            if len(range_entities) > 5:
                range_str += "..."

            relationship_descriptions.append(
                f"- {rel_type}: domain = [{domain_str}], range = [{range_str}]"
            )

        relationship_types_info = "\n".join(relationship_descriptions)

        prompt = f"""You are an expert knowledge graph entity extractor and classifier.

TASK:
1. Analyze the following text and extract entities
2. Classify entities according to the schema provided
3. Identify relationships between entities based on the relationship types in the schema

TEXT TO ANALYZE:
```
{chunk.text[:2000]}  # Limit to 2000 chars to avoid token issues
```

SCHEMA ENTITY TYPES:
{entity_types_info}

SCHEMA RELATIONSHIP TYPES:
{relationship_types_info}

INSTRUCTIONS:
1. Identify and extract entities from the text
2. For each entity:
   - Classify as one of the schema entity types OR suggest a new entity type if it doesn't fit
   - Extract the most relevant properties
   - Provide a confidence score (0.0-1.0)
3. Identify relationships between extracted entities
   - Use schema relationship types OR suggest new relationship types
   - Ensure relationships respect domain and range constraints where possible

FORMAT YOUR RESPONSE AS JSON:
{{
  "entities": [
    {{
      "entity_id": "unique_id_1",
      "text": "extracted text for entity",
      "entity_type": "SchemaEntityType",
      "is_new_type": false,
      "properties": {{
        "property1": "extracted value 1",
        "property2": "extracted value 2"
      }},
      "confidence": 0.85,
      "position": [start_char, end_char]
    }}
  ],
  "relationships": [
    {{
      "source_id": "unique_id_1",
      "target_id": "unique_id_2",
      "relationship_type": "SCHEMA_RELATIONSHIP_TYPE",
      "is_new_type": false,
      "properties": {{}},
      "confidence": 0.80
    }}
  ],
  "chunk_classification": {{
    "entity_type": "PrimaryEntityType",
    "properties": {{}},
    "confidence": 0.85
  }}
}}

IMPORTANT:
- Return ONLY the JSON object
- Choose the most specific entity types possible
- For entities that don't match schema types well, suggest new types with "is_new_type": true
- Similarly for relationships that don't match schema relationship types well
- Ensure relationships respect domain/range constraints when using schema relationship types
- If a text segment doesn't contain identifiable entities, the chunk_classification should use "{self.default_entity_type}" as the entity_type
- For suggested new entity or relationship types, provide higher confidence only when you're certain they represent a truly new concept
"""
        return prompt

    def _build_relationship_prompt(
        self,
        source_chunk: TextChunk,
        target_chunks: List[TextChunk],
        valid_rel_types: List[str],
        source_type: str = None,
        target_type: str = None
    ) -> str:
        """
        Build a prompt to identify relationships between chunks.

        Args:
            source_chunk: Source TextChunk object
            target_chunks: List of target TextChunk objects
            valid_rel_types: Valid relationship types
            source_type: Entity type of the source (optional)
            target_type: Entity type of the target (optional)

        Returns:
            Prompt string for Gemini
        """
        # Extract source text and type
        source_text = source_chunk.text
        if source_type is None:
            source_type = source_chunk.metadata.get("schema_classification", {}).get("entity_type", "Unknown")

        # Format target texts with indices
        formatted_targets = []
        for i, chunk in enumerate(target_chunks):
            # Get target text and type
            text = chunk.text
            chunk_target_type = target_type
            if chunk_target_type is None:
                chunk_target_type = chunk.metadata.get("schema_classification", {}).get("entity_type", "Unknown")

            # Limit text length to avoid huge prompts
            preview = text[:500] + ("..." if len(text) > 500 else "")
            formatted_targets.append(f"TARGET {i+1} ({chunk_target_type}):\n```\n{preview}\n```")

        targets_text = "\n\n".join(formatted_targets)
        rel_types_str = ", ".join(valid_rel_types)

        prompt = f"""You are an expert at identifying relationships between entities in a financial curriculum knowledge graph.

TASK:
Determine if the SOURCE text has any relationships with the TARGET texts.

SOURCE ({source_type}):
```
{source_text[:1000]}  # Limit to 1000 chars
```

{targets_text}

VALID RELATIONSHIP TYPES: {rel_types_str}

INSTRUCTIONS:
1. For each TARGET, determine if there is a relationship from the SOURCE to the TARGET
2. If a relationship exists, specify the relationship type from the list of valid types
3. Include a brief explanation of why this relationship exists

FORMAT YOUR RESPONSE AS JSON:
[
  {{
    "target_index": 1,  # The index of the target (1-based)
    "relationship_type": "RELATIONSHIP_NAME",
    "properties": {{}},
    "explanation": "Brief explanation of why this relationship exists"
  }},
  # Include additional relationships if found
]

IMPORTANT:
- Return ONLY the JSON array
- Only include targets where a relationship exists
- Use the exact relationship type names provided
- If no relationships exist, return an empty array []
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

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity using Levenshtein distance.

        Args:
            str1: First string to compare
            str2: Second string to compare

        Returns:
            Similarity score between 0 and 1
        """
        # Simple utility to normalize strings for comparison
        def normalize(s):
            return s.lower().strip()

        norm_str1 = normalize(str1)
        norm_str2 = normalize(str2)

        # Handle edge cases
        if not norm_str1 and not norm_str2:
            return 1.0
        if not norm_str1 or not norm_str2:
            return 0.0

        # Use Levenshtein distance for similarity
        try:
            from rapidfuzz.distance import Levenshtein
            max_len = max(len(norm_str1), len(norm_str2))
            if max_len == 0:
                return 1.0
            distance = Levenshtein.distance(norm_str1, norm_str2)
            similarity = 1.0 - (distance / max_len)
            return similarity
        except ImportError:
            # Fallback to basic comparison if rapidfuzz not available
            if norm_str1 == norm_str2:
                return 1.0

            common_chars = sum(1 for c in norm_str1 if c in norm_str2)
            avg_len = (len(norm_str1) + len(norm_str2)) / 2
            return common_chars / avg_len if avg_len > 0 else 0.0

    def _find_best_matching_entity_type(self, suggested_type: str) -> Tuple[str, float, bool]:
        """
        Find the best matching entity type in the schema for a suggested type.

        Args:
            suggested_type: The entity type suggested by the LLM

        Returns:
            Tuple containing:
                - The best matching entity type (or the original if no good match)
                - Similarity score
                - Whether this is a new entity type
        """
        best_match = suggested_type
        best_score = 0.0
        is_new_type = True

        # Check similarity against existing entity types
        for entity_type in self.entity_types:
            similarity = self._calculate_string_similarity(suggested_type, entity_type)
            # Update threshold usage:
            if similarity > best_score and similarity >= self.config.get('entity_string_similarity_threshold', settings.ENTITY_STRING_SIMILARITY_THRESHOLD):
                best_match = entity_type
                best_score = similarity
                is_new_type = False

        # If still a new type but very similar to existing, prefix it
        if is_new_type:
            # Update prefix usage:
            tentative_prefix = self.config.get('tentative_entity_prefix', settings.TENTATIVE_ENTITY_PREFIX)
            return f"{tentative_prefix}{suggested_type}", best_score, is_new_type

        return best_match, best_score, is_new_type

    def _find_best_matching_relationship_type(self, suggested_type: str) -> Tuple[str, float, bool]:
        """
        Find the best matching relationship type in the schema for a suggested type.

        Args:
            suggested_type: The relationship type suggested by the LLM

        Returns:
            Tuple containing:
                - The best matching relationship type (or the original if no good match)
                - Similarity score
                - Whether this is a new relationship type
        """
        best_match = suggested_type
        best_score = 0.0
        is_new_type = True

        # Check similarity against existing relationship types
        for rel_type in self.relationship_types:
            similarity = self._calculate_string_similarity(suggested_type, rel_type)
            # Update threshold usage:
            if similarity > best_score and similarity >= self.config.get('entity_string_similarity_threshold', settings.ENTITY_STRING_SIMILARITY_THRESHOLD):
                best_match = rel_type
                best_score = similarity
                is_new_type = False

        # If still a new type but very similar to existing, prefix it
        if is_new_type:
            # Update prefix usage:
            tentative_prefix = self.config.get('tentative_relationship_prefix', settings.TENTATIVE_RELATIONSHIP_PREFIX)
            return f"{tentative_prefix}{suggested_type}", best_score, is_new_type

        return best_match, best_score, is_new_type

    def _parse_classification_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini's classification response.

        Args:
            response_text: Gemini's response

        Returns:
            Parsed classification dict
        """
        try:
            # Extract JSON from the response (Gemini sometimes adds markdown formatting)
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()

            # Parse the JSON
            response = json.loads(json_text)

            # Extract the chunk classification from the response
            classification = response.get("chunk_classification", {})

            # If there's no chunk classification but there are entities, use the first entity
            if not classification and "entities" in response and response["entities"]:
                primary_entity = response["entities"][0]
                classification = {
                    "entity_type": primary_entity.get("entity_type", self.default_entity_type), # Use self.default_entity_type
                    "properties": primary_entity.get("properties", {}),
                    "confidence": primary_entity.get("confidence", 0.5)
                }

            # Ensure required fields are present
            if "entity_type" not in classification:
                logger.warning(f"Missing entity_type in classification response, defaulting to '{self.default_entity_type}'") # Use self.default_entity_type
                classification["entity_type"] = self.default_entity_type # Use self.default_entity_type

            if "properties" not in classification:
                classification["properties"] = {}

            if "confidence" not in classification:
                classification["confidence"] = 0.5

            # Check if this entity type is new and find closest match
            if classification.get("entity_type") not in self.entity_types:
                matched_type, similarity, is_new = self._find_best_matching_entity_type(
                    classification.get("entity_type", "")
                )

                # If sufficient similarity, use the existing type
                if not is_new:
                    logger.info(f"Mapped suggested entity type '{classification['entity_type']}' to existing type '{matched_type}' with similarity {similarity:.2f}")
                    classification["entity_type"] = matched_type
                    # Adjust confidence based on similarity
                    classification["confidence"] = classification.get("confidence", 0.5) * similarity
                else:
                    # New type, store with tentative prefix
                    logger.info(f"Treating '{classification['entity_type']}' as a new entity type with prefix")
                    classification["entity_type"] = matched_type
                    # New type, store with tentative prefix
                    logger.info(f"Treating '{classification['entity_type']}' as a new entity type with prefix")
                    classification["entity_type"] = matched_type
                    classification["is_new_type"] = True

            # Store extracted entities and relationships for later use
            classification["extracted_entities"] = response.get("entities", [])
            classification["extracted_relationships"] = response.get("relationships", [])

            # Process relationships for schema compliance
            tentative_rel_prefix = self.config.get('tentative_relationship_prefix', settings.TENTATIVE_RELATIONSHIP_PREFIX) # Get prefix from config
            for rel in classification.get("extracted_relationships", []):
                if rel.get("relationship_type") not in self.relationship_types:
                    matched_type, similarity, is_new = self._find_best_matching_relationship_type(
                        rel.get("relationship_type", "") # This method now uses config for prefix/threshold
                    )

                    if not is_new:
                        logger.info(f"Mapped relationship type '{rel['relationship_type']}' to existing type '{matched_type}' with similarity {similarity:.2f}")
                        rel["relationship_type"] = matched_type
                        rel["confidence"] = rel.get("confidence", 0.5) * similarity
                    else:
                        logger.info(f"Treating '{rel['relationship_type']}' as a new relationship type with prefix")
                        rel["relationship_type"] = matched_type
                        logger.info(f"Treating '{rel['relationship_type']}' as a new relationship type with prefix")
                        rel["relationship_type"] = matched_type
                        rel["is_new_type"] = True

                # Check domain/range constraints
                if not rel.get("is_new_type", False):
                    source_id = rel.get("source_id")
                    target_id = rel.get("target_id")
                    rel_type = rel.get("relationship_type")

                    # Find entity types for source and target
                    source_entity = next((e for e in response.get("entities", []) if e.get("entity_id") == source_id), None)
                    target_entity = next((e for e in response.get("entities", []) if e.get("entity_id") == target_id), None)

                    if source_entity and target_entity:
                        source_type = source_entity.get("entity_type")
                        target_type = target_entity.get("entity_type")

                        # Get domain/range constraints
                        domain, range_entities = self.schema_loader.get_relationship_domain_range(rel_type)

                        # Check if relationship violates constraints
                        if (domain and source_type not in domain) or (range_entities and target_type not in range_entities):
                            logger.warning(f"Relationship {rel_type} violates domain/range constraints for {source_type} -> {target_type}")
                            # Mark as tentative if constraints violated
                            rel["relationship_type"] = f"{tentative_rel_prefix}{rel_type}" # Use prefix from config
                            rel["is_new_type"] = True
                            rel["constraint_violation"] = True
                            rel["confidence"] = rel.get("confidence", 0.5) * 0.8  # Reduce confidence

            return classification

        except Exception as e:
            logger.error(f"Error parsing classification response: {str(e)}")
            logger.debug(f"Raw response: {response_text}")
            # Return a default classification
            return {
                "entity_type": self.default_entity_type, # Use self.default_entity_type
                "properties": {},
                "confidence": 0.0,
                "extracted_entities": [],
                "extracted_relationships": []
            }

    def _parse_relationship_response(
        self,
        response_text: str,
        source_id: str,
        target_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Parse Gemini's relationship identification response.

        Args:
            response_text: Gemini's response
            source_id: ID of the source chunk
            target_ids: List of target chunk IDs

        Returns:
            List of relationship dictionaries
        """
        relationships = []

        try:
            # Extract JSON from the response
            json_text = response_text
            if "```json" in response_text:
                json_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_text = response_text.split("```")[1].strip()

            # Parse the JSON
            rel_results = json.loads(json_text)

            # Convert to relationship format
            for rel in rel_results:
                target_index = rel.get("target_index")
                if target_index is None or not isinstance(target_index, int) or target_index < 1 or target_index > len(target_ids):
                    continue

                # Convert 1-based to 0-based index
                idx = target_index - 1

                relationship = {
                    "source_id": source_id,
                    "target_id": target_ids[idx],
                    # Update default relationship type usage:
                    "relationship_type": rel.get("relationship_type", self.config.get('default_relationship_type', settings.DEFAULT_RELATIONSHIP_TYPE)),
                    "properties": rel.get("properties", {})
                }

                # Add explanation as a property if available
                if "explanation" in rel:
                    relationship["properties"]["explanation"] = rel["explanation"]

                # Only add if it has the required fields
                if relationship["target_id"]:
                    relationships.append(relationship)

        except Exception as e:
            logger.error(f"Error parsing relationship response: {str(e)}")
            logger.debug(f"Raw response: {response_text}")

        return relationships
