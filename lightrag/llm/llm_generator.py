"""
LLM Generator module for Chain-of-Thought reasoning and enhanced citation handling.
"""

import re
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable, Awaitable

from lightrag.prompt import PROMPTS
from lightrag.config_loader import get_enhanced_config
from lightrag.llm.placeholder_resolver import resolve_placeholders_in_context

# Set up logger
logger = logging.getLogger(__name__)


class LLMGenerationError(Exception):
    """Exception raised for errors in LLM generation."""
    pass


def extract_reasoning_and_answer(response: str) -> Dict[str, str]:
    """
    Extract reasoning and answer from a CoT response.

    Args:
        response: The response from the LLM with reasoning and answer

    Returns:
        Dict with 'reasoning' and 'answer' keys
    """
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)

    # Extract and normalize whitespace in reasoning
    if reasoning_match:
        reasoning_text = reasoning_match.group(1)
        # Normalize whitespace by replacing multiple spaces with a single space
        reasoning_text = re.sub(r'\s+', ' ', reasoning_text)
        # Normalize newlines by replacing multiple newlines with a single newline
        reasoning_text = re.sub(r'\n\s*\n', '\n', reasoning_text)
        reasoning = reasoning_text.strip()
    else:
        reasoning = ""

    # Extract and normalize whitespace in answer
    if answer_match:
        answer_text = answer_match.group(1)
        # Normalize whitespace by replacing multiple spaces with a single space
        answer_text = re.sub(r'\s+', ' ', answer_text)
        # Normalize newlines by replacing multiple newlines with a single newline
        answer_text = re.sub(r'\n\s*\n', '\n', answer_text)
        answer = answer_text.strip()
    else:
        answer = response

    return {
        "reasoning": reasoning,
        "answer": answer
    }


def process_citations(response: str, context: Optional[List[Dict[str, Any]]]) -> str:
    """
    Process citations in a response.

    Args:
        response: The response from the LLM with citations
        context: The context items used for generation

    Returns:
        Response with processed citations
    """
    # Check if context is None or empty
    if context is None or len(context) == 0:
        return response

    # Check if there are any entity citations (case-insensitive and whitespace-tolerant)
    entity_citation_pattern = r"\[Entity\s*ID:?\s*(.*?)\]"
    entity_citations_matches = re.findall(entity_citation_pattern, response, re.IGNORECASE)
    entity_citations = [match.strip() for match in entity_citations_matches]

    # Check if there are any diagram citations (case-insensitive and whitespace-tolerant)
    diagram_citation_pattern = r"\[Diagram\s*ID:?\s*(.*?)\]"
    diagram_citations_matches = re.findall(diagram_citation_pattern, response, re.IGNORECASE)
    diagram_citations = [match.strip() for match in diagram_citations_matches]

    # Check if there are any formula citations (case-insensitive and whitespace-tolerant)
    formula_citation_pattern = r"\[Formula\s*ID:?\s*(.*?)\]"
    formula_citations_matches = re.findall(formula_citation_pattern, response, re.IGNORECASE)
    formula_citations = [match.strip() for match in formula_citations_matches]

    # If no citations of any kind, return the original response
    if not entity_citations and not diagram_citations and not formula_citations:
        return response

    # Create a case-insensitive mapping of entity IDs to context items
    context_map = {}
    for item in context:
        if "id" in item:
            context_map[item["id"].lower()] = item

    # Create case-insensitive mappings for diagrams and formulas
    diagram_map = {}
    formula_map = {}

    # Extract diagrams and formulas from context items
    for item in context:
        if "extracted_elements" in item:
            # Process diagrams
            if "diagrams" in item["extracted_elements"]:
                for diagram in item["extracted_elements"]["diagrams"]:
                    if "diagram_id" in diagram:
                        diagram_map[diagram["diagram_id"].lower()] = diagram

            # Process formulas
            if "formulas" in item["extracted_elements"]:
                for formula in item["extracted_elements"]["formulas"]:
                    if "formula_id" in formula:
                        formula_map[formula["formula_id"].lower()] = formula

    # Replace citations with numbered references
    numbered_response = response
    sources = []
    citation_index = 1
    
    # Keep track of processed citations to avoid duplicates
    processed_entity_ids = set()
    processed_diagram_ids = set()
    processed_formula_ids = set()
    
    # Create a mapping from original IDs to citation indices
    citation_map = {}

    # Process entity citations
    for entity_id in entity_citations:
        entity_id_lower = entity_id.lower()
        if entity_id_lower in context_map and entity_id_lower not in processed_entity_ids:
            # Add the entity ID to the processed set
            processed_entity_ids.add(entity_id_lower)
            
            # Store the citation index for this entity ID
            citation_map[entity_id_lower] = citation_index
            
            # Add the source to the sources list
            item = context_map[entity_id_lower]
            source_content = item.get("content", "")
            if not source_content and "text" in item:
                source_content = item["text"]
            if not source_content and "properties" in item and "name" in item["properties"]:
                source_content = item["properties"]["name"]
            sources.append(f"{citation_index}. {source_content}")
            citation_index += 1

    # Process diagram citations
    for diagram_id in diagram_citations:
        diagram_id_lower = diagram_id.lower()
        if diagram_id_lower in diagram_map and diagram_id_lower not in processed_diagram_ids:
            # Add the diagram ID to the processed set
            processed_diagram_ids.add(diagram_id_lower)
            
            # Store the citation index for this diagram ID
            citation_map[f"diagram:{diagram_id_lower}"] = citation_index
            
            # Add the source to the sources list
            diagram = diagram_map[diagram_id_lower]
            description = diagram.get("description", "Diagram")
            caption = diagram.get("caption", "")
            source_text = f"Diagram: {description}"
            if caption:
                source_text += f" - {caption}"
            sources.append(f"{citation_index}. {source_text}")
            citation_index += 1

    # Process formula citations
    for formula_id in formula_citations:
        formula_id_lower = formula_id.lower()
        if formula_id_lower in formula_map and formula_id_lower not in processed_formula_ids:
            # Add the formula ID to the processed set
            processed_formula_ids.add(formula_id_lower)
            
            # Store the citation index for this formula ID
            citation_map[f"formula:{formula_id_lower}"] = citation_index
            
            # Add the source to the sources list
            formula = formula_map[formula_id_lower]
            formula_text = formula.get("formula", "")
            description = formula.get("description", "")
            source_text = f"Formula: {formula_text}"
            if description:
                source_text += f" - {description}"
            sources.append(f"{citation_index}. {source_text}")
            citation_index += 1
    
    # Now replace citations with their numbered references
    for entity_id in entity_citations:
        entity_id_lower = entity_id.lower()
        if entity_id_lower in citation_map:
            # Find the exact citation pattern in the response
            pattern = re.compile(r"\[Entity\s*ID:?\s*" + re.escape(entity_id) + r"\]", re.IGNORECASE)
            # Replace all occurrences with the numbered reference
            numbered_response = pattern.sub(f"[{citation_map[entity_id_lower]}]", numbered_response)
    
    for diagram_id in diagram_citations:
        diagram_id_lower = diagram_id.lower()
        if f"diagram:{diagram_id_lower}" in citation_map:
            # Find the exact citation pattern in the response
            pattern = re.compile(r"\[Diagram\s*ID:?\s*" + re.escape(diagram_id) + r"\]", re.IGNORECASE)
            # Replace all occurrences with the numbered reference
            numbered_response = pattern.sub(f"[{citation_map[f'diagram:{diagram_id_lower}']}]", numbered_response)
    
    for formula_id in formula_citations:
        formula_id_lower = formula_id.lower()
        if f"formula:{formula_id_lower}" in citation_map:
            # Find the exact citation pattern in the response
            pattern = re.compile(r"\[Formula\s*ID:?\s*" + re.escape(formula_id) + r"\]", re.IGNORECASE)
            # Replace all occurrences with the numbered reference
            numbered_response = pattern.sub(f"[{citation_map[f'formula:{formula_id_lower}']}]", numbered_response)

    # Add sources section if there are valid sources
    if sources:
        numbered_response += "\n\nSources:\n" + "\n".join(sources)

    return numbered_response


class LLMGenerator:
    """
    LLM Generator for Chain-of-Thought reasoning and enhanced citation handling.
    """

    def __init__(self, llm_provider: str, llm_func: Callable[[str, Optional[str], Optional[List], Optional[bool], Any], Awaitable[str]]):
        """
        Initialize the LLM Generator.

        Args:
            llm_provider: The LLM provider (e.g., 'openai', 'google')
            llm_func: The LLM function to use for generation
        """
        self.llm_provider = llm_provider
        self.llm_func = llm_func
        self.config = get_enhanced_config()

    def create_cot_prompt(self, query: str, context: Optional[str] = None) -> str:
        """
        Create a Chain-of-Thought prompt.

        Args:
            query: The query to generate a response for
            context: Optional context to include in the prompt

        Returns:
            str: The CoT prompt
        """
        prompt = PROMPTS["cot_rag_response"]

        # Replace placeholders
        prompt = prompt.replace("{query}", query)
        prompt = prompt.replace("{content_data}", context or "No context provided.")

        return prompt

    async def generate_with_cot(self, query: str, context: Optional[str] = None, context_items: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response using chain-of-thought reasoning.

        Args:
            query: The query to generate a response for
            context: Optional context to include in the prompt
            context_items: Optional list of context items with extracted elements

        Returns:
            str: Generated response with chain-of-thought reasoning
        """
        # If context_items are provided, resolve placeholders
        if context_items:
            # Determine output format based on configuration
            output_format = getattr(self.config, "placeholder_output_format", "detailed")
            logger.debug(f"Using {output_format} format for placeholder resolution in generate_with_cot")

            # Resolve placeholders with the specified format
            resolved_items = resolve_placeholders_in_context(
                context_items,
                output_format=output_format
            )

            # Format the resolved items into a string if context is not provided
            if not context:
                # Simple formatting for demonstration - in practice, use a more sophisticated formatter
                context = "\n\n".join([item.get("content", item.get("text", "")) for item in resolved_items])

        prompt = self.create_cot_prompt(query, context)

        try:
            response = await self.llm_func(prompt)
            # If context_items are provided, process citations
            if context_items:
                response = process_citations(response, context_items)
            return response
        except Exception as e:
            logger.error(f"Error generating CoT response: {str(e)}")
            raise LLMGenerationError(f"Failed to generate CoT response: {str(e)}")

    async def _should_refine_reasoning(self, response: str) -> bool:
        """
        Determine if the reasoning should be refined.

        Args:
            response: The response to check

        Returns:
            bool: True if the reasoning should be refined, False otherwise
        """
        result = extract_reasoning_and_answer(response)

        # Check if reasoning is too short
        if len(result["reasoning"]) < 50:
            return True

        # Check if reasoning doesn't contain citations (case-insensitive)
        if self.config.enable_enhanced_citations:
            # Use regex to find entity citations with case-insensitivity
            entity_citation_pattern = r"\[Entity\s*ID:?\s*(.*?)\]"
            entity_citations = re.findall(entity_citation_pattern, result["reasoning"], re.IGNORECASE)

            if not entity_citations:
                return True

        return False

    async def generate_with_cot_refinement(self, query: str, context: Optional[str] = None, context_items: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate a response with CoT refinement loop.

        Args:
            query: The query to generate a response for
            context: Optional context to include in the prompt
            context_items: Optional list of context items with extracted elements

        Returns:
            str: Generated response with refined reasoning
        """
        response = await self.generate_with_cot(query, context, context_items)
        attempts = 1

        while attempts < self.config.max_cot_refinement_attempts:
            should_refine = await self._should_refine_reasoning(response)
            if not should_refine:
                break

            logger.info(f"Refining CoT response (attempt {attempts+1}/{self.config.max_cot_refinement_attempts})")

            # Create a refined prompt
            prompt = self.create_cot_prompt(query, context)

            # Add specific instructions for diagrams and formulas if context_items contain them
            has_diagrams = False
            has_formulas = False

            if context_items:
                for item in context_items:
                    if "extracted_elements" in item:
                        if "diagrams" in item["extracted_elements"] and item["extracted_elements"]["diagrams"]:
                            has_diagrams = True
                        if "formulas" in item["extracted_elements"] and item["extracted_elements"]["formulas"]:
                            has_formulas = True

            # Add refinement instructions
            refinement_instructions = "\n\nYour previous response had insufficient reasoning. Please provide more detailed reasoning with explicit citations to the knowledge base."

            if has_diagrams:
                refinement_instructions += "\n\nMake sure to reference any relevant diagrams using [Diagram ID: X] format in your reasoning."

            if has_formulas:
                refinement_instructions += "\n\nMake sure to reference any relevant formulas using [Formula ID: X] format in your reasoning."

            prompt += refinement_instructions

            try:
                response = await self.llm_func(prompt)
                # Process citations if context_items are provided
                if context_items:
                    response = process_citations(response, context_items)
            except Exception as e:
                logger.error(f"Error refining CoT response: {str(e)}")
                # Return the last successful response
                break

            attempts += 1

        return response