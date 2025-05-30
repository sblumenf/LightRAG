"""
Advanced Generation module for LightRAG.

This module provides advanced generation capabilities for LightRAG, including:
1. Chain-of-Thought (CoT) reasoning
2. Enhanced citation handling
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Callable, Awaitable, Union, Tuple

from lightrag.config_loader import get_enhanced_config
from lightrag.llm.llm_generator import LLMGenerator, extract_reasoning_and_answer, process_citations
from lightrag.llm.placeholder_resolver import resolve_placeholders_in_context

# Set up logger
logger = logging.getLogger(__name__)


class AdvancedGenerationManager:
    """
    Manager for advanced generation capabilities in LightRAG.
    """

    def __init__(self, llm_provider: str, llm_func: Callable[[str, Optional[str], Optional[List], Optional[bool], Any], Awaitable[str]]):
        """
        Initialize the Advanced Generation Manager.

        Args:
            llm_provider: The LLM provider (e.g., 'openai', 'google')
            llm_func: The LLM function to use for generation
        """
        self.config = get_enhanced_config()
        self.llm_generator = LLMGenerator(llm_provider, llm_func)

    async def generate_response(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generate a response using advanced generation capabilities.

        Args:
            query: The query to generate a response for
            context: The context items to use for generation

        Returns:
            str: The generated response
        """
        # Format context for the LLM
        formatted_context = self._format_context(context)

        # Use Chain-of-Thought if enabled
        if self.config.enable_cot:
            logger.info("Using Chain-of-Thought for response generation")
            response = await self._generate_with_cot(query, formatted_context, context)
        else:
            # Use standard generation
            logger.info("Using standard response generation")
            response = await self.llm_generator.llm_func(
                f"Query: {query}\n\nContext: {formatted_context}\n\nPlease provide a comprehensive answer based on the context."
            )

        return response

    async def _generate_with_cot(self, query: str, formatted_context: str, raw_context: List[Dict[str, Any]]) -> str:
        """
        Generate a response using Chain-of-Thought reasoning.

        Args:
            query: The query to generate a response for
            formatted_context: The formatted context to include in the prompt
            raw_context: The raw context items for citation processing

        Returns:
            str: The generated response with chain-of-thought reasoning
        """
        # Resolve placeholders in context if enabled
        if self.config.enable_diagram_formula_integration and self.config.resolve_placeholders_in_context:
            logger.info("Resolving diagram and formula placeholders in context")

            # Determine output format based on configuration
            output_format = getattr(self.config, "placeholder_output_format", "detailed")
            logger.debug(f"Using {output_format} format for placeholder resolution")

            # Resolve placeholders with the specified format
            resolved_context = resolve_placeholders_in_context(
                raw_context,
                output_format=output_format
            )

            # Use the resolved context for generation
            cot_response = await self.llm_generator.generate_with_cot_refinement(
                query,
                formatted_context,
                context_items=resolved_context
            )
        else:
            # Generate response with CoT refinement using original context
            cot_response = await self.llm_generator.generate_with_cot_refinement(
                query,
                formatted_context,
                context_items=raw_context
            )

        # Extract reasoning and answer
        result = extract_reasoning_and_answer(cot_response)

        # Process citations if enabled
        if self.config.enable_enhanced_citations:
            logger.info("Processing citations in the response")
            result["reasoning"] = process_citations(result["reasoning"], raw_context)
            result["answer"] = process_citations(result["answer"], raw_context)

        # Format the final response
        final_response = f"## Reasoning\n\n{result['reasoning']}\n\n## Answer\n\n{result['answer']}"

        return final_response

    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format context items for the LLM.

        Args:
            context: The context items to format

        Returns:
            str: The formatted context
        """
        formatted_items = []

        for i, item in enumerate(context, 1):
            # Format based on item type
            if "entity_type" in item:
                # Knowledge Graph entity
                formatted_items.append(
                    f"[{i}] Entity ID: {item.get('id', 'unknown')}\n"
                    f"Type: {item.get('entity_type', 'unknown')}\n"
                    f"Name: {item.get('name', 'unknown')}\n"
                    f"Description: {item.get('description', 'No description')}\n"
                )
            elif "relationship_type" in item:
                # Knowledge Graph relationship
                formatted_items.append(
                    f"[{i}] Relationship ID: {item.get('id', 'unknown')}\n"
                    f"Type: {item.get('relationship_type', 'unknown')}\n"
                    f"Source: {item.get('source_id', 'unknown')}\n"
                    f"Target: {item.get('target_id', 'unknown')}\n"
                    f"Description: {item.get('description', 'No description')}\n"
                )
            else:
                # Vector store item
                content = item.get('content', 'No content')
                source = item.get('source', 'unknown')
                file_path = item.get('file_path', '')

                # Check for extracted elements
                has_diagrams = False
                has_formulas = False
                if 'extracted_elements' in item:
                    if 'diagrams' in item['extracted_elements'] and item['extracted_elements']['diagrams']:
                        has_diagrams = True
                    if 'formulas' in item['extracted_elements'] and item['extracted_elements']['formulas']:
                        has_formulas = True

                # Build the formatted item
                formatted_item = f"[{i}] Document ID: {item.get('id', 'unknown')}\n"

                if file_path:
                    formatted_item += f"File: {file_path}\n"

                formatted_item += f"Source: {source}\n"

                # Add diagram/formula information if present
                if has_diagrams or has_formulas:
                    elements_info = []
                    if has_diagrams:
                        diagram_count = len(item['extracted_elements']['diagrams'])
                        elements_info.append(f"{diagram_count} diagram(s)")
                    if has_formulas:
                        formula_count = len(item['extracted_elements']['formulas'])
                        elements_info.append(f"{formula_count} formula(s)")

                    if elements_info:
                        formatted_item += f"Contains: {', '.join(elements_info)}\n"

                formatted_item += f"Content: {content}\n"
                formatted_items.append(formatted_item)

        return "\n".join(formatted_items)
