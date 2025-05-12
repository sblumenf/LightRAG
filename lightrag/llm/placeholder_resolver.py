"""
Placeholder Resolver for LightRAG.

This module provides functionality to resolve placeholders in text content,
particularly for diagrams and formulas extracted during document processing.
"""

import logging
import re
import time
from typing import Dict, Any, List, Optional, Tuple, Union, Literal
from functools import lru_cache

from lightrag.config_loader import get_enhanced_config

# Set up logger
logger = logging.getLogger(__name__)


class PlaceholderResolver:
    """
    Resolver for placeholders in text content.

    This class handles the resolution of placeholders for non-text elements
    like diagrams and formulas, replacing them with appropriate descriptions
    or representations for LLM processing.
    """

    def __init__(self, output_format: Literal["detailed", "concise"] = "detailed"):
        """
        Initialize the PlaceholderResolver.

        Args:
            output_format: Format for resolved placeholders.
                           "detailed" provides full descriptions,
                           "concise" provides minimal information.
        """
        # Placeholder patterns
        self.diagram_pattern = r'\[DIAGRAM-([a-zA-Z0-9-]+)\]'
        self.formula_pattern = r'\[FORMULA-([a-zA-Z0-9-]+)\]'

        # Get configuration
        self.config = get_enhanced_config()

        # Set output format
        self.output_format = output_format

        # Citation formats
        self.diagram_citation_format = self.config.diagram_citation_format or "[Diagram ID: {id}]"
        self.formula_citation_format = self.config.formula_citation_format or "[Formula ID: {id}]"

    def resolve_placeholders(
        self,
        text: str,
        extracted_elements: Dict[str, Any]
    ) -> str:
        """
        Resolve placeholders in text with their descriptions.

        Args:
            text: Text content with placeholders
            extracted_elements: Dictionary of extracted elements

        Returns:
            str: Text with placeholders replaced by descriptions
        """
        if not text:
            logger.debug("Empty text provided to resolve_placeholders, returning as is")
            return text

        if not extracted_elements:
            logger.debug("No extracted elements provided to resolve_placeholders, returning text as is")
            return text

        try:
            # Process diagram placeholders
            if 'diagrams' in extracted_elements:
                text = self._resolve_diagram_placeholders(text, extracted_elements.get('diagrams', []))

            # Process formula placeholders
            if 'formulas' in extracted_elements:
                text = self._resolve_formula_placeholders(text, extracted_elements.get('formulas', []))

            return text
        except Exception as e:
            logger.error(f"Error resolving placeholders: {str(e)}")
            # Return original text in case of error
            return text

    def _resolve_diagram_placeholders(
        self,
        text: str,
        diagrams: List[Dict[str, Any]]
    ) -> str:
        """
        Resolve diagram placeholders in text.

        Args:
            text: Text content with diagram placeholders
            diagrams: List of diagram data

        Returns:
            str: Text with diagram placeholders replaced
        """
        if not text or not diagrams:
            return text

        start_time = time.time()

        # Create a mapping of diagram IDs to diagram data
        diagram_map = {diagram['diagram_id']: diagram for diagram in diagrams}

        # Find all diagram placeholders
        matches = re.finditer(self.diagram_pattern, text)

        # Process matches from end to start to avoid position shifts
        replacements = []
        for match in matches:
            diagram_id = match.group(1)
            if diagram_id in diagram_map:
                diagram = diagram_map[diagram_id]
                replacement = self._format_diagram_from_dict(diagram)
                replacements.append((match.span(), replacement))

        # Apply replacements from end to start
        for (start, end), replacement in sorted(replacements, key=lambda x: x[0][0], reverse=True):
            text = text[:start] + replacement + text[end:]

        end_time = time.time()
        if replacements:
            logger.debug(f"Resolved {len(replacements)} diagram placeholders in {end_time - start_time:.4f} seconds")

        return text

    def _resolve_formula_placeholders(
        self,
        text: str,
        formulas: List[Dict[str, Any]]
    ) -> str:
        """
        Resolve formula placeholders in text.

        Args:
            text: Text content with formula placeholders
            formulas: List of formula data

        Returns:
            str: Text with formula placeholders replaced
        """
        if not text or not formulas:
            return text

        start_time = time.time()

        # Create a mapping of formula IDs to formula data
        formula_map = {formula['formula_id']: formula for formula in formulas}

        # Find all formula placeholders
        matches = re.finditer(self.formula_pattern, text)

        # Process matches from end to start to avoid position shifts
        replacements = []
        for match in matches:
            formula_id = match.group(1)
            if formula_id in formula_map:
                formula = formula_map[formula_id]
                replacement = self._format_formula_from_dict(formula)
                replacements.append((match.span(), replacement))

        # Apply replacements from end to start
        for (start, end), replacement in sorted(replacements, key=lambda x: x[0][0], reverse=True):
            text = text[:start] + replacement + text[end:]

        end_time = time.time()
        if replacements:
            logger.debug(f"Resolved {len(replacements)} formula placeholders in {end_time - start_time:.4f} seconds")

        return text

    @lru_cache(maxsize=128)
    def _format_diagram_description(self, diagram_id: str, caption: str, description: str, page: str, diagram_type: str) -> str:
        """
        Format a diagram description for inclusion in text.

        Args:
            diagram_id: ID of the diagram
            caption: Caption of the diagram
            description: Description of the diagram
            page: Page number of the diagram
            diagram_type: Type of the diagram

        Returns:
            str: Formatted diagram description
        """
        # Generate citation format for reference
        citation = self.diagram_citation_format.format(id=diagram_id)

        if self.output_format == "concise":
            # Concise format - just the essential information
            if description:
                return f"[DIAGRAM: {description}] {citation}"
            elif caption:
                return f"[DIAGRAM: {caption}] {citation}"
            else:
                return f"[DIAGRAM on page {page}] {citation}"
        else:
            # Detailed format - all available information
            # Start with a header
            result = f"[DIAGRAM: {diagram_id}]"

            # Add caption if available
            if caption:
                result += f"\nCaption: {caption}"

            # Add description if available
            if description:
                result += f"\nDescription: {description}"
            else:
                # Basic description if none is available
                result += f"\nDescription: Diagram on page {page}"

            # Add page number
            result += f"\nPage: {page}"

            # Add diagram type if available
            if diagram_type and diagram_type != 'general':
                result += f"\nType: {diagram_type}"

            # Add citation reference
            result += f"\nReference: {citation}"

            return result

    def _format_diagram_from_dict(self, diagram: Dict[str, Any]) -> str:
        """
        Format a diagram description from a dictionary.

        Args:
            diagram: Diagram data

        Returns:
            str: Formatted diagram description
        """
        # Extract values with defaults
        diagram_id = diagram.get('diagram_id', 'unknown')
        caption = diagram.get('caption', '')
        description = diagram.get('description', '')
        page = str(diagram.get('page', 'unknown'))
        diagram_type = diagram.get('diagram_type', '')

        # Use the cached method
        return self._format_diagram_description(diagram_id, caption, description, page, diagram_type)

    @lru_cache(maxsize=128)
    def _format_formula_description(self, formula_id: str, formula_text: str, textual_representation: str, description: str, latex: str) -> str:
        """
        Format a formula description for inclusion in text.

        Args:
            formula_id: ID of the formula
            formula_text: Text of the formula
            textual_representation: Textual representation of the formula
            description: Description of the formula
            latex: LaTeX representation of the formula

        Returns:
            str: Formatted formula description
        """
        # Generate citation format for reference
        citation = self.formula_citation_format.format(id=formula_id)

        if self.output_format == "concise":
            # Concise format - just the essential information
            if formula_text:
                return f"[FORMULA: {formula_text}] {citation}"
            elif textual_representation:
                return f"[FORMULA: {textual_representation}] {citation}"
            else:
                return f"[FORMULA: {formula_id}] {citation}"
        else:
            # Detailed format - all available information
            # Start with a header
            result = f"[FORMULA: {formula_id}]"

            # Add the formula text
            result += f"\nFormula: {formula_text}"

            # Add textual representation if available
            if textual_representation:
                result += f"\nTextual representation: {textual_representation}"

            # Add description if available
            if description:
                result += f"\nDescription: {description}"

            # Add LaTeX if available
            if latex:
                result += f"\nLaTeX: {latex}"

            # Add citation reference
            result += f"\nReference: {citation}"

            return result

    def _format_formula_from_dict(self, formula: Dict[str, Any]) -> str:
        """
        Format a formula description from a dictionary.

        Args:
            formula: Formula data

        Returns:
            str: Formatted formula description
        """
        # Extract values with defaults
        formula_id = formula.get('formula_id', 'unknown')
        formula_text = formula.get('formula', '')
        textual_representation = formula.get('textual_representation', '')
        description = formula.get('description', '')
        latex = formula.get('latex', '')

        # Use the cached method
        return self._format_formula_description(formula_id, formula_text, textual_representation, description, latex)


def resolve_placeholders_in_context(
    context_items: List[Dict[str, Any]],
    output_format: Literal["detailed", "concise"] = "detailed"
) -> List[Dict[str, Any]]:
    """
    Resolve placeholders in context items.

    This function processes a list of context items, resolving any placeholders
    in their content using the extracted_elements data.

    Args:
        context_items: List of context items
        output_format: Format for resolved placeholders.
                      "detailed" provides full descriptions,
                      "concise" provides minimal information.

    Returns:
        List[Dict[str, Any]]: Context items with resolved placeholders
    """
    if not context_items:
        logger.debug("Empty context_items provided to resolve_placeholders_in_context, returning as is")
        return context_items

    try:
        # Create resolver with specified output format
        resolver = PlaceholderResolver(output_format=output_format)
        resolved_items = []

        # Track statistics for logging
        total_items = len(context_items)
        items_with_placeholders = 0
        total_placeholders_resolved = 0

        for item in context_items:
            try:
                # Create a copy of the item to avoid modifying the original
                resolved_item = item.copy()
                placeholders_found = False

                # Check if the item has content and extracted_elements
                if 'content' in item and 'extracted_elements' in item:
                    original_content = item['content']
                    resolved_content = resolver.resolve_placeholders(
                        original_content,
                        item['extracted_elements']
                    )

                    # Check if any placeholders were resolved
                    if original_content != resolved_content:
                        placeholders_found = True
                        # Count approximately how many placeholders were resolved
                        total_placeholders_resolved += (
                            original_content.count('[DIAGRAM-') +
                            original_content.count('[FORMULA-')
                        )

                    resolved_item['content'] = resolved_content

                # Also check for text field which might contain content
                elif 'text' in item and 'extracted_elements' in item:
                    original_text = item['text']
                    resolved_text = resolver.resolve_placeholders(
                        original_text,
                        item['extracted_elements']
                    )

                    # Check if any placeholders were resolved
                    if original_text != resolved_text:
                        placeholders_found = True
                        # Count approximately how many placeholders were resolved
                        total_placeholders_resolved += (
                            original_text.count('[DIAGRAM-') +
                            original_text.count('[FORMULA-')
                        )

                    resolved_item['text'] = resolved_text

                if placeholders_found:
                    items_with_placeholders += 1

                resolved_items.append(resolved_item)
            except Exception as e:
                logger.warning(f"Error resolving placeholders in context item: {str(e)}")
                # Add the original item in case of error
                resolved_items.append(item)

        # Log statistics
        if items_with_placeholders > 0:
            logger.info(
                f"Resolved {total_placeholders_resolved} placeholders in {items_with_placeholders}/{total_items} "
                f"context items using {output_format} format"
            )
        else:
            logger.debug(f"No placeholders found in {total_items} context items")

        return resolved_items
    except Exception as e:
        logger.error(f"Error in resolve_placeholders_in_context: {str(e)}")
        # Return original context items in case of error
        return context_items
