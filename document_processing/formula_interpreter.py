"""
Formula Interpreter for LightRAG.

This module provides functionality to interpret mathematical formulas using LLMs,
offering detailed explanations, component breakdown, and verification of mathematical
accuracy.
"""

import logging
import re
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union, Set

# Configure logger
logger = logging.getLogger(__name__)


class FormulaInterpreter:
    """
    Interprets mathematical formulas using LLMs.
    
    This class provides functionality to generate detailed interpretations of 
    mathematical formulas, including component breakdowns and verification of
    mathematical accuracy. It integrates with the existing formula extraction
    system to enhance formula understanding.
    """
    
    def __init__(self, llm_service=None):
        """
        Initialize the formula interpreter.
        
        Args:
            llm_service: LLM service for formula interpretation
        """
        self.llm_service = llm_service
        
        # Interpretation prompts
        self.formula_interpretation_prompt = """
You are analyzing a mathematical formula in a document. Your task is to provide a detailed explanation of what this formula means, breaking it down step by step.

Context before the formula: "{context_before}"
Formula: "{formula}"
Context after the formula: "{context_after}"

Please provide:
1. A detailed interpretation of what this formula represents and its significance
2. The meaning of each variable or symbol in the formula
3. How to read or verbalize this formula
4. A step-by-step explanation of how to understand or derive this formula
5. How this formula relates to its context in the document
6. Any practical applications or implications of this formula (if discernible from context)

Your explanation should be mathematically accurate, clear, and as detailed as possible while remaining concise.
"""

        self.formula_verification_prompt = """
You are verifying the mathematical accuracy of a formula explanation. Your task is to check if the explanation provided is correct.

Formula: "{formula}"
Explanation: "{explanation}"

Please answer the following:
1. Is the explanation mathematically accurate? (Yes/No)
2. If not, what specific errors or inaccuracies exist?
3. Provide a corrected explanation if needed

Focus exclusively on mathematical correctness, not writing style or clarity.
"""

        self.formula_relationship_prompt = """
You are analyzing relationships between mathematical formulas in a document. Your task is to determine how these formulas are related to each other.

Formula 1: "{formula1}"
Formula 2: "{formula2}"
Document context: "{context}"

Please determine:
1. Are these formulas directly related? (Yes/No)
2. If yes, how are they related? (e.g., One is derived from the other, They represent different aspects of the same concept, etc.)
3. What is the conceptual connection between these formulas?
4. Could knowledge of one formula help in understanding the other?

Provide a concise explanation of the relationship between these formulas.
"""

    async def interpret_formula(self, formula_data: Dict[str, Any], context_items: List[Dict[str, Any]] = None, existing_explanation: str = None) -> Dict[str, Any]:
        """
        Generate an interpretation for a formula.
        
        Args:
            formula_data: Dictionary with formula metadata
            context_items: Optional context items for better interpretation
            existing_explanation: Existing explanation to enhance/verify
            
        Returns:
            dict: Formula interpretation data
        """
        if not self.llm_service:
            logger.warning("No LLM service provided for formula interpretation")
            return self._generate_basic_interpretation(formula_data)
            
        try:
            formula = formula_data.get('formula', '')
            context_before = formula_data.get('context_before', '')
            context_after = formula_data.get('context_after', '')
            
            # If we have an existing explanation, verify it instead of creating a new one
            if existing_explanation:
                return await self._verify_and_enhance_existing_explanation(formula, existing_explanation)
                
            # Prepare context for formula interpretation
            context_data = {
                "context_before": context_before,
                "formula": formula,
                "context_after": context_after
            }
            
            # Generate formula interpretation using LLM
            interpretation = await self._generate_interpretation_with_llm(context_data)
            
            # Extract components from the interpretation
            components = self._extract_components_from_interpretation(interpretation, formula)
            
            # Create the interpretation result
            interpretation_result = {
                "explanation": interpretation,
                "components": components,
                "verified": True
            }
            
            return interpretation_result
            
        except Exception as e:
            logger.error(f"Error interpreting formula: {str(e)}")
            return self._generate_basic_interpretation(formula_data)
            
    async def break_down_complex_formula(self, formula_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Break down a complex formula into simpler components.
        
        Args:
            formula_data: Dictionary with formula metadata
            
        Returns:
            list: List of formula components with explanations
        """
        if not self.llm_service:
            logger.warning("No LLM service provided for formula breakdown")
            return []
            
        try:
            formula = formula_data.get('formula', '')
            context_before = formula_data.get('context_before', '')
            context_after = formula_data.get('context_after', '')
            
            # Build prompt for breaking down the formula
            prompt = f"""
You are analyzing a complex mathematical formula. Your task is to break it down into its fundamental components and explain each one.

Formula: "{formula}"

Please:
1. Identify the main components of this formula
2. For each component, provide:
   a. The mathematical expression of the component
   b. What it represents
   c. How it contributes to the overall formula

Format your response as a numbered list of components.
"""
            
            # Call LLM to break down the formula
            response = await self.llm_service(prompt)
            
            # Parse the response into components
            components = self._parse_components_from_breakdown(response, formula)
            
            return components
            
        except Exception as e:
            logger.error(f"Error breaking down complex formula: {str(e)}")
            return []
            
    async def verify_mathematical_accuracy(self, formula: str, explanation: str) -> Tuple[bool, Optional[str]]:
        """
        Verify the mathematical accuracy of a formula explanation.
        
        Args:
            formula: The formula text
            explanation: The generated explanation
            
        Returns:
            Tuple[bool, Optional[str]]: (is_accurate, corrected_explanation)
        """
        if not self.llm_service:
            logger.warning("No LLM service provided for mathematical verification")
            return True, None
            
        try:
            # Format the prompt for verification
            prompt = self.formula_verification_prompt.format(
                formula=formula,
                explanation=explanation
            )
            
            # Call LLM for verification
            response = await self.llm_service(prompt)
            
            # Parse the verification response
            is_accurate, corrected_explanation = self._parse_verification_response(response)
            
            return is_accurate, corrected_explanation
            
        except Exception as e:
            logger.error(f"Error verifying mathematical accuracy: {str(e)}")
            return True, None
            
    async def identify_formula_relationships(self, formulas: List[Dict[str, Any]], document_text: str = "") -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify relationships between formulas.
        
        Args:
            formulas: List of extracted formulas
            document_text: The document text containing the formulas
            
        Returns:
            dict: Dictionary mapping formula IDs to related formula data
        """
        if not self.llm_service or len(formulas) < 2:
            return {}
            
        try:
            # Create a dictionary to store relationships
            formula_relationships = {}
            
            # Iterate through pairs of formulas to identify relationships
            for i, formula1 in enumerate(formulas):
                formula1_id = formula1.get('formula_id', f'formula-{i}')
                formula_relationships[formula1_id] = []
                
                for j, formula2 in enumerate(formulas):
                    if i == j:
                        continue
                        
                    formula2_id = formula2.get('formula_id', f'formula-{j}')
                    
                    # Get text between the two formulas as context
                    context = self._get_context_between_formulas(formula1, formula2, document_text)
                    
                    # Check if formulas are related
                    is_related, relationship_type, explanation = await self._check_formula_relationship(
                        formula1.get('formula', ''),
                        formula2.get('formula', ''),
                        context
                    )
                    
                    if is_related:
                        formula_relationships[formula1_id].append({
                            "related_formula_id": formula2_id,
                            "relationship_type": relationship_type,
                            "explanation": explanation
                        })
            
            return formula_relationships
            
        except Exception as e:
            logger.error(f"Error identifying formula relationships: {str(e)}")
            return {}
            
    def _generate_basic_interpretation(self, formula_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a basic interpretation when LLM is not available.
        
        Args:
            formula_data: Dictionary with formula metadata
            
        Returns:
            dict: Basic formula interpretation
        """
        formula = formula_data.get('formula', '')
        textual_repr = formula_data.get('textual_representation', '')
        
        # Create a simple explanation
        explanation = f"This is a mathematical formula: {textual_repr}"
        
        # Extract potential variables with simple regex
        variables = re.findall(r'[a-zA-Z]', formula)
        unique_vars = list(set(variables))
        
        # Create basic components list
        components = []
        for var in unique_vars:
            components.append({
                "symbol": var,
                "meaning": f"Variable {var}",
                "description": f"A variable represented by the symbol {var}"
            })
            
        return {
            "explanation": explanation,
            "components": components,
            "verified": False
        }
        
    async def _generate_interpretation_with_llm(self, context_data: Dict[str, str]) -> str:
        """
        Generate formula interpretation using LLM.
        
        Args:
            context_data: Dictionary with context information
            
        Returns:
            str: LLM-generated interpretation
        """
        # Format the prompt with context data
        prompt = self.formula_interpretation_prompt.format(
            context_before=context_data.get('context_before', ''),
            formula=context_data.get('formula', ''),
            context_after=context_data.get('context_after', '')
        )
        
        # Call LLM for interpretation
        interpretation = await self.llm_service(prompt)
        
        return interpretation
        
    def _extract_components_from_interpretation(self, interpretation: str, formula: str) -> List[Dict[str, str]]:
        """
        Extract formula components from the generated interpretation.
        
        Args:
            interpretation: The interpretation text
            formula: The original formula
            
        Returns:
            list: List of component dictionaries
        """
        components = []
        
        # Look for sections that describe variables or symbols
        lines = interpretation.split('\n')
        variable_section = False
        
        for line in lines:
            # Check if we're in a section that describes variables
            if re.search(r'variables|symbols|components|where', line, re.IGNORECASE):
                variable_section = True
                continue
                
            if variable_section:
                # Look for patterns like "x: blah blah" or "x - blah blah"
                var_match = re.match(r'^\s*([a-zA-Z0-9_^{}()\[\]]+)\s*[:-]\s*(.+)$', line)
                if var_match:
                    symbol = var_match.group(1).strip()
                    description = var_match.group(2).strip()
                    
                    # Try to separate meaning from description if possible
                    parts = description.split(',', 1)
                    meaning = parts[0].strip()
                    full_description = description
                    
                    if len(parts) > 1:
                        full_description = parts[1].strip()
                    
                    components.append({
                        "symbol": symbol,
                        "meaning": meaning,
                        "description": full_description
                    })
        
        # If no components were found, try another approach with regex
        if not components:
            # Extract all variables that appear in the formula
            potential_vars = re.findall(r'[a-zA-Z]', formula)
            unique_vars = list(set(potential_vars))
            
            for var in unique_vars:
                # Search for descriptions of this variable in the interpretation
                pattern = r'(?:variable|symbol)\s+' + re.escape(var) + r'\s+(?:is|represents|denotes)\s+([^.]+)'
                matches = re.findall(pattern, interpretation, re.IGNORECASE)
                
                description = f"Variable {var}" if not matches else matches[0].strip()
                components.append({
                    "symbol": var,
                    "meaning": description,
                    "description": description
                })
                
        return components
        
    async def _verify_and_enhance_existing_explanation(self, formula: str, existing_explanation: str) -> Dict[str, Any]:
        """
        Verify and enhance an existing formula explanation.
        
        Args:
            formula: The formula text
            existing_explanation: The existing explanation
            
        Returns:
            dict: Enhanced formula interpretation
        """
        # Verify the mathematical accuracy
        is_accurate, corrected_explanation = await self.verify_mathematical_accuracy(formula, existing_explanation)
        
        # Use the corrected explanation if available, otherwise use the existing one
        explanation = corrected_explanation if corrected_explanation else existing_explanation
        
        # Extract components from the explanation
        components = self._extract_components_from_interpretation(explanation, formula)
        
        return {
            "explanation": explanation,
            "components": components,
            "verified": is_accurate
        }
        
    def _parse_components_from_breakdown(self, breakdown_text: str, formula: str) -> List[Dict[str, Any]]:
        """
        Parse component breakdown from LLM response.
        
        Args:
            breakdown_text: Text breakdown from LLM
            formula: Original formula
            
        Returns:
            list: List of component dictionaries
        """
        components = []
        
        # Split by numbered items
        component_sections = re.split(r'\n\s*\d+\.', breakdown_text)
        
        for section in component_sections[1:]:  # Skip the first element (empty or intro text)
            lines = section.strip().split('\n')
            if not lines:
                continue
                
            # Try to extract component expression, meaning, and description
            expression = ""
            meaning = ""
            description = ""
            
            for i, line in enumerate(lines):
                if i == 0:
                    # First line likely contains the expression
                    expr_match = re.search(r'([^:]+)(?::|$)', line)
                    if expr_match:
                        expression = expr_match.group(1).strip()
                    
                    # Check if the first line also has a meaning after a colon
                    meaning_match = re.search(r':\s*(.+)$', line)
                    if meaning_match:
                        meaning = meaning_match.group(1).strip()
                elif i == 1 and not meaning:
                    # Second line might have the meaning if not found in first line
                    meaning = line.strip()
                else:
                    # Additional lines contribute to description
                    description += line.strip() + " "
            
            description = description.strip()
            
            if expression:
                components.append({
                    "expression": expression,
                    "meaning": meaning,
                    "description": description
                })
                
        return components
        
    def _parse_verification_response(self, verification_text: str) -> Tuple[bool, Optional[str]]:
        """
        Parse verification response from LLM.
        
        Args:
            verification_text: LLM response to verification prompt
            
        Returns:
            Tuple[bool, Optional[str]]: (is_accurate, corrected_explanation)
        """
        # Look for Yes/No answers
        is_accurate = False
        corrected_explanation = None
        
        # Check for 'yes' pattern
        if re.search(r'(?:^|\n)(?:1\.\s*)?yes', verification_text, re.IGNORECASE):
            is_accurate = True
        
        # If not accurate, try to extract corrected explanation
        if not is_accurate:
            # Look for a corrected explanation section
            correction_match = re.search(r'(?:corrected explanation|correct explanation)(?::\s*|\n\s*)(.+?)(?:\n\s*\d|\Z)', 
                                        verification_text, re.IGNORECASE | re.DOTALL)
            
            if correction_match:
                corrected_explanation = correction_match.group(1).strip()
        
        return is_accurate, corrected_explanation
        
    def _get_context_between_formulas(self, formula1: Dict[str, Any], formula2: Dict[str, Any], document_text: str) -> str:
        """
        Get context between two formulas.
        
        Args:
            formula1: First formula data
            formula2: Second formula data
            document_text: Full document text
            
        Returns:
            str: Context text between formulas
        """
        try:
            pos1 = formula1.get('position', (0, 0))
            pos2 = formula2.get('position', (0, 0))
            
            # Determine which formula comes first
            if pos1[0] < pos2[0]:
                start_pos = pos1[1]  # End of first formula
                end_pos = pos2[0]    # Start of second formula
            else:
                start_pos = pos2[1]  # End of second formula
                end_pos = pos1[0]    # Start of first formula
                
            # Get text between formulas, limited to a reasonable length
            context_length = min(end_pos - start_pos, 500)
            context = document_text[start_pos:start_pos + context_length]
            
            return context
        except (KeyError, TypeError, IndexError) as e:
            logger.warning(f"Error getting context between formulas: {str(e)}")
            return ""
            
    async def _check_formula_relationship(self, formula1: str, formula2: str, context: str) -> Tuple[bool, str, str]:
        """
        Check if two formulas are related.
        
        Args:
            formula1: First formula
            formula2: Second formula
            context: Context between formulas
            
        Returns:
            Tuple[bool, str, str]: (is_related, relationship_type, explanation)
        """
        # Format the prompt
        prompt = self.formula_relationship_prompt.format(
            formula1=formula1,
            formula2=formula2,
            context=context
        )
        
        # Call LLM for relationship analysis
        response = await self.llm_service(prompt)
        
        # Parse the response
        is_related = False
        relationship_type = ""
        explanation = ""
        
        # Check if formulas are related
        related_match = re.search(r'(?:^|\n)(?:1\.\s*)?(?:yes|no)', response, re.IGNORECASE)
        if related_match and "yes" in related_match.group(0).lower():
            is_related = True
            
            # Extract relationship type
            type_match = re.search(r'(?:^|\n)(?:2\.\s*)?(?:if yes,\s*)?how.+?:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
            if type_match:
                relationship_type = type_match.group(1).strip()
                
            # Extract explanation
            explanation_match = re.search(r'(?:^|\n)(?:3\.\s*)?(?:what is|connection|conceptual).+?:\s*(.+?)(?:\n\s*\d|\Z)', response, re.IGNORECASE | re.DOTALL)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                
        return is_related, relationship_type, explanation