"""
Formula extraction module for LightRAG.

This module provides functionality to extract and analyze mathematical formulas
from text content using regular expressions and heuristic analysis. It also supports
extraction of formula explanations from surrounding text and identification of
relationships between formulas.
"""
import re
import logging
import base64
import io
from typing import List, Dict, Any, Optional, Tuple, Set, Union

# Try to import PIL for image generation, but make it optional
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    logging.warning("PIL not available. Formula image generation will be disabled.")

# Import formula interpreter if available
try:
    from .formula_interpreter import FormulaInterpreter
except ImportError:
    logging.warning("FormulaInterpreter not available. Advanced formula interpretation will be disabled.")

logger = logging.getLogger(__name__)

class FormulaExtractor:
    """
    Extract and process mathematical formulas from documents.

    This class provides functionality to extract mathematical formulas from text,
    analyze their structure, and generate descriptions. It can also extract
    formula explanations from surrounding text and identify relationships between formulas.
    """
    def __init__(self, llm_service=None):
        """
        Initialize the formula extractor.

        Args:
            llm_service: Optional LLM service for formula description and interpretation
        """
        self.llm_service = llm_service

        # Initialize FormulaInterpreter if LLM service is provided
        self.formula_interpreter = None
        if self.llm_service:
            try:
                self.formula_interpreter = FormulaInterpreter(llm_service)
            except NameError:
                logger.warning("FormulaInterpreter class not available. Using basic formula description.")

        # Common mathematical symbols
        self.math_symbols = set([
            '+', '-', '*', '/', '=', '<', '>', '≤', '≥', '±',
            '∑', '∏', '∫', '√', '∛', '∂', '∞', '≈', '≠', '∝',
            '∅', '∈', '∉', '∩', '∪', '⊂', '⊃', '⊄', '⊅', '⊆', '⊇',
            '⊕', '⊗', 'π', 'θ', 'α', 'β', 'γ', 'δ', 'ε', 'λ', 'μ',
            'σ', 'τ', 'φ', 'ω', '^'
        ])

        # Explanation extraction patterns
        self.explanation_patterns = [
            r'(?:where|in which|with|here)\s+([A-Za-z][A-Za-z0-9_]*)\s+(?:is|represents|denotes|means|refers to)\s+([^.;]+)[.;]?',
            r'([A-Za-z][A-Za-z0-9_]*)\s+(?:is|represents|denotes|means|refers to)\s+([^.;]+)[.;]?',
            r'the\s+(?:equation|formula|expression)\s+([^.;]+)[.;]?\s+(?:means|represents|describes|shows|indicates|tells us)\s+that\s+([^.;]+)[.;]?',
            r'(?:this|the)\s+(?:equation|formula|expression)\s+(?:means|represents|describes|shows|indicates|tells us)\s+that\s+([^.;]+)[.;]?'
        ]

    def extract_formulas(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical formulas from text content.

        Args:
            text: Text content to analyze

        Returns:
            List[Dict[str, Any]]: List of extracted formulas with metadata
        """
        # Handle None or empty text
        if text is None or not text:
            return []

        formulas = []
        processed_spans = set()  # Keep track of matched spans to avoid duplicates

        # Simple formula detection patterns
        formula_patterns = [
            r'([A-Za-z][A-Za-z0-9_]*\s*=\s*[A-Za-z0-9_² \+\-\*\/\(\)\[\]\{\}><\=]+)',  # Simple equations like E = mc²
            r'([A-Za-z0-9_ \+\-\*\/\(\)\[\]\{\}=><]+)[=]([A-Za-z0-9_ \+\-\*\/\(\)\[\]\{\}=><]+)',
            r'([A-Za-z0-9_]+\s*[\+\-\*\/]\s*[A-Za-z0-9_]+\s*[\+\-\*\/=]\s*[\w\s\+\-\*\/\(\)\[\]\{\}]+)',
            r'([A-Za-z0-9_]+[\(\^][\d\w\s\+\-\*\/]+[\)]\s*[\+\-\*\/=]\s*[\w\s\+\-\*\/\(\)\[\]\{\}]+)',
            r'(\$[^\$]+\$)',
            r'(\\\([^\\\)]+\\\))',
            r'(\\\[[^\\\]]+\\\])',
        ]

        for pattern in formula_patterns:
            try:
                matches = re.finditer(pattern, text)
                for match in matches:
                    if not match or not match.group(0):
                        continue

                    formula_text = match.group(0).strip()
                    match_span = match.span()

                    # Check if this formula overlaps with a previously processed one
                    is_overlapping = any(
                        max(start, match_span[0]) < min(end, match_span[1])
                        for start, end in processed_spans
                    )
                    if is_overlapping:
                        continue

                    # Validate that this is likely a formula
                    if len(formula_text) < 3 or not self._is_likely_formula(formula_text):
                        continue

                    # Get surrounding context
                    start_context = max(0, match_span[0] - 100)
                    end_context = min(len(text), match_span[1] + 100)
                    context_before = text[start_context:match_span[0]].strip()
                    context_after = text[match_span[1]:end_context].strip()

                    # Create a unique ID for the formula
                    formula_id = f"formula-{len(formulas)}"

                    # Convert formula to textual representation
                    textual_repr = self._formula_to_text(formula_text)

                    # Extract explanation if possible
                    explanation = self.extract_formula_explanation(text, {
                        'formula': formula_text,
                        'context_before': context_before,
                        'context_after': context_after,
                        'position': match_span
                    })

                    # Add formula to results
                    formulas.append({
                        'formula_id': formula_id,
                        'formula': formula_text,
                        'context_before': context_before,
                        'context_after': context_after,
                        'position': match_span,
                        'textual_representation': textual_repr,
                        'latex': self._extract_latex(formula_text),
                        'explanation': explanation
                    })

                    # Mark this span as processed
                    processed_spans.add(match_span)

            except re.error as e:
                logger.error(f"Regex error in pattern '{pattern}': {e}")
                continue

        # Sort formulas by their position in the text
        formulas.sort(key=lambda x: x['position'][0])
        return formulas

    def _is_likely_formula(self, text: str) -> bool:
        """
        Determine if text is likely a mathematical formula.

        Args:
            text: Text to analyze

        Returns:
            bool: True if text is likely a formula
        """
        if len(text) < 3 or text.isdigit():
            return False

        # Count mathematical symbols
        symbol_count = sum(1 for char in text if char in self.math_symbols)
        has_math_symbols = symbol_count >= 1

        # Check for equal sign
        has_equal_sign = '=' in text

        # Check for variable patterns
        var_pattern = re.search(r'([a-zA-Z](?![a-zA-Z])|\b(sin|cos|tan|log|exp|sqrt)\b)\s*([\+\-\*\/=\(\^]|\s*\()', text)

        # Check for complex mathematical elements
        has_complex_elements = any(pattern in text for pattern in [
            '/', '^', 'sqrt', 'log', 'sin', 'cos', 'tan', 'exp', '\\',
            'sum', 'int', 'prod', 'partial', 'infty',
            'alpha', 'beta', 'gamma', 'delta', 'theta', 'pi', 'sigma', 'omega'
        ])

        has_numbers = any(char.isdigit() for char in text)
        has_letters = any(char.isalpha() for char in text)

        # Handle equations with equal signs
        if has_equal_sign:
            # Reject patterns like "word=word" where both sides are words
            if re.fullmatch(r'[a-zA-Z]{4,}=[a-zA-Z]{4,}', text.strip()):
                return False

            # Accept simple variable assignments like "x=y" or "F = ma"
            if re.match(r'^[a-zA-Z][a-zA-Z0-9_]*\s*=\s*[a-zA-Z0-9_²³]+$', text.strip()):
                # But reject if both sides are the same word
                if re.fullmatch(r'([a-zA-Z]+)=\1', text.strip()):
                    return False
                return True

            return True

        # Check for mathematical expressions
        if (has_math_symbols or has_complex_elements) and (var_pattern or (has_letters and has_numbers)):
            if re.match(r'^[a-zA-Z0-9][\.\)]\s*$', text.strip()):
                return False
            return True

        # Check for LaTeX delimiters
        if (text.startswith('$') and text.endswith('$')) or \
           (text.startswith('\\(') and text.endswith('\\)')) or \
           (text.startswith('\\[') and text.endswith('\\]')):
            return True

        # If there are multiple math symbols, it's likely a formula
        if symbol_count >= 2:
            return True

        return False

    def _formula_to_text(self, formula: str) -> str:
        """
        Convert a formula to a readable text representation.

        Args:
            formula: The formula to convert

        Returns:
            str: Textual representation of the formula
        """
        text = formula.strip()

        # Remove LaTeX delimiters if present
        if text.startswith('$') and text.endswith('$') and len(text) > 1:
            text = text[1:-1]
        elif text.startswith('\\(') and text.endswith('\\)') and len(text) > 3:
            text = text[2:-2]
        elif text.startswith('\\[') and text.endswith('\\]') and len(text) > 3:
            text = text[2:-2]

        # Define LaTeX replacements
        latex_replacements = {
            r'\\alpha': 'alpha', r'\\beta': 'beta', r'\\gamma': 'gamma', r'\\delta': 'delta',
            r'\\epsilon': 'epsilon', r'\\zeta': 'zeta', r'\\eta': 'eta', r'\\theta': 'theta',
            r'\\iota': 'iota', r'\\kappa': 'kappa', r'\\lambda': 'lambda', r'\\mu': 'mu',
            r'\\nu': 'nu', r'\\xi': 'xi', r'\\pi': 'pi', r'\\rho': 'rho',
            r'\\sigma': 'sigma', r'\\tau': 'tau', r'\\upsilon': 'upsilon', r'\\phi': 'phi',
            r'\\chi': 'chi', r'\\psi': 'psi', r'\\omega': 'omega',
            r'\\Gamma': 'Gamma', r'\\Delta': 'Delta', r'\\Theta': 'Theta', r'\\Lambda': 'Lambda',
            r'\\Xi': 'Xi', r'\\Pi': 'Pi', r'\\Sigma': 'Sigma', r'\\Upsilon': 'Upsilon',
            r'\\Phi': 'Phi', r'\\Psi': 'Psi', r'\\Omega': 'Omega',
            r'\\sum': 'sum', r'\\prod': 'product', r'\\int': 'integral',
            r'\\partial': 'partial derivative', r'\\infty': 'infinity',
            r'\\approx': 'approximately equal to', r'\\neq': 'not equal to',
            r'\\geq': 'greater than or equal to', r'\\ge': 'greater than or equal to',
            r'\\leq': 'less than or equal to', r'\\le': 'less than or equal to',
            r'\\times': 'multiplied by', r'\\cdot': 'multiplied by',
            r'\\div': 'divided by', r'\\pm': 'plus or minus',
            r'\\sqrt(?:\[[^\]]*\])?\{([^{}]+)\}': r'square root of (\1)',
            r'\\sqrt\s': 'square root of ',
            r'\\frac\{([^{}]+)\}\{([^{}]+)\}': r'(\1) divided by (\2)',
            r'\_\{([^{}]+)\}': r' subscript (\1)',
            r'\_\s*(\w|\d)': r' subscript \1',
            r'\^\{([^{}]+)\}': r' superscript (\1)',
            r'\^\s*([\w\d\+\-]+)': r' superscript \1',
            r'\\left': '', r'\\right': '',
            r'\\mathrm': '', r'\\mathbf': '', r'\\mathcal': '',
            r'\\ ': ' ', r'\\,': ' ', r'\\;': ' ', r'\\!': '',
        }

        # Apply replacements
        for pattern, replacement in latex_replacements.items():
            try:
                text = re.sub(pattern, replacement, text)
            except re.error as e:
                logger.warning(f"Regex error during LaTeX replacement ('{pattern}' -> '{replacement}'): {e}")

        # Handle carets (^) for superscripts
        text = text.replace('^', ' superscript ')

        # Remove remaining LaTeX commands
        text = re.sub(r'\\[a-zA-Z]+', '', text)

        # Remove braces
        text = re.sub(r'[{}]', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Clean up spacing around operators
        text = re.sub(r'\s*([\(\)\+\-\*\/=,])\s*', r'\1', text)

        # Final cleanup
        text = text.replace(' )', ')').replace('( ', '(')

        return text.strip()

    def _extract_latex(self, formula: str) -> Optional[str]:
        """
        Extract LaTeX content from a formula if it's in LaTeX format.

        Args:
            formula: The formula to extract LaTeX from

        Returns:
            Optional[str]: LaTeX content if present, None otherwise
        """
        formula = formula.strip()

        # Check for LaTeX delimiters
        if formula.startswith('$') and formula.endswith('$') and len(formula) > 1:
            return formula[1:-1].strip()
        elif formula.startswith('\\(') and formula.endswith('\\)') and len(formula) > 3:
            return formula[2:-2].strip()
        elif formula.startswith('\\[') and formula.endswith('\\]') and len(formula) > 3:
            return formula[2:-2].strip()

        return None

    def generate_formula_image(self, formula: str, width: int = 400, height: int = 100) -> Optional[str]:
        """
        Generate an image of a formula and return it as base64.

        Args:
            formula: The formula to render
            width: Image width
            height: Image height

        Returns:
            Optional[str]: Base64-encoded image data, or None if generation fails
        """
        if not PIL_AVAILABLE:
            logger.warning("PIL not available. Cannot generate formula image.")
            return None

        try:
            # Create a new image
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)

            # Set up font
            font_size = 16
            font = None
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                logger.warning("DejaVuSans.ttf not found, using default PIL font.")
                try:
                    font = ImageFont.load_default()
                except Exception as font_e:
                    logger.error(f"Error loading default font: {font_e}")
                    # Create a simple font as last resort
                    font = None

            # Handle line wrapping
            lines = []
            if hasattr(draw, 'textlength'):
                if draw.textlength(formula, font=font) > width - 20:
                    # Split into multiple lines if too long
                    words = formula.split()
                    current_line = ""
                    for word in words:
                        if draw.textlength(current_line + word + " ", font=font) <= width - 20:
                            current_line += word + " "
                        else:
                            lines.append(current_line.strip())
                            current_line = word + " "
                    lines.append(current_line.strip())
                else:
                    lines.append(formula)
            else:
                # Fallback if textlength not available
                if len(formula) * (font_size / 2) > width - 20:
                    split_point = len(formula) // 2
                    lines.append(formula[:split_point])
                    lines.append(formula[split_point:])
                else:
                    lines.append(formula)

            # Draw text
            y_text = (height - (len(lines) * (font_size + 2))) // 2
            for line in lines:
                if font:
                    draw.text((10, y_text), line, fill='black', font=font)
                else:
                    # Fallback if no font is available
                    draw.text((10, y_text), line, fill='black')
                y_text += font_size + 2

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')

        except Exception as e:
            logger.error(f"Error generating formula image: {str(e)}")
            return None

    def generate_formula_description(self, formula_data: Dict[str, Any]) -> str:
        """
        Generate a textual description of a formula.

        Args:
            formula_data: Dictionary containing formula and context

        Returns:
            str: Description of the formula
        """
        formula = formula_data['formula']
        context_before = formula_data.get('context_before', '')
        context_after = formula_data.get('context_after', '')

        # If we have an LLM service, use it to generate a description
        if self.llm_service:
            try:
                # This is a placeholder - actual implementation will depend on the LLM service interface
                prompt = f"""
                Analyze the following mathematical formula found within a document.

                Context Before: "{context_before}"
                Formula: "{formula}"
                Context After: "{context_after}"

                Provide a concise description covering:
                1. **Identification:** What does the formula likely represent?
                2. **Interpretation:** Brief explanation of its meaning.
                3. **Contextual Relevance:** Why it's here.
                4. **Key Components:** What the variables likely mean.

                Focus on clarity and avoid unnecessary jargon.
                """

                # For now, fall back to rule-based description
                return self._generate_rule_based_description(formula_data)

            except Exception as e:
                logger.error(f"Error generating formula description with LLM: {str(e)}")
                return self._generate_rule_based_description(formula_data)

        # If no LLM service, use rule-based description
        return self._generate_rule_based_description(formula_data)

    def _generate_rule_based_description(self, formula_data: Dict[str, Any]) -> str:
        """
        Generate a description of a formula using rule-based heuristics.

        Args:
            formula_data: Dictionary containing formula and context

        Returns:
            str: Description of the formula
        """
        formula = formula_data['formula']
        textual = self._formula_to_text(formula)
        description = f"Mathematical formula: {textual}"

        # Analyze formula structure
        if '=' in formula:
            try:
                parts = formula.split('=', 1)
                left = self._formula_to_text(parts[0]).strip()
                right = self._formula_to_text(parts[1]).strip()
                description += f". This appears to be an equation setting '{left}' equal to '{right}'"
            except IndexError:
                description += ". This appears to be an equation."
        elif any(op in formula for op in ['+', '-', '*', '/', '^', '\\sum', '\\int']):
            description += ". This appears to be a mathematical expression involving calculations."
        elif formula.startswith('\\') or '$' in formula or formula.startswith('$') or formula.endswith('$'):
            description += ". This appears to be written in LaTeX format."

        # Try to extract context from surrounding text
        context_snippet = ""
        context_text_to_use = ""

        # Prefer text before the formula, fall back to text after
        priority_context = formula_data.get('context_before', '')
        fallback_context = formula_data.get('context_after', '')

        use_last_sentence = False
        if priority_context:
            context_text_to_use = priority_context
            use_last_sentence = True
        elif fallback_context:
            context_text_to_use = fallback_context
            use_last_sentence = False

        if context_text_to_use:
            # Split into sentences
            sentences = [s.strip() for s in re.split(r'[.?!:]+', context_text_to_use) if s.strip()]
            if sentences:
                # Get either the last sentence (for context before) or first sentence (for context after)
                raw_snippet = sentences[-1] if use_last_sentence else sentences[0]
                temp_snippet = raw_snippet.lower().strip()

                # Clean up common phrases
                cleaned = False
                for phrase in [
                    "formula is", "formula", "is", "are", "as follows", "given by",
                    "where", "the equation", "we have", "it follows that",
                    "consider", "let", "define"
                ]:
                    if temp_snippet.endswith(" " + phrase):
                        temp_snippet = temp_snippet[:-len(" " + phrase)].strip()
                        cleaned = True
                    if temp_snippet.startswith(phrase + " "):
                        temp_snippet = temp_snippet[len(phrase + " "):].strip()
                        cleaned = True
                    if temp_snippet == phrase:
                        temp_snippet = ""
                        cleaned = True

                # Remove trailing punctuation
                if not cleaned and temp_snippet and temp_snippet[-1] in '.,;:!?':
                    temp_snippet = temp_snippet[:-1].strip()

                # Use the snippet if it's meaningful
                if temp_snippet and not temp_snippet.isdigit() and not all(c in '.,;:!?' for c in temp_snippet):
                    context_snippet = temp_snippet

        # Add context to description if available
        if context_snippet:
            if len(context_snippet) > 80:
                context_snippet = context_snippet[:80] + "..."
            description += f". It seems related to the concept of '{context_snippet}' mentioned in the text"

        return description.strip() + "."

    def extract_formula_explanation(self, text: str, formula_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract explanation for a formula from the surrounding text.

        Args:
            text: The document text containing the formula
            formula_data: Dictionary with formula metadata

        Returns:
            Optional[str]: Extracted explanation or None if not found
        """
        formula = formula_data.get('formula', '')
        position = formula_data.get('position', (0, 0))
        context_before = formula_data.get('context_before', '')
        context_after = formula_data.get('context_after', '')

        # Combine contexts for analysis
        combined_context = context_before + " " + formula + " " + context_after

        # Try to find explicit explanation patterns
        explanation = self._extract_explanation_from_patterns(combined_context, formula)

        if explanation:
            return explanation

        # Look for explanation sentences in the context after the formula
        sentence_explanation = self._extract_explanation_from_sentences(context_after)

        if sentence_explanation:
            return sentence_explanation

        # If no explanation found in text, return None
        return None

    def _extract_explanation_from_patterns(self, text: str, formula: str) -> Optional[str]:
        """
        Extract formula explanation using regex patterns.

        Args:
            text: Text content to search in
            formula: The formula to find explanation for

        Returns:
            Optional[str]: Extracted explanation or None if not found
        """
        # Try each pattern
        for pattern in self.explanation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                # Depending on pattern, we might have different groups
                if len(match.groups()) >= 2:
                    # Pattern with variable definitions
                    var_or_formula = match.group(1).strip()
                    explanation = match.group(2).strip()

                    # Check if this is relevant to our formula
                    if var_or_formula in formula or var_or_formula == formula:
                        return explanation
                elif len(match.groups()) >= 1:
                    # Pattern with direct explanation
                    return match.group(1).strip()

        return None

    def _extract_explanation_from_sentences(self, text: str) -> Optional[str]:
        """
        Extract explanation from sentences following the formula.

        Args:
            text: Text following the formula

        Returns:
            Optional[str]: First sentence or None if not suitable
        """
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)

        if not sentences:
            return None

        # Check if first sentence looks like an explanation
        first_sentence = sentences[0].strip()

        # Filter out sentences that don't look like explanations
        if len(first_sentence) < 15:  # Too short
            return None

        # Check for explanation indicators
        explanation_indicators = [
            'means', 'represents', 'describes', 'shows', 'indicates',
            'tells us', 'defines', 'expresses', 'states', 'explains'
        ]

        if any(indicator in first_sentence.lower() for indicator in explanation_indicators):
            return first_sentence

        # If first sentence starts with "This" or "The", it's likely an explanation
        if first_sentence.lower().startswith(('this', 'the', 'it', 'thus')):
            return first_sentence

        return None

    async def identify_formula_relationships(self, formulas: List[Dict[str, Any]], document_text: str = "") -> Dict[str, List[Dict[str, Any]]]:
        """
        Identify relationships between formulas.

        Args:
            formulas: List of extracted formulas
            document_text: The document text containing the formulas

        Returns:
            dict: Dictionary mapping formula IDs to related formula data
        """
        # If no formula interpreter or less than 2 formulas, return empty dict
        if not self.formula_interpreter or len(formulas) < 2:
            return {}

        # Use the formula interpreter to identify relationships
        return await self.formula_interpreter.identify_formula_relationships(formulas, document_text)

    async def interpret_formula(self, formula_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an interpretation for a formula using the formula interpreter.

        Args:
            formula_data: Dictionary with formula metadata

        Returns:
            dict: Formula interpretation data
        """
        # If no formula interpreter, use basic description
        if not self.formula_interpreter:
            description = self.generate_formula_description(formula_data)
            return {
                "explanation": description,
                "components": [],
                "verified": False
            }

        # Use the formula interpreter for advanced interpretation
        existing_explanation = formula_data.get('explanation')
        return await self.formula_interpreter.interpret_formula(formula_data, existing_explanation=existing_explanation)


# Wrapper function for FormulaExtractor.extract_formulas
def extract_formulas(text: str, **kwargs) -> List[Dict[str, Any]]:
    """
    Extract formulas from text content.

    This is a wrapper around FormulaExtractor.extract_formulas for backward compatibility.

    Args:
        text: Text content to analyze
        **kwargs: Additional arguments for formula extraction

    Returns:
        List[Dict[str, Any]]: List of extracted formulas with metadata
    """
    extractor = FormulaExtractor(**kwargs)
    return extractor.extract_formulas(text)