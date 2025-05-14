"""
Formula extraction module for GraphRAG tutor.
"""
import re
import logging
import base64
import io
from typing import List, Dict, Any, Optional, Tuple, Set, Union

# Try to import PIL for image generation, but make it optional
PIL_AVAILABLE = False
try:
    from PIL import Image, ImageDraw, ImageFont  # noqa: F401
    PIL_AVAILABLE = True
except ImportError:
    logging.warning("PIL not available. Formula image generation will be disabled.")

logger = logging.getLogger(__name__)

class FormulaExtractor:
    """
    Extract and process mathematical formulas from documents.
    """
    def __init__(self, llm_service=None):
        """
        Initialize the formula extractor.

        Args:
            llm_service: Optional LLM service for formula description
        """
        self.llm_service = llm_service

        # Common mathematical symbols - Added '^' and '*'
        self.math_symbols = set([
            '+', '-', '*', '/', '=', '<', '>', '≤', '≥', '±', # Added '*'
            '∑', '∏', '∫', '√', '∛', '∂', '∞', '≈', '≠', '∝',
            '∅', '∈', '∉', '∩', '∪', '⊂', '⊃', '⊄', '⊅', '⊆', '⊇',
            '⊕', '⊗', 'π', 'θ', 'α', 'β', 'γ', 'δ', 'ε', 'λ', 'μ',
            'σ', 'τ', 'φ', 'ω', '^' # Added caret for exponentiation
        ])
    def extract_formulas(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical formulas from text content.

        Args:
            text: Text content to analyze

        Returns:
            List[Dict[str, Any]]: List of extracted formulas with metadata
        """
        formulas = []
        processed_spans = set()  # Keep track of matched spans to avoid duplicates

        # Simple formula detection patterns
        formula_patterns = [
            r'([A-Za-z0-9_ \^\+\-\*\/\(\)\[\]\{\}=><≤≥±∑∏∫√∛∂∞≈≠∝∅∈∉∩∪⊂⊃⊄⊅⊆⊇⊕⊗]+)[=]([A-Za-z0-9_ \^\+\-\*\/\(\)\[\]\{\}=><≤≥±∑∏∫√∛∂∞≈≠∝∅∈∉∩∪⊂⊃⊄⊅⊆⊇⊕⊗]+)',
            r'([A-Za-z0-9_]+\s*[\+\-\*\/\^]\s*[A-Za-z0-9_]+\s*[\+\-\*\/\^\=]\s*[\w\s\^\+\-\*\/\(\)\[\]\{\}]+)',
            r'([A-Za-z0-9_]+[\^\(][\d\w\s\+\-\*\/]+[\)]\s*[\+\-\*\/\=]\s*[\w\s\^\+\-\*\/\(\)\[\]\{\}]+)',
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

                    is_overlapping = any(
                        max(start, match_span[0]) < min(end, match_span[1])
                        for start, end in processed_spans
                    )
                    if is_overlapping:
                        continue

                    if len(formula_text) < 3 or not self._is_likely_formula(formula_text):
                        continue

                    start_context = max(0, match_span[0] - 100)
                    end_context = min(len(text), match_span[1] + 100)
                    context_before = text[start_context:match_span[0]].strip()
                    context_after = text[match_span[1]:end_context].strip()

                    formula_id = f"formula-{len(formulas)}"
                    textual_repr = self._formula_to_text(formula_text)

                    formulas.append({
                        'formula_id': formula_id,
                        'formula': formula_text,
                        'context_before': context_before,
                        'context_after': context_after,
                        'position': match_span,
                        'textual_representation': textual_repr,
                        'latex': self._extract_latex(formula_text)
                    })
                    processed_spans.add(match_span)
            except re.error as e:
                logger.error(f"Regex error in pattern '{pattern}': {e}")
                continue

        formulas.sort(key=lambda x: x['position'][0])
        return formulas
    def _is_likely_formula(self, text: str) -> bool:
        """
        Determine if text is likely a mathematical formula. (Corrected Logic)

        Args:
            text: Text to analyze

        Returns:
            bool: True if text is likely a formula
        """
        if len(text) < 3 or text.isdigit():
            return False

        symbol_count = sum(1 for char in text if char in self.math_symbols)
        has_math_symbols = symbol_count >= 1
        has_equal_sign = '=' in text
        var_pattern = re.search(r'([a-zA-Z](?![a-zA-Z])|\b(sin|cos|tan|log|exp|sqrt)\b)\s*([\+\-\*\/=\(\^]|\s*\()', text)
        has_complex_elements = any(pattern in text for pattern in [
            '/', '^', 'sqrt', 'log', 'sin', 'cos', 'tan', 'exp', '\\',
            'sum', 'int', 'prod', 'partial', 'infty',
            'alpha', 'beta', 'gamma', 'delta', 'theta', 'pi', 'sigma', 'omega'
        ])
        has_numbers = any(char.isdigit() for char in text)
        has_letters = any(char.isalpha() for char in text)

        # ✅ Fixed block to allow "F = ma" but reject "word=word"
        if has_equal_sign:
            # Reject patterns like "word1=word2" but allow "F = ma"
            if re.fullmatch(r'[a-zA-Z0-9]+=[a-zA-Z0-9]+', text.strip()):
                return False
            return True

        if (has_math_symbols or has_complex_elements) and (var_pattern or (has_letters and has_numbers)):
            if re.match(r'^[a-zA-Z0-9][\.\)]\s*$', text.strip()):
                return False
            return True

        if (text.startswith('$') and text.endswith('$')) or \
           (text.startswith('\\(') and text.endswith('\\)')) or \
           (text.startswith('\\[') and text.endswith('\\]')):
            return True

        if symbol_count >= 2:
            return True

        return False
    def _formula_to_text(self, formula: str) -> str:
        text = formula.strip()

        if text.startswith('$') and text.endswith('$') and len(text) > 1:
            text = text[1:-1]
        elif text.startswith('\\(') and text.endswith('\\)') and len(text) > 3:
            text = text[2:-2]
        elif text.startswith('\\[') and text.endswith('\\]') and len(text) > 3:
            text = text[2:-2]

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

        for pattern, replacement in latex_replacements.items():
            try:
                text = re.sub(pattern, replacement, text)
            except re.error as e:
                logger.warning(f"Regex error during LaTeX replacement ('{pattern}' -> '{replacement}'): {e}")

        text = text.replace('^', ' superscript ')
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'[{}]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s*([\(\)\+\-\*\/=,])\s*', r'\1', text)
        text = text.replace(' )', ')').replace('( ', '(')
        return text.strip()

    def _extract_latex(self, formula: str) -> Optional[str]:
        formula = formula.strip()
        if formula.startswith('$') and formula.endswith('$') and len(formula) > 1:
            return formula[1:-1].strip()
        elif formula.startswith('\\(') and formula.endswith('\\)') and len(formula) > 3:
            return formula[2:-2].strip()
        elif formula.startswith('\\[') and formula.endswith('\\]') and len(formula) > 3:
            return formula[2:-2].strip()
        return None

    def generate_formula_image(self, formula: str, width: int = 400, height: int = 100) -> Optional[str]:
        if not PIL_AVAILABLE:
            logger.warning("PIL not available. Cannot generate formula image.")
            return None

        try:
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            font_size = 16
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            except IOError:
                logger.warning("DejaVuSans.ttf not found, using default PIL font.")
                font = ImageFont.load_default()

            lines = []
            if hasattr(draw, 'textlength'):
                if draw.textlength(formula, font=font) > width - 20:
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
                if len(formula) * (font_size / 2) > width - 20:
                    split_point = len(formula) // 2
                    lines.append(formula[:split_point])
                    lines.append(formula[split_point:])
                else:
                    lines.append(formula)

            y_text = (height - (len(lines) * (font_size + 2))) // 2
            for line in lines:
                draw.text((10, y_text), line, fill='black', font=font)
                y_text += font_size + 2

            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error generating formula image: {str(e)}")
            return None

    def generate_formula_description(self, formula_data: Dict[str, Any]) -> str:
        formula = formula_data['formula']
        context_before = formula_data.get('context_before', '')
        context_after = formula_data.get('context_after', '')

        if self.llm_service:
            try:
                prompt = f"""
                Analyze the following mathematical formula found within a document.

                Context Before: "{context_before}"
                Formula: "{formula}"
                Context After: "{context_after}"

                Provide a concise description covering:
                1.  **Identification:** What does the formula likely represent?
                2.  **Interpretation:** Brief explanation of its meaning.
                3.  **Contextual Relevance:** Why it’s here.
                4.  **Key Components:** What the variables likely mean.

                Focus on clarity and avoid unnecessary jargon.
                """
                logger.info("LLM service available but generate call commented out/not implemented.")
                return self._generate_rule_based_description(formula_data)
            except Exception as e:
                logger.error(f"Error generating formula description with LLM: {str(e)}")
                return self._generate_rule_based_description(formula_data)

        return self._generate_rule_based_description(formula_data)

    def _generate_rule_based_description(self, formula_data: Dict[str, Any]) -> str:
        formula = formula_data['formula']
        textual = self._formula_to_text(formula)
        description = f"Mathematical formula: {textual}"

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
        elif formula.startswith('\\') or '$' in formula:
            description += ". This appears to be written in LaTeX format."

        context_snippet = ""
        context_text_to_use = ""
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
            sentences = [s.strip() for s in re.split(r'[.?!:]+', context_text_to_use) if s.strip()]
            if sentences:
                raw_snippet = sentences[-1] if use_last_sentence else sentences[0]
                temp_snippet = raw_snippet.lower().strip()
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

                if not cleaned and temp_snippet and temp_snippet[-1] in '.,;:!?':
                    temp_snippet = temp_snippet[:-1].strip()

                if temp_snippet and not temp_snippet.isdigit() and not all(c in '.,;:!?' for c in temp_snippet):
                    context_snippet = temp_snippet

        if context_snippet:
            if len(context_snippet) > 80:
                context_snippet = context_snippet[:80] + "..."
            description += f". It seems related to the concept of '{context_snippet}' mentioned in the text"

        return description.strip() + "."
