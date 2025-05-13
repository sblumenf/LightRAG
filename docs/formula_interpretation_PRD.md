# Formula Interpretation PRD: Using LLMs to interpret and explain mathematical formulas in context

## 1. Overview

Enhance LightRAG's formula handling capabilities to provide comprehensive mathematical formula interpretation using LLMs, prioritizing existing explanations when available and generating detailed explanations when needed.

## 2. Problem Statement

The current LightRAG system has basic formula extraction and representation capabilities but lacks advanced formula interpretation features. While it can extract and represent formulas, it primarily relies on rule-based approaches for describing formulas, which limits the depth and quality of mathematical explanations.

## 3. Objectives

- Prioritize extracting and using existing formula explanations from documents
- Enhance LLM integration to generate detailed formula explanations when needed
- Provide step-by-step breakdowns of complex formulas
- Ensure mathematical accuracy in formula interpretations
- Identify and establish relationships between formulas in the same context
- Maintain seamless integration with the existing placeholder and citation systems

## 4. User Benefits

- Deeper understanding of mathematical formulas in documents
- Clear, detailed explanations that break down complex mathematical concepts
- Accurate formula interpretation contextually relevant to the document
- Connected formula explanations that show relationships between related concepts
- Consistent experience through integration with existing citation mechanisms

## 5. Technical Specification

### 5.1 Enhanced Formula Extraction

#### Enhancements to FormulaExtractor class
- Add capability to extract existing formula explanations from surrounding text
- Identify formula relationships to connect related mathematical concepts
- Implement context extraction improvements for better formula understanding

```python
class FormulaExtractor:
    # Existing code...
    
    def extract_formula_explanation(self, text, formula_data):
        """
        Extract explanation for a formula from the surrounding text.
        
        Args:
            text: The document text containing the formula
            formula_data: Dictionary with formula metadata
            
        Returns:
            str: Extracted explanation or None if not found
        """
        # Implementation details for extracting explanations...
        
    def identify_formula_relationships(self, formulas):
        """
        Identify relationships between formulas.
        
        Args:
            formulas: List of extracted formulas
            
        Returns:
            dict: Dictionary mapping formula IDs to related formula IDs
        """
        # Implementation details for identifying relationships...
```

### 5.2 LLM-based Formula Explanation

#### Implementation of FormulaInterpreter class

```python
class FormulaInterpreter:
    """
    Interprets mathematical formulas using LLMs.
    """
    def __init__(self, llm_service):
        """
        Initialize the formula interpreter.
        
        Args:
            llm_service: LLM service for formula interpretation
        """
        self.llm_service = llm_service
        
    def interpret_formula(self, formula_data, context_items=None, existing_explanation=None):
        """
        Generate an interpretation for a formula.
        
        Args:
            formula_data: Dictionary with formula metadata
            context_items: Optional context items for better interpretation
            existing_explanation: Existing explanation to enhance/verify
            
        Returns:
            dict: Formula interpretation data
        """
        # Implementation details...
        
    def break_down_complex_formula(self, formula_data):
        """
        Break down a complex formula into simpler components.
        
        Args:
            formula_data: Dictionary with formula metadata
            
        Returns:
            list: List of formula components with explanations
        """
        # Implementation details...
        
    def verify_mathematical_accuracy(self, formula, explanation):
        """
        Verify the mathematical accuracy of a formula explanation.
        
        Args:
            formula: The formula text
            explanation: The generated explanation
            
        Returns:
            bool: True if accurate, False otherwise
        """
        # Implementation details...
```

### 5.3 Integration with PlaceholderResolver

Enhance the existing PlaceholderResolver to include formula interpretations:

```python
class PlaceholderResolver:
    # Existing code...
    
    def _format_formula_from_dict(self, formula):
        # Enhanced version with interpretation
        formula_id = formula.get('formula_id', 'unknown')
        formula_text = formula.get('formula', '')
        textual_representation = formula.get('textual_representation', '')
        description = formula.get('description', '')
        interpretation = formula.get('interpretation', '')
        components = formula.get('components', [])
        related_formulas = formula.get('related_formulas', [])
        latex = formula.get('latex', '')
        
        # Use the cached method with additional parameters
        return self._format_formula_description(
            formula_id, formula_text, textual_representation,
            description, interpretation, components, 
            related_formulas, latex
        )
    
    @lru_cache(maxsize=128)
    def _format_formula_description(self, formula_id, formula_text, 
                                   textual_representation, description,
                                   interpretation, components, 
                                   related_formulas, latex):
        # Enhanced version with interpretation
        # Implementation details...
```

### 5.4 Database Schema Updates

Update the formula schema to include interpretation data:

```json
{
  "formula_id": "formula-1",
  "formula": "E = mc^2",
  "textual_representation": "Energy equals mass times the speed of light squared",
  "description": "Einstein's mass-energy equivalence formula",
  "interpretation": {
    "explanation": "Detailed explanation of the formula...",
    "components": [
      {
        "symbol": "E",
        "meaning": "Energy",
        "description": "The total energy of an object"
      },
      {
        "symbol": "m",
        "meaning": "Mass",
        "description": "The rest mass of the object"
      },
      {
        "symbol": "c",
        "meaning": "Speed of light",
        "description": "The speed of light in vacuum (299,792,458 meters per second)"
      }
    ]
  },
  "related_formulas": ["formula-2", "formula-3"],
  "latex": "E = mc^2"
}
```

### 5.5 New LLM Prompts

#### Formula Interpretation Prompt
```
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
```

#### Formula Verification Prompt
```
You are verifying the mathematical accuracy of a formula explanation. Your task is to check if the explanation provided is correct.

Formula: "{formula}"
Explanation: "{explanation}"

Please answer the following:
1. Is the explanation mathematically accurate? (Yes/No)
2. If not, what specific errors or inaccuracies exist?
3. Provide a corrected explanation if needed

Focus exclusively on mathematical correctness, not writing style or clarity.
```

#### Formula Relationship Prompt
```
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
```

## 6. Implementation Plan

### Phase 1: Formula Extraction Enhancements
- [ ] Enhance FormulaExtractor to identify existing explanations in text
- [ ] Implement context extraction improvements
- [ ] Add formula relationship identification

### Phase 2: Formula Interpreter Implementation
- [ ] Create FormulaInterpreter class 
- [ ] Implement formula interpretation using LLMs
- [ ] Develop complex formula breakdown capability 
- [ ] Add mathematical accuracy verification

### Phase 3: Integration with Existing Systems
- [ ] Enhance PlaceholderResolver to include formula interpretations
- [ ] Update citation handling to include interpretation data
- [ ] Integrate with Chain-of-Thought reasoning for better explanations

### Phase 4: Testing and Refinement
- [ ] Create test cases for formula interpretation
- [ ] Evaluate interpretation accuracy
- [ ] Refine prompts and verification mechanisms
- [ ] Optimize performance

## 7. API Changes

### New Configuration Options

```python
lightrag = LightRAG(
    enable_formula_interpretation=True,
    formula_interpretation_level="detailed",  # Options: "basic", "detailed"
    verify_formula_interpretations=True
)
```

### Extended Response Format

```python
response = lightrag.query("Explain the energy-mass equivalence formula.")

# Response includes formula interpretations
{
    "answer": "...",
    "sources": [
        {
            "type": "formula",
            "id": "formula-1",
            "formula": "E = mc^2",
            "interpretation": {
                "explanation": "...",
                "components": [...]
            },
            "related_formulas": [...]
        }
    ]
}
```

## 8. Test Cases

### Formula Extraction Tests
- Test extraction of formulas with existing explanations
- Test edge cases like complex mathematical notation
- Test formula relationship identification

### Formula Interpretation Tests
- Test basic formula interpretation
- Test complex formula breakdown
- Test mathematical accuracy verification

### Integration Tests
- Test placeholder resolution with interpretations
- Test citation handling with interpretations
- Test Chain-of-Thought integration

### End-to-End Tests
- Test query-response flow with formula interpretations
- Test multi-formula scenarios
- Test accuracy against known formula explanations

## 9. Limitations and Future Enhancements

### Limitations
- Interpretation quality depends on LLM capabilities
- Complex mathematical notation may not be fully interpreted
- Mathematical verification has inherent limitations
- Domain-specific formulas may require specialized knowledge

### Future Enhancements
- Domain-specific formula interpretation
- Multi-language formula support
- Interactive formula exploration
- Formula visualization capabilities
- Support for mathematical proofs

## 10. Documentation Updates

- Update user guides to include formula interpretation
- Provide examples of formula interpretation usage
- Document configuration options
- Create troubleshooting guide
- Add best practices for formula interpretation

## Conclusion

This enhancement will significantly improve LightRAG's ability to handle mathematical formulas, providing users with detailed, accurate interpretations while maintaining seamless integration with the existing system. By prioritizing existing explanations and falling back to LLM-generated explanations when needed, the system will provide the most accurate formula interpretations possible.