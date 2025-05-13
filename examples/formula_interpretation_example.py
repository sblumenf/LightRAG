"""
Example script demonstrating formula interpretation functionality.

This example shows how to use LightRAG's formula interpretation capabilities
to extract and interpret mathematical formulas in documents.
"""

import asyncio
import os
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import LightRAG components
from lightrag.lightrag import LightRAG
from lightrag.llm.openai import openai_complete
from document_processing.formula_extractor import FormulaExtractor
from document_processing.formula_interpreter import FormulaInterpreter

# Sample document with mathematical formulas
SAMPLE_DOCUMENT = """
# Mathematics of Relativity

Einstein's special theory of relativity introduced the famous equation:

E = mc²

where E is energy, m is mass, and c is the speed of light in vacuum. This equation expresses the equivalence of mass and energy.

The time dilation formula in special relativity is:

t' = t / √(1 - v²/c²)

Here, t' is the time in the moving frame, t is the time in the stationary frame, v is the relative velocity, and c is the speed of light.

The Lorentz factor γ appears frequently in relativity and is defined as:

γ = 1 / √(1 - v²/c²)

This allows us to rewrite the time dilation formula as t' = γt.
"""


async def extract_and_interpret_formulas():
    """
    Extract and interpret formulas from a sample document.
    """
    logger.info("Initializing LightRAG with formula interpretation enabled...")
    
    # Initialize LLM service
    async def llm_service(prompt):
        return await openai_complete(prompt)
    
    # Initialize FormulaExtractor with LLM service
    formula_extractor = FormulaExtractor(llm_service=llm_service)
    
    # Extract formulas from sample document
    logger.info("Extracting formulas from sample document...")
    formulas = formula_extractor.extract_formulas(SAMPLE_DOCUMENT)
    
    logger.info(f"Extracted {len(formulas)} formulas.")
    
    # Process each formula
    for i, formula in enumerate(formulas):
        print(f"\n--- Formula {i+1} ---")
        print(f"Formula: {formula['formula']}")
        print(f"Textual: {formula['textual_representation']}")
        
        # Get explanation from document context if available
        if formula.get('explanation'):
            print(f"Extracted Explanation: {formula['explanation']}")
        
        # Get advanced interpretation
        print("\nGenerating interpretation...")
        interpretation = await formula_extractor.interpret_formula(formula)
        
        print(f"\nDetailed Interpretation:")
        print(interpretation['explanation'])
        
        if interpretation.get('components'):
            print("\nComponents:")
            for comp in interpretation['components']:
                print(f"  - {comp['symbol']}: {comp['meaning']}")
    
    # Identify relationships between formulas
    if len(formulas) > 1:
        print("\n--- Formula Relationships ---")
        relationships = await formula_extractor.identify_formula_relationships(formulas, SAMPLE_DOCUMENT)
        
        for formula_id, related_formulas in relationships.items():
            if related_formulas:
                formula_idx = next((i for i, f in enumerate(formulas) if f.get('formula_id') == formula_id), None)
                if formula_idx is not None:
                    print(f"\nFormula {formula_idx+1} is related to:")
                    
                    for rel in related_formulas:
                        related_idx = next((i for i, f in enumerate(formulas) 
                                         if f.get('formula_id') == rel['related_formula_id']), None)
                        
                        if related_idx is not None:
                            print(f"  - Formula {related_idx+1}")
                            print(f"    Relationship: {rel.get('relationship_type', 'Related')}")
                            print(f"    Explanation: {rel.get('explanation', '')}")


async def setup_lightrag_with_formula_interpretation():
    """
    Set up LightRAG with formula interpretation enabled.
    """
    logger.info("Setting up LightRAG with formula interpretation...")
    
    # Define LLM model function
    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete(prompt, api_key=os.getenv("OPENAI_API_KEY"))
    
    # Define embedding function
    async def embedding_func(text, **kwargs):
        # Simple mock embedding function for demonstration
        import hashlib
        import numpy as np
        
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a vector of floats
        vector = np.array([float(int(c, 16)) for c in text_hash]) / 15.0
        
        # Normalize to unit length
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()
    
    # Initialize LightRAG with formula interpretation enabled
    lightrag = LightRAG(
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        enable_formula_analysis=True,
        enable_formula_interpretation=True,
        formula_interpretation_level="detailed",
        verify_formula_interpretations=True,
        extract_formula_explanations=True,
        identify_formula_relationships=True
    )
    
    logger.info("LightRAG initialized with formula interpretation enabled.")
    return lightrag


async def main():
    """
    Main function to demonstrate formula interpretation.
    """
    logger.info("Starting formula interpretation example...")
    
    # Extract and interpret formulas directly
    await extract_and_interpret_formulas()
    
    # Set up LightRAG with formula interpretation
    lightrag = await setup_lightrag_with_formula_interpretation()
    
    # TODO: Add an example of using LightRAG's full pipeline with formula interpretation
    
    logger.info("Formula interpretation example completed.")


if __name__ == "__main__":
    asyncio.run(main())