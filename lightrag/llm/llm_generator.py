"""
LLM Generator module for Chain-of-Thought reasoning and enhanced citation handling.
"""

import re
import logging
from typing import Optional, Dict, Any, List, Tuple, Callable, Awaitable

from lightrag.prompt import PROMPTS
from lightrag.config_loader import get_enhanced_config

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
    
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else response
    
    return {
        "reasoning": reasoning,
        "answer": answer
    }


def process_citations(response: str, context: List[Dict[str, Any]]) -> str:
    """
    Process citations in a response.
    
    Args:
        response: The response from the LLM with citations
        context: The context items used for generation
        
    Returns:
        Response with processed citations
    """
    # Check if there are any citations
    citation_pattern = r"\[Entity ID: (.*?)\]"
    citations = re.findall(citation_pattern, response)
    
    if not citations:
        return response
    
    # Create a mapping of entity IDs to context items
    context_map = {item["id"]: item for item in context}
    
    # Replace citations with numbered references
    numbered_response = response
    sources = []
    
    for i, entity_id in enumerate(citations, 1):
        if entity_id in context_map:
            # Replace the citation with a numbered reference
            numbered_response = numbered_response.replace(f"[Entity ID: {entity_id}]", f"[{i}]")
            # Add the source to the sources list
            sources.append(f"{i}. {context_map[entity_id]['content']}")
    
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
    
    async def generate_with_cot(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate a response using chain-of-thought reasoning.
        
        Args:
            query: The query to generate a response for
            context: Optional context to include in the prompt
            
        Returns:
            str: Generated response with chain-of-thought reasoning
        """
        prompt = self.create_cot_prompt(query, context)
        
        try:
            response = await self.llm_func(prompt)
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
        
        # Check if reasoning doesn't contain citations
        if self.config.enable_enhanced_citations and "[Entity ID:" not in result["reasoning"]:
            return True
        
        return False
    
    async def generate_with_cot_refinement(self, query: str, context: Optional[str] = None) -> str:
        """
        Generate a response with CoT refinement loop.
        
        Args:
            query: The query to generate a response for
            context: Optional context to include in the prompt
            
        Returns:
            str: Generated response with refined reasoning
        """
        response = await self.generate_with_cot(query, context)
        attempts = 1
        
        while attempts < self.config.max_cot_refinement_attempts:
            should_refine = await self._should_refine_reasoning(response)
            if not should_refine:
                break
            
            logger.info(f"Refining CoT response (attempt {attempts+1}/{self.config.max_cot_refinement_attempts})")
            
            # Create a refined prompt
            prompt = self.create_cot_prompt(query, context)
            prompt += "\n\nYour previous response had insufficient reasoning. Please provide more detailed reasoning with explicit citations to the knowledge base."
            
            try:
                response = await self.llm_func(prompt)
            except Exception as e:
                logger.error(f"Error refining CoT response: {str(e)}")
                # Return the last successful response
                break
            
            attempts += 1
        
        return response
