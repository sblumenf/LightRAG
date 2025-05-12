"""
Vision Adapter Module for LightRAG.

This module provides adapters for integrating with LLM services that offer 
vision/multimodal capabilities, primarily used for diagram description generation.
"""

import base64
import logging
import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import time

logger = logging.getLogger(__name__)

class VisionAdapterBase(ABC):
    """
    Base class for vision model adapters used to generate descriptions of visual content.
    
    This class defines the interface that all vision adapters must implement.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the vision adapter.
        
        Args:
            api_key: API key for the LLM service
            model_name: Name of the vision model to use
        """
        self.api_key = api_key
        self.model_name = model_name
        self._is_initialized = False
        
    @property
    def is_initialized(self) -> bool:
        """Check if the adapter is initialized."""
        return self._is_initialized
    
    @property
    def provider_name(self) -> str:
        """Get the name of the LLM provider."""
        return self.__class__.__name__.replace('VisionAdapter', '')
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the vision adapter and verify that it can connect to the LLM service.
        
        Returns:
            bool: True if initialization is successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def generate_description(
        self, 
        image_data: bytes, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a description of the image using the vision model.
        
        Args:
            image_data: Raw image data in bytes
            prompt: The prompt to send to the model for image description
            context: Optional additional context about the image
            
        Returns:
            str: The generated description
        """
        pass
    
    def encode_image_to_base64(self, image_data: bytes) -> str:
        """
        Encode image data to base64 for API requests.
        
        Args:
            image_data: Raw image data in bytes
            
        Returns:
            str: Base64-encoded image data
        """
        return base64.b64encode(image_data).decode('utf-8')
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the vision service is available and functioning.
        
        Returns:
            bool: True if the service is healthy, False otherwise
        """
        pass
    

class OpenAIVisionAdapter(VisionAdapterBase):
    """Vision adapter implementation for OpenAI's vision models."""
    
    DEFAULT_MODEL = "gpt-4o"
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the OpenAI vision adapter.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable
            model_name: Name of the vision model to use. Defaults to gpt-4o
            base_url: Base URL for the OpenAI API. If None, uses the default OpenAI URL
        """
        super().__init__(api_key, model_name or self.DEFAULT_MODEL)
        self.base_url = base_url
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI
            
            # Get API key from environment if not provided
            if not self.api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
                if not self.api_key:
                    logger.error("OpenAI API key not provided and OPENAI_API_KEY not set in environment")
                    return False
            
            # Set up the client
            client_options = {
                "api_key": self.api_key,
                "timeout": 60.0
            }
            
            if self.base_url:
                client_options["base_url"] = self.base_url
            
            self.client = AsyncOpenAI(**client_options)
            self._is_initialized = True
            
            # Verify the client works
            is_healthy = await self.health_check()
            if not is_healthy:
                logger.warning("OpenAI vision service health check failed")
                self._is_initialized = False
                return False
                
            return True
            
        except ImportError:
            logger.error("OpenAI package not installed. Install it with 'pip install openai'")
            return False
        except Exception as e:
            logger.error(f"Error initializing OpenAI vision adapter: {str(e)}")
            return False
    
    async def generate_description(
        self, 
        image_data: bytes, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a description using OpenAI's vision model."""
        if not self.is_initialized:
            initialized = await self.initialize()
            if not initialized:
                return "Error: OpenAI vision adapter not initialized"
        
        context = context or {}
        
        try:
            base64_image = self.encode_image_to_base64(image_data)
            
            # Build the messages
            messages = []
            
            # Add system prompt if provided in context
            if "system_prompt" in context:
                messages.append({
                    "role": "system",
                    "content": context["system_prompt"]
                })
            
            # Create the user message with text and image
            content = [
                {"type": "text", "text": prompt}
            ]
            
            # Add image content
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": context.get("image_detail", "high")
                }
            })
            
            # Add the user message
            messages.append({
                "role": "user",
                "content": content
            })
            
            # Call the API
            start_time = time.time()
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=context.get("max_tokens", 1024),
                temperature=context.get("temperature", 0.5)
            )
            end_time = time.time()
            
            # Get the response
            description = response.choices[0].message.content
            
            logger.debug(f"OpenAI vision description generated in {end_time - start_time:.2f}s")
            logger.debug(f"Description length: {len(description)} chars")
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating description with OpenAI vision: {str(e)}")
            return f"Error generating description: {str(e)}"
    
    async def health_check(self) -> bool:
        """Check if the OpenAI vision service is available."""
        if not self.client:
            return False
        
        try:
            # Use a minimal query to check if the service is responding
            test_image_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            test_prompt = "This is a test image. Describe this image in one word."
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": test_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{test_image_data.decode('utf-8')}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=5,
                temperature=0.0
            )
            
            # If we get a response without an error, the service is healthy
            return True
            
        except Exception as e:
            logger.error(f"OpenAI vision health check failed: {str(e)}")
            return False


class AnthropicVisionAdapter(VisionAdapterBase):
    """Vision adapter implementation for Anthropic's Claude models with vision capabilities."""
    
    DEFAULT_MODEL = "claude-3-opus-20240229"
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the Anthropic vision adapter.
        
        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY environment variable
            model_name: Name of the vision model to use. Defaults to claude-3-opus-20240229
            base_url: Base URL for the Anthropic API. If None, uses the default Anthropic URL
        """
        super().__init__(api_key, model_name or self.DEFAULT_MODEL)
        self.base_url = base_url
        self.client = None
        
    async def initialize(self) -> bool:
        """Initialize the Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
            
            # Get API key from environment if not provided
            if not self.api_key:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
                if not self.api_key:
                    logger.error("Anthropic API key not provided and ANTHROPIC_API_KEY not set in environment")
                    return False
            
            # Set up the client
            client_options = {
                "api_key": self.api_key,
                "default_headers": {
                    "User-Agent": "LightRAG-Vision-Adapter"
                }
            }
            
            if self.base_url:
                client_options["base_url"] = self.base_url
            
            self.client = AsyncAnthropic(**client_options)
            self._is_initialized = True
            
            # Verify the client works
            is_healthy = await self.health_check()
            if not is_healthy:
                logger.warning("Anthropic vision service health check failed")
                self._is_initialized = False
                return False
                
            return True
            
        except ImportError:
            logger.error("Anthropic package not installed. Install it with 'pip install anthropic'")
            return False
        except Exception as e:
            logger.error(f"Error initializing Anthropic vision adapter: {str(e)}")
            return False
    
    async def generate_description(
        self, 
        image_data: bytes, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a description using Anthropic's vision model."""
        if not self.is_initialized:
            initialized = await self.initialize()
            if not initialized:
                return "Error: Anthropic vision adapter not initialized"
        
        context = context or {}
        
        try:
            base64_image = self.encode_image_to_base64(image_data)
            
            # Build the messages
            messages = []
            
            # Add system prompt if provided in context
            system_prompt = context.get("system_prompt", "")
            
            # Create the user message with text and image
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                }
            ]
            
            # Call the API
            start_time = time.time()
            response = await self.client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=context.get("max_tokens", 1024),
                temperature=context.get("temperature", 0.5)
            )
            end_time = time.time()
            
            # Get the response
            description = response.content[0].text
            
            logger.debug(f"Anthropic vision description generated in {end_time - start_time:.2f}s")
            logger.debug(f"Description length: {len(description)} chars")
            
            return description
            
        except Exception as e:
            logger.error(f"Error generating description with Anthropic vision: {str(e)}")
            return f"Error generating description: {str(e)}"
    
    async def health_check(self) -> bool:
        """Check if the Anthropic vision service is available."""
        if not self.client:
            return False
        
        try:
            # Use a minimal query to check if the service is responding
            test_image_data = b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            test_prompt = "This is a test image. Describe this image in one word."
            
            response = await self.client.messages.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": test_prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": self.encode_image_to_base64(test_image_data)
                                }
                            }
                        ]
                    }
                ],
                max_tokens=5,
                temperature=0.0
            )
            
            # If we get a response without an error, the service is healthy
            return True
            
        except Exception as e:
            logger.error(f"Anthropic vision health check failed: {str(e)}")
            return False


class VisionAdapterRegistry:
    """Registry for available vision adapters."""
    
    def __init__(self):
        """Initialize the registry."""
        self.adapters = {
            "openai": OpenAIVisionAdapter,
            "anthropic": AnthropicVisionAdapter
        }
        self.initialized_adapters = {}
        
    def get_adapter_class(self, provider: str) -> Optional[type]:
        """
        Get the adapter class for a given provider.
        
        Args:
            provider: Name of the provider (e.g., "openai", "anthropic")
            
        Returns:
            The adapter class or None if not found
        """
        normalized_provider = provider.lower()
        return self.adapters.get(normalized_provider)
    
    async def get_adapter(
        self, 
        provider: str, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> Optional[VisionAdapterBase]:
        """
        Get or create an initialized adapter instance.
        
        Args:
            provider: Name of the provider (e.g., "openai", "anthropic")
            api_key: API key for the provider
            model_name: Name of the model to use
            base_url: Base URL for the API
            
        Returns:
            An initialized adapter instance or None if not available
        """
        normalized_provider = provider.lower()
        
        # Check if we already have an initialized adapter for this provider
        adapter_key = f"{normalized_provider}:{model_name or 'default'}:{base_url or 'default'}"
        if adapter_key in self.initialized_adapters:
            return self.initialized_adapters[adapter_key]
        
        # Get the adapter class
        adapter_class = self.get_adapter_class(normalized_provider)
        if not adapter_class:
            logger.error(f"No vision adapter available for provider '{provider}'")
            return None
        
        # Create and initialize the adapter
        try:
            adapter = adapter_class(api_key=api_key, model_name=model_name, base_url=base_url)
            initialized = await adapter.initialize()
            
            if initialized:
                self.initialized_adapters[adapter_key] = adapter
                return adapter
            else:
                logger.error(f"Failed to initialize {normalized_provider} vision adapter")
                return None
                
        except Exception as e:
            logger.error(f"Error creating {normalized_provider} vision adapter: {str(e)}")
            return None
    
    def list_available_providers(self) -> List[str]:
        """
        List all available vision adapter providers.
        
        Returns:
            List of provider names
        """
        return list(self.adapters.keys())
    
    async def find_best_available_adapter(self) -> Optional[VisionAdapterBase]:
        """
        Find the best available vision adapter based on initialized status.
        
        Returns:
            The best available vision adapter or None if none are available
        """
        # First check already initialized adapters
        if self.initialized_adapters:
            return next(iter(self.initialized_adapters.values()))
        
        # Try to initialize adapters in priority order
        for provider in ["anthropic", "openai"]:
            adapter_class = self.get_adapter_class(provider)
            if adapter_class:
                try:
                    adapter = adapter_class()
                    initialized = await adapter.initialize()
                    if initialized:
                        adapter_key = f"{provider}:default:default"
                        self.initialized_adapters[adapter_key] = adapter
                        return adapter
                except Exception:
                    continue
        
        return None


# Global registry instance
vision_registry = VisionAdapterRegistry()