"""
Embedding generation module for LightRAG.

This module provides functionality to generate embeddings for text chunks using
different embedding models (Google Gemini, OpenAI).
"""

import logging
import time
import os
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

# Set up logger first so it can be used in the import section
logger = logging.getLogger(__name__)

# Define a dummy class if TextChunk is not essential for basic loading
try:
    from lightrag.text_chunking.text_chunk import TextChunk
except ImportError:
    logger.debug("Could not import TextChunk from lightrag. Using a dummy definition.")
    class TextChunk:
        def __init__(self, text: str, embedding: Optional[List[float]] = None):
            self.text = text
            self.embedding = embedding


# Try to import settings from config, or use dummy settings if unavailable
try:
    from lightrag.config import settings
    logger.debug("Successfully imported settings from lightrag.config")
except ImportError:
    logger.debug("Could not import settings from config. Using dummy settings.")
    # Define dummy settings if config is unavailable (useful for isolated testing)
    class DummySettings:
        class EmbeddingSettings:
            provider = 'google'  # Default provider
            google_model = 'models/embedding-001'  # Default Google model
            openai_model = 'text-embedding-3-small'  # Default OpenAI model
            batch_size = 50
            max_retries = 3
            retry_delay = 2.0
            vector_dimensions = 768  # Default dimension (e.g., for embedding-001)

        class LLMSettings:
            google_api_key = None
            openai_api_key = None

        embedding = EmbeddingSettings()
        llm = LLMSettings()

        # For backward compatibility
        GOOGLE_API_KEY = None
        OPENAI_API_KEY = None
        EMBEDDING_PROVIDER = embedding.provider
        DEFAULT_EMBEDDING_MODEL = embedding.google_model
        DEFAULT_OPENAI_EMBEDDING_MODEL = embedding.openai_model
        EMBEDDING_BATCH_SIZE = embedding.batch_size
        EMBEDDING_MAX_RETRIES = embedding.max_retries
        EMBEDDING_RETRY_DELAY = embedding.retry_delay
        VECTOR_DIMENSIONS = embedding.vector_dimensions

    # Initialize settings with environment variables if available
    settings = DummySettings()
    settings.llm.google_api_key = os.getenv("GOOGLE_API_KEY")
    settings.llm.openai_api_key = os.getenv("OPENAI_API_KEY")


# Handle optional dependencies for provider-specific exceptions
try:
    import google.api_core.exceptions
    import google.generativeai # Needed for type hints/checks later
except ImportError:
    google = None # Flag that google library is not available
try:
    import openai
    # Import specific exceptions for v1.x+
    # Import base APIError and specific retryable errors
    from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, BadRequestError, AuthenticationError
except ImportError:
    openai = None # Flag that openai library is not available
    # Define dummy exceptions if openai is not installed
    APIError = type('DummyOpenAIAPIError', (Exception,), {})
    RateLimitError = APIConnectionError = APITimeoutError = InternalServerError = BadRequestError = AuthenticationError = type('DummyOpenAIError', (APIError,), {})


class EmbeddingError(Exception):
    """Custom exception for embedding generation errors."""
    pass

# Lazy initialization globals
_google_ai_module = None
_openai_client = None # Store the client instance for OpenAI v1.x+

def _load_google_ai():
    """Lazy load Google AI library and configure it."""
    global _google_ai_module
    if _google_ai_module is None:
        if not google: # Check if import failed initially
             logger.error("Google Generative AI SDK not found. Please install it using 'pip install google-generativeai'")
             raise EmbeddingError("Google Generative AI SDK not installed.")
        try:
            # The module itself is stored in the 'google.generativeai' namespace
            _google_ai_module = google.generativeai
            if not settings.llm.google_api_key:
                 raise EmbeddingError("Google API Key not found in settings.")
            _google_ai_module.configure(api_key=settings.llm.google_api_key)
            logger.debug("Google Generative AI SDK loaded and configured.")
        except AttributeError as e:
             logger.error(f"Failed to configure Google AI library. Check API key and library version. Error: {e}")
             raise EmbeddingError(f"Failed to configure Google AI library: {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred during Google AI configuration: {e}", exc_info=True)
            raise EmbeddingError(f"An unexpected error occurred during Google AI configuration: {e}") from e
    return _google_ai_module

def _load_openai():
    """Lazy load OpenAI library and initialize client."""
    global _openai_client
    if _openai_client is None:
        if not openai: # Check if import failed initially
            logger.error("OpenAI SDK not found. Please install it using 'pip install openai>=1.0.0'")
            raise EmbeddingError("OpenAI SDK (>=1.0.0) not installed.")
        try:
            if not settings.llm.openai_api_key:
                 raise EmbeddingError("OpenAI API Key not found in settings.")
            # Use the modern client initialization (v1.x+)
            _openai_client = openai.OpenAI(api_key=settings.llm.openai_api_key)
            logger.debug("OpenAI SDK loaded and client initialized.")
        except Exception as e:
            if "authentication" in str(e).lower() or "auth" in str(e).lower() or "api key" in str(e).lower():
                logger.error(f"OpenAI authentication failed. Check your API key: {e}")
                raise EmbeddingError(f"OpenAI authentication failed: {e}") from e
            else:
                logger.exception(f"An unexpected error occurred during OpenAI client initialization: {e}", exc_info=True)
                raise EmbeddingError(f"An unexpected error occurred during OpenAI client initialization: {e}") from e
    return _openai_client


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using different embedding models.
    Handles batching, retries, and provider-specific logic.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ):
        """
        Initialize the embedding generator.

        Prioritizes configuration sources: config dict > arguments > settings.

        Args:
            config: Optional configuration dictionary for overrides.
            provider: The embedding provider ('google' or 'openai').
            model_name: The name of the embedding model to use.
            batch_size: Number of texts to process in a single API call.
            max_retries: Maximum number of retries for API calls.
            retry_delay: Delay between retries in seconds.
        """
        self.config = config if config is not None else {}

        # Determine configuration values with priority: config > args > settings
        self.provider = self.config.get('provider', provider if provider else settings.embedding.provider).lower()
        self.batch_size = int(self.config.get('batch_size', batch_size if batch_size is not None else settings.embedding.batch_size))
        self.max_retries = int(self.config.get('max_retries', max_retries if max_retries is not None else settings.embedding.max_retries))
        self.retry_delay = float(self.config.get('retry_delay', retry_delay if retry_delay is not None else settings.embedding.retry_delay))

        # Determine model_name based on provider, prioritizing config -> args -> settings default
        _model_name_config_or_arg = self.config.get('model_name', model_name)
        if _model_name_config_or_arg:
            self.model_name = _model_name_config_or_arg
        else:
            if self.provider == 'google':
                self.model_name = settings.embedding.google_model
            elif self.provider == 'openai':
                self.model_name = settings.embedding.openai_model
            else:
                raise ValueError(f"Unsupported or unspecified embedding provider: '{self.provider}'")

        # Initialize the embedding client/module and determine vector dimension
        self.genai = None # Google module/client
        self.openai = None # OpenAI client instance
        self.vector_dim = None # Dimension of the embeddings

        if self.provider == 'google':
            if not google: raise EmbeddingError("Google SDK requested but not installed.")
            self.genai = _load_google_ai()

            # Try to dynamically determine embedding dimensions
            try:
                # First attempt: Try to get a sample embedding to determine dimensions
                logger.info(f"Attempting to dynamically determine embedding dimensions for Google model: {self.model_name}")
                sample_text = "This is a test to determine embedding dimensions."
                embedding_result = self.genai.embed_content(
                    model=self.model_name,
                    content=sample_text,
                    task_type="RETRIEVAL_DOCUMENT"
                )

                if "embedding" in embedding_result and isinstance(embedding_result["embedding"], list):
                    self.vector_dim = len(embedding_result["embedding"])
                    logger.info(f"Successfully determined embedding dimensions dynamically: {self.vector_dim}")
                else:
                    raise ValueError("Could not extract embedding from response")

            except Exception as e:
                # Fallback to settings if dynamic determination fails
                logger.warning(f"Failed to dynamically determine embedding dimensions for Google model '{self.model_name}': {e}")
                logger.warning(f"Falling back to configured dimensions from settings: {settings.embedding.vector_dimensions}")
                self.vector_dim = settings.embedding.vector_dimensions

            logger.info(f"Initialized Google Embedding provider. Model: {self.model_name} (Dimensions: {self.vector_dim})")

        elif self.provider == 'openai':
            if not openai: raise EmbeddingError("OpenAI SDK requested but not installed.")
            self.openai = _load_openai() # Gets the initialized client instance

            # Try to dynamically determine embedding dimensions
            try:
                # First attempt: Try to get model information
                logger.info(f"Attempting to dynamically determine embedding dimensions for OpenAI model: {self.model_name}")

                # OpenAI doesn't have a direct method to get model dimensions, so we'll use a sample embedding
                sample_text = "This is a test to determine embedding dimensions."
                response = self.openai.embeddings.create(
                    model=self.model_name,
                    input=sample_text
                )

                if response.data and len(response.data) > 0 and hasattr(response.data[0], 'embedding'):
                    self.vector_dim = len(response.data[0].embedding)
                    logger.info(f"Successfully determined embedding dimensions dynamically: {self.vector_dim}")
                else:
                    raise ValueError("Could not extract embedding from response")

            except Exception as e:
                # Fallback to settings if dynamic determination fails
                logger.warning(f"Failed to dynamically determine embedding dimensions for OpenAI model '{self.model_name}': {e}")
                logger.warning(f"Falling back to configured dimensions from settings: {settings.embedding.vector_dimensions}")
                self.vector_dim = settings.embedding.vector_dimensions

            logger.info(f"Initialized OpenAI Embedding provider. Model: {self.model_name} (Dimensions: {self.vector_dim})")

        else:
            raise ValueError(f"Initialization failed for unsupported provider: '{self.provider}'")

        if self.vector_dim is None:
             # This should ideally not happen if settings are configured correctly
             logger.error(f"Could not determine vector dimension for provider '{self.provider}' and model '{self.model_name}'. Check settings.embedding.vector_dimensions.")
             raise EmbeddingError(f"Could not determine vector dimension for provider '{self.provider}' and model '{self.model_name}'.")

    def _get_zero_vector(self) -> List[float]:
        """Returns a zero vector of the correct dimension for the provider."""
        if self.vector_dim is None:
            logger.error("Vector dimension is None when trying to create zero vector. Defaulting to 1.")
            return [0.0]
        return [0.0] * self.vector_dim

    def _is_retryable_error(self, e: Exception) -> bool:
        """
        Determines if an error is retryable.

        Args:
            e: The exception to check.

        Returns:
            bool: True if the error is retryable, False otherwise.
        """
        is_retryable = False

        # Check specific library errors first
        if self.provider == 'google' and google:
            try:
                if isinstance(e, (google.api_core.exceptions.ServiceUnavailable, google.api_core.exceptions.ResourceExhausted)):
                    is_retryable = True
            except (TypeError, AttributeError):
                # If there's an issue with the type check, check by name
                error_type = type(e).__name__
                if "ServiceUnavailable" in error_type or "ResourceExhausted" in error_type:
                    is_retryable = True
        elif self.provider == 'openai' and openai:
            # Check for OpenAI error types by name since we can't import them directly
            error_type = type(e).__name__
            if error_type in ["RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"]:
                is_retryable = True

        # Check for generic retryable errors IF NOT already identified by library-specific checks
        if not is_retryable and isinstance(e, (TimeoutError, ConnectionError)): # Add generic checks
            logger.debug(f"Caught generic error '{type(e).__name__}' - considering retryable.")
            is_retryable = True # Treat these as retryable regardless of provider

        # For testing purposes, check if the error message contains 'rate limit' or similar
        if not is_retryable and str(e).lower().find('rate limit') >= 0:
            logger.debug(f"Caught error with 'rate limit' in message - considering retryable for testing.")
            is_retryable = True

        return is_retryable

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text. Handles retries.

        Args:
            text: The text to embed.

        Returns:
           List[float]: The embedding vector. Returns zero vector for empty input.
        """
        logger.debug(f"Generating single embedding for text: {text[:50]}...")
        if not text or not text.strip():
            logger.warning("Received empty or whitespace-only text for single embedding, returning zero vector.")
            return self._get_zero_vector()

        retries = 0
        while retries <= self.max_retries:
            try:
                if self.provider == 'google':
                    if not self.genai: raise EmbeddingError("Google client/module not initialized.")
                    embedding_result = self.genai.embed_content(
                        model=self.model_name,
                        content=text,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    if "embedding" not in embedding_result or not isinstance(embedding_result["embedding"], list):
                        logger.error(f"Google API response missing 'embedding' list for text: {text[:50]}... Response: {embedding_result}")
                        raise EmbeddingError(f"Google API response missing 'embedding' list for text: {text[:50]}...")
                    return embedding_result["embedding"]

                elif self.provider == 'openai':
                    if not self.openai: raise EmbeddingError("OpenAI client not initialized.")
                    response = self.openai.embeddings.create(
                        model=self.model_name,
                        input=text
                    )
                    if response.data and len(response.data) > 0 and hasattr(response.data[0], 'embedding'):
                        return response.data[0].embedding
                    else:
                        logger.error(f"OpenAI API returned unexpected data structure for text: {text[:50]}... Response: {response}")
                        raise EmbeddingError(f"OpenAI API returned unexpected data structure for text: {text[:50]}...")
                else:
                    raise EmbeddingError(f"Unsupported embedding provider configured: {self.provider}")

            # --- Refined Exception Handling ---
            except Exception as e:
                is_retryable = self._is_retryable_error(e)

                # --- Handle Retryable Errors ---
                if is_retryable:
                    retries += 1
                    if retries > self.max_retries:
                        logger.error(f"{self.provider} Embedding API error after {self.max_retries} retries for text '{text[:50]}...': {e}")
                        raise EmbeddingError(f"Failed {self.provider} after {self.max_retries} retries for text '{text[:50]}...': {e}") from e
                    # Calculate delay with exponential backoff
                    current_delay = self.retry_delay * (2 ** (retries - 1))
                    logger.warning(
                        f"{self.provider} Embedding API transient error (attempt {retries}/{self.max_retries}) for text '{text[:50]}...': {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    continue # Go to the next iteration of the while loop

                # --- Handle Non-Retryable or Unexpected Errors ---
                else:
                    is_non_retryable_api_error = False
                    if self.provider == 'google' and google:
                        # Includes InvalidArgument, PermissionDenied etc.
                        try:
                            if isinstance(e, google.api_core.exceptions.GoogleAPIError):
                                is_non_retryable_api_error = True
                        except (TypeError, AttributeError):
                            # If there's an issue with the type check, check by name
                            error_type = type(e).__name__
                            if "GoogleAPIError" in error_type:
                                is_non_retryable_api_error = True
                    elif self.provider == 'openai' and openai:
                         # APIError is base for BadRequest, Auth etc.
                        # Check for OpenAI error types by name since we can't import them directly
                        error_type = type(e).__name__
                        if error_type == "APIError" or "Error" in error_type and "API" in error_type:
                            is_non_retryable_api_error = True

                    if is_non_retryable_api_error:
                         logger.error(f"Non-retryable {self.provider} embedding API error for text '{text[:50]}...': {e}")
                         raise EmbeddingError(f"{self.provider} API error for text '{text[:50]}...': {e}") from e
                    else:
                         # If it wasn't retryable and wasn't a known non-retryable API error, raise as unexpected
                         logger.exception(f"Unexpected error during {self.provider} embedding generation for text '{text[:50]}...': {e}", exc_info=True)
                         raise EmbeddingError(f"Unexpected error for {self.provider} text '{text[:50]}...': {e}") from e
            # --- End of Refined Exception Handling ---

        # Should only be reached if loop finishes without returning (e.g., max_retries < 0)
        logger.error(f"Failed to generate {self.provider} embedding for text '{text[:50]}...' after exhausting retries or unexpected loop exit.")
        raise EmbeddingError(f"Failed to generate {self.provider} embedding for text '{text[:50]}...' after {self.max_retries} retries.")

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts. Handles provider specifics,
        batching, retries, and empty strings.

        Args:
            texts: List of texts to embed.

        Returns:
            List[List[float]]: List of embedding vectors in the same order as input texts.
                               Zero vectors are returned for corresponding empty input texts.
        """
        if not texts:
            return []

        embeddings = [] # Final list of embeddings in the original order
        num_texts = len(texts)
        num_batches = (num_texts + self.batch_size - 1) // self.batch_size
        logger.info(f"Starting batch embedding generation for {num_texts} texts in {num_batches} batches (batch size: {self.batch_size}) using {self.provider}.")

        for i in range(0, num_texts, self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_index = (i // self.batch_size) + 1 # Correct 1-based index for logging

            logger.info(f"Processing batch {batch_index}/{num_batches} ({len(batch_texts)} texts)")

            filtered_batch = [(idx, text) for idx, text in enumerate(batch_texts) if text and text.strip()]

            if not filtered_batch:
                logger.warning(f"Batch {batch_index}/{num_batches} contains only empty/whitespace texts. Adding zero vectors.")
                batch_embeddings = [self._get_zero_vector()] * len(batch_texts)
                embeddings.extend(batch_embeddings)
                continue

            filtered_indices, filtered_texts = zip(*filtered_batch)

            retries = 0
            current_filtered_embeddings = None

            while retries <= self.max_retries:
                try:
                    if self.provider == 'google':
                        if not self.genai: raise EmbeddingError("Google client/module not initialized.")
                        embedding_results = self.genai.batch_embed_contents(
                            model=self.model_name,
                            requests=[{'model': self.model_name, 'content': text, 'task_type': "RETRIEVAL_DOCUMENT"} for text in filtered_texts]
                        )
                        # Check if embeddings key exists and is a list-like object
                        if "embeddings" not in embedding_results or not hasattr(embedding_results["embeddings"], "__iter__"):
                             logger.error(f"Google batch API response missing 'embeddings' list for batch {batch_index}. Response: {embedding_results}")
                             raise EmbeddingError(f"Google batch API response missing 'embeddings' list for batch {batch_index}")
                        batch_result_embeddings = []
                        for result in embedding_results["embeddings"]:
                             # Check if result is dict-like and has embedding key with list-like value
                             if not hasattr(result, "__getitem__") or "embedding" not in result or not hasattr(result["embedding"], "__iter__"):
                                 logger.error(f"Google batch API response item malformed (missing 'embedding' list) in batch {batch_index}. Item: {result}")
                                 raise EmbeddingError(f"Google batch API response item malformed in batch {batch_index}")
                             batch_result_embeddings.append(result["embedding"])
                        if len(batch_result_embeddings) != len(filtered_texts):
                            logger.error(f"Google batch API returned mismatched embedding count for batch {batch_index}. Expected {len(filtered_texts)}, got {len(batch_result_embeddings)}.")
                            raise EmbeddingError(f"Google batch API returned mismatched embedding count for batch {batch_index}")
                        current_filtered_embeddings = batch_result_embeddings

                    elif self.provider == 'openai':
                        if not self.openai: raise EmbeddingError("OpenAI client not initialized.")
                        response = self.openai.embeddings.create(
                            model=self.model_name,
                            input=list(filtered_texts)
                        )
                        if not response.data or len(response.data) != len(filtered_texts):
                             logger.error(f"OpenAI API returned mismatched data length for batch {batch_index}. Expected {len(filtered_texts)}, got {len(response.data) if response.data else 0}. Response: {response}")
                             raise EmbeddingError(f"OpenAI API returned mismatched data for batch {batch_index}")
                        # Check if all items have required attributes
                        missing_attributes = False
                        for item in response.data:
                            if not hasattr(item, 'index') or not hasattr(item, 'embedding'):
                                missing_attributes = True
                                break

                        if missing_attributes:
                             logger.error(f"OpenAI API response item missing 'index' or 'embedding' attribute in batch {batch_index}. Response: {response}")
                             raise EmbeddingError(f"OpenAI API response item missing 'index' or 'embedding' attribute in batch {batch_index}")

                        try:
                            sorted_data = sorted(response.data, key=lambda item: item.index)
                            current_filtered_embeddings = [item.embedding for item in sorted_data]
                        except Exception as e:
                            logger.error(f"Error processing OpenAI API response in batch {batch_index}: {e}")
                            raise EmbeddingError(f"OpenAI API response item missing 'index' or 'embedding' attribute in batch {batch_index}") from e

                    else:
                         raise EmbeddingError(f"Unsupported provider '{self.provider}' encountered during batch processing.")

                    logger.debug(f"Successfully processed API call for batch {batch_index}/{num_batches}")
                    break # Exit the while loop for this batch

                # --- Refined Exception Handling for Batch ---
                except Exception as e:
                    is_retryable = self._is_retryable_error(e)

                    if is_retryable:
                        retries += 1
                        if retries > self.max_retries:
                            logger.error(f"{self.provider} Embedding API error after {self.max_retries} retries for batch {batch_index}/{num_batches}: {e}")
                            raise EmbeddingError(f"Failed {self.provider} batch {batch_index}/{num_batches} after {self.max_retries} retries: {e}") from e
                        # Calculate delay with exponential backoff
                        current_delay = self.retry_delay * (2 ** (retries - 1))
                        logger.warning(
                            f"{self.provider} Embedding API transient error (attempt {retries}/{self.max_retries}) for batch {batch_index}/{num_batches}: {e}. "
                            f"Retrying in {current_delay}s..."
                        )
                        time.sleep(current_delay)
                        continue # Retry the current batch
                    else:
                        # Handle non-retryable or unexpected errors
                        is_non_retryable_api_error = False
                        if self.provider == 'google' and google:
                            # Check if google.api_core.exceptions.GoogleAPIError is available
                            try:
                                if isinstance(e, google.api_core.exceptions.GoogleAPIError):
                                    is_non_retryable_api_error = True
                            except (TypeError, AttributeError):
                                # If there's an issue with the type check, check by name
                                error_type = type(e).__name__
                                if "GoogleAPIError" in error_type:
                                    is_non_retryable_api_error = True
                        elif self.provider == 'openai' and openai:
                            # Check for OpenAI error types by name since we can't import them directly
                            error_type = type(e).__name__
                            if error_type == "APIError" or "Error" in error_type and "API" in error_type:
                                is_non_retryable_api_error = True

                        if is_non_retryable_api_error:
                           logger.error(f"Non-retryable {self.provider} embedding API error for batch {batch_index}/{num_batches}: {e}")
                           raise EmbeddingError(f"{self.provider} API error on batch {batch_index}/{num_batches}: {e}") from e
                        else:
                           logger.exception(f"Unexpected error during {self.provider} embedding for batch {batch_index}/{num_batches}: {e}", exc_info=True)
                           raise EmbeddingError(f"Unexpected error on {self.provider} batch {batch_index}/{num_batches}: {e}") from e
                # --- End of Refined Exception Handling ---

            if current_filtered_embeddings is None:
                 logger.error(f"Failed to process {self.provider} batch {batch_index}/{num_batches} after {self.max_retries} retries (loop exhausted).")
                 raise EmbeddingError(f"Failed to process {self.provider} batch {batch_index}/{num_batches} after {self.max_retries} retries.")

            batch_embeddings = [None] * len(batch_texts)
            for original_batch_idx, original_text_idx in enumerate(filtered_indices):
                batch_embeddings[original_text_idx] = current_filtered_embeddings[original_batch_idx]

            for idx in range(len(batch_texts)):
                 if batch_embeddings[idx] is None:
                     batch_embeddings[idx] = self._get_zero_vector()

            if len(batch_embeddings) != len(batch_texts):
                 logger.error(f"Internal logic error: Reconstructed batch {batch_index} size ({len(batch_embeddings)}) doesn't match original batch size ({len(batch_texts)}).")
                 while len(batch_embeddings) < len(batch_texts): batch_embeddings.append(self._get_zero_vector())
                 batch_embeddings = batch_embeddings[:len(batch_texts)]

            embeddings.extend(batch_embeddings)
            logger.info(f"Finished processing batch {batch_index}/{num_batches}.")

        if len(embeddings) != num_texts:
             logger.error(f"Critical Error: Final embedding count ({len(embeddings)}) does not match input text count ({num_texts}). Padding with zero vectors, but this indicates a significant issue.")
             missing_count = num_texts - len(embeddings)
             embeddings.extend([self._get_zero_vector()] * missing_count)

        logger.info(f"Finished batch embedding generation for all {num_texts} texts.")
        return embeddings

    def embed_text_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """
        Generate embeddings for a list of TextChunk objects, modifying them in place.

        Args:
            chunks: List of TextChunk objects.

        Returns:
            List[TextChunk]: The same list of chunks with embeddings added/updated.
        """
        if not chunks:
            return []

        logger.info(f"Starting embedding process for {len(chunks)} TextChunk objects.")
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batches
        embeddings_list = self.generate_embeddings_batch(texts)

        # Add embeddings back to chunks
        if len(chunks) != len(embeddings_list):
             logger.error(f"Mismatch between number of chunks ({len(chunks)}) and generated embeddings ({len(embeddings_list)}) in embed_text_chunks. This should not happen.")
             min_len = min(len(chunks), len(embeddings_list))
             for i in range(min_len):
                 chunks[i].embedding = embeddings_list[i]
             for i in range(min_len, len(chunks)):
                 logger.warning(f"Assigning zero vector to chunk {i} due to embedding count mismatch.")
                 chunks[i].embedding = self._get_zero_vector()
        else:
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings_list[i]

        logger.info(f"Finished embedding process for {len(chunks)} TextChunk objects.")
        return chunks
