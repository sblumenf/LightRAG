"""
Cached embedding implementation for LightRAG.

This module provides a caching layer for embeddings that can significantly reduce costs
and improve performance for frequently accessed content.
"""

import logging
import hashlib
import json
import time
from functools import lru_cache
from typing import Dict, List, Any, Callable, Union, Optional
import numpy as np

from lightrag.utils import EmbeddingFunc

# Configure logger
logger = logging.getLogger(__name__)

# In-memory LRU cache for embeddings
_EMBEDDING_CACHE_SIZE = 10000  # Default size, can be overridden in config
_token_cache = {}

class CachedEmbedding:
    """
    A caching wrapper for embedding models that reduces API calls by caching results.
    """
    
    def __init__(
        self,
        embedding_func: EmbeddingFunc,
        cache_size: int = _EMBEDDING_CACHE_SIZE,
        use_persistent_cache: bool = False,
        persistent_cache_path: Optional[str] = None,
        ttl: Optional[int] = None,  # Time to live in seconds
    ):
        """
        Initialize the cached embedding wrapper.
        
        Args:
            embedding_func: The embedding function to wrap
            cache_size: Size of the in-memory LRU cache
            use_persistent_cache: Whether to use persistent caching
            persistent_cache_path: Path to store persistent cache
            ttl: Optional time-to-live for cache entries in seconds
        """
        self.embedding_func = embedding_func
        self.embedding_dim = embedding_func.embedding_dim
        self.max_token_size = embedding_func.max_token_size
        self.cache_size = cache_size
        self.cache = {}
        self.use_persistent_cache = use_persistent_cache
        self.persistent_cache_path = persistent_cache_path
        self.ttl = ttl
        self.cache_hits = 0
        self.cache_misses = 0
        
        # If using persistent cache, try to load it
        if use_persistent_cache and persistent_cache_path:
            self._load_persistent_cache()
            
        # Initialize the per-instance token cache
        self._token_cache = {}
        
    def _text_to_key(self, text: str) -> str:
        """
        Convert text to a cache key.
        
        Args:
            text: Text to convert
            
        Returns:
            str: Cache key
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()
        
    def _load_persistent_cache(self) -> None:
        """
        Load embeddings from persistent cache.
        """
        if not self.persistent_cache_path:
            return
            
        try:
            import json
            with open(self.persistent_cache_path, 'r') as f:
                data = json.load(f)
                
            # Convert lists back to numpy arrays
            for key, value in data.items():
                if isinstance(value, list):
                    self.cache[key] = {
                        'vector': np.array(value['vector']),
                        'timestamp': value.get('timestamp', time.time())
                    }
                    
            logger.info(f"Loaded {len(self.cache)} embeddings from persistent cache")
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Could not load persistent cache: {e}")
            
    def _save_persistent_cache(self) -> None:
        """
        Save embeddings to persistent cache.
        """
        if not self.use_persistent_cache or not self.persistent_cache_path:
            return
            
        try:
            import json
            # Convert numpy arrays to lists for serialization
            serializable_cache = {}
            for key, value in self.cache.items():
                serializable_cache[key] = {
                    'vector': value['vector'].tolist(),
                    'timestamp': value['timestamp']
                }
                
            with open(self.persistent_cache_path, 'w') as f:
                json.dump(serializable_cache, f)
                
            logger.info(f"Saved {len(self.cache)} embeddings to persistent cache")
        except IOError as e:
            logger.warning(f"Could not save persistent cache: {e}")
    
    def _is_cache_entry_valid(self, entry: Dict[str, Any]) -> bool:
        """
        Check if a cache entry is still valid based on TTL.
        
        Args:
            entry: Cache entry to check
            
        Returns:
            bool: True if valid, False if expired
        """
        if self.ttl is None:
            return True
            
        current_time = time.time()
        entry_time = entry.get('timestamp', 0)
        return (current_time - entry_time) <= self.ttl
    
    async def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for texts, using cache when possible.
        
        Args:
            texts: Text or list of texts to embed
            
        Returns:
            np.ndarray: Embeddings
        """
        # Handle single text input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
            
        # Process each text, using cache when possible
        results = []
        texts_to_embed = []
        cache_keys = []
        positions = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._text_to_key(text)
            cache_keys.append(cache_key)
            
            if cache_key in self.cache and self._is_cache_entry_valid(self.cache[cache_key]):
                # Cache hit
                self.cache_hits += 1
                results.append(self.cache[cache_key]['vector'])
            else:
                # Cache miss
                self.cache_misses += 1
                positions.append(i)
                texts_to_embed.append(text)
                results.append(None)  # Placeholder
                
        # Generate embeddings for cache misses
        if texts_to_embed:
            new_embeddings = await self.embedding_func(texts_to_embed)
            
            # Store new embeddings in cache
            current_time = time.time()
            for i, embed_idx in enumerate(positions):
                cache_key = cache_keys[embed_idx]
                self.cache[cache_key] = {
                    'vector': new_embeddings[i],
                    'timestamp': current_time
                }
                results[embed_idx] = new_embeddings[i]
                
            # If cache exceeds size limit, remove oldest entries
            if len(self.cache) > self.cache_size:
                # Remove oldest entries based on timestamp
                sorted_keys = sorted(
                    self.cache.keys(),
                    key=lambda k: self.cache[k].get('timestamp', 0)
                )
                for key in sorted_keys[:len(self.cache) - self.cache_size]:
                    del self.cache[key]
                    
            # Save to persistent cache if enabled
            if self.use_persistent_cache:
                self._save_persistent_cache()
                
        # Convert results to numpy array
        result_array = np.array(results)
        
        # Return single embedding or array depending on input
        if single_input:
            return result_array[0]
        return result_array
        
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache hit/miss statistics.
        
        Returns:
            Dict[str, int]: Cache statistics
        """
        total = self.cache_hits + self.cache_misses
        hit_ratio = self.cache_hits / total if total > 0 else 0
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'total': total,
            'hit_ratio': hit_ratio,
            'cache_size': len(self.cache),
            'cache_limit': self.cache_size
        }
        
    def clear_cache(self) -> None:
        """
        Clear the embedding cache.
        """
        self.cache.clear()
        self._token_cache.clear()
        
def create_cached_embedding(
    embedding_func: EmbeddingFunc,
    cache_size: int = _EMBEDDING_CACHE_SIZE,
    use_persistent_cache: bool = False,
    persistent_cache_path: Optional[str] = None,
    ttl: Optional[int] = None,
) -> EmbeddingFunc:
    """
    Create a cached embedding function.
    
    Args:
        embedding_func: Base embedding function to cache
        cache_size: Size of the in-memory cache
        use_persistent_cache: Whether to use persistent caching
        persistent_cache_path: Path to store persistent cache
        ttl: Optional time-to-live for cache entries in seconds
        
    Returns:
        EmbeddingFunc: Cached embedding function
    """
    cached_embed = CachedEmbedding(
        embedding_func=embedding_func,
        cache_size=cache_size,
        use_persistent_cache=use_persistent_cache,
        persistent_cache_path=persistent_cache_path,
        ttl=ttl
    )
    
    return EmbeddingFunc(
        embedding_dim=embedding_func.embedding_dim,
        max_token_size=embedding_func.max_token_size,
        func=cached_embed
    )