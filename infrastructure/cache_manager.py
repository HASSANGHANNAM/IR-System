"""
Cache Manager - Handles caching for search results.
Contains one class: CacheManager.
"""

import logging
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Central cache manager with LRU eviction policy.
    Supports separate caches for different types (e.g., 'query').
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache manager.
        
        Args:
            max_size: Maximum number of items to store in each cache.
        """
        self.max_size = max_size
        self._cache = OrderedDict()      # default cache
        self._query_cache = OrderedDict() # query cache
    
    def get(self, key: Any, cache_type: str = 'default') -> Optional[Any]:
        """
        Retrieve a value from the cache.
        
        Args:
            key: The cache key.
            cache_type: 'default' or 'query'.
            
        Returns:
            The cached value, or None if not found.
        """
        cache = self._query_cache if cache_type == 'query' else self._cache
        value = cache.get(key)
        if value is not None:
            # Move to end to mark as recently used (LRU)
            cache.move_to_end(key)
        return value
    
    def set(self, key: Any, value: Any, cache_type: str = 'default') -> None:
        """
        Store a value in the cache.
        
        Args:
            key: The cache key.
            value: The value to store.
            cache_type: 'default' or 'query'.
        """
        cache = self._query_cache if cache_type == 'query' else self._cache
        
        # If key already exists, update and move to end
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        
        # Evict oldest item if cache exceeds max size
        if len(cache) > self.max_size:
            oldest_key, _ = cache.popitem(last=False)
            logger.debug(f"Cache evicted oldest item: {oldest_key}")
    
    def clear(self, cache_type: Optional[str] = None) -> None:
        """
        Clear the cache.
        
        Args:
            cache_type: 'default', 'query', or None to clear all.
        """
        if cache_type == 'query':
            self._query_cache.clear()
            logger.info("Query cache cleared")
        elif cache_type == 'default':
            self._cache.clear()
            logger.info("Default cache cleared")
        else:
            self._cache.clear()
            self._query_cache.clear()
            logger.info("All caches cleared")
    
    def get_or_compute(self, key: Any, compute_func, cache_type: str = 'default') -> Any:
        """
        Get from cache or compute and store.
        
        Args:
            key: Cache key.
            compute_func: Function that computes the value if not cached.
            cache_type: 'default' or 'query'.
            
        Returns:
            The cached or computed value.
        """
        value = self.get(key, cache_type)
        if value is None:
            value = compute_func()
            self.set(key, value, cache_type)
        return value
    
    def size(self, cache_type: str = 'default') -> int:
        """Return the current size of the specified cache."""
        cache = self._query_cache if cache_type == 'query' else self._cache
        return len(cache)