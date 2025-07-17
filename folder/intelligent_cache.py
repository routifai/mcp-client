"""
Intelligent Caching System - Smart caching with different TTLs based on content type
"""

import time
import hashlib
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

################
# CACHE SYSTEM #
################

class IntelligentCache:
    """Smart caching system with different TTLs based on content type"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        
        # Different TTLs for different content types
        self.ttl_config = {
            "news": 300,                     # 5 minutes for news
            "general": 1800,                 # 30 minutes for general
            "science": 3600,                 # 1 hour for science
            "it": 1800,                      # 30 minutes for IT
            "academic": 7200,                # 2 hours for academic
            "social_media": 600,             # 10 minutes for social media
            "shopping": 1800,                # 30 minutes for shopping
            "images": 3600,                  # 1 hour for images
            "videos": 3600,                  # 1 hour for videos
            "maps": 7200,                    # 2 hours for maps
            "music": 3600,                   # 1 hour for music
            "analysis": 7200,                # 2 hours for query analysis
            "scraped": 3600                  # 1 hour for scraped content
        }
    
    def _get_key(self, query: str, category: str = None, content_type: str = "search") -> str:
        """Generate cache key"""
        key_parts = [query, str(category) if category else "general", content_type]
        return hashlib.md5(":".join(key_parts).encode()).hexdigest()
    
    def get(self, query: str, category: str = None, content_type: str = "search") -> Optional[Any]:
        """Get cached item if not expired"""
        key = self._get_key(query, category, content_type)
        
        if key in self.cache:
            entry = self.cache[key]
            ttl = self.ttl_config.get(category, self.ttl_config.get(content_type, 1800))
            
            if time.time() - entry['timestamp'] < ttl:
                logger.debug(f"Cache hit for: {query[:30]}...")
                return entry['data']
            else:
                del self.cache[key]
        
        return None
    
    def set(self, query: str, data: Any, category: str = None, content_type: str = "search"):
        """Cache item with smart TTL"""
        key = self._get_key(query, category, content_type)
        
        # LRU eviction if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'data': data,
            'timestamp': time.time()
        }
        logger.debug(f"Cached: {query[:30]}... (type: {content_type})")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        now = time.time()
        active_entries = 0
        expired_entries = 0
        
        for key, entry in self.cache.items():
            # Estimate TTL (use default if unknown)
            if now - entry['timestamp'] < 1800:  # Default TTL
                active_entries += 1
            else:
                expired_entries += 1
        
        return {
            "total_entries": len(self.cache),
            "active_entries": active_entries,
            "expired_entries": expired_entries,
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }

# Global cache instance
intelligent_cache = IntelligentCache()