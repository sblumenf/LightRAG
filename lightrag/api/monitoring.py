"""
Monitoring module for LightRAG API.

This module provides monitoring capabilities for the LightRAG API, including:
- Prometheus metrics
- Health checks
- Custom metrics for LightRAG
"""

import time
import logging
import threading
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Configure logger
logger = logging.getLogger(__name__)

# Define Prometheus metrics
REQUESTS_TOTAL = Counter(
    'lightrag_requests_total', 
    'Total number of requests processed',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'lightrag_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'lightrag_active_requests',
    'Number of active requests'
)

EMBEDDING_TIME = Histogram(
    'lightrag_embedding_time_seconds',
    'Time to generate embeddings',
    ['model']
)

LLM_TOKENS_TOTAL = Counter(
    'lightrag_llm_tokens_total',
    'Total number of tokens processed by the LLM',
    ['model', 'operation']
)

DOCUMENT_PROCESSING_TIME = Histogram(
    'lightrag_document_processing_seconds',
    'Document processing time',
    ['doc_type']
)

DOCUMENTS_PROCESSED = Counter(
    'lightrag_documents_processed_total',
    'Total number of documents processed',
    ['status', 'doc_type']
)

RETRIEVAL_TIME = Histogram(
    'lightrag_retrieval_time_seconds',
    'Retrieval operation time',
    ['mode']
)

RETRIEVAL_COUNT = Histogram(
    'lightrag_retrieval_results_count',
    'Number of results returned by retrieval operations',
    ['mode']
)

CACHE_OPERATIONS = Counter(
    'lightrag_cache_operations_total',
    'Total number of cache operations',
    ['operation', 'cache_type']
)

CACHE_HIT_RATIO = Gauge(
    'lightrag_cache_hit_ratio',
    'Cache hit ratio (0.0-1.0)',
    ['cache_type']
)

ERROR_COUNT = Counter(
    'lightrag_errors_total',
    'Total number of errors',
    ['error_type']
)

# System metrics
SYSTEM_MEMORY_USAGE = Gauge(
    'lightrag_system_memory_usage_bytes',
    'System memory usage in bytes'
)

SYSTEM_CPU_USAGE = Gauge(
    'lightrag_system_cpu_usage_percent',
    'System CPU usage percentage'
)

# Last metrics update time
last_system_metrics_update = datetime.now() - timedelta(minutes=5)  # Start with update needed
system_metrics_lock = threading.Lock()

class MetricsMiddleware:
    """Middleware for collecting request metrics."""
    
    async def __call__(self, request: Request, call_next: Callable) -> Response:
        """Process the request and record metrics."""
        # Record request start time
        request_start_time = time.time()
        
        # Track active requests
        ACTIVE_REQUESTS.inc()
        
        method = request.method
        path = request.url.path
        
        try:
            # Call the next middleware or endpoint handler
            response = await call_next(request)
            
            # Record metrics
            status = response.status_code
            latency = time.time() - request_start_time
            
            REQUESTS_TOTAL.labels(method=method, endpoint=path, status=status).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=path).observe(latency)
            
            return response
        except Exception as e:
            # Record error metrics
            ERROR_COUNT.labels(error_type=type(e).__name__).inc()
            REQUESTS_TOTAL.labels(method=method, endpoint=path, status=500).inc()
            
            # Re-raise the exception
            raise
        finally:
            # Always decrement active requests counter
            ACTIVE_REQUESTS.dec()

async def get_metrics():
    """Get Prometheus metrics for the /metrics endpoint."""
    # Update system metrics if needed
    update_system_metrics()
    
    # Generate and return metrics
    return generate_latest()

def update_system_metrics():
    """Update system metrics (CPU, memory) if enough time has passed."""
    global last_system_metrics_update
    
    # Only update every 15 seconds to avoid overhead
    with system_metrics_lock:
        now = datetime.now()
        if (now - last_system_metrics_update).total_seconds() < 15:
            return
            
        try:
            # Memory usage
            if os.path.exists('/proc/meminfo'):
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                
                # Parse memory info
                mem_total = 0
                mem_free = 0
                mem_available = 0
                
                for line in meminfo.split('\n'):
                    if 'MemTotal' in line:
                        mem_total = int(line.split()[1]) * 1024  # Convert to bytes
                    elif 'MemFree' in line:
                        mem_free = int(line.split()[1]) * 1024
                    elif 'MemAvailable' in line:
                        mem_available = int(line.split()[1]) * 1024
                
                if mem_total > 0 and mem_available > 0:
                    mem_used = mem_total - mem_available
                    SYSTEM_MEMORY_USAGE.set(mem_used)
            
            # CPU usage
            if os.path.exists('/proc/stat'):
                with open('/proc/stat', 'r') as f:
                    cpu_info = f.readline()
                
                if cpu_info.startswith('cpu '):
                    # Parse CPU info
                    cpu_parts = cpu_info.split()
                    idle = float(cpu_parts[4])
                    total = sum(float(x) for x in cpu_parts[1:])
                    
                    if total > 0:
                        cpu_usage = 100.0 * (1.0 - idle / total)
                        SYSTEM_CPU_USAGE.set(cpu_usage)
        
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {str(e)}")
        
        # Update last update time
        last_system_metrics_update = now

# Cache statistics tracking
class CacheStats:
    """Helper class to track cache statistics."""
    
    def __init__(self, cache_type: str):
        """Initialize cache stats tracker."""
        self.cache_type = cache_type
        self.hits = 0
        self.misses = 0
        
    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1
        CACHE_OPERATIONS.labels(operation='hit', cache_type=self.cache_type).inc()
        self._update_ratio()
        
    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1
        CACHE_OPERATIONS.labels(operation='miss', cache_type=self.cache_type).inc()
        self._update_ratio()
        
    def record_set(self):
        """Record a cache set operation."""
        CACHE_OPERATIONS.labels(operation='set', cache_type=self.cache_type).inc()
        
    def record_delete(self):
        """Record a cache delete operation."""
        CACHE_OPERATIONS.labels(operation='delete', cache_type=self.cache_type).inc()
        
    def _update_ratio(self):
        """Update the cache hit ratio metric."""
        total = self.hits + self.misses
        if total > 0:
            CACHE_HIT_RATIO.labels(cache_type=self.cache_type).set(self.hits / total)

# Create cache stats trackers for different cache types
query_cache_stats = CacheStats('query')
embedding_cache_stats = CacheStats('embedding')
llm_cache_stats = CacheStats('llm')

# Health check metrics
health_status = {
    'system': True,
    'database': True,
    'vector_db': True,
    'llm_service': True,
    'last_check': datetime.now().isoformat()
}

def update_health_status(component: str, status: bool):
    """Update the health status of a component."""
    health_status[component] = status
    health_status['last_check'] = datetime.now().isoformat()

def get_health_status():
    """Get the current health status."""
    return health_status