# LightRAG Scaling Guide

This document provides comprehensive guidance for scaling LightRAG to handle high-volume production workloads.

## System Architecture Overview

LightRAG's architecture consists of several components that can be scaled independently:

1. **API Service**: The main entry point for queries and document uploads
2. **Vector Database**: Stores embeddings for semantic search
3. **Graph Database**: Stores entity and relationship data
4. **Document Storage**: Persists document content and metadata
5. **LLM Integration**: Handles communication with language models

## Performance Benchmarks

To understand scaling requirements, use the built-in benchmark tools:

```bash
# Run benchmark with default configuration
python scripts/run_benchmarks.py

# Run specific benchmark with custom parameters
python scripts/run_benchmarks.py --component retrieval --queries 1000 --concurrency 10
```

Expected performance metrics on reference hardware (8-core, 16GB RAM):
- Document processing: ~5 pages per second
- Query processing: ~5-20 queries per second (depends on complexity)
- Response generation: ~1-3 seconds per response

## Horizontal Scaling

### API Service Scaling

The LightRAG API service is stateless and can be horizontally scaled:

#### Docker Swarm Configuration

```yaml
# docker-compose.yml with swarm scaling
services:
  lightrag:
    image: lightrag:latest
    deploy:
      replicas: 5
      resources:
        limits:
          cpus: '2'
          memory: 4G
    ports:
      - "9621:9621"
    volumes:
      - shared_data:/app/data
    environment:
      - DATABASE_URL=neo4j://neo4j:7687
      - VECTOR_DB_HOST=milvus

volumes:
  shared_data:
    driver: local
```

#### Kubernetes Configuration

```yaml
# kubernetes/lightrag-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightrag
spec:
  replicas: 5
  selector:
    matchLabels:
      app: lightrag
  template:
    metadata:
      labels:
        app: lightrag
    spec:
      containers:
      - name: lightrag
        image: lightrag:latest
        ports:
        - containerPort: 9621
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: lightrag-config
              key: database_url
        volumeMounts:
        - name: shared-data
          mountPath: /app/data
      volumes:
      - name: shared-data
        persistentVolumeClaim:
          claimName: lightrag-data-pvc
```

### Load Balancing

For high-volume deployments, implement a load balancer:

#### Nginx Load Balancer

```nginx
# /etc/nginx/conf.d/lightrag.conf
upstream lightrag_servers {
    server lightrag1:9621;
    server lightrag2:9621;
    server lightrag3:9621;
    server lightrag4:9621;
    server lightrag5:9621;
}

server {
    listen 80;
    server_name api.lightrag.example.com;

    location / {
        proxy_pass http://lightrag_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Kubernetes Service

```yaml
# kubernetes/lightrag-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: lightrag-service
spec:
  selector:
    app: lightrag
  ports:
  - port: 80
    targetPort: 9621
  type: LoadBalancer
```

## Database Scaling

### Neo4j Scaling

For high-volume graph workloads, use Neo4j's clustering capabilities:

```yaml
# docker-compose-neo4j-cluster.yml
services:
  neo4j-core:
    image: neo4j:5.26.4-enterprise
    environment:
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_dbms_mode=CORE
      - NEO4J_causal__clustering_minimum__core__cluster__size__at__formation=3
      - NEO4J_causal__clustering_minimum__core__cluster__size__at__runtime=3
      - NEO4J_causal__clustering_initial__discovery__members=neo4j-core1:5000,neo4j-core2:5000,neo4j-core3:5000
      - NEO4J_AUTH=neo4j/password
    ports:
      - "7474:7474"
      - "7687:7687"

  neo4j-core2:
    image: neo4j:5.26.4-enterprise
    environment:
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_dbms_mode=CORE
      - NEO4J_causal__clustering_minimum__core__cluster__size__at__formation=3
      - NEO4J_causal__clustering_minimum__core__cluster__size__at__runtime=3
      - NEO4J_causal__clustering_initial__discovery__members=neo4j-core1:5000,neo4j-core2:5000,neo4j-core3:5000
      - NEO4J_AUTH=neo4j/password

  neo4j-core3:
    image: neo4j:5.26.4-enterprise
    environment:
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_dbms_mode=CORE
      - NEO4J_causal__clustering_minimum__core__cluster__size__at__formation=3
      - NEO4J_causal__clustering_minimum__core__cluster__size__at__runtime=3
      - NEO4J_causal__clustering_initial__discovery__members=neo4j-core1:5000,neo4j-core2:5000,neo4j-core3:5000
      - NEO4J_AUTH=neo4j/password
```

Configure LightRAG to use the Neo4j cluster:

```ini
# config.ini
KG_BINDING=neo4j
NEO4J_URI=neo4j://neo4j-load-balancer:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

### Vector Database Scaling

Depending on your vector database choice, scaling options vary:

#### Milvus Cluster

```yaml
# docker-compose-milvus-cluster.yml
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  pulsar:
    container_name: milvus-pulsar
    image: apachepulsar/pulsar:2.8.2
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/pulsar:/pulsar/data
    command: bin/pulsar standalone -nss

  querynode:
    container_name: milvus-querynode
    image: milvusdb/milvus:v2.2.3
    command: ["milvus", "run", "querynode"]
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
      - PULSAR_ADDRESS=pulsar:6650
    depends_on:
      - "etcd"
      - "minio"
      - "pulsar"

  querycoord:
    container_name: milvus-querycoord
    image: milvusdb/milvus:v2.2.3
    command: ["milvus", "run", "querycoord"]
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
      - PULSAR_ADDRESS=pulsar:6650
    depends_on:
      - "etcd"
      - "minio"
      - "pulsar"

  proxy:
    container_name: milvus-proxy
    image: milvusdb/milvus:v2.2.3
    command: ["milvus", "run", "proxy"]
    environment:
      - ETCD_ENDPOINTS=etcd:2379
      - MINIO_ADDRESS=minio:9000
      - PULSAR_ADDRESS=pulsar:6650
    ports:
      - "19530:19530"
    depends_on:
      - "querycoord"
```

Configure LightRAG to use the Milvus cluster:

```ini
# config.ini
VECTOR_DB_BINDING=milvus
MILVUS_HOST=milvus-proxy
MILVUS_PORT=19530
```

## Resource Optimization

### Memory Optimization

Tune memory usage based on workload:

```ini
# config.ini
MAX_EMBED_CHUNK_SIZE=512  # Reduce for lower memory usage
MAX_EMBED_TOKENS=8000     # Limit tokens for embedding
MAX_ASYNC=20              # Adjust concurrent operations
```

### CPU Optimization

For CPU-bound workloads (document processing, embedding generation):

```ini
# config.ini
PROCESSING_THREADS=8      # Match to available CPU cores
CHUNK_OVERLAP=20          # Reduce overlap for faster processing
```

## Cloud Deployment Scaling

### AWS Deployment

For AWS deployments, consider:
- ECS Fargate for stateless API containers
- Neptune for graph database
- OpenSearch for vector storage
- S3 for document storage
- SQS for processing queue

```terraform
# Example Terraform for AWS ECS
resource "aws_ecs_cluster" "lightrag_cluster" {
  name = "lightrag-cluster"
}

resource "aws_ecs_service" "lightrag_service" {
  name            = "lightrag"
  cluster         = aws_ecs_cluster.lightrag_cluster.id
  task_definition = aws_ecs_task_definition.lightrag_task.arn
  desired_count   = 5
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = var.private_subnets
    security_groups = [aws_security_group.lightrag_sg.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.lightrag_tg.arn
    container_name   = "lightrag"
    container_port   = 9621
  }
}
```

### Azure Deployment

For Azure deployments, consider:
- AKS for container orchestration
- Cosmos DB for graph data
- Azure Cognitive Search for vector search
- Blob Storage for documents

### GCP Deployment

For GCP deployments, consider:
- GKE for container orchestration
- Memorystore for caching
- Cloud Storage for documents
- Cloud Run for stateless API services

## Caching Strategies

Implement caching to improve performance:

### Result Caching

```python
# Configure in config.ini
ENABLE_QUERY_CACHE=true
QUERY_CACHE_TTL=3600  # Seconds

# Code example (already implemented in LightRAG)
def get_cached_result(query_hash):
    if cache_enabled and query_hash in query_cache:
        if time.time() - query_cache[query_hash]['timestamp'] < cache_ttl:
            return query_cache[query_hash]['result']
    return None
```

### Embedding Caching

```python
# Configure in config.ini
ENABLE_EMBEDDING_CACHE=true
EMBEDDING_CACHE_SIZE=10000  # Number of embeddings to cache

# Code example (already implemented in LightRAG)
def get_cached_embedding(text_hash):
    if embedding_cache_enabled and text_hash in embedding_cache:
        return embedding_cache[text_hash]
    return None
```

## Monitoring and Observability

For high-volume deployments, implement comprehensive monitoring:

### Prometheus Metrics

```python
# Example metrics (implemented in LightRAG monitoring module)
QUERY_LATENCY = Histogram('lightrag_query_latency_seconds', 'Query latency in seconds')
DOCUMENT_PROCESSING_TIME = Histogram('lightrag_document_processing_seconds', 'Document processing time')
ACTIVE_REQUESTS = Gauge('lightrag_active_requests', 'Number of active requests')
FAILED_REQUESTS = Counter('lightrag_failed_requests_total', 'Number of failed requests')
```

### Grafana Dashboard

Create a Grafana dashboard with:
- Query throughput and latency
- Document processing rate
- Error rates
- Memory and CPU usage
- Database performance

## High Availability Configuration

For mission-critical deployments:

1. Multi-region deployment
2. Automated failover
3. Regular backup and restore testing
4. Load testing under failure conditions

```yaml
# Example Docker Compose for HA setup
services:
  lightrag-primary:
    image: lightrag:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: any
      update_config:
        order: start-first
        failure_action: rollback
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9621/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  lightrag-secondary:
    image: lightrag:latest
    deploy:
      replicas: 3
      restart_policy:
        condition: any
      update_config:
        order: start-first
        failure_action: rollback
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9621/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Conclusion

LightRAG can be scaled to handle high-volume production workloads by:

1. Horizontally scaling the API service
2. Using clustered database solutions
3. Implementing effective caching strategies
4. Optimizing resource usage
5. Deploying to cloud environments with managed services
6. Implementing robust monitoring and observability

For specific scaling questions or optimizations, consult the LightRAG benchmark reports or contact the development team.