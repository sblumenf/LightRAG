# LightRAG Security Guide

This document provides comprehensive security guidance for deploying and operating LightRAG in production environments.

## Authentication and Authorization

### API Key Authentication

LightRAG implements API key authentication for all endpoints. This key must be configured and provided with every request:

```ini
# In config.ini
LIGHTRAG_API_KEY=your-strong-api-key-here
```

Best practices for API keys:
- Use a minimum of 32 characters
- Include a mix of alphanumeric and special characters
- Rotate keys regularly
- Use different keys for different environments

### Authentication Implementation

Authentication is handled in the `auth.py` module:
- All API endpoints are protected by the authentication middleware
- Keys are validated using constant-time comparison to prevent timing attacks
- Failed authentication attempts are logged with client IP addresses

## Transport Security

### TLS/SSL Configuration

Always use HTTPS in production. LightRAG doesn't manage TLS directly - you should configure it at the infrastructure level:

1. Using a reverse proxy (recommended):
   ```nginx
   # Example Nginx configuration
   server {
       listen 443 ssl;
       server_name your-domain.com;
       
       ssl_certificate /path/to/cert.pem;
       ssl_certificate_key /path/to/key.pem;
       ssl_protocols TLSv1.2 TLSv1.3;
       
       location / {
           proxy_pass http://localhost:9621;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

2. Using Docker with Traefik:
   ```yaml
   # docker-compose additions
   services:
     traefik:
       image: traefik:v2.9
       command:
         - "--providers.docker=true"
         - "--entrypoints.websecure.address=:443"
         - "--certificatesresolvers.myresolver.acme.tlschallenge=true"
         - "--certificatesresolvers.myresolver.acme.email=youremail@example.com"
         - "--certificatesresolvers.myresolver.acme.storage=/letsencrypt/acme.json"
       ports:
         - "443:443"
       volumes:
         - /var/run/docker.sock:/var/run/docker.sock:ro
         - ./letsencrypt:/letsencrypt
     
     lightrag:
       # existing configuration...
       labels:
         - "traefik.enable=true"
         - "traefik.http.routers.lightrag.rule=Host(`your-domain.com`)"
         - "traefik.http.routers.lightrag.entrypoints=websecure"
         - "traefik.http.routers.lightrag.tls.certresolver=myresolver"
   ```

## Data Security

### Storage Security

LightRAG stores data in several locations:

1. Document Storage:
   - Raw documents in `/app/data/inputs`
   - Extracted content in `/app/data/rag_storage`
   
2. Vector Database:
   - Embeddings in selected vector DB implementation
   
3. Knowledge Graph:
   - Entity and relationship data in Neo4j or other graph storage

Security recommendations:
- Encrypt volumes at rest using filesystem encryption
- Implement proper backup procedures with encryption
- Apply proper file permissions (read/write only for service user)

### Encryption at Rest

For cloud deployments, use platform-provided volume encryption:
- AWS: EBS encryption
- Azure: Azure Disk Encryption
- GCP: Persistent Disk encryption

For local deployments:
- Linux: Use LUKS encryption
- Docker volumes: Consider using encrypted filesystems

### Credential Management

LightRAG requires several credentials:
- LLM API keys
- Database credentials
- API authentication keys

Best practices:
- Never store credentials in code or Docker images
- Use environment variables or secure secret management
- For Kubernetes: use Secret resources
- For cloud: use cloud secret management services (AWS Secrets Manager, etc.)

## Network Security

### Firewall Configuration

Restrict access to LightRAG and related services:

```bash
# Example iptables rules
iptables -A INPUT -p tcp --dport 9621 -s trusted_ip_range -j ACCEPT
iptables -A INPUT -p tcp --dport 9621 -j DROP
```

Docker network security:
- Use internal Docker networks for inter-service communication
- Only expose necessary ports
- Use `--network=host` with caution

### Rate Limiting

LightRAG includes built-in rate limiting to prevent abuse:
- Configure in `config.ini`:
  ```ini
  RATE_LIMIT_REQUESTS=60
  RATE_LIMIT_WINDOW=60
  ```
- Implement additional rate limiting at the infrastructure level for production

## Dependency Security

### Regular Updates

Keep dependencies updated to mitigate vulnerabilities:
```bash
# Update dependencies
pip install -U -r requirements.txt

# Check for vulnerable packages
pip-audit
```

### Supply Chain Security

- Use dependency pinning in requirements.txt
- Consider using a private PyPI mirror
- Implement container scanning in CI/CD pipeline

## Operational Security

### Logging and Monitoring

LightRAG logs to both console and file by default:
- Configure log levels in `config.ini`:
  ```ini
  LOG_LEVEL=INFO
  LOG_FILE=/var/log/lightrag/app.log
  ```

Important events to monitor:
- Authentication failures
- API rate limit violations
- Server errors
- Unusual query patterns

### Backup and Recovery

Implement regular backups:
```bash
# Example backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf "/backups/lightrag_data_$DATE.tar.gz" /app/data
tar -czf "/backups/lightrag_config_$DATE.tar.gz" /app/config.ini /app/.env
```

For Neo4j, use the built-in backup mechanism:
```bash
neo4j-admin dump --database=neo4j --to=/backups/neo4j_$DATE.dump
```

### Scaling and High Availability

For production deployment with high availability:

1. Use container orchestration (Kubernetes recommended):
   - Deploy multiple LightRAG instances
   - Use a load balancer (e.g., Kubernetes Service)
   - Configure health checks and auto-scaling

2. Database high availability:
   - Use Neo4j clustering for graph data
   - Configure vector database clustering (implementation-specific)

Example Kubernetes deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightrag
spec:
  replicas: 3
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
        image: yourrepo/lightrag:latest
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "500m"
            memory: "1Gi"
        ports:
        - containerPort: 9621
        livenessProbe:
          httpGet:
            path: /health
            port: 9621
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 9621
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /app/config.ini
          subPath: config.ini
        - name: data
          mountPath: /app/data
      volumes:
      - name: config
        configMap:
          name: lightrag-config
      - name: data
        persistentVolumeClaim:
          claimName: lightrag-data
```

## Compliance Considerations

Depending on your use case, consider:
- GDPR: Implement data retention policies and deletion capabilities
- HIPAA: Additional encryption and audit logging requirements
- SOC2: Comprehensive security controls and monitoring

## Security Checklist

Use this checklist for production deployment:

- [ ] Strong API key configured
- [ ] TLS/SSL properly implemented
- [ ] Firewall rules configured
- [ ] Rate limiting enabled
- [ ] Logging configured and monitored
- [ ] Regular backups scheduled
- [ ] Dependencies updated and scanned
- [ ] Volumes encrypted at rest
- [ ] Secure credential management implemented
- [ ] Security updates automated