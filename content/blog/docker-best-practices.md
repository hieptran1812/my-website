---
title: "Docker Best Practices: From Development to Production"
excerpt: "Essential Docker practices, optimization techniques, and deployment strategies for scalable applications."
date: "2024-03-20"
readTime: "7 min read"
tags: ["Docker", "DevOps", "Containerization"]
category: "notes"
featured: true
---

# Docker Best Practices: From Development to Production

Docker has revolutionized how we build, ship, and run applications. This guide covers essential practices for using Docker effectively from development through production deployment.

## Image Optimization

### Multi-stage Builds

Use multi-stage builds to reduce image size:

```dockerfile
# Build stage
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Production stage
FROM node:16-alpine AS production
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

### Layer Optimization

```dockerfile
# Bad - creates unnecessary layers
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y vim
RUN rm -rf /var/lib/apt/lists/*

# Good - minimizes layers
RUN apt-get update && \
    apt-get install -y curl vim && \
    rm -rf /var/lib/apt/lists/*
```

### Base Image Selection

```dockerfile
# Production: Use minimal base images
FROM node:16-alpine  # 50MB vs 900MB for full node image

# Development: Use full images for debugging
FROM node:16  # Includes build tools and utilities
```

## Security Best Practices

### Non-root User

```dockerfile
# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Change ownership
COPY --chown=nextjs:nodejs . .

# Switch to non-root user
USER nextjs
```

### Secret Management

```bash
# Use Docker secrets (Docker Swarm)
echo "mypassword" | docker secret create db_password -

# Use build-time secrets (BuildKit)
# syntax=docker/dockerfile:1
FROM alpine
RUN --mount=type=secret,id=mypassword \
    cat /run/secrets/mypassword
```

### Image Scanning

```bash
# Scan for vulnerabilities
docker scan myapp:latest

# Use Trivy for scanning
trivy image myapp:latest
```

## Development Workflow

### Docker Compose for Local Development

```yaml
version: "3.8"
services:
  app:
    build:
      context: .
      target: development
    volumes:
      - .:/app
      - /app/node_modules
    environment:
      - NODE_ENV=development
    ports:
      - "3000:3000"
    depends_on:
      - database
      - redis

  database:
    image: postgres:13
    environment:
      - POSTGRES_DB=myapp
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Hot Reloading

```dockerfile
# Development stage with hot reloading
FROM node:16-alpine AS development
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
CMD ["npm", "run", "dev"]
```

## Production Optimization

### Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1
```

### Resource Limits

```yaml
services:
  app:
    image: myapp:latest
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M
        reservations:
          cpus: "0.25"
          memory: 256M
```

### Environment Configuration

```dockerfile
# Use build args for build-time config
ARG NODE_ENV=production
ENV NODE_ENV=$NODE_ENV

# Use environment variables for runtime config
ENV PORT=3000
ENV DB_HOST=localhost
```

## Networking and Service Discovery

### Custom Networks

```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

services:
  web:
    networks:
      - frontend
  api:
    networks:
      - frontend
      - backend
  database:
    networks:
      - backend
```

### Service Discovery

```bash
# Services can communicate using service names
curl http://api-service:3001/users
curl http://database:5432
```

## Data Management

### Named Volumes

```yaml
volumes:
  database_data:
    driver: local
  app_logs:
    driver: local

services:
  app:
    volumes:
      - app_logs:/app/logs
  database:
    volumes:
      - database_data:/var/lib/postgresql/data
```

### Bind Mounts for Development

```yaml
services:
  app:
    volumes:
      - ./src:/app/src # Source code
      - /app/node_modules # Preserve node_modules
```

## Logging and Monitoring

### Structured Logging

```javascript
// Use structured logging
const winston = require("winston");

const logger = winston.createLogger({
  format: winston.format.json(),
  transports: [new winston.transports.Console()],
});

logger.info("Application started", {
  port: 3000,
  environment: process.env.NODE_ENV,
});
```

### Log Drivers

```yaml
services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Docker Build and Deploy

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .

      - name: Run tests
        run: docker run --rm myapp:${{ github.sha }} npm test

      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push myapp:${{ github.sha }}
```

### Multi-environment Deployment

```bash
# Build for different environments
docker build --target development -t myapp:dev .
docker build --target production -t myapp:prod .

# Environment-specific configs
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## Performance Optimization

### Image Caching

```dockerfile
# Copy package files first for better caching
COPY package*.json ./
RUN npm ci --only=production

# Copy source code last
COPY . .
```

### Build Context Optimization

```dockerignore
node_modules
npm-debug.log
.git
.gitignore
README.md
.env
.nyc_output
coverage
.nyc_output
```

### Resource Monitoring

```bash
# Monitor container resources
docker stats

# Memory usage
docker exec <container> cat /proc/meminfo

# Disk usage
docker system df
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

## Troubleshooting

### Debug Containers

```bash
# Run debugging session
docker run -it --entrypoint sh myapp:latest

# Execute commands in running container
docker exec -it <container> sh

# View logs
docker logs -f <container>
docker-compose logs -f service-name
```

### Common Issues

```bash
# Permission issues
chown -R $(id -u):$(id -g) ./data

# Network connectivity
docker network ls
docker network inspect bridge

# Port conflicts
netstat -tulpn | grep :3000
```

## Production Deployment

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml myapp

# Scale services
docker service scale myapp_web=3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
        - name: myapp
          image: myapp:latest
          ports:
            - containerPort: 3000
          resources:
            limits:
              memory: "512Mi"
              cpu: "500m"
```

## Monitoring and Alerting

### Prometheus Metrics

```javascript
const promClient = require("prom-client");

// Custom metrics
const httpRequestsTotal = new promClient.Counter({
  name: "http_requests_total",
  help: "Total HTTP requests",
  labelNames: ["method", "status"],
});

// Expose metrics endpoint
app.get("/metrics", (req, res) => {
  res.set("Content-Type", promClient.register.contentType);
  res.end(promClient.register.metrics());
});
```

### Health Check Endpoint

```javascript
app.get("/health", (req, res) => {
  const healthcheck = {
    uptime: process.uptime(),
    message: "OK",
    timestamp: Date.now(),
    checks: {
      database: checkDatabase(),
      redis: checkRedis(),
    },
  };

  res.status(200).json(healthcheck);
});
```

## Conclusion

Following these Docker best practices ensures:

1. **Security**: Non-root users, secret management, vulnerability scanning
2. **Performance**: Optimized images, efficient caching, resource limits
3. **Maintainability**: Clear structure, proper logging, health checks
4. **Scalability**: Stateless containers, service discovery, load balancing
5. **Reliability**: Health checks, graceful shutdowns, proper error handling

Docker's power lies in its simplicity and consistency across environments. These practices will help you build robust, production-ready containerized applications.
