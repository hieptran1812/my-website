---
title: "Building Scalable Microservices with Node.js and Docker"
excerpt: "A comprehensive guide to designing, implementing, and deploying microservices architecture using modern tools and best practices."
date: "2024-03-22"
readTime: "15 min read"
tags: ["Microservices", "Node.js", "Docker", "Architecture"]
difficulty: "Advanced"
featured: true
category: "software-development"
---

# Building Scalable Microservices with Node.js and Docker

Microservices architecture has become the gold standard for building scalable, maintainable applications. This comprehensive guide will walk you through designing, implementing, and deploying a microservices system using Node.js and Docker.

## Architecture Overview

Our microservices system will consist of:

- **User Service**: Authentication and user management
- **Product Service**: Product catalog and inventory
- **Order Service**: Order processing and management
- **API Gateway**: Request routing and load balancing
- **Message Queue**: Asynchronous communication

## Project Structure

```
microservices-app/
├── api-gateway/
├── user-service/
├── product-service/
├── order-service/
├── shared/
├── docker-compose.yml
└── README.md
```

## API Gateway Implementation

Let's start with the API Gateway using Express.js:

```javascript
// api-gateway/src/app.js
const express = require("express");
const httpProxy = require("http-proxy-middleware");
const rateLimit = require("express-rate-limit");
const helmet = require("helmet");
const cors = require("cors");

const app = express();

// Security middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
});
app.use(limiter);

// Service discovery
const services = {
  user: process.env.USER_SERVICE_URL || "http://user-service:3001",
  product: process.env.PRODUCT_SERVICE_URL || "http://product-service:3002",
  order: process.env.ORDER_SERVICE_URL || "http://order-service:3003",
};

// Proxy middleware
const createProxyMiddleware = (target) => {
  return httpProxy({
    target,
    changeOrigin: true,
    timeout: 5000,
    onError: (err, req, res) => {
      console.error(`Proxy error: ${err.message}`);
      res.status(503).json({ error: "Service unavailable" });
    },
  });
};

// Route proxying
app.use("/api/users", createProxyMiddleware(services.user));
app.use("/api/products", createProxyMiddleware(services.product));
app.use("/api/orders", createProxyMiddleware(services.order));

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`API Gateway running on port ${PORT}`);
});

module.exports = app;
```

## User Service

```javascript
// user-service/src/app.js
const express = require("express");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const { Pool } = require("pg");
const redis = require("redis");

const app = express();
app.use(express.json());

// Database connection
const pool = new Pool({
  host: process.env.DB_HOST || "postgres",
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || "userdb",
  user: process.env.DB_USER || "postgres",
  password: process.env.DB_PASSWORD || "password",
});

// Redis connection for caching
const redisClient = redis.createClient({
  host: process.env.REDIS_HOST || "redis",
  port: process.env.REDIS_PORT || 6379,
});

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res.status(401).json({ error: "Access token required" });
  }

  jwt.verify(token, process.env.JWT_SECRET || "secret", (err, user) => {
    if (err) return res.status(403).json({ error: "Invalid token" });
    req.user = user;
    next();
  });
};

// User registration
app.post("/register", async (req, res) => {
  try {
    const { username, email, password } = req.body;

    // Validate input
    if (!username || !email || !password) {
      return res.status(400).json({ error: "All fields are required" });
    }

    // Hash password
    const saltRounds = 10;
    const hashedPassword = await bcrypt.hash(password, saltRounds);

    // Insert user
    const result = await pool.query(
      "INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3) RETURNING id, username, email",
      [username, email, hashedPassword]
    );

    const user = result.rows[0];

    // Generate JWT
    const token = jwt.sign(
      { userId: user.id, username: user.username },
      process.env.JWT_SECRET || "secret",
      { expiresIn: "24h" }
    );

    res.status(201).json({ user, token });
  } catch (error) {
    console.error("Registration error:", error);
    if (error.code === "23505") {
      // Unique constraint violation
      res.status(409).json({ error: "Username or email already exists" });
    } else {
      res.status(500).json({ error: "Internal server error" });
    }
  }
});

// User login
app.post("/login", async (req, res) => {
  try {
    const { username, password } = req.body;

    // Find user
    const result = await pool.query(
      "SELECT id, username, email, password_hash FROM users WHERE username = $1",
      [username]
    );

    if (result.rows.length === 0) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    const user = result.rows[0];

    // Verify password
    const isValidPassword = await bcrypt.compare(password, user.password_hash);
    if (!isValidPassword) {
      return res.status(401).json({ error: "Invalid credentials" });
    }

    // Generate JWT
    const token = jwt.sign(
      { userId: user.id, username: user.username },
      process.env.JWT_SECRET || "secret",
      { expiresIn: "24h" }
    );

    // Cache user session
    await redisClient.setex(`session:${user.id}`, 86400, JSON.stringify(user));

    res.json({
      user: { id: user.id, username: user.username, email: user.email },
      token,
    });
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Get user profile
app.get("/profile", authenticateToken, async (req, res) => {
  try {
    // Try cache first
    const cachedUser = await redisClient.get(`session:${req.user.userId}`);
    if (cachedUser) {
      return res.json({ user: JSON.parse(cachedUser) });
    }

    // Fallback to database
    const result = await pool.query(
      "SELECT id, username, email, created_at FROM users WHERE id = $1",
      [req.user.userId]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: "User not found" });
    }

    const user = result.rows[0];

    // Update cache
    await redisClient.setex(`session:${user.id}`, 86400, JSON.stringify(user));

    res.json({ user });
  } catch (error) {
    console.error("Profile error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "healthy", service: "user-service" });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`User service running on port ${PORT}`);
});
```

## Product Service

```javascript
// product-service/src/app.js
const express = require("express");
const { Pool } = require("pg");
const amqp = require("amqplib");

const app = express();
app.use(express.json());

// Database connection
const pool = new Pool({
  host: process.env.DB_HOST || "postgres",
  port: process.env.DB_PORT || 5432,
  database: process.env.DB_NAME || "productdb",
  user: process.env.DB_USER || "postgres",
  password: process.env.DB_PASSWORD || "password",
});

// Message queue connection
let channel;
const connectRabbitMQ = async () => {
  try {
    const connection = await amqp.connect(
      process.env.RABBITMQ_URL || "amqp://rabbitmq"
    );
    channel = await connection.createChannel();

    // Declare queues
    await channel.assertQueue("inventory.update", { durable: true });
    await channel.assertQueue("product.events", { durable: true });

    console.log("Connected to RabbitMQ");
  } catch (error) {
    console.error("RabbitMQ connection error:", error);
    setTimeout(connectRabbitMQ, 5000); // Retry after 5 seconds
  }
};

connectRabbitMQ();

// Get all products
app.get("/products", async (req, res) => {
  try {
    const { page = 1, limit = 10, category } = req.query;
    const offset = (page - 1) * limit;

    let query = "SELECT * FROM products WHERE 1=1";
    const params = [];

    if (category) {
      query += " AND category = $" + (params.length + 1);
      params.push(category);
    }

    query +=
      " ORDER BY created_at DESC LIMIT $" +
      (params.length + 1) +
      " OFFSET $" +
      (params.length + 2);
    params.push(limit, offset);

    const result = await pool.query(query, params);

    // Get total count
    const countResult = await pool.query("SELECT COUNT(*) FROM products");
    const total = parseInt(countResult.rows[0].count);

    res.json({
      products: result.rows,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / limit),
      },
    });
  } catch (error) {
    console.error("Get products error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Get product by ID
app.get("/products/:id", async (req, res) => {
  try {
    const { id } = req.params;

    const result = await pool.query("SELECT * FROM products WHERE id = $1", [
      id,
    ]);

    if (result.rows.length === 0) {
      return res.status(404).json({ error: "Product not found" });
    }

    res.json({ product: result.rows[0] });
  } catch (error) {
    console.error("Get product error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Create product
app.post("/products", async (req, res) => {
  try {
    const { name, description, price, category, stock_quantity } = req.body;

    const result = await pool.query(
      "INSERT INTO products (name, description, price, category, stock_quantity) VALUES ($1, $2, $3, $4, $5) RETURNING *",
      [name, description, price, category, stock_quantity]
    );

    const product = result.rows[0];

    // Publish event
    if (channel) {
      await channel.sendToQueue(
        "product.events",
        Buffer.from(
          JSON.stringify({
            event: "product.created",
            product,
          })
        )
      );
    }

    res.status(201).json({ product });
  } catch (error) {
    console.error("Create product error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

// Update inventory
app.patch("/products/:id/inventory", async (req, res) => {
  try {
    const { id } = req.params;
    const { quantity_change } = req.body;

    const result = await pool.query(
      "UPDATE products SET stock_quantity = stock_quantity + $1 WHERE id = $2 RETURNING *",
      [quantity_change, id]
    );

    if (result.rows.length === 0) {
      return res.status(404).json({ error: "Product not found" });
    }

    const product = result.rows[0];

    // Publish inventory update event
    if (channel) {
      await channel.sendToQueue(
        "inventory.update",
        Buffer.from(
          JSON.stringify({
            productId: id,
            newQuantity: product.stock_quantity,
            change: quantity_change,
          })
        )
      );
    }

    res.json({ product });
  } catch (error) {
    console.error("Update inventory error:", error);
    res.status(500).json({ error: "Internal server error" });
  }
});

app.listen(3002, () => {
  console.log("Product service running on port 3002");
});
```

## Docker Configuration

```yaml
# docker-compose.yml
version: "3.8"

services:
  api-gateway:
    build: ./api-gateway
    ports:
      - "3000:3000"
    environment:
      - USER_SERVICE_URL=http://user-service:3001
      - PRODUCT_SERVICE_URL=http://product-service:3002
      - ORDER_SERVICE_URL=http://order-service:3003
    depends_on:
      - user-service
      - product-service
      - order-service

  user-service:
    build: ./user-service
    environment:
      - DB_HOST=postgres
      - DB_NAME=userdb
      - DB_USER=postgres
      - DB_PASSWORD=password
      - REDIS_HOST=redis
      - JWT_SECRET=your-secret-key
    depends_on:
      - postgres
      - redis

  product-service:
    build: ./product-service
    environment:
      - DB_HOST=postgres
      - DB_NAME=productdb
      - DB_USER=postgres
      - DB_PASSWORD=password
      - RABBITMQ_URL=amqp://rabbitmq
    depends_on:
      - postgres
      - rabbitmq

  order-service:
    build: ./order-service
    environment:
      - DB_HOST=postgres
      - DB_NAME=orderdb
      - DB_USER=postgres
      - DB_PASSWORD=password
      - RABBITMQ_URL=amqp://rabbitmq
    depends_on:
      - postgres
      - rabbitmq

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

  rabbitmq:
    image: rabbitmq:3-management
    environment:
      - RABBITMQ_DEFAULT_USER=admin
      - RABBITMQ_DEFAULT_PASS=password
    ports:
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq

volumes:
  postgres_data:
  redis_data:
  rabbitmq_data:
```

## Deployment and Scaling

```bash
# Build and start services
docker-compose up --build

# Scale specific services
docker-compose up --scale product-service=3 --scale user-service=2

# View logs
docker-compose logs -f api-gateway

# Health check script
#!/bin/bash
services=("api-gateway:3000" "user-service:3001" "product-service:3002")

for service in "${services[@]}"; do
  if curl -f "http://$service/health"; then
    echo "$service is healthy"
  else
    echo "$service is unhealthy"
  fi
done
```

## Monitoring and Observability

```javascript
// shared/middleware/monitoring.js
const promClient = require("prom-client");

// Metrics
const httpRequestsTotal = new promClient.Counter({
  name: "http_requests_total",
  help: "Total number of HTTP requests",
  labelNames: ["method", "route", "status"],
});

const httpRequestDuration = new promClient.Histogram({
  name: "http_request_duration_seconds",
  help: "Duration of HTTP requests in seconds",
  labelNames: ["method", "route"],
});

const monitoringMiddleware = (req, res, next) => {
  const start = Date.now();

  res.on("finish", () => {
    const duration = (Date.now() - start) / 1000;

    httpRequestsTotal
      .labels(req.method, req.route?.path || req.path, res.statusCode)
      .inc();

    httpRequestDuration
      .labels(req.method, req.route?.path || req.path)
      .observe(duration);
  });

  next();
};

module.exports = { monitoringMiddleware, promClient };
```

## Best Practices

1. **Service Independence**: Each service should be independently deployable
2. **Data Isolation**: Never share databases between services
3. **API Versioning**: Use versioned APIs for backward compatibility
4. **Circuit Breakers**: Implement circuit breakers for resilience
5. **Distributed Tracing**: Use tools like Jaeger for request tracing
6. **Centralized Logging**: Aggregate logs using ELK stack
7. **Health Checks**: Implement comprehensive health checks
8. **Security**: Use JWT tokens and secure service-to-service communication

## Conclusion

This microservices architecture provides a solid foundation for building scalable applications. The combination of Node.js and Docker offers excellent developer experience and deployment flexibility. Key benefits include:

- **Scalability**: Individual services can be scaled independently
- **Maintainability**: Smaller, focused codebases are easier to maintain
- **Technology Diversity**: Different services can use different technologies
- **Fault Isolation**: Failures in one service don't bring down the entire system

Remember to monitor your services closely and implement proper error handling and retry mechanisms for production deployments.
