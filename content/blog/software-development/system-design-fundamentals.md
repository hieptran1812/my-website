---
title: "System Design Fundamentals: Building Scalable Applications"
date: "2024-05-25"
readTime: "18 min read"
category: "Software Development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
tags:
  [
    "System Design",
    "Scalability",
    "Architecture",
    "Load Balancing",
    "Caching",
    "Microservices",
  ]
image: "/blog-placeholder.jpg"
excerpt: "Master the fundamentals of system design with practical examples, architectural patterns, and scalability techniques for building robust distributed systems."
---

# System Design Fundamentals: Building Scalable Applications

System design is the process of defining the architecture, modules, interfaces, and data for a system to satisfy specified requirements. This comprehensive guide covers the essential concepts and patterns needed to design scalable, reliable, and maintainable systems.

## Core System Design Principles

### Scalability

**Horizontal vs Vertical Scaling**

**Vertical Scaling (Scale Up):**

- Add more power to existing machines
- Simpler to implement initially
- Limited by hardware constraints
- Single point of failure

**Horizontal Scaling (Scale Out):**

- Add more machines to the resource pool
- Better fault tolerance
- More complex but unlimited scaling potential
- Distributed system challenges

### Reliability and Availability

**Reliability Patterns:**

- Redundancy and replication
- Failover mechanisms
- Circuit breakers
- Graceful degradation

**Availability Calculations:**

- 99.9% = 8.77 hours downtime/year
- 99.99% = 52.6 minutes downtime/year
- 99.999% = 5.26 minutes downtime/year

### Consistency

**CAP Theorem:**

- **Consistency**: All nodes see the same data simultaneously
- **Availability**: System remains operational
- **Partition Tolerance**: System continues despite network failures

**Trade-offs:**

- CP systems: Consistent and partition-tolerant (sacrifice availability)
- AP systems: Available and partition-tolerant (sacrifice consistency)
- CA systems: Consistent and available (sacrifice partition tolerance)

## System Architecture Patterns

### Layered Architecture

**Presentation Layer:**

```typescript
// API Controllers
class UserController {
  async createUser(req: Request, res: Response) {
    try {
      const userData = req.body;
      const user = await this.userService.createUser(userData);
      res.status(201).json(user);
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  }
}
```

**Business Logic Layer:**

```typescript
// Service Layer
class UserService {
  constructor(private userRepository: UserRepository) {}

  async createUser(userData: CreateUserDto): Promise<User> {
    // Validation
    this.validateUserData(userData);

    // Business logic
    const hashedPassword = await this.hashPassword(userData.password);

    // Persistence
    return this.userRepository.save({
      ...userData,
      password: hashedPassword,
    });
  }
}
```

**Data Access Layer:**

```typescript
// Repository Pattern
interface UserRepository {
  save(user: User): Promise<User>;
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
}

class MongoUserRepository implements UserRepository {
  async save(user: User): Promise<User> {
    return await UserModel.create(user);
  }

  async findById(id: string): Promise<User | null> {
    return await UserModel.findById(id);
  }
}
```

### Microservices Architecture

**Service Decomposition Strategies:**

**By Business Capability:**

- User Management Service
- Order Processing Service
- Inventory Service
- Payment Service

**By Data:**

- Each service owns its data
- Database per service pattern
- Avoid shared databases

**Inter-service Communication:**

**Synchronous Communication:**

```typescript
// HTTP/REST API calls
class OrderService {
  async createOrder(orderData: CreateOrderDto) {
    // Call inventory service
    const inventory = await this.inventoryClient.checkAvailability(
      orderData.productId,
      orderData.quantity
    );

    if (!inventory.available) {
      throw new Error("Product not available");
    }

    // Process order
    const order = await this.orderRepository.save(orderData);

    // Call payment service
    await this.paymentClient.processPayment({
      orderId: order.id,
      amount: order.total,
    });

    return order;
  }
}
```

**Asynchronous Communication:**

```typescript
// Event-driven architecture
class OrderService {
  async createOrder(orderData: CreateOrderDto) {
    const order = await this.orderRepository.save(orderData);

    // Publish event
    await this.eventBus.publish("order.created", {
      orderId: order.id,
      userId: order.userId,
      products: order.items,
    });

    return order;
  }
}

// Event handlers in other services
class InventoryService {
  @EventHandler("order.created")
  async handleOrderCreated(event: OrderCreatedEvent) {
    await this.reserveInventory(event.products);
  }
}
```

## Load Balancing Strategies

### Load Balancer Types

**Layer 4 (Transport Layer):**

- Routes based on IP and port
- Faster, less CPU intensive
- No application-level awareness

**Layer 7 (Application Layer):**

- Routes based on content (HTTP headers, URLs)
- More intelligent routing
- SSL termination capabilities

### Load Balancing Algorithms

**Round Robin:**

```python
class RoundRobinBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0

    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server
```

**Weighted Round Robin:**

```python
class WeightedRoundRobinBalancer:
    def __init__(self, servers):
        # servers = [('server1', 3), ('server2', 1)]
        self.weighted_servers = []
        for server, weight in servers:
            self.weighted_servers.extend([server] * weight)
        self.current = 0

    def get_server(self):
        server = self.weighted_servers[self.current]
        self.current = (self.current + 1) % len(self.weighted_servers)
        return server
```

**Least Connections:**

```python
class LeastConnectionsBalancer:
    def __init__(self, servers):
        self.servers = {server: 0 for server in servers}

    def get_server(self):
        return min(self.servers, key=self.servers.get)

    def add_connection(self, server):
        self.servers[server] += 1

    def remove_connection(self, server):
        self.servers[server] -= 1
```

## Caching Strategies

### Cache Patterns

**Cache-Aside (Lazy Loading):**

```typescript
class ProductService {
  async getProduct(id: string): Promise<Product> {
    // Check cache first
    const cached = await this.cache.get(`product:${id}`);
    if (cached) {
      return JSON.parse(cached);
    }

    // Load from database
    const product = await this.productRepository.findById(id);
    if (product) {
      // Store in cache
      await this.cache.set(`product:${id}`, JSON.stringify(product), 3600);
    }

    return product;
  }
}
```

**Write-Through:**

```typescript
class ProductService {
  async updateProduct(id: string, updates: Partial<Product>): Promise<Product> {
    // Update database
    const product = await this.productRepository.update(id, updates);

    // Update cache
    await this.cache.set(`product:${id}`, JSON.stringify(product), 3600);

    return product;
  }
}
```

**Write-Behind (Write-Back):**

```typescript
class ProductService {
  async updateProduct(id: string, updates: Partial<Product>): Promise<Product> {
    // Update cache immediately
    const product = await this.cache.get(`product:${id}`);
    const updatedProduct = { ...JSON.parse(product), ...updates };
    await this.cache.set(`product:${id}`, JSON.stringify(updatedProduct));

    // Queue database update
    await this.writeQueue.enqueue({
      operation: "update",
      table: "products",
      id,
      data: updates,
    });

    return updatedProduct;
  }
}
```

### Cache Levels

**Browser Cache:**

```typescript
// HTTP headers for caching
app.get("/api/products/:id", (req, res) => {
  const product = getProduct(req.params.id);

  res.set({
    "Cache-Control": "public, max-age=3600", // 1 hour
    ETag: generateETag(product),
    "Last-Modified": product.updatedAt.toUTCString(),
  });

  res.json(product);
});
```

**CDN Cache:**

```typescript
// CDN cache invalidation
class CDNService {
  async invalidateCache(paths: string[]) {
    await this.cdnClient.createInvalidation({
      DistributionId: this.distributionId,
      InvalidationBatch: {
        Paths: { Quantity: paths.length, Items: paths },
        CallerReference: Date.now().toString(),
      },
    });
  }
}
```

**Application Cache:**

```typescript
// Redis implementation
class RedisCache {
  constructor(private redis: Redis) {}

  async get(key: string): Promise<string | null> {
    return await this.redis.get(key);
  }

  async set(key: string, value: string, ttl?: number): Promise<void> {
    if (ttl) {
      await this.redis.setex(key, ttl, value);
    } else {
      await this.redis.set(key, value);
    }
  }

  async invalidate(pattern: string): Promise<void> {
    const keys = await this.redis.keys(pattern);
    if (keys.length > 0) {
      await this.redis.del(...keys);
    }
  }
}
```

## Database Design Patterns

### ACID Properties

**Atomicity:** All operations in a transaction succeed or fail together
**Consistency:** Database remains in a valid state
**Isolation:** Concurrent transactions don't interfere
**Durability:** Committed transactions survive system failures

### Database Scaling Patterns

**Read Replicas:**

```typescript
class DatabaseService {
  constructor(private master: Database, private readReplicas: Database[]) {}

  async write(query: string, params: any[]): Promise<any> {
    return await this.master.query(query, params);
  }

  async read(query: string, params: any[]): Promise<any> {
    const replica = this.getRandomReplica();
    return await replica.query(query, params);
  }

  private getRandomReplica(): Database {
    const index = Math.floor(Math.random() * this.readReplicas.length);
    return this.readReplicas[index];
  }
}
```

**Sharding:**

```typescript
class ShardedDatabase {
  constructor(private shards: Map<string, Database>) {}

  private getShardKey(userId: string): string {
    // Hash-based sharding
    const hash = this.hash(userId);
    const shardCount = this.shards.size;
    const shardIndex = hash % shardCount;
    return `shard_${shardIndex}`;
  }

  async getUserData(userId: string): Promise<User> {
    const shardKey = this.getShardKey(userId);
    const shard = this.shards.get(shardKey);
    return await shard.query("SELECT * FROM users WHERE id = ?", [userId]);
  }
}
```

## API Design Patterns

### RESTful API Design

**Resource-based URLs:**

```typescript
// Good
GET    /api/users          // Get all users
GET    /api/users/:id      // Get specific user
POST   /api/users          // Create user
PUT    /api/users/:id      // Update user
DELETE /api/users/:id      // Delete user

// Nested resources
GET    /api/users/:id/posts     // Get user's posts
POST   /api/users/:id/posts     // Create post for user
```

**HTTP Status Codes:**

```typescript
class UserController {
  async createUser(req: Request, res: Response) {
    try {
      const user = await this.userService.createUser(req.body);
      res.status(201).json(user); // Created
    } catch (error) {
      if (error instanceof ValidationError) {
        res.status(400).json({ error: error.message }); // Bad Request
      } else if (error instanceof ConflictError) {
        res.status(409).json({ error: error.message }); // Conflict
      } else {
        res.status(500).json({ error: "Internal Server Error" });
      }
    }
  }
}
```

### GraphQL API Design

**Schema Definition:**

```typescript
const typeDefs = gql`
  type User {
    id: ID!
    name: String!
    email: String!
    posts: [Post!]!
  }

  type Post {
    id: ID!
    title: String!
    content: String!
    author: User!
    createdAt: DateTime!
  }

  type Query {
    user(id: ID!): User
    users(limit: Int, offset: Int): [User!]!
    post(id: ID!): Post
  }

  type Mutation {
    createUser(input: CreateUserInput!): User!
    updateUser(id: ID!, input: UpdateUserInput!): User!
    deleteUser(id: ID!): Boolean!
  }
`;
```

**Resolvers:**

```typescript
const resolvers = {
  Query: {
    user: async (_, { id }, { dataSources }) => {
      return dataSources.userAPI.getUser(id);
    },
    users: async (_, { limit, offset }, { dataSources }) => {
      return dataSources.userAPI.getUsers({ limit, offset });
    },
  },

  User: {
    posts: async (parent, _, { dataSources }) => {
      return dataSources.postAPI.getPostsByUserId(parent.id);
    },
  },

  Mutation: {
    createUser: async (_, { input }, { dataSources }) => {
      return dataSources.userAPI.createUser(input);
    },
  },
};
```

## Monitoring and Observability

### The Three Pillars

**Metrics:**

```typescript
// Prometheus metrics
import { Counter, Histogram, Gauge } from "prom-client";

const httpRequestsTotal = new Counter({
  name: "http_requests_total",
  help: "Total number of HTTP requests",
  labelNames: ["method", "route", "status"],
});

const httpRequestDuration = new Histogram({
  name: "http_request_duration_seconds",
  help: "Duration of HTTP requests in seconds",
  labelNames: ["method", "route"],
});

// Middleware to collect metrics
app.use((req, res, next) => {
  const start = Date.now();

  res.on("finish", () => {
    const duration = (Date.now() - start) / 1000;

    httpRequestsTotal
      .labels(
        req.method,
        req.route?.path || req.path,
        res.statusCode.toString()
      )
      .inc();
    httpRequestDuration
      .labels(req.method, req.route?.path || req.path)
      .observe(duration);
  });

  next();
});
```

**Logging:**

```typescript
import winston from "winston";

const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({ filename: "error.log", level: "error" }),
    new winston.transports.File({ filename: "combined.log" }),
    new winston.transports.Console({
      format: winston.format.simple(),
    }),
  ],
});

// Structured logging
logger.info("User created", {
  userId: user.id,
  email: user.email,
  timestamp: new Date().toISOString(),
  requestId: req.id,
});
```

**Tracing:**

```typescript
import { trace, SpanStatusCode } from "@opentelemetry/api";

const tracer = trace.getTracer("user-service");

async function createUser(userData: CreateUserDto): Promise<User> {
  const span = tracer.startSpan("createUser");

  try {
    span.setAttributes({
      "user.email": userData.email,
      operation: "create",
    });

    const user = await userRepository.save(userData);

    span.setStatus({ code: SpanStatusCode.OK });
    return user;
  } catch (error) {
    span.recordException(error);
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: error.message,
    });
    throw error;
  } finally {
    span.end();
  }
}
```

## Security Considerations

### Authentication and Authorization

**JWT Implementation:**

```typescript
import jwt from "jsonwebtoken";

class AuthService {
  generateToken(user: User): string {
    return jwt.sign(
      {
        userId: user.id,
        email: user.email,
        roles: user.roles,
      },
      process.env.JWT_SECRET!,
      { expiresIn: "24h" }
    );
  }

  verifyToken(token: string): any {
    return jwt.verify(token, process.env.JWT_SECRET!);
  }
}

// Middleware
const authenticateToken = (req: Request, res: Response, next: NextFunction) => {
  const authHeader = req.headers["authorization"];
  const token = authHeader && authHeader.split(" ")[1];

  if (!token) {
    return res.sendStatus(401);
  }

  try {
    const user = authService.verifyToken(token);
    req.user = user;
    next();
  } catch (error) {
    return res.sendStatus(403);
  }
};
```

### Rate Limiting

```typescript
import rateLimit from "express-rate-limit";

const createRateLimiter = (windowMs: number, max: number) => {
  return rateLimit({
    windowMs,
    max,
    message: "Too many requests from this IP",
    standardHeaders: true,
    legacyHeaders: false,
  });
};

// Different limits for different endpoints
app.use("/api/auth/login", createRateLimiter(15 * 60 * 1000, 5)); // 5 per 15 minutes
app.use("/api/", createRateLimiter(15 * 60 * 1000, 100)); // 100 per 15 minutes
```

## Performance Optimization

### Database Query Optimization

**N+1 Query Problem:**

```typescript
// Problem: N+1 queries
async function getUsersWithPosts() {
  const users = await User.findAll(); // 1 query

  for (const user of users) {
    user.posts = await Post.findAll({ where: { userId: user.id } }); // N queries
  }

  return users;
}

// Solution: Eager loading
async function getUsersWithPosts() {
  return await User.findAll({
    include: [{ model: Post, as: "posts" }], // 1 query with JOIN
  });
}
```

**Database Indexing:**

```sql
-- Single column index
CREATE INDEX idx_users_email ON users(email);

-- Composite index
CREATE INDEX idx_posts_user_created ON posts(user_id, created_at);

-- Partial index
CREATE INDEX idx_active_users ON users(email) WHERE active = true;
```

### Connection Pooling

```typescript
import { Pool } from "pg";

const pool = new Pool({
  host: "localhost",
  port: 5432,
  database: "myapp",
  user: "username",
  password: "password",
  max: 20, // Maximum connections
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

class DatabaseService {
  async query(text: string, params?: any[]): Promise<any> {
    const client = await pool.connect();
    try {
      const result = await client.query(text, params);
      return result.rows;
    } finally {
      client.release();
    }
  }
}
```

## Design Patterns in System Architecture

### Circuit Breaker Pattern

```typescript
enum CircuitState {
  CLOSED = "CLOSED",
  OPEN = "OPEN",
  HALF_OPEN = "HALF_OPEN",
}

class CircuitBreaker {
  private state = CircuitState.CLOSED;
  private failureCount = 0;
  private lastFailureTime?: Date;

  constructor(
    private failureThreshold: number = 5,
    private timeout: number = 60000,
    private retryAttempts: number = 3
  ) {}

  async call<T>(operation: () => Promise<T>): Promise<T> {
    if (this.state === CircuitState.OPEN) {
      if (this.shouldAttemptReset()) {
        this.state = CircuitState.HALF_OPEN;
      } else {
        throw new Error("Circuit breaker is OPEN");
      }
    }

    try {
      const result = await operation();
      this.onSuccess();
      return result;
    } catch (error) {
      this.onFailure();
      throw error;
    }
  }

  private onSuccess() {
    this.failureCount = 0;
    this.state = CircuitState.CLOSED;
  }

  private onFailure() {
    this.failureCount++;
    this.lastFailureTime = new Date();

    if (this.failureCount >= this.failureThreshold) {
      this.state = CircuitState.OPEN;
    }
  }

  private shouldAttemptReset(): boolean {
    return (
      this.lastFailureTime &&
      Date.now() - this.lastFailureTime.getTime() >= this.timeout
    );
  }
}
```

### Saga Pattern for Distributed Transactions

```typescript
// Choreography-based saga
class OrderSaga {
  async processOrder(orderData: CreateOrderDto) {
    try {
      // Step 1: Create order
      const order = await this.orderService.createOrder(orderData);

      // Step 2: Reserve inventory
      await this.inventoryService.reserveItems(order.items);

      // Step 3: Process payment
      await this.paymentService.processPayment(order.total);

      // Step 4: Confirm order
      await this.orderService.confirmOrder(order.id);
    } catch (error) {
      // Compensation logic
      await this.compensate(orderData, error);
      throw error;
    }
  }

  private async compensate(orderData: CreateOrderDto, error: Error) {
    // Reverse operations in reverse order
    try {
      await this.paymentService.refund(orderData.paymentId);
    } catch {}

    try {
      await this.inventoryService.releaseReservation(orderData.items);
    } catch {}

    try {
      await this.orderService.cancelOrder(orderData.orderId);
    } catch {}
  }
}
```

## Conclusion

System design is both an art and a science that requires balancing multiple trade-offs including performance, scalability, reliability, and maintainability. The key to successful system design is:

1. **Understanding Requirements**: Functional and non-functional requirements
2. **Choosing the Right Patterns**: Based on specific use cases and constraints
3. **Planning for Scale**: Design for current needs but plan for growth
4. **Monitoring and Iteration**: Continuous improvement based on real-world usage

Remember that there's no one-size-fits-all solution in system design. The best architecture is one that meets your specific requirements while remaining simple enough to understand, maintain, and evolve.

## Further Reading

- **Books**: "Designing Data-Intensive Applications" by Martin Kleppmann
- **Papers**: Google's MapReduce, Amazon's Dynamo, Facebook's TAO
- **Resources**: High Scalability blog, AWS Architecture Center
- **Practice**: System design interview questions and mock interviews
