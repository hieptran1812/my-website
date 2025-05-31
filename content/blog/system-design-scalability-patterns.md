---
title: "System Design Interview Prep: Scalability Patterns"
excerpt: "Key architectural patterns and design principles for building scalable distributed systems."
date: "2024-03-10"
readTime: "12 min read"
tags: ["System Design", "Architecture", "Scalability"]
category: "notes"
---

# System Design Interview Prep: Scalability Patterns

System design interviews test your ability to architect large-scale distributed systems. This guide covers essential patterns and principles for building scalable systems.

## Core Principles

### Scalability Dimensions

1. **Vertical Scaling (Scale Up)**: Adding more power to existing machines
2. **Horizontal Scaling (Scale Out)**: Adding more machines to the resource pool

### CAP Theorem

You can only guarantee two of the three:

- **Consistency**: All nodes see the same data simultaneously
- **Availability**: System remains operational
- **Partition Tolerance**: System continues despite network failures

## Load Balancing Patterns

### Layer 4 vs Layer 7 Load Balancing

```
Layer 4 (Transport Layer):
- Routes based on IP and port
- Faster, less CPU intensive
- Cannot inspect application data

Layer 7 (Application Layer):
- Routes based on content (HTTP headers, URLs)
- More intelligent routing
- SSL termination, compression
```

### Load Balancing Algorithms

```python
# Round Robin
class RoundRobinBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0

    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# Weighted Round Robin
class WeightedRoundRobinBalancer:
    def __init__(self, servers_weights):
        self.servers = []
        for server, weight in servers_weights.items():
            self.servers.extend([server] * weight)
        self.current = 0

    def get_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# Least Connections
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

```python
# Cache-Aside (Lazy Loading)
def get_user(user_id):
    # Check cache first
    user = cache.get(f"user:{user_id}")
    if user:
        return user

    # Load from database
    user = database.get_user(user_id)

    # Store in cache
    cache.set(f"user:{user_id}", user, ttl=3600)
    return user

# Write-Through
def update_user(user_id, data):
    # Update database
    user = database.update_user(user_id, data)

    # Update cache
    cache.set(f"user:{user_id}", user, ttl=3600)
    return user

# Write-Behind (Write-Back)
def update_user_async(user_id, data):
    # Update cache immediately
    cache.set(f"user:{user_id}", data, ttl=3600)

    # Queue database update for later
    queue.enqueue('update_user_db', user_id, data)
```

### Cache Levels

```
1. Browser Cache (Client-side)
2. CDN Cache (Edge servers)
3. Reverse Proxy Cache (nginx, Varnish)
4. Application Cache (Redis, Memcached)
5. Database Cache (Query result cache)
```

## Database Patterns

### Partitioning Strategies

```python
# Horizontal Partitioning (Sharding)
class UserSharding:
    def __init__(self, shards):
        self.shards = shards

    def get_shard(self, user_id):
        shard_key = hash(user_id) % len(self.shards)
        return self.shards[shard_key]

    def get_user(self, user_id):
        shard = self.get_shard(user_id)
        return shard.get_user(user_id)

# Range-based Partitioning
class RangePartitioning:
    def __init__(self, partitions):
        # partitions = [('A-F', db1), ('G-M', db2), ('N-Z', db3)]
        self.partitions = partitions

    def get_partition(self, key):
        for range_def, db in self.partitions:
            if self.in_range(key, range_def):
                return db
        raise ValueError("No partition found for key")

# Directory-based Partitioning
class DirectoryPartitioning:
    def __init__(self):
        self.lookup_service = {}  # key -> partition mapping

    def get_partition(self, key):
        return self.lookup_service.get(key)
```

### Database Scaling Patterns

```sql
-- Read Replicas
MASTER (Write) -> SLAVE1 (Read)
                -> SLAVE2 (Read)
                -> SLAVE3 (Read)

-- Master-Master Replication
MASTER1 <-> MASTER2
   |           |
SLAVE1      SLAVE2

-- Federation (Split databases by function)
Users DB    |  Products DB  |  Orders DB
User Service|Product Service|Order Service
```

## Microservices Patterns

### Service Communication

```python
# Synchronous Communication (HTTP/REST)
import requests

class UserService:
    def get_user_orders(self, user_id):
        user = requests.get(f"http://user-service/users/{user_id}")
        orders = requests.get(f"http://order-service/users/{user_id}/orders")
        return {
            'user': user.json(),
            'orders': orders.json()
        }

# Asynchronous Communication (Message Queues)
import pika

class EventPublisher:
    def __init__(self):
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters('localhost')
        )
        self.channel = self.connection.channel()

    def publish_event(self, event_type, data):
        self.channel.basic_publish(
            exchange='events',
            routing_key=event_type,
            body=json.dumps(data)
        )

# Event-Driven Architecture
class OrderService:
    def create_order(self, order_data):
        order = self.save_order(order_data)

        # Publish events
        self.publisher.publish_event('order.created', {
            'order_id': order.id,
            'user_id': order.user_id,
            'total': order.total
        })

        return order
```

### Circuit Breaker Pattern

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = 1    # Normal operation
    OPEN = 2      # Failing fast
    HALF_OPEN = 3 # Testing if service recovered

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def _should_attempt_reset(self):
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

# Usage
circuit_breaker = CircuitBreaker()

def call_external_service():
    return circuit_breaker.call(external_api_call)
```

## Content Delivery Network (CDN)

### CDN Strategies

```javascript
// Push CDN - Upload content when it changes
class PushCDN {
  uploadContent(content, path) {
    // Upload to all CDN edge servers
    this.edgeServers.forEach((server) => {
      server.upload(path, content);
    });
  }
}

// Pull CDN - Fetch content on first request
class PullCDN {
  getContent(path, userLocation) {
    const edgeServer = this.getNearestEdgeServer(userLocation);

    if (!edgeServer.hasContent(path)) {
      const content = this.originServer.getContent(path);
      edgeServer.cache(path, content);
    }

    return edgeServer.getContent(path);
  }
}
```

## Message Queues and Event Streaming

### Message Queue Patterns

```python
# Point-to-Point (Queue)
class TaskQueue:
    def __init__(self):
        self.queue = []

    def enqueue(self, task):
        self.queue.append(task)

    def dequeue(self):
        return self.queue.pop(0) if self.queue else None

# Publish-Subscribe (Topic)
class PubSub:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, topic, callback):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def publish(self, topic, message):
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(message)

# Event Sourcing
class EventStore:
    def __init__(self):
        self.events = []

    def append_event(self, event):
        event['timestamp'] = time.time()
        event['version'] = len(self.events) + 1
        self.events.append(event)

    def get_events(self, aggregate_id):
        return [e for e in self.events if e['aggregate_id'] == aggregate_id]

    def replay_events(self, aggregate_id):
        events = self.get_events(aggregate_id)
        state = {}
        for event in events:
            state = self.apply_event(state, event)
        return state
```

## Common System Design Components

### Rate Limiting

```python
import time
from collections import defaultdict

class TokenBucketRateLimiter:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.buckets = defaultdict(lambda: {
            'tokens': capacity,
            'last_refill': time.time()
        })

    def is_allowed(self, key):
        bucket = self.buckets[key]
        now = time.time()

        # Refill tokens
        time_passed = now - bucket['last_refill']
        tokens_to_add = time_passed * self.refill_rate
        bucket['tokens'] = min(self.capacity, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now

        # Check if request is allowed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return True
        return False

class SlidingWindowRateLimiter:
    def __init__(self, window_size, max_requests):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = defaultdict(list)

    def is_allowed(self, key):
        now = time.time()
        window_start = now - self.window_size

        # Remove old requests
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > window_start
        ]

        # Check if under limit
        if len(self.requests[key]) < self.max_requests:
            self.requests[key].append(now)
            return True
        return False
```

### Consistent Hashing

```python
import hashlib
import bisect

class ConsistentHash:
    def __init__(self, nodes=None, replicas=3):
        self.replicas = replicas
        self.ring = {}
        self.sorted_keys = []

        if nodes:
            for node in nodes:
                self.add_node(node)

    def add_node(self, node):
        for i in range(self.replicas):
            key = self.hash(f"{node}:{i}")
            self.ring[key] = node
            bisect.insort(self.sorted_keys, key)

    def remove_node(self, node):
        for i in range(self.replicas):
            key = self.hash(f"{node}:{i}")
            del self.ring[key]
            self.sorted_keys.remove(key)

    def get_node(self, key):
        if not self.ring:
            return None

        hash_key = self.hash(key)
        idx = bisect.bisect_right(self.sorted_keys, hash_key)

        if idx == len(self.sorted_keys):
            idx = 0

        return self.ring[self.sorted_keys[idx]]

    def hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
```

## System Design Process

### 1. Requirements Gathering

- Functional requirements
- Non-functional requirements (scale, performance)
- Constraints and assumptions

### 2. Capacity Estimation

```python
# Example: Social Media Platform
DAU = 100_000_000  # Daily Active Users
Posts_per_user_per_day = 2
Reads_per_user_per_day = 50

# Write QPS
Write_QPS = (DAU * Posts_per_user_per_day) / (24 * 3600)
# = 100M * 2 / 86400 ≈ 2,315 writes/second

# Read QPS
Read_QPS = (DAU * Reads_per_user_per_day) / (24 * 3600)
# = 100M * 50 / 86400 ≈ 57,870 reads/second

# Storage
Post_size = 1_KB  # Average post size
Daily_storage = DAU * Posts_per_user_per_day * Post_size
# = 100M * 2 * 1KB = 200GB per day
```

### 3. High-Level Design

- Major components
- Data flow
- API design

### 4. Detailed Design

- Database schema
- Algorithms
- Scalability solutions

### 5. Scale the Design

- Identify bottlenecks
- Address scaling challenges
- Monitor and optimize

## Conclusion

Key patterns for scalable systems:

1. **Horizontal scaling** over vertical scaling
2. **Microservices** for modularity and independent scaling
3. **Caching** at multiple levels
4. **Load balancing** for distributing traffic
5. **Database sharding** for data scaling
6. **Asynchronous processing** for better performance
7. **Circuit breakers** for fault tolerance
8. **Rate limiting** for protecting resources

Remember: There's no one-size-fits-all solution. Choose patterns based on your specific requirements, constraints, and scale.
