---
title: "Microservices and Distributed Systems: Building Scalable Architecture"
excerpt: "Comprehensive guide to microservices architecture and distributed systems, covering design patterns, communication protocols, data consistency, fault tolerance, and deployment strategies."
date: "2024-12-18"
readTime: "20 min read"
category: "software-development"
subcategory: "Distributed Systems"
tags:
  [
    "Microservices",
    "Distributed Systems",
    "Docker",
    "Kubernetes",
    "Service Mesh",
    "API Gateway",
    "Event-Driven Architecture",
  ]
author: "Hiep Tran"
---

# Microservices and Distributed Systems: Building Scalable Architecture

Modern applications increasingly adopt distributed architectures to handle scale, improve resilience, and enable team autonomy. This comprehensive guide explores microservices architecture, distributed systems principles, and practical implementation strategies.

## Microservices Architecture Overview

### Monolithic vs Microservices

```mermaid
graph TD
    subgraph "Monolithic Architecture"
        A[User Interface] --> B[Business Logic Layer]
        B --> C[Data Access Layer]
        C --> D[(Database)]
    end

    subgraph "Microservices Architecture"
        E[API Gateway] --> F[User Service]
        E --> G[Product Service]
        E --> H[Order Service]
        E --> I[Payment Service]

        F --> J[(User DB)]
        G --> K[(Product DB)]
        H --> L[(Order DB)]
        I --> M[(Payment DB)]
    end
```

### Key Characteristics of Microservices

1. **Business Capability Focus**: Each service owns a specific business domain
2. **Decentralized**: Independent deployment and data management
3. **Fault Isolation**: Failures in one service don't cascade
4. **Technology Diversity**: Teams can choose appropriate tech stacks
5. **Organized Around Teams**: Conway's Law in practice

## Service Design Patterns

### Domain-Driven Design (DDD)

```python
# Domain model example for e-commerce microservices

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

@dataclass
class Product:
    """Product aggregate root"""
    id: str
    name: str
    price: float
    stock_quantity: int
    category: str

    def is_available(self, quantity: int) -> bool:
        return self.stock_quantity >= quantity

    def reserve_stock(self, quantity: int) -> bool:
        if self.is_available(quantity):
            self.stock_quantity -= quantity
            return True
        return False

@dataclass
class OrderItem:
    """Value object"""
    product_id: str
    quantity: int
    unit_price: float

    @property
    def total_price(self) -> float:
        return self.quantity * self.unit_price

@dataclass
class Order:
    """Order aggregate root"""
    id: str
    customer_id: str
    items: List[OrderItem]
    status: OrderStatus
    created_at: datetime

    def add_item(self, product: Product, quantity: int) -> bool:
        if product.is_available(quantity):
            item = OrderItem(
                product_id=product.id,
                quantity=quantity,
                unit_price=product.price
            )
            self.items.append(item)
            return True
        return False

    def calculate_total(self) -> float:
        return sum(item.total_price for item in self.items)

    def confirm(self) -> bool:
        if self.status == OrderStatus.PENDING and self.items:
            self.status = OrderStatus.CONFIRMED
            return True
        return False

# Repository pattern for data access
class OrderRepository:
    def save(self, order: Order) -> None:
        pass

    def find_by_id(self, order_id: str) -> Optional[Order]:
        pass

    def find_by_customer(self, customer_id: str) -> List[Order]:
        pass

# Domain service for complex business logic
class OrderService:
    def __init__(self, order_repo: OrderRepository, product_repo: ProductRepository):
        self.order_repo = order_repo
        self.product_repo = product_repo

    def create_order(self, customer_id: str, items: List[dict]) -> Order:
        order = Order(
            id=generate_uuid(),
            customer_id=customer_id,
            items=[],
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow()
        )

        for item_data in items:
            product = self.product_repo.find_by_id(item_data['product_id'])
            if product and product.reserve_stock(item_data['quantity']):
                order.add_item(product, item_data['quantity'])

        if order.items:
            self.order_repo.save(order)
            return order
        else:
            raise ValueError("No items could be added to order")
```

### Service Boundaries

```python
# Example of well-defined service boundaries

class UserService:
    """User management microservice"""

    def __init__(self):
        self.db = UserDatabase()
        self.auth_service = AuthenticationService()

    def create_user(self, user_data: dict) -> User:
        # Validate user data
        self.validate_user_data(user_data)

        # Hash password
        password_hash = self.auth_service.hash_password(user_data['password'])

        # Create user
        user = User(
            id=generate_uuid(),
            email=user_data['email'],
            password_hash=password_hash,
            profile=UserProfile(**user_data.get('profile', {}))
        )

        # Save to database
        self.db.save_user(user)

        # Publish domain event
        self.publish_event(UserCreatedEvent(user.id, user.email))

        return user

    def get_user_profile(self, user_id: str) -> UserProfile:
        user = self.db.find_user_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")
        return user.profile

class ProductCatalogService:
    """Product catalog microservice"""

    def __init__(self):
        self.db = ProductDatabase()
        self.search_engine = ElasticsearchClient()

    def add_product(self, product_data: dict) -> Product:
        product = Product(**product_data)

        # Save to database
        self.db.save_product(product)

        # Index for search
        self.search_engine.index_product(product)

        # Publish event
        self.publish_event(ProductAddedEvent(product.id))

        return product

    def search_products(self, query: str, filters: dict) -> List[Product]:
        return self.search_engine.search(query, filters)

class OrderManagementService:
    """Order management microservice"""

    def __init__(self):
        self.db = OrderDatabase()
        self.product_client = ProductServiceClient()
        self.payment_client = PaymentServiceClient()

    async def create_order(self, order_data: dict) -> Order:
        # Validate product availability
        products = await self.product_client.check_availability(
            order_data['items']
        )

        # Create order
        order = Order(
            id=generate_uuid(),
            customer_id=order_data['customer_id'],
            items=order_data['items'],
            status=OrderStatus.PENDING
        )

        # Save order
        self.db.save_order(order)

        # Reserve inventory
        await self.product_client.reserve_inventory(order.items)

        # Process payment
        payment_result = await self.payment_client.process_payment(
            order.calculate_total(),
            order_data['payment_method']
        )

        if payment_result.success:
            order.confirm()
            self.db.update_order(order)
            self.publish_event(OrderConfirmedEvent(order.id))

        return order
```

## Communication Patterns

### Synchronous Communication

```python
# REST API implementation with FastAPI

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import httpx
import asyncio

app = FastAPI(title="Order Service")

class CreateOrderRequest(BaseModel):
    customer_id: str
    items: List[dict]
    payment_method: str

class OrderResponse(BaseModel):
    id: str
    customer_id: str
    status: str
    total: float

# Circuit breaker pattern for resilient communication
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self):
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )

    def _on_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# Service client with circuit breaker
class ProductServiceClient:
    def __init__(self):
        self.base_url = "http://product-service:8000"
        self.circuit_breaker = CircuitBreaker()
        self.client = httpx.AsyncClient(timeout=5.0)

    async def check_product_availability(self, product_id: str, quantity: int):
        async def make_request():
            response = await self.client.get(
                f"{self.base_url}/products/{product_id}/availability",
                params={"quantity": quantity}
            )
            response.raise_for_status()
            return response.json()

        return await self.circuit_breaker.call(make_request)

    async def reserve_inventory(self, items: List[dict]):
        async def make_request():
            response = await self.client.post(
                f"{self.base_url}/inventory/reserve",
                json={"items": items}
            )
            response.raise_for_status()
            return response.json()

        return await self.circuit_breaker.call(make_request)

# API endpoint with service composition
@app.post("/orders", response_model=OrderResponse)
async def create_order(
    request: CreateOrderRequest,
    product_client: ProductServiceClient = Depends()
):
    try:
        # Check product availability
        for item in request.items:
            availability = await product_client.check_product_availability(
                item['product_id'],
                item['quantity']
            )
            if not availability['available']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Product {item['product_id']} not available"
                )

        # Create order
        order = await order_service.create_order(request.dict())

        return OrderResponse(
            id=order.id,
            customer_id=order.customer_id,
            status=order.status.value,
            total=order.calculate_total()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Asynchronous Communication

```python
# Event-driven architecture with Apache Kafka

import asyncio
import json
from kafka import KafkaProducer, KafkaConsumer
from dataclasses import dataclass, asdict
from typing import Dict, Any
import logging

@dataclass
class DomainEvent:
    event_id: str
    event_type: str
    aggregate_id: str
    data: Dict[Any, Any]
    timestamp: datetime
    version: int = 1

    def to_dict(self):
        return asdict(self)

class EventPublisher:
    def __init__(self, kafka_config: dict):
        self.producer = KafkaProducer(
            **kafka_config,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )

    def publish(self, event: DomainEvent, topic: str):
        """Publish domain event to Kafka topic"""
        try:
            future = self.producer.send(topic, event.to_dict())
            # Ensure message is sent
            record_metadata = future.get(timeout=10)
            logging.info(f"Event published: {event.event_type} to {topic}")
            return record_metadata
        except Exception as e:
            logging.error(f"Failed to publish event: {e}")
            raise

class EventHandler:
    def __init__(self, kafka_config: dict, group_id: str):
        self.consumer = KafkaConsumer(
            **kafka_config,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )

    def subscribe(self, topics: list):
        self.consumer.subscribe(topics)

    def process_events(self):
        """Process events from subscribed topics"""
        for message in self.consumer:
            try:
                event_data = message.value
                event_type = event_data.get('event_type')

                # Route to appropriate handler
                handler_method = getattr(self, f"handle_{event_type}", None)
                if handler_method:
                    handler_method(event_data)
                else:
                    logging.warning(f"No handler for event type: {event_type}")

            except Exception as e:
                logging.error(f"Error processing event: {e}")
                # Implement dead letter queue logic here

# Order service event handlers
class OrderEventHandler(EventHandler):
    def __init__(self, kafka_config: dict):
        super().__init__(kafka_config, "order-service-group")
        self.inventory_service = InventoryServiceClient()
        self.notification_service = NotificationServiceClient()

    def handle_user_created(self, event_data: dict):
        """Handle user creation event"""
        user_id = event_data['aggregate_id']
        logging.info(f"New user created: {user_id}")
        # Initialize user's order history, send welcome email, etc.

    def handle_payment_processed(self, event_data: dict):
        """Handle payment processing event"""
        order_id = event_data['data']['order_id']
        payment_status = event_data['data']['status']

        if payment_status == 'SUCCESS':
            # Confirm order and notify customer
            order = self.order_service.get_order(order_id)
            order.confirm()
            self.order_service.save_order(order)

            # Send confirmation notification
            self.notification_service.send_order_confirmation(order)
        else:
            # Handle payment failure
            self.handle_payment_failure(order_id)

    def handle_inventory_reserved(self, event_data: dict):
        """Handle inventory reservation event"""
        order_id = event_data['data']['order_id']
        reservation_id = event_data['data']['reservation_id']

        # Update order with reservation details
        order = self.order_service.get_order(order_id)
        order.inventory_reservation_id = reservation_id
        self.order_service.save_order(order)

# Saga pattern for distributed transactions
class OrderSaga:
    def __init__(self, event_publisher: EventPublisher):
        self.event_publisher = event_publisher
        self.steps = []

    def add_step(self, action, compensation):
        """Add step to saga"""
        self.steps.append({
            'action': action,
            'compensation': compensation,
            'executed': False
        })

    async def execute(self, order_data: dict):
        """Execute saga steps"""
        executed_steps = []

        try:
            for i, step in enumerate(self.steps):
                # Execute step
                result = await step['action'](order_data)
                step['executed'] = True
                executed_steps.append(i)

                # Update order data with step result
                order_data.update(result)

        except Exception as e:
            # Compensate executed steps in reverse order
            await self.compensate(executed_steps)
            raise e

    async def compensate(self, executed_steps: list):
        """Compensate executed steps in reverse order"""
        for step_index in reversed(executed_steps):
            try:
                step = self.steps[step_index]
                await step['compensation'](step_index)
            except Exception as e:
                logging.error(f"Compensation failed for step {step_index}: {e}")

# Saga implementation for order creation
async def create_order_saga(order_data: dict):
    saga = OrderSaga(event_publisher)

    # Step 1: Reserve inventory
    saga.add_step(
        action=lambda data: inventory_service.reserve_items(data['items']),
        compensation=lambda _: inventory_service.release_reservation(order_data['reservation_id'])
    )

    # Step 2: Process payment
    saga.add_step(
        action=lambda data: payment_service.charge_payment(data['payment_method'], data['total']),
        compensation=lambda _: payment_service.refund_payment(order_data['payment_id'])
    )

    # Step 3: Create order record
    saga.add_step(
        action=lambda data: order_service.create_order_record(data),
        compensation=lambda _: order_service.delete_order(order_data['order_id'])
    )

    # Step 4: Send confirmation
    saga.add_step(
        action=lambda data: notification_service.send_confirmation(data['customer_id'], data['order_id']),
        compensation=lambda _: notification_service.send_cancellation(order_data['customer_id'])
    )

    await saga.execute(order_data)
```

## Data Management in Microservices

### Database per Service Pattern

```python
# Database per service implementation

class UserService:
    def __init__(self):
        # User service uses PostgreSQL for relational data
        self.db = PostgreSQLDatabase("user_db")

    def create_user(self, user_data: dict):
        with self.db.transaction():
            user = User(**user_data)
            self.db.users.insert(user)

            # Publish event for other services
            event = UserCreatedEvent(user.id, user.email)
            self.event_publisher.publish(event, "user-events")

class ProductCatalogService:
    def __init__(self):
        # Product service uses MongoDB for document-based data
        self.db = MongoDatabase("product_catalog")
        self.search_engine = ElasticsearchClient()

    def add_product(self, product_data: dict):
        product = Product(**product_data)

        # Save to MongoDB
        self.db.products.insert_one(product.to_dict())

        # Index in Elasticsearch for search
        self.search_engine.index_document("products", product.to_dict())

class AnalyticsService:
    def __init__(self):
        # Analytics service uses time-series database
        self.db = InfluxDBClient("analytics_db")

    def record_event(self, event_data: dict):
        point = {
            "measurement": "user_events",
            "tags": {
                "event_type": event_data["type"],
                "user_id": event_data["user_id"]
            },
            "fields": {
                "value": 1,
                "metadata": json.dumps(event_data.get("metadata", {}))
            },
            "time": datetime.utcnow()
        }
        self.db.write_points([point])
```

### Data Consistency Patterns

```python
# Event sourcing implementation

@dataclass
class Event:
    aggregate_id: str
    event_type: str
    event_data: dict
    event_version: int
    timestamp: datetime

class EventStore:
    def __init__(self):
        self.db = PostgreSQLDatabase()

    def append_events(self, aggregate_id: str, events: List[Event], expected_version: int):
        """Append events to event store with optimistic concurrency control"""
        with self.db.transaction():
            # Check current version
            current_version = self.get_current_version(aggregate_id)
            if current_version != expected_version:
                raise ConcurrencyError(f"Expected version {expected_version}, got {current_version}")

            # Append new events
            for i, event in enumerate(events):
                event.event_version = expected_version + i + 1
                self.db.events.insert({
                    'aggregate_id': event.aggregate_id,
                    'event_type': event.event_type,
                    'event_data': json.dumps(event.event_data),
                    'event_version': event.event_version,
                    'timestamp': event.timestamp
                })

    def get_events(self, aggregate_id: str, from_version: int = 0) -> List[Event]:
        """Get events for an aggregate"""
        rows = self.db.events.select(
            where="aggregate_id = %s AND event_version > %s",
            params=[aggregate_id, from_version],
            order_by="event_version"
        )

        return [self._row_to_event(row) for row in rows]

class OrderAggregate:
    def __init__(self, aggregate_id: str):
        self.aggregate_id = aggregate_id
        self.version = 0
        self.uncommitted_events = []
        self.status = OrderStatus.PENDING
        self.items = []
        self.total = 0.0

    def add_item(self, product_id: str, quantity: int, price: float):
        """Add item to order"""
        event = Event(
            aggregate_id=self.aggregate_id,
            event_type="ItemAdded",
            event_data={
                "product_id": product_id,
                "quantity": quantity,
                "price": price
            },
            event_version=self.version + len(self.uncommitted_events) + 1,
            timestamp=datetime.utcnow()
        )

        self.uncommitted_events.append(event)
        self._apply_event(event)

    def confirm_order(self):
        """Confirm the order"""
        if self.status != OrderStatus.PENDING:
            raise ValueError("Order cannot be confirmed")

        event = Event(
            aggregate_id=self.aggregate_id,
            event_type="OrderConfirmed",
            event_data={"confirmed_at": datetime.utcnow().isoformat()},
            event_version=self.version + len(self.uncommitted_events) + 1,
            timestamp=datetime.utcnow()
        )

        self.uncommitted_events.append(event)
        self._apply_event(event)

    def _apply_event(self, event: Event):
        """Apply event to aggregate state"""
        if event.event_type == "ItemAdded":
            data = event.event_data
            item = OrderItem(
                product_id=data["product_id"],
                quantity=data["quantity"],
                price=data["price"]
            )
            self.items.append(item)
            self.total += item.quantity * item.price

        elif event.event_type == "OrderConfirmed":
            self.status = OrderStatus.CONFIRMED

    def load_from_history(self, events: List[Event]):
        """Rebuild aggregate state from events"""
        for event in events:
            self._apply_event(event)
            self.version = event.event_version

    def get_uncommitted_events(self) -> List[Event]:
        """Get uncommitted events"""
        return self.uncommitted_events.copy()

    def mark_events_as_committed(self):
        """Mark events as committed"""
        self.version += len(self.uncommitted_events)
        self.uncommitted_events.clear()

# Repository using event sourcing
class OrderRepository:
    def __init__(self, event_store: EventStore):
        self.event_store = event_store

    def save(self, order: OrderAggregate):
        """Save order aggregate"""
        events = order.get_uncommitted_events()
        if events:
            self.event_store.append_events(
                order.aggregate_id,
                events,
                order.version
            )
            order.mark_events_as_committed()

    def get_by_id(self, order_id: str) -> OrderAggregate:
        """Load order aggregate from events"""
        events = self.event_store.get_events(order_id)
        if not events:
            raise OrderNotFoundError(f"Order {order_id} not found")

        order = OrderAggregate(order_id)
        order.load_from_history(events)
        return order
```

## Service Mesh and Infrastructure

### Istio Service Mesh Configuration

```yaml
# Istio configuration for microservices

# Virtual Service for traffic routing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
spec:
  hosts:
    - order-service
  http:
    - match:
        - headers:
            canary:
              exact: "true"
      route:
        - destination:
            host: order-service
            subset: v2
          weight: 100
    - route:
        - destination:
            host: order-service
            subset: v1
          weight: 100

---
# Destination Rule for load balancing
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: order-service
spec:
  host: order-service
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 10
    circuitBreaker:
      consecutiveErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2

---
# Gateway for external traffic
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: microservices-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
    - port:
        number: 80
        name: http
        protocol: HTTP
      hosts:
        - api.example.com
    - port:
        number: 443
        name: https
        protocol: HTTPS
      tls:
        mode: SIMPLE
        credentialName: api-cert
      hosts:
        - api.example.com
```

### API Gateway Implementation

```python
# API Gateway with FastAPI and authentication

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import jwt
import asyncio
from typing import Dict, Any

app = FastAPI(title="API Gateway")
security = HTTPBearer()

class ServiceRegistry:
    """Service discovery and routing"""

    def __init__(self):
        self.services = {
            "user": "http://user-service:8000",
            "product": "http://product-service:8000",
            "order": "http://order-service:8000",
            "payment": "http://payment-service:8000"
        }

    def get_service_url(self, service_name: str) -> str:
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        return self.services[service_name]

class RateLimiter:
    """Rate limiting middleware"""

    def __init__(self):
        self.requests = {}  # In production, use Redis

    async def check_rate_limit(self, client_id: str, limit: int = 100, window: int = 60):
        current_time = time.time()
        window_start = current_time - window

        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id]
                if req_time > window_start
            ]
        else:
            self.requests[client_id] = []

        # Check limit
        if len(self.requests[client_id]) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )

        # Add current request
        self.requests[client_id].append(current_time)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            "secret-key",  # In production, use proper key management
            algorithms=["HS256"]
        )
        return payload
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Request logging middleware"""
    start_time = time.time()

    # Log request
    logging.info(f"Request: {request.method} {request.url}")

    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logging.info(f"Response: {response.status_code} ({process_time:.3f}s)")

    return response

# Service proxy endpoints
@app.api_route("/{service}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_request(
    service: str,
    path: str,
    request: Request,
    user_data: dict = Depends(verify_token)
):
    """Proxy requests to microservices"""

    # Rate limiting
    rate_limiter = RateLimiter()
    await rate_limiter.check_rate_limit(user_data["user_id"])

    # Get service URL
    service_registry = ServiceRegistry()
    service_url = service_registry.get_service_url(service)

    # Prepare request
    url = f"{service_url}/{path}"
    headers = dict(request.headers)
    headers["X-User-ID"] = user_data["user_id"]
    headers["X-User-Role"] = user_data.get("role", "user")

    # Forward request
    async with httpx.AsyncClient() as client:
        if request.method == "GET":
            response = await client.get(url, headers=headers, params=request.query_params)
        else:
            body = await request.body()
            response = await client.request(
                request.method,
                url,
                content=body,
                headers=headers,
                params=request.query_params
            )

    return response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text

# Health check aggregation
@app.get("/health")
async def health_check():
    """Aggregate health status from all services"""
    service_registry = ServiceRegistry()
    health_statuses = {}

    async def check_service_health(service_name: str, service_url: str):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{service_url}/health")
                health_statuses[service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "response_time": response.elapsed.total_seconds()
                }
        except Exception as e:
            health_statuses[service_name] = {
                "status": "unhealthy",
                "error": str(e)
            }

    # Check all services concurrently
    tasks = [
        check_service_health(name, url)
        for name, url in service_registry.services.items()
    ]
    await asyncio.gather(*tasks)

    # Determine overall health
    overall_status = "healthy" if all(
        status["status"] == "healthy"
        for status in health_statuses.values()
    ) else "degraded"

    return {
        "status": overall_status,
        "services": health_statuses,
        "timestamp": datetime.utcnow().isoformat()
    }
```

## Deployment and Operations

### Kubernetes Deployment

```yaml
# Kubernetes deployment for microservices

# Order Service Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: order-service
  labels:
    app: order-service
    version: v1
spec:
  replicas: 3
  selector:
    matchLabels:
      app: order-service
      version: v1
  template:
    metadata:
      labels:
        app: order-service
        version: v1
    spec:
      containers:
        - name: order-service
          image: order-service:v1.0.0
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: order-service-secrets
                  key: database-url
            - name: KAFKA_BROKERS
              value: "kafka:9092"
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5

---
# Service
apiVersion: v1
kind: Service
metadata:
  name: order-service
  labels:
    app: order-service
spec:
  ports:
    - port: 8000
      targetPort: 8000
  selector:
    app: order-service

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: order-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: order-service
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80

---
# ConfigMap for application configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: order-service-config
data:
  app.yaml: |
    server:
      port: 8000
      host: "0.0.0.0"
    logging:
      level: "INFO"
      format: "json"
    features:
      enable_cache: true
      cache_ttl: 300
```

### Docker Compose for Local Development

```yaml
# docker-compose.yml for local microservices development

version: "3.8"
services:
  # Infrastructure services
  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:latest
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: microservices
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Microservices
  api-gateway:
    build: ./api-gateway
    ports:
      - "8080:8000"
    environment:
      - USER_SERVICE_URL=http://user-service:8000
      - PRODUCT_SERVICE_URL=http://product-service:8000
      - ORDER_SERVICE_URL=http://order-service:8000
    depends_on:
      - user-service
      - product-service
      - order-service

  user-service:
    build: ./user-service
    ports:
      - "8001:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/microservices
      - KAFKA_BROKERS=kafka:9092
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - kafka
      - redis

  product-service:
    build: ./product-service
    ports:
      - "8002:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/microservices
      - KAFKA_BROKERS=kafka:9092
      - ELASTICSEARCH_URL=http://elasticsearch:9200
    depends_on:
      - postgres
      - kafka

  order-service:
    build: ./order-service
    ports:
      - "8003:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/microservices
      - KAFKA_BROKERS=kafka:9092
      - USER_SERVICE_URL=http://user-service:8000
      - PRODUCT_SERVICE_URL=http://product-service:8000
    depends_on:
      - postgres
      - kafka
      - user-service
      - product-service

volumes:
  postgres_data:
```

## Monitoring and Observability

### Distributed Tracing

```python
# OpenTelemetry implementation for distributed tracing

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

# Initialize tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument HTTP requests and database calls
RequestsInstrumentor().instrument()
SQLAlchemyInstrumentor().instrument()

# Manual instrumentation
class OrderService:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)

    async def create_order(self, order_data: dict):
        with self.tracer.start_as_current_span("create_order") as span:
            # Add custom attributes
            span.set_attribute("order.customer_id", order_data["customer_id"])
            span.set_attribute("order.item_count", len(order_data["items"]))

            try:
                # Validate order
                with self.tracer.start_as_current_span("validate_order"):
                    self.validate_order(order_data)

                # Check inventory
                with self.tracer.start_as_current_span("check_inventory") as inventory_span:
                    inventory_result = await self.check_inventory(order_data["items"])
                    inventory_span.set_attribute("inventory.available", inventory_result["available"])

                # Process payment
                with self.tracer.start_as_current_span("process_payment") as payment_span:
                    payment_result = await self.process_payment(order_data)
                    payment_span.set_attribute("payment.transaction_id", payment_result["transaction_id"])

                # Create order record
                with self.tracer.start_as_current_span("save_order"):
                    order = await self.save_order(order_data)

                span.set_attribute("order.id", order.id)
                span.set_status(trace.Status(trace.StatusCode.OK))

                return order

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(
                    trace.StatusCode.ERROR,
                    str(e)
                ))
                raise
```

### Metrics and Logging

```python
# Prometheus metrics and structured logging

from prometheus_client import Counter, Histogram, Gauge, start_http_server
import structlog
import json

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active database connections')
ORDER_PROCESSING_TIME = Histogram('order_processing_seconds', 'Order processing time', ['status'])

# Structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            method = scope["method"]
            path = scope["path"]

            start_time = time.time()

            # Process request
            async def send_with_metrics(message):
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    duration = time.time() - start_time

                    # Record metrics
                    REQUEST_COUNT.labels(
                        method=method,
                        endpoint=path,
                        status=status_code
                    ).inc()

                    REQUEST_DURATION.labels(
                        method=method,
                        endpoint=path
                    ).observe(duration)

                    # Structured logging
                    logger.info(
                        "HTTP request completed",
                        method=method,
                        path=path,
                        status_code=status_code,
                        duration=duration,
                        user_agent=scope.get("headers", {}).get("user-agent")
                    )

                await send(message)

            await self.app(scope, receive, send_with_metrics)
        else:
            await self.app(scope, receive, send)

# Business metrics
class OrderMetrics:
    def __init__(self):
        self.logger = structlog.get_logger()

    def record_order_created(self, order: dict):
        """Record order creation metrics"""
        ORDER_PROCESSING_TIME.labels(status="created").observe(time.time())

        self.logger.info(
            "Order created",
            order_id=order["id"],
            customer_id=order["customer_id"],
            total_amount=order["total"],
            item_count=len(order["items"])
        )

    def record_order_failed(self, order_data: dict, error: str):
        """Record order failure metrics"""
        ORDER_PROCESSING_TIME.labels(status="failed").observe(time.time())

        self.logger.error(
            "Order creation failed",
            customer_id=order_data.get("customer_id"),
            error=error,
            order_data=order_data
        )

# Start metrics server
start_http_server(8000)
```

## Conclusion

Microservices and distributed systems enable organizations to build scalable, resilient applications while supporting team autonomy and technology diversity. However, they introduce complexity in areas like data consistency, service communication, and operational overhead.

Key principles for success:

- **Start with a monolith**: Don't begin with microservices unless you have specific scalability needs
- **Domain-driven design**: Use business capabilities to define service boundaries
- **Embrace eventual consistency**: Design for asynchronous, event-driven communication
- **Invest in observability**: Comprehensive monitoring and tracing are essential
- **Automate everything**: CI/CD, testing, and deployment automation are critical
- **Plan for failure**: Implement circuit breakers, timeouts, and graceful degradation

The microservices architecture pattern is powerful but should be adopted thoughtfully, with proper investment in tooling, processes, and team capabilities.

---

_Ready to implement microservices? Start by identifying bounded contexts in your domain, then gradually extract services from a well-understood monolith._
