---
title: "Anatomy of a Well-Built Microservice: Your First Production-Grade Service"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Build the Orders service of ShopFast the right way — layered internals, ports and adapters, 12-factor config, graceful shutdown, health endpoints, and a Dockerfile a senior will sign off on."
tags:
  [
    "microservices",
    "hexagonal-architecture",
    "12-factor",
    "graceful-shutdown",
    "health-checks",
    "go",
    "docker",
    "distributed-systems",
    "software-architecture",
    "backend",
  ]
category: "software-development"
subcategory: "Microservices"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/anatomy-of-a-well-built-microservice-1.webp"
---

There is a moment every backend engineer remembers: the first time someone hands you a fresh repository and says, "build the Orders service." Not a feature inside a giant codebase that already has all the wiring — a *whole service*, alone, that has to start up, take traffic, talk to a database, emit events, get deployed every afternoon by a CI pipeline you didn't write, get probed by Kubernetes you don't fully understand, and survive a 3am page when something downstream falls over. The blank `main.go` (or empty `app/`) is staring at you. Where do the files go? What does the inside of a *good* service actually look like?

I have reviewed that first service from dozens of engineers. The ones that go badly all fail the same way: they put the database query right inside the HTTP handler, they hard-code the connection string, they `os.Exit(0)` on shutdown and silently drop forty in-flight requests every time the team deploys, and six months later the service is a tiny monolith that nobody can test without spinning up Postgres. The ones that go well share an internal shape — a shape a junior can copy on day one and a staff engineer still respects after a thousand deploys. This post is that shape, built end to end, with real code.

We are going to build the **Orders service of ShopFast**, a fictional e-commerce platform that will be the running example for this whole series. By the end you will be able to lay out the folders, wire the layers so the domain logic never imports a web framework or a database driver, validate your config at boot, expose health endpoints Kubernetes actually understands, drain in-flight requests on `SIGTERM` so a deploy drops zero requests, and ship it in a Docker image that is small, non-root, and fast to start. The figure below is the map of the whole article: four layers stacked with dependencies pointing strictly inward, the platform underneath. We will build each layer, then wrap the service in the production "table stakes" no service ships without.

![Four layers of a single service stacked with dependencies pointing inward toward the domain](/imgs/blogs/anatomy-of-a-well-built-microservice-1.webp)

Read the stack top to bottom. The **transport** layer speaks HTTP and knows about JSON; the **application** layer holds use-cases and the *ports* (interfaces) the service needs; the **domain** layer holds the business rules and knows about nothing else; the **infrastructure** layer holds the Postgres driver, the message-queue client, and the environment. Crucially, the arrows of dependency point *down toward the domain* — the domain is the only layer that imports nobody. Get that one rule right and almost everything else in this post follows from it. This is the practitioner's layer of microservices: not "should we split the monolith" (that was [post 1](/blog/software-development/microservices/what-are-microservices-and-when-not-to-use-them)) but "given that we are building one service, what does the inside look like."

## Why the inside of a service matters more than you think

A junior often believes the hard part of microservices is the *between* — the network calls, the gateways, the service mesh. Those are hard, and we cover them across this series. But the most expensive mistakes I have seen were made *inside* a single service, on day one, by someone who thought the internal structure was a matter of taste. It is not. The internal shape of a service decides three things that compound for years.

First, **testability**. If your order-pricing logic lives inside an HTTP handler that reaches directly into Postgres, you cannot test "an order over \$500 gets free shipping" without standing up an HTTP server and a database. So you don't test it. So it breaks. A service whose business rules live in a pure domain layer can test that rule in microseconds with zero infrastructure, and engineers who can test cheaply actually write tests.

Second, **changeability**. ShopFast will, I promise you, change its datastore at least once — Postgres to a managed Aurora, a hot read path to Redis, an events sink from one broker to another. If the database calls are scattered through forty handlers, that is a forty-file rewrite. If they sit behind one *port* (an interface) implemented by one *adapter*, it is a one-file swap. The internal shape decides whether change is a Tuesday or a quarter.

Third, **the distributed monolith trap**. The single most damaging anti-pattern in microservices is building services that *look* independent but are secretly fused — they share a database, or import a common "core" library full of mutable business logic, so you cannot deploy one without deploying all. We will name exactly what not to do, because a clean internal shape is your first defense against accidentally rebuilding the monolith with network calls in the middle (the slow, painful kind). The boundary you draw with [domain-driven design](/blog/software-development/microservices/service-boundaries-with-domain-driven-design) only holds if the code inside respects it.

There is a fourth reason that is less obvious and more strategic: **the internal shape determines how new people contribute.** A service whose structure is legible — folders that map to layers, a single composition root, business rules in one place — is one a new hire can be productive in within a day, because the *where does this go* question answers itself. A service that is a pile of fat handlers each doing everything is one where every change requires reading the whole thing, where two engineers touch the same 400-line handler and conflict, where the senior becomes the bottleneck because only they hold the map in their head. At a small company this is annoying; at a company running dozens of services across many teams it is existential, because the whole *point* of microservices is to let teams move independently, and they cannot move independently inside a service they cannot read. The internal anatomy is, quietly, an organizational decision dressed up as a technical one — which is the recurring theme of microservices and exactly the Conway's-Law angle we return to in the case studies.

So: the inside matters. It is not taste, it is leverage that compounds — on tests, on changes, on onboarding, on whether you accidentally rebuild the monolith. Let's build it.

## The four layers, concretely

Every well-built service I have worked on can be described as four concentric responsibilities, even when the team has never heard the word "hexagonal." Let me define each in plain terms, with the Orders service as the example.

The **domain layer** is the business itself, expressed in code, with no idea that HTTP, databases, or JSON exist. For ShopFast Orders, the domain is the `Order` type, the rule that an order must have at least one line item, the rule that you cannot confirm an order that has already been cancelled, the calculation of the order total, and the rule that orders over a threshold qualify for free shipping. If you deleted every framework and every database from the universe, the domain layer would still compile and its tests would still pass. This is the heart, and it is the layer juniors most often skip — they spread business rules across handlers and SQL, and the "domain" becomes an anemic bag of fields.

The **application layer** (also called the use-case layer or service layer — careful, "service" is overloaded) orchestrates a single unit of work: *place an order*, *cancel an order*, *get an order*. A use-case does not contain business rules (those are in the domain); it *coordinates*. It loads an `Order` through a port, asks the domain object to do something, persists the result through a port, and publishes an event through a port. The application layer is where transactions, idempotency, and authorization checks live. It defines the **ports** — the interfaces it needs the outside world to satisfy — but it does not know who implements them.

The **transport layer** (or API layer, or "driving adapter") translates the outside world into use-case calls. For an HTTP service it parses the request body, validates the shape, maps it to a command, calls the use-case, and maps the result back to an HTTP status and JSON body. It is deliberately thin. It should contain no business rules and no SQL — if you find an `if order.Total > 500` inside a handler, it has leaked.

The **infrastructure layer** (the "driven adapters") implements the ports the application layer declared. The Postgres `OrderRepository` that turns a `Save(order)` into an `INSERT`. The Kafka publisher that turns `Publish(event)` into a produce call. The config loader that reads the environment. These adapters import drivers and frameworks freely — that is their whole job.

The discipline that makes this work is a single rule: **dependencies point inward.** Transport depends on application; application depends on domain; infrastructure depends on application's ports and on domain. The domain depends on nothing. The application depends on no concrete adapter, only on the port interfaces it owns. In Go this is enforced by package structure and the direction of imports; in Java by package-private constructors and interfaces; in Python by passing dependencies into constructors instead of importing them. The language differs; the arrow does not.

### A note on naming so you don't get lost

The same idea wears three names in the wild: **layered architecture** (the classic top-to-bottom stack), **hexagonal architecture** (Alistair Cockburn's term, also called **ports and adapters**), and **clean / onion architecture** (Robert Martin / Jeffrey Palermo, which add explicit dependency-inversion rules). They are not rivals so much as the same instinct at different resolutions: keep the business logic in the center, keep the I/O at the edges, and make the edges depend on the center rather than the reverse. I will use "ports and adapters" because it is the most operational — it tells you exactly what to build (interfaces and their implementations) — and I will keep the example small enough that the philosophy stays invisible behind working code.

## Ports and adapters, drawn

The word "hexagonal" scares people. It shouldn't, and the hexagon shape is honestly a distraction — Cockburn picked it only because a hexagon has several flat sides to draw adapters on, not because six matters. Here is the whole idea in one figure.

![Driving adapters call inbound ports while the domain core calls outbound ports that driven adapters implement](/imgs/blogs/anatomy-of-a-well-built-microservice-2.webp)

There are two kinds of port and two kinds of adapter, and once you see the symmetry it never confuses you again. **Inbound ports** are the things the service can *do*, declared as interfaces the application layer offers — `PlaceOrder`, `CancelOrder`. **Driving adapters** call those inbound ports: the HTTP handler is a driving adapter, a CLI command would be another, a Kafka consumer that reacts to an event would be a third. They all drive the application the same way, which is why you can add a gRPC interface later without touching the domain. **Outbound ports** are the things the service *needs*, declared as interfaces the application layer requires — `OrderRepository`, `EventPublisher`. **Driven adapters** implement those outbound ports: the Postgres repository, the Kafka publisher. The domain calls *out* through an interface it owns, and the concrete database plugs in from outside.

Here is why this is not academic. Because the application layer depends only on the `OrderRepository` *interface*, your use-case tests can pass in a tiny in-memory map that satisfies that interface, and run in microseconds with no Postgres. Your production code passes in the real Postgres adapter. The use-case can't tell the difference, and doesn't want to. That single property — the ability to swap a real adapter for a fake at the seam — is the entire payoff of the architecture, and it is worth the modest extra boilerplate for any service that will outlive the quarter.

We will wire exactly this for ShopFast. Let's start at the center and work outward, which is also the order in which you should write the code.

## The domain layer: business rules with zero I/O

Start with the thing that makes ShopFast money: an order. Notice what is *absent* from this file. No `import "database/sql"`. No `import "net/http"`. No JSON tags. No logger. The domain knows orders, money, and rules, and nothing else.

```go
// internal/domain/order.go
package domain

import (
	"errors"
	"time"
)

type Status string

const (
	StatusPending   Status = "pending"
	StatusConfirmed Status = "confirmed"
	StatusCancelled Status = "cancelled"
)

var (
	ErrNoLineItems      = errors.New("order must have at least one line item")
	ErrAlreadyCancelled = errors.New("cannot confirm a cancelled order")
	ErrEmptyCustomer    = errors.New("order requires a customer id")
)

type LineItem struct {
	SKU      string
	Quantity int
	UnitCents int64 // money in integer cents, never float
}

type Order struct {
	ID         string
	CustomerID string
	Items      []LineItem
	Status     Status
	CreatedAt  time.Time
}

// NewOrder is the only way to construct a valid Order. It enforces the
// invariants so no caller can build an order that violates the rules.
func NewOrder(id, customerID string, items []LineItem, now time.Time) (*Order, error) {
	if customerID == "" {
		return nil, ErrEmptyCustomer
	}
	if len(items) == 0 {
		return nil, ErrNoLineItems
	}
	return &Order{
		ID:         id,
		CustomerID: customerID,
		Items:      items,
		Status:     StatusPending,
		CreatedAt:  now,
	}, nil
}

// TotalCents is pure business logic — trivially unit-testable, no I/O.
func (o *Order) TotalCents() int64 {
	var sum int64
	for _, it := range o.Items {
		sum += it.UnitCents * int64(it.Quantity)
	}
	return sum
}

// QualifiesForFreeShipping is the kind of rule that must NOT leak into a handler.
func (o *Order) QualifiesForFreeShipping() bool {
	return o.TotalCents() >= 50_00 // orders >= $50.00
}

func (o *Order) Confirm() error {
	if o.Status == StatusCancelled {
		return ErrAlreadyCancelled
	}
	o.Status = StatusConfirmed
	return nil
}
```

Two things to internalize from this file. First, money is in integer cents, never a float — floating-point dollars are a classic production bug that silently overcharges customers fractions of a cent that accumulate into reconciliation nightmares. Second, the constructor `NewOrder` is the *only* way to make an order, and it enforces invariants, so it is impossible anywhere in the codebase to construct an order with zero line items. The domain object protects itself. A handler cannot bypass the rule because the handler never constructs an `Order` by hand — it goes through the constructor. This is the difference between an anemic domain model (a struct of public fields that anyone mutates) and a rich one (a type that guards its own invariants).

Testing this layer is delightful precisely because there is no I/O:

```go
// internal/domain/order_test.go
package domain

import "testing"

func TestFreeShippingThreshold(t *testing.T) {
	o := &Order{Items: []LineItem{{SKU: "BOOK", Quantity: 1, UnitCents: 49_99}}}
	if o.QualifiesForFreeShipping() {
		t.Fatal("$49.99 should NOT qualify for free shipping")
	}
	o.Items[0].UnitCents = 50_00
	if !o.QualifiesForFreeShipping() {
		t.Fatal("$50.00 should qualify for free shipping")
	}
}
```

That test runs in a fraction of a millisecond and needs no container, no network, no fixtures. When the business changes the free-shipping threshold — and it will, for a promotion — the change is one constant and one test, both in the domain. That is what "the rules live in one place" buys you.

## The application layer: use-cases and ports

The use-case is where a single business operation gets orchestrated. The `PlaceOrder` use-case declares the two ports it needs as interfaces, accepts them in its constructor (dependency injection, done with plain function arguments — no framework required), and coordinates the work.

```go
// internal/app/ports.go
package app

import (
	"context"
	"shopfast/orders/internal/domain"
)

// OrderRepository is an OUTBOUND port. The app layer owns this interface;
// the postgres adapter implements it. The app depends on the interface,
// never on the concrete postgres type.
type OrderRepository interface {
	Save(ctx context.Context, o *domain.Order) error
	FindByID(ctx context.Context, id string) (*domain.Order, error)
}

// EventPublisher is an OUTBOUND port for emitting domain events.
type EventPublisher interface {
	Publish(ctx context.Context, topic string, payload []byte) error
}

// IDGenerator and Clock are injected so use-cases are deterministic in tests.
type IDGenerator interface{ NewID() string }
type Clock interface{ Now() time.Time }
```

```go
// internal/app/place_order.go
package app

import (
	"context"
	"encoding/json"

	"shopfast/orders/internal/domain"
)

type PlaceOrderCommand struct {
	CustomerID string
	Items      []domain.LineItem
}

type PlaceOrderResult struct {
	OrderID    string
	TotalCents int64
}

// PlaceOrder is an INBOUND port (a use-case). It depends only on interfaces.
type PlaceOrder struct {
	repo   OrderRepository
	events EventPublisher
	ids    IDGenerator
	clock  Clock
}

func NewPlaceOrder(r OrderRepository, e EventPublisher, ids IDGenerator, c Clock) *PlaceOrder {
	return &PlaceOrder{repo: r, events: e, ids: ids, clock: c}
}

func (uc *PlaceOrder) Handle(ctx context.Context, cmd PlaceOrderCommand) (PlaceOrderResult, error) {
	// 1. Construct a valid domain object (invariants enforced here).
	order, err := domain.NewOrder(uc.ids.NewID(), cmd.CustomerID, cmd.Items, uc.clock.Now())
	if err != nil {
		return PlaceOrderResult{}, err // domain validation error bubbles up
	}

	// 2. Persist through the outbound port.
	if err := uc.repo.Save(ctx, order); err != nil {
		return PlaceOrderResult{}, err
	}

	// 3. Emit a domain event so other ShopFast services can react.
	evt, _ := json.Marshal(map[string]any{
		"order_id":    order.ID,
		"customer_id": order.CustomerID,
		"total_cents": order.TotalCents(),
	})
	// Best-effort publish here for brevity; in production use the outbox
	// pattern so the DB write and the event publish are atomic.
	_ = uc.events.Publish(ctx, "orders.placed", evt)

	return PlaceOrderResult{OrderID: order.ID, TotalCents: order.TotalCents()}, nil
}
```

Notice the comment on the event publish. A naive service publishes the event right after the database write, which means a crash between the two leaves the order saved but the event lost — or the event sent but the transaction rolled back. The correct fix is the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing): write the event to an `outbox` table in the *same* database transaction as the order, then a separate relay publishes it. Because that relay can publish the same event more than once on retry, the consumers must be idempotent — the mechanics of making at-least-once delivery safe with deduplication are covered in [idempotency and deduplication](/blog/software-development/message-queue/idempotency-and-deduplication-making-at-least-once-safe). I'm flagging both here and linking the deep-dives rather than re-deriving them, because keeping the use-case readable matters more than showing the full outbox in this post — and because that is exactly the practitioner's discipline: know where the sharp edge is, cross-link the mechanism, keep moving.

Because `PlaceOrder` depends only on interfaces, its test is pure:

```go
// internal/app/place_order_test.go — uses in-memory fakes, no Postgres.
func TestPlaceOrderRejectsEmptyCart(t *testing.T) {
	uc := NewPlaceOrder(&memRepo{}, &nopPublisher{}, fixedID{"o-1"}, fixedClock{})
	_, err := uc.Handle(context.Background(), PlaceOrderCommand{CustomerID: "c-1", Items: nil})
	if !errors.Is(err, domain.ErrNoLineItems) {
		t.Fatalf("want ErrNoLineItems, got %v", err)
	}
}
```

A junior reads this and asks, "isn't `memRepo` extra code I'll throw away?" No — it is roughly ten lines (a `map[string]*domain.Order` behind the `Save`/`FindByID` methods), and it lets every use-case test run in microseconds with no Docker. That trade is one of the best in software. We will quantify it later when we talk about CI speed.

## The full request flow, end to end

Now stitch it together. A `POST /v1/orders` arrives. The figure shows the path: handler parses and validates, calls the use-case, the use-case persists through the repository, and the result becomes a single `201 Created` response. One job per hop.

![A place order request flowing from handler to use case to repository and back to a created response](/imgs/blogs/anatomy-of-a-well-built-microservice-3.webp)

Here is the transport layer — the driving adapter. Watch how thin it is. It does HTTP things (decode, status codes, content type) and nothing else. It never touches SQL and never decides business rules.

```go
// internal/adapters/http/order_handler.go
package httpadapter

import (
	"encoding/json"
	"errors"
	"net/http"

	"shopfast/orders/internal/app"
	"shopfast/orders/internal/domain"
)

type OrderHandler struct {
	placeOrder *app.PlaceOrder
}

func NewOrderHandler(p *app.PlaceOrder) *OrderHandler {
	return &OrderHandler{placeOrder: p}
}

type placeOrderRequest struct {
	CustomerID string `json:"customer_id"`
	Items []struct {
		SKU       string `json:"sku"`
		Quantity  int    `json:"quantity"`
		UnitCents int64  `json:"unit_cents"`
	} `json:"items"`
}

func (h *OrderHandler) Place(w http.ResponseWriter, r *http.Request) {
	var req placeOrderRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	items := make([]domain.LineItem, 0, len(req.Items))
	for _, it := range req.Items {
		items = append(items, domain.LineItem{SKU: it.SKU, Quantity: it.Quantity, UnitCents: it.UnitCents})
	}

	res, err := h.placeOrder.Handle(r.Context(), app.PlaceOrderCommand{
		CustomerID: req.CustomerID,
		Items:      items,
	})
	switch {
	case errors.Is(err, domain.ErrNoLineItems), errors.Is(err, domain.ErrEmptyCustomer):
		writeError(w, http.StatusUnprocessableEntity, err.Error()) // 422: client sent invalid order
	case err != nil:
		writeError(w, http.StatusInternalServerError, "could not place order") // 500: our fault, hide details
	default:
		writeJSON(w, http.StatusCreated, map[string]any{
			"order_id":    res.OrderID,
			"total_cents": res.TotalCents,
		})
	}
}
```

The error mapping here is a small thing that separates junior from senior code. A *domain* validation error (`ErrNoLineItems`) is the client's fault, so it returns `422 Unprocessable Entity` with a message. An *infrastructure* error (Postgres unreachable) is our fault, so it returns `500` with a generic message and does *not* leak the database error to the caller — leaking internal errors is both a security smell and a support nightmare. The transport layer is the only place that knows about HTTP status codes; the use-case and domain return plain Go errors and stay protocol-agnostic, which is exactly why you could put a gRPC adapter in front of the same use-case tomorrow.

And the driven adapter — the Postgres repository that implements the `OrderRepository` port:

```go
// internal/adapters/postgres/order_repo.go
package postgres

import (
	"context"
	"database/sql"

	"shopfast/orders/internal/domain"
)

type OrderRepo struct{ db *sql.DB }

func NewOrderRepo(db *sql.DB) *OrderRepo { return &OrderRepo{db: db} }

// Save implements app.OrderRepository — the compiler enforces the contract.
func (r *OrderRepo) Save(ctx context.Context, o *domain.Order) error {
	const q = `INSERT INTO orders (id, customer_id, status, total_cents, created_at)
	           VALUES ($1, $2, $3, $4, $5)`
	_, err := r.db.ExecContext(ctx, q, o.ID, o.CustomerID, o.Status, o.TotalCents(), o.CreatedAt)
	return err
}

func (r *OrderRepo) FindByID(ctx context.Context, id string) (*domain.Order, error) {
	const q = `SELECT id, customer_id, status, created_at FROM orders WHERE id = $1`
	row := r.db.QueryRowContext(ctx, q, id)
	var o domain.Order
	if err := row.Scan(&o.ID, &o.CustomerID, &o.Status, &o.CreatedAt); err != nil {
		return nil, err
	}
	return &o, nil
}
```

The repository is the *only* place in the entire service that contains SQL. If ShopFast migrates Orders from Postgres to a managed Aurora, or adds a read replica, the blast radius is this one file. Every `ExecContext` and `QueryRowContext` takes the request's `context.Context`, which is how a client timeout or a deploy-time cancellation propagates all the way down to the database driver and aborts an in-flight query — a detail we will lean on hard when we get to graceful shutdown.

Notice also a rule we will state plainly later: this repository talks *only* to the Orders database. It never `SELECT`s from the Payments service's tables, never joins across service boundaries. Reaching into another service's database is the fastest way to build a distributed monolith, and we will return to why.

## Choosing the architecture: a trade-off you should make on purpose

Ports and adapters is not free. It costs interfaces, constructors, and a little ceremony. For a service that exists to render a single static config blob, that ceremony is pure overhead. So choose deliberately. The three common ways to organize a service trade off along clear axes.

![Trade-off matrix comparing transaction script, layered, and hexagonal organization across five properties](/imgs/blogs/anatomy-of-a-well-built-microservice-4.webp)

| Property | Transaction script | Layered | Hexagonal (ports/adapters) |
|---|---|---|---|
| Boilerplate to start | Lowest — one function per endpoint | Medium | Highest — ports + adapters + DI |
| Unit testability of business rules | Poor (logic tangled with I/O) | OK | Best (pure domain, fake adapters) |
| Boundary clarity | None | Some leakage common | Explicit, compiler-enforced |
| Cost to swap a datastore | Rewrite every handler | Painful, scattered | Swap one adapter |
| Best fit | Trivial CRUD, short-lived tools | Standard CRUD services | Rich rules, long-lived, multi-adapter |

A **transaction script** is the most direct: each endpoint is one function that does everything top to bottom — parse, query, compute, respond. For a service that will live three weeks and has no real business rules, this is correct; reaching for hexagonal there is over-engineering, and a senior calls that out in review just as fast as they call out a handler full of SQL on a service that will live three years. **Layered** architecture (controllers → services → repositories, the default in most Spring and Express codebases) is a fine middle ground and what most teams should reach for by default. **Hexagonal** earns its boilerplate exactly when the service has real domain logic that you want to test in isolation, expects to outlive a datastore choice, or will eventually be driven by more than one transport (HTTP plus an event consumer, say).

My honest default: build the Orders service hexagonal, because orders are the heart of an e-commerce business and will accrete rules for years; build a thin read-only "product-catalog-image-resizer" as a transaction script. The architecture is a tool, not a religion. The mistake juniors make is having no opinion; the mistake mid-levels make is applying hexagonal to everything because they just learned it. Pick on purpose, and be able to say *why* in one sentence.

### Fat handler versus a use-case layer

There is a more granular version of the same decision that you will face on every endpoint: should the logic live in the handler ("fat handler") or in a separate use-case ("thin handler")?

| Concern | Fat handler | Thin handler + use-case |
|---|---|---|
| Lines to add an endpoint | Fewer at first | A few more (the use-case type) |
| Reuse from a second transport | Copy-paste the logic | Call the same use-case |
| Test without an HTTP server | Hard | Trivial |
| Where business bugs hide | In HTTP plumbing | In one named, tested unit |

The fat handler wins for the very first, throwaway endpoint and loses for everything that survives. The moment a second driver appears — a CLI admin tool, a Kafka consumer reacting to an event — the fat handler forces a copy-paste of business logic, and now the rule lives in two places that drift. The use-case layer is the cheap insurance that pays off the first time you add a second way in.

## Config: read it from the environment, validate it at boot

The most boring bug in microservices is also one of the most common: a service that *starts successfully* with a missing or malformed configuration value and then fails on the first request, at 2am, with a nil pointer or a connection to `localhost` in production. The fix is a discipline from the **12-factor app** methodology (more on its origin in the case studies): **store config in the environment**, load it into a typed struct at startup, and *validate it before you accept a single request*. If config is wrong, the process should refuse to start and exit non-zero, so your deploy fails loudly at rollout instead of silently at 2am.

```go
// internal/config/config.go
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	Port            int
	DatabaseURL     string
	KafkaBrokers    string
	ShutdownGrace   time.Duration
	RequestTimeout  time.Duration
	Environment     string // "dev" | "staging" | "prod"
}

// Load reads config from the environment and FAILS LOUDLY if anything required
// is missing or malformed. A misconfigured service must not start.
func Load() (Config, error) {
	cfg := Config{
		Port:           envInt("PORT", 8080),
		DatabaseURL:    os.Getenv("DATABASE_URL"),
		KafkaBrokers:   os.Getenv("KAFKA_BROKERS"),
		ShutdownGrace:  envDuration("SHUTDOWN_GRACE", 25*time.Second),
		RequestTimeout: envDuration("REQUEST_TIMEOUT", 2*time.Second),
		Environment:    envStr("ENVIRONMENT", "dev"),
	}
	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c Config) validate() error {
	if c.DatabaseURL == "" {
		return fmt.Errorf("DATABASE_URL is required")
	}
	if c.KafkaBrokers == "" {
		return fmt.Errorf("KAFKA_BROKERS is required")
	}
	if c.Port < 1 || c.Port > 65535 {
		return fmt.Errorf("PORT %d out of range", c.Port)
	}
	if c.ShutdownGrace < 5*time.Second {
		return fmt.Errorf("SHUTDOWN_GRACE %s too short for a clean drain", c.ShutdownGrace)
	}
	return nil
}

func envStr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
func envDuration(k string, def time.Duration) time.Duration {
	if v := os.Getenv(k); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return def
}
```

Three principles are baked into that loader. **Config comes from the environment, not a checked-in file**, so the same image runs in dev, staging, and prod with only environment differences — that is the 12-factor "dev/prod parity" goal in practice, and it is why you never bake `application-prod.yaml` into the image. **Config is typed and validated**, so `PORT=banana` is caught at boot, not at request time. And **secrets like `DATABASE_URL` are injected at runtime**, never compiled in — the full treatment of secret rotation and vaults is its own topic, covered in the [configuration and secrets management](/blog/software-development/microservices/configuration-and-secrets-management) post; here, the rule is simply that the password is an environment variable the platform supplies, not a string literal in the binary.

A subtle senior move is in there too: the loader exposes `ShutdownGrace` and `RequestTimeout` as config with sane defaults. Timeouts and the shutdown grace period are *operational* knobs you will want to tune per environment without a code change, and putting them in config from day one saves a deploy later. We will use both of these in the next two sections.

## Production table stakes: what every service needs before it ships

Here is the part juniors don't know they don't know. A service that "works on my machine" — handles the happy path, returns the right JSON — is maybe 60% done. The remaining 40% is the operational surface: the parts that let a platform *operate* the service without a human babysitting it. The figure shows that surface for ShopFast Orders. Beyond the business route `/v1/orders`, the kubelet probes `/livez` and `/readyz`, and Prometheus scrapes `/metrics`. These are not optional extras; they are the contract between your service and the platform that runs it.

![Operational surface showing business routes plus livez readyz and metrics endpoints the platform talks to](/imgs/blogs/anatomy-of-a-well-built-microservice-9.webp)

The table stakes, in order of how often I see them missing:

- **Structured JSON logging with correlation/trace IDs.** Logs go to stdout as one JSON object per line, never to a file the service manages. Every log line for a request carries the same `trace_id` so you can grep one request across services. This is the "logs as event streams" 12-factor principle: the service does not route or store its logs; it emits a stream and the platform aggregates it.
- **Health endpoints** — liveness, readiness, and (optionally) startup — that mean *different* things, which is the single most-confused topic for juniors and we will untangle it shortly.
- **Graceful shutdown**: on `SIGTERM`, stop accepting new requests, finish in-flight ones, then exit. Skipping this drops requests on every single deploy.
- **Config validation at boot** — done above.
- **A metrics endpoint** exposing the RED signals (Rate, Errors, Duration) so dashboards and alerts have data.
- **Request timeouts** so one slow dependency cannot pin a request forever and exhaust your connection pool — the gateway to the whole family of [resilience patterns](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads).
- **A clean, versioned API contract** (`/v1/orders`) so you can evolve without breaking callers.

Let's build the three that have real code in them: structured logging with a trace ID, health endpoints, and graceful shutdown.

### Structured logging with a correlation ID

A log line a human reads is a log line a machine can't query. In a fleet of services you will have millions of lines a day, and the only way to make sense of them is structured JSON that a log aggregator indexes. The non-negotiable field is a correlation ID (often the trace ID) that ties every line of one request together, even as that request fans out across services.

```go
// internal/adapters/http/middleware.go
package httpadapter

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/google/uuid"
)

type ctxKey string

const traceIDKey ctxKey = "trace_id"

// WithTracing pulls an incoming trace id or mints one, stamps it on the
// context and on a per-request logger, and logs the RED signals on the way out.
func WithTracing(logger *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		traceID := r.Header.Get("X-Trace-Id")
		if traceID == "" {
			traceID = uuid.NewString()
		}
		reqLog := logger.With(slog.String("trace_id", traceID))
		ctx := context.WithValue(r.Context(), traceIDKey, traceID)

		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(sw, r.WithContext(ctx))

		reqLog.Info("request",
			slog.String("method", r.Method),
			slog.String("path", r.URL.Path),
			slog.Int("status", sw.status),
			slog.Duration("duration", time.Since(start)),
		)
	})
}
```

Configured with Go's standard `slog.NewJSONHandler(os.Stdout, ...)`, every line is a JSON object with `trace_id`, `method`, `path`, `status`, and `duration` — exactly the RED signals, machine-queryable. When ShopFast adds real distributed tracing, this same `trace_id` becomes the OpenTelemetry trace context that propagates across the gateway, the Orders service, and the Payments service, letting you reconstruct a single user action across the whole fleet. That is the subject of the [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry) post; the discipline to start here is small but compounding — a service that never had a `trace_id` field is painful to retrofit.

### Health endpoints: liveness, readiness, and why they differ

This is the topic I most often see juniors get exactly backwards, and getting it backwards causes outages. **Liveness** answers "is this process so broken it should be killed and restarted?" **Readiness** answers "should this instance receive traffic *right now*?" They are different questions with different consequences, and conflating them is dangerous.

```go
// internal/adapters/http/health.go
package httpadapter

import (
	"net/http"
	"sync/atomic"
)

type Health struct {
	ready atomic.Bool // flipped true after deps are up, false on shutdown
	db    interface{ Ping() error }
}

func NewHealth(db interface{ Ping() error }) *Health {
	return &Health{db: db}
}

// Livez: is the process alive? Keep it CHEAP and dependency-free.
// If the DB is down, the FIX is not to restart this process — so livez
// must NOT check the DB, or a DB outage triggers a kill storm.
func (h *Health) Livez(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"alive"}`))
}

// Readyz: should we get traffic? Check dependencies AND the shutdown flag.
func (h *Health) Readyz(w http.ResponseWriter, r *http.Request) {
	if !h.ready.Load() {
		w.WriteHeader(http.StatusServiceUnavailable) // 503 -> remove from rotation
		_, _ = w.Write([]byte(`{"status":"draining"}`))
		return
	}
	if err := h.db.Ping(); err != nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"status":"db_unavailable"}`))
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ready"}`))
}

func (h *Health) SetReady(v bool) { h.ready.Store(v) }
```

The critical rule, learned the hard way by many teams: **liveness must not depend on downstream services.** If `/livez` checks the database and the database has a five-minute blip, Kubernetes will conclude every Orders pod is "dead," kill them all, and restart them — which does nothing to fix the database and turns a recoverable database blip into a total Orders outage as every replica enters a restart loop simultaneously. Liveness checks should be cheap and local: "is the process responsive and not deadlocked?" *Readiness*, by contrast, *should* reflect dependencies and the shutdown state — a pod whose database is unreachable should be pulled from the load-balancer rotation (return `503`) so traffic routes to healthy pods, but it should *not* be killed, because killing won't help. The full taxonomy, including startup probes for slow-booting services and self-healing patterns, is the [health checks, readiness, liveness, and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) post; the rule to carry now is: **liveness = restart me, readiness = route around me.** They are not the same, and treating them the same causes the exact kind of cascading failure you most fear.

## Graceful shutdown: the deploy-time bug that hides in plain sight

Now the centerpiece, because this is the single most common production-grade detail that "working" services skip. Every time ShopFast deploys — and a healthy team deploys many times a day — Kubernetes rolls the pods: it sends `SIGTERM` to an old pod, waits a grace period, then `SIGKILL`s it. If your service hears `SIGTERM` and immediately exits, every request that was *in flight at that instant* dies with it. The client gets a connection reset, sees a `502`, and either retries (extra load) or shows a user an error on a perfectly healthy system — during a routine deploy you initiated.

The fix has two halves that must happen in the right order. First, **fail readiness** so the load balancer stops sending you *new* requests. Then, **drain** the in-flight requests, giving them a bounded grace period to finish, before the server actually closes. Here is the full wiring in `main.go`, which is also where you finally see all the layers assembled.

```go
// cmd/server/main.go
package main

import (
	"context"
	"database/sql"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"shopfast/orders/internal/adapters/http"
	"shopfast/orders/internal/adapters/postgres"
	"shopfast/orders/internal/app"
	"shopfast/orders/internal/config"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))

	// 1. Load + validate config. If it's wrong, exit non-zero NOW.
	cfg, err := config.Load()
	if err != nil {
		logger.Error("config invalid", slog.Any("err", err))
		os.Exit(1)
	}

	// 2. Open infrastructure (driven adapters).
	db, err := sql.Open("pgx", cfg.DatabaseURL)
	if err != nil {
		logger.Error("db open failed", slog.Any("err", err))
		os.Exit(1)
	}
	repo := postgres.NewOrderRepo(db)

	// 3. Wire application layer with concrete adapters (composition root).
	placeOrder := app.NewPlaceOrder(repo, kafkaPublisher(cfg), uuidIDs{}, realClock{})
	orderHandler := httpadapter.NewOrderHandler(placeOrder)
	health := httpadapter.NewHealth(db)

	// 4. Build the router with request timeout + tracing middleware.
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/orders", orderHandler.Place)
	mux.HandleFunc("GET /livez", health.Livez)
	mux.HandleFunc("GET /readyz", health.Readyz)
	// /metrics registered by the prometheus client library.

	handler := httpadapter.WithTracing(logger, mux)
	handler = http.TimeoutHandler(handler, cfg.RequestTimeout, `{"error":"request timeout"}`)

	srv := &http.Server{
		Addr:         ":" + itoa(cfg.Port),
		Handler:      handler,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	// 5. Start serving in the background; mark ready once we're listening.
	go func() {
		health.SetReady(true)
		logger.Info("listening", slog.Int("port", cfg.Port))
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("serve failed", slog.Any("err", err))
			os.Exit(1)
		}
	}()

	// 6. Block until SIGTERM/SIGINT, then drain gracefully.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGTERM, syscall.SIGINT)
	<-stop

	logger.Info("shutdown signal received, draining")
	health.SetReady(false) // (a) fail readiness FIRST -> LB stops new traffic

	// (b) give in-flight requests a bounded window to finish
	ctx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownGrace)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		logger.Error("graceful shutdown timed out, forcing", slog.Any("err", err))
	}
	_ = db.Close()
	logger.Info("shutdown complete, exiting 0")
}
```

The ordering in step 6 is the whole game, and it is subtle enough that I have seen it wrong in production at large companies. `health.SetReady(false)` runs *first*, so `/readyz` starts returning `503`. Kubernetes' readiness probe sees the `503`, and the endpoints controller removes this pod from the Service's endpoints — meaning the load balancer stops routing *new* requests to it. Only *then* does `srv.Shutdown(ctx)` run, which stops accepting new connections but lets the requests already in flight run to completion, up to the `ShutdownGrace` budget. The result: zero dropped requests during a deploy.

There is one more real-world subtlety worth a sentence, because it bites teams constantly. Removing a pod from the endpoints list is *eventually consistent* — it takes a moment to propagate through `kube-proxy` and any external load balancer. So between "readiness flips to `503`" and "the LB actually stops sending traffic," a few requests can still arrive. The standard fix is a short `preStop` sleep (or a brief delay after flipping readiness, before calling `Shutdown`) so the deregistration propagates before you stop accepting. A common value is 5 seconds. The next stress-test section walks through exactly what happens if you skip all of this.

#### Worked example: the cost of NOT having graceful shutdown

Let's make the dropped-requests problem concrete with numbers, because "you might drop some requests" doesn't motivate anyone, but a dollar figure does.

Suppose ShopFast Orders runs **12 replicas**, each serving **80 requests per second** (so ~960 RPS total), with a mean request duration of **120 ms**. By Little's Law, the average number of in-flight requests per replica is roughly `arrival rate × duration = 80 × 0.120 = 9.6`, so call it **~10 in-flight requests per replica** at any instant.

Now deploy. A rolling deploy replaces all 12 pods. With a naive shutdown (`SIGTERM` → immediate `os.Exit`), each pod drops its ~10 in-flight requests the moment it is terminated: `12 replicas × 10 in-flight ≈ 120 dropped requests per deploy`. If ShopFast deploys **8 times a day**, that is `120 × 8 = 960 dropped requests per day`, or about **350,000 per year** — every one of them a customer who saw an error or a checkout that retried. If even 1% of those dropped requests were checkout submissions at an average order value of \$45, and a fraction abandon on the error, the lost-revenue and support-cost story writes itself, and it is entirely self-inflicted by routine deploys.

With graceful shutdown plus readiness gating, the number is **zero**. Readiness flips first, the LB stops sending new requests, the ~10 in-flight finish within the grace window, and the pod exits clean. The fix is about forty lines of `main.go` and a `terminationGracePeriodSeconds` in the pod spec. The payback is the highest-leverage forty lines you will write all quarter: it converts every deploy from a tiny self-inflicted outage into a non-event.

## Stress test: SIGTERM mid-request during a rolling deploy

Let's reason about this like an incident, step by step, because the *order* of events is where the design earns its keep. The timeline figure traces it.

![Timeline of a SIGTERM during a rolling deploy from signal to readiness flip to drain to a clean exit](/imgs/blogs/anatomy-of-a-well-built-microservice-6.webp)

At **T+0**, Kubernetes decides to replace an old Orders pod. Two things happen nearly simultaneously: the pod is marked for termination (which begins removing it from the Service endpoints) and the container receives `SIGTERM`. At **T+0**, our signal handler immediately flips `health.SetReady(false)`, so `/readyz` returns `503`. By **T+1s**, the endpoints controller and `kube-proxy` have removed the pod from rotation; the external load balancer follows shortly after (this is the eventually-consistent part, which the `preStop` delay covers). New requests now route only to healthy pods. At **T+2s**, `srv.Shutdown(ctx)` is draining: it has stopped accepting new connections, and the ~10 requests that were in flight are running to completion. By **T+8s**, the last in-flight request has returned its response, `Shutdown` returns nil, the database connection closes, and the process exits `0` — well inside the grace period. Zero requests dropped.

Now run the naive version to feel the contrast. Without readiness gating, the pod keeps reporting healthy even as it dies, so the load balancer happily routes *new* requests to a pod that is about to be killed — those land in the void. Without `srv.Shutdown`, the in-flight requests are severed mid-write. The before/after figure shows the difference in outcomes.

![Before and after comparison of a naive instant shutdown versus a gated drain showing dropped requests versus zero errors](/imgs/blogs/anatomy-of-a-well-built-microservice-7.webp)

What breaks at the edges, and how the design holds:

- **A request takes longer than the grace period.** If a request needs 30 seconds but `ShutdownGrace` is 25, `srv.Shutdown` returns a deadline-exceeded error and we force-close. The fix is to *bound* request duration with the `RequestTimeout` we put in config (2s here) so no request can outlive the grace window — and to set `terminationGracePeriodSeconds` in the pod spec comfortably larger than `ShutdownGrace`, so Kubernetes does not `SIGKILL` you mid-drain. The numbers must nest: `requestTimeout < shutdownGrace < terminationGracePeriodSeconds`.
- **A downstream dependency is slow during the drain.** Because every repository call carries the request `context`, when the request times out the context is cancelled and the database query is aborted, freeing the connection. A request can't pin a connection forever, so the pool doesn't exhaust during shutdown.
- **The whole fleet deploys at once (10× scenario).** If all 12 pods drained simultaneously you'd briefly halve capacity. That is why rolling deploys replace pods in batches (`maxUnavailable: 25%`) and why readiness gating matters even more at scale: the surviving pods must be the only ones taking traffic while the batch drains. The deployment-strategy mechanics — surge, max-unavailable, canary — are the [deployment strategies](/blog/software-development/microservices/deployment-strategies-blue-green-canary-feature-flags) post; the service-side contract is simply "report readiness honestly and drain on `SIGTERM`," and if every service honors it, rolling deploys are invisible to users.

The lesson a senior internalizes: **the service's job during a deploy is to tell the truth about its readiness and to drain politely.** The platform handles the routing; the service handles the honesty. That division of labor is the heart of why 12-factor's "disposability" principle — fast startup, graceful shutdown — is not a nicety but a load-bearing property of a fleet that deploys constantly.

#### Worked example: cold-start budget and why autoscaling cares

The flip side of graceful shutdown is graceful *startup* — how fast a fresh replica becomes ready. This matters more than juniors expect, because it directly governs how fast you can autoscale into a traffic spike and how fast a rolling deploy completes.

Suppose ShopFast Orders gets a flash-sale spike: traffic triples in 90 seconds. The Horizontal Pod Autoscaler reacts by scheduling new pods. The question is: how quickly can a new pod start *absorbing* load? Walk the budget:

- **Image pull**: with a tiny image (we'll get there — ~18MB), and the layer likely already cached on the node, ~1–3s. With a fat 1.2GB image not yet on the node, this can be 30–60s. **Image size is a cold-start lever.**
- **Process start + config load + DB connection pool warm-up**: a compiled Go binary starts in well under a second; opening the pool and a first `Ping` adds maybe 200–800ms. A JVM service with a large classpath might spend 5–15s here before it's ready — a real difference at autoscale time.
- **Readiness gate**: the pod must report `/readyz` healthy before it gets traffic. If your readiness check waits for the DB ping (good) but also warms a 30-second cache, you've added 30s of cold start. **Keep readiness fast; do expensive warm-up lazily or in the background where you can.**

If your total cold-start budget is ~3 seconds, the autoscaler can field the spike before customers feel it. If it's 60 seconds (fat image, slow JVM warm-up, eager cache priming), the spike is over before the new capacity arrives, and your existing pods shed load or fall over in the meantime. This is precisely why 12-factor lists fast startup under "disposability," why we build a small image, and why a compiled, statically-linked service has a real operational advantage at the autoscaling margin. Measure it: log a `ready_after_ms` metric at the moment readiness flips true, alert if p99 cold-start drifts past your budget, and you've turned a vague "feels slow to scale" into a number you can defend in a capacity review.

## Request timeouts: the one default that saves you at 3am

There is a default in almost every HTTP framework that is wrong for a microservice, and it has caused more cascading outages than any flashy distributed-systems failure: **the default request timeout is "wait forever."** A request that waits forever holds a goroutine (or a thread), a connection-pool slot, and possibly an open database connection for as long as the slowest downstream takes to *not* respond. When a dependency goes slow — not down, just slow — every request piles up waiting, your finite resources fill with stalled requests, and a healthy service becomes unavailable because it ran out of capacity to *wait*. That is the classic resource-exhaustion cascade, and the entry fee to avoiding it is a single timeout you set on purpose.

We already wired the outer timeout in `main.go` with `http.TimeoutHandler(handler, cfg.RequestTimeout, ...)`, which caps the *total* time any request can occupy the server before the client gets a `503` and the handler is told to give up. But a server-side timeout is only half the picture. The deeper rule is that *every* outbound call from your service must itself be bounded, and the budget for those calls must fit inside the request's budget. Here is what that discipline looks like when the Orders service has to call the Inventory service to reserve stock:

```go
// internal/adapters/inventory/client.go — every outbound call is bounded.
func (c *InventoryClient) Reserve(ctx context.Context, sku string, qty int) error {
	// Derive a per-call budget from the request context. If the request has
	// 2s total and we've already spent 600ms, this call gets the remainder,
	// capped at our own 500ms ceiling — whichever is smaller.
	callCtx, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()

	req, _ := http.NewRequestWithContext(callCtx, http.MethodPost, c.baseURL+"/v1/reserve", body)
	resp, err := c.httpClient.Do(req) // httpClient also has a Transport-level timeout
	if err != nil {
		return fmt.Errorf("inventory reserve: %w", err) // ctx deadline shows up here
	}
	defer resp.Body.Close()
	// ...
	return nil
}
```

Two things make this safe. First, `context.WithTimeout(ctx, ...)` derives the call deadline *from the request's deadline*, so a timeout (or a client disconnect, or a shutdown) anywhere up the stack cancels everything below it — the database query, the inventory call — and frees the resources immediately. This is why we threaded `r.Context()` all the way down to `ExecContext` earlier: it is the single mechanism that makes cancellation propagate end to end. Second, the per-call ceiling (500ms) is *smaller* than the request budget (2s), so the budget nests: the request can absorb a slow inventory call and still return a clean error to the client rather than hanging. A junior sets one global timeout and thinks they're done; a senior makes the budgets nest — `db call < inventory call < request timeout < shutdown grace < termination grace` — so that no layer can outlive the one above it. Getting that nesting right is the difference between "one slow dependency degrades one feature" and "one slow dependency takes down the whole service." The full family of defenses around a remote call — retries with jitter, circuit breakers, bulkheads — builds directly on this timeout foundation and is the subject of the [resilience patterns](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) post; the table-stakes rule for *this* service is simply: never make an unbounded call, and make the budgets nest.

## The metrics endpoint: the RED signals, exposed

The last table stake with real code is the metrics endpoint. You cannot operate what you cannot measure, and the minimum viable measurement for a request-serving service is the **RED signals**: the **R**ate of requests, the **E**rror rate, and the **D**uration distribution. Expose those on `/metrics` in Prometheus format and you have the raw material for every dashboard, every SLO, and every alert you will ever want. The instrumentation is a thin middleware that wraps the same handler chain as tracing:

```go
// internal/adapters/http/metrics.go — RED signals via Prometheus.
package httpadapter

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	reqTotal = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "http_requests_total",
		Help: "Total HTTP requests by route, method, status.",
	}, []string{"route", "method", "status"})

	reqDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "http_request_duration_seconds",
		Help:    "Request latency distribution.",
		Buckets: []float64{.005, .01, .025, .05, .1, .25, .5, 1, 2.5}, // for p50/p99
	}, []string{"route", "method"})
)

func WithMetrics(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, status: 200}
		next.ServeHTTP(sw, r)

		route := r.Pattern // Go 1.22 routing exposes the matched pattern
		reqTotal.WithLabelValues(route, r.Method, strconv.Itoa(sw.status)).Inc()
		reqDuration.WithLabelValues(route, r.Method).Observe(time.Since(start).Seconds())
	})
}
```

The histogram buckets are not arbitrary — they are chosen so that Prometheus can compute a meaningful `p99` for a service whose latencies live in the tens-to-hundreds-of-milliseconds range. The cardinality discipline matters too: I label by `route` (the matched *pattern*, `/v1/orders`, not the actual URL with its order ID interpolated) and by `status`, but never by anything unbounded like a customer ID or a full path — high-cardinality labels are the most common way teams blow up their metrics storage bill and take down their own monitoring. With these two metrics you can graph rate (`rate(http_requests_total[1m])`), error rate (`rate(http_requests_total{status=~"5.."}[1m])`), and p99 latency (`histogram_quantile(0.99, ...)`) — the three numbers that tell you whether the service is healthy without reading a single log line. Turning those signals into SLOs and alerts that don't page you for noise is the [SLOs, golden signals, and alerting](/blog/software-development/microservices/slos-golden-signals-and-alerting-for-microservices) topic; the table stake here is simply to *expose* them, because retrofitting metrics into a service that never had them is far more painful than building the four-line middleware on day one.

## The folder layout: structure that mirrors architecture

Now that the pieces exist, here is where they live. A newcomer should be able to guess the location of any file from the folder names alone, because the folders mirror the layers. The figure shows the tree.

![Tree of the Orders service folder layout grouping code by domain application adapters and entrypoint](/imgs/blogs/anatomy-of-a-well-built-microservice-5.webp)

```bash
orders-svc/
├── cmd/
│   └── server/
│       └── main.go            # composition root: wire adapters, start, drain
├── internal/
│   ├── domain/                # business rules, ZERO I/O imports
│   │   ├── order.go
│   │   └── order_test.go
│   ├── app/                   # use-cases + ports (interfaces)
│   │   ├── ports.go           # OrderRepository, EventPublisher (outbound)
│   │   ├── place_order.go     # use-case (inbound port)
│   │   └── place_order_test.go
│   ├── adapters/              # the edges
│   │   ├── http/              # driving adapter: handlers, middleware, health
│   │   │   ├── order_handler.go
│   │   │   ├── middleware.go
│   │   │   └── health.go
│   │   └── postgres/          # driven adapter: the ONLY SQL in the service
│   │       └── order_repo.go
│   └── config/
│       └── config.go          # env loading + boot-time validation
├── migrations/                # SQL schema migrations for THIS service's DB
│   └── 0001_create_orders.sql
├── Dockerfile                 # multi-stage, non-root, tiny
├── go.mod
└── README.md                  # how to run, env vars, ports
```

A few conventions worth calling out. The `internal/` directory is a Go feature that makes everything under it un-importable by other modules — a compiler-enforced wall that stops another service from reaching into your domain types, which is exactly the kind of accidental coupling we want to prevent. The `cmd/server/main.go` is the **composition root**: the one and only place where concrete adapters are constructed and wired into use-cases. Everywhere else, code depends on interfaces. The `migrations/` directory holds this service's own schema — and *only* this service's schema, because in microservices each service owns its data store ([database per service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices) is the rule that defines the style). If you find yourself wanting to put another service's tables in here, stop: that is the distributed-monolith trap opening under your feet.

This layout is idiomatic Go, but the *shape* translates directly. In a Python/FastAPI service you'd have `app/domain/`, `app/use_cases/`, `app/adapters/http/` and `app/adapters/db/`, with FastAPI routers as driving adapters and SQLAlchemy repositories as driven adapters, dependencies injected via FastAPI's `Depends`. In Java/Spring you'd have `domain`, `application`, and `infrastructure` packages with Spring wiring the beans. The language and framework change; the four layers and the inward-pointing arrow do not.

## The Dockerfile, done right

A service ships as a container image. A bad Dockerfile produces a 1.2GB image running as root with a compiler and a shell inside it — slow to pull (hurting cold start, as we saw), and a fat attack surface. A good one is multi-stage, non-root, tiny, and starts fast. The figure shows the two stages.

![Stack showing a multi-stage Docker build compiling in a fat stage then shipping a tiny non-root distroless image](/imgs/blogs/anatomy-of-a-well-built-microservice-8.webp)

```dockerfile
# ---- build stage: has the compiler and the full toolchain ----
FROM golang:1.22 AS build
WORKDIR /src

# Copy go.mod/go.sum first so the dependency layer caches independently of code.
COPY go.mod go.sum ./
RUN go mod download

COPY . .
# Static, stripped binary — no libc dependency, so it runs on a scratch/distroless base.
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
    -o /bin/orders ./cmd/server

# ---- runtime stage: tiny, no shell, no compiler, non-root ----
FROM gcr.io/distroless/static-debian12:nonroot AS runtime

# distroless 'nonroot' already runs as uid 65532; declare it explicitly anyway.
USER nonroot:nonroot

COPY --from=build /bin/orders /orders

EXPOSE 8080
ENTRYPOINT ["/orders"]
```

Every line of that runtime stage is a deliberate decision a senior will check in review:

- **Multi-stage build.** The `golang:1.22` stage has the compiler and is ~800MB; none of it ships. Only the compiled binary is copied into the runtime stage. The final image is the binary plus a minimal base — roughly **18MB**.
- **Distroless base.** `gcr.io/distroless/static` contains no shell, no package manager, no `curl`, almost nothing. There is nothing for an attacker who lands a remote-code-execution to pivot with — no `sh` to spawn, no `apt` to install tools. It also can't run the classic `HEALTHCHECK CMD curl ...` because there is no `curl`, which is fine and actually correct in Kubernetes: the *kubelet* probes `/livez` and `/readyz` over HTTP, so the container doesn't need a self-`HEALTHCHECK`. (On plain Docker without an orchestrator, add a tiny Go-based healthcheck binary or use a base that has a probe tool.)
- **Non-root user.** Running as `uid 65532` (not root) means a container breakout has a far smaller blast radius, and it satisfies the `runAsNonRoot: true` Pod Security Standard that any serious cluster enforces.
- **Layer caching by copying `go.mod` first.** Dependencies change rarely; code changes constantly. Copying and downloading dependencies in a separate, earlier layer means a code-only change reuses the cached dependency layer, cutting build times from minutes to seconds — which compounds across hundreds of CI runs a day.
- **`CGO_ENABLED=0` + static binary.** No dynamic libc linkage means the binary runs on the `static` distroless base with nothing else needed, and there's no glibc-version surprise across base images.

The full set of container best practices — image scanning, SBOMs, base-image pinning by digest, distroless trade-offs, build caching across CI — is the [containerizing microservices](/blog/software-development/microservices/containerizing-microservices-docker-best-practices) post. The takeaway here: a service image should be small, non-root, shell-free, and fast to start, and that is achievable in a dozen lines.

## What NOT to put in a service

A clean internal shape is half the job; the other half is discipline about what crosses the boundary. Two anti-patterns recreate the monolith one innocent decision at a time, and both feel helpful when you commit them.

**Do not reach into another service's database.** It is tempting: the Orders service needs the customer's email for a confirmation, the Customers data is right there in Postgres, so why not `SELECT email FROM customers.users`? Because the instant you do, the Customers team can never change their schema without breaking you, you've coupled the two services' deploys, and you've created an invisible dependency no diagram shows. The Orders service must ask the Customers *service* through its API (or consume an event), never touch its tables. Database-per-service is not bureaucracy; it is the boundary that lets the two teams move independently, and violating it is the textbook on-ramp to a distributed monolith — services that are physically separate but logically fused. The data-coupling failure modes get their own treatment in [shared data anti-patterns and the distributed monolith](/blog/software-development/microservices/shared-data-anti-patterns-and-the-distributed-monolith). The flip side is that once each service owns its own data, a business operation that spans services can no longer be a single database transaction — you trade ACID across the boundary for a [saga](/blog/software-development/database/saga-pattern-distributed-transactions) and for the [eventual consistency](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) it implies, which is a real cost you accept knowingly, not a detail to discover in production.

**Do not put mutable business logic in a shared library.** Cross-cutting *technical* utilities — a logging setup, a tracing helper, a typed config loader, an HTTP client with sane defaults — are fine and good to share, because they change slowly and carry no domain meaning. But the moment you put *business* logic in a shared `commons` library — say, the order-pricing rules, so Orders and the Billing service can both "reuse" them — you have fused those services at the version level. Now a pricing change forces both services to bump the library version and redeploy together, in lockstep, and you've reinvented the monolith's worst property (coordinated deploys) with the added latency of network calls in between. The rule: **share technical plumbing freely; never share mutable domain logic.** If two services genuinely need the same business rule, that is a signal the rule belongs to a *third* service that owns it, exposed behind an API — not a library copied into both.

A third, quieter rule: **keep the service stateless.** No in-memory session that the next request to a different replica won't have, no on-disk scratch state the platform will wipe on restart. State lives in the database, the cache, or the queue — backing services the 12-factor methodology treats as attached resources. A stateless process is one any replica can serve any request, which is what makes horizontal scaling and disposability work at all. If your service *must* hold state, that's a design conversation, not a default.

It is worth making the cost of the shared-library trap concrete, because it is the one that feels most virtuous while you commit it — sharing code is what good engineers do, right? Suppose ShopFast extracts the order-pricing rules into a `shopfast-pricing` library, imported by both Orders and Billing, so the discount logic lives in "one place." It works beautifully for a month. Then marketing wants a new promotion, which is a pricing change. You bump the library to `v2.0`, and now you must redeploy *both* Orders and Billing — coordinated, in the right order, with a compatibility window where one is on `v1` and the other on `v2`. If the change touches a shared type's shape, you cannot deploy them independently at all; you are back to the monolith's defining pain (lockstep releases) with the monolith's defining benefit (a single in-process call) replaced by a network hop. Measure the regression: a team that could deploy Orders 8 times a day independently now deploys both services together perhaps twice a day, after a coordination meeting. You have *negated the entire reason you built microservices* and paid network latency for the privilege. The fix, again, is to ask whether the pricing rules are really a *capability* that deserves its own owning service behind an API — in which case Orders and Billing both call it, and it can evolve behind a versioned contract — or whether the duplication of a small, stable rule in two services is genuinely cheaper than the coupling. Often a little duplication of stable logic beats a lot of coupling, which is a deeply unintuitive lesson for someone trained to never repeat themselves.

## Case studies: where these patterns came from

These aren't ideas I invented; they were forged by teams running services at a scale that made the lessons unavoidable. Three are worth knowing accurately.

**Heroku and the origin of 12-factor.** The twelve-factor methodology was published in 2011 by Adam Wiggins and colleagues at Heroku, the platform-as-a-service company. It distilled patterns they observed across thousands of apps deployed on their platform into twelve principles — config in the environment, treat backing services as attached resources, build/release/run separation, stateless processes, port binding, concurrency via the process model, disposability with fast startup and graceful shutdown, dev/prod parity, logs as event streams, and admin processes as one-offs. Not all twelve are equally load-bearing in 2026 — the methodology predates Kubernetes and containers — but the ones that survived and *matter most for services* are exactly the ones we built in this post: config in env, stateless processes, disposability (graceful shutdown + fast startup), logs as streams, and dev/prod parity. When someone says "make it 12-factor," those are the five that earn their keep.

**Netflix and the "paved road."** Netflix runs thousands of microservices and learned early that letting every team invent service scaffolding from scratch produced chaos and reinvented bugs. Their answer was a *service template* and internal tooling (the era of Eureka for discovery, Ribbon for client-side load balancing, Hystrix for circuit breaking — later largely superseded by service-mesh and platform features) that baked the table-stakes — health checks, metrics, resilient clients, logging — into a generated skeleton. A new Netflix service started life already production-grade because the template made the right thing the default. The lesson for any growing org: the patterns in this post should eventually become a *generated template* (`shopfast new-service orders`), not a tribal-knowledge checklist a senior re-explains in every code review. The paved road is faster *because* it removes choices that don't differentiate you.

**Spotify and the golden path.** Spotify popularized the term "golden path" (sometimes "paved path") and built tooling — most visibly Backstage, their developer portal, later open-sourced and now a CNCF project — to make the well-supported way to build a service the *easy* way. A developer scaffolds a new service from a template that already has CI, observability, health endpoints, and deployment wired, and Backstage tracks the service's ownership, docs, and operational maturity. The insight is organizational: you don't get a fleet of well-built services by writing a wiki page; you get it by making the production-grade default the path of least resistance. Conway's Law cuts both ways — your tooling shapes your services as much as your org chart does (the [Conway's Law and team topologies](/blog/software-development/microservices/conways-law-and-team-topologies-for-microservices) post goes deep on this).

The thread connecting all three: the internal anatomy we built is not meant to be retyped per service forever. At small scale you copy it by hand and review it carefully. At medium scale you template it. At large scale you make it a paved road with tooling that makes the right thing automatic — and that investment, not heroics, is what lets an org run hundreds of services without each one being an artisanal snowflake.

## When to reach for this (and when not to)

Build a service to this full anatomy — hexagonal layers, all the operational table stakes, the multi-stage Dockerfile — when the service is **load-bearing and long-lived**: it holds real business logic, multiple people will work on it over years, it will be deployed constantly, and it sits on a critical path where a dropped request is a lost customer. ShopFast Orders is exactly that service. So is Payments, Inventory, and Checkout.

Dial it down when the cost outweighs the payoff. A **throwaway internal tool** that one person runs occasionally does not need hexagonal layers or a graceful-shutdown dance — a transaction-script handler and a basic Dockerfile are honest engineering, and gold-plating it is just a different kind of waste. A **batch job** that runs to completion and exits doesn't need readiness probes the way a long-running server does, though it still benefits from structured logs and config validation. And if you are still at the stage where the [monolith is the right call](/blog/software-development/microservices/monolith-first-and-the-modular-monolith) — small team, unclear boundaries, pre-product-market-fit — then you have *one* deploy unit, and most of this post collapses into "structure your modules well and you'll thank yourself when you do split." The layered/hexagonal discipline inside a modular monolith is exactly what makes a future extraction cheap.

The senior judgment is not "always do all of it" or "it's overkill"; it's calibrating the anatomy to the service's stakes and lifespan, and being able to say which corners you cut and why. The operational table stakes — health endpoints, graceful shutdown, structured logs, config validation — are the part I almost never let teams skip, because they're cheap and the failure modes (kill storms, dropped-request deploys, silent misconfig) are nasty and recurring. The architectural ceremony (full ports and adapters) is the part that scales with stakes. Cut ceremony for trivial services; never cut the operational table stakes for anything that takes production traffic.

## Key takeaways

- **Dependencies point inward.** The domain imports nobody; the application depends only on ports (interfaces); adapters depend on the application. Get this one arrow right and testability, changeability, and boundary clarity follow.
- **Business rules live in the domain, never in a handler or in SQL.** A handler that contains `if order.total > 500` has leaked, and the rule that lives in two places is the rule that drifts and breaks.
- **Validate config at boot and exit non-zero if it's wrong.** A service that starts misconfigured fails at 2am on the first request; a service that refuses to start fails loudly at deploy time, where you want it.
- **Liveness means "restart me," readiness means "route around me."** Never let liveness depend on a downstream service, or a dependency blip becomes a self-inflicted kill storm across every replica.
- **Graceful shutdown is not optional.** Fail readiness first, then drain in-flight requests within a bounded grace window. The ordering converts every deploy from a tiny outage into a non-event — about forty lines of `main.go` for zero dropped requests.
- **Cold-start budget is an autoscaling property.** A tiny image and a fast-starting process let you absorb a spike before customers feel it; a fat image and slow warm-up mean capacity arrives after the spike is over. Measure `ready_after_ms`.
- **The Dockerfile is part of the service.** Multi-stage, distroless, non-root, statically linked, dependency layer cached. Small, shell-free, fast to pull.
- **Don't recreate the monolith.** Never reach into another service's database; never share *mutable business logic* in a common library (share technical plumbing only); keep the process stateless.
- **The anatomy should become a paved road.** Copy it by hand at small scale, template it at medium scale, make it automatic tooling at large scale — that investment is how orgs run hundreds of services without each being a snowflake.

## Further reading

- Sam Newman, *Building Microservices* (2nd ed.) — the canonical practitioner's book on service design, boundaries, and operability.
- Chris Richardson, *Microservices Patterns* — the pattern catalog (hexagonal, saga, API composition) with concrete code.
- Alistair Cockburn, "Hexagonal Architecture" (the original ports-and-adapters write-up) — short, foundational, worth reading in the source.
- Adam Wiggins et al., *The Twelve-Factor App* (12factor.net) — the origin of config-in-env, disposability, and logs-as-streams.
- *Production-Ready Microservices* by Susan Fowler — a checklist-driven take on the operational table stakes, drawn from running services at Uber.
- Cross-links in this series: [database per service](/blog/software-development/microservices/database-per-service-the-rule-that-defines-microservices), [health checks, readiness, liveness, and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing), [containerizing microservices](/blog/software-development/microservices/containerizing-microservices-docker-best-practices), [distributed tracing and observability with OpenTelemetry](/blog/software-development/microservices/distributed-tracing-and-observability-with-opentelemetry), [inter-service communication fundamentals and fallacies](/blog/software-development/microservices/inter-service-communication-fundamentals-and-fallacies), and the [transactional outbox](/blog/software-development/microservices/the-transactional-outbox-and-reliable-event-publishing).
