---
title: "Design a Ride-Hailing System: Geospatial Matching at Uber Scale"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Design the geospatial heart of a ride-hailing system: choose H3 over geohash, absorb a million location pings per second without a database, match riders to drivers without double-dispatch, and survive a New Year's Eve surge."
tags:
  [
    "system-design",
    "ride-hailing",
    "geospatial",
    "h3",
    "geohash",
    "matching",
    "architecture",
    "distributed-systems",
    "scalability",
    "optimization",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/design-a-ride-hailing-system-1.webp"
---

The interviewer says "design Uber" and most candidates immediately start drawing boxes for the rider app, the driver app, a load balancer, and a database, and then they get stuck, because they have drawn the easy 5% and skipped the part that actually makes ride-hailing hard. Ride-hailing is not a CRUD app with a map on top. It is, at its core, a real-time geospatial matching engine running under a write load that would melt a naively designed database in minutes, and the entire art of the system is keeping that firehose off your disks while still answering "which drivers are near this rider, right now" in a few milliseconds. If you do not get the geospatial index and the location-write problem right, nothing else matters, because the system falls over before the first ride completes.

Let me be precise about what makes this problem distinct from the other "design X" posts in this series. A [URL shortener](/blog/software-development/system-design/design-a-url-shortener) is a read-heavy key-value problem; a [news feed](/blog/software-development/system-design/design-a-news-feed-and-timeline) is a fan-out problem; a [chat system](/blog/software-development/system-design/design-a-chat-and-messaging-system) is a persistent-connection routing problem. Ride-hailing is none of those. Its defining characteristic is that the dominant operation — by two orders of magnitude — is a *continuous stream of location updates from millions of moving drivers*, and the dominant query is a *spatial proximity search* over that constantly-changing set of points. Everything else in the system — trips, payments, ratings, ETAs — is comparatively boring distributed-systems work you already know how to do. The geospatial layer is where seniors earn their title.

By the end of this post you should be able to do five concrete things. First, pick a spatial index — geohash, quadtree, S2, or H3 hexagons — by reasoning from the query you run most, and defend why Uber landed on H3. Second, design the location-update path so that a million pings per second never touch a transactional database, using an in-memory geo-index sharded by geography. Third, build a matching engine that ranks candidates by real ETA and locks a driver atomically so two riders never claim the same car. Fourth, draw the trip state machine and reason about where you need strong consistency (trip state, payment) versus where eventual is fine (driver location). Fifth, stress-test the whole thing against a New Year's Eve surge, a single hot region, and a dispatch race, and explain exactly what breaks and how you contain it. Figure 1 is the skeleton we will spend the rest of the post filling in and defending.

![A high-level architecture diagram showing rider and driver apps connecting through an API gateway to a location service, matching service, and trip service backed by a strongly consistent store](/imgs/blogs/design-a-ride-hailing-system-1.webp)

The shape to notice in figure 1 is that the location path and the trip path diverge almost immediately and never really rejoin. Driver pings flow into a location service backed by an in-memory geo-index; trip and payment events flow into a strongly consistent store. These two halves of the system have *opposite* requirements — one wants to drop data cheaply and never block, the other must never lose a cent — and the single biggest architectural mistake you can make is forcing them through the same datastore. Keep that split in your head; it is the spine of every decision that follows.

## 1. Frame the problem and scope it ruthlessly

Before any boxes, a senior pins down scope, because "design Uber" is a multi-year, thousand-engineer effort and you have forty-five minutes. The discipline of [turning a vague ask into requirements](/blog/software-development/system-design/turning-vague-asks-into-requirements-and-slos) applies directly: name what is in, what is out, and what the system must guarantee. I will scope this to the *core ride-hailing loop*: a rider requests a ride from a pickup location, the system finds nearby available drivers, matches one, the driver navigates to pickup, the trip runs, and payment settles. That is the spine.

In scope, and where the real engineering lives: the geospatial index for "find nearby drivers," the high-frequency driver-location write path, the matching and dispatch algorithm, real-time location streaming back to the rider so they watch the car approach, ETA and routing over a road graph, surge pricing as a demand signal, and the trip state machine with its consistency requirements. Out of scope for this post, because they are either standard or enormous on their own: the payments rails (we forward to [design a payment system](/blog/software-development/system-design/design-a-payment-system) for the money mechanics), driver onboarding and KYC, the rating and review subsystem, fraud detection, the data-warehouse and analytics pipeline, and pooled rides (UberPool-style shared trips), which are a genuinely harder matching problem layered on top of this one. I will mention pooling as an extension but not design it.

Now the functional requirements, stated as capabilities the system must have. Riders must be able to request a ride and see nearby drivers and an ETA before they commit. The system must match a rider to exactly one suitable driver within a few seconds. The matched driver must navigate to the rider, and the rider must watch the driver's live position update on a map. The trip must transition through a well-defined lifecycle and end with an accurate fare and a payment. Drivers must continuously report their location and availability. Surge pricing must adjust fares when demand outstrips supply in an area.

The non-functional requirements are where ride-hailing gets its character, so be specific. *Latency*: a nearby-driver query must return in well under a hundred milliseconds, because it sits on the rider's interactive request path; matching should complete within a couple of seconds end to end; location pings should be ingested with a p99 under a few milliseconds because there are so many of them. *Availability*: matching and location must be highly available — a rider who cannot get a car churns to a competitor immediately — but a brief inability to update a profile photo is fine. *Consistency*: and here is the crucial split that figure 1 hinted at — driver location can be *eventually consistent* and even lossy (a ping dropped means the dot is stale by four seconds, which nobody notices), but trip state and payment must be *strongly consistent* (you cannot dispatch the same driver to two riders, and you cannot charge a card twice). *Scale*: design for a few million concurrently-online drivers globally, tens of millions of rides per day, and the location-update firehose that those drivers generate. We will put real numbers on all of this in section 2.

One scoping decision deserves its own sentence because juniors miss it: ride-hailing is intrinsically *geo-partitioned*. A rider in New York is never matched with a driver in Tokyo, so the system naturally shards by geography, and almost every scaling problem — the location index, the matching engine, the surge calculation — partitions cleanly along city or region lines. This is a gift. Lean on it relentlessly; it is the reason a global system can be built out of thousands of mostly-independent regional systems that rarely talk to each other.

## 2. Estimate the load before you design (the firehose is the whole story)

You cannot design this system without doing the back-of-the-envelope math first, because the numbers do not just inform the design, they *dominate* it. One number — the location-update rate — is so much larger than everything else that it dictates the entire architecture. If you skip this step you will build the wrong thing. The general technique here is the one from [back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design); I will apply it concretely.

Start from drivers online. Take three million drivers online at a typical busy moment globally (Uber's order of magnitude). Each driver's app reports location on a fixed cadence — call it once every four seconds while online, which is frequent enough that the rider sees smooth movement and the matcher has fresh positions. That single assumption produces the number that defines the system:

```
3,000,000 drivers / 4 seconds = 750,000 location writes per second
```

Three quarters of a million writes per second, sustained, every second of every day, just for location. Now compare that to the operations everyone thinks of as the "main" workload. Tens of millions of rides per day spread over a day is on the order of a few thousand ride requests per second on average. Matches happen at roughly the same rate as requests — call it a few thousand per second. So the location write load is *two to three orders of magnitude larger* than the ride-request load. Read that again, because it is the thesis of the entire design: in a ride-hailing system, the thing you optimize for is not rides, it is location updates. The rides are a rounding error on the write budget.

![A capacity estimation matrix showing active drivers, location writes per second, ride requests per second, matches per second, and geo-index memory at average and five times peak load](/imgs/blogs/design-a-ride-hailing-system-7.webp)

Figure 2 lays the numbers out at average and at a five-times peak (a Friday night, a big event, a storm). At peak, five million drivers online pushes location writes toward 1.25 million per second, and ride requests climb to the low tens of thousands per second. Now the punchline that makes the architecture obvious: **you cannot put 750,000 sustained writes per second into a transactional database.** A well-tuned Postgres or MySQL primary handles low tens of thousands of write transactions per second before it is fighting the machine; a million-plus per second is not a tuning problem, it is a category error. Every one of those writes would update an index, flush a write-ahead log entry, and fsync, and the database would fall over within seconds of going live.

But here is the saving grace, and it is the key insight: **the location data is tiny and disposable.** Each location record is essentially a driver ID, a latitude, a longitude, a timestamp, and a status flag — call it under a hundred bytes. Three million drivers is therefore well under a gigabyte of live location state — it fits comfortably in RAM on a single modest machine, and even at five million drivers you are around a gigabyte. And critically, you do not need *history* on the hot path: the matcher only cares about *where each driver is right now*. The previous position is worthless the instant a new ping arrives. So the entire working set is a small, in-memory, last-write-wins map. That realization — a firehose of writes against a tiny, disposable, in-memory dataset — is what unlocks the whole design. We will build the location service around it in section 5.

#### Worked example: sizing the in-memory geo-index

Let us size the index precisely to confirm it fits in RAM, because "it fits" is a claim a senior backs with numbers, not vibes. Per driver, store: driver ID (8 bytes), latitude and longitude as 64-bit doubles (16 bytes), an H3 cell index (8 bytes), a status enum (1 byte), and a last-update timestamp (8 bytes). That is about 41 bytes of payload. Real maps carry overhead — hash-table buckets, pointers, the H3-cell-to-driver-set inverted index — so multiply by a generous factor of four to five for the full structure. Call it 200 bytes per driver, all-in.

```
5,000,000 drivers (peak) * 200 bytes = 1.0 GB
```

One gigabyte at peak. You can hold the *entire global driver location index in RAM on a single large server*, with room to spare — and you would never actually do that (you shard it by region for fault isolation and per-region scaling, which we will cover in section 9), but the fact that it *could* fit on one box tells you the right answer is unambiguously "keep it in memory." Compare the alternative: at 750k writes/sec against disk, even if each write were a cheap 4 KB page touch, you would be pushing 3 GB/s of write I/O sustained, which is absurd for data you are about to overwrite four seconds later. The in-memory approach is not a clever optimization; it is the only sane choice once you have done the math.

Storage that you *do* persist is the durable stuff: trips, payments, ratings. Tens of millions of trips per day at a few kilobytes each (route polyline, timestamps, fare breakdown, both parties' IDs) is on the order of tens of gigabytes per day, a few terabytes per year — entirely manageable for a sharded relational store. That is the workload that belongs in a database, and it is small and slow-moving compared to the location firehose. The whole design is about routing each kind of data to the system built for it.

## 3. The API surface (small, and shaped by the split)

A clean API makes the consistency split explicit. There are three logical clients — the rider app, the driver app, and internal services — and the endpoints fall naturally into the location plane and the trip plane. I will sketch them as REST for readability; in production the high-frequency location channel would be a persistent connection (a WebSocket or a gRPC stream) rather than discrete HTTP calls, for the reasons covered in [API design across REST, gRPC, and GraphQL](/blog/software-development/system-design/api-design-rest-grpc-graphql).

The driver-side location endpoint is the highest-traffic call in the system, so it is deliberately minimal — fire-and-forget, no response body to wait on:

```
# Driver location stream (the firehose — 750k+/sec)
POST /v1/drivers/{driver_id}/location
  body: { lat, lng, heading, speed, status, ts }
  -> 202 Accepted   (fire-and-forget, no read-back)

# Driver availability toggle (rare)
PUT  /v1/drivers/{driver_id}/status
  body: { status: "online" | "offline" | "on_trip" }
```

The rider-side trip endpoints are low-traffic but high-stakes — they create durable state and move money — so they are designed for idempotency and strong consistency:

```
# Rider requests a ride (must be idempotent)
POST /v1/trips
  headers: { Idempotency-Key: <uuid> }
  body: { rider_id, pickup: {lat,lng}, dropoff: {lat,lng}, product }
  -> 201 { trip_id, state: "requested", eta_pickup_s, fare_estimate }

# Rider polls / streams trip state and driver position
GET  /v1/trips/{trip_id}
  -> { state, driver: {lat,lng,heading}, eta_s }
# (in production: server-push over WebSocket, not polling)

# Either party cancels
POST /v1/trips/{trip_id}/cancel
```

Notice two design choices a reviewer will probe. First, the trip-creation call carries an `Idempotency-Key` header, because a rider on a flaky cell connection will retry, and you must never create two trips or dispatch two drivers from one tap — this is the [idempotency-by-design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design) discipline applied at the front door. Second, the location POST returns `202 Accepted` with no body: the driver app does not wait for confirmation, because at 750k/sec you cannot afford a synchronous round-trip per ping, and a lost ping is harmless. The API itself encodes the consistency split: location is fire-and-forget and eventual; trips are idempotent and strong.

## 4. High-level architecture: two planes that barely touch

Return to figure 1 and read it as two planes. The **location plane** ingests driver pings at the gateway, routes each by geography to a location service that maintains an in-memory geo-index, and serves proximity queries from that index. The **trip plane** handles ride requests, runs the matching service (which queries the location plane for nearby drivers), drives the trip state machine, and persists everything durable to a strongly consistent store. The matching service is the one component that straddles both planes — it *reads* from the in-memory geo-index and *writes* to the durable trip store — which is exactly why it is the trickiest component and gets its own deep dive in section 7.

The services, and why each exists as its own thing:

- **API gateway**: terminates TLS, authenticates, and — critically — routes the location firehose down a cheap path and trip requests down the durable path. It is also where you apply [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) so a misbehaving driver app pinging 100×/sec cannot poison a shard.
- **Location service**: owns the in-memory geo-index, sharded by region. Ingests pings, answers "who is near point P." This is the highest-throughput, lowest-latency service in the system, and it is deliberately stateless-with-respect-to-durability — it can lose its memory and rebuild from the next round of pings in seconds.
- **Matching service**: takes a ride request, queries the location service for nearby candidates, ranks them by ETA, and atomically dispatches one. It is the brain.
- **Trip service**: owns the trip state machine and its durable persistence. Strongly consistent. The system of record for "what is happening with this ride."
- **Routing / ETA service**: maintains the road graph and computes travel-time estimates. Read-heavy, cacheable, and largely independent.
- **Pricing / surge service**: consumes demand and supply signals per region and emits a surge multiplier. Eventually consistent by nature.
- **Notification / streaming service**: pushes live driver position and trip updates to the rider's app over a persistent connection.

The communication backbone between these is an event log (the [queues-and-event-streaming](/blog/software-development/system-design/queues-and-event-streaming-for-architects) pattern). When a trip changes state, the trip service emits an event; pricing consumes it to update demand, analytics consumes it, notifications consume it to push to the rider. This decouples the trip plane from a dozen downstream concerns and lets each scale and fail independently. The trip service does not call the pricing service synchronously; it emits "trip.requested" and moves on. Keep the synchronous call graph as short as possible — every synchronous hop is a place the rider's request can stall.

The data model follows the two-plane split directly. The **location plane has essentially no persistent schema** — it is the in-memory `{driver -> position}` and `{cell -> driver set}` maps from section 6, plus an optional append-only ping log streamed to Kafka for analytics, which nothing on the hot path reads. The **trip plane has a small, durable relational schema**: a `drivers` table (driver ID, current status, vehicle, rating — slow-moving profile data, *not* live position), a `trips` table (trip ID, rider ID, driver ID, state, pickup and dropoff points, the chosen route polyline, timestamps for each state transition, fare breakdown), and a `payments` table referenced from the payment system. The `trips` table is the system of record and is sharded by geography (or by trip ID with a region prefix) so a region's trips live together. The deliberate design choice worth defending: **driver *position* is not a column in any table.** Position lives only in the in-memory index; the durable `drivers` row holds the stable status flag, not the lat/lng, because position changes every four seconds and belongs nowhere near a transactional store. Conflating "where is this driver" (volatile, in-memory) with "who is this driver" (stable, durable) is exactly the mistake that leads people to try to persist the firehose.

## 5. The geospatial index — the actual hard part

Here is the core problem stated cleanly: you have millions of points (drivers) that move constantly, and you need to answer "give me all points within radius R of point P" in a few milliseconds, while simultaneously absorbing hundreds of thousands of point-position updates per second. A naive solution — scan all drivers and compute distance to P — is O(n) per query and obviously hopeless at three million drivers. You need a *spatial index* that lets you look at only the drivers plausibly near P. There are four families worth knowing, and choosing among them is the single most discussed decision in any ride-hailing design.

![A matrix comparing geohash, quadtree, S2 cells, and H3 hexagons across point query speed, update cost, neighbor accuracy, range scan support, and uniform cell size](/imgs/blogs/design-a-ride-hailing-system-2.webp)

**Geohash** encodes a lat/lng into a short string by recursively bisecting the world into a grid and interleaving the bits of latitude and longitude. Nearby points share a string prefix, so "find nearby" becomes "find keys sharing this prefix," which is a cheap range scan in any key-value store or sorted structure. Geohash is beautifully simple, works in plain Redis (via sorted sets), and is great when your store already does prefix or range queries. Its two real flaws: cells are rectangular and *distort badly near the poles* (a geohash cell in Norway is a very different size and shape than one at the equator), and — the killer for proximity search — *prefix-adjacency is not spatial-adjacency at cell boundaries*. Two points a meter apart can sit on opposite sides of a major grid line and share no prefix, so a naive prefix query misses them. The standard fix is to query the cell plus its eight neighbors, which works but means you are always doing nine lookups and stitching results.

**Quadtree** is a tree that recursively subdivides space into four quadrants, splitting a node only when it holds too many points. Its strength is *adaptivity*: dense areas (downtown Manhattan) get deep, fine subdivisions; empty areas (the ocean) stay shallow. Range and nearest-neighbor queries are natural tree traversals. The cost is that it is a *mutable tree* — when drivers move, you are inserting and removing points and occasionally rebalancing nodes, and under a 750k/sec update rate that mutation and the locking around it becomes the bottleneck. Quadtrees shine for relatively static spatial data (place a quadtree over points of interest) and struggle under a high-churn moving-point workload. They are also awkward to shard cleanly across machines because the tree structure is global.

**S2** (Google's library) projects the sphere onto the six faces of a cube and uses a space-filling Hilbert curve to assign each cell a 64-bit integer ID, with a clean hierarchy of 30 levels. S2 fixes geohash's pole distortion (cells are far more uniform in area) and the Hilbert curve gives excellent locality — nearby cells have nearby IDs, so range queries on the integer ID work well. S2 cells are quadrilaterals on a cube face, which is a real improvement over lat/lng rectangles. Many systems use S2 happily; its main wrinkle is that cells are still squares, so the *six neighbors of a cell are not all equidistant from its center* — the diagonal neighbors are farther than the edge neighbors, which complicates "expand the search ring evenly" logic.

**H3** (Uber's library, open-sourced) is the one Uber built and adopted, and the reason is specific and worth understanding. H3 tiles the world with *hexagons* (with twelve pentagons to close the sphere, an unavoidable topological quirk). Hexagons have a property squares and rectangles do not: **all six neighbors of a hexagon are equidistant from its center, and they share full edges, not just corners.** This makes "expand the search outward ring by ring" — exactly the operation a proximity search needs — clean and uniform. The `k-ring` operation in H3 returns all cells within k steps of a center cell, and because of hexagon geometry every ring is an even, gap-free annulus. Cells are also far more uniform in area than geohash, and the hierarchy of 16 resolutions lets you pick a cell size matched to the density you expect. H3 is a *pure function from lat/lng to an integer cell ID* with no mutable tree, so updates are trivially cheap — you just recompute which cell a driver is in — and it shards cleanly because cells are independent integers.

Figure 3 (the decision tree later) formalizes the choice, but the short version of why Uber chose H3: the dominant query is "drivers within radius R," which is fundamentally a *ring-expansion* operation, and hexagons make ring expansion uniform and cheap; the dominant write is a position update, which with a stateless cell function is a single integer recomputation. H3 optimizes for exactly the two operations that dominate this workload. The trade-off matrix in figure 2 shows H3 winning on neighbor accuracy and uniform cell size while matching the others on query speed and update cost — there is no column where it is the worst, which is rare and is what makes it a safe default.

```python
import h3

# Resolution 8: hexagons ~0.7 km^2, ~460m edge — a good city granularity.
RES = 8

def driver_cell(lat: float, lng: float) -> str:
    """Pure function: lat/lng -> H3 cell id. No tree, no lock."""
    return h3.latlng_to_cell(lat, lng, RES)

def nearby_cells(lat: float, lng: float, k: int = 2) -> list[str]:
    """All cells within k rings of the rider — the search annulus."""
    center = h3.latlng_to_cell(lat, lng, RES)
    return h3.grid_disk(center, k)   # uniform, gap-free rings
```

Notice how little code this is, and that there is no mutable structure to lock. A driver moving from one cell to another is `old_cell = driver_cell(...)` then `new_cell = driver_cell(...)` and, if they differ, move the driver ID from one cell's set to another's. That is the entire update operation, and it is why H3 absorbs the firehose so gracefully.

#### Worked example: choosing the H3 resolution

Resolution is a real tuning knob, so let us choose it with numbers rather than guessing. At resolution 8, an H3 hexagon is roughly 0.7 km² with an edge of about 460 meters; at resolution 9 it is about 0.1 km². The trade-off: a *coarser* cell (lower resolution) means more drivers per cell, so a single cell lookup returns more candidates and you need fewer rings to find enough drivers — but each cell holds more drivers to scan. A *finer* cell (higher resolution) means fewer drivers per cell, more precise distance bounds, but you must expand more rings to gather enough candidates.

Suppose downtown at peak has roughly 50 available drivers per km². At resolution 8 (0.7 km²), that is about 35 drivers per cell — so the rider's own cell plus a single ring (`k=1`, 7 cells) yields around 245 candidates, far more than the dozen you need. So in a dense area, resolution 8 with `k=1` is plenty. In a sparse suburb with 2 drivers per km², a resolution-8 cell holds barely 1 driver, so you must expand to `k=3` or `k=4` (37 or 61 cells) to find a handful of candidates. The senior move is *adaptive ring expansion*: start at `k=1`, and if you have fewer than (say) 10 candidates, widen to `k=2`, then `k=3`, stopping as soon as you have enough. Resolution 8 as the base with adaptive `k` covers both the dense-downtown and sparse-suburb cases from a single index, which is why a fixed mid-resolution plus dynamic ring count beats trying to pick one perfect resolution per region.

## 6. The location-update write path (keeping the firehose off disk)

Now we design the path that absorbs 750k+ writes per second without a database in sight. The principle, stated once so it sticks: **the location service maintains an in-memory map from H3 cell to the set of drivers currently in that cell, plus a map from driver to their latest position, and a driver ping mutates these in RAM and returns immediately — the disk is never on the write path.**

![A pipeline showing a driver ping routed by H3 cell to an in-memory region shard, with an asynchronous snapshot every thirty seconds](/imgs/blogs/design-a-ride-hailing-system-3.webp)

Figure 4 traces a single ping. A driver app POSTs `{lat, lng, status}`. The gateway routes it by the driver's H3 base cell to the region shard that owns that geography (sharding covered in section 9). The shard computes the new H3 cell, and if the driver changed cells, removes them from the old cell's set and adds them to the new cell's set; it updates the driver-to-position map with the fresh coordinates and timestamp. Then it returns `202`. No fsync, no WAL, no index page flush — just a couple of hash-map operations on data already in RAM. That is why p99 ingest latency is sub-millisecond.

```go
type Position struct {
    Lat, Lng float64
    Cell     h3.Cell
    Status   Status
    Ts       int64
}

type GeoIndex struct {
    mu        sync.RWMutex
    byDriver  map[DriverID]*Position      // latest position
    byCell    map[h3.Cell]map[DriverID]struct{} // cell -> drivers
}

func (g *GeoIndex) Update(id DriverID, lat, lng float64, st Status, ts int64) {
    cell := h3.LatLngToCell(h3.NewLatLng(lat, lng), 8)
    g.mu.Lock()
    defer g.mu.Unlock()
    prev, ok := g.byDriver[id]
    if ok && prev.Cell != cell {
        delete(g.byCell[prev.Cell], id)        // leave old cell
    }
    if g.byCell[cell] == nil {
        g.byCell[cell] = map[DriverID]struct{}{}
    }
    g.byCell[cell][id] = struct{}{}            // enter new cell
    g.byDriver[id] = &Position{lat, lng, cell, st, ts}
}
```

A single global lock around the whole index would itself become the bottleneck at 750k/sec, so in practice you shard the lock — stripe the index by H3 cell or by sub-region so concurrent updates to different areas never contend. With, say, 256 lock stripes keyed by cell, two drivers moving in different neighborhoods update in parallel with no contention, and the write throughput scales with cores. This is the same hot-row-splitting instinct from [partitioning and sharding](/blog/software-development/system-design/partitioning-and-sharding-without-downtime), applied to an in-memory structure instead of a database.

![A before-and-after diagram contrasting persisting every driver ping to a database, which melts it, against an in-memory geo-index that keeps the database idle](/imgs/blogs/design-a-ride-hailing-system-4.webp)

Figure 5 is the comparison every design review wants to see: the naive design that persists every ping versus the in-memory geo-index. On the left, 750k writes/sec hammer Postgres — index thrash, WAL flood, p99 climbing until the database melts. On the right, updates land in RAM by region shard, p99 stays under a millisecond, and the database is *idle* because it never sees a single location write. The optimization is not subtle; it is the difference between a system that works and one that does not exist.

But "in memory" raises an obvious objection a senior must answer: what about durability — if the shard process dies, do you lose all driver locations? And the answer, which is itself a senior insight, is **you do, and it does not matter.** Driver location is *self-healing*: every driver pings again within four seconds, so a shard that loses its memory and restarts rebuilds a complete, fresh index within one ping cycle. You are not storing precious data; you are caching a snapshot of a stream that re-delivers itself constantly. The asynchronous snapshot every thirty seconds in figure 4 exists only to warm a restarted shard slightly faster and to support analytics — it is *not* on the write path and losing it costs you nothing on the hot path. This is the inverse of the usual durability instinct, and getting comfortable with "lose the data, it comes back" is what lets you keep the firehose off disk. The relevant [consistency model](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) for location is the weakest one that works: eventual, last-write-wins, and lossy-tolerant.

#### Worked example: what a dropped or delayed ping actually costs

Quantify the eventual-consistency tolerance so you can defend it. A driver moving at 50 km/h covers about 14 meters per second. If one ping is dropped and the next arrives four seconds later, the rider's map shows the car up to ~56 meters behind its true position for a few seconds before snapping forward — visually a tiny jump, well within what people accept from a moving-dot map. If the matcher uses a position that is one ping stale (four seconds, ~56m), it might rank a driver as marginally closer or farther than reality, changing the chosen driver only in the rare case of a near-tie, and even then the ETA error is a couple of seconds. So the *business cost* of location being eventually consistent and lossy is: occasionally a slightly suboptimal match and a barely-perceptible map jump. Compare that to the *cost of making it strongly consistent and durable*: a melted database and no service at all. The trade is not close. This is the kind of explicit cost-of-consistency reasoning that separates a senior answer from a junior one — you do not get strong consistency for free, and here you do not want it.

## 7. Matching and dispatch (the brain, and the race)

Matching is where the location plane and the trip plane meet, and it is the component with the trickiest correctness requirements, because it must be *fast* (a rider is waiting), *good* (match a nearby driver, not a far one), and *exactly-once* (never dispatch one driver to two riders). Those three pull against each other.

![A graph showing a rider request expanding an H3 ring search to find candidates, ranking them by ETA over a road graph, and atomically locking the winning driver](/imgs/blogs/design-a-ride-hailing-system-6.webp)

Figure 6 traces the dispatch flow. A ride request arrives with the pickup point. The matcher computes the pickup's H3 cell and does an adaptive ring search (section 5) to gather candidate drivers from the in-memory geo-index — say a dozen available cars within a couple of rings. Now it must *rank* them, and the naive ranking — straight-line distance — is wrong, because a driver 200 meters away across a river with no bridge is farther in *time* than one 600 meters away down a clear road. So the matcher calls the routing service to compute a real *ETA-to-pickup* over the road graph for the top candidates, and ranks by that. Then it picks the best driver and *atomically locks* them — and this lock is the entire game.

The dispatch race is the classic failure: two riders request near-simultaneously, both ring-searches return the same nearby driver as best, and without coordination you dispatch that one driver to both. The fix is an atomic compare-and-set on the driver's availability: the matcher attempts to transition the driver from `available` to `reserved` with a CAS (compare-and-swap), and *exactly one* of the two competing matches wins the CAS; the loser sees the CAS fail and immediately falls back to its next-best candidate. This is a coordination problem, and it needs a strongly consistent primitive — a conditional write in the trip store, a Redis `SETNX` with the driver ID, or a lease from a coordination service. It absolutely cannot run on the eventually-consistent location index, which is precisely why availability-for-dispatch lives in the *trip plane* (strong) even though position lives in the *location plane* (eventual).

```python
def dispatch(trip_id, pickup):
    candidates = geo_index.nearby(pickup, k_start=1, min_results=10)
    # Rank by real ETA over the road graph, not straight-line distance.
    ranked = sorted(candidates, key=lambda d: routing.eta_to(d.pos, pickup))
    for driver in ranked:
        # Atomic: only one matcher can win this driver.
        if trip_store.cas_reserve(driver.id, expect="available",
                                  set="reserved", trip=trip_id, ttl=15):
            offer = send_offer(driver.id, trip_id)      # ask the driver
            if offer.accepted:
                trip_store.transition(trip_id, "matched", driver.id)
                return driver
            trip_store.cas_release(driver.id)           # declined -> free
    return None   # no driver -> widen rings or apply surge and retry
```

Three production details that a staff reviewer will look for. First, the reservation has a **TTL** (15 seconds here): if the driver app does not respond to the offer in time — phone in a tunnel, app crashed — the reservation auto-expires and the driver returns to the pool, so a non-responsive driver cannot strand a car in `reserved` forever. The TTL is the difference between a system that self-heals and one that slowly bleeds available drivers into a stuck-reserved limbo; without it, every dropped offer permanently subtracts a car from the supply pool, and after a few thousand such events the matcher cannot find anyone even though the streets are full of idle drivers. A senior always asks "what happens if this lock is never released" and the answer here is "it expires," which is why the reservation is a *lease* (time-bounded) and not a *lock* (held until released). Second, dispatch is a **two-phase** thing: you *reserve* the driver, then *offer* them the ride and wait for them to accept, then *confirm*; a driver can decline, which releases the reservation and moves to the next candidate. Third, at scale you do not match one rider at a time — you **batch**. Instead of greedily assigning each request to its nearest driver the instant it arrives, you collect requests over a short window (a second or two) and solve the *assignment problem* over the batch: minimize total rider wait or total deadhead miles across all pending requests and available drivers simultaneously. Batching produces globally better matches than greedy one-at-a-time assignment because it can avoid the situation where an early request grabs a driver that a slightly-later request needed far more. This is Uber's actual approach, and it is a clean example of trading a tiny bit of latency (the batch window) for a meaningfully better global outcome.

The supply-demand structure also feeds surge. When a region's pending-request count outstrips its available-driver count over a window, the pricing service raises the surge multiplier for that region, which does two things: it rations demand (some riders defer) and it signals supply (drivers move toward the surging area for higher fares). Surge is an *eventually consistent* control signal — it updates every few seconds per region, and being a few seconds stale is fine — so it rides the event stream out of the trip plane rather than sitting on the dispatch hot path. Note that surge is computed *per H3 cell or small cell cluster*, not per city, because demand is spiky at the neighborhood level: the block outside a concert venue can be surging while the block three streets over is normal. Computing surge per cell is only tractable *because* H3 already partitions the world into cells the pricing service can aggregate over — the same index that powers matching powers pricing, which is a quiet but real dividend of picking one good spatial index and using it everywhere.

#### Worked example: when does surge actually trigger

Put numbers on the surge trigger so it is a rule, not a vibe. Take one H3 resolution-8 cell cluster covering a stadium exit. In a normal minute it sees 5 ride requests and has 8 available drivers nearby — supply exceeds demand, so the multiplier is 1.0x (no surge). The event ends and in one minute the cluster sees 200 requests against the same 8 available drivers. The pricing service computes a demand-to-supply ratio of 200/8 = 25, far above the threshold (say 2.0) at which surge engages, and maps that ratio onto a multiplier — capped, because an uncapped multiplier produces the famous \$400 fare screenshots — to perhaps 3.5x. Two effects follow within seconds. On the demand side, a fraction of the 200 riders see 3.5x and defer or walk, dropping effective demand to maybe 120. On the supply side, the surge heatmap pushes to nearby drivers, and over the next few minutes drivers reposition toward the cell, lifting supply from 8 toward 30. The ratio falls from 25 toward 4, the multiplier eases from 3.5x toward 1.6x, and the system converges. The senior point is that surge is a *control loop*, not a price tag: it is the mechanism that rations the one truly scarce resource (drivers) during a spike, and computing it per cell every few seconds — eventually consistent, off the dispatch hot path — is exactly the right consistency choice. Strong consistency on surge would buy nothing and cost latency on the price the rider sees.

## 8. The trip state machine and where consistency actually matters

A trip is not a row you update casually; it is a *state machine* with legal and illegal transitions, and modeling it explicitly is what prevents an entire class of bugs (charging for a cancelled ride, dispatching an already-matched trip, completing a trip that never started). This is where strong consistency is non-negotiable.

![A tree showing the trip lifecycle as a guarded state machine moving through requested, matched, enroute, ongoing, and completed with cancellation and abort branches](/imgs/blogs/design-a-ride-hailing-system-5.webp)

Figure 7 shows the lifecycle. A trip begins `requested`. From there it either becomes `matched` (a driver was found and accepted) or `cancelled` (no driver, or the rider gave up). From `matched` it goes `enroute` (the driver is driving to pickup), then `ongoing` (the rider is in the car), then `completed` (dropoff, fare finalized, card charged) — or, from `ongoing`, it can be `aborted` (an incident, a fault) which triggers a refund path. The crucial property is that *only the listed transitions are legal*: you cannot jump from `requested` straight to `completed`, you cannot re-enter `matched` after `ongoing`, and every transition is a *guarded, conditional write* to the strongly consistent trip store.

```sql
-- Transition is a guarded, conditional UPDATE: it only applies
-- if the trip is in the expected prior state. This is optimistic
-- concurrency control and it makes illegal transitions impossible.
UPDATE trips
   SET state = 'matched', driver_id = $1, updated_at = now()
 WHERE trip_id = $2
   AND state = 'requested';      -- guard: refuse if not 'requested'
-- 0 rows updated  => someone already advanced this trip; reject.
```

That `AND state = 'requested'` guard is the whole trick. It is optimistic concurrency control: if two processes try to advance the same trip, the database serializes them and only the first matching the expected prior state succeeds; the second updates zero rows and is rejected. Combined with the driver-reservation CAS from section 7, this makes double-dispatch *structurally impossible* — the driver cannot be reserved twice, and the trip cannot be matched twice. You are using the database's strong consistency exactly where it earns its cost.

Now the consistency split, stated as a rule a senior carries everywhere: **identify what must never be wrong and pay for strong consistency only there; let everything else be eventual.** In ride-hailing, the strong-consistency set is small and precious: the trip state machine (so a trip has exactly one truth), the driver-reservation flag (so no double-dispatch), and the payment (so no double-charge). Everything else — driver position, surge multiplier, ETAs, the rider's view of where the car is, ratings, the heatmap of demand — is eventually consistent and the system is *better* for it, because forcing strong consistency on the location firehose is what melts the database. The art is drawing that line precisely, and the [CAP and PACELC reasoning](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) gives you the vocabulary: for location you choose availability and low latency over consistency; for trip state and money you choose consistency even at the cost of some latency.

For the durable trip and payment store, you shard by geography or by trip ID and use a strongly consistent relational database (or a NewSQL store like CockroachDB or Spanner if you want global strong consistency without manual sharding — see [choosing a datastore](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql)). The write rate here is the *ride* rate — a few thousand per second, low tens of thousands at peak — which a sharded relational store handles comfortably. The money mechanics themselves (authorization, capture, idempotent charging, refunds) are their own deep subject; this post hands off to [design a payment system](/blog/software-development/system-design/design-a-payment-system) for how the charge at `completed` is made exactly once.

## 9. Sharding by geography and the in-memory index layout

Ride-hailing's natural geo-partitioning is the lever that makes global scale tractable. You do not run one global geo-index; you run *one index per region*, each owning the drivers and requests in its geography, mostly independent of the others.

![A grid showing the geo-index sharded by region, with a geo router fanning to per-city shards, a hot replica for a surging New York shard, and normal shards for other cities](/imgs/blogs/design-a-ride-hailing-system-9.webp)

Figure 8 shows the layout. A geo-router maps an incoming ping or query — by its H3 base cell — to the region shard that owns that geography. New York's drivers live on the NYC shard, San Francisco's on the SF shard, and the two never interact for matching, because no New York rider is matched to a San Francisco driver. This geographic sharding gives you three wins at once. *Isolation*: a surge or outage in one region is contained on that region's shard and does not touch others — the NYC shard can be on fire while SF hums along. *Independent scaling*: a dense city gets a bigger shard (or is itself split into sub-regions), while a quiet one runs on a fraction of a machine. *Locality*: you can place each region's shard in a data center physically near it, cutting the latency on the hot location path.

The one wrinkle is *cross-border queries*. A rider standing near a region boundary — at the edge of NYC and northern New Jersey — has nearby drivers on both sides of the shard line, so their ring search must consult two shards and merge results. You handle this by detecting that the rider's H3 ring spans a boundary and fanning the query to both owning shards, then unioning the candidates. Boundary queries are a small fraction of total queries (most riders are nowhere near a region edge), so the extra fan-out is affordable, but you must handle it or you will silently miss the closest driver for anyone near a seam. Uber's internal geo-distributed coordination here historically used a gossip and consistent-hashing layer (their open-sourced *Ringpop* library) to assign regions to nodes and route requests, so that adding or removing a node rebalances regions with minimal movement — the same [consistent-hashing](/blog/software-development/database/consistent-hashing-and-data-partitioning) idea applied to assigning *geography* to *machines* rather than keys to nodes.

Within a single hot region you apply a second tier of the same trick. New York at peak is too much for one box, so you split the NYC region itself into sub-regions by finer H3 cells, each on its own shard, with a hot read replica for the proximity queries (figure 8 shows the NYC hot replica). This is recursive geo-sharding: the world splits into regions, hot regions split into sub-regions, and the split is always along H3 cell boundaries so the routing stays a pure function of position. Because the index is in memory and self-healing, *adding a shard is cheap* — spin it up, assign it a cell range, and it fills with fresh pings within seconds, no data migration required. Compare that to resharding a stateful database, which is a painful multi-step dual-write-and-backfill operation; the disposable in-memory index makes elastic regional scaling almost trivial, which is one more dividend of keeping the firehose out of a database.

## 10. Choosing the geo-index, formalized

We have argued for H3, but the senior skill is the *decision procedure*, not the answer, because a different workload picks a different index. Figure 3 formalizes it as a decision tree you can defend in a review.

![A decision tree that selects a geospatial index by the dominant query type, routing radius search to H3, global uniform-area needs to S2, and ordered prefix scans to geohash](/imgs/blogs/design-a-ride-hailing-system-8.webp)

Read figure 3 top-down. Start from the question that actually matters: *what query dominates?* If it is "find points within radius R" — the ride-hailing case — you want ring expansion, and the default answer is **H3**, because hexagon geometry makes every ring uniform and gap-free and updates are a stateless cell recomputation. If within the radius case you additionally need *globally uniform cell area* (you are doing analytics that compare cell densities across the whole planet and cannot tolerate geohash-style distortion), **S2** is the better pick, with its near-uniform cube-projected cells and Hilbert-curve locality. If instead your dominant query is an *ordered prefix or range scan* — "give me events in this area in timestamp order," or you are piggybacking on a key-value store's native range queries — then **geohash** wins, because its prefix-equals-proximity property maps directly onto a sorted-key store like Redis or a B-tree, and you do not need the hexagon machinery. The quadtree branch (not on this tree) is for *static or slow-changing* spatial data where adaptive subdivision pays off and the high update churn that kills it never happens.

The meta-lesson, which generalizes far beyond maps: **derive the index from the dominant operation, not from which structure is most elegant or most hyped.** Uber did not adopt H3 because hexagons are pretty; they adopted it because their two dominant operations are radius search and position update, and H3 is the structure that makes *both* cheap. If your dominant operation were different, the right answer would be different, and a senior who reasons from the workload will land on the right index every time, while one who memorized "use H3 for maps" will pick wrong the moment the workload shifts.

## 11. Stress-testing the design (what breaks at 10×?)

A design you have not tried to break is a design you do not understand. Push this one against three concrete failure scenarios and watch where it bends.

**Scenario one: the New Year's Eve surge.** At the stroke of midnight, demand in every major city spikes — perhaps 5× normal ride requests in minutes — while supply (drivers online) rises more slowly. What breaks first? *Not* the location plane: even at 1.25M location writes/sec the in-memory geo-index is barely working, because RAM map updates are cheap and you can add region shards in seconds. The pressure lands on the *matching plane*: many more requests chase a barely-larger pool of drivers, so ring searches widen (fewer available cars per cell), each match does more ETA computations, and the matcher's CPU and the routing service's load climb. The designed responses, in order: surge pricing kicks in automatically to ration demand and pull supply toward hot areas (this is surge's *primary* purpose — it is a load-shedding mechanism, not just revenue); the matcher's batch window can widen slightly to make better global assignments under scarcity; and you autoscale the stateless matching and routing services horizontally. The trip store sees the *ride* rate (tens of thousands/sec at peak), which a sharded relational store handles. The system bends — riders wait longer and pay more — but it does not break, because the firehose plane has enormous headroom and the scarce resource (drivers) is rationed by price rather than by the system falling over. The honest failure mode under extreme surge is *degraded matching quality and longer waits*, which is the correct thing to degrade.

**Scenario two: a single hot region.** A huge event lets out — a stadium empties, 40,000 people want rides from one square kilometer at once. This is a *hot shard* problem: one region's index and matcher get a flood while the rest of the world is normal. Because the index is sharded by geography, the blast radius is contained to that region's shard — no global impact — which is the isolation win from section 9. The mitigations are the [hot-partition](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) playbook applied geographically: add a read replica of the hot shard so proximity *queries* (which vastly outnumber dispatches) spread across replicas; split the hot sub-region into finer H3 cells across more shards; and apply [backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) at the gateway so a thundering herd of retrying riders cannot amplify the load. The location writes themselves are still cheap; the bottleneck is the matcher doing many ETA computations against a saturated supply, so you cache and approximate ETAs under load (straight-line distance as a fallback when the routing service is saturated — a deliberate quality-for-availability trade). The region degrades gracefully and recovers as the crowd disperses; nothing outside the region notices.

**Scenario three: the dispatch race at volume.** Two riders, sometimes ten riders, request near-simultaneously and all ring-searches surface the same nearby driver. We solved correctness with the atomic CAS reservation in section 7 — exactly one wins, the rest fall to their next candidate — so there is no *correctness* failure. The *performance* failure mode under heavy contention is many matchers fighting over the same few drivers in a supply-starved area, each doing a CAS that fails and retrying down its candidate list, burning CPU. The mitigation is batching (section 7): solving the assignment over a window assigns each scarce driver to exactly one rider in a single pass, eliminating the contention storm entirely. So the dispatch race is correct under contention by construction and *efficient* under contention via batching — and naming both the correctness and the performance dimension is exactly the kind of two-layer answer a staff interview is probing for.

One more failure mode is worth naming because it surprises people: a **routing-service brownout**. The road-graph router is the most computationally expensive dependency in the matching path, and under a surge it is the first thing to saturate — not because of the location load, but because the matcher is calling it for ETA rankings far more often. When the router's p99 climbs from a few milliseconds to hundreds, every match stalls waiting on it, and the stall propagates backward: matches queue, riders wait, riders retry, and the retries pile *more* requests onto the already-saturated router. This is the classic [cascading-failure shape](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) — a slow dependency amplified by retries. The defenses are the standard ones: a *timeout* on the router call (if it does not answer in, say, 50ms, give up), a *circuit breaker* that trips the matcher into straight-line-distance fallback mode when the router's error rate spikes, and *load-shedding* at the gateway so retrying riders cannot amplify the load unboundedly. The lesson is that the bottleneck under stress is rarely the obvious firehose; it is the expensive *synchronous* dependency hiding on the hot path, and you protect it with timeouts, breakers, and a degraded fallback long before you need them.

Across all three, the pattern is the same and worth internalizing: the location firehose has enormous headroom because it is cheap in-memory work; the scarce, expensive resource is *drivers and matching CPU*; and the system degrades by rationing the scarce resource (surge, longer waits, approximate ETAs) rather than by falling over. A design that fails *gracefully* by shedding quality is the goal — see [reliability, SLOs, and graceful degradation](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) for the discipline of choosing what to degrade first.

## 12. Real-time location streaming and ETA

Two supporting subsystems round out the design, and each is simpler than the geo-index but has its own twist.

**Streaming the driver's position to the rider.** Once matched, the rider watches the car approach on a live map, which means pushing the driver's position to the rider's app every couple of seconds. You do *not* do this by having the rider poll the location service — polling at scale is wasteful and laggy. You hold a persistent connection (WebSocket) from the rider's app to a notification service, and when the matched driver pings their location, the system forwards that one driver's position to the one (or few) riders watching them. The fan-out is tiny — a driver on a trip is watched by one rider — so this is a cheap, targeted push, not a broadcast. The connection-management and routing here is the same persistent-connection problem as a [chat system](/blog/software-development/system-design/design-a-chat-and-messaging-system): you maintain a registry of which notification node holds which rider's connection, and route the driver's position update to that node. Position streaming is eventually consistent and lossy-tolerant for the same reason ingest is — a missed update just means the dot jumps slightly on the next one.

**ETA and routing.** The routing service owns a *road graph* — intersections as nodes, road segments as weighted edges, weights being real-time travel times — and answers "how long from A to B." It is used in two places: showing the rider an ETA before they request, and ranking candidate drivers by ETA-to-pickup during matching. The hard part is that edge weights are *live* — traffic changes them constantly — so the graph is continuously updated from observed trip speeds, and routing must be fast because matching calls it for several candidates per request. The standard optimization is heavy *caching and precomputation*: contraction hierarchies or precomputed shortest-path structures let you answer a route query in microseconds rather than running Dijkstra over a city graph on every call, and you [cache](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) common ETAs (popular pickup cells to popular destinations) aggressively. Under extreme load you fall back to straight-line distance times a fudge factor — less accurate, but it keeps matching alive when the routing service is saturated, which is the graceful-degradation trade again.

There is a subtle interaction between the matcher and the router worth calling out, because it is a real production tuning decision. The matcher needs an ETA for *every* candidate to rank them, but computing a full road-graph route for a dozen candidates on every one of tens of thousands of requests per second is a lot of routing load. The optimization is a *two-stage rank*: first cheaply pre-rank all candidates by straight-line distance and discard the obvious losers (the cars across the river), then call the expensive real-ETA router only for the top three or four survivors. This cuts router calls by roughly 3-4x with almost no loss in match quality, because the truly-closest driver is almost always in the straight-line top few — straight-line distance is a good *filter* even though it is a bad final *ranker*. Cheap filter, expensive refine: a pattern that recurs anywhere you have a costly scoring function over many candidates.

A second routing subtlety is that the same road graph powers *driver navigation* (turn-by-turn to pickup, then to dropoff) and *fare calculation* (distance and time traveled). These have different freshness needs: navigation must reflect live traffic and road closures to be useful, while the *fare* is computed from the actual distance and time the trip took, observed after the fact, so it does not depend on prediction accuracy at all. Separating "predicted ETA for matching and display" (best-effort, cached, can be slightly wrong) from "actual measured trip distance and duration for the fare" (exact, computed from the recorded GPS trail at trip end) keeps a wrong ETA prediction from ever producing a wrong charge — the fare is grounded in what happened, not what was forecast.

## 13. Trade-offs and rejected alternatives

A senior names the cost of every choice and the alternatives they rejected. The big decisions in this design, with their explicit trade-offs:

| Decision | What you gain | What you pay | When the alternative wins |
| --- | --- | --- | --- |
| **H3 hexagons** for the index | Uniform gap-free rings, cheap stateless updates, clean sharding | A dependency on the H3 library; 12 pentagon edge cases; not built into most datastores | Use **geohash** when you want zero dependencies and your store already does prefix/range scans |
| **In-memory geo-index** (not a DB) | Sub-ms p99, absorbs 750k writes/sec, cheap elastic shards | No durability (acceptable here); custom service to operate | Persist to a **geospatial DB** (PostGIS, Redis Geo) only if write rate is modest and you want one fewer service |
| **Eventual consistency for location** | Survival — the firehose never touches disk | Slightly stale positions, occasional suboptimal match | Strong consistency only if positions were precious and low-rate (they are not) |
| **Strong consistency for trip + payment** | No double-dispatch, no double-charge | Higher latency, sharded relational ops | Nothing — this set must be strong; the skill is keeping the set *small* |
| **Geographic sharding** | Fault isolation, locality, independent scaling | Cross-border fan-out queries at region seams | A single global index only at toy scale |
| **Batched matching** | Globally better assignments, no contention storm | A second or two of added match latency | **Greedy per-request** matching when supply is abundant and latency is paramount |

The two rejected alternatives worth stating explicitly, because interviewers love them. First, **"just use a geospatial database"** (PostGIS with a GiST index, or Redis with `GEOADD`/`GEOSEARCH`). These are excellent and you should reach for them at modest scale — a few thousand drivers, a few thousand updates/sec — where their convenience beats building a custom in-memory service. They fall down at the ride-hailing firehose: Redis Geo is in-memory and fast for queries but you would still be doing 750k writes/sec against a single-threaded server you must shard yourself, and PostGIS hits the disk-write wall from figure 5. The custom in-memory index is what you graduate *to* when the managed option melts; do not build it before you need it. Second, **"persist every ping for analytics/history"** — tempting because the location data seems valuable, but it is the exact mistake that melts the database. The right answer is to keep the *hot* index in memory and stream pings to an *append-only log* (Kafka) for any analytics or historical needs, completely off the hot path — the [event-streaming](/blog/software-development/system-design/queues-and-event-streaming-for-architects) pattern keeps history without putting it on the write-latency-critical path.

## 14. Case studies: how real systems do it

**Uber and H3.** Uber built and open-sourced H3 precisely because the geospatial indexes available did not fit their dominant operations cleanly. Their workload is overwhelmingly radius search over moving points plus high-frequency position updates, and hexagonal cells make ring expansion uniform (every neighbor equidistant) while a pure lat/lng-to-cell function makes updates a stateless integer computation. H3 is now used across Uber for matching, surge pricing (computing demand and supply per cell), ETA bucketing, and analytics, and is open source and widely adopted outside Uber. The lesson: when no off-the-shelf structure fits your dominant operation, building a workload-specific spatial index can be worth it — but only at Uber's scale and only after proving the managed options do not fit. For most teams, geohash or PostGIS is the right answer; H3 is what you reach for when those have demonstrably failed your workload.

**Uber's dispatch and geo-distributed coordination.** Uber's early dispatch system (internally "DISCO") was rebuilt around the realization that matching is fundamentally a supply-demand optimization over a geographic area, not a per-request nearest-neighbor lookup, which is why batched assignment beats greedy. For routing requests to the right node owning a geography, Uber open-sourced *Ringpop*, a library that uses consistent hashing and a gossip (SWIM-style) membership protocol to assign partitions to nodes and rebalance with minimal movement when nodes join or leave — the same consistent-hashing idea from the database track, applied to geography-to-machine assignment. The lesson: geographic partitioning plus consistent hashing for node assignment is the standard way to scale a stateful geo-service elastically.

**Lyft, DoorDash, and the same shape.** The pattern generalizes well beyond ride-hailing. Lyft solves the same nearby-driver and matching problem with the same in-memory-index-plus-strong-trip-state split. Food delivery (DoorDash, Uber Eats) is the same engine with a twist: it is a *three-sided* match (customer, restaurant, courier) with a pickup-then-dropoff routing constraint, which makes the assignment optimization harder but leaves the geospatial-index and consistency-split foundations identical. The lesson: "real-time geospatial matching with an eventual location plane and a strong transaction plane" is a *reusable architecture*, not a one-off — recognizing the shape lets you carry the whole design from ride-hailing to delivery to field-service dispatch to scooter-sharing with the core unchanged.

## 15. When to reach for this architecture (and when not to)

Reach for the full in-memory-geo-index-plus-strong-trip-plane design when you genuinely have a *high-frequency moving-point proximity-search* workload at scale: hundreds of thousands of position updates per second, sub-100ms radius queries on the interactive path, and a matching or dispatch step that must be exactly-once. Ride-hailing, food and grocery delivery, courier and field-service dispatch, real-time fleet management, and location-based games at scale all fit. The defining signal is that the *location write rate dwarfs everything else* and the data is disposable — that combination is what justifies the custom in-memory index.

Do *not* reach for it before you need it, and most teams do not. If you have a few thousand drivers and a few thousand updates per second, a managed geospatial database — PostGIS with a spatial index, or Redis with `GEOSEARCH` — is the correct answer, full stop. It is one service instead of three, it gives you durability and queries for free, and it will carry you for a long time. Building a custom sharded in-memory geo-index, a batched matcher, and a geographic routing tier is a multi-team investment that pays off only when the managed option has *measurably* hit its write-throughput wall. The senior mistake is not under-engineering; it is building Uber's architecture for a workload that PostGIS would have served happily for three more years. Match the architecture to the load you have *measured*, plus a reasonable growth runway — not to the load you fantasize about.

And do not reach for *any* of this if your "location" problem is not actually real-time. Showing the nearest store on a map, geofencing for analytics, or batch routing for next-day delivery are spatial problems but not *high-frequency-moving-point* problems, and they are solved with a static spatial index and a nightly job, not a firehose-absorbing in-memory service. The whole architecture here is justified by *movement at scale*; without that, it is enormous overkill.

## Key takeaways

- **The location-update firehose is the whole problem.** At 750k+ writes/sec it dwarfs rides by two to three orders of magnitude. Design for location updates first; everything else is comparatively easy.
- **Keep the firehose off disk.** Location is tiny (<1 GB for millions of drivers), disposable (overwritten every few seconds), and self-healing (a lost index rebuilds in one ping cycle). An in-memory geo-index, sharded by region, absorbs it at sub-ms p99 while the database stays idle.
- **Derive the spatial index from the dominant query.** Radius search over moving points wants ring expansion, and H3 hexagons make every ring uniform and updates stateless — which is why Uber built and chose H3. A different dominant query (ordered prefix scans) picks geohash instead.
- **Split consistency precisely.** Driver location is eventual, lossy-tolerant, and the system is *better* for it; trip state, driver reservation, and payment are strongly consistent. Keep the strong-consistency set small and pay for it only there.
- **Make double-dispatch structurally impossible.** An atomic CAS reservation on the driver plus a guarded conditional state transition on the trip means exactly one match can win a driver and a trip can advance only along legal edges — correctness by construction, not by hoping.
- **Geo-partition everything.** Ride-hailing shards naturally by city, giving fault isolation, independent scaling, and locality. Hot regions split recursively along H3 boundaries; the disposable in-memory index makes adding a shard nearly free.
- **Batch the matcher.** Trading a second of latency for a global assignment over a window beats greedy per-request matching on both quality and contention — and is how you survive a supply-starved surge.
- **Degrade by rationing the scarce resource.** Under a surge the firehose has headroom; drivers and matching CPU are scarce, so you ration with surge pricing, longer waits, and approximate ETAs rather than falling over.
- **Do not build Uber for a PostGIS workload.** At a few thousand updates/sec a managed geospatial store is the right answer. Graduate to the custom in-memory architecture only when you have *measured* the managed option hitting its wall.

## Further reading

- [Back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) — the technique that makes the location-firehose number obvious.
- [Partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) — the hot-partition playbook applied here to geography.
- [Caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) — for the ETA and routing cache layer.
- [Queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects) — how trip events fan out to pricing, notifications, and analytics off the hot path.
- [Consistency models: a practical guide for architects](/blog/software-development/system-design/consistency-models-a-practical-guide-for-architects) — the vocabulary for the eventual-vs-strong split that defines this design.
- [Articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) — choosing availability for location and consistency for money.
- [Choosing a datastore: SQL, NoSQL, NewSQL](/blog/software-development/system-design/choosing-a-datastore-sql-nosql-newsql) — for the durable trip and payment store.
- [Design a payment system](/blog/software-development/system-design/design-a-payment-system) — how the charge at trip completion is made exactly once.
- The H3 library documentation (Uber, open source) — the geospatial index this entire design is built on.
- Ringpop and the SWIM membership protocol — how a stateful geo-service assigns regions to nodes and rebalances.
