---
title: "The Latency Budget: Speed of Light, Serialization, and Queueing"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Derive the four clocks inside network delay, calculate the Hanoi-to-Frankfurt trade-off, and decide when a closer region beats a fatter pipe."
tags:
  [
    "networking",
    "distributed-systems",
    "latency",
    "bandwidth",
    "queueing",
    "bandwidth-delay-product",
    "network-performance",
    "linux-networking",
    "capacity-planning",
    "multi-region",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 43
image: "/imgs/blogs/the-latency-budget-speed-of-light-serialization-and-queueing-1.webp"
---

A service in Frankfurt takes 190 ms to answer a client in Hanoi. Someone proposes a 10 Gbit/s link. Someone else proposes a second deployment in Singapore. Both changes sound like "make the network faster," but they buy different things. A wider link shortens the time needed to put bits onto the wire. A closer deployment shortens the round trips that no amount of bandwidth can erase. If the response is 10 KB, distance probably owns the budget. If it is 10 MB, distance and link rate may both matter. If a queue is full, neither proposal attacks the first problem.

The diagram below is the mental model: familiar request phases on top, four physical clocks underneath. DNS, TCP, TLS, request upload, server work, first byte, and body transfer are useful observation boundaries. They are not fundamental causes. Each boundary contains some combination of propagation, serialization, processing, and queueing.

![A canonical request latency ladder mapped to propagation, serialization, processing, and queueing](/imgs/blogs/the-latency-budget-speed-of-light-serialization-and-queueing-1.webp)

This post rebuilds the numbers rather than asking you to memorize a latency chart. We will derive a lower bound from distance, price payload bytes at a link rate, distinguish work from waiting, calculate the bandwidth-delay product, and work two Hanoi-to-Frankfurt examples. The final decision is deliberately operational: when should a senior engineer move compute closer, when should they buy bandwidth, and what measurement prevents an expensive guess?

This is post 6 in [Networking for Engineers Who Ship Services](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The [layers post](/blog/software-development/networking/the-layers-are-a-lie-but-a-useful-one) explains where headers and MTU overhead enter the byte count. The [routing post](/blog/software-development/networking/how-a-packet-gets-there-arp-switching-routing-and-ecmp) explains why geographic distance is only a lower bound on the path. The [socket contracts post](/blog/software-development/networking/sockets-and-the-two-transport-contracts-tcp-vs-udp) explains why the transport must keep enough data in flight. The final [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) turns these mechanisms into a complete diagnostic sequence.

> Bandwidth answers how quickly a link can accept bits. Latency answers how long evidence takes to come back. A purchase that improves one does not repeal the other.

## A latency budget is four clocks, not one stopwatch

The useful starting model is additive:

$$
d_{one\text{-}way} = d_{prop} + d_{ser} + d_{proc} + d_{queue}
$$

This is an explanatory model, not an equation mandated by a protocol specification. It gives each elapsed interval an owner:

- **Propagation delay**, $d_{prop}$, is the time for the signal to cross physical distance.
- **Serialization delay**, $d_{ser}$, is the time for a sender to place a finite number of bits onto a link.
- **Processing delay**, $d_{proc}$, is the time devices and software spend examining or transforming the packet or request.
- **Queueing delay**, $d_{queue}$, is the time work waits because some resource is busy.

The sum is simple. The diagnostic value comes from the fact that the terms respond to different levers.

![Four delay components with their formulas, causes, and engineering levers](/imgs/blogs/the-latency-budget-speed-of-light-serialization-and-queueing-2.webp)

| Component | First-principles question | Primary lever | What does not directly fix it |
| --- | --- | --- | --- |
| Propagation | How far must the signal travel, and through what medium? | Shorter physical or routed path | A higher bit rate on the same path |
| Serialization | How many bits must enter the bottleneck, at what bit rate? | More capacity, fewer bytes, compression | Moving an unchanged bottleneck a few kilometers |
| Processing | How much per-packet or per-request work occurs at each hop? | Faster or less work, batching, offload | Empty buffers by themselves |
| Queueing | How much work is ahead of this work item? | Capacity, admission control, pacing, AQM | Faster code at an unrelated hop |

These terms can overlap. While the sender serializes the tail of an object, the leading bits can already propagate. Routers often receive enough of a frame to begin lookup work before later frames arrive. Transport protocols can keep multiple packets in flight. Therefore, adding every per-packet term for a multi-packet transfer can overcount. The model is exact for a specified packet at a specified hop. For an entire object, it is a disciplined approximation whose assumptions must be stated.

The same distinction applies to application timing. `curl` exposes boundaries such as `time_connect`, `time_starttransfer`, and `time_total`. Those boundaries tell us where the client observed elapsed time, not which physical clock caused it. A stable `time_connect` with a growing `time_total - time_starttransfer` points toward body transfer. It still does not tell us whether the body is limited by sender pacing, receive-window pressure, retransmission, serialization at a bottleneck, or a queue. We descend one layer and measure again.

### The first senior move is subtraction

Suppose a request produces these cumulative client timestamps. The values are illustrative, not a production measurement:

| Cumulative field | Value | Derived interval | Source |
| --- | ---: | ---: | --- |
| `time_namelookup` | 0.006 s | DNS: 0.006 s | Illustrative model |
| `time_connect` | 0.096 s | TCP: 0.090 s | Derived here by subtraction |
| `time_appconnect` | 0.190 s | TLS: 0.094 s | Derived here by subtraction |
| `time_starttransfer` | 0.296 s | Request, server, return: 0.106 s | Derived here by subtraction |
| `time_total` | 1.101 s | Body: 0.805 s | Derived here by subtraction |

The total is 1.101 seconds, but "a one-second network" is not a diagnosis. The 90 ms TCP interval suggests roughly one long round trip. The 94 ms TLS interval suggests another. The 805 ms body interval is suspiciously close to the serialization time of 10 MB at 100 Mbit/s, which we will derive shortly. Those similarities form hypotheses, not proof. A packet capture, byte count, socket state, and qdisc counters test them.

Subtraction also prevents a common reporting error. Curl's timestamps are cumulative from the start of the transfer. Treating `time_starttransfer` and `time_connect` as independent costs counts early phases twice. The current curl manual documents the fields in [`--write-out`](https://curl.se/docs/manpage.html#-w). Print the raw cumulative values, preserve them, and derive phase intervals explicitly.

### Build budgets as ranges, not single magic values

A latency budget is often written as if every component were fixed: 30 ms for the network, 20 ms for a cache, 40 ms for a database, and 10 ms of safety. That arithmetic can be useful for ownership, but adding component percentiles does not generally produce the same percentile for the whole request. The slowest 1 percent of network samples does not necessarily occur in the same requests as the slowest 1 percent of database samples. Correlation can make the combined tail better or worse than a naive sum suggests.

Preserve distributions and request-level correlation. Use the additive model to form hypotheses and allocate responsibility, then validate the end-to-end percentile from complete requests. Report at least the client population, route or endpoint, protocol state, payload size, sample count, observation window, and percentile. Without those dimensions, "p99 is 200 ms" is a number without a system.

The fixed and variable terms deserve different treatment. Propagation on a stable route gives a useful floor. Serialization for fixed bytes at a fixed bottleneck gives a calculable contribution. Processing and queueing often vary with workload. A practical budget can therefore be expressed as a base plus load-sensitive allowance:

$$
T_{request} \approx T_{fixed\ path} + T_{bytes} + T_{load\ sensitive}
$$

This is an explanatory accounting model. `T_fixed path` includes protocol turns over the baseline path, `T_bytes` prices the transfer at its effective rate, and `T_load sensitive` includes variable processing and waiting. The model is useful because each term has a different experiment. Measure unloaded round trips for the floor, sweep bytes for serialization, and sweep offered load for processing plus queueing.

## Propagation: rebuild the speed-of-light floor

Propagation begins with a definition, not a folklore table. The [International Bureau of Weights and Measures defines the metre](https://www.bipm.org/en/si-base-units/metre) by fixing the speed of light in vacuum at exactly 299,792,458 m/s. Optical signals in deployed fiber travel more slowly than that vacuum constant. An [AWS engineering article published October 25, 2024](https://aws.amazon.com/blogs/industries/lower-access-latency-for-your-apps-with-aws-wavelength-and-our-telco-partners/) uses about 70 percent of vacuum speed as a planning approximation and gives a rule of roughly 1 ms of round-trip propagation per 100 km of physical path.

For a path of length $d$ and signal velocity $v$, one-way propagation is:

$$
d_{prop} = \frac{d}{v}
$$

Use kilometers and kilometers per second consistently. With the round planning value $v = 200{,}000\ \text{km/s}$, a 1,000 km physical path has this one-way floor:

$$
\frac{1{,}000\ \text{km}}{200{,}000\ \text{km/s}} = 0.005\ \text{s} = 5\ \text{ms}
$$

The propagation-only round trip is approximately 10 ms. That number excludes route stretch, optical regeneration, switching, processing, queues, radio scheduling, access networks, and application work. It is a lower bound for the assumed physical path, not a latency promise.

### Hanoi to Frankfurt: a reproducible geometric lower bound

Take illustrative city-center coordinates of 21.0278° N, 105.8342° E for Hanoi and 50.1109° N, 8.6821° E for Frankfurt. Put those exact inputs into the haversine formula with mean Earth radius 6,371.0088 km. The result is about 8,719.6 km. The `Run it yourself` section includes the complete calculation, so the number is reproducible rather than asserted.

At 200,000 km/s, the great-circle, fiber-speed lower bound is:

$$
d_{one\text{-}way,min} = \frac{8{,}719.6}{200{,}000} \approx 43.6\ \text{ms}
$$

$$
d_{RTT,min} \approx 87.2\ \text{ms}
$$

No terrestrial fiber follows the geodesic through the planet's surface with zero detour. Cables follow available landings, rights of way, conduit, and provider topology. Packets may travel through an exchange or hub that is not on the shortest geographic line. If we apply a clearly labeled 1.2 path-stretch assumption, the illustrative fiber length becomes about 10,463.5 km and the propagation-only RTT becomes about 104.6 ms.

| Hanoi-to-Frankfurt model | Distance | One-way propagation | Propagation RTT | Source |
| --- | ---: | ---: | ---: | --- |
| Great-circle lower bound | 8,719.6 km | 43.6 ms | 87.2 ms | Derived here with haversine inputs shown above |
| 1.2 path-stretch illustration | 10,463.5 km | 52.3 ms | 104.6 ms | Derived here; 1.2 is an explicit modeling assumption |
| Operational example used later | Not inferred | 90 ms | 180 ms | Hypothetical model, deliberately above the geometric floor |

The 180 ms RTT in the worked transfer examples is not presented as a measurement between Hanoi and Frankfurt. It is a hypothetical operational input. A real engineer would measure application-relevant RTT from the actual client network to each candidate endpoint. The geometric calculation serves two narrower purposes: reject physically impossible expectations, and explain why buying bandwidth cannot remove a distance floor.

### Why traceroute cannot give you the fiber length

Traceroute reports responding layer-3 hops and round-trip timing to those responders. It does not disclose conduit length. Routers can decline to respond, rate-limit responses, or send control replies over a different return path. MPLS and tunnels can hide forwarding detail. The forward and return paths can differ. A named city in reverse DNS is an operator label, not a surveyed coordinate.

Use traceroute or `mtr` to find path changes and suspicious increments. Do not multiply hop count by a memorized per-hop cost. One long fiber span can dominate propagation. Ten routers in one building can contribute little propagation but meaningful processing or queueing. The [routing deep dive](/blog/software-development/networking/how-a-packet-gets-there-arp-switching-routing-and-ecmp) owns forwarding and ECMP detail; this latency model prices whatever path routing actually selects.

### Round trips multiply the floor

An interaction that needs $N$ serial round trips before useful work completes pays approximately:

$$
T_{interaction} \ge N \times RTT_{propagation\ floor}
$$

This is another explanatory lower-bound model. It ignores overlapped work, connection reuse, zero-round-trip mechanisms, loss, and every non-propagation term. It remains valuable because it catches architecture errors. With a 180 ms observed RTT, four strictly sequential request-response dependencies consume at least 720 ms before their processing time. Moving from 100 Mbit/s to 1 Gbit/s does not change that 720 ms. Removing a round trip or serving the dependency closer does.

This is why a chatty API can be slow with tiny payloads. Ten calls carrying 1 KB each are not equivalent to one 10 KB call when the calls are causally serialized. The byte total matches. The number of waits for evidence does not.

For the hypothetical 180 ms RTT, ten sequential calls have a propagation-related wait of roughly 1.8 seconds before we price processing or payload. If the ten calls can run concurrently and the server plus bottleneck can absorb them, the propagation contribution to the critical path can approach one RTT instead of ten. Concurrency is not free. It moves pressure into connection limits, worker pools, downstream capacity, and queues. The correct improvement may be one batched request, protocol multiplexing, speculative work, or moving the dependency closer.

This is also why handshake reuse matters. A new TCP connection adds a round-trip dependency before ordinary application data in the classic handshake model. TLS can add protocol turns depending on version, resumption, and transport. Reusing a healthy connection removes those turns from the current request, but it introduces pool sizing, stale connection, load balancing, and failure recovery concerns. Count the turns on the path you actually operate, not on a clean protocol poster.

## Serialization: price the bits at the bottleneck

Serialization delay is the time required to place $L$ bits onto a link whose rate is $R$ bits per second:

$$
d_{ser} = \frac{L}{R}
$$

The bottleneck rate matters, not the fastest NIC printed on a server specification. A client may have a 1 Gbit/s local interface while a 20 Mbit/s access link, shaped tunnel, Wi-Fi airtime share, or sender policer limits the flow. For a first-order payload calculation, convert bytes to bits and divide by the bottleneck bit rate.

For 10 KB in decimal units:

$$
L = 10{,}000\ \text{bytes} \times 8 = 80{,}000\ \text{bits}
$$

$$
d_{ser,100M} = \frac{80{,}000}{100{,}000{,}000} = 0.0008\ \text{s} = 0.8\ \text{ms}
$$

At 1 Gbit/s, the same payload needs 0.08 ms. The upgrade saves 0.72 ms of ideal payload serialization. On a 180 ms RTT, that change is nearly invisible to a one-round-trip, 10 KB exchange.

For 10 MB in decimal units:

$$
L = 10{,}000{,}000\ \text{bytes} \times 8 = 80{,}000{,}000\ \text{bits}
$$

$$
d_{ser,100M} = \frac{80{,}000{,}000}{100{,}000{,}000} = 0.8\ \text{s}
$$

$$
d_{ser,1G} = \frac{80{,}000{,}000}{1{,}000{,}000{,}000} = 0.08\ \text{s}
$$

Now the wider link saves 720 ms of ideal payload serialization. The same purchase that barely touched the 10 KB request materially changes the 10 MB transfer.

### Serialization and propagation overlap

The first bit does not wait for the last bit to leave before it begins traveling. The animation below shows why "RTT plus object serialization" is a useful completion-time approximation while "every packet pays the whole RTT serially" is usually wrong. At 100 Mbit/s, a 10 MB decimal payload takes 0.8 seconds to enter the bottleneck. During that interval, leading bits are already in flight and may already have arrived.

<figure class="blog-anim">
<svg viewBox="0 0 900 330" role="img" aria-label="A 10 MB object serializes over 0.8 seconds while its leading bit is already propagating from sender to receiver" style="width:100%;height:auto;max-width:900px">
<style>
.lat3-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.lat3-link{stroke:var(--text-secondary,#6b7280);stroke-width:5}.lat3-label{font:600 18px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.lat3-note{font:500 15px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}.lat3-fill{fill:var(--accent,#6366f1);opacity:.28}.lat3-bit{fill:var(--accent,#6366f1)}
@keyframes lat3-serialize{0%{transform:scaleX(0)}70%,100%{transform:scaleX(1)}}
@keyframes lat3-propagate{0%{transform:translateX(0);opacity:1}70%{transform:translateX(580px);opacity:1}71%,100%{transform:translateX(580px);opacity:1}}
.lat3-fill-anim{transform-origin:160px 226px;animation:lat3-serialize 12s ease-in-out infinite alternate}.lat3-bit-anim{animation:lat3-propagate 12s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.lat3-fill-anim{animation:none;transform:scaleX(1)}.lat3-bit-anim{animation:none;transform:translateX(580px);opacity:1}}
</style>
<text class="lat3-label" x="450" y="34">Serialization overlaps propagation</text>
<rect class="lat3-box" x="40" y="90" width="120" height="100" rx="12"/><text class="lat3-label" x="100" y="133">sender</text><text class="lat3-note" x="100" y="160">10 MB</text>
<line class="lat3-link" x1="160" y1="140" x2="740" y2="140"/><circle class="lat3-bit lat3-bit-anim" cx="160" cy="140" r="12"/>
<rect class="lat3-box" x="740" y="90" width="120" height="100" rx="12"/><text class="lat3-label" x="800" y="133">receiver</text><text class="lat3-note" x="800" y="160">leading bit</text>
<rect class="lat3-box" x="160" y="214" width="580" height="28" rx="8"/><rect class="lat3-fill lat3-fill-anim" x="160" y="214" width="580" height="28" rx="8"/>
<text class="lat3-note" x="450" y="276">derived decimal model: 10 MB = 80 Mbit</text><text class="lat3-note" x="450" y="302">80 Mbit / 100 Mbit/s = 0.8 s serialization</text>
</svg>
<figcaption>A 10 MB object enters a 100 Mbit/s link over 0.8 s while its leading bits are already in flight.</figcaption>
</figure>

For an uncongested single path with a ready sender, a rough lower-bound completion model is:

$$
T_{object} \gtrsim d_{initial} + \frac{L}{R_{bottleneck}}
$$

Here $d_{initial}$ includes the necessary propagation and protocol waits before or during delivery. The symbol $\gtrsim$ signals an approximation. Real completion can be longer because of transport startup, congestion control, loss, protocol framing, per-packet overhead, queueing, sender stalls, receiver stalls, and application production time.

### Payload is not wire size

A 10 MB response does not put exactly 10 MB on every physical medium. HTTP framing, TLS records, transport headers, IP headers, link-layer headers, acknowledgments, and retransmissions add bytes. MTU determines how many packets carry the object. Tunnels can add headers and reduce the payload available in each frame. Compression can reduce application bytes but adds processing and may change packet count.

The exact wire-time calculation therefore needs a packetization model:

$$
L_{wire} = L_{payload} + L_{protocol\ overhead} + L_{retransmitted}
$$

For our 10 KB versus 10 MB decision, using payload bits is intentionally a first-order comparison. It is sufficient to reveal which term is capable of dominating. When the answer is close, capture the traffic and use observed wire bytes. The [layers and MTU post](/blog/software-development/networking/the-layers-are-a-lie-but-a-useful-one) shows how to compute header overhead on real packets.

### A faster link can expose another bottleneck

Suppose a 100 Mbit/s WAN feeds a proxy that can encrypt at 300 Mbit/s and a client that can consume at 250 Mbit/s. Upgrading the WAN to 1 Gbit/s does not deliver 1 Gbit/s end to end. The proxy or client becomes the next bottleneck. Effective steady rate is bounded by the slowest required stage:

$$
R_{effective} \le \min(R_{sender}, R_{path}, R_{receiver})
$$

That is why `iperf3` and application transfer timing answer different questions. `iperf3` can isolate path throughput with a purpose-built sender and receiver. An application test includes storage, encryption, compression, runtime scheduling, and body generation. Use both when the application underperforms the path.

### Find serialization by sweeping object size

One measurement cannot reliably separate a fixed latency term from a per-byte term. Use at least two payload sizes while holding endpoint, connection state, and path as stable as possible. In the simple model

$$
T(L) = T_0 + \frac{L}{R_{effective}}
$$

$T_0$ is the fixed part and the slope is the inverse effective rate. For two observations $(L_1,T_1)$ and $(L_2,T_2)$, estimate:

$$
R_{effective} \approx \frac{L_2-L_1}{T_2-T_1}
$$

This is an explanatory estimator, not a transport specification. It is most useful when the larger object runs long enough to escape timer noise and both transfers share the same protocol and connection conditions. If the slope changes with object size, startup, congestion control, compression ratio, caching, or a second bottleneck is probably part of the result.

Sweep in both directions. Upload and download can have different access rates, shaping policies, routes, and endpoint costs. Record actual bytes rather than assuming the named object size reached the wire unchanged. A reverse proxy can compress a response, a client can send `Range`, and a cache can serve a different representation.

## Processing: the small term that multiplies

Processing delay is work performed after data reaches a component and before that component can advance it. On the network path, examples include route lookup, ACL evaluation, NAT state lookup, checksum work, tunnel encapsulation, TLS record processing, proxy parsing, decompression, and application dispatch. Some work occurs per packet, some per connection, some per request, and some per byte.

That distinction matters more than a single average. A per-connection 2 ms setup cost is cheap on a connection that carries 10,000 requests and expensive when every request opens a new connection. A 20 µs per-packet cost looks tiny until packets arrive at 100,000 per second, at which point one fully serialized worker would need two seconds of work per second. A per-byte compressor may help a slow link and hurt a fast local path.

Use a simple capacity check for any serialized processing stage:

$$
utilization = arrival\ rate \times service\ time
$$

This is an explanatory model. If one worker receives 40,000 packets per second and spends 20 µs of CPU on each, offered work is:

$$
40{,}000\ \text{packets/s} \times 20\ \mu\text{s/packet} = 0.8\ \text{s/s}
$$

The worker is 80 percent utilized before bursts and unrelated work. If arrival rises to 60,000 packets per second with the same service time, offered work becomes 1.2 seconds per second. The missing 0.2 seconds cannot be processed in the same second. It becomes waiting, parallel work, or loss. Processing pressure turns into queueing pressure.

### Store-and-forward and cut-through change where work starts

A store-and-forward switch receives a complete frame and verifies it before forwarding. A cut-through design can begin forwarding after enough header bytes arrive to select an output. The second design overlaps receive serialization with onward transmission, reducing per-hop latency in the clean case. It can also forward corrupted frames farther before a later check detects them. The right model depends on the actual device and mode.

Do not paste a universal "switch adds X microseconds" number into a budget. Measure or cite the hardware, frame size, forwarding mode, features enabled, and load. An empty-box forwarding benchmark does not include the queue that forms under contention. A firewall with deep inspection does different work from a layer-3 router. A virtual switch under CPU contention does not behave like a fixed-function ASIC.

### Application processing is still part of the request budget

Network diagnosis should not erase server work. If packet capture shows the request's last byte arriving at time $t_1$ and the response's first byte leaving at $t_2$, the interval $t_2 - t_1$ belongs on the server side of the socket. It may include application scheduling, database calls, lock contention, garbage collection, and another network path to a dependency.

The client cannot subdivide that interval alone. Correlate it with a server trace and dependency timings. This is where [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) should provide the request-level spans, while packet timing anchors what crossed the host boundary. The layers complement each other. A trace should not pretend it saw time before the process received bytes, and a packet capture should not pretend it saw work inside the process.

## Queueing: waiting can dwarf every fixed cost

A queue forms whenever work arrives faster than a resource can serve it for long enough to accumulate backlog. The resource can be a link, CPU run queue, NIC ring, socket buffer, proxy worker pool, connection pool, disk, or application semaphore. Queueing is not mysterious. If $B$ bits are ahead of a packet at a link that drains at $R$ bits per second, the packet's approximate waiting time is:

$$
d_{queue} \approx \frac{B}{R}
$$

This deterministic backlog calculation is an explanatory approximation. It assumes the queue drains at the stated rate and ignores arrivals that change ordering or service. If 5 MB of data is ahead at a 100 Mbit/s bottleneck, the existing backlog represents:

$$
\frac{5{,}000{,}000 \times 8}{100{,}000{,}000} = 0.4\ \text{s}
$$

That 400 ms is neither propagation nor serialization of the packet we are timing. It is serialization of other traffic that got there first. This distinction explains why latency can jump under load while the route and physical distance remain unchanged.

### The latency signature of a queue

Propagation is relatively stable until routing changes. Serialization for a fixed byte count and fixed rate is predictable. Processing can vary with CPU state, caching, and workload. Queueing often produces the most load-correlated variance: low latency when idle, rising median as utilization approaches capacity, and a long tail when bursts collide.

Look for three observations together:

1. Baseline RTT is stable when the bottleneck is idle.
2. RTT rises while a bulk flow or offered load fills the bottleneck.
3. Qdisc backlog, drops, overlimits, ECN marks, or socket queues move in the same interval.

The IETF's [RFC 7567, published July 2015](https://datatracker.ietf.org/doc/html/rfc7567), recommends active queue management because persistently full buffers add delay and can harm throughput. Its important operational point is subtle: buffers should absorb normal bursts, but the network should not maintain a large standing queue. "More buffer" is not a context-free reliability improvement.

### A queue can hide until it is almost too late

Loss is a late signal for a tail-drop queue. Before the buffer is full enough to discard packets, every packet behind the backlog already pays additional waiting time. If the alert watches only drops, users can experience degraded latency before the alert fires.

This is why a latency budget needs both idle and loaded measurements. Idle `ping` or TCP handshake time estimates a path floor. Loaded RTT shows how much standing backlog the path permits under traffic. `tc -s qdisc show dev DEVICE` exposes local qdisc backlog, drops, overlimits, and marks where supported. `ss -tin` exposes socket state and transport estimates. A bounded packet capture shows retransmission and timing. No one command proves every queue on an internet path, but correlated changes narrow the owner.

### Queue capacity is not throughput

Making a FIFO queue deeper does not increase the service rate. A 100 Mbit/s link still transmits 100 million bits per second in the ideal model. A larger queue allows more bits to wait before loss. It can smooth a burst, but it can also convert early loss into seconds of delay.

Suppose a queue holds 25 MB in front of a 100 Mbit/s link:

$$
d_{queue,max} = \frac{25{,}000{,}000 \times 8}{100{,}000{,}000} = 2\ \text{s}
$$

That is an upper-bound serialization time for a full 25 MB backlog under the stated assumptions. It does not promise that the queue is always full or that all packets see two seconds. It shows why buffer size must be discussed in time units as well as bytes or packets.

For application resilience, a timeout should not be used to conceal an unbounded queue. [Timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) explains how retries can multiply load precisely when the shared bottleneck is already saturated. Network evidence should feed that policy: rising loaded RTT and backlog are earlier signals than a storm of final timeouts.

### Convert every queue limit into time

Packet counts are convenient configuration units and poor human intuition. A 1,000-packet limit means different delay for small acknowledgments and full-size data frames. Byte limits are better, but their user-visible meaning still depends on drain rate. Convert the configured or observed backlog into time at the departure rate.

For example, 1,000 packets of 1,500 bytes represent at most 1.5 MB before link-layer details. At 100 Mbit/s, serializing that backlog takes approximately 120 ms. At 10 Mbit/s, it takes approximately 1.2 seconds. The same packet-count limit can be tolerable on one link and catastrophic on another.

| Backlog model | Departure rate | Approximate drain time | Source |
| --- | ---: | ---: | --- |
| 1,000 × 1,500-byte packets | 1 Gbit/s | 12 ms | Derived here: backlog bits/rate |
| 1,000 × 1,500-byte packets | 100 Mbit/s | 120 ms | Derived here: backlog bits/rate |
| 1,000 × 1,500-byte packets | 10 Mbit/s | 1.2 s | Derived here: backlog bits/rate |

This calculation is an upper-bound illustration because real packets vary in size, scheduling can interleave flows, AQM may mark or drop before the limit, and arrivals continue while the queue drains. It is still the right review question: how many milliseconds can this configuration add at the actual bottleneck rate?

### Queue location determines who can fix it

A send queue on the application host may respond to pacing, socket behavior, or application concurrency. A qdisc on an egress interface may respond to shaping and AQM. A provider policer may require a contract or traffic-profile change. A mobile radio scheduler is outside the server operator's direct control. A proxy worker queue may not appear in network qdisc statistics at all.

Follow the packet boundary by boundary. If client `time_starttransfer` rises but the server sends promptly after receiving the request, investigate the return path. If the server trace shows a long wait before a worker begins, do not blame fiber. If loaded ping rises only during upstream traffic, inspect the upstream bottleneck and ACK path. Direction matters because the internet path is not guaranteed to be symmetric.

## Hanoi to Frankfurt: 10 KB and 10 MB are different problems

Now combine distance and serialization with a deliberately simple model. These are not benchmark results. They are derived comparisons designed to answer one architecture question.

Assumptions:

- The far-region RTT is 180 ms.
- The near-region RTT is 30 ms.
- The bottleneck is either 100 Mbit/s or 1 Gbit/s.
- Sizes use decimal units: 10 KB is 10,000 bytes and 10 MB is 10,000,000 bytes.
- One request-response dependency is already represented by the RTT.
- The response is ready immediately, the sender can fill the path, and there is no loss or queueing.
- Completion is approximated as one RTT plus payload serialization.

![A derived matrix comparing 10 KB and 10 MB across near and far RTTs and two link rates](/imgs/blogs/the-latency-budget-speed-of-light-serialization-and-queueing-3.webp)

| Region model | Bottleneck | 10 KB model | 10 MB model | Source |
| --- | ---: | ---: | ---: | --- |
| Far, 180 ms RTT | 100 Mbit/s | 180.8 ms | 980 ms | Derived here: RTT + payload bits/rate |
| Far, 180 ms RTT | 1 Gbit/s | 180.08 ms | 260 ms | Derived here: RTT + payload bits/rate |
| Near, 30 ms RTT | 100 Mbit/s | 30.8 ms | 830 ms | Derived here: RTT + payload bits/rate |
| Near, 30 ms RTT | 1 Gbit/s | 30.08 ms | 110 ms | Derived here: RTT + payload bits/rate |

For 10 KB, moving closer saves about 150 ms. Increasing bandwidth saves about 0.72 ms. Distance wins by two orders of magnitude in this simplified comparison.

For 10 MB, moving closer at the same 100 Mbit/s saves 150 ms, while upgrading the far path from 100 Mbit/s to 1 Gbit/s saves 720 ms. The fatter pipe wins for the single large transfer. Doing both produces the 110 ms corner, assuming the transport can fill the path and no other bottleneck appears.

This is not a universal rule that small means region and large means bandwidth. The deciding variables are payload size, effective bottleneck rate, number of serial interactions, attainable in-flight data, and queueing. A 10 MB transfer broken into thousands of sequential range requests can become latency-bound. A 10 KB response behind a 5 MB queue can become queue-bound. A lossy long-distance flow may fail to reach its nominal link rate.

### Solve for the break-even payload

We can calculate the payload size at which a bandwidth upgrade saves the same time as a region move. Let $R_1$ and $R_2$ be the old and new rates, and let $\Delta RTT$ be the RTT reduction. Set serialization savings equal to RTT savings:

$$
L\left(\frac{1}{R_1} - \frac{1}{R_2}\right) = \Delta RTT
$$

$$
L = \frac{\Delta RTT}{\frac{1}{R_1} - \frac{1}{R_2}}
$$

For a 150 ms RTT reduction, 100 Mbit/s old rate, and 1 Gbit/s new rate:

$$
L = \frac{0.150}{\frac{1}{100{,}000{,}000} - \frac{1}{1{,}000{,}000{,}000}}
\approx 16{,}666{,}667\ \text{bits}
$$

That is about 2.08 MB decimal. Below that payload, the 150 ms region move saves more in this one-RTT model. Above it, the rate upgrade saves more. Add another serial round trip and the region move's benefit doubles to 300 ms, moving the break-even payload to about 4.17 MB. Interaction shape changes the answer.

### First byte and last byte deserve separate SLOs

The region move strongly improves the earliest possible response because first-byte latency contains protocol and propagation waits. The bandwidth upgrade strongly improves the drain time of a large body. A single `time_total` percentile hides that distinction.

For interactive APIs, track time to first byte, total time, payload size, and connection reuse. For downloads and streams, track sustained goodput and stall behavior as well. For uploads, separate request-body completion from server response. The [API performance post](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency) owns higher-level payload and compression choices. The network budget supplies the physics those choices must respect.

## Bandwidth-delay product: how many bits fit in the path

Bandwidth-delay product, or BDP, is the amount of data that fits in flight at a chosen rate over a chosen round-trip time:

$$
BDP = R \times RTT
$$

[RFC 6349, published August 2011](https://datatracker.ietf.org/doc/html/rfc6349), uses bottleneck bandwidth and RTT to derive the send and receive socket buffer sizes needed for maximum TCP throughput. The transport's actual in-flight data is constrained by congestion control, receiver flow control, sender availability, and implementation limits. BDP is the capacity target, not a promise that TCP immediately occupies it.

At 1 Gbit/s and 180 ms RTT:

$$
BDP = 1{,}000{,}000{,}000\ \text{bit/s} \times 0.180\ \text{s}
= 180{,}000{,}000\ \text{bits}
$$

$$
BDP = 22{,}500{,}000\ \text{bytes} = 22.5\ \text{MB}
$$

![A bandwidth-delay pipe showing underfilled and full in-flight windows](/imgs/blogs/the-latency-budget-speed-of-light-serialization-and-queueing-4.webp)

If the sender can keep only 4 MB in flight, an idealized upper bound from the window is:

$$
R_{window\ bound} = \frac{4\ \text{MB} \times 8}{0.180\ \text{s}} \approx 177.8\ \text{Mbit/s}
$$

A 1 Gbit/s provisioned path can therefore behave like a roughly 178 Mbit/s path for that flow under the simplified window bound. Buying more line rate without enough in-flight data produces disappointing utilization. The fixes might include socket buffer tuning, transport behavior, parallel flows, request layout, or simply transferring an object large enough to leave startup. Each fix carries trade-offs.

### BDP is not the queue size you should configure

The same units tempt people into a bad shortcut: "BDP is 22.5 MB, so every router needs a 22.5 MB queue." BDP describes data in flight across an end-to-end control loop. A queue is backlog at a particular contention point. The appropriate queue policy depends on burst behavior, number of flows, scheduling, congestion signaling, and acceptable delay.

If a 22.5 MB standing queue formed at 1 Gbit/s, its drain time would be 180 ms. It would add another RTT of waiting to packets at the tail. The pipe could be full while responsiveness was poor. RFC 7567 explicitly separates useful burst absorption from persistently full buffers and recommends active queue management in operational deployments.

### BDP changes every time rate or RTT changes

At 100 Mbit/s and 180 ms, BDP is 18 Mbit, or 2.25 MB decimal. At 1 Gbit/s and 30 ms, BDP is 30 Mbit, or 3.75 MB. The higher-rate far path needs far more in-flight data than the lower-latency near path.

| Path model | Rate | RTT | BDP | Source |
| --- | ---: | ---: | ---: | --- |
| Far, narrower | 100 Mbit/s | 180 ms | 2.25 MB | Derived here: rate × RTT |
| Far, wider | 1 Gbit/s | 180 ms | 22.5 MB | Derived here: rate × RTT |
| Near, narrower | 100 Mbit/s | 30 ms | 0.375 MB | Derived here: rate × RTT |
| Near, wider | 1 Gbit/s | 30 ms | 3.75 MB | Derived here: rate × RTT |

This is a second reason closer compute can improve throughput for one flow even when line rate is unchanged. A smaller BDP is easier for the transport and endpoints to fill. The effect depends on congestion control, windows, loss, and object size, so measure rather than declaring a universal gain.

### One loss costs more on a long feedback loop

BDP also explains why a long path is sensitive to recovery behavior. When the sender learns about congestion or loss only after feedback returns, the control loop spans an RTT. More data can be in flight before the signal arrives on a high-rate, long-RTT path. Recovery details depend on the transport, acknowledgment pattern, loss position, reordering, and congestion-control algorithm, so there is no honest universal "one loss costs one RTT" formula.

The operational rule is narrower. When a nominally fast long-distance path underperforms, inspect retransmissions, congestion window, receive window, pacing, and RTT together. A low application rate with a full receive window suggests a different owner from a low rate with repeated loss recovery. `ss -tin` provides a live Linux view of several transport estimates, while a bounded packet capture establishes packet order and acknowledgment timing.

Parallel connections sometimes raise aggregate throughput by creating more independent in-flight state. They can also take more than a fair share, amplify burstiness, consume connection and NAT state, and hide the single-flow problem. Treat parallelism as an application design choice with externalities, not a free bandwidth knob.

## Public case: AWS Wavelength moved compute closer

On October 25, 2024, AWS published a [Wavelength latency study](https://aws.amazon.com/blogs/industries/lower-access-latency-for-your-apps-with-aws-wavelength-and-our-telco-partners/) comparing mobile user equipment reaching a local Wavelength Zone with reaching its parent AWS Region. The test used Netperf's TCP request-response workload: send one byte, receive one byte, repeat, and calculate average transaction time. Each plotted data point averaged more than 1,000 transactions according to the post.

![AWS 2024 mobile request-response paths to local Wavelength Zones and parent Regions](/imgs/blogs/the-latency-budget-speed-of-light-serialization-and-queueing-5.webp)

The article reports three useful observations. In Manchester, which it describes as around 250 km from London, one operator showed roughly an 8 ms average RTT reduction to the local Wavelength Zone. In Berlin, more than 400 km from Frankfurt, the local zone was about 12 ms lower on average. In Dallas, around 1,800 km from the Northern Virginia parent Region, local-zone RTT was around half the Region RTT. These are the source's measurements in its named mobile networks and test periods, not universal constants.

The mechanism maps cleanly to our budget. The workload carried one byte each way, so payload serialization was negligible compared with network and packet-processing delay. The local Wavelength path reduced physical distance and avoided some packet handling on the longer route toward the parent Region. A fatter link would not remove the serial request-response wait. Shortening the path attacked the dominant terms.

The case also warns against using geography alone. The AWS authors measure TCP request-response latency rather than relying only on straight-line distance. Mobile radio access, packet core placement, backhaul, peering, and route selection can dominate a map's intuition. Even the two Manchester operators showed different reductions and variability. "Nearest region" should mean lowest measured application-relevant path latency for the client population, not nearest city label.

### Evidence ledger

| Field | Verified content | Source |
| --- | --- | --- |
| Case | AWS Wavelength Zones compared with parent AWS Regions | AWS engineering post |
| Event or publication date | October 25, 2024 | Date on source page |
| Source owner | AWS, John Naylon | Byline on source page |
| Mechanism | Shorter physical path and fewer packet-handling stages for mobile traffic | Source explanation and test topology |
| Verified numbers | Roughly 8 ms lower in one Manchester operator, about 12 ms lower in Berlin, Dallas around half; each plotted point averaged more than 1,000 transactions | [AWS Wavelength latency study](https://aws.amazon.com/blogs/industries/lower-access-latency-for-your-apps-with-aws-wavelength-and-our-telco-partners/) |
| Transfer lesson | For tiny request-response exchanges, compare measured path latency to nearer compute before buying bandwidth | Derived here from the published mechanism and workload |

The trigger was not an outage. It was an architecture choice with a measurable latency target. That still makes it a useful public case because it isolates the exact decision in this post. It also avoids a common case-study mistake: attributing a broad user-experience claim to distance without publishing the measurement method.

## The senior decision: closer region or fatter pipe?

Start with the term that dominates the user-visible interval, not the resource that is easiest to buy. The decision tree below makes the order explicit.

![A decision tree choosing closer compute, more bandwidth, queue control, or processing work from measurements](/imgs/blogs/the-latency-budget-speed-of-light-serialization-and-queueing-6.webp)

### Choose a closer region or edge when round trips dominate

This is the likely answer when payloads are small, requests are interactive, dependencies are serial, or handshake and first-byte intervals scale with RTT. Examples include login, autocomplete, game input, remote desktop, synchronous database coordination, and APIs with multiple request-response turns.

Validate the choice with client-population measurements. Compare TCP or application request-response timing from representative networks to candidate endpoints. Inspect routing because a geographically closer site can have worse peering. Model state placement, consistency, failover, and operations because compute without the required data may add a remote backend round trip and erase the gain.

The [multi-region architecture guide](/blog/software-development/system-design/multi-region-and-geo-distribution) owns the system-wide trade-offs. This post contributes one boundary condition: synchronous cross-region interaction cannot be designed as if distance were free.

### Choose a fatter pipe, fewer bytes, or compression when serialization dominates

This is the likely answer when large request or response bodies spend most of their time draining at a measured bottleneck. Compare body bytes with sustained rate. Confirm that sender, receiver, storage, CPU, and transport windows can use the proposed capacity. A 10 Gbit/s port is irrelevant if a 100 Mbit/s policer or 250 Mbit/s encryption stage remains in the path.

Reducing bytes is often better than buying rate because it helps every constrained hop. Compression trades CPU and sometimes latency for fewer bits. Caching avoids repeated transfer but introduces freshness and invalidation policy. Parallelism can fill a path but competes with other flows and may worsen queueing. State the second-order cost with the optimization.

### Choose queue control or capacity when loaded latency moves

If idle RTT is healthy and loaded RTT rises sharply, treat the queue as a first-class suspect. Find the bottleneck, inspect backlog and marks, compare arrival and departure rate, and separate a short burst from sustained overload. Capacity can lower utilization. Admission control and backpressure stop new work from joining an already harmful queue. Pacing smooths bursts. Fair queueing prevents one flow from monopolizing waiting time. AQM signals congestion before a standing queue consumes the latency budget.

The [rate limiting and backpressure guide](/blog/software-development/system-design/rate-limiting-and-backpressure) owns higher-level admission policy. At the network layer, our job is to prove where waiting accumulates and express the backlog in time.

### Choose processing work only after separating it from waiting

High CPU with stable queues can indicate a processing bottleneck. Low CPU does not clear processing if one serialized worker, lock, interrupt target, or accelerator is saturated. Measure service time at the component and queue time before it. Scaling workers helps only when the work can run in parallel and the downstream resource can absorb it.

A useful review table is qualitative because the measurement selects the branch:

| Observation | Dominant hypothesis | Candidate move | Required guardrail |
| --- | --- | --- | --- |
| Small payload, several serial interactions, phase time tracks RTT | Propagation | Closer region, edge termination, fewer turns | Data locality, consistency, failover |
| Large body, stable RTT, body time tracks bytes/rate | Serialization | More bottleneck capacity, fewer bytes, compression | Endpoint and transport can fill rate |
| Idle fast, loaded slow, backlog or marks rise | Queueing | Capacity, pacing, AQM, admission control | Preserve useful burst absorption |
| On-wire gaps align with CPU or service time, no backlog growth | Processing | Remove work, batch, offload, scale | Avoid moving bottleneck downstream |

### Price the architecture, not only the milliseconds

A closer region may require replicated data, deployment automation, observability, incident coverage, capacity headroom, and a consistency policy. A bandwidth upgrade may require new ports, transit commitments, tunnel capacity, firewalls, load balancers, and endpoint work. Compression can increase CPU cost and tail latency. Caching can create stale data. The latency model selects a technically capable lever. It does not make the lever economically or operationally correct by itself.

Frame the proposal as an outcome per unit of complexity. For each candidate, state the latency term it changes, the expected bound from arithmetic, the measurement that will validate it, the failure mode it introduces, and the rollback. A proposal to add a region should name which serial round trips disappear and which data calls remain remote. A proposal to buy bandwidth should name the measured bottleneck and show that endpoints can use the new rate.

A cheap experiment should precede a permanent topology change. Route a small representative cohort to an existing nearer endpoint, test an edge function, compress one response family, shape a staging link, or replay a bounded workload with controlled RTT and rate. Preserve enough telemetry to reject the hypothesis. An experiment that can only report success is a rollout ceremony, not engineering evidence.

### The dominant term can change during one request

A cold HTTPS request may begin propagation-bound because TCP and TLS need serial exchanges. The response body may become serialization-bound. A concurrent backup can fill the access queue and make it queue-bound. The server can pause mid-stream and make it processing-bound. One label for the whole request is too coarse.

Instrument boundaries that let you change your mind. Preserve connection setup, first byte, body duration, bytes, endpoint, protocol, and reuse. Correlate with socket and qdisc state. Segment by client network and region. Averages across paths can manufacture a duration that no individual user experienced.

## Common mistakes that produce confident wrong budgets

### Memorizing a latency table without rebuilding the assumptions

"A cross-continent round trip is 100 ms" might be a useful order of magnitude and a terrible capacity-planning input. Which cities? Which access networks? Which route? Which percentile? Which date? Was the connection warm? Did the measurement use ICMP, TCP request-response, or HTTP? A remembered number has lost the conditions that made it true.

Rebuild the floor from distance when you need a plausibility check. Measure the real path when you need an engineering decision. Cite the population and time window when reporting the result.

### Calling throughput bandwidth

Bandwidth or link rate is a capacity. Throughput is delivered bits over time. Goodput is useful application data over time. Protocol overhead, retransmission, competing traffic, endpoint limits, and window limits separate them. A 1 Gbit/s link can deliver less than 1 Gbit/s throughput, and useful payload goodput is lower still.

Name the quantity and units. `iperf3` throughput is not automatically HTTP goodput. A cloud interface limit is not automatically the per-flow rate. An average over 30 seconds can hide startup and stalls that dominate a 200 ms interaction.

### Treating ping as an application benchmark

Ping is useful for reachability and an ICMP round-trip sample when the path handles ICMP normally. It does not include DNS, TCP, TLS, HTTP, server work, payload transfer, or application queueing. Networks can prioritize or filter ICMP differently. A healthy ping does not prove a healthy service. A failed ping does not prove the TCP port is unreachable.

Use ping in this post's lab to verify that the configured propagation treatment took effect. Use an application or transport test for the service claim.

### Treating an average as a budget

Interactive systems fail at the tail. Queueing, route changes, loss recovery, and runtime pauses make distributions asymmetric. Report percentiles or raw samples with the observation window and population. Preserve sample count. If you compare two paths, compare like with like and include variance.

The AWS Wavelength case is careful about method: repeated one-byte TCP request-response transactions, more than 1,000 per plotted point, with separate operator paths. That context is why its reductions are evidence instead of folklore.

### Improving a component that is not on the critical path

If application work and a network transfer overlap, shortening the non-critical branch may not change completion. If a dependency is issued speculatively and usually completes before it is needed, its latency is real but not user-visible. A budget must represent causality, not merely sum every timer found in telemetry.

Draw the dependency chain. Identify what the caller waits for. Then apply the four-delay model to the critical network intervals. This is where a distributed trace and packet timeline should agree about order even though they observe different layers.

## Run it yourself

### Question

Can we change serialization time by changing the emulated link rate while keeping configured propagation approximately constant?

The experiment uses the canonical `netlab` namespaces from the [series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). It first verifies the Hanoi-to-Frankfurt geometric calculation. It then compares a fixed 10,000,000-byte reverse `iperf3` transfer at 100 Mbit/s and 1 Gbit/s while both directions retain 15 ms of configured delay.

### Preconditions

Run this only inside a disposable Linux VM or lab host. You need `iproute2`, `iputils-ping`, `iperf3`, Python 3, root privileges, and the canonical namespaces `c` and `s` with interfaces `c0` and `s0`. `tc` changes qdiscs and requires `CAP_NET_ADMIN`. Do not apply these commands to an unspecified production interface. Timer granularity, virtualization, offload, CPU, and the host's native link rate affect results.

Confirm the environment without changing it:

```bash
set -euo pipefail

ip netns list
ip -n c addr show dev c0
ip -n s addr show dev s0
ip -n c route get 10.77.0.2
ip netns exec c ping -c 3 -n 10.77.0.2
iperf3 --version
tc -V
```

Read: both namespaces exist, `c0` owns `10.77.0.1/30`, `s0` owns `10.77.0.2/30`, the route selects `c0`, and all three pings succeed.

Expected: before impairment, namespace RTT is commonly below 5 ms on an otherwise idle native Linux host and may be higher in a nested VM. Treat the observed baseline as the control, not as a universal benchmark.

### Rebuild the geographic lower bound

```bash
python3 - <<'PY'
from math import asin, cos, radians, sin, sqrt

hanoi = (21.0278, 105.8342)
frankfurt = (50.1109, 8.6821)
earth_radius_km = 6371.0088
fiber_speed_km_s = 200000.0

lat1, lon1 = map(radians, hanoi)
lat2, lon2 = map(radians, frankfurt)
dlat = lat2 - lat1
dlon = lon2 - lon1
a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
great_circle_km = 2 * earth_radius_km * asin(sqrt(a))

for stretch in (1.0, 1.2):
    path_km = great_circle_km * stretch
    one_way_ms = path_km / fiber_speed_km_s * 1000
    print(
        f"stretch={stretch:.1f} path_km={path_km:.1f} "
        f"one_way_ms={one_way_ms:.1f} rtt_ms={2 * one_way_ms:.1f}"
    )
PY
```

Read: `path_km`, `one_way_ms`, and `rtt_ms` for each explicit stretch assumption.

Expected: about 8,719.6 km and 87.2 ms propagation RTT at stretch 1.0; about 10,463.5 km and 104.6 ms propagation RTT at stretch 1.2. Small last-digit differences can come from coordinate precision or Earth-radius choice.

### Baseline: 100 Mbit/s

Start a bounded server, then apply 15 ms delay and 100 Mbit/s rate in each direction. The [`tc-netem(8)` manual](https://man7.org/linux/man-pages/man8/tc-netem.8.html) documents both `delay` and `rate`, and warns that timer granularity can create artificial packet compression.

```bash
set -euo pipefail

sudo ip netns exec s pkill -x iperf3 2>/dev/null || true
sudo ip netns exec s iperf3 -s -D --pidfile /tmp/netlab-latency-iperf3.pid

sudo ip netns exec c tc qdisc replace dev c0 root \
  netem delay 15ms rate 100mbit limit 1000
sudo ip netns exec s tc qdisc replace dev s0 root \
  netem delay 15ms rate 100mbit limit 1000

sudo ip netns exec c tc -s qdisc show dev c0
sudo ip netns exec s tc -s qdisc show dev s0
sudo ip netns exec c ping -c 10 -n 10.77.0.2
sudo ip netns exec c iperf3 -c 10.77.0.2 -R -n 10000000 -J \
  > /tmp/netlab-latency-100m.json
python3 - <<'PY'
import json
with open('/tmp/netlab-latency-100m.json') as f:
    result = json.load(f)
summary = result['end']['sum_received']
print(f"seconds={summary['seconds']:.3f}")
print(f"bits_per_second={summary['bits_per_second']:.0f}")
print(f"bytes={summary['bytes']}")
PY
```

Read: `delay 15ms` and `rate 100Mbit` in each qdisc, ping's `rtt min/avg/max/mdev`, and `seconds`, `bits_per_second`, and `bytes` from `sum_received`.

Expected: average ping RTT is usually 30–45 ms in an idle lab. The byte count should be 10,000,000. Ideal payload serialization is 0.8 seconds; a practical lab run should commonly land around 0.8–1.4 seconds and below 100 Mbit/s of reported application throughput because protocol overhead, startup, scheduling, and timer granularity remain. If it is far outside that range, inspect qdiscs, CPU saturation, retransmits, and the VM's native path before drawing a conclusion.

### Apply one change: 1 Gbit/s

Change only the emulated rate. Keep 15 ms delay, namespace topology, byte count, direction, and tool unchanged.

```bash
set -euo pipefail

sudo ip netns exec c tc qdisc replace dev c0 root \
  netem delay 15ms rate 1gbit limit 1000
sudo ip netns exec s tc qdisc replace dev s0 root \
  netem delay 15ms rate 1gbit limit 1000

sudo ip netns exec c tc -s qdisc show dev c0
sudo ip netns exec s tc -s qdisc show dev s0
sudo ip netns exec c ping -c 10 -n 10.77.0.2
sudo ip netns exec c iperf3 -c 10.77.0.2 -R -n 10000000 -J \
  > /tmp/netlab-latency-1g.json
python3 - <<'PY'
import json
for label in ('100m', '1g'):
    with open(f'/tmp/netlab-latency-{label}.json') as f:
        result = json.load(f)
    summary = result['end']['sum_received']
    print(
        label,
        f"seconds={summary['seconds']:.3f}",
        f"Mbps={summary['bits_per_second'] / 1e6:.1f}",
        f"bytes={summary['bytes']}",
    )
PY
```

Read: ping average should remain in the same broad range, while the 10,000,000-byte transfer's `seconds` should fall. This discriminates configured propagation from serialization at the emulated bottleneck.

Expected: average ping RTT remains roughly 30–45 ms. Ideal 10 MB serialization at 1 Gbit/s is 0.08 seconds, but short-flow startup and host limits matter much more at this duration. Expect roughly 0.08–0.40 seconds on a capable lab host. Do not call a slower result a protocol failure until you inspect CPU, offloads, congestion window, receiver window, and native veth throughput.

### Observe queueing as a separate experiment

Keep the 100 Mbit/s treatment and watch qdisc state while a longer reverse transfer runs. This deliberately creates contention in the lab. It does not prove where a queue sits on a production internet path.

```bash
set -euo pipefail

sudo ip netns exec c tc qdisc replace dev c0 root \
  netem delay 15ms rate 100mbit limit 1000
sudo ip netns exec s tc qdisc replace dev s0 root \
  netem delay 15ms rate 100mbit limit 1000

sudo ip netns exec c iperf3 -c 10.77.0.2 -R -t 15 \
  > /tmp/netlab-latency-load.txt &
load_pid=$!

sudo ip netns exec c ping -i 0.2 -c 50 -n 10.77.0.2 \
  | tee /tmp/netlab-latency-loaded-ping.txt
sudo ip netns exec s tc -s qdisc show dev s0
wait "$load_pid"
```

Read: compare loaded ping's `rtt min/avg/max/mdev` with the idle sample, then inspect `backlog`, `dropped`, and `overlimits` for `s0`, the response direction's egress qdisc.

Expected: exact queue growth varies with netem, host scheduling, and traffic bursts. The qualitative pass condition is either a loaded RTT increase or observable backlog/overlimit activity while the bulk transfer runs. If neither appears, the lab did not create a standing queue. That is a valid result; reduce the queue limit or add competing flows rather than inventing a latency increase.

### Reset

Remove only the qdiscs and process created by this experiment:

```bash
set -euo pipefail

sudo ip netns exec c tc qdisc del dev c0 root 2>/dev/null || true
sudo ip netns exec s tc qdisc del dev s0 root 2>/dev/null || true
sudo ip netns exec s pkill -x iperf3 2>/dev/null || true
rm -f /tmp/netlab-latency-100m.json \
  /tmp/netlab-latency-1g.json \
  /tmp/netlab-latency-load.txt \
  /tmp/netlab-latency-loaded-ping.txt
```

For a production host, prefer read-only commands: `tc -s qdisc show dev DEVICE`, `ss -tin`, bounded `mtr`, and application phase timings. Packet capture can contain credentials, tokens, personal data, and application payloads. Use a narrow filter, bounded duration, and approved storage when capture is necessary.

The experiment proves a limited claim. With configured delay held approximately constant, changing rate changes a finite transfer much more than it changes small-packet RTT. It does not benchmark Hanoi-to-Frankfurt internet service, predict cloud performance, or establish a universal TCP throughput number.

## Key takeaways

- End-to-end delay is usefully decomposed into propagation, serialization, processing, and queueing. Each term has a different lever.
- Distance creates a floor. At the 200,000 km/s fiber planning approximation, every 100 km of physical path contributes roughly 1 ms of propagation RTT.
- Serialization is payload bits divided by bottleneck bits per second. Always name decimal versus binary units and payload versus wire bytes.
- For the hypothetical Hanoi-to-Frankfurt model, a 10 KB exchange gains far more from reducing 180 ms RTT to 30 ms than from increasing 100 Mbit/s to 1 Gbit/s.
- In the same model, a 10 MB transfer gains more from the bandwidth upgrade because ideal serialization falls from 800 ms to 80 ms.
- BDP is rate times RTT. A 1 Gbit/s, 180 ms path needs about 22.5 MB of in-flight data to remain full in the ideal model.
- BDP is not a queue-sizing command. A standing queue of one BDP adds roughly one RTT of waiting at that rate.
- Loaded latency and backlog distinguish queueing from a fixed propagation floor. Measure both idle and loaded paths.
- Choose a closer region for round-trip-bound interactions, a fatter pipe or fewer bytes for serialization-bound transfers, queue control for load-correlated waiting, and processing optimization for measured work.

## Further reading

- [BIPM: SI base unit definition of the metre](https://www.bipm.org/en/si-base-units/metre), for the exact vacuum speed of light.
- [RFC 6349: Framework for TCP Throughput Testing](https://datatracker.ietf.org/doc/html/rfc6349), published August 2011, for RTT, bottleneck bandwidth, BDP, and TCP throughput methodology.
- [RFC 7567: IETF Recommendations Regarding Active Queue Management](https://datatracker.ietf.org/doc/html/rfc7567), published July 2015, for queue management and standing-delay guidance.
- [Linux `tc-netem(8)`](https://man7.org/linux/man-pages/man8/tc-netem.8.html), for the impairment controls and their limitations.
- [AWS Wavelength latency study](https://aws.amazon.com/blogs/industries/lower-access-latency-for-your-apps-with-aws-wavelength-and-our-telco-partners/), published October 25, 2024, for a public request-response comparison of local edge zones and parent Regions.
- [API performance: payload size, compression, and tail latency](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency), for the application-level choices that change bytes and interaction shape.
