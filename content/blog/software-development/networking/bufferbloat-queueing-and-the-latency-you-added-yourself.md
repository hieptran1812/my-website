---
title: "Bufferbloat: The Queueing Delay You Added Yourself"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn to derive queueing delay, prove bufferbloat under load, and choose CoDel or FQ-CoDel without hiding the bottleneck."
tags:
  [
    "networking",
    "distributed-systems",
    "bufferbloat",
    "queueing",
    "linux-tc",
    "codel",
    "fq-codel",
    "latency",
    "traffic-control",
    "performance-debugging",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 42
image: "/imgs/blogs/bufferbloat-queueing-and-the-latency-you-added-yourself-1.webp"
---

Your video upload reaches the advertised line rate. At the same moment, an SSH keystroke takes half a second to echo, a voice call becomes robotic, and a health check occasionally crosses its timeout. The application servers are idle. There is no packet-loss alarm. The link looks busy but healthy.

That combination is the signature worth remembering: throughput remains good while latency under load becomes terrible. A buffer in front of the bottleneck is storing more work than interactive traffic can tolerate. The buffer did not make the link faster. It merely moved congestion from a visible drop into an invisible wait.

![A bulk upload fills the bottleneck egress queue while a small interactive packet waits behind it](/imgs/blogs/bufferbloat-queueing-and-the-latency-you-added-yourself-1.webp)

The diagram above is the mental model: the application can finish its work before the network begins the longest part of the request. A packet that enters a saturated egress queue is complete from the application's perspective, yet it still waits behind bytes that the bottleneck can transmit only at a fixed rate. This post follows that wait from arithmetic to Linux counters, then replaces an unmanaged FIFO with CoDel and FQ-CoDel.

This is one chapter in [Networking for Engineers Who Ship Services](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). If the words bandwidth-delay product, congestion window, and receive window are still blending together, read [flow control versus congestion control](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe) first. The distinction matters because bufferbloat is queue state, not simply a small window or a slow server.

## 1. Put the queue on the path

**Rule of thumb: a latency investigation is incomplete until you can point to the queue that owns the waiting time.**

A queue forms wherever work arrives faster than the next stage can serve it. On a host, that might be a socket send buffer, a queueing discipline, a driver ring, or a NIC transmit queue. In a home network, it might be the router, Wi-Fi firmware, cable modem, or provider edge. In a service path, it might be a virtual switch, tunnel endpoint, load balancer, or shaped egress interface.

The narrowest link is usually the place to look first. Suppose an application can hand bytes to the kernel at several gigabits per second, but the WAN uplink can serialize only 20 Mbit/s. The application-side arrival process is bursty and fast. The departure process is capped at 20 Mbit/s. A queue absorbs their difference.

The queue is not automatically a defect. A short queue absorbs scheduling jitter and keeps the transmitter busy between bursts. The defect appears when the queue becomes a standing queue: it does not drain back to empty, or close to empty, during normal operation. Every new packet then inherits the stored wait.

| Observation | Naive interpretation | Better queueing question | Source |
| --- | --- | --- | --- |
| Upload holds 20 Mbit/s | The network is healthy | What is loaded RTT while the 20 Mbit/s link is full? | Reproducible in the `netlab` experiment below |
| No packet loss is reported | Congestion is absent | Is a deep FIFO postponing the congestion signal? | Mechanism described by Gettys, December 2010 |
| Server CPU is low | The client must be slow | Does `tc -s qdisc` show backlog on the egress bottleneck? | Linux `tc` counters |
| More buffering reduces drops | Reliability improved | How many milliseconds of serialization work did the extra bytes add? | Derived here as $q/R$ |

This is why an idle `ping` is weak evidence. An empty queue has almost no queueing delay, including a catastrophically oversized one. Jim Gettys called large buffers "dark" in his [December 3, 2010 bufferbloat write-up](https://gettys.wordpress.com/2010/12/03/introducing-the-criminal-mastermind-bufferbloat/): they become visible through their effects when they fill. The useful test is latency during a controlled load, not latency before it.

The path boundary also explains why application traces can mislead. A trace may record the time until `write()` returns, not the time until the last byte crosses the bottleneck. Even a trace that spans the full request reports the symptom without naming the queue. Pair application timing with host and path evidence.

```bash
# Read-only inspection on a Linux host. No qdisc is changed.
ip route get 203.0.113.10
ip -details link show dev eth0
tc -s -d qdisc show dev eth0
ss -tin dst 203.0.113.10

# Look for these fields rather than copying a sample value:
# tc: kind, parent/root, backlog, dropped, overlimits, requeues
# ss: rtt, rto, cwnd, bytes_sent, bytes_acked, pacing_rate
```

`overlimits` is not universally a loss counter. A shaper can increment it when traffic exceeds its configured rate and must wait. `dropped` is a drop count. `backlog` is current stored data, usually shown in bytes and packets. Interpret fields in the context of the qdisc kind and hierarchy.

## 2. Queueing delay is stored work

**Rule of thumb: convert every queue limit from packets or bytes into time at the actual bottleneck rate.**

![Stored bytes become queueing delay when divided by the bottleneck service rate](/imgs/blogs/bufferbloat-queueing-and-the-latency-you-added-yourself-2.webp)

The most useful bufferbloat equation is simple. Treat it as an explanatory serialization model, not as an equation quoted from CoDel:

$$
d_q = \frac{8q}{R}
$$

Here $d_q$ is the queueing delay in seconds, $q$ is the number of bytes already ahead of our packet, and $R$ is the bottleneck rate in bits per second. The factor 8 converts bytes to bits. This calculation assumes the rate remains fixed and ignores link-layer overhead, segmentation, and competing higher-priority traffic. Those omissions make it a clean first estimate, not a promise about an observed RTT.

For a 500 KiB backlog on a 20 Mbit/s uplink:

$$
d_q = \frac{8 \times 500 \times 1024}{20{,}000{,}000} = 0.2048\ \text{s}
$$

The packet waits about 205 ms before its own serialization begins. If request and response traffic each encounter a comparable standing queue, an RTT sample can carry both delays. Do not double the one-way number automatically. First locate which direction is saturated and where the acknowledgement or response travels.

A 1,250,000-byte FIFO at the same 20 Mbit/s rate stores:

$$
d_q = \frac{8 \times 1{,}250{,}000}{20{,}000{,}000} = 0.5\ \text{s}
$$

That number is the reason the lab below uses a 1,250,000-byte baseline FIFO. The expected loaded RTT inflation is on the order of hundreds of milliseconds, large enough to separate it from namespace and scheduler noise.

| Backlog | Bottleneck rate | Derived drain time | Source |
| ---: | ---: | ---: | --- |
| 64 KiB | 20 Mbit/s | 26.2 ms | Derived here with $8q/R$ |
| 500 KiB | 20 Mbit/s | 204.8 ms | Derived here with $8q/R$ |
| 1,250,000 bytes | 20 Mbit/s | 500 ms | Derived here with $8q/R$ |
| 1,250,000 bytes | 100 Mbit/s | 100 ms | Derived here with $8q/R$ |

The same byte limit has five times as much delay when the rate falls from 100 Mbit/s to 20 Mbit/s. That is common on variable-rate Wi-Fi and cellular links. A buffer sized in bytes cannot express one stable delay bound when its drain rate changes.

Packet limits hide another dependency. A limit of 1,000 packets represents about 1.5 MB if packets are near a 1,500-byte MTU, but only 64 KB if they are mostly 64-byte packets. Link-layer overhead and offload can complicate what a tool displays. For a first diagnosis, read both byte and packet backlog, measure the delivered rate, and calculate a range.

### A buffer is not the bandwidth-delay product

The bandwidth-delay product, or BDP, is how much data must be in flight to fill a path:

$$
\mathrm{BDP} = R \times \mathrm{RTT}
$$

For a 20 Mbit/s path with a 40 ms base RTT, the BDP is:

$$
20{,}000{,}000 \times 0.040 = 800{,}000\ \text{bits} = 100{,}000\ \text{bytes}
$$

That 100 KB includes data propagating through the path and data being serialized. It is not an instruction to add another 100 KB of standing queue at every hop. Multiple BDP-sized buffers in series can each add delay. TCP needs enough in-flight data to use the path, but useful flight can reside in propagation rather than waiting in a FIFO.

This distinction connects directly to [the latency budget](/blog/software-development/networking/the-latency-budget-speed-of-light-serialization-and-queueing). Propagation delay is mostly a property of distance and path. Serialization delay is priced by packet size and link rate. Queueing delay is load-dependent stored work, and it is the component we can often remove without moving a data center or buying a faster server.

## 3. Why utilization creates a latency cliff

**Rule of thumb: average utilization is a poor safety target when arrivals are bursty and latency matters.**

<figure class="blog-anim">
<svg viewBox="0 0 880 390" role="img" aria-label="Fixed service drains sparse arrivals, then near-saturation arrivals carry a queue and ping delay across cycles" style="width:100%;height:auto;max-width:896px">
<style>
.bbl11-bg{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.bbl11-title{font:700 18px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.bbl11-label{font:600 14px ui-monospace,SFMono-Regular,monospace;fill:var(--text-primary,#1f2937)}.bbl11-note{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}.bbl11-bulk{fill:var(--border,#d1d5db)}.bbl11-ping{fill:var(--accent,#6366f1)}.bbl11-drain{stroke:var(--text-primary,#1f2937);stroke-width:4}.bbl11-sparse{transform-origin:0 0;animation:bbl11-sparse-cycle 12s ease-in-out infinite}.bbl11-carry{transform-origin:0 0;animation:bbl11-carry-cycle 12s ease-in-out infinite}.bbl11-wait{animation:bbl11-wait-cycle 12s ease-in-out infinite}
@keyframes bbl11-sparse-cycle{0%,8%{transform:scaleY(.25)}25%,42%{transform:scaleY(.8)}58%,100%{transform:scaleY(.08)}}
@keyframes bbl11-carry-cycle{0%,8%{transform:scaleY(.2)}30%{transform:scaleY(.55)}55%{transform:scaleY(.8)}78%,100%{transform:scaleY(1)}}
@keyframes bbl11-wait-cycle{0%,35%{transform:translateX(0)}55%{transform:translateX(120px)}78%,100%{transform:translateX(245px)}}
@media (prefers-reduced-motion:reduce){.bbl11-sparse{animation:none;transform:scaleY(.08)}.bbl11-carry{animation:none;transform:scaleY(1)}.bbl11-wait{animation:none;transform:translateX(245px)}}
</style>
<text class="bbl11-title" x="26" y="34">Sparse arrivals: every burst drains</text>
<rect class="bbl11-bg" x="26" y="54" width="390" height="250" rx="12"/>
<line class="bbl11-drain" x1="70" y1="274" x2="370" y2="274"/>
<text class="bbl11-label" x="76" y="294">fixed drain rate</text>
<rect class="bbl11-bulk bbl11-sparse" x="98" y="114" width="54" height="160" rx="6"/>
<rect class="bbl11-ping" x="188" y="242" width="22" height="32" rx="5"/>
<text class="bbl11-note" x="76" y="334">queue returns to empty between bursts</text>
<text class="bbl11-title" x="462" y="34">Near saturation: delay carries forward</text>
<rect class="bbl11-bg" x="462" y="54" width="390" height="250" rx="12"/>
<line class="bbl11-drain" x1="506" y1="274" x2="806" y2="274"/>
<text class="bbl11-label" x="512" y="294">same fixed drain rate</text>
<rect class="bbl11-bulk bbl11-carry" x="532" y="94" width="54" height="180" rx="6"/>
<rect class="bbl11-bulk" x="606" y="126" width="54" height="148" rx="6"/>
<rect class="bbl11-bulk" x="680" y="158" width="54" height="116" rx="6"/>
<g class="bbl11-wait"><rect class="bbl11-ping" x="532" y="62" width="22" height="28" rx="5"/><text class="bbl11-label" x="560" y="82">ping waits</text></g>
<text class="bbl11-note" x="508" y="334">next burst arrives before the old queue empties</text>
</svg>
<figcaption>The queue fails to return to empty near saturation, so the ping's waiting time grows across cycles.</figcaption>
</figure>

If a link serves exactly 20 Mbit/s and traffic arrives at a perfectly smooth 19 Mbit/s, the queue need not grow. Real traffic is not smooth. TCP sends in windows and bursts. Applications wake on schedulers. NICs batch. Multiple flows align by accident. When a burst arrives faster than 20 Mbit/s, the queue grows. It drains only during the unused capacity after the burst.

Let $\lambda$ be mean arrival work per second, let $\mu$ be service capacity, and define utilization $\rho = \lambda/\mu$. The spare fraction is $1-\rho$. At 50 percent utilization, half the service time remains available to drain bursts. At 95 percent, only 5 percent remains. A burst that would drain quickly at moderate load can persist across many subsequent arrivals near saturation.

The classic M/M/1 queue gives a compact picture of the cliff. It assumes Poisson arrivals, exponentially distributed service times, one server, infinite waiting space, and steady state. Network traffic violates several of those assumptions, so the equation below is explicitly an explanatory model, not a prediction for a production link:

$$
W_q = \frac{\rho}{1-\rho}S
$$

$W_q$ is mean queueing time and $S = 1/\mu$ is mean service time. The multiplier $\rho/(1-\rho)$ is the lesson.

| Utilization $\rho$ | Model multiplier $\rho/(1-\rho)$ | Meaning | Source |
| ---: | ---: | --- | --- |
| 0.50 | 1 | Mean wait equals one mean service time | Derived M/M/1 illustration |
| 0.80 | 4 | Mean wait equals four mean service times | Derived M/M/1 illustration |
| 0.90 | 9 | Mean wait equals nine mean service times | Derived M/M/1 illustration |
| 0.95 | 19 | Mean wait equals nineteen mean service times | Derived M/M/1 illustration |
| 0.99 | 99 | Mean wait equals ninety-nine mean service times | Derived M/M/1 illustration |

The exact multipliers should not be pasted into a capacity plan. The shape should. As utilization approaches one, the denominator approaches zero. Small errors in demand estimates, rate estimates, or burst assumptions create large errors in waiting time.

### Why the model is useful even when its assumptions are wrong

Production packet arrivals are rarely Poisson. A single TCP sender emits paced packets or bursts shaped by its congestion window, segmentation offload, acknowledgements, and the host scheduler. Hundreds of independent flows may look smoother in aggregate, but fan-out, synchronized jobs, and request deadlines can align them again. Packet sizes are not exponentially distributed either. An acknowledgement, a DNS query, and a full-sized data segment demand different service times.

Those violations matter if we want a numeric prediction. They do not erase the conservation law underneath the model. Whenever arrivals exceed departures, backlog grows at their difference. If an application offers 30 Mbit/s to a 20 Mbit/s bottleneck for 200 ms, the excess work is approximately:

$$
q_{\mathrm{excess}} = \frac{(30-20)\times 10^6 \times 0.2}{8} = 250{,}000\ \text{bytes}
$$

Draining that excess through the 20 Mbit/s link takes another:

$$
d_{\mathrm{drain}} = \frac{8\times 250{,}000}{20\times 10^6} = 0.1\ \text{s}
$$

The burst lasted 200 ms, but its latency footprint persists for 100 ms after arrivals return below capacity. If another burst begins inside that drain window, the queue starts from a nonzero base. This carry-over is how individually reasonable bursts become a standing queue.

The arithmetic also explains why average rate can lie. Consider a repeating one-second cycle with 30 Mbit/s offered for 600 ms and nothing for 400 ms. The average offered load is 18 Mbit/s, or 90 percent of a 20 Mbit/s link. During the active portion, backlog grows by 750,000 bytes. The idle portion can drain 1,000,000 bytes, so the queue can empty before the next cycle in this simplified model. Change the burst to 700 ms and idle to 300 ms. Average offered load becomes 21 Mbit/s, and each cycle adds a net 125,000 bytes. A finite drop-tail queue must eventually fill.

This is why a utilization alert averaged over five minutes cannot certify interactive latency. The queue responds on packet, millisecond, and RTT timescales. Capacity planning needs the distribution and correlation of arrivals, not only their long-window mean.

### Little's Law gives a second consistency check

Little's Law states that, for a stable system over a suitable observation interval, mean items in the system equal arrival rate times mean time in the system. Applied carefully to a queue:

$$
L_q = \lambda W_q
$$

$L_q$ is mean queued packets or bytes, $\lambda$ is the corresponding mean arrival rate, and $W_q$ is mean queueing time. This is another explanatory model for checking measurements. If a queue reports a mean backlog near 250 KB while departing near 20 Mbit/s, the implied mean wait is about 100 ms. A loaded RTT increase in that neighborhood is consistent. A 5 ms increase is not, so either the queue snapshot was not representative, the probe bypassed that queue, offload changed the accounting, or a unit was misread.

Do not apply one instantaneous `backlog` sample as though it were a time average. Sample backlog and latency across the same interval. Match bytes with bytes per second or packets with packets per second. Little's Law is a consistency tool, not a replacement for locating the queue.

TCP makes the story more interesting. A long-lived congestion-controlled flow tends to probe for more capacity until it receives a congestion signal. With a shallow, well-managed queue, the signal arrives while delay is still bounded. With a deep drop-tail queue, the sender can keep increasing its in-flight window while packets accumulate. Loss arrives only when the oversized queue finally fills. The queue has converted prompt feedback into delayed feedback.

That is why "zero drops" is not a sufficient objective. Congestion control requires a timely signal. The signal can be a packet drop or an Explicit Congestion Notification mark when both network and endpoint support it. A queue that refuses to signal until it has stored half a second of work protects packets at the expense of every latency-sensitive user sharing the bottleneck.

> The queue is not spare bandwidth. It is a clock that charges every packet behind the bytes already stored.

## 4. Detect bloat with latency under load

**Rule of thumb: compare the same latency probe at idle and during a bounded saturation test.**

An idle baseline answers whether the path is reachable and provides a base RTT distribution. A loaded probe asks a different question: what does a small packet experience when bulk traffic fills the suspected bottleneck?

Use at least three observations together:

1. A bounded bulk transfer saturates one direction.
2. A small, regular probe measures loaded RTT through that direction.
3. The queue owner reports backlog, drops, marks, or overlimit activity.

`ping` is useful here, but only as one piece of the experiment. It does not prove application health, symmetric routing, or TCP behavior. It sends ICMP, which a path may rate-limit or prioritize differently. A loaded `ping` that inflates in step with qdisc backlog is strong local evidence. A loaded `ping` that stays flat does not prove every application flow is safe.

```bash
# Terminal 1: record an idle baseline.
ip netns exec c ping -n -i 0.1 -c 100 10.77.0.2 | tee /tmp/ping-idle.txt

# Terminal 2: create a bounded 20-second upload.
ip netns exec s iperf3 -s -1
ip netns exec c iperf3 -c 10.77.0.2 -t 20 -P 1

# Terminal 3 while the upload runs: record loaded latency and queue state.
ip netns exec c ping -n -i 0.1 -c 100 10.77.0.2 | tee /tmp/ping-loaded.txt
ip netns exec c tc -s -d qdisc show dev c0
```

Read the final `rtt min/avg/max/mdev` line in each ping file. Compare median or percentiles when your tooling records individual samples; do not rely only on a maximum. In `tc`, read `backlog`, `dropped`, `overlimits`, and any qdisc-specific fields. Sample `tc` several times during the transfer because backlog can be zero immediately after the load ends.

An application-level companion probe is valuable. A small HTTP request over the same destination and route tests the protocol the user actually cares about. Keep the payload small so response transfer time does not dominate.

```bash
for i in $(seq 1 50); do
  ip netns exec c curl -sS -o /dev/null \
    -w '%{time_connect} %{time_starttransfer} %{time_total}\n' \
    http://10.77.0.2:8080/echo
  sleep 0.1
done | tee /tmp/http-loaded.txt
```

`time_connect` includes TCP connection establishment. `time_starttransfer` includes connection time plus request handling and time to first response byte. `time_total` includes the body. If all three rise together while server work stays flat and egress backlog grows, queueing is a stronger explanation than application compute.

### Separate location from symptom

The hardest part is often not proving that loaded latency exists. It is locating the queue. Shape the traffic immediately before an opaque downstream bottleneck at a slightly lower rate, then observe whether loaded latency collapses. If it does, the downstream device probably held the harmful queue and the upstream shaper moved queue ownership to a discipline you can control.

Do not perform that experiment casually on a production interface. A wrong shaping rate changes capacity for every flow. In production, begin with read-only inspection, a maintenance window, a rollback command, and an interface identified from `ip route get`. The `Run it yourself` section uses only the isolated `netlab` namespace.

## 5. CoDel controls delay, not depth

**Rule of thumb: manage a queue by the time packets spend inside it, not only by how many packets it can hold.**

![CoDel distinguishes a temporary burst from a persistent queue by minimum sojourn time](/imgs/blogs/bufferbloat-queueing-and-the-latency-you-added-yourself-4.webp)

Controlled Delay, or CoDel, timestamps packets when they enter a queue and measures their sojourn time when they leave. Sojourn time is the packet's residence time in that queue. That gives CoDel a rate-independent unit: milliseconds of waiting rather than a byte or packet count whose time meaning changes with link rate.

CoDel does not drop every packet that exceeds a delay target. Bursts are normal and useful. Instead, it tracks the minimum sojourn time over an interval. If even one packet gets through below the target, the queue demonstrated that it drained sufficiently during that interval. If the minimum remains above target for the interval, the queue is persistent.

[RFC 8289, published in January 2018](https://www.rfc-editor.org/rfc/rfc8289.html), describes the algorithm and its default terrestrial Internet parameters. The default target is 5 ms and the interval is 100 ms. Those numbers are cited defaults, not universal constants. RFC 8290 notes that the target should be at least one MTU serialization time at the prevalent egress rate. At 1 Mbit/s, a 1,500-byte packet alone takes about 12 ms to serialize before link-layer overhead, so a 5 ms target cannot be physically achieved for such a packet.

The state machine is easier to reason about than the reputation of the algorithm suggests:

1. Enqueue the packet and record its arrival time.
2. On dequeue, calculate current sojourn time.
3. Track the minimum sojourn time during the interval.
4. If the minimum stays above target for a full interval and more than an MTU remains queued, enter dropping state.
5. Drop or mark according to the control law, giving responsive senders a congestion signal.
6. Leave dropping state when sojourn time falls below target.

The control law spaces later drops approximately with the inverse square root of the drop count. The purpose is not punishment. It is to increase the signal rate smoothly until responsive senders reduce offered load enough for the standing queue to clear.

### Walk the state machine with two queues

Take a transient burst first. Five packets arrive together because an application woke and wrote a small response. The first packet leaves quickly, later packets see increasing sojourn, and the last packet might exceed the 5 ms target. CoDel does not conclude that the queue is bad from that one sample. Before a full interval passes, the queue drains and another packet records a sojourn below target. The minimum proves that the queue was transient, so CoDel stays out of dropping state.

Now take a long bulk upload. Every dequeued packet has already waited more than target. The minimum cannot fall because the queue never approaches empty. After that condition persists for the interval, CoDel enters dropping state and signals a sender. It then waits long enough for endpoints to react. If the minimum falls below target, control stops. If the standing queue persists, later signals become more frequent according to the control law.

This minimum-over-interval test is the core insight. An average would be polluted by a few extremely delayed packets. An instantaneous sample would mistake normal bursts for persistent congestion. The minimum asks whether the queue demonstrated even one sufficiently low-delay departure during the observation window.

The dequeue location also matters. Sojourn is measured when the packet is about to consume service, which makes it the waiting time the queue actually imposed. A byte limit can be identical on two links and mean radically different delay. The timestamp follows the experienced time directly.

### A drop can be the successful outcome

Operators sometimes see the `dropped` counter increase after enabling CoDel and declare regression. That conclusion confuses packet preservation with service quality. A responsive TCP sender interprets a loss or ECN mark as congestion information and reduces its sending behavior. One early signal can prevent hundreds of later packets from spending excessive time in a bloated queue.

The evaluation must pair signals with outcomes:

- Did loaded latency fall?
- Did useful goodput remain near the bottleneck rate?
- Did retransmission or application error rates remain acceptable?
- Did ECN-capable flows receive marks while non-ECN flows received drops as expected?
- Did a nonresponsive flow continue to overwhelm the link?

The last case is a policy boundary. CoDel supplies congestion signals. It cannot guarantee that every sender obeys them.

| Mechanism | Drop-tail FIFO | CoDel | Source |
| --- | --- | --- | --- |
| Congestion signal | Queue reaches byte or packet limit | Minimum sojourn remains above target for interval | RFC 8289, January 2018 |
| Burst treatment | Accepted until limit | Accepted if queue demonstrates a low-sojourn packet | RFC 8289, January 2018 |
| Main control unit | Packets or bytes | Time | RFC 8289, January 2018 |
| Link-rate sensitivity | One byte limit implies different delay at each rate | Sojourn directly observes waiting time | Derived mechanism plus RFC 8289 |
| Flow isolation | None | None by CoDel alone | RFC 8289 and RFC 8290 |

CoDel alone still has one FIFO. A sparse SSH packet can arrive behind bulk packets already in that FIFO. CoDel bounds persistent delay by signaling the senders, but it does not schedule the SSH packet around the bulk flow. That second problem belongs to flow queueing.

```bash
# Lab-only examples on the canonical netlab interface.
ip netns exec c tc qdisc replace dev c0 root codel \
  target 5ms interval 100ms limit 1000

ip netns exec c tc -s -d qdisc show dev c0
```

The upstream [`tc-codel(8)` manual](https://man7.org/linux/man-pages/man8/tc-codel.8.html) names `ldelay`, `count`, `lastcount`, `drop_next`, `ecn_mark`, and `drop_overlimit` among the detailed fields an implementation may expose. Field availability and formatting vary with iproute2 and kernel version, so record both versions before automating a parser.

## 6. FQ-CoDel stops one flow from owning the queue

**Rule of thumb: active queue management controls standing delay; flow queueing controls who waits behind whom.**

FQ-CoDel combines two mechanisms. First, it hashes packets into internal flow queues, normally using the five-tuple. Second, it runs a CoDel instance for each queue and schedules active queues with a modified deficit round-robin policy.

[RFC 8290, published in January 2018](https://www.rfc-editor.org/rfc/rfc8290.html), calls it flow queueing rather than perfect fair queueing. Hashing can collide, so two flows can share a bucket. The scheduler also treats sparse flows differently from queues that remain active across rounds. A small DNS, SSH, or acknowledgement flow often empties and returns as a new queue, which lets it receive service promptly without a manual application priority rule.

This is isolation, not magic bandwidth. A 20 Mbit/s bottleneck still transmits no more than 20 Mbit/s. FQ-CoDel changes the order and feedback timing so one queue-building flow does not force every sparse flow to sit behind its entire backlog.

### Follow one scheduling round

Imagine three active buckets. Bucket A contains a long upload with dozens of full-sized packets. Bucket B receives one small SSH packet. Bucket C contains a short DNS exchange. A single FIFO would place B and C wherever they happened to arrive behind A. The scheduler cannot see that their service demand is tiny because there is only one line.

FQ-CoDel hashes the packets into separate buckets. Its byte-based deficit scheduler grants each active bucket a quantum of service credit per round. A bucket can send packets while it has credit, and the packet sizes subtract from that credit. Byte accounting avoids giving a flow more link time merely because it splits the same bytes into more packets.

Sparse buckets that empty are treated as new when traffic returns. They tend to receive service before continuously backlogged old buckets, with safeguards against starving the old list. B and C can therefore reach the link after a scheduling decision rather than after all of A's stored packets. CoDel independently controls persistent delay inside A's bucket.

There are two distinct wins. Scheduling limits head-of-line blocking across flows immediately. CoDel changes the longer feedback loop for flows that keep building their own queues. Calling both effects "lower latency" hides the mechanism and makes debugging harder.

### Where the abstraction leaks

![FQ-CoDel hashes traffic into flow queues, schedules them, and runs CoDel per queue](/imgs/blogs/bufferbloat-queueing-and-the-latency-you-added-yourself-5.webp)

FQ-CoDel does not allocate an exact queue object keyed by every possible flow forever. It hashes flow identifiers into a finite number of buckets. Collisions mean unrelated flows can share a FIFO and lose some isolation. The probability depends on active-flow count, bucket count, hash behavior, and traffic identity. Do not turn the 1,024 documented default into a promise that the first 1,024 flows never collide.

Encapsulation creates a different problem. If many inner conversations share one encrypted outer five-tuple, the qdisc may see a single flow. Conversely, a single user who opens many outer flows may obtain many scheduling opportunities. These are reasons to align classification with the fairness boundary, not reasons to abandon active queue management.

The Linux implementation described by RFC 8290 used 1,024 queues by default. Current [`tc-fq_codel(8)` documentation](https://man7.org/linux/man-pages/man8/tc-fq_codel.8.html) also documents defaults including a 5 ms target, 100 ms interval, 1,514-byte quantum, and ECN enabled. Treat installed documentation and `tc -d qdisc show` as authoritative for the system you operate because defaults and available parameters can change across versions.

```bash
ip netns exec c tc qdisc replace dev c0 root fq_codel
ip netns exec c tc -s -d qdisc show dev c0

# Relevant output fields include:
# limit, flows, quantum, target, interval, ecn
# backlog, dropped, overlimits, new_flow_count, ecn_mark
# new_flows_len, old_flows_len
```

Five-tuple hashing does not represent every fairness policy. One user can open many flows. Many users can be hidden inside one encrypted tunnel. RFC 8290 explicitly discusses opaque encapsulation: flows inside a tunnel may appear as one outer flow and share one queue. Per-flow fairness is not per-customer fairness, per-tenant fairness, or a hard priority guarantee.

Flow queueing also does not create the bottleneck in the place where you installed it. If a 1 Gbit/s Ethernet interface feeds a 20 Mbit/s modem, attaching FQ-CoDel directly to the 1 Gbit/s interface may let the kernel empty into the modem faster than the modem drains. The harmful queue then lives downstream, outside FQ-CoDel's control.

The operational pattern is to shape slightly below the true downstream rate so the controllable qdisc becomes the bottleneck. This is why smart queue management configurations combine a shaper with an AQM. The exact percentage is access-technology and workload dependent. Do not copy a universal 90 or 95 percent rule without measurement, especially on variable-rate links.

## 7. `tc qdisc` in practice

**Rule of thumb: inspect the full qdisc tree, identify the real bottleneck, and preserve a rollback command before changing anything.**

Linux traffic control separates queueing disciplines, classes, and filters. A classful root such as HTB can shape traffic into a class. A child qdisc such as `fq_codel` then decides how packets wait and leave within that shaped rate.

```bash
# This mutates only netlab namespace c and interface c0.
RATE_MBIT=20

ip netns exec c tc qdisc replace dev c0 root handle 1: htb default 10
ip netns exec c tc class replace dev c0 parent 1: classid 1:10 \
  htb rate "${RATE_MBIT}mbit" ceil "${RATE_MBIT}mbit" burst 32k
ip netns exec c tc qdisc replace dev c0 parent 1:10 handle 10: fq_codel \
  target 5ms interval 100ms

ip netns exec c tc -s -d qdisc show dev c0
ip netns exec c tc -s -d class show dev c0
```

The root HTB class owns the 20 Mbit/s service rate in this lab. FQ-CoDel owns the waiting policy below that class. On a real multiqueue NIC, the root may be `mq` and the active disciplines may appear on leaf queues. Hardware offload and driver rings can place additional waiting below what a software qdisc reports. Inspect `ethtool -g`, offload state, and device-specific counters when software backlog does not explain loaded latency.

### Replacing versus adding

`tc qdisc add` fails when a qdisc already exists at the specified attachment point. `replace` is convenient and idempotent for a disposable lab. It is dangerous as an unexplained production command because it can discard a carefully constructed hierarchy. Export the existing configuration, understand ownership, and use your system's declarative network configuration where possible.

```bash
# Read-only discovery for a candidate production interface.
TARGET_IP=203.0.113.10
DEV=$(ip route get "$TARGET_IP" | awk '{for (i=1;i<=NF;i++) if ($i=="dev") print $(i+1)}')
test -n "$DEV"
printf 'candidate interface: %s\n' "$DEV"
tc -s -d qdisc show dev "$DEV"
tc -s -d class show dev "$DEV"
ethtool -i "$DEV"
uname -r
tc -Version
```

This only discovers state. Do not pipe the discovered interface into a mutation without review. Routes can change, policy routing can choose another table, and containers can expose a virtual interface whose downstream owner is elsewhere.

### Write the rollback before the change

A safe queue change has an explicit pre-change capture and a restoration path. `tc` does not provide a universal command that serializes every hierarchy into a perfectly replayable configuration. The owning network manager, boot configuration, or infrastructure code should remain the source of truth.

For a maintenance plan, record at least:

```bash
date --iso-8601=seconds
ip -details link show dev "$DEV"
ip -details route show table all
tc -s -d qdisc show dev "$DEV"
tc -s -d class show dev "$DEV"
tc -s -d filter show dev "$DEV"
```

Then write the exact restoration action for that environment. It may be reapplying a NetworkManager profile, restarting `systemd-networkd`, redeploying a declarative host configuration, or replaying a reviewed `tc` hierarchy. "Delete the root qdisc" is a valid reset for the isolated lab because the lab owns the root. It is not a general production rollback.

Change one variable per comparison. If you simultaneously alter shaping rate, qdisc kind, offloads, MTU, and congestion control, a better result teaches almost nothing. Preserve the bottleneck rate while comparing FIFO and FQ-CoDel, as the lab does. Tune rate only after queue policy is understood.

### Define acceptance before the maintenance window

A queue migration needs success and abort criteria written in the units users feel. "FQ-CoDel is installed" is a configuration assertion, not an outcome. A useful rollout record names the traffic mix, direction, offered rate, base RTT, loaded RTT percentile, delivered goodput, loss or mark counters, CPU cost, and observation duration. It also names the previous configuration and the restoration action.

Start with a canary boundary you can isolate. Exercise a bulk flow beside at least one sparse flow, then repeat with the encapsulation and MTU used in production. Test both directions if ingress and egress take different queue paths. If the service carries real-time traffic, include its native probe rather than assuming ICMP predicts jitter and playout behavior.

Abort when the change violates the predeclared service boundary, not merely when a counter becomes nonzero. Examples include sustained goodput below the required floor, application errors, CPU saturation, unexpected class starvation, or loaded latency outside the agreed range. A few deliberate AQM drops can coexist with a successful rollout. A flat drop counter can coexist with a disastrous deep queue.

Keep the observation window long enough to include normal rate changes and traffic mixtures. A five-minute canary on a fixed-rate synthetic stream cannot validate an evening Wi-Fi workload or a tunnel that appears only during backup traffic. The mechanism transfers. The exact parameters and results do not.

### Check where software accounting ends

A qdisc can report a small backlog while packets wait in a driver ring or downstream appliance. Conversely, segmentation offload can make software counters look coarser than wire packets. Compare several boundaries:

- qdisc backlog and control counters;
- interface transmit bytes, packets, drops, and errors;
- driver ring configuration and driver-specific counters;
- downstream device queue or modem telemetry when available;
- loaded latency measured across the suspected bottleneck.

If RTT inflates but the software qdisc stays empty, do not immediately lower its limit. The queue is probably elsewhere, or the probe is taking a different path. Move the observation point before moving the knob.

### Read counters as a time series

One `tc` snapshot can miss the queue. Sample during the event and keep timestamps.

```bash
DEV=c0
for i in $(seq 1 20); do
  date --iso-8601=ns
  ip netns exec c tc -s qdisc show dev "$DEV"
  sleep 0.5
done | tee /tmp/qdisc-series.txt
```

Backlog rising with loaded RTT is the cleanest signal. Drops or ECN marks under CoDel show that the controller is signaling congestion, not that the algorithm failed. The question is whether the queue remains near its intended delay while useful throughput stays acceptable.

## 8. The public story: bufferbloat became measurable

**Rule of thumb: the historical lesson is not "buffers are bad." It is that latency must be measured under load and controlled in units of time.**

![The bufferbloat story progressed from field diagnosis to CoDel and standardized FQ-CoDel](/imgs/blogs/bufferbloat-queueing-and-the-latency-you-added-yourself-6.webp)

### 8.1 Gettys, 2010 to 2011: the hidden queue becomes the suspect

Jim Gettys published ["The criminal mastermind: bufferbloat!"](https://gettys.wordpress.com/2010/12/03/introducing-the-criminal-mastermind-bufferbloat/) on December 3, 2010. He described excessive buffering in network paths and emphasized that buffers cause little trouble while empty but add severe latency when full. His examples included host transmit queues and NIC rings, not only routers.

The user-visible symptom was familiar: a bulk transfer saturated the narrowest link and other traffic became painful. The wrong first hypothesis was insufficient bandwidth. The trigger was saturation. Deep buffers were the contributing condition that converted saturation into long delay. Multiple hidden queues across hosts and access equipment multiplied the places where the symptom could arise.

Gettys' December 8, 2010 [mitigation write-up](https://gettys.wordpress.com/2010/12/08/bufferbloat-mitigations/) described shaping traffic before a problematic upstream device so the downstream buffer would not fill. That remains the operational move behind many modern smart queue management setups: move the bottleneck to a queue you control, then manage that queue.

No single 2010 blog post "discovered queueing." Queueing theory, congestion control, and active queue management predate the name. The defensible historical claim is narrower: Gettys' 2010–2011 investigation named bufferbloat, demonstrated hidden excessive buffering in common systems, and pushed latency under load into the engineering conversation.

### 8.2 Nichols and Jacobson, May 2012: control persistent delay

Kathleen Nichols and Van Jacobson published ["Controlling Queue Delay"](https://dl.acm.org/doi/10.1145/2208917.2209336) in ACM Queue, volume 10 issue 5, in May 2012. The work reframed the signal around packet sojourn time and distinguished a persistent queue from a transient burst by observing the minimum delay over an interval.

The important transfer lesson is measurement choice. Queue length in packets is not stable across packet sizes. Queue length in bytes is not stable across link rates. Sojourn time directly answers the user-facing question: how long did the queue make this packet wait?

CoDel was later documented as Experimental RFC 8289 in January 2018. The RFC preserves the algorithm, rationale, state, and control law in an implementable form. It also records limits: CoDel cannot make an unresponsive offered load fit into insufficient capacity, and parameter assumptions must respect the actual link.

### 8.3 RFC 8290, January 2018: add flow isolation

RFC 8290 documented FQ-CoDel in January 2018. Its authors were Toke Høiland-Jørgensen, Paul McKenney, Dave Täht, Jim Gettys, and Eric Dumazet. The RFC combines modified deficit round robin, sparse-flow treatment, flow hashing, and one CoDel state per queue.

The mechanism maps directly onto the original symptom. CoDel limits persistent delay. Flow queueing prevents one bulk queue from imposing its full backlog on a sparse interactive flow. The blast radius of a queue-building flow becomes smaller, subject to hash collisions, tunnels, and the fact that flow fairness may differ from the operator's desired policy.

The public record also prevents a common exaggeration. RFC 8290 is Experimental, not an Internet Standards Track mandate. It documents a widely implemented approach and its tradeoffs. It does not claim that one default qdisc solves every access link, tenant policy, tunnel, or hardware queue.

### Evidence ledger

| Field | Verified content |
| --- | --- |
| Case | Bufferbloat field diagnosis, CoDel, and FQ-CoDel |
| Event date | Gettys post on 2010-12-03; CoDel paper in May 2012; RFCs in January 2018 |
| Source | Gettys' own writing, Nichols and Jacobson's paper, RFC 8289, RFC 8290 |
| Source owner | Jim Gettys; Kathleen Nichols and Van Jacobson; IETF authors and RFC Editor |
| Mechanism | Saturated bottleneck plus deep queue creates standing delay; CoDel signals persistent delay; FQ isolates flows |
| Verified numbers | CoDel target 5 ms and interval 100 ms are RFC defaults; RFC 8290 documents 1,024 default flow queues in its Linux description |
| Transfer lesson | Measure latency under load, own the bottleneck queue, and validate both delay and throughput |

## 9. Diagnose before you tune

**Rule of thumb: do not change a qdisc until three signals agree on queueing as the cause.**

![A decision tree combines loaded latency and qdisc backlog before recommending a queue change](/imgs/blogs/bufferbloat-queueing-and-the-latency-you-added-yourself-7.webp)

Start with the symptom and narrow it using discriminating evidence.

| Idle RTT | Loaded RTT | Local backlog | Likely next move | Source |
| --- | --- | --- | --- | --- |
| Stable | Rises with upload | Rises on shaped egress | Local egress queue is likely | Reproducible diagnostic pattern |
| Stable | Rises with upload | Flat locally | Inspect downstream device and path | Reproducible diagnostic pattern |
| High | High | Flat | Investigate propagation, route, remote queue, or policy | Diagnostic inference, not a measured result |
| Stable | Stable | Flat | Queueing is not reproduced by this load | Reproducible diagnostic pattern |
| Stable | HTTP rises but ping does not | Flat | Investigate application, proxy, protocol treatment, or ICMP differences | Diagnostic inference |

Use a bounded load. Saturating a shared production uplink without coordination can cause the incident you are trying to diagnose. Prefer a lab, a dedicated test circuit, or a controlled maintenance window. Keep duration, destination, direction, and rate explicit.

### Read combinations, not isolated counters

Backlog without latency can occur when the measured probe uses another class, another hardware queue, or another path. Latency without local backlog can point downstream. Drops without backlog can reflect a hard limit, a policer, stale sampling, or active queue management doing its job. High `overlimits` on a shaper can simply mean it is enforcing its rate.

Direction is another common trap. During an upload, data packets queue on the outbound side, while ping requests and TCP data acknowledgements also need outbound service. The echo replies return in the other direction. During a download, the harmful queue may live at an ISP-facing ingress point that a host egress qdisc cannot control directly. Smart queue management often uses an intermediate functional block or equivalent ingress redirection so downloaded traffic can be shaped before a downstream device accumulates it.

Base RTT also affects interpretation. A loaded increase from 1 ms to 25 ms is a 24 ms queue and a 25-fold ratio. A path from 100 ms to 124 ms has the same added queue but a much smaller ratio. Report both the absolute delta and the distribution. Users experience milliseconds, while ratios help compare paths.

Finally, confirm that the bulk flow actually saturated the intended bottleneck. If receiver goodput reaches only 8 Mbit/s against a 20 Mbit/s shaper because of CPU, loss, or a small window, the queue may never enter the state the test intends to study.

The decision is not simply FIFO bad, FQ-CoDel good. Ask these questions:

1. Is this interface actually the bottleneck?
2. Does the qdisc see the packets before the hidden hardware or downstream queue?
3. Is per-flow isolation the desired policy?
4. Are flows hidden inside encrypted tunnels?
5. Is the rate stable enough for a fixed shaper?
6. Does the traffic use congestion control or is it unresponsive?
7. Is ECN supported and correctly handled end to end?
8. Do latency and goodput both improve under the real workload?

The broader service response belongs in [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design). Network counters should be correlated with request latency and saturation windows, not collected as a decorative dashboard. Admission control and [backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) still matter when offered work exceeds useful capacity. A qdisc manages waiting and signaling. It does not make overload disappear.

## 10. What queue management cannot promise

**Rule of thumb: every latency optimization has a throughput, fairness, or deployment boundary that must be tested.**

### AQM is not capacity planning

If an open-loop sender offers 30 Mbit/s to a 20 Mbit/s bottleneck forever, 10 Mbit/s must be delayed, dropped, marked and heeded, or discarded somewhere. CoDel can signal. It cannot force an unresponsive sender to adapt. Policing or admission control may be necessary.

### FQ-CoDel is not tenant fairness

A tenant with 100 flows can receive different treatment from a tenant with one flow. A VPN can collapse many applications into one outer five-tuple. If the policy is per-customer or per-service, classify at that boundary or use a scheduler designed for it. RFC 8290 discusses both hash collisions and opaque encapsulation.

### A software qdisc cannot control a queue below it

Driver rings, firmware, virtual switches, hypervisors, and modems can buffer after Linux dequeues a packet. Byte Queue Limits can help Linux drivers keep hardware queues from hiding excessive data, but availability and behavior depend on the driver. The diagnostic remains the same: find the point where backlog and loaded delay move together.

### A fixed shaper can waste a variable link

Shape too high and the downstream queue still owns the bottleneck. Shape too low and you leave capacity unused. Wi-Fi airtime, cellular scheduling, and some broadband technologies vary over time. A fixed rate may need conservative headroom or a rate-aware controller. Measure across the range of real link conditions.

### Low latency does not mean zero queue

A transmitter needs packets ready when it becomes available. Too little buffering or an overly aggressive target can reduce utilization, especially on slow links where one MTU already takes several milliseconds to serialize. RFC 8290 explicitly says the FQ-CoDel target should be at least one MTU transmission time at the prevalent egress rate.

### Good averages can hide tail pain

An average loaded RTT can look acceptable while periodic bursts create a painful p99. Keep individual samples, correlate them with queue state, and inspect the distribution. The senior engineering move is to state which percentile and workload a target covers.

| Choice | Improves | Can worsen | Validate with | Source |
| --- | --- | --- | --- | --- |
| Smaller drop-tail limit | Maximum queue delay | Burst tolerance and utilization | Loaded RTT, drops, goodput | Derived tradeoff |
| CoDel | Persistent queue delay | Drops or marks when senders need a signal | Sojourn fields, drops or ECN, goodput | RFC 8289 |
| FQ-CoDel | Flow isolation plus delay control | Hash collisions, per-flow policy mismatch | Per-flow workload and loaded RTT | RFC 8290 |
| Shaping below downstream rate | Queue ownership | Available throughput if set too low | Delivered rate and downstream loaded RTT | Reproducible operational pattern |
| Larger buffer | Burst absorption | Worst-case queue drain time | $8q/R$ and loaded RTT | Derived here |

## Run it yourself

### Question

At a 20 Mbit/s bottleneck, does replacing a 1,250,000-byte FIFO with FQ-CoDel reduce loaded ping latency while keeping a single TCP upload near the configured rate?

### Preconditions

Use Linux with root privileges, `iproute2`, `iputils-ping`, and `iperf3`. The commands assume the canonical `netlab` namespaces and addresses from [the series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url): namespace `c` owns `c0` and `10.77.0.1`; namespace `s` owns `s0` and `10.77.0.2`.

Run this only inside the lab. It replaces the qdisc on `c0`. `ip netns`, qdisc changes, and traffic shaping require root or equivalent capabilities. On macOS, run the lab in a privileged Linux VM such as Lima or Colima with `NET_ADMIN`.

```bash
set -euo pipefail

command -v ip >/dev/null
command -v tc >/dev/null
command -v ping >/dev/null
command -v iperf3 >/dev/null

ip netns list | grep -Eq '^c([[:space:]]|$)'
ip netns list | grep -Eq '^s([[:space:]]|$)'
ip -n c link show dev c0
ip -n s link show dev s0
ip -n c route get 10.77.0.2
ip netns exec c ping -n -c 3 10.77.0.2
tc -Version
uname -r
```

Read: confirm `c0` and `s0` are `UP`, the route selects `c0`, all three preflight pings return, and record the iproute2 and kernel versions.

Expected: on an otherwise idle local namespace pair, RTT is commonly below 2 ms. Treat that as a lab expectation, not a host guarantee. A heavily loaded or virtualized runner may be higher.

### Baseline

Install a 20 Mbit/s HTB class and a 1,250,000-byte FIFO below it. At 20 Mbit/s, the FIFO can store a derived 500 ms of serialization work.

```bash
set -euo pipefail

ip netns exec c tc qdisc replace dev c0 root handle 1: htb default 10
ip netns exec c tc class replace dev c0 parent 1: classid 1:10 \
  htb rate 20mbit ceil 20mbit burst 32k
ip netns exec c tc qdisc replace dev c0 parent 1:10 handle 10: \
  bfifo limit 1250000

ip netns exec c tc -s -d qdisc show dev c0
ip netns exec c tc -s -d class show dev c0

ip netns exec s sh -c 'exec iperf3 -s -1 >/tmp/bloat-fifo-server.txt 2>&1' &
SERVER_PID=$!
sleep 0.5
ip netns exec c sh -c \
  'iperf3 -c 10.77.0.2 -t 20 -P 1 >/tmp/bloat-fifo-iperf.txt 2>&1' &
LOAD_PID=$!
sleep 2
ip netns exec c ping -n -i 0.1 -c 100 10.77.0.2 \
  | tee /tmp/bloat-fifo-ping.txt
ip netns exec c tc -s -d qdisc show dev c0 \
  | tee /tmp/bloat-fifo-qdisc.txt
wait "$LOAD_PID"
wait "$SERVER_PID"
```

Read: in `/tmp/bloat-fifo-ping.txt`, inspect `rtt min/avg/max/mdev`. In `/tmp/bloat-fifo-qdisc.txt`, inspect the `bfifo` `backlog`, `dropped`, and parent HTB `overlimits`. In `/tmp/bloat-fifo-iperf.txt`, inspect the receiver bitrate.

Expected: the receiver bitrate should usually remain in roughly 18 to 20 Mbit/s after TCP reaches steady state. Loaded ping average or maximum should rise by hundreds of milliseconds, commonly into a 300 to 550 ms range when the FIFO stays full. Scheduler timing, offloads, TCP behavior, and virtualization can shift the result. The derived upper queue drain time is 500 ms, not a guaranteed ping value.

### Apply one change

Keep the same 20 Mbit/s HTB bottleneck. Replace only the child FIFO with FQ-CoDel.

```bash
set -euo pipefail

ip netns exec c tc qdisc replace dev c0 parent 1:10 handle 10: fq_codel \
  target 5ms interval 100ms
ip netns exec c tc -s -d qdisc show dev c0
ip netns exec c tc -s -d class show dev c0
```

Read: verify that root `1:` remains HTB at 20 Mbit/s and child `10:` is now `fq_codel`. This preserves the service rate and changes only queue management.

Expected: `tc -d` reports the installed target and interval. Exact extra fields depend on the installed kernel and iproute2.

### Compare

```bash
set -euo pipefail

ip netns exec s sh -c 'exec iperf3 -s -1 >/tmp/bloat-fq-server.txt 2>&1' &
SERVER_PID=$!
sleep 0.5
ip netns exec c sh -c \
  'iperf3 -c 10.77.0.2 -t 20 -P 1 >/tmp/bloat-fq-iperf.txt 2>&1' &
LOAD_PID=$!
sleep 2
ip netns exec c ping -n -i 0.1 -c 100 10.77.0.2 \
  | tee /tmp/bloat-fq-ping.txt
ip netns exec c tc -s -d qdisc show dev c0 \
  | tee /tmp/bloat-fq-qdisc.txt
wait "$LOAD_PID"
wait "$SERVER_PID"

printf '\nFIFO ping summary:\n'
grep '^rtt ' /tmp/bloat-fifo-ping.txt
printf '\nFQ-CoDel ping summary:\n'
grep '^rtt ' /tmp/bloat-fq-ping.txt
printf '\nFIFO receiver bitrate:\n'
grep 'receiver' /tmp/bloat-fifo-iperf.txt | tail -1
printf '\nFQ-CoDel receiver bitrate:\n'
grep 'receiver' /tmp/bloat-fq-iperf.txt | tail -1
```

Read: compare average and maximum RTT, receiver bitrate, FQ-CoDel backlog, drops, `ecn_mark`, `new_flow_count`, and the new and old flow list lengths when those fields are present.

Expected: FQ-CoDel should usually keep loaded ping in roughly the 1 to 30 ms range in this two-flow namespace experiment while the TCP receiver bitrate remains roughly 18 to 20 Mbit/s. A few drops or marks are not a failure. They are congestion signals. Repeat each treatment at least three times and compare medians if the host is noisy.

### Interpret the result without overclaiming

Four result shapes are useful:

1. **FIFO inflates, FQ-CoDel stays low, goodput is similar.** This is the expected demonstration. The queue policy caused most of the avoidable delay.
2. **Both inflate and both show backlog.** Confirm that FQ-CoDel is actually the child of the shaped class, inspect whether the ping and upload hash as expected, and look for a second queue below `c0`.
3. **Neither inflates.** Confirm the upload reached the 20 Mbit/s class and sample qdisc state while it runs. Offload or topology differences may prevent the intended queue from filling.
4. **FQ-CoDel latency improves but goodput falls materially.** Repeat runs, inspect drops and retransmissions, confirm the host is not CPU-limited, and check whether target is below one MTU serialization time for the effective rate.

The ping range is an expected `netlab` result, not a benchmark claim about Linux in general. Kernel version, iproute2 version, timer resolution, virtual machine scheduling, offloads, MTU, and CPU contention all introduce variance. Preserve raw outputs and configuration beside each run so a surprising result is diagnosable.

For a stronger comparison, run at least three repetitions per treatment, randomize treatment order when warm-up effects matter, and summarize the median loaded RTT delta and receiver bitrate. Do not discard a run merely because it contradicts the expected range. Explain the environmental cause or report that the reproduction did not match.

The experiment proves a bounded claim. On this controlled topology, changing queue management reduces the small flow's wait without materially changing the configured bottleneck. It does not prove the same numeric range for Wi-Fi, DOCSIS, cellular, a VPN tunnel, or a multiqueue production NIC.

### Reset

```bash
set -euo pipefail

ip netns exec c tc qdisc del dev c0 root 2>/dev/null || true
ip netns exec c tc qdisc show dev c0
```

The deletion is scoped to `c0` inside namespace `c`. It removes both the HTB root and its child. Do not substitute an unspecified host interface.

### Production translation

Use read-only commands first:

```bash
ip route get 203.0.113.10
tc -s -d qdisc show dev eth0
tc -s -d class show dev eth0
ip -s -s link show dev eth0
ethtool -g eth0
ss -tin dst 203.0.113.10
```

Run a load test only with an approved rate, duration, destination, and rollback. Packet captures can contain credentials, tokens, personal data, and application payloads. If a capture is necessary, filter it to the target flow and bound its duration and size.

## Key takeaways

- Queueing delay is stored transmission work. Convert backlog to time with the explanatory model $d_q = 8q/R$.
- An idle latency probe cannot expose an empty oversized buffer. Compare idle and loaded latency under a bounded transfer.
- As utilization approaches capacity, spare drain time disappears. Bursts become a standing queue.
- CoDel measures packet sojourn time and signals when minimum delay stays above target across an interval. It permits short bursts.
- FQ-CoDel adds hashed per-flow queues and scheduling, which prevents one queue-building flow from imposing its entire backlog on sparse flows in the usual case.
- The qdisc must own the real bottleneck. A faster software interface can otherwise push the harmful queue into a modem, driver, firmware, or virtual switch.
- Validate latency, goodput, drops or marks, and fairness under the workload you actually operate. No qdisc creates capacity.

The next debugging step is to connect queue state to the sender's control loop in [CUBIC, BBR, and what changing congestion control actually does](/blog/software-development/networking/congestion-control-cubic-bbr-and-what-changing-it-actually-does). The full series closes with [the senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model), which places queueing beside naming, routing, transport, security, and application timing.

## Further reading

- Jim Gettys, ["The criminal mastermind: bufferbloat!" (December 3, 2010)](https://gettys.wordpress.com/2010/12/03/introducing-the-criminal-mastermind-bufferbloat/)
- Kathleen Nichols and Van Jacobson, ["Controlling Queue Delay" (May 2012)](https://dl.acm.org/doi/10.1145/2208917.2209336)
- IETF, [RFC 8289: Controlled Delay Active Queue Management (January 2018)](https://www.rfc-editor.org/rfc/rfc8289.html)
- IETF, [RFC 8290: The Flow Queue CoDel Packet Scheduler and Active Queue Management Algorithm (January 2018)](https://www.rfc-editor.org/rfc/rfc8290.html)
- iproute2, [`tc-codel(8)`](https://man7.org/linux/man-pages/man8/tc-codel.8.html) and [`tc-fq_codel(8)`](https://man7.org/linux/man-pages/man8/tc-fq_codel.8.html)
