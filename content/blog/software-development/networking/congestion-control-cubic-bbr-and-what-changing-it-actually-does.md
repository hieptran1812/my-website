---
title: "Congestion Control: CUBIC, BBR, and What Changing It Actually Does"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn how CUBIC and BBR control a TCP sender, how to measure the trade-offs, and how to canary a controller change without mistaking a lab win for a universal result."
tags:
  [
    "networking",
    "distributed-systems",
    "tcp",
    "congestion-control",
    "cubic",
    "bbr",
    "linux",
    "performance-engineering",
    "netlab",
    "capacity-planning",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 44
image: "/imgs/blogs/congestion-control-cubic-bbr-and-what-changing-it-actually-does-1.webp"
---

A file download crosses a path with 20 Mbit/s of capacity. The receiver is healthy. The application is always ready to write. Yet the transfer crawls at a few megabits per second, and changing one Linux setting makes it jump. That looks like free bandwidth. It is not. The setting changed how the sender interprets evidence from the path, how quickly it puts bytes in flight, and how it behaves when another flow contests the same queue.

![Congestion control runs at the sender while responding to evidence from the full network path](/imgs/blogs/congestion-control-cubic-bbr-and-what-changing-it-actually-does-1.webp)

The diagram above is the mental model: congestion control is sender-side code with path-wide consequences. A congestion controller does not enlarge a physical link. It chooses a sending rate and an in-flight budget from acknowledgements, round-trip-time samples, loss, and sometimes Explicit Congestion Notification. Those choices determine whether the bottleneck is full, whether its queue stays occupied, and which flow yields when capacity becomes scarce.

This distinction matters because changing from CUBIC to BBR is unusually easy on Linux. The operational temptation is to turn an interesting experiment into a fleet-wide default. The safe move is the opposite: state the workload, name the controller version, preserve a rollback path, and inspect transport evidence before widening the canary.

This post continues the path built in [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). It assumes the reliability machinery from [sequence numbers, ACKs, retransmits, and RTO](/blog/software-development/networking/reliability-sequence-numbers-acks-retransmits-and-rto) and keeps receive-side limits separate using [flow control versus congestion control](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe). The [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) is the series capstone.

## Congestion control lives at the sender, but prices the path

**Rule of thumb: if you cannot say which sender controls the bytes, you are not ready to tune its controller.**

TCP has at least three limits that engineers routinely collapse into one:

- The congestion window, `cwnd`, is the sender's estimate of how much unacknowledged data the network can safely carry.
- The receive window, `rwnd`, is the receiver's advertised capacity to accept data.
- The application can be unable or unwilling to fill either window.

The useful approximation is:

$$
\text{inflight} \leq \min(\text{cwnd}, \text{rwnd})
$$

This is an explanatory bound, not an equation that fully specifies Linux TCP. Pacing, segmentation offload, ACK behavior, recovery state, and application writes all affect what is actually emitted. It is still the right first model. If `rwnd` is smaller, changing CUBIC to BBR does not repair flow control. If the application writes 2 Mbit/s, neither controller can manufacture 20 Mbit/s of demand.

The sender sees the network through returned evidence. An ACK says that some bytes left the network successfully. The spacing of ACKs can produce a delivery-rate sample. The elapsed time produces an RTT sample. Duplicate ACKs, RACK loss detection, retransmission timeouts, or ECN marks report that some part of the operating point was unsafe. None of those signals names the bottleneck router. They summarize the path as observed by this connection.

That is why congestion control is not a routing algorithm. A route chooses where packets go. A congestion controller decides how aggressively one transport uses the chosen route. Changing it does not change DNS, the next hop, an L4 load balancer decision, or a backend. It can nevertheless change request latency because queues and retransmissions sit beneath all of those layers.

The practical inspection sequence starts read-only:

```bash
uname -r
sysctl net.ipv4.tcp_congestion_control
sysctl net.ipv4.tcp_available_congestion_control
sysctl net.ipv4.tcp_allowed_congestion_control
tc -s qdisc show
ss -tin '( dport = :443 or sport = :443 )'
```

Read the controller name, `cwnd`, `rtt`, `rttvar`, `retrans`, `delivery_rate`, and `pacing_rate` from `ss -tin`. For BBR on a sufficiently recent `iproute2`, also look for `bbr:(bw:...,mrtt:...,pacing_gain:...,cwnd_gain:...)`. The [Google BBR FAQ](https://github.com/google/bbr/blob/master/Documentation/bbr-faq.md) documents those BBR fields and warns that a sender-side `netem` setup can distort experiments through TCP Small Queues and offload behavior.

> A congestion-control change is not a capacity upgrade. It is a new policy for spending evidence about capacity.

## The pipe, the queue, and the bandwidth-delay product

Before comparing algorithms, price the path. The bandwidth-delay product, or BDP, is the amount of data required in flight to keep a path busy when there is no queue:

$$
\text{BDP} = \text{bottleneck bandwidth} \times \text{minimum RTT}
$$

For the lab in this post, the configured bottleneck is 20 Mbit/s and the target RTT is 80 ms. The derived BDP is:

$$
20\ \text{Mbit/s} \times 0.080\ \text{s} = 1.6\ \text{Mbit} = 200\ \text{kB}
$$

With a 1,460-byte TCP payload per full-sized Ethernet packet, that is approximately:

$$
\frac{200{,}000\ \text{bytes}}{1{,}460\ \text{bytes/segment}} \approx 137\ \text{segments}
$$

Both results are derived approximations. The real maximum segment size depends on headers and path MTU, while Linux can represent and emit data through larger offloaded sk_buffs. The calculation answers the architectural question: a sender needs roughly 200 kB acknowledged per 80 ms feedback cycle to sustain the configured ceiling.

Now add a queue. If the sender keeps 400 kB in flight while the propagation pipe holds 200 kB, the remaining 200 kB waits at the bottleneck. At 20 Mbit/s, that queued data adds another 80 ms:

$$
\text{queue delay} = \frac{200{,}000 \times 8}{20{,}000{,}000} = 0.080\ \text{s}
$$

This is the core tension. Too little in flight leaves capacity idle. Too much creates delay and eventually loss. A congestion controller searches for an operating point without direct access to the link's configured rate or buffer depth.

| Quantity | Value | Interpretation | Source |
| --- | ---: | --- | --- |
| Bottleneck rate | 20 Mbit/s | Lab shaping ceiling | `netlab` configuration in this post, 2026-09-29 |
| Minimum RTT target | 80 ms | Two 40 ms one-way delay legs | `netlab` configuration in this post, 2026-09-29 |
| BDP | 200 kB | Rate multiplied by RTT | Derived here from the preceding formula |
| Full-sized segments per BDP | About 137 | Uses a 1,460-byte payload assumption | Derived here; confirm actual `advmss` with `ss -ti` |
| Extra delay from one queued BDP | 80 ms | Queue bytes divided by bottleneck rate | Derived here from the preceding formula |

Do not infer the propagation RTT from a loaded `ping`. The loaded observation includes queue delay. Measure the quiet-path floor, then watch how RTT changes during transfer. The difference is the queue signal an operator can see, even though each controller uses a more specific internal model.

## CUBIC: remember the last cliff, then approach it carefully

![CUBIC reduces its window after congestion, pauses near the previous maximum, and probes beyond it](/imgs/blogs/congestion-control-cubic-bbr-and-what-changing-it-actually-does-2.webp)

**Rule of thumb: CUBIC searches in window space, and a congestion event tells it where the previous search became unsafe.**

CUBIC is loss-based in the broad operational sense, but that phrase should not be mistaken for "it ignores everything except dropped packets." Linux recovery can react to loss detected by modern machinery, and CUBIC can also respond to ECN. The defining feature is that the controller grows a congestion window until congestion evidence causes a multiplicative reduction.

[RFC 9438, published in August 2023](https://www.rfc-editor.org/rfc/rfc9438.html), specifies CUBIC as a Standards Track algorithm and records its adoption in Linux, Windows, and Apple stacks. Its window target follows a cubic function of elapsed time since the congestion epoch:

$$
W_{\text{cubic}}(t) = C(t-K)^3 + W_{\max}
$$

Here, $W_{\max}$ is the remembered window near the previous congestion event, $C$ controls the curve's aggressiveness, $t$ is elapsed time in seconds, and $K$ places the curve's plateau at the remembered maximum. This is the RFC's algorithmic form, not an explanatory invention.

After congestion, CUBIC reduces the window. RFC 9438 recommends a multiplicative decrease factor $\beta_{\text{cubic}}$ of 0.7. If the relevant flight size is 200 segments, the immediate target is roughly 140 segments:

$$
200 \times 0.7 = 140\ \text{segments}
$$

The controller then uses the concave portion of the cubic curve to approach the old maximum. Growth slows near $W_{\max}$ because that region was recently close to saturation. If the path accepts the traffic, the curve crosses the old maximum and becomes convex, probing more quickly for newly available capacity.

That shape is more useful than the shorthand "sawtooth." Reno's additive increase creates a fairly literal linear ramp followed by a drop. CUBIC's time-based curve includes a broad plateau around the remembered operating point. On a graph of `cwnd` over time, loss still produces teeth, but the rising edge is not a straight line.

### Why elapsed time matters

Classic ACK-clocked additive increase gives short-RTT flows more opportunities to increase each second. CUBIC bases its primary curve on wall-clock time, reducing that dependency and scaling better on high-BDP paths. RFC 9438 also includes a Reno-friendly region so CUBIC does not become needlessly timid where Reno already performs well.

The second-order effect is important: loss must occur before a purely loss-driven sender learns that it crossed the safe point. If the bottleneck has a deep drop-tail buffer, the sender can fill that buffer first. Goodput looks excellent while RTT rises. This is the mechanism behind the next post on [bufferbloat, queueing, and the latency you added yourself](/blog/software-development/networking/bufferbloat-queueing-and-the-latency-you-added-yourself).

### Random loss is not always congestion

Wireless corruption, policing, transient bursts, and an overloaded middlebox can drop a packet without a persistent bottleneck queue. CUBIC cannot read the cause printed on a dropped packet. It sees a congestion event and reduces its window. Repeated random loss can therefore keep a large-BDP flow below the path's physical capacity.

This does not make CUBIC broken. Its behavior is conservative and widely understood, and the feedback is compatible with other loss-based senders. The useful statement is conditional: on a path where non-congestive random loss is material, a controller that models delivered bandwidth and propagation RTT may retain more goodput. On a shallow, clean, low-RTT path, the difference can be small. On a shared queue, coexistence can dominate both single-flow results.

## BBR: build a model, then probe whether it is still true

**Rule of thumb: BBR tries to operate near the path's estimated BDP, but the estimate is a moving control input, not an oracle.**

BBR expands to Bottleneck Bandwidth and Round-trip propagation time. The original [Google paper from September to October 2016](https://research.google/pubs/bbr-congestion-based-congestion-control-2/) describes a controller that estimates two properties:

- `BtlBw`, the maximum recently observed delivery rate that appears sustainable.
- `RTprop`, the minimum recently observed RTT, used as an estimate of propagation time without a standing queue.

Their product estimates the pipe:

$$
\widehat{\text{BDP}} = \widehat{\text{BtlBw}} \times \widehat{\text{RTprop}}
$$

This is a model. ACK aggregation can inflate a delivery-rate sample. Competing traffic can hide available bandwidth. Route changes can make an old minimum RTT stale. Receiver or application limits can make the sender appear path-limited. BBR therefore cannot simply measure once and hold a constant rate.

Mainline Linux currently carries the original BBR implementation commonly called BBRv1. Its source names four modes: `STARTUP`, `DRAIN`, `PROBE_BW`, and `PROBE_RTT`. STARTUP raises the pacing rate quickly to find the pipe. DRAIN removes the queue created by that search. PROBE_BW varies pacing around the bandwidth estimate to look for more capacity and to yield some pressure. PROBE_RTT periodically cuts inflight so that a fresh low-delay RTT sample can emerge. The [mainline Linux source](https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_bbr.c) is the source of truth for the implementation actually selected by `TCP_CONGESTION=bbr` on an unmodified mainline kernel.

### How an ACK becomes a bandwidth sample

BBR does not ask a router for its line rate. It estimates delivery from the sender's own bookkeeping. When a packet leaves, TCP records how many bytes had already been delivered and a timestamp. When that packet is acknowledged, the sender can calculate how many additional bytes were delivered during the interval. In simplified explanatory form:

$$
\text{delivery-rate sample} = \frac{\text{newly delivered bytes}}{\text{sampling interval}}
$$

The exact Linux rate-sampling machinery handles details such as retransmissions, application-limited periods, and which elapsed interval constrains the observation. The approximation explains the core idea: ACKs acknowledge bytes and reveal a delivery pace. BBR filters recent samples to estimate the highest bandwidth the path has demonstrated.

That estimate can be wrong in both directions. If the application stops writing, delivery falls even though the path still has capacity. TCP marks such a sample application-limited so it does not automatically drag the bandwidth estimate down. If a receiver or network device releases ACKs in a compressed burst, the apparent delivery rate can exceed the bottleneck's steady service rate. BBR's filtering and aggregation model try to separate useful evidence from artifacts, but no sender-only estimator has perfect visibility.

The minimum RTT filter solves a different problem. Most RTT samples contain propagation, transmission, and whatever queueing happened during that exchange:

$$
\text{sample RTT} = \text{propagation RTT} + \text{serialization} + \text{queueing} + \text{host delay}
$$

This is an explanatory decomposition. Taking the minimum over a window attempts to find a sample with little or no queueing. Mainline BBRv1 uses a 10-second minimum-RTT window in its source. The July 2026 BBRv3 draft also specifies a 10-second `MinRTTFilterLen`, while changing much of the surrounding model and probe machinery. The matching duration does not make the algorithms identical.

Suppose recent delivery samples peak near 18 Mbit/s and the minimum RTT is 82 ms. The model's derived BDP estimate is:

$$
18\ \text{Mbit/s} \times 0.082\ \text{s} = 1.476\ \text{Mbit} \approx 184.5\ \text{kB}
$$

If ordinary RTT samples rise to 160 ms while the bandwidth estimate remains near 18 Mbit/s, roughly 78 ms of the difference is consistent with queueing and host variance. At 18 Mbit/s, 78 ms corresponds to approximately 175.5 kB of extra in-flight data:

$$
18\ \text{Mbit/s} \times 0.078\ \text{s} \div 8 \approx 175.5\ \text{kB}
$$

Those are derived diagnostics, not claims about a capture. They show how to test whether the controller's visible model is plausible against the path you configured.

### Pacing rate and congestion window do different jobs

BBR is sometimes described as rate-based, which can obscure the fact that it still maintains a congestion window. The pacing rate controls the time distribution of packet departures. The congestion window caps the volume allowed in flight. BBR uses gains around its bandwidth and BDP estimates to set both.

In STARTUP, mainline BBRv1 uses a high pacing gain intended to double the sending rate each round trip while delivery continues growing. Once bandwidth growth no longer meets the implementation's full-pipe test, BBR enters DRAIN and sends below the estimated bottleneck rate so excess inflight can leave. In PROBE_BW it cycles pacing gains around the estimate. One phase probes above the estimate, another drains below it, and the remaining phases cruise near it. This deliberate oscillation is how the sender asks whether more capacity appeared.

PROBE_RTT answers a separate identifiability problem. If a standing queue never drains, the sender cannot observe propagation RTT directly. Mainline BBRv1 therefore reduces inflight briefly after its minimum-RTT estimate becomes stale. That can create a visible throughput notch in a long flow. The notch is model maintenance, not necessarily a path outage.

This gives three independent questions when reading `ss -tin`:

1. Is `delivery_rate` close to application goodput over a comparable interval?
2. Is `pacing_rate` temporarily probing, steadily overshooting, or constrained below the useful rate?
3. Is `cwnd` large enough for the modeled BDP but bounded enough to avoid a persistent queue?

One snapshot rarely answers them. Sample the socket through several RTTs and at least one whole probe cycle. Also retain application timing because a healthy transport model cannot make an idle writer busy.

The current [BBRv3 Internet-Draft, version 06 dated July 2026](https://datatracker.ietf.org/doc/draft-ietf-ccwg-bbr/) is a different and much more detailed algorithm. It incorporates loss, ECN, short-term and long-term inflight bounds, aggregation, offload budgets, and explicit Reno/CUBIC coexistence behavior. It is an active Internet-Draft, not an Internet Standard. A mainline host that reports `bbr` should not be described as running every behavior from that draft.

The version distinction changes incident response. A graph labeled only `BBR` is insufficient evidence when a regression may involve loss caps, probe timing, or coexistence logic. Record at least the kernel release, source lineage, controller string, qdisc, and whether the service uses TCP or QUIC. A vendor kernel can backport or replace controller code without changing the marketing name. QUIC implements congestion control in user space and does not inherit the host TCP sysctl.

Treat the implementation identity like a database engine version. `PostgreSQL` alone would not be enough for a query-planner incident. `BBR` alone is not enough for a control-loop incident.

| Question | CUBIC | Mainline Linux BBRv1 | BBRv3 draft | Source |
| --- | --- | --- | --- | --- |
| Primary search coordinate | Congestion window over time | Delivery-rate and minimum-RTT model | Bandwidth, RTT, loss, ECN, and inflight model | RFC 9438; Linux `tcp_bbr.c`; draft-ietf-ccwg-bbr-06 |
| Stable operation | Approach and pass remembered window | Pace around estimated bandwidth | Cycle through bounded probe tactics | Same sources |
| Response to ordinary random loss | Multiplicative window reduction | Loss is not the primary steady-state signal | Loss updates short and long-term bounds | Same sources |
| Standardization status in 2026 | IETF Standards Track | Shipping Linux implementation | Active Internet-Draft | Source documents dated above |

### Pacing is part of the mechanism

`cwnd` limits volume in flight. Pacing controls when packets leave. BBR depends heavily on pacing because a sender that emits a BDP in one burst does not behave like one that spaces the same bytes across an RTT. The Linux BBR source notes that BBR can use the `fq` qdisc for pacing; otherwise the TCP stack falls back to one high-resolution timer per socket and may consume more resources.

Inspect both layers:

```bash
sysctl -n net.ipv4.tcp_congestion_control
tc qdisc show dev eth0
ss -tin dst 203.0.113.10
```

Do not change `eth0` because a blog post named it. Discover the real device with `ip route get <destination>`, then treat a qdisc mutation as a separate change with separate rollback. Controller and qdisc changes can interact, so changing both during one production canary destroys causal clarity.

## Two controllers, two control loops

<figure class="blog-anim">
<svg viewBox="0 0 900 520" role="img" aria-label="Aligned CUBIC and mainline Linux BBRv1 control loops animate over the same interval" style="width:100%;height:auto;max-width:900px">
<style>
.cc-bg{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}.cc-axis{stroke:var(--text-secondary,#6b7280);stroke-width:1.5}.cc-grid{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:5 6}.cc-path{fill:none;stroke:var(--text-primary,#1f2937);stroke-width:3}.cc-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.cc-small{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}.cc-phase{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db)}.cc-dot{fill:var(--accent,#6366f1)}.cc-active{fill:var(--accent,#6366f1);opacity:.18}
@keyframes cc-cubic{0%{transform:translate(0,0)}23%{transform:translate(184px,-62px)}25%{transform:translate(200px,42px)}48%{transform:translate(384px,-68px)}50%{transform:translate(400px,38px)}73%{transform:translate(584px,-72px)}75%{transform:translate(600px,34px)}100%{transform:translate(800px,-78px)}}
@keyframes cc-bbr{0%{transform:translate(0,0)}16%{transform:translate(128px,-96px)}29%{transform:translate(232px,-45px)}45%{transform:translate(360px,-92px)}63%{transform:translate(504px,-40px)}82%{transform:translate(656px,-92px)}100%{transform:translate(780px,-55px)}}
@keyframes cc-phase{0%,18%{transform:translateX(0)}22%,38%{transform:translateX(150px)}42%,82%{transform:translateX(300px)}86%,100%{transform:translateX(450px)}}
.cc-cubic-dot{animation:cc-cubic 12s linear infinite}.cc-bbr-dot{animation:cc-bbr 12s linear infinite}.cc-phase-hi{animation:cc-phase 12s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.cc-cubic-dot,.cc-bbr-dot,.cc-phase-hi{animation:none}.cc-cubic-dot{transform:translate(384px,-68px)}.cc-bbr-dot{transform:translate(400px,-95px)}.cc-phase-hi{transform:translateX(300px)}}
</style>
<text class="cc-label" x="24" y="30">Two controllers, one interval, different control loops</text>
<rect class="cc-bg" x="50" y="55" width="820" height="175" rx="10"/>
<text class="cc-label" x="68" y="83">CUBIC: window grows toward loss</text>
<line class="cc-grid" x1="70" y1="120" x2="850" y2="120"/><line class="cc-axis" x1="70" y1="205" x2="850" y2="205"/>
<path class="cc-path" d="M70 185 C130 180 190 145 254 123 L270 198 C330 190 390 150 454 117 L470 194 C530 184 590 143 654 112 L670 190 C730 180 790 138 850 105"/>
<circle class="cc-dot cc-cubic-dot" cx="70" cy="185" r="7"/>
<text class="cc-small" x="70" y="220">time</text><text class="cc-small" x="760" y="104">loss frontier</text>
<rect class="cc-bg" x="50" y="250" width="820" height="220" rx="10"/>
<text class="cc-label" x="68" y="278">mainline Linux BBRv1: model probes, then drains</text>
<line class="cc-axis" x1="70" y1="330" x2="850" y2="330"/><line class="cc-grid" x1="70" y1="300" x2="70" y2="445"/>
<path class="cc-path" d="M70 420 C120 410 155 365 200 320 C235 290 265 345 300 375 C345 405 390 350 440 325 C490 300 535 355 580 380 C625 402 680 345 730 325 C775 305 820 350 850 365"/>
<circle class="cc-dot cc-bbr-dot" cx="70" cy="420" r="7"/>
<rect class="cc-active cc-phase-hi" x="70" y="292" width="145" height="145" rx="8"/>
<rect class="cc-phase" x="70" y="440" width="145" height="24" rx="4"/><rect class="cc-phase" x="220" y="440" width="145" height="24" rx="4"/><rect class="cc-phase" x="370" y="440" width="145" height="24" rx="4"/><rect class="cc-phase" x="520" y="440" width="145" height="24" rx="4"/>
<text class="cc-small" x="100" y="457">STARTUP</text><text class="cc-small" x="270" y="457">DRAIN</text><text class="cc-small" x="397" y="457">PROBE_BW</text><text class="cc-small" x="545" y="457">PROBE_RTT</text>
<text class="cc-small" x="690" y="457">phase cycle repeats</text>
</svg>
<figcaption>The synchronized loop shows why identical path evidence can produce different sending patterns: CUBIC follows a window sawtooth while mainline Linux BBRv1 cycles through model probes.</figcaption>
</figure>

The moving comparison should be read as control logic, not a waveform promised by every packet capture. CUBIC's window rises, slows near its remembered maximum, then probes beyond it until congestion causes another reduction. BBR's model goes through acquisition, drain, steady bandwidth probing, and occasional RTT remeasurement. Application-limited periods, recovery, ACK compression, offload, and route changes perturb both patterns.

The strongest operational difference is the choice of evidence. CUBIC treats a congestion event as the boundary that just proved unsafe. BBR asks what delivery rate the path demonstrated and what RTT looked like when the queue was smallest. That can make BBR resilient to random packet loss, but it also creates new failure modes when the model is biased.

### Why loss tolerance is not loss blindness

The slogan that BBR "ignores loss" is inaccurate. Mainline BBRv1 does not use ordinary random loss as its primary steady-state signal in the way CUBIC does, but TCP still detects and retransmits lost data, recovery still consumes time, and sufficiently severe loss can constrain delivery samples and inflight behavior. BBRv3 goes further by making loss and ECN explicit inputs to short-term and long-term bounds.

This nuance matters in a postmortem. If BBR traffic shows retransmissions, the right question is not why the controller ignored them. Ask whether the losses reduced useful delivery, whether probes created queue pressure, whether a policer enforced a burst-sensitive limit, and whether the implementation's loss response matches the version actually running.

Random independent packet loss is also a model, not a faithful description of every link. Real losses cluster around queues, radio fades, handovers, policers, offload-sized bursts, and device failures. The Google BBR FAQ notes that `netem` can drop an entire offloaded sk_buff containing many MTU-sized packets, producing a burstier process than its percentage suggests. Disabling GRO, GSO, and TSO in the lab reduces that distortion at the cost of more CPU work. It does not turn a veth experiment into a cellular network.

### Model errors have recognizable signatures

An overestimated bandwidth sample raises pacing pressure. A stale or inflated minimum RTT raises the estimated BDP. ACK aggregation can make delivery arrive in clumps that do not reflect bottleneck service. A policer can punish probing bursts. A route change can invalidate both bandwidth and RTT history. In each case, "BBR is enabled" is not a diagnosis.

Use `ss -tin` repeatedly during a long flow and correlate:

- `delivery_rate` with application goodput;
- `pacing_rate` with the configured or observed bottleneck;
- `minrtt` with quiet-path RTT;
- `rtt` with queue growth;
- `retrans` and loss counters with the drop process;
- `cwnd` and bytes in flight with the derived BDP.

If pacing is 100 Mbit/s into a 20 Mbit/s bottleneck for a brief probe, that is not itself a bug. If the queue remains inflated, retransmissions persist, and competing flows lose their share, the control loop is not producing an acceptable operating point for that topology.

## What actually changes when you switch

![Switching controllers changes sender estimation and probing, not the physical path](/imgs/blogs/congestion-control-cubic-bbr-and-what-changing-it-actually-does-3.webp)

Changing the default controller affects new sockets. Existing established connections generally retain the controller attached to their socket. Long-lived connections, connection pools, HTTP/2 sessions, databases, and service-mesh upstreams can therefore make a fleet change look gradual or inconsistent.

The change also acts only where that host is the sender of the relevant bytes. Enabling BBR on a web server affects large responses sent from the server. It does not control uploads sent by a client that you do not operate. For a bidirectional protocol, each direction has its own sender and its own congestion-control state.

What stays fixed is just as important:

- The route and bottleneck capacity do not change.
- Receiver flow control does not change.
- Application concurrency and payload size do not change.
- A broken MTU, bad NIC, or overloaded CPU remains broken.
- Fairness policy in the network does not automatically adapt to the new controller.

| Operational layer | What a controller switch can change | What it cannot change |
| --- | --- | --- |
| Sender estimation | Which path signals dominate and how history is filtered | The truth of the physical path |
| Packet emission | Pacing pattern, bursts, and in-flight budget | Link line rate |
| Queue interaction | Standing queue, transient probe pressure, loss timing | Buffer implementation in the bottleneck |
| Flow coexistence | How this flow yields or probes against others | Tenant or business priority by itself |
| Application result | Goodput and latency when transport is limiting | Server computation or receiver consumption rate |

This is why application dashboards are insufficient. A lower download time can be caused by transport, cache hit rate, object size, or backend work. A credible canary retains per-flow transport evidence and controls the object, destination, route, and time window.

### Global, per-socket, and per-route selection

Linux exposes several scopes. The global sysctl provides a default for new TCP sockets:

```bash
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr
```

That is simple and broad. It is a poor first production experiment because every eligible new socket changes together.

An application can set `TCP_CONGESTION` on a socket before connecting or listening. Tools such as `iperf3` expose this as `--congestion`. A service can canary a process, listener, or selected outbound connection without changing unrelated workloads.

Linux also supports a route metric:

```bash
sudo ip route replace 203.0.113.0/24 via 192.0.2.1 dev eth0 congctl bbr
ip route show 203.0.113.0/24
```

The [`ip-route(8)` manual](https://man7.org/linux/man-pages/man8/ip-route.8.html) documents `congctl NAME` and `congctl lock NAME` for Linux 3.20 and later. Without `lock`, an application can override the route suggestion. With `lock`, it cannot. A locked route is therefore policy enforcement, not merely a canary hint.

Never copy the example prefix or gateway into a real host. Resolve the exact existing route first, preserve all its metrics, and prepare the reverse command. Replacing a route with an incomplete reconstruction can remove unrelated attributes. In the lab, the direct namespace route is disposable and fully known, so per-route selection is safe.

### Existing sockets make rollouts look haunted

A controller is attached to a socket, so connection lifetime becomes rollout state. Imagine a service with an HTTP/2 upstream pool whose connections remain open for hours. At 10:00, an operator changes the host default from CUBIC to BBR. New connections use BBR, while established pool members continue with CUBIC. Traffic distribution, backend health, or routine pool churn then changes the mixture over several hours. A chart can show a gradual effect even though the sysctl changed instantly.

Before the canary, inventory long-lived transports:

```bash
ss -tinp state established
ss -tinp state established | awk '/users:/ {print}' | head
```

Do not kill production connections merely to make the experiment tidy. Instead, label sockets by controller, wait for controlled turnover, or canary a fresh process pool. Record the connection creation time when the application exposes it.

The same principle applies to rollback. Restoring the sysctl changes the default for future sockets. It does not necessarily move every live socket back. A rollback plan must state whether natural drain is fast enough, whether the canary pool will be removed from service, or whether connections can be closed safely by the application.

### Preserve the route before you replace it

Per-route `congctl` is attractive because it scopes by destination, but `ip route replace` reconstructs a route. On a production host, capture the exact route and its policy context first:

```bash
DEST=203.0.113.10
ip -details route get "$DEST"
ip -details route show table all match "$DEST"
ip rule show
```

Policy routing, multiple tables, VRFs, source constraints, metrics, ECN features, MTU locks, and multipath next hops can all matter. The rollback is not always `ip route del`. It may be an exact restoration of the prior object. Have a second management path if the experiment touches the route used for access to the host.

For a first canary, an application-level `TCP_CONGESTION` option is often safer because it avoids route reconstruction and narrows the blast radius. The trade-off is code or process configuration. A per-route hint is useful when a destination class is the experiment and the route is already managed declaratively. A global sysctl is appropriate only after those narrower scopes have answered the mechanism and coexistence questions.

### Define rollback as an executable decision

"Roll back if latency gets worse" is not a rollback plan. Write the predicate before treatment:

For an illustrative policy, trigger rollback after 15 consecutive minutes when the canary loaded-RTT p99 exceeds 1.20 times the control loaded-RTT p99 and canary median goodput remains below 1.05 times control goodput. Trigger it immediately through a separate condition when latency-sensitive cross-traffic goodput falls below 0.90 times its baseline.

This is an illustrative policy model, not a universal threshold. The values must come from the service SLO and normal variance. The important structure is explicit: duration prevents one noisy interval from deciding, a control limits shared environmental bias, goodput prevents paying a latency cost without benefit, and cross-traffic catches harm outside the canary's own dashboard.

Automate evidence capture before automating the mutation. A rollback that fires without retaining socket, qdisc, route, and application evidence protects users but teaches little. A rollback that waits for a human to assemble evidence may protect nobody.

## The fairness fight at one bottleneck

![CUBIC and BBR interpret feedback from one shared bottleneck through different control loops](/imgs/blogs/congestion-control-cubic-bbr-and-what-changing-it-actually-does-4.webp)

**Rule of thumb: judge a congestion controller beside the traffic it must coexist with, not only in an empty-link benchmark.**

Two identical controllers still need time to converge. Two different controllers have a harder problem because they assign different meaning to the same events. A CUBIC flow can reduce after loss while a BBR flow continues pacing from a bandwidth model. That can give the BBR flow more of the bottleneck. A BBR probe can also cause loss that pushes the CUBIC flow down, reinforcing the imbalance.

The reverse is possible in another regime. A deep standing queue created by loss-based traffic can contaminate BBR's RTT observations. A policer can drop BBR's paced probes. Many short CUBIC flows can repeatedly enter and leave, preventing a long BBR flow from seeing a stable environment. An RTT difference changes both controllers' feedback timing.

There is no context-free fairness verdict. At minimum, record:

- controller and exact implementation version for each class of sender;
- base RTT and RTT distribution;
- bottleneck rate and buffer or active queue management policy;
- number of flows and their start times;
- application-limited versus continuously backlogged behavior;
- random loss, congestion loss, and ECN separately;
- per-flow goodput, retransmissions, and queue delay.

### Fairness has at least four meanings

Engineers often say "fair" when they mean one of four different properties:

- **Intra-controller fairness:** two CUBIC flows or two BBR flows under comparable conditions converge toward comparable shares.
- **Inter-controller coexistence:** a CUBIC flow and a BBR flow can share without one persistently suppressing the other.
- **RTT fairness:** otherwise similar flows with different round-trip times receive an acceptable share.
- **Application fairness:** short requests, streaming traffic, and bulk transfers meet their service objectives while sharing a queue.

Success on one axis does not imply success on the others. CUBIC was designed to reduce Reno's strong RTT bias on high-BDP paths, yet equal rates across different RTTs are not guaranteed in every loss model. BBR's bandwidth probing can help low-share BBR flows discover unused capacity, yet interaction with a loss-based flow depends on buffer size and probe timing. A queue can report a good long-window Jain index while short RPCs still suffer damaging latency spikes.

The experiment must match the fairness claim. To test intra-controller convergence, start two long flows with staggered start times and watch whether the incumbent yields. To test coexistence, run one flow from each controller through the same bottleneck and reverse their start order. To test RTT fairness, give flows different propagation delays without changing the bottleneck. To test application fairness, add the actual short-flow distribution and measure completion-time percentiles, not only bytes per second.

### Version nuance is part of the result

The BBRv3 draft contains an explicit dual-time-scale strategy for Reno and CUBIC coexistence. Its July 2026 text describes a BBR-native probe interval randomized between 2 and 3 seconds and a Reno-conscious interval bounded at 62 or 63 round trips. Those are cited design parameters in draft-ietf-ccwg-bbr-06, not mainline BBRv1 constants and not measurements from this post.

The draft's reasoning exposes the trade-off. Probe too frequently and the BBR flow can cause loss before a loss-based peer regrows its window. Probe too slowly and BBR takes too long to discover freed capacity. Randomization helps prevent multiple flows from synchronizing, but it cannot make heterogeneous control loops equivalent.

When a benchmark says `BBR`, ask five follow-ups:

1. Which BBR generation and source commit?
2. TCP in the kernel or QUIC in user space?
3. Which pacing qdisc and offload settings?
4. Which competing controller versions?
5. Which RTT, buffer, AQM, and loss process?

Without those fields, a fairness number is not portable. It may still describe the test that produced it, but it cannot responsibly choose a fleet default.

The July 2026 BBRv3 draft explicitly treats coexistence with Reno and CUBIC as a design constraint. It varies bandwidth-probe timing using a BBR-native interval and a Reno-conscious round-trip interval. That is evidence that coexistence is a hard control problem, not evidence that every topology is now fair. The document remains a draft, and mainline BBRv1 does not inherit BBRv3 by name.

### Use a fairness metric carefully

For $n$ flows with throughputs $x_i$, Jain's fairness index is:

$$
J(x_1, \ldots, x_n) = \frac{\left(\sum_{i=1}^{n} x_i\right)^2}{n\sum_{i=1}^{n}x_i^2}
$$

The value is 1 when all measured throughputs are equal. That is a derived mathematical property of the metric, not proof that equal throughput is the correct business policy. Different RTTs, paid service classes, deadlines, or object sizes can make equal shares inappropriate. Also, a high index with low total utilization is not success.

For two flows at 15 Mbit/s and 5 Mbit/s, the derived index is:

$$
J = \frac{(15+5)^2}{2(15^2+5^2)} = \frac{400}{500} = 0.8
$$

Report the raw per-flow rates, total goodput, queue delay, and the index. The index compresses distribution into one number and can hide starvation intervals.

This boundary matters to service architecture. [Rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) owns admission and business priority above TCP. Congestion control owns how admitted byte streams contest a path. Do not ask one layer to secretly implement the other's policy.

## The dated public case: Google's BBR rollout

On July 20, 2017, Google Cloud published [TCP BBR congestion control comes to GCP](https://cloud.google.com/blog/products/networking/tcp-bbr-congestion-control-comes-to-gcp-your-internet-just-got-faster), written by Neal Cardwell and Yuchung Cheng. The post said BBR already powered TCP traffic from `google.com` and reported 4 percent higher YouTube network throughput on average globally, with more than 14 percent in some countries. It also described enabling BBR on traffic from certain Google Cloud services and on responses served through Cloud Load Balancing or Cloud CDN.

That is a useful dated deployment case because the owner states the rollout and bounds the measured result. It is not a promise that an arbitrary service will gain 4 percent. Google's client population, path mix, object sizes, server stack, rollout controls, and definition of network throughput define the observation.

The same article included a synthetic scenario: a 10 Gbit/s sender, 100 ms RTT, and 1 percent loss, where it reported about 3.3 Mbit/s for CUBIC and over 9,100 Mbit/s for BBR. Those values are reported by Google for that named scenario, not reproduced in this post. They illustrate the mechanism at an extreme high-BDP, lossy operating point. Quoting the ratio without the 10 Gbit/s link, 100 ms RTT, and 1 percent loss would be benchmark laundering.

| Reported result | Context | What transfers | Source |
| --- | --- | --- | --- |
| 4 percent average throughput improvement | YouTube global network throughput reported in 2017 | Large-scale deployment can produce a modest aggregate gain even when some paths gain much more | Google Cloud, 2017-07-20 |
| More than 14 percent in some countries | Country subsets in the same rollout | Path mix matters | Google Cloud, 2017-07-20 |
| 3.3 versus over 9,100 Mbit/s | Synthetic 10 Gbit/s, 100 ms RTT, 1 percent loss scenario | Random loss can collapse a loss-based large-BDP flow | Google Cloud, 2017-07-20 |

### Evidence ledger

- **Case:** Google and YouTube deployment of TCP BBR.
- **Event date:** public announcement on 2017-07-20; the article does not give one single cutover date for all preceding deployment.
- **Source owner:** Google Cloud, Neal Cardwell and Yuchung Cheng.
- **Mechanism:** BBR estimates bottleneck bandwidth and propagation RTT instead of using loss as the primary congestion signal.
- **Verified numbers:** 4 percent average YouTube network throughput improvement globally, more than 14 percent in some countries, plus the separately labeled synthetic scenario above.
- **Transfer lesson:** segment results by path and client population, separate production aggregates from synthetic extremes, and deploy from the sending side.

The trigger was not a public incident. It was the mismatch between modern path behavior and the assumption that packet loss reliably marks a full bottleneck. Contributing conditions included shallow buffers and random loss on high-speed, long-distance paths. The potential blast-radius multiplier was enormous because server-side TCP policy affects many clients. Google's result therefore argues for measurement discipline as much as for BBR.

## Read the result without fooling yourself

![Expected netlab result bands for CUBIC and BBR under a controlled lossy path](/imgs/blogs/congestion-control-cubic-bbr-and-what-changing-it-actually-does-5.webp)

The lab below is intentionally smaller than Google's published synthetic case. It uses a 20 Mbit/s ceiling, 80 ms RTT, 1 percent random loss on the data direction, and the same 64 MiB file for both controllers. The goal is not to reproduce Google's ratio. It is to make the control-loop difference visible on a machine an engineer can inspect.

The expected bands are acceptance ranges, not measurements claimed by this article. Kernel, CPU, virtualization, offload, random seed, qdisc implementation, HTTP server, and socket reuse all move the result. Record three runs and compare medians. If both controllers hit roughly 20 Mbit/s, your impairment is probably not acting on the intended packets. If neither gets beyond a few hundred kilobits per second, inspect loss placement, MTU, CPU, and receiver limits before blaming the algorithm.

| Controller | Expected median goodput | Expected quiet RTT | Expected loaded RTT | Source |
| --- | ---: | ---: | ---: | --- |
| CUBIC | 1–8 Mbit/s | 75–90 ms | 80–250 ms | Reproducible acceptance range for this post's `netlab` configuration, verify locally |
| BBRv1 | 8–19 Mbit/s | 75–90 ms | 80–180 ms | Reproducible acceptance range for this post's `netlab` configuration, verify locally |

These broad bands deliberately overlap possible platform behavior. The causal expectation is stronger than the exact number: under independent random data loss, CUBIC should reduce its congestion window repeatedly, while BBRv1 should preserve more of its bandwidth estimate. Loaded RTT is less predictable because qdisc depth and the sender's probe behavior matter.

### Separate four questions

1. **Did the chosen socket use the intended controller?** Read `cong_algo` from `ss -tin` or `tcp_info`, not the sysctl alone.
2. **Did the impairment create the intended path?** Verify quiet RTT with multiple probes, qdisc counters with `tc -s`, and the HTTP transfer's direction.
3. **Did useful application bytes arrive faster?** Compare `curl`'s `speed_download` and `time_total` for the identical object.
4. **What did the improvement cost?** Compare RTT during load, retransmissions, CPU, and coexistence with another flow.

A single download answers only part of the third question. It does not prove fleet safety.

### Interpret combinations, not isolated metrics

The following patterns are more useful than a winner column:

- **BBR goodput rises and loaded RTT stays near the quiet floor.** The model-based controller is probably using capacity that random loss denied to CUBIC without building a large standing queue. Confirm the live controller and qdisc before accepting the explanation.
- **BBR goodput rises and loaded RTT rises sharply.** The flow may be probing into a queue, the qdisc may differ from the intended setup, or an RTT estimate may be stale. The throughput gain is real but not free.
- **Both controllers stop near the same ceiling.** The shaper, receiver, CPU, HTTP server, or application can be the common limit. A controller switch is not the discriminating variable.
- **CUBIC varies widely across three runs.** A random 1 percent loss process can create different loss epochs in a 64 MiB transfer. Add runs, retain the full distribution, and inspect qdisc drop counts.
- **BBR reports high pacing rate but low delivery rate.** The sender wants to emit faster than useful bytes are being acknowledged. Look for policers, downstream limits, recovery, receiver pressure, and application stalls.
- **Application goodput rises while TCP retransmissions also rise.** More delivered work can coexist with more network waste. Decide whether CPU, link cost, tail completion time, and competing traffic tolerate it.

Convert units before comparing layers. `curl` reports `speed_download` in bytes per second. Link and qdisc rates are commonly bits per second. Multiply the `curl` field by 8, then divide by 1,000,000 for decimal Mbit/s. Do not divide by 1,048,576 and label the result Mbit/s.

Also distinguish a transfer average from an instantaneous socket estimate. `curl` averages the complete object, including startup and recovery. `ss` can show a recent delivery-rate sample during the connection. A short object's average can be lower even when the steady section reaches the bottleneck. That is not a contradiction. It means startup occupies a material fraction of the transfer.

### When the expected range misses

An out-of-range result is diagnostic evidence. Work from the path inward:

1. Confirm 67,108,864 bytes arrived on every run.
2. Confirm the route attached `cubic` or `bbr` to the server's live sending socket.
3. Confirm quiet RTT is 75–90 ms and the IFB qdisc reports drops during transfer.
4. Confirm the bottleneck ceiling is on the data direction and ACK delay is on the reverse direction.
5. Confirm GRO, GSO, and TSO state, CPU saturation, and Python server throughput.
6. Confirm no prior qdisc, route metric, or namespace process survived from another run.

Do not silently widen the acceptance range after seeing a surprising value. Record the host, kernel, `iproute2`, qdisc output, offload state, and all three runs. If the topology is correct, the surprise may be the most valuable part of the exercise.

## A safe switching runbook

![A staged runbook for canarying and rolling back a Linux congestion-control change](/imgs/blogs/congestion-control-cubic-bbr-and-what-changing-it-actually-does-6.webp)

Start with availability. `tcp_available_congestion_control` lists registered algorithms. More may exist as unloaded modules. Loading `tcp_bbr` is a host-level action and can require privilege. In containers, the relevant network namespace and kernel belong to the host, even if a container exposes part of `/proc/sys`.

Then select the narrowest scope:

1. One test socket with `TCP_CONGESTION`.
2. One disposable namespace or host.
3. One destination route without `lock`.
4. One canary service pool.
5. A fleet default only after coexistence and rollback evidence are boring.

Define rollback before treatment. A useful threshold names an observable regression, a duration, and an owner. For example: roll back if the canary's p99 loaded RTT increases by more than 20 percent for 15 minutes while goodput improves by less than 5 percent, or if control traffic sharing the same link loses more than 10 percent of its baseline goodput. Those values are illustrative policy choices, not universal thresholds. Derive them from the service SLO and traffic mix.

The application boundary belongs in the review. A faster bulk transfer may increase burst pressure on a downstream decompressor, cache, or storage service. [Observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) explains how to connect transport evidence to application latency without turning one dashboard into a causal oracle. [Timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) owns the retry policy that can amplify a transport regression.

### Production read-only checklist

```bash
# Replace the documentation address with the real destination.
DEST=203.0.113.10

ip route get "$DEST"
sysctl -n net.ipv4.tcp_congestion_control
sysctl -n net.ipv4.tcp_available_congestion_control
tc -s qdisc show
ss -tin dst "$DEST"
nstat -az | grep -E 'TcpRetransSegs|TcpExtTCPLostRetransmit|TcpExtTCPTimeouts'
```

The counters are host-wide unless you isolate them by namespace or use more targeted instrumentation. Take deltas across the canary interval and correlate them with socket-level evidence. Never attribute a global counter jump to one route merely because the timestamps overlap.

## Run it yourself

### Question

On the same Linux namespace path, with a verified RTT near 80 ms, a 20 Mbit/s data-direction ceiling, and 1 percent random data loss, does a 64 MiB HTTP download using mainline BBRv1 retain more goodput than the same download using CUBIC?

### Preconditions

Use Linux with root or equivalent `CAP_NET_ADMIN`, `iproute2`, `tc`, `curl`, Python 3, `ethtool`, and a kernel exposing both `cubic` and `bbr`. macOS readers should use the privileged Linux VM described in [the series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). Namespace, qdisc, route, and offload changes below are scoped to the disposable `c` and `s` namespaces, but `modprobe tcp_bbr` loads a host kernel module. Do not run the mutation against a production interface.

The setup places `netem` on receiver ingress through IFB because the Google BBR FAQ warns that sender-side `netem` interacts unrealistically with TCP Small Queues. It disables GRO, GSO, and TSO on the namespace veths to make the configured loss process closer to per-packet loss. This increases CPU work and is one reason the result is a lab result.

### Preflight

```bash
set -euo pipefail

sudo modprobe tcp_bbr
sudo modprobe ifb

available=$(sysctl -n net.ipv4.tcp_available_congestion_control)
case " $available " in
  *" cubic "*) ;;
  *) echo "cubic is unavailable: $available" >&2; exit 1 ;;
esac
case " $available " in
  *" bbr "*) ;;
  *) echo "bbr is unavailable: $available" >&2; exit 1 ;;
esac

command -v ip >/dev/null
command -v tc >/dev/null
command -v curl >/dev/null
command -v python3 >/dev/null
command -v ethtool >/dev/null

ip netns list
ip -n c link show c0
ip -n s link show s0
ip -n c addr show dev c0
ip -n s addr show dev s0
ip -n c route get 10.77.0.2
ip -n s route get 10.77.0.1
```

Expected: `c0` is `10.77.0.1/30`, `s0` is `10.77.0.2/30`, both links are up, and each route resolves through the named veth. If the namespaces do not exist, complete the setup post first.

### Baseline

Create the same deterministic 64 MiB file for both treatments. The content does not need entropy because HTTP does not compress it here. Start one server and verify the clean path before applying impairment.

```bash
set -euo pipefail

sudo ip netns exec s sh -c \
  'dd if=/dev/zero of=/tmp/netlab-cc.bin bs=1M count=64 status=none'

sudo ip netns exec s sh -c \
  'cd /tmp && exec python3 -m http.server 8080 --bind 10.77.0.2' \
  >/tmp/netlab-cc-http.log 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" | sudo tee /tmp/netlab-cc-http.pid >/dev/null
sleep 1

sudo ip netns exec s ip route replace 10.77.0.0/30 dev s0 src 10.77.0.2 congctl cubic
sudo ip netns exec c ping -q -c 10 10.77.0.2
sudo ip netns exec c curl --fail --silent --show-error \
  --output /dev/null \
  --write-out 'controller=cubic-clean bytes=%{size_download} speed_Bps=%{speed_download} time_s=%{time_total}\n' \
  http://10.77.0.2:8080/netlab-cc.bin
sudo ip netns exec s ss -tin dst 10.77.0.1
```

Read: `bytes` should be 67,108,864, `speed_Bps` is application goodput in bytes per second, `time_s` is total transfer time, and `ss -tin` should name CUBIC while a connection exists. The final `ss` may miss a short clean connection; the impaired repetitions below sample during transfer.

Expected: quiet RTT on a direct namespace veth is usually below 2 ms, and the clean transfer is limited mainly by CPU and veth performance. This clean run is a topology check, not the CUBIC versus BBR comparison.

### Apply one change

Create an ingress IFB inside namespace `c`. Data from `s` enters `c0`, is redirected to `ifb0`, and encounters the 40 ms delay, 1 percent random loss, and 20 Mbit/s ceiling there. ACKs leaving `c0` encounter a separate 40 ms delay. `s0` uses `fq` so BBR can use qdisc pacing.

```bash
set -euo pipefail

for nsdev in 'c c0' 's s0'; do
  set -- $nsdev
  sudo ip netns exec "$1" ethtool -K "$2" gro off gso off tso off
done

sudo ip netns exec c ip link del ifb0 2>/dev/null || true
sudo ip netns exec c ip link add ifb0 type ifb
sudo ip netns exec c ip link set ifb0 up

sudo ip netns exec c tc qdisc replace dev c0 clsact
sudo ip netns exec c tc filter replace dev c0 ingress \
  matchall action mirred egress redirect dev ifb0
sudo ip netns exec c tc qdisc replace dev ifb0 root netem \
  delay 40ms loss random 1% rate 20mbit limit 1000
sudo ip netns exec c tc qdisc replace dev c0 root netem delay 40ms limit 1000
sudo ip netns exec s tc qdisc replace dev s0 root fq

sudo ip netns exec c ping -q -c 20 10.77.0.2
sudo ip netns exec c tc -s qdisc show dev ifb0
sudo ip netns exec c tc -s qdisc show dev c0
sudo ip netns exec s tc -s qdisc show dev s0
```

Read: the `ping` summary's average RTT should be about 80 ms. `tc -s` on `ifb0` should show the 20 Mbit/s rate and later accumulate dropped packets. `c0` should show the 40 ms ACK-path delay. `s0` should show `fq`.

Expected: quiet RTT should usually land between 75 and 90 ms. Twenty probes can all survive 1 percent loss, so zero lost pings does not disprove the impairment. The `ifb0` qdisc's `dropped` counter during the 64 MiB transfers is better evidence.

### Compare

The function below selects a per-route controller for new server-side connections, downloads the identical file three times, samples the live socket during each transfer, and writes only generated measurements under `/tmp`. The route is replaced before each new connection, so the server's sending socket inherits the selected controller.

```bash
set -euo pipefail

run_controller() {
  algo=$1
  sudo ip netns exec s ip route replace \
    10.77.0.0/30 dev s0 src 10.77.0.2 congctl "$algo"
  sudo ip netns exec s ip route show 10.77.0.0/30

  run=1
  while [ "$run" -le 3 ]; do
    out="/tmp/netlab-cc-${algo}-${run}.txt"
    sudo ip netns exec c curl --fail --silent --show-error \
      --output /dev/null \
      --write-out "controller=${algo} run=${run} bytes=%{size_download} speed_Bps=%{speed_download} time_s=%{time_total}\n" \
      http://10.77.0.2:8080/netlab-cc.bin >"$out" &
    curl_pid=$!
    sleep 1
    sudo ip netns exec s ss -tin dst 10.77.0.1 | tee -a "$out"
    wait "$curl_pid"
    cat "$out"
    run=$((run + 1))
  done
}

run_controller cubic
run_controller bbr

awk '
  /controller=/ {
    algo=""; speed=""
    for (i=1; i<=NF; i++) {
      if ($i ~ /^controller=/) { split($i,a,"="); algo=a[2] }
      if ($i ~ /^speed_Bps=/)  { split($i,a,"="); speed=a[2] }
    }
    if (algo != "" && speed != "") {
      printf "%s %.3f Mbit/s\n", algo, speed*8/1000000
    }
  }
' /tmp/netlab-cc-{cubic,bbr}-*.txt | sort

sudo ip netns exec c tc -s qdisc show dev ifb0
sudo ip netns exec s nstat -az | grep -E 'TcpRetransSegs|TcpExtTCPTimeouts'
```

Read: confirm the live `ss` line names `cubic` or `bbr` as intended. Compare three `speed_Bps` values per controller after converting to Mbit/s. Record the median, not the fastest run. For BBR, inspect `bw`, `mrtt`, `pacing_rate`, and `delivery_rate`. For CUBIC, inspect `cwnd`, `rtt`, and retransmission fields. Read `tc -s` drops and TCP counter deltas as supporting evidence.

Expected: on a reasonably idle modern Linux host, CUBIC will often land in the broad 1–8 Mbit/s range and BBRv1 in the 8–19 Mbit/s range. Both should download exactly 67,108,864 bytes. Quiet RTT should remain about 75–90 ms; loaded RTT can be higher. Treat values outside the bands as a reason to inspect topology and implementation, not as permission to edit the result until it agrees with this post.

Known variance includes kernel version, BBR availability, IFB and qdisc implementation, random-loss sequence, CPU scheduling, Python HTTP server throughput, veth offload behavior, and virtualization. The lab proves a conditional claim about this topology. It does not prove that BBR is always faster or fairer.

### Reset

The cleanup removes only state created for this experiment and restores the direct route without a controller metric. It then terminates the recorded lab HTTP server and deletes the generated file and measurements.

```bash
set -euo pipefail

sudo ip netns exec c tc qdisc del dev c0 clsact 2>/dev/null || true
sudo ip netns exec c tc qdisc del dev c0 root 2>/dev/null || true
sudo ip netns exec s tc qdisc del dev s0 root 2>/dev/null || true
sudo ip netns exec c ip link del ifb0 2>/dev/null || true
sudo ip netns exec s ip route replace 10.77.0.0/30 dev s0 src 10.77.0.2

if [ -f /tmp/netlab-cc-http.pid ]; then
  sudo kill "$(cat /tmp/netlab-cc-http.pid)" 2>/dev/null || true
fi
sudo rm -f /tmp/netlab-cc-http.pid /tmp/netlab-cc-http.log
sudo ip netns exec s rm -f /tmp/netlab-cc.bin
sudo rm -f /tmp/netlab-cc-cubic-*.txt /tmp/netlab-cc-bbr-*.txt
```

Expected: `tc qdisc show` no longer lists the experiment's `netem`, `clsact`, or `fq` qdiscs, `ifb0` is absent, and the `s` route no longer has a `congctl` metric. Offload settings on `c0` and `s0` remain disabled until the namespaces are recreated because the safe original state is not knowable from this script. The series teardown recreates the veth pair and restores defaults.

### Production translation

On a real host, keep this phase read-only until a reviewed canary exists:

```bash
DEST=203.0.113.10
ip route get "$DEST"
sysctl -n net.ipv4.tcp_available_congestion_control
tc qdisc show
ss -tin dst "$DEST"
```

For the canary, prefer an application-set `TCP_CONGESTION` socket option or an exact per-route `congctl` suggestion with a captured rollback command. Do not introduce a qdisc change and a controller change in the same comparison. Do not use a locked route until you intentionally want to prevent application overrides.

## Common failed interpretations

### "BBR gave us more bandwidth"

It gave the flow a different share or better utilization of existing bandwidth. Verify the bottleneck rate before and after. If total link goodput did not increase but one flow improved, another flow or queue absorbed the cost.

### "CUBIC is obsolete because it reacts to loss"

CUBIC is a current Standards Track algorithm with broad deployment and known coexistence behavior. Loss remains a valid congestion signal in many paths. The question is whether loss is sufficiently correlated with congestion for the workload and topology.

### "The sysctl says BBR, so this connection uses BBR"

The sysctl is a default for new sockets. An existing socket, route metric, namespace default, or application `TCP_CONGESTION` setting can differ. Read the live socket.

### "The single-flow benchmark proves fairness"

An empty bottleneck contains no competitor to be fair to. Add representative CUBIC, BBR, short, long, and latency-sensitive flows. Stagger start times. Record per-flow distributions.

### "Lower RTT proves the application is faster"

Lower queue delay helps only where queueing was on the critical path. Application think time, retries, and receive-side flow control can dominate. Use the latency ladder and correlate transport with application phases.

### "BBRv3 results describe mainline BBR"

They do not. Name the source tree, commit or kernel, and algorithm version. Mainline Linux BBRv1 and the July 2026 BBRv3 draft share a lineage and broad goal, but their control logic differs materially.

## When to reach for BBR, and when not to

### Reach for a BBR canary when

- Long-lived or bulk transfers cross high-BDP paths with measurable non-congestive random loss.
- CUBIC goodput is below the derived pipe capacity and repeated loss reductions explain the gap.
- You control the sending host and can inspect live socket state.
- The qdisc and kernel implementation are known, versioned, and reproducible.
- You can test coexistence with the traffic classes that share the real bottleneck.
- You have a narrow socket, route, or canary-pool scope and an automatic rollback threshold.

### Stay with CUBIC, or delay the switch, when

- The application, receiver window, CPU, storage, or route is the actual limit.
- Most transfers are too short to leave startup, so the steady controller difference is not the dominant mechanism.
- You cannot observe the shared bottleneck or representative competing traffic.
- The environment depends on policers or unusual middleboxes that have not been exercised in a canary.
- The expected gain is based on an unbounded public benchmark rather than your RTT, loss, rate, and workload.
- A controller change would be bundled with kernel, qdisc, NIC, or application changes.

## Key takeaways

Congestion control is a sender-side feedback loop. CUBIC grows a congestion window along a time-based cubic curve, reduces it when congestion is detected, and spends extra time near the previous maximum. BBR estimates delivery bandwidth and minimum RTT, paces around a model of the pipe, and periodically probes whether that model remains true.

Neither changes physical capacity. Each changes utilization, queue pressure, loss response, and coexistence. The winner depends on RTT, buffer, loss process, flow duration, controller version, qdisc, and competing traffic.

The discriminating measurement is not the global sysctl. It is a controlled application transfer paired with live socket state, qdisc counters, quiet and loaded RTT, retransmissions, and per-flow goodput. Use the same object and route. Record multiple runs. Compare medians. Add competing flows before making a fairness claim.

The safe rollout starts with one socket or route, keeps BBRv1 separate from BBRv3 claims, defines rollback in advance, and expands only after the improvement survives representative coexistence.

## Further reading

- [RFC 9438: CUBIC for Fast and Long-Distance Networks](https://www.rfc-editor.org/rfc/rfc9438.html), August 2023.
- [BBR: Congestion-Based Congestion Control](https://research.google/pubs/bbr-congestion-based-congestion-control-2/), Google Research and ACM Queue, September to October 2016.
- [BBR Congestion Control, draft-ietf-ccwg-bbr-06](https://datatracker.ietf.org/doc/draft-ietf-ccwg-bbr/), July 2026 Internet-Draft.
- [Linux mainline `tcp_bbr.c`](https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_bbr.c), the implementation source for mainline BBRv1.
- [Google BBR FAQ](https://github.com/google/bbr/blob/master/Documentation/bbr-faq.md), including `netem`, offload, pacing, and `ss` cautions.
- [`ip-route(8)`](https://man7.org/linux/man-pages/man8/ip-route.8.html), including per-route `congctl` and `congctl lock`.
