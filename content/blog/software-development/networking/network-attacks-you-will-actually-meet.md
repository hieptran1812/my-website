---
title: "Network attacks you will actually meet: Find the resource before choosing the defense"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Trace volumetric, connection-state, and application attacks to the first saturated resource, then measure and place defenses where they can work."
tags:
  [
    "networking",
    "distributed-systems",
    "ddos",
    "syn-flood",
    "http2",
    "rate-limiting",
    "incident-response",
    "security-engineering",
    "tcp",
    "observability",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 44
image: "/imgs/blogs/network-attacks-you-will-actually-meet-1.webp"
---

Your API is returning timeouts. The origin CPU is low. The load balancer reports fewer completed requests than usual, yet inbound traffic is high. One engineer proposes increasing the request rate limit. Another proposes adding instances. Both may be acting on the wrong side of the bottleneck.

A network attack works by making some finite resource run out before useful work can finish. That resource could be transit bandwidth, packet processing, pending TCP connection state, open sockets, HTTP/2 request dispatch, an expensive application dependency, or the authority that answers DNS for your name. The packet path below is the mental model: find the earliest saturated point, then put the control before it. A limiter at the origin cannot recover packets already discarded on the link to the origin.

![Path map from transit through edge, connection state, HTTP processing, and origin, with distinct attack bottlenecks](/imgs/blogs/network-attacks-you-will-actually-meet-1.webp)

This article is about defensive diagnosis. We will not manufacture a flood against a public service. The lab at the end uses a local namespace and a small number of slow connections to make resource accounting visible without imitating Internet scale. If you need the broader request path first, start with [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) connects this diagnosis to the rest of the series.

## The first question is what ran out

**Count at the resource boundary, not at the dashboard that happens to be easiest to open.** An HTTP request metric counts requests that made it far enough to be parsed. It says little about dropped packets at a router, unanswered TCP handshakes, or queries rejected by a DNS provider. A low request count during a bad outage may mean the application was protected by accident while users still could not enter.

Three labels are useful, provided we treat them as resource descriptions rather than rigid layers. A *volumetric* attack fills a link or exhausts packet forwarding capacity. A *protocol or state* attack forces network equipment or a server stack to spend state per incoming message. An *application* attack presents enough valid protocol to reach expensive work or tie up a concurrency slot. The same incident can cross these boundaries. A high-rate SYN stream may fill a link and a SYN backlog. A high-rate HTTP/2 stream-reset pattern can stress proxy request dispatch while the origin sees few requests. A botnet can send traffic to a DNS authority, leaving an otherwise healthy web server unreachable by name.

| First constrained resource | What the user may see | Discriminating evidence | First place a useful control can act |
| --- | --- | --- | --- |
| Transit bits or packets | Connect timeout from several client populations | Provider or edge ingress bps and pps, interface drops | Upstream provider or distributed edge before the narrow link |
| Pending TCP handshake state | New connections stall; established connections may survive | `ss` SYN-RECV state and `nstat` TCP listen counters | TCP listener or upstream SYN proxy before state allocation |
| Accepted connection or worker slot | Slow headers coexist with many established sockets | Open connections by age and header completion rate | HTTP proxy timeout and per-connection limits |
| Request dispatch or backend work | Errors and latency despite modest wire volume | Requests opened, reset, dispatched, canceled, completed | HTTP-aware edge before expensive dispatch |
| Authoritative DNS capacity | Name resolution fails before a connection attempt | Resolver query result, authoritative health, provider status | DNS provider's distributed authority and mitigation edge |

This table is a diagnostic map, not a claim that all devices expose the same counters. The counter names and accounting stages vary. We need the provider's ingress metrics when the link is full, kernel counters when the TCP listener is at issue, and protocol-aware metrics when a proxy does work on an apparently canceled request.

### Measure the boundary from both sides

A useful incident snapshot has a client probe, an edge probe, and a host probe taken over the same interval. From a client, separate DNS, connection, TLS, first byte, and total time with `curl -w`. From the edge or provider, ask for inbound bits per second, packets per second, drops, and mitigation actions. On the host, inspect listener and socket state with `ss` and TCP counters with `nstat`. If the client cannot resolve the name, a healthy origin endpoint is not evidence that the user path works.

```bash
curl --connect-timeout 3 --max-time 10 -sS -o /dev/null \
  -w 'dns=%{time_namelookup} connect=%{time_connect} tls=%{time_appconnect} first_byte=%{time_starttransfer} total=%{time_total} status=%{http_code}\n' \
  https://example.com/
dig +time=2 +tries=1 example.com A
ss -lnt
nstat -az | rg 'TcpExtListen|TcpExtSyncookies|TcpExtTCPReqQFull'
```

The `curl` values are cumulative timestamps since the request began. Subtract adjacent values to obtain phase durations; `time_connect` is not an isolated TCP duration. A DNS failure often leaves later timing fields at zero. A TLS handshake failure can leave `time_starttransfer` at zero. This is why the [TLS handshake post](/blog/software-development/networking/tls-what-the-handshake-buys-and-what-it-costs) places connection and TLS on distinct rungs. Run these probes against a service you operate or a permitted test endpoint, and observe the provider's corresponding interval rather than treating a single successful curl as proof of availability.

### Capacity is a vector, not a single number

An interface advertised in Gbit/s also has finite packets-per-second forwarding capability. A TLS proxy has handshake and request parsing capacity. A DNS authority has a query and cache-miss profile. An application has a bounded number of active requests and downstream calls. You cannot infer one limit from another. If a link is receiving many small packets, pps can be the constraint well before the bit rate reaches the nominal link speed. If HTTP handlers are expensive, an attacker may use relatively little network bandwidth while consuming most application concurrency.

Our explanatory model is a vector of utilization ratios, one per relevant resource: $u_i = a_i / c_i$, where $a_i$ is arrival demand at resource $i$ per second and $c_i$ is that resource's sustainable service capacity in the same units. This is a diagnostic abstraction, not a protocol equation or a measurement of any named incident. The first component with $u_i$ persistently above $1$ will queue or drop work, though dependencies and bursts can make the symptoms visible elsewhere. A request limiter changes one $a_i$. It does not automatically change traffic already admitted onto a congested upstream link.

## Volumetric pressure: when traffic arrives before your filter

**When the link fills, moving application code is irrelevant until traffic is removed upstream.** Imagine a narrow bridge into a data center. A guard standing after the bridge can identify every unwanted car, but each car has already occupied bridge capacity. An origin WAF has the same limitation when unwanted packets have crossed the origin's transit link before the WAF rejects them.

The two useful rate measurements are bits per second and packets per second. Bits price the transmission capacity; packets price per-packet forwarding, rule evaluation, and interrupt or polling work. Record both in and out, per interface and per provider if possible. An ingress/egress imbalance is a clue, not proof of malice: downloads and asymmetric applications can also be asymmetric. GitHub's 2018 incident report says its monitoring detected an anomaly in that ratio during the attack, which made it actionable in their context.

For scale intuition, derive serialization time from a stated link speed, not from a vague claim about packet rates. On a hypothetical 10 Gbit/s link, transmitting 10 Gbit of data takes at least 1 second because $10\,\mathrm{Gbit}/(10\,\mathrm{Gbit/s})=1\,\mathrm{s}$. That is before framing, queuing, routing, and competing legitimate traffic. If offered traffic is 15 Gbit/s and the hypothetical link can carry 10 Gbit/s, the minimum excess is 5 Gbit/s while the burst lasts. No per-user HTTP policy at the far end can recover those excess bits. The arithmetic is illustrative; it is not an estimate of any case below.

Packet capacity needs the same care. A 64-byte packet contains 512 payload-and-header bits at that counting boundary, so a purely illustrative 10 million such packets per second carries 5.12 Gbit/s at that boundary. Ethernet wire rate includes framing and inter-frame overhead, and provider counters may count at another layer. State the denominator and the accounting point before comparing a pps number with a link's advertised bit rate. Small packets can demand many more forwarding decisions per transmitted bit than large packets.

### The useful path for mitigation

At the first sign of saturation, identify where drops begin. If provider ingress is already above the customer's purchased or provisioned capacity, ask the provider or mitigation partner to filter or absorb traffic before the constrained handoff. An anycast edge can spread arrival across many locations, but it still needs a rule or capacity at each site. Blackholing a destination can protect adjacent customers or infrastructure at the cost of making the protected service unavailable. That is a last-resort containment choice, not successful service restoration.

If the link is healthy and only the host's receive queue or firewall is dropping, move down the path. A network ACL near the provider edge may help if the signal can be expressed there. A stateful rule at the host might be too late if connection tracking itself is the exhausted resource. A request rule after TLS termination might be necessary for a path-specific HTTP attack, but it cannot inspect encrypted traffic before TLS is terminated. Every control pays an observation cost and has a deployment boundary.

This is the operational distinction from [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7): a load balancer can distribute admitted work, but it does not create capacity on the link feeding it. It can also become the earliest stateful bottleneck. During an incident, record where traffic disappears: provider link, edge filter, TCP accept path, HTTP proxy, or application. That position decides who can help.

## Reflection and amplification: one request, somebody else's reply

Reflection uses a third-party service as the sender visible to the victim. The attacker sends a request to an exposed service with the victim's IP address forged as the source. The service sends its response to the victim. IP source spoofing is feasible where the attacker's network does not validate that the source address belongs to its customer. [RFC 2827](https://www.rfc-editor.org/rfc/rfc2827) describes ingress filtering intended to stop such packets near their source. It helps the Internet only when deployed by networks that can check their customers' source addresses; the victim cannot retrofit it into an attacker network.

![Reflection graph showing a forged-source request to a UDP reflector and a larger response toward the victim](/imgs/blogs/network-attacks-you-will-actually-meet-3.webp)

An amplification factor is a ratio with a precise measurement boundary. For one request and its responses, define $A = B_{\mathrm{response}}/B_{\mathrm{request}}$, where both byte counts are measured at the same protocol layer. If, purely as an example, a 100-byte UDP request elicits 4,000 response bytes, then $A=4{,}000/100=40$. At an offered request rate of 1,000 such requests per second, the illustrative request traffic is 100,000 B/s and the reflected response traffic is 4,000,000 B/s, before IP and link framing. These numbers are an arithmetic example, not a reported protocol ratio. Fragmentation, loss, payload choice, server configuration, and where bytes are counted can change an observed value.

**Never transfer a maximum factor from a protocol description to an incident as if it were measured.** GitHub's [March 1, 2018 incident report](https://github.blog/news-insights/company-news/ddos-incident-report/) describes memcached over UDP as capable of amplification *up to* 51,000 based on a contemporaneous Cloudflare explanation. The same report says the attack against GitHub peaked at 1.35 Tbps and 126.9 million packets per second. It does not say the observed ratio for the attack was 51,000. The factor describes the vector's possible behavior under particular conditions; the traffic figures describe one observed event. Treating them as one equation would invent a measurement.

For defenders, the source IP on a reflected response often belongs to a real reflector, not the machine controlling the attack. A blocklist of apparent senders can be brittle. Filter by validated protocol characteristics and cooperate with an upstream provider when the link is saturated. For operators of a potential reflector, disable unnecessary UDP exposure, require authentication or restrict clients where the service supports it, and remove public configurations that permit disproportionate responses. Source address validation at access networks addresses the spoofing prerequisite. It is a separate administrative boundary from the victim's filtering problem.

### Two different multiplication mechanisms

Reflection amplifies *wire bytes*: a small request produces a larger response, often using spoofing to direct it elsewhere. Application amplification multiplies *work*: a cheap request might invoke a costly query, decompression, cache miss, or fan-out. HTTP/2 Rapid Reset is largely about request handling work per connection and per byte rather than a UDP byte ratio. A rate policy based only on inbound bandwidth therefore misses some expensive application requests. Conversely, an HTTP request limit cannot solve a saturated ingress link. Keep the units visible: bytes, packets, handshakes, open streams, handler invocations, or downstream operations.

## Protocol state: why a SYN can be expensive

TCP normally begins with SYN, SYN-ACK, and ACK. On a listening server, an incoming SYN asks the stack to reserve enough state to remember an incomplete handshake while it sends SYN-ACK and waits for the final ACK. A flood of SYNs that never complete can consume that pending state. [RFC 4987](https://www.rfc-editor.org/rfc/rfc4987) documents the mechanism and the trade-offs among mitigations. The central symptom is new connection establishment failing while some existing connections remain usable.

![Packet timeline comparing ordinary pending TCP handshake state with SYN-cookie validation](/imgs/blogs/network-attacks-you-will-actually-meet-2.webp)

The queue is not the whole service. A listener can have pending handshake state, a queue of completed connections waiting for `accept`, established sockets, file descriptors, and application worker capacity. Different Linux counters correspond to different stages. `ss -lnt` shows listener state and backlog-related columns, while `ss -nt state syn-recv` shows incomplete connection state visible to the kernel. `nstat` exposes TCP extended counters, including syncookie activity and listen drops where supported. Capture a before and after interval, since cumulative counters alone cannot tell you whether an event is happening now.

```bash
date -u +%FT%TZ
ss -lnt '( sport = :443 )'
ss -Hnt state syn-recv '( sport = :443 )' | wc -l
nstat -az | rg 'TcpExt(Syncookies|Listen|TCPReqQFull)'
```

Use the actual listener port and note the kernel and namespace in which you run the command. A low `SYN-RECV` snapshot does not rule out an attack if the server is dropping early, using cookies, or sitting behind a proxy that owns the public handshake. A large number of established sockets points toward a different stage. A SYN flood is not equivalent to a slow HTTP header attack: the former may never complete TCP, while the latter deliberately completes TCP and occupies resources higher up.

### What a SYN cookie changes

A SYN cookie encodes enough information in the server's SYN-ACK sequence number to validate the final ACK without retaining the usual per-SYN state while the queue is under pressure. If the ACK validates, the server reconstructs the connection state. That moves the expensive allocation to a point after the client has demonstrated it can receive the SYN-ACK. It does not give the link infinite bandwidth, prevent pps exhaustion, or ensure that accepted application connections can do useful work.

Do not advise changing `tcp_syncookies` blindly. The [Linux kernel IP sysctl documentation](https://docs.kernel.org/networking/ip-sysctl.html) describes syncookies as a fallback when the SYN backlog overflows and discusses compatibility and performance trade-offs for using them as a substitute for correct capacity planning. RFC 4987 also notes that cookie schemes can constrain information available during handshake and the support for TCP options. Modern implementation details vary by kernel version. Check the actual default, kernel, and counters first:

```bash
uname -r
sysctl net.ipv4.tcp_syncookies
nstat -az | rg 'TcpExtSyncookies(Sent|Recv|Failed)'
```

If the provider's ingress link is already saturated, SYN cookies at the origin are behind the problem. If a fronting proxy handles client TCP and speaks a separate connection to the origin, inspect and defend the proxy listener. A SYN proxy or edge can validate handshakes before they reach the origin, but then that proxy is itself a capacity and availability dependency. The right question is not whether cookies are enabled in general. It is whether legitimate ACKs can still arrive and whether the path after validation can serve them.

### A worked state budget

Suppose a hypothetical listener can sustain 20,000 pending handshakes and each unmatched SYN occupies a slot for an average of 30 seconds. In a simple steady-state model, the arrival rate that would fill the queue is $20{,}000/30\,\mathrm{s}\approx667$ unmatched SYNs per second. This is an explanatory approximation based on Little's Law, $L=\lambda W$, where $L$ is average concurrent pending state, $\lambda$ is arrival rate, and $W$ is residence time. It is not the Linux backlog size, timer, or attack threshold for your host. Real servers retransmit, use caches or cookies, apply per-source logic, and see bursty arrivals. The example shows why holding time matters as much as raw arrival rate. Measure the real stage and timeout on your implementation.

## Accepted sockets can be the scarce resource too

A Slowloris-style slow-header pattern completes TCP and then sends HTTP request headers so slowly that a server waits on many incomplete requests. The bytes per second can be low while open sockets or parser contexts approach a configured limit. The [OWASP Denial of Service Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Denial_of_Service_Cheat_Sheet.html) describes slow HTTP attacks as fragmented, delayed request delivery that stalls server resources. The defense depends on which component owns the waiting state.

At a reverse proxy, set sensible header read deadlines, maximum header sizes, total request deadlines, and concurrency limits. At an application server behind the proxy, verify that the proxy does not pass an unbounded number of incomplete bodies or long-lived streams. A blanket idle timeout is not enough if an adversarial client sends just enough data to reset it indefinitely. A minimum sustained progress policy can help, but it may exclude real users on slow or unstable networks. Test the policy against legitimate mobile and accessibility-heavy clients before tightening it broadly.

The state budget again provides intuition. If a proxy permits 5,000 open connections and each slow request occupies one for 100 seconds, only $5{,}000/100\,\mathrm{s}=50$ new long-lived connections per second are enough to keep it full in an idealized steady state. Those are hypothetical inputs and a Little's Law approximation. A real proxy may multiplex, limit per-client sockets, time out headers sooner, or route slow traffic into cheaper async state. The important point is that low bit rate does not imply low resource use.

You can distinguish this from SYN pressure by looking for *established* connections with old ages and unfinished HTTP request headers, rather than many incomplete TCP handshakes. An edge proxy's active connection count and header timeout counts are stronger evidence than origin CPU. A downstream application may show no request at all until the proxy receives a complete header block. If your WAF counts only completed requests, it sees less traffic as the proxy approaches exhaustion.

## HTTP/2 Rapid Reset: concurrency can hide work rate

HTTP/2 multiplexes streams on one TCP connection. A server advertises a maximum number of concurrently active streams. That controls simultaneous open streams, which is useful for ordinary resource management. It does not by itself bound the total number of streams created and closed per second. A client can send a HEADERS frame that opens a request stream, then a RST_STREAM frame that cancels it. The slot becomes free, so another stream can start immediately. The server may already have parsed headers, allocated objects, dispatched an upstream job, or queued cancellation work. [RFC 9113](https://www.rfc-editor.org/rfc/rfc9113) defines the stream states and reset behavior; Cloudflare's [October 10, 2023 technical analysis](https://blog.cloudflare.com/technical-breakdown-http2-rapid-reset-ddos-attack/) explains the operational abuse labeled [CVE-2023-44487](https://www.cve.org/CVERecord?id=CVE-2023-44487).

<figure class="blog-anim">
<svg viewBox="0 0 860 300" role="img" aria-label="One HTTP/2 connection repeatedly opens a request stream and resets it; server work accumulates even though active stream count returns to zero" style="width:100%;height:auto;max-width:900px">
<style>
.na24-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.na24-t{font:600 19px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.na24-small{font:15px ui-monospace,monospace;fill:var(--text-secondary,#6b7280)}.na24-hot{fill:var(--accent,#6366f1)}.na24-line{stroke:var(--border,#d1d5db);stroke-width:3}.na24-job{fill:#ffec99;stroke:#b98b00;stroke-width:1.5}
@keyframes na24-sweep{0%,8%{transform:translateX(0);opacity:0}15%,27%{transform:translateX(180px);opacity:1}34%,45%{transform:translateX(360px);opacity:1}52%,63%{transform:translateX(540px);opacity:1}70%,100%{transform:translateX(540px);opacity:0}}
@keyframes na24-work{0%,26%{opacity:0}34%,100%{opacity:1}}
@keyframes na24-work2{0%,50%{opacity:0}58%,100%{opacity:1}}
.na24-pulse{animation:na24-sweep 10s ease-in-out infinite}.na24-w1{animation:na24-work 10s steps(1,end) infinite}.na24-w2{animation:na24-work2 10s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.na24-pulse,.na24-w1,.na24-w2{animation:none}.na24-pulse{opacity:1;transform:translateX(540px)}.na24-w1,.na24-w2{opacity:1}}
</style>
<text class="na24-t" x="26" y="36">One connection, repeated stream churn</text>
<line class="na24-line" x1="120" y1="116" x2="730" y2="116"/>
<rect class="na24-box" x="24" y="72" width="150" height="86" rx="12"/><text class="na24-t" x="42" y="105">HEADERS</text><text class="na24-small" x="42" y="135">open stream</text>
<rect class="na24-box" x="205" y="72" width="150" height="86" rx="12"/><text class="na24-t" x="222" y="105">dispatch</text><text class="na24-small" x="222" y="135">queue work</text>
<rect class="na24-box" x="386" y="72" width="150" height="86" rx="12"/><text class="na24-t" x="405" y="105">RST_STREAM</text><text class="na24-small" x="405" y="135">close stream</text>
<rect class="na24-box" x="567" y="72" width="254" height="86" rx="12"/><text class="na24-t" x="585" y="105">next HEADERS</text><text class="na24-small" x="585" y="135">slot reused</text>
<circle class="na24-hot na24-pulse" cx="125" cy="181" r="11"/>
<text class="na24-small" x="25" y="217">Active-stream limit sees a short-lived stream.</text>
<text class="na24-small" x="25" y="244">Already dispatched jobs may still need cancellation.</text>
<rect class="na24-job na24-w1" x="575" y="205" width="102" height="36" rx="7"/><text class="na24-small na24-w1" x="585" y="229">job 1</text>
<rect class="na24-job na24-w2" x="695" y="205" width="102" height="36" rx="7"/><text class="na24-small na24-w2" x="705" y="229">job 2</text>
</svg>
<figcaption>A reset releases an active-stream slot, but request dispatch and cancellation still consume work. Repeating the cycle can outpace cleanup.</figcaption>
</figure>

The moving dot is not a recipe for generating traffic. It shows why the average number of active streams can stay low while request creation and cancellation rates are high. This is a queueing mismatch: the admission check is on simultaneous state, but the expensive work is associated with transitions. A safe implementation must bound or cheaply account for transitions, and cancellation must actually stop downstream work where feasible. A proxy that asynchronously dispatched the request before reading the reset can have work in flight after the local stream state has closed.

Cloudflare's technical post reports that its attacks beginning August 25, 2023 eventually peaked just above 201 million requests per second. That figure is Cloudflare's reported peak across the attack it observed, not a throughput requirement for a single origin. It describes one event at a large distributed edge. The useful lesson for a smaller service is the mismatch between stream concurrency and stream churn, not the headline rate.

The same post records a mitigation trade-off. Cloudflare reduced advertised maximum stream concurrency to 64 and discovered some legitimate clients began with an assumption of 100 before processing the server setting. Resetting the excess 36 streams interacted with their protective logic and caused legitimate page load failures, so they returned to 100. These numbers are Cloudflare's reported implementation observations, not universal HTTP/2 requirements. It is an unusually clear example of a defense causing a second failure mode. Measure resets by direction, reason, and connection. Test real client behavior under the candidate policy.

### Count transitions and expensive work

The metrics that separate this mechanism from ordinary high request traffic include requests opened per connection, client-initiated resets per connection, frames parsed, requests dispatched, cancellations delivered upstream, and jobs still running after cancellation. An RPS counter based on completed responses misses canceled requests. An origin request counter may miss work absorbed by the edge. A proxy's CPU and queue depth can rise while origin traffic is flat. If you only have L4 flow logs, you may see a small number of long-lived connections and miss the stream churn entirely.

An HTTP-aware edge can enforce per-connection churn limits, detect abnormal reset patterns, and stop costly dispatch earlier. A patch or upgrade to a server implementation may change cleanup behavior. An ordinary global IP-based rate limit can be a poor fit behind NAT because many real clients share an address, and an attacker can distribute across addresses. The [rate limiting and backpressure post](/blog/software-development/system-design/rate-limiting-and-backpressure) handles the general policy design; here the wire-specific question is which stage pays before the counter increments.

## Put the limiter before the resource it protects

The most common mistake in a DDoS design review is placing a correct policy at the wrong hop. A rule that rejects requests after a database query protects neither database CPU nor queue depth. A WAF that rejects an HTTP request after TLS termination protects the origin but still spends edge TLS and parser work. A host firewall that drops reflected UDP responses after they traverse the handoff cannot protect that handoff. A DNS failover policy does not help if all hostnames depend on the same unavailable authoritative platform.

![Layered defense placement showing upstream bandwidth control, connection validation, and request admission before application work](/imgs/blogs/network-attacks-you-will-actually-meet-4.webp)

For each control, write down three things: the first resource it can save, the work it has already paid to decide, and the legitimate traffic it might reject. An upstream filter can save transit capacity but has less application context. A TCP listener has flags and connection state but cannot distinguish an expensive URL. An HTTP edge can use path and header semantics but pays connection, TLS, and parsing cost. An origin has the richest business context and the worst position to protect the link. There is no universally best point; there is an earliest point with enough information to make a safe decision.

| Control | Resource saved first | Work already spent | Main false-positive risk |
| --- | --- | --- | --- |
| Provider filtering or scrubbing | Customer ingress link | Provider forwarding and classification | Broad rules discard legitimate shared-source traffic |
| SYN proxy or cookies | Pending TCP state | Packet receipt and SYN-ACK response | Option compatibility or proxy capacity |
| Proxy header deadline | Open sockets and parser slots | TCP and possibly TLS | Slow legitimate clients |
| HTTP/2 churn guard | Proxy dispatch and cleanup work | HTTP/2 parsing | Legitimate cancellations or unusual browser behavior |
| Application rate policy | Expensive handlers or dependencies | All earlier network and proxy work | Shared identities, NAT, or uneven user cost |

This table has no reported measurement. It is a causal comparison. Its practical use is to expose a proposed rule that acts after the first saturated resource. If the edge cannot classify safely with the information it has, distribute or expand its capacity while moving a narrower rule deeper, and monitor both stages. A robust design often combines cheap coarse filtering early with specific admission control later.

### Per-IP limits are a weak identity

An IP address is a routing identifier, not a stable person or tenant. Carrier-grade NAT and enterprise proxies concentrate many legitimate users behind one address. IPv6 clients may use privacy addresses. Botnets distribute across many addresses. A per-IP cap is still useful as one coarse signal, especially when protecting a scarce per-connection resource, but it is not a complete fairness policy. Prefer authenticated tenant or API-key budgets where available, and place a cheap unauthenticated guard before authentication work if that work itself is costly.

Avoid writing a single threshold into a runbook without the dimensions: requests per second, concurrent connections, bytes per second, cost per request, and reset rate are different budgets. Use a finite burst allowance because healthy clients are bursty. Track the rejected count and the success rate of clients behind large NATs. The side effect of aggressive limiting may be a new availability incident that looks just like the attack it was meant to prevent.

### Challenge, cache, shed, or block?

These controls solve different cost problems. A cache can remove backend work for repeatable responses, but it cannot stop ingress bytes or cache-busting requests. A challenge can separate some human traffic from automation, but it adds round trips and can exclude API clients or accessibility tools. Load shedding protects a dependent service by returning a bounded failure promptly; it does not make a full transit link usable. Blocking can stop recognizable unwanted traffic at the earliest appropriate boundary, but a brittle signature may reject real clients. During an incident, choose the control based on the resource at risk and preserve a rollback path.

The [cascading failures, circuit breakers, and bulkheads post](/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads) owns the application dependency side of the story. A network attack can trigger the same downstream cascade, but the initial discriminator is often on the wire: packets and handshakes arrive without a corresponding rise in successfully parsed application requests.

## Three incidents, three different places to look

![Comparison matrix mapping GitHub, Dyn, and HTTP/2 Rapid Reset to the resource first stressed and the useful measurement](/imgs/blogs/network-attacks-you-will-actually-meet-5.webp)

The cases are selected because they punish the same diagnostic mistake from three directions. GitHub needed upstream capacity and routing intervention for reflected traffic. Dyn's managed DNS was itself the attacked dependency, so users could fail before connecting to customer sites. HTTP/2 Rapid Reset stressed request processing despite connection concurrency controls. None is a template for assuming every outage is an attack; each gives a measurement that distinguishes it from a nearby alternative.

### GitHub, February 28, 2018: the choke point was upstream

GitHub's [incident report published March 1, 2018](https://github.blog/news-insights/company-news/ddos-incident-report/) states that GitHub.com was unavailable from 17:21 to 17:26 UTC on February 28 and intermittently unavailable until 17:30 UTC. The report attributes the event to a memcached UDP amplification attack. It reports a peak of 1.35 Tbps and 126.9 million packets per second. It also says the traffic originated from more than a thousand autonomous systems across tens of thousands of endpoints. These are GitHub's incident observations, not numbers we infer from protocol theory.

The trigger was reflected traffic directed at GitHub addresses. The contribution was an ecosystem of publicly reachable UDP memcached instances and spoofable source addresses, not a bug in a GitHub web handler. The immediate bottleneck showed up on transit. GitHub's monitoring noticed an inbound/outbound traffic-ratio anomaly, and the report says one facility saw inbound transit bandwidth rise above 100 Gbps. The response changed the route: at 17:26 UTC GitHub initiated withdrawal of transit BGP announcements and announced its ASN through Akamai links, where edge capacity and access lists could mitigate. GitHub reports full recovery at 17:30 UTC. The exact routing steps are specific to their network; the transferable principle is to have a prearranged upstream absorption path and a measured trigger for using it.

The blast radius was user availability, not data confidentiality or integrity according to the report. A host-level HTTP rate limit would not have prevented the transit pressure. The figure's first choke point is therefore before the web application. If you operate a smaller service, you need not reproduce GitHub's BGP arrangement. You do need to know who can act before your narrowest link, what signal activates them, and how you will verify the legitimate path after diversion.

### Dyn, October 21, 2016: the dependency was DNS authority

Dyn's [first-party statement about October 21, 2016](https://cyber-peace.org/wp-content/uploads/2016/10/Dyn-Statement-on-10_21_2016-DDoS-Attack.pdf) says its Managed DNS infrastructure was attacked in multiple waves. The first began about 7:00 a.m. Eastern Time and affected East Coast points of presence. The second, before noon, was more global. Dyn explicitly says there was no system-wide outage; some customer sites were unreachable for affected users, while other regions could still work. Dyn confirmed that devices infected by Mirai were one source of traffic, while its investigation described multiple attack vectors and locations. Do not turn that into a claim that Mirai alone explains every observed packet.

For a user, the symptom can resemble a dead web service: the browser cannot load the hostname. But the failure can happen at resolution before the TCP connect attempt. That distinction changes the first probes. Check the resolver's answer and error, query multiple authoritative nameservers where permitted, and compare regions. A direct request to an already-known origin IP can help isolate resolution, but it is not a substitute for the user's normal path because TLS name validation, CDN routing, and virtual hosting depend on the hostname. The [DNS production post](/blog/software-development/networking/dns-in-production-ttls-caching-layers-and-stale-answers) explains how cache state and TTLs can make impact uneven.

Cloudflare's [firsthand account of the Dyn outage's effect on Cloudflare](https://blog.cloudflare.com/how-the-dyn-outage-affected-cloudflare/) is useful as a second vantage point, not as Dyn's incident owner report. It describes how an upstream DNS dependency can affect systems that are otherwise functioning. The transfer lesson is to include authoritative DNS in the availability path and in incident probes. Origin scale-out does not restore a name clients cannot resolve. DNS provider diversity and failover can reduce concentration, but only if delegation, record state, and operational procedures are actually independent and rehearsed.

### HTTP/2 Rapid Reset, disclosed October 10, 2023: the queue hid behind cancellation

Cloudflare's [technical breakdown](https://blog.cloudflare.com/technical-breakdown-http2-rapid-reset-ddos-attack/) was published on October 10, 2023. It says attacks were observed beginning August 25 and eventually peaked just above 201 million requests per second. The event date and publication date are different. The mechanism was the ability to start an HTTP/2 request stream and cancel it quickly, allowing the client to create a high request rate without leaving many streams concurrently active. Cloudflare describes how its proxy could dispatch work to an upstream component before the reset caught up, causing load despite cancellation.

The user-visible effect included increased 502 errors in the most affected data centers, according to Cloudflare. The misleading first hypothesis would be that low active stream count means the stream limit is working. The actual counter needed is churn and work: opens, resets, dispatches, cancellations, queue length, and errors, all at the component that performs them. Cloudflare says it extended detection of client RST_STREAM behavior and made request-dispatch and queuing changes. It also describes the legitimate-client failure caused by lowering stream concurrency too aggressively. The transferable guardrail is to test mitigation against real client behavior, and to constrain expensive transitions rather than trusting a concurrent-state limit alone.

| Case and event date | Reported figure | Mechanism boundary | Source |
| --- | --- | --- | --- |
| GitHub, 2018-02-28 | Peak 1.35 Tbps and 126.9 Mpps | Reflected memcached UDP traffic stressed transit | [GitHub incident report, 2018-03-01](https://github.blog/news-insights/company-news/ddos-incident-report/) |
| Dyn, 2016-10-21 | No traffic-rate figure used here | Managed DNS attacked; Mirai devices were one confirmed source | [Dyn statement, 2016-10](https://cyber-peace.org/wp-content/uploads/2016/10/Dyn-Statement-on-10_21_2016-DDoS-Attack.pdf) |
| Cloudflare, attacks beginning 2023-08-25 | Peak just above 201 million requests/s | HTTP/2 stream churn stressed proxy request processing | [Cloudflare technical report, 2023-10-10](https://blog.cloudflare.com/technical-breakdown-http2-rapid-reset-ddos-attack/) |

## A diagnostic runbook for the first ten minutes

![Decision tree from client symptom and first saturated counter to an appropriate early defense](/imgs/blogs/network-attacks-you-will-actually-meet-6.webp)

The decision tree starts with one question: what is the earliest stage that stops making progress? Do not start by deciding which attack name sounds plausible. A rate spike may be legitimate demand, a retry storm, a deployment bug, or hostile traffic. The first task is to preserve evidence and service while separating those possibilities.

First, run a client probe that records DNS, connection, TLS, first-byte, and total timings. Run it from at least two independent network locations if available, because a regional provider or DNS failure can look global from one office. Check the status and actual response, not just `curl` exit code. Keep the timestamp in UTC so it can be aligned with edge and host metrics.

Second, read the upstream boundary. Ask whether provider ingress bps or pps, interface drops, or mitigation events changed before application error rate. If the link is at capacity, engage the upstream path immediately and avoid trying to fix it by only changing application settings. Record the exact provider interface and counter definition. A graph that sums across regions can hide one saturated point of presence.

Third, if the link is healthy, inspect connection state. Compare SYN-RECV, accepted sockets, listener drops, syncookie counters, TLS handshake rates, and proxy connection age. A rising incomplete-handshake state with failing new connections suggests a TCP-state issue. Many established, very old, low-progress connections suggest a slow-request issue. A large number of completed handshakes with high proxy CPU points higher. Do not change global sysctls during this diagnostic step.

Fourth, compare HTTP requests opened, completed, reset, and dispatched. If the active request gauge is low but create/reset rates and queue depth are high, inspect protocol churn such as Rapid Reset. If a small set of expensive routes dominates backend time, an application-aware admission control or cache policy may help. If origin metrics are quiet while client DNS fails, leave the origin alone and inspect DNS authority.

Finally, choose a reversible control and measure collateral damage. Record both blocked or challenged traffic and successful legitimate requests. Test from a shared NAT, a slow network, and a normal browser path if those populations matter. A defense that protects CPU but breaks real clients has changed the failure mode, not restored availability. Keep the rollback command or provider action next to the activation command in the runbook.

### A compact incident evidence sheet

Write down the same fields every time. The discipline prevents a famous attack label from substituting for observed facts.

| Field | Record during the incident |
| --- | --- |
| User symptom | Resolver error, connect timeout, TLS error, HTTP status, or high first-byte time |
| First bad boundary | DNS authority, provider handoff, edge, listener, proxy, origin, dependency |
| Arrival and completion | bps, pps, SYNs, completed handshakes, opened and completed requests |
| State and queue | SYN backlog, established sockets, active streams, dispatch queue, worker saturation |
| Control and location | Exact rule, component, activation time, and upstream dependencies |
| Collateral effect | Legitimate success rate, false positives, latency, and rollback signal |

These are categories, not invented benchmark values. In particular, avoid a single "attack traffic" line that mixes wire packets, accepted connections, and HTTP requests. Their ratios tell you where work disappears. If 100,000 packets arrive and only 1,000 HTTP requests complete, the missing work may have been dropped or rejected at several stages. You need stage-specific counters to attribute the difference.

### Compare rates across adjacent stages

Build a short accounting chain for a single window: packets received at the provider handoff, packets admitted by the edge, TCP handshakes completed, HTTP requests opened, HTTP requests dispatched, and responses completed. These are not expected to be equal. One HTTP/2 connection carries many requests. Retransmissions add packets without adding requests. A cached response may finish at the edge without reaching origin. A canceled request may be opened but never produce a response. The point is to notice a *change in the relationship* under incident conditions, then explain it with a corresponding counter.

For example, suppose a hypothetical ten-second baseline has 100,000 packets at the edge and 5,000 opened requests, while the incident window has 500,000 packets and still 5,000 opened requests. That comparison alone says the extra packets did not become opened requests. It does not identify the cause. Check edge drops, SYN attempts, retransmissions, and protocol distribution. In another hypothetical window, packets remain near 100,000 but opened requests rise to 40,000 while completed responses fall. That points toward request-level work or cancellation rather than a pure link-capacity problem. These figures are illustrative, derived from the stated windows, and are not a production trace. They show why stage ratios are more informative than any single line.

Beware of dashboard sampling. A one-minute average can hide a short burst that fills a queue long enough to cause user timeouts. A provider may report a peak at a one-second resolution while your application panel uses five-minute bins. Align interval length, timestamp, timezone, and aggregation before comparing. Use a rate calculated from counter deltas over the same period when possible. If a counter resets after a process restart, a naive difference can turn negative or produce a false spike. Keep raw counter samples in the incident record so another engineer can reconstruct the rate.

Sampling also changes what a packet capture can prove. A narrow capture on the origin sees only packets that reached the origin. It cannot prove that the provider received no other traffic. Conversely, a provider flow sample may characterize a large attack but miss the exact HTTP frame sequence that made one proxy expensive. Choose the observation point based on the disputed mechanism. Do not collect full payloads just because the capture command is convenient. Capture traffic for a specific interface, port, and short duration, with access controls, then stop when the question is answered.

### Distinguish hostile traffic from your own retry storm

A demand spike is not automatically an attack. A dependency failure can make well-behaved clients retry, and retries can consume the same connection and request resources as hostile traffic. The shape of the traffic helps. Did client software or a deploy change just before the surge? Are requests authenticated and distributed across normal tenants? Do traces show the same request attempted repeatedly after a timeout? Did upstream errors rise *before* incoming request rate, suggesting failures triggered retries? A traffic source can be legitimate and still overload the service. The immediate control may be load shedding and retry backoff rather than a permanent deny rule.

The distinction matters because a source-IP block can hurt the exact users who are retrying from a shared gateway. If a service returns a slow 503 and clients retry immediately, a harsher edge rule may make the dashboard look cleaner while user success falls further. Prefer an explicit `Retry-After` where clients honor it, idempotency for operations that can be repeated, and bounded exponential backoff in clients you control. The [timeouts, retries, and backoff post](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) owns those application policies. Here the network observation tells you whether admitted request load is self-generated after another stage failed.

There is also a middle ground: apparently normal HTTP traffic can be abusive because it disproportionately invokes an expensive path. A route that runs a full-text search or cold storage fetch might cost orders of magnitude more than a cached health check. A global requests-per-second limit treats them equally and therefore prices the wrong resource. Instrument work per route and tenant: CPU time, downstream calls, bytes read, and queue residence. Put a cheap admission decision before the expensive step. If the expense is not visible until after authentication or parsing, cap the earlier unauthenticated work separately and apply the more precise budget once identity is known.

### Understand why a defense can move the bottleneck

Traffic does not disappear when a control changes the path. It can shift. If an edge challenges suspicious clients, challenge generation and verification become new work. If a CDN bypasses cache for a path, origin demand rises. If a SYN proxy validates connections, the proxy must handle enough SYN-ACK traffic and maintain its own policy state. If a provider diverts routes to a scrubbing center, route convergence and the return path can add latency or alter source visibility. A good rollout watches the next hop for new queueing rather than declaring success when the original counter falls.

Consider a hypothetical 10,000 request-per-second arrival stream. If a cheap edge rule rejects 80% before application dispatch, then $10{,}000\times(1-0.8)=2{,}000$ requests per second reach the application, assuming the rule and arrival mix remain stable. This is a derived example, not a claim about any mitigation vendor. If each accepted request still fans out to five synchronous backend calls, the backend receives a modeled 10,000 calls per second. That may simply move the bottleneck from application admission to the backend. The model also assumes rejection is cheap enough that the edge can sustain all 10,000 arrivals. Measure rule-evaluation CPU and queue depth as well as origin relief.

The same accounting applies to cancellation. A canceled HTTP/2 stream saves downstream work only if cancellation reaches the worker before the expensive step completes and the worker responds to it. If dispatch is async and cancellation waits behind queued work, the protocol state can look clean while backend work still runs. Correlate a request identifier across edge acceptance, dispatch, cancellation, and backend completion. A missing or delayed cancellation event is a concrete implementation bug or capacity issue, not a reason to lower a concurrency setting without testing.

### Preserve a clean rollback

Emergency controls often outlive the emergency unless their expiry is explicit. Put an owner and review time on temporary source blocks, path rules, connection limits, and provider diversions. Record the baseline value and the exact changed value. For a stateful configuration, test rollback in the same environment before the incident if possible. A limit that seems safe under attack can reject a normal traffic peak the next morning. A diversion that helped absorb traffic can leave an unexpected routing dependency after the attack stops.

Use a canary population when the component supports it. Apply a more restrictive HTTP policy to a small fraction of edge traffic, then compare legitimate completion, reset rate, 4xx and 5xx rates, and tail latency with an unaffected population. During a severe outage you may not have time for a slow experiment, but you can still choose a reversible control and watch its effect minute by minute. Label the moment of activation on every graph. Without that marker, later analysts may attribute the recovery to the wrong action and preserve an unnecessary rule.

Finally, make the page useful after the incident. Store the source of each graph, the resolution and unit of each counter, the observed ordering of symptoms, and the exact action that restored user success. A postmortem that says only "large DDoS mitigated" cannot improve the next response. The useful artifact states where the first queue formed, how you proved it, which control acted before that queue, and what collateral effect you measured.

### Build a capacity envelope before an incident

An on-call engineer should not have to discover the normal operating range during a flood. Record a modest capacity envelope for every public ingress point: ordinary peak ingress bps and pps, TLS handshakes per second, concurrent accepted sockets, HTTP request opens and completions, proxy queue length, and origin concurrency. These are measurements from your own service, not universal thresholds. Add the software version, hardware or cloud instance shape, region, and sampling interval. A number copied from another company's incident has no threshold value for your deployment.

Use a load test only on infrastructure you own and within its agreed limits. A controlled test can tell you where latency bends upward as offered work increases, but it cannot safely simulate every hostile traffic shape. A synthetic HTTP request generator may miss the cost of partial headers. A normal HTTP/2 benchmark may have low reset churn. A local test will not reveal whether your transit provider can absorb a reflected UDP event. Separate what you have measured from what you are trusting a provider to supply, and obtain the provider's escalation procedure in advance.

For each envelope, note *which resource* the number describes. Suppose, as a derived planning example, a proxy completes 8,000 ordinary requests per second in a controlled test while a backend completes 3,000 expensive search operations per second. If ordinary traffic contains 10% search and the rest cached or cheap routes, 8,000 incoming requests imply 800 search operations per second. Under that mix the backend has headroom. If the mix changes to 50% search at the same overall request rate, expected search demand becomes 4,000 per second, above the stated 3,000 capacity. The attack need not increase total RPS to create a backend queue. This is illustrative arithmetic based on assumed test results, not a measured service.

The control should reflect the expensive dimension. A path-specific or tenant-specific search budget could protect the backend while cheap traffic remains available. It has to sit before the search work begins, but after the proxy knows enough to identify the route and tenant. If authentication itself is costly, place a separate cheap guard before it. Test both budgets with real request mixes and observe latency, rejection rates, and downstream work. A single global RPS cap at 8,000 would not protect the example backend under the changed mix, even though it matches the proxy's ordinary-request capacity.

An envelope also needs failure behavior. What happens when the DNS provider is unreachable? How long can existing resolvers answer from cache, and what happens after their cached records expire? What does the edge return when the origin is unavailable? Does the application fail fast when a dependency queue grows, or does it retain accepted requests until clients give up and retry? The answers determine whether one initial bottleneck fans out into a larger outage. The [reliability SLOs and graceful degradation post](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) covers the policy boundary; here the point is to identify the packet or request stage where a graceful failure must begin.

Runbook ownership matters as much as the graph. Identify who can change provider filtering, who owns DNS delegation, who can tune the edge proxy, and who can roll back application admission. Record how to reach each team when the primary chat or DNS path is impaired. A control that requires an unavailable control plane is not an effective incident control. GitHub's report is especially instructive here: it describes monitoring, a ChatOps routing action, and an upstream partner, not merely an attack signature. The mechanism and organizational path were both necessary for recovery.

## Run it yourself

The safe experiment demonstrates one necessary part of the main claim: an accepted HTTP connection can occupy state before a complete request is counted. It does not generate a SYN flood, spoof packets, or reproduce Rapid Reset. Use a Linux lab namespace from [post 1](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). `ip netns` and `ss` require Linux; namespace setup needs root or `CAP_NET_ADMIN`. Run this only inside the lab namespaces `c` and `s`. The small client cap keeps it a diagnostic demonstration rather than a load test.

### Question

Can the server hold established TCP connections with incomplete HTTP headers while its completed-request count stays at zero? If so, a completed-request limiter cannot be the first control for this resource.

### Preconditions

The lab has namespaces `c` and `s`, client `c0` at `10.77.0.1/30`, server `s0` at `10.77.0.2/30`, and Python 3 installed. The commands below use an isolated listener at `10.77.0.2:8080`; check that port is free in `s`. The server script uses Python's standard library and handles each accepted connection with an async task. It counts a request only after the header terminator arrives. This is intentionally a toy parser for observation, not production HTTP server code.

```bash
set -euo pipefail
ip netns list | rg '^(c|s)( |$)'
ip netns exec c ip -brief address show dev c0
ip netns exec s ip -brief address show dev s0
ip netns exec c ip route get 10.77.0.2
ip netns exec s ss -lnt '( sport = :8080 )'
python3 --version
```

The expected route is through `c0`, and the `ss` output should show no listener on port 8080 before the experiment. If a listener exists, stop instead of killing an unknown process. If the namespaces are absent, complete the setup in post 1 first. This lab changes no qdisc, route, firewall, or sysctl state.

### Baseline

Create a temporary server file with a header deadline and counters. The `timeout` is an explicit bound on the waiting resource. The server prints `accepted`, `complete`, `timeout`, and `active` fields after each connection closes. Start it only in namespace `s` and retain its PID for scoped cleanup.

```bash
set -euo pipefail
LAB_DIR="$(mktemp -d /tmp/netattacks24.XXXXXX)"
cat >"$LAB_DIR/server.py" <<'PY'
import asyncio

accepted = 0
complete = 0
timed_out = 0
active = 0

async def handle(reader, writer):
    global accepted, complete, timed_out, active
    accepted += 1
    active += 1
    try:
        await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=8.0)
        complete += 1
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nOK")
        await writer.drain()
    except (asyncio.TimeoutError, asyncio.IncompleteReadError):
        timed_out += 1
    finally:
        active -= 1
        writer.close()
        await writer.wait_closed()
        print(f"accepted={accepted} complete={complete} timeout={timed_out} active={active}", flush=True)

async def main():
    server = await asyncio.start_server(handle, "10.77.0.2", 8080)
    async with server:
        await server.serve_forever()

asyncio.run(main())
PY
ip netns exec s python3 "$LAB_DIR/server.py" >"$LAB_DIR/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" >"$LAB_DIR/server.pid"
sleep 1
ip netns exec s ss -lnt '( sport = :8080 )'
ip netns exec c curl --max-time 3 -sS -o /dev/null -w 'status=%{http_code}\n' http://10.77.0.2:8080/
cat "$LAB_DIR/server.log"
```

Read the `ss` listener row and the server log. Expected: one `status=200`, then `accepted=1 complete=1 timeout=0 active=0`. The exact listener backlog fields vary by kernel. The baseline proves the toy server can finish a normal request. Keep `LAB_DIR` and `SERVER_PID` in the same shell for the next steps.

### Apply one change

Now the client opens four connections and sends incomplete request headers. It does not send the blank line that ends the header section. Four is an intentionally small, local count. The client sleeps long enough to observe the sockets, then exits. Do not point this code at a public endpoint or raise the count; the purpose is resource accounting in your own namespace.

```bash
set -euo pipefail
cat >"$LAB_DIR/slow.py" <<'PY'
import socket
import time

sockets = []
for _ in range(4):
    sock = socket.create_connection(("10.77.0.2", 8080), timeout=2)
    sock.sendall(b"GET / HTTP/1.1\r\nHost: lab.local\r\nX-Wait: ")
    sockets.append(sock)
print(f"open_incomplete={len(sockets)}", flush=True)
time.sleep(5)
for sock in sockets:
    sock.close()
PY
ip netns exec c python3 "$LAB_DIR/slow.py" >"$LAB_DIR/client.log" 2>&1 &
CLIENT_PID=$!
sleep 1
ip netns exec s ss -Hnt state established '( sport = :8080 )' | wc -l
cat "$LAB_DIR/client.log"
```

Read the established socket count while the client is sleeping. Expected: about four server-side established sockets, with zero new complete requests in the log at that moment. A count below four can occur if local scheduling delays the client or a connection fails; inspect `client.log` and repeat once rather than asserting an exact kernel timing. The treatment changed only header completion, not the destination, server, or namespace path.

### Compare

Wait for the client and the server's bounded header deadline, then inspect the log again. The server should release the state without counting a completed request. The precise exception path can be timeout or incomplete read, depending on whether the client closes first; both are recorded in the toy `timeout` field.

```bash
set -euo pipefail
wait "$CLIENT_PID"
sleep 4
cat "$LAB_DIR/server.log"
ip netns exec s ss -Hnt state established '( sport = :8080 )' | wc -l
```

Expected: the final log reaches `accepted=5 complete=1`, `timeout=4`, and `active=0`, with zero established server-side sockets after cleanup. Those are expected values for this exact five-connection lab, not a benchmark. TCP close timing can leave a transient socket in another state; the established count should settle to zero. The observation proves why a completed-request metric misses waiting header state. Production proxies may use different parsers and deadlines, so verify the equivalent counter and timeout in the component you operate.

### Reset

Stop only the PID started for this lab and remove only its temporary directory. This does not delete namespaces or change the shared series setup. If the shell was restarted and lost `SERVER_PID`, read the saved PID from the specific `LAB_DIR` only after confirming it is the expected `server.py` process.

```bash
set -euo pipefail
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null || true
rm -rf -- "$LAB_DIR"
```

For a real host, use read-only observations such as `ss -nt state established`, proxy active-connection and header-timeout metrics, and a bounded capture with a narrow filter. Packet captures can contain credentials, tokens, personal data, and request bodies; restrict access and retention. Do not run the lab's namespace commands, parser, or client against an unspecified production interface or endpoint.

## What to change after you know the resource

**Choose a control by its position on the path and its false-positive cost.** If provider ingress is dropping, prearrange upstream filtering or scrubbing and a tested diversion path. If SYN state is the constraint, verify cookie or SYN-proxy behavior and track successful handshakes, not just blocked packets. If slow accepted connections are the problem, use bounded header and total deadlines at the proxy that owns them. If HTTP/2 stream churn is the problem, update the affected implementation, observe reset and dispatch rates, and apply a tested churn guard. If a route is expensive, place admission before the expensive dependency and use application identity where available.

No one control replaces capacity planning. An upstream provider must still absorb traffic long enough to classify it. A proxy must have CPU headroom to inspect HTTP. An application limit must be backed by a clear fairness policy. Your best incident action may be a temporary coarse rule plus a narrower follow-up rule after evidence arrives. Note the time of every rule change, graph both attack and legitimate traffic, and roll back when collateral harm exceeds protection.

There is also a prevention layer outside the victim's network. Operators should not expose UDP services that can reflect large responses without a reason. Access networks should validate customer source addresses as in RFC 2827. Protocol implementers should make cancellation cheap and bounded. These actions do not give any single service immunity, but they remove multiplication opportunities before they become somebody else's incident.

## Key takeaways

- Find the earliest saturated resource: DNS authority, transit, packet processing, TCP state, accepted sockets, proxy dispatch, or application work.
- Measure both bits and packets at network boundaries. Measure opened, canceled, dispatched, and completed work separately at HTTP boundaries.
- Reflection's byte ratio belongs to a defined request and response measurement. A protocol maximum is not an incident's observed ratio.
- SYN cookies protect pending TCP state under specific conditions. They do not create upstream bandwidth or application capacity.
- A stream concurrency limit does not necessarily bound stream creation rate. Rapid cancellation can leave expensive work behind.
- Put each limiter before the resource it is meant to protect, and measure legitimate traffic after activating it.

## Further reading

- [RFC 4987: TCP SYN flooding attacks and common mitigations](https://www.rfc-editor.org/rfc/rfc4987)
- [RFC 2827: network ingress filtering](https://www.rfc-editor.org/rfc/rfc2827)
- [RFC 9113: HTTP/2](https://www.rfc-editor.org/rfc/rfc9113)
- [OWASP Denial of Service Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Denial_of_Service_Cheat_Sheet.html)
- [GitHub's February 28, 2018 DDoS incident report](https://github.blog/news-insights/company-news/ddos-incident-report/)
- [Dyn's statement on the October 21, 2016 attack](https://cyber-peace.org/wp-content/uploads/2016/10/Dyn-Statement-on-10_21_2016-DDoS-Attack.pdf)
- [Cloudflare's HTTP/2 Rapid Reset technical breakdown](https://blog.cloudflare.com/technical-breakdown-http2-rapid-reset-ddos-attack/)
