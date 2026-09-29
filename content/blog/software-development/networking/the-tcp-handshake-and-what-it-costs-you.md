---
title: "The TCP Handshake and What It Costs You"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn to price a TCP handshake in RTTs, distinguish Linux's two listener queues, and prove which queue dropped a connection."
tags:
  [
    "networking",
    "distributed-systems",
    "tcp",
    "linux",
    "latency",
    "socket-programming",
    "observability",
    "performance",
    "syn-cookies",
    "tcp-fast-open",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 42
image: "/imgs/blogs/the-tcp-handshake-and-what-it-costs-you-1.webp"
---

A service can be healthy, idle, and ready to answer in two milliseconds while a caller still spends eighty milliseconds doing no application work. The missing time lives one rung below HTTP. A new TCP connection has to cross the path, come back, and cross it again before the ordinary request is useful to the server. That first round trip is easy to ignore on a laptop and impossible to hide across regions.

The latency ladder is the opening map for this investigation. It keeps DNS, TCP, TLS, request transfer, server work, first byte, and response transfer separate. The diagram below is the mental model: this post lights the TCP rung, then follows that rung down into packets, kernel queues, counters, and the application's `accept()` loop.

![Latency ladder with the TCP connection segment highlighted](/imgs/blogs/the-tcp-handshake-and-what-it-costs-you-1.webp)

This is not a post about memorizing three arrows. The operational question is what happens when one of those arrows meets a slow application, a full listener, spoofed traffic, uneven workers, or a middlebox that dislikes data in a SYN. By the end, we will be able to price the clean path, identify the queue that is full, and choose a measurement before choosing a tuning knob.

If you want the complete path around this layer, start with [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The handshake is one component of that request, not the whole request.

## Put the handshake on the latency ladder

**Senior rule: count network crossings before counting packets.** Packet count is not latency. Dependencies between packets are latency.

Suppose `ping` in the series `netlab` reports an RTT between 78 and 84 ms after we configure an intended 80 ms path. A conventional active open proceeds like this:

1. The client sends SYN at elapsed time zero.
2. The server receives it after roughly half an RTT and returns SYN-ACK.
3. The client receives SYN-ACK after roughly one RTT, enters `ESTABLISHED`, and sends ACK.
4. The client can attach ordinary request data to that third packet, depending on the API and write timing. The server receives it after another one-way propagation interval.

The application-visible cost depends on which boundary we name. A blocking `connect()` usually returns at the client when SYN-ACK arrives and the kernel can emit the final ACK, so its clean-path duration is approximately one RTT. A server cannot consume a conventional request until request bytes arrive. If the caller waits for `connect()` and only then writes, the first request bytes reach the server after approximately one and a half RTTs from the original SYN. If we measure until the first response byte returns, we must add request transit, server time, and response transit too.

That distinction prevents a common accounting error. Engineers say "the TCP handshake costs three packets," then accidentally price it as three RTTs. The three messages contain one causal round trip before the client learns the server's initial sequence number. The final ACK does not need its own reply before the client writes.

For an 80 ms RTT and a two millisecond handler, this explanatory approximation puts client `connect()` completion at 1.0 RTT, or 80 ms. The request reaches the server at about 1.5 RTT, or 120 ms. The first response byte reaches the client at about 2.0 RTT plus two milliseconds of handler work, or 162 ms.

The approximation assumes symmetric 40 ms one-way delay, no loss, negligible serialization for the small control packets, and a request sent immediately with the third packet. It is a model, not a formula promised by TCP. Asymmetric routes, delayed application scheduling, SYN retransmission, TLS, proxy hops, and request size all move the observed number.

Connection reuse changes the ledger. A warm HTTP keep-alive connection pays zero new TCP RTTs for the next request. A connection pool that already holds an established socket removes the highlighted rung rather than making it faster. This is why connection lifetime often matters more than micro-optimizing the listener. It is also why a cold-start benchmark and a warm throughput benchmark answer different questions.

| Request condition | TCP setup before request | Derived clean-path consequence | Source |
| --- | ---: | --- | --- |
| New conventional connection | 1 RTT to complete `connect()` | About 80 ms on the reproducible 80 ms path | Derived here from configured and verified RTT |
| Reused established connection | 0 RTT | No new TCP establishment delay | Derived from reuse of existing state |
| Later connection using valid TFO cookie | Request data may ride in SYN | Up to 1 RTT saved for suitable application data | [RFC 7413, December 2014](https://www.rfc-editor.org/rfc/rfc7413.html) |
| Lost initial SYN | At least one retransmission wait | Dominated by the client's SYN retry timer, not clean RTT | Reproducible with packet loss; inspect `TCPSynRetrans` |

The phrase "up to" matters. TCP Fast Open can move suitable data earlier, but it does not repeal path delay, server work, or response transit. It also has a first-connection cookie acquisition path and replay constraints. We will come back to those boundaries.

### The second-order cost is amplification

One cold connection is usually affordable. Thousands of short-lived connections can turn one RTT into a capacity problem. Each new connection creates packet processing, request state, queue occupancy, timers, and eventually a connected socket. If a caller opens a connection per request, latency and listener work scale together.

The useful derived quantity is the number of handshakes in flight. As an approximation based on Little's Law, incomplete handshakes equal new connections per second multiplied by handshake residence time.

At 20,000 new connections per second and an 80 ms residence time, the average is approximately `20,000/s x 0.080 s = 1,600` incomplete handshakes before allowing for retries, bursts, scheduler delay, or attacks. That is not a recommendation for a 1,600-entry queue. It is a reason to measure arrival rate and residence time together. A burstier workload needs headroom above its average. A spoofed SYN never supplies the final ACK, so it can remain until retry policy expires rather than for one clean RTT.

This connects the latency ladder to capacity. RTT is not only what the caller waits for. RTT also controls how long the server retains handshake state under normal traffic.

## What the three packets establish

**Senior rule: the handshake synchronizes state; it does not merely ask permission.**

![TCP three-way handshake packet timeline and state transitions](/imgs/blogs/the-tcp-handshake-and-what-it-costs-you-2.webp)

[RFC 9293, published in August 2022](https://www.rfc-editor.org/rfc/rfc9293.html), specifies TCP's connection-establishment behavior. Each endpoint chooses an initial sequence number, commonly abbreviated ISN. The client must learn the server's ISN and confirm it. The server must learn the client's ISN and confirm it. Combining the server's SYN with its acknowledgment produces the familiar three-message exchange.

Let the client's initial sequence number be `x` and the server's be `y`. The first message is a client-to-server SYN with `seq=x`. The second is a server-to-client SYN-ACK with `seq=y, ack=x+1`. The third is a client-to-server ACK with `ack=y+1`.

The `+1` surprises people because the SYN may carry no application payload. SYN consumes one position in TCP's sequence space. So does FIN. That accounting lets acknowledgments unambiguously confirm the control event.

On the client, an active open moves from `CLOSED` to `SYN-SENT` when SYN leaves. Receipt of an acceptable SYN-ACK confirms the client's sequence space, teaches the server's sequence space, and permits the client to enter `ESTABLISHED` while returning ACK. On the passive side, the listener receives SYN, creates request state in `SYN-RECV`, and sends SYN-ACK. The final acceptable ACK permits creation or promotion of the connected child socket to `ESTABLISHED`.

The listening socket itself stays in `LISTEN`. `accept()` does not transform the listener into a connected socket. It removes a completed connection from the pending queue and returns a new file descriptor for that connection. The distinction matters during incidents because the kernel can finish handshakes while the application is too slow to accept them.

### What the handshake protects against

The exchange gives both endpoints evidence that packets can make the round trip and that the peer saw the current sequence choice. It reduces the risk that an old duplicate segment is mistaken for a fresh connection. It negotiates options carried in SYN and SYN-ACK, such as maximum segment size, window scaling, selective acknowledgment permission, and timestamps when supported.

It does not authenticate the application peer. An on-path observer can see and manipulate ordinary TCP. TLS owns cryptographic server identity and confidentiality. The [service-to-service security boundary](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) belongs above TCP, even though its handshake adds more network dependencies.

It also does not prove that the application is ready. A completed TCP handshake proves kernel transport state exists. The process might not call `accept()` promptly. It might accept and then stall before reading. A proxy might complete downstream TCP while its upstream is unavailable. Health checks that only connect to a port therefore prove less than many dashboards imply.

### Packet capture settles order, not ownership

A capture is the best evidence for packet order:

```bash
sudo tcpdump -ni any -s 128 -tttt \
  'host 10.77.0.2 and tcp port 8080 and (tcp[tcpflags] & (tcp-syn|tcp-ack) != 0)'
```

The filter is intentionally scoped to the lab endpoint and port. A production capture needs a short duration, a narrow filter, appropriate access control, and a storage plan because packets can contain sensitive metadata or payload. `tcpdump` usually requires root or `CAP_NET_RAW`.

Read the elapsed gap from SYN to SYN-ACK at the capture point. Then compare the sequence and acknowledgment fields. A SYN retransmission without a SYN-ACK narrows the failure toward the forward path, server ingress, listener lookup, SYN queue pressure, or return-path visibility. Repeated SYN-ACK followed by no final ACK narrows it toward the client, reverse delivery, asymmetric filtering, or spoofed source traffic.

The capture cannot by itself tell us whether a Linux accept queue was full. For that we need socket state and kernel counters at the server. Evidence becomes strong when packet behavior and counter deltas agree.

## TCP Fast Open moves data into the handshake

**Senior rule: removing a dependency changes semantics, not only latency.**

<figure class="blog-anim">
<svg viewBox="0 0 900 430" role="img" aria-label="Standard TCP request data waits for the handshake while later TCP Fast Open sends replay-safe data with the SYN" style="width:100%;height:auto;max-width:900px">
<style>
.tcp7-title{font:700 20px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.tcp7-label{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.tcp7-note{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}.tcp7-line{stroke:var(--border,#d1d5db);stroke-width:2}.tcp7-packet{fill:var(--accent,#6366f1)}.tcp7-data{fill:var(--surface,#f3f4f6);stroke:var(--text-primary,#1f2937);stroke-width:1.5}.tcp7-fast{fill:var(--accent,#6366f1);opacity:.18;stroke:var(--accent,#6366f1);stroke-width:2}@keyframes tcp7-standard{0%,12%{transform:translateX(0);opacity:0}22%,42%{opacity:1}52%,100%{transform:translateX(650px);opacity:1}}@keyframes tcp7-fastopen{0%,12%{transform:translateX(0);opacity:0}22%,100%{transform:translateX(650px);opacity:1}}.tcp7-standard-move{animation:tcp7-standard 10s ease-in-out infinite alternate}.tcp7-fast-move{animation:tcp7-fastopen 10s ease-in-out infinite alternate}@media (prefers-reduced-motion:reduce){.tcp7-standard-move,.tcp7-fast-move{animation:none;opacity:1;transform:translateX(650px)}}
</style>
<text class="tcp7-title" x="20" y="30">When request data is allowed to move</text>
<text class="tcp7-label" x="20" y="82">Standard cold connection</text>
<text class="tcp7-note" x="20" y="108">SYN, SYN-ACK, ACK first; ordinary request data follows</text>
<line class="tcp7-line" x1="180" y1="145" x2="830" y2="145"/>
<circle class="tcp7-packet tcp7-standard-move" cx="180" cy="145" r="11"/>
<rect class="tcp7-data" x="560" y="120" width="130" height="50" rx="8"/><text class="tcp7-label" x="625" y="151" text-anchor="middle">request data</text>
<text class="tcp7-note" x="180" y="186">client</text><text class="tcp7-note" x="792" y="186">server</text>
<text class="tcp7-label" x="20" y="252">Later TCP Fast Open connection</text>
<text class="tcp7-note" x="20" y="278">valid cookie learned earlier; only replay-safe data rides in SYN</text>
<line class="tcp7-line" x1="180" y1="315" x2="830" y2="315"/>
<g class="tcp7-fast-move"><rect class="tcp7-fast" x="145" y="287" width="150" height="56" rx="10"/><text class="tcp7-label" x="220" y="310" text-anchor="middle">SYN + cookie</text><text class="tcp7-note" x="220" y="331" text-anchor="middle">replay-safe data</text></g>
<text class="tcp7-note" x="180" y="365">client</text><text class="tcp7-note" x="792" y="365">server</text>
<text class="tcp7-note" x="450" y="405" text-anchor="middle">Fast Open can remove one request round trip; it does not remove replay constraints.</text>
</svg>
<figcaption>The standard cold path waits for the third handshake packet before request data; a later TFO connection can carry replay-safe data in the SYN after the client has learned a cookie.</figcaption>
</figure>

Conventional TCP permits data in a SYN but withholds it from the application until the three-way handshake validates the connection. [RFC 7413, published in December 2014](https://www.rfc-editor.org/rfc/rfc7413.html), defines TCP Fast Open, or TFO, so a server can accept data during the handshake when the client presents a valid Fast Open cookie.

The first interaction is still a cookie-learning connection. A client sends a SYN requesting a TFO cookie. A supporting server returns an opaque cookie in SYN-ACK. On a later connection, the client can place that cookie and application data in SYN. If the server validates the cookie, it may deliver the data early and acknowledge both SYN and data. This can remove up to one full RTT from a suitable transaction.

"Suitable" is doing heavy work. TFO data can be replayed. RFC 7413 explicitly warns applications not to send operations that cannot tolerate replay in a SYN. A read-like, idempotent request may be a candidate. A payment mutation or non-idempotent job submission is not safe merely because its HTTP method happens to be named one way. Replay safety is an application invariant.

| Property | Conventional cold TCP | Later TFO connection with valid cookie | Source |
| --- | --- | --- | --- |
| Earliest ordinary request delivery | After handshake validation | During handshake | [RFC 7413, December 2014](https://www.rfc-editor.org/rfc/rfc7413.html) |
| Latency opportunity | No setup RTT removed | Up to one RTT saved | [RFC 7413](https://www.rfc-editor.org/rfc/rfc7413.html) |
| Prior state | None required | Client needs a server cookie from an earlier exchange | [RFC 7413](https://www.rfc-editor.org/rfc/rfc7413.html) |
| Duplicate-data concern | Application data waits for validation | Early data can be replayed | [RFC 7413 Section 2.1](https://www.rfc-editor.org/rfc/rfc7413.html#section-2.1) |
| Middlebox fallback | Normal SYN path | Client may retry without data or TFO option | [RFC 7413 Section 4.2](https://www.rfc-editor.org/rfc/rfc7413.html#section-4.2) |

TFO is therefore not a switch labeled "one RTT faster." It is a protocol and application decision. The client and server need support. The service port needs explicit enablement. The operation needs replay analysis. The path needs to tolerate the option and data-bearing SYN. Observability needs to distinguish successful Fast Open from fallback.

On Linux, useful `nstat` fields include `TcpExtTCPFastOpenActive`, `TcpExtTCPFastOpenActiveFail`, `TcpExtTCPFastOpenPassive`, and failure-specific fields exposed by the running kernel. Always inspect available names rather than assuming a version has every counter:

```bash
nstat -az | grep -E 'TCPFastOpen|TCPSynRetrans'
sysctl net.ipv4.tcp_fastopen
uname -r
```

An increasing `TcpExtTCPFastOpenActiveFail` says attempts failed because the remote side did not accept them or the attempt timed out, according to the [Linux kernel SNMP counter documentation](https://kernel.org/doc/html/next/networking/snmp_counter.html). It does not say which middlebox or endpoint caused the failure. Pair it with a capture and a conventional connection control group.

### Fast Open does not fix connection churn

If a service opens a fresh connection for every small request, TFO may hide part of the latency while leaving connection creation, listener work, state allocation, and cleanup intact. Pooling and reuse eliminate more work when the protocol and failure model permit them. TFO is useful when new connections are genuinely necessary and repeated client-server contact makes cookie reuse likely.

This is a recurring systems lesson: make the dependency unnecessary before making it faster. The later post on [reliability, sequence numbers, ACKs, retransmits, and RTO](/blog/software-development/networking/reliability-sequence-numbers-acks-retransmits-and-rto) follows what happens after establishment when packets disappear.

## One listener, two queues

**Senior rule: say which queue before saying backlog.**

A Linux TCP listener has two waiting areas with different occupants and different exits.

![Linux listener SYN queue and accept queue anatomy](/imgs/blogs/the-tcp-handshake-and-what-it-costs-you-3.webp)

The SYN queue holds incomplete connection requests, typically represented by compact request-socket state while the server is in `SYN-RECV`. These entries have received SYN and caused SYN-ACK transmission, but they have not completed the final acknowledgment path. They may need SYN-ACK retransmission and eventual expiration.

The accept queue holds fully established child sockets waiting for the application to call `accept()` or `accept4()`. These connections completed the transport handshake. The kernel is ready to give each one to the process, but the process has not drained it yet.

The final ACK is the transition between the two concepts. When the ACK is valid and capacity permits, incomplete request state becomes a connected child socket and joins the accept queue. When the process calls `accept()`, that child leaves the accept queue and becomes an application-owned file descriptor.

This model makes symptoms legible:

| Observation | Likely waiting area | What it means | Source |
| --- | --- | --- | --- |
| Many `SYN-RECV` entries | SYN queue | Handshakes are waiting for final ACK or expiry | Linux socket state; inspect with `ss` |
| Listener `Recv-Q` near `Send-Q` in `ss -lnt` | Accept queue | Completed connections are waiting for application acceptance | Reproducible in the lab; confirm semantics on installed `ss` |
| `TcpExtListenOverflows` increases | Accept queue pressure | A full accept queue prevented normal admission | [Linux SNMP counter documentation](https://kernel.org/doc/html/next/networking/snmp_counter.html) |
| `TcpExtTCPReqQFullDoCookies` increases | SYN request pressure with cookies | Kernel handled full request queue by using cookies | Kernel-version-dependent `nstat` field; inspect locally |
| `TcpExtTCPReqQFullDrop` increases | SYN request pressure with drops | Request queue pressure resulted in drops | Kernel-version-dependent `nstat` field; inspect locally |

The `backlog` argument to `listen(fd, backlog)` is often described as though it names one universal TCP queue. On modern Linux, the `listen(2)` manual defines it as the queue length for completely established sockets waiting to be accepted. `/proc/sys/net/core/somaxconn` silently caps the requested value. The incomplete request limit is governed separately by `net.ipv4.tcp_max_syn_backlog`, subject to implementation details and SYN-cookie behavior.

Read the running machine rather than copying a number from a tuning guide:

```bash
sysctl net.core.somaxconn
sysctl net.ipv4.tcp_max_syn_backlog
sysctl net.ipv4.tcp_syncookies
ss -lnt '( sport = :8080 )'
ss -nt state syn-recv '( sport = :8080 )'
```

The [current Linux IP sysctl documentation](https://www.kernel.org/doc/html/v6.17/networking/ip-sysctl.html) describes `tcp_max_syn_backlog` as a per-listener maximum for remembered `SYN_RECV` requests. It also says a request socket consumes about 304 bytes in the documented kernel version. Treat that byte count as versioned implementation evidence, not an eternal constant. Cloudflare's 2018 write-up reported 256 bytes for `struct inet_request_sock` on its Linux 4.14 environment. Both numbers can be honest because kernel structures change.

### Backlog is a burst buffer, not a throughput fix

A deeper accept queue absorbs a longer scheduling pause before callers see loss or retry. It does not increase the rate at which workers call `accept()` and begin useful work. If completed connections arrive at 10,000 per second while the process drains 8,000 per second, a queue merely delays the overflow.

Using a simple explanatory fluid model, queue growth per second equals arrival rate minus drain rate.

At 10,000 arrivals per second and 8,000 accepts per second, a 4,096-slot queue has approximately `4,096 / 2,000 = 2.048` seconds of capacity from empty if rates remain steady. This is derived arithmetic, not a prediction of a real server. Bursts, multiple listeners, scheduler behavior, connection expiration, kernel admission rules, and application batching alter the trace. The important result is the direction: no finite queue repairs a sustained negative service margin.

A queue also hides pain. Callers may complete TCP but then sit unread, converting a connect failure into a request timeout. Larger queues can increase memory use and widen tail latency. Capacity work should start with why the application is not draining, whether workers are runnable, and whether accepted connections become useful work.

### Measure occupancy and rate together

One `ss` snapshot shows current occupancy. One `nstat` snapshot shows cumulative counters, often since boot or since the tool's saved baseline. Neither alone gives event rate. Sample explicitly:

```bash
nstat -az > /tmp/nstat.before
sleep 10
nstat -az > /tmp/nstat.after
diff -u /tmp/nstat.before /tmp/nstat.after || true

ss -lnt '( sport = :8080 )'
ss -nt state syn-recv '( sport = :8080 )' | wc -l
```

For automation, parse numeric fields and calculate deltas rather than scraping this illustrative `diff`. Label dashboards with the kernel and exporter behavior. A counter reset after reboot or namespace recreation is not a miraculous recovery.

## What overflows where

**Senior rule: a connect timeout is an outcome, not a root cause.**

There are at least four different paths to a similar caller symptom.

First, the SYN may never reach the server. Routing, security policy, a load balancer, or path loss owns the failure. Server TCP counters may remain quiet because the server never saw the packet.

Second, the server may receive SYN but lack room or memory for ordinary request state. Depending on configuration and kernel behavior, it may drop the SYN or use SYN cookies. The caller retransmits because no acceptable SYN-ACK arrives.

Third, the handshake may approach completion while the accept queue is full. Modern Linux can drop packets and increment `TcpExtListenOverflows` and `TcpExtListenDrops`. The caller retries, hoping the application drains before the next attempt.

Fourth, TCP may complete and the application may accept, but later work stalls. That is no longer a handshake failure. Socket and application evidence should move the investigation upward.

Cloudflare's ["SYN packet handling in the wild," published January 15, 2018](https://blog.cloudflare.com/syn-packet-handling-in-the-wild/), is valuable because it joins these layers. The article names the SYN queue and accept queue, uses `ss` to observe `SYN-RECV`, and shows `TcpExtListenOverflows` plus `TcpExtListenDrops` when a slow application stops draining. It also records the exact environment-specific details rather than pretending Linux behavior never changes.

The transferable lesson is not "set backlog to Cloudflare's value." Their article reported `net.core.somaxconn = 16384` on their servers at that time and 119 SYN queue entries on port 80 plus 78 on port 443 in one example. Those are dated observations from Cloudflare's fleet, not defaults or targets. The lesson is to name the queue, observe occupancy, record counter deltas, and relate both to the application's drain rate.

### Trigger, contributing condition, blast-radius multiplier

For the Cloudflare scenario, a slow `accept()` loop is the trigger for accept-queue accumulation. The configured backlog and traffic arrival process are contributing conditions. Retry behavior can be a blast-radius multiplier because clients retransmit while the service is still behind. Increasing the queue can buy time, but it can also retain more completed sockets and defer failure until request deadlines expire.

This framing is useful during any incident:

- **Trigger:** what changed the arrival or drain rate?
- **Contributing condition:** which queue limit, scheduler delay, or path property turned the change into drops?
- **Multiplier:** which retries, connection churn, or synchronized clients increased offered load?
- **Discriminating measurement:** which packet, occupancy value, or counter delta would be different under the competing hypothesis?

Without those distinctions, teams tune the nearest integer and call the temporary quiet a fix.

## SYN cookies trade memory for information

**Senior rule: SYN cookies are overload fallback, not a capacity plan.**

A spoofed SYN flood exploits a painful asymmetry. The attacker can send SYN with a forged source and disappear. A conventional server records request state and retransmits SYN-ACK while waiting for a final ACK that will never arrive. Enough such requests can exhaust the incomplete queue and block legitimate connection attempts.

SYN cookies let the server encode enough connection state into its SYN-ACK sequence number instead of retaining ordinary per-request state. When a valid final ACK returns, the server reconstructs what it needs and continues. A source that cannot receive the SYN-ACK cannot easily produce the right acknowledgment. The mechanism preserves listener availability under a class of SYN-queue pressure.

Linux documentation is deliberately blunt about the trade-off. `tcp_syncookies` should be used as a fallback when the SYN backlog overflows, not as a way to handle legitimate overload. If legitimate traffic routinely fills the queue, cookies do not fix the application's capacity or the arrival pattern.

Inspect rather than assume:

```bash
sysctl net.ipv4.tcp_syncookies
nstat -az | grep -E 'Syncookies|TCPReqQFull|Listen'
```

Useful fields can include `TcpExtSyncookiesSent`, `TcpExtSyncookiesRecv`, `TcpExtSyncookiesFailed`, `TcpExtTCPReqQFullDoCookies`, and `TcpExtTCPReqQFullDrop`. Exact availability depends on kernel lineage. `SyncookiesSent` increasing during a legitimate traffic spike is evidence that incomplete request pressure crossed the fallback threshold. It does not prove an attack. Confirm source distribution, completion rate, packet traces, and edge controls.

### The tuning order

When SYN pressure is legitimate, use this order:

1. Confirm packets reach the intended listener and measure new connections per second.
2. Measure `SYN-RECV` occupancy and residence time.
3. Check retransmission and request-queue counters as deltas.
4. Verify the service's connection reuse and keep-alive behavior.
5. Check whether an upstream L4 device or proxy already terminates TCP.
6. Size the queue for a measured burst and RTT distribution, with memory headroom.
7. Keep SYN-cookie fallback enabled according to platform policy, but alert when it activates.

If the traffic is hostile, queue tuning on the origin is late defense. Filtering, source validation where feasible, anycast distribution, upstream mitigation, and stateless or semi-stateless edge mechanisms can absorb attack traffic before the application listener becomes the scarce resource.

## Which queue dropped the connection?

**Senior rule: use paired counters because one counter rarely has exclusive meaning.**

![Linux TCP listener counters mapped to queues and confirmation commands](/imgs/blogs/the-tcp-handshake-and-what-it-costs-you-4.webp)

`nstat` reads Linux networking counters with names that are much easier to use than positional fields in `/proc/net/netstat`. Use `-a` to include zero values when building a stable parser and `-z` carefully because tool history behavior varies by invocation. For an incident, capture a before snapshot and calculate explicit deltas.

The [Linux kernel SNMP counter documentation](https://kernel.org/doc/html/next/networking/snmp_counter.html) gives the most important relationship. On newer kernels described there, a SYN dropped because the accept queue is full increments both `TcpExtListenOverflows` and `TcpExtListenDrops`. `ListenDrops` is broader and can increase without `ListenOverflows`, for example when allocation fails. A positive `ListenOverflows` delta therefore strongly implicates accept queue pressure in the documented path. A positive `ListenDrops` delta with no `ListenOverflows` increase says the listener dropped packets, but needs another explanation.

This is a diagnostic implication, not a protocol guarantee across every historical kernel. Record `uname -r`, distribution backports, and network namespace. Cloudflare's 2018 article and the current kernel documentation both note that behavior changed across kernel versions.

| Counter delta | Primary interpretation | Confirmation | Source |
| --- | --- | --- | --- |
| `TcpExtListenOverflows > 0` | Accept queue admission failed on documented modern path | `ss -lnt`, application accept rate, capture retries | [Linux SNMP counter docs](https://kernel.org/doc/html/next/networking/snmp_counter.html) |
| `TcpExtListenDrops > 0` | Listener dropped packets for overflow or another reason | Compare `ListenOverflows`; check memory and tracepoints | [Linux SNMP counter docs](https://kernel.org/doc/html/next/networking/snmp_counter.html) |
| `TcpExtTCPReqQFullDoCookies > 0` | Full request queue invoked SYN-cookie handling | Check `tcp_syncookies`, `SYN-RECV`, completion ratio | Linux kernel counter exposed when supported |
| `TcpExtTCPReqQFullDrop > 0` | Full request queue caused drops | Check cookie configuration and ingress shape | Linux kernel counter exposed when supported |
| `TcpExtSyncookiesSent > 0` | Server sent SYN cookies | Correlate with request pressure and traffic legitimacy | Linux TCP extended counter |
| `TcpExtTCPSynRetrans > 0` | SYN or SYN-ACK retransmission occurred | Capture direction and flags | [Linux SNMP counter docs](https://kernel.org/doc/html/next/networking/snmp_counter.html) |
| `TcpExtTCPFastOpenActiveFail > 0` | Active TFO attempt failed or timed out | Compare normal TCP and capture SYN options | [Linux SNMP counter docs](https://kernel.org/doc/html/next/networking/snmp_counter.html) |

The table includes numbers only as delta predicates, not reported measurements. A production dashboard should plot rates and reset-aware increases. For example, alerting on the absolute lifetime value of `ListenOverflows` pages forever after one old incident.

### `ss` tells us current state

Counters say an event happened. `ss` says what exists now:

```bash
# Incomplete handshakes for this service.
ss -Hnt state syn-recv '( sport = :8080 )'

# Listener queue occupancy and configured maximum.
ss -Hlnt '( sport = :8080 )'

# Summary counts across TCP states.
ss -s
```

On a listening TCP socket, `Recv-Q` is the current number of completed connections waiting for acceptance and `Send-Q` is the configured maximum backlog as reported by current iproute2. Validate these semantics against the installed tool if building automation. For an established socket, the same column names mean bytes not copied or acknowledged, so context matters.

A full-looking `Recv-Q` plus increasing `ListenOverflows` and a slow accept rate is a coherent story. Thousands of `SYN-RECV` entries plus cookie counters but an empty accept queue is a different story. An empty server and repeated client SYNs with no server capture is a path story.

### Measure the application drain

Kernel evidence should lead to the owner above it. Track accepted connections per second, accept-loop scheduling latency, worker run-queue delay, file descriptor pressure, and time from accept to first read. An application can drain the accept queue quickly and still stall every connection afterward. Conversely, CPU can look idle while a serialized accept loop or lock prevents drain.

This is where [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) matters. A TCP counter without application context identifies the layer but may not identify the process bottleneck.

## SO_REUSEPORT changes who owns the accept queue

**Senior rule: distribution before `accept()` changes contention and failure isolation.**

![Shared accept queue compared with SO_REUSEPORT per-worker listeners](/imgs/blogs/the-tcp-handshake-and-what-it-costs-you-5.webp)

The traditional multi-worker design creates one listening socket. Workers inherit or share it, block in `accept()`, and compete for completed connections from one accept queue. Modern kernels avoid the worst historical wake-everyone behavior in common paths, but the design still centralizes queue state and can create contention or uneven scheduling depending on the server model.

`SO_REUSEPORT`, available for TCP sockets on Linux since 3.9 according to `socket(7)`, permits multiple sockets with the same address and port when every participant sets the option before `bind()` and the effective-user requirements are satisfied. Each worker can own a distinct listener and accept queue. The kernel selects a listener for an incoming flow before the worker calls `accept()`.

The [Linux `socket(7)` manual](https://www.man7.org/linux/man-pages/man7/socket.7.html) describes the goal as improved accept load distribution compared with one accepting thread distributing work or several threads competing on one socket. Cloudflare's [October 23, 2017 analysis of NGINX worker balancing](https://blog.cloudflare.com/the-sad-state-of-linux-socket-balancing/) shows the benefit and the caveat in a real server: hashing flows across separate queues improved worker distribution in its experiment, but independent queues can worsen latency when one worker stalls while another worker has spare capacity.

That caveat is easy to miss. A single shared queue is work-conserving at the accept boundary: any ready worker can take the next connection. Per-worker queues improve locality and reduce shared contention, but work assigned to a stalled worker does not automatically jump to a healthy worker's queue. The choice moves the failure shape.

| Design | Queue ownership | Strength | Failure mode | Source |
| --- | --- | --- | --- | --- |
| One shared listener | One accept queue shared by workers | Ready workers can drain common work | Shared contention and scheduling imbalance | Linux socket model |
| `SO_REUSEPORT` listener per worker | One accept queue per listener | Pre-accept distribution and locality | One worker's queue can fill while another is idle | [`socket(7)`](https://www.man7.org/linux/man-pages/man7/socket.7.html), Cloudflare 2017 |
| `SO_REUSEPORT` plus BPF selection | Program selects a socket in reuseport group | Policy can reflect CPU or application state | More policy complexity and verification burden | [`socket(7)`](https://www.man7.org/linux/man-pages/man7/socket.7.html) |

Cloudflare's ["Perfect locality and three epic SystemTap scripts," published November 7, 2017](https://blog.cloudflare.com/perfect-locality-and-three-epic-systemtap-scripts/), explains the locality motivation. A listener per worker can align packet processing, accept queue, and worker CPU more closely. Linux also exposes reuseport BPF selection for applications that need more control than default hashing.

### Do not call every accept problem a thundering herd

A thundering herd means many waiters wake for work that only one can consume, wasting scheduling and cache resources. It is not a synonym for any uneven accept distribution. Kernel wakeup behavior, event mechanism, server architecture, and version matter. `EPOLLEXCLUSIVE`, accept mutexes in servers, and kernel wake-one behavior all affect the result.

`SO_REUSEPORT` avoids worker competition on one listening socket by giving workers distinct sockets, but it introduces multiple queues and a selection policy. Phrase the change precisely: it can mitigate shared accept contention and herd-like wakeups in designs that suffer them. It does not guarantee equal application load because connections can have radically different lifetimes and costs.

### Verify distribution, not only throughput

Measure per-worker accepts, queue occupancy, request latency, and CPU. A connection-count hash can be perfectly even while one worker receives expensive or long-lived connections. Cloudflare's 2017 post reported a specific NGINX experiment where the busiest worker used 13.2 percent CPU and the least busy used 9.3 percent with reuseport. Those numbers belong to that dated configuration. They are evidence that the mechanism helped there, not a reusable target.

When the service needs L4 or L7 balancing beyond one host, [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) owns the larger architecture. `SO_REUSEPORT` only decides which local listener receives a flow.

## A handshake incident runbook

**Senior rule: branch on evidence, not on the most familiar sysctl.**

![Decision tree for diagnosing TCP handshake failures](/imgs/blogs/the-tcp-handshake-and-what-it-costs-you-6.webp)

Start from a user-visible symptom such as connect timeout, reset, or a step increase in cold-request latency. Keep one affected destination, source population, time window, and protocol version in scope. Then walk the tree.

### 1. Did the SYN leave the client?

Use a client-side capture, connection tracing, or a narrowly scoped eBPF probe. If no SYN leaves, inspect name resolution result, route selection, local policy, ephemeral-port availability, and application socket errors. Server tuning is irrelevant until traffic reaches the path.

```bash
ip route get 10.77.0.2
ss -s
sudo tcpdump -ni c0 -c 20 'host 10.77.0.2 and tcp port 8080'
```

### 2. Did the SYN reach the listener host?

Capture at the correct interface and network namespace. If the client sees SYN leave but the server sees nothing, investigate routing, ACLs, load balancer selection, NAT state, and asymmetric observation points. A host capture behind an upstream TCP terminator will never see the original handshake.

### 3. Did SYN-ACK leave?

If SYN reaches the server but SYN-ACK does not leave, confirm a listener exists for the destination address and port. Check SYN request pressure, memory failures, policy, and kernel tracepoints. Sample `SYN-RECV` and `TCPReqQFull` or cookie counters.

```bash
ss -Hlnt '( sport = :8080 )'
ss -Hnt state syn-recv '( sport = :8080 )'
nstat -az | grep -E 'Listen|Syncookies|TCPReqQFull|TCPSynRetrans'
```

### 4. Did the final ACK return?

Repeated SYN-ACK without final ACK points toward reverse-path loss, client filtering, spoofed source traffic, or a capture point that cannot see the return. Sequence and acknowledgment numbers distinguish a valid final ACK from unrelated ACK traffic.

### 5. Is the accept queue full?

Inspect listener `Recv-Q` and `Send-Q`, then calculate `ListenOverflows` and `ListenDrops` deltas during the symptom. If occupancy is near the configured maximum and overflows rise, measure why the process is not draining. A larger backlog is only justified after the pause or burst distribution is understood.

### 6. Did Fast Open fall back?

If only TFO-enabled attempts regress, compare `TCPFastOpenActiveFail`, SYN options, data-bearing SYN behavior, and a conventional TCP control. Middleboxes can drop unknown options or SYN payload. RFC 7413 specifies fallback behavior, but fallback still costs time.

### 7. Did TCP complete and the application stall?

Once a child socket is accepted, move up. Measure first read, TLS progress, proxy upstream connect, handler queueing, and deadline propagation. Do not keep tuning SYN queues after evidence leaves the handshake layer.

| Evidence bundle | Best next owner | Avoid this reflex | Source |
| --- | --- | --- | --- |
| Client SYN absent | Client application or host networking | Raising server backlog | Reproducible capture and route inspection |
| Client SYN present, server SYN absent | Network path or upstream device | Blaming `accept()` | Two-sided capture |
| Server SYN present, no SYN-ACK, request counters rise | Listener request path | Enlarging accept queue only | `nstat`, `ss`, capture |
| SYN-ACK repeats, final ACK absent | Client or reverse path | Scaling application workers | Packet timeline |
| `Recv-Q` full and `ListenOverflows` rises | Application accept path | Calling it a SYN flood | Linux SNMP counter docs |
| TCP completes, request begins late | Application, TLS, or proxy | Tuning handshake retry policy | Application timing plus socket state |

The runbook ends with a measurement, not a setting. A useful incident note records the hypothesis that was rejected as carefully as the one confirmed.

## Design choices that actually reduce handshake pain

**Senior rule: fix connection economics before queue limits.**

The strongest option is usually connection reuse. Persistent HTTP connections, client pools, proxy upstream pools, and multiplexed protocols amortize establishment across useful requests. Reuse needs lifetime limits, health checks, load-balancing awareness, and failure handling. An immortal pool can pin traffic to stale endpoints or concentrate load.

The next option is placement. A smaller RTT shortens client latency and normal SYN-request residence time. Edge termination can absorb geographically distributed handshakes near callers, but it adds another connection boundary upstream. The full path may contain client-to-edge TCP and edge-to-origin TCP. Price both and identify which is reused.

Then address application drain. Keep the accept loop cheap. Hand connected sockets to workers without blocking acceptance on expensive initialization. Bound downstream work so accepting faster does not merely move unbounded queues into user space. [Rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) covers that system-wide boundary.

Tune backlog only from measured burst size, pause duration, RTT distribution, and memory budget. Monitor both occupancy and overflow. Treat SYN cookies as a protective fallback with an alert. Consider `SO_REUSEPORT` when shared-listener contention or locality is proven, then monitor each worker queue and tail latency.

Use TFO only when repeated connections, path support, and replay-safe operations line up. Test fallback explicitly. A feature that wins one RTT in the success path but triggers retransmission on a meaningful path segment may lose overall.

### A decision table

| Symptom | Primary move | Why | New risk | Source |
| --- | --- | --- | --- | --- |
| High cold latency, low connection rate | Reuse or place termination closer | Removes or shrinks setup RTT | Stale pooled connections, more edge state | Derived from latency ladder |
| Accept queue bursts during short scheduler pauses | Size backlog from measured burst | Buffers transient mismatch | Hides latency and uses more state | Derived queue model |
| Sustained accept queue growth | Increase drain rate or shed load | Finite queue cannot repair rate deficit | Earlier rejection needs client policy | Derived queue model |
| SYN request pressure from spoofed sources | Upstream mitigation plus cookie fallback | Avoids retaining ordinary state for every SYN | Reduced negotiation information, attack still consumes processing | Linux docs and RFC 4987 context |
| Shared-listener contention | Evaluate `SO_REUSEPORT` | Moves selection before per-worker accept | Per-worker queue imbalance | `socket(7)`, Cloudflare 2017 |
| Repeated short, replay-safe transactions | Evaluate TFO | May move data into handshake | Replay and middlebox fallback | RFC 7413 |

Every row moves cost rather than deleting it. Connection reuse retains state longer. Edge termination adds infrastructure and another trust boundary. Deeper queues exchange immediate drops for waiting and memory. Reuseport exchanges shared contention for partitioned queues. TFO exchanges a dependency for replay analysis.

## Work the numbers before changing the knob

**Senior rule: size a queue from a traffic envelope, then validate it under the same envelope.**

The most tempting listener calculation is also the least useful one: take peak connections per second, multiply by RTT, and paste the result into every backlog setting. That product is a starting estimate for incomplete handshakes under clean, steady traffic. It is not the required accept backlog, and it is not enough for bursty traffic.

Consider a service that receives 12,000 new connections per second. Its clients have a measured p50 RTT of 20 ms, p95 of 90 ms, and p99 of 180 ms. If every handshake completes cleanly and arrivals are steady, an explanatory estimate at each RTT is:

| RTT point | Arrival rate | Estimated incomplete requests | What the estimate omits | Source |
| --- | ---: | ---: | --- | --- |
| p50, 20 ms | 12,000/s | 240 | Bursts, retransmits, spoofed SYNs | Derived here: rate multiplied by 0.020 s |
| p95, 90 ms | 12,000/s | 1,080 | Correlation between client geography and arrival rate | Derived here: rate multiplied by 0.090 s |
| p99, 180 ms | 12,000/s | 2,160 | Tail beyond p99 and server scheduling | Derived here: rate multiplied by 0.180 s |

These rows describe the SYN-side residence of clean handshakes. They do not tell us how many completed children wait for `accept()`. Accept-queue demand depends on a different time: how long completed sockets wait before the application drains them.

Suppose the same listener experiences a 100 ms runtime pause while 12,000 connections per second complete. A first-order accept-queue burst estimate is `12,000/s x 0.100 s = 1,200` connections. If the pause is 500 ms, the estimate becomes 6,000. If arrivals double during a synchronized retry wave, both numbers double. This arithmetic explains why a queue that is comfortable during normal scheduling can overflow during a stop-the-world pause or worker rollout.

The estimates also show why `tcp_max_syn_backlog` and `listen(backlog)` solve different residence problems. Long RTT inflates time in incomplete state. Application pauses inflate time in completed-but-unaccepted state. One workload can pressure both, but the evidence and remediation differ.

### Add burst shape, not a magical safety factor

A generic "multiply by four" rule conceals the variable that matters. Record connections per second in intervals shorter than the queue drain event. A one-minute rate can look safe while a 200 ms burst fills a listener. Histograms or high-resolution counters should retain the burst shape relevant to the queue.

For the accept queue, a useful explanatory bound is the maximum, over a short interval, of completed arrivals minus accepts during that interval.

This estimate is not a Linux admission formula. It tells us what to measure: completed arrivals and accepts over aligned short intervals. If the maximum observed surplus is 900 during ordinary deploy pauses, a configured effective maximum of 128 is predictably fragile. If the surplus grows without bound for seconds, increasing from 128 to 4,096 only changes when failure begins.

Queue memory is only part of the cost. A completed socket can own receive buffers, protocol state, accounting, and application-facing deadlines. Clients that believe they connected may immediately send data, so a full or slow accept path can accumulate bytes and timeouts. Model memory from measurements on the actual kernel and workload instead of multiplying a historical request-socket size by every queue.

### Retry timing changes the offered load

When admission drops a SYN or final ACK, clients do not necessarily disappear. TCP retransmission produces another attempt after a timer. Application libraries may also abandon the connection and create a new one. Load balancers may retry a failed upstream on another endpoint. These layers can turn a short accept pause into a longer arrival surge.

Separate transport retransmission from application retry in telemetry. `TcpExtTCPSynRetrans` counts SYN and SYN-ACK retransmissions in kernels documented by the Linux SNMP guide. A new source port or new socket attempt may instead be an application retry. A packet capture can distinguish retransmission of the same flow from a new flow, while traces can identify a higher-level attempt.

The safe control is a coordinated deadline and backoff policy, not an arbitrarily large listener. The [timeouts, retries, and backoff guide](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) covers the application policy. At this layer, our job is to expose whether transport admission caused the retry and how much extra offered load it created.

## The socket API changes what latency the application sees

**Senior rule: kernel state transition and application wakeup are separate timestamps.**

A blocking client `connect()` returns when the local kernel considers the connection established or reports an error. That is usually near receipt of SYN-ACK on the clean active-open path. It does not wait for the remote process to call `accept()`. A client can therefore report successful connect while the server's application is paused and the child socket waits in the accept queue.

On the server, a blocking `accept()` returns a connected child. With nonblocking sockets, `accept4()` can return `EAGAIN` after the readiness indication has already been consumed by another worker or after an asynchronous network error changes the queue. Linux also passes certain pending network errors on the new socket in ways portable applications need to handle, as described by the platform `accept(2)` manual.

This separation creates four useful timestamps:

1. Client sends SYN.
2. Client `connect()` completes.
3. Server `accept()` returns the child.
4. Server reads the first application byte.

The gaps own different mechanisms. Timestamp 1 to 2 is primarily handshake and path. Timestamp 2 to 3 can include accept-queue waiting and server scheduling, but the clocks are on different hosts unless instrumentation normalizes them. Timestamp 3 to 4 can expose application dispatch, event-loop delay, TLS handling, or a client that connected but sent nothing.

For a controlled lab on one host, monotonic clocks in both namespaces share the host clock and can be compared cautiously. Across production hosts, use packet capture at one observation point or distributed tracing with known clock-error bounds. Do not subtract unsynchronized wall clocks and call the result queue time.

### A minimal listener exposes the control points

This Python example is intentionally small enough to audit. It requests a backlog, accepts continuously, and reports the delay from process start to each accepted child. It does not claim the requested backlog is the effective one because `somaxconn` can cap it.

```python
import argparse
import socket
import time


def serve(host: str, port: int, backlog: int) -> None:
    started = time.monotonic_ns()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(backlog)
        print({"event": "listening", "requested_backlog": backlog})
        while True:
            conn, peer = listener.accept()
            accepted = time.monotonic_ns()
            print(
                {
                    "event": "accepted",
                    "peer": peer,
                    "process_age_ms": (accepted - started) / 1_000_000,
                }
            )
            conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="10.77.0.2")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backlog", type=int, default=128)
    args = parser.parse_args()
    serve(args.host, args.port, args.backlog)
```

Production code needs signal handling, structured metrics, error policy, file descriptor limits, connection ownership, and usually an event loop or worker system. The point is to locate controls. `listen()` requests completed-queue capacity. `accept()` drains it. Work done before the next `accept()` reduces drain rate. Moving expensive work after dispatch can keep admission flowing, but only if downstream work is bounded.

### Nonblocking accept is not automatically faster

Nonblocking I/O prevents a thread from sleeping on one socket, but it does not create CPU or remove synchronization. An event loop that spends 200 ms handling one callback can neglect a ready listener. Multiple workers around one readiness source can contend. A busy loop on `EAGAIN` can burn CPU that should drain actual work.

Measure event-loop lag, accept batch size, and time between readiness and accept. If one worker deliberately accepts several sockets per wakeup, check fairness with existing connections. If workers use `SO_REUSEPORT`, expose each listener's queue rather than summing away the imbalance.

The practical boundary is simple. Kernel tuning controls how much transient mismatch is buffered. Application architecture controls how quickly the mismatch is removed.

## Run it yourself

### Question

When a Linux application's accept queue stops draining, do listener queue occupancy and `TcpExtListenOverflows` or `TcpExtListenDrops` distinguish that condition from a healthy accept loop?

### Preconditions

Use the Linux-only `netlab` topology from [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url): namespaces `c` and `s`, interfaces `c0` and `s0`, client `10.77.0.1`, server `10.77.0.2`, and TCP port `8080`. Run inside a disposable Linux VM if your host is macOS. The experiment needs root or `CAP_NET_ADMIN` for namespaces and `tc`, plus Python 3, iproute2 (`ip`, `ss`, `nstat`), and `tcpdump` if you capture packets.

Do not run the mutating commands on an unspecified production interface. Packet captures can contain sensitive data. The commands below scope all mutations to `c0`, namespace `s`, port `8080`, and `/tmp/tcp-handshake-lab`.

Preflight:

```bash
set -euo pipefail
ip netns list | grep -E '^(c|s)( |$)'
ip -n c addr show dev c0
ip -n s addr show dev s0
ip -n c route get 10.77.0.2
ip netns exec c ping -c 5 -q 10.77.0.2
ip netns exec s python3 --version
ip netns exec s nstat -az | grep -E 'ListenOverflows|ListenDrops'
```

Read: confirm both namespaces and interfaces exist, the route uses `c0`, ping succeeds, and note the measured RTT. Without added delay, a same-host namespace RTT is often below 2 ms, but CPU load and virtualization vary. Record the actual range instead of assuming it.

### Baseline

Create a server that calls `accept()` continuously with backlog 1. It closes each connection immediately so application work does not become the bottleneck.

```bash
set -euo pipefail
LAB=/tmp/tcp-handshake-lab
mkdir -p "$LAB"

ip netns exec s sh -c 'cat > /tmp/tcp-handshake-lab/server.py <<"PY"
import socket

HOST = "10.77.0.2"
PORT = 8080

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, PORT))
    listener.listen(1)
    while True:
        conn, _ = listener.accept()
        conn.close()
PY'

ip netns exec s sh -c \
  'python3 /tmp/tcp-handshake-lab/server.py > /tmp/tcp-handshake-lab/server.log 2>&1 & echo $! > /tmp/tcp-handshake-lab/server.pid'
sleep 1

ip netns exec s nstat -az > "$LAB/nstat.baseline.before"
ip netns exec c python3 - <<'PY'
import socket

ok = 0
for _ in range(50):
    with socket.create_connection(("10.77.0.2", 8080), timeout=1):
        ok += 1
print({"attempts": 50, "connected": ok})
PY
ip netns exec s nstat -az > "$LAB/nstat.baseline.after"
ip netns exec s ss -Hlnt '( sport = :8080 )'
```

Read: the Python client field `connected`, listener `Recv-Q`, and the before/after values of `TcpExtListenOverflows` and `TcpExtListenDrops`.

Expected: 50 of 50 connections should normally complete on an otherwise idle lab. Listener `Recv-Q` should return to 0 or briefly remain near 0. Both overflow-counter deltas should normally be 0. A heavily loaded VM can produce a different result, which is why we retain both snapshots.

Compare the exact fields:

```bash
for key in TcpExtListenOverflows TcpExtListenDrops; do
  before=$(awk -v k="$key" '$1 == k {print $2}' /tmp/tcp-handshake-lab/nstat.baseline.before)
  after=$(awk -v k="$key" '$1 == k {print $2}' /tmp/tcp-handshake-lab/nstat.baseline.after)
  printf '%s delta=%d\n' "$key" "$((after - before))"
done
```

### Apply one change

Replace the server with one that still uses backlog 1 but waits 15 seconds before beginning to accept. The controlled change is application drain behavior. Queue configuration stays the same.

```bash
set -euo pipefail
LAB=/tmp/tcp-handshake-lab
ip netns exec s sh -c 'kill "$(cat /tmp/tcp-handshake-lab/server.pid)" 2>/dev/null || true'

ip netns exec s sh -c 'cat > /tmp/tcp-handshake-lab/server.py <<"PY"
import socket
import time

HOST = "10.77.0.2"
PORT = 8080

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, PORT))
    listener.listen(1)
    time.sleep(15)
    while True:
        conn, _ = listener.accept()
        conn.close()
PY'

ip netns exec s sh -c \
  'python3 /tmp/tcp-handshake-lab/server.py > /tmp/tcp-handshake-lab/server.log 2>&1 & echo $! > /tmp/tcp-handshake-lab/server.pid'
sleep 1
ip netns exec s nstat -az > "$LAB/nstat.treatment.before"
```

### Compare

Start 50 connection attempts with short client timeouts while the server is intentionally not accepting. Sample queue occupancy during the pause, then read counter deltas.

```bash
set -euo pipefail
LAB=/tmp/tcp-handshake-lab

ip netns exec c python3 - <<'PY' &
import socket

ok = 0
failed = 0
for _ in range(50):
    try:
        with socket.create_connection(("10.77.0.2", 8080), timeout=0.25):
            ok += 1
    except OSError:
        failed += 1
print({"attempts": 50, "connected": ok, "failed_or_timed_out": failed})
PY
client_pid=$!

for _ in 1 2 3 4 5; do
  ip netns exec s ss -Hlnt '( sport = :8080 )'
  sleep 0.2
done
wait "$client_pid"

ip netns exec s nstat -az > "$LAB/nstat.treatment.after"
for key in TcpExtListenOverflows TcpExtListenDrops; do
  before=$(awk -v k="$key" '$1 == k {print $2}' "$LAB/nstat.treatment.before")
  after=$(awk -v k="$key" '$1 == k {print $2}' "$LAB/nstat.treatment.after")
  printf '%s delta=%d\n' "$key" "$((after - before))"
done
```

Read: listener `Recv-Q` versus `Send-Q`, client `connected` versus `failed_or_timed_out`, and both counter deltas. The queue should reach its small limit while the process sleeps. On a modern Linux kernel following the behavior documented by the kernel SNMP guide, `ListenOverflows` and `ListenDrops` should both increase as additional attempts and retries arrive.

Expected: at least one or two connections may complete because Linux queue accounting and the `backlog + 1` behavior described by the kernel test can admit a small number, but many of 50 attempts should time out during the 15-second pause. Both deltas should be greater than zero on the documented modern path. Exact counts vary with kernel version, retransmission timing, the 250 ms client timeout, scheduler timing, and namespace implementation. The qualitative result is the invariant: healthy drain keeps occupancy and deltas low; stopped drain fills the accept queue and produces admission evidence.

For a read-only production translation, run only:

```bash
uname -r
ss -Hlnt '( sport = :8080 )'
ss -Hnt state syn-recv '( sport = :8080 )'
nstat -az | grep -E 'ListenOverflows|ListenDrops|TCPReqQFull|Syncookies|TCPSynRetrans'
```

Do not change sysctls during an incident until queue identity and application drain are proven.

### Reset

Remove only the process and files created by this experiment. The base namespaces remain for later posts.

```bash
set -euo pipefail
ip netns exec s sh -c \
  'test ! -f /tmp/tcp-handshake-lab/server.pid || kill "$(cat /tmp/tcp-handshake-lab/server.pid)" 2>/dev/null || true'
ip netns exec s rm -rf /tmp/tcp-handshake-lab
rm -rf /tmp/tcp-handshake-lab
```

The experiment proves the main diagnostic claim without generating a SYN flood or changing a global sysctl. The queue fills because the application stops draining. `ss` shows current occupancy, and `nstat` records the kernel's admission failures.

## Production review checklist

Before changing TCP listener settings, write down these answers:

- What is the measured RTT distribution for new connections?
- What fraction of requests use a new connection rather than a pooled one?
- Where does TCP terminate: client host, edge, L4 load balancer, sidecar, or application?
- How many new connections arrive per second at median and burst percentiles?
- How many sockets are in `SYN-RECV` during the symptom?
- What are listener `Recv-Q` and `Send-Q` for the affected socket?
- Which `nstat` counters increased during the same interval?
- How quickly does the application call `accept()` and begin its first read?
- Is traffic legitimate, retry-amplified, or spoofed?
- Does `SO_REUSEPORT` create per-worker queues that need per-worker visibility?
- If TFO is enabled, are early operations replay-safe and are fallback counters monitored?

If these answers are unavailable, the safest change is usually better measurement. A bigger queue can turn a crisp refusal into a slow timeout. A cookie can preserve SYN admission while legitimate capacity remains insufficient. More workers can deepen downstream overload.

## Key takeaways

- A conventional TCP `connect()` costs approximately one clean RTT because SYN-ACK depends on SYN. Three packets do not mean three RTTs.
- The first ordinary request reaches the server later than `connect()` completion. Define the measurement boundary before comparing latency.
- Reusing an established connection removes TCP setup from the next request. TFO can move replay-safe data into a later handshake, but it changes application semantics and has path fallback.
- Linux exposes an incomplete SYN queue and a completed accept queue. `backlog` without a queue name is an incomplete diagnosis.
- `TcpExtListenOverflows` implicates accept-queue pressure on the documented modern path. `TcpExtListenDrops` is broader. Pair both with `ss`, packet capture, and application drain rate.
- SYN cookies protect availability under request-queue pressure by avoiding ordinary retained state. They are fallback, not a substitute for legitimate capacity.
- `SO_REUSEPORT` replaces one shared listener queue with per-listener queues selected before `accept()`. It can improve locality and reduce contention while creating per-worker imbalance.
- Tune from measured arrival rate, RTT, burst duration, occupancy, and drain rate. Never copy a backlog value from another fleet.

The broader [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) is the same move repeated at every layer: put the symptom on the path, name the state that can retain it, and choose the counter or packet that rejects the competing explanation. The next queue is not automatically the right queue.

## Further reading

- [RFC 9293: Transmission Control Protocol](https://www.rfc-editor.org/rfc/rfc9293.html), August 2022.
- [RFC 7413: TCP Fast Open](https://www.rfc-editor.org/rfc/rfc7413.html), December 2014.
- [Linux IP sysctl documentation](https://www.kernel.org/doc/html/v6.17/networking/ip-sysctl.html), including `tcp_max_syn_backlog`, `tcp_syncookies`, and related controls.
- [Linux networking SNMP counter documentation](https://kernel.org/doc/html/next/networking/snmp_counter.html), including listener overflow examples and TCP extended counters.
- [Cloudflare: SYN packet handling in the wild](https://blog.cloudflare.com/syn-packet-handling-in-the-wild/), January 15, 2018.
- [Cloudflare: Why does one NGINX worker take all the load?](https://blog.cloudflare.com/the-sad-state-of-linux-socket-balancing/), October 23, 2017.
- [Cloudflare: Perfect locality and three epic SystemTap scripts](https://blog.cloudflare.com/perfect-locality-and-three-epic-systemtap-scripts/), November 7, 2017.
- [Flow control versus congestion control](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe), for the two limits that govern an established connection after the handshake.
