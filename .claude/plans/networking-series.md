# Networking for Engineers Who Ship Services

**Subcategory:** `networking` (NEW folder, renders "Networking")
**Path:** `content/blog/software-development/networking/<slug>.md`
**frontmatter `subcategory`:** `"Networking"` · **`category`:** `"software-development"`
**Size:** **48 posts, 8 tracks (A to H), 8 waves of 6.** **STATUS: APPROVED, Wave 0 complete; Wave 1 next.**
**Language:** English (repo convention, verify gate enforces it). Request came in VN: figures must be *trực quan*, every post must be *dễ hiểu* and *áp dụng cao*.

### Folder-name note (checked, do not re-litigate)
`src/lib/postPath.ts::derivePostLocation`: the folder wins, frontmatter `subcategory` is only a fallback. Display name comes from `formatSubcategoryName` in `src/app/blog/software-development/page.tsx:19`, which title-cases word by word against an `ACRONYMS` allow-list. `networking` renders **"Networking"**. Nothing to register anywhere: subcategories are derived from the folder tree.

---

## Angle: this series owns the wire

The repo has 43 posts on system design, 40 on microservices, 40 on SRE, 40 on API design. Every one of them *assumes* the network and none of them *explains* it. There are exactly zero posts on TCP, DNS, TLS handshakes, BGP, congestion control, or packet capture. That is the gap: engineers who can draw a service diagram but cannot say why the same call takes 2 ms in staging and 240 ms in production.

This is not a CCNA course. It is networking for the person who writes and operates services, organised around one claim:

> **The network is not transparent, and every abstraction above it leaks in a specific, predictable, measurable way.** A senior is someone who knows which layer a symptom comes from before opening a single dashboard.

Three things in every post, without exception:

1. **Intuition first and heavily visual.** Where the milliseconds and the bytes actually go, drawn, before any protocol detail. Four recurring figures (below) carry the whole series so the reader builds one mental picture across 48 posts instead of 48 unrelated ones.
2. **At least one real, dated, publicly documented case.** A named incident postmortem, a vendor engineering write-up, a measurement paper, or an RFC-era story, with a link. This is the user's headline requirement and it is a hard review gate. The case bank below is pre-verified: use it, extend it, never invent one.
3. **Something you can run today.** Every post ends with a **Run it yourself** block: exact commands against `netlab` or against the reader's own production, the output to look at, and the range to expect. "Highly applicable" means the reader can reproduce the claim on a laptop within five minutes.

### `netlab`: the recurring lab (this series' `nanosearch` / `nanoserve`)

One small, copy-pasteable Linux lab that grows across the series: two `ip netns` namespaces joined by a veth pair, `tc netem` for RTT / loss / reorder, `tbf` for bandwidth, and a tiny Go echo+HTTP server and client. Built in post #1, extended by later posts (TLS terminator in Track D, a proxy hop in Track F, a 3-node fabric in Track G).

- Tooling fixed once, used everywhere: `curl -w`, `ss -tin`, `tcpdump` / `tshark`, `dig +trace`, `mtr`, `iperf3`, `nghttp` / `h2load`, `bpftrace` (`tcpretrans`, `tcpconnlat`), `ip route get`, `conntrack -S`, `nstat` / `netstat -s`.
- **macOS note (the author's box):** `netem` is Linux only. `netlab` runs in Lima / Colima or a `--privileged --cap-add NET_ADMIN` container. Say so in post #1 once and never repeat it.
- The lab is what makes the numbers honest: most claims in this series are class (c) below, not class (b).

### Honesty rule (hard gate, inherited from `inference-engineering`)

No post may claim "I measured X in production and got Y" with a fabricated number. Every number is one of:

- **(a) Derived** from a stated formula, with the arithmetic shown: BDP, serialization delay, retry fan-out, ephemeral-port ceiling, Little's Law pool size, cross-AZ dollars per month.
- **(b) Cited** to a public postmortem, paper, vendor benchmark or RFC, **with a link and a date**, and with the version or region it applies to.
- **(c) Reproducible** in `netlab` by the reader, with the command and an expected range ("at 80 ms RTT and 1% loss, cubic should land near 1.5 to 3 Mbit/s on this lab, bbr near 8 to 12").

Tables that carry numbers carry a `Source` column. **Never invent a company, an incident, a customer, or a benchmark.** The house first-person voice ("I have watched a health check yank a whole pool") stays, but the moment a name, a date or a number appears it must be sourceable.

Cloud prices move: any dollar figure names the region and the date it was checked.

---

## Gates (same tier as `inference-engineering` / `performance-engineering`)

- **>= 6k words** floor, target 9k to 11k · **>= 5 figures** floor, target **7** · **>= 4 distinct kinds** · **>= 1 animated figure** where motion carries meaning.
- Verify: `bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> deep-dive`
- **No em dash anywhere** (house rule since 2026-08-11): no `—`, no spaced ` – `, no ` -- `. Unspaced `–` in a numeric range is fine.
- **LaTeX traps:** brace-wrap any inline math starting with a digit (`${10}^{6}$`), and write `\lt ` instead of a bare `<` followed by a letter. Both are known silent-corruption bugs in this repo.
- Kit: `.cache/blog-writer/_networking-series-kit.md` (BUILD in the infra step, READ FULLY before writing any post).
- Render helper: `bash .cache/blog-writer/_render-net.sh <slug>` (DSL / in.json to scene to PNG to lossless webp to `public/imgs/blogs/`).
- **Renderer reminder:** `render-scene-batch.mjs` lives **outside this repo** at `/Users/hieptran1812/Documents/mcp_excalidraw/scripts/`. The sharpness gate passes vacuously with 0 webp on disk, so **always count files**, never trust a green gate.
- Check the post does not already exist on disk before running the figure pipeline: a same-slug render clobbers committed webps.

**Animation candidates (motion carries the idea):** the sliding window advancing and stalling on a zero window · cwnd sawtooth under loss versus BBR's probe cycle · a retransmit timer firing · HOL blocking in HTTP/1.1 versus /2 versus /3 · a BGP withdrawal propagating hop by hop across ASes · recursive DNS walking root to TLD to authoritative · a queue filling and draining under bufferbloat, with latency rising · retry amplification fanning out 3 levels deep · ECMP hashing flows onto links with one elephant flow pinning a link.

---

## The recurring visual language (define in post #1, reuse in all 47)

1. **The latency ladder.** One horizontal bar of a single request's wall clock, split into DNS, TCP, TLS, request, server think, first byte, transfer. Every post opens with it and lights up the segment it owns. By the capstone the reader has seen the same bar 48 times with different segments dominating.
2. **The path map.** `client → resolver → edge / anycast → L4 LB → L7 proxy → mesh sidecar → app → backend`, with the current hop lit and the rest greyed. Answers "where am I standing" in one glance.
3. **The packet timeline.** A fixed two-column client / server sequence diagram, time downward, milliseconds on the left rail. Used for every handshake, every loss-recovery story, every HOL example, so the reader reads them all with the same eyes.
4. **The throughput / latency frontier.** One fixed pair of axes (goodput versus RTT, or p99 versus offered load). Every tuning knob in the series lands on it as a point or a curve. By Track H it is a populated map.

---

## Boundary table: what this series does NOT re-explain

| Neighbour | It owns | We own |
| --- | --- | --- |
| `api-design/` | HTTP semantics, status codes, auth schemes, contract and versioning design | HTTP framing, connection reuse, HOL blocking, the bytes and the milliseconds |
| `system-design/` (`load-balancing-from-l4-to-l7`) | the architect's choice of LB and algorithm | what L4 and L7 physically cannot see, DSR, client-IP preservation, health-check timing math |
| `microservices/` | mesh as an architecture decision, discovery as a pattern | the extra hops, the sidecar's measured CPU and latency tax, xDS on the wire |
| `site-reliability-engineering/` | incident process, SLOs, on-call | the diagnostic ladder for a network symptom, packet capture, partition semantics |
| `debugging/` (`its-the-network-packet-and-protocol-tracing`) | the general debugging method | protocol-by-protocol reading of a capture |
| `distributed-training/` | collectives, parallelism strategy | RDMA / RoCEv2, PFC, incast, fabric topology |
| `database-scaling/` | replication and sharding strategy | what a 43-second partition does to a failover |

Cross-link into these rather than restating them. Two or three sibling links per post.

**Cross-link spine (every post links both):**
- **Intro:** `what-actually-happens-when-you-curl-a-url` (A #1)
- **Capstone:** `the-senior-engineers-network-mental-model` (H #48)

---

## Track A: the mental model and the fundamentals (Wave 1)

1. `what-actually-happens-when-you-curl-a-url`: **INTRO.** One request, every hop and every millisecond named, from typing the command to the last byte. Introduces the latency ladder, the path map, the packet timeline, the frontier. Sets up `netlab` and the honesty rule. The map of all 48 posts.
2. `the-layers-are-a-lie-but-a-useful-one`: OSI versus the stack that actually runs (link, IP, TCP/UDP, TLS, HTTP), encapsulation and header overhead computed on a real packet, what each layer can and cannot know about the one above, and why "it's a network problem" is nearly always a layering mismatch. MTU as the first leak.
3. `addresses-subnets-nat-and-the-packets-two-identities`: IPv4 and CIDR arithmetic you can do in your head, private ranges, IPv6 in its real state of deployment, NAT and its state table, and the first preview of the two limits NAT imposes (conntrack entries, ephemeral ports).
4. `how-a-packet-gets-there-arp-switching-routing-and-ecmp`: L2 versus L3 in one figure, MAC and ARP, switches and VLANs, the default gateway, longest-prefix match, `ip route get`, and ECMP hashing with the rule that surprises people most: a single flow never splits across links.
5. `sockets-and-the-two-transport-contracts-tcp-vs-udp`: the socket API as the boundary between your code and the kernel, stream versus datagram semantics, blocking versus non-blocking, what UDP forces you to rebuild (ordering, reliability, congestion control, MTU discovery) and when that trade is correct. `ss` output read line by line.
6. `the-latency-budget-speed-of-light-serialization-and-queueing`: the four components of delay derived from first principles, the latency numbers table rebuilt rather than memorised, bandwidth-delay product, and the senior decision it drives: when a closer region beats a fatter pipe. Worked: Hanoi to Frankfurt, 10 KB versus 10 MB.

## Track B: TCP, properly (Wave 2)

7. `the-tcp-handshake-and-what-it-costs-you`: SYN, SYN-ACK, ACK priced in RTT, TCP Fast Open, the SYN queue versus the accept queue, `backlog` and what overflows where, SYN cookies, `SO_REUSEPORT` and the accept thundering herd. Counters in `nstat` that prove which queue dropped. **Case:** Cloudflare's SYN-handling write-ups.
8. `reliability-sequence-numbers-acks-retransmits-and-rto`: cumulative ACK, SACK, duplicate ACKs and fast retransmit, RTO estimation from RTT samples, tail loss probe, RACK, and why a single lost packet at the end of a response costs an entire RTO. **Animated:** loss and recovery on the packet timeline.
9. `flow-control-vs-congestion-control-two-windows-one-pipe`: rwnd versus cwnd as two independent limits, zero-window and window probes, Linux receive-buffer autotuning, and the BDP arithmetic that explains the classic complaint. **Worked:** why a 10 Gbit link delivers 40 Mbit/s over a 150 ms RTT, and exactly which sysctl fixes it.
10. `congestion-control-cubic-bbr-and-what-changing-it-actually-does`: loss-based versus model-based control, the sawtooth, BBR's bottleneck bandwidth and RTT model, the fairness fight when they share a link, and how to switch algorithms per route. **Lab:** the same file over 80 ms / 1% loss under cubic and bbr, side by side.
11. `bufferbloat-queueing-and-the-latency-you-added-yourself`: why deep buffers destroy interactivity, queueing delay derived from utilisation, CoDel and FQ-CoDel, `tc qdisc` in practice, and how to spot bloat from a `ping` during a large upload. **Case:** Gettys' 2010 to 2011 bufferbloat discovery, CoDel (Nichols and Jacobson 2012), RFC 8290.
12. `nagle-delayed-acks-and-the-40ms-mystery`: the interaction that has cost more engineer-hours than any other single line of TCP, the write-write-read anti-pattern, `TCP_NODELAY` and `TCP_CORK`, and how it surfaces in an RPC layer as a bimodal latency histogram at exactly 40 ms.

## Track C: names, discovery, and the control plane (Wave 3)

13. `dns-the-distributed-database-your-request-starts-in`: recursive versus iterative resolution, the root / TLD / authoritative walk, record types that matter, TTL, negative caching, EDNS0 and the 512-byte legacy, TCP fallback. **Animated:** the resolution walk.
14. `dns-in-production-ttls-caching-layers-and-stale-answers`: the five caches between your process and authoritative (app, JVM `networkaddress.cache.ttl`, libc / nscd, resolver, forwarder), CNAME chains, DNS-based failover and its honest recovery time, split-horizon. **Case:** Salesforce 2021-05-11 (DNS change via emergency break-fix, global disruption), Akamai Edge DNS 2021-07-22.
15. `kubernetes-dns-ndots-search-domains-and-the-5-second-timeout`: why one lookup becomes five, `ndots:5`, the conntrack DNAT race in the kernel that produces exactly 5-second hangs, NodeLocal DNSCache, and the three fixes ranked by cost. **Case:** the Weave and Xing write-ups on racy conntrack and `ndots`.
16. `service-discovery-from-dns-to-registries-to-xds`: static config to DNS to Consul / etcd to EDS and xDS, active versus passive health checks, and the property that decides everything: staleness versus correctness during a partition. **Case:** Roblox's 73-hour 2021-10-28 outage (Consul streaming plus BoltDB).
17. `bgp-how-the-internet-decides-where-your-packet-goes`: ASes, transit versus peering, path attributes and route selection, propagation and withdrawal, anycast, and why your traffic takes a route no one chose deliberately. **Animated:** a withdrawal propagating.
18. `bgp-incidents-hijacks-leaks-and-the-day-facebook-withdrew-itself`: **case-heavy post.** Meta 2021-10-04 (backbone audit command, DNS self-withdrawal, out-of-band access lost), Rogers 2022-07-08 (filter removal, nationwide, 911 down), KT Korea 2021-10-25, Pakistan Telecom versus YouTube 2008-02-24, the 2018 Route 53 / MyEtherWallet hijack. RPKI, ROA, MANRS, and what an application engineer can actually do about any of it.

## Track D: trust on the wire (Wave 4)

19. `tls-what-the-handshake-buys-and-what-it-costs`: 1.2 versus 1.3 priced in round trips, key exchange and forward secrecy in plain language, session resumption, 0-RTT and its replay caveat, and the handshake's exact place on the latency ladder. **Lab:** `curl -w` splitting connect from appconnect at 5 ms and at 150 ms RTT.
20. `certificates-chains-and-the-trust-you-inherit`: X.509, chain building and the intermediates you forgot to serve, SNI, SAN, expiry monitoring that actually fires, OCSP versus CRL versus stapling, Certificate Transparency. **Case:** the DST Root CA X3 expiry on 2021-09-30 and exactly which clients it broke.
21. `mtls-and-service-identity-at-scale`: client certificates, SPIFFE and SVIDs, why short-lived certificates are the only revocation that works, rotation without downtime, and the boundary with `microservices/service-to-service-security-mtls-and-zero-trust`.
22. `terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end`: where to terminate and what each choice gives up, the CPU economics of TLS at line rate, kernel TLS and hardware offload, and the honest version of the "it's encrypted inside the VPC" argument. **Case:** Netflix Open Connect serving TLS at 400 Gb/s with kTLS (Gallatin, EuroBSDcon 2021).
23. `protocol-ossification-middleboxes-and-why-quic-looks-like-that`: the senior mental model post of this track. Why TLS 1.3 pretends to be 1.2 on the wire, what GREASE is for, how middleboxes froze TCP in place, and why the only way to ship a new transport was to encrypt the headers and ride UDP.
24. `network-attacks-you-will-actually-meet`: volumetric versus protocol versus application layer, SYN floods and cookies, reflection and amplification factors computed, slow-loris, and where rate limiting must sit to work. **Case:** GitHub's 1.35 Tbps memcached amplification 2018-02-28, Dyn / Mirai 2016-10-21, HTTP/2 Rapid Reset CVE-2023-44487 (2023-10-10). Defensive framing throughout.

## Track E: the application protocols (Wave 5)

25. `http-1-1-keep-alive-and-the-six-connection-tax`: connection reuse and what a cold connection costs, the per-origin connection limit and the sharding hacks it spawned, why pipelining died, chunked transfer, and `Connection: close` as a debugging tool.
26. `http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer`: streams and frames, HPACK, per-stream **and** per-connection flow control (the window that silently caps gRPC throughput), priority and its failure, and the head-of-line blocking that multiplexing moved rather than removed. **Animated:** the same six requests under 1.1, 2 and 3.
27. `quic-and-http-3-what-moving-to-udp-actually-changed`: streams without transport HOL, connection IDs and migration across networks, 0-RTT, congestion control in user space, and the CPU cost that is the real deployment blocker. Numbers cited to public measurement work, not asserted.
28. `grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap`: HTTP/2 framing for RPC, deadline propagation down a call chain, cancellation, keepalive pings versus idle proxies, message size limits, and the trap that bites every team once: an L4 load balancer pins every request to one backend because there is only one connection.
29. `websockets-sse-and-long-lived-connections`: the upgrade, framing, what proxies and idle timeouts do to a connection nobody is using, backpressure on a socket you do not control, reconnect storms after a deploy, and fan-out patterns that survive a restart.
30. `payloads-on-the-wire-compression-record-sizes-and-the-mtu-you-forgot`: gzip versus brotli versus zstd as a CPU-for-bytes trade with the break-even derived, TLS record size and time to first byte, JSON versus protobuf measured on the wire, MSS clamping, jumbo frames, and the failure everyone meets exactly once. **Case:** PMTUD black holes behind a firewall that drops ICMP, RFC 2923 and RFC 4821.

## Track F: the path, proxies, and the edge (Wave 6)

31. `load-balancing-l4-vs-l7-what-each-one-cannot-do`: NAT mode versus DSR, balancing connections versus balancing requests, where TLS must terminate, preserving the client IP (`X-Forwarded-For`, PROXY protocol) and why it breaks silently, and the decision table. Wire-level complement to the system-design post, not a rewrite.
32. `balancing-algorithms-round-robin-least-request-and-the-power-of-two-choices`: why round robin hurts the tail, the P2C result and the intuition for why two samples is nearly as good as all of them, EWMA and least-request, slow start for cold backends, outlier ejection. **Lab:** a small simulation in `netlab` reproducing the p99 gap.
33. `health-checks-draining-and-the-graceful-shutdown-nobody-implements`: active versus passive detection, the arithmetic of detection time versus false ejection, connection draining, the SIGTERM to 503 race that produces errors on every deploy, and the keepalive mismatch that turns an idle timeout into an RST storm. Derived: interval, threshold and blast radius.
34. `anycast-cdns-and-the-edge`: how anycast picks a PoP and why it is not the nearest one, cache keys and hierarchy, origin shield, purge semantics, TTL strategy, and the caveat about long-lived TCP on anycast. **Case:** Fastly 2021-06-08 (one customer config triggered a latent bug, 85% of the network erroring).
35. `connection-pools-and-where-your-tail-latency-actually-lives`: pool sizing from Little's Law with the arithmetic shown, queueing at the pool as the hidden p99, keepalive tuning that must agree across every hop, retry budgets, hedged requests. **Case:** HikariCP's pool-sizing analysis, Dean and Barroso's "The Tail at Scale" (CACM 2013).
36. `nat-conntrack-and-port-exhaustion-the-limits-nobody-documents`: the 64k ceiling and its real arithmetic per destination tuple, conntrack table sizing and what happens at 100%, TIME_WAIT explained honestly, why `tcp_tw_recycle` broke NAT'd clients and was removed in Linux 4.12, SNAT in Kubernetes. **Case:** Bernat's TIME-WAIT analysis (2014), Cloudflare on running out of ephemeral ports (2022).

## Track G: networking inside the systems you run (Wave 7)

37. `kubernetes-networking-from-a-packets-point-of-view`: the pod network model, CNI, veth pairs and bridges, and one packet traced pod to pod, pod to service, node to node, with the commands to watch it at each step.
38. `kube-proxy-iptables-ipvs-and-ebpf`: how a Service VIP is actually implemented, why iptables rule count grows the way it does and what that costs at thousands of services, IPVS, Cilium and eBPF, and how each choice changes what you can debug. Scaling numbers cited to public benchmarks with versions.
39. `service-mesh-on-the-wire-what-a-sidecar-actually-adds`: the two extra hops drawn on the path map, the measured latency and CPU tax with version-dated sources, what you get back (mTLS, retries, circuit breaking, telemetry), and the sidecarless / ambient alternative.
40. `cloud-networking-vpcs-subnets-and-the-bill-you-didnt-expect`: VPC and subnet design, route tables, security groups versus NACLs, NAT gateway throughput limits and per-GB pricing, cross-AZ transfer charges with a worked monthly bill, PrivateLink versus peering versus transit gateway. **Case:** Slack 2021-01-04 (traffic ramp, Transit Gateway scaling lag, network saturation). Every price dated and region-named.
41. `datacenter-networks-clos-fabrics-ecmp-and-incast`: leaf-spine, oversubscription ratios, ECMP flow hashing and the elephant-flow problem, and TCP incast in any scatter-gather workload with the collapse derived. DCTCP and ECN as the fix. **Case:** the 2008 incast measurement work and DCTCP (SIGCOMM 2010).
42. `the-network-for-ai-workloads-rdma-rocev2-and-collective-traffic`: why training traffic is synchronised, bursty and intolerant of loss, RDMA and RoCEv2, PFC and congestion spreading, InfiniBand versus Ethernet, rail-optimised topologies. **Case:** Microsoft's "RDMA over Commodity Ethernet at Scale" (SIGCOMM 2016) and the Llama 3 paper's RoCE cluster section. Cross-links `distributed-training/`.

## Track H: diagnosis, design, and the senior mental model (Wave 8)

43. `a-diagnostic-ladder-for-network-problems`: the ordered ladder (is it resolution, is it the path, is it the handshake, is it the transport, is it the protocol, is it the app) with the one command that settles each rung and the output that proves it. The flowchart is the figure people will screenshot.
44. `reading-a-packet-capture-without-fear`: a guided capture in `netlab` where the reader sees a handshake, a retransmit, a zero window, an RST and a FIN, and learns what each looks like in `tshark`; capture filters that do not lie, ring buffers, and how to capture in production without making things worse.
45. `timeouts-retries-and-backoff-the-three-knobs-that-cause-outages`: timeout budgets down a call chain, retry amplification derived (three levels of three retries is twenty-seven), jitter, circuit breakers, retry budgets as a percentage rather than a count, idempotency as the precondition. **Case:** the AWS Builders' Library piece on timeouts, retries and backoff with jitter.
46. `partitions-split-brain-and-what-the-network-does-to-your-database`: asymmetric and gray partitions, failure detectors and phi-accrual, why "the network is reliable" is fallacy number one, and the failover policy question that is really a consistency question. **Case:** GitHub 2018-10-21 (a 43-second partition, an automated failover, 24 hours of degraded service).
47. `case-files-five-outages-and-the-layer-each-one-broke`: a compact anatomy of Meta's BGP withdrawal, Fastly's config push, Slack's Transit Gateway saturation, GitHub's partition and Roblox's discovery collapse, each mapped onto the same path map, each ending in the transferable guardrail: blast radius of a config push, control plane independent of data plane, out-of-band access, staleness that fails open.
48. `the-senior-engineers-network-mental-model`: **CAPSTONE.** The eight fallacies of distributed computing revisited with the evidence from 47 posts, the latency ladder and path map as permanent artefacts, a design checklist, the numbers worth memorising and the ones worth measuring forever, and the reading list.

---

## Case bank (pre-verified, public, cite with link and date)

Drafting agents: **verify each one against its public source before using it**, and quote the date. Do not add a case you cannot link.

**Outages and incidents:** Meta 2021-10-04 (BGP withdrawal) · Rogers Canada 2022-07-08 · KT Korea 2021-10-25 · Fastly 2021-06-08 · Cloudflare 2022-06-21 (19 data centres) · Cloudflare / CenturyLink 2020-08-30 (flowspec) · AWS us-east-1 2021-12-07 (internal network congestion) · Slack 2021-01-04 · GitHub 2018-10-21 (43-second partition) · Roblox 2021-10-28 to 10-31 · Salesforce 2021-05-11 (DNS) · Akamai Edge DNS 2021-07-22 · Dyn / Mirai 2016-10-21 · GitHub memcached DDoS 2018-02-28.

**Hijacks and routing:** Pakistan Telecom / YouTube 2008-02-24 · Amazon Route 53 / MyEtherWallet 2018-04-24 · RPKI and MANRS adoption data.

**Protocol and security events:** Heartbleed CVE-2014-0160 · DST Root CA X3 expiry 2021-09-30 · HTTP/2 Rapid Reset CVE-2023-44487.

**Engineering write-ups and papers:** Cloudflare (SYN packet handling, one latency spike, one NGINX worker taking all the load, ephemeral ports) · Vincent Bernat on TIME-WAIT (2014) · Weave and Xing on Kubernetes conntrack races and `ndots` · Netflix kTLS at 400 Gb/s (2021) · Dropbox web-server tuning (2017) · BBR (ACM Queue 2016) and BBR-versus-CUBIC fairness modelling (2019) · CoDel (2012) and RFC 8290 · QUIC at internet scale (SIGCOMM 2017) and "Taking a Long Look at QUIC" (IMC 2017) · Mitzenmacher on two choices, and NGINX's random-two · "The Tail at Scale" (CACM 2013) · TCP incast (2008) and DCTCP (2010) · RDMA over commodity Ethernet at scale (SIGCOMM 2016) · gRPC's own load-balancing post · HikariCP pool sizing · AWS Builders' Library on retries and jitter.

---

## Waves (one fresh session per wave, per CLAUDE.md)

| Wave | Track | Posts | Theme |
| --- | --- | --- | --- |
| 0 | infra | 0 | build the kit and `_render-net.sh`, write the figure primitives for the four recurring figures |
| 1 | A | 1 to 6 | the mental model and fundamentals |
| 2 | B | 7 to 12 | TCP properly |
| 3 | C | 13 to 18 | names, discovery, control plane |
| 4 | D | 19 to 24 | trust on the wire |
| 5 | E | 25 to 30 | the application protocols |
| 6 | F | 31 to 36 | the path, proxies, the edge |
| 7 | G | 37 to 42 | inside the systems you run |
| 8 | H | 43 to 48 | diagnosis and the senior model |

### Progress checklist

- [x] Wave 0: infrastructure kit, render helper, and four recurring figure primitives. COMPLETE 2026-09-29 in `f360ab43`: four author-scene and render gates passed, four visual reviews passed, 0 post WebPs by design.
- [x] Wave 1: Track A, the mental model and fundamentals, posts 1 to 6. COMPLETE 2026-09-29 in `f31ffa02`: 6 posts, 36 WebPs, 6 inline animations; all static visual reviews and post gates passed. Live browser playback was unavailable for the routing and latency animations, whose source, reduced-motion behavior, structural validation, and production rendering passed.
- [ ] Wave 2: Track B, TCP properly, posts 7 to 12
- [ ] Wave 3: Track C, names, discovery, and the control plane, posts 13 to 18
- [ ] Wave 4: Track D, trust on the wire, posts 19 to 24
- [ ] Wave 5: Track E, the application protocols, posts 25 to 30
- [ ] Wave 6: Track F, the path, proxies, and the edge, posts 31 to 36
- [ ] Wave 7: Track G, networking inside the systems you run, posts 37 to 42
- [ ] Wave 8: Track H, diagnosis, design, and the senior mental model, posts 43 to 48

Per-wave loop: dispatch 6 drafting agents in one message, let each run its own `figure-author` for Phase C and C2, gate each finished post through one `post-verifier`, commit that wave's `.md` plus `<slug>-N.webp` with explicit paths, `git pull --rebase --autostash`, push, then **close the session**.

## Drafting-agent brief (paste into every dispatch)

1. Read `.cache/blog-writer/_networking-series-kit.md` in full first.
2. Read only your own post's section of this plan (`offset` / `limit` or grep), never the whole file.
3. Depth `deep-dive`. 9k to 11k words, 7 figures, 4 distinct kinds, 1 animated figure.
4. Open with the latency ladder or the path map with your segment lit. Reuse the series' visual language, do not invent a new one.
5. At least one dated, linked, public case. Never invent an incident, a company or a benchmark.
6. Every number is derived, cited with a link and date, or reproducible in `netlab` with an expected range.
7. Close with a **Run it yourself** block: exact commands, the output to read, the range to expect.
8. No em dash. Brace-wrap inline math starting with a digit. Use `\lt ` not a bare `<` before a letter.
9. Link the intro post, the capstone, and two or three siblings. Cross-link into `system-design/`, `microservices/`, `sre/`, `api-design/` rather than restating them.
10. Delegate the whole figure pipeline to `figure-author` and never read a rendered webp yourself.
