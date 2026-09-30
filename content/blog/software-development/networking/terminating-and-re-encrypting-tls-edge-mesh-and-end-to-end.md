---
title: "Terminating and re-encrypting TLS: Edge, mesh, and end-to-end trust"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Choose a TLS termination boundary, prove what each hop protects, and diagnose the CPU and memory cost of encrypted traffic at scale."
tags:
  [
    "networking",
    "distributed-systems",
    "tls",
    "encryption-in-transit",
    "reverse-proxy",
    "service-mesh",
    "kernel-tls",
    "performance",
    "security-boundaries",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 43
image: "/imgs/blogs/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end-1.webp"
---

A browser shows a padlock. The edge proxy logs a successful TLS 1.3 handshake. A security review asks whether the request is encrypted all the way to the service. Three engineers answer yes for three different reasons: the public connection uses HTTPS, the internal hop stays in a private VPC, and a mesh sidecar has mutual TLS. All three observations can be true while the application receives plaintext from a local sidecar. They describe different segments and different principals.

![Path map locating TLS termination and the internal legs](/imgs/blogs/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end-1.webp)

The path map above is the mental model. A connection terminates where a process obtains the record keys and can read the HTTP request. After that point, any further protection requires a new security channel with its own peer identity, key material, policy, and monitoring. "HTTPS at the edge" is a statement about one connection. It is not a statement about the entire request path.

This article follows a request from client to edge to service. We will decide where TLS should end, separate link encryption from service identity, and price the record-encryption work that grows with traffic. At the end, a small Linux lab will show one plaintext internal hop and the effect of replacing it with TLS. For the handshake mechanics themselves, start with [what the TLS handshake buys and costs](/blog/software-development/networking/tls-what-the-handshake-buys-and-what-it-costs). For the full request path, use [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url).

## 1. A padlock identifies one peer, not a journey

**Name the process that has the keys before saying the request is encrypted.** A TLS client authenticates the peer named by the certificate it validates, negotiates traffic secrets, and sends protected records on that connection. In ordinary browser HTTPS, that peer may be a CDN edge, reverse proxy, API gateway, or origin. If a CDN edge presents the certificate, the browser has authenticated the edge's authority to serve that name. The browser has not thereby authenticated the private application process behind the edge.

TLS 1.3 gives integrity and confidentiality to records between the endpoints of one TLS connection. The [TLS 1.3 specification, RFC 8446, August 2018](https://www.rfc-editor.org/rfc/rfc8446) defines a handshake that derives traffic secrets and a record layer that protects application data. This is a cryptographic property of a channel. The statement says nothing about what an authorized endpoint does with plaintext after decryption. A reverse proxy may inspect an HTTP header, select an upstream, write a log line, run a web application firewall, or send a new request. Those are reasons to terminate TLS there, and each is also a reason that endpoint is in the trust boundary.

There are two separate questions in a design review. First: **Where can application plaintext exist?** Second: **Who is authenticated at each leg?** Encryption answers only part of the first. A network transport can be encrypted while the receiving workload is the wrong service, or while a compromised proxy legitimately holds the keys. Conversely, a service may be strongly authenticated by a private channel on one leg even though a neighboring local hop is plaintext. The architecture has to say which threat each protection addresses.

Suppose a request takes four stages: browser to edge, edge to origin ingress, ingress to service sidecar, and sidecar to application over loopback. You can choose TLS for the first three legs. The fourth is usually a local socket handoff, and it is plaintext unless you configure an additional protected connection. Calling this arrangement "end-to-end TLS" without qualification would hide the proxy and local termination points. Call it **hop-by-hop TLS with service identity on selected hops**. If the browser's payload must remain unreadable to the edge, application-layer encryption must run above the edge's HTTP processing, and the edge loses the ability to inspect that payload. That is a different product requirement.

![Four transport legs showing TLS endpoints and the local plaintext handoff](/imgs/blogs/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end-2.webp)

The four-row boundary matrix makes the scope explicit. The client-to-edge, edge-to-origin-ingress, and ingress-to-service-sidecar connections each have a distinct TLS endpoint pair. The final sidecar-to-app handoff is local and has no TLS in this example. A loopback address is a routing choice, not a cryptographic identity. A process with suitable host privileges, a compromised sidecar, or an overly broad Unix-socket permission can still affect that boundary. Its security comes from host isolation and access control unless another channel is added.

This distinction is practical. If the edge sees `Host`, path, cookies, or a bearer token, then it can route and enforce HTTP policy, but its logs and crash dumps become potential data exposure paths. If the sidecar owns service certificates, the app may not know which peer certificate was validated unless that identity is securely conveyed across the local handoff. A forwarded `X-Client-Cert` header is not automatically trustworthy. It must be stripped from untrusted inbound requests and injected by a trusted proxy, or conveyed through a channel that cannot be spoofed by the client. The adjacent [service identity and mTLS article](/blog/software-development/networking/mtls-and-service-identity-at-scale) treats rotation, SVIDs, and identity policy in detail. Here we care about where those properties begin and end on the wire.

### A precise vocabulary for a design document

| Phrase | What it should mean | What to verify |
| --- | --- | --- |
| TLS terminated at edge | Edge has traffic keys and can read HTTP | Listener configuration, certificate ownership, edge logs and memory boundary |
| Re-encrypted to origin | Edge is a new TLS client to a named upstream | Upstream scheme, CA bundle, name verification, certificate rotation |
| Mesh mTLS | Sidecars or workloads authenticate each other on a mesh leg | Identity in peer certificate, authorization policy, bypass paths |
| Local plaintext handoff | Proxy passes bytes to app without TLS on the local link | Socket permissions, loopback binding, namespace and process isolation |
| End-to-end payload secrecy | Intermediaries cannot read protected payload fields | Encryption at the actual producer and decryption at the actual consumer |

The table deliberately separates transport and payload. A client-to-edge TLS session can be perfectly secure against a passive observer yet fail the requirement "the edge operator must not read this medical field." That requirement needs a payload-level cryptographic design and a key boundary outside the edge. It may conflict with edge features such as content inspection, caching, compression, and request routing by encrypted fields. State the conflict before choosing the mechanism.

## 2. Terminate for a reason, then draw the new connection

**A TLS terminator is an application participant, not a transparent wire segment.** The moment the edge completes the handshake, it has a trusted role. It can validate a client certificate, inspect HTTP, normalize headers, decompress content, cache a response, choose an origin, apply a rate limit, and log request metadata. That is often exactly why an edge exists. Pretending that it merely passes an opaque tunnel leads teams to miss where authorization, data retention, and key control have moved.

An L4 load balancer can forward TCP while leaving the original TLS session to the origin. Its routing information is restricted to network and transport details, with some protocol-specific exceptions such as inspecting a ClientHello without terminating. An L7 proxy terminates or otherwise understands the application protocol, which permits routing by path or header and consistent policy enforcement. That difference belongs beside [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7), which owns the broader architecture choices. On the wire, the discriminating observation is simple: which process sends the ServerHello and owns the traffic keys, and which process opens the next socket?

Re-encryption means a proxy acts as a TLS client on the next leg. It does **not** mean it forwards the browser's ciphertext unchanged. The upstream TLS session has a different handshake, different traffic secrets, a different certificate check, and potentially a different protocol version or cipher suite. The browser's trust in the edge certificate does not imply the edge validated the origin certificate. A proxy configured to skip upstream verification can make a green padlock coexist with an unauthenticated internal connection. A proxy that uses TLS to the wrong upstream name can likewise give confidentiality to the wrong peer. The right test checks both encryption and identity.

The exact configuration syntax varies, but the review is portable. Find the edge's upstream URL scheme. Find the trusted CA roots or pinned trust domain. Find the expected DNS name or SPIFFE identity. Verify hostname checking is enabled. Confirm what happens when the origin certificate expires or rotates. Then deliberately break the upstream trust in staging and observe the failure. A connection that continues after an unknown certificate is accepted is not satisfying a server-authentication requirement, even if packet capture shows ciphertext.

Consider three possible edge-to-origin patterns. The first uses HTTP over a private subnet. This reduces TLS setup and cryptographic work, but plaintext can be read by any component with access at the relevant host or network boundary. The second uses HTTPS with server authentication. The edge verifies the origin, protecting the hop against passive reading and certain active redirection attacks. The third adds client authentication, so the origin also verifies the edge identity. That can support an explicit allow policy, provided direct bypass routes are closed. The [certificate-chain discussion](/blog/software-development/networking/certificates-chains-and-the-trust-you-inherit) explains the trust material and expiry behavior behind these checks.

At the service side, a mesh may run mTLS from ingress to sidecar and then hand plaintext to the app on localhost. The sidecar may carry authorization identity forward in a header or metadata. Its correctness depends on a trusted local interface and header sanitation. If an application can be reached directly on its plaintext port from another namespace or host, the mesh policy can be bypassed. The safe operational question is not merely "is mTLS enabled?" It is "can every request to this app only arrive through the authenticated listener?" Firewall rules, Kubernetes NetworkPolicy, socket binding, and pod layout are part of that answer. They are not a substitute for naming the TLS endpoints.

### Who pays for the extra hop?

Each new upstream connection may add handshake latency and CPU, though pooling and resumption can amortize that cost across many requests. The extra TCP socket has receive buffers, congestion state, timers, file descriptors, and failure modes. The proxy also must parse, copy, or otherwise move request and response bytes. An internal hop can fail independently of the browser-to-edge hop, so a client sees a 502 or timeout while its own TLS handshake succeeded. The [TLS handshake latency article](/blog/software-development/networking/tls-what-the-handshake-buys-and-what-it-costs) gives the round-trip model; here the key point is that the edge and origin have separate clocks. Measure them separately.

For a small response, a new connection's handshake can dominate. For a long-lived streaming response, record encryption and data movement dominate. A useful approximation for an uncached request is

$$
T_{\mathrm{total}} \approx T_{\mathrm{client\to edge}} + T_{\mathrm{edge\ work}} + T_{\mathrm{edge\to origin}} + T_{\mathrm{origin\ work}} + T_{\mathrm{return}},
$$

where each term is measured on a specific segment and overlapping work makes the sum an upper-level accounting model rather than a TLS protocol equation. Do not add an upstream handshake to every request if the proxy reuses its origin connection. Do not subtract it from a cold-connect tail merely because steady-state connection pooling works. The model is useful only when the measurement labels tell you whether a connection was warm.

## 3. The record layer has a different cost curve from the handshake

**Price handshakes per connection and records per byte.** Teams often respond to high TLS CPU by tuning connection reuse. That helps if the expensive work is key exchange or certificate verification at high connection churn. It does little for a video server that sends many gigabits on a modest number of established sessions. The heavy work there is symmetric record protection and moving data through memory and the NIC.

A full handshake negotiates keys and authenticates the server. The record layer then protects application bytes with an authenticated encryption algorithm. TLS 1.3 defines the key schedule and record protection in [RFC 8446](https://www.rfc-editor.org/rfc/rfc8446). In a conventional userspace implementation, a server can read file data, encrypt it in userspace, and write encrypted records to a socket. In a kernel TLS path, userspace still establishes the session, then installs the required record-protection state in the kernel. The [Linux kernel TLS documentation](https://docs.kernel.org/networking/tls.html) is explicit that its kernel path handles symmetric encryption after the handshake. The FreeBSD implementation and interface used by Netflix are their own implementation, so treat Linux details as a conceptual comparison rather than an assertion about FreeBSD code.

The place where encryption executes matters. Software kTLS can keep file-to-socket transfer on a kernel path and avoid some userspace copying, while host CPU cycles and memory bandwidth still pay for record encryption. NIC TLS offload gives the network device enough connection state to encrypt transmitted records as data leaves, shifting work away from host CPU and sometimes reducing host memory traffic. It creates different constraints: device session memory, firmware behavior, PCIe transaction throughput, supported cipher suites, retransmission handling, and observability. "Offloaded" never means "free." It names a new bottleneck location.

![Software and NIC kTLS data paths for a large transfer](/imgs/blogs/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end-3.webp)

The data-path comparison is useful. In the software path, data traverses host memory as the CPU reads and writes bytes for encryption before transmission. In the NIC path, plaintext reaches the NIC and encryption happens there. The exact number of memory passes depends on implementation, cache state, DMA behavior, file cache, NUMA placement, and copy avoidance. Drew Gallatin's [EuroBSDcon 2021 Netflix slides](https://papers.freebsd.org/2021/eurobsdcon/gallatin-netflix-freebsd-400gbps.files/gallatin-netflix-freebsd-400gbps-slides.pdf) present an illustrative 400 Gb/s target as 50 GB/s of payload. They estimate about 200 GB/s of memory bandwidth for the software kTLS flow and about 100 GB/s for NIC kTLS. Those are the slide's workload model, not universal constants for every TLS stack.

We can audit the first conversion without importing a benchmark claim. Using decimal network units, 400 gigabits each second divided by 8 bits per byte equals 50 gigabytes each second. If the data path transfers roughly four payload-sized streams through memory, the modeled traffic is 4 × 50 = 200 GB/s. If NIC offload reduces that to roughly two payload-sized streams, the modeled traffic is 2 × 50 = 100 GB/s. The point is not that every architecture has precisely four or two passes. The point is that a memory system with a measured sustainable bandwidth below a required movement rate cannot be rescued by a faster cipher instruction alone. Count the bytes crossing each bus.

The same distinction applies at smaller scale. Suppose a service sends 10 Gbit/s of large responses. The payload is 1.25 GB/s. Under a four-pass explanatory memory-traffic model, the host moves about 5 GB/s for that path, before accounting for other application work and contention. Under a two-pass model it moves about 2.5 GB/s. These are *derived scenario bounds*, not measurements. The actual choice requires CPU profiles, memory-controller counters, and NIC counters on the intended host. A system with an idle CPU can still be memory-bound; one with high CPU can be bottlenecked in another function entirely. The graph of utilization must be tied to the path, not to a dashboard label.

### What kernel TLS does and does not do

Kernel TLS is often misunderstood as putting the complete TLS protocol into the kernel. The [Linux interface documentation](https://docs.kernel.org/networking/tls.html) describes the userspace handshake and transfer of symmetric record state. The kernel handles subsequent record operations for configured directions. [Linux's offload documentation](https://docs.kernel.org/networking/tls-offload.html) distinguishes software cryptography from NIC packet-based offload and notes that transmit and receive state are installed independently. Application code and library support still matter. A browser-to-proxy or proxy-to-origin flow does not become offloaded because the NIC feature flag exists. The socket must use the relevant API, negotiate a supported cipher and TLS version, and qualify for the device path.

The same document explains that a failed hardware-offload attempt can fall back to software. From outside, packets remain TLS ciphertext either way. You therefore cannot prove hardware offload by observing encrypted packets in a capture. Inspect the application's kTLS usage, kernel or driver offload counters, NIC features, and CPU or memory behavior under a controlled workload. A high throughput number alone does not prove the NIC encrypted the records. Conversely, a small benchmark that shows no benefit may be dominated by handshake or request parsing rather than bulk bytes. Separate the workload phases first.

## 4. Trust boundaries are also failure boundaries

**Every termination point is a place to audit keys, plaintext, and policy.** An edge proxy can be a valuable choke point, but it is also a concentration of credentials and sensitive data. The origin may trust traffic from that proxy so strongly that it skips its own authentication. That works only if the network path, proxy identity, and bypass prevention are tested as carefully as the browser certificate.

Start by listing who can read plaintext. At edge termination, operators with access to the edge host, proxy process, memory dump, request logs, and tracing pipeline may see data. At origin termination, the origin and its host can. At mesh sidecar termination, the sidecar and potentially the app can. In a payload encryption design, those intermediaries can read routing metadata but not the protected fields, provided keys never reach them. None of these statements assigns moral trust to a component. They enumerate technical access.

Then list who can impersonate whom. A browser validates the public DNS name in an edge certificate against a trusted authority. An edge should validate an origin name or identity. A service sidecar should validate its peer workload identity and authorize that identity for the intended route. A certificate proves possession of a key and a bound name or identity. It does not prove that the request is allowed, that a header is correct, or that the process is bug-free. The distinction between authentication and authorization is especially important at a mesh boundary: mTLS can tell an ingress which workload is calling; policy decides whether that workload may call this API.

Finally, list what happens when any TLS leg fails. An expired public certificate blocks the client at the first boundary. An expired origin certificate may yield a proxy-side upstream TLS error while the public certificate remains valid. A sidecar rotation fault may break only one service-to-service leg. A diagnostic dashboard that reports merely "TLS errors" collapses these different owners. Label errors by listener, upstream cluster, peer name, certificate chain result, and protocol stage. This is the same principle as the recurring path map: locate the first boundary where the symptom appears.

| Symptom | Boundary to inspect | Discriminating evidence | First safe action |
| --- | --- | --- | --- |
| Client sees certificate error | Client to edge | Client validation error and served chain | Compare served SAN, dates, and chain with intended name |
| Browser handshake succeeds, then 502 | Edge to origin | Proxy upstream TLS error and origin listener logs | Confirm upstream CA and name verification without disabling them |
| One service call fails after ingress | Ingress to sidecar | Peer identity rejection or policy denial | Check the service identity and authorization rule |
| Packet capture shows plaintext on loopback | Sidecar to app | Local listener and capture scope | Verify intended local trust boundary and socket exposure |
| CPU rises with established bulk sessions | Record path | CPU profile, memory bandwidth, kTLS counters | Measure bytes and offload state before changing handshakes |

There is no number in this table because the diagnostic split is qualitative. The four channels should have different counters and logs. If the edge emits one aggregate "upstream handshake failed" metric, include the upstream cluster and verification failure in labels or structured logs. A packet capture can confirm which connection was encrypted, but it will not explain why a certificate check failed unless you also inspect application error reporting.

## 5. The Netflix 400 Gb/s goal exposes the real bottleneck

**Treat a conference result as a bounded experiment, then transfer the mechanism.** Drew Gallatin's talk, ["Serving Netflix Video at 400Gb/s on FreeBSD," EuroBSDcon, September 2021](https://papers.freebsd.org/2021/eurobsdcon/gallatin-netflix-freebsd-400gbps/), describes Netflix's Open Connect video-serving workload and a set of experiments to increase per-server encrypted delivery capacity. The title names an engineering target. The deck gives several measured results under different platforms, firmware versions, and offload modes. It should not be flattened into "Netflix universally served 400 Gb/s in production."

The baseline workload matters. The slides describe FreeBSD-current, NGINX, file serving through `sendfile(2)`, and software kTLS. The production platform described in the deck used an AMD EPYC 7502P Rome processor with 32 cores at 2.5 GHz, 256 GB DDR4-3200 across eight channels, two Mellanox ConnectX-6 Dx adapters with four 100 GbE ports in total, and 18 NVMe drives. Those details are not decorations. They explain why the relevant ceiling was not simply "four ports times 100 Gb/s." Disk DMA, host memory movement, NUMA locality, PCIe reads, record crypto, and NIC session state all participated in the same path.

The first measured software-kTLS result on that configuration was 240 Gb/s. Gallatin says it was limited by memory bandwidth, which the team checked with AMD uProf PCM. The deck's 400 Gb/s accounting requires 50 GB/s of transmitted payload and models about 200 GB/s of host memory bandwidth for software kTLS. The slide also cites about 150 GB/s single-node STREAM bandwidth as a proxy and about 175 GB/s across four NUMA nodes in that experiment. STREAM is a synthetic bandwidth proxy, not a direct video-serving measurement, but it helps explain why the arithmetic pointed to a memory wall. Adding CPU cores would not automatically remove traffic across the memory hierarchy.

![Gallatin's measured stages on specific platforms and firmware](/imgs/blogs/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end-4.webp)

The stage chart preserves the order of the deck's experiments. The early ConnectX-6 Dx NIC kTLS run with prerelease firmware peaked at about 125 Gb/s per NIC and sustained about 75 Gb/s per NIC, roughly 150 Gb/s total sustained. A later firmware change enabled PCIe Relaxed Ordering and the deck reports 160 Gb/s per NIC, about 320 Gb/s total, with peak and sustained effectively identical from that version forward. Production firmware added a `TLS_OPTIMIZE` setting, and the test reached roughly 190 Gb/s per NIC, about 380 Gb/s total peak and sustained. Those are *reported test results for the stated hardware and firmware progression*. The same slides say further quality-of-experience testing was needed before use of NIC TLS in Netflix production. They do not establish a universal production throughput number.

There is a second lesson inside the slide deck that is easy to miss. Hardware TLS offload stores per-session state on the NIC. The deck discusses about 400,000 active sessions for a 400 Gb/s workload and finite NIC memory that causes state to move between device and host buffers. Thus a small number of large flows and a very large number of active sessions can have different throughput on the same ports. The first limiting resource can shift from host memory bandwidth to NIC state handling and PCIe read behavior. When the team changed firmware, the change was not "faster encryption math" in the abstract. It altered the device's ability to keep its data path fed.

The deck also describes a safety concern with NIC encryption under loss. Retransmits need correct TLS record handling. The team considered moving sessions with significant retransmission back to software. At a threshold of 1 percent of bytes retransmitted, roughly one third of connections moved to software in their experiment, and maximum stable bandwidth fell from about 380 Gb/s to about 350 Gb/s. That result is particularly useful for capacity planning. A no-loss peak is not enough if normal traffic or an attack causes many connections to take the slower fallback path. A headroom budget should include the expected retransmission distribution, session count, and the proportion eligible for offload.

| Reported stage | Hardware and condition | Result | Interpretation | Source |
| --- | --- | --- | --- | --- |
| Software kTLS baseline | AMD EPYC 7502P Rome, NGINX, `sendfile(2)`, FreeBSD-current | 240 Gb/s | Memory-bandwidth-limited result in the cited test | [Gallatin, EuroBSDcon 2021 slides](https://papers.freebsd.org/2021/eurobsdcon/gallatin-netflix-freebsd-400gbps.files/gallatin-netflix-freebsd-400gbps-slides.pdf) |
| Initial NIC kTLS | Two ConnectX-6 Dx NICs, prerelease firmware | About 150 Gb/s total sustained | Early firmware/device behavior limited the test | [Gallatin, EuroBSDcon 2021 slides](https://papers.freebsd.org/2021/eurobsdcon/gallatin-netflix-freebsd-400gbps.files/gallatin-netflix-freebsd-400gbps-slides.pdf) |
| Relaxed Ordering enabled | Same NIC family, subsequent firmware | About 320 Gb/s total | PCIe transaction behavior mattered | [Gallatin, EuroBSDcon 2021 slides](https://papers.freebsd.org/2021/eurobsdcon/gallatin-netflix-freebsd-400gbps.files/gallatin-netflix-freebsd-400gbps-slides.pdf) |
| `TLS_OPTIMIZE` firmware | Same NIC family, production firmware in the deck | About 380 Gb/s total peak and sustained | Cited test approached the 400 Gb/s goal | [Gallatin, EuroBSDcon 2021 slides](https://papers.freebsd.org/2021/eurobsdcon/gallatin-netflix-freebsd-400gbps.files/gallatin-netflix-freebsd-400gbps-slides.pdf) |
| Loss-triggered software fallback | Threshold of 1% bytes retransmitted; about one third of sessions in software | About 350 Gb/s maximum stable | Mixed modes reduced the reported stable capacity | [Gallatin, EuroBSDcon 2021 slides](https://papers.freebsd.org/2021/eurobsdcon/gallatin-netflix-freebsd-400gbps.files/gallatin-netflix-freebsd-400gbps-slides.pdf) |

The table is intentionally repetitive about source and scope. A number copied without its firmware and traffic conditions becomes a false design target. For example, a team buying a NIC because it advertises TLS offload still needs to validate the supported TLS version, cipher, driver, firmware, kernel interface, session count, active-flow distribution, retransmit behavior, and PCIe topology. Otherwise it is comparing a feature checkbox to a whole-system benchmark.

### NUMA and PCIe are part of the TLS data path

Nonuniform memory access, or NUMA, means a core accesses some memory and devices more cheaply than others. The deck explores whether distributing work among NUMA nodes could move the software-kTLS ceiling. Locality matters because file data arrives from storage, is read for encryption, and is sent toward a NIC. If the storage controller, encryption worker, memory pages, and NIC sit on inconvenient nodes, bytes cross a shared interconnect. Moving an established connection toward a disk can improve data locality but can also change egress NIC and cause reordering or asymmetry. Gallatin's slides describe these trade-offs rather than claiming that one placement rule always wins.

With NIC TLS, the host may avoid a host-CPU encryption pass, but the NIC still pulls data over PCIe. The Ampere Altra experiment in the same deck illustrates the new ceiling: software kTLS was CPU-limited at 180 Gb/s, while initial NIC TLS was limited around 240 Gb/s with low CPU use and output drops. Enabling PCIe extended tags raised reported NIC TLS throughput to about 320 Gb/s on that platform. The slide explains the analogy to a larger window of outstanding DMA reads: more transactions can be in flight. This was an Ampere-platform observation, not a knob that guarantees the same result on an arbitrary server.

If a production profile shows low CPU during poor throughput, the wrong response is to conclude that encryption has no cost. The cost may now be hidden in host memory bandwidth, PCIe transactions, NIC on-board state, or retransmission fallback. Each is measurable, but with different counters. Ask which component must touch each byte, then identify the first constrained transfer between components.

## 6. Capacity math: useful bounds before a benchmark

**Turn a claimed line rate into bytes, passes, and state before selecting an accelerator.** A marketing number such as "400 Gb/s TLS" hides at least four dimensions: bytes per second, connections per second, concurrent sessions, and packet loss. Bulk video is mostly record throughput. A small-request API can be dominated by handshakes and HTTP parsing. A traffic mix with many short sessions can exhaust NIC state even if total bitrate is moderate.

For a bulk response, begin with a unit conversion:

$$
B_{\mathrm{payload}} = \frac{R_{\mathrm{wire}}}{8}.
$$

Here $R_{\mathrm{wire}}$ is an assumed payload rate in bits per second and $B_{\mathrm{payload}}$ is payload bytes per second. This is an explanatory capacity model; real Ethernet wire rate also includes framing, headers, retransmissions, and other overhead. If an application requires 80 Gbit/s of *payload*, the division gives 10 GB/s of payload before those overheads. If a software path moves four payload-sized copies through memory, the modeled traffic is 40 GB/s. If offload reduces it to two, the model gives 20 GB/s. Verify the pass count with profiler and controller counters. Do not treat the number as a vendor benchmark.

For sessions, Little's Law gives a useful occupancy estimate:

$$
N \approx \lambda_{\mathrm{conn}} W_{\mathrm{conn}},
$$

where $N$ is mean concurrent connections, $\lambda_{\mathrm{conn}}$ is new connections each second, and $W_{\mathrm{conn}}$ is mean connection lifetime in seconds under a steady-state approximation. At 2,000 new connections per second and 20 seconds mean lifetime, the model predicts 40,000 concurrent sessions. At the same bitrate but 200 seconds mean lifetime, it predicts 400,000. The byte rate could remain unchanged while the NIC's session-state demand changes by an order of magnitude. That is why Gallatin's hundreds-of-thousands-of-sessions discussion matters to an offload buyer.

The third bound is loss or fallback. If a fraction $f$ of bytes uses software crypto, a simple first-order CPU-work model is $C \approx f B c_{\mathrm{sw}}$, where $B$ is bytes per second and $c_{\mathrm{sw}}$ is CPU work per byte in the specific implementation. This is a modeling approximation, not a protocol equation or a measured value. It also misses memory contention and per-flow setup. Gallatin's measured mixed-mode reduction from 380 to 350 Gb/s shows why a linear cost assumption can be optimistic. Moving a third of sessions to software did more than add a tidy third of a benchmark's CPU graph; it changed system-wide memory and scheduling pressure.

![Conceptual constraints that can become the first TLS throughput limit](/imgs/blogs/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end-5.webp)

The constraint graph is a map of possible ceilings, not an empirical performance curve. Start with the record path, then ask whether the host CPU, memory system, PCIe path, NIC session store, or network port reaches a limit first. A change that removes one ceiling exposes the next. The conclusion is operational: capacity tests must use the target traffic mix, keep session count and loss visible, and record firmware and topology alongside throughput.

### A test matrix that can falsify the offload story

Do not benchmark only one request size and one connection count. Hold the certificate, TLS version, cipher suite, application payload, and server response constant, then vary one dimension at a time. A useful matrix has a short-response run to expose handshake and connection cost; a long-stream run to expose record throughput; a high-concurrency run to expose state limits; and a controlled-loss run to expose retransmit behavior. Repeat each run long enough to see sustained behavior, not only a warm-cache spike. Report throughput as application goodput as well as NIC bits per second so retransmissions are visible.

Record the host CPU profile, memory-controller bandwidth, per-NUMA-node placement, PCIe or driver counters, NIC offload counters, session counts, and retransmitted bytes. If hardware offload is truly active on bulk streams, a plausible observation is lower host crypto CPU at the same delivered bytes, but exact savings depend on the CPU, library, kernel, NIC, and data path. The result to publish is a table of the measured configuration, not a bare "TLS is faster" claim. State whether content came from page cache or storage, whether clients were on the same rack, and whether the bottleneck was the sender, receiver, or link.

## 7. The honest version of "encrypted inside the VPC"

**A private address and an encrypted link are different properties.** A VPC gives routing isolation and security controls in a cloud provider's network. It does not, by its name alone, prove that every packet is encrypted from workload to workload or that the receiving application has the expected identity. At the same time, "inside a VPC is always plaintext" is also too broad. Modern infrastructure can provide hardware link encryption for qualifying paths. The useful answer specifies the provider, instance families, path, region, controls, and the exact security boundary the feature promises.

For example, the [AWS Nitro System security whitepaper](https://docs.aws.amazon.com/pdfs/whitepapers/latest/security-design-of-aws-nitro-system/security-design-of-aws-nitro-system.pdf), checked September 30, 2026, describes transparent VPC traffic encryption between supported EC2 instance types in the same Region and VPC or peered VPCs. It explicitly conditions that property on the traffic avoiding a virtual network device or service such as a load balancer or transit gateway. This is a provider-host hardware guarantee for an eligible network segment. It is not equivalent to browser-to-application TLS, and it does not authenticate an HTTP service identity to the client application.

AWS also documents [VPC Encryption Controls](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-encryption-controls.html), checked September 30, 2026. Its monitor mode can expose encryption status through an enriched Flow Logs field; enforce mode blocks resources that do not meet its encryption requirements, subject to documented exclusions and service support. The controls can rely on application-layer encryption or Nitro hardware encryption. Their existence does not retroactively make every legacy VPC enforce encryption. The policy must be enabled, its status observed, and exceptions reviewed. AWS separately documents [Transit Gateway encryption support](https://docs.aws.amazon.com/vpc/latest/tgw/tgw-encryption-support.html) and cases where encryption is guaranteed only up to a gateway boundary. That is why a topology diagram belongs next to any compliance claim.

Suppose an edge proxy forwards HTTP to an origin on two supported Nitro EC2 instances. If the exact path qualifies for transparent hardware encryption, the provider may protect the bytes while they cross that network segment. The HTTP process still receives plaintext, and the edge does not validate an origin TLS certificate on that hop. If the traffic traverses a load balancer or other excluded device, the transparent encryption claim may no longer apply under the stated whitepaper conditions. If the organization requires application identity or cryptographic continuity across arbitrary intermediate networks, use an application or transport channel that supplies it and verify every peer. Provider encryption can be a useful layer, but the security statement must identify what it does.

The distinction also affects captures. A packet capture inside the guest may show HTTP plaintext even when the provider encrypts traffic below the guest at the Nitro card. That capture proves what the application emits, not the bits on the physical network. Conversely, a guest capture showing TLS ciphertext does not prove that hostname verification was enabled or that keys were not copied into an untrusted process. To audit the provider claim, use the provider's documented eligibility and encryption-status controls; to audit the application claim, inspect TLS configuration, peer identity, and endpoint logs. Neither observation replaces the other.

| Claim you want to make | Evidence that can support it | Claim it does not establish |
| --- | --- | --- |
| This internal hop is TLS protected | Successful peer-verified TLS handshake on the edge-to-origin socket | The edge cannot read the request |
| This qualifying cloud path has provider encryption | Supported path plus provider encryption-status evidence and control mode | The origin service was authenticated at application level |
| The app only accepts mesh-authenticated callers | Sidecar policy plus proof that direct app access is blocked | The local sidecar-to-app handoff is encrypted |
| The browser's sensitive payload stays secret from proxies | Payload encryption whose decryption key is held only at the intended consumer | The proxy can inspect encrypted fields for routing or WAF rules |

The table is a review checklist. It prevents a team from presenting one valid control as evidence for a different requirement. A control is useful when its promised boundary matches the threat. It becomes dangerous when the description quietly expands from "this eligible provider link" to "the whole application path."

## 8. Watch one request become three TLS sessions

The animation below follows a request rather than pretending that ciphertext is a single sealed envelope passed intact from the browser to the app. The edge ends the browser's TLS session, reads the permitted HTTP fields, then starts an edge-to-ingress TLS session. The ingress ends that session and starts a separate protected service-sidecar leg. In this concrete example, the sidecar hands a plaintext request to the local app. That last step can be designed differently, but it must be drawn and configured rather than inferred from the word "mesh."

<figure class="blog-anim">
<svg viewBox="0 0 840 260" role="img" aria-label="One request passes through separate TLS legs; edge and ingress decrypt then start new protected connections, while the sidecar passes plaintext to the local app" style="width:100%;height:auto;max-width:840px">
<style>
.tls22-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.tls22-txt{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.tls22-small{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.tls22-line{stroke:var(--border,#d1d5db);stroke-width:5;stroke-linecap:round}
.tls22-clear{stroke:#e9b44c;stroke-width:5;stroke-linecap:round}
.tls22-dot{fill:var(--accent,#6366f1);stroke:var(--background,#fff);stroke-width:3}
@keyframes tls22-hop{0%{transform:translateX(0);opacity:0}6%{opacity:1}23%{transform:translateX(170px)}28%{transform:translateX(170px)}46%{transform:translateX(340px)}51%{transform:translateX(340px)}69%{transform:translateX(510px)}74%{transform:translateX(510px)}94%{transform:translateX(680px);opacity:1}100%{transform:translateX(680px);opacity:0}}
.tls22-move{animation:tls22-hop 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.tls22-move{animation:none;transform:translateX(510px);opacity:1}}
</style>
<text class="tls22-txt" x="420" y="30">A request crosses four distinct trust legs</text>
<line class="tls22-line" x1="80" y1="125" x2="760" y2="125"/>
<line class="tls22-clear" x1="650" y1="125" x2="760" y2="125"/>
<rect class="tls22-box" x="20" y="70" width="120" height="100" rx="12"/>
<rect class="tls22-box" x="190" y="70" width="120" height="100" rx="12"/>
<rect class="tls22-box" x="360" y="70" width="120" height="100" rx="12"/>
<rect class="tls22-box" x="530" y="70" width="120" height="100" rx="12"/>
<rect class="tls22-box" x="700" y="70" width="120" height="100" rx="12"/>
<text class="tls22-txt" x="80" y="109">client</text>
<text class="tls22-txt" x="250" y="109">edge</text>
<text class="tls22-txt" x="420" y="109">ingress</text>
<text class="tls22-txt" x="590" y="109">sidecar</text>
<text class="tls22-txt" x="760" y="109">app</text>
<text class="tls22-small" x="165" y="205">TLS A</text>
<text class="tls22-small" x="335" y="205">TLS B</text>
<text class="tls22-small" x="505" y="205">TLS C</text>
<text class="tls22-small" x="675" y="205">local plaintext</text>
<text class="tls22-small" x="250" y="229">decrypt, inspect, re-encrypt</text>
<text class="tls22-small" x="420" y="248">decrypt, route, re-encrypt</text>
<circle class="tls22-dot tls22-move" cx="80" cy="125" r="10"/>
</svg>
<figcaption>The moving request reaches a TLS endpoint at each proxy; a new TLS session protects the next remote leg, while the local sidecar-to-app handoff remains plaintext in this example.</figcaption>
</figure>

The moving dot is a logical request, not a single TLS record. The records are created and authenticated independently on each protected leg, and the dot pauses at each proxy because that is where the old channel ends and a new one begins. The pause is where routing, authentication, logging, and policy can change. A packet capture taken only on the public listener sees the first session and cannot certify the others. Take evidence at each boundary.

## 9. Diagnose the first constrained boundary

**Start with the symptom and choose one measurement that can reject a hypothesis.** A burst of `ssl_handshake` errors at the edge is not a NIC line-rate problem. Low bulk goodput with flat handshake rate may be a record-path limit. A single upstream 502 with a successful public handshake points toward the origin leg. A mesh authorization denial is an identity or policy problem, even when all links are encrypted. These failures demand different tools.

![Decision tree for TLS hop failures and bulk encryption limits](/imgs/blogs/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end-6.webp)

The diagnostic tree orders the tests. First identify the leg. For handshake trouble, inspect certificate validation, peer name, version, and error counters. For established bulk flows, compare application goodput to socket and NIC counters, then profile CPU, memory bandwidth, offload state, and retransmits. For a proxy-to-app issue, inspect the proxy's upstream error and the app's listener exposure. This is the networking-series habit formalized in [the senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model): move down the path until a measurement changes, then work at that boundary.

### Separate a failed handshake from a slow data path

Use `curl` timing fields with caution. `time_connect` is time to complete TCP connection establishment to the peer that `curl` directly contacted. `time_appconnect` is the time until a TLS or other application connection setup completes. Their difference is a useful client-side approximation for the first TLS setup, but it may include scheduling and protocol behavior. It does not measure the edge's separate connection to its origin. Proxy logs or tracing must expose that upstream leg. A warm origin pool can hide new-connection cost in most requests while occasional cold requests dominate p99. Tag the measurements by connection reuse before computing an average.

At the host, a CPU profile can distinguish symmetric crypto functions from HTTP parsing, compression, cache lookup, and kernel network processing. Memory bandwidth tools can reveal pressure when CPU utilization is not the ceiling. On Linux, `ethtool -k` and driver counters can show whether the NIC advertises and uses TLS offload, but names and availability vary by kernel and driver. The [Linux kernel offload documentation](https://docs.kernel.org/networking/tls-offload.html) is the source for the conceptual software and hardware modes. On FreeBSD, use the implementation and driver counters appropriate to the release in use; do not paste Linux counter names into a FreeBSD incident report.

For the packet evidence, capture only the relevant interface and port. A TLS record capture will show ciphertext, sizes, timing, and TCP sequence behavior. It will not reveal whether an upstream certificate was accepted by a permissive client configuration. A loopback capture can prove that a local proxy-to-app hop carries plaintext in the guest namespace. Packet captures may contain credentials, tokens, and personal data when a plaintext hop is involved. Use a narrow filter, short duration, controlled lab payload, and protected storage. Do not collect arbitrary production traffic to settle a design debate.

### A runbook for the 502 that follows a green padlock

First, reproduce with a request ID and record whether the browser-to-edge handshake succeeded. If it did, the public certificate is not the first failed step. Second, correlate the edge's upstream connection log by request ID. Look for DNS failure, TCP connect timeout, TLS verification error, application timeout, or origin response status. Third, attempt a peer-verified TLS connection from the edge environment to the configured origin name and compare the served certificate chain and SAN with the expected name. Fourth, check whether the origin listener was restarted or its certificate rotated near the first failure. Fifth, confirm direct calls cannot bypass the edge policy. Only then change configuration. Disabling upstream verification to make the 502 disappear would move the failure from availability to trust.

The same method applies to throughput. If NIC counters show the port at capacity, upgrading crypto is unlikely to help. If the port is below capacity while host memory bandwidth is saturated, investigate data movement. If CPU crypto is hot, check cipher support, record path, and whether connections actually qualify for offload. If offload is active until retransmits rise, quantify the fallback distribution. One dashboard cannot answer all these questions because the system has several independent limits.

### Choosing a boundary without a slogan

The design choice starts with who must read the request. If the edge needs to route by URL path, block abusive HTTP payloads, or cache responses, it must see at least the relevant application fields. Terminate there and treat the edge as a privileged application component. Restrict access to its keys, set a retention policy for request logs, and avoid copying sensitive headers into generic telemetry. If the edge needs only to steer opaque TCP connections, preserving TLS to an origin may be simpler for the trust model, though it gives up most L7 features at that edge. Neither choice is inherently superior; each assigns work and authority to a different principal.

Next, decide what must be authenticated on the internal leg. Server-authenticated TLS answers "did the edge connect to the named origin?" Mutual TLS additionally lets the origin authenticate the edge's presented identity. That may be valuable when multiple callers share a network and origin policy needs a stable machine identity. The [service-to-service security article](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) owns the broader identity architecture; the networking decision is narrower: prove that the next socket uses the intended certificate and that an unauthenticated bypass socket is not available. A mesh policy cannot protect an app port that clients can reach directly.

Then decide whether provider link encryption counts toward the requirement. If the requirement is protection against physical-network observation on a documented eligible cloud segment, provider hardware encryption may satisfy it. If the requirement is cryptographic authentication of a workload, a provider link property alone does not supply that identity. If the requirement is that the proxy cannot read a payload, neither provider encryption nor re-encrypting after proxy termination solves it. Put those three requirements in separate rows of the threat model. The phrase "encryption in transit" is too broad to carry all three.

Finally, price the steady-state data path. An API with a high rate of short-lived client connections may benefit more from connection pooling, certificate-chain optimization, and resumption than from NIC record offload. A video server with huge established transfers may have the opposite shape. A mesh of small services adds sidecar hops and policy processing but may not come close to saturating a modern NIC. Spend effort where a profile and counter show the first constrained resource. The presence of TLS in a flame graph does not establish that it is the limiting factor if queueing, connection churn, or upstream service time dominates user latency.

| Requirement | Candidate termination design | Necessary proof | Main cost or lost capability |
| --- | --- | --- | --- |
| Edge must inspect and route HTTP | Client TLS ends at edge; edge starts verified origin TLS | Edge owns public cert; origin peer name and CA are checked; logs are controlled | Edge can read plaintext and must be trusted |
| Origin alone must read application payload | Pass through TLS or encrypt selected fields at the producer | Keys stay out of intermediary processes; origin validates intended client context | Edge loses inspection of protected fields |
| Workloads need mutual identity | Verified mTLS between designated workloads or sidecars | Peer identity, policy, rotation, and bypass prevention are tested | Certificate and policy lifecycle complexity |
| Bulk stream needs lower host crypto work | Software kTLS or qualified NIC offload, measured in the actual stack | Application eligibility, counters, firmware, traffic mix, loss, and sustained throughput | Host-memory or device constraints may replace CPU bottleneck |
| Eligible cloud link needs physical-path secrecy | Provider hardware encryption with monitored eligibility | Instance and path eligibility, policy state, and exceptions | Does not itself identify the application peer |

This table is a starting design record, not a product recommendation. It makes the proof obligation visible. A good review asks the team to demonstrate one success and one intentional failure per protected leg: a valid peer succeeds, a wrong-name or untrusted peer fails, and a direct bypass path fails. Do that in a test environment with disposable certificates. On the performance side, show a no-loss baseline and a controlled-loss or high-session test. These negative tests reveal much more than a screenshot of a green HTTPS lock.

There is also a migration trap. Teams often turn on edge-to-origin TLS in stages, first changing `http://` to `https://`, then planning to enable certificate verification later. The first stage encrypts bytes but leaves a gap in origin authentication. If the later stage never ships, the configuration can look secure to a superficial scanner while accepting an arbitrary upstream certificate. Define the target state at the start, test certificate rejection, and keep the two changes in one reviewable rollout when possible. If availability demands a staged change, record the interim risk and an expiry date rather than letting a permissive setting become permanent.

## Run it yourself

### Question

Can the client see a valid TLS connection to an edge while the edge-to-origin hop is plaintext, and can we change only the upstream hop to peer-verified TLS? This lab tests the *boundary claim*, not a throughput claim. It deliberately sends a harmless `GET /` so a local capture can show the HTTP method in the baseline. It does not benchmark Netflix's FreeBSD implementation or NIC offload.

### Preconditions

Use a Linux environment with the series' `netlab` namespaces `c` and `s`, address `10.77.0.1` on `c0`, and address `10.77.0.2` on `s0`. The [series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) owns namespace creation and teardown. Do not run namespace or interface mutations on a production host. You need `iproute2`, `nginx`, OpenSSL, Python 3, `curl`, `tcpdump`, `timeout`, and `sudo` or equivalent privileges for namespace execution and packet capture. Use a throwaway Linux VM if your workstation is macOS. The capture is scoped to loopback and the test ports; real plaintext captures can contain credentials.

Check the environment before creating anything:

```bash
set -euo pipefail
ip netns list | grep -E '^(c|s)( |$)'
ip -n c -br addr show c0
ip -n s -br addr show s0
ip -n c route get 10.77.0.2
ip netns exec s ss -lnt '( sport = :8080 or sport = :8443 or sport = :9443 )'
for tool in nginx openssl python3 curl tcpdump timeout; do command -v "$tool"; done
```

The listener check should show these three ports unused. If it does not, stop and choose an isolated `netlab` VM. These commands do not modify a running service. The lab creates files only under its named output directory and uses only the `s` namespace for listeners. The edge certificate is self-signed for `edge.netlab`; the origin certificate is self-signed for `origin.netlab`. They are lab trust anchors, not a production certificate pattern.

Prepare the certificates and a single tiny origin program. The same program serves both the HTTP baseline and the TLS treatment, so the application response remains `ok` while the upstream transport changes. The origin TLS process runs in advance, but the edge does not use it until the treatment.

```bash
set -euo pipefail
LAB="$PWD/netlab/out/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end"
mkdir -p "$LAB"
openssl req -x509 -newkey rsa:2048 -nodes -days 1 \
  -keyout "$LAB/edge.key" -out "$LAB/edge.crt" \
  -subj '/CN=edge.netlab' -addext 'subjectAltName=DNS:edge.netlab' \
  >/dev/null 2>&1
openssl req -x509 -newkey rsa:2048 -nodes -days 1 \
  -keyout "$LAB/origin.key" -out "$LAB/origin.crt" \
  -subj '/CN=origin.netlab' -addext 'subjectAltName=DNS:origin.netlab' \
  >/dev/null 2>&1
cat > "$LAB/origin.py" <<'PY'
import argparse
import http.server
import ssl

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--tls', action='store_true')
parser.add_argument('--cert')
parser.add_argument('--key')
args = parser.parse_args()

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'ok\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

server = http.server.HTTPServer(('127.0.0.1', args.port), Handler)
if args.tls:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.cert, args.key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
server.serve_forever()
PY
sudo ip netns exec s sh -c "nohup python3 '$LAB/origin.py' --port 8080 >'$LAB/http.log' 2>&1 & echo \$! >'$LAB/http.pid'"
sudo ip netns exec s sh -c "nohup python3 '$LAB/origin.py' --port 9443 --tls --cert '$LAB/origin.crt' --key '$LAB/origin.key' >'$LAB/tls.log' 2>&1 & echo \$! >'$LAB/tls.pid'"
```

The `LAB` variable is deliberately set in the shell you will use for the later snippets. If you open a new shell, set it again to the same path. The one-day certificates reduce the lifetime of this disposable lab material; do not import them into a real trust store. The `sudo` child processes need read access to the files. Keep the directory private if your host has other users.

### Baseline

Configure NGINX as the public-facing edge. It terminates client TLS on `10.77.0.2:8443` and forwards to the origin's plaintext loopback listener at `127.0.0.1:8080`. We configure origin TLS verification even in the baseline so that the treatment can change exactly the upstream URL and port.

```bash
cat > "$LAB/nginx.conf" <<EOF
pid $LAB/nginx.pid;
error_log $LAB/nginx.error.log notice;
events { worker_connections 128; }
http {
  access_log $LAB/nginx.access.log;
  server {
    listen 10.77.0.2:8443 ssl;
    server_name edge.netlab;
    ssl_certificate $LAB/edge.crt;
    ssl_certificate_key $LAB/edge.key;
    location / {
      proxy_pass http://127.0.0.1:8080;
      proxy_ssl_server_name on;
      proxy_ssl_name origin.netlab;
      proxy_ssl_verify on;
      proxy_ssl_trusted_certificate $LAB/origin.crt;
    }
  }
}
EOF
sudo ip netns exec s nginx -t -c "$LAB/nginx.conf"
sudo ip netns exec s nginx -c "$LAB/nginx.conf"
sudo ip netns exec s timeout 4 tcpdump -i lo -s 256 -A -c 12 'tcp port 8080' \
  > "$LAB/plain.capture" 2> "$LAB/plain.capture.err" &
CAP=$!
sleep 0.3
sudo ip netns exec c curl --silent --show-error --fail \
  --cacert "$LAB/edge.crt" \
  --resolve edge.netlab:8443:10.77.0.2 \
  https://edge.netlab:8443/
wait "$CAP" || true
grep -a 'GET /' "$LAB/plain.capture"
```

Read: the client response should be `ok`; `curl` should verify the edge certificate because its SAN matches `edge.netlab`; and `grep` should find `GET /` in the loopback capture. The exact number of captured packets varies, so the expected result is the method line rather than a packet-count claim. The packet capture is taken inside namespace `s` on `lo`, after edge termination. It does not reveal the public connection's plaintext. If you capture on `s0` at the public port instead, you should see TLS records rather than the HTTP method.

### Apply one change

Replace only the edge's upstream URL, from HTTP on port `8080` to HTTPS on port `9443`, and reload NGINX. The origin code and response body remain the same. NGINX must verify the `origin.netlab` certificate against the explicit lab trust anchor configured above. A successful response after this mutation is meaningful only because verification is enabled.

```bash
python3 - "$LAB/nginx.conf" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
before = path.read_text()
old = 'proxy_pass http://127.0.0.1:8080;'
new = 'proxy_pass https://127.0.0.1:9443;'
assert before.count(old) == 1
path.write_text(before.replace(old, new))
PY
sudo ip netns exec s nginx -t -c "$LAB/nginx.conf"
sudo ip netns exec s nginx -s reload -c "$LAB/nginx.conf"
```

### Compare

Repeat the same client request and capture the origin loopback leg on port `9443`. The capture is intentionally short and filtered. Do not interpret the absence of readable `GET /` as proof of peer verification by itself; NGINX's upstream TLS configuration and a negative certificate test establish that part.

```bash
sudo ip netns exec s timeout 4 tcpdump -i lo -s 256 -X -c 16 'tcp port 9443' \
  > "$LAB/tls.capture" 2> "$LAB/tls.capture.err" &
CAP=$!
sleep 0.3
sudo ip netns exec c curl --silent --show-error --fail \
  --cacert "$LAB/edge.crt" \
  --resolve edge.netlab:8443:10.77.0.2 \
  https://edge.netlab:8443/
wait "$CAP" || true
if grep -a 'GET /' "$LAB/tls.capture"; then
  echo 'unexpected readable HTTP method on protected origin leg' >&2
  exit 1
else
  echo 'no readable HTTP method on protected origin leg'
fi
sudo ip netns exec s openssl s_client \
  -connect 127.0.0.1:9443 -servername origin.netlab \
  -CAfile "$LAB/origin.crt" -verify_return_error </dev/null 2>&1 \
  | grep 'Verification: OK'
```

Read: the response remains `ok`; the loopback capture should show TLS handshake and application-data bytes rather than a readable HTTP method; `openssl s_client` should report `Verification: OK`. A short capture can miss the request altogether, so check that it contains packets on port `9443` before treating the `grep` result as evidence. A stricter negative test is to point NGINX's `proxy_ssl_trusted_certificate` at the edge certificate, reload, and expect an upstream verification error, then restore the origin certificate. That is an optional second experiment, not part of this one-change baseline comparison.

The result connects directly to the main claim: the client's verified TLS connection was present in both runs. Only the internal origin leg changed. Therefore the client padlock alone cannot certify whether that internal leg was protected. The two origin captures distinguish the configurations at the relevant boundary. This test proves transport behavior in the lab, not the security of every process on a real host. The local app still sees plaintext after its TLS server decrypts it.

### Reset

Stop only the processes launched for this lab, then remove only this lab's generated files. Leave `netlab` namespaces and interfaces intact for other posts.

```bash
sudo ip netns exec s nginx -s quit -c "$LAB/nginx.conf" || true
for pidfile in "$LAB/http.pid" "$LAB/tls.pid"; do
  if test -f "$pidfile"; then sudo kill "$(cat "$pidfile")" 2>/dev/null || true; fi
done
rm -rf -- "$LAB"
```

On a production host, use read-only checks first: inspect the proxy's upstream scheme and TLS verification settings, compare listener ports with `ss -lnt`, inspect certificate expiry and peer names, and use structured upstream TLS error logs. Do not install self-signed lab keys, reload a production proxy, or capture arbitrary payloads to answer this question.

## Key takeaways

1. A browser padlock authenticates one TLS peer. Each proxy termination ends that connection and creates a new trust decision.
2. Draw every leg, including local sidecar-to-app handoffs. State where plaintext and traffic keys exist.
3. Re-encryption is useful only when the next peer's identity is verified and bypass routes are controlled.
4. Bulk TLS capacity depends on record bytes, host memory traffic, PCIe, NIC state, and fallback behavior, not only handshake CPU.
5. Gallatin's 2021 Netflix deck reports a 400 Gb/s goal and configuration-specific results up to about 380 Gb/s in the NIC-offload test. Preserve the hardware and firmware scope.
6. A VPC can provide isolation and, on eligible paths, provider hardware encryption. Verify the exact path and control state before making an application-level encryption claim.
7. Diagnose the first failing or constrained boundary with a targeted measurement. Public handshake success does not rule out an origin TLS error or a plaintext internal hop.

## Further reading

- [Drew Gallatin, "Serving Netflix Video at 400Gb/s on FreeBSD," EuroBSDcon 2021](https://papers.freebsd.org/2021/eurobsdcon/gallatin-netflix-freebsd-400gbps/), with the [primary slide deck](https://papers.freebsd.org/2021/eurobsdcon/gallatin-netflix-freebsd-400gbps.files/gallatin-netflix-freebsd-400gbps-slides.pdf).
- [RFC 8446, TLS 1.3, August 2018](https://www.rfc-editor.org/rfc/rfc8446), for handshake and record semantics.
- [Linux kernel TLS](https://docs.kernel.org/networking/tls.html) and [TLS offload](https://docs.kernel.org/networking/tls-offload.html), for a platform-specific explanation of software and NIC modes.
- [AWS VPC Encryption Controls](https://docs.aws.amazon.com/vpc/latest/userguide/vpc-encryption-controls.html), for path eligibility, monitoring, enforcement, and exceptions as checked September 30, 2026.
