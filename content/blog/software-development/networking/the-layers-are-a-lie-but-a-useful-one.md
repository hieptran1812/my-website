---
title: "The Layers Are a Lie, but a Useful One"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn to locate a network failure by the first broken contract, price every header, and recognize MTU as the leak that crosses layers."
tags:
  [
    "networking",
    "distributed-systems",
    "osi-model",
    "tcp-ip",
    "encapsulation",
    "packet-analysis",
    "mtu",
    "path-mtu-discovery",
    "tls",
    "http",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 45
image: "/imgs/blogs/the-layers-are-a-lie-but-a-useful-one-1.webp"
---

An alert says checkout is timing out. The application team sees no completed requests. The platform team sees healthy pods. The network team sees interfaces up and packets moving. Everyone is telling the truth, and the incident is still unresolved.

The phrase "network problem" is usually where useful reasoning stops. It collapses name resolution, routing, packet delivery, transport state, encryption, proxy behavior, and application semantics into one bucket. The better question is: **which contract failed first, and which observer could have seen that failure?**

The diagram below is the mental model: one request crosses several contracts, but no component owns or understands all of them. A switch can forward a TLS record without knowing it is TLS. A router can discard an IP packet without knowing it carried the final bytes of an HTTP response. An HTTP server can return `503` without knowing which physical links carried the request. Our job is to locate the first broken contract, not to pick a team to blame.

![A request crosses link, IP, transport, TLS, and HTTP contracts, with different components terminating different contracts](/imgs/blogs/the-layers-are-a-lie-but-a-useful-one-1.webp)

This post builds the working stack used by the rest of [Networking for Engineers Who Ship Services](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). We will replace the seven-box classroom diagram with the stack that actually runs, compute the byte cost of one packet, draw the visibility boundary at every layer, and use MTU as the first constraint that leaks upward. The companion posts on [packet forwarding](/blog/software-development/networking/how-a-packet-gets-there-arp-switching-routing-and-ecmp), [TCP and UDP sockets](/blog/software-development/networking/sockets-and-the-two-transport-contracts-tcp-vs-udp), and [latency budgets](/blog/software-development/networking/the-latency-budget-speed-of-light-serialization-and-queueing) take the individual contracts deeper. The [series capstone](/blog/software-development/networking/the-senior-engineers-network-mental-model) assembles them into an incident model.

## The useful stack is a set of contracts, not seven boxes

The Open Systems Interconnection model is useful vocabulary. It is not a literal call graph, a packet format, or a faithful diagram of every production path. Treating its seven layers as physical compartments creates two recurring mistakes: assuming every packet passes through one implementation of every layer, and assuming one layer has no effect on another.

The deployed Internet model is more compact. [RFC 1122, published in October 1989](https://www.rfc-editor.org/rfc/rfc1122.html), organizes host communication into link, Internet Protocol, transport, and application layers. In real services we usually split out TLS because its termination boundary is operationally decisive. We also name HTTP because its messages, status codes, routing fields, and retry semantics are the evidence application engineers actually inspect.

That gives us this working stack:

| Contract | Unit we reason about | Primary promise | Typical owner | Evidence that settles a dispute |
| --- | --- | --- | --- | --- |
| Link | frame | Deliver across one local link | NIC, driver, bridge, switch | interface counters, neighbor state, frame capture |
| IP | packet or datagram | Address and forward across networks | host IP stack, router | route lookup, TTL, ICMP, packet capture |
| TCP | ordered byte stream | Reliable, ordered delivery with flow and congestion control | endpoint kernels | socket state, sequence numbers, retransmissions |
| UDP | datagram | Preserve message boundaries with minimal transport machinery | endpoint kernels and application | socket state, datagram lengths, loss observed by the application |
| TLS | record and handshake message | Authenticate peers and protect record contents | client, terminator, service | handshake transcript, alerts, certificate verification |
| HTTP | request, response, stream | Express application methods, metadata, status, and content | client, proxy, handler | access log, trace, status, protocol-native capture |

The word *contract* matters. IP promises best-effort datagram delivery, not reliable delivery. TCP adds ordering and retransmission, but does not promise that the application will read promptly. TLS protects content, but does not hide every traffic property. HTTP gives method and status semantics, but does not know which router discarded a packet.

The boundaries are also termination points. A router terminates one incoming link frame, processes the IP packet, then creates a new outgoing link frame. It does not forward the original Ethernet frame end to end. A TLS proxy terminates one TLS connection and can create another. An L7 load balancer can choose a backend from an HTTP path only after the request becomes visible at its termination point. This is why the distinction between [L4 and L7 load balancing](/blog/software-development/system-design/load-balancing-from-l4-to-l7) is a visibility distinction before it is a product-category distinction.

### OSI names are coordinates, not proof

Saying "layer 4" can locate a conversation quickly. It cannot prove that TCP caused the symptom. A firewall may inspect transport ports while maintaining connection state. A QUIC connection carries transport machinery inside encrypted UDP payloads. A tunnel inserts another IP header under what an application thought was its network path. A service mesh terminates and originates TLS while presenting an ordinary socket to the application.

Use layer names the way a building engineer uses floor numbers. They locate a problem. They do not tell you whether the failed object is plumbing, power, structure, or software.

> A layer is a promise about what one component may assume. It is not permission to ignore evidence from the layer below.

The strongest consequence is diagnostic: start with the first contract whose expected evidence is absent. If DNS returns no address, do not begin with TCP congestion control. If the SYN receives no SYN-ACK, an HTTP status dashboard cannot tell you why. If TLS completes and the server returns a valid `503`, the link did its job for that exchange even if a network dependency later contributed to overload.

### The model names contracts, not implementation processes

One process can implement several contracts, and one contract can be split across several components. A browser implements HTTP, TLS, transport-facing socket behavior, DNS policy, connection pooling, and retry logic. The kernel usually implements IP and TCP, while the NIC may complete checksums or segment large buffers. A proxy terminates a client-side TCP and TLS connection, parses HTTP, chooses an upstream, then originates another transport and security context.

That physical co-location does not erase the logical boundaries. It makes them more important because a single process can transform evidence as bytes cross from one contract to another. When a proxy reports `upstream_connect_time`, it describes a new connection created after the downstream request reached the proxy. It does not describe the client's original connect time. When a kernel reports retransmissions, it describes one TCP connection. It does not identify which HTTP request was delayed if several requests share that connection.

Likewise, one logical layer can have several implementations along a path. A client TLS connection might terminate at an edge proxy. The edge could start a second TLS connection to a regional proxy, which starts a third to a service sidecar. Saying "TLS succeeded" is ambiguous until we name which leg and which endpoint observed success.

This is the operational version of the end-to-end argument: preserve a narrow lower-layer service, then put semantics where the endpoints can enforce them. The principle does not ban middleboxes. It requires us to acknowledge every place that terminates, translates, or originates a contract.

Use a connection ledger during difficult incidents:

| Leg | Local endpoint | Remote endpoint | Transport evidence | Security evidence | Application evidence |
| --- | --- | --- | --- | --- | --- |
| Client to edge | client socket | edge listener | client capture and edge socket | client handshake and edge termination log | request visible after edge decrypts |
| Edge to proxy | edge upstream socket | regional proxy | both endpoint socket states | independent TLS context | forwarded request and proxy timing |
| Proxy to service | proxy upstream socket | service or sidecar | final transport leg | service-side identity and alert | handler trace and response status |

The table prevents a common correlation error: using success on one leg as proof of success on every leg. A client can complete TLS to an edge while the edge cannot connect to an origin. The resulting HTTP `502` is valid application evidence about the edge's failure to satisfy the request. It is not proof that the client-to-edge transport failed.

## The stack that actually runs

The stack is easiest to understand by following one outbound write. Suppose an HTTP/1.1 client writes a request onto a TLS-protected TCP connection. The exact APIs vary, but the responsibilities are stable:

1. The HTTP implementation serializes a request line, header fields, the empty line that ends the header section, and an optional body. [RFC 9112, published in June 2022](https://www.rfc-editor.org/rfc/rfc9112.html) defines that HTTP/1.1 message shape.
2. TLS packages plaintext into records, adds an inner content type, optionally pads it, encrypts it, and adds record metadata and an authentication tag.
3. TCP accepts bytes, chooses segments according to its effective maximum segment size and current state, and adds ports, sequence information, acknowledgement information, flags, and options.
4. IP adds source and destination addresses, a next-protocol identifier, hop lifetime, and other network-layer fields.
5. The link layer wraps the IP packet for one hop with local addressing and frame integrity fields.
6. The NIC and physical medium transmit a signal. Offload may delay or change where the host software appears to perform some segmentation or checksum work, but it does not remove the wire contract.

At the receiver, the order reverses. The NIC validates and admits a frame. IP validates enough of the packet to deliver it to the appropriate transport. TCP reconstructs the byte stream. TLS authenticates and decrypts records. HTTP parses messages. Each step removes context that was meaningful to that step and hands the remaining payload upward.

This is why a packet capture is not automatically "the truth." It is the truth at a named observation point. A capture above a virtual tunnel sees an inner packet that will later receive an outer header. A capture on a transmit host with TCP segmentation offload may show a buffer larger than the physical MTU because the NIC will segment it later. A capture at a switch mirror sees what crossed that switch, which may differ from what a host capture saw before checksum offload completed.

Before interpreting any capture, write down four facts:

| Question | Why it changes interpretation |
| --- | --- |
| Where was the capture taken? | Before and after a tunnel or proxy are different protocol paths. |
| Which interface was captured? | Loopback, bridge, veth, tunnel, and physical NIC expose different envelopes. |
| Which offloads were enabled? | GSO, TSO, GRO, and checksum offload can make host captures look unlike wire frames. |
| Was traffic encrypted at that point? | HTTP fields are unavailable before TLS termination even when packet delivery is visible. |

On Linux, these read-only commands establish the observation point before we infer anything:

```bash
ip -details link show dev eth0
ip route get 203.0.113.10
ethtool -k eth0
ss -tin dst 203.0.113.10
tcpdump --version
```

`ip link` reports the interface MTU. `ip route get` asks the kernel which route it would use. `ethtool -k` exposes offload state. `ss -tin` shows TCP state and transport metrics for matching sockets. None alone proves application health. Together they tell us where a later observation sits.

## Encapsulation: price every header on one real packet

Encapsulation is often taught as boxes nesting inside boxes. The useful version is a budget. Every lower-layer limit must pay for every header above it, and every optional header reduces what remains for application bytes.

Start with a common Ethernet IP maximum transmission unit of 1,500 bytes. [RFC 894, published in April 1984](https://www.rfc-editor.org/rfc/rfc894.html) specifies that the maximum IP datagram carried in the Ethernet data field is 1,500 octets. Here an octet is an 8-bit byte. The MTU counts the IP packet, not the Ethernet header, frame check sequence, preamble, or inter-frame gap.

![Header accounting from a 1500-byte Ethernet MTU through IPv4, TCP, TLS 1.3, and application data](/imgs/blogs/the-layers-are-a-lie-but-a-useful-one-2.webp)

Under deliberately stated minimum-header assumptions, the arithmetic is:

$$
\text{TCP payload budget} = 1500 - 20_{\text{IPv4}} - 20_{\text{TCP}} = 1460\ \text{bytes}
$$

[RFC 791, published in September 1981](https://www.rfc-editor.org/rfc/rfc791.html) encodes the IPv4 Internet Header Length in 32-bit words, with a minimum value of five words, so the minimum is ${5 \times 4 = 20}$ bytes. [RFC 9293, published in August 2022](https://www.rfc-editor.org/rfc/rfc9293.html) gives 20 bytes as the fixed TCP header size when no TCP options are present. Options make the header larger and the data allowance smaller for that segment.

Now protect application bytes with TLS 1.3 using AES-128-GCM, no optional padding, and one TLS record aligned within this TCP payload for a simple accounting example. This is an explanatory packetization model, not a requirement that TLS records align one-to-one with TCP segments.

The minimum overhead in this stated model is:

$$
5_{\text{record header}} + 1_{\text{inner content type}} + 16_{\text{GCM tag}} = 22\ \text{bytes}
$$

[RFC 8446, published in August 2018](https://www.rfc-editor.org/rfc/rfc8446.html) defines the five-byte TLSCiphertext header from a one-byte opaque content type, two-byte legacy record version, and two-byte length. It also places the real content type inside TLSInnerPlaintext. [RFC 5116, published in January 2008](https://www.rfc-editor.org/rfc/rfc5116.html) specifies that AEAD_AES_128_GCM produces ciphertext exactly 16 octets longer than its plaintext because of the authentication tag.

Therefore:

$$
\text{application allowance} = 1460 - 22 = 1438\ \text{bytes}
$$

The number is not a universal TLS payload size. It is the result of explicit assumptions: IPv4 without options, TCP without options, TLS 1.3, AES-128-GCM, no TLS padding, and a record boundary chosen to fit one TCP payload. Real TCP options, IPv6, another AEAD construction, TLS padding, HTTP framing, tunnels, or a different record strategy change it.

| Item | Bytes | Included in 1,500-byte IP MTU? | Source |
| --- | ---: | --- | --- |
| Ethernet header | 14 | No | [Linux kernel netdevice documentation](https://docs.kernel.org/networking/netdevices.html), checked September 2026 |
| IPv4 minimum header | 20 | Yes | RFC 791, September 1981 |
| TCP minimum header | 20 | Yes | RFC 9293, August 2022 |
| TLSCiphertext record header | 5 | Yes, inside TCP payload | RFC 8446, August 2018 |
| TLSInnerPlaintext content type | 1 | Yes, encrypted | RFC 8446, August 2018 |
| AES-128-GCM tag | 16 | Yes, inside TCP payload | RFC 5116, January 2008 |
| Application allowance in this model | 1,438 | Yes | Derived here: ${1500-20-20-5-1-16}$ |
| Ethernet frame check sequence | 4 | No | [Linux kernel netdevice documentation](https://docs.kernel.org/networking/netdevices.html), checked September 2026; capture visibility depends on hardware and driver |
| Preamble plus start delimiter | 8 | No | [IEEE 802.3 task-force material](https://www.ieee802.org/3/cg/public/adhoc/cordaro_8023cg_short_reach_new_preamble_proposal_1220.pdf), December 2017 |
| Inter-frame gap equivalent | 12 | No | [IEEE 802.3 task-force material](https://www.ieee802.org/3/cg/public/adhoc/cordaro_8023cg_short_reach_new_preamble_proposal_1220.pdf), December 2017 |

For a full 1,500-byte IP packet, Ethernet carries ${14 + 1500 + 4 = 1518}$ bytes from destination MAC through frame check sequence. The [Linux kernel's network-device documentation](https://docs.kernel.org/networking/netdevices.html), checked in September 2026, uses the same 1,518-byte accounting. If we price medium occupancy, add the 8 bytes of preamble and start delimiter plus the 12-byte minimum inter-frame gap documented in [IEEE 802.3 task-force material from December 2017](https://www.ieee802.org/3/cg/public/adhoc/cordaro_8023cg_short_reach_new_preamble_proposal_1220.pdf): ${1518 + 8 + 12 = 1538}$ byte-times. Packet captures commonly omit the preamble, gap, and often the frame check sequence, so never subtract fields merely because a capture UI does not display them.

### UDP changes the budget, not the principle

[RFC 768, published in August 1980](https://www.rfc-editor.org/rfc/rfc768.html) defines a minimum UDP length of eight bytes, including header and data. On the same 1,500-byte IPv4 path:

$$
\text{UDP application datagram budget} = 1500 - 20 - 8 = 1472\ \text{bytes}
$$

UDP exposes more payload per datagram than minimum-header TCP, but it does not provide TCP's reliable ordered stream. If an application builds reliability, ordering, congestion control, and cryptographic framing over UDP, those mechanisms consume bytes and state somewhere else. QUIC is not "free TCP plus TLS." It deliberately moves transport machinery into a user-space encrypted protocol over UDP.

### Header percentage depends on payload size

For a tiny 40-byte application message in our TLS model, the IP packet carries ${40 + 22 + 20 + 20 = 102}$ bytes. The application fraction of the IP packet is:

$$
\frac{40}{102} \times 100 \approx 39.2\%
$$

For 1,438 application bytes, the same headers fill the 1,500-byte IP packet, so the application fraction is:

$$
\frac{1438}{1500} \times 100 \approx 95.9\%
$$

This does not mean "batch everything" without limit. Larger writes can improve byte efficiency while increasing buffering delay, memory use, loss recovery cost, or head-of-line blocking. The correct trade-off belongs to the application protocol and transport behavior, not to header arithmetic alone.

### Work one HTTP request all the way down

Take this exact HTTP/1.1 request, where `\r\n` means the two terminating bytes carriage return and line feed:

```http
GET /health HTTP/1.1\r\n
Host: svc.example\r\n
Connection: close\r\n
\r\n
```

Count the ASCII bytes rather than estimating them:

```bash
printf 'GET /health HTTP/1.1\r\nHost: svc.example\r\nConnection: close\r\n\r\n' \
  | wc -c
```

Expected output is `62`: the request line is 22 bytes, the Host field is 19, the Connection field is 19, and the final empty line contributes 2. The arithmetic is ${22 + 19 + 19 + 2 = 62}$ bytes. The values follow directly from the shown string and can be reproduced with POSIX `printf` and `wc`.

Protect those 62 bytes using the same TLS 1.3 AES-128-GCM assumptions. TLS contributes 22 bytes, producing an 84-byte TCP payload:

$$
62_{\text{HTTP}} + 5_{\text{TLS header}} + 1_{\text{inner type}} + 16_{\text{tag}} = 84\ \text{bytes}
$$

Add minimum TCP and IPv4 headers:

$$
84 + 20_{\text{TCP}} + 20_{\text{IPv4}} = 124\ \text{bytes in the IP packet}
$$

Add Ethernet header and frame check sequence:

$$
124 + 14_{\text{Ethernet}} + 4_{\text{FCS}} = 142\ \text{bytes in the MAC frame}
$$

Then price medium occupancy:

$$
142 + 8_{\text{preamble and SFD}} + 12_{\text{gap}} = 162\ \text{byte-times}
$$

The request's 62 application bytes occupy only ${62 / 162 \approx 38.3\%}$ of those byte-times in this simplified one-frame model. This result deliberately excludes TCP options, VLAN tags, acknowledgements, handshake traffic, DNS, ARP or neighbor discovery, and any tunnel. It also assumes the request fits one TLS record and one TCP segment. The point is not that every health check costs exactly 162 byte-times. The point is that "a 62-byte request" names only one layer's contribution.

| Accounting boundary | Cumulative bytes or byte-times | Derivation | Source |
| --- | ---: | --- | --- |
| HTTP/1.1 request | 62 bytes | Exact shown string counted by `wc -c` | Reproducible command above |
| TLSCiphertext | 84 bytes | ${62+5+1+16}$ | RFC 8446 and RFC 5116 |
| IPv4 packet | 124 bytes | ${84+20+20}$ | Minimum headers from RFC 791 and RFC 9293 |
| Ethernet MAC frame | 142 bytes | ${124+14+4}$ | Derived Ethernet II accounting |
| Medium occupancy | 162 byte-times | ${142+8+12}$ | Derived on-wire accounting |

Acknowledgements make the exchange bidirectional even before an HTTP response arrives. TCP can acknowledge bytes without the application having produced a response. TLS handshake traffic may dwarf a tiny request on a cold connection. DNS may precede connection establishment. The [latency-budget post](/blog/software-development/networking/the-latency-budget-speed-of-light-serialization-and-queueing) prices those phases in time; this example prices one application write in bytes.

### Why packet boundaries are not message boundaries

The worked example chose convenient alignment for teaching. Production stacks are free to package data differently within protocol rules. One HTTP header section can span multiple TLS records. One TLS record can span multiple TCP segments. A TCP segment can be split into IP fragments in legacy IPv4 situations, although avoiding fragmentation is usually preferable. Offload can cause the segmentation visible in host software to differ from the frames emitted by the NIC.

Conversely, one TCP segment can carry bytes from more than one application write or more than one small TLS record. TCP exposes an ordered stream. It does not preserve application write calls as receiver-visible message boundaries. Code that assumes one `read()` returns one HTTP message is broken even on a lossless local network.

That distinction is why packet count is a poor proxy for request count. It is also why a capture must be reassembled at the correct layer before we interpret higher-level fields. First reconstruct IP where fragmentation exists, then TCP byte order, then TLS records at an authorized endpoint, then HTTP message framing. Skipping a step can turn a partial header into a false parser error or make an ordinary segment boundary look like application truncation.

## Encapsulation is reversible context, not decorative wrapping

The animation below makes the important part explicit. Outbound, each contract adds the context its peer needs. Inbound, the peer removes that context only after validating or acting on it. The payload is not blindly wrapped and unwrapped by one monolithic network function.

<figure class="blog-anim">
<svg viewBox="0 0 920 560" role="img" aria-label="A payload descends the client stack as HTTP, TLS, TCP, IP, and Ethernet headers accumulate, crosses the link, then ascends the server stack as those headers are removed" style="width:100%;height:auto;max-width:920px">
<style>
.lay2-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}.lay2-focus{fill:var(--accent,#6366f1);fill-opacity:.14;stroke:var(--accent,#6366f1);stroke-width:2}.lay2-text{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.lay2-small{font:500 12px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}.lay2-path{fill:none;stroke:var(--border,#d1d5db);stroke-width:3}.lay2-packet{fill:var(--accent,#6366f1);stroke:var(--text-primary,#1f2937);stroke-width:1.5}.lay2-tag{font:700 11px ui-monospace,SFMono-Regular,monospace;fill:var(--background,#fff);text-anchor:middle}.lay2-h1,.lay2-h2,.lay2-h3,.lay2-h4{opacity:0}.lay2-packet{offset-path:path('M220 108 L220 418 L700 418 L700 108');offset-rotate:0deg;animation:lay2-travel 12s ease-in-out infinite}.lay2-h1{animation:lay2-wrap1 12s ease-in-out infinite}.lay2-h2{animation:lay2-wrap2 12s ease-in-out infinite}.lay2-h3{animation:lay2-wrap3 12s ease-in-out infinite}.lay2-h4{animation:lay2-wrap4 12s ease-in-out infinite}@keyframes lay2-travel{0%,4%{offset-distance:0%;opacity:0}8%{offset-distance:0%;opacity:1}44%,56%{offset-distance:50%;opacity:1}92%{offset-distance:100%;opacity:1}96%,100%{offset-distance:100%;opacity:0}}@keyframes lay2-wrap1{0%,10%,90%,100%{opacity:0}16%,84%{opacity:1}}@keyframes lay2-wrap2{0%,18%,82%,100%{opacity:0}24%,76%{opacity:1}}@keyframes lay2-wrap3{0%,26%,74%,100%{opacity:0}32%,68%{opacity:1}}@keyframes lay2-wrap4{0%,34%,66%,100%{opacity:0}40%,60%{opacity:1}}@media (prefers-reduced-motion:reduce){.lay2-packet{animation:none;offset-distance:50%;opacity:1}.lay2-h1,.lay2-h2,.lay2-h3,.lay2-h4{animation:none;opacity:1}}
</style>
<text class="lay2-text" x="220" y="34">client: encapsulate downward</text>
<text class="lay2-text" x="700" y="34">server: decapsulate upward</text>
<path class="lay2-path" d="M220 108 L220 418 L700 418 L700 108"/>
<g><rect class="lay2-box" x="80" y="70" width="280" height="58" rx="9"/><text class="lay2-text" x="220" y="96">HTTP payload</text><text class="lay2-small" x="220" y="116">application meaning</text><rect class="lay2-box" x="80" y="140" width="280" height="58" rx="9"/><text class="lay2-text" x="220" y="166">TLS record</text><text class="lay2-small" x="220" y="186">security context</text><rect class="lay2-box" x="80" y="210" width="280" height="58" rx="9"/><text class="lay2-text" x="220" y="236">TCP segment</text><text class="lay2-small" x="220" y="256">transport context</text><rect class="lay2-box" x="80" y="280" width="280" height="58" rx="9"/><text class="lay2-text" x="220" y="306">IP packet</text><text class="lay2-small" x="220" y="326">routing context</text><rect class="lay2-box" x="80" y="350" width="280" height="58" rx="9"/><text class="lay2-text" x="220" y="376">Ethernet frame</text><text class="lay2-small" x="220" y="396">local-link context</text></g>
<g><rect class="lay2-box" x="560" y="70" width="280" height="58" rx="9"/><text class="lay2-text" x="700" y="96">HTTP payload</text><text class="lay2-small" x="700" y="116">delivered to handler</text><rect class="lay2-box" x="560" y="140" width="280" height="58" rx="9"/><text class="lay2-text" x="700" y="166">TLS record removed</text><text class="lay2-small" x="700" y="186">authenticated and decrypted</text><rect class="lay2-box" x="560" y="210" width="280" height="58" rx="9"/><text class="lay2-text" x="700" y="236">TCP header removed</text><text class="lay2-small" x="700" y="256">ordered socket bytes</text><rect class="lay2-box" x="560" y="280" width="280" height="58" rx="9"/><text class="lay2-text" x="700" y="306">IP header removed</text><text class="lay2-small" x="700" y="326">local delivery</text><rect class="lay2-box" x="560" y="350" width="280" height="58" rx="9"/><text class="lay2-text" x="700" y="376">Ethernet header removed</text><text class="lay2-small" x="700" y="396">frame accepted</text></g>
<rect class="lay2-focus" x="370" y="382" width="180" height="72" rx="12"/><text class="lay2-text" x="460" y="410">link transit</text><text class="lay2-small" x="460" y="434">fully wrapped frame</text>
<g class="lay2-packet"><rect x="-48" y="-22" width="96" height="44" rx="8"/><text class="lay2-tag" x="0" y="4">payload</text><rect class="lay2-h1" x="-58" y="-30" width="20" height="60" rx="3"/><rect class="lay2-h2" x="-68" y="-34" width="18" height="68" rx="3"/><rect class="lay2-h3" x="-78" y="-38" width="18" height="76" rx="3"/><rect class="lay2-h4" x="-88" y="-42" width="18" height="84" rx="3"/></g>
</svg>
<figcaption>Motion shows the payload gaining HTTP, TLS, TCP, IP, and Ethernet context on the outbound path, then losing those wrappers in reverse order at the receiving endpoint.</figcaption>
</figure>

An HTTP parser cannot skip directly to the Ethernet frame because it does not own the NIC receive path. A router does not decrypt TLS because it generally lacks the connection keys and does not terminate that security contract. A kernel cannot invent an HTTP retry policy from a TCP retransmission because byte-stream delivery has no request boundary.

This separation creates useful independence. HTTP can evolve while routers continue forwarding IP. Ethernet can be replaced by another link technology without changing an application's method semantics. TCP can retransmit a lost segment without asking the HTTP handler to regenerate the missing bytes.

It also creates information loss. Once a proxy terminates client TCP and opens a new backend TCP connection, the backend capture no longer contains the client's original transport sequence space or congestion history. Once a router removes an incoming link header, the next hop does not receive the original source MAC address. Once TLS decrypts a record, an application log can record the request but cannot reconstruct which encrypted record boundaries or packets carried it unless separate evidence was retained.

Three rules follow:

1. **Name the observation boundary.** "The packet had a 1,500-byte length" is incomplete without interface and capture point.
2. **Do not infer unavailable semantics.** A router forwarding TCP port 443 does not thereby know an HTTP route.
3. **Correlate across termination points.** Use timestamps, request IDs, connection tuples, and proxy logs carefully, knowing that each intermediary can create a new connection and clock domain.

That is also why end-to-end tracing and packet evidence complement each other. Tracing explains logical work after an instrumented component has accepted a request. Packet evidence explains whether bytes reached that component and what the transport did. [Observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) owns the broader telemetry architecture. Here the network-specific rule is narrower: never ask one evidence source to reveal fields that were opaque or already discarded at its observation point.

## What each layer can know, and what it must not pretend to know

The visibility matrix is the fastest antidote to vague incident claims. It states what a component can inspect directly, what it can change, and what remains opaque without terminating another contract.

![Visibility matrix for link, IP, transport, TLS, and HTTP layers](/imgs/blogs/the-layers-are-a-lie-but-a-useful-one-3.webp)

### Link: one local delivery domain

An Ethernet receiver can inspect destination and source MAC addresses, EtherType, VLAN information when present, and frame integrity. A switch can learn source locations and select an output port. A router receiving Ethernet can remove that frame and build another for the next hop.

The link layer cannot determine that an encrypted payload is an HTTP `POST /checkout` from link fields alone. It also cannot promise end-to-end reachability. A local frame can be delivered perfectly to the default gateway while every later route to the destination is absent.

Useful questions at this boundary are concrete: Did neighbor resolution complete? Did the NIC record receive errors? Did the bridge learn the expected MAC? Did the capture see the frame on the intended interface?

### IP: destination and forwarding, not a conversation

IP can inspect addresses, the next-protocol field, TTL or Hop Limit, fragmentation-related fields, and other IP metadata. A router chooses a next hop, decrements lifetime, and may emit ICMP. [RFC 1122](https://www.rfc-editor.org/rfc/rfc1122.html) is blunt about the contract: IP is a connectionless internetwork service with no end-to-end delivery guarantee.

IP does not know whether two packets belong to one HTTP request. It does not provide ordered delivery. It cannot tell whether a missing packet was harmless duplicate data, a TLS handshake fragment, or the last bytes needed to finish a response.

Useful questions are: Which route did the kernel select? Where does TTL expire? Did ICMP report an unreachable destination or packet too big? Is source-address selection correct?

### TCP and UDP: endpoint multiplexing with different promises

TCP sees ports, sequence and acknowledgement state, flags, windows, and options. It can retransmit, reorder received bytes, pace or limit sending, and apply flow and congestion control. It gives an application an ordered byte stream, not packets and not HTTP messages.

UDP sees ports, a datagram length, and checksum state. It preserves datagram boundaries but does not retransmit or order them. The application or a protocol above UDP must decide what loss, duplication, reordering, and congestion response mean.

Neither TCP nor UDP can read TLS-protected HTTP plaintext. TCP can show that bytes were acknowledged. That acknowledgement does not prove the application parsed them, committed a transaction, or returned success. It proves the remote TCP endpoint accepted the bytes into its transport receive path.

Useful questions are: Did a socket exist? Did the handshake complete? Are retransmission counters rising? Is the receiver window zero? For UDP, did the destination socket receive the datagram, and does the application protocol expose its own acknowledgement?

### TLS: identity and protected content at the termination point

Before termination, an observer can see IP addresses, ports, packet sizes, timing, and parts of handshake metadata that the negotiated protocol leaves visible. It cannot read protected application data merely because it sees the records.

At a legitimate endpoint, TLS can authenticate the peer according to configuration, derive keys, authenticate records, decrypt content, and emit alerts. Once a proxy terminates TLS, it may inspect HTTP and establish a separate protected connection downstream. That creates two security and transport contracts, not one continuous connection.

TLS cannot report a remote switch queue or identify the physical link that dropped a packet. A handshake timeout can be caused by packet loss, routing, an overloaded endpoint, a certificate-validation dependency, or a policy mismatch. The alert, capture, and endpoint logs separate those causes.

### HTTP: meaning without the path that carried it

HTTP endpoints understand methods, target paths, header fields, status codes, content framing, and protocol-specific stream behavior. They can apply authentication policy, routing, caching, retries, and application semantics.

HTTP does not directly observe Ethernet loss or the exact router where an IP packet disappeared. A server log has no entry for a request that never completed TLS or never reached the listener. Absence from an access log is therefore not proof that the client never sent anything. It is proof only that the logging point did not record a completed event under its configured rules.

This boundary matters for API design. [HTTP for API designers](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers) owns method and status semantics. The network layer supplies the delivery substrate and failure evidence. Mixing them leads to retries on non-idempotent operations, treating transport timeout as a server rejection, or interpreting an HTTP error as packet loss.

## Why "network problem" is nearly always a layering mismatch

A layering mismatch occurs when a symptom is described at one contract but investigated with evidence from another that cannot distinguish the competing causes. The network may still be involved. The mistake is treating that involvement as a diagnosis.

Consider six superficially similar client reports:

| User-visible report | First discriminating observation | Competing causes it separates |
| --- | --- | --- |
| "Host not found" | `dig +tries=1 +time=2 name A` | no DNS answer versus later connection failure |
| "Connection timed out" | SYN and SYN-ACK in a targeted capture | routing or filtering versus listener acceptance |
| "Connection refused" | RST after SYN plus listener state | reachable host with no accepting listener versus silent drop |
| "TLS failed" | TLS alert and certificate verification output | policy or identity failure versus transport stall |
| "Request timed out" | `curl -w` phase timings plus server trace | connect or handshake time versus first-byte or body time |
| "Small request works, upload hangs" | packet sizes, retransmissions, and ICMP too-big evidence | MTU or PMTUD failure versus generic application slowness |

The phrase "the network is up" is equally weak. An interface can be administratively up while neighbor resolution fails. A route can exist while a firewall drops the flow. TCP can connect while PMTU failure blocks larger packets. TLS can complete while the HTTP handler is unhealthy. HTTP can return `200` while a downstream dependency corrupts the result.

Replace binary health with contract-specific questions:

- **Resolution:** Did the client obtain the intended address from the intended resolver, and was the answer current enough for the operation?
- **Local link:** Did the host resolve the next-hop neighbor and emit the frame on the expected interface?
- **Routing:** Which source address and next hop did the kernel choose? Did the return path work?
- **Transport:** Did the handshake complete? Did bytes advance? Were they retransmitted or flow-controlled?
- **Security:** Did the peers agree on protocol and identity? Was an alert sent?
- **Application:** Was a request parsed? Which status and trace identify its handling?

The ordering is not a rigid checklist. If an HTTP `503` is already captured, we know resolution, some route, transport, TLS, and HTTP parsing worked for that exchange. Start at the first failed or slow contract supported by evidence. Do not restart from layer one out of ritual.

### Negative evidence has a scope

"There is nothing in the log" sounds decisive, but absence is meaningful only after we define what event creates the log entry. An HTTP access log commonly records after a request has been parsed, and some configurations write only after the response completes. A request that stalls during TLS, is rejected before HTTP parsing, or reaches a different replica can leave no entry in the file being inspected.

A packet capture has the same limitation. No SYN on a server interface proves only that the capture hook did not observe a matching SYN during the capture window. The client may have selected another address, another address family, another interface, or another destination because of proxy configuration. The packet may have been dropped earlier. The capture filter may be wrong. The server clock and client clock may not align.

Make negative evidence testable by stating its boundary:

- "No packet matching destination `203.0.113.10:443` appeared on `eth0` during the client's single bounded attempt" is useful.
- "The server got nothing" is not.
- "No completed HTTP access entry contains request ID `abc` on any of the three selected replicas during the trace interval" is useful.
- "The app did not see it" is not.

Positive evidence also has limits. A SYN-ACK proves that something accepted the transport opening at the observed address. It does not prove the intended process owns the listener. A TLS certificate proves only what the verifier and policy actually checked. An HTTP `200` proves that an endpoint generated that status. It does not prove the response body is semantically correct.

This discipline prevents premature closure. Each observation should remove some hypotheses and leave others explicit. The investigation ends when one causal chain explains the symptom and all discriminating evidence, not when one dashboard turns green.

### Time is another boundary

Layered observations frequently come from different clocks. A packet timestamp can use a host clock or adapter clock. A proxy duration can use a monotonic clock even when its log timestamp uses wall time. A distributed trace can contain spans from hosts with residual clock skew. Ordering events from raw wall-clock strings can therefore manufacture impossible causality.

Prefer elapsed durations inside one component when locating a slow contract. Use synchronized wall time to correlate components, but allow for uncertainty. A packet leaving a client and a proxy log entry on another machine need a correlation window, not an assumption of nanosecond alignment. When exact order matters, carry an application request ID and combine it with connection tuple, stream identifier, and local monotonic durations.

Layer boundaries are evidence boundaries. Clock boundaries deserve the same explicit treatment as TLS termination and proxy connection splitting.

### Error ownership and causal ownership are different

The component that reports an error may not have caused it. A client resolver reports a timeout because authoritative service was unreachable. TCP reports a timeout because a tunnel discarded packets too large for its inner budget. HTTP returns `502` because its upstream TLS handshake failed. These reporters are valuable witnesses, not automatically root causes.

A clean incident statement has four parts:

1. **Symptom:** what the user or caller observed.
2. **First broken contract:** the earliest expected event missing from evidence.
3. **Mechanism:** the state transition or resource limit that prevented it.
4. **Trigger:** the change or condition that activated the mechanism.

For example: "Uploads above roughly 1.2 KiB stalled" is the symptom. "Large packets failed after TCP connected" locates the contract boundary. "ICMP Packet Too Big was filtered, so the sender retained an excessive PMTU" is the mechanism. "A new tunnel added outer headers without lowering the inner MTU" is the trigger.

That sentence is actionable. "The VPN broke networking" is not.

## MTU is the first leak in the abstraction

Maximum transmission unit is the largest network-layer packet a link can carry in one piece under that link's contract. Applications do not usually manipulate link frames, yet MTU reaches upward into transport segment sizing, TLS record packetization, HTTP body progress, and user-visible timeouts. It is the first clean demonstration that layers isolate responsibilities without erasing constraints.

![A direct 1500-byte path compared with a tunnel whose illustrative 50-byte wrapper reduces the inner budget to 1450 bytes](/imgs/blogs/the-layers-are-a-lie-but-a-useful-one-4.webp)

Suppose a physical path admits 1,500-byte IP packets. A direct inner packet of 1,500 bytes fits. Now insert a tunnel whose outer headers consume 50 bytes in this illustrative scenario. The outer packet size becomes:

$$
\text{outer packet} = \text{inner packet} + \text{tunnel overhead}
$$

$$
1550 = 1500 + 50\ \text{bytes}
$$

That does not fit a 1,500-byte outer limit. The available inner budget is instead:

$$
\text{inner MTU} = 1500 - 50 = 1450\ \text{bytes}
$$

The 50-byte wrapper is an explicit example, not a claim about every tunnel. Real overhead depends on outer IP version, tunnel protocol, encryption, options, and implementation. The invariant is subtraction: lower-layer capacity must cover every outer header added below the sender's original packetization decision.

### How the stack should adapt

For IPv4, classic Path MTU Discovery from [RFC 1191, published in November 1990](https://www.rfc-editor.org/rfc/rfc1191.html) sends packets with Don't Fragment set. A router that cannot forward an oversized packet discards it and returns ICMP Destination Unreachable, Fragmentation Needed, including the constraining next-hop MTU in compliant behavior. The source lowers its PMTU estimate.

For IPv6, routers do not fragment forwarded packets. [RFC 8201, published in July 2017](https://www.rfc-editor.org/rfc/rfc8201.html) specifies that the constraining node discards an oversized packet and returns ICMPv6 Packet Too Big. The source then lowers its PMTU estimate and sends smaller packets or notifies the packetization layer to do so. [RFC 8200, also published in July 2017](https://www.rfc-editor.org/rfc/rfc8200.html) sets the IPv6 minimum link MTU at 1,280 octets.

TCP translates the packet budget into an effective maximum segment size. With minimum IPv4 and TCP headers on a 1,450-byte inner path:

$$
\text{TCP data allowance} = 1450 - 20 - 20 = 1410\ \text{bytes}
$$

With minimum IPv6 and TCP headers:

$$
\text{TCP data allowance} = 1450 - 40 - 20 = 1390\ \text{bytes}
$$

Those are upper bounds under minimum-header assumptions. TCP options and IP extension headers reduce the actual data in an individual packet. TLS and HTTP then divide that transport data budget further.

### The black-hole symptom crosses every layer

Path MTU Discovery depends on feedback or successful probing. If a middlebox drops the oversized packet and the relevant ICMP feedback is also blocked, the source can keep sending packets that never fit. [RFC 2923, published in September 2000](https://www.rfc-editor.org/rfc/rfc2923.html) describes the classic PMTUD black hole: small control traffic can work while bulk transfer fails when the first large packet disappears.

That produces a deceptive sequence:

1. DNS works because the exchange is small or takes another transport path.
2. TCP's handshake works because SYN packets are small.
3. TLS may work if handshake records happen to fit the constraining path and chosen packetization.
4. A small HTTP request can work.
5. A large response or upload stalls when packets exceed the actual PMTU.

Every dashboard can look partly healthy. The link is up. A route exists. TCP connected. The certificate is valid. The handler began work. Only packet size separates success from failure.

The decisive test is not "can I ping it?" It is "what is the largest packet that progresses with fragmentation forbidden, and do I see the required too-big feedback when I exceed it?" Even that test needs care because ICMP echo treatment can differ from application traffic. Packet captures, route state, transport counters, and a protocol-native reproduction should agree before changing production MTUs.

## Read captures without being fooled by offload

Packet analysis is a layer exercise disguised as a tool exercise. `tcpdump` and Wireshark decode fields, but they cannot choose the correct observation boundary for us.

On a Linux sender with TCP segmentation offload, the kernel may hand a large buffer to the NIC. A host capture can observe that pre-segmentation buffer, making an IP packet appear larger than interface MTU. The NIC later emits compliant wire frames. On receive, generic receive offload can coalesce multiple packets before a higher capture hook sees them. The [Linux kernel segmentation-offload documentation](https://docs.kernel.org/networking/segmentation-offloads.html), checked in September 2026, describes TSO, GSO, and GRO and their segmentation or coalescing roles. Check offload state before declaring a giant frame or impossible MSS.

```bash
ethtool -k eth0 | grep -E \
  'tcp-segmentation-offload|generic-segmentation-offload|generic-receive-offload'

ip -details link show dev eth0

sudo tcpdump -ni eth0 -s 128 -c 50 \
  'host 203.0.113.10 and (tcp or icmp or icmp6)'
```

The `-n` flag prevents name resolution from adding unrelated DNS traffic and delay. The `-i` flag names the observation interface. The snap length limits captured bytes, which reduces exposure of application payload. The filter narrows collection to one peer and the relevant control protocols. Packet captures can still contain addresses, identifiers, tokens, or content, so collect the minimum necessary and protect the output.

### Separate packet length from captured length

A capture record can store fewer bytes than the original packet because of snap length. It can also omit link fields the hardware stripped. Distinguish:

- **wire length**, the original packet or frame length reported by capture metadata;
- **captured length**, the bytes retained in the capture file;
- **IP total length or IPv6 payload length**, protocol fields inside the packet;
- **TCP payload length**, derived after header lengths;
- **application content length**, defined by the application protocol, not by the packet.

If a packet says IPv4 total length 1,500 and the capture stored 128 bytes, the packet was not 128 bytes on the wire. If a TLS record spans multiple TCP segments, no single packet contains a complete record. If one TCP segment contains parts of two records, packet count does not equal record count.

### Use filters that match the contract

These `tshark` commands answer different questions:

```bash
# Network-layer size and fragmentation-related fields.
tshark -r trace.pcapng -Y 'ip' \
  -T fields -e frame.number -e ip.len -e ip.flags.df -e ip.frag_offset

# Transport progress and retransmission analysis.
tshark -r trace.pcapng -Y 'tcp' \
  -T fields -e frame.number -e tcp.stream -e tcp.seq -e tcp.len \
  -e tcp.analysis.retransmission

# ICMP feedback relevant to IPv4 PMTU discovery.
tshark -r trace.pcapng -Y 'icmp.type == 3 && icmp.code == 4' \
  -T fields -e frame.number -e ip.src -e icmp.mtu
```

Do not start with the widest possible capture and hope a display filter will make it safe later. Scope at collection time. On a busy host, use duration, count, ring-buffer, host, port, and protocol bounds appropriate to the question.

## Replace "network problem" with a discriminating observation

The decision tree below is not a ritual sequence. It is a map from a vague timeout to the first observation that splits plausible causes. If evidence already places the failure later, jump there.

![Decision tree that narrows a timeout through DNS, route, transport, TLS, HTTP timing, and MTU evidence](/imgs/blogs/the-layers-are-a-lie-but-a-useful-one-6.webp)

### Resolution branch

```bash
dig +tries=1 +time=2 service.example A
dig +tries=1 +time=2 service.example AAAA
```

Read the status, answer section, responding server, and query time. An empty or failed answer keeps the investigation at resolution. A valid answer moves the investigation to the address actually chosen by the client. It does not prove that address is reachable or correct for every region.

### Route and neighbor branch

```bash
ip route get 203.0.113.10
ip neigh show nud failed,stale,delay,probe,reachable
```

Read the selected next hop, interface, and source address from `ip route get`. Then check the next-hop neighbor on a local Ethernet-like link. A correct route with failed neighbor resolution is a different mechanism from no route. A route lookup still does not prove the return path.

### Transport branch

```bash
ss -lntp 'sport = :443'
ss -tin 'dst 203.0.113.10'
sudo tcpdump -ni eth0 -c 20 'host 203.0.113.10 and tcp port 443'
```

On the server, verify a listener in the correct network namespace. On the client, inspect socket state, retransmissions, congestion window, and RTT fields when available. In the capture, distinguish no SYN-ACK, an explicit RST, and a completed handshake followed by stalled data.

### TLS and HTTP branch

```bash
curl --silent --show-error --output /dev/null \
  --connect-timeout 3 --max-time 10 \
  --write-out 'remote_ip=%{remote_ip}\nconnect=%{time_connect}\nappconnect=%{time_appconnect}\nstarttransfer=%{time_starttransfer}\ntotal=%{time_total}\ncode=%{http_code}\n' \
  https://service.example/health
```

`time_connect` ends after the transport connection. `time_appconnect` includes the TLS handshake for HTTPS. `time_starttransfer` reaches the first response byte after request transfer and server work. `total` includes response transfer. These fields locate elapsed time, but they do not assign causality without packet, endpoint, or trace evidence.

If connect is fast and appconnect is slow, inspect TLS and transport. If both are fast and start-transfer dominates, inspect proxy and server work. If first byte is fast but total is slow, inspect body size, flow control, congestion, loss, and receiver consumption.

### MTU branch

The strongest clue is size dependence: small exchanges succeed while large ones repeatedly stall at a stable packet size. Look for retransmitted large packets and ICMP too-big feedback. On a Linux lab interface, use `ping -M do` to forbid IPv4 fragmentation for a controlled packet-size test. On production paths, avoid mutating interfaces and remember that echo traffic can be treated differently from the service protocol.

## Case study: a routing failure looked like DNS and application failure

On October 4, 2021, Facebook, Instagram, WhatsApp, and related services became unreachable. The public symptom looked broad enough to invite every possible label: DNS failure, website outage, backbone failure, authentication failure, or operational lockout.

![Causal graph of Meta's October 4, 2021 backbone disconnection, DNS BGP withdrawal, and user-visible application unreachability](/imgs/blogs/the-layers-are-a-lie-but-a-useful-one-5.webp)

Meta's direct account, [published October 5, 2021](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/), provides the causal chain. During maintenance, a command intended to assess global backbone capacity unintentionally disconnected the data centers from one another. Auditing should have stopped the command, but a bug in the audit tool allowed it to execute. The lost backbone connectivity then affected the facilities that answer DNS queries.

Those DNS facilities advertised their authoritative server IP addresses through BGP. Their health design withdrew those advertisements when the facilities could not communicate with Meta data centers. With the backbone removed, the facilities declared themselves unhealthy and withdrew the routes. Meta states that the DNS servers remained operational but became unreachable.

The evidence ledger is deliberately explicit:

| Field | Verified case detail |
| --- | --- |
| Case | Meta global service outage |
| Event date | October 4, 2021 |
| Source | Meta Engineering incident detail, published October 5, 2021 |
| Source owner | Meta |
| Mechanism | Backbone disconnection led DNS facilities to withdraw BGP advertisements, making authoritative DNS unreachable |
| Verified numbers | No duration, traffic, or user-count number is required for this mechanism and none is asserted here |
| Transfer lesson | A higher-layer symptom can be generated by a lower-layer reachability failure plus health automation that amplifies blast radius |

### Place each fact at its contract

The initiating maintenance command was a control action, not a packet property. The resulting backbone disconnection was a network reachability failure. BGP withdrawal removed routes to authoritative DNS server addresses. Recursive resolvers and clients then could not obtain useful service addresses. Applications became unfindable even though the named DNS server processes were still running.

Calling this only a "DNS outage" describes where many users observed failure but hides the reachability mechanism. Calling it only a "BGP outage" hides the trigger and the health automation that converted internal disconnection into public DNS unreachability. Calling it only an "application outage" says almost nothing about the first broken contract.

The precise incident statement is stronger: an erroneous maintenance action disconnected the backbone; DNS health logic responded by withdrawing BGP advertisements; authoritative DNS became unreachable; users could no longer resolve services. Each semicolon crosses a distinct contract boundary.

### Trigger, contributing condition, and multiplier

Separating these avoids shallow remediation:

- **Trigger:** the maintenance command disconnected backbone links.
- **Contributing condition:** an audit-tool bug failed to block a command that should have been rejected.
- **Blast-radius multiplier:** DNS reachability depended on health checks to the disconnected data centers, so facilities withdrew their public advertisements.
- **Recovery complication:** the outage also affected tools and physical access patterns needed by responders, according to Meta's account.
- **User symptom:** names and services were unreachable.

If remediation focuses only on the DNS process, it misses the backbone trigger and safety check. If it focuses only on BGP, it misses why health automation withdrew routes. If it focuses only on the command, it misses why one control action reached so much of the system.

The transferable control is to model dependencies across layers during failure, especially dependencies used by the recovery path. Health withdrawal can prevent traffic from reaching a broken service, which is useful. It can also remove the very reachability needed to diagnose or restore it. Test the failure graph, not just each component's steady-state health rule.

## Design rules that survive layer leaks

Layering is still one of our best complexity tools. The goal is not to abolish it. The goal is to design interfaces that expose the lower-layer facts an upper layer must react to without making the upper layer reimplement the lower one.

### Make limits explicit

An interface MTU, a tunnel overhead, a proxy header limit, a TLS record limit, and an HTTP body limit are different quantities. Name them separately. Derive effective payload from all relevant envelopes. Avoid a global constant named `MAX_PACKET_SIZE` unless its precise layer and observation point are encoded in the name and documentation.

For an encapsulated path, keep the arithmetic beside the configuration:

```yaml
physical_mtu_bytes: 1500
outer_tunnel_overhead_bytes: 50
inner_mtu_bytes: 1450
tcp_ipv4_min_header_bytes: 20
ipv4_min_header_bytes: 20
derived_tcp_data_upper_bound_bytes: 1410
```

The 50-byte value remains illustrative. In a real deployment, generate or verify it from the exact tunnel mode, address family, encryption, and options. Then confirm with packet evidence.

### Preserve feedback

Dropping all ICMP as a generic security measure breaks legitimate control feedback. Filter by type, code, direction, state, rate, and policy appropriate to the environment. For PMTU, the sender needs trustworthy information that a packet did not fit, or it needs packetization-layer probing that can discover a working size without relying only on ICMP.

This does not mean accepting every unauthenticated control message blindly. RFC 8201 discusses validation and limits for Packet Too Big processing. The design objective is controlled feedback, not no feedback.

### Instrument termination points

Every proxy, tunnel, NAT, TLS terminator, and service boundary can destroy or transform correlation fields. Record enough information to connect the two sides without pretending they are the same connection. Depending on policy and privacy constraints, that can include timestamps, local and remote tuples, chosen upstream, protocol version, TLS alert, HTTP request ID, response status, byte counts, and termination reason.

Do not log secrets merely to make debugging convenient. Do not collect packet bodies when header metadata answers the question. Make retention, access, and redaction part of the design.

### Set timeouts from the contract you are waiting on

A DNS timeout, connect timeout, TLS handshake timeout, response-header timeout, idle body timeout, and total request deadline protect different waits. One undifferentiated 30-second timeout turns all failures into the same symptom and delays recovery.

Retries also belong above a failure classification. A TCP retransmission repairs missing bytes inside one connection. An HTTP retry creates another application attempt and can duplicate effects. The broader service policy belongs with [timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right). The network-specific rule is to record which contract expired before retry policy erases the distinction.

### Test size, direction, and address family

"Connectivity passed" is incomplete if the test used one small IPv4 request in one direction. Layer leaks often depend on:

- packet size;
- request versus response direction;
- IPv4 versus IPv6;
- warm versus cold DNS and connection state;
- direct versus tunneled path;
- one region or source network;
- reused versus new TLS connection;
- small header-only response versus a large body.

A minimal canary set should vary the dimensions relevant to the mechanism. Do not multiply tests blindly. Choose cases that split plausible failure modes.

## Run it yourself

This lab proves one narrow claim: the interface MTU is a hard packet budget, and lowering it changes the largest IPv4 echo payload that can be sent with fragmentation forbidden. It uses the series' canonical Linux namespaces `c` and `s` and their veth pair `c0` and `s0`.

### Question

Does reducing the client interface MTU from 1,500 to 1,300 bytes change a 1,472-byte IPv4 ICMP payload from success to a local "message too long" failure, while a 1,272-byte payload still succeeds?

The payload sizes come from the minimum IPv4 plus ICMP headers used by ordinary echo: ${20 + 8 = 28}$ bytes. Therefore ${1500 - 28 = 1472}$ and ${1300 - 28 = 1272}$.

### Preconditions

Run this only in a disposable Linux environment with root privileges. Namespace and interface changes require `CAP_NET_ADMIN`; packet capture generally requires root or capture capabilities. macOS users should run the Linux lab inside Lima, Colima, or another privileged Linux VM. The introductory [curl path and netlab setup](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) owns the reusable environment.

The commands below create only the canonical `c` and `s` namespaces. They refuse to continue if either name already exists, which avoids deleting unrelated state. `iproute2` and `iputils-ping` are required.

```bash
set -euo pipefail

command -v ip >/dev/null
command -v ping >/dev/null
test "$(id -u)" -eq 0

if ip netns list | awk '{print $1}' | grep -qxE 'c|s'; then
  echo 'namespace c or s already exists; inspect it before continuing' >&2
  exit 1
fi

ip netns add c
ip netns add s
ip link add c0 type veth peer name s0
ip link set c0 netns c
ip link set s0 netns s
ip -n c address add 10.77.0.1/30 dev c0
ip -n s address add 10.77.0.2/30 dev s0
ip -n c link set lo up
ip -n s link set lo up
ip -n c link set c0 up mtu 1500
ip -n s link set s0 up mtu 1500

ip -n c -details link show dev c0
ip -n s -details link show dev s0
ip -n c route get 10.77.0.2
```

Read: in each link line, find `mtu 1500`; in the route output, find `10.77.0.2 dev c0 src 10.77.0.1`.

Expected: both interfaces are `UP` with MTU 1,500. The route is directly connected through `c0`. Interface indices and queue metadata vary by kernel and environment.

### Baseline

```bash
ip netns exec c ping -4 -c 3 -W 1 -M do -s 1472 10.77.0.2
```

Read: the summary line for packets transmitted, received, and packet loss. `-M do` sets IPv4 Don't Fragment behavior for the probe, and `-s 1472` requests 1,472 ICMP data bytes. With 20 bytes of minimum IPv4 header and 8 bytes of ICMP header, the IP packet totals 1,500 bytes.

Expected: all three replies normally arrive with 0% loss in an idle local namespace lab. RTT is commonly below 1 ms on a native Linux host and can be several milliseconds in a busy or nested VM. The correctness test is successful delivery, not a specific latency.

### Apply one change

Lower only the client interface MTU:

```bash
ip -n c link set dev c0 mtu 1300
ip -n c -details link show dev c0
```

Read: confirm `mtu 1300` on `c0`. The server remains at MTU 1,500. This treatment tests the sender's local packet budget; it is not a complete routed PMTU-discovery experiment.

### Compare

```bash
set +e
ip netns exec c ping -4 -c 1 -W 1 -M do -s 1472 10.77.0.2
large_status=$?
set -e

ip netns exec c ping -4 -c 3 -W 1 -M do -s 1272 10.77.0.2
printf 'oversized_probe_exit=%s\n' "$large_status"
```

Read: the oversized probe should report a local error such as `message too long, mtu=1300` and exit nonzero. The 1,272-byte payload should receive three replies. Its total IPv4 packet size is ${1272 + 8 + 20 = 1300}$ bytes.

Expected: the 1,472-byte probe fails before a compliant 1,500-byte packet is emitted through `c0`; wording varies by `iputils` version. The 1,272-byte probe normally shows 0% loss. If it does not, confirm the two interface states, addresses, and direct route before interpreting MTU behavior.

To observe that the oversized local send produces no matching echo request while the fitting probe does, run this bounded capture in another terminal during the comparison:

```bash
ip netns exec s timeout 10 \
  tcpdump -ni s0 -c 6 'icmp and host 10.77.0.1'
```

Read: echo requests and replies for the fitting probes. Expected: no 1,500-byte IP echo request from the oversized treatment reaches `s0`; fitting 1,300-byte IP echo traffic does. Capture text varies by tcpdump version. Captures can contain sensitive payloads in real environments, so keep filters and duration narrow.

### Reset

Restore MTU before removing only the two lab namespaces:

```bash
ip -n c link set dev c0 mtu 1500
ip netns delete c
ip netns delete s
```

Verify cleanup:

```bash
if ip netns list | awk '{print $1}' | grep -qxE 'c|s'; then
  echo 'cleanup incomplete' >&2
  exit 1
fi
```

The treatment changed no production interface and no global sysctl. Its result is intentionally local: a lower-layer byte limit directly changes which higher-layer payload can be emitted. A routed tunnel lab would add an intermediate namespace, explicit outer headers, and ICMP feedback to test full PMTU discovery.

## Key takeaways

- The seven-layer OSI model is a coordinate system. The deployed service stack is better reasoned about as link, IP, TCP or UDP, TLS, and HTTP contracts with explicit termination points.
- Encapsulation is byte accounting. Under the stated minimum-header model, a 1,500-byte IPv4 path leaves 1,460 bytes after IPv4 and TCP, then 1,438 bytes after a minimal TLS 1.3 AES-128-GCM record envelope.
- Every observation belongs to a boundary. Host captures, wire captures, proxy logs, TLS logs, and HTTP traces expose different fields and can all be correct.
- A layer can act only on fields it owns or legitimately terminates. Do not ask a switch for HTTP meaning or an application log for a remote loss hop.
- "Network problem" is a symptom bucket. Replace it with the first broken contract, the discriminating measurement, the mechanism, and the trigger.
- MTU is the first obvious contract leak. Tunnel overhead shrinks the inner packet budget, and broken PMTU feedback can let small exchanges succeed while large transfers stall.
- Layering remains useful when limits, feedback, observation points, and termination boundaries are explicit.

## Further reading

- [RFC 1122: Requirements for Internet Hosts, Communication Layers](https://www.rfc-editor.org/rfc/rfc1122.html), October 1989.
- [RFC 9293: Transmission Control Protocol](https://www.rfc-editor.org/rfc/rfc9293.html), August 2022.
- [RFC 8446: TLS 1.3](https://www.rfc-editor.org/rfc/rfc8446.html), August 2018.
- [RFC 8201: IPv6 Path MTU Discovery](https://www.rfc-editor.org/rfc/rfc8201.html), July 2017.
- [RFC 2923: TCP Problems with Path MTU Discovery](https://www.rfc-editor.org/rfc/rfc2923.html), September 2000.
- [Meta Engineering: More details about the October 4 outage](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/), published October 5, 2021.
