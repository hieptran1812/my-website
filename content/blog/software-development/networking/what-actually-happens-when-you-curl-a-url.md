---
title: "What Actually Happens When You curl a URL: Every Hop and Every Millisecond"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Trace one curl request from URL parsing to the last response byte, assign each delay to an owner, and reproduce the path in a safe Linux network lab."
tags:
  [
    "networking",
    "distributed-systems",
    "curl",
    "dns",
    "tcp",
    "tls",
    "http",
    "latency",
    "linux-networking",
    "network-debugging",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 44
image: "/imgs/blogs/what-actually-happens-when-you-curl-a-url-1.webp"
---

You type `curl https://service.example/orders/42`, press Enter, and wait. The terminal gives you a body or an error. Between those two moments, the machine parses a URL, finds an address, chooses a route, opens a transport connection, authenticates a peer, writes an HTTP message, waits for application work, and drains response bytes through several queues. Calling all of that "the network" throws away the evidence we need most.

The diagram below is the mental model for this entire series: one wall clock split at observable boundaries. Its labels are deliberately symbolic because a duration is honest only when it is derived, cited, or reproduced. `curl` reports cumulative timestamps. We turn them into phase costs by subtraction, then ask which component owns the first phase that changed.

![A latency ladder separating curl name lookup, TCP, TLS, request, server wait, first byte, and body transfer](/imgs/blogs/what-actually-happens-when-you-curl-a-url-1.webp)

This post is the [introduction and index](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) to **Networking for Engineers Who Ship Services**. It gives us four reusable views: the latency ladder, the path map, the packet timeline, and the throughput-latency frontier. Later posts zoom into individual mechanisms. The [layers post](/blog/software-development/networking/the-layers-are-a-lie-but-a-useful-one) explains encapsulation, [the packet-routing post](/blog/software-development/networking/how-a-packet-gets-there-arp-switching-routing-and-ecmp) follows forwarding decisions, and [the latency-budget post](/blog/software-development/networking/the-latency-budget-speed-of-light-serialization-and-queueing) derives the physical limits. The final [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) puts every diagnostic move back together.

The promise here is narrower and more useful than "learn networking." By the end, you will be able to look at one slow `curl`, name the boundary where time accumulated, choose the next discriminating command, and reproduce the causal chain in an isolated lab.

> A total duration tells you that a request was slow. A boundary tells you who must explain it.

## One command, seven clocks

Start with a command that discards the body and prints structured timing data:

```bash
curl --silent --show-error --output /dev/null \
  --write-out '{"remote_ip":"%{remote_ip}","remote_port":%{remote_port},"http_version":"%{http_version}","code":%{response_code},"bytes":%{size_download},"dns":%{time_namelookup},"connect":%{time_connect},"tls":%{time_appconnect},"pretransfer":%{time_pretransfer},"first_byte":%{time_starttransfer},"total":%{time_total}}\n' \
  https://service.example/orders/42
```

The variable names and meanings come from the current curl man page. As of curl 8.22.0, released on September 2, 2026, `time_namelookup` ends after name resolution, `time_connect` ends after the connection to the remote host or proxy, `time_appconnect` ends after the TLS handshake, `time_starttransfer` ends when the first response byte arrives, and `time_total` ends when the operation finishes. The project documents all of these in [`--write-out`](https://curl.se/docs/manpage.html#-w), and its [release table](https://curl.se/docs/releases.html) dates the version. Check your installed build with `curl --version` because protocol support, resolver backends, and TLS libraries vary.

Those fields are not independent stopwatch readings. They all start at the beginning of the transfer. The useful quantities are deltas. For a cold HTTPS request with no redirect, use this explanatory decomposition:

$$
T_{total} = T_{dns} + T_{tcp} + T_{tls} + T_{request} + T_{server+return} + T_{body}
$$

This is an operational model, not an equation stated by an RFC. It assigns observable intervals to likely owners:

$$
\begin{aligned}
T_{dns} &= t_{namelookup} \\
T_{tcp} &= t_{connect} - t_{namelookup} \\
T_{tls} &= t_{appconnect} - t_{connect} \\
T_{pre} &= t_{pretransfer} - t_{appconnect} \\
T_{first} &= t_{starttransfer} - t_{pretransfer} \\
T_{body} &= t_{total} - t_{starttransfer}
\end{aligned}
$$

`T_first` still combines several things: request transmission, proxy and application work, backend work, and the response's return path. A client-side timer cannot separate work that happens behind the remote socket. That limitation is not a flaw. It tells us when to correlate a packet capture with an application trace rather than guessing.

| Observed delta | Boundary that moved | First useful check | What it still cannot prove |
| --- | --- | --- | --- |
| `namelookup` | resolver returned an address late | `dig`, resolver logs, cache state | why the chosen service was slow |
| `connect - namelookup` | TCP establishment or proxy connect was late | `ip route get`, `ss -tin`, SYN capture | whether the application handler is healthy |
| `appconnect - connect` | TLS negotiation was late | `openssl s_client`, TLS trace, certificate path | whether HTTP routing was correct |
| `starttransfer - pretransfer` | request-to-first-byte interval grew | packet capture plus server trace | client-side body drain time |
| `total - starttransfer` | response body completed slowly | byte count, receive window, retransmits, rate | which server computation produced the body |

`curl -v` is excellent for protocol narration, but verbose lines are not a latency profile. The curl project recommends verbose mode as a first diagnostic step in [Everything curl](https://everything.curl.dev/usingcurl/verbose/index.html). Use `-v` to see address attempts, negotiated protocol, request headers, and response headers. Use `-w` to measure boundaries. Use a packet capture when order, retransmission, or on-wire timing is the question.

### Cold, warm, and reused are different experiments

A cold request may perform DNS, TCP, and TLS work. A second URL transfer in the same curl process may reuse a connection. A second process can reuse an operating-system DNS cache but not libcurl's in-memory connection. An HTTP redirect can add another resolution and connection sequence. A proxy can make `time_connect` describe the proxy connection rather than the origin connection.

Control those variables explicitly:

```bash
# Cold process, one transfer.
curl --silent --output /dev/null --write-out '%{num_connects} %{time_total}\n' \
  https://service.example/health

# Two transfers in one process. The second may reuse the first connection.
curl --silent --output /dev/null --write-out '%{url_effective} %{num_connects} %{time_total}\n' \
  https://service.example/health \
  https://service.example/health

# Follow redirects and expose how many occurred.
curl --location --silent --output /dev/null \
  --write-out '%{num_redirects} %{time_redirect} %{time_total}\n' \
  http://service.example/health
```

If the second transfer reports `num_connects` as zero, curl reused an existing connection for that transfer. That does not mean TCP or TLS are free in general. It means this measurement priced the warm path. State the path you measured every time.

## The path is not one path

Engineers often draw a client on the left, a server on the right, and one arrow between them. That picture hides the components most likely to own the failure.

![The DNS control lookup and the separate request data path through edge, load balancers, proxies, application, and backend](/imgs/blogs/what-actually-happens-when-you-curl-a-url-2.webp)

DNS is consulted to obtain a destination. HTTP request packets do not then flow through the resolver. The solid path begins when the client sends toward an address. Depending on the service, that address may terminate at an anycast edge, a layer 4 load balancer, a layer 7 proxy, or the application host itself. A service mesh sidecar may add another local or remote hop. The application may then wait on a database, cache, queue, or another service.

Each box can produce a different symptom:

- A resolver failure prevents an address from being chosen. No SYN leaves for the service address.
- A routing or filtering failure lets resolution succeed but prevents the TCP handshake from completing.
- A layer 4 load balancer can accept a connection but choose an unhealthy transport endpoint.
- A layer 7 proxy can complete TLS yet reject a route or wait for an upstream.
- A sidecar can add connection pools, queues, retries, and policy decisions invisible to the client.
- An application can accept the request quickly and then wait on a backend.
- A backend can return the first bytes promptly but stream the rest slowly.

The map is a hypothesis generator, not proof. `remote_ip` says where curl connected. It does not reveal every forwarding device. A traceroute shows some responding routers, not necessarily a symmetric or complete service path. A response header may identify a proxy, but headers can be removed or rewritten. The correct habit is to combine observations whose blind spots differ.

### The URL selects more than a host

For `https://service.example:8443/orders/42?expand=items`, curl must interpret at least these fields:

| URL field | Example | Consequence |
| --- | --- | --- |
| scheme | `https` | select HTTPS behavior and a secure transport |
| host | `service.example` | resolve an address and set HTTP authority |
| port | `8443` | connect to a non-default service port |
| path | `/orders/42` | form the request target used by HTTP routing |
| query | `expand=items` | pass application parameters in the target |

The host has two jobs that must not be conflated. It supplies a name for address resolution, and it supplies the HTTP authority plus the TLS Server Name Indication used to select a virtual service and certificate. Overriding the destination with `--resolve` changes address selection while preserving the URL host:

```bash
curl --resolve service.example:443:203.0.113.10 \
  https://service.example/orders/42
```

`203.0.113.0/24` is documentation space, so replace it with an address you control. Do not use the example as a real target. The diagnostic value is the separation: if `--resolve` repairs a request, the resolver path or its answer is implicated. If it does not, the failure lies later or is shared by both paths.

Changing the URL to `https://203.0.113.10/` is not equivalent. It changes the TLS name and HTTP authority, which can select a different certificate and virtual host. A common debugging mistake fixes DNS by accidentally changing the application identity too.

## The packet timeline: order is the mechanism

Wall-clock phases are useful because packet order constrains what can happen next. A cold HTTPS request cannot receive an HTTP response before it has an address, a connected transport, negotiated security keys, and a delivered request.

<figure class="blog-anim">
<svg viewBox="0 0 900 860" role="img" aria-label="Cold HTTPS packet timeline from DNS resolution through TCP, TLS, HTTP request, first byte, and body transfer" style="width:100%;height:auto;max-width:896px">
<style>
.curl1-head{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.curl1-life{stroke:var(--border,#d1d5db);stroke-width:2;stroke-dasharray:7 8}.curl1-wire{stroke:var(--text-secondary,#6b7280);stroke-width:2}.curl1-control{stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:5 6}.curl1-label{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.curl1-step{font:500 13px ui-monospace,SFMono-Regular,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}.curl1-rtt{font:600 12px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280)}.curl1-packet{fill:var(--accent,#6366f1);opacity:0}.curl1-path{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5;marker-end:url(#curl1-arrow)}@keyframes curl1-ltr{0%,2%{transform:translateX(0);opacity:0}3%{opacity:1}8%{transform:translateX(500px);opacity:1}9%,100%{transform:translateX(500px);opacity:0}}@keyframes curl1-rtl{0%,2%{transform:translateX(0);opacity:0}3%{opacity:1}8%{transform:translateX(-500px);opacity:1}9%,100%{transform:translateX(-500px);opacity:0}}@keyframes curl1-dnsout{0%,2%{transform:translateX(0);opacity:0}3%{opacity:1}8%{transform:translateX(250px);opacity:1}9%,100%{transform:translateX(250px);opacity:0}}@keyframes curl1-dnsback{0%,2%{transform:translateX(0);opacity:0}3%{opacity:1}8%{transform:translateX(-250px);opacity:1}9%,100%{transform:translateX(-250px);opacity:0}}.curl1-go{animation:curl1-ltr 12s ease-in-out infinite}.curl1-back{animation:curl1-rtl 12s ease-in-out infinite}.curl1-dns-go{animation:curl1-dnsout 12s ease-in-out infinite}.curl1-dns-back{animation:curl1-dnsback 12s ease-in-out infinite}.curl1-p2{animation-delay:1.05s}.curl1-p3{animation-delay:2.10s}.curl1-p4{animation-delay:3.15s}.curl1-p5{animation-delay:4.20s}.curl1-p6{animation-delay:5.25s}.curl1-p7{animation-delay:6.30s}.curl1-p8{animation-delay:7.35s}.curl1-p9{animation-delay:8.40s}.curl1-p10{animation-delay:9.45s}.curl1-p11{animation-delay:10.50s}@media (prefers-reduced-motion:reduce){.curl1-packet{animation:none;display:none}}
</style>
<defs><marker id="curl1-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="var(--text-secondary,#6b7280)"/></marker></defs>
<rect class="curl1-head" x="110" y="24" width="180" height="62" rx="10"/><rect class="curl1-head" x="360" y="24" width="180" height="62" rx="10"/><rect class="curl1-head" x="610" y="24" width="180" height="62" rx="10"/>
<text class="curl1-label" x="200" y="61">client</text><text class="curl1-label" x="450" y="61">resolver</text><text class="curl1-label" x="700" y="61">server</text>
<line class="curl1-life" x1="200" y1="86" x2="200" y2="820"/><line class="curl1-life" x1="450" y1="86" x2="450" y2="230"/><line class="curl1-life" x1="700" y1="86" x2="700" y2="820"/>
<path class="curl1-path curl1-control" d="M210 128 H440"/><text class="curl1-step" x="325" y="119">DNS query</text><circle class="curl1-packet curl1-dns-go" cx="200" cy="128" r="8"/>
<path class="curl1-path curl1-control" d="M440 184 H210"/><text class="curl1-step" x="325" y="175">DNS answer</text><circle class="curl1-packet curl1-dns-back curl1-p2" cx="450" cy="184" r="8"/>
<path class="curl1-path" d="M210 246 H690"/><text class="curl1-step" x="450" y="237">SYN</text><circle class="curl1-packet curl1-go curl1-p3" cx="200" cy="246" r="8"/>
<path class="curl1-path" d="M690 302 H210"/><text class="curl1-step" x="450" y="293">SYN-ACK</text><circle class="curl1-packet curl1-back curl1-p4" cx="700" cy="302" r="8"/>
<path class="curl1-path" d="M210 358 H690"/><text class="curl1-step" x="450" y="349">ACK</text><text class="curl1-rtt" x="40" y="306">+1 RTT</text><circle class="curl1-packet curl1-go curl1-p5" cx="200" cy="358" r="8"/>
<path class="curl1-path" d="M210 414 H690"/><text class="curl1-step" x="450" y="405">TLS ClientHello</text><circle class="curl1-packet curl1-go curl1-p6" cx="200" cy="414" r="8"/>
<path class="curl1-path" d="M690 470 H210"/><text class="curl1-step" x="450" y="461">TLS ServerHello + Finished</text><circle class="curl1-packet curl1-back curl1-p7" cx="700" cy="470" r="8"/>
<path class="curl1-path" d="M210 526 H690"/><text class="curl1-step" x="450" y="517">TLS client Finished</text><text class="curl1-rtt" x="40" y="474">+1 RTT</text><circle class="curl1-packet curl1-go curl1-p8" cx="200" cy="526" r="8"/>
<path class="curl1-path" d="M210 598 H690"/><text class="curl1-step" x="450" y="589">HTTP GET</text><circle class="curl1-packet curl1-go curl1-p9" cx="200" cy="598" r="8"/>
<rect class="curl1-head" x="620" y="626" width="160" height="54" rx="9"/><text class="curl1-step" x="700" y="658">server work</text>
<path class="curl1-path" d="M690 716 H210"/><text class="curl1-step" x="450" y="707">first byte</text><circle class="curl1-packet curl1-back curl1-p10" cx="700" cy="716" r="8"/>
<path class="curl1-path" d="M690 772 H210"/><text class="curl1-step" x="450" y="763">body chunks</text><circle class="curl1-packet curl1-back curl1-p11" cx="700" cy="772" r="8"/><circle class="curl1-packet curl1-back curl1-p11" cx="730" cy="772" r="6"/><circle class="curl1-packet curl1-back curl1-p11" cx="760" cy="772" r="5"/>
<text class="curl1-rtt" x="40" y="650">request before response</text>
</svg>
<figcaption>The moving packet shows why a cold HTTPS request cannot deliver response bytes until DNS, TCP, TLS, request delivery, server work, and return transfer occur in causal order.</figcaption>
</figure>

The sequence above is deliberately a causal model. [RFC 9293, published August 2022](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.5), specifies TCP's three-way handshake: SYN, SYN plus ACK, then ACK. [RFC 8446, published August 2018](https://www.rfc-editor.org/rfc/rfc8446.html), specifies TLS 1.3 handshake message flows and distinguishes ordinary 1-RTT application data from weaker replay-sensitive 0-RTT data. [RFC 9110, published June 2022](https://www.rfc-editor.org/rfc/rfc9110.html#section-3.4), defines HTTP as request and response messages exchanged over a connection.

On a clean path, a simplified cold sequence looks like this:

1. The resolver returns one or more destination addresses.
2. The client chooses an address and sends a TCP SYN.
3. The server returns SYN-ACK; the client returns ACK and reaches `ESTABLISHED`.
4. The TLS client sends `ClientHello`, including the service name and supported parameters.
5. The server authenticates its identity and both peers derive traffic keys.
6. The client writes an HTTP request through the encrypted connection.
7. The server or an intermediary routes the request and performs application work.
8. Response control data and headers arrive, establishing the first-byte boundary.
9. Content bytes continue until HTTP framing says the message is complete.

Reality adds branches. The resolver can return IPv6 and IPv4 candidates. Curl can race address families. A proxy adds a tunnel or a separate HTTP exchange. TLS resumption changes handshake work. HTTP/2 multiplexes streams on one connection. HTTP/3 uses QUIC over UDP rather than TCP. None of those invalidate the method: name the actual path, identify observable boundaries, then inspect the first boundary that moved.

### Read the capture as a sequence of claims

A packet capture is not a movie of "the network." It is an observation at one capture point. Packets dropped before that point never appear. Offload can make packets look larger or checksums look invalid on a host capture. A capture on the client and one on the server can disagree about timestamps because their clocks differ. Start by stating where the capture was taken and which clock its timestamps use.

For a single HTTP lab flow, this display keeps the conversation narrow:

```bash
tshark -r delayed.pcap \
  -Y 'tcp.port == 8080' \
  -T fields \
  -e frame.time_relative \
  -e ip.src -e tcp.srcport \
  -e ip.dst -e tcp.dstport \
  -e tcp.flags.str \
  -e tcp.seq -e tcp.ack -e tcp.len
```

Read each row as a falsifiable statement. A SYN with no ACK flag says the active opener tried to create state. A SYN-ACK says something at the destination address and port returned compatible transport state. An ACK covering a byte range says the receiver reports progress through that range. A retransmitted sequence range says the sender lacked sufficient evidence of delivery and sent those bytes again. These facts are narrower than "the server is up," which is exactly why they are useful.

Absolute sequence numbers are not needed for most diagnosis. Relative sequence numbers make the first data byte easy to follow. What matters is whether sequence space advances, whether acknowledgements cover it, whether advertised windows close, and whether gaps are repaired. TCP acknowledges bytes, not application messages. One HTTP header block can span packets, and one packet can contain bytes from more than one application write.

Capture location changes interpretation. On a client host, seeing a SYN leave proves only that the local stack handed a packet toward the interface. Seeing no SYN-ACK there does not distinguish a forward-path drop, a closed service hidden by a filter, or a return-path drop. A simultaneous server-side capture can split those hypotheses: no arriving SYN implicates the forward path; an arriving SYN with a leaving SYN-ACK implicates the return path; an arriving SYN followed by RST implicates local endpoint behavior.

Encrypted traffic preserves enough metadata for many transport questions. You can observe endpoints, ports, packet sizes, direction, timing, retransmissions, and connection closure without decrypting content. You cannot infer an HTTP status code merely from a packet size. Keep protocol and transport claims at the layer the evidence supports.

### One RTT is a dependency cost, not a magic constant

Round-trip time, RTT, is the elapsed time for information to reach the peer and for a response to return. If a protocol step requires information from the peer before it can continue, that dependency costs roughly one RTT plus endpoint processing and queueing.

The word "roughly" matters. TCP's handshake is one RTT from SYN send to SYN-ACK receipt at the client. The final ACK takes another one-way trip to the server, but application data can often accompany it. TLS 1.3's ordinary handshake can make client application data available after one handshake round trip, but certificates, key exchange, loss, and implementation behavior add work. HTTP first-byte time then includes request travel, server-side work, and response travel.

Do not compute a universal HTTPS time as exactly `3 * RTT`. Treat that as an explanatory lower-order model for a specific cold path, then measure the actual cumulative boundaries. Connection reuse can remove the TCP and TLS setup from a later request. QUIC combines transport security differently. A cache can answer before an origin is consulted. Dependency graphs survive these variations better than slogans.

## What curl asks the operating system to do

`curl` does not push an Ethernet frame itself. It asks libraries and the operating system to perform work through several boundaries.

![A layered stack from curl and name service through sockets, TCP, routing, qdisc, interface, wire, and the peer](/imgs/blogs/what-actually-happens-when-you-curl-a-url-3.webp)

At the top, curl parses command-line options, chooses a protocol handler, manages connection reuse, constructs messages, and reports timings. Name resolution may run through a threaded resolver, an asynchronous resolver library, or the platform resolver. That platform path can consult `/etc/hosts`, local caches, multicast mechanisms, or DNS according to host configuration. `dig` asks DNS directly; `getent hosts` is often closer to what normal applications see through the system name service.

Once curl has an address, it asks the kernel for a socket and calls `connect`. The kernel selects a source address and ephemeral port, consults routing policy and route tables, creates transport state, and queues packets. A queueing discipline schedules packets onto a network interface. The interface and lower network deliver frames to the next hop. The remote side reverses the process and wakes a listening process.

This separation explains why one tool is never enough:

| Boundary | Read-only command | Evidence returned |
| --- | --- | --- |
| application and protocol | `curl -v -w ...` | address attempts, protocol narration, cumulative timings |
| system name service | `getent ahosts service.example` | addresses visible through the configured host lookup path |
| DNS protocol | `dig service.example A +stats` | answer, TTL, responding server, DNS query time |
| route choice | `ip route get 10.77.0.2` | selected route, source address, output interface |
| socket and TCP | `ss -tin dst 10.77.0.2` | state, RTT estimate, congestion and retransmission fields |
| queueing discipline | `tc -s qdisc show dev c0` | backlog, drops, overlimits, transmitted bytes and packets |
| packets | `tcpdump -ni c0 host 10.77.0.2` | flags, sequence, acknowledgements, and capture timestamps |

The [load-balancing deep dive](/blog/software-development/system-design/load-balancing-from-l4-to-l7) owns architecture choices across layer 4 and layer 7. Here we care about what is visible on the wire and where a timing boundary lands. Likewise, the [observability post](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) owns telemetry design. Here we use a trace only after client and packet evidence show that server-side time needs decomposition.

### Namespaces make the lab safe and legible

The recurring `netlab` environment uses two Linux network namespaces. Namespace `c` contains the client address `10.77.0.1` on interface `c0`. Namespace `s` contains the server address `10.77.0.2` on interface `s0`. A virtual Ethernet pair connects them. The `/30` prefix gives this point-to-point lab exactly the address space it needs without implying an external route.

Network namespaces isolate interfaces, routes, sockets, and queueing disciplines. That makes them the right place to mutate latency and loss. It also creates a hard safety rule: every destructive command in this series names `c`, `s`, `c0`, or `s0`. Never paste a `tc`, firewall, route-deletion, or namespace-deletion command against an unspecified production interface.

On macOS, `ip netns`, veth, and `tc netem` are not native facilities. Run the lab inside a privileged Linux VM such as Lima or Colima, with the capability needed for network administration. The experiment time does not include installing the VM, compiler, or packages.

## DNS: turn a service name into candidate addresses

The Domain Name System does not return "the server." It returns records that help a client choose an address or continue another lookup. [RFC 1035, published November 1987](https://www.rfc-editor.org/rfc/rfc1035.html), defines the DNS message sections and resource-record structure. Modern deployments add many extensions, but the diagnostic foundation remains: inspect the question, answer, authority, TTL, responding server, and elapsed query time.

Compare the application-visible path and a direct DNS query:

```bash
getent ahosts service.example
dig service.example A +noall +answer +stats
dig service.example AAAA +noall +answer +stats
```

If `dig` works while `getent` fails, do not declare DNS healthy. The application may use the system name service, which can apply search domains, `/etc/hosts`, NSS modules, local stub resolvers, and container-specific configuration. Conversely, a successful `getent` result may come from `/etc/hosts` rather than a DNS exchange.

TTL is cache eligibility, not a global stopwatch. Different resolvers can cache at different moments. Clients can impose minimums or maximums. A service can publish multiple addresses, and connection logic can choose among them. Therefore, "the DNS record changed" does not imply "every client now uses the new address."

For one curl measurement, preserve these fields:

- URL host and effective URL
- `remote_ip` and `remote_port`
- address family if it matters
- whether `--resolve`, a proxy, or a hosts-file entry was used
- resolver response and TTL when DNS is the hypothesis
- whether the run was cold or followed an earlier lookup in the same process

The later DNS track will separate recursive resolution, authoritative service, TTL rollout, negative caching, and split-horizon behavior. For now, the key boundary is simple: if `time_namelookup` moved, investigate resolution before opening a transport dashboard.

## TCP: synchronize state before sending the stream

TCP presents an ordered byte stream. The application does not see packet boundaries, and the network does not promise that one application write maps to one packet. TCP must establish synchronized state, sequence bytes, acknowledge progress, control a sending window, retransmit loss, and react to congestion.

For a new active open, the client sends SYN and enters `SYN-SENT`. The listening peer replies with SYN-ACK. The client acknowledges it and enters `ESTABLISHED`. The handshake exists partly to reject confusion from old duplicate connection attempts, a motivation described directly in RFC 9293.

The quickest on-host view is often:

```bash
ip route get 10.77.0.2
ss -tin dst 10.77.0.2
nstat -az | grep -E 'Tcp(ActiveOpens|RetransSegs|AttemptFails)'
```

`ip route get` answers the kernel's route lookup for this destination. It is more useful than staring at the whole routing table because it resolves policy and source selection for the specific address. `ss -tin` exposes per-socket transport information. `nstat` exposes host-wide counters, which need a before-and-after delta around the request to be meaningful.

Three failure shapes that users all describe as "curl timed out" are mechanically different:

| Wire observation | Socket implication | Likely next boundary |
| --- | --- | --- |
| repeated SYN, no SYN-ACK | connect cannot finish | route, filter, listener, or return path |
| immediate RST after SYN | peer actively rejected | listener absent or policy rejection |
| handshake completes, then silence | transport connected | TLS, proxy, or application phase |

Do not infer a packet drop from elapsed time alone. A retransmission in a targeted capture, or a counter delta tied to the experiment, is evidence. A timeout is policy: it says how long the caller waited, not what the network did.

The dedicated [sockets and transport contracts post](/blog/software-development/networking/sockets-and-the-two-transport-contracts-tcp-vs-udp) explains what TCP and UDP promise to an application. This introduction needs one rule from it: the socket API is where application intent becomes kernel transport state, so measure both sides of that boundary.

### Queues turn load into waiting before they turn it into loss

Packets can wait in application buffers, socket buffers, queueing disciplines, interface rings, switches, proxies, and upstream services. A queue is not automatically bad. It absorbs short bursts and lets a slower stage continue working. It becomes a latency problem when arrival rate exceeds departure rate long enough for backlog to grow.

Little's Law provides a useful explanatory model for a stable system:

$$
L = \lambda W
$$

Here $L$ is the average number of items in the system, $\lambda$ is the average arrival rate, and $W$ is the average time an item spends there. This is a general queueing identity, not a TCP-specific equation. If a service sustains ${1000}$ requests per second and requests spend an average of ${0.050}$ seconds in one stable stage, the implied average occupancy is derived as ${1000} \times {0.050} = {50}$ requests. If service rate stays fixed while arrivals rise, waiting can grow before drops or explicit errors appear.

That ordering shapes diagnosis. Rising RTT with growing qdisc backlog can precede packet loss. Rising `time_starttransfer` with stable packet travel can precede application timeouts. A dashboard that alerts only on drops detects the failure after users have already paid queueing delay.

Inspect a lab qdisc with:

```bash
sudo ip netns exec c tc -s qdisc show dev c0
sudo ip netns exec s tc -s qdisc show dev s0
```

Read backlog, dropped packets, overlimits, and transmitted bytes as a set. A nonzero backlog during load says packets are waiting at that specific qdisc. It does not reveal every queue elsewhere. A drop counter that never changes does not prove a queue is absent. The queue may be upstream, in a socket, or below the visibility of this command.

The frontier is the recurring tuning view: as offered load or queue depth rises, useful throughput can flatten while tail latency keeps climbing. A larger buffer can reduce short-term drops and worsen waiting. A tighter buffer can bound delay and discard bursts. There is no context-free best queue depth. The safe choice depends on burst tolerance, congestion control, application deadlines, and where backpressure can be applied.

## TLS: authenticate the peer and derive traffic keys

HTTPS inserts a security protocol between the connected transport and the HTTP exchange. The client sends a `ClientHello` that advertises supported parameters and usually the intended server name. The server selects parameters, proves its identity with a certificate chain and signature, and both sides derive keys. Only then does ordinary protected application traffic proceed.

Inspect a service without disabling verification:

```bash
openssl s_client \
  -connect service.example:443 \
  -servername service.example \
  -verify_return_error \
  -brief </dev/null
```

The TCP destination and `-servername` are separate on purpose. The destination chooses where packets go. The server name participates in virtual-host selection and certificate validation. If a raw IP succeeds at TCP but `openssl s_client` reports a certificate or name error, the transport path is not the primary failure.

TLS timing depends on protocol version, session state, certificate path, key exchange, peer processing, and loss. Report the negotiated version and reuse state when comparing runs. Do not claim that every TLS handshake costs one exact RTT. TLS 1.3 provides message flows that allow ordinary client application data after its main handshake dependency, while 0-RTT has weaker replay properties. That is a protocol trade-off, not a free latency switch.

The client-side delta `time_appconnect - time_connect` ends when curl considers the secure connection ready. If it grows while the TCP delta stays stable, capture the TLS handshake or use library-specific tracing. If both grow together, first determine whether the route, address family, proxy, or packet loss changed. Fixing a certificate chain will not repair missing SYN-ACK packets.

## HTTP: send intent, then wait for visible response

HTTP defines request and response semantics independently of a service's internal architecture. RFC 9110 describes a request message as control data, header fields, optional content, and optional trailers. A response has a status code, header fields, optional content, and optional trailers. HTTP/1.1, HTTP/2, and HTTP/3 encode and transport these semantics differently.

For a simple HTTP/1.1 request, verbose curl output might show the request line and fields prefixed with `>`, then response control data and fields prefixed with `<`. Treat secrets carefully. Authorization headers, cookies, query parameters, and bodies can contain credentials or personal data. Redact captures and verbose logs before sharing them.

The interval from `time_pretransfer` to `time_starttransfer` is often casually called server time. It is broader:

$$
T_{first} \approx T_{request\ wire} + T_{proxy\ queues} + T_{application} + T_{backends} + T_{response\ wire}
$$

This is an explanatory approximation. The client clock sees only the combined interval. To split it, correlate a targeted packet capture with a server trace using a request identifier. If the request bytes arrive promptly but the first response packet leaves much later, time accumulated at or behind the service. If the server sends promptly but the client receives late, the return path or client receive side is implicated.

Status codes are not a substitute for path evidence. A `503` proves that an HTTP participant generated a response. It does not by itself prove whether the origin, proxy, service mesh, or edge generated it. Response fields, trace context, and hop-specific logs can narrow ownership.

The [HTTP for API designers post](/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers) owns resource semantics and contract design. The networking series owns how those messages are framed, transported, queued, retried, and observed.

### Follow one request from keystroke to completion

We can now narrate the whole cold request without hiding a handoff.

**Command parsing.** The shell builds an argument vector and starts curl. Shell quoting determines which characters reach the program. Curl reads configuration unless options suppress it, evaluates proxy settings, parses the URL, chooses a protocol handler, and creates transfer state. None of this has placed a service packet on the wire yet. A malformed URL or local configuration error can fail before name resolution.

**Name service.** Curl asks its configured resolver path for candidate addresses. The platform can consult local policy before DNS. A DNS query, if needed, travels to a recursive resolver; that resolver may answer from cache or pursue other name servers. Curl eventually receives addresses, not a guarantee that a service is reachable at them. The `time_namelookup` boundary closes here.

**Candidate and route selection.** Curl and its resolver backend can have more than one address. Connection logic chooses or races candidates according to build and options. For each attempt, the kernel chooses a source address, route, and interface. Neighbor discovery may be required on a local link. NAT or stateful firewalls can create mappings as packets cross them. Curl's `remote_ip` records the selected peer address for the completed transfer, which is why it belongs in every evidence bundle.

**Transport establishment.** For HTTPS over TCP, `connect` initiates the three-way handshake. The client allocates socket state and an ephemeral source port. Each stateful device along the path may create its own flow entry. The listener or an upstream transport proxy accepts the connection. Curl's `time_connect` boundary closes when its connection requirement is satisfied, but that timer alone does not enumerate hidden stateful hops.

**Security establishment.** Curl's TLS backend sends and receives handshake messages, validates the peer according to trust and hostname policy, and derives traffic keys. A layer 7 proxy may terminate this connection and open or reuse a different connection upstream. The client can prove only the identity of the peer at its secure connection boundary. Curl's `time_appconnect` boundary closes when the application protocol can proceed securely.

**Request transmission.** Curl encodes the selected HTTP version, writes control data and headers, and writes a body when the method has one. The kernel copies or references bytes into socket buffers, TCP maps stream bytes to sequence space, IP supplies network-layer addressing, and the output path schedules packets. A successful local write says bytes entered the local stack. It does not prove the application at the far end consumed them.

**Intermediary and application work.** An edge can apply policy, a layer 7 proxy can choose a route, a mesh sidecar can select an upstream connection, and an application can call backends. Queues can form at each boundary. The client sees their combined effect in the request-to-first-byte interval. A distributed trace can split server-side spans, but it should be correlated with wire evidence when the disputed time sits between hosts.

**First response byte.** Response control data returns through the selected path and reaches curl. This closes `time_starttransfer`. The first byte proves that some HTTP participant responded. It does not prove that the response body will complete, that the origin generated the status, or that later chunks follow the same pacing.

**Body transfer and completion.** Curl drains response content according to HTTP framing. The sender can pause, the receiver can constrain the window, packets can be lost, and the output sink can apply backpressure. Completion means the framing rules say all expected content arrived, or the connection-close rule made the message complete. RFC 9110 notes that explicit framing helps distinguish a complete message from a connection that closed early. Curl's `time_total` boundary closes only when the operation ends.

**Cleanup or reuse.** Curl can retain an eligible connection for another transfer in the same process. The kernel retains TCP state as required by the close sequence. NATs, proxies, and load balancers can retain their own flow state on different timers. The terminal prompt returning does not mean every network device forgot the flow.

This narration gives each later post a clean entry point. The layers post owns headers and encapsulation. The routing post owns neighbor and forwarding choices. TCP posts own sequence, windows, loss recovery, and close. DNS posts own resolver behavior and control-plane staleness. TLS posts own identity and keys. HTTP posts own framing and multiplexing. Edge and proxy posts own the extra connections hidden behind one client socket. The capstone will turn the whole chain into an incident decision system.

## The last byte has a different budget from the first

Time to first byte and time to last byte answer different questions. A server can return headers quickly and then stream a large object slowly. A small response can spend nearly all of its time in fixed setup. A large response can make serialization, congestion control, receiver capacity, or application pacing dominate.

![A derived frontier showing fixed path cost dominating small bodies and serialization dominating large bodies](/imgs/blogs/what-actually-happens-when-you-curl-a-url-4.webp)

The figure uses an explicit explanatory model, not a benchmark. Let payload size be $B$ bytes and bottleneck rate be $R$ bits per second. Serialization time is:

$$
T_{serialization} = \frac{8B}{R}
$$

At $R = {20}\ \text{Mbit/s}$, using decimal megabits per second and binary payload units:

| Payload | Substitution | Serialization time | Fixed model cost | Modeled total | Source |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 KiB | `${8} \times {1024} / ({20} \times {10}^{6})` | 0.410 ms | 40 ms | 40.410 ms | Derived here from $8B/R$ |
| 100 KiB | `${8} \times {102400} / ({20} \times {10}^{6})` | 40.960 ms | 40 ms | 80.960 ms | Derived here from $8B/R$ |
| 1 MiB | `${8} \times {1048576} / ({20} \times {10}^{6})` | 419.430 ms | 40 ms | 459.430 ms | Derived here from $8B/R$ |

The 40 ms fixed cost is a scenario assumption chosen to expose the crossover. It is not an internet baseline. The arithmetic says that making the body smaller barely changes a 1 KiB request whose cost is fixed elsewhere, while doubling link rate materially changes a large transfer. Conversely, moving a service closer can matter greatly for small sequential requests even when bandwidth is abundant.

Real transfers depart from this lower-order model. Headers add bytes. TLS records and transport packets add framing. TCP congestion control may not immediately fill the bottleneck. Loss can cause retransmission and head-of-line waiting. The receiver can advertise a small window. An application can pace or pause output. Compression trades CPU for fewer bytes. The model is still valuable because it tells us which term could plausibly dominate before we reach for a knob.

Measure the body interval and effective rate together:

```bash
curl --silent --output /dev/null \
  --write-out 'bytes=%{size_download} body_s=%{time_total}-%{time_starttransfer} speed_Bps=%{speed_download}\n' \
  https://service.example/object
```

The literal subtraction is not evaluated by curl in that format. Record the two cumulative values and subtract them in a shell, `jq`, or analysis tool. Keep the raw fields so rounding does not hide a short interval. `speed_download` is an operation-level average, not an instantaneous congestion-control trace.

## Diagnose the first boundary that moved

Random dashboard hopping is expensive because most graphs cannot falsify the current hypothesis. Start from the earliest client-visible boundary that differs from a known comparable run.

![A decision tree mapping the first changed curl timing boundary to a discriminating network command](/imgs/blogs/what-actually-happens-when-you-curl-a-url-5.webp)

Use this sequence:

1. Confirm that the URL, proxy settings, address family, curl build, request method, payload, and cold or warm state match.
2. Compute phase deltas from cumulative timestamps.
3. Find the earliest delta that changed materially across repeated runs.
4. Run the command that observes that boundary directly.
5. Change one variable, then repeat the same evidence collection.

### What the clocks cannot tell you

Client timing is necessary but underdetermined. Several mechanisms can produce the same delta:

| Same observed symptom | Explanation A | Explanation B | Discriminating evidence |
| --- | --- | --- | --- |
| slow `time_namelookup` | recursive lookup walked the hierarchy | local resolver was CPU-starved | resolver logs, query trace, local scheduling evidence |
| slow connect delta | SYN packets were lost | listener accepted slowly behind a proxy | packet sequence at both boundaries |
| slow TLS delta | packet loss delayed handshake messages | certificate or key operation was slow | TLS message timing plus retransmission evidence |
| slow first-byte delta | application computation grew | request waited in a proxy queue | hop timing and application trace |
| slow body delta | bottleneck rate fell | receiver stopped advancing its window | `ss -tin` window fields and packet acknowledgements |

This is why a label such as "DNS time" should be read as "elapsed until curl's name-resolution boundary," not "CPU time consumed by a DNS server." A timing boundary localizes the next question. It rarely supplies the complete root cause.

Clock quality also matters. Curl's phase fields share one process's elapsed-time basis, which makes subtraction useful. Comparing packet timestamps across two hosts requires synchronized clocks or a method that avoids absolute cross-host time. The namespace lab uses one Linux host, so its client-side capture has a single clock. In a distributed incident, compare round trips from one capture point or document clock uncertainty.

Measurement perturbs systems too. Verbose logging adds output work. Packet capture consumes CPU and storage. DNS cache flushes change the workload. Running a command once selects an anecdote; running it forever can become load. Bound duration and sample count, preserve raw output, and write down the intervention.

### Compare like with like

A useful baseline matches the request dimensions that affect the path:

- same URL scheme, host, port, path class, and method
- same proxy and `NO_PROXY` behavior
- same address family policy
- same cold or warm connection state
- same payload size and response class
- same client region or namespace
- same curl build and TLS backend when protocol details matter

If any of these differ, record the difference rather than averaging it away. A warm HTTP/2 stream and a cold HTTPS connection are valid measurements of different questions. Combining them into one latency number destroys both.

Repeated measurements should report a distribution or, for a tiny lab, all runs plus a declared summary such as the median. Avoid selecting the fastest result as "network latency." The fastest run can be useful as a lower-bound clue, but users experience the distribution. Avoid selecting only the slowest result too, unless the investigation is explicitly about the tail and the sampling method supports that claim.

The word "materially" needs a declared rule. For the local lab later in this post, the treatment adds a configured 40 ms one-way delay to each direction. We expect a large step relative to namespace noise, so broad ranges suffice. In production, compare a distribution from equivalent traffic rather than declaring a regression from one request. The [SLO and graceful degradation post](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) owns the policy for deciding what users can tolerate. This post owns locating the wire boundary.

### A compact evidence bundle

When escalating a network symptom, send a bundle that another engineer can reason from:

```bash
date -u +'%Y-%m-%dT%H:%M:%SZ'
curl --version
env | grep -iE '^(http|https|all|no)_proxy=' || true
getent ahosts service.example
ip route get 203.0.113.10
curl --silent --show-error --output /dev/null \
  --write-out '{"ip":"%{remote_ip}","port":%{remote_port},"code":%{response_code},"dns":%{time_namelookup},"connect":%{time_connect},"tls":%{time_appconnect},"pre":%{time_pretransfer},"first":%{time_starttransfer},"total":%{time_total}}\n' \
  https://service.example/health
```

Replace documentation addresses and names with the real target. Add a bounded packet capture only when policy permits:

```bash
sudo timeout 20 tcpdump -ni any \
  'host 203.0.113.10 and (tcp port 443)' \
  -c 200 -w curl-request.pcap
```

Packet captures can contain credentials, tokens, personal data, and application payloads. Restrict the filter, packet count, duration, access, and retention. On an encrypted request, metadata is still sensitive even if content is not readable.

## Case study: Fastly on June 8, 2021

On June 8, 2021, Fastly experienced a broad global outage. In its [incident-owner summary published that day](https://www.fastly.com/blog/summary-of-june-8-outage), Fastly wrote that an undiscovered software bug was triggered by a valid customer configuration change. The company reported detecting the disruption within one minute. After identifying and isolating the cause and disabling the triggering configuration, it reported that 95 percent of its network was operating normally within 49 minutes.

![Fastly's June 8, 2021 incident mapped from a valid configuration change through a latent edge bug to isolation and recovery](/imgs/blogs/what-actually-happens-when-you-curl-a-url-6.webp)

Those are incident-owner numbers with a publication date, not a generic availability benchmark. The source does not justify inventing per-request latency, affected-request counts, or a packet-level failure mode. We should use only what it supports.

Map the event onto our path. The customer configuration was a control-plane input. A latent software bug at the edge was the vulnerable condition. Activating it impaired the data plane that served requests. Users saw failures across many unrelated sites because the shared edge sat early on their paths. Disabling the triggering configuration was a recovery action; the longer-term action was to correct the bug and improve the processes that had not exposed it.

The important diagnostic lesson is not "CDNs fail." It is that a request can fail before an origin sees it, and a valid control-plane change can alter data-plane behavior at enormous fan-out. If DNS succeeds and TCP or HTTP behavior changes at a shared edge address across many hostnames, an origin-only investigation has started too deep in the path.

Separate four ideas when reading any incident:

- **Trigger:** the valid customer configuration change.
- **Vulnerable condition:** the undiscovered software bug.
- **Blast-radius multiplier:** shared edge infrastructure serving many customers.
- **Recovery action:** isolate the cause and disable the triggering configuration.

This separation prevents shallow remedies. Blocking one customer action might remove the trigger while leaving the vulnerable condition. Restarting origins might change nothing because the request never reached them. A useful guardrail tests configuration interactions in the same execution path that serves traffic, constrains rollout scope, and preserves a rapid way to disable a harmful change.

### Evidence ledger for the case

| Field | Verified entry | Source |
| --- | --- | --- |
| Case | Fastly global service outage | Fastly incident summary |
| Event date | June 8, 2021 | Fastly incident summary |
| Source owner | Fastly | Fastly incident summary |
| Direct source | Summary of June 8 outage | [Fastly, June 8, 2021](https://www.fastly.com/blog/summary-of-june-8-outage) |
| Mechanism relevant here | valid configuration activated a latent edge software bug | Fastly incident summary |
| Verified numbers | detection within 1 minute; 95% of network operating normally within 49 minutes | [Fastly, June 8, 2021](https://www.fastly.com/blog/summary-of-june-8-outage) |
| Transfer lesson | locate the earliest shared path component and separate trigger from vulnerable condition | Derived here from the incident mechanism |

The case also marks the boundary with system design. Multi-region origin architecture cannot help a request that fails at a shared edge before origin selection. The system-design question is how much shared fate to accept. The networking question is which packets and timings prove where failure first appeared.

## The honesty rule: every number needs a provenance class

Networking prose becomes folklore when precise-looking numbers lose their conditions. This series uses three evidence classes.

**Derived numbers** show a formula, units, assumptions, substitution, and result. The 20 Mbit/s serialization table is derived. Its fixed 40 ms term is labeled as a scenario assumption.

**Cited numbers** link a primary source and include the relevant date, version, region, topology, or population. Fastly's one-minute detection and 49-minute recovery statement are cited to the incident owner and dated June 8, 2021.

**Reproducible numbers** name the lab configuration, exact command, field to inspect, and expected range. The `netlab` experiment below configures delay explicitly and tells you how scheduler and virtualization noise can shift the result.

If a number fits none of these classes, remove it or turn it into a qualitative claim. "Usually fast" is still weak, but it is less deceptive than an unexplained `20 ms` presented as fact.

This rule changes how we write tables. Any table containing reported measurements, benchmark results, limits, or prices needs a `Source` column. It changes how we write incidents. A publish date is not automatically an event date. It changes how we write labs. One run is not a universal benchmark, and a VM result is not bare-metal truth.

It also changes debugging. Capture the effective configuration before the observation:

```bash
uname -a
curl --version
ip -Version
tc -Version
ip netns exec c ip -br address
ip netns exec c ip route
ip netns exec c tc -s qdisc show dev c0
```

A result without configuration is a story. A result with configuration, a command, and a bounded expected range is an experiment someone else can challenge.

## Build the base netlab once

This first post owns the recurring lab topology. Later posts keep the names and show only their experiment's delta. The lab is Linux-only and requires root or equivalent network-administration capabilities.

Create a setup script at `netlab/setup.sh` with the following contents:

```bash
#!/usr/bin/env bash
set -euo pipefail

for command in ip tc python3 curl; do
  command -v "$command" >/dev/null || {
    printf 'missing required command: %s\n' "$command" >&2
    exit 1
  }
done

# Idempotent cleanup of only the namespaces owned by this lab.
ip netns del c 2>/dev/null || true
ip netns del s 2>/dev/null || true

ip netns add c
ip netns add s
ip link add c0 type veth peer name s0
ip link set c0 netns c
ip link set s0 netns s

ip -n c address add 10.77.0.1/30 dev c0
ip -n s address add 10.77.0.2/30 dev s0
ip -n c link set lo up
ip -n s link set lo up
ip -n c link set c0 up
ip -n s link set s0 up

ip netns exec c ip route get 10.77.0.2
ip netns exec s ip route get 10.77.0.1
ip netns exec c ping -c 3 -W 1 10.77.0.2
```

Run it from a Linux shell:

```bash
sudo bash netlab/setup.sh
```

The stable contract is now visible: namespaces `c` and `s`, interfaces `c0` and `s0`, addresses `10.77.0.1` and `10.77.0.2`, subnet `10.77.0.0/30`, HTTP port `8080`, later HTTPS port `8443`, and debug port `9090`. Later installments introduce the tiny `netserver` and `netclient` Go binaries without renaming these endpoints.

For this introductory timing experiment, Python's standard static server keeps the treatment focused on the network. Create a deterministic 64 KiB payload. The byte count is derived as ${64} \times {1024} = {65536}$ bytes:

```bash
mkdir -p netlab/www netlab/out/what-actually-happens-when-you-curl-a-url
head -c 65536 /dev/zero > netlab/www/payload.bin

sudo ip netns exec s \
  python3 -m http.server 8080 \
  --bind 10.77.0.2 \
  --directory "$PWD/netlab/www" \
  >netlab/out/what-actually-happens-when-you-curl-a-url/server.log 2>&1 &
printf '%s\n' "$!" > netlab/out/what-actually-happens-when-you-curl-a-url/server.pid
```

The server command is intentionally plain HTTP. It isolates DNS and TLS out of this first treatment so the packet and curl boundaries are easy to predict. Track D adds the stable TLS terminator at port `8443`.

## Run it yourself

### Question

If we add 40 ms of one-way delay in both directions of the isolated veth path, do the TCP-connect and first-byte boundaries grow by the round trips their dependencies require?

This is falsifiable. The route, server process, payload, and request remain fixed. The only treatment is queueing delay on `c0` and `s0`.

### Preconditions

Use Linux with `iproute2`, `tc netem`, Python 3, curl, and privileges to create network namespaces and queueing disciplines. Record versions with `uname -a`, `ip -Version`, `tc -Version`, `python3 --version`, and `curl --version`. On macOS, run these commands inside a privileged Linux VM. First run the `netlab/setup.sh` and server commands from the previous section.

Mutating commands below are scoped to namespaces `c` and `s`. Do not adapt them to a production interface without a separate change review. Read-only production equivalents include `ip route get`, `ss -tin`, and targeted counter inspection.

Preflight proves the namespace, interface, route, qdisc, process, and tool state:

```bash
set -euo pipefail
sudo ip netns list
sudo ip -n c -br address show dev c0
sudo ip -n s -br address show dev s0
sudo ip netns exec c ip route get 10.77.0.2
sudo ip netns exec s ss -lnt '( sport = :8080 )'
sudo ip netns exec c tc -s qdisc show dev c0
sudo ip netns exec s tc -s qdisc show dev s0
sudo ip netns exec c curl --version | head -n 1
```

Read: `c0` must show `10.77.0.1/30`, `s0` must show `10.77.0.2/30`, the route must select `c0`, the server must be `LISTEN` on `10.77.0.2:8080`, and neither interface should have an old `netem` treatment.

Expected: interface and route state match exactly. The qdisc name depends on distribution defaults, so the required qualitative state is "no configured netem delay."

### Baseline

Run five baseline requests and retain the raw cumulative fields:

```bash
set -euo pipefail
out=netlab/out/what-actually-happens-when-you-curl-a-url/baseline.jsonl
: >"$out"
for run in 1 2 3 4 5; do
  sudo ip netns exec c curl \
    --silent --show-error --output /dev/null \
    --write-out '{"run":'"$run"',"ip":"%{remote_ip}","bytes":%{size_download},"dns":%{time_namelookup},"connect":%{time_connect},"pre":%{time_pretransfer},"first":%{time_starttransfer},"total":%{time_total}}\n' \
    http://10.77.0.2:8080/payload.bin | tee -a "$out"
done
```

Read: `bytes` must be `65536`. Because the URL uses an IP address and plain HTTP, DNS and TLS are intentionally absent from the mechanism under test. Compare `connect`, `first`, and `total`. Compute `first - pre` and `total - first` if you load the JSON lines into another tool.

Expected: on an otherwise idle local Linux namespace pair, most baseline totals should be below 20 ms. This is a broad reproducible lab range, not a promise about every VM. A busy or nested virtualized host can exceed it. Preserve the baseline from your machine rather than deleting a run that looks inconvenient.

### Apply one change

Add 40 ms of egress delay to each end. A packet from `c` to `s` experiences the `c0` delay. A reply from `s` to `c` experiences the `s0` delay. The configured round-trip contribution is therefore derived as ${40}\ \text{ms} + {40}\ \text{ms} = {80}\ \text{ms}$.

```bash
set -euo pipefail
sudo ip netns exec c tc qdisc replace dev c0 root netem delay 40ms
sudo ip netns exec s tc qdisc replace dev s0 root netem delay 40ms
sudo ip netns exec c tc -s qdisc show dev c0
sudo ip netns exec s tc -s qdisc show dev s0
sudo ip netns exec c ping -c 5 -W 1 10.77.0.2
```

Read: both qdisc reports must name `netem` and `delay 40ms`. In `ping`, inspect each `time=` value and the final `rtt min/avg/max/mdev` line.

Expected: ping RTT should usually land between 75 and 95 ms. The center is the derived 80 ms treatment; timer granularity, scheduling, virtualization, and command overhead explain the range.

### Compare

Repeat the same curl measurement under treatment:

```bash
set -euo pipefail
out=netlab/out/what-actually-happens-when-you-curl-a-url/delayed.jsonl
: >"$out"
for run in 1 2 3 4 5; do
  sudo ip netns exec c curl \
    --silent --show-error --output /dev/null \
    --write-out '{"run":'"$run"',"ip":"%{remote_ip}","bytes":%{size_download},"dns":%{time_namelookup},"connect":%{time_connect},"pre":%{time_pretransfer},"first":%{time_starttransfer},"total":%{time_total}}\n' \
    http://10.77.0.2:8080/payload.bin | tee -a "$out"
done

sudo ip netns exec c ss -tin dst 10.77.0.2
sudo ip netns exec c tc -s qdisc show dev c0
sudo ip netns exec s tc -s qdisc show dev s0
```

Read: compare medians rather than the best single run. `connect` prices the TCP handshake seen from the client. `first` includes that handshake, request travel, server scheduling, and first-byte return. `total` includes body completion. Verify that `remote_ip` stays `10.77.0.2` and `bytes` stays `65536`, so the treatment did not change destination or payload.

Expected: `connect` should usually fall between 70 and 110 ms. `first` and `total` should usually fall between 140 and 240 ms for this small local payload. The causal estimate is about one configured RTT to complete connect, then another configured RTT for request delivery and first-byte return. Server scheduling and the fact that the body spans packets widen the range. Treat these as netlab expectations for this configuration, not internet benchmarks.

For packet-level confirmation, capture only this lab flow while issuing one request:

```bash
sudo ip netns exec c timeout 10 tcpdump -ni c0 \
  'host 10.77.0.2 and tcp port 8080' -c 80 -w \
  netlab/out/what-actually-happens-when-you-curl-a-url/delayed.pcap &
capture_pid=$!
sudo ip netns exec c curl --silent --output /dev/null \
  http://10.77.0.2:8080/payload.bin
wait "$capture_pid" || true
tcpdump -nn -tttt -r \
  netlab/out/what-actually-happens-when-you-curl-a-url/delayed.pcap
```

Read: locate SYN, SYN-ACK, the client's ACK and request bytes, then the first server response packet. Capture timestamps come from the client namespace's host clock, so no cross-host clock synchronization is required in this lab. Do not publish packet captures without reviewing their contents.

### Reset

Remove only the qdiscs added by this experiment, stop the recorded server, and optionally delete only the two lab namespaces:

```bash
set -euo pipefail
sudo ip netns exec c tc qdisc del dev c0 root 2>/dev/null || true
sudo ip netns exec s tc qdisc del dev s0 root 2>/dev/null || true

server_pid=$(cat netlab/out/what-actually-happens-when-you-curl-a-url/server.pid)
sudo kill "$server_pid" 2>/dev/null || true

# Optional full teardown of this lab topology.
sudo ip netns del c 2>/dev/null || true
sudo ip netns del s 2>/dev/null || true
```

The observation connects directly to the article's claim. One command produced several cumulative timestamps. By applying one controlled network change, we moved the TCP and first-byte boundaries in predictable dependency-sized steps. We did not need to label the whole request "network slow."

## How to use this map in an incident

The fastest useful incident loop is small:

1. State the user-visible symptom in one sentence.
2. Preserve one failing request's effective URL, destination, protocol, byte count, and cumulative timings.
3. Place the first changed boundary on the path map.
4. Capture the packet or state transition that boundary depends on.
5. Compare against a controlled baseline or unaffected path.
6. Change one variable and repeat.

This avoids two familiar traps. The first is layer drift: a team sees a timeout and immediately tunes application thread pools even though no connection reached the listener. The second is metric drift: a team opens dozens of dashboards and finds a graph that moved at roughly the right time without proving causality.

A boundary makes ownership provisional but testable. A slow name lookup belongs first to the resolver path. A slow connect belongs first to addressing, route, filters, listeners, and transport. A stable connect with a slow TLS phase belongs first to TLS negotiation. A stable secure setup with a slow first byte belongs at or behind HTTP handling. A fast first byte with a slow last byte belongs to streaming, bytes, windows, loss, pacing, or client consumption.

Retries deserve special suspicion. A successful final `curl` can hide an earlier failed attempt if retry flags are enabled. Record `num_retries` on curl versions that expose it and include retry policy in the experiment. The [timeouts, retries, and backoff post](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) owns retry policy. The networking task is to show which attempt crossed which wire boundary.

Proxies also deserve explicit treatment. Environment variables can redirect curl through HTTP, HTTPS, or SOCKS proxies. `NO_PROXY` matching can make two apparently identical hosts take different paths. Print relevant environment settings, inspect verbose output, and record both `remote_ip` and the effective URL. The address curl connects to may be the proxy, not the origin.

## Key takeaways

- A curl request is a causal chain, not one opaque network duration.
- Curl timing fields are cumulative. Subtract adjacent boundaries to price phases.
- DNS chooses an address; request packets follow a separate data path.
- The URL host carries both resolution input and service identity. `--resolve` preserves identity while overriding address selection.
- TCP, TLS, HTTP first byte, and body completion expose different dependencies and owners.
- Time to first byte includes request travel, service work, backend waits, and response travel. Client timing alone cannot split them.
- Small bodies tend to expose fixed dependency costs. Large bodies can expose serialization, congestion, receive windows, pacing, and loss.
- Diagnose the earliest boundary that moved, then use the command that observes that boundary directly.
- Every number must be derived, cited with conditions, or reproducible with a command and expected range.
- Mutate networks only inside an explicitly scoped lab. Production diagnosis begins with read-only evidence.

## Further reading

- [curl `--write-out` documentation](https://curl.se/docs/manpage.html#-w), current online manual, consulted September 29, 2026
- [RFC 9293: Transmission Control Protocol](https://www.rfc-editor.org/rfc/rfc9293.html), August 2022
- [RFC 8446: TLS 1.3](https://www.rfc-editor.org/rfc/rfc8446.html), August 2018
- [RFC 9110: HTTP Semantics](https://www.rfc-editor.org/rfc/rfc9110.html), June 2022
- [RFC 1035: Domain Names, Implementation and Specification](https://www.rfc-editor.org/rfc/rfc1035.html), November 1987
- [Fastly: Summary of June 8 outage](https://www.fastly.com/blog/summary-of-june-8-outage), June 8, 2021

The rest of this series keeps the same four views. We will light one path segment, price one latency term, draw the state transition that carries it, and show the frontier where a tuning win becomes a new failure mode. The destination is not memorizing protocols. It is being able to predict what should happen, observe what did happen, and explain the difference.
