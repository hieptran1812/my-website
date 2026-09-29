---
title: "How a Packet Gets There: ARP, Switching, Routing, and ECMP"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn to follow one packet from route lookup through ARP, switching, longest-prefix match, and flow-based ECMP without guessing which layer made the decision."
tags:
  [
    "networking",
    "distributed-systems",
    "ethernet",
    "arp",
    "switching",
    "vlan",
    "routing",
    "ecmp",
    "linux-networking",
    "packet-debugging",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 41
image: "/imgs/blogs/how-a-packet-gets-there-arp-switching-routing-and-ecmp-1.webp"
---

A service can be healthy, its DNS answer can be correct, and the client can still fail before sending one useful byte. The route points at the wrong interface. The next-hop neighbor never answers ARP. A switch learned a MAC address in the wrong VLAN. Or four equal-cost links exist, yet one large connection fills only one of them. These failures feel unrelated when we start from protocol names. They become one problem when we start from the forwarding decision.

![The host chooses a layer-3 next hop, then resolves that next hop to a layer-2 destination](/imgs/blogs/how-a-packet-gets-there-arp-switching-routing-and-ecmp-1.webp)

The diagram above is the mental model: IP chooses *where the packet should go next*, Ethernet chooses *which interface on this local link receives the next frame*, and every routed hop repeats that translation. A remote destination IP normally remains unchanged from source to destination, while the source and destination MAC addresses are replaced at every Ethernet hop. Once this distinction is automatic, `ip route get`, `ip neigh`, `bridge fdb`, and flow-aware probes stop looking like four unrelated tools.

This post follows the packet at that boundary. It assumes the base Linux namespaces introduced in [the series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url), then adds only the route state needed to prove the mechanism. For CIDR arithmetic and address identity, see [addresses, subnets, NAT, and the packet's two identities](/blog/software-development/networking/addresses-subnets-nat-and-the-packets-two-identities). For what the socket API asks the kernel to send, continue to [TCP versus UDP sockets](/blog/software-development/networking/sockets-and-the-two-transport-contracts-tcp-vs-udp).

> A host never asks ARP, "Where is the internet?" It asks, "Which MAC owns the next-hop IP on this link?"

## The forwarding decision starts before the first frame

When an application calls `connect`, the kernel does not begin with Ethernet. It begins with an IP destination, a source-selection problem, and a routing policy lookup. Only after that lookup says `dev c0` and identifies either an on-link destination or a gateway does the kernel know which IPv4 address must be resolved to a MAC address.

That order matters. Suppose the client is `10.77.0.1/24` and wants `10.77.0.42`. The connected prefix says the destination is on-link, so the next-hop IP is `10.77.0.42`. The host resolves that address directly. Now suppose the destination is `203.0.113.9`, with a default route through `10.77.0.254`. The next-hop IP is the gateway, `10.77.0.254`, not `203.0.113.9`. The Ethernet frame goes to the gateway's MAC while the IP header still names `203.0.113.9` as its destination.

This is the clean separation between layer 2 and layer 3:

| Question | Layer 2 answer | Layer 3 answer | Command that exposes it |
| --- | --- | --- | --- |
| Who receives this frame on the current link? | Destination MAC | Not decided here | `ip neigh show dev c0` |
| Which logical destination is the packet trying to reach? | Not decided here | Destination IP | `ip route get 203.0.113.9` |
| Which local interface transmits? | Port selected after route lookup | Egress device from the route | `ip route get` field `dev` |
| Which local-link peer receives a remote packet? | Gateway MAC | Gateway IP is the next hop | `ip route get` field `via` |
| What changes at a router? | Ethernet header | TTL and header checksum for IPv4 | Packet capture on both links |

The last row deserves care. A normal router removes the incoming link-layer header, processes the IP packet, decrements IPv4 TTL, updates the IPv4 header checksum, performs a new route lookup, resolves the new next hop if necessary, and emits a new frame. NAT can also change IP addresses or ports, but routing itself does not require NAT. Treating routing and NAT as the same operation makes packet captures unnecessarily confusing.

### The two destination test

When a path is unclear, compare one known on-link address with one remote address:

```bash
ip -n c route get 10.77.0.2
ip -n c route get 203.0.113.9
```

Read `dev`, `src`, and, when present, `via`. The first query should choose `dev c0 src 10.77.0.1` without a gateway in the canonical lab. The second query requires a route before it can succeed; once configured, it should add `via 10.77.0.2`. These lines are the kernel's resolved decisions, not a scan of routes that might match.

The `ip-route(8)` manual describes `ip route get` as printing a single route exactly as the kernel sees it. It also supports inputs such as source address, incoming interface, mark, IP protocol, source port, and destination port. That matters when policy routing or ECMP makes a destination-only question underspecified. A bare destination is a good first question, not always the whole question.

### A useful latency model

ARP is not one of the canonical application timing phases, but a cold neighbor lookup can sit in front of the first transmitted IP packet. A useful explanatory model is:

$$
T_{\text{first packet}} \approx T_{\text{route lookup}} + T_{\text{neighbor resolution}} + T_{\text{queue}} + T_{\text{serialization}}
$$

This is an operational decomposition, not an equation stated by the ARP specification. On a neighbor-cache hit, the resolution term is effectively removed from the critical path. On a miss, the first packet can wait while ARP runs. The important diagnostic is therefore not to memorize a universal ARP latency. It is to compare warm and flushed-neighbor attempts while capturing ARP and the first IPv4 frame.

## ARP is the bridge between an IP decision and an Ethernet frame

![An ARP miss precedes the first IPv4 frame](/imgs/blogs/how-a-packet-gets-there-arp-switching-routing-and-ecmp-2.webp)

ARP, the Address Resolution Protocol, maps a protocol address such as an IPv4 address to a local-network hardware address. [RFC 826](https://www.rfc-editor.org/rfc/rfc826.html), published in November 1982, describes the original Ethernet mechanism: if the mapping is absent, the sender broadcasts a request that identifies the target protocol address; the owner returns its hardware address; the sender can then transmit the queued packet in a frame.

The sequence is simple, but four details prevent most ARP debugging mistakes.

First, the request is broadcast at Ethernet because the sender does not yet know the target MAC. Every station in that broadcast domain may receive the frame, but only the owner of the requested IPv4 address should answer under normal operation. The reply is normally unicast back to the requester because the target learned the requester's addresses from the request.

Second, ARP is link-local. Routers do not forward an ARP broadcast into another subnet. If the IP destination is remote, the host resolves the gateway address chosen by the route lookup. Asking why `arping 203.0.113.9` receives no answer on a private LAN is usually asking the wrong layer to find a remote host.

Third, a neighbor entry is state, not eternal truth. Linux exposes IPv4 ARP and IPv6 Neighbor Discovery through the neighbor table. Entries move among states such as `INCOMPLETE`, `REACHABLE`, `STALE`, `DELAY`, `PROBE`, and `FAILED`. The state tells us whether the kernel has a link-layer address and how confident it is that the neighbor remains reachable. The exact transition timers are implementation and configuration details, so inspect the running host rather than copying a timeout from an old article.

Fourth, ARP has no authentication. A received mapping can update local state according to the implementation's rules. That property enables legitimate movement and recovery, but it also creates spoofing risk on a hostile layer-2 segment. Network access controls, switch features, segmentation, and cryptographic protection above Ethernet address different parts of that risk. ARP itself does not prove identity.

### Read the cache and the wire together

Use the table to see state:

```bash
ip -n c neigh show dev c0
```

Use a bounded capture to see why the state changed:

```bash
sudo ip netns exec c timeout 10 \
  tcpdump -ni c0 -c 8 -e 'arp or host 10.77.0.2'
```

The `-e` flag includes the Ethernet header. Without it, a capture can show correct IP addresses while hiding the MAC decision we are trying to verify. The filter and packet limit also matter: captures can contain payloads, credentials, tokens, and personal data, so collect only what answers the question.

On a cold neighbor table, expect an ARP request, an ARP reply, then the first IPv4 packet. On a warm entry, expect the IPv4 packet without a preceding ARP exchange. Scheduler and virtual-machine noise can change elapsed time, but it cannot reverse this causal order.

### Incomplete is different from wrong

An `INCOMPLETE` or `FAILED` entry means resolution did not finish. Common causes include a wrong VLAN, a down peer, a missing address, filtering, or a link problem. A complete entry with the wrong MAC is a different class of failure. It may reflect a duplicate address, stale movement, proxy ARP, spoofing, or an operator assumption that the IP belongs to a different interface.

That distinction sets the next command. For an incomplete entry, capture the request and ask whether it leaves the correct interface and reaches the correct broadcast domain. For a complete but suspicious entry, compare the learned MAC with the switch forwarding database and the peer's actual interface. Deleting the neighbor entry can reproduce a lookup, but it is a mutation. Do it only in an isolated lab or during a controlled production procedure:

```bash
sudo ip -n c neigh del 10.77.0.2 dev c0
```

The safer production move is read-only first: `ip neigh show`, a narrowly filtered capture, and switch-side forwarding state. Repeatedly flushing a live cache can manufacture the very packet loss or latency spike under investigation.

## Switches learn locally, and VLANs bound that learning

![A switch learns source MAC locations inside one VLAN](/imgs/blogs/how-a-packet-gets-there-arp-switching-routing-and-ecmp-3.webp)

An Ethernet switch forwards frames using a forwarding database, usually shortened to FDB. The important verb is *learns*. When a frame enters a port, the switch observes the source MAC and associates it with that ingress port in the relevant VLAN. When it later sees a frame for a known destination MAC, it can send the frame only to the learned egress port.

When the destination is unknown, the switch cannot invent a port. It floods the frame to eligible ports in the same VLAN, excluding the ingress port. Broadcast frames follow a similar VLAN-scoped fan-out. Once a reply returns, source learning usually makes later forwarding selective.

This produces a compact troubleshooting rule:

| Observation | Likely decision point | Next evidence | Source |
| --- | --- | --- | --- |
| ARP request never appears on the host interface | Host route or neighbor path | `ip route get`, `ip neigh` | Reproducible in `netlab` |
| Request leaves host but not expected switch port | VLAN membership or switch forwarding | `bridge vlan show`, switch counters | Reproducible on Linux bridge |
| Unknown unicast floods briefly, then becomes selective | Normal source learning | `bridge fdb show` before and after traffic | Linux bridge state |
| MAC repeatedly moves between ports | Loop, dual attachment, or address duplication | FDB event log and topology | Deployment-specific evidence |
| ARP reply arrives but neighbor remains failed | Host filtering or malformed reply | `tcpdump -e -vv`, kernel logs | Reproducible in controlled lab |

Every row with observed behavior points to a command rather than a conclusion. Flooding is not automatically a loop. One short burst can be ordinary unknown-unicast behavior. Conversely, a stable FDB does not prove end-to-end IP reachability; it proves only that the bridge believes a MAC is reachable on a port.

### VLANs create separate forwarding contexts

A VLAN is a logical layer-2 segment. With VLAN-aware switching, FDB identity is effectively scoped by both MAC and VLAN. Port membership determines where broadcasts and unknown unicasts may travel. A router or layer-3 switch connects IP subnets across VLAN boundaries; the bridge does not forward a VLAN 10 broadcast into VLAN 20 simply because both live in the same chassis.

This is why "the switch learned the MAC" is incomplete. Ask four questions:

1. Which VLAN owns the entry?
2. Which port learned it?
3. Is that port forwarding for the VLAN?
4. Is the frame tagged or untagged as the receiving side expects?

On a Linux bridge, these commands separate those facts:

```bash
bridge link show
bridge vlan show
bridge fdb show br br0
```

The Linux kernel's [Ethernet bridging documentation](https://docs.kernel.org/networking/bridge.html) describes the bridge as the in-kernel Ethernet switching component and documents VLAN filtering and forwarding-database controls. The exact hardware offload path can differ, so on a physical switch use its authoritative MAC-address and VLAN tables. The conceptual questions remain the same.

### Why an ARP failure can really be a VLAN failure

An ARP request is a broadcast within a layer-2 domain. Put the client access port in VLAN 10 and the gateway interface in VLAN 20, and both endpoints can be administratively up while the request never reaches the gateway. From the client, the symptom is neighbor resolution failure. From the switch, the cause is membership. From an application dashboard, the symptom may appear as a connect timeout.

The layers are not competing diagnoses. They are consecutive decisions. The application cannot connect because no IP packet leaves. No IP packet leaves because the next-hop MAC is unresolved. The MAC is unresolved because the ARP request cannot cross the VLAN boundary. A useful incident note preserves that chain instead of stopping at "ARP was broken."

### Second-order effect: moving hosts and stale state

Virtual machines, containers, and failover addresses move more often than physical servers once did. Source learning lets a switch update a MAC location when frames arrive on a new port. Gratuitous ARP or ordinary traffic can refresh host neighbor state. But convergence is not instantaneous across every cache, bridge, offload table, and peer.

During a move, prove each layer separately. Verify the new owner emits traffic, the switch learns the MAC on the new port in the correct VLAN, and peers update their neighbor entries. Avoid assigning one fixed timer to the whole event. Different devices use different aging, reachability, and control-plane rules. What matters is which table still points at the old location.

## Routers choose the longest matching prefix

![Longest-prefix match keeps the most specific matching route](/imgs/blogs/how-a-packet-gets-there-arp-switching-routing-and-ecmp-4.webp)

A routing table is not an ordered firewall rule list. Several prefixes can match one destination, and the route with the longest matching prefix is the most specific. [RFC 1812](https://www.rfc-editor.org/rfc/rfc1812.html), published in June 1995, states the longest-match rule for IPv4 routers and gives the classic hierarchy from host routes through network prefixes to the default route.

For destination `10.77.8.42`, all of these prefixes match:

| Prefix | Matching leading bits | Matches? | Outcome | Source |
| --- | ---: | --- | --- | --- |
| `0.0.0.0/0` | 0 | Yes | Candidate default | Derived here from prefix definition |
| `10.0.0.0/8` | 8 | Yes | Beats `/0` | RFC 1812 longest match |
| `10.77.0.0/16` | 16 | Yes | Beats `/8` | RFC 1812 longest match |
| `10.77.8.0/24` | 24 | Yes | Selected before equal-cost choice | RFC 1812 longest match |
| `10.77.9.0/24` | 23 before first difference | No | Discarded | Derived bit comparison |

The first four all contain the destination, but `/24` constrains the most leading bits. It wins even if the default route has a lower numeric metric. Metrics compare candidates at the same specificity within the relevant routing process or implementation policy; they do not make a shorter prefix beat a longer one in ordinary forwarding.

### Derive one match instead of trusting the notation

The last octet of `10.77.8.42` is irrelevant to a `/24` match because the first 24 bits are the three octets `10.77.8`. In binary, the comparison is:

```console
destination: 00001010 01001101 00001000 00101010
route /24:  00001010 01001101 00001000 --------
```

This is a bit comparison printed for the derivation, not an architecture diagram. The first 24 destination bits equal the route prefix, so it matches. The `/16` also matches because its first 16 bits are the same, but it provides eight fewer matching bits. Longest-prefix match therefore selects `/24`.

For a second worked example, `203.0.113.9` does not match any `10.0.0.0/8` route because the first octet differs. A configured `203.0.113.0/24` wins over default. Without that specific route, `0.0.0.0/0` remains the candidate because a zero-length prefix imposes no bit constraint.

### Routing policy comes before the selected table's longest match

Linux can hold multiple routing tables and a routing policy database. Source prefixes, marks, incoming interfaces, or other selectors can choose which lookup rules apply. Within that world, `ip route show` is necessary but not always sufficient. It dumps routes; it does not prove which policy inputs the packet carries.

Use `ip rule show` to inspect the policy chain, then ask a route question with the same relevant fields as the real traffic:

```bash
ip rule show
ip route get 203.0.113.9 from 10.77.0.1 ipproto tcp sport 40000 dport 443
```

The [iproute2 route manual](https://man7.org/linux/man-pages/man8/ip-route.8.html), viewed on 2026-09-29, documents those route-get selectors and explains that `fibmatch` returns the full matched FIB route rather than only the resolved destination entry. The command is valuable because it asks the kernel to perform a lookup. It is still not a packet capture, and it cannot prove that a downstream router makes the same choice.

### Recursive next hops and the final neighbor

A route may point to a gateway that itself must be reachable through a connected route. The kernel resolves that dependency until it obtains an egress device and an on-link next hop. ARP applies to that final on-link IPv4 address.

This explains an apparently contradictory observation: `ip route get 203.0.113.9` names the remote destination and a gateway, while `ip neigh show` contains no entry for `203.0.113.9`. Nothing is missing. The route says to send the remote packet *via* an on-link gateway, and the neighbor table contains the gateway's MAC.

## Diagnose the chosen path without guessing

![A decision tree maps route, neighbor, bridge, and flow evidence](/imgs/blogs/how-a-packet-gets-there-arp-switching-routing-and-ecmp-6.webp)

Start at the highest forwarding decision the local kernel can answer, then descend only when the evidence tells you to. This prevents a common failure pattern: opening a broad packet capture, seeing thousands of frames, and deciding the network is mysterious.

### Step 1: ask the route question precisely

```bash
ip route get 203.0.113.9 \
  from 10.77.0.1 \
  ipproto tcp \
  sport 40000 \
  dport 443
```

Read these fields:

- `via`: the on-link gateway IP, if routing requires one;
- `dev`: the selected egress interface;
- `src`: the source address the kernel would choose;
- `table`: a non-main table when output includes it;
- cache or route flags that qualify the result.

If any field differs from the real socket, refine the query. A bound source address, packet mark, VRF, or policy rule can change the result. Use `ss -ntp` or application configuration to learn the actual local and remote tuple before inventing probe values.

### Step 2: inspect the selected next hop

If the route says `via 10.77.0.254 dev eth0`, inspect exactly that neighbor:

```bash
ip neigh show to 10.77.0.254 dev eth0
```

An absent entry before traffic is not itself a failure. Generate one controlled probe and watch the state. `INCOMPLETE` followed by `FAILED` focuses the investigation on the local link, VLAN, peer, or ARP handling. `REACHABLE` or `STALE` provides a MAC but does not prove the peer forwards the packet onward.

### Step 3: inspect bridge state only if a bridge owns the segment

```bash
bridge vlan show
bridge fdb show br br0
```

Do not run bridge commands on an ordinary routed host and treat empty output as evidence. First establish whether a Linux bridge, virtual switch, top-of-rack switch, or cloud fabric owns layer-2 forwarding. Then query the authoritative surface.

### Step 4: capture the smallest proof

```bash
sudo timeout 15 tcpdump -ni eth0 -c 30 -e \
  'arp or (host 203.0.113.9 and tcp port 443)'
```

Read the Ethernet source and destination, ARP target, IP source and destination, and TCP tuple. A frame addressed to the gateway MAC with the remote service IP is correct for routed traffic. Do not label that combination inconsistent.

### Step 5: move one hop at a time

If the local evidence is correct, the next unknown lives downstream. Compare captures or interface counters on both sides of one boundary. The goal is not to collect more data. It is to find the first point where the packet or forwarding state differs from the expected path.

The wider operational strategy belongs in [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design). This post owns the packet-level evidence. Likewise, [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) owns endpoint lifecycle above the wire, while this post explains how a chosen endpoint becomes a next hop and a frame.

## ECMP balances flows, not packets

<figure class="blog-anim">
<svg viewBox="0 0 1200 700" role="img" aria-label="Stable ECMP flow hashing across two equal-cost links while one elephant flow stays pinned to the upper link" style="width:100%;height:auto;max-width:1200px">
<title>ECMP hashes stable flows, not individual packets</title>
<style>
.pkt4-bg{fill:#e9ecef;stroke:#1e1e1e;stroke-width:3}.pkt4-active{fill:#a5d8ff;stroke:#1e1e1e;stroke-width:4}.pkt4-ext{fill:#d0bfff;stroke:#1e1e1e;stroke-width:3}.pkt4-link{fill:none;stroke:#1e1e1e;stroke-width:8;stroke-linecap:round}.pkt4-label{font:26px system-ui,sans-serif;fill:#1e1e1e;text-anchor:middle}.pkt4-code{font:21px ui-monospace,monospace;fill:#1e1e1e;text-anchor:middle}.pkt4-small{font:19px system-ui,sans-serif;fill:#1e1e1e;text-anchor:middle}.pkt4-f1,.pkt4-f2,.pkt4-f3,.pkt4-elephant{stroke:#1e1e1e;stroke-width:2}.pkt4-f1{fill:#a5d8ff;animation:pkt4-top-a 10s linear infinite}.pkt4-f2{fill:#d0bfff;animation:pkt4-bottom-a 10s linear infinite 1.2s}.pkt4-f3{fill:#b2f2bb;animation:pkt4-bottom-b 10s linear infinite 2.4s}.pkt4-elephant{fill:#ffec99;animation:pkt4-elephant-top 10s linear infinite}.pkt4-pulse{animation:pkt4-hash-pulse 10s ease-in-out infinite;transform-box:fill-box;transform-origin:center}@keyframes pkt4-top-a{0%,8%{transform:translate(0,0);opacity:1}32%{transform:translate(300px,0)}58%{transform:translate(580px,-145px)}84%,100%{transform:translate(910px,-145px);opacity:1}}@keyframes pkt4-bottom-a{0%,8%{transform:translate(0,0);opacity:1}32%{transform:translate(300px,0)}58%{transform:translate(580px,145px)}84%,100%{transform:translate(910px,145px);opacity:1}}@keyframes pkt4-bottom-b{0%,8%{transform:translate(0,0);opacity:1}32%{transform:translate(300px,0)}58%{transform:translate(580px,145px)}84%,100%{transform:translate(910px,145px);opacity:1}}@keyframes pkt4-elephant-top{0%,8%{transform:translate(0,0)}32%{transform:translate(300px,0)}58%{transform:translate(580px,-145px)}84%,100%{transform:translate(910px,-145px)}}@keyframes pkt4-hash-pulse{0%,25%,40%,100%{transform:scale(1)}32%{transform:scale(1.08)}}@media (prefers-reduced-motion:reduce){.pkt4-f1,.pkt4-elephant{animation:none;transform:translate(700px,-145px)}.pkt4-f2,.pkt4-f3{animation:none;transform:translate(700px,145px)}.pkt4-pulse{animation:none}}
</style>
<rect class="pkt4-bg" x="50" y="245" width="210" height="210" rx="28"/>
<text class="pkt4-label" x="155" y="305">flows arrive</text>
<text class="pkt4-code" x="155" y="350">src,dst,ports</text>
<text class="pkt4-small" x="155" y="395">packet fields stay fixed</text>
<path class="pkt4-link" d="M260 350 H420"/>
<g class="pkt4-pulse">
<rect class="pkt4-active" x="420" y="245" width="240" height="210" rx="28"/>
<text class="pkt4-label" x="540" y="305">stable flow hash</text>
<text class="pkt4-code" x="540" y="350">same 5-tuple</text>
<text class="pkt4-small" x="540" y="395">same next-hop bucket</text>
</g>
<path class="pkt4-link" d="M660 350 L805 205 H1025"/>
<path class="pkt4-link" d="M660 350 L805 495 H1025"/>
<rect class="pkt4-ext" x="1025" y="115" width="130" height="180" rx="24"/>
<rect class="pkt4-ext" x="1025" y="405" width="130" height="180" rx="24"/>
<text class="pkt4-label" x="1090" y="180">next hop A</text>
<text class="pkt4-code" x="1090" y="225">equal cost</text>
<text class="pkt4-label" x="1090" y="470">next hop B</text>
<text class="pkt4-code" x="1090" y="515">equal cost</text>
<circle class="pkt4-f1" cx="120" cy="328" r="18"/>
<circle class="pkt4-f2" cx="120" cy="350" r="18"/>
<circle class="pkt4-f3" cx="120" cy="372" r="18"/>
<rect class="pkt4-elephant" x="75" y="405" width="90" height="30" rx="15"/>
<text class="pkt4-small" x="600" y="630">The large amber flow stays on one link; distinct smaller flows can occupy both.</text>
</svg>
<figcaption>A stable flow hash pins every packet of one flow to one equal-cost next hop, so one elephant flow cannot use both links.</figcaption>
</figure>

Equal-cost multipath, or ECMP, exists after longest-prefix match has identified a route with multiple equally eligible next hops. The forwarding device needs a repeatable way to choose among them. A common design hashes fields that identify a flow, maps the hash into a bucket, and assigns that bucket to a next hop.

[RFC 2992](https://www.rfc-editor.org/rfc/rfc2992.html), published in November 2000, analyzes a hash-threshold ECMP algorithm. It describes hashing packet-header fields that identify a flow and dividing the key space among next hops. [RFC 6438](https://www.rfc-editor.org/rfc/rfc6438.html), published in November 2011 for IPv6 flow labels, explains the underlying tension: traffic should share available paths while individual flows should avoid reordering.

For TCP or UDP, implementations often use some version of the five-tuple:

$$
K = (\text{source IP}, \text{destination IP}, \text{protocol}, \text{source port}, \text{destination port})
$$

That tuple is a conceptual flow key. The exact hash inputs and algorithm are implementation and configuration choices. Some devices use fewer fields, add fields, support symmetric hashing, or use tunnel headers. Linux exposes IPv4 multipath hash policy through `net.ipv4.fib_multipath_hash_policy`, whose meaning depends on kernel version. Inspect the running documentation and sysctl rather than assuming every router uses the same key.

### The rule that surprises people

With conventional per-flow ECMP, one stable flow selects one next hop. Sending a single TCP connection over a fabric with two 100 Gbit/s paths does not make that connection a 200 Gbit/s pipe. The paths increase aggregate capacity across many independently hashed flows. One elephant flow can still pin one member while another remains underused.

This is not a defect in the hash. It is the cost of avoiding packet reordering. If consecutive packets of one flow alternated between paths with different queueing or propagation delay, later packets could arrive first. TCP can interpret reordering as a loss signal in some circumstances, spend receiver buffer, and complicate measurement. Per-flow affinity trades perfect instantaneous balance for stable ordering.

A simple derived occupancy example shows the limit. Suppose eight equal-cost links carry eight equal-rate flows. Uniform hashing does **not** guarantee one flow per link. There are ${8}^8 = 16{,}777{,}216$ possible link assignments if each flow independently selects one of eight links. Only ${8}! = 40{,}320$ assignments put exactly one flow on every link. Therefore the probability of perfect one-per-link occupancy under the idealized independent uniform model is:

$$
P(\text{perfect occupancy}) = \frac{8!}{8^8} = \frac{40{,}320}{16{,}777{,}216} \approx 0.0024
$$

That is about 0.24 percent. This is a derived balls-into-bins model, not a measured fabric result. Real flow sizes are unequal, hashes may not be perfectly independent, and routing hardware has implementation-specific buckets. The model still proves the operational point: "eight flows and eight links" is not enough to expect perfect balance.

### More flows improve aggregate balance, not one flow's ceiling

If a load test uses one connection, it measures one hashed path plus endpoints. If it uses many connections with varying source ports, it samples more flow keys and can occupy more members. That is why throughput tools expose parallel-stream options, and why their results must state concurrency.

The same fact explains uneven backend load behind a layer-4 balancer. Persistent connections reduce connection churn and handshake cost, but they also reduce the number of independent balancing decisions. The architectural consequences are covered in [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7). At the wire, the key question is simpler: which fields create a new hash input, and how many independently hashed flows actually exist?

### ECMP member changes can remap live flows

Removing a next hop changes the set of available buckets. A naive modulo or contiguous hash-threshold mapping can move flows that did not use the failed member. That disruption matters when different paths have different delay or when an ECMP group distributes traffic toward stateful servers.

The Linux kernel's [resilient next-hop group documentation](https://docs.kernel.org/networking/nexthop-group-resilient.html), consulted on 2026-09-29, explains an indirection table that aims to minimize flow movement when group membership or weights change. Buckets assigned to a removed next hop can be reassigned while busy buckets on surviving next hops remain stable when possible. The document also warns why movement matters: a flow can reach an unexpected server, or packets can experience reordering across different-latency paths.

Resilient hashing reduces disruption. It does not make a failed path harmless, preserve every flow, or aggregate one flow across all members. Keep those claims separate.

### Test ECMP with flow keys, not repeated pings

One ICMP echo stream often keeps the fields used by a device's hash stable. Sending it repeatedly can prove that one path works, but it cannot prove that an ECMP group distributes traffic. Even a classic traceroute may change UDP destination ports between probes and therefore sample several paths while presenting the result as one trace. The measurement must match the question.

For one-flow stability, hold the tuple constant and repeat. For group coverage, change exactly one hashed field, usually the source port, and record which member each flow reaches. On Linux, a route-get query can include a transport tuple:

```bash
for sport in 40000 40001 40002 40003 40004 40005 40006 40007; do
  ip route get 203.0.113.9 \
    from 10.77.0.1 \
    ipproto tcp \
    sport "$sport" \
    dport 443
done
```

Whether the local kernel varies next hops depends on its route, kernel, hash policy, and seed. Treat this loop as a lookup method, not as a promise that eight adjacent ports must cover every member. The expected qualitative result with an active multipath route is that a given complete tuple resolves consistently while a set of distinct tuples may select different next hops.

At a hardware router, use the vendor's flow-aware forwarding lookup or member counters. Hold source and destination addresses, protocol, and ports constant when checking affinity. Vary one input across many probes when checking distribution. If encapsulation exists, establish whether the router hashes the outer headers, inner headers, or both. A beautifully even inner five-tuple is irrelevant if the device sees one invariant tunnel header.

### Diagnose one bad member from repeatable subsets

A particularly revealing failure is "some connections always time out, but retries sometimes succeed." If DNS rotates addresses, a load balancer chooses backends, and ECMP chooses fabric paths, several decisions could create that subset. Stabilize them one at a time.

Pin the destination address first. Then generate a known set of source ports. If the same source ports repeatedly fail while others pass, and the mapping is stable over short intervals, suspect a deterministic hash partition such as one ECMP member. Compare interface error, discard, and queue counters on every member over the same test interval. Do not infer a bad member merely from unequal byte counts; unequal flow sizes create unequal bytes even when every path works.

If failures move after a route update, that is further evidence of remapping, but it is not proof by itself. Route updates can coincide with neighbor churn, ACL programming, tunnel changes, or backend membership. The strongest diagnosis combines a controlled flow-key matrix, the device's selected-next-hop lookup, and packet or counter evidence at the suspected member.

### Hash balance has two different scoreboards

Operators often say "ECMP is balanced" without naming the unit. Flow-count balance asks whether roughly equal numbers of flow keys map to each member. Byte balance asks whether members carry roughly equal traffic volume. A perfect flow count can coexist with terrible byte balance when one flow is much larger than the rest.

In a derived example with four links and eight flows, seven flows each send 1 unit while one elephant sends 100 units. A hash could place two flows on every link, which is perfect flow-count balance. If the elephant shares one member with a 1-unit flow, that member carries 101 units while each other member carries 2 units. Total traffic is 107 units, so the elephant's member carries ${101} / 107 \approx 94.4\%$ of bytes despite owning only ${2} / 8 = 25\%$ of flows.

This arithmetic is a workload model, not a reported measurement. It tells us which counters to read. Compare member flow counts, bytes, queue depth, drops, and latency. When the skew is caused by a small number of elephants, changing the hash seed merely moves the pain. More independent flows, application sharding, multipath transport, or a carefully designed flowlet scheme changes the granularity of balancing. Each option introduces its own ordering, state, or complexity cost.

### Per-packet and flowlet switching are explicit exceptions

Some systems can distribute packets more finely than a full connection. Per-packet load sharing accepts reordering risk. Flowlet switching looks for sufficiently large gaps within a flow and may move a later burst when the previous burst is unlikely to overlap. Multipath transport protocols can expose several subflows below one application-level session.

Those are designed behaviors, not reasons to discard the per-flow rule. When diagnosing an ordinary ECMP fabric, assume stable flow hashing until device configuration and packet evidence show otherwise. If a vendor says "dynamic load balancing," ask whether the unit of movement is a packet, flowlet, flow, bucket, or route. The noun determines the failure mode.

## What changes at every routed hop

It is worth following one remote packet across two Ethernet segments because captures from different points can otherwise look like different conversations.

At the client, the kernel has an IP packet from `10.77.0.1` to `203.0.113.9`. The route selects gateway `10.77.0.254` on `eth0`. ARP supplies the gateway's MAC. The emitted frame therefore has the client's source MAC and gateway's destination MAC, while its payload still contains the original source and destination IP addresses.

The router accepts the frame because the destination MAC belongs to its incoming interface. It removes the Ethernet header, validates and processes the IP header, decrements TTL, and performs a route lookup for `203.0.113.9`. Suppose the selected outgoing interface reaches next hop `198.51.100.2`. The router resolves that on-link address and emits a new frame whose source MAC belongs to the router's outgoing interface and whose destination MAC belongs to `198.51.100.2`.

The invariant and changing fields are:

| Field | Client segment | Next routed segment | Why | Source |
| --- | --- | --- | --- | --- |
| Destination IPv4 | `203.0.113.9` | `203.0.113.9` | Routing forwards toward the IP destination | Reproducible with two-link capture |
| Source IPv4 | `10.77.0.1` | `10.77.0.1` | No NAT in this example | Reproducible with two-link capture |
| Destination MAC | Gateway MAC | Next-hop MAC | Each frame targets a local-link receiver | RFC 826 plus routed forwarding |
| Source MAC | Client MAC | Router egress MAC | Each frame originates on a new link | Reproducible with two-link capture |
| IPv4 TTL | Initial value | Initial value minus 1 | Router advances the packet one hop | RFC 1812 |
| TCP source port | Unchanged | Unchanged | Routing alone does not rewrite transport ports | Reproducible with two-link capture |

The addresses are examples from documentation ranges, not a captured production trace. Their value is structural. If a capture after a router shows a different destination MAC, that is expected. If it shows the same destination IP in a no-NAT topology, that is expected. Comparing only one header layer leads to false contradictions.

### The default gateway is a route, not a special Ethernet role

A default gateway is the next hop attached to a default route, usually `0.0.0.0/0` for IPv4. Ethernet does not mark one MAC as "the gateway." ARP does not return a gateway flag. The IP routing table decides that unmatched remote destinations use a particular next-hop IP, and neighbor resolution turns that IP into a local-link address.

More specific routes can bypass the default. A host route can send one destination through a different gateway. A connected prefix can make a destination on-link. Policy routing can select a different table. This is why reading a configured `default via ...` line is not enough to explain one packet. Ask the resolved lookup for that packet's actual inputs.

Two default routes may exist with different metrics for failover, or as equal-cost alternatives where supported and configured. A default route can therefore lead into ECMP, but longest-prefix match still happens first. Any matching `/1` through `/32` route is more specific than `/0` and wins the prefix comparison.

### Proxy ARP deliberately bends the ordinary boundary

Proxy ARP is an explicit exception worth recognizing. A router can answer an ARP request on behalf of an IPv4 address that is not actually on that local segment, then route received packets onward. To the sender, the destination appears directly reachable at layer 2 even though a router sits in the path.

This can make migrations or unusual subnet layouts work, but it hides the routed boundary from hosts and expands the set of addresses for which a router must answer ARP. When a supposedly on-link destination resolves to the gateway's MAC, check for proxy ARP before concluding that the peer stole an address. On Linux, inspect the relevant per-interface sysctls and routing design. Do not enable proxy behavior as a generic fix for a wrong subnet mask.

### ARP failure and routing loop have different signatures

An ARP failure prevents the first frame from reaching a known next hop. A routing loop forwards the IP packet through one or more routers repeatedly until TTL expires. Both can look like a timeout at the application, but the wire evidence differs.

For ARP failure, repeated requests name the same on-link target and no valid reply completes resolution. For a routing loop, frames do leave each local segment, router interfaces resolve their next hops, and the IP TTL falls at each hop. A traceroute may expose repeated routers, but a capture and route inspection at the loop boundary give stronger evidence. Fixing neighbor state cannot repair contradictory layer-3 routes.

## The control plane proposes; the data plane forwards

Routing protocols, static configuration, controllers, and operators create candidate reachability. The forwarding information base, or FIB, is the data-plane structure used to forward packets. A route can exist in a protocol's database yet fail to reach the active FIB. Conversely, a stale programmed entry can outlive the control-plane fact that once created it.

On a Linux host, `ip route show` and `ip route get` expose closely related kernel state. On a network device with hardware forwarding, separate control-plane and hardware-table commands may be necessary. During an incident, record both the intended route and the programmed route. "BGP has it" is incomplete if the forwarding ASIC does not.

This distinction also sharpens the meaning of ECMP. The control plane determines that several next hops are equally eligible under routing policy and metric. The data plane chooses one of those members for a packet or flow according to the installed group and hash behavior. A routing-protocol neighbor count is not proof that traffic uses every member.

### A route can be valid while its next hop is unusable

The FIB can select a route through a gateway whose neighbor resolution fails. That is not a contradiction. The layer-3 decision succeeded; the layer-2 delivery prerequisite did not. Depending on queue limits and neighbor state, packets can wait, fail, or be dropped while resolution proceeds.

Likewise, a complete neighbor entry can point to a gateway that has no route onward. Local delivery succeeds, remote forwarding fails. The boundary test is simple: capture on the sender and the gateway. If the frame reaches the gateway with the expected MAC and IP destination, the sender's route and ARP have done their jobs. The next question belongs to the gateway's FIB, policy, filters, and outgoing neighbor.

### Metrics do not outrank specificity

Consider these two routes:

```console
10.77.8.0/24 via 10.0.0.2 metric 500
10.0.0.0/8  via 10.0.0.3 metric 10
```

For `10.77.8.42`, the `/24` remains more specific and wins ordinary longest-prefix selection despite its larger metric. The metric can help choose among comparable candidates, such as two routes to the same prefix. Reversing this hierarchy would make route aggregation unsafe because a cheap default or covering prefix could unexpectedly steal traffic from every specific route.

When operators say "the lower metric route should win," silently append "among routes at the relevant prefix and policy stage." That qualifier prevents hours of staring at the wrong line.

## Case study: Meta's October 4, 2021 backbone withdrawal

![Meta's backbone disconnection propagated into BGP withdrawal and DNS unreachability](/imgs/blogs/how-a-packet-gets-there-arp-switching-routing-and-ecmp-5.webp)

On October 4, 2021, Meta experienced a global outage that is useful here because it shows the forwarding hierarchy at infrastructure scale. This is not an ARP incident, and reducing it to "BGP broke" misses the causal chain. It demonstrates that a service can remain operational as a process while route withdrawal makes it unreachable.

Meta's incident owner published [More details about the October 4 outage](https://engineering.fb.com/2021/10/05/networking-traffic/outage-details/) on October 5, 2021. The post says a command intended to assess global backbone capacity unintentionally took down backbone connections. A bug in the audit tool failed to stop the command. The resulting backbone disconnection made facilities unable to reach the data centers they used as a health signal. Those facilities then declared themselves unhealthy and withdrew the BGP advertisements for Meta's authoritative DNS servers. The DNS servers were still operational, but the rest of the internet could no longer reach them.

The evidence ledger is explicit:

| Field | Verified value | Source |
| --- | --- | --- |
| Case | Meta global service outage | Meta engineering, 2021-10-05 |
| Event date | 2021-10-04 | Meta engineering, 2021-10-05 |
| Trigger | Maintenance command disconnected the global backbone | Meta engineering, 2021-10-05 |
| Control failure | Audit-tool bug did not stop the command | Meta engineering, 2021-10-05 |
| Route mechanism | Unhealthy facilities withdrew BGP advertisements | Meta engineering, 2021-10-05 |
| Blast-radius multiplier | Authoritative DNS addresses became unreachable | Meta engineering, 2021-10-05 |
| Transfer lesson | Couple health to reachability carefully and preserve independent recovery paths | Derived operational lesson |

### Map the case onto the local mental model

On one Linux host, a route lookup chooses a prefix and next hop before ARP resolves a local MAC. On the internet, BGP advertisements contribute reachability information that eventually programs forwarding tables across many networks. When the advertisement disappears, remote routers no longer have the same route toward the destination. There is no remote ARP trick that can repair a missing inter-domain route. ARP operates only after a router has selected an on-link next hop.

The trigger, contributing condition, and multiplier were different:

- The trigger was the backbone-changing command.
- The contributing control failure was the audit bug that allowed it.
- The health dependency connected backbone reachability to BGP advertisement state.
- DNS unreachability multiplied the visible failure because clients could not resolve service names.

This separation matters in design review. A health check that withdraws a route can prevent traffic from reaching an unhealthy destination. It can also remove the path needed for diagnosis and recovery if its dependencies are not independent. "Automate fail-closed" is not a complete design principle when the closed state hides every repair interface.

### The diagnostic lesson

Begin with the observable lookup at the affected scope. On a host, that might be `ip route get`. At an edge, it may be the selected BGP route, FIB entry, and next hop. Then verify that the next hop is locally resolvable. Do not start by flushing neighbor caches when the prefix itself is absent, and do not start by changing BGP when the selected gateway merely lacks a local neighbor mapping.

Meta's post does not publish a packet benchmark for this mechanism, so none is invented here. The case establishes the causal relationship among backbone connectivity, health evaluation, route withdrawal, and DNS reachability. The lab below proves the smaller host-level distinction between a remote destination and its local next hop.

## Failure patterns that look alike from the application

A connect timeout is an endpoint symptom. It does not tell us which forwarding decision failed. The table below maps similar application behavior to discriminating evidence.

| Symptom | Competing explanations | Discriminating check | Safe first action |
| --- | --- | --- | --- |
| Immediate `Network is unreachable` | No matching route or rejecting route | `ip route get` exits with an error or names an unreachable route | Inspect rules and routes, do not flush ARP |
| Connect attempt stalls before SYN appears | Neighbor resolution or local queueing | `ip neigh` plus ARP-filtered capture | Verify selected next hop and VLAN |
| SYN leaves with correct IP but wrong egress | Policy route, mark, VRF, or source selection | Flow-complete `ip route get` and `ip rule show` | Reproduce lookup inputs |
| SYN reaches gateway but not remote path | Downstream routing, ACL, or ECMP member | Hop-by-hop counters and flow-stable probes | Compare one boundary at a time |
| Many small flows spread, one large flow saturates one link | Normal per-flow ECMP | Vary source ports and observe path/member counters | Add flows or use multipath transport where appropriate |
| Some flows fail consistently while others pass | One bad ECMP member or hash bucket | Repeat probes with distinct five-tuples | Drain or repair the member after confirmation |
| MAC alternates between ports | Loop, mobility, duplicate address | Time-correlated FDB events | Verify topology before pinning an entry |

The most dangerous action is a broad mutation made before locating the layer. Clearing every route, neighbor entry, or switch table destroys evidence and creates a recovery event. Read the selected route, selected neighbor, selected VLAN, and selected flow path first.

### Why traceroute is not the first answer

Traceroute can reveal responding layer-3 hops by eliciting TTL-expired messages, but it has limits. Return paths can differ. Devices can rate-limit or filter responses. ECMP can send probes with different flow keys down different paths unless the probe mode stabilizes them. A missing hop response does not prove that data forwarding stopped there.

Use traceroute or `mtr` after the local route and next hop are known. Keep the probe tuple stable when mapping one ECMP path, and vary it deliberately when looking for multiple members. The distinction between "same-flow repeat" and "new-flow sample" is essential. Otherwise the tool changes the path while pretending to measure it.

### The single-flow capacity trap

Suppose two equal-cost links each have capacity $C$. The aggregate path set may carry up to roughly ${2}C$ under a workload with enough well-distributed flows and no other bottleneck. Under conventional per-flow ECMP, a single flow is still bounded by approximately $C$ before protocol, endpoint, and queueing constraints. This is a derived upper-bound model, not a benchmark:

$$
\text{aggregate bound} \approx 2C, \qquad \text{single-flow bound} \lesssim C
$$

If $C = 100$ Gbit/s, the model yields a 200 Gbit/s aggregate link bound and a 100 Gbit/s single-member bound. It does not promise either throughput because hosts, congestion control, packet size, overhead, and competing traffic still matter. The arithmetic exists to stop one invalid inference: two links do not double one ordinary flow automatically.

## Design rules that survive different vendors

Vendor commands change. These rules remain useful because they follow the forwarding stages.

### Keep layer-2 domains intentionally small

ARP broadcasts, unknown-unicast flooding, accidental loops, and address duplication are bounded by the VLAN. Smaller failure domains reduce how far those events spread. They also create more routed boundaries, which adds configuration and requires clear gateway redundancy. The trade-off is not "VLAN good" versus "routing good." It is the scope of shared layer-2 fate versus the cost of more layer-3 boundaries.

### Make route intent observable

Store the intended prefix, next hop, table, and policy selector in a form operators can compare with the FIB. A route present in a control-plane database is not enough if hardware programming failed. Conversely, a stale dashboard does not outweigh a flow-complete kernel lookup and packet evidence.

### Design ECMP around the traffic distribution

ECMP spreads hash keys, not bytes. A workload with millions of similarly sized short flows is easier to balance than one with a few long, unequal flows. Connection pooling, source-port reuse, NAT, tunnels, and encapsulation can reduce the entropy visible to a particular hashing device. State the fields the device hashes and measure member bytes alongside flow counts.

### Preserve a recovery path independent of the failing path

The Meta case makes this rule concrete. If health evaluation, route advertisement, management access, and recovery all depend on one backbone, a single disconnection can collapse both service and repair paths. Independence is never absolute, but dependencies should be drawn and tested.

### Prefer a discriminating command over a generic dashboard

`ip route get` answers what the local kernel would choose. `ip neigh` answers whether the on-link identity is known. `bridge fdb` answers where a bridge learned a MAC. A packet capture answers what crossed an interface. Member counters and stable flow probes answer ECMP placement. A dashboard can summarize these signals, but it cannot replace knowing which question each signal answers.

## Run it yourself

### Question

Does a Linux host resolve the final remote destination with ARP, or does it resolve only the gateway selected by its route?

### Preconditions

Use the Linux-only `netlab` base topology from [the series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url): namespace `c` has `c0` at `10.77.0.1/30`, and namespace `s` has `s0` at `10.77.0.2/30`. Run as root or with the capabilities required for namespaces, addresses, routes, and packet capture. The experiment uses `iproute2` and `ping`; record versions with `ip -Version` and `ping -V`. On macOS, run it inside a privileged Linux VM such as Lima or Colima with `NET_ADMIN`.

The treatment adds only `192.0.2.1/32` to the loopback interface in namespace `s`, enables IPv4 forwarding in that namespace, and adds one host route in namespace `c`. `192.0.2.0/24` is documentation space, so this address is deliberately not a public target.

### Preflight

```bash
set -euo pipefail

ip netns list | grep -E '^(c|s)( |$)'
ip -n c -br addr show dev c0
ip -n s -br addr show dev s0
ip -n c route show
ip -n c neigh show dev c0
ip -Version
ping -V 2>&1 | head -n 1
```

Read: namespace names, `c0` address `10.77.0.1/30`, `s0` address `10.77.0.2/30`, and the connected route `10.77.0.0/30 dev c0`. If those names or addresses differ, stop and repair the base lab rather than adapting destructive commands to an unknown interface.

Expected: both links report `UP`; the connected route exists; the neighbor table may be empty or may contain `10.77.0.2` in a valid state. Exact neighbor state can vary because prior traffic warms the cache.

### Baseline

Flush only the lab peer's neighbor entry, ask for the direct route, and generate one packet:

```bash
sudo ip -n c neigh del 10.77.0.2 dev c0 2>/dev/null || true

ip -n c route get 10.77.0.2
sudo ip netns exec c ping -c 1 -W 1 10.77.0.2
ip -n c neigh show to 10.77.0.2 dev c0
```

Read: in `ip route get`, inspect `dev` and `src`. There should be no `via` field because `10.77.0.2` is on-link. In `ip neigh`, inspect the MAC and neighbor state.

Expected: the route contains `10.77.0.2 dev c0 src 10.77.0.1` with formatting that may vary by iproute2 version. The ping should report zero percent packet loss in this isolated two-namespace lab. The neighbor should have a link-layer address and commonly be `REACHABLE`, though `STALE` or `DELAY` can appear depending on timing and kernel state.

### Apply one change

Make namespace `s` act as the gateway for one documentation address, then add a host route through it:

```bash
sudo ip -n s address replace 192.0.2.1/32 dev lo
sudo ip netns exec s sysctl -qw net.ipv4.ip_forward=1
sudo ip -n c route replace 192.0.2.1/32 via 10.77.0.2 dev c0
sudo ip -n c neigh del 10.77.0.2 dev c0 2>/dev/null || true
```

This is a controlled lab mutation. Do not apply the route, sysctl, or neighbor deletion to an unspecified production host.

### Compare

Ask the route question, capture only relevant frames, then send one packet:

```bash
sudo ip netns exec c timeout 5 \
  tcpdump -ni c0 -c 4 -e 'arp or icmp' &
CAP_PID=$!

sleep 0.2
ip -n c route get 192.0.2.1
sudo ip netns exec c ping -c 1 -W 1 192.0.2.1
wait "$CAP_PID" || true

ip -n c neigh show dev c0
```

Read: `ip route get` must name `via 10.77.0.2 dev c0 src 10.77.0.1`. In the capture, the ARP target protocol address must be `10.77.0.2`, while the ICMP echo request's destination IP is `192.0.2.1`. In the final neighbor table, look for `10.77.0.2`; there should not be a `192.0.2.1` neighbor on `c0`.

Expected: one ARP request/reply pair may appear before the ICMP exchange after the targeted flush. The echo request and reply should complete with zero percent loss in this isolated lab. The exact elapsed time varies with the host scheduler and virtualization. The qualitative result does not: the remote IP is routed through `10.77.0.2`, and only that on-link gateway is resolved to a MAC on `c0`.

If `tcpdump` starts too slowly and misses the first ARP, repeat only the targeted neighbor deletion and compare step. If ping fails, verify `net.ipv4.ip_forward=1` inside `s`, both links are up, and no lab firewall rejects forwarding.

### Reset

Remove only the state added by this experiment:

```bash
sudo ip -n c route del 192.0.2.1/32 via 10.77.0.2 dev c0 \
  2>/dev/null || true
sudo ip -n s address del 192.0.2.1/32 dev lo \
  2>/dev/null || true
sudo ip netns exec s sysctl -qw net.ipv4.ip_forward=0
sudo ip -n c neigh del 10.77.0.2 dev c0 \
  2>/dev/null || true
```

The production translation is read-only: use a flow-complete `ip route get`, inspect the chosen gateway with `ip neigh`, and capture a bounded sample with Ethernet headers. Do not toggle forwarding, add routes, or flush neighbors on a production host merely to imitate this lab.

Repeat the comparison once without deleting the neighbor entry. The second capture should usually contain the ICMP exchange without the preceding ARP request because the gateway mapping is already usable. Treat the presence or absence of ARP as the assertion, not a tiny timing difference: virtual-machine scheduling can dominate sub-millisecond observations. If ARP still repeats, record the neighbor state before changing anything. Repetition may indicate that the entry never became usable, that another process removed it, or that the lab was reset between runs. This warm repeat connects the table to the wire and prevents a successful first run from being mistaken for proof that resolution occurs before every packet.

## Key takeaways

- Route lookup happens before ARP. The route chooses an egress interface and an on-link next hop; ARP resolves that next hop to a MAC.
- A remote packet normally carries the remote destination IP inside a frame addressed to the local gateway's MAC.
- Switches learn source MAC locations, forward known destinations selectively, and constrain flooding to the relevant VLAN.
- Longest-prefix match selects the most specific route before equal-cost next-hop selection becomes relevant.
- `ip route get` is stronger evidence than visually scanning `ip route show`, especially when supplied with the real source, protocol, and ports.
- Conventional ECMP balances flows. One stable flow stays on one member, so aggregate fabric capacity does not become single-flow capacity.
- Similar application timeouts can originate at route, neighbor, VLAN, switch, or ECMP state. Inspect the first failing decision instead of clearing every table.

The capstone, [the senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model), combines these decisions with transport, name resolution, security, load balancing, and queueing. The habit to carry forward is smaller: for every packet, name the destination IP, selected route, next-hop IP, destination MAC, VLAN, and flow key. If one of those nouns is unknown, that is the next measurement.

## Further reading

- [RFC 826: An Ethernet Address Resolution Protocol](https://www.rfc-editor.org/rfc/rfc826.html), November 1982.
- [RFC 1812: Requirements for IP Version 4 Routers](https://www.rfc-editor.org/rfc/rfc1812.html), June 1995.
- [RFC 2992: Analysis of an Equal-Cost Multi-Path Algorithm](https://www.rfc-editor.org/rfc/rfc2992.html), November 2000.
- [RFC 6438: Using the IPv6 Flow Label for ECMP and Link Aggregation](https://www.rfc-editor.org/rfc/rfc6438.html), November 2011.
- [Linux `ip-route(8)` manual](https://man7.org/linux/man-pages/man8/ip-route.8.html), consulted 2026-09-29.
- [Linux resilient next-hop groups](https://docs.kernel.org/networking/nexthop-group-resilient.html), consulted 2026-09-29.
