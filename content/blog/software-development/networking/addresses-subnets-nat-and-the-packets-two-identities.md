---
title: "Addresses, Subnets, NAT, and a Packet's Two Identities"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn to calculate IPv4 prefixes, reason about IPv6 scope, and diagnose the two finite state pools hidden behind NAT."
tags:
  [
    "networking",
    "distributed-systems",
    "ipv4",
    "ipv6",
    "cidr",
    "nat",
    "conntrack",
    "ephemeral-ports",
    "linux",
    "capacity-planning",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 42
image: "/imgs/blogs/addresses-subnets-nat-and-the-packets-two-identities-1.webp"
---

A service dials `203.0.113.80:443` from `10.77.0.1:40000`. The connection succeeds, yet the server says the peer is `198.51.100.7:53012`. Both observations are correct. The socket has one identity inside the private network and another after a Network Address Translator, or NAT, rewrites it. That rewrite looks like a small header edit. Operationally, it creates state, consumes capacity, changes what logs can prove, and gives a packet two identities that must be correlated during an incident.

![A private socket becomes a public flow at the NAT boundary](/imgs/blogs/addresses-subnets-nat-and-the-packets-two-identities-1.webp)

The diagram above is the mental model: an address is meaningful within a routing scope, a prefix defines that scope, and a stateful boundary may replace the source address and port before the packet reaches its peer. We will build the arithmetic from bits, connect it to IPv6 as it is actually deployed, open the NAT state table, and derive the first two hard ceilings: connection-tracking entries and ephemeral ports.

This is the third post in [Networking for Engineers Who Ship Services](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The intro owns the complete request path and the lab setup. The next useful companions are [how a packet gets there](/blog/software-development/networking/how-a-packet-gets-there-arp-switching-routing-and-ecmp), [sockets and the two transport contracts](/blog/software-development/networking/sockets-and-the-two-transport-contracts-tcp-vs-udp), and [the layers are a useful lie](/blog/software-development/networking/the-layers-are-a-lie-but-a-useful-one). We will stay at the addressing and translation boundary here. [Load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) owns the architecture built above that boundary.

> An IP address is not a machine's permanent name. It is a locator with a scope, attached to an interface for some interval, and sometimes rewritten in flight.

## One packet has two identities

The five fields that usually identify a transport flow are source address, source port, destination address, destination port, and transport protocol. Engineers call this the five-tuple. For the outbound TCP flow in the opening example, the private-side tuple is:

```console
10.77.0.1:40000 -> 203.0.113.80:443, TCP
```

After source NAT, the public-side tuple is:

```console
198.51.100.7:53012 -> 203.0.113.80:443, TCP
```

The translator stores a mapping between them. When a reply arrives for `198.51.100.7:53012`, it consults that state and restores `10.77.0.1:40000` as the private destination. The remote server never needs to know the private address. The client application usually never sees the public port.

Those example addresses are deliberately documentation ranges. RFC 5737, published in January 2010, reserves `192.0.2.0/24`, `198.51.100.0/24`, and `203.0.113.0/24` for examples, so a tutorial does not accidentally point readers at a real subscriber. The values are not a measurement and do not describe a production network.

The rewrite is not encryption. The payload is unchanged unless a protocol-aware gateway does additional work. NAT is not authentication either. A private address does not establish trust, and a public address does not establish hostility. The boundary controls reachability as a side effect of its mappings, but a firewall policy is still a separate control.

The two identities explain several common debugging disagreements:

| Observation point | Source identity recorded | What it can establish | Source |
| --- | --- | --- | --- |
| Client socket | `10.77.0.1:40000` | Which local process opened the flow | Derived from the example tuple |
| Private-side capture | `10.77.0.1:40000` | Packet left the client before translation | Derived from the example tuple |
| Public-side capture | `198.51.100.7:53012` | Packet left the translator after rewrite | Derived from the example mapping |
| Remote access log | `198.51.100.7:53012` | Which translated flow reached the service | Derived from the example mapping |

A remote log with only the public source address may collapse thousands of private clients into one label. A remote log with address and source port is better, but only if the NAT mapping still exists or was exported when the event occurred. A client timestamp, gateway mapping, and server timestamp need synchronized clocks or a sufficiently narrow time window. This is why address translation is also an observability design problem.

## CIDR arithmetic without a calculator

![CIDR fixes the leftmost bits and leaves the rest for addresses](/imgs/blogs/addresses-subnets-nat-and-the-packets-two-identities-2.webp)

IPv4 is a 32-bit number printed as four decimal octets. Classless Inter-Domain Routing, or CIDR, appends a prefix length such as `/20`. RFC 4632, published in August 2006, defines that suffix as the count of significant leftmost bits. A `/20` fixes 20 bits and leaves 12 bits variable.

That gives the first formula:

$$
\text{addresses in an IPv4 prefix} = {2}^{32-p}
$$

Here, $p$ is the prefix length. For `10.77.16.0/20`:

$$
{2}^{32-20} = {2}^{12} = 4096\text{ addresses}
$$

This is the number of address bit patterns in the block. It is not automatically the number of assignable hosts. Traditional broadcast subnets reserve the all-zero host part as the network address and the all-one host part as the broadcast address, giving ${4096} - 2 = 4094$ conventional host addresses. Point-to-point `/31` links follow a different rule in RFC 3021, so do not apply the subtraction as an eternal law.

### Find the boundary octet

You can do most prefix arithmetic in your head with two moves.

First, count complete octets. A `/20` contains two complete octets, or 16 bits, plus four bits in the third octet. Second, compute the block step in the boundary octet:

$$
\text{step} = {2}^{8-r}
$$

Here, $r$ is the number of prefix bits in the boundary octet. With four prefix bits, the step is `${2}^{4} = 16`. Valid third-octet boundaries are therefore 0, 16, 32, 48, and so on. The address `10.77.23.9` lies between 16 and 31, so its `/20` network is `10.77.16.0` and its last address is `10.77.31.255`.

The same method works for awkward prefixes:

| Prefix | Boundary bits $r$ | Step in boundary octet | Addresses | Source |
| --- | ---: | ---: | ---: | --- |
| `/24` | 8 | 1 | `${2}^{8} = 256` | Derived here from RFC 4632 notation |
| `/23` | 7 | 2 | `${2}^{9} = 512` | Derived here from RFC 4632 notation |
| `/22` | 6 | 4 | `${2}^{10} = 1024` | Derived here from RFC 4632 notation |
| `/20` | 4 | 16 | `${2}^{12} = 4096` | Derived here from RFC 4632 notation |
| `/18` | 2 | 64 | `${2}^{14} = 16384` | Derived here from RFC 4632 notation |
| `/16` | 8 | 1 in the third octet | `${2}^{16} = 65536` | Derived here from RFC 4632 notation |

For `/16`, the boundary is exactly between octets, so the third and fourth octets are both host bits. The table says a step of one in the third octet because every third-octet value belongs to the same block until the first two octets change.

### Membership is a mask operation

The formal test is still useful because it maps directly to routing code. Construct a mask with $p$ leading one bits and compare the masked addresses:

$$
(A \mathbin{\&} M) = (N \mathbin{\&} M)
$$

$A$ is the candidate address, $N$ is the advertised network, and $M$ is the prefix mask. This is an explanatory equality for membership, not an equation copied from an RFC. For a `/20`, the dotted mask is `255.255.240.0` because the boundary octet is binary `11110000`, or 240.

You rarely need to convert the full address to binary. Use the step to locate the interval, then use `ipcalc` or a language library to check your work:

```bash
python3 - <<'PY'
import ipaddress

net = ipaddress.ip_network("10.77.23.9/20", strict=False)
print("network", net.network_address)
print("broadcast", net.broadcast_address)
print("addresses", net.num_addresses)
print("contains", ipaddress.ip_address("10.77.31.200") in net)
print("outside", ipaddress.ip_address("10.77.32.1") in net)
PY
```

Expected output is `10.77.16.0`, `10.77.31.255`, `4096`, `True`, then `False`. This is deterministic arithmetic, not a benchmark.

### Prefixes answer two different questions

An interface assignment such as `10.77.0.1/30` says both "my address is `10.77.0.1`" and "destinations in `10.77.0.0/30` are on-link." A route such as `10.77.0.0/16 via 10.0.0.1` says "send this whole destination prefix to a next hop." Confusing assignment with route is a frequent cause of a host attempting neighbor discovery for a destination that should have gone through a gateway.

Routing uses longest-prefix match. If a table has `10.0.0.0/8` and `10.77.0.0/16`, then `10.77.3.4` matches both, but `/16` wins because 16 fixed bits are more specific than 8. The following post on [ARP, switching, routing, and ECMP](/blog/software-development/networking/how-a-packet-gets-there-arp-switching-routing-and-ecmp) follows that decision all the way to the next hop.

## Private is a scope, not a security property

![Address families define routing scope rather than trust](/imgs/blogs/addresses-subnets-nat-and-the-packets-two-identities-3.webp)

RFC 1918, published in February 1996, reserves three IPv4 blocks for private internets: `10.0.0.0/8`, `172.16.0.0/12`, and `192.168.0.0/16`. The exact ranges matter. `172.16.0.0` through `172.31.255.255` is private. `172.32.0.0` is not. Treating all of `172.0.0.0/8` as private creates routing and filtering bugs.

The address counts are derived directly from prefix length:

| Block | Address count | Globally routed | Intended scope | Source |
| --- | ---: | --- | --- | --- |
| `10.0.0.0/8` | `${2}^{24} = 16,777,216` | No | One cooperating private network | [RFC 1918, February 1996](https://www.rfc-editor.org/rfc/rfc1918.html) |
| `172.16.0.0/12` | `${2}^{20} = 1,048,576` | No | One cooperating private network | [RFC 1918, February 1996](https://www.rfc-editor.org/rfc/rfc1918.html) |
| `192.168.0.0/16` | `${2}^{16} = 65,536` | No | One cooperating private network | [RFC 1918, February 1996](https://www.rfc-editor.org/rfc/rfc1918.html) |
| `100.64.0.0/10` | `${2}^{22} = 4,194,304` | No across provider boundaries | Shared address space between subscriber and carrier NAT | [RFC 6598, April 2012](https://www.rfc-editor.org/rfc/rfc6598.html) |

The fourth row is not RFC 1918 space. RFC 6598 created `100.64.0.0/10` for carrier-grade NAT, commonly abbreviated CGN. An operator that casually uses it as ordinary private space may collide with an access provider that uses the same range between customer equipment and its translator.

Private means "not globally unique and not globally routed." It does not mean "safe." Two companies can both assign `10.0.0.7`, then discover the ambiguity when a merger, VPN, or private interconnect joins their networks. The packet cannot express which organization's `10.0.0.7` the sender intended. Renumbering, translation between overlapping spaces, or a new addressing plan becomes necessary.

Likewise, possession of a private source address proves very little. An attacker already inside the routing domain can use one. A proxy can originate one. A workload can be compromised. Authorization must bind to an authenticated identity and policy, not to the comforting shape of `10.x.y.z`. The [service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) layer can tell a client where a service is, but it does not change these routing semantics.

### Special does not always mean private

Loopback, link-local, documentation, multicast, and benchmark ranges also have special semantics. Do not build a binary classifier called `is_private` and assume its `false` branch means globally reachable. The IANA special-purpose registries are the right source for classification. In code, prefer a maintained address library and test the exact property you need: global reachability, unicast eligibility, routability at a boundary, or membership in an organization-owned prefix.

This distinction matters in allowlists. A rule that accepts all non-public addresses may unexpectedly admit loopback or link-local destinations, enabling server-side request forgery against local agents and metadata services. The secure question is not "is this address private?" It is "is this exact destination and resolution result allowed for this request?"

### Plan prefixes as a hierarchy

A good address plan mirrors ownership and routing boundaries. Suppose an organization allocates `10.64.0.0/10` to one global environment. It can divide that block into four `/12` regional aggregates because the prefix grows by two bits and therefore creates `${2}^{2} = 4` children. Those children begin at `10.64.0.0/12`, `10.80.0.0/12`, `10.96.0.0/12`, and `10.112.0.0/12`. The step is 16 in the second octet because a `/12` fixes four bits there.

Each region could then reserve `/16` units for virtual networks. Growing from `/12` to `/16` adds four subnet bits, so one regional aggregate contains `${2}^{4} = 16` non-overlapping `/16` blocks. A router outside the region can retain one `/12` route while routers inside know the more specific `/16` routes. That is the scaling purpose of aggregation.

Do not interpret this worked plan as a universal recommendation. The parent block, region count, workload density, cloud constraints, and interconnect model are illustrative. The arithmetic is derived here. The important habit is to allocate from a parent with a documented owner instead of selecting a familiar prefix independently.

The plan should answer these questions before any instance launches:

- Which parent prefix owns this allocation?
- Which bits encode region, environment, or failure domain?
- Can routes be summarized at a stable boundary?
- How much contiguous space remains for growth?
- What happens when this network connects to an acquisition, partner, or developer VPN?
- Are service virtual IPs, nodes, pods, and load balancers drawn from distinct pools?

Kubernetes makes overlap especially painful because nodes, pods, services, and external destinations can occupy different logical spaces. A packet whose destination matches a local pod or service prefix may never follow the path the application owner expected. Translation can paper over one overlap, but it makes captures and policy more difficult. Preventing overlap in the allocation system is cheaper than debugging overlapping identity later.

### Derive child counts and reserve room

If a parent prefix has length $p$ and each child has length $q$, where $q$ is greater than $p$, the number of equal child prefixes is:

$$
\text{children} = {2}^{q-p}
$$

A `/20` split into `/24` children yields `${2}^{24-20} = 16` subnets. Each `/24` contains `${2}^{8} = 256` address patterns. Sixteen times 256 returns the parent's 4,096 patterns, which is a useful arithmetic check.

Address plans often reserve capacity rather than maximize immediate utilization. That is not automatically waste. Contiguous free space preserves aggregation and lets a failure domain grow without renumbering. The trade-off is between local packing efficiency and future routing simplicity. Record the reservation as an intentional capacity decision so a later cleanup project does not fill it opportunistically.

An IP address may change when an interface is recreated, a lease expires, a pod moves, or a failover occurs. DNS and service discovery give applications a level of indirection, but their caches create a separate staleness window. Hard-coded addresses turn an allocation detail into an API contract. Use address management for allocation, discovery for location, and authenticated workload identity for authorization.

## IPv6 is deployed, but the transition is not finished

IPv6 expands addresses from 32 bits to 128 bits. RFC 4291, published in February 2006, writes an address as eight 16-bit hexadecimal groups and permits one run of zero groups to be compressed with `::`. The loopback address is `::1/128`; link-local unicast begins with `fe80::/10`; and global unicast is structured as a routing prefix, subnet identifier, and interface identifier.

The size difference is easiest to state without pretending every bit is freely assignable:

$$
\frac{{2}^{128}}{{2}^{32}} = {2}^{96}
$$

The IPv6 identifier space has `${2}^{96}` times as many bit patterns as IPv4. That does not mean one operator receives all of them. Allocation policy and subnet conventions determine usable structure. A common `/64` leaves 64 interface-identifier bits, or `${2}^{64}` patterns, on one subnet. The engineering point is not to pack that subnet tightly. It is to make renumbering, autoconfiguration, and aggregation tractable without depending on widespread address sharing.

IPv6 is also not a future-only protocol. Google's public adoption graph continuously measures the share of users reaching Google over IPv6. Its snapshot for September 5, 2026 reported 50.30 percent, scoped to Google users rather than every Internet endpoint. That is enough deployment to make IPv6 a production path, and enough remaining IPv4 to make dual-stack behavior a production concern. See [Google IPv6 statistics, accessed September 29, 2026](https://www.google.com/intl/en/ipv6/statistics.html).

### Dual stack creates a race, not a clean migration flag

A dual-stack name can return both an A record for IPv4 and an AAAA record for IPv6. The client then needs a connection strategy. Modern "Happy Eyeballs" algorithms avoid waiting through a long failure on one family before trying the other. This means two users with the same DNS answers may take different address-family paths depending on local connectivity, timing, and cached history.

For operations, separate the paths in telemetry. Record address family, chosen destination, connect duration, and error class. A single success-rate line can hide IPv6 failures behind quick IPv4 fallback. Conversely, a service may appear healthy from an IPv4-only probe while real dual-stack clients spend extra time failing the preferred path.

IPv6 often restores end-to-end addressing, but it does not eliminate stateful firewalls, load balancers, proxies, or address translation in every design. NAT64, for example, lets IPv6-only clients reach IPv4 services through algorithmic address synthesis and stateful translation. Treat "IPv6 means no state in the path" as an unsafe assumption.

### Address scope still matters

IPv6 link-local addresses work only on one link and often require an interface zone such as `%eth0` in textual commands. Unique-local addresses come from `fc00::/7` under RFC 4193 and are not expected on the global Internet. Global unicast addresses are globally unique, but reachability still depends on routing and policy. The same discipline applies across both families: classify scope, inspect the chosen route, and observe the packet at the boundary you care about.

Useful read-only commands are:

```bash
ip -4 address show
ip -6 address show scope global
ip -6 address show scope link
ip -4 route get 203.0.113.80
ip -6 route get 2001:db8::80
getent ahosts example.com
```

The `src` field in `ip route get` is the kernel's selected source address for that destination under the current policy. It does not prove the packet passed through a NAT later. For that, capture on both sides of the translator or inspect its mapping state.

## NAT is a state machine

The simplest description of source NAT is "replace the source address." That is incomplete. A useful implementation must choose a translated address and usually a translated port, remember the original tuple, recognize later packets as members of the same flow, reverse the translation for replies, and eventually reclaim the mapping. That is a state machine.

RFC 3022, published in January 2001, calls the address-only form Basic NAT and the address-and-port form Network Address Port Translation, or NAPT. Everyday usage often calls both NAT. Linux implements translation on top of Netfilter connection tracking, commonly shortened to conntrack.

The animated figure below follows the state that a static before-and-after picture hides. The first outbound packet has no mapping. The translator allocates one and records both directions. A reply matches the public tuple and is rewritten toward the private socket. After the flow is closed or idle long enough for the relevant protocol state, the entry expires and its capacity can be reused.

<figure class="blog-anim">
<svg viewBox="0 0 896 300" role="img" aria-label="NAT state machine moving from no entry to allocation, reply match, and idle expiry" style="width:100%;height:auto;max-width:896px">
<style>
.nat3-stage{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.nat3-active{fill:var(--accent,#6366f1);opacity:.18}.nat3-title{font:700 20px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.nat3-label{font:600 15px ui-monospace,SFMono-Regular,monospace;fill:var(--text-primary,#1f2937);text-anchor:middle}.nat3-note{font:500 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.nat3-arrow{stroke:var(--text-secondary,#6b7280);stroke-width:2;fill:none;marker-end:url(#nat3-arrowhead)}
@keyframes nat3-walk{0%,18%{transform:translateX(0)}25%,43%{transform:translateX(214px)}50%,68%{transform:translateX(428px)}75%,93%{transform:translateX(642px)}100%{transform:translateX(0)}}
@keyframes nat3-entry{0%,18%{opacity:.18}25%,68%{opacity:1}75%,100%{opacity:.18}}
.nat3-sweep{animation:nat3-walk 10s steps(1,end) infinite}.nat3-state{animation:nat3-entry 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.nat3-sweep,.nat3-state{animation:none}.nat3-sweep{transform:translateX(428px)}.nat3-state{opacity:1}}
</style>
<defs><marker id="nat3-arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="var(--text-secondary,#6b7280)"/></marker></defs>
<text class="nat3-title" x="448" y="32">NAT is a state machine</text>
<path class="nat3-arrow" d="M200 128 H238"/><path class="nat3-arrow" d="M414 128 H452"/><path class="nat3-arrow" d="M628 128 H666"/>
<rect class="nat3-stage" x="24" y="72" width="176" height="112" rx="10"/><rect class="nat3-stage" x="238" y="72" width="176" height="112" rx="10"/><rect class="nat3-stage" x="452" y="72" width="176" height="112" rx="10"/><rect class="nat3-stage" x="666" y="72" width="176" height="112" rx="10"/>
<rect class="nat3-active nat3-sweep" x="24" y="72" width="176" height="112" rx="10"/>
<text class="nat3-label" x="112" y="116">no entry</text><text class="nat3-note" x="112" y="146">tuple unknown</text>
<text class="nat3-label" x="326" y="108">outbound SYN</text><text class="nat3-note" x="326" y="138">allocate mapping</text><text class="nat3-note nat3-state" x="326" y="160">entry present</text>
<text class="nat3-label" x="540" y="108">reply matches</text><text class="nat3-note" x="540" y="138">reverse translation</text><text class="nat3-note nat3-state" x="540" y="160">entry reused</text>
<text class="nat3-label" x="754" y="108">idle expiry</text><text class="nat3-note" x="754" y="138">delete mapping</text><text class="nat3-note" x="754" y="160">slot released</text>
<text class="nat3-note" x="448" y="226">The first packet creates state; later packets depend on it until timeout.</text>
</svg>
<figcaption>Motion carries the state transition: the first packet allocates a mapping, the reply matches it, and idle expiry removes it.</figcaption>
</figure>

The lookup needs both views. A conceptual entry for our running example contains:

```console
original:  TCP 10.77.0.1:40000     -> 203.0.113.80:443
translated: TCP 198.51.100.7:53012 -> 203.0.113.80:443
reply seen: TCP 203.0.113.80:443    -> 198.51.100.7:53012
deliver to: TCP 203.0.113.80:443    -> 10.77.0.1:40000
```

That entry is explanatory, not copied command output. Real `conntrack -L` formatting depends on protocol, state, tool version, and selected output options. The invariant is the relationship among original and reply tuples.

### What creates a mapping

For TCP, the first SYN commonly creates tracked state. The tracker then observes enough of the handshake and later flags to move through protocol-specific states. Conntrack is not the application socket table, and it is not a full TCP implementation. It observes packets crossing a hook and maintains the state needed for filtering and translation.

UDP has no handshake and no FIN. A translator infers a flow from tuples and timers. Two applications can therefore disagree about whether a UDP conversation is still alive while the translator has already discarded its mapping. A later packet may allocate a different public port. Protocols that need stable reachability send keepalives or refresh traffic, but every keepalive spends bandwidth, wakes devices, and prolongs state. The correct interval depends on the path, not on a universal NAT timeout.

ICMP also needs matching logic. An ICMP error often quotes the beginning of the packet that caused it, and a translator may need to rewrite the embedded header so the private endpoint can associate the error with its flow. This is one example of why "change one address" is not a sufficient operational model.

### Mapping behavior changes reachability

Translators differ in how strongly a mapping is tied to a remote endpoint. A mapping might be reusable for traffic to any destination, restricted to one destination address, or restricted to one destination address and port. Filtering behavior can be equally important for inbound packets. RFC 4787, published in January 2007, defines behavioral requirements for unicast UDP NATs and precise terms for these variations.

This matters to peer-to-peer and real-time systems. A public mapping learned through one rendezvous server might accept packets from a peer on one NAT and reject them on another. The application-level consequence is not "UDP is unreliable" in the abstract. It is that reachability depends on observable mapping and filtering behavior, and the fallback path may need a relay.

For ordinary client-to-server traffic, the safer assumption is narrower: an outbound packet creates state, replies that match the state can return, unsolicited inbound packets do not automatically gain a mapping, and the entry has a finite lifetime. Confirm vendor behavior before relying on anything more specific.

### Translation is directional

Source NAT changes the source tuple, usually for outbound traffic. Destination NAT changes the destination tuple, commonly to publish a private service behind a public address or virtual IP. A single packet path can encounter both. A load balancer might rewrite the destination on ingress, while the backend's return path uses source translation so replies traverse the same device.

Direction matters during capture analysis. On an ingress destination-NAT boundary, the public destination exists before translation and the private backend destination exists after it. On egress source NAT, the private source exists before translation and the public source exists after it. Saying "capture before NAT" without naming direction and interface is ambiguous.

Hairpin NAT adds another case. A private client connects to the public address of a service that is also private. The gateway translates the destination toward the internal server and may translate the source so the reply returns through the gateway. Without symmetric handling, the server can reply directly to the client from a private address the client did not dial, and the transport tuple no longer matches.

This yields a practical capture plan:

1. Record the tuple at the application socket.
2. Capture on the private side of the translation boundary.
3. Capture on the public or post-translation side.
4. Query the mapping table with a tuple and narrow time window.
5. Confirm that replies traverse a boundary able to reverse the mapping.

One capture is often insufficient. If the private trace contains a SYN and the public trace does not, the boundary is implicated. If both contain corresponding SYNs but no reply arrives, move toward the destination. If the public reply exists but the private reply does not, inspect reverse lookup, state expiry, asymmetric routing, and filtering at the translator.

### Rewriting reaches beyond visible fields

IPv4 has a header checksum. TCP and UDP checksums cover a pseudo-header that includes IP addresses. Changing an address or port therefore requires checksum adjustment. Implementations usually do this efficiently, sometimes with offload assistance, but packet captures can be confusing when captured before a NIC computes an outbound checksum.

A tool may label such a packet checksum incorrect even though the packet put on the wire is valid. Check capture location and offload settings before diagnosing corruption. Compare with a capture beyond the transmitting interface when possible. The observation point changes not only the visible tuple but sometimes the apparent completion of packet processing.

Only the first IPv4 fragment contains the transport header with ports. Later fragments carry the IP identification and offset but not the TCP or UDP port fields. A translator needs enough fragment association state to apply a consistent rewrite. Dropped first fragments, reassembly pressure, or inconsistent paths can therefore produce failures that look unrelated to NAT at the application layer.

Avoid relying on fragmentation as a normal transport strategy. Path MTU Discovery and protocol-level sizing are covered later in the series. For this post, remember that a five-tuple is easiest to classify when the packet containing the transport header is available at the stateful boundary.

### NAT does not preserve end-to-end identity

The remote service sees the translator as the network peer. If many clients share one public address, address-based rate limiting aggregates independent users. If a blocklist rejects that address, it rejects the whole population behind it. If a forensic process has only the public address and omits the source port and time, it may be impossible to identify the originating private flow.

This does not make NAT inherently bad. It means the address is the wrong identity primitive for many application policies. Authenticate requests at the appropriate layer, propagate trace context carefully, and retain privacy-conscious translation logs only for the period justified by the operational or legal requirement. [Observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) owns the broader telemetry architecture.

## Conntrack is the first NAT ceiling

![Conntrack retains original and reply tuples until cleanup](/imgs/blogs/addresses-subnets-nat-and-the-packets-two-identities-4.webp)

Every active tracked flow consumes an entry. The first capacity ratio to inspect on a Linux translator is an explanatory utilization model:

$$
U_c = \frac{C}{C_{max}}
$$

$C$ is `nf_conntrack_count`, the number of currently allocated flow entries. $C_{max}$ is `nf_conntrack_max`, the configured maximum. The Linux kernel documentation identifies the first as read-only and the second as the maximum allowed entries. This ratio is a diagnostic model, not a protocol formula and not a claim that failure begins at one universal percentage.

Read both values without changing the host:

```bash
count=$(cat /proc/sys/net/netfilter/nf_conntrack_count)
limit=$(cat /proc/sys/net/netfilter/nf_conntrack_max)
awk -v c="$count" -v m="$limit" 'BEGIN {
  printf "count=%d max=%d utilization=%.1f%%\n", c, m, 100*c/m
}'
```

If the files do not exist, connection tracking may be unavailable, unloaded, namespaced differently, or hidden by the container environment. Do not treat missing visibility as zero utilization.

The kernel's current documentation also explains that the conntrack hash table stores entries for both original and reply directions. Its note about average hash-chain length is an implementation detail, not evidence that one user flow should be counted twice in `nf_conntrack_count`. Use the exported count and maximum as documented instead of deriving a flow count from bucket internals.

### Capacity is arrivals multiplied by lifetime

Little's Law gives a useful approximation for steady state:

$$
C \approx \lambda W
$$

$\lambda$ is the average arrival rate of new tracked flows in flows per second, and $W$ is their average residence time in seconds. This is an explanatory capacity model. It assumes a stable observation interval and says nothing about burst distribution.

Suppose a gateway sees a steady 8,000 new short flows per second and entries remain for an average of 45 seconds across active time and cleanup. The expected population is:

$$
8000\ \frac{\text{flows}}{\text{s}} \times 45\ \text{s} = 360{,}000\ \text{flows}
$$

If a retry storm doubles arrivals to 16,000 flows per second without changing residence time, the model predicts 720,000 entries. Bandwidth might remain below line rate because many failed handshakes carry few bytes. State can exhaust before the traffic graph looks large.

This is the first important NAT lesson: capacity follows flow churn and lifetime, not just bit rate. A million long-lived idle connections and a million rapid failed handshakes stress different code paths, but both can occupy finite tracking state.

### Bursts need headroom beyond the average

Little's Law describes an average population. It does not size a burst safely by itself. Suppose the steady population is 360,000 entries, and a deployment causes an extra 50,000 connection attempts per second for 8 seconds. If entries created by those attempts live through the whole burst, the added state is approximately:

$$
50{,}000\ \frac{\text{flows}}{\text{s}} \times 8\ \text{s} = 400{,}000\ \text{entries}
$$

The transient population can reach roughly 760,000 before cleanup catches up. That is a derived rectangular-burst model. Real arrivals and cleanup overlap, so measure a time series at a resolution that can see the burst. A five-minute average can flatten an eight-second exhaustion event into an innocent-looking line.

Retries can make the burst endogenous. A lost initial connection triggers a retry, the retry allocates or attempts more state, the added load increases loss, and clients retry again. Jitter, retry budgets, circuit breaking, and connection reuse break that feedback loop at higher layers. The gateway still needs headroom for the residual burst.

Track both the level and the derivative. A table at 60 percent utilization with a steep positive slope may offer less response time than one stable at 85 percent. Alerting only on a fixed utilization threshold misses that distinction.

### Distribution matters across gateways

Two gateways with equal configured maxima do not guarantee equal load. Routing, consistent hashing, availability-zone affinity, long-lived flows, or one hot source prefix can skew state. Aggregate utilization can be 50 percent while one gateway is full.

Report per-device count, maximum, new-flow rate, drop or insertion-failure counters, and source or destination concentration. When scaling out, verify that new flows actually reach the new capacity. Existing mappings may remain pinned to their original gateway, which is usually necessary for reverse translation.

Stateful failover has its own cost. If mappings are not replicated, a gateway failure resets or blackholes active flows until clients reconnect. If mappings are replicated, the replication channel, lag, ordering, and failover semantics become part of capacity and correctness. There is no free conversion from stateful translation to stateless availability.

### What failure looks like

At or near the limit, new flows may fail while established flows continue. Depending on topology and rules, symptoms can include dropped first packets, connection timeouts, kernel messages about a full table, or counters that show insertion failures. Application dashboards may misclassify this as a remote dependency outage because existing pooled connections work and only new connection attempts stall.

Use several signals together:

```bash
cat /proc/sys/net/netfilter/nf_conntrack_count
cat /proc/sys/net/netfilter/nf_conntrack_max
conntrack -S
nstat -az | grep -E 'Conntrack|Listen|Timeout'
journalctl -k --since '-10 min' | grep -i conntrack
```

`conntrack -S` usually requires elevated privileges. Field names vary with conntrack-tools and kernel versions, so record `conntrack -V` and `uname -r` with an incident snapshot. Read the count-to-max trend, failed insert indicators when exposed, and the time correlation with new-connection errors. A single high count without failures is a capacity warning, not proof of the root cause.

### Raising the maximum moves the trade-off

Increasing `nf_conntrack_max` can create headroom, but it also retains more kernel state. Memory cost varies with kernel configuration, enabled extensions, accounting, architecture, and allocator overhead. A universal "bytes per entry" constant would be misleading. Measure on the kernel and rule set you operate.

The safe sequence is:

1. Confirm that the table is the constrained resource.
2. Identify arrival rate and residence-time contributors.
3. Reduce unnecessary new flows through connection reuse when semantics allow it.
4. Fix retry amplification and unreachable destinations.
5. Validate memory and lookup behavior before raising the limit.
6. Scale the number of translators or public identities when one failure domain is too concentrated.

Changing timeouts can reclaim state sooner, but too-short timeouts break legitimate idle flows and UDP mappings. Raising a limit without fixing an unbounded retry storm makes the crash take longer and potentially consume more memory. [Timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) covers the policy above the wire.

## Ephemeral ports are the second NAT ceiling

![Port allocation changes when the kernel can reuse a source tuple](/imgs/blogs/addresses-subnets-nat-and-the-packets-two-identities-5.webp)

A TCP connection is unique by its transport tuple. A client that does not request a source port lets the kernel choose one from the ephemeral range. On current Linux documentation, `ip_local_port_range` defaults to 32768 through 60999, inclusive. Always read the running namespace instead of assuming that default.

The raw range size is:

$$
R = h - l + 1
$$

$l$ is the low endpoint and $h$ is the high endpoint. For the documented default:

$$
60999 - 32768 + 1 = 28232\ \text{ports}
$$

Reserved local ports and ports already bound for incompatible uses reduce availability. Selection and reuse also depend on protocol, socket options, connection state, destination tuple, and kernel behavior. Therefore 28,232 is a range size, not a universal connection limit.

Read the effective inputs:

```bash
sysctl net.ipv4.ip_local_port_range
sysctl net.ipv4.ip_local_reserved_ports
ss -s
ss -tan state established
```

The source port can be reused across distinct destinations when the complete tuples remain unique. For TCP connections from one source address to one fixed destination address and port, however, each simultaneous connection needs a distinct source port. The simple upper bound is then:

$$
\text{connections to one destination tuple} \le R - R_{reserved} - R_{unavailable}
$$

This is a derived bound, not a guarantee. `TIME_WAIT`, explicit binds, listening sockets, policy, and allocation search behavior can reduce the practically available set.

### NAT adds another port allocator

The host chooses a private source port. A NAPT device may then choose a public source port. These are separate pools at separate boundaries. A host can have local ports available while the gateway cannot allocate a collision-free public mapping. Conversely, the gateway can have ample capacity while one client process exhausts its local range toward a single upstream tuple.

For a public address with $P$ usable translated ports and $D$ identical destination tuples, a simplistic planning model might suggest up to $P \times D$ unique public four-tuples if the translator reuses source ports across destinations. Real behavior is narrower because mappings, filtering modes, reserved ports, hash selection, collision handling, timeouts, and per-subscriber quotas differ. Label such multiplication as a model and validate it against the actual gateway.

Adding public source addresses increases tuple space. If $A$ addresses each offer an equivalent usable port pool $P$, the idealized address-port combinations are:

$$
A \times P
$$

Again, this is an upper-bound model. Distribution may be uneven, one destination can impose its own acceptance limits, and conntrack may fill first. Capacity planning has to take the minimum of several limits, not celebrate the largest one.

### `TIME_WAIT` is not pointless debris

After an active TCP close, the endpoint may keep a `TIME_WAIT` record so delayed segments from an old connection cannot be mistaken for a new connection with the same tuple. Aggressively removing that protection to reclaim ports can trade a visible capacity problem for subtle correctness failures.

Connection reuse is usually the first lever for request-oriented protocols. A pool amortizes handshakes, reduces tuple churn, and reduces conntrack arrivals. The pool also needs bounds, health checks, idle eviction, and backpressure. Unlimited keepalive merely converts port churn into persistent socket and upstream-capacity consumption.

HTTP/2 and HTTP/3 can multiplex requests over fewer transport connections, but multiplexing changes failure coupling and congestion behavior. The later protocol posts own those trade-offs. The address-layer lesson is simply that a request count and a connection count are different capacity dimensions.

### Work the bound from workload demand

Start with concurrency rather than requests per second. If one process sends 2,000 requests per second to a single destination and the mean connection lifetime is 30 seconds because each request opens a fresh connection and then remains in a closing state relevant to tuple reuse, the explanatory population model is:

$$
2000\ \frac{\text{connections}}{\text{s}} \times 30\ \text{s} = 60{,}000\ \text{connection tuples}
$$

That exceeds a 28,232-port pool for one source address and one destination tuple. The result is a warning from the model, not a prediction of the exact failure second. Connection close behavior, kernel reuse rules, request duration, and the configured range need measurement.

If the application instead reuses 400 established connections and sequences or multiplexes requests over them, its port population can remain near 400 while request throughput stays 2,000 per second. The derived average is five requests per second per connection. That change cuts allocator pressure, but it transfers responsibility to pool scheduling and per-connection backpressure.

Adding a second source address ideally doubles source address-port combinations for that destination. Yet a policy route, explicit bind, container namespace, or gateway may keep traffic on only one address. Prove distribution with `ss`, `ip route get`, and captures. An address configured on an interface is not useful capacity until socket selection and the NAT path use it.

### Containers add more allocation boundaries

The application may read an ephemeral range inside its network namespace, while the node performs another source translation for pod egress, and a cloud gateway performs a third translation before the Internet. Each boundary can have its own tuple allocator and conntrack table.

When a pod reports `EADDRNOTAVAIL`, inspect its namespace first. When the pod emits a SYN that disappears at the node, inspect node translation and tracking. When the node emits a translated SYN that disappears before the remote peer, inspect the upstream gateway. A cluster-wide dashboard that sums all three layers can hide which allocator returned the failure.

This is why a packet can have more than two identities in a container platform. "Two" is the minimum useful model: before and after one stateful boundary. A real path may chain pod address, node address, load-balancer address, and provider egress address. Write each tuple and boundary explicitly instead of calling all of them "the client IP."

## A dated public case: Cloudflare's ephemeral-port ceiling

On February 2, 2022, Cloudflare engineer Marek Majkowski published [How to stop running out of ephemeral ports and start to love long-lived connections](https://blog.cloudflare.com/how-to-stop-running-out-of-ephemeral-ports-and-start-to-love-long-lived-connections/). The write-up reports production symptoms including `ssh` to localhost failing with `Cannot assign requested address` and a DNS query failing with `address in use`. Cloudflare attributed both examples to ephemeral-port exhaustion.

The important mechanism was not merely "many connections." Cloudflare needed applications to select source addresses so different traffic classes would use the intended egress identities. A straightforward `bind(src_IP, 0)` before `connect()` forced the kernel to reserve a unique local address-port pair before it knew the destination. That prevented source two-tuple reuse across different destinations and constrained total concurrent connections by the ephemeral range.

The article shows the Linux range `32768 60999` on its example system and derives 28,232 ports with inclusive arithmetic. It contrasts three behaviors:

| Connection technique | Exhaustion error reported | Source address-port reuse across destinations | Source |
| --- | --- | --- | --- |
| Plain TCP `connect()` | `EADDRNOTAVAIL` | Yes | [Cloudflare, February 2, 2022](https://blog.cloudflare.com/how-to-stop-running-out-of-ephemeral-ports-and-start-to-love-long-lived-connections/) |
| `bind(src_IP, 0)` then `connect()` | `EADDRINUSE` | No | [Cloudflare, February 2, 2022](https://blog.cloudflare.com/how-to-stop-running-out-of-ephemeral-ports-and-start-to-love-long-lived-connections/) |
| `IP_BIND_ADDRESS_NO_PORT`, bind, then connect | `EADDRNOTAVAIL` | Yes | [Cloudflare, February 2, 2022](https://blog.cloudflare.com/how-to-stop-running-out-of-ephemeral-ports-and-start-to-love-long-lived-connections/) |

The option `IP_BIND_ADDRESS_NO_PORT` tells Linux to delay port reservation until `connect()` supplies the destination. The kernel can then select a port while considering the full tuple and reuse a local address-port pair against distinct destinations when safe. Current Linux IP sysctl documentation also names this option as the preferred solution over enabling broad automatic bind reuse.

The transfer lesson is precise. Before expanding a port range or adding source addresses, inspect how the application creates sockets. Two programs with the same destination set and connection count can consume radically different tuple capacity because one reserves ports too early. The syscall error distinguishes useful branches: `EADDRNOTAVAIL`, `EADDRINUSE`, and a network timeout do not point to the same allocator.

This is a public engineering case, not a claim that every port incident has Cloudflare's cause. Its reported symptoms and figures are bounded to the Linux behaviors and application patterns described in that February 2022 article. Reproduce your own kernel and workload before applying the remedy.

## Diagnose the ceiling before changing it

![A read-only decision tree separates three state pools](/imgs/blogs/addresses-subnets-nat-and-the-packets-two-identities-6.webp)

The fastest diagnosis starts at the failing boundary and moves outward. Do not begin by widening every range.

### Branch 1: did the local syscall fail before a packet left?

Capture the actual error. `EADDRNOTAVAIL` on TCP connect toward one destination can indicate no suitable local tuple. `EADDRINUSE` after explicit bind suggests a different allocation conflict. Confirm with a small reproducer and a packet capture scoped to the destination. If no SYN leaves while the syscall immediately returns an allocation error, remote routing is not the first hypothesis.

Read:

```bash
sysctl net.ipv4.ip_local_port_range
sysctl net.ipv4.ip_local_reserved_ports
ss -s
ss -tan state time-wait | wc -l
ss -tan state established | wc -l
```

The `wc -l` values include a header on some `ss` versions and are snapshots, not precise time-series counters. Use them to establish scale and state distribution, then collect application error rates and socket lifecycle telemetry.

### Branch 2: did the gateway reject or drop a new flow?

If the SYN leaves the client but does not emerge from the expected egress boundary, compare conntrack count and maximum, inspect insertion/drop counters, and query the exact NAT device. On Linux:

```bash
printf 'count='; cat /proc/sys/net/netfilter/nf_conntrack_count
printf 'max='; cat /proc/sys/net/netfilter/nf_conntrack_max
conntrack -S
sudo conntrack -L -p tcp --dport 443 -o extended | head -50
```

Listing a large table can be expensive and exposes network metadata. Filter aggressively, limit output, and avoid running an unbounded listing during an incident. Packet captures may contain credentials or payloads. Use a narrow BPF filter, a duration limit, and approved storage.

### Branch 3: are local resources healthy but the managed translator is saturated?

Cloud NAT gateways and firewalls expose platform-specific metrics for allocated ports, active connections, dropped flows, or source-address use. Names and semantics change, so consult the current provider documentation for the exact region, SKU, and metric definition. Correlate the gateway's event time with client attempts and remote observations.

A healthy local range does not clear the upstream allocator. A healthy upstream port pool does not clear conntrack. A low average does not clear a per-destination or per-subscriber quota. Draw every allocator on the path:

| Resource | Owned by | Exhaustion clue | Safe first action | Source |
| --- | --- | --- | --- | --- |
| Local ephemeral ports | Client kernel namespace | Immediate allocation errno, no outbound SYN | Inspect range, reservations, socket construction, and states | Linux kernel IP sysctl docs, current tree accessed 2026-09-29 |
| Conntrack entries | Stateful host or gateway | Count approaches max, new-flow insertion failures | Reduce churn, verify timeouts, validate memory, scale state | Linux conntrack sysctl docs, current tree accessed 2026-09-29 |
| Translated public ports | NAT gateway | Local SYN exists, mapping allocation or egress fails | Inspect per-address and per-destination allocation, add identities only after proof | Device or provider documentation for deployed version |
| Remote accept capacity | Destination service | SYN reaches peer but is dropped, reset, or queued | Inspect listener and upstream policy | Reproduce with captures at both boundaries |

The table contains no universal threshold. That is intentional. The same utilization percentage can be safe under stable long-lived flows and dangerous under a steep burst of short flows.

### Correlate one failed flow end to end

During an incident, pick one connection attempt instead of starting with aggregates. Ask the client to log a monotonic attempt identifier, wall-clock timestamp, destination name, resolved addresses, selected destination, local tuple after `connect()`, and errno. If policy permits, capture only the target host and port for a short interval.

Suppose the client reports this private tuple:

```console
2026-09-29T08:15:12.240Z attempt=8f31
TCP 10.77.0.1:40000 -> 203.0.113.80:443
```

The translator log reports a mapping at the same boundary:

```console
2026-09-29T08:15:12.241Z
TCP 10.77.0.1:40000 -> 198.51.100.7:53012
destination 203.0.113.80:443
```

The server reports:

```console
2026-09-29T08:15:12.268Z
peer 198.51.100.7:53012 local 203.0.113.80:443
```

These are illustrative records for the documentation addresses, not claimed production output. They show the join keys. The derived client-to-server timestamp difference is 28 milliseconds, but that difference is meaningful only if the clocks are synchronized and the log points represent comparable events. It is not a round-trip-time measurement.

Now consider where correlation can break. If the translator reuses public port 53012 after the first mapping expires, the public tuple alone is ambiguous across time. If two clocks differ by seconds, a narrow time join selects the wrong mapping. If the server logs only an address and discards the source port, concurrent clients behind the same translator become indistinguishable at that layer. If an L7 proxy terminates the connection, the backend sees the proxy's tuple rather than the original public tuple.

The remedy is not to trust an arbitrary forwarding header. A proxy should overwrite, validate, or cryptographically protect client-address metadata according to a clear trust boundary. Applications should still use authenticated identity for authorization. Network tuple correlation is for locating a flow, not for proving who a human is.

When there is no mapping log, simultaneous captures on both interfaces can correlate sequence numbers, flags, payload length, and timing for a narrowly scoped flow. Be aware that TCP sequence numbers differ only if another proxy terminates and originates a new connection. A pure address translator rewrites headers but does not create a new TCP connection endpoint.

### Distinguish no route, no state, and no reply

Three failures that look like "connect timeout" need different evidence:

1. **No route:** `ip route get` fails or chooses an unintended interface or source. No packet reaches the expected boundary.
2. **No translation state:** the private SYN reaches the gateway, but no public SYN appears, and allocation or conntrack evidence points to the boundary.
3. **No reply:** the public SYN leaves with a valid mapping, but no SYN-ACK returns. Move toward the remote route, firewall, and listener.

There is also a fourth case: the public reply arrives but cannot be matched or routed back to the private endpoint. That implicates expired state, asymmetric return routing, conflicting policy, or reverse-path filtering. A client timeout alone cannot distinguish these cases. One carefully chosen observation on each side of the stateful boundary can.

Use packet capture defensively. Limit duration and size, filter to the required hosts and ports, and treat captures as sensitive because they can contain tokens, personal data, and payloads. Prefer header-only snap lengths when payload is unnecessary, but remember that truncation can hide the field needed for a later protocol question.

## Design rules that survive scale

### Size subnets for routing and failure domains

Address abundance is not a reason to create one enormous broadcast domain. Prefixes are operational boundaries for routing policy, fault isolation, ownership, and summarization. Leave expansion room, but prefer an aggregation plan that lets routers advertise a small number of stable prefixes.

Overlapping RFC 1918 plans become expensive when networks interconnect. Allocate from an organization-wide plan rather than letting every team choose `10.0.0.0/16`. Record owner, purpose, region, environment, and parent prefix. Enforce non-overlap in infrastructure review.

### Treat dual stack as two production paths

Probe IPv4 and IPv6 separately. Log the selected family and destination. Test fallback behavior. Avoid assuming that a DNS AAAA record proves the whole IPv6 path works, or that one successful IPv6 probe proves all clients have equivalent routing.

### Reuse connections deliberately

Connection reuse lowers new-flow rate $\lambda$ in the conntrack model and reduces ephemeral-port churn. It also concentrates work onto fewer sockets. Bound pools per destination, expire unhealthy connections, honor server limits, and apply backpressure rather than opening an unbounded replacement wave.

### Scale state, not only bandwidth

For every translator or stateful firewall, capacity-plan at least:

- tracked entries and entry creation rate;
- translated ports per public identity and destination behavior;
- memory under the deployed kernel and enabled extensions;
- cleanup time after close, timeout, or failure;
- failure-domain size if the device restarts and loses mappings;
- observability needed to connect private and public identities.

The system limit is the smallest constrained pool. A 100 Gbit/s interface does not help a full conntrack table. A large conntrack maximum does not help one exhausted public port pool. Many public ports do not help an application that reserves one private port per long-lived connection.

### Preserve evidence across the boundary

At minimum, retain timestamps, transport protocol, private source tuple, translated source tuple, destination tuple, and the device or node that created the mapping. Synchronize clocks. Define retention and access policies because these records can reveal user behavior and internal topology.

At the application layer, prefer authenticated principals and request identifiers over raw source IP for security and attribution. Source address remains useful network evidence, but it is one observation at one boundary, not a durable person or workload identity.

## Run it yourself

### Question

Can we prove that a narrow local ephemeral-port range limits simultaneous TCP connections to one destination tuple, then restore the namespace without changing the host's production networking?

### Preconditions

Use the Linux `netlab` environment from [the series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). You need root or `CAP_NET_ADMIN` for network-namespace sysctls, Python 3, `iproute2`, and the canonical namespaces `c` and `s` with `10.77.0.1/30` and `10.77.0.2/30`. Run this inside the disposable lab VM, not on an unspecified production interface. The experiment changes only the `c` namespace and restores its original range.

Confirm the state first:

```bash
set -euo pipefail
ip netns list
ip -n c addr show dev c0
ip -n s addr show dev s0
ip -n c route get 10.77.0.2
ip netns exec c python3 --version
ip netns exec s python3 --version
ip netns exec c sysctl net.ipv4.ip_local_port_range
```

Read the namespace names, `c0` and `s0` addresses, selected `src 10.77.0.1`, Python version, and original port range. Stop if those do not match the lab.

### Baseline

Start a disposable server that keeps accepted sockets open. The output directory stores only process IDs and logs for this slug:

```bash
set -euo pipefail
SLUG=addresses-subnets-nat-and-the-packets-two-identities
OUT="netlab/out/$SLUG"
mkdir -p "$OUT"

sudo ip netns exec s python3 -u - <<'PY' >"$OUT/server.log" 2>&1 &
import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("10.77.0.2", 8080))
s.listen(128)
held = []
print("LISTEN 10.77.0.2:8080", flush=True)
while True:
    conn, peer = s.accept()
    held.append(conn)
    print("ACCEPT", peer[0], peer[1], "held", len(held), flush=True)
PY
SERVER_PID=$!
echo "$SERVER_PID" >"$OUT/server.pid"
sleep 1
sudo ip netns exec s ss -lnt sport = :8080
```

Open four client sockets without changing the range:

```bash
sudo ip netns exec c python3 -u - <<'PY'
import socket
sockets = []
for i in range(4):
    s = socket.create_connection(("10.77.0.2", 8080), timeout=2)
    sockets.append(s)
    print("OPEN", i + 1, s.getsockname(), "->", s.getpeername())
input("Press Enter to close baseline sockets: ")
PY
```

Read: each `OPEN` line must show local address `10.77.0.1`, four source ports, and peer `10.77.0.2:8080`.

Expected: all four connects succeed. Source-port values vary because the unmodified namespace range and allocator state vary. This baseline proves the server and route work before the treatment.

### Apply one change

Save the original range, then constrain only namespace `c` to four ports:

```bash
set -euo pipefail
SLUG=addresses-subnets-nat-and-the-packets-two-identities
OUT="netlab/out/$SLUG"
sudo ip netns exec c cat /proc/sys/net/ipv4/ip_local_port_range \
  >"$OUT/original-port-range"
sudo ip netns exec c sysctl -w net.ipv4.ip_local_port_range='40000 40003'
sudo ip netns exec c sysctl net.ipv4.ip_local_port_range
```

Read: the final line must be `40000 40003`. The derived pool size is ${40003} - 40000 + 1 = 4$ ports.

### Compare

Attempt five simultaneous connections to the same destination tuple and keep successful sockets open:

```bash
sudo ip netns exec c python3 -u - <<'PY'
import errno
import socket

held = []
for i in range(5):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    try:
        s.connect(("10.77.0.2", 8080))
        held.append(s)
        print("OPEN", i + 1, s.getsockname(), "->", s.getpeername())
    except OSError as exc:
        print("FAIL", i + 1, "errno", exc.errno,
              errno.errorcode.get(exc.errno), str(exc))
        s.close()
print("SUMMARY open", len(held), "failed", 5 - len(held))
input("Press Enter to close treatment sockets: ")
PY
```

Read: inspect each `OPEN` source port, the `FAIL` errno name, and the `SUMMARY` counts.

Expected: four simultaneous connections normally succeed with source ports in `40000` through `40003`, and the fifth fails locally with `EADDRNOTAVAIL`, producing `SUMMARY open 4 failed 1`. A recently used tuple may temporarily reduce successes because TCP state protects against delayed packets. If you see three successes, wait for the lab's prior TCP state to clear, confirm `ss -tan` in namespace `c`, and repeat. Treat the expected range as three to four successes followed by at least one allocation failure, not as a benchmark.

This experiment isolates the client allocator. It does not claim that a NAT gateway has the same four-port configuration. It proves the necessary mechanism: simultaneous TCP connections to one destination tuple need distinct local source ports, and exhaustion can occur before an outbound SYN exists.

### Reset

Restore the saved range and terminate only this experiment's server:

```bash
set -euo pipefail
SLUG=addresses-subnets-nat-and-the-packets-two-identities
OUT="netlab/out/$SLUG"
ORIGINAL=$(cat "$OUT/original-port-range")
sudo ip netns exec c sysctl -w net.ipv4.ip_local_port_range="$ORIGINAL"

if [ -f "$OUT/server.pid" ]; then
  SERVER_PID=$(cat "$OUT/server.pid")
  sudo kill "$SERVER_PID" 2>/dev/null || true
fi

sudo ip netns exec c sysctl net.ipv4.ip_local_port_range
sudo ip netns exec s ss -lnt sport = :8080
```

Read: the port range must equal the baseline value, and no listener from this experiment should remain on `10.77.0.2:8080`. If another canonical `netserver` was already using that port, do not run this experiment until you stop it through the intro lab's scoped teardown.

The safe production equivalent is read-only: inspect `ip_local_port_range`, reservations, `ss` state counts, application errno, and a narrowly filtered capture. Do not narrow a production port range to reproduce an outage.

## Key takeaways

- A prefix fixes the leftmost bits. IPv4 block size is `${2}^{32-p}`, and the boundary-octet step is `${2}^{8-r}`.
- RFC 1918 private space and RFC 6598 shared carrier space have different purposes. Neither is an authentication boundary.
- IPv6 is a live production path. Dual stack means two paths whose success, latency, and fallback behavior should be observed separately.
- Stateful NAT preserves a mapping between private and public tuples. That mapping has a lifecycle and an observability cost.
- Conntrack capacity depends on active state, new-flow arrival rate, and residence time. Bit rate alone does not predict it.
- Ephemeral-port capacity depends on the full tuple and socket API behavior. A range size is not a universal connection count.
- Diagnose from the failing syscall and boundary outward. Inspect local ports, conntrack, translated ports, and the remote listener as separate finite pools.
- The series capstone, [the senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model), will combine these address and state checks with routing, transport, security, and application evidence.

## Further reading

- [RFC 4632: Classless Inter-Domain Routing, August 2006](https://www.rfc-editor.org/rfc/rfc4632.html)
- [RFC 1918: Address Allocation for Private Internets, February 1996](https://www.rfc-editor.org/rfc/rfc1918.html)
- [RFC 6598: Shared Address Space, April 2012](https://www.rfc-editor.org/rfc/rfc6598.html)
- [RFC 4291: IPv6 Addressing Architecture, February 2006](https://www.rfc-editor.org/rfc/rfc4291.html)
- [RFC 3022: Traditional IP Network Address Translator, January 2001](https://www.rfc-editor.org/rfc/rfc3022.html)
- [RFC 4787: NAT Behavioral Requirements for Unicast UDP, January 2007](https://www.rfc-editor.org/rfc/rfc4787.html)
- [Linux conntrack sysctl documentation, accessed September 29, 2026](https://docs.kernel.org/networking/nf_conntrack-sysctl.html)
- [Linux IP sysctl documentation, accessed September 29, 2026](https://docs.kernel.org/networking/ip-sysctl.html)
- [Cloudflare's ephemeral-port case, February 2, 2022](https://blog.cloudflare.com/how-to-stop-running-out-of-ephemeral-ports-and-start-to-love-long-lived-connections/)
