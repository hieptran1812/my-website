---
title: "Protocol ossification: Why TLS 1.3 wears an old coat and QUIC rides UDP"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Trace how middleboxes freeze protocol fields, why TLS 1.3 adopted compatibility camouflage, and which QUIC bytes the network can still see."
tags:
  [
    "networking",
    "distributed-systems",
    "protocol-ossification",
    "middleboxes",
    "tls-13",
    "quic",
    "udp",
    "grease",
    "http-3",
    "packet-capture",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 43
image: "/imgs/blogs/protocol-ossification-middleboxes-and-why-quic-looks-like-that-1.webp"
---

A browser and an origin can both implement a new protocol correctly and still fail to connect. A box between them might have learned that a field always contains one particular value, that a handshake always includes one particular message, or that a TCP connection always looks like the versions it inspected during testing. It can drop a connection that both endpoints would have accepted. That is protocol ossification: an extension point exists in the specification, but deployed observers turn yesterday's common behavior into tomorrow's accidental rule.

The path map below is the mental model for this post. The resolver supplies an address; application packets do not pass through it. Once the client sends packets toward the edge, an enterprise firewall, carrier gateway, NAT, load balancer, or inspection device can decide the flow's fate before the origin sees a byte. The highlighted inspection boundary is where an endpoint-only test stops explaining the result. If you have followed [one request from `curl` through the stack](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url), place this post at the point where the transport and TLS bytes first cross administrative control.

![A path map showing the client, resolver, edge, L4 and L7 hops, and the middlebox decision point before the origin](/imgs/blogs/protocol-ossification-middleboxes-and-why-quic-looks-like-that-1.webp)

The practical question is not whether a new protocol is elegant. It is whether packets carrying it traverse the paths that real users have. TLS 1.3 answered by keeping an old-looking outer handshake while negotiating new security semantics inside an extension. GREASE keeps selected extension points exercised so endpoints do not learn an overly narrow vocabulary. QUIC answered a harder transport problem by using UDP as a deployable outer envelope and protecting much of its transport state cryptographically. None of these strategies grants universal reachability. They reduce the number of assumptions a transit device can safely enforce.

We will separate four claims that are often compressed into a misleading slogan:

1. TLS 1.3 can present TLS 1.2-looking legacy fields while actually negotiating TLS 1.3.
2. GREASE tests whether TLS participants tolerate values they do not understand; it is primarily a defense against endpoint intolerance and does not make every middlebox tolerant.
3. QUIC protects many transport details, but IP, UDP, some QUIC header fields, lengths, timing, and the Initial handshake remain observable in important ways.
4. UDP is widely usable, not universally usable. A service needs a measured fallback and a way to tell a blocked datagram path from an application failure.

## 1. What it means for a protocol to ossify

**The diagnostic rule:** a connection can fail because of a rule at a transit hop even when both endpoints obey the RFC.

An extensible protocol reserves space for future behavior. A receiver might be required to ignore an unknown extension, or a version field might be intended to carry a higher value someday. The problem begins when an implementation sees only the common case. An intermediary developer captures a few handshakes, writes a parser around those examples, and treats the observed layout as the protocol. A decade of successful traffic then reinforces the mistake. The path has an invisible validation rule that no endpoint negotiated.

Think of a building whose doors were designed to accept several badge formats. If guards have only ever seen one shape, a new badge can be valid according to the building's policy and still be refused at the lobby. The tenant and the credential issuer may both be correct. The guard's private interpretation has become the effective protocol. In networking the guard may also alter packets, terminate TLS, rewrite headers, or create state based on fields it was never promised would stay fixed.

An ordinary router primarily forwards by network-layer information. The devices relevant here do more. A network address translator tracks flows and rewrites addresses or ports. A firewall permits or rejects flows by policy. A load balancer directs traffic to backends. An intrusion prevention system may parse a handshake, classify an application, and terminate traffic it considers malformed. A TLS-intercepting proxy is an endpoint for each of two separate TLS connections, with its own TLS implementation and trust arrangement. These are different roles. They share the ability to affect reachability between the user and intended service.

The distinction between *observation* and *authority* matters. A passive box can classify traffic and ask a firewall to block it. An active box can inject a reset, silently drop a packet, or rewrite a field. An endpoint that simply cannot understand a new version is normal negotiation. A box that rejects a conversation the endpoints would have negotiated creates a deployment constraint the protocol designer did not control.

The [2016 paper *Using UDP for Internet Transport Evolution*](https://arxiv.org/abs/1612.07816) begins from precisely this deployment problem. Its authors measured whether UDP encapsulation could carry new transport behavior through real paths. Their answer was broadly favorable but qualified: UDP worked on most tested networks, and impairments were concentrated in access networks. That result supports choosing UDP as an envelope. It does not prove that an arbitrary enterprise, hotel, mobile carrier, or cloud path will pass a particular UDP flow. The paper's population and date are part of the claim.

### The two tests an extension must pass

An extension succeeds only if endpoints understand it *and* transit devices allow its wire image through. The first test is visible in server and client implementation matrices. The second is path dependent. A connection can work in a lab and fail for one office, one carrier, or one proxy fleet. It may fail only after a device firmware update or only when a particular ClientHello length crosses a parser limit.

This leads to a useful debugging split. If a server rejects an unknown extension with a TLS alert, ask whether that server follows the negotiated-version and extension rules. If the client sees a timeout or TCP reset before the server logs a ClientHello, inspect the intervening path. If the server did receive the ClientHello but the client never saw the response, inspect the reverse path as well. A single `curl` error code cannot locate the box. A capture at both ends can.

### A rough risk model, explicitly an abstraction

The following is a planning model, not an RFC equation. Let (p_i) be the probability that path class (i) contains an intolerant observer, and (w_i) be the fraction of users on that class. Then the population-weighted failure fraction attributable to that observer class is approximately (sum_i w_i p_i), assuming those classes are mutually exclusive and the endpoint support test has already passed. The equation is useful because it shows why a small, high-traffic enterprise or carrier class can block a browser launch even while the protocol works perfectly on most developer networks. It cannot estimate the real failure rate without representative measurements, and correlated paths violate the simple independence intuition.

The senior move is to keep three denominators separate: attempts that reached the network, attempts whose endpoints supported the new protocol, and attempts that completed through the observed path. Many misleading adoption numbers silently mix them.

## 2. Why TLS 1.3 looks older than it is

**The wire rule:** do not infer the negotiated TLS version from the legacy version field alone.

TLS 1.3's ClientHello deliberately sets `legacy_version` to `0x0303`, the TLS 1.2 value. Its actual version preference appears in the `supported_versions` extension, where `0x0304` means TLS 1.3. The ServerHello also uses a legacy version value of `0x0303` while signaling the selected version through `supported_versions`. This is not a downgrade and not a hidden TLS 1.2 connection. It is the standardized representation in [RFC 8446, sections 4.1.2, 4.1.3, and 4.2.1, August 2018](https://www.rfc-editor.org/rfc/rfc8446.html).

The figure compares the clean conceptual expectation with the deployed compatibility shape. Read the right side as the actual TLS 1.3 wire image, not as a second protocol running inside TLS 1.2. A parser that only reads the outer legacy field will report the wrong version. A parser that rejects a nonempty session ID or an apparently redundant change-cipher-spec record may reject a valid handshake.

![A before-and-after comparison of a conceptual TLS 1.3 hello and its deployed compatibility form](/imgs/blogs/protocol-ossification-middleboxes-and-why-quic-looks-like-that-2.webp)

Why preserve these old-looking bytes? [RFC 8446 Appendix D.4](https://www.rfc-editor.org/rfc/rfc8446.html#appendix-D.4) says field measurements found that a significant number of middleboxes misbehaved when a client and server negotiated TLS 1.3. Compatibility mode increases the chance of traversing them. A client offers a nonempty legacy session ID. If it lacks a pre-TLS-1.3 session ID to reuse, it generates a new 32-byte value. The server echoes that value. The peers may send a dummy `change_cipher_spec` record at specified points. The record is ignored by a TLS 1.3 peer; it is a wire-image concession to equipment that expected the old transition marker. The `legacy_compression_methods` vector still carries the null compression byte.

The word *dummy* does not mean arbitrary traffic may be inserted anywhere. The RFC constrains placement and peer behavior. It also distinguishes compatibility mode from the negotiated cryptographic protocol. TLS 1.3 removes older key exchange and cipher choices even while selected framing details resemble TLS 1.2. For the handshake's security and round-trip consequences, see [the TLS handshake post](/blog/software-development/networking/tls-what-the-handshake-buys-and-what-it-costs). Here the question is why those bytes have their particular shape.

### What a capture should say

In a packet capture, inspect `ClientHello.legacy_version` and the `supported_versions` extension together. A modern decoder can label a TLS 1.3 ClientHello even though the legacy field reads `0x0303`. A plain grep for `0x0304` in raw bytes is less reliable because extensions move and GREASE changes surrounding values. When a tool reports a TLS 1.2 record layer followed by a TLS 1.3 handshake, do not assume the application negotiated TLS 1.2. Confirm the selected version from the ServerHello extension or from the client's TLS library result.

Use an endpoint probe and a capture as complementary observations. `openssl s_client -connect host:443 -tls1_3` tests a TLS 1.3 handshake with one implementation and one network path. `tcpdump` or `tshark` tells you what crossed a particular interface. A server log tells you whether the origin actually saw it. A proxy can terminate one TLS session and create another, so client-side and origin-side captures can each be truthful while showing different handshakes. [The TLS termination post](/blog/software-development/networking/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end) explains that boundary in detail.

### The compatibility cost

Every accommodation has a cost. Retaining old-looking fields consumes parser complexity and creates the possibility of misleading telemetry. More seriously, it signals that the supposedly flexible parts of a deployed protocol can become hard to change. TLS 1.3 recovered a path through the ecosystem, but it cannot force already deployed observers to implement the original extensibility rules. Future versions must account for the same installed base. This is an engineering constraint, not an argument that the older security protocol is equivalent.

## 3. GREASE exercises the joints before they rust

**The extension rule:** unknown values must be handled according to the protocol, not treated as impossible just because production rarely uses them.

GREASE stands for Generate Random Extensions And Sustain Extensibility. [RFC 8701, January 2020](https://www.rfc-editor.org/rfc/rfc8701.html) reserves a set of values such as `0x0a0a`, `0x1a1a`, and `0x2a2a` for TLS extension-related registries. A client can offer one of these intentionally meaningless values. A compliant peer should respond as it would to another unknown value in that position. The traffic continuously tests that an extension point remains extensible while a future legitimate value has not yet been assigned.

![A matrix showing how a tolerant TLS endpoint ignores a GREASE value while an intolerant endpoint rejects it](/imgs/blogs/protocol-ossification-middleboxes-and-why-quic-looks-like-that-3.webp)

This is best understood as a preventative maintenance probe. Suppose every deployed ClientHello contains only a fixed list of cipher suite identifiers. A server parser may accidentally hard-code that list. The day a new identifier appears, the parser rejects the entire ClientHello rather than ignoring an unknown offered option. If clients have been sending reserved GREASE identifiers all along, that bug is exposed earlier, while it is still an interoperability bug rather than a release blocker for the real extension.

GREASE is aimed primarily at endpoint behavior, especially servers that parse client offers. It may reveal some intolerance along a path, but it cannot guarantee that every middlebox passes a new handshake. A device can ignore a GREASE value yet reject a different field length, record sequence, UDP packet, or new version. A transit device can also block for policy reasons even when it parses correctly. Treat GREASE as one tool that preserves unused protocol joints, not as a universal bypass.

There is a subtle policy lesson here. For an offer list, an unknown element normally means the peer does not select that element. It does not mean the connection itself is invalid. For a selected value or a message in a different handshake phase, the rules can differ. RFC 8701 names positions where GREASE values are sent and positions where they must not be used. A production parser should implement those exact rules rather than a simplistic rule to accept any unknown number anywhere.

### Why a fixed fake value is insufficient

If every client forever sent only `0x0a0a`, an implementation could accidentally learn to special-case that number while remaining intolerant to the next unknown value. The reserved set varies the exercise. It makes the behavioral requirement visible across values and positions. The goal is not randomness for cryptographic secrecy. It is repeated exposure to harmless novelty so a parser cannot quietly make the extension point fixed again.

### The measurement you actually want

When a rollout fails, compare success by client build, TLS library, offered extension set, and network population. An endpoint-owned server can log the reason it rejected a ClientHello. A middlebox-owned path may leave no server log at all. If a GREASE-related regression is suspected, do not remove the unknown values globally as the first permanent fix. Reproduce the rejection with the exact byte sequence and determine whether the rejecting component is a server implementation or a transit observer. Removing the exercise may restore one path while allowing intolerance to grow elsewhere.

## 4. A public deployment case: TLS 1.3 in 2017

**The case rule:** distinguish early failure data from later compatibility experiments and keep their populations separate.

On [December 26, 2017, Cloudflare published Nick Sullivan's account of why TLS 1.3 was not yet enabled by default in major browsers](https://blog.cloudflare.com/why-tls-1-3-isnt-in-browsers-yet/). The report described experiments that began when Chrome and Firefox enabled TLS 1.3 for subsets of users in February 2017. Some users could establish TLS 1.2 connections but could not establish TLS 1.3 connections to otherwise reachable sites. Cloudflare attributed many of these failures to deployed middleboxes, including both intercepting and passive devices. This is a deployment case with a date, named participants, and measured outcome, rather than an invented packet trace.

Cloudflare's early Draft 18 comparison reported Chrome-to-Gmail success of 98.3% for TLS 1.2 and 92.3% for TLS 1.3. Its Firefox-to-Cloudflare comparison reported 97.8% for TLS 1.2 and 96.1% for TLS 1.3. These are separate client/server populations and should not be averaged. They also include all causes of connection failure in the observed attempts; the gap is evidence of rollout trouble, not a direct measurement of one named firewall model's drop rate.

The figure maps the early comparisons and later compatibility trials onto a timeline. Cloudflare reported that changes associated with PR 1091 made TLS 1.3 look more like TLS 1.2 to middleboxes. In a later Chrome experiment, the reported success rates were 98.6% for TLS 1.2 and 98.8% for the experimental changes. In the later Firefox experiment, they were 98.42% and 98.37% respectively. Do not subtract an early Chrome rate from the later experimental rate as if it were a controlled before-and-after effect on the same users. The cohorts, time, endpoints, and draft changed. The defensible conclusion is that the compatibility wire image reached success rates comparable to the TLS 1.2 control *within each later experiment*.

![A timeline of the 2017 TLS 1.3 deployment case with early failures and later compatibility cohorts kept separate](/imgs/blogs/protocol-ossification-middleboxes-and-why-quic-looks-like-that-4.webp)

The trigger was not a certificate outage or a newly broken server cryptographic primitive. The changed wire image violated assumptions in deployed TLS parsers. Contributing conditions included years in which old fields appeared in a stable arrangement and middleboxes learned that arrangement as if it were normative. The blast-radius multiplier was browser distribution: one client-side change reached many heterogeneous enterprise and access networks, most of which the browser team did not administer. Browser rollback or version fallback could mask the symptom for a user, but it did not explain which device had rejected the new handshake.

Cloudflare reported that features such as the old session ID and ChangeCipherSpec had been removed or changed in early TLS 1.3 designs, and that some middleboxes considered them essential. The revised design preserved enough familiar structure to cross those paths. [RFC 8446's middlebox compatibility appendix](https://www.rfc-editor.org/rfc/rfc8446.html#appendix-D.4), published in August 2018, codified the accommodation. The transfer lesson is to test new wire images across representative paths before making an endpoint rollout global, then keep distinct success denominators for client version, network, and negotiated protocol.

### What the incident would look like in service telemetry

A service may see fewer completed TLS 1.3 connections while reporting no increase in application 5xx responses. That is expected: a failed handshake never becomes an HTTP request. A successful TCP connect followed by a TLS timeout can sit between a firewall's logs and an application's logs. At an edge that terminates TLS, compare accepted TCP connections, received ClientHellos, completed handshakes, and HTTP requests. If accepted TCP rises but completed handshakes falls for a new client cohort, inspect handshake alerts and captures. If the edge never receives the ClientHello, inspect the client-side path and upstream middleboxes.

This is where the [latency ladder from the series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) remains useful. `curl`'s `time_connect` can complete while `time_appconnect` does not. A request that never reaches HTTP has no server think time to optimize. Tuning a handler or a database pool cannot repair a dropped ClientHello.

## 5. Why changing TCP itself was harder

**The transport rule:** a new endpoint capability is not enough when transit devices maintain TCP state.

TCP offers a familiar, widely permitted path. It also exposes fields that devices use to track connections: ports, flags, sequence and acknowledgement numbers, options, and timing. A NAT needs enough state to translate the flow. A firewall may use a state machine to admit packets after a SYN, and a performance device may try to infer loss or retransmission from sequence numbers. That installed machinery makes a wholly new transport protocol difficult to deploy and makes changes to TCP's wire behavior risky. The 2016 UDP evolution paper frames that problem as transport ossification by middleboxes, not as a deficiency in the TCP endpoints' ability to ship code.

This does **not** mean every TCP extension is impossible. Many have been deployed. It means every change must survive the devices that interpret or normalize the traffic. A TCP option can be removed, rewritten, or mishandled. A packet pattern can trigger a policy that assumes an older implementation. Even when a change is legal at the endpoints, the observable wire image has to pass through a heterogeneous network that upgrades on a different schedule. An endpoint team can usually ship a new user-space library far faster than it can upgrade every firewall and NAT in the world.

There is a control problem as well as a parser problem. A transport algorithm may want to change loss detection, packet numbers, acknowledgement format, multiplexing, or connection migration. If these are visible as TCP fields, in-path devices can come to depend on their precise meaning. That dependency constrains later evolution. If instead those details are carried inside an encrypted payload, an on-path observer can still measure sizes and timing, but it cannot safely write a policy against an internal acknowledgement frame it cannot parse.

QUIC's choice was to keep the outer envelope that common networks can route and track: IP plus UDP. Inside that envelope, endpoints implement a transport with connection setup, streams, flow control, congestion control, loss recovery, and TLS-based security. [RFC 9000, May 2021](https://www.rfc-editor.org/rfc/rfc9000.html) specifies the transport. [RFC 9001, May 2021](https://www.rfc-editor.org/rfc/rfc9001.html) specifies how TLS secures it. The word *inside* is important. UDP itself does not provide reliability or congestion control. QUIC supplies those functions above UDP, subject to QUIC's own rules, and applications such as HTTP/3 use them.

### UDP is an envelope, not a magic tunnel

An outbound UDP datagram still crosses firewalls, NATs, traffic policers, and load balancers. It needs return traffic to find the client. A path may filter UDP entirely, filter port 443 specifically, time out idle UDP mappings quickly, or throttle datagrams. A new protocol that rides UDP has exchanged one set of deployment constraints for another. The 2016 measurement study found broad UDP usability in its tested networks and also documented impairment. That is why a serious deployment records protocol selection and failure by network population and retains a TCP-based option for users whose UDP path fails.

Do not make the stronger claim that UDP *always* traverses because DNS uses it. DNS may use a different destination port, a local resolver, a short request/response pattern, or an explicit firewall exception. A long-lived UDP flow to an HTTPS edge is a different policy object. Likewise, successful UDP `ping` to one host says little about MTU, idle timeout, or bidirectional reachability for a production QUIC connection.

### The animated mental model

The following animation is a conceptual deployment path, not a packet capture. A new transport behavior on the TCP path reaches a stateful inspector that expects the old pattern. A QUIC flow places new transport semantics inside a UDP datagram, reducing that inspector's ability to require a particular internal format. The bottom lane also marks the limit: a firewall can still block UDP at the outer envelope. Motion matters because the failure happens at a particular hop, before the application can answer.

<figure class="blog-anim"><style>.ossq23-wrap{max-width:960px;margin:0 auto}.ossq23-svg{width:100%;height:auto;max-width:960px}.ossq23-old{animation:ossq23-oldstop 10s linear infinite}.ossq23-new{animation:ossq23-newpass 10s linear infinite}.ossq23-limit{animation:ossq23-limitshow 10s linear infinite}@keyframes ossq23-oldstop{0%,8%{transform:translateX(0);opacity:1}32%,58%{transform:translateX(265px);opacity:1}62%,100%{transform:translateX(265px);opacity:.25}}@keyframes ossq23-newpass{0%,12%{transform:translateX(0);opacity:1}65%,88%{transform:translateX(645px);opacity:1}100%{transform:translateX(0);opacity:1}}@keyframes ossq23-limitshow{0%,64%{opacity:0}68%,94%{opacity:1}100%{opacity:0}}@media (prefers-reduced-motion:reduce){.ossq23-old,.ossq23-new,.ossq23-limit{animation:none}.ossq23-old{transform:translateX(265px);opacity:.25}.ossq23-new{transform:translateX(645px)}.ossq23-limit{opacity:1}}</style><div class="ossq23-wrap"><svg class="ossq23-svg" style="width:100%;height:auto;max-width:960px" viewBox="0 0 960 430" role="img" aria-label="A changed TCP wire pattern stops at a stateful middlebox; a QUIC datagram carries protected transport semantics over UDP, while UDP blocking remains possible"><title>Two deployment paths through a middlebox</title><rect x="10" y="20" width="940" height="390" rx="24" fill="#f8f9fa" stroke="#ced4da"/><text x="55" y="70" font-size="27" fill="#1e1e1e">A new transport must cross the path</text><text x="52" y="140" font-size="22" fill="#1e1e1e">New TCP pattern</text><line x1="52" y1="180" x2="900" y2="180" stroke="#868e96" stroke-width="4"/><rect x="360" y="123" width="180" height="115" rx="14" fill="#ffec99" stroke="#1e1e1e"/><text x="377" y="160" font-size="21" fill="#1e1e1e">Stateful</text><text x="377" y="190" font-size="21" fill="#1e1e1e">inspector</text><circle class="ossq23-old" cx="75" cy="180" r="20" fill="#ffc9c9" stroke="#1e1e1e"/><text x="52" y="302" font-size="22" fill="#1e1e1e">QUIC over UDP</text><line x1="52" y1="338" x2="900" y2="338" stroke="#868e96" stroke-width="4"/><rect x="360" y="282" width="180" height="112" rx="14" fill="#e9ecef" stroke="#1e1e1e"/><text x="377" y="320" font-size="20" fill="#1e1e1e">Outer UDP</text><text x="377" y="350" font-size="20" fill="#1e1e1e">still visible</text><circle class="ossq23-new" cx="75" cy="338" r="20" fill="#a5d8ff" stroke="#1e1e1e"/><text class="ossq23-limit" x="565" y="270" font-size="20" fill="#a61e4d">Policy can still block UDP</text><text x="720" y="151" font-size="20" fill="#1e1e1e">origin</text><text x="720" y="310" font-size="20" fill="#1e1e1e">origin</text></svg></div><figcaption>A fixed TCP inspection assumption can stop a new wire pattern; QUIC protects transport details inside UDP, while outer UDP policy can still stop the flow.</figcaption></figure>

The animation deliberately does not claim that a TCP connection always fails or that QUIC always passes. It shows two possible constraints. In the first lane, the middlebox is intolerant to the changed TCP pattern. In the second, the inner QUIC transport fields are unavailable for that same kind of parsing, but the outer UDP policy remains. This is the exact scope of the design argument.

## 6. What a QUIC observer can and cannot see

**The visibility rule:** QUIC encrypts transport payloads and protects parts of its header, but it does not encrypt every header field.

Start outside the QUIC packet. IP source and destination addresses remain visible to forwarding devices. UDP source and destination ports remain visible. A path observer sees datagram sizes, packet timing, direction, and any resulting loss or retransmission patterns it can infer. A network operator can still route, rate limit, or block by these properties. Encryption does not remove the network's authority over reachability.

Now inspect QUIC itself. [RFC 8999, May 2021](https://www.rfc-editor.org/rfc/rfc8999.html) defines a small version-independent set of invariants so endpoints can recognize the broad packet form and negotiate versions without freezing every detail of QUIC version 1. A long header exposes a version field and connection ID lengths and values, among other invariant structure. A short header exposes a smaller invariant shape and a destination connection ID whose interpretation depends on the connection context. [RFC 9000 section 17](https://www.rfc-editor.org/rfc/rfc9000.html#section-17) defines version 1 packet formats. Some bits and packet number material receive header protection under RFC 9001. Payload protection covers QUIC frames, including transport-control information that middleboxes might otherwise learn to parse.

![A layered wire layout distinguishing visible IP, UDP, and QUIC invariant fields from protected header parts and payload](/imgs/blogs/protocol-ossification-middleboxes-and-why-quic-looks-like-that-5.webp)

It is tempting to call the entire QUIC header encrypted. That is wrong. Version negotiation requires visible information, routing and connection association need connection IDs in some packets, and the outer network headers necessarily remain accessible. It is also tempting to say a QUIC Initial packet is secret because it is encrypted. That is wrong in a different way. [RFC 9001 section 5.2](https://www.rfc-editor.org/rfc/rfc9001.html#section-5.2) derives Initial secrets from a version-specific salt and the client's Destination Connection ID, both available to an observer. The RFC explicitly says Initial packets have no confidentiality protection against an observer. Initial encryption provides a consistent protected packet machinery and some integrity properties, while later handshake and application data use keys that a passive observer cannot derive in the same way.

The consequence for packet analysis is precise. A passive capture may decode a QUIC Initial and recover the TLS ClientHello inside it, depending on tooling and capture completeness. It should not be expected to decode 1-RTT application frames without endpoint secrets. A network appliance that assumes every byte after UDP is opaque will miss visible invariant fields and Initial information. An appliance that assumes every QUIC version has the same version 1 layout violates the deliberate invariant boundary.

### Connection IDs are operationally useful and privacy relevant

A connection ID allows endpoints and load balancers to associate packets with a connection even if the client's address or port changes. That supports connection migration, one of QUIC's distinctive transport capabilities. It also creates metadata visible to path observers while the ID is in use. The protocol includes rules for issuing, retiring, and changing connection IDs; an operator should not treat them as permanent user identifiers. At a load balancer, connection IDs can help maintain routing continuity, but their format is an implementation choice with privacy implications. The network can use them for routing without learning the protected stream and acknowledgement frames.

### The version-independent contract is intentionally narrow

RFC 8999 tries to avoid repeating the TLS experience. It lists only the wire properties that must remain stable across QUIC versions. Its appendix warns against assuming version 1 details are universal. For example, an observer should not assume a particular meaning for every short-header bit in a future version. The protocol's design limits what transit devices can validly require, while still exposing enough to distinguish a QUIC packet from arbitrary UDP data and to perform version negotiation.

This is a social and operational contract as much as a cryptographic one. A middlebox can always choose to drop unfamiliar traffic. It cannot be made physically incapable of doing so. But if the standard exposes only a narrow, explicit invariant set, a vendor that wants future compatibility has a clear boundary: parse the invariants needed for its job and treat the rest as endpoint-owned. That boundary makes protocol evolution more feasible than a wire image full of stable, readable transport internals.

## 7. The performance argument, with its conditions

**The performance rule:** choose QUIC or TLS 1.3 for the workload and path you measured, not for a context-free speed slogan.

Protocol ossification is mainly about the ability to deploy improvements. A successful new transport can then implement lower connection setup cost, independent streams, or better migration behavior. These benefits depend on the application, connection reuse, loss pattern, and path. A warm HTTP/2 connection with many requests pays no fresh TCP and TLS handshake for each request. A cold connection across a high-RTT path pays more. HTTP/3's QUIC stream independence avoids TCP's connection-level head-of-line blocking when one packet is lost, but a stalled application dependency or a server queue can dominate either protocol.

Here is a derived setup example, deliberately simplified. Assume a client has already resolved the address, the path round-trip time is 80 ms, there is no packet loss, and neither endpoint can reuse an existing connection. A TCP three-way handshake costs about one RTT before the client can complete a conventional TLS 1.3 handshake. TLS 1.3's full handshake costs about one more RTT before application data can be exchanged in the usual case. A QUIC connection integrates transport and TLS setup, so the comparable full setup can be around one RTT before protected application exchange. Under this model, one saved RTT is about 80 ms. This is a protocol-flight comparison, not a promised `curl` delta. Server processing, address validation, retransmissions, connection reuse, resumption, and implementation scheduling change observed wall time. The detailed TLS flight accounting is in [the handshake post](/blog/software-development/networking/tls-what-the-handshake-buys-and-what-it-costs).

The same arithmetic shows when setup optimization does little. Assume a reused connection and a 5 ms RTT between services. Both protocols avoid a fresh connection handshake for the request, so the previously derived 80 ms advantage is irrelevant. If the handler spends 120 ms waiting on a backend, shaving a connection flight that is not present changes nothing. The right metric is a distribution of request paths split by cold versus warm connection, RTT, failure, and protocol. A global mean blends together the very cases the design is intended to improve.

The second worked example concerns fallback cost. Treat the following as an explanatory model, not a QUIC specification. Suppose a client tries UDP first, waits a locally configured 250 ms before starting TCP in parallel, and the UDP path is blocked. If TCP setup and TLS take a further 100 ms under the assumed path, the first usable connection completes at about 350 ms from the start, ignoring DNS and server work. If the client instead starts TCP immediately in parallel, the first usable connection could complete around 100 ms, at the expense of extra connection attempts and server work when UDP succeeds. The numbers are illustrative inputs to the arithmetic, not a claim about a particular browser algorithm. Real clients use their own timers and racing behavior, and those change by version. Measure your actual client's attempt timeline before attributing a 250 ms tail to QUIC fallback.

That is why an origin's HTTP/3 enablement decision includes telemetry design. Record which protocol completed, whether the request used a new connection, the apparent client network or region at an appropriate privacy-preserving aggregation level, and the fallback delay visible to the user. A QUIC success rate of 90% means little without knowing the population that attempted it and whether the unsuccessful 10% completed quickly over TCP or waited long enough to hurt p99. A policy that only optimizes QUIC success may hide the experience of the blocked cohort.

### The four different head-of-line problems

Engineers often say QUIC “fixes head-of-line blocking” without naming the queue. There are several queues. HTTP/1.1 request sequencing on one connection can prevent a later response from being delivered until an earlier response is ready, unless the application uses more connections or pipelining mechanisms with their own costs. HTTP/2 multiplexes streams at the application layer, but all their bytes ride one ordered TCP byte stream. A lost TCP segment can delay delivery of bytes from other HTTP/2 streams that arrived later. QUIC multiplexes streams over its own packet transport, so loss for one stream need not prevent delivery of received data on another stream. None of this eliminates a shared server queue, congestion-control limit, or an application lock. The network can still be the bottleneck.

An operator should therefore avoid inferring protocol superiority from one screenshot of a waterfall. Measure the condition in which the difference is predicted: multiple concurrent streams, loss or reordering, a sufficiently long connection, and a comparable congestion and server configuration. If the request has one short stream on a clean local path, the purported head-of-line benefit has little room to appear. If it has dozens of independent resources over a lossy access path, the mechanism has a real opportunity to matter. This is a falsifiable expectation, not a universal benchmark number.

### Security is part of the deployment mechanism

QUIC's encrypted transport fields prevent a passive observer from learning or enforcing the detailed internal frame grammar of established connections. That is a deployment feature, but it is also a security and privacy choice. The price is that an on-path performance appliance cannot transparently rewrite acknowledgements or inspect streams in the same way it might with unencrypted TCP metadata. Operators need endpoint telemetry and intentional load-balancer interfaces instead of relying on accidental packet visibility. This shifts some diagnostic work from a network tap to the endpoints that own the transport state.

It also changes how to reason about a failure. A TCP capture can expose retransmissions and receive-window advertisements directly. A QUIC capture without keys may show datagram loss and timing, but cannot directly identify every protected ACK or flow-control frame. The endpoint can export those signals. If your observability design assumes a packet tap will always reconstruct transport state, QUIC will expose that assumption. The solution is not to weaken encryption. It is to specify the operational measurements you need, collect them at endpoints or a QUIC-aware edge, and correlate them with path-level packet counts.

## 8. A diagnostic path for a failed rollout

**The triage rule:** ask where the last confirmed packet or handshake stage was observed.

The decision tree below starts with the symptom that matters to a service team: a client reports a connection or protocol failure. First check whether the TCP-based path to the same service works. Then check whether UDP datagrams reach the edge, whether the QUIC handshake progresses, and whether the application request reaches the handler. For a TLS 1.3-over-TCP failure, compare the client-side and server-side ClientHello. This avoids treating every failure as an undifferentiated “network issue.”

![A diagnostic decision tree that separates UDP blockage, TLS handshake intolerance, QUIC handshake failure, and application errors](/imgs/blogs/protocol-ossification-middleboxes-and-why-quic-looks-like-that-6.webp)

The first fork is basic reachability. Does TCP port 443 complete and serve the same hostname? If both TCP and UDP fail, start with DNS, address selection, routing, edge health, or a broader policy. A working TCP path and failed UDP path makes an outer UDP policy or QUIC-specific problem plausible. It does not prove filtering: the server might not listen on UDP, the address might differ, or the client might never attempt HTTP/3. Confirm the listener and actual packets before naming a firewall.

The second fork is packet arrival. Capture a narrow UDP/443 slice at the client and edge. If the client sends datagrams and the edge sees none, the drop lies between those capture points. If the edge sees them but sends no response, inspect its QUIC listener and policy. If it responds but the client receives nothing, investigate the return path, NAT mapping, firewall state, and asymmetric routing. If bidirectional packets exist but the QUIC handshake stalls, endpoint logs, version negotiation, address validation, and TLS handshake errors become the next evidence. A packet capture alone may not decode all protected state.

For TLS 1.3 over TCP, inspect whether the ClientHello leaves the client and reaches the TLS terminator. If it arrives and the server emits an alert, the TLS endpoint is giving you a specific failure. If it never arrives or a reset appears from an intermediate address, investigate the transit path. If a TLS-inspecting proxy is deployed, remember that the client and origin see different connections. The proxy can fail one leg while the other leg is healthy. A successful origin-side `openssl` probe does not test the client-to-proxy leg.

| Observation | Next discriminating measurement | Likely layer, not yet a verdict | Source |
| --- | --- | --- | --- |
| TCP/443 succeeds, UDP/443 has no edge arrival | Client and edge UDP capture with the same destination and time window | Outer UDP path or policy | Derived diagnostic procedure here |
| UDP reaches edge, no response leaves | Edge listener and QUIC error counters | Edge listener, admission, version, or resource policy | Derived diagnostic procedure here |
| ClientHello leaves client but never reaches terminator | Captures bracketing the suspected proxy or firewall | Transit TLS parser or routing path | Derived diagnostic procedure here |
| ClientHello reaches terminator and TLS alert returns | Alert description and server TLS logs | TLS endpoint negotiation or certificate policy | Derived diagnostic procedure here |
| Handshake completes, HTTP fails | Request trace and status at first HTTP-aware hop | Application or L7 policy | Derived diagnostic procedure here |

This table is intentionally a sequence of observations, not a mapping from symptoms to automatic root causes. A client may have multiple addresses or proxies; a capture on the wrong interface can produce a false “never arrived.” Clock skew makes timestamps from two hosts deceptive. Use a stable flow identifier such as the address pair, ports, and time window, and align clocks or compare packet content rather than subtracting unsynchronized timestamps.

### What to log before enabling HTTP/3 broadly

At the edge, log completed and failed QUIC handshakes by failure stage, negotiated QUIC version, address family, and coarse network cohort. Record UDP datagrams received and responses sent. For TCP, log accepted connections, ClientHellos, completed TLS handshakes, and HTTP requests. At the client, record attempted protocol, selected protocol, connection reuse, fallback timing, and the first failure reason available from the library. Keep personally identifying metadata out of long-lived dashboards unless you have a specific operational need and retention policy.

Roll out by cohort, not only by aggregate percentage. An organization with a restrictive egress firewall can disappear in a global success average. A mobile access network can behave differently from fixed broadband. IPv4 and IPv6 paths may traverse different middleboxes. A region can have a different anycast edge and policy chain. If a rollout degrades one cohort, the right temporary response may be to steer that cohort to TCP while you investigate. Calling the protocol a failure everywhere throws away useful evidence.

### Fallback is part of correctness

If a service advertises HTTP/3, it should also keep a working HTTP/2 or HTTP/1.1-over-TCP path appropriate to its clients. The exact client discovery and fallback behavior depends on the application and implementation, so test the clients you actually support. The key operational property is that a blocked UDP attempt does not turn a reachable service into an indefinite timeout. Monitor the time to successful fallback, not merely whether fallback eventually succeeds. A 300 ms penalty repeated at every cold connection can be significant for an interactive client even when error rate remains low.

For internal service meshes, the policy can be different because you may control the entire network and the endpoints. Yet a claim of control should include the NAT, host firewall, CNI, sidecar, cloud security group, and load balancer. A packet travels through all of them. [Service identity and mTLS](/blog/software-development/networking/mtls-and-service-identity-at-scale) address who the peers are. This post addresses what their handshake and transport look like to everything in between. [Service-to-service security](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) owns the higher-level trust architecture.

## 9. Designing extensions that survive their own success

**The design rule:** an extension point is only useful if deployed participants continue to exercise its unknown-value behavior.

The TLS 1.3 and QUIC stories suggest a practical checklist for protocol and platform teams. First, define a small invariant surface: what must an intermediary parse to perform a legitimate job? Second, state how every participant handles unknown values on extension points. Third, send unknown but reserved values routinely, where the protocol permits it, so intolerance appears before a critical feature launch. Fourth, encrypt or authenticate fields that intermediaries have no business rewriting. Fifth, retain a deployment path that works when the preferred transport is filtered. These are design choices, not a guarantee of universal reachability.

The inverse checklist helps when reviewing a middlebox or L7 proxy. Does it reject unknown extensions in a ClientHello? Does it assume `legacy_version` tells the negotiated TLS version? Does it require a TLS record sequence that the current RFC allows to vary? Does it parse QUIC version 1 fields and then apply those assumptions to unknown versions? Does it key UDP state to an idle timeout shorter than the application's normal pause? Does its logging distinguish “policy denied,” “could not parse,” and “upstream did not answer”? A box that cannot explain which rule it applied is difficult to operate during a protocol rollout.

There is a legitimate reason for some inspection. Enterprises enforce policy, edges defend against abuse, and load balancers need routing information. The design question is which information must be visible for each job. IP and UDP headers are enough for basic forwarding. A QUIC-aware load balancer may intentionally use connection IDs. An HTTP-aware proxy must terminate the encrypted connection by design and become an endpoint of that leg. These arrangements can be made explicit. Trouble starts when an opaque transit device silently takes dependency on fields it was never promised would remain stable.

Standards language alone does not protect an extension point. A parser bug can be present for years before a new value exercises it. The 2017 TLS case demonstrates that large browser and edge teams had to measure across real networks and reshape a standardized handshake to pass deployed equipment. GREASE is a response to that evidence. QUIC's narrow invariants and protected internals are another response. Neither mechanism removes the need for staged measurement. A path is an evolving collection of software and policy, not a static channel.

The relation to [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) is worth making explicit. L4 routing can use addresses, ports, and sometimes intentional connection-ID structure without understanding HTTP semantics. L7 routing needs application information and therefore a termination point that can decrypt and parse it. Treating an accidental peek into a handshake as a stable L7 interface is fragile. A deliberately terminated connection has ownership, logs, certificates, and a clear upgrade boundary. A passive parser with hidden assumptions often has none of those operational affordances.

## 10. Three rollout reviews I would insist on

**The review rule:** ask what the change looks like to each component that did not participate in the design.

The first review is a TLS library update. A team may see a routine dependency bump and focus on cryptographic support. I would ask for a sample ClientHello from the old and new builds, including extension ordering, offered version, session ID behavior, and record sequence. Then I would ask whether egress proxies, inbound WAFs, and TLS inspection products on the relevant paths have test coverage for that exact wire image. If they do not, roll out to a cohort that includes those paths. A successful `openssl` command from a CI runner tests the CI runner's path, not the office network or mobile carrier where users live. Keep an error budget for handshake completion, not only HTTP status.

The second review is enabling HTTP/3 at an edge. I would verify that UDP/443 is listening on every advertised edge address, that the load balancer preserves a stable route for the QUIC connection, that reverse traffic is allowed, and that TCP/443 remains healthy. I would ask which client versions attempt HTTP/3 and how they learn the alternative. I would measure first-connection latency for successful QUIC, successful TCP, and failed-UDP-then-TCP paths separately. If a firewall blocks UDP, the service should continue to work through TCP with a bounded penalty. If the fallback penalty is larger than the gain for the successful cohort, the rollout needs tuning even when its error rate is zero.

The third review is a middlebox firmware or policy change. A security team may describe a parser update as invisible because addresses and ports remain the same. I would ask whether the update changes TLS extension tolerance, QUIC version handling, UDP idle timers, or the behavior for unknown packet forms. A parser that drops what it does not recognize has made itself a participant in every future endpoint upgrade. Test current protocols plus deliberately unknown values that the standard says should pass. Log the rule ID and packet stage when traffic is denied. Otherwise the first symptom may be a browser protocol fallback far from the box that caused it.

These reviews connect to [SRE observability design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design), but the wire measurements remain specific. A generic “network healthy” dashboard cannot tell whether the first ClientHello reached the edge, whether the edge answered UDP, or which protocol completed. The operational record needs those facts. The organizational incident process is separate from the packet diagnosis.

### A failure that changes with one network

Suppose a browser gets HTTP/3 at home and HTTP/2 at work, with no application errors in either location. This is not automatically a regression. The office may block UDP/443 by policy, a proxy may intercept TLS, or the two paths may receive different edge advertisements. Start by recording the resolved address, selected protocol, and whether UDP packets left the client. If they left, compare an edge-side capture for the same interval. If the office intentionally blocks UDP, verify fallback time and document the supported policy. If UDP reaches the edge and stalls, inspect QUIC endpoint state. The comparison is useful precisely because the application and browser are held mostly constant while the path changes.

### A failure that appears only after a client update

Suppose one browser release has a TLS 1.3 handshake failure in a subset of enterprise networks while older builds succeed. Do not conclude immediately that TLS 1.3 itself is unsupported there. The new build may have changed extension ordering, GREASE placement, key shares, session ID behavior, or record fragmentation. Compare old and new ClientHellos at the same capture point. Then compare with what the server receives. If the new bytes disappear between those points, a transit parser is implicated. If the server receives and rejects them, inspect its TLS stack or the terminating proxy. A version label alone is too coarse: two TLS 1.3 ClientHellos can be materially different to an intolerant parser.

### A failure that disappears when a proxy is bypassed

This is strong evidence of a path-specific effect, but “the proxy is broken” remains too broad. A TLS-intercepting proxy is a client of the origin and a server to the browser. The browser-to-proxy leg may accept the new ClientHello while the proxy-to-origin leg does not. A passive proxy may drop the browser ClientHello before it gets anywhere. A bypass can also change DNS resolution, source IP, and edge selection. Capture and log both legs before choosing a remedy. If a proxy must inspect application data for policy, make its termination role explicit and upgrade its TLS implementation. If it only needs coarse flow policy, avoid parsing more than the stable fields required for that policy.

## Run it yourself

### Question

Can we observe a TLS 1.3 handshake whose legacy field reads like TLS 1.2, then prove that a UDP flow can be stopped by a rule outside the application while a separate TCP service remains reachable? This lab proves the wire-image and outer-policy mechanisms. It does **not** reproduce Cloudflare's 2017 population measurements or prove that QUIC itself traverses every network.

### Preconditions

Use the Linux `c` and `s` namespaces from [the series' setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url), with `c0` at `10.77.0.1/30` and `s0` at `10.77.0.2/30`. You need `iproute2`, OpenSSL with TLS 1.3 support, Python 3, `tcpdump`, `tshark`, and `nft`. Namespace, packet capture, and firewall commands require root or appropriate capabilities. Run the mutation only inside the named `s` namespace. Do not paste the `nft` rule onto a production interface. Captures can contain credentials and application payloads; use this lab's self-signed test service only.

The commands below assume a shell in which variables and background process IDs remain available between blocks. The temporary directory is scoped to this experiment. Check the topology and tool versions first:

```bash
set -euo pipefail
for tool in ip openssl python3 tcpdump tshark nft; do command -v "$tool"; done
openssl version
tshark --version | head -n 1
sudo ip netns list
sudo ip netns exec c ip -brief addr show c0
sudo ip netns exec s ip -brief addr show s0
sudo ip netns exec c ip route get 10.77.0.2
sudo ip netns exec c ping -c 1 -W 1 10.77.0.2
```

Read: `ip route get` should select `c0`, the addresses should be `10.77.0.1` and `10.77.0.2`, and the one-packet ping should receive a reply. Ping establishes this narrow base reachability only. It says nothing about TCP, UDP, TLS, or application health. If your namespace setup differs, repair it using the setup post rather than renaming this experiment's endpoints.

### Baseline

Create a short-lived test certificate, start a TLS 1.3 server in `s`, capture its handshake at `c0`, and connect from `c`. The certificate is intentionally self-signed because this test examines version framing, not certificate trust. Restrict the capture to the one lab address and port.

```bash
LAB_DIR=$(mktemp -d /tmp/ossq23.XXXXXX)
openssl req -x509 -newkey rsa:2048 -nodes -days 1 \
  -subj '/CN=netlab.local' \
  -keyout "$LAB_DIR/key.pem" -out "$LAB_DIR/cert.pem" >/dev/null 2>&1
sudo ip netns exec s openssl s_server -accept 10.77.0.2:8443 \
  -cert "$LAB_DIR/cert.pem" -key "$LAB_DIR/key.pem" \
  -tls1_3 -www >"$LAB_DIR/server.log" 2>&1 &
TLS_PID=$!
sleep 1
sudo ip netns exec s ss -lnt '( sport = :8443 )'
sudo ip netns exec c tcpdump -i c0 -s 0 -U -w "$LAB_DIR/tls.pcap" \
  'host 10.77.0.2 and tcp port 8443' >/dev/null 2>&1 &
CAP_PID=$!
sleep 1
sudo ip netns exec c openssl s_client -connect 10.77.0.2:8443 \
  -servername netlab.local -tls1_3 -brief </dev/null \
  >"$LAB_DIR/client.log" 2>&1 || true
sleep 1
sudo kill "$CAP_PID" 2>/dev/null || true
wait "$CAP_PID" 2>/dev/null || true
cat "$LAB_DIR/client.log"
tshark -r "$LAB_DIR/tls.pcap" -d tcp.port==8443,tls \
  -Y 'tls.handshake.type == 1' -V |
  grep -E 'Version: TLS 1.2|Supported Version|TLS 1.3|Session ID Length' |
  head -n 20
```

Read: the client log should identify `TLSv1.3` as the negotiated protocol. In the decoded ClientHello, look for the legacy version `0x0303` or `TLS 1.2` label and a `supported_versions` entry containing TLS 1.3. On common OpenSSL builds in compatibility mode, the legacy session ID is nonempty. `tshark` field wording varies by version, so inspect the full `-V` ClientHello if the grep misses a label. The expected state is qualitative: negotiated TLS 1.3 alongside the older-looking legacy field. This is the exact distinction RFC 8446 makes. Do not infer the negotiated version from the legacy field alone.

Next, use a tiny UDP echo service on port 9443. Port 9443 keeps this lab separate from any HTTP/3 listener. It models only the outer datagram reachability property; it is not a QUIC implementation. The service returns the same bytes the client sent, which gives an unambiguous baseline observation.

```bash
cat >"$LAB_DIR/udp_echo.py" <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(('10.77.0.2', 9443))
while True:
    data, peer = s.recvfrom(2048)
    s.sendto(data, peer)
PY
sudo ip netns exec s python3 "$LAB_DIR/udp_echo.py" \
  >"$LAB_DIR/udp.log" 2>&1 &
UDP_PID=$!
sleep 1
sudo ip netns exec s ss -lun '( sport = :9443 )'
sudo ip netns exec c python3 -c \
  'import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(1); s.sendto(b"probe",("10.77.0.2",9443)); print(s.recvfrom(2048)[0].decode())'
```

Read: the last line should be `probe`. A one-second timeout is a lab bound, not a performance requirement or a measured Internet RTT. The `ss -lun` line proves the UDP socket is listening in `s` before we alter the path.

### Apply one change

Add one scoped rule in namespace `s` that drops inbound UDP to the echo port. The rule models an outer firewall policy. It does not parse QUIC and it does not alter the TLS service on TCP/8443. Check whether a table with this lab-only name already exists before creating it; if it does, inspect and remove only your own prior experiment state.

```bash
sudo ip netns exec s nft list table inet ossq23 2>/dev/null && \
  { echo 'ossq23 table already exists; inspect it before continuing' >&2; exit 1; } || true
sudo ip netns exec s nft add table inet ossq23
sudo ip netns exec s nft 'add chain inet ossq23 input { type filter hook input priority 0; policy accept; }'
sudo ip netns exec s nft add rule inet ossq23 input udp dport 9443 counter drop
sudo ip netns exec s nft list table inet ossq23
```

The table name is unique to this post. On a system whose policy or container security settings prohibit `nft` in a namespace, stop here and do not translate the rule onto a real interface. That environment cannot run this treatment as written.

### Compare

Repeat the exact UDP request, then inspect the rule counter and make sure the TCP TLS endpoint still accepts a connection. This is a controlled change: same endpoints, same UDP payload, same port, one new policy rule.

```bash
sudo ip netns exec c python3 -c \
  'import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(1); s.sendto(b"probe",("10.77.0.2",9443)); exec("try:\n print(s.recvfrom(2048)[0].decode())\nexcept socket.timeout:\n print(\"timeout: UDP reply absent\")")'
sudo ip netns exec s nft list table inet ossq23
sudo ip netns exec c openssl s_client -connect 10.77.0.2:8443 \
  -servername netlab.local -tls1_3 -brief </dev/null 2>&1 |
  grep -E 'Protocol version|Protocol  *:' || true
```

Read: the UDP client should print `timeout: UDP reply absent`; the `nft` rule's packet counter should increase by at least one for the probe; the separate TCP connection should still report TLS 1.3. If the counter stays at zero, the packet did not reach that hook or the address/port is wrong, so the timeout does not prove this rule caused the loss. If the counter rises and the client receives a reply, inspect an earlier response or an unexpectedly duplicated server. This lab's expected result is a qualitative before/after state with a one-second client timeout, not a promised Internet throughput number.

The production translation is read-only: compare client and edge packet captures or protocol counters for a short, consented test flow; inspect firewall rule counters and edge QUIC handshake counters; do not install a probe drop rule on a live interface. The lab demonstrates why “UDP exists on this machine” is weaker than “this user path passes UDP to this service.” It also shows why a successful TCP/TLS fallback can coexist with a blocked UDP attempt.

### Reset

Remove only the `ossq23` table and stop only the processes started above. The certificate and capture live in the temporary lab directory. Do not delete the base namespaces or their veth pair; other series experiments use them.

```bash
sudo ip netns exec s nft delete table inet ossq23
sudo kill "$UDP_PID" "$TLS_PID" 2>/dev/null || true
wait "$UDP_PID" "$TLS_PID" 2>/dev/null || true
rm -rf "$LAB_DIR"
```

### What this experiment cannot establish

The `nft` rule is an explicit drop policy that we installed. Cloudflare's case involved heterogeneous real middleboxes whose behavior the browser teams had to infer and test. The lab does not emulate those proprietary parsers. The UDP echo service is not QUIC, so the lab cannot measure QUIC congestion control, TLS-in-QUIC setup, connection migration, or HTTP/3 fallback. It isolates two necessary observations: a TLS 1.3 ClientHello can carry a TLS 1.2-looking legacy version, and an outer UDP policy can prevent an otherwise working datagram application from receiving traffic. To test a real HTTP/3 deployment, add a client and server with named QUIC implementation versions, capture the same path, and record fallback behavior for the actual client population.

## Key takeaways

- A specification can leave room for change while the installed path quietly forbids that change. Check transit behavior when endpoints disagree with the observed outcome.
- TLS 1.3 uses `legacy_version = 0x0303` and `supported_versions` to negotiate TLS 1.3. Its compatibility session ID and dummy change-cipher-spec behavior respond to measured middlebox intolerance, not a return to TLS 1.2 security.
- GREASE routinely exercises unknown-value handling, chiefly at TLS endpoints. It cannot guarantee a new version or a UDP flow will pass every intermediary.
- QUIC carries transport behavior over UDP and protects much of its internal state. IP, UDP, invariant QUIC fields, sizes, timing, and derivable Initial contents remain observable in defined ways.
- UDP reachability is a path property. Keep TCP fallback healthy and measure the unsuccessful UDP cohort's added time.
- Debug from the last confirmed observation: client packet, edge packet, handshake stage, then HTTP request. A generic connection error does not identify the layer.

## Further reading

- [RFC 8446: TLS 1.3, especially Appendix D.4](https://www.rfc-editor.org/rfc/rfc8446.html#appendix-D.4), August 2018.
- [RFC 8701: GREASE for TLS extensibility](https://www.rfc-editor.org/rfc/rfc8701.html), January 2020.
- [RFC 8999: QUIC version-independent properties](https://www.rfc-editor.org/rfc/rfc8999.html), May 2021.
- [RFC 9000: QUIC transport](https://www.rfc-editor.org/rfc/rfc9000.html) and [RFC 9001: TLS in QUIC](https://www.rfc-editor.org/rfc/rfc9001.html), May 2021.
- [Cloudflare's TLS 1.3 deployment account](https://blog.cloudflare.com/why-tls-1-3-isnt-in-browsers-yet/), December 26, 2017.
- [Using UDP for Internet Transport Evolution](https://arxiv.org/abs/1612.07816), December 2016.
- [The senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) ties protocol failures back to the complete request path.
