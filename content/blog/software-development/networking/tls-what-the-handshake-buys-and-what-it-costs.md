---
title: "TLS: What the Handshake Buys and What It Costs"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Locate TLS in a request's latency budget, price its message flights, and decide when resumption and early data are safe."
tags:
  ["networking", "distributed-systems", "tls", "tls-handshake", "https", "latency", "forward-secrecy", "session-resumption", "zero-rtt", "security"]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 43
image: "/imgs/blogs/tls-what-the-handshake-buys-and-what-it-costs-1.webp"
---

An HTTPS request is slow. The TCP connect completed in 150 ms, but the client did not send the HTTP request until 300 ms. An engineer might call the entire delay “network latency” and move on. That loses the useful distinction. TCP established a transport path; TLS then used that path to agree on keys, authenticate the peer, and prove that both sides saw the same handshake. The second 150 ms has a different owner and different remedies.

The opening latency ladder below is the mental model for this post. DNS chooses an address, TCP establishes the connection, TLS establishes an authenticated encrypted channel, and only then does an ordinary HTTPS client send application data. The labels are phases rather than sample timings. The TLS cell is the one we will measure and explain.

![The request latency ladder highlights TLS after TCP and before the HTTP request.](/imgs/blogs/tls-what-the-handshake-buys-and-what-it-costs-1.webp)

For a full handshake over an already established TCP connection, the familiar approximation is two network round trips for TLS 1.2 and one for TLS 1.3. That is a statement about message flights under a clean path, not a promise about every deployed connection. TLS 1.2 resumption is shorter than a full handshake. A TLS 1.3 HelloRetryRequest can add a flight. Certificate validation, packet loss, a proxy, and CPU can add time. Connection reuse removes the handshake from the next request entirely. We need to know which case we actually have before assigning a price.

This post extends the [request path walkthrough](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) and the [TCP handshake cost](/blog/software-development/networking/the-tcp-handshake-and-what-it-costs-you). It stays at the secure channel boundary. Certificate chain operations get their own treatment in [certificates and trust](/blog/software-development/networking/certificates-chains-and-the-trust-you-inherit), while [service identity and mTLS](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) owns the policy question of which workload is allowed to talk to which.

## What the handshake has to establish

**A fast encrypted socket is useless if it belongs to the wrong peer.** The handshake has four jobs. First, negotiate a mutually supported protocol version and cryptographic algorithms. Second, authenticate the server against an identity the client intended to reach. Third, agree on fresh traffic secrets that a passive observer cannot calculate from captured packets. Fourth, bind the negotiation to a transcript so an attacker cannot silently edit the choices or substitute messages.

Encryption of the application bytes is the most visible outcome, but it is not the entire service. Without authentication, a person in the middle can establish one encrypted connection with each endpoint and read both. Without transcript integrity, a downgrade or substitution can change what the endpoints believe they negotiated. Without fresh secrets, compromising a long-lived key later may expose old recordings. Without record authentication, an attacker could modify ciphertext or inject apparently valid responses.

TLS does not prove that the application is correct, that the DNS answer was the one you intended, or that a service behind a reverse proxy is trustworthy. It authenticates a cryptographic peer under a configured trust and name-checking policy. If a client terminates at an edge proxy, that client has authenticated the edge endpoint. Any separate edge-to-origin channel has its own handshake and trust decision. The broader architecture choice belongs to [termination and re-encryption](/blog/software-development/networking/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end).

Think of the handshake as two people agreeing on a sealed meeting room. They need to know which room, verify the other person's badge, create a room key neither brought in a reusable form, and confirm that both heard the same instructions. The subsequent conversation uses a much cheaper symmetric key. The public-key operations establish and authenticate that key; they do not encrypt every application byte with the certificate's public key.

The [TLS 1.3 specification, published in August 2018](https://www.rfc-editor.org/rfc/rfc8446), names the handshake, record, and alert protocols. [RFC 9846](https://www.rfc-editor.org/rfc/rfc9846), published in 2025, obsoletes that edition. I use RFC 8446 for the historically deployed TLS 1.3 message flow and its detailed replay discussion, and RFC 9846 for the current standard status. The [TLS 1.2 specification](https://www.rfc-editor.org/rfc/rfc5246) supplies the comparison. The wire observations in a particular library can differ in compatibility records and extension details, so read the negotiated version and packet sequence rather than assuming every capture looks like a textbook diagram.

## Count flights, then convert them to time

**A round trip is a scheduling constraint, not a byte count.** RTT is the elapsed time for a packet to reach the peer and for a response to return. The handshake cannot advance past a message it has not received. A protocol that needs two sequential request-response dependencies therefore has a larger latency floor than one that needs a single dependency, even if its cryptography is faster.

![A message-flight comparison shows the two TLS 1.2 flights and the one-flight TLS 1.3 full handshake.](/imgs/blogs/tls-what-the-handshake-buys-and-what-it-costs-2.webp)

After TCP is ready, a typical full TLS 1.2 exchange starts with ClientHello. The server responds with ServerHello, its certificate, key-exchange material when that suite requires it, and ServerHelloDone. The client checks the server, sends its key-exchange contribution and Finished, then waits for the server's Finished before treating the full exchange as complete. In flight terms, the client sees the server's first response after about one RTT and receives the server's Finished after roughly a second RTT. [RFC 5246 section 7.4](https://www.rfc-editor.org/rfc/rfc5246#section-7.4) gives the exact message flow. The cipher suite matters: TLS 1.2 could use ephemeral Diffie-Hellman or static RSA key transport. “TLS 1.2 lacks forward secrecy” is therefore false as a universal statement.

For a typical full TLS 1.3 exchange, the client sends a key share in ClientHello. The server can choose parameters, send its own key share, authenticate itself, and send Finished in its first response flight. The client can verify that flight and send its Finished together with the first protected application request. The server's response can follow. [RFC 8446 section 2](https://www.rfc-editor.org/rfc/rfc8446#section-2) calls this a 1-RTT handshake. The client does not have to wait for a separate server Finished after sending its own Finished because the server already sent it in the first response flight.

The phrase “TLS 1.3 saves one RTT” compares **full handshakes over the same established transport path**. It does not compare a resumed TLS 1.2 session with a full TLS 1.3 session, and it does not mean TCP plus TLS takes one RTT. With conventional HTTPS over TCP, add the TCP connection's approximately one RTT to the TLS flights. A cold full request with no DNS cost and no server work has a simplified setup floor near three RTTs for full TLS 1.2 and two RTTs for full TLS 1.3 before a response can travel back. The HTTP request and first response contribute another network propagation interval to time-to-first-byte. Actual packet timing overlaps in places; these are explanatory floors, not exact equations for a particular curl result.

Let $R$ be the path RTT, $C$ the non-network TLS work visible to the client, and $L$ the TLS setup time after TCP. A useful **planning model**, not a protocol equation, is $L_{1.2,\mathrm{full}} \approx 2R+C$ and $L_{1.3,\mathrm{full}} \approx R+C$. At a verified 5 ms RTT, that model prices TLS flights at about 10 ms versus 5 ms before local work. At a verified 150 ms RTT, it prices them at about 300 ms versus 150 ms. The one-RTT difference becomes large because the client is waiting for causally necessary information, not because TLS 1.2 necessarily performs a huge amount of CPU work.

| Full-handshake scenario | RTT assumption | TLS 1.2 network-flight floor | TLS 1.3 network-flight floor | Approximate difference | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| Nearby peer | 5 ms | 10 ms | 5 ms | 5 ms | Derived here: 2R versus R from RFC 5246 and RFC 8446 |
| Distant peer | 150 ms | 300 ms | 150 ms | 150 ms | Derived here: 2R versus R from RFC 5246 and RFC 8446 |

This table contains no measurements. To compare it with a request, verify the RTT to the actual TLS peer, identify whether this was a full handshake, then subtract the appropriate `curl` timing fields. An anycast edge, forward proxy, or service mesh sidecar may be the TLS peer even when the hostname names a distant application.

The 1.3 handshake can be longer than the clean one-RTT case if the server rejects the client's proposed key share and sends HelloRetryRequest. That is an extra exchange specified by [RFC 8446 section 2.1](https://www.rfc-editor.org/rfc/rfc8446#section-2.1). A dropped ClientHello, retransmitted TCP segment, certificate verification delay, or large certificate flight can dominate the neat RTT model. The RTT arithmetic is a hypothesis to test, never a performance guarantee.

## The keys are separate from the certificate

**The certificate proves an identity; the ephemeral exchange creates the session secret.** In a full TLS 1.3 handshake, the client and server contribute ephemeral Diffie-Hellman shares. Each can derive the same shared secret using its private share and the peer's public share. A passive observer sees both public shares but cannot feasibly calculate the secret under the chosen group's security assumptions. The key schedule then derives separate handshake and application traffic secrets. The server's certificate and CertificateVerify signature authenticate the transcript and connect the handshake to the certified identity.

![A key exchange graph separates identity proof, ephemeral shared secret, and traffic-key derivation.](/imgs/blogs/tls-what-the-handshake-buys-and-what-it-costs-3.webp)

That separation answers a recurring incident question: if the server certificate's private signing key is stolen next month, can an attacker decrypt a packet capture from last month? For a properly executed ephemeral full handshake whose ephemeral secrets and traffic keys were erased, the signing key alone does not reconstruct the old session secret. This is forward secrecy. It has conditions. A compromised endpoint that retained plaintext or session keys defeats it. An attacker who controlled the endpoint during the session can read the traffic. The statement is about later compromise of a long-lived authentication key, not a magic promise that data remains secret after every possible compromise.

TLS 1.2 can also provide forward secrecy when it negotiates an ephemeral ECDHE or DHE suite. Static RSA key transport in TLS 1.2 does not have the same property: possession of the RSA private key can expose a recorded premaster secret exchange. TLS 1.3 removed static RSA and static Diffie-Hellman key-exchange suites, a change documented in [RFC 8446 section 1.2](https://www.rfc-editor.org/rfc/rfc8446#section-1.2). Version labels alone do not tell you what a 1.2 connection negotiated; inspect the cipher suite.

TLS 1.3 resumption adds a nuance we cannot hide under the phrase “TLS 1.3 has forward secrecy.” The server can offer a pre-shared key through a session ticket. On the next connection, the endpoints can combine that PSK with a fresh ephemeral DHE exchange, called `psk_dhe_ke`, or use the PSK without a fresh DHE exchange, called `psk_ke`. The latter loses forward secrecy for the application data if the relevant PSK is compromised. The former preserves forward secrecy for the later 1-RTT traffic under the stated key-erasure assumptions. [RFC 8446 section 2.2](https://www.rfc-editor.org/rfc/rfc8446#section-2.2) states this distinction explicitly. Early 0-RTT data has weaker forward-secrecy guarantees even when the later connection uses fresh DHE, because its encryption keys are derived before the new server share arrives.

An implementation's default selection, ticket lifetime, key rotation, and storage policy matter operationally. It is unsafe to infer the resumption mode from “TLS 1.3” in an access log. Capture or instrument the negotiated handshake, and use a TLS library that exposes the relevant session and early-data state if the risk decision depends on it. The certificate chain's identity policy is separate; [certificate chains and trust](/blog/software-development/networking/certificates-chains-and-the-trust-you-inherit) covers that boundary in depth.

## Session resumption is a remembered secret

**Resumption is a latency optimization with state behind it.** A successful initial handshake can leave the client with material it presents on a future connection. TLS 1.2 used session IDs and session tickets, allowing an abbreviated handshake. TLS 1.3 uses a PSK derived from a prior connection, commonly represented to the client by a NewSessionTicket. The client includes a ticket identity and binder in a later ClientHello; the server must recover or look up the corresponding secret and decide whether to resume. A server that cannot find or accept it falls back to a full handshake.

![A matrix distinguishes full handshakes, PSK-DHE resumption, PSK-only resumption, and early data.](/imgs/blogs/tls-what-the-handshake-buys-and-what-it-costs-4.webp)

The ticket is not a universal pass. It is bound to the server configuration and protocol context that issued it. It expires, can be rotated, can be deliberately rejected, and might not be available at a different edge cluster. A client with a ticket can still have a one-RTT TLS handshake. Resumption often saves expensive authentication and certificate work even when it does not remove the dependency on the server's first flight. Saying “resumed equals zero RTT” confuses resumption with the optional early-data feature.

The security cost of tickets is persistence. A server that encrypts tickets needs to control the ticket-protection keys and their lifetime across a fleet. Sharing those keys broadly makes resumption reliable across nodes but increases the number of machines whose compromise could reveal the resumption material. Rotating them aggressively reduces exposure but may lower the resumption hit rate. That is a real operational trade-off, not a recommendation to maximize either metric. [RFC 8446 section 8](https://www.rfc-editor.org/rfc/rfc8446#section-8) discusses single-use tickets and anti-replay state because early data makes these details much more important.

The application can often get a larger benefit without changing TLS at all: reuse an existing secure connection. A warm HTTP/2 or HTTP/1.1 keep-alive connection has already paid DNS, TCP, and TLS setup. If the request fits that connection, the client's request starts without a new handshake. Connection pools therefore change the denominator of a “handshake percentage” metric. A fleet that creates a new connection for every request pays the setup cost repeatedly; a fleet that reuses connections pays it per connection lifetime. The other side of that choice is stale or overloaded connections, balancing behavior, and idle-resource cost. Diagnose [connection reuse and pooling](/blog/software-development/networking/http-1-1-keep-alive-and-the-six-connection-tax) at the application-protocol layer rather than treating resumption as a substitute for pooling.

For a simple cost estimate, suppose a service sends $N$ requests over $K$ distinct TLS connections during the measurement window, all to peers with RTT $R$. If every connection uses a full TLS 1.3 handshake, the flight contribution per request averaged across the window is approximately $KR/N$. This is an **amortization model**, not a property of TLS. For 10,000 requests, 100 connections, and 50 ms RTT, the model allocates $100\times50/10000=0.5$ ms of handshake flight time per request. With 10,000 one-request connections, it allocates 50 ms per request. The model ignores CPU, queuing, failures, and overlap, but it shows why reducing connection churn can matter more than trimming a cryptographic primitive.

The connection count must be observed, not inferred from request count. A load balancer may close idle connections, an autoscaler may continually add cold instances, or DNS and routing changes may send the next request to a peer that cannot resume. Authentication metadata and [service discovery](/blog/software-development/networking/service-discovery-from-dns-to-registries-to-xds) also influence which endpoint receives the ticket. A blanket statement that a repeat user will resume is too strong.

## Zero round trips still needs a prior connection

**Zero RTT means early application bytes ride with the first TLS flight; it does not mean a new client skips trust establishment.** A client must first complete a connection and receive a usable ticket. On a later connection it can send a ClientHello containing a PSK identity and encrypted early application data before seeing the new server's response. The server can process that early data if its policy and ticket state permit it. If it rejects early data, the connection can continue as a normal handshake, but the application must decide whether a request should be retried. [RFC 8446 section 2.3](https://www.rfc-editor.org/rfc/rfc8446#section-2.3) is precise about these alternatives.

The animation below makes the timing claim. On an ordinary one-RTT TLS 1.3 connection, the first application request waits for the server's handshake flight. On an accepted early-data connection, request bytes leave with ClientHello. In both cases the new TCP connection, if we are using TLS over TCP, still has its own transport establishment. “Zero RTT” names the TLS application-data start relative to the TLS flight, not the total wall-clock time of a new HTTPS request.

<figure class="blog-anim" aria-label="TLS one RTT and early-data timing"><style>.tls19-svg{width:100%;height:auto;max-width:960px;display:block;margin:auto}.tls19-flow{animation:tls19-flight 10s ease-in-out infinite}.tls19-early{animation:tls19-early 10s ease-in-out infinite}.tls19-late{animation:tls19-late 10s ease-in-out infinite}@keyframes tls19-flight{0%,8%{transform:translateX(0);opacity:1}35%,45%{transform:translateX(370px);opacity:1}55%,100%{transform:translateX(370px);opacity:0}}@keyframes tls19-early{0%,8%{transform:translateX(0);opacity:1}35%,45%{transform:translateX(370px);opacity:1}55%,100%{transform:translateX(370px);opacity:0}}@keyframes tls19-late{0%,48%{opacity:0;transform:translateX(0)}55%{opacity:1;transform:translateX(0)}85%,100%{opacity:1;transform:translateX(370px)}}@media (prefers-reduced-motion:reduce){.tls19-flow,.tls19-early,.tls19-late{animation:none;opacity:1;transform:none}}</style><svg class="tls19-svg" style="width:100%;height:auto;max-width:960px" viewBox="0 0 960 350" role="img" aria-label="On a resumed TLS connection, accepted early data travels with ClientHello before the server flight; ordinary application data waits"><rect x="20" y="20" width="920" height="310" rx="18" fill="#fff" stroke="#adb5bd"/><text x="65" y="60" font-size="25" fill="#212529">Application-data start on a new TLS connection</text><text x="70" y="110" font-size="18" fill="#495057">client</text><text x="800" y="110" font-size="18" fill="#495057">server</text><line x1="130" y1="130" x2="830" y2="130" stroke="#495057" stroke-width="3"/><line x1="130" y1="230" x2="830" y2="230" stroke="#495057" stroke-width="3"/><text x="30" y="164" font-size="18" fill="#212529">1-RTT</text><text x="30" y="264" font-size="18" fill="#212529">0-RTT</text><circle class="tls19-flow" cx="200" cy="130" r="17" fill="#a5d8ff" stroke="#1c7ed6" stroke-width="3"/><text x="250" y="122" font-size="17" fill="#212529">ClientHello</text><circle class="tls19-late" cx="200" cy="130" r="10" fill="#ffec99" stroke="#e67700" stroke-width="3"/><text x="610" y="161" font-size="16" fill="#495057">request after server flight</text><circle class="tls19-early" cx="200" cy="230" r="17" fill="#a5d8ff" stroke="#1c7ed6" stroke-width="3"/><text x="250" y="221" font-size="17" fill="#212529">ClientHello + early request</text><text x="130" y="305" font-size="17" fill="#495057">Requires a prior ticket; server may reject early data</text></svg><figcaption>Accepted early data moves the request into the first TLS flight; the prior ticket and replay policy are prerequisites.</figcaption></figure>

That timing opportunity comes with a specific security cost. Early data is encrypted using secrets derived from the prior session ticket, before the new server's random contribution and key share arrive. An observer cannot simply read the early bytes from the wire, but an active attacker who records the ClientHello and encrypted early-data records can replay them into another acceptable connection. TLS cannot promise the same cross-connection uniqueness for early data that it provides for ordinary one-RTT application data. [RFC 8446 section 8 and Appendix E.5](https://www.rfc-editor.org/rfc/rfc8446#section-8) describe both the attack and limits of protocol-level defenses.

![A replay threat graph shows a copied early-data flight reaching a second acceptance boundary.](/imgs/blogs/tls-what-the-handshake-buys-and-what-it-costs-5.webp)

Suppose the early request is `POST /transfer` and its application handler moves funds. If the same protected early-data flight is accepted twice, two transfers can occur even though the attacker never learned the plaintext. The precise amount is irrelevant to the mechanism. The vulnerable operation is a non-idempotent state change: doing it twice has a different effect from doing it once. An early `GET` is often a safer candidate, but method names are only a starting point. Some `GET` endpoints trigger billing, side effects, expensive work, or timing-sensitive cache behavior. The RFC also warns that many replays can matter even for otherwise idempotent operations.

There are multiple defenses at different layers. The server can use single-use tickets, a replay cache, freshness checks, and cluster scoping. These reduce acceptance of duplicate early flights when all accepting nodes share a consistent view. They do not prove that a retried application operation is safe. The application can decline early data for state-changing routes, require an idempotency key stored with the operation's result, or defer processing until the handshake reaches the stronger one-RTT state. The client must be prepared for an early-data rejection and must choose whether a retry is safe. A TLS library should not silently transform a rejected early request into a new operation with changed semantics.

The cluster boundary is important. An edge fleet may have many locations or independently replicated replay stores. If two locations accept the same ticket before their replay state converges, the protocol guard can be weaker than the diagram on a single server suggests. A load balancer that sends retries to a different shard creates the same concern. This is one reason “we enabled anti-replay” is not a complete application guarantee. Document the consistency scope and the operation's idempotency policy together. [RFC 8446 section 8](https://www.rfc-editor.org/rfc/rfc8446#section-8) describes the cluster assumptions explicitly.

There is also a correctness cost. The server may reject early data because the ticket is old, the selected application protocol changed, the ticket key rotated, or anti-replay policy did not accept the flight. The client then has to continue with the regular handshake and decide whether it can send the request again. A benchmark that counts only accepted early-data connections selects the best case. A production latency report needs acceptance rate, rejection rate, fallback latency, and the distribution of request types that were eligible. A high hit rate for tickets does not automatically imply a high acceptance rate for early data.

## Measure the boundary that curl can see

**A single `time_total` hides the layer.** Curl's `--write-out` exposes timestamps relative to the start of the transfer. For HTTPS over a new TCP connection, `time_connect` marks the completion of the TCP connection and `time_appconnect` marks completion of the TLS or other application-layer handshake. Subtracting them gives a client-observed TLS interval. `time_namelookup` marks name resolution; `time_starttransfer` marks the first received response byte. The current [curl write-out documentation](https://curl.se/docs/manpage.html#-w) defines the fields. The subtraction is an observation of curl's boundaries, not a packet-level decomposition of certificate verification, key exchange, and network waiting.

![A diagnostic tree turns curl timing deltas into the next packet or certificate check.](/imgs/blogs/tls-what-the-handshake-buys-and-what-it-costs-6.webp)

Use a new curl process for each cold connection test. A second URL in one process may reuse its first connection, which makes the second transfer's setup fields represent a different situation. A proxy changes the peer represented by `time_connect`. HTTP/3 uses QUIC rather than a separate TCP handshake, so this post's TCP-plus-TLS subtraction model is the wrong mental picture for that transport. Pin HTTP/1.1 or HTTP/2 over TCP for this experiment, and record the negotiated protocol. A redirect can open another connection; suppress redirects while establishing the baseline.

```bash
curl --http1.1 --no-progress-meter --output /dev/null \
  --write-out 'dns=%{time_namelookup} tcp=%{time_connect} tls=%{time_appconnect} first=%{time_starttransfer} total=%{time_total} version=%{http_version}\n' \
  https://example.com/
```

This is a read-only diagnostic against a public example, not a controlled latency experiment. The elapsed phases are `DNS = time_namelookup`, `TCP = time_connect - time_namelookup`, `TLS = time_appconnect - time_connect`, `post-TLS to first byte = time_starttransfer - time_appconnect`, and `remaining transfer = time_total - time_starttransfer`. These differences are an explanatory partition of one curl transfer. A client may perform additional local work inside each boundary, and the `post-TLS` interval includes request transmission and server time. It does not isolate server CPU.

Suppose a new connection produces `time_namelookup=0.004`, `time_connect=0.154`, `time_appconnect=0.307`, and `time_starttransfer=0.500` seconds. These numbers are an **illustrative calculation**, not measured output. The DNS phase is 4 ms, the TCP phase is $154-4=150$ ms, the TLS phase is $307-154=153$ ms, and the interval from secure readiness to first byte is $500-307=193$ ms. If a separate ping and capture confirm roughly 150 ms RTT to the TLS endpoint and the negotiated protocol is a normal TLS 1.3 full handshake, one RTT of TLS flight is plausible. We still have to check local validation and packet loss before declaring the cause.

| Illustrative field | Cumulative value | Derived phase | Source |
| --- | ---: | ---: | --- |
| `time_namelookup` | 0.004 s | DNS 4 ms | Illustrative values; arithmetic shown above |
| `time_connect` | 0.154 s | TCP 150 ms | Illustrative values; subtract `time_namelookup` |
| `time_appconnect` | 0.307 s | TLS 153 ms | Illustrative values; subtract `time_connect` |
| `time_starttransfer` | 0.500 s | Post-TLS to first byte 193 ms | Illustrative values; subtract `time_appconnect` |

If `time_connect - time_namelookup` grows but `time_appconnect - time_connect` remains stable, inspect TCP setup, route, and retransmits. If only the TLS delta grows, inspect handshake version and resumption status, certificate validation, HelloRetryRequest, and packet loss during the TLS flight. If both remain stable but `time_starttransfer - time_appconnect` grows, investigate request transmission, an upstream proxy, server work, or queueing. [The latency-budget post](/blog/software-development/networking/the-latency-budget-speed-of-light-serialization-and-queueing) provides the physical and queueing terms that sit underneath these boundaries.

Do not use `-k` as a performance fix. It disables certificate validation, changing the security contract of the measurement. If you need to distinguish validation cost from network cost in an isolated lab, record that experiment explicitly and never turn its insecure mode into a service recommendation. A larger certificate chain can also cause extra TCP segments or losses, so packet capture and library timing can be necessary when the curl delta is suspicious. Chain construction and expiry are covered in the sibling certificate post.

## A dated deployment lesson: Cloudflare's early-data gate

The public case here is a feature rollout, not an outage. On **March 15, 2017**, Cloudflare engineer Nick Sullivan described the company's introduction of TLS 1.3 0-RTT resumption in [Cloudflare's own engineering write-up](https://blog.cloudflare.com/introducing-0-rtt/). The article is unusually useful because it states the desired latency improvement and names the new failure mode in the same place. It also contains a later update saying 0-RTT is no longer enabled by default; the page's historical and current statements must not be flattened into one timeless product setting.

The user-visible promise was a faster repeat visit. The network mechanism was precise: a returning client that held a usable ticket could place its first encrypted request in the ClientHello flight rather than waiting for the server's new flight. Cloudflare's article illustrated a browser example with a roughly 250 ms shorter waiting bar. That was the article's demonstration, not a universal Cloudflare median, and the page did not publish enough conditions for us to use it as a general benchmark. Its exact benefit depends on path RTT and whether the request is actually eligible and accepted.

The trigger for the security concern was not a discovered break of encryption. It was the protocol's intentional ordering: the server can act on early data before it has fresh proof of a unique new handshake. A recorded early flight can be replayed. The contributing condition is any application route that treats a repeated request as a new state-changing command. The blast radius multiplier is an edge fleet or origin integration where anti-replay state and application idempotency are not consistently enforced across all accepting paths.

Cloudflare's 2017 description says its rollout answered only GET requests with no query parameters over 0-RTT, limited early-request size and replay time, and passed a `Cf-0rtt-Unique` header toward the origin to identify a resumption attempt. Those are source-attributed choices from that deployment, not requirements of TLS 1.3 and not proof that every GET is safe. The article's header example is a token format, not a protocol guarantee for arbitrary origins. The later note that the feature is no longer default is material to any current deployment decision. Check the provider's current settings and application behavior before assuming its historical gate applies to your account.

The transferable guardrail is a three-part review. First, name the exact requests eligible for early data. Second, prove those requests remain correct when repeated, including when they reach a different edge or origin. Third, measure accepted, rejected, and retried early-data requests separately. If you cannot write down those answers, a one-RTT resumed connection is the safer baseline. It still gets the cryptographic and certificate-work benefits of resumption without moving application execution before the new server flight.

This case also shows why a performance optimization can change the trust contract. A feature flag called `0-RTT` sounds like a timing knob. In fact it changes when an application may see a request and what uniqueness it may assume. Security reviewers should read the request path and retry policy, not merely the cipher list. That is the same principle behind [timeouts and retries](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right): retries need an application operation whose semantics remain correct under duplication.

## Where the clean RTT model breaks

**The client-observed TLS interval has several possible owners.** Network flight count is the first hypothesis, not the final diagnosis. A slow `time_appconnect - time_connect` can come from the path, the negotiated protocol, the credential chain, local work, or an intermediary. Each has a different discriminating observation.

The path can lose a handshake segment. A large certificate flight may span multiple TCP segments. Losing one of them can stall completion while TCP recovers, even though the successful packets show the expected TLS 1.3 message order. In that case, a packet capture reveals a retransmission or duplicate acknowledgements, and the timing delta is no longer approximately one RTT. Disabling certificate validation would not repair a missing TCP segment. A larger receive window would not repair a path MTU black hole. First look for the point where progress stopped.

The protocol can take an extra turn. In TLS 1.3, HelloRetryRequest asks the client to send a new ClientHello with a suitable key share. The exchange is legitimate, but it changes the flight count. A deployment with inconsistent supported groups between client and server may pay that extra delay repeatedly. The right fix is to examine negotiated groups and client key-share offers, not to declare TLS 1.3 “slow.” The current standard and [RFC 8446's HelloRetryRequest section](https://www.rfc-editor.org/rfc/rfc8446#section-4.1.4) define its conditions.

The credential chain can cause local and network cost. The server transmits its certificate chain during a full certificate-authenticated handshake, and the client validates identity, dates, signatures, and policy. The chain may not fit in one transport segment. Verification can consult local trust stores and application policy. Some clients perform revocation-related work under their own policy. If the endpoint sends an incomplete chain, validation may fail outright or trigger implementation-specific recovery. Never assume that a successful handshake everywhere means the chain is healthy for every client. The sibling certificate post owns chain construction; this post cares that its bytes and validation work live inside the observed TLS interval.

The peer may not be the origin. A corporate forward proxy, CDN edge, L7 gateway, or mesh sidecar can terminate the client-side TLS connection. Curl then times the client-to-terminator handshake. A new proxy-to-origin handshake may occur later and show up in `time_starttransfer - time_appconnect`, not in curl's `time_appconnect - time_connect`. In a multi-hop path, one client timer cannot expose every secure channel. Trace identifiers and proxy connection metrics are needed to assign the hidden hop. [Load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) explains why those boundaries exist; our concern is where the wire handshake is visible.

The client can change the comparison without changing a server. Curl's TLS backend and protocol preferences vary by build. A TLS 1.3 client may choose a different group, offer a ticket, or negotiate HTTP/2 through ALPN. A command run with `--tlsv1.2` can be misleading because curl options often set a minimum version unless a maximum is also specified. For a controlled TLS 1.2 comparison, use `--tls-max 1.2` and verify the negotiated version with `curl -v` or a packet capture. A laboratory server can be pinned to `-tls1_2` or `-tls1_3`. A public endpoint might refuse old versions or route to a different edge; that refusal is data, not a reason to weaken the experiment.

The resumed case has different visibility. Curl's `time_appconnect` tells you when this transfer's handshake completed. It does not, by itself, prove that a ticket was accepted or whether early data was sent. A fresh curl process may not persist a TLS session ticket across invocations. A reused connection may show almost no setup at all. For a resumption study, use a client that exposes session state and perform two connections with explicit ticket handling, then classify full, PSK-DHE, PSK-only, and early-data acceptance. Do not derive these labels solely from a shorter curl number.

Finally, CPU can dominate on a near peer. At a 5 ms RTT, a few milliseconds of validation, scheduler delay, or slow entropy and hardware operations can be comparable with a whole network flight. At a 150 ms RTT, the RTT term usually dominates a healthy full handshake, though that is still an empirical question. The same TLS configuration can therefore look CPU-bound inside one region and flight-bound across an ocean. A single threshold such as “TLS should be under 20 ms” has no meaning without the peer RTT and the handshake class.

| Observed pattern | Leading question | Discriminating evidence | Safe next move |
| --- | --- | --- | --- |
| `time_connect` rises, TLS delta stable | Is transport establishment delayed? | SYN/SYN-ACK timing, route, retransmits | Diagnose TCP path before TLS settings |
| TLS delta near one verified RTT | Is this a normal full TLS 1.3 flight? | Negotiated version and no HelloRetryRequest | Leave the handshake alone; consider peer proximity or reuse |
| TLS delta near two verified RTTs | Is this full TLS 1.2 or a retry path? | Negotiated version and packet sequence | Prefer 1.3 when both ends support it; fix retry cause |
| TLS delta far above flight model | Did a packet or validation step stall? | Capture and client TLS logs | Repair the observed loss or validation step |
| First-byte interval rises after TLS stays flat | Is a proxy or origin slow? | Proxy upstream connection and server traces | Move to the downstream owner |

The “near one RTT” and “near two RTTs” rows are diagnostic shapes derived from the protocol flights. They are not measured limits. Loss, local work, coalescing, and retries make overlap normal. Use the table to choose a capture, then let the capture overrule the model.

## HTTP has to participate in the early-data decision

The TLS layer can tell the application that bytes arrived as early data. It cannot decide whether `POST /orders`, `GET /export`, or a cache miss has safe business semantics. [RFC 8470, published in September 2018](https://www.rfc-editor.org/rfc/rfc8470), specifies how HTTP uses early data. It defines the `Early-Data` request header for intermediaries and the `425 Too Early` response for an origin that is unwilling to risk processing the request before the handshake completes. A client that receives 425 can retry under the RFC's rules after the early-data condition is gone. The route, method, authentication context, and retry behavior all matter.

This is why an edge terminator and origin need a shared story. The edge may accept 0-RTT from the client, then forward the HTTP request over an already established origin connection. The origin does not see the original TLS handshake. If the edge fails to label the request as early data, the origin cannot apply its own replay policy. Conversely, an origin that rejects every request with an `Early-Data` indication may negate the latency benefit while remaining correct. The right metric is completed safe request latency, including 425 and retry paths, not just the count of accepted TLS early records.

Idempotency needs a concrete key and storage boundary. Imagine an order API that accepts a client-generated operation ID and atomically records the result with that ID. If the same early request arrives twice, the handler returns the stored result instead of creating a second order. That can make a repeated operation safe, provided every processing region consults the same authoritative idempotency store or uses a conflict-safe protocol. A cache local to one edge does not provide that property across two edges. The first successful operation and the deduplication record must commit together; otherwise a crash between them can still leave a duplicate on retry. This is an application design requirement, not a TLS record-layer feature.

For an early-data rollout, write the eligibility predicate as if it were a security policy. Which methods and paths are permitted? Are query strings used as commands? Can a read cause a purchase, send an email, consume a one-time token, or reveal information through repeated timing measurements? Does the origin understand `Early-Data` and 425? Can it distinguish an ordinary application retry from a replayed early flight? Is anti-replay state shared across regions that hold the same ticket key? A “safe methods only” checkbox answers only the first line of this review.

The performance comparison also needs the rejected branch. Let $p$ be the fraction of eligible returning connections whose early data is accepted, $S$ the RTT saved when accepted, and $F$ the additional fallback cost when rejected. A simple **planning model**, not an RFC formula, is expected latency change $\Delta \approx -pS+(1-p)F$. If $p=0.9$, $S=150$ ms, and $F=150$ ms, the model predicts $-0.9(150)+0.1(150)=-120$ ms on average among eligible returning connections. If $p=0.2$ with the same costs, it predicts $-30+120=+90$ ms. These illustrative numbers are arithmetic, not a Cloudflare or browser measurement. They show why the acceptance rate and fallback path can reverse the result. They also exclude application replay harm, which cannot responsibly be priced as a small latency term.

| Illustrative early-data model input | Value | Meaning | Source |
| --- | ---: | --- | --- |
| Acceptance fraction $p$ | 0.9 or 0.2 | Hypothetical eligible returning connections | Illustrative model in this section |
| Saving $S$ | 150 ms | One RTT saved on accepted early data | Derived here from assumed 150 ms RTT |
| Fallback cost $F$ | 150 ms | Assumed extra RTT-equivalent cost after rejection | Illustrative model assumption |
| Expected change at $p=0.9$ | -120 ms | $-pS+(1-p)F$ | Derived here |
| Expected change at $p=0.2$ | +90 ms | $-pS+(1-p)F$ | Derived here |

The model is deliberately narrow. Real clients may hide some retry work in connection setup, and 425 handling depends on the HTTP stack. Security policy can refuse early data entirely on sensitive routes. Measure the four branches separately: eligible and accepted, eligible and rejected, ineligible but resumed, and full handshake. Report request latency percentiles for each branch and their traffic weights. A single aggregate p50 can improve while a rare rejected path becomes worse, which is exactly the path that may define user-visible tail latency.

## Capture the message sequence when timing is ambiguous

`curl -w` gives a boundary, not the message list. A targeted packet capture can tell whether the client sent ClientHello, whether the server replied with ServerHello or HelloRetryRequest, whether TCP retransmitted a handshake segment, and whether a proxy is the endpoint actually answering. On a lab namespace, capture only port `8443` with a bounded duration and a small snap length appropriate to the question. Do not collect broad production packet traces without access controls: even when application payload is encrypted, names, addresses, certificates or metadata, and timing can be sensitive.

```bash
timeout 15 ip netns exec c tcpdump -i c0 -nn -s 256 \
  -w /tmp/tls19-handshake.pcap 'host 10.77.0.2 and tcp port 8443'
tshark -r /tmp/tls19-handshake.pcap \
  -Y 'tls.handshake or tcp.analysis.retransmission' \
  -T fields -e frame.time_relative -e ip.src -e ip.dst \
  -e tls.handshake.type -e tcp.analysis.retransmission
```

Run the capture in one terminal while the curl request runs in another. The `timeout` bounds the capture; delete the lab file afterward. `tshark` field availability and decryption visibility depend on version. TLS 1.3 encrypts most handshake messages after ServerHello, so an ordinary passive capture without secrets will not necessarily display Certificate and Finished as named handshake messages. It can still reveal ClientHello, ServerHello, record sizes, timing, and TCP recovery. If you need the encrypted handshake details in a disposable lab, use the TLS library's key-log support only where authorized and protect that file as a traffic-decryption secret.

The key diagnostic is the **gap**, not the mere presence of a packet. A client-side trace that shows ClientHello leaving and ServerHello arriving one RTT later is consistent with the clean full 1.3 flight. A second ClientHello after HelloRetryRequest explains another dependency. Repeated TCP sequence ranges with a long idle gap point to loss recovery. A long delay before any ClientHello leaves points toward client scheduling or local cryptographic preparation. A long gap after the server's first flight but before curl reports `time_appconnect` suggests client validation or a missing later record. Pair the capture with the cumulative curl fields; neither alone proves every cause.

On a proxy path, capture at both sides of the terminator if possible. The client capture may show a perfect one-RTT TLS 1.3 handshake to the edge while the edge starts a separate TLS connection to an origin. In that case, the client's TLS delta stays healthy and its first-byte interval grows. The appropriate fix may be an origin connection pool, upstream reachability, or proxy certificate validation, not a change to the client-facing TLS suite. This is where the [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) earns its keep: name the precise boundary, then choose the observation at that boundary.

## Run it yourself

### Question

If we keep the client, TLS version, certificate, and server fixed but increase the RTT between the same two Linux namespaces from about 5 ms to about 150 ms, does the client-observed TLS interval grow by roughly the same 145 ms? That is the one-RTT TLS 1.3 flight hypothesis. The experiment measures a new full TLS 1.3 connection on each curl invocation. It does not claim to measure 0-RTT or resumption.

### Preconditions

Use the Linux `netlab` from the [first networking post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). It needs the existing `c` and `s` network namespaces, client interface `c0` at `10.77.0.1`, server interface `s0` at `10.77.0.2`, and a route between them. On macOS, run that lab inside the privileged Linux environment described in post 1. Use `iproute2` with `tc netem`, `ping`, OpenSSL with TLS 1.3 and `-addext` support, and curl with `time_appconnect` write-out support. Record their actual versions. Root or `CAP_NET_ADMIN` is required for qdisc changes; the following commands are for the named lab veth interfaces only. Do not apply them to a production interface.

The commands below assume an interactive root shell so the same shell can retain the server PID and lab path. Run `sudo -s` if that is how the lab VM grants access, then inspect the state before adding any qdisc. The shell blocks are exact commands for that root shell. The certificate and private key stay in a temporary lab directory. They are short-lived, and curl explicitly trusts only that lab certificate for this test.

```bash
set -euo pipefail
ip netns list
ip -n c addr show dev c0
ip -n s addr show dev s0
ip -n c route get 10.77.0.2
ip netns exec c tc qdisc show dev c0
ip netns exec s tc qdisc show dev s0
curl --version | head -1
openssl version
tc -V
```

Read the namespace and interface names, the route target, and both qdisc lines. Expected state: `c` and `s` exist, `10.77.0.2` routes through `c0`, and neither interface already has a lab impairment that you need to preserve. If either interface has a qdisc you care about, stop and use a disposable lab rather than overwriting it. Confirm that port `8443` is not already occupied in `s` with `ip netns exec s ss -lnt '( sport = :8443 )'`. No listening line is the expected state before this test.

### Baseline

Create a dedicated ephemeral certificate, start OpenSSL's built-in test server in `s`, then add 2.5 ms one-way delay on **each** veth end. The resulting RTT should be near 5 ms because a packet and its reply each traverse the delayed egress path. Verify it with `ping`; netem timing can vary across hosts and VMs.

```bash
TLS19_DIR=$(mktemp -d /tmp/tls19-lab.XXXXXX)
openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 1 \
  -subj '/CN=netlab.local' \
  -addext 'subjectAltName=DNS:netlab.local,IP:10.77.0.2' \
  -keyout "$TLS19_DIR/server.key" -out "$TLS19_DIR/server.crt"
ip netns exec s openssl s_server -accept 10.77.0.2:8443 \
  -cert "$TLS19_DIR/server.crt" -key "$TLS19_DIR/server.key" \
  -tls1_3 -www >"$TLS19_DIR/server.log" 2>&1 &
TLS19_SERVER_PID=$!
sleep 1
ip netns exec s ss -lnt '( sport = :8443 )'
ip netns exec c tc qdisc add dev c0 root netem delay 2.5ms
ip netns exec s tc qdisc add dev s0 root netem delay 2.5ms
ip netns exec c ping -n -c 5 10.77.0.2
```

Read the `ping` `rtt min/avg/max` line. Expected: the average is in the neighborhood of 5 ms, often with a few milliseconds of scheduling variance. If it is not close, use the **observed** RTT in the calculation; do not call this a 5 ms case by configuration alone. The certificate creation may print key-generation progress; it is not a timing measurement.

Now make five independent curl process invocations. `--resolve` gives curl the lab address while preserving the hostname for certificate checking and SNI. Each curl process starts a new connection and does not inherit an in-memory TLS session ticket from the previous process. OpenSSL's `-tls1_3` pins the server to the version under test. The response body goes to `/dev/null`; the write-out line records the timing fields we need.

```bash
: >"$TLS19_DIR/timings.tsv"
for n in 1 2 3 4 5; do
  ip netns exec c curl --http1.1 --noproxy "*" --no-progress-meter \
    --connect-timeout 5 --max-time 10 \
    --cacert "$TLS19_DIR/server.crt" \
    --resolve netlab.local:8443:10.77.0.2 \
    --output /dev/null \
    --write-out "5ms\t%{time_namelookup}\t%{time_connect}\t%{time_appconnect}\t%{time_starttransfer}\n" \
    https://netlab.local:8443/ >>"$TLS19_DIR/timings.tsv"
done
cat "$TLS19_DIR/timings.tsv"
awk -F '\t' '{printf "%s dns=%.3f tcp=%.3f tls=%.3f first_after_tls=%.3f\n", $1,$2,$3-$2,$4-$3,$5-$4}' "$TLS19_DIR/timings.tsv"
```

Read column four minus column three, printed as `tls`. Expected: a positive interval around one observed RTT plus local TLS work, commonly single-digit to low-tens of milliseconds on an unloaded VM with a 5 ms path. That is an experiment expectation, not a claimed measurement. The self-signed lab certificate is explicitly trusted through `--cacert`; no `-k` is needed. If curl reports a certificate failure, inspect the generated SAN and trust path instead of disabling verification.

### Apply one change

Change only the emulated one-way delay from 2.5 ms to 75 ms on each lab veth interface. Keep the same addresses, server process, TLS version, certificate, client, and URL. Because both directions are impaired, this should produce about 150 ms RTT. Again, the measured ping value wins over the configured value.

```bash
ip netns exec c tc qdisc replace dev c0 root netem delay 75ms
ip netns exec s tc qdisc replace dev s0 root netem delay 75ms
ip netns exec c tc qdisc show dev c0
ip netns exec s tc qdisc show dev s0
ip netns exec c ping -n -c 5 10.77.0.2
```

Read the two qdisc `delay` values and the `rtt min/avg/max` line. Expected: both qdiscs say 75 ms and the observed average RTT is around 150 ms, with VM-dependent variance. This is the treatment. We have changed one parameter, propagation delay, in both required directions; we have not changed the TLS server or application handler.

### Compare

Repeat the same independent requests and calculate the same deltas. The output is stored in the same temporary directory under a new label so the two conditions can be compared without copying numbers by hand.

```bash
for n in 1 2 3 4 5; do
  ip netns exec c curl --http1.1 --noproxy "*" --no-progress-meter \
    --connect-timeout 5 --max-time 10 \
    --cacert "$TLS19_DIR/server.crt" \
    --resolve netlab.local:8443:10.77.0.2 \
    --output /dev/null \
    --write-out "150ms\t%{time_namelookup}\t%{time_connect}\t%{time_appconnect}\t%{time_starttransfer}\n" \
    https://netlab.local:8443/ >>"$TLS19_DIR/timings.tsv"
done
awk -F '\t' '{printf "%s dns=%.3f tcp=%.3f tls=%.3f first_after_tls=%.3f\n", $1,$2,$3-$2,$4-$3,$5-$4}' "$TLS19_DIR/timings.tsv"
python3 - "$TLS19_DIR/timings.tsv" <<'PY'
import statistics
import sys
from collections import defaultdict
samples = defaultdict(list)
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        label, dns, connect, appconnect, first = line.strip().split("\t")
        samples[label].append(float(appconnect) - float(connect))
for label in ("5ms", "150ms"):
    values = samples[label]
    print(f"{label}: median TLS delta={statistics.median(values)*1000:.1f} ms, samples={len(values)}")
print(f"increase={1000*(statistics.median(samples['150ms'])-statistics.median(samples['5ms'])):.1f} ms")
PY
```

Read the median `TLS delta` for each label and the final `increase`. Expected: the increase should be broadly near the **measured RTT change**, about 145 ms in the intended setup, rather than near zero or twice that value. A range such as 110 to 190 ms is a reasonable sanity band for a lightly loaded disposable VM, not a benchmark guarantee. CPU scheduling, certificate generation differences across hosts, qdisc timer granularity, and transient loss widen it. The TCP delta should also grow by about one RTT; that is a separate transport cost. The post's claim is about the incremental TLS interval after TCP. If the TLS increase is closer to two RTTs, check for retransmits, unexpected version negotiation, and HelloRetryRequest before changing the explanation.

The application-first-byte delta is not expected to be constant: it includes request travel to the server, server handling, and response travel. Its growth is therefore a useful reminder that TLS is only one component of total request time. OpenSSL's built-in `-www` handler is a lab endpoint, not a production HTTP server. We are isolating the handshake flight, not measuring application performance.

### Reset

Remove only the qdiscs and process created for this test, then delete only its temporary files. The `c` and `s` namespaces and addresses belong to the shared `netlab` and remain in place for other posts.

```bash
ip netns exec c tc qdisc del dev c0 root
ip netns exec s tc qdisc del dev s0 root
kill "$TLS19_SERVER_PID"
wait "$TLS19_SERVER_PID" 2>/dev/null || true
rm -rf -- "$TLS19_DIR"
ip netns exec c tc qdisc show dev c0
ip netns exec s tc qdisc show dev s0
```

For production, use the read-only `curl --write-out` fields against a known hostname and collect a small packet capture at the client or TLS terminator if policy permits. Capture filters and access controls matter because handshake metadata and application traffic may contain sensitive information. Do not inject `tc` delay into a live interface to prove the point. The lab's causal change is useful precisely because it is isolated from production traffic.

## Choose the optimization at the right layer

The default performance question should be “why did this request need a new secure connection?” before it becomes “which handshake trick is fastest?” A full TLS 1.3 handshake over a 150 ms RTT has an unavoidable flight cost when the peer is that far away. If a client opens a new connection for each request, a healthy TLS stack will repeatedly pay it. Reusing a sound connection may remove DNS, TCP, and TLS setup from many requests at once. A close edge or regional peer can shorten the physical RTT. Resumption can reduce work on new connections that still need to exist. Early data can move eligible bytes earlier on returning connections, but it adds replay and fallback obligations. These are different interventions with different failure modes.

Consider a mobile client making one request after a long idle period. Its connection may have been closed by the phone, carrier, proxy, or server. Reuse may be impossible, but a valid ticket might allow resumption. If the request is a harmless fetch and the HTTP stack implements RFC 8470 correctly, early data may improve the visible wait. For a payment command, the same optimization is a poor default unless the application's idempotency contract is proven. The user has a stronger interest in one correct charge than in saving one network flight.

Consider instead a service inside one data center that makes hundreds of RPCs per second to the same peer. Its RTT may be under a millisecond, and the main issue may be TLS CPU, certificate verification, or connection churn rather than network flights. A larger, stable pool can amortize setup, but an oversized pool can create its own queues, memory pressure, and uneven load. The choice belongs to observed connection lifetimes and request concurrency. The [connection-pool deep dive](/blog/software-development/networking/connection-pools-and-where-your-tail-latency-actually-lives) owns that queueing analysis; the TLS contribution is the work paid when a socket becomes secure.

Consider a CDN or service mesh that terminates TLS twice. The user-facing connection may be warm while an upstream connection pool constantly churns during scale-out. The user's `time_appconnect` stays flat, but first-byte time rises because the proxy is handshaking upstream. Switching the browser-facing connection from TLS 1.2 to 1.3 cannot erase an origin flight on another leg. Instrument both legs and label their peers. A trace span called “TLS” without its local and remote endpoint is too vague to guide a change.

| Situation | Best first action | Why | New obligation |
| --- | --- | --- | --- |
| Many cold connections to the same peer | Measure and improve connection reuse | Avoids repeated TCP and TLS setup | Bound idle sockets and pool queues |
| Distant peer, unavoidable new connections | Enable a supported TLS 1.3 full handshake | Removes one full-handshake flight compared with full 1.2 | Verify actual negotiation and retry behavior |
| Repeat connections with usable tickets | Measure resumption hit rate | Reuses authenticated context and reduces work | Manage ticket lifetimes and key scope |
| Replay-safe returning requests | Evaluate early data with RFC 8470 | Moves eligible request bytes into first TLS flight | Prove replay safety, 425 handling, and fallback |
| Stable client TLS, slow first byte | Trace downstream proxy and origin | Client handshake is not the growing phase | Attribute hidden upstream connections |

None of these rows promises a context-free winner. An HTTP/3 connection uses QUIC and integrates TLS 1.3 differently, so its handshake timing should be measured under its own transport model. A client pinned to a proxy may see the proxy's TLS endpoint rather than the advertised service. A mobile path may have variable RTT and packet loss that dwarf an optimized cipher choice. Name the actual path, peer, and request class before making a performance claim.

## The security contract in one sentence per feature

A full certificate-authenticated TLS 1.3 handshake gives the client an authenticated server identity, an ephemeral key agreement, transcript integrity, and protected application traffic after the server flight. It takes a network flight after TCP under the clean-path model. A full TLS 1.2 handshake can provide the same broad goals, including forward secrecy when it negotiates an ephemeral suite, but its standard full exchange needs another sequential flight. Neither version makes an invalid certificate valid, prevents application bugs, or eliminates transport loss.

TLS 1.3 PSK resumption reuses a secret derived from an earlier successful connection. With fresh DHE, later one-RTT application traffic can retain forward secrecy under the normal erasure assumptions. With PSK-only key exchange, that property is weaker. A ticket therefore represents a continuing security dependency on previous state and the server's ticket handling. Resumption is not a general license to send application bytes before the server answers.

Zero-RTT early data is an optional, narrower feature. It encrypts a returning client's early bytes, but they can be replayed across connections and do not get the same forward-secrecy property as later traffic. The server can reject them. An HTTP application needs an explicit replay-safe profile and retry behavior. This is why the [current TLS 1.3 standard](https://www.rfc-editor.org/rfc/rfc9846) and [RFC 8470](https://www.rfc-editor.org/rfc/rfc8470) belong together in an early-data review. The protocol makes early data possible; HTTP and the application decide whether processing it is safe.

The practical senior-engineer rule is to keep **latency, authentication, and replay** as three separate columns in a design review. A change can improve one and weaken another. If a proposed optimization is described only as “faster TLS,” ask which message flight disappears, for which fraction of requests, and what happens when the shortcut is rejected or repeated. If nobody can answer those questions with a packet sequence and an application rule, measure before enabling it.

## Key takeaways

- On a new HTTPS-over-TCP connection, TLS begins after TCP connect and before the ordinary HTTP request. `time_appconnect - time_connect` is the client-observed TLS interval for curl's connection boundary.
- A standard full TLS 1.2 handshake costs roughly two sequential TLS RTTs; a normal full TLS 1.3 handshake costs roughly one. These are message-flight models, and resumption, HelloRetryRequest, loss, and local work change the observation.
- The certificate authenticates the server; ephemeral key agreement supplies fresh shared secret material. Forward secrecy depends on the negotiated exchange and secret erasure, not merely on a version label.
- TLS 1.3 resumption uses a remembered PSK. PSK-DHE and PSK-only resumption have different forward-secrecy properties, and a ticket can be rejected.
- Accepted 0-RTT lets a returning client send eligible application bytes with ClientHello. Early data can be replayed and can be rejected, so anti-replay state, HTTP 425 behavior, and application idempotency are part of the feature.
- A growing curl TLS delta calls for a handshake trace; a stable TLS delta with a growing first-byte interval calls for a downstream proxy or server trace. A timer is a location hint, not a complete root cause.

## Further reading

- [RFC 5246, TLS 1.2](https://www.rfc-editor.org/rfc/rfc5246), August 2008, for the full and abbreviated handshake flows.
- [RFC 8446, TLS 1.3](https://www.rfc-editor.org/rfc/rfc8446), August 2018, for the deployed 1-RTT and early-data design; [RFC 9846](https://www.rfc-editor.org/rfc/rfc9846) is the current successor.
- [RFC 8470, Using Early Data in HTTP](https://www.rfc-editor.org/rfc/rfc8470), September 2018, for `Early-Data` and `425 Too Early`.
- [Cloudflare's March 2017 0-RTT deployment account](https://blog.cloudflare.com/introducing-0-rtt/), read together with its later default-setting update.
- [Curl's `--write-out` field reference](https://curl.se/docs/manpage.html#-w), for the exact cumulative timing definitions.
