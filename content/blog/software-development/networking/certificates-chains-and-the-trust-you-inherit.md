---
title: "Certificates, Chains, and the Trust You Inherit"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Trace a TLS certificate failure from SNI and SAN through chain building, revocation, expiry, and the client trust store."
tags:
  [
    "networking",
    "distributed-systems",
    "tls",
    "x509",
    "certificate-chains",
    "public-key-infrastructure",
    "sni",
    "certificate-transparency",
    "incident-response",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 43
image: "/imgs/blogs/certificates-chains-and-the-trust-you-inherit-1.webp"
---

The service is healthy. DNS returns the right address. TCP connects. Then one client reports `certificate verify failed`, another works, and a third works only after you open the same URL in a browser. The server logs may show no HTTP request at all. This is a trust-path failure, and the useful question is not whether the certificate is “valid” in isolation. It is whether this particular client can construct an acceptable path from the certificate the server actually sent to a trust anchor this client actually has, for the name the client actually requested, at this moment.

![The TLS certificate gate on the request path](/imgs/blogs/certificates-chains-and-the-trust-you-inherit-1.webp)

The diagram above is the mental model: certificate validation sits inside the TLS portion of a cold request, after a connection has been established and before authenticated HTTP can proceed. An application timeout and a trust failure can both be reported as a failed request, but their packet and timing signatures differ. In the [series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url), we followed a URL down to packets. Here we stop at the boundary where a packet peer becomes an authenticated service identity. The neighboring [TLS handshake post](/blog/software-development/networking/tls-what-the-handshake-buys-and-what-it-costs) prices the round trips and key exchange. This post examines what the client accepts as proof of *who* owns the public key.

There are four independent questions to answer. Did the server select the right certificate? Does its subject alternative name cover the requested identity? Can the client build and validate a certificate path under its policy? Is the result still acceptable under time, revocation, and transparency rules? “The browser shows a padlock” answers those questions only for that browser, at that time, with its trust store, cache, path builder, and policy. It says little about an old embedded OpenSSL binary or a container image frozen years ago.

## 1. Put the failure in the right latency segment

**First separate transport reachability from identity verification.** A certificate failure does not mean the remote host was unreachable. The client must normally exchange a ClientHello and receive enough of the server's handshake to examine its certificate. `curl` makes that distinction visible:

```bash
curl -v --connect-timeout 3 --max-time 8 \
  -w '\nremote_ip=%{remote_ip} connect=%{time_connect} appconnect=%{time_appconnect} starttransfer=%{time_starttransfer} http=%{http_code}\n' \
  https://example.com/ -o /dev/null
```

`time_connect` is elapsed time until the TCP connection finishes. `time_appconnect` is elapsed time until the SSL or TLS handshake is complete. A verification error can prevent `time_appconnect` from becoming a useful completed-handshake timestamp. Read the verbose certificate error and the exit status alongside the timing fields. A successful TCP connect followed by a verification error places the fault in authentication, even if a generic application metric labels the whole request a connect failure. `curl`'s timing definitions are documented in its [write-out variables reference](https://curl.se/docs/manpage.html#-w).

The certificate chain may also increase the bytes in the handshake. A larger chain can require more records and more packets; on a constrained path, that can matter to latency or loss recovery. Do not convert chain length into a fixed millisecond tax. TLS version, MTU, record packing, congestion window, packet loss, and whether a connection is resumed all matter. A warm connection can skip the entire certificate exchange for that request. Measure the cold handshake when diagnosing chain behavior and keep connection reuse explicit. The [latency-budget post](/blog/software-development/networking/the-latency-budget-speed-of-light-serialization-and-queueing) supplies the serialization and propagation vocabulary.

One deliberately simple model is useful for investigation: **cold-request latency ≈ DNS + TCP setup + TLS setup and verification + server time + response transfer**. This is an explanatory accounting model, not a protocol equation; several phases can overlap or be skipped. If TCP finished and TLS failed, optimizing server think time cannot fix the failure. If `curl -k` appears to “fix” it, you have only proved that bypassing authentication changes the outcome. That is a diagnostic experiment, never a deployment remedy.

## 2. A certificate is a signed statement, not a magic badge

**The public key in a leaf certificate is useful only when the client accepts the chain of authority and the name binding.** X.509 is a structured signed object. The certificate has a subject, issuer, public key, validity interval, serial number, extensions, and a signature by the issuer. For a TLS server, the end-entity or *leaf* certificate states that a key is authorized for identities listed in its Subject Alternative Name (SAN) extension, under constraints inherited from the issuing certification authorities. The client checks the server has the corresponding private key during the TLS handshake; merely sending a certificate would not prove possession.

![The leaf, intermediate, and trust anchor stack](/imgs/blogs/certificates-chains-and-the-trust-you-inherit-2.webp)

Start at the leaf and walk toward an anchor. The leaf is usually signed by an intermediate CA. That intermediate may be signed by another intermediate or a root. A **trust anchor** is an input to path validation, chosen by the client or administrator. It is often distributed as a self-signed root certificate in a trust store, but its trusted public key and name are the important inputs. The server does not make a root trusted by sending it over the network. [RFC 5280, Section 6.1](https://www.rfc-editor.org/rfc/rfc5280.html#section-6.1) defines the basic path validation model and explicitly treats obtaining the candidate sequence of certificates as separate from validating it. That distinction explains a large class of operational failures: the certificate may be good, but the client lacks the material needed to construct its path.

The signature checks form a chain of delegated authority. Suppose a leaf says issuer `Lab Intermediate` and its signature verifies with the intermediate's public key. That proves the intermediate signed it; it does not prove the intermediate had authority to sign server certificates. The intermediate must itself validate under a trusted path, with `basicConstraints` permitting CA operation and `keyUsage` permitting certificate signing where those extensions apply. The leaf must be valid for TLS server use under the relevant extended key usage and application policy. Time validity, name constraints, policy constraints, path length constraints, and algorithm restrictions can still reject the chain. A single green signature check is therefore insufficient.

The path builder may have choices. Cross-signing can create more than one certificate with the same subject and public key but different issuers. An implementation can pick different routes through a graph of candidate issuers, and a different trust store can terminate at a different anchor. The identity proof is a conjunction of several predicates, which we can write as an **explanatory model**, not as a verbatim RFC formula:

$$
\text{accept} = \text{name match} \land \text{valid path to trusted anchor} \land \text{time and usage policy} \land \text{other client policy}.
$$

“Other client policy” can include revocation information, Certificate Transparency (CT), or local restrictions. A browser, Java runtime, containerized Go service, and firmware image need not make identical decisions. [RFC 9525, published November 2023](https://www.rfc-editor.org/rfc/rfc9525.html), discusses service identity checking separately from PKIX path validation, and this separation is the diagnostic key: a trustworthy issuer does not make a certificate valid for the wrong hostname.

You can inspect a local certificate without contacting a network service:

```bash
openssl x509 -in leaf.pem -noout -subject -issuer -dates -serial \
  -ext subjectAltName -ext basicConstraints -ext keyUsage -ext extendedKeyUsage
openssl x509 -in intermediate.pem -noout -subject -issuer -dates \
  -ext basicConstraints -ext keyUsage
openssl verify -show_chain -CAfile root.pem -untrusted intermediate.pem leaf.pem
```

The first two commands reveal what the files claim. The last asks OpenSSL to build and validate a path using a chosen anchor and an untrusted intermediate. `-show_chain` tells you what path *this invocation* selected. Use the target runtime's verifier too: an OpenSSL result is evidence about OpenSSL with these flags and store, not a universal verdict about every client.

### The fields that change a diagnosis

`subject` and `issuer` give human-readable names, but names alone do not link certificates. Signature verification and key identifiers do. `notBefore` and `notAfter` bound the certificate's validity in time; a correct server clock is not enough if the client clock is wrong. SAN carries the DNS or IP identities the application checks. `basicConstraints: CA:TRUE` identifies a CA-capable intermediate, and `keyUsage: Certificate Sign` allows the key to sign certificates when the extension is present. A leaf typically has `CA:FALSE` and a server-authentication extended key usage. The exact extension rules are in [RFC 5280, Sections 4.2 and 6](https://www.rfc-editor.org/rfc/rfc5280.html).

Do not treat a certificate's `CN` as a rescue path when SAN is wrong or absent. Modern service identity rules center the SAN identities. [RFC 9525, Section 6.2.1](https://www.rfc-editor.org/rfc/rfc9525.html#section-6.2.1) says the common name is not a valid source of server identity. Some legacy stacks retain compatibility behavior, which is precisely why a test must use the actual client library and configuration. The service's security contract should specify the reference name the client is trying to authenticate, not depend on accidental fallback behavior.

## 3. Serve the intermediates the client needs

**A server should provide the leaf and appropriate intermediates, while the client supplies the trusted root.** The client cannot validate a signature by an issuer whose certificate it has never seen, unless it already cached that intermediate or obtains it by some other mechanism. This is why a missing intermediate can be invisible in one browser and fatal in a fresh container. A browser may have an issuer cached from another visit, while an API client starts with only its packaged trust store. Some clients fetch issuer certificates from the leaf's Authority Information Access (AIA) URL; others do not, or cannot due to network policy. Neither cache nor AIA fetching is a deployment contract you should rely on.

![Missing intermediate compared with a complete served chain](/imgs/blogs/certificates-chains-and-the-trust-you-inherit-3.webp)

The operational file is often called `fullchain.pem`: leaf first, then the intermediate or intermediates required to reach a common root. The name is a packaging convention, not a guarantee that every possible client will accept it. The trust anchor is commonly omitted from the server's handshake because sending a root cannot grant trust and costs bytes. If you operate several ingress layers, verify the *served* chain at each public endpoint, not merely the contents of an ACME output directory. A load balancer can keep an old bundle even after the host's files rotate.

Use `s_client` with SNI and a verification store you control:

```bash
openssl s_client -connect api.example.com:443 \
  -servername api.example.com \
  -verify_hostname api.example.com \
  -verify_return_error \
  -showcerts </dev/null
```

Inspect `Certificate chain`, every PEM block, `Verify return code`, and the reported negotiated protocol. The bytes shown under `-showcerts` are what that endpoint served; they are not automatically the path selected by a particular verifier. Run from a network position that reaches the actual edge or ingress. If DNS load balances across addresses, probe each address while preserving the desired SNI and reference hostname. `curl --resolve api.example.com:443:203.0.113.10 https://api.example.com/` is a useful shape for pinning a particular address in a test; replace the documentation address with your real target. The TLS name remains `api.example.com`.

An `unable to get local issuer certificate` result can mean a missing served intermediate, an absent trust anchor, or a path-building incompatibility. Do not prescribe “install the root” from that string alone. First compare the served chain, the intended trust store, and the candidate path. If the intermediate is missing, install the correct full chain on every TLS terminator. If the client has no acceptable anchor, decide whether to update its trust store, support a different issuing path, or stop supporting that client. Importing a leaf or arbitrary intermediate as a trust anchor broadens trust in ways operators often fail to notice.

Here is a rough byte calculation, labeled as a **derived illustration**. If an additional DER-encoded certificate contributes 1,200 bytes, then the TLS handshake carries approximately 1,200 additional certificate bytes, before TLS and record overhead. On an otherwise idle 10 Mbit/s bottleneck, serialization alone is $1{,}200 \times 8 / 10{,}000{,}000 = 0.00096$ seconds, or $0.96$ ms. This is not a measured handshake delta: loss, packet boundaries, delayed processing, and congestion may dominate. The purpose of the arithmetic is to keep “long chain” in the right category. It costs bytes and can amplify loss sensitivity, but it does not add a fixed round trip solely because another certificate exists.

## 4. SNI chooses a certificate; SAN proves a name

**Selection at the server and verification at the client are separate operations.** Server Name Indication (SNI) is a TLS ClientHello extension carrying the hostname the client wants. It lets a server sharing one address select a certificate and configuration for that name. [RFC 6066, Section 3](https://www.rfc-editor.org/rfc/rfc6066.html#section-3) specifies the extension. The server may select a default certificate when SNI is absent or unrecognized. The client then checks the identity it intended to reach against the selected certificate's SAN entries. The HTTP `Host` header comes later and cannot repair a TLS certificate mismatch that has already aborted the handshake.

![SNI selection and SAN verification are separate decisions](/imgs/blogs/certificates-chains-and-the-trust-you-inherit-4.webp)

Think of SNI as choosing which ID card the server presents and SAN verification as checking whether that card names the destination the client requested. A server can select the wrong card because of a missing virtual-host entry, a stale edge deployment, or a client that omits SNI. The selected card can still have a perfectly valid path to a public root. It will fail because the name is wrong. Conversely, a certificate can list the right SAN but fail because its issuer path is incomplete. These errors have different fixes.

Make the name test explicit:

```bash
openssl s_client -connect 10.77.0.2:8443 \
  -servername api.lab.example \
  -verify_hostname api.lab.example \
  -CAfile root.pem -verify_return_error </dev/null
openssl s_client -connect 10.77.0.2:8443 \
  -servername other.lab.example \
  -verify_hostname other.lab.example \
  -CAfile root.pem -verify_return_error </dev/null
```

In a multi-certificate server, the first line should select and validate the `api.lab.example` identity, while the second tests the other name. In a single-certificate lab, the second may fail for a SAN mismatch even though the TCP path and CA path are identical. Compare the served leaf's SAN as well as `Verify return code`. `-servername` controls SNI; `-verify_hostname` controls the reference identity used for the check. A command that sets only one can prove the wrong thing. For an IP literal, use an IP address SAN and the client's IP verification mode; a DNS SAN containing the textual address is not equivalent.

Wildcard names are also narrower than casual config reviews assume. A certificate for `*.example.com` generally matches one label such as `api.example.com`, not an arbitrary depth such as `v1.api.example.com`. [RFC 9525's matching rules](https://www.rfc-editor.org/rfc/rfc9525.html#section-6.3) constrain wildcard matching. Public CA issuance policy and client libraries can add further restrictions. Test every externally used hostname, including preview, regional, and failover names, through the same path your client will use.

SNI itself is not proof of identity. It is client-provided routing metadata. An attacker can claim any SNI in a ClientHello, and the server should not treat that string as an authenticated principal. This distinction matters when a reverse proxy maps SNI to a tenant, then later trusts a different HTTP authority or route. The TLS layer authenticates the server to the client after certificate verification; authorization of an HTTP request is a separate application concern.

## 5. Expiry monitoring must probe the served endpoint

**An alert on a certificate file is not an alert on the certificate customers receive.** A certificate can renew successfully in the ACME client's directory while a reverse proxy, CDN, or ingress controller continues presenting the old leaf. A process may need a reload, a secret may fail to propagate, or one region may lag another. Checking the file's `notAfter` on one node answers only whether that file is near expiry. It does not prove the right leaf, chain, SNI mapping, and listener are live on every client-visible path.

The animated timeline below follows the operational transition from issued certificate to deployed certificate to a warning threshold and the actual expiry boundary. Its point is the gap between *renewal succeeded* and *the endpoint changed*. That gap is where an expiry alert must observe.

<figure class="blog-anim">
<svg viewBox="0 0 880 260" role="img" aria-label="Certificate lifecycle from issuance through deployment, endpoint expiry alert, and expiration" style="width:100%;height:auto;max-width:880px">
<style>
.c5-card{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.c5-text{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.c5-note{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.c5-line{stroke:var(--text-secondary,#6b7280);stroke-width:3}
.c5-dot{fill:var(--accent,#6366f1)}
.c5-ring{fill:none;stroke:var(--accent,#6366f1);stroke-width:4}
@keyframes c5-walk{0%,15%{transform:translateX(0)}25%,40%{transform:translateX(210px)}50%,65%{transform:translateX(420px)}75%,90%{transform:translateX(630px)}100%{transform:translateX(0)}}
.c5-move{animation:c5-walk 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.c5-move{animation:none}}
</style>
<line class="c5-line" x1="85" y1="82" x2="795" y2="82"/>
<circle class="c5-dot c5-move" cx="85" cy="82" r="10"/>
<rect class="c5-card" x="20" y="112" width="130" height="94" rx="12"/>
<rect class="c5-card" x="230" y="112" width="130" height="94" rx="12"/>
<rect class="c5-card" x="440" y="112" width="130" height="94" rx="12"/>
<rect class="c5-card" x="650" y="112" width="130" height="94" rx="12"/>
<rect class="c5-ring c5-move" x="20" y="112" width="130" height="94" rx="12"/>
<text class="c5-text" x="85" y="150">Issue</text>
<text class="c5-note" x="85" y="179">New certificate</text>
<text class="c5-text" x="295" y="150">Deploy</text>
<text class="c5-note" x="295" y="179">Serve on endpoint</text>
<text class="c5-text" x="505" y="150">Alert</text>
<text class="c5-note" x="505" y="179">Probe served cert</text>
<text class="c5-text" x="715" y="150">Expire</text>
<text class="c5-note" x="715" y="179">TLS fails if stale</text>
<text class="c5-note" x="440" y="242">A file renewed on disk is not proof that the endpoint serves it.</text>
</svg>
<figcaption>Monitoring must inspect the certificate actually served by the endpoint before expiry.</figcaption>
</figure>

A useful monitor records at least hostname, address or edge, certificate fingerprint or serial, `notAfter`, chain summary, validation result, and observation time. It probes with the intended SNI and verifies against the intended client trust store. Probe the public edge as well as any internal TLS terminator with a separate identity. When one hostname resolves to multiple edge addresses, sample the distinct addresses or run probes from multiple regions; otherwise a single healthy endpoint can hide a stale member. The owner should know which deployment artifact, certificate issuer, and reload path corresponds to each observed serial.

For a first manual inspection:

```bash
host=api.example.com
openssl s_client -connect "${host}:443" -servername "$host" </dev/null 2>/dev/null |
  openssl x509 -noout -subject -issuer -serial -fingerprint -enddate
```

This pipeline deliberately examines the served leaf. It does *not* perform full verification, because the first `s_client` command omits `-verify_return_error` and a controlled CA file. Use the explicit verification command from the preceding section for that separate check. `openssl x509 -checkend 604800` exits nonzero when the certificate expires within the next 604,800 seconds, which is exactly seven days. It is a useful local gate, but a production monitor should include alert delivery, ownership, and a second warning horizon that leaves enough time for issuance and deployment remediation.

We can derive the monitoring margin without pretending one threshold fits everyone. Let $R$ be the maximum time from renewal attempt to confirmed deployment, $D$ the longest alert detection and paging delay, and $H$ the human remediation allowance. As an **operational planning model**, alert before expiry by more than $R+D+H$. If a team budgets $R=24$ hours, $D=1$ hour, and $H=48$ hours, its minimum margin is $73$ hours; a seven-day warning offers slack for weekends or unexpected issuer problems. These are illustrative budgets, not measured facts about a vendor. If you cannot state your own $R$, instrument issuance and deployment separately before arguing about the exact alert threshold.

An alert should fire on *failure to observe the new served serial* after renewal, not only on proximity to expiry. That detects a broken reload before the expiry countdown becomes urgent. A second alert should fire when the served certificate enters the warning window, regardless of whether the renewal job reports success. A third can warn when validation fails for the intended hostname and trust profile. These three signals distinguish failed issuance, failed deployment, and client-path incompatibility. A dashboard that collapses all three into “TLS certificate unhealthy” slows response.

Watch clock behavior too. X.509 validity uses absolute times. A client whose clock is behind `notBefore` or ahead of `notAfter` can reject an otherwise sound certificate. That is not a reason to disable verification. It is a reason to verify time synchronization on the failing client and compare the error to the certificate's validity interval. In constrained devices with unreliable clocks, boot-time trust decisions need an explicit design, because “wait for NTP” may itself depend on TLS to reach a time service.

### Make the alert actionable

The alert payload should contain the exact hostname and port, SNI value, observed peer address, certificate serial, earliest expiring certificate in the served chain, and the validation error from the probe. Include the previous good observation and the source of the expected certificate. A page saying only “certificate expires in seven days” can send the engineer to the wrong ingress or to a renewed file that is not deployed. If your deployment intentionally serves different chains to different client classes, record the client profile and selected chain in the alert.

Treat certificate rotation as a deployment with rollback semantics. The new leaf should arrive with its matching private key and intermediates. Atomic replacement of the bundle avoids a transient state where leaf and key do not match. Reload should be tested for the actual listener. A successful configuration parse is not proof that the new chain is being served; follow it with a network probe. The expected post-deploy signal is a changed serial or fingerprint at every target endpoint and a successful verification from representative clients.

## 6. Revocation is a policy and availability trade-off

**A certificate can be within its validity window and still need to be distrusted.** The obvious cases are compromised private keys, mistaken issuance, or loss of control over a name. CAs publish revocation information, but the way a client obtains and enforces that information varies. This is one reason “valid until next month” is weaker than it sounds. Validity time is only one condition in acceptance.

![Revocation mechanisms and their operational costs](/imgs/blogs/certificates-chains-and-the-trust-you-inherit-6.webp)

A Certificate Revocation List (CRL) is a signed list of revoked certificate serials for an issuer. The client obtains a list, checks its signature and freshness, then checks the target serial. CRLs can be cached or distributed efficiently, but the list may grow and freshness depends on publishing and fetching schedules. [RFC 5280, Section 6.3](https://www.rfc-editor.org/rfc/rfc5280.html#section-6.3) describes CRL processing. Online Certificate Status Protocol (OCSP) asks a responder about a particular certificate's status, typically returning `good`, `revoked`, or `unknown`. [RFC 6960](https://www.rfc-editor.org/rfc/rfc6960.html) defines the protocol. `good` means the responder has not marked that serial revoked under its status model; it is not a fresh proof that the domain operator still controls the key.

An OCSP lookup can expose the sites a client visits to the responder and can insert another network dependency into connection setup. **OCSP stapling** moves the response fetch to the server: the server presents a CA-signed response in the handshake, and the client checks its signature and freshness. This reduces per-client responder lookups, but the server must refresh the staple. A stale or missing staple can become an outage if the certificate and client policy require it. The TLS status request extension is specified in [RFC 6066, Section 8](https://www.rfc-editor.org/rfc/rfc6066.html#section-8); [RFC 7633](https://www.rfc-editor.org/rfc/rfc7633.html) defines a TLS feature extension used for the “must-staple” requirement. Enforcement and support are client-specific. Do not assume a stapled response is checked merely because the server sends one.

The decision that often dominates real incidents is *soft fail versus hard fail*. If a revocation service is unreachable, a soft-fail client may continue, preserving availability while losing assurance about revocation. A hard-fail client rejects the certificate, preserving the revocation rule while making responder reachability part of the service's availability path. There is no context-free winner. Browser revocation strategies often involve browser-specific mechanisms beyond straightforward live OCSP fetches, so test the concrete user agent and its policy rather than projecting a generic RFC diagram onto every browser.

Short-lived certificates reduce the maximum time an unrevoked compromised key can remain useful, but they do not make key compromise harmless. Their shorter life moves risk into automation: issuance, distribution, reload, and endpoint verification must be reliable. A 24-hour certificate whose renewal pipeline fails is a more immediate availability problem than a 90-day certificate whose pipeline fails. These are example durations for reasoning, not claims that either duration is a universal standard. The companion [mTLS and service identity post](/blog/software-development/networking/mtls-and-service-identity-at-scale) applies this trade-off to workload identities, where short lifetimes and automated rotation are especially important.

Revocation mechanisms also change over time. In a [December 5, 2024 announcement](https://letsencrypt.org/2024/12/05/ending-ocsp), Let's Encrypt said it would add CRL URLs to certificates by May 7, 2025, stop including OCSP URLs on that date, and turn off its OCSP responders on August 6, 2025. That is issuer-specific and dated. It makes an old “always monitor the OCSP staple for every Let's Encrypt certificate” runbook stale. Check the actual certificate's extensions and issuer policy as deployed, then decide what your clients enforce. A certificate without an OCSP URL cannot produce the same operational behavior as one issued under an older policy.

| Mechanism | What the client obtains | Main operational constraint | Failure question | Source |
| --- | --- | --- | --- | --- |
| CRL | Signed issuer list containing revoked serials | Distribution and freshness of the list | Does the client have a current list for this issuer? | [RFC 5280 §6.3](https://www.rfc-editor.org/rfc/rfc5280.html#section-6.3) |
| OCSP | Signed status response for a certificate | Responder reachability, privacy, response age | Does policy reject an unavailable or stale response? | [RFC 6960](https://www.rfc-editor.org/rfc/rfc6960.html) |
| OCSP stapling | Server-delivered signed OCSP response | Server refresh and client enforcement | Is the staple present, current, and required? | [RFC 6066 §8](https://www.rfc-editor.org/rfc/rfc6066.html#section-8), [RFC 7633](https://www.rfc-editor.org/rfc/rfc7633.html) |
| Short lifetime | A bounded validity interval rather than a revocation query | Reliable renewal and deployment | Can every endpoint rotate before expiry? | Operational inference from validity checking in [RFC 5280 §6.1](https://www.rfc-editor.org/rfc/rfc5280.html#section-6.1) |

The table is a mechanism comparison, not a claim that every client uses every row. For a real endpoint, examine the AIA and CRL Distribution Points extensions and the target client's revocation policy. `openssl x509 -in leaf.pem -noout -text` displays the advertised locations. Advertised locations alone do not prove the client fetched them or enforced a result. A packet capture or client debug log can settle whether a particular program made a network request.

## 7. Certificate Transparency detects issuance you did not authorize

**CT makes publicly trusted certificate issuance observable; it does not validate a server's runtime configuration for you.** A CA can issue a certificate for a domain after its domain-control process. If that process or the CA is compromised, the domain owner needs a way to discover the unexpected certificate. CT logs are append-only public logs of certificates or precertificates. A log returns a Signed Certificate Timestamp (SCT), which is a promise to include an entry within the log's maximum merge delay. Monitors can watch for names they own and investigate unfamiliar issuances. The [CT project's explanation](https://certificate.transparency.dev/howctworks/) describes the precertificate, SCT, inclusion, and Merkle-tree audit model; [RFC 9162](https://www.rfc-editor.org/rfc/rfc9162.html) specifies CT version 2.0.

Do not collapse three different checks into “CT says it is safe.” First, an SCT says a log promised inclusion; it is not, by itself, proof that every monitor saw the entry at handshake time. Second, a logged certificate can still be maliciously or mistakenly issued. Logging lets owners detect and respond. Third, a perfectly logged certificate can be served with a missing intermediate or for the wrong SNI. CT is an issuance visibility control, not a replacement for path validation, hostname verification, or endpoint probes.

There is a useful operational distinction between *expected issuance* and *unexpected issuance*. A normal renewal often produces a new serial and SCTs even when nothing suspicious happened. Your CT monitor needs an allowlist or reconciliation path tied to issuance automation, but it must not auto-dismiss every certificate from a familiar CA. A compromised ACME credential could produce a valid certificate at that same CA. Review the SAN set, key lineage, requested account or order where available, and whether the certificate reached your deployment inventory. Log search is a detective control whose value depends on routing an unexpected result to an owner who can act.

CT can also reveal names you thought were private if a public CA issued a certificate for them. That is a design implication, not a flaw in CT: public log visibility is part of the transparency mechanism. Do not put sensitive internal naming in public certificates without accepting that disclosure. For private PKI, the trust and logging model is different. The [service-to-service security post](/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust) owns application-side identity policy; this article stays at the certificate and wire boundary.

The immediate diagnostic command remains the served-chain probe. CT is where you ask, “Did someone issue a certificate for this name that we did not expect?” `s_client` is where you ask, “Which certificate did this endpoint present to this client?” Those are related investigations with different evidence. When a browser warns about CT policy, collect the exact browser build, certificate, SCT delivery route, and policy; an OpenSSL `verify` pass will not reproduce browser-specific CT enforcement.

## 8. The DST Root CA X3 expiry: one chain, different client outcomes

**The expiry was not a universal failure of every Let's Encrypt certificate.** On September 30, 2021, IdenTrust's DST Root CA X3 expired. Let's Encrypt had used that root to establish broad trust before its own ISRG Root X1 was present in many client stores. By 2021, modern clients generally trusted ISRG Root X1 directly. Older devices that had not received an updated trust store could depend on the DST path. That mixed client population created a path-building problem, not a single binary answer to “does the site use Let's Encrypt?”

![Alternative trust paths after DST Root CA X3 expiry](/imgs/blogs/certificates-chains-and-the-trust-you-inherit-7.webp)

The source ledger for this case is concrete. **Event date:** September 30, 2021, the DST Root CA X3 expiration, as stated by [Let's Encrypt's expiry notice](https://letsencrypt.org/ca/docs/dst-root-ca-x3-expiration-september-2021/). **Source owner:** Let's Encrypt, the issuing CA and chain operator. **Mechanism:** the server's default chain included a cross-signed ISRG Root X1 certificate issued by DST Root CA X3, while an alternate shorter chain ended at an ISRG Root X1 trust anchor. **Verified compatibility facts:** Let's Encrypt reported that old Android devices lacking ISRG Root X1 could continue using the special cross-sign because Android did not enforce the expiry of the DST certificate when used as a trust anchor. It also warned that OpenSSL 1.0.x clients could reject the Android-compatible long chain even when ISRG Root X1 was in their trust store. **Transfer lesson:** test the chain actually served against representative client trust stores and path builders, not just a modern desktop browser.

The default RSA chain described in [Let's Encrypt's 2021 production-chain announcement](https://community.letsencrypt.org/t/production-chain-changes/150739) was leaf signed by R3, R3 signed by ISRG Root X1, and a cross-signed ISRG Root X1 certificate issued by DST Root CA X3. The alternative omitted that final cross-sign. The self-signed DST root itself was a trust anchor in client stores, not a certificate that a server needed to send. The same public key for ISRG Root X1 could participate in different paths. A modern path builder with ISRG Root X1 trusted could stop there. An older Android device that lacked ISRG Root X1 could use the DST trust anchor and the cross-sign. The details of trust-anchor treatment matter: [Let's Encrypt's December 2020 explanation](https://letsencrypt.org/2020/12/21/extending-android-compatibility.html) says Android intentionally did not enforce expiration dates for certificates used as trust anchors. That was the basis of the extension, not a claim that all expired certificates may be ignored.

Now consider the opposite client. [Let's Encrypt's January 2021 compatibility announcement](https://community.letsencrypt.org/t/openssl-client-compatibility-changes-for-let-s-encrypt-certificates/143816) says OpenSSL versions before 1.1.0 could reject the longer default chain because of path-verification behavior. Its proposed workaround for operators who did not need older Android compatibility was the shorter alternate chain. The same announcement explicitly warns that choosing the short chain would make Android older than 7.1.1 reject it if those devices lacked ISRG Root X1. It also says ISRG Root X1 had to be present in the trust store for the affected OpenSSL clients to use either chain. This is the compatibility frontier: a server-side chain choice could improve one old client group and harm another. “Switch to the shorter chain” was a conditional mitigation, not a universal fix.

The blast radius followed client diversity. Typical up-to-date browsers were not the main concern, according to Let's Encrypt's expiry notice. Embedded software, old operating systems, and API clients with bundled TLS libraries or trust stores deserved closer inspection. The exact affected set could not be inferred merely from a device brand or OS label: the TLS implementation, trust-store update state, and served chain all mattered. [Let's Encrypt's compatibility post](https://community.letsencrypt.org/t/openssl-client-compatibility-changes-for-let-s-encrypt-certificates/143816) also mentioned similar issues for LibreSSL before 3.2.0 and GnuTLS before 3.6.14, while stating that the platform verifiers on Windows and macOS did not have that specific OpenSSL path-building problem. These are source-attributed historical version boundaries, not a recommendation to use those old libraries today.

Separate trigger, contributors, and recovery. The trigger was the scheduled expiry of DST Root CA X3. The contributing condition was a transition period with multiple roots and cross-signs, plus clients that differed in their anchor stores and path-building behavior. The blast-radius multiplier was an API or device population that could not quickly update its root store or TLS library. Recovery for an operator might mean upgrading clients and trust stores, or selecting the alternate chain after checking the supported client population. Replacing a server's leaf certificate alone did not necessarily alter the failing path. Disabling verification would remove the failure by removing the security property the service needed.

The incident is a useful warning against an oversimplified expiry monitor. A monitor that checked only the leaf's `notAfter` would report green. A monitor that used only a modern browser's verifier would report green. A monitor that scanned the chain and declared “an expired root appears, so all clients are broken” would also be wrong: trust-anchor treatment and path choice differ. A useful compatibility test matrix names the actual client library version, trust-store snapshot, served chain variant, SNI, reference hostname, and observation date. If your service contract includes old devices, keep a synthetic client or a reproducible container for each supported verifier family. The [RFC 5280 trust-anchor model](https://www.rfc-editor.org/rfc/rfc5280.html#section-6.1.1) explains why the trusted starting point is an input, while the 2021 case shows how implementation choices can still change the result.

The historical window has moved. [Let's Encrypt's July 10, 2023 write-up](https://letsencrypt.org/2023/07/10/cross-sign-expiration.html) explained that the special cross-sign was a time-limited compatibility measure and that its expiry would require another transition. Do not copy a 2021 chain-selection runbook into a 2026 deployment. The case is about *how to reason* when anchors and chain alternatives change: inventory client trust, inspect the current issuer's chain options, test the actual path, then plan the transition before the anchor or cross-sign ages out.

## 9. A diagnostic sequence that avoids false fixes

When a ticket says “TLS is broken,” I want one failed client, one target endpoint, and one timestamp before changing the server. Record the URL or expected DNS name, resolved address, SNI sent, client program and TLS library version, trust store location or version, and the full error string. If a proxy is involved, record whether TLS ends at the edge, sidecar, or application. Then reproduce with the client's own runtime. The [path map from the introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) is still useful: a certificate at the edge and a certificate on an internal mesh hop are different endpoints with different trust contracts.

The first discriminating probe tests selection and presentation, not a guessed root-store edit:

```bash
target=api.example.com
openssl s_client -connect "${target}:443" -servername "$target" \
  -showcerts -verify_hostname "$target" -verify_return_error </dev/null
```

If the wrong leaf appears, inspect SNI mapping, edge routing, and deployment state. If the leaf's SAN does not cover the reference hostname, issue or select a certificate that does. If a middle certificate is absent, fix the served chain. If the full chain is present but only one client class fails, compare that client's anchors, algorithm policy, time, and path builder. A client may reject an algorithm or key size even when the issuer path exists. In a production postmortem, attach the served PEM chain or fingerprints and the command line, not merely a screenshot of a browser warning.

| Symptom | Evidence to collect | Likely boundary | First safe action |
| --- | --- | --- | --- |
| TCP connects, wrong certificate shown | Leaf SAN, SNI, peer IP, ingress mapping | Server certificate selection | Probe each ingress with explicit SNI |
| Right SAN, `unable to get local issuer` | Served intermediates, trust store, selected path | Chain construction or anchor | Compare fullchain and client store |
| Valid path in one client, expiry error in another | Client clock, chain dates, verifier version | Time or path policy | Check clock and selected anchor |
| Leaf near expiry despite renewal success | Served serial versus renewed serial | Deployment or reload | Trace bundle to listener and edge |
| Revocation status error | AIA or CRL extension, client policy, response freshness | Revocation dependency | Inspect status source and client policy |
| Browser-specific CT warning | SCT delivery and browser policy | Transparency enforcement | Inspect certificate SCTs and browser details |

The table intentionally gives safe first actions, not one-line repairs. `curl -k` or an equivalent `InsecureSkipVerify` flag suppresses the identity check and can make a request appear successful, but it turns an authentication failure into exposure to an active intermediary. Use it only as a controlled local diagnostic to show that verification is the failing stage, then remove it. Adding a broad root to the system store can be just as consequential. A root is a grant of signing authority to that client; scope private PKI anchors to the intended service and process when possible.

Be precise about what a packet capture can and cannot show. It can show the TCP connection, TLS handshake progression, alerts, and, depending on TLS version and key material, some certificate details. TLS 1.3 encrypts more handshake messages after ServerHello, so a passive capture alone may not expose the certificate. Client or server TLS debug output is often more direct for certificate inspection. Capture filters should include only the target host and port, with limited duration, and be handled as sensitive data: captures can contain credentials, tokens, personal data, and application payloads. The [TLS handshake post](/blog/software-development/networking/tls-what-the-handshake-buys-and-what-it-costs) explains the encrypted handshake boundary.

When the failure is intermittent, stratify by endpoint and client type before averaging. A global success rate can hide one stale edge, one old trust store, or one SNI route. Compare fingerprints, not just `notAfter`, because two certificate variants can share a validity window. If traffic is split by DNS or anycast, the failing client may reach a different address than your shell. `curl --resolve` pins an address while retaining the hostname for SNI and verification. Repeat from the client network or a comparable vantage point and record the resolved address. A DNS query from your laptop does not prove what the failing device resolved minutes earlier.

## 10. Decide what to change at the trust boundary

There are several levers, each with a different failure mode. You can change the certificate and served intermediates, the client's trust store, the client's verifier implementation, or the fleet's supported-client policy. Make the least broad change that restores a correct identity check. A missing intermediate is normally a server packaging defect, so fix the served chain. A stale root store on a managed client is normally a client update problem. A path-builder bug in an unsupported library calls for an upgrade; a short alternate chain might be a temporary compatibility measure if its effect on other clients is tested. A name mismatch calls for correcting SNI, routing, or SAN, not importing more roots.

This is also a dependency question. Publicly trusted TLS inherits the CA ecosystem, root programs, CT policy, revocation distribution, and client software updates. Private PKI shifts much of that control to your organization, along with the duty to distribute anchors, rotate intermediates, and prevent accidental trust expansion. Neither removes the need to know which private key terminates which connection. A CDN or service mesh can multiply the number of termination points. The [edge and mesh termination post](/blog/software-development/networking/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end) treats where encryption ends; the certificate inventory must follow those boundaries.

The practical decision table is short. If the server presents the wrong leaf, fix selection or deployment. If it presents the right leaf without the needed intermediate, serve the complete chain. If a client has no trusted path, update its trust store or select an issuer and chain that meet the supported-client contract. If only a legacy verifier fails on a longer cross-sign path, upgrade it and test an alternate chain only with the full client matrix. If revocation or CT policy fails, inspect the actual policy and evidence before changing it. Every repair should end with the exact failing client successfully verifying the real endpoint with verification fully enabled.

### Inventory the clients before choosing a chain

A trust-store inventory sounds administrative until a root transition makes it the only useful map. The unit of inventory is not “Linux” or “Android.” It is an application build with a TLS implementation and a source of anchors. A process may use the operating system store, a runtime-maintained store, a file baked into a container image, or a bundle explicitly passed in configuration. Two applications on one host can therefore make different decisions about the same server. Even two replicas of one application can diverge when one image is stale. The inventory should capture image digest or package build, TLS library and version, trust-bundle origin and update date, and the expected server names. Without those fields, a compatibility test may accidentally exercise only the healthiest client.

There are two questions to ask of every profile. First, which anchor can terminate the selected certificate path? Second, which verifier behavior and policy determine whether it actually chooses and accepts that path? The DST case showed why both matter. A root can be present yet a legacy path builder can still fail on a longer supplied chain. Conversely, a browser with a fresh root can succeed even while a device with an old embedded store fails. If the service supports those devices contractually, put them in the deployment gate. If it does not, document the minimum supported trust-store and TLS-library versions so an incident responder can distinguish an unsupported device from a new regression.

Test from the client side with the deployed endpoint. For a containerized workload, execute the probe in the same image and with the same CA bundle path as the application, ideally using the same library. Merely mounting the host's `/etc/ssl/certs` into a diagnostic container can hide the defect you need to reproduce. For a mobile app, check whether it uses platform trust or a bundled store and whether a shipped app version pins a key or certificate. Pinning changes the trust contract: a public CA path can verify while a pin still rejects the peer. If you pin, plan a safe rotation mechanism and overlapping pins before an emergency. The certificate path and the pin are separate predicates, so log them separately.

This inventory also clarifies what a root-store update means. Adding a public root to an old client may restore a path, but a privileged update mechanism must deliver it reliably and securely. Adding a private root to every process on a machine can grant authority far beyond one service. Prefer a scoped bundle loaded by the client that needs the private PKI, with ownership and rotation procedures. If the trust anchor itself must change, distribute the new anchor before switching the server chain, then prove representative clients accept both during the overlap. Remove the old anchor only after the old server paths have retired and rollback no longer requires it. That ordering is an operational inference from the path-validation dependency, not a universal timeline prescribed by RFC 5280.

For recurring checks, keep a small matrix rather than a huge collection of ad hoc screenshots. Rows are actual client profiles; columns are candidate served chains. Each cell records verification result and the path or error observed. Add the SNI and hostname to the test invocation so a certificate-selection defect does not masquerade as path incompatibility. Re-run the matrix when the CA changes an intermediate, an ACME client changes its preferred chain, a root nears retirement, or a base image updates its CA package. These events can alter trust without changing application code. A green synthetic using only the newest client profile is a reachability check, not a compatibility gate.

### Verify every terminator, not every source file

The production object is the listener that answers a ClientHello. A team may own a certificate in a secret manager, but a CDN edge owns the bytes the public client sees. An ingress controller may copy that secret into memory. A service mesh may establish a second TLS session using a different internal identity. List each terminator and its associated hostname, chain source, reload behavior, and verifier population. Then probe each one from the appropriate side of the boundary. A public probe cannot tell you which internal sidecar certificate the application sees; an internal probe cannot guarantee the public edge serves the new leaf.

This matters during rollback. Rolling back an application release might restore an old ingress configuration or secret reference, even though certificate renewal succeeded independently. If your monitor keys only on the certificate file in a central store, the rollback can reintroduce an expiring or mismatched served leaf without changing that file. An endpoint fingerprint check catches it. Keep the previous known-good chain available long enough for a controlled rollback, but do not let “rollback” mean restoring an expired certificate. A release gate should check the actual serial, SAN, issuer path, and verification result at the listener after traffic shifts.

When a service advertises several names, use a probe per name, not one per address. SNI routes by name before HTTP, and the cert selected for `api.example.com` may differ from the one for `admin.example.com` on the same IP. Conversely, one name may resolve to several addresses or regions, so one name probe from one vantage point can miss a stale member. Build a two-dimensional coverage map of names and endpoints. A full Cartesian product may be unnecessary if you know routing policy, but any omitted pair should be an explicit assumption. This is the same discipline used for [service discovery](/blog/software-development/networking/service-discovery-from-dns-to-registries-to-xds): know which address the client reaches and which identity it expects there.

## Run it yourself

### Question

Can a client that trusts the root and requests the correct SAN fail solely because the TLS server omits an intermediate certificate? We will hold the leaf, private key, root trust store, SNI, hostname, address, and port constant. The only treatment is adding the intermediate to the certificate chain the server sends. This deliberately tests path *construction*, not expiry or a cross-sign compatibility quirk. Those other mechanisms need their own client matrix.

### Preconditions

Use the Linux `netlab` from the [series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). It provides namespaces `c` and `s`, with `c0` at `10.77.0.1/30`, `s0` at `10.77.0.2/30`, and the route between them. On macOS, run inside the privileged Linux VM described there; `ip netns` is a Linux facility. You need permission to enter namespaces and bind port `8443`, `iproute2`, and OpenSSL 3.x. [OpenSSL 3.0 documents](https://docs.openssl.org/3.0/man1/openssl-s_server/) the `s_server -cert_chain` option used below, and its [client manual](https://docs.openssl.org/3.0/man1/openssl-s_client/) documents `-servername`, `-verify_hostname`, and `-verify_return_error`. Record your exact version with `openssl version -a` if the result differs.

This is a local test CA. Do not add its root to your system trust store or use its private key outside the lab. The generated key files belong in a temporary directory, not source control. The script tests namespace, interface, route, qdisc, listener, and tool state before generating certificates. It neither changes qdiscs nor alters routes. The only namespace operation is starting and stopping a server process in `s`; cleanup targets that process and the temporary directory. A certificate probe can reveal server names and chain details; when using packet captures in a different experiment, treat them as potentially containing credentials and personal data.

### Baseline

Run the next three code blocks in order in the **same interactive Bash session** on the Linux lab host after the base topology exists. Do not run the first block as a standalone script and then start a new shell, because its local variables and temporary CA would be removed at exit. The `trap` ensures the server and temporary files are removed when that Bash session exits early. The `openssl req` and `openssl x509` commands create a root, one constrained intermediate, and a leaf for `api.lab.example`. The `s_server` baseline presents only the leaf. The client trusts only the generated root and has no cached intermediate.

```bash
#!/usr/bin/env bash
set -euo pipefail

for tool in ip openssl ss; do
  command -v "$tool" >/dev/null || { echo "missing $tool" >&2; exit 1; }
done
openssl version
ip netns list | grep -Eq '^c([[:space:]]|$)'
ip netns list | grep -Eq '^s([[:space:]]|$)'
ip -n c addr show dev c0
ip -n s addr show dev s0
ip -n c route get 10.77.0.2
ip netns exec c ping -c 1 -W 2 10.77.0.2
ip -n c qdisc show dev c0
ip -n s qdisc show dev s0
ip netns exec s ss -lnt '( sport = :8443 )'

labdir=$(mktemp -d)
server_pid=''
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
  rm -rf -- "$labdir"
}
trap cleanup EXIT

openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 2 \
  -subj '/CN=Netlab Root' \
  -addext 'basicConstraints=critical,CA:TRUE,pathlen:1' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign' \
  -keyout "$labdir/root.key" -out "$labdir/root.crt" >/dev/null 2>&1

openssl req -new -newkey rsa:2048 -sha256 -nodes \
  -subj '/CN=Netlab Intermediate' \
  -keyout "$labdir/intermediate.key" \
  -out "$labdir/intermediate.csr" >/dev/null 2>&1
cat >"$labdir/intermediate.ext" <<'EOF'
basicConstraints=critical,CA:TRUE,pathlen:0
keyUsage=critical,keyCertSign,cRLSign
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid,issuer
EOF
openssl x509 -req -in "$labdir/intermediate.csr" \
  -CA "$labdir/root.crt" -CAkey "$labdir/root.key" -CAcreateserial \
  -days 2 -sha256 -extfile "$labdir/intermediate.ext" \
  -out "$labdir/intermediate.crt" >/dev/null 2>&1

openssl req -new -newkey rsa:2048 -sha256 -nodes \
  -subj '/CN=api.lab.example' \
  -keyout "$labdir/leaf.key" -out "$labdir/leaf.csr" >/dev/null 2>&1
cat >"$labdir/leaf.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:api.lab.example
authorityKeyIdentifier=keyid,issuer
EOF
openssl x509 -req -in "$labdir/leaf.csr" \
  -CA "$labdir/intermediate.crt" -CAkey "$labdir/intermediate.key" \
  -CAcreateserial -days 2 -sha256 -extfile "$labdir/leaf.ext" \
  -out "$labdir/leaf.crt" >/dev/null 2>&1

openssl verify -CAfile "$labdir/root.crt" \
  -untrusted "$labdir/intermediate.crt" "$labdir/leaf.crt"
ip netns exec s openssl s_server -accept 10.77.0.2:8443 \
  -cert "$labdir/leaf.crt" -key "$labdir/leaf.key" \
  -quiet >"$labdir/server.log" 2>&1 &
server_pid=$!
sleep 1
ip netns exec s ss -lnt '( sport = :8443 )'

set +e
ip netns exec c openssl s_client -connect 10.77.0.2:8443 \
  -servername api.lab.example -verify_hostname api.lab.example \
  -CAfile "$labdir/root.crt" -verify_return_error \
  -showcerts </dev/null >"$labdir/baseline.txt" 2>&1
baseline_rc=$?
set -e
echo "baseline_exit=$baseline_rc"
grep -E 'verify error|Verify return code|Certificate chain' "$labdir/baseline.txt" || true

# Continue with the treatment and comparison commands below before exit.
```

Read `baseline_exit`, the verification error, and the `Certificate chain` section of `baseline.txt`. With OpenSSL 3.x and the temporary store containing only `root.crt`, expect a nonzero exit and an issuer-resolution error such as `unable to get local issuer certificate`. The exact error depth and wording can vary by OpenSSL build. The server should show one leaf in the served chain. The local `openssl verify` line immediately before the server starts should succeed, proving that the leaf and intermediate signatures are usable when the intermediate is available. If the baseline unexpectedly succeeds, inspect whether your client loaded an intermediate from an additional store or whether the server sent more than the leaf.

### Apply one change

Keep using the same shell process and variables. Stop only the baseline server, then restart with exactly the same leaf and private key plus `-cert_chain "$labdir/intermediate.crt"`. We are changing what the server presents, not what the client trusts.

```bash
kill "$server_pid"
wait "$server_pid" 2>/dev/null || true
server_pid=''
ip netns exec s openssl s_server -accept 10.77.0.2:8443 \
  -cert "$labdir/leaf.crt" -key "$labdir/leaf.key" \
  -cert_chain "$labdir/intermediate.crt" \
  -quiet >"$labdir/server.log" 2>&1 &
server_pid=$!
sleep 1
ip netns exec s ss -lnt '( sport = :8443 )'
```

### Compare

Run the same client command with the same SNI, hostname, root file, and peer address. The certificate chain is the single treatment. Because `s_client` output format changes between releases, inspect both process exit and the textual verification result.

```bash
set +e
ip netns exec c openssl s_client -connect 10.77.0.2:8443 \
  -servername api.lab.example -verify_hostname api.lab.example \
  -CAfile "$labdir/root.crt" -verify_return_error \
  -showcerts </dev/null >"$labdir/treatment.txt" 2>&1
treatment_rc=$?
set -e
echo "treatment_exit=$treatment_rc"
grep -E 'Verify return code|Certificate chain|Verification:|verify error' \
  "$labdir/treatment.txt" || true
```

Expected qualitative state: `treatment_exit=0` and `Verify return code: 0 (ok)` or equivalent success wording, with the served `Certificate chain` including the leaf and intermediate. The baseline should fail while the treatment succeeds. No latency target is promised: this lab proves a validation state change, and both handshakes occur on a local veth pair. Scheduler timing and RSA key generation can vary; neither changes the expected verify result. If `s_server` does not start, inspect `server.log` and the listener output. If both commands fail, verify `root.crt`, the intermediate's CA constraints, SAN, and whether the generated files are readable from the host namespace. If both succeed, check that your baseline did not inherit a cached intermediate or a second trust store.

This experiment is deliberately narrower than the 2021 case. It shows why the client needs intermediate material. The Let's Encrypt case adds cross-signs and implementation-dependent path selection even when the server sends a full chain. Do not infer that appending every possible issuer certificate is a safe fix. Serve a deliberate chain selected against the clients you support.

### Reset

At normal script exit, the `trap` kills the lab server process and removes only the `mktemp` directory it created. If you run the blocks interactively, execute the same scoped cleanup yourself:

```bash
if [[ -n "${server_pid:-}" ]]; then
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
fi
if [[ -n "${labdir:-}" && -d "$labdir" ]]; then
  rm -rf -- "$labdir"
fi
ip netns exec s ss -lnt '( sport = :8443 )'
```

The final `ss` output should show no listener for this experiment. The script leaves namespace addresses, routes, qdiscs, and the base `netlab` processes unchanged. Run the full sequence in one shell because `labdir` and `server_pid` are shell variables, and the temporary root trust file is deleted by reset.

### Production translation

On a production endpoint, use read-only probes with explicit SNI and verification. Do not run the lab's certificate-generation or server commands on a production interface. The first command below prints what the endpoint serves. The second uses your real client trust store to verify it; choose the same trust profile the affected application uses.

```bash
openssl s_client -connect api.example.com:443 \
  -servername api.example.com -showcerts </dev/null
openssl s_client -connect api.example.com:443 \
  -servername api.example.com -verify_hostname api.example.com \
  -CAfile /path/to/client-trust-bundle.pem \
  -verify_return_error </dev/null
```

Read the PEM count and issuers in the first output, then the verification result in the second. If the real application still differs, test with its own TLS library and trust store. A server can serve a complete chain and still fail a client because of trust-anchor age, algorithm policy, revocation policy, or a path-building bug. The lab's controlled change gives you one causal mechanism to recognize; it does not turn every issuer error into the same diagnosis.

## Key takeaways

Certificate verification is a client-specific path computation. The server presents a leaf and supporting intermediates. The client contributes trust anchors, reference identity, time, algorithm and revocation policy, and a path-building implementation. Each ingredient can change the result. Inspect the served chain with explicit SNI, then reproduce with the failing client's verifier and trust store before making a change.

Operate certificate renewal as an endpoint deployment, not a filesystem event. Monitor the serial and expiry actually served at each TLS terminator, with the SNI and hostname customers use. Alert on failure to deploy a renewed serial as well as impending expiry. Keep chain choice in your supported-client test matrix, especially during root transitions. The DST Root CA X3 transition is the proof that a chain can preserve compatibility for one old client population while causing trouble for another.

The larger [network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) ends at the same diagnostic rule: locate the boundary and choose evidence that distinguishes explanations. For certificates, that evidence is the exact leaf, chain, trust anchor, reference name, client version, and verification result. Once those are written down, “TLS is broken” becomes a repairable statement.

## Further reading

- [RFC 5280: X.509 certificate and CRL profile](https://www.rfc-editor.org/rfc/rfc5280.html), May 2008, for path validation and certificate extensions.
- [RFC 9525: Service Identity in TLS](https://www.rfc-editor.org/rfc/rfc9525.html), November 2023, for reference names and SAN matching.
- [RFC 6066: TLS extensions](https://www.rfc-editor.org/rfc/rfc6066.html), January 2011, for SNI and certificate-status request behavior.
- [Let's Encrypt: DST Root CA X3 expiration](https://letsencrypt.org/ca/docs/dst-root-ca-x3-expiration-september-2021/), updated February 2024, for the historical client compatibility split.
- [Certificate Transparency: How CT works](https://certificate.transparency.dev/howctworks/) for issuance visibility and SCTs.
