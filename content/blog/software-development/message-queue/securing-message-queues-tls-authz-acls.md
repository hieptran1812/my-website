---
title: "Securing Message Queues: TLS, Authentication, Authorization, and Multi-Tenancy"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A message broker holds your most sensitive data in motion, and the default install ships wide open. This is the layer-by-layer hardening guide: TLS in transit and the zero-copy tax it imposes, encryption at rest, SASL versus mTLS authentication, default-deny ACLs and RBAC, per-tenant quotas and namespaces, audit logging, secrets management, and why unsecured brokers keep ending up dumped on the internet."
tags:
  [
    "message-queue",
    "security",
    "tls",
    "authentication",
    "authorization",
    "kafka",
    "rabbitmq",
    "multi-tenancy",
    "distributed-systems",
    "event-driven",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/securing-message-queues-tls-authz-acls-1.webp"
---

There is a genre of news story that recurs every few months, almost word for word, with only the company name swapped out. "Researchers discover unsecured database exposing millions of records." "Misconfigured server leaks customer data to the open internet." The database in the headline is usually Elasticsearch, MongoDB, or Redis, but increasingly it is a message broker — a Kafka cluster, a RabbitMQ node, a NATS server — sitting on a public IP with no authentication, no encryption, and no access control, quietly streaming a company's entire event history to anyone who runs a port scan. The data was not stolen through some clever exploit. It was handed over, because the broker was configured to hand it over to anyone who asked, and the only reason it had not happened sooner was that nobody had asked yet.

This post is about making sure nobody can just ask. A message broker is, by its nature, one of the most security-sensitive systems you run, because it is where your data lives *in motion* and where it briefly pools at rest. Every order, every payment event, every user action, every change-data-capture stream off your primary database — it all flows through the broker, often in plaintext on the wire and in plaintext on disk, and any principal that can connect can frequently read all of it. A compromised broker is not one leaked table; it is a live feed of everything your business does, replayable from the beginning of retention. That is a worse blast radius than most databases, and yet brokers are routinely treated as plumbing that lives safely inside the network and therefore needs no locks.

The security model of a broker is best understood as a stack of independent layers, each guarding a distinct failure, and the figure below is the spine of this entire post. At the bottom is TLS, which makes the wire confidential so a network observer learns nothing. Above it is authentication, which proves who is connecting. Above that is authorization, which decides what that proven identity is allowed to do. Above that are quotas, which keep one tenant from starving the others. And wrapping all of it is audit logging, which records who did what so that when something does go wrong you can reconstruct it. The critical property is that these layers compose but do not substitute: TLS without authentication encrypts a conversation with an attacker; authentication without authorization proves the identity of someone who can then read everything; authorization without quotas lets an authorized client take down the cluster by accident. You need all of them, and you need to understand what each one does and does not protect against.

![A vertical stack of the five broker security layers showing TLS in transit at the base, then authentication, then authorization, then per-tenant quotas, then audit logging wrapping the whole stack](/imgs/blogs/securing-message-queues-tls-authz-acls-1.webp)

By the end of this post you will be able to do several concrete things. You will be able to reason about a broker's threat model — who the realistic attackers are and what each layer stops. You will be able to enable TLS for client and inter-broker traffic and quantify the throughput you give up when you do, because TLS defeats the zero-copy read path that makes Kafka fast. You will be able to choose a SASL mechanism — SCRAM, Kerberos, OAuth, or mTLS certificates — with a clear sense of what each is good for. You will be able to write default-deny ACLs that grant a service exactly the operations it needs and nothing more, and layer RBAC on top when ACLs get unwieldy. You will be able to carve a cluster into tenants with quotas and namespaces so a noisy neighbor cannot starve the rest. And you will understand, in operational detail, why the unsecured-broker catastrophe keeps happening and the specific configuration mistakes that cause it. This post is a companion to [Choosing a message broker](/blog/software-development/message-queue/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs), because security capabilities differ sharply across brokers and should weigh in that choice, and to [RabbitMQ Deep Dive Part 1](/blog/software-development/message-queue/rabbitmq-amqp-exchanges-bindings-routing), because the exchange-and-binding model shapes how you scope permissions. It forward-links the operational sibling on [durability and disaster recovery](/blog/software-development/message-queue/durability-and-disaster-recovery-for-message-queues), since your backups inherit whatever encryption-at-rest posture you set here.

## 1. The threat model for a message broker

Before you configure a single security control, you should be able to name what you are defending against. Security work that skips the threat model produces theater — elaborate controls against attacks nobody will mount, and gaping holes where the real risk lives. For a message broker, the realistic threats fall into a small number of categories, and naming them tells you which layer of the stack addresses each one.

The first threat is the **network eavesdropper**. Someone with visibility into the network path between a client and the broker — a compromised host on the same subnet, a malicious actor on a shared cloud network, an attacker who has pivoted into your VPC — can read every byte that crosses the wire if that wire is plaintext. Because brokers move your most sensitive data continuously, a passive tap is enormously valuable. This is what TLS defends against: it makes the bytes on the wire unreadable to anyone who does not hold the session keys.

The second threat is the **unauthenticated connector**. This is the headline-grabbing one. If the broker accepts connections without proving identity, then anyone who can reach the listening port can connect, list topics, and consume data. On a broker bound to a public interface this means the entire internet; on a broker inside a flat network this means any compromised host, any contractor's laptop, any service that should not have access. Authentication defends against this by refusing to do anything for a connection that has not proven who it is.

The third threat is the **over-privileged insider or compromised credential**. Even with authentication, a single set of credentials that can read and write everything is a catastrophe waiting for a leaked password, a hardcoded secret in a git repo, or a malicious employee. The blast radius of a leaked credential is exactly the set of operations that credential is authorized to perform. Authorization — least-privilege ACLs — shrinks that blast radius from "everything" to "the one topic this service legitimately uses."

The fourth threat is the **noisy or hostile tenant**. In any cluster shared across teams or customers, one tenant can degrade or deny service to the others, whether maliciously or through a runaway loop that produces ten million messages a second. This is an availability threat rather than a confidentiality one, and it is addressed by quotas and isolation.

The fifth threat, often forgotten, is the **data-at-rest exposure**: someone who gets access to the broker's disks, its backups, or a decommissioned drive can read the log segments directly, bypassing every network control. Encryption at rest defends the bytes once they have landed on storage.

The figure below is the request-path view of how these layers actually intercept a request in sequence. A client's produce request does not reach the topic log until it has cleared the TLS gate, proven an identity, and passed an ACL check. There is no path around the gates — the broker will not append a record for a connection that failed authentication, and it will not append for a principal whose ACLs do not permit writing to that topic. Each gate is checked in order, and a failure at any gate short-circuits the request.

![A grid showing a produce request flowing from client through a TLS handshake gate then an authentication gate then an ACL check before reaching the topic log, with the principal bound to the connection throughout](/imgs/blogs/securing-message-queues-tls-authz-acls-2.webp)

The reason it helps to enumerate threats this explicitly is that it immediately exposes the most common security failure: implementing one layer and assuming it covers another. A team that turns on TLS and declares the broker "encrypted and therefore secure" has defended against the eavesdropper and done nothing about the unauthenticated connector — the attacker simply completes the TLS handshake (the server's certificate is public, after all) and connects to a fully encrypted, fully open broker. A team that adds authentication but gives every service the same admin credential has defended against anonymous connection and done nothing about credential compromise. The threat model is what keeps you honest about which holes are actually plugged.

### The trust boundary fallacy

The single most damaging assumption in broker security is "it is inside the network, so it is safe." This is the trust boundary fallacy, and it is responsible for the majority of the breaches we will discuss in section 8. The reasoning goes: the broker is not exposed to the internet, only internal services connect to it, the network perimeter is the security boundary, so the broker itself does not need authentication. Every clause of that reasoning is fragile. "Not exposed to the internet" is one security-group misconfiguration away from false. "Only internal services connect" assumes no host inside the network is ever compromised, which is the assumption every successful breach violates. "The network perimeter is the security boundary" is the flat-network model that modern security explicitly rejects in favor of zero trust, where every connection authenticates regardless of where it originates.

The correct posture is to assume the network is hostile — that an attacker is already inside it — and to make the broker secure on its own merits, not because of where it sits. This is not paranoia; it is the lesson of every internal-pivot breach. The broker should require authentication from every client, including ones on the same host, and it should encrypt traffic even on a trusted link, because "trusted" is a property that evaporates the instant any host on that link is compromised. The rest of this post assumes the zero-trust posture, because the alternative is the recurring headline.

### What each layer does and does not protect

A useful exercise before configuring anything is to map each threat to the layer that stops it and, crucially, to note what that layer leaves open, because the gaps are where breaches live. TLS stops the eavesdropper but does nothing about authentication — an attacker can complete a TLS handshake to a broker whose certificate is public and then connect to a fully encrypted, fully open broker. Authentication stops the anonymous connector but does nothing about an over-privileged or compromised credential — a verified identity with admin rights is still a full breach if its credential leaks. Authorization shrinks the blast radius of a compromised credential but does nothing about availability — an authorized principal can still take down the cluster by flooding it. Quotas protect availability but do nothing about confidentiality — a throttled tenant is still reading whatever its ACLs permit. Encryption at rest protects offline storage but does nothing against a compromised running process that holds the keys. Audit logging protects nothing in real time — it does not stop anything; it only records, so its entire value is realized after the fact.

Reading that list, the discipline becomes obvious: no single layer is sufficient, each one has a gap that exactly the next layer fills, and removing any one of them reopens a class of attack that the others cannot cover. This is why the checklist in section 9 walks every branch and why a "secured" broker that implements four layers and skips the fifth is not eighty percent secure — it is fully exposed to the one threat the missing layer addresses. Security here is conjunctive, not additive: you need all of the controls true at once, and the weakest one defines your actual posture.

## 2. Encryption in transit with TLS (and its cost)

TLS — Transport Layer Security, the protocol formerly known as SSL — is the foundation, because it is the layer that makes everything above it meaningful. Authentication credentials sent over a plaintext connection can be sniffed and replayed; ACL checks protect data that is then shipped in the clear to the consumer. TLS wraps the TCP connection in an encrypted, integrity-protected tunnel: an eavesdropper sees only ciphertext, and any tampering with the bytes in flight is detected and the connection torn down. For a broker, there are two distinct traffic flows to encrypt, and they have different threat profiles.

The first is **client-to-broker** traffic: producers and consumers talking to the broker. This is the obvious one, and it is where the eavesdropper threat is sharpest, because client traffic often traverses more network hops — across availability zones, sometimes across the public internet for clients outside the VPC. The second is **inter-broker** traffic: the replication streams that copy partitions between brokers, plus controller and metadata traffic. This is easy to forget because it is internal, but it carries exactly the same data — the replication stream is a full copy of every message — so leaving it plaintext leaves your data exposed on the very links you most assumed were safe. A serious deployment encrypts both.

### How the handshake works, and why it costs you

When a client connects to a TLS-enabled broker, a handshake runs before any application data flows, and understanding its steps explains both the security it buys and the latency it costs. The figure below lays out the sequence for a TLS handshake followed by a SASL authentication exchange, which is the full identity-establishment dance on a secured Kafka connection.

![A timeline of the TLS and SASL handshake showing ClientHello, server certificate verification against a trusted CA, key exchange, the SASL mechanism negotiation, a SCRAM challenge and proof, a failure branch that closes the connection, and finally the principal being bound so requests can flow](/imgs/blogs/securing-message-queues-tls-authz-acls-9.webp)

The client opens with a `ClientHello` advertising the TLS versions and cipher suites it supports. The broker responds with its **certificate**, which the client validates against a trusted certificate authority — this is what proves the client is talking to the real broker and not a man-in-the-middle. The two sides then perform a key exchange (an elliptic-curve Diffie-Hellman in modern TLS) to derive a shared session key without ever transmitting it, and from that point the link is encrypted. With TLS 1.3 this takes one network round trip; with the older TLS 1.2 it takes two. On top of that comes the SASL exchange to establish identity, which we cover in section 4. The practical cost of the handshake is a one-time latency hit at connection setup — a few milliseconds of round trips plus some asymmetric crypto — which is why you want **long-lived connections** with a broker, not a fresh connection per request. A client that reconnects constantly pays the handshake tax over and over.

The handshake cost is real but bounded; the cost that actually shapes broker capacity is the **per-byte encryption cost on the data path**, and specifically what it does to zero-copy. This is the single most important performance fact about broker TLS, and it deserves its own treatment.

### The zero-copy tax

A high-throughput broker like Kafka serves reads using the `sendfile` system call, which streams bytes directly from the page cache to the network socket without ever copying them into application memory. The kernel reads the file data and hands it to the network card via DMA, and the broker's CPU never touches the bytes. This is the zero-copy read path, and it is a large part of why Kafka can saturate a network interface with a modest CPU. I dig into the mechanics of this in [the broker I/O optimization companion](/blog/software-development/message-queue/broker-io-optimization-zero-copy-tiered-storage); the relevant point here is what TLS does to it.

TLS requires the broker to **encrypt every byte before it goes on the wire**, and you cannot encrypt bytes the CPU never touches. So enabling TLS forces the broker to abandon `sendfile` and instead copy each segment of data from the page cache into user space, encrypt it, and write the ciphertext to the socket. The zero-copy fast path is gone. The CPU now processes every outbound byte, and on a read-heavy broker that is a substantial new cost. In practice, enabling TLS on a Kafka cluster typically costs somewhere between 20% and 40% of throughput on the same hardware, with the exact figure depending on the cipher suite, whether the CPU has AES-NI hardware acceleration (modern server CPUs do, which helps enormously), and how read-heavy the workload is. A write-heavy cluster with little read fan-out loses less, because the write path was already copying data; a read-heavy cluster with large fan-out loses more, because it gives up the most.

This is not a reason to skip TLS — the security is non-negotiable for sensitive data — but it is a reason to *plan capacity for it*. If you benchmark a cluster without TLS, decide it needs six brokers, and then turn on TLS, you may find you need eight. Budget the zero-copy tax up front. Modern mitigations exist: AES-NI makes the AES-GCM cipher suites cheap enough that the copy itself, not the crypto, becomes the dominant cost; kernel TLS (kTLS) can push encryption back into the kernel and partially restore zero-copy on supported platforms; and newer cipher suites like ChaCha20-Poly1305 help on CPUs without AES acceleration. But the baseline assumption should be that TLS costs you a meaningful slice of read throughput, and you should measure it on your hardware rather than guess.

### Configuring TLS on Kafka

Here is the concrete configuration. On the broker, you define a listener that uses the `SSL` security protocol and point it at a keystore holding the broker's certificate and private key, and a truststore holding the CA that signs client certificates if you are doing mutual TLS:

```properties
# server.properties on each broker
listeners=SSL://0.0.0.0:9093,SASL_SSL://0.0.0.0:9094
advertised.listeners=SSL://broker1.internal:9093,SASL_SSL://broker1.internal:9094

ssl.keystore.location=/etc/kafka/secrets/broker1.keystore.jks
ssl.keystore.password=${file:/etc/kafka/secrets/creds.properties:keystore-pw}
ssl.key.password=${file:/etc/kafka/secrets/creds.properties:key-pw}
ssl.truststore.location=/etc/kafka/secrets/truststore.jks
ssl.truststore.password=${file:/etc/kafka/secrets/creds.properties:truststore-pw}

# Encrypt the replication and controller traffic too, not just clients
security.inter.broker.protocol=SSL

# Modern TLS only; refuse the old, attackable versions
ssl.enabled.protocols=TLSv1.3,TLSv1.2
ssl.cipher.suites=TLS_AES_256_GCM_SHA384,TLS_AES_128_GCM_SHA256
```

The `security.inter.broker.protocol=SSL` line is the one teams forget. Without it, your client traffic is encrypted while your replication traffic — a full copy of the same data — flows in plaintext between brokers. Note also that the passwords are pulled from a file via the `${file:...}` config-provider syntax rather than written inline; section 7 covers why credentials should never be literals in a config file.

On the client side, the producer or consumer needs the truststore so it can validate the broker's certificate:

```properties
# client.properties
security.protocol=SSL
ssl.truststore.location=/etc/kafka/secrets/truststore.jks
ssl.truststore.password=changeit
# Verify the broker hostname matches its certificate (prevents MITM)
ssl.endpoint.identification.algorithm=https
```

That last line is quietly critical. `ssl.endpoint.identification.algorithm=https` tells the client to verify that the hostname it connected to matches the hostname in the broker's certificate. Disabling it — which some teams do to silence a certificate-mismatch error during a hurried setup — reopens the man-in-the-middle hole that TLS was supposed to close, because now the client will accept *any* valid certificate, including one an attacker presents. The single most common TLS misconfiguration on brokers is turning hostname verification off to make an error go away. Do not do it; fix the certificate's subject alternative names instead.

## 3. Encryption at rest

TLS protects bytes while they move. Encryption at rest protects them once they stop moving — when they have been appended to a log segment on disk, written to a snapshot, or copied into a backup. The threat here is different: not a network eavesdropper but someone with physical or logical access to the storage. A stolen disk, a decommissioned drive that was not wiped, a leaked cloud storage snapshot, a backup file copied to an attacker's bucket, or simply another process on the broker host reading the log files directly. None of these touch the network, so TLS does nothing for them.

The honest framing is that **encryption at rest defends a narrower threat than people assume**, and it is worth being precise about what it does and does not buy you. It protects against offline access to the storage medium: if someone gets the raw disk or the backup file but not the running system and its keys, the data is gibberish. It does *not* protect against an attacker who has compromised the running broker process, because that process necessarily holds the keys to decrypt its own data — it has to, in order to serve reads. So encryption at rest is a control against storage-layer exposure, not against application-layer compromise. This matters because it tells you where to invest: encryption at rest is most valuable when your storage and your compute have different trust boundaries, which is exactly the case for backups and for cloud block storage.

### The two places to encrypt at rest

There are two practical layers at which to encrypt broker data at rest, and the choice between them is a real tradeoff.

The first is **volume-level encryption**: the disk or cloud block volume is encrypted underneath the filesystem, transparently to the broker. On Linux this is LUKS/dm-crypt; in the cloud it is EBS encryption, GCP persistent-disk encryption, or Azure disk encryption, usually with keys managed by the cloud KMS. The broker writes plaintext to what it thinks is a normal filesystem, and the block layer encrypts on the way to physical storage and decrypts on the way back. The advantages are huge: it is transparent to the broker (zero application changes), it covers everything on the volume including indexes and snapshots, and with hardware acceleration its performance cost is negligible — often under a few percent. The limitation is that it protects only against someone getting the raw storage; any process that can read the mounted filesystem sees plaintext, because decryption happens at the block layer below the filesystem. For the overwhelming majority of deployments, **volume-level encryption is the right default**: it is cheap, transparent, and covers the realistic at-rest threats of stolen disks and leaked snapshots.

The second is **application-level or message-level encryption**, where the data is encrypted by the producer before it is ever sent, so the broker only ever stores ciphertext and never holds the keys. This is the strongest model — even a fully compromised broker leaks nothing useful, because it cannot decrypt — but it is also the most operationally demanding. The broker can no longer do anything that requires reading message content: no log compaction by key if the key is encrypted, no server-side filtering, no schema validation. Key management becomes the producers' and consumers' problem, and key rotation means re-encrypting or maintaining old keys for the retention window. Reserve message-level encryption for genuinely high-sensitivity data — payment card numbers, health records, secrets in transit — where the broker should be a zero-knowledge pipe, and accept the operational weight that comes with it. For everything else, volume encryption plus strong transit and access controls is the pragmatic posture.

### Do not forget the backups

The most common at-rest failure is not the live disks; it is the **backups**. A team encrypts its production EBS volumes, feels secure, and then runs a nightly backup that copies log segments to an object-storage bucket in plaintext, or with the bucket left world-readable. The backup is a perfect copy of all your data, sitting in a place with its own, often weaker, access controls. Every encryption-at-rest decision must extend to the backup and disaster-recovery path — the snapshots, the object-storage tiers, the cross-region copies. This is precisely where this post hands off to [the durability and disaster recovery sibling](/blog/software-development/message-queue/durability-and-disaster-recovery-for-message-queues): your DR copies inherit whatever encryption posture you set here, and an unencrypted backup of an encrypted cluster is the weakest link that defines your actual security.

## 4. Authentication: SASL mechanisms and mTLS

Authentication answers the question "who is this connection?" and it is the layer whose absence causes the catastrophic breaches. A broker without authentication will happily serve anyone; a broker with authentication refuses to do anything for a connection until it has proven an identity, which the broker then binds to the connection as a **principal**. Every subsequent authorization decision is made against that principal. Get authentication wrong and nothing above it matters, because the ACLs are checking the permissions of an identity that was never really verified.

Kafka and most brokers support two broad families: SASL mechanisms and mutual TLS. SASL — the Simple Authentication and Security Layer — is a pluggable framework that supports several mechanisms with very different properties. The figure below lays them out across the axes that actually drive the decision: credential strength, operational cost, and the deployment where each one shines.

![A matrix comparing SASL PLAIN, SCRAM-SHA-512, Kerberos, OAUTHBEARER, and mTLS certificates across credential strength, operational cost, and best-fit deployment, showing PLAIN as weak cleartext suitable only over TLS for development and SCRAM as the low-cost default](/imgs/blogs/securing-message-queues-tls-authz-acls-3.webp)

### SASL/PLAIN — only ever over TLS

SASL/PLAIN sends a username and password in the clear. By itself this is indefensible — it is a cleartext credential on the wire. Its only legitimate use is **inside a TLS tunnel** (the `SASL_SSL` listener), where TLS encrypts the cleartext so the password is never exposed on the network. Even then, PLAIN has the weakness that the broker must have access to verify the password, which usually means storing it or a hash of it in a config file or a callback. PLAIN is acceptable for quick development setups and for cases where the credential is a short-lived API token validated by a custom callback, but for a username-and-password world there is a strictly better option that costs almost nothing more.

### SASL/SCRAM — the sensible default

SCRAM — Salted Challenge Response Authentication Mechanism — is the right default for most clusters. Instead of sending the password, the client and server perform a challenge-response exchange that proves the client knows the password without ever transmitting it, and the broker stores only a salted hash, not the password itself. This means a leaked broker credential store does not directly leak passwords (an attacker gets salted hashes they must crack), and a network capture even without TLS does not reveal the password (though you still want TLS for the data). SCRAM credentials are stored in the cluster metadata, so you create and rotate them with admin commands rather than redeploying config. For the vast majority of deployments, **SCRAM-SHA-512 over a TLS listener is the correct, boring, secure default**, and you should reach for it unless you have a specific reason to use something else.

```bash
# Create a SCRAM credential for a service principal, stored in cluster metadata
kafka-configs --bootstrap-server broker1:9094 \
  --command-config admin.properties \
  --alter --add-config 'SCRAM-SHA-512=[password=s3rv1ce-secret]' \
  --entity-type users --entity-name orders-producer
```

### SASL/GSSAPI (Kerberos) — when you already have a KDC

Kerberos, exposed through SASL as GSSAPI, is ticket-based authentication tied to a Key Distribution Center. Its great strength is single sign-on across an enterprise: if your organization already runs Active Directory or another Kerberos realm, Kerberos lets the broker plug into that existing identity infrastructure, so principals are managed centrally and users authenticate with the same identity they use everywhere else. Its great weakness is operational weight: you need a KDC, keytab files distributed to every client, clock synchronization across the fleet (Kerberos is famously unforgiving of clock skew), and the general operational complexity of the Kerberos ecosystem. Reach for Kerberos when you are in an enterprise that already lives in Active Directory and wants the broker to honor that identity system; do not stand up a KDC just for the broker.

### SASL/OAUTHBEARER — cloud-native and short-lived

OAUTHBEARER authenticates with an OAuth 2.0 bearer token — typically a short-lived JWT issued by an identity provider like Okta, Auth0, Azure AD, or a cloud IAM system. The client obtains a token from the IdP, presents it to the broker, and the broker validates it (checking the signature against the IdP's public keys and the expiry). This is the cloud-native, zero-trust-friendly choice: credentials are short-lived (a token might live fifteen minutes, sharply limiting the value of a leak), centrally issued and revocable, and tied into your existing single-sign-on. The cost is that you need an identity provider and the plumbing for clients to fetch and refresh tokens. For greenfield cloud deployments where an IdP already exists, OAUTHBEARER is increasingly the best answer, because short-lived tokens are simply a better credential model than long-lived passwords.

### mTLS — identity from the certificate itself

Mutual TLS reuses the TLS layer for authentication: instead of the client only verifying the broker's certificate, the *broker also requires and verifies a client certificate*, and the client's identity is the certificate's distinguished name. There is no separate password or token; proving possession of the private key for a CA-signed certificate *is* the authentication. This is elegant — one mechanism for both encryption and identity — and it is the natural fit for service-mesh environments and for inter-broker authentication, where every component already has a certificate. The cost is **certificate lifecycle management**: you need a certificate authority, a way to issue certificates to every client, and, most painfully, a way to rotate and revoke them. Certificate rotation at scale is a genuine operational discipline, and revocation (via CRLs or OCSP) is notoriously fiddly. mTLS shines when you already have PKI infrastructure — a service mesh, an internal CA with automated issuance like Vault or cert-manager — and becomes a burden when you are managing certificates by hand. It is also the standard choice for the inter-broker link, where the brokers' own certificates serve double duty for encryption and mutual authentication.

The decision among these is less about which is "most secure" — SCRAM, OAuth, and mTLS are all strong — and more about which fits your existing identity infrastructure. If you already run Active Directory, Kerberos is natural. If you already run an OAuth IdP, OAUTHBEARER is natural. If you already run a service mesh with PKI, mTLS is natural. If you run none of these and just need solid authentication, SCRAM over TLS is the path of least resistance and least regret.

### Connecting a SCRAM client and binding the principal

It helps to see the client side of authentication, because it makes concrete how the credential becomes a principal that authorization then checks. A Kafka client on a `SASL_SSL` listener configures both the SASL mechanism and the JAAS login that supplies the credential:

```properties
# producer/consumer client.properties for SCRAM over TLS
security.protocol=SASL_SSL
sasl.mechanism=SCRAM-SHA-512
sasl.jaas.config=org.apache.kafka.common.security.scram.ScramLoginModule required \
  username="fulfillment-svc" \
  password="${file:/etc/secrets/creds.properties:fulfillment-pw}";

# Still validate the broker's certificate and hostname
ssl.truststore.location=/etc/kafka/secrets/truststore.jks
ssl.truststore.password=changeit
ssl.endpoint.identification.algorithm=https
```

When this client connects, the broker runs the SCRAM challenge-response, and on success it binds the principal `User:fulfillment-svc` to the connection. From that moment, the principal is fixed for the life of the connection — it cannot be changed or escalated by anything the client sends — and every request the client makes is authorized against `User:fulfillment-svc` specifically. This binding is the hinge between authentication and authorization: authentication runs once and establishes the principal; authorization then runs per request against that fixed identity. Note again the password comes from a file reference, not a literal in the client config, for the reasons section 7 details. The principal name the broker derives — `User:<username>` for SASL, or the certificate's distinguished name for mTLS — is exactly the string you will name in your ACLs, so consistency between how a client authenticates and how you write its ACLs matters: a mismatch between the authenticated principal name and the ACL's principal name is a quiet way to lock a service out or, worse, to write an ACL that matches nothing and silently grants nothing.

### A note on rotating credentials without downtime

Authentication is not a one-time setup; credentials must rotate, and how a mechanism handles rotation is a real operational property. SCRAM credentials can be added and removed in cluster metadata while the broker runs, so the rotation pattern is to add the new credential, roll clients onto it, then remove the old one — a clean overlap with no downtime. Certificates are harder: you must issue the new certificate, distribute it, and reload it on both clients and brokers before the old one expires, and the failure mode of a forgotten certificate expiry is a cluster-wide outage when every connection suddenly fails validation at once. This is the strongest practical argument for short-lived OAuth tokens and for automated certificate issuance (cert-manager, Vault PKI): they make rotation a background process rather than a calendar emergency. A security posture that depends on someone remembering to rotate a credential before it expires is a posture with a built-in incident.

## 5. Authorization: ACLs and default-deny

Authentication tells the broker *who* is connecting. Authorization decides *what they may do*. The two are independent and both necessary: a perfectly authenticated principal with no access control is just a verified identity that can read all your data. The unit of authorization in Kafka and most brokers is the **Access Control List entry**, and the mental model is a triple: a *principal* is allowed (or denied) an *operation* on a *resource*.

The principal is the authenticated identity — `User:orders-producer`, or a certificate DN, or an OAuth subject. The operation is one of a fixed set: Read, Write, Create, Describe, Alter, Delete, and a few cluster-level ones like ClusterAction. The resource is a topic, a consumer group, the cluster itself, or a few other types like transactional IDs. The figure below lays out the operation-by-resource grid, which is the space in which you write ACLs: each meaningful cell is a permission you can grant, and least privilege means granting only the cells a service genuinely needs.

![A matrix of ACL operations Read, Write, Create, Describe, and Alter against the resource types Topic, Group, and Cluster, marking which combinations are the common grants for a producer or consumer and which are dangerous admin-only powers](/imgs/blogs/securing-message-queues-tls-authz-acls-7.webp)

### Default-deny is the whole game

The single most important property of an authorization system is its **default**. Kafka's authorizer, when enabled, denies any operation that is not explicitly allowed by a matching ACL. This is default-deny, and it is the only safe default. The opposite — default-allow, where everything is permitted unless explicitly denied — is a security disaster, because it means every new topic, every new operation, every gap in your deny rules is an open door. With default-deny, a principal starts with zero permissions and you grant exactly what it needs; a forgotten ACL means a service cannot do something (a loud, obvious failure you fix immediately), rather than a service being able to do something it should not (a silent hole you discover after the breach).

There is one configuration trap here that has caused real incidents. Kafka has a setting `allow.everyone.if.no.acl.found`. When `true`, it means: if a resource has *no* ACLs at all, allow everyone to access it. This was offered as a migration convenience — turn on the authorizer without immediately breaking everything — but it quietly converts default-deny into default-allow-for-unprotected-resources. A new topic created without ACLs is wide open. **Set `allow.everyone.if.no.acl.found=false`** so that a resource with no ACLs is denied to everyone, which is what default-deny is supposed to mean. Leaving it `true` is one of the subtle ways a "secured" cluster turns out to have open topics.

```properties
# server.properties — turn on the authorizer with a strict default
authorizer.class.name=org.apache.kafka.metadata.authorizer.StandardAuthorizer
allow.everyone.if.no.acl.found=false
# Bootstrap superusers who can administer ACLs (no other powers come for free)
super.users=User:kafka-admin;User:cluster-operator
```

#### Worked example: least-privilege ACLs for a producer-plus-consumer service

Let me make this concrete with the exact scenario the threat model demands. We have a service, authenticated as the principal `User:fulfillment-svc`, that should be able to do exactly two things: **produce to the `orders` topic** and **consume as part of the consumer group `fulfillment`**. It should be able to do nothing else — not read other topics, not create topics, not touch the cluster. Default-deny handles "nothing else" for free; we just grant the two things it needs.

To produce to `orders`, the service needs Write on the topic, and Describe on the topic so it can fetch metadata (partition count, leadership):

```bash
# Allow fulfillment-svc to WRITE to the orders topic (produce)
kafka-acls --bootstrap-server broker1:9094 --command-config admin.properties \
  --add --allow-principal User:fulfillment-svc \
  --operation Write --operation Describe \
  --topic orders

# Allow it to READ from the orders topic (consume) and DESCRIBE for metadata
kafka-acls --bootstrap-server broker1:9094 --command-config admin.properties \
  --add --allow-principal User:fulfillment-svc \
  --operation Read --operation Describe \
  --topic orders

# Allow it to use exactly the consumer group "fulfillment" (join + commit offsets)
kafka-acls --bootstrap-server broker1:9094 --command-config admin.properties \
  --add --allow-principal User:fulfillment-svc \
  --operation Read \
  --group fulfillment
```

Notice what is *not* granted. There is no Create on any resource, so the service cannot auto-create topics — if it tries to produce to a topic that does not exist, it is denied rather than silently spawning a new topic. There is no ClusterAction, so it cannot touch cluster-wide settings. The group ACL names `fulfillment` exactly, so the service cannot commit offsets under some other group and read another team's stream. And crucially, there is no wildcard. A lazy version of this would grant `User:fulfillment-svc` Read and Write on `--topic '*'`, which "works" and is a catastrophe: the service can now read every topic in the cluster, so a compromise of this one service leaks everything. The whole point of least privilege is that the blast radius of a compromised `fulfillment-svc` credential is exactly the `orders` topic and the `fulfillment` group, and nothing more.

You can verify the principal's permissions by listing them:

```bash
kafka-acls --bootstrap-server broker1:9094 --command-config admin.properties \
  --list --principal User:fulfillment-svc
```

This is the discipline at the heart of broker authorization: enumerate exactly what each service does, grant exactly that, and let default-deny take care of everything else. It is more upfront work than one admin credential shared everywhere, and it is the difference between a leaked credential being a contained incident and being a company-ending breach.

### RBAC layered on top

Raw ACLs are powerful but they do not scale gracefully. When you have two hundred services and fifty teams, writing per-principal ACLs by hand becomes an unmanageable sprawl of rules, and the natural failure mode is over-granting to reduce the toil. **Role-Based Access Control** layers a level of indirection on top: you define roles (`orders-team-producer`, `analytics-readonly`) as bundles of permissions, and you assign principals to roles rather than writing individual ACLs. Confluent's RBAC, Redpanda's RBAC, and similar systems implement this, often with the ability to scope roles to resource prefixes and to bind roles to groups from your identity provider. RBAC does not replace the ACL primitive — internally each role still expands into the same per-resource permissions — but it makes the permission model comprehensible at scale, and it lets you reason about "what can the analytics team do" rather than auditing a thousand individual rules. For small clusters, raw ACLs are fine; once the rule count and the team count grow, RBAC is how you keep authorization maintainable without sliding into over-grant.

### The authorization decision flow

It is worth tracing the full decision once, because it shows how authentication and authorization compose. The figure below is the authorization flow as a directed graph: a connection comes up over TLS, authenticates to resolve a principal, and then every operation triggers an ACL lookup that branches one of two ways — an explicit allow rule is found and the operation executes, or no rule is found and the request is denied and logged.

![A directed graph of the authorization flow from connect to authenticate to an ACL lookup that branches to either an explicit allow that executes the operation or a default deny that is logged to the audit trail](/imgs/blogs/securing-message-queues-tls-authz-acls-5.webp)

The thing to internalize from this flow is that authorization is checked *per operation*, not once at connection time. A principal that is allowed to read `orders` and tries to read `payments` is denied on the `payments` read specifically, even though the connection and authentication succeeded. This per-operation granularity is what makes least privilege meaningful: the principal is verified once, but its right to each individual action is checked every time, against the specific resource it is touching.

## 6. Multi-tenancy: quotas, namespaces, isolation

So far every layer has been about confidentiality and access — keeping the wrong people out. Multi-tenancy is about a different threat: keeping authorized tenants from harming each other. When a single cluster is shared across teams, applications, or customers, you have a tragedy-of-the-commons problem. The cluster has finite resources — network bandwidth, request-handling threads, disk I/O, connection slots — and any tenant can consume a disproportionate share, whether through a malicious flood, a runaway retry loop, or simply an honest workload that scaled faster than anyone planned. The figure below contrasts the two worlds: a cluster with no isolation where one tenant's flood starves everyone, versus one where quotas confine the damage to the offender.

![A before-and-after comparison showing a single-tenant cluster with no isolation where tenant A floods at 400 megabytes per second and starves tenants B and C, versus a multi-tenant cluster with quotas where tenant A is capped at 50 megabytes per second and throttled while B and C are unaffected](/imgs/blogs/securing-message-queues-tls-authz-acls-8.webp)

### Quotas: bandwidth and request-rate throttling

Kafka's primary isolation mechanism is **quotas**, and there are two kinds that matter. **Network bandwidth quotas** cap the bytes per second a principal or client can produce or fetch. **Request-rate quotas** cap the fraction of broker request-handling capacity (the I/O and network threads) a client can consume, expressed as a percentage — this catches clients that send a torrent of small requests, which a byte quota alone would miss. Quotas are enforced per broker and can be scoped to a user principal, a client ID, or the combination, which lets you give a tenant a total budget across all its clients.

The enforcement mechanism is elegant and worth understanding, because it shapes how a client experiences a quota. When a client exceeds its quota, the broker does **not** reject the request or drop the connection. Instead, it computes how long the client would need to pause to bring its rate back under the limit, and it **delays the response** by that amount — it holds the produce or fetch response, then sends it late with a `throttle_time_ms` field telling the client how long it was throttled. The client, seeing the delay, naturally slows down. This is graceful back-pressure rather than hard rejection: the offending tenant is slowed to its allotment, but it is not erroring out, and the other tenants are protected because the throttled client is not consuming more than its share of broker capacity.

#### Worked example: capping a noisy tenant at 50 MB/s

Suppose tenant `analytics-batch` runs a nightly job that, left unchecked, produces at 400 MB/s and saturates the cluster's network, causing the latency-sensitive `payments` tenant to see its produce latencies spike from 5 ms to 800 ms. We want to cap `analytics-batch` at 50 MB/s of produce bandwidth so it cannot starve the others.

```bash
# Cap the analytics-batch principal at 50 MB/s produce and 50 MB/s fetch
kafka-configs --bootstrap-server broker1:9094 --command-config admin.properties \
  --alter --add-config 'producer_byte_rate=52428800,consumer_byte_rate=52428800' \
  --entity-type users --entity-name analytics-batch

# Also cap its share of request-handler threads at 200% (2 full threads worth)
kafka-configs --bootstrap-server broker1:9094 --command-config admin.properties \
  --alter --add-config 'request_percentage=200' \
  --entity-type users --entity-name analytics-batch
```

Now walk through what happens when the nightly job kicks off and tries to push 400 MB/s. The job's producer sends batches as fast as it can. The broker accepts them but tracks the principal's rate; the moment the trailing rate crosses 50 MB/s (52,428,800 bytes), the broker starts delaying the produce responses. The producer's in-flight request window fills up because acknowledgments are coming back late, so the producer's own back-pressure kicks in and it stops sending faster than ~50 MB/s. The effective throughput of `analytics-batch` settles at its quota. The nightly job now takes eight times longer to drain its backlog — 400 MB/s of demand throttled to 50 MB/s of supply — which is exactly the intended outcome: the batch job runs slower, and the `payments` tenant keeps its 5 ms latencies because the cluster's bandwidth is no longer being consumed by analytics. The noisy neighbor has been quieted without being broken. The one thing to watch is that the batch job's *backlog* now grows during the night, so you size the quota to be generous enough that the job still finishes by morning, but tight enough that it cannot crowd out latency-sensitive traffic at peak. Quotas are a budgeting exercise, not a one-size-fits-all number.

### Namespaces and prefix conventions

Quotas handle resource contention; **namespaces** handle naming and access isolation. Kafka does not have first-class namespaces the way Pulsar does — Pulsar has a genuine tenant/namespace/topic hierarchy built into the broker, which is one of its real advantages for multi-tenant operation. Kafka instead uses a **topic-prefix convention** combined with prefixed ACLs. Each tenant gets a prefix — `teamA.`, `payments.`, `customer-1234.` — and you grant that tenant ACLs scoped to its prefix using Kafka's prefixed ACL matching:

```bash
# Grant teamA full read/write on any topic starting with "teamA."
kafka-acls --bootstrap-server broker1:9094 --command-config admin.properties \
  --add --allow-principal User:teamA-svc \
  --operation Read --operation Write --operation Describe \
  --topic teamA. --resource-pattern-type prefixed
```

This gives each tenant a private slice of the topic namespace that other tenants cannot read or write, enforced by the prefixed ACL, and it makes quota assignment, monitoring, and cost allocation natural because everything a tenant does is grouped under its prefix. The discipline is to make the prefix mandatory — a tenant can only create and use topics under its own prefix — so that the namespace boundary is enforced by authorization, not by everyone agreeing to be polite.

Pulsar's first-class multi-tenancy is worth a sentence here because it is a genuine point of differentiation covered in [the broker comparison post](/blog/software-development/message-queue/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs): if hard multi-tenancy is a primary requirement — many independent tenants, strong isolation, per-tenant quotas and namespaces as a built-in concept rather than a convention — Pulsar's model is more native than Kafka's prefix convention. RabbitMQ takes yet another approach with **virtual hosts (vhosts)**, which are fully isolated logical brokers within one physical cluster, each with its own exchanges, queues, and permissions. The right tool depends on how hard your tenancy boundary needs to be.

### Isolation beyond quotas

Quotas and namespaces address the common cases, but true hard isolation between mutually distrusting tenants — the kind you need when tenants are external customers who must never affect each other — sometimes calls for stronger measures. Connection limits cap how many connections a single principal can open, preventing a connection-exhaustion attack. Dedicated brokers or even dedicated clusters for the most sensitive or noisiest tenants provide physical isolation that no quota can match — if a tenant absolutely cannot be allowed to affect others, the cleanest answer is to not share the hardware. The general principle is that isolation comes in degrees: quotas give you soft isolation (fair sharing of a common pool), namespaces give you access isolation (you cannot see my data), and dedicated infrastructure gives you hard isolation (you cannot touch my resources at all). Match the degree to the trust level of your tenants.

## 7. Audit logging and secrets management

The layers so far prevent bad things. Audit logging and secrets management are about the bad things that get through anyway — knowing they happened, and not handing attackers the keys in the first place.

### Audit logging: who did what

An audit log records the security-relevant events on the broker: who authenticated (and who failed to), who was granted or denied an operation, who created or deleted a topic, who changed an ACL or a quota. The value of an audit log is realized at exactly two moments: during an active incident, when you need to know what an attacker touched and how far they got; and during forensics afterward, when you need to reconstruct the timeline for the post-mortem and for any legal or compliance obligation. A broker without audit logging is a broker where a breach is invisible — you may never know it happened, and if you do, you cannot tell what was taken.

The most security-relevant events to capture are the **authorization denials** and the **authentication failures**, because those are the signature of an attack in progress. A sudden spike in authentication failures from one source is a credential-stuffing or brute-force attempt. A spike in authorization denials from a legitimately authenticated principal is either a misconfiguration or a compromised credential probing for what it can reach. Kafka can log authorizer decisions (enable the authorizer logger at the appropriate level), and Confluent, Redpanda, and the cloud-managed offerings provide structured audit logs as a first-class feature. The key operational point is that audit logs must go somewhere **the broker's own operators cannot quietly edit** — shipped off-box to a separate, append-only log store — because an audit log an insider can rewrite is no audit log at all. The whole point is non-repudiation, and that requires the record to live outside the trust boundary of the people it might incriminate.

There is a cost to be honest about: audit logging at the per-request level on a high-throughput broker is expensive, both in the logging overhead and in the volume of logs produced. You do not audit every fetch on a million-message-per-second topic. The pragmatic posture is to audit the **control-plane and security events** comprehensively — authentication, authorization decisions (at least the denials), administrative operations, ACL and quota changes — and to sample or omit the high-volume data-plane operations. The signal you actually need for security lives in the control plane.

### Secrets management: never a literal in a config file

Every layer in this post depends on credentials: TLS keystore passwords, SCRAM passwords, OAuth client secrets, certificate private keys. The fastest way to undo all of it is to manage those credentials badly, and the single most common mistake is putting them as **literals in a config file checked into version control**. A keystore password in `server.properties` committed to git is a credential that lives forever in the repository history, visible to everyone with repo access and to anyone who ever clones it. The recurring "secrets leaked in a public GitHub repo" stories are this mistake at scale.

The discipline has a few rules. First, **secrets come from a secrets manager**, not from config files — HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager, or Kubernetes Secrets backed by one of these. The application fetches the credential at startup (or has it injected) rather than reading it from a static file. Kafka supports this directly through its **config provider** mechanism, which is why the TLS configuration in section 2 used `${file:...}` indirection rather than inline passwords — and in production that indirection points at a Vault provider, not a flat file:

```properties
# Reference a secret from a provider instead of writing it inline
config.providers=vault
config.providers.vault.class=io.confluent.kafka.security.config.provider.VaultConfigProvider
ssl.keystore.password=${vault:secret/kafka/broker1:keystore-password}
```

Second, **rotate credentials regularly and on compromise**, which means the system must support rotation without downtime — short-lived OAuth tokens and automated certificate issuance make this far easier than long-lived passwords, which is a strong argument for those mechanisms. Third, **scope credentials tightly**: a credential that can only do what one service needs (because of the ACLs from section 5) limits the damage when it leaks. Fourth, **scan for leaked secrets** in your repositories and CI logs with automated tooling, because the secret that leaks is usually the one someone pasted somewhere temporary and forgot. Secrets management is not glamorous, but it is the layer that determines whether all the other layers are actually protecting anything or just protecting a lock whose key is taped to the door.

## 8. The unsecured-broker catastrophe

Now we come to the recurring disaster that motivates this entire post. Every few months, security researchers announce another unsecured data store dumped to the internet — Elasticsearch clusters, MongoDB instances, Redis servers, and increasingly Kafka brokers — exposing millions of records to anyone who ran a scan. These are not sophisticated attacks. They are brokers (or databases) configured exactly as the unsecured-broker side of the figure below: bound to a public interface, accepting connections from anyone, serving plaintext data on the wire, with no authentication and no access control. The attacker's "exploit" is to connect and ask. The figure contrasts that open broker with the secured posture this post has built — the same cluster differing only in three controls, but in blast radius differing by everything.

![A before-and-after comparison of an open broker bound to all interfaces where anyone connects without credentials, traffic is plaintext and sniffable, and the full topic data is dumped to an attacker, versus a secured broker with an encrypted TLS link, proven SASL or mTLS identity, and a default-deny ACL enforcing least privilege](/imgs/blogs/securing-message-queues-tls-authz-acls-4.webp)

### Why it keeps happening

The mechanics of how a broker ends up open are depressingly consistent, and understanding them is how you avoid being the next headline. There are four recurring root causes.

The first is **insecure defaults**. Historically, many data systems shipped with no authentication enabled by default and bound to all network interfaces. The original sin of the early MongoDB and Redis breaches was exactly this: out of the box, the system listened on `0.0.0.0` with no password, optimized for a frictionless first-run experience and trusting the operator to lock it down later — which the operator, under deadline, never did. Brokers have improved here, but the lesson stands: never trust that the default configuration is secure, and explicitly verify the listener address and authentication state before exposing anything.

The second is the **trust boundary fallacy** from section 1, manifesting in production. A team stands up a broker for internal use, reasons that it lives inside the VPC so it needs no authentication, and then a single security-group rule gets widened — someone opens a port for debugging, a load balancer is misconfigured, a VPC peering connection is added — and the "internal" broker is suddenly reachable from the internet, still with no authentication because nobody ever added it. The broker did not change; the network around it did, and the broker had no defense of its own. This is why the zero-trust posture is non-negotiable: a broker that requires authentication is not exposed even when the network is misconfigured.

The third is **abandoned or forgotten infrastructure**. A test cluster spun up for a proof of concept, never decommissioned, drifting out of memory, still running, still holding a copy of production data someone loaded into it "just to test." It has no owner, no monitoring, and no security, and it sits there until a scanner finds it. The defense is infrastructure inventory and ruthless decommissioning: every broker is owned, monitored, and either secured or destroyed.

The fourth is **the disabled-for-debugging change that became permanent**. Authentication broke a deployment at 2 a.m., someone disabled it to get the system back up, filed a ticket to re-enable it, and the ticket died in the backlog. The temporary insecure state became the permanent one. The defense is to treat security configuration as something that fails the deployment if it is wrong, not something you can quietly switch off — and to alert on a broker that is accepting unauthenticated connections, so the "temporary" disable is loud and short-lived.

### The cost when it happens

The reason this is worth dwelling on is the magnitude of the consequence. When a broker is dumped, the attacker does not get one snapshot — they get the **entire retention window**, replayable from the beginning, because that is what a log broker holds. A database breach leaks the current state; a broker breach leaks the full history of events, often including the change-data-capture stream off your primary database, which means the broker breach can be a superset of the database breach. The data includes whatever flows through your event backbone: orders, payments, user actions, internal service calls, and frequently personal data subject to regulation. The consequences cascade: regulatory penalties under GDPR or CCPA, mandatory breach notification, the reputational damage of being the company in the headline, and the direct cost of the data in criminal hands. Some of these incidents have included **ransom demands** — attackers who find an open broker may delete the data and leave a note demanding payment for its return, which is a particularly cruel outcome for a broker that was the only copy of in-flight events. The asymmetry is stark: the security controls in this post are a few days of careful work, and the breach they prevent can be an existential event for the business.

## 9. A security hardening checklist

Everything above distills into an operational checklist you can run against any broker. The taxonomy figure below organizes the controls into the five branches you must cover — in transit, at rest, authentication, authorization, and tenancy — and a gap in any branch is an exposure regardless of how well you covered the others.

![A taxonomy tree of broker security controls branching into in-transit TLS for client and inter-broker traffic, at-rest volume encryption, authentication via SASL or mTLS, authorization via default-deny ACLs and RBAC, and tenancy via quotas and namespaces](/imgs/blogs/securing-message-queues-tls-authz-acls-6.webp)

Walk the tree branch by branch. For **in transit**: TLS is enabled on client listeners, hostname verification is on, and — the commonly forgotten one — inter-broker traffic is encrypted too, not just client traffic. Weak TLS versions and ciphers are disabled; only TLS 1.2 and 1.3 with modern AEAD ciphers are permitted. For **at rest**: volumes are encrypted (and so are the backups and snapshots, which is where the real gap usually hides), with message-level encryption reserved for the genuinely high-sensitivity streams. For **authentication**: every listener requires authentication, no anonymous access anywhere, with SCRAM-over-TLS as the default unless you have a reason for OAuth, Kerberos, or mTLS; the inter-broker link authenticates too. For **authorization**: the authorizer is on, `allow.everyone.if.no.acl.found` is `false`, every principal has least-privilege ACLs scoped to exactly its topics and groups, there are no wildcard grants outside of deliberately shared resources, and the superuser list is short and tightly held. For **tenancy**: quotas cap per-tenant bandwidth and request rate so no tenant can starve the cluster, namespaces or prefixes isolate tenants' topics, and connection limits prevent exhaustion attacks.

Then the cross-cutting controls that wrap the tree: audit logging captures authentication and authorization events to an off-box, append-only store; secrets come from a secrets manager and never from config literals, and they rotate; the broker is on a private network with security groups that default-deny inbound and are reviewed; and there is monitoring that alerts if the broker ever accepts an unauthenticated connection or if authentication-failure or authorization-denial rates spike. Run this checklist before any broker carries production data, and run it again periodically, because security configuration drifts — the "temporary" disable, the widened security group, the forgotten test cluster all happen between audits, not during them.

## Case studies and war stories

### The TLS that halved throughput

A payments team enabled TLS on a Kafka cluster as part of a compliance mandate, having sized the cluster's broker count from a benchmark run without TLS. Within a week of the rollout, consumer lag was climbing on the heaviest topics and produce latencies were spiking at peak. The cluster had not changed in message volume; it had simply lost its zero-copy read path. The read-heavy fan-out topics — consumed by a dozen downstream services each — were now copying every byte through user space to encrypt it, and the brokers' CPUs, previously idle while `sendfile` did the work, were saturated. The fix was partly more brokers (the cluster genuinely needed about 35% more capacity for the same workload with TLS) and partly ensuring the brokers ran on CPUs with AES-NI and used AES-GCM cipher suites so the crypto itself was cheap, leaving the copy as the dominant remaining cost. The lesson is the one from section 2: **the zero-copy tax is real, and you must budget capacity for TLS rather than enabling it on a cluster sized without it.** Benchmark with TLS on, or expect to add brokers after you turn it on.

### The wildcard ACL that leaked everything

A company enabled the Kafka authorizer and dutifully wrote ACLs, but under deadline pressure a developer granted the main application principal Read and Write on `--topic '*'` to "make it work" without chasing down every topic the service touched. The authorizer was on, the cluster looked secured, and the audit passed a cursory review. Months later that application's credentials leaked through a compromised dependency, and because the principal had wildcard access, the attacker could read every topic in the cluster — including the change-data-capture stream carrying the full customer database. The authorizer was working perfectly; it was the policy that was wrong. **Default-deny only protects you if you do not undo it with a wildcard.** The post-mortem replaced the wildcard with explicit per-topic ACLs and added a policy check in CI that rejected any ACL with a `*` topic resource outside an approved allowlist. Least privilege is not a feature you turn on; it is a discipline you enforce on every grant.

### The internal broker that wasn't

A team ran a RabbitMQ cluster for internal service communication with the default `guest`/`guest` credentials, reasoning it was inside the VPC and therefore safe — the trust boundary fallacy in its purest form. (RabbitMQ's `guest` account is, sensibly, restricted to localhost by default precisely to prevent this, but the team had relaxed that restriction during setup to allow remote management and never undid it.) A misconfigured security group, added during an unrelated networking change, exposed the management port to the internet. A scanner found it within days, logged in with the well-known default credentials, and had full administrative access to the message bus. The data exposure was bad; the fact that the attacker could *publish* messages — injecting fake events into the system — was arguably worse, because it turned a read breach into a potential integrity attack on every consumer downstream. The lessons stack up: never run default credentials, never relax a security-by-default restriction without a compensating control, and never rely on the network to be the only thing standing between an open broker and the internet.

### The noisy tenant that took down the cluster

A shared analytics platform ran many teams' workloads on one Kafka cluster with no quotas, on the theory that everyone was a good citizen. One team deployed a backfill job with a bug that produced in a tight loop, pushing several hundred megabytes per second into a topic. With no quota, that job consumed the cluster's entire network and request-handling capacity, and every other team's producers and consumers stalled — a cluster-wide outage caused by one buggy job. The incident review added per-principal byte-rate and request-rate quotas across the board, sized so that no single tenant could consume more than a defined share of cluster capacity. The runaway job, replayed against the quota'd cluster, was simply throttled to its allotment and slowly drained while everyone else kept running. **Quotas turn a cluster-wide outage into one tenant's slow night.** The cost of not having them is that the cluster's stability is only as good as the best-behaved tenant's worst day.

## When to reach for this (and when not to)

The honest answer is that **every broker carrying real data needs the core of this**, and the only question is how much of the full stack each deployment warrants. There is no deployment where "no authentication on a broker holding production data" is the right call, and the unsecured-broker catastrophe is precisely what happens when teams convince themselves otherwise. So the baseline — TLS in transit, authentication on every listener, default-deny ACLs, encrypted volumes — is non-negotiable for any production broker. Skipping it is not a tradeoff; it is an unmitigated risk waiting to become a headline.

Where judgment enters is the depth of the upper layers. **Multi-tenancy controls** (quotas, namespaces, hard isolation) are essential when a cluster is genuinely shared across teams or customers, and largely unnecessary when a cluster serves a single application — though even single-application clusters benefit from quotas as a guardrail against runaway loops. **Message-level encryption at rest** is warranted only for genuinely high-sensitivity data where the broker must be a zero-knowledge pipe; for everything else, volume encryption is the right cost-benefit. **Comprehensive audit logging** scales with your compliance obligations: a fintech or healthcare deployment audits everything in the control plane and keeps it for years, while an internal logging pipeline might reasonably audit only administrative and security events. **Kerberos and full RBAC** make sense at enterprise scale with existing identity infrastructure and hundreds of services, and are overkill for a five-service startup where SCRAM and a few dozen ACLs are entirely sufficient.

The one place to push back hard is the temptation to defer security to "after we ship." Security controls are far cheaper to design in than to retrofit — adding authentication to a running cluster that a dozen services connect to anonymously is a delicate, coordinated migration, while building it in from day one is a config file. The brokers that end up dumped on the internet are almost never the ones where someone decided security was not worth it; they are the ones where someone decided it could wait, and then it waited forever. Build the baseline in from the start, and add the upper layers as your tenancy and compliance needs grow.

## Key takeaways

- **Security is a stack of independent layers, not a single switch.** TLS, authentication, authorization, quotas, and audit each guard a distinct failure; implementing one and assuming it covers another is the most common mistake. TLS without authentication encrypts a conversation with an attacker.
- **Assume the network is hostile.** The trust boundary fallacy — "it is internal, so it is safe" — is the root cause of most broker breaches. Require authentication from every client regardless of where it connects, and encrypt traffic even on trusted links.
- **TLS defeats zero-copy, so budget for it.** Enabling TLS typically costs 20–40% of read throughput because the broker can no longer use `sendfile` and must copy and encrypt every byte. Benchmark with TLS on; do not size a cluster without it and then turn it on.
- **SCRAM over TLS is the sensible default for authentication.** Reach for Kerberos if you already run Active Directory, OAUTHBEARER if you already run an OAuth identity provider, and mTLS if you already run PKI or a service mesh. Choose by your existing identity infrastructure, not by which is "most secure."
- **Default-deny is the whole game in authorization.** Set `allow.everyone.if.no.acl.found=false`, grant each principal least-privilege ACLs scoped to exactly its topics and groups, and never use wildcard grants outside deliberately shared resources. A leaked credential's blast radius is exactly its ACLs.
- **Quotas turn a cluster-wide outage into one tenant's slow night.** Cap per-tenant bandwidth and request rate; the broker throttles by delaying responses, so the offender is slowed gracefully while everyone else is protected. In a shared cluster, quotas are not optional.
- **Encryption at rest is mostly about the backups.** Volume encryption is the right default and is nearly free; the real gap is unencrypted backups and snapshots of an otherwise encrypted cluster. Extend the at-rest posture to your entire disaster-recovery path.
- **Secrets come from a secrets manager, never from config literals.** A keystore password committed to git is a credential that lives forever in the repository history. Use a config provider backed by Vault or a cloud secrets manager, rotate regularly, and scope credentials tightly.
- **The unsecured-broker catastrophe is a configuration failure, not a sophisticated attack.** It happens through insecure defaults, the trust boundary fallacy, abandoned infrastructure, and disabled-for-debugging changes that became permanent. Alert on any broker accepting unauthenticated connections.

## Further reading

- [Choosing a message broker: Kafka, RabbitMQ, Pulsar, NATS, SQS](/blog/software-development/message-queue/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs) — security capabilities differ sharply across brokers; Pulsar's native multi-tenancy and RabbitMQ's vhosts are real differentiators that belong in the selection decision.
- [RabbitMQ Deep Dive Part 1: AMQP exchanges, bindings, and routing](/blog/software-development/message-queue/rabbitmq-amqp-exchanges-bindings-routing) — the exchange-and-binding model shapes how you scope RabbitMQ permissions and vhost isolation.
- [Durability and disaster recovery for message queues](/blog/software-development/message-queue/durability-and-disaster-recovery-for-message-queues) — your backups inherit the encryption-at-rest posture set here, and an unencrypted backup of an encrypted cluster is the weakest link.
- [Broker I/O optimization: zero-copy, the page cache, and tiered storage](/blog/software-development/message-queue/broker-io-optimization-zero-copy-tiered-storage) — the deep mechanics of the `sendfile` read path that TLS forces you to give up.
- [Apache Kafka Security documentation](https://kafka.apache.org/documentation/#security) — the authoritative reference for listeners, SASL mechanisms, the authorizer, and ACL commands.
- [RFC 8446: The Transport Layer Security (TLS) Protocol Version 1.3](https://datatracker.ietf.org/doc/html/rfc8446) — the handshake and cipher specification underlying everything in section 2.
- [The OWASP Cheat Sheet Series](https://cheatsheetseries.owasp.org/) — practical guidance on TLS configuration, secrets management, and authentication that applies directly to broker hardening.
