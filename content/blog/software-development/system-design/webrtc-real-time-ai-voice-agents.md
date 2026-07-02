---
title: "WebRTC for Real-Time AI: Protocol Internals, Voice Agents, and Production Troubleshooting"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "A protocol-first deep dive into WebRTC — signaling, ICE/STUN/TURN, DTLS-SRTP, RTP/RTCP, and Google Congestion Control — applied to real-time voice AI, with runnable code, a latency budget, ten production war stories, and a troubleshooting playbook."
tags:
  [
    "webrtc",
    "real-time-communication",
    "voice-agents",
    "ice",
    "stun-turn",
    "dtls-srtp",
    "rtp",
    "congestion-control",
    "aiortc",
    "system-design",
    "low-latency",
    "ai-infrastructure",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 50
---

The first time most engineers touch WebRTC, they treat it like every other networking API they have shipped: open a connection, send bytes, handle a callback. Then the demo works flawlessly on the office Wi-Fi, works on the reviewer's laptop, works in the staging cluster — and falls apart the moment a real user opens it from a hotel network, a corporate VPN, or an LTE hotspot in a parking garage. Suddenly there is no audio, or audio in one direction only, or a two-second delay that turns a conversation into a walkie-talkie exchange. The API did not lie to you. WebRTC is simply not "a socket." It is a peer-to-peer, UDP-first, self-encrypting, self-rate-adapting media transport that has to punch through the entire adversarial reality of the public internet — and it hands you a deceptively small JavaScript surface on top of an enormous amount of machinery.

That machinery is exactly what makes WebRTC the transport of choice for the current wave of real-time AI products. When you build a voice agent — a thing a human talks to and that talks back within a few hundred milliseconds — you are not really building an "AI feature." You are building a low-latency, bidirectional, loss-tolerant media pipeline that happens to have a large language model wired into the middle of it. Get the model right and the transport wrong, and users will describe your beautifully-tuned agent as "laggy," "robotic," or "it keeps talking over me." The transport *is* the product experience. OpenAI shipped its Realtime API with WebRTC as the recommended browser path for precisely this reason: for interactive voice, WebRTC gives more consistent latency and resilience than a WebSocket stream, and it does the audio plumbing so you do not have to.

![WebRTC has two planes: a signaling plane you build and a media plane the browser runs](/imgs/blogs/webrtc-real-time-ai-voice-agents-1.webp)

The diagram above is the mental model for this entire article, and internalizing it will save you more debugging hours than any single API trick. WebRTC has **two planes**. The **signaling plane** — how two peers find each other and agree on what they will send — is *not part of WebRTC at all*. The spec deliberately leaves it to you: you exchange session descriptions and network candidates over whatever channel you like (a WebSocket, an HTTPS POST, a carrier pigeon). The **media plane** — NAT traversal, encryption, packetization, retransmission, congestion control, jitter buffering — is what the browser and the STUN/TURN servers run *for* you, and it is where all the hard, interesting, and failure-prone engineering lives. Nearly every production incident I have seen resolves to a misunderstanding of which plane owns a problem.

This is a protocol-first tour. We will walk down the media plane layer by layer — signaling, ICE, DTLS-SRTP, RTP and the jitter buffer, congestion control, data channels — and at each layer connect the mechanism to what it means for a voice agent's latency, robustness, and cost. Then we build a server-side agent with real Python, budget its latency stage by stage, handle the genuinely hard problem of barge-in, and close with a troubleshooting playbook and ten production war stories. If you want a batteries-included platform that wraps all of this, [LiveKit](/blog/software-development/system-design/livekit-real-time-communication) is the reference implementation; this post is about the protocol underneath it, so that when the platform's abstractions leak — and they will — you know what you are looking at.

## Why WebRTC feels like a different animal

Before the layer-by-layer tour, it is worth being explicit about *why* WebRTC violates the intuitions you built shipping REST APIs, gRPC services, and WebSocket streams. Every row in the table below is a place I have watched a competent engineer lose a day.

| You assume… | The naive mental model | The WebRTC reality |
| --- | --- | --- |
| The client connects to your server | Client → your endpoint, TLS, done | Two peers connect to *each other*; your server only brokers the introduction, then usually steps out of the media path |
| A connection is a connection | One TCP socket, ordered, reliable | Media rides **UDP** by default; packets are lost, reordered, and duplicated — and that is the *correct* behavior for real-time |
| You provide the address | `wss://api.example.com` and go | Neither peer knows its own reachable address; **ICE** discovers it at runtime, and it changes per network |
| Reliability is free | TCP retransmits, so I never lose data | Retransmitting late audio is *worse* than losing it; WebRTC chooses concealment and FEC over blind retransmission |
| Bandwidth is fixed | I send at my bitrate | **Google Congestion Control** constantly re-estimates the path and moves your bitrate up and down without asking you |
| Encryption is a checkbox | Enable TLS if you care | There is **no unencrypted WebRTC**; DTLS-SRTP is mandatory and non-negotiable |
| Firewalls allow outbound | Port 443 is open, I'm fine | UDP is often blocked entirely; without a **TURN** relay on TCP/443, a meaningful fraction of real users cannot connect at all |
| It works in the demo | Ship it | The demo ran peer-to-peer on one LAN; production is symmetric NATs, VPNs, and carrier-grade NAT that only relaying survives |

> WebRTC is the only widely-deployed protocol where *losing data on purpose* and *changing your own send rate without being asked* are not bugs but the entire point. If you fight those two facts, you will lose.

Hold that table in mind. The rest of the article is essentially a detailed defense of each "reality" column, and a guide to building AI products that lean into it instead of fighting it.

## 1. Signaling: the handshake WebRTC deliberately leaves to you {#signaling}

The single most common source of confusion for newcomers is discovering that **WebRTC has no built-in way for two peers to find each other.** There is no `connect(host, port)`. Instead, each peer produces a **Session Description Protocol (SDP)** blob describing what it wants to send and receive — codecs, media directions, encryption fingerprints, network candidates — and the two peers must exchange these blobs through a channel you build. That channel is *signaling*, and it is your responsibility.

![Offer/answer with trickle ICE is a multi-step handshake WebRTC leaves you to transport](/imgs/blogs/webrtc-real-time-ai-voice-agents-2.webp)

The flow is the classic **offer/answer** exchange, shown above. The initiating peer calls `createOffer()` to produce an SDP offer, calls `setLocalDescription()` (which kicks off ICE candidate gathering — more on that next section), and ships the offer through your signaling channel. The remote peer calls `setRemoteDescription()` with that offer, `createAnswer()` to produce its own SDP, `setLocalDescription()`, and ships the answer back. As candidates are discovered asynchronously, each peer **trickles** them across the same channel rather than waiting for gathering to finish. Once both descriptions are set and at least one candidate pair connects, media flows.

Here is the minimal browser side, using real APIs, for a peer that captures the microphone and connects:

```javascript
// The signaling transport is yours — here a plain WebSocket to your server.
const signaling = new WebSocket("wss://api.example.com/rtc");

const pc = new RTCPeerConnection({
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    {
      urls: ["turn:turn.example.com:3478?transport=udp",
             "turns:turn.example.com:443?transport=tcp"],
      username: "ephemeral-user",
      credential: "ephemeral-secret", // short-lived, from your auth server
    },
  ],
  // Pool a couple of ICE candidates before the first offer to shave setup time.
  iceCandidatePoolSize: 2,
});

// Capture mic audio and add it to the connection.
const stream = await navigator.mediaDevices.getUserMedia({
  audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  video: false,
});
for (const track of stream.getTracks()) pc.addTrack(track, stream);

// Trickle our candidates to the peer as they arrive.
pc.onicecandidate = ({ candidate }) => {
  if (candidate) signaling.send(JSON.stringify({ type: "ice", candidate }));
};

// Play whatever the remote peer sends us.
pc.ontrack = ({ streams: [remote] }) => {
  document.getElementById("audio").srcObject = remote;
};

// Initiate.
const offer = await pc.createOffer();
await pc.setLocalDescription(offer);
signaling.send(JSON.stringify({ type: "offer", sdp: offer.sdp }));

signaling.onmessage = async (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "answer") {
    await pc.setRemoteDescription({ type: "answer", sdp: msg.sdp });
  } else if (msg.type === "ice") {
    await pc.addIceCandidate(msg.candidate);
  }
};
```

Notice what the API does *not* do: it never mentions IP addresses, ports, or NAT. You describe *intent* (capture audio, use these ICE servers), hand the SDP to your transport, and the browser handles the rest. The `getUserMedia` audio constraints — `echoCancellation`, `noiseSuppression`, `autoGainControl` — are quietly doing heavy lifting we will return to when we discuss barge-in; for a voice agent they are not optional niceties, they are load-bearing.

### What SDP actually contains

Reading an SDP blob once, slowly, demystifies half of WebRTC. A trimmed audio offer looks like this:

```sdp
v=0
o=- 46117317 2 IN IP4 127.0.0.1
s=-
t=0 0
m=audio 9 UDP/TLS/RTP/SAVPF 111 63 9 0 8
c=IN IP4 0.0.0.0
a=rtcp-mux
a=ice-ufrag:F7gI
a=ice-pwd:x9cml/YzichV2+XlhiM9Rte6
a=fingerprint:sha-256 D2:FA:0E:C3:22:59:5E:14:95:69:92:3D:13:B4:84:24:...
a=setup:actpass
a=mid:0
a=sendrecv
a=rtpmap:111 opus/48000/2
a=fmtp:111 minptime=10;useinbandfec=1
```

Every line is load-bearing. `m=audio 9 UDP/TLS/RTP/SAVPF` declares an audio stream over the DTLS-SRTP profile. `a=ice-ufrag` and `a=ice-pwd` are the ICE credentials the peers use to authenticate connectivity checks. `a=fingerprint:sha-256 …` is the hash of the peer's DTLS certificate — this is the line that binds the signaling plane to the media plane, and we will see why it matters for security. `a=rtpmap:111 opus/48000/2` says payload type 111 is stereo 48 kHz Opus, and `a=fmtp:111 …useinbandfec=1` enables Opus in-band forward error correction, a single toggle that will show up in a case study about robotic audio. The SDP is a contract; both peers must agree on it before a single media packet moves.

### The OpenAI Realtime twist: SDP over HTTPS

Modern AI voice APIs collapse the signaling round-trip into a single HTTPS request, which is worth seeing because it shows how flexible "signaling is yours" really is. With OpenAI's Realtime API, the browser mints its local offer and simply **POSTs the raw SDP** to the API with an ephemeral token, and the response body *is* the answer SDP:

```javascript
// 1) Your server mints a short-lived token so the real API key never ships to the browser.
const { client_secret } = await fetch("/api/realtime-token").then((r) => r.json());

const pc = new RTCPeerConnection();
pc.ontrack = (e) => (audioEl.srcObject = e.streams[0]);
pc.addTrack((await navigator.mediaDevices.getUserMedia({ audio: true })).getAudioTracks()[0]);

// A data channel carries events (transcripts, function calls) alongside the audio.
const events = pc.createDataChannel("oai-events");

const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

// 2) POST the SDP offer directly; the answer comes back in the HTTP body.
const answer = await fetch("https://api.openai.com/v1/realtime/calls", {
  method: "POST",
  body: offer.sdp,
  headers: { Authorization: `Bearer ${client_secret}`, "Content-Type": "application/sdp" },
});
await pc.setRemoteDescription({ type: "answer", sdp: await answer.text() });
```

There is no WebSocket, no long-lived signaling server, no trickle loop in the application code — the whole handshake is one POST because the server already knows all its candidates. The important architectural detail is the **ephemeral token**: your backend authenticates with the real API key and hands the browser a short-lived secret, so the key never touches the client. Everything after the handshake — audio in both directions, plus a **data channel** for events — is standard WebRTC, which is why the WebRTC path "just works" without granular audio-buffer handling on your side.

### Second-order gotcha: glare and perfect negotiation

The moment either peer can initiate renegotiation — adding a screen share, switching cameras, an agent that starts sending video mid-call — you can hit **glare**: both sides send an offer simultaneously and the state machines deadlock. The fix is the **perfect negotiation** pattern, where one peer is designated *polite* and yields on collision:

```javascript
let makingOffer = false, ignoreOffer = false;
const polite = false; // one peer true, the other false — decided out of band

pc.onnegotiationneeded = async () => {
  try {
    makingOffer = true;
    await pc.setLocalDescription(); // implicit createOffer in modern browsers
    signaling.send({ description: pc.localDescription });
  } finally { makingOffer = false; }
};

signaling.onmessage = async ({ description, candidate }) => {
  if (description) {
    const collision = description.type === "offer" && (makingOffer || pc.signalingState !== "stable");
    ignoreOffer = !polite && collision;      // impolite peer ignores the colliding offer
    if (ignoreOffer) return;
    await pc.setRemoteDescription(description);
    if (description.type === "offer") {
      await pc.setLocalDescription();
      signaling.send({ description: pc.localDescription });
    }
  } else if (candidate) {
    try { await pc.addIceCandidate(candidate); }
    catch (e) { if (!ignoreOffer) throw e; }   // swallow candidates for an ignored offer
  }
};
```

For a one-way "browser talks to an AI server" call you can often avoid renegotiation entirely by declaring all media up front. But the instant your product grows a "share your screen with the assistant" button, glare is waiting, and perfect negotiation is the only pattern that reliably survives it.

## 2. ICE, STUN, and TURN: getting through NAT {#ice}

If signaling is where beginners get confused, **ICE is where production breaks.** The Interactive Connectivity Establishment protocol is WebRTC's answer to a brutal fact: neither peer knows an address at which it can actually be reached. Your laptop has a private address like `192.168.1.20` that means nothing on the public internet; it sits behind a NAT that rewrites addresses and ports, often unpredictably. ICE's job is to discover every address at which a peer *might* be reachable, try them all, and keep the best pair that works.

![ICE gathers host, server-reflexive, and relay candidates and picks the lowest-latency working pair](/imgs/blogs/webrtc-real-time-ai-voice-agents-3.webp)

As the figure shows, ICE gathers three kinds of **candidates**:

- **Host candidates** — the peer's own local IP:port pairs (`192.168.1.20:54321`). Fastest and free, but only work if the peers are on the same network.
- **Server-reflexive (srflx) candidates** — the peer's *public* IP:port as seen from outside, discovered by asking a **STUN** server "what address did my packet appear to come from?" STUN is cheap and stateless; it just reflects your mapped address back. Most NAT crossings succeed on a srflx-to-srflx pair.
- **Relay candidates** — an IP:port on a **TURN** server that agrees to forward all media on your behalf. TURN is the fallback of last resort: it always works (both peers just talk to the relay), but it adds a hop of latency and it costs real money because every byte of media transits your server.

Each peer gathers all its candidates, trickles them to the other side, and then both run **connectivity checks** — STUN binding requests across every plausible candidate pair — to find which pairs actually pass traffic. The pairs are prioritized (host > srflx > relay, roughly), the best working pair is **nominated**, and media starts flowing on it. The whole process is designed to be greedy and parallel: try everything at once, use the first good thing.

| Candidate type | How it's found | Typical added latency | Success across NAT | Cost to you |
| --- | --- | --- | --- | --- |
| Host | Local interfaces | ~0 ms | Same LAN only | Free |
| Server-reflexive | STUN reflection | ~0 ms (direct path) | ~80–90% of NATs | Negligible (STUN is stateless) |
| Relay (TURN/UDP) | TURN allocation | +10–40 ms (extra hop) | ~100% | Bandwidth × minutes, all media |
| Relay (TURN/TCP or TLS-443) | TURN over TCP/443 | +30–100 ms | Survives UDP-blocking firewalls | Highest (TCP head-of-line + relay) |

The configuration you pass to `RTCPeerConnection` is just the list of STUN and TURN servers to use:

```javascript
const pc = new RTCPeerConnection({
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    {
      urls: [
        "turn:turn.example.com:3478?transport=udp",   // preferred relay
        "turn:turn.example.com:3478?transport=tcp",    // survives UDP block
        "turns:turn.example.com:443?transport=tcp",     // survives everything short of DPI
      ],
      username: "1719900000:alice",   // time-limited, HMAC'd credential
      credential: "base64hmac...",
    },
  ],
  iceTransportPolicy: "all", // set "relay" to *force* TURN — useful for testing and locked-down networks
});
```

### The rule nobody wants to hear: you always need TURN

The most expensive lesson in WebRTC is that **STUN is not enough for production.** On a friendly home NAT, srflx candidates connect the majority of the time. But symmetric NATs — common on corporate networks and universal on carrier-grade NAT used by mobile carriers — assign a *different* external port for every destination, so the port a STUN server reflects back is useless for talking to a peer. Add firewalls that block UDP entirely, and you get a population of users for whom *no direct path exists at all.* Industry measurements have long put the fraction of connections that require relaying somewhere in the **8–20% range** depending on the user population; for enterprise or mobile-heavy products it skews higher. If you skip TURN to save money, you are shipping a product that silently fails for one in five or ten users, and those users cannot tell you why — they just see "no audio."

The `turns:` entry on port 443 over TLS is the single most important line in a production ICE config. It makes relayed media look like ordinary HTTPS traffic, which is the only thing that survives the strictest corporate firewalls and captive portals. It is slow and expensive, so ICE will only fall back to it when nothing better works — but when it is the difference between a working call and a dead one, latency is the right trade.

You can stand up your own relay with **coturn**, the de facto open-source TURN server:

```bash
# /etc/turnserver.conf — a production-ish coturn config
listening-port=3478
tls-listening-port=443
fingerprint
use-auth-secret
static-auth-secret=YOUR_SHARED_SECRET      # HMAC time-limited creds, not static passwords
realm=turn.example.com
cert=/etc/letsencrypt/live/turn.example.com/fullchain.pem
pkey=/etc/letsencrypt/live/turn.example.com/privkey.pem
min-port=49152
max-port=65535
no-cli
# Lock down relaying so your TURN box can't be abused as an open proxy:
no-multicast-peers
denied-peer-ip=10.0.0.0-10.255.255.255
denied-peer-ip=169.254.0.0-169.254.255.255
```

And you can — and should — verify it before you trust it, both with the bundled client and with the browser-based trickle-ICE tester:

```bash
# Prove the relay actually allocates and forwards over UDP:
turnutils_uclient -T -u alice -w YOUR_SHARED_SECRET turn.example.com

# Then in a browser, open the WebRTC samples "Trickle ICE" page, paste your TURN
# URL + credential, and confirm a candidate of type "relay" appears. If it does
# not, your ICE config is a placebo — clients will silently fall back to failing.
```

> If you have never watched your own connection fall back to `relay` on a hostile network, you do not yet know whether your TURN setup works. "It connected in the demo" tells you nothing about the 15% of users who need the relay.

### Second-order gotcha: TURN credentials and cost

Never ship static TURN passwords to the browser — they will be scraped and your relay will become someone else's free proxy. Use coturn's `use-auth-secret` mode, where your backend computes a time-limited HMAC credential (`username = expiry-timestamp`, `credential = HMAC-SHA1(secret, username)`) that the client uses for a few minutes and that cannot be replayed later. And watch the cost: because relayed media transits your server both ways, a single relayed one-hour voice call at Opus bitrates is cheap, but a relayed 1080p video call is not, and a fleet of them will dominate your egress bill. This is why region-local TURN placement matters — a relay in the user's region adds a few milliseconds, while a relay across an ocean adds a hundred, and that shows up directly in your voice agent's response latency.

## 3. DTLS-SRTP: encryption is mandatory, and that is a feature {#dtls-srtp}

There is no such thing as unencrypted WebRTC. You cannot turn encryption off; the spec forbids it and browsers enforce it. Media is carried in **SRTP** (Secure RTP), and the keys for SRTP are negotiated by a **DTLS** (Datagram TLS) handshake that runs *over the ICE-selected path* once connectivity is established. This is called **DTLS-SRTP**, and understanding the layering explains a whole class of "it connects but there is no audio" bugs.

![There is no unencrypted WebRTC: a DTLS handshake over the ICE path keys the SRTP carrying every frame](/imgs/blogs/webrtc-real-time-ai-voice-agents-4.webp)

The stack in the figure reads bottom-up. At the bottom is the **UDP transport** (or the TCP/TLS-443 fallback that TURN provides). On top of that sits the **ICE-selected path** — the nominated candidate pair from the previous section. Once ICE picks a path, a **DTLS handshake** runs across it: the two peers exchange certificates and derive a shared secret, exactly like TLS but over datagrams. The keys from that handshake are then exported to key the **SRTP/SRTCP** streams that carry the actual encrypted audio and video frames. Every media packet, in every WebRTC session, is encrypted with keys that no signaling server ever saw.

### Where the two planes finally connect

Here is the elegant part, and the reason the `a=fingerprint` line in the SDP matters so much. During DTLS, each peer presents a self-signed certificate. How does peer A know the certificate peer B presents is really B's, and not an attacker who sat in the middle of the signaling channel? Because the **SDP carried the SHA-256 fingerprint of the certificate**, and A verifies that the certificate presented in the DTLS handshake hashes to the fingerprint it received in the SDP. This is the mechanism that binds the signaling plane to the media plane: as long as your signaling channel has integrity (which is why it must be HTTPS/WSS), a man-in-the-middle on the media path cannot impersonate either peer, because they cannot produce a certificate matching the fingerprint. It is a beautifully economical design — the signaling plane you built vouches for the media plane the browser runs.

The practical consequences for debugging:

- **A DTLS handshake failure looks exactly like "connected but silent."** ICE succeeds, `iceConnectionState` reaches `connected`, and then… nothing, because DTLS never completed and SRTP was never keyed. When you see ICE connected but no media, suspect DTLS.
- **Clock skew breaks DTLS.** Self-signed certificates have validity windows. If a device's clock is badly wrong — dead phone that just booted, a misconfigured kiosk — the certificate can be "not yet valid" and the handshake fails. This is a real and maddening field failure; we will see it in the case studies.
- **`a=setup:actpass` decides who initiates the handshake.** The offerer says `actpass` (I can be client or server); the answerer picks `active` (I'll initiate) or `passive`. A mismatch here, usually from an SDP-munging proxy, stalls the handshake.

For most application engineers, DTLS-SRTP is something you should be grateful you never have to configure — but you must recognize its failure signature, because "ICE connected, DTLS failed" is one-way-silent in a way that no amount of staring at ICE candidates will explain.

## 4. RTP, RTCP, and the jitter buffer: making UDP sound like a phone call {#rtp}

Now media is flowing, encrypted, over a working path. But it is flowing over **UDP**, which means packets arrive late, out of order, duplicated, or not at all. Turning that mess back into smooth audio is the job of the **RTP** family of protocols and the **jitter buffer**, and this is where WebRTC's "lose data on purpose" philosophy becomes concrete.

Media travels in **RTP** packets. Each RTP header carries three fields that do almost all the work: a **sequence number** (so the receiver can detect loss and reorder), a **timestamp** (the sampling instant, so the receiver can reconstruct timing regardless of arrival jitter), and an **SSRC** (a synchronization source ID that identifies which stream the packet belongs to). Alongside RTP runs **RTCP**, the control channel, where receivers report back what they are seeing — packet loss, jitter, round-trip time — and request repairs. RTCP feedback messages like **NACK** (I'm missing sequence number N, please resend), **PLI** (Picture Loss Indication — send me a fresh keyframe), and **REMB/TWCC** (bandwidth feedback) are the receiver's voice in a constant negotiation with the sender.

The receiver does not play packets the instant they arrive — that would produce audio that speeds up and stutters with every fluctuation in network delay. Instead it feeds them into a **jitter buffer**: a small, adaptive holding area that absorbs the variance in arrival time and releases packets to the decoder at a steady cadence.

<figure class="blog-anim">
<svg viewBox="0 0 780 300" role="img" aria-label="RTP packets arrive from the network with jitter and loss, pass through the jitter buffer, and leave as a steady playout stream; one lost packet is repaired mid-flight." style="width:100%;height:auto;max-width:840px">
<style>
.jb-zone{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.jb-buf{fill:none;stroke:var(--accent,#6366f1);stroke-width:2.5;stroke-dasharray:6 5}
.jb-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.jb-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.jb-pkt{fill:var(--accent,#6366f1)}
.jb-num{font:700 14px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
@keyframes jb-flow{0%{transform:translateX(0);opacity:0}9%{opacity:1}91%{opacity:1}100%{transform:translateX(620px);opacity:0}}
@keyframes jb-heal{0%,42%{fill:#ef4444}56%,100%{fill:var(--accent,#6366f1)}}
.jb-p{animation:jb-flow 8s linear infinite}
.jb-p2{animation-delay:1.1s}
.jb-p3{animation-delay:3.4s}
.jb-p4{animation-delay:4.3s}
.jb-fix rect{animation:jb-heal 8s linear infinite}
.jb-fix{animation:jb-flow 8s linear infinite;animation-delay:2.2s}
@media (prefers-reduced-motion:reduce){.jb-p,.jb-fix{animation:none;opacity:1}.jb-fix rect{animation:none;fill:var(--accent,#6366f1)}}
</style>
<text class="jb-lbl" x="90" y="40">network</text>
<text class="jb-sub" x="90" y="62">jittery + lossy</text>
<text class="jb-lbl" x="390" y="40">jitter buffer</text>
<text class="jb-sub" x="390" y="62">reorder + wait + repair</text>
<text class="jb-lbl" x="690" y="40">playout</text>
<text class="jb-sub" x="690" y="62">steady 20 ms</text>
<rect class="jb-zone" x="30" y="90" width="120" height="150" rx="10"/>
<rect class="jb-buf" x="300" y="90" width="180" height="150" rx="10"/>
<rect class="jb-zone" x="630" y="90" width="120" height="150" rx="10"/>
<g class="jb-p"><rect class="jb-pkt" x="40" y="150" width="44" height="44" rx="8"/><text class="jb-num" x="62" y="178">1</text></g>
<g class="jb-p jb-p2"><rect class="jb-pkt" x="40" y="150" width="44" height="44" rx="8"/><text class="jb-num" x="62" y="178">2</text></g>
<g class="jb-fix"><rect class="jb-pkt" x="40" y="150" width="44" height="44" rx="8"/><text class="jb-num" x="62" y="178">3</text></g>
<g class="jb-p jb-p3"><rect class="jb-pkt" x="40" y="150" width="44" height="44" rx="8"/><text class="jb-num" x="62" y="178">4</text></g>
<g class="jb-p jb-p4"><rect class="jb-pkt" x="40" y="150" width="44" height="44" rx="8"/><text class="jb-num" x="62" y="178">5</text></g>
<text class="jb-sub" x="390" y="268">packet 3 lost, rebuilt by NACK / FEC / PLC before playout</text>
</svg>
<figcaption>The jitter buffer absorbs irregular arrival timing and repairs loss, so the decoder sees a smooth, in-order stream.</figcaption>
</figure>
The animation captures the essential trade the jitter buffer makes on every call: it deliberately adds a little latency so it can smooth out timing variance and reorder late packets. A **deeper buffer** tolerates more jitter but adds delay; a **shallower buffer** is snappier but stutters on bursty networks. Modern implementations size it adaptively from measured jitter, typically holding tens of milliseconds of audio. For a voice agent, this buffer is a direct, unavoidable line item in your latency budget — and, as a later case study shows, a buffer that grows unboundedly under bursty loss is a classic cause of mysterious, creeping delay.

### Repair without retransmission

When a packet is genuinely lost, WebRTC has a menu of repair strategies, and the choice among them is governed by one variable: **round-trip time**. This is where "retransmit everything like TCP" is exactly the wrong instinct.

| Strategy | Mechanism | Cost | When it wins | When it fails |
| --- | --- | --- | --- | --- |
| **NACK + retransmit** | Receiver asks sender to resend a lost packet | One extra RTT of delay, minimal bandwidth | Low-RTT paths where the resend still arrives before playout | High-RTT paths — the resend arrives too late to use |
| **FEC (RED / Opus in-band)** | Send redundant data so loss is recoverable without a round trip | Extra bandwidth on every packet | Any RTT; bursty loss up to the redundancy level | Wastes bandwidth when the network is clean |
| **PLC (packet-loss concealment)** | Decoder synthesizes plausible audio for the gap | Free, no network cost | Isolated small gaps | Audible artifacts on long or frequent gaps |
| **DTX (discontinuous transmission)** | Stop sending during silence, send comfort noise | Saves bandwidth | Reduces load so real speech has headroom | Not a repair — a bandwidth optimization |

For **audio**, the workhorse is **Opus in-band FEC** plus **PLC**, because voice cannot tolerate the extra RTT of waiting for a NACK retransmit — by the time the resend arrives, the moment for that 20 ms of speech has passed and playing it late is worse than concealing it. That single SDP toggle from the signaling section — `a=fmtp:111 …useinbandfec=1` — is what lets the Opus decoder reconstruct a lost packet from redundant data carried in the *next* packet, no round trip required. For **video**, NACK and PLI dominate because a lost keyframe is catastrophic (the picture freezes until a fresh one arrives) and worth an RTT to repair. A voice agent is almost pure audio, so your resilience story is Opus FEC + DTX + a well-sized jitter buffer — and if you disable FEC to "save bandwidth," you will hear it the first time a user is on a lossy network.

## 5. Congestion control: GCC and transport-wide feedback {#congestion}

Here is the WebRTC reality that surprises even experienced engineers the most: **you do not control your own send bitrate.** WebRTC does, continuously, on the fly, based on a live estimate of how much the network path can carry. The algorithm that does this in every Chromium-based browser and most production media servers is **Google Congestion Control (GCC)**, and it is one of the most consequential and least-understood pieces of the stack.

![GCC runs a delay-based and a loss-based estimator off TWCC feedback and sends at the lower target](/imgs/blogs/webrtc-real-time-ai-voice-agents-6.webp)

The control loop in the figure runs constantly for the life of the call. The **sender** emits RTP packets across the **network path**. The **receiver** logs the precise arrival time of every packet and, via **Transport-Wide Congestion Control (TWCC)** feedback, reports those arrival times back to the sender. The sender feeds that feedback into two estimators running in parallel:

- The **delay-based estimator** watches the *trend* in one-way delay. If packets are starting to arrive later and later relative to when they were sent, the queue on the bottleneck link is filling — congestion is building *before* any packet is actually dropped. This is the early-warning system, and it is why WebRTC can back off gracefully instead of slamming into loss.
- The **loss-based estimator** watches the packet loss rate. The classic thresholds: above roughly **10% loss**, cut the bitrate; below **2%**, cautiously increase it; in between, hold. Loss is the confirmation signal — by the time you are losing packets, the queue already overflowed.

The sender then sends at the **minimum of the two estimates** — the lower, more conservative target wins — and reconfigures its encoder to match. When the estimate rises, GCC probes upward by briefly sending faster to see if the extra capacity is real. The whole thing is a feedback controller steering your bitrate up and down, many times a second, entirely without your involvement.

Why does an audio-only voice agent care about a mechanism built for adaptive video? Three reasons. First, **the delay-based estimator is your friend** — it detects the bufferbloat that would otherwise inflate your latency, and it can tell your application, via `getStats()`, that the path is degrading *before* users hear it. Second, **bandwidth probing and TWCC feedback are why WebRTC audio stays intelligible on a flaky LTE connection** where a naive constant-bitrate stream would either overflow the link or waste half of it. Third, if you ever add video (a talking-head avatar, screen share, a camera feed to a vision model), GCC is suddenly load-bearing for your product's quality, and you need to understand that audio and video share one bandwidth budget that GCC allocates. WebRTC's congestion control is a distant cousin of the [backpressure and rate-limiting](/blog/software-development/system-design/rate-limiting-and-backpressure) problems you solve in distributed systems — the difference is it happens on a 10-millisecond timescale and the "queue" is a cell tower's radio buffer you cannot see.

> The IETF has standardized alternatives — SCReAM, NADA, and the newer RFC 8888 congestion-control feedback with ECN/L4S — but in practice, in 2026, if you are debugging WebRTC bandwidth behavior in a browser, you are debugging GCC. Learn its two estimators and the min() rule and most of its behavior stops being mysterious.

### Second-order: probing, priorities, and audio-only reality

Two GCC behaviors surprise people building audio agents. First, **bandwidth probing.** When the delay-based estimator thinks capacity has grown, GCC does not just raise the target and hope — it sends a short burst of padding or duplicate packets to actively *measure* whether the extra headroom is real, then commits only if the probe arrives without added delay. This is why a WebRTC stream can climb back up quickly after a network hiccup instead of creeping up conservatively. For an audio-only agent the probing is gentle (there is little to probe with in a ~40 kbps Opus stream), but the moment you add video the probes become substantial and briefly visible in `availableOutgoingBitrate`.

Second, **audio and video share one budget, and priority decides who starts.** When both a talking-head avatar and the voice stream compete on a constrained link, GCC allocates the estimated bandwidth across them by priority, and a naive configuration will let video steal from audio — the worst possible trade, because a frozen avatar is tolerable but garbled speech kills the interaction. Set the audio track's `RTCRtpSender` priority high (`setParameters` with `priority: "high"`) so that when the pipe narrows, the picture degrades before the voice does. For a pure voice agent this is moot, but the day a product manager asks for "a face for the assistant," it becomes the difference between a call that gracefully drops video under congestion and one that turns the actual product — the conversation — to mush.

The practical upshot for operators: **do not fight GCC, instrument it.** Log `availableOutgoingBitrate` alongside loss and RTT, and when you see the estimate collapse, you are watching the algorithm correctly protect the call from a degrading path — not misbehaving. The failure mode to watch for is the opposite: an estimate that never rises because a middlebox is silently dropping the probe packets, leaving the stream stuck at a needlessly low bitrate on a link that could carry more.

## 6. Data channels: the reliable/unreliable sibling {#data-channels}

WebRTC is not only media. The **RTCDataChannel** gives you a peer-to-peer channel for arbitrary application data, and for AI products it is where transcripts, function-call events, and control messages travel. In the OpenAI Realtime example earlier, `pc.createDataChannel("oai-events")` is the channel over which the model streams partial transcripts, signals when it starts and stops speaking, and emits tool-call requests — all alongside the audio, over the same encrypted transport.

Data channels run over **SCTP** (Stream Control Transmission Protocol) tunneled inside the same DTLS association as the media. SCTP is the unsung hero here because, unlike TCP (always reliable, always ordered) or UDP (never either), it lets you choose *per channel* whether delivery is ordered and whether it is reliable.

![SCTP data channels expose a 2x2 ordering and reliability matrix that TCP and UDP alone cannot](/imgs/blogs/webrtc-real-time-ai-voice-agents-7.webp)

The matrix above is the whole design space, and it maps directly to `RTCDataChannel` configuration:

- **Ordered + reliable (the default)** behaves like TCP: every message arrives, in order. This is what you want for **transcripts and events** — you cannot have a function-call argument arrive before the function name, and you cannot drop a transcript token. `createDataChannel("events")` with no options gives you this.
- **Ordered + `maxRetransmits: N`** caps retries but preserves order — bounded reliability for data where staleness eventually makes a message worthless but order still matters.
- **Ordered + `maxPacketLifetime: ms`** is the time-bounded variant: retry only for a window, then give up, still in order.
- **Unordered + reliable** delivers everything but in any order — good for independent chunks like a file transfer where each chunk carries its own offset.
- **Unordered + partial reliability** is the real-time regime: **latest-wins** state (a cursor position, a game entity, a live pose) where an old update is useless, or **drop-stale telemetry** where you would rather lose a sample than delay the next one.

```javascript
// Reliable, ordered — transcripts and tool-call events must never drop or reorder.
const events = pc.createDataChannel("events");

// Unordered, at most one retransmit — live "user is typing / speaking" indicators
// where a stale signal is worse than a missing one.
const presence = pc.createDataChannel("presence", { ordered: false, maxRetransmits: 1 });

events.onmessage = (e) => {
  const evt = JSON.parse(e.data);
  if (evt.type === "response.audio_transcript.delta") appendTranscript(evt.delta);
  if (evt.type === "response.function_call_arguments.done") dispatchTool(evt);
};
```

For a voice agent, the pattern that matters most is running the **event/transcript channel reliable-and-ordered** while keeping media unreliable. The audio can afford to lose a packet — the jitter buffer and PLC handle it — but the transcript and the tool-call JSON cannot. Getting this split right is the difference between an agent that occasionally garbles a function-call argument and one that never does.

## 7. The real-time AI voice pipeline and its latency budget {#pipeline}

Everything so far has been WebRTC as a general-purpose media transport. Now we assemble it into the thing you are actually building: a voice agent. Conceptually, a voice turn is a pipeline — the user's speech is captured, transported, recognized, reasoned over, synthesized, and played back — and because the stages run *in sequence*, the latencies **add up**. The end-to-end delay a user feels is the sum of every stage, and that sum is the single number that decides whether your agent feels like a conversation or a transaction.

<figure class="blog-anim">
<svg viewBox="0 0 900 300" role="img" aria-label="Seven sequential stages of a voice-agent turn, each adding milliseconds; a highlight walks stage by stage while a cumulative bar grows to about 800 milliseconds to first audio." style="width:100%;height:auto;max-width:900px">
<style>
.lb-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.lb-name{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.lb-ms{font:700 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.lb-sweep{fill:var(--accent,#6366f1);opacity:.22}
.lb-track{fill:var(--border,#d1d5db);opacity:.4}
.lb-grow{fill:var(--accent,#6366f1)}
.lb-total{font:700 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes lb-walk{0%{transform:translateX(0)}100%{transform:translateX(732px)}}
@keyframes lb-fill{0%{transform:scaleX(0)}100%{transform:scaleX(1)}}
.lb-sw{animation:lb-walk 10.5s steps(7,jump-none) infinite}
.lb-bar{transform-box:fill-box;transform-origin:left center;animation:lb-fill 10.5s steps(7,jump-none) infinite}
@media (prefers-reduced-motion:reduce){.lb-sw{animation:none}.lb-bar{animation:none;transform:scaleX(1)}}
</style>
<rect class="lb-box" x="20"  y="110" width="110" height="90" rx="10"/>
<rect class="lb-box" x="142" y="110" width="110" height="90" rx="10"/>
<rect class="lb-box" x="264" y="110" width="110" height="90" rx="10"/>
<rect class="lb-box" x="386" y="110" width="110" height="90" rx="10"/>
<rect class="lb-box" x="508" y="110" width="110" height="90" rx="10"/>
<rect class="lb-box" x="630" y="110" width="110" height="90" rx="10"/>
<rect class="lb-box" x="752" y="110" width="110" height="90" rx="10"/>
<rect class="lb-sweep lb-sw" x="20" y="110" width="110" height="90" rx="10"/>
<text class="lb-name" x="75"  y="150">capture</text><text class="lb-ms" x="75"  y="178">15 ms</text>
<text class="lb-name" x="197" y="150">network</text><text class="lb-ms" x="197" y="178">30 ms</text>
<text class="lb-name" x="319" y="150">buffer</text><text class="lb-ms" x="319" y="178">40 ms</text>
<text class="lb-name" x="441" y="150">endpoint</text><text class="lb-ms" x="441" y="178">200 ms</text>
<text class="lb-name" x="563" y="150">STT</text><text class="lb-ms" x="563" y="178">100 ms</text>
<text class="lb-name" x="685" y="150">LLM TTFT</text><text class="lb-ms" x="685" y="178">300 ms</text>
<text class="lb-name" x="807" y="150">TTS</text><text class="lb-ms" x="807" y="178">120 ms</text>
<text class="lb-name" x="450" y="90">a voice-agent turn is a chain of sequential stages — latency adds up</text>
<rect class="lb-track" x="20" y="235" width="842" height="26" rx="13"/>
<rect class="lb-grow lb-bar" x="20" y="235" width="842" height="26" rx="13"/>
<text class="lb-total" x="450" y="290">cumulative ≈ 800 ms to first audio (overlap STT+LLM+TTS to hit sub-300 ms)</text>
</svg>
<figcaption>Because the stages run in sequence, end-to-end voice latency is their sum; the only way under 300 ms is to overlap recognition, reasoning, and synthesis.</figcaption>
</figure>
Walk the budget stage by stage, because each one is a decision you own:

- **Capture + Opus encode (~15 ms).** The browser packetizes audio into 20 ms frames. Not much to squeeze here, but the frame size is why sub-20 ms end-to-end is physically impossible.
- **Network transport (~30 ms one way).** This is your ICE path. A direct srflx path in-region might be 20 ms; a TURN relay across an ocean might be 150 ms. This is the number region-local TURN placement moves.
- **Jitter buffer (~40 ms).** The unavoidable smoothing delay from the RTP section. Adaptive, but never zero.
- **Endpointing / VAD (~200 ms).** The agent must decide the user has *stopped speaking* before it responds. Wait too little and you cut users off mid-sentence; wait too long and the agent feels sluggish. This "end-of-turn" silence window is frequently the single largest and most-overlooked term in the budget.
- **Speech-to-text (~100 ms).** Streaming STT emits partial transcripts as audio arrives, so the marginal cost after endpointing is small if you have been transcribing continuously.
- **LLM time-to-first-token (~300 ms).** The model's TTFT, not its full generation time — because you stream. This is usually the largest *compute* term and the one your model choice and prompt length control.
- **TTS first audio (~120 ms).** Streaming TTS emits the first chunk of synthesized audio as the first tokens arrive.

Summed naively, that is roughly **800 ms to first audio** — perceptibly laggy. The entire engineering game of a good voice agent is **overlapping these stages** rather than running them serially: transcribe *while* the user is still talking, start the LLM on partial transcripts, begin TTS on the first tokens, and stream that audio back over WebRTC as it is produced. Done well, the perceived latency collapses toward the sum of the *irreducible* terms (network + buffer + a slice of endpointing + first-token + first-audio-chunk), which is how production systems hit the 200–400 ms "first audible phoneme" range that feels genuinely conversational. This overlapping is exactly the streaming discipline covered in [real-time and full-duplex speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech); WebRTC is the transport that makes it possible to *deliver* that streamed audio without adding its own delay.

## 8. Barge-in, VAD, and turn-taking: the genuinely hard part {#barge-in}

Latency is a budgeting problem you can solve with a spreadsheet and streaming. **Turn-taking** is a systems problem, and it is where most voice agents feel broken. Real conversation is full-duplex: humans interrupt, back-channel ("uh-huh"), and talk over each other, and they expect the machine to stop talking the instant they start. That capability — **barge-in** — is deceptively hard, and it lives at the intersection of WebRTC audio handling and your agent logic.

<figure class="blog-anim">
<svg viewBox="0 0 780 300" role="img" aria-label="Two lanes over time: the agent begins speaking, the user interrupts partway, a VAD marker fires, and the agent's remaining planned speech is cancelled so it can listen." style="width:100%;height:auto;max-width:840px">
<style>
.bt-lane{font:700 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:end}
.bt-cap{font:600 14px ui-sans-serif,system-ui;text-anchor:middle}
.bt-agent{fill:var(--accent,#6366f1)}
.bt-ghost{fill:none;stroke:var(--accent,#6366f1);stroke-width:2;stroke-dasharray:7 5;opacity:0}
.bt-user{fill:#f59e0b}
.bt-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.bt-mark{stroke:#ef4444;stroke-width:2.5;stroke-dasharray:5 4;opacity:0}
.bt-grow{transform-box:fill-box;transform-origin:left center}
@keyframes bt-spoke{0%{transform:scaleX(0)}38%{transform:scaleX(1)}100%{transform:scaleX(1)}}
@keyframes bt-usr{0%,38%{transform:scaleX(0)}72%,100%{transform:scaleX(1)}}
@keyframes bt-show{0%,34%{opacity:0}44%,90%{opacity:1}100%{opacity:0}}
@keyframes bt-ghostshow{0%,38%{opacity:0}46%,90%{opacity:.75}100%{opacity:0}}
.bt-spk{animation:bt-spoke 9s ease-in-out infinite}
.bt-usrb{animation:bt-usr 9s ease-in-out infinite}
.bt-mk{animation:bt-show 9s ease-in-out infinite}
.bt-gh{animation:bt-ghostshow 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.bt-spk{animation:none}.bt-usrb{animation:none;transform:scaleX(1)}.bt-mk,.bt-gh{animation:none;opacity:1}}
</style>
<text class="bt-cap" x="390" y="34" fill="var(--text-primary,#1f2937)">Barge-in: the user talks over the agent, which must stop and listen</text>
<text class="bt-lane" x="104" y="118">User</text>
<text class="bt-lane" x="104" y="208">Agent</text>
<rect class="bt-user bt-usrb bt-grow" x="420" y="94" width="240" height="46" rx="8"/>
<rect class="bt-agent bt-spk bt-grow" x="120" y="184" width="300" height="46" rx="8"/>
<rect class="bt-ghost bt-gh" x="420" y="184" width="300" height="46" rx="8"/>
<line class="bt-mark bt-mk" x1="420" y1="72" x2="420" y2="250"/>
<text class="bt-cap bt-mk" x="470" y="66" fill="#ef4444">VAD → cancel TTS</text>
<line class="bt-axis" x1="120" y1="266" x2="720" y2="266"/>
<text class="bt-cap" x="150" y="288" fill="var(--text-secondary,#6b7280)">agent speaking</text>
<text class="bt-cap" x="560" y="288" fill="var(--text-secondary,#6b7280)">user speaking → agent listens</text>
</svg>
<figcaption>Turn-taking requires detecting user speech mid-response and cancelling the agent's in-flight TTS so the dashed remainder is never played.</figcaption>
</figure>
The timeline above is the barge-in contract. The agent is speaking (the solid bar). The user starts talking partway through. A **voice activity detector (VAD)** on the incoming audio fires. The moment it does, three things must happen, fast: the agent must **stop synthesizing** (cancel the in-flight TTS request), it must **stop sending** the audio it already generated (flush any TTS audio still queued for playout), and it must **switch to listening**. The dashed remainder — the rest of the response the agent *would* have said — must never be played. Get any of those three wrong and users experience the agent "talking over them," which is the single most-reported complaint about voice agents.

Every step here has a WebRTC dimension:

- **Echo cancellation is a prerequisite, not a nicety.** When the agent speaks, that audio comes out of the user's speakers and back into the user's microphone. Without acoustic echo cancellation (AEC), your server's VAD hears the *agent's own voice* on the inbound stream and either falsely triggers barge-in or, worse, transcribes the agent's speech as if the user said it. This is why the `echoCancellation: true` constraint in the very first code sample was load-bearing — the browser's AEC removes the agent's playback from the captured audio before it ever hits the wire.
- **Endpointing is a tunable, and both directions of getting it wrong are bad.** Server-side VAD with an aggressive threshold cuts users off during natural pauses; a lax one leaves awkward silences. There is no universal right answer — it depends on whether users speak in short commands or long rambles.
- **Cancellation must propagate end-to-end.** Cancelling the LLM generation is necessary but not sufficient; you must also drop the TTS audio *already in the playout buffer*, or the agent keeps talking for a second after it "stopped." This is a case study below.

This full-duplex, listen-while-speaking behavior is exactly what models like Moshi and the GPT-4o-realtime family push further — and whether you use a cascaded STT→LLM→TTS pipeline or an end-to-end speech model, the WebRTC transport and its AEC/VAD handling are what make barge-in physically possible. The agent's decision loop here is a specialized instance of the general [agent loop anatomy](/blog/machine-learning/ai-agent/agent-loop-anatomy): perceive (VAD + STT), decide (LLM), act (TTS) — except every arrow has a hard real-time deadline.

## 9. Server-side WebRTC with aiortc: joining as a peer {#server-side}

To run inference, you need the *decoded audio* on your server — which means your server has to be a WebRTC peer, terminating the media, not just brokering signaling. In Python, the tool for this is **aiortc**, a pure-Python WebRTC implementation. Your AI backend becomes a "bot" that joins the call as a peer, receives the user's audio track, runs it through STT→LLM→TTS, and sends synthesized audio back on its own track.

![Real-time AI terminates media at a server (bot-as-peer or SFU), not peer-to-peer](/imgs/blogs/webrtc-real-time-ai-voice-agents-10.webp)

The figure makes the architectural point that trips up first-time builders. A **naive peer-to-peer mesh** — the left column — is the natural first instinct because WebRTC is "peer to peer," but it is wrong for AI: there is *nowhere to run the model.* The media flows directly between browsers over SRTP; your server never sees it; you cannot transcribe it, cannot record it, cannot run an LLM on it, and as the call count grows you get an N-squared explosion of connections. The right architecture — the right column — **terminates the media at a server**: either a **bot that joins as a peer** (aiortc) for a one-on-one agent, or a **Selective Forwarding Unit (SFU)** for multi-party rooms. The server decodes the audio, drives the STT→LLM→TTS loop, and can scale, record, and observe everything centrally.

Here is a real aiortc server that accepts an offer, receives the caller's audio, and wires it into an inference loop:

```python
# server.py — an AI voice agent that joins a WebRTC call as a peer.
# pip install aiortc aiohttp av
import asyncio, fractions
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame

pcs = set()

class AgentAudioTrack(MediaStreamTrack):
    """Outbound track: pulls synthesized PCM from a queue and emits 20 ms Opus-ready frames."""
    kind = "audio"

    def __init__(self):
        super().__init__()
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._timestamp = 0

    async def push_tts(self, pcm_20ms: bytes):
        await self._queue.put(pcm_20ms)

    def flush(self):
        # Barge-in: drop everything we were about to say.
        while not self._queue.empty():
            self._queue.get_nowait()

    async def recv(self) -> AudioFrame:
        pcm = await self._queue.get()
        frame = AudioFrame(format="s16", layout="mono", samples=960)  # 48 kHz * 20 ms
        frame.planes[0].update(pcm)
        frame.sample_rate = 48000
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, 48000)
        self._timestamp += 960
        return frame

async def consume_user_audio(track: MediaStreamTrack, agent_track: AgentAudioTrack):
    """Inbound track: run VAD + streaming STT, and on end-of-turn drive the LLM+TTS."""
    from my_pipeline import StreamingVAD, StreamingSTT, run_llm_and_tts  # your code
    vad, stt = StreamingVAD(), StreamingSTT()
    while True:
        frame = await track.recv()                 # a decoded 20 ms AudioFrame
        pcm = bytes(frame.planes[0])
        speech = vad.process(pcm)
        if speech.user_started and agent_track:      # barge-in!
            agent_track.flush()                       # stop talking immediately
            run_llm_and_tts.cancel_current()          # cancel in-flight generation
        stt.feed(pcm)
        if speech.end_of_turn:
            transcript = stt.finalize()
            # Stream tokens -> TTS -> agent_track.push_tts(...) as audio is produced.
            asyncio.create_task(run_llm_and_tts(transcript, agent_track))

async def offer(request):
    params = await request.json()
    pc = RTCPeerConnection(RTCConfiguration([
        RTCIceServer("stun:stun.l.google.com:19302"),
        RTCIceServer(urls="turn:turn.example.com:3478",
                     username=params["turnUser"], credential=params["turnCred"]),
    ]))
    pcs.add(pc)
    agent_track = AgentAudioTrack()
    pc.addTrack(agent_track)

    @pc.on("track")
    def on_track(track):
        if track.kind == "audio":
            asyncio.ensure_future(consume_user_audio(track, agent_track))

    @pc.on("connectionstatechange")
    async def on_state():
        if pc.connectionState in ("failed", "closed"):
            await pc.close(); pcs.discard(pc)

    await pc.setRemoteDescription(RTCSessionDescription(params["sdp"], "offer"))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

app = web.Application()
app.router.add_post("/offer", offer)
web.run_app(app, port=8080)
```

The shape of this server is the shape of nearly every production voice agent, whether it is 200 lines of aiortc or a managed platform: **one inbound track feeds VAD + STT, end-of-turn triggers the LLM+TTS, the synthesized audio streams out on an outbound track, and a VAD-triggered `flush()` + `cancel()` implements barge-in.** Everything else — scaling, recording, observability — hangs off the fact that the server sits in the media path.

### Second-order: P2P vs SFU vs MCU, and where AI fits

For completeness: a **mesh** (every peer connects to every other) is fine for 2–3 participants and never involves your server in media. An **SFU** receives each participant's stream once and *forwards* it to the others without decoding — the standard for scalable multi-party video, and where you would run a *shared* AI agent across many participants. An **MCU** decodes and re-composites everything into one stream — expensive, rarely used now. For a one-on-one voice agent, the aiortc bot-as-peer above is the simplest correct choice; for a multi-party meeting with an AI participant, you attach the agent to an SFU. In all cases the load balancing is not the L4/L7 HTTP kind you are used to — a WebRTC session is sticky, long-lived, and UDP, so you route the *signaling* through your normal [L4-to-L7 load balancer](/blog/software-development/system-design/load-balancing-from-l4-to-l7) but the *media* pins to a specific media server for the life of the call.

## 10. Observability and scaling: you cannot fix what you cannot measure {#observability}

Because WebRTC hides so much, the only way to operate it in production is to pull the statistics it exposes and watch them like you watch any other SLI. The `getStats()` API returns a dictionary of live metrics per connection; the ones that actually predict user pain are few.

```javascript
// Poll the connection every 2s and surface the metrics that correlate with "it feels bad".
setInterval(async () => {
  const stats = await pc.getStats();
  for (const report of stats.values()) {
    if (report.type === "inbound-rtp" && report.kind === "audio") {
      const lossPct = 100 * report.packetsLost / (report.packetsReceived + report.packetsLost);
      console.log({
        jitter_ms: (report.jitter * 1000).toFixed(1),
        loss_pct: lossPct.toFixed(2),
        // jitterBufferDelay / emittedCount ~= current buffer depth in seconds
        buffer_ms: (1000 * report.jitterBufferDelay / report.jitterBufferEmittedCount).toFixed(0),
        concealment_pct: (100 * report.concealedSamples / report.totalSamplesReceived).toFixed(2),
      });
    }
    if (report.type === "candidate-pair" && report.nominated) {
      console.log({ rtt_ms: (report.currentRoundTripTime * 1000).toFixed(0),
                     send_kbps: (report.availableOutgoingBitrate / 1000).toFixed(0) });
    }
  }
}, 2000);
```

| Metric (from `getStats`) | What it tells you | Threshold to alarm on |
| --- | --- | --- |
| `currentRoundTripTime` (candidate-pair) | Path latency; jumps when you fall back to a distant relay | > 300 ms sustained |
| `packetsLost` / received | Network quality; drives GCC and FEC | > 3% for audio |
| `jitter` | Arrival-time variance; sizes the jitter buffer | > 30 ms |
| `jitterBufferDelay` / emitted | Actual buffer depth — creeping delay lives here | > 200 ms or steadily rising |
| `concealedSamples` / total | Fraction of audio the decoder had to fake (PLC) | > 5% means audible artifacts |
| `availableOutgoingBitrate` | GCC's current estimate; collapses on congestion | Sudden drop = degrading path |

> The three numbers that predict a bad voice-agent call, in order: round-trip time (are we relayed and far?), concealment rate (is the audio being faked?), and jitter-buffer depth (is delay creeping up?). Chart those three and you will diagnose most incidents before the user finishes typing the complaint.

Scaling WebRTC is unlike scaling stateless HTTP. Each media session is stateful, long-lived, CPU-heavy on the server (decode + inference), and UDP. You scale media servers horizontally and route new sessions to the least-loaded one *in the caller's region*, you provision TURN capacity per region (relayed calls are the expensive ones), and you plan for the fact that a media server draining for deploy means gracefully migrating or ending live calls — you cannot just kill the pod. This is closer to operating a stateful database fleet than a web tier.

## Troubleshooting playbook {#troubleshooting}

When a WebRTC call misbehaves, the symptom points almost deterministically at the layer that owns it. The decision tree below is the one I run in my head on every incident; it turns "WebRTC is broken" into a specific hypothesis in under a minute.

![Most WebRTC failures diagnose along a short symptom-to-layer-to-fix path](/imgs/blogs/webrtc-real-time-ai-voice-agents-11.webp)

Read it as symptom → most-likely cause → first fix:

- **No audio at all.** Almost always **ICE failed** — no candidate pair connected. Check `iceConnectionState`; if it is `failed`, your candidates never found a working path. First fix: confirm TURN actually works (the trickle-ICE test), including the `turns:` on 443. If ICE reached `connected` but there is still no audio, suspect **DTLS** — the handshake did not complete and SRTP was never keyed.
- **One-way audio.** A classic **asymmetric NAT or inbound-firewall** problem: one peer can send, the other cannot get packets back. First fix: force relay in both directions (`iceTransportPolicy: "relay"` to confirm the hypothesis), then check that both peers have a working TURN candidate.
- **Choppy or robotic audio.** **Packet loss above ~3% with no FEC or PLC** kicking in. Check `concealedSamples` and `packetsLost` in `getStats`. First fix: enable Opus in-band FEC (`useinbandfec=1`) and DTX, and let the jitter buffer size up.
- **High latency.** You are almost certainly on a **TCP/TLS TURN relay** (often far away) and/or your **jitter buffer has grown deep** under bursty loss. First fix: region-local TURN placement and a cap on buffer depth; verify with `currentRoundTripTime` and `jitterBufferDelay`.
- **Echo or the agent talking over the user.** Not in the tree but worth stating: missing AEC on capture, or TTS audio not being flushed on barge-in. First fix: confirm `echoCancellation: true` and that your cancellation path flushes the playout buffer, not just the LLM.

The meta-lesson: **let the symptom name the layer.** No audio is ICE/DTLS; one-way is NAT/firewall; choppy is loss/FEC; laggy is relay/buffer. You rarely need to inspect the SDP byte-by-byte if you start from the symptom and check the one layer that owns it.

## Case studies from production

Ten incidents, lightly fictionalized from real ones, each following the same arc: the symptom, the wrong first guess, the actual root cause, the fix, and the lesson.

### 1. The corporate firewall that only allowed 443

**Symptom:** A B2B voice-assistant pilot worked for everyone on the team and failed for the entire customer — no audio, every time, for every employee. **Wrong first guess:** "Their microphone permissions are blocked." **Root cause:** The customer's corporate firewall blocked all UDP and all non-443 TCP. Our ICE config had `stun:` and `turn:...:3478?transport=udp` but no TLS relay on 443, so *no* candidate pair could form. **Fix:** Added `turns:turn.example.com:443?transport=tcp`. Connections that had zero working candidates suddenly had one — slow, relayed, on 443, but working. Success rate on that customer went from 0% to 100% overnight. **Lesson:** The `turns:` on 443 is not optional; it is the candidate of last resort that defines whether locked-down enterprise networks can use your product at all.

### 2. One-way audio on the mobile app

**Symptom:** On the web it was flawless; in the iOS app, roughly a third of calls had audio from the agent to the user but nothing back. **Wrong first guess:** "aiortc isn't receiving the track." **Root cause:** Those users were on carrier-grade NAT (symmetric), so their server-reflexive candidate was useless for inbound packets — the agent's STUN checks toward the user never got a reply, so the pair for user→agent never nominated, while agent→user worked over a different pair. **Fix:** Ensured a relay candidate was always gathered (region-local TURN) so a symmetric bidirectional path existed. **Lesson:** One-way audio is the fingerprint of asymmetric NAT; the fix is always a relay that both directions can traverse, and mobile carriers make this the common case, not the edge case.

### 3. The two-second delay nobody could explain

**Symptom:** Calls started crisp and, over 30–60 seconds, developed a growing delay until the agent was responding two seconds behind the user. **Wrong first guess:** "The LLM is getting slower as context grows." **Root cause:** The user's network had bursty packet loss. Each burst triggered the jitter buffer to grow to absorb it — but it never shrank back, and the client had additionally fallen back to a **TCP** TURN relay, whose head-of-line blocking compounded the buffering. Delay accumulated monotonically. **Fix:** Capped the jitter buffer's maximum depth, preferred UDP relay, and alarmed on `jitterBufferDelay` rising. **Lesson:** Creeping latency is almost never the model; it is buffer growth and TCP relays. Chart `jitterBufferDelay` over the call and the culprit is obvious.

### 4. Barge-in that never triggered

**Symptom:** Users interrupted the agent and it kept talking; barge-in "didn't work" maybe half the time. **Wrong first guess:** "Our VAD threshold is too high." **Root cause:** Echo. The user's laptop speakers fed the agent's own voice back into the microphone, and with AEC misconfigured the server's VAD could not distinguish the user's speech from the agent's playback, so it either ignored the user (treating everything as the agent's echo) or the STT transcribed the agent's own words. **Fix:** Enabled `echoCancellation`, `noiseSuppression`, and `autoGainControl` on capture, and gated the server VAD to ignore inbound audio during known agent-playback windows as a belt-and-suspenders. **Lesson:** Barge-in is an echo-cancellation problem before it is a VAD problem. If the agent can hear itself, turn-taking is impossible.

### 5. The voice agent that talked over itself

**Symptom:** Barge-in *detected* the user correctly and cancelled the LLM, but the agent still spoke for about a second after the user started. **Wrong first guess:** "The cancel signal is slow to reach the model." **Root cause:** We cancelled the LLM generation, but the TTS audio it had *already produced* was sitting in the outbound track's playout queue and kept draining to the user. We flushed the model but not the audio buffer. **Fix:** On barge-in, call `agent_track.flush()` to drop queued PCM *and* cancel generation — both, in that order. **Lesson:** Cancellation must propagate to the very end of the pipeline. The last buffer in the chain is the one users hear.

### 6. Robotic voice under 3% loss

**Symptom:** On mildly lossy networks the agent's voice sounded robotic and warbly, even though loss was only 2–3%. **Wrong first guess:** "The TTS model quality is bad." **Root cause:** Opus in-band FEC was disabled (the SDP was being munged by a proxy that stripped `useinbandfec=1`), so every lost packet became a concealment artifact instead of being reconstructed from redundant data. `concealedSamples` was 6% of total. **Fix:** Restored the fmtp line, enabled DTX to free bandwidth headroom during silence, and confirmed FEC in the negotiated SDP. Concealment dropped below 1% and the robotic quality vanished. **Lesson:** Audio quality problems under mild loss are almost always a missing FEC toggle, not a model problem. Check `concealedSamples` before you retrain anything.

### 7. TURN costs that exploded

**Symptom:** The monthly egress bill for the TURN fleet was five times the forecast. **Wrong first guess:** "Usage grew faster than expected." **Root cause:** A misconfiguration disabled host and srflx candidate gathering on the client (an over-aggressive `iceTransportPolicy: "relay"` shipped by mistake), so **100% of calls relayed** through TURN even when a direct path existed. Every byte of every call transited our servers. **Fix:** Reverted to `iceTransportPolicy: "all"` so ICE preferred direct paths and only relayed the ~12% that needed it. Cost fell back to forecast. **Lesson:** `iceTransportPolicy: "relay"` is a fantastic *debugging* tool and a catastrophic *production default*. Relay only when nothing else works.

### 8. Works on Wi-Fi, fails on LTE

**Symptom:** A user reported the agent worked at home and failed on their phone's cellular connection, consistently. **Wrong first guess:** "Cellular is just too lossy." **Root cause:** Two things. The cellular path was IPv6-only in a way our TURN server (IPv4-only) could not relay, and separately the path MTU was smaller, so large DTLS handshake packets fragmented and were dropped, stalling the handshake. **Fix:** Dual-stack TURN (IPv4 + IPv6) and ensured DTLS used a conservative MTU. **Lesson:** "Works on Wi-Fi, fails on LTE" is a network-topology fingerprint — IPv6 reachability, MTU, and symmetric NAT all differ on cellular. Test on a real phone on real cellular, not just office Wi-Fi.

### 9. SDP negotiation glare during screen share

**Symptom:** When a user clicked "share screen with the assistant" while the agent happened to be adding a video avatar track, the connection wedged and had to be reloaded. **Wrong first guess:** "Screen capture is failing." **Root cause:** Both peers fired `onnegotiationneeded` and sent offers simultaneously — glare — and neither yielded, deadlocking the signaling state machine. **Fix:** Implemented the perfect-negotiation pattern with the agent designated *polite*, so on collision it rolled back and accepted the client's offer. **Lesson:** The moment either side can renegotiate, you need perfect negotiation. It is not optional the day you add a second media source.

### 10. Clock skew broke DTLS on kiosks

**Symptom:** A fleet of in-store kiosks running the voice agent failed to connect every morning after power-cycling, then started working an hour later. **Wrong first guess:** "The network isn't up yet at open." **Root cause:** The kiosks booted with a wrong system clock (dead RTC battery) and only synced NTP after a delay. During that window, the self-signed DTLS certificate's "not before" time was in the *future* relative to the kiosk's clock, so the peer rejected it and the handshake failed — ICE connected, DTLS did not, audio was silent. **Fix:** Forced NTP sync before launching the agent and added a startup guard. **Lesson:** DTLS depends on a sane clock. "ICE connects, no audio, resolves itself later" on embedded devices is a clock-skew fingerprint, not a network one.

## Best practices checklist

Distilled from the sections and the war stories, the things I would put in a code review checklist for any WebRTC-based AI product:

- **Always ship TURN, including `turns:` on 443/TCP.** Test it with the trickle-ICE tool. STUN-only is a product that silently fails for 10–20% of users.
- **Keep `iceTransportPolicy: "all"` in production.** Use `"relay"` only to reproduce relay-path bugs, never as a default.
- **Use ephemeral, HMAC-time-limited TURN credentials.** Never static passwords in client code.
- **Enable `echoCancellation`, `noiseSuppression`, `autoGainControl` on capture.** They are prerequisites for barge-in, not audio polish.
- **Keep Opus in-band FEC and DTX on, and verify them in the negotiated SDP** — proxies strip fmtp lines.
- **Run the events/transcript data channel reliable + ordered; keep media unreliable.** Never let a tool-call argument reorder.
- **On barge-in, cancel the LLM *and* flush the TTS playout buffer**, in that order.
- **Terminate media at a server (aiortc bot or SFU) for anything AI.** P2P has nowhere to run inference.
- **Overlap STT, LLM, and TTS.** The latency budget is a sum; the only way under 300 ms is to run stages concurrently.
- **Place TURN and media servers region-locally** and route sessions to the caller's region; relay hops across oceans are pure latency.
- **Chart `currentRoundTripTime`, `concealedSamples`, and `jitterBufferDelay`** as your top-line voice SLIs.
- **Sync the clock before connecting on embedded/kiosk devices.** DTLS depends on it.

## When to reach for WebRTC — and when not to

WebRTC is powerful and complex, and the complexity is only worth it for a specific shape of problem.

**Reach for WebRTC when:**

- You need **sub-500 ms bidirectional media** between a browser (or mobile) and something else — the defining case for voice and video agents.
- Your client is a **browser** and you cannot install native networking — WebRTC is the only real-time media transport browsers expose.
- You must **cross NATs and hostile networks** and cannot control the client's firewall — ICE + TURN is the mature, battle-tested answer.
- You need the media plane's **adaptivity** — congestion control, FEC, jitter buffering — because your users are on unpredictable networks (mobile, home Wi-Fi, cafés).
- You are building **conversational AI** where barge-in and low latency define the experience.

**Skip WebRTC when:**

- The communication is **server-to-server.** Use gRPC, HTTP/2, or a message queue — you control both ends and NAT traversal is a non-problem you would be paying for.
- You need **one-way, high-quality streaming to many viewers** (a webinar, a broadcast). Use HLS or LL-HLS/DASH with a CDN; they scale to millions where WebRTC's per-connection state does not.
- Your real-time need is **text or small events, not media.** A plain WebSocket or Server-Sent Events is simpler, debuggable with `curl`, and has none of the ICE/DTLS/SRTP machinery.
- You **cannot operate TURN** (no budget, no ops capacity for a stateful media fleet). Without a working relay, WebRTC will fail for a meaningful fraction of users — in that case a managed platform like LiveKit or a realtime API that hides the transport is the honest choice.
- **Latency does not actually matter.** If a 2–3 second response is fine, a request/response API over HTTP is far less to build and operate.

The through-line of this entire article is the two-plane mental model from the very first diagram. Signaling is yours; the media plane is the browser's — and every hard WebRTC problem, from a dead call on a corporate firewall to an agent that talks over its users, is really a question of understanding which layer of that media plane owns the symptom. Learn the layers — ICE, DTLS-SRTP, RTP and the jitter buffer, GCC — and WebRTC stops being a black box that "sometimes doesn't work" and becomes what it actually is: a remarkably well-engineered real-time transport that is finally getting the AI applications it was built for.

## Further reading

- [High Performance Browser Networking — WebRTC chapter](https://hpbn.co/webrtc/) — Ilya Grigorik's canonical, free, protocol-level treatment.
- [WebRTC for the Curious](https://webrtcforthecurious.com/) — a vendor-neutral, deeply technical open book covering ICE, DTLS-SRTP, RTP, and congestion control.
- [MDN WebRTC API](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API) — the authoritative reference for `RTCPeerConnection`, `RTCDataChannel`, perfect negotiation, and `getStats`.
- [aiortc documentation](https://aiortc.readthedocs.io/) — the Python WebRTC library used for the server-side agent above.
- [OpenAI Realtime API — WebRTC guide](https://platform.openai.com/docs/guides/realtime-webrtc) — the SDP-over-HTTPS handshake and event data channel in practice.
- [LiveKit: real-time communication and AI voice agents](/blog/software-development/system-design/livekit-real-time-communication) — the platform that wraps everything in this post, when you would rather not build it yourself.
- [Real-time, streaming, and full-duplex speech](/blog/machine-learning/audio-generation/real-time-streaming-and-full-duplex-speech) — the model side of the latency and barge-in story.
