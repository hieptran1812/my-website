---
title: "TCP Reliability: Sequence Numbers, ACKs, Retransmits, and RTO"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn to read TCP loss recovery from sequence ranges, ACK evidence, retransmission timers, RACK, and tail-loss probes."
tags:
  [
    "networking",
    "distributed-systems",
    "tcp",
    "reliability",
    "retransmission",
    "sack",
    "rto",
    "rack",
    "tail-loss-probe",
    "packet-analysis",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 41
image: "/imgs/blogs/reliability-sequence-numbers-acks-retransmits-and-rto-1.webp"
---

A request finishes its DNS lookup, reuses an established connection, reaches the server, and receives almost the entire response in a few milliseconds. Then it stops. CPU is quiet. The server log says the handler completed. No queue is obviously full. A few hundred milliseconds later, one TCP segment appears again and the request instantly completes. The application reports a latency spike, but the work was already done.

That shape is not mysterious once we stop treating reliability as a promise and start treating it as an evidence system. TCP can retransmit only after the sender has evidence that bytes are missing. A hole in the middle of a flight produces evidence because later segments arrive. A hole at the tail can produce silence. Silence is expensive because the sender must create evidence with a probe or wait for a retransmission timer.

![A request latency ladder with the transfer phase highlighted as the owner of a tail-loss stall](/imgs/blogs/reliability-sequence-numbers-acks-retransmits-and-rto-1.webp)

The diagram above is the mental model: TCP reliability lives inside the transfer phase, but one recovery delay can dominate the wall clock of the whole request. We will build that mechanism from byte sequence numbers, cumulative acknowledgments, selective acknowledgments, duplicate ACKs, RTT estimation, retransmission timeout, Recent Acknowledgment, and Tail Loss Probe. We will finish with packet and kernel measurements that tell these paths apart.

This post continues the packet journey introduced in [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). It assumes the socket contract from [TCP versus UDP](/blog/software-development/networking/sockets-and-the-two-transport-contracts-tcp-vs-udp) and the connection setup from [the TCP handshake and what it costs you](/blog/software-development/networking/the-tcp-handshake-and-what-it-costs-you). The goal is narrower than congestion control. Here we ask one operational question: what evidence made this sender retransmit now?

## 1. Reliability lives in the transfer phase

**Senior rule:** when a request is almost complete and then pauses, inspect the final data sequence range before blaming server work.

TCP provides an application with a reliable, in-order byte stream. The network underneath provides IP datagrams that may be dropped, duplicated, or reordered. TCP bridges those contracts by assigning positions to bytes, reporting which positions arrived, retaining unacknowledged data, and sending missing ranges again. [RFC 9293, published August 2022](https://www.rfc-editor.org/rfc/rfc9293.html) is the current base TCP specification and describes reliability as loss detection using sequence numbers plus correction through retransmission.

The important word is detection. Retransmission is the repair action, not the detector. Different evidence paths detect loss at different speeds:

| Evidence path | What the sender learns | Typical recovery trigger | What can delay it | Source |
| --- | --- | --- | --- | --- |
| Advancing cumulative ACK | All bytes below a new boundary arrived | Free acknowledged data and advance the send window | ACK loss or delayed acknowledgment | [RFC 9293, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html) |
| Duplicate ACKs | Later data arrived while the same next byte is still missing | Classic fast retransmit after the specified threshold | Too few later segments, reordering ambiguity | [RFC 5681, September 2009](https://www.rfc-editor.org/rfc/rfc5681.html) |
| SACK blocks | Specific ranges beyond the cumulative boundary arrived | Scoreboard-driven recovery of one or more holes | Option negotiation, ACK loss, receiver reneging | [RFC 2018, October 1996](https://www.rfc-editor.org/rfc/rfc2018.html) and [RFC 6675, August 2012](https://www.rfc-editor.org/rfc/rfc6675.html) |
| RACK timing | A newer transmission was delivered while an older one remained unacknowledged long enough | Time-based loss inference and fast recovery | Reordering window and sparse feedback | [RFC 8985, February 2021](https://www.rfc-editor.org/rfc/rfc8985.html) |
| TLP probe ACK | A probe creates fresh feedback near the tail | RACK or ACK processing exposes the missing tail | Probe loss, RTO already sooner | [RFC 8985, February 2021](https://www.rfc-editor.org/rfc/rfc8985.html) |
| RTO expiration | The oldest outstanding data remained unacknowledged through the timer | Timeout retransmission and conservative congestion response | A deliberately guarded timer | [RFC 6298, June 2011](https://www.rfc-editor.org/rfc/rfc6298.html) |

These are not six interchangeable retries. They are six ways of learning enough to act. That distinction is the key to reading a capture. A retransmitted packet tells us that the sender made a decision. The ACK history, SACK blocks, elapsed time, and kernel counters tell us why.

The application has even less information. A blocking read merely waits for the next byte in stream order. If bytes after a hole have already arrived, the receiver may hold them in its out-of-order queue, but the application still cannot consume through the missing range. A 40 KiB response can therefore be physically present except for one segment and still be logically incomplete.

This is also why an HTTP retry is not the first diagnostic move. The TCP connection may be repairing the original transfer below the application. An application retry can add another request, another connection, or duplicate side effects while the first connection is still making progress. The policy boundary belongs in [timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right). Our job here is to identify the wire mechanism before changing that policy.

### The transfer can be longer than the work

Consider an explanatory timing model, not a protocol equation:

$$
T_{request} = T_{pretransfer} + T_{server} + T_{data} + T_{recovery}
$$

Suppose a warm connection has 4 ms of request and server time, an 80 ms round-trip time, and a response that would otherwise finish in one flight. If middle loss creates three duplicate ACKs, a retransmission can be sent without waiting for the RTO, and the repair cost can be near one additional RTT under a favorable flight pattern. If the last segment is lost and no later data arrives, the same amount of lost data may instead wait for a TLP probe or the RTO path. The bytes lost are identical. The available evidence is not.

Do not convert that example into a latency guarantee. ACK policy, congestion window, pacing, reordering, implementation, and whether the retransmission itself is lost all change the observed result. The invariant is structural: later delivery creates feedback for a middle hole; tail loss does not naturally create later delivery.

## 2. Sequence numbers turn a byte stream into recoverable ranges

**Senior rule:** read `seq` and `ack` as half-open byte ranges, not as packet counters.

![TCP send and receive sequence space divided into acknowledged, outstanding, and future byte ranges](/imgs/blogs/reliability-sequence-numbers-acks-retransmits-and-rto-2.webp)

TCP numbers bytes. A segment's Sequence Number is the number of its first data octet. If a segment carries 1,000 bytes beginning at sequence number 4,000, it covers the half-open range `[4000,5000)`. An ACK value of 5,000 says, in ordinary language, "I have every byte below 5,000 and expect byte 5,000 next."

That half-open notation removes several common off-by-one errors. The first endpoint is included. The second endpoint is not. Segment length is therefore `end - start`. Packet boundaries can change because of segmentation offload, retransmission, or path constraints while the byte ranges remain the same. A capture may show one original segment carrying `[4000,5000)` and two retransmissions carrying `[4000,4500)` and `[4500,5000)`. Reliability is still defined over the byte positions.

[RFC 9293 Section 3.4](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.4) defines a finite 32-bit sequence space with arithmetic modulo ${2}^{32}$. It also names the variables that make captures easier to reason about:

| Variable | Meaning | Operational question | Source |
| --- | --- | --- | --- |
| `SND.UNA` | Oldest unacknowledged sequence number | What data must the sender still retain? | [RFC 9293, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.3.1) |
| `SND.NXT` | Next sequence number to send | How far has the sender emitted new data? | [RFC 9293, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.3.1) |
| `RCV.NXT` | Next sequence number expected | Where is the receiver's first hole? | [RFC 9293, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.3.1) |
| `SEG.SEQ` | First sequence number in this segment | Which byte range does this packet begin? | [RFC 9293, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.4) |
| `SEG.ACK` | Acknowledgment carried by this segment | How far has the peer cumulatively advanced? | [RFC 9293, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.4) |

SYN and FIN each consume one position in sequence space even though they are control flags rather than application bytes. That is why the first data byte follows the initial sequence number plus one. An ACK that covers a FIN also advances by one. ACK itself consumes no sequence space.

### Worked range example

Use deliberately small, illustrative offsets. A sender has `SND.UNA = 1000` and sends three data segments:

| Segment | Byte range | Payload bytes | Receiver result | Source |
| --- | ---: | ---: | --- | --- |
| A | `[1000,2000)` | 1,000 | Arrives | Illustrative range derived here |
| B | `[2000,3000)` | 1,000 | Lost | Illustrative range derived here |
| C | `[3000,4000)` | 1,000 | Arrives out of order | Illustrative range derived here |

After A arrives, the receiver advances `RCV.NXT` to 2,000 and sends ACK 2,000. C cannot advance the cumulative boundary because B is absent, so C causes another ACK 2,000. If SACK was negotiated, that ACK can also report that `[3000,4000)` is present. When a retransmission fills `[2000,3000)`, the receive stream becomes contiguous through 4,000, and ACK 4,000 can cover both the repaired range and the buffered range.

Notice what did not matter. Segment C did not need to be delivered to the application before it could generate evidence. The receiver's TCP stack could acknowledge and buffer it out of order. The application still waits at byte 2,000 because TCP promises in-order delivery at its interface.

### Why packet numbers in Wireshark look friendlier

Packet analyzers often display relative sequence numbers by subtracting the connection's initial sequence number. That makes the first data byte appear near 1 rather than as a large 32-bit value. Relative numbers are excellent for human inspection, but they are a presentation transform. When two captures disagree, confirm whether relative numbering is enabled before concluding that the endpoints used different sequence space.

Use fields, not the Info column, for reproducible analysis:

```bash
tshark -r capture.pcapng \
  -Y 'tcp.port == 8080' \
  -T fields \
  -e frame.time_relative \
  -e ip.src -e tcp.srcport \
  -e tcp.seq -e tcp.len -e tcp.ack \
  -e tcp.options.sack_le -e tcp.options.sack_re
```

The exact SACK field names can vary across Wireshark releases, so check them with `tshark -G fields | rg 'tcp.options.sack'` on the analysis host. Keep the `tshark --version` output with serious incident artifacts. Packet dissection is software, not scripture.

### Wraparound is real, but rarely your first hypothesis

Because sequence arithmetic is modulo ${2}^{32}$, ordinary integer comparisons are unsafe near wraparound. At 100 Gbit/s, a purely arithmetic lower bound for sending ${2}^{32}$ bytes is:

$$
\frac{{2}^{32}\ \text{bytes} \times 8\ \text{bits/byte}}{{100}\times {10}^{9}\ \text{bits/s}} \approx 0.344\ \text{s}
$$

This is a derived serialization bound that ignores headers, pacing, congestion control, and every real bottleneck. It simply shows why modern stacks cannot treat 32-bit values as an ever-increasing global counter. TCP timestamp mechanisms and careful modular comparisons exist because high-rate connections can traverse the numeric space quickly. For a single stalled request, however, a missing byte range, ACK behavior, or timeout is usually a more productive first hypothesis than sequence wrap.

## 3. Cumulative ACK is the baseline; SACK adds a scoreboard

**Senior rule:** the ACK field names the first missing byte; a SACK block describes evidence beyond that boundary.

![Comparison of cumulative acknowledgments alone with cumulative acknowledgments plus a SACK scoreboard](/imgs/blogs/reliability-sequence-numbers-acks-retransmits-and-rto-3.webp)

The base acknowledgment is cumulative. ACK X means every byte before X has been received. This is compact and robust. One number can retire a large prefix of transmitted data, and a later ACK can cover an earlier ACK that was itself lost.

The cost of that compactness is ambiguity above the first hole. If ACK 2,000 repeats, the sender knows the receiver still expects byte 2,000. Without more information, the sender does not know exactly which later ranges arrived. Several later segments may be buffered, one may be buffered, or the repeated ACK may result from reordering or duplication.

Selective Acknowledgment, specified by [RFC 2018 in October 1996](https://www.rfc-editor.org/rfc/rfc2018.html), adds TCP options that report noncontiguous blocks received above the cumulative ACK point. SACK does not replace the cumulative ACK. The cumulative field remains the authoritative left edge of delivered sequence space. SACK adds a map of islands beyond the hole.

In the earlier example, the receiver can send `ACK 2000, SACK 3000-4000`.

If another segment `[4000,5000)` arrives, the receiver can report a larger contiguous block as `ACK 2000, SACK 3000-5000`.

The sender can maintain a scoreboard marking `[3000,5000)` as selectively acknowledged while `[2000,3000)` remains missing. [RFC 6675, published August 2012](https://www.rfc-editor.org/rfc/rfc6675.html), specifies a conservative SACK-based recovery algorithm and variables such as `HighACK`, `HighData`, `HighRxt`, and `Pipe`. The central operational benefit is straightforward: multiple losses in one flight can be recovered using explicit range evidence rather than rediscovering each hole through cumulative progress alone.

### SACK is advisory, not permission to forget data

RFC 6675 states that SACKed data must not be removed from the retransmission buffer until it is cumulatively acknowledged. A receiver is allowed to renege, meaning it can discard data it previously reported in a SACK block under exceptional conditions. That is uncommon, but it explains the contract boundary. A SACK block is strong recovery evidence. It is not the final delivery commitment represented by cumulative ACK progress.

This matters when reading memory pressure or unusual recovery behavior. A sender that retains SACKed bytes is not wasting memory accidentally. It is preserving its ability to satisfy the reliable byte-stream contract until the cumulative boundary passes them.

### ACK loss is not data loss

Suppose the receiver sends ACK 3,000 and that ACK is lost. It later sends ACK 5,000. The later cumulative ACK covers all bytes below 5,000, including the range acknowledged by the missing packet. TCP does not need to retransmit an ACK merely because that ACK packet vanished.

Data loss and ACK loss can still interact. Sparse ACK feedback delays RTT samples and can delay the evidence used by RACK. An ACK carrying unique SACK information can be lost, requiring later ACKs to repeat useful blocks. But an isolated ACK loss does not automatically create an application-visible gap because the next cumulative ACK can subsume it.

### Delayed ACK is not the same as duplicate ACK

A receiver may delay ordinary ACKs to reduce overhead, subject to protocol rules and implementation policy. An out-of-order segment is different. RFC 5681 says the receiver should send an immediate duplicate ACK when a segment arrives above a gap, precisely to accelerate loss recovery. Treating every repeated ACK as "the receiver is slow to ACK" throws away the strongest clue in the capture.

The next post on [Nagle, delayed ACKs, and the 40 ms mystery](/blog/software-development/networking/nagle-delayed-acks-and-the-40ms-mystery) owns the interaction between small writes and acknowledgment policy. Here the boundary is simple: duplicate ACKs created by out-of-order delivery are a loss signal, while delayed ACK policy governs ordinary in-order acknowledgment timing.

## 4. Duplicate ACKs make middle loss fast and tail loss slow

**Senior rule:** count how many packets can arrive after the hole. That count determines whether classic fast retransmit can even obtain its evidence.

<figure class="blog-anim">
<svg viewBox="0 0 920 560" role="img" aria-label="Middle loss produces duplicate acknowledgements and fast retransmit, while tail loss stays silent until a TLP probe or RTO fallback" style="width:100%;height:auto;max-width:920px">
<style>
.tcp8-title{font:700 18px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.tcp8-lbl{font:600 13px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.tcp8-small{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}.tcp8-axis{stroke:var(--border,#d1d5db);stroke-width:3}.tcp8-flow{stroke:var(--text-secondary,#6b7280);stroke-width:2}.tcp8-loss{stroke:#ef4444;stroke-width:4}.tcp8-ack{stroke:var(--accent,#6366f1);stroke-width:2}.tcp8-probe{stroke:#f59e0b;stroke-width:4}.tcp8-dot{fill:var(--accent,#6366f1)}.tcp8-warn{fill:#f59e0b}.tcp8-bad{fill:#ef4444}.tcp8-card{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}.tcp8-gap{fill:none;stroke:#f59e0b;stroke-width:2;stroke-dasharray:6 5}
@keyframes tcp8-middle{0%,8%{opacity:.18}14%,44%{opacity:1}50%,100%{opacity:.18}}
@keyframes tcp8-tail{0%,46%{opacity:.18}52%,94%{opacity:1}100%{opacity:.18}}
@keyframes tcp8-pulse{0%,18%,100%{opacity:.25}24%,38%{opacity:1}44%{opacity:.25}}
@keyframes tcp8-wait{0%,54%{stroke-dashoffset:36;opacity:.25}64%,78%{stroke-dashoffset:0;opacity:1}88%,100%{opacity:.25}}
.tcp8-midanim{animation:tcp8-middle 14s ease-in-out infinite}.tcp8-tailanim{animation:tcp8-tail 14s ease-in-out infinite}.tcp8-fast{animation:tcp8-pulse 14s ease-in-out infinite}.tcp8-waitanim{animation:tcp8-wait 14s linear infinite}
@media (prefers-reduced-motion:reduce){.tcp8-midanim,.tcp8-tailanim,.tcp8-fast,.tcp8-waitanim{animation:none;opacity:1}}
</style>
<rect class="tcp8-card" x="10" y="10" width="440" height="520" rx="14"/><rect class="tcp8-card" x="470" y="10" width="440" height="520" rx="14"/>
<text class="tcp8-title" x="230" y="42">A. Middle loss: later packets expose the hole</text><text class="tcp8-title" x="690" y="42">B. Tail loss: silence needs a probe</text>
<g class="tcp8-midanim"><text class="tcp8-lbl" x="50" y="75">sender</text><text class="tcp8-lbl" x="360" y="75">receiver</text><line class="tcp8-axis" x1="80" y1="88" x2="80" y2="470"/><line class="tcp8-axis" x1="390" y1="88" x2="390" y2="470"/>
<line class="tcp8-flow" x1="82" y1="115" x2="388" y2="145"/><text class="tcp8-small" x="96" y="108">segment 1</text><line class="tcp8-loss" x1="82" y1="160" x2="230" y2="176"/><text class="tcp8-small" x="96" y="154">segment 2 lost</text><circle class="tcp8-bad" cx="238" cy="177" r="7"/>
<line class="tcp8-flow" x1="82" y1="205" x2="388" y2="235"/><text class="tcp8-small" x="96" y="199">segment 3</text><line class="tcp8-flow" x1="82" y1="250" x2="388" y2="280"/><text class="tcp8-small" x="96" y="244">segment 4</text><line class="tcp8-flow" x1="82" y1="295" x2="388" y2="325"/><text class="tcp8-small" x="96" y="289">segment 5</text>
<line class="tcp8-ack" x1="388" y1="250" x2="82" y2="275"/><line class="tcp8-ack" x1="388" y1="295" x2="82" y2="320"/><line class="tcp8-ack" x1="388" y1="340" x2="82" y2="365"/><text class="tcp8-lbl" x="244" y="252">ACK 2</text><text class="tcp8-lbl" x="244" y="297">ACK 2</text><text class="tcp8-lbl" x="244" y="342">ACK 2</text>
<line class="tcp8-loss tcp8-fast" x1="82" y1="390" x2="388" y2="420"/><text class="tcp8-lbl" x="96" y="388">fast retransmit 2</text><text class="tcp8-small" x="96" y="454">three duplicate ACKs create evidence within +RTT</text></g>
<g class="tcp8-tailanim"><text class="tcp8-lbl" x="510" y="75">sender</text><text class="tcp8-lbl" x="820" y="75">receiver</text><line class="tcp8-axis" x1="540" y1="88" x2="540" y2="470"/><line class="tcp8-axis" x1="850" y1="88" x2="850" y2="470"/>
<line class="tcp8-flow" x1="542" y1="115" x2="848" y2="145"/><text class="tcp8-small" x="556" y="108">segment 1</text><line class="tcp8-flow" x1="542" y1="160" x2="848" y2="190"/><text class="tcp8-small" x="556" y="154">segment 2</text><line class="tcp8-flow" x1="542" y1="205" x2="848" y2="235"/><text class="tcp8-small" x="556" y="199">segment 3</text><line class="tcp8-flow" x1="542" y1="250" x2="848" y2="280"/><text class="tcp8-small" x="556" y="244">segment 4</text><line class="tcp8-loss" x1="542" y1="295" x2="690" y2="310"/><text class="tcp8-small" x="556" y="289">segment 5 lost</text><circle class="tcp8-bad" cx="698" cy="311" r="7"/>
<rect class="tcp8-gap tcp8-waitanim" x="565" y="330" width="260" height="58" rx="10"/><text class="tcp8-lbl" x="640" y="354">idle gap</text><text class="tcp8-small" x="590" y="375">no later arrival, no duplicate ACK</text>
<line class="tcp8-probe tcp8-waitanim" x1="542" y1="410" x2="848" y2="440"/><text class="tcp8-lbl" x="556" y="408">TLP probe at PTO</text><text class="tcp8-small" x="556" y="470">probe creates ACK evidence; RTO remains fallback</text></g>
<circle class="tcp8-dot" cx="230" cy="508" r="6"/><text class="tcp8-small" x="245" y="512">feedback already exists</text><circle class="tcp8-warn" cx="650" cy="508" r="6"/><text class="tcp8-small" x="665" y="512">feedback must be created</text>
</svg>
<figcaption>Watch the same loss position change the recovery signal: duplicate ACKs trigger fast retransmit in the middle, but the tail needs TLP or RTO.</figcaption>
</figure>

[RFC 5681 Section 3.2](https://www.rfc-editor.org/rfc/rfc5681.html#section-3.2) specifies classic fast retransmit using three duplicate ACKs without intervening ACKs that advance `SND.UNA`. Each later out-of-order segment prompts the receiver to repeat the same cumulative ACK. Three such reports are treated as evidence that the missing segment was probably lost rather than merely delayed by modest reordering.

Why three? The threshold is a conservative disambiguation rule in the classic algorithm, not a law of physics. One duplicate ACK can result from reordering. Several duplicate ACKs show that later packets are continuing to leave the network and reach the receiver while one sequence range remains absent. RFC 5681 then allows repair without waiting for the retransmission timer.

Fast retransmit and fast recovery are related but distinct. Fast retransmit chooses a missing segment to send again. Fast recovery controls how transmission proceeds while ACK evidence continues. RFC 5681 connects duplicate ACKs to the ACK clock: each duplicate ACK proves that some segment arrived and left the network, so the sender can continue carefully under a reduced congestion window instead of falling all the way back to timeout recovery.

The congestion response is not the focus of this post. [Flow control versus congestion control](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe) separates the receiver's advertised window from the sender's congestion window, and [CUBIC, BBR, and what changing congestion control actually does](/blog/software-development/networking/congestion-control-cubic-bbr-and-what-changing-it-actually-does) covers the control algorithms. Reliability supplies the loss evidence that those mechanisms must react to.

### Worked middle-loss example

Assume a sender emits five full-sized segments and segment 2 is lost. Segments 3, 4, and 5 arrive. Each is above the same gap, so the receiver immediately repeats the ACK for the first byte of segment 2. After the third duplicate ACK, the classic trigger exists. The sender can retransmit segment 2 before RTO expiration.

Now move the loss from segment 2 to segment 5. Segments 1 through 4 arrive in order and advance the cumulative ACK. Segment 5 vanishes. No segment 6 exists because the application response has ended. The receiver has no out-of-order arrival to report, so it sends no duplicate ACK caused by this hole. The connection becomes quiet with one outstanding range.

This is the tail-loss problem. A short response is especially exposed because it may not have enough later segments to produce three duplicate ACKs even when the loss is not literally the last segment. An application-limited sender can have a large congestion window but no new data to transmit. Capacity is available. Evidence is not.

### Retransmission ambiguity

When an ACK arrives after retransmission, the sender may not know whether it acknowledges the original transmission or the retransmission. Using that ambiguous observation as an RTT sample can corrupt the timer. [RFC 6298 Section 3](https://www.rfc-editor.org/rfc/rfc6298.html#section-3) requires Karn's algorithm: RTT samples must not be made from retransmitted segments unless timestamps remove the ambiguity.

This is an easy capture-reading trap. An ACK arriving quickly after a retransmission does not prove the retransmission crossed the path that quickly. The original may have been delayed rather than lost. Duplicate Selective Acknowledgment can provide evidence of a spurious retransmission, but the timing attribution still requires care.

### Reordering can imitate loss

Duplicate ACKs are evidence of a gap, not proof of a drop. Reordering can create the same pattern. That is the trade-off behind any early loss detector. Act too soon and retransmit data that is merely late. Act too slowly and leave the application stalled.

RACK makes this trade-off explicit with a reordering window. Classic duplicate-ACK counting embeds it indirectly in the threshold. In both cases, the operational question is not "did I see a retransmission?" It is "what evidence and tolerance caused the sender to classify this range as lost?"

## 5. RTO is a guarded estimate, not a fixed retry delay

**Senior rule:** inspect `rto`, `rtt`, and `rttvar` together. A timer without its uncertainty term is only half the explanation.

![RTO calculation showing how smoothed RTT and RTT variation react to stable and jittery samples](/imgs/blogs/reliability-sequence-numbers-acks-retransmits-and-rto-4.webp)

The retransmission timeout is the backstop. It must fire when ACK-driven mechanisms cannot establish loss, yet it must avoid firing on every ordinary RTT fluctuation. [RFC 6298, published June 2011](https://www.rfc-editor.org/rfc/rfc6298.html), specifies the standard estimator.

Before any RTT measurement, a sender should use an initial RTO of 1 second. After the first RTT sample $R$, it initializes:

$$
SRTT \leftarrow R
$$

$$
RTTVAR \leftarrow \frac{R}{2}
$$

$$
RTO \leftarrow SRTT + \max(G, K \times RTTVAR)
$$

Here $SRTT$ is smoothed round-trip time, $RTTVAR$ is smoothed RTT variation, $G$ is clock granularity, and $K=4$. For later sample $R'$, RFC 6298 requires updating variation before the smoothed mean:

$$
RTTVAR \leftarrow (1-\beta)RTTVAR + \beta\left|SRTT-R'\right|
$$

$$
SRTT \leftarrow (1-\alpha)SRTT + \alpha R'
$$

with $\alpha=\frac{1}{8}$ and $\beta=\frac{1}{4}$. The RTO is then recomputed from $SRTT + \max(G,4RTTVAR)$. These are protocol formulas from RFC 6298, not an explanatory model invented for this article.

RFC 6298 also says a computed RTO below 1 second should be rounded up to 1 second. Real operating systems may implement later standards, additional mechanisms, and different internal timer behavior while remaining interoperable, so do not infer a host's precise timeout solely from the RFC floor. Measure the socket and capture.

### Worked estimator example

Take a first eligible RTT sample of 80 ms and assume timer granularity $G$ is smaller than the variation term. The raw formula gives:

$$
SRTT = 80\ \text{ms},\quad RTTVAR = 40\ \text{ms}
$$

$$
RTO_{raw}=80+4\times40=240\ \text{ms}
$$

The subscript matters. `240 ms` is the estimator output before applying RFC 6298's recommended 1-second lower bound. It is not a claim that a conforming connection will expose a 240 ms RTO.

Now take a second sample of 80 ms. Update variation first:

$$
RTTVAR = \frac{3}{4}\times40 + \frac{1}{4}\times|80-80| = 30\ \text{ms}
$$

$$
SRTT = \frac{7}{8}\times80 + \frac{1}{8}\times80 = 80\ \text{ms}
$$

The raw RTO becomes $80 + 4\times30 = 200$ ms. Stability narrows the uncertainty margin.

Now let the next eligible sample jump to 160 ms:

$$
RTTVAR = \frac{3}{4}\times30 + \frac{1}{4}\times|80-160| = 42.5\ \text{ms}
$$

$$
SRTT = \frac{7}{8}\times80 + \frac{1}{8}\times160 = 90\ \text{ms}
$$

The raw RTO becomes $90 + 4\times42.5 = 260$ ms. The mean moved by only 10 ms, but the uncertainty term grew enough to widen the timer by 60 ms. This is the intended behavior: jitter matters because a premature timeout creates needless retransmission and unnecessary congestion response.

| Eligible RTT sample | SRTT after update | RTTVAR after update | Raw formula RTO | Standards handling | Source |
| ---: | ---: | ---: | ---: | --- | --- |
| 80 ms, first | 80 ms | 40 ms | 240 ms | Apply RFC minimum rule | Derived from RFC 6298 equations |
| 80 ms, second | 80 ms | 30 ms | 200 ms | Apply RFC minimum rule | Derived from RFC 6298 equations |
| 160 ms, third | 90 ms | 42.5 ms | 260 ms | Apply RFC minimum rule | Derived from RFC 6298 equations |

### Timeout backoff protects the network

When the RTO expires, RFC 6298 says the sender retransmits the earliest unacknowledged segment, doubles the RTO, and restarts the timer. Exponential backoff prevents a broken or congested path from being hammered at a fixed rate. It also means repeated timeout loss creates a distinctive capture: widening intervals between retransmissions.

Do not use an application deadline shorter than the transport's recovery behavior without understanding the consequence. The application may abandon a request while the kernel still owns an established socket and retransmits below it. Conversely, waiting for every TCP retry can exceed the service's usefulness window. The right application timeout is a product decision informed by the transport path, not a replacement for transport diagnosis.

### Read the live estimator

On Linux, `ss -tin` exposes transport information for live sockets. Field availability and formatting vary by iproute2 and kernel version, but commonly include `rtt`, RTT variation after the slash, `rto`, congestion window, unacknowledged segments, SACKed segments, and retransmission information.

```bash
sudo ip netns exec c ss -tin \
  '( dst 10.77.0.2 and dport = :8080 )'
```

Sample only while the connection exists. For a short client that opens and closes quickly, add a debug hold flag to the lab client or run a longer transfer. Do not invent a missing socket by quoting output from another connection.

## 6. RACK and TLP change what counts as evidence

**Senior rule:** modern Linux recovery is not explained completely by counting duplicate ACKs.

![Decision graph showing RACK time-based inference, TLP feedback creation, and RTO fallback](/imgs/blogs/reliability-sequence-numbers-acks-retransmits-and-rto-5.webp)

Classic fast retransmit asks how many later segments arrived above a hole. Recent Acknowledgment, or RACK, asks a time-order question: has a sufficiently newer transmission been delivered while this older transmission remains unacknowledged beyond a reordering window?

[RFC 8985, published February 2021](https://www.rfc-editor.org/rfc/rfc8985.html), standardizes RACK-TLP. RACK records transmit times for segments, including retransmissions. ACK and SACK feedback identify recently delivered data. Older unacknowledged transmissions can then be marked lost after the reordering allowance. This works for patterns that defeat a simple duplicate-ACK count, including lost retransmissions and application-limited flights.

RACK still needs ACK feedback. A silent tail provides none. Tail Loss Probe, or TLP, creates feedback before the RTO when conditions allow. It sends at most one probe beyond the congestion window limit at a time. The probe can be new data if available or a retransmission selected by the algorithm. If the probe reaches the receiver, the resulting ACK gives RACK or ordinary ACK processing new evidence about the tail.

RFC 8985's PTO calculation starts with `PTO = 2 * SRTT` when SRTT is available. If the flight contains one segment, it adds the maximum ACK delay. Without an SRTT estimate, PTO is 1 second. In every case, the probe must not be scheduled later than the current RTO expiration.

This pseudocode is a compact restatement of RFC 8985 Section 7.2. It is not a universal Linux timer printout. The RFC defines the algorithmic relationship. An implementation maps that relationship onto its timers, ACK behavior, and kernel state.

### The 100-segment tail example

RFC 8985 gives a concrete scenario: a sender transmits 100 new data segments and the last three are lost. Traditional recovery reaches RTO because there are no later arrivals to produce enough evidence. The RFC explains that this transfer can take three RTTs plus one RTO and reset the congestion window. With TLP, a probe after two round trips can solicit an ACK; RACK can then identify the earlier missing tail segments, recover in four RTTs in the stated example, and avoid the full timeout response.

Those numbers belong to the RFC's scenario, not to every flow. Their value is comparative. Moving the same loss three segments earlier would have allowed fast recovery because later packets existed. The location of loss inside the flight changes the evidence path and therefore the latency.

### RACK's reordering window is a safety margin

If any older unacknowledged transmission were declared lost immediately when a newer one was acknowledged, ordinary reordering would cause spurious retransmissions. RACK therefore maintains a reordering window. RFC 8985 derives a minimum based on the minimum RTT and adjusts behavior from reordering observations and recovery state.

This creates a useful debugging distinction:

- Duplicate-ACK recovery is primarily packet-count evidence.
- RACK recovery is primarily transmit-time order plus acknowledgment evidence.
- TLP is feedback generation for a sparse or silent tail.
- RTO is the conservative backstop when earlier mechanisms do not complete recovery.

These paths can interact in one connection. A TLP probe can generate the ACK that lets RACK mark another packet lost. A probe can itself be lost, leaving RTO as the fallback. A reordered packet can arrive after a retransmission and produce DSACK evidence that recovery was spurious.

### Linux exposes aggregate corroboration

The [Linux kernel SNMP counter documentation](https://docs.kernel.org/networking/snmp_counter.html) describes several useful namespace-wide counters:

| Counter | Meaning in Linux documentation | Scope caution | Source |
| --- | --- | --- | --- |
| `TcpExtTCPLossProbes` | TLP probe packets sent | Namespace aggregate, not one socket | [Linux SNMP counter documentation, accessed 2026-09-29](https://docs.kernel.org/networking/snmp_counter.html) |
| `TcpExtTCPLossProbeRecovery` | Loss detected and recovered by TLP | Namespace aggregate | [Linux SNMP counter documentation, accessed 2026-09-29](https://docs.kernel.org/networking/snmp_counter.html) |
| `TcpExtTCPFastRetrans` | Retransmission attempted outside Loss state | Namespace aggregate | [Linux SNMP counter documentation, accessed 2026-09-29](https://docs.kernel.org/networking/snmp_counter.html) |
| `TcpExtTCPSackRecovery` | Entry into Recovery state using SACK | Namespace aggregate | [Linux SNMP counter documentation, accessed 2026-09-29](https://docs.kernel.org/networking/snmp_counter.html) |
| `TcpExtTCPSlowStartRetrans` | Retransmission attempted in Loss state | Namespace aggregate | [Linux SNMP counter documentation, accessed 2026-09-29](https://docs.kernel.org/networking/snmp_counter.html) |

Use `nstat -az` before and after a controlled experiment and subtract. On a production host, unrelated connections can change the same counters, so pair them with a filtered capture or per-socket `ss` data. An increase proves that the namespace executed a path. It does not by itself attribute that event to your request.

## 7. Why the final response packet is disproportionately expensive

**Senior rule:** tail latency is often an evidence drought, not a bandwidth shortage.

The final packet has three properties that make its loss special. First, no later data packet naturally arrives to expose the gap. Second, the application cannot declare the message complete without those bytes. Third, the connection may become application-limited immediately after sending it, so a large congestion window does not create additional packets.

Imagine a server sends an HTTP response body in six segments. The first five arrive. The sixth carries the last body bytes. At the receiver, the stream is contiguous up to the start of segment six. There is no out-of-order queue because nothing follows it. At the sender, one range remains outstanding. Both endpoints may otherwise be idle.

The application-visible completion time can be approximated with this explanatory model:

$$
T_{complete} \approx T_{last\ good\ byte} + T_{loss\ detection} + RTT_{repair}
$$

For middle loss, $T_{loss\ detection}$ may be the time needed for later packets and three duplicate ACKs or RACK's reordering window. For tail loss, it may be the TLP PTO. Without effective probe recovery, it can reach RTO. The model is intentionally simplified. It excludes delayed application scheduling, ACK loss, retransmission loss, and congestion-control pacing.

### A deadline can turn one lost segment into duplicate work

Suppose a service has a 300 ms client deadline and ordinarily completes in 90 ms. A tail loss adds a recovery delay large enough to cross the deadline. The client cancels or retries. The server may already have committed the operation, and the missing bytes may only be the response. A transport event has now become an application consistency problem.

This is why retry safety requires idempotency, deduplication, and budgets. [Resilience patterns for timeouts and retries](/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads) covers that policy layer. The networking contribution is to reveal whether a deadline spike came from tail recovery rather than server execution.

### More bandwidth does not manufacture ACK evidence

A common wrong turn is to raise socket buffers or choose a faster link. Those changes may improve throughput for a capacity-bound transfer. They do not make a finished application write contain later segments. A connection with room for 100 more packets and zero bytes left to send is application-limited. The missing resource is feedback.

Likewise, enlarging the receive window does not repair a tail loss. The receive window says how much data the receiver is willing to accept. The congestion window says how much the network path may safely have in flight. Neither is the loss detector. The distinction is developed fully in [two windows, one pipe](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe).

### Bufferbloat can stretch every recovery signal

Queueing increases RTT samples, delays duplicate ACKs, delays probe ACKs, and widens the time base on which recovery operates. A packet need not be dropped for an oversized queue to make loss recovery look slow. Inspect queue delay alongside retransmissions. [Bufferbloat and the latency you added yourself](/blog/software-development/networking/bufferbloat-queueing-and-the-latency-you-added-yourself) owns that mechanism.

## 8. A public case: RACK-TLP moved tail recovery into RTT time

**Senior rule:** standards history is useful when it explains which production failure pattern forced the algorithm to change.

The case is the development and standardization of RACK-TLP by Yuchung Cheng, Neal Cardwell, Nandita Dukkipati, and Priyaranjan Jha of Google. The direct public source is [RFC 8985](https://www.rfc-editor.org/rfc/rfc8985.html), published by the IETF in February 2021 as a Standards Track document. The source owner is the IETF, and the named authors describe the algorithm and motivating failure patterns.

The user-visible symptom is disproportionate completion delay for application-limited transfers, tail losses, and lost retransmissions. The trigger is missing data. The contributing condition is sparse ACK feedback, especially when the lost packets are the last packets of a flight. The blast-radius multiplier is RTO recovery, which both adds timer delay and resets the congestion window more severely than successful fast recovery.

The RFC's 100-segment example gives the mechanism without asking us to trust a vague production anecdote. Losing the final three segments prevents later data from triggering classic recovery. TLP sends a probe to solicit feedback. RACK uses that feedback and transmit-time ordering to infer which older segments are lost. If the probe succeeds, recovery remains on an RTT-scale path rather than waiting for RTO. If the probe is also lost, RTO remains the safety net.

The standard also identifies cases where duplicate-ACK counting is weaker: application-limited flights, lost retransmissions, and reordering. This is not a claim that RACK eliminates every timeout or that every timeout is a bug. A complete reliability design still needs a conservative backstop. The change is that the sender can use richer evidence earlier.

The transferable control is diagnostic and architectural:

1. Preserve packet timing and ACK/SACK detail when investigating tail spikes.
2. Correlate filtered captures with `TCPLossProbes`, `TCPFastRetrans`, and timeout-related counters.
3. Keep application deadlines and retries safe even when transport recovery is working as designed.
4. Treat kernel and middlebox diversity as part of the path. Confirm negotiated options and observed behavior rather than assuming the RFC's newest algorithm is active end to end.

This case passes the honesty boundary because the mechanism, publication date, authorship, example values, and algorithm behavior come directly from RFC 8985. We do not attach invented fleet percentages or latency wins to Google. The RFC supplies a standards story and a reproducible mechanism, not permission to manufacture a company benchmark.

## 9. Diagnose the recovery path, not just the retransmission count

**Senior rule:** one retransmission counter is a symptom. Packet order plus path-specific counters is a diagnosis.

![Decision tree for distinguishing fast retransmit, RACK or TLP recovery, RTO, and reordering](/imgs/blogs/reliability-sequence-numbers-acks-retransmits-and-rto-6.webp)

Start with a narrowly filtered capture. Identify the first retransmitted sequence range. Then walk backward from that retransmission to the evidence available at the sender.

### Step 1: establish direction and byte range

Use endpoints and ports, not the analyzer's conversational labels alone. Confirm that the retransmitted segment overlaps an earlier transmitted range. Hardware offload can make captures on the sending host look unusual because segmentation may occur after the capture point. Capturing on the peer or disabling offloads in an isolated lab can clarify packetization, but do not disable production offloads casually.

```bash
tshark -r tail-loss.pcapng \
  -Y 'tcp.port == 8080 && (tcp.analysis.retransmission || tcp.analysis.fast_retransmission || tcp.analysis.spurious_retransmission)' \
  -T fields \
  -e frame.number -e frame.time_relative \
  -e ip.src -e tcp.seq -e tcp.len -e tcp.ack \
  -e _ws.col.Info
```

Analyzer flags are heuristics derived from the packets visible at that capture point. If the capture missed packets, their conclusion may be wrong. Always inspect the raw sequence and ACK fields around the event.

### Step 2: look for advancing SACK evidence or repeated cumulative ACKs

Three repeated ACKs with later sequence ranges arriving above one hole support classic fast retransmit. SACK blocks that advance across newer ranges show a scoreboard with useful evidence. A retransmission after newer data is acknowledged, without exactly three simple duplicate ACKs, can be consistent with RACK or SACK-based recovery.

If no later data arrived, stop trying to explain the event with a duplicate-ACK threshold. Look for a probe near the tail and the interval from the last original transmission. Then compare namespace counters.

### Step 3: distinguish probe from timeout

Snapshot counters inside the namespace before the workload:

```bash
sudo ip netns exec s nstat -az \
  | rg 'TcpExt(TCPLossProbes|TCPLossProbeRecovery|TCPFastRetrans|TCPSackRecovery|TCPSlowStartRetrans|TCPTimeouts)'
```

Repeat after one isolated run and subtract values. `TCPLossProbes` increasing points to probe activity. `TCPFastRetrans` or `TCPSackRecovery` increasing supports ACK-driven recovery. `TCPTimeouts` or retransmission in Loss state supports timeout recovery. Counter definitions and availability vary with kernel version, so record `uname -r` and keep the full `nstat -az` output rather than parsing a field that may not exist.

### Step 4: check for reordering and spurious recovery

If the original packet later arrives, the receiver may report a duplicate range through DSACK. Linux also exposes reordering-related counters. This can reveal that the sender's loss detector acted on reordering rather than actual loss. The fix may be in path consistency, bonding, ECMP behavior, or the reordering tolerance, not in making retransmission more aggressive.

### A practical evidence matrix

| Capture and counter pattern | Strongest hypothesis | What would weaken it | Next measurement | Source |
| --- | --- | --- | --- | --- |
| Same ACK repeats while later segments arrive; fast retransmission follows | Classic duplicate-ACK or SACK recovery | Capture misses reverse-path ACKs | Inspect exact SACK blocks and `TCPFastRetrans` delta | RFC 5681 and Linux counter docs |
| Newer transmission acknowledged; older one retransmitted after reordering allowance | RACK inference | Timestamp or packet capture order is unreliable | Correlate ACK time, transmit time, and kernel version | RFC 8985 |
| Tail becomes silent; one probe is sent before RTO; ACK unlocks recovery | TLP | Probe is ordinary new application data | Check `TCPLossProbes` delta and payload range | RFC 8985 and Linux counter docs |
| Long idle interval; oldest range retransmitted; later intervals back off | RTO recovery | Capture began after earlier probes | Inspect `rto` in `ss`, full capture, `TCPTimeouts` delta | RFC 6298 |
| Retransmission followed by DSACK or late original | Reordering or spurious retransmission | Duplicate came from capture artifact | Capture at receiver and inspect DSACK | RFC 2883 and packet evidence |

### Do not collapse the layers

An application trace can tell you the request stalled. A server trace can tell you handler work ended. A socket snapshot can tell you unacknowledged data and timer state. A packet capture can show sequence and ACK evidence. A kernel counter can corroborate a recovery path over an interval. None replaces the others.

For a durable incident workflow, integrate these boundaries into [observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design). Store connection identifiers carefully, protect packet data, and avoid turning high-cardinality per-flow details into an always-on metrics bill. Captures can contain credentials, tokens, personal data, and payloads. Filter tightly and handle them as sensitive artifacts.

## 10. Design consequences for services

Reliability below the application is valuable, but it does not make request semantics reliable by itself. The transport guarantees ordered byte delivery while the connection survives. It does not tell the client whether a timed-out mutation committed, whether the response was lost after commit, or whether a retry is safe.

### Match deadlines to useful work, then make retries safe

If the business value expires at 500 ms, allowing a TCP connection to retry for minutes does not make the request useful. If the application aborts at 100 ms on an 80 ms path, ordinary recovery may never get a chance. Set deadlines from end-to-end service objectives and observed network behavior, then make retryable operations idempotent or deduplicated.

Do not "fix" tail loss by simply lengthening every deadline. That can increase queue occupancy, retain memory, and amplify retries elsewhere. Likewise, shortening every deadline can convert recoverable transport loss into retry storms. The correct adjustment is workload-specific and belongs with concurrency limits, backoff, and admission control.

### Keep responses resumable when payloads are large

For large downloads, range requests, chunk checksums, and application-level resume can reduce the cost of connection failure. TCP will repair packet loss within a live connection, but it cannot resume an application object across an unrelated replacement connection without protocol support.

For small RPC responses, object-level resume is usually excessive. Idempotent retry and deduplication matter more. The transport diagnosis still matters because a burst of tail-loss recovery can predict deadline pressure before application error rates explode.

### Separate loss recovery from congestion-control tuning

Changing from CUBIC to BBR changes how the sender estimates and controls its sending rate. It does not remove sequence numbers, ACKs, RTO, or the need to repair lost bytes. A different congestion controller can change flight shape and timing, which influences available evidence, but it is not a replacement for loss detection.

Similarly, increasing socket buffers can allow a larger flight. That may produce enough later segments for ACK-driven recovery, but it can also add queueing and memory pressure. Treat any such effect as a second-order consequence, not as the primary reliability mechanism.

### Preserve the final-byte signal in telemetry

Many service dashboards record time to first byte and total duration but not the interval from last useful progress to completion. A tail stall hides inside the latter. If your client permits progress callbacks, record response bytes over time in sampled diagnostics. At the host, correlate retransmission counters with request latency histograms over the same bounded interval.

A useful diagnostic ratio is an explanatory metric, not a standards formula:

$$
tail\ stall\ share = \frac{T_{complete}-T_{last\ progress}}{T_{complete}-T_{start}}
$$

Define "progress" precisely for the client library before using this ratio. Buffering above TCP can make application reads coarser than packet arrival. The metric is valuable for comparing incidents, not for identifying the transport path by itself.

## Run it yourself

### Question

Can a controlled lossy path produce both ACK-driven recovery and tail-loss probe or timeout evidence, and can we distinguish the paths using packet sequence fields plus Linux namespace counters?

### Preconditions

Use Linux with `iproute2`, `tc`, `tcpdump`, `tshark`, `nstat`, `curl`, and the series `netlab` binaries. Namespace and qdisc changes require root or `CAP_NET_ADMIN`; packet capture normally requires root or `CAP_NET_RAW`. On macOS, run this inside the privileged Linux VM described in [the series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). Do not apply these `tc` commands to an unspecified production interface.

The canonical topology must already exist: namespace `c` owns `c0` at `10.77.0.1/30`, namespace `s` owns `s0` at `10.77.0.2/30`, and `netserver` listens on `10.77.0.2:8080`. This experiment uses random loss to collect examples. It does not claim that every run loses the literal last segment.

Run this preflight first:

```bash
set -euo pipefail

ip netns list | rg '^(c|s)\b'
ip -n c -br addr show dev c0
ip -n s -br addr show dev s0
ip -n c route get 10.77.0.2
ip -n s route get 10.77.0.1
ip netns exec c ping -c 5 -W 1 10.77.0.2
ip netns exec c tc -s qdisc show dev c0
ip netns exec s tc -s qdisc show dev s0
ip netns exec s ss -lnt '( sport = :8080 )'
ip netns exec c tshark --version | sed -n '1p'
ip netns exec c nstat --version 2>&1 | sed -n '1p'
```

Read: both interfaces must be `UP`, `ip route get` must select `c0` or `s0`, ping should report five received packets before impairment, and `ss` must show a listener on `10.77.0.2:8080` or `*:8080`.

Expected: on an unloaded local Linux VM, the baseline namespace RTT is commonly below 2 ms. That is a reproducible lab expectation, not a protocol limit. Virtualization and host scheduling can raise it. Record the measured value rather than forcing the environment to match.

### Baseline

The baseline adds 40 ms one-way delay on both egress interfaces, producing an intended RTT near 80 ms, with no configured random loss. It records a bounded capture and counter snapshots.

```bash
set -euo pipefail

SLUG=reliability-sequence-numbers-acks-retransmits-and-rto
OUT=netlab/out/$SLUG
mkdir -p "$OUT"

sudo ip netns exec c tc qdisc replace dev c0 root netem delay 40ms
sudo ip netns exec s tc qdisc replace dev s0 root netem delay 40ms

sudo ip netns exec c ping -c 10 -W 1 10.77.0.2 | tee "$OUT/baseline-ping.txt"
sudo ip netns exec s nstat -az > "$OUT/baseline-nstat-before.txt"

sudo ip netns exec c timeout 20 tcpdump -i c0 -nn -s 128 \
  -w "/tmp/$SLUG-baseline.pcap" \
  'tcp port 8080' &
CAP_PID=$!
sleep 1

for i in $(seq 1 20); do
  sudo ip netns exec c curl -fsS -o /dev/null \
    -w '%{time_total}\n' \
    'http://10.77.0.2:8080/echo?bytes=65536'
done | tee "$OUT/baseline-time-total.txt"

wait "$CAP_PID" || true
sudo mv "/tmp/$SLUG-baseline.pcap" "$OUT/baseline.pcap"
sudo chown "$(id -u):$(id -g)" "$OUT/baseline.pcap"
sudo ip netns exec s nstat -az > "$OUT/baseline-nstat-after.txt"

tshark -r "$OUT/baseline.pcap" \
  -Y 'tcp.analysis.retransmission || tcp.analysis.fast_retransmission' \
  -T fields -e frame.number -e frame.time_relative \
  -e tcp.seq -e tcp.len -e tcp.ack
```

Read: verify `ping` reports an average RTT roughly in the 75–95 ms range. Inspect `time_total` for the request distribution. The final `tshark` command should normally print no retransmission rows because the qdisc adds delay only.

Expected: all 20 requests should succeed, with most completion times in a compact band determined by the 80 ms path, request size, server behavior, and connection policy. Exact times are intentionally not promised because the endpoint implementation and whether `curl` reuses a connection affect the flight structure.

### Apply one change

Add 5 percent independent random loss to both directions while preserving 40 ms one-way delay. Random loss is used because it can produce middle, tail, data, and ACK loss across repeated trials. We classify observed events after capture instead of pretending a particular packet will be lost.

```bash
set -euo pipefail

sudo ip netns exec c tc qdisc replace dev c0 root netem delay 40ms loss 5%
sudo ip netns exec s tc qdisc replace dev s0 root netem delay 40ms loss 5%

sudo ip netns exec c tc -s qdisc show dev c0
sudo ip netns exec s tc -s qdisc show dev s0
```

Read: each `tc` output must show `netem delay 40ms loss 5%`. The loss model is independent random loss as implemented by the host's `netem`; it is not a model of every production path.

### Compare

Run enough bounded requests to make recovery events likely, capture only port 8080, and snapshot the server namespace counters before and after. The server is the data sender for response traffic, so its loss-recovery counters are the relevant aggregate for that direction.

```bash
set -euo pipefail

SLUG=reliability-sequence-numbers-acks-retransmits-and-rto
OUT=netlab/out/$SLUG

sudo ip netns exec s nstat -az > "$OUT/loss-nstat-before.txt"
sudo ip netns exec c timeout 45 tcpdump -i c0 -nn -s 160 \
  -w "/tmp/$SLUG-loss.pcap" \
  'tcp port 8080' &
CAP_PID=$!
sleep 1

for i in $(seq 1 100); do
  sudo ip netns exec c curl --max-time 5 -fsS -o /dev/null \
    -w '%{http_code} %{time_total}\n' \
    'http://10.77.0.2:8080/echo?bytes=65536' || true
done | tee "$OUT/loss-time-total.txt"

wait "$CAP_PID" || true
sudo mv "/tmp/$SLUG-loss.pcap" "$OUT/loss.pcap"
sudo chown "$(id -u):$(id -g)" "$OUT/loss.pcap"
sudo ip netns exec s nstat -az > "$OUT/loss-nstat-after.txt"

tshark -r "$OUT/loss.pcap" \
  -Y 'tcp.analysis.retransmission || tcp.analysis.fast_retransmission || tcp.analysis.out_of_order || tcp.analysis.duplicate_ack' \
  -T fields \
  -e tcp.stream -e frame.number -e frame.time_relative \
  -e ip.src -e tcp.seq -e tcp.len -e tcp.ack \
  -e tcp.analysis.duplicate_ack_num \
  > "$OUT/recovery-events.tsv"

paste \
  <(rg 'TcpExt(TCPLossProbes|TCPLossProbeRecovery|TCPFastRetrans|TCPSackRecovery|TCPSlowStartRetrans|TCPTimeouts)' "$OUT/loss-nstat-before.txt") \
  <(rg 'TcpExt(TCPLossProbes|TCPLossProbeRecovery|TCPFastRetrans|TCPSackRecovery|TCPSlowStartRetrans|TCPTimeouts)' "$OUT/loss-nstat-after.txt") \
  | tee "$OUT/recovery-counter-pairs.txt"

sed -n '1,80p' "$OUT/recovery-events.tsv"
```

Read: in `recovery-events.tsv`, group rows by `tcp.stream`. A middle-loss candidate has later sequence ranges followed by repeated identical ACK values and a retransmission of the hole. A tail candidate has the final original range outstanding with no later response data before a probe or retransmission. In `recovery-counter-pairs.txt`, subtract the left value from the right value for each named counter.

Expected: with 5 percent independent loss over 100 transfers, at least one recovery event is likely, but no particular counter is guaranteed to move because loss positions, ACK loss, kernel version, offload, and implementation vary. If no retransmission appears, repeat the treatment up to three times and report that outcome rather than raising loss until the experiment becomes a different workload. Successful runs commonly show a mix of completion times near the baseline band and slower outliers separated by one or more recovery intervals.

The discriminating result is qualitative and packet-based: middle holes have later arrivals that create ACK evidence; silent tails lack those arrivals. A `TCPLossProbes` increase corroborates TLP activity. A `TCPFastRetrans` or `TCPSackRecovery` increase corroborates ACK-driven recovery. A `TCPTimeouts` increase plus a long quiet interval supports RTO. Because counters are namespace-wide, the isolated namespace is what makes the deltas attributable here.

### Reset

Remove only the qdiscs added to the canonical lab interfaces. Captures and reports remain under this post's output directory for inspection.

```bash
set -euo pipefail

sudo ip netns exec c tc qdisc del dev c0 root 2>/dev/null || true
sudo ip netns exec s tc qdisc del dev s0 root 2>/dev/null || true

sudo ip netns exec c tc -s qdisc show dev c0
sudo ip netns exec s tc -s qdisc show dev s0
sudo ip netns exec c ping -c 5 -W 1 10.77.0.2
```

Read: neither interface should retain the configured `netem` qdisc, and RTT should return toward the preflight range.

### Production translation

Do not inject loss on a production interface. Use read-only commands and a narrowly scoped capture approved for the environment:

```bash
uname -r
ss -tin '( dport = :8080 or sport = :8080 )'
nstat -az | rg 'TcpExt(TCPLossProbes|TCPLossProbeRecovery|TCPFastRetrans|TCPSackRecovery|TCPSlowStartRetrans|TCPTimeouts)'
sudo timeout 30 tcpdump -i any -nn -s 128 -C 20 -W 3 \
  -w /secure/path/tcp-recovery-%02d.pcap \
  'host 203.0.113.10 and tcp port 8080'
```

Replace the documentation address and port with an explicitly approved target. A capture can include sensitive headers and payload. Use access controls, rotation, minimal snap length, retention limits, and incident approval. On a busy host, counter changes are only background evidence because they include unrelated sockets.

### Four traps when interpreting the lab

First, a retransmission label from `tshark` is an inference based on the packets in that file. If capture began late, dropped packets, or observed traffic after segmentation or receive offload, the analyzer can classify an event incorrectly. The byte ranges and timestamps are the evidence. The generated `tcp.analysis.*` fields are a convenient index into that evidence.

Second, the 5 percent loss setting describes independent random impairment at each configured qdisc. It does not mean 5 percent of completed requests fail, nor does it mean exactly 5 percent of visible TCP segments disappear. Retransmissions, ACKs, handshake packets, and packets in both directions all encounter their own qdisc. Report the effective qdisc configuration, packet count, and request count together.

Third, a counter delta can be zero even when the capture contains a retransmission. The kernel may classify recovery under a different counter than expected, a counter may be unavailable in that kernel, or the capture may be taken in a different namespace from the sender whose counters you read. Prove namespace placement with `ip netns identify`, socket ownership with `ss -tinp` when permissions allow, and direction from the four-tuple.

Fourth, a successful HTTP status does not mean loss was absent. TCP is supposed to hide recoverable loss from the application. Compare completion-time distributions and transport evidence, not only success counts. Conversely, a slow successful request is not automatically a retransmission. Queueing, application scheduling, delayed ACK policy, and receiver flow control can all add time. The recovery diagnosis passes only when packet order and transport state support the same causal story.

## Key takeaways

- TCP reliability is an evidence system over byte ranges. Retransmission is the repair action after a loss detector has enough evidence.
- A cumulative ACK names the next byte expected. SACK preserves that boundary while reporting noncontiguous ranges already received.
- Three duplicate ACKs drive the classic fast retransmit path because later packets expose a middle hole. A tail loss may produce no later packets and therefore no duplicate-ACK train.
- RTO combines smoothed RTT with an uncertainty margin based on RTT variation. Jitter can move the timer even when average RTT barely changes.
- RACK uses transmit-time order plus ACK feedback. TLP creates feedback near a silent tail. RTO remains the conservative fallback.
- A lost final response segment can turn completed server work into a client-visible deadline failure. Application retries must still be safe.
- Diagnose with sequence ranges, ACK and SACK history, elapsed time, per-socket state, and isolated kernel-counter deltas. A retransmission count alone cannot name the recovery path.

The capstone, [the senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model), puts this recovery workflow back into the full request path. The habit to carry forward is simple: when bytes stall, ask what evidence the sender had, what evidence it lacked, and which timer or ACK supplied the missing fact.

## Further reading

- [RFC 9293: Transmission Control Protocol, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html)
- [RFC 2018: TCP Selective Acknowledgment Options, October 1996](https://www.rfc-editor.org/rfc/rfc2018.html)
- [RFC 5681: TCP Congestion Control, September 2009](https://www.rfc-editor.org/rfc/rfc5681.html)
- [RFC 6298: Computing TCP's Retransmission Timer, June 2011](https://www.rfc-editor.org/rfc/rfc6298.html)
- [RFC 6675: A Conservative SACK-Based Loss Recovery Algorithm, August 2012](https://www.rfc-editor.org/rfc/rfc6675.html)
- [RFC 8985: The RACK-TLP Loss Detection Algorithm for TCP, February 2021](https://www.rfc-editor.org/rfc/rfc8985.html)
- [Linux kernel SNMP counter documentation](https://docs.kernel.org/networking/snmp_counter.html)
