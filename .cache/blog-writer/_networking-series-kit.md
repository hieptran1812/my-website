# Networking for Engineers Who Ship Services: Series Agent Kit

Read this file in full before planning, researching, illustrating, or drafting any post in the networking series. This kit is the shared contract for all 48 posts. The individual post brief in `.claude/plans/networking-series.md` decides the topic. This file decides the voice, evidence standard, lab interface, visual grammar, cross-links, and release gates.

## 1. Series identity

You are writing one deep-dive in **Networking for Engineers Who Ship Services**, a 48-post series for engineers who build and operate services rather than configure networks for a certification exam.

The series owns the wire. Its central claim is:

> The network is not transparent. Every abstraction above it leaks in a specific, predictable, measurable way. A senior engineer learns to locate the leaking layer before opening a dashboard.

Every post must make that claim concrete in three ways:

1. **Intuition first.** Draw where the packet, byte, queue, window, route, or millisecond goes before naming protocol machinery.
2. **Public evidence.** Include at least one real, dated, linked case from an incident report, engineering write-up, standards document, or measurement paper.
3. **Reproduction.** End with a `## Run it yourself` section containing exact commands, the exact output fields to inspect, and an expected numeric range or qualitative result.

Write in the voice of a principal engineer who has debugged production paths with packet captures and kernel counters. Be opinionated, practical, calm, and precise. Define jargon on first use. Build from a concrete request or symptom, then descend through the layers. Never use marketing filler, vague claims of scale, or AI throat-clearing.

The reader should finish knowing:

- which layer can create the symptom;
- which measurement distinguishes competing explanations;
- what the important limit or trade-off is;
- what command settles the question;
- what change is safe, and what new failure that change can introduce.

## 2. Scope and placement

- Target directory: `content/blog/software-development/networking/`
- Frontmatter category: `software-development`
- Frontmatter subcategory: `Networking`
- Depth: `deep-dive`
- Language: English only
- Target length: 9,000 to 11,000 words
- Absolute floor: 6,000 words
- Required figures: exactly 7 total, exactly 6 static WebPs and exactly 1 inline animated figure
- Figure diversity: at least 4 distinct visual kinds across the 7 figures
- Static image format: lossless `.webp` only
- Post date: the actual drafting date in `YYYY-MM-DD`

Use this frontmatter shape:

```yaml
---
title: "Sentence-case Title: Optional Explanatory Subtitle"
date: "YYYY-MM-DD"
publishDate: "YYYY-MM-DD"
description: "One sentence promising the concrete skill the reader gains."
tags:
  [
    "networking",
    "distributed-systems",
    "<topic-tag>",
    "<topic-tag>",
    "<more-specific-tags>",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 45
image: "/imgs/blogs/<slug>-1.webp"
---
```

Use 8 to 12 lowercase, hyphenated tags. Recompute `readTime` from the finished prose at roughly 220 words per minute. The `image` is the first static figure. If the first conceptual figure is animated, also create a static opening figure and use that as the card image.

Before drafting or rendering, prove the target does not already exist:

```bash
test ! -e content/blog/software-development/networking/<slug>.md
test -z "$(find public/imgs/blogs -maxdepth 1 -name '<slug>-*.webp' -print -quit)"
```

Never overwrite a same-slug post or its figures silently.

## 3. The series reasoning spine

Each post is one move in the same diagnostic sequence:

1. **Name the user-visible symptom.** Slow lookup, connect timeout, reset, low goodput, a 5-second spike, uneven backend load, or a partition.
2. **Place it on the path map.** Identify the hop or boundary where the symptom can first be introduced.
3. **Price it on the latency ladder.** Separate resolution, connection, security, server, and transfer time.
4. **Show the packets or state.** Use the packet timeline, kernel queue, route, socket, or flow table that carries the mechanism.
5. **Locate the trade-off on the frontier.** State what improves and what worsens when the knob moves.
6. **Settle it with one discriminating measurement.** Prefer a command and output field over a dashboard screenshot.
7. **Connect the mechanism to a public case.** Explain the transfer lesson, not merely the incident chronology.
8. **Let the reader reproduce it.** The lab must prove the post's main claim or one necessary part of it.

Do not present networking as an inventory of acronyms. Teach causal chains. A useful paragraph often has this shape:

> The queue grows because arrival rate exceeds departure rate. That adds waiting time before it drops packets. The application therefore sees rising RTT before it sees loss. `tc -s qdisc` distinguishes this from slow server work because backlog and overlimit counters rise while application CPU stays flat.

## 4. Evidence and honesty contract

Every nontrivial number belongs to exactly one evidence class.

### A. Derived

State the formula, units, assumptions, substitution, and result. Examples include serialization delay, bandwidth-delay product, retry multiplication, ephemeral-port capacity, Little's Law, and monthly transfer cost.

```text
serialization delay = payload bits / link bits per second
10 MiB x 8 / 1 Gbit/s = 83.9 ms, before propagation or queueing
```

If an equation is an explanatory model rather than a protocol equation or a formula stated by a cited source, label it explicitly as a model or approximation.

### B. Cited

Link the primary source and state its date. Include the version, region, topology, traffic mix, or client population that bounds the result. Prefer, in order:

1. incident owner postmortem;
2. RFC or standards body publication;
3. peer-reviewed paper or conference talk with measurements;
4. maintainer or vendor engineering write-up;
5. high-quality secondary analysis only when no primary source exists.

Do not cite a search result, copied quote collection, anonymous benchmark, or another blog that merely repeats the claim. Never write a precise company result without a direct source.

### C. Reproducible

Name the exact `netlab` configuration, command, tool version when relevant, output field, and expected range. A range is required because scheduler noise, kernel versions, CPU architecture, and container virtualization change exact results.

Bad:

> BBR is much faster.

Good:

> In this lab with 80 ms RTT, 1% random loss, and a 20 Mbit/s bottleneck, three 30-second runs should put CUBIC in roughly 1.5 to 3 Mbit/s and BBR in roughly 8 to 12 Mbit/s. Record all three runs and report the median. Treat this as a lab result, not a universal internet benchmark.

### Numeric table rule

Every table containing reported measurements, prices, limits, or benchmark numbers has a `Source` column. A derived table may use `Derived here` with the formula referenced in the adjacent prose. A lab table may use `netlab, configuration name, date`.

### Forbidden claims

- Never invent a company, incident, customer, benchmark, packet capture, quote, or production measurement.
- Never turn an illustrative example into a named company's result.
- Never say "I measured this in production" unless the measurement is actually available and attributable.
- Never generalize one vendor benchmark beyond its named hardware, software version, topology, and date.
- Never quote a price without cloud region and date checked.
- Never confuse an incident publish date with the incident date.

## 5. Case-study gate

Every post needs at least one case. Before drafting the case section, write a private evidence ledger with these fields:

| Field | Required content |
| --- | --- |
| Case | Organization, system, paper, or RFC story |
| Event date | Exact date or explicitly stated date range |
| Source | Direct public URL |
| Source owner | Incident owner, authors, maintainer, or standards body |
| Mechanism | The network layer and causal chain relevant to this post |
| Verified numbers | Only numbers present in the source, with context |
| Transfer lesson | Guardrail or diagnostic move that generalizes |

The case passes only if all seven fields are present. If the direct source is unavailable, replace the case. Do not pad the post with an unverifiable anecdote.

Use the case bank in `.claude/plans/networking-series.md` as a lead list, not as a citation. Open and verify the source again at drafting time. Cases named in the plan include Meta's 2021-10-04 backbone withdrawal, Rogers Canada on 2022-07-08, Fastly on 2021-06-08, GitHub's 2018-10-21 partition, Slack on 2021-01-04, Roblox from 2021-10-28 through 2021-10-31, DST Root CA X3 on 2021-09-30, and HTTP/2 Rapid Reset in 2023. The plan contains more.

The case section must do more than tell a story:

1. establish what users observed;
2. map the failure onto the recurring path map;
3. explain the mechanism at the layer this post owns;
4. distinguish trigger, contributing conditions, and blast-radius multipliers;
5. end with a transferable control or diagnostic test.

## 6. `netlab` contract

`netlab` is the recurring, Linux-only lab. Its topology and names stay stable across the series so commands compose from one post to the next.

### Canonical base topology

```text
namespace c                     namespace s
client process                  server process
c0 10.77.0.1/30  < veth pair > s0 10.77.0.2/30
```

Stable names:

- namespaces: `c`, `s`
- veth interfaces: `c0`, `s0`
- subnet: `10.77.0.0/30`
- client address: `10.77.0.1`
- server address: `10.77.0.2`
- HTTP port: `8080`
- HTTPS port after Track D: `8443`
- metrics or debug port when needed: `9090`
- tiny Go binaries: `netclient`, `netserver`

Stable source layout:

- lab root: `netlab/`
- Go module: `github.com/hieptran1812/netlab`
- server entry point: `netlab/cmd/netserver/main.go`
- client entry point: `netlab/cmd/netclient/main.go`
- shared wire helpers: `netlab/internal/wire/`
- result formatting: `netlab/internal/report/`
- experiment scripts: `netlab/experiments/<post-slug>/run.sh`
- generated captures and measurements: `netlab/out/<post-slug>/`, never committed as source evidence

Keep the CLI stable: `netserver --listen 10.77.0.2:8080`, `netclient --url http://10.77.0.2:8080/echo --requests N --concurrency C`, and machine-readable output through `--json`. Extend flags without changing these names or defaults.

Post #1 owns setup and teardown. Later posts show only an idempotent preflight and the delta for that experiment. Do not silently rename namespaces, interfaces, addresses, or binaries.

### Base impairment model

Apply delay, loss, reorder, rate, and queue limits through named shell variables or script flags. Every experiment prints the effective configuration before it runs.

```bash
RTT_MS=80
LOSS_PCT=1
RATE_MBIT=20
QUEUE_PKTS=100
```

If one-way `tc netem delay` is placed on only one direction, say so. If the post promises an 80 ms RTT, prove the resulting RTT with `ping`; do not assume a 40 ms setting always becomes exactly 80 ms through a virtualized host.

### Extension topology by track

- Track A to C: base client and server namespaces.
- Track D: add the series TLS terminator without changing endpoint names.
- Track E: add protocol-specific clients such as `nghttp`, `h2load`, or QUIC tooling with exact versions.
- Track F: add a proxy namespace `p` only when the extra hop is the point.
- Track G: add routers or leaf-spine namespaces with explicit names and a topology diagram.
- Track H: reuse earlier faults so captures contain known SYN, loss, zero-window, RST, FIN, retry, and partition signatures.

### Required experiment shape

Every lab has these subsections:

1. **Question.** One falsifiable statement.
2. **Preflight.** Commands proving namespace, interface, route, qdisc, process, and tool state.
3. **Baseline.** One run without the treatment.
4. **Treatment.** One controlled change.
5. **Observation.** The exact output field or packet property to compare.
6. **Expected range.** A range with the reason it can vary.
7. **Reset.** Commands that remove only the state this experiment added.
8. **Production translation.** The safer read-only equivalent for a real host, when one exists.

Labs must be copy-pasteable and idempotent. Use `set -euo pipefail` in complete shell scripts. Validate prerequisites and privileges. Explain that `tcpdump`, `bpftrace`, `conntrack`, namespace changes, qdisc changes, and packet filters may require root or capabilities.

### Safety rules

- Never tell readers to run `tc`, `iptables`, `nft`, route deletion, namespace deletion, or sysctl mutation on an unspecified production interface.
- Scope every destructive cleanup to the lab names above.
- Show the read-only production command separately from the mutating lab command.
- Capture only traffic needed for the claim. Use filters, snap length when appropriate, duration limits, and ring buffers.
- Warn that packet captures can contain credentials, tokens, personal data, and application payloads.
- If a command differs by kernel, distribution, container runtime, or tool version, name the tested environment and provide a detection command.

### macOS contract

`ip netns`, `tc`, and `netem` are Linux facilities. Post #1 explains once that macOS readers run the lab inside Lima, Colima, or another privileged Linux VM or container with `NET_ADMIN`. Later posts link back to post #1 instead of repeating setup instructions.

### Tool vocabulary

Use the tool that answers the question:

| Question | Preferred evidence |
| --- | --- |
| Resolution path | `dig`, `dig +trace`, resolver logs |
| Chosen route | `ip route get`, `mtr` with caveats |
| Socket state | `ss -tin`, `ss -lnt`, `/proc/net/*` where justified |
| TCP counters | `nstat`, `netstat -s`, `ss -ti` |
| Packet sequence | `tcpdump`, `tshark` |
| Throughput | `iperf3`, protocol-native load tool |
| HTTP timing | `curl -w` with named timing fields |
| HTTP/2 behavior | `nghttp`, `h2load` |
| Conntrack | `conntrack -S`, targeted table inspection |
| Kernel event | documented `bpftrace` scripts such as retransmit or connect latency |

Never use `ping` alone to prove application health, throughput, or path symmetry.

## 7. Recurring visual language

The four primitives below are a shared coordinate system. A post should open with either the latency ladder or the path map, with its subject lit. Packet-level posts should also include the packet timeline. Tuning posts should use the frontier. Do not redesign these from scratch.

### Shared semantic palette

| Role | Static fill | Meaning |
| --- | --- | --- |
| Primary | `#a5d8ff` | Current layer, active path, mechanism being taught |
| Caution | `#ffec99` | Queue, bottleneck, trade-off, approaching limit |
| Danger | `#ffc9c9` | Loss, failure, timeout, attack, invalid state |
| Success | `#b2f2bb` | Recovered path, completed handshake, verified outcome |
| External | `#d0bfff` | Off-host system, third party, control plane |
| Neutral | transparent or `#e9ecef` grouping | Context that locates the reader |

Use no more than three strong accents in one figure. `primary`, `caution`, `danger`, and `success` are strong. Neutral and external do not count. The active subject is normally primary. Use danger only for an actual failure, not for generic emphasis.

### Lighting convention

"Light" means semantic emphasis, not decoration:

- exactly one region is the current focus;
- the current region gets primary blue, a 3 px outline or the single hot arrow;
- prerequisite and downstream context stays neutral but readable;
- a demonstrated bottleneck may use amber;
- a demonstrated failure may use red;
- a verified recovery or desired terminal state may use green;
- never light two unrelated regions to make the picture colorful;
- never gray context so aggressively that its label becomes unreadable;
- across sibling posts, the same component keeps the same name and left-to-right position.

### Typography and labels

- Virgil, `fontFamily: 1`, for title, caption, and prose labels.
- Cascadia, `fontFamily: 3`, only for packet flags, field names, commands, addresses, ports, and numeric literals that read like code.
- Title 32, caption 28 or 24, section label 24, body 22, code or edge label 18 to 20.
- Manual line breaks. Keep a line to about 24 Virgil characters or 22 Cascadia characters.
- Label nodes with a noun plus a useful qualifier: `SYN queue\n128 entries`, not `queue`.
- At least 60 percent of nodes carry a quantity, protocol role, state, or qualifier.
- No `[` or `]` in labels. No Markdown emphasis in labels.
- No legend if position, color, and direct labels can carry the meaning.

### Layout convention

- Logical canvas: 2400 by 1600.
- Snap coordinates and dimensions to 20 px.
- Title at `y: 60`, caption immediately below, first body row near `y: 200`.
- Minimum sibling gap: 40 px, preferably 60 to 80 px.
- Reading direction: left to right for causality and time, top to bottom for layers or protocol stack.
- Internal content must occupy all four quadrants. No blank band or decorative filler.
- Static output must be at least 1600 by 900 px and 40 KB.
- One claim per figure. `_claim` or `claim` must contain at least 8 words and be directly supported by nearby prose.
- Caption states the takeaway, not a second title.

### Arrow convention

- Solid: real data or request flow.
- Dotted: control-plane or metadata relationship.
- Dashed: optional, asynchronous, deferred, or failure path.
- Normal arrow: data movement or causal flow.
- Triangle: synchronous blocking request.
- Bar: blocked or terminated.
- Dot: observation or subscription.
- Bind both endpoints to nodes for static Excalidraw figures.
- Prefer orthogonal arrows. Use a direct diagonal only when it clears every node and the diagonal has meaning.
- More than two visible crossings is a redesign, not a polish task.
- Edge labels are at most four words or one quantity.

## 8. Primitive A: latency ladder

### Claim and use

The ladder answers: **which component owns the wall-clock time of one request?** Its canonical order never changes:

`DNS | TCP | TLS | request | server think | first byte | transfer`

Use it at the start of every post unless the path map is clearer. Light only the segment the post owns. The values are scenario-specific and must be derived, cited, or measured.

### Preferred DSL

Use `type: "grid"`, one row, seven columns. Equal cell width is intentional: it keeps the recurring landmark stable. Put the measured duration in the label rather than pretending visual width is proportional. If a post needs a truly proportional waterfall, make that a separate one-off figure and keep this canonical ladder unchanged.

```json
{
  "type": "grid",
  "title": "Where this request spends time",
  "caption": "The current post explains the TCP segment; the rest of the request remains visible as context.",
  "claim": "Connection establishment consumes the highlighted share of this measured request budget.",
  "gridRows": 1,
  "gridCols": 7,
  "nodes": [
    {"id":"dns","row":0,"col":0,"label":"DNS\n12 ms","kind":"neutral","anchor":"latency-budget"},
    {"id":"tcp","row":0,"col":1,"label":"TCP\n38 ms","kind":"primary","anchor":"tcp-handshake"},
    {"id":"tls","row":0,"col":2,"label":"TLS\n41 ms","kind":"neutral"},
    {"id":"request","row":0,"col":3,"label":"request\n2 ms","kind":"neutral"},
    {"id":"server","row":0,"col":4,"label":"server think\n27 ms","kind":"neutral"},
    {"id":"ttfb","row":0,"col":5,"label":"first byte\n39 ms","kind":"neutral"},
    {"id":"transfer","row":0,"col":6,"label":"transfer\n8 ms","kind":"neutral"}
  ],
  "edges": [
    {"from":"dns","to":"tcp"},
    {"from":"tcp","to":"tls"},
    {"from":"tls","to":"request"},
    {"from":"request","to":"server"},
    {"from":"server","to":"ttfb"},
    {"from":"ttfb","to":"transfer"}
  ]
}
```

Replace the sample numbers. Do not copy them into prose. If DNS is cached, say `cache hit` and use the measured sub-millisecond or local value. If connection reuse removes TCP or TLS from a warm request, retain the cells and label them `reused, 0 RTT`; do not delete the segments.

### Animated variant

Use a calm sweep only when showing how cold and warm requests differ or how RTT changes successive phases. Freeze reduced motion on the final fully labeled ladder. A moving highlight that merely visits seven static labels adds no meaning and should remain static.

## 9. Primitive B: path map

### Claim and use

The map answers: **where on the end-to-end path can this mechanism or symptom originate?** Canonical order:

`client -> resolver -> edge / anycast -> L4 LB -> L7 proxy -> mesh sidecar -> app -> backend`

Keep all eight stations even when some are bypassed. Label bypassed nodes `not on this path` or keep them neutral. Do not silently collapse L4 and L7 or merge the resolver into the client.

### Preferred DSL

Use a `pipeline` for the standard map. The engine may wrap the eight stations into a balanced serpentine. That layout is the canonical form for this renderer. If the post adds a branch, such as origin shield versus origin or control plane versus data plane, use a real `graph` with at least two nodes in one layer.

```json
{
  "type": "pipeline",
  "title": "Where this failure enters the path",
  "caption": "The L7 proxy is the first component that can inspect an HTTP route and choose a request-level backend.",
  "claim": "Only components after TLS termination can balance requests using application-layer fields.",
  "nodes": [
    {"id":"client","label":"client\n10.0.0.12:53144","kind":"neutral"},
    {"id":"resolver","label":"resolver\ncache TTL 30 s","kind":"external"},
    {"id":"edge","label":"edge / anycast\nPoP sin","kind":"neutral"},
    {"id":"l4","label":"L4 LB\n5-tuple only","kind":"neutral"},
    {"id":"l7","label":"L7 proxy\nroute + headers","kind":"primary","anchor":"l7-routing"},
    {"id":"sidecar","label":"mesh sidecar\n2 extra hops","kind":"neutral"},
    {"id":"app","label":"app\nHTTP handler","kind":"neutral"},
    {"id":"backend","label":"backend\nstateful store","kind":"external"}
  ],
  "edges": [
    {"from":"client","to":"resolver"},
    {"from":"resolver","to":"edge"},
    {"from":"edge","to":"l4"},
    {"from":"l4","to":"l7"},
    {"from":"l7","to":"sidecar"},
    {"from":"sidecar","to":"app"},
    {"from":"app","to":"backend"}
  ]
}
```

The resolver relationship is logically consulted before the connection path begins. In prose, state that the query returns the destination used by the client; do not imply that application packets traverse the resolver. When that distinction is the claim, use a branched graph with a dotted DNS control edge and a separate solid data path.

### Lighting rules by topic

- DNS: resolver primary, returned edge address external, data path neutral.
- BGP or anycast: edge primary, control relationship dotted, affected route danger.
- TCP or TLS: light the client-to-edge or client-to-L4 boundary, not an arbitrary box.
- L4 versus L7: light both compared boxes with primary and caution, but only if the comparison is the single claim.
- Mesh: sidecar primary; duplicate hop or queue amber only when measured.
- Backend partition: backend danger, first detecting component primary.

## 10. Primitive C: packet timeline

### Claim and use

The timeline answers: **which endpoint sends what, when, and what delay or state transition follows?** It always has two vertical roles, client on the left and server on the right, with time increasing downward and a millisecond rail on the far left.

Use exact flags, sequence or acknowledgement ranges, stream IDs, or protocol messages only when the post explains them. Do not fill the diagram with decorative packets.

### Preferred DSL

Use `type: "raw"` because a two-lifeline sequence diagram is not the same shape as the built-in horizontal timeline. Reuse this coordinate contract:

- left time rail: `x: 140`
- client lifeline: `x: 620`
- server lifeline: `x: 1780`
- body starts near `y: 240`
- event stride: 180 to 220 px
- client and server headers: 360 by 100
- packet arrows span the two lifelines and bind to small event nodes placed on each rail
- all coordinates and dimensions land on the 20 px grid

Logical raw DSL skeleton:

```json
{
  "type": "raw",
  "title": "One loss changes the timeline",
  "caption": "A tail packet loss adds an RTO-sized gap because no later packet arrives to trigger fast retransmit.",
  "claim": "Tail loss can delay completion by a retransmission timeout even when earlier packets arrived quickly.",
  "raw": {
    "elements": [
      {"id":"client-head","type":"rectangle","x":440,"y":220,"width":360,"height":100,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"client\n10.77.0.1"},
      {"id":"server-head","type":"rectangle","x":1600,"y":220,"width":360,"height":100,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"server\n10.77.0.2"},
      {"id":"c-syn","type":"ellipse","x":580,"y":440,"width":80,"height":80,"backgroundColor":"#a5d8ff","strokeWidth":2,"label":"SYN"},
      {"id":"s-syn","type":"ellipse","x":1740,"y":440,"width":80,"height":80,"backgroundColor":"#a5d8ff","strokeWidth":2,"label":"recv"},
      {"id":"syn-flight","type":"arrow","x":660,"y":480,"width":1080,"height":0,"points":[[0,0],[1080,0]],"strokeWidth":2,"endArrowhead":"arrow","startBinding":{"elementId":"c-syn","focus":0,"gap":8},"endBinding":{"elementId":"s-syn","focus":0,"gap":8}},
      {"id":"t0","type":"text","x":140,"y":460,"width":180,"height":40,"text":"0 ms","fontSize":20,"fontFamily":3},
      {"id":"c-rto","type":"ellipse","x":580,"y":1040,"width":80,"height":80,"backgroundColor":"#ffc9c9","strokeWidth":2,"label":"RTO"},
      {"id":"s-recv","type":"ellipse","x":1740,"y":1040,"width":80,"height":80,"backgroundColor":"#b2f2bb","strokeWidth":2,"label":"recv"},
      {"id":"retry","type":"arrow","x":660,"y":1080,"width":1080,"height":0,"points":[[0,0],[1080,0]],"strokeWidth":2,"strokeStyle":"dashed","endArrowhead":"arrow","startBinding":{"elementId":"c-rto","focus":0,"gap":8},"endBinding":{"elementId":"s-recv","focus":0,"gap":8}},
      {"id":"trto","type":"text","x":140,"y":1060,"width":240,"height":40,"text":"RTO fires","fontSize":20,"fontFamily":3}
    ],
    "positions": {
      "client-head":{"x":440,"y":220,"w":360,"h":100},
      "server-head":{"x":1600,"y":220,"w":360,"h":100},
      "c-syn":{"x":580,"y":440,"w":80,"h":80},
      "s-syn":{"x":1740,"y":440,"w":80,"h":80},
      "c-rto":{"x":580,"y":1040,"w":80,"h":80},
      "s-recv":{"x":1740,"y":1040,"w":80,"h":80}
    }
  }
}
```

This is a geometry template, not a complete figure. Add the missing events that prove the post's claim, while staying within 5 to 9 semantic event pairs. Use neutral lifelines, blue normal packets, amber waiting or queueing, red loss or reset, and green completion.

Rules:

- Time labels are actual measured or derived offsets, or symbolic labels such as `+1 RTT`.
- If clocks are not synchronized, use elapsed time from the capture point and say so.
- A missing packet is shown as a dashed arrow ending at a red loss marker, not as an arrow that reaches the peer.
- An ACK arrow points back toward the sender of the acknowledged bytes.
- Retransmission labels include the same sequence range when that is the point.
- Do not draw a response before the request reaches the server.

### Animated variant

Packet motion is appropriate when the order, stall, retransmission, fan-out, or head-of-line blocking is the claim. Animate a packet group or highlight, not every label. The start and end frames must remain meaningful. Freeze reduced motion on the final explanatory state.

## 11. Primitive D: throughput and latency frontier

### Claim and use

The frontier answers: **what throughput or goodput do we gain, and what latency or tail cost do we pay, as load or a tuning knob moves?** It is not a generic line chart. It must expose the trade-off and label the knee, unstable region, or dominated configuration.

Pick one axis pairing and name it exactly:

- x: offered load; y: p99 latency;
- x: RTT; y: goodput;
- x: queue depth; y: goodput and delay as separate aligned panels;
- x: throughput; y: latency, with better direction explicitly marked.

Never put unrelated units on one unlabeled axis. Never draw a smooth empirical curve from two points. Show observed points, uncertainty, or label a curve as a conceptual model.

### Preferred DSL

Use `type: "raw"` with unbound axis arrows, point nodes, and straight line segments. Do not use the horizontal `timeline` engine for a scientific chart.

Coordinate contract:

- plot left: `x: 300`
- plot right: `x: 2200`
- plot top: `y: 300`
- plot bottom: `y: 1380`
- x axis points right; y axis points up
- plot title and caption are supplied by the common DSL header
- axis labels use Cascadia 20
- tick labels use Cascadia 18
- observed points are 40 to 60 px circles
- primary series blue; comparison series amber; failure region red; useful knee green

Logical skeleton:

```json
{
  "type": "raw",
  "title": "The queue hides a latency cliff",
  "caption": "Goodput flattens near saturation while p99 rises sharply, so a deeper queue buys little useful work.",
  "claim": "Past the saturation knee, additional offered load increases tail latency without increasing useful throughput.",
  "raw": {
    "elements": [
      {"id":"x-axis","type":"arrow","x":300,"y":1380,"width":1900,"height":0,"points":[[0,0],[1900,0]],"strokeWidth":2,"endArrowhead":"arrow","unbound":true},
      {"id":"y-axis","type":"arrow","x":300,"y":1380,"width":0,"height":-1080,"points":[[0,0],[0,-1080]],"strokeWidth":2,"endArrowhead":"arrow","unbound":true},
      {"id":"x-label","type":"text","x":1760,"y":1440,"width":440,"height":40,"text":"offered load (req/s)","fontSize":20,"fontFamily":3},
      {"id":"y-label","type":"text","x":100,"y":260,"width":300,"height":40,"text":"p99 latency (ms)","fontSize":20,"fontFamily":3},
      {"id":"p1","type":"ellipse","x":620,"y":1180,"width":40,"height":40,"backgroundColor":"#a5d8ff","strokeWidth":2,"label":""},
      {"id":"p2","type":"ellipse","x":1140,"y":1080,"width":40,"height":40,"backgroundColor":"#a5d8ff","strokeWidth":2,"label":""},
      {"id":"knee","type":"ellipse","x":1600,"y":800,"width":60,"height":60,"backgroundColor":"#b2f2bb","strokeWidth":3,"label":"knee"},
      {"id":"cliff","type":"ellipse","x":1960,"y":420,"width":60,"height":60,"backgroundColor":"#ffc9c9","strokeWidth":2,"label":"p99 cliff"}
    ],
    "positions": {
      "p1":{"x":620,"y":1180,"w":40,"h":40},
      "p2":{"x":1140,"y":1080,"w":40,"h":40},
      "knee":{"x":1600,"y":800,"w":60,"h":60},
      "cliff":{"x":1960,"y":420,"w":60,"h":60}
    }
  }
}
```

Add explicit line segments between points only when interpolation is justified. Label the source and experiment configuration in the nearby prose. The figure caption must say whether it is measured data, a derived bound, or a conceptual curve.

### Animated variant

Animation is appropriate when offered load sweeps upward, a queue fills, cwnd evolves, or two controllers diverge over the same interval. Keep axes and labels static. Move one marker or reveal one curve. Use 8 to 16 seconds, a calm easing function, and a reduced-motion frame at the knee or final comparison.

## Element-form `.in.json` primitive templates

The four DSL examples above are the semantic contracts. The four standalone element-form templates below are the copy-adaptable starting points for `author-scene.mjs`. Save one block as `.cache/blog-writer/<slug>/<slug>-N.in.json`, replace every illustrative number and caption, add the topic-specific detail, then validate. Do not ship the sample values.

### Latency ladder `.in.json`

```json
{"title":"Where this request spends time","_claim":"Connection establishment consumes the highlighted share of this measured request budget.","_caption":"The highlighted TCP segment is the current post's contribution to end-to-end wall clock.","elements":[{"type":"text","id":"title","x":660,"y":60,"width":1080,"height":40,"text":"Where this request spends time","fontSize":32,"fontFamily":1},{"type":"text","id":"caption","x":360,"y":120,"width":1680,"height":40,"text":"The highlighted TCP segment is the current post's contribution to end-to-end wall clock.","fontSize":28,"fontFamily":1},{"type":"rectangle","id":"dns","x":120,"y":260,"width":280,"height":480,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"DNS\n12 ms\nresolution"},{"type":"rectangle","id":"tcp","x":420,"y":260,"width":280,"height":480,"backgroundColor":"#a5d8ff","strokeWidth":3,"label":"TCP\n38 ms\n1 RTT"},{"type":"rectangle","id":"tls","x":720,"y":260,"width":280,"height":480,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"TLS\n41 ms\n1 RTT"},{"type":"rectangle","id":"request","x":1020,"y":260,"width":280,"height":480,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"request\n2 ms\nwrite"},{"type":"rectangle","id":"server","x":1320,"y":260,"width":280,"height":480,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"server think\n27 ms\nhandler"},{"type":"rectangle","id":"ttfb","x":1620,"y":260,"width":280,"height":480,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"first byte\n39 ms\nreturn RTT"},{"type":"rectangle","id":"transfer","x":1920,"y":260,"width":280,"height":480,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"transfer\n8 ms\nbody"}],"export":{"padding":48,"minWidth":1600,"minHeight":900}}
```

### Path map `.in.json`

```json
{
  "title": "Where this mechanism enters the path",
  "_claim": "Before L7 termination, hops use network and transport fields; afterward, route and header fields become visible.",
  "_caption": "TLS stays opaque through L4; the L7 proxy terminates it and exposes route and header fields downstream.",
  "elements": [
    {"type":"text","id":"title","x":620,"y":60,"width":1160,"height":40,"text":"Where this mechanism enters the path","fontSize":32,"fontFamily":1},
    {"type":"text","id":"caption","x":300,"y":120,"width":1800,"height":40,"text":"TLS stays opaque through L4; the L7 proxy terminates it and exposes route and header fields downstream.","fontSize":28,"fontFamily":1},
    {"type":"text","id":"pre-label","x":620,"y":220,"width":520,"height":40,"text":"pre-TLS: IP, port, 5-tuple","fontSize":22,"fontFamily":3},
    {"type":"text","id":"post-label","x":1480,"y":220,"width":680,"height":40,"text":"post-TLS: route + headers visible","fontSize":22,"fontFamily":3},
    {"type":"line","id":"tls-boundary","x":1210,"y":200,"width":0,"height":620,"points":[[0,0],[0,620]],"strokeWidth":3,"strokeStyle":"dashed","strokeColor":"#a5d8ff"},
    {"type":"text","id":"boundary-label","x":1120,"y":840,"width":360,"height":40,"text":"TLS termination boundary","fontSize":22,"fontFamily":3},
    {"type":"rectangle","id":"client","x":80,"y":300,"width":220,"height":400,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"client\n:53144"},
    {"type":"rectangle","id":"resolver","x":370,"y":300,"width":220,"height":400,"backgroundColor":"#d0bfff","strokeWidth":2,"label":"resolver\nTTL 30 s"},
    {"type":"rectangle","id":"edge","x":660,"y":300,"width":220,"height":400,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"edge\npre-TLS\nIP + port"},
    {"type":"rectangle","id":"l4","x":950,"y":300,"width":220,"height":400,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"L4 LB\npre-TLS\n5-tuple"},
    {"type":"rectangle","id":"l7","x":1240,"y":300,"width":220,"height":400,"backgroundColor":"#a5d8ff","strokeWidth":3,"label":"L7 proxy\nTLS\ntermination\nroute +\nheaders"},
    {"type":"rectangle","id":"sidecar","x":1530,"y":300,"width":220,"height":400,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"sidecar\npost-TLS\nservice hop"},
    {"type":"rectangle","id":"app","x":1820,"y":300,"width":220,"height":400,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"app\npost-TLS\nHTTP handler"},
    {"type":"rectangle","id":"backend","x":2110,"y":300,"width":220,"height":400,"backgroundColor":"#d0bfff","strokeWidth":2,"label":"backend\npost-TLS\nstate store"},
    {"type":"arrow","id":"e1","x":308,"y":500,"width":54,"height":0,"points":[[0,0],[54,0]],"startBinding":{"elementId":"client","focus":0,"gap":8},"endBinding":{"elementId":"resolver","focus":0,"gap":8},"endArrowhead":"arrow","strokeWidth":2},
    {"type":"arrow","id":"e2","x":598,"y":500,"width":54,"height":0,"points":[[0,0],[54,0]],"startBinding":{"elementId":"resolver","focus":0,"gap":8},"endBinding":{"elementId":"edge","focus":0,"gap":8},"endArrowhead":"arrow","strokeWidth":2},
    {"type":"arrow","id":"e3","x":888,"y":500,"width":54,"height":0,"points":[[0,0],[54,0]],"startBinding":{"elementId":"edge","focus":0,"gap":8},"endBinding":{"elementId":"l4","focus":0,"gap":8},"endArrowhead":"arrow","strokeWidth":2},
    {"type":"arrow","id":"e4","x":1178,"y":500,"width":54,"height":0,"points":[[0,0],[54,0]],"startBinding":{"elementId":"l4","focus":0,"gap":8},"endBinding":{"elementId":"l7","focus":0,"gap":8},"endArrowhead":"arrow","strokeWidth":2},
    {"type":"arrow","id":"e5","x":1468,"y":500,"width":54,"height":0,"points":[[0,0],[54,0]],"startBinding":{"elementId":"l7","focus":0,"gap":8},"endBinding":{"elementId":"sidecar","focus":0,"gap":8},"endArrowhead":"arrow","strokeWidth":2},
    {"type":"arrow","id":"e6","x":1758,"y":500,"width":54,"height":0,"points":[[0,0],[54,0]],"startBinding":{"elementId":"sidecar","focus":0,"gap":8},"endBinding":{"elementId":"app","focus":0,"gap":8},"endArrowhead":"arrow","strokeWidth":2},
    {"type":"arrow","id":"e7","x":2048,"y":500,"width":54,"height":0,"points":[[0,0],[54,0]],"startBinding":{"elementId":"app","focus":0,"gap":8},"endBinding":{"elementId":"backend","focus":0,"gap":8},"endArrowhead":"arrow","strokeWidth":2}
  ],
  "export": {"padding":48,"minWidth":1600,"minHeight":900}
}
```

### Packet timeline `.in.json`

```json
{
  "title": "One loss changes the packet timeline",
  "_claim": "Tail loss delays completion until the retransmission timer fires when no later packet triggers recovery.",
  "_caption": "Time runs downward; the missing tail packet creates a visible idle gap before retransmission.",
  "elements": [
    {"type":"text","id":"title","x":620,"y":60,"width":1160,"height":40,"text":"One loss changes the packet timeline","fontSize":32,"fontFamily":1},
    {"type":"text","id":"caption","x":420,"y":120,"width":1560,"height":40,"text":"Time runs downward; the missing tail packet creates a visible idle gap before retransmission.","fontSize":28,"fontFamily":1},
    {"type":"rectangle","id":"client-head","x":440,"y":220,"width":360,"height":100,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"client\n10.77.0.1"},
    {"type":"rectangle","id":"server-head","x":1600,"y":220,"width":360,"height":100,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"server\n10.77.0.2"},
    {"type":"arrow","id":"time-rail","x":320,"y":360,"width":0,"height":900,"points":[[0,0],[0,900]],"strokeWidth":2,"endArrowhead":"arrow","unbound":true},
    {"type":"text","id":"time-label","x":80,"y":1260,"width":220,"height":40,"text":"elapsed time","fontSize":22,"fontFamily":3},
    {"type":"line","id":"client-life","x":640,"y":320,"width":0,"height":980,"points":[[0,0],[0,980]],"strokeWidth":2,"strokeStyle":"dashed","strokeColor":"#1e1e1e"},
    {"type":"line","id":"server-life","x":1800,"y":320,"width":0,"height":980,"points":[[0,0],[0,980]],"strokeWidth":2,"strokeStyle":"dashed","strokeColor":"#1e1e1e"},
    {"type":"ellipse","id":"c1","x":580,"y":420,"width":120,"height":100,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"SYN"},
    {"type":"ellipse","id":"s1","x":1740,"y":420,"width":120,"height":100,"backgroundColor":"#e9ecef","strokeWidth":2,"label":"recv"},
    {"type":"arrow","id":"p1","x":708,"y":470,"width":1024,"height":0,"points":[[0,0],[1024,0]],"startBinding":{"elementId":"c1","focus":0,"gap":8},"endBinding":{"elementId":"s1","focus":0,"gap":8},"endArrowhead":"arrow","strokeWidth":2},
    {"type":"text","id":"t1","x":120,"y":440,"width":180,"height":40,"text":"0 ms","fontSize":22,"fontFamily":3},
    {"type":"ellipse","id":"c2","x":580,"y":700,"width":120,"height":100,"backgroundColor":"#ffec99","strokeWidth":2,"label":"tail"},
    {"type":"ellipse","id":"s2","x":1740,"y":700,"width":120,"height":100,"backgroundColor":"#ffc9c9","strokeWidth":2,"label":"loss"},
    {"type":"arrow","id":"p2","x":708,"y":750,"width":1024,"height":0,"points":[[0,0],[1024,0]],"startBinding":{"elementId":"c2","focus":0,"gap":8},"endBinding":{"elementId":"s2","focus":0,"gap":8},"endArrowhead":"bar","strokeStyle":"dashed","strokeWidth":2},
    {"type":"text","id":"t2","x":120,"y":720,"width":180,"height":40,"text":"+1 RTT","fontSize":22,"fontFamily":3},
    {"type":"rectangle","id":"idle-gap","x":900,"y":860,"width":600,"height":160,"backgroundColor":"#ffec99","strokeWidth":2,"label":"idle gap\nno later ACK arrives\nretransmit timer runs"},
    {"type":"ellipse","id":"c3","x":580,"y":1100,"width":120,"height":100,"backgroundColor":"#ffc9c9","strokeWidth":2,"label":"RTO"},
    {"type":"ellipse","id":"s3","x":1740,"y":1100,"width":120,"height":100,"backgroundColor":"#b2f2bb","strokeWidth":2,"label":"recv"},
    {"type":"arrow","id":"p3","x":708,"y":1150,"width":1024,"height":0,"points":[[0,0],[1024,0]],"startBinding":{"elementId":"c3","focus":0,"gap":8},"endBinding":{"elementId":"s3","focus":0,"gap":8},"endArrowhead":"arrow","strokeStyle":"dashed","strokeWidth":2},
    {"type":"text","id":"t3","x":120,"y":1120,"width":180,"height":40,"text":"+RTO","fontSize":22,"fontFamily":3}
  ],
  "export": {"padding":48,"minWidth":1600,"minHeight":900}
}
```

### Throughput and latency frontier `.in.json`

```json
{
  "title": "The queue hides a latency cliff",
  "_claim": "Beyond the 80 percent knee, p99 latency rises sharply while measured goodput remains nearly flat.",
  "_caption": "Aligned panels show the same load sweep: tail latency climbs after the knee while goodput plateaus.",
  "elements": [
    {"type":"text","id":"title","x":700,"y":60,"width":1000,"height":40,"text":"The queue hides a latency cliff","fontSize":32,"fontFamily":1},
    {"type":"text","id":"caption","x":360,"y":120,"width":1680,"height":40,"text":"Aligned panels show the same load sweep: tail latency climbs after the knee while goodput plateaus.","fontSize":28,"fontFamily":1},
    {"type":"text","id":"top-panel","x":80,"y":220,"width":400,"height":40,"text":"p99 latency (ms)","fontSize":24,"fontFamily":3},
    {"type":"line","id":"top-h10","x":300,"y":740,"width":1900,"height":0,"points":[[0,0],[1900,0]],"strokeWidth":2,"strokeStyle":"dashed","strokeColor":"#e9ecef"},
    {"type":"line","id":"top-h50","x":300,"y":620,"width":1900,"height":0,"points":[[0,0],[1900,0]],"strokeWidth":2,"strokeStyle":"dashed","strokeColor":"#e9ecef"},
    {"type":"line","id":"top-h100","x":300,"y":500,"width":1900,"height":0,"points":[[0,0],[1900,0]],"strokeWidth":2,"strokeStyle":"dashed","strokeColor":"#e9ecef"},
    {"type":"line","id":"top-h250","x":300,"y":300,"width":1900,"height":0,"points":[[0,0],[1900,0]],"strokeWidth":2,"strokeStyle":"dashed","strokeColor":"#e9ecef"},
    {"type":"arrow","id":"top-x","x":300,"y":820,"width":1900,"height":0,"points":[[0,0],[1900,0]],"strokeWidth":2,"endArrowhead":"arrow","unbound":true},
    {"type":"arrow","id":"top-y","x":300,"y":820,"width":0,"height":-540,"points":[[0,0],[0,-540]],"strokeWidth":2,"endArrowhead":"arrow","unbound":true},
    {"type":"text","id":"top-t10","x":120,"y":720,"width":140,"height":40,"text":"10 ms","fontSize":22,"fontFamily":3},
    {"type":"text","id":"top-t50","x":120,"y":600,"width":140,"height":40,"text":"50 ms","fontSize":22,"fontFamily":3},
    {"type":"text","id":"top-t100","x":100,"y":480,"width":160,"height":40,"text":"100 ms","fontSize":22,"fontFamily":3},
    {"type":"text","id":"top-t250","x":100,"y":280,"width":160,"height":40,"text":"250 ms","fontSize":22,"fontFamily":3},
    {"type":"line","id":"top-curve","x":600,"y":740,"width":1420,"height":-400,"points":[[0,0],[500,-40],[980,-120],[1420,-400]],"strokeWidth":3,"strokeColor":"#1e1e1e"},
    {"type":"ellipse","id":"lat25","x":520,"y":680,"width":180,"height":120,"backgroundColor":"#a5d8ff","strokeWidth":2,"label":"25% load\np99 12 ms"},
    {"type":"ellipse","id":"lat60","x":1020,"y":640,"width":180,"height":120,"backgroundColor":"#a5d8ff","strokeWidth":2,"label":"60% load\np99 18 ms"},
    {"type":"ellipse","id":"lat80","x":1480,"y":560,"width":220,"height":140,"backgroundColor":"#ffec99","strokeWidth":3,"label":"80% knee\np99 45 ms"},
    {"type":"ellipse","id":"lat95","x":1900,"y":280,"width":220,"height":140,"backgroundColor":"#ffc9c9","strokeWidth":2,"label":"95% load\np99 240 ms"},
    {"type":"text","id":"bottom-panel","x":80,"y":880,"width":440,"height":40,"text":"goodput (Mbit/s)","fontSize":24,"fontFamily":3},
    {"type":"line","id":"bottom-h10","x":300,"y":1280,"width":1900,"height":0,"points":[[0,0],[1900,0]],"strokeWidth":2,"strokeStyle":"dashed","strokeColor":"#e9ecef"},
    {"type":"line","id":"bottom-h20","x":300,"y":1040,"width":1900,"height":0,"points":[[0,0],[1900,0]],"strokeWidth":2,"strokeStyle":"dashed","strokeColor":"#e9ecef"},
    {"type":"arrow","id":"bottom-x","x":300,"y":1400,"width":1900,"height":0,"points":[[0,0],[1900,0]],"strokeWidth":2,"endArrowhead":"arrow","unbound":true},
    {"type":"arrow","id":"bottom-y","x":300,"y":1400,"width":0,"height":-460,"points":[[0,0],[0,-460]],"strokeWidth":2,"endArrowhead":"arrow","unbound":true},
    {"type":"text","id":"bottom-t0","x":160,"y":1380,"width":100,"height":40,"text":"0","fontSize":22,"fontFamily":3},
    {"type":"text","id":"bottom-t10","x":100,"y":1260,"width":160,"height":40,"text":"10 Mbit/s","fontSize":22,"fontFamily":3},
    {"type":"text","id":"bottom-t20","x":100,"y":1020,"width":160,"height":40,"text":"20 Mbit/s","fontSize":22,"fontFamily":3},
    {"type":"line","id":"bottom-curve","x":600,"y":1300,"width":1420,"height":-240,"points":[[0,0],[500,-160],[980,-235],[1420,-240]],"strokeWidth":3,"strokeColor":"#1e1e1e"},
    {"type":"ellipse","id":"gp25","x":520,"y":1240,"width":180,"height":120,"backgroundColor":"#a5d8ff","strokeWidth":2,"label":"25% load\n8 Mbit/s"},
    {"type":"ellipse","id":"gp60","x":1020,"y":1080,"width":180,"height":120,"backgroundColor":"#a5d8ff","strokeWidth":2,"label":"60% load\n17 Mbit/s"},
    {"type":"ellipse","id":"gp80","x":1480,"y":1000,"width":220,"height":140,"backgroundColor":"#ffec99","strokeWidth":3,"label":"80% knee\n20 Mbit/s"},
    {"type":"ellipse","id":"gp95","x":1900,"y":1000,"width":220,"height":140,"backgroundColor":"#ffc9c9","strokeWidth":2,"label":"95% load\n20.5 Mbit/s"},
    {"type":"line","id":"knee-boundary","x":1600,"y":240,"width":0,"height":1160,"points":[[0,0],[0,1160]],"strokeWidth":3,"strokeStyle":"dashed","strokeColor":"#ffec99"},
    {"type":"text","id":"knee-label","x":1380,"y":1440,"width":440,"height":40,"text":"80% knee: latency cost starts","fontSize":22,"fontFamily":3},
    {"type":"text","id":"x-label","x":1820,"y":1500,"width":380,"height":40,"text":"offered load (%)","fontSize":22,"fontFamily":3},
    {"type":"text","id":"method","x":540,"y":1500,"width":980,"height":40,"text":"illustrative observations: median of 3 x 30 s runs","fontSize":22,"fontFamily":3}
  ],
  "export": {"padding":48,"minWidth":1600,"minHeight":900}
}
```

These templates fix recurring positions, names, and semantics. They do not authorize copying the illustrative measurements. Each drafting agent replaces values with derived, cited, or `netlab`-reproduced evidence and re-runs both mechanical and visual gates.

## 12. Figure set contract

Plan seven figures before prose. A strong default set is:

1. opening latency ladder or path map;
2. mechanism figure using packet timeline, graph, stack, or hand-authored internals;
3. one animated sequence where motion carries the claim;
4. comparison matrix or before-after;
5. dated case mapped onto a recurring primitive;
6. measurement or frontier figure;
7. diagnostic decision tree or runbook figure.

This is an exact count: six numbered static WebPs plus one inline animated SVG. The animation counts as one of the seven and never creates an eighth figure. Do not substitute a second animation for a static figure without changing the series contract.

Across the set:

- at least four distinct kinds;
- no kind accounts for more than half the set;
- no adjacent figures reuse the same layout skeleton;
- every abstract mechanism in prose has a figure within about 30 lines;
- figure 1 is referenced in the introduction;
- every static image path is `/imgs/blogs/<slug>-N.webp`;
- all nodes, arrows, labels, and numbers appear in or are justified by nearby prose;
- exactly one animated figure is required, and motion must carry meaning;
- never use ASCII art, Unicode box drawing, Mermaid, or a prose-only substitute.

Static figures use the current blog-writer pipeline. Delegate the complete author, render, visual-review, fix, and re-review loop to `figure-author`. Do not inspect a rendered WebP in the drafting or wave-orchestrator context. The figure reviewer owns pixel-level judgment.

## 13. Animated figure contract

Use inline SVG plus CSS keyframes inside one `<figure class="blog-anim">` raw HTML block. No JavaScript.

Hard rules:

- The opening `<figure` begins at column 0.
- There are no blank lines anywhere inside the figure block.
- The SVG has a `viewBox`, `role="img"`, and an `aria-label` or `<title>`.
- Responsive style includes `width:100%;height:auto;max-width:...px`.
- The figure has a `<figcaption>` that states what the motion proves.
- At least one namespaced `@keyframes` rule is referenced by `animation:`.
- Each post-specific class and keyframe name begins with a unique prefix such as `tls1-` or `tcp8-`.
- Loop duration is 6 to 20 seconds.
- Use no more than three accents. Prefer one moving accent plus neutrals.
- Include `@media (prefers-reduced-motion:reduce)` and freeze on a meaningful, fully legible frame.
- No `<script>`, `on*=` handler, `javascript:`, remote `href`, or remote `src`.
- Start and end states are both true; the change between them is exactly the caption's claim.
- The loop resets cleanly or uses `alternate`.

Preferred networking motion patterns:

- packet flow across fixed waypoints;
- loss followed by timeout or fast retransmit;
- receive window shrinking to zero and reopening;
- cwnd sawtooth compared with a model-based probe cycle;
- DNS resolution walking root, TLD, and authoritative servers;
- BGP withdrawal propagating across ASes;
- HTTP/1.1, HTTP/2, and HTTP/3 requests diverging under one lost packet;
- queue fill and drain with latency rising and falling;
- retry amplification fanning out through call-chain levels;
- ECMP hashing stable flows while one elephant flow pins a link.

Validate each source:

```bash
node .agents/skills/blog-writer/scripts/check-anim.mjs \
  .cache/blog-writer/<slug>/<slug>-anim-<n>.fig.html
```

Embed the validated block verbatim. Watch one complete loop in the local site before release. A static screenshot cannot validate motion semantics.

## 14. Source-aware prose structure

A typical post uses 9 to 13 H2 sections:

1. concrete production symptom and promise;
2. recurring figure that locates the topic;
3. intuition before terminology;
4. mechanism from packet or state point of view;
5. one or more worked derivations;
6. measurement and commands;
7. trade-offs and failure modes;
8. real case study;
9. design or diagnostic decision table;
10. `## Run it yourself`;
11. `## Key takeaways`;
12. `## Further reading`.

Adapt the structure to the topic. Do not force a protocol history section when it does not help diagnosis. Do include at least two worked examples with concrete values when the post contains quantitative mechanisms.

Use real, idiomatic artifacts: shell commands, Go snippets, Wireshark or `tshark` filters, config fragments, sysctl inspection, qdisc state, route output, or load-tool invocations. Do not invent fake command output. When showing output, identify whether it is representative, excerpted from the lab, or quoted from a source.

## 15. Cross-link spine

Every post links to:

1. the intro: `/blog/software-development/networking/what-actually-happens-when-you-curl-a-url`;
2. the capstone: `/blog/software-development/networking/the-senior-engineers-network-mental-model`;
3. two or three networking siblings whose mechanisms compose with this post;
4. one to three existing posts that own a higher-level concern.

Planned sibling links are allowed before the sibling is published, but the slug must come exactly from `.claude/plans/networking-series.md`.

Use existing series to establish a boundary, not to repeat them:

| If this post touches | Networking series owns | Link out for |
| --- | --- | --- |
| System design | wire behavior, protocol visibility, flow limits | architecture choices and system-wide trade-offs |
| Microservices | packets, connections, discovery staleness, proxy behavior | service boundaries and resilience patterns |
| SRE | network signals and discriminating commands | SLOs, alerting, incident process, operational policy |
| API design | framing, multiplexing, wire cost, protocol constraints | resource semantics and contract design |
| Distributed training | interconnect, RDMA, collectives on the wire | parallelism algorithms and training architecture |

Verified useful targets include:

- `/blog/software-development/system-design/load-balancing-from-l4-to-l7`
- `/blog/software-development/system-design/rate-limiting-and-backpressure`
- `/blog/software-development/system-design/cascading-failures-circuit-breakers-and-bulkheads`
- `/blog/software-development/system-design/observability-metrics-logs-traces-by-design`
- `/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation`
- `/blog/software-development/microservices/service-discovery-and-load-balancing`
- `/blog/software-development/microservices/resilience-patterns-timeouts-retries-circuit-breakers-bulkheads`
- `/blog/software-development/microservices/service-to-service-security-mtls-and-zero-trust`
- `/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing`
- `/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right`
- `/blog/software-development/site-reliability-engineering/circuit-breakers-bulkheads-and-load-shedding`
- `/blog/software-development/api-design/http-for-api-designers-methods-status-codes-headers`
- `/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming`
- `/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency`
- `/blog/machine-learning/distributed-training/the-interconnect-physics`
- `/blog/machine-learning/distributed-training/collectives-from-scratch`

Before using an out-link, verify its file exists. Use relative blog URLs, omit `content/`, and omit `.md`.

## 16. `Run it yourself` gate

The final lab section is not optional and does not pass as a loose list of commands.

Required template:

````markdown
## Run it yourself

### Question

State the one claim the experiment will test.

### Preconditions

Name OS, privileges, tools, versions where relevant, and the post #1 setup link.

### Baseline

```bash
# exact commands
```

Read: name the exact field, packet, counter, or timing line.

Expected: give a range or a precise qualitative state.

### Apply one change

```bash
# one controlled mutation
```

### Compare

```bash
# exact repeat and evidence command
```

Read: name what changed and what did not.

Expected: give the range, direction, and known sources of variance.

### Reset

```bash
# scoped cleanup for this lab only
```
````

The section fails if it lacks any of these:

- exact commands;
- the output field or packet property to inspect;
- expected range or state;
- baseline and one controlled treatment;
- reset command for mutations;
- safety or privilege note when relevant;
- an explanation connecting the observation back to the post's main claim.

Do not promise that five minutes includes initial VM, container-image, compiler, or package installation. It means the experiment itself is short once post #1 prerequisites exist.

## 17. Math, LaTeX, and punctuation traps

### LaTeX

- Preserve every LaTeX command with its leading backslash: `\frac`, `\sum`, `\in`, `\mathbb`, `\left`, `\right`, `\mid`, `\prod`, `\lambda`, `\pi`.
- Brace-wrap inline math that begins with a digit. Write `${10}^{6}$`, not `$10^6$`.
- Use `\lt ` instead of a literal `<` immediately before a letter in math or raw HTML-sensitive prose.
- Include units outside or consistently inside math. Define every symbol on first use.
- Label explanatory abstractions as models or approximations. Do not present an inferred objective or approximation as a protocol's exact equation.
- After editing equations, scan for form-feed, backspace, tab, and malformed control sequences.

Useful scan:

```bash
LC_ALL=C grep -n $'[\f\b\t]' content/blog/software-development/networking/<slug>.md
rg -n '(?<!\\)\b(frac|sum|in|mathbb|left|right|mid|prod|lambda|pi)\b' \
  content/blog/software-development/networking/<slug>.md
```

Review matches manually because normal English words such as `in`, `left`, and `right` are not always math errors.

### Em dash

No em dash anywhere in frontmatter, headings, body, captions, alt text, DSL labels, or animated SVG text. Also ban a spaced en dash and a spaced double hyphen. Use a period, colon, comma, or parentheses. An unspaced en dash in a numeric range such as `2018–2022` is allowed.

Scan prose and figure sources:

```bash
rg -n $'\u2014| \u2013 | -- ' \
  content/blog/software-development/networking/<slug>.md \
  .cache/blog-writer/<slug>
```

### Currency and shell variables

Escape literal currency dollars in Markdown prose, such as `\$0.01/GB`. Leave shell variables unescaped inside code fences. Use real math delimiters only for math.

## 18. Research and source rules by topic

- Protocol behavior: start with the relevant RFC and current implementation documentation. State when deployed behavior differs from the clean model.
- Linux kernel behavior: name kernel version or source lineage when a detail changed over time.
- Cloud limits and prices: use official provider documentation, region, SKU, and check date.
- Security events: use the CVE, vendor advisory, standards response, and responsible defensive framing.
- Benchmarks: state hardware, NIC, CPU, kernel, software version, concurrency, payload, RTT, loss, and measurement duration when the source provides them.
- Historical incidents: distinguish trigger from root cause, contributing conditions, and recovery blockers.
- Comparative claims such as CUBIC versus BBR or HTTP/2 versus HTTP/3: define workload and loss model. There is no context-free winner.

Do not use a case because it is famous. Use it because its mechanism is the mechanism of the post.

## 19. Phase and delegation contract

For each post:

1. Read this kit in full.
2. Read only the assigned post line and necessary nearby lines from the series plan.
3. Research primary sources and build the evidence ledger.
4. Plan the outline, abstraction inventory, seven figures, case, lab, and cross-links.
5. Delegate all static figure authoring, rendering, WebP conversion, visual review, fixing, and re-review to `figure-author`.
6. Author and validate the inline animated figure according to the animation contract.
7. Draft prose around verified evidence and passed figures.
8. Delegate the final post verification to `post-verifier`.
9. Fix every reported failure. Never reduce substantive prose to evade an abstraction-coverage failure.
10. Run the local page and watch the animation before release.

The drafting agent must not open rendered WebPs. The wave orchestrator must not read full finished posts. Return concise manifests and gate verdicts upward.

## 20. Verification and release gates

Run the repository verifier through the designated verification agent:

```bash
bash .agents/skills/blog-writer/scripts/verify-post.sh \
  content/blog/software-development/networking/<slug>.md \
  <slug> \
  deep-dive
```

Confirm separately:

- target length 9,000 to 11,000 words and absolute floor 6,000;
- 7 useful figures, with at least 4 kinds and at least 1 meaningful animation;
- every static embed is WebP and matches the slug;
- every WebP exists, is at least 1600 by 900, and is at least 40 KB;
- the opening figure is referenced in the introduction;
- every abstract mechanism has a nearby figure;
- every static figure passed pixel-level visual review;
- animation source passed `check-anim.mjs` and one live loop was watched;
- at least one case passed the seven-field case-study gate;
- every number is derived, cited, or reproducible;
- numeric tables have a `Source` column;
- `Run it yourself` passed all required fields;
- intro, capstone, sibling, and boundary cross-links are present and valid;
- no H1 exists in body;
- English only;
- no ASCII diagram, Unicode box drawing, or Mermaid;
- no em dash, spaced en dash, or spaced double hyphen;
- LaTeX commands and control characters were scanned;
- code fences and raw HTML blocks are closed;
- `readTime` matches finished prose.

The sharpness check can pass vacuously when no files exist. Always count the expected WebPs on disk. Do not trust a green gate without the file count.

Only after all gates pass may the per-post cache be removed. Never remove the shared kit.

## 21. Final drafting checklist

- [ ] The post teaches one layer deeply and keeps higher-level boundaries explicit.
- [ ] The opening symptom is concrete and mapped to the recurring visual language.
- [ ] Jargon is defined before it is used to explain another term.
- [ ] The main mechanism is causal, measurable, and falsifiable.
- [ ] Two worked examples show arithmetic where the topic is quantitative.
- [ ] Seven figures are planned; the first is a ladder or path map; the set has four kinds.
- [ ] The one animated figure needs motion to make its claim.
- [ ] The public case has organization, event date, primary link, mechanism, and transferable guardrail.
- [ ] No reported number lacks derivation, source, or lab reproduction.
- [ ] The lab preserves canonical names and changes one variable at a time.
- [ ] The lab names the output field and expected range.
- [ ] The lab has scoped cleanup and a safety note.
- [ ] Intro, capstone, siblings, and existing higher-level posts are linked.
- [ ] There is no em dash and no malformed LaTeX.
- [ ] Static figures ship as lossless WebP; animation ships as contiguous inline SVG.
- [ ] Verification, visual review, source review, and local animation review all pass.

The quality bar is simple: a reader should be able to predict the symptom, locate its layer, name the measurement that distinguishes it, and reproduce the core claim without trusting the author's authority.
