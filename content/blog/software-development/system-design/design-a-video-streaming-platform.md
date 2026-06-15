---
title: "Design a Video Streaming Platform: CDN, Transcoding, and Adaptive Bitrate"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Design a Netflix-scale streaming platform from the upload pipeline to the edge: how to transcode once into many renditions, stream adaptively, and make the CDN cache-hit ratio the lever that keeps the egress bill from bankrupting you."
tags:
  [
    "system-design",
    "video-streaming",
    "cdn",
    "transcoding",
    "adaptive-bitrate",
    "architecture",
    "distributed-systems",
    "scalability",
    "performance",
    "cost-optimization",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/design-a-video-streaming-platform-1.webp"
---

A friend of mine once joked that running a video streaming platform is "the business of moving the heaviest objects on the internet, repeatedly, to people who will leave if it stutters for two seconds." That sentence is the entire design brief. Every hard decision in this system traces back to one of two brutal facts: video files are enormous, and viewers are impatient. The enormity drives your storage and your bandwidth bill, which is the single largest line item in the company. The impatience drives your latency-to-play and your tolerance for rebuffering, which is the single biggest determinant of whether viewers stay. Design the platform well and you serve a hundred petabytes a day at a cost that does not capsize the business. Design it naively and your egress bill alone — the cost of bytes leaving your servers and reaching viewers — will be larger than your entire engineering payroll before you finish reading this post.

So let us design it. By the end you will be able to size the storage and bandwidth, reason about the upload-and-transcode pipeline that turns one source file into a fan of renditions, explain why adaptive bitrate streaming is the key user-experience optimization and how it works, and — the part that actually decides whether the company lives — understand why the content delivery network is the whole game, and how maximizing its cache-hit ratio is the cost lever that everything else serves. We will draw the architecture, do the napkin math, deep-dive the transcoding farm and the adaptive-bitrate client, then stress-test the design against a viral video, a live-event spike, and a thundering herd at a new release.

The pipeline at the heart of this whole system is the transcoding stage, where a single uploaded master file fans out into many renditions, so let us put that picture on the table first and refer back to it as we build out the rest.

![A directed acyclic graph showing a source upload splitting into chunks, fanning out to parallel encoders for multiple renditions, and merging into a packager that emits HLS and DASH manifests](/imgs/blogs/design-a-video-streaming-platform-1.webp)

## Framing the problem and drawing the scope

Before a single box goes on the whiteboard, a senior engineer pins down what the system is and is not. "Build YouTube" is not a spec; it is a wish. The first move is to carve the problem into the parts that share a design and separate the parts that do not.

The biggest cut is **video-on-demand (VOD) versus live**. VOD means the content already exists — a movie, an uploaded clip — and you have all the time in the world to process it before anyone watches. Live means the content is being created right now, and you have seconds, not hours, to ingest, transcode, package, and deliver it. These share a CDN and a player, but their pipelines are fundamentally different in their latency budgets. We will design VOD in depth because it is the cleaner problem and the larger workload at most platforms, then call out where live diverges.

The second cut is **the control plane versus the data plane**. The control plane is everything that decides *what* to play and *whether you are allowed*: the catalog and metadata, authentication and entitlement, search, recommendations, the watch history, the billing. It is request-response traffic, measured in modest QPS, and it looks like every other web backend you have ever built. The data plane is the video bytes themselves: terabits per second of egress flowing from edge caches to player apps. These two planes have almost nothing in common operationally — different scale, different cost structure, different failure modes — and conflating them is the most common mistake junior designs make. The bytes must never flow through your application servers; they flow from the CDN, and the application servers only ever hand the player a signed URL. We will return to this separation repeatedly because it is the structural backbone of the design.

For functional scope, let us say the platform must let creators upload video; transcode it into multiple qualities; stream it adaptively to web, mobile, and TV clients; show a catalog with search and recommendations; track watch progress; and support both VOD and a basic live mode. For non-functional requirements, the ones that actually constrain the architecture are: start playback in **under two seconds** (time-to-first-frame is the metric viewers feel most), keep rebuffering **under 0.5% of playback time**, serve a global audience with regional latency, sustain a catalog of millions of titles and hundreds of millions of daily viewers, and — stated as a first-class requirement, not an afterthought — do all this at a unit cost that the business can afford, because at this scale cost is a correctness constraint, not a finance problem. If you have read the [cost as a design constraint post](/blog/software-development/system-design/cost-as-a-design-constraint-finops), you already know the punchline: in a streaming business, egress is the cost, and the entire architecture is a machine for not paying full price for egress.

## Estimating the system: where the cost actually lives

You cannot design this system until you have done the napkin math, because the math tells you, unambiguously, that one number dominates everything. If you want the general technique, the [back-of-the-envelope estimation post](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) is the companion to this section; here we apply it to video.

Let us anchor on a mid-large platform: **200 million daily viewers**, each watching about **two hours** per day. That is 400 million viewer-hours per day. At an average delivered bitrate of roughly **3 Mbps** — a blended number across phones on cellular at 1.5 Mbps and TVs on fiber at 8 Mbps — one hour of viewing moves about 1.35 GB. So daily egress is on the order of 400 million × 1.35 GB ≈ **540 petabytes per day**, which works out to roughly **16 exabytes per month**. That is the data plane. Hold that number; it is the one that decides the company's fate.

Now storage. Suppose creators upload, or the catalog grows by, **100,000 hours of new VOD content per year** (a number well within reach for a large platform once you count the long tail). A source master might be, say, 10–20 Mbps; but we do not store the master for streaming — we store the **renditions**. A typical encoding ladder has six to eight rungs (from a 240p mobile rung up to 4K), and the sum of all rung bitrates, plus multiple codecs (H.264 for compatibility, AV1 for efficiency), comes out to roughly **6× the source bitrate** in stored bytes when you store a full ladder in two codecs. So each hour of content might consume on the order of 10–20 GB across all renditions. A hundred thousand hours per year is then 1–2 PB per year of *new* rendition storage, growing the catalog. Storage is large but it is linear and it is cheap per byte; it is not the thing that kills you.

![A six-row estimation matrix mapping daily viewers, hours watched, average bitrate, daily and monthly egress, and per-title storage to their capacity implications, with egress flagged as the dominant cost](/imgs/blogs/design-a-video-streaming-platform-8.webp)

Now put a price on it and watch the asymmetry. Object storage runs around \$0.02 per GB-month; storing a few petabytes is a few tens of thousands of dollars a month — real money, but a rounding error against the rest. Egress is the killer. Public-cloud internet egress lists around \$0.05–\$0.09 per GB; even at a heavily negotiated \$0.02/GB, sixteen exabytes a month is **16 × 10⁹ GB × \$0.02 = \$320 million per month** if you served every byte from your origin at cloud egress rates. That is not a typo. Origin egress at this scale is a number that ends the company. This is precisely why the next section is the most important one in the entire design, and why "maximize the CDN cache-hit ratio" is the optimization that the whole architecture exists to enable.

#### Worked example: the egress bill with and without a CDN

Take the 16 EB/month figure. Suppose your CDN charges roughly \$0.01/GB for delivered bytes (bulk, committed, peering-heavy — realistic at scale) and your origin egress to *fill* the CDN costs the cloud rate of \$0.05/GB.

If you serve **everything from origin** (no CDN): 16 × 10⁹ GB × \$0.05 = **\$800 million/month**. Game over.

If the CDN achieves a **95% cache-hit ratio**, the CDN delivers 95% of bytes (15.2 × 10⁹ GB × \$0.01 = \$152M) and your origin only fills the 5% of misses (0.8 × 10⁹ GB × \$0.05 = \$40M). Total ≈ **\$192 million/month** — a 4× reduction, and that is before you negotiate, peer, or build your own appliances.

If you push the hit ratio to **99%** (the realistic target for a mature platform that pre-positions popular content): CDN delivers 99% (\$158M) and origin fills 1% (\$8M), total ≈ **\$166M**. The difference between 95% and 99% hit ratio — four percentage points — is roughly **\$26 million a month**. That is why the cache-hit ratio is not a vanity metric; it is the single most valuable number in the operation, and every design choice downstream either protects it or erodes it.

## The high-level architecture

With the math making the priorities obvious, the architecture almost writes itself. There are three subsystems: the **ingest-and-transcode pipeline** (offline, batch, runs once per video), the **delivery path** (online, the CDN-fronted data plane that serves every playback), and the **control plane** (the metadata, auth, catalog, and recommendation services that orchestrate playback without ever touching the bytes).

A playback session goes like this. The player opens a title; it calls the control plane's API gateway, which checks the user's entitlement and returns the title's metadata plus a **manifest URL** — a pointer to a small text file (the HLS or DASH manifest) that lists the available renditions and the segment URLs for each. The player fetches the manifest from the CDN, picks a starting rendition, and begins requesting **segments** — small chunks of video, typically 2–10 seconds each — from the CDN edge. The edge serves them from cache (the common case) or, on a miss, fetches them up the CDN hierarchy and ultimately from your origin object store. As playback proceeds, the player measures its download throughput and **switches renditions** up or down at segment boundaries. The bytes never pass through your servers; they flow edge-to-player, and your origin only ever sees cache-miss fills.

That clean split between the control plane (services and metadata) and the data plane (bytes from the edge) is worth drawing explicitly, because it is the structural decision that keeps your application tier small and your bandwidth bill on the CDN where it belongs.

![A directed graph showing a player calling the API gateway which fans out to a metadata service and a recommendation service on the control plane, while video bytes flow separately from the CDN edge to the player with origin fill only on a miss](/imgs/blogs/design-a-video-streaming-platform-9.webp)

Notice what this buys you. The control plane handles maybe tens of thousands of QPS of small JSON requests — a completely ordinary backend you could run on a modest fleet. It scales like any web service, and the patterns from the rest of this series apply directly: cache the catalog reads (see [caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite)), shard the watch-history store, fan recommendation computation out to an offline batch. The data plane, meanwhile, scales by money and physics: more edge capacity, more peering, more cache. The two planes fail independently — a recommendation outage degrades discovery but does not stop the movie you already started; a CDN PoP failure reroutes to another PoP without touching your catalog database. Keeping them decoupled is what lets each scale and fail on its own terms.

The upload-and-ingest path deserves a moment because it is where the pipeline meets the control plane and where a lot of naive designs go wrong. A creator's client does not stream a 4 GB file through your API server; that would pin an application process for the duration of an upload, route the heaviest bytes through your most expensive tier, and turn a flaky mobile connection into a failed multi-gigabyte transfer. Instead the client requests a pre-signed URL and PUTs the file in resumable chunks directly to object storage, so a dropped connection resumes from the last completed part rather than from zero. When the final part lands, a `complete` call publishes an event onto a durable queue — and that event is what kicks off the transcode DAG. Routing the kickoff through a queue rather than a synchronous call is deliberate: it decouples the upload's success from the pipeline's availability, lets you absorb ingest bursts by letting the queue grow, and gives you at-least-once delivery with idempotent consumers so a redelivered "transcode this" event re-runs safely. This is exactly the durable-log-as-backbone pattern from [queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects), and it is why the ingest side of a streaming platform looks like an event-driven system rather than a request-response one.

## The API surface

Keep the API boring; the cleverness belongs in the pipeline and the edge, not the contract. Three groups of endpoints matter. The first is **upload**, which is deliberately not a single POST of a multi-gigabyte file through your API — that would tie up an application server for the length of an upload and route the heaviest bytes through the wrong tier. Instead, the client requests a **pre-signed upload URL** and PUTs the file directly to object storage, often in resumable chunks so a dropped connection does not restart a 4 GB upload from zero.

```http
POST /v1/uploads
  -> { "uploadId": "u_8f3a", "url": "https://upload-store/...&X-Amz-Signature=...",
       "partSize": 8388608, "method": "resumable-PUT" }

PUT  {url}            # client streams bytes straight to object storage, by part
POST /v1/uploads/u_8f3a/complete   # triggers the transcode pipeline
```

The second group is **playback**. The player never asks your servers for video; it asks for a manifest URL and renders from the CDN.

```http
GET /v1/titles/{titleId}/playback
  -> { "manifestUrl": "https://cdn.example.com/t/9c2/master.m3u8?token=...",
       "drmLicenseUrl": "https://drm.example.com/license",
       "startPositionSec": 412 }     # resume where the viewer left off
```

The manifest URL carries a short-lived signed token so the CDN can authorize delivery at the edge without calling back to your control plane on every segment — critical, because a two-hour movie at 6-second segments is **1,200 segment requests per viewer**, and you cannot afford an auth round-trip on each. The third group is **control-plane CRUD** — catalog browse, search, watch-progress updates, recommendations — ordinary REST/gRPC, designed per the [API design post](/blog/software-development/system-design/api-design-rest-grpc-graphql). The watch-progress update is a fire-and-forget event, not a synchronous write; the player heartbeats its position every few seconds onto an event stream rather than hammering a database, which is exactly the kind of write you route through a log (see [queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects)).

## The data model: what you actually store

The control-plane data model is small and conventional. A **Title** has metadata (name, description, cast, artwork, duration, rights/availability windows by region). A title has many **Assets** — the source master plus every rendition. A rendition row records its codec, resolution, bitrate, the container/packaging, the manifest path, and the storage location and tier. A **WatchEvent** stream records who watched what, how far, on what device — feeding both resume-playback and recommendations. Entitlements link users to what they may play.

```sql
CREATE TABLE titles (
  title_id      UUID PRIMARY KEY,
  name          TEXT NOT NULL,
  duration_sec  INT,
  status        TEXT,            -- uploading | transcoding | ready | failed
  region_rights JSONB            -- availability windows by territory
);

CREATE TABLE renditions (
  rendition_id  UUID PRIMARY KEY,
  title_id      UUID REFERENCES titles,
  codec         TEXT,            -- h264 | hevc | vp9 | av1
  height        INT,             -- 240 .. 2160
  bitrate_kbps  INT,
  manifest_path TEXT,            -- object key of the segment playlist
  storage_tier  TEXT             -- hot | warm | standard | cold
);
```

The heavy data — the segments themselves — does not live in a database at all. It lives in **object storage**: billions of small immutable files (each segment of each rendition of each title), addressed by key, served to the CDN on miss. This is the right home for it because segments are write-once, read-many, large in aggregate but individually modest, and need no transactional semantics. Object storage gives you eleven-nines durability, near-infinite capacity, and lifecycle policies to move cold objects to cheaper tiers automatically — which is the seam where storage tiering lives, a topic we will return to.

## Deep dive 1: the transcoding pipeline

This is the first genuinely hard part, and it embodies a principle that runs through the whole design: **transcode once, store many, serve forever.** You pay the compute cost of transcoding exactly once per title, then amortize it across every future view. Get this stage wrong and you either re-encode on the fly (insane at scale) or store a single quality (a terrible experience). Get it right and the rest of the system has clean inputs.

When the upload completes, an event lands on a queue and the pipeline begins. The first job **validates and analyzes** the source: probes the codec, resolution, frame rate, color space, audio tracks, and runs quality analysis that will inform the encoding ladder. The second decision is the **encoding ladder** — the set of (resolution, bitrate, codec) rungs you will produce. A naive platform uses one fixed ladder for everything. A sophisticated one uses **per-title encoding**: a low-motion talking-head video does not need the same bitrate as a fast-action sports clip to look equally good, so you analyze the content and assign each title a ladder tuned to its complexity, saving bitrate (and therefore egress) on easy content without sacrificing quality on hard content. Netflix's per-title and later per-shot encoding work is the canonical example, and it is a pure egress optimization dressed up as a quality feature.

Now the expensive part. Transcoding a two-hour 4K source serially would take many hours of CPU. You do not have many hours, and you certainly do not want one machine pinned for that long per title when thousands of titles are in flight. So you **chunk and parallelize**. The source is split at **GOP boundaries** (group-of-pictures boundaries — points where a frame is independently decodable, so a chunk can be encoded without needing frames outside it) into segments of a few seconds each. Each chunk becomes an independent encode job. A two-hour video at 6-second chunks is 1,200 chunks; multiply by the number of ladder rungs and codecs, and a single title fans out into **thousands of small, independent, embarrassingly-parallel encode jobs**. You schedule these across a transcoding farm — a fleet of worker machines (often spot/preemptible instances, since the work is retryable and idempotent) — and a single title that would take six hours serially finishes in minutes because you ran a thousand chunks at once.

This is a DAG, not a line. The split fans out to parallel per-chunk-per-rendition encodes; the encodes fan back in to a per-rendition **packager** that stitches the encoded chunks into a continuous segment stream and writes the manifest. Look back at figure 1: that is the shape — source → split → parallel encode → packager → manifests. The DAG matters because it makes the parallelism explicit and the failure handling tractable: if chunk 487 of the 1080p rendition fails (a spot instance got reclaimed mid-encode), you retry just that chunk, not the whole title. Every job is idempotent — encoding the same chunk twice yields the same bytes — which is what makes spot instances and retries safe, and which is the same idempotency discipline covered in [idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design).

```python
# Orchestrator: fan one title into chunk-encode jobs, then package per rendition.
def submit_title(title_id, source_key, ladder):
    chunks = split_at_gop(source_key, chunk_sec=6)   # e.g. 1200 chunks
    jobs = []
    for rung in ladder:                              # e.g. 6 rungs x 2 codecs
        for ci, chunk in enumerate(chunks):
            jobs.append(EncodeJob(
                title_id=title_id, chunk_index=ci, chunk=chunk,
                height=rung.height, bitrate=rung.bitrate, codec=rung.codec,
                # deterministic output key => idempotent retries are safe
                out_key=f"{title_id}/{rung.codec}/{rung.height}/{ci:05d}.m4s"))
    queue.submit_all(jobs)                            # ~14,400 parallel jobs
    # a barrier fires the packager per rendition once all its chunks exist
    for rung in ladder:
        on_all_complete(rung, lambda r: package_rendition(title_id, r))
```

The packager is where **HLS and DASH** get produced. HLS (Apple's HTTP Live Streaming) and DASH (the MPEG standard) are the two dominant adaptive-streaming formats; both work by chopping each rendition into segments and writing a **manifest** (an `.m3u8` playlist for HLS, an `.mpd` for DASH) that lists, for each rendition, its bitrate and the URLs of its ordered segments. You typically produce both formats from the same encoded segments (modern CMAF packaging lets one set of segment files serve both), because TV and Apple devices favor HLS and others favor DASH. The output is what the CDN serves and the player consumes.

#### Worked example: sizing the transcoding farm

Suppose 100,000 hours of new content arrive per year, but it is bursty — most of it lands in working hours, so peak ingest is roughly 4× the average. Average is 100,000 / 8,760 ≈ **11.4 hours of content per hour**; peak ≈ **46 hours of content per hour**.

Encoding a full ladder (say 12 rendition-codec combinations) typically costs on the order of **5–10× real-time of CPU per rung** on commodity cores for the heavy codecs, so a full ladder might be ~80× real-time of single-core CPU per hour of content. At peak, 46 content-hours/hour × 80 core-hours per content-hour ≈ **3,680 cores** running flat out — call it ~115 32-core machines kept busy at peak, fewer if you lean on hardware encoders or accept slower turnaround off-peak.

The lever here is **parallelism for latency, not throughput**: chunking does not reduce total core-hours, but it collapses the *wall-clock* time per title from hours to minutes, which is what lets you promise creators fast availability and lets you absorb spikes by borrowing spot capacity. Because every job is idempotent and retryable, you run the farm almost entirely on **preemptible instances at a fraction of on-demand price** — turning a latency win into a cost win, which is exactly the kind of second-order optimization a senior looks for.

For live, this whole pipeline collapses into a streaming version: you cannot wait for the file to finish, so you transcode the incoming stream in real time, package into short segments on the fly, and push them to the CDN with a latency of a few seconds. The ladder is smaller (fewer rungs, faster codecs) because you are racing the clock, and you accept lower compression efficiency to hit the latency budget. That trade — quality and ladder depth sacrificed for latency — is the defining tension of live, and it is why the same platform runs two different encoding configurations.

One more pipeline concern that bites teams in production is **failure isolation and observability inside the farm**. A title is "ready" only when every chunk of every rendition has encoded successfully and the packager has written every manifest; until then it is in a partial state that must never be exposed to viewers. The orchestrator therefore tracks completion as a per-rendition barrier — the packager for the 1080p H.264 rendition fires only when all 1,200 of its chunks exist — and the title's `status` flips to `ready` only when all renditions have packaged. If a small number of chunks are stuck (a worker class is starved, a particular codec is throwing on a corrupt GOP), you want that visible as "title X is 99.8% encoded, blocked on three 4K-AV1 chunks" rather than as a vague "transcoding…" that hangs forever. The practical move is to emit a metric per chunk-job completion and alert on titles whose encode progress stalls, so a poison chunk surfaces as a targeted retry or a quarantine rather than a silent backlog. This is the same operational hygiene the [observability post](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) argues for, applied to a batch farm: instrument the unit of work, not just the aggregate.

A subtle but important correctness rule ties the whole pipeline together: **the renditions of a title must share a common segment timeline.** Adaptive bitrate only works if the player can switch from the 720p segment that ends at 18.0 seconds to the 1080p segment that *starts* at 18.0 seconds without a glitch — which requires that every rendition is segmented at exactly the same boundaries, aligned to the same keyframes. That is why you split the source at GOP boundaries once, up front, and force every encoder to honor those same boundaries, rather than letting each encoder choose its own keyframe placement. Get this wrong and renditions drift out of alignment, switches stutter or fail, and you have a bug that only shows up on the network conditions that trigger a switch — the hardest kind to reproduce. Aligned segmentation across the ladder is a pipeline invariant, not an optional nicety.

## The manifest, packaging, and content protection

Between "the pipeline produced segments" and "the player rendered them" sits a thin but load-bearing layer: the manifest and the content-protection scheme. It is worth a section because it is where a surprising amount of real-world breakage and cost lives.

The **manifest** is a small text file that the player reads to understand what is available. For HLS it is a master `.m3u8` that points to one media playlist per rendition; each media playlist lists the rendition's segments in order, with their durations. For DASH it is an `.mpd` XML that describes the renditions (called representations) and how to construct each segment's URL. The manifest is tiny — kilobytes — but it is requested by every viewer at the start of playback and sometimes refreshed (for live, constantly, as new segments appear), so it must be cached at the edge like any other object. A live manifest that updates every few seconds and is *not* cacheable will hammer your origin with manifest fetches even when the segments themselves cache beautifully; the fix is short-TTL caching of the live manifest at the edge so that ten million viewers refreshing a live playlist generate a manageable trickle of origin fetches rather than a flood.

```m3u8
#EXTM3U
#EXT-X-VERSION:7
#EXT-X-STREAM-INF:BANDWIDTH=5000000,RESOLUTION=1920x1080,CODECS="avc1.640028"
1080p/h264/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=2800000,RESOLUTION=1280x720,CODECS="avc1.64001f"
720p/h264/playlist.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=1200000,RESOLUTION=854x480,CODECS="av01.0.04M.08"
480p/av1/playlist.m3u8
```

Modern packaging leans on **CMAF** (Common Media Application Format), which lets a single set of fragmented-MP4 segment files serve both HLS and DASH. This matters for the cache-hit ratio: if HLS and DASH each needed their own copy of every segment, you would double your storage *and* split your cache entries, halving your effective hit ratio on the segments. CMAF packaging means web, mobile, and TV clients requesting the same content hit the same cached segment regardless of whether they speak HLS or DASH — one cache entry, one stored copy, maximum reuse. Choosing CMAF over format-specific packaging is a quiet but real optimization that protects the metric that matters most.

**Content protection** is where money meets law. Premium content (licensed movies, live sports rights) typically requires **DRM** (digital rights management) — the segments are encrypted, and the player must fetch a decryption license from a license server, gated on the user's entitlement, to play them. The industry uses a handful of DRM systems (Widevine for Android/Chrome, FairPlay for Apple, PlayReady for Microsoft), and you commonly encrypt once with a common encryption scheme and issue per-system licenses. The architectural consequence is that DRM splits the auth into two checks: a coarse entitlement check on the control plane when the player asks for the manifest, and a fine-grained license issuance on the *license server* when the player needs the decryption key. The license server is a separate, security-sensitive control-plane service with its own scaling profile — it is hit once per playback session (or per key rotation), not per segment, which keeps it manageable. Cheaper content may skip DRM and use plain **signed-URL / token authorization** at the CDN edge, where the edge validates a short-lived signed token before serving a segment and never calls back to your origin for auth. The senior decision is matching the protection level to the content's value: full DRM for licensed premium, signed tokens for user-generated content, public caching for genuinely public assets — because each step up in protection costs latency, complexity, and edge-cacheability, and you pay it only where the content's value demands it. This is the same authn/authz layering discussed in the [architecture-level security post](/blog/software-development/system-design/architecture-level-security-authn-authz-zero-trust), specialized to the fact that on the hot path you cannot afford an auth round-trip per chunk.

## Deep dive 2: adaptive bitrate streaming

Here is the single best user-experience decision in the entire system, and it is worth understanding precisely because it is *why* you built the encoding ladder in the first place. The problem: a viewer's available bandwidth is unknown at the start and **changes constantly** during playback — a phone moves from WiFi to cellular, a home network gets saturated when someone else starts a download, a train enters a tunnel. If you commit to one bitrate, you are making a bet you cannot win. Pick too high and the player runs out of buffered video and **rebuffers** — the spinning wheel, the conversion-killer. Pick too low and you waste a great connection on a blurry picture.

Adaptive bitrate streaming (ABR) refuses to make that bet. Because you encoded the content as a ladder of renditions, all segmented to the same timeline, the player can switch between rungs **at every segment boundary**. It downloads a segment, measures how long that took relative to the segment's playback duration, estimates current throughput, and chooses the rendition for the *next* segment accordingly. If segments are arriving faster than real-time and the buffer is filling, step up a rung. If they are arriving slowly and the buffer is draining toward empty, step down — immediately, before the buffer hits zero and playback stalls. The decision is the player's, made continuously, on the live network. That is the optimization, and it is the difference between fixed and adaptive that the next figure makes concrete.

![A before-and-after comparison showing a fixed bitrate that buffers when bandwidth drops versus adaptive bitrate where the client measures throughput and switches renditions at each segment boundary to avoid rebuffering](/imgs/blogs/design-a-video-streaming-platform-2.webp)

The ABR decision is governed by an algorithm in the player, and the choice of algorithm is a real engineering decision. The simplest is **throughput-based**: estimate bandwidth from recent segment downloads and pick the highest rung that fits. It is responsive but jittery — a single slow segment can trigger a needless downshift. **Buffer-based** algorithms (the BOLA family, and Netflix's published buffer-based approach) instead key the decision off how much video is buffered: a full buffer can afford to gamble on a higher rung, a draining buffer must play safe. Modern players blend the two: use throughput estimates when the buffer is short and you have no slack, lean on buffer occupancy when you have a comfortable cushion. The goal function is explicit and multi-objective: maximize delivered quality, minimize rebuffering (weighted heavily — viewers hate stalls far more than they value resolution), and minimize quality oscillation (constant up-and-down switching is itself annoying).

```python
# A blended ABR rule, simplified, run by the player per segment.
def pick_rendition(buffer_sec, throughput_kbps, ladder, current):
    safe = throughput_kbps * 0.8                      # headroom factor
    # candidate by throughput
    by_tput = max((r for r in ladder if r.bitrate <= safe), default=ladder[0])
    if buffer_sec < 10:                               # buffer is shallow: be safe
        return by_tput
    if buffer_sec > 30:                               # deep buffer: allow a step up
        higher = next((r for r in ladder if r.bitrate > current.bitrate), current)
        return higher
    return current                                    # comfortable: avoid oscillation
```

Two design parameters from earlier now reveal their consequences. **Segment length** is a trade-off: short segments (2s) let ABR react fast and cut startup latency, but mean more requests, more manifest overhead, and slightly worse compression; long segments (10s) compress better and reduce request volume but make ABR sluggish and slow to start. Most VOD lands at 4–6 seconds; low-latency live pushes toward 1–2 seconds. **Time-to-first-frame** — that two-second startup budget — is won by starting playback on a *low* rung (which downloads fast) and ramping up once the buffer fills, rather than insisting on the best quality from frame one. The viewer would rather see a slightly soft picture instantly than a perfect picture after five seconds of spinner. That single heuristic — start low, ramp up — is responsible for a large share of perceived startup performance, and it costs nothing but a few lines in the player.

## Deep dive 3: the CDN is the whole game

Everything so far has been setup. This is the section the entire architecture exists to serve, because this is where the 16-exabytes-a-month live, and where the difference between a viable company and a bankrupt one is decided.

Recall the math: serving your egress from the origin at cloud rates is a number that ends the company. The content delivery network is the answer, and its job is brutally simple to state and endlessly hard to execute: **cache the bytes physically close to viewers so they almost never travel from your origin.** A CDN is a global mesh of caching servers (points of presence, PoPs) sitting in data centers and increasingly *inside ISP networks*, near the viewer. When a player requests a segment, the nearest edge PoP serves it from local cache. The first viewer in a region to request a given segment causes a miss that fills the cache from a higher tier; every subsequent viewer in that region is served locally. Because popular content is requested by millions, the hit ratio on hot content approaches 100%, and your origin serves a trickle.

The CDN is **multi-tier**, and the tiering is what makes the hit ratio robust. The edge tier is closest to viewers and largest in count but smallest per-node in cache; a miss there does not go straight to your origin — it goes to a **regional/mid-tier cache** that aggregates misses from many edges, and only a miss *there* goes to an **origin shield** (a single designated cache in front of your origin) before finally hitting the origin object store. Each tier collapses a wider fan of misses into a narrower one, so that even content that is not hot enough to live at every edge still gets served from cache most of the time, and your origin sees only the genuinely cold long tail. That layered absorption is the structure worth drawing.

![A layered stack showing a client request passing through an edge point of presence with high hit ratio, then a regional cache, then an origin shield that collapses misses, before reaching the origin store which serves under two percent of bytes](/imgs/blogs/design-a-video-streaming-platform-3.webp)

The optimization that everything serves is the **cache-hit ratio**, and you have real levers on it. The first is **pre-positioning**: do not wait for the first viewer to cause a miss on a hot title. When you know a blockbuster releases Friday at midnight, you *push* its segments out to edge caches *before* the rush, so the launch is served entirely from warm caches with a near-100% hit ratio from the first second. Netflix calls this proactive fill and runs it during off-peak hours when its appliances and the networks they sit in are idle. The second lever is **cache key discipline**: every needless variation in your URLs — a tracking query parameter, an inconsistent host, a per-user signature baked into the path instead of a separate token — splits one cache entry into many and craters your hit ratio. Senior teams treat the cache key as a designed artifact: the segment URL is identical for every viewer (the auth token lives in a header or a short-lived signed query the CDN is configured to ignore for caching), so all viewers of a segment share one cache entry. The third lever is **origin shielding**, which we will see again under stress: funneling all misses through a single shield means that a thousand simultaneous edge misses for a brand-new segment become *one* origin fetch, not a thousand.

The most important architectural fact about CDNs at the very top of the scale is that the biggest platforms **build their own**. Netflix's **Open Connect** program places custom caching appliances — purpose-built boxes stuffed with storage — *physically inside ISP data centers and internet exchanges*, given to ISPs for free. The economics are irresistible for everyone: the ISP saves on its own transit costs because Netflix traffic no longer crosses expensive peering links, and Netflix gets its bytes served from a box that is one network hop from the viewer, at essentially zero marginal egress cost. During off-peak hours Netflix fills these appliances with the content its models predict the region will watch tomorrow, so that during peak the appliances serve nearly all traffic locally with no backhaul at all. This is the logical endpoint of "maximize the hit ratio and minimize origin egress": you stop renting CDN capacity and put your cache inside the last-mile network itself. YouTube does the analogous thing with Google's Global Cache nodes inside ISPs. At sufficient scale, the CDN stops being a vendor you pay and becomes infrastructure you build, because the egress savings dwarf the capital cost. The before-and-after on egress is the whole business case in one picture.

![A before-and-after comparison showing all egress served from origin at cloud rates costing far more than serving ninety-five percent from the CDN edge at bulk rates with only a small origin fill cost](/imgs/blogs/design-a-video-streaming-platform-5.webp)

## Deep dive 4: storage tiering for the long tail

Storage is not the cost villain that egress is, but a few petabytes of careless storage still adds up, and the structure of the catalog hands you an easy optimization. Viewership is wildly skewed: a small fraction of titles account for the overwhelming majority of views (the head), while a vast catalog of old or niche content gets played rarely (the long tail). It is wasteful to keep a documentary nobody has streamed in eight months on the same fast, replicated, edge-cached storage as this week's blockbuster.

So you **tier storage by popularity**. The hottest content lives at the edge (in CDN caches) and on fast regional storage so misses fill quickly. Recent and trending content sits on standard, performant object storage. The cold long tail drops to cheaper, slower object-storage tiers (infrequent-access or archive classes), where bytes cost a fraction as much but a first-byte fetch is slower. Because the long tail is, by definition, rarely requested, that slower fetch happens rarely and on content where a slightly longer time-to-first-frame is acceptable. Lifecycle policies on the object store automate the demotion: an object not accessed in N days moves to a colder tier automatically; a sudden spike of interest (an old title goes viral, a catalog promotion) promotes it back. The mapping of popularity to tier is the optimization, and it is worth seeing laid out.

![A layered stack showing storage tiers from a hot edge cache holding the top titles, to regional SSD for trending content, to standard object storage for the full catalog, down to cold archive storage for the rarely played long tail](/imgs/blogs/design-a-video-streaming-platform-6.webp)

The senior insight here is that storage tiering and CDN caching are the *same optimization at two layers*: both are popularity-aware placement that puts hot bytes on fast, expensive media close to viewers and cold bytes on slow, cheap media far away. The CDN cache is just the hottest, closest tier of a storage hierarchy that runs all the way down to tape-grade archive. Designing them together — a unified popularity signal feeding both your edge pre-positioning and your storage lifecycle — is more coherent than treating them as separate systems, and it means one good prediction of "what will people watch" pays off twice.

## Measuring the thing that matters: QoE and the cost levers

You cannot optimize what you do not measure, and in streaming the metrics that matter are not the ones a generic web backend tracks. The defining metric is **Quality of Experience (QoE)**, and it decomposes into a handful of numbers that map directly to whether viewers stay. **Time-to-first-frame (startup latency)** is how long from pressing play to seeing video; the target is under two seconds and ideally under one, because abandonment climbs sharply with each additional second of spinner. **Rebuffer ratio** is the fraction of playback time spent stalled (the spinning wheel mid-stream); you want it under half a percent, and it correlates with engagement and churn more strongly than almost anything else. **Average delivered bitrate** and **bitrate switches per session** capture the quality-versus-stability trade: high bitrate is good, but constant oscillation is annoying, so you watch both. **Playback failure rate** — sessions that never start or die mid-stream — is the floor you must keep near zero. These are measured at the *player*, reported back as telemetry, and aggregated per region, per device class, per ISP, per CDN, so you can see that, say, viewers on one mobile carrier in one region are getting a 3% rebuffer ratio while everyone else is fine — which points you at a specific peering or PoP problem.

On the cost side, two operational metrics dominate. The **cache-hit ratio** per tier (edge, regional, shield) is the lever we have returned to throughout; you alert when it drops, because a falling hit ratio is money leaking to origin egress in real time, often the first symptom of a cache-key regression or a content mix shift. The **egress cost per delivered hour** — your total egress bill divided by viewer-hours — is the unit economic that tells you whether the optimizations are working; drive it down with better codecs, better pre-positioning, and a higher hit ratio, and watch it as religiously as a finance team watches gross margin, because at this scale it *is* the gross margin. The discipline mirrors the SLO-and-error-budget thinking in the [reliability post](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation): set explicit targets (p99 startup under 2s, rebuffer ratio under 0.5%, edge hit ratio above 95%), measure them continuously, and treat a breach as a paging incident rather than a quarterly review item.

#### Worked example: turning a QoE regression into a fix

Telemetry shows the global rebuffer ratio has crept from 0.3% to 0.9% over a week, concentrated in one region on mobile networks. The naive read is "the CDN is slow." The senior read is to decompose. You check the **edge hit ratio** for that region: it has dropped from 96% to 88%. That extra 8% of misses is now backhauling from origin, adding latency that pushes mobile players — already on thin bandwidth — into rebuffering. Why did the hit ratio drop? You inspect cache keys and find a new client build appended a per-session analytics parameter to segment URLs, splitting each segment into thousands of distinct cache entries that never reused. The fix is one line — move that parameter out of the cached path — and the hit ratio recovers to 96%, the backhaul disappears, and the rebuffer ratio falls back under 0.4%. The lesson: a user-experience regression and a cost regression were the *same* regression, surfaced through different metrics, and the fix was to protect the cache key. This is why a senior instruments the hit ratio and the QoE metrics together — they move together far more often than people expect.

## Choosing the encoding ladder: a decision you make per title

We have touched the ladder twice; let us make the decision procedure explicit, because it is a place where seniors and juniors visibly differ. A junior picks one ladder and applies it to everything. A senior recognizes that the right ladder depends on the content and the audience, and reasons through it. Is this live or VOD? Live forces a short ladder of fast-to-encode rungs to meet the latency budget. Is the title premium 4K content with a large expected audience, where the extra storage and encode cost of a deep ladder (including AV1 4K rungs) is justified by the egress savings across millions of views and by the experience on big-screen TVs? Or is it a short, low-traffic clip where a lean H.264-only ladder up to 1080p is plenty and the cost of encoding AV1 4K would never be recouped? The decision tree captures the reasoning.

![A decision tree branching from a new title to live versus video on demand, then for video on demand to premium high-audience content getting a full multi-codec four-K ladder versus low-traffic content getting a lean H.264 ladder](/imgs/blogs/design-a-video-streaming-platform-7.webp)

Underneath the ladder sits the **codec choice**, which is itself a trade-off with no single winner — which is why a real ladder mixes codecs rather than betting on one. H.264 (AVC) is the universal baseline: it plays on literally everything, encodes cheaply, but compresses worst, so it costs the most egress per unit quality. H.265 (HEVC) and VP9 cut bitrate by roughly a third over H.264 — meaningful egress savings — at the cost of more encode time and narrower device support, with HEVC carrying royalty complications. AV1 is the efficiency champion, cutting bitrate by up to half, and is royalty-free, but encoding it is extremely expensive (orders of magnitude slower than H.264) and device support, while growing fast, is not yet universal. The practical answer is a ladder that ships H.264 for guaranteed compatibility *and* a modern codec (AV1 or HEVC) for the large fraction of devices that support it, so most viewers get the efficient codec and fall back gracefully. The codec matrix lays the trade-off bare.

![A matrix comparing H.264, HEVC, VP9, and AV1 across bitrate saving, encode cost, device support, and royalty status, showing AV1 saving the most bitrate but costing the most to encode](/imgs/blogs/design-a-video-streaming-platform-4.webp)

The reason this matters in dollars: every percentage point of bitrate you shave off a rung is a percentage point off the egress for every view of that rung, forever. At sixteen exabytes a month, a codec that cuts the average delivered bitrate by even 15% saves a staggering amount of money — which is why platforms invest heavily in expensive AV1 encoding even though the encode cost is high. You pay the encode cost *once* and harvest the bitrate savings on *every* view. This is the same "transcode once, serve forever" amortization logic that justified the transcoding farm, applied to codec selection.

## Trade-offs: the decisions a senior defends in review

No design is free; every choice buys something and costs something else. The discipline of a senior is to name both, every time. Here is the trade-off matrix for the major decisions in this system, framed exactly as you would defend them in a design review.

| Decision | What you gain | What you pay | When it wins |
|---|---|---|---|
| **Deeper encoding ladder** (more rungs/codecs) | Better quality match per viewer, lower delivered bitrate, less rebuffering | More transcode compute, more storage, longer time-to-available | High-traffic premium titles where egress savings dwarf encode cost |
| **Shorter segments** (2s vs 6s) | Faster ABR reaction, lower startup latency, lower live latency | More requests, more manifest overhead, slightly worse compression | Live and latency-sensitive playback |
| **Build your own CDN** (Open Connect style) | Near-zero marginal egress, bytes one hop from viewer, ISP goodwill | Huge capital outlay, hardware ops, ISP relationships | Only at the very top of scale where egress dwarfs capex |
| **Aggressive pre-positioning** | Near-100% hit ratio at launch, no thundering herd | Wasted fill bytes on content that flops, prediction complexity | Predictable big releases and known regional taste |
| **Per-title / per-shot encoding** | Lower bitrate at equal quality → direct egress savings | Heavy content analysis, more pipeline complexity | Large catalog where per-title savings compound over millions of views |
| **AV1 over H.264** | Up to 50% less egress per view, royalty-free | Orders-of-magnitude more encode cost, imperfect device support | Hot content on supported devices; pair with H.264 fallback |
| **Cold storage tiering** | Cheap storage for the long tail | Slower first-byte on cold fetches | Rarely-watched catalog where latency-to-play is non-critical |

The two decisions worth dwelling on are the ones that look expensive but pay for themselves. **Building your own CDN** looks like an absurd capital commitment until you put it next to a \$200-million-a-month egress bill, at which point the appliances pay back in months. **Per-title encoding** looks like over-engineering until you multiply a 15% bitrate reduction across exabytes of monthly egress. In both cases the senior move is the same: recognize that an up-front, one-time cost (capex, encode compute) is dwarfed by a recurring, per-view cost (egress), and spend the one-time cost to crush the recurring one. That asymmetry — pay once to save on every view — is the through-line of the entire design, and it is the lens through which the [cost-as-a-design-constraint](/blog/software-development/system-design/cost-as-a-design-constraint-finops) thinking turns architecture decisions into a defensible business case.

### Rejected alternatives, and why

A few designs look tempting and are wrong; naming why you *rejected* them is half of a good design review. **Transcoding on demand** (encode renditions lazily on first request, instead of up front) saves storage on never-watched content, but it is fatal: the first viewer of any segment waits seconds-to-minutes for an encode, blowing the startup budget, and a viral spike triggers a transcode stampede that no farm can absorb. You transcode ahead of time precisely so the read path is pure cache. **Serving from origin with a fat pipe** — "just buy more bandwidth" — ignores that the cost is per-byte egress, not link capacity; a bigger pipe at the same per-GB rate is the same bankrupting bill, faster. **One global encoding ladder** is simpler but leaves bitrate (money) on the table for easy content and quality on the table for hard content. And **a single CDN tier with no shield** seems simpler until the first viral video melts your origin under a synchronized miss storm, which is exactly the failure we turn to next.

## Stress-testing the design: what breaks at 10×?

A design you have not tried to break is a hypothesis, not a plan. Let us subject this one to the three failure modes that actually take down streaming platforms.

**The viral video.** A clip explodes from a thousand views an hour to ten million. The danger is the **cache stampede / thundering herd at the edge**: the moment the video goes hot, every edge PoP gets a flood of requests for segments it does not yet have cached, and naively each of those misses races up to the origin simultaneously — a synchronized miss storm that can put more load on your origin in one minute than it normally sees in a day. Two mechanisms save you. **Origin shielding** funnels all misses through a single mid-tier cache per origin, so a thousand edges asking for the same new segment become one origin fetch that the shield then fans back out. **Request coalescing** (also called request collapsing) at each cache means that if a thousand viewers request the same uncached segment in the same instant, the cache makes *one* upstream request and lets the other 999 wait for its result, instead of forwarding all thousand. Together these turn a multiplicative storm into a single fetch per segment per tier. This is the same stampede problem covered in the [caching strategies post](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite), and the same coalescing solution; the streaming twist is that the objects are huge and immutable, so once filled, the segment is hot forever and the storm is brief.

**The live-event spike.** Ten million people press play on a live final at the same minute. Unlike VOD, you cannot pre-position content that does not exist yet, and the audience arrives *synchronized* — the worst possible request pattern. The defenses are different. First, the live encoder produces segments just ahead of playback, so the working set is tiny (the last few segments) and stays hot in every edge cache — a live stream is almost the perfect CDN workload because everyone watches the same few segments at once, so the hit ratio is enormous after the first viewer per PoP. Second, you provision the control plane (the entitlement and manifest path) for the synchronized join, because ten million simultaneous "start playback" calls hit your API at the same second even though the bytes are fine; you smooth this with a low-latency entitlement check, cached signed tokens, and sometimes a virtual waiting room. Third, the latency budget is unforgiving — viewers comparing notes on social media notice a thirty-second delay — so the live ladder runs short segments and you accept the compression penalty. The bottleneck in a live spike is rarely the bytes; it is the synchronized control-plane join and the encoder keeping up in real time.

**The new-release thundering herd.** A tentpole release drops Friday at midnight across every timezone's midnight. This is the most *predictable* of the three, which is exactly why it should never hurt you. Because you know the title and the time in advance, you **pre-position** aggressively: in the hours before launch, you push the full ladder of the new title out to every relevant edge cache and Open-Connect-style appliance, during off-peak when the fill bytes are cheap and the network is idle. When midnight hits, the launch is served entirely from warm caches at a near-100% hit ratio, and your origin never feels it. The failure mode is *not* pre-positioning a known release and letting the launch generate a cold-miss storm — an entirely self-inflicted wound. The senior lesson generalizes: a predictable spike is a scheduling problem, not a scaling problem. You do not scale up to meet a herd you saw coming; you arrange for the herd to find the data already there.

Across all three, notice the unifying principle: **protect the origin and protect the hit ratio.** Every failure mode is some path by which traffic that should have been absorbed by cache instead reaches your origin, and every defense is a way of collapsing many would-be origin requests into few or zero. The origin is the one tier you cannot scale by money alone, so the whole design is an apparatus for keeping it idle.

#### Worked example: surviving a viral segment

A video goes from cold to **2 million concurrent viewers** in five minutes. At 3 Mbps that is 6 Tbps of egress — large but exactly what the CDN exists to serve. The real risk is the origin.

Without coalescing: as the video heats up, each of, say, **2,000 edge PoPs** misses on each new segment. For a fresh segment, that is 2,000 simultaneous origin fetches *per segment*; across the segments being requested in the ramp, the origin could see tens of thousands of concurrent fetches — enough to saturate its egress and tip it over, which would then fail the misses for *everything else* too.

With a two-level shield and coalescing: the 2,000 edge misses for a given segment collapse at the regional tier (say 20 regions) into ~20 mid-tier requests, which collapse at the origin shield into **1 origin fetch per segment**. The origin serves each new segment exactly once; the shield fans it out to the 20 regions; each region fans it out to its ~100 edges; each edge serves its share of the 2M viewers from then on. The origin's load is bounded by *the number of distinct new segments*, not by viewer count — so 2 million viewers cost the origin the same as 2 viewers would. That bound is the whole point of the hierarchy.

## Live versus VOD: the same backbone, two pipelines

We have treated live as an aside throughout; let us pull the contrast together, because understanding *why* the two diverge sharpens the whole design. They share the player, the ABR algorithm, the CDN, and the control plane. They differ entirely in the upstream half — and the difference is driven by one variable: how much time you have between content existing and content being watched.

For **VOD**, that gap is effectively infinite. The content already exists, so you can take hours to transcode it into a deep, multi-codec ladder with maximum compression efficiency, pre-position it to the edge before anyone watches, and serve every byte from a warm cache. You optimize for compression (every saved bit is egress saved forever) and for breadth of ladder (every device gets its ideal rendition). Latency-to-available is a convenience, not a constraint; the only hard latency budget is the viewer's time-to-first-frame, which the CDN and start-low-ramp-up heuristic handle.

For **live**, the gap is *seconds*. The content is being created right now, so the encoder must transcode the incoming feed in real time, package it into short segments on the fly, and push them to the edge while the event is still happening. This inverts the optimization priorities. You cannot afford a deep ladder or the slowest, most efficient codec, because you are racing a clock measured in seconds — so the live ladder is shallow and uses fast codecs, accepting worse compression (more egress per view) as the price of hitting the latency budget. Glass-to-glass latency (camera to viewer screen) becomes the headline metric; standard HLS/DASH live runs around 10–30 seconds of delay, and low-latency variants (LL-HLS, LL-DASH) push toward 2–5 seconds with smaller segments and chunked transfer, at the cost of more requests and a thinner buffer that is more fragile to network jitter.

The audience pattern differs too, and in live's favor on one axis. VOD viewers are spread across the catalog and across time, so any given segment is requested by a diffuse population. Live viewers are *synchronized* — everyone watches the same few most-recent segments at the same instant — which is the *best* possible CDN workload, because the working set is tiny (the last handful of segments) and white-hot in every edge cache after the first viewer per PoP. The flip side is the synchronized *join*: ten million people pressing play in the same minute hammer the control plane's entitlement and manifest path simultaneously, which is the live-specific scaling problem we stress-tested earlier. So live trades a harder control-plane join and a brutal encoder latency budget for an easier data-plane caching story, while VOD trades an easy, deferred pipeline for a diffuse, long-tail caching story. Same backbone, opposite pressures — and a mature platform runs both configurations side by side, routing each title down the pipeline that fits it, exactly as the encoding-ladder decision tree above prescribes.

## How real systems do it

The architecture above is not hypothetical; it is, in its shape, how the platforms you use actually work. A few concrete examples, each with the lesson it teaches.

**Netflix — Open Connect and per-title encoding.** Netflix runs its own CDN, Open Connect, placing storage-dense caching appliances directly inside ISP networks and internet exchanges, given to ISPs at no cost. It pre-fills these appliances during off-peak hours with content its models predict each region will watch, so peak traffic is served from a box one hop from the viewer with essentially zero backhaul. On the encoding side, Netflix pioneered per-title and per-shot encoding — analyzing each title's (and each scene's) complexity to assign a custom bitrate ladder, shaving delivered bitrate without hurting quality. **The lesson:** at sufficient scale, the CDN and the codec are not vendor relationships to manage but core competencies to own, because the egress savings dwarf the engineering cost. Egress is the business; everything optimizes it.

**YouTube — transcoding at planetary scale and the long tail.** YouTube ingests an extraordinary volume of new content continuously and must transcode all of it into multiple formats and codecs, which is why Google built custom silicon (the Argos VCU, a video transcoding ASIC) to make transcoding cheaper and faster than general-purpose CPUs could. It serves a catalog with an enormous long tail — most videos get few views — which makes storage tiering and selective rendition generation (don't make every codec for a video nobody watches) essential. Google's Global Cache nodes sit inside ISPs, mirroring the Open Connect strategy. **The lesson:** when transcoding volume is the bottleneck, specialized hardware and *demand-aware* rendition generation (transcode lazily or partially for the cold tail) are the levers, and the long tail makes tiering non-optional.

**Live sports and the synchronized-join problem.** Platforms that stream live sports — where tens of millions join within seconds of kickoff — consistently report that the hard part is not the steady-state bytes but the synchronized control-plane join and the unforgiving latency budget. The defenses are the ones above: short low-latency segments, heavily cached entitlement and manifest paths, virtual waiting rooms to shape the join, and aggressive monitoring of glass-to-glass latency. **The lesson:** a synchronized audience attacks the control plane and the encoder, not the data plane; the bytes are easy because everyone wants the same few segments, which is the best cache workload there is.

**Multi-region resilience.** Serious platforms run the control plane and origin storage across multiple regions, so that a regional outage degrades but does not down the service. The data plane is inherently multi-region (the CDN is global), but the origin and control plane need deliberate geo-distribution — replicated metadata, regional origin stores, failover that does not strand viewers mid-stream. The mechanics of doing this without splitting your data or your latency are the subject of [multi-region and geo-distribution](/blog/software-development/system-design/multi-region-and-geo-distribution); the streaming-specific point is that the data plane gives you geographic resilience nearly for free, but the control plane and origin must earn it. **The lesson:** decoupling the planes pays a resilience dividend — the heavy, hard-to-replicate bytes are already everywhere, so only the light metadata needs careful geo-distribution.

## When to reach for this architecture (and when not to)

This is a big, expensive design, and most products that play video do not need most of it. The discipline is matching the architecture to the actual scale and stakes.

**Do not build this if** your video is incidental — a few product demos, a marketing reel, occasional webinars. Upload to a managed video service (a Mux, a Cloudflare Stream, an AWS MediaConvert plus CloudFront, a hosted player) and let them run the pipeline and the CDN. You get adaptive bitrate, transcoding, and a global CDN as a product, for a per-minute fee, with none of the operational burden. At low-to-moderate volume the managed service is dramatically cheaper than the engineering time to replicate even a fraction of this, and you should not write a line of transcoding orchestration. The break-even is real and it is high; respect it.

**Build the custom pipeline when** video is your product and the volume is large enough that the managed-service per-minute fees exceed the cost of running your own transcoding farm and negotiating your own CDN deal — typically when you are delivering many petabytes a month and your egress bill has become a top line item the CFO asks about. Even then, you build *incrementally*: start on a commercial CDN and a managed transcoder, instrument your hit ratio and your egress relentlessly, and in-source each piece only when the math proves it pays. Building your own CDN appliances is the *last* thing you do, justified only when your egress is so large that the capex is a rounding error against it — which is to say, only at Netflix/YouTube scale. Most platforms live happily forever on commercial CDNs with good pre-positioning and never build a box.

**The universal parts** — regardless of scale — are the principles, not the infrastructure: transcode once into an adaptive ladder, separate the control plane from the data plane so bytes never touch your app servers, serve everything you can from cache, and treat the cache-hit ratio as the cost metric that matters most. Those hold whether you are on a managed service or running your own appliances. The infrastructure scales with you; the principles are constant from day one.

## Key takeaways

- **Egress is the business.** At streaming scale, the bandwidth bill dwarfs every other cost, so the entire architecture is a machine for not serving bytes from your origin. Do the egress math first; it decides everything downstream.
- **The cache-hit ratio is your most valuable number.** The gap between a 95% and a 99% edge hit ratio is millions of dollars a month. Protect it with cache-key discipline, pre-positioning, and origin shielding — and measure it like revenue.
- **Transcode once, serve forever.** Pay the encode cost a single time per title to amortize it across every future view. This logic justifies the transcoding farm, per-title encoding, and expensive AV1 alike: a one-time cost to crush a recurring one.
- **Chunk to parallelize.** Splitting a source at GOP boundaries turns one multi-hour serial encode into thousands of idempotent parallel jobs, collapsing wall-clock time and letting you run the farm on cheap preemptible instances.
- **Adaptive bitrate is the key UX win.** Because you built a ladder, the player switches rendition per segment to track the live network — start low and ramp up to nail the two-second startup, then weight rebuffering avoidance above raw resolution.
- **Separate the control plane from the data plane.** Metadata and auth flow through your services; bytes flow from the CDN. This split keeps your app tier small, your bill on the CDN, and your failure modes independent.
- **Tier storage by popularity.** The CDN cache and cold archive are the same optimization at two layers — popularity-aware placement. One good prediction of what viewers will watch pays off twice, at the edge and in the storage bill.
- **A predictable spike is a scheduling problem, not a scaling one.** Pre-position known releases ahead of the herd; you do not scale to meet a launch you saw coming, you arrange for the data to be already there.
- **At the top of scale, the CDN stops being a vendor and becomes infrastructure.** Open Connect and Global Cache put your cache inside the ISP because, against a nine-figure egress bill, the appliances pay back in months.

## Further reading

- [Back-of-the-envelope estimation for system design](/blog/software-development/system-design/back-of-the-envelope-estimation-for-system-design) — the technique behind the egress and storage math that drives this whole design.
- [Caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) — multi-tier caching, stampedes, and the hit-ratio discipline that the CDN section depends on.
- [Cost as a design constraint (FinOps)](/blog/software-development/system-design/cost-as-a-design-constraint-finops) — why egress dominates content-heavy systems and how to make cost a first-class design input.
- [Queues and event streaming for architects](/blog/software-development/system-design/queues-and-event-streaming-for-architects) — the transcode-job pipeline and the watch-event firehose both ride on a durable log.
- [Multi-region and geo-distribution](/blog/software-development/system-design/multi-region-and-geo-distribution) — how to make the origin and control plane survive a regional failure while the data plane stays global.
- [Idempotency and exactly-once by design](/blog/software-development/system-design/idempotency-and-exactly-once-by-design) — why deterministic, retryable encode jobs are what make a spot-instance transcoding farm safe.
- Next in this series: [Design a search and autocomplete system](/blog/software-development/system-design/design-a-search-and-autocomplete-system) — from inverted indexes to type-ahead latency budgets, the read-path counterpart to this delivery-heavy build.
- Apple HLS and the MPEG-DASH specifications; Netflix's Open Connect and per-title/per-shot encoding engineering posts; the Pensieve and BOLA papers on adaptive bitrate algorithms — for the mechanism depth behind the player and the pipeline.
