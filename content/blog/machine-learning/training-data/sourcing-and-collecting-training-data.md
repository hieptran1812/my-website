---
title: "Sourcing and Collecting Training Data: From Common Crawl to Your Own Crawler"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Where pretraining data actually comes from, why serious labs mine WARC instead of the lossy WET, what a snapshot really contains, and how to estimate the clean-token yield before you spend a dollar on storage."
tags: ["training-data", "common-crawl", "web-crawling", "data-pipeline", "warc", "deduplication", "llm-pretraining", "data-engineering", "fineweb", "data-sourcing"]
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 35
---

Ask someone where a large language model's training data comes from and you will usually get one of two answers: "the internet" or "Common Crawl." Both are technically true and operationally useless. The internet is not a dataset; it is a hostile, duplicated, mostly-boilerplate stream of bytes that has to be hunted, captured, filtered, and re-filtered before a single token is worth a forward pass. And "Common Crawl" is not your dataset either — it is the quarry. What you ship to the model is the rock you cut out of it and grind down, and most of what you start with ends up on the floor.

This post is the first real step of the data pipeline: **getting raw data in**. Not cleaning it (that comes next), not deduping it, not tokenizing it — just sourcing and collecting the raw material. We will walk the full menu of where data comes from, then spend most of our time inside Common Crawl because it is the dominant source and the one people misunderstand most. We will look at the difference between WARC, WAT, and WET records and why that distinction quietly decides the quality ceiling of your entire model. We will build the mental model of running your own crawler and when that is worth it. And we will do the arithmetic that nobody shows you: how many clean tokens actually survive when you pour ten petabytes of HTML into the top of the funnel.

![The training-data sourcing menu](/imgs/blogs/sourcing-and-collecting-training-data-1.webp)

The diagram above is the mental model for the whole post: six ways data gets into a training set, each scored on the four axes that decide which one you reach for — cost, quality, legal risk, and freshness. There is no free lunch on this menu. The cheapest, most abundant source (web crawl) is also the lowest quality and the legally murkiest. The cleanest source (human-written, or licensed under a signed contract) is the most expensive and the slowest to scale. Every serious data effort is a blend across this table, weighted by what the model needs and what the lawyers will sign off on. The rest of this article is a tour of that table, with Common Crawl front and center because it is where the petabytes live.

> Collecting data is the cheap part. Deciding what to throw away — and proving you were allowed to keep the rest — is the job.

## 1. The sourcing menu: six ways data gets in

Before we go deep on the web, it is worth laying out the whole menu so you know what Common Crawl is *instead of*. There are six broad sources of pretraining and post-training data, and a mature program uses all of them in different proportions.

**Senior rule of thumb: pick your source from the constraint that binds, not the one that is easiest.** If you are token-starved, you reach for web crawl. If you are quality-starved on a narrow capability, you reach for human-written or licensed. If you are starved for a capability that does not exist on the web yet, you reach for synthetic. The menu is a set of trades, not a ranking.

| Source | Cost | Quality (raw) | Legal risk | Freshness | Scale ceiling | Reach for it when |
| --- | --- | --- | --- | --- | --- | --- |
| **Web crawl** (Common Crawl or your own) | ~\$0 ingest, \$\$ storage/compute | Low, very noisy | Gray area (copyright, robots, TDM) | Monthly | Petabytes / trillions of tokens | You need raw scale and breadth |
| **Public datasets** (HF Hub, academic) | Free | Often high (pre-curated) | Varies per dataset license | Static (frozen at release) | Tens of TB | You want a known-good baseline fast |
| **Licensed / purchased** | \$\$\$ | High | Clean (signed contract) | Negotiable | Bounded by the deal | You need provenance you can defend |
| **Product telemetry** | Infra only | Very high (domain-matched) | High (PII, privacy, consent) | Real-time | Grows with usage | You have users and a feedback loop |
| **Human-generated** (annotation, writing) | \$\$\$, slow | Top | Clean (you commissioned it) | On demand | Thousands–millions of items | You need a capability nothing else covers |
| **Synthetic** (model-generated) | Compute | Variable | Source-model terms of service | On demand | Compute-bound | The capability exists in a teacher model |

A few things on this table are worth saying out loud because they trip people up.

**Public datasets are not free of the web.** When you `load_dataset("HuggingFaceFW/fineweb")` you are pulling fifteen trillion tokens that *came from* Common Crawl — someone else already paid the extraction and filtering cost and published the result. This is the single highest-leverage move in data sourcing: stand on the shoulders of a published, documented, reproducible corpus instead of re-mining the quarry yourself. The catch is that you inherit their decisions (which snapshots, which extractor, which filters) and their license. Read the dataset card like a contract, because it is one.

**Product telemetry is the highest-quality web-scale source you can have, and the one most likely to get you sued or fined.** Logs of how users actually phrase requests, which completions they accept, what they edit — this is gold for post-training, because it is exactly the distribution your model is deployed against. It is also full of personal data, governed by your privacy policy and laws like GDPR and CCPA, and frequently subject to consent you did not collect for "train a model." We will come back to this as the data flywheel, because it is the engine that compounds, but the legal surface area is enormous.

**Synthetic data is not a way to make data from nothing.** It is a way to *transfer* a capability that already exists in a teacher model into a smaller or more specialized student, or to amplify a narrow seed of human data. It is bounded by the teacher's quality and by the terms of service of whatever model you generated it with — many commercial model providers explicitly forbid using outputs to train competing models. Synthetic data also risks model-collapse-style feedback loops if you train on your own outputs without fresh human grounding.

The remaining three rows — web crawl, licensed, human-generated — are the load-bearing sources for pretraining, post-training, and capability-specific work respectively. For the rest of this post we focus on the first one, because it is where the petabytes are and where the most engineering judgment is required. If you want the theory of *how many tokens you actually need* and what happens when you run out, that is its own rabbit hole — see [data-constrained scaling laws](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) for the demand side of this equation. This post is the supply side.

## 2. Inside Common Crawl: WARC vs WAT vs WET

Common Crawl is a nonprofit that has been crawling the web since 2008 and publishing the results for free. Roughly once a month it releases a **snapshot** (the project calls them "crawls" or "dumps"), named like `CC-MAIN-2024-22` — year and ISO week. A recent snapshot contains on the order of **2.5 to 3.5 billion web pages** and weighs roughly **90 to 130 TB compressed** in its primary format. The full archive since 2008 is well over **250 billion pages** and several petabytes. It lives in a public S3 bucket (`s3://commoncrawl/`, us-east-1) and is mirrored over HTTPS at `data.commoncrawl.org`. You do not need an account or an API key. You need bandwidth and a plan.

The single most important thing to understand about Common Crawl — the thing that separates people who get good data out of it from people who get garbage — is that **each crawled page is published in three different record types**, and they are not interchangeable.

![Common Crawl ships three record types per page](/imgs/blogs/sourcing-and-collecting-training-data-2.webp)

When the crawler fetches one URL, it captures the full HTTP response. That single response is then written out in three derived forms:

- **WARC** (Web ARChive, the ISO 28500 standard) is the source of truth. It contains the request, the full HTTP response *including headers*, and the complete raw payload — the original HTML bytes exactly as the server sent them. Everything else is derived from this. If you want to do your own extraction with a modern tool, this is the only record type that has enough information to do it well.
- **WAT** (Web Archive Transformation) contains *metadata only*, serialized as JSON: the HTTP headers, the page title, the outbound and inbound links, and other structured fields. It is fantastic for link-graph analysis (building a web graph, computing PageRank-like signals, finding hub domains) and useless as a text source. Do not train on WAT.
- **WET** (WARC Encapsulated Text) contains *plain text only* — Common Crawl's own extraction of the page's text, with HTML stripped. This is the trap.

Here is the comparison that should be tattooed on the wall of every data team:

| Property | WARC | WAT | WET |
| --- | --- | --- | --- |
| Contents | Full HTTP response + raw HTML | Metadata + links (JSON) | Plain text extraction |
| Size per snapshot | ~90–130 TB | ~20–25 TB | ~8–12 TB |
| HTML structure preserved | Yes (it *is* the HTML) | N/A | No |
| Boilerplate removed | No (you do it) | N/A | Mostly not — nav/footer text survives |
| Suitable for training text | After your own extraction | No | Tempting, but lossy |
| Who should use it | Serious pretraining | Web-graph work | Quick experiments only |

**Senior rule of thumb: if you are building a real corpus, extract from WARC. WET is a demo, not a dataset.** WET is seductive because it is small (a tenth the size of WARC) and pre-extracted (no HTML parsing on your side). But Common Crawl's WET extraction is deliberately generic and conservative. It keeps a great deal of boilerplate — navigation menus, cookie banners, "related articles" rails, footer link farms — because a one-size-fits-all extractor cannot know which `<div>` is the article and which is the sidebar. It also loses document structure entirely (no paragraph boundaries you can trust, no headings, no list semantics) and, on some page layouts, silently drops the main content while keeping the chrome. The net effect is that a WET-derived corpus has a *lower quality ceiling* than a WARC-derived one, and no amount of downstream filtering fully recovers the gap — you cannot filter back in text that was never extracted.

This is not a hypothetical. The two corpora that reset everyone's expectations for web data — RefinedWeb and FineWeb — both went back to WARC and ran their own extraction, and both reported measurable downstream gains over the WET baseline. We will dig into exactly what they did in the case study. The headline: the WARC-vs-WET choice is made once, costs you ~10x the storage and a real extraction pipeline, and quietly sets the ceiling on everything that follows. The actual extraction techniques — boilerplate removal, main-content detection, structure recovery — are a deep topic of their own, covered in the sibling post on [text extraction and boilerplate removal](/blog/machine-learning/training-data/text-extraction-and-boilerplate-removal).

### Reading a WARC, in practice

Let us make this concrete. A WARC file is a sequence of records; you stream it, filter for HTTP responses, and pull the raw HTML. The `warcio` library makes this a dozen lines. The paths to every WARC file in a snapshot are listed in a gzipped manifest at `s3://commoncrawl/crawl-data/CC-MAIN-2024-22/warc.paths.gz`.

```python
import requests
from warcio.archiveiterator import ArchiveIterator

# One WARC file from the May 2024 crawl (CC-MAIN-2024-22). A full snapshot is
# ~90,000 of these, each ~1 GB compressed.
warc_url = (
    "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-22/"
    "segments/1715971057216.0/warc/"
    "CC-MAIN-20240517041940-20240517071940-00000.warc.gz"
)

resp = requests.get(warc_url, stream=True)
for record in ArchiveIterator(resp.raw):
    # Three record types live in a WARC: 'request', 'response', 'metadata'.
    # We only want successful page bodies.
    if record.rec_type != "response":
        continue
    url = record.rec_headers.get_header("WARC-Target-URI")
    status = record.http_headers.get_statuscode()
    ctype = (record.http_headers.get_header("Content-Type") or "")
    if status != "200" or "text/html" not in ctype:
        continue

    html = record.content_stream().read()      # raw bytes — the source of truth
    print(url, len(html), "bytes of HTML")
    # Hand `html` to a real extractor (trafilatura / resiliparse), NOT to WET.
    # text = trafilatura.extract(html, ...)
```

The thing to notice is that `record.content_stream().read()` hands you the *exact bytes the server returned*. You now own the extraction decision. That is the whole point of starting from WARC: you can run trafilatura, resiliparse, jusText, or your own DOM-aware extractor, tuned for the kinds of pages you care about, instead of accepting Common Crawl's generic WET output.

### Finding pages without downloading the haystack

Streaming 90 TB to find a handful of pages from one domain is absurd. Common Crawl publishes two indexes so you can locate exactly which WARC file, at which byte offset, holds the capture you want.

The first is the **CDX index**, a per-crawl, queryable URL index with an HTTP API:

```bash
# Every capture of example.com/* in the May 2024 crawl, as JSON lines.
curl -s 'https://index.commoncrawl.org/CC-MAIN-2024-22-index?url=example.com/*&output=json&limit=3'
```

```json
{"urlkey": "com,example)/", "timestamp": "20240517...", "url": "https://example.com/",
 "mime": "text/html", "status": "200", "digest": "SHA1:...",
 "filename": "crawl-data/CC-MAIN-2024-22/segments/.../warc/CC-MAIN-...-00042.warc.gz",
 "offset": "738291043", "length": "9214"}
```

That `filename` + `offset` + `length` triple is a recipe for a **single HTTP range request**: you fetch 9 KB out of a 1 GB file and decode exactly one record. This is how targeted collection (build a corpus of one domain, one TLD, one language) is done without moving petabytes.

The second is the **columnar index**: the same capture metadata, but published as Parquet and partitioned by crawl and subset. You can query it with DuckDB or AWS Athena and never touch a WARC until you have decided what you want.

```sql
-- Which registered domains contributed the most English HTML pages to one
-- crawl? Answered straight from the Parquet index — zero WARC downloads.
INSTALL httpfs; LOAD httpfs;
SET s3_region = 'us-east-1';

SELECT url_host_registered_domain AS domain,
       count(*)                   AS pages
FROM read_parquet(
       's3://commoncrawl/cc-index/table/cc-main/warc/'
       'crawl=CC-MAIN-2024-22/subset=warc/*.parquet')
WHERE content_languages = 'eng'
  AND fetch_status      = 200
GROUP BY domain
ORDER BY pages DESC
LIMIT 20;
```

The columnar index carries the columns you actually plan filters around: `url_host_registered_domain`, `content_languages`, `content_mime_type`, `fetch_status`, and the `warc_filename` / `warc_record_offset` / `warc_record_length` pointers back into the WARC. **Second-order optimization: do your source selection in the index, not in the crawl.** Deciding "English, status 200, these TLDs, not these spam domains" against a few terabytes of Parquet is cheap; deciding it by streaming 90 TB of WARC is not. The index is the cheapest place to be picky.

## 3. The monthly snapshots and their biases

People hear "monthly snapshots since 2008" and assume that means roughly 200 independent samples of the web, so 200x the data of one snapshot. That intuition is wrong in two important ways, and both have direct consequences for how many snapshots you should actually pull.

![Monthly snapshots overlap more than they refresh](/imgs/blogs/sourcing-and-collecting-training-data-3.webp)

The figure above is the key insight: **successive snapshots overlap heavily.** The crawler re-discovers the same high-popularity "head" domains every single month — Wikipedia, major news sites, large e-commerce, popular forums — because those domains are densely linked and the frontier keeps finding them. So the head of the distribution is re-crawled constantly, and a large fraction of any new snapshot's URLs were already in the previous one. The "long tail" — small sites, new pages, churning news content — is what genuinely rotates between snapshots. As a rough order of magnitude, a large fraction of domains in one monthly snapshot were also present in the prior one; the genuinely new material is a minority of each dump.

This produces two biases you must plan around:

**Recency and coverage bias.** Common Crawl is a *sample*, not a census. Its frontier is seeded and link-driven, so it over-represents well-linked, SEO-optimized, English-language, commercially valuable pages and under-represents the deep web, paywalled content, login-gated communities, non-Latin scripts, and anything that actively blocks crawlers (which increasingly includes the highest-quality publishers). The web Common Crawl sees is the *public, linkable, crawler-tolerant* web — which is systematically different from "the web."

**Duplication-across-snapshots bias.** Because the head is re-crawled monthly, pulling N snapshots does not give you N times the unique content. It gives you something far smaller, because near-duplicate detection across snapshots collapses the repeats. This is the single biggest reason that naively "just grab all the snapshots" is a bad plan — you pay N times the storage and extraction cost for a sublinear gain in unique tokens. The cross-snapshot deduplication problem is severe enough that it gets its own treatment in [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale); here, just internalize that **more snapshots is a decision with sharply diminishing returns, not a free win.**

The practical guidance: choose snapshots deliberately. For breadth and recency you want a spread across time (a snapshot from each year captures different web eras and different content) rather than ten consecutive months (which are near-copies of each other). FineWeb, as we will see, sampled 96 dumps spanning 2013 to 2024 precisely to get temporal breadth, then deduplicated *within* each dump but treated cross-dump overlap carefully.

## 4. Positioning the sources: cost, quality, legal, freshness

We have now seen the dominant source in detail. Step back and place all six sources on the same map, because the choice is never "Common Crawl or not" in isolation — it is "what blend, given my constraints."

![Positioning the sources: cheap-and-risky vs costly-and-clean](/imgs/blogs/sourcing-and-collecting-training-data-4.webp)

The pattern in this figure is the central tension of data sourcing: **cost and legal cleanliness move together, inversely to scale.** The cheap, web-scale sources in the top row (web crawl, public datasets, product telemetry) carry the messy provenance — gray-area copyright, varying licenses, PII. The costly sources in the bottom row (licensed, human-written) are the ones where you can actually point to a contract or a commission and say "we are allowed to use this, here is the paper trail." Synthetic sits awkwardly between: cheap to scale but legally entangled with the terms of whatever model produced it.

There is a deeper point here about *risk-adjusted* value. A token of Common Crawl text and a token of licensed-news text are not worth the same, even at the same quality, because they do not carry the same liability. As the legal landscape around training data hardens — and it is hardening fast, with active litigation and new text-and-data-mining regulation — the provenance of your corpus is becoming a first-class engineering concern, not a footnote. The closing sibling post, [legal, ethics, and the future of training data](/blog/machine-learning/training-data/legal-ethics-and-the-future-of-training-data), is entirely about this axis; for sourcing, the rule is simple.

**Senior rule of thumb: tag every byte with its provenance at ingest time, not later.** The moment data enters your system, record where it came from, under what license or terms, when, and which filters it passed. Provenance you do not capture at ingest is provenance you will never reconstruct, and a corpus you cannot account for is a corpus you cannot ship. This single discipline — a provenance column next to every shard — is the cheapest insurance in the entire pipeline.

## 5. Running your own crawler

Common Crawl is the right default, but there are real reasons to run your own crawler: you need fresher data than monthly, you need domains or content types Common Crawl under-samples, you need to respect a specific publisher's terms via a direct relationship, or you need the *raw response* fidelity for something Common Crawl strips. Before you do, understand what you are signing up for, because a crawler is mostly infrastructure that has nothing to do with machine learning.

![Your own crawl pipeline: fetch, store, index, extract](/imgs/blogs/sourcing-and-collecting-training-data-5.webp)

The pipeline above is the honest picture. The model only ever sees the last box — the clean text shards. Everything before it is storage and politeness plumbing:

1. **Seed URLs + frontier queue.** You start from seed URLs and maintain a frontier — the set of discovered-but-not-yet-fetched URLs, prioritized. The frontier is where most crawler intelligence lives: how you prioritize determines whether you spend your budget on quality pages or drown in calendar widgets and faceted-search URL explosions.
2. **Politeness: robots.txt + rate limiting.** This is non-negotiable and we cover it below.
3. **Fetcher: HTTP + URL dedup.** The actual HTTP client, plus deduplication of URLs (canonicalization, stripping tracking params, collapsing `http`/`https` and `www`) so you do not fetch the same page a thousand times under a thousand URLs.
4. **WARC store: raw bytes, append-only.** You write what you fetch to WARC, append-only, exactly like Common Crawl does — because you want the same "extract later, extract again, extract better" optionality.
5. **CDX / columnar index.** You build your own index over your WARCs so you can find and re-process captures without rescanning.
6. **Extractor: WARC to clean text.** The same WARC-to-text step we discussed, now on your own bytes.
7. **WebDataset / Parquet shards.** The final form the training loader reads — more on storage formats below.

### Politeness is the whole game

The fastest way to get your crawler IP-banned, your company a cease-and-desist, and your data effort shut down is to crawl rudely. Politeness is a hard requirement, and it is mostly four rules: identify yourself with a real User-Agent and contact, honor `robots.txt`, rate-limit per host, and back off on errors.

```python
import time
import urllib.robotparser as robotparser
from collections import defaultdict

UA = "AcmeResearchBot/1.0 (+https://acme.example/crawler; data@acme.example)"
PER_HOST_DELAY = 2.0  # seconds between hits to the SAME host — be generous

_last_hit: dict[str, float] = defaultdict(float)
_robots: dict[str, robotparser.RobotFileParser | None] = {}

def allowed(host: str, path: str) -> bool:
    """Honor robots.txt. Cache per host. Unreachable robots is NOT open season."""
    if host not in _robots:
        rp = robotparser.RobotFileParser()
        rp.set_url(f"https://{host}/robots.txt")
        try:
            rp.read()
        except Exception:
            rp = None  # could not read robots -> be conservative, treat as allowed
        _robots[host] = rp
    rp = _robots[host]
    return True if rp is None else rp.can_fetch(UA, path)

def polite_get(host: str, path: str):
    if not allowed(host, path):
        log_excluded(host, path)   # respect the exclusion and record it
        return None
    wait = PER_HOST_DELAY - (time.time() - _last_hit[host])
    if wait > 0:
        time.sleep(wait)           # per-host rate limit, not global
    _last_hit[host] = time.time()
    return fetch(f"https://{host}{path}",
                 headers={"User-Agent": UA}, timeout=20)
```

The per-host (not global) rate limit is the subtle part: ten thousand hosts at one request every two seconds is five thousand requests per second in aggregate, which is plenty of throughput, while never hammering any single server. The Robots Exclusion Protocol is now standardized as RFC 9309; many sites also advertise a non-standard `Crawl-delay`, which polite crawlers honor. And `429 Too Many Requests` / `503 Service Unavailable` mean *slow down*, with exponential backoff — not retry-immediately.

**Second-order optimization: budget the frontier, or it will explode.** The classic own-crawler failure is not bandwidth, it is the frontier ballooning into infinity. A single calendar page links to next-month, which links to next-month, forever. A faceted-search page generates a combinatorial explosion of `?color=red&size=M&sort=price` URLs that are all the same page. Crawler traps like these will fill your storage with worthless near-duplicates and starve the quality pages of budget. The defense is URL canonicalization, per-domain page caps, depth limits, and trap-pattern detection in the frontier — long before the extractor ever runs.

### When your own crawler is worth it

| Situation | Common Crawl | Your own crawler |
| --- | --- | --- |
| General-purpose pretraining scale | Yes — default | Rarely worth it |
| Need data fresher than ~1 month | No | Yes |
| Domain Common Crawl under-samples (niche, non-English, deep web) | Partial | Yes |
| Specific publisher relationship / licensed feed | No | Yes (or direct feed) |
| Reproducibility / "documented public source" | Yes (huge plus) | You own the documentation burden |
| Team size to operate it | Zero | A real, ongoing eng investment |

The honest summary: for 90% of pretraining-scale needs, Common Crawl (or a published derivative like FineWeb) is the right answer and rolling your own is a distraction. The 10% where you crawl yourself is real but specific — freshness, coverage gaps, or a licensing relationship — and you should know which of those you are buying before you commit an engineering team to running fetchers forever.

### A note on storage formats

One detail that matters for everything downstream: **the format you collect in is not the format you train from.** You *collect* in WARC because WARC preserves the raw response and lets you re-extract. But WARC is a terrible format to feed a training data loader — it is row-oriented, mixes metadata and payload, and is not built for random access or columnar filtering at training time. So after extraction, the corpus is rewritten into a training-friendly format:

- **WebDataset** — POSIX tar shards of records, streamed sequentially, great for high-throughput sharded reading across many workers.
- **Parquet** — columnar, compressed, filterable; great when you want to slice by language, quality score, or source at load time without rewriting.

The pattern is always the same: **WARC for collection (re-extractable), Parquet/WebDataset for consumption (loadable).** Keep the WARCs. The whole reason to start from raw bytes is so you can extract again with a better tool in six months without re-crawling — and that optionality dies the moment you throw the WARCs away and keep only the extracted text.

## 6. The data flywheel

Everything above is about getting the *first* dataset. The most valuable data source over the long run is the one that did not exist before you shipped: your own product's telemetry. This is the data flywheel, and it is why incumbents with deployed products have a structural data advantage that is very hard to out-crawl.

![The data flywheel: shipping the model is how you collect the next dataset](/imgs/blogs/sourcing-and-collecting-training-data-6.webp)

The loop is simple and compounding. You ship a product. Users interact with it, and that interaction *is data* — the prompts they write, the completions they accept or reject, the edits they make to your model's output, the thumbs-up and thumbs-down. You instrument the product to capture that telemetry. You curate it (dedup, filter, de-identify, label) into a dataset. You train a better model on it. The better model makes a better product, which wins more users, who generate more and better telemetry. Each turn of the wheel compounds the data moat.

This is the highest-quality web-scale source available, because it is *distribution-matched*: it is literally the data your model will be evaluated against in production, not a proxy for it. A million real user interactions are worth more for post-training than a billion random web pages, because they sit exactly on the distribution you care about.

Two things keep the flywheel from being a free lunch, and both are serious:

**The legal and privacy surface is enormous.** Telemetry is personal data. Using it to train requires a lawful basis, almost always consent, and frequently de-identification that survives the fact that language models can memorize and regurgitate. "We logged it, so we can train on it" is how companies end up in regulatory trouble. The flywheel must be built with privacy engineering — consent capture, PII scrubbing, retention limits, opt-outs — as a first-class part of the loop, not a cleanup step.

**Feedback loops can poison the well.** If your model's outputs become its own training data without enough fresh human grounding, you get drift and, at the extreme, model collapse — the distribution narrows around the model's own quirks. The curation step exists partly to keep the flywheel honest: weight human signal heavily, sample synthetic and self-generated content carefully, and keep injecting external (web, licensed, human) data so the loop stays anchored to reality.

The flywheel does not replace web crawl — you still need the broad pretraining base — but it is what turns a one-time data acquisition into a compounding asset.

## 7. A worked scenario: the survival funnel

Now the arithmetic nobody shows you. The question every data engineer eventually has to answer is: *if I pour N Common Crawl snapshots into the top, how many clean training tokens come out the bottom?* The answer is "shockingly few," and the only way to plan storage, compute, and timeline is to walk the funnel with real percentages.

<figure class="blog-anim">
<svg viewBox="0 0 760 470" role="img" aria-label="A six-stage survival funnel: raw HTML at 100 percent shrinks through extraction, language filtering, quality filtering and deduplication to roughly 2 to 3 percent of the original volume as final tokens" style="width:100%;height:auto;max-width:760px">
<style>
.fn-bar{fill:var(--accent,#6366f1)}
.fn-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:end}
.fn-pct{font:700 14px ui-monospace,SFMono-Regular,monospace;fill:var(--text-secondary,#6b7280);text-anchor:start}
.fn-hd{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:start}
.fn-band{fill:var(--accent,#6366f1);opacity:.14}
@keyframes fn-sweep{0%,13%{transform:translateY(0)}15%,28%{transform:translateY(64px)}30%,43%{transform:translateY(128px)}45%,58%{transform:translateY(192px)}60%,73%{transform:translateY(256px)}75%,100%{transform:translateY(320px)}}
.fn-sweep{animation:fn-sweep 13s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.fn-sweep{animation:none}}
</style>
<text class="fn-hd" x="20" y="22">Share of the raw crawl volume that survives each collection stage</text>
<rect class="fn-band fn-sweep" x="0" y="28" width="760" height="48" rx="6"/>
<text class="fn-lbl" x="235" y="58">Raw HTML (WARC)</text>
<rect class="fn-bar" x="250" y="36" width="430" height="34" rx="4"/>
<text class="fn-pct" x="692" y="58">100%</text>
<text class="fn-lbl" x="235" y="122">Main text extracted</text>
<rect class="fn-bar" x="250" y="100" width="95" height="34" rx="4"/>
<text class="fn-pct" x="357" y="122">~22%</text>
<text class="fn-lbl" x="235" y="186">Language filtered</text>
<rect class="fn-bar" x="250" y="164" width="60" height="34" rx="4"/>
<text class="fn-pct" x="322" y="186">~14%</text>
<text class="fn-lbl" x="235" y="250">Quality filtered</text>
<rect class="fn-bar" x="250" y="228" width="30" height="34" rx="4"/>
<text class="fn-pct" x="292" y="250">~7%</text>
<text class="fn-lbl" x="235" y="314">Deduplicated</text>
<rect class="fn-bar" x="250" y="292" width="15" height="34" rx="4"/>
<text class="fn-pct" x="277" y="314">~3.5%</text>
<text class="fn-lbl" x="235" y="378">Final clean tokens</text>
<rect class="fn-bar" x="250" y="356" width="10" height="34" rx="4"/>
<text class="fn-pct" x="272" y="378">~2.4%</text>
<text class="fn-hd" x="20" y="438">The highlight steps down one stage at a time; each cut removes most of what the prior stage kept.</text>
</svg>
<figcaption>The collection funnel is brutal: of the raw bytes a crawl pulls in, only about 2 to 3 percent survive extraction, language and quality filtering, and deduplication to become trainable tokens.</figcaption>
</figure>

Let us run the numbers for a concrete plan: **you decide to source 10 Common Crawl snapshots, spread one per recent quarter for temporal breadth.** Here is the funnel, stage by stage, with the surviving fraction expressed as a share of the original raw volume.

| Stage | Survives (of raw) | What gets cut | Typical tool |
| --- | --- | --- | --- |
| Raw HTML (WARC) | 100% | — | warcio |
| Main text extracted | ~22% | HTML markup, boilerplate, nav/footers | trafilatura / resiliparse |
| Language filtered | ~14% | Non-target-language pages | fastText langid / CLD3 |
| Quality filtered | ~7% | Spam, SEO farms, gibberish, too-short | heuristics + classifier |
| Deduplicated (within snapshot) | ~3.5% | Near- and exact-duplicate documents | MinHash / suffix array |
| Final clean tokens | ~2.4% | PII removal, final formatting, edge drops | PII scrub + tokenizer |

Walk it with absolute numbers. A single recent snapshot is roughly **3 billion pages, ~90 TB compressed WARC**. The *text-bearing* portion of that HTML — what you could in principle extract — is on the order of **~25 TB of raw HTML text** per snapshot. Apply the funnel:

- Extracted main text: 25 TB x 22% ... but we track against raw, so ~22% of 25 TB ≈ **5.5 TB** of main text.
- Language-filtered (English, say ~64% of extracted survives): ~14% of raw ≈ **3.5 TB**.
- Quality-filtered (drop ~50%): ~7% of raw ≈ **1.75 TB**.
- Deduplicated within the snapshot (drop ~50%): ~3.5% of raw ≈ **875 GB**.
- Final clean text after PII scrub and formatting: ~2.4% of raw ≈ **600 GB** of clean text.

Now convert text to tokens. English BPE tokenizers land near **~4 characters per token ≈ ~4 bytes per token**. So 600 GB of clean text ≈ **~150 billion tokens per snapshot**. That number is not a coincidence — it is right in line with what FineWeb reports per dump (15 trillion tokens across 96 dumps ≈ ~156 billion tokens per dump). The funnel checks out against a real, published corpus.

So far, 10 snapshots looks like **10 x 150B = 1.5 trillion tokens**. Here is the twist that the snapshot-overlap discussion set up: **that 1.5T is before cross-snapshot deduplication, and it is a lie.** Because the snapshots overlap (the head domains are re-crawled every period), global dedup across the 10 snapshots removes a large fraction — call it ~55% — as cross-snapshot duplicates. The actual unique yield is closer to:

```python
snapshots          = 10
tokens_per_snap    = 150e9      # ~150B clean tokens per snapshot (post per-snapshot funnel)
naive_total        = snapshots * tokens_per_snap          # 1.5e12
cross_snap_dup_rate = 0.55      # ~55% of the naive sum are cross-snapshot dups
unique_tokens      = naive_total * (1 - cross_snap_dup_rate)

print(f"naive:  {naive_total/1e12:.2f} T tokens")    # naive:  1.50 T tokens
print(f"unique: {unique_tokens/1e12:.2f} T tokens")  # unique: 0.68 T tokens
print(f"effective snapshots: {unique_tokens/tokens_per_snap:.1f}")  # 4.5
```

**Ten snapshots yield about 0.68 trillion unique tokens — roughly 4.5 snapshots' worth of unique data.** You paid 10x the download, storage, extraction, and dedup compute for ~4.5x the unique tokens. That is the diminishing-returns curve made concrete, and it is *exactly* why "just grab everything" is a bad plan and why snapshot selection (breadth over consecutiveness) matters.

The storage planning falls out of the same numbers. If you keep the WARCs (you should), budget ~90 TB compressed per snapshot, so ~900 TB just to hold 10 raw snapshots. The extracted-and-filtered output is tiny by comparison (~600 GB of clean text per snapshot before dedup), but the *intermediate* artifacts — extracted-but-unfiltered text, dedup signatures, quality scores — can easily 2–3x your peak storage during processing. **Senior rule of thumb: your peak storage is during processing, not at rest, and it is dominated by the raw WARCs plus intermediates — size your cluster's scratch for the funnel's widest point, not its output.**

## 8. Troubleshooting: where collection quietly goes wrong

Collection failures are insidious because they do not crash — they silently degrade the corpus, and you only discover them as an unexplained ceiling on model quality three weeks into training. Here is the field guide, symptom to root cause to fix.

### Symptom: the model is weirdly good at SEO copy and bad at reasoning

**Root cause: snapshot and domain bias.** Common Crawl over-samples the well-linked, SEO-optimized, commercially valuable web. If you do not actively rebalance, your corpus is disproportionately product pages, listicles, and content-farm articles, because that is what is densely linked and crawler-tolerant. The model learns the *texture* of that text — fluent, shallow, keyword-stuffed — at the expense of the long-form, reasoning-heavy text that is rarer on the public web.

**Fix:** rebalance at source-selection time using the index. Up-weight high-quality domains (reference, academic, documentation, long-form journalism) and down-weight known content farms before extraction. Quality classification (the next stage) helps, but it is far cheaper to not collect the farm than to filter it out after extracting 25 TB. Diversify snapshots across time, not just within one.

### Symptom: storage explodes and unique-token yield is far below plan

**Root cause: near-duplicate explosion across snapshots.** You pulled 10 consecutive monthly snapshots, expected 10x the tokens, and got 4x. The head domains were re-crawled every month, so most of what you extracted is the same documents over and over. Worse, *near*-duplicates (same article with a different ad, a different timestamp, a different "related posts" rail) evade exact-hash dedup and bloat the corpus while teaching the model nothing new.

**Fix:** (1) Select snapshots for temporal breadth, not consecutiveness — one per quarter or year, not ten in a row. (2) Plan for global, cross-snapshot near-duplicate detection (MinHash/LSH), not just per-document exact hashing. (3) Use the WARC `digest` (SHA-1 of the payload) in the index to cheaply drop *exact* re-captures before you even extract them. See [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) for the algorithms.

### Symptom: text quality is mediocre no matter how hard you filter

**Root cause: you extracted from WET, or your WARC extractor is silently dropping content.** WET keeps boilerplate and loses structure, so the quality ceiling is low and downstream filtering cannot recover it. Even from WARC, a misconfigured extractor can silently fail on common layouts — single-page-app shells where the content is JavaScript-rendered (and absent from the raw HTML), unusual encodings, or main-content detectors that mistake the article for chrome and keep the nav instead.

**Fix:** extract from WARC, not WET, full stop. Then *measure* your extractor: sample a few hundred extracted documents and eyeball them against the rendered pages. Track the extraction yield (chars of text per KB of HTML) and alarm on outliers — a domain where yield suddenly drops to near-zero is usually a JS-rendered site or a layout your extractor chokes on. Be explicit that Common Crawl captures the *raw* HTML response, so JavaScript-rendered content that is not in the initial HTML is simply not there — no extractor can recover it.

### Symptom: crawl traps and spam farms dominate your own crawl

**Root cause: an unbudgeted frontier.** A calendar widget, an infinite-scroll endpoint, or a faceted-search page generated millions of near-identical URLs, and your fetcher dutifully crawled them all, filling storage with garbage and starving the quality pages of your crawl budget. Link farms and auto-generated spam sites do this on purpose to manipulate crawlers.

**Fix:** budget the frontier before it explodes — per-domain page caps, URL-pattern trap detection, depth limits, and aggressive URL canonicalization (strip tracking/session params, collapse `?sort=` permutations). Prioritize the frontier by a quality prior (domain reputation, link structure) so budget flows to good pages first. This is a frontier problem, and it must be solved upstream of the extractor.

### Symptom: legal sends you a takedown after the model ships

**Root cause: robots/legal pitfalls ignored at collection time.** You crawled domains that disallowed crawling in `robots.txt`, or you trained on content under a license that forbids it, or you used product telemetry without a lawful basis, and you have no provenance trail to even figure out which shards are affected.

**Fix:** honor `robots.txt` at crawl time and *log every exclusion* (so you can prove you respected it). Tag every byte with provenance — source, license, timestamp, robots status — at ingest. For Common Crawl, understand that the legal status of training on crawled copyrighted text is genuinely unsettled and actively litigated; "Common Crawl published it" is not a license. Build the ability to *remove* a domain's data from a future training run (because you will be asked to), which requires provenance tags you can filter on. The deep treatment is in [legal, ethics, and the future of training data](/blog/machine-learning/training-data/legal-ethics-and-the-future-of-training-data).

## 9. Case study: how FineWeb and RefinedWeb source from Common Crawl

The two corpora that most changed how the field sources web data are RefinedWeb and FineWeb. Both are master classes in the decisions this post is about, and both made the same pivotal call: **go back to WARC.**

### RefinedWeb (TII, 2023)

RefinedWeb, built by the Technology Innovation Institute for the Falcon models, set out to test a then-controversial thesis: that *web data alone*, filtered and deduplicated aggressively enough, could match or beat the curated mixtures (books, Wikipedia, code, papers) that everyone assumed were necessary. To do that credibly, they could not start from the lossy WET files — they needed maximum-fidelity input. So they sourced from Common Crawl's **WARC** records and ran their own extraction with **trafilatura**, a DOM-aware main-content extractor, rather than accepting WET's generic output.

The pipeline was extraction from WARC, then language identification, then a cascade of quality heuristics, then *very* aggressive deduplication — both exact and fuzzy (MinHash), applied at large scale. The result was on the order of **5 trillion tokens**, of which a **600-billion-token** extract was released publicly. The headline finding validated the thesis: models trained on properly refined web-only data were competitive with models trained on curated mixtures. The lesson for sourcing: the *quality of your collection and extraction* matters more than the *prestige of your sources*. Starting from WARC and extracting well beat starting from someone else's pre-digested text.

### FineWeb (Hugging Face, 2024)

FineWeb pushed the same philosophy further and, crucially, documented every decision in the open. The team sourced **96 Common Crawl snapshots spanning summer 2013 through early 2024** — note the deliberate temporal breadth, exactly the snapshot-selection discipline from Section 3, rather than a block of consecutive recent dumps. And they were explicit about the WARC-vs-WET decision: they extracted text from **WARC using trafilatura**, and reported in their public write-up that WARC-plus-trafilatura produced *measurably better* downstream models than the WET baseline. That is the single most-cited empirical confirmation of the central claim of this post.

The full FineWeb pipeline — built on their `datatrove` library — runs extraction, then language filtering, then a carefully ablated set of quality filters (many borrowed and adapted from C4 and Gopher/MassiveText heuristics), then both per-dump and cross-dump deduplication, with each filtering choice validated by training small models and measuring the effect. The output is roughly **15 trillion tokens** of English text, and the FineWeb-Edu variant adds an educational-quality classifier on top to distill a smaller, higher-quality subset. The whole thing is published as a Hugging Face dataset with a detailed technical report.

The contrast worth drawing is with **C4** (the corpus behind T5, 2019), which was built from a *single* Common Crawl **WET** snapshot with heuristic cleaning. C4 was enormously influential and perfectly reasonable for its time, but it sits on the other side of the WARC/WET line — and the trajectory from C4 (one WET dump) to RefinedWeb and FineWeb (dozens of WARC dumps, custom extraction) is the field collectively learning the lessons in this post. The direction of travel is unambiguous: more snapshots chosen for breadth, extracted from WARC, with the extraction and filtering treated as the core IP.

**The takeaways from both:** (1) source from WARC, extract yourself; (2) choose snapshots for temporal breadth; (3) deduplicate hard, including across snapshots; (4) document every decision so the corpus is reproducible and accountable. And the highest-leverage move of all for most teams: these corpora are *published*. Unless you have a specific reason to re-mine the quarry, `load_dataset("HuggingFaceFW/fineweb")` gives you the output of all this work for the cost of bandwidth.

## 10. When to reach for Common Crawl, and when to crawl yourself

### Reach for Common Crawl (or a published derivative) when

- You need **pretraining-scale breadth** — trillions of tokens across the general web.
- You want a **documented, reproducible, public source** you can point to in a data card and a paper.
- You do not have a team to operate fetchers, frontiers, and politeness infrastructure forever.
- You can start from a **published derivative** (FineWeb, RefinedWeb, DCLM, Dolma, RedPajama) and inherit someone else's extraction and filtering — this is the default, and it is the right default.
- Monthly freshness is good enough for your use case.

### Crawl yourself when

- You need data **fresher than ~1 month**, or continuously.
- You need **domains, languages, or content types Common Crawl systematically under-samples** (deep web, niche communities, non-Latin scripts, specialist sources).
- You have a **direct relationship or license** with publishers and want to honor specific terms via your own fetch.
- You need **raw-response fidelity** for something Common Crawl strips or does not capture.

### Skip the whole "build it ourselves" instinct when

- You are tempted to re-extract Common Crawl from scratch when FineWeb already did it — unless you have a *specific* extraction or filtering need their public corpus does not meet, you are reinventing 15 trillion tokens of work.
- You are about to pull 10 consecutive monthly snapshots "to be safe" — that is overlap you will pay for and then delete in dedup.
- You are reaching for WET because it is smaller — that is a quality ceiling you cannot raise later.
- You have not yet tagged a single byte with provenance — fix that before you scale collection, not after.

Sourcing and collecting is the unglamorous front of the data pipeline, but it sets the ceiling on everything after it. You cannot filter quality into a corpus that was extracted lossily, you cannot dedupe your way out of snapshots you should not have collected, and you cannot license your way out of provenance you never recorded. Get the collection right — WARC over WET, breadth over consecutiveness, provenance from byte one — and the rest of the pipeline has something worth its compute. The next step is turning those raw HTML bytes into clean text, which is its own deep and surprisingly hard problem: [text extraction and boilerplate removal](/blog/machine-learning/training-data/text-extraction-and-boilerplate-removal).

## Further reading

- [Data-constrained scaling laws](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) — the demand side: how many tokens you actually need and what happens when the web runs out.
- [Text extraction and boilerplate removal](/blog/machine-learning/training-data/text-extraction-and-boilerplate-removal) — turning the raw WARC bytes you just collected into clean, structured text.
- [Deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) — the cross-snapshot near-duplicate problem, MinHash/LSH, and why it dominates collection economics.
- [Legal, ethics, and the future of training data](/blog/machine-learning/training-data/legal-ethics-and-the-future-of-training-data) — provenance, licensing, robots, and the hardening regulatory landscape around crawled data.
- Common Crawl documentation and the per-crawl WARC/WAT/WET layout at `commoncrawl.org`; the `warcio` and `cdx_toolkit` libraries; the FineWeb technical report and the `datatrove` pipeline; the RefinedWeb paper (Penedo et al., 2023).
