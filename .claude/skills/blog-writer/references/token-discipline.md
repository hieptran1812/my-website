# Token discipline for series production

Read this before dispatching a series wave. It is not a style guide — it is where the money goes.

## The measurement

Aggregated from 71 orchestrator transcripts of this repo (`.claude/scripts/token-report.mjs` reproduces it):

| Line item | Volume | Share of spend |
| --- | ---: | ---: |
| **Cache read** | **2,582 M tokens** | **largest** |
| Cache write | 149 M tokens | second |
| Output | 32 M tokens | third |
| Plain input | 1.6 M tokens | negligible |

14,540 API calls at a **median context of 148k tokens** — p75 246k, p90 382k, p99 632k. 1,265 calls ran above 400k. One session spanned 299 hours and 997 turns with a 755k peak.

Cache read is billed at 0.1× input and cache write at 1.25–2×, so at Opus 5 rates ($5 in / $25 out per MTok) that mix is roughly **$1.3k cache-read + $0.9–1.5k cache-write + $0.8k output**. The orchestrator — which writes almost no prose — outspends the drafting it supervises.

**The governing identity:**

```
cache-read tokens  =  Σ over turns of (context size at that turn)
```

Not "how much did I write." Not "how many posts." Every token you pull into a session is re-billed on **every remaining turn of that session**. A 20k-token post read at turn 40 of a 600-turn session costs 20k × 560 ≈ 11 M cache-read tokens. Reading it once *feels* free. It isn't.

Everything below follows from that one line.

## The four rules

### 1. One session per wave — never per series

A session that spans 12 days and 997 turns re-bills its earliest turns a thousand times. Finish a wave, commit and push it, then **start a fresh session for the next wave**. Carry forward only the plan file path and the wave number; the plan on disk is the state, not the transcript.

This single change moves the median context from ~148k toward ~60k. It is worth more than every other item on this page combined.

### 2. Nothing large enters the orchestrator — only verdicts

The orchestrator's job is dispatch, gate, commit. It does not need to *see* the artifacts.

| Artifact | Cost if read inline | Instead |
| --- | --- | --- |
| Rendered figure (`.webp`) | ~2k tok, re-billed every turn after | `figure-reviewer` subagent → one verdict line |
| Post prose (8–10k words) | ~15–20k tok, re-billed every turn after | `post-verifier` subagent → pass line or FAIL list |
| Scene JSON / validator output / render logs | 5–30k tok per post | `figure-author` subagent → one-screen manifest |
| Full series plan (up to 42 KB) | ~11k tok for the whole file | Read only the wave's own section |

Measured: 484 images read into orchestrator sessions accounted for **134 M cache-read tokens** — tokens spent re-reading pictures that had already been judged. `Read` on a WebP in a long session is the most expensive single call in this pipeline.

The rule is not "avoid subagents to save tokens." It is the opposite: **subagent context is disposable, orchestrator context is not.** Push every bulky read down into something that dies.

### 3. Tier the model to the stage, not the post

Quality lives almost entirely in Phase B (outline) and Phase D (prose). Those stay on Opus. The mechanical stages do not need it, and Sonnet 5 is ~40% of Opus's rate while Haiku 4.5 is ~20%.

| Stage | Model | Why |
| --- | --- | --- |
| Phase A/B — intake, research, outline, abstraction inventory | **Opus** | Sets the thesis, the figure plan, and the case studies. Every downstream quality ceiling is set here. |
| Phase C — DSL authoring, render, WebP (`figure-author`) | **Sonnet** | Schema-driven and validator-checked. The engine enforces layout; the model just fills the shape. |
| Phase C2 — visual gate (`figure-reviewer`) | **Sonnet** | Vision + a fixed 6-point rubric returning one line. |
| Phase D — the prose | **Opus** | This *is* the product. Never downgrade. |
| Phase E — verify gate (`post-verifier`) | **Haiku** | Runs a bash script and formats the output. |

This is where "fewer tokens, same quality" actually comes from: you are not writing less or thinking less, you are paying Opus rates only for the two stages that determine whether the post is good.

### 4. Read the wave, not the plan

Series plans run to 42 KB. Grep or `offset`/`limit` the wave you are running. Same for `MEMORY.md` entries and prior-wave reports.

## Checklist before dispatching a wave

- [ ] Fresh session (previous wave committed, pushed, and closed)
- [ ] Only this wave's plan section in context — not the whole plan file
- [ ] Figure work delegated to `figure-author` (Sonnet)
- [ ] Every figure gated by a `figure-reviewer` (Sonnet), one per figure, dispatched concurrently
- [ ] Verification delegated to `post-verifier` (Haiku)
- [ ] No `Read` of any `.webp` in this session
- [ ] No `Read` of a finished post's full prose in this session
- [ ] `.cache/blog-writer/<slug>/` cleared per Phase F once each post is green

## Re-measuring

```bash
node .claude/scripts/token-report.mjs
```

Prints per-session cache-read / output / turns / median-and-peak context, the context percentile distribution, and image-carry cost. Run it after a wave and compare the median context against the previous wave — that number is the whole scoreboard.
