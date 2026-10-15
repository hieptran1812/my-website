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

Since every agent in the pipeline now runs Opus 5 (rule 3), that rate card applies end to end: there is no cheaper tier absorbing the mechanical stages, so the shape of this table *is* the bill.

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

**Split on the drafts-on-disk boundary, not on "the wave is finished."** A wave is two jobs, and only the first is drafting. Once every post's `.md` exists, the remaining work — figure gating, fix cycles, verify, commit — needs none of the drafting transcript. Close and restart, carrying only the slug list and the outstanding failure list.

Waiting for the wave to *finish* means never splitting, because the second job is exactly the one that blows up: crypto-players W5 ran **427 turns over ~30h in one session** and landed at p50 252k / p90 383k — ~1.7× worse than the 148k baseline *with rules 2–4 being followed*. Rework (a usage-limit reset mid-wave, agents clobbering each other, 70–90% of figures failing the visual gate) all lands in the second job. Assume it will, and give it a fresh context.

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

#### Two leaks this table doesn't cover

**Verdict *count* is a cost, even though each verdict is tiny.** "One `figure-reviewer` per figure" is right when the *drafting agent* gates its own post — that context dies. It is wrong when the **orchestrator** gates a whole wave: 60+ figures means 60 dispatches, 60 completion notifications and a relay message per failure, all in the context that survives. Same tokens per verdict, ~10× the turns, and turns are the multiplier in `Σ context`.

> When gating more than one post, dispatch **one aggregator subagent per post**. It fans out the per-figure `figure-reviewer` calls itself and returns a single list: `fig N: PASS` / `fig N: FAIL — <what to change>`. Review quality is identical — each figure still gets its own fresh-context reviewer and the same rubric — only the collection moves down a level.

**Bulk can arrive unbidden.** Rule 2 forbids *reading* large things; it does not stop the harness *pushing* them. A subagent that spawns its own research subagents delivers each child's full return value into the orchestrator as a task notification. In W5 that was 8 dense fact-ledger tables nobody asked the orchestrator to see, then re-billed for the rest of the session. When you dispatch an agent that will itself fan out, tell it explicitly: *summarise your children's findings; do not pass their raw returns upward.*

### 3. One model — Opus 5 everywhere. Tier the *effort*, not the model

**Every stage of this pipeline runs Opus 5.** No Sonnet, no Haiku, in any agent. The three subagent definitions pin it explicitly (`model: opus` in `.claude/agents/*.md`) rather than inheriting, so a 1M-context orchestrator session does not drag short-lived children into the long-context premium tier.

The reason is that the cheap-tier saving was being paid back in *turns*, which is the term that costs quadratically. A figure author that mis-reads the DSL schema fails the visual gate; every failure is another author → render → gate cycle. W5 measured **70–90% of figures failing the visual gate**, and each of those is a round trip that a per-token rate card does not show. A gate model that mis-judges is worse in both directions: a wrong PASS ships a broken figure, a wrong FAIL buys a re-author nobody needed. Single-model removes model quality as a variable in the loop that dominates the bill.

What replaces model tiering is **reasoning effort**, set per agent in the same frontmatter. Effort is a per-turn multiplier on thinking tokens only — it does not touch the `Σ context` term at all, so it is safe to spend where judgment is real and safe to drop where the work is a checklist.

| Stage | Model | Effort | Why |
| --- | --- | --- | --- |
| Phase A/B — intake, research, outline, abstraction inventory | Opus 5 | session default | Sets the thesis, the figure plan, and the case studies. Every downstream quality ceiling is set here. |
| Phase C — DSL authoring, render, WebP (`figure-author`) | Opus 5 | `medium` | Schema-driven and validator-checked, but it also owns the fix loop — enough effort to converge in one pass is cheaper than a second pass at `low`. |
| Phase C2 — visual gate (`figure-reviewer`) | Opus 5 | `low` | One image against a fixed 6-point rubric, one line out. Judgment is in the rubric, not the model's deliberation. |
| Phase D — the prose | Opus 5 | session default | This *is* the product. Never downgrade. |
| Phase E — verify gate (`post-verifier`) | Opus 5 | `low` | Runs a bash script and reports the FAIL lines. |

The two one-line agents also carry a `maxTurns` cap (`figure-reviewer: 6`, `post-verifier: 25`). Neither has any legitimate reason to loop; the cap is a backstop against the one failure mode — a stuck agent burning turns — that costs more than everything else on this page.

With a single model, rules 1, 2 and 4 are no longer *most* of the saving, they are **all** of it. Hold them tighter than before.

### 4. Read the wave, not the plan

Series plans run to 42 KB. Grep or `offset`/`limit` the wave you are running. Same for `MEMORY.md` entries and prior-wave reports.

## Checklist before dispatching a wave

- [ ] Fresh session (previous wave committed, pushed, and closed)
- [ ] Only this wave's plan section in context — not the whole plan file
- [ ] Figure work delegated to `figure-author` (Opus 5, `medium` effort)
- [ ] Every figure gated by a `figure-reviewer` (Opus 5, `low` effort), one per figure, dispatched concurrently
- [ ] Verification delegated to `post-verifier` (Opus 5, `low` effort)
- [ ] No `model:` override passed at dispatch — the agent definitions already pin Opus 5
- [ ] No `Read` of any `.webp` in this session
- [ ] No `Read` of a finished post's full prose in this session
- [ ] `.cache/blog-writer/<slug>/` cleared per Phase F once each post is green

## Re-measuring

```bash
node .claude/scripts/token-report.mjs
```

Prints per-session cache-read / output / turns / median-and-peak context, the context percentile distribution, and image-carry cost. Run it after a wave and compare the median context against the previous wave — that number is the whole scoreboard.
