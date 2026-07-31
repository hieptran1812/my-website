# my-website

Long-form technical and finance blog (Next.js). Most work here is **blog-series production**: 40–60 post series drafted in waves of ~6 by parallel subagents, using the `blog-writer` / `finance-writer` / `paper-writer` skills. Series plans live in `.claude/plans/`.

## Token discipline — read before dispatching a series wave

Measured across 71 orchestrator transcripts and 990 subagent transcripts (`node .claude/scripts/token-report.mjs`): **cache read is 7.4 B tokens versus 64 M output tokens.** The articles themselves are a small minority of spend. Writing shorter saves almost nothing and costs quality — it is the wrong lever.

The governing identity is:

```
cache-read tokens  =  Σ over turns of (context size at that turn)
```

Every token pulled into a session is re-billed on **every remaining turn of that session**. A 20k-token post read at turn 40 of a 600-turn session costs ~11 M cache-read tokens. Reading it once feels free; it isn't. The scoreboard is **median context per turn**, not word count.

**Turns are the multiplier, and the cost is quadratic in them.** Context grows roughly linearly with turn count, so substituting into the identity gives:

```
Σ context  ≈  N² · g / 2        (N = turns, g = growth per turn)
```

Measured on crypto-players W5/W6: a drafting agent ran **279 turns** and climbed 30k → 339k, costing ~40 M cache-read tokens — one post, ~$20 in re-reading alone. Its tool results were only ~5 MB; the bulk of that context was **its own accumulated turns**.

The practical consequence: **halving an agent's turns quarters its cost.** Cutting bytes-per-turn is linear and small; cutting turns is quadratic and large. When something is expensive, ask *what is making this take so many turns* before asking what is making each turn big. Iterative fix-loops — figure fails the gate, re-author, re-render, re-gate — are the worst offenders, and they belong inside a short-lived subagent, never in the drafter or the orchestrator.

Four rules follow. Full rationale and the measurements: `.claude/skills/blog-writer/references/token-discipline.md`.

### 1. One session per wave — never one per series

Finish a wave, commit, push, **close the session, start a fresh one for the next wave.** State lives in the plan file on disk, not in the transcript; carry forward only the plan path and the wave number.

A session split into *k* parts cuts its Σ-context by roughly *k*×. The worst measured session ran 997 turns over 300 hours at a 275k median context — its first turns were re-billed nearly a thousand times. This rule is worth more than everything else on this page combined.

**Split when the drafts hit disk, not when the wave finishes.** Figure gating, fix cycles and commit need none of the drafting transcript — and that second half is where rework lands (limit resets, clobbering agents, failed figure gates). Crypto-players W5 ignored this: 427 turns, ~30h, one session, p50 252k — worse than baseline despite following rules 2–4.

### 2. Nothing bulky enters a long-lived context — only verdicts

The orchestrator dispatches, gates, and commits. It never needs to *see* the artifacts. Same discipline applies inside a drafting agent.

| Never read inline | Delegate to |
| --- | --- |
| The whole figure pipeline — author, render, **gate, fix, re-gate** | `figure-author` (returns one manifest of passed WebPs) |
| Rendered figure (`.webp`) — ~2k tok, re-billed every later turn | `figure-reviewer` (one per figure, returns one verdict line) |
| A finished post's prose — ~15–20k tok | `post-verifier` (returns the pass line or the FAIL list) |
| A whole 42 KB series plan | Grep or `offset`/`limit` to the wave's own section |

Subagent context is **disposable**; orchestrator and drafter context is not. Push every bulky read down into something that dies.

Two leaks the table misses:
- **Verdict count costs turns.** One `figure-reviewer` per figure is right inside a drafting agent, wrong for an orchestrator gating a whole wave — 60 figures becomes 60 dispatches + 60 notifications + a relay per failure. Gate a wave through **one aggregator subagent per post** that fans out the reviewers itself and returns a single PASS/FAIL list. Same reviewers, same rubric, same quality — one turn instead of ten.
- **Bulk arrives unbidden.** A subagent that spawns its own children pushes each child's full return into the orchestrator as a task notification. Tell any agent that will fan out: *summarise your children's findings; never pass their raw returns upward.*

### 3. Tier the model to the stage

Quality is decided in the outline and the prose. Everything else is mechanical and validator-checked.

| Stage | Model |
| --- | --- |
| Phase A/B — intake, research, outline, abstraction inventory | **Opus** |
| Phase C — figures (`figure-author`) | Sonnet |
| Phase C2 — visual gate (`figure-reviewer`) | Sonnet |
| Phase D — the prose | **Opus** |
| Phase E — verify gate (`post-verifier`) | Haiku |

This is what makes the saving free rather than a tradeoff: you pay Opus rates only for the two stages that decide whether the post is good.

### 4. Re-measure after each wave

```bash
node .claude/scripts/token-report.mjs
```

Watch one number: **p50 context**. Baseline was 148k (p90 380k). Target for series waves is **p50 under 60k, p90 under 120k**.

## Series conventions

- **Commit per wave, explicit paths.** Only that wave's `.md` + `.webp`. Never `git add -A`.
- **Always `git pull --rebase --autostash` before pushing** — several sessions share `main`.
- **Check the post doesn't already exist** before running the figure pipeline; a same-slug re-render clobbers committed WebPs. Restore with `git checkout` if it happens.
- **Agents can go idle mid-pipeline** with no `.md` written. Verify on disk before committing; resume via `SendMessage` rather than re-dispatching.
- Clear `.cache/<skill>/<slug>/` once each post passes its gates.
- Production domain is **halleywiki.com**; ISR 3600 plus a 15–30 min build lag.
