---
name: wave
description: Run one wave of a blog series end-to-end — dispatch the wave's posts as parallel agents, gate them, commit and push. Reads only that wave's section of the plan and keeps figures, render logs, and prose out of this session.
argument-hint: [series-slug] [wave-number]
arguments: [series, wave]
disable-model-invocation: true
---

# Run wave $wave of the `$series` series

## 0. Session check — do this first

If **this session has already completed a wave**, stop and tell the user to `/clear` and re-run `/wave $series $wave`. Do not continue.

Everything the previous wave loaded gets re-billed on every turn of this one — a few hundred turns carrying ~250k of dead context is ~$50 of pure waste, and it compounds each wave until auto-compaction fires mid-run and silently drops detail. A fresh context is one keystroke; this is the highest-value rule in `CLAUDE.md`.

A session that has only been used for reading, planning, or chat is fine. The rule is about a *completed wave*, not about being brand new.

## 1. Load only what this wave needs

Plan file: `.claude/plans/$series-series.md`

- `grep -n "^## WAVE $wave" .claude/plans/$series-series.md` to find the line, then read **only** that section (up to the next `## `). Plans run to 42 KB; reading the whole file costs ~11k tokens on every subsequent turn for no benefit.
- Read the `## Conventions` section — it names the **engine** (`finance-writer`, `blog-writer`, or `paper-writer`), the depth, the verify script, and any series-specific hard gates (fact-checking, sourcing, honesty rules).
- Read the progress checklist at the bottom to confirm wave $wave is actually next and the previous wave is committed.

Then stop reading the plan.

## 2. Preflight

- `git status --porcelain content/blog/<category>/<subcategory>/` — the target folder must be clean. Uncommitted work from a previous wave means finish that first.
- For each planned slug, check `[ -f <target>/<slug>.md ]`. **A post that already exists is not re-run** — a same-slug figure render clobbers committed WebPs. If one exists, drop it from this wave and say so.
- `git pull --rebase --autostash origin main` — several sessions share `main`.

## 3. Dispatch

One agent per post, **all in a single message** so they run concurrently. Each agent gets: its slug, its one-line brief from the plan, the engine skill to use, the depth, and the series' hard gates.

Every agent must:

- Use the engine skill named in `## Conventions` — not a generic drafting approach.
- **Delegate Phase C to the `figure-author` subagent.** Scene JSON, validator retries, and render logs are the bulk of Phase C's tokens and are worthless once the WebPs exist.
- **Gate Phase C2 with `figure-reviewer`, one instance per figure**, dispatched concurrently. **Never `Read` a `.webp` directly** — an image costs ~2k tokens in and is re-billed every later turn of that agent.
- Run the gate via the **`post-verifier`** subagent, telling it which engine wrote the post.
- Honor the series' hard gates verbatim. For fact-checked series, every number (fund size, deal, loss, market cap, FDV, unlock date, enforcement action) is **sourced or dated-and-attributed**; contested claims are framed as *reported/alleged* with the source; nothing invented.

## 4. Collect

Agents can go idle mid-pipeline with **no `.md` written** — an idle agent is not a finished agent. Before treating a post as done:

- `[ -f <target>/<slug>.md ]` and check the word count
- Confirm its WebPs exist in `public/imgs/blogs/`

For anything incomplete, resume the agent with `SendMessage` rather than dispatching a new one — the original still holds its research and outline. If the account hits a usage limit mid-wave, wait and resume the same way; re-dispatching pays for all that work twice.

## 5. Ship

- Commit **only this wave's** `.md` and `.webp` files, by explicit path. **Never `git add -A`** — the repo routinely holds other sessions' in-flight work.
- Commit subject: `feat(blog): $series Wave $wave — <wave title> (N posts)`.
- `git pull --rebase --autostash origin main` then push.
- Tick wave $wave in the plan's checklist with the commit SHA, the date, the WebP count, and any deviation worth remembering (a thin post, a dropped slug, a salvage).
- Clear `.cache/<engine>/<slug>/` for each green post.

## 6. Report and hand off

Tell the user:

- Which posts shipped, word counts, figure counts, the commit SHA
- Anything dropped or thinned, and why
- The next wave number and its title

Then run `node .claude/scripts/token-report.mjs --since <today>` and report the **p50 context** against the 148k baseline and the 60k target — that one number says whether the wave stayed disciplined.

Finish by telling the user to `/clear` before the next wave.
