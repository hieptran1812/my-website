---
name: finance-writer
description: Draft a long-form, deep-dive-yet-accessible finance blog post for the my-website blog. Opens with a TL;DR box, builds every concept from first principles for a curious beginner, grounds it in worked dollar examples, and illustrates it with Excalidraw figures (cash-flow timelines, payoff charts, balance-sheet stacks). Writes valid frontmatter into content/blog/trading/<finance|crypto|quantitative-finance>/. Triggers on /finance-writer, "write a finance post about", "explain <finance topic> as a blog", "deep dive on <financial concept>".
---

# finance-writer

Drafts a long-form finance article that is **deep but never gatekept**: it assumes a smart, curious reader with *no finance background*, defines every term, builds intuition with everyday-money analogies before any formula, anchors each idea in a worked example with real dollar figures, and illustrates the abstractions with Excalidraw diagrams. Every post opens with a TL;DR box.

This is the finance sibling of [`blog-writer`](../blog-writer/SKILL.md). It **reuses blog-writer's diagram tooling verbatim** — the validator, the layout engines, the batch renderer are all domain-neutral. What changes is the *voice* (accessible expert, not principal-engineer war stories), the *structure* (TL;DR → foundations → deep dive → worked examples → misconceptions → real markets), the *diagram vocabulary* (finance-native figures + a finance semantic palette), and the *verify gate* (adds TL;DR-at-top, foundations-section, and worked-example checks).

## When to use

- User types `/finance-writer` or `/finance-writer <topic>`
- User says "write a finance blog post about X", "explain X (a finance topic) clearly", "deep dive on <financial instrument/concept>"
- Topic is about money, markets, instruments, valuation, accounting, macro, banking, crypto/DeFi, or quantitative finance

Do NOT use for: ML/engineering posts (use `blog-writer`), short notes, or social-length copy. If a topic is finance-flavored but really an engineering deep-dive (e.g. "building a low-latency matching engine"), prefer `blog-writer` and route to `trading/`.

## Core promise (the things every post must deliver)

These are the user's hard requirements, enforced in Phase E:

1. **TL;DR at the very top** — a callout box summarizing the whole post in ≤ 6 bullets, *before* the reader scrolls.
2. **Complete foundations** — a beginner can read top-to-bottom with zero prior finance knowledge. Every term defined on first use; the basics are not assumed.
3. **Genuinely deep** — past the basics it reaches the depth a practitioner would respect (mechanics, edge cases, second-order effects, how it really behaves in markets).
4. **Easy worked examples** — concrete dollar/number walkthroughs ("You buy 1 share at \$100…") so the abstraction is always grounded.
5. **Clear, supportive visuals** — diagrams that *prove* a claim and sit right next to the prose they illustrate, never decoration.
6. **Real, sourced numbers — never fabricated.** Every factual figure (a market level, rate, size, regulatory limit, dated historical magnitude) is traceable to a real source and is internally consistent across the post. If a value cannot be verified, it is ranged or dated-and-attributed, never invented. This is the **fact-check** promise, enforced in Phase D2; see `references/fact-check.md`. (Round hypothetical numbers inside worked examples — "suppose you buy 1 share at \$100" — are illustrative arithmetic, not factual claims, and are fine.)

## Reference files (lazy-load)

Read each at the start of the phase that needs it — don't read them all upfront.

| File                                                       | Read at start of                                              |
| ---------------------------------------------------------- | ------------------------------------------------------------- |
| `references/finance-diagrams.md`                           | Phase B (planning figures) and Phase C (authoring them)       |
| `../blog-writer/references/diagram-authoring.md`           | Phase C — the mechanical authoring rules (fonts, grid, arrows) |
| `references/finance-voice.md`                              | Phase D (before writing prose)                                |
| `references/fact-check.md`                                 | Phase D2 (the numbers fact-check protocol)                    |
| `templates/{deep-dive,explainer,concept}.md`              | Phase B (skeleton for the chosen depth)                       |

The diagram *mechanics* (canvas size, font families, palette hexes, arrow binding, snap grid, anti-dead-space rule) live in blog-writer's `diagram-authoring.md` and are authoritative. `references/finance-diagrams.md` is a thin overlay: it re-assigns the palette's *meaning* for finance and catalogs finance-native figure kinds. Read both in Phase C.

## Frontmatter contract

```yaml
---
title: "Sentence-case Title: Optional Clarifying Subtitle"
date: "YYYY-MM-DD" # today's date, absolute
publishDate: "YYYY-MM-DD" # optional; equals date if absent
description: "One sentence: what a beginner walks away understanding."
tags: ["tag1", "tag2", "..."] # 5–12 tags, lowercase-hyphen
category: "trading" # always 'trading' for this skill
subcategory: "Finance" # "Finance" | "Crypto" | "Quantitative Finance"
author: "Hiep Tran"
featured: true # true for deep-dives, false for explainer/concept
readTime: 27 # integer minutes, recomputed in Phase E
---
```

`category` is always `trading` (the directory). `subcategory` is the display label matching the chosen subfolder. Convert any relative dates ("yesterday", "next Monday") to absolute `YYYY-MM-DD`. Legacy optional fields used by older posts (`excerpt`, `image`, `collection`) are allowed but not required — `description` is the canonical summary field.

## Workflow

### Phase A — Topic intake

Ask via `AskUserQuestion` only what's missing:

1. **Topic** — required.
2. **Depth** — `deep-dive` | `explainer` | `concept`. Default = `deep-dive` (the user's standing preference is deep dives). Use `explainer` for a single self-contained mechanism, `concept` for a short "what is X" primer.
3. **Subcategory / routing** — `Finance` | `Crypto` | `Quantitative Finance` (see routing table). Infer from the topic; only ask if genuinely ambiguous.
4. **Cross-link targets** — search `content/blog/trading/**/*.md` for 2–4 related posts to link.

Audience is fixed: **a curious smart beginner with no finance background**. Do not ask about audience. Output is **English** regardless of how the user phrased the request.

### Phase B — Research & outline (STOP for approval)

1. **Read `references/finance-diagrams.md`.**
2. Fetch context and **capture sources as you go**: `WebSearch` for primary sources / current figures (rates, prices, market sizes), `WebFetch` for user-provided URLs. Finance facts go stale — prefer recent, citable numbers and note the as-of date. Every figure you plan to use should arrive with a source attached *now*; retrofitting citations in Phase D2 onto numbers you can no longer trace is how fabrication creeps in. Route each figure to its native source tier (legal text → securities press → mainstream/official press → international — see `references/fact-check.md`). If web tools are unavailable, draw figures only from a cited data kit (`.cache/finance-writer/_<series>/data*.py`) and attribute them; otherwise range-or-date them per the no-fabrication rule.
3. Pick the matching skeleton from `templates/`.
4. Produce the outline as markdown with:
   - Proposed title and slug
   - Target path (`content/blog/trading/<subfolder>/<slug>.md`)
   - **TL;DR draft** — the ≤ 6 bullets that will open the post (so the user approves the thesis up front)
   - **Foundations checklist** — the list of terms/prerequisites that will be defined from zero before the deep part begins
   - Section list (H2s with one-line summaries), ending with **Common misconceptions** and **How it shows up in real markets / case studies**
   - **Abstraction inventory + figure plan** — one bullet per abstract concept: claim / caption / section anchor / figure kind (see `finance-diagrams.md`) / sketch. Figure count = abstraction count.
   - **Worked-example plan** — 3+ named numeric walkthroughs (deep-dive), each one line: the scenario + the number being computed.
   - 2–4 cross-links to existing `trading/` posts
5. **Print the outline and stop.** Wait for "go" / "approved" / edits.

### Phase C — Diagrams (parallel, headless)

Identical pipeline to blog-writer — the scripts are shared. **Read `../blog-writer/references/diagram-authoring.md`** (mechanics) and keep `references/finance-diagrams.md` open (finance palette meaning + figure kinds).

Per figure (run all N in parallel for one post, render as one batch):

1. Author element JSON → `.cache/finance-writer/<slug>/<slug>-<i>.in.json`. (For canned shapes you may author a DSL and use the layout engine — step below.)
2. Validate + normalize: `node .claude/skills/blog-writer/scripts/author-scene.mjs <in.json> <scene.json>`. The validator enforces fonts, the 6-hex palette, containment, no-overlap, density, claim length, caption, snap grid, arrow binding, and anti-dead-space. Read its errors — they name the rule and element. Do not bypass.
   - DSL shortcut for canned kinds (pipeline/graph/before-after/matrix/tree/timeline/grid/layered-stack): `node .claude/skills/blog-writer/scripts/layout-scene.mjs <dsl.json> <scene.json>`.
3. Batch render all scenes **to PNG inside the cache** via the mcp_excalidraw renderer with a manifest:
   ```bash
   slug="<slug>"
   manifest=".cache/finance-writer/$slug/manifest.json"
   ls .cache/finance-writer/$slug/*.scene.json \
     | jq -R -s 'split("\n") | map(select(length>0)) | map({in: ., out: (. | sub(".scene.json"; ".png"))})' \
     > "$manifest"
   node /Users/hieptran1812/Documents/mcp_excalidraw/scripts/render-scene-batch.mjs "$manifest"
   ```
4. **Convert each cache PNG → lossless WebP** at `public/imgs/blogs/<slug>-<i>.webp`:
   ```bash
   mkdir -p public/imgs/blogs
   for png in .cache/finance-writer/$slug/*.png; do
     [ -e "$png" ] || continue
     cwebp -quiet -lossless -m 6 "$png" -o "public/imgs/blogs/$(basename "${png%.png}").webp"
   done
   ```
5. Verify each WebP ≥ 1600×900 px, ≥ 40 KB.

If the renderer or `cwebp` exits non-zero, or any WebP fails the sharpness floor: **stop and surface to the user.** Never substitute ASCII art, `text` boxes, Unicode box-drawing, prose-only "diagrams", or inline `mermaid` source — those are hard failures (the verify gate rejects them).

Do NOT use the `mcp__excalidraw__*` MCP tools here — they target the live canvas, not this headless code path.

### Phase C2 — Visual self-review (vision gate, mandatory)

The validator checks geometry; it cannot *see* the pixels. Before writing prose, **open every `public/imgs/blogs/<slug>-*.webp` with `Read`** and write a one-line verdict per figure (`PASS`, or `FAIL: <criterion> — <what's wrong>`). Use the full rubric in `../blog-writer/references/diagram-authoring.md §Visual self-review`. Finance-specific additions to that rubric:

- **Sign convention is unambiguous** — inflows/gains are green and outflows/losses are red *consistently* across every figure; a reader never has to guess whether an arrow means "money in" or "money out".
- **Axes are labeled with units** — any chart (payoff, yield curve, growth, distribution) labels both axes with what they measure and the unit (\$, %, years), and marks the reference points the prose names (strike, breakeven, par, spot).
- **Numbers match the worked example** — if a figure illustrates a worked example, its numbers are the *same* numbers the prose computes. No invented figures for visual balance.

Any `FAIL` → re-author that figure (fix `.in.json`, re-validate, re-render, re-convert), then re-review. Never "fix" a bad figure by editing the prose. Advance only when every figure is a clean PASS. For ≥ 8 figures you may dispatch parallel reviewer subagents (one per figure).

### Phase D — Draft

1. **Read `references/finance-voice.md`.**
2. Write the full markdown via `Write` to the resolved target path. Frontmatter exactly per contract; today's date.
3. **First thing in the body: the TL;DR callout box.**
   ```markdown
   > [!important]
   > **TL;DR** — <one-sentence thesis>.
   >
   > - <key takeaway 1>
   > - <key takeaway 2>
   > - <key takeaway 3>
   > - <the one number / fact a reader should remember>
   ```
   (`tldr`/`summary` are not supported callout types on this site — `important` renders the prominent box; the bold `**TL;DR**` lead names it.)
4. Then the hook, then the **mental-model figure** referenced in the intro, then the **Foundations** section (define everything from zero), then the deep sections, then **Common misconceptions**, then **How it shows up in real markets** (case studies / worked scenarios), then **When this matters / further reading**. No generic "Conclusion".
5. Embed each WebP immediately under the heading it illustrates: `![alt](/imgs/blogs/<slug>-<n>.webp)`. Every image is `.webp` — no `.png`/`.jpg`/`.svg`. The first figure is referenced in the intro.
6. Each worked example is a clearly marked walkthrough with explicit numbers (a `#### Worked example:` sub-heading or a numbered step list), and ends with the single sentence of intuition it teaches.
7. Cross-link inline with relative paths: `[the volatility surface](/blog/trading/quantitative-finance/volatility-surface)` (drop `content/` and `.md`).

### Phase D2 — Fact-check the numbers (mandatory, before the verify gates)

**Read `references/fact-check.md`** and run the protocol on the drafted post. The promise is: *every factual figure is real, sourced, internally consistent, and never fabricated.* Four steps:

1. **Extract every quantitative claim.** Run the extractor — it builds your worklist:
   ```bash
   bash .claude/skills/finance-writer/scripts/extract-claims.sh <post.md>
   ```
   It prints a CLAIM LEDGER (every numeric line), a NUMBER INDEX (each figure → the lines it appears on), POSSIBLE INTERNAL CONTRADICTIONS (an anchor term near two different numbers), and UNSOURCED LIVE-NUMBER LINES. Copy the ledger into a working table in the scratchpad (`claim | value | type | tier | source | as-of | status`) — never into the post.
2. **Grep for internal contradictions.** Make the post agree with itself before reaching outward: resolve every flagged anchor, confirm a figure repeated across the TL;DR / sections / captions is identical, check for unit drift (bps vs %, million vs billion, VND vs USD), and recompute every derived number so worked examples actually add up.
3. **Cross-check external sources via the tier waterfall.** Route each claim to its native authoritative tier and escalate on conflict: **legal / primary text** (laws, decrees, circulars, regulator & exchange filings, company reports) → **securities / financial press & market data** → **mainstream / official press & statistics agencies** → **international cross-check** (IMF/World Bank/BIS/FT/Reuters). Double-source anything load-bearing (TL;DR figures, section theses); higher tier breaks ties. The full tier→claim-type mapping and the Vietnam + global source lists are in `references/fact-check.md`.
4. **Resolve and annotate.** Each claim exits as verified / corrected / **ranged** / **dated-and-attributed** / cut. **Never invent a number** — if you cannot source it, you write a sourced range or a timestamped value, or you delete the claim. Live numbers carry an inline as-of date; deep-dives end with a `## Sources & further reading` section listing the primary sources behind the headline figures. Re-run the extractor after edits to confirm no new contradiction slipped in.

Do not advance to Phase E with any claim left in an unverified state, any relative date ("last month", "recently") left unanchored, or any figure standing as false precision.

### Phase E — Verify (hard gates)

```bash
bash .claude/skills/finance-writer/scripts/verify-finance-post.sh <post.md> <slug> <depth>
```

`<depth>` ∈ `deep-dive` | `explainer` | `concept`. The script checks everything blog-writer checks (word-count floor, diagram-count floor, abstraction coverage, WebP sharpness, webp-only embeds + no stray non-webp artifacts, forbidden text-diagram substitutes, slug-match, no-H1, English-only, frontmatter sanity) **plus four finance gates**:

- **TL;DR-at-top** — a `[!important]`/`[!note]`/`[!info]` callout or a `**TL;DR**` blockquote within the first ~25 body lines.
- **Foundations section** — a heading matching the basics (`foundation|fundamentals|basics|first principles|primer|how .* works|background|the building blocks`) before the deep sections.
- **Worked examples** — ≥ N concrete numeric walkthroughs (deep-dive ≥ 4, explainer ≥ 3, concept ≥ 2), detected by `worked example` / `example:` markers co-located with `$`/number figures.
- **Sourcing** — provenance for real figures: a deep-dive must carry a `## Sources`/`References`/`Further reading` section **or** inline citation links, and the gate WARNs with a count of live-number lines (price/index/rate/size) that lack a nearby citation, link, or as-of date. A hard FAIL only when a deep-dive has *zero* sourcing of any kind. This gate backstops Phase D2 — it does not replace the manual fact-check; an all-green gate on an unverified post still means the work isn't done.

Any FAIL → re-enter the named phase and fix. A WARN on the sourcing gate means re-run the Phase D2 extractor and source or date those lines before shipping. The fix for a missing figure is *always* to add the figure, never to delete the prose. The fix for a missing worked example is to add a real numeric walkthrough, never to relabel prose.

**Final figure pass (prose-aware).** Now that the prose exists, re-check the prose-dependent things — re-open any figure you're unsure about with `Read`:

- **Placement & faithfulness**: each figure sits under the heading it illustrates; every node label appears in the prose ±200 lines around its anchor.
- **Arrow / sign direction vs. prose**: each arrowhead points the way money/causality flows *in the text you wrote*; inflow/outflow colors match the worked numbers. Reversed arrows or flipped signs = re-author the figure (return to Phase C → re-run C2), not edit the prose.

Then report to the user:

- Final file path, word count, recomputed `readTime`
- New images written (paths and sizes)
- Which gates passed/failed and what was fixed
- 2–4 suggested cross-links the *user* should consider adding to *other* existing `trading/` posts (don't edit those unless asked)
- Reminder to run `npm run dev` (or `bun run dev`) and load the page locally before committing

### Phase F — Clean up the diagram cache (only after Phase E passes)

Once `verify-finance-post.sh` exits 0 **and** the visual review is clean, delete this post's diagram cache. It holds only intermediate artifacts (`*.in.json`, `*.dsl.json`, `*.scene.json`, `*.png`, `manifest*.json`); the final WebPs live in `public/imgs/blogs/` and the prose in `content/blog/`.

```bash
# Run ONLY if Phase E gates passed. Scoped to this slug — never wipe .cache/finance-writer wholesale.
rm -rf .cache/finance-writer/<slug>
```

Rules: gate it on a fully-green post (the cache is needed to re-author a failing figure); scope strictly to `<slug>`; mention the cleanup in the final report. Clearing each post's cache the moment it ships keeps stale cramped layouts from leaking into the next article's figures.

## Path / routing (always under `content/blog/trading/`)

| Topic keywords                                                                                     | Subfolder                | `subcategory`            |
| -------------------------------------------------------------------------------------------------- | ------------------------ | ------------------------ |
| personal finance, budgeting, saving, credit, mortgage, taxes, retirement                           | `finance/`               | `Finance`                |
| corporate finance, accounting, balance sheet, cash flow, capital structure, M&A, IPO, valuation     | `finance/`               | `Finance`                |
| how markets work, banking, central banks, monetary policy, inflation, macro, the four firm types    | `finance/`               | `Finance`                |
| stocks, ETFs, mutual funds, bonds (intro), portfolio basics, asset allocation                       | `finance/`               | `Finance`                |
| crypto, bitcoin, ethereum, blockchain, stablecoins, DeFi, on-chain, tokens, x402, wallets           | `crypto/`                | `Crypto`                 |
| options, futures, swaps, derivatives pricing, Black-Scholes, Greeks, implied vol, exotics            | `quantitative-finance/`  | `Quantitative Finance`   |
| yield curve, fixed-income analytics, bond pricing math, short-rate models, duration/convexity        | `quantitative-finance/`  | `Quantitative Finance`   |
| stochastic calculus, risk-neutral measure, Monte Carlo pricing, vol surface, quant risk              | `quantitative-finance/`  | `Quantitative Finance`   |

If two subfolders match (e.g. "pricing a crypto option"), or none match cleanly, use `AskUserQuestion` with the top 2–3 candidates. Never guess silently.

## Slug rules

- kebab-case, derived from the title.
- Drop stop-words (`a`, `the`, `of`, `for`, `with`, `to`, `and`, `or`).
- ≤ 60 chars. Trim trailing partial words.
- If `<target>/<slug>.md` already exists: ask the user to (a) overwrite, (b) append `-v2`, or (c) pick a new slug.

## What this skill shares with blog-writer (do not duplicate)

- `author-scene.mjs`, `layout-scene.mjs` — the validator and DSL layout engines.
- The mcp_excalidraw `render-scene-batch.mjs` headless renderer.
- `diagram-authoring.md` — the mechanical diagram rules (canvas 2400×1600, fonts 1/3, the 6-hex palette, arrow binding, snap grid, anti-dead-space).

What is finance-writer's own: this `SKILL.md`, `references/finance-voice.md`, `references/finance-diagrams.md`, `references/fact-check.md`, the three `templates/`, `scripts/extract-claims.sh`, and `scripts/verify-finance-post.sh`.

## Parallel execution

All phases run concurrently across sessions and within a single post. Phase C diagrams render as N independent puppeteer subprocesses with isolated `--user-data-dir` — no shared state, no canvas collision. Use `.cache/finance-writer/<slug>/` so finance caches never collide with blog-writer's `.cache/blog-writer/`.
