---
name: blog-writer
description: Draft a long-form, principal-engineer-voice blog post for the my-website blog. Plans an outline, generates Excalidraw diagrams, then writes the full markdown into the correct content/blog/<category>/<subcategory>/ folder with valid frontmatter. Triggers on /blog-writer, "write a blog post about", "draft an article on", "new blog:".
---

# blog-writer

Drafts a long-form blog article in the house style of `my-website` — opinionated, principal-engineer voice, intuition before math, runnable code, comparison tables, case studies, and Excalidraw diagrams.

## When to use

- User types `/blog-writer` or `/blog-writer <topic>`
- User says "write a blog post about X", "draft an article on X", "new blog: X"
- User asks for a deep-dive / explainer / paper-reading post

Do NOT use for: short notes, README updates, comments on existing posts, social-media-length copy.

## Reference files (lazy-load)

This skill is intentionally split. Read each reference at the start of the phase that needs it — don't read them all upfront.

| File                                               | Read at start of                                                                   |
| -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `references/diagram-triggers.md`                   | Phase B (planning the abstraction inventory)                                       |
| `references/diagram-authoring.md`                  | Phase C (before authoring the first figure); §Visual self-review again at Phase C2 |
| `references/animated-figures.md`                   | Phase C — **only if** Phase B marked any abstraction `animated`                    |
| `references/voice-cheatsheet.md`                   | Phase D (before writing prose)                                                     |
| `templates/{deep-dive,explainer,paper-reading}.md` | Phase B (skeleton for the chosen depth)                                            |

Phase E gates run via `scripts/verify-post.sh` — no need to inline bash.

## Frontmatter contract

```yaml
---
title: "Sentence-case Title with Optional Subtitle After a Colon"
date: "YYYY-MM-DD" # today's date, absolute form
publishDate: "YYYY-MM-DD" # optional; equals date if absent
description: "One-sentence promise of what the reader walks away with."
tags: ["tag1", "tag2", "..."] # 5–12 tags, lowercase-hyphen
category: "machine-learning" # top-level folder name
subcategory: "Large Language Model" # human-readable; optional
author: "Hiep Tran"
featured: true # true for deep-dives, false for explainers/notes
readTime: 27 # integer minutes, recomputed in Phase E
---
```

`category` is the directory name under `content/blog/` (kebab-case). `subcategory` is a display label, not a path. Convert any relative dates the user mentions ("yesterday", "next Monday") to absolute `YYYY-MM-DD` before writing.

## Workflow

### Phase A — Topic intake

Ask via `AskUserQuestion` only what's missing:

1. **Topic** — required.
2. **Depth** — `deep-dive` | `explainer` | `paper-reading`. Default = `deep-dive` for engineering-heavy topics, `explainer` for conceptual, `paper-reading` if the user references a paper title or arxiv link.
3. **Audience** — default "senior ML engineers / staff-level interviewees".
4. **Cross-link targets** — search `content/blog/**/*.md` for 2–4 related posts.

### Phase B — Research & outline (STOP for approval)

1. **Read `references/diagram-triggers.md`.**
2. Optionally fetch context: `WebSearch` for primary sources, `WebFetch` for user-provided URLs, `claude-obsidian:wiki-query` if a vault exists.
3. Pick the matching skeleton from `templates/`.
4. Produce the outline as markdown with:
   - Proposed title and slug
   - Target path (`content/blog/<category>/[<subcategory>/]<slug>.md`)
   - Section list (H2s with one-line summaries)
   - **Abstraction inventory + diagram plan** — one bullet per abstract concept, each with claim / caption / section anchor / sketch (see `diagram-triggers.md`). Figure count = abstraction count. **Tag each figure `static` or `animated`.** Mark `animated` only when *motion carries the meaning* — a process unfolding step by step, a before→after state transition (cache fill/evict, memory compaction, rebalancing), something flowing along a path, a quantity sweeping/growing, or two strategies diverging "in the same round." Everything else stays `static`. Cap animated figures at **1–3 per post**; static Excalidraw WebP remains the default workhorse. (Motion-worthy criteria: see `animated-figures.md §When to animate`.)
   - Case-study list (deep-dives): 3–12 named incidents, each one line
   - Code snippet plan: language, what it demonstrates, ~lines
   - 2–4 cross-links to existing posts
5. **Print the outline and stop.** Wait for "go" / "approved" / edits.

### Phase C — Diagrams (parallel, headless)

1. **Read `references/diagram-authoring.md`.** Diagrams must be **diverse** (vary the figure kind per the plan — see §Diversity), **accurate** (every node/edge/number traces to the prose), and have **no meaningless empty space** (content fills the cropped frame).
2. For each planned figure: author element JSON → `.cache/blog-writer/<slug>/<slug>-<i>.in.json`.
3. Validate + normalize each: `node scripts/author-scene.mjs <in.json> <scene.json>`. The validator enforces fonts, palette, containment, no-overlap, density, claim length, caption presence, snap grid, **and anti-dead-space (no blank quadrant/band)**. Read its error messages — they name the rule and offending element. Do not bypass.
4. Batch render all scenes **to PNG inside the cache** via `mcp_excalidraw/scripts/render-scene-batch.mjs` with a manifest (one browser, ~150 ms per figure after startup).
5. **Convert each cache PNG → lossless WebP** at `public/imgs/blogs/<slug>-<i>.webp` (`cwebp -lossless -m 6`). The post ships `.webp` only; the PNG is a throwaway intermediate. (Exact loop in `diagram-authoring.md §Batch render`.)
6. Verify each WebP ≥ 1600×900 px, ≥ 40 KB.

If the renderer or `cwebp` exits non-zero, or any WebP fails the sharpness floor: **stop and surface to the user**. Never substitute ASCII art, `text` boxes, Unicode box-drawing, prose-only "diagrams", or inline `mermaid` source. Those are hard failures.

Do NOT use the `mcp__excalidraw__*` MCP tools in this phase — they target the live canvas, which is not on this code path.

**Phase C-anim — animated figures (only for abstractions tagged `animated` in Phase B).** These do **not** go through the Excalidraw → PNG → WebP pipeline. They are self-contained inline `<svg>` blocks driven by CSS `@keyframes`, embedded as raw HTML in the markdown (the blog renders with `allowDangerousHtml` and injects via `dangerouslySetInnerHTML`, so inline SVG survives; `<script>` does **not** run, so motion must be declarative).

1. **Read `references/animated-figures.md`** for the technique, the hard rules (no blank lines in the block, `prefers-reduced-motion` guard, accessibility, responsive sizing), the theming tokens, and the copy-adaptable motion patterns (step sweep, slide-and-compact, flow-along-path, sweep/grow, A↔B crossfade).
2. Author each as a full `<figure class="blog-anim">…</figure>` block into `.cache/blog-writer/<slug>/<slug>-anim-<i>.fig.html` (multi-line OK, **zero blank lines**, `<figure` at column 0).
3. Validate: `node scripts/check-anim.mjs <fig.html>`. It enforces the hard rules and prints `PASS`/`FAIL <rule>`. Read the messages; do not bypass.
4. These animated figures **count** toward the Phase E figure floor and **satisfy** abstraction-coverage (the gate recognizes `blog-anim` blocks) — they are real figures, not extras.

### Phase C2 — Visual self-review (vision gate, mandatory)

`author-scene.mjs` and `verify-post.sh` check geometry and structure, but they cannot _see_ the rendered pixels — a figure can pass every mechanical rule and still be a tangle of arrows, lopsided, half-empty, or off-topic. **This gate looks at the actual image.** Run it _before_ Phase D so a bad figure is re-authored before any prose is built around it.

1. **Read `references/diagram-authoring.md §Visual self-review`** for how to judge each criterion.
2. **Open every `public/imgs/blogs/<slug>-*.webp` with `Read`** (it renders WebP) and write a one-line verdict per figure — `PASS`, or `FAIL: <criterion> — <what's wrong>`:

   ```
   fig 1 (pipeline): PASS
   fig 2 (graph):    FAIL: arrows — 3 edges cross in the middle layer; head of e4 floats off its node
   fig 3 (matrix):   FAIL: empty space — bottom third is blank; rows not extended to a shared height
   ```

3. Per-figure rubric — **fail the figure if any answer is "no"**:
   1. **Faithful to the content** — every box, arrow, color, and number maps to the figure's `_claim`/`_caption` and the section it illustrates (from the Phase B outline); nothing invented for visual filler; the figure actually _proves_ its claim. ("đã thể hiện nội dung bài viết")
   2. **Arrows legible, not a tangle** — count the crossings: arrows don't cross each other or pierce unrelated nodes (> 2 visible crossings = re-author); every head/tail lands cleanly on a node edge (not floating, not buried inside a box); directions match the causal flow; orthogonal where the relationship is axial. ("các mũi tên có rối không")
   3. **Balanced composition** — visual weight is distributed, not dumped in one corner; the figure reads as centered; aspect ratio matches content shape (pipeline wide-short, stack tall-narrow). ("hình cân đối")
   4. **No meaningless empty space** — content fills the cropped frame; no wide empty band, no blank quadrant, no single card stretched to fake fullness. ("có nhiều khoảng trống không")
   5. **Text renders correctly** — all labels in Virgil/Cascadia (no system-font fallback), nothing overflows its box, nothing overlaps, no label sits on an arrow stroke; readable at a glance.
   6. **Squint test (< 5 s)** — at 25% the main path / bottleneck / outcome is still clear from color and position; one reading direction; ≤ 3 accent colors; no legend needed.

4. Set-level (review all figures together, once): 7. **Diversity** — no single figure kind is > ~½ the set; no two adjacent figures share a layout skeleton. If they do, recast one (see §Diversity).

**Decision rule:** any `FAIL` → re-author that figure (back to Phase C step 2: fix the `.in.json`, re-validate, re-render, re-convert), then re-review. Never "fix" a bad figure by editing the prose. Advance to Phase D **only when every figure is a clean PASS.**

For a large post (≥ 8 figures) you may dispatch parallel reviewer subagents — one per figure, each opening its WebP with `Read` and returning the verdict line — then act on the FAILs. Same gate, only the fan-out differs.

**Animated figures** can't be judged by a still `Read` — it shows one frozen frame, not the motion. Review the **source** instead (see `animated-figures.md §Self-review`): does the `0%` state *and* the `100%` state each map to something true in the prose, and is the *change between them* exactly what the caption claims? Does it loop cleanly, and does the `prefers-reduced-motion` branch freeze on a meaningful frame? Their static start/end frames still pass the rubric above (faithful, balanced, no dead space, text renders, squint). For real confidence, run `npm run dev` and watch each loop once.

### Phase D — Draft

1. **Read `references/voice-cheatsheet.md`.**
2. Write the full markdown via `Write` to the resolved target path. Frontmatter exactly per contract; today's date.
3. Embed each WebP immediately under the section heading it illustrates: `![alt](/imgs/blogs/<slug>-<n>.webp)`. Every embedded image must be `.webp` — no `.png`/`.jpg`/`.svg`. The first figure is referenced in the intro paragraph.
   - **Animated figures**: paste the validated `<figure class="blog-anim">…</figure>` block from `.cache/blog-writer/<slug>/<slug>-anim-<i>.fig.html` **verbatim** under its heading — not as `![]()`. Keep it one contiguous block with **no blank lines inside** and `<figure` at column 0 (a blank line makes CommonMark shatter the SVG into escaped text). A blank line *before* the opening `<figure` and *after* the closing `</figure>` is required, as for any block.
4. Add cross-links inline using relative paths: `[KV cache](/blog/machine-learning/large-language-model/kv-cache)` (drop the `content/` prefix and `.md` extension).

### Phase E — Verify (hard gates)

Run:

```bash
bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> <depth>
```

`<depth>` is one of `deep-dive`, `explainer`, `paper-reading`. The script checks: word-count floor, diagram-count floor (static WebP embeds **+** inline animated figures), abstraction coverage (a WebP **or** a `blog-anim` figure within 30 lines of every prose abstraction), WebP sharpness, webp-only embeds (no `.png`/`.jpg`/`.gif`) + no leftover non-webp render artifacts, forbidden text-diagram substitutes (animated-figure blocks are excluded from the ASCII/Unicode scan), **animated-figure safety** (each `blog-anim` block is contiguous/no-blank-line, declarative with no `<script>`/`on*=`, accessible, and reduced-motion-aware), slug-match on every image, no-H1-in-body, English-only, frontmatter sanity.

Any FAIL means re-enter the named phase and fix. The fix for missing figures is _always_ to add the figure, never to delete the prose.

**Final figure pass (prose-aware).** Phase C2 already cleared every figure against the visual rubric. Now that the prose exists, re-check the things that depend on the _written text_ — re-open any figure you're unsure about with `Read`:

- **Placement & faithfulness**: each figure sits under the heading it illustrates, and every node label appears in the prose ±200 lines around its anchor (no orphaned or mis-placed figure introduced during drafting).
- **Arrow direction vs. prose**: each arrowhead points the way causality flows _in the text you actually wrote_; reversed or dangling arrows = re-author the figure (return to Phase C → re-run Phase C2), not edit the prose.
- If you changed or re-authored any figure here, **re-run the Phase C2 rubric** on it before shipping.

The deep visual criteria (tangled arrows, balance, empty space, text rendering, squint test, diversity) are owned by **Phase C2** — don't re-litigate them here unless a figure changed.

Then report to the user:

- Final file path, word count, recomputed `readTime`
- New images written (paths and sizes)
- Which gates passed/failed and what was added on the second pass
- 2–4 suggested cross-links the _user_ should consider adding to _other_ existing posts (don't edit those unless asked)
- Reminder to run `npm run dev` (or `bun run dev`) and load the page locally before committing

### Phase F — Clean up the diagram cache (only after Phase E passes)

Once `verify-post.sh` exits 0 **and** the visual review is clean, delete this post's entire diagram cache. The cache (`.cache/blog-writer/<slug>/`) holds only intermediate authoring artifacts — `*.in.json`, `*.dsl.json`, `*.scene.json`, `*.png` (the pre-WebP render intermediates), `*.fig.html` (animated-figure sources, already pasted into the post), `manifest*.json`. The final WebPs live in `public/imgs/blogs/<slug>-*.webp`, the animated figures are inline in the markdown, and the prose in `content/blog/...`, so nothing the published post depends on is in the cache.

```bash
# Run ONLY if Phase E gates passed. Scoped to this slug — never `rm -rf .cache/blog-writer` wholesale.
# Leave the path unquoted (slugs are kebab-case, no spaces) so it matches the pre-approved
# Bash(rm -rf .cache/blog-writer/*) allowlist rule and runs without a permission prompt.
rm -rf .cache/blog-writer/<slug>
```

Why this matters: stale per-article caches accumulate across posts, and reusing or glancing at an earlier article's `.dsl.json`/`.scene.json` reintroduces the same cramped layouts and ugly diagram designs into new posts. Clearing each post's cache the moment it ships keeps `.cache/blog-writer/` holding only in-progress work, so every article's figures are authored fresh.

Rules:

- **Gate it.** If any Phase E gate FAILED, do NOT delete — the cache is needed to re-author the failing figure. Only clean up on a fully green post.
- **Scope to the slug.** Delete `.cache/blog-writer/<slug>/` only. Never touch sibling slug folders, the shared `.cache/blog-writer/excalidraw.log`, or `.cache/blog-writer/` itself.
- Mention the cleanup in the final report (e.g. "cleared `.cache/blog-writer/<slug>/`").

## Path / category routing

| Topic keywords                                                | Target folder                                    |
| ------------------------------------------------------------- | ------------------------------------------------ |
| llm, transformer, attention, kv-cache, tokenizer, decoding    | `machine-learning/large-language-model/`         |
| docker, kubernetes, vllm, triton, serving, gpu, inference-ops | `machine-learning/mlops/`                        |
| agent, tool-use, langgraph, react-agent, planner              | `machine-learning/ai-agent/`                     |
| sae, steering, interpretability, probe, circuit               | `machine-learning/ai-interpretability/`          |
| diffusion, vit, segmentation, detection, image                | `machine-learning/computer-vision/`              |
| backprop, optimizer, lr-schedule, init                        | `machine-learning/deep-learning/`                |
| sft, dpo, rlhf, lora, peft, finetuning                        | `machine-learning/training-techniques/`          |
| xgboost, lightgbm, catboost, random-forest, tabular           | `machine-learning/traditional-machine-learning/` |
| dsp, fft, audio, wavelet                                      | `machine-learning/signal-processing/`            |
| linear-algebra, probability, optimization-theory              | `machine-learning/mathematics/`                  |
| paper title / arxiv link                                      | `paper-reading/<area>/`                          |
| react, next.js, build-tool, framework                         | `software-development/`                          |
| backtest, market-microstructure, alpha                        | `trading/`                                       |
| short scratch notes                                           | `notes/`                                         |
| named OSS library deep-dives (lmcache, vllm, sglang, ray)     | `machine-learning/open-source-library/`          |

If two folders match, or none match cleanly, use `AskUserQuestion` with the top 2–3 candidates. Never guess silently.

## Slug rules

- kebab-case, derived from the title.
- Drop stop-words (`a`, `the`, `of`, `for`, `with`, `to`, `and`, `or`).
- ≤ 60 chars. Trim trailing partial words.
- If `<target>/<slug>.md` already exists: ask the user to (a) overwrite, (b) append `-v2`, or (c) pick a new slug.

## Parallel execution

All phases run concurrently across sessions and within a single post. Phase C diagrams render as N independent puppeteer subprocesses with isolated `--user-data-dir` — no shared state, no canvas collision. The old `/tmp/blog-writer-excalidraw.lock` is gone; if you see it, delete it.
