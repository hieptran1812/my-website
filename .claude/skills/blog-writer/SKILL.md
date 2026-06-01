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

| File | Read at start of |
|---|---|
| `references/diagram-triggers.md` | Phase B (planning the abstraction inventory) |
| `references/diagram-authoring.md` | Phase C (before authoring the first figure) |
| `references/voice-cheatsheet.md` | Phase D (before writing prose) |
| `templates/{deep-dive,explainer,paper-reading}.md` | Phase B (skeleton for the chosen depth) |

Phase E gates run via `scripts/verify-post.sh` — no need to inline bash.

## Frontmatter contract

```yaml
---
title: "Sentence-case Title with Optional Subtitle After a Colon"
date: "YYYY-MM-DD"            # today's date, absolute form
publishDate: "YYYY-MM-DD"     # optional; equals date if absent
description: "One-sentence promise of what the reader walks away with."
tags: ["tag1", "tag2", "..."] # 5–12 tags, lowercase-hyphen
category: "machine-learning"  # top-level folder name
subcategory: "Large Language Model"  # human-readable; optional
author: "Hiep Tran"
featured: true                # true for deep-dives, false for explainers/notes
readTime: 32                  # integer minutes, recomputed in Phase E
aiGenerated: true             # always true when produced by this skill
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
   - **Abstraction inventory + diagram plan** — one bullet per abstract concept, each with claim / caption / section anchor / sketch (see `diagram-triggers.md`). Figure count = abstraction count.
   - Case-study list (deep-dives): 3–12 named incidents, each one line
   - Code snippet plan: language, what it demonstrates, ~lines
   - 2–4 cross-links to existing posts
5. **Print the outline and stop.** Wait for "go" / "approved" / edits.

### Phase C — Diagrams (parallel, headless)

1. **Read `references/diagram-authoring.md`.**
2. For each planned figure: author element JSON → `.cache/blog-writer/<slug>/<slug>-<i>.in.json`.
3. Validate + normalize each: `node scripts/author-scene.mjs <in.json> <scene.json>`. The validator enforces fonts, palette, containment, no-overlap, density, claim length, caption presence, snap grid. Read its error messages — they name the rule and offending element. Do not bypass.
4. Batch render all scenes via `mcp_excalidraw/scripts/render-scene-batch.mjs` with a manifest (one browser, ~150 ms per figure after startup).
5. Verify each PNG ≥ 1600×900 px, ≥ 80 KB.

If the renderer exits non-zero or any PNG fails the sharpness floor: **stop and surface to the user**. Never substitute ASCII art, ```text``` boxes, Unicode box-drawing, prose-only "diagrams", or inline ```mermaid``` source. Those are hard failures.

Do NOT use the `mcp__excalidraw__*` MCP tools in this phase — they target the live canvas, which is not on this code path.

### Phase D — Draft

1. **Read `references/voice-cheatsheet.md`.**
2. Write the full markdown via `Write` to the resolved target path. Frontmatter exactly per contract; today's date.
3. Embed each PNG immediately under the section heading it illustrates: `![alt](/imgs/blogs/<slug>-<n>.png)`. The first figure is referenced in the intro paragraph.
4. Add cross-links inline using relative paths: `[KV cache](/blog/machine-learning/large-language-model/kv-cache)` (drop the `content/` prefix and `.md` extension).

### Phase E — Verify (hard gates)

Run:

```bash
bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> <depth>
```

`<depth>` is one of `deep-dive`, `explainer`, `paper-reading`. The script checks: word-count floor, diagram-count floor, abstraction coverage (figure within 30 lines of every prose abstraction), PNG sharpness, forbidden text-diagram substitutes, slug-match on every image, no-H1-in-body, English-only, frontmatter sanity.

Any FAIL means re-enter the named phase and fix. The fix for missing figures is *always* to add the figure, never to delete the prose.

For diagram visual review (composition, faithfulness, text containment in the rendered PNG), open each `public/imgs/blogs/<slug>-*.png` with `Read` and inspect:
- Every text element renders in Virgil or Cascadia (no system font leakage)
- Bounding box covers ≥ 70% of canvas; no wide empty bands
- Every node label appears in the prose ±200 lines around its anchor
- No text overflow, no overlapping bboxes, no arrow-through-label
- **Arrow accuracy**: every arrow direction matches the causal claim in the prose; head and tail land on the intended node edges (not floating, not piercing a bbox); orthogonal where the relationship is axial; arrowhead style (`arrow` / `triangle_outline` / `bar` / `dot`) and stroke style (solid / dashed / dotted) are consistent across the figure and used per the semantics in `diagram-authoring.md`. Reversed or dangling arrows = re-author the figure, not the prose. The validator's rule 3c now rejects any arrow polyline segment that crosses a non-endpoint node bbox; if it fires on a `graph`-engine figure, re-author the DSL (the layer membership is likely wrong — two nodes in one layer that should be in adjacent layers).

Then report to the user:
- Final file path, word count, recomputed `readTime`
- New images written (paths and sizes)
- Which gates passed/failed and what was added on the second pass
- 2–4 suggested cross-links the *user* should consider adding to *other* existing posts (don't edit those unless asked)
- Reminder to run `npm run dev` (or `bun run dev`) and load the page locally before committing

## Path / category routing

| Topic keywords                                                | Target folder                                           |
| ------------------------------------------------------------- | ------------------------------------------------------- |
| llm, transformer, attention, kv-cache, tokenizer, decoding    | `machine-learning/large-language-model/`                |
| docker, kubernetes, vllm, triton, serving, gpu, inference-ops | `machine-learning/mlops/`                               |
| agent, tool-use, langgraph, react-agent, planner              | `machine-learning/ai-agent/`                            |
| sae, steering, interpretability, probe, circuit               | `machine-learning/ai-interpretability/`                 |
| diffusion, vit, segmentation, detection, image                | `machine-learning/computer-vision/`                     |
| backprop, optimizer, lr-schedule, init                        | `machine-learning/deep-learning/`                       |
| sft, dpo, rlhf, lora, peft, finetuning                        | `machine-learning/training-techniques/`                 |
| xgboost, lightgbm, catboost, random-forest, tabular           | `machine-learning/traditional-machine-learning/`        |
| dsp, fft, audio, wavelet                                      | `machine-learning/signal-processing/`                   |
| linear-algebra, probability, optimization-theory              | `machine-learning/mathematics/`                         |
| paper title / arxiv link                                      | `paper-reading/<area>/`                                 |
| react, next.js, build-tool, framework                         | `software-development/`                                 |
| backtest, market-microstructure, alpha                        | `trading/`                                              |
| short scratch notes                                           | `notes/`                                                |
| named OSS library deep-dives (lmcache, vllm, sglang, ray)     | `machine-learning/open-source-library/`                 |

If two folders match, or none match cleanly, use `AskUserQuestion` with the top 2–3 candidates. Never guess silently.

## Slug rules

- kebab-case, derived from the title.
- Drop stop-words (`a`, `the`, `of`, `for`, `with`, `to`, `and`, `or`).
- ≤ 60 chars. Trim trailing partial words.
- If `<target>/<slug>.md` already exists: ask the user to (a) overwrite, (b) append `-v2`, or (c) pick a new slug.

## Parallel execution

All phases run concurrently across sessions and within a single post. Phase C diagrams render as N independent puppeteer subprocesses with isolated `--user-data-dir` — no shared state, no canvas collision. The old `/tmp/blog-writer-excalidraw.lock` is gone; if you see it, delete it.
