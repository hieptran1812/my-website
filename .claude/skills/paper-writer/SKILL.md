---
name: paper-writer
description: Analyze a research paper end-to-end and draft a detailed paper-reading blog post for the my-website blog. Reads the actual PDF, explains every technique from intuition up to the math, extracts the paper's own figures (architecture diagrams, plots, tables) straight out of the PDF and embeds them, adds redrawn Excalidraw diagrams that clarify, and writes valid frontmatter into content/blog/paper-reading/<area>/. Triggers on /paper-writer, "analyze this paper", "read this paper", "explain this paper as a blog", "phân tích paper", an arxiv link, or an attached paper PDF.
---

# paper-writer

Drafts a **detailed, technique-by-technique analysis of a single research paper** in the house style of `my-website` — accessible-expert voice, intuition before math, every symbol defined, the paper's *own* figures cut out of the PDF and embedded alongside redrawn Excalidraw diagrams that make the ideas click.

This is the paper-analysis sibling of [`blog-writer`](../blog-writer/SKILL.md) and [`finance-writer`](../finance-writer/SKILL.md). It **reuses blog-writer's diagram tooling verbatim** — the same validator (`author-scene.mjs`), layout engines (`layout-scene.mjs`), batch renderer (`render-scene-batch.mjs`), and semantic palette. Three things are new:

1. **It reads the real paper.** Fetch the PDF, extract its text with `pdftotext`, render its pages, and analyze the actual method — not a memory of it.
2. **It extracts the paper's own figures.** A dedicated `scripts/extract-figures.py` renders a page at high DPI, crops a fractional bounding box, auto-trims the margins, and ships a lossless WebP. The reader sees Figure 1 as the authors drew it, credited.
3. **It explains every technique from intuition to math.** A fixed explanation ladder (problem → analogy → mechanism → math → worked example → why/when-it-fails) applied to *each* technique in the paper, so a smart reader with no prior exposure can follow the derivation.

## When to use

- User types `/paper-writer`, `/paper-writer <arxiv-link>`, or `/paper-writer <topic/title>`
- User says "analyze this paper", "read this paper and explain it", "explain <paper> as a blog", "phân tích paper", "deep dive on the <X> paper"
- User pastes an arxiv link, a PDF URL, or attaches a paper PDF

Do NOT use for: general concept explainers with no single source paper (use `blog-writer`), short notes, literature surveys spanning many papers (that's a `blog-writer` deep-dive), or social-length copy. If the topic is finance/markets, use `finance-writer`.

## Core promise (every post must deliver)

- **Faithful to the source.** Every claim, number, and symbol traces to the actual paper text you read — not to prior knowledge of the method. Where you extrapolate beyond the paper, say so.
- **The paper's real figures.** At least **2 figures cut straight out of the PDF** (the architecture diagram is almost always one), each credited to the paper.
- **Redrawn clarity.** At least **2 Excalidraw diagrams** you author to explain something the paper's own figures leave implicit (a data-flow the paper only describes in prose, a loss decomposition, a before/after).
- **Intuition → math for every technique.** Each core technique gets the full explanation ladder (see `references/technique-explanation.md`): the problem, an analogy, the mechanism, the math with every symbol defined, a worked micro-example, and when it breaks.
- **Detailed, not a summary.** Cover *every* load-bearing technique in the paper. Depth floor is higher than a normal paper-reading post (≥ 6,500 words).

## Reference files (lazy-load)

Read each reference at the start of the phase that needs it — don't read them all upfront. Files under `blog-writer/` are shared verbatim.

| File                                                              | Read at start of                                            |
| ---------------------------------------------------------------- | ----------------------------------------------------------- |
| `references/figure-extraction.md`                                | Phase C1 (before cropping the first figure)                 |
| `references/technique-explanation.md`                             | Phase B (planning) **and** Phase D (before writing prose)   |
| `../blog-writer/references/diagram-triggers.md`                   | Phase B (planning the redrawn-diagram inventory)            |
| `../blog-writer/references/diagram-authoring.md`                  | Phase C2 (before authoring the first redrawn figure)        |
| `../blog-writer/references/voice-cheatsheet.md`                   | Phase D (before writing prose)                              |
| `templates/paper-analysis.md`                                    | Phase B (section skeleton)                                  |

Phase E gates run via `scripts/verify-paper-post.sh` — no need to inline bash.

## Frontmatter contract

```yaml
---
title: "Paper Title, or a Sentence-case Framing of It: Optional Subtitle"
date: "YYYY-MM-DD" # today's date, absolute form
description: "One-sentence promise of what the reader walks away understanding."
tags: ["paper-reading", "tag2", "..."] # 5–12 tags, lowercase-hyphen; include the method name
category: "paper-reading" # top-level folder name
subcategory: "Large Language Model" # human-readable area label; optional
author: "Hiep Tran"
featured: true # paper analyses are deep-dives
readTime: 32 # integer minutes, recomputed in Phase E
paper:
  title: "The exact paper title"
  authors: "First Author et al."
  venue: "arXiv 2024 / NeurIPS 2023 / ..."
  url: "https://arxiv.org/abs/XXXX.XXXXX"
---
```

`category` is always `paper-reading`. The post lands in `content/blog/paper-reading/<area>/<slug>.md`, where `<area>` is one of the existing subfolders (see §Area routing). Convert any relative dates to absolute `YYYY-MM-DD` before writing.

## Workflow

### Phase A — Paper intake & acquisition

1. **Get the paper into the cache as a local PDF.** Ask via `AskUserQuestion` only what's missing:
   - **Source** — arxiv link / PDF URL / local path / attached file. Required.
   - **Area** — which `paper-reading/<area>/` subfolder (see §Area routing). Default: infer from the abstract, confirm if ambiguous.
   - **Audience** — default "engineers who know the basics of the field but haven't read this paper".
   - **Depth focus** — optional: "which techniques matter most to you?" Default: cover all load-bearing methods.
2. Resolve to a local PDF and extract text + page renders:
   ```bash
   bash .claude/skills/paper-writer/scripts/get-paper.sh <arxiv-id | url | local.pdf> <slug>
   ```
   This writes `.cache/paper-writer/<slug>/paper.pdf`, `paper.txt` (via `pdftotext -layout`), and `pages/page-*.png` (≈120 DPI thumbnails for figure hunting). For an arxiv *abstract* URL it rewrites to the PDF URL automatically.
3. Find 2–4 cross-link targets: search `content/blog/**/*.md` for related posts (same method family, prior work, sibling papers).

### Phase B — Deep read & outline (STOP for approval)

1. **Read `paper.txt` in full** (`Read` the cached text). This is the ground truth for every claim, symbol, and number. For long papers, read in sections; don't skim the method or the equations.
2. **Read `references/technique-explanation.md`** and **`../blog-writer/references/diagram-triggers.md`.**
3. **Hunt the figures.** `Read` each `pages/page-*.png` thumbnail. Note every figure/plot/table worth extracting: page number, what it shows, a rough fractional bounding box `[x0,y0,x1,y1]`, and whether it's a *keeper original* (extract) or something you'll *redraw more clearly* (Excalidraw). The architecture/method figure is almost always a keeper.
4. Pick the skeleton from `templates/paper-analysis.md`.
5. Produce the outline as markdown with:
   - Proposed title, slug, and target path (`content/blog/paper-reading/<area>/<slug>.md`)
   - `paper:` metadata (title, authors, venue, url)
   - Section list (H2s with one-line summaries)
   - **Technique inventory** — one bullet per load-bearing technique in the paper, each with: the equation(s) it hinges on, the intuition/analogy you'll open with, and whether it needs a redrawn diagram.
   - **Figure plan — two tracks:**
     - **Extract** (originals): one bullet per keeper figure → `<slug>-fig<n>` · page · rough box · one-line caption. (≥ 2.)
     - **Redraw** (Excalidraw): one bullet per authored figure → claim / caption / section anchor / kind (`pipeline`/`graph`/`before-after`/`matrix`/`layered-stack`/hand-authored) · `static`|`animated`. (≥ 2. Tag `animated` only when motion carries the meaning; cap 1–3.)
   - Results table plan: which headline numbers, which baselines, which datasets.
   - 2–4 cross-links to existing posts.
6. **Print the outline and stop.** Wait for "go" / "approved" / edits.

### Phase C1 — Extract original figures from the PDF

1. **Read `references/figure-extraction.md`.**
2. For each keeper figure, refine the fractional box: render the source page at low DPI, `Read` it, and estimate `[x0,y0,x1,y1]` in 0..1 page coordinates. Trim reclaims margin but never recovers clipped ink, so estimate **asymmetrically**: measure tightly only the edges that touch the caption or body text (**exclude the "Figure N:" caption line**), and push the whitespace-facing edges *generously* past the outermost stroke — including feedback loops, return arrows, and side connectors, not just the visible box cluster. When unsure, go wider; auto-trim tightens the rest for free.
3. Extract via a manifest (renders each page once):
   ```bash
   python3 .claude/skills/paper-writer/scripts/extract-figures.py --manifest .cache/paper-writer/<slug>/figures.json
   ```
   Output: `public/imgs/blogs/<slug>-fig<n>.webp` (the `-fig<n>` infix marks an **extracted original**, distinct from redrawn `<slug>-<n>.webp`).
4. **`Read` each extracted WebP.** Confirm it is a tight crop of *only that figure*: no caption line, no body text bleeding in, no adjacent figure, nothing clipped. If it clipped or included stray text, adjust the box (or `pad`/`trim_thresh`) and re-run — the reference has the tuning knobs.
5. Extracted figures must be **≥ 900 px on the long side and ≥ 20 KB**. If smaller, raise `dpi` (try 500–600) and re-extract. `pdfimages` is a fallback for pulling a high-res embedded bitmap when the rendered crop is still soft (see reference §Fallback).

Never fabricate a figure, screenshot a browser, or paste a low-res thumbnail. If a figure cannot be cleanly extracted, redraw it in Excalidraw instead (Phase C2) and say "redrawn from Figure N" in the caption.

### Phase C2 — Author redrawn Excalidraw diagrams (reuse blog-writer)

Identical to blog-writer Phase C. **Read `../blog-writer/references/diagram-authoring.md`**, then per redrawn figure:

1. Author element JSON → `.cache/paper-writer/<slug>/<slug>-<i>.in.json`.
2. Validate + normalize: `node ../blog-writer/scripts/author-scene.mjs <in.json> <scene.json>` (or `layout-scene.mjs` for a DSL shape). Read its errors; don't bypass.
3. Batch render to PNG via `/Users/hieptran1812/Documents/mcp_excalidraw/scripts/render-scene-batch.mjs` with a manifest.
4. Convert each cache PNG → lossless WebP at `public/imgs/blogs/<slug>-<i>.webp` (`cwebp -quiet -lossless -m 6`).
5. Redrawn WebP must be **≥ 1600×900 px, ≥ 40 KB** (blog-writer's strict floor — these are full-canvas diagrams, held to the same bar).

If any redrawn figure was tagged `animated` in Phase B, follow blog-writer Phase C-anim (**read `../blog-writer/references/animated-figures.md`**, author `<figure class="blog-anim">` blocks, validate with `../blog-writer/scripts/check-anim.mjs`). Do NOT use the `mcp__excalidraw__*` MCP tools here.

### Phase C3 — Visual self-review (vision gate, mandatory)

`Read` **every** figure — extracted and redrawn — and write a one-line verdict (`PASS` / `FAIL: <what's wrong>`) before any prose is built around it.

- **Extracted originals** — judge the *crop*, not the design (you didn't draw it): tight to the figure, caption excluded, nothing clipped, nothing foreign bleeding in, legible at the shipped size. A FAIL means re-extract with a better box/DPI (back to Phase C1), never edit the prose.
- **Redrawn diagrams** — the full blog-writer Phase C2 rubric (read `../blog-writer/references/diagram-authoring.md §Visual self-review`): faithful, arrows legible, balanced, no dead space, text renders, squint test; and set-level diversity.

Advance to Phase D only when every figure is a clean PASS.

### Phase D — Draft (intuition → math)

1. **Read `../blog-writer/references/voice-cheatsheet.md` and re-read `references/technique-explanation.md`.**
2. Write the full markdown via `Write` to the resolved target path. Frontmatter exactly per contract (including the `paper:` block); today's date.
3. **Open with the TL;DR box** — a `> [!tldr]` blockquote: what the paper claims, why it matters, the most surprising result, where it fails.
4. **Apply the explanation ladder to every technique** (from `technique-explanation.md`): problem it solves → intuition/analogy → mechanism step-by-step → the math (define every symbol on first use, annotate tensor shapes) → a worked micro-example (tiny numbers or pseudocode) → why it works / when it fails. Math in `$...$` / `$$...$$`; brace-wrap any inline math starting with a digit as `${...}$`.
5. **Embed the paper's own figures where the method is introduced**, credited: `![Figure 1 from Vaswani et al. (2017): the Transformer](/imgs/blogs/<slug>-fig1.webp)`. Embed redrawn diagrams under the section they clarify: `![alt](/imgs/blogs/<slug>-2.webp)`. Every embed is `.webp`. Paste any animated `<figure class="blog-anim">` block verbatim (no blank lines inside, `<figure` at column 0).
6. **Results section**: a table with baselines named, datasets, and compute scale. One paragraph on "what's load-bearing in their setup that might not transfer".
7. **Critique section**: what's strong, what's weak/unfalsifiable/cherry-picked, what ablation is missing, and an explicit "what would change my mind" line.
8. Add cross-links with relative paths: `[attention](/blog/paper-reading/large-language-model/attention-is-all-you-need)` (drop `content/` and `.md`).
9. **End the post with a `## References`** section: the paper (arxiv/DOI), the code repo if any, and 2–3 sibling posts. Never write a generic "Conclusion".

### Phase E — Verify (hard gates)

```bash
bash .claude/skills/paper-writer/scripts/verify-paper-post.sh <post.md> <slug>
```

The script checks everything blog-writer's gate does — word-count floor (≥ 6,500), figure-count floor (≥ 6), abstraction coverage, webp-only embeds, no stray non-webp renders, no ASCII/Unicode/mermaid diagram substitutes, slug-match, no-H1-in-body, English-only, frontmatter sanity — split into two sharpness tiers (**extracted** `-fig<n>` ≥ 900 px long-side / ≥ 20 KB; **redrawn** `-<n>` ≥ 1600×900 / ≥ 40 KB) — **plus paper-specific gates**:

- **TL;DR** callout near the top.
- **≥ 2 extracted originals** (`-fig<n>.webp`) and **≥ 2 redrawn diagrams** (`-<n>.webp`) embedded.
- **Math rigor**: ≥ 3 display-math `$$…$$` blocks (a paper analysis that never writes an equation is a summary, not an analysis).
- **References** section present with the paper URL.
- **"what would change my mind"** line in the critique (warn if absent).

Any FAIL means re-enter the named phase and fix. **The fix for a missing figure is always to add the figure, never to delete the prose.**

**Final figure pass (prose-aware).** Re-check placement and faithfulness now that prose exists: each figure sits under the section it illustrates; each extracted figure's caption credits the paper; every node label in a redrawn figure appears in the prose ±200 lines; arrow directions match the text you actually wrote. If you re-authored any redrawn figure, re-run the Phase C3 rubric on it.

Then report: final path, word count, recomputed `readTime`; new images (extracted vs redrawn, with sizes); which gates passed/failed; 2–4 cross-links the *user* might add to other posts; reminder to run `npm run dev` and load the page before committing.

### Phase F — Clean up the cache (only after Phase E passes)

Once `verify-paper-post.sh` exits 0 and the visual review is clean, delete this post's cache. The published post depends only on `public/imgs/blogs/<slug>-*.webp` and the markdown; the cache holds throwaway intermediates (`paper.pdf`, `paper.txt`, `pages/`, `*.in.json`, `*.scene.json`, `*.png`, `figures.json`, manifests).

```bash
# Run ONLY if Phase E gates passed. Scoped to this slug — never wipe .cache/paper-writer wholesale.
rm -rf .cache/paper-writer/<slug>
```

Rules: gate it (a FAILed post keeps its cache for re-authoring), scope to the slug (never touch sibling folders), and mention the cleanup in the final report.

## Area routing

`category` is always `paper-reading`. Pick `<area>` from the existing subfolders under `content/blog/paper-reading/`:

| Paper topic                                                        | `<area>` folder            |
| ----------------------------------------------------------------- | -------------------------- |
| llm architecture/training/inference, transformers, attention      | `large-language-model`     |
| agents, tool-use, multi-agent, planning                           | `ai-agent`                 |
| reasoning, chain-of-thought, RL for reasoning                     | `reasoning`                |
| diffusion, flow-matching, score-based generation                  | `diffusion-model`          |
| GANs, VAEs, autoregressive image/audio generation                 | `generative-model`         |
| vision, detection, segmentation, ViT                              | `computer-vision`          |
| vision-language, any-to-any, multimodal fusion                    | `multimodal`              |
| RLHF, PPO/GRPO, policy optimization, RL theory                    | `reinforcement-learning`   |
| retrieval, RAG, dense/sparse search, rerankers                    | `information-retrieval`    |
| SAE, steering, probing, circuits, mech-interp                     | `ai-interpretability`      |
| alignment, safety evals, red-teaming, oversight                   | `ai-safety`                |
| ASR, TTS, speech representation                                   | `speech-processing`        |
| optimization, generalization, learning theory                     | `machine-learning-theory`  |
| backprop, optimizers, normalization, general DL                   | `deep-learning`            |
| everything else ML                                                | `machine-learning`         |

If two areas fit or none fits cleanly, `AskUserQuestion` with the top 2–3 candidates. Never guess silently. If the paper genuinely needs a new subfolder, propose it and confirm.

## Slug rules

- kebab-case, derived from the paper title (or its common short name).
- Drop stop-words (`a`, `the`, `of`, `for`, `with`, `to`, `and`, `or`, `via`, `in`).
- ≤ 60 chars. Trim trailing partial words. Prefer the memorable name if the paper has one (`flash-attention`, not `fast-and-memory-efficient-exact-attention`).
- If `<target>/<slug>.md` already exists: ask to (a) overwrite, (b) append `-v2`, or (c) pick a new slug.

## Parallel execution

Figure extraction (C1), redrawn-diagram rendering (C2), and per-figure visual review (C3) all parallelize. Extraction renders each page once and crops N figures from it; redrawn diagrams render as independent puppeteer subprocesses. For a figure-heavy paper (≥ 8 figures), dispatch parallel reviewer subagents in Phase C3 — one per figure, each opening its WebP with `Read` and returning the verdict line.
