---
name: blog-writer
description: Draft a long-form, principal-engineer-voice blog post for the my-website blog. Plans an outline, generates Excalidraw diagrams, then writes the full markdown into the correct content/blog/<category>/<subcategory>/ folder with valid frontmatter. Triggers on /blog-writer, "write a blog post about", "draft an article on", "new blog:".
---

# blog-writer

Drafts a long-form blog article in the house style of `my-website` — opinionated, principal-engineer voice, with intuition before math, runnable code, comparison tables, case studies, and Excalidraw diagrams.

## When to use

- User types `/blog-writer` or `/blog-writer <topic>`
- User says "write a blog post about X", "draft an article on X", "new blog: X"
- User asks for a deep-dive / explainer / paper-reading post for the blog
- User points at a topic and an existing similar post and says "do one like that for X"

Do NOT use for: short notes, README updates, comments on existing posts, social-media-length copy.

**Parallel execution (fully concurrent, including diagrams).** Every phase runs in parallel — across sessions and within a single post. Diagrams in Phase C are no longer serialized through the live MCP canvas: each figure is rendered by `mcp_excalidraw/scripts/render-scene.mjs`, a stateless headless puppeteer process that owns its own browser tab and exports the PNG entirely client-side. N figures launch as N independent subprocesses; two `/blog-writer` invocations interleave with zero shared state. The old `/tmp/blog-writer-excalidraw.lock` is gone — if you see it on disk, delete it.

## Reference articles (read before drafting — calibrate voice)

The skill should `Read` ~80 lines from the matching reference at the start of Phase D:

- **Deep-dive gold standard** — `content/blog/machine-learning/mlops/docker-optimization-for-llm-and-ai-workloads.md`
- **Explainer gold standard** — `content/blog/machine-learning/large-language-model/kv-cache.md`
- **Featured + aiGenerated frontmatter** — `content/blog/machine-learning/large-language-model/kv-cache-optimization-and-management.md`

## House style guardrails

Voice & structure:
- **Always write in English.** Regardless of the language the user uses to invoke this skill or describe the topic, the entire blog post (title, frontmatter values, body, captions, code comments) must be in English. Never write any portion of the article in Vietnamese, Chinese, or any other language.
- **Never use H1 (`#`) headings inside the article body.** The post title lives in frontmatter (`title:`), and Next.js renders it as the page H1. Top-level section headings start at `##`, sub-sections at `###`, sub-sub-sections at `####`. A `#` line anywhere in the markdown body is a hard failure.
- First-person plural (`we`) for shared reasoning. First-person singular (`I have personally debugged…`) only for war-stories.
- Open with the **real problem** or a sharp mismatch — never a dictionary definition.
- Build intuition with a concrete analogy (library, restaurant, city map) before introducing math.
- Math goes in `$...$` / `$$...$$`; symbols are defined the first time they appear.
- Code blocks must look runnable: real imports, real flags, real version numbers. Avoid pseudocode unless explicitly labeled.
- Use comparison tables for "naive vs optimized", "assumption vs reality", "strategy / when to use / trade-off".
- Reach for headings `##` for sections and `###` for sub-sections. Numbered top-level sections (`## 1. …`, `## 2. …`) are appropriate for deep-dives, not for explainers.
- Every deep-dive ends with a **case study** section: 6–12 numbered, named incidents (each ~250–400 words) followed by a closing **"When to reach for X / when not to"** section. Never a generic "Conclusion".
- **Length is non-negotiable. Deep-dives MUST be ≥ 50 minutes read time (≈ 11,000 words minimum, target 12,000–16,000).** Explainers ≥ 25 min (≈ 5,500 words). Paper readings ≥ 30 min (≈ 6,500 words). If a draft comes in short, do not ship it — expand the weakest sections (more case studies, deeper internals, more comparison tables, more code) until the floor is met. The word-count check in Phase E is a hard gate, not a warning.
- **Deep analysis is required.** Every section must answer "why does this work, when does it fail, what are the second-order consequences." Surface-level summaries are unacceptable. For every claim: name the mechanism, quantify the trade-off, and give at least one concrete number, benchmark, or failure mode. Treat each H2 as if it could be its own short blog post.
- Inside each major section, include at least one of: a comparison table, a runnable code block (≥ 15 lines), a measured benchmark with units, or a worked numerical example. Sections that are pure prose are a smell.

Diagrams:
- **Excalidraw is mandatory and must be used correctly.** Every figure in every post is authored through the Excalidraw MCP — no exceptions, no shortcuts. "Used correctly" means: native Excalidraw fonts only (`fontFamily: 1` Virgil for prose, `fontFamily: 3` Cascadia for code, set explicitly on every text element), the documented sloppiness/roughness/strokeWidth defaults, the semantic color palette applied by meaning not aesthetics, and bound text via `containerId` for any label that lives inside a shape. Misusing the API (omitting `fontFamily`, free-floating text on top of shapes, custom colors outside the palette, decorative-only elements) is a hard failure — re-author the figure.
- **Scientific layout is mandatory.** Every diagram is a figure in a technical article and must be laid out like one: one provable claim per figure, balanced composition, ≥ 70% canvas bounding-box coverage with no empty quadrants, aspect ratio chosen by content (pipelines wide-and-short, stacks tall-and-narrow), tight ≥ 40 px inter-element spacing without overlap, and arrows/labels that match the surrounding prose word-for-word. Loose, decorative, or aesthetically-driven layouts fail the Phase E composition gate.
- **Anything that needs illustration earns a figure — there is no upper limit.** Do not think of figures as a budget; think of them as the visual track running parallel to the prose. Every claim, mechanism, structure, comparison, or relationship that a reader would understand faster from a picture gets a picture. If the post needs 5 figures, draw 5; if it needs 25, draw 25. The old "1–4 PNGs" rule is removed.
- **Trigger a figure whenever any of these appears in the prose** — this list is illustrative, not exhaustive. Any *other* place where a picture would clarify the idea also qualifies:
  - Multi-step processes, pipelines, request lifecycles
  - Layered or hierarchical structures (stacks, taxonomies, nesting)
  - Before/after comparisons, naive vs optimized, A/B variants
  - State machines, transitions, lifecycles
  - Data structures and memory layouts (cache, page table, KV block, queue, buffer, ring, hashmap)
  - Math or physics intuitions that map to a picture (a curve, a region, a vector field, a tensor reshape)
  - System architectures with ≥ 3 components
  - Tradeoff matrices (axes × choices), capability tables
  - Algorithm walkthroughs, decision trees, recursion shapes
  - Timing or sequencing diagrams, parallelism layouts, GPU stream timelines
  - Control- or data-flow graphs, dependency graphs
  - Concrete-but-non-obvious mechanics that benefit from a picture: a config's effect, a CLI's mental model, a regex's matching shape, a packet's wire format, a file system's on-disk layout, a transformer's attention pattern on a small example
  - Any prose phrase that reaches for a visual analogy: "imagine", "think of it as", "consider the case where", "the way this works is", "under the hood", "looks something like"
- **Coverage rule.** If the prose introduces an idea and the next paragraph does not have a figure within 30 lines, **add one** — a missing figure is a defect, not a stylistic choice. This applies to abstract concepts AND to concrete-but-illustration-worthy details (a confusing API surface, an unintuitive flag interaction, a benchmark comparison).
- The first figure (the "mental model" image) is referenced in the intro paragraph with a sentence like "The diagram above is the mental model: …". Subsequent figures sit directly under the section heading or paragraph that introduces the concept.
- **Sanity-check floors by depth (minimums, never caps).** Explainer ≥ 4 figures, paper-reading ≥ 5, deep-dive ≥ 8 — but most posts in any of these categories will exceed these floors substantially, often by 2–3×. The ceiling is set by the post's content, not by these numbers. If a 50-min deep-dive ends up with 20 figures because it has 20 things worth illustrating, that's correct, not excessive.
- Hand-drawn Excalidraw style (sloppiness 1, roughness 1, default Virgil/Cascadia fonts).
- Saved to `public/imgs/blogs/<slug>-<n>.png`. Embedded as `![alt text](/imgs/blogs/<slug>-<n>.png)` directly under the heading they illustrate.
- **Diagrams MUST be real Excalidraw PNGs. Never substitute ASCII art, ```text``` boxes, Unicode box-drawing, code-block "diagrams", or prose-bulleted "diagrams". These are not acceptable fallbacks under any circumstance.** If the Phase C headless renderer fails (puppeteer crash, missing build, validation error that you cannot fix), **stop and surface the failure to the user** — do not ship the post. Mermaid sources are acceptable only when first converted to scene JSON via `@excalidraw/mermaid-to-excalidraw` and then rendered through the same `render-scene.mjs` pipeline. Posts without proper image diagrams are considered incomplete and fail the Phase E diagram gate.

## Frontmatter contract

Use exactly these keys (mirror what existing posts already do — don't invent new ones):

```yaml
---
title: "Sentence-case Title with Optional Subtitle After a Colon"
date: "YYYY-MM-DD"            # always today's date in absolute form
publishDate: "YYYY-MM-DD"     # optional; equal to date if absent
description: "One-sentence promise of what the reader walks away with." # OR excerpt:
tags: ["tag1", "tag2", "..."] # 5–12 tags, lowercase-hyphen
category: "machine-learning"  # top-level folder name
subcategory: "Large Language Model"  # human-readable; optional but recommended
author: "Hiep Tran"
featured: true                # true for deep-dives, false for explainers/notes
readTime: 32                  # integer minutes, recomputed in Phase E
aiGenerated: true             # always true when produced by this skill
---
```

Notes:
- `category` is the directory name under `content/blog/` (kebab-case). `subcategory` is a display label, not a path.
- Today's date resolves via the runtime context (`currentDate`). Convert any relative dates the user mentions ("yesterday", "next Monday") to absolute `YYYY-MM-DD` before writing.

## Workflow

### Phase A — Topic intake

Capture from the user (ask via `AskUserQuestion` only what's missing):

1. **Topic** — required.
2. **Depth** — `deep-dive` | `explainer` | `paper-reading`. Default = `deep-dive` if the user says "blog" without qualifier and the topic is engineering-heavy; `explainer` if the topic is conceptual; `paper-reading` if the user references a paper title or arxiv link.
3. **Audience** — default "senior ML engineers / staff-level interviewees". Override only if the user says otherwise.
4. **Cross-link targets** — search `content/blog/**/*.md` for 2–4 related posts to link to from the new article.

### Phase B — Research & outline (STOP for approval)

1. Optionally fetch context: `WebSearch` for primary sources, `WebFetch` for specific URLs the user provided, or `claude-obsidian:wiki-query` if a vault exists.
2. Pick the matching template from `templates/`:
   - `deep-dive.md` for layered/operational topics
   - `explainer.md` for "what is X and how does it work"
   - `paper-reading.md` for paper summaries
3. Produce the outline as markdown with:
   - Proposed title and slug
   - Target path (`content/blog/<category>/[<subcategory>/]<slug>.md`)
   - Section list (H2s, with one-line summaries)
   - **Abstraction inventory + diagram plan.** First, walk the outline section by section and write a bullet list of every abstract concept the post will introduce — each mechanism, each layered structure, each before/after, each state machine, each data layout, each tradeoff matrix, each algorithm flow, each math intuition (per the trigger list in the house-style guardrails). One bullet per abstraction. Then, for each bullet, plan one figure with: the **single claim** it proves (≥ 8 words — `_claim` field), the **caption** (one sentence, not a label restatement — `_caption` field), the section it sits in, the markdown anchor it binds to (used by Phase E's anchor-faithfulness gate), and a **rough sketch** of which boxes / arrows / labels appear. **Author each figure at the Excalidraw element level** — give every figure the freedom to take whatever shape its claim needs (annotated photos, hybrid layouts, math intuitions, asymmetric flows, unusual geometries). The figure count equals the abstraction count — if you found 9 abstractions, plan 9 figures. Element-level reference templates live at `.claude/skills/blog-writer/diagrams/{layered-stack,before-after}.elements.json`; the older DSL shortcuts (`*.dsl.json`) are still available for the most regular shapes (pipeline, stack, before/after, matrix, tree, timeline, grid, graph) but should NOT be the default — DSL trades flexibility for consistency, and the figure-quality mandates apply just as well to hand-authored elements.
   - Case-study list (for deep-dives): 3–12 named incidents, each one line
   - Code snippet plan: language, what it demonstrates, ~lines
   - 2–4 cross-links to existing posts
4. **Print the outline and stop.** Wait for the user to say "go" / "approved" / "looks good", or to give edits. If they give edits, revise and re-emit. Do not proceed silently.

### Phase C — Diagrams (headless render, fully parallel)

Diagrams are rendered by `mcp_excalidraw/scripts/render-scene.mjs` — a stateless `scene.json → PNG` puppeteer process. Each render owns its own browser instance with an isolated `--user-data-dir`, exports the PNG entirely client-side (no MCP server state is ever read or written), then exits. **Two consequences**: (1) all N figures for a post run in parallel as N independent subprocesses, and (2) parallel `/blog-writer` sessions cannot collide — there is no shared canvas to corrupt. Do not use the `mcp__excalidraw__*` MCP tools in this phase; they exist for the live canvas, which is not on this code path.

**One-time prerequisites (skip if already done).** Verify and bootstrap:

```bash
MCP_REPO=/Users/hieptran1812/Documents/mcp_excalidraw   # adjust if relocated
test -d "$MCP_REPO/dist/frontend" && test -f "$MCP_REPO/dist/frontend/headless.html" \
  || ( cd "$MCP_REPO" && npm install && npm run build:frontend )
test -d "$MCP_REPO/node_modules/puppeteer" \
  || ( cd "$MCP_REPO" && npm install )
```

If the user has not built the frontend before (no `dist/frontend/headless.html`) or has not installed puppeteer (`devDependencies` was added when this skill was redesigned), run the install/build above once. After that, the renderer is fully self-contained.

**Per-diagram pipeline (run all N in parallel).** Authoring is **element-level by default** — you write the Excalidraw element list directly so each figure is free to take whatever shape its claim needs. The DSL shortcut is *opt-in* for the most regular shapes (pipeline / stack / before-after / matrix / tree / timeline / grid / graph) where you want to skip coordinates; the moment a figure needs something irregular (annotated photo, hybrid layout, asymmetric flow, math intuition with arrows pointing into a formula, custom matrix with merged cells), drop the DSL and write the elements yourself.

For each planned diagram `i` (1-indexed):

1. **Write the figure as element JSON.** Build a payload of the form:
   ```json
   {
     "title": "<figure title>",
     "_claim": "<≥ 8 word sentence the figure proves>",
     "_caption": "<one-sentence thesis under the title>",
     "elements": [
       { "type": "rectangle", "id": "node1", "x": 200, "y": 240, "width": 360, "height": 200,
         "backgroundColor": "#a5d8ff", "strokeWidth": 2, "label": "scheduler" },
       { "type": "arrow", "x": 560, "y": 340, "width": 200, "height": 0,
         "strokeWidth": 2, "endArrowhead": "arrow", "points": [[0,0],[200,0]] },
       { "type": "text", "x": 600, "y": 100, "width": 800, "height": 40,
         "text": "vLLM request lifecycle", "fontSize": 32, "fontFamily": 1, "id": "title" },
       ...
     ],
     "export": { "padding": 48, "minWidth": 1600, "minHeight": 900 }
   }
   ```
   Save it to `.cache/blog-writer/<slug>/<slug>-<i>.in.json`. Authoring rules (all enforced in step 2 by `author-scene.mjs`):
   - Every text element has explicit `fontFamily` (`1`=Virgil for prose, `3`=Cascadia for code/identifiers) and `fontSize` ≥ 22.
   - Every shape's `backgroundColor` is from the semantic palette: `#a5d8ff` primary, `#ffec99` caution, `#ffc9c9` danger, `#b2f2bb` success, `#d0bfff` external, `#e9ecef` soft-gray, or `transparent`.
   - Bound text uses the shape's `label` field (the validator expands `{label: "..."}` on a rectangle into a centered text element with `containerId` automatically) — don't hand-write free-floating text on top of shapes.
   - Containers are sized to fit their label per `width ≥ chars × fontSize × 0.6 + 48`.
   - **Header layout (mandatory exact y-coordinates).** Every figure includes one `text` element with `id: "title"` (fontSize 32, Virgil) at **y=60**, and one caption text element matching `_caption` (fontSize 28, Virgil) at **y=120**. Body content (the first row/column of node rectangles) **must start at y ∈ [180, 220]** — i.e., directly under the caption with at most a 60-px gap. A body that starts at y=300 or beyond creates a dead vertical band between the header and the figure body; this is a layout bug, not a stylistic choice. The DSL layout engines do this automatically; for hand-authored element-level figures, hard-code `y: 60` for title, `y: 120` for caption, and `y: 200` (or close) for the first body row. The same rule applies horizontally: the bounding box of body content should start within ~140 px of the canvas left edge — leaving a 600-px margin reads as accidental empty space.
   - **Card sizing rule (mandatory): cards are sized to fit their content, not stretched to fill the canvas.** Width = `chars × fontSize × 0.6 + 48`, height = `lines × fontSize × 1.25 + 60`, both rounded up to the nearest 20 px. **Do not** declare a card 1100 px tall just because the canvas allows it — the text renders centered, leaving 80% of the card empty (this is the most common authoring mistake). If a card has a 5-line label at fontSize 22, its height should be `5 × 22 × 1.25 + 60 ≈ 200` px, not 1100. To make a row of cards visually fill more vertical space, give every card the same height (the tallest content's required height), not an arbitrarily large one. Empty space inside a shape is a defect; use the bounding-box coverage rule (≥ 70 % dominant axis, ≥ 40 % minor) by *adding more cards or rows*, not by stretching individual shapes.
   - **Free-floating text widths (mandatory): the declared `width` of a text element must equal the rendered text width.** Excalidraw renders text using its actual character widths, but the validator currently only enforces this on shape-bound text. For free-floating text (edge labels, annotations, axis ticks), if you declare `width: 60` and the text is `"compute"` (renders ~92 px), the rendered text *overflows the declared bbox to the right* — usually into an adjacent node. Always set `width = ceil(chars × fontSize × 0.6 / 20) × 20`. For a 7-char label at fontSize 22 that's `ceil(92.4 / 20) × 20 = 100`. The validator was extended to catch this; if you see "text element X: declared width N below rendered M", widen the declared box.
   - **Edge label placement (mandatory): in the gap between nodes, never on top of a node.** Compute the gap between two consecutive nodes' x-coordinates (right edge of A to left edge of B). The edge label's `x..x+width` interval must fall entirely within that gap. Vertically, place the label *above* the arrow line (`y = arrow.y - fontSize - 16`) so it sits in the empty band between the title-band and the body row. If the gap is < `chars × fontSize × 0.6 + 24`, either widen the gap (move B further right), shorten the label, or break it across two lines. Never accept a label whose bbox crosses a node's bbox — this is a hard fail at visual review.
   - Reference templates: `.claude/skills/blog-writer/diagrams/{layered-stack,before-after}.elements.json`.
   - **Optional shortcut**: if the figure happens to fit one of the canned shapes (regular pipeline, stack, before-after, matrix, tree, timeline, grid, graph), you can write a DSL JSON instead and pass it through `node .claude/skills/blog-writer/scripts/layout-scene.mjs <dsl.json> <scene.json>` first — the engine emits an element-level scene JSON the renderer consumes identically. Treat the DSL as a labor-saver, not a constraint; whenever the shape doesn't quite fit, abandon DSL and author the elements directly.
2. **Validate + normalize the element JSON.** `author-scene.mjs` normalizes labels into bound-text pairs, fills missing element defaults, and enforces every invariant (containment, sibling overlap, ≥ 70 % bounding-box coverage on the dominant axis with ≥ 40 % on the minor axis, fontFamily ∈ {1, 3}, palette compliance, `_claim` ≥ 8 words, `_caption` present and non-restating, information density ≥ 6 unique non-stopword tokens):
   ```bash
   node .claude/skills/blog-writer/scripts/author-scene.mjs \
     ".cache/blog-writer/<slug>/<slug>-<i>.in.json" \
     ".cache/blog-writer/<slug>/<slug>-<i>.scene.json"
   ```
   Non-zero exit names the offending element/rule. Fix the input JSON (resize the container, separate two siblings, swap to a palette color, set fontFamily, add density to labels) and re-run. **Do not edit the scene file directly** — it's regenerated.
3. **Render all validated scenes in ONE puppeteer browser via the batch script.** `render-scene-batch.mjs` pays the ~1.5 s browser startup once, then loops in the already-loaded harness page calling `__renderScene` / `__renderMermaid` for each input. Per-figure cost after the first drops from ~1.5 s to ~150 ms, and total memory is one chromium process (~250 MB) instead of N. Build a manifest and hand it off:
   ```bash
   slug="<slug>"
   manifest=".cache/blog-writer/$slug/manifest.json"
   ls .cache/blog-writer/$slug/*.scene.json \
     | jq -R -s 'split("\n") | map(select(length>0)) | map({in: ., out: ("public/imgs/blogs/" + (. | split("/") | last | sub(".scene.json"; ".png")))})' \
     > "$manifest"
   node /Users/hieptran1812/Documents/mcp_excalidraw/scripts/render-scene-batch.mjs "$manifest"
   ```
   The batch script sniffs each input's shape: element-form scenes route to `__renderScene`, `{ mermaid, export }` scenes (produced by `type: "graph"`) route to `__renderMermaid`. Both share the same sharpness floor and isolation. If you have multiple parallel `/blog-writer` sessions, each session spawns its own batch process — they don't share state. The legacy `render-scene.mjs <in> <out>` one-shot path still works (and `render-scene-batch.mjs <in> <out>` accepts the same argv), but inside Phase C always prefer the manifest form so you're not paying the startup cost N times.
4. **Sharpness gate — per PNG, after all renders settle.**
   ```bash
   for f in public/imgs/blogs/${slug}-*.png; do
     dims=$(sips -g pixelWidth -g pixelHeight "$f" 2>/dev/null | awk '/pixel/ {print $2}' | paste -sd' ' -)
     bytes=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
     echo "$f  $dims  ${bytes}B"
   done
   ```
   Required for every PNG: **width ≥ 1600**, **height ≥ 900**, **size ≥ 80 KB**. The renderer scales small bounding boxes up to this floor automatically (via `getDimensions` in `headless.tsx`), so a failing gate here usually means an underweight figure (too few elements, sparse layout, all-neutral fills, missing detail) — go back to step 1 and add information density: more nodes, real numbers/units in labels, internal structure, annotations.

If the renderer exits non-zero for any diagram, **stop and surface the failure to the user**. Do not ship a post with a missing or corrupt diagram, and do not substitute ASCII art, ```text``` boxes, Unicode box-drawing, or prose-only "diagrams" — those are still hard failures regardless of how the renderer is configured.

**Accuracy bar for diagrams.** A diagram that is decorative is a failure. Every diagram must:
- Reflect the exact terminology used in the article (variable names, component names, arrow labels match prose 1:1).
- Have arrows that point in the correct direction of data/control flow — verify by re-reading the prose section it illustrates.
- Use the accent palette below *semantically* (blue = main path, amber = caution/cost, red = failure/loss, green = win/cached). Do not pick colors aesthetically.
- Be readable at 800px width on a blog page — minimum `fontSize: 22` for body labels, `fontSize: 32` for titles (raised from 18/28 to keep text sharp after blog-page downscale).
- Include a short caption text element inside the canvas (2nd line under the title) explaining what the reader should take away.

**Native Excalidraw fonts only (mandatory).** Every text element must use one of Excalidraw's built-in `fontFamily` values — never set custom font names, never rely on system fonts, never leave `fontFamily` unset (the export will fall back inconsistently across machines and the PNG will look "off-brand"). Allowed values:

| `fontFamily` | Name        | Use for                                                          |
| ------------ | ----------- | ---------------------------------------------------------------- |
| `1`          | Virgil      | **Default** for all titles, labels, captions, prose annotations. |
| `2`          | Helvetica   | Reserved — only for formal/technical figures explicitly framed as non-sketch (rare). Do not mix with Virgil in the same diagram. |
| `3`          | Cascadia    | Code, identifiers, file paths, shell commands, hex/numeric literals inside boxes labeled as code. |

Rules:
- **(a) Every `text` element in the scene payload must include `"fontFamily": 1` (or `3` for code), explicitly — never omit.** `author-scene.mjs` rejects any text element with a missing or non-`{1,3}` `fontFamily`, because omitting the field causes the renderer to fall back to a default that varies across versions/machines, and the resulting PNG looks visibly different from the rest of the post's figures.
- **(b) Font must be synchronized across every text element in the same diagram.** Pick one primary family per figure (default: Virgil = `1`) and apply it to *every* text element — title, captions, node labels, edge labels, annotations. The only allowed exception is code/identifier text, which uses Cascadia (`3`); these must be visually grouped (inside boxes labeled as code, or in a code panel), not sprinkled into prose labels.
- **(c) Same `fontSize` tier across peers.** All node labels in the same figure share one `fontSize` (e.g., 22). All edge labels share one (e.g., 20). Title is one (32). Caption is one (24). Do not vary sizes within a tier for emphasis — use color or stroke for emphasis instead.
- **(d) Do not switch families mid-label.** A single `text` element is one family, period.
- **(e) Cross-figure consistency.** All diagrams in the same post use the same family choices and the same size tiers. Do not author figure 1 in Virgil-22 and figure 2 in Virgil-26.
- **(f) Math-heavy / paper-reading posts**: still use Virgil for prose labels; render math as text with Cascadia, not Helvetica.

**Text must never overflow its container (mandatory).** If a `text` element is bound to a shape via `containerId`, or visually placed inside any rectangle/ellipse/diamond, the rendered glyphs must stay strictly inside the shape's borders — no character may cross the stroke, no descender may clip the bottom edge, no line may run past the right edge. Excalidraw does NOT auto-wrap or auto-shrink, so this is the author's responsibility:

1. **Compute required width before sizing the box.** For text of `N` characters at `fontSize F` (Virgil), required interior width ≈ `N × F × 0.6`; for Cascadia (mono), ≈ `N × F × 0.62`. Multi-line text: take `max(N_per_line)`. The container's `width` must be **`required_width + 2 × 24` (horizontal padding ≥ 24 px on each side)**, rounded up to the nearest 20 px.
2. **Compute required height.** `lineCount × F × 1.25 + 2 × 20` (vertical padding ≥ 20 px). Round up to the nearest 20 px.
3. **Wrap manually before sizing.** Cap any single line at ~24 chars (Virgil) / ~22 chars (Cascadia). Insert explicit `\n` in the text payload — Excalidraw respects newlines but never inserts them. Recompute height for the new line count.
4. **Bind every in-shape label via `containerId`** with `verticalAlign: "middle"`, `textAlign: "center"`. Free-floating text on top of shapes is forbidden; the shape's bounding box and the text's bounding box must be the same element-pair.
5. **Pre-render overflow audit.** `author-scene.mjs` performs this check on the JSON before render: for every text element it computes the required box from the formulas above and rejects the scene if `container.width < required_width` or `container.height < required_height`. When the validator fails, **fix the layout before re-running it** — enlarge the container, shrink the font (down to the floor: 22 body / 32 title), or break the label across more lines. Do not bypass the validator.
6. **Visual confirmation in Phase E.** When opening the exported PNG in the composition gate, scan every shape: text must be fully inside the stroke with visible padding on all four sides. Any clipped, overhanging, or edge-touching glyph fails the gate and forces a re-author.

**Scientific, content-faithful design (mandatory).** Every diagram is a figure in a technical article — treat it like one in a paper, not a marketing illustration:
- **One claim per figure.** Decide the single takeaway sentence the figure proves (e.g., "KV-cache writes dominate latency past sequence length 4k"). If you cannot state it in one sentence, the figure is doing too much — split it.
- **Bind to the prose.** Every node, edge label, and color must correspond to a noun, verb, or quantity that appears in the surrounding ±200 lines of markdown. If a viewer asks "what is this box?" and the answer is not in the article, delete the box.
- **Quantify when possible.** Prefer concrete numbers (latencies in µs/ms, sizes in MB/GB, rates in tok/s, percentages) over vague adjectives ("fast", "large"). Numbers must match the values cited in the prose — do not invent figures for visual balance.
- **Show structure, not decoration.** Information density matters: each pixel should encode either a component, a relationship, or a measurement. No purely decorative shapes, no logos, no clip-art icons, no "vibes" elements.
- **Caption discipline.** The 2nd-line caption under the title is the figure's thesis statement, not a label restatement. Bad: "KV-cache architecture diagram." Good: "Decode-time bandwidth grows linearly with batch × seq-len; prefill is bounded by FLOPs."

**Tight composition — no wasted whitespace.** Diagrams must use the canvas, not float in it. Loose layouts with large empty regions look amateurish, waste vertical space on the blog page, and make text relatively smaller after downscale.
- **Element bounding box must fill ≥ 70% of the authored 2400×1600 canvas** in both width and height. `author-scene.mjs` enforces this at validation time: take min/max of all element `x, y, x+width, y+height`; if the bounding box is smaller than 1680×1120, the layout is too sparse — enlarge shapes proportionally or move the camera in (i.e., re-author with everything scaled up to fill the canvas).
- **Outer margin ≤ 80 px on all sides** between the element bounding box and the canvas edge. The 60 px inter-element padding (set elsewhere) is for *between* elements, not for an extra border around the whole figure.
- **No empty quadrants.** If a 2×2 split of the bounding box leaves any quadrant >70% empty, rebalance — either move elements to fill it or shrink the canvas region. Title and caption count as content; one of them per quadrant is enough to keep it non-empty.
- **Aspect ratio matches content.** A 5-step pipeline is wide-and-short; a layered stack is tall-and-narrow. Do not force everything into 2400×1600 — use `exportPadding: 48` and let the bounding box dictate aspect, then verify the result is still ≥ 1600×900 px after export.

**Text containment & no-overlap mandate (mandatory — diagrams with overflowing or overlapping text fail Phase E).** Excalidraw does NOT auto-wrap or auto-shrink text inside shapes. If a label is wider than its container, it spills out and visually collides with neighbors; if two elements overlap in their bounding boxes, the text becomes unreadable. Enforce this before exporting:

1. **Size every container to its label, not the other way around.** For a text label of `N` characters at `fontSize F`, the rendered text occupies roughly:
   - **width ≈ `N × F × 0.6`** (Virgil/Cascadia at default tracking)
   - **height per line ≈ `F × 1.25`**
   Set the parent rectangle's `width` to **`textWidth + 2 × horizontalPadding`** with `horizontalPadding ≥ 24 px`, and `height` to **`(lineCount × lineHeight) + 2 × verticalPadding`** with `verticalPadding ≥ 20 px`. If the label is dynamic, round up — clipping is worse than empty space.
2. **Wrap long labels manually.** Never put more than ~28 characters on a single line inside a node. Break into 2–3 lines with explicit `\n` in the text payload (Excalidraw respects newlines but does not insert them). Recompute the container height for the new line count.
3. **Bind text to its shape using `containerId`.** When a label belongs *inside* a rectangle/ellipse, create the text element with `containerId` set to the shape's id and `verticalAlign: "middle"`, `textAlign: "center"`. Excalidraw will then center-clip the text to the box; combined with rule 1 (sizing the box to fit), this prevents overflow. Free-floating text elements (no `containerId`) must be placed in empty canvas regions, never on top of another shape.
4. **Minimum gap between sibling elements: 40 px on every side** (60 px preferred — see padding rule above). After laying out, check every pair of bounding boxes: if `|x1 - x2| < width/2 + 40` AND `|y1 - y2| < height/2 + 40`, they will visually crowd. Lay out rows/columns of nodes with even strides in the JSON itself (the validator's overlap check will reject any two siblings whose bounding boxes intersect).
5. **Arrow labels never sit on top of arrows or other text.** Place edge labels in the gap between source and target nodes, offset perpendicular to the arrow by ≥ 20 px. If two arrows run parallel, label only one or fan the labels out vertically.
6. **Pre-render overlap check.** `author-scene.mjs` walks every pair `(a, b)` of non-parent/child elements and computes axis-aligned bounding-box intersection. Any overlap (other than a `containerId`-bound text inside its parent) is a hard fail. When the validator rejects a scene for overlap, **fix the layout before re-running it** — move elements apart, shrink fonts to the floor (22 body / 32 title), or split the diagram into two figures. Do not bypass the validator.

**Sharpness mandate.** Excalidraw exports the PNG at roughly the bounding-box resolution of the elements on canvas — there is no DPI knob. The only way to get a crisp blog image is to author at high resolution. Concretely:
- **Canvas size: 2400 × 1600 logical units** (not 1200 × 800). Every element's `x`, `y`, `width`, `height` should scale up accordingly.
- **`strokeWidth: 2` minimum** for primary shape borders (the default 1 looks faint at high resolution).
- **`fontSize`: title 32, section labels 24, body labels 22, code/mono 18.** Smaller fonts may shrink to illegibility after blog-page downscale.
- **Padding ≥ 60 px** between elements — tight layouts compress badly when re-rendered.
- **Output PNG must be ≥ 1600 px wide and ≥ 80 KB**, verified by the Phase C sharpness gate. PNGs below that bar are blurry on Retina/4K screens and unacceptable.

**Arrow correctness mandate (mandatory).** Arrows carry the figure's *grammar* — they're the single most-misused element in technical diagrams. Every arrow must obey:
- **Direction matches semantics.** Data-flow arrows point in the direction data moves; control-flow arrows point from caller to callee; dependency arrows point from dependent to dependency. Re-read the prose section the figure illustrates and verify each arrowhead is on the correct end. A reversed arrow is a hard fail — readers will trust the picture over the text.
- **Arrowhead style encodes meaning, not aesthetic.** `endArrowhead: "arrow"` (filled triangle) for normal flow; `endArrowhead: "triangle_outline"` for "creates / produces"; `startArrowhead` set only for bidirectional sync (rare — pick one direction or split into two arrows). Never put an arrowhead on both ends just for symmetry.
- **Geometry: orthogonal first, diagonal only with reason.** Horizontal pipelines use horizontal arrows (height 0); layered stacks use vertical arrows (width 0); the only diagonals allowed are graph-type figures where Mermaid's dagre layout produced them. Any hand-authored diagonal must be load-bearing — a deliberate cross-cutting concern, an aside flowing back into the main path. Random diagonals read as sloppy.
- **No arrow crosses a node, label, or unrelated arrow.** If two arrows must cross, route one with an explicit jog (two-segment polyline via `points: [[0,0],[mid,0],[mid,Δy],[end,Δy]]`) so the crossing reads as intentional. If three+ arrows would cross, the figure is too dense — split it.
- **Label placement: perpendicular to the arrow, in the gap, not on it.** Edge labels sit ≥ 20 px off the arrow line, on the side that doesn't collide with adjacent nodes. For horizontal arrows place the label above the line; for vertical, to the right. Never let a label straddle the arrowhead — readers parse arrowhead first, label second.
- **Length encodes nothing; thickness can.** Don't make a long arrow mean "slow" or a short one "fast" — length is determined by node positions. If you want to mark a hot path, use `strokeWidth: 3` plus the primary blue palette on that arrow specifically; everything else stays at 2.

**Detail-density mandate (mandatory).** Figures must be detailed enough that a reader who skipped the prose can still extract the mechanism. Mere boxes-with-labels is "abstract decoration", not an explanation. Concretely:
- **At least 60 % of nodes carry a quantity, type, or qualifier**, not just a name. Bad: a box labeled `cache`. Good: `KV cache (16 KB / page)`. Bad: `decode`. Good: `decode (autoregressive, 1 token / step)`. Numbers, units, sizes, rates, and worst-case bounds belong on the figure, not just in the prose. The pre-render validator's information-density check counts unique tokens — push past the floor of 6 toward 12+ for a deep-dive figure.
- **Internal structure where it's the point.** A diagram of paged KV-cache must show pages-inside-the-cache, not a single grey box. A diagram of a multi-head attention block must show ≥ 2 heads, not "one box labeled MHA". Use nested rectangles, sub-grids, or a small inset to reveal the structure the prose claims to explain. The rule: if the figure's claim is about an internal mechanism, the figure must contain the mechanism.
- **Show the failure mode alongside the success.** Before/after figures get this right by construction; for non-comparison figures, mark the danger / caution / external-system parts with the matching `kind` so the reader sees what can go wrong. A figure that is uniformly green or uniformly blue is usually under-detailed.
- **Annotate the non-obvious.** If two boxes look similar but mean different things, add a short Cascadia (`fontFamily: 3`) annotation underneath spelling out the difference (e.g., `// shared across requests`, `// per-request`). If a quantity scales nonlinearly with another, write `O(n²)` or `~ b·s` next to the relevant edge.

**Layout precision mandate (mandatory — figures must look authored, not improvised).** The eye reads alignment as competence; misaligned shapes read as careless. Author every element on a 20-px grid and enforce visual symmetry:
- **20-px snap grid.** All `x`, `y`, `width`, `height` values must be multiples of 20. The DSL layout engines already round to 20 — when authoring a `raw` figure manually, snap by hand. Misaligned shapes (off by 3–7 px) make sibling rows look "wobbly" even when no individual shape is wrong.
- **Equal-stride rows / columns.** Nodes in a pipeline row share one stride; layers in a stack share one height; rows in a matrix share one row-height. Variable strides are reserved for figures whose claim *is* the variation (e.g., showing skewed bucket sizes).
- **Optical alignment for text.** When centering text inside a container of width `W`, set `x = container.x + (W - textWidth)/2`, then round to the nearest even pixel. The DSL engine handles this; for `raw` figures, compute `textWidth ≈ chars × fontSize × 0.6` and set the text element's `x` accordingly. Free-floating text labels that look "almost centered" are a smell.
- **Consistent stroke weights across peers.** All same-tier shapes share one `strokeWidth`. A mix of `1.5` / `2` / `2.5` within one figure (without a semantic reason) reads as noise. The accent kind already varies fill — don't also vary stroke.
- **No rotated text, no rotated shapes.** Excalidraw supports `angle`, but rotated text in technical figures looks like an infographic, not a paper. Keep `angle: 0` everywhere. If a label genuinely doesn't fit horizontally, shrink the font (down to the 22-pt floor) or break it across two lines — never rotate.
- **Margins are deliberate, not residual.** The bounding-box coverage gate enforces ≥ 70 % on the dominant axis and ≥ 40 % on the minor axis. When you are at the floor, *use the space* — make nodes wider, give labels two lines, add an inset showing internals. Empty canvas at the bottom of a figure is wasted vertical real estate on the rendered blog page.

**Intuitiveness mandate (mandatory — the figure should "click" in under 5 seconds).** A reader should grasp the figure's claim from the title + caption + general shape alone, before parsing any node label. Verify by self-test:
- **Squint test.** Mentally blur the figure (or actually shrink it to 25 % in your head). Can you still tell apart main path / bottleneck / outcome from color and position? If everything blurs into one mass, the palette assignment is wrong (too many neutrals, or main-path nodes scattered instead of in a row).
- **One reading direction.** Left-to-right for time/causality, top-to-bottom for layered abstraction or inheritance. Mixed reading directions (some left-to-right, some bottom-up) confuse the eye — pick one per figure and stick with it.
- **Visual hierarchy matches conceptual hierarchy.** The most important node is biggest, brightest, or in the visual center; supporting nodes are smaller and at the periphery. If the prose has a "main character", the figure should too.
- **No legend required.** A reader who has the title and caption should not need a legend explaining what each color means — the prose already calls primary blue the "main path", caution amber the "bottleneck", etc. If you find yourself wanting to add a legend, the figure is using too many kinds; reduce to ≤ 3 distinct accents per `assertOneAccent`.

### Phase D — Draft

1. Read 60–80 lines from the matching reference article (above) so voice stays calibrated.
2. Write the full markdown via `Write` to the resolved target path. Frontmatter exactly per contract above; today's date.
3. Embed each PNG immediately under the section heading it illustrates: `![alt](/imgs/blogs/<slug>-<n>.png)`.
4. Add cross-links inline using relative paths: `[KV cache](/blog/machine-learning/large-language-model/kv-cache)` (note: link path drops the `content/` prefix and `.md` extension, matching how Next.js routes the blog).

### Phase E — Verify (hard gates)

After writing, run these checks. If any gate fails, **expand and rewrite — do not ship a short post.**

1. **Word count gate.** `wc -w <path>`. Compute `readTime = round(words / 220)`.
   - Deep-dive: `readTime >= 50` (≥ 11,000 words). If under, identify the 2–3 thinnest sections and add: more case studies, deeper internals, more numbers, more code, more comparison tables. Re-run the gate.
   - Explainer: `readTime >= 25`. Paper-reading: `readTime >= 30`.
2. **Diagram gate.** Every planned diagram exists as a **sharp** PNG under `public/imgs/blogs/`, and is embedded in the markdown with `![alt](/imgs/blogs/<slug>-<n>.png)`. The first ("mental model") diagram must be referenced in the intro paragraph.

   **Abstraction-coverage sub-gate (mandatory): every abstract concept has a figure within 30 lines.** Scan the post for abstract-concept signals and verify each one has an embedded `![](/imgs/blogs/<slug>-<n>.png)` within 30 lines (downward) of the trigger. If any signal lacks a nearby figure, return to Phase B, add a DSL entry to the abstraction inventory for it, run it through Phase C, and embed the resulting PNG. Run:
   ```bash
   path="<post.md>"
   # 1) flag prose-only abstractions (any of these signals)
   grep -nE 'imagine|think of (it|this) as|consider (the|a) case|the way (this|it) works|under the hood|conceptually|in essence|abstract(ly|ion)|(\b(layer|tier|stage|phase|step)s?\b.*(\1\b|next|then))' "$path" \
     | while IFS=: read -r line _; do
         # check for an embedded figure within 30 lines below
         next30=$(awk -v L="$line" 'NR>=L && NR<=L+30' "$path")
         if ! grep -q '!\[[^]]*\](/imgs/blogs/' <<< "$next30"; then
           echo "MISSING FIGURE near line $line: $(awk -v L=$line 'NR==L' "$path" | cut -c1-80)"
         fi
       done
   # 2) flag math intuitions and architecture descriptions without a nearby figure
   grep -nE '\$\$|\\begin\{(equation|align)|architecture|pipeline|state machine|data structure|page table|kv.cache|cache layout|tradeoff|before.+after' "$path" \
     | while IFS=: read -r line _; do
         next30=$(awk -v L="$line" 'NR>=L && NR<=L+30' "$path")
         grep -q '!\[[^]]*\](/imgs/blogs/' <<< "$next30" \
           || echo "MISSING FIGURE for abstraction at line $line"
       done
   ```
   Any "MISSING FIGURE" line is a hard fail. The fix is *never* to delete the prose — always to add the figure. Also enforce the per-depth **minimum** (not a cap): explainer ≥ 4 figures, paper-reading ≥ 5, deep-dive ≥ 8. There is **no upper limit** — if a post has 25 things worth illustrating, ship 25 figures. Count with `grep -cE '^!\[' "$path"` (or count distinct PNG paths) and reject only if under the minimum; do not warn or "trim" when over.

   **Composition sub-gate (mandatory): tight, scientific, content-faithful.** Open each PNG with `Read` and verify:
   - **Font check**: every visible text element renders in Virgil (hand-drawn) or Cascadia (mono code) — no Helvetica/Arial/Times leakage. If any label looks like a generic system font, the `fontFamily` was omitted in the payload — re-author with explicit `"fontFamily": 1` (or `3`) on every text element.
   - **Density check**: the actual content fills the frame. If the figure has wide empty bands on any side, or the title/caption float far from the body, the bounding-box coverage is below the 70% floor — re-author tighter (scale elements up, reduce inter-element gaps to the 40–60 px floor, or shrink the export region).
   - **Faithfulness check (anchor-based, deterministic)**: every node in the DSL has an `anchor` field naming the markdown section it illustrates. For each node, slice the markdown to the ±200 lines around its anchor and `grep` for the node's `label`. Zero hits = the diagram diverged from the prose it claims to illustrate. Failing nodes must be fixed by either (a) editing the prose to mention the component, or (b) renaming the node label to match the prose. Bash sketch:
     ```bash
     # for each node {label, anchor} in <slug>-<i>.dsl.json:
     line=$(grep -nE "^#+ .*${anchor#\#}" "<post.md>" | head -1 | cut -d: -f1)
     awk -v L="$line" 'NR>=L-200 && NR<=L+200' "<post.md>" \
       | grep -Fq "$label" || echo "FAIL: $label not found near $anchor"
     ```
     Replaces the older "pick 3 random labels" spot-check — every node is checked, and the failure message names which node and which section it should have been in.
   - **Single-claim check**: the figure proves one specific sentence from the prose. If you cannot point to that sentence, the figure is decorative — replace it with one that earns its space.

   **Layout sub-gate (mandatory): no text overflow, no overlapping elements.** Open each exported PNG with `Read` and inspect visually. The diagram fails if any of: (a) text spills past the edge of its container rectangle/ellipse, (b) two non-parent/child elements have visibly overlapping bounding boxes, (c) an arrow line crosses through a text label, (d) labels are clipped at the canvas edge, (e) two text elements visually touch or run into each other. If any failure is observed, return to Phase C: re-size containers to fit their labels per the formula `width ≈ chars × fontSize × 0.6 + 48`, bind in-shape text via `containerId`, enforce ≥ 40 px gap between siblings, and re-export. Do not ship a diagram with known overflow or overlap — this is a hard gate.

   **Sharpness sub-gate (mandatory):** for every diagram PNG, verify resolution and size:
   ```bash
   for f in public/imgs/blogs/<slug>-*.png; do
     dims=$(sips -g pixelWidth -g pixelHeight "$f" 2>/dev/null | awk '/pixel/ {print $2}' | paste -sd' ' -)
     bytes=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
     echo "$f  $dims  ${bytes}B"
   done
   ```
   Required for every PNG: width ≥ 1600, height ≥ 900, size ≥ 80 KB. A diagram that fails any of these is blurry on Retina/4K displays and must be re-authored on the larger 2400×1600 canvas (see Phase C sharpness mandate). Do not ship blurry images — this is a hard gate, not a warning.

   Grep the markdown for forbidden text-based diagram substitutes — if any of these appear in the file the gate fails and the post must be reworked with real Excalidraw PNGs:
   ```bash
   # All of these should return zero matches in a finished post:
   grep -nE '^```text' <path>            # ASCII "diagrams" in fenced text blocks
   grep -nE '[│┌┐└┘├┤┬┴┼─]' <path>       # Unicode box-drawing
   grep -nE '\+--+\+|--->|<---' <path>   # ASCII art arrows/boxes
   ```
   Mermaid sources are fine when first converted to scene JSON via `@excalidraw/mermaid-to-excalidraw` and then rendered through `render-scene.mjs`. Inline ```mermaid``` source blocks left in the markdown are not.

   **Slug-match sub-gate (defensive sanity check on the markdown).** Every embedded image path in the post must start with this post's slug. Run:
   ```bash
   slug="<slug>"
   grep -oE '\!\[[^]]*\]\(/imgs/blogs/[^)]+\)' "<path>" \
     | grep -v "/imgs/blogs/${slug}-" \
     && echo "FAIL: foreign image embedded — wrong-slug typo or stale path" \
     || echo "ok: all images match slug"
   ```
   With the headless renderer there is no shared canvas, so a foreign-slug image now means a typo in the markdown or a stale reference, not a parallel-session collision — fix the `![alt](...)` path or re-author the right diagram under the correct slug and replace it.
3. **Frontmatter gate.** `readTime` matches the recomputed value; `date` is today; `aiGenerated: true`; tags 5–12; category is a real folder under `content/blog/`.
4. **Substance gate.** Every H2 has at least one of: comparison table, ≥15-line code block, measured benchmark with units, or worked numerical example. Spot-check by reading the file and counting.
5. **No-H1 gate.** `grep -nE '^# [^#]' <path>` must return zero matches. Any single-`#` heading in the body fails the gate — convert it to `##` and re-run.
6. **English-only gate.** The entire body (and frontmatter values like `title`, `description`, `tags`) must be English. Spot-check for non-ASCII letters that indicate other languages: `grep -nP '[\x{00C0}-\x{1EF9}\x{4E00}-\x{9FFF}]' <path>` should return zero matches outside of legitimate proper nouns or quoted technical terms. If any section was drafted in another language, rewrite it in English before shipping.

Then report to the user:
- Final file path, word count, recomputed `readTime`.
- List of new images written (paths and sizes).
- Which gates passed/failed and what was added on the second pass if applicable.
- 2–4 suggested cross-links the *user* should consider adding to *other* existing posts (do not edit those unless asked).
- Reminder to run `npm run dev` (or `bun run dev`) and load the page locally before committing.

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
| named open-source library deep-dives (lmcache, vllm, sglang, ray, deepspeed) | `machine-learning/open-source-library/`     |

If two folders match, or none match cleanly, use `AskUserQuestion` with the top 2–3 candidates as options. Never guess silently.

## Slug rules

- kebab-case, derived from the title.
- Drop stop-words (`a`, `the`, `of`, `for`, `with`, `to`, `and`, `or`).
- ≤ 60 chars. Trim trailing partial words.
- If `<target>/<slug>.md` already exists: ask the user to (a) overwrite, (b) append `-v2`, or (c) pick a new slug.

## Diagram style guide

Match the look of existing PNGs in `public/imgs/blogs/`. Defaults for every shape in the scene payload (the JSON the skill writes to `.cache/blog-writer/<slug>/<slug>-<i>.in.json`):

```json
{
  "strokeColor": "#1e1e1e",
  "backgroundColor": "transparent",
  "fillStyle": "hachure",
  "strokeWidth": 2,
  "strokeStyle": "solid",
  "roughness": 1,
  "roundness": { "type": 3 },
  "fontFamily": 1,
  "fontSize": 22
}
```

(Stroke width 2 and font size 22 are minimums for sharpness — see Phase C sharpness mandate. Bump to 2.5 / 24 for primary shapes if the canvas is dense.)

**Semantic color palette (mandatory — colors carry meaning, not decoration).** Pick colors based on what the element represents in the prose, never aesthetically. Use 2–3 colors per figure max; the rest stays transparent.

| Color | Hex | Use for |
| --- | --- | --- |
| Primary blue | `#a5d8ff` | The main happy-path component; the layer being explained; "what we use" |
| Caution amber | `#ffec99` | Trade-off points; bottlenecks; cost; "watch out here" |
| Danger red | `#ffc9c9` | Failure modes; eviction; loss; "this is what breaks" |
| Success green | `#b2f2bb` | Cache hits; wins; the optimized path in before/after diagrams |
| Neutral lavender | `#d0bfff` | External systems / third-party / off-host components |
| Soft gray | `#e9ecef` | Background grouping, "context" boxes, things the reader can ignore |

For a before/after diagram: left column uses red/amber accents (the bad path), right column uses green/blue (the good path). For a layered stack: blue for the layer being discussed in the section, gray for the layers above/below. Never two strong accents competing in the same figure — pick one to dominate.

Three reusable layouts (full payload templates live in `diagrams/`):

1. **Layered stack** (`diagrams/layered-stack.json`) — nested rectangles for "host / image / runtime / serving" or "data / model / serving / observability".
2. **Before / after comparison** (`diagrams/before-after.json`) — two columns separated by a vertical divider with a centered arrow; left column is the naive flow, right is the optimized one.
3. **Pipeline / flow** — author the Mermaid source, run it through `@excalidraw/mermaid-to-excalidraw` to get an element list, drop that into your scene payload, and feed it to the same `author-scene.mjs` → `render-scene.mjs` pipeline as any other figure.

Always:
- **2400×1600 logical canvas** (was 1200×800 — bumped for sharpness); export with `exportPadding: 48`.
- **`fontFamily: 1` (Virgil) for prose; `fontFamily: 3` (Cascadia) for code** — set explicitly on every `text` element, never omit. No system fonts, no custom names. (See "Native Excalidraw fonts only" mandate above.)
- Title text at the top in `fontSize: 32` (was 28). Section labels at 24, body at 22.
- `strokeWidth: 2` minimum; primary shapes 2.5.
- Label edges/arrows with one-line annotations, not paragraphs.
- One accent color per figure, never two.
- **Fill the canvas (≥ 70% bounding-box coverage in both axes)** — see "Tight composition" mandate. Loose, airy layouts fail the design gate.
- After export, run the Phase C sharpness gate: width ≥ 1600 px, height ≥ 900 px, file size ≥ 80 KB. Re-author or fall back to SVG → high-DPI rasterize if any check fails.

## Templates

The three outline skeletons live at `templates/{deep-dive,explainer,paper-reading}.md`. Pick by depth (Phase A) and copy the section list as the starting outline emitted in Phase B.
