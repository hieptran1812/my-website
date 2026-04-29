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
- 1–4 PNGs per post. The first one (the "mental model" image) is referenced in the intro paragraph with a sentence like "The diagram above is the mental model: …".
- Hand-drawn Excalidraw style (sloppiness 1, roughness 1, default Virgil/Cascadia fonts).
- Saved to `public/imgs/blogs/<slug>-<n>.png`. Embedded as `![alt text](/imgs/blogs/<slug>-<n>.png)` directly under the heading they illustrate.
- **Diagrams MUST be real Excalidraw PNGs. Never substitute ASCII art, ```text``` boxes, Unicode box-drawing, code-block "diagrams", or prose-bulleted "diagrams". These are not acceptable fallbacks under any circumstance.** If the Excalidraw MCP cannot be reached after the Phase C recovery procedure, **stop and surface the failure to the user** — do not ship the post. The only allowed non-Excalidraw figure type is an inline Mermaid block rendered through `mcp__excalidraw__create_from_mermaid` (which still produces a real PNG via Excalidraw). Posts without proper image diagrams are considered incomplete and fail the Phase E diagram gate.

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
   - Diagram plan (1–4 figures, what each shows, which section it sits in)
   - Case-study list (for deep-dives): 3–12 named incidents, each one line
   - Code snippet plan: language, what it demonstrates, ~lines
   - 2–4 cross-links to existing posts
4. **Print the outline and stop.** Wait for the user to say "go" / "approved" / "looks good", or to give edits. If they give edits, revise and re-emit. Do not proceed silently.

### Phase C — Diagrams (Excalidraw MCP)

**Connection bootstrap (do this FIRST, before any drawing call).** The Excalidraw MCP needs a browser frontend connected to the canvas server, otherwise `export_to_image` fails with `No frontend client connected`. Try to recover automatically before falling back:

1. Probe with `mcp__excalidraw__describe_scene` (cheap call). If it returns without a connection error, skip to step 5.
2. If it errors with "No frontend client connected", try to start/open the canvas:
   ```bash
   # Default canvas URL the MCP server exposes; check the user's MCP config if the port differs
   open -a "Google Chrome" "http://localhost:3333" 2>/dev/null \
     || open "http://localhost:3333" 2>/dev/null \
     || true
   ```
   Wait ~3 seconds, then re-probe with `describe_scene`. If still failing, check whether the MCP server itself is running:
   ```bash
   pgrep -fl excalidraw-mcp || echo "excalidraw-mcp server not running"
   ```
   If the server is not running, ask the user to start it (`claude mcp` lists configured servers; the canvas typically runs via `npx excalidraw-mcp` or similar).
3. After the user opens the canvas / starts the server, re-probe. Retry up to 3 times with 5-second waits.
4. If after 3 retries it still fails, **stop the workflow and ask the user to fix the canvas connection**. Do not ship the post without real diagrams. Acceptable user replies: (a) "I opened it, retry" → re-probe and continue, (b) "use Mermaid via create_from_mermaid" → still produces real PNG output through Excalidraw, acceptable. **Unacceptable**: skipping diagrams, using ASCII art, using ```text``` boxes, using Unicode box-drawing, or any prose-only substitute. Posts without proper Excalidraw PNGs are not allowed to ship.
5. Once connected, for each planned diagram (loop):
   1. `mcp__excalidraw__clear_canvas`
   2. `mcp__excalidraw__batch_create_elements` with the payload (see Diagram style guide). **Author on a high-resolution canvas: 2400×1600 logical units, not 1200×800.** This is the single most important factor in sharpness — the export resolution is proportional to the bounding box of your elements. A 1200-wide canvas exports to a ~1200-px PNG, which looks blurry on Retina/4K displays. A 2400-wide canvas exports to a ~2400-px PNG, which is crisp.
   3. Optionally `mcp__excalidraw__align_elements` / `distribute_elements` to clean up.
   4. `mcp__excalidraw__export_to_image` with `format: "png"`, `background: true`, `filePath: "public/imgs/blogs/<slug>-<n>.png"`.
   5. **Sharpness gate (mandatory).** Verify the PNG exists, is non-empty, AND meets resolution + size minimums:
      ```bash
      sips -g pixelWidth -g pixelHeight "<path>" 2>/dev/null \
        || identify -format "%w %h" "<path>"
      stat -f%z "<path>" 2>/dev/null || stat -c%s "<path>"   # bytes
      ```
      Required: **width ≥ 1600 px**, **height ≥ 900 px**, **size ≥ 80 KB**. If any of these fail, the diagram is blurry. Do NOT ship it. Recovery options, in order:
      1. Re-author on a 2400×1600 canvas with larger fonts and re-export.
      2. If still small, export with `format: "svg"` instead, then rasterize at 2× via `rsvg-convert -w 2400 in.svg -o out.png` or `magick -density 200 in.svg out.png` (whichever is available locally).
      3. If neither rasterizer is installed, ask the user to install `librsvg` (`brew install librsvg`) — do not silently ship a low-resolution PNG.

**Accuracy bar for diagrams.** A diagram that is decorative is a failure. Every diagram must:
- Reflect the exact terminology used in the article (variable names, component names, arrow labels match prose 1:1).
- Have arrows that point in the correct direction of data/control flow — verify by re-reading the prose section it illustrates.
- Use the accent palette below *semantically* (blue = main path, amber = caution/cost, red = failure/loss, green = win/cached). Do not pick colors aesthetically.
- Be readable at 800px width on a blog page — minimum `fontSize: 22` for body labels, `fontSize: 32` for titles (raised from 18/28 to keep text sharp after blog-page downscale).
- Include a short caption text element inside the canvas (2nd line under the title) explaining what the reader should take away.

**Sharpness mandate.** Excalidraw exports the PNG at roughly the bounding-box resolution of the elements on canvas — there is no DPI knob. The only way to get a crisp blog image is to author at high resolution. Concretely:
- **Canvas size: 2400 × 1600 logical units** (not 1200 × 800). Every element's `x`, `y`, `width`, `height` should scale up accordingly.
- **`strokeWidth: 2` minimum** for primary shape borders (the default 1 looks faint at high resolution).
- **`fontSize`: title 32, section labels 24, body labels 22, code/mono 18.** Smaller fonts may shrink to illegibility after blog-page downscale.
- **Padding ≥ 60 px** between elements — tight layouts compress badly when re-rendered.
- **Output PNG must be ≥ 1600 px wide and ≥ 80 KB**, verified by the Phase C sharpness gate. PNGs below that bar are blurry on Retina/4K screens and unacceptable.

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
2. **Diagram gate.** Every planned diagram exists as a **sharp** PNG under `public/imgs/blogs/`, and is embedded in the markdown with `![alt](/imgs/blogs/<slug>-<n>.png)`. The first ("mental model") diagram must be referenced in the intro paragraph. **Sharpness sub-gate (mandatory):** for every diagram PNG, verify resolution and size:
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
   Mermaid blocks rendered through `mcp__excalidraw__create_from_mermaid` are fine — they produce real PNGs. Inline ```mermaid``` source blocks left in the markdown are not.
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

Match the look of existing PNGs in `public/imgs/blogs/`. Defaults for every shape created via `batch_create_elements`:

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
3. **Pipeline / flow** — author as Mermaid and pass through `mcp__excalidraw__create_from_mermaid` for nodes-and-edges figures.

Always:
- **2400×1600 logical canvas** (was 1200×800 — bumped for sharpness); export with `exportPadding: 48`.
- Title text at the top in `fontSize: 32` (was 28). Section labels at 24, body at 22.
- `strokeWidth: 2` minimum; primary shapes 2.5.
- Label edges/arrows with one-line annotations, not paragraphs.
- One accent color per figure, never two.
- After export, run the Phase C sharpness gate: width ≥ 1600 px, height ≥ 900 px, file size ≥ 80 KB. Re-author or fall back to SVG → high-DPI rasterize if any check fails.

## Templates

The three outline skeletons live at `templates/{deep-dive,explainer,paper-reading}.md`. Pick by depth (Phase A) and copy the section list as the starting outline emitted in Phase B.
