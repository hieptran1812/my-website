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
- First-person plural (`we`) for shared reasoning. First-person singular (`I have personally debugged…`) only for war-stories.
- Open with the **real problem** or a sharp mismatch — never a dictionary definition.
- Build intuition with a concrete analogy (library, restaurant, city map) before introducing math.
- Math goes in `$...$` / `$$...$$`; symbols are defined the first time they appear.
- Code blocks must look runnable: real imports, real flags, real version numbers. Avoid pseudocode unless explicitly labeled.
- Use comparison tables for "naive vs optimized", "assumption vs reality", "strategy / when to use / trade-off".
- Reach for headings `##` for sections and `###` for sub-sections. Numbered top-level sections (`## 1. …`, `## 2. …`) are appropriate for deep-dives, not for explainers.
- Every deep-dive ends with a **case study** section: 3–12 numbered, named incidents (each ~150–250 words) followed by a closing **"When to reach for X / when not to"** section. Never a generic "Conclusion".
- Length targets: deep-dives 25–45 min read (≈ 6,000–11,000 words); explainers 8–15 min (≈ 2,000–4,000 words); paper readings 10–20 min.

Diagrams:
- 1–4 PNGs per post. The first one (the "mental model" image) is referenced in the intro paragraph with a sentence like "The diagram above is the mental model: …".
- Hand-drawn Excalidraw style (sloppiness 1, roughness 1, default Virgil/Cascadia fonts).
- Saved to `public/imgs/blogs/<slug>-<n>.png`. Embedded as `![alt text](/imgs/blogs/<slug>-<n>.png)` directly under the heading they illustrate.

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

For each planned diagram (loop):

1. `mcp__excalidraw__clear_canvas` (or `restore_snapshot` from a template).
2. `mcp__excalidraw__batch_create_elements` with the diagram payload — see **Diagram style guide** below for templates.
3. Optionally `mcp__excalidraw__align_elements` / `distribute_elements` to clean up.
4. `mcp__excalidraw__export_to_image` with `format: "png"`, `theme: "light"`, `exportBackground: true`, `exportPadding: 24`, `path: "public/imgs/blogs/<slug>-<n>.png"`.
5. Verify the file exists; if export failed, try `export_scene` → manual PNG, or fall back to a Mermaid figure rendered via `create_from_mermaid`.

If the Excalidraw MCP is unavailable (no `mcp__excalidraw__*` tools in the environment), tell the user and offer to either (a) skip diagrams, (b) emit ASCII art instead, or (c) wait for the MCP to be enabled.

### Phase D — Draft

1. Read 60–80 lines from the matching reference article (above) so voice stays calibrated.
2. Write the full markdown via `Write` to the resolved target path. Frontmatter exactly per contract above; today's date.
3. Embed each PNG immediately under the section heading it illustrates: `![alt](/imgs/blogs/<slug>-<n>.png)`.
4. Add cross-links inline using relative paths: `[KV cache](/blog/machine-learning/large-language-model/kv-cache)` (note: link path drops the `content/` prefix and `.md` extension, matching how Next.js routes the blog).

### Phase E — Verify

After writing, report to the user:
- Final file path and word count (`wc -w`).
- **Recomputed** `readTime` (words / 220, rounded). If it differs from the frontmatter, edit the frontmatter to match.
- List of new images written.
- 2–4 suggested cross-links the *user* should consider adding to *other* existing posts that point back to this one (do not edit those other posts unless asked).
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
  "strokeWidth": 1.5,
  "strokeStyle": "solid",
  "roughness": 1,
  "roundness": { "type": 3 },
  "fontFamily": 1,
  "fontSize": 20
}
```

Accent palette (use sparingly to highlight one or two elements per figure):
- Primary blue fill: `#a5d8ff`
- Caution amber fill: `#ffec99`
- Danger red fill: `#ffc9c9`
- Success green fill: `#b2f2bb`

Three reusable layouts (full payload templates live in `diagrams/`):

1. **Layered stack** (`diagrams/layered-stack.json`) — nested rectangles for "host / image / runtime / serving" or "data / model / serving / observability".
2. **Before / after comparison** (`diagrams/before-after.json`) — two columns separated by a vertical divider with a centered arrow; left column is the naive flow, right is the optimized one.
3. **Pipeline / flow** — author as Mermaid and pass through `mcp__excalidraw__create_from_mermaid` for nodes-and-edges figures.

Always:
- 1200×800 logical canvas; export with `exportPadding: 24`.
- Title text at the top in `fontSize: 28`.
- Label edges/arrows with one-line annotations, not paragraphs.
- One accent color per figure, never two.

## Templates

The three outline skeletons live at `templates/{deep-dive,explainer,paper-reading}.md`. Pick by depth (Phase A) and copy the section list as the starting outline emitted in Phase B.
