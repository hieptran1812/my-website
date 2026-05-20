# Diagram authoring reference

Read at the start of Phase C, before authoring the first figure. The validator (`scripts/author-scene.mjs`) enforces most rules below mechanically — when it rejects a scene, its error names the rule and the offending element. This file is the canonical reference for the *why* and the *math* behind those rules.

## Pipeline

Diagrams are rendered by `mcp_excalidraw/scripts/render-scene-batch.mjs` — a stateless headless puppeteer process. No shared MCP canvas state; parallel `/blog-writer` sessions cannot collide. Do **not** use `mcp__excalidraw__*` MCP tools in Phase C.

Per figure (run all N in parallel for one post; run as one batch via `render-scene-batch.mjs` to amortize browser startup):

1. Author element JSON → `.cache/blog-writer/<slug>/<slug>-<i>.in.json`
2. Validate + normalize: `node scripts/author-scene.mjs <in.json> <scene.json>`
3. Batch render: build manifest of all `*.scene.json` and pass to `render-scene-batch.mjs`
4. Sharpness gate: every PNG ≥ 1600×900 px, ≥ 80 KB

### One-time prerequisites

```bash
MCP_REPO=/Users/hieptran1812/Documents/mcp_excalidraw
test -f "$MCP_REPO/dist/frontend/headless.html" \
  || ( cd "$MCP_REPO" && npm install && npm run build:frontend )
test -d "$MCP_REPO/node_modules/puppeteer" \
  || ( cd "$MCP_REPO" && npm install )
```

### Batch render

```bash
slug="<slug>"
manifest=".cache/blog-writer/$slug/manifest.json"
ls .cache/blog-writer/$slug/*.scene.json \
  | jq -R -s 'split("\n") | map(select(length>0)) | map({in: ., out: ("public/imgs/blogs/" + (. | split("/") | last | sub(".scene.json"; ".png")))})' \
  > "$manifest"
node /Users/hieptran1812/Documents/mcp_excalidraw/scripts/render-scene-batch.mjs "$manifest"
```

Element-form scenes route to `__renderScene`; `{ mermaid, export }` scenes route to `__renderMermaid`. If render exits non-zero, **stop and surface to the user** — never substitute ASCII art, ```text``` boxes, Unicode box-drawing, or prose-only "diagrams".

## Element JSON shape

```json
{
  "title": "<figure title>",
  "_claim": "<≥ 8 word sentence the figure proves>",
  "_caption": "<one-sentence thesis under the title>",
  "elements": [
    { "type": "rectangle", "id": "node1", "x": 200, "y": 200, "width": 360, "height": 200,
      "backgroundColor": "#a5d8ff", "strokeWidth": 2, "label": "scheduler" },
    { "type": "arrow", "x": 560, "y": 300, "width": 200, "height": 0,
      "strokeWidth": 2, "endArrowhead": "arrow", "points": [[0,0],[200,0]] },
    { "type": "text", "id": "title", "x": 600, "y": 60, "width": 800, "height": 40,
      "text": "vLLM request lifecycle", "fontSize": 32, "fontFamily": 1 }
  ],
  "export": { "padding": 48, "minWidth": 1600, "minHeight": 900 }
}
```

The validator expands a shape's `label` field into a properly bound text element automatically. Don't hand-write free-floating text on top of shapes.

## Defaults

```json
{ "strokeColor": "#1e1e1e", "backgroundColor": "transparent", "fillStyle": "hachure",
  "strokeWidth": 2, "strokeStyle": "solid", "roughness": 1, "roundness": { "type": 3 },
  "fontFamily": 1, "fontSize": 22 }
```

## Fonts (mandatory)

| `fontFamily` | Name     | Use for |
|---|---|---|
| `1` | Virgil    | **Default** — titles, labels, captions, prose annotations |
| `3` | Cascadia  | Code, identifiers, paths, shell commands, hex/numeric literals |

Rules:
- Every `text` element must include explicit `fontFamily` (`1` or `3`). Omitting it makes the renderer fall back inconsistently across machines.
- One primary family per figure. Cascadia only for code/identifier text, visually grouped (not sprinkled into prose labels).
- Same `fontSize` tier across peers: title 32, section labels 24, body 22, edge labels 20. Don't vary sizes within a tier — use color/stroke for emphasis.
- Cross-figure consistency: all diagrams in the same post share family + size tiers.

## Semantic palette (mandatory)

Pick by meaning, never aesthetically. Max 2–3 colors per figure; rest stays transparent.

| Color | Hex | Meaning |
|---|---|---|
| Primary blue | `#a5d8ff` | Main happy-path; layer being explained; "what we use" |
| Caution amber | `#ffec99` | Tradeoffs, bottlenecks, cost |
| Danger red | `#ffc9c9` | Failure modes, eviction, loss |
| Success green | `#b2f2bb` | Cache hits, wins, optimized path |
| Neutral lavender | `#d0bfff` | External / third-party / off-host |
| Soft gray | `#e9ecef` | Background grouping, "context" boxes |

Before/after: left = red/amber, right = green/blue. Layered stack: blue for the layer being discussed, gray for above/below. Never two strong accents competing.

## Sizing math (containers fit content, not canvas)

For text of `N` chars at fontSize `F`:
- **Width** ≈ `N × F × 0.6` (Virgil) or `N × F × 0.62` (Cascadia)
- **Height per line** ≈ `F × 1.25`

Container size:
- `width = ceil((textWidth + 48) / 20) × 20` (≥ 24 px horizontal padding each side)
- `height = ceil((lineCount × F × 1.25 + 40) / 20) × 20` (≥ 20 px vertical padding each side)

**Cards are sized to fit content, not stretched to fill canvas.** A 5-line label at fontSize 22 is `5 × 22 × 1.25 + 60 ≈ 200` px tall, not 1100. To fill vertical space, give cards a uniform tallest-content height or add more rows — never stretch a single card.

**Free-floating text width must equal rendered width.** For a 7-char label at fontSize 22: `width = ceil(7 × 22 × 0.6 / 20) × 20 = 100`. Excalidraw renders by actual character widths; declared bbox short of rendered width causes overflow into adjacent nodes. The validator catches this with "declared width N below rendered M".

**Wrap manually.** Cap any single in-shape line at ~24 chars (Virgil) / ~22 (Cascadia). Insert explicit `\n` — Excalidraw never inserts newlines.

## Layout (mandatory exact y-coordinates)

- Title text element (`id: "title"`, fontSize 32, Virgil): **`y: 60`**
- Caption text matching `_caption` (fontSize 28 or 24, Virgil): **`y: 120`**
- First body row: **`y ∈ [180, 220]`** (max 60 px gap below caption)
- Body bounding-box left edge within ~140 px of canvas left

A body that starts at y=300+ creates a dead band — layout bug, not stylistic choice.

### 20-px snap grid

All `x`, `y`, `width`, `height` are multiples of 20. Misaligned shapes (off by 3–7 px) make rows look "wobbly".

### Equal-stride rows / columns

Pipeline rows share one stride; stack layers share one height; matrix rows share one row-height. Variable strides only when the figure's claim *is* the variation.

### Canvas & coverage

- **Canvas: 2400×1600 logical units**, `exportPadding: 48`
- **Element bounding box ≥ 70% canvas width AND height** (≥ 1680×1120). The validator rejects sparser layouts.
- **Outer margin ≤ 80 px** between bbox and canvas edge
- **No empty quadrants**: 2×2 split — no quadrant >70% empty
- **Aspect ratio matches content**: pipelines wide-and-short, stacks tall-and-narrow

### No-overlap

Min gap **40 px** between sibling bounding boxes (60 px preferred). Validator checks every non-parent/child pair for axis-aligned bbox intersection.

## Arrows (mandatory)

Arrow accuracy is non-negotiable: a diagram with a reversed, dangling, or mis-anchored arrow is *worse* than no diagram, because it teaches the reader the wrong causal chain. Treat every arrow as a load-bearing claim.

### Anchoring (endpoints must bind to shapes)

- **Bind both endpoints with `startBinding` / `endBinding`** referencing the source and target node `id`s. Free-floating arrows (no bindings) drift on re-layout and are a hard fail unless the arrow's tail or head intentionally points at empty space (e.g., "data leaves the system").
  ```json
  { "type": "arrow", "id": "e1", "x": 560, "y": 300,
    "width": 200, "height": 0, "points": [[0,0],[200,0]],
    "startBinding": { "elementId": "node1", "focus": 0, "gap": 8 },
    "endBinding":   { "elementId": "node2", "focus": 0, "gap": 8 },
    "endArrowhead": "arrow" }
  ```
- **Endpoint coordinates must land on the source/target edge,** not inside the bbox and not past it. For a horizontal arrow from `node1` (right edge `x = node1.x + node1.width`) to `node2` (left edge `x = node2.x`), set `arrow.x = node1.x + node1.width + gap` and `arrow.x + dx = node2.x - gap` with `gap ≥ 8`. The validator flags arrows whose head/tail falls inside any sibling bbox.
- **`gap` ≥ 8 px** so the arrowhead doesn't visually merge into the target stroke. Increase to 12 px when target `strokeWidth ≥ 3`.
- **One arrow = one edge.** Don't fan a single arrow into multiple targets by drawing extra `points` — author one arrow per (source, target) pair.

### Direction & semantics

- **Direction matches the prose.** Re-read the surrounding ±200 lines; verify each arrowhead points the way causality flows. Reversed arrows are a hard fail.
- **Arrowhead style encodes meaning** (be consistent across the post):
  - `endArrowhead: "arrow"` — normal flow / call / data movement
  - `endArrowhead: "triangle"` — synchronous request, blocking
  - `endArrowhead: "triangle_outline"` — creates / produces / instantiates
  - `endArrowhead: "bar"` — terminates, blocked, hard stop
  - `endArrowhead: "dot"` — subscribe / observe (non-consuming)
  - `startArrowhead` set **only** for true bidirectional sync (RPC round-trip, handshake). Otherwise leave it `null`.
- **Stroke style encodes channel:** `strokeStyle: "solid"` = real call/data; `"dashed"` = async / deferred / optional; `"dotted"` = control-plane / metadata only. Pick once per figure and stick to it.

### Geometry (orthogonal-first)

- **Orthogonal by default.** Horizontal pipelines → arrows with `height: 0` and `points: [[0,0],[dx,0]]`. Vertical stacks → `width: 0` and `points: [[0,0],[0,dy]]`. A diagonal arrow is a *claim* that the relationship is non-axial — only use one when that's true.
- **Right-angle jogs use exactly 3 segments**, not curves: `points: [[0,0],[mid,0],[mid,Δy],[end,Δy]]`. Pick `mid` so the vertical leg sits in empty space (≥ 40 px from any sibling bbox), not grazing a node.
- **No arrow crosses a node, label, or unrelated arrow.** If two arrows must cross, jog the *less important* one (lower stroke weight). Three+ crossings in a figure = split the figure.
- **Equal-stride arrows** in a row/column share the same length and y (or x). Mismatched strides imply mismatched semantics; the reader will look for a difference that isn't there.

### Labels on arrows

- **Edge labels live in the gap, not on the line.** For horizontal arrows, place the label above the arrow at `y = arrow.y - fontSize - 12`; for vertical arrows, place it to the right at `x = arrow.x + 16`. Center it on the arrow midpoint.
- **Label width must fit the gap.** If the gap between source and target is `G` px, the label's rendered width (using the sizing math above) must be `≤ G - 24`. Otherwise shorten the label or widen the gap — never let a label cross a node bbox or a parallel arrow.
- **One label per arrow, ≤ 4 words or one quantity** (e.g., `~12 µs`, `1 token/step`, `evict LRU`). Longer commentary belongs in a Cascadia annotation near the source node, not on the edge.
- **The validator now enforces this geometrically.** `author-scene.mjs` runs a segment-vs-bbox test (`segIntersectsRect`) of every free-floating text element against every arrow/line polyline segment. A label whose bbox (inset 4 px) crosses any arrow stroke is a hard fail: `text element … is drawn over arrow …`. The DSL layout engines already place labels clear of the stroke — above the source-side horizontal run for `graph`, above the arrow for `pipeline`, perpendicular-offset for `grid` — so this only bites hand-authored `type: "raw"` figures. When it fires, move the label into the gap; don't widen the bbox to dodge it.

### Emphasis

- **Length encodes nothing; thickness and color do.** Hot path = `strokeWidth: 3` + primary blue (`#a5d8ff` stroke is illegal — keep stroke `#1e1e1e` and tint the *source/target nodes* blue). Failure path = `strokeWidth: 2` + `strokeStyle: "dashed"` + danger-red node tints.
- **At most one "hot" arrow per figure.** If everything is bold, nothing is.

## Detail density (mandatory)

- **≥ 60% of nodes carry a quantity, type, or qualifier** — not just a name. Bad: `cache`. Good: `KV cache (16 KB / page)`. Bad: `decode`. Good: `decode (autoregressive, 1 token / step)`.
- **Internal structure where it's the point.** Paged KV-cache shows pages-inside-the-cache, not one grey box. Multi-head attention shows ≥ 2 heads.
- **Show the failure mode alongside success.** Mark danger / caution / external parts with the matching palette. Uniformly green or blue figures are usually under-detailed.
- **Annotate non-obvious distinctions** with short Cascadia annotations: `// shared across requests`, `// per-request`, `O(n²)`, `~ b·s`.
- Validator's information-density check: ≥ 6 unique non-stopword tokens; push toward 12+ for deep-dive figures.

## Sharpness (mandatory floor)

- Canvas 2400×1600 logical units (not 1200×800)
- `strokeWidth: 2` minimum (2.5 for primary)
- `fontSize`: title 32, section 24, body 22, code 18
- Padding ≥ 60 px between elements
- **PNG output ≥ 1600 px wide, ≥ 900 px tall, ≥ 80 KB** (Phase C sharpness gate)

A failing sharpness gate usually means under-density: too few elements, sparse layout, all-neutral fills. Add information density rather than scaling up empty space.

## Intuitiveness ("clicks" in <5s)

- **Squint test**: at 25%, can you still tell main path / bottleneck / outcome from color and position?
- **One reading direction per figure**: left-to-right for time/causality, top-to-bottom for layered abstraction. Don't mix.
- **Visual hierarchy matches conceptual hierarchy**: most important node = biggest, brightest, most central.
- **No legend required.** If you want to add one, the figure uses too many kinds — reduce to ≤ 3 distinct accents.

## Scientific, content-faithful design

- **One claim per figure.** State it in one sentence. If you can't, split it.
- **Every node, edge, color corresponds to a noun/verb/quantity in the surrounding ±200 lines of markdown.** If a viewer asks "what is this box?" and the answer isn't in the article, delete the box.
- **Quantify** with concrete numbers (µs/ms, MB/GB, tok/s, %). Numbers must match the prose — don't invent for visual balance.
- **No purely decorative shapes, logos, or clip-art.**
- **Caption is a thesis, not a label restatement.** Bad: "KV-cache architecture diagram." Good: "Decode-time bandwidth grows linearly with batch × seq-len; prefill is bounded by FLOPs."

## DSL shortcut (opt-in only)

For canned shapes (pipeline, stack, before-after, matrix, tree, timeline, grid, graph), you may write a DSL JSON and convert with:

> **`graph` engine note (post-2026-05 update):** the `graph` type used to emit Mermaid; it now produces deterministic element-form layered DAG layouts with a real title and caption rendered at the top. Nodes in a layer share one uniform size so edges align on a clean grid. Arrows are bound to source/target IDs and routed orthogonally: adjacent-layer edges jog at a shared channel x (the inter-layer gap midpoint) so fan-out/fan-in reads as one tidy bus; multi-layer-jump and back-edges detour through channels below the node block. The figure is then scaled to fill the canvas and top-aligned under the caption (no dead band). The validator no longer flags arrow ↔ shape bbox overlap, since arrows are routes and their correctness is enforced by binding fidelity and visual review.
>
> **Crossing reduction & spacing (post-2026-05 update):** within each layer, nodes are reordered by a barycenter sweep (down, up, down) that pulls every node toward the mean position of its neighbours in the adjacent layer — this minimises edge crossings so the DAG reads as a scientific layered graph, not a tangle. The vertical gap between sibling nodes is adaptive: a sparse, wide graph (few branches, many layers) spreads its busiest layer to fill ~82% of the body band instead of floating as a thin ribbon. Edge labels are placed above the *source-side* horizontal run of each arrow (never on the vertical routing channel), so they are provably clear of every stroke.
>
> **A linear flow is not a graph.** If every layer has at most one node, the `graph` engine rejects the DSL (`graph has at most one node per layer …`) — that figure is a `pipeline`, which serpentines into rows and fills the canvas. Use `graph` only when the flow actually branches or merges.
>
> **Author `graph` DSLs as true DAGs.** A cyclic edge set (A→B→A) cannot layer cleanly in any engine and renders as a tangle of detour arrows. Model the *flow* forward: e.g. for a request/response cycle, route the response to a distinct "result" node rather than back to the origin.


```bash
node .claude/skills/blog-writer/scripts/layout-scene.mjs <dsl.json> <scene.json>
```

The moment a figure needs something irregular (annotated photo, hybrid layout, asymmetric flow, math intuition, custom matrix with merged cells), drop the DSL and author elements directly. Element-level reference templates: `diagrams/{layered-stack,before-after}.elements.json`.

## What the validator enforces

Read its error messages — they name the rule and the offending element. Don't bypass.

- Containment (text inside container, with padding)
- Sibling bbox no-overlap
- Bounding-box coverage (≥ 70% dominant axis, ≥ 40% minor)
- `fontFamily ∈ {1, 3}` on every text element
- Palette compliance (only the 6 hex values)
- `_claim` ≥ 8 words; `_caption` present and non-restating
- Information density ≥ 6 unique non-stopword tokens
- Free-floating text declared width ≥ rendered width
- 20-px snap grid
- Arrow endpoints bind to a node `id` (or are explicitly marked `"unbound": true` for "leaves the system" arrows)
- Arrow head/tail coordinates fall in the `gap` band outside source/target bbox, not inside any sibling bbox
- Arrow `endArrowhead` set; `startArrowhead` only when `"bidirectional": true` is declared
- Orthogonal arrows have `height: 0` (horizontal) or `width: 0` (vertical); diagonals require `"diagonal": true` opt-in
- Edge label rendered width ≤ source-target gap − 24 px, and label bbox doesn't intersect any node bbox
