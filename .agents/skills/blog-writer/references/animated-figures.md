# Animated figures — inline SVG + CSS keyframes

Some concepts only land in **motion**: a process unfolding step by step, memory
compacting as survivors slide forward and holes drift to the tail, tokens flowing
through layers, a window sweeping across a series. The NVIDIA *KV-cache compression*
post animates exactly these — a step-by-step geometric mechanism on real Q/K data,
and two memory-packing strategies running "the same round of decoding." Static
Excalidraw still can't show the *change over time*. This file is how this blog does.

Read this at Phase C **only if** the Phase B plan marked one or more abstractions
`animated`. Animated figures are seasoning — **1–3 per post**, never the default.

---

## When to animate (and when not to)

Animate **only when motion carries the meaning**. If the still already says
everything, ship a static WebP — that is the workhorse and stays the majority of
every post's figures.

**Motion-worthy** (mark these `animated` in the Phase B inventory):

- A **process unfolds in steps / phases** — an algorithm iterating, a pipeline
  executing stage by stage, a geometric mechanism built up incrementally.
- A **before → after state transition** — memory compaction/eviction, cache
  fill-then-evict, rebalancing, a buffer draining, a tree rotating.
- Something **flows along a path** — tokens through layers, a request through a
  queue, a packet across a network, gradients backward.
- A **quantity sweeps / grows** — a progress bar, a sliding window, attention
  accumulating, a clip region filling.
- **Two strategies run "the same round"** side by side or A↔B — so the reader
  watches them diverge.

**Keep it static** (plain WebP): taxonomies, architecture block diagrams,
comparison tables, equations, decision trees, anything where the snapshot is the
whole point. Animating these is noise.

---

## The one technique this site supports

**Inline `<svg>` with an embedded `<style>` block using CSS `@keyframes`**, wrapped
in a `<figure class="blog-anim">` raw-HTML block in the markdown. Nothing else.
Here is *why* the alternatives are out — so you don't reach for them:

- The blog renders markdown → HTML with `allowDangerousHtml: true` and **no
  sanitizer**, then injects it via `dangerouslySetInnerHTML`. Raw inline SVG passes
  through **verbatim**. (Verified against the real `getArticle.ts` pipeline.)
- **No JavaScript.** `<script>` inserted via `innerHTML` does **not execute** in
  any browser. So JS-driven animation, Lottie, canvas, requestAnimationFrame — all
  dead on arrival. Animation must be **declarative**: CSS or SMIL.
- **Inline, not `<img src="*.svg">`.** An inline SVG lives in the page DOM, so it
  is **theme-aware** (reads the site's CSS variables) and is **not** subject to the
  strict image CSP. An SVG referenced by `<img>` goes through next/image's
  `script-src 'none'; sandbox` policy and its animations may be frozen. Inline
  avoids all of it — and needs no render/WebP/manifest step.
- **CSS `@keyframes`, not SMIL.** Better support, themeable, trivially honors
  `prefers-reduced-motion`, and easier to author/review. (SMIL `<animate>` works too
  and the validator allows it, but prefer CSS.)

---

## The hard rules (the validator + Phase E gate enforce these)

1. **No blank lines inside the `<figure>…</figure>` block.** CommonMark ends a
   raw-HTML block at the **first blank line** and reparses the rest as markdown —
   your SVG shatters into escaped text. Newlines are fine; *blank* lines are fatal.
   Keep the whole figure one contiguous block.
2. **`<figure class="blog-anim"` starts at column 0.** The block-start tag must be
   at the line start (CommonMark allows ≤3 spaces; use 0). No leading indentation.
3. **No `<script>`, no `on*=` handlers, no `javascript:`, no remote `href`/`src`.**
   Declarative only. (It wouldn't run anyway, but it's a hard fail.)
4. **Honor `prefers-reduced-motion: reduce`.** Include a media query in `<style>`
   that freezes motion — `animation: none` (or `animation-play-state: paused`) on
   the animated classes. Vestibular accessibility, and a gate.
5. **Accessible.** The `<svg>` carries `role="img"` and `aria-label="…"` (or a
   `<title>` child), plus a `<figcaption>` stating what the motion shows.
6. **Responsive sizing.** Use a `viewBox` plus
   `style="width:100%;height:auto;max-width:NNNpx"`. **No** large fixed
   `width="…"`/`height="…"` pixel attributes — the content column is ~896px and
   overflow is clipped.
7. **It must actually animate** — at least one `@keyframes` *and* an `animation:`
   referencing it (or a `<animate>` element).
8. **Calm motion.** 6–20 s loop, `infinite`, eased timing, `alternate` or a clean
   reset. Never flash more than ~3×/second. ≤ 3 accent colors (same squint
   discipline as static figures).

---

## Theming — match light/dark

An inline SVG reads the page's CSS variables. Use them so the figure flips with the
site theme, and always give a literal fallback:

| Token                   | Use for                          |
| ----------------------- | -------------------------------- |
| `var(--accent, #6366f1)`| the one thing the eye follows    |
| `var(--text-primary, #1f2937)` | labels, strong strokes    |
| `var(--text-secondary, #6b7280)`| secondary labels, axes   |
| `var(--surface, #f3f4f6)`| block / card fills               |
| `var(--border, #d1d5db)`| outlines, grid lines             |
| `var(--background, #fff)`| knockouts                        |

Keep it to one accent + neutrals. The moving element gets the accent; everything
else is neutral so the motion reads at a glance.

---

## Motion patterns (adapt these skeletons)

Each is a self-contained `<figure>` block — multi-line but **zero blank lines**.
Copy, rename the classes per figure (`a1-…`) so two figures on one page never share
keyframe names, retarget to your real data, and revalidate. Skeletons are minimal;
real figures carry labels, a caption row, and 4–12 elements.

### A. Step sequence with a sweeping highlight (geometric mechanism, algorithm steps)

A row of stages; a highlight sweeps across to "narrate" each step; stage labels fade
in on a staggered delay. Mirrors NVIDIA's step-by-step mechanism.

```html
<figure class="blog-anim">
<svg viewBox="0 0 640 200" role="img" aria-label="Four-stage mechanism; a highlight sweeps each stage in turn" style="width:100%;height:auto;max-width:760px">
<style>
.a1-stage{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a1-sweep{fill:var(--accent,#6366f1);opacity:.18}
@keyframes a1-move{0%{transform:translateX(0)}100%{transform:translateX(480px)}}
.a1-anim{animation:a1-move 8s steps(1,end) infinite}
@media (prefers-reduced-motion:reduce){.a1-anim{animation:none}}
</style>
<rect class="a1-stage" x="20"  y="70" width="140" height="60" rx="8"/>
<rect class="a1-stage" x="180" y="70" width="140" height="60" rx="8"/>
<rect class="a1-stage" x="340" y="70" width="140" height="60" rx="8"/>
<rect class="a1-stage" x="500" y="70" width="120" height="60" rx="8"/>
<rect class="a1-sweep a1-anim" x="20" y="70" width="140" height="60" rx="8"/>
<text class="a1-lbl" x="90"  y="105">project</text>
<text class="a1-lbl" x="250" y="105">rotate</text>
<text class="a1-lbl" x="410" y="105">score</text>
<text class="a1-lbl" x="560" y="105">select</text>
</svg>
<figcaption>The mechanism walks one stage at a time; the highlight marks the active step.</figcaption>
</figure>
```

### B. Slide-and-compact / eviction (memory packing)

Token squares; survivors translate left to close gaps; an evicted slot fades to a
ghost. Mirrors NVIDIA's "survivors slide forward, holes drift to the tail."

```html
<figure class="blog-anim">
<svg viewBox="0 0 560 140" role="img" aria-label="Cache compaction: an evicted slot empties and survivors slide left to close the gap" style="width:100%;height:auto;max-width:680px">
<style>
.a2-tok{fill:var(--accent,#6366f1);rx:6}
.a2-lbl{font:600 14px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
@keyframes a2-evict{0%,15%{opacity:1}35%,100%{opacity:.12}}
@keyframes a2-slideL{0%,40%{transform:translateX(0)}65%,100%{transform:translateX(-80px)}}
.a2-gone{animation:a2-evict 6s ease-in-out infinite}
.a2-mv{animation:a2-slideL 6s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a2-gone,.a2-mv{animation:none}}
</style>
<rect class="a2-tok" x="20"  y="40" width="64" height="64"/>
<rect class="a2-tok a2-gone" x="100" y="40" width="64" height="64"/>
<rect class="a2-tok a2-mv" x="180" y="40" width="64" height="64"/>
<rect class="a2-tok a2-mv" x="260" y="40" width="64" height="64"/>
<text class="a2-lbl" x="52"  y="78">k0</text>
<text class="a2-lbl" x="212" y="78">k2</text>
<text class="a2-lbl" x="292" y="78">k3</text>
</svg>
<figcaption>Evict k1; k2 and k3 slide left so live tokens stay contiguous.</figcaption>
</figure>
```

### C. Flow along a path (tokens through layers, request through a queue)

A packet travels between waypoints; staggered copies make a stream. Use CSS
`offset-path` (well supported) or translate keyframes between waypoints.

```html
<figure class="blog-anim">
<svg viewBox="0 0 600 160" role="img" aria-label="Tokens stream left to right through three layers" style="width:100%;height:auto;max-width:720px">
<style>
.a3-layer{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a3-dot{fill:var(--accent,#6366f1)}
@keyframes a3-flow{0%{transform:translateX(0);opacity:0}10%{opacity:1}90%{opacity:1}100%{transform:translateX(520px);opacity:0}}
.a3-p{animation:a3-flow 5s linear infinite}
.a3-p2{animation-delay:1.6s}
.a3-p3{animation-delay:3.2s}
@media (prefers-reduced-motion:reduce){.a3-p{animation:none;opacity:1}}
</style>
<rect class="a3-layer" x="40"  y="40" width="90" height="80" rx="8"/>
<rect class="a3-layer" x="255" y="40" width="90" height="80" rx="8"/>
<rect class="a3-layer" x="470" y="40" width="90" height="80" rx="8"/>
<circle class="a3-dot a3-p"  cx="40" cy="80" r="9"/>
<circle class="a3-dot a3-p a3-p2" cx="40" cy="80" r="9"/>
<circle class="a3-dot a3-p a3-p3" cx="40" cy="80" r="9"/>
</svg>
<figcaption>A token stream advances through the layer stack, one hop at a time.</figcaption>
</figure>
```

### D. Sweep / fill / grow (progress, sliding window, attention building)

A clip region or rect grows 0 → full, or a window rectangle slides across a series.

```html
<figure class="blog-anim">
<svg viewBox="0 0 600 120" role="img" aria-label="A sliding window sweeps across a sequence of tokens" style="width:100%;height:auto;max-width:720px">
<style>
.a4-cell{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a4-win{fill:none;stroke:var(--accent,#6366f1);stroke-width:3}
@keyframes a4-slide{0%{transform:translateX(0)}100%{transform:translateX(420px)}}
.a4-mv{animation:a4-slide 7s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.a4-mv{animation:none}}
</style>
<g>
<rect class="a4-cell" x="40"  y="45" width="44" height="44" rx="6"/>
<rect class="a4-cell" x="100" y="45" width="44" height="44" rx="6"/>
<rect class="a4-cell" x="160" y="45" width="44" height="44" rx="6"/>
<rect class="a4-cell" x="220" y="45" width="44" height="44" rx="6"/>
<rect class="a4-cell" x="280" y="45" width="44" height="44" rx="6"/>
<rect class="a4-cell" x="340" y="45" width="44" height="44" rx="6"/>
<rect class="a4-cell" x="400" y="45" width="44" height="44" rx="6"/>
<rect class="a4-cell" x="460" y="45" width="44" height="44" rx="6"/>
<rect class="a4-cell" x="520" y="45" width="44" height="44" rx="6"/>
</g>
<rect class="a4-win a4-mv" x="36" y="41" width="112" height="52" rx="8"/>
</svg>
<figcaption>The attention window slides along the sequence; only tokens inside it are visible to the step.</figcaption>
</figure>
```

### E. Before ↔ after crossfade (A vs B, two strategies)

Two groups cross-fade on the same loop: the reader sees state A, then B, then back.

```html
<figure class="blog-anim">
<svg viewBox="0 0 560 180" role="img" aria-label="Layout alternates between contiguous packing and fragmented free list" style="width:100%;height:auto;max-width:680px">
<style>
.a5-b{fill:var(--accent,#6366f1)} .a5-g{fill:var(--border,#d1d5db)}
.a5-cap{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes a5-fadeA{0%,40%{opacity:1}55%,95%{opacity:0}100%{opacity:1}}
@keyframes a5-fadeB{0%,40%{opacity:0}55%,95%{opacity:1}100%{opacity:0}}
.a5-A{animation:a5-fadeA 8s ease-in-out infinite}
.a5-B{animation:a5-fadeB 8s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a5-A{animation:none;opacity:1}.a5-B{animation:none;opacity:0}}
</style>
<g class="a5-A">
<rect class="a5-b" x="40" y="40" width="48" height="48" rx="6"/>
<rect class="a5-b" x="96" y="40" width="48" height="48" rx="6"/>
<rect class="a5-b" x="152" y="40" width="48" height="48" rx="6"/>
<text class="a5-cap" x="160" y="120">contiguous</text>
</g>
<g class="a5-B">
<rect class="a5-b" x="40"  y="40" width="48" height="48" rx="6"/>
<rect class="a5-g" x="96"  y="40" width="48" height="48" rx="6"/>
<rect class="a5-b" x="152" y="40" width="48" height="48" rx="6"/>
<text class="a5-cap" x="160" y="120">fragmented</text>
</g>
</svg>
<figcaption>The same four slots, packed contiguously vs. fragmented across a free list.</figcaption>
</figure>
```

> SVG `transform` on a child element animates fine in modern browsers. If you
> scale/rotate (not just translate), add `transform-box:fill-box;transform-origin:center`
> to the animated class so the origin is the element, not the SVG root.

---

## Authoring & validation flow (Phase C-anim)

1. Author the **full embeddable block** (figure → svg → figcaption) into
   `.cache/blog-writer/<slug>/<slug>-anim-<i>.fig.html`. Multi-line is fine and
   encouraged for readability — just **zero blank lines** between `<figure>` and
   `</figure>`, and `<figure` at column 0.
2. Validate:

   ```bash
   node .claude/skills/blog-writer/scripts/check-anim.mjs .cache/blog-writer/<slug>/<slug>-anim-<i>.fig.html
   ```

   It prints `PASS`/`FAIL <rule> — <detail>` per rule and exits non-zero on any
   FAIL. Read the messages — they name the offending line. **Do not bypass.**
3. Embed the validated block **verbatim** under the heading it illustrates (Phase
   D) — paste, don't `![]()`. Animated figures **count** toward the figure floor and
   **satisfy** abstraction-coverage; `verify-post.sh` recognizes `blog-anim` blocks.
4. The `.fig.html` is a cache artifact — Phase F's `rm -rf .cache/blog-writer/<slug>`
   removes it. The published source of truth is the block inside the `.md`.

---

## Self-review — the part a still `Read` can't see

Opening the figure with `Read` shows **one frozen frame**, not the motion. So review
the **source** and (ideally) the running page:

1. **Trace the keyframes.** Does the `0%` state *and* the `100%` state each map to
   something true in the prose? Is the **change between them** the exact thing the
   `_claim`/`figcaption` asserts? (Motion for its own sake = re-author or make it
   static.)
2. **Clean loop.** No jarring snap-back unless you used `alternate`. The reset reads
   as intentional, not a glitch.
3. **Reduced-motion path.** The `prefers-reduced-motion` branch freezes on a
   **meaningful** frame — not a blank or half-built one.
4. **Static legibility.** At the loop's start and end frames the figure is readable
   on its own (squint test): one reading direction, ≤ 3 accent colors, no legend.
5. **Watch it run.** For any post with animated figures, run `npm run dev` (or
   `bun run dev`), open the page, and watch each loop once before shipping —
   recommend the same to the user in the final report. The validator checks
   structure; only your eyes confirm the motion *says the right thing*.

Animated figures still pass through the **Phase C2** rubric for their static frames
(faithful, balanced, no dead space, text renders, squint) — animation is an extra
axis on top, not a waiver.
