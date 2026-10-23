# Math for AI Research: Series Agent Kit

Read this file in full before drafting any post. Then read only your own row of the plan (`.claude/plans/math-for-ai-series.md`) plus your Track header, and grep `#<your number>` in its §Scope extensions. Never read the whole plan.

## 1. Series identity

- **Folder:** `content/blog/machine-learning/mathematics/<slug>.md`. Folder wins over frontmatter; it renders as "Mathematics".
- **Skill:** `blog-writer`, depth `deep-dive`. English. First-person principal-engineer voice from `voice-cheatsheet.md`, but warmer and slower than usual: the reader may be a strong engineer who last did math at university.
- **Promise:** after the series a reader can (1) follow the math and the proofs in LLM, diffusion, flow-matching, optimal-transport, MoE, RL and theory papers, and (2) write a correct theory section of their own.
- **Thesis line (use once per post, in your own words):** paper math is a small, closed vocabulary; learn each idea once, with a picture, a derivation, a proof and a numerical check, and the appendix stops being a wall.

### Frontmatter

```yaml
---
title: "Sentence-case Title: Optional Subtitle"
date: "YYYY-MM-DD"
publishDate: "YYYY-MM-DD"
description: "One sentence: what the reader can now read or prove."
tags: ["mathematics", "math-for-ai", "<topic>", "..."]   # 6–12, always the first two
category: "machine-learning"
subcategory: "Mathematics"
author: "Hiep Tran"
featured: true
readTime: 18
---
```

No `aiGenerated` field (repo convention; the gate's demand for it is an accepted deviation). No em dash in title or description.

## 2. The post skeleton (in this order)

1. **Difficulty tag + Needs / Unlocks** as the first lines after frontmatter, as a short blockquote:
   > **Level:** Core · **Needs:** [Post A](/blog/machine-learning/mathematics/slug-a), [Post B](...) · **Unlocks:** Score SDE, Flow Matching, DDIM appendices
   Levels: **Foundation** (Tracks A, B, D, E), **Core** (C, F, G, R, H, I, K), **Advanced** (J, L, M, N, O, P, Q, S), **Research** (T and proof-heavy rows). Use your judgement if the row is clearly harder or easier.
2. **Hook (1–2 paragraphs):** a real equation or sentence from a real paper that the reader cannot yet read, and the promise that by the end they will. Name the paper.
3. **The picture first:** the opening figure (one of the four primitives, adapted) before any symbol.
4. **Body:** each concept climbs the accessibility ladder (§3).
5. **Paper sentence, decoded** (§4).
6. **Proof, read line by line** (§5). From Track B on. Track A posts teach proof reading itself, so they decode a short proof instead.
7. **Check it numerically** (§6).
8. **Where this shows up in practice** (§7).
9. **What to remember:** 3–5 bullets in a blockquote or short list.
10. **Where next:** links to the intro (#1), the capstone (#72) and 2–3 siblings. Plain sentences, not a bare link list.

Headings: `##` for sections, `###` inside. No H1 in the body.

## 3. Accessibility contract (user requirement: easy to understand, visual)

Every new concept climbs this ladder, in order. Skipping a rung is a review failure.

1. **One plain-language sentence** saying what the idea is *for*, with no symbols. ("A gradient tells you which way is uphill and how steep it is.")
2. **An analogy or physical picture.** A hiker in fog (gradient descent), sand moved by shovel (optimal transport), dye spreading in water (diffusion, heat equation), a crowd flowing through a station (continuity equation), a rubber sheet stretched by a matrix (linear maps, determinants). Keep the analogy honest and say where it breaks.
3. **A figure** in the series' visual language (§8).
4. **A worked example with small numbers that the reader can check by hand:** a 2×2 matrix, a 3-outcome distribution, a two-step gradient descent. Show every arithmetic step. Recompute it before publishing; the verify gate never checks arithmetic.
5. **The formal definition or theorem.** Define every symbol at first use, in the sentence right after the display. State assumptions explicitly.
6. **The general / high-dimensional statement and the paper connection.** "In a 4096-dimensional embedding the same picture holds, except..."

Style rules that make math readable:
- One idea per paragraph. Short sentences around displays.
- After every display equation, one sentence saying what it means in words.
- Prefer named quantities over letters when first introduced: "the step size $\eta$", not just "$\eta$".
- Introduce notation only when it will be used at least twice.
- Bold the term at the moment it is defined; never bold for emphasis elsewhere.
- Never write "obviously", "trivially", "it is easy to see", or "it can be shown".
- Avoid the words that trip the verify gate's abstraction scan unless a figure follows within 30 lines: *imagine, think of it/this as, consider the case, the way it works, under the hood, conceptually, in essence, abstraction, abstractly*. ("Abstract algebra" is safe; "abstraction" is not.)

## 4. Paper sentence, decoded

A clearly marked block (a `###` heading "Paper sentence, decoded") containing:
- the paper's full title, authors (first author et al.), year, and a link to `https://arxiv.org/abs/<id>` (or OpenReview / proceedings link);
- the equation reproduced in LaTeX, with its number in the paper ("Eq. 6") or the section if the number is unclear;
- a symbol-by-symbol decode table or list: symbol, what it is, where in this post it was built;
- one sentence: what the equation says, in words.

**WebFetch the abstract page once to confirm the id and title.** If you cannot see the equation itself (PDF unreadable), cite the section and paraphrase, saying so. Never construct an equation that the paper does not contain. Ids marked "find id" in the plan must be found and confirmed or the paper replaced by one you can confirm.

## 5. Proof, read line by line

This is the user's headline requirement: *the reader must be able to read the proofs in papers*. Format:

1. **Statement**, with every assumption listed as (A1), (A2), ...
2. **The idea in one sentence** before any step ("bound the error by a telescoping sum, then use smoothness on each term").
3. **Numbered steps** (≤ ~15), each a display or a short line, followed by a justification in italics-free plain text: *by Cauchy–Schwarz*, *swap ∇ and E by dominated convergence (A2 gives the integrable bound)*, *integrate by parts; the boundary term vanishes because of (A3)*.
4. **Where each assumption was used:** a short table, assumption → step number.
5. **What breaks without it:** one concrete counterexample for the most important assumption.
6. **Proof moves used:** tag line, from this fixed vocabulary (extend it only in the capstone):
   `add-and-subtract` · `telescoping` · `Cauchy–Schwarz` · `Jensen` · `Young / AM–GM` · `triangle inequality` · `union bound` · `Fubini / DCT swap` · `integration by parts` · `change of variables` · `Grönwall` · `induction (layers / steps)` · `contraction / fixed point` · `change of measure` · `coupling` · `ε/3 argument` · `Taylor with remainder` · `duality` · `convexity (first-order condition)` · `concentration + union bound` · `reduction (hardness)` · `probabilistic method`.

Source the proof: a named paper appendix (with section), or a standard textbook (author, title, theorem number). If you simplify a published proof, say "simplified from".

## 6. Check it numerically

- One fenced Python block, ≤ ~30 lines, NumPy (or PyTorch only when the post is about autodiff / training), seeded, runnable as-is.
- It verifies the post's central identity or theorem: finite-difference vs analytic gradient, Monte Carlo vs closed form, Sinkhorn plan vs LP, Euler vs exact flow, empirical coverage vs nominal.
- **Run it once with Bash** (in the scratchpad, `python3 -I`) and paste the real output in a second fenced block labelled `text`. Never write output you did not see.
- One sentence after it: what agreement you expected and why the residual error has the size it has.

## 7. Where this shows up in practice

2–3 named, real applications, each one paragraph:
- what the system or recipe is (LoRA adapters, RLHF KL penalty, DP-SGD, conformal triage, minibatch OT in a flow-matching trainer, FP8 training in DeepSeek-V3);
- which exact piece of this post's math it uses;
- a link to the source (paper, engineering blog, official docs) with year.

Never invent a company, a deployment, a user count or a benchmark. If you cannot source a production use, use a well-known training recipe or open-source library (with a link to its code or docs) instead.

## 8. Recurring visual language

Four primitives, defined once in `.cache/blog-writer/_math-primitives/` (`ladder-map.in.json`, `plane-2d.in.json`, `density-filmstrip.in.json`, `loss-landscape.in.json`). Copy and adapt; do not redesign.

| Primitive | Use it for | Convention |
| --- | --- | --- |
| **Ladder map** | opening figure of Track A posts and any post that orients the reader | bottom-to-top rungs A → N–T; your rung lit primary; arrows to the paper families it unlocks |
| **2D plane** | vectors, matrices, eigenvectors, projections, rotations, gradients, vector fields | fixed axes and grid, unit circle, e₁ e₂; exact coordinates at a stated scale |
| **Density filmstrip** | change of variables, pushforward, diffusion, Fokker–Planck, flow matching, OT, discrete diffusion | 3–5 equal panels p₀ → p₁, same y-scale, time arrow under the strip |
| **Loss landscape** | GD, SGD, momentum, Adam, natural gradient, implicit bias, saddles | contour ellipses of ½(x² + 10y²) or a saddle variant; GD in amber, the improved method in primary |

**Primitive coordinate conventions** (keep them so sibling posts line up; `gen_primitives.py` in the same folder computes every point, adapt it rather than placing by eye):
- **Frame:** canvas 2400×1600, title y=60 (32 Virgil), caption y=120 (28 Virgil). Free text is left-anchored: to centre, set `x = centre − width/2`. Title / caption need real widths.
- **Ladder:** 14 rungs, rails x=360 and x=1000, rung height 80, stride 100, A at y=1500 up to N–T at y=200. Below the lit rung: grey fill; lit: blue, stroke 4; above: unfilled. Paper families at x=1700, 620×140, lavender. Lit rung's arrows blue width 5, others grey width 2. Keep arrows monotone (higher rung → higher destination) so none cross.
- **2D plane:** 1 unit = 200 px, origin (760, 760), grid x −3..3, y −2..2; circles / ellipses as 120-point closed polylines. Every label must fit inside one grid cell (the text-over-stroke check includes dashed gridlines). Positive tick labels right-below (x) / right-above (y) their gridline, negative ones left-above / left-below.
- **Filmstrip:** 5 panels 416 px wide, 40 px gap, from x=80; each spans x ∈ −3.5..3.5; density baseline y=700, density 1.0 = 1100 px on every panel; sample every 0.125 (57 points). Optional mass-bar row at baseline y=1260 (mass 1.0 = 520 px); time arrow y=1380.
- **Loss landscape:** equal axes, 1 unit = 170 px, minimum at (1200, 600), view x ±6.4, y ±2.2, axes on bottom and left edges; contours f = a²/2, a = 1..6 (semi-axes a and a/√10). Naive path amber (stroke `#f08c00`), improved path blue (stroke `#1971c2`), minimum green (stroke `#2f9e44`). Mark iterates with arrowheads; dots only at start and minimum.
- **Strokes:** blue fill pairs with stroke `#1971c2`, amber with `#f08c00`, green with `#2f9e44`; contours `#868e96`; grids `#ced4da` dashed. Minus sign is U+2212 `−`.

**Renderer / validator gotchas (measured in Wave 0):**
- `line` elements count as solid boxes in the overlap check: start every curve or closed outline at its lowest on-screen point and give it a negative `height`, or its bbox collides with every label inside.
- Arrows may not cross shapes larger than 12 px; marker dots must be ≤ 12 px.
- `"endArrowhead": null` is turned into `'arrow'` by `author-scene.mjs`; omit arrows you do not want instead.
- Each batch render logs `[page error] Cannot read properties of undefined (reading 'length')` plus a 404 before drawing: harmless.
- Monotone arrow ordering beats rerouting for crossing fixes.

Palette: primary `#a5d8ff` (the thing being taught), caution `#ffec99`, danger `#ffc9c9`, success `#b2f2bb`, external `#d0bfff`, neutral `#e9ecef`. At most 3 strong accents. Virgil (`fontFamily: 1`) for prose labels, Cascadia (`3`) for numbers and symbols. Math in figures is plain Unicode (θ, ∇, ‖x‖, p₀, ∑), never LaTeX.

**Figure budget per post:** 5–8 static WebPs, target 6, plus 1 animated `blog-anim` figure where motion carries meaning (gradient-descent path, ODE / SDE trajectories, a density morphing, a rotation, a transport plan). At least half pictorial (xy-chart, space, cells, filmstrip, worked-example, anatomy); box-and-arrow flow at most 2. No kind more than twice. **Ask for exact-proportion data in every chart brief:** the reviewer does not check data fidelity. Compute curve points with a script, never by eye.

Math-specific figure kinds that work well: worked-example panels (a 2×2 multiplication laid out cell by cell), cells (a matrix as a heatmap of its entries), space (vector fields as arrow grids), xy-chart (a function and its Taylor approximations), filmstrip (iterates of an algorithm), anatomy (an equation with each symbol called out).

Delegate the whole figure pipeline to `figure-author` (render helper: `bash .cache/blog-writer/_render-math.sh <slug>`). Never read a rendered WebP yourself. The renderer lives outside the repo (`/Users/hieptran1812/Documents/mcp_excalidraw/scripts/render-scene-batch.mjs`); the sharpness gate passes with zero files, so **count the WebPs on disk**.

## 9. LaTeX rules (this series' biggest risk)

The site renders math with the regex in `src/app/components/MathJax.tsx`. There is no remark-math. Rules that follow from that regex:

- **Inline** `$...$` must sit on one line; no whitespace right after the opening `$` or right before the closing `$`; the closing `$` must not be followed by a letter or digit (write "$x$ values", never "$x$s").
- **Inline math starting with a digit** renders only if it contains one of `\ { } ^ _`. Write `${2}^{n}$`, `${0.5}$`, never `$2n$` or `$0.5$`. Plain numbers stay plain text.
- **`<` followed by a letter inside math** is parsed as an HTML tag and swallows text. Write `\lt ` (with a space): `$a \lt b$`. Same caution for `>` before a letter: use `\gt `.
- **Display math:** `$$` on its own line, before and after. Never inside a list item or a table cell.
- **Never** use `\(...\)` or `\[...\]`. Never write `\$` adjacent to math. For currency write "USD 5" or "5 dollars".
- **Odd count of `$`** outside code fences = desync. Before verify, run a fence-aware count: strip fenced blocks, then count `$` per paragraph.
- Prefer `\mathbb{E}`, `\nabla_\theta`, `\operatorname{tr}`, `\lVert x \rVert`, `\mid` for conditioning, `\,` before `dx`. Use `\begin{aligned}` inside `$$` for multi-line derivations.
- Markdown tables: no `|` inside math in a table (use `\lvert ... \rvert` or `\mid`).

Pre-verify scan (run it, fix every hit):

```bash
python3 - "$POST" <<'EOF'
import re,sys
t=open(sys.argv[1]).read()
t=re.sub(r"```.*?```","",t,flags=re.S)
t=re.sub(r'<figure class="blog-anim".*?</figure>',"",t,flags=re.S)
body=re.sub(r"\$\$.*?\$\$","",t,flags=re.S)
for i,l in enumerate(body.split("\n"),1):
    if l.count("$")%2: print("odd $:",i,l[:90])
    for m in re.finditer(r"(?<![\\$])\$(\d[^$]*?)\$",l):
        if not re.search(r"[\\{}^_]",m.group(1)): print("digit-start:",i,m.group(0))
for m in re.finditer(r"\$[^$]*<[A-Za-z][^$]*\$",t): print("raw <:",m.group(0)[:80])
for bad in [r"\\\(",r"\\\[","—"," – "," -- "]:
    for m in re.finditer(bad,t): print("banned:",repr(m.group(0)),t[max(0,m.start()-30):m.start()+30].replace("\n"," "))
EOF
```

## 10. Honesty contract

Every mathematical claim is (a) proved or derived in the post, (b) cited to a named paper or textbook with theorem / equation number and link, or (c) numerically checked in §6. Statements beyond the post's scope get "we state this without proof" plus a precise citation. Every number in prose is recomputed before publishing. Never invent a paper, an equation, an id, a company, a benchmark or a code output.

## 11. Boundaries and cross-links

| Neighbour | Link, do not re-derive |
| --- | --- |
| `trading/math-for-quants/` | Itô, measure theory, OT, information geometry with finance examples. Link once where it helps a reader who wants a second angle. |
| `machine-learning/image-generation/` | `the-math-of-ddpm`, `score-based-models-and-the-sde-view`, `flow-matching-and-rectified-flow`, `normalizing-flows-and-change-of-variables`: the models. We own the math under them. |
| `machine-learning/large-language-model/` | MoE and attention systems posts. |
| `machine-learning/reinforcement-learning/` | the RL series; we only do the math needed for RLHF / DPO / GRPO and MDP theory. |
| `machine-learning/scaling-laws/` | empirical results; we do the fitting and the Lagrangian. |

Link format: `/blog/machine-learning/mathematics/<slug>` (no trailing slash, no `.md`). Every post links #1 `how-to-read-the-math-in-an-ml-paper` and #72 `the-math-roadmap-for-ml-researchers`, 2–3 siblings, 1–2 neighbours. Links to series posts not yet written are fine (they resolve when the wave ships), but use the exact slug from the plan.

## 12. Gates (definition of done)

1. `WORD_MIN=3000 WORD_MAX=4000 bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> deep-dive` through one `post-verifier`; accepted FAIL: `aiGenerated` missing.
2. The §9 LaTeX scan prints nothing.
3. 5–8 WebPs on disk named `<slug>-N.webp`, every one passed `figure-reviewer`; 1 animated figure validated by `check-anim.mjs`.
4. The §6 snippet ran and its real output is in the post.
5. Paper id(s) confirmed by WebFetch; every "find id" resolved.
6. No em dash anywhere (`—`, spaced ` – `, ` -- `).
7. `.cache/blog-writer/<slug>/` cleared after the post passes.

## 13. Agent hygiene

- Check `content/blog/machine-learning/mathematics/<slug>.md` and `git ls-files public/imgs/blogs/<slug>-*` before starting; a same-slug render clobbers committed WebPs (`_render-math.sh` refuses, do not work around it).
- Delegate figures to `figure-author`; never open a rendered image yourself.
- Summarise children's findings; never pass raw child returns upward.
- Write the draft to disk early and keep it there; agents can go idle mid-pipeline.
