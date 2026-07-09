# Paper-analysis outline skeleton

For posts under `content/blog/paper-reading/<area>/`. Target: 30+ min read, ≥ 6,500 words, ≥ 6 figures (**≥ 2 extracted from the PDF + ≥ 2 redrawn in Excalidraw**). This is the detailed sibling of blog-writer's `paper-reading.md` — deeper, with the paper's own figures and the full intuition→math ladder per technique.

## Section list

1. **TL;DR box** — a `> [!tldr]` blockquote at the very top: 4–6 bullets covering (a) what the paper proposes, (b) the key mechanism in one line, (c) why it matters / what it beats, (d) the single most surprising result, (e) where it fails or what it doesn't show. Reference the first extracted figure right after ("The diagram above is the whole method at a glance; the rest of this post unpacks it.").

2. **`## The problem` (Context: what came before)** — 2–4 paragraphs framing the pain the paper attacks and the lineage of prior work it builds on / beats. End with the specific gap this paper claims to fill. Link prior-work posts on this blog.

3. **`## Contributions`** — the paper's stated contributions, tightened into a numbered list (3–6 items), in your words. This is the map for the rest of the post.

4. **`## Method`** — the heart, and the longest part. Lead with the **extracted architecture/method figure** (credited: "Figure N from <Authors> (<year>)"). Then one `###` sub-section **per load-bearing technique**, each climbing the full explanation ladder (`references/technique-explanation.md`): problem → intuition/analogy → mechanism → math (every symbol defined, shapes annotated) → worked micro-example → why-it-works/when-it-fails. Embed the redrawn diagrams where they make a mechanism or the math visible. At least one pytorch-shaped / pseudocode block reproducing the key idea.

5. **`## Experiments & results`** — a table of the headline numbers with **baselines named**, datasets, and compute scale (extract the paper's results figure/plot if it's a keeper). One paragraph on "what's load-bearing in their setup that might not transfer" (data scale, a specific tokenizer, a tuned hyperparameter, eval choice). If there's an ablation that carries the argument, walk it.

6. **`## Critique`** — the senior-engineer lens:
   - What's genuinely strong.
   - What's weak / unfalsifiable / cherry-picked / an unstated assumption.
   - What ablation or baseline is missing.
   - An explicit **"what would change my mind"** line.

7. **`## What I'd build with this`** — 3–5 concrete extensions, applications, or follow-up experiments. Flag these as *your* extrapolation, not the paper's claims.

8. **`## References`** — the paper (arxiv/DOI/openreview), the code repo if any, and 2–3 sibling posts on this blog. Never a generic "Conclusion".

## Figure plan defaults (two tracks)

**Extract (originals, ≥ 2):**
- The **architecture / method figure** — almost always Figure 1. Extract it; it anchors the Method section.
- The **headline results plot or table** — if the paper's own plot tells the story better than a redraw.
- Any figure whose *visual detail is the point* (an attention-pattern heatmap, a qualitative sample grid, a scaling curve).

**Redraw (Excalidraw, ≥ 2):**
- A **clarified data-flow** the paper only describes in prose, or a cleaner version of a too-dense original — shape-annotated where tensors move.
- A **loss / objective decomposition** (base term → regularizer → constraint) or a **before/after** of the quantity the method changes (naive vs proposed).
- Pick the kind from the concept (pipeline / graph / before-after / matrix / layered-stack / hand-authored); keep the set diverse; tag `animated` only if motion carries the meaning (a step-by-step process, an A↔B transition).

## Required patterns to include

- The `[!tldr]` callout at the top.
- The extracted architecture figure, credited to the paper, in the Method section.
- The full intuition→math ladder on every load-bearing technique.
- One results table with the baselines named and one derived/quantified claim.
- At least 3 display-math `$$…$$` blocks; every symbol defined on first use.
- An explicit "what would change my mind" line in the Critique.
- 2–4 cross-links to existing posts.
