# Paper-reading outline skeleton

For posts under `content/blog/paper-reading/<area>/`. Target: 9–14 min read, 2–3k prose words, 4–6 figures (≥ half pictorial, ≤ 2 box-flow).

## Section list

1. **TL;DR box**, a `> [!tldr]`-style blockquote at the very top: 3–5 bullets covering (a) what the paper claims, (b) why it matters, (c) the single most surprising finding, (d) where it fails.
2. **`## Context: what came before`**: 2–4 paragraphs framing the problem and the lineage of prior work the paper builds on. End with the gap this paper claims to fill.
3. **`## Contributions`**: numbered list of the paper's stated contributions, in the authors' words but tightened. 3–6 items.
4. **`## Method`**: the heart of the post:
   - One diagram (the architecture / training pipeline / loss).
   - Sub-sections per component. Define every symbol the first time it appears.
   - At least one snippet of pseudocode or pytorch-shaped code reproducing the key idea.
5. **`## Experiments`**: table of the headline results. Note which baselines, which datasets, which compute scale. One paragraph of "what's load-bearing in their setup that might not transfer".
6. **`## Critique`**: the senior-engineer lens:
   - What's strong.
   - What's weak / unfalsifiable / cherry-picked.
   - What ablation is missing.
7. **`## What I'd build with this`**: 3–5 concrete extensions or applications. Optional: a sketch of a follow-up experiment.
8. **`## References`**: link to the arxiv PDF, code repo (if any), and 2–3 sibling posts on this blog.

## Diagram plan defaults

- Figure 1: redraw the paper's key idea in Excalidraw style, as a `worked-example` or `cells` figure on a tiny instance where possible, rather than a box-per-module architecture chart.
- Figure 2: the headline result as `bars` or an `xy-chart` with the baselines.
- Figure 3+: whatever the method's shape is (`filmstrip` of training state, `space` for representations, `lanes` for compute).

## Required patterns to include

- The TL;DR callout at the top.
- One results table with the baselines named.
- An explicit "what would change my mind" line in the Critique section.
