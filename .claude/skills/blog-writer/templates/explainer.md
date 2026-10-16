# Explainer outline skeleton

Mirrors `kv-cache.md`. Target: 9–14 min read, 2–3k prose words, 4–6 figures (≥ half pictorial, ≤ 2 box-flow), no case studies (instead: worked examples).

## Section list

1. **Hook (`## Introduction`)**: open with a phenomenon the reader has noticed (e.g. "If you've ever used ChatGPT…"), then surface the hidden mechanism. End the intro with: "Let's build up the intuition step by step."
2. **Embed the primary diagram** right after the hook paragraph.
3. **`## Prerequisites: How <thing> Works`**: 1–3 sub-sections covering the minimum background. Always include:
   - A concrete analogy (library, restaurant, city map). Set in a fenced code block as ASCII or as a plain prose paragraph.
   - The minimal math, with every symbol defined.
4. **`## The <problem>: Why We Need <thing>`**: show the pain without the optimization. Include a worked numerical example.
5. **`## How <thing> Works`**: the core explanation. Walk through one full step-by-step example with small numbers (e.g. 4-token sequence, 2-dim vectors).
6. **`## Memory / Cost / Trade-offs`**: quantify. Tables of "without X vs with X" for typical model sizes.
7. **`## Optimizations and Variants`**: 3–6 named techniques (e.g. MQA, GQA, MLA, paged-attention). One paragraph + one comparison row each.
8. **`## Summary`**: 5–8 bullets. The last bullet should be the single most important takeaway.

## Diagram plan defaults

- Figure 1: the pain made visible, e.g. `bars` of wasted work or an `xy-chart` of cost growing, embedded in section 1 or 2.
- Figure 2: the mechanism as a `worked-example` on the same small numbers the prose uses (the 4-token sequence, the 2-dim vectors).
- Figure 3: a `cells` memory layout or a `filmstrip` of the state changing step by step.
- Figure 4: the trade-off or variants as `bars` or a `matrix`.

## Required patterns to include

- One concrete analogy in a fenced block.
- One math display block with every symbol defined inline below it.
- One small worked example with explicit numbers.
- One comparison table.
