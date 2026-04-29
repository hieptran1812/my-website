# Explainer outline skeleton

Mirrors `kv-cache.md`. Target: 8–15 min read, 2–4k words, 1–2 diagrams, no case studies (instead: worked examples).

## Section list

1. **Hook (`## Introduction`)** — open with a phenomenon the reader has noticed (e.g. "If you've ever used ChatGPT…"), then surface the hidden mechanism. End the intro with: "Let's build up the intuition step by step."
2. **Embed the primary diagram** right after the hook paragraph.
3. **`## Prerequisites: How <thing> Works`** — 1–3 sub-sections covering the minimum background. Always include:
   - A concrete analogy (library, restaurant, city map). Set in a fenced code block as ASCII or as a plain prose paragraph.
   - The minimal math, with every symbol defined.
4. **`## The <problem>: Why We Need <thing>`** — show the pain without the optimization. Include a worked numerical example.
5. **`## How <thing> Works`** — the core explanation. Walk through one full step-by-step example with small numbers (e.g. 4-token sequence, 2-dim vectors).
6. **`## Memory / Cost / Trade-offs`** — quantify. Tables of "without X vs with X" for typical model sizes.
7. **`## Optimizations and Variants`** — 3–6 named techniques (e.g. MQA, GQA, MLA, paged-attention). One paragraph + one comparison row each.
8. **`## Summary`** — 5–8 bullets. The last bullet should be the single most important takeaway.

## Diagram plan defaults

- Diagram 1: the before/after or the data-flow that motivates the topic. Embed in section 1 or 2.
- Diagram 2 (optional): a memory-layout or shape diagram for section 5 or 6.

## Required patterns to include

- One concrete analogy in a fenced block.
- One math display block with every symbol defined inline below it.
- One small worked example with explicit numbers.
- One comparison table.
