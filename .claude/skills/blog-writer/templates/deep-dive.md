# Deep-dive outline skeleton

Mirrors `docker-optimization-for-llm-and-ai-workloads.md`. Target: 9–14 min read, 2–3k prose words, 5–7 figures (≥ half pictorial, ≤ 2 box-flow), 2–4 case studies. Breadth with economy: each layer gets its rule of thumb, one code block or number, and its gotcha in a few tight paragraphs.

## Section list

1. **Hook / why this is different**: open with the mismatch between common assumption and reality. Include a `## Why X is different` table (assumption | naive view | reality).
2. **The mental model figure**: embed the primary diagram immediately after the intro paragraph and explain it in 2–3 sentences. The rest of the article is a tour of that diagram.
3. **Section per layer / dimension**: number them (`## 1. The host layer`, `## 2. Image engineering`, …). Each section:
   - Opens with a senior rule of thumb in bold.
   - Has at least one runnable code block or shell snippet with realistic flags.
   - Ends with a "second-order optimization" sub-section noting a non-obvious gotcha.
4. **Cross-cutting concerns**: observability, security, cost, multi-tenancy. Pick 2–3.
5. **Case studies**: `## Case studies from production`. 2–4 numbered entries:
   - `### N. Short evocative name`
   - 120–200 words: the symptom, the wrong first hypothesis, the actual root cause, the fix, the lesson.
   - Always concrete: real tool versions, real flag names, real model sizes.
6. **When to reach for X / when not to**: the closing section. Two sub-sections:
   - "Reach for X when …": 4–6 bullet conditions.
   - "Skip X when …": 3–5 anti-patterns / overkill cases.
7. (Optional) **Further reading**: 3–6 links (official docs, key papers, sibling posts on this blog).

## Diagram plan defaults

- Figure 1 (mental model): the system drawn as what it *is*, e.g. memory or cost as `bars` to scale, or the request as `lanes` over time; a layered stack only if the layers themselves are the point.
- Figure 2: the optimization as a `filmstrip` or `bars` before vs after, with the real numbers.
- Figure 3: the tradeoff as an `xy-chart` (where it pays off, where it stops paying).
- Figure 4+: `cells` for any data structure, `lanes` for timing, a `worked-example` for any formula; at most two box-flow figures in the whole post.

## Required patterns to include

- At least one `| col | col | col |` markdown table.
- At least one fenced shell block with real CLI flags.
- At least one fenced code block in the production language (Python, Dockerfile, YAML, etc.).
- At least one `> blockquote` for a memorable senior-engineer aphorism.
