# Finance deep-dive outline skeleton

Target: 38–60 min read, 8.5–13k words, ≥ 7 figures, ≥ 4 worked examples. The depth a practitioner respects, the on-ramp a beginner can climb. Mirrors the rigor of `quantitative-finance/black-scholes.md` but stays accessible the whole way down.

## Section list

1. **TL;DR callout** (`> [!important]` with a `**TL;DR**` lead) — at the very top, before anything else. ≤ 6 bullets: the thesis, the 3–4 key takeaways, and the one number/fact to remember.
2. **Hook** — a question or surprising number the reader has actually wondered about. No definitions yet. End the intro by referencing the mental-model figure.
3. **The mental-model figure** — embed the primary diagram right after the intro and explain it in 2–3 sentences. The rest of the article tours this picture.
4. **`## Foundations: the building blocks`** — define *everything* from zero: the instrument(s), the players, the units (bps, notional, par, yield, spread — whatever the post needs). 2–4 sub-sections. A beginner must be able to proceed from here; a pro can skim. Include the first worked example here (the simplest possible case).
5. **Numbered deep sections** (`## 1. …`, `## 2. …`) — one per mechanism/layer. Each:
   - Opens with the plain-English intuition + an analogy, *then* the formal definition / math.
   - Carries at least one worked example **or** a figure **or** a comparison table.
   - Ladders difficulty: simple case first, then add a realistic wrinkle (fees, taxes, time, uncertainty) and show how much the answer moves.
   - Ends with a short "what this costs / when it breaks" note.
6. **`## Common misconceptions`** — 4–6 wrong beliefs beginners hold, each corrected in 2–4 sentences with the *why*.
7. **`## How it shows up in real markets`** — 4–8 named scenarios or historical episodes:
   - `### N. Short evocative name`
   - 150–300 words: what happened, the mechanism from this post in action, the lesson. Real instruments, real dates, real numbers (cite the as-of date; every figure here must survive the Phase D2 fact-check).
8. **`## When this matters to you`** — where this touches the reader's money or decisions; honest about risk; "educational, not advice".
9. **`## Sources & further reading`** — the provenance closer (also satisfies the verify sourcing gate). A short bulleted list: the primary sources behind the headline numbers (link text + publisher + as-of date), then 3–6 further-reading links (primary sources + sibling posts on this blog). Sourced inline as natural markdown links, not academic `[12]` superscripts.

## Figure plan defaults (vary the kinds — ≥ 4 distinct)

- Figure 1 (mental model): the system at a glance — often a `pipeline`/`graph` of how the instrument or market works, or a `cash-flow-timeline`.
- Figure 2: a `cash-flow-timeline` or `payoff` chart for the core instrument.
- Figure 3: a `balance-sheet` / capital stack, or a `yield-curve` / growth chart.
- Figure 4: a `matrix` comparing the alternatives.
- Plus one figure per remaining abstraction (a figure within 30 lines of every prose abstraction).

## Required patterns to include

- The TL;DR callout at the very top.
- A Foundations section that assumes zero background.
- ≥ 4 worked numeric examples with round, friendly numbers and every step shown.
- At least one `| col | col | col |` comparison table.
- At least one `> blockquote` for a memorable, plain-English aphorism.
- Common misconceptions + real-markets sections (the finance closers).
- A `## Sources & further reading` section listing the primary sources behind the headline numbers.
- Every real-world figure sourced and dated (Phase D2); no fabricated or unverifiable numbers — range or date-and-attribute instead.
- No generic conclusion.
