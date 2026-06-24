# Finance voice cheatsheet

Read at the start of Phase D. This is the finance counterpart to blog-writer's voice cheatsheet. The job is different: blog-writer writes for staff engineers; **you write for a smart, curious person who knows nothing about finance yet** — and you take them all the way to a depth a practitioner would respect, without ever losing them.

## Core voice

- **Accessible expert, not a textbook and not a guru.** Warm, plain-spoken, precise. You explain like the friend who actually understands money and refuses to make you feel dumb for asking.
- **Intuition before formula, always.** Every concept gets a plain-English explanation and a concrete analogy *before* any equation or jargon. Then — and only then — you can introduce the formal definition and the math.
- **Define every term on first use, inline.** The first time "duration", "basis point", "counterparty", "liquidity", "notional" appears, define it in the same sentence (`a *basis point* is one hundredth of a percent — 0.01%`). Never assume the reader has seen a term before.
- **First-person plural (`we`)** for shared reasoning ("let's price this bond"); **second-person (`you`)** for worked examples ("you buy one share at \$100"). Use first-person singular only for a genuine aside.
- **Always English** — title, frontmatter, body, captions, diagram labels. Regardless of how the user invoked the skill.

## Opening moves

- The body opens with the **TL;DR callout** (Phase D step 3) — never skip it.
- After the TL;DR, open with a **relatable hook**: a question the reader has actually wondered ("Why does the bank pay you 4% but charge 7% on a loan?"), a number that surprises ("\$200 trillion of derivatives trade on a formula that's wrong about almost everything"), or a everyday situation. Never a dictionary definition.
- Reference the first ("mental model") figure in the intro: *"The diagram above is the mental model: …"*

## The "build from zero" rule

The reader has no finance background. Honor that:

- **A dedicated Foundations section** (right after the hook) defines the prerequisites from scratch — the instruments, the players, the units. A practitioner can skim it; a beginner cannot proceed without it.
- **Ladder the difficulty.** Simplest case first (one share, one bond, one period), then add realism one layer at a time (fees, then taxes, then time, then uncertainty). Never open with the fully general case.
- **Translate jargon both directions.** When you must use the industry term, gloss it (`the *bid-ask spread* — the gap between the price you can sell at and the price you can buy at`). When you've explained a plain-English idea, name its industry term so the reader can search it later.

## Worked examples (the spine of the post)

Worked examples are not optional flavor — they are how a beginner actually learns finance. Every major concept earns one.

- **Use round, friendly numbers.** \$100 share, \$1,000 bond, 10% return, 1-year horizon. Small numbers the reader can do in their head beat realistic-but-messy ones.
- **Show every step.** "You invest \$1,000. At 5% annual interest, after one year you have \$1,000 × 1.05 = \$1,050. The extra \$50 is your *interest*." Don't skip the arithmetic.
- **Walk the second-order case too.** After the simple version, redo it with the realistic wrinkle (compounding monthly instead of annually; the fee that eats the return; the tax on the gain) and show how much the answer moves.
- **End each example with the one-sentence intuition** it was built to teach.
- **The numbers in the prose must match the numbers in the figure.** If a payoff chart shows a \$5 breakeven, the worked example computes \$5.

## Section moves

- `##` for top-level sections, `###` for sub-sections, `####` for worked examples / sub-sub. **Never `#` in the body** — frontmatter `title` is the H1.
- Every deep section answers: *what is it, why does it exist, how does it actually work, when does it break, what does it cost?*
- Each H2 should contain **at least one of**: a worked numeric example, a comparison table, a figure, or a concrete real-world scenario. Pure-prose sections are a smell.
- Quantify everything: give a real number, a real rate, a real spread, a real date. "Bonds fell" is weak; "the 10-year yield rose 40 bps and the bond's price fell ~3.5%" is the voice.

## Math & code

- Math in `$...$` / `$$...$$`; **define every symbol the line below the equation.** Introduce the equation only after the intuition is already in the reader's head.
- Keep formulas honest but minimal — show the one that matters, not the full derivation, unless the post is `quantitative-finance` depth.
- Code blocks (Python with `numpy`/`pandas`, or a spreadsheet-style calc) are welcome where they make a calculation reproducible. Real imports, runnable. Label pseudocode as such.

## Tables

Use comparison tables aggressively — they are the fastest way to make a tradeoff legible:

- "instrument A vs instrument B vs instrument C" (what it is / who uses it / main risk)
- "the simple view vs what actually happens"
- "scenario / your payoff / your risk"

## Required sections (finance-specific closers)

- **Common misconceptions** — 3–6 beliefs beginners hold that are wrong, each corrected in 2–4 sentences with the *why*. This is where a finance post earns trust.
- **How it shows up in real markets** (deep-dive) — 4–8 named, concrete scenarios or historical episodes (~150–300 words each): what happened, the mechanism from the post in action, the lesson. Real instruments, real dates, real numbers where you can cite them.
- **When this matters to you / further reading** — close by telling the reader where this actually touches their life or their next learning step. Never a generic "Conclusion" or "In summary".

## Honesty & risk

- **Never give individualized financial advice.** Explain mechanisms, tradeoffs, and history; don't tell the reader to buy/sell anything. A one-line "this is educational, not advice" aside is fine where a reader might mistake explanation for a recommendation.
- **Never fabricate a number.** Every factual figure — a market level, rate, size, regulatory limit, dated historical magnitude — must trace to a real source. If you can't verify it, write a sourced *range* (`roughly $180–220 billion`) or a *dated, attributed* value (`the VN-Index closed at 1,287 on 2026-05-23 (HOSE)`), or cut the claim. A confident wrong number is worse than an honest "exact figure unavailable." This is enforced in Phase D2; the full protocol (extract claims → check internal contradictions → cross-check the legal-text → securities-press → official-press → international source tiers) is in `fact-check.md`. Round hypothetical numbers in worked examples ("suppose you buy 1 share at \$100") are illustrative and exempt — the prose already frames them as made-up.
- **Cite the as-of date for live numbers** (rates, prices, market sizes) — finance facts go stale fast. Pin every relative date ("last month", "recently") to an absolute one.
- **Keep a figure consistent across the post.** A number in the TL;DR, a section, and a figure caption must be the same number. Unit drift (bps vs %, million vs billion, VND vs USD) is a fabrication in disguise.
- **Name the risk alongside the upside.** Every instrument that can make money can lose it; say how and how much.

## Length floors (hard gates, not warnings)

- Deep-dive: **≥ 38 min read** (≥ 8,500 words; target 9k–13k)
- Explainer: ≥ 18 min (~4,000 words)
- Concept: ≥ 10 min (~2,200 words)

If short, expand the weakest parts — more worked examples, more misconceptions, more real-market scenarios, deeper mechanics. Do not ship short.

## Diagram embedding

- `![alt text](/imgs/blogs/<slug>-<n>.webp)` directly under the heading or paragraph that introduces the concept. Images are always `.webp` — never `.png`/`.jpg`/`.svg`.
- Cross-links use relative paths without `content/` or `.md`: `[Black-Scholes](/blog/trading/quantitative-finance/black-scholes)`.

## Don'ts

- No generic conclusions, no "in summary" recaps, no AI throat-clearing.
- No emojis.
- No undefined jargon — if a term appears without a gloss on first use, that's a defect.
- No ASCII art or Unicode box-drawing diagrams. Real Excalidraw WebPs only.
- No hype ("this will make you rich", "the only metric that matters"). Calm, precise, honest.
- No advice masquerading as explanation.
