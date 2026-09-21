# Senior Quant Math series - drafting brief

Given to every drafting agent in this series. Lives in the repo rather than a
scratchpad because session restarts wipe the scratchpad, and without this brief
each new wave repeats the defects waves 1 and 2 already paid for.

You are writing one post for the **Senior Quant Math** series in
`content/blog/trading/math-for-quants/`. Use the **finance-writer** skill and run
its full pipeline: Phase A intake, B outline, C figures (delegate to
`figure-author`; never render or Read a WebP yourself), D draft, D2 fact-check,
E verify gate, F cache cleanup.

Depth tier is **explainer**. Target band **3,600-4,000 words of body**.

The band is a discipline on prose, not a ceiling over required content. Aim inside
it on the first draft, because every post so far has needed a compression pass it
should not have needed. But if the only remaining cut is a mandated element - a
worked example, a misconception the brief named, a section the reader needs - then
stop and say so rather than dropping it. Waves 2 and 3 shipped at 4,396 to 5,121
for exactly that reason. Report the count and the reason; the call is the
orchestrator's, not yours.

## The worked-examples gate bites every post in this series

Gate 5 is an **AND**: it needs `worked example` markers *and* `$`-figures on the
same lines. A post full of clean unitless walkthroughs (eigenvalues, bounds,
probabilities, ICs) scores "0 $-figures below floor 3" and FAILS, even with five
perfectly good examples. Every post here is maths-with-no-money, so plan for this
from the outline, not after the gate.

The fix is never to relabel prose. Give at least three worked examples a real book
size and carry the arithmetic through to money: state a \$500m portfolio, a \$50m
position, a \$100k per-trade risk, and show what the maths does to that number. It
satisfies the gate *and* makes the point land harder, because a reader feels
"\$18m in one name" in a way they never feel "weight 0.36".

## Check your own arithmetic; the gate cannot

The verify gate counts examples, measures figures and catches em dashes. It does
**not** recompute your numbers. Wave 3 shipped two t-statistics that did not
reproduce, and a sweep of the already-published posts found two more.

Every defect was the same shape: **a denominator rounded one step early, then
carried**. `65 x 9.2` was written as 600 rather than 598, and the downstream
`sqrt(600) x 0.03 = 0.735` then failed to round to the 0.73 the post printed,
while the true 598 gives 0.7335 and does.

So before you declare a post done, take every stated division and product in the
prose and recompute it at the precision you printed. Carry unrounded intermediates
and round only at the end. If a figure and the prose disagree, work out which one
is wrong before assuming it is the figure.

### Verify a closed form against an *independently derived* closed form

When the post derives a pricing formula, checking your own algebra can only catch a
wrong arithmetic step, never a wrong formula. Re-deriving your own result by hand
reproduces your own mistake, so that is not a check. Price the same instrument by a
**different derivation** and compare.

The local-time post checked its method-of-images down-and-out against the eight-term
Reiner-Rubinstein formula: they agree to 13 decimals only if the image weight
`(S/H)^(1-2r/sigma^2)` is right, which is the one place a barrier post can be
confidently wrong. That is what makes the derivation trustworthy rather than merely
self-consistent.

Do this whenever the post's central claim *is* a formula.

### Re-grep before you act on a line-level claim

Any claim about a specific line, from the orchestrator or from a checker, is a
hypothesis about a file that may have changed since it was measured. Re-grep the
line before touching anything. It costs one tool call and it turns a rewrite into a
no-op when the claim is already stale, which in wave 5 it was four times out of four.

### The audit is reader-reproducibility, not truth

This is the framing that matters, and it took three waves to find. An audit that
compares your computed values against each other passes while the page is still
wrong, because the page carries a **rounded input beside a result computed from the
unrounded one**. Both numbers are correct. The row is not.

    printed   0.3551 x 5,000,000 = $1,775,701
    from page 0.3551 x 5,000,000 = $1,775,500   (the 1,775,701 needs p = 0.3551402)

Its cousin bites whenever two figures are differenced or ratioed: two correctly
rounded numbers on the page do not differ by the printed difference, because
rounded(A) minus rounded(B) is not rounded(A minus B). A reader who subtracts the
two numbers in front of them gets a third number.

So recompute **using only the numbers printed on the page**, at the precision
printed. One wave-5 post had already audited 45 values against `erfc` and `Fraction`
and still shipped eight bad rows; re-running the same post under the printed-string
rule found all eight.

When a row does not reproduce, fix it by **naming the unrounded value in the text**,
not by degrading the money to match the rounded input. In that post the headline
ratio was 2.12 from the true values and would have become 2.13 the other way.

**Check the figures before rewriting prose.** If a figure displays one of the
disputed numbers, changing the prose freely desyncs figure and text, which is worse
than the defect you are fixing. Read the `.scene.json` in the cache as text; never
open the WebP. If the cache is already cleared, say so and escalate rather than
guessing.

### Assert against exact fractions, not floats, and round half-up

Eye-checking cannot find the worst version of this defect, because every row looks
individually plausible. Wave 4 shipped a shrinkage table whose printed equalities
were built from **rounded** factors while the reported results came from the
**exact** ones: `0.75 + 0.50 x 0.59 = 1.05` is really 1.045, and all five rows
shared the flaw because `B` was displayed at 2 dp while the arithmetic carried
4/5, 4/7, 4/9, 1/2 and 1/3.

So write the check as assertions, not as a reading pass:

- compute with `fractions.Fraction`, never floats, so `4/9` stays `4/9`
- round **half-up** explicitly; Python's default banker's rounding hides ties like
  1.045 exactly where this defect lives
- assert the *printed* string against the computed value at the precision printed
- if a table displays a rounded intermediate, either print the exact fraction
  beside it or carry the result to enough places that the row is a true identity

The fix that worked was printing each `B` as its exact fraction next to the
decimal and carrying results to three places. It made every row an exact identity
*and* reinforced the closed form derived above it.

## Two gate quirks that cost a pass each

- **Do not write "the picture above".** The abstraction-coverage gate matches
  `picture (a|this)` with no word boundary, so "picture ab-ove" trips it, and its
  search window only looks *forward*, so a figure sitting immediately above cannot
  clear it. Write "that figure" instead.
- A `## Sources and further reading` section clears the sourcing gate to a WARN,
  never a FAIL, at this depth. The WARN listing your illustrative worked-example
  dollar figures is expected; state once in the text that they are illustrative
  arithmetic on assumed inputs and move on.

## Maths rendering rules for this repo

These four have cost the corpus hundreds of fixes. They are not style choices.

1. **Brace-wrap any inline span that starts with a digit.** `$1/n$` is silently
   dropped by the site renderer and the reader sees literal dollar signs. Write
   `${1/n}$`. Any span whose content begins with a digit and contains no
   `\`, `{`, `}`, `^` or `_` must be brace-wrapped.
2. **Escape currency everywhere, including image alt text**: `\$50m`, `\$100k`,
   never a bare `$50m`. A bare currency dollar pairs with the next real formula's
   delimiter and swallows the sentence between them. Alt text is the easy one to
   forget, and a single unpaired dollar there is invisible to any checker that looks
   for matched spans, so it survives every sweep until a reader sees it.
3. **Display math needs blank lines either side.** A `$$...$$` block jammed against
   prose, or inside a list item without its own block, is folded into the paragraph
   and ships as literal source. Inside a numbered list, indent the block to the
   list marker width (3 spaces for `1. `) so the numbering survives.
4. **Never put LaTeX inside a backtick code span.** It renders as monospace source.

Also: no `\*` (undefined in KaTeX, use `\ast`), and no raw `<` followed by a letter
inside math (use `\lt `), which the HTML parser eats along with the closing
delimiter.

## House rules

- **No em dashes.** None, anywhere, including spaced ` - ` and ` -- ` variants. The
  verify gate fails on them. Use a comma, a colon, or two sentences.
- **No fabricated numbers.** Do not invent empirical coefficients, R-squared
  values, decay half-lives, or study results. Where the literature has a number,
  cite it and name the source. Where it does not, present the arithmetic as
  clearly labelled illustrative work on assumed inputs, and say so in the text.
- Every post ends with `## Sources and further reading` (real, checkable
  references), then the required closing section
  `## In the interview room and on the desk`, 250-400 words: the question as it is
  actually asked, the strong answer in order, and the trap that makes a candidate
  look rigorous while being wrong. Name which firms weight it.
- Set frontmatter `readTime` to whatever the verify gate recomputes.
- Cross-link existing posts rather than re-deriving what they cover.

## Finish condition

Run the gate until green, then report slug, word count, figure count, example count:

```
bash .claude/skills/finance-writer/scripts/verify-finance-post.sh <path> <slug> explainer
```
