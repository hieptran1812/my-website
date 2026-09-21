# Senior Quant Math series - drafting brief

Given to every drafting agent in this series. Lives in the repo rather than a
scratchpad because session restarts wipe the scratchpad, and without this brief
each new wave repeats the defects waves 1 and 2 already paid for.

You are writing one post for the **Senior Quant Math** series in
`content/blog/trading/math-for-quants/`. Use the **finance-writer** skill and run
its full pipeline: Phase A intake, B outline, C figures (delegate to
`figure-author`; never render or Read a WebP yourself), D draft, D2 fact-check,
E verify gate, F cache cleanup.

Depth tier is **explainer**. Target band **3,600-4,000 words of body**. Two posts
in wave 2 overshot and needed a whole extra compression pass; aim inside the band
the first time.

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

## Maths rendering rules for this repo

These four have cost the corpus hundreds of fixes. They are not style choices.

1. **Brace-wrap any inline span that starts with a digit.** `$1/n$` is silently
   dropped by the site renderer and the reader sees literal dollar signs. Write
   `${1/n}$`. Any span whose content begins with a digit and contains no
   `\`, `{`, `}`, `^` or `_` must be brace-wrapped.
2. **Escape currency in prose**: `\$50m`, `\$100k`, never a bare `$50m`. A bare
   currency dollar pairs with the next real formula's delimiter and swallows the
   sentence between them.
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
