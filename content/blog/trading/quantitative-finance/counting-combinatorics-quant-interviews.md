---
title: "Counting and combinatorics for quant interviews"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Most 'hard' probability questions in a quant interview are counting in disguise. This deep dive builds the whole counting toolkit from zero, grounds every technique in worked numbers, and solves real interview problems step by step."
tags:
  [
    "quant-interviews",
    "combinatorics",
    "counting",
    "probability",
    "permutations",
    "combinations",
    "stars-and-bars",
    "inclusion-exclusion",
    "birthday-problem",
    "poker-probability",
    "brainteasers",
    "trading",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Most "hard" probability questions in a quant interview are counting problems wearing a costume. A small fixed toolkit cracks almost all of them.
>
> - **Probability of an equally-likely event = (favorable outcomes) / (total outcomes).** So the whole game is *counting carefully* — count the numerator, count the denominator, divide.
> - **Six tools cover the field:** the product rule, the sum rule, permutations (order matters), combinations (order does not), stars-and-bars (distributing identical items), and inclusion-exclusion (overlapping "or" events).
> - **The single most common mistake is overcounting** — counting the same outcome twice. Symmetry arguments ("divide by the number of equivalent arrangements") fix it.
> - **You will compute real numbers in the room:** a flush is `5108 / 2,598,960 ≈ 0.197%`; in a room of just **23** people there is a **50.7%** chance two share a birthday; there are exactly **9** ways to hand back four hats so nobody gets their own.
> - **The skill that gets you hired** is narrating a clean, checkable count out loud — not memorizing formulas. We build every formula from scratch so you can re-derive it under pressure.

If you have ever been asked "what's the probability of a flush?" or "how many people until two share a birthday?" and felt your mind go blank, this post is for you. Those questions feel like they need some deep probability theory. They don't. They need careful *counting*. Once you see that, a huge slice of the quant-interview question bank collapses into one repeatable procedure.

The diagram below is the mental model for the whole article. Almost every probability question a trading firm throws at you reduces to the same first move — *reframe it as a count* — and then routes to one of a handful of counting tools depending on the shape of the question.

![The counting toolkit: a decision map from a probability question to the right counting technique](/imgs/blogs/counting-combinatorics-quant-interviews-1.png)

The thesis of this post is exactly that picture: **a probability question is a counting question in disguise, and a small toolkit solves it fast and correctly.** We will build each branch of that map from the ground up — assuming you remember nothing past middle-school multiplication — and then walk into a simulated interview room and solve real problems out loud.

A quick word on *why this matters for the job*, not just the interview. Quant trading and research run on expected value. Pricing a bet, sizing a position, estimating the chance a market-making strategy gets picked off — these are counting-and-probability calculations done quickly and correctly under uncertainty. The interview tests counting because the job *is* counting. Everything here is educational; none of it is financial advice.

## Foundations: sample spaces, equally-likely outcomes, and "favorable over total"

Before any technique, we need three plain-English ideas. Define them once and the rest of the post is arithmetic.

**An outcome** is one specific result of a random experiment. Roll one die: the outcomes are 1, 2, 3, 4, 5, 6. Flip one coin: heads or tails.

**The sample space** is the set of *all* possible outcomes. For one die it is ${1, 2, 3, 4, 5, 6}$ — six outcomes. The sample space is the universe of "everything that could happen."

**An event** is any collection of outcomes we care about — a subset of the sample space. "The die shows an even number" is the event ${2, 4, 6}$. "The card is a heart" is the event of all 13 hearts.

Here is the one rule that powers the whole post. When every outcome is **equally likely** — each is exactly as probable as any other — the probability of an event is just a ratio of counts:

$$
P(\text{event}) = \frac{\text{number of favorable outcomes}}{\text{number of total outcomes}}
$$

- *Favorable outcomes* = the outcomes inside the event you want.
- *Total outcomes* = the size of the whole sample space.

Roll one fair die. `P(even) = 3 / 6 = 1/2`, because three of the six equally-likely faces (2, 4, 6) are even. That is the entire idea. Every problem below is some clever way of **counting the top and the bottom of that fraction.**

The phrase *equally likely* is load-bearing. The formula only works when the outcomes you are counting are equiprobable. A classic trap: "a coin is flipped twice; the outcomes are zero heads, one head, two heads — so `P(two heads) = 1/3`." Wrong, because those three outcomes are *not* equally likely. The genuinely equally-likely outcomes are the ordered pairs `HH, HT, TH, TT`, each with probability `1/4`. One head happens in two of those four (`HT`, `TH`), so `P(one head) = 2/4 = 1/2`. **Always count at the level where outcomes are equally likely.** When in doubt, make the outcomes ordered and distinguishable — that almost always restores equal likelihood.

### A picture of a sample space

Two dice make this concrete. Each die is equally likely to land on 1 through 6, and the two dice are independent, so every ordered pair `(d1, d2)` is equally likely. That is a 6-by-6 grid of 36 outcomes.

![Sample space for two dice: a 6 by 6 grid of 36 equally likely outcomes, with the sum-7 diagonal highlighted](/imgs/blogs/counting-combinatorics-quant-interviews-2.png)

The grid *is* the sample space. Want `P(sum = 7)`? Count the green diagonal: `(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)` — six cells. So `P(sum = 7) = 6 / 36 = 1/6`. Want `P(sum = 2)`? Only the red cell `(1,1)` works: `1/36`. Want `P(sum = 12)`? Only `(6,6)`: `1/36`. The whole skill of this post is building grids like this *without drawing them* — counting the favorable cells and the total cells with a formula instead of a picture, because real problems have millions of cells.

One more foundational habit: **decide whether outcomes are ordered or unordered, and whether items are distinguishable, before you count.** Half of all counting errors trace back to fuzzing this. The two dice above are *ordered* — `(2,5)` and `(5,2)` are different cells — which is exactly why the 36 outcomes are equally likely. We will return to this distinction constantly.

### How this shows up in interviews

Interviewers rarely say "use the multiplication principle." They hand you a story — cards, dice, coins, people in a room, balls in urns — and watch how you set it up. The tell that a problem is *counting in disguise* is that it asks for the probability of a clearly-defined event over a finite, symmetric sample space: a shuffled deck, fair dice, random seatings, random assignments. When you hear those, your inner monologue should immediately become:

1. **What is the sample space, and are its outcomes equally likely?** If not, refine it (make outcomes ordered/distinguishable) until they are.
2. **What is the total count (denominator)?** Usually a clean product, factorial, or binomial coefficient.
3. **What is the favorable count (numerator)?** Pick the matching tool from the map.
4. **Divide, then sanity-check** the magnitude (is it between 0 and 1, and does it feel right?).

Graders are scoring your *process*, not just the final number. A candidate who says "let me define the sample space first" and narrates each count has already differentiated themselves from one who lunges at a formula. Throughout the interview section we'll model that narration explicitly.

A second thing interviewers probe: **can you re-derive a formula you forgot?** Nobody will fault you for blanking on the derangement formula, but they will absolutely notice if you can rebuild it from inclusion-exclusion on the spot. That is why this post derives everything rather than asking you to memorize it — under pressure, a derivation you understand beats a formula you half-remember.

## The product rule: independent choices multiply

The product rule (also called the multiplication principle) is the workhorse. It says:

> If a process has **stage 1 with `a` choices**, then **stage 2 with `b` choices** (no matter what you picked in stage 1), then **stage 3 with `c` choices**, and so on, the total number of ways to complete the whole process is `a × b × c × …`.

The intuition is a tree. Each first-stage choice branches into all the second-stage choices, and so on. Count the leaves.

![The product rule as a tree: 3 shirts times 2 pants times 2 shoes gives 12 outfits](/imgs/blogs/counting-combinatorics-quant-interviews-3.png)

#### Worked example: how many outfits?

You own 3 shirts, 2 pairs of pants, and 2 pairs of shoes. How many distinct outfits (one of each)?

- Stage 1 — pick a shirt: 3 choices.
- Stage 2 — pick pants: 2 choices, regardless of the shirt.
- Stage 3 — pick shoes: 2 choices, regardless of the rest.

Total: `3 × 2 × 2 = 12` outfits. The tree above shows exactly this: 3 first-level branches, each splitting into `2 × 2 = 4` leaves, for `3 × 4 = 12` leaves. *The intuition: independent stages multiply.*

The product rule scales effortlessly to large numbers, which is where it earns its keep.

#### Worked example: counting license plates and PINs

A license plate is 3 letters followed by 3 digits. How many plates exist? Each of the 3 letter slots has 26 choices and each of the 3 digit slots has 10:

$$
26 \times 26 \times 26 \times 10 \times 10 \times 10 = 26^3 \times 10^3 = 17{,}576{,}000
$$

A 4-digit PIN where digits can repeat: `10 × 10 × 10 × 10 = 10^4 = 10,000` PINs — which is exactly why a 4-digit PIN is guessable and a 6-digit one (`10^6 = 1,000,000`) is a hundred times harder. *The intuition: every independent slot you add multiplies the count by the slot's size.*

The product rule also produces our first probability number directly. What is the chance a random 4-digit PIN is `1234`? There is one favorable outcome and `10,000` total, so `1 / 10,000`. Favorable over total — counting both sides.

### With replacement vs without: the one distinction that changes the count

When you fill slots one at a time, the crucial question is whether earlier choices *remove* options from later ones. This is the **with-replacement vs without-replacement** distinction, and naming it explicitly prevents a whole class of errors.

| Scenario | Each slot's choices | Count of `k` slots from `n` items | Example |
|---|---|---|---|
| **With replacement, order matters** | always `n` | `n^k` | PINs (`10^4`), repeated dice rolls (`6^k`) |
| **Without replacement, order matters** | `n, n-1, n-2, …` | $P(n,k) = \frac{n!}{(n-k)!}$ | medals to runners (`8·7·6`), drawing ordered cards |
| **Without replacement, order doesn't matter** | — | $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ | poker hands ($\binom{52}{5}$), committees |
| **With replacement, order doesn't matter** | — | $\binom{n+k-1}{k}$ | multisets, ice-cream scoops, stars-and-bars |

The PIN counted `10^4` because digits repeat (with replacement). The medals counted `8·7·6` because a runner can't win two medals (without replacement). The poker hand divides by `k!` because order is irrelevant. The bottom row — with replacement but unordered — is secretly stars-and-bars, which we'll reach shortly. *The intuition: before you multiply, ask "does using an item remove it from the pool, and do I care about order?" — those two yes/no answers pick your formula.*

## The sum rule: disjoint cases add

The sum rule is the product rule's quieter sibling. It says:

> If an event can happen in **case A** (with `a` outcomes) **or** **case B** (with `b` outcomes), and the cases **cannot both happen at once** (they are *disjoint*, or *mutually exclusive*), then the total number of outcomes is `a + b`.

The keyword that triggers the sum rule is "or," and the critical condition is *disjoint* — no outcome belongs to two cases. When cases overlap, plain addition double-counts, and you need inclusion-exclusion (later). When they don't overlap, you just add.

![The sum rule: split a card-counting question into three non-overlapping cases and add the counts](/imgs/blogs/counting-combinatorics-quant-interviews-4.png)

#### Worked example: a face card or a red card

Draw one card from a standard 52-card deck. (Reminder for the uninitiated: 52 cards = 4 suits — hearts and diamonds are red, clubs and spades are black — each suit having 13 ranks: Ace, 2–10, Jack, Queen, King. The Jack, Queen, King are the "face cards.") What is `P(face card OR red card)`?

The trap is to compute `P(face) + P(red)` and double-count the red face cards. Instead, **split into disjoint cases** so nothing is counted twice:

- **Case A — black face cards:** Jack, Queen, King in clubs and spades = `3 × 2 = 6` cards.
- **Case B — red face cards:** Jack, Queen, King in hearts and diamonds = `3 × 2 = 6` cards.
- **Case C — red non-face cards:** Ace through 10 in hearts and diamonds = `10 × 2 = 20` cards.

These three cases never overlap (a card is in exactly one), and together they are precisely "face OR red." So the count is `6 + 6 + 20 = 32`, and

$$
P(\text{face OR red}) = \frac{32}{52} = \frac{8}{13} \approx 61.5\%.
$$

*The intuition: when cases can't overlap, counts simply add — the art is carving the event into clean, non-overlapping pieces.* Mastering the "split into disjoint cases" move is half of interview probability. Whenever you feel an "or" question getting tangled, partition it.

## Permutations vs combinations: does order matter?

This is the distinction that decides most counting problems, and the one candidates most often get backwards. The question to always ask: **does the order of the chosen items matter?**

- If order **matters** — `AB` is different from `BA` — you want **permutations** (ordered arrangements).
- If order **does not matter** — `AB` and `BA` are the same selection — you want **combinations** (unordered subsets).

![Permutations vs combinations: picking 2 of 3 gives 6 ordered permutations but only 3 unordered combinations](/imgs/blogs/counting-combinatorics-quant-interviews-5.png)

### Permutations: counting ordered arrangements

A **permutation** of `k` items chosen from `n` distinct items is an *ordered* selection. Build it with the product rule: the first slot has `n` choices, the second has `n − 1` (one item is used up), the third `n − 2`, and so on for `k` slots:

$$
P(n, k) = n \times (n-1) \times \cdots \times (n-k+1) = \frac{n!}{(n-k)!}
$$

Here `n!` ("n factorial") means `n × (n−1) × … × 2 × 1`, the number of ways to order all `n` items. By convention `0! = 1`.

#### Worked example: arranging books and awarding medals

How many ways to line up 5 distinct books on a shelf? That is `P(5,5) = 5! = 120`. (First slot 5 choices, then 4, 3, 2, 1.)

How many ways to award gold, silver, and bronze medals among 8 sprinters? Order matters — gold-then-silver is a different result from silver-then-gold — so it is `P(8,3) = 8 × 7 × 6 = 336`. *The intuition: each filled slot removes one option from the next, so you multiply a descending run of factors.*

### Combinations: counting unordered selections

A **combination** chooses `k` items from `n` where order is irrelevant — you want the *set*, not the arrangement. The trick to get the formula: count the ordered version, then **divide out the orderings you don't care about.** Every unordered set of `k` items can be arranged in `k!` orders, so:

$$
C(n, k) = \binom{n}{k} = \frac{P(n,k)}{k!} = \frac{n!}{k!\,(n-k)!}
$$

The symbol $\binom{n}{k}$ is read "n choose k." In the figure above, picking 2 from ${A, B, C}$ gives `P(3,2) = 6` ordered pairs, but each unordered set (like ${A,B}$) shows up `2! = 2` times (as `AB` and `BA`), so there are `6 / 2 = 3` combinations. That division — *strip out the orderings that don't matter* — is the single most important idea in combinatorics and the engine behind every symmetry argument later.

#### Worked example: choosing a committee and a poker hand

How many 3-person committees can you form from 10 people? Order does not matter (a committee of Alice, Bob, Carol is the same as Carol, Bob, Alice), so:

$$
\binom{10}{3} = \frac{10 \times 9 \times 8}{3 \times 2 \times 1} = \frac{720}{6} = 120.
$$

How many distinct 5-card poker hands from a 52-card deck? A hand is an unordered set of 5 cards, so:

$$
\binom{52}{5} = \frac{52 \times 51 \times 50 \times 49 \times 48}{5!} = \frac{311{,}875{,}200}{120} = 2{,}598{,}960.
$$

Hold onto that number — `2,598,960` total hands — because it is the denominator for every poker-probability question in an interview. *The intuition: combinations are permutations with the irrelevant orderings divided away.*

> A fast sanity check you should internalize: $\binom{n}{k} = \binom{n}{n-k}$. Choosing which `k` to *include* is the same as choosing which `n − k` to *exclude*. So $\binom{52}{5} = \binom{52}{47}$, and $\binom{10}{3} = \binom{10}{7}$. This symmetry both saves arithmetic and catches errors.

## Binomial coefficients and Pascal's triangle

The numbers $\binom{n}{k}$ are called **binomial coefficients**, and they have a structure worth knowing cold because it lets you compute small ones instantly and reason about them under pressure.

Stack them in a triangle: row `n` holds $\binom{n}{0}, \binom{n}{1}, \dots, \binom{n}{n}$. Every interior entry equals the sum of the two entries directly above it. This is **Pascal's triangle.**

![Pascal's triangle: each entry is the sum of the two above it, and row n sums to 2 to the n](/imgs/blogs/counting-combinatorics-quant-interviews-6.png)

Two facts from the picture earn their keep in interviews:

1. **The addition rule:** $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$. In the figure, the `4` and `6` in row 4 add to the `10 = C(5,2)` in row 5. The combinatorial reason: to choose `k` of `n` items, either you include the last item (then choose `k−1` of the remaining `n−1`) or you exclude it (choose `k` of the remaining `n−1`). Those two cases are disjoint and cover everything — the sum rule again.
2. **Row sums are powers of two:** the entries in row `n` add to `2^n`. Why? $\sum_k \binom{n}{k}$ counts *all* subsets of an `n`-element set (subsets of size 0, plus size 1, plus … plus size `n`), and there are `2^n` subsets total — each element is independently in or out, so `2 × 2 × … = 2^n` by the product rule.

#### Worked example: the binomial theorem in one line

These coefficients are exactly the ones in the algebra identity

$$
(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}.
$$

For `n = 5`: `(x+y)^5 = x^5 + 5x^4y + 10x^3y^2 + 10x^2y^3 + 5xy^4 + y^5` — the coefficients `1, 5, 10, 10, 5, 1` are read straight off row 5 of the triangle. This is why they are called *binomial* coefficients. It also gives a clean probability hook: flip a fair coin `n` times, and $P(exactly k heads) = \binom{n}{k} / 2^n$. For 5 flips, `P(exactly 2 heads) = 10 / 32 = 31.25%`. *The intuition: $\binom{n}{k}$ counts the number of ways to place `k` "successes" among `n` trials.*

## Stars and bars: distributing identical items

Here is a problem the basic tools don't obviously handle: **how many ways can you distribute identical items into distinct bins?** The items are interchangeable; only *how many land in each bin* matters.

The trick — and it is one of the most beautiful in combinatorics — is to represent a distribution as a row of symbols: a **star** for each item, and a **bar** for each divider between bins. With `k` bins you need `k − 1` bars to separate them.

![Stars and bars: distributing 7 identical units into 3 bins maps to arranging 7 stars and 2 bars](/imgs/blogs/counting-combinatorics-quant-interviews-7.png)

In the figure, `**|***|**` means "2 in bin 1, 3 in bin 2, 2 in bin 3." Every arrangement of 7 stars and 2 bars is a valid distribution, and every distribution corresponds to exactly one arrangement. So we just need to count arrangements of `7 + 2 = 9` symbols where we choose which 2 of the 9 positions are bars:

$$
\binom{9}{2} = \frac{9 \times 8}{2} = 36.
$$

The general formula: distributing `n` identical items into `k` distinct bins (bins allowed to be empty) is

$$
\binom{n + k - 1}{\,k - 1\,}.
$$

#### Worked example: dice outcomes summing to a target

Stars-and-bars cracks a classic interview question. **In how many ways can three dice show a sum of 10?** Each die shows 1 to 6. Write `d1 + d2 + d3 = 10` with each `di ≥ 1`.

First, remove the "at least 1" floor by substituting `ei = di − 1`, so each `ei ≥ 0` and `e1 + e2 + e3 = 7`. That is distributing 7 identical units into 3 bins: $\binom{7 + 3 - 1}{3 - 1} = \binom{9}{2} = 36$ — the exact figure above.

But we have over-counted: stars-and-bars allows a bin to exceed 5 (i.e. `ei > 5`, meaning `di > 6`), which a die can't do. Subtract those. If `e1 ≥ 6`, set `f1 = e1 − 6 ≥ 0`, giving `f1 + e2 + e3 = 1`, which has $\binom{1 + 3 - 1}{2} = \binom{3}{2} = 3$ solutions. The same happens for `e2` and `e3`, and two bins can't both exceed 5 (that would need at least 12 > 7), so by the sum rule we subtract `3 × 3 = 9` illegal arrangements:

$$
36 - 9 = 27 \text{ ways}.
$$

So `P(three dice sum to 10) = 27 / 216 = 1/8 = 12.5%` (the denominator is `6^3 = 216`). *The intuition: stars-and-bars counts non-negative integer solutions of a sum, and you patch the upper limits with a correction.* That correction is itself an inclusion-exclusion — which is our next tool.

## Inclusion-exclusion: when "or" events overlap

The sum rule needed disjoint cases. What if the cases *overlap*? Then plain addition counts the overlap twice, and you must subtract it back out. This is the **inclusion-exclusion principle.** For two sets:

$$
|A \cup B| = |A| + |B| - |A \cap B|
$$

In words: the size of "A or B" is the size of A plus the size of B, minus the size of "A and B" (which was counted in both `|A|` and `|B|`).

![Inclusion-exclusion: the union equals the sum of the sets minus their double-counted overlap](/imgs/blogs/counting-combinatorics-quant-interviews-8.png)

#### Worked example: numbers divisible by 2 or 5

How many integers from 1 to 100 are divisible by 2 or 5?

- `|A| =` divisible by 2 `= 50`.
- `|B| =` divisible by 5 `= 20`.
- `|A ∩ B| =` divisible by both, i.e. by 10 `= 10`.

If you naively did `50 + 20 = 70`, you'd count the multiples of 10 (which are in both sets) twice. Correcting:

$$
|A \cup B| = 50 + 20 - 10 = 60.
$$

The figure decomposes it cleanly: 40 numbers divisible by 2 but not 5, plus 10 divisible by both, plus 10 divisible by 5 but not 2, totals 60. *The intuition: add the sets, then subtract every overlap exactly once so nothing is double-counted.*

For three sets the pattern continues — add singles, subtract pairs, add the triple:

$$
|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|.
$$

#### Worked example: divisible by 2, 3, or 5

How many integers from 1 to 1000 are divisible by 2, 3, or 5? Count each set with the floor function `⌊1000/d⌋` (the number of multiples of `d` up to 1000):

- Singles: `⌊1000/2⌋ = 500`, `⌊1000/3⌋ = 333`, `⌊1000/5⌋ = 200`.
- Pairs (divisible by the product): `⌊1000/6⌋ = 166`, `⌊1000/10⌋ = 100`, `⌊1000/15⌋ = 66`.
- Triple: `⌊1000/30⌋ = 33`.

Apply the alternating signs:

$$
500 + 333 + 200 - 166 - 100 - 66 + 33 = 734.
$$

So 734 of the first 1000 integers are divisible by at least one of 2, 3, 5 — and `1000 − 734 = 266` are divisible by none (these are the integers coprime to 30). *The intuition: inclusion-exclusion scales to any number of overlapping sets — add odd-sized intersections, subtract even-sized ones.*

The alternating `+ − + − …` signs are the heart of inclusion-exclusion, and they show up most dramatically in the derangement problem, which we'll meet in the interview section.

### The complement: count the opposite when it's easier

One more idea belongs here because it pairs so often with inclusion-exclusion: **counting the complement.** If the event you want is messy ("at least one X") but its opposite is clean ("zero X"), count the opposite and subtract from the total:

$$
P(\text{at least one}) = 1 - P(\text{none}).
$$

"At least one" is a red flag that screams *use the complement.* Example: roll 4 dice — what's the probability of *at least one* six? Counting "at least one six" directly forces messy cases (exactly one, exactly two, …). The complement is trivial: the probability of *no* six on a single die is `5/6`, and the four dice are independent, so $P(\text{no six}) = (5/6)^4 = 625/1296$. Therefore

$$
P(\text{at least one six}) = 1 - \frac{625}{1296} = \frac{671}{1296} \approx 51.8\%.
$$

The birthday problem in the interview section is the same move on a grand scale: "at least one shared birthday" is hard, "all distinct" is a clean product. *The intuition: whenever you see "at least one," check whether counting "none" and subtracting is dramatically easier — it usually is.*

## Overcounting and symmetry: divide out what you double-counted

If there is one meta-skill that separates a clean count from a wrong one, it is recognizing **overcounting** and fixing it by **dividing by symmetry.** We've already used it twice (combinations divide permutations by `k!`; the coin problem made outcomes distinguishable). Now we name it.

The recipe: **count as if everything is distinguishable and ordered (easy with the product rule), then divide by the number of arrangements you treated as different but should treat as the same.**

The canonical interview setting is **circular arrangements.**

![Circular seating: row arrangements collapse to fewer distinct circles because rotations are equivalent](/imgs/blogs/counting-combinatorics-quant-interviews-9.png)

#### Worked example: seating people around a round table

In how many distinct ways can 4 people sit around a round table? In a *row*, the answer is `4! = 24`. But around a round table, only the *relative* order matters — rotating everyone one seat clockwise gives the "same" arrangement. Each distinct circular arrangement corresponds to `4` rotations of a row (one for each starting seat). So we divide:

$$
\frac{4!}{4} = \frac{24}{4} = 6 \text{ distinct circular arrangements}.
$$

The general result: `n` people around a round table can be seated in `(n − 1)!` distinct ways. *The intuition: fix one person to break the rotational symmetry, then arrange the other `n − 1` in a line — or equivalently, count `n!` orderings and divide by the `n` rotations that are secretly the same.*

A second flavor of overcounting: **identical items.** How many distinct arrangements of the letters in `BANANA`? If all 6 letters were distinct it would be `6! = 720`. But there are 3 identical A's (which can be permuted `3! = 6` ways without changing the word) and 2 identical N's (`2! = 2` ways). Divide out both:

$$
\frac{6!}{3!\,2!\,1!} = \frac{720}{6 \times 2 \times 1} = 60.
$$

This "divide by the factorials of the repeats" rule (the *multinomial coefficient*) is just the same symmetry logic applied to repeated objects. Whenever your raw product-rule count treats two truly-identical outcomes as different, divide by the size of that symmetry group.

## In the interview room

Now we put it together the way a real interview unfolds: a problem stated in plain English, and a clean, narrated count to the answer. The thing being graded is *your reasoning out loud* — define the sample space, count the denominator, count the numerator, sanity-check. Several of these end with the move that matters most on a desk: attaching a dollar payoff to the probability and reading off the expected value.

### Problem 1 — Probability of a flush (and a full house)

**"You're dealt 5 cards from a standard deck. What's the probability of a flush?"**

A **flush** is 5 cards all of the same suit (and, for this count, not a straight flush — but interviewers usually accept either; state your assumption).

**Denominator.** Total 5-card hands: $\binom{52}{5} = 2,598,960$. (Order doesn't matter in a poker hand, so combinations.)

**Numerator.** Build a flush with the product rule:
- Choose the suit: $\binom{4}{1} = 4$ ways.
- Choose 5 of that suit's 13 cards: $\binom{13}{5} = 1287$ ways.

That gives `4 × 1287 = 5148` flushes *including* straight flushes. There are `40` straight flushes (10 per suit — A-2-3-4-5 up to 10-J-Q-K-A — times 4 suits), so flushes excluding them: `5148 − 40 = 5108`.

$$
P(\text{flush}) = \frac{5108}{2{,}598{,}960} \approx 0.00197 = 0.197\%.
$$

To make that number tangible, suppose a friend offers you \$3 every time you're dealt a flush if you stake \$1 each hand. With `P ≈ 0.197%` the bet is wildly in your friend's favor: a fair payout would be about \$508 (one over the probability), not \$3 — a first taste of how a probability becomes a price. We'll do that conversion properly in two later problems.

The figure below ranks every hand by how many ways it can be made — and rarity is exactly what makes a hand beat another.

![Five-card poker hands ranked by count: rarer hands sit lower, and a full house is rarer than a flush](/imgs/blogs/counting-combinatorics-quant-interviews-10.png)

**Follow-up they always ask: "Why does a full house beat a flush?"** Because a full house is *rarer*. Count it: choose the rank for the three-of-a-kind (`13` ways), choose 3 of its 4 suits ($\binom{4}{3} = 4$), choose a different rank for the pair (`12` ways), choose 2 of its 4 suits ($\binom{4}{2} = 6$):

$$
13 \times 4 \times 12 \times 6 = 3744 \text{ full houses}, \quad P = \frac{3744}{2{,}598{,}960} \approx 0.144\%.
$$

Since `3744 < 5108`, a full house is harder to make, so it ranks higher. *The poker hand ranking is literally a sorted list of these counts — the whole game's hierarchy is combinatorics.*

### Problem 2 — The birthday problem

**"How many people need to be in a room before there's a better-than-even chance that two share a birthday?"**

Most people guess 180-ish (half of 365). The answer is **23**, and the gap between intuition and truth is what makes it a perennial favorite.

The move that makes it tractable is the **complement**: instead of counting the messy event "at least one shared birthday," count its opposite, "*all* birthdays distinct," and subtract from 1. (Ignore leap years; assume 365 equally likely birthdays.)

For `n` people, count using the product rule with everyone distinguishable:
- Total ways to assign birthdays: `365^n` (each person independently has 365 choices).
- Ways for all `n` to be *distinct*: `365 × 364 × 363 × … × (365 − n + 1)` — the first person has 365 free days, the second must avoid 1, the third must avoid 2, and so on.

So

$$
P(\text{all distinct}) = \frac{365 \times 364 \times \cdots \times (365 - n + 1)}{365^n}, \qquad P(\text{shared}) = 1 - P(\text{all distinct}).
$$

Plug in `n = 23`:

$$
P(\text{all distinct}) = \frac{365}{365}\cdot\frac{364}{365}\cdots\frac{343}{365} \approx 0.4927,
$$

so `P(at least one shared) ≈ 1 − 0.4927 = 0.5073 = 50.7%` — just over half. At `n = 22` it's `47.6%` (under half), so 23 is the crossover.

![The birthday problem: the probability of a shared birthday crosses 50 percent at only 23 people](/imgs/blogs/counting-combinatorics-quant-interviews-11.png)

**Why so few?** Because the relevant quantity is the number of *pairs*, not people. With `n` people there are $\binom{n}{2}$ pairs, and $\binom{23}{2} = 253$ pairs — each a chance for a match. The growth is quadratic in `n`, so collisions arrive far faster than linear intuition expects. By `n = 70` the probability is `99.9%`. *The intuition: count pairs, not people — $\binom{n}{2}$ grows fast, so coincidences are the norm, not the exception.* (This same "collisions are common" math underlies hash-collision attacks and why trading systems need wide-enough unique IDs.)

### Problem 3 — Anagrams with repeated letters

**"How many distinct arrangements are there of the letters in the word `MISSISSIPPI`?"**

Count the letters: `M` ×1, `I` ×4, `S` ×4, `P` ×2 — eleven letters total. If all 11 were distinct, the answer would be `11!`. But the four I's are interchangeable (permuting them among themselves changes nothing), as are the four S's and the two P's. Divide out every block of identical letters:

$$
\frac{11!}{1!\,4!\,4!\,2!} = \frac{39{,}916{,}800}{1 \times 24 \times 24 \times 2} = \frac{39{,}916{,}800}{1152} = 34{,}650.
$$

*The intuition: overcounting from identical items is fixed by dividing by the factorial of each repeat count.* If the interviewer pushes — "now how many arrangements have all four S's together?" — glue the four S's into one super-letter `[SSSS]`, leaving 8 objects (`M, I, I, I, I, P, P, [SSSS]`) to arrange: $\frac{8!}{4!\,2!} = \frac{40320}{48} = 840$. Treating a constrained block as a single unit, then counting the rest, is a standard follow-up technique.

### Problem 4 — The hat-check (derangement) problem

**"Four people check their hats. The clerk loses the tickets and returns the four hats at random, one to each person. What's the probability that *nobody* gets their own hat back?"**

A permutation where *no* element stays in its original position is called a **derangement**. We want the number of derangements of 4 items, written `D₄`, divided by the total `4! = 24` permutations.

Counting `D₄` directly is awkward, so use inclusion-exclusion on the "bad" events. Let `Aᵢ` be the event "person `i` gets their own hat." We want permutations in *none* of the `Aᵢ`, i.e. `4! − |A₁ ∪ A₂ ∪ A₃ ∪ A₄|`. By inclusion-exclusion:

- **At least one person correct:** choose which person ($\binom{4}{1}$) and arrange the rest freely (`3!`): $\binom{4}{1}\cdot 3! = 4 × 6 = 24$.
- **At least two correct:** $\binom{4}{2}\cdot 2! = 6 × 2 = 12$. (Subtract, because pairs were double-counted.)
- **At least three correct:** $\binom{4}{3}\cdot 1! = 4 × 1 = 4$. (Add back.)
- **All four correct:** $\binom{4}{4}\cdot 0! = 1 × 1 = 1$. (Subtract.)

The alternating signs give the size of the union, and the derangement count is `4!` minus it:

$$
D_4 = 4! - 24 + 12 - 4 + 1 = 24 - 24 + 12 - 4 + 1 = 9.
$$

![Counting derangements of 4 items by inclusion-exclusion: 24 minus 24 plus 12 minus 4 plus 1 equals 9](/imgs/blogs/counting-combinatorics-quant-interviews-12.png)

So `P(nobody gets their hat) = 9 / 24 = 37.5%`. To price the result: if a colleague bets you \$1 that *someone* gets their own hat, the fair payoff to you on the "nobody" side is the stake divided by `0.375`, about \$2.67 — pay less than that and the bet tilts your way, a clean illustration of turning a counted probability into odds.

**The elegant punchline interviewers love:** the derangement formula is $D_n = n!\left(1 - \frac{1}{1!} + \frac{1}{2!} - \frac{1}{3!} + \cdots\right)$, and the term in parentheses is the Taylor series for $e^{-1}$. So `D_n / n! → 1/e ≈ 0.3679` as `n` grows. **No matter how many people check hats, the probability that nobody gets their own is essentially `1/e ≈ 36.8%`** — astonishingly, it barely depends on `n`. (For `n = 4` it's `37.5%`; for `n = 5` it's `44/120 = 36.67%`; it converges fast.) Knowing that `1/e` shows up here is the kind of fact that signals you've seen the territory.

### Problem 5 — Lattice paths on a grid

**"You're on a grid. Starting at the bottom-left corner, you can only move right or up. How many shortest paths reach a point 4 steps right and 3 steps up?"**

Any shortest path is a sequence of exactly `4` rights (`R`) and `3` ups (`U`) — total `7` moves. A path is determined by *which* of the 7 moves are the rights. So choose 4 positions out of 7 for the R's:

$$
\binom{7}{4} = \frac{7!}{4!\,3!} = 35 \text{ paths}.
$$

Equivalently, it's the number of distinct arrangements of `RRRRUUU` — the anagram view — which is $\frac{7!}{4!3!} = 35$. *The intuition: a monotone lattice path is just a word in two letters, so counting paths is counting arrangements.* Lattice-path counting is a favorite because it connects to combinations, anagrams, *and* Pascal's triangle (each grid point's path-count is a binomial coefficient).

### Problem 6 — Two aces in a five-card hand (counting meets conditional probability)

**"You're dealt 5 cards. What's the probability you get exactly two aces?"**

This problem looks like it needs conditional probability, but it's a pure count — and it shows the *favorable-over-total* recipe at its cleanest.

**Denominator.** All 5-card hands: $\binom{52}{5} = 2,598,960$.

**Numerator.** Build a hand with exactly two aces using the product rule on two independent choices:
- Choose which 2 of the 4 aces you hold: $\binom{4}{2} = 6$.
- Choose the other 3 cards from the 48 non-ace cards: $\binom{48}{3} = 17,296$.

So the count is `6 × 17,296 = 103,776`, and

$$
P(\text{exactly two aces}) = \frac{103{,}776}{2{,}598{,}960} \approx 0.0399 = 3.99\%.
$$

The structure — *choose the special cards, then choose the rest* — is the template for "exactly `k` of a kind" questions across cards, lottery, and quality-control sampling. *The intuition: split the hand into the part you constrain and the part you fill freely, count each with a binomial coefficient, and multiply.*

A revealing follow-up: **"Now, the probability that the *first two* cards dealt are aces?"** That's a different, ordered question. Dealing in order, $P = \frac{4}{52}\cdot\frac{3}{51} = \frac{12}{2652} = \frac{1}{221} \approx 0.45\%$. Notice this is much smaller than the 3.99% above, because "exactly two aces somewhere in five cards" is a far broader event than "aces in the first two specific positions." Spotting whether the question fixes positions (ordered) or just counts a set (unordered) is the whole game.

### Problem 7 — The Monty-Hall-style counting check

**"In a lottery you pick 6 numbers from 1 to 49. What's the probability you match all 6?"**

Order doesn't matter (the balls are drawn and sorted), and there's no replacement, so the total number of tickets is $\binom{49}{6}$:

$$
\binom{49}{6} = \frac{49 \times 48 \times 47 \times 46 \times 45 \times 44}{6!} = \frac{10{,}068{,}347{,}520}{720} = 13{,}983{,}816.
$$

Exactly one ticket matches all six, so $P(\text{jackpot}) = 1 / 13{,}983{,}816 \approx 7.15 \times 10^{-8}$ — about 1 in 14 million. **Follow-up: "What's the probability of matching exactly 5 of the 6?"** Choose 5 of the 6 winning numbers ($\binom{6}{5} = 6$) and 1 of the 43 losing numbers ($\binom{43}{1} = 43$):

$$
P(\text{match 5}) = \frac{\binom{6}{5}\binom{43}{1}}{\binom{49}{6}} = \frac{6 \times 43}{13{,}983{,}816} = \frac{258}{13{,}983{,}816} \approx 1.85 \times 10^{-5}.
$$

*The intuition: "match exactly `j` of the winners" is — once again — choose `j` from the winning pool and the rest from the losing pool, over the total.* This "winners pool × losers pool / total" structure is the **hypergeometric** count, and it's the same shape as Problem 6's two aces. Recognizing that one template solves cards, lottery, defective-widget sampling, and audit problems is exactly the pattern-matching interviewers are testing.

#### Worked example: the expected value of a lottery ticket

**"A lottery ticket costs \$2. The jackpot is \$10,000,000, paid only if you match all 6 of 6 numbers from 1 to 49. Should you buy? And how big would the jackpot have to be before the bet is fair?"**

This is the question that turns the counting from Problem 7 into a decision — and it is the *exact* shape of every "should I take this trade?" question a desk asks. We already counted the denominator: there are $\binom{49}{6} = 13{,}983{,}816$ equally-likely tickets, and exactly one matches all six, so the win probability is `p = 1 / 13,983,816`.

**Step 1 — expected payout (gross).** Expected value of a bet is *probability × payoff*, summed over outcomes. Here there is one paying outcome — the jackpot — so the expected gross winnings of a single ticket are

$$
\mathbb{E}[\text{winnings}] = p \times \$10{,}000{,}000 = \frac{10{,}000{,}000}{13{,}983{,}816} \approx \$0.715.
$$

In words: averaged over millions of tickets, each one returns about **72 cents** of jackpot.

**Step 2 — net expected value (subtract the cost).** You paid \$2 to play, so the expected *profit* per ticket is

$$
\mathbb{E}[\text{net}] = \$0.715 - \$2 = -\$1.285.
$$

Every \$2 ticket loses about **\$1.29 on average** — the house (or the state) keeps roughly 64% of every dollar wagered. That negative EV is *why* it's a tax-on-hope, not an investment, and it's the same sign a quant looks for in reverse: a desk only wants bets whose net EV is positive after costs.

**Step 3 — the breakeven jackpot.** When does the ticket become a *fair* bet — net EV exactly zero? Set expected winnings equal to the \$2 cost and solve for the jackpot `J`:

$$
p \times J = \$2 \;\Longrightarrow\; J = 2 \times 13{,}983{,}816 = \$27{,}967{,}632.
$$

So the jackpot must exceed roughly **\$28 million** before a \$2 ticket even breaks even on expectation — and that *still* ignores taxes, the lump-sum discount, and the chance of *splitting* the pot with another winner. That last effect is itself a counting problem worth a sentence, because it's the trap that fools people who *do* know EV. If `m` other tickets are sold and each independently has win probability `p`, the number who *also* hit the jackpot is roughly Poisson with mean `λ = m·p`, and your expected share of a fixed jackpot is *not* the full \$10,000,000 — it's the jackpot times `E[1/(1 + K)]` where `K` is the number of *other* winners. When a huge jackpot draws hundreds of millions of tickets, `λ` climbs past 1, the expected split can halve or third your prize, and the breakeven jackpot you just computed roughly *doubles*. The deeper point: a naive EV that ignores how *your own demand changes the payoff* is exactly the mistake a desk punishes — your fill moves the price, your size splits the pot. *The transferable lesson: EV = probability × payoff minus cost; flip the equation to solve for the payoff that makes any bet fair, and you've priced it — but check whether the payoff itself depends on how many others took the same bet.*

#### Worked example: pricing a pay-to-draw poker side bet

**"A casino offers this side bet: pay \$5 and you're dealt 5 cards from a fresh deck. You win \$100 for any flush, and \$5,000 for a straight flush. What's the expected value — is it a good bet?"**

This problem reuses the exact poker counts from Problem 1, so there's no new counting — only the step that matters on a trading desk: *attaching dollars to the probabilities and summing.*

**Step 1 — pull the probabilities from the counts we already have.** Out of $\binom{52}{5} = 2{,}598{,}960$ hands, Problem 1 found `5108` plain flushes (flush, not straight flush) and `40` straight flushes. So

$$
P(\text{flush}) = \frac{5108}{2{,}598{,}960} \approx 0.0019654, \qquad P(\text{straight flush}) = \frac{40}{2{,}598{,}960} \approx 0.0000154.
$$

**Step 2 — weight each payoff by its probability.** Expected gross winnings are the sum of *payoff × probability* over the paying hands:

$$
\mathbb{E}[\text{winnings}] = \$100 \times \frac{5108}{2{,}598{,}960} + \$5000 \times \frac{40}{2{,}598{,}960} = \frac{510{,}800 + 200{,}000}{2{,}598{,}960} = \frac{710{,}800}{2{,}598{,}960} \approx \$0.273.
$$

A neat sanity check on the arithmetic: the two payout pools are `$100 × 5108 = $510,800` and `$5000 × 40 = $200,000`, summing to `$710,800` of total prize money spread across all 2,598,960 equally-likely hands — divide and you get the ~27-cent average directly.

**Step 3 — subtract the cost.** You paid \$5, so

$$
\mathbb{E}[\text{net}] = \$0.273 - \$5 = -\$4.73.
$$

The bet returns about **27 cents** for every **\$5** wagered — a brutal house edge of roughly 95%. *Sanity-check the magnitude:* flushes are a 1-in-509 event and straight flushes 1-in-65,000, so even a \$5,000 top prize can't rescue a \$5 stake. **Follow-up they may push:** "How large must the flush payout be to make the bet fair, holding the \$5,000 straight-flush prize fixed?" Set expected winnings to the \$5 cost and solve for the flush payout `x`:

$$
x \cdot \frac{5108}{2{,}598{,}960} + \$5000 \cdot \frac{40}{2{,}598{,}960} = \$5 \;\Longrightarrow\; x = \frac{5 \times 2{,}598{,}960 - 5000 \times 40}{5108} \approx \$2{,}505.
$$

The flush would need to pay about **\$2,505** — not \$100 — for the side bet to be break-even. One more point a sharp interviewer will probe: *EV is not the whole story — variance is.* Even at a hypothetical fair price, this bet pays \$0 on more than 99.8% of hands and a big lump on the rest, so its outcome distribution is wildly skewed. A desk that took thousands of such bets a day would care about that spread (it drives how much capital must back the position), which is why real pricing carries the EV *and* a risk charge for the variance. The counting gives you the probabilities; turning them into a position size needs both moments. *The transferable lesson: to price any multi-outcome wager, multiply each payoff by its (counted) probability, sum, and compare to the stake — exactly how a desk prices an option's discrete payoff scenarios — then remember that a fair EV with fat variance still needs capital and a margin of safety.*

## Common misconceptions

**"Order never matters for probability."** It matters constantly — and getting it wrong is the most common error. The fix isn't "always use combinations"; it's to count numerator and denominator *at the same level* of ordering. If your total outcomes are ordered (like the 36 ordered dice pairs), your favorable outcomes must be ordered too. Mixing an ordered denominator with an unordered numerator gives nonsense.

**"More people roughly proportionally raises the birthday chance, so you'd need ~180."** No — the chance depends on the number of *pairs*, $\binom{n}{2}$, which grows quadratically. That's why 23 people suffice. Linear intuition badly underestimates how fast coincidences accumulate.

**"You can just add probabilities for an 'or' event."** Only if the events are disjoint. `P(face) + P(red)` overcounts the red face cards. Either split into disjoint cases (sum rule) or subtract the overlap (inclusion-exclusion). Adding overlapping events is the inclusion-exclusion trap.

**"Stars-and-bars handles any distribution problem."** It handles *identical* items into *distinct* bins, with no per-bin cap. The moment items become distinguishable, or bins get upper limits (like a die capped at 6), you need a correction (subtract the illegal overflows) or a different tool entirely. Forgetting the cap is how people get the three-dice-sum count wrong.

**"Big factorials mean you need a calculator."** Almost never in an interview. Cancel aggressively: $\binom{52}{5}$ is $\frac{52·51·50·49·48}{120}$, and the `120` divides cleanly (`50/(5·2) = 5`, `48/(4·3·2)=2`, etc.). Interviewers want to see you *simplify the ratio before multiplying*, not produce a 68-digit `52!`.

**"A probability is the answer; the dollars are a separate step."** On a trading desk the probability is only the *halfway* point — the question is almost always "what's it worth?" The bridge is one line: expected value = Σ (probability × payoff), then subtract the cost. A \$2 lottery ticket with a one-in-14-million shot at \$10,000,000 has gross EV `≈ \$0.72` and net EV `≈ −\$1.28`; the \$5 poker side bet nets `≈ −\$4.73`. The discipline interviewers reward is *not stopping at the probability* — count the chance, attach the payoff, subtract the stake, and only then say whether the bet is good. A candidate who computes `0.197%` for a flush but can't turn it into "the fair payout on a \$1 stake is about \$508" has done half the job.

**"The answer is the formula."** The answer is the *narration*. Saying "I'll count the denominator as all hands, `52 choose 5`; the numerator as: pick a suit 4 ways, pick 5 of its 13 cards, `13 choose 5`; divide" earns the offer. A bare number with no audible reasoning, even a correct one, does not.

## How it shows up in real trading and research

Counting isn't a party trick that vanishes after the interview — it's the substrate of how trading firms reason about risk and edge.

**Expected value and bet sizing.** Every trade is a wager with a probability and a payoff. A market maker quoting a two-sided price is implicitly counting the outcomes where they get adversely selected versus the outcomes where they capture the spread. Sizing that position correctly *is* a favorable-over-total calculation, just with dollars attached. Firms like Optiver and SIG run options-market-making businesses where pricing an exotic payoff means enumerating and weighting scenarios — combinatorics with money on the line.

**Combinatorial explosions in strategy search.** A researcher backtesting feature combinations faces $\binom{n}{k}$ candidate models; understanding how fast that grows (and how many "discoveries" are spurious by chance — the multiple-comparisons problem) is pure counting. With 100 candidate signals and 5-feature models, there are $\binom{100}{5} \approx 75$ million combinations — and at a 5% false-positive rate you'd expect millions of accidentally "significant" backtests. Knowing the count tells you how skeptical to be of any single result.

**The birthday paradox in systems.** The "collisions arrive fast" lesson is everywhere in low-latency infrastructure. Order IDs, hash keys, and random nonces all face birthday-style collision risk: if your ID space is `N`, collisions become likely around $\sqrt{N}$ items, not `N`. A trading system generating millions of order IDs per day needs an ID width chosen with the birthday math in mind, or it will collide far sooner than naive intuition suggests.

**Risk-neutral pricing and lattice models.** The binomial option-pricing model builds a tree of up/down moves and prices a derivative by counting paths — each terminal node's probability is a binomial coefficient over `2^n`, exactly the Pascal's-triangle structure from earlier. The same counting that solves the lattice-path interview problem prices a real option on a binomial tree. If you want to see where this leads, our deep dives on [options theory](/blog/trading/quantitative-finance/options-theory) and [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) pick up exactly where this counting leaves off — Black-Scholes is the continuous-time limit of that binomial path-counting.

**Poker, game theory, and adverse selection.** Many quant traders cut their teeth on poker precisely because the hand-ranking is combinatorics and the betting is decision-making under incomplete information — the same shape as quoting in a market where some counterparties know more than you. The flush-versus-full-house count is the toy version of "how often does the rare, dangerous event happen, and how should I price it?" — a question that recurs every time a firm prices tail risk or calibrates a [volatility surface](/blog/trading/quantitative-finance/volatility-surface).

## When this matters to you and where to go next

If you are preparing for quant interviews, the highest-leverage move is not to memorize formulas — it's to **practice narrating clean counts out loud** until the toolkit becomes reflexive. Re-derive `C(n,k)` from the product rule. Re-derive the round-table answer by dividing out rotations. Do the poker counts until `2,598,960` is muscle memory. When a new problem lands, run the decision map from the first figure: *reframe as a count → does order matter → are items identical → do events overlap?*

Concretely, build the reflex with these drills, each a variation on what we solved:
- Count the other poker hands (two pair, straight, four of a kind) and verify they sort into the standard ranking.
- Redo the birthday problem for "shared *month*" (12 equally-likely buckets) — the crossover drops to just 5 people.
- Extend inclusion-exclusion to three sets: integers 1–1000 divisible by 2, 3, or 5.
- Use stars-and-bars to count the ways to make change for a dollar, then add coin-count caps and correct for them.
- Take any hand count you computed and turn it into a price: pick a stake and a payoff, compute the EV, and solve for the breakeven payoff — the move that converts a flush count into a \$508 fair payout or a \$2 lottery ticket into its \$28 million breakeven jackpot.

For the deeper machinery these techniques feed into, the natural next steps are conditional probability and expectation (how counting becomes decision-making), then the pricing models that turn path-counting into dollar values — see [derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing) for the bridge from counting outcomes to pricing risk under the risk-neutral measure. The counting never goes away; it just gets a payoff attached.

The one sentence to carry out of here: **when a probability problem looks hard, stop trying to be clever about probability and start counting carefully — the favorable outcomes over the total outcomes — with the right tool for the shape of the question.**
