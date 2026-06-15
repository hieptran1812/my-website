---
title: "Quadratic and convex programming in practice: portfolio optimization with real constraints"
date: "2026-06-15"
description: "How quadratic and convex programming turn portfolio optimization into a problem a computer can solve exactly, and how to encode the messy real-world rules a trading desk actually lives by."
tags: ["quadratic-programming", "convex-optimization", "portfolio-optimization", "socp", "constraints", "transaction-costs", "interior-point", "quantitative-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Quadratic programming is the practical engine that turns "I want the least-risky portfolio that obeys all my rules" into a single math problem a solver can answer exactly, in milliseconds, with a guarantee that the answer is the genuine best one.
>
> - A **quadratic program (QP)** minimizes a quadratic objective — portfolio variance, $\tfrac{1}{2}w^\top Q w + c^\top w$ — subject to *linear* constraints. That one shape covers almost every real portfolio problem.
> - Real desk rules become constraints: **long-only** ($w \ge 0$), **budget** ($\mathbf{1}^\top w = 1$), **position caps**, **sector limits**, **turnover penalties** for transaction costs, and a **tracking-error bound** that needs the next tier up, a second-order cone program (SOCP).
> - The whole field nests: **LP ⊂ QP ⊂ SOCP ⊂ SDP**. Because every one of these problems is *convex*, the solver is guaranteed to find the single **global** optimum — there are no traps to get stuck in.
> - The one number to remember: on a \$1,000,000 book, adding a transaction-cost term to the optimizer cut a rebalance from \$6,000 of trading cost down to \$1,200 — a real \$4,800 saved — while keeping almost all of the risk improvement.

Why can a computer find the single best portfolio out of an infinity of possibilities, in the blink of an eye, and *prove* that nothing better exists — yet the same kind of computer can spend a week on a chess position and still not be sure it played the best move? The answer is a property called *convexity*, and it is the quiet reason that portfolio optimization, which sounds like it should be impossibly hard, is one of the most reliably solved problems in all of finance. A modern solver will take a portfolio of ten thousand stocks, a tangle of position limits, sector budgets, and cost penalties, and hand you the provably optimal weights faster than you can refill your coffee.

This post is about the machinery that makes that possible: **quadratic programming** and its broader family, **convex optimization**. We will start from what an optimization problem even *is*, build the quadratic program piece by piece, and then spend most of our time on the part that actually matters in a trading job — encoding the real, messy constraints a desk lives by, and understanding what the solver does with them. Along the way we will solve four concrete portfolios by hand, watch a constraint "bite," save real dollars with a cost term, and deliberately break a problem so you learn to read the error message when a solver says "infeasible."

![Stack diagram showing LP nested inside QP nested inside SOCP nested inside SDP](/imgs/blogs/quadratic-convex-programming-math-for-quants-1.png)

## The convex hierarchy from LP to SDP

The figure above is the map of the whole territory, so let us read it once before we build anything. Each box sits *inside* the one above it, and that nesting is the single most useful fact in this entire subject. At the bottom is the **linear program (LP)** — the simplest, where everything in sight is a straight line. One step out is the **quadratic program (QP)**, which is our main character: it keeps the linear constraints but lets the *objective* curve, which is exactly what risk does. One step further out is the **second-order cone program (SOCP)**, which can handle a constraint that says "the length of this vector must stay under a limit" — that turns out to be exactly how you cap tracking error. The outermost box is the **semidefinite program (SDP)**, the most general and most expensive, which can optimize over whole matrices at once.

The reason the nesting matters so much is practical. Because an LP is a special case of a QP, any solver that can handle a QP can also handle an LP. Because a QP is a special case of an SOCP, a cone solver can handle your variance problem too. So when you reach for a more powerful tool, you never lose the ability to solve the simpler problems — you only pay a little more in computation. This is why a quant rarely needs to know more than "which is the smallest box my problem fits in," because the solver for that box will take care of everything inside it.

There is one more thread running through all four boxes, and it is the thread that makes them special among all possible optimization problems: every one of them is **convex**. We will define convexity carefully in a moment, but the headline is this — for a convex problem, the local best is always the global best. There is exactly one valley, and any sensible algorithm that rolls downhill lands in it. That is the difference between portfolio optimization and chess.

## Foundations: the building blocks

Before we can talk about quadratic programs we need to agree, from zero, on what an optimization problem is, what the pieces are called, and what makes one easy or hard. A practitioner can skim this; a beginner should read every line, because the rest of the post leans on these definitions.

### What an optimization problem is, in plain words

An **optimization problem** is a question of the form: *out of all the choices I am allowed to make, which one is best?* It has exactly three parts.

The first part is the **decision variables** — the things you get to choose. In portfolio optimization the decision variables are the *weights*, the fractions of your money you put into each asset. We bundle them into a single list (a *vector*) and call it $w$. If you hold three assets and you put 50% in the first, 30% in the second, and 20% in the third, then $w = [0.5,\ 0.3,\ 0.2]$.

The second part is the **objective function** — the single number you want to make as small (or as large) as possible. It is a recipe that takes your choices $w$ and spits out one number: the thing you care about. For a cautious investor that number is *risk*. For someone chasing return it might be *expected profit*. The objective is what "best" means, made precise.

The third part is the **constraints** — the rules your choice has to obey. You can't put more money into the market than you have. Maybe you're not allowed to short (bet against) a stock. Maybe compliance caps any single position at 40% of the book. Each rule narrows down which choices are even allowed. The set of all allowed choices is called the **feasible region** — the playing field — and the optimizer's job is to find the best spot *on* that field.

So every optimization problem reads: **minimize** [objective] **subject to** [constraints], where you choose the [variables]. That phrase — "minimize … subject to …" — is the grammar of the entire field. Memorize it and you can read any problem in this post.

> Optimization is just "what is the best choice, given the rules?" written carefully enough that a machine can answer it.

### What "linear" and "quadratic" mean here

A function is **linear** if its graph is a straight line (or, in many dimensions, a flat plane). The defining feature is that it has no curves and no products of variables with themselves. Your portfolio's *expected return* is linear in the weights: if asset A is expected to return 8% and asset B 4%, a portfolio with weights $w_A, w_B$ has expected return $0.08\,w_A + 0.04\,w_B$. Double a weight, double its contribution. No curvature anywhere.

A function is **quadratic** if it involves variables *squared* or *multiplied together* — terms like $w_A^2$ or $w_A w_B$. Its graph is a curve: a parabola in one dimension, a bowl in many. Portfolio *risk* is quadratic in the weights, and that is the whole reason we need quadratic programming rather than the simpler linear kind. We'll see exactly why risk curves in the next section.

A **constraint** is **linear** if the rule is a straight-line relationship: "the weights add up to 1" ($w_A + w_B + w_C = 1$) is linear; "no weight exceeds 0.4" ($w_A \le 0.4$) is linear. Almost every desk rule is naturally linear, which is a gift — it means the hard, curvy part (the risk) lives only in the objective, and the constraints stay flat and well-behaved.

### What convexity is, and why it is the whole game

A problem is **convex** when two things are true: the objective is a "bowl" (it curves upward, never downward), and the feasible region has no dents (if two points are allowed, every point on the straight line between them is allowed too). The picture to hold onto is a marble in a salad bowl. Wherever you release it, it rolls to the *same* bottom. There is one lowest point and no false bottoms, no little side-dimples where the marble could get stuck thinking it had found the answer.

That property is worth its weight in gold. For a convex problem, **any** local minimum is automatically the **global** minimum. An algorithm that just keeps stepping downhill cannot fail — there is nowhere to get trapped. This is why the solver can *prove* it found the best portfolio: convexity is a mathematical certificate that the bottom it reached is the only bottom there is. A non-convex problem (chess, scheduling, training a deep neural net) is a bumpy landscape with thousands of valleys, and you never quite know whether a deeper one is hiding over the next ridge.

Portfolio variance is convex because the covariance matrix that defines it is **positive semidefinite** — a technical way of saying "risk can never be negative, so the bowl always cups upward." We built that idea from the ground up in [the covariance matrix, from the ground up](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants); for today you only need the consequence: because risk is a genuine bowl and the constraints are flat-sided, the entire problem is convex, and convex problems get solved exactly.

#### Worked example: the smallest possible optimization

Let's make the grammar concrete with the tiniest problem imaginable. You have \$100 to split between two assets, a stock fund and a cash account. Cash has zero risk and zero return. The stock fund has some risk. Your *objective* is to minimize risk; your *constraint* is that the two weights add to 1 (you invest all \$100) and neither is negative.

With only "minimize risk" and no return target, the answer is trivial but instructive: put everything in cash. Weight vector $w = [0,\ 1]$, risk = 0, expected return = 0. The optimizer has done its job perfectly — it found the genuinely least-risky allowed portfolio. The lesson is that **an optimizer answers exactly the question you ask**. If the all-cash portfolio looks useless, it's because we forgot to tell the optimizer we also want return. Every interesting portfolio problem is a tug-of-war between an objective that wants to lower risk and a constraint (or a second term) that forces some return or some exposure into the mix. Hold that tension in mind; it is the engine of everything below.

## The anatomy of a quadratic program

Now we assemble the star of the show. A **quadratic program** is the optimization problem that minimizes a quadratic objective subject to linear constraints. In full dress it is written:

$$ \min_{w}\ \tfrac{1}{2} w^\top Q\, w + c^\top w \quad \text{subject to}\quad A w \le b,\ \ E w = f. $$

That looks like a wall of symbols, so let's translate every piece into English, because each one is a concrete object on a trading desk.

- $w$ is the **weight vector** — your decision, the fractions you'll hold in each asset.
- $Q$ is a **symmetric matrix** that defines the curvature of the objective. In portfolio optimization $Q$ *is* the covariance matrix $\Sigma$ (sometimes scaled by a risk-aversion number). The term $\tfrac{1}{2} w^\top Q w$ is therefore exactly portfolio variance — the risk. The $\tfrac{1}{2}$ is a cosmetic convention that makes the calculus cleaner; it changes nothing about which portfolio wins.
- $c$ is a **linear coefficient vector**. In a pure minimum-variance problem $c = 0$. When you want to *reward return*, you set $c = -\gamma\,\mu$, where $\mu$ is the vector of expected returns and $\gamma$ is how much you care about return versus risk. The minus sign is there because we're minimizing: subtracting return from the objective is the same as rewarding it.
- $A w \le b$ packs all the **inequality constraints** into one line: each row of $A$ and matching entry of $b$ is one rule like "weight on asset 3 is at most 0.4."
- $E w = f$ packs all the **equality constraints**: the classic one is the budget rule, $\mathbf{1}^\top w = 1$, written with $E = \mathbf{1}^\top$ (a row of ones) and $f = 1$.

The figure below shows the data flow: the objective and the constraint list both feed a solver, which checks that the problem is feasible and convex, and then returns the one weight vector that is globally best.

![Pipeline from risk objective and constraints into a QP solver out to optimal weights](/imgs/blogs/quadratic-convex-programming-math-for-quants-2.png)

The way this works in practice is that you, the quant, never touch the inside of the solver. You assemble four objects — $Q$, $c$, $A/b$, $E/f$ — hand them over, and receive $w$. The art of the job is almost entirely in *building those four objects correctly* so they encode what you actually mean. Get the matrices right and the solver is infallible; get them wrong and it will cheerfully solve the wrong problem to perfection.

The reason this particular shape — quadratic objective, linear constraints — is so dominant in finance is that it matches the structure of risk and rules exactly. Risk is quadratic (variance is a sum of squared and cross terms), and almost every real-world rule is linear (sums, caps, floors). Markowitz's mean-variance portfolio, which we develop as a pure optimization elsewhere in this series, is *literally* a quadratic program. So is risk-parity, so is most factor-based allocation, so is the index-tracking problem. Learn QP and you have learned the working form of modern portfolio construction.

#### Worked example: a three-asset long-only minimum-variance portfolio

Time to actually solve one. You manage a \$100,000 book and can hold three assets: a US equity fund (A), a bond fund (B), and a gold fund (C). Your goal is the **minimum-variance portfolio** — the mix that wobbles the least — under two rules: invest everything (budget) and no shorting (long-only, every weight $\ge 0$).

First we need the covariance matrix. Suppose the annual volatilities are 18% for equities, 6% for bonds, and 15% for gold, with these correlations: equities–bonds $-0.2$ (they hedge each other a bit), equities–gold $0.1$, bonds–gold $0.0$. Volatility squared gives variances; volatility-times-volatility-times-correlation gives covariances. The covariance matrix (in units of "fraction-squared") is approximately:

$$ \Sigma = \begin{bmatrix} 0.0324 & -0.00216 & 0.0027 \\ -0.00216 & 0.0036 & 0.0 \\ 0.0027 & 0.0 & 0.0225 \end{bmatrix}. $$

The objective is $\min_w \tfrac{1}{2} w^\top \Sigma w$ with $c = 0$, subject to $w_A + w_B + w_C = 1$ and $w \ge 0$. Solving this QP (we will see how the solver does it later; for now trust the result) gives roughly:

$$ w \approx [0.12,\ 0.79,\ 0.09]. $$

The portfolio piles into bonds, the calmest asset, and sprinkles a little into equities and gold for diversification — note it does *not* go 100% bonds, because the small negative equity–bond correlation means a dash of equities actually *lowers* total wobble. Plugging these weights back in, portfolio variance is $w^\top \Sigma w \approx 0.00306$, so volatility is $\sqrt{0.00306} \approx 5.5\%$ per year. On a \$100,000 book a one-standard-deviation year is therefore about a \$5,500 swing. Compare that to holding all equities, where a one-sigma year is \$18,000 — the optimizer cut the risk by two thirds, mostly by leaning on bonds and exploiting the hedge.

The intuition this teaches: the minimum-variance portfolio is *not* "hold the safest single asset" — it is "hold the combination whose pieces cancel each other out the most," and the solver finds that cancellation automatically from the covariance matrix.

## Encoding the rules a real desk lives by

The textbook problem above had two clean constraints. A real mandate has a dozen, and the entire skill of applied portfolio optimization is translating fuzzy human rules into the precise rows of $A$, $b$, $E$, and $f$ that the solver reads. The figure below catalogs the common ones and the math each becomes.

![Matrix table of portfolio constraint types mapped to their math form and kind](/imgs/blogs/quadratic-convex-programming-math-for-quants-3.png)

Let's walk the catalog, because each row is a rule you will genuinely have to encode.

### The budget constraint: spend exactly what you have

The most basic rule is that your weights add up to 1 — you invest the whole book, no more, no less. Written as math, $\mathbf{1}^\top w = 1$, where $\mathbf{1}$ is a column of ones. This is an *equality* constraint, so it lives in the $E w = f$ block with $E = [1, 1, \dots, 1]$ and $f = 1$. If you want to hold cash, you either add a cash "asset" with zero risk and return, or you relax the rule to $\mathbf{1}^\top w \le 1$ (invest *at most* the whole book), which turns it into an inequality.

### Long-only: no betting against anything

Many mandates — most mutual funds, most retirement accounts — forbid *short selling*, which is borrowing a stock to sell it now and buy it back cheaper later. Forbidding shorts means every weight must be non-negative: $w \ge 0$, read element by element. Each becomes one row of the inequality block, $-w_i \le 0$. This single rule has an outsized effect: without it, optimizers love to take enormous offsetting long and short positions that look great on paper and blow up in practice, so the long-only constraint is also a crude but effective form of risk control.

### Position caps: don't let one name dominate

A position cap says no single holding can exceed some fraction — say 40% — of the book. As math, $w_i \le 0.40$ for each asset $i$, one inequality row apiece. Caps exist because optimizers, left alone, will happily bet 95% on whichever asset has the best risk-adjusted numbers *in the historical data*, and historical numbers are noisy. The cap is humility encoded as a constraint: "I don't trust my estimates enough to bet the farm on one name." We will watch a cap bite in a worked example shortly.

### Sector and group limits: spread the bets

A sector limit caps the *combined* weight of a group of assets — "no more than 30% in technology." If assets 1, 4, and 7 are tech names, the rule is $w_1 + w_4 + w_7 \le 0.30$, which is again one linear inequality row (a row of $A$ with ones in the tech columns and zeros elsewhere, and $b = 0.30$). Group floors work the same way with the inequality flipped. This is how a portfolio manager expresses a view like "I want diversified sector exposure" in a form the machine understands.

### Turnover and transaction-cost penalties

This one is subtler and deserves its own section below, because it can be written either as a constraint ("don't trade more than 20% of the book") or as a *penalty added to the objective* ("each dollar of trading costs me money, so subtract it"). Both are how you stop the optimizer from churning your portfolio into oblivion. The penalty form involves an absolute value, which we'll see is still convex.

### The tracking-error bound: the constraint that breaks the QP mold

Every constraint so far has been linear — a flat wall. But one extremely common rule is *not*: keeping your **tracking error** (how far your returns drift from a benchmark like the S&P 500) under a cap. Tracking error is itself a square-root-of-a-quadratic — a *length* — so the constraint "tracking error $\le 3\%$" reads $\sqrt{(w - w_b)^\top \Sigma (w - w_b)} \le 0.03$, where $w_b$ is the benchmark's weights. That square root of a quadratic is exactly the shape of a **second-order cone**, and a constraint of this form is what lifts the problem from a QP up to an **SOCP**. We'll come back to it; for now just register that this single common rule is the reason quants need the next tier of the hierarchy.

#### Worked example: adding a 40% position cap that binds

Let's see a constraint actually change the answer. Suppose instead of pure minimum-variance, you're maximizing risk-adjusted return on three assets where asset A genuinely has the best numbers. Without any cap, the optimizer hands you a concentrated book:

$$ w_{\text{uncapped}} \approx [0.58,\ 0.27,\ 0.15]. $$

Asset A soaks up 58% of the \$100,000 book — \$58,000 in one name. Compliance says no single position may exceed 40%. We add the row $w_A \le 0.40$ to the inequality block and re-solve. The new answer:

$$ w_{\text{capped}} \approx [0.40,\ 0.38,\ 0.22]. $$

![Before-after columns contrasting an uncapped concentrated book with a 40 percent capped book](/imgs/blogs/quadratic-convex-programming-math-for-quants-4.png)

Look at what happened, shown in the before-after above. The cap on A *binds* — meaning the solver pushes A right up against its 40% ceiling and stops — and the \$18,000 it had to take out of A flows into B and C, the runners-up. The book is more diversified: \$40,000 in A, \$38,000 in B, \$22,000 in C. The objective is *slightly worse* than the uncapped version, because we forced the optimizer off its preferred point — that small loss is the literal price of the constraint, and it has a name we'll meet next (the shadow price). But the portfolio is far less exposed to a single name being wrong.

A constraint that the solver pushes right up against is called **binding** (or *active*); one that the optimal solution satisfies with room to spare is **slack**. The 40% cap on A binds; the (implicit) cap on B and C does not. The intuition: a binding constraint is the desk rule that is *actually* shaping your portfolio right now — the slack ones are just watching from the sidelines.

## Shadow prices: what a constraint actually costs you

When you add a constraint and the objective gets worse, exactly *how much* worse is a number worth knowing — it tells you the price of the rule. That number is the **shadow price** of the constraint, and it falls out of the optimization for free.

Here is the idea, stated simply. The 40% cap on asset A made our portfolio a little riskier (or a little less profitable) than it could have been. Suppose compliance offered to raise the cap to 41% — would that buy you much? The shadow price answers exactly that: it is the rate at which your objective improves per unit of loosening the constraint. If the shadow price of the 40% cap is, say, \$200 of risk-adjusted value per 1% of extra room, then bumping the cap to 41% is worth \$200 to you, and you now have a concrete number to take into the negotiation with compliance.

Mathematically, shadow prices are the **Lagrange multipliers** of the constraints — one number per constraint, produced automatically by the solver. A binding constraint has a non-zero shadow price (loosening it helps). A slack constraint has a shadow price of exactly zero (loosening it does nothing, because you weren't up against it anyway). That on-off relationship — *either* a constraint binds and has a price, *or* it's slack and has zero price, never both — is called **complementary slackness**, and it is one of the most useful diagnostic facts in all of optimization. The full machinery of multipliers and the conditions that characterize a constrained optimum is the Karush–Kuhn–Tucker (KKT) system, which is the natural companion to this post.

#### Worked example: reading a shadow price as a dollar figure

Return to the capped book. Say the solver reports that the 40% cap on asset A has a shadow price (Lagrange multiplier) of $\lambda = 0.018$ in the objective's natural units, and your objective is scaled so one unit equals \$1,000,000 of book value times annual risk-adjusted return. Then loosening the cap by one percentage point — from 40% to 41% — would improve the objective by about $\lambda \times 0.01 \times \$1{,}000{,}000 = 0.018 \times 0.01 \times \$1{,}000{,}000 = \$180$ per year.

That \$180 is small, which is itself the lesson: a tight-looking cap might be costing you almost nothing, so fighting compliance to relax it would be a poor use of political capital. Conversely, if the shadow price came back at \$18,000 per percentage point, you'd know the cap is genuinely strangling the strategy and the conversation is worth having. The intuition: shadow prices turn "this rule is annoying" into "this rule costs me exactly \$X," which is the only language a risk committee actually listens to.

## Transaction costs and the no-trade region

So far our portfolios have appeared from nowhere, fully formed. In reality you already *hold* a portfolio, and optimizing means *trading* from where you are to where you'd like to be — and trading costs money. Every share you buy or sell crosses a *bid-ask spread* (the gap between the price you can buy at and the price you can sell at), pays a commission, and pushes the price against you (*market impact*). A naive optimizer ignores all this: it computes the theoretically ideal weights and trades all the way there, no matter how small the improvement or how large the trade. That is a recipe for bleeding money on churn.

The fix is to put the cost *into the optimization*, so the solver only trades when the benefit beats the cost. There are two equivalent ways to do it.

### Turnover as a penalty in the objective

**Turnover** is the total amount you trade, measured as the sum of the absolute changes in each weight: $\sum_i |w_i - w_i^{\text{old}}|$, where $w^{\text{old}}$ is your current portfolio. If transaction costs are roughly proportional to how much you trade — a reasonable first model — then total cost is (cost per unit) $\times$ turnover. You fold that straight into the objective:

$$ \min_w\ \tfrac{1}{2} w^\top \Sigma w - \gamma\, \mu^\top w + \kappa \sum_i |w_i - w_i^{\text{old}}|. $$

The new third term, $\kappa \sum_i |w_i - w_i^{\text{old}}|$, charges the optimizer $\kappa$ dollars for every unit of trading. Now the solver faces a genuine trade-off: a trade only happens if the risk-and-return improvement it buys exceeds the cost it incurs. Crucially, the absolute value $|x|$ is a **convex** function — a "V" shape, which cups upward — so the problem stays convex and still solves exactly. (To feed it to a standard QP solver you split each $|w_i - w_i^{\text{old}}|$ into a buy part and a sell part with a small linear trick, but conceptually it's just "charge for distance moved.")

### The no-trade region this creates

Here is the beautiful consequence. Because there's now a fixed-rate cost on *any* trade, tiny improvements aren't worth acting on. Around your current portfolio there forms a **no-trade region** — a band where the optimal move is *do nothing*, because the gain from rebalancing wouldn't cover the cost. Only when your portfolio drifts far enough outside the band does the solver decide a trade pays. This is not a hack; it falls directly out of the convex math, and it is exactly how real desks avoid the death-by-a-thousand-trades that kills a strategy's net returns. The width of the no-trade region grows with the cost rate $\kappa$: the more expensive trading is, the more drift you tolerate before acting.

This is the same economic logic that governs execution algorithms, which decide not just *whether* to trade but *how* to slice a large order across time to minimize impact — we go deep on that in [execution algorithms: VWAP, TWAP, POV](/blog/trading/quantitative-finance/execution-algorithms-vwap-twap-pov-quant-research). Portfolio optimization decides the *target*; execution decides the *path*; both are dominated by the same enemy, cost.

#### Worked example: rebalancing a \$1,000,000 book with and without a cost term

Let's put real dollars on it. You run a \$1,000,000 book. Over the quarter your holdings have drifted, and a fresh minimum-variance solve says the *ideal* portfolio is meaningfully different from what you hold. You run the optimizer two ways.

**Without a cost term**, the optimizer chases the ideal weights exactly. The total turnover comes out to 60% — meaning \$600,000 of buying and selling to move \$1,000,000 of book onto the ideal point. At a realistic all-in cost of 1% of the traded value (spread plus impact plus commission), that's:

$$ \text{cost}_{\text{no term}} = 0.60 \times \$1{,}000{,}000 \times 0.01 = \$6{,}000. $$

And what did all that trading buy? Suppose the risk improvement is worth about \$1,500 over the quarter. You spent \$6,000 to gain \$1,500. You *lost* \$4,500 net by rebalancing. The optimizer did exactly what you asked — it just asked the wrong question.

**With a turnover penalty** set to the true cost rate ($\kappa = 0.01$), the solver now refuses trades that don't pay for themselves. It trims the rebalance to only the moves that matter — turnover drops to 12%, or \$120,000 traded:

$$ \text{cost}_{\text{with term}} = 0.12 \times \$1{,}000{,}000 \times 0.01 = \$1{,}200. $$

![Before-after columns contrasting an uncosted rebalance with a turnover-penalized rebalance](/imgs/blogs/quadratic-convex-programming-math-for-quants-6.png)

The before-after above tells the story. By moving only partway toward the ideal — landing on the *edge* of the no-trade region rather than the exact center — the cost-aware solve keeps almost all of the risk improvement (say \$1,300 of the original \$1,500) while spending \$1,200 instead of \$6,000. You saved \$4,800 of trading cost and gave up only \$200 of benefit. The intuition: the ideal portfolio on paper is rarely worth the cost of reaching it exactly, and a cost term in the objective is how you let the math find the point where chasing perfection stops paying.

## Inside the solver: interior-point versus active-set

You don't have to know how the solver works to use it, any more than you need to understand an internal combustion engine to drive. But knowing the two main families helps you reason about speed, about why some problems are slow, and about what "the solver failed" might mean. There are two dominant algorithms for QPs and convex problems, and they reach the same answer by different routes.

![Stack of interior-point solver steps from inside the region to the optimal weights](/imgs/blogs/quadratic-convex-programming-math-for-quants-7.png)

### Interior-point methods: walk through the middle

The **interior-point method**, sketched in the stack above, starts at a point *strictly inside* the feasible region — comfortably away from every wall — and walks toward the optimum from within. To keep itself from crashing into a constraint wall too early, it adds a *barrier function* to the objective that shoots up to infinity as you approach any boundary, like a force field around the edges of the playing field. Then it takes a **Newton step** (a smart, curvature-aware downhill jump — the same Newton's method we develop in [matrix calculus for optimization](/blog/trading/math-for-quants/matrix-calculus-optimization-math-for-quants)) toward the barriered minimum, gradually *weakens* the barrier, and repeats. As the barrier fades, the path it traces — the "central path" — homes in on the true optimum, which usually sits right on a boundary. Interior-point methods are the workhorse for large problems: they reach high accuracy in a small, *predictable* number of iterations (often 10–50) almost regardless of problem size, which is why they dominate when you have thousands of assets.

### Active-set methods: walk along the walls

The **active-set method** takes the opposite tack. It guesses which constraints are *binding* (the "active set"), solves the much simpler equality-constrained problem that results, checks whether the guess was right, and if not, swaps one constraint in or out and tries again. It effectively walks *along the edges* of the feasible region, vertex to vertex, the way the famous simplex method does for linear programs. Active-set methods are superb for small-to-medium problems and especially when you're re-solving a slightly changed problem repeatedly (a *warm start*), because the active set rarely changes much between solves — exactly the situation in live rebalancing.

### Why convexity is what makes either one trustworthy

The deep reason both methods *work* — the reason either one can declare victory and be believed — is convexity. In a convex problem, the moment an algorithm finds a point where it can't improve in any feasible direction, it has found the global optimum, full stop, no caveats. There's no "but maybe a better point exists elsewhere," because convexity forbids elsewhere. Run the same convex problem through an interior-point solver and an active-set solver and they land on the *same* weights, because there is only one place to land. This is the property that lets a risk committee sign off on an automated optimizer: it isn't trusting the algorithm's cleverness, it's trusting a theorem.

#### Worked example: counting the cost of a solve

Concreteness helps. Say you run a 500-asset long-only optimizer with a budget constraint, position caps, and ten sector limits. An interior-point solver converges in about 25 iterations, each iteration dominated by solving a linear system whose cost scales like the number of assets cubed for a dense problem — but real covariance matrices are structured, so a good solver exploits that and the whole solve finishes in well under a second on a laptop. Suppose your data vendor charges \$0.001 of compute per solve and you re-optimize every minute across a trading day; that's roughly 390 solves, or about \$0.39 a day — trivial. Now imagine the problem were *non-convex* (it isn't, but suppose): you'd need a global solver that might run for *hours* and still not prove optimality, turning a \$0.39 task into an intractable one. The intuition: convexity isn't just a theoretical nicety — it's the difference between a sub-second, sub-penny solve and a problem you can't reliably solve at all.

## The tracking-error bound and the jump to SOCP

We flagged earlier that one common rule breaks the QP mold. Let's give it the full treatment, because it's the cleanest example of *why* the hierarchy exists and *when* you climb it.

![Tree of convex problem classes branching into LP, QP, and SOCP](/imgs/blogs/quadratic-convex-programming-math-for-quants-5.png)

The tree above shows where each problem class sits. **Tracking error** is the volatility of the difference between your portfolio's returns and a benchmark's. If your weights are $w$ and the benchmark's are $w_b$, the *active weights* are $w - w_b$, and tracking error is

$$ \text{TE}(w) = \sqrt{(w - w_b)^\top \Sigma (w - w_b)}. $$

The square root of a quadratic form is a **norm** — a length. A constraint that says "this length is at most 3%," $\text{TE}(w) \le 0.03$, is geometrically the statement that the active-weight vector must stay inside a ball (an ellipsoid, really). That kind of constraint is called a **second-order cone constraint**, and a problem with a quadratic objective and one or more cone constraints is a **second-order cone program (SOCP)** — one rung up the ladder from a QP.

Why not just *square* both sides to get rid of the root and stay in QP-land? Because squaring a constraint can change its geometry in ways that break convexity in general, and because cone solvers handle the un-squared form natively and robustly. The honest answer is: you reach for an SOCP solver, which (per the nesting) also handles every QP and LP you might throw at it, and you stop worrying about which rung you're on. Modern convex modeling tools (CVXPY, MOSEK, and friends) let you write $\text{TE}(w) \le 0.03$ literally and dispatch the right solver automatically.

The practical payoff is large. Index-enhancement strategies — "beat the S&P 500 by a little, but never drift more than 3% away from it" — are *defined* by a tracking-error cone constraint. Risk-budgeting and minimum-tracking-error portfolios live here too. The moment your mandate mentions a benchmark and a leash, you've left pure QP and entered the cone.

#### Worked example: a tracking-error leash in dollars

You run a \$1,000,000 enhanced-index book against the S&P 500 with a 3% annual tracking-error limit. Three percent of \$1,000,000 is a \$30,000 standard-deviation *active* swing per year — that is, in a typical year your portfolio's return will land within about \$30,000 of the index's return, in either direction. You want to tilt toward a few names your alpha model likes (the kind of signal we build in [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research)), but the cone constraint caps how far you can stray.

Suppose your unconstrained tilt would produce a tracking error of 5% — a \$50,000 active swing — which compliance forbids. The SOCP solver shrinks your tilts proportionally until tracking error hits exactly 3%, the binding cone constraint, giving up some expected alpha to fit the leash. If the full 5% tilt was expected to add \$25,000 of alpha, the leashed 3% version might keep about \$15,000 of it — you sacrificed \$10,000 of expected edge to honor a \$30,000 risk leash. The intuition: a tracking-error bound is a cone, not a flat wall, and respecting it is precisely the job a QP cannot do but an SOCP can.

## When a problem has no answer: infeasibility

Everything so far assumed the solver *can* find a feasible portfolio. Sometimes it can't, and the solver returns not weights but a verdict: **infeasible**. This is not a bug; it's the solver telling you that the constraints you wrote *contradict each other* — there is no portfolio in the universe that satisfies all of them at once, so the feasible region is empty. Learning to read and diagnose infeasibility is a genuine on-the-job skill, because it usually means a rule was specified wrong, not that the market is broken.

### The two ways a problem dies

A convex problem can fail in two distinct ways, and they mean opposite things.

**Infeasibility** means the feasible region is empty — no allowed point exists. Example: you require long-only ($w \ge 0$) *and* a budget ($\mathbf{1}^\top w = 1$) *and* every single position capped at 20% ($w_i \le 0.20$) — but you only have *four* assets. Four weights, each at most 0.20, can sum to at most 0.80, which can never equal 1. The constraints contradict. The solver reports infeasible.

**Unboundedness** is the opposite failure: the objective can be improved forever without limit, usually because you forgot a constraint. Example: maximize return with no budget and no position limits — the optimizer happily puts infinite money into the best asset, and "best" never tops out. The fix is always a missing constraint (a budget, a cap, a no-short rule) that fences in the playing field. Unboundedness is rarer in practice precisely because the budget constraint usually fences things in, but it bites beginners who forget it.

### How to diagnose an infeasible problem

When a solver says infeasible, the move is *not* to panic or to assume the solver is broken — convex solvers are extremely reliable and will tell you the truth. The move is to find the contradicting constraints. Three techniques, in order of how often they crack the case:

1. **Count the arithmetic.** Most infeasibilities are dumb in hindsight: caps that can't sum to the budget, a sector floor plus other floors that exceed 100%, a long-only book asked to hit a return higher than its best single asset. Add up the bounds by hand and you usually spot the clash in a minute.
2. **Relax constraints one at a time.** Temporarily drop each constraint and re-solve. The moment the problem becomes feasible, the constraint you just removed was part of the conflict. This is the optimization equivalent of pulling fuses to find the short.
3. **Ask the solver for an irreducible infeasible set.** Good solvers can return a minimal subset of constraints that is *itself* infeasible — the smallest group that contradicts. That points you straight at the offending rules without trial and error.

The deeper lesson is that infeasibility is *information*. A well-built optimization tool doesn't just crash; it surfaces *which* rules collided, so a quant can go back to the portfolio manager and say "your 20% cap and your full-investment rule can't both hold with only four names — which do you want to relax?"

#### Worked example: diagnosing an infeasible \$500,000 mandate

You're handed a \$500,000 mandate with these rules: long-only, fully invested, no single position over 15%, *and* at least 50% in a "defensive" sleeve of two specific bond funds. You set it up, hit solve, and the solver returns **infeasible**. Let's find the clash.

Count the arithmetic. The defensive sleeve must hold at least 50% — that's \$250,000 — split across exactly two funds. But each fund is capped at 15% (\$75,000). Two funds at 15% each sum to at most 30% (\$150,000). You're requiring \$250,000 in a sleeve that, under the cap, can hold at most \$150,000. The 50% floor and the 15% cap on those two names *directly contradict*. The fix is a conversation, not a code change: either widen the defensive sleeve to more funds (so the 15% caps can sum past 50%), raise the cap, or lower the floor. The intuition: an "infeasible" verdict is rarely the solver's fault — it's the solver faithfully reporting that two of your rules can't both be true, and the diagnosis is almost always grade-school arithmetic on the bounds.

## Putting it together: a constraint-by-constraint build

Let's consolidate by building a realistic optimizer in words, one constraint at a time, so you can see how the four objects ($Q$, $c$, $A/b$, $E/f$) accumulate. This is the mental checklist you'd actually run through on a desk.

Start with the **objective**. You want low risk and some return, so $\tfrac{1}{2} w^\top \Sigma w - \gamma \mu^\top w$: that's $Q = \Sigma$ and $c = -\gamma\mu$. The risk-aversion knob $\gamma$ slides you along the trade-off between safe and aggressive.

Add the **budget**: $\mathbf{1}^\top w = 1$, one equality row. Add **long-only**: $w \ge 0$, one inequality row per asset. Add **position caps**: $w_i \le 0.40$, another row per asset. Add a **sector limit**: $w_{\text{tech}} \le 0.30$, one row summing the tech columns. Add a **turnover penalty**: append $\kappa \sum_i |w_i - w_i^{\text{old}}|$ to the objective (split into buy/sell halves to keep it a clean QP). If the mandate has a benchmark leash, add the **tracking-error cone** $\text{TE}(w) \le 0.03$ — and now you've climbed from QP to SOCP, so you point the problem at a cone solver.

Hand the assembled problem to the solver. It checks feasibility, confirms convexity, and returns the global-optimal $w$ plus a shadow price for every constraint. You read the shadow prices to see which rules are actually biting and what they cost. If it returns "infeasible," you count the arithmetic to find the clash. That entire loop — assemble, solve, read multipliers, diagnose — is the day-to-day craft of applied portfolio optimization.

> The hard part of portfolio optimization isn't the optimizer. It's writing down what you actually mean, in constraints, without contradicting yourself.

#### Worked example: the risk-aversion knob in dollars

One more number to make $\gamma$ concrete. Take a two-asset book, \$1,000,000, choosing between equities (12% expected return, 18% vol) and bonds (4% expected return, 6% vol), lowly correlated. At a high risk-aversion $\gamma$ (you hate risk), the optimizer might land near 20% equities / 80% bonds: expected return $0.20 \times 12\% + 0.80 \times 4\% = 5.6\%$, or \$56,000 a year, with modest volatility. Crank $\gamma$ down (you tolerate risk), and the optimizer slides to 70% equities / 30% bonds: expected return $0.70 \times 12\% + 0.30 \times 4\% = 9.6\%$, or \$96,000 a year, but the one-sigma swing roughly doubles. The same QP, the same constraints, just a different $\gamma$, traces out the whole **efficient frontier** — the menu of best-possible portfolios at every risk level. The intuition: $\gamma$ is the single dial that turns "I want the best portfolio" into "I want the best portfolio *for my appetite*," and the QP solves it for any setting in milliseconds.

## Common misconceptions

**"The optimizer finds the best portfolio, full stop."** It finds the portfolio that's best *according to the objective and constraints you gave it, using the estimates you fed it.* If your expected returns are noisy (and they always are), the optimizer will confidently over-concentrate in whatever asset got lucky in your sample — a problem called *error maximization*. The optimizer is a flawless answer to a question that may be poorly posed. Garbage in, confidently optimal garbage out. This is precisely why regularization and robust methods exist, and why position caps are good practice even when they cost a little.

**"Adding constraints can only hurt my returns."** Every constraint does mathematically *narrow* the feasible region, so the in-sample objective can only get worse or stay the same — that's true. But out-of-sample, constraints like caps and long-only routinely *improve* real performance, because they stop the optimizer from making huge bets on noisy estimates. A constraint is a way of injecting humility the data doesn't have. The in-sample "cost" of a sensible constraint is often an out-of-sample *gift*.

**"Convex means the problem is easy/small."** Convexity is about the *shape* of the problem (one bowl, no dents), not its *size*. A convex problem with a million variables is still convex and still solvable to global optimality; a non-convex problem with three variables can be genuinely hard. The dividing line in optimization isn't big-versus-small, it's convex-versus-not. That's the single most important classification to make about any problem you face.

**"Quadratic programming is just for portfolios."** The QP shape — quadratic objective, linear constraints — shows up everywhere: support vector machines in machine learning, model-predictive control in robotics and process engineering, least-squares fitting with bounds, and optimal-execution scheduling. Learning QP for finance hands you a tool that's useful far beyond finance. The covariance matrix is just one of many things that happen to be a nice quadratic.

**"If the solver says infeasible, something is broken."** Almost never. Convex solvers are among the most reliable software in finance, and "infeasible" is a *correct and useful answer*: it means your constraints contradict each other. The bug is in your constraint specification, not the solver. The right response is to diagnose the clash (count the arithmetic, relax constraints one at a time), not to swap solvers or assume a numerical glitch.

**"More risk aversion always means less risk."** Turning up $\gamma$ shifts the optimizer toward lower-variance portfolios, yes — but only within the feasible region and only as measured by the model's covariance matrix. If your covariance estimate is wrong (it underweights a tail risk, say), a "low-risk" optimized portfolio can still be a sitting duck for the risk the model didn't see. The optimizer minimizes the risk you *told* it about, not the risk that's actually out there.

## How it shows up in real markets

### 1. The everyday rebalance on a long-only fund

The most common use of QP in the world is utterly mundane: a long-only equity fund re-optimizing its holdings, daily or weekly, to stay near its target risk profile while honoring its prospectus rules. The objective is minimum tracking error or a risk-return blend; the constraints are budget, long-only, per-name caps (often 5%, the "diversified fund" regulatory line in many jurisdictions), and sector bands. A turnover penalty keeps trading costs in check. This single template — a QP with maybe a thousand rows — runs at thousands of asset managers every single trading day. It's invisible precisely because it works: the math is settled, the solver is fast, and the answer is provably optimal given the inputs.

### 2. Risk parity and the 2018 "volmageddon" stress

Risk-parity funds size each asset so that every asset contributes *equal* risk to the portfolio — a constrained optimization problem (often solved as a sequence of convex sub-problems) rather than a single textbook QP, but built from the same covariance-matrix machinery and the same convex logic. These strategies ballooned to hundreds of billions of dollars in the 2010s. In early February 2018, a sudden spike in equity volatility (the VIX more than doubling in days) forced volatility-targeting and risk-parity strategies to de-risk roughly in unison, because their optimizers all read the same rising covariance estimates and all cut exposure. The mechanism from this post — an optimizer responding to its risk inputs — operated correctly at each fund, but the *collective* response amplified the move. The lesson is sobering: a convex optimizer is individually rational and globally optimal for its owner, yet a crowd of them keyed to the same inputs can move markets in concert.

### 3. Index-enhancement and the tracking-error cone

When a pension fund wants the S&P 500's return "plus a little," it hires a manager to run an enhanced-index mandate: beat the benchmark modestly while keeping tracking error under a hard cap, often 1–3% annually. This is the SOCP-with-a-cone-constraint problem made real, run on hundreds of billions of institutional dollars. The cone constraint is the *defining* feature of the mandate — it's what separates "enhanced indexing" from "active management." When a manager blows through their tracking-error limit (as several "low-vol" and "smart-beta" products effectively did in the 2020 COVID crash, when correlations spiked and historical covariance matrices badly understated active risk), it's a direct, observable failure of the cone constraint to bind on *forward* risk because it was calibrated on *stale* covariance. The math was right; the inputs were a quarter out of date.

### 4. Transaction costs and the high-turnover trap

A famous and recurring failure mode in systematic strategies is the backtest that looks spectacular gross of costs and dies net of them. A signal might be genuinely predictive but decay in hours, so capturing it requires constant trading. Without a turnover penalty in the optimizer, the "optimal" portfolio churns 200%+ per month, and at even 10 basis points a side, costs swamp the alpha. Desks that survive build the cost term in from day one — exactly the \$6,000-vs-\$1,200 trade-off from our worked example, scaled to billions. The 2007 "quant quake," when many statistical-arbitrage funds suffered simultaneous losses as they tried to deleverage crowded positions at once, was in part a transaction-cost-and-impact story: the optimizers wanted to trade out, but everyone trading the same way at the same time made the *cost* of trading explode, far beyond what any static cost term anticipated.

### 5. Bank capital and regulatory optimization

Banks run enormous convex optimizations to allocate capital across business lines under regulatory constraints (Basel capital ratios, liquidity-coverage rules, leverage limits). Each rule is a constraint; the objective is return on regulatory capital. These are often LPs or QPs with thousands of constraints, and the *shadow prices* are the prize: the multiplier on the capital constraint tells the bank exactly how much an extra dollar of regulatory capital is worth, which drives real decisions about which businesses to grow and which to shrink. When regulators tighten a rule, the bank's optimizer instantly reprices every business through the changed shadow prices — convex optimization as a live management dashboard.

### 6. The Black–Litterman tilt as a constrained QP

A persistent practical problem is that naive mean-variance optimizers produce wild, unstable weights because expected-return estimates are so noisy. The Black–Litterman model, developed at Goldman Sachs, blends market-implied returns with the manager's views to produce *stable* expected returns, which then feed a standard constrained QP. The optimization machinery is exactly what we've built here; the innovation is feeding it sane inputs. It's the industry's standing acknowledgment that the optimizer is only as good as its estimates, and that the cure for "error maximization" is partly better inputs (Black–Litterman, shrinkage) and partly better constraints (caps, turnover penalties) — two halves of the same robustness story.

## When this matters to you

If you're learning quant finance, quadratic and convex programming is one of the highest-leverage topics you can master, because it's where the abstract math of the earlier posts — covariance matrices, gradients, Lagrange multipliers — finally turns into *a portfolio you can actually hold*. The mean-variance theory tells you the shape of the answer; convex programming is how you get the answer with the real-world rules attached. Almost every quant role, from portfolio construction to risk to execution, touches an optimizer, and understanding what it does (and what it can't do) separates someone who *runs* an optimizer from someone who *trusts it blindly*.

If you're an investor rather than a quant, the practical takeaway is humility about optimized portfolios. An "optimal" portfolio is optimal only for the estimates and constraints behind it — change the covariance estimate or the cost assumption and the "best" portfolio moves. The healthiest stance is to treat optimization as a disciplined way to *encode your rules and trade-offs*, not as an oracle that divines the future. The constraints — caps, diversification, cost-awareness — often do more for your real-world results than the objective does, because they protect you from your own noisy estimates. *(This is educational, not individualized financial advice.)*

A few honest cautions to carry forward. The optimizer's confidence is not your confidence: it will report a single best portfolio to many decimal places, but that precision is about the *math*, not the *world*. The biggest dollar risks in optimized portfolios come from inputs you got wrong (a covariance matrix that missed a tail correlation) and rules you forgot (a missing turnover penalty), not from the solver. And every constraint that protects you also costs you a little expected return — the shadow price is the exact dollar figure of that trade-off, so read it before you fight a rule.

**Further reading and next steps.** To see where the *objective* of these problems comes from, work through the mean-variance theory and the efficient frontier in this series, which derives the risk-return trade-off the QP optimizes. For the constrained-optimization theory underneath the shadow prices — the Lagrange multipliers and KKT conditions — read the companion post on the Lagrangian. For the calculus that powers the solver's Newton steps, see [matrix calculus for optimization](/blog/trading/math-for-quants/matrix-calculus-optimization-math-for-quants). For the covariance matrix $Q$ that defines the bowl, see [the covariance matrix from the ground up](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants). And for the two problems that bracket portfolio construction in the trading workflow — where the return estimates come from and how the trades actually get executed — see [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) and [execution algorithms: VWAP, TWAP, POV](/blog/trading/quantitative-finance/execution-algorithms-vwap-twap-pov-quant-research). On the software side, the open-source modeling layer CVXPY (with solvers like OSQP for QPs and ECOS/MOSEK for cones) lets you write every problem in this post in a few lines and dispatch the right solver automatically — the fastest way to turn this understanding into working code.
