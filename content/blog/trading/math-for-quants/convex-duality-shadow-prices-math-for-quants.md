---
title: "Convex duality and shadow prices: what a constraint actually costs"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "Every optimiser a desk runs is a constrained problem, and the constraints are where the argument happens. The dual solution prices each one, turning 'risk wants this tighter, PMs want it looser' into a number in dollars per year."
tags: ["convex-duality", "shadow-price", "lagrange-multiplier", "kkt-conditions", "slater-condition", "complementary-slackness", "portfolio-constraints", "risk-limits", "mean-variance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 20
---

> [!important]
> **TL;DR:** The dual of a constrained optimisation attaches a price to every rule the desk runs under, in the objective's own units, which converts a governance argument into arithmetic.
>
> - A **shadow price** is the multiplier on a constraint. Its units are objective-per-constraint-unit, and that division is the entire intuition.
> - **Weak duality** is free: any price you name certifies a bound. **Strong duality**, the thing that makes the price meaningful, needs convexity plus a constraint qualification, and **Slater's condition** is the one to check.
> - **Complementary slackness** makes a solver's output readable at a glance: a slack constraint prices at exactly zero, and a positive price means the constraint is binding.
> - The number to remember: on the \$500m book below, the gross exposure limit prices at **\$27,727 per year per extra point of gross**, but relaxing that limit all the way is worth **\$294,643**, not the \$589,285 the price implies. The shadow price is a tangent line, and for a mean-variance objective it overstates the full relaxation by exactly a factor of two.

## The argument your optimiser can settle

Somewhere this week a portfolio manager told a risk officer that the gross exposure cap is strangling the strategy, and the risk officer said the cap is there for a reason. Both of them are right, and neither has said anything checkable. The argument has no unit.

It does have a unit, and the optimiser already computed it. Every constrained optimisation carries a shadow companion, the **dual problem**, whose variables are prices, one per constraint. Solve it and each rule on the mandate arrives with a tag: this many dollars of objective per year, per unit of limit. The gross cap costs \$27,727 a year per point. The long-only rule on the rates sleeve costs \$132,366 a year per point. The sector cap costs nothing at all, because nobody is pressed against it. Now the argument is about numbers.

This post builds that machinery from zero and then spends most of its time on the part people get wrong: a shadow price is a **local derivative**, it decays, and there are mandates where no finite price exists at all. The mechanics of setting up a Lagrangian and solving the KKT system are covered in [the Lagrangian and the KKT conditions](/blog/trading/math-for-quants/lagrangian-kkt-conditions-math-for-quants), and the portfolio problem itself in [mean-variance optimization and the efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants); this one is about reading the prices those methods hand back. Every dollar figure below is illustrative arithmetic on assumed inputs, computed exactly and cross-checked against a numerical solver.

## Foundations: the primal, the Lagrangian, and the dual

### The problem on the desk

A \$500m multi-strategy book runs three sleeves. Equity is expected to earn 8% excess return a year at 20% volatility, credit 5% at 12%, rates 3% at 9%. Correlations are 0.30 between equity and credit, 0.20 between equity and rates, and 0.30 between credit and rates. Write ${w}$ for the vector of sleeve weights as a fraction of capital, ${\mu}$ for expected excess returns and ${\Sigma}$ for the covariance matrix.

The desk maximises **certainty-equivalent excess return**, expected return minus a penalty for variance, with risk aversion ${\gamma = 4}$:

$$\max_{w} \; \mu^\top w - \frac{\gamma}{2} w^\top \Sigma w \qquad \text{subject to} \qquad \mathbf{1}^\top w \le L.$$

The single constraint is a **gross exposure limit**: total sleeve notional cannot exceed ${L}$ times capital. Risk has set ${L = 1.25}$, so \$625m of notional on \$500m of equity. This is the **primal** problem. It searches over portfolios.

### The Lagrangian: buy your way out of the fence

The trick that unlocks everything is to stop treating the limit as a wall and start treating it as a toll. Pick a price ${\lambda \ge 0}$ per unit of gross exposure, then let the portfolio go anywhere it likes as long as it pays:

$$\mathcal{L}(w, \lambda) = \mu^\top w - \frac{\gamma}{2} w^\top \Sigma w - \lambda\left(\mathbf{1}^\top w - L\right).$$

That is the **Lagrangian**. For any feasible ${w}$ the bracket is negative or zero, so ${\mathcal{L}(w,\lambda) \ge \mu^\top w - \tfrac{\gamma}{2} w^\top \Sigma w}$: charging a non-negative toll can only flatter a portfolio that was obeying the rule anyway.

### The dual function: name a price, get a bound

Now fix the price and let the portfolio run free. Define

$$g(\lambda) = \sup_{w} \; \mathcal{L}(w, \lambda).$$

This is the **dual function**, and it has one property that does all the work. Take the optimal feasible portfolio ${w^\star}$. Because it is feasible, ${\mathcal{L}(w^\star,\lambda) \ge}$ its objective value; because the supremum is over all ${w}$, ${g(\lambda) \ge \mathcal{L}(w^\star,\lambda)}$. Chain them and ${g(\lambda) \ge p^\star}$ for **every** ${\lambda \ge 0}$. That is **weak duality**, and it costs nothing: any price you invent hands you a provable ceiling on how good the book could possibly be.

For this problem the supremum is a plain unconstrained quadratic, so it solves in closed form. With the three standard mean-variance scalars ${A = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}}$, ${B = \mathbf{1}^\top \Sigma^{-1} \mu}$ and ${C = \mu^\top \Sigma^{-1} \mu}$:

$$g(\lambda) = \frac{C - 2\lambda B + \lambda^2 A}{2\gamma} + \lambda L.$$

The **dual problem** is to push that ceiling down as far as it will go: minimise ${g(\lambda)}$ over ${\lambda \ge 0}$. Differentiating and setting to zero gives the optimal price directly,

$$\lambda^\star = \frac{B - \gamma L}{A}.$$

![Two ladders of numbers converging: feasible portfolios rising from below and dual prices falling from above, meeting at the optimum](/imgs/blogs/convex-duality-shadow-prices-math-for-quants-1.webp)

That figure is the whole idea in one image. On the left, every portfolio you can actually hold is a floor under the answer. On the right, every price you can name is a ceiling over it. The primal pushes up, the dual pushes down, and when the gap closes to zero you have not just found the optimum, you have **proved** it.

#### Worked example 1: pricing the gross exposure limit on a \$500m book

For this covariance matrix, ${A = 153.3035}$, ${B = 5.85013}$ and ${C = 0.290904}$. With ${\gamma = 4}$ and ${L = 1.25}$:

$$\lambda^\star = \frac{5.85013 - 4 \times 1.25}{153.3035} = \frac{0.85013}{153.3035} = 0.0055454.$$

The optimal portfolio is ${w^\star = \tfrac{1}{\gamma}\Sigma^{-1}(\mu - \lambda^\star \mathbf{1})}$, which comes out at weights 0.336964, 0.513376 and 0.399660. On \$500m that is **\$168.5m equity, \$256.7m credit, \$199.8m rates**, \$625.0m of gross notional against the 125% cap, running at 12.01% volatility. Expected excess return is \$32.31m a year, the variance penalty costs \$14.42m, and the certainty equivalent is **\$17,886,866 a year**.

Now read the price. One point of gross exposure is 1% of \$500m, so \$5m of notional. The shadow price says that point is worth

$$0.0055454 \times \$5{,}000{,}000 = \$27{,}727 \text{ per year}.$$

The dual side checks it. At ${\lambda = 0}$ the certificate is \$18,181,509 a year; at ${\lambda = 0.0020}$ it tightens to \$18,007,303; at ${\lambda = 0.0040}$ to \$17,909,749, only \$22,883 above the truth. At ${\lambda^\star = 0.0055454}$ the ceiling lands exactly on \$17,886,866 and the gap is zero. The best portfolio a naive guess produces, equal weights at 125% gross, earns \$17,607,639, so before solving anything you could already bracket the answer to \$302,110 a year.

**Verification.** The analytic KKT solution above was checked against two independent numerical solvers on the same problem, SLSQP and a trust-region interior-point method. The weights agree to within ${5 \times 10^{-8}}$, and the trust-region solver's own reported Lagrange multiplier comes back as 0.0055454, matching the closed form to seven digits. The intuition: the price is not an artefact of one derivation, because two routes that share no algebra arrive at the same number.

## The units are the intuition

Ask what ${\lambda}$ measures and the rest follows. The objective is measured in dollars of certainty-equivalent return per year. The constraint is measured in points of gross exposure. The multiplier sits between them, so its units are **dollars per year, per point of gross**. Nothing else it could be.

![Four stacked rows deriving the shadow price unit by dividing the objective unit by the constraint unit](/imgs/blogs/convex-duality-shadow-prices-math-for-quants-2.webp)

Every reading of a shadow price is that division applied somewhere. If the objective is annual profit and the constraint is capital, the multiplier is a return on the marginal dollar. If the objective is tracking error and the constraint is a sector band, it is basis points of tracking error per point of band. If the objective is Sharpe and the constraint is turnover, it is Sharpe per unit of turnover. Get the division right and the number interprets itself, which is why the first thing to say about any reported multiplier is its unit, not its size.

Formally the statement is the **sensitivity theorem**: if ${p^\star(u)}$ is the best achievable objective when the limit is moved to ${L + u}$, then ${\lambda^\star = \left. \mathrm{d}p^\star / \mathrm{d}u \right|_{u=0}}$. Hold on to the word *derivative*. It comes back.

## Weak duality is free; strong duality is an assumption

Weak duality, ${g(\lambda) \ge p^\star}$, needed nothing. It holds for non-convex problems, integer problems, problems nobody can solve. **Strong duality**, the statement that the best ceiling equals the truth, is a real assumption and it can fail.

The standard sufficient condition for a convex problem is **Slater's condition**: there exists a point that satisfies the equality constraints and satisfies every inequality constraint **strictly**, with room to spare. Not "at the limit", strictly inside it. When Slater holds, the duality gap is zero and an optimal multiplier exists. When it fails, the optimum may still exist while no finite price does, and a solver asked for one will return garbage or diverge.

### Slater's condition, and what it looks like when it breaks

Change the mandate to a shape risk committees actually write. Maximise expected return, fully invested, subject to a volatility ceiling:

$$\max_w \; \mu^\top w \quad \text{s.t.} \quad \mathbf{1}^\top w = 1, \quad \sqrt{w^\top \Sigma w} \le s.$$

This book's minimum achievable volatility is 8.0765%. Set ${s}$ anywhere above that and Slater holds, because portfolios exist strictly inside the ceiling. Set ${s}$ exactly at 8.0765% and exactly one portfolio is feasible, the minimum-variance one, with no strict interior at all. Slater fails.

The consequence is sharp rather than vague. Stationarity at that point would require ${\mu}$ to equal a constant vector plus a multiple of ${\Sigma w}$, which for the minimum-variance portfolio means all three sleeves must have identical expected returns. They have 8%, 5% and 3%. So the optimum exists and is perfectly well defined, and **no finite multiplier exists**. There is no price.

![Shadow price of a volatility limit rising without bound as the limit approaches the minimum achievable volatility](/imgs/blogs/convex-duality-shadow-prices-math-for-quants-5.webp)

Approach the wall and the price runs away, as that figure shows. At an 8.10% ceiling, 2.4 basis points above the minimum, the shadow price reads **\$17.1m per year per point of volatility** on the \$500m book. At 9% it is \$2.95m, at 10% \$2.21m, at 12% \$1.76m. The 8.10% number is not information about how much money the mandate is leaving on the table. It is information about how close the mandate sits to a geometric wall. The desk reading: a shadow price an order of magnitude larger than its neighbours is usually a symptom, and the right response is to ask whether the constraint set still has an interior.

## Complementary slackness, and how to read a solver's dual output

The KKT conditions bundle everything a convex optimum must satisfy: **stationarity** (the Lagrangian's gradient vanishes), **primal feasibility** (the portfolio obeys the rules), **dual feasibility** (${\lambda \ge 0}$ on inequalities), and **complementary slackness** (${\lambda_i \cdot \text{slack}_i = 0}$ for every constraint). Under convexity plus Slater they are necessary and sufficient, which is the licence every production solver runs on.

Complementary slackness is the one you read every morning. It says a constraint is either tight with a positive price, or loose with a price of exactly zero, and never both. So a solver's dual output is a sorted list of what is actually costing you.

![Table of three constraints showing which bind with a positive shadow price and which is slack at zero](/imgs/blogs/convex-duality-shadow-prices-math-for-quants-3.webp)

#### Worked example 2: two limits bite, one does not

Keep the gross limit at 125% and add two concentration caps: credit no more than 45% of capital, equity no more than 60%. Solving the KKT system with both caps active gives ${\lambda = 0.0045493}$ on gross and ${\nu = 0.0036175}$ on the credit cap, with weights 0.350073, 0.450000 and 0.449927, so **\$175.0m equity, \$225.0m credit, \$225.0m rates**. SLSQP reproduces those weights to eight decimals.

Read the three constraints straight off, as in that figure:

- **Gross exposure at 125.0%**, on the limit. Price ${0.0045493 \times \$5{,}000{,}000 = \$22{,}747}$ per point per year. Binding.
- **Credit sleeve at 45.0%**, on the limit. Price ${0.0036175 \times \$5{,}000{,}000 = \$18{,}088}$ per point per year. Binding.
- **Equity sleeve at 35.0%**, against a 60% cap. Price **\$0**. Slack, and no amount of arguing with compliance about it will earn a cent.

The credit cap costs the book \$57,316 a year: certainty equivalent falls from \$17,886,866 to \$17,829,550. There is a second-order lesson in the first row. Tightening the credit cap **lowered** the gross limit's price from \$27,727 to \$22,747 per point, because a book forced away from its preferred mix has less use for the marginal unit of notional. Shadow prices are a joint output. Quoting one without saying what else was in the problem is meaningless.

One practical note on active sets. If you guess wrong about which constraints bind and force the credit cap to hold with equality at 60%, the algebra happily returns ${\nu = -0.0049445}$. A negative multiplier on an inequality violates dual feasibility, and that sign is exactly how you learn your guess was wrong: at the true optimum credit sits at 51.3%, comfortably inside a 60% cap, and its price is zero.

## The mean-variance case: what a long-only multiplier confesses

Long-only constraints deserve their own reading, because their multipliers tell you something the weights cannot: **what the optimiser wanted to do and was not allowed to**.

Suppose the research view on rates turns negative, to an expected excess return of ${-1\%}$, with the book now fully invested at 100% net rather than levered. Attach a multiplier ${\nu_i \ge 0}$ to each non-negativity constraint ${w_i \ge 0}$. Stationarity reads

$$\mu_i - \gamma(\Sigma w)_i - \lambda + \nu_i = 0,$$

so ${\nu_i}$ is the rate at which the objective would improve per unit of shorting you were permitted in name ${i}$.

#### Worked example 3: the \$3m a year that "no shorting" costs

Solve with shorting allowed and the optimiser goes to weights 0.453227, 1.000356 and ${-0.453583}$: **\$226.6m equity, \$500.2m credit, and a \$226.8m short in rates**, for a certainty equivalent of \$19,004,197 a year. Note that this is 190.7% gross, so a long-only ban is quietly also a leverage ban.

Impose long-only and the answer becomes exactly 0.3675 and 0.6325 with rates pinned at zero: **\$183.75m equity, \$316.25m credit, \$0 rates**, certainty equivalent \$16,002,250 a year. SLSQP agrees to ten decimals. The budget multiplier is ${\lambda = 0.002984}$ and the multiplier on the rates non-negativity constraint is

$$\nu_3 = 0.0264732 \quad \Longrightarrow \quad 0.0264732 \times \$5{,}000{,}000 = \$132{,}366 \text{ per year per point.}$$

So the first 1% of capital of permitted shorting in rates is worth \$132,366 a year, nearly five times the gross limit's price, and the long-only mandate as a whole costs **\$3,001,947 a year** on this view. That gap is the number to take to an investment committee, not the observation that the optimiser "wanted to short rates".

These multipliers are not bookkeeping. Jagannathan and Ma (2003) showed that a long-only minimum-variance portfolio is exactly the **unconstrained** solution for a modified covariance matrix built from those multipliers, which is why long-only constraints empirically behave like covariance shrinkage and often improve out-of-sample risk. The multiplier is a real object with real statistical consequences.

## The null result: a shadow price is a tangent, not a price list

Here is where the technique misleads, and it misleads in the direction of optimism.

The sensitivity theorem gave ${\lambda^\star = \mathrm{d}p^\star/\mathrm{d}u}$ at ${u = 0}$. A derivative prices an infinitesimal relaxation. Quote it for a large one and you are extrapolating a tangent line off a curve that bends away from you, because for a concave maximisation the value function ${p^\star(u)}$ is concave. For this problem it is an exact parabola:

$$p^\star(L) = \frac{D}{2\gamma} + \frac{B}{A}L - \frac{\gamma}{2A}L^2, \qquad D = C - \frac{B^2}{A} = 0.067661,$$

which makes the price fall linearly, ${\lambda(L) = B/A - (\gamma/A) L}$, at ${\gamma/A = 0.026092}$ per unit of leverage. It hits zero at ${L^\star = B/\gamma = 1.46253}$, which is the leverage the book would choose if nobody stopped it. Call the distance from the limit to that point the **room**, here 21.25 points of gross.

![Two curves against gross exposure: the straight tangent implied by the shadow price and the concave curve of what relaxation is actually worth](/imgs/blogs/convex-duality-shadow-prices-math-for-quants-4.webp)

#### Worked example 4: relaxing the gross limit all the way

Against the \$27,727 per point the price quotes, the true annual gain from relaxing the 125% limit runs:

| Relaxation | Tangent says | Actually worth | Ratio |
| --- | --- | --- | --- |
| +1 point | \$27,727 | \$27,075 | 97.6% |
| +5 points | \$138,635 | \$122,327 | 88.2% |
| +10 points | \$277,269 | \$212,039 | 76.5% |
| +21.25 points (all the way) | \$589,285 | \$294,643 | 50.0% |

The decay has a closed form worth memorising. For a quadratic objective the ratio of the true gain to the tangent's claim over a relaxation of ${\Delta}$ is

$$\frac{\text{actual}}{\text{tangent}} = 1 - \frac{\Delta}{2\,\Delta_{\max}},$$

where ${\Delta_{\max}}$ is the room. Every row above is that formula with ${\Delta_{\max} = 0.2125}$. Walk the whole room and the ratio is exactly one half: \$294,643 against \$589,285 (unrounded, \$294,642.63 and \$589,285.25, exactly two to one). The same factor appeared in worked example 2, where the credit cap's \$18,088 per point implies \$114,632 over the 6.3376 points it was squeezed but actually costs \$57,316, and in worked example 3, where \$132,366 per point over 45.4 points implies roughly \$6.0m but the ban actually costs \$3,001,947.

**Half the quoted value, whenever the constraint is relaxed all the way, for any mean-variance objective.** That is the null result, and it is not a small correction. A researcher who takes a \$27,727 per point price into a meeting and multiplies by the 21 points they want has just overstated the case by 100%.

The volatility example from the Slater section is the violent version of the same defect. At an 8.10% ceiling the price reads \$17.1m per point, but moving the ceiling from 8.10% to 9.10% is actually worth \$4,651,487, about 27% of the claim. At a 10% ceiling the price reads \$2,205,651 per point and a full point is worth \$2,043,639, or 93%. The tangent is trustworthy far from the wall and worthless near it, and the shadow price alone does not tell you which regime you are in. You need the room.

## Where this earns its place on a desk

Three uses, in order of how often they come up.

**Pricing a limit.** Risk asks what the gross cap costs. The answer is a rate and a total: \$27,727 a year per point at the margin, \$294,643 a year to remove it entirely. Quote both, because the second is the one being negotiated.

**Choosing which fight to have.** With several binding constraints, rank them by price multiplied by the room, not by price. In worked example 2 the gross limit prices higher than the credit cap, but the general case flips: a constraint quoting \$17m per point with two basis points of room is worth less than one quoting \$2m per point with five points of room.

**Spotting alpha that is really constraint-fighting.** When a book's performance is dominated by large multipliers, much of the deviation from the raw signal is the constraint set, not the forecast. Grinold and Kahn's fundamental law, refined into the *transfer coefficient* by Clarke, de Silva and Thorley (2002), makes this explicit: constraints cap the fraction of a forecast that reaches the portfolio. A rising multiplier on a long-only constraint is a direct measurement of signal being thrown away.

## Common misconceptions

**"The multiplier is just a solver artefact."** It is the derivative of the value function, it has units, and in worked example 3 it was \$132,366 a year per point of capital. Jagannathan and Ma's result goes further: the multipliers on long-only constraints reconstruct a modified covariance matrix, so they change the portfolio's statistical behaviour, not just its bookkeeping.

**"A bigger shadow price means a more important constraint."** It means a more **expensive** one at the margin, which is not the same claim. The 8.10% volatility ceiling priced at \$17.1m per point, five times anything else in this post, and a full point of relief was worth \$4.65m. A huge price often measures proximity to a wall where the feasible set is about to vanish, which is a reason to question the mandate rather than to buy the relaxation.

**"Duality is a theoretical nicety."** It is what interior-point solvers compute. They walk the primal and dual together and stop when the gap is small, so the duality gap *is* the convergence criterion, and the multipliers arrive at no extra cost. Weak duality is also the only honest certificate available on a hard problem: for a cardinality constraint, "hold at most ${K}$ names", the feasible set is not convex, strong duality fails, and the dual value is a genuine bound on what you are leaving on the table rather than a price you can act on.

## Sources and further reading

- Stephen Boyd and Lieven Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004. Chapter 5 is the canonical treatment: 5.2.3 for Slater's condition, 5.5.3 for the KKT conditions, 5.6.3 for the perturbation and sensitivity analysis behind the shadow price. Free at [stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/).
- Richard Grinold and Ronald Kahn, *Active Portfolio Management*, 2nd edition, McGraw-Hill, 2000. Chapters 5 and 14 for constrained portfolio construction and the fundamental law.
- Roger Clarke, Harindra de Silva and Steven Thorley, "Portfolio Constraints and the Fundamental Law of Active Management", *Financial Analysts Journal* 58(5), 2002, 48-66. The transfer coefficient.
- Ravi Jagannathan and Tongshu Ma, "Risk Reduction in Large Portfolios: Why Imposing the Wrong Constraints Helps", *Journal of Finance* 58(4), 2003, 1651-1683. Long-only multipliers as covariance shrinkage.
- Harry Markowitz, "Portfolio Selection", *Journal of Finance* 7(1), 1952, 77-91.
- Related posts here: [the Lagrangian and the KKT conditions](/blog/trading/math-for-quants/lagrangian-kkt-conditions-math-for-quants), [quadratic and convex programming in practice](/blog/trading/math-for-quants/quadratic-convex-programming-math-for-quants), [convexity and Jensen's inequality](/blog/trading/math-for-quants/convexity-jensen-math-for-quants), and [mean-variance optimization and the efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants).

The portfolio figures in this post are illustrative arithmetic on assumed inputs, computed exactly with rational arithmetic and cross-checked against SLSQP and a trust-region interior-point solver. They are not quotes from any real book.

## In the interview room and on the desk

The question arrives as a statement with a trap inside it: *"your optimiser says this limit costs 40 basis points, what does that mean?"*

Answer in this order. First, the unit: 40 basis points of **what, per unit of what**. Basis points of annual certainty-equivalent return per percentage point of gross exposure is a completely different number from basis points of tracking error per unit of turnover, and until you say which, you have said nothing. Second, the source: it is the Lagrange multiplier on that constraint, which by the sensitivity theorem is the derivative of the optimal objective with respect to the limit. Third, and this is the sentence that separates candidates, say **it is a local derivative**. It prices the next basis point of relaxation. It does not price the relaxation anybody is actually asking for, and for a mean-variance objective, relaxing a constraint all the way is worth exactly half what the multiplier implies.

Then close the loop with the two checks a senior researcher does automatically. Is it binding? Complementary slackness means a zero price is a constraint nobody should be arguing about, and a positive price means the optimiser is genuinely pressed against it. And is the price even meaningful? Strong duality requires convexity plus a constraint qualification, so if the problem has integer or cardinality constraints, the dual value is a bound rather than a price, and if a multiplier comes back enormous relative to its neighbours, suspect that Slater's condition is nearly failing and the mandate has squeezed the feasible set toward a point.

The trap is quoting the multiplier as though it were a price list. A candidate who says "40 basis points per point of gross, so give me ten points and I will make you 400" has demonstrated they do not know what a derivative is, and the honest answer of roughly 300 is available from one line of algebra. The complementary trap is treating a huge multiplier as a huge opportunity when it is usually a signal that the feasible set is collapsing.

Citadel, Two Sigma, and any portfolio-construction, risk, or optimisation seat weight this heavily, because reading the dual output correctly is most of what separates running an optimiser from understanding one. Jane Street and Jump ask it less often, but the units-first habit transfers to every marginal-value question they do ask.
