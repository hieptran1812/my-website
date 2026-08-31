# Senior Quant Math: the gap-filling series

**Why this series exists.** A 101-post audit of `trading/math-for-quants` and
`trading/quantitative-finance` found the eight core pillars already covered:
measure theory, stochastic calculus, inference, linear algebra, optimization,
time series, financial ML, derivatives pricing. What is missing is the layer that
separates *running a model* from *knowing where the model is wrong*: estimation
under high dimensions, causality, Bayesian computation, processes beyond
diffusion, and the numerics that decide whether a price is real.

Measured gaps (occurrences across `content/blog/trading/`, 2026-08-30):
`random matrix` 0 · `causal inference` 0 · `instrumental variable` 0 ·
`concentration inequality` 0 · `Gaussian process` 0 · `Metropolis` 0 ·
`optimal transport` 0 · `convex duality` 0 · `Milstein` 0 · `Levy process` 0 ·
`Fokker-Planck` 0 · `Doob` 0 · Marchenko-Pastur 1 post · Hawkes 1 · HMM 1.

## Who this series is for

Explicitly: a reader trying to pass the research rounds at **Jane Street,
Citadel / Citadel Securities, Two Sigma, WorldQuant and Jump Trading**. The 101
posts already in `math-for-quants` and `quantitative-finance` cover the interview
*mechanics* well (probability puzzles, mental math, market-making games, Kelly,
order-book simulator, C++ latency). Microstructure is also well covered already:
Glosten-Milgrom appears in 20 posts, adverse selection in 71, inventory risk in
46, Almgren-Chriss in 4.

What this series adds is the layer those firms actually probe once you are past
the puzzles: **can you say why a number is wrong, and what you would do about it.**

### What each firm weights

| Firm | What the round is really testing | Posts that carry the weight |
| --- | --- | --- |
| **Jane Street** | fast probabilistic reasoning, sequential decisions under adverse selection, betting sizing | 3 (concentration), 24 (optimal stopping), 14 (stopping times) |
| **Citadel Securities / Jump** | short-horizon prediction, order flow, queueing, execution | 10 (Hawkes), 25 (order-book imbalance), 19-21 (numerics) |
| **Citadel (multi-strat)** | portfolio construction, risk models, defending an alpha | 1 (RMT), 2 (shrinkage), 4-6 (causality), 22 (duality) |
| **Two Sigma** | statistical learning, causality, high-dimensional inference, Bayesian modelling | 1, 3, 4, 5, 6, 7, 8, 9 |
| **WorldQuant** | alpha construction at scale, multiple testing, decay and turnover discipline | 1, 3, 26 (combining weak alphas) |

### The section every post in this series must carry

Matching the house convention already used by existing posts
(`## In the interview room and the take-home`), each post ends with:

`## In the interview room and on the desk` (250-400 words) covering:
1. the question this topic is actually asked as, in the firm's words, not the textbook's;
2. what a strong answer contains, in order;
3. the follow-up they push on, and the trap that makes a candidate look rigorous while being wrong;
4. which firms weight it most, one line, no hype.

## Conventions

- Skill: **finance-writer**, depth **`explainer`** (floor 3,500 words, ≥3 figures,
  ≥3 worked examples). Target 3,500–4,000 words per post.
- Directory: `content/blog/trading/math-for-quants/`, subcategory
  `Quantitative Finance`, slug suffix `-math-for-quants`.
- Every post cross-links 2–4 existing posts and never re-derives what they cover.
- Verify: `bash .claude/skills/finance-writer/scripts/verify-finance-post.sh <md> <slug> explainer`

## Track A — Estimation when the dimensions fight back (3)

1. `random-matrix-theory-covariance-cleaning-math-for-quants`
   Marchenko-Pastur, the eigenvalue bulk, clipping, and why a sample covariance
   with N≈T is mostly noise. **Highest-priority post in the series.**
2. `shrinkage-stein-paradox-math-for-quants`
   James-Stein, Ledoit-Wolf, and why the sample mean is inadmissible in 3+ dims.
3. `concentration-inequalities-sample-complexity-math-for-quants`
   Hoeffding, Bernstein, McDiarmid, union bound: how many trades before an edge
   is believable. Replaces the roadmap item in
   `probability-statistics-martingales-quant-series.md`.

## Track B — Causality, not correlation (3)

4. `causal-inference-alpha-research-math-for-quants`
   Confounders, colliders, DAGs, the back-door criterion, and what a regression
   coefficient does *not* mean.
5. `instrumental-variables-natural-experiments-math-for-quants`
   IV, 2SLS, exclusion restrictions, and the market events that act as instruments.
6. `event-studies-diff-in-diff-synthetic-control-math-for-quants`
   Abnormal returns done properly, parallel trends, synthetic controls.

## Track C — Bayesian computation (3)

7. `mcmc-metropolis-gibbs-math-for-quants`
   Why sampling beats integrating, Metropolis-Hastings, Gibbs, diagnosing chains.
8. `hierarchical-bayes-pooling-math-for-quants`
   Partial pooling across assets, sectors and regimes; shrinkage as a prior.
9. `gaussian-processes-kernels-math-for-quants`
   Kernels as covariance, GP regression for yield curves and vol surfaces.

## Track D — Processes beyond diffusion (4)

10. `hawkes-point-processes-order-flow-math-for-quants`
    Self-excitation, branching ratio, clustered trades and contagion.
11. `regime-switching-hidden-markov-math-for-quants`
    Markov-switching models, the forward-backward algorithm, regime inference.
12. `levy-processes-jumps-math-for-quants`
    Lévy-Khintchine, variance gamma, jumps as a first-class citizen.
13. `rough-volatility-fractional-brownian-math-for-quants`
    The Hurst exponent, roughness of realised vol, why H≈0.1 changes hedging.

## Track E — Stochastic calculus, completed (5)

14. `stopping-times-optional-stopping-math-for-quants`
    Stopping times, the optional stopping theorem, and why "quit while ahead" fails.
15. `martingale-representation-hedging-math-for-quants`
    Why a complete market is exactly one where every payoff is a stochastic integral.
16. `change-of-numeraire-math-for-quants`
    Pricing under the measure that makes the problem easy; forward measure.
17. `fokker-planck-kolmogorov-forward-math-for-quants`
    The density's own PDE, and the other side of Feynman-Kac.
18. `local-time-barriers-reflection-math-for-quants`
    Reflection principle, first-passage, and the math under barrier options.

## Track F — Numerics that decide the price (3)

19. `monte-carlo-variance-reduction-math-for-quants`
    Control variates, antithetics, importance sampling, quasi-MC. One rigorous
    home for what is currently scattered across eight posts.
20. `numerical-sdes-euler-milstein-math-for-quants`
    Strong vs weak convergence, Euler-Maruyama, Milstein, discretisation bias.
21. `finite-difference-pde-pricing-math-for-quants`
    Explicit/implicit/Crank-Nicolson, stability, Greeks, American exercise.

## Track G — Frontier tools (2)

22. `convex-duality-shadow-prices-math-for-quants`
    Lagrangian duality, strong duality, and reading constraints as prices.
    *Citadel: every position limit and risk bound has a dual price; a senior
    researcher reads the binding constraint off the dual.*
23. `optimal-transport-wasserstein-math-for-quants`
    Wasserstein distance and distributionally robust portfolios.

## Track H — Firm-facing gaps found in the coverage audit (3)

Added after re-scoping the series toward the five firms. Each fills a hole the
audit measured rather than a hole in the textbook.

24. `optimal-stopping-secretary-when-to-take-the-trade-math-for-quants`
    Secretary problem, house-hunting, the 1/e rule, and the trading version: when
    to take the fill you have rather than the one you hope for. Measured gap:
    `secretary problem` appears **once** in the whole corpus, and it is a Jane
    Street staple. Sits next to post 14, which does the measure-theoretic version.
25. `order-book-imbalance-short-horizon-prediction-math-for-quants`
    Queue imbalance as a predictor, the sign-flip horizon, why the edge decays in
    seconds. Measured gap: `order book imbalance` appears **zero** times, despite
    10 posts touching queue position. Citadel Securities and Jump.
26. `combining-weak-alphas-math-for-quants`
    Blending many low-IC signals: equal weight vs IC weight vs covariance-aware,
    the fundamental law with correlated signals, and why naive stacking
    over-counts. `building-an-alpha-signal` introduces this in one section; this
    post does it properly and leans on posts 1 and 2 for the covariance.

## Track S — What a senior actually owns (7)

**Why this is a track and not a series.** `trading/quant-careers` already has 45
posts covering the senior/career layer, including firm playbooks for Jane Street,
Citadel, Two Sigma/D.E. Shaw, WorldQuant and Jump/HRT, plus "What Senior Actually
Means at a Quant Firm", "Decision-Making Under Uncertainty: The Senior's Edge",
"Owning P&L and Owning Research", "Intellectual Honesty and Killing Your Own
Ideas" and "The IC vs Management Fork". A second career series would duplicate it.

What neither that series (behavioural) nor this one (mathematical) covers is the
**production research layer a senior is handed the keys to**. Measured gaps in
`content/blog/trading/` on 2026-08-30:

`live vs backtest` **0** · `alpha lifecycle / signal retirement` **0** ·
dedicated factor-risk-model post **0** (5 passing mentions only) ·
`model validation` 3 posts/4 mentions · `model governance` 2/8 ·
`performance attribution` 4/6 · `drawdown control` 3/4 · `feature store` 1.

(Note: an earlier grep reported "Barra" in 127 posts. That was `embarrassing`
matching as a substring. The factor-risk-model gap is real.)

27. `factor-risk-model-build-math-for-quants`
    From a returns panel to a covariance you can defend in a risk meeting:
    factor exposures, specific risk, and why the cleaned covariance from post 1
    is the input. *Citadel, Two Sigma, any portfolio-construction seat.*
28. `pnl-attribution-math-for-quants`
    Where the money actually came from: factor vs specific, alpha vs beta vs
    execution. The senior is the person who can decompose a P&L on demand.
29. `live-vs-backtest-divergence-math-for-quants`
    A diagnostic ladder for the most common senior question after a bad month:
    costs, capacity, regime, look-ahead, or the signal is simply dead. **0
    coverage today.**
30. `alpha-lifecycle-decay-retirement-math-for-quants`
    Monitoring decay on a live signal, the statistics of "is it dead or unlucky",
    and the discipline of retiring it. **0 coverage today.** *WorldQuant.*
31. `strategy-capacity-market-impact-math-for-quants`
    How much can this hold before impact eats the edge: the square-root law,
    turnover, and the capacity-adjusted Sharpe.
32. `model-validation-governance-math-for-quants`
    What an independent reviewer checks, and how to write a model document that
    survives it.
33. `risk-budgeting-drawdown-control-math-for-quants`
    Book-level risk budgets, vol targeting, and drawdown control that does not
    quietly become market timing.

## Waves

Ordered by hiring value, not by mathematical dependency.

- **W1 (in flight):** 1 RMT, 4 causal inference, 3 concentration
  *the three that appear in every Two Sigma / Citadel research round*
- **W2 (firm-facing):** 24 optimal stopping, 26 combining weak alphas, 25 order-book imbalance
  *one post each for Jane Street, WorldQuant, Jump*
- **W3:** 2 shrinkage, 5 instrumental variables, 6 event studies
- **W4:** 7 MCMC, 8 hierarchical Bayes, 9 Gaussian processes
- **W5:** 10 Hawkes, 11 regime switching, 12 Lévy, 13 rough vol
- **W6:** 14, 15, 16, 17, 18 (stochastic calculus completions)
- **W7:** 19, 20, 21 (numerics)
- **W8:** 22, 23 (frontier)
- **W9:** 27, 28, 29 (the risk-and-attribution core of Track S)
- **W10:** 30, 31, 32, 33 (the rest of Track S)

Commit per wave, explicit paths (`<slug>.md` + `public/imgs/blogs/<slug>-*.webp`),
never `git add -A`. Pull --rebase --autostash before every push.

## Tracker

| # | slug | wave | status |
|---|------|------|--------|
| 1 | random-matrix-theory-covariance-cleaning | W1 | **SHIPPED** f43eaba4 |
| 2 | shrinkage-stein-paradox | W2 | TODO |
| 3 | concentration-inequalities-sample-complexity | W1 | **SHIPPED** f43eaba4 |
| 4 | causal-inference-alpha-research | W1 | **SHIPPED** f43eaba4 |
| 5 | instrumental-variables-natural-experiments | W2 | TODO |
| 6 | event-studies-diff-in-diff-synthetic-control | W2 | TODO |
| 7 | mcmc-metropolis-gibbs | W3 | TODO |
| 8 | hierarchical-bayes-pooling | W3 | TODO |
| 9 | gaussian-processes-kernels | W3 | TODO |
| 10 | hawkes-point-processes-order-flow | W4 | TODO |
| 11 | regime-switching-hidden-markov | W4 | TODO |
| 12 | levy-processes-jumps | W4 | TODO |
| 13 | rough-volatility-fractional-brownian | W4 | TODO |
| 14 | stopping-times-optional-stopping | W5 | TODO |
| 15 | martingale-representation-hedging | W5 | TODO |
| 16 | change-of-numeraire | W5 | TODO |
| 17 | fokker-planck-kolmogorov-forward | W5 | TODO |
| 18 | local-time-barriers-reflection | W5 | TODO |
| 19 | monte-carlo-variance-reduction | W6 | TODO |
| 20 | numerical-sdes-euler-milstein | W6 | TODO |
| 21 | finite-difference-pde-pricing | W6 | TODO |
| 22 | convex-duality-shadow-prices | W7 | TODO |
| 23 | optimal-transport-wasserstein | W8 | TODO |
| 24 | optimal-stopping-secretary-when-to-take-the-trade | W2 | IN PROGRESS: 4 figures rendered, prose not started |
| 25 | order-book-imbalance-short-horizon-prediction | W2 | IN PROGRESS: 5 figures rendered + 5 scenes cached, prose not started |
| 26 | combining-weak-alphas | W2 | **SHIPPED** 4,396 words |
| 27 | factor-risk-model-build | W9 | TODO |
| 28 | pnl-attribution | W9 | TODO |
| 29 | live-vs-backtest-divergence | W9 | TODO |
| 30 | alpha-lifecycle-decay-retirement | W10 | TODO |
| 31 | strategy-capacity-market-impact | W10 | TODO |
| 32 | model-validation-governance | W10 | TODO |
| 33 | risk-budgeting-drawdown-control | W10 | TODO |
