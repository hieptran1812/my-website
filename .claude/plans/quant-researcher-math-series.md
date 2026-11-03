# Quant Researcher Math: the math that finds alpha and defends it

47-post series for readers applying to **Quant Researcher** roles at top firms
(Jane Street, Citadel / Citadel Securities, Two Sigma, HRT, Jump, D. E. Shaw,
Renaissance, Millennium, Squarepoint, XTX, G-Research, Optiver/SIG research desks).

Status: **PLANNED 2026-11-03, not started.** Wave 1 next.

## Why this series exists (gap analysis, measured 2026-11-03)

The repo already holds 83 `trading/math-for-quants` posts (Math for Quants 50 +
Senior Quant Math 33) and ~50 `trading/quantitative-finance` posts (Quant Interview
39 + quant-research). Those cover the eight core pillars, measure theory, stochastic
calculus, RMT, causal inference, IV, concentration, GPs, OT, duality, Hawkes, rough vol.
**This series must not re-teach any of that.** It links to it as prerequisites.

What a QR is actually tested on in research rounds, take-homes and the research
presentation, and which appears **zero** times in `content/blog/trading/` (grep count):

| Missing term | Hits | Missing term | Hits |
| --- | --- | --- | --- |
| Fama-MacBeth / Shanken | 0 | GMM / Hansen J-test | 0 |
| stochastic discount factor / HJ bounds | 0 | delta method | 0 |
| Stambaugh bias / IVX | 0 | Romano-Wolf / SPA | 0 |
| quantile regression | 0 | conformal prediction | 0 |
| Rademacher / PAC-Bayes | 0 | double descent / virtue of complexity | 0 |
| kernel ridge / random features | 0 | mutual information | 0 |
| VAR / impulse response | 0 | HAR / two-scale realized vol | 0 |
| Hayashi-Yoshida / Epps effect | 0 | propagator / Obizhaeva-Wang | 0 |
| Garleanu-Pedersen | 0 | universal portfolios / OGD | 0 |
| IPCA / conditional factors | 0 | queue position / fill models | 0 |

Thin (1 to 4 hits, mentioned never derived): Newey-West, HJB, Avellaneda-Stoikov,
Huber, lasso, markouts, Malliavin, COS/Carr-Madan, AAD, Grinold's fundamental law.

The through-line: **at a signal-to-noise ratio near 0.01, the job is not fitting a
model, it is knowing whether the number you found is real.** Every post is judged by
two tests: (1) does it let the reader *find* a class of alpha they could not find
before, and (2) can they *defend* it in front of a skeptical senior researcher.

### Goal: produce a researcher who finds alpha, not one who passes interviews

User requirement (2026-11-03): the math must make the reader a top QR who researches
top alpha. So every post is written around a research loop, not a syllabus:

```
economic mechanism -> data -> signal construction -> statistical evidence
-> survives costs/capacity -> combines with the book -> monitored until it dies
```

Each track owns a stage of that loop: I (mechanism + construction), A/C (evidence),
B (is it new vs known factors), D (time behaviour), E (survives execution),
F (enters the book), G (options-desk alpha), H (communicating it).

Honesty rule, stated in the hub and never broken: math is necessary, not sufficient.
Top alpha also comes from data access, infrastructure, and an economic reason someone
is on the other side of the trade. Posts never promise returns, never present a
synthetic backtest as evidence of a live edge, and every "alpha idea" is labeled as
a research hypothesis with the test that would kill it.

## Conventions

- Engine: `finance-writer`, depth `deep-dive`.
- Folder: `content/blog/trading/math-for-quants/`, slug suffix `-math-for-quants`
  (folder path is the URL; see memory `blog-slug-is-folder-path`).
- Tag every post `quant-researcher-math` so the hub (H4) can list the series.
- Length: **3,000 to 4,000 prose words** via `WORD_MIN=3000 WORD_MAX=4000` (same
  override as Math for AI / CVPR; math-dense derivations do not fit in 2–3k).
- 6 to 8 figures, at least half pictorial (memory `post-length-and-pictorial-figures`);
  1 animated figure per post where a dynamic process is the point (impact decay,
  double-descent curve, CUSUM alarm).
- No em dashes; English; literal LaTeX with the digit-span and `\lt` rules.
- Verify: `WORD_MIN=3000 WORD_MAX=4000 bash .claude/skills/finance-writer/scripts/verify-finance-post.sh <post> <slug> deep-dive`, through `post-verifier`.
- Then `python3 .claude/scripts/check-printed-arithmetic.py <post>`: the verifier never
  recomputes a number (memory `verify-gates-never-check-arithmetic`).

### Required post skeleton (beyond finance-writer defaults)

1. **The research question it answers** (one paragraph, the desk situation).
2. **Intuition → derivation ladder**: every result derived, not quoted. Proof sketches
   where the interview asks for them.
3. **Synthetic-data experiment**: a short, runnable numpy/statsmodels snippet whose
   printed output appears in the post. Numbers come from running it, never invented.
   Fix the seed. This is the honesty spine: no fabricated backtest, no fabricated
   firm-specific claim.
4. `## From this math to an alpha`: one concrete signal hypothesis this math lets you
   build or test (mechanism, who is on the other side, construction, the kill test),
   run on the synthetic data from step 3. Labeled a hypothesis, never a claimed edge.
5. **Where it breaks**: compute the failure case (memory: "compute the null result").
6. `## In the research interview`: 3 to 5 questions **labeled representative** (never
   attributed to a specific firm unless sourced to a public, citable writeup), each with
   the trap and what a strong answer contains.
7. `## Take-home drill`: one exercise in the shape of a QR take-home, with the check
   the grader will run.
8. Prerequisites box linking the existing posts named in the table row.

## Series map

Columns: slug stem (append `-math-for-quants`) | core math | the QR use | prerequisites already on disk.

### Track A: Inference you defend in review (6)

| # | Slug stem | Core math | QR use | Prereqs |
| --- | --- | --- | --- | --- |
| A1 | `delta-method-sharpe-ratio-standard-error` | delta method, Lo (2002) SE of Sharpe, autocorrelation correction, PSR | "is Sharpe 1.4 on 3 years real?" | estimators, LLN-CLT, deflated-Sharpe (quant-research) |
| A2 | `robust-standard-errors-hac-newey-west` | sandwich estimator, HAC, Bartlett kernel, bandwidth, overlapping returns, Hansen-Hodrick | t-stats on overlapping-horizon forecasts | regression-ols-gls, stationarity |
| A3 | `fama-macbeth-panel-regressions` | two-pass regression, Shanken correction, errors-in-variables, clustered SE (firm/time) | pricing a characteristic in the cross-section | regression, A2 |
| A4 | `gmm-generalized-method-of-moments` | moment conditions, efficient weighting, Hansen J-test, two-step vs iterated | estimating models without a likelihood | mle-method-of-moments, A2 |
| A5 | `data-snooping-white-reality-check-spa-romano-wolf` | FWER vs FDR, BH/BY under dependence, White RC, Hansen SPA, Romano-Wolf stepdown, t>3 | testing 2,000 signals honestly | hypothesis-testing, bootstrap, overfitting-purged-cv |
| A6 | `predictive-regressions-stambaugh-bias` | persistent regressors, Stambaugh bias derivation, IVX, long-horizon R² illusion, Campbell-Thompson OOS R² | "does dividend yield predict returns?" | ar-ma-arima, A2 |

### Track B: Empirical asset pricing math (5)

| # | Slug stem | Core math | QR use | Prereqs |
| --- | --- | --- | --- | --- |
| B1 | `stochastic-discount-factor-hansen-jagannathan` | SDF, Euler equation, HJ bound derivation, HJ distance | one equation behind every factor model | martingales-risk-neutral, change-of-numeraire |
| B2 | `linear-factor-models-apt-grs-test` | APT, time-series vs cross-sectional tests, GRS F-test, alpha as mispricing | "is your signal just beta to known factors?" | B1, A3, factor-risk-model-build |
| B3 | `portfolio-sorts-spanning-regressions` | univariate/double sorts, spanning tests, characteristic vs covariance, microcap weighting | standard anomaly evaluation protocol | B2 |
| B4 | `conditional-factor-models-ipca` | instrumented PCA, alternating least squares, characteristics as betas | modern factor construction | eigendecomposition-pca, svd-least-squares |
| B5 | `machine-learning-cross-section-low-snr` | Gu-Kelly-Xiu setup, OOS R² near 0.4%, why monthly R² is tiny yet Sharpe large, ensembling | what ML buys in return prediction | tree-models-cross-sectional (quant-research), C2, C3 |

### Track C: Learning theory at low signal-to-noise (7)

| # | Slug stem | Core math | QR use | Prereqs |
| --- | --- | --- | --- | --- |
| C1 | `lasso-high-dimensional-regression-theory` | restricted eigenvalue, oracle inequality, support recovery, post-selection inference | 500 features, 2,000 days | regression-regularized, concentration-inequalities |
| C2 | `ridge-kernel-ridge-random-features` | ridge as shrinkage on eigen-directions, representer theorem, kernel trick, random Fourier features | nonlinear signals without overfitting | svd-least-squares, gaussian-processes |
| C3 | `generalization-bounds-vc-rademacher-pac-bayes` | uniform convergence, Rademacher complexity, PAC-Bayes, why bounds are vacuous and still useful | the number of trials you can afford | concentration-inequalities |
| C4 | `double-descent-virtue-of-complexity` | ridgeless regression, interpolation threshold, benign overfitting, Kelly-Malamud-Zhou | when bigger models help on returns, and the caveats | C2, C3 |
| C5 | `conformal-prediction-distribution-free-intervals` | split conformal, exchangeability, weighted/adaptive conformal under drift | calibrated risk forecasts | bootstrap |
| C6 | `quantile-regression-robust-m-estimators` | check loss, Huber, influence functions, breakdown point | fat-tailed targets, outlier days | tail-risk-evt, estimators |
| C7 | `mutual-information-entropy-signal-research` | entropy, MI, estimators (KSG, binning bias), data-processing inequality, information coefficient vs MI | nonlinear dependence screening | information-geometry, copulas |

### Track D: Time series a researcher actually fits (5)

| # | Slug stem | Core math | QR use | Prereqs |
| --- | --- | --- | --- | --- |
| D1 | `var-models-impulse-response-granger` | VAR(p), stability, Cholesky vs generalized IRF, Granger causality and its limits | lead-lag across assets | ar-ma-arima, cholesky |
| D2 | `realized-volatility-microstructure-noise-har` | RV, noise bias, two-scale/realized kernel, signature plot, HAR | intraday vol forecasting | arch-garch, quadratic variation (ito) |
| D3 | `asynchronous-data-epps-effect-hayashi-yoshida` | non-synchronous trading bias, Epps effect, Hayashi-Yoshida estimator | correlation at high frequency | covariance-matrix, D2 |
| D4 | `change-point-detection-cusum-bayesian-online` | CUSUM, Page's test, ARL, Bayesian online change-point | regime breaks, signal death detection | regime-switching-hmm, alpha-lifecycle |
| D5 | `nonstationary-learning-exponential-forgetting` | EWMA as weighted LS, effective sample size, bias-variance of the half-life, recursive least squares | choosing the lookback | kalman-filter, D4 |

### Track E: Microstructure and execution math (6)

| # | Slug stem | Core math | QR use | Prereqs |
| --- | --- | --- | --- | --- |
| E1 | `kyle-glosten-milgrom-derivations` | Kyle equilibrium (lambda), Glosten-Milgrom spread via Bayes | adverse selection from first principles | conditional-expectation, bayesian-inference |
| E2 | `propagator-models-transient-impact` | Bouchaud propagator, Obizhaeva-Wang, square-root law, no-dynamic-arbitrage | impact model for a backtest | strategy-capacity-market-impact |
| E3 | `stochastic-control-hjb-avellaneda-stoikov` | HJB derivation, verification, Avellaneda-Stoikov quotes, inventory penalty | market-making research at HFT firms | dynamic-programming-optimal-execution, ito |
| E4 | `transaction-cost-aware-portfolios-garleanu-pedersen` | quadratic tcost, aim portfolio, partial adjustment rate, multi-horizon alpha | turning signals into trades | quadratic-convex-programming, E2 |
| E5 | `queue-position-fill-probability-survival` | queue dynamics, survival/hazard models, censoring, Kaplan-Meier | fill models for passive orders | order-book-simulator (quant-research), hawkes |
| E6 | `markouts-adverse-selection-measurement` | markout curves, conditioning bias, toxicity, VPIN and its critique | measuring whether you are being picked off | E1, E5 |

### Track F: From forecast to position (4)

| # | Slug stem | Core math | QR use | Prereqs |
| --- | --- | --- | --- | --- |
| F1 | `fundamental-law-active-management-derivation` | Grinold IR = IC·√BR derivation, transfer coefficient, where breadth is not independent | how many bets you really have | evaluating-alpha-signals, mean-variance |
| F2 | `optimizer-error-maximization-alpha-alignment` | estimation error amplification, Michaud, alpha-risk misalignment, factor alignment | why the optimizer loves your mistakes | robust-regularized-portfolios, shrinkage |
| F3 | `online-learning-regret-universal-portfolios` | regret, OGD, multiplicative weights, Cover's universal portfolio, follow-the-regularized-leader | adaptive signal weighting without a model | stochastic-gradient-optimizers, convexity |
| F4 | `bandits-strategy-capital-allocation` | UCB, Thompson sampling, non-stationary bandits, regret lower bound | allocating capital across signals/PMs | F3, bayesian-inference |

### Track G: Pricing-research specialist (options desks) (4)

| # | Slug stem | Core math | QR use | Prereqs |
| --- | --- | --- | --- | --- |
| G1 | `characteristic-functions-fourier-pricing-cos` | Lévy-Khintchine recap, Carr-Madan FFT, COS method, damping, truncation error | fast pricing under Heston/jumps | levy-processes, jump-diffusion-stoch-vol |
| G2 | `malliavin-greeks-pathwise-likelihood-ratio` | pathwise vs likelihood-ratio vs Malliavin weights, digital greeks | Monte Carlo greeks that do not explode | monte-carlo-variance-reduction, girsanov |
| G3 | `arbitrage-free-vol-surface-svi` | static arbitrage (calendar, butterfly), Durrleman condition, SVI/SSVI fitting | building a surface a trader trusts | volatility-surface (quant-research) |
| G4 | `automatic-differentiation-aad-risk` | forward vs reverse mode, tape, checkpointing, AAD for greeks at scale | full-book risk in seconds | matrix-calculus |

### Track I: The alpha-research craft (6)

Gap-measured: `limits to arbitrage`, `lead-lag`, text `embedding` signals = 0 hits;
winsorize/orthogonalize = 1 hit each. This track is what turns the math into alpha.

| # | Slug stem | Core math | QR use | Prereqs |
| --- | --- | --- | --- | --- |
| I1 | `where-alpha-comes-from-limits-to-arbitrage` | risk premium vs mispricing vs liquidity provision, limits-to-arbitrage model (noise-trader risk, capital constraints), who pays you and why they keep paying | choosing ideas with a mechanism, the first filter on every hypothesis | B1, causal-inference |
| I2 | `signal-construction-transforms-rank-winsorize-neutralize` | rank/Gaussianize, winsorization bias, cross-sectional z-scores, neutralization as projection onto factor null space, orthogonalization order effects | the 80% of signal quality decided before any model | conditional-expectation-projection, regression |
| I3 | `signal-horizon-ic-decay-term-structure` | IC as a function of horizon, decay fitting, optimal holding period vs turnover cost, overlapping-horizon blending | matching a signal to its natural horizon | A2, E4, alpha-lifecycle |
| I4 | `lead-lag-networks-cross-asset-signals` | cross-correlation at lags, economic-link graphs (supplier/customer, sector), graph diffusion, spurious lead-lag from asynchronicity | a family of cross-sectional alphas | D1, D3 |
| I5 | `text-and-alternative-data-signals-math` | embeddings as features, noisy-label sentiment, point-in-time alignment, coverage bias, decay of a public dataset | alt-data alpha without look-ahead | filtrations-no-lookahead, C1, C7 |
| I6 | `crowding-and-alpha-capacity-detection` | crowding metrics (short interest, co-holdings, factor return correlation), unwind dynamics, capacity curve from impact | knowing when your alpha is everyone's alpha | strategy-capacity, E2, D4 |

### Track H: The QR job itself (4)

| # | Slug stem | Core math | QR use | Prereqs |
| --- | --- | --- | --- | --- |
| H1 | `research-interview-derivation-gauntlet` | the derivations asked at the whiteboard, mapped to posts in this series, with time budgets | rehearsal map | all |
| H2 | `research-take-home-end-to-end` | one synthetic-data alpha taken through the full loop: I1 mechanism, I2 construction, A1/A2/A5 evidence, B2 spanning, I3 horizon, E4 costs, I6 capacity | the deliverable a grader reads | quant-research-writeup-killing-ideas |
| H3 | `defending-a-result-research-presentation` | the referee questions (snooping, SE, capacity, decay) each answered with the math | the onsite research talk | A-track, F1 |
| H4 | `quant-researcher-math-roadmap` | hub: 12-week plan over the ~170 existing + new posts, by firm archetype (HFT / stat-arb / options / fundamental-quant) | entry point, written last | all |

## Waves (6 posts each, dependency order)

| Wave | Posts | Theme |
| --- | --- | --- |
| 1 | A1–A6 | Inference everything else cites |
| 2 | I1, I2, I3, B1, B2, B3 | Where alpha comes from + is it new vs known factors |
| 3 | B4, C1, C2, C3, C4, I5 | Regularization, learning theory, alt-data signals |
| 4 | B5, C5, C6, C7, D1, D3 | Low-SNR ML + lead-lag prerequisites |
| 5 | D2, D4, D5, F3, I4, I6 | Time behaviour, lead-lag and crowding alphas |
| 6 | E1–E6 | Microstructure and execution |
| 7 | F1, F2, F4, G1, G2, G3 | Position sizing + pricing research |
| 8 | G4, H1, H2, H3, H4 | AAD + the job itself + hub (5) |

Per CLAUDE.md: one session per wave, split again when the drafts hit disk; gate each
post through one aggregator subagent; commit only that wave's `.md` + `.webp` with
explicit paths; `git pull --rebase --autostash` before push; re-run
`node .claude/scripts/token-report.mjs` after each wave.

Before each wave: check every slug is absent from `content/blog/` (memory
`check-post-exists-before-figure-pipeline`). Lessons from the sibling series (arithmetic
audit, calibrating two methods on a known case, display fence alone on its line):
`git show HEAD:.claude/plans/senior-quant-math-brief.md`.

## Progress

- [ ] Wave 1: A1–A6
- [ ] Wave 2: I1, I2, I3, B1, B2, B3
- [ ] Wave 3: B4, C1–C4, I5
- [ ] Wave 4: B5, C5–C7, D1, D3
- [ ] Wave 5: D2, D4, D5, F3, I4, I6
- [ ] Wave 6: E1–E6
- [ ] Wave 7: F1, F2, F4, G1–G3
- [ ] Wave 8: G4, H1–H4
