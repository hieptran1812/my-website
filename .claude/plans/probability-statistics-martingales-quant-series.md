# Probability, Statistics and Martingales for Quant Trading

## Conventions

- Engine: `finance-writer`
- Depth: `deep-dive`
- Audience: curious beginner progressing toward quant practitioner
- Target directory: `content/blog/trading/math-for-quants/`
- Verify: `bash .claude/skills/finance-writer/scripts/verify-finance-post.sh <post> <slug> deep-dive`
- Every article must use English prose, literal LaTeX, no em dashes, sourced real-world figures, clearly labeled hypothetical examples, and WebP figures only.
- Existing posts are audited and adapted rather than duplicated. A post counts as shipped only after the finance verifier and `npm run blog-validate` pass.

## Series map

1. `probability-spaces-random-variables-math-for-quants` | Probability as a trading language: spaces, events, and P&L | COMPLETE, audited 2026-08-11
2. `probability-distributions-for-markets-math-for-quants` | The probability distributions that markets actually use | EXISTING, audit required
3. `bayesian-inference-traders-math-for-quants` | Bayesian inference for traders | EXISTING, audit required
4. `expectation-variance-moments-math-for-quants` | Expectation, variance, and higher moments | EXISTING, audit required
5. `covariance-matrix-linear-algebra-math-for-quants` | Covariance, correlation, and portfolio dependence | EXISTING, audit required
6. `estimators-bias-variance-consistency-math-for-quants` | Estimators, bias, variance, and consistency | EXISTING, audit required
7. `law-large-numbers-central-limit-theorem-math-for-quants` | Law of large numbers and central limit theorem | EXISTING, audit required
8. `bootstrap-cross-validation-math-for-quants` | Bootstrap and validation without leakage | EXISTING, audit required
9. `hypothesis-testing-pvalues-math-for-quants` | Hypothesis testing and p-values | EXISTING, audit required
10. `stationarity-autocorrelation-math-for-quants` | Stationarity, autocorrelation, and ergodicity | EXISTING, audit required
11. `arch-garch-volatility-math-for-quants` | Volatility clustering and conditional variance | EXISTING, audit required
12. `conditional-expectation-projection-math-for-quants` | Conditional expectation as the best forecast | EXISTING, audit required
13. `filtrations-no-lookahead-math-for-quants` | Filtrations and no look-ahead | EXISTING, audit required
14. `martingales-risk-neutral-measure-math-for-quants` | Martingales and risk-neutral measures | EXISTING, audit required
15. `stopping-times-optional-stopping-math-for-quants` | Stopping times and the optional stopping trap | NEW
16. `martingale-differences-trading-returns-math-for-quants` | Martingale differences and unpredictable returns | NEW
17. `self-financing-martingale-transforms-math-for-quants` | Self-financing strategies and martingale transforms | NEW
18. `concentration-inequalities-tail-risk-math-for-quants` | Concentration inequalities and tail risk | NEW
19. `monte-carlo-simulation-quant-trading-math-for-quants` | Monte Carlo simulation for pricing and risk | EXISTING candidate audit: search repository before drafting
20. `statistical-trading-system-capstone` | From signal to statistical trading system | NEW capstone

## WAVE 1: Foundations and uncertainty

Posts 1 through 5. Post 1 is already complete. Audit posts 2 through 5 for current house style, no em dashes, WebP-only embeds, sourcing, equations, and cross-links. Preserve the existing math-for-quants route.

## WAVE 2: Estimation and evidence

Posts 6 through 10. Audit or repair existing estimator, LLN, bootstrap, hypothesis-testing, and stationarity posts. Ensure LLN is the explicit bridge from per-trade edge to empirical convergence.

## WAVE 3: Information and martingales

Posts 11 through 15. Audit GARCH, conditional expectation, filtrations, and martingale posts, then write stopping times as the first new article. Cross-link the information progression carefully.

## WAVE 4: Martingale mechanics and capstone

Posts 16 through 20. Write the three new martingale mechanics articles, audit or add Monte Carlo, and finish with the end-to-end statistical trading system capstone.

## Progress

- [x] Wave 1, Post 1: probability spaces and random variables, audited and verified.
- [ ] Wave 1: posts 2 to 5 audited and verified.
- [ ] Wave 2: posts 6 to 10 audited and verified.
- [ ] Wave 3: posts 11 to 15 audited and verified.
- [ ] Wave 4: posts 16 to 20 audited and verified.
