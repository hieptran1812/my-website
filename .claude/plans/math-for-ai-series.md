# Math for AI Research: From First Principles to the Theory Section

**Subcategory:** `mathematics` (folder already exists under ML, currently empty; renders "Mathematics")
**Path:** `content/blog/machine-learning/mathematics/<slug>.md`
**frontmatter:** `category: "machine-learning"` · `subcategory: "Mathematics"`
**Size:** **130 posts, 20 tracks (A to T, incl. R), 21 waves (0, 1–4, 4R, 5–19).** Rows 1–120 plus inserted 18a, 24a, 25a, 32a, 32b, 47a, 48a, 54a, 66a, 79a; numbering stays stable, suffixed rows run in their track's wave. Tracks O–T come from a 4-venue oral / spotlight scan (2026-10-23). **STATUS: APPROVED 2026-10-23 as a run-gradually roadmap (all 130 posts kept); Wave 0 next.** User overrides: **3,000–4,000 words**, must be easy to understand, highly visual, and show real-world applications.
**Skill:** `blog-writer`, depth `deep-dive`. **Language:** English (repo convention, verify gate enforces it). Request came in VN.

### The user's request (2026-10-23, verbatim intent)
A math series for AI and deep learning, basic to advanced (linear algebra, abstract algebra, geometry, calculus, statistics and probability, ...), so the reader can (1) **read** research papers on LLMs, diffusion, flow matching, optimal transport, mixture of experts, and (2) **write** papers for top conferences.

### Folder-name note (checked)
`src/lib/postPath.ts::derivePostLocation`: folder wins. `formatSubcategoryName` in `src/app/blog/machine-learning/page.tsx:38` title-cases `mathematics` to **"Mathematics"**. Nothing to register.

---

## Angle: every post ends inside a real paper

The repo already has the *models* (38 posts in `image-generation/`, MoE and attention posts in `large-language-model/`) and an 83-post `trading/math-for-quants` series. What it lacks is the bridge for an ML reader: **the exact math a 2024–2026 ML paper assumes, taught so the reader can follow the derivation and then produce one.**

The series is organised around one claim:

> **Paper math is a small, closed vocabulary.** About 70 ideas cover nearly every equation in a NeurIPS / ICML / ICLR main paper. Learn each one once, with a picture, a derivation and a numerical check, and the appendix stops being a wall.

Three fixed devices in every post, no exceptions:

1. **Picture first.** The idea drawn before any symbol: a vector field, a density morphing, a level-set, a transport plan. 2D toy first, then "the same thing in 4096 dimensions".
2. **Paper sentence, decoded.** A boxed block quoting **one real equation from one named paper** (arXiv id + equation number), then decoding it symbol by symbol with the post's tools. Use the paper bank below; **WebFetch the arXiv abstract page to confirm the id and the equation before citing.** Never paraphrase an equation into one the paper does not contain.
3. **Check it numerically.** A NumPy / PyTorch snippet of at most ~30 lines that verifies the post's central identity (finite-difference gradient check, Monte Carlo KL vs closed form, Sinkhorn vs LP plan, Euler ODE vs exact flow). **The agent runs it once with Bash and pastes the real output.** No invented output.

4. **Proof, read line by line** (user requirement, 2026-10-23: *the math must be enough to read and understand the proofs in papers*). From Track B on, every post contains one **complete** proof (≤ ~15 steps) of a real lemma or theorem, taken from a named paper's appendix or a standard textbook. Every step is annotated with the rule that licenses it (*by Cauchy–Schwarz*, *swap ∫ and ∇ by dominated convergence*, *integrate by parts, boundary term vanishes because p decays*), and a closing **"where each assumption was used"** check maps every hypothesis to the line that needs it, plus one counterexample showing what breaks without it. Each post also tags the **proof moves** it teaches; the kit keeps the running index (add-and-subtract, telescoping, Fubini / DCT swap, integration by parts, Grönwall, union bound, change of measure, coupling, ε/3, induction over layers or time steps), and #71 is the compendium.

5. **Where this shows up in practice** (user requirement, 2026-10-23). One section per post with 2–3 concrete, real applications of the post's math in a shipped system, a known training recipe or an industrial setting: SVD → LoRA adapters and model compression; KL → the RLHF penalty and distillation; conformal prediction → coverage guarantees in medical triage; Sinkhorn → minibatch OT couplings in image generators; differential privacy → DP-SGD in production telemetry. Each one is named and sourced (paper, engineering blog or docs, with a link). Never invent a company use case.

### Accessibility contract (user requirement: *dễ hiểu, trực quan*)

The reader may be an engineer who last did math in university. Every new concept follows the same ladder, in order:
1. **One plain-language sentence** saying what it is for, before any symbol.
2. **An analogy or physical picture** (mass being moved for OT, a fluid for the continuity equation, a hiker for gradient descent).
3. **A figure** in the series' visual language.
4. **A worked example with small numbers, by hand** (a 2×2 matrix, a 3-point distribution), so the reader can check each step with a pencil.
5. **Then the formal definition / theorem**, with every symbol defined at first use.
6. **Then the high-dimensional or general statement** and the paper connection.

Each post also ends with a **"What to remember"** box (3–5 bullets) and opens with a difficulty tag (**Foundation / Core / Advanced / Research**). Proofs stay complete but are introduced with the one-sentence idea of the proof ("we squeeze the error between two telescoping sums") before the steps.

Plus, at the top of every post: a **Needs / Unlocks** line. *Needs:* the 1–3 earlier posts it builds on. *Unlocks:* the paper families the reader can now read.

### Honesty rule (hard gate)
Every mathematical claim is one of: **(a) proved or derived** in the post, with the steps shown; **(b) cited** to a named paper / textbook with a link (and theorem number when there is one); **(c) numerically checked** in the snippet. Statements we cannot do (a) for get a precise citation and the phrase "we state without proof". **No hand-waved "it can be shown".** Watch the arithmetic: the verify gate never recomputes a number (see memory `verify-gates-never-check-arithmetic`).

---

## Gates

- **3,000–4,000 prose words** (user override 2026-10-23, same exception as the CVPR and interpretability series; run the gate with `WORD_MIN=3000 WORD_MAX=4000`, which fails both ends). Code fences and `blog-anim` blocks do not count, so the derivation goes in prose and display math, the check goes in code.
- **5–8 figures, target 6;** ≥ ½ pictorial (xy-chart, space, cells, filmstrip, worked-example, anatomy); box-flow ≤ ⅓, max 2; no kind more than twice. **≥ 1 animated figure** where motion carries meaning (ODE / SDE trajectories, gradient descent paths, OT plans, rotations). Ask for **exact-proportion** data in chart briefs: figure-reviewer does not check data fidelity.
- Verify: `WORD_MIN=3000 WORD_MAX=4000 bash .claude/skills/blog-writer/scripts/verify-post.sh <post.md> <slug> deep-dive` via one `post-verifier`.
- **No em dash** (no `—`, no spaced ` – `, no ` -- `).
- **LaTeX is this series' main risk** (every post is math-dense). Before verify, run the fence-aware `$` scan and fix:
  - inline math starting with a digit → brace-wrap `${2}^{n}$`;
  - `<` followed by a letter inside math → `\lt `;
  - **no** `\(...\)` / `\[...\]`, no `\$` next to math (site has no remark-math; `MathJax.tsx` regex is the gate);
  - display math `$$` on its own lines, never inside a list item;
  - odd `$` count in a file = desync, fix before anything else.
- Kit: `.cache/blog-writer/_math-for-ai-series-kit.md` (built in Wave 0, READ FULLY before drafting).
- Renderer lives **outside the repo**: `/Users/hieptran1812/Documents/mcp_excalidraw/scripts/render-scene-batch.mjs`. The sharpness gate passes with 0 webp on disk: **count files**.
- Check the post does not already exist before the figure pipeline (same-slug render clobbers committed webps).

## The recurring visual language (defined in Wave 0, reused everywhere)

1. **The ladder map.** The 12 tracks as rungs, current post lit, arrows to the paper families it unlocks. Opens Track A posts and every capstone-style post.
2. **The 2D plane.** One fixed pair of axes, unit circle and basis vectors drawn the same way every time. Every linear map, eigenvector, projection, rotation, gradient and vector field in the series lands on it.
3. **The density filmstrip.** `p_0 → p_t → p_1` as 3–5 panels of a 1D or 2D density. Used for change of variables, diffusion, Fokker–Planck, flow matching, OT.
4. **The loss landscape.** One fixed contour plot (an anisotropic bowl plus a saddle variant). GD, momentum, Adam, natural gradient, implicit bias all draw their paths on it.

---

## Boundary table

| Neighbour | It owns | We own |
| --- | --- | --- |
| `trading/math-for-quants/` (83 posts) | Itô, measure theory, OT, information geometry **for pricing and portfolios** | the same tools with ML examples and ML papers; cross-link, never re-derive their finance examples |
| `image-generation/` (`the-math-of-ddpm`, `score-based-models-and-the-sde-view`, `flow-matching-and-rectified-flow`, `normalizing-flows-and-change-of-variables`) | the models, training recipes, samplers, code | the math those posts assume: SDEs, Fokker–Planck, continuity equation, pushforwards, OT |
| `large-language-model/` (MoE, attention posts) | architectures and systems | routing as a mixture model, load-balance loss derivation, attention as kernel smoothing |
| `reinforcement-learning/` | the RL series | only the policy-gradient / KL-regularised math needed for RLHF / DPO / GRPO |
| `scaling-laws/` | empirical scaling results | power-law fitting and the Chinchilla Lagrangian |
| `paper-reading/` | single-paper walkthroughs | the reusable tool each walkthrough relies on |

Cross-link spine (every post links both): **intro** `how-to-read-the-math-in-an-ml-paper` (#1) and **capstone** `the-math-roadmap-for-ml-researchers` (#72). Plus 2–3 siblings and 1–2 neighbour posts from the table.

---

## Track A — The language of papers (Wave 1)

| # | Slug | Post | Paper sentence decoded |
| --- | --- | --- | --- |
| 1 | `how-to-read-the-math-in-an-ml-paper` | Notation decoder: `E_{x~p}`, `∇_θ`, `argmin`, `∝`, `:=`, `‖·‖`, subscripts vs superscripts; a 3-pass method for any equation | DDPM loss (2006.11239, Eq. 14) |
| 2 | `sets-functions-and-maps-the-grammar-of-ml-math` | Sets, domains, codomains, composition, inverses, `f: R^d → R^k`; a network as a composed map | ResNet residual map (1512.03385, Eq. 1) |
| 3 | `proof-techniques-for-ml-researchers` | Direct, contrapositive, contradiction, induction, construction; reading "w.l.o.g." and "it suffices to show" | Universal approximation statement (Cybenko 1989, cited) |
| 4 | `sums-products-einsum-and-log-sum-exp` | Σ / Π manipulation, swapping sums, Einstein notation, log-sum-exp and numerically stable softmax | Attention (1706.03762, Eq. 1) in einsum |
| 5 | `the-inequality-toolkit-every-appendix-uses` | Triangle, Cauchy–Schwarz, Jensen, Hölder, Young, AM–GM, Markov, Chebyshev, union bound | ELBO via Jensen (VAE 1312.6114, Eq. 1–3) |
| 6 | `limits-big-o-and-asymptotics-in-ml-papers` | Limits, O / Θ / Ω / o, `Õ`, "with high probability", rates `O(1/√T)` | SGD rate statement in Adam paper (1412.6980, Thm. 4.1) |

## Track B — Linear algebra I: the geometry of vectors and maps (Wave 2)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 7 | `vectors-dot-products-and-the-geometry-of-embeddings` | Vector spaces, span, basis, dot product as similarity, cosine, high-dim near-orthogonality | Word2vec analogy / CLIP contrastive logits (2103.00020) |
| 8 | `matrices-as-linear-maps-what-a-layer-does` | Matrix as a map, columns = images of basis vectors, composition = multiplication, affine layers | Linear layer + GELU MLP (1706.03762, Eq. 2) |
| 9 | `rank-null-space-and-low-rank-structure` | Rank, null space, rank–nullity, low-rank factorisation, why updates are low rank | LoRA (2106.09685, Eq. 3) |
| 10 | `norms-inner-products-and-distances` | ℓp, Frobenius, spectral, nuclear norms; dual norms; which norm a paper's bound uses | Spectral normalisation (1802.05957) |
| 11 | `orthogonality-projections-and-qr` | Orthonormal bases, projection matrices, Gram–Schmidt, QR, least squares as projection | Orthogonal init / orthogonalised updates (Muon, 2502.16982) |
| 12 | `determinants-volume-and-invertibility` | Determinant as signed volume, triangular Jacobians, log-det tricks | RealNVP coupling log-det (1605.08803, Eq. 6) |

## Track C — Linear algebra II: spectra and decompositions (Wave 3)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 13 | `eigenvalues-eigenvectors-and-dynamics` | Eigenpairs, diagonalisation, powers of a matrix, exploding / vanishing recurrences, power iteration | S4 / linear RNN state matrix (2111.00396) |
| 14 | `symmetric-matrices-psd-and-the-spectral-theorem` | Spectral theorem, PSD, quadratic forms, covariance and Hessian as PSD objects | Gaussian covariance in VAE posterior |
| 15 | `svd-the-decomposition-deep-learning-runs-on` | SVD geometry, Eckart–Young, pseudo-inverse, PCA as SVD, weight spectra | Eckart–Young in LoRA / SVD-init papers |
| 16 | `matrix-calculus-gradients-of-matrix-functions` | Differentials, `d tr`, `d log det`, `∂‖AX−B‖²/∂X`, layout conventions | Backprop through a linear layer by hand |
| 17 | `kronecker-products-and-structured-matrices` | Kronecker, vec trick, block / Toeplitz / circulant, FFT diagonalisation | K-FAC (1503.05671), Shampoo (1802.09568) |
| 18 | `spectral-norm-lipschitz-constants-and-stability` | Operator norm, Lipschitz constant of a network, bounds by products of spectral norms | WGAN-GP / spectral norm (1802.05957) |
| 18a | `matrix-functions-polar-decomposition-and-newton-schulz` | Matrix functions via eigen / SVD (sqrt, inverse root, sign), polar decomposition, Newton–Schulz iteration, why orthogonalised updates are steepest descent under the spectral norm | Muon (2502.16982); Shampoo inverse 4th root (1802.09568) |

## Track D — Calculus and analysis (Wave 4)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 19 | `derivatives-gradients-and-directional-change` | Partial derivatives, gradient as steepest ascent, directional derivative, level sets | Gradient of cross-entropy w.r.t. logits |
| 20 | `chain-rule-jacobians-and-backpropagation` | Multivariable chain rule, Jacobians, forward vs reverse mode, VJP / JVP cost | Backprop as reverse-mode AD (Baydin et al. 1502.05767) |
| 21 | `taylor-expansion-and-local-approximation` | 1st / 2nd order Taylor, Hessian, Newton step, Laplace approximation, error terms | Adam / sharpness papers' 2nd-order argument |
| 22 | `multivariable-integration-and-change-of-variables` | Fubini, polar / spherical coordinates, Jacobian determinant in integrals, Gaussian integral | Normalizing-flow density formula |
| 23 | `odes-for-deep-learning` | Existence / uniqueness (Lipschitz), Grönwall's inequality, Euler and RK, stiffness, error vs step count | Neural ODE (1806.07366, Eq. 1) |
| 24 | `function-spaces-and-operators-lite` | Metric / normed / Hilbert spaces, L², completeness, operators, universal approximation in context | Neural operators / RKHS teaser |
| 24a | `fourier-analysis-for-deep-learning` | Fourier series / DFT, convolution theorem, FFT, frequencies of RoPE, spectral bias, Fourier features, long convolutions | Fourier features (2006.10739); Hyena (2302.10866); YaRN frequency view (2309.00071) |

## Track R — Rigor for reading proofs (Wave 4R, between Waves 4 and 5)

*Added 2026-10-23 for the proof-reading requirement. The posts in Tracks A–D give intuition; Track R gives the precise definitions and theorems that appendix proofs invoke by name. Needs #3, #5, #19–#23.*

| # | Slug | Post | Proof read line by line |
| --- | --- | --- | --- |
| 85 | `quantifiers-epsilon-delta-and-reading-a-definition` | ∀ / ∃ order, negating statements, ε–δ limits and continuity, unpacking a definition into a checklist, building examples and counterexamples | Continuity of softmax; Lipschitz constant of softmax |
| 86 | `sequences-series-and-uniform-convergence` | Convergence, Cauchy sequences, series tests, pointwise vs uniform convergence, why uniform convergence lets you swap limits | Robbins–Monro step-size conditions (Σηₜ = ∞, Σηₜ² < ∞) used in SGD proofs |
| 87 | `compactness-completeness-and-fixed-points` | Compactness, Bolzano–Weierstrass, extreme value theorem (existence of minimisers), completeness, Banach fixed-point theorem | Picard–Lindelöf existence for Neural ODEs; Sinkhorn as a contraction (Hilbert metric, cite) |
| 88 | `interchanging-limits-integrals-and-derivatives` | Monotone / dominated convergence, Fubini–Tonelli, differentiating under the integral (Leibniz), when each swap is illegal | Flow Matching Thm. 1–2 proof (2210.02747, App. A): ∇ and E swapped |
| 89 | `vector-calculus-divergence-theorem-and-integration-by-parts` | Divergence, Laplacian, divergence theorem, integration by parts in ℝᵈ, vanishing boundary terms, Stein's identity | Hyvärinen score-matching theorem (JMLR 2005, Thm. 1); Tweedie's formula |
| 90 | `pdes-for-generative-models-transport-heat-and-characteristics` | Transport / continuity equation, heat equation = Gaussian smoothing, method of characteristics, weak solutions lite | Continuity equation ⇔ pushforward by the ODE flow (2210.02747; Villani, cite) |

## Track E — Probability (Wave 5)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 25 | `probability-spaces-random-variables-and-distributions` | Sample space, events, σ-algebra lite, pmf / pdf / cdf, measure only where needed | Data distribution `p_data` as used in GAN (1406.2661) |
| 25a | `measure-theoretic-probability-for-reading-proofs` | σ-algebras, measurable functions, "almost surely", modes of convergence (a.s., in probability, L², in distribution), Radon–Nikodym, conditional expectation as L² projection, product measures | Proofs that write `E[· | x_t]` and densities w.r.t. Lebesgue; DSM ⇔ score matching equivalence (Vincent 2011, App.) |
| 26 | `expectation-variance-and-the-tower-rule` | LOTUS, linearity, variance, covariance, law of total expectation, tower rule | Denoising objective as conditional expectation |
| 27 | `conditional-probability-bayes-and-latent-variables` | Conditioning, Bayes, marginalisation, graphical models, latent-variable models | VAE generative model (1312.6114) |
| 28 | `the-gaussian-everything-you-need` | Multivariate Gaussian, affine transforms, marginals, conditionals, product of Gaussians, KL closed form | DDPM forward marginal `q(x_t | x_0)` (2006.11239, Eq. 4) |
| 29 | `monte-carlo-importance-sampling-and-the-reparameterization-trick` | MC estimators, variance, importance weights, score-function (REINFORCE) vs pathwise gradients, Gumbel-softmax | VAE reparameterisation (1312.6114); Gumbel-softmax (1611.01144) |
| 30 | `pushforward-measures-and-change-of-variables-for-densities` | Pushforward `T#p`, density transform with Jacobian, why generative models are pushforwards | Flow Matching's probability path (2210.02747) |

## Track F — Statistics and information theory (Wave 6)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 31 | `maximum-likelihood-and-why-cross-entropy-is-the-loss` | Likelihood, log-likelihood, MLE = forward KL minimisation, next-token loss, perplexity | GPT next-token objective |
| 32 | `entropy-and-cross-entropy` | Entropy as expected surprise (bits vs nats), cross-entropy, why cross-entropy ≥ entropy, label smoothing, perplexity as exp(cross-entropy) | Label smoothing (Inception-v3, 1512.00567, Sec. 7) |
| 32a | `kullback-leibler-divergence-definition-properties-and-proofs` | **KL deep-dive I (user add, 2026-10-23).** KL = cross-entropy − entropy; Gibbs' inequality (KL ≥ 0, via Jensen); not symmetric, no triangle inequality (so not a metric); support / absolute-continuity condition and when KL = ∞; chain rule; invariance under reparameterisation; closed form between two multivariate Gaussians, derived step by step; KL's local quadratic form = Fisher (forward link to #60) | VAE KL term (1312.6114, App. B); DDPM per-step Gaussian KL (2006.11239, Eq. 5–7) |
| 32b | `kl-divergence-in-practice-forward-vs-reverse-and-estimators` | **KL deep-dive II.** Forward KL (mode-covering, = MLE) vs reverse KL (mode-seeking, = VI) on a bimodal target, drawn; which direction VAEs, diffusion, distillation, RLHF and DPO minimise and why; Monte Carlo estimators k1 / k2 / k3 (unbiasedness vs variance, numerically checked); the KL penalty in PPO-RLHF vs GRPO's in-loss KL; KL vs JS vs Wasserstein when supports don't overlap (forward link to #52) | Schulman's KL estimators note (joschu.net, 2020, cite as blog); InstructGPT KL-penalised reward (2203.02155, Eq. 2); knowledge distillation (1503.02531) |
| 33 | `the-elbo-and-variational-inference-from-scratch` | Evidence, ELBO derivation two ways, EM algorithm as coordinate ascent on the ELBO, amortised inference, the gap = KL | DDPM as hierarchical ELBO (2006.11239, Eq. 5) |
| 34 | `f-divergences-mutual-information-and-their-estimators` | f-divergences, variational (Fenchel) bounds, MI, InfoNCE as a lower bound | f-GAN (1606.00709); CPC / InfoNCE (1807.03748) |
| 35 | `bayesian-inference-priors-posteriors-and-regularization` | Priors as regularisers (L2 = Gaussian prior), MAP, posterior predictive, uncertainty | Weight decay as MAP |
| 36 | `concentration-and-high-dimensional-probability` | Hoeffding, sub-Gaussian, Azuma / martingale differences, Gaussian annulus, random projections (JL), curse and blessing of dimension | JL lemma in retrieval / attention approximations |

## Track G — Optimization (Wave 7)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 37 | `convex-sets-convex-functions-and-why-they-matter` | Convexity definitions, first / second-order tests, strong convexity, smoothness | Logistic regression convexity |
| 38 | `gradient-descent-convergence-rates-and-step-sizes` | L-smooth descent lemma, `O(1/T)` and linear rates, condition number, step-size choice | Descent lemma as used in optimiser papers |
| 39 | `sgd-noise-variance-and-batch-size` | Unbiased gradient estimate, variance, batch size vs learning rate, linear scaling rule, SDE view | Goyal et al. linear scaling (1706.02677) |
| 40 | `momentum-adam-and-adaptive-methods-the-math` | Heavy ball, Nesterov, preconditioning, Adam bias correction, decoupled weight decay | Adam (1412.6980, Alg. 1); AdamW (1711.05101) |
| 41 | `lagrangians-kkt-and-duality-in-ml` | Constrained optimisation, Lagrange multipliers, KKT, duality, max-entropy solutions | KL-constrained policy closed form (DPO 2305.18290, Eq. 4) |
| 42 | `nonconvex-landscapes-saddles-and-implicit-bias` | Critical points, saddles, escape by noise, implicit regularisation of GD, edge of stability | Implicit bias of GD (Soudry et al. 1710.10345) |

## Track H — Stochastic processes: the math of diffusion (Wave 8)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 43 | `markov-chains-stationary-distributions-and-mcmc` | Transition kernels, stationarity, detailed balance, Metropolis–Hastings, mixing | DDPM forward chain as a Markov chain |
| 44 | `brownian-motion-and-the-wiener-process` | Random walk limit, independent Gaussian increments, `√dt` scaling, quadratic variation | Score SDE forward process (2011.13456) |
| 45 | `stochastic-differential-equations-and-ito-calculus` | SDEs, Euler–Maruyama, Itô's lemma, VP / VE SDEs, OU process | Score SDE (2011.13456, Eq. 5) |
| 46 | `fokker-planck-and-the-continuity-equation` | How densities evolve under drift + noise; continuity equation as conservation of mass | Probability-flow derivation (2011.13456, App. D) |
| 47 | `score-functions-score-matching-and-langevin-dynamics` | Score `∇ log p`, Hyvärinen score matching, denoising score matching (Tweedie), Langevin sampling, classifier-free guidance as Bayes' rule on scores | NCSN (1907.05600); Vincent 2011 DSM |
| 47a | `log-concave-sampling-and-langevin-convergence` | Log-concavity, mixing time of (unadjusted / Metropolis-adjusted) Langevin, discretisation error vs ε, why diffusion sampling has guarantees; first-order rejection sampling | ICML 2026 Outstanding: *High-Accuracy Sampling for Diffusion Models and Log-Concave Distributions* (find arXiv id) |
| 48 | `reverse-time-sdes-and-the-probability-flow-ode` | Anderson's reverse SDE, probability-flow ODE, same marginals, DDIM as its discretisation | Score SDE (2011.13456, Eq. 6, 13); EDM (2206.00364) |
| 48a | `girsanov-and-kl-between-path-measures` | Path measures, Girsanov's theorem, KL between two SDEs = ½∫ E‖drift difference‖², data-processing inequality; how diffusion convergence bounds are proved | Chen et al., *Sampling is as easy as learning the score* (2209.11215, main theorem proof sketch) |

## Track I — Flows and optimal transport (Wave 9)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 49 | `continuous-normalizing-flows-and-the-trace-trick` | Instantaneous change of variables, `d log p/dt = −tr(∂f/∂x)`, Hutchinson estimator | FFJORD (1810.01367); Neural ODE (1806.07366, Thm. 1) |
| 50 | `flow-matching-from-the-continuity-equation` | Probability paths, marginal vs conditional vector field, why CFM's gradient equals FM's | Flow Matching (2210.02747, Thm. 1–2); stochastic interpolants (2303.08797) |
| 51 | `optimal-transport-monge-and-kantorovich` | Monge map, Kantorovich plan, LP formulation, Brenier's theorem in 2D pictures | OT primer (Peyré & Cuturi 1803.00567) |
| 52 | `wasserstein-distances-and-kantorovich-duality` | W1 / W2, Kantorovich–Rubinstein duality, Lipschitz critics, 1D closed form | WGAN (1701.07875, Eq. 3) |
| 53 | `entropic-ot-and-the-sinkhorn-algorithm` | Entropic regularisation, Sinkhorn iterations as matrix scaling, convergence, GPU cost | Sinkhorn distances (Cuturi 1306.0895) |
| 54 | `dynamic-ot-rectified-flow-and-schrodinger-bridges` | Benamou–Brenier, straight paths, minibatch OT couplings, rectified flow, Schrödinger bridge teaser | Rectified Flow (2209.03003); OT-CFM (2302.00482); DSB (2106.01357) |
| 54a | `consistency-models-and-mean-flows-the-math-of-few-step-generation` | Self-consistency along the PF-ODE, consistency loss, average velocity and the MeanFlow identity, shortcut models | Consistency Models (2303.01469); MeanFlow (2505.13447) |

## Track J — Geometry, symmetry and abstract algebra (Wave 10)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 55 | `groups-and-symmetry-for-deep-learning` | Group axioms, actions, orbits, invariance vs equivariance, CNNs as translation-equivariant maps | Geometric Deep Learning (2104.13478) |
| 56 | `group-representations-and-equivariant-networks` | Representations, irreps, Schur's lemma lite, permutation equivariance (sets, GNNs, attention), E(3) | Deep Sets (1703.06114); EGNN (2102.09844) |
| 57 | `rotations-lie-groups-and-rope` | SO(2), complex exponentials, Lie algebra / exp map, deriving RoPE's relative-position property, context extension as frequency rescaling (PI, NTK-aware, YaRN) | RoFormer (2104.09864, Eq. 13–16); YaRN (2309.00071) |
| 58 | `manifolds-tangent-spaces-and-the-manifold-hypothesis` | Charts, tangent spaces, embeddings, intrinsic dimension, why data lives near low-dim manifolds | Manifold hypothesis in diffusion theory papers |
| 59 | `riemannian-geometry-geodesics-and-hyperbolic-embeddings` | Metrics, geodesics, curvature lite, exp / log maps, Poincaré ball, Riemannian SGD | Poincaré embeddings (1705.08039) |
| 60 | `information-geometry-fisher-information-and-natural-gradient` | Fisher as a metric, KL's local quadratic form, natural gradient, K-FAC, trust regions | Natural gradient (Amari 1998); TRPO (1502.05477) |

## Track K — The math inside modern architectures (Wave 11)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 61 | `attention-as-kernel-smoothing-the-math-of-softmax-attention` | Softmax as soft argmax, `1/√d` derivation from variance, attention as Nadaraya–Watson, linear / random-feature attention, attention sinks and log-n length scaling, sigmoid output gating | 1706.03762; Performer (2009.14794); linear attention (2006.16236); NeurIPS 2025 Best Paper *Gated Attention for LLMs* (find id) |
| 62 | `signal-propagation-initialization-and-mup` | Variance through layers, Xavier / He derivation, residual + norm scaling, μP's width rules | He init (1502.01852); μP (2203.03466) |
| 63 | `mixture-of-experts-the-math-of-routing` | MoE as a mixture model, softmax / top-k gating, load-balance aux loss derived, capacity factor, router z-loss, aux-loss-free bias updates | Switch (2101.03961, Eq. 4–6); DeepSeekMoE (2401.06066); aux-loss-free (2408.15664) |
| 64 | `state-space-models-linear-recurrences-and-convolutions` | Continuous SSM, ZOH discretisation, recurrence = convolution, parallel scan, selectivity | S4 (2111.00396); Mamba (2312.00752, Alg. 2); Mamba-2 (2405.21060) |
| 65 | `power-laws-and-the-chinchilla-lagrangian` | Power laws on log-log axes, fitting honestly, compute-optimal allocation via Lagrange multipliers | Chinchilla (2203.15556, Eq. 2–4); Kaplan (2001.08361) |
| 66 | `policy-gradients-and-the-math-of-rlhf-dpo-grpo` | Policy-gradient theorem, baselines, KL-regularised objective, Bradley–Terry, DPO derivation, GRPO's group baseline, entropy collapse, pass@k vs pass@1 under RLVR | DPO (2305.18290, Eq. 4–7); PPO (1707.06347); GRPO (2402.03300); NeurIPS 2025 runner-up *Does RL Really Incentivize Reasoning Capacity…* (find id) |
| 66a | `reward-fine-tuning-of-diffusion-and-flow-models-the-math` | Split from #66 (8+ ICLR 2026 orals on preference / policy optimisation). KL-regularised reward maximisation over path measures, DDPO / DPO for diffusion, Adjoint Matching as stochastic optimal control, memoryless noise schedules | Adjoint Matching (2409.08861, confirm); DiffusionNFT (ICLR 2026 oral, find id) |

## Track L — Learning theory and writing the theory section (Wave 12)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 67 | `generalization-bounds-vc-rademacher-and-pac-bayes` | ERM, uniform convergence, VC / Rademacher, PAC-Bayes, why classical bounds fail for deep nets | Zhang et al. (1611.03530); Dziugaite & Roy (1703.11008) |
| 68 | `kernels-rkhs-and-the-neural-tangent-kernel` | Kernels, RKHS, representer theorem, infinite-width limit, NTK and lazy training | NTK (1806.07572) |
| 69 | `random-matrix-theory-for-deep-learning` | Wigner, Marchenko–Pastur, Stieltjes transform lite, weight-matrix spectra, heavy tails as a training diagnostic | Martin & Mahoney (1810.01075) |
| 70 | `anatomy-of-a-theory-section` | Setup → assumptions → theorem → proof sketch → appendix; stating assumptions reviewers accept; worked dissection of one real paper | Flow Matching (2210.02747) theory section dissected |
| 71 | `writing-proofs-that-survive-review` | The recurring lemma bag (add-and-subtract, telescoping, Grönwall, union bound, Young, change of measure), checking a proof numerically, common reviewer objections | DDIM (2010.02502) Thm. 1 proof walked |
| 72 | `the-math-roadmap-for-ml-researchers` | Capstone: the 130-post map, which tracks each paper family needs (LLM, diffusion, FM, OT, MoE, SSM, RL), study orders by goal, textbook list | the ladder map, fully lit |

**Paper bank:** the ids above are the pre-picked sources. Each agent WebFetches `arxiv.org/abs/<id>` once to confirm title and equation before citing. If an equation number is wrong, cite by section instead; never guess.

---

## Track M — The math of LLM practice: numerics, sampling, compression, evaluation (Wave 13)

*Added 2026-10-23 after a gap review against 2025–2026 LLM papers and the NeurIPS 2025 / ICLR 2026 / ICML 2026 award lists. Needs Tracks D–F.*

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 73 | `floating-point-arithmetic-and-numerical-stability` | IEEE floats, bf16 / fp8 formats, rounding error, overflow in softmax, loss scaling, accumulation precision, stochastic rounding | DeepSeek-V3 FP8 training (2412.19437, Sec. 3.3) |
| 74 | `quantization-math-rounding-error-hessians-and-rotations` | Uniform quantisers, MSE of rounding, outliers, Optimal Brain Surgeon → GPTQ (Hessian inverse via Cholesky), Hadamard rotations that spread outliers | GPTQ (2210.17323); QuaRot (2404.00456) |
| 75 | `sampling-from-language-models-temperature-top-p-and-gumbel-max` | Softmax temperature as an exponential family knob, top-k / top-p as truncated distributions, Gumbel-max trick, beam search vs sampling | Nucleus sampling (1904.09751) |
| 76 | `rejection-sampling-and-the-speculative-decoding-proof` | Rejection sampling, acceptance rate, the modified-rejection rule that keeps the target distribution exact, expected tokens per step | Leviathan et al. (2211.17192, Thm. 1); Chen et al. (2302.01318) |
| 77 | `compression-is-prediction-arithmetic-coding-and-mdl` | Source coding theorem, arithmetic coding, bits per byte, MDL, Kolmogorov complexity, measuring memorisation in bits per parameter | Language Modeling Is Compression (2309.10668); ICML 2026 HM *How much can language models memorize?* (find id) |
| 78 | `statistics-of-evaluation-error-bars-pass-at-k-and-significance` | Standard errors for benchmark accuracy, clustered / paired tests, the unbiased pass@k estimator, multiple comparisons, how many seeds | Adding Error Bars to Evals (2411.00640); Codex pass@k (2107.03374, Eq. 1) |

## Track N — Frontier theory: what 2025–2026 award papers assume (Wave 14)

*Each post is anchored on a recent award paper or the line it belongs to. Needs Tracks C, E, H, L.*

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 79 | `continuous-time-markov-chains-and-discrete-diffusion` | Rate matrices, Kolmogorov forward equation for discrete states, masking / absorbing processes, the masked-diffusion ELBO as weighted cross-entropy, concrete score | MDLM (2406.07524); SEDD (2310.16834); LLaDA (2502.09992); ICML 2026 Outstanding *The Flexibility Trap* |
| 79a | `discrete-diffusion-theory-ii-order-planning-and-reasoning` | Second discrete-diffusion post (4+ ICML 2026 and 3+ ICLR 2026 orals): unmasking order as a policy, any-order vs left-to-right likelihood, planners, why fixed order helps RL | ICML 2026 Outstanding *The Flexibility Trap*; *On the Reasoning Abilities of Masked Diffusion Language Models* (ICLR 2026 oral, find id) |
| 80 | `linear-attention-delta-rule-and-test-time-regression` | Attention as fast-weight memory, linear attention as a recurrence, delta rule as online least squares, gating, Householder products, chunked parallel form | DeltaNet (2406.06484); Gated DeltaNet (2412.06464); TTT layers (2407.04620) |
| 81 | `sparse-coding-superposition-and-compressed-sensing` | Sparse dictionary learning, near-orthogonal packing of features (JL), compressed sensing and RIP, superposition and its link to scaling laws, sparse autoencoders | Toy Models of Superposition (2209.10652); NeurIPS 2025 runner-up *Superposition Yields Robust Neural Scaling* (find id) |
| 82 | `high-dimensional-asymptotics-random-features-and-double-descent` | The proportional limit, ridge regression and random-features risk curves, deterministic equivalents, double descent, why generalisation is decided by spectra | NeurIPS 2025 Best Paper *Why Diffusion Models Don't Memorize*; ICML 2026 HM *A Random Matrix Perspective on the Consistency of Diffusion Models*, *Provable Grokking in Ridge Regression* (find ids) |
| 83 | `online-learning-regret-and-multiplicative-weights` | Regret, Hedge / multiplicative weights, online gradient descent, mirror descent, the regret view behind AdaGrad / Adam, mistake bounds | AdaGrad (Duchi et al. 2011, JMLR); NeurIPS 2025 runner-up *Optimal Mistake Bounds for Transductive Online Learning* (find id) |
| 84 | `expressivity-of-transformers-formal-languages-and-circuits` | Formal languages and automata, circuit classes (TC0), what fixed-depth transformers can and cannot compute, chain-of-thought as extra depth, succinctness | Merrill & Sabharwal (2207.00729); ICLR 2026 Outstanding *Transformers are Inherently Succinct* (find id) |

**Award-paper ids marked "find id" must be looked up and confirmed on arXiv / OpenReview by the drafting agent; titles above come from the official award posts (blog.neurips.cc, blog.icml.cc, ICLR / MPI-SWS announcement).**

---

## Conference scan (2026-10-23): what orals and spotlights use that Tracks A–N did not cover

Four parallel scans: **NeurIPS 2025** (40 orals + 357 spotlights, ~65 theory abstracts read), **ICLR 2026** (~215 of 223 orals; no spotlight tier), **ICML 2026** (~168 orals; 45 theory papers), **AAAI 2026/25** (9 outstanding papers, 3,072 titles keyword-scanned, ~35 abstracts). A topic was promoted to a new post when it showed up at **two or more venues** or was a whole standard subfield; the rest became scope extensions. Raw scrape data was saved in the session scratchpad (not in the repo). Tracks O–T follow. Every example id below came back from the scans; ids marked "find id" must be confirmed before citing.

**Cross-venue frequency (venues where it appeared):** causal inference 4/4 · differential privacy 4/4 · game theory / equilibria 4/4 · conformal prediction 4/4 · MDP theory 3/4 · minimax lower bounds 3/4 · bandits 3/4 · rate–distortion / IB 2/4 · approximation theory 2/4 · implicit differentiation 2/4 · Shapley 2/4 · spectral graph / sheaves 2/4 · Hermite / single-index / SQ 2/4.

## Track O — Statistical theory and guarantees (Wave 15)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 91 | `m-estimation-consistency-and-misspecification` | M-estimators, consistency, asymptotic normality, sandwich covariance, what happens when the model is wrong | *Why DPO is a Misspecified Estimator and How to Fix It* (2510.20413) |
| 92 | `minimax-lower-bounds-le-cam-fano-and-assouad` | Minimax risk, reduction to testing, Le Cam two-point, Fano, Assouad, change-of-measure lower bounds for bandits | *Minimax Optimal Strategy for Delayed Observations in Online RL* (2603.03480); *Non-Asymptotic Analysis of (Sticky) Track-and-Stop* (2505.22475) |
| 93 | `hermite-polynomials-single-index-models-and-sq-lower-bounds` | Hermite expansions, information exponent, single / multi-index models, statistical query and low-degree lower bounds, statistical–computational gaps | *An Optimized Franz–Parisi Criterion and its Equivalence with SQ Lower Bounds* (NeurIPS 2025 oral, OR:U8BwT6Rmw4); *Joint Learning in the Gaussian Single Index Model* (2505.21336) |
| 94 | `algorithmic-stability-and-information-theoretic-generalization` | Uniform stability, MI / CMI generalisation bounds, why noisy SGD / Langevin generalise | *Tighter CMI-Based Generalization Bounds…* (NeurIPS 2025 oral, OR:VYLdKb5dzO); *Temperature is All You Need for Generalization in Langevin Dynamics…* (2505.19087) |
| 95 | `conformal-prediction-and-exchangeability` | Exchangeability, de Finetti lite, split conformal and its finite-sample coverage proof (order statistics), conditional coverage limits | *Optimal Decision-Making Based on Prediction Sets* (2602.00989); AAAI-26 cost-sensitive conformal training (ojs 39384) |
| 96 | `proper-scoring-rules-calibration-and-fenchel-young-losses` | Proper scoring rules, calibration error, swap regret ⇒ calibration, Fenchel–Young losses, surrogate regret bounds | *High-Dimensional Calibration from Swap Regret* (NeurIPS 2025 oral, OR:UVDihUz0iT); OR:A4Xx9irvpp |

## Track P — Decisions, games and RL theory (Wave 16)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 97 | `mdps-bellman-operators-and-value-iteration` | MDPs, Bellman operators as γ-contractions (uses #87), value / policy iteration, robust MDPs, POMDP belief states | *State Entropy Regularization for Robust RL* (NeurIPS 2025 oral, 2506.07085); AAAI-25 Outstanding *Revelations* (ojs 34845) |
| 98 | `stochastic-bandits-ucb-thompson-and-regret-lower-bounds` | Explore–exploit, UCB via concentration, Thompson sampling, Lai–Robbins lower bound, GP-UCB | *Improved Regret Bounds for GP-UCB* (2506.01393) |
| 99 | `nash-equilibria-minimax-and-fixed-point-existence` | Normal / extensive-form games, von Neumann minimax, Nash existence via Brouwer / Kakutani, saddle points, PPAD | *The Complexity of Symmetric Equilibria in Min-Max Optimization…* (OR:75LMvs1CjG); *Multiplayer Nash Preference Optimization* (2509.23102) |
| 100 | `no-regret-dynamics-and-learning-in-games` | No-regret ⇒ coarse correlated equilibrium, swap / Φ-regret, optimistic gradient, CFR, bilinear saddle-point dynamics | *Asymmetric Perturbation in Solving Bilinear Saddle-Point Optimization* (2506.05747); OR:rSsc9uCVBl |
| 101 | `stochastic-optimal-control-hjb-and-adjoint-methods` | Control problems, HJB, adjoint method, path-space KL = control cost, Schrödinger bridge as control, IPF | *Adjoint Schrödinger Bridge Sampler* (NeurIPS 2025 oral, 2506.22565); 2508.12511 |
| 102 | `distributionally-robust-optimization-and-risk-measures` | Ambiguity sets (KL, Wasserstein), DRO duality, CVaR and coherent risk measures, stochastic dominance | *Distributional Inverse RL* (2510.03013) |

## Track Q — Causality, privacy and trustworthy-ML math (Wave 17)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 103 | `structural-causal-models-do-calculus-and-identifiability` | SCMs, d-separation, do-operator, back-door / front-door, identifiability, counterfactuals | *Distributional Equivalence in Linear Non-Gaussian Latent-Variable Cyclic Causal Models* (2603.04780) |
| 104 | `causal-discovery-and-doubly-robust-estimation` | Markov equivalence, score vs constraint-based discovery, NOTEARS-style continuous relaxations, potential outcomes, doubly robust / semiparametric estimators | AAAI-26 Outstanding *Causal Structure Learning for Dynamical Systems…* (2512.14361); OR:htN3xhUpjD |
| 105 | `differential-privacy-definitions-mechanisms-and-composition` | (ε, δ)-DP, Laplace / Gaussian mechanisms, sensitivity, post-processing, composition, privacy as hypothesis testing | *Gaussian certified unlearning in high dimensions* (2510.13094) |
| 106 | `privacy-accounting-renyi-f-dp-and-dp-sgd` | Rényi / f-DP, amplification by subsampling, moments accountant, DP-SGD analysis, privacy–utility rates | *Privacy amplification by random allocation* (OR:MiPIjE5onj); OR:F0JzotXYgC |
| 107 | `cooperative-games-and-shapley-values` | Characteristic functions, core, Shapley axioms and uniqueness proof, Möbius / interaction indices, sampling approximations, SHAP | AAAI-26 *Shapley Value Approximation Based on k-Additive Games* (ojs 38770); *Exact Functional ANOVA…* (2603.02673) |
| 108 | `social-choice-mechanism-design-and-fair-division` | Arrow / Gibbard–Satterthwaite, incentive compatibility, distortion, EF1 / EFX / MMS existence arguments | AAAI-25 Outstanding *Every Bit Helps* (ojs 33507); *Epistemic EFX Allocations Exist…* (ojs 33476) |

## Track S — Discrete structures, complexity and algorithms (Wave 18)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 109 | `computational-complexity-and-reductions-for-ml-researchers` | P / NP / coNP / #P / PSPACE / PPAD, polynomial reductions, hardness proofs, parameterised complexity, approximation ratios and submodular greedy (1 − 1/e) | AAAI-26 *Matrix Editing Meets Fair Clustering* (10.1609/aaai.v40i23.38984) |
| 110 | `logic-sat-and-model-counting` | Propositional / first-order semantics, decidability, SAT, model counting, knowledge compilation (d-DNNF), probabilistic circuits | AAAI-26 Outstanding *Model Change for Description Logic Concepts* (10.1609/aaai.v40i23.39008); *Proof Systems That Tightly Characterise Model Counting Algorithms* (10.1609/aaai.v40i17.38431) |
| 111 | `spectral-graph-theory-laplacians-and-oversmoothing` | Graph / hypergraph / sheaf Laplacians, Dirichlet energy, Cheeger, oversmoothing as heat diffusion, graph Fourier / wavelets | AAAI-26 Outstanding *High-Pass Matters…* (10.1609/aaai.v40i27.39469); *Foundations of Equivariant Deep Learning: Graph and Sheaf NNs* (2607.03798) |
| 112 | `tensor-decompositions-cp-tucker-and-identifiability` | CP / Tucker, tensor rank, Kruskal uniqueness, ALS, identifiability via tensors | *Identifiability of Deep Polynomial Neural Networks* (NeurIPS 2025 oral, 2506.17093); OR:9FDErIfoVE |
| 113 | `approximation-theory-chebyshev-and-minimax-polynomials` | Weierstrass, Chebyshev polynomials, equioscillation, minimax polynomial approximation, spectral collocation | *The Polar Express: Optimal Matrix Sign Methods… Muon* (2505.16932); *Chebyshev Policies and the Mountain Car Problem* (2605.22305) |
| 114 | `implicit-differentiation-bilevel-optimization-and-operator-splitting` | Implicit function theorem, differentiating through argmins / fixed points (DEQs), bilevel optimisation, projections, ADMM / Douglas–Rachford | *Pinet: hard-constrained NNs with orthogonal projection layers* (2508.10480); 2601.22970 |

## Track T — Advanced dynamics and generative theory (Wave 19, with capstone)

| # | Slug | Post | Paper |
| --- | --- | --- | --- |
| 115 | `mean-field-limits-and-wasserstein-gradient-flows` | Interacting particles, propagation of chaos, mean-field two-layer nets, Wasserstein gradient flows (JKO), mean-field transformers | *A multiscale analysis of mean-field transformers…* (NeurIPS 2025 oral, OR:WCRPgBpbcA) |
| 116 | `solvable-training-dynamics-gradient-flow-in-toy-models` | Exact gradient flow for linear / diagonal nets, two-timescale arguments, Gaussian equivalence, grokking and phase transitions | ICML 2026 HM *To Grok Grokking* (find id); *Fast Escape, Slow Convergence…* (2511.18661) |
| 117 | `sequential-monte-carlo-and-inference-time-steering` | Importance resampling, Feynman–Kac models, twisted SMC, annealing; reward-guided sampling from diffusion models | *Progressive Inference-Time Annealing of Diffusion Models…* (OR:vf2GHcxzMV); *Ψ-Sampler* (OR:slVqJAI5sT) |
| 118 | `rate-distortion-and-the-information-bottleneck` | Rate–distortion function, Blahut–Arimoto, rate–distortion–perception, information bottleneck | *Optimal Neural Compressors for the Rate-Distortion-Perception Tradeoff* (2503.17558); *Information Shapes Koopman Representation* (2510.13025) |
| 119 | `large-deviations-and-rare-event-estimation` | Cramér / Chernoff, rate functions, Sanov, importance sampling for rare events, tail probabilities of LLM outputs | *Rare Event Analysis of Large Language Models* (2602.06791) |
| 120 | `point-processes-poisson-and-hawkes` | Poisson processes, intensities, thinning, Hawkes self-excitation, likelihoods, branching representation | *A Representer Theorem for Hawkes Processes…* (2510.08916); 2508.11727 |

## Scope extensions from the conference scan (drafting agents: grep your `#N` here too)

- **#17** add tensor-network notation pointer to #112. **#20** implicit differentiation pointer to #114. **#23** exponential / reversible RK integrators. **#24** Koopman operators. **#24a** Fourier on discrete domains, functional ANOVA, graph Fourier pointer to #111. **#25a** exchangeability, de Finetti. **#29** rare-event importance sampling pointer to #119. **#32** von Neumann entropy aside. **#36** Bernstein, sub-exponential. **#38** quasi-Newton / BFGS, self-concordance. **#39** SVRG variance reduction, Robbins–Monro ODE method, two-timescale stochastic approximation. **#41** LP duality and complementary slackness, SDP and sum-of-squares relaxations. **#42** non-Euclidean steepest descent and edge of stability. **#43** SMC pointer to #117. **#54** OT on manifolds, constrained / conditional OT. **#55 / #58** quotient spaces under group actions. **#59** carré du champ / diffusion geometry. **#62** spectral-sphere / Muon-style constrained training. **#67** graph-learning example; information-theoretic bounds pointer to #94. **#68** Gaussian processes and Bayesian optimisation. **#69** spiked models and the BBP transition, free probability. **#78** conformal pointer to #95. **#82** Hermite pointer to #93, SGD summary-statistic ODEs. **#83** link to #98 / #100. **#84** communication-complexity and memory lower bounds, hyperplane arrangements / linear regions of ReLU nets. **#87** Brouwer and Kakutani fixed-point theorems.

---

## Waves (one fresh session per wave, per CLAUDE.md)

| Wave | Track | Posts | Theme |
| --- | --- | --- | --- |
| 0 | infra | 0 | kit (voice, Needs / Unlocks block, paper-sentence block, check-it block, LaTeX lint), `_render-math.sh`, the 4 recurring figure primitives |
| 1 | A | 1–6 | the language of papers |
| 2 | B | 7–12 | linear algebra I |
| 3 | C | 13–18 + 18a | linear algebra II (7) |
| 4 | D | 19–24 + 24a | calculus, analysis, Fourier (7) |
| 4R | R | 85–90 | rigor for reading proofs: quantifiers, convergence, compactness, limit swaps, integration by parts, PDEs |
| 5 | E | 25–30 + 25a | probability, incl. measure-theoretic (7) |
| 6 | F | 31–36 + 32a, 32b | statistics and information theory, incl. the KL deep-dive (8 agents) |
| 7 | G | 37–42 | optimization |
| 8 | H | 43–48 + 47a, 48a | stochastic processes / diffusion math, incl. Girsanov (8) |
| 9 | I | 49–54 + 54a | flows, optimal transport, few-step generation (7) |
| 10 | J | 55–60 | geometry, symmetry, abstract algebra |
| 11 | K | 61–66 + 66a | architecture math (attention, init, MoE, SSM, scaling, RLHF) |
| 12 | L | 67–71 | learning theory and research craft (capstone #72 moves to Wave 19) |
| 13 | M | 73–78 | LLM practice: numerics, quantisation, sampling, compression, eval statistics |
| 14 | N | 79–84 + 79a | frontier theory (7) |
| 15 | O | 91–96 | statistical theory and guarantees |
| 16 | P | 97–102 | decisions, games, RL theory, control |
| 17 | Q | 103–108 | causality, privacy, Shapley, social choice |
| 18 | S | 109–114 | complexity, logic, graphs, tensors, approximation, implicit differentiation |
| 19 | T | 115–120 + 72 | advanced dynamics and generative theory + capstone (7) |

Waves are ordered by prerequisite; do not reorder without checking each post's *Needs*. The user's named topics (diffusion, FM, OT, MoE) live in Waves 8, 9 and 11. They need Tracks D–F (calculus, probability, information theory) but not G, J or L, so a "fast path" order 0, 1, 4, 4R, 5, 6, 8, 9, 11, 13, then the rest is valid if the user wants those posts sooner.

### Progress checklist

- [ ] Wave 0: kit, render helper, 4 figure primitives
- [ ] Wave 1: Track A, posts 1–6
- [ ] Wave 2: Track B, posts 7–12
- [ ] Wave 3: Track C, posts 13–18 + 18a
- [ ] Wave 4: Track D, posts 19–24 + 24a
- [ ] Wave 4R: Track R, posts 85–90
- [ ] Wave 5: Track E, posts 25–30 + 25a
- [ ] Wave 6: Track F, posts 31–36 + 32a, 32b (KL deep-dive)
- [ ] Wave 7: Track G, posts 37–42
- [ ] Wave 8: Track H, posts 43–48 + 47a, 48a
- [ ] Wave 9: Track I, posts 49–54 + 54a
- [ ] Wave 10: Track J, posts 55–60
- [ ] Wave 11: Track K, posts 61–66 + 66a
- [ ] Wave 12: Track L, posts 67–71
- [ ] Wave 13: Track M, posts 73–78
- [ ] Wave 14: Track N, posts 79–84 + 79a
- [ ] Wave 15: Track O, posts 91–96
- [ ] Wave 16: Track P, posts 97–102
- [ ] Wave 17: Track Q, posts 103–108
- [ ] Wave 18: Track S, posts 109–114
- [ ] Wave 19: Track T, posts 115–120 + capstone 72

Per-wave loop: write a shared brief to scratch, dispatch 6 drafting agents in one message (one short paragraph each pointing at the brief + their row), each runs its own `figure-author`; when drafts hit disk, **split the session**; gate each post through one `post-verifier` plus the LaTeX scan; commit only that wave's `.md` + `<slug>-N.webp` with explicit paths; `git pull --rebase --autostash`; push; close the session.

## Drafting-agent brief (paste into every dispatch)

1. Read `.cache/blog-writer/_math-for-ai-series-kit.md` in full first.
2. Read only your own row and your Track's header in this plan (grep / offset), never the whole file. Also grep `#<your number>` in §Scope extensions.
3. Depth `deep-dive`, **3,000–4,000 prose words** (gate with `WORD_MIN=3000 WORD_MAX=4000`), 6 figures (5–8), ≥ ½ pictorial, 1 animated figure where motion carries meaning.
3b. Follow the **Accessibility contract** ladder for every concept (plain sentence → analogy → figure → small-number worked example → formal → general), end with "What to remember", open with the difficulty tag.
3c. Include **Where this shows up in practice**: 2–3 named, sourced real applications.
4. Open with the **Needs / Unlocks** line, then the picture, then the symbols. 2D toy first, then the high-dimensional statement.
5. Include the **Paper sentence, decoded** box: one real equation from your row's paper, arXiv id confirmed by WebFetch.
6. Include the **Check it numerically** snippet (≤ ~30 lines), run it once, paste the real output.
7. Every claim derived, cited (with theorem / equation number) or numerically checked. No "it can be shown".
7b. Include the **Proof, read line by line** block: one complete real proof, every step annotated with its justification, the "where each assumption was used" check, one counterexample, and the proof-move tags.
8. No em dash. LaTeX rules from §Gates: brace-wrap digit-leading inline math, `\lt ` before letters, no `\(`/`\[`, `$$` on its own lines and never in a list.
9. Link #1 and #72, 2–3 siblings, 1–2 neighbours from the boundary table (`math-for-quants`, `image-generation`, `large-language-model`).
10. Delegate the whole figure pipeline to `figure-author`; never read a rendered webp yourself. Summarise children's findings; never pass raw returns up.
