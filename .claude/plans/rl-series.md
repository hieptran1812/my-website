# Reinforcement Learning: From Rewards to Real Systems

Series of **71 deep-dive posts** in **`content/blog/machine-learning/reinforcement-learning/`** (NEW folder, subcategory `Reinforcement Learning`).
**blog-writer** voice (principal-engineer, intuition→math→runnable code→tables→case studies), **English**,
≥ 11,000 words, 8 figures each, `.png` embeds + `optimize-blog-images` (→ webp+cover+manifest; never rewrite embeds to `.webp`).
Commit + push **each wave** (explicit paths, never `git add -A`).

## Angle (non-negotiable)
"Every RL algorithm is a different answer to: *which objective to optimize, and how to estimate the gradient.*"
Every post is **theory + algorithms + applications**: the *why* (math/stats/info-theory behind the method), the *how*
(real runnable code in PyTorch/SB3/Gymnasium/TRL), and the *proof* (a named benchmark result or before→after metric).
Three pillars: **Theory of Diversity** (Track L) is a first-class track alongside RLHF and Finance.

Kit: `.cache/blog-writer/_rl-series-kit.md`
Render: `.cache/blog-writer/_render-rl.sh`
Folder: `content/blog/machine-learning/reinforcement-learning/`
Completion marker: `.claude/plans/.rl-done`

---

## Track A — Foundations (Wave 1, 6 posts)
A1 `what-is-reinforcement-learning` — The reward hypothesis, the agent↔environment loop, why RL differs from SL/UL. **Series intro + unified-map figure.**
A2 `markov-decision-processes` — States, actions, transition dynamics, reward function, horizon, discount γ; MDP formalism and why Markov property matters.
A3 `value-functions-and-the-bellman-equation` — V(s), Q(s,a), advantage A(s,a); Bellman expectation and optimality equations; derivation from first principles.
A4 `policies-deterministic-vs-stochastic` — Policy definitions, parameterization (tabular/softmax/Gaussian); what makes a policy optimal; the policy improvement theorem.
A5 `exploration-vs-exploitation-the-core-tension` — Multi-armed bandits, ε-greedy, UCB, Thompson sampling; regret bounds; why this tension never goes away in full RL.
A6 `the-credit-assignment-problem` — Long-horizon credit, the temporal-credit gap, return decomposition, eligibility traces preview; why RL is hard beyond bandits.

## Track B — Classic Algorithms (Wave 2, 6 posts)
B1 `dynamic-programming-for-rl` — Policy evaluation, policy improvement, policy iteration, value iteration; the DP → RL bootstrap connection.
B2 `monte-carlo-methods-in-rl` — MC prediction, MC control (GLIE), first-visit vs every-visit; high variance but unbiased; when MC beats TD.
B3 `temporal-difference-learning-td0-and-sarsa` — TD(0), SARSA (on-policy); the TD target; why TD converges and MC doesn't need a model; cliff-walking example.
B4 `q-learning-off-policy-td-control` — Q-learning derivation, off-policy nature, convergence conditions, max-bias problem; comparison with SARSA.
B5 `n-step-returns-and-td-lambda` — n-step returns, the λ-return, forward/backward views, eligibility traces; the bias–variance spectrum from TD to MC.
B6 `tabular-rl-in-practice` — Implementation guide: Q-table, ε-schedule, replay tricks, GridWorld/FrozenLake/Taxi case studies; when tabular is enough.

## Track C — Function Approximation (Wave 3, 5 posts)
C1 `function-approximation-why-tables-dont-scale` — Curse of dimensionality, linear FA, SGD for RL, the semi-gradient issue; why supervised-learning intuitions break.
C2 `linear-function-approximation-and-convergence` — Tile coding, Fourier basis, RBF; convergence of semi-gradient TD with linear FA; the on-policy stability theorem.
C3 `neural-networks-as-value-approximators` — Nonlinear FA, stochastic gradient, the deadly triad (bootstrapping × off-policy × FA); why vanilla DQN diverges.
C4 `the-deadly-triad-stability-in-deep-rl` — Divergence examples, gradient TD, emphatic TD, target networks as a fix; the theoretical landscape.
C5 `experience-replay-and-offline-data` — Replay buffer mechanics, i.i.d. trick, prioritized ER (PER), the off-policy correction problem; bridge to offline RL.

## Track D — Deep RL: Value-Based (Wave 4, 6 posts)
D1 `deep-q-networks-dqn` — DQN architecture, target network, ε-greedy schedule, loss; the 2015 Atari breakthrough; what it solved (stability) and what it left (overestimation).
D2 `dqn-improvements-double-dueling-per` — Double DQN (max-bias fix), Dueling architecture (V+A decomposition), Prioritized ER (TD-error sampling); each improvement's gain.
D3 `rainbow-dqn-combining-six-improvements` — Rainbow: DDQN + PER + Dueling + Multi-step + Distributional + Noisy Nets; ablation results; when Rainbow is overkill.
D4 `distributional-rl-c51-qr-dqn-iqn` — Return distributions vs expected return; C51, QR-DQN, IQN; risk-sensitive policies; why distributional beats expectation on Atari.
D5 `offline-rl-learning-from-fixed-datasets` — The distributional shift / out-of-distribution action problem; behavior cloning baseline; CQL, IQL, TD3+BC; offline→online fine-tuning.
D6 `conservative-q-learning-cql` — CQL objective derivation; pessimism under uncertainty; comparison with IQL/TD3+BC; D4RL benchmark results; when to use offline RL.

## Track E — Policy Gradient Methods (Wave 5, 7 posts)
E1 `the-policy-gradient-theorem` — REINFORCE derivation (score function estimator); the log-prob trick; high variance analysis; baseline subtraction; convergence.
E2 `actor-critic-a2c-a3c` — Actor-Critic architecture, advantage estimation, A2C (synchronous) vs A3C (asynchronous); shared backbone; wall-clock vs sample efficiency.
E3 `trust-region-policy-optimization-trpo` — The monotonic improvement theorem, KL constraint, natural policy gradient, conjugate gradient solver; why TRPO is stable but slow.
E4 `proximal-policy-optimization-ppo` — Clipped surrogate objective; why PPO approximates TRPO; entropy bonus; GAE advantage estimation; SB3 PPO recipe; MuJoCo results.
E5 `soft-actor-critic-sac` — Maximum entropy RL framework; SAC actor/critic/temperature; automatic entropy tuning; why SAC dominates continuous-control; HalfCheetah benchmark.
E6 `deterministic-policy-gradient-ddpg-td3` — DPG theorem, DDPG, TD3 (twin critics + delayed update + target smoothing); continuous action spaces; when SAC vs TD3.
E7 `on-policy-vs-off-policy-a-practical-guide` — Sample efficiency vs stability trade-offs; the on/off policy spectrum; choosing PPO vs SAC vs TD3 vs DQN by environment type.

## Track F — Model-Based RL (Wave 6, 5 posts)
F1 `model-based-rl-learning-world-models` — The model-based vs model-free trade-off; Dyna-Q architecture; sample efficiency wins; compounding model error; when MBRL helps.
F2 `dyna-q-and-planning-with-a-model` — Dyna-Q algorithm; planning steps vs environment steps; the balance; background planning vs decision-time planning; MPC preview.
F3 `world-models-dreamer-planet` — PlaNet (latent planning), Dreamer v1/v2/v3 (imagination rollouts, GRU world model, actor-critic in latent space); visual control benchmarks.
F4 `muzero-mastering-games-without-rules` — MuZero: MCTS + learned model; value/policy/reward prediction; how it beats AlphaZero on Atari; planning without a simulator.
F5 `model-based-vs-model-free-when-to-use-which` — Sample efficiency vs asymptotic performance; model capacity; compounding error; a decision guide for practitioners.

## Track G — Multi-Agent RL (Wave 7, 5 posts)
G1 `multi-agent-rl-fundamentals` — The MARL problem: joint action spaces, non-stationarity, communication; cooperative / competitive / mixed; Dec-POMDP formalism.
G2 `nash-equilibria-and-game-theory-for-marl` — Nash equilibrium in MARL, best-response dynamics, fictitious play, minimax-Q; the game-theory↔MARL bridge.
G3 `maddpg-centralised-training-decentralised-execution` — MADDPG: centralised critic with joint observations; CTDE framework; cooperative navigation case study; emergent communication.
G4 `emergent-behaviour-and-multi-agent-games` — OpenAI Five, StarCraft II (AlphaStar), emergent tool use, hide-and-seek; what makes multi-agent qualitatively different.
G5 `marl-applications-auctions-traffic-robotics` — RL for auction mechanism design, traffic signal control, multi-robot coordination; practical deployment lessons.

## Track H — RLHF & LLM Alignment (Wave 8, 8 posts)
H1 `why-language-models-need-rlhf` — The reward misspecification problem in SFT; sycophancy, verbosity bias; RLHF as a solution; InstructGPT overview; the pipeline.
H2 `reward-modeling-from-human-preferences` — Bradley-Terry preference model; reward model training; data collection; calibration; reward model accuracy vs downstream RLHF quality.
H3 `ppo-for-llm-fine-tuning-the-instructgpt-recipe` — KL-penalized PPO objective; reference policy; per-token reward; the 4-model setup; implementation in TRL; instability gotchas.
H4 `dpo-direct-preference-optimization` — DPO derivation: reparameterising PPO's optimal policy; the closed-form reward; no RL loop; why DPO is simpler and often competitive.
H5 `grpo-group-relative-policy-optimization` — GRPO: group-level relative rewards, no critic, DeepSeek-R1 recipe; comparison with PPO/DPO; reasoning task results.
H6 `constitutional-ai-and-rlaif` — Anthropic's Constitutional AI; RLAIF (AI feedback vs human feedback); self-critique + revision; scalable oversight; critique-revision loop.
H7 `reward-hacking-and-goodharts-law` — Specification gaming, reward over-optimization (the KL–reward Pareto curve), length exploitation; mitigations: KL constraint, reward ensemble.
H8 `the-alignment-problem-through-rl` — Value alignment, RLHF limitations (preference data quality, distribution shift), debate/amplification/IDA; open problems in AI safety from an RL view.

## Track I — RL in Finance & Trading (Wave 9, 8 posts)
I1 `rl-for-portfolio-optimization` — MDP formulation for multi-asset allocation; state (prices/factors), action (weights), reward (risk-adjusted return); DQN/PPO portfolio agents vs Markowitz.
I2 `optimal-execution-with-rl` — Almgren-Chriss baseline; RL for TWAP/VWAP improvement; market-impact MDP; action space (order sizes/rates); slippage reward shaping.
I3 `market-making-with-rl` — The market-maker's MDP: bid-ask spread, inventory risk, adverse selection; DQN/SAC agents; spread vs inventory trade-off; Avellaneda-Stoikov comparison.
I4 `crypto-trading-agents` — High-frequency crypto RL: exchange APIs, tick data, latency, position sizing; reward shaping for drawdown control; backtesting pitfalls.
I5 `risk-aware-rl-cvar-and-constrained-mdps` — CVaR as a risk measure; constrained MDPs (CMDP); Lagrangian relaxation; CPO (Constrained Policy Optimization); Sharpe vs CVaR objectives.
I6 `multi-asset-allocation-with-rl` — Cross-asset (equities/FX/bonds/commodities) RL portfolio; factor features; correlation regime changes; walk-forward evaluation; DDPG/SAC results.
I7 `backtesting-rl-trading-agents` — The backtesting trap: look-ahead bias, overfitting to history; walk-forward validation; realistic transaction costs; regime robustness testing.
I8 `case-study-end-to-end-rl-trading-system` — Building and deploying a complete RL trading system: data pipeline → env design → training → evaluation → live paper-trading; lessons learned.

## Track J — Applications & Advanced (Wave 10, 5 posts)
J1 `rl-for-robotics-sim-to-real-transfer` — Sim-to-real gap; domain randomisation; DAGGER; Hindsight ER; MuJoCo → real arm transfer; what breaks and why.
J2 `game-playing-atari-to-alphago-and-beyond` — Atari DQN 2015; AlphaGo (policy + value networks + MCTS); AlphaGo Zero (self-play); MuZero; what each breakthrough required.
J3 `safe-rl-constrained-and-risk-sensitive` — Constrained MDPs, CPO, Lagrangian RL; shielding; risk-sensitive objectives (CVaR, entropic); safety benchmarks (Safety Gym).
J4 `hierarchical-rl-options-and-subgoals` — Options framework, subgoal discovery, feudal networks, HIRO; when hierarchy helps (long-horizon, sparse rewards); skill reuse.
J5 `the-future-of-rl-open-problems-and-frontiers` — Sample efficiency, sim-to-real, sparse rewards, multi-task, offline-to-online, foundation RL models, world models; what's unsolved.

## Track K — Playbook & Capstone (Wave 11, 4 posts)
K1 `debugging-rl-agents-common-failure-modes` — Reward shaping bugs, observation normalization, action scaling, entropy collapse, NaN in policy gradient; the RL debugging checklist.
K2 `hyperparameter-tuning-for-rl` — What matters (γ, learning rates, n_steps, batch size, entropy coefficient); sensitivity analysis; Optuna/Ray Tune for RL; common footguns.
K3 `rl-engineering-production-systems` — Distributed RL (IMPALA/APPO/RLlib), environment vectorization, async actors, throughput profiling; the gap between research and production.
K4 `the-reinforcement-learning-playbook` — **Capstone**: unified algorithm-selection framework; the decision tree (model-free/based × on/off-policy × discrete/continuous × sample budget); links all 70 siblings; the "complete RL engineer" checklist.

## Track L — Theory of Diversity in RL (Wave 12, 6 posts)
L1 `why-diversity-matters-in-rl` — The diversity thesis: why a single deterministic policy is fragile; behavioral diversity for exploration, robustness, multi-task; diversity as intrinsic motivation.
L2 `diayn-diversity-is-all-you-need` — DIAYN: skill discovery via discriminability; the information-theoretic objective I(s;z); unsupervised skill pre-training; downstream task fine-tuning.
L3 `quality-diversity-algorithms-map-elites` — MAP-Elites: behavior characterisation + archive + illumination; CMAME; QD vs RL vs evolutionary search; open-ended exploration.
L4 `population-based-training-and-evolutionary-rl` — PBT (exploit + explore on hyperparameters); CMA-ES; OpenAI Evolution Strategies; when evolution outperforms gradient descent.
L5 `ensemble-policies-and-diverse-policy-sets` — Policy ensembles for exploration (EDAC, SAC-N); disagreement-based exploration; diverse policy cover; multi-task policy sets.
L6 `intrinsic-motivation-curiosity-and-empowerment` — ICM (curiosity via prediction error), RND (random network distillation), empowerment (mutual information I(a;s')); comparison on sparse-reward envs; Montezuma's Revenge benchmark.

---

## Wave schedule
| Wave | Track | Posts | Status |
|------|-------|-------|--------|
| 1 | A: Foundations | 6 | TODO |
| 2 | B: Classic Algorithms | 6 | TODO |
| 3 | C: Function Approximation | 5 | TODO |
| 4 | D: Deep RL — Value-Based | 6 | TODO |
| 5 | E: Policy Gradients | 7 | TODO |
| 6 | F: Model-Based RL | 5 | TODO |
| 7 | G: Multi-Agent RL | 5 | TODO |
| 8 | H: RLHF & LLM Alignment | 8 | TODO |
| 9 | I: Finance & Trading | 8 | TODO |
| 10 | J: Applications & Advanced | 5 | TODO |
| 11 | K: Playbook & Capstone | 4 | TODO |
| 12 | L: Theory of Diversity | 6 | TODO |
