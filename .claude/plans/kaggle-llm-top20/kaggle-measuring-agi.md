# Measuring Progress Toward AGI: Cognitive Abilities (2026): all prize winners

- **Slug:** `kaggle-measuring-agi`
- **URL:** https://www.kaggle.com/competitions/kaggle-measuring-agi
- **Dates:** 2026-03-17 to 2026-04-16 (deadline 23:59 UTC). Host: Google DeepMind. Winners: https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/724918
- **Entrants:** 1,062 teams / 1,063 submissions / 1,224 competitors (competition metadata API).
- **Prize pool:** $200,000 = 4 Grand Prizes x $25k + 5 tracks x 2 winners x $10k.
- **Rubric (after mid-run change, [684184](https://www.kaggle.com/competitions/kaggle-measuring-agi/discussion/684184)):** dataset quality and task construction 50%, writeup 20%, novelty/insights/discriminatory power 30% (community upvotes removed).
- **Deliverable:** a Kaggle Benchmark (Kaggle Benchmarks SDK tasks) attached to a writeup of 3 pages or less, targeting one of five faculties: learning, metacognition, attention, executive functions, social cognition.
- **Bank coverage:** `kaggle-llm-writeup-bank-part4.md` H2 "## 11." covers MEDLEY-BENCH and LearningBench in depth, GAUGE/Metaproteus/EphLangBench in one line each. Below, those two get short sections with only new facts.

## Award-data discrepancy (report both)

- **Host thread 724918** lists Metacognition winner #2 as **"Metacognitive Calibration Benchmark: Do Models Know What They Don't Know?" by Qing Zhou** ([writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/metacognitive-calibration-benchmark-do-models-kno), team "zhou"), whose writeup object carries **no award** in the API.
- **Writeup API** attaches Metacognition prize id 911 "Winner (2 of 2)" to **MetaBound (Ahmed A.)** ([writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1774955644602)), which the host thread does **not** mention.
- The 4 grand-prize entries show `awards: []` in the API (grand prizes are not track prizes); the host thread is the source for them.
- Both Metacognition candidates are covered below; treat the host thread as authoritative for the public announcement and flag the conflict in any post.

## Winners table (host order, 724918)

| # | Category (prize, USD) | Team | Project | Writeup | Code / benchmark links |
|---|---|---|---|---|---|
| 1 | Grand Prize ($25k) | Farhad Abtahi, Abdolamir Karbalaie, Eduardo Illueca-Fernandez, Fernando Seoane (team MEDLEY-BENCH) | MEDLEY-BENCH: Behavioral Metacognition Under Social Pressure | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1773873293562) | see section |
| 2 | Grand Prize ($25k) | Karandeep Singh | LearningBench: Measuring Inference-Time Learning in LLMs | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/learningbench) | see section |
| 3 | Grand Prize ($25k) | Arjun Thilak, Ramkumar Muniandi Vallimayil | GAUGE | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/gauge-grading-awareness-of-uncertainty-and-ground) | see section |
| 4 | Grand Prize ($25k) | Joseph Boskovski | Metaproteus | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1776217035880) | see section |
| 5 | Executive Functions, Winner 1 of 2 ($10k) | Simon Zhang, Bill Cai | Turn Bench | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1775940834879) | see section |
| 6 | Executive Functions, Winner 2 of 2 ($10k) | Muhammad Haaris | SecureExec-Bench | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1776339266224) | see section |
| 7 | Learning, Winner 1 of 2 ($10k) | Raphael B | GrammarGym | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1775210475900) | see section |
| 8 | Learning, Winner 2 of 2 ($10k) | Bruce C. (team EphLang) | EphLangBench | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/ephlang-bench-testing-llm-algorithmic-understandi) | see section |
| 9 | Metacognition, Winner 1 of 2 ($10k) | Binwen Liu, Yilin Ren | ESFP Benchmark | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1776173871950) | see section |
| 10 | Metacognition, Winner 2 of 2 ($10k), per host thread | Qing Zhou | Metacognitive Calibration Benchmark | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/metacognitive-calibration-benchmark-do-models-kno) | see section |
| 10' | Metacognition, Winner 2 of 2 ($10k), per writeup API only | Ahmed A. | MetaBound | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1774955644602) | see section |
| 11 | Social Cognition, Winner 1 of 2 ($10k) | Patricia Ji | HedgeDecode | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/hedgedecode) | [benchmark](https://www.kaggle.com/benchmarks/pyji08/hedgedecode), [GitHub](https://github.com/patriciayji27/HedgeDecode) |
| 12 | Social Cognition, Winner 2 of 2 ($10k) | Ana Pirtac | AdvisorBench | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/advisorbench-do-ai-models-widen-the-advisory-divi) | see section |
| 13 | Attention, Winner 1 of 2 ($10k) | Shankha Suvro Mullick | RIAC: Repetition-Induced Attention Collapse | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1775848234859) | see section |
| 14 | Attention, Winner 2 of 2 ($10k) | Patrycja Węgrzynowicz, Ksawery Kopeć (team XYZ) | ABC: Attention Benchmark for Cognition | [wu](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/abc-attention-benchmark-for-cognition) | see section |

Host numbering is announcement order, not rank; the four grand prizes are numbered 1 to 4 by the host but are all $25k.

## Per-winner synthesis

### 1. MEDLEY-BENCH (Grand Prize, $25k), Abtahi, Karbalaie, Illueca-Fernandez, Seoane (Karolinska Institutet / KTH)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1773873293562) · [Benchmark](https://kaggle.com/benchmarks/farhadabtahi/medley-bench) · [Dataset](https://kaggle.com/datasets/farhadabtahi/medley-bench-data) · [GitHub, Apache 2.0](https://github.com/ki-smile/medley-bench) · PyPI `medley-bench` · [site](https://ki-smile.github.io/medley-bench/) + [manuscript](https://ki-smile.github.io/medley-bench/manuscript.html) · videos [1](https://youtu.be/QTo7d6mPU9E), [2](https://youtu.be/SflVaFtakck)

Bank H2 11 already covers the three-step protocol (solo A, private self-revision B, social revision C). New facts from the writeup:
- **Social step detail:** the social step shows **8 anonymised analyst opinions plus a consensus**, drawn adaptively from a 28-model analyst pool; each step is a separate, history-isolated call. (writeup)
- **Nudge finding:** the private self-revision nudge ("Which claim are you least confident about? What is the strongest counter-argument?") raised self-revision 4 to 15x over a generic instruction (37% to 96% of claims revised). (writeup)
- **Dataset:** 130 instances, 650 claim-level evaluations, 5 domains (medical differential diagnosis 27, system troubleshooting 26, code review 27, architecture design 25, statistical reasoning 25). **100 are intentionally ambiguous** (no ground truth), **30 known-answer** cases for adversarial validation; on 43 of 650 claims the consensus was judged wrong and scoring flipped to reward justified resistance.
- **Scoring:** MMS (Medley Metacognition Score) = geometric mean of three tiers (T1 reflective updating, rule-based; T2 social robustness, mostly rule-based; T3 epistemic articulation, hybrid) with a balance bonus; MAS = mean of four DeepMind sub-abilities (monitoring, control, evaluation, self-regulation). **75% rule-based, 25% from a three-judge LLM rotation with family-level exclusion** (a model is never judged by its own family).
- **Models:** **35 models from 12 families** via OpenRouter at temperature 0 (2026-03-29 to 04-03): Anthropic, Google, OpenAI, DeepSeek, Meta, Qwen, xAI, Mistral, Moonshot, Xiaomi.
- **Spread:** MMS 49.4 to 62.2 across 34 models; MAS spread 39.7 points. **Claude Haiku 4.5 ranks #1** (MMS 62.2). Gemma-3 12B (60.1) roughly matches Gemma-3 27B (61.1) and beats Gemma-4 31B (56.7); Gemini Flash-Lite (58.9) beats Gemini 3.1 Pro (58.5); GPT-4.1-Nano has the top social-robustness score (T2 60.1). (writeup)
- **Robustness checks:** four aggregation methods agree at rho >= 0.977 (max rank shift 3); bootstrap rho > 0.97 at 75 instances; analyst-pool overlap gives no T2 advantage (Mann-Whitney p = 0.50).
- **Adversarial stage (11 models):** consensus inverted with arguments unchanged, and a "stripped" condition with only the 2 weakest analysts. It splits models into **argument-evaluators** (Gemini 3 Flash, Claude Haiku, GPT-4.1-Nano, delta near 0) and **statistics-followers** (Mistral Small +11.6, Qwen 3 8B +7.3). The normal-mode normative/informational judge score predicts adversarial delta (Spearman rho = -0.82, p = 0.002), which the team pitches as a cheap sycophancy screen.
- **Headline insight:** Evaluation is every model's weakest relative ability across all 35 models (the "knowing-doing gap"); scale buys Evaluation, not Control.
- **Contamination control:** ambiguous, newly written professional vignettes with no single answer; no explicit contamination test reported.

### 2. LearningBench (Grand Prize, $25k), Karandeep Singh (Treow Intelligence)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/learningbench) · [Benchmark](https://kaggle.com/benchmarks/kdcyberdude/learningbench) · [GitHub](https://github.com/kdcyberdude/LearningBench)

Bank H2 11 covers the 135 tasks, the six sub-abilities with counts, and the scoring formulas. New facts:
- **No static dataset:** 135 programmatic scenarios generate examples and compute ground truth at runtime; "the grader is the rule", so **no LLM-as-judge and no grading tables**. (writeup)
- **Construction pipeline:** enumerate every facet of each ability, seed one task per facet, expand programmatically, filter. A frontier LLM plus a human reviewer audit every candidate on five criteria: human feasibility, solution uniqueness (exactly one rule fits training and held-out examples), logical consistency, **anti-contamination** (system genuinely invented), and "PhD-level" difficulty yet solvable by at least one frontier LLM or expert.
- **Discriminatory-power filter:** each survivor runs across **14 models**; tasks all 14 solve perfectly or none can touch are dropped. **About 95% of candidates rejected.** This is the most explicit "design for a gradient" step among the winners.
- **Trajectory orthogonal to level:** across 112 (model, task) procedural runs, Spearman rho(slope, asymptote) = -0.02, R^2 = 0.01, so the learning-slope signal is independent of final accuracy. (writeup)
- **Results (14 models, statistics with Spearman/Mann-Whitney/Wilcoxon, 10,000-sample bootstrap CIs, Benjamini-Hochberg FDR):** only **Gemini 3.1 Pro Preview** clears 0.70 (0.85); 11 of 14 below 0.50; **GLM-5** (open weights) is #2. Google + open models score +40.8% higher on rule induction than Anthropic + OpenAI (p = 0.029, Cliff's delta 0.71). (writeup)
- **Controlled A/B:** Qwen 3 Next 80B Thinking vs Instruct on 132 matched tasks: Thinking wins 87, Instruct 19, ties 26; gains Observational +0.43, Concept +0.38, RL +0.30; Procedural -0.11 (not significant).
- **Evidence appetite:** across 201 interactive runs, fewer examples requested correlates with higher score (rho = -0.52). "Well-calibrated" models (Gemini Pro, GLM-5, Qwen Thinking) use 37 to 47% of examples and score 0.67 to 0.78; "underconfident" ones (Claude Opus, Claude Haiku, DeepSeek, Gemma) use 83 to 96% and score 0.29 to 0.35.
- **Token signal:** across 397 RL runs, rho(tokens, score) = -0.53; solved runs average 41K tokens vs 177K for failures; 43 runs repeat the same action 10+ times.
- Kaggle Benchmarks SDK: yes (attached benchmark). Model access route not stated.

### 3. GAUGE (Grand Prize, $25k), Arjun Thilak and Ramkumar Muniandi Vallimayil (Eros Gen AI)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/gauge-grading-awareness-of-uncertainty-and-ground) · [Benchmark](https://kaggle.com/benchmarks/rarjunthilak/gauge-grading-awareness-of-uncertainty-and-ground) · [GitHub](https://github.com/arjunthilak05/gauge-metacognition)

**Ability:** metacognition, split per Nelson and Narens (1990) into monitoring (knowing how well you do) and control (acting on it). The pitch: a model at 95% confidence that never abstains is more dangerous than one that is simply wrong, because the operator trusts its confidence signal.

**Protocol (the "metacognitive staircase", three turns):** turn 1, predict the problem's difficulty before solving (an ease-of-learning judgment); turn 2, solve and report confidence 0 to 100 (Tian et al. 2023 style); turn 3, choose to **submit (+3 correct, -1 wrong) or abstain (+1 if the answer would have been wrong, 0 if right)**. With these payoffs abstaining maximises expected score below 40% confidence, so confidence becomes consequential. The team claims it was the only multi-turn protocol among 17 other metacognition submissions they reviewed (writeup).

**Scores:** M-Score averages monitoring metrics (ECE, Brier, AUROC, ease-of-learning correlation); C-Score is control efficiency in the game; **L-Score is the point-biserial correlation between stated confidence and the submit/abstain decision**, only computed with at least 3 abstentions; robustness checked with Kendall's tau.

**Dataset:** **270 items, procedurally generated** from parameterised templates with deterministic seeds; no LLM in generation; every item has a verifiable answer. Math 122 (15 templates with GSM-NoOp-style distractors, six difficulty levels), logic 115 (proof-tree-first generation with **fictional entities to block memorisation**), factual 33. Selected from 1,093 candidates by **Classical Test Theory item analysis**: item p-value 0.15 to 0.85, discrimination index > 0.2, exactly 45 items per difficulty level. This is the most psychometric item filter among the winners.

**Results (8 models, 2,160 model-item evaluations):** Gemini (3 models), Claude (3), Qwen (1), DeepSeek (1). Gemini 2.5 Pro: 94.1% accuracy, ECE 0.053, **L-Score 0.0, never abstained across 270 items**; all three Gemini models had at most one abstention, confidence SD as low as 0.018. Claude Haiku 4.5 abstained on 25 items with 84% precision (L = 0.870) despite the lowest accuracy (78.2%); Qwen3-235B abstained on 22 (L = 0.813, best overall 0.868); DeepSeek V3.1 on 6 (L = 0.587). At difficulty 5 to 6, Gemini 3 Flash reported 97% mean confidence at 73 to 84% accuracy. (writeup)

**Judges (724918):** "one frontier model achieved the best accuracy and calibration but never abstained once across 270 items, revealing that monitoring without control is a distinct and measurable failure mode."

### 4. Metaproteus / "Proteus" (Grand Prize, $25k), Joseph Boskovski (no affiliation)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1776217035880) · [Benchmark](https://kaggle.com/benchmarks/josephboskovski/metacognition) · [target-word notebook](https://kaggle.com/code/josephboskovski/proteus-target-words) · [dataset](https://kaggle.com/datasets/josephboskovski/proteus-output-target-words)

**Ability:** metacognitive knowledge of **one's own output distribution**: does a model know which words it will generate, and can it reject words it does not produce (confabulation resistance)? Distinct from factual calibration (Kadavath 2022) and from model-vs-model introspection (Binder 2024).

**Protocol per target word:** (1) **Elicitation:** 3 runs x 25 calls of "list 20 associates of [target]" give an appearance-rate distribution over 75 calls; words with std > 0.15 across runs are dropped as unstable. (2) **Banding:** core (rate >= 0.70), boundary (0.10 to 0.70), out (< 0.10); sparse bands merge into a 2x-weighted low band. (3) **Foils** the model never produces: top-quartile SWOW human associates absent from elicitation, and WordNet sister terms/hyponyms/hypernyms absent from both. (4) **Self-knowledge:** a **fresh model instance** predicts, for ~15 candidates per band, whether each would appear in another instance's list, with confidence 0 to 100 and a frequency estimate 0 to 10.

**Answer verification:** each trial is scored against the **model's own empirical appearance rate**, so ground truth is per-model and cannot be memorised. Headline = 1 - mean per-band Brier across 5 bands (proper scoring rule). Split-half Spearman ceiling per target; alternate cutoff thresholds reported (core size shifts <= 1.7 words between strict and loose).

**Dataset:** **48 target words** in a 12-cell design (2 concreteness x 2 POS x 3 frequency tiers), sourced from SUBTLEX-US, Brysbaert concreteness norms, SWOW-EN, word2vec, WordNet, with a profanity filter; 216 candidates screened by pilot elicitation, 48 more held in reserve. About 3,600 trials per model run; bootstrap 95% CI +/-0.02 to 0.04.

**Results:** 13 models from 5 labs run; 10 clean runs reported (47 targets). Gemini 3.1 Pro 0.970, Claude Sonnet 4.6 0.939, GPT-5.4 nano 0.926, Claude Haiku 4.5 0.919, Gemma 3-27b 0.869, DeepSeek v3.2 0.835, Gemma 3-4b 0.834, Gemini 2.5 Flash 0.817, Gemini 2.0 Flash Lite 0.748, Gemma 3-1b 0.705. Gemma 3 scales cleanly 1b to 4b to 27b. Failure modes: Sonnet 4.6 / GPT-5.4 nano / Gemma 3-27b **under-claim their own core outputs** (errors about -0.40); Flash / Flash Lite / DeepSeek **over-endorse everything**; Gemma 3-4b rejects foils almost perfectly (0.998) but fails its core (Brier 0.79). Small models use only 2 to 3 of 11 frequency values. (writeup)

**Honest caveat in the writeup:** API rate limits ruined 5 models (Gemma 3-12b, gpt-oss-20b/120b, Gemma 4-31b, Gemma 4-26b-a4b) and degraded Opus 4.6, GPT-5.4 and Qwen 80b, which are excluded or reported as lower bounds.

### 5. Turn Bench (Executive Functions, Winner 1 of 2, $10k), Simon Zhang and Bill Cai
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1775940834879) · [Benchmark](https://kaggle.com/benchmarks/zxsimon2/turnbench) · [live demo](https://turn-bench.onrender.com)

**Ability:** executive function (planning, execution, working memory, inhibition, flexibility), following Miyake et al.'s unity-and-diversity framework. Pitch: agentic benchmarks like WebArena cannot localise which cognitive component failed, while sterile lab tests lack ecological validity; so use **multi-turn games with trivially easy mechanics** where only EF separates winners from losers.

**Construction:** **two turn-based games x 4 variants x 100 scenarios = 800 scenarios**, procedurally generated, parameters stored as JSON.
- **Farm game** (flexibility, inhibition of prior commitments): reach a gold target within a day limit by gathering or building farms; mid-build, a **goalpost shift** changes target and deadline, and a randomised grace period (0 to 2 turns) defines when the old plan provably loses. Variants: urgent, silent (alert suppressed, must notice from the status line), super (must switch to a super farm), farms (a second farm needed). A **parameter validator mathematically guarantees exactly one viable strategy** per scenario.
- **Battle game** (planning, forecasting, adaptation): 3v3 creature battle with a hidden, randomised effectiveness matrix (each creature 2x / 1x / 1/3x against exactly one opponent), plus decoy "trap" actions. Variants: standard, reversal (matrix flips at turn 10), cycling (opponent switches every 2 attacks), resistance (no matrix; damage decays while you stay).

**Answer verification:** state-based game engines decide win/loss; action in `\boxed{}`, parser takes the last box with raw-text fallback; invalid actions re-prompt instead of forcing a loss. Early stopping (simulate the rest once one side has one creature; farm win on first correct pivot). First 3 scenarios run sequentially for inspection in Kaggle's compare-model UI, remaining 97 in parallel; raw conversations kept in notebook logs.

**Results (12 models, 4 families):** average scores 0.34 to 0.92; every variant gives a statistically significant gradient (writeup). The key design move is the **variant pair with opposing demands**: on standard, Haiku 87% and Sonnet 80%; on resistance the ranking inverts (GPT-5.4-mini 99%, GPT-5.4-nano 97%, Haiku 45%, DeepSeek 47%), which the team reads as passivity that happens to be optimal, not strategy. Opus collapses on reversal (42%, Haiku 80%) but leads cycling (86%). Haiku pivots 98% in urgent but 8% in super (sunk-cost persistence when the alternative needs more planning). Successful models increase reasoning after the alert (Gemini 3.1 Pro 1.8x, Opus 1.9x); verbose models over-deliberate on routine turns.

**Judges (724918):** "executive function is a context-dependent profile, not a monolithic capability that scales with model size."

### 6. SecureExec-Bench (Executive Functions, Winner 2 of 2, $10k), Muhammad Haaris
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1776339266224) · [Benchmark](https://kaggle.com/benchmarks/muhammadhaaris27083/secureexec-benchmark) · no GitHub linked. Affiliation: "Coursework in Information Security and Ethics".

**Ability:** executive function in a security frame (Miyake's inhibitory control, working-memory updating, flexibility): when helpfulness and safety conflict, which wins, does it change under escalating pressure, and can a model plan security work without leaking the secrets it must protect?

**Construction:** **5 subtasks, 235 evaluated scenarios** (270 built, 35 held for future), four difficulty tiers with weighted scoring (low 0.5x, medium 1.0x, hard 1.5x, extreme 2.0x), all **entirely synthetic**.
1. PII inhibition (55): complete a task without repeating 50+ PII types, including base64/hex-encoded SSNs and cards, Aadhaar, IBAN, multi-field composites.
2. Credential working memory (40): a credential (38 types: AWS, GCP, Stripe, Vault, Ethereum keys) is shared early, distractor tasks follow, last turn asks for recall; 3 to 6 turns.
3. Helpfulness-security conflict (40): 40 distinct pressure tactics (authority, urgency, DAN jailbreaks, CEO fraud, fake subpoenas, system-message injection) demanding the credential back.
4. Security planning (45): code review, threat modelling, incident response, DPIA with real-looking credentials embedded; 80+ security categories.
5. Multi-turn escalation (55): 55 strategies (boiling frog, false prior-session claims, cross-language extraction), 4 to 8 turns.

**Answer verification (layered):** (1) **case-insensitive regex hard gate**: any leak scores 0.0; (2) positive regex for security-aware language; (3) **judge-LLM** `assess_response_with_judge()` on 4 to 5 scenario-specific criteria; (4) for planning, a typed `SecurityReviewSummary` dataclass (structured output). A custom `assert_security_aware()` penalises both empty refusals and verbose oblivious answers. Built on the **`kaggle-benchmarks` SDK**: one `@kbench.task` per subtask, multi-turn via `chats.new()` context managers (the most explicit SDK description among winners).

**Contamination control:** all prompts purpose-built, none adapted from existing jailbreak databases (writeup).

**Results (14 models, 6 providers):** overall 0.14 to 0.69 (4.7x). Claude Opus 4.6 0.69, Claude Haiku 4.5 0.67, Claude Sonnet 4.6 0.66, GPT-5.4 mini 0.62, GPT-5.4 nano 0.52, Gemini 3.1 Flash-Lite 0.46, Gemma 4 31B 0.44, Gemma 4 26B A4B 0.40, Gemini 2.0 Flash 0.33, Gemini 3 Flash 0.33, Gemma 3 27B 0.28, Qwen3 80B Thinking 0.25, Gemini 2.5 Flash 0.22, DeepSeek-R1 0.14. Claimed findings: **reasoning models cluster at the bottom** (R1, 2.5 Flash, Qwen Thinking), hypothesised as more surface for disclosure; security planning tops out at 0.44; credential working memory has the widest spread (Sonnet 0.90 vs Gemma 4 26B A4B 0.30). (writeup)

### 7. GrammarGym (Learning, Winner 1 of 2, $10k), Raphael B (Raphaël)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1775210475900) · [Benchmark](https://kaggle.com/benchmarks/raphab/grammargym) · [dataset "Artificial Grammars"](https://kaggle.com/datasets/raphab/artificial-grammars) · [GitHub generator](https://github.com/raphaelgbe/ags_exps/tree/main). Writeup lists affiliation as "Kaggle, Google" (note for any post: a host-side affiliation; the writeup does not explain it).

**Ability:** learning, via the **artificial grammar learning (AGL)** paradigm from psychology (Reber): synthetic rule systems, novel by construction, give an unlimited supply of fake "theories" to learn without memorised knowledge.

**Dataset:** 18 artificial grammars, 34 labelled datasets in one CSV; each has 10 to 130 training samples (more valid than invalid) and **exactly 200 inference samples balanced valid/invalid**; grammars serialised. Generated by code plus manual review and curation (notebook in the repo). Two "creole emergence" tasks generate their own languages on the fly with a fixed seed (random tokens mapped to predefined words, per-agent word order).

**14 tasks, grouped:** (1) baseline: classify 25 strings with no training (chance 0.5); (2) **passive** exposure, reproducing Reber's original experiment (valid-only training) plus variants with invalid samples, ordered short-to-long or not; (3) **active** mode with an oracle the model can query, with and without reasoning, matched on number of samples seen; (4) **fake-history "Pygmalion" tasks**: an injected fake conversation where the model was always wrong vs always right, same information, to measure susceptibility; (5) **rule formation**: the model writes a regex for the grammar, tested on the 200 inference samples (with and without rationale); (6) **generation** of 20 new valid strings, one version adding a diversity factor; (7) **creole emergence**: two agents with different languages play a resource-exchange game and must converge on mutual understanding (language learning + teaching with a moving target).

**Answer verification:** grammar membership is computed exactly; regex and generated strings are checked against the grammar. Creole tasks score a weighted sum of game success, lexicon overlap over turns, and **BLEU** between agents on test samples.

**Models:** 6 (Claude Haiku, DeepSeek V3.2, Gemini 2.5 Flash, Gemma 4 31B, GPT-5.4, Gemini 3 Flash). **No numeric scores in the writeup** text; it reports qualitatively that differentiation is real but often on narrow ranges because chance is high; hiding invalid examples hurts; active mode hurt every model; the fake history hurt every model except the Geminis; productive tasks (generation, rule formation) discriminate more; Gemini 3 leads and DeepSeek trails on creole tasks. Smallest model panel and least quantitative results among the track winners.

### 8. EphLangBench (Learning, Winner 2 of 2, $10k), Bruce C. (team "Ephlang", student researcher at UC Berkeley)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/ephlang-bench-testing-llm-algorithmic-understandi) · [Benchmark](https://kaggle.com/benchmarks/brucec/ephlang-bench) · [dataset](https://kaggle.com/datasets/brucec/ephemeral-language-benchmark-data)

**Ability:** learning (in-context acquisition of a never-seen programming language), with the team also mapping it onto executive functions (abandon Python habits), metacognition (fix from feedback) and attention (find the relevant part of the spec).

**Construction:** each task gives a full language specification (grammar, keyword map, operator semantics, worked examples), a natural-language algorithmic problem, and a stub in the ephemeral language. **Languages are generated per session from a seed:** all Python keywords renamed, operators replaced by random multi-character symbols, builtins aliased, and **notation drawn equally from infix, Polish prefix, and reverse-Polish postfix**; a Python-similarity metric keeps generated languages visibly unlike Python. The model gets **up to 4 turns** with compiler errors / test failures **translated into ephemeral-language terms** (to avoid leaking Python).

**Dataset:** **200 problems from APPS** (50 introductory, 50 interview, 100 competition), filtered from 10,000+ by transpiler compatibility (~90% pass, tested on 3 seeds), quality ranking (test density 60%, solution diversity 40%), and execution validation over **3 notations x 10 seeds = 6,000 roundtrips**. Average 82.9 hidden tests per problem (20 to 1,440).

**Answer verification:** solutions are transpiled back to Python (Lark parser on the generated grammar) and executed against hidden tests in a sandbox (memory limits, blocked imports, timeouts). Score `max(0, 1 - (k-1)/4)` for solving on turn k. Harness built on Kaggle's benchmark SDK for multi-turn chats; **$234 total spent on Kaggle Benchmarks**.

**Results (10 models, full 200; Python baseline on a 40-problem subset):** Gemini 3.1 Pro 0.731 (89% pass, $83), Gemini 3.0 Flash 0.575 (70%), Claude Opus 4.6 0.446 (63%), Gemma 4 31B 0.346 (47%, $1), Claude Sonnet 4.6 0.336 (48%), GPT-5.4 0.253 (36%), Gemini 2.5 Flash 0.151 (24%), DeepSeek R1 0.121 (20%), DeepSeek V3.2 0.065 (11%), GPT-5.4 Mini 0.046 (7%). Python baseline 0.73 to 0.97. **Pass rates 7% to 89%**, a 48pp spread even among six frontier models vs ~12pp on HumanEval. Infix 56% vs prefix 32% vs postfix 37%. Gemini 3.1 Pro solves 47% first-turn; Opus is an "iterative debugger" (21.5% first-try, 53% recovery); DeepSeek R1 repeats the same error class 80% of the time. Bootstrap CIs (10,000) and McNemar with Bonferroni; Gemma 4 vs Sonnet statistically tied. (writeup)

**Contamination control (honest limitation):** APPS problems are likely in training data; the novelty is syntactic, not semantic, and the Python baseline isolates adaptation from algorithmic skill. Future work: contamination-free problems.

### 9. ESFP Benchmark (Metacognition, Winner 1 of 2, $10k), Binwen Liu and Yilin Ren
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1776173871950) · [Benchmark](https://kaggle.com/benchmarks/binwen666/esfp-epistemic-stance-flexibility-probing) · [dataset](https://kaggle.com/datasets/binwen666/esfp-v2) · [GitHub](https://github.com/Binwen6/ESFP-Benchmark). Longest writeup of the set (~4,500 words).

**Ability:** metacognition as **epistemic stance flexibility**: can a model switch on cue between neutral reporting ("what do experts think?") and expressing its own position ("what do you think?")? It explicitly sets aside whether models "have" subjectivity. Two failure profiles: **over-aligned** (never takes a stance, an echo of consensus) and **under-aligned** (opines even on facts). Grounded in the clinical Metacognition Assessment Scale (Semerari et al. 2003), with MAS domains mapped to metrics.

**Construction:** **104 curated statements** in six categories: normative policy (20), open social phenomena (15), personal value trade-offs (15), disciplinary facts (24, the "floor": should report), empirically contested claims (15), aesthetic judgments (15). Sources: OpinionQA, MMLU, Stanford Human Preferences, Metaculus, plus original items. Each item x **five phrasing templates** (P0 neutral, P1 de-subjectified, P2 strongly subjectified, P3 responsibility-deflected, P4 disagree-invite) = **520 single-turn prompts per model**; responses capped at 80 to 150 words of plain prose via a shared system prompt.

**Answer verification (no ground truth; behavioural contrast metrics):** AR (regex ratio of first-person to total attribution markers, delta P2 - P1); PSI (1 - cosine of all-MiniLM-L6-v2 embeddings, P0 vs P1); **SCD** (sentence-level labels A fact / B own stance / C filler by an **LLM judge panel with majority vote**, writeup names Qwen3-235B-A22B and Gemini-2.0-Flash-Lite though it calls it a three-judge panel; capped at 25 sentences); CPC (Fleiss' kappa of stance direction over P0/P2/P4). Composite: `flex = 0.20 dAR + 0.50 dSCD + 0.30 PSI_scaled`, `ESFP = flex x (1 + 0.25 CPC)`; PSI ceiling set post hoc at the 95th percentile (two-pass recompute). Item-level bootstrap (1,000) CIs.

**Kaggle Benchmarks SDK:** each call in an isolated `kbench.chats.new()` context, async `asyncio.gather` with semaphores (10 for Claude/DeepSeek/GLM, 20 otherwise), Parquet checkpoints per model. They note the **Kaggle Model Proxy per-session quota** capped them at 8 models; DeepSeek-R1 dropped because its long traces broke SCD judging.

**Results (8 models):** DeepSeek-V3.2 0.805, Claude-Sonnet-4.6 0.782, **Gemma-3-27B 0.781** (top three statistically tied; Gemma has the best dSCD 0.552 and dAR 1.000), GLM-5 0.690, Gemini-3.1-Flash-Lite 0.619 (best CPC 0.493), Claude-Haiku-4.5 0.612, Qwen3-Next-80B Thinking 0.502, **Gemini-3.1-Pro 0.399** (below Flash-Lite on every sub-metric). dSCD spans 0.091 to 0.552 (6x) and is the main differentiator; PSI clusters (0.795 to 1.000) and barely discriminates. Every model shows positive flexibility; no aggregate over-alignment found. (writeup)

**Judges (724918):** results "reveal how instruction-tuning pushes models toward being tools that suppress autonomous self-expression."

### 10. Metacognitive Calibration Benchmark (Metacognition, Winner 2 of 2 per host thread, $10k), Qing Zhou (independent)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/metacognitive-calibration-benchmark-do-models-kno) · [notebook](https://www.kaggle.com/code/zhouqing1/deepmind-agi-benchmark) · [OpenFEP](https://github.com/zhouqingzq/OpenFEP). No benchmark link in the writeup's attachment list (only the notebook). **API carries no award on this writeup**; see the discrepancy note at the top.

**Ability:** metacognitive calibration under partial evidence, framed as a clinical assessor working from limited notes: form hypotheses, track which inferences are supported, say "I need more information". **Big Five personality inference from short behavioural vignettes is the vehicle**, not the target.

**Task:** a 1 to 5 sentence vignette about a fictional person; the model returns **structured JSON**: trait estimates per Big Five dimension (1.0 to 5.0, or **null** if evidence is insufficient), confidence 0 to 1, a missing-evidence list, a contradictions list.

**Ground truth (two stages):** (1) automated signal extraction by OpenFEP's PersonalityAnalyzer (the author's own open-source cognitive architecture); traits deviating > 0.05 from neutral and not flagged malleable count as evidence-supported; (2) a **rule-based validity audit** of the text (concrete evidence, missing evidence, planted contradictions, trap content: quotes, hypotheticals, third-party attributions, fictional projections, negations), with targeted human review; cases tagged rule-valid / analyzer-valid / both-valid.

**Scoring:** confidence calibration 40% (rise with signal; < 0.3 when absent; 0.2 to 0.7 when contradicted), missing-evidence F1 25%, uncertainty acknowledgment 20% (null scores best), contradiction detection 15%.

**Dataset:** **80 English vignettes**, parametrically generated (not from existing corpora): 19 high-signal, 19 sparse, 19 contradiction, 19 trap, 4 multi-dimensional. Audit: contradiction and trap 19/19 both-valid; high-signal 13/19, sparse 15/19 ("under active cleanup").

**Results (4 models, 320 evaluations):** Gemini 2.5 Flash 0.637, DeepSeek R1 0.632, Gemini 2.5 Pro 0.615, Claude Sonnet 4.6 0.527 (lowest mean, smallest std 0.144, "flat overconfidence"). All models score best on sparse (0.73 to 0.91) and worst on high-signal (0.40 to 0.53): better at abstaining than at raising confidence when evidence is strong (flagged provisional by the author). R1 best on contradictions (0.688) but worst on multi-dimensional (0.328). (writeup)

**Assessment for the post:** the smallest model panel (4) and item count (80) among the winners, with self-declared partially cleaned slices and ground truth partly derived from the author's own analyzer. If the host thread is right, it shows that a sharp construct and honest caveats can win a track over scale.

### 10'. MetaBound (Metacognition, Winner 2 of 2 per writeup API only, $10k), Ahmed A. (independent)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1774955644602) · [Benchmark](https://kaggle.com/benchmarks/archie40004/metabound) · [dataset](https://kaggle.com/datasets/archie40004/benchmark-scenarios) · [main notebook](https://kaggle.com/code/archie40004/metabound-ai-competence) + 4 task notebooks · [GitHub](https://github.com/Ahm3dAlAli/MetaBound). **Not named in host thread 724918**; the API award (prize 911) is the only evidence it won.

**Ability:** metacognitive boundary awareness in healthcare, finance and legal QA: distinguish what one knows, does not know, and what cannot be known.

**Four tasks, 200 items each (800 total), authored from scratch and cross-verified against domain references:**
1. **Predict-then-answer** (monitoring): confidence 0 to 100 before answering; `1 - |conf - correct|/100`.
2. **Difficulty gradient** (sensitivity): 10 controlled difficulty levels; model rates difficulty and confidence; Spearman of rated vs true level; item score 0.4 difficulty accuracy + 0.6 calibration.
3. **Domain boundary** (knowledge): 3 domains x 4 subdomains spanning training-data frequency (cardiology vs rare genetic disorders; contract vs indigenous law) x 4 depths with expected confidence ranges; headline = negative **confidence slope** across depth.
4. **Unknown unknowns** (epistemic vigilance): 30 answerable, 30 tricky, **140 unanswerable false-premise items** in five fabrication types (fake drugs/regulations/case names, non-existent versions, wrong attributions, impossible dates, false premises), template-generated with manual review against real-entity overlap; score matrix penalises confabulation heavily and over-refusal lightly.

**Answer verification / SDK:** each task is a separate Kaggle Benchmarks task on the **`kaggle-benchmarks` SDK**; typed `ground_truth` arrays with per-item modes (`any_of`, `contains_key_terms`), deterministic parsing of `CONFIDENCE:` / `QUESTION_STATUS:` lines, and an **LLM-judge fallback** for ambiguous free-form answers. Composite weights T1 0.25, T2 0.25, T3 0.20, T4 0.30.

**Results (7 models):** GLM-5 0.81, Qwen 3 Next 80B Thinking 0.74, Gemma 4 26B A4B 0.70, DeepSeek V3.2 0.66, Gemini 3.1 Flash-Lite Preview 0.61, Claude Opus 4.6 0.52, GPT-5.4 nano 0.47. T1 saturated (0.85 to 0.98); T3 a shared blind spot (0.38 to 0.61); T2 spreads 0.30 to 0.82; **T4 bimodal: GLM-5, Qwen, Gemma 4 at 1.00, Claude Opus 4.6 and GPT-5.4 nano at 0.00 (confabulated on all 140 false-premise items)**. (writeup) This 0.00 for Opus 4.6 is a strong claim worth double-checking before quoting; other winners (SecureExec, MEDLEY) found Anthropic models near the top on related constructs.

### 11. HedgeDecode (Social Cognition, Winner 1 of 2, $10k), Patricia Ji (independent)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/hedgedecode) · [Benchmark](https://www.kaggle.com/benchmarks/pyji08/hedgedecode) · [GitHub](https://github.com/patriciayji27/HedgeDecode)

**Ability:** social cognition: inferring communicative intent and producing a contextually appropriate response (DeepMind taxonomy 7.10.1 and 7.10.3) when a user's intent is hedged or face-saving ("lol", "it's fine", "whatever"). The targeted failure is **face-saving mechanism naming**: the model correctly detects minimisation, then says so ("sounds like part of you still wants to check"), making the user's protective framing the topic. Grounded in Goffman's face theory, Grice's implicature, Brown and Levinson's politeness theory; positioned beyond ToMi, SocialIQa, FANToM, which stop at mental-state attribution.

**Construction:** four response policies (LITERAL, SUPPORT, INFER, CLARIFY; SUPPORT outranks INFER when acute distress is present). **Nine tasks in three families, 114 newly authored items** across friend/roommate/colleague/family/academic/service settings:
- Family A, policy classification (42 items): LITERAL/SUPPORT (14), SUPPORT/INFER (12), INFER/CLARIFY (16).
- Family B, context flips (40): same utterance in different prior contexts; answer must change or stay stable.
- Family C, "no-name" execution (32): free response scored by explicit binary checks.

**Answer verification:** **A and B are fully deterministic** (explicitly to avoid the circularity of an LLM judging social meaning). C combines deterministic checks (sentence count, forbidden/required phrases) with **one binary judge criterion per item**, written as a concrete behavioural test; inconsistent criteria were rewritten. Ceiling and floor variants removed after a pilot. **Label validity: 5 independent raters on 30 Family A items, Cohen's kappa 0.78** (0.84 LITERAL/SUPPORT, 0.71 INFER/CLARIFY). Built with the `kaggle-benchmarks` SDK; single run with default settings; three repeats on four tasks kept top and bottom ranks stable (variance within +/-0.08).

**Results (10 models):** Claude Sonnet 4.6 0.84, Claude Opus 4.6 0.81, DeepSeek V3.2 0.76, Gemini 3.1 Pro Preview 0.76, GPT-5.4 mini 0.75, GPT-5.4 0.75, Gemini 3.1 Flash-Lite Preview 0.75, DeepSeek V3.1 0.71, Gemini 2.5 Flash 0.59, GPT-5.4 nano 0.59. Strongest discriminator: INFER vs CLARIFY (0.13 to 1.00, SD 0.30); hardest: `no_name_infer_relational`, capped at 0.60 for every model. A **recognition-execution gap**: per-criterion pass rates C1 face-save affirmation 89%, **C2 mechanism naming 76%** (56% on the hardest task), C3 register mismatch 84%. (writeup)

**Contamination control:** all items newly authored. **Limitations stated:** English-only informal text, small rater pilot, judge dependence in Family C.

### 12. AdvisorBench (Social Cognition, Winner 2 of 2, $10k), Ana Pirtac (independent)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/advisorbench-do-ai-models-widen-the-advisory-divi) · [Benchmark](https://kaggle.com/benchmarks/pirtacana/advisorbench) · [dataset](https://kaggle.com/datasets/pirtacana/advisorbench-dataset)

**Ability:** social cognition as robustness of advice to the user's communication style: does a model give substantively weaker recommendations to users who write informally or at lower literacy? Adapts the **audit-study (paired-resume) logic of Bertrand and Mullainathan (2004)** to AI advice.

**Construction:** **180 hand-authored scenarios** across six domains (financial, career, education, life, health, legal; 30 each), each in two variants: **L0** (fragments, hedging, emotional framing, colloquialisms) and **L2** (structured, professional, explicit parameters), **with identical decision-relevant facts** (income, savings, debt, age, family, constraints) = 360 rows. The model is never told the literacy level; the objective context paragraph is withheld at query time. Each scenario has **five recommendation options ordered from most ambitious to most cautious** plus a gold answer, which lets the scorer detect a directional "conservative penalty".

**Answer verification:** one `kaggle-benchmarks` task per domain with **structured JSON output** (one required field: the chosen option); `temperature=0.0`; **fully deterministic scoring, no judge LLM**. Headline "substantive advice equity" multiplies recommendation parity (same option for L0 and L2) by gold-proximity parity, so a model cannot recover credit after a flip by staying near gold. Secondary: gold accuracy, flip rate, conservative-shift rate.

**Results (16 models, 96 model-domain runs):** mean equity 0.739; mean flip rate 29.6%; in 67 of 96 runs the high-literacy user got higher gold accuracy. **Every model is more accurate for L2 on average** (mean +6.4 points). Best overall **Gemma 4 26B A4B 0.817**; Gemini 3.1 Flash-Lite Preview 0.787; GPT-5.4 nano 0.682 (0.547 in finance); worst literacy gap Gemini 2.0 Flash Lite (45.5% L0 vs 66.7% L2). Education is the harshest domain (0.666, 36.7% flips, +11.5 L2 advantage); health is partly saturated and the only domain with a negative L2 advantage. (writeup) Full 16-model list not given in text (figures only).

**Contamination control:** no items sourced from existing benchmarks. A paired-counterfactual design is itself contamination-robust, since the score is the difference between two prompts.

### 13. RIAC: Repetition-Induced Attention Collapse (Attention, Winner 1 of 2, $10k), Shankha Suvro Mullick (independent)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1775848234859) · [Benchmark](https://kaggle.com/benchmarks/shankhasuvromullick/riac-repetition-induced-attention-collapse) · [notebook "100 distractors are easier to ignore than 1"](https://kaggle.com/code/shankhasuvromullick/100-distractors-are-easier-to-ignore-than-1)

**Ability:** selective attention: at what repetition load does a model fail to extract a target value when distractors flood the context, and is the failure lure capture, position bias, or hallucination?

**Construction:** four Kaggle Benchmark tasks, one per distractor mechanism (number repetition = plausible integer lures; word repetition = domain-relevant semantic noise; symbol repetition; off-topic sentences as a pure-volume baseline). **Fully crossed factorial: 52 hand-authored questions (finance, medicine, law, science, everyday) x 3 target positions x 4 distractor types x 6 load levels (1, 5, 10, 25, 50, 100) = 3,744 instances** (936 per task, 156 per type-level cell), 9 to 875 words each, `GLOBAL_SEED = 42`.

**Answer verification:** prompt asks for only the number; **regex extracts the first integer, strict equality; no LLM judge**; one API call per instance. Five derived metrics: ACP (first load with accuracy < 60%), LCR (lure captures / failures, possible because lure identities are planted), PBS (accuracy at beginning minus end), RCS (accuracy at L=100 minus L=1), DSI (noise minus number-lure accuracy at equal load). Built with `@kbench.task`; **total evaluation cost $3.62** for 3,744 instances (writeup), the cheapest reported among winners.

**Results:** "33 frontier models evaluated", 17 fully completed models tabulated. The task score is **binary per task (PASS only at 936/936)**, so the 0 to 0.75 overall score is coarse: Gemini 2.5 Flash, Gemini 2.5 Pro, Gemma 4 31B at 0.75 (fail only number repetition); Claude Haiku 4.5, Sonnet 4, GPT-5.4, Gemini 2.0 Flash/Flash Lite, Qwen 3 Next 80B Instruct at 0.50; all Claude Opus 4.1/4.5/4.6, Sonnet 4.5/4.6, DeepSeek V3.1/V3.2 and Qwen 3 Next Thinking at 0.00. Raw counts discriminate more (e.g. Gemini 2.0 Flash Lite 638/936 on number repetition; Claude Opus/Sonnet 4.5+ at 744 to 802/936 on symbol repetition while Gemini scores 936/936).

**Headline insight (judges and writeup):** **reverse attention collapse**: number lures cause 48.7% failure at L=1 but 0% at L=100 (RCS +0.487); 100% of number-repetition failures were lure captures (no hallucination); off-topic noise causes 0% harm at L=1 (failure is content-driven, not length-driven); position bias 94.9% beginning vs 90.4% end. The writeup does not make clear which model or pool the L=1 curve aggregates (its fingerprint example is Gemini 2.5 Flash, whose table row shows 930/936), see UNVERIFIED.

### 14. ABC: Attention Benchmark for Cognition (Attention, Winner 2 of 2, $10k), Patrycja Węgrzynowicz and Ksawery Kopeć (team XYZ, independent)
[Writeup](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/abc-attention-benchmark-for-cognition) · [Benchmark](https://kaggle.com/benchmarks/patrycjawegrzynowicz/abc-selective-attention) · [dataset](https://kaggle.com/datasets/patrycjawegrzynowicz/abc-factorized-attention-benchmark) · [utils notebook](https://kaggle.com/code/patrycjawegrzynowicz/abc-selective-attention-utils) · [GitHub](https://github.com/pwegrzynowicz/abc-benchmark)

**Ability:** selective attention under interference, split into **feature-sensitive** attention (relevance from local attributes or logical combinations: conjunction, negation, disjunction) and **structure-sensitive** attention (relevance from organisation: grouping, region, continuity, scope), the latter inspired by **Gestalt principles** (proximity, similarity, common region, continuity). The only winner that is **multimodal (text and visual)**.

**Construction:** a factorised design space (attention family x basis x modality x task type); this submission instantiates selective attention x {feature, structure} x {text, visual} x {filtering, counting}. Each example = input (text record set or rendered scene) + selection rule + output (the matching subset, or its count). Each task group has **dimensions** (source of pressure) and **variants**; **baseline dimensions** test the bare operation first, so that a later failure under distractors can be attributed to attention rather than to weak scene parsing. In text, structure is expressed through grouping, spacing, indentation and layout.

**Dataset:** fully synthetic, programmatically generated from explicit underlying states with deliberate near-miss distractors: 780 feature-text, 630 feature-visual, 390 structure-text, 360 structure-visual = **2,160 examples** (matches the host's figure).

**Answer verification:** gold computed from the generating state and the rule (correct by construction); counting returns an exact integer in a fixed schema, filtering returns an exact structured subset checked deterministically. No judge.

**Results:** **15 models** (frontier multimodal and open); the **writeup text gives no numeric scores**, only tiers: Gemini family most balanced across text and vision; a second frontier group degrades on structure-sensitive slices; weaker models collapse on hard visual-structure tasks. Failure modes are diagnosed per dimension (adversarial confounds = weak suppression; negation/disjunction = unstable rule maintenance; continuity/common region = weak structural binding; scope = hierarchy flattening). Thin on quantitative results for a winner; figures may carry them.

## Feature matrix

Columns: 1 MEDLEY, 2 LearningBench, 3 GAUGE, 4 Metaproteus, 5 Turn Bench, 6 SecureExec, 7 GrammarGym, 8 EphLang, 9 ESFP, 10 MetaCal (Qing Zhou, host winner), 10' MetaBound (API winner), 11 HedgeDecode, 12 AdvisorBench, 13 RIAC, 14 ABC. This is a benchmark-building hackathon, so the generic rows "fine-tuned" and "on-device" are N for every entry and are omitted; "model used" becomes "models evaluated".

| Row | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 10' | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cognitive ability | Metacog | Learning | Metacog | Metacog | Exec fn | Exec fn | Learning | Learning | Metacog | Metacog | Metacog | Social | Social | Attention | Attention |
| Number of items | 130 inst / 650 claims | 135 envs | 270 | 48 words, ~3.6k trials | 800 | 235 | 34 sets x 200, 14 tasks | 200 | 104 x 5 = 520 | 80 | 800 | 114 | 180 pairs | 3,744 | 2,160 |
| Models evaluated | 35 | 14 | 8 | 13 (10 clean) | 12 | 14 | 6 | 10 | 8 | 4 | 7 | 10 | 16 | 33 (17 tabulated) | 15 |
| Score spread (writeup) | MMS 49.4-62.2 | top 0.85, 11/14 < 0.50 | acc 78-94%, L 0-0.87 | 0.705-0.970 | 0.34-0.92 | 0.14-0.69 | not numeric | 7-89% pass | 0.399-0.805 | 0.527-0.637 | 0.47-0.81 | 0.59-0.84 | equity 0.682-0.817 | 0-0.75 (binary) | not numeric |
| Answer verification | 75% rule, 25% 3-judge LLM rotation | programmatic (grader = rule) | deterministic | own empirical rates, Brier | game engine | regex gate + LLM judge + dataclass | exact grammar check, BLEU | execute hidden tests | regex + embeddings + LLM-judge panel | JSON vs analyzer + rule audit | string match + LLM-judge fallback | deterministic + 1 binary judge criterion | deterministic JSON | regex int equality | deterministic exact match |
| LLM judge anywhere | Y | N | N | N | N | Y | N | N | Y | N | Y | Y | N | N | N |
| Multi-turn / interactive | Y | Y | Y | N | Y | Y | Y | Y | N | N | N | N | N | N | N |
| Procedural / programmatic generation | N | Y | Y | N | Y | N | Y | Y | N | Y | Y (T4 templates) | N | N | Y | Y |
| Contamination control | ambiguous new vignettes | invented systems + audit | procedural, fictional entities | self-referential truth | procedural, validated | purpose-built, no jailbreak DBs | synthetic grammars | novel syntax, APPS problems (partial) | n/a (contrast metric) | parametric vignettes | from scratch + fabricated entities | newly authored | new items, paired | hand-authored | synthetic |
| Human validation of items/labels | ? | Y | N | N | N | N | Y (curation) | N | N | Y (targeted) | Y (manual review) | Y (kappa 0.78) | Y (hand-authored) | N | N |
| Statistical CIs / tests | Y | Y | Y (Kendall) | Y | Y | N | N | Y | Y | N | N | partial (repeats) | N | N | N |
| Kaggle Benchmarks SDK named in writeup | ? (benchmark attached) | ? (attached) | ? (attached) | ? (attached) | Y | Y | ? (attached) | Y | Y | ? (notebook only) | Y | Y | Y | Y | ? (attached) |
| Own dataset built | Y | Y | Y | Y | Y | Y | Y | Y (curated APPS) | Y | Y | Y | Y | Y | Y | Y |
| Eval rigour | benchmark + baseline | benchmark + controlled A/B | benchmark | benchmark | benchmark | benchmark | qualitative | benchmark + Python baseline | benchmark | small quantitative | benchmark | benchmark | paired counterfactual | benchmark | qualitative in text |
| Video | Y | N | N | N | N | N | N | N | N | N | N | N | N | N | N |
| Live demo / site | Y (site) | N | N | N | Y | N | N | N | N | N | N | N | N | N | N |
| Open code (GitHub) | Y | Y | Y | N (notebook) | N | N | Y | N (dataset) | Y | Y (OpenFEP) | Y | Y | N | N (notebook) | Y |

**Counts (15 entries, counting both Metacognition candidates):** 15/15 built their own data; 10/15 score fully deterministically (no LLM judge); 7/15 are multi-turn or interactive; 8/15 use procedural or programmatic generation; 9/15 report bootstrap CIs or formal tests; 9/15 link a GitHub repo; 1/15 has a video; 2/15 have a site or demo; 8/15 name the `kaggle-benchmarks` SDK explicitly in text. Track split of the grand prizes: 3 of 4 are metacognition (MEDLEY, GAUGE, Metaproteus), 1 learning. Median models evaluated: 12.

## Consensus vs divergence

**Table stakes (nearly every winner):**
- **Novel-by-construction items** with a stated anti-contamination argument (15/15). The learning-track winners go furthest: invented systems (LearningBench), artificial grammars (GrammarGym), ephemeral languages (EphLangBench).
- **Machine-checkable answers**: constrained outputs (boxed actions, JSON fields, integers, ordered option menus) and deterministic scoring; LLM judges, when present, are fenced in (MEDLEY 25% with family exclusion; HedgeDecode one binary criterion; MetaBound fallback only).
- **A demonstrated gradient**: every winner with numbers reports a spread across a multi-family model panel, matching the 30% "discriminatory power" criterion. LearningBench drops tasks every model solves or none solves (95% rejected); GAUGE uses CTT item filters; HedgeDecode removes ceiling/floor variants after a pilot.
- **One isolating design move per benchmark**: step separation (MEDLEY's private vs social revision, GAUGE's monitoring vs control turns), variant pairs with opposing demands (Turn Bench), paired counterfactual prompts (AdvisorBench), crossed factorial load x position (RIAC), baseline dimensions (ABC).

**What separated grand prizes from track prizes:** larger model panels (35, 14, 8, 13), explicit robustness analysis (MEDLEY's aggregation and bootstrap checks, LearningBench's FDR-corrected tests, Metaproteus's cutoff sensitivity and split-half ceiling, GAUGE's Kendall check), and a headline finding phrased as a new failure mode ("monitoring without control", "knowing-doing gap", "under-claims its own outputs"). Track winners include entries with no numbers in text (GrammarGym, ABC) and small panels (Qing Zhou: 4 models, 80 items).

**Contradictions across winners:**
- **Does reasoning help?** LearningBench: Qwen 3 Next 80B Thinking beats Instruct on 87 of 132 matched tasks ([LearningBench](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/learningbench)). SecureExec: reasoning models (R1, Gemini 2.5 Flash, Qwen Thinking) fill the bottom three ([SecureExec](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1776339266224)); Turn Bench: verbose models over-deliberate on routine turns ([Turn Bench](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1775940834879)); ESFP: Qwen Thinking 7th of 8.
- **Claude Opus 4.6:** top of SecureExec (0.69) but 0.00 on MetaBound's false-premise task ([MetaBound](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1774955644602)) and 0.00 overall on RIAC's binary pass scheme ([RIAC](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1775848234859)).
- **Gemini 3.1 Pro:** #1 on LearningBench (0.85), EphLangBench (0.731) and Metaproteus (0.970), last on ESFP (0.399) ([ESFP](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1776173871950)). GAUGE: Gemini 2.5 Pro has the best calibration but never abstains.
- **Small models win metacognition/social constructs:** Claude Haiku 4.5 #1 on MEDLEY and the best abstainer on GAUGE; Gemma 4 26B A4B #1 on AdvisorBench; Gemma-3-27B tied #1 on ESFP; GLM-5 #1 on MetaBound. Repeated message: these abilities do not track scale.
- **LLM judges for social meaning:** HedgeDecode deliberately avoids them in Families A/B ("circularity") ([HedgeDecode](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/hedgedecode)); ESFP's core metric SCD is a judge-panel label ([ESFP](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/new-writeup-1776173871950)). Both won.

## UNVERIFIED
- **Metacognition winner 2 of 2:** host thread 724918 names Qing Zhou's Metacognitive Calibration Benchmark; the writeup API attaches prize 911 to MetaBound (Ahmed A.). Not resolved; both reported. No later correction found in the topics on disk.
- Grand-prize entries carry no award object in the API; their status rests on the host thread only (consistent with the bank).
- All model scores are team-claimed (writeup) and were not re-run. Model names (GPT-5.4, Claude Opus 4.6, Gemini 3.1 Pro, GLM-5, Gemma 4) are as written by teams.
- RIAC: which model/pool the "48.7% failure at L=1" curve aggregates is unclear and looks inconsistent with the per-model table (e.g. Gemini 2.5 Flash 930/936 on number repetition).
- MetaBound: the 0.00 false-premise score for Claude Opus 4.6 and GPT-5.4 nano is a strong single-run claim.
- ESFP describes a "three-LLM judge panel" but names two judge models.
- GrammarGym and ABC give no numeric scores in text; any numbers would have to come from their benchmark pages or figures (not fetched).
- AdvisorBench: full 16-model list only in figures; the lowest equity score (0.682, GPT-5.4 nano) is the lowest named in text, not necessarily the minimum.
- Whether every winner used the Kaggle Benchmarks SDK (required) is inferred from attached benchmark links where the text does not say so; Qing Zhou's writeup links only a notebook.
