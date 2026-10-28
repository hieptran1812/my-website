# Kaggle LLM competitions — verified writeup bank, part 2

Scope: math reasoning (AIMO 1/2/3), misconception mining (Eedi, MAP), ARC Prize 2024/2025/2026, Konwinski Prize. Built 2026-10-27 for the "Learn from Kaggle: LLM competitions" series. Companion files: `kaggle-rl-writeup-bank-part1.md`, `kaggle-rl-writeup-bank-part2.md`.

## How this bank was verified

- Competition facts (dates, team/competitor/submission counts, runtime limits, prize totals, leaderboard split, team-size caps) come from Kaggle's internal JSON API via `.claude/scripts/kaggle-api.sh`: `CompetitionService/GetCompetition` (metadata), `PageService/ListPages` (Overview, Evaluation, Prizes, Code Requirements, Data, Rules text), `LeaderboardService/GetLeaderboard` (private and public rows joined to the teams array; `solutionWriteUpUrl` gives the writeup link). Raw JSON saved under `/private/tmp/claude-501/kaggle-bank-p2/`.
- Writeups and threads: `DiscussionsService/GetTopicListByForumId` (sortBy TOP, 8 pages per forum) to rank topics by votes, then `GetForumTopicById` (writeup body + top comments) for ~120 topics. Every technical claim in a "place" subsection is paraphrased from that team's own writeup or paper.
- Off-Kaggle primary sources: Hugging Face Numina blog; ARC Prize 2024 technical report (arXiv 2412.04604) and 2025 results blog (arcprize.org/blog/arc-prize-2025-results-analysis); ARChitects 2024 paper PDF (github.com/da-fr/arc-prize-2024); arXiv abstracts/HTML for BARC (2411.02272) and TTT (2411.07279); PR Newswire for AIMO1 award.
- Kaggle recap posts (registrations, countries) are quoted separately from API counts because they differ slightly.
- After finalization Kaggle zeroes the public score in the API for some competitions (AIMO1/2/3, Konwinski); public scores there are marked as not retrievable.
- ARC Prize 2026 is still running (deadline 2026-11-02): all its numbers are a public-leaderboard snapshot from 2026-10-27.
- Anything not confirmed from a primary source is listed in each section's UNVERIFIED block. Derived numbers (e.g. Konwinski correct/wrong counts recomputed from the metric) are labelled "derived".

## Summary table

| # | Slug | Title | Dates | Teams (API) | Prize pool | Metric |
|---|---|---|---|---|---|---|
| 1 | ai-mathematical-olympiad-prize | AIMO Progress Prize 1 | 2024-04-01 → 2024-06-27 | 1,161 | $1,048,576 fund ($253,952 to top 5; grand prize rolled over) + $10k early sharing | Accuracy on 50 private problems, integer answers mod 1000 |
| 2 | ai-mathematical-olympiad-progress-prize-2 | AIMO Progress Prize 2 (stub) | 2024-10-17 → 2025-04-01 | 2,212 | $2,117,152 fund | Accuracy /50, mod 1000 |
| 3 | ai-mathematical-olympiad-progress-prize-3 | AIMO Progress Prize 3 | 2025-11-20 → 2026-04-15 | 4,138 (recap 3,450) | $2,207,152 fund + $110k extra prizes | Two-run penalized accuracy /50, answers 0–99999 |
| 4 | eedi-mining-misconceptions-in-mathematics | Eedi — Mining Misconceptions in Mathematics | 2024-09-12 → 2024-12-12 | 1,446 | $55,000 (LB + CPU efficiency track) | MAP@25 |
| 5 | map-charting-student-math-misunderstandings | MAP — Charting Student Math Misunderstandings | 2025-07-10 → 2025-10-15 | 1,857 | $55,000 | MAP@3 |
| 6 | arc-prize-2024 | ARC Prize 2024 (ARC-AGI-1) | 2024-06-11 → 2024-11-10 | 1,427 | $1.1M advertised; $125k progress paid; $600k grand prize unclaimed | % exact grids, 2 attempts (public = private) |
| 7 | arc-prize-2025 | ARC Prize 2025 (ARC-AGI-2) | 2025-03-26 → 2025-11-03 | 1,455 | $1.0M advertised; $125k progress; $700k grand prize unclaimed | % exact grids, 2 attempts, 50/50 split |
| 8 | arc-prize-2026-arc-agi-2 / arc-prize-2026-arc-agi-3 / arc-prize-2026-paper-track | ARC Prize 2026 (ongoing) | 2026-03-25 → 2026-11-02 (papers 11-09; winners 12-04) | 3,425 / 4,905 / 328 | $700k / $850k / $450k (much conditional) | ARC-AGI-2 % exact grids; ARC-AGI-3 action-efficiency vs humans; paper rubric |
| 9 | konwinski-prize | Konwinski Prize | 2024-12-11 → 2025-03-12 (forecast scoring to Jul 2025) | 617 (recap 471) | $1,225,000 ($100k paid to top 5; thresholds and $775k grand prize unclaimed) | (correct − wrong − skipped/10⁴)/total on ~120 post-deadline issues |

Winners at a glance: AIMO1 Numina 29/50 · AIMO2 NemoSkills 34/50 · AIMO3 Exalted Joseph 44/50 · Eedi MTH 101 0.639 · MAP tascj 0.949 · ARC 2024 the ARChitects 53.5% (MindsAI 55.5% ineligible) · ARC 2025 NVARC 24.03% · ARC 2026 TBD · K Prize Eduardo Rocha de Andrade 0.058 (9/120 resolved).

## 1. AI Mathematical Olympiad — Progress Prize 1 (2024)

### Facts
- URL: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize (slug `ai-mathematical-olympiad-prize`, competition id 73231)
- Dates: start 2024-04-01; entry and team-merge deadline 2024-06-20; final submission 2024-06-27 23:59 UTC. Private rerun finished and scores posted 2024-07-03 (host Ryan Holbrook, https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/516823). Prize presented at IMO 2024 in Bath (PR Newswire, 2024-07-22).
- Task: 110 new problems written by an international team "similar to an intermediate-level high school math challenge", text-only LaTeX, difficulty "a bit easier than AIME, between AIME and AMC 12". Answer is a non-negative integer reported modulo 1000 (0–999). 10 train problems, 50 public, 50 private (disjoint). During the submission period only the 50 public problems are served; the private 50 are served only at the final rerun. Problems arrive one at a time in random order through a Python evaluation API.
- Metric: accuracy (count of exact matches, shown as x/50). Baseline stated by host: Gemma 7B = 3/50.
- Prize: total fund $1,048,576 (2^20). Top 5: $131,072 / $65,536 / $32,768 / $16,384 / $8,192 (quartered if a team does not beat 3/50 on both sets). Overall Progress Prize (at least $794,624) for the first team scoring >= 47/50 on both public and private: NOT awarded, remainder rolled over to the next AIMO competition. Early Sharing Prize $10,000 for the first public notebook >= 20/50 before 2024-04-22 (the community credits Abdur Rafae's "Improved Code Interpretation" notebook as the early-sharing winner — 4th-place writeup calls it "His Early Sharing Prize is well-deserved").
- Teams: 1,161 teams / 1,401 competitors / 1,831 submissions (Kaggle API). Max 2 submissions/day. Press release: 16,104 registrations from 81 countries.
- Hardware / runtime: notebook-only; CPU or GPU notebook <= 9 h; internet off; freely available external data and pre-trained models allowed. Accelerator per Hugging Face blog: either 1x P100 or 2x T4 (https://huggingface.co/blog/winning-aimo-progress-prize). Runtime obfuscated by up to 30 min. Winner license: open source (OSI).
- Model-date rule: community post (Tong Hui Kang, https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/490691) quotes "only AI models and tools that are open source and were released prior to 23 February 2024" — not found in the current rules text pulled from the API, so treat as UNVERIFIED for the final rules.

### Top placements (private, out of 50)
| Rank | Team | Private | Approach (one line) | Writeup |
|---|---|---|---|---|
| 1 | Numina (Numina + Hugging Face) | 29 | Two-stage SFT of DeepSeekMath-7B-Base (CoT then tool-integrated reasoning), 8-bit GPTQ, SC-TIR decoding N=48, depth 4 | https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/numina-numina-1st-place-solution |
| 2 | CMU_MATH | 22 | Fine-tuned DeepSeekMath-7B-RL policy + separate 7B outcome reward model, ORM-weighted majority vote | https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/cmu-math-2nd-place-solution-all-code-and-datasets- |
| 3 | after exams | 21 | Un-finetuned DeepSeekMath-7B-RL in vLLM, 120–160 samples, penalty on small numbers and numbers from the statement | https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/after-exams-3rd-place-solution |
| 4 | codeinter | 21 | DeepSeekMath-7B-RL on both T4s in parallel, adaptive TimeManager, fewer attempts on `[asy]` geometry | https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/codeinter-4th-place-solution |
| 5 | Conor #2 | 20 | no writeup found | — |
Ranks 5–12 all scored 20. Public scores were wiped from the API after finalization (all show 0), so public ranks for these teams cannot be verified from Kaggle.

### 1st place — Numina (Jia Li, Yann Fleureau, Guillaume Lample, Stan Polu, Hélène Evain; with Lewis Tunstall, Ed Beeching of Hugging Face)
Sources: Kaggle writeup above; https://huggingface.co/blog/winning-aimo-progress-prize; notebook https://www.kaggle.com/code/lewtun/numina-1st-place-solution; model https://huggingface.co/AI-MO/NuminaMath-7B-TIR (+ `-GPTQ` 8-bit).
- Base model: DeepSeekMath-Base 7B. Larger bases (InternLM-20B, CodeLlama-33B, Mixtral-8x7B) tried and dropped: DeepSeek's math continued pretraining was hard to beat and big models were slow with "mysterious timeouts".
- Recipe follows MuMath-Code (arXiv 2405.07551), two stages, full fine-tuning (no LoRA), TRL SFTTrainer with packing, gradient checkpointing, DeepSpeed ZeRO-3, 1 node of 8x H100, ~10 h total.
  - Stage 1: large diverse natural-language math problems with CoT-templated solutions ("several hundred thousand problems" per HF blog). LR 2e-5, batch 32, block 2048, 3 epochs, cosine, warmup 0.1.
  - Stage 2: synthetic tool-integrated reasoning (ToRA format, arXiv 2309.17452): GPT-4 prompted to write rationale -> Python -> execution output -> ... ; ~60k problems with numerical outputs (HF blog). LR 2e-5, batch 32, block 1024, 4 epochs.
- Trajectory on public LB: Stage-1-only CoT model 8/50 (maj@32) -> MMOS single-program data 16/50 (maj@32) -> multi-step TIR data à la DeepSeekMath-Instruct/RL gave the big jump. Integer-only Stage-2 filtering was slightly worse than keeping all solution types.
- Inference — SC-TIR (self-consistency with tool-integrated reasoning): copy the prompt N times as a vLLM batch; sample until a complete Python block; execute each block and append output (tracebacks included); repeat M rounds; prune traces with broken outputs; majority vote on the post-processed answers. Final: N=48, M=4; going larger did not help. Motivation: problems arrive in random order so early-stop/time-budget tricks gave high variance; SC-TIR reduced variance on internal evals and LB.
- Quantization: 8-bit GPTQ (AutoGPTQ, calibrated on training data) because a 16-bit 7B needs ~32 GB just for weights on 2x T4 and model upload to Kaggle was twice as fast; small accuracy drop offset by more samples. T4s lack bf16, FlashAttention-2, and torch.compile support.
- Validation ("good validation sets are all you need"): AMC 2022–23 integer-answer problems (83, designed to mimic the private set; models solved ~60–65%), AIME 22/23/24, MATH level 4 and level 5 integer-answer subsets (~750 each). Small sets run with 5–10 seeds (1–3% variance). Released as `AI-MO/aimo-validation-*` on HF.
- Did not make the final: on-policy KTO (sample 4 completions per Stage-2 prompt, label by answer correctness, KTO on SFT model) gave a few points internally and 27/50 public, but they ran out of time to apply it to the final SFT model; RLOO RL with code-execution reward on DeepSeekMath-7B-RL gave no significant gain and was slow; static KV cache + torch.compile gave 2–3x on H100 but broke on T4; model merging (DARE, TIES, WARP via mergekit) regressed or was unfinished.

### 2nd place — CMU_MATH (Zhiqing Sun et al.)
Sources: Kaggle writeup above; code+data https://github.com/AIMO-CMU-MATH/CMU_MATH-AIMO; notebook https://www.kaggle.com/code/edwardsun0909/aimo-2nd-place-solution; HF `AIMO-CMU-MATH/policy-model`, `AIMO-CMU-MATH/reward-model`.
- Two fine-tuned DeepSeekMath-7B-RL models: a policy (generator) and an outcome reward model (ORM), one T4 each (policy in vLLM, ORM in HF transformers).
- Policy data: AMC + AIME + Odyssey-Math, integer-answer problems only, multiple-choice options removed; GPT-4 few-shot sampled code solutions, keep only correct ones. 3 epochs, LR 2e-5.
- Reward data built for diversity and label balance: (1) MATH problems sampled from weight-interpolated models M_alpha = alpha*M_RL + (1-alpha)*M_Base for alpha in {0.3..1.0} (8 models, 2 samples each) — interpolation gives a spread of skill levels so each problem gets both right and wrong solutions; (2) AMC/AIME/Odyssey sampled from the policy fine-tuned for 0,1,2,3 epochs (4 models x 12 samples). Filter: drop wrong non-integer answers; keep positive:negative 1:1 per problem.
- Inference: 42 samples per problem, T=0.9, 2 rounds; if code output is not an integer, append feedback ("The code output is not an integer...") and let the model retry.
- Weighted majority vote: ORM scores problem+code (not the prose); weight of answer a = N x geometric mean of its N scores (geometric mean resists a single false-positive high score); answer 0 penalized (score / 10) because the policy over-produces 0.
- Effect on the 10 train problems: base 1/10 -> fine-tuned policy 2/10 -> + ORM 4/10.
- Did not help: deduplicating policy data, adding GPT-4's wrong solutions, keeping all reward samples, positive:negative ratios between 1:2 and 2:1.

### 3rd place — after exams (David Dinucu-Jianu et al.)
Source: Kaggle writeup above; notebook https://www.kaggle.com/code/davidd2/3rd-place-solution-afterexams.
- No fine-tuning: DeepSeekMath-7B-RL, adapted from Abdur Rafae's public notebook; vLLM; 120–160 candidates per problem in one batch; >6 code-execution iterations helped; FP16 KV cache improved score; batch-parallel code execution.
- Forced answer: if generation ends without `\boxed{}`, append "The final answer is \boxed{" and generate a few tokens.
- Scoring rule: penalize (as a percentage of total generations) answers that are small (<10, often code errors) and numbers that appear in the problem statement — "a significant boost".
- Tried but not selected: bf16 on T4 by patching vLLM to compute in fp32 and cast (github.com/RD211/vllm-bf16) — best validation but unstable, never above 25 public; a logits-processor trick that injects code output tokens directly into generation to avoid repeated vLLM calls. Chose an older version for the final because of tie risk.

### 4th place — codeinter (Zonglin Han et al.)
- DeepSeekMath-7B-RL, T=0.9, top_p 1.0, 2048 max tokens; one model copy per T4 in parallel raised samples per problem from 21 to 30–40. Local CV: latest 50 AMC 12A problems from a 71k AoPS dump (no `[asy]`), correlated with public LB. `TimeManager` class shrinks attempts when behind schedule; only 3 attempts for `[asy]` geometry problems.

### Key threads and notebooks
- Abdur Rafae, "Improved Code Interpretation" — https://www.kaggle.com/code/abdurrafae/improved-code-interpretation — the public baseline that 1st, 3rd and 4th all credit; DeepSeekMath-7B-RL + code execution + majority vote.
- Chan Kha Vu, "Solutions to 10 samples in train set" (114 votes) — https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/490640 — found a wrong train label; organizers fixed it; human CoT as a guide to what the model must do.
- Tong Hui Kang, "My prediction of how the top solutions will look like" — https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/490691 — prediction of code-tool + voting designs.
- Khoi Nguyen, "If someone manage to solve 47/50 ... would they really want to turn the solution in for ~800k?" (44 votes) — https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/488441 — the incentive question around the grand prize.
- Geremie Yeo, "Strange Behavior on the LB [current 3rd place user deleted account]" — https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/511414 — a 27/50 account vanished mid-competition; commenters suspected a multi-account.
- ISAKA Tsuyoshi, "[Public LB20 / Private FAIL]" — https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/isaka-tsuyoshi-public-lb20-private-fail-write-up — a strict answer parser alone moved LB 14 -> 19; then the private rerun failed (runtime robustness matters).
- Abdullah Meda, >15k MATH/GSM8K samples annotated with code — https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/507796.

### Shake-up
Final-score thread comments call it a big shake-up ("Yikes talk about shakeup! Random seed not in my favour", discussion/516823). Numina's validation discipline is the community's explanation of the 29 vs 22 gap. Exact public ranks/scores are not retrievable from the API after finalization — UNVERIFIED.

### Transferable lessons
1. Tool-integrated reasoning (interleave prose and executed Python) beat pure CoT by a wide margin on integer-answer olympiad math (8 -> 16 -> 29 trajectory) — Numina writeup.
2. Build several small, hard, distribution-matched validation sets and run them with multiple seeds; do not trust a 50-problem LB — Numina writeup / HF blog.
3. A learned verifier (ORM) on top of sampling picks better answers than plain majority vote (2/10 -> 4/10 on the train problems); geometric-mean weighting resists single false positives — CMU_MATH writeup.
4. Cheap priors on wrong answers (penalize 0, small numbers, numbers in the statement) are worth points — CMU_MATH and after-exams writeups.
5. Budget-aware inference (fixed width/depth, time managers) matters when problems arrive in random order; variance reduction is a goal in itself — Numina and codeinter writeups.
6. Old GPUs shape the design: 8-bit quantization, no bf16, no FlashAttention — Numina writeup.

### UNVERIFIED
- Public LB scores of the top teams (API zeroed). Exact size of Numina's Stage 1 CoT set (blog says only "several hundred thousand"; the later NuminaMath-CoT release on HF is ~860k — not re-checked here). The Feb-23-2024 model cutoff rule (community quote only). Early Sharing Prize recipient (Abdur Rafae per community; no host announcement fetched).

## 2. AI Mathematical Olympiad — Progress Prize 2 (2024–25) — timeline stub only (covered by an existing post)
- URL https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2 ; 2024-10-17 to 2025-04-01; 2,212 teams / 2,672 competitors; GPU notebook <= 5 h (API `maxGpuRuntimeMinutes` 300), internet off, 1 submission/day, team size up to 7.
- Fund $2,117,152: 1st $262,144 / 2nd $131,072 / 3rd $65,536 / 4th $32,768 / 5th $16,384; Early Sharing $20,000 (first public notebook >= 20/50 before 2025-01-01); Overall Progress Prize (>= 47/50 both sets, >= $1,589,248) NOT awarded, rolled to AIMO3. Answers still integer mod 1000.
- Private top 3 (out of 50): 1 NemoSkills (NVIDIA) 34; 2 imagination-research 31; 3 Aliev 30; 4 sravn 29; 5 usernam 29. Writeups: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills , .../imagination-research-2nd-place-solution-team-imagi , .../aliev-3rd-place-solution-report
- AIMO3 overview text: commercial models solved 50/50 AIMO2 public problems "with sufficient compute" in an OpenAI x AIMO eval (March 2025) versus the best Kaggle score of 34/50 (https://aimoprize.com/updates/2025-09-05-the-gap-is-shrinking).
- Hardware for AIMO2 (4x L4 per community memory) not re-verified from the API pages — UNVERIFIED.

## 3. AI Mathematical Olympiad — Progress Prize 3 (2025–26)

### Facts
- URL: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3 (id 118448). Status: FINISHED. Start 2025-11-20; entry/merge deadline 2026-04-08; final submission 2026-04-15 23:59 UTC. Kaggle recap posted 2026-05-21 (discussion/702188); extra prizes announced 2026-06-15 (discussion/708484); main winners announced 2026-06-26 after a longer compliance check (discussion/714435).
- Task: 110 original problems (algebra, combinatorics, geometry, number theory) "from National Olympiad level all the way up to IMO standard", designed to be "AI hard" against current open LLMs. 10 reference problems with a PDF of full solutions, 50 public, 50 private. NEW: answers are integers 0–99999 (5 digits), any modulo is stated explicitly in the problem (AIMO1/2 used implicit mod 1000).
- Metric: public LB = number correct out of 50. Private = notebook run TWICE on the 50 private problems; per problem 1 if both runs correct, 0.5 if one, 0 if neither ("penalized accuracy", metric named "AIMO 3 Multirun-Accuracy"). A submission must finish both reruns to be scored. Public order random; private uses a fixed random order.
- Prize fund $2,207,152 (API: "$507,904 in Intermediate Prizes, $110,000 in bonus prizes, and a $1,589,248 Grand Prize contingent on achieving 47/50"). Top 5: $262,144 / $131,072 / $65,536 / $32,768 / $16,384. Grand prize (>= 47/50 public AND private): NOT awarded (top private score was 44). New extra prizes ($110k): Longest Leader $20k, Hard Problem $30k, Math Corpus $30k, 2 x Writeup $15k.
- Teams: 3,450 teams / 4,065 participants / 3,450 submissions per host recap (API team count field shows 4,138), 22,980 registrations, 113 countries.
- Hardware: Kaggle added H100 machines usable only for AIMO3 notebooks, internet must be off (Overview "Upgraded Kaggle Hardware"). Code page: GPU notebook <= 5 h, CPU <= 9 h (API fields show 600 / 1080 min, likely covering the double private rerun — UNVERIFIED interpretation). Max team size 20. 1 submission/day.
- Rule changes vs AIMO1/2: open-weight LLMs allowed if "reasonably accessible to all" and of "minimal cost" (rules section 6, Reasonableness Standard); winners must release on HuggingFace and pass stricter reproducibility checks (winners post). Compute partnerships: Fields Model Initiative (128 H100s, applications) and Thinking Machines' Tinker (up to $400 API credit).

### Top placements (private, two-run penalized accuracy out of 50)
| Rank | Team | Private | Approach | Writeup |
|---|---|---|---|---|
| 1 | Exalted Joseph | 44.0 | gpt-oss-120b on 1x H100, no training; fork of the Parthenos/Andreas notebook: page-cache preload, prefix caching, 8-bit KV, entropy-weighted vote, persistent Jupyter sandbox, adaptive time | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/1st-place-solution-for-the-aimo3-competition |
| 2 | varianceofx | 44.0 | Same base, 7 targeted edits: answer-range prompt, tail-window entropy, vote-share-dominant scoring, uncatchable-leader early stop | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/2nd-place-solution-ai-mathematical-olympiad-prog |
| 3 | SKobayak | 44.0 | Baseline by ZaynYu, prompt rewritten from LLM-assisted error analysis of 1,120 outputs | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/3rd-place-solution-for-the-aimo3-competition |
| 4 | taku | 44.0 | no writeup; NOT among announced prize winners | — |
| 5 | TAMU-TACO | 43.5 | (prize winner) writeup not read | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3-tamu-taco-solution |
| 7 | yemao ye | 43.5 | (prize winner) writeup not read | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/7th-place-solution-for-the-aimo3-competition |
Announced winners are ranks 1, 2, 3, 5, 7 (discussion/714435): ranks 4 (taku) and 6 (Man Penguin) did not receive prizes; the post cites stricter compliance (HF release) but does not name reasons — treat reasons as UNVERIFIED. Four teams tied at 44.0; tie order on Kaggle is by earlier submission. Public scores are zeroed in the API.

Extra prizes (discussion/708484): Math Corpus — Yi-Chia Chen, CrystalMath (Contrast-Augmented Verification filter for hard problems; host spot-check ~50% residual issue rate on hard items vs ~80% in comparable sets), runner-up NGUYEN AstralMath (~431k tool-use traces from 7 models + 50-problem AstralBench). Write-Up — Geremie Yeo & Chan Kha Vu "A Collection of Delightful Side Quests" and Natapong Nitarach "Model Capability Dominates" ($15k each); runner-up team Hail Mary "A Practitioner's Plateau on gpt-oss-120b-MXFP4". Hardest Problem — Md. Hamid Hosen (only submission to solve problem "ACUTES" in both runs, private 43.5); problem "ROLLER" solved by nobody in both runs. Longest Leader — OSSMath (yiyangzheng, shreyansh01m). Follow-up: invite-only "AIMO Proof Pilot" (7 teams, proofs graded by humans, only fully open models like OLMo; 3 nodes of 8x H200 per team from Japan's NII).

### The shared backbone (what everyone forked)
Andreas Bisiadis, "GPT-OSS-120B (with tools)" — notebook https://www.kaggle.com/code/andreasbis/aimo-3-gpt-oss-120b-with-tools ; writeup https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/gpt-oss-120b-with-tools-technical-writeup . >1,000 upvotes; host calls it the foundation of many top notebooks and invited him to the Proof Pilot. His own selected version: public 41, private 41.5 (place 546).
- gpt-oss-120b in vLLM 0.11.2 on one H100, Harmony format, `ReasoningEffort.HIGH`, 65,536 context, FP8 E4M3 KV cache, prefix caching, async scheduling.
- Up to 8 parallel attempts per problem, seeds (42+i)^2, T=1.0, min_p=0.02; Python tool backed by persistent Jupyter kernels (pool overprovisioned 2x, port retry, output truncation, CPU isolation).
- Early stop when 4 attempts agree. Answer = `\boxed{}` integer in [0, 99999] (plus a narrow "final answer is N" fallback). Selection: inverse mean token entropy (from top logprobs) summed per answer. No training at all.
- Finding: inference-system reliability beat extra reasoning machinery; strong notebooks all used pass@8, short prompts, FP8 KV, prefix caching, long context.
Parthenos's derivative ("44/50 let me over cook", https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook) added a 5-step UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY system prompt; 1st and 2nd both build on it.

### 1st place — Exalted Joseph (private 44.0)
Source: writeup above (largely descriptive, few numbers). "Based on the work of Parthenos."
- Model gpt-oss-120b (~117B total, ~12B active per the writeup — note the arXiv paper below says 5.1B active; the writeup's number is its own), single H100 80 GB, no fine-tuning.
- Systems: OS-level preload (read the weight files once to put them in Linux page cache before init), prefix caching across parallel samples, 8-bit KV cache (longer context, more parallel generations).
- Prompts: tried Gemini-generated prompts and community templates, ended up keeping the original notebook prompt; cites https://www.kaggle.com/code/datasciencegrad/aimo-3-42-50-stable-lb-possible-43-luck .
- Entropy-weighted self-consistency; persistent Jupyter kernels whose outputs feed back into generation; adaptive scheduling: baseline budget per problem, leftover time redistributed, near the timeout reduce sampling depth and return the best valid partial answer.
- Stated lesson: careful inference design and prompting beat naive scaling. No ablation numbers given — the bank cannot say which piece mattered.

### 2nd place — varianceofx (private 44.0)
Source: writeup above; notebook https://www.kaggle.com/code/varianceofx/let-me-improve-this-cooking ; HF https://huggingface.co/varianceofx/aimo3-2nd-place-solver-adapted. Pure inference, no training. 8 attempts, 16 Jupyter kernels, FP8 KV, 65,536 context, ~300–400 s per problem, ~4–5 h total on 1x H100.
Seven changes over the Parthenos base:
1. System prompt with a mandatory answer-range block ("MUST be a non-negative integer in [0, 99999]... if your raw answer exceeds 99999 the problem is asking for something modular"), a 4-question pre-boxing checklist, and a pitfalls list. Called "the highest-leverage change": the model often solved the problem and then boxed the raw large integer.
2. Library hint condensed from ~200 tokens to one line (math, numpy, sympy, mpmath at 64 digits, itertools, collections).
3. Entropy measured only on the last 256 tokens (the commit phase) instead of the whole stream.
4. Answer extraction: only `\boxed{}` (dropped the "final answer is N" fallback, which fired on intermediate steps during streaming), comma/whitespace handling, last match wins.
5. Scoring: score = 2.0 x vote_share + 0.3 x (1 / median entropy) + 0.5 x consensus bonus, replacing pure sum of 1/entropy, which a single very confident wrong attempt could hijack.
6. Median instead of mean entropy per answer.
7. "Uncatchable leader" early stop: stop when second place + remaining attempts cannot exceed the leader (in addition to the 4-agree rule).
Did not help: lower temperature (less diversity; kept T=1.0, min_p=0.02); more than 8 attempts (time budget); fine-tuning (not attempted, inference gave more per hour); different system prompts per attempt (no gain).

### 3rd place — SKobayak (private 44.0)
Source: writeup above (slides attached on Kaggle). Base: ZaynYu "40/50 gpt-oss-120b TIR DynamicTime KernelPool" (https://www.kaggle.com/code/zaynyu/40-50-gpt-oss-120b-tir-dynamictime-kernelpool). gpt-oss-120b as-is + Python tool, 8 parallel solvers, majority vote, baseline early stopping.
- Offline validation instead of LB-chasing: 1,120 full outputs = 32 runs x 10 reference problems + 16 runs x 50 external hard problems shared by participants (discussion/672643). Collected into `llm_history.csv`; a stronger LLM categorized failures, fed with examples of gpt-oss's own correct answers so its advice stayed within gpt-oss's reach.
- Mental model of the public LB: ~35 always-solved, ~10 coin-flip, ~5 unsolvable problems — so LB-tuning overfits to the 10 unstable ones.
- Prompt changes from the error analysis: use Python for verification; extract and check every explicit constraint; narrow the search space and estimate complexity before loops (brute-force loops were timing out). Full system/tool/preference prompts are in the writeup.
- Did not work: reasoning in Japanese; forced draft-then-finalize two-step self-correction; complexity-based consistency prompts; 16 instead of 8 solvers (early finishers were less precise; net more errors).

### Key threads, notebooks and writeups
- "Model Capability Dominates" (Natapong Nitarach, writeup prize; arXiv 2603.27844; https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/model-capability-dominates): diverse per-voter prompts never beat baseline across 3 models and 23+ experiments; N_eff = N / (1 + (N-1) rho); capability gap ~8 points vs +-2 from prompts; best 42, 39.3 mean over 21 runs; gap to pass@20 ~45.5 is selection loss, 82–86% of unrecovered cases lose by one vote; public-private shake-up of 4.5 points within 3 sigma of run-to-run variance; private 40.5.
- "A collection of delightful side quests" (Chan Kha Vu & Geremie Yeo, writeup prize; https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3-a-collection-of-delightful-side-quests): FP8/NVFP4 quants of Nemotron-Cascade-2-30B-A3B, vLLM fork making prefix caching + speculative decoding coexist on hybrid Mamba models ("mamba cache poisoning" fixed with per-request ghost states), EAGLE-3 speculator with 3.55 mean accepted tokens at K=7.
- Tong Hui Kang, "[Math Corpus Prize] Annotated Solution Traces" — https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/672528 (commended).
- ippeiogawa, "Blackboard and Fork" — https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3test — shared tool-call ring buffer across parallel rollouts + forked self-critique (commended).
- IceBeam777, "Recursive Self Aggregation with Difficulty Aware Scheduler" — https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/recursive-self-aggregation-with-di — public mean 38 -> 41 with lower variance (host).
- Utkarsh Sharma, "Every Improvement Failed" — https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/every-improvement-failed-aimo3 — negative results (not read in detail).
- Victor Ogobi, "What happened to the Hard Problem, Math Corpus and Writeup Prizes?" — discussion/705484 — community pressure on prize timing.

### Shake-up
Public scores clustered around 40–44 with run-to-run noise of several points; the two-run private metric was designed to reduce luck. Nitarach measured a 4.5-point public/private swing on an identical config. Andreas's base notebook: public 41 -> private 41.5 (place 546) while minor forks reached 44 — the top of the board is separated by noise-sized margins. Host: "the main AIMO3 competition ended up being dominated by one notebook."

### Transferable lessons
1. When one strong open model dominates and is hard to train, the competition becomes inference engineering: KV precision, prefix caching, sandbox reliability, time allocation — Andreas writeup; host winners post (discussion/714435).
2. Fix output-contract failures first: an answer-range reminder recovered problems the model had already solved — 2nd place writeup.
3. Use confidence (entropy) as a tiebreaker, not the primary vote; measure it where the model commits (last 256 tokens) — 2nd place writeup.
4. Validate offline on many runs of many hard problems and read failures, rather than chasing a 50-problem LB — 3rd place writeup; Nitarach writeup.
5. Prompt diversity does not decorrelate errors beyond what T=1.0 sampling already does; remaining loss is selection — Nitarach writeup / arXiv 2603.27844.
6. Extra prizes for datasets and writeups produced reusable science even when the main board was a notebook monoculture — discussion/708484.

### UNVERIFIED
- Why ranks 4 and 6 were not among winners. Public LB scores of the top 3. Contents of the TAMU-TACO (5th) and yemao ye (7th) writeups (not fetched). Meaning of the 600-min GPU API field vs the 5 h code rule. Active-parameter count for gpt-oss-120b differs between sources (12B in 1st writeup vs 5.1B in Nitarach) — use the model card when writing.

## 4. Eedi — Mining Misconceptions in Mathematics (2024)

### Facts
- URL: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics (id 82695). Hosts: Eedi, Vanderbilt University, The Learning Agency Lab.
- Dates: 2024-09-12 to 2024-12-12 (API). Recap posted 2024-12-20.
- Task: a Diagnostic Question is a 4-option math MCQ (one correct, three distractors, each written to catch a specific misconception). For every (question, wrong answer) pair, rank misconceptions from a pool of ~2,587 ("2.5k+" per 1st place) by how well they explain the wrong answer. The model should generalize to misconceptions never seen in train. Text came from images via human-in-the-loop OCR. Fields: Question/Construct/Subject text and ids, answer texts, per-option MisconceptionId; `misconception_mapping.csv`.
- Data: train ~1.8k questions (1st place: "1.8k examples"); hidden test ~1,000 questions per data page (participants estimated ~685, discussion/550875). Public LB = 28% of test (API `leaderboardPercentage`).
- Metric: MAP@25, one correct label per row.
- Prizes: $55,000 total. Leaderboard $12k / $8k / $5k / $5k; Efficiency (CPU-only) $12k / $8k / $5k, efficiency score = MAP@25 / (Base - maxMAP@25) + RuntimeSeconds/32400 (lower is better).
- Teams: 1,446 teams / 1,902 competitors / 41,411 submissions (API); recap: 1,471 teams, 1,964 participants, 8,047 registrations, 82 countries.
- Runtime: CPU or GPU notebook <= 9 h, internet off, public pre-trained models allowed. GPU was Kaggle's 2x T4 (implied by top thread "Qwen2.5-72B & Llama 3.3-70B on 2xT4", discussion/550223, and both top writeups running 72B quantized models).

### Top placements
| Rank | Team | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|
| 1 | MTH 101 (Raja Biswas) | 0.63881 | 0.64379 (6) | Qwen2.5-14B retriever -> 14B pointwise -> 32B pointwise -> 72B listwise rerank; Claude-generated grouped synthetic MCQs; distilled CoT | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution |
| 2 | CQYR | 0.60480 | 0.62947 (8) | Linq-Embed-Mistral (ArcFace) + 2x Qwen2.5-14B retrievers; sliding-window listwise rerank (14B on 8–17, then 72B + Llama-3.3-70B on 1–10) | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution |
| 3 | Waseda Pochi (Yi-Chia Chen) | 0.60248 | 0.67081 (1) | Public Qwen-14B embedders + Qwen-32B-AWQ reranker (6 LoRAs) + LB-probed "unseen misconception" boost | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/waseda-pochi-3rd-place-solution-with-magic-boost |
| 4 | takoi & charmq & kami | 0.60044 | 0.65568 (3) | writeup not read | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/takoi-charmq-kami-4th-place-solution |
| 5 | Ebi & ktr | 0.59765 | 0.65374 (4) | writeup not read | https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/ebi-ktr-5th-place-solution |
Efficiency track: 1st Dipam Chakraborty (model merging, https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/dipam-chakraborty-efficiency-track), 2nd Jack (Japan) (discussion/552686), 3rd Ulrich G. (public model + simple post-processing).

### 1st place — MTH 101 (Raja Biswas, solo)
Sources: writeup above; summary https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551402 ; code https://github.com/rbiswasfc/eedi-mining-misconceptions ; inference https://www.kaggle.com/code/conjuring92/eedi-a2-pipeline ; data https://www.kaggle.com/datasets/conjuring92/eedi-mcq-dataset .
- Framing: retrieve-and-rerank. LLMs are good at solving math but bad at counterfactual reasoning (play a novice who makes a specific error, or a tutor naming it); the pool has near-duplicate misconceptions; must generalize to unseen ones.
- Pipeline: retriever keeps top 32 + up to 32 more above a dynamic threshold (top score - constant) -> fine-tuned Qwen2.5-14B pointwise ranker picks top 8 -> Qwen2.5-32B pointwise narrows to top 5 -> Qwen2.5-72B listwise orders top 5. Final 25 = 72B ranks 1–5, 32B ranks 6–8, 14B ranks 9–25.
- Synthetic data (key): cluster closely related misconceptions using co-occurrence in retriever/ranker predictions on validation; for each cluster, Claude 3.5 Sonnet writes 5 new MCQs with 4–5 reference MCQs from the cluster (prompt built with Anthropic's metaprompt; asks for analysis, planning, exact misconception wording, harder than references). GPT-4o as judge scores 0–10 the logical link misconception -> wrong answer, filters low scores. LLM-invented misconceptions (4k+) kept after dedup: string normalization, gte-base similarity >= 0.995 = match, 0.95–0.995 = drop, rest kept. Result: 1.8k competition + 10.6k synthetic MCQs, 4,791 misconceptions.
- CoT distillation: Claude 3.5 Sonnet given question, correct answer, wrong answer, true misconception and related ones writes the student's likely faulty reasoning (5–6 sentences). Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B fine-tuned as "reasoners" WITHOUT the misconception in input; at inference vLLM T=0.7, top_p 0.8, 256 tokens.
- CV: GroupKFold on ConstructId (QuestionId split was too optimistic, SubjectId too pessimistic); train folds 1–4, validate fold 0; synthetic as fold 99; final full fit.
- Retrievers: MultipleNegativesRankingLoss, FlagEmbedding-style; optimized recall@32 over MAP; LoRA r=64, alpha=128, LR A 1e-5 / B 5e-5, batch 128, 12 epochs; temperature 0.01 (not 0.02); one example per misconception per batch (avoid false in-batch negatives); pretrain 14B encoder on all uncurated synthetic data. Chose the high-recall Qwen2.5-14B encoder. Best encoder-only: 0.524 public / 0.475 private. Did not work: iterative hard-negative mining, cross-device negatives, subject-grouped batches, bidirectional conversion à la NV-Embed-v2.
- Rerankers: pointwise, score = logit("Yes") - logit("No") at last position, groups of 1 positive + (batch-1) negatives, cross-entropy; LoRA r=64, alpha=128, lm_head frozen. 14B ranker private LB ablation: few-shot reference examples in context 0.495 -> 0.531; pseudo-labels on synthetic data from Qwen2.5-Math-72B and Qwen2.5-72B pointwise teachers 0.531 -> 0.575; negatives 16 -> 24 and 2x synthetic 0.575 -> 0.596; 50% of inputs with CoT from the 14B reasoner 0.596 -> 0.615. Add 32B pointwise -> 0.625. Add 72B listwise (sees 3 CoTs from 7B/14B/32B reasoners, up to 5 reference examples, all 5 candidates) -> 0.638. No position-shuffle TTA (time).
- Quantization: AutoAWQ with task-specific calibration sets. Did not work: LoRA-merge "continuous fine-tuning" recipe; QwQ-32B-Preview fine-tuned the same way.

### 2nd place — CQYR (RihanPiggy et al.)
Sources: writeup above; train code https://github.com/wangqihanginthesky/Eedi_kaggle ; inference https://www.kaggle.com/code/honglihang/2nd-place-inference-code .
- Added parent-subject info from the NeurIPS 2022 Eedi challenge's `subject_metadata.csv` (same SubjectIds).
- Synthetic questions in 3 generations: LLM writes question for a misconception with few-shot examples; Qwen-Math solves it for the correct answer, then solves "under the misconception" for the wrong answer; gpt-4o-mini scores quality (keep > 2 of 5). Gen1 random few-shot, Gen2 same-misconception few-shot, Gen3 prompt adapted from a Preferred Networks blog.
- Misconception augmentation: Llama-3.1-70B-Instruct and Qwen2.5-72B-Instruct write a detailed explanation for each misconception (better than gpt-4o-mini); +2–4% retriever; free at submission time because misconception embeddings are precomputed.
- CoT from Qwen2.5-32B-Instruct-AWQ ("explain the mistake ... misunderstanding at conceptual level").
- Retrievers: Linq-Embed-Mistral with ArcFace loss + CoT input; Qwen2.5-14B/32B and QwQ-32B with MNRL, no CoT (time), last-token pooling > mean. Singles (private/public): Linq 0.461/0.484; Qwen14B 0.485/0.492; Qwen32B 0.495/0.536; QwQ 0.500/0.531. Training on H100: 2–5 days per retriever. Final ensemble Mistral + 2x Qwen14B (0.513 / 0.530) to leave time for 72B.
- Listwise rerank with sliding windows: Qwen2.5-14B-Instruct reorders ranks 8–17, then Qwen2.5-72B-Instruct and Llama-3.3-70B-Instruct finalize ranks 1–10; TTA with reversed candidate order. QLoRA training (based on atmaCup17 1st place); random top-N lists (3/5/15/25); 8k train + 14k synthetic records; negatives mined with fine-tuned stella_en_400M_v5 + TF-IDF. 14B ~2 h, 72B ~8 h on H100.
- Quantization: intel/auto-round 4-bit, group 128, symmetric, 512 calibration samples, 500 iters (<2% loss); padded Qwen2.5-72B intermediate size 29568 -> 29696 for multi-GPU. vLLM with prefix caching (~10% faster); reranking via logits processors (+100 bias on candidate tokens) and logprobs instead of a classification head.
- Ablation (private/public): retriever 0.513/0.530; +Qwen14B 0.568/0.583; +Qwen72B alone 0.593/0.582; +14B+72B+Llama70B 0.596/0.609; + window 8–17 0.604/0.622. 72B looked worse on public but better in validation; they trusted CV. Athene-V2-Chat variant hit 0.609 but timed out.

### 3rd place — Waseda Pochi (Yi-Chia Chen, solo)
Source: writeup above; inference https://www.kaggle.com/code/threerabbits/eedi-11-21-myq14b-q32b-rerank-mod-novel-local-suf .
- Retrieval: public Qwen-14B embedder (by anhvth226) ensembled with a FlagEmbedding-trained Qwen-14B; top 35. His own retriever was weak; removing it helped.
- Rerank: Qwen-32B-instruct-AWQ with 6 LoRAs ensembled; ~2,000 GPT-4o-mini synthetic samples added little. Score 0.590 public / 0.564 private.
- "Magic": >900 misconceptions in the mapping never appear in train; test has many unseen subjects. LB probing with one prediction per row: seen-only 0.154, unseen-only 0.444 -> guessed seen:unseen ~1:3. Post-processing multiplies unseen-misconception probabilities by a constant until 75% of top-1 predictions are unseen: 0.590 -> 0.658 public, 0.564 -> 0.602 private (with shuffled reranker input order: 0.670 / 0.602).

### Key threads and notebooks
- sayoulala, "Some tricks for training LLM Recall Model: CV 0.490, LB 0.352" (171 votes) — https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/543519 — LLM-as-embedder recipe that most top teams started from (content not read in detail).
- sinchir0, Retriever-Reranker baseline (LB 0.189) and fine-tuned BGE baseline (LB 0.246) — discussion/534317.
- moth, "Most misconceptions appear only once in train" — discussion/533920 — the long tail that made synthetic data and unseen-label handling central.
- Ahmet Erdem, "Logits Processors" (78 votes) — discussion/546978 — constrain vLLM output to candidate tokens; 2nd place used it.
- Jagat Kiran, "Qwen2.5-72B & Llama 3.3-70B on 2xT4" — discussion/550223 — how to fit 70B-class rerankers in the box.
- NguyenThanhNhan, "Private dataset accidentally shared by Eedi?" (63 votes) — https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550619 — a member of the (eventual) 7th-place team had requested Eedi's non-public misconception-labelled dataset used in a UMass NAACL paper (arXiv 2404.02124, GitHub issue on umass-ml4ed/prompt_distractor_generation_NAACL); participants asked whether it overlaps the test set; a commenter also flagged an exposed OpenAI key in that repo's history. No host ruling found in the fetched thread.
- eugenkrylov / zhudong1949 test-size probing (~685 questions, many unseen subjects) — discussion/550875 (cited by 3rd place).

### Shake-up
Moderate: public 1st (Waseda Pochi, 0.671) finished 3rd private; public 2nd (terekaerumasahmet 0.663) finished 7th; MTH 101 was 6th public and 1st private with a 0.039 margin over 2nd. Public was only 28% of the test, so LB-probed priors and public-tuned choices were fragile while CV-trusted large rerankers held up (2nd place writeup).

### Transferable lessons
1. Retrieve-then-rerank with progressively larger LLM rerankers on shrinking candidate lists fits a 9 h, 2x T4 budget — 1st place writeup.
2. Synthetic data must be generated in groups of confusable labels and filtered by an LLM judge; otherwise it teaches the wrong boundaries — 1st place writeup.
3. Distill reasoning (CoT) from a frontier model into small "reasoner" models and feed it to rankers as optional context — 1st and 2nd place writeups.
4. Enrich short label texts (misconception explanations) offline — label-side compute is free at inference — 2nd place writeup.
5. Choose CV split by the generalization axis you care about (ConstructId here) and trust it over a 28% public LB — 1st and 2nd place writeups.
6. Label-prior shift (unseen labels dominate test) can be estimated and corrected, but it is LB probing — 3rd place writeup.

### UNVERIFIED
- Exact misconception pool size (2,587 is commonly cited; 1st place says "2.5k+"). Host ruling on the private-dataset thread (none found) and whether any team used that data. 4th/5th place methods. Efficiency-track scores.

## 5. MAP — Charting Student Math Misunderstandings (2025)

### Facts
- URL: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings (id 104383). Hosts: Vanderbilt University and The Learning Agency (Eedi questions).
- Dates: 2025-07-10 to 2025-10-15 (API). Recap 2025-10-20.
- Task: students answered an Eedi diagnostic MCQ and wrote a free-text explanation. Predict a combined label `Category:Misconception`: Category = {True,False} x {Correct, Misconception, Neither} (whether the chosen answer is right, and whether the explanation shows a misconception), plus the specific misconception (e.g. `False_Misconception:Incomplete`). 65 combined classes; 37 once True/False is removed (the True/False part can be derived from whether the selected answer equals the correct one — discussion/589400). Only 15 QuestionIds in train and test; each question has just 2–5 possible misconceptions (3rd place writeup).
- Data: ~36k training rows (1st place: 35,960 after dropping duplicates); re-run test ~16,000 rows. Public LB = 50% of test.
- Metric: MAP@3 over up to 3 `Category:Misconception` predictions.
- Prizes: $55,000 — $20k / $12k / $8k / $5k / $5k / $5k (6 places).
- Teams: 1,857 teams / 2,175 competitors / 39,742 submissions (API); recap 1,858 teams, 2,176 participants, 8,465 registrations, 94 countries.
- Runtime: CPU or GPU notebook <= 9 h, internet off. Inference GPU = 2x T4 (1st place: "around 65 minutes ... with T4x2"; 3rd place notes no bf16 on T4).

### Top placements (scores separated in the 4th decimal)
| Rank | Team | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|
| 1 | tascj | 0.94919 | 0.95204 (14) | Suffix classification with shared-prefix FlexAttention masks; full fine-tune Qwen3-32B + GLM-Z1-32B (3 seeds); W8A8 SmoothQuant + layer-wise 32B inference on one T4 | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/1st-place-solution |
| 2 | looking for remote job (HZM + Baiph) | 0.94890 | 0.95373 (2) | 80k LLM-generated explanations for rare labels, soft labels from 4 teacher LLMs, Qwen3-14B with generation + soft-label CE | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map2025-private-and-public-2nd |
| 3 | monsaraida & Masaya | 0.94866 | 0.95443 (1) | Qwen3-14B classifier (multi-task, R-Drop, AWP, EMA) + Qwen2.5 14B/32B/72B CausalLM restricted to the question's label set; low-confidence re-inference | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/3rd-place-solution |
| 4 | THLUO | 0.94835 | 0.95211 (13) | single model, private 0.948 / public 0.951 (not read) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/4th-place-solution-single-model-pb-0-948-lb-0-951 |
| 5 | Naoism | 0.94830 | 0.95133 (24) | not read | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/5th-place-solution |
| 6 | Manan Jhaveri | 0.94828 | 0.95145 (22) | "Qwen-semble" (not read) | https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/mananjhaveri-map-6th-place-solution |
Gap between 1st and 6th private: 0.00091. Public LB was displayed to 3 decimals, which participants complained about (discussions/611010, 611060, 611982).

### 1st place — tascj (solo)
Sources: writeup above; notebook https://www.kaggle.com/code/tascj0/map-submit ; training https://github.com/tascj/kaggle-map-charting-student-math-misunderstandings .
- Data: deduplicated to 35,960 rows; 5-fold stratified by Category.
- Formulation: "suffix classification". Prefix = chat prompt with Question, Choices, Correct Answer, Common Misconceptions (candidates for this question), Student Answer, Student Explanation. Each candidate label is a suffix in submission format (e.g. `False_Correct:NA<|im_end|>`); each QuestionId has 8, 10 or 12 candidates. Last-token hidden state of prefix+suffix_i -> Linear(hidden,1) -> logits over candidates -> cross-entropy. Implemented as one packed sequence `prefix ++ suffix0 ++ suffix1 ++ ...` with a FlexAttention mask: causal AND (same suffix OR key is prefix) AND same document — the prefix is encoded once.
- Training: full-parameter fine-tuning with his own `offload_adam` (https://github.com/tascj/offload_adam), which fits up to 32B on one A100 80GB / RTX Pro 6000 Blackwell. epoch 1, batch 32, LR 1e-5.
- Validation: labels are noisy (mostly "Neither"), single-seed scores swing; ran 5 folds x 5 seeds with Qwen3-8B, then used the hardest fold with a 3-seed ensemble as the yardstick. Trust loss over MAP@3. Larger is better: 3-seed MAP@3 — R1-Distill-Qwen-7B 0.9444, Qwen3-8B 0.9455, GLM-Z1-9B 0.9469, R1-Distill-Qwen-14B 0.9464, Qwen3-14B 0.9477, Qwen3-32B 0.9484, GLM-Z1-32B 0.9480, Qwen3-32B + GLM-Z1-32B 0.9496. Multi-seed ensembles beat multi-fold ensembles.
- Final: Qwen3-32B and GLM-Z1-32B-0414 trained on full data, 3 seeds each with slightly different formats. An auxiliary SFT loss on short label justifications written by Qwen3-235B-A22B-Thinking-2507-FP8 helped R1-Distill-7B, not Qwen3-8B; included, marginal.
- Inference on 2x T4: T4 reaches only ~20 TFLOPS in practice in FP16; LMDeploy W8A8 INT8 kernels sustain 40+ TFLOPS, so models were SmoothQuant-quantized (alpha 0.75) with no measurable ensemble loss. Memory: layer-wise inference — keep 2 transformer layers on GPU, overlap running layer k with loading layer k+1 from disk; 640 samples per forward (40 micro-batches of 16). ~65 min for 16k rows. Prefix caching not implemented.
- Gotcha: Kaggle `/tmp` is copy-on-write with a cap (deleted files do not free space) — layer checkpoints there crashed runs; only 4 of 6 uploaded models made the final ensemble. Use `/kaggle/working`.

### 2nd place — HZM + Baiph (public and private 2nd)
Source: writeup above.
- View: an "easy" classification with ambiguous labels under a ranking metric, so denoise / soft-label.
- Synthetic data: for >30 rare labels, commercial LLMs (GPT-4, GPT-5, Claude Sonnet, Gemini, Seed, Doubao) wrote 80k student explanations, prompted with contrastive examples (same misconception on the same and on different problems, under both True_ and False_ variants).
- Soft labels: 4 teacher LLMs (Phi-4-reasoning-14B, Qwen3-32B, Mistral-12B, Qwen2.5-72B) trained on train.csv label the synthetic data; 5-fold OOF averages give soft labels for train.
- HZM's best single model: Qwen3-14B with a modified TRL `SFTTrainer` — generation loss on the hard label letter + cross-entropy between the option-token logits and the soft label. The prompt lists all 65 options as letters A..BM. Fold-0 CV 0.955, public 0.950, private 0.946.
- Baiph: merge 65 -> 37 classes, DeepSeek-written one-line explanations of every misconception placed in the prompt as lettered options; LR 2e-4, batch 4x2, 2 epochs. Qwen2.5-14B-AWQ public 0.948 / private 0.946, ~20 min inference. Extra data + distillation helped each model but not the fused ensemble.
- Turned classification into multiple-choice generation so vLLM could serve it in time.

### 3rd place — monsaraida & Masaya (public 1st)
Source: writeup above.
- monsaraida: prompt shows all Choices plus the Selected one (> +0.001 CV/public, also helped Masaya), and a "Common mistakes for this question" hint list ordered by frequency (+0.0003). Base model Qwen3-14B (beat gemma-2-9b, Qwen2.5-14B, Qwen2.5-Math-7B, deepseek-math-7b, R1-Distill-14B, Mistral-Nemo, Mixtral-8x7B, Mathstral, a pruned gpt-oss). LoRA + `AutoModelForSequenceClassification`, multi-task heads (65-class main + True/False, Correct/Misconception/Neither, 36-class misconception). R-Drop (> +0.001 CV), AWP on top (slight), EMA (stability). Inference: fp16 instead of bf16 on T4 (2x faster) and no max-length padding (2x) = 4x, so more ensemble members; second stage re-infers the lowest-confidence 50%.
- Masaya: CausalLM answering a letter from ONLY that question's options (Correct, Neither, and its 2–5 misconceptions; True/False added back by rule): +0.002–0.003 and ~3x faster; listing all choices +0.003–0.004. Qwen2.5 LoRA r=16, alpha 32, dropout 0.05, all projections, 2 epochs, LR 1e-4, batch 4, grad accum 4 (higher LR better until it collapses). Results (CV / public / private): 14B-AWQ 0.9469 / 0.943 / 0.942 (30 min); 32B-AWQ 0.9498 / 0.949 / 0.944 (60 min); 72B-GPTQ — / 0.950 / 0.946 (190 min); 72B-GPTQ with EMA — / 0.950 / 0.948 (single best, not used because public looked weaker). 72B: LoRA merged, GPTQ, CPU offload; only low top-1-probability rows re-inferred by 72B (~1 h). Did not work: Llama-3.3-70B, Gemma-2, pairwise CausalLM, fixed per-label letters, Eedi-pretrained weights, flipping True/False on QuestionId 31778 to handle label noise.

### Key threads and notebooks
- Chris Deotte, "Does Test Data Have Questions Different Than Train Data?" (98 votes) — https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/589400 — same 15 questions; True/False derivable; label-noise examples.
- Chris Deotte, "Ettin-Encoder-1B achieves LB 0.940" and "Is Bigger Better? Gemma2-9B vs Ettin-1B achieves LB 0.942" — discussions/590326, 590923 — small encoders were already near the ceiling.
- Chris Deotte, "DeBERTa versus ModernBERT?" and "Deberta versus Tf-idf?" — discussions/590096, 589452.
- Chris Deotte, "Is Public LB 0.950 a Resistance Level?" — discussion/591054 — the ceiling set by label noise.
- Chris Deotte, 18th place "Pyramid Ensemble" — https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/18th-place-pyramid-ensemble — small models on 100% of rows, medium on the 50% most uncertain (2x weight), large on 10% (4x), extra-large on 6% (8x); ~1,000 models trained over 3 months on ~4x A100; CV 0.951 / public 0.952 / private 0.947. Paradigms compared: sequence classification (65 or 37 classes), single-token CausalLM, multiple choice (<= 6 options per question).
- Martin Kovacevic Buvinic, "Place 22 with only 1 model" — writeup/place-22-with-only-1-model — one Qwen3-14B, 4-fold, prompt listing all 4 answers + possible misconceptions.
- hengck23, "The Rise of the 1-Shot Submit" — https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/discussion/611998 — the winner made ONE submission all competition (API: tascj submissionCount = 1, submitted 2025-10-15 18:32 UTC, ~5.5 h before the deadline) versus 202 for 2nd and 286 for 3rd. A clean story about trusting offline multi-seed validation over the public LB.

### Shake-up
Large in rank, tiny in score: public-1st team finished 3rd, the winner was 14th public; many writeups titled "Public 32 / Private 203", "Upshake 319 places", "Public 8th / Private 24th". With ~0.001 separating dozens of teams and noisy labels, ranks were close to noise (1st-place writeup on seed variance; Deotte "resistance level" thread).

### Transferable lessons
1. With noisy labels, judge changes on multi-seed ensembles and on loss, not single-seed MAP — 1st place writeup.
2. Restrict the output space to what is possible for this input (per-question label sets) — faster and more accurate — 1st (8–12 suffix candidates) and 3rd place (Masaya) writeups.
3. Encode a shared prefix once (FlexAttention masks or prefix caching) when scoring many candidates — 1st place writeup.
4. Old GPUs reward systems work: INT8 W8A8 kernels, layer-wise offload, fp16 not bf16, no padding — 1st and 3rd place writeups.
5. Spend big-model compute only on uncertain rows (cascade / pyramid) — 3rd place, Deotte 18th.
6. Soft labels from teacher ensembles plus targeted synthetic data for rare classes — 2nd place writeup.

### UNVERIFIED
- Exact number of training rows before dedup (~36.7k commonly cited). 4th–6th place methods. Content of Prashant Kikani's "Has this happened on Kaggle before?" writeup (title only).

## 6. ARC Prize 2024 (ARC-AGI-1)

### Facts
- URL: https://www.kaggle.com/competitions/arc-prize-2024 (id 67357). Hosts: ARC Prize Foundation (François Chollet, Mike Knoop; Greg Kamradt ran it).
- Dates: 2024-06-11 to 2024-11-10; paper submissions by 2024-11-12. Winners announced 2024-12-05 with the technical report (arXiv 2412.04604, https://arcprize.org/media/arc-prize-2024-technical-report.pdf).
- Task: each ARC task gives a few demonstration input/output grid pairs (up to 30x30, 10 colors) and one or more test inputs; produce the exact output grid (size and every cell). 2 attempts per test output; score = fraction of test outputs solved exactly. Private set: 100 hidden tasks. Public data: 400 training + 400 evaluation tasks with solutions. Leaderboard was 100% of the hidden set — the public LB score IS the private score (API `leaderboardPercentage` 100; alijs: "the Public=Private leaderboard").
- Prizes: total advertised $1,100,000. Progress prizes $125k: top-5 $25k / $10k / $5k / $5k / $5k; Paper Awards $75k (winner $50k, runner-up $20k, second runner-up $5k — third award added 2024-09-18, discussion/534826). Grand Prize $600k for >= 85% (raised by $100k on 2024-09-18): NOT claimed. $375k "to be announced" off-Kaggle. Only teams that open-source can win (MindsAI scored 55.5% but did not open-source, so not ranked).
- Teams: 1,427 teams / 1,637 competitors / 16,956 submissions (API); recap 1,451 teams, 1,663 participants, 16,383 registrations, 89 countries.
- Limits: CPU or GPU notebook <= 12 h, no internet, public data and models allowed. Technical report: "a single P100 GPU"; the ARChitects paper says they were "constrained by Kaggle's 2 Nvidia T4 GPUs" — both accelerator choices existed; treat the exact one per team as what each team reports. Daily submissions cut from 5 to 3 on 2024-06-27 to discourage probing (discussion/515322). Report: each Kaggle entry was equivalent to ~$10 of compute.

### Top placements (private = public, % of 100 tasks)
| Rank | Team | Score | Approach | Writeup / paper |
|---|---|---|---|---|
| (—) | MindsAI | 55.5 | test-time fine-tuning (AIRV); not open-sourced, ineligible | report |
| 1 | the ARChitects (Daniel Franzen, Jan Disselhoff, David Hartmann) | 53.5 | Mistral-NeMo-Minitron-8B, 64-token vocab, Re-ARC/ConceptARC/ARC-Heavy pretraining, test-time LoRA, DFS sampling, augmentation-stability scoring | paper https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf |
| 2 | Guillermo Barbadillo | 40.0 | Omni-ARC: Qwen2.5-0.5B multi-task (outputs + new inputs), per-task test-time fine-tuning, AIRV voting, + 2020 DSL ensemble | https://www.kaggle.com/competitions/arc-prize-2024/writeups/guillermo-barbadillo-2nd-place-solution-for-the-ar ; https://ironbar.github.io/arc24/05_Solution_Summary/ |
| 3 | alijs | 40.0 | solution summary inside notebook https://www.kaggle.com/code/alijs1/arc-prize-2024-solution-3rd-place-score-40 (not read) | https://www.kaggle.com/competitions/arc-prize-2024/writeups/alijs-3rd-place-solution |
| 4 | William Wu | 37.0 | No LLM: 2020-era DSL search over a DAG (icecuber etc.), decision trees, CNNs, ensembling | https://www.kaggle.com/competitions/arc-prize-2024/writeups/william-wu-4th-place-solution |
| 5 | PoohAI (gromml) | 37.0 | Ensemble of classic solvers + post-processing filters + task-ID probing | https://www.kaggle.com/competitions/arc-prize-2024/writeups/poohai-5th-place-solution |
4th place notes he was originally 7th and moved up "after some teams withdrew".

Paper Awards (technical report): 1st Li et al. "Combining Induction and Transduction for Abstract Reasoning" (MIT/Cornell, BARC, arXiv 2411.02272); 2nd Akyürek et al. "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning" (MIT, arXiv 2411.07279); 3rd Bonnet & Macfarlane "Searching Latent Program Spaces" (LPN). Runners-up: Franzen et al. (ARChitects); Barbadillo "Omni-ARC"; Fletcher-Hill "Mini-ARC"; Ouellette "Towards Efficient Neurally-Guided Program Induction for ARC-AGI"; Puget "A 2D nGPT Model For ARC Prize".
ARC-AGI-Pub (off-Kaggle, up to $10k API compute; semi-private / public eval): Jeremy Berman 53.6 / 58.5; Akyürek et al. 47.5 / 62.8; Ryan Greenblatt 43 / 42; o1-preview pass@1 18 / 21; Claude 3.5 Sonnet 14 / 21; GPT-4o 5 / 9.

### 1st place — the ARChitects (53.5; 56.5 in a run that finished after the deadline)
Source: paper "The LLM ARChitect: Solving ARC-AGI Is A Matter of Perspective", https://github.com/da-fr/arc-prize-2024 . On 100 split-off public-eval tasks (other 300 merged into training) they solve 72.5.
- Model constraints: fit 16 GB (one T4) for training and inference with >= 8,000-token context. Best: Mistral-NeMo-Minitron-8B-Base; also an uncensored Llama-3.2-3B-instruct.
- Data representation: tokenizer and embeddings shrunk to only 64 symbols so every grid cell is one token (dense, unambiguous).
- Data: public training replaced by Re-ARC (Hodel's procedural generators, a superset of training), plus 75% of the public evaluation set, ConceptARC and ARC-Heavy (BARC's 200k). Best private model "Nemo-heavy" trained on 531,318 examples.
- Augmentation everywhere (train, inference, scoring): the 8 D8 symmetries (rotations/transpositions), color permutations, shuffling the order of demonstration examples. Augmentation means no exact repetition in training.
- Preliminary training: LoRA rank 256 on all layers incl. input/output embeddings, LR 1e-4 (different for embeddings), 4-bit + gradient checkpointing via unsloth, on one H100.
- Secondary (test-time) training on Kaggle: LoRA rank 64 on the hidden tasks' demonstration pairs (with augmentations), time-boxed; computing loss on all outputs. A separate fine-tune per task helped slightly but was not runtime-efficient; instead they split the test set across the two T4s and ran the whole pipeline in parallel on each half.
- Candidate generation: greedy and multinomial sampling were suboptimal; a custom depth-first search enumerates every completion whose cumulative probability exceeds a cutoff (figure uses 5%), using KV caches, run on the original and augmented versions of each task — better score and faster.
- Selection ("model as both generator and classifier"): re-score each candidate under the 8 augmentations and pick argmax of the product of probabilities (sum of log-softmax). ~25% better than baseline selection for 10–17 extra min per 100 tasks on an H100. DFS log-probs reused in scoring for a free small boost.

### 2nd place — Guillermo Barbadillo, "Omni-ARC" (40.0)
Sources: Kaggle writeup; full lab notebook https://ironbar.github.io/arc24/ ; code https://github.com/ironbar/arc24 ; submission https://www.kaggle.com/code/ironbar/single-task-test-time-fine-tuning-for-arc24 .
- Idea: the representation is the hard part; train one model on many ARC-derived tasks so it learns a shared representation: examples+input->output (main), inputs->new input (learn the input distribution), and planned code/verification/selection tasks.
- Data (1,420 unique tasks): ARC 800, RE-ARC 400, PQA 7, Tama 50, Mini-ARC 149, nosound 9, Andy Penrose 5. Augment: rotations, flips, color changes, swap train/test examples; "problem augmentation" composes the task with a reversible transform applied to inputs only or outputs only (rotate/flip, pad, upscale, mirror).
- Grid text format: Markdown code block with a "grid shape: HxW" line and numbered rows.
- Training: Qwen2.5-0.5B-Instruct, LoRA rank 128, LR 5e-5 linear with warmup, batch 16, 2e5 steps, max length 8196, 2x A6000 (Veridas cluster; A100s from Strong Compute later). Full fine-tune no better.
- Test-time fine-tuning: one model per test task (100 per submission), leave-one-out over the task's n demonstrations (n-1 as examples, 1 as target), with augmentation, batch size 1, ~300 steps, LR 8e-5, continuing the trained LoRA. Raised one model from 11 to 33 solved.
- Inference: AIRV (augment, infer, reverse, vote, after MindsAI): 96 predictions per task, temperature 0, vLLM, then vote. Ensemble: first attempt from his model + first attempt from the public 2020 DSL solution (run on CPU in parallel).
- Did not pan out: verifier/selector models (selection top-2 81.0% vs voting 70.0% offline on frozen-model predictions, no LB gain — likely because TTT models' vote is already better); code-generation (285 hand-written DSL tasks, solved 5% of eval); BARC's 400k synthetic tasks (no gain); smaller (SmolLM2-135M, NanoLM-0.3B) worse; larger (1.5B, 7B) more data-efficient but too slow for the 12 h budget. Practical trick: used 3 daily submissions as extra compute (36 h/week vs 30 h/week of notebook GPU).

### 3rd place — alijs (40.0)
Only a pointer to the notebook (summary inside; not read for this bank). UNVERIFIED approach.

### 4th / 5th (classic, no LLM)
- William Wu (37): DSL search over a DAG, decision trees, CNNs, augmentation (flips, color swaps, symmetry) on top of 2020 winners (icecuber, golubev, szabo7zoltan, ilialar, mehrankazeminia) and Hodel's arc-dsl; benefited from 2024's 30 GB RAM and 12 h (vs 9 h in 2020). Ensemble by vote with icecuber prioritized.
- PoohAI (37): ensemble + post-processing filters for typical errors (genetic algorithm draws redundant lines; wrong shape) + explicit leaderboard probing: tasks are served in the same ID-sorted order every run, so with <= 100 submissions they identified which 26 tasks the well-known public "26" notebook solves, then tested new algorithms for single new solves and located them by binary search. Lesson stated: "removing wrong attempts is easier" than picking the right one.

### Key threads and notebooks
- Greg Kamradt, "Daily Submission Limit Change" (5 -> 3) — https://www.kaggle.com/competitions/arc-prize-2024/discussion/515322 — anti-probing; pointed people to ARC-AGI-Pub.
- Addison Howard, "September 18 Prize Increase!" — discussion/534826.
- Andreas Bisiadis, "What happened to first place?" — https://www.kaggle.com/competitions/arc-prize-2024/discussion/550442 — MindsAI 55.5% removed for not open-sourcing (quotes the report).
- "Solution approaches that have led to the current state-of-the-art" — discussion/535811 (survey thread, not read).
- mehrankazeminia, "3-arc24-developed-2020-winning-solutions" notebook — https://www.kaggle.com/code/mehrankazeminia/3-arc24-developed-2020-winning-solutions — the public 26-task CPU baseline used in many ensembles.
- Nikola Hu, sharing a 31-point solution and attention visualizations — writeup/nikola-hu-sharing-my-arc-prize-2024-code-31-points, discussion/545747.
- ivan, "Enhance 400k BARC models with 850k VLM" — writeup/ivan-enhance-400k-barc-models-with-850k-vlm (not read).

### Paper-award methods (for the series' "ideas" post)
- BARC (Li et al., arXiv 2411.02272): 100 hand-written seed Python programs -> GPT-4o-mini remixes into ~200k (ARC-Heavy) / 400k (ARC-Potpourri, incl. 100k Re-ARC transduction examples) synthetic tasks; Llama3.1-8B-instruct trained two ways: induction (write a Python program, sample up to 20k, vote by execution) 38.0% and transduction (predict grid directly, with TTT + reranking) 43.0% on the 400 public eval tasks; ensemble 56.75%. Induction is good at precise computation and composition, transduction at fuzzy perceptual concepts. Scaled-down Kaggle version scored only 19% private.
- TTT (Akyürek et al., arXiv 2411.07279): test-time training on in-context examples gives "up to 6x" over fine-tuned baselines; 8B LM reaches 53.0% public eval; ensembled with program synthesis 61.9%.

### Shake-up
None by design: public = private (same 100 tasks). The cost was probing risk, which the host countered by lowering daily submissions; 5th place documented probing openly.

### Transferable lessons
1. Test-time fine-tuning on each task's demonstrations is the single biggest lever for LLMs on novel tasks (11 -> 33 in Barbadillo's ablation; MindsAI, ARChitects, TTT paper) — 2nd place writeup; technical report.
2. Exploit the task's symmetry group: augment at train, test, and selection time; score candidates by stability across augmentations — ARChitects paper.
3. Tailor the tokenizer to the domain (64 tokens, one per cell/symbol) — ARChitects paper.
4. Enumerate high-probability outputs (DFS with cutoff) instead of sampling when you need exact answers — ARChitects paper.
5. Induction (programs) and transduction (direct prediction) solve different tasks; ensemble them — BARC paper.
6. If the private set is also the public set and order is fixed, the leaderboard leaks; design for it — PoohAI writeup; discussion/515322.

### UNVERIFIED
- alijs' 3rd-place method. Exact accelerator per team (P100 vs 2x T4). Which teams withdrew above William Wu. MindsAI's method details (not public).

## 7. ARC Prize 2025 (ARC-AGI-2)

### Facts
- URL: https://www.kaggle.com/competitions/arc-prize-2025 (id 91496). Dates: 2025-03-24/26 to 2025-11-03; paper deadline 2025-11-09; winners announced 2025-12-05 (https://arcprize.org/blog/arc-prize-2025-results-analysis ; technical report arXiv 2601.10904).
- Task: same format as 2024 (exact output grids, 2 attempts) but on ARC-AGI-2: new human-calibrated tasks every task solved by at least 2 humans in testing; overview text: "Humans have collectively scored 100% ... the best AI systems only score 4%" at launch. Hidden set ~240 tasks split 120 public / 120 private (Lonnie writeup; API `leaderboardPercentage` 50). Test-set fixes after community reports (Greg Kamradt, discussion/573301; changelog https://github.com/arcprize/ARC-AGI-2/blob/main/changelog.md).
- Rule/format changes vs 2024 (host tweet quoted in discussion/570243): Grand Prize $600k -> $700k; ARC-AGI-1 -> ARC-AGI-2; 2x compute — Kaggle L4x4 machines (4x L4, 96 GB) ~$50/submission, L4s usable only for this competition, internet off, double quota burn; public/private split; open source required before final standing; "additional overfit protection"; longer competition. 1 submission/day (API). Runtime 12 h, obfuscated +-10 min.
- Prizes: total $1,000,000. Top-5 $25k / $10k / $5k / $5k / $5k; Paper Awards $50k / $20k / $5k (plus 5 runners-up at $2.5k each, results blog); Grand Prize $700k for >= 85%: NOT claimed; $175k TBA off-Kaggle.
- Teams: 1,455 teams / 1,684 competitors / 15,141 submissions (API); blog: 1,455 teams, 15,154 entries; 90 papers (47 in 2024). Top score cost ~$0.20/task (blog).

### Top placements (private / public, % of tasks)
| Rank | Team | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|
| 1 | NVARC (Ivan Sorokin, Jean-François Puget — NVIDIA) | 24.03 | 27.64 (1) | 103k synthetic puzzles from gpt-oss-120b (description-mixing + code), 3.2M augmented samples, Qwen3-4B full SFT, ARChitects-style TTT + batched DFS + augmentation rescoring; TRM experiments | https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc ; code https://github.com/1ytic/NVARC |
| 2 | the ARChitects | 16.53 | 21.67 (2) | Masked-diffusion LLaDA-8B with 2D RoPE, soft-mask recursive refinement, separate shape predictor | https://www.kaggle.com/competitions/arc-prize-2025/writeups/the-architects-solution-2025 ; report https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/ |
| 3 | MindsAI & Tufa Labs | 12.64 | 15.42 (3) | Pruned 660M CodeT5-Large trained on >100M examples; TTT (~45k steps) + AIRV (10k augmented inferences) | https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution |
| 4 | Lonnie | 6.67 | 4.17 (343) | Public fork of ARChitects-2024 baseline; changed random seed | https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place |
| 5 | Guillermo Barbadillo @ Veridas | 6.53 | 11.94 (5) | Best LB from last year's transduction+TTT; research on search-and-learn program synthesis with hindsight relabeling | https://www.kaggle.com/competitions/arc-prize-2025/writeups/exploring-the-combination-of-search-and-learn-for |
Note: MindsAI's writeup labels 15.42% as "Private Leaderboard Score"; the Kaggle API and arcprize.org both give 12.64 private / 15.42 public — use 12.64. Lonnie's writeup title says "5th place" but the official rank is 4th. A mid-competition arcprize.org snapshot listed Giotto.ai at 27.08%; Giotto.ai is not in the final top 5 and no reason was found — UNVERIFIED.

Paper Awards 2025 (results blog): 1st A. Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks" (TRM, ~7M params, 45% ARC-AGI-1 / 8% ARC-AGI-2 per blog); 2nd J. Pourcel, C. Colas, P.-Y. Oudeyer, "Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI" (SOAR); 3rd I. Liao & A. Gu, "ARC-AGI Without Pretraining". Runners-up ($2.5k): Joffe & Eliasmith (Vector Symbolic Algebras); J. Berman ("From Parrots to Von Neumanns", evolutionary test-time compute); E. Pang ("Efficient Evolutionary Program Synthesis"); Guichard et al. (ARC-NCA); M. Ho et al. (ArcMemo). Honorable mentions include Franzen/Disselhoff/Hartmann "Product of Experts with LLMs", Barbadillo, Hu et al. "ARC-AGI is a Vision Problem!", McGovern "Test-time Adaptation of Tiny Recursive Models", Cole & Osman, Sorokin & Puget (NVARC).
Commercial context (blog): Opus 4.5 (Thinking 64k) 37.6% at $2.20/task top verified model; Gemini 3 Pro + Poetiq refinement 54% at ~$30/task (Gemini 3 Pro alone 31%). Theme: "the Year of the Refinement Loop".

### 1st place — NVARC (private 24.03)
Source: Kaggle writeup above.
- Synthetic data generation (SDG) with gpt-oss-120b via NVIDIA NeMo-Skills on Slurm (8x H100 node ~15k tokens/s):
  1. Puzzle descriptions from H-ARC (natural-language solution descriptions written by 1,700+ humans — "surprisingly underestimated") + 160 human-written BARC descriptions -> descriptions for 716 ARC-AGI-2 training puzzles; LLM summarizes them.
  2. Mix two summaries into a new, harder puzzle summary: 266,593 new summaries.
  3. Generate a Python program for the INPUT grid generator, with an LLM-written unit test: 126,901 kept.
  4. Generate OUTPUT (transformation) programs several times per input program; keep only puzzles whose output programs agree: 103,253 puzzles.
- Training data for the ARChitects-style model: 3,255,481 augmented samples (up to 7 pairs each): MINI-ARC 147 puzzles x256, ConceptARC 160 x256, RE-ARC 400 x256, ARC-AGI-2 609 x256, NVARC-training 47,337 x24 (34.8%), NVARC-full 55,886 x32 (54.9%; uses eval-puzzle descriptions too).
- Format: Qwen3 chat template, user turn = input grid rows, assistant turn = output grid rows; only 16 tokens needed (10 digits, newline, "user", "assistant", 2 special tokens, padding).
- Model and training: Qwen3 (4B final; 2B also tried), full SFT with NVIDIA NeMo RL (Megatron backend); 4B took 4 nodes x 8 H100 for 27 h. Local validation = loss on augmented 120 eval puzzles; correlated with public LB.
- Test time: per-puzzle LoRA TTT r=256, alpha=32, bf16, no gradient checkpointing, no 4-bit, FlashAttention-2 via Unsloth. Batched DFS decoding (their main optimization); batch-invariant kernels from Thinking Machines fixed nondeterminism but ran ~17% slower, so not used. Rescoring with exactly the same 8 augmentations for every candidate (comparable scores). Post-deadline finding: count how often DFS found a candidate, combined with geometric-mean of augmentation log-probs, selects better.
- TRM (Tiny Recursive Model) in the last ~10 days: ARC Prize's own run got 6.9% using 4 nodes x 8 H100; within ~2 h of Kaggle time they pretrain then test-time fine-tune: batch 3072 / LR 3e-4 matched original accuracy in 24 h on 8x H100 but scored 2.08%; tuned (4 H-cycles, 10 halt steps, 2,000 epochs, 200 warmup) -> 7.5% (best not-trained-on-eval pass@2 9.44%); post-deadline 4,000 epochs -> 10.0% in <4 h. TRM emits 10 attempts added to the Qwen candidates and rescored by Qwen; Qwen3-2B 21.53 -> 22.50 with TRM; Qwen3-4B (10 h) 27.22 -> 27.22 (no gain).
- Conclusion stated: scaling pretraining on good synthetic puzzles was the right way to "compress knowledge" given limited test-time compute.

### 2nd place — the ARChitects (private 16.53)
Source: Kaggle writeup (solution cards) + technical report link above.
- First half of 2025: their 2024 autoregressive pipeline (Mistral-NeMo-Minitron-8B incl. ARC-Heavy) with per-task TTT instead of multi-task, speculative decoding (16–32 tokens ahead) + prefix caching for faster DFS, probability cutoff lowered 17% -> 7%, scoring augmentations 8 -> 32. Peaked at 16.94% public (Aug 11). Weakness: tasks needing global restructuring (diagonals, simulations) — sequential generation cannot revise earlier tokens.
- Final: masked-diffusion LLM LLaDA-8B, cross-entropy on masked positions; "soft masking" (add the mask embedding to every token) lets the model refine its own prediction; recursive latent sampling feeds outputs back until stable ("token algebra" on continuous embeddings). 2D positional encoding (based on "Golden Gate RoPE") replaces 1D. Data: ReARC, ARC-GEN-100K, ARC-1 and ARC-2, ARC-Heavy, ConceptARC. Pretraining ~175k steps, batch 8, 8x H100. TTT: 128 steps per task on an L4, batch 1. A second LLaDA predicts output grid shape (~85% +-2%). With known shape ~30.5% +-1% on held-out eval after 102 inference steps (2 x 51 with cold restart); expected ~26% combined; got 21.67 public / 16.53 private. An unselected submission scored 19.17 / 19.17 — chosen against because of lower public score.

### 3rd place — MindsAI & Tufa Labs (private 12.64)
Sources: Kaggle writeup; full PDF https://github.com/jcole75/arc_2025_mindsai/blob/main/MindsAI_Tufa_Labs_2025_Solution.pdf ; dataset https://huggingface.co/datasets/mindware/arc-agi-mega ; arXiv 2506.14276.
- Team: Jack Cole (lead; pioneered TTT and AIRV for ARC since 2023), Dries Smit, Isaiah Pressman, Mohamed Osman, Michael Hodel (RE-ARC, ARC-DSL).
- Model: Salesforce CodeT5-Large encoder–decoder pruned to 660M (24 encoder layers kept, decoder pruned to 16), trained on >100M examples (~70M ARC-style) on Google TPUs over many months (some runs >2 years cumulative). T5 span corruption, prompt/answer reversals, BPE dropout.
- Test time: TTT with permutation-based labeling (~45k steps) + AIRV with 10k augmented inferences per task; ensemble of two checkpoints; ~11 h on 4x L4.
- Augmentations ranked: geometric + color permutations (essential); NEW mixup/combine/combine-mixup (+6.3% top-2 on ARC 1.5); input/output swap in 30% of training; reversals; BPE dropout.
- Findings: TTT and AIRV nearly perfectly additive (~812% combined vs ~410–430% each over zero-shot); self-ensembling seeds beats 2x more TTT/AIRV samples at equal compute (+6.2%); encoder depth matters far more than decoder depth; refinement, DPO on beam pairs and targeted ARC-2 data were not additive; "ARC-AGI-2 appears partially adversarial to the current TTT/AIRV paradigm". A simplified version (geometric+color only, TTT 20–30k, AIRV 5–10k) keeps 90–95% of the gain and runs on one consumer GPU.

### Key threads and notebooks
- ARChitects 2024 public notebook — https://www.kaggle.com/code/dfranzen/arc-prize-2024-solution-by-the-architects — base of most 2025 LB entries (ippeiogawa writeup; Lonnie writeup).
- Lonnie, 4th private from 343rd public by changing the random seed (19920627) on a public fork; argues seed is a hyperparameter when 120 tasks decide ranks — writeup above.
- ippeiogawa, "LB 10.00 with Small Tweaks to 2024ARChitects" — https://www.kaggle.com/competitions/arc-prize-2025/writeups/lb10-00-with-small-change-to-2024architects — picked "consistent" submissions instead of best LB; 6th private.
- jojobagel, "LB 8.33 with ARChitects approach and a 0.5B model (private runtime ~5 h)" — writeup/lb-8-33-with-architects-approach-and-a-0-5b- (not read).
- Younus_Mohamed, "ARC 2024 Solutions and Key Takeaways" — discussion/575595.
- Greg Kamradt, "ARC Prize 2025 Test Set Edits" — discussion/573301.

### Shake-up
Large. Top 3 held (NVARC, ARChitects, MindsAI) but each lost 3–5 points public -> private; Lonnie 343rd -> 4th; Ggmam1 55th -> 7th; Ttkuma 113th -> 12th. With 120 tasks per split, one or two tasks move ranks a lot (Lonnie writeup). ARChitects' better private submission was not selected.

### Transferable lessons
1. Synthetic data quality comes from human descriptions + program generation + consistency filtering (multiple output programs must agree), not raw volume — NVARC writeup (contrast Barbadillo 2024 finding BARC volume did not help).
2. Spend pretraining compute off-Kaggle to compress skills into a small model; keep test-time compute for TTT and search — NVARC writeup; MindsAI writeup.
3. Autoregressive decoders struggle with tasks needing global edits; iterative refinement (diffusion, recursion, TRM) is the 2025 frontier — ARChitects cards; ARC results blog ("Year of the Refinement Loop").
4. Test-time adaptation methods compose (TTT + AIRV additive; seed ensembles beat more samples) — MindsAI writeup.
5. Small hidden splits make seed and submission choice a lottery; pick robust submissions — Lonnie, ippeiogawa, ARChitects cards.

### UNVERIFIED
- Whether the NVARC final used Qwen3-4B exactly as the selected submission (the writeup reports "best model scored 27.64%" and Qwen3-4B 27.22 in a 10 h variant). Giotto.ai's disappearance from the top. Exact hidden-set size (~240 tasks inferred from Lonnie's "240 evaluation tasks").

## 8. ARC Prize 2026 (ONGOING as of 2026-10-27) — three Kaggle competitions

Status: submissions close 2026-11-02 23:59 UTC (both prediction tracks), paper track closes 2026-11-09, winners announced 2026-12-04. Everything below is a snapshot of the PUBLIC leaderboard on 2026-10-27; no private scores exist yet. Re-check before publishing.

### 8a. ARC-AGI-2 track — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2 (id 133469)
- Format: identical to ARC Prize 2025 ("a relaunch of ARC Prize 2025"): exact grids, 2 attempts; public/private split 50/50; 12 h notebooks, no internet; 1 submission/day; team size up to 5. Started 2026-03-25 on 2x T4 / P100; upgraded to L4x4 (96 GB) on 2026-04-07 (discussion/689054).
- Prizes: $700,000 total ($550k guaranteed). Progress prizes $275k for ranks 1–8: $75k / $50k / $40k / $35k / $25k / $20k / $15k / $15k. NEW: the $275k "Grand Prize" is no longer for 85% — it goes to the best Solution Writeup judged on Accuracy, Universality, Progress, Theory, Completeness, Novelty (0–5 each, averaged), artifacts open-sourced within 7 days of the deadline. Bonus $150k only if >= 85% on the leaderboard (split among top 5: $75k / $25k / $20k / ...).
- Teams (API, 2026-10-27): 3,425 teams / 3,815 competitors / 32,013 submissions.
- Public LB top 8 snapshot: 1 Tufa Labs 88.06; 2 Yi-Chia Chen @ Fields Model 84.72; 3 Domdolus Tolus 82.08; 4 nvbanana 81.39; 5 rabbithole 80.83; 6 Majkel1337 76.81; 7 ALbert RUehlman 75.56; 8 Dawid Kopiczko 75.14. (2025's winning score was 27.64 public.)
- How the year unfolded (forum):
  - Chris Deotte, "Last Years 1st Place Solution" — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/685142 — NVARC ported to 2x T4 scored 9.58; once L4x4 arrived, NVARC could run as-is. CPMP (NVARC co-author), "When will a new model be top?" (discussion/691081): NVARC-based public notebooks were setting new highs, e.g. 32.64%.
  - CPMP, "95% on ARC AGI1 evaluation" (discussion/694698): NVARC's 2025 code scores >95% on ARC-AGI-1 public eval, mostly because ARC-AGI-1 eval tasks are inside ARC-AGI-2 training data — "All frontier models use ARC AGI2 training data, hence their scores on ARC AGI1 are meaningless."
  - CPMP, "Good to see movement on LB" (discussion/724647, July): the 2nd team moved "beyond what could be done with knob fitting" on NVARC.
  - Anil Thomas, "Qwen3.8 or something else?" (discussion/750546, Oct 18): a sudden flood of 55+ scores; he went 35 -> 45 by ensembling an NVARC-based solution with a Qwen3.8-27B run; suspects a Python-REPL harness around Qwen3.8 is the recipe but 12 h on 4x L4 is tight.
  - Darren Amadeus Martin, "Are solutions using Qwen3.8-Flash-Next eligible for prizes?" (discussion/752709): open license question (Qwen Community License 1.0) affecting both tracks; no host answer captured.
  - Chris Deotte, "Watching LLM Agents Solve ARC-AGI-2" (discussion/750577): quotes the ARC Prize Foundation's ARC-AGI-2 leaderboard with frontier systems at 92–95% (GPT-6 Astra (Max) 95.00% at $1.119/task, Claude Opus 5.5 (High) 93.33% at $0.408/task, etc.). These are off-Kaggle API results quoted secondhand — verify on arcprize.org before using.
  - Infra pain in October: "Kaggle Error" submissions and multi-day queues (discussions/747515, 746520, 747193, 751344).
  - Rules threads on AI coding assistants during development (discussions/701528, 737679) — the offline-notebook rule applies to scoring, questions about assistants were asked; host answer not captured.
  - Tong Hui Kang, historical leaderboard with runtimes — https://arc3.huikang.dev/leaderboard/?comp=arc2 (discussion/709522).

### 8b. ARC-AGI-3 track — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 (id 133468)
- Task: interactive. Hand-crafted turn-based games; the agent receives frames (JSON with a grid up to 64x64, values 0–15, state NOT_FINISHED/WIN/GAME_OVER) and sends up to 7 actions (RESET, ACTION1–5 simple, ACTION6 with (x,y) coordinates, ACTION7); the meaning of actions must be discovered by exploration. Each game has several levels. 25 public games for development (arcprize.org/arc-agi/3); evaluation on a private set of 110 games, half for public LB and half for private LB.
- Metric: per level, agent action count vs a human baseline (efficiency), averaged over levels and games; 100% = beats every game with human-level action counts; capped at 100% (https://docs.arcprize.org/methodology). Jeroen Cottaar noted the second-best human's `baseline_actions` were visible to agents during submission (discussion/687655).
- Runtime: notebook <= 9 h, internet off, team size up to 8; RTX Pro 6000 machines (`g4-standard-48`, 96 GB) added for this competition only. 110 games in any order within 9 h (Tufa write-up).
- Prizes: $850,000 total. Final leaderboard $75k: $40k / $15k / $10k / $5k / $5k. Milestone prizes $75k (notebook must be public under an open-source license by the date): Milestone 1 (2026-06-30) and Milestone 2 (2026-09-30), each $25k / $7.5k / $5k. Bonus $700k only at 100% (split $350k / $175k / $70k / $70k / $35k).
- Teams (API, 2026-10-27): 4,905 teams / 5,593 competitors / 61,007 submissions.
- Public LB top snapshot: 1 Yi-Chia Chen @ Fields Model 75.56; 2 Tufa Labs 68.91; 3 Team Turducken 56.02; 4 the last dance 50.69; 5 mtg 50.37; 6 NVARC3 48.84.
- Trajectory: early public notebooks ~0.66% on 10 public games (CPMP, discussion/691696); Milestone 1 won at 1.21% official (1.30% retracted); Milestone 2 public LB ~28%; top now ~75% — driven by stronger open models (Qwen3.6-27B -> Qwen3.8-Flash-Next) and serving tricks rather than new harness logic (Tong Hui Kang comparison).

Milestone 1 (Greg Kamradt, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/725002):
1. Tufa Labs, "The Duck" (Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Dries Smit, Michal Tešnar, Stefano Viel) — write-up https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133 ; code https://github.com/Tufalabs/duck-harness ; blog https://tufalabs.ai/research/duck-harness. Successor to Dries Smit's "Stochastic Goose" (won the ARC-AGI-3 agent preview). Base model Qwen3.6-27B FP8 on one RTX Pro 6000; inspired by Symbolica's ARCgentica (recursive sub-agents, 36% with frontier models) and the RGB Agent (OpenCode harness, near-human efficiency on 3 preview games). Design: game state exposed as Python variables in a REPL (`current_frame.ascii`, `.segmentation`, `previous_frame`, `history`, `transitions`, `valid_actions`, `last_action_result`); 30 s / 4,096-char tool limit; actions with natural names (UP/DOWN/LEFT/RIGHT/SPACE/MOUSE/RESET), UNDO withheld because the model wasted it; a carried "World model:" note; 4x-upscaled image of the board each turn (Qwen's 16x16 patches); a 4-connected-component segmentation tool so the model does not dump whole grids into context; context capped at 64k, maintained near 32k by evicting the oldest user/assistant turns ("infinite play via eviction"). Scores: 1.21% Kaggle, 1.6% on public games; same submission ranged down to 0.77% (high variance). Philosophy: keep the harness lightweight and generic; gains came from multimodality and better base models.
2. Reki — vision-LLM-as-policy with Gemma-4-31B: rendered recent frames -> JSON {what changed, short plan, next 1–4 actions}, reflection memory every ~10 steps, numpy click heuristics favoring small rare-colored button-like shapes, "dead-signature" to stop clicking object types that never change anything. Notebook https://www.kaggle.com/code/ruichardliu/milestone1-2nd-solution .
3. Md Boktiar Mahbub Murad, "forge" — similar Gemma-4-31B agent with generator/arbiter and confidence prompts, but the top run turned all extra machinery off.

Milestone 2 (announced on X, not on Kaggle; compared by Tong Hui Kang, https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792): all three winners — dfranzen (public 27.89), lordhansolo (23.84), sirikilohit (22.53) — run Tufa's Duck harness on the Tufa ARC-AGI Framework (TAAF) with Qwen3.8-Flash-Next (hybrid with Mamba layers): 4-bit weights (Intel AutoRound W4A16 or NVFP4/FP8 mixed), ~70 GB weights on GPU, per-layer embeddings (~95 GB) offloaded to host, FP8 KV cache, 3-token speculative decoding with an MTP/draft head, SGLang ("Pennyroyal" builds) or vLLM nightly. The differences are serving engineering, not agent logic.
- Tufa Labs declined to open-source for Milestone 2 (Jeroen Cottaar, discussion/742801): after their M1 release "all high-scoring public notebooks" built on the Duck, and they feared the milestone would go to "the luckiest" of ~4,000 copies; they plan to release at the end. Same author asked for a freeze-then-claim window (discussion/705043) because open-sourcing drops you hundreds of places.
- Other threads: Greg Kamradt, "500 Submissions Analyzed - Common Errors" (discussion/727119) — a third of failures hang without visible error, ~20% forget to enable the GPU; Chan Kha Vu on a team (Mostik.AI) publicly admitting 12 members vs the 8 allowed (discussion/739186); Komil Parmar, "Is Offline BFS 'Cheating the Spirit' of ARC-AGI-3?" (discussion/687950); Jeroen Cottaar "Simplified submission framework" (discussion/686416).

### 8c. Paper track — https://www.kaggle.com/competitions/arc-prize-2026-paper-track (id 133724)
- 2026-03-25 to 2026-11-09. Submit a Kaggle Writeup (<= 1,500 words, cover image, attached public notebook, optional project link/PDF) documenting a submission to ARC-AGI-2 or ARC-AGI-3. Judged on the same 6 criteria (0–5 each, averaged); three Paper Awards from $75,000 guaranteed; a $375,000 bonus split among papers scoring above 4.5/5. 328 teams registered (API, 2026-10-27). Results not yet available.

### Transferable lessons (provisional — competition not finished)
1. When an open-weights model generation jump lands mid-competition, a lightweight generic harness inherits the gain; "let the model drive" — Tufa Duck write-up (discussion/717133) and M2 comparison (discussion/744792).
2. Context management (eviction + persistent world-model note) and perception tools (segmentation, upscaled images) matter more than clever planners for small local models — Duck write-up.
3. Serving engineering (4-bit MoE weights, offloaded embeddings, speculative decoding, FP8 KV) decides how many actions you can afford in 9 h — discussion/744792.
4. Milestone prizes that require open-sourcing create a copy-and-pray dynamic; prize design shapes sharing — discussion/742801, 705043.
5. Benchmark contamination: ARC-AGI-1 eval is inside ARC-AGI-2 training data, so ARC-AGI-1 scores of modern systems are not evidence — discussion/694698.

### UNVERIFIED
- Every score above is public-LB only and may change. Frontier-model ARC-AGI-2 numbers (GPT-6 Astra 95% etc.) are from a Kaggle post, not checked on arcprize.org. Milestone 2 official winner list (only via Tong Hui Kang's post and an X link). License eligibility of Qwen3.8-Flash-Next. Team compositions (e.g. Tufa Labs in both tracks) and whether "Yi-Chia Chen @ Fields Model" is the AIMO3 Math Corpus winner (same name; not confirmed).

## 9. Konwinski Prize (K Prize, 2024–25) — SWE-bench with a contamination-free future test set

### Facts
- URL: https://www.kaggle.com/competitions/konwinski-prize (id 84795). Host: Andy Konwinski (the overview is written in first person; follow-up at https://kprize.ai). "$1M for the AI that can close 90% of new GitHub issues."
- Dates: start 2024-12-11; final submission deadline 2025-03-12 (extended selection window after L4x4 queue problems, discussion/567896); forecasting phase on issues collected AFTER the deadline, end date 2025-07-09 per timeline, extended twice (discussions/582799, 588708); API deadline 2025-07-23; recap 2025-07-30; host wrap-up 2025-09-24 (discussion/609204).
- Task: an agent gets a repo snapshot and a GitHub issue's problem statement (SWE-bench style, Python repos) and returns a git diff, or skips. Scored by running the repo's fail-to-pass (F2P) and pass-to-pass (P2P) tests. Training phase: public LB on ~70 historical instances (71 established from score decimals, discussion/557158; labels scrapeable from GitHub, so the public LB was known to leak). Forecasting phase: new issues collected after the deadline, "approximately 150–200" announced; the private set turned out to be 120 instances (discussion/557148; consistent with 1st place's 9+2+109).
- Metric: score = (a - b - c/10,000) / (a + b + c), a = correct, b = wrong, c = skipped; if a = 0 the score is -1. A wrong patch costs as much as a right one earns, so skipping is the main strategy. Metric revised 2025-01-17 so that skipping everything no longer scores 0 and skips break ties (discussion/557133).
- Prizes: total $1,225,000. Leaderboard $50k / $20k / $10k / $10k / $10k. Threshold pools: +$50k at each of 30%, 40%, ..., 90% split among top-5 teams reaching it. Grand Prize: +$775,000 if 1st place exceeds 90% (bringing them to $1M). Only open-source code and open-weight models are eligible. None of the thresholds was reached.
- Teams: 617 teams / 531 competitors / 784 submissions in the API (final LB has 472 rows); recap: 471 teams, 531 participants, 4,961 registrations, 61 countries. Host wrap-up mentions "600+ teams" and "9000+ submissions" (differs from Kaggle's count — likely includes training-phase submissions; UNVERIFIED).
- Runtime: internet off; each issue must finish within 30 minutes of being served; notebooks <= 9 h in training phase, <= 24 h in forecasting phase; L4x4 (96 GB) machines available for this competition only, double quota burn.

### Top placements (private = forecasting set of 120 new issues)
| Rank | Team | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|
| 1 | Eduardo Rocha de Andrade | 0.058242 (9 correct, 2 wrong, 109 skipped — his edit) | 0.0 shown (366) | Agentless 1.5 fork + Qwen2.5-Coder-32B; context-aware reproduction-test generation; reject patches failing F2P or regressing P2P | https://www.kaggle.com/competitions/konwinski-prize/writeups/eduardo-rocha-de-andrade-1st-place-solution-write- |
| 2 | Camaro | 0.016576 | (142) | Agentless-style localize -> generate test -> patch -> verify, Qwen2.5-Coder-32B with JSON structured output, many skip gates | https://www.kaggle.com/competitions/konwinski-prize/writeups/camaro-public-2nd-place-solution |
| 3 | akami (akiyama) | 0.016575 | (388) | huikang's Select-Patch-Verify + LoRA difficulty classifier on DeepSeek-R1-Distill-Llama-70B-AWQ to skip non-easy issues | https://www.kaggle.com/competitions/konwinski-prize/writeups/3rd-place-solution |
| 4 | Genxxsky | 0.016571 | (302) | huikang's notebook base | https://www.kaggle.com/competitions/konwinski-prize/writeups/4th-place-solution |
| 5 | Kenj | 0.008239 | (361) | huikang's notebook + regex traceback parser + git-conflict-marker edit format | https://www.kaggle.com/competitions/konwinski-prize/writeups/5th-place-solution |
Derived from the metric (not stated by teams 2–5): 0.0165x corresponds to ~2 correct and 0 wrong; 0.0082 to 1 correct and 0 wrong out of ~120. Rank 11 onward were slightly negative (0 correct). Public column in the API shows 0.000000 for everyone after the rescoring, so public ranks are as listed by the API but public scores are not retrievable. Host summary: "The first place winner resolved 8%" (9/120 = 7.5%).

### 1st place — Eduardo Rocha de Andrade (joined ~1 month before the deadline)
Source: writeup above; code https://www.kaggle.com/code/arc144/custom-agentless-fork .
- Base: Agentless 1.5, modified for local models and runtime. Model Qwen2.5-Coder-32B (reasoning models produced long "but wait..." traces and were not better).
- Reproduction (F2P) test generation first: 5 candidate tests per issue with diverse context — 1 with example imports + relevant classes/functions, 1 with imports only, 3 with no context (original Agentless); one greedy, others sampled. Context found in two stages: model picks relevant unit-test files, then sees skeletons (classes/functions) and picks what to inspect; imports extracted from those files. Small models got the idea right but failed on imports and class usage — context fixes that. Keep only tests that print exactly "Issue reproduced" (anything that also prints "Other issues" is discarded). If none of 5 reproduces, generate 5 more at higher temperature; if none of 10, SKIP (no way to validate a patch).
- Localization: files -> classes/functions -> line-level edit locations (greedy for first two, sampled 2x for fine-grained). Repo structure parsed once (saved 5–20 s per step).
- Repair: 4 samples per edit location (1 greedy + 3 sampled) = 8 candidates, in SEARCH/REPLACE format (far more valid patches than asking for a unified diff directly), converted to git diff.
- Selection: `git apply` dry-run; keep patches that make at least one F2P test pass; run the model-selected P2P unit tests and require results equal or better than the unpatched reference; otherwise skip. Tests run in parallel with timeouts; package installation concurrent with test runs.
- Time: best runs 5–7 h (~3–10 min per issue); guards skip everything after 20 h globally and any issue after 12 min.
- Robustness: resubmissions scored 0.056–0.098 on the public LB. Hardware for dev: one RTX 4090 (24 GB, ~8k context), so little local validation (a few Django instances from SWE-bench Verified).
- Did not work / not tried enough: agent mode (feed test errors back to fix) tried too late; reasoning models; a full local validation on SWE-bench Verified.

### 2nd place — Camaro (~3 months of work)
Source: writeup above.
- Pipeline: (1) issue evaluation — LLM judges solvability to skip early (Qwen cautious, Gemini overconfident; phrasing matters; lenient setting chosen); (2) file localization — list Python files, pick 5 (more candidates, second pass, embeddings: no help; even giving ground-truth files helped little); (3) file selection — extract the relevant parts rather than yes/no; (4) reproduction-test generation, the most important step — few-shot examples help up to a point, syntax errors are fed back; tracked "good_test_rate" (reproduces AND is fixed by the gold patch); (5) repair as function-calling edits (path, old string, new string), retries on apply failure and on failing tests, up to 5 attempts; (6) verification of test logic, resolution, and no regressions — too strict a verifier rejects everything.
- Model: Qwen2.5-Coder-32B served by vLLM as an OpenAI-compatible server with JSON-schema structured outputs; beat Qwen2.5-72B, QwQ-32B, DeepSeek-R1 on stability and speed; 14B -> 32B fixed most context-loss and syntax problems. Dev on two rented RTX 3090s (vast.ai).
- Prompt engineering by "LLM consortium": compared ChatGPT, Grok, Gemini, DeepSeek outputs; 44 prompt variants across 6 steps. Large parts of the code written with an LLM.
- Evaluation: SWE-bench Verified subset of 100 and full 500 (0.04–0.06); 10 late submissions averaged 0.052 (0.028–0.070). Tracked per-stage metrics (file recall, selected-line recall, reproduced rate, replacement and syntax success rate); improving a stage metric did not always improve the final score. Warns that ~half of SWE-bench Verified is Django, so local scores can be gamed.
- Thread "How random is your submission?" (discussion/568626): same code scored 0.0703 (7,2,62) and -0.0142 (4,5,62).

### 3rd place — akami (akiyama et al.)
Source: writeup above.
- Base: Tong Hui Kang's "Select-Patch-Verify" starter notebook. One model for everything (load once): DeepSeek-R1-Distill-Llama-70B-AWQ.
- Difficulty gate: LoRA (rank 32) on the same base as a 3-class classifier (easy < 15 min / medium 15–60 min / hard > 1 h, using SWE-bench Verified's human difficulty labels, 400 train / 100 validation); skip if P(easy) < 0.5.
- Search: 6 parallel query generations at T=0.6 over the issue + directory tree; take matching lines +-12 lines as context. 6 patches per search result. Verify by LLM yes/no judgment, unidiff parse, `patch` dry run; submit one passing patch or skip.
- Their view: "it was just luck that we ended up in 3rd place."

### Key threads and notebooks
- Tong Hui Kang, "Starter notebook with LLMs" / Select-Patch-Verify — https://www.kaggle.com/code/huikang/starter-notebook-select-patch-verify (discussion/553294) — base of 3rd, 4th, 5th and 8th place.
- Tong Hui Kang, "The first published notebook not scoring -1" — discussion/561695 — 1 correct, 37 wrong, 33 skipped = -0.507: the metric punishes attempts.
- Tong Hui Kang, "How to compute (n_correct, n_wrong, n_skipped) given leaderboard score" — discussion/557148 — LB decoder (public 71, private 120).
- Geremie Yeo, "The public test set has 71 instances" — discussion/557158.
- Sohier Dane, "Competition Upgrade Plan Details" (discussion/552449) and "Metric Update and Leaderboard Rescore" (discussion/557133) — test-environment access for running unit tests inside submissions; `predict()` signature change (discussion/557125); metric fix.
- Alex Golubev (Nebius), 80,036 SWE-agent trajectories + 6,411 GitHub issues — discussion/552605 — training data for fine-tuning.
- Tong Hui Kang, "Changes that you could have made to the Select-Patch-Verify notebook" — discussion/568634 — he could not pass any example test cases himself; some required fixes not mentioned in the issue.
- Khoi Nguyen, "This is the most gigachad competition ever held on this website" (84 votes) — discussion/551229 — a $1M prize from one individual host.
- ISAKA Tsuyoshi (GraphNINJAs), public 4th / private 15th — writeup/graphninjas-public-4th-place-private-15th-place-so — best public 0.056 with 4 correct, 0 incorrect.

### Shake-up
Total. Public scores were wiped and private ranks bear little relation to public ranks (winner was 366th public by the API listing). With ~1–9 correct answers out of 120 and a symmetric penalty, one issue moves a team many places; several top writeups call their result luck (3rd place; 1st place "luck will play a significant role").

### Transferable lessons
1. When wrong answers cost as much as right ones earn, the system is a selective classifier: invest in knowing when to abstain — metric (Evaluation page); 3rd place difficulty gate; 1st place skip-if-no-reproducing-test.
2. Generate a reproduction test before the fix; it is both the gate and the verifier — 1st and 2nd place writeups.
3. Ask small models for SEARCH/REPLACE or old/new-string edits, never raw unified diffs — 1st place, 2nd place, 5th place (git conflict markers).
4. Give small models the repo context frontier models have memorized (imports, class skeletons) — 1st place writeup.
5. A fresh, post-deadline test set removes contamination and also exposes how far open models are from benchmark numbers (8% vs SWE-bench headlines) — host recap discussion/609204.
6. Track per-stage metrics, but optimize the end metric; stage gains can hurt the total — 2nd place writeup.

### UNVERIFIED
- Exact (correct, wrong, skipped) for places 2–5 (only derived). Host's "600+ teams / 9000+ submissions" vs Kaggle's 617 / 784. Whether a second K Prize round has launched (kprize.ai mailing list only).

