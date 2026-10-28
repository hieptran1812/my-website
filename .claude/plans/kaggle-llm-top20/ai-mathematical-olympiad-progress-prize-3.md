# AI Mathematical Olympiad – Progress Prize 3 (2025–26): top-20 addendum

- Slug `ai-mathematical-olympiad-progress-prize-3`, competition id 118448, forumId 9129558.
- Ranks 1 to 3 are covered in depth in `.claude/plans/kaggle-llm-writeup-bank-part2.md`, H2 "## 3. AI Mathematical Olympiad — Progress Prize 3 (2025–26)" (also the shared Andreas Bisiadis / Parthenos backbone, extra prizes, Nitarach's "Model Capability Dominates"). Not repeated here.
- Leaderboard pulled from `competitions.LeaderboardService/GetLeaderboard` on 2026-10-28. Raw JSON and topic dumps: `/private/tmp/claude-501/kaggle-top20/aimo3/`.
- Forum sweep: all 466 topics paged, filtered by title and by author (every top-22 member's username). Writeup links from the API's `solutionWriteUpUrl` field. Host posts read: winners announcement (discussion/714435) and extra prizes (discussion/708484).

## 1. Top-20 table

Metric: two private reruns, per problem 1 if both correct, 0.5 if one, out of 50. **Public rank is unavailable:** the API shows 0.0 public for every team after finalization, so the API's public "rank" is an ordering of zeros and is not reported.

| Rank | Team | Members (username) | Private | Public rank | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | Exalted Joseph | Exalted Joseph (theexaltedone) | 44.0 | unavailable | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/1st-place-solution-for-the-aimo3-competition (topic 703222) | gpt-oss-120b, no training; Parthenos "44/50 let me over cook" notebook (commenters say unchanged v1) |
| 2 | varianceofx | varianceofx | 44.0 | unavailable | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/2nd-place-solution-ai-mathematical-olympiad-prog (topic 702423) | Same base + answer-range prompt, tail-window entropy, vote-share scoring, uncatchable-leader stop |
| 3 | SKobayak | SKobayak | 44.0 | unavailable | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/3rd-place-solution-for-the-aimo3-competition (topic 702479) | ZaynYu baseline, prompt-only change from LLM-assisted error analysis |
| 4 | taku | taku (taku56) | 44.0 | unavailable | no writeup found | unknown; not among announced prize winners |
| 5 | TAMU-TACO | Fangzhou Lin (linfangzhou), Peiran Li (peirancs) | 43.5 | unavailable | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3-tamu-taco-solution (topic 704510) | gpt-oss-120b, no training; inference policy (300/900 s timeouts, 8 attempts, stop at 4) tuned by an "AutoResearchAgent" against hard-problem order |
| 6 | Man Penguin | Man Penguin (manpenguin) | 43.5 | unavailable | no writeup found | unknown; not among announced prize winners |
| 7 | yemao ye | yemaoye | 43.5 | unavailable | https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/7th-place-solution-for-the-aimo3-competition (topic 704412) | Andreas/Parthenos base, short IMO prompt, simple inverse-entropy vote |
| 8 | BarrryYang | BarrryYang | 43.5 | unavailable | no writeup found | unknown |
| 9 | Yew Jin Lim | Yew Jin Lim (limyewjin) | 43.5 | unavailable | no writeup found | unknown |
| 10 | Mehedi | Mehedi (mehedi457) | 43.5 | unavailable | no writeup found | unknown |
| 11 | Dylancelia | DYLAN (dingqiuyang) | 43.5 | unavailable | no writeup found | unknown |
| 12 | twotodevnull | twotodevnull | 43.5 | unavailable | no writeup found | unknown |
| 13 | stpete_ishii | stpete_ishii (stpeteishii) | 43.5 | unavailable | no writeup found | unknown |
| 14 | Md. Hamid Hosen | Md. Hamid Hosen (hosen42) | 43.5 | unavailable | no writeup found (Hardest Problem Prize winner) | only submission to solve problem ACUTES in both runs |
| 15 | Abhimanyu Karki | Abhimanyu Karki (abhimanyukarki0) | 43.5 | unavailable | no writeup found | unknown |
| 16 | Naji | Naji (alazharn) | 43.5 | unavailable | no writeup found | unknown |
| 17 | Micheal1994 | HUG (micheal1994) | 43.5 | unavailable | no writeup found | unknown |
| 18 | Mert Kuzu | Mert Kuzu (mertkuzu1) | 43.5 | unavailable | no writeup found | unknown |
| 19 | Greg | Greg (gregkoja) | 43.5 | unavailable | no writeup found | unknown |
| 20 | VelvetStorm | VelvetStorm | 43.5 | unavailable | no writeup found | unknown |

Ties as given by the API: ranks 1 to 4 at 44.0; ranks 5 to 20 at 43.5, and the 43.5 tie continues past the cut (21 Telorina – Yuran Wang (prylone), Yuran Bi (telorina); 22 Makar Baderko; possibly more). Kaggle breaks ties by earlier submission, so ranks 5 to 20+ are ordered by submission time, not skill.

**Writeups found: 5 of 20** (ranks 1, 2, 3, 5, 7 – exactly the five announced prize winners).

## 2. Per-team synthesis

### Ranks 1 to 3: facts the bank is missing
- **Exalted Joseph (1st).** Commenters checked the notebook and say the submission is an **exact copy of Parthenos's "[44/50] LET ME (over)COOK!!!" version 1** (https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook?scriptVersionId=291499043; comments id3464271, id3464272). Asked where the "adaptive runtime scheduling by difficulty, verification complexity, reasoning depth" lives in the code, the author concedes it is only a global time budget plus early stopping at 4 matching answers: unused time rolls over to later problems, so harder (less consistent) problems implicitly get more time; there is no explicit difficulty score (id3464286, id3464298). Treat the bank's description of his "adaptive scheduling" accordingly.
- **varianceofx (2nd).** The selected private submission is **version 10** of the notebook; later versions (one shows 35) were experiments (id3463127, id3463293).
- **SKobayak (3rd).** Confirms in comments that the final notebook changed **only the prompt**; parallel-computation and voting variants were tried and dropped because the score appeared to fall (id3463613). The same final notebook submitted three times scored **36, 37 and 39 public** (id3464485), then 44 private – a direct measure of the noise.

### Rank 5 – TAMU-TACO (43.5), topic 704510
HF Space: https://huggingface.co/spaces/peiranli0930/AIMO3-TAMU-TACO . No training: pretrained gpt-oss-120b on one H100 80GB.

Framing: AIMO3 is **a sequential resource-allocation problem** – one hidden problem at a time, 5-hour budget, no knowledge of what comes next, unused VRAM cannot be "saved" for later problems. They argue private robustness is governed mainly by **where hard problems fall in the order** and by four coupled parameters: `base_problem_timeout=300`, `high_problem_timeout=900`, `attempts=8`, `early_stop=4`.

- **Local set S (100 problems):** the 10 reference problems were used as difficulty anchors (solved by GPT-5 high at pass@1 = basic, only at pass@3 = harder, unsolved = hardest). Candidates screened from AIMO2, OpenMathReasoning and Putnam-AXIOM with GPT-5 high judging difficulty and answer format, plus human review. Mix 70 basic / 20 harder / 10 hardest, tuned as 70 Basic + 30 Hard.
- **Stress test:** every candidate setting runs with the 30 hard problems placed at the front, middle, back and random, 3 repeats each = 12 full runs of S per setting.
- **"AutoResearchAgent":** a human-in-the-loop experiment engine that runs the suites, logs correctness, time, entropy, early stops, timeouts, OOM, GPU/CPU/RAM, summarises with GPT-5 high and proposes the next setting inside human-set bounds (timeouts 200 to 348 base, 348 to 1000 high; early_stop 3 to 5; attempts 3 to 8). Bounds came from the clock: (18,000 s − 600 s setup) / 50 = 348 s; fastest correct reference solution took 212 s; 8-way inference at 16k to 32k tokens per path used 68.6 to 74.2 GB.
- **Two-phase search:** phase 1 (attempts 6, stop 3) tuned the timeouts, 46 rounds = 552 runs, stable region high 894 to 906 / base 288 to 306, locked at 900/300; phase 2 tuned attempts/stop, 18 rounds, best score-variance at 8/4. Total 64 rounds, up to 768 full runs, on **64 H100s** of their own.
- **Inference:** per-problem budget = remaining time / problems left, clipped to [300, 900]; up to 8 attempts via a local vLLM OpenAI server; 16 pre-warmed persistent Jupyter kernels; an attempt stops as soon as it emits a valid boxed integer; stop all at 4 matching answers; else 1/mean-entropy-weighted vote over top-5 logprobs; clamp to [0, 99999]; fallback 0. vLLM settings: FP8 E4M3 KV cache, `gpu_memory_utilization` 0.96, 65,536 context, batch 256, async scheduling, prefix caching.
- **Pilot facts:** at attempts 6 / stop 3, 16 of 20 basic problems reached 3 matching answers; 6-way used 67.8 to 71.6 GB.
- **Did not do:** no fine-tuning or RL ("larger gain came from configuring inference"); did not tune on public LB or a single local shuffle.
- **Distinguishing idea:** optimising for robustness over problem orderings rather than peak score, with a reproducible tuning harness.

### Rank 7 – yemao ye (43.5), topic 704412
Base: Andreas Bisiadis's and Parthenos's notebooks (both weighted-entropy versions cited). No training.
- **Prompt:** tried Parthenos's 5-step UNDERSTAND/EXPLORE/PLAN/EXECUTE/VERIFY prompt and added checklists; longer prompts "did not always give a stable advantage", used more context and made reasoning less direct. Final system prompt is three sentences: world-class IMO competitor, answer a non-negative integer in [0, 99999], put it in `\boxed{}`; tool prompt says the Jupyter is stateful and to use `print()`; preference prompt lists math, numpy, sympy.
- **Aggregation ablation (public LB, same prompt):** simple inverse-entropy weighting, 6 submissions, 36 to 40 (40 twice); a complex entropy metric (mean + position-weighted entropy, variance, high-entropy penalty, low-entropy-streak reward), 3 submissions, 36 to 39. Kept the simple one: fewer hand-set hyperparameters, and token entropy is only a rough confidence signal that a complex transform may amplify as noise.
- **Runtime:** 1x H100, FP8 KV, up to 81,920 context, 8 parallel attempts, 16 Jupyter kernels, ~300 to 400 s per problem, ~4 to 5 h total.
- **Not selected:** longer per-problem budgets (eat time early), more aggressive early stopping (locks in repeated wrong answers), wider tuning of temperature/min_p/timeouts (not enough submissions).
- **Distinguishing idea:** deliberate simplification – the shortest prompt and simplest vote of any documented top team.

### Rank 4 – taku, rank 6 – Man Penguin
No writeup, no forum posts. Both are absent from the five announced winners (discussion/714435), which cites stricter compliance including releasing submissions on HuggingFace; the post names no reasons for any team.

### Rank 14 – Md. Hamid Hosen
No writeup. Host (discussion/708484): his submission was the only one to solve problem "ACUTES" in both evaluation runs; awarded the $30,000 Hardest Problem Prize; overall 43.5. Host adds that ACUTES was not necessarily the hardest problem – "ROLLER" was solved by no one in both runs, even with pass@N effectively ~4,000 across teams.

### Ranks 8 to 13, 15 to 20
No writeup and no posts by any member in the forum; one web search found no rank-attributed writeups. Likely forks of the shared backbone (host: "the main AIMO3 competition ended up being dominated by one notebook"), but this is UNVERIFIED per team.

## 3. Technique frequency matrix

Columns with any information: ranks 1, 2, 3, 5, 7 (writeups) and 14 (host note). All other columns are `?` throughout and omitted.

| Technique | 1 | 2 | 3 | 5 | 7 |
|---|---|---|---|---|---|
| gpt-oss-120b backbone | Y | Y | Y | Y | Y |
| Any training / fine-tuning | N | N | N | N | N |
| Built on a public notebook | Y (Parthenos) | Y (Parthenos) | Y (ZaynYu) | ? (same stack, not credited by name) | Y (Andreas + Parthenos) |
| Python tool via persistent Jupyter | Y | Y | Y | Y (16 kernels) | Y (16 kernels) |
| 8 parallel attempts | Y | Y | Y | Y | Y |
| Early stop on k agreeing answers | Y (4) | Y (4 + uncatchable leader) | Y (baseline) | Y (4) | Y |
| Entropy-weighted vote | Y | Y (vote share dominant, tail entropy) | N (majority) | Y | Y (simple) |
| FP8 KV cache | Y (8-bit) | Y | ? | Y | Y |
| Prefix caching | Y | ? | ? | Y | ? |
| Custom prompt changes | N (kept original) | Y (answer range) | Y (error analysis) | N | Y (shortened) |
| Explicit per-problem timeout policy | N (global pool) | ? | ? | Y (300/900) | Y (dynamic) |
| Offline validation set | ? | ? | Y (1,120 outputs) | Y (100-problem S, 768 runs) | N (public LB) |
| External LLM used in development | Y (Gemini prompts tried) | ? | Y (error categorisation) | Y (GPT-5 high) | N |

Counted summary (M = 5 teams with writeups): 5 of 5 used gpt-oss-120b with no training; 5 of 5 used 8 attempts with Python tools and early stopping; 4 of 5 used entropy-weighted voting; 3 of 5 changed the prompt (and 2 of those 3 went in opposite directions: longer and shorter); 2 of 5 built a serious offline validation harness; 4 of 5 explicitly credit a public notebook.

## 4. Consensus vs divergence

- **Table stakes:** gpt-oss-120b, vLLM on one H100 with FP8 KV and long context, 8 parallel tool-using attempts, early stop at 4 agreeing answers, `\boxed{}` extraction in [0, 99999]. Every documented top team did this; nobody trained.
- **What separated top 5 from 6 to 20:** nothing measurable. Ranks 1 to 4 scored 44.0 and 5 to 20+ scored 43.5 – a half-point, i.e. one problem right in one of the two runs. Rank 1 is reportedly an unmodified public notebook; rank 3's final was the same notebook scoring 36 to 39 public. The honest summary is that the top-20 order is noise on top of a shared notebook, which the host acknowledges.
- **Contradictory findings:**
  - Prompt length: SKobayak (3rd) added constraint-checking and complexity-estimation instructions from error analysis and credits the prompt; varianceofx (2nd) added an answer-range block and checklist; yemao ye (7th) found longer structured prompts unstable and shortened to three sentences; Exalted Joseph (1st) tried Gemini-generated prompts and kept the original. URLs: 702479, 702423, 704412, 703222.
  - Entropy aggregation: varianceofx found pure 1/entropy can be hijacked by one confident wrong attempt and made vote share dominant; yemao ye found a more complex entropy metric no better and kept plain 1/entropy; SKobayak used plain majority; TAMU-TACO used 1/entropy only as fallback when 4-agreement is not reached.
  - More attempts: SKobayak found 16 solvers worse than 8; TAMU-TACO capped at 8 for memory; varianceofx says >8 exceeded the time budget.

## 5. Aggregated "what did not work"
- More than 8 attempts / 16 solvers – SKobayak, varianceofx (bank); TAMU-TACO (OOM risk).
- Lower temperature – varianceofx (bank).
- Per-attempt diverse prompts – varianceofx (bank); Nitarach writeup (bank).
- Fine-tuning – not attempted by any documented top team (TAMU-TACO: inference gave more).
- Long structured prompts (5-step protocol + checklists) – yemao ye.
- Complex entropy metric with position decay and streak rewards – yemao ye (36 to 39 vs 36 to 40).
- Reasoning in Japanese; draft-then-finalize self-correction; complexity-consistency prompts – SKobayak (bank).
- Parallel-computation and voting changes on top of the baseline – SKobayak (comment id3463613).
- Longer per-problem budgets and aggressive early stopping – yemao ye; TAMU-TACO's stress tests found the 894 to 906 / 288 to 306 s region robust instead.
- Tuning to one shuffle or to the public LB – TAMU-TACO, SKobayak.

## 6. Shake-up note
Public scores are not retrievable from the API, so movement cannot be computed. Direct evidence of noise: SKobayak's final notebook scored 36, 37, 39 public across three submissions and 44 private; yemao ye's chosen config ranged 36 to 40 public over 6 submissions; Nitarach measured a 4.5-point public/private swing on an identical config (bank). The two-run private metric reduces but does not remove luck; ranks 5 to 20+ share 43.5 and are ordered by submission time.

## 7. UNVERIFIED
- All public ranks and scores (API zeroed).
- Why ranks 4 (taku) and 6 (Man Penguin) received no prize; host cites stricter compliance (HF release) generically.
- That rank 1 is a byte-identical copy of Parthenos v1 (asserted by two commenters, not refuted by the author; not diffed here).
- Approaches of ranks 4, 6, 8 to 20; whether Md. Hamid Hosen's notebook differs from the shared backbone.
- TAMU-TACO's "64 H100" tuning cluster and run counts are as stated in their post.
