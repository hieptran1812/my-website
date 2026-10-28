# AI Mathematical Olympiad – Progress Prize 1 (2024): top-20 addendum

- Slug `ai-mathematical-olympiad-prize`, competition id 73231, forumId 4783756.
- Ranks 1 to 3 (and a short rank-4 note) are covered in depth in `.claude/plans/kaggle-llm-writeup-bank-part2.md`, H2 "## 1. AI Mathematical Olympiad — Progress Prize 1 (2024)". This file does not repeat that detail; it adds only what the bank lacks, plus ranks 4 to 20.
- Leaderboard pulled from `competitions.LeaderboardService/GetLeaderboard` on 2026-10-28. Raw JSON: `/private/tmp/claude-501/kaggle-top20/aimo1/`.
- Forum sweep: all 388 topics paged, filtered by title (place/solution/rank/writeup) and by author (every top-22 member's username). Writeup links come from the API's `solutionWriteUpUrl` field.

## 1. Top-20 table

Metric is the count of correct answers out of 50 (private set). **Public rank is unavailable:** the API zeroes every public score after finalization (all show 0), so the public rank the API returns is an ordering of zeros and means nothing. The column is kept blank on purpose.

| Rank | Team | Members (username) | Private | Public rank | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | Numina | Jia Li (jiali4), Edward Beeching (edwardbeeching), Lewis Tunstall (lewtun), Hélène Evain (hlneevain) | 29 | unavailable | https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/numina-numina-1st-place-solution (topic 519303) | 2-stage full SFT of DeepSeekMath-7B-Base (CoT then ToRA-style TIR), 8-bit GPTQ, SC-TIR N=48 M=4 |
| 2 | CMU_MATH | Zhiqing SUN (edwardsun0909) | 22 | unavailable | https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/cmu-math-2nd-place-solution-all-code-and-datasets- (topic 518964) | Fine-tuned DeepSeekMath-7B-RL policy + 7B ORM, geometric-mean-weighted vote |
| 3 | after exams | mirceatlx, David Dinucu-Jianu (davidd2), Cătălin Grîu (catalin4444), Catalin Brita (catalinbrita) | 21 | unavailable | https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/after-exams-3rd-place-solution (topic 517206) | Un-finetuned DeepSeekMath-7B-RL in vLLM, 120 to 160 samples, small-number and in-statement-number penalty |
| 4 | codeinter | Zonglin Han (zonglinhan), winter2003, Yz H (ahahahaheihei), HUIHUANG LV (huihuanglv), Aristotle.Chen (aristotlechen) | 21 | unavailable | https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/codeinter-4th-place-solution (topic 518960) | DeepSeekMath-7B-RL, one copy per T4, TimeManager, 3 attempts for `[asy]` problems, small-integer down-weighting |
| 5 | Conor #2 | conor (conormacamhlaoibh) | 20 | unavailable | no writeup found (comment explains: unmodified fork, prize declined) | Unmodified fork of the Early Sharing Prize notebook |
| 6 | TheEighthDay | TheEighthDay (theeighthday) | 20 | unavailable | no writeup found | unknown |
| 7 | [Deleted] cefcc80d-454d-475d-b140-9f8db89e85e6 | [Deleted User] | 20 | unavailable | no writeup found | unknown (account deleted) |
| 8 | Vaishnavi Mudaliar | Vaishnavi Mudaliar (vaishnavimudaliar) | 20 | unavailable | no writeup found | unknown |
| 9 | kaiwu | kaiwu (kaiwu123) | 20 | unavailable | no writeup found | unknown |
| 10 | curiousgeorge | James C (jamesyc23), itswu.alex (itswualex), Abhi Sivaprasad (metacuriousgeorge), Tanner Hoke (tannerhoke), Zachary Chroman (zacharychroman) | 20 | unavailable | no writeup found | unknown |
| 11 | Hoang Tu Anh | Tu Anh Hoang (poiewq77) | 20 | unavailable | no writeup found | unknown |
| 12 | Quickpanda | Quickpanda, Doranap, JudyWatermelon | 20 | unavailable | no writeup found | unknown |
| 13 | zby | zby (herminw) | 19 | unavailable | no writeup found | unknown |
| 14 | aigood | aigood | 19 | unavailable | no writeup found | unknown |
| 15 | ZhujunXu | ZhujunXu (xzjcs1992) | 19 | unavailable | no writeup found | unknown |
| 16 | BU123 | xi7997, Akshat Gurbuxani (akshatgurbuxani), Mao Mao (maomaoai), Iddo Drori (iddodrori), Zack M123 (zackm123) | 19 | unavailable | no writeup found | unknown |
| 17 | Practice makes perfect | AbdurRafae (abdurrafae) | 19 | unavailable | no writeup found (author of the baseline notebook) | Presumably his own "Improved Code Interpretation" line (UNVERIFIED) |
| 18 | rl12 | ywm123, CJ Zhang (ruiyaoyang), Zelin Huang (blueshyy), EulKaggle, ALickerHasNoName | 19 | unavailable | no writeup found | unknown |
| 19 | Dileep Jayamal | Dileep Jayamal (dileepjayamal) | 19 | unavailable | no writeup found | unknown |
| 20 | aha | soso (tangtang1999) | 18 | unavailable | no writeup found | unknown |

Ties as given by the API: ranks 3 to 4 at 21; ranks 5 to 12 at 20 (eight teams); ranks 13 to 19 at 19; rank 20 at 18, and ranks 21 (Ali, asalhi) and 22 (naocanzouyihui) also at 18, so the 18-point tie continues past the cut. Within a tie, Kaggle orders by earlier submission.

**Writeups found: 4 of 20** (ranks 1 to 4), plus an explanatory comment from rank 5.

## 2. Per-team synthesis

### Ranks 1 to 3: facts the bank is missing
- **Numina (1st), from the comments of topic 519303.** Asked how they jumped from 16/50 to 27/50, Jia Li says the jump came from switching the Stage-2 data format from MMOS to ToRA (multi-step tool-integrated) (comment id2917230). The full fine-tuning corpus was "around 1M samples", built as a dataset first and then used for the competition. AIME problems 1983 to 2021 were in the training set; AIME 2022 to 2024 were held out as `AI-MO/aimo-validation-aime` (id2916299). The dataset and report were released at https://github.com/project-numina/aimo-progress-prize and https://huggingface.co/collections/AI-MO/numinamath-6697df380293bcfdbc1d978c (id2930688). Commenters note the later NuminaMath-72B release (not part of the Kaggle submission).
- **CMU_MATH (2nd).** Full training and data-collection code released after the writeup at https://github.com/AIMO-CMU-MATH/CMU_MATH-AIMO (comment id2916257). Nothing else new in the comments; the CV-LB correlation question (id2913049) went unanswered.
- **after exams (3rd).** Ablation from the comments (David Dinucu-Jianu, id2906161): a second submission that penalized only 0 plus the "number in the statement" rule scored **20** private versus **21** for the full small-number penalty, so the small-number rule was worth one problem. Most wrong small answers were also numbers in the statement. Geremie Yeo (not top 20) reports the same post-processing gave up to 6 to 7 CV points on lucky runs and that 25 of 61 AIME 2022 to 2024 problems had a single-digit modal answer. GPUs sit idle during code execution; they did not overlap generation and execution because more generations no longer improved validation; AsyncLLM was tried and abandoned near the deadline (id2906420). Speed was measured as generations per problem on the Kaggle validation runs, not with a profiler.

### Rank 4 – codeinter (21), writeup topic 518960
Code: https://www.kaggle.com/code/winter2003/4th-dual-gpu-reasoning-and-comparing-answers . Built on Abdur Rafae's "Improved Code Interpretation" notebook and Anren's reorganisation of it, using Anren's two prompts (one CoT, one write-code).

- **Data / CV:** no training. Local validation was the latest 50 AMC 12A problems from a 71k Art-of-Problem-Solving dump (Kaggle notebook "AIMO-24: Processor"), excluding problems that contain `[asy]`. They report it "showed good correlation with the public leaderboard". Local score ~21, pass@1 32, pass@2 23 (as written in the post; the pass@k labels are theirs).
- **Model and decoding:** `deepseek-math-7b-rl`, T=0.9, top_p 1.0, max 2,048 new tokens. One model copy on each T4 run in parallel with `ThreadPoolExecutor`, raising repetitions per problem from 21 to 30 to 40. If the first GPU to finish already has enough candidates, the slower GPU's attempts are cut and the candidates merged.
- **Time management:** a `TimeManager` class tracks time spent; when the running average per problem exceeds the remaining budget per problem, the next problem's attempts shrink to `0.8 x default x (budget / average)`. Problems with `[asy]` (geometry with a figure) get only 3 repetitions, because the model rarely solved them locally and any success looked like memorisation.
- **Tool loop:** at most 3 code-revision rounds per repetition (`while_limit` 6), because long contexts degraded output quality.
- **Voting:** small integers 0 to 5 get weight 0.25, all others 1. When the code output is not a valid answer (error, decimal, complex), the text answer is ignored instead of parsed, since it was "almost always incorrect".
- **Shake-up note:** different seeds moved the public score by about 3 points.
- **Distinguishing idea:** turning the 2xT4 box into two independent samplers plus an adaptive time budget, i.e. buying more votes rather than a better model.

### Rank 5 – Conor #2 (20)
No writeup. In topic 523125 (Igor Kotenkov asking for the 5th-place solution), conor explains that his selected private submissions were **unmodified forks of the Early Sharing Prize winner's notebook**, auto-selected by Kaggle, and that he **declined the prize money** because the work was not his own. He made the notebook public: https://www.kaggle.com/code/conormacamhlaoibh/aimo-fork-of-23-public-score/notebook ("fork of 23 public score"). JM's reply: this shows "where the high end of the variance was for that notebook" (https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/523125).

### Ranks 6 to 20
No writeup found on Kaggle (forum title sweep and author sweep over all 388 topics) or via web search. One-line notes:
- 7: account deleted; compare Geremie Yeo's thread on a vanished 3rd-place account (discussion/511414) – the link between the two is UNVERIFIED.
- 10 curiousgeorge: only post is "URGENT: Dataset Upload not Working" (discussion/514865), no method detail.
- 17 Practice makes perfect = Abdur Rafae, author of the "Improved Code Interpretation" baseline (https://www.kaggle.com/code/abdurrafae/improved-code-interpretation) and of "Notebook Scoring 20 points without Probing" (discussion/495362) and "Train vs LB score" (discussion/499464). His own final configuration is not documented.
- 21 Ali (asalhi), just outside the cut, posted "Speeding DeepSeekMath Inference from 8+ hours to 3 hours" (discussion/492578) and a P100 vs T4x2 comparison (discussion/493345).
- Others (6, 8, 9, 11 to 16, 18 to 20): nothing posted.

Below the top 20, writeups exist at 36th (topic 516966), 41st (516868), 50th (516909), 53rd (518543; also Zenn https://zenn.dev/morim34/articles/3a23b8c6a7abbb), 56th (516856), 96th on P100 (517137). Not read; useful only as context.

## 3. Technique frequency matrix

Only ranks 1 to 5 have any information; 6 to 20 are all `?` and omitted as columns. Y = used, N = not used, ? = unknown.

| Technique | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Backbone DeepSeekMath-7B family | Y (Base) | Y (RL) | Y (RL) | Y (RL) | Y (RL, via fork) |
| Any fine-tuning | Y (full FT) | Y | N | N | N |
| External human data (AMC/AIME/AoPS) for training | Y | Y | N | N | N |
| Synthetic data from GPT-4 | Y (ToRA traces) | Y (code solutions) | N | N | N |
| LoRA | N | ? | N | N | N |
| Learned verifier / reward model | N | Y (ORM) | N | N | N |
| Code execution (tool-integrated reasoning) | Y | Y | Y | Y | Y |
| vLLM | Y | Y (policy) | Y | ? | ? |
| Quantisation | Y (8-bit GPTQ) | N | N (FP16 KV noted) | ? | ? |
| Majority / weighted vote | Y | Y (ORM-weighted) | Y (penalised) | Y (weighted) | Y |
| Penalty on small/0 answers | N | Y (0 only) | Y | Y | ? |
| Penalty on numbers in the statement | N | N | Y | N | ? |
| Adaptive time management | N (fixed N, M by design) | N | N | Y | ? |
| Both T4s used as independent samplers | N (one vLLM batch) | Y (policy + ORM split) | ? | Y | ? |
| Built on Abdur Rafae public notebook | Y (credited) | ? | Y | Y | Y |
| Offline validation sets | Y (AMC, AIME, MATH L4/5) | ? (10 train problems reported) | Y | Y (AMC 12A) | N |

Counted summary (M = 4 teams with writeups, ranks 1 to 4): 4 of 4 used DeepSeekMath-7B and code execution with voting; 2 of 4 fine-tuned (both with GPT-4 generated solutions); 1 of 4 used a learned verifier; 3 of 4 down-weighted suspicious answers; 1 of 4 used quantisation; 1 of 4 used adaptive time management; 3 of 4 explicitly credit Abdur Rafae's notebook. Rank 5 is a pure fork of that notebook.

## 4. Consensus vs divergence

- **Table stakes:** DeepSeekMath-7B (RL or Base), Python execution in the loop, many samples per problem, majority vote. Every documented top-5 entry did this; rank 5 shows that the public baseline alone reached 20 private.
- **What separated the top 2 from 3 to 20:** training. Only Numina (29) and CMU_MATH (22) fine-tuned; ranks 3 to 5 (21, 21, 20) were inference-only on the same baseline. The score gap 29 vs 20 to 21 is the clearest evidence in this competition that data (TIR traces) beat inference tricks.
- **Contradictory or divergent findings:**
  - Time management: codeinter used an adaptive TimeManager; Numina deliberately used fixed N=48/M=4 because problems arrive in random order and early-stop tricks gave high variance (bank). Both writeups: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/codeinter-4th-place-solution and the Numina writeup.
  - More generations: after exams found more generations stopped helping on validation (comment id2906420) while codeinter's whole design maximised repetitions (21 -> 30 to 40).
  - vLLM: top 3 used vLLM; Geremie Yeo (outside top 20) could not get vLLM above 13 and suffered "mysterious timeouts", which Numina also reported (topic 515519).

## 5. Aggregated "what did not work"
- Larger bases (InternLM-20B, CodeLlama-33B, Mixtral-8x7B): slow, timeouts – Numina (bank).
- Integer-only Stage-2 filtering; RLOO RL; model merging (DARE/TIES/WARP); static KV cache + torch.compile on T4 – Numina (bank).
- Deduplicating policy data, adding GPT-4's wrong solutions, other positive:negative ratios – CMU_MATH (bank).
- bf16 on T4 via patched vLLM (unstable, never above 25 public); MLP in bf16 "very slow"; logits-processor code-injection – after exams (bank + comments).
- Penalising only 0 instead of all small numbers: one problem fewer (20 vs 21) – after exams comment id2906161.
- Long code-revision loops (more than 3 rounds) degrade output; parsing the text answer when the code fails – codeinter.
- Spending attempts on `[asy]` geometry – codeinter.

## 6. Shake-up note
Public scores are not retrievable, so movement cannot be measured. Indirect evidence: codeinter saw about 3 points of seed variance on the public LB; conor's unmodified fork of a "23 public" notebook landed at 20 private and 5th; eight teams tied at 20. The final-score thread calls it a big shake-up (discussion/516823, in the bank). ISAKA Tsuyoshi's public-20 submission failed on the private rerun (bank).

## 7. UNVERIFIED
- All public ranks and scores (API zeroed).
- Approaches of ranks 6 to 20; whether rank 17 (Abdur Rafae) used his own public notebook unchanged.
- Whether the deleted rank-7 account is the vanished account discussed in discussion/511414.
- codeinter's "pass1@ 32, pass2@ 23" figures are copied as written; their exact meaning is unclear.
