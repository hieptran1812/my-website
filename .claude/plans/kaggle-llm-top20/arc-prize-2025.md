# ARC Prize 2025 (ARC-AGI-2): top-20 addendum

- Slug: `arc-prize-2025` (competition id 91496, forum 6794744). Final board pulled from `competitions.LeaderboardService/GetLeaderboard` on 2026-10-28.
- Deep coverage of ranks 1 to 3 (NVARC, the ARChitects, MindsAI & Tufa Labs) and short notes on 4 to 6 already live in `.claude/plans/kaggle-llm-writeup-bank-part2.md`, H2 `## 7. ARC Prize 2025 (ARC-AGI-2)` (lines ~419 to 488). This file adds ranks 4 to 20, plus NVARC post-deadline facts the bank lacks.
- Board mechanics: ~240 hidden tasks, 120 scored on the public board and 120 on the private (Lonnie writeup). One task = 0.83 points, so the 5.00 to 6.67 band of ranks 4 to 20 is a spread of only 2 tasks. 1 submission/day; 12 h on 4x L4.

## Top-20 table

| Priv rank | Team | Members (display name (username)) | Private | Public rank (score) | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | NVARC | CPMP (cpmpml); ivan (sorokin) | 24.03 | 1 (27.64) | https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc (topic 651671) | gpt-oss-120b synthetic puzzles + Qwen3-4B full SFT + ARChitects-style TTT/DFS/rescoring (bank) |
| 2 | the ARChitects | Jan Disselhoff (insanitycheck); Daniel Franzen (dfranzen); David Hartmann (dvhrtm) | 16.53 | 2 (21.67) | https://www.kaggle.com/competitions/arc-prize-2025/writeups/the-architects-solution-2025 (topic 656966) | Masked-diffusion LLaDA-8B, 2D RoPE, recursive soft-mask refinement (bank) |
| 3 | MindsAI & Tufa Labs | Jack Cole (jcole75); Dries Smit (driessmit1); IsaiahP (pressman1); Mohamed Osman (moeosman); Michael Hodel (michaelhodel) | 12.64 | 3 (15.42) | https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution (topic 629790) | Pruned 660M CodeT5, >100M examples, TTT + AIRV (bank) |
| 4 | Lonnie | Lonnie (lonnieqin) | 6.67 | 343 (4.17) | https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place (topic 617939) | Fork of public ARChitects-2024 baseline chain; changed random seed |
| 5 | Guillermo Barbadillo @ Veridas | Guillermo Barbadillo (ironbar) | 6.53 | 5 (11.94) | https://www.kaggle.com/competitions/arc-prize-2025/writeups/exploring-the-combination-of-search-and-learn-for (topic 614521) ; https://ironbar.github.io/arc25/05_Solution_Summary/ | ARChitects-2024 with per-task TTT on 4 GPUs (scored); research on search-and-learn program synthesis (unscored) |
| 6 | ippeiogawa | ippeiogawa (ippeiogawa) | 6.25 | 7 (10.28) | https://www.kaggle.com/competitions/arc-prize-2025/writeups/lb10-00-with-small-change-to-2024architects (topic 615041) | ARChitects-2024 with per-task TTT, 24 epochs, DFS top_k=4; picked "consistent" subs |
| 7 | Ggmam1 | Ggmam1 (ggmam1); Wenbo Li (eweing); Zihan Xu (account99); CleAAAAAA (sunning11) | 6.25 | 55 (7.08) | no writeup found | unknown |
| 8 | rxe | Richard Emslie (albartrose) | 6.11 | 6 (10.42) | micro-writeup in notebook https://www.kaggle.com/code/albartrose/10-4-notebook-arcitects-tweaked (pointed to by topic 614586) | ARChitects-2024 with master/worker per-task TTT on L4s, tweaked DFS, skip-example collator |
| 9 | Abdelmoneim Desouki | AMD (amadesouki) | 5.83 | 33 (7.50) | no writeup found | unknown |
| 10 | Boiled-Sweet-Potato | Boiled-Sweet-Potato (boiledsweetpotato) | 5.69 | 48 (7.08) | no writeup found | unknown |
| 11 | MenghaoZhao | MenghaoZhao (menghaozhao); Alex GUAN (guanyuzhen) | 5.42 | 18 (7.92) | no writeup found | unknown |
| 12 | Ttkuma | Ttkuma (ttkuma) | 5.42 | 113 (6.25) | no writeup found | unknown |
| 13 | yarkhom | yarkhom (yarkhom) | 5.28 | 144 (5.83) | no writeup found | unknown |
| 14 | Hira Norm | Hira Norm (hiranorm) | 5.28 | 38 (7.08) | no writeup found | unknown |
| 15 | PoerMP | Xiaohan Chang (xiaohanchang); YmcjZkbr (zelinzl); Zhengye Li (zhengyeli); Tiancheng Zhang (alphaxyz); Suofei Liu (suofeiliu) | 5.28 | 31 (7.50) | no writeup found | unknown |
| 16 | vldmrc | vldmrc (vldmrc) | 5.00 | 132 (5.83) | no writeup found | unknown |
| 17 | The  Unforgiven7 | Yone (theunforgiven7) | 5.00 | 100 (6.25) | no writeup found | unknown |
| 18 | sanyul | sanyul (sanyul) | 5.00 | 23 (7.50) | no writeup; public notebook https://www.kaggle.com/code/sanyul/lb-5-83-baseline-from-1st-place-of-2024 | ARChitects-2024 fork with "active layer" freezing and LoRA tuning (public baseline others forked) |
| 19 | Hestu | Hestu (myeongkyuncho) | 5.00 | 54 (7.08) | no writeup found | unknown |
| 20 | 暗黑AGI | 暗黑AGI (boristown); haogood666 (haogood666); inszva (inszva) | 5.00 | 184 (5.42) | no writeup; public notebook https://www.kaggle.com/code/boristown/baseline-from-1st-place-of-2024 | ARChitects-2024 fork extended to 4 GPUs with seeded reproducibility (the root public baseline of 2025) |

Writeups found: 8 of 20 with a Kaggle writeup or a notebook write-up (1, 2, 3, 4, 5, 6, 8 + Barbadillo's site), plus 2 public baseline notebooks with change logs (18, 20). Forum scanned in full (215 topics, 12 pages).

## Per-team synthesis

### Rank 1 NVARC: facts the bank lacks
Sources: topics 614436, 615018, 652927.
- At the deadline the team did not pick the version they thought best: it had been stuck 10 h in the submission queue. The picked version ran out of time on the last ~5% of tasks and had high score variance; the unpicked one was "much more stable" and scored the same public LB in a late submission (CPMP and Ivan, topic 614436).
- A post-deadline late submission by Ivan scored **29.72** public (topic 615018).
- TRM follow-up (topic 652927): during the competition TRM scored 10% semi-private and 10.41% on public eval; 10 more days of training with the same recipe reached a best pass@2 of 18.05% on public eval, within Kaggle compute limits. CPMP estimates that would have been ~15% public / ~8% private on Kaggle.
- Why an LLM won (CPMP, 652927): "Kaggle compute limits forces us to focus on efficiency"; reasoning models and test-time code generation were tried and were too slow, so "the heavy lifting (generative LLM for puzzle understanding + code gen)" was moved into pretraining-time synthetic data.
- Giotto.ai led the public board "till the last minute" (topic 614436). Giotto.ai is absent from the final API board entirely (neither public nor private).

### Rank 4: Lonnie, 6.67 private (343rd public, 4.17)
Source: topic 617939; notebook https://www.kaggle.com/code/lonnieqin/lb-5-83-baseline-from-1st-place-of-2024 .
- Pipeline is the public chain ARChitects-2024 -> boristown's 4-GPU baseline -> sanyul's "LB 5.83" fork. Components: `model_runner.py` (load/train/infer, vocabulary shrinking), `selection.py`, `async_tools.py` (multi-GPU), `common_stuff.py` (config).
- Model: "Nemo Mini" (Mistral-NeMo-Minitron, the ARChitects' 2024 fine-tuned checkpoint), 4-bit quantised base, LoRA rank 32, only the first 32 layers trainable "to reduce overfitting". LR 1e-4 main, 1e-5 embeddings. Max length 4,224 train / 8,192 inference. 4 epochs, 240 steps.
- Augmentation: transposition, rotation, colour permutation, example shuffling. Per-task "priming" (brief retraining on each test task's demos), Turbo DFS, augmented scoring.
- His only change: random seed set to 19920627. He reports the same notebook scores 3.33% to 6.67% across seeds (4 tasks of 120), and argues the seed is a first-class hyperparameter when 120 tasks decide ranks. His writeup says public 344th; the API says 343rd.
- Writeup title says "5th place"; final API rank is 4th.

### Rank 5: Guillermo Barbadillo, 6.53 private (5th public, 11.94)
Sources: topic 614521; https://ironbar.github.io/arc25/05_Solution_Summary/ .
- Scored submission: ARChitects-2024 with three changes: TTT per task instead of over a group of tasks, efficient use of 4 GPUs, hyperparameter tuning. First team over 10% (11.94 in April 2025). Lost 5.4 points public to private.
- Research (none of it solved a private ARC-AGI-2 task): a deep-learning-guided program-synthesis loop that searches program space and adapts at test time by TTT on hindsight-relabelled programs ("search and learn").
  - Toy drawing DSL (draw_pixel/line/rectangle): trained on up to 5 calls, with hindsight-relabel TTT it solved tasks with 100+ drawn elements.
  - Own DSL (~40 primitives, 32 generators): solved no real ARC task; low data diversity blamed.
  - Qwen2.5-Coder with the BARC DSL; BARC induction model replication ~22% on ARC-AGI-1 eval at ~500 predictions (paper reports <15%), but 0.8% on ARC-AGI-2 eval.
  - At a fixed 512-prediction budget on ARC-AGI-1 eval: pure search 23.3%; 256 + 1 relabel epoch 26.0%; 128 x 4 rounds (3 epochs) 28.3%. About +5 points from search-and-learn.
  - RL: 22.3% -> 27% at 480 predictions, but training repeatedly collapsed (larger data, 8 to 128 generations, higher KL, lower grad norm, reward changes, repetition penalty, no 4-bit, LoRA rank changes did not fix it). RL-trained refinement 16.3% -> 17.8% after 17k steps.
- Did not work: synthetic task generators for a DSL model; diversity prompting for small models (worse than plain "solve the task"); prompt-based refinement with execution feedback (no gain vs independent sampling).

### Rank 6: ippeiogawa, 6.25 private (7th public, 10.28)
Source: topic 615041.
- Lightly modified ARChitects-2024 public notebook; "none of my other experiments managed to beat it".
- Training: per-task TTT, num_epoch = 24, num_train reduced to 3, batch 4, grad accumulation 2; one of two final subs removed background lines for faster training.
- Inference: n = 1, num_train 2, two subprocesses, DFS top_k = 4 to cut search time.
- Not all tasks finish in 12 h with these settings; lowering epochs "dropped a lot", so he kept 24.
- Submission choice: did not pick the best LB; picked the two with the most consistent behaviour across runs. Ended 7th public -> 6th private.

### Rank 8: rxe (Richard Emslie), 6.11 private (6th public, 10.42)
Source: notebook markdown at https://www.kaggle.com/code/albartrose/10-4-notebook-arcitects-tweaked ; topic 614586.
- ARChitects-2024 restructured into a master/worker client-server: the master hands tasks one by one to workers on the L4s, each doing TTT and inference per task. Inference ran on LoRA adapters (not merged), slower than the ARChitects train-and-merge path.
- Hyperparameters: 10 train epochs, 1 inference epoch, LoRA rank 32, alpha 32, min_prob 0.17, max_seq_length_train 6144.
- A collator that skipped examples in training ("not sure that was even working") and a DFS tweak that may skip 1 to 2 low-probability tokens and extends the probability budget for longer grids (aimed at ambiguous tasks). These two moved him from 8 to 10.4 public.
- A later full rewrite to support his own models dropped scores to 6 to 7% ("I obviously broke something"); the restored mid-May notebook is the final.

### Rank 18: sanyul, 5.00 (public notebook, no writeup)
Source: notebook https://www.kaggle.com/code/sanyul/lb-5-83-baseline-from-1st-place-of-2024 .
- Fork of boristown's baseline. Adds `num_active_layers` to train only the first N layers of Mistral-NeMo-Minitron and freeze the rest; LoRA/training settings tuned to fit 12 h.
- Reported variance: the notebook scores about 115 on the public ARC-AGI-2 evaluation set (units as written; presumably tasks/points of 120, UNVERIFIED) but five runs on the semi-private set gave 4.17, 5, 5, 5.42, 5.83. "The reason for this significant discrepancy remains unknown." Lonnie (4th) forked this notebook.

### Rank 20: 暗黑AGI (boristown and team), 5.00 (public notebook, no writeup)
Source: notebook https://www.kaggle.com/code/boristown/baseline-from-1st-place-of-2024 .
- The root public baseline of the 2025 board: ARChitects-2024 extended from 2 to 4 GPUs (4 training + 4 inference processes, even task split, 2x throughput), global seed 42 with per-GPU deterministic seeding and non-deterministic algorithms disabled, extensive visualisation and per-attempt accuracy stats. Team pledged to open-source all submissions.
- Documented components (from the notebook): tokenizer vocabulary shrinking and embedding resizing, 4-bit loading via transformers or unsloth, adapter merging.

### Ranks 7, 9 to 17, 19: no writeup
- 7 Ggmam1 (4-person team, 55th public -> 7th), 9 Abdelmoneim Desouki, 10 Boiled-Sweet-Potato, 11 MenghaoZhao, 12 Ttkuma, 13 yarkhom, 14 Hira Norm, 15 PoerMP, 16 vldmrc, 17 The Unforgiven7, 19 Hestu: nothing found in the forum or the team entries' writeup links. Given that ranks 4, 5, 6, 8, 18 and 20 are all ARChitects-2024 forks and that public notebooks of that lineage scored 4 to 6 private, it is likely but UNVERIFIED that most of these are too.

## Technique frequency matrix

Y = stated, N = stated or clearly implied not used, ? = no source.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Built on ARChitects-2024 public notebook | partly (their TTT/DFS/scoring ideas, own model) | Y (own work) | N | Y | Y | Y | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | ? | Y |
| Own large-scale pretraining off-Kaggle | Y (Qwen3-4B SFT, 32 H100 x 27 h) | Y (LLaDA-8B, 8x H100) | Y (TPUs, >100M ex.) | N | N (for scored sub) | N | ? | N | ? | ? | ? | ? | ? | ? | ? | ? | ? | N | ? | N |
| New synthetic data | Y (103k puzzles, gpt-oss-120b) | Y (ARC-GEN, ReARC, Heavy) | Y | N | N | N | ? | N | ? | ? | ? | ? | ? | ? | ? | ? | ? | N | ? | N |
| Per-task TTT | Y | Y | Y | Y ("priming") | Y | Y | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | ? | Y |
| LoRA at test time | Y (r256) | ? | N (full TTT) | Y (r32) | Y | Y | ? | Y (r32, a32) | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | ? | Y |
| DFS / enumerative decoding | Y | N (diffusion) | N (AIRV sampling) | Y | Y | Y (top_k 4) | ? | Y (tweaked) | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | ? | Y |
| Augmentation rescoring / AIRV vote | Y | ? | Y | Y | Y | Y | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | ? | Y |
| Non-autoregressive / recursive model | partly (TRM add-on) | Y (LLaDA) | N | N | N | N | ? | N | ? | ? | ? | ? | ? | ? | ? | ? | ? | N | ? | N |
| Program synthesis in scored sub | N | N | N | N | N | N | ? | N | ? | ? | ? | ? | ? | ? | ? | ? | ? | N | ? | N |
| Picked submission for stability, not best LB | N (forced by queue) | N (picked higher LB, lost) | ? | N | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Seed treated as hyperparameter | ? | ? | Y (seed ensembles) | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y (fixed for reproducibility) |

Counted summary (10 sources: ranks 1, 2, 3, 4, 5, 6, 8, 18, 20 + Barbadillo research site):
- 9 of 9 scored entries with sources used per-task test-time training.
- 8 of 9 used augmentation-based rescoring or voting (rank 2's candidate selection for the diffusion model is not described in the bank).
- 6 of 9 (ranks 4, 5, 6, 8, 18, 20) are forks of the ARChitects-2024 public notebook with no new pretraining; they land at 5.00 to 6.67 private.
- 3 of 9 (ranks 1 to 3) pretrained their own model off-Kaggle on large synthetic/augmented data; they are the only entries above 7.
- 0 of 9 used program synthesis in the scored submission (Barbadillo researched it; NVARC moved code generation into pretraining data).

## Consensus vs divergence

- Table stakes: per-task TTT + augmentation rescoring. Every documented entry has both.
- The single line separating top 3 from 4 to 20: **off-Kaggle pretraining compute on ARC-specific data** (NVARC: 32 H100 for 27 h on 3.2M samples; ARChitects: 175k steps on 8x H100; MindsAI: TPU months). Ranks 4 to 20 reused the public 2024 checkpoint. The gap between 3rd (12.64) and 4th (6.67) is larger than between 4th and 20th (5.00).
- Contradiction 1 (submission selection): ippeiogawa (6th, topic 615041) picked consistent submissions over best LB and gained a place; the ARChitects (2nd) picked the higher-public sub and left a 19.17 private on the table (bank); NVARC picked a time-limited sub because the stable one was stuck in the queue (topic 614436) and still won.
- Contradiction 2 (is variance noise or skill?): Lonnie (4th, 617939) argues the seed is a legitimate hyperparameter worth tuning; sanyul's notebook documents 4.17 to 5.83 across five identical runs and calls the discrepancy unexplained. Both agree the 120-task split makes rank 4 to 20 order largely noise.
- Contradiction 3 (reasoning/code at test time): Barbadillo (5th) spent the year on search-and-learn program synthesis and found +5 points on ARC-AGI-1 but 0 private ARC-AGI-2 solves; NVARC (1st, 652927) says reasoning and code generation were too slow under Kaggle limits and belong in pretraining.

## Aggregated "what did not work"

- Reasoning models and test-time code generation: too slow for 12 h on 4x L4 (1st, 652927).
- Bigger public-baseline rewrites: rxe's refactor dropped from 10.4 to 6 to 7 (8th).
- Lowering TTT epochs to fit time: score "dropped a lot" (6th).
- Own DSL + synthetic generators: no real ARC task solved (5th). BARC induction on ARC-AGI-2: 0.8% (5th).
- Diversity prompting for small code models: worse than plain prompting (5th).
- Execution-feedback refinement by prompting: no gain over independent samples (5th).
- RL for program synthesis: repeated collapse; no stable long-run fix (5th).
- Selecting the higher-public submission (2nd, bank) and selecting a sub that runs out of time (1st, 614436) both cost private points.
- Refinement, DPO on beam pairs, targeted ARC-2 data: not additive (3rd, bank).

## Shake-up note

Large, and mostly noise given 120 tasks per split. Biggest movers into the top 20 (public -> private): Lonnie 343 -> 4; 暗黑AGI 184 -> 20; yarkhom 144 -> 13; vldmrc 132 -> 16; Ttkuma 113 -> 12; The Unforgiven7 100 -> 17; Ggmam1 55 -> 7; Hestu 54 -> 19; Boiled-Sweet-Potato 48 -> 10. Teams that held: ranks 1 to 3 and Barbadillo (5 -> 5), rxe (6 -> 8), ippeiogawa (7 -> 6). Only explanations offered in writeups: the 1 to 2 task granularity and seed variance (Lonnie; sanyul notebook), and stable-submission picking (ippeiogawa). Giotto.ai, which led the public board until the last days (topic 614436), does not appear on the final API board; no reason found.

## UNVERIFIED

- Methods of ranks 7, 9 to 17 and 19; the inference that they are ARChitects-2024 forks.
- Why Giotto.ai is missing from the final board.
- sanyul's "about 115 on the public ARC-AGI-2 evaluation set" units.
- Whether NVARC's 29.72 late-submission change is the same as the "stable" unpicked version.
- Lonnie's public rank (writeup 344th vs API 343rd); API value used.
