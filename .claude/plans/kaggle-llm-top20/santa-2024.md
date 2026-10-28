# Santa 2024 - The Perplexity Permutation Puzzle: top-20 writeup synthesis

- Slug: `santa-2024` (competition id 88046, forum id 6256227)
- Existing deep coverage: `.claude/plans/kaggle-llm-writeup-bank-part3.md`, H2 `## Santa 2024 - The Perplexity Permutation Puzzle`
  (facts, metric, ranks 1 to 3 in depth, a one-line 4th/10th/11th, key public threads). This file does not repeat that detail.
- Leaderboard pulled from `competitions.LeaderboardService/GetLeaderboard` on 2026-10-28. The board is 100% public, so public rank = private rank for every team (no shake-up is possible).
- **Ties:** ranks 1 to 13 are all at **246.81784** (13-way tie at the optimum). Ranks are given in API order, which per the rules is submission-time order (tie broken by who submitted first). Ranks 16 and 17 are also tied at 246.88339. `last` = the team's `lastSubmissionDate` from the API (not the date the score was first reached; for tied teams the time of the scoring submission is what decides order).

## Top-20 table

| Priv rank | Team | Members (Kaggle usernames) | Score (private = public) | Public rank | Last sub / #subs | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|---|
| 1 (tie) | Hamiltonians | kibuna, nagiss, cnumber | 246.81784 | 1 | 2024-12-14 / 196 | https://www.kaggle.com/competitions/santa-2024/discussion/560560 (bank) | ILS + thresholded DFS local search, online-trained 1.3M NN prunes neighbours |
| 2 (tie) | Word Salad | zaburo, solverworld, danielphalen | 246.81784 | 2 | 2025-01-03 / 111 | 560540, 560533, 560565 (bank) | ILS with restricted k-opt, shuffle kicks, priority-queue / backtracking climbers |
| 3 (tie) | SantaMizers | siavrez, yuanzhezhou, asalhi, yyyu54 | 246.81784 | 3 | 2025-01-07 / 295 | 560620 (bank) | SA with adaptive-weight operator mix; stopwords+verbs+others sorted seed |
| 4 (tie) | Santa Master | onodera, daiwakun | 246.81784 | 4 | 2025-01-09 / 58 | https://www.kaggle.com/competitions/santa-2024/discussion/560536 | ILS: delete N words, re-insert by beam search, then local moves |
| 5 (tie) | Moș Noël | cpmpml, horeahoreastefan | 246.81784 | 5 | 2025-01-30 / 293 | https://www.kaggle.com/competitions/santa-2024/discussion/560597 (CPMP part) | Batched multi-point threshold-accepting SA with crossover to best, ATSP moves, "discounted perplexity" |
| 6 (tie) | Santa Seekers | chris62, cdeotte, shun2741, yukiohkawa | 246.81784 | 6 | 2025-01-19 / 115 | https://www.kaggle.com/competitions/santa-2024/discussion/560625 (Shun) + https://www.kaggle.com/competitions/santa-2024/discussion/560796 (Yuki.O); no team writeup linked in API | SA with insertion-only moves; two-alphabetical-block structure for id5 |
| 7 (tie) | Miner2024 | max2020 | 246.81784 | 7 | 2025-01-24 / 38 | https://www.kaggle.com/competitions/santa-2024/discussion/561091 | ILS wrapped around batched SA; learned bigram "distance matrix" prunes 1M candidates |
| 8 (tie) | KS | kyohei1 | 246.81784 | 8 | 2025-01-25 / 11 | no writeup found | - |
| 9 (tie) | Fu Ryo | fujiwararyo | 246.81784 | 9 | 2025-01-27 / 29 | https://www.kaggle.com/competitions/santa-2024/discussion/560601 | GA with Edge Assembly Crossover (ATSP), Gemma logits guide edge choice |
| 10 (tie) | B.A.G.E.L. 🥯 | titericz, deeper, lucienderubempre, egortrushin, nlztrk | 246.81784 | 10 | 2025-01-28 / 240 | https://www.kaggle.com/competitions/santa-2024/discussion/560531 (egortrushin's take) | Plain SA (swap + insert only); seed = stopwords + two random sorted blocks |
| 11 (tie) | Ryota | ryotak12 | 246.81784 | 11 | 2025-01-29 / 43 | https://www.kaggle.com/competitions/santa-2024/discussion/560542 | SA with insertion family (distance-weighted, alphabetical, variable-length); "magi" first for id3 |
| 12 (tie) | juvenjiang | mineqihang, juvenjiang | 246.81784 | 12 | 2025-01-30 / 21 | no writeup found | - |
| 13 (tie) | 本家Santa Master | cashfeg, yukari17, kkkggg, prgckwb | 246.81784 | 13 | 2025-01-30 / 107 | https://www.kaggle.com/competitions/santa-2024/discussion/560683 | see below |
| 14 | adakoda | adakoda | 246.82532 | 14 | 2025-01-31 / 70 | no writeup found | (246.82532 = the 28.57 id5 basin; see 4th place's table) |
| 15 | Rist Santa | takoihiraokazu, justy0300, tomfuj, ystsuji | 246.86686 | 15 | 2025-01-13 / 30 | no writeup found | - |
| 16 (tie) | Santa Magister | woosungyoon, annafabris, johnmichaelwu, martinapreusse, hikari30 | 246.88339 | 16 | 2025-01-31 / 190 | member posts 560541, 560674, 560713 (see below) | - |
| 17 (tie) | just team | atamazian, tetsuya3510, watanabekazuo, mahamhc | 246.88339 | 17 | 2025-01-28 / 94 | member post 560770 | - |
| 18 | BeerMind + Denden + Gilles | group16, denden12, sdtycc4434, isakatsuyoshi, waddlegassa | 246.88983 | 18 | 2025-01-29 / 188 | member posts 560550, 560557, 560564, 560567 | - |
| 19 | bono | kjoonha | 246.89181 | 19 | 2025-01-30 / 94 | no writeup found | - |
| 20 | zBere | deyury, dwchen, milanpeelman, gzlang, snorfyang | 246.92798 | 20 | 2025-01-28 / 147 | no writeup found | - |

Score ladder below the optimum (derived; every team shares ids 0 to 2 optimum, so the gaps are ids 3 to 5): 246.82532 is exactly the "28.574555 on id5" step that 4th place reports before its final "remove `and` from the function-word block" fix (560536). The 246.88 cluster is roughly one more decimal step on id5 (UNVERIFIED which id each of ranks 15 to 20 missed).

## Per-team synthesis

### Ranks 1 to 3: facts the bank lacks
- **1st Hamiltonians, NN details** (nagiss, in comments on https://www.kaggle.com/competitions/santa-2024/discussion/560560): the pruning network is **not a transformer** but a **ConvNet inspired by SE-ResNet**, chosen because "how nearby words are arranged seemed to be more important than their global relationships" (and nagiss was less experienced training transformers). It is not a Gemma replacement, only a filter that discards sequences that clearly worsen the score before Gemma sees them. **Pretraining turned out to be unnecessary; online learning during the search was enough.** Same thread: they had the optimum by mid-December and, after seeing equal scores lining up at the top, nagiss turned the GPU off; c-number kept his running. Nobody in the thread names the Rudolph Prize winner.
- A commenter on 560560 argues the key to id5 is **block rotation** (insert blocks into near neighbours), and that the 31.5 and 28.52 solutions are different local minima of the same basin (opinion, not a team writeup).

### 4th - Santa Master (onodera, daiwakun)
Source: https://www.kaggle.com/competitions/santa-2024/discussion/560536 ; gist https://gist.github.com/KazukiOnodera/ff74dd5d9171cddac773e03f7d3457f3
- **Search:** an ILS loop by daiwakun. (1) *Insert optimise:* delete N randomly chosen words and re-insert them with **beam search**. (2) *Local optimise* until no improvement: shuffle subsections, swap words, move words, exclude identical arrangements, remove duplicates.
- **Plain ILS results (before structure):** ids 0 to 2 optimal in 3 to 10 minutes; id3 stuck at 198.93 (10 h), id4 74.33 (1 day), id5 36.58 (1 day).
- **Finding 1 - "valid_length" words first:** a small set of words (reindeer, jingle, sleigh, gingerbread, peppermint, decorations, ornament, wreath, magi, stocking, chimney, fireplace) increase the run of well-predicted tokens if placed at the start. With **"magi" as the first word of id3**, their algorithm finds the optimum in ~3 hours. id0 and id1 start with "reindeer", id2 with "sleigh".
- **Function-words block:** grouping the function words (and, as, from, have, in, is, it, not, of, that, the, to, we, with, you) at the start lowers loss. id4's optimum is exactly *function words + content words*, cutting 50! to 14! + 36!; id5 to 20! + 80!. With this, id3 191.73 in 3 h, id4 67.54 in 6 h, id5 32.4.
- **Finding 2 (id5):** content words in alphabetical order lower loss, and *splitting* them into two sorted runs lowers it further, so id5 ~ function + sorted(content1) + sorted(content2) (search space 20! x C(80,40)). This reached 28.574555 (LB 246.82532). The final step: **"and" must be removed from the function-word block** (28.529695, LB 246.81784).
- Distinguishing idea: turning oracle observations into a **reduced, structured search space** rather than a smarter generic optimiser.

### 5th - Moș Noël (cpmpml, horeahoreastefan), CPMP part
Source: https://www.kaggle.com/competitions/santa-2024/discussion/560597
- Teammate Horea had the best id3 when they merged; CPMP's own method could not find id3's optimum without his input (word grouping). For other ids it finds the best score more than half the time from a **random shuffle**.
- **SA variant = threshold accepting:** accept a new solution if its score is below a global upper bound that is lowered from time to time (instead of the exp(-delta/T) test); "as effective and more efficient".
- **Multi-point search on GPU:** run N SA chains in parallel where N = the largest batch the metric fits (104 for id5 on an A100). Each round: modify every solution, score all in one batch, keep those within margin M of the best, and replace the others by a **crossover with the best solution** (the only GA element; purpose is to keep diverse paths alive).
- **Moves (from ATSP literature):** k-opt (split into k segments, shuffle, k up to ~10, with variants fixing some segments); remove-insert (random subset or contiguous subsequence of length k, the latter more effective; random or best-position insertion); and **"double root and stem"** ejection chains from a doubly-rooted ATSP paper.
- **Speed:** GPU at ~98% use; main lever a global text-to-score cache.
- **Discounted perplexity:** changes near the start move the score far more than changes at the end, so search drifts to editing the tail. Horea restricted moves to a prefix; CPMP instead normalised each position's logit by its running average, making early and late moves equally likely to be accepted.
- **Did not work:** Sinkhorn-relaxed permutation matrix trained by KL (like simjeg's idea), iterated ATSP on the logit matrix (failed completely), EMA of logits before Sinkhorn (better, still below local search), and **MCTS + RL** to generate sequences.

### 6th - Santa Seekers (chris62, cdeotte, shun2741, yukiohkawa)
No team writeup is linked in the API; three member posts.
- **Shun Fukuda** (https://www.kaggle.com/competitions/santa-2024/discussion/560625): SA for all ids, differing only in parameters. Neighbourhood eventually simplified to **random insertion of 1 word and of 2 to 5 consecutive words**; swaps worsened solutions and were rarely accepted, and insertions could emulate them. From random starts id5 reached ~32. After merging with Yuki (30.5 solution), they saw his permutation had **alphabetical sorting split into two segments**; fixing parts and re-running SA reached 28.5. Settings: temperature decreased slowly from ~0.1 to 0.2 (sic) so the search could wander in the low 30s; parallel chains with batch size ~8; dictionary cache; fixed words at the start and end (stop words at the start).
- **Yuki Ohkawa** (https://www.kaggle.com/competitions/santa-2024/discussion/560796): SA with a diverse move set (swap, shift, 2-opt, 3-opt, Or-opt); score gains from seeds, per-id initial temperature, decay rate, and **reheating** at a threshold. Alphabetical starting order key for ids 4 and 5.
- **Chris (chris62)** (https://www.kaggle.com/competitions/santa-2024/discussion/560553, "How I sped up finding good moves"): a per-string **LSTM "mini-net"** with word embeddings predicts the score (within ~2 points on id5; transformers did worse). Generate 10k to 100k moves, rank with the LSTM, score the top ~100 with Gemma: ~100x faster. It generalises poorly outside the basin it was trained on, so he retrained it whenever the score jumped to a new basin. Final pipeline finds all 6 optima in a couple of days on one RTX 3090. Independently the same surrogate idea as 1st place.

### 7th - Miner2024 (max2020, solo)
Source: https://www.kaggle.com/competitions/santa-2024/discussion/561091 ; code https://www.kaggle.com/code/max2020/santa24-7th-place-solution
- Learned metaheuristics from Sean Luke's *Essentials of Metaheuristics* during the competition. TL;DR: ILS around SA, with pruned distance matrices, caching and distributed parallel compute.
- **Batched SA:** tweaks = insertion of a block of length m1, swap of two blocks of length m2, local shuffle of length m3. All candidate indices precomputed and generated with NumPy (m1, m2 up to 20 gives ~500k neighbours; up to 65 gives ~1M). Each step scores k = 1,000 to 5,000 random candidates in a batch and accepts the best; if none improves, accept one of the top 10 with SA probability.
- **Repeat SA:** increase steps per temperature each round; early-stopping patience (3 failures early, 1 late).
- **ILS kick:** shuffle `perturb_rank` random positions (start 35); multiply by 2/3 on a new global best, 5/6 on a new local best, reset to 35 otherwise.
- **Cache:** words as integers, packed binary (685 to 121 bytes per sentence), persisted with **msgpack**.
- **Distinguishing idea - bigram distance-matrix pruning:** treat PPL as a TSP over word-to-word edges (102x102 with bos/eos). Initialise edges seen in ~20 recorded local optima low (e.g. 47), others high (100); later update each edge to the **mean perplexity of all good solutions containing it**. Ranking 1M candidates by summed edge cost takes a few hundred ms, ~100x search efficiency (keep top 5k/6 by distance plus k/6 random).

### 8th - KS (kyohei1, solo)
No writeup found; 11 submissions in total.

### 9th - Fu Ryo (fujiwararyo, solo)
Source: https://www.kaggle.com/competitions/santa-2024/discussion/560601 ; code https://gist.github.com/FujiwaraRyo/9a30b1cc7b615789de0aa264d2e63faa
- **GA with Edge Assembly Crossover (GA-EAX for ATSP,** Nagata & Soler 2012) for all six ids (ids 0 to 2 were finished by the initial SA).
- Loop: first generation from SA; children by EAX ("EAX-1AB" AB-cycle selection); children improved by local search and SA; selection per "SEL1" (replace a parent if a nearby child beats it); diversity penalty for similar individuals; eliminated individuals can come back as parents.
- **Gemma logits guide crossover repair:** since PPL is computed from logits, keeping each parent's logits gives a rough estimate of PPL when another word is placed next to a given word, used to choose edges when reconstructing permutations.
- Seeds: random for ids 0 to 3 (random 3-opt); id4 [stop words][non-stop words]; id5 [stop words][sorted group A][sorted group B] with random split, moves that shift words between groups while keeping alphabetical order, crossovers allowed to break groups. Other crossovers barely worked; only EAX did.

### 10th - B.A.G.E.L. (titericz, deeper, lucienderubempre, egortrushin, nlztrk)
Source: https://www.kaggle.com/competitions/santa-2024/discussion/560531 (egortrushin's personal take); code https://github.com/EgorTrushin/Kaggle-Competitions/tree/main/Santa24 ; mid-competition notebook https://www.kaggle.com/code/egortrushin/santa24-improving-sample-2
- **Only SA.** Keys: choice of starting sentence, understanding/controlling SA, compute.
- **Starting sentence decides the basin:** [stopwords] + [alphabetical rest] (~48 on id5) cannot reach gold. Building on woosungyoon's public two-block notebook, his final seed was **[stopwords without "and"] + [sorted block 1] + [and and and] + [sorted block 2]** with random split; this reached 28.5.
- **Only two moves at the end:** swap two words, and extract-and-reinsert one word (plus stopword-only versions early; cyclic rotation for a while, maybe for id2). More complex public neighbourhoods gave no gain and burned compute.
- **Three SA regimes by hyperparameters:** high-temperature exploration from a non-optimised start; "warm-up" reheat to escape the current minimum; low-temperature exploration around a minimum. Monitored acceptance rate. Recommends the online mini-book *Simulated Annealing Afternoon*.
- Compute mattered for fast experimentation, though a known path can be followed on Kaggle's free GPUs.

### 11th - Ryota (ryotak12, solo)
Source: https://www.kaggle.com/competitions/santa-2024/discussion/560542
- Started from a public 250.10105 submission (veniaminnelin notebook) where ids 0 to 2 were already optimal, so all compute went to ids 3 to 5. Pure SA, linear cooling.
- Moves (insert family, because swap disrupts too much): random insert; **distance-based probabilistic insert** (p(j) proportional to 1/|i-j|); variable-length insert (L=1..4 with p .4/.3/.2/.1); **alphabetical insert** (next to a word with the same first letter); partial permutation of a length-4 window.
- id3: random start reaches the known 197.5 trap; **fixing "magi" as first word** escaped to 191.7 (100k iterations/run, T 10 to 0.1, batch 8 to 32). id4: public seed + distance-based insert, 67.54 in one run (T 2 to 0.1). id5: 31.66, then 30.70 with combined moves; noticed three alphabetical blocks, re-seeded with 1 to 4 blocks, **two blocks best** (28.57); variable-length insert fixed multi-word units ("card game", "wrapping paper") to 28.52; the "and and and" needed to sit between blocks.
- Hardware: rented A100 for 10 days (1 GPU for 5 days, then 3).

### 12th - juvenjiang (mineqihang, juvenjiang)
No writeup found.

### 13th - 本家Santa Master (cashfeg, yukari17, kkkggg, prgckwb)
Source: https://www.kaggle.com/competitions/santa-2024/discussion/560683
- University of Electro-Communications team (two faculty, a master's student, an alumnus); GPU allocation managed centrally, progress tracked in W&B.
- Ids 0, 1, 2, 4 matched public best; stuck on 3 and 5. **id3:** fixing the first word as "magi", SA reached 195, then GA reached 191.73 (Jan 23). **id5:** stuck at 32 until Jan 29; only after running woosungyoon's accidentally-public two-block notebook for a day (31.5) did they understand the structure (49.5 h before deadline); then the ILS pushed to 28.8, SA and GA to 246.82532, and SA made the final step 25.5 h before deadline.
- **SA (kumanomi):** T 10 to 0.5/0.1, exponential, 100k steps; moves word_insert, swaps, 2-opt, 3-opt, Or-opt; **restart cut-off** if score not below 220 by step 60k; many fast ~30-min trials.
- **GA (prgckwb):** initial population = diverse existing local optima, crossover of two parents, SA-style mutations, switch to SA when stalled. id3 195.0 to 191.7, id5 28.8 to 28.5.
- **ILS "Sano Weapon" (yukari17):** exhaustive local search (3-part reorder O(N^2), adjacent and n-spaced swaps O(N), all insertions O(N^2)), accepting worsening up to a slowly lowered threshold X (threshold accepting, not probabilistic); a moderate kick applied to the head and tail; a tabu rule (never pass the same score twice). Took id5 from 31.5 to 28.8.

### 14th - adakoda (solo), 15th - Rist Santa (takoihiraokazu, justy0300, tomfuj, ystsuji)
No writeup found for either.

### 16th - Santa Magister (woosungyoon, annafabris, johnmichaelwu, martinapreusse, hikari30)
Member posts, no team writeup in API.
- **woosungyoon** (https://www.kaggle.com/competitions/santa-2024/discussion/560541; notebooks trial-1-beam-simulated-annealing, trial-2-simulated-annealing, alphabetical-sample-5): beam-stacked SA early; the **neighbour function is the most critical component** (must let you return easily to the good solution and reach the whole space). Temperature schedule = **max of a linear and a logistic curve** (logistic alone collapses early, linear alone is too cold late). His public **alphabetical-sample-5** notebook (letter-block sharding, SA moves words between blocks; explained in https://www.kaggle.com/competitions/santa-2024/discussion/559339) is what unlocked id5 for several teams (10th, 13th, 18th cite it; it was posted after an "accidental disclosure", 559317).
- **johnmichaelwu** (https://www.kaggle.com/competitions/santa-2024/discussion/560713): vanilla SA with simple TSP moves on the full permutation space ("free SA") for ids 1 to 4; id3's 192 needed manually fixed first words; id4 found after ~500k evaluations seeded from jazivxt's public solution. For id5, free SA converged to ~50, so they changed the search space to block-alphabetical structures; found double beats single, then were "trolled" by **triple alphabetical** beating double and spent their remaining effort there (best 28.92, sequence posted). Never tried "card game" as a fixed fragment. Peaked at LB 8.
- **hikari30** (https://www.kaggle.com/competitions/santa-2024/discussion/560674): SA + beam search explore near the initial solution but do not find the global optimum, so focus moved to finding better initial solutions.

### 17th - just team (atamazian, tetsuya3510, watanabekazuo, mahamhc)
Member post (atamazian): https://www.kaggle.com/competitions/santa-2024/discussion/560770. Reached id3's optimum with SA modelled on egortrushin's public notebook: hashed perplexity cache persisted as .pkl; moves = random swap, shift of one element, swap/shift adjacent pairs; T0 5, Tmin 1, 200 steps per temperature, exponential; chained runs with different seeds and manual word adjustments when stuck.

### 18th - BeerMind + Denden + Gilles (group16, denden12, sdtycc4434, isakatsuyoshi, waddlegassa)
Member posts: https://www.kaggle.com/competitions/santa-2024/discussion/560550 (denden), 560557 (isakatsuyoshi), 560564 (litlit), 560567 (Gassano).
- **denden:** SA with periodic **GA mixing of elite solutions**, batched scorer, cross-experiment cache, parallel SA, elite pool kept **diverse by edit distance**. Optimal ids 1 to 3; ineffective on 4 to 5. id5 hypotheses [stopword][A-Z][A-Z][A-Z][others] and 4-block variants did not break 28.5 (based on utm529fg's tutorial notebooks).
- **isakatsuyoshi:** Optuna over per-word priority parameters (optimal for id0 in hours, not scalable); fixing the sentence head and merging phrases ("visit workshop") to shrink the space (risk of excluding the optimum); evolutionary random playouts with alphabetical-order-aware inserts/swaps. Reached 8th a week before the end, finished 18th.
- **litlit:** SA with reheat to 50 to 90% of T0 after 10k non-improving iterations, per-id temperatures; block-pattern seeds for id5 did not help.
- **Gassano:** SA; tried patterns such as "[stopwords][free][a-z] yuletide [a-z][free]" ("yuletide"/"scrooge" as block connectors) and alternating stopword/non-stopword optimisation; none worked.

### 19th - bono (kjoonha, solo), 20th - zBere (deyury, dwchen, milanpeelman, gzlang, snorfyang)
No writeup found for either. (kjoonha posted "Solutions from past Santa competitions and Notebook materials for NLP", https://www.kaggle.com/competitions/santa-2024/discussion/548725, a resource list, not a solution.)

## Technique frequency matrix

Columns are private ranks. Y = used, N = not used / explicitly rejected, ? = unknown (no writeup, or the writeup is silent). Ranks 8, 12, 14, 15, 19, 20 have no writeup. For 6, 16, 17, 18 the cells merge the member posts. Context: there is no model training in this competition; every "technique" is search craft around a frozen Gemma 2 9B scorer.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Simulated annealing (any variant) as a main engine | N | N | Y | N | Y | Y | Y | ? | Y | Y | Y | ? | Y | ? | ? | Y | Y | Y | ? | ? |
| Iterated local search with explicit kicks | Y | Y | N | Y | N | N | Y | ? | N | N | N | ? | Y | ? | ? | N | N | N | ? | ? |
| Threshold accepting instead of exp(-d/T) | Y | N | N | N | Y | N | N | ? | N | N | N | ? | Y | ? | ? | N | N | N | ? | ? |
| GA / crossover | N | N | N | N | Y | N | N | ? | Y | N | N | ? | Y | ? | ? | N | N | Y | ? | ? |
| Beam search inside the loop | N | N | N | Y | N | N | N | ? | N | N | N | ? | N | ? | ? | Y | N | N | ? | ? |
| Remove-and-reinsert (insertion) moves | Y | Y | Y | Y | Y | Y | Y | ? | ? | Y | Y | ? | Y | ? | ? | ? | Y | ? | ? | ? |
| Block moves / k-opt without reversal | Y | Y | Y | Y | Y | Y | Y | ? | Y | N | Y | ? | Y | ? | ? | Y | N | ? | ? | ? |
| Cheap surrogate to prune candidates before Gemma | Y | N | N | N | N | Y | Y | ? | Y | N | N | ? | N | ? | ? | N | N | N | ? | ? |
| Global score cache | ? | Y | ? | ? | Y | Y | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | Y | ? | ? |
| Batched GPU scoring of many candidates | Y | ? | Y | ? | Y | Y | Y | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | Y | ? | ? |
| Structured seed: stopwords first + alphabetical block(s) (id4/id5) | ? | Y | Y | Y | ? | Y | ? | ? | Y | Y | Y | ? | Y | ? | ? | Y | ? | Y | ? | ? |
| Two (not one, not three) alphabetical blocks for id5 | ? | Y | ? | Y | ? | Y | ? | ? | Y | Y | Y | ? | Y | ? | ? | N | ? | N | ? | ? |
| Pin a first/last word for id3 ("magi" first, or last word) | ? | Y | ? | Y | ? | Y | ? | ? | ? | ? | Y | ? | Y | ? | ? | Y | ? | ? | ? | ? |
| Built on a public notebook/submission | ? | ? | ? | ? | ? | ? | ? | ? | ? | Y | Y | ? | Y | ? | ? | Y | Y | Y | ? | ? |
| Re-scored on Kaggle T4 to guard hardware drift | ? | Y | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |

Notes on cells: 1st's "threshold" is the DFS depth threshold; 9th's surrogate is Gemma's own logits reused as edge estimates, not a trained network; 7th's surrogate is a learned bigram edge-cost matrix; 6th's is chris62's LSTM; 3rd's operator set includes reversal operators (so "without reversal" is about block moves existing, not about excluding reversals).

**Counted summary (14 of 20 teams have at least one writeup or member post: ranks 1-7, 9-11, 13, 16-18):**
- 11 of 14 used SA in some form as a main engine (all except 1st, 2nd, 4th, which used ILS).
- 13 of 14 used insertion-style moves (remove a word or block and re-insert); it is the one move almost everyone converged on. 2 (6th Shun, 11th) explicitly said insertion beats swap; 2nd measured that delete+insert perturbs PPL less than arbitrary swaps.
- 10 of 14 explicitly used a stopwords-first and/or alphabetical-block seed or search space for id4/id5; 7 of them say **two** sorted blocks were the shape that reached 28.52 (16th and 18th explored three or more blocks and ended below gold).
- 6 of 14 report pinning a word (usually "magi" first) to escape id3's 197.5 trap.
- 4 of 14 built a cheap surrogate (ConvNet, LSTM, bigram matrix, Gemma logits) to filter neighbours; **3 of those 4 are in the top 9**, and both learned-network surrogates (1st, 6th) report ~10x to ~100x fewer Gemma calls.
- 4 of 14 used GA/crossover; 2nd explicitly reported GA failed (no good crossover), 9th made it work only with EAX, the ATSP-specific crossover.

## Consensus vs divergence

**Table stakes (nearly every top-20 team):**
- Treat the puzzle as an asymmetric TSP over words with a frozen LLM as the cost oracle. No team trained a generator; every finisher used local search.
- Batched, correctly padded Gemma scoring plus a cache; cheap moves that disturb perplexity little (insertion, short block moves).
- Exploit structure discovered from the oracle: function words first, content words in alphabetical runs, a "valid_length" word first, specific multi-word units ("card game", "wrapping paper").
- ids 0 to 2 were solved early by everyone (id0 brute force); the whole board is decided on id3 (197.5 trap vs 191.73) and id5 (32 / 30.x / 28.57 / 28.52 basins).

**What separated the top of the tie from 14 to 20:** finding id5's last steps (two sorted blocks; "and" moved out of the function block or "and and and" between the blocks; multi-word units). 14th sits exactly on the 28.57 basin (246.82532). Within the 13-way tie, rank is **time-to-optimum** (Hamiltonians last submitted 2024-12-14; ranks 5 to 13 all finished in the last week of January). The teams that got there earliest had either a general search strong enough to find structure unaided (1st: ILS + NN pruning, optimum by mid-December; 5th: from random shuffles for all but id3; 6th/chris62: all six optima in a couple of days on one RTX 3090 with the LSTM filter) or found the structure analytically (4th). Teams from 10th down mostly report reaching id5 only after woosungyoon's public alphabetical-block notebook (10th, 13th explicitly).

**Contradictory findings:**
- *Swap moves:* hurt and rarely accepted (6th Shun, https://www.kaggle.com/competitions/santa-2024/discussion/560625; 11th, 560542) vs one of only two moves kept to the end (10th, 560531).
- *Many operators vs few:* diverse operator sets mattered (6th Yuki, 560796; 3rd's 12 to 15 adaptive operators, 560620) vs complex neighbourhoods "burn compute" and add nothing over swap + insert (10th, 560531).
- *SA:* "not methodical enough", dropped (2nd, 560540) vs the main engine of most of the tie (3rd, 6th, 7th, 10th, 11th).
- *GA:* failed for lack of a good crossover (2nd, 560540) vs GA-EAX on ATSP worked (9th, 560601) and GA delivered id3's 191.7 and id5's 28.5 (13th, 560683).
- *Beam search:* no non-myopic evaluation, did not work (2nd, 560540) vs beam re-insertion of deleted words as the core of 4th's ILS (560536).
- *Surrogate architecture:* ConvNet (SE-ResNet style) chosen over transformers for locality (1st, comment on 560560) vs LSTM, transformers "didn't do as well" (chris62, 560553); both work.
- *Restricting the search space:* 4th and 16th shrink to structured spaces deliberately; 16th's johnmichaelwu concludes the full-permutation teams were right because the restricted triple-block space could not contain the optimum (560713).

## Aggregated "what did not work"
- Simulated annealing and late-acceptance hill climbing as main engine - 2nd (560540).
- Branch-and-bound with exact prefix losses (bound too loose) - 2nd.
- Beam search without a non-myopic evaluation - 2nd. (Contrast 4th.)
- GA with generic crossover - 2nd; "almost only EAX worked" among crossovers - 9th.
- Multi-armed bandit over random permutations; scoring 14M random id5 permutations - 2nd.
- Gradient-based relaxation: Sinkhorn permutation matrix + KL, iterated ATSP on the logit matrix, EMA-of-logits Sinkhorn - 5th (560597).
- MCTS + RL sequence generator - 5th.
- Optuna over per-word priority parameters (fine for 10 words, does not scale) - 18th isakatsuyoshi (560557).
- Swap moves (too disruptive) - 6th Shun, 11th.
- Complex public neighbourhood operators - 10th.
- Free SA on id5 from scratch (converges ~50) - 16th; [stopwords]+[single alphabetical block] seed cannot reach gold - 10th.
- Three or more alphabetical blocks for id5 - 16th (best 28.92), 18th denden/litlit.
- Phrase merging / fixing the sentence head to shrink the space (risks excluding the optimum) - 18th isakatsuyoshi; "sleigh of the magi" is not in the id3 optimum.
- "yuletide"/"scrooge" connector patterns, alternating stopword/content optimisation - 18th Gassano.
- Surrogate generalising across basins (had to retrain per basin) - 6th chris62.
- Logistic-only or linear-only temperature schedules - 16th woosungyoon.

## Shake-up note
None by construction: the leaderboard is 100% public, so the API's public and private ranks are identical for all 20 teams. The "shake" that mattered was **chronological**: 13 teams tied at 246.81784 are ordered by submission time. 18th (isakatsuyoshi) reports being 8th a week before the end and 16th (johnmichaelwu) reports peaking at LB 8; both were pushed down as other teams reached the optimum in the final days (ranks 5 to 13 have last-submission dates of 2025-01-19 to 2025-01-30). The publication of woosungyoon's alphabetical-block notebook (559317/559339, "Accidental Disclosure") in late January is cited by 10th and 13th as the unlock for id5, i.e. a public leak compressed the final ordering. The gold line equalled the optimum score ("gold == best score", 560713; 13th: "the same score was exactly the border for the gold medal").

## UNVERIFIED
- Rudolph Prize recipient (no primary source read names it; Hamiltonians held 1st from mid-December by the API dates, so likely them).
- Which id(s) ranks 15 to 20 missed; the 28.57 basin explains only rank 14's 246.82532 (derived arithmetic).
- Whether 1st, 3rd, 5th, 7th used an alphabetical-block seed for id5 (writeups are silent; 7th's distance matrix and 5th's random-start claim suggest not necessarily).
- 6th Shun's "decreased the temperature from around 0.1 to 0.2" is quoted as written (likely means a narrow 0.1 to 0.2 band).
- No writeup located for ranks 8, 12, 14, 15, 19, 20 (forum list of 187 topics checked by title; GitHub/Zenn not searched).
