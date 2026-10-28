# Kaggle LLM Competitions: Writeup Bank, Part 3 (2024-2026 "LLM as metric / tool / target")

Built 2026-10-27. Scope: Santa 2024, Drawing with LLMs, Jigsaw Agile Community Rules, Make Data Count,
Google Code Golf 2025, NeuroGolf 2026, Name That LLM (ongoing), plus competitions discovered by search.

## How this bank was verified (read first)

- **Primary source = Kaggle's internal JSON API** (anonymous session + XSRF cookie, helper
  `.claude/scripts/kaggle-api.sh`): `CompetitionService/GetCompetition` (dates, teams, reward, metric name,
  submission limits, public %), `PageService/ListPages` (Overview, Evaluation, Data, Rules/Prizes, Timeline,
  Code Requirements text), `LeaderboardService/GetLeaderboard` (public + private rank, score, team names, members,
  linked writeup topic id), `DiscussionsService/GetTopicListByForumId` (all topics, paged, sorted locally by votes)
  and `GetForumTopicById` (full markdown of writeups and threads; writeups live under `forumTopic.writeUp.message`).
  Raw JSON is in `/private/tmp/claude-501/kaggle-bank-p3/`.
- Every rank, score and team name comes from the leaderboard API. Every technical claim is paraphrased from the
  team's own post or repo README; numbers are copied, not recomputed. GitHub READMEs fetched via the GitHub API.
  One external blog (NVIDIA recap) and one Zenn article are cited where used.
- `GetForumTopicById` is rate-limited (HTTP 429 after ~20 rapid calls); topics were fetched one every ~15 s.
- **Two competitions are ongoing** (Name That LLM, Gemma 4 Developer Agent): their sections describe the state on
  2026-10-27 and must be refreshed before publishing.
- Scope note for writers: in several "LLM" competitions the LLM was the *judge or objective* (Santa 2024, Drawing with
  LLMs, You Can't Please Them All), a *data engineer* (Make Data Count, Deep Past), or an *agent doing the work*
  (Code Golf, NeuroGolf). Each section says which.
- **UNVERIFIED** marks anything not confirmed from a primary source.

### Summary table
| Slug | Title | Dates | Teams | Prize | Metric |
|---|---|---|---|---|---|
| `santa-2024` | Santa 2024 - The Perplexity Permutation Puzzle | 2024-11-21 → 2025-01-31 | 1,514 | $50,000 | mean Gemma-2-9B perplexity (lower) |
| `drawing-with-llms` | Drawing with LLMs | 2025-02-25 → 2025-05-27 | 1,309 | $50,000 | SVG Image Fidelity: F-β(β=2) of VQA×OCR-penalty and CLIP aesthetic |
| `llms-you-cant-please-them-all` | LLMs - You Can't Please Them All (discovered) | 2024-12-03 → 2025-03-04 | 1,692 | $50,000 | judge-disagreement formula |
| `make-data-count-finding-data-references` | Make Data Count - Finding Data References | 2025-06-11 → 2025-09-09 | 1,282 | $100,000 | global F1 on (article, dataset, type) |
| `jigsaw-agile-community-rules` | Jigsaw - Agile Community Rules Classification | 2025-07-23 → 2025-10-23 | 2,445 | $100,000 | mean per-rule AUC |
| `google-code-golf-2025` | NeurIPS 2025 - Google Code Golf Championship | 2025-07-31 → 2025-10-30 | 1,142 | $100,000 | Σ max(1, 2500 − bytes) |
| `deep-past-initiative-machine-translation` | Deep Past Challenge - Akkadian → English (discovered) | 2025-12-16 → 2026-03-23 | 2,674 | $50,000 | √(BLEU·chrF++) |
| `nvidia-nemotron-model-reasoning-challenge` | NVIDIA Nemotron Model Reasoning Challenge (discovered) | 2026-03-16 → 2026-06-15 | 4,185 | $106,388 (cash + DGX Sparks) | accuracy, greedy vLLM, 7,680 tokens |
| `neurogolf-2026` | The 2026 NeuroGolf Championship | 2026-04-15 → 2026-07-15 | 2,963 | $50,000 | Σ max(1, 25 − ln(params + memory)) |
| `autonomous-agent-prediction-beta` | Autonomous Agent Prediction (Beta) (discovered) | 2026-07-07 → 2026-08-06 | 570 | merchandise | AUC of agent's own mini-competition |
| `gemma-4-developer-agent` | Gemma 4 Developer Agent (discovered, ongoing) | 2026-09-23 → 2026-12-02 | 3,932 so far | $65,000 (+$10k small-model) | % issues resolved |
| `name-that-llm` | Name That LLM (ongoing) | 2026-10-14 → 2027-01-13 | 3,523 so far | $50,000 (10 × $5k) | balanced accuracy, 18 classes |

Sections below appear in the order they were written (scope list first, discoveries after), and a
"Considered and excluded" table closes the file.

## Santa 2024 - The Perplexity Permutation Puzzle

**Why it belongs in an LLM series:** the LLM is not the model you train, it is the *objective function*. Every
candidate is scored by a forward pass of Gemma 2 9B, so the whole competition is "black-box combinatorial
optimisation where one function evaluation costs an LLM forward pass".

### Facts
- URL: https://www.kaggle.com/competitions/santa-2024 (competition id 88046, forum id 6256227)
- Dates: 2024-11-21 → 2025-01-31 (entry/team-merger deadline 2025-01-24). Host: Kaggle.
- Task: 6 rows of Christmas words (lengths 10, 20, 20, 30, 50, 100 words; ids 0-5). Submit a permutation of
  each row's words. Words cannot be added, removed or changed; only re-ordered.
- Metric (Evaluation page): average perplexity over the 6 sequences, computed with **Gemma 2 9B**, lower is better.
  `PPL(seq) = exp( (1/T) * Σ_t −log p(token_t | tokens_<t) )`, score = mean of the 6 PPLs. Metric notebook:
  https://www.kaggle.com/code/metric/santa-2024-metric/
- Leaderboard: 100% public (no hidden test). Final LB = public LB. Ties are broken by submission time.
- Prizes: $50,000. 1st $12k, 2nd $10k, 3rd $10k, 4th $8k, plus a **$10k "Rudolph Prize"** for the team that
  held 1st place on the LB for the longest time. (Rules/Prizes page.)
- Teams: 1,514 teams, 1,720 competitors, 23,729 submissions. Max team size 5; 100 submissions/day.
- Hardware: no runtime limit on the optimisation (it ran on participants' own GPUs; 1st place thanks Kaggle for
  "no restrictions on program execution time"). Only the scoring runs on Kaggle (T4s, per the 2nd-place
  solverworld post, which notes some orderings flipped polarity between T4 and RTX 4090).
- Metric changes (Ryan Holbrook, staff, https://www.kaggle.com/competitions/santa-2024/discussion/547676):
  the launch metric diverged with batch size; it was fixed to score each instance individually, then fixed
  twice more (label shift and sequence-boundary tokens, final fix 2024-11-25) with a full rescore each time.
- Exploit closed early: "Hacking the LB [fixed]" (c-number, https://www.kaggle.com/competitions/santa-2024/discussion/547505)
  — inserting ~83 spaces between words gave a super-low perplexity; patched.

### Final leaderboard (top 12; from LeaderboardService)
13 teams finished at the optimum-looking **246.81784**; ranks among them are submission-time order.

| Rank | Team | Members | Private (=public) | Approach (one line) | Writeup |
|---|---|---|---|---|---|
| 1 | Hamiltonians | kibuna, nagiss, c-number | 246.81784 | Iterated Local Search + DFS local search; online-trained 1.3M-param NN prunes neighbours before Gemma scoring | https://www.kaggle.com/competitions/santa-2024/writeups/hamiltonians-1st-place-solution (repo https://github.com/Lgeu/santa2024) |
| 2 | Word Salad | zaburo, solverworld, danielphalen | 246.81784 | ILS with restricted k-opt (no flips) + custom shuffle kicks; priority-queue hill climber | https://www.kaggle.com/competitions/santa-2024/writeups/word-salad-2nd-place-solution |
| 3 | SantaMizers | siavrez, yuanzhezhou, asalhi, yyyu54 | 246.81784 | Simulated annealing with adaptive operator weights; start = sorted stopwords + sorted verbs + sorted others | https://www.kaggle.com/competitions/santa-2024/writeups/santamizers-3rd-place-solution |
| 4 | Santa Master | onodera, daiwakun | 246.81784 | ILS: beam-search re-insertion of N deleted words + local moves; function-words-first structure | https://www.kaggle.com/competitions/santa-2024/discussion/560536 |
| 5 | Moș Noël | cpmpml, horeahoreastefan | 246.81784 | (CPMP part) | https://www.kaggle.com/competitions/santa-2024/discussion/560597 |
| 6 | Santa Seekers | chris62, cdeotte, shun2741, yukiohkawa | 246.81784 | (Shun Fukuda part) | https://www.kaggle.com/competitions/santa-2024/discussion/560625 |
| 7 | Miner2024 | max2020 | 246.81784 | – | topic 561091 |
| 9 | Fu Ryo | fujiwararyo | 246.81784 | – | topic 560601 |
| 10 | B.A.G.E.L. | titericz, deeper, lucienderubempre, egortrushin, nlztrk | 246.81784 | random-sampled alphabetical blocks start → 28.5 on id5 | https://www.kaggle.com/competitions/santa-2024/discussion/560531 |
| 11 | Ryota | ryotak12 | 246.81784 | random/“distance-based probabilistic” insert; "magi" first word for id3 | https://www.kaggle.com/competitions/santa-2024/discussion/560542 |

Per-id best scores (from 4th-place table, which matches the final 246.81784 mean): id0 469.77, id1 424.38,
id2 298.93, id3 191.73, id4 67.54, id5 28.52.

Hamiltonians' `lastSubmissionDate` in the LB API is 2024-12-14, i.e. they had their final submission in
about three weeks; the other 12 teams matched it later (2nd's last submission 2025-01-03, 3rd's 2025-01-07,
4th's 2025-01-09). **UNVERIFIED:** who won the Rudolph Prize (likely Hamiltonians given the dates, but no
primary source read says so).

### 1st place — Hamiltonians (kibuna, nagiss, c-number)
Source: writeup (points to repo) + https://github.com/Lgeu/santa2024 README (en/ja).
- **Framework: Iterated Local Search (ILS).** Loop: from the best solution, 3 times: `kick` then `local_search`;
  keep if better.
- **Kicks:** swap 2 random words; or remove a random contiguous segment and re-insert each removed word at a
  random position.
- **Local search = depth-first search with a threshold.** For each neighbour (take a small number of words out
  and insert them elsewhere): if it improves, take it and restart; else if its score is below
  `get_threshold(best, depth)` (≈ `best*1.1` at shallow depth, `best` deep), recurse into it. So it can walk
  through a slightly-worse intermediate to reach an improvement — a bounded "look past one bad step".
- Extras: node-count early termination, efficient batched Gemma scoring.
- **Key trick: NN-based pruning.** A ~1.3M-parameter network (≈1/5000 of Gemma 9B) screens neighbours, so fewer
  than 1/10 of candidates get an exact Gemma score. The NN is trained **online during the search**, "in a manner
  similar to reinforcement learning" (repo has `pretrain.py` too). This is the surrogate-model idea: cheap
  learned proxy, expensive oracle only for the shortlist.
- Repo layout: `code_nagiss` (main, pretrain, evaluation), `code_kibuna`, `code_ethylene` (experiments).

### 2nd place — Word Salad (danielphalen, zaburo, solverworld; three posts)
Sources: https://www.kaggle.com/competitions/santa-2024/discussion/560540 (overview),
https://www.kaggle.com/competitions/santa-2024/discussion/560533 (zaburo), https://www.kaggle.com/competitions/santa-2024/discussion/560565 (solverworld).
- **Framing as TSP-like.** Two steps: (1) a local neighbourhood to descend; (2) a way to kick out of local minima.
  id0 was solved by brute force (publicly known).
- **Neighbourhoods** (danielphalen): (1) delete one word and reinsert (restricted 3-opt, smallest perplexity
  disturbance); (2) move a phrase (all 3-opts without reversal); (3) swap two phrases (double-bridge).
  Reversals excluded because word order reversal wrecks perplexity. On an RTX A6000, nbhd 3 for ids 1-3,
  nbhd 2 (sometimes 3) for id4, nbhd 1 (sometimes 2) for id5.
- **Restricted k-opt** (zaburo): no flips; limit the length of the *moving* subsequence but allow long
  stationary ones; used (k=3, max_moving=5) and (k=4, max_moving=1). Full neighbourhood of id5 fits in ~18 min
  on a 4090.
- **Kicks:** double-bridge and random 2-opts early; later the key one was "shuffle a random subsequence of
  n=5 or 10 words" (break up adjacent-word groups). For id5 a problem-specific kick swapped words between the
  `(stop words)(optimised words)` prefix and the nearly-sorted suffix (code in post); found 28.5.
- **id5 structure:** `(sorted stop words)(sorted non-stop words)` stalls at ~32.x; breaking into
  `(some stop words)(some optimised words)(other stop words)(nearly-sorted other words)` broke the barrier.
  They seeded id5 using solutions of ids 1, 3, 4 (id5's words are made from them).
- **id3:** pin the *last* word (zaburo) / test each word in the last position → escaped 197.5 → 191.5.
- **Backtracking climber** (danielphalen): on a local minimum, jump back a random number of steps in the search
  history and pick a good unexplored solution; kick if the basin seems huge. Notebook:
  https://www.kaggle.com/code/danielphalen/santabasinclimberv7-final
- **Priority-queue search** (solverworld): pop lowest-perplexity node, generate neighbours, keep up to 2
  improved (or 1 worse) nodes, drop a node after 5 moves without improving on its best ancestor, 15-min rounds,
  kicks = 5-word shuffle or 6-opt. Finds id2's 298.9 within 24 h on a 4090. Cached scores in a pickle.
  Code: https://github.com/dgrunberg/Santa-2024-public. Measured that "cycler" (delete+insert) perturbs PPL
  less than arbitrary swaps.
- **Hardware drift guard:** re-scored best solutions and all neighbours in a Kaggle T4 notebook to make sure no
  neighbour was better under the official hardware.
- **Did not work:** simulated annealing and late-acceptance hill climbing (not methodical enough; SA's
  temperature schedule bad for long intermittent runs), phrase-level brute force, **branch-and-bound** (prefix
  losses are exact because later words don't change earlier token losses, and next-word loss can be read off one
  forward pass's logits, a 100× saving on id5; but the optimistic bound was too loose to prune), multi-armed
  bandit over random permutations, beam search (no non-myopic evaluation function), GA (no good crossover), and
  scoring 14M random id5 permutations (best ~500 vs 28.5).

### 3rd place — SantaMizers (siavrez, yuanzhezhou, asalhi, yyyu54)
Source: https://www.kaggle.com/competitions/santa-2024/discussion/560620 (written by asalhi).
Title: "Sorted stopwords, Sorted verbs, and Sorted Other words are all you need!"
- **Approach 1 — SA with a weighted candidate generator** (notebook
  https://www.kaggle.com/code/asalhi/3rd-place-sa-weighted-candidate-generator-sol1): 12 operators (swap,
  reverse, remove-insert, remove-insert-n, rotate, shift, swap-adjacent, circular shift, reverse-all,
  scramble-except-first, block shift, block swap). Operator weights adapt: best operator ×1.2, others ×(0.9 to
  1.0) by relative score, then normalise. Word importance = 1/PPL of the single word. Accepts texts with *equal*
  best score (ties happen on A100) to escape dead ends; restart from best/shuffled text on stagnation. Batched
  metric. Best for ids 0-4, 30.x on id5.
- **Approach 2** (https://www.kaggle.com/code/siavrez/3rd-place-sa-weighted-candidate-generator-sol2): SA
  starting from 2-word swap, adding ~15 block/shift/interleave/bridge operators, **cyclic cooling**, a bounded
  acceptance band, chaining runs with random operator subsets, top-k parallel paths. Reached 28.5x on id5.
- **Starting point mattered most:** alphabetical sort (best for id2); stopwords-sorted-then-rest-sorted (ids 3, 4);
  random restarts in parallel (id3); and **sorted stopwords + sorted verbs + sorted others** (best seed for 3, 4
  and especially 5, even though its initial score is not the lowest).

### Key threads and notebooks
| Thread / notebook | Takeaway |
|---|---|
| Chris Deotte, "How To Use Batch_Size>1 and get Correct Perplexity Score" (92 votes) https://www.kaggle.com/competitions/santa-2024/discussion/548249 | Right-pad the tokenizer and mask pad tokens in the loss; left padding made the label shift drop a pad instead of `<bos>`. Batching = the main throughput lever. |
| Ryan Holbrook (staff), "Metric Revision and Rescore" https://www.kaggle.com/competitions/santa-2024/discussion/547676 | Batch-size-dependent metric fixed three times; full rescores. |
| c-number, "Hacking the LB [fixed]" https://www.kaggle.com/competitions/santa-2024/discussion/547505 | Whitespace injection fooled the tokenizer-level PPL; patched. |
| simjeg, "Brute forcing" https://www.kaggle.com/competitions/santa-2024/discussion/547564 | 10/20/20/30/50/100 words ≈ 14/24/25/36/59/117 tokens; 10! is brute-forceable, 20! is not. |
| simjeg, "Quantum Santa - or how to use gradient descent" https://www.kaggle.com/competitions/santa-2024/discussion/549535 (notebook https://www.kaggle.com/code/simjeg/relax-it-s-santa) | Relax permutations to bistochastic matrices, optimise with REINFORCE; cites Gumbel-Sinkhorn. Original but not used by winners. |
| Giba (titericz), "Sorting sample 6 scores 53.46" https://www.kaggle.com/competitions/santa-2024/discussion/550287 | Alphabetical sort is a strong prior for the 100-word bag; seed of every top id5 solution. |
| Tanaka, "Accelerating ppx calculation…" https://www.kaggle.com/competitions/santa-2024/discussion/552346 | `attn_implementation="sdpa"` ≈ +20% throughput on A6000. |
| ONODERA, "Each Optimal(?) Score has been revealed!" https://www.kaggle.com/competitions/santa-2024/discussion/556784 | Per-id target scores posted mid-January, telling everyone where the floor was. |
| Maksim Sinitsyn, "Publishing top solution is dangerous" https://www.kaggle.com/competitions/santa-2024/discussion/555881 | 100%-public LB + shared optimal files flatten the board; 13-way tie at the top shows it. |

### Shake-up
None possible (100% public LB). The tie at 246.81784 makes **time-to-optimum** the real ranking.

### Transferable lessons
1. **An LLM can be a frozen objective; optimisation craft beats ML craft.** Every winner used TSP-style local
   search (ILS, k-opt, kicks), not learned generators. (2nd: https://www.kaggle.com/competitions/santa-2024/discussion/560533)
2. **Make the oracle cheap before making the search smart:** correct batching (548249), SDPA (552346), and a
   learned 1.3M-param surrogate that filters >90% of Gemma calls (https://github.com/Lgeu/santa2024).
3. **Neighbourhood design should respect the model's inductive bias:** no reversals (causal LM), small moves
   disturb PPL least (solverworld plot, https://www.kaggle.com/competitions/santa-2024/discussion/560565).
4. **Structure discovered from the oracle is a seed:** function-words-first + alphabetically sorted content blocks
   (https://www.kaggle.com/competitions/santa-2024/discussion/560536, 560620).
5. **Pin a variable to escape a basin:** fixing the first ("magi") or last word unlocked id3's 191.73
   (560542, 560533).
6. **Verify on the scoring hardware:** fp ordering on T4 vs 4090/A100 can flip near-ties (560565, 560620).

### UNVERIFIED
- Rudolph Prize recipient. 5th/6th place content not summarised beyond titles.

## Drawing with LLMs

**Why it belongs:** text → SVG code generation, judged by a VLM (PaliGemma) + CLIP aesthetic model. The twist the
series should make explicit: **the winning solutions did not use an LLM to write SVG at all.** They used a
text-to-image diffusion model, vectorised its output, and then attacked the VLM judge's OCR/preprocessing gap.
It is the canonical "model-based metric gets gamed" case.

### Facts
- URL: https://www.kaggle.com/competitions/drawing-with-llms (id 89659, forum 6478774)
- Dates: 2025-02-25 → 2025-05-27 (entry/merger deadline 2025-05-19). Host: Kaggle.
- First competition that required **Kaggle Packages**: submit a `class Model` with `predict(description) -> svg`.
  Internet off; CPU or GPU notebook ≤ 9 h; each SVG within 5 minutes; all SVGs within 9 h.
- Data: ~500 test descriptions of everyday objects/scenes (≤200 chars, avg ~50), ~a dozen categories. `landscapes`,
  `abstract`, `fashion` appear in train/public/private; other categories are unique to public vs private. No
  people, brands or names. train.csv has only the 3 shared categories.
- SVG constraints: ≤10,000 bytes, element/attribute allowlist, no CSS `style`, no raster data or external refs.
- Metric "SVG Image Fidelity" (final version, from 2025-03-20): render with cairosvg → preprocessing (random
  crop/resize, JPEG compression, median filter, FFT low-pass, bilateral filter; per 3rd place's re-implementation)
  → **VQA**: PaliGemma answers 4 image-specific yes/no or multiple-choice questions per description (TIFA
  style); → **OCR penalty**: PaliGemma OCR on a lightly processed image, exponential penalty if > 4 characters
  are read; VQA = OCR penalty × TIFA score → **Aesthetic**: CLIP aesthetic predictor (improved-aesthetic-predictor)
  → per-image score = weighted harmonic mean of VQA and aesthetic with β=2 favouring VQA
  (`F_β = (1+β²)·V·A / (β²·A + V)`), averaged over images. Metric notebook: https://www.kaggle.com/code/metric/svg-image-fidelity
- Public/private split: public LB = 52% of test (`leaderboardPercentage` 52).
- Prizes: $50,000 (1st $12k, 2nd $10k, 3rd $10k, 4th $10k, 5th $8k). Winner licence Apache 2.0.
- Teams: 1,309 teams, 1,725 competitors, 38,770 submissions; 5 submissions/day.
- **Metric changes (three versions):** launch metric was exploited (rendered text, edge-pixel attacks; 1st place
  says its old-metric image scored 1.014 with aesthetic > 1.2 using 237 optimised squares). Ryan Holbrook announced
  a VQA(yes/no) + aesthetic harmonic-mean metric (https://www.kaggle.com/competitions/drawing-with-llms/discussion/566429,
  rolled out in https://www.kaggle.com/competitions/drawing-with-llms/discussion/566981, ~45 min to score one
  submission), then the final TIFA + preprocessing + OCR-penalty version
  (https://www.kaggle.com/competitions/drawing-with-llms/discussion/567872, details in comment #3154907, 2025-03-20).
  Even the final version was beaten by hiding text from OCR but not from VQA (see winners).

### Final leaderboard (top 10)
| Rank | Team | Members | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|---|
| 1 | new team name | conormacamhlaoibh, huanligong, kenyeungtech | 0.79058 | 0.79038 (1) | SSD-1B ×64 images → 6-colour polygon SVG → hidden prompt text + OCR decoy + searched aesthetic artifacts → pick max aesthetic | https://www.kaggle.com/competitions/drawing-with-llms/writeups/new-team-name-1st-place-solution |
| 2 | move on | takuji, arsenypoyda, samson8, brianwilliamzhao, guoooooooss | 0.77776 | 0.77187 (3) | Pure OCR attack: dashed-line mask hides text from OCR, reveals it after preprocessing; "Zoe" aesthetic letters | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581023 |
| 3 | CP GMs | cnumber, daiwakun | 0.76947 | 0.77523 (2) | FLUX.1-schnell → vtracer → SigLIP rerank → diffvg optimisation through a differentiable copy of the preprocessing; no text hack | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581024 |
| 4 | Kevin Clark | kevlark | 0.74314 | 0.74108 (6) | SD3.5 Medium LoRA trained with DRaFT on reward models → heuristic SVG → diffvg; best-of-n by VQA proxy | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581108 |
| 5 | tomirol | tomirol | 0.73168 | 0.74255 (4) | SDXL-turbo → VTracer + DiffVG; optimised ~3.5 KB "aesthetic patch" lifts aesthetic to ~0.7 | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581128 |
| 6 | Grand Painting Artists | tascj0, liushuzhi, kapenon | 0.72474 | 0.74194 (5) | – | none listed |
| 7 | JJ | jerry4083 | 0.72123 | 0.71321 (15) | – | none listed |
| 8 | Psiilu | philippsinger, ilu000 | 0.71961 | 0.73005 (11) | – | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581095 |
| 9 | linrock | linrock | 0.71473 | 0.71924 (13) | SDXL LoRA fine-tune | https://www.kaggle.com/competitions/drawing-with-llms/discussion/581061 |
| 10 | BERA | benbla, raki21 | 0.71389 | 0.73824 (7) | – | topic 581213 |

### 1st place — "new team name" (conor, huanligong, kenyeungtech)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581027 ; notebook
https://www.kaggle.com/code/conormacamhlaoibh/new-team-name-1st-place-solution
- **Pipeline:** (1) 64 images from Segmind **SSD-1B**; (2) each → SVG; (3) add OCR decoy, hidden prompt text and
  aesthetic artifacts; (4) re-render through a re-implementation of the metric's preprocessing and compute the
  aesthetic score; (5) return the SVG with the highest aesthetic score.
- **Diffusion speed-ups:** Tiny AutoEncoder for SDXL (taesdxl) instead of the VAE; DPM-Solver multistep with 12
  steps; DeepCache (`cache_interval=2`, `cache_branch_id=0`); 384×384, guidance 8; 32 images per T4 on both GPUs
  (~15 s multi-threaded, ~30 s single-threaded because of a Kaggle Packages multithreading bug).
  Prompt: "flat vector illustration, flat color blocks, hard color boundaries, solid fills, clipart style… vector
  art of {description}", with a negative prompt against photorealism, gradients, text, frames.
- **Vectorisation:** downsample, quantise to 6 colours, OpenCV contours per colour, simplify to polygons, rank paths
  by size/centrality/simplicity, add until 6,000 bytes (~2.5 s per batch of 64).
- **OCR exploits:** (a) three visible characters top-left to stop OCR hallucinating; a search over 3-letter strings
  found "ZOK" gave the best aesthetic with no VQA loss. (b) **Hidden prompt text:** the description (stopwords
  removed) drawn as compressed SVG paths under a 10×10 repeating pattern of 1-px coloured squares at opacity .1.
  The text is invisible to the OCR pass (raw/lightly processed image) but appears after the heavier preprocessing
  that feeds VQA and aesthetic.
- **Layout:** the diffusion SVG shrunk to 40%, top centre; hidden text below; plus searched artifacts: 5×4 patch of
  coloured squares (4 variants tested per image), green "KUZ", green "3Q3", grey "SSQ", green "FRANCE XK ARTS".
  Artifacts were found by genetic algorithm, hill-climbing or manual trial.
- Validation: avg aesthetic ~0.563, VQA ~0.88. Fallback text-only SVG for timeouts.
- **Did not work:** adding CLIP/VQA scores to the selection rule; >64 images (though it looked good on private);
  averaging aesthetic across multiple preprocessing draws; other artifact placements.

### 2nd place — "move on" (takuji, guo dashuai and team)
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581023 ; notebook
https://www.kaggle.com/code/takuji/2nd-place-solution-ocr-attack
- Started with Stable Diffusion → SVG like everyone, stalled for a week, then went all-in on the metric.
- Reasoning: VQA weighs 2× aesthetic (β=2), the only way to very high VQA is text, and the **OCR penalty is
  computed before the enhancement while VQA is computed after it**. So make text invisible before, visible after.
- A carefully designed **dashed-line mask** over the text; only the uncovered letters are recognised by OCR.
  Letters and masks hand-designed by takuji to pack a lot of content in ~4,000 letters; layout code keeps each
  word inside one column. Reached 0.652, then all further work went into the aesthetic score.
- Brute-forced (on an A100) all letter/number combinations placed outside the mask, scored over 15 prompts:
  "Zoe" won. Then hearts, position tweaks, doubling short prompts, random inserted words selected by aesthetic.
- DiffVG failed for them because the augmentations were non-differentiable, so the final is heuristic.
  A last-minute text-format rewrite kept 2nd ahead of 3rd.

### 3rd place — CP GMs (c-number, daiwakun): "VQA/AES = 0.81/0.64", no text hack
Source: https://www.kaggle.com/competitions/drawing-with-llms/discussion/581024 ; final notebook
https://www.kaggle.com/code/cnumber/fork-of-flux-baseline-75c113?scriptVersionId=242056669
1. **FLUX.1-schnell** bitmaps from 4 prompt templates ("{description}", "Beautiful illustration of …", "Simple svg
   illustration of … colorful background", "… background of minimal flat vector illustration of colorful layered
   big mountains"), 2 images each → 8 candidates.
2. **vtracer** to SVG; binary search over input PNG resolution to land under the size cap.
3. Rank with **SigLIP so400m-patch14-384** (the image encoder inside PaliGemma 2) vs the text; keep top 2; 10-step
   short optimisation; keep the one with best aesthetic.
4. **pydiffvg optimisation, 200 iterations, Adam, cosine LR with warmup**, over path control points, fill/stroke
   colours and widths. Loss = −aesthetic score − SigLIP (siglip2-base-patch16-224) similarity to the target
   bitmap + MSE to the target bitmap.
5. **ImageProcessorTorch:** a differentiable (or straight-through) copy of the metric's preprocessing — random
   crop/resize, JPEG (diff_jpeg), median filter, FFT low-pass, bilateral filter — applied before the losses, with
   synced crop seeds for the MSE target.
6. Compression to fit 10 KB: drop opacity, hex colours, rounding, merge same-fill paths, relative/H/V commands,
   binary search on render size, prune paths. A small fixed signature path is added.

### 4th and 5th (short, useful for the "fine-tune the generator to the judge" angle)
- 4th, Kevin Clark (https://www.kaggle.com/competitions/drawing-with-llms/discussion/581108; code
  https://github.com/clarkkev/svg-diffusion): **SD3.5 Medium without T5**, LoRA trained with **DRaFT-LV**
  (backprop the reward through the diffusion sampler). 1,000 synthetic prompts + questions from GPT-4o. Rewards:
  HPSv2, PickScore, competition aesthetic, VQA proxy ("Does the image show {prompt}?" — faster and better than
  the real question set), and −LPIPS(image, svg(image)) to make images easy to vectorise. +~10% aesthetic, +~5%
  VQA, ~+0.03 LB. Then heuristic SVG (after Rich Olson) refined with diffvg on L1; best-of-n within ~60 s by VQA
  proxy; OCR decoy from Rich Olson.
- 5th, tomirol (https://www.kaggle.com/competitions/drawing-with-llms/discussion/581128): SDXL-turbo → SVG, plus an
  optimised ~3.5 KB **aesthetic patch** that by itself lifts average aesthetic to ~0.7; a VQA proxy picks the SVG.
- 12th, Rock (https://www.kaggle.com/competitions/drawing-with-llms/discussion/581034; code
  https://github.com/yanyanhuang/12th-Place-Solution-for-Kaggle-Drawing-with-LLMs): SD3.5M fine-tuned with
  **Flow-GRPO** directly on the metric.

### Key threads and notebooks
| Thread | Takeaway |
|---|---|
| Bilzard, "Text Rendering: OCR-Exploit [LB=0.305]" https://www.kaggle.com/competitions/drawing-with-llms/discussion/565396 | Draw the description as glyph paths; the judge was pretrained on OCR pairs, so text "matches" text. Started the arms race. |
| Bilzard, "Inversion Attack" https://www.kaggle.com/competitions/drawing-with-llms/discussion/566160 | White-box gradient search for images near the text embedding; judges can score faithful images lower than adversarial ones. |
| Bilzard, "Text not present" attack https://www.kaggle.com/competitions/drawing-with-llms/discussion/567449 and "Vulnerability Report: Edge Pixel Attack to the New Metric (v15)" https://www.kaggle.com/competitions/drawing-with-llms/discussion/569825 | Each metric version got a reported exploit within days. |
| David Austin, "Edge pixel attack - Aesthetic 0.7214" https://www.kaggle.com/competitions/drawing-with-llms/discussion/567487 | Aesthetic predictor can be driven by border pixels. |
| Rich Olson, "Stable Diffusion -> SVG -> Scoring Metric [LB .507]" https://www.kaggle.com/competitions/drawing-with-llms/discussion/567633 | The diffusion→SVG baseline that most top teams built on; his OCR-decoy notebook https://www.kaggle.com/code/richolson/let-s-defeat-ocr-easy-lb-boost was reused by 4th. |
| Zhuang Jia, "Comparing the SVG generation capabilities of mainstream LLMs" https://www.kaggle.com/competitions/drawing-with-llms/discussion/569847 | Direct LLM SVG on train: Claude-3.7-Sonnet 0.670, Qwen2.5-14B 0.659, GPT-4o 0.643 — far below the ~0.79 diffusion+hack winners. |
| taku_sid, "Fine-tune Your LLM with GRPO and Custom Reward Functions" https://www.kaggle.com/competitions/drawing-with-llms/discussion/565980 | The "honest" LLM-RL path; not what won. |
| Ravi Ramakrishnan, "A nearly perfect leaderboard dynamics" https://www.kaggle.com/competitions/drawing-with-llms/discussion/581017 | Top-10% stable public→private; gold-zone teams averaged ~159 submissions. |

### Shake-up
Small. Top 5 public ranks 1,3,2,6,4 → private 1-5. 2nd and 3rd swapped.

### Transferable lessons
1. **A learned judge is an attack surface.** Every metric version got exploited; the final winners exploited the
   gap between the OCR pass and the VQA pass. (581027, 581023)
2. **"Use the LLM" is not the best tool for the job.** Diffusion + vectorisation beat direct LLM SVG by a wide
   margin. (569847 vs LB)
3. **Re-implement the judge's preprocessing, then optimise through it** (differentiably, if you can). (581024)
4. **Train the generator on the judge's rewards** (DRaFT, Flow-GRPO) and add a "convertibility" reward. (581108, 581034)
5. **Best-of-n with a cheap proxy of the metric** is a strong, simple inference-time lever. (581027, 581108, 581128)
6. Hosts: model-based metrics need red-teaming before launch; three rescores mid-competition. (566429, 567872)

### UNVERIFIED
- 6th-7th approaches; the full text of 8th/9th writeups was fetched but not summarised.

## Jigsaw - Agile Community Rules Classification

**Why it belongs:** the cleanest example of "LLM as a fine-tuned binary classifier with test-time training".
Every top solution LoRA-fine-tuned 7-14B instruct LLMs **inside the 12-hour submission notebook** on the
labelled few-shot examples that ship *in test.csv*, and read P(Yes) from one next-token logit.

### Facts
- URL: https://www.kaggle.com/competitions/jigsaw-agile-community-rules (id 94635, forum 7097115)
- Dates: 2025-07-23 → 2025-10-23 (entry/merger 2025-10-16). Host: Jigsaw/Conversation AI.
- Task: given a Reddit comment (`body`), a `rule` and a `subreddit`, predict the probability the comment broke
  the rule. Each row also carries `positive_example_{1,2}` and `negative_example_{1,2}` (labelled comments that
  do / do not violate that rule). train.csv has only **2 rules**; test has **additional unseen rules** (number not
  disclosed). Data drawn from the Chandrasekharan & Gilbert moderated-comments corpus; labels from a small
  "dev set" process. Public/private split 30:70 (API `leaderboardPercentage` 29).
- Metric: **column-averaged AUC** — AUC computed per rule, then averaged (metric "94635_Jigsaw_Rules_AUC").
- Code competition: CPU/GPU notebook ≤ 12 h, internet off; public external data and models allowed.
- Prizes: $100,000 (1st $35k, 2nd $25k, 3rd $15k, 4th-8th $5k each). Winner licence Apache 2.0.
- Teams: 2,445 teams; 5 submissions/day.
- **Hidden-rule probing:** c-number proposed reading the hidden rule text by encoding characters into the score
  (https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/604092); shlomoron organised a
  "Public joint probing initiative" (one submission = one character,
  https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/607118) and the full text of all
  6 test rules was posted in "The rules are revealed"
  (https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/607941): public = no advertising,
  no legal advice; private = no financial advice, no medical advice, no promotion of illegal activity, no spoilers.
- **Train-on-test grey area:** neibyr's "Train on test data" notebook (LB 0.878 with Qwen2.5-0.5B) used the
  test-set positive/negative examples as training data; discussed as a possible rule grey area
  (https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/598099). It stayed legal and
  became the foundation of every winning solution.

### Final leaderboard (top 10)
| Rank | Team | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|
| 1 | Guanshuo Xu (wowfattie) | 0.92930 | 0.93476 (1) | Online LoRA SFT of 6 models (Qwen3-14B/8B/4B, Qwen2.5-14B, Llama3.1-8B, Ettin-400M) on test examples; loss only on Yes/No token; per-rule rank-normalised ensemble | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/1st-place-solution |
| 2 | ^^ (andyy12345, yaoqingjin, trudih, yfr999, hoatha) | 0.92923 | 0.93316 (6) | Label cleaning by majority vote, drop subreddit, 4 unsloth 4-bit LoRA models (phi-4-14B, Qwen3-14B/8B, Qwen2.5-7B) on 2 GPUs | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/2nd-place-solution |
| 3 | Sergio Manuel Papadakis (socom20) | 0.92871 | 0.93335 (5) | Pretrained LoRAs + test-time training; LLM last-token embeddings → LightGBM/XGB/SVC; "slow" LLM + "fast" embedding ensembles | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613324 |
| 4 | Dream Team (rusalmighty) | 0.92860 | 0.93382 (3) | SALSA single-pass logit classifier, Generalized CE (q=0.955), soft labels, 2-stage pseudo-labels 4B → 14B | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/4th-place-solution |
| 5 | Flash (ivanisaev, benbla, raki21, cnumber, kishanvavdara) | 0.92855 | 0.93472 (2) | Diverse ensemble: DeBERTa NLI, embeddings+LR/kNN, TTT Llama3.1-8B/Qwen3-8B (r=256 LoRA), Qwen-32B inference-only | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/5th-place-solution-diverse-ensemble |
| 6 | ducnh279 | 0.92781 | 0.93249 (12) | Deep Mutual Learning (online distillation) between Qwen3-14B/8B/Qwen3Guard-4B | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/6th-place-solution |
| 7 | ktr (kcotton21) | 0.92760 | 0.93281 (9) | – | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613215 |
| 8 | Reddit comment moderators | 0.92759 | 0.93370 (4) | 3× Qwen3-14B + Llama2-13B + bge-base, QLoRA via unsloth, seed ensembles | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/writeups/8th-place-solution-qwen3-14b3-llama2-13b1-b |
| 9 | lllleeeo + mango | 0.92747 | 0.93315 (7) | – | https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613185 |
| 10 | Nikita Babych | 0.92745 | 0.93244 (13) | – | none listed |

### 1st place — Guanshuo Xu (solo)
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613305 ; code
https://www.kaggle.com/code/wowfattie/notebookc4e4a9206d
- **Validation = public LB.** Reasoning: host said the split is random, and 30% of a large test set gives low
  variance; local CV on 2 rules can't represent the 4 new rules. All models were fine-tuned online (in the
  submission run).
- **Data:** starts from neibyr's train-on-test notebook (https://www.kaggle.com/code/neibyr/lb-0-878-train-on-test-data-qwen2-5-0-5b):
  the test-set positive/negative examples become training rows. Changes: use each LLM's native chat template;
  **drop the subreddit column before deduplication** ("the performance improvement is significant").
- **Fine-tuning:** started with plain transformers+PyTorch (0.931 public early with Qwen3-4B-instruct-2507,
  Llama3.2-3B, Phi-4-mini-instruct), then **Unsloth** to fit 14B models in 16 GB. Test-set examples upsampled once,
  1 epoch (= 1 epoch on train, 2 on test examples). **Loss masked to the single Yes/No token** — chat templates add
  tokens like `<think>\n\n</think>\n\n` around the answer; excluding them speeds training and makes convergence
  consistent across models so one hyperparameter set works for all.
- **Inference:** P(yes) = sigmoid(logsumexp(Yes-variants) − logsumexp(No-variants)) over first-token variant sets
  {Yes, YES, Y, yes, True} vs {No, NO, N, no, False} plus space-prefixed forms; sort by length; `forward()` only
  (no `generate`).
- **Post-processing:** per-rule rank → normalise to [0,1] → ensemble (0.929 → 0.931 public early on), because
  the metric is per-rule AUC.
- **Models (public / private):** Qwen3-14B 0.9297/0.9239; Qwen2.5-14B 0.9287/0.9232; Qwen3-8B 0.9272/0.9236;
  Qwen3-4B-instruct-2507 0.9258/0.9198; Llama3.1-8B 0.9257/0.9202; Ettin-400M 0.8991/0.8944;
  **weighted ensemble 0.9344/0.9293**. Larger was better; Qwen3 beat other families.

### 2nd place — team "^^"
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/614279
- Start point: mks2192's "llama3.1-8b instruct training one epoch" notebook proving Unsloth online LoRA fits Kaggle.
- **Local CV** (unlike 1st): StratifiedKFold on `rule_violation`, 10% holdout, mean AUC over 3 seeds; used because
  public LB swung a lot and identical notebooks sometimes crashed.
- **Label cleaning:** same (body, rule) pair sometimes had opposite labels; key on `body || rule` (ignoring
  subreddit), majority-vote the label, fill, then deduplicate. Fewer but cleaner rows suit 1-epoch LoRA.
- **Drop subreddit** entirely (shortcut/leakage feature; the same behaviour judged differently by subreddit).
  Prompt: system "Decide if the Reddit comment violates the rule. Reply strictly with Yes or No." user
  `Rule: … / Comment: …`.
- Models: **phi-4 (14B), Qwen3-14B, Qwen3-8B, Qwen2.5-7B-Instruct**, all bnb 4-bit via
  `unsloth.FastLanguageModel`, LoRA, `trl.SFTTrainer`, 1 epoch; score = Yes/No next-token logit difference;
  weighted fusion.
- Engineering: two model groups on two T4s in parallel (`CUDA_VISIBLE_DEVICES`), 11 h total runtime; deliberately
  use only 80-90% of memory (moderate batch + grad accumulation) because Kaggle reruns of the same notebook
  sometimes threw "Notebook Threw Exception"/timeout.

### 3rd place — Sergio Manuel Papadakis (solo)
Source: https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613324 ; notebook
https://www.kaggle.com/code/socom20/t73t74-ens-7b-9b-unsloth-qwen14bx4-qwen8bx1-qwen7
- **Test-Time Training** for the hidden rules; validated on public LB; fill the 12-hour window with models.
- **Pretraining step offline:** LoRAs pretrained 1 epoch on all public rules/bodies/examples
  (https://www.kaggle.com/code/socom20/jigsaw-pretraining-v82), zero-shot prompt ("…classify if it violates the given
  rule. Only respond Yes or No. Rule: {rule}" / "1. ```{body}```\nViolation:{answer}"), so TTT starts from a LoRA
  that already knows the task.
- **Slow ensemble:** 2× Qwen2.5-14B-instruct, 2× Qwen2.5-7B-instruct, 2× Qwen3-14B, 2× Qwen3-8B; Unsloth, one model
  per T4; vLLM inference outputs the **embedding of the final "Violation:" token**; per-rule LightGBM/XGBoost/SVC
  fitted on those embeddings (pairs of 7B LoRAs merged to save time).
- **Fast ensemble:** 2× bge-base-en-v1.5 with triplet loss (anchor = rule; one version with *negated* rules),
  features = body − positive centroid ‖ body − negative centroid; Qwen3-Embedding-0.6B with ArcFace (2 classes);
  DeBERTa-v3-small CE; each followed by classic ML heads.
- Partial ensembles (public/private): 2× Qwen3-14B 0.92571/0.92049 (best single group); 2× Qwen2.5-7B
  0.91024/0.90637; 2× bge 0.90952/0.89714; DeBERTa-v3-small 0.89546/0.88946. `subreddit` discarded.

### 4th-8th (one-liners worth a figure)
- 4th, Ruslan Berdichevsky — **SALSA** (Single-pass Autoregressive LLM Structured Classification, arXiv
  2510.22691, his paper): read logits over {"0","1"} as a Bayesian estimator; soft targets from duplicate
  vote ratios; **Generalized CE (Tsallis) q=0.955**; AWQ Qwen3-4B/14B; comment last in the prompt + batch by rule
  for prefix KV-cache reuse; Stage-1 small model → ~8,000 most-uncertain test rows as soft pseudo-labels →
  Stage-2 Qwen3-14B; blend 0.8/0.2 → 0.9286 private. Seed noise ≈ 0.002 AUC.
  (https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613473)
- 5th, Flash — diverse ensemble with per-rule rank weighting; Llama-3.1-8B TTT 0.92618 private was their best
  single (LoRA r=256, α=32, 2×T4 DDP, seq 512). (https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613421)
- 6th, ducnh279 — **Deep Mutual Learning** (arXiv 1706.00384) as online distillation among Qwen3 peers, because
  teacher softmax was too sharp for classic KD; single Qwen3-14B on test examples 0.925/0.921.
  (https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613150)
- 8th — 3× Qwen3-14B + Llama2-13B + bge, seed ensembles, QLoRA via Unsloth on 2×T4 within 12 h; trusted LB
  (29% of ~55k rows). (https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613195)
- 120th, Chris Deotte — RAG over the test examples (retrieve most similar positive/negative examples per row) +
  4× TTA + Qwen2.5-32B "reranker" on the 20% most uncertain rows; per-rule rank averaging.
  (https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613168)

### Key threads and notebooks
| Thread | Takeaway |
|---|---|
| neibyr, "Possible Grey Area… Training on Test Data via Positive/Negative Examples" https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/598099 | The test file contains its own labelled few-shot set; using it is the whole game. |
| c-number, "The rules are revealed" (127 votes) https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/607941 | Community LB probing recovered all hidden rule texts. |
| monnu, "Qwen2.5 LoRA Finetune Baseline LB 0.872" https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/591445 | Offline Qwen2.5-32B-GPTQ LoRA + vLLM baseline. |
| Chris Deotte, "Text Generation vs Sequence Classification" https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/591549 | Framing comparison for LLM classifiers (short post). |
| Chris Deotte, "Is LB 0.931 a Ceiling?" https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/610635 | Public LB plateaued at ~0.931 for weeks. |
| Psi, "Was the labeling synthetic?" https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613100 | Labels from the 2M-comment source felt random; external data was hard to use. |

### Shake-up
Small at the top: public ranks 1-5 finished private 1, 5, 4, 8, 3; 2nd place rose from public 6th. Gaps at the
top are ~0.0001-0.0008 AUC, i.e. below the ~0.002 seed noise reported by 4th place.

### Transferable lessons
1. **When the test set ships labelled examples, test-time LoRA is the model.** (598099, 613305, 613324)
2. **Read a classifier out of one token:** mask loss to the answer token, then take Yes-vs-No log-odds over token
   variants; `forward()` not `generate()`. (613305, 613473)
3. **Delete shortcut features** (subreddit) and **clean conflicting labels** by majority vote. (614279, 613421)
4. **Match the ensemble to the metric:** rank-normalise per rule before averaging for per-rule AUC. (613305, 613168)
5. **Engineer for the 12-hour/2×T4 box:** Unsloth 4-bit, two models in parallel, 80-90% memory headroom. (614279, 613195)
6. **LB probing can de-hide a hidden test design** — hosts should not rely on hidden prompts/rules. (604092, 607118)

### UNVERIFIED
- 2nd-place author names inside the writeup (the API lists team "^^" members only). 7th/9th not summarised.

## Make Data Count - Finding Data References

**Why it belongs (with a caveat for the writer):** an information-extraction task over scientific PDFs where LLMs
were *one* component, not the whole solution. The winners showed that **retrieval from the external corpora the
labels were built from** beat any extractor, and that for the type classifier an LLM won on accession IDs while
CatBoost on metadata won on DOIs. Good "know when not to use the LLM" episode.

### Facts
- URL: https://www.kaggle.com/competitions/make-data-count-finding-data-references (id 82370, forum 5476697)
- Dates: 2025-06-11 → 2025-09-09 (merger 2025-09-02). Host: Make Data Count (DataCite International Data
  Citation Initiative e.V.); prizes from The Navigation Fund and Chan Zuckerberg Initiative.
- Task: for each paper (PDF, plus XML for ~75%), list every research dataset it references — **DOIs** (e.g.
  `https://doi.org/10.5061/dryad.6m3n9`) or **accession IDs** (e.g. GSE12345, PDB 1Y2T, E-MEXP-568,
  SAMN…, CHEMBL…) — and tag each as **Primary** (data generated for this paper) or **Secondary** (reused).
  Papers from the Europe PMC open-access subset. Rerun test ≈ 2,600 articles. Only 524 training PDFs
  (per https://zenn.dev/mkj/articles/0d42ec3db44749).
- Metric: global F1 over unique (article_id, dataset_id, type) tuples ("82370_MDC_Global_F1"). A right dataset with
  the wrong type costs both a FP and a FN (3rd place's point). Articles with no references must be omitted.
- **Label/metric update** ("IMPORTANT: Updated Train Labels and Metric",
  https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/589314): some articles
  were only partially labelled with "Missing" true negatives; train_labels.csv was updated to list them, and the
  metric was changed to ignore unlabelled citations in those articles; all submissions rescored.
- Labels were not human-labelled from scratch: per the host (quoted by 1st place) they came from **Europe PMC
  NER text-mined accession IDs and DataCite dataset-DOI ↔ article mappings**, so they inherit that NER's misses and
  the host's regex filters.
- Code competition: CPU/GPU ≤ 9 h, internet off. Public/private 30/70.
- Prizes: $100,000 (1st $40k, 2nd $20k, 3rd $17k, 4th $13k, 5th $10k). Winner licence MIT. Data CC0.
- Teams: 1,282 teams, 1,531 competitors, 36,850 submissions.

### Final leaderboard (top 10)
| Rank | Team | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|
| 1 | Ali \| Kea (asalhi, keakohv) | 0.79732 | 0.89095 (9) | Candidates from Data Citation Corpus v4.1 + EUPMC mined terms; DOI type = 6× CatBoost on title/author similarity; accession type = 0-shot Qwen2.5-Coder-32B-AWQ | https://www.kaggle.com/competitions/make-data-count-finding-data-references/writeups/1st-place-solution |
| 2 | Mohsin and Nikhil (tezdhar, nikhilmishradev) | 0.78648 | 0.88680 (10) | DataCite dump + EUPMC terms for 100% recall; heuristics; MedGemma-4B LoRA + BiomedBERT classifiers | https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606786 |
| 3 | BRBR (arc144, tomirol) | 0.77292 | 0.87635 (12) | Regex + DataCite/Crossref dumps; heuristics + 6-fold DeBERTa-v3-large with EMA + grad clipping | https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606879 |
| 4 | [NVIDIA] Raja & Ahmet & David | 0.76788 | 0.91837 (1) | Tool-calling agent makes synthetic labels → Qwen2.5 warm-up; cascade Qwen2.5-14B → 32B → 72B | https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606921 |
| 5 | cm391 (mccocoful) | 0.76167 | 0.89711 (7) | "standing on the shoulders of giants" | https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606769 |
| 6 | Team (cakiki, conormacamhlaoibh, samson8) | 0.75866 | 0.91080 (2) | – | topic 606867 |
| 7 | 反卷局重新起航 | 0.75232 | 0.90413 (5) | – | topic 606757 |
| 8 | zhang2dog | 0.74574 | 0.86573 (13) | – | topic 606768 |
| 9 | flash attention (Geremie Yeo) | 0.74556 | 0.85390 (14) | solo gold | https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606743 |
| 10 | Khoi Nguyen | 0.74105 | 0.83063 (19) | probing-driven rules | https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606740 |

### 1st place — Ali | Kea (Kea Kohv, Ali Salhi)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606853 ; notebook
https://www.kaggle.com/code/keakohv/mdc-1st-place-solution-catboost-and-qwen/notebook?scriptVersionId=262235918
- **Insight:** because labels were built from EUPMC NER + DataCite mappings, using those sources as the candidate
  generator beats any regex/NER extractor.
- **Validation:** discard "sparsely labelled" articles (those with a Missing label) from local F1, per mccocoful;
  correlation with LB still shaky, better for DOIs. Used LB submissions to learn which accession families exist
  in the public test. Estimated mix ~70% accession / 30% DOI.
- PDF → text with PyMuPDF (whitespace cleaning + unicode normalisation before matching).
- **DOI stage 1:** candidates = Data Citation Corpus **v4.1** pairs with source == DataCite, kept only if the DOI
  appears in the PDF/XML. On train: 321 predictions, 302 TP, 19 FP, 23 misses; mention-level F1 0.9350. Adding the
  DataCite dump directly gave 100% recall but more FPs.
- **DOI stage 2:** no text context at all — **metadata only**. Article metadata via Crossref (authors, title, year,
  publisher, journal), dataset metadata via DataCite (authors, year). Features compare titles, authors, years;
  publisher/journal/repository as categoricals; #DOIs per article. **6-fold CatBoost, stratified by type, grouped
  by article**. OOF type F1 0.87; DOI triple F1 0.82.
- **Accession stage 1:** Corpus v4.1 with source "eupmc" + the original EUPMC TextMinedTerms mappings
  (https://europepmc.org/pub/databases/pmc/TextMinedTerms/), with a hand-chosen whitelist of families from each
  (e.g. ArrayExpress, InterPro, ChEMBL, UniProt, PDB, BioSample, GEO…; ENA only with prefix filters). Keep only IDs
  present in text. **Predict accession IDs only for articles with no predicted DOI** (an LB boost, contrary to
  public discussion).
- **Accession stage 2:** **Qwen2.5-Coder-32B (AWQ) + vLLM**, simple 0-shot prompt on the context snippet; Coder
  beat the base model. `MultipleChoiceLogitsProcessor` constrains output to A/B. For BioSample (SAMN) IDs the prompt
  adds article title/authors/year and the BioSample submitter and date (from NCBI API). Post-rule: majority type
  per accession family within an article.
- **Scores (F1):** DOI-only 0.362 public / 0.350 private; accession-only 0.754 / 0.664; full 0.890 / 0.797.
  Runtime ~29 min on train.
- **Did not work:** Qwen for DOI type (even with metadata + snippet) or as a CatBoost reviewer; DataCite relation
  types (IsSupplementTo); article-context features for DOIs; fancier or Chinese prompts; Marker OCR (better text,
  too slow).

### 2nd place — Mohsin and Nikhil
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606786
- Stage 1 DOIs: Data Citation Corpus (v2 validated best) + DataCite Public Data File 2024 → 100% recall; drop
  figshare / CCDC / HEPData (tagged Missing in train); keep only IDs found by a whitespace-tolerant regex in PDF/XML;
  strip Dryad version tags → 0.964 type-agnostic F1 on train.
- Stage 1 accession: EUPMC TextMinedTerms → 100% recall; drop hgnc, gca, go, rrid and other `:`-style families
  (chebi, orpha, rhea, efo) and ebisc/hipsci/omim/eudract; drop accessions in articles that have a DOI; a rule for
  SAMN IDs in an online-only table (+0.003 public / +0.004 private) → 0.98 type-agnostic F1.
- **Heuristics alone = gold** (0.869 public / 0.739 private, "No model required to get a gold medal!"): SAMN and
  EMDB → Primary; for a DOI cited by several papers, earliest = Primary; `isSupplementTo` → Primary; >4 same-repo
  DOIs nearby → Secondary.
- Models: Mohsin — **MedGemma-4B LoRA** for DOI type with a ≤2,048-token context (first regex match context + top-3
  BM25 chunks vs the DataCite abstract) → 0.880 / 0.784. Nikhil — two BiomedBERT
  (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) classifiers; replace non-target IDs with
  "other dataset id", mark the target "Prediction {ID}"; batch size 256 for stability; multi-seed OOF thresholds;
  context 300 chars (accession) / 150 chars (DOI) + "More than 3 mentions" and repository tokens.

### 3rd place — BRBR (arc144, tomirol)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606879
- Regex candidates matched against the 2024 DataCite dump and EUPMC mined terms; Crossref 2025 dump for metadata;
  XML only as fallback; drop figshare DOIs, `GCA_`, `HGNC:`, `rs*`; if both accessions and DOIs are present, drop
  the DOIs (LB boost).
- Heuristics first (Dryad DOIs and SAMN → Primary; DOI title/authors similar to paper → Primary; accession cited
  >5 times in EUPMC → Secondary; DOI with ≥5 citations → Secondary), then **6-fold DeBERTa-v3-large**
  (StratifiedGroupKFold by type/article) on 1k-char context + first 500 chars of the article + titles.
- Qwen2.5-7B with a classification head was unstable; DeBERTa faster and better. Stabilisers: gradient clipping
  and **weight EMA** (decay 0.9995, code in post).

### 4th place — NVIDIA team (Raja Biswas, Ahmet Erdem, David)
Source: https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606921 ; code
https://github.com/rbiswasfc/mdc-4th-place-solution
- Candidates from Data Citation Corpus v3.0 (minus czi source, HEPData, CCDC, figshare) and PMC TextMinedTerms
  (minus gca, go, hgnc, rrid, nct); regex + LLM filter as fallback.
- **Tool-calling agent for synthetic labels** over Europe PMC OA articles (keyword + semantic search tools,
  50k+-token articles) → warm up Qwen2.5 models, then fine-tune on competition data (+~0.02);
  demo https://www.kaggle.com/code/conjuring92/mdc-tool-calling-agent-demo, thread
  https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606752. Pseudo-labels
  from Qwen2.5-72B for diversity; cap 24 mentions/article; mask IDs as `<mask_i>`; EMA.
- Context: first 1,400 chars, mention snippets, data-availability section, nearby DOI snippets, table headers.
- **Cascade inference:** Qwen2.5-14B → route uncertain (50% of DOIs, 20% of accessions) to Qwen2.5-32B → hardest
  10% to Qwen2.5-72B. Several accession families forced to Secondary.
- Public 1st (0.91837) → private 4th (0.76788).

### Key threads
| Thread | Takeaway |
|---|---|
| Roderic Page, "Alternative, manually constructed training data" (75 votes) https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/586075 | Hand-annotated the train PDFs (https://github.com/rdmpage/make-data-count-training-data); official labels scored F1 0.446 against them. |
| Host, "IMPORTANT: Updated Train Labels and Metric" https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/589314 | Missing-type negatives and metric fix; rescore. |
| inversion (host), "Data Quality - Mega Thread" https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/584899 | Host acknowledged label issues. |
| Khoi Nguyen, "Quality of train vs test labels" https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/584792 | LB favoured fewer references → "retrieval vs agreeing with labels". |
| Khoi Nguyen, "Tricks (10char)" https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/606740 | DOIs from corpus v3, accessions from v4 eupmc, one prefix worth +0.08 when removed, all SAMN primary; "nothing beat probing". |
| Hzwlf, "My Rule-Based Method Beats My LLM Attempts" https://www.kaggle.com/competitions/make-data-count-finding-data-references/discussion/598550 | All-SAMN-primary rule 0.660 beat LLM classification 0.600. |

### Shake-up
Large absolute drop (public ~0.89-0.92 → private ~0.74-0.80) and a reorder: public 1st (NVIDIA) → 4th; private 1st-3rd
came from public 9th, 10th, 12th. Probing-tuned family whitelists fit the public 30% only.

### Transferable lessons
1. **Find how the labels were made, then use that generator.** (606853, 606786)
2. **LLM where language matters, tabular model where metadata matters:** Qwen for accession context, CatBoost for
   DOI metadata. (606853)
3. **Heuristics can be gold;** an LLM must beat the rule baseline first. (606786, 598550)
4. **Agents can make training data:** a tool-calling LLM labelling external papers lifted LB ~0.02. (606752)
5. **Cascade by uncertainty** (14B → 32B → 72B) to fit big models in the runtime. (606921)
6. **Do not over-probe a 30% public split.** (shake-up above; 606740)

### UNVERIFIED
- Content of 5th-8th writeups not summarised.

## NeurIPS 2025 - Google Code Golf Championship

**Why it belongs:** the cleanest "humans vs LLM agents" scoreboard on Kaggle. The task (shortest Python for all
400 ARC-AGI-1 training tasks) is checkable, so LLMs can be run in generate-test-shorten loops. The verdict at the
top: **1st, 2nd, 3rd were human golfers** (LLMs used only for first drafts or tooling); **4th "FuunAgent" was ~98%
LLM-generated** (gpt5-codex), and Sakana's ALE-Agent finished 14th.

### Facts
- URL: https://www.kaggle.com/competitions/google-code-golf-2025 (id 95282, forum 7155893). Host: Google DeepMind
  (organiser Michael D. Moffitt). Part of the NeurIPS 2025 Competition Track.
- Dates: 2025-07-31 → 2025-10-30 (entry/merger 2025-10-23).
- Task: for each of the 400 ARC-AGI-1 public training tasks submit `taskNNN.py` defining `p(grid)` that returns the
  exact output for every pair in `train` + `test` + `arc-gen` (extra pairs from the ARC-GEN-100K dataset).
  Python standard library only, self-contained files (security constraints deployed 2025-08-19).
- Metric: per task `max(1, 2500 − length_in_bytes)` if correct on all pairs, else 0.001; total over 400 tasks
  (max 1,000,000). LB is 100% public; 100 submissions/day; 1 scored submission.
- Prizes: $100,000 — 1st $30k, 2nd $20k, 3rd $10k, 4th-10th $5k each, plus a $5k "Longest Leader" prize.
  Top teams invited to present at NeurIPS and co-author a PMLR retrospective.
- Teams: 1,142 teams, 1,414 competitors, 41,315 submissions.
- **Exploits / metric fixes:** byte length was measured *after* running the submitted code, so monkey-patching
  `os.path.getsize` produced impossible billion-point scores ("Critical Scoring Exploit and a Call for a Fair
  Leaderboard", https://www.kaggle.com/competitions/google-code-golf-2025/discussion/597956); another 960K+ method
  was reported privately and fixed by the 2025-08-25 patch
  (https://www.kaggle.com/competitions/google-code-golf-2025/discussion/599368, write-up notebook
  https://www.kaggle.com/code/isakatsuyoshi/resolved-past-vulnerabilities). Shell calls (`os.popen` → perl) were
  ruled legal late; the top 3 agreed not to use them (1st-place writeup).

### Final leaderboard (top 10 + notable LLM teams)
| Rank | Team | Score | Approach | Writeup / code |
|---|---|---|---|---|
| 1 | Code Golf International (lukeg10081, mukundan314, seekerstyle, bruhhhhd, sisyphuscg) | 962,070 | 5 experienced golfers, all by hand; zlib on 21 tasks | https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614092 ; https://github.com/Seek64/NeurIPS-Code-Golf-2025 |
| 2 | jailctf merger | 961,805 | CTF team; char-level fuzzer "charbrute"; GA for compression; LLM drafts for long tasks | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/solution |
| 3 | ox jam! (xsot and others) | 961,784 | code.golf veterans by hand; pysearch for expressions | https://www.kaggle.com/competitions/google-code-golf-2025/writeups/3rd-place-submission-ox-jam ; https://github.com/xsot/google-code-golf-2025 |
| 4 | FuunAgent (confirm, jonaskreutz, thunderthunder) | 957,810 | gpt5-codex parallel sampling (N=4) with AST-rule prompts; ~98% LLM for one member | https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614124 ; dashboard https://gcgc.pages.dev/ |
| 5 | HIMAGINE THE FUTURE. (keymoon …) | 957,568 | student CP/CTF team; custom deflate optimiser stronger than Zopfli (~+700 pts); seed cracking | https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614225 |
| 6 | Theoretical Syntax Golfers | 957,527 | – | topic 615145 |
| 7 | LogicLynx | 957,513 | – | topic 619321 |
| 8 | import itertools | 957,177 | – | https://www.kaggle.com/competitions/google-code-golf-2025/discussion/615039 |
| 9 | Santa Golf Club 2500 | 957,151 | – | topic 614150 |
| 10 | lv1.dev | 956,458 | – | topic 628087 |
| 14 | ALE-Agent (Sakana AI) | 954,571 | agent tooling incl. prompt generators for ChatGPT/Claude | https://github.com/SakanaAI/google-code-golf-2025 |
| 19 | Team (Yuchen20 part) | 953,039 | evolutionary LLM program-synthesis agent (Gemini free tier) then manual | https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614729 |
| 25 | Ravi Annaswamy | 945,774 | "Teaching LLMs to Golf" with Claude, GPT-5, Gemini | https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614024 |

All solution repos collected by c-number: https://www.kaggle.com/competitions/google-code-golf-2025/discussion/613968

### 1st place — Code Golf International
Source: https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614092
- **No LLMs; very manual.** A few hours per task per author + reviewers, many tasks revisited when new tricks
  appeared. Only automation: zlib compression (21 of 400 tasks). "No particular technique… gave a decisive
  advantage over 2nd or 3rd."
- Building blocks: `zip(*g)` transpose, `g[::-1]` mirror, `zip(*g[::-1])` rotate, `sum(g,[])` flatten,
  `filter(any,g)`, `{}.fromkeys(r)` ordered unique colours.
- **Recursion templates:** `p=lambda g,n=3:-n*g or F(p(g,n-1))` ("do X, rotate, repeat 4×"); depth-2 variants;
  "slice recursion" `p=lambda g:F(g[70:]or p(g*2))` (task 093 uses `g[57330:]` = 14·(2¹²−1) for 12 iterations
  on 14×14); `[g:=F(g)for _ in g][3]` loops; `exec("…"*9)`.
- List comprehensions with walrus state; `list.pop()` for look-ahead and implicit reversal.
- **Regex on `str(g)`** then `eval` (46 solutions), e.g. task 294
  `p=lambda g:eval(re.sub("(?<=5.{34})5(?=.{34}5)","2",str(g)))`; rotate-and-replace loops; `*[\g<0>][::-1]`
  replacement to reverse a match.
- **"Dimension recursion":** one `p` handles grid/row/cell via `g*0!=0`.
- **Guess-and-check** (task 124), hash-based lookup tables brute-forced for efficiency.
- **zlib golfing:** zopfli/libdeflate first pass, then a re-encoder that reuses the Huffman tree and uses DP to
  account for escape costs inside a Python literal; search over variable names for compressibility. Under
  compression, repetition is nearly free, so write repetitive code.

### 2nd place — jailctf merger
Source: https://www.kaggle.com/competitions/google-code-golf-2025/discussion/618673 ; repos
https://github.com/orgs/jailctf-merger-goog-golf/repositories
- 5 CTF players (one seasoned golfer, lydxn, did most golfing). Early investment in compression tooling and a
  shared spreadsheet.
- **charbrute fuzzer:** random character-level deletions/insertions/substitutions centred on a hotspot, chunk
  mover, run on PyPy 3.11; mostly small saves, occasionally big; saved 100+ bytes in the final stretch.
- Compression: strip zlib header/checksum, `wbits=-9`; variable-rename optimiser driven by a **genetic algorithm
  ("implemented by gemini")**; 24 compressed tasks at the end.
- **LLM role:** for long tasks, an LLM produced a first working solution from the ARC-GEN generator code + golf tips
  + human hints; then manual golfing. "Most of the llm solutions we golfed got obsoleted by either public notebooks
  or with a solution rewrite."

### 3rd place — ox jam!
Source: https://www.kaggle.com/competitions/google-code-golf-2025/discussion/619624 ; repo
https://github.com/xsot/google-code-golf-2025 (members per README: ovs-code, xsot, JoKing, att/zacjl, MeWhenI).
- Short post: solutions in repo (`merged/all_tasks.py` pre-compression); `merge_and_submit.py` takes each member's
  best per task and applies zlib. Credits pysearch (https://github.com/lynn/pysearch) and garrymoss's public
  notebook. No LLM use is mentioned.

### 4th place — FuunAgent (the strongest LLM-driven entry)
Source: https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614124
- One member's solutions were "98% generated by an LLM and only about 2% by human effort".
- **Parallel sampling:** gpt5-codex with tools analyses the task JSON in a REPL; each session 10-30 turns; each turn
  generates **4 parallel versions**, evaluates them, keeps the shortest valid one, then a follow-up prompt.
- **Rule-based prompt generation from the AST:** if no regex → "Try to use the regex re package…"; loops →
  recursion; `def` → lambda. Base prompt: "Please make it as short as possible in bytes for Code Golf and tell me the
  file size." Stop after *s* turns without improvement; after *t<s*, exploration prompts ("Analyze the test cases
  generation code.", "Rethink a more compact algorithm."). "Shorter code is known to exist." keeps the model going.
- Ran ~15 Codex Cloud sessions in parallel on a ChatGPT Pro subscription, by hand from a phone for three months;
  later `codex exec` in Docker, logged to SQLite. **Fresh empty sandboxes** break local optima (the repo with old
  code anchored the model).
- Last two weeks: optimise for **post-compression size** ("minimal Kolmogorov complexity… evaluated based on the
  file size after zip compression").
- Grok-4 variants tried and dropped (tool misuse / cost). Human teammates broke stalls where the AI approach failed.

### Key threads
| Thread | Takeaway |
|---|---|
| Chris Deotte, "Solutions By Hand or Solutions By LLM?" https://www.kaggle.com/competitions/google-code-golf-2025/discussion/608310 | The framing question of the competition; answered by the final LB. |
| Ravi Annaswamy, "Getting to Rank 25 by Teaching LLMs to Golf" https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614024 | Teach the LLM named golf patterns and iterate; agent found sub-LB-best solutions late. |
| Yuchen20, "19th place (a LLM agentic solution)" https://www.kaggle.com/competitions/google-code-golf-2025/discussion/614729 ; blog https://yuchen-mao.vercel.app/blog/google-code-golf | Evolutionary LLM agent hit ~0.940 normalised mid-competition, then plateaued; ARC's spatial abstractions make LLMs hallucinate. |
| Luke G, "Share individual task scores here!" https://www.kaggle.com/competitions/google-code-golf-2025/discussion/596679 | code.golf community's public per-task spreadsheet = the benchmark everyone chased. |
| c-number, "Final 3 Days: Could People Pause Solution Releases?" https://www.kaggle.com/competitions/google-code-golf-2025/discussion/613583 | Public notebooks with strong golfed solutions moved the board late (title only read). |

### Shake-up
None (100% public LB). Gap 1st-2nd = 265 points (~0.1 byte per task on average).

### Transferable lessons
1. **Verifiable tasks make LLM loops cheap, but the last 0.5% still belonged to human experts.** (614092 vs 614124)
2. **Best-of-N + automatic verification + rule-based critique prompts** is a practical agent recipe. (614124)
3. **Start agents from a blank sandbox** to escape anchoring on existing solutions. (614124)
4. **Optimise the real objective** (post-compression bytes), not a proxy. (614092, 614225, 614124)
5. **Fuzzers and GAs complement LLMs** for micro-optimisation. (618673)
6. **Measure after, not during, untrusted execution** (host lesson from the getsize exploit). (597956)

### UNVERIFIED
- Longest Leader prize recipient. Exact share of LLM-written code in teams other than 4th, 19th, 25th.

## The 2026 NeuroGolf Championship

**Why it belongs:** the sequel to Code Golf, where the artefact is a tiny ONNX network per ARC task and the work
was done **almost entirely by fleets of LLM coding agents** (Codex, Claude Code, ChatGPT Pro) managed by humans.
The 1st-place team is literally called "Kaggle Agent". This is the best case study of *agent operations*: task
notes, cookbooks, targets, rewrite-vs-optimise economics, consensus validation.

### Facts
- URL: https://www.kaggle.com/competitions/neurogolf-2026 (id 116438, forum 8930897). Host: The Neurosynthetic
  Research Institute (organiser Michael D. Moffitt). IJCAI-ECAI 2026 Competitions Track.
- Dates: 2026-04-15 → 2026-07-15 (entry/merger 2026-07-08).
- Task: for each of the 400 ARC-AGI-1 training tasks submit `taskNNN.onnx`. Input is a one-hot tensor
  [1, 10, 30, 30] (zero-hot outside the grid); output must reproduce every pair in train + test + arc-gen, plus a
  small **private** example set per task (anti-overfitting).
- Constraints: static shapes only; banned ops `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`; each file
  ≤ 1.44 MB.
- Metric (Evaluation page): per task `max(1, 25 − ln(cost))`, cost = total parameters + total memory footprint (bytes);
  sum over 400 tasks. Every factor of e cut from cost = +1 point (2nd-place README). LB 100% public;
  submissions/day raised to 100 on 2026-04-21.
- Prizes: $50,000 — 1st $12k, 2nd $10k, 3rd $10k, Top Student Team $8k, **Longest Leader $10k** (period
  2026-05-06 → 2026-07-15).
- Teams: 2,963 teams, 3,469 competitors, **492,086 submissions**.
- **Metric patches:** "NeuroGolf Update for April 21st" (https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711):
  100 subs/day, ignore >30×30 test cases, report failing networks, patched a **negative-memory exploit**
  (calibrator, https://www.kaggle.com/competitions/neurogolf-2026/discussion/693247: chaining onnx-tool bugs —
  `PadNode.shape_infer` negative shapes × `get_memsize` product — made every task score 25; at that time cost
  included MACs), onnx-tool profiler fixes; planned enforcement of static shapes and counting `Constant` params.
  Further loopholes found **by agents** were reported privately: Kaggle Agent's "LB 9550 potential scoring loophole"
  (https://www.kaggle.com/competitions/neurogolf-2026/discussion/696365) and Chris Deotte's "LB 10000 - Convert any
  task into score 25 - [fixed]" (https://www.kaggle.com/competitions/neurogolf-2026/discussion/696377, "My agent
  just discovered…", under the 2026-04-30 metric).
- **Integrity:** teams were removed at the end, including some near the top; an investigation post reports two
  teams' final submissions sharing 74 bit-identical graphs and two banned participants
  (https://www.kaggle.com/competitions/neurogolf-2026/discussion/729721). A dataset claiming a 7957 solution was the
  public baseline encrypted with ZipCrypto (https://www.kaggle.com/competitions/neurogolf-2026/discussion/726541).

### Final leaderboard (top 10)
| Rank | Team | Score | Approach | Writeup |
|---|---|---|---|---|
| 1 | Kaggle Agent (titericz, jiweiliu, maxjeblick, arc144, cdeotte) | 8314.03 | Agent fleets (Claude Code Opus 4.8 / Fable 5, Codex, ChatGPT Pro 5.6-Sol/5.5); targets + task notes + cookbooks; vertical single-task + horizontal multi-task pipelines; evolved agent contexts | https://www.kaggle.com/competitions/neurogolf-2026/writeups/1st-place-kaggle-agent (+ 726799, 726883) |
| 2 | neurogolf team (jacekwl, yhirano, chankhavu, yeoyunsianggeremie, gufanmingmie) | 8243.89 | 5 independent agent pipelines; ChatGPT 5.5 Pro Extended + Codex; Claude Code drives ChatGPT web via Chrome; TECHNIQUES/TRICKS.md | https://www.kaggle.com/competitions/neurogolf-2026/writeups/work-in-progress-2nd-place-neurogolf-team ; https://github.com/bogoconic1/neurogolf-2026-2nd-place-solution |
| 3 | The Final Ensemble (christofhenkel, tomirol, fritzcremer, dziyanavalenta, tonylica) | 8221.27 | Codex workers + ChatGPT UI with shared attempt ledgers; forced full rewrites; consensus validation from top-10 Code Golf programs | https://www.kaggle.com/competitions/neurogolf-2026/discussion/727356 |
| 4 | M & H & M & N & M | 8142.02 | – | https://www.kaggle.com/competitions/neurogolf-2026/discussion/727134 |
| 5 | claudex (binhdth, buphmquc, pntan17, hainguyenthien, myotterspace) | 8118.97 | – | https://www.kaggle.com/competitions/neurogolf-2026/discussion/726691 |
| 6 | Slow and Steady | 8067.63 | – | https://www.kaggle.com/competitions/neurogolf-2026/discussion/726660 |
| 7 | kluges clueless | 8048.16 | – | topic 726693 |
| 8 | Pavel & CroDoc | 8038.62 | – | topic 726782 |
| 9 | Timeout exceeded : ( (yiheng, solo) | 8028.25 | – | https://www.kaggle.com/competitions/neurogolf-2026/discussion/726653 |
| 10 | Kimura@SeSDA & OverfitOracle | 8025.82 | – | topic 727156 |

### 1st place — Kaggle Agent (Chris Deotte, Giba, Jiwei Liu, Max Jeblick, Eduardo Rocha de Andrade)
Sources: intro https://www.kaggle.com/competitions/neurogolf-2026/discussion/726654 ; pipeline
https://www.kaggle.com/competitions/neurogolf-2026/discussion/726799 (Eduardo) ; prompts
https://www.kaggle.com/competitions/neurogolf-2026/discussion/726883 (Chris) ; dashboard
https://daxiongshu.github.io/kaggle-agent-neurogolf-2026/
- Team = 5 humans + "ChatGPT, Codex and Claude Code". Shared dashboard of per-task ONNX and scores.
- **Rewrite beats optimise:** from task score histories, architectural rewrites averaged **+0.5 per task**, optimising
  existing graphs **+0.05** — so agents were pushed to explore new architectures.
- **Token efficiency (726799):** open-ended "improve this task for 2 hours" gave +0.05…+0.2 per round until a
  plateau, then a rewrite threw away all that pruning work. Fix: **concrete targets + "headstrong" prompts**
  ("improve this task by at least 1.5 points! It is possible and has been achieved by another competitor…"), which
  made agents first derive a theoretical max before editing.
- **Task notes** pre-baked per task (pair counts, shapes, palettes, symmetry probes, baseline cost/ops/initializers,
  archetype + family median cost, a drafted rule with confidence) and appended by each worker with what it tried and
  dead ends. Reusable Python utils; agents dump **builder scripts** for ONNX.
- **Cookbooks** = "central pillar": every new trick goes into a shared cookbook given to all workers.
- **Two pipelines:** vertical single-task, single-turn workers (`codex exec` spawned by a script; ChatGPT Pro chats
  with a fixed template + task bundle incl. targets, cookbooks, prior notes, human ARC solutions) and horizontal
  multi-task sessions that push a new trick across similar tasks using a **task clustering report**. Loop:
  single-task exploration → trace analysis to mine tricks → cookbook update → multi-task exploitation.
- Local validation on given pairs + ARC-GEN pairs, sometimes random inputs/colour permutations for colour-agnostic
  tasks. Models used: Claude Code (Opus 4.8, Fable 5), Codex, ChatGPT Pro (5.6-Sol, 5.5).
- **"Self Evolving Prompts" (726883):** run several agents in 4-hour "mini contests" ("achieve the most total onnx LB
  score gain"), humans mentor mid-run and agents self-review; after three rounds promote the best-performing agent
  (its *context* is the asset) to manager of the others and have it write the new `cookbook.md`.

### 2nd place — neurogolf team (5 independent pipelines)
Source: https://www.kaggle.com/competitions/neurogolf-2026/discussion/728549 ; repo README
https://github.com/bogoconic1/neurogolf-2026-2nd-place-solution
- Members ran **independent pipelines** on purpose (diversity against local optima); shared a private HF leaderboard
  Space for candidates.
- Jacek: Claude Code → ChatGPT 5.5 Pro Extended (solved harder tasks from scratch) with Codex for verification,
  risk checks and transfers; reused Code Golf 2025 knowledge, then **removed golf solutions** as references because
  short code ≠ small graph; TECHNIQUES.md grew to ~50k lines. Example tricks: fp32 **subnormal underflow** as a count
  threshold inside one `Einsum`; ONNX **`RNN` with ReLU** as a static recurrence since `Loop/Scan` are banned
  (task138 cost 2505 → 2300). Brute-force search didn't pay; 5.6 Sol Pro sessions often ended "Thinking Failed".
- Geremie: Claude Code "gives up way too easily"; Codex 5.5 xhigh, then a **Claude Code skill that drives ChatGPT
  web through Chrome** and downloads ONNX; prompts force a **zero-memory Einsum**; a 24/7 Codex session reorders
  Einsum operands for runtime; TRICKS.md entries auto-deleted after 14 days unused; Einsum has a **52-variable**
  limit, so split into two Einsums with a tiny intermediate; "repeated-power homotopy factorisation" of dense
  Einsum cores.
- README pipeline: Claude Code as driver round-robins ChatGPT sessions (30-90 min each, nine nudges over ten
  rounds), harvests files named `task{NNN}_m{memory}_p{params}.onnx`, auto-submits to the Space; five lanes
  (cost-golf, compress-einsum, convert-to-0-memory-einsum, runtime, daily-bundle).

### 3rd place — The Final Ensemble (Christof Henkel and team)
Source: https://www.kaggle.com/competitions/neurogolf-2026/discussion/727356
- **Codex** main optimiser (one task per worker) + **ChatGPT UI** (fresh chat per task in a Project holding scorer
  rules, technique libraries, corpus, negative knowledge). In the last week ~70% of gains from Codex, ~30% from UI.
- Warm-start bundle per run: task data, ARC-GEN generator, RE-ARC verifier, 2025 Code Golf solution, current ONNX +
  builder + per-tensor cost view, exact cost target, previous promotions, rejected branches, similar solved tasks.
- **Before optimising, force rewrites:** price a direct implementation, a finite certificate, and a re-read-input-
  at-output design, each ending with a prototype/price/failing case.
- Four kinds of history: `history/taskNNN.jsonl`, `attempt_ledger` (accepted + rejected with reasons),
  `retry_state` (do_not_repeat, leads, forced rewrites), `negative_knowledge.md`. Imported even rejected UI runs.
- Example: task 349 cost 4,839 → 205, score 16.516 → 19.677 (+3.161) by combining ideas from 5 other tasks; the
  same technique transferred to task 367 (+3.094) with one prompt.
- **Consensus validation:** sample new ARC-GEN examples, run the **top-10 Code Golf 2025 programs**, keep only
  examples all ten reproduce → 2,000-example hidden-test proxy.

### Key threads
| Thread | Takeaway |
|---|---|
| calibrator, "The highest single task score is 25 (fixed)" https://www.kaggle.com/competitions/neurogolf-2026/discussion/693247 | Static profilers can be gamed (negative memory). |
| Jiwei Liu, "Honest audit of Kaggle Agent's current best submission (9538)" https://www.kaggle.com/competitions/neurogolf-2026/discussion/694772 | An agent-written self-audit of dynamic-shape and Constant-param loopholes before the rules changed. |
| Chris Deotte, "Web GUI to Build ONNX by Hand" https://www.kaggle.com/competitions/neurogolf-2026/discussion/699313 | Human tooling for ONNX (title/votes only read). |
| Tony Li, "ChatGPT Plus — $20/month — is all you need" https://www.kaggle.com/competitions/neurogolf-2026/discussion/704942 | Web ChatGPT in many tabs reached ~7600 without API costs. |
| Jiwei Liu, NVIDIA Kaggle agent skills report https://www.kaggle.com/competitions/neurogolf-2026/discussion/717854 ; https://github.com/NVIDIA/nvidia-kaggle | Open-sourced agent skills for Kaggle. |
| Sergey Bryansky, "Internal Investigation — Finally" https://www.kaggle.com/competitions/neurogolf-2026/discussion/729721 | Bit-identical graphs across teams; bans. |

### Shake-up
None (100% public), but late disqualifications reordered the top of the board (see 729721).

### Transferable lessons
1. **Give agents targets, not time budgets.** A stated gain target changed behaviour and cut tokens. (726799)
2. **Persist memory outside the agent:** task notes, attempt ledgers, negative knowledge, cookbooks. (726799, 727356, 728549)
3. **Rewrite > tweak** when the metric is logarithmic in cost. (726654, 727356)
4. **Transfer every win across similar tasks** (task clustering, horizontal sessions). (726799, 727356)
5. **Build a hidden-test proxy by consensus of independent solvers.** (727356)
6. **Agents find metric exploits fast** — hosts need red-teaming, and agent outputs need honest audits. (696365, 696377, 694772)

### UNVERIFIED
- Longest Leader and Top Student Team recipients. 4th-10th writeups fetched in part, not summarised.

## Name That LLM (ongoing; state as of 2026-10-27)

**Why it belongs:** model attribution — which of 18 frontier LLMs wrote a 40-250-word passage. It is the
"fingerprinting" sequel to Detect AI Generated Text, and the first weeks already show the classic Kaggle arc:
CV design, data-generation arms race, rules questions about API-generated data, and a structural exploit idea.
**No final results exist yet; everything below is the live state on 2026-10-27, 13 days in.**

### Facts
- URL: https://www.kaggle.com/competitions/name-that-llm (id 170484, forum 13688374). Host: Kaggle.
- Dates: opened 2026-10-14; final submission deadline 2027-01-13 per the Timeline page (API `deadline`
  2027-01-14T23:59); team-merger deadline 2027-01-06 (API).
- Data: `train.csv` = **2,699 passages for 150 prompts** (`id`, `prompt_id`, `text`, `model`); hidden test ≈ **4,000
  passages** written for prompts **not in train**; passage length varies randomly 40-250 words; only light
  formatting cleanup. Topics: philosophy, physics, mathematics, biology, CS, creative prose.
- Labels (18): OpenAI `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-6-astra`; Google
  `gemini-3.1-pro-preview`, `gemini-3.6-flash`, `gemini-3.7-flash`, `gemini-3.8-flash`; Anthropic `claude-fable-5`,
  `claude-opus-4-6`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`; Qwen `qwen3.8-max`; ZAI `glm-5.3`,
  `glm-5.3-flash`; Moonshot `kimi-k3`.
- Metric: **balanced accuracy** (mean per-class recall over K=18). Per-model test counts are unequal and the mix differs
  between public and private. Public LB = 40% (`leaderboardPercentage`). Chance ≈ 0.056.
- Code competition: CPU/GPU ≤ 9 h, internet off; output `submission.parquet` (`id`, `model`).
- Prizes: $50,000 = **ten prizes of $5,000** (1st-10th). Winner licence CC-BY 4.0 (Rules §1); data CC BY-SA 4.0.
- External data: allowed if public and equally accessible at no cost, or meeting the "Reasonableness" standard of
  §2.6.b (minimal cost, reasonably accessible). Whether **paid-API generations from the 18 candidate models** qualify
  is the open rules question (threads 749613, 751011, 751954; no host answer read).
- Teams (2026-10-27): 3,523 teams, 3,752 competitors, 29,111 submissions; 5/day.

### Current public leaderboard (2026-10-27; public 40% only, will change)
| Rank | Team | Public BA |
|---|---|---|
| 1 | Hod Carriers (tononnh) | 0.728 |
| 2 | Team Aillis (huuquan1994, analokamus, yujiariyasu, horikitasaku, shookiyama) | 0.704 |
| 3 | Lian Matsuo | 0.702 |
| 4 | Le Viet (sleepyssr) | 0.701 |
| 5 | t.shiga (takumashiga) | 0.691 |
Rank 20 ≈ 0.669, rank 100 ≈ 0.651, median team ≈ 0.536.

### What the forum has established so far
| Thread | Finding |
|---|---|
| Aleksey Osipenko, "Random-split CV underestimates you here: 0.22 vs 0.37 – split by prompt_id" https://www.kaggle.com/competitions/name-that-llm/discussion/749593 | Each prompt is answered by every model once, so random K-fold puts 17 "siblings" in train and topic words push to wrong labels: random 0.217 vs grouped-by-prompt 0.369 vs LB 0.355 (TF-IDF + stylometry). Lab-level BA 0.587 vs model-level 0.369; Gemini Flash models confuse each other; GLM → claude-opus-5. |
| Adisak Sukul, "Grouped CV that tracks the LB, CPU baseline (LB 0.353), pitfalls, 2xT4 tips" https://www.kaggle.com/competitions/name-that-llm/discussion/750390 | Char + word TF-IDF + logistic regression: grouped CV 0.343, public 0.353; keep case/punctuation (`lowercase=False`); pitfalls: cleaned text, bf16 on T4. |
| Aleksey Osipenko, "Early Qwen layers know the author better than late ones: 0.39 vs 0.31" https://www.kaggle.com/competitions/name-that-llm/discussion/750462 | Frozen Qwen3-4B, mean-pool per layer + LR: layer 3 0.391 … last layer 0.311; 6 layers concatenated 0.414; + TF-IDF blend 0.439 CV / 0.420 LB. Style lives in early layers, topic in late ones. Hard classes: claude-fable-5 recall 0.11, gpt-5.6-sol 0.21, GLM ~0.21. |
| Masood Ahmed, "What the punctuation already tells you about the 18 models" https://www.kaggle.com/competitions/name-that-llm/discussion/750664 | Claude and GLM use em dashes + straight quotes; GPT and Gemini use curly apostrophes; Gemini rarely em-dashes; within-lab siblings differ little (OpenAI em-dash rate 9%-37%). |
| Jean Brandicourt, "train has a structure worth exploiting" https://www.kaggle.com/competitions/name-that-llm/discussion/750706 | If test keeps "each prompt answered once per model", cluster passages by prompt (91% nearest-neighbour hit rate from first ~15 words) and solve a per-group assignment (`linear_sum_assignment`, at-most-one per model). Unverified on test. |
| pjmathematician, "Reconstructed train prompts for all 150 prompt_ids" https://www.kaggle.com/competitions/name-that-llm/discussion/749455 | Claude Sonnet 5 reverse-engineered the hidden prompts (85 explain-a-concept, 45 describe/narrate, …) to guide data generation. |
| Corey Levinson, "Will the winner just be whoever can get the most extra training data?" (127 votes) https://www.kaggle.com/competitions/name-that-llm/discussion/749447 | The pay-to-win worry that dominates the forum. |
| Oleg Zholobov, "18-model wide response additional corpus" https://www.kaggle.com/competitions/name-that-llm/discussion/749629 | 135,171 external responses, 1,902 prompt groups with all 18 models; no overlap with train found. |
| Carrot Production, "Trust-Worthy Pool Dataset - 240k+ passages" https://www.kaggle.com/competitions/name-that-llm/discussion/752766 | 50 community datasets merged (1.4M raw → 226k-240k unique), 6k-19k per label. Provenance is self-reported. |
| prvsiyan, "Useful helper dataset" https://www.kaggle.com/competitions/name-that-llm/discussion/750267 | Community generation with free Kaggle AI credits ($100/month mentioned) to keep data public. |
| Darragh1Walsh, "Gemini-3.7 Flash requests routed to Gemini-3.8 Flash" https://www.kaggle.com/competitions/name-that-llm/discussion/751370 | After 3.7's 2026-10-08 deprecation, API calls may be served by 3.8 → externally generated "3.7" data may be mislabelled. |

### Transferable lessons (so far)
1. **Group CV by prompt** whenever every class answers every prompt. (749593, 750390)
2. **Authorship is style, not content:** use early-layer features and character n-grams; avoid topic leakage. (750462, 750664)
3. **How the data was built is a feature** (one answer per model per prompt → assignment decoding). (750706)
4. **Data generation is the arms race;** watch label noise from API routing. (749447, 751370)

### UNVERIFIED / open
- Host rulings on API-generated data; whether the test keeps the one-per-model-per-prompt structure; final results.

## LLMs - You Can't Please Them All (discovered)

**Why it belongs:** the purest adversarial-LLM-as-judge competition on Kaggle. Submissions are ~100-word "essays";
a hidden committee of three LLM judges scores them, and the metric rewards **making the judges disagree**. Every
winner used prompt injection tuned to fire on some judge families and not others. Pairs naturally with Drawing with
LLMs (judge exploitation) and the LMSYS/WSDM arena comps (judges as targets).

### Facts
- URL: https://www.kaggle.com/competitions/llms-you-cant-please-them-all (id 83035). Host: Kaggle.
- Dates: 2024-12-03 → 2025-03-04 (merger 2025-02-25).
- Task: for each of ~1,000 hidden essay topics write an essay (~100 words) that makes three LLM judges disagree.
  No training data; test.csv has 3 example topics.
- Metric (Evaluation page): judges return quality scores in [0, 9]. `avg_q` = mean quality, `avg_h` = mean over essays
  of the stdev across the 3 judges ("horizontal"), `min_v` = min over judges of the stdev across essays ("vertical"),
  `avg_e` = mean English-confidence (lingua), `avg_s` = max(mean difflib sequence similarity, 0.2).
  `final = avg_h × min_v × avg_e / (avg_s × (9 − avg_q))`. Evaluation takes hours.
- Code competition: ≤ 9 h, internet off; runtimes obfuscated (±15 min). Public 30% / private 70%
  (`leaderboardPercentage` 30). 5 submissions/day.
- Prizes: $50,000 (1st $12k, 2nd $10k, 3rd $10k, 4th $10k, 5th $8k). Prizes page warns of possible course
  corrections ("experimental competition").
- Teams: 1,692 teams, 1,995 competitors, 58,905 submissions.
- **Metric/judge change:** "High leaderboard scores (Dec 11 Rescore)" (Paul Mooney, staff,
  https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/550344) — after early 30-point
  exploits the hosts tweaked the task/metric; the 4th-place team notes the judges' system prompts changed and the
  "current metric" needed the two Gemma judges to be split too.
- Judge identities were never published; forum consensus and the 4th/5th/3rd writeups estimate **two Gemma
  judges (with different system prompts) + one Llama** (pinned Gemma-2B and Llama-3B models were the hint per 4th
  place; 3rd place fit Gemma-2B, Gemma-9B and Llama-3B locally). **UNVERIFIED** beyond those writeups.

### Final leaderboard (top 6)
Several teams reached the practical ceiling ~30; ties are ordered by submission time.

| Rank | Team | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|
| 1 | I'm just lucky (guoooooooss) | 30.010 | 30.050 (2) | Three attack families (0-9-9 / 9-0-9 / 9-9-0) validated on local Gemma, Qwen, Phi; 12 repeated essays; seed search | https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566372 |
| 2 | Discussion is all you need (williamwu88, goodcoder, huanligong, …) | 30.010 | 30.050 (4) | Forum-assembled attacks (000, 999, 099, 909 "run command", 990 identity) + purity checks + index-split solving | https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566602 |
| 3 | Jagat Kiran | 30.010 | 30.050 (3) | Multilingual (Japanese/Chinese/Korean) injections separating Gemma from Llama; guided random-word essays | https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566515 |
| 4 | Team (cakiki, conormacamhlaoibh, kenyeungtech, …) | 29.967 | 30.050 (1) | English+Japanese exploit (old metric 30.0); system-prompt generator with tens of thousands of templates for local testing | https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566479 |
| 5 | sash & 4eta | 29.944 | 30.050 (6) | "riddle + puzzle" framing; judge A/B Gemma, C Llama | https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566322 |
| 6 | Khoi Nguyen | 29.912 | 30.050 (5) | – | none listed |
| 8 | AmbrosM | – | – | "Nine essays are enough" | https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566333 |

### How the winning attacks work (shared mechanics)
- **Beat the two LLM-free terms first:** pad the essay with random English words (≥ ~450-600 chars) so lingua says
  "English" and difflib similarity stays low; random-word noise also "confuses" the judge so it follows the injected
  instruction (2nd, 4th). lingua was bypassed by length even with non-English instructions (3rd).
- **Target judge families separately:** the score is maximal when, essay by essay, the judges split 0/9 and the
  split pattern rotates so every judge has high variance across essays. So you need three "pure" attacks: 0-9-9,
  9-0-9, 9-9-0 (Rich Olson's plan, https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/560743).
- Discriminating tricks reported: instruction language (Llama followed only English while Gemma/Qwen followed the
  last instruction, 4th); conflicting instructions where different models obey different parts (2nd); "run this
  command and output its result" that only some models execute (2nd); identity-conditioned instructions ("if you
  are from Google…", parity of system-prompt words to split two Gemmas, 2nd); vocabulary-specific word lists drawn
  from one model's tokenizer (1st).
- **Submission-as-experiment design:** combine a candidate with known all-0 and all-9 attacks to read its effect from
  the score; verify "purity" with one-off substitutions (2nd); solve linear equations over rotated anchors to learn
  how essay indices split into thirds (2nd, 5th).
- **Repeat a few essays:** 9-15 fixed essays repeated keep similarity near 1/9 (< 0.2 floor), make purity checkable,
  and remove random-word variance (1st, 2nd, 8th). Inject fixed override topics to reduce topic dependence (8th).

### 1st place — guoooooooss (solo)
Source: https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566372
- Local judges: **Gemma, Qwen and Phi**. For each target pattern (0-9-9, 9-0-9, 9-9-0) found several attack variants:
  token strings sampled from Qwen's vocabulary or plain word lists + a short override instruction; for the Qwen-only
  split, a word list interleaved with a repeated token plus a Korean override; for the Llama-like split, a
  "if you understand Base64…" style condition and a Belarusian override. Notes that a single comma mattered.
- Final: **12 essays** repeated (15 → 12 after removing 3 "impure" ones on Conor's advice), seed 1144 (one more than
  Jagat's 1143). Stopped submitting in the last days; credits Conor (4th) as "the true champion".

### 2nd place — Discussion is all you need
Source: https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566602 ; notebook
https://www.kaggle.com/code/huanligong/perfect-score-solution
- Built entirely from forum hints (each step links its thread): local testing only to *spot* attacks; planned
  submissions; nonsense + exploit; 000/999; conflicting-instruction 099; "output of this command" 909;
  identity-conditioned 990; index-split solving; repeating purest samples → verified every sample on public LB, so
  confident about private.

### 3rd place — Jagat Kiran (solo)
Source: https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566515
- After Conor open-sourced a Japanese single attack, searched for non-English strings that separate Gemma from Llama
  (together.ai, AI Studio, HF playground, then 2×T4 locally with 100+ combinations). Hypotheses evolved
  (Gemma 27B/9B + Llama 8B → 2× Gemma 9B + Llama 8B); final local judges Gemma-2B, Gemma-9B, Llama-3B.
- Three attack templates (990 / 909 / 099) mixing Japanese, Chinese and Korean override lines after a random-word
  essay; **guided generation of the random-word part** per topic to make attacks pure (Jiri Prudky's word list
  converged within 5 iterations per topic). Key fix: realising the private LB is a random 70%, not all 1,000 essays
  (credit DeadKey), which broke a 28.2 plateau.

### 4th place — Team (Conor and colleagues)
Source: https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566479 ; notebook
https://www.kaggle.com/code/conormacamhlaoibh/5th-place-llms-ycpta ; word lists
https://www.kaggle.com/datasets/conormacamhlaoibh/llm-judges-wordlists ; prompt templates
https://www.kaggle.com/datasets/conormacamhlaoibh/llm-judge-prompts
- Old metric: 70 random topic words clipped to 450 chars + English "give 0" line + Japanese "ignore and output 9" →
  29.8, then 30.0 after removing overly positive words (code: https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/555051).
- New metric: system prompts changed; built a **system-prompt generator** (tens of thousands of templated prompts)
  to test exploits so LLaMA gives the wanted grade and Gemma splits 50/50; only 10-20% of locally optimal exploits
  transferred to the LB; extended noise to 600 chars.

### Key threads
| Thread | Takeaway |
|---|---|
| Paul Mooney (staff), "High leaderboard scores (Dec 11 Rescore)" https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/550344 | Hosts patched early exploits; promised not to play endless cat-and-mouse. |
| conor, "[Old Metric] 30.0 Exploit" https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/555051 | Public bilingual exploit + word-list scaffold. |
| Rich Olson, "Top Secret LB 30 Attack Plan (or maybe LB 22.5)" https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/560743 | Rotate three split-attacks; a single split-attack scores 0 because min_v = 0. |
| Filtered, "Guess the LLMs Judges" https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/558117 | Crowd inference of judge families. |
| Jagat Kiran, "Solution leak or Problem Statement too easy now?" https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/562972 | Late entrants jumped to 25+ in a few submissions; public notebooks (e.g. "20.293 solution", https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/563137) collapsed the board. |
| AmbrosM, "#8: Nine essays are enough" https://www.kaggle.com/competitions/llms-you-cant-please-them-all/discussion/566333 | Repeating 9 essays satisfies the similarity floor and makes testing deterministic. |

### Shake-up
Public top-6 all at 30.050; private 29.91-30.01. 4th-place team was public 1st.

### Transferable lessons
1. **LLM-judge committees are not robust:** family-specific instruction following (language, identity, tool-use
   framing) lets one input split judges deterministically. (566479, 566602)
2. **Non-LLM guard metrics are weak** (lingua by length, difflib by repetition). (566333, 566515)
3. **Use the LB as an oracle with designed experiments** (anchors, purity probes, linear solves). (560743, 566602)
4. **Repeat a few verified samples** instead of generating many noisy ones. (566333, 566372)
5. **Local judge simulation needs system-prompt diversity**, and still transfers only 10-20%. (566479)

### UNVERIFIED
- Exact judge models and system prompts (never published by the host). The full text of the 5th-place post is not
  summarised.

## NVIDIA Nemotron Model Reasoning Challenge (discovered)

**Why it belongs:** the most "LLM-post-training" competition of 2026: everyone fine-tunes the *same* model
(Nemotron-3-Nano-30B-A3B, a hybrid Mamba/attention MoE) with a **rank ≤ 32 LoRA**, scored by greedy vLLM decoding
with a 7,680-token budget. Winners did **SFT on solver-generated, deterministic chain-of-thought** — no RL, no
distillation — and won by *trace design*: what to memorise vs compute, token-efficient encodings, signature catalogs.

### Facts
- URL: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge (id 129716). Host: NVIDIA
  (compute: Google Cloud G4 VMs, RTX PRO 6000 Blackwell).
- Dates: 2026-03-16 → 2026-06-15 (merger 2026-06-08); midpoint cut-off 2026-04-09 (Open Progress Prize);
  Open Contribution Award entries by 2026-06-22.
- Task: puzzles with input-output examples of a hidden transformation; infer it and answer the final query.
  train.csv = **9,500** labelled problems in 7 families: bit manipulation (8-bit), equation numeric, cryptarithm
  (symbol-encoded equations), text cipher, numeral system (Roman), unit conversion, gravity (1st-place README).
  Hidden test = several hundred problems.
- Submission: a LoRA adapter (rank ≤ 32, with `adapter_config.json`) for Nemotron-3-Nano-30B-A3B-BF16 in
  `submission.zip`. No internet, no inference-code changes, no full-model submissions.
- Metric: accuracy. vLLM with max_lora_rank 32, **max_tokens 7680, temperature 0.0, top_p 1.0**, max_model_len 8192,
  max_num_seqs 64; answer extracted from `\boxed{}` (fallback heuristics/last number); correct if exact string match
  or relative numeric error ≤ 1e-2. A fix for answers containing `}` was deployed without rescoring
  (https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698106).
  Public 50% / private 50%.
- Prizes: $106,388 = $50k cash + 8 DGX Sparks (~$4,699 each): 1st $25k + 5 Sparks, 2nd $15k + 2, 3rd $5k + 1;
  Open Progress Prize $5k + 1 Spark (best LB at 2026-04-09); three Open Contribution Awards (best data/synthetic,
  RL, fine-tuning method) of 1 Spark each, top-10% only. Public notebook + writeup required for prizes.
- Teams: 4,185 teams, 5,234 competitors, 71,952 submissions; 5/day.
- NVIDIA recap blog (2026-07-14): https://developer.nvidia.com/blog/lessons-from-the-leaderboard-what-5000-kagglers-taught-us-about-improving-ai-reasoning/

### Final leaderboard (top 6)
| Rank | Team | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|
| 1 | NullSira (goodmeatday, refv00, reopon) | 0.920 | 0.912 (1) | Two-stage SFT on 220k + 140k solver traces; cryptarithm signature catalog + DFS; bit-manip in HEX with catalog repair | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution ; https://github.com/xrwr/kaggle-nvidia-nemotron-model-reasoning-challenge-1st-place |
| 2 | vli (victorlifire) | 0.908 | 0.884 (5) | Reverse-engineered generators; 57,600 synthetic examples; Megatron-Bridge LoRA (exact PEFT export) | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution ; https://github.com/livctr/nvidia-nemotron |
| 3 | YS-L (liauys) | 0.900 | 0.872 (42) | Compact Hui-Kang-style traces; CSP cryptarithm with multiplication patterns; **two-stage memorise-then-solve** training | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution ; https://github.com/YS-L/nvidia-nemotron-reasoning-3rd-place-solution |
| 4 | /make-no-mistakes (dipamc77) | 0.900 | 0.892 (3) | "Trainable traces"; bit manipulation 99.4% via 5 templates; ~$120 compute | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution |
| 5 | Domdolus Tolus | 0.896 | 0.888 (4) | – | topic 708538 |
| 6 | Alehandreus & Yurnero | 0.896 | 0.900 (2) | validation-centric ("2nd public / 6th private") | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-public-6th-private-place-solution |

Open Progress Prize: Tong Hui Kang (huikang), "SFT to maximize minimum logprob"
(https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915; code
https://github.com/tonghuikang/nemotron) — the base every top team credits.

### Progress Prize — Tong Hui Kang (the shared foundation)
Source: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 and
https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307
- Bet: "Nemotron can act as a simple computer after LoRA training." **Against RL** ("I already have the optimal policy")
  and **against distilling larger models** (generate the policy with code).
- Trace design rules: **deterministic** (temperature 0 → only the argmax token matters; objective = **maximise the
  minimum log-prob** over trace tokens); simple per-token derivations (division as subtraction/addition; iterate and
  classify instead of choosing among 77 words); coverage of rare operations; < 7,680 tokens; tokenisation-aware;
  generalisable (memorise only things like cipher word lists). Non-goals: diversity, conciseness, self-verification.
- Numbers: 27,850,703 training tokens for the winning midpoint submission (598,958,637 total explored); spent
  $212.48 Tinker + ~$60 Modal + $10 subscriptions; aimed for 0.877.
- Bit manipulation: per-output-bit expression search (input bit, NOT, constant, 2-input AND/OR/XOR), then choose
  consistent left/right sequences → solves 1,364/1,602 (85.1%) vs 35/160 in the earlier public notebook (llkh0a).

### 1st place — NullSira
Sources: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/709231 ; repo README
https://github.com/xrwr/kaggle-nvidia-nemotron-model-reasoning-challenge-1st-place
- **Core question: what should the model memorise through traces, and what should it compute inside the trace?**
- **Synthetic data, not LLM-written:** fit statistics from train.csv, sample prompt structure, instantiate hidden
  rules, render with the competition template, attach a **rule-based solver trace**. Main set 220,000 samples /
  890.1M tokens (bit 40k, equation numeric 10k, cryptarithm 120k, cipher 20k, numeral/unit/gravity 10k each);
  continuation pool 140,000 / 590.7M tokens, bit-heavy. Validation 1,424 (15% of train.csv).
- **SFT (Unsloth):** LoRA r=32, α=32, targets q/k/v/o, Mamba `in_proj`/`out_proj`, up/down/gate, **lm_head**;
  adamw_bnb_8bit; cosine; effective batch 16; main LR 2e-4 for 1 epoch (~119 h), continuation LR 1e-5 for 6,500 steps
  (~51 h); one RTX PRO 6000 Blackwell 96 GB. Final checkpoint `sft/106` = 91.57% on validation.
- **Equation numeric:** all operators in train explained by a **24-rule inventory** (join, join_swap, add±1,
  flip_add±1 with digit reversal, sub/absdiff/maxmodmin variants, mul±1, flips); prompt-level **mode** constraint
  (all non-join operators normal or all flip) and **group** constraint (non-join operators don't share a group);
  operator priors for `+ - *`.
- **Cryptarithm:** naive space ≈ 10!·24³ ≈ 5e10. First check join/join_swap (~5%). Then **signature catalog**:
  normalise `left+right+output` digit pattern by first occurrence (e.g. `ABCCCDD`) → enumerate 00-99 × 00-99 × 22
  rules → **4,205 signatures**, each mapping to candidate (rule, digit-string) rows. The model **memorises the catalog
  through repeated SFT traces**, retrieves candidates for the first equation, then runs DFS consistency checks on the
  rest in-trace. Credit: youkinasa's pattern-matching post
  (https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/701981). This alone
  reached 0.89 public / 0.908 private.
- **Bit manipulation:** start from huikang's solver; (1) write bit strings ≥4 long as **HEX** — Nemotron's tokenizer
  is ~one token per bit/digit, so median trace length 6,771 → 4,888 tokens (−27.8%); (2) add majority, 3-input
  parity, conditional select, 3-input compositions; (3) **repair** the selected 8-rule sequence by checking nearby
  valid sequences from a precomputed rule-sequence catalog.
- Coverage on train.csv (solver within budget / model): bit 97.50% / 93.45%; equation numeric deduce 95.47 / 95.30;
  guess 52.94 / 52.21; cryptarithm deduce 42.94 / 38.09; guess 12.80 / 12.20; cipher, numeral, unit, gravity ≈ 100%;
  total 93.16% / 92.00% (leaky, since rules were derived from train).
- Checkpoint selection: periodic LB submissions as a noisy signal; temperature 0 still not fully deterministic, so
  repeated submissions vary; final = the continuation adapter submitted twice.
- "All code was written with Codex… the ideas… almost never came from Codex."

### 2nd place — vli (Victor Li, solo)
Sources: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/711703 ; repo
https://github.com/livctr/nvidia-nemotron
- Three ingredients: rapid iteration on synthetic prompt/trace generation, discipline in **uncovering the
  data-generation process**, clean traces. Claude wrote ~95% of the data-generation code.
- **Simple CE beat** focal loss, token reweighting and multi-stage schedules.
- train.csv as validation (all training data synthetic): bit manipulation synthetic 0.9906 / LoRA 0.9732; overall
  0.9107 / 0.9074; cryptarithm left mostly untouched (0.096). Public 0.884, private 0.908.
- **Infrastructure matters:** Tinker LoRA needed a lossy SVD to convert to PEFT; switching to **Megatron-Bridge**
  (exact mapping) raised bit-manip accuracy 0.81 → 0.89 with nothing else changed. A late-found QKV-interleave bug in
  his exporter barely mattered (public 0.884 → 0.880, private 0.908 unchanged).
- Final: 57,600 examples, 159M tokens, 1 epoch, LR 2e-4 → 1e-5 cosine, seq 8192 packed, global batch 12, EP=4
  (accidentally tied LoRA across experts ~4×), per-expert LoRA on MoE FC, targets incl. Mamba in/out proj and
  output layer ("extremely important"); 4× A100 80GB, 11 h. Strong diminishing returns after ~80M tokens.

### 3rd place — YS-L (solo)
Sources: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/709136 ; repo
https://github.com/YS-L/nvidia-nemotron-reasoning-3rd-place-solution
- SFT on deterministic traces "without any hidden deduction or jumps". Minimal numeric solver with better
  guessing; huikang-style bit traces compacted with hex + majority search; augmentation for most categories;
  CSP-like cryptarithm with multiplication patterns + backtracking solving 24.2% within budget.
- **Two-stage training:** stage 1 teaches the model to *recall* digit domains / candidate mappings for any
  signature (drill prompts like `<RECALL_DOMAINS> sig=… type=…`); stage 2 starts from that LoRA and trains on full
  traces (+3K drill samples to retain memory). On 12K cryptarithm samples: direct 0/11 solvable problems vs
  two-stage 7/11; final mix ablation: direct 12/17 (private 0.888) vs two-stage 16/17 (private 0.900).
- Training: huikang's loop with cut cross-entropy, micro-batch 4, **effective batch 4** (bigger under-trained),
  LR 2e-4 linear, LoRA r=32 α=32 fp32, targets incl. in/out proj and lm_head, **MoE tie weights off**. Stage 1:
  44,136 examples, 159M tokens, 25.2 h; stage 2: 72,377 examples, 299M tokens, 32.7 h, one RTX PRO 6000.
- **Validation:** 10% stratified holdout (954 rows) excluding augmentations of held-out items; tracked private LB,
  **not** public — "the wall at 0.86 is clearly visible on the public LB"; one 0.85-public submission was 0.90
  private. Extra metric: **perfect-trace rate** (>85% bit, >95% cipher/numeric).

### 4th place — /make-no-mistakes (dipamc77)
Source: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/712419
- "Trainable traces": refine bit-manip traces until >95% accurate with only ~2,500 examples (huikang's ≈82% at
  that size); bit manipulation 99.4% because it maps to only 5 templates; MoE experts untied, batch 12/16; ~$120
  total compute (Colab Pro G4 + Kaggle); 9,500 originals + 15,000 synthetic traces; code by Claude (Opus 4.8).

### Key threads
| Thread | Takeaway |
|---|---|
| Tong Hui Kang, "SFT to maximize minimum logprob" (242 votes) https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 | The design doctrine; won the Progress Prize. |
| Tong Hui Kang, "Strategy to solve 85% of bit manipulation" https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307 | Per-bit expression search; starting point of 1st/3rd. |
| Tong Hui Kang, "Visualize the problems and completions from the base model" https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212 ; https://nemotron.huikang.dev/base | Base-model behaviour on all 9,500 problems. |
| Donald Galliano III, "Answers To Everything Data: 100% Solve Rate" https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461 | Claimed reverse-engineering of all generators; complaint that Kaggle Blackwell GPUs made training slow ("pay-to-play"). |
| Komil Parmar, "Why GRPO is Painfully Slow on Nemotron (and the Fix)" https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690161 | RL rollout throughput (~2 tok/s) was a practical barrier; SFT dominated. |
| Zhuang Jia, "Mainstream LLM Performance Comparison" https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684283 | Zero-shot frontier models: Gemini-3.1-Pro 0.81, Claude-Opus-4.6 0.78, DeepSeek-V3.2 0.74 — below the fine-tuned 30B LoRA winners (0.92). |
| youkinasa, "A Pattern-Matching Approach to Solving Symbolic Arithmetic Puzzles" https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/701981 | Signature-keyed operation tables (seed of the 1st-place catalog). |

### Shake-up
Moderate: 3rd place rose from public 42nd; public 2nd (Alehandreus & Yurnero) finished 6th. A visible public "wall" at
~0.86-0.87 hid real progress (3rd-place validation analysis).

### Transferable lessons
1. **For greedy-decoded reasoning, write the policy in code and SFT on it**; RL and distillation lost. (689915, 709231, 711703)
2. **Decide memorise vs compute:** memorised catalogs (signatures, rule sequences) + short in-trace search. (709231, 709136)
3. **Token budget is a design constraint:** HEX encoding cut traces 28% because the tokenizer is digit-level. (709231)
4. **Train memory first, execution second** (two-stage LoRA). (709136)
5. **Reverse-engineer the generator; measure per category** — aggregates hide regressions. (711703, NVIDIA blog)
6. **Your training stack can be the bottleneck:** lossy LoRA export cost 8 points of bit-manip accuracy. (711703)

### UNVERIFIED
- Recipients of the three Open Contribution Awards. Hui Kang's Progress Prize is stated in his repo README
  (https://github.com/tonghuikang/nemotron) and thread title; amount per the Prizes page.

## Deep Past Challenge - Translate Akkadian to English (discovered; LLMs as data engineers)

**Why it belongs (scope note):** the *submitted* models were ByT5 seq2seq models, not LLMs, but every top team used
frontier LLMs as the **data factory** (Gemini multimodal OCR-to-parallel-text extraction, LLM sentence alignment,
Claude-generated vocabulary/grammar drills) and 3rd place used a fine-tuned **Qwen3-8B reward model** to pick
translations. Include it as the "LLM builds the training set for a small specialist model" episode.

### Facts
- URL: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation (id 121150). Host: Deep Past
  Initiative (https://www.deeppast.org).
- Dates: 2025-12-16 → 2026-03-23 (entry/merger 2026-03-16).
- Task: translate transliterated **Old Assyrian** (Akkadian) tablet text into English, one sentence per test row.
  Data: >8,000 transliterated texts with metadata, aligned English for a subset (train.csv ≈ 1,500 per the Data page;
  the 2nd-place post describes ~6,500 documents after a data update — **UNVERIFIED which version**), plus unprocessed
  text from ~900 scholarly publications to mine. Mid-competition data updates (Ryan Holbrook "Data Update", "Dataset
  Update - Mind the Gaps").
- Metric: geometric mean of corpus-level BLEU and chrF++ (SacreBLEU micro-averaged). Public 34% (`leaderboardPercentage`).
- Code competition: CPU/GPU ≤ 9 h, internet off.
- Prizes: $50,000 (1st $15k, 2nd $10k, 3rd $8k, 4th $7k, 5th-6th $5k). Teams: 2,674; 68,346 submissions.

### Final leaderboard (top 6)
| Rank | Team | Private | Public (rank) | Approach | Writeup |
|---|---|---|---|---|---|
| 1 | DataTech Club (guoqingu, hongori, leechengju, xiakie) | 42.8957 | 41.6694 (1) | 11× ByT5-XL ensemble, 10 candidates each, MBR selection; data from PDFs via GLM-OCR layout + Gemini-3.0-pro page-image extraction | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684353 |
| 2 | KE WU (wukeneth) | 41.9311 | 41.0478 (3) | Vanilla ByT5-large; three-stage LLM sentence alignment + ~60 extra OCR'd publications + orthography normalisation | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684345 |
| 3 | How I Learned to Stop Worrying (darraghdog, conjuring92) | 41.5091 | 41.6451 (2) | ByT5-Large + ByT5-XL with continued pretraining on Claude-generated drills; Qwen3-8B pairwise reward model selects per sentence | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684425 |
| 4 | Wisdom and Vitaly | 40.8899 | 39.9734 (8) | – | topic 685925 |
| 5 | ebi & okumura | 40.8077 | 40.4156 (4) | – | topic 684847 |
| 6 | M & J & M | 40.7360 | 40.0310 (7) | 15-model ensemble on diverse data sources | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684231 |
| 24 | Geremie Yeo | – | – | post-trained Qwen2.5 32B and 72B on Gemini-OCR'd pairs (LLM-as-translator route) | https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684189 |

### 1st place — DataTech Club ("Data Quality Dictates Everything")
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684353
- **Discarded train.csv**; rebuilt training data from host-provided and self-collected PDFs. Structured books (AKT
  series, ICK 4, Larsen 2002, Michel 2020, POAT): GLM-OCR layout analysis + regex → tablet → page-range map; then
  **Gemini-3.0-pro-preview** reads the page images per tablet and emits sentence-level transliteration/translation
  pairs (translating German/Turkish to English). Poorly structured sources via a vibe-coded screenshot-to-API front end.
  Prompts (in Chinese) segment by logical sentence using the English translation as anchor, not by tablet line.
- Model: **11 ByT5-XL** variants on different data mixes (10 with Seq2SeqTrainer bs 64, 3 epochs; 1 with a
  data-quality-weighted loss, bs 48, 4 epochs); each produces 10 candidates (beam + sampling at several temperatures);
  a community **MBR selector** picks the output; CTranslate2 int8-float32 on 2×T4, using almost the full 9 h.
- Best (unselected) submission 41.5 public / 43.2 private; selected one added repeated-n-gram cleanup (41.6 / 42.8).
  Principle: preprocess carefully, post-process as little as possible (LB-tuned post-processing = shake-up risk).

### 2nd place — KE WU (solo)
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684345
- "Every point of improvement came from building a larger, cleaner training corpus"; model unchanged ByT5-large.
- Three LLM pipelines over set partitions: **Breaker** (split full translations into sentence pairs using OARE
  word indices; 1,558 docs → 9,378 pairs), **Fixer** (re-anchor corrupt pre-segmented fragments), **Generator**
  (synthesise translations from morphological metadata for untranslated texts). Plus ~60 extra OCR'd academic
  publications and orthography normalisation.

### 3rd place — darraghdog & conjuring92
Source: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684425
- Two stages per model: **continued seq2seq pretraining** on synthetic data that teaches Old Assyrian fundamentals,
  then fine-tuning on scholar-quality pairs. **Claude 4.5 Sonnet** generated vocab drills (~1k lemmas from CAD/eSAD
  × senses → ~90k pairs, plus ~60k polysemy mixes) and grammar drills from ~500 rules extracted from Kouwenberg 2017.
- Ensemble ByT5-Large + ByT5-XL with a fine-tuned **Qwen3-8B pairwise reward model** choosing the better translation
  per sentence. Claude Code used for extraction, auditing, pipelines and research.

### Transferable lessons
1. **Frontier LLMs as data engineers** (OCR + alignment + synthetic drills) beat bigger models. (684353, 684345, 684425)
2. **Small byte-level specialists win under a 9 h offline budget;** LLM translators (24th) did not. (684189 title)
3. **Teach fundamentals with synthetic curricula, then style with real data.** (684425)
4. **Selection matters:** MBR or a learned pairwise judge over many candidates. (684353, 684425)

## Google - The Gemma 4 Developer Agent Competition (discovered; ongoing, state as of 2026-10-27)

**Why it belongs:** post-train Gemma 4 into an offline SWE-bench-style coding agent. The submission is an **agent
config** (Google ADK YAML: prompts, sub-agents, skills, LoRA adapters), not a notebook of predictions. Running until
December 2026, so only the setup and early forum findings exist.

### Facts
- URL: https://www.kaggle.com/competitions/gemma-4-developer-agent (id 149921). Host: Google DeepMind. Companion
  paper track: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper (391 teams, $35k, deadline 2026-11-12).
- Dates: 2026-09-23 → 2026-12-02 (entry/merger 2026-11-25; optional paper 2026-11-12).
- Task: fix real Python issues (bug fixes / feature requests). Training set = **129 public tasks** from fastapi,
  Textualize/rich, psf/requests, encode/httpx with `patch`, `test_patch`, repo snapshots, NetworkX AST call graphs,
  256-d node embeddings (for `search_similar_code`), offline wheels, sandbox Dockerfiles. Hidden test ≈ **120 tasks**
  from **private repositories**, split evenly public/private; curated by fail-to-pass / pass-to-pass verification and
  a "frontier model can nearly solve it" check.
- Submission: `submission.zip` with `agent.yaml` (ADK Agent Config), optional prompts, sub-agents, ADK skills (scripts
  run in the sandbox), PEFT LoRA adapters. Models allowed: `gemma-4-31b-it-qat-w4a16-ct`, `-12b-`, `-e4b-`, `-e2b-`
  (smaller ones added later, https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748287); one base
  model per submission, different LoRAs per agent allowed. Tools: run_command, submit_patch, get_status, read/edit/
  write_file, get_code_neighbors, search_similar_code, get_code_subgraph.
- Metric: % of tasks whose patch makes the validation tests pass (SWE-bench style). 12 h limit for all tasks.
  1 submission/day. Public 50%.
- Prizes: $65,000 in the Rules (1st $37k, 2nd $18k, 3rd $10k; API reward field says $75,000) plus a **$10,000 Small
  Model bonus** for the best E2B/E4B entry (https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/751104).
  Winner licence Apache 2.0.
- Teams (2026-10-27): 3,932; 25,462 submissions. Public LB top = 0.24 (5 teams tied), then 0.22.

### Early findings from the forum
| Thread | Finding |
|---|---|
| Geremie Yeo, "The public leaderboard has exactly 58 tasks" https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743506 | Floor-rounded score set {0.00, 0.01, 0.03, …} is only consistent with n = 58 public tasks; 0.24 ≈ 14 tasks. |
| Ruijie Zhu, "Stronger models barely beat Gemma 4 31B in this harness" https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747168 | Same harness, 124 local tasks: Gemma 4 31B 54/124 and 53/123 on two passes; DeepSeek V4 Pro 58/124; Qwen3.6-27B 59/123. 19 of 123 tasks flipped between identical Gemma passes → harness and variance dominate. |
| Ayush Thakur, "Gemma 4 agent experiments: sampling, thinking, time limits, tool ablations" https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746250 | Keep thinking on (temp 1.0, top-p 0.95, 4,096 thinking tokens); 15-min per-task limit matched 60-min quality (54/129); graph retrieval and analyzer sub-agent gave no clear gain. |
| Dmitrii Gluzdov, "31B vs 12B vs E4B with the same agent files" https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/750974 | On 31 unseen-repo tasks: 31B solved 9-11; 12B 4 (writes 2× faster but 3× more, so slower per task); E4B 0. |
| Nur, "fastapi tasks can't pass even with the gold patch" https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747248 | Missing `typing_inspection` wheel breaks pydantic imports in the public harness container (local repro; host response not read). |
| c-number, "Clarification on distillation from external LLMs" https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742807 | Open question whether trajectories from proprietary models may train the Gemma LoRA. |

### UNVERIFIED / open
- Final results; host rulings on distillation and the fastapi environment; reason for the $65k vs $75k figure.

## Autonomous Agent Prediction (Beta) (discovered; short)

**Why it belongs:** Kaggle's first "the submission is an LLM agent that competes in Kaggle" format. Swag prizes only,
so treat as a format preview, not a solution bank.

- URL: https://www.kaggle.com/competitions/autonomous-agent-prediction-beta (id 151211). Host: Kaggle.
  Dates 2026-07-07 → 2026-08-06. 570 teams, 2,369 submissions, 1/day. Prize: Kaggle merchandise for top 3.
- Task: submit an ADK agent config (prompts, tools, skills, sub-agents) that, inside a Docker sandbox on a CPU,
  runs **two complete mini-competitions per submission** (binary classification, AUC) on hidden synthetic datasets
  from the same generating family as 16 provided training datasets: explore, train, submit (≤30 submissions), pick 2
  finals. Budgets per session: **60 minutes and $2.00 of LLM tokens**; models from `models.yaml` (e.g. Gemini 3.1
  Flash Lite, 3.1 Pro preview, 3.5 Flash with listed token prices).
- Final top: adamdavis99 0.783, then six teams at 0.782 (public ranks 4, 278, 252, 430, 198, 183 → heavy shake-up
  from a near-ceiling public LB). No top-2 writeups.
- Threads: 3rd place (VISHALINI K, https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/737407):
  agent trains LightGBM/XGBoost/CatBoost + LR, chooses best single vs blend by honest OOF, submits a fast baseline
  early as an anchor. Naji, "Exploiting the Freeroll Fallback Rule"
  (https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/730539): secure a Flash-Lite
  baseline (~$0.10, 10 min) then let gemini-3.1-pro-preview iterate with the remaining budget, since failure falls
  back to the best public submission. Chris Deotte, "Is LLM budget only $2?"
  (https://www.kaggle.com/competitions/autonomous-agent-prediction-beta/discussion/723806).

## Considered and excluded

Discovery used `CompetitionService/ListCompetitions` with search terms (LLM, language model, large language, Gemma,
Qwen, Nemotron, agent, coding agent, reasoning, chatbot, prompt, text generation, judge, AI generated, fine-tune,
translation, math) plus the TaeyanG4 solutions index (https://github.com/TaeyanG4/kaggle-top-solutions-and-strategies,
`data/competitions.csv`) filtered to 2023+ and Featured/Research.

| Competition | Dates | Reason excluded / short note |
|---|---|---|
| AI Agent Security - Multi-Step Tool Attacks (`ai-agent-security-multi-step-tool-attacks`, OpenAI/Google/IEEE) | 2026-06-11 → 2026-09-01 | LLM-centric (red-teaming GPT-OSS-20B and Gemma 4 agents in a sandbox), 4,186 teams, $50k + 2×$2.5k working-note awards. Left without a full section here: the winning material is attack-method detail. Facts for the writer: private LB 1st xz 46.425 (public rank 19), 2nd "no shake thanks :)" 42.99, 3rd "live to win" 42.175; public leader poijio 147.53 finished 6th at 37.69 — a hidden stricter private guardrail collapsed public scores. Writeups: https://www.kaggle.com/competitions/ai-agent-security-multi-step-tool-attacks/discussion/739181 , 739208, 739040, 738981 ("Transfer Was the Real Attack"). Evaluator changes: 712642, 733058 (leaderboard invalidated and refreshed). |
| ARC Prize 2026 (ARC-AGI-2, ARC-AGI-3, Paper Track) | 2026-03-25 → 2026-11 | Excluded by brief (ARC Prize family). |
| Gemma 4 Developer Agent Paper Track (`gemma-4-developer-agent-paper`) | 2026-09-23 → 2026-11-12 | Paper track, no leaderboard. |
| Kaggle Measuring AGI - Cognitive Abilities (`kaggle-measuring-agi`) | 2026-03-17 → 2026-04-16 | Benchmark-design hackathon, no metric leaderboard. |
| MedGemma Impact Challenge, Gemma 4 Good, Gemma 3n Impact, Gemini Long Context, Unlock Global Communication with Gemma, AI Assistants for Data Tasks with Gemma, Tunix, BigQuery AI, gpt-oss red-teaming, OpenAI to Z, 5-Day AI Agents | 2024-2026 | Hackathons / courses (excluded by brief or no leaderboard). |
| LLM Classification Finetuning (`llm-classification-finetuning`) | 2024-10-16 → 2030 | Getting-started clone of LMSYS Arena (excluded family). |
| USPTO - Explainable AI for Patent Professionals | 2024-04-24 → 2024-07-24 | Boolean patent-search query generation; not LLM-centric by design (UNVERIFIED whether any top solution used LLMs). |
| Stable Diffusion - Image to Prompts | 2023-02-13 → 2023-05-15 | Image → prompt embedding (CLIP/BLIP-style), text-encoder metric; not LLM-centric. |
| CommonLit Evaluate Student Summaries; Learning Agency Lab AES 2; Linking Writing Processes | 2023-2024 | NLP regression won mostly by DeBERTa ensembles (UNVERIFIED for every top-5 entry). |
| AI Village Capture the Flag @ DEFCON31 | 2023 | Mixed security CTF; only some challenges involve LLMs. |
| Benetech - Making Graphs Accessible | 2023 | Chart-to-data with vision-encoder-decoder models; not LLM-centric. |
| Pokémon TCG AI Battle Challenge (+ Strategy, Playground), Orbit Wars, Kaggriculture | 2026 | Game-agent simulations (covered by the game-agents series). |
| Squares in Squares | 2026-10-21 → 2027-07 | Packing optimisation, not LLM. |
| Classification of Math Problems by KAChallenges | 2025 | Community competition (1st place ensembled Qwen2.5-14B/Qwen3-14B per search snippet); small, no prize money — UNVERIFIED. |
