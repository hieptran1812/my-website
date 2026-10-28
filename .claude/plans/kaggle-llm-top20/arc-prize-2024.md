# ARC Prize 2024 (ARC-AGI-1): top-20 addendum

- Slug: `arc-prize-2024` (competition id 67357, forum 4374556). Final board pulled from `competitions.LeaderboardService/GetLeaderboard` on 2026-10-28.
- Deep coverage of ranks 1 to 5 already lives in `.claude/plans/kaggle-llm-writeup-bank-part2.md`, H2 `## 6. ARC Prize 2024 (ARC-AGI-1)` (lines ~340 to 418). This file does not repeat it; it adds the 3rd-place method (missing from the bank), extra detail for 4th/5th, rank 13, and the state of ranks 6 to 20.
- Board mechanics that matter for every row below: the leaderboard was 100% of the 100 hidden tasks, so public = private; scores are % of 100 tasks, so they move in 1-point steps and ties are everywhere. The API gives distinct ranks within a tie (tiebreak = earlier submission); ties are shown exactly as given.

## Top-20 table

| Priv rank | Team | Members (display name (username)) | Private | Public rank (score) | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | the ARChitects | Jan Disselhoff (insanitycheck); Daniel Franzen (dfranzen) [David Hartmann is a paper co-author but not on the Kaggle team roster] | 53.50 | 1 (53.50) | paper https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf (no Kaggle writeup link on the team entry) | Mistral-NeMo-Minitron-8B, 64-token vocab, Re-ARC/ARC-Heavy pretraining, test-time LoRA, DFS sampling, augmentation-stability scoring (bank) |
| 2 | Guillermo Barbadillo | Guillermo Barbadillo (ironbar) | 40.00 | 2 (40.00) | https://www.kaggle.com/competitions/arc-prize-2024/writeups/guillermo-barbadillo-2nd-place-solution-for-the-ar (topic 545671) | Omni-ARC: Qwen2.5-0.5B multi-task + per-task TTT + AIRV voting, ensembled with the 2020 DSL (bank) |
| 3 | alijs | alijs (alijs1) = Agnis Liukis (named in the technical report) | 40.00 | 3 (40.00) | https://www.kaggle.com/competitions/arc-prize-2024/writeups/alijs-3rd-place-solution (topic 550328) -> notebook https://www.kaggle.com/code/alijs1/arc-prize-2024-solution-3rd-place-score-40 | Public "26" DSL notebook + ~14 points of hand-written, LB-pruned Python task functions; no ML |
| 4 | William Wu | William Wu Chengyuan (williamwu88) | 37.00 | 4 (37.00) | https://www.kaggle.com/competitions/arc-prize-2024/writeups/william-wu-4th-place-solution (topic 550414) | 2020 DSL/DAG search + decision trees + CNNs, ensembled by vote; no LLM |
| 5 | PoohAI | Aleksandr Emchinov (greylord1996); gromml (gromml); lead_n_validate (leadnvalidate); Sam (samalkubentayeva) | 37.00 | 5 (37.00) | https://www.kaggle.com/competitions/arc-prize-2024/writeups/poohai-5th-place-solution (topic 550336) | Ensemble of classical solvers + error-pattern post-filters + LB probing by task order |
| 6 | [Deleted] eff96e59-... | [Deleted User] | 37.00 | 6 (37.00) | no writeup found (account deleted) | unknown |
| 7 | vialactea | vialactea (vialactea) | 36.00 | 7 (36.00) | no writeup found | unknown |
| 8 | Dawid Kopiczko | Dawid Kopiczko (dawkopi) | 36.00 | 8 (36.00) | no writeup found | unknown |
| 9 | GOA | gaonan (gaonan) | 36.00 | 9 (36.00) | no writeup found | unknown |
| 10 | optima | starpentagon (starpentagon); Ueddy (ku0807) | 35.00 | 10 (35.00) | no writeup found | unknown |
| 11 | Giotto.ai | Wallyson R. Lemes De Oliveira (wallyoliveira); Gabriele (gabrielebeltramo); Matteo Caorsi (matteocaorsi); Luca Crosato (lucacrosato5) | 35.00 | 11 (35.00) | no writeup found | unknown (company team) |
| 12 | Junhua Yang | Junhua Yang (junhuay) | 30.00 | 12 (30.00) | no writeup found | unknown |
| 13 | Nikola Hu | Nikola Hu (zoenguyenramirez) | 30.00 | 13 (30.00) | https://www.kaggle.com/competitions/arc-prize-2024/writeups/nikola-hu-sharing-my-arc-prize-2024-code-31-points (topic 546302) + topic 545747 | From-scratch 162M transformer, 19-token vocab, 2D grid positional encoding, test-time training, reverse augmentation, ensembled with public notebook |
| 14 | OuterloopAI | MarcoMascorro (mascobot); Andres Campero (andrescampero) | 30.00 | 14 (30.00) | no writeup found | unknown |
| 15 | Ahmed El Fazouani | Ahmed El Fazouani (ahmedelfazouan) | 29.00 | 15 (29.00) | no writeup found | unknown |
| 16 | h309 = o1411 | h309 = o1411 (h309o1411) | 29.00 | 16 (29.00) | no writeup found | unknown |
| 17 | LIGHT | LIGHT (berserker408) | 29.00 | 17 (29.00) | no writeup found | unknown |
| 18 | 暗黑AGI | 暗黑AGI (boristown) | 29.00 | 18 (29.00) | no writeup found | unknown |
| 19 | Benjamin Kovacs | Benjamin Kovacs (benjaminkovacs) | 29.00 | 19 (29.00) | no writeup found | unknown |
| 20 | linear regression | yuanzhe zhou (yuanzhezhou) | 29.00 | 20 (29.00) | no writeup found | unknown (see note) |
| (21) | Lyrialtus | lyrialtus | 28.00 | 21 (28.00) | https://www.kaggle.com/competitions/arc-prize-2024/discussion/550209 | Recolouring preprocessing for icecuber on top of the 26 baseline; listed because it explains what a 26-to-29 score typically is |

Writeups found: 6 of 20 (ranks 2, 3, 4, 5, 13 on Kaggle; rank 1 as a paper). Ranks 6 to 12 and 14 to 20 have no writeup in the forum (241 topics scanned, all pages) and none on arcprize.org (its winners page lists only ranks 1 to 5 with code links).

## Per-team synthesis

### Rank 1 (the ARChitects): only roster fact the bank lacks
The Kaggle team entry has two members (insanitycheck, dfranzen); the team entry carries no writeup link. The bank's three-author line comes from the paper. arcprize.org links their winning code as Greg Kamradt's re-run notebook https://www.kaggle.com/code/gregkamradt/arc-prize-v8 .

### Rank 3: alijs (Agnis Liukis), 40.00 (bank marked this UNVERIFIED; now read)
Source: the notebook's markdown summary, https://www.kaggle.com/code/alijs1/arc-prize-2024-solution-3rd-place-score-40 (pulled with Kaggle's anonymous kernel-pull endpoint).
- The final submission is an ensemble of two parts: the best public notebook, Gholami & Kazeminia's "3-arc24-developed-2020-winning-solutions" (the icecuber C++ 2020 winner plus other 2020 solvers; scores 26 alone; uses Hansueli Jud's dataset of icecuber source files), and his own code, which added **14 points**.
- His part is "some kind of guided brute-force": a library of hand-written Python functions, each tried against the task's demonstration pairs; any function that reproduces at least one demonstration is used on the test input.
- **The leaderboard was the development loop.** Because public = private, he used LB feedback to tune functions toward specific hidden tasks and to delete every function or code path that did not raise the score. Deleting dead code cut runtime, which is what made room to run the public 26-point solver in the same 12 h notebook. He says plainly the result is "extremely overfitted to the given test-set", and congratulates teams who resisted that.
- Focus areas: symmetry and pattern-completion tasks; re-colouring squares vertically/horizontally by rule.
- The one transferable idea: many tasks are "a known reasoning step plus one small extra step" (mirror the shape and also recolour it; draw the spiral maze and also colour it). So solve first in a **colour-independent binary view** (background vs not-background); if the geometry works, search separately for the right colouring.
- The ARC technical report credits this as the best single-submission brute-force/DSL score (40%), and contrasts it with transduction-only systems at about the same level.
- No ML at all; no hardware claims beyond the standard notebook.

### Rank 4: William Wu, 37.00 (additions to the bank)
Source: topic 550414; code https://www.kaggle.com/code/williamwu88/fork-of-small-sample-arc24-7d97ca .
- Components ranked by importance: DSL search over a DAG; decision trees; CNNs; augmentation/preprocessing (diagonal flips, colour switching, symmetry).
- Sources merged: 2020 solutions by icecuber, golubev, szabo7zoltan, ilialar, mehrankazeminia and somayyehgholami, plus Hodel's arc-dsl.
- The gain over 2020 came mostly from the bigger box: 30 GB RAM and 12 h (vs 9 h in 2020) let him search deeper, train longer and fit more models into the ensemble.
- Ensembling: majority vote, with icecuber manually prioritised and exempt from the vote; plus "probabilistic trials", choosing a model with probability roughly proportional to how many new problems it solves.
- He was 7th at the deadline and moved to 4th after teams above withdrew (see shake-up). His own conclusion: TTT approaches of the top teams are the state of the art.

### Rank 5: PoohAI, 37.00 (additions to the bank)
Source: topic 550336; code https://www.kaggle.com/code/gromml/arc-prize-2024-poohai-solution ; arcprize.org links a paper PDF on Google Drive and a re-run notebook titled "arc-prize-2024-8th-place-solution" (the team's pre-withdrawal rank).
- Three ideas: ensemble different solvers; post-processing filters for each solver's typical mistakes (the genetic-algorithm solver draws redundant vertical/horizontal lines; wrong output shape); and brute-force probing (bank covers the probing mechanics).
- Lessons stated: removing wrong attempts is easier than picking the right one; LLMs "could be added" to the ensemble but would need fine-tuning.
- Community reaction in the thread: Barbadillo called leaderboard probing "totally against the spirit of the competition" and suggested the AIMO-1 style API that releases one task only after the previous answer (topic 550336 comments).

### Rank 13: Nikola Hu, 30.00 (writeup says 31 points)
Sources: topic 546302; topic 545747; code https://github.com/zoenguyenramirez/arc-prize-2024 ; notebook https://www.kaggle.com/code/zoenguyenramirez/fs-1-inherted-newzoe-v7 .
- The only top-20 entry besides ranks 1 and 2 with a neural transduction model, and the only one trained **from scratch**: a custom decoder transformer, 162M parameters, 18 layers, 4 attention heads, joined in the final month.
- Vocabulary of 19 tokens: 10 colour tokens plus PAD, START, START/END_INPUT, START/END_OUTPUT, END, ROW_SEPARATOR.
- **Custom positional encoding**: each token carries row, column, a "topology/chart" index and a grid (sample) index, each looked up in a learned grid-encoding table, scaled and concatenated. He reports it beats standard 1D NLP positional encoding.
- Data: Re-ARC and BARC datasets (acknowledged).
- **Active inference** (test-time training on the task's demonstrations), adopted from MindsAI's public interview; it was the "game-changer" when he was stuck at 4 points. "Reverse augmentation" (augment, predict, invert, compare), which he first called "consistency" and later found matched MindsAI's AIRV.
- Loss only on output tokens. Training: 3x RTX 4090, then 2x 4090 from 6 Nov.
- Did not work: a "mask hack" letting input tokens attend to all other input tokens (no gain, disabled); a "progressive head" (impact unclear because of bugs). Early versions scored 30 to 40% on the public eval set but only ~4 on the hidden set.
- Final 31 = ensemble with the public notebook approach (exact split not stated).

### Ranks 6 to 12 and 14 to 20: no writeup
- 6 [Deleted user], 7 vialactea, 8 Dawid Kopiczko, 9 GOA (gaonan), 10 optima (starpentagon, Ueddy), 11 Giotto.ai, 12 Junhua Yang, 14 OuterloopAI, 15 Ahmed El Fazouani, 16 h309 = o1411, 17 LIGHT, 18 暗黑AGI (boristown), 19 Benjamin Kovacs, 20 linear regression (yuanzhe zhou): no writeup in the forum, the writeups list linked from the team entries, arcprize.org winners page, or web search (Giotto.ai, OuterloopAI, Dawid Kopiczko searched by name).
- Rank 20 yuanzhe zhou asked in topic 542428 (https://www.kaggle.com/competitions/arc-prize-2024/discussion/542428) whether C++ solvers are "the only valid way", noting the best public notebook is C++ icecuber-based; a reply estimated tree models (XGBoost/LGBM) solve only 3 to 5 hidden tasks. Indicative only.
- Context for the 29-point cluster (ranks 15 to 21): the public "26" notebook was the shared floor (alijs, PoohAI, Lyrialtus all build on it); Lyrialtus (21st, 28) reached +2 with a recolouring preprocessor for icecuber. It is plausible but UNVERIFIED that most 29-point teams are 26-notebook ensembles plus a few solver tweaks.

## Technique frequency matrix

Columns are private ranks. Y = stated in a writeup/paper, N = writeup implies not used, ? = no source.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM / transformer transduction | Y | Y | N | N | N | ? | ? | ? | ? | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? |
| Pretrained LLM backbone | Y (8B) | Y (0.5B) | N | N | N | ? | ? | ? | ? | ? | ? | ? | N (scratch 162M) | ? | ? | ? | ? | ? | ? | ? |
| Test-time training on demos | Y | Y | N | N | N | ? | ? | ? | ? | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? |
| D8 / colour augmentation | Y | Y | N | Y | ? | ? | ? | ? | ? | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? |
| Augment-invert-vote (AIRV) or aug. rescoring | Y | Y | N | N | N | ? | ? | ? | ? | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? |
| Synthetic / generated data (Re-ARC, BARC) | Y | Y | N | N | N | ? | ? | ? | ? | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? |
| 2020 DSL / icecuber search in ensemble | N | Y | Y | Y | Y | ? | ? | ? | ? | ? | ? | ? | Y | ? | ? | ? | ? | ? | ? | ? |
| Hand-written task functions | N | N | Y | N | Y | ? | ? | ? | ? | ? | ? | ? | N | ? | ? | ? | ? | ? | ? | ? |
| Classical ML (trees / CNN) | N | N | N | Y | ? | ? | ? | ? | ? | ? | ? | ? | N | ? | ? | ? | ? | ? | ? | ? |
| Post-processing error filters | N | N | N | N | Y | ? | ? | ? | ? | ? | ? | ? | N | ? | ? | ? | ? | ? | ? | ? |
| LB feedback used to tune / probe | N | Y (subs as extra compute only) | Y | ? | Y | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| Custom tokenizer / minimal vocab | Y (64) | N (Markdown text) | N | N | N | ? | ? | ? | ? | ? | ? | ? | Y (19) | ? | ? | ? | ? | ? | ? | ? |

Counted summary (6 sources: ranks 1, 2, 3, 4, 5, 13):
- 3 of 6 used a neural transducer with test-time training (1, 2, 13); all three also used augmentation-and-vote/rescoring.
- 5 of 6 leaned on the 2020 DSL / icecuber lineage in some form (2, 3, 4, 5, 13); only rank 1 had no classical component.
- 3 of 6 had no ML of any kind in the scored path (3, 5) or only classical ML (4).
- 2 of 6 openly used leaderboard feedback on the hidden set to shape the solution (3 tuned/pruned, 5 probed by task order).

## Consensus vs divergence

- Table stakes: the public 26-point notebook (icecuber + 2020 solvers) was the floor under almost every team that wrote up, LLM or not. Ranks 2, 3, 4, 5 and 13 all ensemble with it or with its components.
- What separated rank 1 from 2 to 20: rank 1 alone did not need the DSL; a single 8B transducer with TTT, DFS enumeration and augmentation-product scoring reached 53.5. The gap from 1st (53.5) to 2nd (40) is larger than from 2nd to 20th (29).
- Top 5 vs 6 to 20 is not visible: 3rd to 11th span 40 to 35 (5 tasks), and nobody from 6 to 12 wrote up.
- Contradiction 1 (how to use the public = private board): alijs (3rd) used LB feedback to keep only functions that scored and called the result "extremely overfitted" (notebook); PoohAI (5th) probed task order explicitly (topic 550336); Barbadillo (2nd) called probing "totally against the spirit" (same thread) and used extra submissions only as compute. The host cut daily submissions 5 to 3 for this reason (discussion/515322).
- Contradiction 2 (do LLMs help a classical ensemble?): William Wu (4th) won money with zero LLMs and concluded TTT is the future (topic 550414); PoohAI (5th) thought LLMs could be added but "fine-tuning is needed" (topic 550336); Barbadillo (2nd) gained from adding the DSL's first attempt to his LLM's (bank).
- Contradiction 3 (pretrained vs from-scratch transformer): ARChitects and Barbadillo fine-tune pretrained LLMs; Nikola Hu (13th) trained a 162M model from scratch with 2D positional encoding and reached 30 in one month, asking openly whether that can scale (topic 546302).

## Aggregated "what did not work"

- Full attention among input tokens ("mask hack") in a from-scratch transformer: no gain (13th, topic 546302).
- Progressive output head: unclear, bugs (13th).
- High public-eval accuracy without TTT transferred poorly: 30 to 40% public eval but ~4 hidden (13th).
- Picking the right attempt among candidates is hard; removing wrong ones via filters is easier (5th).
- Recolouring preprocessing can help some tasks and "ruins some tasks" (21st, topic 550209).
- Graph-based and neural-cellular-automata approaches: under 1% (21st).
- Verifier/selector models, code generation, BARC 400k data, smaller/bigger models (2nd; detail in bank).
- Using several accounts to get around the 3-submissions/day cap: suspected cause of a top team's disappearance (see shake-up).

## Shake-up note

No public/private shake-up by design: every top-20 team has the same public and private rank (API). What moved the board was removals at the end:
- The team "torino", in the top 3 the day before the deadline, vanished; its account shows as no longer active. CPMP suspected multiple accounts used to bypass the 3/day submission limit (topic 545733, https://www.kaggle.com/competitions/arc-prize-2024/discussion/545733). Unconfirmed by the host.
- MindsAI (55.5) was removed from the prize ranking for not open-sourcing (bank; topic 550442).
- William Wu moved 7th to 4th "after some teams withdrew"; PoohAI's arcprize.org notebook is titled "8th place solution", so they moved about 8th to 5th.
- Rank 6 is now a deleted account.

## UNVERIFIED

- Methods of every team at ranks 6 to 12 and 14 to 20 (no writeups found). The guess that the 29-point cluster are 26-notebook ensembles.
- Identity of all withdrawn/removed teams above William Wu besides MindsAI and "torino"; the cause of torino's removal.
- How much of Nikola Hu's 30/31 came from his transformer vs the ensembled public notebook.
- David Hartmann's absence from the Kaggle team roster (paper lists him; API does not).
- Content of PoohAI's Google Drive paper (not read).
