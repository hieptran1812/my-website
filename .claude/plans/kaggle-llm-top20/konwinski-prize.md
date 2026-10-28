# Konwinski Prize (K Prize, 2024–25): top-20 addendum

- Slug `konwinski-prize`, competition id 84795, forumId 5846300.
- Ranks 1 to 3 (plus one-line notes on 4 and 5) are covered in depth in `.claude/plans/kaggle-llm-writeup-bank-part2.md`, H2 "## 9. Konwinski Prize (K Prize, 2024–25) — SWE-bench with a contamination-free future test set". This file does not repeat that detail.
- Leaderboard pulled from `competitions.LeaderboardService/GetLeaderboard` on 2026-10-28. Raw JSON and topic dumps: `/private/tmp/claude-501/kaggle-top20/kprize/`.
- Forum sweep: all 182 topics paged, filtered by title and by author (every top-22 member's username). Writeup links from the API's `solutionWriteUpUrl`.

## 1. Top-20 table

**Correction to the bank.** The bank says 0.0165x "corresponds to ~2 correct and 0 wrong". That is wrong. The metric is `(a − b − c/10,000) / 120` (private n = 120), and the displayed score is **truncated** to 6 decimals. Under that rule every top-20 score decodes to exactly one (correct, wrong, skipped) triple, and the decoding reproduces both self-reported triples (1st 9/2/109 – confirmed by Tong Hui Kang in comment id3253023; 4th 4/2/114 – stated in the 4th-place writeup). No triple matches under rounding. The decoded column below is derived, not stated by the teams except where marked.

**Public rank:** the API's public column is 0.000000 for every team after the forecasting-phase rescore, so the API's public "rank" is an ordering of zeros. It is reported as unavailable, except where a team states its own public placing in its writeup title or text (marked "self-reported").

| Rank | Team | Members (username) | Private | Decoded (a, b, c) | Public rank | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|---|
| 1 | Eduardo Rocha de Andrade | arc144 | 0.058242 | 9, 2, 109 (confirmed) | 1st self-reported context: Giba calls him first "in public and private" (id3253309); API unavailable | https://www.kaggle.com/competitions/konwinski-prize/writeups/eduardo-rocha-de-andrade-1st-place-solution-write- (topic 568884) | Agentless 1.5 fork, Qwen2.5-Coder-32B, reproduction tests + F2P/P2P gate |
| 2 | Camaro | bamps53 | 0.016576 | 7, 5, 108 | 2nd (self-reported, title "Public 2nd Place") | https://www.kaggle.com/competitions/konwinski-prize/writeups/camaro-public-2nd-place-solution (topic 568888) | Agentless-style pipeline, Qwen2.5-Coder-32B, JSON-structured edits, test generation |
| 3 | akami | akiyama (mtfall), sakami, kami (kami634) | 0.016575 | 6, 4, 110 | unavailable | https://www.kaggle.com/competitions/konwinski-prize/writeups/3rd-place-solution (topic 597207) | Select-Patch-Verify + LoRA difficulty classifier on DeepSeek-R1-Distill-Llama-70B-AWQ |
| 4 | Genxxsky | genxxsky | 0.016571 | 4, 2, 114 (confirmed) | unavailable (its public score −0.000097 = 1, 1, 69 per writeup) | https://www.kaggle.com/competitions/konwinski-prize/writeups/4th-place-solution (topic 596181) | Select-Patch-Verify + tree-sitter context + generated pytest before/after tests |
| 5 | Kenj | manhnguyen315 | 0.008239 | 4, 3, 113 | unavailable | https://www.kaggle.com/competitions/konwinski-prize/writeups/5th-place-solution (topic 596375) | Select-Patch-Verify + regex traceback locator, SEARCH/REPLACE via git, skip issues without traceback |
| 6 | 1M | Quan Nguyen (quannguyn12), tishiu | 0.008237 | 3, 2, 115 | unavailable | no writeup found | unknown |
| 7 | snoopyyyyy | snoopyyyyy | 0.008237 | 3, 2, 115 | unavailable | no writeup found | unknown |
| 8 | first lucky gold! | yuanzhe zhou (yuanzhezhou) | 0.008235 | 2, 1, 117 | unavailable | https://www.kaggle.com/competitions/konwinski-prize/writeups/first-lucky-gold-8th-place-solution (topic 590920) | Select-Patch-Verify; picked the model weights with the best correct:wrong ratio |
| 9 | bianfu | bianfu | 0.008234 | 1, 0, 119 | unavailable | no writeup found | unknown |
| 10 | Đàm Văn Tài | damvantai | 0.008234 | 1, 0, 119 | unavailable | no writeup found | unknown |
| 11 | Myles and Trevor | Trevor Dunlap (tdunlap607), Myles Dunlap (dunlap0924) | −0.000093 | 4, 4, 112 | unavailable | no writeup found | unknown |
| 12 | nico | lllzzzyyylearning (luzhenyi), TianOrange (tianorange) | −0.000096 | 2, 2, 116 | unavailable | no writeup found | unknown |
| 13 | Kenleung Leung | kenleungleung | −0.000096 | 2, 2, 116 | unavailable | no writeup found | unknown |
| 14 | Marcelo Piovan | piovan | −0.000096 | 2, 2, 116 | unavailable | no writeup found | unknown |
| 15 | GraphNINJAs | ISAKA Tsuyoshi (isakatsuyoshi), yawanago (kazukishida), NOZOMI YAMAMOTO (nozomiyamamoto) | −0.000096 | 2, 2, 116 | 4th (self-reported; public 0.056243 = 4, 0) | https://www.kaggle.com/competitions/konwinski-prize/writeups/graphninjas-public-4th-place-private-15th-place-so (topic 568799) | Select-Patch-Verify v94, R1-Distill-Qwen-32B-AWQ, edit-distance patch selection, skip gates tuned on public LB, odd challenge count |
| 16 | PRASHANT SHUKLA91 | prashantshukla91 | −0.000098 | 1, 1, 118 | unavailable | no writeup found | unknown |
| 17 | BIT Research Alliance | atom1231, Kevin L (luozihjia1221), Ren-Di Wu (whats2000), JimmyChu169 | −0.000098 | 1, 1, 118 | unavailable | no writeup found | unknown |
| 18 | Mohammad Albaddawi | m7mdalbaddawi | −0.000098 | 1, 1, 118 | unavailable | no writeup found | unknown |
| 19 | yellowleaf | wcqyfly | −0.000098 | 1, 1, 118 | unavailable | no writeup found | unknown |
| 20 | Chew Kok Wah | chewkokwahibrainai | −0.008422 | 6, 7, 107 | unavailable | no writeup found | unknown |

Ties exactly as the API gives them (distinct ranks, equal displayed scores): 6 = 7 (0.008237), 9 = 10 (0.008234), 12 = 13 = 14 = 15 (−0.000096), 16 = 17 = 18 = 19 (−0.000098). Ranks 21 (trip, soso) and 22 (Junhua Yang) sit just below at −0.008424 / −0.008427.

**Writeups found: 7 of 20** (ranks 1, 2, 3, 4, 5, 8, 15).

What the decoding shows: the top 3 attempted 11 to 12 issues each and got more right than wrong; ranks 2 and 3 were **7–5 and 6–4**, not cautious 2–0 records. Rank 20 got 6 right but 7 wrong. Only ranks 1 to 10 have correct > wrong.

## 2. Per-team synthesis

### Ranks 1 to 3: facts the bank is missing
- **Eduardo (1st).** On robustness to new data (reply to Giba, comment id3253328): after his best public score he spent the last two weeks making the result consistent across seeds. What he thinks helped: (1) requiring **both F2P and P2P tests**, being strict about accepting a patch (precision); (2) **greedy or very low temperature at almost every step**, alongside a few higher-temperature samples for diversity; (3) **varying context per sample** (full imports + classes, imports only, no context) because the best amount of context varied a lot per issue; (4) a faster pipeline buying more samples per step. He also tried running two LLMs (one per GPU) and ensembling, but "couldn't make it work": every reasoning LLM was slow and did not improve on Qwen2.5-Coder-32B.
- **Camaro (2nd).** Decoded private 7 correct, 5 wrong. In comments (id3153931) he says he never tried an agentic pipeline (OpenHands) because it is hard to control and evaluate with a small model, and that SWE-bench (especially Verified) is easier and cleaner than the K Prize public set, which had higher skip and failure rates and "more noisy instances". Eduardo and Camaro agree their solutions were "quite similar" (id3153250).
- **akami (3rd).** Decoded private 6 correct, 4 wrong. No comments on the post.

### Rank 4 – Genxxsky (0.016571; 4 correct, 2 wrong, 114 skipped), topic 596181
Code: https://www.kaggle.com/code/genxxsky/4th-place-solution-select-patch-verify-test . Built on Tong Hui Kang's Select-Patch-Verify starter and his LB-decoder thread (discussion/557148). Model not named in the writeup (the starter notebook's model is assumed – UNVERIFIED).

Two final strategies, with private/public decoded by the author:
- **Select-Patch-Verify-Test (winner):** private 0.016571 (4, 2, 114); public −0.000097 (1, 1, 69). Fewer patch attempts and less thinking time; for each candidate patch the model writes a **pytest file with two functions**, `test_before_patch` (should pass before the fix, fail after) and `test_after_patch` (the reverse), and runs them.
- **Select-Patch-Verify-Choose:** private −0.008429 (2, 3, 115); public −0.000094 (2, 2, 67). More tokens and patches, then picks the best by a multi-dimensional evaluation; better on complex issues but sometimes very slow.

Pipeline: Select (code search with **tree-sitter**, returning the whole enclosing function/class around each matched line, with line numbers) → Patch (diff) → Verify (LLM self-check) → Test (generated pytest) → Choose. Efficiency: parallel processing, token compression, caching, early stopping; quick confidence estimates to prioritise patches.

Did not help or random effect: rule-based diff repairs (random impact); multi-method code lookup, multi-step analysis, cross-attempt code sharing; varying prompts or parameters across attempts; longer patch generation; AST-based checks; reusing existing pytest functions. Self-criticism: a late refactor applied improvements inconsistently across the two strategies, and many changes had no ablation.

Distinguishing idea: a generated before/after test pair as the acceptance gate, i.e. Agentless-style reproduction testing grafted onto the Select-Patch-Verify notebook. Note the public/private reversal: the winning strategy was slightly negative on public.

### Rank 5 – Kenj (0.008239; decoded 4, 3, 113), topic 596375
Notebook: https://www.kaggle.com/code/manhnguyen315/5th-place-solution . Base: Select-Patch-Verify. Joined late.
- **Skip gate:** skip every issue without an error traceback, because only with a traceback could the right edit location be found in his tests. From public-LB probing he estimated 6 < n < 11 public issues contain tracebacks.
- **Localisation:** replaced the LLM keyword search (`get_selection_query`, which invented phrases or picked too-generic ones) with a **regex traceback parser** (`get_selection_query_x`) that maps each traceback frame's path to the repo copy and extracts the sub-traceback line; supports Python 3.10 to 3.12 formats; falls back to the LLM search if nothing is found.
- **Patch:** model **DeepSeek-R1-Distill-Qwen-14B**, 8 queries per issue, Agentless SEARCH/REPLACE prompt ("git conflict markers"); edits applied to a fake git repo, committed, then `git diff` produces a valid patch – the same idea as 1st place.
- **Selection:** first valid patch; no tests run, no evaluation, no environment setup. His second, fuller submission broke at random, forcing the simpler one.
- **Did not work:** checking/fixing diff line counts; DeepSeek-R1-Distill-Qwen-32B (no gain over 14B in his tests).
- Distinguishing idea: deterministic localisation from tracebacks plus a hard "no traceback, no attempt" rule.

### Rank 8 – first lucky gold! (yuanzhe zhou; decoded 2, 1, 117), topic 590920
Short writeup. Base: Select-Patch-Verify. Joined late and chose to focus on filtering out high-risk issues ("statistical arbitrage"). Keyword filtering (e.g. dropping patches that return "error") helped only marginally. The breakthrough: across different **model weight configurations**, one set failed on many cases but had a notably high **correct-to-wrong ratio** on the ones it did answer; he built the final submission around that model's predictions. Both selected submissions using this logic reached gold. Model names not given.

### Rank 15 – GraphNINJAs (private 2, 2, 116; public 4th at 0.056243 = 4, 0), topic 568799
Team: ISAKA Tsuyoshi, kazukishida, nozomiyamamoto.
- **Validation:** 6 training instances, ~6,400 external instances (Nebius, discussion/552605) with a different distribution, 71 public. Strategy stated as "ignore CV, care about public LB, trust methodology".
- **Base and model:** Select-Patch-Verify **version 94** (7 correct on public); **DeepSeek-R1-Distill-Qwen-32B-AWQ** (the non-AWQ model did worse, oddly). QwQ-32B (AWQ and full) disappointed. Temperatures 0.6 / 0.6 / 0.0 for read / patch / verify.
- **Verifier prompt** with a third label `None` for "unsure" – no significant difference, not used in the final.
- **Edit-distance selection:** drop patches longer than 1,000 characters, then pick the candidate with the smallest summed edit distance to all others (the "most average" patch). Turned an early −1 submission into a valid score.
- **Skip mechanism, modelled with a hypergeometric simulation:** odd numbers of attempted issues ("challenges") beat adjacent even numbers because even counts allow ties like 2–2. They planned 3 challenges, then switched to **5** after Select-Patch-Verify v113 occasionally scored (3, 0) and was being copied. Gates tuned on public: problem statement length kept only within **1,802 to 3,400 characters** (above 3,400 never correct; under 1,200 only one correct), and **file count ≤ 388** (500 gave (5, 4), 388 gave (4, 2), 403 gave (5, 4)). A plan-B rule extends to an odd challenge count if the target is not reached by turn 100.
- **Time:** per-step limits 7 / 15 / 19 min for read / patch / verify; global limits 8 h public, 23 h private.
- **Engineering:** GitHub with code review, 12 modules, 98 tests, 80% coverage; only one critical bug.
- **Outcome:** public 4th, private 15th – the gates tuned on 71 public issues did not carry over to 120 curated new ones; they note that the private set "should be curated", which could invalidate the length-based skipping.

### Ranks 6, 7, 9 to 14, 16 to 20
No writeup and no forum posts by any member (author sweep). No external writeups found. Decoded records are in the table.

## 3. Technique frequency matrix

Columns with information: 1, 2, 3, 4, 5, 8, 15. All others `?`, omitted.

| Technique | 1 | 2 | 3 | 4 | 5 | 8 | 15 |
|---|---|---|---|---|---|---|---|
| Base pipeline | Agentless 1.5 | Agentless-style (own) | Select-Patch-Verify | Select-Patch-Verify | Select-Patch-Verify | Select-Patch-Verify | Select-Patch-Verify v94 |
| Coder model (Qwen2.5-Coder-32B) | Y | Y | N | ? | N | ? | N |
| Reasoning-distill model (DeepSeek-R1-Distill) | N (tried, worse) | N (worse) | Y (Llama-70B-AWQ) | ? | Y (Qwen-14B) | ? | Y (Qwen-32B-AWQ) |
| Any fine-tuning | N | N | Y (LoRA r32 classifier) | N | N | N | N |
| Generated reproduction / F2P tests | Y | Y | N | Y (before/after pytest) | N | N | N |
| Runs repo tests (P2P regression) | Y | Y | N | ? | N | N | N |
| LLM verifier yes/no | N | Y | Y | Y | N | ? | Y |
| SEARCH/REPLACE or old/new edits (not raw diff) | Y | Y (JSON) | N | N | Y | ? | N |
| Structural code context (skeletons, tree-sitter) | Y | Y | N (grep ±12 lines) | Y (tree-sitter) | N (traceback regex) | ? | N |
| Explicit difficulty / skip classifier | Y (no reproducing test → skip) | Y (LLM solvability) | Y (LoRA easy-prob) | N | Y (no traceback → skip) | Y (model-choice) | Y (length, file-count, challenge count) |
| Multiple candidate patches | Y (8) | Y (5 retries) | Y (6 per search) | Y | Y (8) | ? | Y |
| Patch selection beyond "first valid" | Y (tests) | Y (tests + verify) | N | Y (tests) | N | N | Y (edit distance) |
| Tuned skip gates on public LB | N | N | N | N | Y (traceback count) | Y | Y |

Counted summary (M = 7 writeups): 5 of 7 built on Tong Hui Kang's Select-Patch-Verify; 2 of 7 (the top 2) built Agentless-style pipelines with Qwen2.5-Coder-32B; 3 of 7 generated their own tests, and those are ranks 1, 2 and 4; 3 of 7 used SEARCH/REPLACE-type edit formats; 7 of 7 had an explicit skip rule; 1 of 7 fine-tuned anything (a classifier, not the patcher); 0 of 7 used agent loops in the final.

## 4. Consensus vs divergence

- **Table stakes:** skip almost everything (every top-20 team skipped ≥ 107 of 120); an explicit abstention rule; multiple candidate patches; open 14B to 70B models on 4x L4.
- **What separated the top 2 (and 4) from the rest:** **executable verification**. Ranks 1, 2 and 4 generated reproduction tests and ran them; ranks 3, 5, 8, 15 relied on LLM yes/no checks, first-valid-patch or heuristics. Ranks 1 and 2 additionally used a coder model rather than a reasoning distill. Rank 1 had 9 correct; nobody else had more than 7.
- **Contradictory findings:**
  - Reasoning models: Eduardo and Camaro found R1-style reasoning models slower and no better than Qwen2.5-Coder-32B (topics 568884, 568888); akami (3rd), Kenj (5th) and GraphNINJAs (15th) used R1-Distill models; Kenj found the 32B distill no better than the 14B.
  - Bigger model: Camaro says 14B → 32B fixed most context and syntax problems; Kenj says 14B → 32B did not help (different model families).
  - Public LB as a guide: GraphNINJAs tuned skip gates on the 71 public issues and fell from 4th to 15th; Eduardo instead tuned for seed-consistency and kept 1st; 4th place's winning strategy was slightly negative on public.
  - Context: Camaro found giving even ground-truth files helped little; Eduardo found context (imports, class skeletons) essential for small models writing tests.

## 5. Aggregated "what did not work"
- Reasoning models (QwQ-32B, DeepSeek-R1, long "but wait" traces) – Eduardo, Camaro (bank), GraphNINJAs (QwQ).
- Two-model ensemble across GPUs – Eduardo (comment id3253328).
- Agent mode / OpenHands – Eduardo (too late), Camaro (not attempted: hard to control).
- More file candidates, second pass, embeddings for localisation – Camaro (bank).
- A too-strict verifier rejects everything – Camaro (bank).
- Asking small models for raw unified diffs – Eduardo, Kenj (SEARCH/REPLACE instead).
- Rule-based diff repairs; multi-method code lookup; cross-attempt code sharing; per-attempt prompt variation; AST checks; reusing repo pytest functions – Genxxsky.
- Diff line-count checks; R1-Distill-Qwen-32B over 14B – Kenj.
- Keyword filters on patches – yuanzhe zhou (marginal).
- `None` ("unsure") verifier label – GraphNINJAs (no significant difference).
- Non-AWQ R1-Distill-Qwen-32B – GraphNINJAs (worse than AWQ).
- Exploratory many-patch "Choose" strategy – Genxxsky (2–3 private vs 4–2 for the lean strategy).

## 6. Shake-up note
Total, and mostly explained by sample size and the symmetric penalty. One issue moves a team many places. The clearest documented drop is GraphNINJAs, public 4th (4, 0) to private 15th (2, 2), after tuning length and file-count skip gates on 71 public issues. Genxxsky's selected strategy was −0.000097 public (1, 1) yet 4th private (4, 2). Eduardo (first on public per Giba's comment) held first. Camaro's same code once scored 0.0703 and −0.0142 on public (bank, discussion/568626). The API's public column cannot be used: it is zero for all teams.

## 7. UNVERIFIED
- Decoded (correct, wrong, skipped) for ranks 2, 3 and 5 to 20 are derived from the metric assuming n = 120 and truncation to 6 decimals. The assumption reproduces both self-reported triples and gives unique solutions, but teams 2, 3, 5 to 20 did not confirm them. This supersedes the bank's "~2 correct and 0 wrong" reading.
- Eduardo's public rank 1: from Giba's comment, not from the API.
- Model used by Genxxsky (4th) and yuanzhe zhou (8th).
- Approaches of ranks 6, 7, 9 to 14, 16 to 20.
