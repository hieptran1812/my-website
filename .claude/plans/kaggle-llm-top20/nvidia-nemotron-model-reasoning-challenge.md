# NVIDIA Nemotron Model Reasoning Challenge (2026): top-20 addendum

- Slug `nvidia-nemotron-model-reasoning-challenge`, competition id 129716, forumId 10085877.
- Ranks 1 to 3 (plus a short rank-4 note and Tong Hui Kang's Open Progress Prize baseline) are covered in depth in `.claude/plans/kaggle-llm-writeup-bank-part3.md`, H2 "## NVIDIA Nemotron Model Reasoning Challenge (discovered)". Not repeated here.
- Leaderboard pulled from `competitions.LeaderboardService/GetLeaderboard` on 2026-10-28. Raw JSON and topic dumps: `/private/tmp/claude-501/kaggle-top20/nemotron/`.
- Forum sweep: all 302 topics paged, filtered by title and by author (every top-22 member's username). Writeup links from the API's `solutionWriteUpUrl`.
- Shared vocabulary: "OPP" = Tong Hui Kang's Open Progress Prize writeup and code (discussion/689915, https://github.com/tonghuikang/nemotron). Every team below with a writeup builds on it. Categories: bit manipulation, cipher, cryptarithm (deduce/guess), equation numeric (deduce/guess), gravity, numeral (Roman), unit conversion; train.csv has 9,500 rows (bit 1,602; cryptarithm 659 + 164; equation 596 + 136).

## 1. Top-20 table

Public scores and ranks here are real (unlike AIMO/K Prize). Shake-up = public rank minus private rank.

| Rank | Team | Members (username) | Private | Public (rank) | Shake-up | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|---|
| 1 | NullSira | GoodMeatDay (goodmeatday), re (refv00), reopon | 0.920 | 0.912 (1) | 0 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution (topic 709231) | 2-stage SFT on 220k + 140k solver traces; cryptarithm signature catalog + DFS; hex bit traces |
| 2 | vli | vli (victorlifire) | 0.908 | 0.884 (5) | +3 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution (topic 711703) | Reverse-engineered generators, 57,600 synthetic examples, Megatron-Bridge LoRA |
| 3 | YS-L | YS-L (liauys) | 0.900 | 0.872 (42) | +39 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution (topic 709136) | Compact traces, CSP cryptarithm, two-stage memorise-then-solve LoRA |
| 4 | /make-no-mistakes | Dipam Chakraborty (dipamc77) | 0.900 | 0.892 (3) | −1 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution (topic 712419) | "Trainable traces" audited by code; bit manipulation 99.4% via 5 templates + anchor tables |
| 5 | Domdolus Tolus | Darren Amadeus Martin (darrenamadeusmartin), Tri Ardian Rinalda (rinnnntt), Alwin Theodoric (alwintheodoric), Muhammad Yassir Ramadhan (yassirramadhan), Billy Anggara (billyanggara) | 0.896 | 0.888 (4) | −1 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/domdolus-tolus-solution (topic 708538, "Jack of Trades, Loser of Cryptarithm") | Infinite synthetic generator (no train.csv in training), 15 SFT iterations with LoRA soups |
| 6 | Alehandreus & Yurnero | Alehandreus (alehandreus), Yurnero (samson8) | 0.896 | 0.900 (2) | −4 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-public-6th-private-place-solution (topic 709120) | Full train.csv as validation, per-domain LB probing with DUMMY outputs, DFS cryptarithm, 0.25B-token synthetic set |
| 7 | Solved Bit Manipulation | adypd (adityap97), Prateek (prateekagnihotri), Shubham Jain (shubham068), Sanchit Jain (sanchit1609), Prabhat Agnihotri (agnihotriprabhat) | 0.888 | 0.864 (531) | +524 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/7th-place-solution-solving-bitmanipulation-via-st (topic 712395) | Bit manipulation via string matching + backtracking, single-bit tokens, dynamic loss masking; no cryptarithm training |
| 8 | Researcher 7919 | Researcher 7919 (researcher7919) | 0.888 | 0.876 (9) | +1 | no writeup found | unknown |
| 9 | Fate | Chi-Han Tsai (chihantsai) | 0.884 | 0.880 (7) | −2 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot (topic 711570) | No synthetic expansion; rewrote OPP CoT targets to be human-auditable; 1 epoch LoRA |
| 10 | kansai-kanto-kaggler | eikichi (ekichi), neilus, Rick (rikitomo0526) | 0.880 | 0.872 (24) | +14 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/10th-place-solution (topic 708535) | Rotate/shift key-table bit reasoner (98.75% oracle), corpus-frequency equation priors, alpha/r = 4 |
| 11 | Hello World | WCao24 (wcao24) | 0.872 | 0.868 (82) | +71 | no writeup found | unknown |
| 12 | Team | Chris Akiki (cakiki), conor (conormacamhlaoibh), kenyeung.tech (kenyeungtech), Bilal (gfvdcujuzg35) | 0.872 | 0.876 (10) | −2 | no writeup found | unknown |
| 13 | Haraguchi-T | Haraguchi-T (haraguchit) | 0.872 | 0.868 (84) | +71 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/13th-place-solution (topic 709646) | OPP reasoners + heuristics (89.4% oracle), 3-stage Unsloth SFT on full data |
| 14 | TaiDoDuc | TaiDoDuc (taidoduc) | 0.872 | 0.872 (62) | +48 | no writeup found | unknown |
| 15 | MAJ0RT0M | MAJ0RT0M (maj0rt0m) | 0.872 | 0.876 (16) | +1 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/wip-lb15-extension-to-huikangs-progress-prize (topic 711727) | OPP + arity-3 truth-table bit solver + memorised-multiplication-table cryptarithm via an execution-trace mini-language |
| 16 | toxu | toxu (tonyxu) | 0.868 | 0.868 (71) | +55 | no writeup found | unknown |
| 17 | Frank Yang | Frank Yang (ffy208) | 0.868 | 0.856 (1736) | +1719 | no writeup found | unknown |
| 18 | Y \| M \| F | yanqiangmiffy (quincyqiang), MOONMOON (kevin1742064161), ForcewithMe (forcewithme) | 0.868 | 0.876 (8) | −10 | https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/18th-solution-from-deterministic-solvers-to-learn (topic 715330) | OPP reasoners extended, GT-assisted tie-breaks, SVD LoRA fusion; no local validation |
| 19 | Ryuto Yoda | Ryuto Yoda (ryutoyoda) | 0.868 | 0.856 (1781) | +1762 | no writeup found | unknown |
| 20 | T T | Julien Genzling (juliengenzling), Gauthier Avite (gauthieravite) | 0.868 | 0.864 (322) | +302 | no writeup found | unknown |

Ties exactly as the API gives them (distinct ranks, equal scores): 3 = 4 (0.900); 5 = 6 (0.896); 7 = 8 (0.888); 11 to 15 (0.872); 16 to 20 at 0.868, and the 0.868 tie continues past the cut (21 EQV UK, 22 xinyue wang, more below). With ~250 private problems, 0.004 = one problem (6th place's own arithmetic: 0.01 ≈ 3 problems on 250).

**Writeups found: 12 of 20** (ranks 1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 15, 18; rank 7 also has an arXiv preprint, 2606.23672).

Note on public scores: the API's public score for 10th (0.872) differs from the 0.860 that team reports for its selected final run; the API public column seems to be the team's best public submission, not necessarily the selected one (UNVERIFIED).

## 2. Per-team synthesis

### Ranks 1 to 3: facts the bank is missing
- **NullSira (1st).** The idea of a trace that follows the solver but ends in the ground-truth `\boxed{}` came from an accident: in a bit problem where the local solver disagreed with ground truth, they built a sample with the solver trace and the true answer swapped in. "The model learned the format. In fact, we saw cases where the solver would be wrong, but the trained model changed only the final answer and became correct" (topic 709231). MAJ0RT0M (15th) independently saw the model beat its own reference solver by ~1% on bit manipulation (topic 711727).
- 2nd and 3rd: nothing material beyond the bank found in this pass.

### Rank 4 – /make-no-mistakes (Dipam Chakraborty; 0.900, public 3rd), topic 712419
- **Principle – "trainable traces":** (1) no hidden compute: every token derivable from visible context plus a small rule set; (2) locality: each line computable from ~10 to 20 lines above it or a verbatim copy of a named line; (3) causal consistency: trace shape depends only on what is already written (no special block only for the winner); (4) reference integrity: cite only names/values already in the trace; (5) few, uniform rules (a rule that fires in a handful of traces cannot be learned). Rules 2 and 4 were enforced by scripts (reference-integrity audit; locality audit measuring token distance to the farthest source line).
- **Metric:** first-token divergence against the gold trace measured at training time; a generation viewer diffing model output against gold; Claude (Opus 4.8) wrote essentially all code and was pointed at the viewer to classify failures and propose format fixes ("train, diff, classify, reformat, retrain", ~1 h per cycle).
- **Bit manipulation:** everything not solvable by 1 or 2 ops fits **5 templates** over three shifted leaves (Majority, Choice, M5, C3, C4). Brute force does not fit in 7,500 tokens, so the trace reads shift amounts off two **anchor tables** at bit 7 (no left-shift alive) and bit 0 (no right-shift alive), classifying each output column's four quadrants into a 2-input function, joins the two on the shared rotate amount, then fully verifies each joint candidate. Gold reasoner: 1,593/1,602 (99.4%); mean trace 4,148 tokens (p95 ~4,600, max 6,739). His traces hit >95% accuracy with ~2,500 examples where OPP traces reached ~82%.
- **Other categories:** equation operations tried in measured corpus frequency (signed subtraction before absolute difference); guess ~39% solve rate; cryptarithm symbol-splitting + operator-glyph prior.
- **Training:** OPP code; LoRA r 32 / α 32 / dropout 0; q/k/v/o, up/down, Mamba in/out proj, lm_head; **MoE experts untied**; batch 16 with micro-batch 4 ("lower batch size converged faster"); LR 2e-4 cosine with warmup and cooldown; Cut Cross-Entropy. RTX PRO 6000 via Colab Pro G4 + Kaggle quota, no Tinker; ~$120 total. Data: 9,500 originals + 15,000 synthetic traces.
- **Distinguishing idea:** making trace learnability a testable property and letting an LLM coding agent iterate formats against those tests.

### Rank 5 – Domdolus Tolus (0.896, public 4th), topic 708538
Self-described near-zero fine-tuning experience; most time spent solving puzzles by hand and designing CoT.
- **Data:** no train.csv rows in training at all; an "infinite" synthetic generator per category matched to train.csv statistics (e.g. gravity g 4.9 to 19.6 with noise; conversion multipliers 0.05 to 20; cipher limited to the ~80-word vocabulary and 4 sentence templates; Roman 1 to 100); train.csv used only as validation. They attribute OPP's CV/LB instability to training and validating on the same train.csv.
- **CoT:** OPP traces for gravity/conversion/cipher; Roman via a written 1-to-100 lookup table. Bit manipulation: analysed train.csv into 4 stages (unary, pairwise, simplified ternary, true ternary – MAJORITY, CHOICE, XOR-NOT_OR), ~140 continuations and ~7,500 valid 8-bit sequences. A first trace using a 5-bits-per-letter encoding failed badly (the model hallucinated rules or output `11111111`): "encoding is breaking the model". The working trace returned to raw bits, keeps only 7 examples (~94% still pinned), drops ternary search (extrapolate the middle from prefix/suffix pairs) and filters candidate groups by shift length. Equation: 2 "worlds" (normal/reverse) × 4 families, 13 rules; guess order tuned to ~40%. Cryptarithm: their own guess-style CoT "failed miserably", fell back to OPP.
- **Training:** OPP trainer on Kaggle via the Kaggle CLI (plus Colab Pro to access Kaggle compute, ~$100); r 32, α 32, LR 1e-4 to 2e-4, grad clip 1.0, linear schedule; ~**2 billion tokens** in total. Phase 1: 6 sequential SFT iterations at high LR. Phase 2: **LoRA averaging** – train level k+1 on top of the previous soup, then average with it. Phase 3: train 3 models from the same soup on different problem sets and average. Data per iteration 13k to 78k samples, bit-manipulation-heavy.
- **CV vs LB:** validation 79.0 → 90.4 over 15 iterations; public rose with it through phase 1 then plateaued at ~88; private kept rising to 90.0. Final pick = the most stable adapter (88.8 public, 89.6 private in 3 of 5 resubmissions); their highest-CV, highest-private (90.0) adapter was not selected because it was unstable.
- **Did not work:** short CoT without logical steps (even 250k examples); pseudo-GRPO via iterated DPO (5 rounds, could not learn easy bit problems); real GRPO (air-gapped notebook, C++ compile errors, OOM); Unsloth with 4-bit quantisation and packing (~100k examples, no learning); DoRA (lower loss but vLLM rejects it; conversion lossy). ORPO/KTO/DPOP planned, not run.

### Rank 6 – Alehandreus & Yurnero (0.896, public 2nd), topic 709120
- **Training:** Unsloth on Kaggle/G4 for experiments, Tinker late; batch 32, LR 3.5e-4, r 32, α 32, 1 epoch.
- **Validation:** all 9,500 train rows used as validation (10x the usual 950-row holdout), which forced an all-synthetic training set. **Per-domain LB probing:** train the target domain normally and every other domain to output `I will ignore this problem \boxed{DUMMY}`, giving a pure per-domain public signal; assumed 500 test problems split 250/250, so 0.01 ≈ 3 problems.
- **Bit manipulation:** a small template set covers all of train (e.g. rotr 89, shl XOR shr 368, rotr AND NOT shl 86 problems); iterate shift values per template with monotonic skips; **all arithmetic in hex**, including ≤ comparisons, with a hex multiplication cheatsheet printed first. Reasoner 1,508/1,602; trained LLM 1,480.
- **Cryptarithm:** DFS over symbol mappings, equation by equation, with backtracking; tries reversed numbers if forward fails (~40% of solves are reversed); operation deduction from result length and sign and the one-op-per-family rule; leading-zero and leading-one pruning; equations with fewest unknowns first. Best version 240/654 deduce problems after ~100M tokens; a lighter 170-problem version was used in the final. Equation numeric: same solver, 550/594 guess problems.
- **Synthetic data:** distributions fitted from train.csv; targeted cipher subsets fixed hallucinations like `s-t-u-d-e-n-t -> sent`; cryptarithm generator rerolls digit mappings until the reasoner solves within the token limit.
- **Final decision:** adding data for any domain barely hurt others except bit manipulation, so 66 to 69% of tokens were bit problems. A ~0.25B-token set (best CV) vs a ~0.5B-token set that doubled cryptarithm (150 → 241 solved) but dropped bit CV; they submitted two seeds of the 0.25B adapter, which they say cost them 2nd or 3rd place.

### Rank 7 – Solved Bit Manipulation (0.888, public 531st), topic 712395 + arXiv 2606.23672
Team: Prateek, Sanchit, Prabhat Agnihotri, Shubham Jain, adypd. Joined in the last 10 days; **no cryptarithm training at all**.
- Bit manipulation reframed as feature selection: string similarity (minimal bit flips) between outputs isolates primitive transformations ("bases") and their truth tables with no arithmetic; a structured search tests candidate bases, detects collisions across the examples and backtracks deterministically.
- **Single-bit tokenisation:** forced the tokenizer to emit one token per bit (avoiding chunks like `1010|00|11`).
- **Dynamic loss masking** to simulate oracle feedback, so the model learns to hypothesise, self-evaluate and backtrack. Claimed >96% bit validation accuracy ("highest in this category by any participating team"; compare 4th's 99.4% reasoner).
- Notebooks for every category's solver/CoT and synthetic generation are linked in the post. Hyperparameters not stated in the post or the arXiv abstract.
- The largest public→private jump in the top 10 (531 → 7).

### Rank 8 – Researcher 7919
No writeup, no forum posts.

### Rank 9 – Fate (Chi-Han Tsai, solo; 0.884), topic 711570
- **No synthetic expansion.** Rewrote the OPP traces so targets are correct, consistent and readable; verified-correct targets: bit 1,364 → 1,447 /1,602; equation 561 → 633 /732; cryptarithm 65 → 73 /823.
- Fixes: gravity/unit multiplication expanded from the highest digit down with zero positions kept explicit and fixed decimal places (near-perfect after); compact cipher pipeline (map, partial decode, dictionary match only unresolved words grouped by length, stop at 2 mismatches); bit notation `Z`/`U` for constants, Identity as its own family, dedupe symmetric pairs, edge anchoring (left runs prefer bit 7, right runs bit 0), alternate `ORNOT` fill and a composite `OR(XORNOT, I)` fill, majority-family fallback instead of default `U`; equation with explicit normal/reverse mode and family-exclusion fallback.
- **Training:** one fold held out; numeral cut to ~400 examples; bit ×5, cipher ×2, equation ×2 oversampling (final mix bit 6,405, cipher 2,522, ...); LoRA r 32, α 32, dropout 0, q/k/v/o, in/out proj, up/down, lm_head; 1 epoch, per-device batch 2 × grad-accum 16 = 32, LR 2e-4 linear, no warmup, max length 8,192, bf16.
- **Validation:** fold-0 accuracy numeral 1.000, gravity/unit 0.997, cipher 0.994, bit 0.891, equation 0.885, cryptarithm 0.043; separates target errors from imitation errors (only 8 bit cases where the source CoT was right and the model wrong). CV, LB and final were positively correlated.

### Rank 10 – kansai-kanto-kaggler (0.880), topic 708535
- **Bit reasoner rewritten** to the true generator form `out[i] = F(rotl(x,a)[i], shl(x,b)[i], shr(x,c)[i])` with F read from the examples as a 3-bit **key table**; offsets recovered at boundary bits (bit 0 has no shr, bit 7 has no shl). Oracle 1,364 → **1,582/1,602 (98.75%)**; remaining 20 = 11 underdetermined + 9 candidate-selection errors; 24 traces over 8,192 tokens dropped. Removing the candidate `List:` enumeration also stopped runaway loops.
- **Equation:** corpus stats (`add` 250, `mul` 247, `sub` 233, `abs` 159; wrapper shared within a problem in 688/688; operations not repeated in 676/688) → reorder deduce candidates (+14, 541 → 555) and rebuild guess (21 → 56/136).
- **Training:** full data, no OOF exclusion, >8,192-token samples dropped; priority samples (fragile bit and cipher demos, low-logprob problems) duplicated 2x; LoRA r 32, **α 128 (α/r = 4, improved CV)**, dropout 0, all projections incl. lm_head; LR 2e-4 linear, 2 epochs, effective batch 32.
- **Selection:** best-public run (val 0.876, public 0.872, private 0.852) vs chosen run (val 0.904, public 0.860, private 0.880) – trusting validation over public LB paid.
- **Did not work:** curriculum easy-to-hard; two-stage training on failure demos (redistributed wins/losses); 3x priority (overfit); excluding low-confidence/failed traces (hurt); add/sub/mul cryptarithm traces (logical jumps, unlearnable); LoRA SVD merge and seed soup (soup fell to the worst member).

### Ranks 11, 12, 14
No writeups, no forum posts. Rank 12 "Team" includes conor (conormacamhlaoibh), the same user who placed 5th in AIMO1.

### Rank 13 – Haraguchi-T (0.872, public 84th), topic 709646
- OPP reasoners plus heuristics; oracle 87.7% → 89.4% (bit 89.7%, equation deduce 95.3%, guess 50.7%, cryptarithm ~10%).
- Bit: explicit MAJORITY/XNOR 3-input checks; search restricted by operand bit positions; left-edge propagation fallback; dropped a parsing display to save tokens. Cryptarithm: answer-format cleanup (no extra braces), lenient matching against transcription errors, rebalanced teacher-correct:wrong 1:1 and ~3x more samples seeded from equation problems. Equation: four exclusive families, shared reversal by majority vote, guess tie order subtraction > addition > multiplication > concatenation.
- **Training:** Unsloth (konbu17's notebook, dgxchen's improved version); r 32 / α 32, q/k/v/o, in/out, up/down (no lm_head listed); deliberately overfit on all data in 3 stages: 9,277 samples × 2 epochs at 2e-4 linear; same × 1 at 2e-5 cosine; 11,315 (bit and cryptarithm upsampled) × 1 at 5e-6 cosine; effective batch 32.

### Rank 15 – MAJ0RT0M (0.872), topic 711727
First Kaggle competition. OPP code and hyperparameters, on Tinker; only change ~200 → ~2,000 steps.
- **Bit:** builds an arity-3 truth table for the middle bits from OPP's offsets, backtracking over the third offset, and fills slices from the flanking arity-2 functions; reference 85% → 92.1%, model 93.3% on train (higher than the solver).
- **Cryptarithm:** following kenshim's approach (discussion/703479) – pick 4-digit-output examples (must be concat or multiplication), recall a **memorised multiplication table** keyed by digit pattern, pin ~6 digits, then backtrack. Implemented as a tiny language (GOTO + registered Python functions) whose interpreter emits an execution trace; only outputs are trained on. Reference 28%, model 136/659 (~20%).
- **Data:** 536,583 examples / 256M tokens, of which 503,682 are multiplication-table "drills" (48.7% of tokens), 10,000 synthetic cryptarithm, 4,000 synthetic arity-3 bit.
- **Did not work:** varying LR (2e-4 optimal); **focal loss** raised minimum logprob but made inference unstable by pushing tokens toward the −0.69 threshold. Tinker treats Mamba projections as separate matrices (rank 32 becomes rank 64 in vLLM); OPP's SVD conversion did not measurably increase divergence.
- Estimates his changes were worth ~7 private problems (+0.02 to 0.03) over OPP; never reproduced OPP's 0.85 himself.

### Ranks 16, 17, 19, 20
No writeups, no forum posts. Ranks 17 and 19 were ~1,736th and ~1,781st on public – the biggest risers in the top 20.

### Rank 18 – Y | M | F (0.868, public 8th), topic 715330
- OPP pipeline extended over versions expv001 to expv006: backtracking cryptarithm with larger budgets; whole-byte logical shifts for bit manipulation; reasoner-filtered synthetic problems (400 bit, 1,400 equation deduce, 1,400 cryptarithm deduce); answer-guided selection among rule/format candidates; no-GT broader search (cryptarithm deduce 54 → 160); conservative GT-assisted tie-breaking among example-consistent candidates (equation deduce 596/596, cryptarithm deduce 184).
- **LoRA fusion:** rescale each adapter by α/r, average the updates, truncated SVD back to rank 32; beat TIES and DARE in their tests.
- **Self-diagnosed failure:** no local validation; trained on full data and used public LB. Public vs private per version: expv001 0.872/0.864; expv003 (synthetic) 0.868/0.872; expv006 (GT-assisted) 0.864/**0.880**; selected fusion expv001_002 0.876/0.868. They dropped the synthetic direction because public looked worse; the private board says it was better.

## 3. Technique frequency matrix

Columns = ranks with writeups (1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 15, 18). Ranks 8, 11, 12, 14, 16, 17, 19, 20 are all `?` and omitted. Y / N / ? as in the brief. Ranks 1 to 3 from the bank.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 10 | 13 | 15 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Built on OPP (huikang) code/traces | Y | ? | Y | Y | Y | Y | ? | Y | Y | Y | Y | Y |
| SFT only (no RL, no distillation from bigger LLM) | Y | Y | Y | Y | Y (RL failed) | Y | Y | Y | Y | Y | Y | Y |
| Deterministic solver-generated CoT | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| Large synthetic set (beyond train.csv) | Y (220k+140k) | Y (57.6k) | Y | Y (15k) | Y (no train rows used) | Y (all synthetic) | Y | N | ? (full train + priority dupes) | Y (crypt ×3) | Y (536k incl. drills) | Y (small) |
| train.csv used as validation (not training) | Y (15%) | Y | Y (10% holdout) | ? | Y (all) | Y (all) | ? | Y (1 fold) | Y (950) | N | N | N |
| Rank 32, α 32 | Y | ? | Y | Y | Y | Y | ? | Y | N (α 128) | Y | Y | ? |
| lm_head in LoRA targets | Y | Y | Y | Y | ? | ? | ? | Y | Y | N | ? | ? |
| MoE experts untied | ? | N (accidentally tied) | Y | Y | ? | ? | ? | ? | ? | ? | ? | ? |
| Tinker used | ? | Y→Megatron | N | N | N | Y (late) | ? | ? | ? | N | Y | ? |
| Unsloth | Y | N | N | N | N (failed with 4-bit/packing) | Y | ? | ? | ? | Y | N | ? |
| Hex / compact encoding of bits | Y | ? | Y | N (anchor tables) | N (encoding failed) | Y | N (single-bit tokens) | N | N | N | N | N |
| Memorised catalog / tables in traces | Y (signatures) | ? | Y (two-stage recall) | N | N | N | N | N | N | N | Y (mult tables) | N |
| Real cryptarithm solver (beyond concat) | Y | N (mostly untouched) | Y | ~ (prior only) | N | Y (DFS) | N | N | N | N | Y | Y |
| Multi-stage / continuation training | Y (2) | N (single CE beat it) | Y (2) | N | Y (15 iterations) | N | ? | N | N (2-stage hurt) | Y (3) | N | N |
| Weight averaging / LoRA merge | N | N | N | N | Y (soups) | N | ? | N | N (soup failed) | N | N | Y (SVD fusion) |
| Ground truth used to pick traces | Y (accidental GT box) | ? | ? | N | N | N | ? | N | N | N | N | Y (tie-breaks) |
| Selected by validation over public LB | ~ | Y | Y | ? | N (public + stability) | Y | ? | Y | Y | N | ? | N |

Counted summary (M = 12 teams with writeups): 12 of 12 used SFT on deterministic solver traces and none shipped RL or distillation; 11 of 12 explicitly built on Tong Hui Kang's OPP (the 12th, rank 7, is unstated; rank 2 cites the generator approach, bank); 11 of 12 used synthetic data beyond train.csv (only rank 9 did not); 7 of 12 kept train.csv (or a fold) out of training for validation; 9 of 12 that state it used r 32 / α 32 (rank 10 used α 128); 5 of 12 built a real cryptarithm solver beyond concatenation, and all 5 are in ranks 1, 3, 6, 15, 18 – i.e. cryptarithm did not by itself decide the top 5; 2 of 12 merged adapters in the final (5th soups, 18th SVD fusion), while 10th found soups fell to the worst member.

## 4. Consensus vs divergence

- **Table stakes:** OPP's trainer and trace philosophy (deterministic, temperature-0 friendly, minimum-logprob mindset); rank-32 LoRA covering attention, Mamba in/out projections and MLPs (most also lm_head); SFT on solver-written traces; near-perfect cipher, numeral, gravity and unit conversion. Every documented gold team did this.
- **What separated the top 5 from 6 to 20:**
  1. **Bit manipulation coverage near 99%** with a trace the model can actually execute: 1st (catalog repair), 4th (5 templates, 99.4%), 10th (key table, 98.75%) vs OPP's 85%. But 10th shows a 98.75% bit solver alone gave 0.880, so bit was necessary, not sufficient.
  2. **Cryptarithm via memorisation + search** (1st signature catalog, 3rd two-stage recall) – the only teams above 0.90 with real cryptarithm coverage. 2nd reached 0.908 with cryptarithm "mostly untouched" by having the cleanest everything-else and the best LoRA export.
  3. **Validation discipline:** top teams held train.csv out and chose by validation; 18th (no validation) and 13th (trained on everything) self-report choosing worse submissions.
- **Contradictory findings (with URLs):**
  - Multi-stage training: helped 1st (continuation), 3rd (memorise-then-solve, 16/17 vs 12/17), 5th (iterations + soups), 13th (3-stage); did not help 2nd (single CE beat multi-stage, bank) or 10th (two-stage redistributed wins and losses). Topics 709231, 709136, 708538, 709646 vs 711703, 708535.
  - Adapter averaging: 5th credits LoRA soups for stability (708538); 18th's SVD fusion was its selected submission (715330); 10th found seed soup "degraded to the level of the worst member" and SVD merge could not beat the best single adapter (708535).
  - Encoding bits compactly: 1st and 6th won with hex; 5th's 5-bits-per-letter encoding "broke the model completely"; 7th went the opposite way, one token per bit. Topics 709231, 709120 vs 708538, 712395.
  - Batch size: 3rd (effective batch 4) and 4th (16, micro 4) found smaller batches learn better; 9th, 10th, 13th used 32; 6th used 32 at LR 3.5e-4.
  - LoRA alpha: 10th found α/r = 4 improved CV; nearly everyone else kept α = r = 32.
  - Excluding wrong traces: 10th found removing low-confidence/failed traces hurt; 13th rebalanced teacher-correct:wrong to 1:1; 1st found a GT-substituted final answer taught the model to correct its own solver.
  - Focal loss / reweighting: 2nd (bank) and 15th both found it worse than plain CE (15th: it moved tokens toward the −0.69 greedy threshold and destabilised inference).

## 5. Aggregated "what did not work"
- RL / GRPO (throughput, air-gapped notebooks, compile errors): 5th; Komil Parmar thread (bank); OPP author argues against it (bank).
- Pseudo-GRPO via iterated DPO: 5th.
- Distilling a larger LLM: OPP (bank); no top team did it.
- Short CoT without visible logic (even 250k examples): 5th.
- Compact letter encoding of bit columns: 5th.
- Unsloth with 4-bit quantisation and packing on Nemotron: 5th.
- DoRA (vLLM rejects, conversion lossy): 5th.
- Focal loss, token reweighting, multi-stage schedules: 2nd (bank); focal loss: 15th.
- Tinker LoRA with lossy SVD export (cost 0.81 vs 0.89 bit accuracy): 2nd (bank); 15th found the SVD conversion harmless in its own analysis – contradictory.
- Curriculum learning: 10th.
- Two-stage targeted fine-tuning: 10th.
- 3x priority duplication: 10th.
- Excluding failed/low-confidence traces: 10th.
- add/sub/mul cryptarithm traces with logical jumps: 10th; 5th's guess-style cryptarithm CoT.
- Seed soup and SVD merge of complementary adapters: 10th.
- TIES and DARE merging (vs own SVD fusion): 18th.
- Doubling cryptarithm data at the expense of bit tokens (0.5B set): 6th.
- Including NOT-CHOICE bit formula in training: 6th (left out as too hard).
- Trusting public LB to abandon synthetic data: 18th.
- Varying LR around 2e-4: 15th (2e-4 optimal).

## 6. Shake-up note
Moderate at the very top, large below. Biggest risers inside the top 20: 19 Ryuto Yoda (public 1,781 → 19), 17 Frank Yang (1,736 → 17), 7 Solved Bit Manipulation (531 → 7), 20 T T (322 → 20), 11 Hello World and 13 Haraguchi-T (82/84 → 11/13), 3 YS-L (42 → 3). Fallers: public 2nd → 6th (Alehandreus & Yurnero), public 8th → 18th (Y | M | F). Explanations from writeups:
- A visible public "wall" at ~0.86 to 0.88 hid real progress: 3rd (bank), 5th (public flat at ~88 while validation and private rose), 18th (synthetic version looked worse on public, was better on private).
- 10th's chosen run had lower public (0.860) but higher validation; it was +0.028 better on private than its best-public run.
- 7th joined in the last 10 days with a strong bit-only approach; its public 0.864 understated it.
- vLLM is not fully deterministic at temperature 0: 1st (bank) and 5th report identical adapters varying by up to ~1% between submissions; with ~250 private problems each problem is 0.004.
- 6th explains its own drop: chose two seeds of the safer 0.25B adapter over the 0.5B set.

## 7. UNVERIFIED
- Approaches of ranks 8, 11, 12, 14, 16, 17, 19, 20 (no posts found).
- Rank 7's training hyperparameters (not in the post or arXiv abstract) and their ">96% bit, highest of any team" claim (4th reports a 99.4% reasoner; model-level numbers are not comparable).
- Whether the API public column is the selected or best public submission (10th's writeup 0.860 vs API 0.872).
- Whether 2nd place built on OPP code (bank describes generator reverse-engineering; not re-checked here).
- Test-set size: teams estimate ~250 public / ~250 private; not confirmed by the host in this pass.
- Recipients of the three Open Contribution Awards (carried over from the bank).
