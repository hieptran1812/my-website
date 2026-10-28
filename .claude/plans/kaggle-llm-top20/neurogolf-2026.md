# The 2026 NeuroGolf Championship: top-20 addendum

- Slug: `neurogolf-2026` (competition id 116438, forum 8930897). Final board pulled from `competitions.LeaderboardService/GetLeaderboard` on 2026-10-28.
- Deep coverage of ranks 1 to 3 (Kaggle Agent, neurogolf team, The Final Ensemble) and one-line stubs for 4 to 10 live in `.claude/plans/kaggle-llm-writeup-bank-part3.md`, H2 `## The 2026 NeuroGolf Championship` (lines ~758 to 890). This file adds ranks 4 to 20.
- Board mechanics: LB 100% public, so public rank = private rank for every top-20 team (API). Score per task `max(1, 25 - ln(cost))`, cost = parameters + charged intermediate memory (MACs free after the May 6 reset). Ranks 1 to 20 span 8314.03 to 7956.78.
- **Rank numbering warning:** several teams were removed after the deadline (see shake-up), so most writeup titles carry a pre-removal rank one or two higher than the API rank: "6th" claudex is API 5th, "7th" Slow and Steady is 6th, "8th" kluges clueless is 7th, "9th" Pavel & CroDoc is 8th (title says 8th, Chris Deotte's comment says 9th), "11th" Kimura is 10th, "15th" KOKOROMI is 14th, "18th" Fususu is 17th, "21st" robga is 19th. API ranks are used below.

## Top-20 table

| Rank | Team | Members (display name (username)) | Score | Public rank | Writeup | Approach (one line) |
|---|---|---|---|---|---|---|
| 1 | Kaggle Agent | Giba (titericz); Jiwei Liu (jiweiliu); Max Jeblick (maxjeblick); Eduardo Rocha de Andrade (arc144); Chris Deotte (cdeotte) | 8314.03 | 1 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/1st-place-kaggle-agent (topic 726654; + 726799, 726883) | Agent fleets, targets, task notes, cookbooks, self-evolving prompts (bank) |
| 2 | neurogolf team | jacekwl (jacekwl); Yuichiro Hirano (yhirano); Chan Kha Vu (chankhavu); Geremie Yeo (yeoyunsianggeremie); Jin Niu (gufanmingmie) | 8243.89 | 2 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/work-in-progress-2nd-place-neurogolf-team (topic 728549) | 5 independent agent pipelines, ChatGPT 5.5 Pro + Codex (bank) |
| 3 | The Final Ensemble | Dieter (christofhenkel); matheus (tomirol); Fritz Cremer (fritzcremer); Diana (dziyanavalenta); Tony Li (tonylica) | 8221.27 | 3 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/3rd-place-solution-writeup (topic 727356) | Codex workers + ChatGPT UI, attempt ledgers, forced rewrites, consensus validation (bank) |
| 4 | M & H & M & N & M | mrmldjr (mrmldjr); Maciej Sypetkowski (maciejsypetkowski); MaxChen303 (maxchen303); hongan (honganzhu); Natalia Jaśkowska (nataliajakowska) | 8142.02 | 4 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/4th-place-solution-writeup (topic 727134) | Async agentic loop + task registry + voted idiom catalogue + grouped LB-probe bisection + runtime-budgeted assembly |
| 5 | claudex | Bình (binhdth); Bửu Phạm Quốc (buphmquc); Phạm Ngọc Thiên Ân (pntan17); Hai (Tom) Nguyen Thien (hainguyenthien); Otter (myotterspace) | 8118.97 | 5 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/6th-place-solution-claudex (topic 726691) | Two sub-teams swapping 200-task halves; Codex + web ChatGPT streams; score-band operator library; strict validation gate |
| 6 | Slow and Steady | David Austin (tivfrvqhs5); M Sato (masasato1999); yash bhaskar (yash9439); Paritosh Kumar Tripathi (paritoshtripathi5); cg (chaitanyagarg2) | 8067.63 | 6 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/7th-place-solution-slow-and-steady (topic 726660) | ChatGPT Web (GPT 5.5 High) with prompt + per-task zip + 68-rule "ONNX golfing" rulebook; later Codex |
| 7 | kluges clueless | mudesteven (fastlight2007); NIkita Chervov (nikitachervov); Ivan Gevorkov (ivangevorkov); alexeyshmelev (alexeyshmelev); noobanot (noobanot) | 8048.16 | 7 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-writeup-draft (topic 726693) | 330-worker ChatGPT desktop factory on 11 flat-rate accounts, git as blackboard, probe-gated certification |
| 8 | Pavel & CroDoc | Pavel (pavelsavchenkov); CroDoc (crodoc) | 8038.62 | 8 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-gold-pavel-and-crodoc (topic 726782) | Two people, overlapping tasks; coding agents with MEMORY.md + archive; Codex CLI pipeline then 25 ChatGPT 5.5 Pro sessions |
| 9 | Timeout exceeded : ( | Yiheng Wang (yiheng) | 8028.25 | 9 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/9th-place-solution (topic 726653) | Solo; multi-turn Codex scheduler with transactional three-file task units + ChatGPT Pro architecture search |
| 10 | Kimura@SeSDA & OverfitOracle | jazivxt (jazivxt); Ryuhki Kimura (kimura0415); OverfitOracle (alisalmanrana); hwe owe (hweowe); SOLID ACE (avijaynandha) | 8025.82 | 10 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/11th-place-solution (topic 727156) | Model memory access + score-discrepancy bisection of failing tasks with Claude doing exhaustive subset search (short writeup) |
| 11 | Carlos, Egor, Rubempre & Bill | Bill (deeper); Lucien de Rubempre (lucienderubempre); Egor Trushin (egortrushin); Carlos Pérez Ricardo (carlosperez97) | 8022.05 | 11 | no writeup found | unknown |
| 12 | Sol | Yang (trudih) | 8015.02 | 12 | no writeup found | unknown (solo; posted "GPT 5.6 is coming soon!", topic 723782) |
| 13 | Santa Neuro Club 10000 | Ueddy (ku0807); 4eta (bio4eta); ymg_aq (ymgpalaqs); Kazuo Watanabe (watanabekazuo); yu_w (yuw000) | 8013.53 | 13 | no writeup found | unknown (largely the 9th-place Code Golf 2025 team; ymg_aq proposed file-size scoring, topic 697079) |
| 14 | KOKOROMI | Stmx35 (stmx35); tomo2677 (tomo1314); rook (tknakamu); re (refv00); igaiga (schrodinger0112) | 8012.12 | 14 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/14th-place-writeup (topic 728123) | Autonomous Claude+Codex harness abandoned for one-task-per-chat ChatGPT Web with task notes as constraints and numeric targets |
| 15 | carbonapi | carbonapi (nonempty) | 8000.10 | 15 | no writeup found | unknown (solo) |
| 16 | nmdcddwzml | Qsj (qiaoshiji); Sinan Calisir (snnclsr); tmp (baosenguo); nmdcddwzml (mujian63); insuperabilehart (insuperabile) | 7998.91 | 16 | no writeup found | unknown |
| 17 | Fususu | Fususu (phuongncn) | 7994.76 | 17 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/i-deleted-my-entire-project-6-days-before-the-dead (topic 726670) | Solo, 21 days; 64 concurrent coding agents, "seven-year-old" advisor, 250+ hand-written rules; survived `rm -rf` |
| 18 | Token is my sugar daddy! | Damien Mourot (nahman); KKY (evilpsycho42); NT (mitubant); leo (calibrator); Russell Kirk (russcore) | 7960.75 | 18 | no writeup found | unknown (calibrator found the negative-memory exploit, topic 693247) |
| 19 | robga | robga (robga) | 7958.58 | 19 | https://www.kaggle.com/competitions/neurogolf-2026/writeups/21st-place-solution-representation-collapse (topic 726748) | Solo; ChatGPT Pro 5.5 reasons about memory representations, Codex lowers to Python/ONNX; later 6 Codex CLIs on 5.6 Sol Ultra |
| 20 | Just A game on your lips | SangGold (sanggold); Tue Nguyen (proptiter) | 7956.78 | 20 | no writeup found | unknown |

Writeups found: 14 of 20 (ranks 1 to 10, 14, 17, 19; 10 is a stub). 291 forum topics scanned (16 pages).

## Per-team synthesis (ranks 4 to 20)

### Rank 4: M & H & M & N & M (8142.02)
Source: topic 727134. LLM vendor not named in the writeup.
- **Asynchronous agentic loop**, one task per loop, many loops in parallel. Prompt: pre-crafted task description + reference algorithm, current best ONNX with its cost profile, a scorer reference, and the evolving idiom system. Tools: model evaluation, vote on or add idioms. **Bottleneck-targeted hints**: rethink from scratch after a plateau; repair after local or LB failure; optimise runtime when evaluation is slow.
- **Task registry** keeps every model version by hash (not only the best) with score, local eval time and LB status (unknown / correct / broken). Registration checks include conv bias shorter than output channels (nondeterministic reads on the grader).
- **Idiom system**: a catalogue built by agent tools plus semi-automatic mining; agents vote and comment per task; repeatedly misleading idioms get disabled; each prompt gets only the task's own comment plus a summary of all comments, keeping context small. Example idiom "ULP-collision finite-difference selector" (fp32 rounding collisions as an exact equality gate) carried 89 measured +2 confirmations vs 8 failures.
- **Verification by grouped LB probes**: broken models score exactly 0, so adaptive bisection over groups isolates them.
- **Submission assembly** as an approximate knapsack: maximise total score subject to summed local eval time, since some 1-parameter Einsums take hours on one CPU.
- Final: 9 tasks at the perfect 25; 224 models are a single node, 193 of them a single Einsum; 349 contain an Einsum; 5,747 nodes total; largest model task 023 with 491 nodes (18.151). Examples: task 267 as one Einsum with a 2-element fp32 tensor (`0xC5BCCB9A`, `0x45BCCF43`) whose rounding cancels unwanted colour routes (cost 2, 24.307; contraction order is part of the program); task 053 as one parameter-free `LpPool` with dilation 3, negative padding and stride -2 acting as a row gather (cost 0, score 25).

### Rank 5: claudex (8118.97)
Source: topic 726691; repo https://github.com/PNTAN17/neurogolf .
- Two sub-teams each took 200 tasks, then **swapped ranges every one to two days** to escape local maxima; this still gained 10+ points in the final minutes (author reply). Within each sub-team: a Codex stream and a web ChatGPT stream.
- Only strict improvements (or equal score, better runtime) promoted per round; LB 0.0 scores, timeouts and format failures fed back as task-specific warnings.
- Codex prompt config block: ONNX/ORT versions, paths, current score and best source, official scorer and validation commands, task JSON, ARC-GEN generator, and **`target_base` of +0.1 to +0.5**. Web prompts: the same packaged as a zip, with examples from nearby score bands. A **score-band operator library** suggested the operator families common in a task's band.
- Tricks: single Einsum first for linear/recolour/reflection/tiling; analytic coordinate grids instead of stored masks; `ConvTranspose` for tiling/upscaling; `QLinearConv`/int8 or 1x1 conv for colour classification; build the full one-hot only at the output; bool/uint8/int8/fp16 intermediates; make the largest tensor the (free) output; switch representation when shrinking stalls; reorder Einsum operands for runtime.
- Validation gate: static I/O `[1,10,30,30]`, no sparse initializers or fragile ops, checker + shape inference, conv bias length = output channels (added after a forum comment), ORT 1.24.4 run, public JSON exact, **1,000 fresh ARC-GEN samples**, file size limit; singleton submissions to localise risky failures.
- Humans supplied most high-level insights; Codex implemented and searched; attempt logs stopped Codex retrying failed operators (author reply).

### Rank 6: Slow and Steady (8067.63)
Source: topic 726660.
- Early: Codex, Claude Code and Gemini to ~7200, most gains from GPT 5.5. After merging, M Sato's rulebook `onnx_golfing_expert_hints.md` became the backbone (R1 to R57 in the prompt text; 68 rules by the end per Sato). Final workflow centred on **ChatGPT Web**, later extended to Codex.
- Each chat: `NEUROGOLF_PROMPT.md` (role "Expert ONNX Golfer", file guide) + `taskXXX.zip` with 7 files: ARC-GEN generator (read first for invariants that license cheaper graphs), `common.py` (mass-simulate thousands of cases), task JSON, current ONNX baseline ("at-floor claims expire"), official scorer utils, list of 163 usable ONNX ops with caveats, and the rulebook. Example rule R36: zero-intermediate networks; task 110 as one 5-input Einsum, cost 22871 -> 7310 -> 110.
- **ChatGPT memory turned off**, which they found increased thinking time; multi-turn follow-ups; extra ARC-GEN validation left to the model (50 to 50,000 cases).
- Runtime: give a fresh session the fast baseline + slow improved ONNX, ask for speed; done for the slowest 20 tasks periodically.
- Regrets: over-reliance on ChatGPT Web with 5.5 High; Pro models often failed ("Thinking Failed"); should have used Codex with 5.5 Pro/xHigh and later 5.6 Sol; H-ARC references (from 1st) and all-400-task context (from 3rd) would have helped.
- Comment thread: Kaggle Agent and kluges clueless report the same agent pessimism ("this task is at its floor"); kluges scrubbed thousands of such phrases from logs.

### Rank 7: kluges clueless (8048.16)
Source: topic 726693 (part by fastlight2007 and noobanot; the other three members ran separate pipelines, not described).
- **No metered API**: 1 ChatGPT Pro + 10 Plus accounts, about $300/month; volume is free, Kaggle submissions are the bottleneck.
- Key finding: a batch of 100 candidates with zero errors on full local ARC-GEN lost **~20% of projected points** to hidden overfits on a sparse Kaggle probe. "Local clean is not hidden clean."
- **Factory**: ChatGPT desktop app copied into per-account bundles (binary-patched to allow isolation), driven by Playwright over CDP, 30 background workers per account, ~330 concurrent chats. Prompt bans ops proven to score 0 on hidden (`Loop`, `Scan`, `NonZero`, `Unique`, `Compress`, `If`, sequence ops, sparse initializers) and forbids fitting shown examples. Plus a Codex CLI fallback fleet, three algorithmic search engines, and a competitor-model scraper, all scored by one CPU scorer.
- **Git as blackboard**: workers upload builders through ChatGPT's GitHub connector using only `create_branch` + `create_file`; a GitHub Action builds and checks; `instructions/CURRENT.md` prepended to every prompt steers all workers with one commit. ~24,371 launches, 19,038 builds, 5,058 branches, 14,771 commits.
- **Gate**: static-cost tracing of shape ops, runtime gate (whole bundle times out at 30 min; drop swaps under 0.05 points per added second; one swap added 118 s for 0.036 points and killed a bundle), risk-pool split, sparse probe, conv-bias scan.
- **Probe economy**: probes are additive, so probe one half and infer the other; naive subset-sum decoding failed (one ambiguous bank regressed -74.79; now require a unique subset); alternating-split bisection is the default; region-residual consistency decoded a 128-task pool for +12.73.
- Tricks: single-Einsum collapse in 136 of 400 banked models (four families by how many input views); ConvTranspose with the input as weight (task 326 cost 30 -> 4 with NaN taps); negative-pad crops (works on Conv/LpPool/ConvInteger; MaxPool/AveragePool/Crop crash the grader); fp32 underflow as a free threshold (task 129 cost 1); Einsum operand reorder (task 294 39 s -> 0.07 s).
- Did not work: huge multi-operand Einsums (an LLM-claimed +12.49 round delivered +6.49; blow-up predicted by peak index footprint, peak RSS ~ 90 MB + 2.7 x footprint); generator ops (`ConstantOfShape`, `EyeLike`, `Range` are charged as memory); the "large-pad flaky" hypothesis (cost ~3 clean points; the only flakiness cause is short conv bias); dynamic-dim intermediates make the scorer return memory None (phantom-cheap); ORT-GPU per graph (~30% unrunnable) though a batched ONNX-to-torch screen gave ~450x; search engines (MAP-Elites/MCTS, typed Einsum grammar, equality saturation) plateaued at ~+1.5 but seeded stuck agents.
- Biggest banked jumps: team merger +83.99; first GPT 5.6 Sol High pass +43.20; second +33.17. **Realisation rate of LLM-claimed gains ~29%.**

### Rank 8: Pavel & CroDoc (8038.62)
Source: topic 726782; repos https://github.com/CroDoc/neurogolf-2026 , https://github.com/PavelSavchenkov/kaggle-golf-8th-place-neurogolf-2026 .
- Did not split tasks; both worked the same tasks with different prompts and starting points, merging by verified score (ties: faster model).
- CroDoc: coding-agent sessions given examples, current ONNX + builder, task history, scorer constraints, shared notes. **Two-tier memory**: `MEMORY.md` (~200 lines, loaded every session) and `MEMORY_ARCHIVE.md` (full collection, searchable), periodically de-duplicated. Biggest gains from representation changes; "recompute instead of store" since MACs are free; end with an op writing directly to output. 202 of 400 models single-node, 273 end in Einsum.
- Did not work: from-scratch re-solve runs (rediscovered rules, rebuilt baselines, repeated failures); asking for a major rewrite while showing the current model and notes worked better. Runtime underestimated until it became the main constraint: best local set 8041.20 missed the time limit, submitted 8038.62.
- Pavel: local solution DB with web frontend, agents talk to it only via a submit API (full validation on entry). ~2 months of autonomous **Codex CLI** pipeline (recipes.txt framed several ways, append-only per-task `ideas_pool.jsonl`, streamed thinking logs; re-runs act as session resume) reached low gold unattended. Then a ChatGPT project with recipes + per-task logs and **25 manual GPT 5.5 Pro sessions** to ~7880 (3rd at the time).
- Trick: "gaslight AI that better score is possible": "Other people achieved [current]+1.0. Achieve it as well."

### Rank 9: Timeout exceeded : ( (Yiheng Wang, solo) (8028.25)
Source: topic 726653.
- Inspired by FuunAgent's Code Golf 2025 writeup (parallel Codex + verification + structured prompts); GPT-5 Codex was not reliable enough then, GPT-5.5 was. Built the system first.
- Already constrained his targets against loopholes before the May 6 metric reset; first safe bundle after the reset was 2nd, then **1st for ~15 days**.
- **Three-file task unit**: `taskNNN.onnx`, builder `attack_tNNN.py`, memory `docs/task-NNN.md` (rule, architecture, cost history, experiments, hidden-test risks); promoted or restored together. Workers also get the Code Golf 2025 winning Python solution for the task as an alternative algorithm description.
- **Multi-turn Codex scheduler**: bounded worker pool; one persistent Codex session per task (`codex exec` then `codex exec resume <id>`); transactional backup -> experiment -> validate (checker, shape inference, all examples, official cost) -> promote only if strictly cheaper, else restore; next prompt chosen by outcome (lower target / repair / rethink); checkpointed for resume.
- After a wave: submit changed tasks, compare expected vs observed online gain, bisect with differential submissions; then one serialized Codex step merges only online-confirmed tricks into `docs/tricks.md` (failures recorded as negative examples).
- A 5-day unattended run while on vacation: 7516.01 -> 7575.78 (+59.77), 271 models changed, 256 strict reductions, 0 regressions.
- ChatGPT Pro pipeline: Project with stable knowledge; per-task bundle; infer rule, compare with code-golf method, decide whether implementation or representation is wrong, propose at least three architecture families, separate candidate builder. **Five-task bucket chats** aimed at Fritz Cremer's published per-bucket totals (topic 708377). He found ChatGPT Pro reasoning "consistently outperformed" Codex xhigh for open-ended architecture search.
- Tricks: write directly to output (task 001 as one 226-operand Einsum, cost 98); recompute instead of carry state; one paid basis serving several roles; finite-state/number-theoretic algebra instead of tables; contraction order as model state.

### Rank 10: Kimura@SeSDA & OverfitOracle (8025.82)
Source: topic 727156 (stub; full writeup promised). Two stated keys: letting the model access its memory ("memory-based reconstruction"), and analysing the evaluator's error handling: bisect by score discrepancies of 200, 100, 50 points, then have **Claude** exhaustively search subset combinations below 50 and cross-check with previously known error tasks. Team found three competition bugs (Kimura's comment). Thread is mostly an internal dispute about contributions.

### Rank 14: KOKOROMI (8012.12)
Source: topic 728123.
- Joined ~40 days before the end. An autonomous Claude + Codex harness gave under 10 points/day and gave up on hard tasks. Switched to **ChatGPT Web, one task per conversation**: ~7,780 after the first pass, 7,780.86 -> 7,878.26 in the second, ~8,000 after revisiting "saturated" tasks.
- Inputs per chat: task data, current ONNX + solver, official scorer, task note. Lessons were **rewritten as constraints** in the task note rather than pasting failure logs, because long verbatim history anchored the model on earlier ideas.
- **Numeric targets**: +0.1 needs cost <= 90.5%, +0.5 <= 60.7%, +1.0 <= 36.8% of current; asking for +1.0 forces a representation change. Asked for three candidates: safe, maximum-cut, unconventional. Example task 305: 19.1449 -> 22.6974, cost 10, via scalar/bit representation of the diagonal period.
- Dashboard (tasks, scores, history, tags, I/O stepping) used mainly to choose next tasks and spot stale notes; human rule understanding did not translate directly into cheap ONNX.

### Rank 17: Fususu (solo) (7994.76)
Source: topic 726670.
- Joined with 21 days left; hand baseline ~6000, then a shared 7100 public notebook moved the field.
- Claude/Opus at max effort declared hard tasks "already optimal"; explaining the task "like I was seven" cracked one two exchanges later. Built an "advisor" that asks childlike questions when an agent is about to give up, and hand-wrote rules for **250+ tasks in seven-year-old language**; rewrites beat tweaks ~10x in his logs.
- Peak **64 concurrent coding agents** racing on the same 400 tasks, sharing bests and notes; 330,924 candidate models in three weeks; manual paste lane 5,452 attempts.
- `rm -rf *` in `/home` six days before the end deleted the infrastructure; rebuilt in a day from the AI assistant's conversation memory. Reached 13th, finished 17th (writeup says 18th), ~6 points short of gold, with validated improvements unbanked for lack of submission slots and a claim-flow bug that told idle agents "YOU WIN!".

### Rank 19: robga (solo) (7958.58)
Source: topic 726748.
- One ChatGPT Pro account and a work MacBook. Treated **graph representation as the goal and ONNX lowering as mechanical**: ChatGPT Pro 5.5 (90% of the timeline) reasons only about memory representations from a compressed task-history packet and is never asked to build ONNX; Codex implements a Python builder, then ONNX (an MLIR detour worked but was too complex).
- Prompt library to manage "pressure": guards against retreating to "floor" claims or fingerprint (overfit) solutions; phrases like "semantic collapse" and "adversarial basis swap" to push to new basins.
- GPT 5.6 changed things: latecomers probably got better basins from 5.6; switched to 5.6 Sol Ultra in Codex, with Codex managing 6 Codex CLIs. Would run two independent repos next time: "Diversity of representation mattered more than tuning one representation."
- **Architecture-sensitive numerics**: some Einsum solutions relying on exact cancellation produced 0 on x86 (Kaggle) but tiny positive residues (~1e-11) on ARM due to different SIMD/FMA reduction trees; since output is decoded as `> 0`, they pass LB and fail on a Mac. Deliberate underflow/Inf/NaN tricks share this property (task 262 removed ten parameters this way).

### Ranks 11, 12, 13, 15, 16, 18, 20: no writeup
11 Carlos, Egor, Rubempre & Bill; 12 Sol (solo); 13 Santa Neuro Club 10000 (overlaps Code Golf 9th team); 15 carbonapi (solo); 16 nmdcddwzml (raised the post-deadline scoring issue, topic 726658); 18 Token is my sugar daddy! (includes calibrator, finder of the negative-memory exploit, 693247); 20 Just A game on your lips. No method details found.

## Technique frequency matrix

Y = stated, N = stated not used, ? = no source. Ranks 11 to 13, 15, 16, 18, 20 have no source.

| Technique | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 14 | 17 | 19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Codex (CLI/exec) workers | Y | Y | Y | ? | Y | Y (late) | Y (fallback) | Y | Y | ? | Y (early) | ? | Y |
| ChatGPT web/Pro sessions | Y | Y | Y | ? | Y | Y (core) | Y (core, 330 chats) | Y (25 Pro) | Y | ? | Y (core) | ? | Y |
| Claude Code | Y | Y (driver) | ? | ? | ? | Y (early) | N | ? | N | Y (subset search) | Y (early) | Y | N |
| Shared trick cookbook / rulebook | Y | Y | Y | Y (voted idioms) | Y (operator library) | Y (68 rules) | Y (CURRENT.md) | Y (MEMORY.md) | Y (tricks.md) | ? | Y | Y | Y |
| Per-task notes / attempt history | Y | Y | Y | Y (registry) | Y | Y (zip) | Y | Y | Y (3-file unit) | Y | Y (as constraints) | Y | Y |
| Explicit numeric targets | Y | ? | Y | ? | Y (+0.1 to 0.5) | ? | ? | Y ("+1.0 achieved") | Y | ? | Y (+0.1/0.5/1.0) | ? | Y (30% lower) |
| Forced rewrite / rethink prompts | Y | ? | Y | Y | Y | ? | ? | Y | Y | ? | Y | Y | Y |
| Independent parallel pipelines for diversity | Y | Y | ? | ? | Y (swap halves) | ? | Y | Y | N | ? | N | ? | N (regret) |
| ARC-GEN / generator-based local validation | Y | ? | Y (consensus) | Y | Y (1,000) | Y (50 to 50k) | Y | ? | Y | ? | ? | Y | ? |
| LB probe bisection to find hidden failures | ? | ? | ? | Y | Y (singletons) | ? | Y | ? | Y | Y | ? | ? | ? |
| Runtime-aware selection | ? | Y (runtime lane) | ? | Y (knapsack) | Y | Y | Y | Y | ? | ? | ? | Y | Y (30 s cap) |
| Code Golf 2025 solutions as references | ? | Y (then removed) | Y | Y? (reference algorithm) | ? | ? | ? | ? | Y | ? | ? | ? | ? |
| Algorithmic (non-LLM) graph search | ? | N (brute force did not pay) | ? | ? | ? | ? | Y (mixed) | ? | ? | ? | ? | ? | ? |
| Single-Einsum terminal node dominant | Y | Y | ? | Y (193 of 224) | Y | Y | Y (136) | Y (273 end in Einsum) | Y | ? | ? | ? | Y |
| fp rounding / underflow tricks | ? | Y | ? | Y (ULP collision) | ? | ? | Y | ? | ? | ? | ? | ? | Y |

Counted summary (13 teams with a source: 1 to 10, 14, 17, 19):
- 13 of 13 drove LLM coding agents; none hand-built more than a fraction of graphs.
- 10 of 13 used ChatGPT web/Pro sessions (unknown for 4, 10, 17); 10 of 13 used Codex (unknown for 4, 10, 17).
- 12 of 13 kept a shared trick store (cookbook, rulebook, idioms, MEMORY.md, CURRENT.md, tricks.md); unknown for 10.
- 13 of 13 kept per-task persistent notes or histories outside the agent.
- 7 of 13 set explicit numeric improvement targets (1, 3, 5, 8, 9, 14, 19).
- 5 of 13 described submission-probe bisection to localise hidden failures (4, 5, 7, 9, 10).

## Consensus vs divergence

- Table stakes: GPT-5.5/5.6-class models in Codex and/or ChatGPT Pro; persistent per-task memory; a shared trick file; single Einsum writing straight to output; local validation against freshly generated ARC-GEN cases; runtime screening for the 30-minute bundle limit.
- What separated top 3 from 4 to 20: more independent pipelines/people with merge discipline (1, 2, 3 are five-person teams running several distinct lanes), explicit targets plus forced rewrites, and hidden-test proxies (3rd's consensus of top-10 Code Golf programs). 4th to 9th have equally elaborate infrastructure; the gap from 3rd (8221) to 4th (8142) is ~80 points, and from 4th to 10th ~116.
- Contradiction 1 (ChatGPT web vs Codex): Yiheng (9th) says ChatGPT Pro reasoning "consistently outperformed" Codex xhigh for architecture search (726653); Slow and Steady (6th) regret relying on ChatGPT Web and wish they had used Codex with Pro/xHigh (726660); robga (19th) moved entirely to Codex after 5.6 Sol Ultra (726748); KOKOROMI (14th) abandoned an autonomous Claude+Codex harness for ChatGPT Web (728123).
- Contradiction 2 (autonomy): Pavel (8th) ran a fully autonomous Codex pipeline to low gold "without babysitting" (726782) and Yiheng gained +59.77 unattended over 5 days (726653); KOKOROMI's autonomous harness made under 10 points/day and gave up too quickly (728123); Fususu found max-effort models quit on "optimal" claims without human coaching (726670).
- Contradiction 3 (from scratch vs incumbent): CroDoc (8th) found from-scratch re-solves mostly rebuilt old baselines; rewrite prompts that still show the current model worked better (726782). Kaggle Agent (1st) and The Final Ensemble (3rd) push forced rewrites and found rewrite >> optimise (bank). KOKOROMI warns verbatim failure history anchors the model (728123), while kluges and claudex feed failures back as warnings.
- Contradiction 4 (search engines): neurogolf team (2nd) says brute-force search did not pay (bank); kluges (7th) found algorithmic engines plateau at ~+1.5 but useful as diverse seeds (726693).

## Aggregated "what did not work"

- Trusting local ARC-GEN cleanliness: ~20% of projected points lost to hidden overfits (7th); fresh-sample validation and probes needed (5th, 9th).
- Huge multi-operand Einsums: paper-cheap but time out or core-dump; LLM-claimed gains realised at ~29% overall, worst for these (7th); runtime-slow Einsums forced knapsack selection (4th) and cost 8th its best set (8041.20 not submittable).
- Generator ops (`ConstantOfShape`, `EyeLike`, `Range`) as parameter savers: counted as memory (7th).
- MaxPool/AveragePool/Crop negative-pad crops: crash the grader (7th).
- Dynamic-dim intermediates: scorer returns memory None, phantom-cheap (7th).
- Short conv bias vectors: nondeterministic reads, flaky scores (4th, 5th, 7th).
- Naive subset-sum decoding of probe deficits (7th).
- ORT-GPU evaluation per graph (7th).
- Autonomous Claude + Codex harness without targets (14th); agents declaring "floor" (6th, 7th, 17th, 19th, bank 1st).
- From-scratch re-solve runs (8th); pasting full failure logs (14th).
- ChatGPT Pro web sessions failing with "Thinking Failed" (6th; bank 2nd).
- Fine-tuning a single representation rather than running diverse repos (19th, hindsight).
- ARM-vs-x86 floating-point differences: solutions valid on Kaggle fail on Mac (19th).

## Shake-up note

No public/private shake-up (100% public board). Movement came from three sources:
- **Removals after the deadline**: per robga's list (topic 727985, at time of writing): 4th "weluveleven", 19th "where did my tokens go?", 24th "NeuroSOLVers", 27th Md Boktiar (solo), and others at 80th, 82nd, 111th, 120th. The NeuroSOLVers leader's investigation (topic 729721) found 74 bit-identical graphs shared with "where did my tokens go?", traced to a couple sharing a server and a ChatGPT account; one member of each team was banned. Ken Krige (NeuroSOLVers) disputes the lack of evidence from Kaggle. These removals shifted every writeup's rank by one or two places.
- **Post-deadline scoring**: a submission created before the deadline finished scoring ~28 minutes later and displaced nmdcddwzml, who had received a 16th-place email; Kaggle (Addison Howard) confirmed it was valid (topic 726658).
- **Solution selling/leak in the last weeks**: a 7300+ and then claimed 8100 public release triggered calls to freeze the LB (topic 724778); a dataset claiming 7957 was the public baseline encrypted (bank). Fususu's late climb (21 days, solo) and robga's note that latecomers "raced from 100+ to top 20 in a few days" are attributed by them to GPT 5.6, not leaks.

## UNVERIFIED

- Which LLMs rank 4 used (not named).
- Methods of ranks 11, 12, 13, 15, 16, 18, 20.
- The second half of 7th's writeup (three other members' pipelines).
- Full 10th-place method (writeup is a stub with a diagram).
- Whether the removed 4th team "weluveleven" removal is final (right of reply per 727985).
- Exact pre-removal rank of Pavel & CroDoc (title 8th, Deotte's comment 9th).
