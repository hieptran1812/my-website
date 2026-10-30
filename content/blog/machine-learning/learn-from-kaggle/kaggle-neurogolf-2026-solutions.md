---
title: "NeuroGolf 2026: the smallest network that solves an ARC task"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "NeuroGolf 2026 scored 400 tiny ONNX networks by the log of their parameters plus memory, and the top 20 teams built them with fleets of LLM coding agents. This is the metric worked by hand, the 14 published top-20 writeups synthesised, and what the agent operations that won actually looked like."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "neurogolf", "arc-agi", "onnx", "coding-agents", "agent-operations", "model-compression", "einsum"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** The 2026 NeuroGolf Championship (April 15 to July 15, 2026) asked 2,963 teams for one ONNX network per ARC-AGI-1 training task, 400 networks in all, each scored $\max(1, 25 - \ln(\text{params} + \text{memory}))$. Kaggle Agent (Giba, Jiwei Liu, Max Jeblick, Eduardo Rocha de Andrade, Chris Deotte) won with 8314.03 of a possible 10,000. Almost nobody drew graphs by hand: every one of the 13 top-20 teams with a usable writeup drove LLM coding agents (Codex, ChatGPT Pro, Claude Code) and spent their own effort on the harness around them. **Decisive lever: metric insight.** A logarithmic cost means a rewrite that changes the representation is worth ten tweaks, and the winners built their agent operations around forcing rewrites. **The one lesson:** when the work is done by agents, the edge moves to what you keep outside the agent: targets, task notes, a shared trick cookbook, and validation the agent cannot fool.

![The lever strip: of the five levers that decided Kaggle LLM competitions, NeuroGolf 2026 was decided by metric insight, with fleets of LLM coding agents as the labour](/imgs/blogs/kaggle-neurogolf-2026-solutions-1.webp)

NeuroGolf is the sequel to the [NeurIPS 2025 Google Code Golf Championship](/blog/machine-learning/learn-from-kaggle/kaggle-google-code-golf-2025-solutions), where you wrote the shortest Python program for each ARC task. This time the artefact is a neural network, and the "golf" is about how few parameters and how little intermediate memory it needs. Unlike [ARC Prize 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions) and [ARC Prize 2025](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2025-solutions), which ask whether a model can infer an unseen rule, NeuroGolf hands you the rule (the public training set) and asks how small a network can *be* that rule.

The surprise is in who did the work. The 1st-place team is literally called "Kaggle Agent", and its writeup lists the team as five humans plus "ChatGPT, Codex and Claude Code". That makes this the series' best case study in **agent operations**: organising hundreds of LLM coding sessions so they converge. The series opener on [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work) explains the lever strip above.

## 1. The task in 60 seconds

![The NeuroGolf budget card and an illustrative toy grid encoded as ten one-hot channel planes on a 30 by 30 canvas](/imgs/blogs/kaggle-neurogolf-2026-solutions-2.webp)

Each ARC task is a tiny puzzle: a few input grids of coloured cells, each with its output grid, and a hidden rule connecting them ("mirror the shape"). In NeuroGolf you submit `taskNNN.onnx`, a static computation graph that maps input grid to output grid.

Formally, the input is a one-hot tensor of shape `[1, 10, 30, 30]`: ten channels, one per colour, on a 30 by 30 canvas. A cell of colour 2 puts a 1 in channel 2 at that position. Cells outside the actual grid are "zero-hot": all ten channels are 0. The output has the same shape. Your network must reproduce every pair in the task's train and test examples, plus the ARC-GEN generated pairs, plus a small **private** example set per task that exists only to punish overfitting.

The rules are what make it a craft problem rather than a deep-learning problem:

- **Static shapes only.** No graph may depend on a runtime-sized tensor.
- **Banned operators:** `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`. No dynamic control flow.
- **File size:** each ONNX file at most 1.44 MB.
- **Runtime:** teams report that the whole bundle had to finish evaluation within 30 minutes, which became a real constraint late in the competition (more in section 4).
- **Board:** the leaderboard was 100% public, submissions were raised to 100 a day on April 21, and the competition logged 492,086 submissions from 3,469 competitors.

The prize pool was USD 50,000: 12,000 for 1st, 10,000 each for 2nd and 3rd, 8,000 for the top student team, and 10,000 for the "Longest Leader" (longest time at 1st between May 6 and July 15). I could not verify who won the last two, so this post does not name them.

## 2. The metric, with a worked toy score

Each task gets a cost, and the score is logarithmic in it:

$$
\text{score}_{\text{task}} = \max\bigl(1,\; 25 - \ln(\text{cost})\bigr), \qquad \text{cost} = \text{parameters} + \text{memory bytes}
$$

$$
\text{Total} = \sum_{t=1}^{400} \text{score}_t \quad\le\quad 400 \times 25 = 10{,}000
$$

"Parameters" counts the elements of the stored weights and constants. "Memory" is the footprint of intermediate tensors in bytes; the final output is not charged, which matters a lot. Multiply-accumulates were part of the cost early on but are free after the May 6 reset. A network costing 1 or less (a parameter-free graph can cost 0) is capped at the perfect 25. A failing network scores 0.

![The NeuroGolf task score against cost on a log axis: a straight line from 25 at cost 1 down to the floor of 1, with real tasks from the writeups marked along it](/imgs/blogs/kaggle-neurogolf-2026-solutions-3.webp)

Now a worked example. Take an illustrative recolouring task, "every cell of colour 3 becomes colour 5". Here are three hypothetical designs, priced with the definition above.

```python
# Illustrative toy, not any team's code: the NeuroGolf score and three
# hypothetical designs for a "recolour colour 3 to colour 5" task.
import math

def task_score(params, memory_bytes):
    cost = params + memory_bytes
    if cost <= 1:
        return 25.0                     # capped at a perfect 25
    return max(1.0, 25.0 - math.log(cost))

designs = {
    # 1x1 Conv: 10x10 weights + 10 biases, writes straight to the output
    "conv 1x1 to output":           (110, 0),
    # same Conv, but via one fp32 [1,10,30,30] intermediate (9,000 x 4 bytes)
    "conv 1x1 + fp32 intermediate": (110, 9000 * 4),
    # channel Gather with a 10-entry index table, straight to the output
    "gather 10 indices to output":  (10, 0),
}
for name, (p, m) in designs.items():
    print(f"{name:30s} cost={p + m:6d}  score={task_score(p, m):6.3f}")

def max_cost_for_gain(current_cost, gain):
    """Largest cost that still earns `gain` more points."""
    return current_cost * math.exp(-gain)

for gain in (0.1, 0.5, 1.0):
    print(f"+{gain}: cost must fall from 4839 to <= {max_cost_for_gain(4839, gain):.0f}")
```

```console
conv 1x1 to output             cost=   110  score=20.300
conv 1x1 + fp32 intermediate   cost= 36110  score=14.506
gather 10 indices to output    cost=    10  score=22.697
+0.1: cost must fall from 4839 to <= 4379
+0.5: cost must fall from 4839 to <= 2935
+1.0: cost must fall from 4839 to <= 1780
```

Three things fall out of those numbers, and every top team's strategy follows from them.

**Memory dominates parameters.** One full-size fp32 intermediate costs 36,000 bytes, over 300 times the convolution's weights, and drops the score by about 5.8 points. Hence "write directly to the output, which is free" in nearly every writeup.

**Every factor of e is one point.** The 2nd-place README states it plainly: each factor of e (about 2.718) cut from the cost adds one point. A design ten times cheaper is worth $\ln 10 \approx 2.30$ points, whatever the starting cost.

**Small edits are worth almost nothing.** To gain 0.1 points you must cut cost by about 9.5%. To gain a full point you must cut it to 36.8% of where it is. KOKOROMI (14th) turned this into their prompt: "+0.1 needs cost at most 90.5%, +0.5 at most 60.7%, +1.0 at most 36.8%", and they noted that asking for +1.0 forces the model to change representation, because no amount of pruning gets there.

One sanity check makes the leaderboard concrete. Kaggle Agent's 8314.03 is an average of 20.785 per task, so ${25 - 20.785 = 4.215}$ and $e^{4.215} \approx 68$.: the winner's networks have a geometric-mean cost of about 68. The 20th-place 7956.78 implies about 165, so the 357.25-point top-20 spread is a factor of about 2.44 in typical network size.

## 3. What made it hard

![What one point costs: the share of cost you must keep to gain 0.1 to 3.161 points, and the 1st-place measurement that rewrites earned ten times more per task than optimisation](/imgs/blogs/kaggle-neurogolf-2026-solutions-4.webp)

Three constraints decided this competition, and none of them is "train a better model".

**The logarithm punishes incrementalism.** Kaggle Agent mined their per-task score histories: architectural rewrites averaged **+0.5 per task**, optimising an existing graph **+0.05**. An agent left to "improve this task for two hours" pruned for +0.05 to +0.2 per round, plateaued, and a later rewrite threw that pruning away.

**Agents give up, convincingly.** Five writeups (1st, 6th, 7th, 17th, 19th) report models declaring a task "at its floor" or "already optimal" when it was not. kluges clueless scrubbed thousands of such phrases from their logs so later sessions would not inherit the pessimism.

**Local clean is not hidden clean.** A network could pass every local test and still score 0. kluges clueless measured it: a batch of 100 candidates with zero errors on full local ARC-GEN validation lost about **20% of its projected points** to hidden overfits. The grader also had sharp edges: a convolution bias shorter than its output channels read nondeterministic memory (4th, 5th and 7th all hit it), and some huge single-`Einsum` graphs cost almost nothing on paper but took hours to evaluate.

Then there was the metric itself. On April 21 the hosts patched a **negative-memory exploit**: calibrator chained two profiler bugs (a negative `Pad` shape times the memory-size product) so every task scored 25. Agents found more: Kaggle Agent reported a "LB 9550 potential scoring loophole" and Chris Deotte an "LB 10000, convert any task into score 25", opening "My agent just discovered...". Both were fixed, and the May 6 reset is the board that counted.

## 4. The top-20 scorecard

The leaderboard was 100% public, so public rank equals final rank for every team and there is no public-to-private move to report. One warning before the table: several teams were removed after the deadline (section 8), so most writeup titles carry a rank one or two places higher than the final board. claudex's writeup says "6th" and the API says 5th; robga's says "21st" and the API says 19th. This post uses the API ranks throughout.

| Rank | Team | Score | Public-rank move | Key idea | Source |
|---|---|---|---|---|---|
| 1 | Kaggle Agent | 8314.03 | none | Agent fleets, concrete targets, task notes, cookbooks | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/1st-place-kaggle-agent) |
| 2 | neurogolf team | 8243.89 | none | Five independent agent pipelines | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/work-in-progress-2nd-place-neurogolf-team) |
| 3 | The Final Ensemble | 8221.27 | none | Attempt ledgers, forced rewrites, consensus validation | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/3rd-place-solution-writeup) |
| 4 | M & H & M & N & M | 8142.02 | none | Task registry, voted idioms, probe bisection | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/4th-place-solution-writeup) |
| 5 | claudex | 8118.97 | none | Sub-teams swapping 200-task halves | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/6th-place-solution-claudex) |
| 6 | Slow and Steady | 8067.63 | none | ChatGPT Web plus a 68-rule rulebook | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/7th-place-solution-slow-and-steady) |
| 7 | kluges clueless | 8048.16 | none | ~330 ChatGPT chats, git as blackboard | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-writeup-draft) |
| 8 | Pavel & CroDoc | 8038.62 | none | Same tasks, two prompts; MEMORY.md | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-gold-pavel-and-crodoc) |
| 9 | Timeout exceeded : ( | 8028.25 | none | Solo; transactional Codex scheduler | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/9th-place-solution) |
| 10 | Kimura@SeSDA & OverfitOracle | 8025.82 | none | Score-discrepancy bisection (stub) | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/11th-place-solution) |
| 11 | Carlos, Egor, Rubempre & Bill | 8022.05 | none | unknown | no writeup |
| 12 | Sol | 8015.02 | none | unknown (solo) | no writeup |
| 13 | Santa Neuro Club 10000 | 8013.53 | none | unknown (largely the 9th-place Code Golf 2025 team) | no writeup |
| 14 | KOKOROMI | 8012.12 | none | Lessons rewritten as constraints | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/14th-place-writeup) |
| 15 | carbonapi | 8000.10 | none | unknown (solo) | no writeup |
| 16 | nmdcddwzml | 7998.91 | none | unknown | no writeup |
| 17 | Fususu | 7994.76 | none | Solo, 21 days, 64 agents | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/i-deleted-my-entire-project-6-days-before-the-dead) |
| 18 | Token is my sugar daddy! | 7960.75 | none | unknown | no writeup |
| 19 | robga | 7958.58 | none | Solo; representation first | [writeup](https://www.kaggle.com/competitions/neurogolf-2026/writeups/21st-place-solution-representation-collapse) |
| 20 | Just A game on your lips | 7956.78 | none | unknown | no writeup |

![The final top-20 board: Kaggle Agent 70 points clear, a 79-point step between 3rd and 4th, and ranks 4 to 20 packed into about 185 points, with writeup teams marked](/imgs/blogs/kaggle-neurogolf-2026-solutions-5.webp)

Fourteen of twenty teams published something. Rank 10's is a stub, so 13 writeups carry enough method detail to analyse. The shape of the board says a lot: 1st is 70.14 points clear of 2nd, there is a 79.25-point step between 3rd and 4th, and everything from 4th to 10th fits in 116.20 points.

## 5. The top-20 landscape

![Top-20 technique matrix: 13 teams with writeups as columns, 13 techniques as rows; per-task notes kept outside the agent are the only row filled for all 13](/imgs/blogs/kaggle-neurogolf-2026-solutions-6.webp)

The matrix counts what the 13 teams with a usable writeup said they did; unknown cells stay unknown, and the seven teams with no writeup are a grey block.

**Table stakes:**

- **LLM coding agents did the building: 13 of 13.** Codex appears in 10 of 13 and ChatGPT web or Pro sessions in 10 of 13 (unknown for 4th, 10th and 17th). Claude Code is used by 6, explicitly not used by 3 (7th, 9th, 19th) and unknown for 4.
- **Per-task notes or attempt history kept outside the agent: 13 of 13.** This is the one universal row.
- **A shared trick store: 12 of 13** (unknown for the 10th-place stub). The names vary: cookbook, TECHNIQUES.md, idiom catalogue, operator library, rulebook, CURRENT.md, MEMORY.md, tricks.md.
- **A single `Einsum` as the dominant node: 9 of 13.** At 4th place, 224 of the 400 models are one node and 193 of those are a single `Einsum`; at 8th, 273 models end in `Einsum`. `Einsum` can contract the one-hot input against tiny tensors and write straight to the free output, which is exactly what the metric rewards.
- **Validation against freshly generated ARC-GEN cases: 8 of 13**, and runtime-aware selection: 8 of 13.

**What separated the top 3 from 4 to 20.** The infrastructure from 4th to 9th is as elaborate (see 7th's 330-chat factory). The top 3 add *diversity* plus discipline: all are five-person teams running several distinct lanes, 1st and 3rd set explicit targets and forced rewrites, and 3rd built a hidden-test proxy nobody else matched. Numeric targets appear in 7 of 13 writeups, forced-rewrite prompts in 9 of 13.

**Contradictions worth knowing.** The writeups disagree on four things:

1. **ChatGPT Pro versus Codex.** Yiheng (9th) found ChatGPT Pro "consistently outperformed" Codex xhigh for architecture search; Slow and Steady (6th) regret relying on ChatGPT Web; robga (19th) moved to Codex once GPT 5.6 Sol Ultra arrived.
2. **Autonomy.** Pavel (8th) ran an autonomous Codex pipeline to low gold, and Yiheng gained +59.77 in a five-day unattended run on vacation; KOKOROMI's (14th) autonomous Claude plus Codex harness made under 10 points a day, and Fususu's models quit without coaching.
3. **From scratch versus incumbent.** 1st and 3rd force rewrites. CroDoc (8th) found from-scratch re-solves mostly rebuilt old baselines, and asking for a major rewrite *while showing the current model and notes* worked better. KOKOROMI warns that pasting verbatim failure history anchors the model on old ideas, while kluges and claudex feed failures back as warnings.
4. **Search engines.** The 2nd-place team says brute-force search did not pay. kluges clueless (7th) found its algorithmic engines plateau at about +1.5 points but useful as seeds for stuck agents.

The reconciliation: the winners showed the agent the current best model, the target, and a *summary* of what failed, then demanded a different representation.

## 6. The winning solution, explained: Kaggle Agent

![Kaggle Agent's trick-mining loop: single-task exploration feeds trace analysis, which updates a shared cookbook, which drives multi-task exploitation across clustered tasks](/imgs/blogs/kaggle-neurogolf-2026-solutions-7.webp)

Kaggle Agent's agents ran Claude Code (Opus 4.8 and Fable 5), Codex, and ChatGPT Pro (5.6-Sol and 5.5). The method is spread over three posts: Chris's intro, Eduardo's pipeline post, and Chris's post on prompts.

**The intuition.** Treat each agent session as a cheap, forgetful contractor: good at one task for one sitting, bad at remembering anything afterwards. Everything that should persist lives in files, and the humans' job is to write those files and choose what each contractor sees.

**Task notes.** Before any agent touched a task, the team pre-baked a note for it: number of pairs, grid shapes, palettes, symmetry probes, the baseline's cost, ops and initialisers, the task's archetype and its family's median cost, and a drafted rule with a confidence. Each worker appends what it tried and its dead ends. Agents also dump **builder scripts**, the Python that generates the ONNX, so a graph can be regenerated and edited rather than reverse-engineered.

**Targets, not time budgets.** This is the change Eduardo credits with saving the most tokens. Open-ended "improve this task for two hours" prompts produced the +0.05 to +0.2 per round plateau from section 3. They replaced them with concrete, "headstrong" targets: "improve this task by at least 1.5 points! It is possible and has been achieved by another competitor". Faced with a target that pruning cannot reach, agents first derived a theoretical maximum and only then started editing.

**The cookbook.** The team calls it the central pillar. Every new trick goes into a shared cookbook given to every worker. A trick found on one task is worthless until it reaches the other tasks where it applies.

**Two pipelines.** The vertical pipeline runs single-task, single-turn workers: `codex exec` spawned by a script, and ChatGPT Pro chats with a fixed template plus a task bundle (target, cookbook, prior notes, and human ARC solutions). The horizontal pipeline runs multi-task sessions that push one new trick across similar tasks, guided by a **task clustering report**. The loop is: explore single tasks, analyse the traces to mine tricks, update the cookbook, exploit across clusters, repeat.

**Self-evolving prompts.** Chris's post adds a tournament on top. Several agents run four-hour "mini contests" scored on "the most total onnx LB score gain". Humans mentor mid-run and agents self-review. After three rounds, the best-performing agent is promoted to manager of the others and writes the new `cookbook.md`. The asset being selected is the agent's *context*, the accumulated understanding in its session, not its weights.

**Validation.** Locally they checked the given pairs plus ARC-GEN pairs, and for colour-agnostic tasks sometimes random inputs and colour permutations. Kaggle Agent also published an agent-written "honest audit" of their own 9538 submission before the rules changed, examining dynamic-shape and `Constant`-parameter loopholes. That is a good habit to steal: make the agent argue against its own score.

None of this is specific to ONNX. Per-task notes and a shared trick store are the episodic and procedural memory of the [agent memory taxonomy](/blog/machine-learning/ai-agent/agent-memory-taxonomy), and a target the agent must justify before editing is a form of [agent output validation](/blog/machine-learning/ai-agent/agent-output-validation).

## 7. 2nd and 3rd in depth, then the rest of the top 20

### 2nd: neurogolf team (8243.89)

jacekwl, Yuichiro Hirano, Chan Kha Vu, Geremie Yeo and Jin Niu ran **independent pipelines on purpose**, as protection against everyone settling into the same local optimum, and shared candidates through a private leaderboard Space.

Jacek's pipeline used Claude Code to drive ChatGPT 5.5 Pro Extended, which solved harder tasks from scratch, with Codex for verification, risk checks and transfers. He started from Code Golf 2025 knowledge, then **removed the golf solutions as references**, because short code is not a small graph. His TECHNIQUES.md grew to about 50,000 lines. Two of his tricks show the flavour. One uses fp32 **subnormal underflow** as a counting threshold inside a single `Einsum`. Another uses ONNX's `RNN` operator with ReLU as a static recurrence, because `Loop` and `Scan` are banned (task 138 went from cost 2505 to 2300, about +0.085 points).

Geremie found Claude Code "gives up way too easily", moved to Codex 5.5 xhigh, then built a **Claude Code skill that drives ChatGPT in the browser** and downloads the ONNX. His prompts force a zero-memory `Einsum`, TRICKS.md entries auto-delete after 14 days unused, and graphs that hit `Einsum`'s 52-variable limit are split in two around a tiny intermediate. The repo README describes five lanes: cost-golf, compress-einsum, convert-to-0-memory-einsum, runtime, and a daily bundle.

### 3rd: The Final Ensemble (8221.27)

Dieter (Christof Henkel), matheus, Fritz Cremer, Diana and Tony Li used **Codex** as the main optimiser, one task per worker, plus the **ChatGPT UI** with a fresh chat per task inside a Project that held the scorer rules, technique libraries, a corpus and "negative knowledge". In the last week about 70% of gains came from Codex and 30% from the UI.

Each run got a warm-start bundle: the task data, its ARC-GEN generator, the RE-ARC verifier, the 2025 Code Golf solution, the current ONNX with its builder and a per-tensor cost view, an exact cost target, previous promotions, rejected branches, and similar solved tasks. Before optimising anything, the agent had to **price three rewrites**: a direct implementation, a finite certificate, and a design that re-reads the input at the output. Each had to end in a prototype, a price, or a failing case.

History came in four kinds: a per-task JSONL history, an attempt ledger (accepted and rejected, with reasons), a retry state (do-not-repeat, leads, forced rewrites), and `negative_knowledge.md`. Their headline example is task 349, cut from cost 4,839 to 205 (16.516 to 19.677, +3.161) by combining ideas from five other tasks. The same technique moved task 367 by +3.094 with one prompt.

![Two validation patterns: 3rd place keeps only generated examples all ten top Code Golf programs agree on, and grouped leaderboard probes bisect a bundle to find the one hidden-broken network](/imgs/blogs/kaggle-neurogolf-2026-solutions-8.webp)

Their most original idea is **consensus validation**. Sample new ARC-GEN examples, run the **top-10 Code Golf 2025 programs** on them, and keep only the examples all ten reproduce. The result is a 2,000-example proxy for the hidden test, built from ten independent solvers rather than one generator you might have misread.

### Ranks 4 to 20, by approach family

**Registries and statistics over the agents (4th, 10th).** M & H & M & N & M (8142.02) never names its LLM. It ran an asynchronous loop per task with bottleneck-targeted hints (rethink after a plateau, repair after a failure, optimise runtime when slow). A **task registry** keeps every model version by hash with its score, eval time and leaderboard status. Agents vote on an **idiom catalogue**; misleading idioms get disabled, and the "ULP-collision finite-difference selector" carried 89 measured confirmations against 8 failures. Final assembly was an approximate knapsack, maximising score under total eval time, and nine tasks hit the perfect 25 (task 053 is one parameter-free `LpPool` with negative padding acting as a row gather). Kimura@SeSDA & OverfitOracle (10th, 8025.82) posted only a stub: memory-based reconstruction, plus bisecting score discrepancies of 200, 100 and 50 points and having Claude search subset combinations below that.

**Two lanes that swap (5th, 8th).** claudex (8118.97) split into two sub-teams of 200 tasks each and **swapped halves every one to two days** to escape local maxima, still gaining 10+ points in the final minutes. Codex prompts carried a `target_base` of +0.1 to +0.5 and a **score-band operator library** suggesting operator families common at the task's score band. Pavel & CroDoc (8038.62) did the opposite split: both worked the *same* tasks with different prompts and merged by verified score. CroDoc kept a two-tier memory (`MEMORY.md` of about 200 lines loaded every session, plus a searchable archive). Their favourite prompt line: "Other people achieved [current]+1.0. Achieve it as well." Runtime cost them: their best local set, 8041.20, missed the time limit.

**ChatGPT Web factories (6th, 7th, 14th).** Slow and Steady (8067.63) sent each chat a prompt plus a seven-file zip: the generator, a mass-simulation helper, task JSON, the current ONNX, scorer utilities, 163 usable ops with caveats, and M Sato's rulebook (68 rules by the end). Rule R36 (zero intermediates) took task 110 from cost 22,871 to 110. kluges clueless (8048.16, writeup covers two of five members) ran about 330 concurrent chats on 1 Pro and 10 Plus accounts for about USD 300 a month, with git as the blackboard. Their most sobering number: LLM-claimed gains realised at about **29%**. KOKOROMI (8012.12) joined 40 days before the end, abandoned an autonomous harness for one task per ChatGPT Web chat, and **rewrote lessons as constraints** in the task note instead of pasting logs.

**Solo operators (9th, 17th, 19th).** Yiheng Wang (8028.25) held 1st for about 15 days after the reset. His unit of work is three files (`taskNNN.onnx`, a builder, a markdown memory) promoted or restored together, under a multi-turn Codex scheduler that only promotes strictly cheaper models. Fususu (7994.76) joined with 21 days left, ran up to 64 concurrent agents, and wrote rules for 250+ tasks "in seven-year-old language" after finding that explaining a task "like I was seven" cracked one an agent had declared optimal. robga (7958.58) let ChatGPT Pro reason only about representations and had Codex lower them to ONNX, and found that "diversity of representation mattered more than tuning one representation".

**No writeup (11, 12, 13, 15, 16, 18, 20).** No method details found. Two forum footnotes: Santa Neuro Club 10000 overlaps the 9th-place Code Golf 2025 team, and Token is my sugar daddy! includes calibrator, who found the negative-memory exploit.

**The best contrasting entry** is kluges clueless, because it measured what everyone else assumed: its 29% realisation rate and 20% hidden-overfit loss are the most honest numbers here on what an LLM's own claim is worth.

## 8. What the forum taught everyone

Tony Li showed that ChatGPT Plus at USD 20 a month, in many tabs, reached about 7600 with no API spend, the economics kluges clueless later industrialised. Jiwei Liu open-sourced NVIDIA's Kaggle agent skills, and Fritz Cremer's per-bucket score totals became Yiheng's targets.

Five teams used probe bisection. A broken network scores exactly 0, so a group's deficit (local claim minus leaderboard score) shows whether it hides one, and since probes are additive you submit only one half:

```python
# Illustrative toy, not any team's code: find hidden-broken models by
# grouped probes. A broken model scores exactly 0 on the hidden set, so a
# group's deficit (local claim minus leaderboard score) reveals it.
import random

random.seed(7)
tasks = list(range(64))
broken = set(random.sample(tasks, 2))         # unknown to us
claim = {t: 20.0 for t in tasks}              # local score of each model
probes = 0

def probe(group):                             # costs one submission
    global probes
    probes += 1
    lb = sum(0.0 if t in broken else claim[t] for t in group)
    return sum(claim[t] for t in group) - lb  # the deficit

def find(group, deficit):
    if deficit == 0:
        return []
    if len(group) == 1:
        return group
    mid = len(group) // 2
    left = probe(group[:mid])                 # probe one half...
    right = deficit - left                    # ...infer the other (additive)
    return find(group[:mid], left) + find(group[mid:], right)

found = find(tasks, probe(tasks))
print("broken:", sorted(broken), "found:", sorted(found), "probes:", probes)
```

```console
broken: [19, 41] found: [19, 41] probes: 12
```

Twelve submissions instead of 64 singletons. kluges clueless add a warning: naive subset-sum decoding of deficits fails when several subsets explain the same number (one ambiguous bank regressed by 74.79).

**No shake-up, but the board moved,** for three reasons. **Removals after the deadline:** robga's list names teams removed at 4th, 19th, 24th, 27th and lower. One investigation found 74 bit-identical graphs shared by two teams, traced to people sharing a server and a ChatGPT account; one member of each was banned, and the evidence is still disputed. **Post-deadline scoring:** a submission made before the deadline finished scoring about 28 minutes later and displaced nmdcddwzml, who had already received a 16th-place email; Kaggle confirmed it valid. **Leaks:** a 7300+ and then a claimed 8100 public release in the last weeks triggered calls to freeze the board, and a dataset claiming a 7957 solution turned out to be the public baseline, encrypted.

## 9. What did not work

Aggregated across the top-20 writeups, with who reported it:

- **Trusting local validation:** about 20% of projected points lost to hidden overfits (7th).
- **Huge multi-operand Einsums.** Cheap on paper, but they time out or crash. An LLM-claimed +12.49 round delivered +6.49 (7th); slow Einsums forced knapsack assembly (4th) and cost 8th its best set.
- **Generator ops to save parameters.** `ConstantOfShape`, `EyeLike` and `Range` are charged as memory (7th).
- **Negative-pad crops on the wrong ops.** They work on `Conv`, `LpPool` and `ConvInteger` but crash the grader on `MaxPool`, `AveragePool` and `Crop` (7th).
- **Dynamic-dimension intermediates** look phantom-cheap locally because the scorer returns no memory figure (7th); **short conv biases** give flaky scores (4th, 5th, 7th); **ORT-GPU evaluation** could not run about 30% of graphs (7th).
- **Autonomous harnesses without targets** (14th), accepting "floor" claims (1st, 6th, 7th, 17th, 19th), from-scratch re-solves (8th), pasting full failure logs (14th), Pro sessions ending in "Thinking Failed" (2nd, 6th), and tuning one representation instead of several (19th, in hindsight).
- **Exact-cancellation numerics across CPUs.** Some `Einsum` solutions produce exactly 0 on Kaggle's x86 but residues around 1e-11 on ARM, so they pass the leaderboard and fail on a Mac (19th).

## 10. Lessons that transfer

1. **Give agents targets, not time budgets** (1st; 7 of 13 teams). With a logarithmic metric, make the target big enough that only a new representation can hit it.
2. **Keep memory outside the agent.** Task notes, attempt ledgers, negative knowledge, cookbooks: 13 of 13 teams kept per-task history and 12 of 13 a shared trick store. The session is disposable; the files are the asset.
3. **Summarise failures as constraints.** KOKOROMI's rewrite of logs into constraints and CroDoc's "show the current model, demand a rewrite" are the same idea: give the agent the lesson, not the transcript.
4. **Transfer every win.** Task 349's +3.161 came from combining five other tasks' ideas and transferred to task 367 with one prompt (3rd); Kaggle Agent built a whole horizontal pipeline for this.
5. **Build a proxy the agent cannot fool:** ten-solver consensus (3rd), fresh generator samples (5th), probe bisection (5 of 13).
6. **Discount agent claims.** kluges clueless realised about 29% of LLM-claimed gains. Measure the gap in your own setup before you plan around claimed numbers.

### If you enter a competition like this

- Work out what one point costs in the metric before writing any prompt, and put that arithmetic in the prompt.
- Create a per-task note file and a shared trick file on day one; make every agent read and append.
- Run at least two independent lanes and merge by verified score.
- Validate on freshly generated cases, keep a risk pool, and budget submissions for bisection probes.
- Track runtime from the start; 8th place lost its best set to it.
- Ban "this is optimal" as an answer: require a theoretical floor with an argument, or a rewrite.
- Ask your agent to audit its own score for loopholes, and report what it finds to the host.

For the human side of agent fleets, see [running AI in a team](/blog/software-development/ai-assisted-development/running-ai-in-a-team). Related in this wave: the [Konwinski Prize](/blog/machine-learning/learn-from-kaggle/kaggle-konwinski-prize-solutions) and the [NVIDIA Nemotron Reasoning Challenge](/blog/machine-learning/learn-from-kaggle/kaggle-nvidia-nemotron-reasoning-challenge-solutions). Coming later: the ARC Prize 2026 post.

## Sources

- Competition page: https://www.kaggle.com/competitions/neurogolf-2026
- 1st, Kaggle Agent: https://www.kaggle.com/competitions/neurogolf-2026/writeups/1st-place-kaggle-agent ; pipeline https://www.kaggle.com/competitions/neurogolf-2026/discussion/726799 ; prompts https://www.kaggle.com/competitions/neurogolf-2026/discussion/726883 ; dashboard https://daxiongshu.github.io/kaggle-agent-neurogolf-2026/
- 2nd, neurogolf team: https://www.kaggle.com/competitions/neurogolf-2026/writeups/work-in-progress-2nd-place-neurogolf-team ; repo https://github.com/bogoconic1/neurogolf-2026-2nd-place-solution
- 3rd, The Final Ensemble: https://www.kaggle.com/competitions/neurogolf-2026/writeups/3rd-place-solution-writeup
- 4th, M & H & M & N & M: https://www.kaggle.com/competitions/neurogolf-2026/writeups/4th-place-solution-writeup
- 5th, claudex: https://www.kaggle.com/competitions/neurogolf-2026/writeups/6th-place-solution-claudex ; repo https://github.com/PNTAN17/neurogolf
- 6th, Slow and Steady: https://www.kaggle.com/competitions/neurogolf-2026/writeups/7th-place-solution-slow-and-steady
- 7th, kluges clueless: https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-writeup-draft
- 8th, Pavel & CroDoc: https://www.kaggle.com/competitions/neurogolf-2026/writeups/8th-place-gold-pavel-and-crodoc ; repos https://github.com/CroDoc/neurogolf-2026 , https://github.com/PavelSavchenkov/kaggle-golf-8th-place-neurogolf-2026
- 9th, Timeout exceeded : (: https://www.kaggle.com/competitions/neurogolf-2026/writeups/9th-place-solution
- 10th, Kimura@SeSDA & OverfitOracle: https://www.kaggle.com/competitions/neurogolf-2026/writeups/11th-place-solution
- 14th, KOKOROMI: https://www.kaggle.com/competitions/neurogolf-2026/writeups/14th-place-writeup
- 17th, Fususu: https://www.kaggle.com/competitions/neurogolf-2026/writeups/i-deleted-my-entire-project-6-days-before-the-dead
- 19th, robga: https://www.kaggle.com/competitions/neurogolf-2026/writeups/21st-place-solution-representation-collapse
- Metric update and negative-memory exploit: https://www.kaggle.com/competitions/neurogolf-2026/discussion/693711 , https://www.kaggle.com/competitions/neurogolf-2026/discussion/693247
- Agent-found loopholes: https://www.kaggle.com/competitions/neurogolf-2026/discussion/696365 , https://www.kaggle.com/competitions/neurogolf-2026/discussion/696377 ; honest audit https://www.kaggle.com/competitions/neurogolf-2026/discussion/694772
- Forum tooling: https://www.kaggle.com/competitions/neurogolf-2026/discussion/704942 , https://www.kaggle.com/competitions/neurogolf-2026/discussion/717854 , https://www.kaggle.com/competitions/neurogolf-2026/discussion/708377
- Integrity and board movement: https://www.kaggle.com/competitions/neurogolf-2026/discussion/729721 , https://www.kaggle.com/competitions/neurogolf-2026/discussion/727985 , https://www.kaggle.com/competitions/neurogolf-2026/discussion/726658 , https://www.kaggle.com/competitions/neurogolf-2026/discussion/724778 , https://www.kaggle.com/competitions/neurogolf-2026/discussion/726541
