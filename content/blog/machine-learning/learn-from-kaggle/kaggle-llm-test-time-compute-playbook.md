---
title: "The test-time compute playbook from Kaggle LLM competitions: sample, vote, stop, and train on the test"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "What eight Kaggle LLM competitions taught about spending a fixed notebook clock at inference time: sampling, voting, reward-model reranking, tool-integrated reasoning, early stopping, time management, test-time training and augmentation voting, each traced to the team that shipped it."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "test-time-compute", "self-consistency", "test-time-training", "aimo", "arc-prize", "inference"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 13
---

> **TL;DR.** Test-time compute on Kaggle came in two families. **Sample and select** (AIMO 1 to 3, Drawing with LLMs): many candidates, code inside the reasoning, a vote or reranker, early stop, saved seconds moved to harder items. **Train on the test** (ARC 2024 and 2025, Jigsaw): the hidden file carries its own labelled examples, so winners fine-tuned on them in the notebook, then voted across augmented views. The lesson: the vote is almost free, the selector holds the headroom, and the clock decides which selector you can afford.

NemoSkills won [AIMO Progress Prize 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions) with 34 of 50 and left its best inference idea on the shelf. Their trained answer selector, GenSelect, lifted their 14B model on a held-out math set from 76.3 to 86.7, but it needed a second model in memory and extra minutes per problem, and more samples were the cheaper bet on a five-hour clock. That decision is the subject of this post: you are not choosing the best technique, you are choosing the best technique *per second of notebook*.

![Cross-competition matrix: eight competitions as rows, eight test-time techniques as columns; AIMO and Drawing rows fill the sampling and selection columns, ARC and Jigsaw rows fill test-time training and augmentation voting, and Nemotron's row is locked by the host](/imgs/blogs/kaggle-llm-test-time-compute-playbook-1.webp)

The matrix above is the mental model. A filled cell means the winner shipped the technique, a half cell means another top-20 team did; every fact comes from the published competition posts. Math and drawing sample and select, grid puzzles and the rules classifier train on the test, and Nemotron's row is all locks because the host ran the decoder.

## 1. The clock is the budget

Every technique in this post spends the same currency: wall-clock seconds per hidden item, on hardware you did not choose, with the internet off.

![Minutes of wall clock per hidden test item on a log scale: AIMO 1 at 10.8, ARC 2024 at 7.2, AIMO 2, AIMO 3 and ARC 2025 at 6.0, Drawing with LLMs capped at 5, Jigsaw at 0.79 seconds per row, and Nemotron fixed by the host](/imgs/blogs/kaggle-llm-test-time-compute-playbook-2.webp)

Five generative competitions on four hardware generations all landed between 5 and 11 minutes per item. Jigsaw sits three orders of magnitude lower, which is why its test-time compute went into training: at 0.79 seconds a row you get one forward pass per model, not a vote.

| Competition | Notebook | Per item | Winner | Winning test-time recipe |
|---|---|---|---|---|
| [AIMO 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions) | 2x T4, 9 h | 10.8 min | Numina, 29/50 | 48 samples, 4 code rounds, vote |
| [AIMO 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions) | 4x L4, 5 h | 6.0 min | NemoSkills, 34/50 | 12 samples, early stop, time buffer |
| [AIMO 3](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions) | 1x H100, 5 h | 6.0 min | Exalted Joseph, 44.0 | 8 tool attempts, stop at 4 agreeing |
| [ARC 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions) | 2x T4, 12 h | 7.2 min | the ARChitects, 53.5 | Test-time LoRA, DFS, 8-view scoring |
| [ARC 2025](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2025-solutions) | 4x L4, 12 h | 6.0 min | NVARC, 24.03 | Per-puzzle LoRA, DFS, 8-view rescoring |
| [Jigsaw Rules](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions) | 2x T4, 12 h | 0.79 s | Guanshuo Xu, 0.92930 | Six LoRAs trained on the test's examples |
| [Drawing with LLMs](/blog/machine-learning/learn-from-kaggle/kaggle-drawing-with-llms-solutions) | 2x T4, 9 h | at most 5 min | new team name, 0.79058 | 64 candidates, keep the best-scored |
| [Nemotron](/blog/machine-learning/learn-from-kaggle/kaggle-nvidia-nemotron-reasoning-challenge-solutions) | host-run | 1 greedy sample | NullSira, 0.920 | None: compute moved into the trace |

## 2. Sample many, then vote

Self-consistency, which every math winner shipped, draws several independent solutions and takes the most frequent answer. Reasoning fails in many ways and succeeds in one, so wrong answers scatter while right ones pile up. The [test-time compute scaling post](/blog/machine-learning/scaling-laws/test-time-compute-scaling) covers the general curve; Kaggle adds the clock.

<figure class="blog-anim">
<svg viewBox="0 0 800 440" role="img" aria-label="Eight tool-using attempts run in parallel; as each finishes its answer drops into a vote bin; when the bin for 512 reaches 4 votes the question stops and the two unfinished attempts are cancelled" style="width:100%;height:auto;max-width:820px">
<style>
.t1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.t1-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.t1-big{font:700 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.t1-track{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1}
.t1-bar{fill:var(--text-secondary,#6b7280);opacity:.55;transform-box:fill-box;transform-origin:left center}
.t1-half{transform:scaleX(.5)}
.t1-third{transform:scaleX(.4)}
.t1-tok{fill:var(--accent,#6366f1)}
.t1-tokn{fill:var(--text-secondary,#6b7280);opacity:.7}
.t1-toktx{font:700 13px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.t1-bin{fill:none;stroke:var(--border,#d1d5db);stroke-width:1.5}
.t1-thr{stroke:var(--accent,#6366f1);stroke-width:2;stroke-dasharray:6 5}
.t1-ban{fill:var(--accent,#6366f1)}
.t1-bantx{font:700 15px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.t1-gone{fill:var(--border,#d1d5db)}
.t1-gonetx{font:600 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
@keyframes t1-g15{0%{transform:scaleX(0)}15%,100%{transform:scaleX(1)}}
@keyframes t1-k15{0%,14%{opacity:0;transform:translateY(-14px)}15%,100%{opacity:1;transform:translateY(0)}}
.t1-g15{animation:t1-g15 10s linear infinite}
.t1-k15{animation:t1-k15 10s ease-out infinite}
@keyframes t1-g25{0%{transform:scaleX(0)}25%,100%{transform:scaleX(1)}}
@keyframes t1-k25{0%,24%{opacity:0;transform:translateY(-14px)}25%,100%{opacity:1;transform:translateY(0)}}
.t1-g25{animation:t1-g25 10s linear infinite}
.t1-k25{animation:t1-k25 10s ease-out infinite}
@keyframes t1-g35{0%{transform:scaleX(0)}35%,100%{transform:scaleX(1)}}
@keyframes t1-k35{0%,34%{opacity:0;transform:translateY(-14px)}35%,100%{opacity:1;transform:translateY(0)}}
.t1-g35{animation:t1-g35 10s linear infinite}
.t1-k35{animation:t1-k35 10s ease-out infinite}
@keyframes t1-g45{0%{transform:scaleX(0)}45%,100%{transform:scaleX(1)}}
@keyframes t1-k45{0%,44%{opacity:0;transform:translateY(-14px)}45%,100%{opacity:1;transform:translateY(0)}}
.t1-g45{animation:t1-g45 10s linear infinite}
.t1-k45{animation:t1-k45 10s ease-out infinite}
@keyframes t1-g55{0%{transform:scaleX(0)}55%,100%{transform:scaleX(1)}}
@keyframes t1-k55{0%,54%{opacity:0;transform:translateY(-14px)}55%,100%{opacity:1;transform:translateY(0)}}
.t1-g55{animation:t1-g55 10s linear infinite}
.t1-k55{animation:t1-k55 10s ease-out infinite}
@keyframes t1-g65{0%{transform:scaleX(0)}65%,100%{transform:scaleX(1)}}
@keyframes t1-k65{0%,64%{opacity:0;transform:translateY(-14px)}65%,100%{opacity:1;transform:translateY(0)}}
.t1-g65{animation:t1-g65 10s linear infinite}
.t1-k65{animation:t1-k65 10s ease-out infinite}
@keyframes t1-gA4{0%{transform:scaleX(0)}65%,100%{transform:scaleX(.5)}}
@keyframes t1-gA6{0%{transform:scaleX(0)}65%,100%{transform:scaleX(.4)}}
.t1-gA4{animation:t1-gA4 10s linear infinite}
.t1-gA6{animation:t1-gA6 10s linear infinite}
@keyframes t1-fade{0%,92%{opacity:1}98%,100%{opacity:0}}
.t1-fade{animation:t1-fade 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.t1-fade,.t1-g15,.t1-g25,.t1-g35,.t1-g45,.t1-g55,.t1-g65,.t1-k15,.t1-k25,.t1-k35,.t1-k45,.t1-k55,.t1-k65,.t1-gA4,.t1-gA6{animation:none}}
</style>
<text class="t1-lbl" x="20" y="26">8 tool-using attempts</text>
<text class="t1-lbl" x="470" y="26">answer bins (votes)</text>
<g class="t1-fade">
<text class="t1-lbl" x="20" y="62">A1</text>
<rect class="t1-track" x="56" y="46" width="300" height="22" rx="4"/>
<rect class="t1-bar t1-g35" x="56" y="46" width="300" height="22" rx="4"/>
<text class="t1-lbl t1-k35" x="368" y="62">512</text>
<text class="t1-lbl" x="20" y="106">A2</text>
<rect class="t1-track" x="56" y="90" width="300" height="22" rx="4"/>
<rect class="t1-bar t1-g55" x="56" y="90" width="300" height="22" rx="4"/>
<text class="t1-lbl t1-k55" x="368" y="106">47</text>
<text class="t1-lbl" x="20" y="150">A3</text>
<rect class="t1-track" x="56" y="134" width="300" height="22" rx="4"/>
<rect class="t1-bar t1-g15" x="56" y="134" width="300" height="22" rx="4"/>
<text class="t1-lbl t1-k15" x="368" y="150">512</text>
<text class="t1-lbl" x="20" y="194">A4</text>
<rect class="t1-track" x="56" y="178" width="300" height="22" rx="4"/>
<rect class="t1-bar t1-half t1-gA4" x="56" y="178" width="300" height="22" rx="4"/>
<g class="t1-k65"><rect class="t1-gone" x="56" y="178" width="300" height="22" rx="4"/><text class="t1-gonetx" x="368" y="194">cancelled</text></g>
<text class="t1-lbl" x="20" y="238">A5</text>
<rect class="t1-track" x="56" y="222" width="300" height="22" rx="4"/>
<rect class="t1-bar t1-g45" x="56" y="222" width="300" height="22" rx="4"/>
<text class="t1-lbl t1-k45" x="368" y="238">512</text>
<text class="t1-lbl" x="20" y="282">A6</text>
<rect class="t1-track" x="56" y="266" width="300" height="22" rx="4"/>
<rect class="t1-bar t1-third t1-gA6" x="56" y="266" width="300" height="22" rx="4"/>
<g class="t1-k65"><rect class="t1-gone" x="56" y="266" width="300" height="22" rx="4"/><text class="t1-gonetx" x="368" y="282">cancelled</text></g>
<text class="t1-lbl" x="20" y="326">A7</text>
<rect class="t1-track" x="56" y="310" width="300" height="22" rx="4"/>
<rect class="t1-bar t1-g25" x="56" y="310" width="300" height="22" rx="4"/>
<text class="t1-lbl t1-k25" x="368" y="326">128</text>
<text class="t1-lbl" x="20" y="370">A8</text>
<rect class="t1-track" x="56" y="354" width="300" height="22" rx="4"/>
<rect class="t1-bar t1-g65" x="56" y="354" width="300" height="22" rx="4"/>
<text class="t1-lbl t1-k65" x="368" y="370">512</text>
<g class="t1-k15"><rect class="t1-tok" x="474" y="344" width="72" height="30" rx="5"/><text class="t1-toktx" x="510" y="364">A3</text></g>
<g class="t1-k25"><rect class="t1-tokn" x="594" y="344" width="72" height="30" rx="5"/><text class="t1-toktx" x="630" y="364">A7</text></g>
<g class="t1-k35"><rect class="t1-tok" x="474" y="308" width="72" height="30" rx="5"/><text class="t1-toktx" x="510" y="328">A1</text></g>
<g class="t1-k45"><rect class="t1-tok" x="474" y="272" width="72" height="30" rx="5"/><text class="t1-toktx" x="510" y="292">A5</text></g>
<g class="t1-k55"><rect class="t1-tokn" x="714" y="344" width="72" height="30" rx="5"/><text class="t1-toktx" x="750" y="364">A2</text></g>
<g class="t1-k65"><rect class="t1-tok" x="474" y="236" width="72" height="30" rx="5"/><text class="t1-toktx" x="510" y="256">A8</text></g>
</g>
<path class="t1-bin" d="M470 196 V378 H550 V196"/>
<text class="t1-big" x="510" y="400">512</text>
<path class="t1-bin" d="M590 196 V378 H670 V196"/>
<text class="t1-big" x="630" y="400">128</text>
<path class="t1-bin" d="M710 196 V378 H790 V196"/>
<text class="t1-big" x="750" y="400">47</text>
<line class="t1-thr" x1="460" y1="234" x2="796" y2="234"/>
<text class="t1-sub" x="470" y="226">4 votes: stop</text>
<g class="t1-fade"><g class="t1-k65"><rect class="t1-ban" x="470" y="50" width="320" height="40" rx="8"/><text class="t1-bantx" x="630" y="76">stop: 4 agree on 512</text></g></g>
<text class="t1-sub" x="20" y="430">bars grow as each attempt reasons and runs code; a finished attempt drops one vote</text>
</svg>
<figcaption>Illustrative: the AIMO 3 backbone rule, up to 8 attempts, stop at 4 agreeing answers.</figcaption>
</figure>

The animation shows the AIMO 3 backbone rule. Every one of the five AIMO 3 teams that published (ranks 1, 2, 3, 5 and 7) ran `gpt-oss-120b` with 8 tool-using attempts at temperature 1.0 and ended the question when 4 attempts agreed. Sample counts scaled with the hardware: in AIMO 1 Numina drew 48 on two T4s and after exams (3rd) drew 120 to 160 per problem in one vLLM batch, NemoSkills drew 12 in AIMO 2, and AIMO 3 settled on 8 for a 120B model.

More samples stopped paying early wherever it was measured. Numina found nothing beyond N=48; after exams saw more generations stop improving validation; in AIMO 3, SKobayak (3rd) found 16 solvers worse than 8 and varianceofx (2nd) found more than 8 broke the time budget. A vote needs enough samples to stabilise the mode and no more.

Diversity matters as much as count. In AIMO 2, imagination-research (2nd, 31/50) split 15 samples into 7 chain-of-thought and 8 code prompts so failures would not correlate. Aliev (3rd, 30/50, no training) branched 5 traces to 10 to 14 with [prefix caching](/blog/machine-learning/model-serving/prefix-caching-and-radixattention) making the shared prefix free: cheap votes, paid for in correlation.

Drawing with LLMs is the same idea outside math. The winner generated 64 SSD-1B images per prompt and kept the one with the highest aesthetic score; 9 of 16 sourced top-20 entries generated 8 or more candidates. With no discrete answer to vote on, best-of-n needs a scorer, which is the next column.

## 3. Smarter selection: weighted votes and reward models

Plain majority treats every sample as an equal vote. Every serious selection rule on these leaderboards was a way of saying "not all votes are equal".

![Dumbbell chart on NemoSkills' validation set: majority vote at 62.9 and 66.8 against pass at 16 of 76.2 and 80.1, a 13.3-point gap each; GenSelect moves a 14B model from 76.3 to 86.7 and a 32B model from 78.4 to 93.3, and was not shipped](/imgs/blogs/kaggle-llm-test-time-compute-playbook-3.webp)

NemoSkills measured maj@16 at 66.8 against pass@16 at 80.1 for their tool-integrated model: 13.3 points of problems where a correct trace existed and lost the vote. Three selection families shipped:

- **Penalties on suspicious answers.** In AIMO 1, after exams (3rd) down-weighted answers below 10 and numbers from the problem statement, worth one problem (20 private without the small-number penalty, 21 with it); codeinter (4th) down-weighted small integers. In AIMO 2, arek-paterek (20th) discounted programs sharing a long prefix and boosted code answers that agreed with a boxed one.
- **A learned reward model.** CMU_MATH (AIMO 1, 2nd) ran the policy on one T4 and a 7B outcome [reward model](/blog/machine-learning/reinforcement-learning/reward-modeling-from-human-preferences) on the other, combining scores per answer by geometric mean so one over-confident sample could not carry an answer. On 10 training problems: base 1, fine-tuned policy 2, policy plus reward model 4.
- **Confidence weighting.** In AIMO 3, 4 of 5 published teams weighted answers by inverse token entropy. varianceofx warned that a pure inverse-entropy sum can be hijacked by one confident wrong attempt, so vote share must dominate.

The toy below runs three rules over one invented set of attempts and gets three different answers. It is illustrative, not any team's code.

```python
# Illustrative toy of three selection rules seen on AIMO leaderboards. Not any team's code.
# Each attempt: (answer, mean token entropy). Lower entropy = the model was more sure.
from collections import defaultdict

ATTEMPTS = [(512, 0.60), (512, 0.70), (512, 0.80),   # three ordinary, agreeing attempts
            (64, 0.05),                              # one confident attempt, wrong
            (7, 0.50), (7, 0.55), (7, 0.60), (7, 0.65)]  # a number lifted from the statement
STATEMENT_NUMBERS = {7, 12}

def majority(attempts):
    votes = defaultdict(float)
    for ans, _ in attempts:
        votes[ans] += 1
    return max(votes, key=votes.get), dict(votes)

def inverse_entropy_sum(attempts):                   # pure confidence sum
    votes = defaultdict(float)
    for ans, h in attempts:
        votes[ans] += 1 / h
    return max(votes, key=votes.get), {a: round(v, 2) for a, v in votes.items()}

def share_first(attempts, penalty=0.25):             # vote share dominates, entropy breaks ties,
    votes = defaultdict(float)                       # suspicious answers are down-weighted
    for ans, h in attempts:
        w = penalty if (ans < 10 or ans in STATEMENT_NUMBERS) else 1.0
        votes[ans] += w + 0.01 * (1 / h)
    return max(votes, key=votes.get), {a: round(v, 2) for a, v in votes.items()}

for rule in (majority, inverse_entropy_sum, share_first):
    pick, scores = rule(ATTEMPTS)
    print(f"{rule.__name__:20s} -> {pick:4d}   {scores}")
```

```console
majority             ->    7   {512: 3.0, 64: 1.0, 7: 4.0}
inverse_entropy_sum  ->   64   {512: 4.35, 64: 20.0, 7: 7.02}
share_first          ->  512   {512: 3.04, 64: 1.2, 7: 1.07}
```

Majority picks the statement number, the entropy sum picks the confident wrong attempt (one vote at entropy 0.05 is worth 20), and share-first with a penalty picks 512.

Learned selectors mostly did not ship. NemoSkills' outcome reward model faded on long tool-integrated traces, and GenSelect lost to the clock. In ARC 2024, Guillermo Barbadillo (2nd) trained a verifier that beat voting offline and gained nothing on the leaderboard. In Drawing with LLMs, 11 of 16 sourced entries reranked with a PaliGemma yes/no question as a proxy for the judge; how the top five then attacked the judge belongs to the metric-exploits post in this track.

## 4. Tool-integrated reasoning: samples that check themselves

A vote only works if correct traces outnumber wrong ones. Running Python inside the reasoning raises that ratio more than any selector.

Numina's AIMO 1 run is the cleanest evidence. Same DeepSeekMath-7B base and hardware: chain of thought scored 8 public, one program per solution 16, and multi-step traces that write code, read the output and write more code 27, then 29 private. SC-TIR samples 48 copies, executes each code block, appends the output, repeats up to 4 rounds, prunes broken traces and votes.

Tools cost time, and AIMO 2 showed how to buy it back. NemoSkills' tool model averaged 2.73 code calls and 15,834 tokens per solution against 11,203 for chain of thought; a 0.3 CoT plus 0.7 TIR weight merge scored 69.1 maj@16, above both parents, at 12,489 tokens and 0.85 code calls. By AIMO 3 all five published teams ran a persistent Jupyter kernel per attempt, with the kernel pool overprovisioned two to one because a hung kernel kills an attempt.

## 5. Early stopping and spending the clock

Once votes are cheap, stop paying for votes that cannot change the outcome. In AIMO 2, imagination-research stopped each *sample* at its first executable code block or first `\boxed{}`, after measuring that wrong first answers were rarely revised. NemoSkills stopped each *question* when 4 of the first 5 generations agreed, or after 10 of 12 finished so one looping sample could not hold it hostage. farsail (11th) also stopped when it was hopeless, for example 6 distinct answers among 8. In AIMO 3, varianceofx stopped once the leader could not be caught.

Saved seconds matter only if something spends them. NemoSkills gave each question 350 seconds, pooled the unused time, and let a hard question draw up to 210 more, a ceiling of 560. The toy compares that rule to a flat split on eight invented questions.

![Eight toy questions as bars of seconds needed, a flat 360-second line and per-question buffer ceilings: the flat split solves 4 of 8, the buffer pool solves 7 of 8, and reversing the order drops the pool to 6 of 8](/imgs/blogs/kaggle-llm-test-time-compute-playbook-4.webp)

```python
# Illustrative toy: a fixed per-question split against a NemoSkills-style time buffer.
# Not any team's code. Eight questions arrive in a fixed order; each needs `need` seconds
# of generation before the vote converges. Numbers are invented.
NEED = [100, 520, 150, 610, 90, 480, 200, 400]

def fixed_split(need, per_q=360):                 # 5 h / 50 questions = 360 s each
    return [n <= per_q for n in need]

def buffer_policy(need, base=350, max_draw=210):  # NemoSkills: 350 s base, draw up to 210 s
    pool, solved, ceilings = 0, [], []
    for n in need:
        ceiling = base + min(max_draw, pool)      # never more than 560 s on one question
        used = min(n, ceiling)
        ceilings.append(ceiling)
        solved.append(n <= ceiling)
        pool += base - used                       # unused base time flows into the pool
    return solved, ceilings

fixed = fixed_split(NEED)
buf, ceil = buffer_policy(NEED)
print("ceilings :", ceil)
print("fixed    :", sum(fixed), "of", len(NEED), "solved")
print("buffer   :", sum(buf), "of", len(NEED), "solved")
print("reversed order, buffer:", sum(buffer_policy(NEED[::-1])[0]), "of", len(NEED))
```

```console
ceilings : [350, 560, 430, 560, 420, 560, 550, 560]
fixed    : 4 of 8 solved
buffer   : 7 of 8 solved
reversed order, buffer: 6 of 8
```

The pool solves 7 of 8 where the flat split solves 4. Reverse the order and it solves 6, because hard questions arrive before easy ones have banked time. That is why sravn (AIMO 2, 4th, 29 private) refused time management: questions arrive in random order, so a policy tuned on one ordering is partly fitted to it. Numina made the same call in AIMO 1, fixing N and M because time-budget tricks gave high variance, while codeinter (4th) built an adaptive TimeManager and also reached the money. Both camps can be right. Gentler schemes exist too: imagination-research dropped to 10 samples when average remaining time fell below five minutes per question, and WXYZ123 (Drawing, 19th) banked time with early stopping.

## 6. Test-time training: when the test set carries its own training data

ARC and Jigsaw had a property the math competitions did not: each hidden item arrives with labelled examples of its own rule. That turns inference into a small training job.

![Two panels: an ARC task's three demonstration pairs become 24 leave-one-out training episodes across eight symmetry views and a per-task LoRA; a Jigsaw test row's two positive and two negative example comments peel off into four labelled training rows](/imgs/blogs/kaggle-llm-test-time-compute-playbook-6.webp)

In ARC 2024, the ARChitects (53.5) trained a rank-64 LoRA on the hidden tasks' augmented demonstrations inside the notebook. Barbadillo (2nd, 40.00) trained one model per test task, leave-one-out over the demonstrations, about 300 steps each, and that change alone took one model from 11 to 33 solved tasks. Nikola Hu (13th) reached 30 with a 162M model trained from scratch plus test-time training, and William Wu (4th) placed with no LLM yet concluded test-time training is the state of the art.

By ARC 2025 it was universal: 9 of 9 documented scored entries trained per task. NVARC (24.03) fitted a rank-256 LoRA per puzzle on its own Qwen3-4B. MindsAI and Tufa Labs (3rd, 12.64) published the most transferable ablation: test-time training and augmentation voting each gave roughly 410 to 430% over zero-shot, and together about 812%, so they compose almost perfectly.

Jigsaw is the same move on text. Each test row carried two positive and two negative example comments for its rule, and 14 of the 15 top-20 writeups trained on them. Guanshuo Xu (1st) trained six LoRAs inside an 11-hour run on two T4s; Sergio Papadakis (3rd) started from a LoRA already trained on public rules, so the in-notebook epoch began from a model that knew the task.

## 7. Augmentation voting: ask the question eight ways

If a model's answer is right, it should stay right when you rotate the puzzle. That is the whole idea behind AIRV (augment, infer, reverse, vote).

![Eight symmetry views of one toy ARC task, each prediction mapped back to the original frame: six agree on the correct grid, two have a single wrong cell in different places, and the tally is 6 to 1 to 1](/imgs/blogs/kaggle-llm-test-time-compute-playbook-5.webp)

Errors made in different views land on different cells once mapped back, so they split their votes while the correct grid collects the rest. Barbadillo (ARC 2024) made 96 predictions per task this way; MindsAI (ARC 2025) ran 10,000 augmented inferences per task.

The ARChitects replaced counting with scoring. Their depth-first search enumerates every completion above a probability cutoff instead of sampling noisy duplicates, then ranks each candidate by the product of its probabilities under 8 views; their paper reports this selection about 25% better than their baseline for 10 to 17 extra minutes per 100 tasks. NVARC kept the 8 views in 2025. In Jigsaw, takaito's team (11th) ran test-time augmentation over different combinations of the example comments, the text analogue of rotating a grid. MindsAI also found that at equal compute, ensembling seeds beat doubling the training and voting samples by 6.2%, which fits Lonnie (ARC 2025, 4th) seeing one notebook score 3.33 to 6.67% across seeds.

## 8. When the host owns the decoder

The [Nemotron Reasoning Challenge](/blog/machine-learning/learn-from-kaggle/kaggle-nvidia-nemotron-reasoning-challenge-solutions) is the control experiment. Teams shipped only a LoRA adapter; Kaggle ran it in vLLM at temperature 0, one sample, 7,680 output tokens. No sampling, no vote, no tool, no early stop.

So test-time compute moved into the trace. Tong Hui Kang, whose Open Progress Prize code became the base for nearly every top team, framed training as maximising the minimum log-probability over trace tokens, because greedy decoding fails on its weakest token. NullSira (1st, 0.920) wrote bit strings in hex and cut the median bit-manipulation trace from 6,771 to 4,888 tokens, leaving room under the cap to check candidate rules. When you cannot buy inference compute, you pay for it in training data.

## 9. What did not work, across competitions

| Idea | Competition, team | What happened |
|---|---|---|
| More samples | AIMO 1, Numina and after exams; AIMO 3, SKobayak | Beyond N=48 no gain; more generations stopped improving validation; 16 solvers worse than 8 |
| Trained verifier instead of a vote | ARC 2024, Barbadillo | Beat voting offline, nothing on the leaderboard |
| Outcome reward model and GenSelect | AIMO 2, NemoSkills | ORM gains faded on long TIR traces; GenSelect cost too much memory and time |
| Pure inverse-entropy vote | AIMO 3, varianceofx | One confident wrong attempt can hijack it |
| Adaptive time budgets | AIMO 1, Numina; AIMO 2, sravn | High variance; sensitive to random question order |
| Program search at test time | ARC 2025, NVARC | Too slow under Kaggle limits, moved into training data instead |

## 10. When to reach for each technique

| Your competition looks like | Reach for first | Then | Skip |
|---|---|---|---|
| Discrete answers, minutes per item (AIMO) | Many samples plus Python in the loop | Early stop at k agreeing, a time pool, share-first weighting | A second model for selection unless it fits beside the policy |
| Each item ships labelled examples (ARC, Jigsaw) | Test-time training on those examples | Augmentation voting or multi-view rescoring | Large sample counts from an untrained model |
| Continuous output scored by a model (Drawing) | Best-of-n with a local copy of the judge | More candidates per minute: few-step samplers, pipelining | Voting; there is nothing discrete to count |
| Host fixes decoding (Nemotron) | Shorter, more checkable traces | Token-efficient encodings | Everything in this post that happens at inference |

A checklist for the next one:

1. Divide the clock by the hidden item count before picking a model.
2. Measure maj@k against pass@k; the gap is your selection budget.
3. Find where more samples stop helping, and stop there.
4. Add stop rules for "decided" and for "hopeless".
5. If the test file contains labelled examples, train on them.
6. Evaluate time policies under several question orders.
7. Submit the same notebook twice before trusting a half-point difference.

## Sources

Published competition posts in this series: [AIMO 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions), [AIMO 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions), [AIMO 3](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions), [ARC 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions), [ARC 2025](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2025-solutions), [Jigsaw Rules](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions), [Nemotron](/blog/machine-learning/learn-from-kaggle/kaggle-nvidia-nemotron-reasoning-challenge-solutions), [Drawing with LLMs](/blog/machine-learning/learn-from-kaggle/kaggle-drawing-with-llms-solutions).

Writeups cited:

- AIMO 1: [Numina](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/numina-numina-1st-place-solution), [CMU_MATH](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/cmu-math-2nd-place-solution-all-code-and-datasets-), [after exams](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/after-exams-3rd-place-solution), [codeinter](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/codeinter-4th-place-solution)
- AIMO 2: [NemoSkills](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/nemoskills-1st-place-solution-nemoskills), [imagination-research](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/572948), [Aliev](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573314), [sravn](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573671), [farsail](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/discussion/573086), [arek-paterek](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-2/writeups/arek-paterek-20th-place-solution-generate-lots-of-)
- AIMO 3: [Exalted Joseph](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/1st-place-solution-for-the-aimo3-competition), [varianceofx](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/2nd-place-solution-ai-mathematical-olympiad-prog), [SKobayak](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/3rd-place-solution-for-the-aimo3-competition), [TAMU-TACO](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3-tamu-taco-solution), [yemao ye](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/7th-place-solution-for-the-aimo3-competition)
- ARC 2024: [the ARChitects paper](https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf), [Guillermo Barbadillo](https://www.kaggle.com/competitions/arc-prize-2024/writeups/guillermo-barbadillo-2nd-place-solution-for-the-ar), [William Wu](https://www.kaggle.com/competitions/arc-prize-2024/writeups/william-wu-4th-place-solution), [Nikola Hu](https://www.kaggle.com/competitions/arc-prize-2024/writeups/nikola-hu-sharing-my-arc-prize-2024-code-31-points)
- ARC 2025: [NVARC](https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc), [the ARChitects](https://www.kaggle.com/competitions/arc-prize-2025/writeups/the-architects-solution-2025), [MindsAI and Tufa Labs](https://www.kaggle.com/competitions/arc-prize-2025/writeups/mindsai-and-tufa-labs-arc-prize-2025-solution), [Lonnie](https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place)
- Jigsaw: [Guanshuo Xu](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613305), [Sergio Manuel Papadakis](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613324), [takaito / Muj!rush! / yukko / miyaki](https://www.kaggle.com/competitions/jigsaw-agile-community-rules/discussion/613506)
- Nemotron: [NullSira](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution)
- Drawing with LLMs: [new team name](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581027), [WXYZ123](https://www.kaggle.com/competitions/drawing-with-llms/discussion/581012)
