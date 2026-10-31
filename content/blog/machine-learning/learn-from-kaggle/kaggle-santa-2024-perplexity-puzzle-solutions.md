---
title: "Kaggle Santa 2024: the top 20 teams, an LLM as the objective, and a 13-way tie"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A synthesis of the top-20 writeups from Santa 2024, the Perplexity Permutation Puzzle: why every finisher treated Gemma 2 9B as a frozen cost function and won with iterated local search, simulated annealing and cheap surrogate filters, how two alphabetical blocks cracked the 100-word row, and why 13 teams tied at the optimum and were ranked by the calendar."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "combinatorial-optimization", "local-search", "simulated-annealing", "perplexity", "gemma", "surrogate-models"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** Santa 2024 (November 21, 2024 to January 31, 2025, 1,514 teams) handed out six bags of Christmas words and asked for the ordering of each bag that a frozen Gemma 2 9B found least surprising. The score was the mean perplexity of the six sequences. Nobody trained a model. Every top-20 team with a writeup treated the LLM as an expensive black-box cost function, the puzzle as an asymmetric travelling salesman problem over words, and won with classical search: iterated local search (1st, 2nd, 4th), simulated annealing (11 of 14 teams with a source), and at the top a cheap learned filter that kept nine in ten candidates away from Gemma. **13 teams finished at exactly 246.81784**, the optimum as far as anyone ever found. The 100% public leaderboard ordered them by submission time, so the race was not about who reached the floor but who reached it first. Hamiltonians did it by December 14, 48 days before the close. **The lesson:** when an LLM is the objective, the craft is optimisation, and the first thing to optimise is the cost of asking the LLM.

![Lever strip with Test-time compute lit: search against a frozen Gemma 2 9B scorer](/imgs/blogs/kaggle-santa-2024-perplexity-puzzle-solutions-1.webp)

The strip above is the series frame from the [orientation post](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work): five levers, one lit per competition. Here the LLM is not the thing you build; it is the judge. You can only feed it orderings of fixed words and read back a number. The lit lever is test-time compute in its purest form: how many orderings you can get Gemma to score, and how cleverly you choose them. It sits in the same grey lane as [Google Code Golf 2025](/blog/machine-learning/learn-from-kaggle/kaggle-google-code-golf-2025-solutions), where classical craft beat LLM-driven entries outright.

## The task in 60 seconds

Kaggle's annual Santa competition is a combinatorial optimisation puzzle; in 2024 it was built around a language model. The data was six rows of Christmas words, with ids 0 to 5, of 10, 20, 20, 30, 50 and 100 words. For each row you submitted a permutation of its words. You could not add, drop or change a word, only reorder. The bags held words like "reindeer", "sleigh", "magi" and plenty of function words ("the", "of", "and"); some repeated.

![Budget card: own GPUs with no runtime limit, scoring on Kaggle T4, six rows, 100% public leaderboard](/imgs/blogs/kaggle-santa-2024-perplexity-puzzle-solutions-2.webp)

The budget card is the inverse of a normal Kaggle LLM competition: no hidden test and no notebook clock. The optimisation ran on participants' own hardware for as long as they liked; only the scoring ran on Kaggle, on T4 GPUs. The leaderboard was 100% public, and ties were broken by submission time. The prize pool was USD 50,000: 12k, 10k, 10k and 8k for places 1 to 4, plus a 10k **Rudolph Prize** for the team that held 1st place longest. No primary source we found names its winner; the dates make Hamiltonians likely, but that is unverified.

![The six rows drawn to scale, with the number of orderings and the best final perplexity of each](/imgs/blogs/kaggle-santa-2024-perplexity-puzzle-solutions-3.webp)

The figure shows why the rows are not equally hard. Row id0 has ${10! = 3{,}628{,}800}$ orderings: a public "Brute forcing" thread (simjeg) showed it can be enumerated outright. Row id1 already has ${20! \approx 2.4 \times 10^{18}}$, and id5 has ${100! \approx 9.3 \times 10^{157}}$. Ids 0 to 2 were solved early by essentially everyone; the board was decided on id3 and id5.

## The metric, worked on a toy

The score of one sequence is its perplexity under Gemma 2 9B, and the leaderboard number is the plain mean of the six perplexities. Lower is better.

$$
\mathrm{PPL}(s) = \exp\!\Big(\frac{1}{T}\sum_{t=1}^{T} -\log p_\theta(x_t \mid x_{\lt t})\Big),
\qquad
\text{score} = \frac{1}{6}\sum_{i=0}^{5} \mathrm{PPL}(s_i)
$$

Here $x_1, \dots, x_T$ are the tokens of the sequence, $p_\theta$ is the frozen Gemma, and $T$ is the token count. Perplexity is the geometric mean of ${1/p}$: how many equally likely choices the model felt it faced, on average.

![Two orderings of four words: per-token loss columns drawn to scale, perplexity 7.07 against 56.23](/imgs/blogs/kaggle-santa-2024-perplexity-puzzle-solutions-4.webp)

The toy uses made-up probabilities, one token per word. "Reindeer pulled the sleigh" gets per-token probabilities 0.02, 0.10, 0.50 and 0.40, a mean loss of 1.956 and a perplexity of 7.07. "Sleigh the pulled reindeer" gets 0.01, 0.05, 0.01 and 0.02, a mean loss of 4.030 and a perplexity of 56.23. Same words, eight times the score. Notice the side note in the figure: a token's loss depends only on the words before it. Two teams built on that fact, as we will see.

```python
# Illustrative toy: the Santa 2024 score with made-up token probabilities.
import math

def perplexity(token_probs):
    nll = [-math.log(p) for p in token_probs]      # loss of each token
    return math.exp(sum(nll) / len(nll))           # exp(mean loss)

a = perplexity([0.02, 0.10, 0.50, 0.40])   # "reindeer pulled the sleigh"
b = perplexity([0.01, 0.05, 0.01, 0.02])   # "sleigh the pulled reindeer"
print(f"ordering A: {a:6.2f}   ordering B: {b:6.2f}")

# The leaderboard score is the plain mean of six perplexities.
best = [469.77, 424.38, 298.93, 191.73, 67.54, 28.52]   # per-id optima
print(f"mean of the six optima: {sum(best) / 6:.2f}")
# One id5 basin step (28.574555 -> 28.529695) moves the mean by:
print(f"id5 step on the leaderboard: {(28.574555 - 28.529695) / 6:.5f}")
```

```console
ordering A:   7.07   ordering B:  56.23
mean of the six optima: 246.81
id5 step on the leaderboard: 0.00748
```

The per-id optima come from 4th place's table; their mean, 246.81, matches the final 246.81784 within rounding. The last line matters: the 100-word row has the lowest perplexity, so a whole basin of improvement on it moves the mean by less than a hundredth, while the short, high-perplexity rows dominate the absolute number and decide nothing.

Two early incidents shaped the metric. Kaggle staff revised it three times in the first week, because the launch version scored differently at different batch sizes and then needed label-shift and boundary-token fixes, with a full rescore each time. And c-number, later on the 1st-place team, showed that inserting about 83 spaces between words drove perplexity absurdly low; it was patched. Batch-dependent numerics are an old enemy, covered in [sampling numerics and batch invariance](/blog/machine-learning/inference-engineering/sampling-numerics-determinism-and-batch-invariance).

## What made it hard

Three things, and none of them is the language model.

**The oracle is expensive.** Every candidate costs a forward pass of a 9-billion-parameter model, and one full restricted neighbourhood of id5 took about 18 minutes on an RTX 4090 (2nd place). Throughput was the first battleground, which is why Chris Deotte's thread on batching correctly was among the most-voted: right-pad, mask the pads in the loss, or the label shift silently drops `<bos>`. The same tax is the subject of [static batching and the padding tax](/blog/machine-learning/inference-engineering/static-batching-and-the-padding-tax). A later thread reported about 20% more throughput on an A6000 from `attn_implementation="sdpa"`.

**The landscape is full of deep traps.** id3 had a famous basin at 197.5; the optimum, 191.73, sat behind a different first word. id5 had basins at about 32, 30.x, 28.57 and 28.52, each a structurally different arrangement.

**Scoring hardware drifts.** 2nd place found near-ties that flipped between a T4 and an RTX 4090, so they re-scored their best solutions and neighbours on a Kaggle T4. At five decimal places, floating-point ordering is part of the problem.

## The top-20 scorecard

The leaderboard is 100% public, so every "public-rank move" is 0. Ranks 1 to 13 share 246.81784 and are listed in API order, which is submission-time order. Teams with no writeup are stated as such.

| Rank | Team | Score | Public move | Key idea | Source |
|---|---|---|---|---|---|
| 1 | Hamiltonians | 246.81784 | 0 | ILS + thresholded DFS; online-trained 1.3M-param ConvNet prunes neighbours | [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560560) |
| 2 | Word Salad | 246.81784 | 0 | ILS with restricted k-opt, shuffle kicks, priority-queue climber | [overview](https://www.kaggle.com/competitions/santa-2024/discussion/560540) |
| 3 | SantaMizers | 246.81784 | 0 | SA with adaptive operator weights; sorted stopwords + verbs + others seed | [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560620) |
| 4 | Santa Master | 246.81784 | 0 | Delete N words, beam-search re-insertion; function-words-first structure | [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560536) |
| 5 | Moș Noël | 246.81784 | 0 | Batched multi-point threshold accepting, crossover to best, ATSP moves | [CPMP part](https://www.kaggle.com/competitions/santa-2024/discussion/560597) |
| 6 | Santa Seekers | 246.81784 | 0 | Insertion-only SA; LSTM surrogate ranks moves | [Shun](https://www.kaggle.com/competitions/santa-2024/discussion/560625) |
| 7 | Miner2024 | 246.81784 | 0 | ILS around batched SA; learned bigram cost matrix prunes 1M candidates | [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/561091) |
| 8 | KS | 246.81784 | 0 | unknown | no writeup |
| 9 | Fu Ryo | 246.81784 | 0 | GA with Edge Assembly Crossover, Gemma logits guide edges | [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560601) |
| 10 | B.A.G.E.L. | 246.81784 | 0 | Plain SA, two moves; stopwords + two sorted blocks seed | [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560531) |
| 11 | Ryota | 246.81784 | 0 | SA with an insertion family; "magi" first for id3 | [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560542) |
| 12 | juvenjiang | 246.81784 | 0 | unknown | no writeup |
| 13 | Honke Santa Master | 246.81784 | 0 | SA + GA + threshold-accepting ILS, W&B-managed team | [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560683) |
| 14 | adakoda | 246.82532 | 0 | unknown (score = the 28.57 id5 basin) | no writeup |
| 15 | Rist Santa | 246.86686 | 0 | unknown | no writeup |
| 16 | Santa Magister | 246.88339 | 0 | Free SA, then two and three alphabetical blocks for id5 | [member post](https://www.kaggle.com/competitions/santa-2024/discussion/560713) |
| 17 | just team | 246.88339 | 0 | SA modelled on a public notebook, hashed cache | [member post](https://www.kaggle.com/competitions/santa-2024/discussion/560770) |
| 18 | BeerMind + Denden + Gilles | 246.88983 | 0 | SA with GA mixing of a diverse elite pool | [member post](https://www.kaggle.com/competitions/santa-2024/discussion/560550) |
| 19 | bono | 246.89181 | 0 | unknown | no writeup |
| 20 | zBere | 246.92798 | 0 | unknown | no writeup |

Rank 13's team name is Japanese for "the original Santa Master". 14 of the 20 teams left a writeup or member post.

![Thirteen tied teams placed by last-submission date, and the gaps above the optimum for ranks 14 to 20](/imgs/blogs/kaggle-santa-2024-perplexity-puzzle-solutions-6.webp)

This replaces the usual shake-up chart, because here the only shake-up was the calendar. The top panel places each tied team at its last-submission date from the Kaggle API, an upper bound on when it reached the optimum: 5th place last submitted on January 30 yet ranks above teams that finished earlier, because the tie is broken by the time of the scoring submission. Hamiltonians' last submission is December 14, 23 days into a 71-day competition. nagiss says they had the optimum by mid-December and, after seeing equal scores line up, turned his GPU off. 2nd, 3rd and 4th followed in early January; ranks 5 to 13 last submitted between January 19 and 30.

The bottom panel shows the near-misses. 14th place, adakoda, sits at +0.00748, and ${0.00748 \times 6 = 0.0449}$, which is exactly the difference between id5 at 28.574555 and at 28.529695 (0.04486; the two agree to rounding). 4th place's writeup reports both values. So 14th was one id5 basin away from a 13th-place tie. The 246.88 cluster is a larger miss; which rows ranks 15 to 20 missed is not stated anywhere we found.

## The top-20 landscape

![Top-20 technique matrix: 14 of 20 teams left a writeup, 15 techniques](/imgs/blogs/kaggle-santa-2024-perplexity-puzzle-solutions-5.webp)

Filled means used, outlined means not used or rejected, "?" means the writeup is silent. We recounted every row from the cells; two counts differ from our first-pass notes (insertion moves are 11 of 14, not 13, and all four surrogate users are in the top 9, not three).

**Table stakes.** Every team with a source treated the puzzle as an asymmetric TSP over words with Gemma as the cost oracle. Nobody trained a generator; every finisher used local search. 11 of 14 used remove-and-reinsert moves and 11 of 14 used block moves or k-opt without reversal. 7 of 14 report batched GPU scoring and 6 of 14 a global score cache; the rest are silent rather than opposed. 10 of 14 seeded id4 or id5 with a stopwords-first or alphabetical structure.

**The engine split.** 11 of 14 ran simulated annealing (SA) as the main engine, in some variant. The three that did not are 1st, 2nd and 4th, which all ran iterated local search (ILS). 5 of 14 used ILS with explicit kicks in total (1st, 2nd, 4th, 7th and 13th, the last two wrapping it around SA). This is not "ILS beat SA": 3rd place won with SA. But the most methodical searches got there earliest.

**What separated the top of the tie from 14 to 20.** Finding id5's last steps: two sorted blocks, not one or three; "and" moved out of the function-word block; units like "card game" kept together. Within the tie, the separator was **time to optimum**. The early teams had either a general search strong enough to find structure unaided (1st's neural filter; 5th, which found the best score from random shuffles on every row but id3; 6th's chris62, who found all six optima in a couple of days on one RTX 3090), or an analytic read of the structure (4th). Teams from 10th down mostly report reaching id5's optimum only after a public notebook revealed the two-block shape.

**Contradictory findings**, each with its source:

| Question | One side | Other side |
|---|---|---|
| Swap moves | Hurt, rarely accepted (6th Shun, 11th) | One of only two moves kept to the end (10th) |
| Many operators or few | Diverse operators mattered (6th Yuki; 3rd's 12 to 15 adaptive operators) | Complex neighbourhoods burn compute for nothing (10th) |
| Simulated annealing | "Not methodical enough", dropped (2nd) | Main engine of most of the tie (3rd, 6th, 7th, 10th, 11th) |
| Genetic algorithms | Failed, no good crossover (2nd) | Worked with EAX (9th); delivered id3 and id5 optima (13th) |
| Beam search | No non-myopic evaluation, failed (2nd) | Core of the re-insertion step (4th) |
| Surrogate architecture | ConvNet for locality (1st) | LSTM; transformers did worse (6th chris62) |

## The winning solution: Hamiltonians

Hamiltonians were kibuna, nagiss and c-number; their code is in [Lgeu/santa2024](https://github.com/Lgeu/santa2024). The design has three parts: an ILS outer loop, a thresholded depth-first local search, and a small neural network that decides which neighbours deserve a Gemma call.

![Hamiltonians pipeline: kick, thresholded DFS local search, a ConvNet funnel that passes fewer than 1 in 10 neighbours to Gemma](/imgs/blogs/kaggle-santa-2024-perplexity-puzzle-solutions-8.webp)

### The outer loop: iterated local search

ILS is the oldest good idea in combinatorial search. Run local search to a local minimum, perturb the best solution a little (a "kick"), search again, keep the result if it beats the best. A random restart throws away everything you know; a kick does not.

<figure class="blog-anim">
<svg viewBox="0 0 760 330" role="img" aria-label="Iterated local search on a one-dimensional perplexity landscape: a ball descends into a basin, gets kicked over a ridge, descends into a deeper basin, is kicked back into the same basin once with no gain, then kicked again into the deepest basin; a dashed best-so-far line steps down at each improvement" style="width:100%;height:auto;max-width:760px">
<style>
.i1-land{fill:none;stroke:var(--text-secondary,#6b7280);stroke-width:2.5}
.i1-ax{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.i1-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);opacity:0}
.i1-ball{fill:var(--accent,#6366f1)}
.i1-best{stroke:var(--accent,#6366f1);stroke-width:1.5;stroke-dasharray:6 5;opacity:.7}
.i1-btxt{font:500 12px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
@keyframes i1-move{0.0%{transform:translate(70px,65px)} 2.1%{transform:translate(87px,69px)} 4.3%{transform:translate(104px,78px)} 6.4%{transform:translate(120px,97px)} 8.6%{transform:translate(137px,127px)} 10.7%{transform:translate(154px,156px)} 12.9%{transform:translate(171px,169px)} 15.0%{transform:translate(192px,117px)} 17.1%{transform:translate(214px,80px)} 19.3%{transform:translate(236px,58px)} 21.4%{transform:translate(257px,52px)} 23.6%{transform:translate(278px,62px)} 25.7%{transform:translate(300px,87px)} 27.9%{transform:translate(314px,102px)} 30.0%{transform:translate(329px,123px)} 32.1%{transform:translate(343px,150px)} 34.3%{transform:translate(357px,181px)} 36.4%{transform:translate(372px,206px)} 38.6%{transform:translate(386px,216px)} 40.7%{transform:translate(396px,168px)} 42.9%{transform:translate(406px,133px)} 45.0%{transform:translate(416px,112px)} 47.1%{transform:translate(425px,105px)} 49.3%{transform:translate(435px,110px)} 51.4%{transform:translate(445px,129px)} 53.6%{transform:translate(435px,144px)} 55.7%{transform:translate(425px,162px)} 57.9%{transform:translate(416px,181px)} 60.0%{transform:translate(406px,199px)} 62.1%{transform:translate(396px,211px)} 64.3%{transform:translate(386px,216px)} 66.4%{transform:translate(408px,157px)} 68.6%{transform:translate(431px,115px)} 70.7%{transform:translate(453px,89px)} 72.9%{transform:translate(475px,81px)} 75.0%{transform:translate(498px,89px)} 77.1%{transform:translate(520px,113px)} 79.3%{transform:translate(534px,130px)} 81.4%{transform:translate(547px,156px)} 83.6%{transform:translate(560px,189px)} 85.7%{transform:translate(574px,226px)} 87.9%{transform:translate(588px,256px)} 90.0%{transform:translate(601px,267px)} 100%{transform:translate(601px,267px)}}
.i1-anim{animation:i1-move 16s ease-in-out infinite}
@keyframes i1-bestk{0%,12.9%{transform:translateY(178px)}13.0%,38.6%{transform:translateY(178px)}38.7%,90.0%{transform:translateY(225px)}90.1%,100%{transform:translateY(276px)}}
.i1-bmove{animation:i1-bestk 16s steps(1,end) infinite}
@keyframes i1-l1k{0%,-0.1%{opacity:0}0%,18.9%{opacity:1}19.0%,100%{opacity:0}}
.i1-l1{animation:i1-l1k 16s linear infinite}
@keyframes i1-l2k{0%,18.799999999999997%{opacity:0}18.9%,34.6%{opacity:1}34.7%,100%{opacity:0}}
.i1-l2{animation:i1-l2k 16s linear infinite}
@keyframes i1-l3k{0%,34.5%{opacity:0}34.6%,50.6%{opacity:1}50.7%,100%{opacity:0}}
.i1-l3{animation:i1-l3k 16s linear infinite}
@keyframes i1-l4k{0%,50.5%{opacity:0}50.6%,72.6%{opacity:1}72.69999999999999%,100%{opacity:0}}
.i1-l4{animation:i1-l4k 16s linear infinite}
@keyframes i1-l5k{0%,77.9%{opacity:0}78.0%,100%{opacity:1}100.1%,100%{opacity:0}}
.i1-l5{animation:i1-l5k 16s linear infinite}
@media (prefers-reduced-motion:reduce){.i1-anim,.i1-bmove,.i1-lbl{animation:none}.i1-lbl{opacity:1}.i1-l1,.i1-l2,.i1-l3,.i1-l4{opacity:0}.i1-anim{transform:translate(601px,267px)}.i1-bmove{transform:translateY(276px)}}
</style>
<text class="i1-ax" x="14" y="24">perplexity (lower is better)</text>
<text class="i1-ax" x="560" y="322">orderings of the word bag</text>
<path class="i1-land" d="M40,70.0 L48,70.9 L56,71.8 L64,72.8 L72,74.0 L80,75.7 L88,78.2 L96,82.0 L104,87.5 L112,95.2 L120,105.6 L128,118.6 L136,133.3 L144,148.4 L152,162.2 L160,172.4 L168,177.6 L176,176.9 L184,170.6 L192,160.0 L200,146.9 L208,133.3 L216,120.9 L224,110.6 L232,102.7 L240,96.8 L248,92.4 L256,88.9 L264,86.4 L272,85.2 L280,85.7 L288,88.4 L296,93.2 L304,99.8 L312,108.2 L320,118.5 L328,130.7 L336,145.2 L344,161.6 L352,178.8 L360,195.4 L368,209.7 L376,219.8 L384,224.4 L392,222.9 L400,215.7 L408,203.9 L416,189.3 L424,173.7 L432,158.7 L440,145.4 L448,134.5 L456,125.7 L464,118.9 L472,113.6 L480,109.7 L488,107.7 L496,107.9 L504,110.4 L512,115.3 L520,122.4 L528,131.7 L536,143.4 L544,158.2 L552,176.1 L560,196.8 L568,219.0 L576,240.4 L584,258.6 L592,271.1 L600,276.0 L608,272.7 L616,261.8 L624,245.2 L632,225.4 L640,204.9 L648,186.0 L656,170.1 L664,157.8 L672,149.0 L680,143.3 L688,140.0 L696,138.3 L704,137.8 L712,137.8 L720,138.3"/>
<g class="i1-bmove"><line class="i1-best" x1="40" y1="0" x2="720" y2="0"/><text class="i1-btxt" x="660" y="-6">best so far</text></g>
<circle class="i1-ball i1-anim" cx="0" cy="0" r="9"/>
<text class="i1-lbl i1-l1" x="131" y="206.05308084521874">descend to a local minimum</text>
<text class="i1-lbl i1-l2" x="215" y="98.05308084521874">kick: shuffle a few words</text>
<text class="i1-lbl i1-l3" x="346" y="252.55195362159117">descend: better, keep it</text>
<text class="i1-lbl i1-l4" x="300" y="129.55195362159117">kick lands back in the same basin: no gain</text>
<text class="i1-lbl i1-l5" x="521" y="304.03487305136423">kick, descend: new best</text>
</svg>
<figcaption>Iterated local search, the engine of 1st, 2nd and 4th place: descend until no move helps, kick the best solution a little, descend again, keep the result only if it beats the best so far. A kick is small enough to stay near good solutions and big enough to leave the current basin.</figcaption>
</figure>

Hamiltonians' loop runs from the best solution, three times per round: kick, then local search, keep if better. They used two kicks: swap two random words, or remove a random contiguous segment and re-insert each removed word at a random position.

### The inner loop: depth-first search with a threshold

Their local search looks at neighbours made by taking a small number of words out and inserting them elsewhere. If a neighbour improves on the best, take it and restart. If it does not, but its score is below a threshold, recurse into it and look at its neighbours. The threshold is about ${1.1 \times}$ the best at shallow depth and the best itself deeper down. That lets the search walk through one or two slightly worse orderings to reach an improvement, a bounded look past a bad step, which plain first-improvement descent can never do. Node-count limits stop the recursion from exploding.

### The trick: a 1.3M-parameter filter in front of Gemma

The decisive idea is economic. A network of about 1.3 million parameters, roughly 1/5000 of Gemma 2 9B, screens every neighbour, and fewer than one in ten gets an exact Gemma score. nagiss added in the comments that it is **a ConvNet inspired by SE-ResNet, not a transformer**, because "how nearby words are arranged seemed to be more important than their global relationships". It only throws away sequences that will clearly be worse. It is trained **online during the search**, "in a manner similar to reinforcement learning", on the Gemma scores the search generates anyway; pretraining turned out to be unnecessary.

This is the surrogate-model pattern applied to an LLM oracle: a cheap learned proxy ranks, the expensive truth decides. The toy below shows why it is worth more than a better search loop. The "LLM" is a stand-in: pair costs between consecutive words plus a weaker three-word term, so there are real local minima. The surrogate sees only the pair costs. Every mode gets the same budget of 60,000 oracle calls.

```python
# Illustrative toy, not any team's code. A stand-in "LLM" scores an ordering
# of 30 words; we compare four ways of spending the same oracle budget.
import math, random

N, BUDGET = 30, 60_000
rng = random.Random(0)
# Asymmetric pair costs plus a weaker three-word term, so the landscape has
# real local minima (a pure pair model would be an easy ATSP).
pair = [[rng.uniform(0.2, 4.0) for _ in range(N)] for _ in range(N)]
tri = {(a, b, c): rng.uniform(-0.6, 0.6)
       for a in range(N) for b in range(N) for c in range(N)}

calls = 0
def ppl(order):                      # the expensive oracle
    global calls; calls += 1
    nll = [pair[a][b] for a, b in zip(order, order[1:])]
    nll += [tri[t] for t in zip(order, order[1:], order[2:])]
    return math.exp(sum(nll) / (N - 1))

def insertion_neighbours(order):     # take one word out, put it back elsewhere
    for i in range(N):
        rest = order[:i] + order[i + 1:]
        for j in range(N):
            if j != i:
                yield rest[:j] + [order[i]] + rest[j:]

def cheap(order):                    # surrogate: pair costs only, no oracle call
    return sum(pair[a][b] for a, b in zip(order, order[1:]))

def descend(order, score, filt=False):  # first-improvement local search
    improved = True
    while improved and calls < BUDGET:
        improved = False
        nbs = list(insertion_neighbours(order))
        if filt:                     # oracle sees only the surrogate's top 10%
            nbs = sorted(nbs, key=cheap)[:len(nbs) // 10]
        for nb in nbs:
            s = ppl(nb)
            if s < score:
                order, score, improved = nb, s, True
                break
            if calls >= BUDGET:
                break
    return order, score

def kick(order, r):                  # shuffle a random 4-word window
    i = r.randrange(N - 3)
    window = order[i:i + 4]; r.shuffle(window)
    return order[:i] + window + order[i + 4:]

def run(mode, seed):
    global calls; calls = 0
    r = random.Random(seed)
    start = list(range(N)); r.shuffle(start)
    best, best_s = start, ppl(start)
    if mode == "random":
        while calls < BUDGET:
            cand = list(range(N)); r.shuffle(cand)
            s = ppl(cand)
            if s < best_s: best, best_s = cand, s
        return best_s
    filt = mode == "ils+filter"
    best, best_s = descend(best, best_s, filt)
    while calls < BUDGET:
        if mode == "restart":        # forget everything, start again
            cand = list(range(N)); r.shuffle(cand)
        else:                        # ILS: perturb the best, not from scratch
            cand = kick(best, r)
        cand, s = descend(cand, ppl(cand), filt)
        if s < best_s: best, best_s = cand, s
    return best_s

for mode in ("random", "restart", "ils", "ils+filter"):
    scores = [run(mode, seed) for seed in range(10)]
    print(f"{mode:10s} median best PPL {sorted(scores)[5]:6.3f}  "
          f"worst {max(scores):6.3f}  best {min(scores):6.3f}")
```

```console
random     median best PPL  3.497  worst  3.674  best  3.228
restart    median best PPL  1.449  worst  1.540  best  1.395
ils        median best PPL  1.431  worst  1.472  best  1.315
ils+filter median best PPL  1.280  worst  1.363  best  1.243
```

Random sampling is hopeless, matching 2nd place's report that 14 million random id5 permutations found nothing better than about 500. Restarted local search is a huge step; ILS beats it modestly (its worst run beats restart's median). The filter is the big jump: the same 60,000 oracle calls go much further because the surrogate spends them only on promising neighbours. On this toy, the surrogate's worst run (1.363) beats plain ILS's median (1.431). That is the shape of Hamiltonians' advantage: not a cleverer walk, the same walk with ten times the legs.

## 2nd and 3rd in depth

### 2nd: Word Salad, local search as a TSP engineer would build it

Word Salad (zaburo, solverworld, danielphalen) framed it like a TSP solver: a neighbourhood to descend and a kick to leave local minima.

**Neighbourhoods that respect the causal model.** danielphalen used three, in increasing size: delete and reinsert one word, move a phrase, swap two phrases (a double bridge). Reversals were banned because reversing a run wrecks perplexity under a left-to-right model. zaburo's restricted k-opt limits the length of the *moving* subsequence (k=3 with at most 5 moving words; k=4 with 1). solverworld measured that delete-and-insert moves disturb perplexity less than arbitrary swaps, which is the quantitative reason insertion won across the whole top 20.

**Kicks.** The decisive kick was shuffling a random run of 5 or 10 words, breaking up groups that had locked together. For id5 a problem-specific kick swapped words between the stop-word prefix and the nearly sorted suffix, and id5 was seeded from their ids 1, 3 and 4 solutions, since its words come from those rows.

**The id3 trick.** zaburo pinned the *last* word, testing each word in the last position, which escaped the 197.5 trap into the 191s.

**Searchers.** danielphalen's backtracking climber jumps back a random number of steps on a local minimum and continues from a good unexplored solution. solverworld's priority-queue search pops the lowest-perplexity node, keeps up to two improved neighbours (or one worse) and drops a node after five moves without beating its best ancestor; it found id2's 298.9 within 24 hours on a 4090. They also wrote the competition's most useful negative-results list.

### 3rd: SantaMizers, simulated annealing with an adaptive toolbox

SantaMizers (siavrez, yuanzhezhou, asalhi, yyyu54) titled their writeup "Sorted stopwords, Sorted verbs, and Sorted Other words are all you need!". Two SA engines did the work.

The first uses 12 operators (swaps, reversals, remove-insert, rotations, shifts, block moves) with **adaptive weights**: the best-performing operator's weight is multiplied by 1.2, the others by 0.9 to 1.0 according to relative score, then renormalised. It accepts orderings that *equal* the best score to drift across plateaus. It was best on ids 0 to 4 and reached 30.x on id5. The second adds about 15 block operators, **cyclic cooling** and top-k parallel paths; it reached 28.5x on id5.

Their main finding is in the title: **the starting point mattered most.** Sorted stopwords, then sorted verbs, then sorted other words was the best seed for ids 3, 4 and especially 5, though its initial score is not the lowest.

## Ranks 4 to 20, by approach family

### Structure first: 4th place

![id5 basin ladder: about 500 random, about 50 free SA, about 32 one block, 28.574555 two blocks, 28.529695 with "and" moved out](/imgs/blogs/kaggle-santa-2024-perplexity-puzzle-solutions-7.webp)

**Santa Master (4th; onodera, daiwakun)** turned observations of the oracle into a smaller search space. Their ILS deletes N random words and re-inserts them by beam search. Plain, it stalled at 198.93 on id3, 74.33 on id4 and 36.58 on id5. Then three findings. A few words lengthen the run of well-predicted tokens when placed first; with "magi" first, id3's optimum fell in about three hours. Grouping the function words at the start lowers loss: id4's optimum is exactly function words followed by content words, which shrinks 50! orderings to a 14-word block and a 36-word block (a factor of about ${9.4 \times 10^{11}}$). For id5, splitting the content words into *two* alphabetical runs reached 28.574555; removing "and" from the function-word block gave 28.529695. Each rung of the ladder is a change of search space, not a better optimiser.

### Simulated annealing as the engine: 6th, 10th, 11th, 17th

**Santa Seekers (6th)** posted as individuals. Shun Fukuda ran SA with insertion moves only (one word, or 2 to 5 consecutive words), because swaps worsened solutions; a teammate's 30.5 id5 solution revealed the two-block split that reached 28.5. Yuki Ohkawa used a diverse move set with reheating. chris62's surrogate is covered below.

**B.A.G.E.L. (10th)**, in egortrushin's account, used only SA, and at the end only two moves: swap two words, and extract and reinsert one word. The seed decided the basin: stopwords plus one alphabetical block (about 48 on id5) cannot reach gold, while stopwords without "and", sorted block 1, "and and and", sorted block 2 reached 28.5. He ran three SA regimes (hot exploration, a reheat to escape, cold refinement) and watched the acceptance rate.

**Ryota (11th)** started from a public 250.10105 submission with ids 0 to 2 already optimal and spent a rented A100 on ids 3 to 5. His moves are an insertion family: random, distance-weighted (probability of target position $j$ proportional to ${1/|i-j|}$), alphabetical (next to a word with the same first letter) and variable-length. Fixing "magi" first escaped id3's 197.5 trap to 191.7; trying one to four alphabetical blocks for id5 showed two was best (28.57), and the variable-length insert kept "card game" and "wrapping paper" together for 28.52.

### Population methods: 5th, 9th, 13th, 18th

**Moș Noël (5th)**, in CPMP's part, replaced the exp(-delta/T) test with **threshold accepting** (accept anything under a slowly lowered bound) and ran it **multi-point on the GPU**: one chain per batch slot (104 for id5 on an A100), with chains far from the best replaced by a crossover with the best. Moves came from the asymmetric-TSP literature. His most original idea is **discounted perplexity**: early positions move the score far more than late ones, so search drifts to editing the tail; normalising each position's logit by its running average made early and late moves equally acceptable. Teammate Horea supplied the id3 grouping CPMP's method could not find.

**Fu Ryo (9th)** ran a genetic algorithm with Edge Assembly Crossover (GA-EAX for the ATSP, Nagata and Soler 2012) on all six rows. The twist: keeping each parent's Gemma logits gives a rough estimate of perplexity when a word is placed after another, which guided edge choices during repair. Other crossovers barely worked; only EAX did.

**Honke Santa Master (13th)**, a university team, combined SA with a restart cut-off, a GA seeded with diverse local optima that took id3 from 195.0 to 191.7, and yukari17's ILS with exhaustive neighbourhoods, threshold accepting and a tabu rule (never pass the same score twice), which took id5 from 31.5 to 28.8. Stuck at 32 on id5 until January 29, they ran a public two-block notebook and made the final step 25.5 hours before the deadline.

**BeerMind + Denden + Gilles (18th)** mixed elite solutions with a GA inside SA, keeping the pool diverse by edit distance; they were 8th a week before the end. **just team (17th)** reached id3's optimum with SA modelled on a public notebook.

### Surrogate filters: 1st, 6th, 7th, 9th

Four teams independently built a cheap model to rank candidates before Gemma, and all four finished in the top 9. **chris62 (6th)** trained a per-row LSTM that predicts the score within about 2 points on id5: generate 10k to 100k moves, rank them, score the top 100 or so with Gemma, about 100 times faster. It generalised poorly across basins, so he retrained it on each jump. **Miner2024 (7th)** built a 102 by 102 word-to-word cost matrix where each edge costs the mean perplexity of the good solutions containing it; ranking a million candidates takes a few hundred milliseconds. **9th's** Gemma-logit edge estimates are the fourth, free rather than trained.

### Searching the structure space: 16th, and the no-writeup teams

**Santa Magister (16th)** shows the other side of structure search. johnmichaelwu found free SA converged to about 50 on id5, moved to block-alphabetical spaces, then was "trolled" by three blocks beating two at an intermediate stage and stayed there (best 28.92). A restricted space that cannot contain the optimum is a trap. Teammate woosungyoon matters to the whole board: his public alphabetical-block notebook, posted after what the forum called an accidental disclosure, is what 10th, 13th and 18th cite as their id5 unlock.

**No writeup found:** KS (8th, solo, only 11 submissions), juvenjiang (12th), adakoda (14th), Rist Santa (15th), bono (19th) and zBere (20th). We do not guess their methods.

**The best contrasting entry** is 5th's CPMP part: no structure seeding, no learned surrogate, and still the optimum from a random shuffle on five rows out of six, on brute GPU parallelism and a cache.

## What the forum taught everyone

The forum mattered more here than anywhere in the series, because the score was fully public. simjeg settled that 10! is enumerable and 20! is not, and his "Quantum Santa" thread relaxed permutations into doubly stochastic matrices trained with REINFORCE, original and unused by any winner. Giba showed alphabetical order is a strong prior for the 100-word bag, the seed of every top id5 solution. In mid-January ONODERA posted the per-row target scores, telling everyone where the floor was. And Maksim Sinitsyn's "Publishing top solution is dangerous" named the mechanism behind the tie: a 100%-public leaderboard plus shared optimal files flattens the board. Gold and the optimum were the same score.

## What did not work

Aggregated across the top-20 sources, with who reported it:

- **SA and late-acceptance hill climbing** as a main engine (2nd): not methodical enough.
- **Branch and bound** (2nd). Prefix losses are exact because later words never change earlier token losses, and one forward pass gives the next-word loss for every candidate, a 100-times saving on id5. The optimistic bound was still too loose to prune.
- **Beam search without a non-myopic evaluation** (2nd); contrast 4th, who used it only for re-insertion.
- **GA with generic crossover** (2nd); only EAX worked among crossovers (9th).
- **Multi-armed bandits over random permutations, and 14 million random id5 orderings** (best about 500) (2nd).
- **Gradient relaxations**: Sinkhorn permutation matrices trained by KL, iterated ATSP on the logit matrix (5th).
- **MCTS plus RL** to generate sequences (5th).
- **Optuna over per-word priority parameters**: fine for 10 words, does not scale (18th).
- **Swap moves**, too disruptive (6th Shun, 11th); **complex public neighbourhoods** (10th).
- **Free SA on id5 from scratch** (about 50, 16th); **one alphabetical block** (10th); **three or more blocks** (16th, 18th).
- **Phrase merging and fixing the head** risks excluding the optimum; connector-word patterns (18th).
- **A surrogate that generalises across basins**: it did not; retrain per basin (6th chris62).
- **Logistic-only or linear-only temperature schedules** (16th; the max of the two worked).

## Lessons that transfer

1. **An LLM can be a frozen objective, and then optimisation craft beats ML craft.** Every finisher with a source used local search, not a learned generator. If you score outputs with a model (a judge, a perplexity filter), the same toolkit applies; see [You Can't Please Them All](/blog/machine-learning/learn-from-kaggle/kaggle-llms-you-cant-please-them-all-solutions) for what happens when that judge is exploitable.
2. **Make the oracle cheap before making the search smart.** Correct batching, SDPA, a cache, and then a learned filter. All four surrogate teams finished in the top 9, and both learned-network filters report about 10 to 100 times fewer Gemma calls.
3. **Design moves around the model's inductive bias.** A causal LM punishes reversals, so insertion and short block moves win (2nd measured it; 6th and 11th rediscovered it).
4. **Read structure off the oracle and turn it into a search space.** Function words first, sorted content runs, "magi" first: 4th turned 50! into a 14-word and a 36-word problem.
5. **Pin a variable to escape a basin.** 6 of 14 teams with a source pinned a first or last word to break id3's 197.5 trap.
6. **Verify on the scoring hardware.** At five decimals, T4 versus 4090 numerics can flip a near-tie (2nd).

### If you enter a competition like this

- Get the metric bit-exact first: batch size, padding side, `<bos>` handling, scoring GPU.
- Measure oracle calls per second and treat it as your main hyperparameter.
- Cache every scored candidate, keyed on the token sequence.
- Start with insertion moves and an ILS loop; add SA or a population only if you can say why.
- Train a cheap ranker online on your own oracle calls; retrain when you jump basins.
- Spend an afternoon looking at your best solutions for structure, and reseed from it.
- On a fully public board with a tie-break by time, submit the moment you reach a new best.

## Sources

- Competition and metric: [Santa 2024 overview](https://www.kaggle.com/competitions/santa-2024), [metric notebook](https://www.kaggle.com/code/metric/santa-2024-metric/), [Metric Revision and Rescore](https://www.kaggle.com/competitions/santa-2024/discussion/547676), [Hacking the LB (fixed)](https://www.kaggle.com/competitions/santa-2024/discussion/547505)
- 1st, Hamiltonians: [writeup](https://www.kaggle.com/competitions/santa-2024/writeups/hamiltonians-1st-place-solution), [discussion with comments](https://www.kaggle.com/competitions/santa-2024/discussion/560560), [code](https://github.com/Lgeu/santa2024)
- 2nd, Word Salad: [writeup](https://www.kaggle.com/competitions/santa-2024/writeups/word-salad-2nd-place-solution), [overview](https://www.kaggle.com/competitions/santa-2024/discussion/560540), [zaburo](https://www.kaggle.com/competitions/santa-2024/discussion/560533), [solverworld](https://www.kaggle.com/competitions/santa-2024/discussion/560565), [basin climber notebook](https://www.kaggle.com/code/danielphalen/santabasinclimberv7-final), [priority-queue code](https://github.com/dgrunberg/Santa-2024-public)
- 3rd, SantaMizers: [writeup](https://www.kaggle.com/competitions/santa-2024/writeups/santamizers-3rd-place-solution), [discussion](https://www.kaggle.com/competitions/santa-2024/discussion/560620), [solution 1](https://www.kaggle.com/code/asalhi/3rd-place-sa-weighted-candidate-generator-sol1), [solution 2](https://www.kaggle.com/code/siavrez/3rd-place-sa-weighted-candidate-generator-sol2)
- 4th, Santa Master: [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560536), [gist](https://gist.github.com/KazukiOnodera/ff74dd5d9171cddac773e03f7d3457f3)
- 5th, Moș Noël: [CPMP part](https://www.kaggle.com/competitions/santa-2024/discussion/560597)
- 6th, Santa Seekers: [Shun Fukuda](https://www.kaggle.com/competitions/santa-2024/discussion/560625), [Yuki Ohkawa](https://www.kaggle.com/competitions/santa-2024/discussion/560796), [chris62 surrogate](https://www.kaggle.com/competitions/santa-2024/discussion/560553)
- 7th, Miner2024: [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/561091), [code](https://www.kaggle.com/code/max2020/santa24-7th-place-solution)
- 9th, Fu Ryo: [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560601), [gist](https://gist.github.com/FujiwaraRyo/9a30b1cc7b615789de0aa264d2e63faa)
- 10th, B.A.G.E.L.: [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560531), [code](https://github.com/EgorTrushin/Kaggle-Competitions/tree/main/Santa24), [notebook](https://www.kaggle.com/code/egortrushin/santa24-improving-sample-2)
- 11th, Ryota: [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560542)
- 13th, Honke Santa Master: [writeup](https://www.kaggle.com/competitions/santa-2024/discussion/560683)
- 16th, Santa Magister: [woosungyoon](https://www.kaggle.com/competitions/santa-2024/discussion/560541), [hikari30](https://www.kaggle.com/competitions/santa-2024/discussion/560674), [johnmichaelwu](https://www.kaggle.com/competitions/santa-2024/discussion/560713), [alphabetical-block explanation](https://www.kaggle.com/competitions/santa-2024/discussion/559339)
- 17th, just team: [atamazian](https://www.kaggle.com/competitions/santa-2024/discussion/560770)
- 18th, BeerMind + Denden + Gilles: [denden](https://www.kaggle.com/competitions/santa-2024/discussion/560550), [isakatsuyoshi](https://www.kaggle.com/competitions/santa-2024/discussion/560557), [litlit](https://www.kaggle.com/competitions/santa-2024/discussion/560564), [Gassano](https://www.kaggle.com/competitions/santa-2024/discussion/560567)
- Forum threads: [batching correctly](https://www.kaggle.com/competitions/santa-2024/discussion/548249), [brute forcing](https://www.kaggle.com/competitions/santa-2024/discussion/547564), [Quantum Santa](https://www.kaggle.com/competitions/santa-2024/discussion/549535), [sorting sample 6](https://www.kaggle.com/competitions/santa-2024/discussion/550287), [SDPA speed-up](https://www.kaggle.com/competitions/santa-2024/discussion/552346), [optimal scores revealed](https://www.kaggle.com/competitions/santa-2024/discussion/556784), [publishing top solutions](https://www.kaggle.com/competitions/santa-2024/discussion/555881)
