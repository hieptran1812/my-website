---
title: "AIMO Progress Prize 3: what a leaderboard of ties teaches"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "In AIMO 3 every documented top team ran the same open model from the same public notebook, ranks 1 to 4 tied at 44.0 and ranks 5 to 20 tied at 43.5. This is what the five published top-20 writeups did, how much of the order is noise, and what still transfers."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "aimo", "math-reasoning", "gpt-oss", "self-consistency", "tool-integrated-reasoning", "test-time-compute", "llm-inference", "evaluation-noise"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** AIMO Progress Prize 3 (November 2025 to April 2026) asked 3,450 teams to solve 50 hidden olympiad problems on one H100 in five hours, and scored the private set by running each notebook twice. Exalted Joseph took first with 44.0 out of 50, but so did ranks 2, 3 and 4, and ranks 5 to 20 all scored 43.5. Every one of the five teams that published a writeup ran `gpt-oss-120b` with no training, forked from the same public notebook. **Decisive lever: test-time compute, specifically inference engineering on a shared notebook.** **The one lesson:** when the field shares a backbone, the leaderboard measures run-to-run noise more than skill, so the useful knowledge is in the writeups that measured that noise, not in the rank order.

![The lever strip: of the five levers that decided Kaggle LLM competitions, AIMO 3 was decided by test-time compute, and above 43.5 the order is noise](/imgs/blogs/kaggle-aimo-progress-prize-3-solutions-1.webp)

The strip above is the mental model for this post. In the [prequel on AIMO Progress Prize 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions), the winner trained its own model and data construction decided first place. A year later the hardware became one big GPU, an open-weight 120B reasoning model became legal, and the competition collapsed onto one recipe. The question stopped being "what did the winner do differently" and became "how would you even know whether anyone did".

That is the theme of this post: the scorecard is mostly ties and missing writeups, so the value is in the shared backbone, the five published solutions, and the writeups that measured the noise floor. The series opener, [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work), explains the notebook format this post assumes.

## 1. The task in 60 seconds

![The AIMO 3 notebook budget card and an illustrative toy problem showing the out-of-range answer trap](/imgs/blogs/kaggle-aimo-progress-prize-3-solutions-2.webp)

AIMO 3 ran from 2025-11-20 to a final submission deadline of 2026-04-15. The host wrote 110 original problems in algebra, combinatorics, geometry and number theory, "from National Olympiad level all the way up to IMO standard", and designed them to be hard for the open LLMs of the day. Ten were reference problems released with full worked solutions, 50 were public, 50 were private. Your notebook receives one problem at a time through an evaluation API and returns an integer.

Two details changed from earlier rounds. Answers are integers from 0 to 99999 with any modulus stated explicitly (AIMO 1 and 2 used an implicit "mod 1000"), and Kaggle added H100 machines available only to AIMO 3 notebooks. The card in the figure is the whole budget: one H100 80 GB, a 5-hour GPU notebook, internet off, one submission a day, teams of up to 20, and 3,450 teams in the field (host recap).

The rules allowed open-weight models that are "reasonably accessible to all" and of "minimal cost", and required winners to release their solutions on HuggingFace and pass stricter reproducibility checks. That rule matters later. The pool was USD 2,207,152: USD 262,144, 131,072, 65,536, 32,768 and 16,384 for places 1 to 5, a USD 1,589,248 grand prize for 47 out of 50 on both sets (not awarded), and USD 110,000 of new extra prizes.

The toy problem in the figure is illustrative, not from the competition. It shows the failure the 2nd-place team found most often: the model solves the problem, then boxes the raw integer 123456789 instead of its remainder modulo 100000, which is 56789.

## 2. The metric, with a worked toy score

The public leaderboard was simply the number of public problems solved, out of 50. The private leaderboard used what the host called **AIMO 3 Multirun-Accuracy**, or penalized accuracy: the notebook runs twice over the 50 private problems, and each problem scores

$$
s_i = \tfrac{1}{2}\left(\mathbb{1}[\text{run 1 correct}] + \mathbb{1}[\text{run 2 correct}]\right) \in \{0,\ 0.5,\ 1\},
\qquad \text{score} = \sum_{i=1}^{50} s_i .
$$

A submission had to finish both reruns to be scored at all.

![Two-run penalized accuracy on five toy problems: both runs right scores 1, one run right scores 0.5, the total is 3.0 out of 5](/imgs/blogs/kaggle-aimo-progress-prize-3-solutions-3.webp)

On the toy above, P1 and P4 are solved in both runs (1 each), P2 and P5 in exactly one run (0.5 each), and P3 in neither. The total is ${1 + 0.5 + 0 + 1 + 0.5 = 3.0}$ out of 5.

Why bother running twice? Suppose a notebook solves problem $i$ with probability $p$ on any one run, independently. Then the two-run score has

$$
\mathbb{E}[s_i] = p^2 \cdot 1 + 2p(1-p)\cdot 0.5 = p,
\qquad
\operatorname{Var}[s_i] = \tfrac{1}{2}\,p(1-p).
$$

The expected score equals the single-run expectation; the metric only halves the variance per problem. The host knew temperature-1.0 sampling makes every notebook a random variable and paid for a second run to shrink the luck. But halving the variance shrinks the standard deviation by $\sqrt{2}$, not by two, and that arithmetic is the story of the final leaderboard (section 4).

## 3. What made it hard

Three things, in order of how much they decided.

**The problems.** Per the host, "ROLLER" was solved by no submission in both runs, even with the field's effective pass@N around 4,000; "ACUTES" was solved in both runs by exactly one.

**The clock.** TAMU-TACO (5th) did the arithmetic: ${(18{,}000 - 600) / 50 = 348}$ seconds per problem after a 600-second setup, while the fastest correct reference solve they saw took 212 seconds.

**The single dominant model.** OpenAI's `gpt-oss-120b` (the model card lists about 117B total and 5.1B active parameters; the 1st-place writeup says about 12B active, its own number) fit on one H100, ran in vLLM, used Python as a tool, and was hard to improve by training on anyone's budget. Once that is true, the competition becomes inference engineering on a shared stack, and the differences between entries shrink toward the noise floor. The prequel is the clearest contrast:

| | AIMO 2 (2024–25) | AIMO 3 (2025–26) |
|---|---|---|
| Teams | 2,212 | 3,450 |
| Hardware | 4x L4, 24 GB each (per the AIMO 2 post) | 1x H100, 80 GB |
| Answer format | integer mod 1000 | integer 0 to 99999, modulus explicit |
| Private metric | correct out of 50 | two-run penalized accuracy out of 50 |
| Top private score | 34 | 44.0 (four teams) |
| Dominant model | 4-bit DeepSeek-R1-Distill-Qwen-14B (12 of 16 writeups) | gpt-oss-120b (5 of 5 writeups) |
| Winner's edge | trained its own model on a dataset it built | no training; a public notebook |
| Grand prize | not awarded, rolled over | not awarded |

## 4. The top-20 scorecard

Public scores are not retrievable (the Kaggle API shows 0.0 public for every team after finalization), so there is no public-to-private move to report.

| Rank | Team | Private | Public-rank move | Key idea | Source |
|---|---|---|---|---|---|
| 1 | Exalted Joseph | 44.0 | unavailable | gpt-oss-120b, no training; the Parthenos "44/50 let me over cook" public notebook (commenters say unchanged v1) | [writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/1st-place-solution-for-the-aimo3-competition) |
| 2 | varianceofx | 44.0 | unavailable | Same base plus seven edits: answer-range prompt, tail-window entropy, vote-share-dominant scoring, uncatchable-leader stop | [writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/2nd-place-solution-ai-mathematical-olympiad-prog) |
| 3 | SKobayak | 44.0 | unavailable | ZaynYu baseline; prompt-only change from LLM-assisted analysis of 1,120 outputs | [writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/3rd-place-solution-for-the-aimo3-competition) |
| 4 | taku | 44.0 | unavailable | unknown; not among announced prize winners | no writeup |
| 5 | TAMU-TACO | 43.5 | unavailable | Inference policy tuned for robustness across problem orders by a human-in-the-loop experiment agent | [writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3-tamu-taco-solution) |
| 6 | Man Penguin | 43.5 | unavailable | unknown; not among announced prize winners | no writeup |
| 7 | yemao ye | 43.5 | unavailable | Andreas/Parthenos base, three-sentence prompt, simple inverse-entropy vote | [writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/7th-place-solution-for-the-aimo3-competition) |
| 8 | BarrryYang | 43.5 | unavailable | unknown | no writeup |
| 9 | Yew Jin Lim | 43.5 | unavailable | unknown | no writeup |
| 10 | Mehedi | 43.5 | unavailable | unknown | no writeup |
| 11 | Dylancelia | 43.5 | unavailable | unknown | no writeup |
| 12 | twotodevnull | 43.5 | unavailable | unknown | no writeup |
| 13 | stpete_ishii | 43.5 | unavailable | unknown | no writeup |
| 14 | Md. Hamid Hosen | 43.5 | unavailable | Only submission to solve problem ACUTES in both runs (Hardest Problem Prize) | no writeup |
| 15 | Abhimanyu Karki | 43.5 | unavailable | unknown | no writeup |
| 16 | Naji | 43.5 | unavailable | unknown | no writeup |
| 17 | Micheal1994 | 43.5 | unavailable | unknown | no writeup |
| 18 | Mert Kuzu | 43.5 | unavailable | unknown | no writeup |
| 19 | Greg | 43.5 | unavailable | unknown | no writeup |
| 20 | VelvetStorm | 43.5 | unavailable | unknown | no writeup |

Read the Private column top to bottom. It contains two numbers.

![The leaderboard of ties: ranks 1 to 4 at 44.0, ranks 5 to 20 at 43.5, ordered by submission time, with prize winners and the base notebook marked](/imgs/blogs/kaggle-aimo-progress-prize-3-solutions-4.webp)

Ranks 1 to 4 scored 44.0. Ranks 5 to 20 scored 43.5, and the tie continues past the cut (ranks 21 and 22 are also 43.5). Kaggle breaks ties by earlier submission, so the order inside each tie is the order people clicked submit. The 4th-to-5th gap is half a point: one problem solved in one of the two runs.

Three more facts, each reported only as far as the sources support:

- **Only 5 of 20 published**: ranks 1, 2, 3, 5 and 7, exactly the five announced prize winners.
- **Ranks 4 and 6 took no prize, and nobody has said why.** The winners announcement (2026-06-26, after a longer compliance check) cites stricter compliance, including the HuggingFace release, in general terms and names no reason for any team. Neither taku nor Man Penguin posted on the forum. Any explanation is a guess.
- **The 1st-place notebook may be an unchanged public notebook.** Two commenters on the 1st-place writeup said the submission is an exact copy of version 1 of Parthenos's public "[44/50] LET ME (over)COOK!!!" notebook. Asked where the described "adaptive runtime scheduling" lives, the author answered that it is a global time budget plus early stopping at four matching answers. He did not dispute the copy claim. We have not diffed the notebooks, so this is the commenters' claim, not fact. Either way, a public notebook run unchanged is plausibly enough to tie for first.

### What a leaderboard of ties actually measures

The 3rd-place team's mental model of the public set: about 35 always-solved problems, about 10 coin flips, about 5 never solved. Take it literally, with $p = 0.5$ per run on the coin flips, and compute what two identical notebooks would score.

```python
# Illustrative toy: two identical notebooks under SKobayak's mental model of the set.
# 35 always-solved problems, 10 coin flips (p = 0.5 per run), 5 never solved.
import math
from collections import Counter

def total_distribution(per_problem, n):
    """Distribution of the sum of n i.i.d. per-problem scores (in half-points)."""
    dist = {0: 1.0}
    for _ in range(n):
        new = Counter()
        for a, pa in dist.items():
            for b, pb in per_problem.items():
                new[a + b] += pa * pb
        dist = new
    return dist

one_run = {0: 0.5, 2: 0.5}               # single run: 0 or 1 point (2 half-points)
two_run = {0: 0.25, 1: 0.5, 2: 0.25}     # two runs: 0, 0.5 or 1 point

for name, per in [("one run", one_run), ("two runs", two_run)]:
    dist = total_distribution(per, 10)
    mean = sum(k / 2 * v for k, v in dist.items())
    sd = math.sqrt(sum((k / 2 - mean) ** 2 * v for k, v in dist.items()))
    p_same = sum(v * v for v in dist.values())   # two independent copies land on the same score
    print(f"{name:8s} mean {35 + mean:.1f}  sd {sd:.3f}  P(identical notebooks tie) {p_same:.3f}")
```

```console
one run  mean 40.0  sd 1.581  P(identical notebooks tie) 0.176
two runs mean 40.0  sd 1.118  P(identical notebooks tie) 0.125
```

The two-run metric does exactly what the algebra promised: same mean, standard deviation down from 1.58 to 1.12 points. But 1.12 points is still more than twice the half-point that separated the four teams at 44.0 from the sixteen at 43.5. And two copies of the same notebook land on the identical score only 12.5% of the time in this toy. A cluster of 20 teams on two adjacent scores is what you would expect from a field running variants of one notebook whose true quality differs by less than the noise.

![Run-to-run noise against the half-point gap: the same notebook spans several points across submissions, while ranks 4 and 5 differ by 0.5](/imgs/blogs/kaggle-aimo-progress-prize-3-solutions-5.webp)

The measured numbers agree. SKobayak (3rd) submitted the same final notebook three times and scored 36, 37 and 39 public, then 44 private. yemao ye (7th) scored 36 to 40 public over six submissions of one configuration. Natapong Nitarach measured a 4.5-point public/private swing on an identical configuration, within three standard deviations of his run-to-run variance. The shared base notebook scored 41.5 private for its own author (546th) while minor forks reached 44. The host put it in one sentence: "the main AIMO3 competition ended up being dominated by one notebook."

The top 20 are a shared notebook plus noise. What you can learn is which changes the five published teams made, which they could measure, and which they found useless.

## 5. The top-20 landscape

![Top-20 technique matrix: only ranks 1, 2, 3, 5 and 7 have writeups, every other column is unknown; the shared backbone rows are filled across all five](/imgs/blogs/kaggle-aimo-progress-prize-3-solutions-6.webp)

Fifteen of the 20 columns are question marks: that is the information content of the top 20. Counted over the five with writeups:

- 5 of 5 ran `gpt-oss-120b`; 0 of 5 trained or fine-tuned anything.
- 5 of 5 used a Python tool through persistent Jupyter kernels, 8 parallel attempts, and early stopping when enough attempts agreed.
- 4 of 5 used an entropy-weighted vote (SKobayak used plain majority).
- 4 of 5 used an FP8 or 8-bit KV cache (SKobayak's writeup does not say).
- 4 of 5 explicitly built on a named public notebook (TAMU-TACO describes the same stack without crediting one by name).
- 3 of 5 changed the prompt from their base, and two of those went in opposite directions.
- 2 of 5 built a serious offline validation harness; 2 of 5 had an explicit per-problem timeout policy.

**Table stakes**: `gpt-oss-120b` in vLLM on one H100, FP8 KV cache, about 65,536 tokens of context, 8 tool-using attempts at temperature 1.0, stop at 4 agreeing answers, extract the `\boxed{}` integer. **What separated the top 5 from 6 to 20**: nothing measurable; half a point is inside the noise. **Contradictory findings** show where the noise hides the answer:

| Question | Who said what |
|---|---|
| Should the prompt be longer? | varianceofx (2nd) added an answer-range block and a checklist and calls it the highest-leverage change. SKobayak (3rd) added constraint-checking and complexity-estimation instructions from error analysis. yemao ye (7th) found long structured prompts unstable and cut to three sentences. Exalted Joseph (1st) tried Gemini-generated prompts and kept the original. |
| How should entropy enter the vote? | varianceofx: a pure sum of inverse entropy can be hijacked by one confident wrong attempt, so vote share must dominate. yemao ye: a more complex entropy metric did no better than plain inverse entropy. SKobayak: plain majority. TAMU-TACO: inverse entropy only as a fallback when 4 answers never agree. |
| More than 8 attempts? | SKobayak found 16 solvers worse than 8. TAMU-TACO capped at 8 for memory. varianceofx found more than 8 exceeded the time budget. |

## 6. The winning solution, explained: the shared backbone

Since the 1st-place notebook is, by commenters' account, the Parthenos notebook, and the 2nd and 7th places also build on it, the right thing to explain is the backbone itself. It is Andreas Bisiadis's public "GPT-OSS-120B (with tools)" notebook, which passed 1,000 upvotes and which the host called the foundation of many top notebooks. Parthenos's derivative added a five-step system prompt (UNDERSTAND, EXPLORE, PLAN, EXECUTE, VERIFY).

**The intuition.** Think of eight students sitting the same exam problem in parallel, each with a calculator and scratch paper they can keep using. You do not grade any one of them. You wait until four hand in the same answer and accept it, then send them all home and give the leftover time to the next problem. If they never agree, you trust the students who sounded most sure of themselves.

![The shared backbone as eight lanes: a cached prompt prefix feeds eight tool-using attempts, four agree on 512 at 240 s and the rest are cancelled](/imgs/blogs/kaggle-aimo-progress-prize-3-solutions-7.webp)

**The mechanism**, piece by piece, from the Andreas writeup:

1. **Model and server.** `gpt-oss-120b` in vLLM 0.11.2, Harmony format, high reasoning effort, 65,536-token context, async scheduling.
2. **Memory.** FP8 (E4M3) KV cache, which lets eight long traces share one GPU; see [KV-cache quantization and its accuracy cliff](/blog/machine-learning/inference-engineering/kv-cache-quantization-fp8-int8-and-the-accuracy-cliff).
3. **Prefix caching.** All attempts share the prompt, so its KV blocks are computed once; see [prefix caching and RadixAttention](/blog/machine-learning/model-serving/prefix-caching-and-radixattention).
4. **Sampling.** Up to 8 attempts, seeds $(42 + i)^2$, temperature 1.0, `min_p` 0.02.
5. **The tool.** Python in a persistent Jupyter kernel per attempt, pool overprovisioned two to one, with port retry and output truncation, because a hung kernel kills an attempt.
6. **Selection.** Extract the `\boxed{}` integer; stop when 4 attempts agree; otherwise weight answers by inverse mean token entropy.
7. **Time.** A global budget: early convergence leaves time for later problems.

Andreas's conclusion is the one to remember: inference-system reliability beat extra reasoning machinery. None of it improves the model; all of it lets the model's existing ability show up within the clock. See our [gpt-oss timeline](/blog/machine-learning/frontier-models/openai-gpt-oss-open-weight-models-timeline) and the [vLLM deep dive](/blog/machine-learning/inference-frameworks/vllm-deep-dive).

**A toy version** of the selection rule follows. It is illustrative, not any team's code.

```python
# Illustrative toy, not any team's code: the selection rule of the shared backbone.
from collections import Counter, defaultdict

def select(attempts, stop_at=4):
    """attempts: list of (finish_time_s, answer, mean_token_entropy) in finish order."""
    seen = Counter()
    for t, ans, _ in sorted(attempts):
        seen[ans] += 1
        if seen[ans] >= stop_at:            # early stop: k answers agree
            return ans, f"early stop at {t} s"
    score = defaultdict(float)              # no agreement: entropy-weighted vote
    for _, ans, h in attempts:
        score[ans] += 1.0 / h               # confident (low-entropy) attempts weigh more
    best = max(score, key=score.get)
    return best, {a: round(s, 2) for a, s in score.items()}

# Run 1: the toy timeline from the figure. Four attempts agree on 512 at 240 s.
run1 = [(140, 512, 0.8), (170, 512, 0.9), (190, 740, 0.6), (210, 512, 1.0),
        (240, 512, 0.7), (260, 512, 0.8), (280, 33, 1.2), (300, 512, 0.9)]
print(select(run1))

# Run 2: three attempts agree on 512, one very confident attempt says 740.
run2 = [(150, 512, 0.9), (200, 512, 1.0), (230, 512, 1.1), (120, 740, 0.2),
        (300, 88, 1.3), (300, 91, 1.4), (300, 17, 1.5), (300, 64, 1.6)]
print(select(run2))
print("majority:", Counter(a for _, a, _ in run2).most_common(1))
```

```console
(512, 'early stop at 240 s')
(740, {512: 3.02, 740: 5.0, 88: 0.77, 91: 0.71, 17: 0.67, 64: 0.62})
majority: [(512, 3)]
```

Run 1 is the figure's happy path: four agree at 240 seconds and the rest are cancelled. Run 2 is the failure the 2nd-place team wrote about: three attempts on 512 sum to 3.02, one attempt very sure of 740 (entropy 0.2) contributes 5.0 alone, so the inverse-entropy sum picks 740 while majority picks 512. Whether that confident attempt is right is exactly what entropy cannot tell you.

## 7. 2nd and 3rd in depth, then the rest of the top 20

### 2nd: varianceofx (44.0)

varianceofx made seven targeted changes to the Parthenos base. Pure inference: 8 attempts, 16 Jupyter kernels, FP8 KV, about 300 to 400 seconds per problem, 4 to 5 hours on one H100. The selected submission was notebook version 10; later versions (one scored 35) were experiments.

1. **An answer-range block in the prompt**: the answer is an integer in [0, 99999], and a larger raw answer means the problem wants something modular. Plus a pre-boxing checklist. He calls this the highest-leverage change: the model often solved the problem, then boxed the unreduced integer.
2. **A one-line library hint** instead of about 200 tokens.
3. **Entropy over the last 256 tokens only**, where the model commits.
4. **Only `\boxed{}` extraction**; the "final answer is N" fallback fired on intermediate steps.
5. **Vote share dominates**: 2.0 times vote share, plus 0.3 times inverse median entropy, plus 0.5 times a consensus bonus, replacing the hijackable sum from run 2.
6. **Median instead of mean** entropy per answer.
7. **"Uncatchable leader" stop**: stop when second place plus remaining attempts cannot overtake the leader.

The first change is the one that transfers: an output-contract fix, the cheapest points in any LLM competition because the model already did the hard part.

### 3rd: SKobayak (44.0)

SKobayak started from ZaynYu's public "40/50 gpt-oss-120b TIR DynamicTime KernelPool" notebook: the model as-is with a Python tool, 8 parallel solvers, majority vote, baseline early stopping. He confirmed in the comments that his final notebook changed **only the prompt**.

Instead of chasing a 50-problem leaderboard, he built an offline set of 1,120 full outputs: 32 runs over the 10 reference problems plus 16 runs over 50 external hard problems shared by participants. A stronger LLM categorized the failures, fed with examples of `gpt-oss`'s own correct answers so its advice stayed within `gpt-oss`'s reach. The resulting prompt tells the model to use Python for verification, to extract and check every explicit constraint, and to narrow the search space and estimate complexity before writing loops, because brute-force loops were timing out.

With 16 solvers instead of 8, early finishers were less precise and the net was more errors. His public scores of 36, 37 and 39 for one notebook are the cleanest noise measurement anyone published.

### Ranks 4 to 20, by approach family

**Robustness-tuned inference policy: TAMU-TACO (5th, 43.5).** Fangzhou Lin and Peiran Li framed AIMO 3 as a sequential resource-allocation problem: one hidden problem at a time, a fixed clock, no knowledge of what comes next. Their claim is that private robustness depends mainly on where the hard problems fall in the order and on four coupled knobs: base timeout, high timeout, attempts, and early-stop count. Section 8 covers how they tuned them; the final values were 300 seconds, 900 seconds, 8 and 4. No fine-tuning: "the larger gain came from configuring inference".

**Deliberate simplification: yemao ye (7th, 43.5).** Built on the Andreas and Parthenos notebooks, tried the five-step prompt with checklists, and found that longer prompts "did not always give a stable advantage", used more context and made reasoning less direct. The final system prompt is three sentences: a world-class IMO competitor, a non-negative integer answer in [0, 99999], put it in `\boxed{}`. His aggregation ablation is the most honest table in the competition: simple inverse entropy over 6 public submissions scored 36 to 40, a complex entropy metric (position-weighted, variance, high-entropy penalty, low-entropy streak reward) over 3 submissions scored 36 to 39. He kept the simple one because it has fewer hand-set hyperparameters and token entropy is only a rough confidence signal that a complex transform may amplify as noise. FP8 KV, up to 81,920 context, 8 attempts, 16 kernels.

**No writeup, a host note: Md. Hamid Hosen (14th, 43.5).** His submission was the only one in the whole competition to solve "ACUTES" in both evaluation runs, which won the USD 30,000 Hardest Problem Prize. The host adds that ACUTES was not necessarily the hardest problem, since ROLLER was solved by nobody in both runs. Whether his notebook differs from the shared backbone is unknown.

**No writeup: taku (4th), Man Penguin (6th), and ranks 8 to 13 and 15 to 20** (BarrryYang, Yew Jin Lim, Mehedi, Dylancelia, twotodevnull, stpete_ishii, Abhimanyu Karki, Naji, Micheal1994, Mert Kuzu, Greg, VelvetStorm). None posted on the forum. Given the host's remark they are plausibly forks of the shared backbone, but that is unverified for each, so their matrix columns stay question marks.

### The best contrasting entries, off the board

The two USD 15,000 writeup prizes went to teams outside the top 20 that did the science the leaderboard could not.

**"Model Capability Dominates" (Natapong Nitarach).** Across 3 models and more than 23 experiments, diverse per-voter prompts never beat the baseline. He models why: with $N$ voters whose errors have pairwise correlation $\rho$, the effective number of independent voters is

$$
N_{\text{eff}} = \frac{N}{1 + (N-1)\,\rho},
$$

so with $N = 8$ and $\rho = 0.5$ you get ${8 / 4.5 \approx 1.8}$ effective voters, and temperature-1.0 sampling already supplies most of the decorrelation that prompt diversity could add. He measured a capability gap between models of about 8 points against about plus or minus 2 from prompts. His best run scored 42 and his mean over 21 runs was 39.3, against a pass@20 of about 45.5; the gap between the two is selection loss, and 82 to 86% of the unrecovered cases lost by a single vote. His private score was 40.5.

**"A Collection of Delightful Side Quests" (Chan Kha Vu and Geremie Yeo).** They tried to leave the backbone: FP8 and NVFP4 quantizations of Nemotron-Cascade-2-30B-A3B, and a vLLM fork where prefix caching and speculative decoding coexist on a hybrid Mamba model, after fixing "mamba cache poisoning" with per-request ghost states. Their EAGLE-3 speculator averaged 3.55 accepted tokens at K = 7.

## 8. What the forum taught everyone

![TAMU-TACO's two budgets: 18,000 seconds split into setup plus 50 slices of 348 s, and four placements of the 30 hard problems in a 100-problem local set](/imgs/blogs/kaggle-aimo-progress-prize-3-solutions-8.webp)

**TAMU-TACO's harness is the method worth copying.** Their 100-problem local set used the 10 reference problems as difficulty anchors (solved by GPT-5 at high effort at pass@1 = basic, only at pass@3 = harder, unsolved = hardest), with candidates from AIMO 2, OpenMathReasoning and Putnam-AXIOM screened by GPT-5 and then humans: 70 basic plus 30 hard. Every setting ran with the 30 hard problems at the front, middle, back and random, three repeats each, so 12 full runs per setting.

Their "AutoResearchAgent" ran the suites, logged correctness, time, entropy, early stops, timeouts, out-of-memory events and hardware use, had GPT-5 summarise, and proposed the next setting inside human-set bounds. The bounds came from the clock (348 seconds average, 212 seconds fastest correct reference solve) and memory (68.6 to 74.2 GB for 8-way inference at 16k to 32k tokens per path). Phase 1 (6 attempts, stop at 3) tuned timeouts over 46 rounds, 552 runs, finding a stable region around 894 to 906 seconds high and 288 to 306 base, locked at 900 and 300. Phase 2 found the best score-variance trade at 8 attempts and stop at 4 over 18 rounds. That is up to 768 full runs on 64 H100s of their own, as stated in their post. The lesson is not "use 900 and 300"; it is to optimize for robustness over orderings, not peak score.

**Other threads.** ippeiogawa's "Blackboard and Fork" shares a ring buffer of tool calls across parallel rollouts. IceBeam777's recursive self-aggregation reports a public mean of 38 to 41 with lower variance. Utkarsh Sharma's "Every Improvement Failed" collects negative results (not read in detail here). The Math Corpus Prize went to Yi-Chia Chen's CrystalMath, with NGUYEN's AstralMath (about 431k tool-use traces) as runner-up.

## 9. What did not work

Aggregated across the five top-20 writeups and the two writeup-prize winners, with who reported it:

| Tried | Result | Reported by |
|---|---|---|
| More than 8 attempts (16 solvers) | worse or did not fit the clock | SKobayak, varianceofx; TAMU-TACO capped at 8 for memory |
| Lower temperature | less diversity, worse votes | varianceofx |
| Different prompts per attempt | no gain over baseline | varianceofx, Nitarach (3 models, 23+ experiments) |
| Fine-tuning | not attempted; inference gave more per hour | varianceofx, TAMU-TACO; no documented top team trained |
| Long structured prompts with checklists | unstable, more context, less direct reasoning | yemao ye |
| Complex entropy metric | 36 to 39 versus 36 to 40 for simple inverse entropy | yemao ye |
| Reasoning in Japanese | worse | SKobayak |
| Draft-then-finalize self-correction | worse | SKobayak |
| Complexity-consistency prompts | worse | SKobayak |
| Parallel-computation and voting changes on the baseline | score appeared to fall | SKobayak |
| Longer per-problem budgets, aggressive early stopping | eat time early; lock in repeated wrong answers | yemao ye; TAMU-TACO's stress tests found the 300/900 region robust instead |
| Tuning to the public leaderboard or one problem order | overfits the coin-flip problems | TAMU-TACO, SKobayak |

Almost every attempt to add reasoning machinery did nothing measurable or made things worse. What survived: output-contract fixes, more honest confidence signals, and time allocation.

## 10. Lessons that transfer

1. **When everyone forks one notebook, the leaderboard measures noise.** Ranks 1 to 4 tied, ranks 5 to 20 tied, and one team's identical notebook spanned 36 to 39 publicly. Before you read meaning into a rank, compute the run-to-run standard deviation of a single configuration. (Section 4; SKobayak and Nitarach writeups; host winners post.)
2. **Fix the output contract before the reasoning.** The answer-range reminder recovered problems the model had already solved. (varianceofx, 2nd.)
3. **Use confidence as a tiebreaker, not the primary vote, and measure it where the model commits.** A single over-confident attempt can hijack an entropy sum; tail-window entropy and vote-share-dominant scoring are cheap guards. (varianceofx, 2nd; yemao ye, 7th.)
4. **Validate offline on many runs of many hard problems, and read the failures.** 1,120 logged outputs told SKobayak more than 50 public problems at one submission a day could. (SKobayak, 3rd; TAMU-TACO, 5th.)
5. **Prompt diversity does not decorrelate errors beyond what temperature already does.** The remaining loss is selection, and most of it is cases lost by one vote. (Nitarach, arXiv 2603.27844.)
6. **Tune for the worst plausible ordering, not the average one.** In a sequential, clock-bound evaluation, where the hard problems land matters as much as how many there are. (TAMU-TACO.)

### If you enter a competition like this

- Run your baseline five times first; any change smaller than that spread is unmeasured.
- Log every output on a local set that includes failures; fix format and range errors first.
- Make the stack boring: persistent kernels with retries, FP8 KV, prefix caching, a global time pool.
- Use the simplest vote that guards against one confident outlier.
- Stress-test problem order, especially hard problems at the front.
- Read the release and reproducibility rules early; two of the top six took no prize, for unpublished reasons.

## Sources

- Competition page and overview: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3
- 1st place, Exalted Joseph: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/1st-place-solution-for-the-aimo3-competition
- 2nd place, varianceofx: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/2nd-place-solution-ai-mathematical-olympiad-prog ; notebook https://www.kaggle.com/code/varianceofx/let-me-improve-this-cooking ; model card https://huggingface.co/varianceofx/aimo3-2nd-place-solver-adapted
- 3rd place, SKobayak: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/3rd-place-solution-for-the-aimo3-competition ; base notebook https://www.kaggle.com/code/zaynyu/40-50-gpt-oss-120b-tir-dynamictime-kernelpool
- 5th place, TAMU-TACO: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3-tamu-taco-solution ; space https://huggingface.co/spaces/peiranli0930/AIMO3-TAMU-TACO
- 7th place, yemao ye: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/7th-place-solution-for-the-aimo3-competition
- Shared backbone, Andreas Bisiadis: notebook https://www.kaggle.com/code/andreasbis/aimo-3-gpt-oss-120b-with-tools ; writeup https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/gpt-oss-120b-with-tools-technical-writeup
- Parthenos derivative: https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook
- Prompt reference cited by 1st place: https://www.kaggle.com/code/datasciencegrad/aimo-3-42-50-stable-lb-possible-43-luck
- Natapong Nitarach, "Model Capability Dominates": https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/model-capability-dominates (arXiv 2603.27844)
- Chan Kha Vu and Geremie Yeo, "A Collection of Delightful Side Quests": https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3-a-collection-of-delightful-side-quests
- ippeiogawa, "Blackboard and Fork": https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/aimo3test
- IceBeam777, "Recursive Self Aggregation with Difficulty Aware Scheduler": https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/recursive-self-aggregation-with-di
- Utkarsh Sharma, "Every Improvement Failed": https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/every-improvement-failed-aimo3
- Tong Hui Kang, annotated solution traces: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/672528
- External hard problems shared by participants: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/672643
- Host recap, extra prizes and winners: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/702188 , https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/708484 , https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/714435
- AIMO update on the shrinking gap: https://aimoprize.com/updates/2025-09-05-the-gap-is-shrinking
- Prequel: [AIMO Progress Prize 2: every published solution](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions). Sibling posts in this wave: [AIMO Progress Prize 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions), [Eedi Mining Misconceptions](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions), [MAP Charting Student Math Misunderstandings](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions).
