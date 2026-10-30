---
title: "AIMO Progress Prize 1: how Numina won with tool-integrated reasoning on two T4s"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A synthesis of every published top-20 writeup from Kaggle's first AI Mathematical Olympiad prize: why all of them sampled, ran Python and voted, how Numina's two-stage fine-tuning and SC-TIR reached 29 of 50, how CMU's reward model and two inference-only teams got to 22 and 21, and why 16 of the top 20 published nothing."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "math-reasoning", "tool-integrated-reasoning", "self-consistency", "test-time-compute", "reward-model", "deepseekmath", "vllm", "gptq"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** Kaggle's first AI Mathematical Olympiad Progress Prize (April to June 2024, 1,161 teams, a prize fund of USD 1,048,576) asked an offline notebook on two T4 GPUs to answer 50 new olympiad-style problems in nine hours, each answer an integer from 0 to 999. Team **Numina** won with 29 of 50 by fine-tuning DeepSeekMath-7B-Base in two stages, ending on tool-integrated reasoning traces, and then sampling 48 copies of each problem that write Python, execute it, read the output and vote. Every one of the four top-20 teams that published a writeup used the same backbone family, the same Python-in-the-loop and the same vote; the two that fine-tuned took first and second. **The lesson:** test-time compute was the price of entry. Training data was the difference.

![The lever strip: AIMO Progress Prize 1 was decided on test-time compute, with training data separating first place from the pack](/imgs/blogs/kaggle-aimo-progress-prize-1-solutions-1.webp)

The strip above is the mental model for this series, set up in [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work): five levers, one lit per competition. AIMO 1 is where the series turns from retrieval (the [LLM Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions) story) to test-time compute. It also has an unusual shape for a top-20 synthesis: **only four of the top 20 teams published a writeup.** Ranks 6 to 20 published nothing, and rank 5 explained in a forum comment that its submission was an unmodified fork of a public notebook. So this post goes deep on ranks 1 to 4, says plainly what is unknown about the other sixteen, and treats the rank-5 fork as the competition's most useful control experiment.

## The task in 60 seconds

The host released 110 new problems written by an international team, pitched "a bit easier than AIME, between AIME and AMC 12". They were text only, in LaTeX, and every answer was a non-negative integer reported modulo 1000. Ten problems were public training examples, 50 formed the public leaderboard and 50 the private one, with no overlap. During the competition your notebook only ever saw the public 50; the private 50 were served once, at the final rerun. Problems arrived **one at a time, in random order**, through a Python evaluation API, so you could not look ahead, sort by difficulty or batch them.

![The notebook budget: two T4 GPUs or one P100, at most nine hours, internet off, 50 problems served one at a time in random order, about 10.8 minutes per problem on average](/imgs/blogs/kaggle-aimo-progress-prize-1-solutions-2.webp)

The budget card above is the whole constraint. Two T4s (or one P100), nine hours per rerun with up to 30 minutes of deliberate runtime obfuscation, no internet, and open pre-trained models allowed. Nine hours over 50 problems is 540 minutes divided by 50, or **10.8 minutes per problem on average**, and that average hides the problem: you do not know in advance which problems are cheap. Teams got two submissions a day. Places 1 to 5 paid USD 131,072, 65,536, 32,768, 16,384 and 8,192. An Overall Progress Prize of at least USD 794,624 waited for the first team to reach 47 of 50 on both sets; nobody came close, and it rolled over. An Early Sharing Prize of USD 10,000 went to the first public notebook to reach 20 of 50 before 22 April 2024; the community credits Abdur Rafae's "Improved Code Interpretation" notebook with it, and that notebook matters more than any other artifact in this post.

The data is not ours to reproduce, so here is an illustrative problem in the same style and the kind of trace every top team wanted its model to write.

![An illustrative problem in the competition's style and a tool-integrated reasoning trace: a sentence of reasoning, a Python block, the executed output, and the boxed answer](/imgs/blogs/kaggle-aimo-progress-prize-1-solutions-3.webp)

Notice the four parts: a sentence of plan, a short program, the interpreter's output pasted back into the context, and a `\boxed{}` answer. This is **tool-integrated reasoning** (TIR): the model reasons, Python counts. Almost every idea in this competition is about getting more traces of this shape, getting better ones, or choosing among them.

## The metric: correct answers out of 50

The score is plain exact-match accuracy, shown as a count:

$$
\text{score} = \sum_{i=1}^{50} \mathbf{1}\left[\hat{a}_i = a_i\right], \qquad \hat{a}_i, a_i \in \{0, 1, \dots, 999\}
$$

where $a_i$ is the true answer to problem $i$ and $\hat{a}_i$ is the integer your notebook returned. There is no partial credit and no probability; a wrong integer and a crash both score zero. The host's reference point was Gemma 7B at 3 of 50.

![Accuracy on the 50 private problems as filled cells: the host's Gemma 7B baseline solved 3, rank 20 solved 18, an unmodified public notebook fork 20, CMU_MATH 22 and Numina 29, far short of the 47 needed for the grand prize](/imgs/blogs/kaggle-aimo-progress-prize-1-solutions-4.webp)

Read the cells as a worked score: Numina's 29 is 29 right and 21 wrong. The striking row is rank 5, an **unmodified fork of a public notebook** at 20, only two below second place. A metric this coarse, on 50 items, has a large noise floor. One problem is two percentage points, and the fourth-place writeup reports that the random seed alone moved its public score by about 3 problems. Eight teams tied at 20 and seven at 19, and Kaggle breaks ties by earlier submission, so ranks 5 to 12 are not a skill ordering.

## What made it hard

Three things, and the third is the one that decided the winner.

**Old, small GPUs.** The T4 has no bf16, no FlashAttention-2 and no working `torch.compile` path, so every speed trick of mid-2024 was off the table. Numina found that static KV cache plus `torch.compile` gave 2 to 3 times faster generation on an H100 and broke on the T4. It also fixed the model: [DeepSeekMath-7B](/blog/machine-learning/frontier-models/deepseekmath-data-pipeline-and-grpo-origin) was the strongest open math model that fit. Numina tried InternLM-20B, CodeLlama-33B and Mixtral-8x7B and dropped them as slow, with "mysterious timeouts".

**Random order and a fixed clock.** You cannot budget by difficulty in advance. Spend too long on an early hard problem and the easy ones at the end starve; stop early and you lose problems you could have solved. The top teams disagreed on what to do about it.

**A 50-problem leaderboard you cannot trust.** With seed variance near 3 problems, a public improvement of 2 is indistinguishable from luck. Numina's writeup puts it bluntly, "good validation sets are all you need", and the team that took this most seriously won by seven problems.

## The top-20 scorecard

Public scores were wiped from the Kaggle API after the competition finalized (every team shows 0), so **public rank is unavailable for every team**, and the column is blank on purpose. Scores are private, out of 50.

| Rank | Team | Private | Public-rank move | Key idea | Source |
|---|---|---|---|---|---|
| 1 | Numina | 29 | unavailable | Two-stage full SFT of DeepSeekMath-7B-Base (CoT, then ToRA-style TIR), 8-bit GPTQ, SC-TIR with N=48, M=4 | [writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/numina-numina-1st-place-solution) |
| 2 | CMU_MATH | 22 | unavailable | Fine-tuned DeepSeekMath-7B-RL policy plus a 7B outcome reward model, geometric-mean weighted vote | [writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/cmu-math-2nd-place-solution-all-code-and-datasets-) |
| 3 | after exams | 21 | unavailable | No fine-tuning; DeepSeekMath-7B-RL in vLLM, 120 to 160 samples, penalty on small numbers and numbers from the statement | [writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/after-exams-3rd-place-solution) |
| 4 | codeinter | 21 | unavailable | One model copy per T4, adaptive TimeManager, 3 attempts for `[asy]` geometry, small-integer down-weighting | [writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/codeinter-4th-place-solution) |
| 5 | Conor #2 | 20 | unavailable | Unmodified fork of the Early Sharing Prize notebook; prize declined | [comment](https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/523125) |
| 6 | TheEighthDay | 20 | unavailable | unknown | no writeup |
| 7 | [Deleted account] | 20 | unavailable | unknown (account deleted) | no writeup |
| 8 | Vaishnavi Mudaliar | 20 | unavailable | unknown | no writeup |
| 9 | kaiwu | 20 | unavailable | unknown | no writeup |
| 10 | curiousgeorge | 20 | unavailable | unknown | no writeup |
| 11 | Hoang Tu Anh | 20 | unavailable | unknown | no writeup |
| 12 | Quickpanda | 20 | unavailable | unknown | no writeup |
| 13 | zby | 19 | unavailable | unknown | no writeup |
| 14 | aigood | 19 | unavailable | unknown | no writeup |
| 15 | ZhujunXu | 19 | unavailable | unknown | no writeup |
| 16 | BU123 | 19 | unavailable | unknown | no writeup |
| 17 | Practice makes perfect (Abdur Rafae) | 19 | unavailable | Author of the baseline notebook; final configuration undocumented | no writeup |
| 18 | rl12 | 19 | unavailable | unknown | no writeup |
| 19 | Dileep Jayamal | 19 | unavailable | unknown | no writeup |
| 20 | aha | 18 | unavailable | unknown | no writeup |

The 18-point tie continues past the cut to ranks 21 and 22. The sweep behind this table paged all 388 discussion topics by title and by every top-22 member's username and found writeups for ranks 1 to 4 only. A few exist further down (36th, 41st, 50th, 53rd, 56th, 96th) and are not used here.

## The top-20 landscape

Sixteen of the top 20 published nothing, so any "N of 20" statistic would be invented. The honest denominator is **4 teams with writeups**, plus what rank 5 told us in a comment. The matrix below draws ranks 6 to 20 anyway, as grey unknowns, because the absence is itself a finding: in this competition, the leaderboard tells you what scored and almost nothing about why.

![Technique matrix for the top 20: rows are techniques, columns are private ranks 1 to 20; ranks 1 to 4 have writeups, rank 5 explained itself in a comment, and ranks 6 to 20 are all unknown](/imgs/blogs/kaggle-aimo-progress-prize-1-solutions-5.webp)

Counting from the cells, with the four writeups as the denominator: 4 of 4 used a DeepSeekMath-7B backbone, Python in the loop and a vote; 3 of 4 built on Abdur Rafae's notebook, kept an offline validation set, down-weighted 0 or small answers and ran vLLM; 2 of 4 fine-tuned, both on GPT-4 generated solutions; 1 of 4 each used a learned reward model (CMU_MATH), 8-bit quantisation (Numina), a statement-number penalty (after exams) and an adaptive time budget (codeinter).

**Table stakes.** DeepSeekMath-7B, Python in the loop, many samples, vote. Every documented entry did all four, and rank 5 is direct evidence of how far that alone goes: the public baseline, unchanged, landed at 20.

**What separated the top 2 from everyone else: training.** Only Numina and CMU_MATH fine-tuned, both on solutions generated by GPT-4. Ranks 3 to 5, at 21, 21 and 20, were inference-only on the same baseline. The gap from 20 or 21 to 29 is the clearest evidence in the competition that better reasoning traces in the weights beat cleverer selection at inference time.

**Where the writeups contradict each other.**

- *Time management.* codeinter built an adaptive `TimeManager`; Numina deliberately fixed the width and depth of search because the time-budget tricks it tried gave high variance. Both reached the money.
- *More samples.* codeinter's design maximised samples per problem (21 to 30 to 40). after exams reported that more generations had stopped improving validation.
- *vLLM.* The top three used it. Outside the top 20, Geremie Yeo could not get vLLM past 13 and hit the same "mysterious timeouts" Numina described.

## The winning solution: Numina

Numina (Jia Li, Yann Fleureau, Guillaume Lample, Stan Polu and Hélène Evain, with Lewis Tunstall and Edward Beeching of Hugging Face) wrote the most complete writeup of the competition, published a Hugging Face blog, the notebook, the model (`AI-MO/NuminaMath-7B-TIR` and its GPTQ version), the validation sets and the dataset. The pipeline is short to describe and expensive to build.

![Numina's pipeline: DeepSeekMath-7B-Base, a chain-of-thought SFT stage, a tool-integrated SFT stage on GPT-4 traces, 8-bit GPTQ, SC-TIR with 48 samples and 4 rounds, and a majority vote; the public score climbed from 8 to 16 to 27 as the data changed](/imgs/blogs/kaggle-aimo-progress-prize-1-solutions-6.webp)

### Two stages of fine-tuning

The recipe follows MuMath-Code: teach the model to reason in prose first, then to reason with a tool.

- **Stage 1, chain of thought.** A large, diverse set of natural-language math problems with solutions templated as chain of thought, "several hundred thousand problems" per the Hugging Face blog. Learning rate 2e-5, batch 32, block size 2048, 3 epochs, cosine schedule, 0.1 warmup.
- **Stage 2, tool-integrated reasoning.** Synthetic traces in the ToRA format: GPT-4 was prompted to write a rationale, then Python, then read the execution output, and repeat. About 60k problems with numerical outputs. Same learning rate and batch, block size 1024, 4 epochs.

Both stages were **full fine-tuning, not LoRA**, with TRL's `SFTTrainer` (packing, gradient checkpointing, DeepSpeed ZeRO-3) on one node of 8 H100s, about 10 hours in total. In the comments Jia Li puts the whole corpus at "around 1M samples". AIME 1983 to 2021 was in training; AIME 2022 to 2024 was held out.

The score trajectory is the most instructive number sequence in the competition. A Stage-1-only chain-of-thought model scored **8** of 50 on the public board (majority over 32 samples). Stage-2 data from MMOS, one program per solution, took it to **16**. Switching to multi-step ToRA traces, where the model writes code, reads the output and writes more code, jumped it to **27**; Jia Li confirms in the comments that the format switch was the jump. The final private score was 29. Same base, same hardware, same sampling; what moved 11 problems was the shape of the training traces. And against intuition, filtering Stage 2 to integer answers only was slightly worse than keeping every solution type.

Numina also tried RL. On-policy KTO (4 completions per Stage-2 prompt, labelled by answer correctness) gave a few points internally and 27 on public, but missed the deadline. RLOO with a code-execution reward gave no significant gain and was slow.

### Fitting it on two T4s

A 7B model in 16-bit is a tight fit on a pair of 16 GB T4s once you add a KV cache for dozens of parallel samples. Numina quantised to **8-bit with GPTQ** (AutoGPTQ, calibrated on their training data; background in the [weight-only quantization post](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq)). The writeup reports a small accuracy drop that was more than paid back by fitting more samples, and a practical bonus: uploading the model to Kaggle was twice as fast. Generation ran in [vLLM](/blog/machine-learning/inference-frameworks/vllm-deep-dive).

### SC-TIR: self-consistency with tool-integrated reasoning

At inference, the algorithm is called SC-TIR:

1. Copy the prompt N times and send it to vLLM as one batch.
2. Sample each copy until it completes a Python block.
3. Execute every block and append its output, tracebacks included, to that copy's context.
4. Repeat for M rounds.
5. Prune traces whose outputs are broken, then take a majority vote over the post-processed answers.

The final setting was **N=48, M=4**, and going larger did not help.

![SC-TIR illustrated as four frames: 48 copies of a prompt advance through up to four code rounds, finished traces turn green, broken ones are pruned, and the survivors vote](/imgs/blogs/kaggle-aimo-progress-prize-1-solutions-7.webp)

The counts in the frames are illustrative; the point is that the vote is taken only over traces that ran cleanly, and that N and M are **fixed**. Early stopping lets a problem that happens to arrive first consume budget the last one needed. SC-TIR traded peak score for predictable runtime and lower run-to-run variance, a good trade on a 50-problem test.

Here is the control loop as an illustrative toy, not Numina's code: the "model" is a stub that emits canned Python, so the only real part is the loop that samples, executes, prunes and votes.

```python
# Illustrative toy of the SC-TIR control loop. Not Numina's code: the "model"
# is a stub that returns canned Python, so the loop logic is the only real part.
import random, re, subprocess
from collections import Counter

def run_python(code, timeout=5):
    """Execute one code block in a subprocess; return (ok, stdout or error)."""
    try:
        p = subprocess.run(["python3", "-c", code], capture_output=True,
                           text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "TimeoutError"
    return p.returncode == 0, (p.stdout or p.stderr).strip()

def fake_model(trace, rng):
    """Stand-in for one vLLM sampling step: emit a code block or a final answer."""
    if "```output" in trace and rng.random() < 0.7:
        out = trace.rsplit("```output\n", 1)[1].split("\n```")[0]
        return f"So the answer is \\boxed{{{out}}}."
    good = "print(sum(1 for n in range(1, 1001) if (n%3==0 or n%5==0) and n%15))"
    sloppy = "print(sum(1 for n in range(1, 1001) if n%3==0 or n%5==0))"
    broken = "print(undefined_name)"
    return "```python\n" + rng.choices([good, sloppy, broken], [6, 3, 1])[0] + "\n```"

def sc_tir(problem, n=48, m=4, seed=0):
    rng = random.Random(seed)
    traces, answers = [problem] * n, []
    for _ in range(m):                      # at most m code rounds
        alive = []
        for t in traces:                    # in the real thing: one vLLM batch
            step = fake_model(t, rng)
            t += "\n" + step
            boxed = re.search(r"\\boxed\{(\d+)\}", step)
            if boxed:
                answers.append(int(boxed.group(1)) % 1000)
                continue
            code = step.split("```python\n")[1].split("\n```")[0]
            ok, out = run_python(code)
            if not ok:                      # prune traces whose code broke
                continue
            alive.append(t + f"\n```output\n{out}\n```")
        traces = alive
    return Counter(answers)

votes = sc_tir("How many n <= 1000 are divisible by 3 or 5 but not by 15?")
print(votes.most_common(), "->", votes.most_common(1)[0][0])
# [(401, 24), (467, 14)] -> 401
```

With seed 0 the stub's 48 copies end with 38 answers: 24 votes for the correct 401 and 14 for 467, the answer you get by forgetting to exclude multiples of 15. The remaining 10 copies were pruned for broken code or never boxed an answer within 4 rounds. Even a sloppy sampler wins the vote when the correct program is the most likely one, which is the entire bet behind self-consistency.

### Validation was the real product

Numina built several offline sets: 83 integer-answer AMC 2022 and 2023 problems designed to mimic the private set (their models solved about 60 to 65 percent), AIME 2022 to 2024, and MATH level 4 and level 5 integer-answer subsets of about 750 each. Small sets ran with 5 to 10 seeds, bringing measurement variance down to 1 to 3 percent; they are released as `AI-MO/aimo-validation-*`. Every decision above, from data format to fixed N and M, was taken on these sets rather than on the public board, which is why the community explains the 29 versus 22 gap as validation discipline as much as model quality.

## Second and third place

### 2nd: CMU_MATH (22), a policy plus a judge

CMU_MATH (Zhiqing Sun et al.) used both T4s for two different models: a fine-tuned DeepSeekMath-7B-RL **policy** in vLLM on one, and a separately fine-tuned DeepSeekMath-7B-RL **outcome reward model** (ORM) in Hugging Face transformers on the other. They released everything: code, data, the notebook, and both models on Hugging Face.

The policy was fine-tuned on AMC, AIME and Odyssey-Math problems with integer answers, with multiple-choice options stripped, using code solutions sampled few-shot from GPT-4 and kept only when correct. Three epochs, learning rate 2e-5.

The [reward model](/blog/machine-learning/reinforcement-learning/reward-modeling-from-human-preferences)'s data is the clever part. It needs, for each problem, both right and wrong solutions, roughly balanced. To get a spread of skill levels they **interpolated weights**, $M_\alpha = \alpha M_{\text{RL}} + (1 - \alpha) M_{\text{Base}}$, for 8 values of $\alpha$ from 0.3 to 1.0, drawing 2 MATH samples from each. For AMC, AIME and Odyssey they sampled 12 times from each of the policy's checkpoints after 0, 1, 2 and 3 epochs. Wrong non-integer answers were dropped and positives and negatives held at 1:1 per problem.

At inference the policy drew 42 samples per problem at temperature 0.9 over 2 rounds, with feedback ("The code output is not an integer...") appended when a code output was not an integer. The ORM scored the problem plus the code (not the prose), and an answer $a$ with $N_a$ samples got weight

$$
w(a) = N_a \cdot \left(\prod_{j=1}^{N_a} r_j\right)^{1/N_a}
$$

where $r_j$ is the ORM score of the $j$-th sample with that answer. Using the geometric mean rather than the maximum means a single over-confident false positive cannot carry an answer. Answer 0 had its score divided by 10, because the policy produced it far too often. On the 10 training problems the base model solved 1, the fine-tuned policy 2, and the policy plus ORM 4. It is a tiny sample, but the ORM doubled the policy's score.

![Three voting rules on the same 12 illustrative samples: plain majority picks 15, a number from the statement; a penalised vote and a reward-model geometric-mean vote both pick the correct 401, while the single highest-scored sample would have picked 533](/imgs/blogs/kaggle-aimo-progress-prize-1-solutions-8.webp)

The figure runs three voting rules over the same 12 invented samples for the problem from earlier. Plain majority picks 15, a number lifted from the statement, with 4 votes to the correct answer's 3. Down-weighting suspicious answers or weighting by a reward model's geometric mean both recover 401, and the bottom note shows why CMU did not use the maximum score: the single best-scored sample is a wrong 533. The toy below reproduces the numbers; it is illustrative, not either team's code.

```python
# Illustrative toy: three ways to turn 12 sampled answers into one.
# Not any team's code; answers and reward scores are invented.
from collections import Counter, defaultdict
from math import prod

STATEMENT_NUMBERS = {1000, 3, 5, 15}
samples = [  # (final answer, toy reward-model score)
    (401, 0.8), (401, 0.7), (401, 0.9),
    (467, 0.6), (467, 0.5),
    (15, 0.3), (15, 0.2), (15, 0.4), (15, 0.1),
    (0, 0.9), (0, 0.1),
    (533, 0.95),
]

def plain_majority(s):
    return Counter(a for a, _ in s).most_common(1)[0][0]

def penalised_majority(s, w=0.25):
    votes = defaultdict(float)
    for a, _ in s:
        suspicious = a < 10 or a in STATEMENT_NUMBERS
        votes[a] += w if suspicious else 1.0
    return max(votes, key=votes.get), dict(votes)

def orm_weighted(s):
    scores = defaultdict(list)
    for a, r in s:
        scores[a].append(r)
    weight = {}
    for a, rs in scores.items():
        w = len(rs) * prod(rs) ** (1 / len(rs))   # N x geometric mean
        weight[a] = w / 10 if a == 0 else w      # answer 0 divided by 10
    return max(weight, key=weight.get), {a: round(w, 2) for a, w in weight.items()}

def best_single_score(s):
    return max(s, key=lambda x: x[1])[0]

print("plain majority  :", plain_majority(samples))
print("penalised vote  :", penalised_majority(samples))
print("ORM N x geomean :", orm_weighted(samples))
print("max single score:", best_single_score(samples))
# plain majority  : 15
# penalised vote  : (401, {401: 3.0, 467: 2.0, 15: 1.0, 0: 0.5, 533: 1.0})
# ORM N x geomean : (401, {401: 2.39, 467: 1.1, 15: 0.89, 0: 0.06, 533: 0.95})
# max single score: 533
```

The 0.25 weight in `penalised_majority` is borrowed from codeinter's rule for small integers; after exams did not publish its exact penalty, only that it was a percentage of the total generations.

### 3rd: after exams (21), no fine-tuning at all

after exams (David Dinucu-Jianu, Cătălin Grîu, Catalin Brita and mirceatlx) is the strongest argument that inference engineering alone was worth something. They took Abdur Rafae's public notebook with the un-finetuned DeepSeekMath-7B-RL, moved it to vLLM, and generated **120 to 160 candidates per problem in one batch**, with code executed in parallel across the batch. More than 6 code-execution iterations helped. An FP16 KV cache improved the score.

Two small rules did the rest. First, a **forced answer**: if a generation ended without a `\boxed{}`, they appended "The final answer is \boxed{" and generated a few more tokens, so an unfinished trace still voted. Second, a **penalty**, applied as a percentage of the total generations, on answers that were small (below 10, often the residue of a code error) and on numbers that appear in the problem statement. The writeup calls it "a significant boost", and the comments quantify it: a submission that penalised only 0 plus in-statement numbers scored 20 private against 21 for the full rule, so the small-number penalty was worth one problem. Geremie Yeo, outside the top 20, found that 25 of 61 AIME 2022 to 2024 problems had a single-digit modal answer under the baseline, the failure this rule targets.

What they left out is as telling. They patched vLLM to run bf16 weights on a T4 by computing in fp32 and casting back; it gave their best validation score but was unstable and never beat 25 on public. A logits-processor trick that injected code output tokens straight into generation was not selected either. They chose an older version for the final because of tie risk, an honest note about how much luck sits in a 50-problem score.

## Ranks 4 to 20, by approach family

### Inference-only, buy more votes: codeinter (4th, 21)

codeinter (Zonglin Han et al.) also built on Abdur Rafae's notebook, via another user's reorganisation with one chain-of-thought prompt and one write-code prompt. Their distinguishing idea was to treat the two T4s as **two independent samplers**: one copy of DeepSeekMath-7B-RL per GPU, run in parallel with a `ThreadPoolExecutor`, which raised repetitions per problem from 21 to 30 to 40 at temperature 0.9 and 2,048 max new tokens. If the first GPU already had enough candidates, the slower one was cut off and the two pools merged.

They also built the adaptive `TimeManager` that Numina avoided: when the running average time per problem exceeded the remaining budget per problem, the next problem's attempts shrank to 0.8 times the default times budget over average. Problems containing `[asy]` (geometry with a drawn figure) got only 3 attempts, since any local success looked like memorisation. Code revision was capped at 3 rounds, small integers 0 to 5 got weight 0.25 in the vote, and when the code output was not a valid answer the text answer was ignored as "almost always incorrect". Validation was the latest 50 AMC 12A problems from a 71k Art of Problem Solving dump, which they report correlated well with the public board.

### The control experiment: Conor #2 (5th, 20)

When another competitor asked for the 5th-place solution, conor replied that his selected submissions were **unmodified forks of the Early Sharing Prize winner's notebook**, picked automatically by Kaggle, and that he had **declined the prize money** because the work was not his. He made the fork public as "fork of 23 public score". Another competitor's reply sums up its value: it shows "where the high end of the variance was for that notebook". A notebook that scored 23 on public scored 20 on private and finished 5th of 1,161. That is the best single measurement of what the shared baseline was worth, and the reason to read ranks 3 to 5 as "baseline plus small, real, noisy improvements" rather than as distinct methods.

### No writeup: ranks 6 to 20

Fifteen teams, no published method. Here is everything that is known:

- **Rank 7** is a deleted account. Mid-competition, a 27-of-50 account in 3rd place on public vanished and commenters suspected a multi-account; whether rank 7 is that account is unverified.
- **Rank 10, curiousgeorge**, posted only about a dataset upload failure.
- **Rank 17** is Abdur Rafae himself, author of the baseline notebook. His final configuration is not documented.
- **Ranks 6, 8, 9, 11 to 16, and 18 to 20**: nothing posted.

## What the forum taught everyone

- **Abdur Rafae, "Improved Code Interpretation".** DeepSeekMath-7B-RL plus code execution plus majority vote, public early. Ranks 1, 3 and 4 credit it, rank 5 forked it unchanged, and its author finished 17th.
- **Chan Kha Vu, "Solutions to 10 samples in train set".** Solved every training problem by hand and found a wrong label, which the organisers fixed.
- **Tong Hui Kang** predicted code-tool plus voting designs well before the end, and was right.
- **Khoi Nguyen** asked whether a team at 47 of 50 would really hand over its solution for about USD 800,000. Moot: the best score was 29.
- **ISAKA Tsuyoshi, "Public LB20 / Private FAIL".** A strict answer parser alone moved his public score from 14 to 19; then the private rerun failed. Robustness on unseen problems is part of the score.

On the **shake-up**: it cannot be measured, because public scores are gone from the API. The indirect evidence (3 problems of seed variance, a 23-public fork landing at 20 private, eight teams tied at 20, a final-scores comment reading "Yikes talk about shakeup! Random seed not in my favour") points one way, but any public-versus-private chart would be fabricated, so this post has none.

## What did not work

Aggregated across the four writeups and their comment threads:

| Tried | Result | Reported by |
|---|---|---|
| Larger bases (InternLM-20B, CodeLlama-33B, Mixtral-8x7B) | Slow, timeouts, no better than DeepSeekMath | Numina |
| Integer-only filtering of Stage-2 data | Slightly worse than keeping all solution types | Numina |
| RLOO with a code-execution reward | No significant gain, slow | Numina |
| Model merging (DARE, TIES, WARP) | Regressed or unfinished | Numina |
| Static KV cache plus `torch.compile` | 2 to 3x faster on H100, broke on T4 | Numina |
| Early stopping and time-budget tricks | Higher variance | Numina |
| Deduplicating policy data; adding GPT-4's wrong solutions; ratios other than 1:1 | No help | CMU_MATH |
| bf16 on T4 via patched vLLM | Unstable, never above 25 public | after exams |
| Penalising only 0 instead of all small numbers | One problem fewer (20 vs 21) | after exams |
| More than 3 code-revision rounds; parsing text when code failed | Worse output; almost always wrong | codeinter |
| Full attempts on `[asy]` geometry | Rarely solved | codeinter |

Most failed ideas squeezed inference; the one large idea that worked late (Numina's KTO) was about training.

## Lessons that transfer

1. **Put the tool in the loop.** Prose-only reasoning took Numina's model to 8 of 50, single-program tool use to 16, multi-step tool-integrated reasoning to 27, on the same base. (Numina writeup and comments.)
2. **Change the data before you change the decoder.** The two teams that fine-tuned on GPT-4 generated solutions finished first and second; the three inference-only entries behind them scored 20 to 21. (Top-20 matrix.)
3. **Build validation you can trust and run it with several seeds.** With about 3 problems of seed noise, the public board cannot tell you whether a change worked; Numina's sets with 5 to 10 seeds got variance to 1 to 3 percent. (Numina, codeinter.)
4. **A learned verifier beats a head count, and the aggregation matters.** CMU's ORM doubled the policy's training-problem score from 2 to 4 of 10, and the geometric mean resists a single confident false positive. (CMU_MATH.)
5. **Cheap priors on wrong answers are free points.** Zero, tiny numbers and numbers copied from the statement are disproportionately wrong; down-weighting them was worth a problem to after exams. (after exams, CMU_MATH.)
6. **Decide about variance explicitly.** Numina fixed N and M; codeinter adapted to the clock. Both worked; leaving the budget to chance does not. (Numina, codeinter.)

### If you enter a competition like this

- Submit the strongest public notebook unchanged first: that is your noise floor, and rank 5 shows it can be 20 of 50.
- Build offline sets that match the hidden test's difficulty and answer format; run every change with at least 5 seeds.
- Generate tool-integrated training traces with the strongest model allowed and filter them by answer correctness.
- Quantise if it buys more samples.
- Prune traces whose code failed, force an answer on unfinished ones, down-weight 0, small and in-statement numbers.
- Leave margin under the time limit; a private rerun that times out scores nothing.

The sequel raised every number: [AIMO Progress Prize 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions) moved to harder problems and bigger hardware, and the [AIMO Progress Prize 3 post](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions) follows the same stack onto H100s. The math thread continues with [Eedi's misconception retrieval](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions) and the [MAP student-misunderstanding classifiers](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions).

## Sources

- Competition: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize ; final scores thread https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/516823
- 1st, Numina: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/numina-numina-1st-place-solution ; Hugging Face blog https://huggingface.co/blog/winning-aimo-progress-prize ; notebook https://www.kaggle.com/code/lewtun/numina-1st-place-solution ; model https://huggingface.co/AI-MO/NuminaMath-7B-TIR ; code and report https://github.com/project-numina/aimo-progress-prize ; datasets https://huggingface.co/collections/AI-MO/numinamath-6697df380293bcfdbc1d978c
- 2nd, CMU_MATH: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/cmu-math-2nd-place-solution-all-code-and-datasets- ; code and data https://github.com/AIMO-CMU-MATH/CMU_MATH-AIMO ; notebook https://www.kaggle.com/code/edwardsun0909/aimo-2nd-place-solution
- 3rd, after exams: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/after-exams-3rd-place-solution ; notebook https://www.kaggle.com/code/davidd2/3rd-place-solution-afterexams ; bf16 vLLM patch https://github.com/RD211/vllm-bf16
- 4th, codeinter: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/codeinter-4th-place-solution ; notebook https://www.kaggle.com/code/winter2003/4th-dual-gpu-reasoning-and-comparing-answers
- 5th, Conor #2: comment https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/discussion/523125 ; notebook https://www.kaggle.com/code/conormacamhlaoibh/aimo-fork-of-23-public-score/notebook
- Baseline: Abdur Rafae, https://www.kaggle.com/code/abdurrafae/improved-code-interpretation ; forum posts discussion/495362 and discussion/499464
- Forum: discussion/490640 (Chan Kha Vu), discussion/490691 (Tong Hui Kang), discussion/488441 (Khoi Nguyen), discussion/511414 (Geremie Yeo), discussion/515519, discussion/507796 (Abdullah Meda), discussion/492578 and discussion/493345 (rank 21); ISAKA Tsuyoshi https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/isaka-tsuyoshi-public-lb20-private-fail-write-up
- Papers: MuMath-Code, arXiv 2405.07551; ToRA, arXiv 2309.17452
