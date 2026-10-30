---
title: "Konwinski Prize: how the top 20 teams handled a metric that punishes wrong patches"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A synthesis of the top 20 finishers in Kaggle's Konwinski Prize, a contamination-free SWE-bench: why the winner resolved only 9 of 120 new GitHub issues, how to decode a leaderboard score into right, wrong and skipped, and why executable reproduction tests separated the top from the rest."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "swe-bench", "code-agents", "agentless", "selective-prediction", "benchmark-contamination", "reproduction-tests", "qwen2-5-coder"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 23
---

> **TL;DR.** The Konwinski Prize (December 2024 to July 2025) was SWE-bench with the answers not yet written: the private test was 120 GitHub issues opened *after* the submission deadline, so no model could have memorised them. A wrong patch cost exactly as much as a right one earned. The winner, Eduardo Rocha de Andrade (private 0.058242), resolved 9 issues, broke 2 and skipped 109: a 7.5% resolve rate at 82% precision. The decisive lever was **test-time compute spent on verification**: the model wrote its own reproduction test before touching the code, and a patch shipped only if that test flipped and the repo's own tests did not regress. The one lesson: when wrong answers are expensive, the system you are really building is a judge of its own work, and the best judge is something you can execute.

![Lever strip with Test-time compute lit for verification and Metric insight lit secondarily](/imgs/blogs/kaggle-konwinski-prize-solutions-1.webp)

The strip above is the frame for every post in this series, with the decisive lever lit. Here it is test-time compute, but not the kind the AIMO winners spent on sampling more answers. The Konwinski winners spent their compute on *rejecting* answers. Every top-20 team skipped at least 107 of 120 issues, and what separated the podium from the rest was how they chose the 11 or 12 to keep.

This post synthesises the writeups of the top 20 private finishers. Only seven published one (ranks 1, 2, 3, 4, 5, 8 and 15); the other thirteen are named as "no writeup", with their record decoded from the score and nothing invented about their method. If you are new to the series, start with [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work).

## 1. The task in 60 seconds

Andy Konwinski put up the money himself and wrote the overview in the first person, offering a million dollars "for the AI that can close 90% of new GitHub issues." The competition opened on 2024-12-11 and closed submissions on 2025-03-12. Then came a **forecasting phase**: the hosts collected fresh GitHub issues from real Python repositories after the deadline and ran every frozen submission against them. Kaggle's recap counts 471 teams from 61 countries.

![Budget card next to an illustrative issue: issue and repo snapshot in, a git diff or SKIP out, scored by F2P and P2P tests](/imgs/blogs/kaggle-konwinski-prize-solutions-2.webp)

The input is SWE-bench shaped: a repository snapshot plus one issue's problem statement. The output is a git diff, or nothing. The scorer applies the diff and runs the repo's own tests: **fail-to-pass (F2P)** tests, which failed before the real fix and must pass after it, and **pass-to-pass (P2P)** tests, which must still pass. Both must hold for the issue to count as resolved.

Internet is off. Each issue must finish within 30 minutes of being served. The notebook gets 9 hours in the training phase and 24 hours in the forecasting phase, on a 4x NVIDIA L4 machine (96 GB of VRAM in total). Only open-source code and open-weight models were eligible for prizes.

The **public** leaderboard ran on 71 historical issues, a count players worked out from the score decimals. Their real fixes could be scraped from GitHub, so everyone knew the public leaderboard leaked. The **private** set, the one that paid, was 120 new issues; the hosts had announced "approximately 150 to 200", and 120 is what the scores imply, consistent with the winner's own 9 + 2 + 109.

The prize pool was USD 1,225,000: USD 50k, 20k, 10k, 10k and 10k for the top five, a further USD 50k at each of 30%, 40% and so on up to 90% for top-five teams reaching it, and USD 775,000 more for a first place above 90%. None of the thresholds was reached.

## 2. The metric, worked on a toy

Let $a$ be the number of issues resolved, $b$ the number of wrong patches, and $c$ the number skipped. The score is

$$
\text{score} = \frac{a - b - c / 10{,}000}{a + b + c}, \qquad \text{score} = -1 \text{ if } a = 0 .
$$

Read it as a bet on every issue. A right patch earns one point, a wrong patch loses one, and a skip costs a ten-thousandth. The skip cost came in a metric revision on 2025-01-17, so that skipping everything no longer scores zero and so that, between two teams with the same right-minus-wrong, the one that attempted more ranks higher. The $a = 0$ rule means you cannot sit the competition out.

![Five strategies on 120 issues: skip all scores -1, attempt all at 25% scores -0.5, 15 right and 15 wrong about zero, one sure fix +0.008234, the winner +0.058242](/imgs/blogs/kaggle-konwinski-prize-solutions-3.webp)

The winning score of the whole competition is a sliver next to what careless attempting costs. Attempting all 120 and getting a quarter right scores $(30 - 90)/120 = -0.5$. Getting 15 right but also submitting 15 wrong scores slightly *below* zero, because the 90 skips each cost a ten-thousandth. One sure fix and 119 skips beats both.

The decision rule falls out in one line. If you would get an issue right with probability $p$, attempting it changes the numerator by $p \cdot 1 + (1 - p) \cdot (-1) = 2p - 1$, while skipping costs ${-0.0001}$. Attempt only when $p$ is above one half, and in practice well above it, because your estimate of $p$ is itself noisy. This is a **selective classification** problem wearing a code-generation costume: the patcher matters, but the abstention policy decides the score.

### Decoding a leaderboard score

Because the denominator is fixed at 120 for every private entry and the skip term gives each record a tiny unique offset, a displayed score can be inverted into its (right, wrong, skipped) triple. Tong Hui Kang published the decoder idea during the competition, and the top-20 pass for this post applied it to every finisher. Two assumptions are needed: **the private set has 120 issues, and the displayed score is truncated (not rounded) to six decimals.** Under them every top-20 score decodes to exactly one triple, and the decoding reproduces both records the teams stated themselves: 1st place's 9 / 2 / 109 and 4th place's 4 / 2 / 114.

![Seven records with right minus wrong equal to 2 on a number line; only 7 right, 5 wrong, 108 skipped displays as 0.016576](/imgs/blogs/kaggle-konwinski-prize-solutions-4.webp)

One decode worked through, for 2nd place, Camaro, displayed as 0.016576. Truncation means the true score lies in $[0.016576, 0.016577)$; multiply by 120 and the numerator $a - b - c/10{,}000$ lies in $[1.98912, 1.98924)$. The skip term is at most 0.012, so $a - b$ must be 2, and the deficit below 2 gives $c/10{,}000 \in (0.01076, 0.01088]$, so $c = 108$. Then $a + b = 12$ and $a - b = 2$ give $a = 7$, $b = 5$. Forward check: $(7 - 5 - 0.0108)/120 = 0.0165766...$, which truncates to 0.016576. Rounding would display 0.016577, and no integer triple rounds to 0.016576, which is why truncation is the rule that fits.

This corrects the research bank behind this series, which first read the 0.0165x scores as "about 2 right and 0 wrong"; that record would display 0.016568. Ranks 2 and 3 were in fact **7 right, 5 wrong** and **6 right, 4 wrong**, aggressive records rather than cautious ones. The triples for ranks 2, 3 and 5 to 20 are derived, not confirmed by those teams.

```python
# Illustrative toy, not any team's code: the K Prize metric, the leaderboard
# decoder, and the break-even precision that makes skipping the default move.
from decimal import Decimal, ROUND_DOWN

def kprize_score(right: int, wrong: int, skipped: int) -> Decimal:
    if right == 0:
        return Decimal(-1)
    n = right + wrong + skipped
    raw = (Decimal(right - wrong) - Decimal(skipped) / 10_000) / n
    return raw.quantize(Decimal("0.000001"), rounding=ROUND_DOWN)  # truncate

def decode(displayed: str, n: int = 120):
    target = Decimal(displayed)
    return [(a, b, n - a - b)
            for a in range(1, n + 1) for b in range(0, n - a + 1)
            if kprize_score(a, b, n - a - b) == target]

for shown in ["0.058242", "0.016576", "0.016575", "0.016571", "-0.008422"]:
    print(f"{shown:>10} -> {decode(shown)}")

# Expected change in (right - wrong) from attempting one issue you would
# get right with probability p, versus the 1/10,000 a skip costs.
for p in [0.3, 0.5, 0.6, 0.82]:
    attempt = p * 1 + (1 - p) * -1
    print(f"p={p:.2f}  attempt={attempt:+.2f}  skip={-1/10_000:+.4f}")
```

```console
  0.058242 -> [(9, 2, 109)]
  0.016576 -> [(7, 5, 108)]
  0.016575 -> [(6, 4, 110)]
  0.016571 -> [(4, 2, 114)]
 -0.008422 -> [(6, 7, 107)]
p=0.30  attempt=-0.40  skip=-0.0001
p=0.50  attempt=+0.00  skip=-0.0001
p=0.60  attempt=+0.20  skip=-0.0001
p=0.82  attempt=+0.64  skip=-0.0001
```

The last decode is rank 20, Chew Kok Wah: 6 right, as many as 3rd place, but 7 wrong.

## 3. What made it hard: why the scores were so low

The host's summary was blunt: the winner resolved about 8% of the new issues (9 of 120 is 7.5%), a number the host recap set against SWE-bench headlines that had trained everyone to expect far more. Five forces, each documented in the writeups, explain the gap.

![Four 120-cell grids for the top four: 9 and 2, 7 and 5, 6 and 4, 4 and 2 right and wrong, everything else skipped](/imgs/blogs/kaggle-konwinski-prize-solutions-6.webp)

**First, nothing to remember.** A benchmark built from old GitHub issues is built from text that may sit in pretraining data: the issue, its discussion, sometimes the merged fix, and certainly the surrounding code. The forecasting phase removed that by construction; the issues did not exist when the models were frozen. (See [decontamination and benchmark leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage).) Eduardo's writeup gives an indirect symptom: his small model understood the fix but kept failing on imports and on how to use the repo's classes, until he fed it import lines and class skeletons. His lesson is to give small models the repository context that frontier models seem to have memorised.

**Second, a small model on small hardware.** The L4 box holds a 32B model comfortably and a 70B model only quantised. Eduardo developed on a single RTX 4090 with about 8k tokens of context, Camaro on two rented RTX 3090s. Thirty minutes per issue leaves little room for long agent loops.

**Third, the metric caps the resolve rate on purpose.** The grids above are the key picture. The winner's precision on attempted issues was 9 of 11, about 82%; his resolve rate was 7.5% because he attempted only 11. SWE-bench's "% resolved" has no penalty for a wrong patch, so a system there should attempt everything. The K Prize number answers the question a maintainer cares about: of the PRs you open, how many should I merge?

**Fourth, real issues are messier than curated ones.** Camaro said SWE-bench, especially the Verified subset, is easier and cleaner than the K Prize public set, which had higher skip and failure rates and "more noisy instances". Tong Hui Kang, author of the most-used starter notebook, could not pass some example tests himself because the required fix included changes the issue never mentioned.

**Fifth, almost no reliable validation.** The 71 public issues were leaky and tiny, and six training instances came with the competition. The largest external set, about 6,400 GitHub issues from Nebius, had a different distribution. Camaro validated on SWE-bench Verified and warned that about half of it is Django, so local scores are easy to over-fit. His same code scored 0.0703 (7 right, 2 wrong) and ${-0.0142}$ (4 right, 5 wrong) on the public set in two runs.

## 4. Top-20 scorecard

The API's public column reads 0.000000 for every team after the forecasting rescore, so public ranks are unavailable ("n/a") except where a team stated its own. Records are decoded as above; "confirmed" means the team stated it.

| Rank | Team | Private | Record (right, wrong, skipped) | Public rank | Key idea | Source |
|---|---|---|---|---|---|---|
| 1 | Eduardo Rocha de Andrade | 0.058242 | 9, 2, 109 (confirmed) | 1st (forum comment) | Agentless fork, reproduction tests, F2P/P2P gate | [writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/eduardo-rocha-de-andrade-1st-place-solution-write-) |
| 2 | Camaro | 0.016576 | 7, 5, 108 | 2nd (self-reported) | Agentless-style, JSON edits, test generation | [writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/camaro-public-2nd-place-solution) |
| 3 | akami | 0.016575 | 6, 4, 110 | n/a | LoRA difficulty classifier as skip gate | [writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/3rd-place-solution) |
| 4 | Genxxsky | 0.016571 | 4, 2, 114 (confirmed) | n/a | Generated before/after pytest | [writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/4th-place-solution) |
| 5 | Kenj | 0.008239 | 4, 3, 113 | n/a | Regex traceback locator; no traceback, skip | [writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/5th-place-solution) |
| 6 | 1M | 0.008237 | 3, 2, 115 | n/a | unknown | no writeup |
| 7 | snoopyyyyy | 0.008237 | 3, 2, 115 | n/a | unknown | no writeup |
| 8 | first lucky gold! | 0.008235 | 2, 1, 117 | n/a | Model chosen by right-to-wrong ratio | [writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/first-lucky-gold-8th-place-solution) |
| 9 | bianfu | 0.008234 | 1, 0, 119 | n/a | unknown | no writeup |
| 10 | Đàm Văn Tài | 0.008234 | 1, 0, 119 | n/a | unknown | no writeup |
| 11 | Myles and Trevor | -0.000093 | 4, 4, 112 | n/a | unknown | no writeup |
| 12 | nico | -0.000096 | 2, 2, 116 | n/a | unknown | no writeup |
| 13 | Kenleung Leung | -0.000096 | 2, 2, 116 | n/a | unknown | no writeup |
| 14 | Marcelo Piovan | -0.000096 | 2, 2, 116 | n/a | unknown | no writeup |
| 15 | GraphNINJAs | -0.000096 | 2, 2, 116 | 4th (self-reported) | Skip gates tuned on public, edit-distance selection | [writeup](https://www.kaggle.com/competitions/konwinski-prize/writeups/graphninjas-public-4th-place-private-15th-place-so) |
| 16 | PRASHANT SHUKLA91 | -0.000098 | 1, 1, 118 | n/a | unknown | no writeup |
| 17 | BIT Research Alliance | -0.000098 | 1, 1, 118 | n/a | unknown | no writeup |
| 18 | Mohammad Albaddawi | -0.000098 | 1, 1, 118 | n/a | unknown | no writeup |
| 19 | yellowleaf | -0.000098 | 1, 1, 118 | n/a | unknown | no writeup |
| 20 | Chew Kok Wah | -0.008422 | 6, 7, 107 | n/a | unknown | no writeup |

Ranks 6 and 7, 9 and 10, 12 to 15 and 16 to 19 share identical scores and records; the order inside each tie is the leaderboard's. Only ranks 1 to 10 have more right than wrong. Rank 11 resolved four issues, twice as many as rank 8, and finished below it. A team that resolved one issue and attempted nothing else (ranks 9 and 10) beat every team from 11 to 20.

## 5. The top-20 landscape

![Technique matrix: seven writeups as columns, thirteen no-writeup teams as question marks, technique rows with N of 7 counts](/imgs/blogs/kaggle-konwinski-prize-solutions-5.webp)

With 7 writeups of 20, every count below is out of 7, recounted from the matrix cells.

**Table stakes.** All 7 had an explicit skip rule, and every top-20 team skipped at least 107 issues. Six of 7 generated several candidate patches per issue (the 8th-place writeup does not say). All ran open models between 14B and 70B parameters. And 0 of 7 used an agent loop in the final: the field converged on fixed Agentless-style pipelines (localise, repair, validate), not on agents that act and observe.

**Two lineages.** Five of 7 built on Tong Hui Kang's public Select-Patch-Verify notebook: an LLM picks files and lines, writes several diffs, and an LLM verifier answers yes or no. The top two did not: Eduardo forked Agentless 1.5 and Camaro wrote his own Agentless-style pipeline, both with Qwen2.5-Coder-32B. Three of 7 used a DeepSeek-R1-Distill reasoning model (ranks 3, 5 and 15).

**What separated the top: executable verification.** Three of 7 had the model write a reproduction test and then *ran* it: ranks 1, 2 and 4. Two of 7, ranks 1 and 2, also ran the repository's pass-to-pass tests. Everyone else accepted a patch on an LLM's yes, on the first patch that applied, or on a heuristic. An LLM judging its own patch is a second sample from the distribution that wrote it; a test that fails before the patch and passes after it is evidence from outside the model. The winner had 9 right; nobody else had more than 7.

**Contradictory findings.**

| Question | One side | Other side |
|---|---|---|
| Reasoning or coder model? | Eduardo, Camaro: R1-style models slower and no better than Qwen2.5-Coder-32B | akami, Kenj, GraphNINJAs all used R1-Distill models |
| Does a bigger model help? | Camaro: 14B to 32B fixed most context and syntax problems | Kenj: R1-Distill 32B no better than 14B |
| Trust the public leaderboard? | GraphNINJAs tuned skip gates on 71 public issues and fell from public 4th to private 15th | Eduardo tuned for consistency across seeds and held 1st |
| How much context? | Camaro: even ground-truth files helped little | Eduardo: imports and class skeletons were essential |

The model rows are less contradictory than they look: the coder-model teams also ran tests, the reasoning-model teams mostly did not. The top two attribute their result to the verifier.

## 6. The winning solution, explained

Eduardo Rocha de Andrade joined about a month before the deadline; his code is a public Kaggle notebook.

### The intuition: test first, then you know when to stop

A maintainer triaging an issue does two things. First, reproduce it: if you cannot make the bug happen, you cannot know whether your change fixed it, so you should not open a PR. Second, after the fix, check that the reproduction passes and nothing else broke. Eduardo's pipeline turns both into hard gates, and the skip decision comes for free: an issue the model cannot reproduce is one it cannot verify, and on this metric an unverifiable patch is a coin flip with a negative edge.

![First-place funnel: reproduction tests, localisation, eight SEARCH/REPLACE candidates, three acceptance gates, one patch or SKIP](/imgs/blogs/kaggle-konwinski-prize-solutions-7.webp)

### The mechanism, stage by stage

**Reproduction tests first.** The model writes 5 candidate tests per issue with different context: one with example imports plus the relevant classes and functions, one with imports only, and three with none, as in the original Agentless. One is greedy, the rest sampled. Context is found in two stages: the model picks relevant unit-test files, then sees their skeletons and picks what to inspect. A test counts only if it prints exactly "Issue reproduced"; one that also prints "Other issues" is discarded. If none of the 5 reproduces, 5 more are generated at higher temperature; if none of 10 does, the issue is skipped.

**Localisation.** Files, then classes and functions, then line-level edit locations. The repository structure is parsed once and reused, saving 5 to 20 seconds per step.

**Repair.** Four samples per edit location (1 greedy, 3 sampled), 8 candidates in all, written in **SEARCH/REPLACE** format: the model quotes old and new lines, and code converts them into a git diff. Asking a small model for a unified diff directly gave far fewer valid patches.

**Selection.** A `git apply` dry run; then at least one reproduction (F2P) test must pass; then the model-selected P2P tests must do as well as or better than the unpatched code. If nothing survives, skip. His best runs took 5 to 7 hours, about 3 to 10 minutes per issue, with guards that skip any issue after 12 minutes and everything after 20 hours.

```python
# Illustrative toy, not Eduardo's code: the shape of the 1st-place acceptance
# gate. A patch ships only if a reproduction test flips and nothing regresses.
from dataclasses import dataclass

@dataclass
class Candidate:
    name: str
    applies: bool            # `git apply --check` succeeded
    f2p_passing: int         # reproduction tests that now pass
    p2p_failures: int        # selected repo tests failing after the patch

def choose(cands, repro_tests: int, baseline_p2p_failures: int):
    if repro_tests == 0:
        return "SKIP: no test reproduces the issue, nothing can verify a patch"
    for c in cands:          # candidates arrive greedy-first
        if not c.applies:
            continue
        if c.f2p_passing == 0:
            continue
        if c.p2p_failures > baseline_p2p_failures:
            continue
        return f"SUBMIT {c.name}"
    return "SKIP: no candidate passed every check"

cands = [Candidate("greedy", True, 0, 0), Candidate("s1", False, 0, 0),
         Candidate("s2", True, 2, 3), Candidate("s3", True, 1, 1)]
print(choose(cands, repro_tests=0, baseline_p2p_failures=1))
print(choose(cands, repro_tests=3, baseline_p2p_failures=1))
print(choose(cands[:3], repro_tests=3, baseline_p2p_failures=1))
```

```console
SKIP: no test reproduces the issue, nothing can verify a patch
SUBMIT s3
SKIP: no candidate passed every check
```

In the toy, the greedy patch applies but flips no reproduction test, `s1` does not apply, and `s2` fixes the bug while breaking two more repo tests. Only `s3` clears all three gates. How Eduardo ranked several passing patches is not stated; the toy takes the first.

### Why it held up on new data

Eduardo's public resubmissions scored between 0.056 and 0.098, and a forum comment from Giba places him first on both leaderboards. He said he spent his last two weeks not on raising his best public score but on making the result consistent across seeds. What he thinks mattered: requiring **both** F2P and P2P tests; **greedy or very low temperature at almost every step**, with a few higher-temperature samples for diversity; **varying the context per sample**, because the best amount varied a lot by issue; and a faster pipeline that bought more samples. He tried two LLMs, one per GPU, ensembled, but could not make it work: every reasoning model he tried was slow and did not beat Qwen2.5-Coder-32B.

## 7. Second and third in depth

### 2nd: Camaro, an Agentless pipeline built by hand

Camaro worked for about three months. An LLM first judges whether an issue is solvable, to skip early; Qwen was cautious and Gemini overconfident, and he chose a lenient setting. Then file localisation (pick 5 Python files; more candidates, a second pass and embeddings did not help), then extraction of the relevant parts. The step he calls most important is **reproduction-test generation**, with few-shot examples and syntax errors fed back. Repair is done as function-calling edits (path, old string, new string) with up to 5 retries. A final verifier checks test logic, resolution and regressions; he warns that a verifier set too strict rejects everything.

The model is Qwen2.5-Coder-32B served by [vLLM](/blog/machine-learning/inference-frameworks/vllm-inference) with JSON-schema [structured outputs](/blog/machine-learning/inference-engineering/structured-output-in-production-streaming-json-and-tool-calls), chosen over Qwen2.5-72B, QwQ-32B and DeepSeek-R1 for stability and speed. He tuned 44 prompt variants across the 6 steps by comparing outputs from several chat assistants, an "LLM consortium". He tracked a "good_test_rate", the share of generated tests that reproduce the issue and are fixed by the gold patch, and found that improving a stage metric did not always improve the final score. His decoded 7 right and 5 wrong is 58% precision, not far above the 50% break-even: his gates let through more wrong patches than Eduardo's did.

### 3rd: akami, a classifier that says "too hard"

akami (akiyama, sakami and kami) kept the Select-Patch-Verify pipeline and loaded one model for everything: DeepSeek-R1-Distill-Llama-70B, quantised with AWQ to fit (see [weight-only quantisation with GPTQ and AWQ](/blog/machine-learning/edge-ai/llm-quantization-weight-only-gptq-awq)). Their addition is a **difficulty gate**: a [LoRA](/blog/machine-learning/debugging-training/debugging-lora-and-peft) of rank 32 on the same base, trained as a three-class classifier on SWE-bench Verified's human difficulty labels (easy under 15 minutes, medium 15 to 60, hard over an hour), with 400 training and 100 validation examples. Any issue with P(easy) under 0.5 is skipped.

The rest is lean: six parallel search queries at temperature 0.6, each matching line taken with 12 lines either side, 6 patches per search result, and a patch must pass an LLM yes/no check, parse as a unified diff and survive a `patch` dry run. The only top-20 fine-tuning was of the abstention policy, not the patcher. Their verdict: "it was just luck that we ended up in 3rd place."

## 8. Ranks 4 to 20, by approach family

### Family A: the starter notebook plus executable tests (4)

**Genxxsky (4th, 4 right, 2 wrong, confirmed).** Built on the Select-Patch-Verify starter and on the decoder thread, and decoded both of his own submissions. For each candidate patch, the model writes a pytest file with `test_before_patch` (passes before the fix, fails after) and `test_after_patch` (the reverse). Code search uses tree-sitter to return the whole enclosing function or class around each match. His other final strategy, more patches picked by a multi-factor evaluation, scored 2 right and 3 wrong on private. The lean, tested one was slightly negative on public (1 right, 1 wrong) and 4th on private.

### Family B: deterministic localisation and a hard skip (5)

**Kenj (5th, 4 right, 3 wrong).** Joined late. He skipped every issue without an error traceback, because only then could he find the right edit location; from public probing he estimated more than 6 and fewer than 11 public issues had one. A **regex traceback parser** replaced the starter's LLM keyword search, which invented phrases. Patches come from DeepSeek-R1-Distill-Qwen-14B, 8 per issue, in SEARCH/REPLACE format ("git conflict markers"), applied to a scratch git repo and turned into a diff with `git diff`. Selection is the first valid patch; no tests run.

### Family C: choose the model by its error ratio (8)

**yuanzhe zhou, "first lucky gold!" (8th, 2 right, 1 wrong).** He calls the approach "statistical arbitrage". Keyword filters on patches helped only marginally. The breakthrough was noticing that one set of model weights failed on many issues but had a notably high right-to-wrong ratio on those it answered, and building the submission around it. Model names are not given. Pick the system by precision, not recall.

### Family D: gates tuned on the public set (15)

**GraphNINJAs (15th, 2 right, 2 wrong; public 4th with 4 right, 0 wrong).** The most engineered writeup: code review, 12 modules, 98 tests, 80% coverage. Select-Patch-Verify version 94 with DeepSeek-R1-Distill-Qwen-32B-AWQ, and selection by **edit distance**: drop patches over 1,000 characters, then pick the one with the smallest summed edit distance to all others. A hypergeometric simulation told them an odd number of attempts beats the adjacent even number, because even counts allow 2 to 2 ties. Their gates, tuned on the 71 public issues, kept problem statements between 1,802 and 3,400 characters and repos with at most 388 files. They scored 4 and 0 on public, then 2 and 2 on 120 new issues: exactly the tie their simulation warned against.

### No writeup (6, 7, 9 to 14, 16 to 20)

Thirteen teams published nothing, and a sweep of all 182 forum topics found no posts by any member. Two of their records are instructive. Myles and Trevor (11th) resolved 4 issues, as many as 4th and 5th place, and broke 4. Chew Kok Wah (20th) resolved 6 and broke 7.

### The best contrasting entries

**GraphNINJAs** is the cleanest case of tuning abstention rules on a tiny, leaky public set. **Genxxsky's** two strategies are the cleanest ablation: same author, same base, and the leaner, test-gated variant won on the only set that mattered.

## 9. What the forum taught everyone

![Diverging bars of right and wrong for ranks 1 to 20; only ranks 1 to 10 have more right than wrong](/imgs/blogs/kaggle-konwinski-prize-solutions-8.webp)

With 1 to 9 right answers out of 120 and a symmetric penalty, one issue moves a team many places; 3rd place called their result luck, and Eduardo wrote that luck would play a significant role. A few threads shaped the field:

- **Tong Hui Kang's Select-Patch-Verify starter** (discussion 553294) was the base of ranks 3, 4, 5, 8 and 15. His thread on the first notebook not scoring -1 (discussion 561695) showed what naive attempting costs: 1 right, 37 wrong, 33 skipped, for $(1 - 37 - 0.0033)/71 = -0.507$.
- **The leaderboard decoder** (discussion 557148) turned every score into a record; Geremie Yeo's thread established the 71-issue public set (discussion 557158).
- **The metric fix** (discussion 557133) and the upgrade that let submissions run unit tests inside the sandbox (discussion 552449). The winning approach depends on the latter.
- **Nebius's 80,036 SWE-agent trajectories and 6,411 GitHub issues** (discussion 552605): the largest external source, yet none of the seven writeups reports fine-tuning a patcher on it.
- **"How random is your submission?"** (discussion 568626), Camaro's thread with the 7/2 versus 4/5 result from the same code.

## 10. What did not work

| Idea | Reported by | Outcome |
|---|---|---|
| Reasoning models (QwQ-32B, DeepSeek-R1, long "but wait" traces) | Eduardo, Camaro, GraphNINJAs (QwQ) | Slower, no better than Qwen2.5-Coder-32B |
| Two models on separate GPUs, ensembled | Eduardo | Could not make it work |
| Agent mode or OpenHands | Eduardo (too late), Camaro (not attempted: hard to control) | In no final |
| More file candidates, a second pass, embeddings | Camaro | No help |
| A very strict verifier | Camaro | Rejects everything |
| Raw unified diffs from small models | Eduardo, Kenj | Far fewer valid patches than SEARCH/REPLACE |
| Rule-based diff repairs, multi-method lookup, prompt variation, AST checks, reusing repo pytest functions | Genxxsky | Random or no effect |
| Diff line-count checks; R1-Distill-Qwen-32B over 14B | Kenj | No gain |
| Keyword filters on patches | yuanzhe zhou | Marginal |
| An "unsure" third verifier label | GraphNINJAs | No significant difference |
| Many-patch "Choose" strategy | Genxxsky | 2 right, 3 wrong, versus 4 and 2 for the lean one |

The pattern: every idea that added generation without adding *evidence* failed.

## 11. Lessons that transfer

1. **When a wrong answer costs as much as a right one earns, build a selective classifier first.** Break-even precision is 50%, and Eduardo's "no reproducing test, no attempt", akami's LoRA gate and Kenj's traceback rule were each worth more than any change to the patcher.
2. **Make the model write a test before the fix.** The reproduction test is both the skip signal and the verifier; ranks 1, 2 and 4 executed tests, ranks 3, 5, 8 and 15 did not. The same idea underlies [repeated sampling](/blog/machine-learning/scaling-laws/repeated-sampling-scaling-laws): more samples help only when something can pick the right one.
3. **Ask small models for SEARCH/REPLACE edits, never raw unified diffs.** Ranks 1, 2 and 5 say so independently.
4. **Give small models the context large ones memorised.** On a contamination-free set you cannot count on the model having seen the repo.
5. **Do not tune abstention thresholds on a tiny public set.** Tune for stability across seeds, as the winner did.
6. **Track stage metrics, optimise the end metric.** Camaro's stage recall rose without the score following, the point our post on [evaluating agent trajectories](/blog/machine-learning/ai-agent/evaluating-agent-trajectories-beyond-final-answer) makes from the other side.

### If you enter a competition like this

- Decode the leaderboard on day one: with a fixed denominator and a tie-break term, every score is a confusion matrix.
- Write down your break-even precision and measure your system's precision before you tune recall.
- Put an executable check between generation and submission; LLM self-verification is a weak substitute.
- Use greedy decoding for steps that must be right; spend sampling diversity on candidates the verifier will filter.
- Expect a contamination-free private set to be harder than any public benchmark you validate on.

Related: [AIMO Progress Prize 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions), [AIMO 3](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions), [ARC Prize 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions), [ARC Prize 2025](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2025-solutions) and [Google Code Golf 2025](/blog/machine-learning/learn-from-kaggle/kaggle-google-code-golf-2025-solutions).

## Sources

Writeups (7 of the top 20):

- 1st, Eduardo Rocha de Andrade: https://www.kaggle.com/competitions/konwinski-prize/writeups/eduardo-rocha-de-andrade-1st-place-solution-write- ; code: https://www.kaggle.com/code/arc144/custom-agentless-fork
- 2nd, Camaro: https://www.kaggle.com/competitions/konwinski-prize/writeups/camaro-public-2nd-place-solution
- 3rd, akami: https://www.kaggle.com/competitions/konwinski-prize/writeups/3rd-place-solution
- 4th, Genxxsky: https://www.kaggle.com/competitions/konwinski-prize/writeups/4th-place-solution ; code: https://www.kaggle.com/code/genxxsky/4th-place-solution-select-patch-verify-test
- 5th, Kenj: https://www.kaggle.com/competitions/konwinski-prize/writeups/5th-place-solution ; notebook: https://www.kaggle.com/code/manhnguyen315/5th-place-solution
- 8th, first lucky gold!: https://www.kaggle.com/competitions/konwinski-prize/writeups/first-lucky-gold-8th-place-solution
- 15th, GraphNINJAs: https://www.kaggle.com/competitions/konwinski-prize/writeups/graphninjas-public-4th-place-private-15th-place-so

Competition and forum:

- Competition: https://www.kaggle.com/competitions/konwinski-prize ; host site: https://kprize.ai
- Select-Patch-Verify starter: https://www.kaggle.com/code/huikang/starter-notebook-select-patch-verify
- Leaderboard decoder: https://www.kaggle.com/competitions/konwinski-prize/discussion/557148
- Public set has 71 instances: https://www.kaggle.com/competitions/konwinski-prize/discussion/557158
- Metric update and rescore: https://www.kaggle.com/competitions/konwinski-prize/discussion/557133
- Competition upgrade plan: https://www.kaggle.com/competitions/konwinski-prize/discussion/552449
- First notebook not scoring -1: https://www.kaggle.com/competitions/konwinski-prize/discussion/561695
- Nebius trajectories and issues: https://www.kaggle.com/competitions/konwinski-prize/discussion/552605
- How random is your submission?: https://www.kaggle.com/competitions/konwinski-prize/discussion/568626
- Host wrap-up: https://www.kaggle.com/competitions/konwinski-prize/discussion/609204
