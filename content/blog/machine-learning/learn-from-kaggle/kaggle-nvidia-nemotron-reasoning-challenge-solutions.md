---
title: "NVIDIA Nemotron Reasoning Challenge: teaching a frozen model to run a solver"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "4,185 teams fine-tuned the same 30B model with a rank-32 LoRA, scored by greedy decoding under a 7,680-token cap. All 12 published top-20 writeups won with supervised fine-tuning on solver-written traces and nobody shipped RL. This is how the top 20 designed those traces, what the winner memorised and what it computed, and why the public board hid real progress."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "nemotron", "lora", "supervised-fine-tuning", "synthetic-data", "chain-of-thought", "greedy-decoding", "reasoning"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 22
---

> **TL;DR.** The NVIDIA Nemotron Model Reasoning Challenge (March to June 2026) gave 4,185 teams the same frozen model, Nemotron-3-Nano-30B-A3B, and let them submit exactly one thing: a LoRA adapter of rank at most 32. Kaggle scored it with greedy vLLM decoding, a 7,680-token answer budget and accuracy on several hundred hidden puzzles. NullSira won with 0.920 by writing rule-based solvers, generating 360,000 synthetic puzzles with deterministic solver traces, and training the model to replay them. **Decisive lever: data, specifically trace design.** **The one lesson:** under greedy decoding, the best policy is one you write in code and teach by supervised fine-tuning; the art is deciding what the model memorises and what it computes inside the trace.

![The lever strip: of the five levers that decided Kaggle LLM competitions, the Nemotron challenge was decided by data, in the form of solver-written chain-of-thought traces](/imgs/blogs/kaggle-nvidia-nemotron-reasoning-challenge-solutions-1.webp)

The strip above is the mental model for this post. Most Kaggle LLM competitions let you pick the model, so the field fights over which backbone fits the notebook. Here two levers were bolted down: the model was fixed and so was decoding. What was left was the text you trained on, and the top 20 turned into a contest of curriculum design for a machine that reads one token at a time and never gets a second sample.

In the [AIMO Progress Prize 3 post](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions) everyone sampled many answers and voted. Here voting is impossible: temperature 0 gives you one trace. The series opener, [how Kaggle LLM competitions work](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work), explains the format this post assumes.

## 1. The task in 60 seconds

![The Nemotron challenge budget card next to a redrawn toy bit-manipulation puzzle: five 8-bit input and output pairs and a query, with the hidden rule revealed underneath](/imgs/blogs/kaggle-nvidia-nemotron-reasoning-challenge-solutions-2.webp)

The competition ran from 2026-03-16 to 2026-06-15, hosted by NVIDIA with compute on Google Cloud G4 machines carrying RTX PRO 6000 Blackwell GPUs. Each puzzle shows a handful of input-output examples of a hidden transformation and asks for the output of one more input. `train.csv` held 9,500 labelled puzzles in seven families: bit manipulation on 8-bit strings (1,602 rows), symbol-encoded cryptarithms (823 rows: 659 "deduce", 164 "guess"), equations with numeric operators (732 rows: 596 deduce, 136 guess), text ciphers, Roman numerals, unit conversion and gravity. The hidden test was several hundred puzzles split 50/50 public and private; teams estimate about 250 in each half, which the host did not confirm in our sources.

You did not ship inference code. You shipped `submission.zip` with a LoRA adapter for `Nemotron-3-Nano-30B-A3B-BF16`, a hybrid Mamba and attention mixture-of-experts model. Kaggle loaded it into vLLM with `max_lora_rank 32`, `temperature 0.0`, `top_p 1.0`, `max_tokens 7680` and `max_model_len 8192`. Training happened wherever you liked, so budgets ranged from a Kaggle quota to rented clusters.

The toy in the figure is illustrative, not a competition row: the hidden rule is "rotate left by 1, XOR with shift right by 2", and the query `11001010` maps to `10100111`. Real bit puzzles combine up to three shifted or rotated copies of the input through a boolean function, recovered from about seven examples inside the trace. If the [Nemotron-H architecture](/blog/machine-learning/frontier-models/nemotron-h-hybrid-mamba-transformer) is new to you, the fact that matters is that its Mamba layers have their own `in_proj` and `out_proj` matrices, and the best teams put LoRA on them.

The field: 4,185 teams, 5,234 competitors, 71,952 submissions at five a day. The pool was USD 106,388: USD 50,000 cash plus twelve DGX Sparks at about USD 4,699 each. First took USD 25,000 and five Sparks, second USD 15,000 and two, third USD 5,000 and one; an Open Progress Prize (USD 5,000 and one Spark) went to the best score on 2026-04-09, and three Open Contribution Awards of one Spark each covered data, RL and fine-tuning contributions. (Our bank says "8 Sparks", but the allocation sums to 12 and only 12 matches the total. The contribution award recipients are unverified, so we do not name them.)

## 2. The metric, with a worked toy score

Kaggle extracts the answer from the last `\boxed{}` (falling back to heuristics such as the last number). A puzzle is correct if the string matches exactly or, for numbers, if the relative error is within one percent:

$$
\text{correct}_i = \mathbb{1}\left[\hat{y}_i = y_i\right] \;\lor\; \mathbb{1}\left[\frac{|\hat{y}_i - y_i|}{|y_i|} \le 10^{-2}\right],
\qquad
\text{score} = \frac{1}{N}\sum_{i=1}^{N} \text{correct}_i .
$$

![The metric on five toy answers: an exact match and a 0.64% numeric miss count as correct, a 1.9% miss, a wrong Roman numeral and a truncated trace with no box count as wrong, for a score of 2 out of 5](/imgs/blogs/kaggle-nvidia-nemotron-reasoning-challenge-solutions-3.webp)

Walk the toy rows. `42` against `42` matches. `3.16` against `3.14` is off by ${0.02 / 3.14 = 0.64\%}$ and counts. `10.7` against `10.5` is off by ${0.2 / 10.5 = 1.9\%}$ and fails. `XLI` against `XLII` is a string, and strings get no tolerance. The last row is the failure that shaped every trace in this competition: the model was still reasoning at 7,680 tokens, never wrote a box, and scored zero. The toy total is ${2/5 = 0.40}$.

With about 250 private puzzles, one puzzle moves the score by 0.004, so the 0.020 gap between first and third is five puzzles. The metric also had a bug: answers containing `}` were mishandled until a fix was deployed, without rescoring earlier submissions ([discussion 698106](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698106)).

## 3. What made it hard

**Greedy decoding turns training into a worst-case problem.** At temperature 0 the model emits the most likely token at every step, so a trace right on 999 tokens and wrong on one is a wrong answer. Tong Hui Kang, who won the Open Progress Prize and whose public code became the base for nearly every top team, framed the objective as **maximising the minimum log-probability** over trace tokens. If the gold token has probability above 0.5, its log-probability is above $\ln 0.5 = -0.693$ and it is guaranteed to be the argmax. Below that line it is not. A toy shows why the average lies:

```python
# Illustrative toy: why greedy decoding makes the MINIMUM token log-prob the
# number to watch. Per-token log-probs of the gold trace under two adapters.
import math

SAFE = math.log(0.5)   # -0.693: above this, the gold token is guaranteed argmax

adapters = {
    "A (great on average)": [-0.01, -0.02, -0.01, -1.20, -0.01, -0.02, -0.01, -0.01],
    "B (dull but even)":    [-0.15, -0.20, -0.18, -0.30, -0.22, -0.19, -0.25, -0.20],
}
for name, lps in adapters.items():
    mean, worst = sum(lps) / len(lps), min(lps)
    risky = [i for i, lp in enumerate(lps) if lp < SAFE]
    print(f"{name}: mean {mean:+.3f}  min {worst:+.2f}  "
          f"p(worst token) {math.exp(worst):.2f}  tokens below {SAFE:.3f}: {risky}")
```

```console
A (great on average): mean -0.161  min -1.20  p(worst token) 0.30  tokens below -0.693: [3]
B (dull but even): mean -0.211  min -0.30  p(worst token) 0.74  tokens below -0.693: []
```

Adapter A has the better average loss and can still derail at token 3, where the gold token has only 30% probability. Adapter B is never in danger. Cross-entropy minimises the average, so the work moved into the data: make every step so simple and consistent that no token is ever in doubt. Hui Kang's rules followed: deterministic traces; each token a tiny derivation from what is already written (division as repeated subtraction, classification by iterating rather than choosing among 77 words); rare operations covered; under budget. Diversity, conciseness and self-verification were explicit non-goals.

**The token budget is tight and the tokenizer is literal.** Nemotron's tokenizer spends roughly one token per bit or digit, and a trace that checks candidate rules against seven 8-bit examples burns tokens fast.

![Writing bit strings in hex: about 8 tokens per binary byte against about 2 in hex, and the 1st-place median bit-manipulation trace falling from 6,771 to 4,888 tokens, well under the 7,680 cap](/imgs/blogs/kaggle-nvidia-nemotron-reasoning-challenge-solutions-4.webp)

The chart is the 1st-place measurement: writing bit strings of four or more bits in hex cut the median bit trace from 6,771 to 4,888 tokens, a 27.8% reduction that turned "usually fits" into "fits with room for more search". The [tokenization bugs post](/blog/machine-learning/debugging-training/tokenization-bugs) covers the general trap.

**Two families are genuine search problems.** Every serious team solved cipher, numeral, unit and gravity at close to 100%. Bit manipulation and cryptarithms decided the board, and the 1st-place writeup puts the naive cryptarithm space at ${10! \cdot 24^3 \approx 5 \times 10^{10}}$ assignments, far beyond 7,680 tokens of greedy search.

The obvious alternative, reinforcement learning, was tried and dropped. Komil Parmar measured GRPO rollouts at about 2 tokens per second on Nemotron ([discussion 690161](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690161)), and Hui Kang made the deeper point: RL searches for a policy, but here "I already have the optimal policy". It is the solver. For calibration, Zhuang Jia's zero-shot comparison put Gemini-3.1-Pro at 0.81, Claude-Opus-4.6 at 0.78 and DeepSeek-V3.2 at 0.74 ([discussion 684283](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684283)). The winning adapter on a 3B-active model scored 0.920.

## 4. The top-20 scorecard

Public scores here are real. "Move" is public rank minus private rank; positive means the team rose. Ranks 3 and 4 tie at 0.900, 5 and 6 at 0.896, 7 and 8 at 0.888, 11 to 15 at 0.872, and 16 to 20 at 0.868. Twelve of the twenty published a writeup.

| Rank | Team | Private | Public (rank) | Move | Key idea | Source |
|---|---|---|---|---|---|---|
| 1 | NullSira | 0.920 | 0.912 (1) | 0 | Solver traces, signature catalog, hex bits | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution) |
| 2 | vli | 0.908 | 0.884 (5) | +3 | Reverse-engineered generators, exact LoRA export | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution) |
| 3 | YS-L | 0.900 | 0.872 (42) | +39 | Two-stage memorise-then-solve LoRA | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution) |
| 4 | /make-no-mistakes | 0.900 | 0.892 (3) | −1 | Script-audited "trainable traces" | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution) |
| 5 | Domdolus Tolus | 0.896 | 0.888 (4) | −1 | Infinite generator, LoRA soups | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/domdolus-tolus-solution) |
| 6 | Alehandreus & Yurnero | 0.896 | 0.900 (2) | −4 | All train as validation, per-domain probing | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-public-6th-private-place-solution) |
| 7 | Solved Bit Manipulation | 0.888 | 0.864 (531) | +524 | Bit string matching, no cryptarithm | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/7th-place-solution-solving-bitmanipulation-via-st) |
| 8 | Researcher 7919 | 0.888 | 0.876 (9) | +1 | unknown | no writeup |
| 9 | Fate | 0.884 | 0.880 (7) | −2 | Rewritten base traces, no synthetic | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot) |
| 10 | kansai-kanto-kaggler | 0.880 | 0.872 (24) | +14 | Key-table bit reasoner, α/r = 4 | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/10th-place-solution) |
| 11 | Hello World | 0.872 | 0.868 (82) | +71 | unknown | no writeup |
| 12 | Team | 0.872 | 0.876 (10) | −2 | unknown | no writeup |
| 13 | Haraguchi-T | 0.872 | 0.868 (84) | +71 | Heuristic reasoners, 3-stage overfit | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/13th-place-solution) |
| 14 | TaiDoDuc | 0.872 | 0.872 (62) | +48 | unknown | no writeup |
| 15 | MAJ0RT0M | 0.872 | 0.876 (16) | +1 | Memorised multiplication tables | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/wip-lb15-extension-to-huikangs-progress-prize) |
| 16 | toxu | 0.868 | 0.868 (71) | +55 | unknown | no writeup |
| 17 | Frank Yang | 0.868 | 0.856 (1736) | +1719 | unknown | no writeup |
| 18 | Y \| M \| F | 0.868 | 0.876 (8) | −10 | GT tie-breaks, SVD LoRA fusion | [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/18th-solution-from-deterministic-solvers-to-learn) |
| 19 | Ryuto Yoda | 0.868 | 0.856 (1781) | +1762 | unknown | no writeup |
| 20 | T T | 0.868 | 0.864 (322) | +302 | unknown | no writeup |

Caveat: the API gives 10th a public 0.872 but its writeup reports 0.860 for the selected run, so this column may show best rather than selected submissions (unverified).

![Public versus private rank for the top 20 on a log scale: the top six barely moved, while nine teams climbed from public rank 40 or worse, including 7th from 531 and 19th from 1,781](/imgs/blogs/kaggle-nvidia-nemotron-reasoning-challenge-solutions-5.webp)

The top was stable: the public top five all finished in the private top six. Below that, nine of the twenty came from public rank 40 or worse, two from beyond 1,700. At 0.004 a puzzle, two unlucky puzzles cost dozens of places in a field this crowded at 0.86 to 0.88.

## 5. The top-20 landscape

![Top-20 technique matrix across the 12 published writeups: SFT on deterministic solver traces is used by all 12, synthetic data by 10, held-out validation by 7, and hex encoding and memorised catalogs by only 3 each](/imgs/blogs/kaggle-nvidia-nemotron-reasoning-challenge-solutions-6.webp)

The matrix covers the twelve ranks with writeups. The eight without (8, 11, 12, 14, 16, 17, 19, 20) are unknown and we do not guess. Counts are recounted from the cells, "of 12" throughout.

**Table stakes.** All 12 trained by supervised fine-tuning on deterministic, solver-written traces; none shipped RL or distillation from a larger model. 10 of 12 say they built on Hui Kang's Progress Prize code or traces (2nd and 7th do not say). 10 of 12 trained on synthetic puzzles beyond `train.csv` (9th did not; 10th is unclear). Of the nine teams that state their LoRA shape, eight used rank 32 with α 32. Six put `lm_head` in the targets, and 2nd called his targets, Mamba projections and output layer included, "extremely important".

**What separated the top five from six to twenty.**

1. **Bit manipulation near 99%, in a trace the model can execute.** The Progress Prize solver covered 85.1% of bit puzzles (1,364 of 1,602). 1st reached 97.50% solver coverage, 4th a 99.4% reasoner, 10th a 98.75% one. But 10th finished at 0.880 with that solver: necessary, not sufficient.
2. **Cryptarithm via memorisation plus search.** Five of 12 built a real cryptarithm solver (ranks 1, 3, 6, 15, 18). Only 1st and 3rd paired it with memorised tables in the trace, and they are the only teams at or above 0.900 with real cryptarithm coverage. The counterexample is 2nd: 0.908 with cryptarithm "mostly untouched", carried by clean everything-else and an exact LoRA export.
3. **Validation discipline.** 7 of 12 kept `train.csv`, or a fold, out of training. 2nd, 3rd and 6th chose by validation, and 6th attributes its drop from public 2nd to a selection made against its own evidence. 18th (no local validation) and 13th (trained on everything) say they selected worse submissions than they had.

**Contradictions worth knowing.** Multi-stage training helped 1st, 3rd, 5th and 13th; 2nd found single-stage cross-entropy better and 10th found a second stage just "redistributed" wins and losses. LoRA soups carried 5th and SVD fusion was 18th's final submission, while 10th's seed soup "degraded to the level of the worst member". Hex encoding won for 1st and 6th, a 5-bits-per-letter code broke 5th's model, and 7th went the other way to one token per bit. Small batches won for 3rd (4) and 4th (16) while 9th, 10th and 13th used 32. And removing traces whose solver was wrong hurt 10th, while 1st found a deliberately mismatched trace taught the model something valuable. That last one is the best story in the competition.

## 6. The winning solution, explained: NullSira

NullSira (GoodMeatDay, re and reopon) framed everything as one question: **what should the model memorise through traces, and what should it compute inside the trace?**

![The 1st-place pipeline: train.csv statistics drive a rule sampler, a rule-based solver writes each trace, then a 220,000-sample main SFT and a 140,000-sample bit-heavy continuation produce the submitted checkpoint](/imgs/blogs/kaggle-nvidia-nemotron-reasoning-challenge-solutions-7.webp)

**Synthetic data, and no LLM wrote it.** They fitted statistics from `train.csv`, sampled a prompt structure, instantiated a hidden rule, rendered it in the competition template, and attached a rule-based solver trace. The main set was 220,000 samples and 890.1M tokens: 120,000 cryptarithm, 40,000 bit, 20,000 cipher, and 10,000 each of equation, numeral, unit and gravity. A bit-heavy continuation pool added 140,000 samples and 590.7M tokens. Validation was 1,424 rows, 15% of `train.csv`.

**Ordinary training on purpose.** Unsloth, LoRA rank 32 and α 32 on attention q/k/v/o, Mamba `in_proj`/`out_proj`, MLP up/down/gate and `lm_head`; `adamw_bnb_8bit`, cosine, effective batch 16. The main stage ran one epoch at 2e-4 for about 119 hours, the continuation 6,500 steps at 1e-5 for about 51 hours, on one 96 GB RTX PRO 6000. The final checkpoint, `sft/106`, scored 91.57% on validation. For the knobs, see [effective LLM fine-tuning techniques](/blog/machine-learning/large-language-model/effective-llm-fine-tuning-techniques) and [debugging LoRA and PEFT](/blog/machine-learning/debugging-training/debugging-lora-and-peft).

**Equation numeric: a 24-rule inventory.** Every operator in train was one of 24 rules (join, swapped join, add or multiply with ±1 offsets, digit-reversing "flip" variants, absolute difference, max-mod-min). Two constraints shrank the search: within a puzzle all non-join operators are normal or all flipped, and they never share a rule group.

**Cryptarithm: a signature catalog the model memorises.** This moved them from good to first. Join and swapped join solve about 5% cheaply, so the trace tries them first. For the rest, the digit pattern of left operand, right operand and output is normalised by first appearance, so `89 + 11 = 100` becomes `ABCCCDD`. Enumerating 00 to 99 per operand across 22 rules gave 4,205 signatures, each pointing to a few candidate (rule, digits) rows. The model learns that table by seeing it used across 120,000 traces, recalls the candidates for the first equation, then runs a depth-first consistency check on the rest inside the trace. The seed was youkinasa's public post on signature-keyed tables ([discussion 701981](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/701981)). That component alone reached 0.89 public and 0.908 private.

![A toy signature catalog: the symbol equation normalises to the signature ABCBBDA, the memorised catalog returns three candidates, the second equation rejects two of them, and the survivor answers the query](/imgs/blogs/kaggle-nvidia-nemotron-reasoning-challenge-solutions-8.webp)

The figure and code are a toy with five rules and two-digit operands, not the team's code. The expensive enumeration happens once, offline; the model only recalls one row and verifies a few candidates.

```python
# Illustrative toy, not any team's code: the signature-catalog idea behind
# the 1st-place cryptarithm traces, shrunk to 5 rules and 2-digit operands.
from itertools import product

RULES = {
    "add":     lambda a, b: a + b,
    "sub":     lambda a, b: a - b,
    "absdiff": lambda a, b: abs(a - b),
    "mul":     lambda a, b: a * b,
    "concat":  lambda a, b: int(f"{a:02d}{b:02d}"),
}

def signature(s: str) -> str:
    """Rename symbols by first occurrence: '8911100' -> 'ABCCCDD'."""
    names = {}
    return "".join(names.setdefault(ch, chr(65 + len(names))) for ch in s)

# Offline: enumerate every (rule, a, b) once and index it by signature.
# This is the table the model memorises through repeated SFT traces.
CATALOG = {}
for (rule, fn), a, b in product(RULES.items(), range(100), range(100)):
    out = fn(a, b)
    if out >= 0:
        CATALOG.setdefault(signature(f"{a:02d}{b:02d}{out}"), []).append((rule, a, b, out))

def solve(equations, query):
    """equations: list of (left, right, output) symbol strings, one shared operator."""
    left, right, out = equations[0]
    for rule, a, b, o in CATALOG.get(signature(left + right + out), []):
        digits = dict(zip(left + right + out, f"{a:02d}{b:02d}{o}"))
        ok = True
        for l, r, res in equations[1:]:          # in-trace consistency check
            if any(ch not in digits for ch in l + r + res):
                ok = False; break
            got = RULES[rule](int("".join(digits[c] for c in l)), int("".join(digits[c] for c in r)))
            want = int("".join(digits[c] for c in res))
            print(f"  try {rule}: {''.join(digits[c] for c in l)} op {''.join(digits[c] for c in r)} = {got}, needs {want}")
            ok = ok and got == want
        if ok:
            inv = {d: s for s, d in digits.items()}
            val = RULES[rule](int("".join(digits[c] for c in query[0])), int("".join(digits[c] for c in query[1])))
            return rule, val, "".join(inv.get(d, "?") for d in str(val))
    return None

print(len(CATALOG), "signatures in the toy catalog")
print(solve([("@#", "%#", "#&@"), ("@%", "&@", "%&%")], ("&@", "%&")))
```

```console
2189 signatures in the toy catalog
  try add: 28 op 02 = 30, needs 808
  try mul: 14 op 31 = 434, needs 434
('mul', 1333, '@&&&')
```

The first equation's signature `ABCBBDA` has three toy candidates: `21 + 81 = 102`, `19 × 49 = 931` and `48 × 18 = 864`. The second equation kills the addition and accepts the first multiplication, so the code never needs the third (the figure checks all three). The query `&@ } %&` becomes `31 × 43 = 1333`, re-encoded as `@&&&`. Even five rules give 2,189 signatures.

**Bit manipulation: hex, more primitives, catalog repair.** Starting from Hui Kang's per-output-bit solver, they wrote bit strings in hex (the 27.8% cut above), added majority, three-input parity, conditional select and three-input compositions, and "repaired" a doubtful eight-rule sequence by checking nearby valid sequences from a precomputed catalog. Memorisation again, used to make search cheap.

**The accident that taught the format.** In a bit puzzle where the local solver disagreed with ground truth, they built a sample with the solver's trace but the true answer in the final `\boxed{}`. "The model learned the format. In fact, we saw cases where the solver would be wrong, but the trained model changed only the final answer and became correct" ([discussion 709231](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/709231)). MAJ0RT0M (15th) independently saw his model beat its reference solver by about 1% on bit manipulation. A trained model is not a copy of its teacher.

On `train.csv` (solver within budget, then model): bit 97.50% and 93.45%; cryptarithm deduce 42.94% and 38.09%, guess 12.80% and 12.20%; equation guess 52.94% and 52.21%; total 93.16% and 92.00%, which they flag as leaky because the rules came from those rows. Even at temperature 0 vLLM was not fully deterministic, so leaderboard checks were noisy; the final submission was the continuation adapter, submitted twice. And: "All code was written with Codex… the ideas… almost never came from Codex."

## 7. 2nd and 3rd in depth, then the rest of the top 20

### 2nd: vli (Victor Li, solo, 0.908)

Victor Li named three ingredients: rapid iteration on synthetic prompts and traces, discipline in uncovering the data-generation process, and clean traces. Claude wrote about 95% of his data-generation code. All 57,600 training examples (159M tokens) were synthetic, so the whole of `train.csv` was validation: 0.9107 for the solver, 0.9074 for the LoRA, bit manipulation 0.9906 and 0.9732, cryptarithm 0.096. Plain cross-entropy beat focal loss, token reweighting and multi-stage schedules.

His most transferable finding is about infrastructure. Tinker's LoRA needed a lossy SVD to become the PEFT format vLLM loads; switching to Megatron-Bridge, which maps weights exactly, raised bit accuracy from 0.81 to 0.89 with nothing else changed. Training ran one epoch, 2e-4 to 1e-5 cosine, packed 8,192-token sequences, global batch 12, four A100 80 GB GPUs for 11 hours; expert parallelism accidentally tied the LoRA across experts about four ways. Returns diminished sharply after about 80M tokens. (15th found Tinker's SVD conversion harmless in his setup, so the loss is configuration-dependent.)

### 3rd: YS-L (solo, 0.900, from public 42nd)

YS-L wanted traces "without any hidden deduction or jumps": a minimal numeric solver with better guessing, hex-compacted bit traces with majority search, and a CSP-style cryptarithm solver with multiplication patterns and backtracking that solved 24.2% within budget. The distinctive move was **two-stage training**. Stage 1 taught only recall, with drill prompts like `<RECALL_DOMAINS> sig=… type=…`; stage 2 started from that LoRA and trained on full traces with 3,000 drills mixed in so the memory did not decay. On 12,000 cryptarithm samples, direct training solved 0 of 11 solvable problems and two-stage 7 of 11; on the final mix, 12 of 17 against 16 of 17, which was private 0.888 against 0.900.

He used Hui Kang's loop with cut cross-entropy, rank 32 and α 32 in fp32 including `lm_head`, MoE weights untied, and an **effective batch of 4** (larger batches under-trained): 25.2 hours for stage 1 and 32.7 for stage 2 on one RTX PRO 6000. He validated on a 954-row stratified holdout, tracked a "perfect-trace rate", and ignored the public board: "the wall at 0.86 is clearly visible on the public LB". One of his 0.85-public submissions was 0.90 private.

### Ranks 4 to 20, by approach family

**Trace engineers.** *4th, /make-no-mistakes (Dipam Chakraborty, 0.900)* made learnability testable: no hidden compute, every line derivable from the 10 to 20 lines above it, trace shape dependent only on what is written, few uniform rules. Scripts audited locality and references, and Claude (Opus 4.8) iterated formats in an hour-long "train, diff, classify, reformat, retrain" loop. Everything beyond one or two ops fit five templates; his reasoner solved 1,593 of 1,602 (99.4%), his traces hit over 95% with about 2,500 examples where the base reached about 82%, and the whole run cost about USD 120. *9th, Fate (Chi-Han Tsai, 0.884)* used no synthetic data: he rewrote the base traces until targets were correct and auditable (bit 1,364 to 1,447 correct targets) and separated target errors from imitation errors in validation. *10th, kansai-kanto-kaggler (0.880)* rewrote the bit reasoner to the generator form, with the boolean function read off as a 3-bit key table, reaching 98.75%; corpus statistics reordered equation candidates (guess 21 to 56 of 136), and α/r = 4 improved validation. *13th, Haraguchi-T (0.872)* added heuristics (oracle 87.7% to 89.4%) and overfit on purpose in three stages at falling learning rates.

**Generator builders.** *5th, Domdolus Tolus (0.896)*, five people with near-zero fine-tuning experience, used no `train.csv` rows in training: an "infinite" generator per family, train kept for validation, about 2 billion tokens over 15 SFT iterations, the later ones as LoRA soups. Validation rose from 79.0 to 90.4 while public stalled near 88; they picked the most stable adapter (89.6 private) over the highest-validation one (90.0 private) because the latter varied between resubmissions. *6th, Alehandreus & Yurnero (0.896, public 2nd)* used all 9,500 rows as validation and probed the board per domain by training every other family to answer `\boxed{DUMMY}`. Their DFS cryptarithm tried reversed numbers when forward failed (about 40% of solves). Choosing a 0.25B-token set over a 0.5B set that doubled cryptarithm solves, they believe, cost them 2nd or 3rd. *18th, Y | M | F (0.868)* used ground truth to break ties among example-consistent candidates and fused adapters by averaging and truncated SVD back to rank 32, which beat TIES and DARE in their tests.

**Memorisers.** *15th, MAJ0RT0M (0.872)*, in his first Kaggle competition, kept the base hyperparameters but trained about 2,000 steps instead of 200, built arity-3 truth tables for middle bits (reference 85% to 92.1%), and followed kenshim's idea ([discussion 703479](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/703479)) of recalling a memorised multiplication table keyed by digit pattern. Of 536,583 examples, 503,682 were table drills. He estimates his changes were worth about seven private problems.

**Best contrasting entry: 7th, Solved Bit Manipulation (0.888).** Five people who joined in the last ten days, did **no cryptarithm training at all**, and rose from public 531st. They treated bit puzzles as feature selection (string similarity between outputs isolates primitive "bases" with no arithmetic), forced one token per bit, the opposite of hex, and used dynamic loss masking so the model learns to hypothesise and backtrack (arXiv 2606.23672). Their hyperparameters are not stated, and their claim of the best bit accuracy of any team is unverified; 4th's 99.4% is a reasoner figure, not comparable.

**No writeup:** ranks 8, 11, 12, 14, 16, 17, 19 and 20. We found no posts, so their methods are unknown.

## 8. What the forum taught everyone

One thread defined the field. **Hui Kang's "SFT to maximize minimum logprob"** ([discussion 689915](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915), 242 votes, [code](https://github.com/tonghuikang/nemotron)) won the Progress Prize at the midpoint and gave everyone the doctrine, the trainer and the reasoners. Its cost discipline is a model: 27,850,703 training tokens in the winning submission out of 598,958,637 explored, for about USD 212 on Tinker, USD 60 on Modal and USD 10 of subscriptions. His bit thread ([discussion 690307](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307)) searched per output bit over inputs, NOT, constants and two-input AND/OR/XOR, solving 1,364 of 1,602 against 35 of 160 for the earlier public notebook. He also published a browser of base-model completions ([discussion 684212](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212)).

Donald Galliano III called slow Kaggle Blackwell training "pay-to-play" ([discussion 688461](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461)), yet 4th and 5th each spent only about USD 100 to 120.

The shake-up lesson came from writeups. A public "wall" around 0.86 to 0.88 hid progress for 3rd, 5th and 18th. 10th chose a run with lower public (0.860) and higher validation (0.904) that beat its best-public run by 0.028 on private. 18th dropped its synthetic direction because public looked worse (0.868 against 0.872) though private was better (0.872 against 0.864). And 1st and 5th saw the same adapter vary by up to about 1% between submissions: two or three puzzles of pure noise.

## 9. What did not work

| Idea | Who reported it | What happened |
|---|---|---|
| RL / GRPO | 5th; Parmar; Hui Kang | Air-gapped notebooks, compile errors, OOM, ~2 tok/s rollouts |
| Pseudo-GRPO via iterated DPO | 5th | Could not learn easy bit problems |
| Distilling a larger LLM | Hui Kang | Rejected by design; no top team did it |
| Short CoT without visible logic | 5th | Failed even with 250k examples |
| Compact letter encoding of bits | 5th | "Encoding is breaking the model" |
| Unsloth with 4-bit and packing; DoRA | 5th | No learning; vLLM rejects DoRA |
| Focal loss, token reweighting | 2nd; 15th | Worse than plain CE; pushed tokens toward −0.69 |
| Multi-stage schedules | 2nd; 10th | Single-stage won; wins and losses redistributed |
| Lossy SVD LoRA export | 2nd (15th disagrees) | Bit accuracy 0.81 versus 0.89 |
| Curriculum; 3× duplication; dropping failed traces | 10th | No gain; overfit; hurt |
| Cryptarithm traces with logical jumps | 10th; 5th | Unlearnable |
| Seed soup, SVD merge; TIES and DARE | 10th; 18th | Fell to the worst member; lost to SVD fusion |
| Doubling cryptarithm at the expense of bit | 6th | Bit validation fell |

The pattern: anything that made the target less deterministic, or a step less locally derivable, lost. If you want RL anyway, [fine-tuning LLMs with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) shows where it shines: when you do not have a solver, which is exactly the case this competition removed.

## 10. Lessons that transfer

1. **When decoding is greedy and you can write the policy, write it in code and teach it by SFT.** All 12 published teams did, and frontier models zero-shot scored 0.74 to 0.81 against 0.920. (689915; 709231; 711703)
2. **Optimise the worst token, not the average.** Make every step a trivial derivation and measure first-token divergence or a perfect-trace rate, not just loss. (689915; 712419; 709136)
3. **Decide what to memorise and what to compute.** A recalled table plus a short in-trace search beats a long search, and teaching the table first helps. (709231; 709136; 711727)
4. **The tokenizer sets your budget.** Hex cut 1st's bit traces by 27.8%; a bad encoding broke 5th's model. (709231; 708538)
5. **Hold out real data and believe it over the public board.** 3rd, 5th, 10th and 18th document public scores that hid private gains. (709136; 708535; 715330)
6. **Your training stack is part of the model.** A lossy export cost 2nd eight points of bit accuracy. (711703)

### If you enter a competition like this

- Read the grader's decoding settings first; temperature 0 and a token cap redefine good training data.
- Write a solver per family before a training loop, and measure its coverage within the budget.
- Generate synthetic data from fitted train statistics and keep train for validation.
- Audit traces mechanically: no jumps, no hidden compute.
- Count tokens per reasoning step under the real tokenizer.
- Put LoRA on every projection, including Mamba `in_proj`/`out_proj` and `lm_head`.
- Round-trip the adapter through the grader's loader and compare per-family accuracy.
- Resubmit one adapter to measure your noise floor before trusting a 0.004 move.

For the same question with free model choice and sampling, compare [AIMO Progress Prize 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions) and [AIMO Progress Prize 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions). For hidden-transformation puzzles without a fixed model, see [ARC Prize 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions) and [ARC Prize 2025](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2025-solutions); for programs under a size budget, [Google Code Golf 2025](/blog/machine-learning/learn-from-kaggle/kaggle-google-code-golf-2025-solutions) and [NeuroGolf 2026](/blog/machine-learning/learn-from-kaggle/kaggle-neurogolf-2026-solutions). The model family is covered in the [Nemotron model family timeline](/blog/machine-learning/frontier-models/nemotron-model-family-timeline).

## Sources

- Competition: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- NVIDIA recap (2026-07-14): https://developer.nvidia.com/blog/lessons-from-the-leaderboard-what-5000-kagglers-taught-us-about-improving-ai-reasoning/
- Progress Prize, Tong Hui Kang: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915 ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307 ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684212 ; https://github.com/tonghuikang/nemotron
- 1st: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution ; https://github.com/xrwr/kaggle-nvidia-nemotron-model-reasoning-challenge-1st-place
- 2nd: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution ; https://github.com/livctr/nvidia-nemotron
- 3rd: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution ; https://github.com/YS-L/nvidia-nemotron-reasoning-3rd-place-solution
- 4th: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution
- 5th: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/domdolus-tolus-solution
- 6th: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-public-6th-private-place-solution
- 7th: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/7th-place-solution-solving-bitmanipulation-via-st (arXiv 2606.23672)
- 9th: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot
- 10th: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/10th-place-solution
- 13th: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/13th-place-solution
- 15th: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/wip-lb15-extension-to-huikangs-progress-prize
- 18th: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/18th-solution-from-deterministic-solvers-to-learn
- Threads: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/701981 ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/703479 ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690161 ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/684283 ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/688461 ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/698106
