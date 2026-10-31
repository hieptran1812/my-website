---
title: "Synthetic data as the edge: how Kaggle LLM winners generated data that matched a test they could not see"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "A cross-competition synthesis of ten Kaggle LLM competitions (DAIGT, PII, Eedi, MAP, AIMO 1 and 3, ARC 2024 and 2025, Deep Past, Nemotron): which generator the winners chose, how they bought diversity, injected noise, filtered, staged and scaled synthetic data, and when it did not help."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "synthetic-data", "data-generation", "distribution-shift", "data-augmentation", "filtering", "curriculum"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 13
---

> **TL;DR.** In ten Kaggle LLM competitions, the team that won was usually the team whose *training data* looked most like a hidden test set nobody could see. The model was a commodity: DeBERTa-v3-large in all 13 PII writeups, Qwen2.5 in almost every Eedi stack, ByT5 in all 17 Deep Past writeups. What differed was the data. Winners picked a generator that wrote like the test, bought diversity instead of volume, put the test's noise on both classes, filtered every generated row with something independent of the generator, and trained broad first and target-like last. Where nobody trained at all (AIMO 3), synthetic data decided nothing, and that is a finding too.

![One generator's cluster misses the hidden test region; many generators surround it](/imgs/blogs/kaggle-llm-synthetic-data-as-the-edge-1.webp)

The picture above is the mental model for this post. You never see the hosts' generator; you see a training file that is too small, too clean or from the wrong prompts. One generator at any scale lands in one place. Many generators, decoding settings and kinds of noise surround the region, so wherever the test sits, something you trained on is nearby. That is how the DAIGT winners described their 160k-row mix.

This is a technique post in the [Learn from Kaggle LLM series](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work). Each claim names the competition and team and links its top-20 synthesis, where the counts were recomputed from the writeups.

## 1. Why the training file is never enough

Every competition here had the same shape of problem: the labelled data you were given did not cover what the private leaderboard would ask.

- **DAIGT** (detect AI-written essays): training covered 2 of 7 prompts and was "nearly all" human; the hidden test had other prompts, unknown generators and injected character noise.
- **PII Data Detection**: 6,807 training essays, only 945 (about 14%) with any PII.
- **Eedi**: about 2,587 misconceptions to rank, and "most misconceptions appear only once in train".
- **ARC 2024**: 400 public training tasks, which the winners called too few.
- **Deep Past**: roughly 1,500 translated documents, with tens of thousands more locked in scanned books.
- **Nemotron**: 9,500 labelled puzzles, and you shipped only a LoRA adapter, so every skill had to be taught through training traces.

No model capacity closes a gap like that; a model interpolates between what it has seen. So the winners' first question was not "which backbone" but "what does the hidden test look like, and how do I manufacture rows near it".

## 2. Ten competitions, side by side

Each row: the gap the competition posed and the synthetic move that answered it. "Measured" is what a writeup reported.

| Competition | Team | Hidden-test gap | Synthetic move | Measured |
|---|---|---|---|---|
| [DAIGT](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions) | 1st, Bamba, Biswas, Broad | Unknown generators, injected noise | 160k mix from many LLM families plus attack augmentations | AUC 0.987824; "the modeling approach was less important" |
| [DAIGT](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions) | 2nd, Guanshuo Xu | Student style | 500k SlimPajama pairs, then essays from LLMs fine-tuned to mimic students | Private 0.967 to 0.98 |
| [PII](/blog/machine-learning/learn-from-kaggle/kaggle-pii-data-detection-solutions) | 1st, Fold Zero; 9th, Broad | Rare PII | Mixtral essays with planted PII; ~2,000 own essays | Broad's set "by far the most useful" |
| [Eedi](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions) | 1st, MTH 101 | Unseen labels | Claude questions for confusable clusters, judged, deduplicated | +0.044 from teacher pseudo-labels on them |
| [MAP](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) | 2nd, 7th | Label noise ceiling | 80k (2nd), ~50k of ~200k (7th) LLM explanations | +0.002 to +0.005 (7th); none after fusion (2nd) |
| [AIMO 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions) | 1st, Numina | Olympiad problems, integer answers | ~60k GPT-4 tool-integrated traces after a CoT stage | Public 8 to 16 to 27 of 50 |
| [AIMO 3](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions) | top 5 | None addressed by data | No training at all: 0 of 5 | Ties at 44.0 and 43.5 |
| [ARC 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions) | 1st, the ARChitects | 400 tasks too few | Re-ARC generators plus ARC-Heavy: 531,318 examples | 53.5% private |
| [ARC 2025](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2025-solutions) | 1st, NVARC | Harder ARC-AGI-2 | 103,253 gpt-oss-120b puzzles kept by program agreement | 24.03%; only the 3 pretrained teams cleared 7% |
| [Deep Past](/blog/machine-learning/learn-from-kaggle/kaggle-deep-past-akkadian-translation-solutions) | 1st DataTech Club; 3rd | Data locked in scans | Gemini-3.0-pro reads page images (1st); ~150k Claude drills (3rd) | +3.5 from continued pre-training (5th) |
| [Nemotron](/blog/machine-learning/learn-from-kaggle/kaggle-nvidia-nemotron-reasoning-challenge-solutions) | 1st, NullSira | Skill must live in a LoRA | 360,000 puzzles with solver traces, no LLM | 0.920; 10 of 12 writeups used synthetic puzzles |

The matrix turns the same ten rows into strategies. A filled cell means the winner used the strategy; a half cell means another top-20 team did; an empty cell means no top-20 writeup reports it.

![Matrix of ten competitions by seven synthetic-data strategies](/imgs/blogs/kaggle-llm-synthetic-data-as-the-edge-2.webp)

The first column is the densest: an LLM wrote or extracted the data in 7 of the 10 winners' pipelines, and 5 winners matched the generator to the target. Noise injection is the sparsest column, because only DAIGT had a test with deliberate noise in it. The AIMO 3 row is empty across the board, and its winners still tied at the top. Read it as "what the competition demanded", not "what is always good".

## 3. Choosing the generator

The competitions give a consistent answer to "which model writes the data": the one whose *style* matches the test, not the one with the best benchmark.

**Base beats instruct when the test is not chat.** In DAIGT, Hao Mei (7th) generated only with base Falcon-7B, Mistral-7B and Llama-2-7B, arguing the hosts' samples lacked chat formatting. Uesugi Erii (12th) and RB (13th) fine-tuned LLMs on PERSUADE and saw quality drop; RB's own diagnosis was that she used Mistral-7B-*Instruct* while 2nd used base models.

**Fine-tune the generator on the target domain.** Guanshuo Xu (DAIGT 2nd) fine-tuned LLMs on student essays with H2O LLM Studio "down to citations and typos". In PII, Yusuke Nakayama (6th) was the only top-20 team to fine-tune the generator: Mistral-7B-Instruct mapped a tool name, the essay's PII and the hosts' instruction to an essay, 1,722 of them.

**Use a frontier model where reasoning quality matters.** Eedi needed a model that understands *why* a student is wrong: 13 of 15 writeups generated synthetic questions, 10 with Claude 3.5 Sonnet, GPT-4o or Gemini 1.5 Pro. AIMO 1's two prize-winning trainers, Numina and CMU_MATH, both fine-tuned on GPT-4-written solutions; the inference-only teams at ranks 3 to 5 scored 21, 21 and 20 against Numina's 29.

**Use no LLM at all when a program can write the answer.** NullSira (Nemotron 1st) fitted statistics from `train.csv`, sampled a prompt structure, instantiated a hidden rule and attached a rule-based solver trace. Their main set was 220,000 samples (890.1M tokens) plus a 140,000-sample bit-heavy pool. Victor Li (2nd) reverse-engineered the generators and trained on 57,600 purely synthetic examples, freeing all of `train.csv` for validation.

**Use the LLM as a reader, not a writer.** Deep Past's winner, DataTech Club, let Gemini-3.0-pro read page images of scholarly books into sentence pairs; at least 15 of 17 writeups had an LLM or VLM build their training data, but only 2 of 17 let a decoder LLM translate in the final submission.

## 4. Diversity and matching the test distribution

Volume without diversity buys the left panel of the first figure. The winners spent effort on *coverage*.

**Surround the unknown.** The DAIGT winners built their mix round by round, each round plugging a gap: human text from all 15 PERSUADE prompts, generated text from proprietary, open and PERSUADE-fine-tuned LLMs, and decoding diversity (contrastive search, typical_p, high temperature, large top-k, rewrites, fill-in-the-blank). James Day's forum thread made the point with a number: a DeBERTa at roughly 0.75 to 0.80 on the leaderboard reached 0.958 to 0.963 with about 500k Pile continuations from 12 LLMs. 7 of the 14 DAIGT writeups generated more than 100k rows, including four of the top five.

**Generate for the decision boundary, not the label.** Raja Biswas (Eedi 1st, also on the DAIGT-winning team) clustered misconceptions that his retriever and rankers confused on validation, then had Claude write 5 new, harder questions per cluster from 4 to 5 reference questions. The model trains on exactly the distinctions it was getting wrong.

**Fit the generator to the real data's statistics.** NullSira and Domdolus Tolus (Nemotron 5th, an "infinite" generator per family, about 2 billion tokens) rendered puzzles in the competition template from statistics of the training file, so synthetic and hidden puzzles share a distribution.

**Mix seeds to reach harder regions.** NVARC (ARC 2025 1st) described 716 ARC-AGI-2 training puzzles with gpt-oss-120b, using H-ARC's human descriptions and 160 BARC descriptions, then mixed pairs of summaries into 266,593 new, harder puzzle descriptions. Mixing moves outward from the seeds without leaving the task.

## 5. Noise injection: make the shortcut uninformative

DAIGT is the cleanest case. Typos were the easiest signal: students make them, LLMs do not, and a 14-line typo counter scored 0.714. Then the hosts injected character-level noise into the test, so generated essays carried typos too, and any model that had learned "typo means human" was wrong in a systematic direction.

![Typos as signal in train, as injected noise in test, and on both classes after the fix](/imgs/blogs/kaggle-llm-synthetic-data-as-the-edge-3.webp)

The field split in two. Removing the noise (spell checkers, a T5 deobfuscator) was the riskier bet: 8th and 12th reported it hurt, and LLMLab's deobfuscation (9th) gave +0.006 public and -0.001 private. Training through it won. Ertugrul & Chase (4th) injected random typos into 30% of generated essays; the winners applied character delete, insert and swap, back-translation and capitalisation attacks to a random subset. The toy below, which is illustrative and not any team's code, shows the mechanism:

```python
# Illustrative toy, not any team's code: put noise on BOTH classes so it stops carrying the label.
import random

def add_typos(text: str, rate: float, rng: random.Random) -> str:
    """Randomly insert, delete or replace letters, like a character-level obfuscation."""
    out = []
    for ch in text:
        r = rng.random()
        if ch.isalpha() and r < rate / 3:
            continue                                          # delete
        if ch.isalpha() and r < 2 * rate / 3:
            out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))  # replace
            continue
        out.append(ch)
        if ch.isalpha() and r < rate:
            out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))  # insert
    return "".join(out)

def noisy_share(rows, label):
    return sum(t != c for t, c, y in rows if y == label) / sum(y == label for _, _, y in rows)

rng = random.Random(0)
clean = "Students should have more choice in their classes because it helps them learn."
rows = []
for i in range(1000):
    y = i % 2                       # 1 = generated, 0 = human
    if y == 0:
        text = add_typos(clean, 0.03, rng)                # humans make typos
    else:
        text = add_typos(clean, 0.03, rng) if rng.random() < 0.30 else clean  # 30% of generated
    rows.append((text, clean, y))

before = [(c if y else t, c, y) for t, c, y in rows]   # generated essays left clean
print(f"human essays with a typo:              {noisy_share(rows, 0):.2f}")
print(f"generated with a typo, no augmentation: {noisy_share(before, 1):.2f}")
print(f"generated with a typo, 30% augmented:  {noisy_share(rows, 1):.2f}")
```

```console
human essays with a typo:              0.88
generated with a typo, no augmentation: 0.00
generated with a typo, 30% augmented:  0.22
```

Without augmentation, a typo is a perfect "human" flag and the model leans on it. With 30% of generated rows noised, about a fifth of generated essays carry one too, so the flag stops being a certainty. The rule: find the shortcut the test will break, and put it on both sides of the label.

## 6. Filtering: the generator is not the judge

Generation is cheap; a wrong answer key is expensive, because the model learns it with full confidence. Strong pipelines put something *independent of the generator* between generation and training.

![NVARC's program-agreement funnel and Eedi 1st place's similarity dedup zones](/imgs/blogs/kaggle-llm-synthetic-data-as-the-edge-4.webp)

**Agreement between independent programs.** NVARC went from 266,593 mixed summaries to 126,901 puzzles with a working input generator and unit test, then to 103,253 where independently written output programs agreed. Agreement is not proof, but it cheaply removes the commonest failure:

```python
# Illustrative toy, not NVARC's code: keep a synthetic puzzle only if
# independently written output programs agree on every generated input.
import random

def gen_input(rng):                      # the "input generator" an LLM would write
    return [rng.randint(0, 9) for _ in range(rng.randint(3, 8))]

# Three candidate programs for the same description: "reverse the row, then add 1 mod 10".
def prog_a(row): return [(x + 1) % 10 for x in reversed(row)]
def prog_b(row): return [(row[-1 - i] + 1) % 10 for i in range(len(row))]
def prog_c(row): return [(x + 1) % 10 for x in row[::-1]][:-1] + [row[0]]  # off-by-one: no +1 on last cell

def agree(programs, trials=50, seed=0):
    rng = random.Random(seed)
    for _ in range(trials):
        row = gen_input(rng)
        outs = {tuple(p(row)) for p in programs}
        if len(outs) > 1:
            return False
    return True

print("a + b agree:", agree([prog_a, prog_b]))     # keep the puzzle, the answer key is trusted
print("a + c agree:", agree([prog_a, prog_c]))     # drop it rather than teach a wrong answer
```

```console
a + b agree: True
a + c agree: False
```

**A different model as judge.** Eedi's winner had GPT-4o score 0 to 10 how logically each misconception leads to the wrong answer, dropped low scores, and deduplicated Claude's invented misconceptions by embedding similarity: at least 0.995 counts as the same label, 0.95 to 0.995 is dropped as ambiguous, anything below is new. Over 4,000 new misconceptions survived. CQYR (Eedi 2nd) had gpt-4o-mini rate rows and dropped those scoring 2 or less out of 5.

**Your own model as the filter.** Study Hard (Eedi 9th) kept GPT-4o questions for known misconceptions only when the best retriever put the target in its top 5 to 10. CMU_MATH (AIMO 1 2nd) kept GPT-4 code solutions only when they reached the correct answer.

**Iterative cleaning passes.** Eleftheria 14 (Deep Past 9th) extracted about 95k pairs and ran more than 30 cleaning passes: sample a batch, have Claude Opus judge alignment and OCR errors, find the systematic error class, write a programmatic fix. Amed (PII 8th) dropped two external sets after per-class error analysis: their idea of an ID number disagreed with the hosts'.

The generic generate, verify, deduplicate loop is in [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation); the Kaggle lesson is that the verifier must not share the generator's blind spots.

## 7. Scale, and the order you train in

Scale mattered where the task was procedural: 531,318 pretraining examples for the ARChitects (ARC 2024), 3,255,481 augmented samples for NVARC, 360,000 puzzles for NullSira. PII's generated sets, by contrast, ran from 1,722 to about 4,600 essays. What mattered almost everywhere was *order*: broad or synthetic data first, data closest to the test last.

![Two-stage training lanes: broad or synthetic data first, target-like data last](/imgs/blogs/kaggle-llm-synthetic-data-as-the-edge-5.webp)

Guanshuo Xu's DAIGT run is the cleanest measurement: 500k SlimPajama pairs gave a general detector at 0.967 private, and adapting on student-mimicking essays took it to 0.98. Numina (AIMO 1) taught prose reasoning, then roughly 60k tool-integrated traces. The ARChitects pretrained on generated tasks, then trained a LoRA on the hidden tasks' own demonstrations at test time. Hrithik Reddy (Deep Past 8th) went 350k noisy to 65k clean; darraghdog and conjuring92 (Deep Past 3rd) pre-trained ByT5 on about 150k Claude-written vocabulary and grammar drills before scholar-quality pairs; Rai (PII 3rd) pre-trained on mpware's Mixtral essays before the competition's. In Deep Past, 9 of 17 writeups had a noisy stage before clean fine-tuning.

The counterexample is worth keeping: Karl (Deep Past 20th) found gold fine-tuning after silver pre-training *lowered* his score. Check the order on your own data.

## 8. What it bought, measured

![Measured gains from data: Numina, Eedi 1st place and Deep Past ablations](/imgs/blogs/kaggle-llm-synthetic-data-as-the-edge-6.webp)

Numina held the model fixed and changed only the training data: chain of thought alone scored 8 of 50 public, single-program MMOS traces 16, multi-step ToRA traces 27. In Eedi's winning ablation, teacher pseudo-labels on the synthetic questions were worth +0.044, more than the 72B reranker (+0.013) and the 32B (+0.010) combined. In Deep Past, the largest single step on two ablation ladders was data: +4.0 from Gemini pseudo-labels for Karl (20th, 31.1 to 38.9) and +3.5 from continued pre-training for ebi & okumura (5th, 35.4 to 40.8).

Note where the Eedi gain came from: *labelling* the synthetic rows with a strong teacher's soft scores, not trusting the generator's labels. S T I T Y (MAP 7th) did the same with a Qwen2.5-32B trained on real data.

## 9. When synthetic data did not help

A cross-competition view also shows where the same move failed.

| Finding | Where | Who reported it |
|---|---|---|
| Synthetic rows "did not work" or "added little" in the reranker | Eedi | 6th, 3rd (13th tried several variants); 1st and 4th gained |
| LLM-written training data dropped | MAP | 12th, 13th, 18th; Chris Deotte used none ("mostly not" an improvement) |
| Helped single models, "no benefit after fusion" | MAP | 2nd |
| More BARC synthetic tasks: no gain | ARC 2024 | Barbadillo (2nd) |
| Own DSL with synthetic generators: no real ARC task solved | ARC 2025 | Barbadillo (5th) |
| Faker PII injection did not help | PII | 8th; core of 4th's and 6th's recipes |
| Scaling back-translation 10k to 100k: -0.1 public, +0.3 private | Deep Past | 5th |
| Synthetic direction dropped because public looked worse, though private was better | Nemotron | 18th (0.868 vs 0.872 public; 0.872 vs 0.864 private) |
| No training at all, top five tied | AIMO 3 | 0 of 5 writeups trained |

Three patterns explain most failures. **Label noise caps the gain**: in MAP, synthetic rows helped single models and washed out in ensembles. **Validation must not come from your own generator**: DAIGT teams reported CV "always close to 1" for that reason, while iws (MAP 13th) flipped it and used synthetic data as a held-out yardstick when public could not separate submissions. **A strong enough base model moves the lever**: with `gpt-oss-120b` in AIMO 3, the edge was test-time compute, not data.

## 10. If your next competition has a hidden-test gap

1. Write down what the hidden test probably looks like (prompts, style, noise, labels, length); trace every synthetic choice to that list.
2. Pick the generator by style match: base models for non-chat text, a fine-tuned generator for a narrow domain, a frontier model where reasoning is the label, a program where the answer can be computed.
3. Buy diversity before volume, and generate for the boundaries your model confuses.
4. Find the shortcut the test will break and put it on both classes.
5. Filter with something independent of the generator: program agreement, a different judge model, your own retriever, deduplication thresholds.
6. Train broad first, target-like last, and measure the order.
7. Keep real data, or a held-out synthetic set from a different pipeline, as validation. Never validate on your own generator's output.
8. Prefer soft labels from a teacher trained on real data over the generator's own labels.

Sibling technique posts cover the other levers: *RAG under a notebook budget*, *fitting big models in small GPUs*, *the test-time compute playbook* and *metric exploits and shake-ups*.

## Sources

- DAIGT: https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295 (1st) ; https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395 (2nd) ; https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179 (4th) ; https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470643 (7th) ; https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470255 (9th) ; https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470396 (12th) ; https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470593 (13th) ; https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465882 (James Day thread)
- PII: https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374 (1st) ; https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497482 (3rd) ; https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497367 (4th) ; https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/498304 (6th) ; https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497323 (8th) ; https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497177 (9th)
- Eedi: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution ; https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/cqyr-2nd-place-solution ; https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/waseda-pochi-3rd-place-solution-with-magic-boost ; https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/takoi-charmq-kami-4th-place-solution ; https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/nikita-babich-6th-place-solution ; https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/study-hard-private-9th-public-7th-place-solution
- MAP: https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map2025-private-and-public-2nd ; https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/private-7th-public-10th-place-solution ; https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings/writeups/map-competition-13th-place-gold-medal-solution
- AIMO 1: https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/numina-numina-1st-place-solution ; https://www.kaggle.com/competitions/ai-mathematical-olympiad-prize/writeups/cmu-math-2nd-place-solution-all-code-and-datasets-
- AIMO 3: see the [AIMO 3 synthesis](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions) for all five writeups
- ARC 2024: https://github.com/da-fr/arc-prize-2024/blob/main/the_architects.pdf ; https://www.kaggle.com/competitions/arc-prize-2024/writeups/guillermo-barbadillo-2nd-place-solution-for-the-ar
- ARC 2025: https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc ; https://www.kaggle.com/competitions/arc-prize-2025/writeups/exploring-the-combination-of-search-and-learn-for
- Deep Past: https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684353 (1st) ; https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684425 (3rd) ; https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684847 (5th) ; https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684329 (8th) ; https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/685928 (9th) ; https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/discussion/684267 (20th)
- Nemotron: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/domdolus-tolus-solution ; https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/18th-solution-from-deterministic-solvers-to-learn
