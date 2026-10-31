---
title: "The Kaggle LLM competition playbook: which lever to pull first, from three years of winners"
date: "2026-10-31"
publishDate: "2026-10-31"
description: "The capstone of the Learn from Kaggle LLM series: every published competition placed by the lever that decided it, the people who kept winning and what they carried between competitions, a six-question playbook for a new competition, and the honesty habits the series had to learn."
tags: ["kaggle", "llm", "learn-from-kaggle", "competition", "playbook", "test-time-compute", "synthetic-data", "retrieval", "distillation", "evaluation"]
category: "machine-learning"
subcategory: "Learn from Kaggle"
author: "Hiep Tran"
featured: true
readTime: 13
---

> **TL;DR.** Across 23 scored Kaggle LLM competitions from 2023 to 2026, the decisive lever was **data** 7 times, **test-time compute** 6 times, **metric insight** 5 times, **model fit** 3 times and **retrieval** twice. The model itself was never the lever on its own. A small set of people (Raja Biswas, tascj, sayoulala, the H2O and Team Danube trio, Eduardo Rocha de Andrade, the ARChitects, NVIDIA's NemoSkills and NVARC teams) placed again and again because they carried a specific artifact from one competition into the next. **The playbook:** read the scorer, find out where the test came from, check whether each test item ships its own examples, check whether answers are checkable, and only then decide between building data and fitting a bigger model. **The honesty rule:** most of a top-20 order is noise, most writeup numbers are self-reported, and every number in a bank needs recomputing.

![Decisive lever per competition: data 7, test-time compute 6, metric insight 5, model fit 3, retrieval 2](/imgs/blogs/kaggle-llm-competitions-playbook-1.webp)

The figure above is the series in one picture: the five-lever strip from the [orientation post](/blog/machine-learning/learn-from-kaggle/kaggle-llm-competitions-how-they-work), with one tile per scored competition stacked on the lever that decided it. No column holds a majority. That is the first lesson of three years of Kaggle LLM competitions: there is no default winning recipe, only a default order of questions to ask before you pick one.

This post pulls the 36 published posts of the series together. It places every competition on one timeline, names the people who kept finishing on top and what they reused, turns the pattern into a decision playbook with a rule per lever and the competitions behind each rule, and closes with the reading habits the series had to adopt to stay honest. It is a synthesis of the series' own verdicts, not a re-analysis: where a number appears, it comes from the linked post, and the linked post traces it to the writeups.

## 1. Every competition on one timeline

**Rule of thumb: the deciding lever moved with the hardware and the benchmark, but metric insight never left.**

![Every published competition by final-submission month and decisive lever; hollow rings have not closed](/imgs/blogs/kaggle-llm-competitions-playbook-2.webp)

Read the lanes left to right. In 2023 and early 2024 the money went to retrieval and data: [LLM Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions) was won by a 30 to 60 million chunk Wikipedia index, and [Detect AI Generated Text](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions) by a 160k-row training mix built for diversity. From mid-2024, on the same two T4s, the edge moved to model fit: the [LMSYS Chatbot Arena](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions) winner distilled 70B teachers into Gemma-2-9B. Then test-time compute took over, starting with [AIMO Progress Prize 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions) and [ARC Prize 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions). In 2025 and 2026 data came back in a new form: off-Kaggle pretraining on synthetic puzzles in [ARC Prize 2025](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2025-solutions), and solver-written traces in the [Nemotron reasoning challenge](/blog/machine-learning/learn-from-kaggle/kaggle-nvidia-nemotron-reasoning-challenge-solutions).

| Lever | Competitions it decided | What the winning move looked like |
|---|---|---|
| Data (7) | [DAIGT](/blog/machine-learning/learn-from-kaggle/kaggle-detect-ai-generated-text-solutions), [PII](/blog/machine-learning/learn-from-kaggle/kaggle-pii-data-detection-solutions), [Eedi](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions), [AIMO 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions), [ARC 2025](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2025-solutions), [Deep Past](/blog/machine-learning/learn-from-kaggle/kaggle-deep-past-akkadian-translation-solutions), [Nemotron](/blog/machine-learning/learn-from-kaggle/kaggle-nvidia-nemotron-reasoning-challenge-solutions) | A generator that wrote like the hidden test, filtered by something independent of the generator |
| Test-time compute (6) | [AIMO 1](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-1-solutions), [ARC 2024](/blog/machine-learning/learn-from-kaggle/kaggle-arc-prize-2024-solutions), [Santa 2024](/blog/machine-learning/learn-from-kaggle/kaggle-santa-2024-perplexity-puzzle-solutions), [Konwinski](/blog/machine-learning/learn-from-kaggle/kaggle-konwinski-prize-solutions), [Jigsaw](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions), [AIMO 3](/blog/machine-learning/learn-from-kaggle/kaggle-aimo-progress-prize-3-solutions) | Sample and vote, verify before shipping, search, or fine-tune on the test file's own examples |
| Metric insight (5) | [Prompt Recovery](/blog/machine-learning/learn-from-kaggle/kaggle-llm-prompt-recovery-solutions), [You Can't Please Them All](/blog/machine-learning/learn-from-kaggle/kaggle-llms-you-cant-please-them-all-solutions), [Drawing with LLMs](/blog/machine-learning/learn-from-kaggle/kaggle-drawing-with-llms-solutions), [Code Golf](/blog/machine-learning/learn-from-kaggle/kaggle-google-code-golf-2025-solutions), [NeuroGolf](/blog/machine-learning/learn-from-kaggle/kaggle-neurogolf-2026-solutions) | Read the scorer's implementation, then optimise against it directly |
| Model fit (3) | [LMSYS](/blog/machine-learning/learn-from-kaggle/kaggle-lmsys-chatbot-arena-solutions), [WSDM Cup](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions), [MAP](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) | Distil 70B-class teachers into 9B to 32B, then quantise or stream layers to fit two T4s |
| Retrieval (2) | [Science Exam](/blog/machine-learning/learn-from-kaggle/kaggle-llm-science-exam-solutions), [Make Data Count](/blog/machine-learning/learn-from-kaggle/kaggle-make-data-count-solutions) | Retrieve from the corpus the questions or labels were built from |

The grey lane holds the 16 judged hackathons. They sit outside the five levers because a panel, not a scorer, picks the winner, and the series found the same answer in all of them: the rubric is the metric. The [BigQuery AI](/blog/machine-learning/learn-from-kaggle/kaggle-bigquery-ai-hackathon-winners) rubric gave 30 of its 110 points to things that need no model at all; the [Gemma impact hackathons](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-impact-hackathons-winners) gave 70 of 100 to impact and video; 0 of 40 [vibe-coding sprint winners](/blog/machine-learning/learn-from-kaggle/kaggle-gemini-vibe-coding-and-capstones) reported a quantitative result. The exceptions were the ones with a real measurement inside: [OpenAI to Z](/blog/machine-learning/learn-from-kaggle/kaggle-openai-to-z-challenge-winners) was won on labelled data, the [gpt-oss red-teaming](/blog/machine-learning/learn-from-kaggle/kaggle-gpt-oss-red-teaming-challenge-findings) prize on methodology (18 of 20 awarded entries ran an automated harness), [Tunix](/blog/machine-learning/learn-from-kaggle/kaggle-tunix-hack-reasoning-winners) on reward design, [Measuring AGI](/blog/machine-learning/learn-from-kaggle/kaggle-measuring-agi-cognitive-abilities-winners) on protocol design, and the [2024 Gemma and Gemini hackathons](/blog/machine-learning/learn-from-kaggle/kaggle-gemma-gemini-early-hackathons-winners) on one visible, checkable extra.

The second-order point is about the "most LLM" competitions. In [Santa 2024](/blog/machine-learning/learn-from-kaggle/kaggle-santa-2024-perplexity-puzzle-solutions) the LLM was the objective and classical search won; in [Code Golf](/blog/machine-learning/learn-from-kaggle/kaggle-google-code-golf-2025-solutions) places 1 to 3 went to human golfers and the best LLM-driven team finished 4th, 3,974 points behind 3rd. Being an LLM competition says nothing about whether the LLM is where the points are.

## 2. The people who kept winning, and what they carried

**Rule of thumb: repeat winners do not reuse models; they reuse an artifact and a habit.**

![Recurring top-20 finishes, by person or team](/imgs/blogs/kaggle-llm-competitions-playbook-3.webp)

The grid shows eleven people or teams who finished in the top 20 of at least two competitions. Read across a row and the lever changes almost every time; what stays constant is the craft each one carried.

**Raja Biswas** (Kaggle handle conjuring92) is the clearest case of a data specialist. He won DAIGT with Udbhav Bamba and Nicholas Broad, won [Eedi](/blog/machine-learning/learn-from-kaggle/kaggle-eedi-mining-misconceptions-solutions) solo on 10.6k synthetic questions that Claude 3.5 Sonnet wrote for groups of easily confused misconceptions, placed 4th in [Make Data Count](/blog/machine-learning/learn-from-kaggle/kaggle-make-data-count-solutions) with an NVIDIA team whose agent wrote warm-up labels (worth about +0.02), and 3rd in [Deep Past](/blog/machine-learning/learn-from-kaggle/kaggle-deep-past-akkadian-translation-solutions) with darraghdog after pretraining ByT5 on about 150k Claude-written drills. Four competitions, three different levers, one habit: when labels are scarce, write more of them.

**tascj and sayoulala** are the model-fit pair. sayoulala won LMSYS by distilling Llama-3-70B and Qwen2-72B into Gemma-2-9B; tascj took 2nd with full fine-tuning and a rewritten inference engine. In the [WSDM Cup](/blog/machine-learning/learn-from-kaggle/kaggle-wsdm-cup-multilingual-arena-solutions) a few months later, the winner stacked sayoulala's distillation recipe on tascj's padding-free engine, and 2nd place started from tascj's LMSYS checkpoints. tascj then won [MAP](/blog/machine-learning/learn-from-kaggle/kaggle-map-student-math-misunderstandings-solutions) with one submission in three months, running 32B models on two T4s through INT8 kernels and layer-by-layer weight streaming. The artifact was the inference engine; the habit was systems work to afford a bigger model.

**Psi (Philipp Singer), Yauhen Babakhin and Pascal Pfeiffer** won Science Exam as Team H2O LLM Studio, took 2nd in [Prompt Recovery](/blog/machine-learning/learn-from-kaggle/kaggle-llm-prompt-recovery-solutions) as Team Danube by regressing directly into the scorer's embedding space, and Psi and Pfeiffer placed 5th in LMSYS by fine-tuning from reward models. Their tool, H2O LLM Studio, also turns up in Guanshuo Xu's 2nd-place DAIGT pipeline. **Guanshuo Xu** went on to win [Jigsaw](/blog/machine-learning/learn-from-kaggle/kaggle-jigsaw-agile-community-rules-solutions) solo, on craft: loss on the Yes/No token only, per-rule rank normalisation, six models.

**Eduardo Rocha de Andrade** placed 3rd in Prompt Recovery (beam search against the real scorer), won [Konwinski](/blog/machine-learning/learn-from-kaggle/kaggle-konwinski-prize-solutions) solo by making the model write a reproduction test before touching code, and won [NeuroGolf](/blog/machine-learning/learn-from-kaggle/kaggle-neurogolf-2026-solutions) as part of Kaggle Agent with Giba, Jiwei Liu, Max Jeblick and Chris Deotte. The habit: build something the model cannot fool, then let it work.

**The ARChitects, MindsAI and NVIDIA** own the ARC and AIMO lines. The ARChitects won ARC 2024 at 53.5% and placed 2nd in 2025 at 16.53%; MindsAI scored 55.5 in 2024 without being ranked (their method was not open-sourced) and 3rd in 2025. Ivan Sorokin was on both NVIDIA winners: NemoSkills in [AIMO 2](/blog/machine-learning/learn-from-kaggle/aimo-progress-prize-2-published-solutions) (34 of 50, by building a 5.5M-solution dataset) and NVARC in ARC 2025 (24.03%). Chris Deotte's two top-20 finishes in the series are LMSYS 16th and NeuroGolf 1st, but his threads set baselines everywhere: the 60k Science Exam dataset, the check that MAP's test used the same 15 questions, the right way to batch Gemma in Santa.

![What winners carried forward between competitions](/imgs/blogs/kaggle-llm-competitions-playbook-4.webp)

The lineage figure is the practical version. Each lane is a chain where one competition's open winner became the next competition's starting line. The second-order consequence is the uncomfortable one: when a winner publishes, the published artifact becomes the floor. ARC 2025 ranks 4 to 20 were all lightly tuned forks of the ARChitects' 2024 notebook, packed between 5.00 and 6.67; the only three teams above 7% pretrained their own model. Reusing the last winner gets you into the pack. It has not once, in this series, got anyone out of it.

## 3. The decision playbook

**Rule of thumb: ask about the scorer and the test before you ask about the model.**

![The decision playbook: first lever for a new LLM competition](/imgs/blogs/kaggle-llm-competitions-playbook-5.webp)

The tree orders the questions by how cheap they are to answer and how much they decide. Reading the metric's code costs an afternoon and decided five competitions. Finding where the test came from costs a day of forum reading and decided two more. Everything below that needs experiments. Each rule below names the competitions that justify it.

| If the new competition has... | Pull first | Justified by |
|---|---|---|
| A panel of judges and a weighted rubric | Score yourself against the rubric; ship a working build and one measured number | BigQuery AI, Gemma impact, vibe-coding sprints |
| A learned scorer (embedding cosine, LLM judges, VQA) or an odd formula (bytes, log parameters) | Metric insight: read the implementation, probe it with submissions | Prompt Recovery (`lucrarea`), You Can't Please Them All (30.010 ceiling), Drawing with LLMs (hidden text), Code Golf, NeuroGolf |
| A test written from a corpus you can obtain | Retrieval from that corpus | Science Exam (Wikipedia), Make Data Count (14 of 15 writeups looked DOIs up in the label source; the one that did not finished 20th) |
| Labelled examples inside each test item | Test-time training on those examples | ARC 2024, Jigsaw |
| Checkable answers and inference you control | Sample, verify, vote; spend saved seconds on hard items | AIMO 1 (vote over 48), AIMO 3, Konwinski (verify, then ship or skip) |
| Scarce labels, unseen labels or unseen generators | Data: generate rows that look like the hidden test | DAIGT, PII (945 essays with any PII), Eedi, Deep Past, Nemotron (host fixed greedy decoding) |
| Plenty of labels and a fixed small box | Model fit: distil, quantise, stream layers | LMSYS, WSDM Cup, MAP |

Budget is a modifier, not a branch. On two 16 GB T4s, model fit is a tax on every lever: `GB = params x bits / 8` says a 32B model needs 16 GB at 4 bits before activations, which is why [fitting big models in small GPUs](/blog/machine-learning/learn-from-kaggle/kaggle-llm-fitting-big-models-in-small-gpus) mattered even in competitions decided by data. On one H100 (AIMO 3) or four L4s (ARC 2025), test-time compute becomes affordable and the clock replaces memory as the constraint. Data size sets the retrieval stack: millions of candidates behind each query wants a dense index, about 2,587 labels wants an LLM reranker, 8 to 12 candidates wants no retrieval at all, as [RAG under a notebook budget](/blog/machine-learning/learn-from-kaggle/kaggle-llm-rag-under-a-notebook-budget) shows.

Here is the tree as code. It is an illustrative toy, not any team's code, run on the series' own history to check that it reproduces the verdicts:

```python
# Illustrative toy, not any team's code: the decision playbook as a function.
from dataclasses import dataclass

@dataclass
class Comp:
    name: str
    judged: bool = False            # panel of judges instead of a scorer
    readable_scorer: bool = False   # learned scorer or odd formula you can read
    corpus_source: bool = False     # test written from a corpus you can obtain
    ships_examples: bool = False    # each test item carries labelled examples
    checkable: bool = False         # integer, exact grid, or tests that run
    own_inference: bool = True      # False when the host fixes decoding
    scarce_labels: bool = False     # labels scarce, unseen, or generator unseen

def first_lever(c: Comp) -> str:
    if c.judged:          return "the rubric"
    if c.readable_scorer: return "metric insight"
    if c.corpus_source:   return "retrieval"
    if c.ships_examples:  return "test-time compute: train on the test"
    if c.checkable and c.own_inference:
        return "test-time compute: sample, verify, vote"
    if c.scarce_labels:   return "data"
    return "model fit"

history = [
    Comp("Science Exam", corpus_source=True),
    Comp("DAIGT", scarce_labels=True),
    Comp("Prompt Recovery", readable_scorer=True),
    Comp("LMSYS"),
    Comp("ARC 2024", ships_examples=True, checkable=True),
    Comp("Eedi", scarce_labels=True),
    Comp("Konwinski", checkable=True),
    Comp("MAP"),
    Comp("Jigsaw", ships_examples=True),
    Comp("ARC 2025", ships_examples=True, checkable=True),
    Comp("Nemotron", checkable=True, own_inference=False, scarce_labels=True),
]
for c in history:
    print(f"{c.name:16s} -> {first_lever(c)}")
```

```console
Science Exam     -> retrieval
DAIGT            -> data
Prompt Recovery  -> metric insight
LMSYS            -> model fit
ARC 2024         -> test-time compute: train on the test
Eedi             -> data
Konwinski        -> test-time compute: sample, verify, vote
MAP              -> model fit
Jigsaw           -> test-time compute: train on the test
ARC 2025         -> test-time compute: train on the test
Nemotron         -> data
```

Ten of eleven match the decisive lever. The miss is instructive: ARC 2025 says test-time training, and every top-20 team did train on the test, but that was table stakes; the three teams above 7% won on off-Kaggle pretraining data. The playbook picks the **first** lever, the one you cannot skip. In a sequel, expect the first lever to be everyone's floor and the second lever to decide.

> Read the scorer, then read the test, then read the box. The model is the last thing you choose.

Three short cases show the playbook working on the series' own record.

**MAP, one submission.** tascj was 14th on public and won on private with one submission in three months, while the top 20 sat inside 0.00171 and seed noise was about 0.002. The playbook says model fit; the second-order lesson is that a trusted local validation is worth more than the public board.

**Konwinski, the metric that charges for mistakes.** A wrong patch cost as much as a right one earned. The winner resolved 9, broke 2 and skipped 109, a 7.5% resolve rate at 82% precision. "Checkable answers" led to verification, and verification led to skipping.

**Make Data Count, the smallest LLM.** The winner's LLM, a 0-shot Qwen2.5-Coder-32B, only typed accession IDs. "Where did the labels come from?" answered the competition before any model was chosen.

## 4. Three honesty habits the series had to learn

**Rule of thumb: treat a leaderboard as a measurement with error bars, and a writeup as a claim with a source.**

![Ties and noise bands in the top 20](/imgs/blogs/kaggle-llm-competitions-playbook-6.webp)

**Ties are noise, so say so.** In AIMO 3 ranks 1 to 4 scored 44.0 and ranks 5 to 20 scored 43.5 out of 50, half a problem apart on a notebook scored by running it twice. In Santa 2024, 13 teams hit exactly 246.81784 and the board ordered them by submission time. In You Can't Please Them All, three teams tied at 30.010 and the metric's own sampling ordered them. The [metric exploits and shake-ups](/blog/machine-learning/learn-from-kaggle/kaggle-llm-metric-exploits-and-shake-ups) post simulates how far an identical notebook moves between reruns. When reading any top-20 table, ask what the measured seed noise was before reading anything into the order; when no one measured it, say that too.

**Writeups are claims.** Every hackathon number in this series is the team's own claim; judges never re-ran them, and the OpenAI to Z sites are not verified on the ground. Code-competition writeups need the same care: a secondary source's 8x A100 claim for LMSYS stayed labelled unverified, and teams with no writeup were never given an invented method.

**Banks make arithmetic errors.** The source banks were wrong often enough that recomputation became a gate:

| Item | Bank or source said | Recomputed |
|---|---|---|
| Konwinski 2nd place score 0.016576 | About 2 correct, 0 wrong | 7 right, 5 wrong, 108 skipped: $(7 - 5 - 108/10{,}000)/120 = 0.0165767$ |
| Nemotron contribution awards | 8 Sparks | The allocation sums to 12 |
| ARC 2025 MindsAI | 15.42 private (writeup) | 12.64 private, 15.42 was public (API) |
| Deep Past places | Two self-titled places | Each off by one from the API rank |
| Gemma language-tuning winners | Swahili scores in the bank | Wrong; the top-20 file corrected them |
| "N of M" technique counts | Top-20 summary lines | Off by one in three Wave 1 posts; recounted from the matrix cells |

The Konwinski row is worth checking yourself: 2 right and 0 wrong gives $(2 - 0 - 118/10{,}000)/120 = 0.016568$, which does not display as 0.016576. A number that looks plausible is not the same as a number that decodes. For readers, the habit transfers directly: before you build on a forum claim, recompute it from its own inputs.

## 5. What is still to come

> **Still to come in this series** (titles only; each post will be drafted after its competition closes and its writeups are fetched):
> - **ARC Prize 2026** (ARC-AGI-2, ARC-AGI-3 and a paper track): final submission 2026-11-02, papers 2026-11-09, winners announced 2026-12-04.
> - **Gemma 4 Developer Agent**: final submission 2026-12-02, with an optional paper track due 2026-11-12.
> - **Name That LLM**: final submission 2027-01-13.

They are the three hollow rings in the timeline. The playbook makes a prediction for each that you can check later: ARC 2026 is a sequel, so expect last year's test-time training and pretraining to be the floor; Name That LLM asks which of 18 frontier LLMs wrote a passage, a fingerprinting sequel to DAIGT, which on this series' record points at data first.

## When to reach for this playbook, and when not to

Reach for it when a new competition, or a production problem shaped like one, has a fixed compute box, a hidden evaluation set and a metric you did not design. Start with the tree, then read the technique post for the lever it picks: [synthetic data](/blog/machine-learning/learn-from-kaggle/kaggle-llm-synthetic-data-as-the-edge), [test-time compute](/blog/machine-learning/learn-from-kaggle/kaggle-llm-test-time-compute-playbook), [LLMs as classifiers](/blog/machine-learning/learn-from-kaggle/kaggle-llm-as-classifier-fine-tuning-recipes), [model fit](/blog/machine-learning/learn-from-kaggle/kaggle-llm-fitting-big-models-in-small-gpus), [retrieval](/blog/machine-learning/learn-from-kaggle/kaggle-llm-rag-under-a-notebook-budget) or [metric exploits](/blog/machine-learning/learn-from-kaggle/kaggle-llm-metric-exploits-and-shake-ups). Then read the competition post whose budget card looks most like your hardware.

Do not reach for it to pick a frontier model, to rank people, or to treat a 23-competition tally as a law. Each lever was assigned by reading writeups, not by ablation, and several competitions were decided by two levers at once. Some winning moves (a metric token, a leak, hidden text for a judge) are lessons about evaluation design, not recipes to copy into production.
