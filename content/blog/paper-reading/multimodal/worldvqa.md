---
title: "WorldVQA: Measuring Atomic World Knowledge in Multimodal LLMs"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "paper-reading"
subcategory: "Multimodal"
tags:
  - multimodal
  - vqa
  - benchmark
  - visual-knowledge
  - hallucination
  - calibration
  - evaluation
  - long-tail
description: "A close reading of WorldVQA, a 3,500-pair benchmark that decouples visual recognition from reasoning and shows no frontier multimodal model clears 50% accuracy on naming what it actually sees."
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/worldvqa-1.png"
readTime: 30
---

When a multimodal model answers "what is this building?" with the right landmark name, we tend to credit the whole stack: the vision encoder saw the facade, the language model retrieved the fact, the reasoning glue connected the two. But that credit is almost never earned cleanly. Most visual question-answering benchmarks bundle three abilities into one score — recognizing the percept, recalling the encyclopedic fact, and reasoning over both — so when a model is wrong, we cannot say *which* of the three failed. The benchmark told us the model scored 39%; it did not tell us whether the model failed to *see* the entity, failed to *know* the entity, or simply tripped on a multi-hop chain that had nothing to do with the picture.

WorldVQA, from the Moonshot AI / Kimi team, is a deliberate attack on that ambiguity. It is a 3,500-pair benchmark engineered so that every question requires exactly one thing: look at an image, name the specific entity it depicts. No OCR. No arithmetic. No multi-hop deduction. Each image is "definitive evidence for exactly one atomic fact," and the grading rewards the model only when it produces the *specific* identity — the species, not "a flower"; the exact aircraft variant, not "a jet." The result is the cleanest measurement we have of what a multimodal model actually *memorizes* about the visual world, stripped of the reasoning scaffolding that usually inflates and obscures the number.

The headline is uncomfortable: across sixteen evaluated frontier models — Gemini-3-pro, Kimi K2.5, the GPT-5.x family, Claude Opus 4.5, Qwen3-VL, and more — **no model exceeds 50% accuracy**. The single best, Gemini-3-pro, lands at 47.4%. And nearly every model is severely overconfident: they assert wrong answers with near-certainty rather than abstaining. WorldVQA is short on architecture (it trains no model) but long on a lesson that matters for anyone shipping vision systems: the part of the multimodal stack we assume is solved — naming what you see — is the part that is most broken, and our evals have been hiding it.

![WorldVQA construction and difficulty pipeline](/imgs/blogs/worldvqa-1.png)

The diagram above is the mental model: WorldVQA is not a model, it is a construction-and-evaluation pipeline. Seed entities from expert annotators fan out into a global-expansion branch (GPT and Kimi association search, with a hard cap on region-specific entities) and a visual-deduplication branch (embedding-based leakage control against web corpora). Those branches merge at a dual-gate verification stage — one machine auditor, one blind human — and only fully validated triplets survive into the 3,500-pair pool. Then, crucially, a five-model ensemble scores every surviving sample to assign difficulty *from the data itself*, not from annotator guesswork. Hold that flow in your head; the rest of the article unpacks each stage and why each design choice changes what the final number means.

> [!tldr] TL;DR
> - **What it claims:** A 3,500-pair VQA benchmark that strictly decouples visual *knowledge retrieval* from *reasoning*, measuring an MLLM's atomic ability to ground and name a single visual entity in one hop — plus a refusal-aware metric suite (CGA, F-score, calibration) to separate "doesn't know" from "won't say."
> - **Why it matters:** It exposes a measurement gap. Comprehensive and expert suites treat world knowledge as an implicit prerequisite or bury it under reasoning chains, so factual deficits are invisible. WorldVQA makes the factual deficit the *only* thing being measured.
> - **Most surprising finding:** No frontier model clears 50% accuracy (top is Gemini-3-pro at 47.4%), and every model is badly overconfident — Gemini-3-pro asserts ≥95% confidence in over 85% of cases regardless of whether it is right. Open-source Kimi K2.5 (46.3% accuracy, 46.8% F-score) nearly matches the best closed model.
> - **Where it fails:** It measures factuality in a "highly atomic" setting; whether atomic entity-naming correlates with performance on complex downstream multimodal tasks is an open question the authors explicitly do not answer. The People category is also excluded from the overall average because closed-model safety guardrails refuse rather than answer.

## Context: what came before

To understand why WorldVQA exists, you have to see what the existing VQA landscape was *not* measuring. The paper sorts prior work into three buckets, and the gap it fills sits precisely between them.

The first bucket is **comprehensive multimodal suites** — MME, MMBench, SEED-Bench, MMStar. These are broad capability surveys: they probe perception, OCR, spatial reasoning, attribute recognition, and dozens of other skills across thousands of questions. World knowledge is *in there*, but only as an implicit prerequisite. A question like "which country does this dish come from?" needs the model to know the dish, but the benchmark never isolates that knowledge step. If the model whiffs, the aggregate score moves a fraction of a percent and nobody learns that the model didn't recognize the dish — the signal is diluted across the whole capability matrix. Worse, these suites suffer a *ceiling effect*: frontier models now saturate large chunks of them, so the remaining headroom is dominated by edge cases and ambiguity rather than genuine knowledge.

The second bucket is **expert reasoning suites** — MMMU and MMMU-Pro. These deliberately stack difficulty by demanding complex, college-level reasoning chains over diagrams, charts, and domain figures. They are excellent at measuring reasoning. But the paper's critique is sharp: those reasoning chains "obscure purely factual deficits." If a model gets an MMMU physics question wrong, was it because it misread the diagram, misremembered a constant, or botched the algebra? You cannot tell, and the factual-recognition component is the smallest, most-confounded term in that error.

The third bucket is **factuality probes** — SimpleVQA, VisualSimpleQA. These are the closest kin to WorldVQA; they explicitly target factual recall. But the paper argues they still carry *secondary dependencies*: a question may require language knowledge layered on top of the visual fact, or OCR to read a label, or a short multi-hop step. So even these "simple" probes are not atomic. A correct answer can come from the wrong path — reading the caption rather than recognizing the object.

The gap, then, is a benchmark that asks one and only one thing: *does the model recognize the specific entity it sees, or is it hallucinating from visual patterns?* That is the core question WorldVQA isolates. It is a narrow question on purpose. By refusing to test reasoning, OCR, or multi-hop logic, WorldVQA buys the ability to attribute every failure to a single cause — the model's internal, unassisted association between a visual percept and a factual identity. The cost of that narrowness is real (more on that in the Critique), but the diagnostic clarity it buys is exactly what the field was missing.

## Contributions

The paper's contributions are best read as a set of *design decisions*, each of which removes a specific confound that would otherwise leak reasoning, leakage, or refusal behavior into the score.

1. **Atomic isolation as a hard constraint.** Every question is single-hop and requires only direct identification of a visual entity or its name. OCR, arithmetic, and multi-hop reasoning are explicitly excluded. Each image is definitive evidence for exactly one atomic fact. This is the foundational move: it makes failures attributable to memory rather than reasoning.

2. **A stratified 9-category taxonomy** spanning globally recognized "head-class" landmarks and logos down to "long-tail" biological species and artisanal artifacts — so the benchmark probes both high-frequency and rare entities rather than only the easy famous ones.

3. **Granularity alignment.** The grading enforces strict alignment between the specificity of the question and the entity, penalizing generic hypernyms. Answering "flower" when the species is asked loses points. This single rule is why models that "revert to generic hypernyms" collapse on Nature and Culture.

4. **Model-performance-based difficulty stratification.** Difficulty is assigned by an *ensemble of five frontier MLLMs*, not by annotators. Samples all five get right are discarded (they carry no signal); the rest are stratified into Easy (>3 models correct), Medium (1–2), and Hard (0), and the Easy bucket is randomly downsampled so the benchmark is not dominated by trivial entities.

5. **Distributional balancing and LLM-in-the-loop global expansion.** A 50% per-category cap on China-unique entities (yielding 36% aggregate Chinese-specific entities) keeps the benchmark from over-indexing on one region, and an LLM-driven association search using GPT and Kimi supplements under-represented categories with global entities.

6. **Visual deduplication for leakage control.** Instance-level Semantic Content (ISC) embeddings are compared by cosine similarity against LAION and Common Crawl at a strict 0.95 threshold; leaked images are discarded, and affected entities get fresh assets re-collected from video screenshots — so a correct answer reflects internal knowledge, not memorized image-answer pairs scraped during pretraining.

7. **Dual-gate verification.** A few-shot-prompted Gemini-3-Pro audits each triplet for visual clarity, semantic exclusivity, and contextual completeness; then an independent human annotator answers blind, and any divergence triggers manual audit and permanent removal of flawed samples.

8. **A refusal-aware metric suite.** Beyond raw Accuracy, the paper reports Correct Given Attempted (CGA), an F-score that balances coverage against precision, and calibration metrics (ECE, Slope) — so over-conservative refusal and over-aggressive guessing are both penalized, and "doesn't know" is distinguished from "won't say."

## Method

WorldVQA's "method" is its construction-and-evaluation pipeline. There is no training run, no model weights, no MoE config — and the dossier is explicit that architecture fields are "not applicable." So we read this paper the way we'd read a measurement protocol: every stage exists to remove a specific source of error from the final number. Let me walk the stages in the order data flows through them, defining each design term on first use.

### Atomic isolation: the load-bearing principle

The single most important design choice is **atomic isolation**. An *atomic* question is one whose answer depends on exactly one fact, retrievable in a single recognition hop, with no intermediate computation. Formally, if we let $v$ be the visual percept and $y$ the answer, an atomic question requires only the model's learned association $P(y \mid v)$ — not a chain $P(y \mid r_k(\dots r_1(v)))$ through reasoning operators $r_1 \dots r_k$, and not a fusion $P(y \mid v, t)$ with on-image text $t$ read via OCR.

![Composite VQA versus WorldVQA atomic isolation](/imgs/blogs/worldvqa-2.png)

The before/after above is the whole argument in one frame. On the left, a composite VQA item entangles recognition with OCR and a 2–4 hop reasoning chain, so a wrong answer is unattributable — was it memory or logic? On the right, a WorldVQA item is one image equals one definitive fact: a single granularity-aligned recognition hop, and a failure attributes cleanly to knowledge. This is not a cosmetic distinction. It changes the *meaning* of the score. When a composite benchmark reports 40%, that number is a tangled product of perception, recall, and reasoning accuracies. When WorldVQA reports 47.4%, that number is — by construction — an estimate of one thing: the probability the model has the right percept-to-identity mapping internalized.

The practical consequence is that the benchmark refuses a large class of questions that *look* like world-knowledge questions but secretly aren't. "How many windows does this famous building have?" is out — it needs counting. "What year was the monument in this photo completed?" is out — the year is a second fact, a second hop. "What does the sign say?" is out — that's OCR. What survives is the irreducible core: *that* building is the Shanghai Stadium, *that* aircraft is the J-20 fighter, *that* performance is Swan Lake.

### Granularity alignment: penalizing the hypernym escape hatch

Atomic isolation has a failure mode the authors anticipated: a model can dodge a hard recognition by answering with a true-but-vague hypernym. Asked to name a rare orchid species, a model might answer "a flower" — technically not false, but useless. **Granularity alignment** is the grading rule that closes this escape hatch. It enforces a strict match between the specificity of the question and the specificity of the expected entity. "Flower" does not satisfy a question that asks for the species; "a fighter jet" does not satisfy a question that asks for the J-20.

This rule is the mechanism behind one of the paper's sharpest findings: models do *worst* on Nature and Culture not because those images are blurry, but because models "frequently revert to generic hypernyms" there, and granularity alignment refuses to give them partial credit. The grading model — GPT-oss-120b, the primary judge — has to enforce this consistently, which is why the authors validate it (more below). Here is the shape of the granularity check as pseudocode, so the rule is concrete rather than hand-wavy:

```python
def grade_atomic_answer(question, gold_entity, model_answer, judge):
    """Grade a single WorldVQA triplet under granularity alignment.

    The judge (GPT-oss-120b) must reward only answers at the SAME
    taxonomic specificity as the gold entity. A correct-but-generic
    hypernym (e.g. 'flower' for a specific orchid) is graded WRONG,
    because the question's specificity demands the leaf-level name.
    """
    verdict = judge.classify(
        prompt=GRANULARITY_JUDGE_PROMPT,  # few-shot, Appendix A
        question=question,
        gold=gold_entity,
        candidate=model_answer,
    )
    # The judge returns one of three labels, not a similarity score:
    #   CORRECT          -> same entity, same specificity
    #   HYPERNYM_REVERT  -> true but too generic; counts as WRONG
    #   WRONG            -> different entity or hallucination
    if verdict == "CORRECT":
        return {"correct": True,  "attempted": True}
    if verdict == "HYPERNYM_REVERT":
        return {"correct": False, "attempted": True}
    if model_answer_is_refusal(model_answer):
        # 'I cannot identify this' -> attempted=False, not penalized as wrong
        return {"correct": False, "attempted": False}
    return {"correct": False, "attempted": True}
```

Note the third branch: refusals are tracked separately as `attempted=False`. That single distinction is what makes the refusal-aware metrics downstream possible, and it is why the People category had to be handled specially — closed models refuse there for safety reasons, not knowledge reasons.

### Distributional balancing and global expansion

Seed entities come from internal lexicons curated by 10 expert annotators, each with over a year of experience in MLLM evaluation, under "Encyclopedic Knowledge Coverage" criteria and the granularity standard. But a naive seed set drawn from any single team's lexicon will skew regionally. The authors enforce a **50% per-category cap on China-unique entities**, which yields a 36% aggregate share of Chinese-specific entities and an overall language split of English 64.00% (2,240 pairs) and Chinese 36.00% (1,260 pairs) — a stated 1:1.78 Chinese-to-English ratio.

For under-represented categories, they run an **LLM-in-the-loop expansion**: GPT and Kimi perform association search to surface supplemental global entities the human curators might miss. This is a pragmatic use of LLMs as breadth amplifiers rather than as judges — the LLMs propose candidate entities, but every candidate still has to survive deduplication and dual-gate verification before it enters the benchmark.

### Visual deduplication: the leakage problem

This is the stage that separates "the model knows the entity" from "the model memorized this exact image during pretraining." If a benchmark image was scraped into LAION or Common Crawl with its caption, a model can answer correctly by pattern-matching the image to a remembered image-answer pair — that is retrieval, not knowledge, and it inflates the score.

The defense uses **Instance-level Semantic Content (ISC) descriptors**: an embedding of each candidate image's semantic content. The pipeline computes cosine similarity between each ISC embedding and the embeddings of images in LAION and Common Crawl, and applies a **strict 0.95 threshold** — anything above it is treated as a likely duplicate or leaked image and discarded. For entities that lose their assets this way, the team **re-collects new visual assets from video screenshots**, which are far less likely to appear verbatim in web-image corpora. A frame grabbed from a video is a fresh percept of a known entity, so a correct answer on it reflects internalized knowledge rather than image memorization.

```python
def dedup_against_web(candidate_img, web_corpus_index, isc_encoder, tau=0.95):
    """Drop candidate images that are near-duplicates of web-scraped data.

    web_corpus_index covers LAION + Common Crawl. A hit above tau=0.95
    means the model may have seen this exact image (with caption) in
    pretraining, so a correct answer would reflect retrieval, not
    internal knowledge. Such entities are re-shot from video frames.
    """
    q = isc_encoder.embed(candidate_img)           # ISC descriptor
    sims = web_corpus_index.cosine_topk(q, k=8)     # nearest web images
    if max(sims) >= tau:
        return "LEAKED"                             # discard or re-collect
    return "CLEAN"
```

The dossier does not state the exact ISC embedding model variant, so we should not invent one — but the procedure (embed, retrieve nearest web neighbors, threshold at 0.95, re-shoot from video on a hit) is fully specified.

### Difficulty stratification driven by five models

Here is where WorldVQA does something most benchmarks skip: it lets the *models* decide difficulty, then validates that the difficulty is real.

![Five-model difficulty stratification with MetaCLIP validation](/imgs/blogs/worldvqa-4.png)

The pipeline above shows the flow. Every candidate sample is scored by an ensemble of five frontier MLLMs. Samples all five answer correctly are trivial and discarded — keeping them would just pad the benchmark with entities that carry no discriminative signal and re-introduce the ceiling effect. The rest are stratified by how many models got them right: **Easy (>3 models correct), Medium (1–2 correct), Hard (0 correct)**. The Easy bucket is then randomly downsampled so the benchmark is not dominated by simple, famous entities — the focus stays on long-tail knowledge. The final split is Easy 31.16%, Medium 40.77%, Hard 28.07%.

The reason this is more than a labeling convenience is the validation step. A skeptic could argue the "Hard" samples are just visually ambiguous — bad crops, occlusions, confusing angles — rather than genuinely rare entities. The authors rebut this with **MetaCLIP rank-frequency percentiles** as a proxy for real-world entity frequency. They show that Trivial and Easy samples cluster near the 0th percentile (common entities), while Medium and Hard distributions shift progressively rightward toward higher percentiles (rarer entities). In other words, difficulty correlates with genuine knowledge scarcity in the long tail, not with visual confounds. That is the difference between a benchmark whose hard split is hard *for a reason you can name* and one whose hard split is just noise.

### Dual-gate verification

Before any sample is final, it passes two independent gates. The first is **model-based auditing**: a few-shot-prompted Gemini-3-Pro evaluates each (image, question, answer) triplet for three properties — *visual clarity* (the entity is actually visible and identifiable), *semantic exclusivity* (the image supports exactly one answer, not several plausible ones), and *contextual completeness* (the question is answerable from the image alone). The second gate is **human blind validation**: an independent annotator answers the question without seeing the ground-truth answer. If the human's blind prediction diverges from the gold answer, the sample is flagged for manual audit and, if flawed, permanently removed. Only fully validated samples are retained.

This dual gate is what lets the authors claim the benchmark measures knowledge rather than annotation artifacts. A sample that two independent processes — one machine, one human — both find clean and unambiguous is far more likely to be a fair test of recognition.

### The grading judge and its validation

All sixteen models are evaluated "with unified prompts and official inference parameters" (the exact temperature is not stated in the source). The grader is **GPT-oss-120b**, prompted with the judge template in the paper's Appendix A. Because the whole benchmark's credibility rests on the judge applying granularity alignment consistently, the authors validate it directly: they manually audit 160 random samples and find **98.1% alignment with human expertise — only 3 disagreements**. That is a strong number for an LLM judge on a task where the failure mode (accepting hypernyms) is subtle, and it is the kind of validation that should be table stakes for any LLM-as-judge eval but frequently isn't.

One more protocol detail with outsized importance: the **overall average aggregates only the first eight categories**. "Notable People & Public Figures" is excluded because systematic refusals in closed-source models — driven by privacy and safety guardrails, not knowledge deficits — would otherwise corrupt the comparison. In Table 3, a hyphen in the People column means the score was omitted due to excessive refusal. This is a measurement-hygiene decision, and it matters when you read the leaderboard: the overall numbers are *not* contaminated by safety-refusal behavior on faces.

### Comparison: what WorldVQA tests that prior suites don't

| Benchmark family | Primary target | Isolates atomic recognition? | Leakage control | Refusal-aware metrics |
|---|---|---|---|---|
| Comprehensive (MME, MMBench, SEED-Bench, MMStar) | Broad capability survey | No — knowledge is an implicit prerequisite | Not emphasized | No |
| Expert reasoning (MMMU, MMMU-Pro) | Complex reasoning chains | No — reasoning obscures factual deficits | Not emphasized | No |
| Factuality probes (SimpleVQA, VisualSimpleQA) | Factual recall | Partially — but carries OCR / language / multi-hop dependencies | Limited | Partial |
| **WorldVQA** | **Atomic visual factuality** | **Yes — single-hop, no OCR / arithmetic / multi-hop** | **ISC vs LAION+CC at 0.95; video re-collection** | **Yes — CGA, F-score, ECE, Slope** |

## Experiments

The experiments are a single, brutal leaderboard plus three analyses (category disparity, difficulty validation, calibration). The leaderboard is the part you'll quote in meetings, so let's start there.

![WorldVQA leaderboard matrix](/imgs/blogs/worldvqa-3.png)

The matrix above shows the top of the table and the floor. Gemini-3-pro leads on accuracy (47.4%) and F-score (47.5%); Kimi K2.5 is a hair behind on accuracy (46.3%) but is the top open-source model and effectively ties on F-score (46.8%); the worst evaluated model, Kimi-VL-16B-A3B, sits at 12.0% accuracy. The "Not Att." (Not Attempted) column is the refusal signal — note GPT-5.2 at 5.4% and the much larger refusal rates that show up in the full table for the GPT-5.1 generation. Here is the full leaderboard, verbatim from the paper's Table 3 (overall metrics aggregate the first eight categories; the People column uses "-" where a model refused too often to score):

| Model | Accuracy | Not Attempted | CGA | F-score | Nature | Geography | Culture | Objects | Transport | Entertain | Brands | Sports | People |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gemini-3-pro (closed) | **47.4** | 0.6 | 47.7 | **47.5** | 45.1 | 44.7 | 47.2 | 48.1 | 45.1 | 47.6 | 52.4 | 59.4 | - |
| Kimi K2.5 (open) | 46.3 | 2.1 | 47.3 | 46.8 | 40.6 | 46.8 | 43.0 | 44.7 | 47.4 | 48.1 | 52.6 | 64.8 | 50.9 |
| Gemini-2.5-pro (closed) | 36.9 | 0.1 | 36.9 | 36.9 | 37.1 | 33.8 | 32.6 | 39.6 | 39.9 | 34.2 | 38.8 | 54.2 | - |
| Claude-opus-4.5 (closed) | 36.8 | 3.4 | 38.1 | 37.5 | 32.5 | 36.5 | 34.1 | 39.6 | 43.5 | 29.0 | 47.6 | 54.9 | - |
| Seed-1.5-vision-pro (closed) | 34.9 | 1.6 | 35.5 | 35.2 | 41.4 | 36.1 | 33.4 | 32.8 | 35.0 | 33.6 | 32.3 | 43.7 | - |
| GPT-5.2 (closed) | 28.0 | 5.4 | 29.5 | 28.7 | 24.3 | 29.1 | 26.7 | 26.6 | 30.7 | 24.8 | 39.1 | 40.8 | - |
| GPT-5.1 (closed) | 24.5 | 16.3 | 29.3 | 26.7 | 27.3 | 25.1 | 22.5 | 26.6 | 31.6 | 18.5 | 36.0 | 45.4 | - |
| Qwen3-VL-235B-A22B-Instruct (open) | 23.5 | 0.0 | 23.5 | 23.5 | 26.1 | 24.8 | 22.9 | 26.1 | 28.8 | 15.5 | 22.3 | 26.1 | 26.2 |
| GPT-4o (closed) | 22.2 | 9.1 | 24.4 | 23.3 | 25.6 | 20.6 | 17.8 | 19.1 | 26.2 | 19.1 | 35.2 | 44.5 | - |
| Grok-4-1-fast-reasoning (closed) | 21.1 | 0.1 | 21.1 | 21.1 | 18.4 | 23.6 | 20.2 | 25.2 | 23.5 | 11.4 | 25.8 | 30.3 | - |
| Claude-sonnet-4.5 (closed) | 20.0 | 8.0 | 21.8 | 20.9 | 19.4 | 21.0 | 17.4 | 22.9 | 24.8 | 11.6 | 32.2 | 31.0 | - |
| GLM-4.6V (open) | 19.0 | 0.0 | 19.0 | 19.0 | 24.5 | 21.5 | 17.8 | 19.2 | 18.6 | 12.5 | 20.4 | 23.2 | 10.7 |
| Grok-4-fast-reasoning (closed) | 18.9 | 0.2 | 19.0 | 18.9 | 17.8 | 19.0 | 18.6 | 22.0 | 20.3 | 8.3 | 26.6 | 34.5 | - |
| Qwen3-VL-32B-Instruct (open) | 17.7 | 0.0 | 17.7 | 17.7 | 18.1 | 18.0 | 16.8 | 19.0 | 19.0 | 12.1 | 23.8 | 20.4 | 13.1 |
| GLM-4.6V-Flash (open) | 14.8 | 0.1 | 14.8 | 14.8 | 16.0 | 16.3 | 13.2 | 14.9 | 19.0 | 7.8 | 18.8 | 20.4 | 8.2 |
| Kimi-VL-16B-A3B (open) | 12.0 | 3.3 | 12.4 | 12.2 | 11.2 | 13.9 | 10.1 | 10.8 | 13.5 | 7.9 | 20.8 | 17.7 | 7.4 |

Three things are load-bearing in this table, and one of them might not transfer.

**First, the absolute ceiling.** The top accuracy is 47.4%. On a benchmark where a correct answer means "named the specific entity in the picture," the best multimodal model on Earth is wrong more than half the time. That is the whole point of the paper, and it is robust: it is the consequence of the difficulty stratification (trivial items removed) and granularity alignment (no hypernym credit) working as designed.

**Second, the open-vs-closed near-parity at the top.** Kimi K2.5 (open) at 46.3% accuracy and 46.8% F-score is within a point of Gemini-3-pro. This is a genuine result, but read it with the difficulty-construction in mind: the five-model ensemble that set difficulty is *not* disclosed to be disjoint from the evaluated set, and benchmarks scored by frontier ensembles can subtly favor models that resemble the ensemble. The authors state the stratification is "explicitly not designed to trap specific models," and the MetaCLIP analysis argues difficulty is real, but the exact composition of the five-model ensemble is the kind of detail that, if it overlapped with top scorers, could nudge the top of the leaderboard. I'd want that disjointness stated explicitly before treating the open-source parity as fully clean.

**Third, the refusal column tells a second story.** Look at GPT-5.1: 24.5% accuracy but a 16.3% Not-Attempted rate and a CGA of 29.3 that exceeds its accuracy. That is a *conservative, honest-refusal* profile — when GPT-5.1 commits, it is more often right (29.3 CGA) than its raw accuracy suggests, because it declines the questions it would have gotten wrong. Contrast the smaller models (Qwen3-VL variants, GLM-4.6V) with 0.0% Not-Attempted: they essentially never abstain, so they hallucinate rather than decline, and their CGA equals their accuracy. The refusal-aware suite is what surfaces this difference; raw accuracy alone would flatten two very different behaviors into one number.

### Category disparity: where the long tail bites

The category columns confirm a clean pattern. Models do best on **Brands and Sports** — Gemini-3-pro's Sports F-score is 59.4, and Kimi K2.5's Sports F-score is 64.8, the single highest cell in the table. The paper attributes this to over-representation of brands and sports content in web-scale pretraining data. Models do *worst* on **Nature and Culture**, where they "frequently revert to generic hypernyms" and granularity alignment refuses partial credit. This is the long tail biting: rare species, regional crafts, and artisanal artifacts are exactly the entities that are sparse in pretraining corpora and specific enough that "a flower" or "a vase" earns nothing.

![Head-class versus long-tail taxonomy](/imgs/blogs/worldvqa-5.png)

The taxonomy tree above splits the nine categories into the head-class winners (Sports, Brands, Entertainment) and the long-tail losers (Nature, Culture) plus the excluded People category. The asymmetry is the actionable finding: if you are deploying a vision model into a domain that looks like the long tail — biodiversity surveys, cultural-heritage cataloging, niche product identification — the leaderboard's headline 47% is an *optimistic* upper bound, because those domains live in the categories where every model is weakest.

### Calibration: the universal overconfidence

The last analysis is the one I'd put on a slide for any team shipping an MLLM into a high-stakes loop. Every model is **severely overconfident**, measured by Expected Calibration Error (ECE, optimal 0) and a confidence-accuracy Slope (optimal 1.0). The reported values:

| Model | Slope (optimal 1.0) | ECE (optimal 0) | Reading |
|---|---|---|---|
| Kimi K2.5 | 0.550 | 37.9% | Best calibration of all models, still far from honest |
| Gemini-3-pro | 0.541 | 44.0% | Top accuracy, near-binary confidence |
| GPT-5.1 | 0.396 | 42.3% | Only model that meaningfully distinguishes low confidence |
| Qwen3-VL-235B-A22B-Thinking | 0.261 | 64.8% | Thinking variant barely helps calibration |
| Qwen3-VL-235B-A22B-Instruct | 0.252 | 69.0% | Worst calibration in the set |

Every Slope is below 1.0, which means every model's confidence rises faster than its accuracy — they get more sure without getting more right. The most striking single statistic: **Gemini-3-pro asserts ≥95% confidence in over 85% of cases regardless of whether it is correct.** Its confidence is essentially a constant near 1.0, carrying almost no information about correctness. GPT-5.1 is the lone partial exception — it is the only model that meaningfully distinguishes low-confidence cases and gives more honest uncertainty, which is the calibration-side mirror of its high refusal rate.

![The refusal-aware metric stack](/imgs/blogs/worldvqa-6.png)

The stack above is the metric hierarchy that exposes all of this. Accuracy sits at the bottom — it games refusal, because a model that declines its hard questions can look better than one that attempts them. CGA adds precision-on-attempt, isolating hallucination when the model commits. F-score balances coverage against precision so neither over-refusal nor over-guessing wins. And calibration sits on top, where the verdict is that best-in-class Slope is only 0.550 and every model is overconfident. The paper attributes the overconfidence to two causes: a lack of uncertainty-bearing samples in training data, and alignment objectives that favor assertiveness. Both are fixable, and both are squarely outside what a benchmark can fix on its own — which is exactly why the benchmark measuring it matters.

### What might not transfer

The number that is most likely to *not* transfer is the open-source/closed-source parity at the top, for the ensemble-composition reason above. The number most likely to transfer — because it is a direct consequence of the construction — is the sub-50% ceiling and the universal overconfidence. If you rebuilt WorldVQA with a different judge and a different five-model ensemble, I would bet the ceiling stays under 50% and every Slope stays under 1.0, while the exact ordering of the top three models could shuffle by a point or two.

To make the metric interplay concrete, here is a worked example using the table's own numbers. Take GPT-5.1: accuracy 24.5%, Not-Attempted 16.3%, CGA 29.3, F-score 26.7. The CGA exceeds the accuracy because CGA is precision computed only over the samples the model *attempted* — it answered roughly 83.7% of the benchmark, got 24.5% of the full set right, and that 24.5% correct-of-total works out to about 29.3% correct-of-attempted once you remove the 16.3% it declined. The F-score (26.7) then sits between the two, harmonically penalizing the model for the coverage it sacrificed by refusing. Now contrast Qwen3-VL-235B-A22B-Instruct: accuracy 23.5%, Not-Attempted 0.0%, CGA 23.5, F-score 23.5 — all four numbers collapse to the same value because the model never abstains. GPT-5.1 and Qwen3-VL-235B land within a point of each other on raw accuracy (24.5 vs 23.5), but they are behaviorally opposite: one declines its hard questions and is more reliable when it commits, the other answers everything and hallucinates the gap. Raw accuracy alone would tell you these two models are equivalent. The refusal-aware suite tells you they are not, and that difference is precisely what determines which one is safe to put in front of a user who will trust a confident answer.

A second worked example shows why the calibration Slope is the metric I would actually monitor in production. A Slope of 1.0 means a model that reports 70% confidence is right 70% of the time. Gemini-3-pro's Slope of 0.541, combined with its ≥95%-confidence-in-over-85%-of-cases behavior, means its confidence is nearly a constant: it reports something like 0.95+ whether the true probability of correctness is 0.9 or 0.2. So if you build a downstream gate that only accepts answers above 90% model confidence, that gate passes essentially everything Gemini-3-pro says — including the more-than-half of WorldVQA questions it gets wrong. The gate is useless because the signal it keys on carries no information. Kimi K2.5's marginally better Slope of 0.550 and ECE of 37.9% does not rescue this; it is the best of a badly-calibrated field, not a well-calibrated model. The actionable read is that *no model in this set has confidence you can threshold on for entity recognition* — a finding that is invisible to accuracy and central to deployment.

## Critique

**What's strong.** The atomic-isolation framing is genuinely useful and, as far as I can tell, correctly executed. The decision to assign difficulty from a five-model ensemble *and then validate that difficulty against MetaCLIP frequency* is the move that elevates this from "yet another VQA set" to a credible measurement instrument — most benchmarks assert their difficulty splits are meaningful without ever checking. The leakage control (ISC embeddings vs LAION and Common Crawl at 0.95, with video re-collection) addresses the single most damning critique of knowledge benchmarks, that they measure memorization of scraped image-caption pairs. And the refusal-aware metric suite, plus the explicit exclusion of the People category from the overall average, shows the authors thought hard about the ways closed-model safety behavior corrupts knowledge measurement. The 98.1% judge-human alignment on 160 samples is the right validation to run and a strong result.

**What's weak or unfalsifiable.** The five-model ensemble used for difficulty stratification is not disclosed (in the dossier) to be disjoint from the evaluated models. If it overlaps with the top scorers, the difficulty stratification could subtly favor those models, and the "explicitly not designed to trap specific models" claim is hard to falsify without the ensemble roster. The MetaCLIP frequency argument is good but indirect — rank-frequency percentile is a proxy for "rarity," not a direct measure of whether a model *should* know an entity. And the single largest interpretive risk is in the granularity-alignment grading: the line between "acceptable specific answer" and "hypernym revert" is judge-dependent, and while 98.1% alignment on 160 samples is reassuring, the categories where this boundary matters most (Nature, Culture) are exactly the categories with the lowest scores — so a small systematic judge bias toward strictness there would amplify the very gap the paper highlights.

**The missing ablation.** I want a judge-sensitivity ablation: re-grade the full benchmark with a *second* judge (a different model family) and report how much the leaderboard moves. The whole result rests on GPT-oss-120b applying granularity alignment consistently; 160 hand-checked samples validate the judge on average, but they do not tell us whether judge choice changes the *ordering* of models, which is what readers actually use the leaderboard for. A second missing piece is a held-out human ceiling: how do expert humans score on the Hard split? Without it, we know models are below 50% but not how far below the human bar that is.

**What would change my mind.** If the authors released the five-model stratification roster and showed it is disjoint from (or robustly uncorrelated with) the top of the leaderboard, *and* a second-judge re-grade preserved the model ordering within a point or two, I would treat the open-source parity and the exact rankings as solid rather than provisional. Conversely, if a second judge reshuffled the top three, I would downgrade every claim about specific model ordering and keep only the aggregate finding that no model clears 50%.

## What I'd build with this

1. **A long-tail recognition fine-tuning eval harness.** WorldVQA's Hard split, sorted by MetaCLIP percentile, is a ready-made curriculum signal. I'd build a harness that fine-tunes a small open VLM on rare-entity recognition and tracks movement specifically on the Hard/Nature/Culture cells, since those are where the headroom is. The benchmark's per-category breakdown makes it trivial to detect whether a fine-tune actually moved long-tail knowledge or just re-fit the head classes.

2. **An abstention-trained variant.** Given that GPT-5.1's honest-refusal profile is the calibration outlier, I'd train a model with explicit "I cannot identify this" supervision on the Hard split and measure CGA and F-score, not accuracy. The hypothesis is that a model that abstains correctly on the bottom MetaCLIP percentiles can beat a more knowledgeable but overconfident model on F-score. WorldVQA's refusal-aware metrics are exactly the scoreboard for that bet.

3. **A calibration probe pipeline for production.** The finding that Gemini-3-pro asserts ≥95% confidence in over 85% of cases is a deployment red flag. I'd lift the ECE/Slope methodology out of the paper and run it continuously against production traffic on entity-naming tasks, alerting when a model's confidence distribution collapses toward the 90–100% band — that collapse is the leading indicator of silent hallucination.

4. **A leakage-audit service.** The ISC-vs-web-corpus dedup at a 0.95 threshold is a reusable component. I'd package it as a pre-eval gate for *any* internal vision benchmark: before trusting a score, check what fraction of the eval images are near-duplicates of LAION/Common Crawl, and re-collect the flagged ones from video. Most internal evals skip this and silently measure memorization.

5. **A bilingual recognition stress test.** The 64/36 EN/CN split with a 50% China-unique cap is a clean template for measuring whether a model's visual knowledge is anglocentric. I'd extend the construction protocol to a third language and measure the per-language recognition gap, which is a sharper signal of training-data coverage than any aggregate score.

## When to reach for WorldVQA (and when not to)

Reach for WorldVQA when your question is specifically *does this model recognize and correctly name the entities it sees, and how confidently does it lie when it doesn't?* It is the right instrument for vetting a vision model destined for entity-identification work — product recognition, landmark or species identification, media tagging, brand detection — and for any setting where a confidently-wrong label is more dangerous than an abstention. The refusal-aware metrics make it uniquely good at separating "this model knows less" from "this model lies more," which is the distinction that actually drives deployment risk. If you care about long-tail coverage or about whether your model's confidence means anything, this benchmark will tell you something your comprehensive suites are hiding.

Do *not* reach for WorldVQA as a proxy for general multimodal capability. The authors are explicit that it measures factuality in a highly atomic setting and that whether atomic entity-naming correlates with complex downstream multimodal performance is an open question they do not answer. A model that scores well here is not thereby a good chart-reader, document-understander, or multi-step visual reasoner — those abilities were deliberately excluded so that this one ability could be measured cleanly. And do not over-index on the exact model ordering at the top until the judge-sensitivity and ensemble-disjointness questions are settled; treat "no frontier model clears 50%, and all of them are overconfident" as the durable takeaway, and the specific rank of any one model as provisional. Used that way — as a sharp, narrow diagnostic rather than a general report card — WorldVQA is one of the more honest measurements the multimodal field has produced.

## References

- WorldVQA: Measuring Atomic World Knowledge in Multimodal Large Language Models — arXiv abstract: [https://arxiv.org/abs/2602.02537](https://arxiv.org/abs/2602.02537)
- Code and dataset (GitHub): [https://github.com/MoonshotAI/WorldVQA](https://github.com/MoonshotAI/WorldVQA)
- Related reading on the model families evaluated here:
  - [Kimi-VL: A Mixture-of-Experts Vision-Language Model](/blog/paper-reading/multimodal/kimi-vl)
  - [Kimi K2.5: Visual Agentic Intelligence](/blog/paper-reading/large-language-model/kimi-k2-5)
  - [Kimi K2 Thinking: An Open-Source Reasoning Model Built on K2](/blog/paper-reading/large-language-model/kimi-k2-thinking)
  - [CombiBench: Benchmarking LLMs on Combinatorial Mathematics](/blog/paper-reading/reasoning/combibench)
