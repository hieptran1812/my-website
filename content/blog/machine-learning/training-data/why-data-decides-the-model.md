---
title: "Why data decides the model: the data-centric turn in training"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Same architecture, better data, better model. A principal-engineer tour of why the corpus, not the network, is the dominant lever in modern training, and how to tell a data problem from a model problem."
tags:
  - training-data
  - data-centric-ai
  - large-language-models
  - data-quality
  - deduplication
  - decontamination
  - fineweb
  - scaling-laws
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 34
---

Here is a bet I will win more often than not. Take any two teams with the same compute budget, the same model size, and free choice of architecture. Tell me only one thing about each: how much care went into their training data. I will predict which model wins, and I will be right far more than chance — usually by a margin that no amount of architecture cleverness on the losing side can close.

That is not how most of us were trained to think. We came up reading architecture papers — a new attention variant, a new normalization, a new positional scheme — and we internalized that progress comes from the network. It does not, anymore. The architecture of a frontier language model in 2026 is, to a first approximation, a solved and *converged* artifact: a decoder-only transformer with rotary positions, RMSNorm, SwiGLU, and grouped-query attention. Swap the logo on the report and the model diagrams are nearly interchangeable. What is *not* interchangeable — what teams guard like a trade secret and what actually moves the evaluation numbers — is the data.

The diagram above is the mental model for this entire series: stop iterating on the model, freeze it, and iterate on the corpus instead. That single change in where you spend your effort is the data-centric turn, and it is the most important shift in applied machine learning of the last several years.

![From model-centric to data-centric iteration](/imgs/blogs/why-data-decides-the-model-1.webp)

This is post #1 of a 25-part series on the full lifecycle of training data — collection, extraction, cleaning, deduplication, decontamination, filtering, selection, mixing, and per-modality recipes. Before we zoom into any single stage, this post sets the frame: *why* data is the dominant lever, *how* the three axes of quality, diversity, and quantity trade against each other, *what* the evidence says, and — most practically — *how to tell a data problem from a model problem* when your eval numbers disappoint.

## Why this is different from what you were taught

Most engineers carry a mental model of training that quietly overweights the network. Here is the mismatch between the folklore and the reality, laid out plainly.

| Assumption | The naive view | The reality in 2026 |
| --- | --- | --- |
| Where gains come from | A better architecture or a clever loss | A better corpus at the *same* architecture and compute |
| What a "good dataset" is | Bigger is better; scrape more | Higher signal-per-token; often *smaller after filtering* |
| Who owns model quality | The modeling / research team | The data team — collection, filtering, mixing |
| What you iterate on | Layers, heads, learning-rate schedules | Dedup thresholds, quality classifiers, domain weights |
| How you debug bad eval | Tune hyperparameters, add regularization | Read 100 random training documents first |
| What is the moat | The weights and the recipe | The *data pipeline* that produced the weights |

None of this means architecture is irrelevant. It means architecture is *table stakes*: necessary, well understood, and no longer where the differentiated leverage lives. The leverage moved. This series is about following it.

> The model is a lossy compression of its training set. If the training set is mediocre, no decoder can decompress greatness out of it.

## 1. The model-centric era, and why it ended

For most of deep learning's history, the data was treated as a fixed, given thing — a benchmark you downloaded — and the research happened on top of it. You took ImageNet as a constant and competed on the architecture: AlexNet, VGG, ResNet, DenseNet. You took a fixed translation corpus and competed on the encoder-decoder. The dataset was the *board*; the model was the *move*.

Andrew Ng put a name on the inversion in 2021 with his "data-centric AI" campaign, and his framing has aged extremely well. His observation, from years of deploying models in industry, was that teams systematically over-invested in model changes and under-invested in data quality — and that on most real problems, *systematically improving the data* beat *systematically improving the model*, often by a wide margin. In one widely shown example, a defect-detection model stuck at 76% accuracy did not move when the team tried bigger networks and hyperparameter sweeps; it jumped to over 90% when they cleaned and made the *labels* consistent. Same model. Different data.

Two forces made this inversion structural for large language models specifically:

**Architectures converged.** Since "Attention Is All You Need" in 2017, the field ran an enormous, distributed architecture search, and it landed. The decoder-only transformer won. The remaining deltas — RoPE versus learned positions, RMSNorm versus LayerNorm, SwiGLU versus GELU, multi-head versus grouped-query attention — are real but small, worth low single-digit percentages, and largely shared across every serious lab. When everyone runs the same engine, the engine stops being the differentiator.

**Compute became the constraint, which made tokens precious.** Once you accept the [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) result — that for a fixed compute budget you should train a smaller model on more tokens, roughly 20 tokens per parameter — the question stops being "how clever is my network?" and becomes "what are the *best possible* tokens I can feed it within my budget?" The moment tokens are the bottleneck, the *quality* of each token is the lever. This is the bridge between scaling laws and data-centric AI, and it is why [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) are now their own field of study.

It helps to look at the historical arc, because the inversion was gradual and then sudden. The ImageNet decade (2012–2017) was the purest model-centric era: a frozen, hand-labeled benchmark of 1.2M images, and a leaderboard that moved entirely on architecture — eight layers (AlexNet), then nineteen (VGG), then residual connections that unlocked a hundred-plus (ResNet), then dense connectivity (DenseNet). The data was a constant by construction; that was the point of a benchmark. Natural language went through the same phase with fixed translation and language-modeling corpora. The lesson everyone absorbed — *progress equals architecture* — was *correct for that regime* and is the source of the folklore we are now unlearning.

What flipped the regime was a change in economics. A frontier pretraining run is now a multi-million-dollar capital expenditure measured in tens of thousands of GPU-hours, and you get *one* serious shot at it per quarter. When a single run costs that much, two things follow. First, you cannot afford to discover after the fact that 30% of your tokens were duplicated boilerplate — that is 30% of a seven-figure budget spent teaching the model nothing. Second, the marginal return on an architecture tweak that the rest of the field has already found is close to zero, while the marginal return on a corpus that is genuinely cleaner than your competitor's is enormous and *yours alone*. The money moved the incentive, and the incentive moved the engineering effort onto the data.

The practical consequence is the loop in the figure above. In the **model-centric loop**, you *freeze the dataset*, then *tweak architecture and hyperparameters*, then *train, evaluate, repeat* — and you hit *diminishing returns*, because you are searching a space that the whole field has already mapped. In the **data-centric loop**, you do the opposite: *freeze the architecture*, then *improve data: clean, dedup, filter, mix*, then *train, evaluate, repeat* — and you get *compounding gains at equal compute*, because the data space is vast, under-explored, and specific to your goals. The habit you are trying to build is the second loop.

## 2. The three axes: quality, diversity, and quantity

"Better data" is not one thing. It is a negotiation between three axes that pull against each other under a fixed budget. Get specific about which one you are buying, because you cannot buy all three at once.

![The quality-diversity-quantity triangle](/imgs/blogs/why-data-decides-the-model-2.webp)

- **Quality** is *signal per token*: how much of each document is information a model should learn versus boilerplate, spam, broken markup, or noise. Raising quality means *throwing tokens away* — you keep the top fraction and discard the rest. The cost of quality is paid in *tokens dropped*.
- **Diversity** is coverage across *domains and skills*: code, math, prose, dialogue, multiple languages, multiple registers. A model only learns what is represented; a corpus that is 95% news and blogs will be weak at code and arithmetic no matter how clean it is. The cost of diversity is that broadening the net *adds noise* — the long tail of domains is messier and harder to filter.
- **Quantity** is raw scale: *more tokens*, which (per Chinchilla) you genuinely need to feed a larger model. The cost of quantity is that the marginal token is *low-value text* — once you have exhausted the high-signal sources, scaling up means scraping deeper into the sludge.

The reason these form a triangle and not three independent dials is the budget in the middle: a *fixed token and compute budget*. You can buy aggressive quality filtering, but only by dropping tokens, which forces you to either shrink the corpus (losing quantity) or scrape harder (losing quality on the margin, or narrowing diversity). The figure's claim is exactly this — under a fixed budget, **pushing one corner pulls resources away from the other two.**

Here is how the three axes cash out as concrete decisions you will actually face:

| Lever | What it buys | What it costs | When to pull it |
| --- | --- | --- | --- |
| Aggressive quality filter (keep top 10–30%) | Higher per-token signal; faster convergence | Fewer tokens; risk of over-pruning a domain | Compute-rich, token-rich regime |
| Broaden sources (more domains/languages) | Coverage of under-served skills | Noisier tail; harder dedup/filter | When eval shows a domain gap |
| Scale up raw scraping | More tokens for a bigger model | Diminishing marginal quality | When you are token-limited for your model size |
| Upweight a high-value domain in the mix | Targeted skill gains (e.g. code, math) | Less budget for everything else | When a benchmark is the priority |

Make the tradeoff concrete with a budget you can actually allocate. Say you have a 300B-token budget and a raw pool of 1T deduplicated web tokens plus 50B code tokens and 20B math tokens. You face a real choice. Option (a): take the top 30% of the web pool by quality (300B tokens), all web. Clean, but you have spent your entire budget on a single domain — strong prose, weak code and math, a diversity failure. Option (b): take the top 10% of web (100B, *higher* average quality because you filtered harder), all 50B code tokens, all 20B math tokens, and repeat code/math ~1.5× to reach 300B. Now you have bought diversity and per-token quality, but you paid for it by *repeating* the scarce domains and by shrinking the unique web set. Option (c): take the top 20% of web (200B) plus the 70B code+math, and backfill the last 30B with books. The point is not that one option is universally right — it is that the three numbers (quality threshold, domain mix, repetition factor) are *coupled*, and moving any one of them forces a move in the others under the fixed 300B ceiling. That coupling is the triangle.

A useful instinct: **at a fixed compute budget, quality usually wins the first round and diversity wins the second.** Cleaning the corpus buys you the biggest, cheapest gain; once it is clean, the next gain comes from filling the domains you are missing. Quantity is the axis you scale *last*, when model size forces your hand. We will quantify all three in the [data-scaling-laws-and-budgets](/blog/machine-learning/training-data/data-scaling-laws-and-budgets) post and turn quality into a measurable number in [measuring-data-quality](/blog/machine-learning/training-data/measuring-data-quality).

## 3. The data lifecycle, end to end

Everything in this series is a zoom-in on one stage of a single pipeline. This is the master map.

![The end-to-end training-data lifecycle](/imgs/blogs/why-data-decides-the-model-3.webp)

Read it left to right, top row then bottom row. Raw text becomes a model through a dozen ordered stages, and *each one is able to make or break final quality.* A quick tour, because the rest of the series lives inside these boxes:

1. **Collect** (web, code, books). Decide what universe of text you are willing to train on — CommonCrawl dumps, code hosts, book corpora, scientific archives. Source selection is the first and most consequential bias you introduce.
2. **Extract** (HTML to text). Turn raw HTML/PDF/markup into clean text. This is where boilerplate, navigation chrome, and "HTML soup" either get stripped or silently poison your corpus.
3. **Clean** (normalize, strip). Unicode normalization, encoding fixes, removing control characters, fixing mojibake, dropping documents that are mostly symbols or whitespace.
4. **Dedup** (MinHash, exact). Remove exact and near-duplicate documents. Duplicates inflate the apparent loss curve, waste compute, and cause memorization. This is the highest-leverage single stage for most corpora.
5. **Decontaminate** (remove eval). Strip any text that overlaps your evaluation benchmarks, so your scores measure generalization rather than leakage. The easiest stage to skip and the most embarrassing to skip.
6. **Filter** (quality classifier). Keep high-signal documents and drop low-signal ones, usually with a learned classifier or heuristic ensemble. This is the axis that "data quality" most directly refers to.
7. **Select** (curriculum, hard). Within what survives filtering, decide ordering and emphasis — easy-to-hard curricula, upsampling hard or rare examples.
8. **Mix** (domain weights). Set the proportion of web, code, math, books, and languages. The mixture is a hyperparameter as important as the learning rate, and it is invisible if you do not track it.
9. **Format** (pack, template). Tokenize, pack into sequences, apply chat/instruction templates, choose document-separator and packing strategies.
10. **Train** (fixed arch). The part everyone pictures when they say "training," and the *least* differentiated step in the whole chain.
11. **Eval** (benchmarks). Measure what you got — and, critically, decide whether a bad number is a data problem or a model problem.
12. **Flywheel** (data feedback). Use what eval and production tell you to go back and fix collection, filtering, and mixing. The loop that turns a one-shot dataset into a compounding asset.

The single most important property of this diagram is that **errors propagate downstream and masquerade as something else.** A contamination bug at stage 5 looks like a "surprisingly good model" at stage 11. A mixture mistake at stage 8 looks like a "reasoning weakness in the architecture." Most of the debugging skill in this field is learning to walk *backward* up this pipeline from a confusing symptom to its true stage. Sections 7 and 8 are entirely about that skill.

### The second-order rule: every stage interacts with the others

The stages are not independent filters you can tune in isolation, and the most expensive mistakes come from forgetting it. Three interactions to keep in your head:

- **Filtering silently rewrites the mixture.** A quality classifier trained mostly on clean prose will score code, tables, and math as "low quality" and drop them disproportionately. You set out to raise quality and accidentally narrowed diversity. *Always re-measure your domain proportions after the filter stage*, not before.
- **Dedup must run across splits, not just within them.** Deduplicating the training set is necessary but insufficient: if a near-duplicate of a validation document survives in training, your validation loss is corrupted and every decision you make from it is poisoned. Dedup is a *global* operation over train, validation, and eval together.
- **Decontamination has to happen after collection but its scope is defined by eval.** You cannot decontaminate against benchmarks you have not enumerated. New eval added late means re-running decontamination, or accepting that your new benchmark might already be in the corpus.

> A pipeline is only as honest as its leakiest stage. One skipped decontamination pass invalidates every number downstream of it.

### The flywheel is the whole game

Stage 12 — *flywheel* — is the one that turns a dataset from a one-shot artifact into a compounding asset, and it is the stage most teams never build. The idea is to close the loop: production traffic and eval failures tell you exactly which domains are weak, which sources are noisy, and which skills are missing; you feed that signal back into *collection*, *filtering*, and *mixing* for the next run. The teams that win over multiple model generations are not the ones with the best single corpus — they are the ones whose corpus gets measurably better every quarter because the feedback loop is instrumented. A static dataset depreciates; a flywheel appreciates.

## 4. The evidence: equal tokens, better corpus, better model

The data-centric claim is falsifiable, and it has been tested cleanly. The gold-standard experiment is an **ablation at a fixed token budget**: hold the architecture, the parameter count, the optimizer, and the *number of training tokens* constant, change *only the corpus*, and compare downstream evals. If data were a minor lever, the curves would overlap. They do not.

![FineWeb wins at equal token budgets](/imgs/blogs/why-data-decides-the-model-4.webp)

The clearest public example is **FineWeb** (Penedo et al., 2024), a 15-trillion-token English web corpus built with HuggingFace's `datatrove` pipeline. The FineWeb team ran exactly the ablation above: a fixed ~1.7B-parameter model trained on a fixed token budget, swapping only the source corpus. Across the standard suite — *HellaSwag*, *ARC*, *MMLU*, and others rolled into an *aggregate* score — models trained on **FineWeb** consistently edged out the same model trained on **C4**, **RefinedWeb**, Dolma, RedPajama-v2, and The Pile. The figure shows the shape of the result (numbers are representative of the published ablations): at equal tokens, the curated web corpus leads on every benchmark, and the aggregate gap is the kind of margin you would otherwise chase with a much larger model.

The follow-up, **FineWeb-Edu**, makes the point even sharper. The team used a large model to annotate documents for "educational value," trained a small classifier on those annotations, and kept only the high-scoring documents. That single *filtering* change — same collection, same dedup, just a stricter quality gate — produced large jumps specifically on knowledge-and-reasoning benchmarks like MMLU and ARC, at a *fraction* of the tokens. More aggressive filtering, fewer tokens, better model. That is the quality axis winning a round outright.

The other landmark is **phi** (Gunasekar et al., 2023, "Textbooks Are All You Need"). The phi-1 model is 1.3B parameters trained on only ~7B tokens — a rounding error by frontier standards — but the tokens were "textbook-quality": filtered code plus synthetically generated textbooks and exercises. phi-1 reached **50.6% pass@1 on HumanEval**, beating models an order of magnitude larger trained on two orders of magnitude more data. phi-1.5 extended the recipe to general reasoning with ~30B synthetic tokens and stayed competitive with models 5–10× its size. The phi line is the most extreme demonstration that *what* the tokens are can dominate *how many* there are.

A small but load-bearing piece of code: the experiment that produces these claims is mechanically simple, which is exactly why it is convincing. You are not allowed to change anything but the data.

```python
# The equal-token ablation, in pseudocode-but-runnable form.
# Same model, same optimizer, same token budget — only `corpus` differs.
from dataclasses import dataclass

@dataclass(frozen=True)
class RunConfig:
    n_params: int = 1_700_000_000     # identical architecture
    n_tokens: int = 350_000_000_000   # identical token budget
    seq_len: int = 2048
    lr: float = 3e-4
    warmup: int = 2000
    seed: int = 0                     # identical init

def train_and_eval(corpus_path: str, cfg: RunConfig) -> dict:
    model = build_decoder_only(cfg.n_params)        # fixed recipe
    opt = adamw(model, lr=cfg.lr, warmup=cfg.warmup)
    loader = token_stream(corpus_path, cfg.seq_len) # the ONLY thing that varies
    for step in range(cfg.n_tokens // (cfg.seq_len * GLOBAL_BATCH)):
        loss = model.step(next(loader), opt)
    return run_lm_eval(model, tasks=["hellaswag", "arc_easy", "mmlu"])

# Everything is frozen except corpus_path. Any eval delta is attributable to data.
results = {c: train_and_eval(c, RunConfig()) for c in
          ["c4", "refinedweb", "fineweb"]}
```

When you control everything else and the only free variable is the corpus, the resulting eval delta is *causally* attributable to the data. That is what makes these ablations the strongest evidence in the field — and what makes the data-centric claim more than a slogan.

## 5. The filtering funnel: petabytes in, trillions of tokens out

Where do these corpora come from? Not from a clean source — from an enormous, filthy one that is then *filtered down by orders of magnitude.* The mental model is a funnel: you start with petabytes of raw web crawl and end with a few trillion clean tokens, shedding most of the volume at each stage. Watch the stages narrow.

<figure class="blog-anim">
<svg viewBox="0 0 720 440" role="img" aria-label="A filtering funnel: roughly 100 petabytes of raw web crawl is reduced stage by stage to about 15 trillion training tokens, while a highlight sweeps down through the stages." style="width:100%;height:auto;max-width:760px">
<style>
.a5-bar{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a5-final{fill:var(--accent,#6366f1);stroke:var(--accent,#6366f1)}
.a5-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a5-out{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:start}
.a5-win{font:700 14px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
.a5-glow{fill:var(--accent,#6366f1);opacity:0}
@keyframes a5-pulse{0%,6%{opacity:.8}22%,100%{opacity:0}}
.a5-g0{animation:a5-pulse 10s ease-in-out infinite;animation-delay:0s}
.a5-g1{animation:a5-pulse 10s ease-in-out infinite;animation-delay:2s}
.a5-g2{animation:a5-pulse 10s ease-in-out infinite;animation-delay:4s}
.a5-g3{animation:a5-pulse 10s ease-in-out infinite;animation-delay:6s}
.a5-g4{animation:a5-pulse 10s ease-in-out infinite;animation-delay:8s}
@media (prefers-reduced-motion:reduce){.a5-g0,.a5-g1,.a5-g2,.a5-g3,.a5-g4{animation:none}}
</style>
<rect class="a5-bar" x="60" y="30" width="600" height="56" rx="8"/>
<rect class="a5-bar" x="125" y="108" width="470" height="56" rx="8"/>
<rect class="a5-bar" x="190" y="186" width="340" height="56" rx="8"/>
<rect class="a5-bar" x="250" y="264" width="220" height="56" rx="8"/>
<rect class="a5-bar a5-final" x="295" y="342" width="130" height="56" rx="8"/>
<rect class="a5-glow a5-g0" x="60" y="30" width="600" height="56" rx="8"/>
<rect class="a5-glow a5-g1" x="125" y="108" width="470" height="56" rx="8"/>
<rect class="a5-glow a5-g2" x="190" y="186" width="340" height="56" rx="8"/>
<rect class="a5-glow a5-g3" x="250" y="264" width="220" height="56" rx="8"/>
<rect class="a5-glow a5-g4" x="295" y="342" width="130" height="56" rx="8"/>
<text class="a5-lbl" x="360" y="64">Raw web crawl    ~100 PB</text>
<text class="a5-lbl" x="360" y="142">URL + language filter    ~12 PB</text>
<text class="a5-lbl" x="360" y="220">Dedup (MinHash)    ~3 PB</text>
<text class="a5-lbl" x="360" y="298">Quality filter    ~600 TB</text>
<text class="a5-win" x="360" y="376">~15T tokens</text>
<text class="a5-out" x="440" y="376">decontaminated, packed</text>
</svg>
<figcaption>The data funnel: ~100 PB of raw crawl is filtered stage by stage down to ~15T training tokens; the highlight marks the active stage. Volumes are illustrative.</figcaption>
</figure>

The numbers are illustrative but the *shape* is real and routinely surprising to people who have not built a pipeline: you discard 99%+ of what you collect, and the discarding is where the quality comes from. A typical English-web funnel:

- **Raw web crawl** — on the order of 100 PB of raw HTML across a multi-year crawl archive.
- **URL + language filter** — drop adult/spam domains and non-target languages; an order of magnitude gone immediately.
- **Dedup (MinHash)** — remove near-duplicate documents; web crawls are *extraordinarily* repetitive (mirrors, syndication, templated pages), and this stage alone often removes 50–80% of what remains.
- **Quality filter** — keep the high-signal fraction with a classifier; another large cut.
- **Decontaminate and pack** — strip eval overlap, tokenize, and you land at the few trillion tokens that actually train the model.

Two non-obvious consequences fall out of the funnel's geometry. First, **the dedup stage is doing quiet, enormous work**: because it removes the most-repeated content, and the most-repeated content is disproportionately low-value (boilerplate, SEO spam, license headers), dedup improves quality *and* shrinks compute at the same time — a rare win on two axes at once. Second, **the yield is the hidden constraint on quantity.** If your model size demands 3T tokens (Chinchilla-style) and your funnel yields 1% of a 200 PB crawl, you can back out exactly how much raw data you must collect. The funnel is not just a cleaning step; it is the equation that ties your collection budget to your model size.

```python
# Back-of-envelope: how much RAW crawl must I collect for a target token budget?
target_tokens = 3_000_000_000_000     # 3T tokens for a Chinchilla-optimal ~150B model
bytes_per_token = 4                    # ~4 bytes/token of clean UTF-8 text after tokenization
funnel_yield = 0.01                    # 1% survives URL+lang+dedup+quality, end to end

clean_bytes  = target_tokens * bytes_per_token          # ~12 TB of clean text
raw_bytes    = clean_bytes / funnel_yield               # ~1.2 PB of raw text to collect
print(f"clean: {clean_bytes/1e12:.1f} TB   raw needed: {raw_bytes/1e15:.2f} PB")
# clean: 12.0 TB   raw needed: 1.20 PB
```

If you have ever wondered why a "small" 12 TB dataset implies a petabyte-scale collection-and-storage operation, this is why. The yield term is the whole story, and it is exactly the lever the quality filter controls.

## 6. A worked scenario: two runs, same architecture

Abstract arguments are easy to nod along to and easy to forget. Let us walk a concrete one, with numbers, because this is the scenario you will actually live through.

![Same architecture, different data, different model](/imgs/blogs/why-data-decides-the-model-6.webp)

You are training a **1.5B-parameter** decoder-only model. You fix *everything* about the run — the architecture, the optimizer (AdamW, identical learning-rate schedule and warmup), the global batch size, the random seed, and a **100B-token budget** that costs roughly the same GPU-hours either way. The only thing you change is the corpus:

- **Corpus A: raw crawl (dedup only).** A single CommonCrawl dump, MinHash-deduplicated and nothing else. No quality filter, no decontamination beyond what dedup incidentally removes. 100B tokens, easy to assemble.
- **Corpus B: filtered + decontaminated.** The same crawl, but URL/language filtered, MinHash *and* exact deduplicated, passed through a quality classifier (keep the top ~30%), and benchmark-decontaminated. Also 100B tokens, drawn from a larger filtered pool.

The figure's claim is the punchline before we see the numbers: with architecture, compute, and hyperparameters held identical, **the curated corpus yields a measurably stronger model.** Here is what the eval looks like at checkpoints through training, the kind of table you would paste into a run report.

| Metric (at tokens) | 10B | 30B | 60B | 100B |
| --- | --- | --- | --- | --- |
| **Model A** — val bits/byte | 0.93 | 0.89 | 0.87 | 0.86 |
| **Model A** — HellaSwag | 41 | 47 | 50 | 52 |
| **Model A** — ARC-Easy | 48 | 53 | 56 | 58 |
| **Model A** — MMLU (5-shot) | 25.6 | 26.1 | 26.8 | 27.1 |
| **Model B** — val bits/byte | 0.91 | 0.85 | 0.82 | 0.80 |
| **Model B** — HellaSwag | 44 | 53 | 58 | 61 |
| **Model B** — ARC-Easy | 52 | 61 | 66 | 69 |
| **Model B** — MMLU (5-shot) | 27.4 | 31.0 | 34.2 | 36.0 |

Read the curves, not just the endpoints. Three things are happening, and each one is a lesson:

**The gap widens over training.** At 10B tokens the two models are close — a couple of points apart. By 100B the gap is roughly 9 points on HellaSwag and 11 on ARC. *Better data does not give you a constant bonus; it gives you a steeper slope.* The cleaner corpus keeps teaching the model something new for longer, while the raw corpus runs out of signal and the curve flattens. **Model A plateaus early; Model B is still climbing** at the budget's end — which is exactly what the figure's red and green tracks encode.

**MMLU separates the men from the boys.** Model A's MMLU sits at ~27 the whole way — barely above the 25% random-chance floor for 4-way multiple choice. The raw corpus simply does not contain enough structured, knowledge-dense text for the model to learn the material. Model B climbs to 36 because the quality filter preferentially *kept* the knowledge-dense documents. A benchmark hugging the chance floor is almost never a model-capacity problem at this scale; it is a "your data does not contain this skill" problem.

**Token efficiency is the number that should scare you.** Model B reaches Model A's *final* HellaSwag score (52) at around **38B tokens** — well under half the budget. On the metrics where data matters most, the curated corpus is delivering ~2.5× the *effective compute*. Put differently: the team that filtered its data got a free 2.5× compute multiplier on those metrics, without buying a single extra GPU. That is the quantitative shape of the data-centric claim, and it is why this is the highest-ROI work in the building.

```python
# Token efficiency: at what token count does B match A's FINAL score?
# Linear-interpolate B's HellaSwag curve to find where it crosses A's endpoint (52).
b_tokens = [10e9, 30e9, 60e9, 100e9]
b_hellaswag = [44, 53, 58, 61]
a_final = 52

def crossing(xs, ys, target):
    for i in range(1, len(xs)):
        if ys[i-1] <= target <= ys[i]:
            t = (target - ys[i-1]) / (ys[i] - ys[i-1])
            return xs[i-1] + t * (xs[i] - xs[i-1])
    return None

match = crossing(b_tokens, b_hellaswag, a_final)
print(f"B matches A's final HellaSwag at ~{match/1e9:.0f}B tokens "
      f"=> {100e9/match:.1f}x effective compute")
# B matches A's final HellaSwag at ~38B tokens => 2.6x effective compute
```

This is not a contrived example. It is a compressed, numerically explicit version of what the FineWeb and phi results show at scale: hold the model fixed, improve the data, and the eval curve bends upward and keeps bending.

## 7. Telling a data problem from a model problem

Here is the skill that separates engineers who *believe* data matters from engineers who can *act* on it: when an eval number disappoints, can you localize the cause to the right place? Reach for a hyperparameter sweep when the problem is in the corpus and you will burn a week and a rack of GPUs learning nothing.

![Is it a data problem or a model problem?](/imgs/blogs/why-data-decides-the-model-7.webp)

The figure is a decision tree from *symptom* to *likely cause*, and its claim is deliberately provocative: **most disappointing-eval symptoms localize to a specific data defect; only a truly stuck loss points first at the model.** Walk the branches:

- **Train loss is low but eval is flat.** The model is fitting the training distribution beautifully and not generalizing. The usual culprit is **duplicates**: the model is memorizing repeated content, which drives training loss down without teaching anything transferable. The fix lives in the *dedup* stage, not in regularization.
- **A benchmark score is suspiciously high but the model fails on paraphrases of the same questions.** That is the signature of **contamination** — the test set, or something n-gram-close to it, is in your training data. The fix is *decontamination*, and the tell is that performance collapses the moment you perturb the question wording.
- **The model is good on web-style text but weak on reasoning or code.** This is a **mixture gap**: a domain is under-represented in the corpus. No architecture change will conjure a skill the data never contained. The fix is *reweighting domains* in the mix.
- **The loss curve drops in stair-steps** — flat, then a sudden cliff, then flat again. Those cliffs usually align with *epoch boundaries*: you are seeing the model recognize **data repeating**. The fix is to check how many times your corpus is being seen, and whether your "100B tokens" are really 20B tokens seen five times.
- **The loss will not drop at all** — *now* you have earned the right to suspect the model. A stuck loss from step one points at **the model or optimizer**: a learning rate that is too high or too low, a bad initialization, a broken data loader, or an honest architecture bug.

Notice the asymmetry the figure is built to convey: four of the five branches end in a *data* fix (red), and only one ends in a *model* fix (blue). That is not a quirk of this diagram; it is the base rate. When eval disappoints at a model that is otherwise training stably, the prior should be *data first*. The single most valuable habit you can build is the one in the companion post: [look at your data before you train](/blog/machine-learning/debugging-training/look-at-your-data-before-you-train). Sample 100 random documents and *read them* before you touch a hyperparameter.

```python
# The cheapest, highest-yield diagnostic in all of training: read your data.
import random, json

def sample_and_read(path: str, k: int = 100, seed: int = 0):
    """Print k random training documents so a human can actually look."""
    rng = random.Random(seed)
    with open(path) as f:
        docs = [json.loads(line)["text"] for line in f]
    for i, doc in enumerate(rng.sample(docs, k)):
        head = doc[:600].replace("\n", " ")
        print(f"\n--- doc {i} ({len(doc)} chars) ---\n{head}")

# You will find the bug here far more often than in your config:
# duplicated boilerplate, menu soup, base64 blobs, a single language
# you didn't expect, or — the worst — your eval set staring back at you.
```

## 8. Troubleshooting: when data masquerades as a model bug

The hardest data bugs are the ones wearing a model bug's costume. This section is a field guide: symptom, root cause, and fix, for the failures that hide at each stage of the pipeline.

![Where data bugs hide in the pipeline](/imgs/blogs/why-data-decides-the-model-8.webp)

The figure stacks the lifecycle stages and names the characteristic failure that lurks in each — because *every stage of the data pipeline harbors a distinct failure mode that masquerades later as a model problem.* The stack is your checklist; walk it top to bottom when a number looks wrong.

### Symptom: training loss is great, eval is mediocre, and the gap is huge

**Likely root cause — near-duplicate leakage (the *Dedup* stage).** Web crawls are full of near-identical documents. If they survive into training, the model memorizes them, which crushes training loss while doing nothing for generalization. Worse, if a near-duplicate of a *validation* document leaks into *training*, your val loss is also corrupted and you are flying blind.

**How you catch it.** Measure the duplicate rate directly, and measure train/val overlap explicitly. Lee et al. (2021), "Deduplicating Training Data Makes Language Models Better," found that some popular corpora contained large fractions of duplicated text and that removing it both reduced memorization and improved held-out perplexity.

```python
# Estimate near-duplicate rate with MinHash + LSH (datasketch).
from datasketch import MinHash, MinHashLSH

def minhash(text, num_perm=128, ngram=5):
    m = MinHash(num_perm=num_perm)
    tokens = text.split()
    for i in range(max(1, len(tokens) - ngram + 1)):
        m.update(" ".join(tokens[i:i+ngram]).encode())
    return m

def near_dup_rate(docs, threshold=0.8):
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    dups = 0
    for i, d in enumerate(docs):
        mh = minhash(d)
        if lsh.query(mh):       # an existing doc is >= threshold similar
            dups += 1
        else:
            lsh.insert(f"d{i}", mh)
    return dups / len(docs)

# A near-dup rate above ~10-15% means dedup is your highest-leverage fix.
```

**The fix.** Run exact dedup (hash) and near-dup dedup (MinHash/LSH) over the whole corpus, *including across the train/val split*, before training. If the gap closes after dedup, you found it.

### Symptom: a benchmark score that is too good to be true

**Likely root cause — benchmark contamination (the *Decontamination* stage).** Eval text leaked into training. The model is not solving the task; it is recalling the answer. This is the most common way teams accidentally lie to themselves, and it is endemic — GPT-3's own paper (Brown et al., 2020) included a contamination analysis precisely because n-gram overlap between web crawls and public benchmarks is unavoidable unless you actively remove it.

**How you catch it.** Run an n-gram overlap scan between your training corpus and every eval set, and watch for the *paraphrase cliff*: if the score collapses when you reword the questions, the model was matching strings, not reasoning.

```python
# Contamination check: does any eval n-gram appear verbatim in training?
def ngrams(text, n=13):
    toks = text.split()
    return {" ".join(toks[i:i+n]) for i in range(len(toks) - n + 1)}

def contamination_rate(eval_docs, train_ngram_index, n=13):
    hit = 0
    for doc in eval_docs:
        if ngrams(doc, n) & train_ngram_index:   # any 13-gram overlap
            hit += 1
    return hit / len(eval_docs)

# train_ngram_index is a precomputed set of 13-grams from the corpus.
# A nonzero contamination rate means your eval numbers are inflated.
```

**The fix.** Remove every training document with significant n-gram overlap with any benchmark *before* training, and keep a clean held-out canary set the corpus has provably never seen.

### Symptom: strong on prose, weak on a specific skill (code, math, multilingual)

**Likely root cause — a mixture gap (the *Mixing* stage), or a source gap (the *Collection* stage).** The skill is under-represented or absent. The model cannot learn what is not there in sufficient quantity, and no learning-rate change will fix a coverage problem.

**How you catch it.** Compute the actual domain proportions of your *final, post-filter* corpus — not the proportions you intended, the proportions you got. Filtering silently changes the mix: a quality classifier trained on prose will preferentially drop code, leaving you with *less* code than you put in.

**The fix.** Reweight the mixture to upsample the weak domain, or go back to collection and add a dedicated source. We devote a whole post to this in the series.

### Symptom: eval regresses randomly between otherwise-identical runs

**Likely root cause — a toxic slice (the *Extraction* / *Clean* stages).** A small fraction of garbled documents — encoding errors, HTML soup, repeated-token spam, base64 blobs — can destabilize training and inject noise that shows up as run-to-run variance.

**How you catch it.** Apply document-level quality heuristics and look at the *outliers*: documents with extreme symbol-to-word ratios, very short or very long mean line length, low type-token ratio (a sign of repetition), or high fractions of non-alphabetic characters.

```python
# Cheap document-quality heuristics — flag the toxic tail for inspection.
def doc_flags(text):
    toks = text.split()
    if not toks:
        return {"empty": True}
    symbols = sum(c in "{}[]<>|\\^~`" for c in text)
    return {
        "symbol_ratio": symbols / max(1, len(text)),       # high => HTML/code soup
        "ttr": len(set(toks)) / len(toks),                 # low  => repetitive spam
        "mean_word_len": sum(map(len, toks)) / len(toks),  # extreme => garbled
        "alpha_ratio": sum(c.isalpha() for c in text) / max(1, len(text)),
    }

# Sort by symbol_ratio descending and read the top 50. The bug is usually visible.
```

**The fix.** Add the failing heuristic as a filter in the *Clean* stage and re-run. The discipline is the same every time: the fix for a data problem is *always* to fix the data, never to paper over it with a model-side knob.

> When the eval surprises you, the model is rarely lying. The data is. Walk back up the pipeline until you find the stage that lied.

## Case studies from production

Named incidents, with specifics, because the patterns above are easier to internalize as stories.

### 1. FineWeb: the ablation that settled the argument

The FineWeb team (Penedo et al., 2024) did not assert that their data was better — they *proved* it with the equal-token ablation, then open-sourced the entire 15T-token corpus, the `datatrove` pipeline, and the ablation harness. The methodology is the lesson: a fixed ~1.7B model, a fixed token budget, and a swap of *only* the corpus, evaluated on a curated benchmark suite chosen for low run-to-run variance (so that small, real deltas were not drowned in noise). FineWeb beat C4, RefinedWeb, Dolma, RedPajama-v2, and The Pile on the aggregate. The decisive design choice was making the comparison *causal* by freezing everything else — the same move as our worked scenario, run at scale and published. When someone tells you data matters, this is the experiment to ask them for.

### 2. FineWeb-Edu: one filter, a step-change on knowledge

The FineWeb-Edu variant isolated the *quality filter* as a single intervention. The team prompted a strong model to rate documents for educational value, distilled those ratings into a small, cheap classifier, and kept only high-scoring documents. The result was a pronounced jump on MMLU and ARC at a fraction of the token count — the cleanest public demonstration that aggressive quality filtering trades quantity for capability and *wins* on knowledge benchmarks. It is also a cautionary tale for the mixing stage: an educational-value filter is implicitly a domain filter, and you have to re-measure your mixture after applying it.

### 3. phi-1: textbooks beat tonnage

phi-1 (Gunasekar et al., 2023) is the existence proof for the extreme end of the quality axis. 1.3B parameters, ~7B tokens, a corpus of filtered code plus synthetic "textbook" content and exercises — and **50.6% pass@1 on HumanEval**, outscoring code models many times larger trained on vastly more data. The follow-ups (phi-1.5, and the later phi line) generalized the recipe beyond code. The takeaway is not "use synthetic data for everything"; it is that the *information density* of the corpus can dominate its size by orders of magnitude. When your tokens are curriculum-grade, you need far fewer of them.

### 4. The deduplication paper: a free quality upgrade

Lee et al. (2021), "Deduplicating Training Data Makes Language Models Better," quantified what practitioners suspected: standard corpora contain large amounts of duplicated text, and removing it (a) reduces verbatim memorization, (b) reduces train-test overlap that inflates eval, and (c) improves held-out perplexity — all while *shrinking* the dataset and saving compute. This is the rare intervention that improves quality and reduces cost simultaneously, which is exactly why dedup is the first heavy stage every serious pipeline runs.

### 5. GPT-3 and the contamination disclosure

The GPT-3 paper (Brown et al., 2020) is notable for *publishing its own contamination analysis*: the authors measured n-gram overlap between the training corpus and benchmark test sets, reported which benchmarks were affected, and re-scored on cleaned subsets. The lesson is professional honesty as much as technique — at web scale, *some* contamination is the default, and the responsible move is to measure and disclose it rather than quietly enjoy the inflated numbers. Every decontamination stage downstream traces its lineage to this kind of analysis.

### 6. LLaMA and Dolma: the recipe is the result

LLaMA (Touvron et al., 2023) demonstrated that a carefully filtered mixture of *entirely public* data — CommonCrawl via CCNet, C4, GitHub, Wikipedia, books, arXiv, StackExchange, in deliberate proportions — could produce models competitive with closed ones. AllenAI's Dolma open-sourced a 3T-token corpus *and* the full toolkit and documentation of how it was filtered and mixed. Both make the same point: the differentiated artifact is the *data pipeline and mixture*, and publishing it (LLaMA published the recipe; Dolma published the tooling) is how the field's data-centric knowledge compounds. The model weights are the easy part to reproduce; the corpus is the hard part.

## When to reach for a data-centric push (and when not to)

Data work is high-leverage, not infinite-leverage. Spend your effort where it pays.

**Reach for a data-centric push when:**

- Your architecture is standard and stable, and you have stopped finding real gains from model changes. (This is the common case.)
- A benchmark is hugging the random-chance floor — that is a coverage/quality problem, almost never a capacity problem.
- Train loss is healthy but eval is flat, or the train/val gap is large — classic dedup/contamination signatures.
- You are compute-constrained and want more effective compute without more GPUs — better data is the cheapest multiplier available.
- You are building something durable and want a *moat*: the data pipeline is far harder for a competitor to copy than the architecture.

**Skip it (or deprioritize) when:**

- Your loss will not drop from step one — fix the model, optimizer, or data loader *first*; a broken run cannot tell you anything about data quality.
- You have not yet *looked at your data* — running an expensive filtering pipeline before reading 100 documents is optimizing blind.
- The corpus is already aggressively filtered and your real gap is model scale — sometimes you genuinely do need a bigger model and more tokens, and no amount of polishing a small clean corpus substitutes for that.
- You are in a tiny-data, transfer-learning regime where a strong pretrained base already encodes most of what you need — there the leverage is in fine-tuning data, not pretraining corpora.

The throughline of this series is that the corpus is a first-class engineering artifact — designed, measured, version-controlled, and debugged with the same rigor you bring to code. The next posts make each stage of the funnel concrete: how to set [data scaling laws and budgets](/blog/machine-learning/training-data/data-scaling-laws-and-budgets) for your model size, and how to turn "quality" from a vibe into a number in [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality). Freeze the model. Iterate on the data. That is where the wins are.

## Further reading

- Penedo et al., "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale" (2024) — the equal-token ablation, the `datatrove` pipeline, and FineWeb-Edu.
- Gunasekar et al., "Textbooks Are All You Need" (2023) — phi-1 and the information-density argument.
- Lee et al., "Deduplicating Training Data Makes Language Models Better" (2021) — why dedup is the first heavy stage.
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (2022) — the Chinchilla result that makes tokens precious; see also [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling).
- On this blog: [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) and [look at your data before you train](/blog/machine-learning/debugging-training/look-at-your-data-before-you-train).
