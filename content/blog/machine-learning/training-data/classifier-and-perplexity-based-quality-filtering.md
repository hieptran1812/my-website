---
title: "Classifier and Perplexity Quality Filtering: Teaching a Model to Pick Your Data"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Heuristics catch obvious garbage; model-based filters catch what quality actually looks like. A principal-engineer's guide to KenLM perplexity bucketing, fastText quality classifiers, the FineWeb-Edu distillation trick, LLM-as-judge scoring, calibration, and the cost ladder that makes it all affordable."
tags: ["training-data", "data-quality", "quality-filtering", "perplexity", "kenlm", "fasttext", "classifier", "fineweb-edu", "ccnet", "llm-as-judge", "data-curation", "pretraining-data", "ai-safety"]
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 35
---

Every serious pretraining pipeline eventually hits the same wall. You have written the heuristic filters — the ones that drop documents by length, by symbol ratio, by the fraction of lines ending in a bullet, by whether the language detector is confident it is English. They cleared out the obvious sewage: the SEO doorway pages, the Lorem Ipsum, the pages that are 90% navigation chrome. And the model got better. Then you tighten the thresholds a little more, and the model stops getting better, and you realize the rules have run out of road. The next ten points of quality are not hiding behind a symbol-ratio cutoff. They are hiding in the difference between a document that *reads like something worth learning from* and one that does not, and no regex will ever draw that line.

That is the moment you reach for model-based filtering. Instead of writing rules about what bad data looks like, you train a model to recognize what good data looks like — and then you let that model score your entire corpus. There are three broad ways to do it, and this post is a tour of all three: **perplexity filtering** against a reference language model (score every document by how surprised a clean model is to read it), **classifier filtering** (train a cheap classifier to tell your curated data from raw web), and **LLM-as-judge** scoring (ask a capable model to grade documents directly, then distill that judgment into something you can afford to run at scale). The figure below is the mental model for the whole space — four methods, arranged by what signal they capture and what they cost.

![Four ways to filter data by quality, arranged by cost and signal — from cheap surface heuristics through perplexity and classifiers up to an expensive LLM judge](/imgs/blogs/classifier-and-perplexity-based-quality-filtering-1.webp)

Read that table as a ladder, not a menu. You do not pick one row and stop. The cheap methods run on everything; the expensive methods run on a sample and teach the cheap ones what to look for. The single most important idea in this post — the one that separates a filtering pipeline that ships from one that bankrupts your compute budget — is that **the expensive judgment and the cheap execution are different jobs, and you distill the former into the latter.** FineWeb-Edu, the case study we keep returning to, is exactly this trick: a 70-billion-parameter model labels a few hundred thousand documents, a tiny classifier learns to mimic those labels, and the tiny classifier filters fifteen trillion tokens. The result was one of the largest quality jumps the open pretraining community has seen.

This post assumes you have already read [Measuring Data Quality](/blog/machine-learning/training-data/measuring-data-quality) — because everything here is downstream of being able to *tell whether a filtering change actually helped* — and [Language Identification and Heuristic Quality Filters](/blog/machine-learning/training-data/language-identification-and-heuristic-quality-filters), which covers the rules-based layer that runs before any of this. If perplexity and classifiers are the second story of the building, those two posts are the foundation.

## Why model-based filtering is different from what you expect

The trap most engineers fall into on their first quality-filter project is treating it like spam classification. It is not. Spam classification has a stable, adversarial-but-labeled target: an email is spam or it is not, humans agree most of the time, and you can measure accuracy against ground truth. Pretraining quality filtering has none of that. There is no ground-truth label for "is this document good training data." The only real definition of "good" is *does including it make the trained model better on the things you care about* — and that is a signal you can only measure through the ablation loop, expensively and noisily.

| Assumption | The naive view | The reality |
| --- | --- | --- |
| "Quality is a property of the document" | Score each doc in isolation, keep the high scorers | Quality is relative to your target distribution and your model's needs; the same doc can be a keep or a drop depending on the mix |
| "The classifier learns quality" | Train on good-vs-bad, trust the probability | The classifier learns *your label choice* — it will faithfully reproduce whatever bias is in your positive set |
| "Higher threshold, better data" | Keep only the top few percent | Aggressive thresholds shrink the corpus and delete diversity; past a point you trade tokens for a quality gain that no longer helps |
| "Perplexity measures quality" | Low perplexity = clean = good | Perplexity measures *fluency relative to a reference*; boilerplate and repetition score beautifully, and technical prose scores badly |
| "One filter is enough" | Pick the best method, apply it | Real pipelines stack a cheap filter on everything and an expensive one on the residual; no single method wins on cost *and* signal |

Hold onto the second and fourth rows. They are the two failure modes that cost teams the most, and both get their own section later. A perplexity filter that quietly prefers repetitive boilerplate, and a classifier that quietly deletes your best technical and code documents because they do not look like Wikipedia — these are not edge cases. They are the default behavior of the naive implementation, and you will ship them unless you go looking.

One more framing before we get into mechanics. Model-based filtering is a **proxy**. The classifier's score is a proxy for "resembles my positive set," which is itself a proxy for "good training data." Every proxy invites Goodhart's law: optimize the proxy hard enough and it stops tracking the thing you actually wanted. The whole discipline of quality filtering is managing that gap — choosing proxies that correlate well, setting thresholds that do not overfit the proxy, and validating against the one measurement that is not a proxy: a downstream eval on a model trained with and without the filter.

## 1. Perplexity filtering against a reference model: the CCNet approach

The oldest and cheapest model-based filter is **perplexity scoring**, and the canonical implementation is CCNet, the pipeline Facebook AI Research built to extract usable multilingual text from Common Crawl. The idea is disarmingly simple. Train a small n-gram language model on a clean, high-quality target — CCNet used Wikipedia in each language — then score every candidate document by its perplexity under that model. Perplexity is the exponentiated average negative log-likelihood the reference model assigns to the document:

$$\text{PPL}(d) = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N} \ln p(w_i \mid w_{i-1}, \dots, w_{i-n+1})\right)$$

where $d$ is the document, $w_1 \dots w_N$ its tokens, and $p$ the reference model's probability. Low perplexity means the reference model found the document unsurprising — it looks like the clean target it was trained on. High perplexity means the reference model was constantly surprised — the document is full of tokens and sequences that never appear in clean Wikipedia-like text: markup fragments, keyword spam, garbled encoding, machine-translation sludge.

CCNet does not use a single threshold. It ranks every document by perplexity and cuts the distribution into three buckets — head, middle, and tail — as shown below.

![CCNet perplexity bucketing: a KenLM trained on Wikipedia scores every document, splitting the corpus into a low-perplexity head to keep, a mid-perplexity middle to blend, and a high-perplexity tail to drop](/imgs/blogs/classifier-and-perplexity-based-quality-filtering-2.webp)

The head — the lowest-perplexity third — is the fluent, clean material you keep for a quality tier. The tail — the highest-perplexity third — is where the spam and garble concentrate, and you drop or heavily downweight it. The middle is ordinary usable web text; you keep it and blend it into the mix. The exact percentile cuts are a knob you tune per language and per corpus, but the three-bucket structure is the durable idea: perplexity gives you a *ranking*, and you decide where to cut based on how much data you need and how clean you need it.

Here is what perplexity scoring actually looks like in code, using KenLM — the fast n-gram library everyone uses for this — trained on a SentencePiece-tokenized Wikipedia dump.

```python
import kenlm
import sentencepiece as spm

# Prerequisites (run once, offline):
#   spm_train --input=wiki.txt --model_prefix=wiki_spm --vocab_size=32000
#   spm_encode --model=wiki_spm.model < wiki.txt > wiki.tok
#   lmplz -o 5 < wiki.tok > wiki.arpa        # 5-gram model
#   build_binary wiki.arpa wiki.5gram.bin    # fast binary format

lm = kenlm.Model("wiki.5gram.bin")
sp = spm.SentencePieceProcessor(model_file="wiki_spm.model")

def doc_perplexity(text: str) -> float:
    """Perplexity of a document under the Wikipedia reference LM."""
    tokens = " ".join(sp.encode(text, out_type=str))
    # KenLM returns log10 of the sentence probability, with BOS/EOS.
    log10_prob = lm.score(tokens, bos=True, eos=True)
    n_tokens = len(tokens.split()) + 1          # +1 for the EOS token
    # log10 -> natural log per token, then exponentiate.
    return 10.0 ** (-log10_prob / n_tokens)

# CCNet-style bucketing: rank the whole shard, cut into thirds.
scored = sorted(((doc_id, doc_perplexity(t)) for doc_id, t in corpus),
                key=lambda x: x[1])
n = len(scored)
head   = scored[: n // 3]              # low perplexity  -> quality tier
middle = scored[n // 3 : 2 * n // 3]   # mid perplexity  -> keep / blend
tail   = scored[2 * n // 3 :]          # high perplexity -> drop / downweight
```

**The senior rule of thumb for perplexity filtering: it is a fluency filter, not a quality filter, and you must never forget the difference.** Perplexity rewards text that statistically resembles your reference. That is exactly right for catching garble and spam, and exactly wrong for a lot of legitimately valuable data. A dense mathematics paper, a Python stack trace, a transcript of casual spoken dialogue, a legal contract — all of these have high perplexity against a Wikipedia reference, and all of them may be data you desperately want. This is the first place the method quietly betrays you, and we return to it in the troubleshooting section.

The second-order gotcha is more insidious: **perplexity loves repetition.** A document that says "buy now buy now buy now" a thousand times has *extremely* low perplexity, because after the reference model sees "buy now" a few times it predicts the next "buy now" with near certainty. Boilerplate, repeated headers, and templated spam all score as beautifully clean. If you filter on perplexity alone you will happily keep a warehouse of repetitive junk while dropping a brilliant, information-dense essay that used three words the reference model had never seen in that order. The fix is to run perplexity *after* your deduplication and repetition heuristics, never instead of them — a topic covered in the sibling post on [text extraction and boilerplate removal](/blog/machine-learning/training-data/text-extraction-and-boilerplate-removal).

Why use KenLM and not a neural model for the reference? Cost. A 5-gram KenLM scores documents at hundreds of thousands of tokens per second per core, on CPU, with no GPU in sight. CCNet was designed to process *all* of Common Crawl in dozens of languages; a neural reference model would have made that economically impossible. The lesson generalizes: the reference model in a perplexity filter should be the cheapest thing that captures fluency, because you are going to run it on every token you own.

## 2. Training a quality classifier: the "does this look like the good stuff?" filter

Perplexity asks "is this fluent?" A quality classifier asks a sharper question: "does this look like the specific kind of high-quality data I curated?" This is the approach GPT-3 used to filter Common Crawl, and it is still the workhorse of large-scale web curation. You assemble a **positive set** of documents you consider high quality — GPT-3 used pages referenced from Reddit with enough karma, plus WebText, plus books and Wikipedia — and a **negative set** of unfiltered web pages. Then you train a fast binary classifier to tell them apart, and you score your whole corpus by its probability of belonging to the positive class.

![Training a quality classifier: a curated positive set and a raw-web negative set are featurized and used to train a fastText or logistic-regression scorer that emits P(good) for every document in the full corpus](/imgs/blogs/classifier-and-perplexity-based-quality-filtering-3.webp)

The classifier only ever learns one thing: to separate your positives from your negatives. That is worth saying twice, because it is the source of both the method's power and its central danger. **The classifier's bias is exactly your label choice.** If your positive set is Wikipedia and books, the classifier learns "formal, edited, expository prose = good" — and it will faithfully downrank forum discussions, code, transcripts, and anything conversational, because those are not what you showed it. Choose your positive set as if it were the definition of quality, because to the classifier, it is.

The standard tool is fastText — a linear classifier over word and character n-gram embeddings that trains in minutes on a few hundred thousand documents and scores at enormous throughput. Here is the full loop.

```python
import fasttext

# Build train.ft with one document per line:
#   __label__good  <normalized document text on a single line>
#   __label__web   <normalized document text on a single line>
# Positives: curated set (Wikipedia refs, books, edu pages, high-karma links).
# Negatives: a random sample of raw Common Crawl of similar size.

model = fasttext.train_supervised(
    input="train.ft",
    lr=0.5,
    epoch=5,
    wordNgrams=2,      # bigrams capture a little local style
    dim=100,
    minCount=3,
    loss="softmax",
)
model.save_model("quality.ftz")

def quality_score(text: str) -> float:
    """P(document belongs to the 'good' class), in [0, 1]."""
    text = text.replace("\n", " ").strip()
    labels, probs = model.predict(text, k=2)   # both classes
    scores = dict(zip(labels, probs))
    return scores.get("__label__good", 0.0)

# Score the corpus and keep by a threshold chosen via ablation (Section 6).
kept = [doc for doc in corpus if quality_score(doc.text) >= 0.5]
```

Three engineering notes that matter more than they look. First, **normalize the text identically for positives, negatives, and the corpus you score** — lowercasing, whitespace collapse, URL stripping. A classifier will happily learn "documents with `\n` characters are negatives" if your positive set happened to be pre-cleaned and your negatives were not. That is a spurious feature, and it will wreck your filter. Second, **balance the classes** — roughly equal positive and negative counts, or the classifier's probabilities become uncalibrated and your threshold stops meaning anything. Third, **use character n-grams (fastText does this by default via `minn`/`maxn`)** so the classifier is robust to typos and rare words rather than memorizing a vocabulary.

The second-order optimization here is about the *negative* set, which teams routinely neglect. If your negatives are pure random Common Crawl, the classifier learns an easy task — telling curated prose from raw sludge — and its scores collapse into a bimodal "obviously good / obviously bad" split that gives you no resolution in the middle, where all the interesting decisions live. A sharper approach is **hard negatives**: seed the negative set with web pages that already passed your heuristic filters, so the classifier has to learn the finer distinction between "passed the rules but still mediocre" and "genuinely good." That is the difference between a classifier that reproduces your heuristics and one that adds signal beyond them.

## 3. FineWeb-Edu: distilling an LLM judge into a filter you can afford

Everything so far has a ceiling. Perplexity captures fluency; a Wikipedia-vs-web classifier captures "resembles curated prose." Neither captures *educational value* — whether a document actually teaches something, explains a concept, presents a worked example, reasons through a problem. That is a semantic judgment, and until recently the only thing that could make it was a human or a very capable language model. Both are far too expensive to run on trillions of tokens.

FineWeb-Edu, from the Hugging Face team, is the pipeline that broke that impasse, and it is the single most instructive case study in this entire post. The trick is distillation, and it runs in four stages.

![The FineWeb-Edu distillation pipeline: sample a few hundred thousand documents from raw web, score each 0-5 for educational value with a large LLM, train a small classifier on those labels, then score and threshold the full corpus cheaply](/imgs/blogs/classifier-and-perplexity-based-quality-filtering-4.webp)

Stage one: **sample.** Draw a few hundred thousand documents from the raw FineWeb corpus — the actual FineWeb-Edu work annotated around 460,000. Stage two: **annotate with a big model.** Prompt a large instruction-tuned LLM (the original used Llama-3-70B-Instruct) to score each sampled document from 0 to 5 on an educational-value rubric: 0 for pure navigation or spam, rising through "has some educational content but is cluttered" to 5 for a clear, self-contained, textbook-quality explanation. Stage three: **distill.** Train a small classifier — the FineWeb-Edu team put a lightweight regression head on a BERT-style encoder — to predict the LLM's 0-to-5 score from the raw text. Stage four: **filter.** Run that small, fast classifier over the *entire* corpus, and keep documents scoring at or above a threshold (they used a score of 3).

The economics are the whole point. Running Llama-3-70B over fifteen trillion tokens is a fantasy — it would cost more than training the model you are trying to feed. Running it over 460,000 documents is a weekend job on a modest GPU cluster. The small distilled classifier costs almost nothing per document, so it scales to the full corpus. You pay for expensive judgment *once, on a sample*, and amortize it across the whole dataset through the cheap student model.

The results made the whole field pay attention. A model trained on FineWeb-Edu matched or beat models trained on several times more tokens of unfiltered web data, with the gains concentrated exactly where educational value should help: knowledge-and-reasoning benchmarks like MMLU and ARC. The reference numbers the team published showed the edu-filtered data reaching MMLU and ARC scores at a fraction of the token budget that raw FineWeb needed to approach — an efficiency multiple, not a rounding-error improvement. When people say "data quality is a scaling-law lever," this is the concrete demonstration; the relationship is explored further in [Data Quality Scaling Laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws).

A few design decisions inside this pipeline are worth stealing:

- **Regression, not classification, for the annotation.** Asking the big model for a 0-to-5 score rather than a binary good/bad gives the distilled model a richer target and lets you move the keep-threshold *after* training without re-annotating. Keeping score 3 and above is one policy; keeping 4 and above is another, and you can ablate both from a single labeled set.
- **A rubric with anchored examples.** The annotation prompt does not just say "rate educational value." It defines what each score means, with concrete descriptions of what a 1 versus a 4 looks like. Vague rubrics produce noisy labels, and noisy labels produce a weak student. This is the same discipline as writing a good grading rubric for [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) — the rubric *is* the definition of quality, so write it carefully.
- **A small, fast student.** The distilled classifier has to run on the whole corpus, so it must be cheap. An encoder with a regression head hits the sweet spot: far more semantically aware than fastText, far cheaper than the teacher.

The one caution — and it is the same caution as every classifier — is that the distilled model inherits the teacher's biases plus the sample's biases. An edu classifier trained to reward textbook-style explanations will downrank code, casual dialogue, and non-Western-academic prose unless those are represented in the sampled-and-scored set. FineWeb-Edu is spectacular for knowledge benchmarks; it is not the filter you would use unmodified to build a coding model. Match the teacher's rubric to the capabilities you are trying to build.

## 4. LLM-as-judge and Ask-LLM: scoring by meaning

FineWeb-Edu uses an LLM as a *teacher* to label a sample. But you can also use an LLM directly as the *scorer* — an LLM-as-judge — when the sample you need to score is small enough to afford it. This is the top rung of the quality ladder: the most semantically sensitive signal available, and the most expensive per document.

The mechanics are a single prompt with a rubric that returns a structured score. Because the task involves classifying natural-language documents and the provider is up to you, here is a concrete, runnable version using the Anthropic SDK with a cheap model — Haiku 4.5 is the right tier for a bulk judge where you want semantic sensitivity without Opus pricing.

```python
import json
import anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment

EDU_RUBRIC = """You are grading a web document for its value as pretraining data
for a language model. Score its educational value from 0 to 5:

0 = no educational content (nav bars, ads, link farms, keyword spam)
1 = mostly non-educational, a stray useful sentence
2 = some educational content but cluttered, shallow, or off-topic
3 = clearly educational: explains a concept or presents useful information
4 = high quality: coherent, self-contained, well-structured explanation
5 = outstanding: textbook- or tutorial-grade, with examples or reasoning

Return ONLY a JSON object: {"score": <int 0-5>, "reason": "<one short sentence>"}"""

def edu_score(document: str) -> dict:
    msg = client.messages.create(
        model="claude-haiku-4-5",       # cheap tier for a bulk judge
        max_tokens=100,
        system=EDU_RUBRIC,
        messages=[{"role": "user", "content": document[:6000]}],  # cap the prompt
    )
    return json.loads(msg.content[0].text)

# Score a small held-out sample; use the labels to train/validate a cheap filter.
labels = [(doc.id, edu_score(doc.text)) for doc in sample]
```

Two production notes. First, **truncate the document you send** — the first several thousand characters carry almost all of the quality signal, and paying to send a 50KB page to the judge is waste. Second, for the *labeling* pass that feeds a distilled classifier (the FineWeb-Edu pattern), you generally want a stronger model than Haiku — a Sonnet- or Opus-tier judge produces cleaner labels, and since you only run it on the sample, the higher per-document cost is affordable. Use the cheap tier when the LLM-judge score is the *final* signal on a small set; use the strong tier when its output becomes *training labels* that a million documents will inherit.

A useful variant is **Ask-LLM**, which reframes the judgment as a question the model answers with its own probabilities rather than a free-form score: prompt the model with "Does this document contain information useful for learning? Answer yes or no," and use the probability it assigns to "yes" as the quality score. This is cheaper and more calibrated than parsing a free-text rating, and it turns the judge into something closer to a soft classifier — a continuous score you can threshold, rather than a discrete rating you have to bucket. The published Ask-LLM work found that scoring with a capable model this way could select a small fraction of a corpus that trained a *better* model than the full corpus, which is the same "quality beats quantity" result FineWeb-Edu showed, arrived at from a different direction.

The reason you cannot just run LLM-as-judge on everything and skip the classifier entirely is throughput and money. A judge call is milliseconds-to-seconds and cents-per-thousand-documents; a corpus is trillions of tokens. Do the arithmetic once and it is obvious why every large-scale pipeline distills: the LLM judge is the teacher that defines quality, and a cheap classifier is the only thing that can afford to apply that definition at scale. Which brings us to the practical question every one of these methods eventually forces — where, exactly, do you draw the line?

## 5. A worked scenario: build a keep-filter and choose its threshold

Let me make this concrete with a scenario you could run on a laptop and a modest GPU box, with the kind of numbers you should actually expect. The goal: filter a 100-million-document raw-web shard down to a high-quality keep set, using a fastText quality classifier, and choose the keep-threshold with a small ablation rather than a guess.

**Step one — assemble the training sets.** Take 200,000 documents from a curated positive source (a mix of Wikipedia references, an open books corpus, and a set of pages a strong LLM judge already scored 4 or 5 for educational value) and 200,000 random documents from the raw shard as negatives. Normalize both identically. Train the fastText classifier exactly as in Section 3 — on 400,000 documents this takes a couple of minutes on CPU.

**Step two — score a held-out sample.** Score 50,000 held-out documents (not in the training set) and look at the distribution of `P(good)`. In a healthy run it is roughly bimodal: a hump of clearly-bad documents near 0, a hump of clearly-good near 1, and a spread in between where the real decisions live. The animated figure below is that distribution, with the keep-threshold as a line you slide left and right — everything to its right is kept.

<figure class="blog-anim">
<svg viewBox="0 0 720 410" role="img" aria-label="A histogram of quality scores from low on the left to high on the right; a vertical keep-threshold line sweeps left and right, and the green keep region to its right grows and shrinks as the threshold moves" style="width:100%;height:auto;max-width:820px">
<style>
.qf-bar{fill:var(--surface,#eef0f3);stroke:var(--border,#c7ccd3);stroke-width:1.5}
.qf-keep{fill:var(--accent,#22a06b);opacity:.16}
.qf-line{stroke:var(--accent,#22a06b);stroke-width:3}
.qf-axis{stroke:var(--text-secondary,#6b7280);stroke-width:2}
.qf-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.qf-sub{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.qf-mv{font:700 14px ui-sans-serif,system-ui;fill:var(--accent,#22a06b);text-anchor:middle}
@keyframes qf-sweep{0%{transform:translateX(-165px)}100%{transform:translateX(150px)}}
.qf-sweeper{animation:qf-sweep 9s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.qf-sweeper{animation:none}}
</style>
<text class="qf-lbl" x="360" y="26">Quality-score distribution and the keep-threshold</text>
<rect class="qf-bar" x="90"  y="270" width="28" height="30"/>
<rect class="qf-bar" x="122" y="230" width="28" height="70"/>
<rect class="qf-bar" x="154" y="180" width="28" height="120"/>
<rect class="qf-bar" x="186" y="150" width="28" height="150"/>
<rect class="qf-bar" x="218" y="180" width="28" height="120"/>
<rect class="qf-bar" x="250" y="230" width="28" height="70"/>
<rect class="qf-bar" x="282" y="260" width="28" height="40"/>
<rect class="qf-bar" x="314" y="275" width="28" height="25"/>
<rect class="qf-bar" x="346" y="275" width="28" height="25"/>
<rect class="qf-bar" x="378" y="255" width="28" height="45"/>
<rect class="qf-bar" x="410" y="210" width="28" height="90"/>
<rect class="qf-bar" x="442" y="160" width="28" height="140"/>
<rect class="qf-bar" x="474" y="140" width="28" height="160"/>
<rect class="qf-bar" x="506" y="170" width="28" height="130"/>
<rect class="qf-bar" x="538" y="220" width="28" height="80"/>
<rect class="qf-bar" x="570" y="260" width="28" height="40"/>
<line class="qf-axis" x1="80" y1="300" x2="610" y2="300"/>
<text class="qf-sub" x="185" y="332">low score (web-like)</text>
<text class="qf-sub" x="490" y="332">high score (edu-like)</text>
<text class="qf-sub" x="200" y="356">DROP as threshold rises</text>
<text class="qf-sub" x="500" y="356">KEEP region (right of line)</text>
<text class="qf-sub" x="360" y="392">Raise the line: precision up, tokens down. Lower it: recall up, noise up.</text>
<g class="qf-sweeper">
<rect class="qf-keep" x="330" y="120" width="390" height="180"/>
<line class="qf-line" x1="330" y1="112" x2="330" y2="300"/>
<text class="qf-mv" x="330" y="104">keep threshold</text>
</g>
<figcaption>The keep-threshold sweeps across the score axis; everything to its right is kept. Raising it trades corpus size for average quality, lowering it does the reverse.</figcaption>
</figure>

**Step three — run the ablation.** This is the step teams skip, and it is the step that matters. Pick three or four candidate thresholds, keep the documents above each, and for each keep set, hand-audit a random sample of 200 kept documents to estimate precision (what fraction are genuinely good) and record the keep-rate (what fraction of the corpus survives). Then — the part that is not optional — train a small proxy model on each keep set at equal token budget and measure a downstream eval. The table below is a realistic result shape.

| Threshold | Keep-rate | Est. precision (audit of 200) | Proxy MMLU delta vs. raw |
| --- | --- | --- | --- |
| 0.30 | 62% | ~71% | +0.6 pts |
| 0.50 | 38% | ~84% | +1.4 pts |
| 0.70 | 19% | ~93% | +1.7 pts |
| 0.85 | 8% | ~97% | +1.1 pts |

Read that table the way a principal engineer reads it. Precision climbs monotonically with the threshold — no surprise, you are keeping only the classifier's most confident picks. But the downstream eval *peaks in the middle and then falls*. At 0.85 the kept data is almost pure, and the model does worse than at 0.70, because you have thrown away so much data that the model is now token-starved and has lost distributional diversity. The right threshold is 0.70 here — not the one that maximizes precision, and not the one that maximizes keep-rate, but the one that maximizes the only number that matters, the downstream eval. **The threshold is not a property of the classifier; it is the output of an ablation.** Anyone who hands you a filter with a threshold and no ablation behind it is guessing.

Notice also that the gap between 0.50 and 0.70 on the eval is small (+1.4 vs +1.7), while the keep-rate halves (38% to 19%). If tokens are scarce for you — if you are training a large model and every token counts — 0.50 may be the better operating point even though 0.70 scores marginally higher, because it gives you twice the data at nearly the same quality. This is exactly the quality-versus-quantity tension quantified in [data scaling laws and budgets](/blog/machine-learning/training-data/data-scaling-laws-and-budgets); the classifier gives you the knob, the scaling law tells you where to set it.

## 6. Calibration, thresholds, and Goodhart's law

A raw classifier score is not a probability, and treating it like one is a common and expensive mistake. When fastText tells you a document is `P(good) = 0.8`, that does not mean 80% of documents with that score are good. The number is an uncalibrated output of a linear model, and its relationship to true quality depends entirely on how you built the training set. If you want the score to *mean* something — to say "keep everything with at least a 70% chance of being genuinely good" — you have to calibrate it, by fitting a simple monotonic map (Platt scaling or isotonic regression) from raw scores to audited-precision on a held-out labeled set.

But here is the honest truth about calibration for filtering: **most of the time you do not need calibrated probabilities, you need a well-chosen threshold, and those are different problems.** A filter does not care whether 0.8 means "80% good"; it cares where to cut so that the downstream model improves the most. Calibration matters when you want to *combine* signals — say, blend a perplexity score and a classifier score into one keep decision — because you cannot average two numbers that live on incomparable scales. It also matters when you want to set a threshold on one shard and have it mean the same thing on another shard with a different quality mix. For a single classifier on a single corpus, the ablation-chosen threshold from Section 5 is what you ship; calibration is polish.

The deeper issue lurking under every threshold decision is **Goodhart's law**: when a measure becomes a target, it ceases to be a good measure. Your classifier score is a proxy for quality. The moment you filter aggressively on it, you are optimizing the proxy, and the proxy starts to diverge from the real thing in three predictable ways:

- **The classifier's blind spots become your corpus's blind spots.** If the classifier under-scores code because your positives had little code, filtering hard on the classifier deletes code, and now your model cannot code. You did not decide to remove code; the proxy did, silently.
- **You overfit the threshold to the proxy, not the goal.** If you tune the threshold to maximize the classifier's average kept-score, you will drive it as high as it goes and starve the model. The threshold must be tuned against the downstream eval, which is *not* a proxy — it is the actual objective. This is why Section 5's table has a "proxy MMLU delta" column and not just a "precision" column.
- **Adversarial and distribution drift.** A classifier trained on today's web will slowly mismatch tomorrow's web as content shifts (more machine-generated text, new formats, new spam). A filter is not a fixed asset; it decays, and it needs periodic re-annotation against fresh samples.

The discipline that keeps Goodhart at bay is simple to state and hard to practice: **validate the filter against the objective, not against itself.** The classifier's precision, its ROC curve, its cross-validation accuracy — all of these measure how well it reproduces your labels. None of them measure whether the filtered data trains a better model. Only the ablation does. Every other number is a diagnostic; the ablation is the verdict.

## 7. Cost control: the filtering cost ladder

If you tried to run an LLM judge on every document in a fifteen-trillion-token corpus, you would spend more than the cost of training the model several times over. If you ran only heuristics, you would leave the biggest quality gains on the table. The resolution is a **cost ladder**: cheap methods run on everything, and each successively more expensive method runs only on the shrinking residual of documents the cheaper methods could not confidently classify.

![The cost ladder for quality filtering: a cheap fastText scorer runs on all documents, most are confidently kept or dropped, and only the small borderline residual escalates to perplexity rechecks and finally an LLM-as-judge on a tiny fraction](/imgs/blogs/classifier-and-perplexity-based-quality-filtering-7.webp)

Read the ladder top to bottom. At the base, a fastText classifier scores 100% of documents at roughly a dollar per million — cheap enough to run on the whole corpus. Most documents — call it 95% — come out with a confident score: either clearly above or clearly below your keep-threshold, decided and done. Only the borderline band, the roughly 5% clustered around the threshold, is worth spending more on. Those escalate to a perplexity recheck or a small-LM score — more expensive than fastText, but you are only running it on a twentieth of the corpus. That resolves most of them, and only the still-ambiguous residual — a tenth of a percent — escalates to the LLM-as-judge, the most expensive tier, now running on a rounding error of the original corpus.

The arithmetic is what makes the ladder worth building. Here is the cost shape for a hypothetical billion-document corpus:

| Tier | Tool | Coverage | Cost basis | Rough total |
| --- | --- | --- | --- | --- |
| 1 | fastText classifier | 100% (1B docs) | ~\$1 / 1M docs | ~\$1,000 |
| 2 | perplexity / small-LM recheck | ~5% (50M docs) | ~\$20 / 1M docs | ~\$1,000 |
| 3 | LLM-as-judge | ~0.1% (1M docs) | ~\$5,000 / 1M docs | ~\$5,000 |

Total: about \$7,000 to apply high-quality-judgment-grade filtering to a billion documents. Now compare the naive alternative — running the LLM judge on all billion documents at \$5,000 per million — which comes to \$5 million. The ladder is roughly three orders of magnitude cheaper, and the quality of its decisions on the documents that matter (the borderline ones) is nearly as good, because those are exactly the ones that got escalated to the expensive judge.

This is the same idea as the FineWeb-Edu distillation, generalized: **spend expensive compute only where cheap compute is uncertain.** FineWeb-Edu spends the LLM once on a sample to train a cheap classifier; the cost ladder spends it repeatedly but only on the residual. In practice you often use both — a distilled classifier as the tier-1 workhorse, and an LLM judge as the tier-3 tiebreaker for the band where the distilled classifier is least confident. The one rule that makes the ladder pay off: **the cheap tier must produce a confidence, not just a decision**, so you know which documents to escalate. A classifier that only outputs "keep" or "drop" gives you nowhere to route the hard cases; one that outputs a calibrated score tells you exactly which 5% deserve a second look.

## Troubleshooting: when quality filtering silently hurts you

Every failure mode in this section shares a property that makes it dangerous: the filter looks like it is working. Precision is high, the offline metrics are green, the kept documents look clean when you spot-check them. The damage shows up only in the trained model, weeks later, on evals you may not even be running. Here is the field guide — symptom, cause, and fix.

| Symptom | Root cause | Fix |
| --- | --- | --- |
| Model's coding/math ability drops after adding the filter | Classifier trained on prose positives scores code and formulas as low-quality and deletes them | Add code, math, and technical docs to the positive set; or run the classifier per-domain with domain-specific thresholds |
| Filter keeps clean-looking repetitive junk | Perplexity rewards repetition; templated spam scores as low-perplexity | Run dedup and repetition heuristics *before* perplexity, never instead of |
| High offline precision, no downstream gain | Threshold tuned to maximize classifier precision, not the eval; corpus over-shrunk | Re-tune the threshold against a proxy-model ablation, not against the classifier's own metrics |
| Filter works on one shard, fails on another | Distribution shift between the shard the classifier was trained on and the one being scored | Re-sample and re-annotate per major domain/language; recalibrate the threshold per shard |
| Non-English or dialectal text disappears | Reference LM / positives are English-centric; everything else scores as high-perplexity or low-P(good) | Train per-language reference models (CCNet's actual design); stratify the classifier by language |
| Aggressive filter, worse model than no filter | Kept only the top few percent; token-starved and lost diversity | Lower the threshold; the eval-optimal cut is almost never the precision-optimal cut |

The one that costs the most, by a wide margin, is the first row — and it deserves its own picture, because it is the perfect illustration of "the classifier learns your label choice, not quality."

![How classifier bias deletes good technical data: a prose-only classifier scores a great CUDA kernel write-up at 0.08 and drops it, while a domain-aware classifier with code in its positives scores the same document at 0.71 and keeps it](/imgs/blogs/classifier-and-perplexity-based-quality-filtering-6.webp)

Trace both columns of that figure. On the left, a classifier whose positive set was Wikipedia and books encounters a genuinely excellent write-up of a CUDA kernel — dense with code, terse prose, unusual tokens. The classifier has never seen anything like it in its positives, so it scores it 0.08 and drops it. Nobody decided to delete great engineering content; the *label choice* decided, silently, and the only place you would ever find out is a coding eval on the final model. On the right, the same document scored by a classifier whose positives *include* code and technical writing: 0.71, kept. The document did not change. The definition of quality did.

The general debugging move for all of these is the same, and it is worth internalizing as a habit: **audit what the filter removed, not just what it kept.** Everyone spot-checks the keep set and pronounces it clean. Almost nobody pulls a random sample of the *dropped* documents and reads them, and that is precisely where the silent damage hides — the brilliant technical post scored 0.08, the dialectal transcript scored as garble, the dense math paper flagged as high-perplexity noise. A quality filter is defined as much by its false negatives as its true positives, and false negatives only show up in the reject pile. Make reading the reject pile part of every filter's acceptance test.

## Case studies from production

### 1. FineWeb-Edu and the educational-value classifier

The headline case, revisited for its lessons rather than its mechanics. The Hugging Face team sampled ~460,000 documents from FineWeb, scored each 0-5 for educational value with Llama-3-70B-Instruct against an anchored rubric, distilled those labels into a small encoder-plus-regression-head classifier, and filtered the full corpus at a keep-threshold of 3. The trained model reached MMLU and ARC levels that raw FineWeb needed several times the tokens to approach. The durable lesson is the distillation pattern — expensive teacher on a sample, cheap student on the corpus — but the subtler lesson is that the *rubric* was the product. The team iterated on the annotation prompt more than on the model architecture, because the rubric is the operational definition of quality, and a distilled classifier can be no better than the labels it learns from.

### 2. CCNet perplexity bucketing across a hundred languages

CCNet's contribution was proving that a dirt-cheap n-gram perplexity filter, applied per-language against a Wikipedia reference, could turn raw Common Crawl into usable multilingual pretraining data at web scale. The head/middle/tail bucketing gave downstream users a quality dial rather than a binary keep/drop, and the per-language reference models were the critical design choice — a single English reference would have flagged every non-English document as garbage. CCNet's data underpinned a generation of multilingual models. The lesson: the cheapest model-based filter, done carefully and per-language, still delivers enormous value, and it is the right first thing to reach for when heuristics run out.

### 3. GPT-3's Common Crawl classifier

GPT-3's training data was filtered with a logistic-regression classifier that separated a positive set (WebText, Wikipedia, books, and Reddit-referenced pages) from raw Common Crawl, keeping documents by a probabilistic threshold with a bit of noise injected so the filter was not perfectly deterministic. It was one of the first large-scale demonstrations that a simple linear classifier over a well-chosen positive set could materially improve web data. It also seeded the field's default bias: because the positive set was formal, edited, expository text, the filter — and the many pipelines that copied it — systematically favored that register. Much of the later work on code and math data was, in effect, correcting for the register that this original positive set baked in.

### 4. DCLM and the fastText classifier that beat fancier methods

The DataComp for Language Models (DCLM) benchmark ran a controlled bake-off of filtering methods on a fixed corpus and fixed compute, and one of its sharpest findings was how well a *simple* fastText classifier performed when its positive set was chosen well — specifically, positives drawn from instruction-formatted and high-quality conversational data rather than just encyclopedic prose. It outperformed several more elaborate model-based filters. The lesson landed hard: the *choice of positive set* dominates the *choice of classifier architecture*. Teams routinely over-invest in the model and under-invest in the labels; DCLM is the receipt showing the labels matter more.

### 5. The war story: the filter that deleted the company's own docs

A team I worked alongside shipped a quality classifier trained on a broad web-prose positive set and rolled it across their pretraining corpus, including a valuable internal slice of technical documentation and API references. Every offline metric was green. Two weeks into a training run, an eval on technical question-answering came back *worse* than the pre-filter baseline. The audit-the-reject-pile move found it in an afternoon: the classifier had scored their densest, most valuable API docs — code-heavy, terse, full of identifiers — as low-quality web sludge and dropped roughly 70% of them. The positive set had no code in it, so the classifier had learned that code looks like noise. The fix was one line of intent and a day of work: add the technical docs and a code corpus to the positive set, retrain, re-score. The eval recovered and then exceeded the baseline. The lesson costs nothing to learn from someone else's run: your positive set is your definition of quality, and anything absent from it is, to the filter, garbage.

## When to reach for model-based filtering (and when not to)

Reach for it when:

- **Your heuristic filters have plateaued** and the next quality gains are semantic — "reads like something worth learning from" rather than "has a sane symbol ratio."
- **You have a clear, capability-aligned notion of quality** you can express as a positive set or an annotation rubric. If you can write the rubric, you can build the filter.
- **The corpus is large enough that a cheap filter's per-document savings dwarf the setup cost** — which is essentially any web-scale pretraining corpus.
- **You can run the ablation loop.** Model-based filtering without a downstream ablation to set the threshold is guessing with extra steps; the [measurement infrastructure](/blog/machine-learning/training-data/measuring-data-quality) is a prerequisite, not a nice-to-have.
- **You want a tunable dial**, not a binary decision — perplexity buckets and calibrated classifier scores let you move the quality/quantity operating point as your token budget changes.

Skip it, or apply it with great care, when:

- **The corpus is already curated and small** (a books collection, a vetted domain corpus). A quality classifier trained on web data will misjudge it, and the setup cost is not amortized over enough documents to pay off.
- **You cannot articulate what "good" means** for your target model. A filter built on a fuzzy positive set will faithfully encode your confusion and delete things you did not mean to delete.
- **You are building a specialist model** (code, math, a specific language) and only have generalist filters. A prose-quality classifier applied to a code corpus is the single most reliable way to delete exactly the data you need — see the war story above.
- **You have not built the reject-pile audit into your process.** Without it you will not catch the silent false negatives, and a filter you cannot debug is a liability, not an asset.

The throughline of this entire post is one sentence: **model-based filtering encodes a definition of quality, so the definition — your positive set, your rubric, your threshold — is the real work, and the model is just the machinery that applies it at scale.** Get the definition right, validate it against a downstream ablation rather than against itself, and read what it throws away. Do those three things and a fastText classifier will beat a fancy method with a careless label set every time. The next step in the pipeline — deciding *how much* of the filtered data to keep and in what mix, and pruning near-duplicates and low-value documents from the survivors — is covered in [data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning).

## Further reading

- [Measuring Data Quality: The Ablation Loop That Drives Every Curation Decision](/blog/machine-learning/training-data/measuring-data-quality) — the measurement backbone every threshold decision here depends on.
- [Language Identification and Heuristic Quality Filters](/blog/machine-learning/training-data/language-identification-and-heuristic-quality-filters) — the rules-based layer that runs before model-based filtering.
- [Data Quality Scaling Laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) — why a better filter behaves like more compute.
- [Data Scaling Laws and Budgets](/blog/machine-learning/training-data/data-scaling-laws-and-budgets) — how to set the quality/quantity operating point the threshold controls.
- [Data Selection and Pruning](/blog/machine-learning/training-data/data-selection-and-pruning) — what to do with the documents that survive the filter.
- The CCNet paper (Wenzek et al.), the FineWeb / FineWeb-Edu technical report (Penedo et al.), the DataComp-LM benchmark (Li et al.), and the Ask-LLM data-selection work (Sachdeva et al.) are the primary sources behind the methods above.
