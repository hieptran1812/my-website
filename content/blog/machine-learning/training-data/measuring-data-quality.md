---
title: "Measuring Data Quality: The Ablation Loop That Drives Every Curation Decision"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "You cannot curate what you cannot measure. A principal-engineer's guide to the signal ladder — heuristics, perplexity, classifiers, LLM-judges — and the gold standard: clean proxy-model ablations with honest noise bands."
tags: ["training-data", "data-quality", "data-curation", "ablation", "proxy-model", "perplexity-filtering", "quality-classifier", "llm-as-judge", "evaluation", "seed-variance", "goodharts-law", "fineweb", "pretraining-data"]
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 41
---

There is a sentence that gets repeated in every data-curation talk and quietly ignored in every data-curation pipeline: *you cannot improve what you do not measure*. The reason it gets ignored is that measuring data quality is genuinely hard, and the easy measurements — the ones a dashboard can show you in green — are exactly the ones that lie to you. A corpus can have a beautiful deduplication rate, a tidy length distribution, a low average perplexity against your reference model, and still produce a worse model than the raw web dump you started from. I have personally watched a team ship a filtering change that improved every offline statistic they tracked and cost them about a point of downstream accuracy, because none of those statistics were the thing they actually cared about.

This post is the measurement backbone of the whole training-data series. Every later decision — which classifier to train, how aggressively to deduplicate, whether to keep the top 10% or the top 30%, how much synthetic data to mix in — is downstream of a single capability: *can you tell, reliably and cheaply enough, whether change X made the data better?* If you cannot, you are not curating data, you are decorating it. The loop below is the mental model for everything that follows, and the rest of the article is a tour of how to run each step of it without fooling yourself.

<figure class="blog-anim">
<svg viewBox="0 0 720 470" role="img" aria-label="The curation feedback loop: measure, change one knob, re-train the proxy, compare eval deltas, then repeat — a dot travels the loop one iteration per lap" style="width:100%;height:auto;max-width:760px">
<style>
.a1-ring{fill:none;stroke:var(--accent,#6366f1);stroke-width:3;opacity:.4}
.a1-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.a1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a1-mid{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.a1-dot{fill:var(--accent,#6366f1)}
@keyframes a1-orbit{0%{transform:translate(0px,0px)}25%{transform:translate(150px,150px)}50%{transform:translate(0px,300px)}75%{transform:translate(-150px,150px)}100%{transform:translate(0px,0px)}}
.a1-orbit{animation:a1-orbit 12s linear infinite}
@media (prefers-reduced-motion:reduce){.a1-orbit{animation:none}}
</style>
<defs><marker id="a1-ah" markerWidth="11" markerHeight="11" refX="7" refY="3.5" orient="auto"><path d="M0,0 L8,3.5 L0,7 Z" fill="var(--accent,#6366f1)"/></marker></defs>
<path class="a1-ring" d="M 360 60 A 150 150 0 0 1 510 210" marker-end="url(#a1-ah)"/>
<path class="a1-ring" d="M 510 210 A 150 150 0 0 1 360 360" marker-end="url(#a1-ah)"/>
<path class="a1-ring" d="M 360 360 A 150 150 0 0 1 210 210" marker-end="url(#a1-ah)"/>
<path class="a1-ring" d="M 210 210 A 150 150 0 0 1 360 60" marker-end="url(#a1-ah)"/>
<rect class="a1-box" x="255" y="34" width="210" height="72" rx="10"/>
<text class="a1-lbl" x="360" y="64">1. Measure</text>
<text class="a1-lbl" x="360" y="86">(a ladder signal)</text>
<rect class="a1-box" x="505" y="168" width="205" height="84" rx="10"/>
<text class="a1-lbl" x="607" y="204">2. Change</text>
<text class="a1-lbl" x="607" y="226">exactly one knob</text>
<rect class="a1-box" x="255" y="314" width="210" height="84" rx="10"/>
<text class="a1-lbl" x="360" y="350">3. Re-train proxy</text>
<text class="a1-lbl" x="360" y="372">on equal tokens</text>
<rect class="a1-box" x="10" y="168" width="205" height="84" rx="10"/>
<text class="a1-lbl" x="112" y="204">4. Compare</text>
<text class="a1-lbl" x="112" y="226">eval deltas</text>
<text class="a1-mid" x="360" y="198">each lap =</text>
<text class="a1-mid" x="360" y="218">one ablation</text>
<text class="a1-mid" x="360" y="238">iteration</text>
<text class="a1-mid" x="150" y="120">keep if eval rises,</text>
<text class="a1-mid" x="150" y="138">revert otherwise</text>
<circle class="a1-dot a1-orbit" cx="360" cy="60" r="11"/>
</svg>
<figcaption>Every curation decision rides this loop: measure a signal, change one knob, re-train a small proxy, compare eval deltas, keep or revert — then go around again.</figcaption>
</figure>

The loop above is the mental model. The four boxes are deceptively simple, and three of them hide most of the failure modes in practice. "Measure" is where you choose a signal, and the cheap signals lie. "Change exactly one knob" is where confounds creep in. "Compare eval deltas" is where seed noise gets misread as a result. The rest of this post is about doing each of those steps honestly, because a loop that turns fast but turns on bad measurements just lets you make mistakes more efficiently.

## Why measurement is different from what you think

The first thing to internalize is that "data quality" is not a property of a document. It is a property of a *(document, model, objective, budget)* tuple. A snippet of dense legal boilerplate is garbage for a chatbot and gold for a contract-analysis model. A page of competitive-programming solutions is noise for a 100M-parameter model that will never learn to code and signal for a 7B model that might. The moment you accept that quality is relative to what you are training and how much compute you have, most of the popular shortcuts reveal themselves as proxies — useful, but proxies — and the question becomes which proxy to trust and when.

| The assumption | The naive view | The reality |
| --- | --- | --- |
| Quality is intrinsic to a document | Score each doc once, reuse forever | Quality depends on the target model size, objective, and compute budget |
| Lower perplexity means higher quality | Filter out high-perplexity docs | Perplexity rewards text that *looks like the reference model* — including bland, repetitive, near-duplicate text |
| Offline stats predict downstream quality | Track dedup rate, length, symbol ratio | Those stats catch gross junk but barely correlate with eval deltas past the first cleanup pass |
| A better classifier score is a better corpus | Keep the highest-scoring decile | Aggressive filtering shrinks diversity; the optimal cut depends on your token budget |
| One eval number tells you if it worked | Read the average, ship it | A 0.3-point move is usually seed noise; you need the noise band before you read the number |

The through-line of that table is that almost every cheap measurement is measuring *resemblance to something* — resemblance to fluent text, resemblance to a reference distribution, resemblance to a labeled "good" set — and resemblance is correlated with quality right up until you optimize against it, at which point the correlation breaks. That is not a reason to abandon cheap signals. It is a reason to know exactly how far each one can be trusted, and to anchor all of them against a measurement that actually trains a model. This relationship between data curation and the loss-versus-compute curve is the subject of [data quality as a scaling axis](/blog/machine-learning/scaling-laws/data-quality-scaling-laws); here we are concerned with the measurement methodology that lets you find those curves in the first place.

> Cheap data metrics answer "does this look like good data?" Only an ablation answers "does a model trained on this get better?" Those are different questions, and the gap between them is where most curation projects quietly fail.

## The ladder of signals: cheap to expensive

Picture your measurement options as a ladder. At the bottom are signals that cost milliseconds per document and tell you almost nothing about downstream quality beyond catching obvious garbage. At the top is the one signal that *is* downstream quality — training a model and evaluating it — which costs GPU-hours per data point. Every rung trades compute for fidelity, and the engineering skill is climbing exactly as high as a given decision warrants and no higher.

![A vertical ladder of five data-quality signals from cheapest at the bottom (heuristic stats) to most expensive at the top (proxy ablation), with side arrows showing that cost and downstream fidelity both increase as you climb](/imgs/blogs/measuring-data-quality-1.webp)

The ladder has a property worth stating explicitly: the cheap rungs are *necessary but not sufficient*, and the expensive rungs are *sufficient but not always necessary*. You always run the heuristic filters — they are basically free and they catch the catastrophic stuff. You rarely run a full ablation for every micro-decision, because you cannot afford 200 ablations. The art is using the cheap rungs to narrow the search space and the expensive rungs to make the calls that matter. Let me walk up the ladder rung by rung.

### Rung 1: heuristic statistics

Heuristic stats are the regex of data quality: crude, fast, and indispensable for the first pass. Document length, mean word length, ratio of symbols to words, fraction of lines ending in punctuation, fraction of duplicate lines within a document, presence of a minimum count of stop-words — these are the filters that Gopher, C4, and RefinedWeb all leaned on for the initial cleanup. They are cheap enough to run over a 15-trillion-token corpus, and they reliably remove the stuff that is unambiguously broken: navigation-bar soup, JavaScript dumps, pages that are 90% punctuation, documents three words long.

```python
import re

def heuristic_quality_flags(text: str) -> dict:
    """Cheap per-document signals. Each is a weak proxy; together they
    catch gross junk. None of these should be trusted past the first pass."""
    words = text.split()
    n_words = len(words)
    lines = text.splitlines()
    n_lines = max(len(lines), 1)

    alpha_chars = sum(c.isalpha() for c in text)
    total_chars = max(len(text), 1)
    symbol_chars = sum(c in "#{}[]<>|\\^~" for c in text)

    dup_lines = n_lines - len(set(lines))

    return {
        "n_words": n_words,
        "mean_word_len": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "alpha_ratio": alpha_chars / total_chars,
        "symbol_word_ratio": symbol_chars / max(n_words, 1),
        "dup_line_frac": dup_lines / n_lines,
        "frac_lines_end_punct": sum(l.rstrip().endswith((".", "!", "?", "\"")) for l in lines) / n_lines,
    }

def passes_gopher_style_filters(flags: dict) -> bool:
    return (
        50 <= flags["n_words"] <= 100_000
        and 3.0 <= flags["mean_word_len"] <= 10.0
        and flags["alpha_ratio"] >= 0.6
        and flags["symbol_word_ratio"] <= 0.1
        and flags["dup_line_frac"] <= 0.30
    )
```

Here is the crucial thing about Rung 1: **the marginal value of a heuristic filter collapses after the first pass.** The first time you apply length and symbol-ratio filters to raw Common Crawl, you remove a large slug of junk and you will measure a real downstream gain. The tenth heuristic you add — some clever rule about the ratio of bullet points to paragraphs — almost never moves a downstream eval, because the documents it catches are rare and the model was already largely ignoring them. Teams burn months tuning Rung 1 thresholds because the offline statistics keep improving. The offline statistics improving is not the goal. I have seen a "97.3% of documents now pass our quality gate" slide presented as a win when the actual model had not moved in five iterations. Heuristics are a floor, not a lever.

The second-order trap with heuristics is that they are *language- and domain-biased*. A symbol-word-ratio threshold tuned on English prose will shred code, LaTeX, and tabular data. A "minimum stop-word count" filter will delete perfectly good Chinese or Thai, which do not use spaces the way the filter assumes. Every heuristic is a tiny implicit model of what text should look like, and if you do not check what it removes, it will quietly amputate whole domains. This is exactly the discipline argued in [look at your data before you train](/blog/machine-learning/debugging-training/look-at-your-data-before-you-train): pull a random sample of what each filter *removes* and read it, because the removed set is where the surprises live.

### Rung 2: perplexity against a reference language model

The next rung up scores each document by how surprising it is to a small reference language model. Concretely, you take a cheap pretrained model — often a 5-gram KenLM trained on a clean corpus like Wikipedia, or a small transformer — and compute the per-token perplexity of each document. Low perplexity means the reference model finds the text predictable and fluent; high perplexity means it finds the text surprising. The CCNet pipeline that underlies a lot of multilingual web data does exactly this, bucketing documents into perplexity terciles and treating the low-perplexity bucket as "head" (clean) and the high-perplexity bucket as "tail" (noisy).

![A right-skewed histogram of document counts versus reference-LM perplexity, with a dashed cut line separating a green low-perplexity keep region from a red high-perplexity drop region, and a callout noting the tail holds spam and machine translation but also some genuinely good documents](/imgs/blogs/measuring-data-quality-2.webp)

The figure shows the move and its hazard in one picture. Most documents pile up at moderate perplexity, with a long right tail of high-perplexity text. Set a cut line, keep the left, drop the tail. The tail genuinely is enriched for junk: spam, machine-translation artifacts, OCR garbage, keyword-stuffed SEO pages. But the tail *also* contains genuinely good documents that happen to be surprising to your small reference model — technical writing full of rare terminology, code, poetry, non-mainstream dialects, anything stylistically far from the reference corpus. Every document you cut from the green-shaded part of the tail is a false positive, and you cannot see those false positives in the perplexity histogram. You can only see them by sampling what gets dropped, or by running an ablation.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ref_name = "EleutherAI/pythia-160m"  # a small, cheap reference model
tok = AutoTokenizer.from_pretrained(ref_name)
ref = AutoModelForCausalLM.from_pretrained(ref_name, torch_dtype=torch.bfloat16).cuda().eval()

@torch.no_grad()
def doc_perplexity(text: str, max_len: int = 2048) -> float:
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.cuda()
    if ids.shape[1] < 2:
        return float("inf")
    out = ref(ids, labels=ids)
    # HF returns mean token NLL in out.loss; perplexity is exp(NLL)
    return torch.exp(out.loss).item()

# A perplexity *cut* is one knob. The right cut is an empirical question:
# different cuts -> different corpora -> different ablation results.
def keep_by_perplexity(docs, cut: float):
    return [d for d in docs if doc_perplexity(d) <= cut]
```

The deep failure mode of perplexity filtering is that **the reference model defines "good," and a small reference model has a narrow idea of good.** Optimize hard against low perplexity and you select for text that resembles your reference corpus — which is usually clean, formal, Wikipedia-like prose — and you systematically strip out the diversity that makes a model robust. This is the single most common way perplexity filtering backfires, and it is the textbook setup for Goodhart's law, which gets its own treatment in the troubleshooting section. For now, the rule of thumb: perplexity is a fine *coarse* tail-trimmer and a dangerous *fine* selector. Use it to drop the worst tercile, not to rank the best decile.

### Rung 3: trained quality classifiers

The third rung replaces a hand-built reference distribution with a *learned* one. You assemble a set of documents you consider high quality — a common recipe is to treat curated sources like Wikipedia, books, or pages linked from a trusted aggregator as positives, and random Common Crawl as negatives — and train a fast classifier to distinguish them. A linear fastText classifier over n-gram features is the workhorse here because it scores documents at hundreds of thousands per second per core; a small BERT-style encoder is the heavier, more accurate option. GPT-3, the Pile, and DataComp-LM all used a learned classifier as a central quality filter.

The classifier is more expressive than perplexity because it can learn arbitrary lexical and stylistic signals of "the kind of text we put in the positive set." That is also its weakness: it learns *the kind of text in the positive set*, full stop. If your positives are Wikipedia and your negatives are raw web, you have trained a Wikipedia-detector, and "high quality" now silently means "Wikipedia-like." That can be exactly right (knowledge-dense, well-edited text is genuinely valuable) or exactly wrong (you have just down-weighted forums, dialogue, and code, which the model also needs to learn). The classifier's decision boundary is only as good as the contrast you built into its training labels.

```python
import fasttext

# Train a fast quality classifier from a labeled contrast set.
# __label__hq lines are curated/high-quality; __label__lq are random web.
model = fasttext.train_supervised(
    input="quality_train.txt",   # "__label__hq <text>" / "__label__lq <text>"
    lr=0.5, epoch=5, wordNgrams=2, dim=100, minCount=3,
)

def quality_score(text: str) -> float:
    # P(high-quality) for this document, in [0, 1].
    labels, probs = model.predict(text.replace("\n", " "), k=2)
    return dict(zip(labels, probs)).get("__label__hq", 0.0)
```

Two second-order points decide whether a classifier helps. First, **the threshold matters more than the model.** A classifier that gives you a clean score still leaves you with the question of where to cut, and that cut is a knob you must ablate, not guess — the optimal aggressiveness depends on your compute budget, because aggressive filtering buys quality at the cost of quantity. Second, **the positive set leaks its biases into the model.** FineWeb-Edu's large knowledge-benchmark gains came specifically from defining "quality" as "educational value" via an LLM-annotated positive set, which is a much more targeted contrast than "Wikipedia versus web." The lesson is that you are not training a quality classifier, you are training a *resembles-my-positives* classifier, and the entire game is choosing positives whose resemblance actually predicts downstream gains. The mechanics of building and tuning these filters get a dedicated post in [classifier and perplexity quality filtering](/blog/machine-learning/training-data/classifier-and-perplexity-quality-filtering).

### Rung 4: LLM-as-judge scoring

Near the top of the cheap-to-moderate band sits LLM-as-judge: prompt a strong model with a rubric and ask it to rate each document, or to compare two documents. This is the most semantically rich of the proxy signals — an LLM can actually read a document and assess whether it is coherent, factual-looking, educational, or toxic in a way no n-gram classifier can. FineWeb-Edu's classifier was bootstrapped exactly this way: a strong model scored a few hundred thousand documents for educational quality, and a small classifier was distilled from those labels so the expensive judgment could be applied at corpus scale.

```python
JUDGE_RUBRIC = """Rate the educational value of the following web text for
pretraining a language model, on a 0-5 integer scale:
0 = no usable content (spam, boilerplate, gibberish)
3 = useful, coherent, some informational value
5 = highly educational, dense, well-structured, like a textbook or article
Respond with ONLY the integer.

TEXT:
{document}
SCORE:"""

def llm_judge_score(client, document: str) -> int:
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4,
        messages=[{"role": "user", "content": JUDGE_RUBRIC.format(document=document[:6000])}],
    )
    try:
        return int(resp.content[0].text.strip()[0])
    except (ValueError, IndexError):
        return 0
```

LLM-as-judge has three costs that keep it off the bottom rungs. It is *slow and expensive* — dollars per thousand documents at API prices, which is prohibitive for a multi-trillion-token corpus, hence the distill-to-a-classifier pattern. It is *subjective and non-deterministic* — the same document scored twice can get different ratings, and the rubric's wording shifts the distribution; you have to validate judge consistency the way you would validate a human annotator. And it has a *self-preference bias* — a judge tends to rate text that resembles its own outputs more highly, which quietly pulls your corpus toward the judge's distribution. Use it where its richness pays off (nuanced quality dimensions a classifier cannot capture, or generating labels to distill) and never treat its score as ground truth. Like every rung below it, it answers "does this look good to a model that has opinions," not "does training on this make a model better."

## The gold standard: ablation on a proxy model

Here is the rung that is not a proxy at all, because it measures the actual quantity of interest. You take your candidate corpus, train a small model on it under a fixed budget, evaluate that model on downstream tasks, and read the result. There is no resemblance heuristic in the loop — you are directly observing "a model trained on this data scores X." Everything below this rung is an attempt to *predict* this number cheaply; this rung *is* the number.

The reason ablation is the gold standard is the reason every cheap signal is suspect: the cheap signals measure properties of the data, and the only property that ultimately matters is "what happens to a model trained on it." Two corpora can have identical perplexity distributions, identical classifier-score histograms, identical dedup rates, and produce models a full point apart on MMLU, because the difference lives in something none of your cheap metrics captured — topic coverage, the long tail of rare-but-important documents, subtle distributional shifts. The ablation sees all of it, because the model sees all of it.

The catch, of course, is cost. You cannot train a 70B model to test every filter; you train a *proxy*: a small model (commonly in the 1B to 3B range) on a reduced token budget (tens of billions of tokens), as a stand-in for the full run. This works because data-quality effects are, to a useful approximation, transferable across scale — a filter that helps a 1.6B model on 30B tokens usually helps a 7B model on 1T tokens, in the same direction if not the same magnitude. "Usually" is doing real work in that sentence, and the cases where the proxy's verdict does not transfer are exactly the cases the troubleshooting section is about. But as a default, the proxy ablation is the most honest signal you can afford, and the entire discipline of data curation is built on running it cleanly.

> A proxy ablation is the only measurement on the ladder that can disagree with all the others and be right. When the heuristics, perplexity, and classifier all say "better" and the ablation says "worse," believe the ablation. That is the whole reason it sits on top.

## Designing a clean ablation

An ablation is a controlled experiment, and a controlled experiment is only as good as its control. The cardinal rule is the same one from any science: **change exactly one variable and hold everything else fixed.** In a data ablation, "everything else" is a long list — model architecture and size, total tokens, optimizer and hyperparameters, learning-rate schedule, batch size, sequence length, the random seed, and the evaluation suite — and the one variable you are allowed to change is the data knob under test. If you change the filter *and* the token count at the same time, your result is uninterpretable: you cannot attribute the eval delta to the filter.

![A two-column control table comparing Run A and Run B across params, tokens, optimizer, learning-rate schedule, seeds, and eval suite, with every row identical and held fixed except the highlighted bottom row where the data filter differs between keep-top-30-percent and keep-top-10-percent](/imgs/blogs/measuring-data-quality-3.webp)

The table is the entire discipline in one picture: six rows held identical, one row varied. When you read an ablation result and cannot point to the single row that differs, you do not have an ablation, you have an anecdote. Four design decisions turn this principle into a measurement you can actually trust.

**Hold compute and tokens fixed, not just "roughly equal."** The most common silent confound is comparing a filter that keeps 30% of the data against one that keeps 10%, training both for "one epoch." One epoch of the 10%-filter corpus is far fewer tokens than one epoch of the 30%-filter corpus, so you have changed both the filter and the token budget. The fix is to fix the *token budget* — train both runs on exactly the same number of tokens, repeating data if necessary — so the only thing that differs is which documents those tokens came from. This is the difference between asking "is this filter better?" and the muddier "is this filter better, at a different data scale, for reasons I cannot disentangle?" The compute-dependence of the right answer is the central result of the [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws), and it is precisely why the budget must be a controlled variable, not a free one.

**Pick evals with signal at small scale.** A 1.6B model trained on 30B tokens is not going to do well on hard reasoning benchmarks — it will sit near random on GSM8K and barely above chance on the hardest MMLU subsets. An eval where your proxy scores at random has zero signal: the delta you measure is pure noise. FineWeb's team was explicit about this and curated a set of "early-signal" benchmarks chosen by four criteria: the metric is *above random* even for a small model, it is *monotonic* (rises as training proceeds rather than bouncing), it has *low variance* across seeds, and it *ranks known-good and known-bad corpora correctly*. HellaSwag, ARC, CommonsenseQA, and a few others clear that bar at 1.6B; MATH and GPQA do not. Choosing the wrong evals is the fastest way to run a clean experiment and learn nothing.

**Control seed variance — measure it before you trust any delta.** Two training runs that differ *only* in their random seed will not produce identical eval scores. Data shuffling, dropout, and initialization all inject noise, and that noise is often ±0.3 to ±0.7 accuracy points on a small-model eval. If your filter produces a +0.4 point change and your seed noise is ±0.5, you have measured nothing. The only defense is to *quantify the noise floor*: run the same configuration with two or three different seeds, measure the spread, and treat any delta smaller than that spread as zero. This single practice separates teams that make real progress from teams that chase noise for a quarter.

**Watch for eval saturation.** The opposite of a too-hard eval is a too-easy one. If your proxy already scores 92% on some benchmark, there is almost no headroom for a data improvement to show up, and a ceiling effect will make every filter look identical on that metric. Saturated evals are as useless as random ones — they just fail in the other direction. The fix is to track a *suite* with a range of difficulties and to retire any eval your proxy has saturated.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class AblationConfig:
    """Everything that must be held FIXED across runs in one ablation.
    The only field that may vary between Run A and Run B is `data_recipe`."""
    params: str = "1.6B"
    tokens: int = 30_000_000_000     # fixed token BUDGET, not epochs
    optimizer: str = "adamw"
    weight_decay: float = 0.1
    lr_schedule: str = "cosine"
    warmup_steps: int = 2_000
    seq_len: int = 2048
    eval_suite: tuple = ("hellaswag", "arc_easy", "arc_challenge", "mmlu", "commonsense_qa")
    seeds: tuple = (0, 1, 2)         # multiple seeds -> a noise band, not a point

def run_ablation(data_recipe: str, cfg: AblationConfig) -> dict:
    """Train cfg.params on exactly cfg.tokens drawn from `data_recipe`,
    once per seed, and return per-eval mean and std across seeds."""
    per_seed = []
    for seed in cfg.seeds:
        model = train_proxy(data_recipe, cfg, seed=seed)   # your trainer
        per_seed.append(evaluate(model, cfg.eval_suite))   # {eval: acc}
    return aggregate_mean_std(per_seed)                     # {eval: (mean, std)}
```

The structure of that config object is the methodology made executable: a frozen bundle of everything held constant, one free `data_recipe` argument, and a `seeds` tuple that turns every result from a point estimate into a distribution. If your ablation harness does not return a standard deviation, it is hiding the most important number on the page.

## A worked scenario: comparing two filters

Let me make this concrete with numbers. Suppose you are deciding how aggressively to apply a quality classifier, and the two candidates are *keep the top 30%* (Filter A) and *keep the top 10%* (Filter B). The intuition is that the stricter filter yields cleaner data and therefore a better model. The intuition is testable, so we test it. We fix the proxy at 1.6B parameters and the budget at 30B tokens, draw those 30B tokens from each filter's output (repeating the smaller corpus as needed so the token count is identical), train three seeds per filter, and evaluate on a five-eval suite. Then we report the delta of Filter B minus Filter A, *with* a noise band derived from the seed spread.

![A horizontal forest plot of Filter B minus Filter A eval deltas with confidence whiskers: HellaSwag +1.6, ARC-easy +1.1, and the 10-task average +0.8 clear the zero line as real wins, while ARC-challenge +0.4 and MMLU -0.3 have intervals that straddle zero and are within seed noise](/imgs/blogs/measuring-data-quality-4.webp)

Read the figure the way a careful analyst reads it. Filter B beats Filter A by +1.6 points on HellaSwag with a noise band of ±0.4 — the whole interval is on the positive side of zero, so that is a real win. ARC-easy, +1.1 ± 0.5, also clears zero. The aggregate, +0.8 ± 0.3 across the suite, is real and is the headline number. But look at ARC-challenge: +0.4 ± 0.6. The interval straddles zero. And MMLU: −0.3 ± 0.5, also straddling zero. The naive reading of this table — the one a manager skims off a dashboard — is "Filter B is better on three evals, tied on one, and *worse on MMLU*." The correct reading is "Filter B is genuinely better on average and on two evals, and indistinguishable from Filter A on the other two, including MMLU, because those moves are inside the noise."

The difference between those two readings is a quarter of wasted work. The naive reading triggers a panic about why the stricter filter "hurts MMLU," a round of investigation into a −0.3 that does not exist, and possibly a decision to abandon a filter that is actually better. The correct reading ships Filter B and moves on. The only thing standing between them is the noise band, which is why I keep hammering on it: **a delta without a noise band is not a measurement, it is a rumor.**

How do you get the noise band? The cheap version is the seed standard deviation, scaled to an interval. The honest version, when you have a handful of seeds, is a small bootstrap over the per-seed scores so you do not over-trust three samples.

```python
import numpy as np

def delta_with_band(scores_a: list[float], scores_b: list[float], z: float = 1.96):
    """Per-eval delta (B - A) with a noise band from seed variance.
    scores_* are per-seed accuracies for one eval, e.g. [54.1, 53.7, 54.6]."""
    a, b = np.array(scores_a), np.array(scores_b)
    delta = b.mean() - a.mean()
    # Pooled seed standard error of the difference of two means.
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return delta, (delta - z * se, delta + z * se)

def is_real(delta, interval) -> bool:
    """A delta is 'real' only if its interval excludes zero."""
    lo, hi = interval
    return lo > 0 or hi < 0

hella_a = [52.4, 52.0, 52.8]   # Filter A, 3 seeds
hella_b = [54.1, 53.7, 54.6]   # Filter B, 3 seeds
d, ci = delta_with_band(hella_a, hella_b)
print(f"HellaSwag delta = {d:+.2f}, band = [{ci[0]:+.2f}, {ci[1]:+.2f}], real = {is_real(d, ci)}")
# HellaSwag delta = +1.73, band = [+1.10, +2.37], real = True
```

A few practical notes on this scenario. First, three seeds is the floor, not the target — with three seeds your variance estimate is itself noisy, which is why the bootstrap or a conservative interval matters. Second, the aggregate across a suite is more reliable than any single eval, because averaging independent noise shrinks the band; this is why "the 10-task average moved +0.8 ± 0.3" is a stronger claim than any one eval's delta. Third, and most importantly, this whole scenario consumed six proxy training runs (two filters times three seeds) — a real but bounded cost. That cost is the budget you spend to *know* instead of *guess*, and it is almost always worth it for a decision as consequential as filter aggressiveness, which will be applied to your entire pretraining corpus.

## The curation feedback loop, run for real

Now we can put the loop from the opening figure into practice, because every piece of it has a method attached. The loop is: measure a signal, change exactly one knob, re-train the proxy on equal tokens, compare eval deltas with noise bands, and keep the change if it helps or revert it if it does not. Then go around again. The discipline of the loop is what turns a pile of intuitions into a curation pipeline you can defend.

The single most important rule of the loop is **one knob per lap.** It is tempting, when you have a six-hour proxy run, to bundle three changes into one ablation to save time — tighten the perplexity cut, add a new heuristic, and bump the classifier threshold all at once. Do not. If the bundle helps, you do not know which of the three did it (maybe two helped and one hurt, and you have left value on the table). If the bundle hurts, you do not know which to revert. The bundle destroys the attribution that is the entire point of running the ablation. The slow, boring, correct path is one knob per lap, and over a quarter it is dramatically faster than the clever bundling path because you never have to re-litigate a confounded result.

The second rule is **keep a ledger.** Every lap is a row: the knob you changed, the corpus before and after, the per-eval deltas with bands, and the keep/revert decision. The ledger is what lets a later you (or a teammate) ask "why is the perplexity cut at 35 and not 50?" and get an answer that is a measurement instead of a shrug. Curation pipelines that lack this ledger ossify into cargo cults — nobody remembers why a threshold is what it is, so nobody dares change it.

```python
def curation_loop(base_recipe, knobs, cfg):
    """One lap per knob. Each lap changes exactly one thing, re-trains,
    compares to the current champion with noise bands, keeps or reverts."""
    champion = run_ablation(base_recipe, cfg)
    ledger = [("baseline", base_recipe, champion, "kept")]

    for knob in knobs:                       # e.g. "ppl_cut=35", "clf_keep=0.2"
        challenger_recipe = apply_knob(base_recipe, knob)   # change ONE thing
        challenger = run_ablation(challenger_recipe, cfg)   # re-train proxy

        avg_delta, band = suite_delta_with_band(champion, challenger)
        if avg_delta > 0 and band[0] > 0:    # real improvement, band clears zero
            base_recipe, champion = challenger_recipe, challenger
            decision = f"kept (+{avg_delta:.2f}, band {band})"
        else:
            decision = f"reverted ({avg_delta:+.2f}, band {band})"
        ledger.append((knob, challenger_recipe, challenger, decision))

    return base_recipe, ledger
```

The third rule, the one that is hardest to follow under deadline pressure, is **revert by default.** A change has to *earn* its way in by producing a delta whose band clears zero. "It didn't hurt" is not a reason to keep a change; it is a reason to revert it, because every knob you keep is complexity you will maintain forever and a potential confound for every future ablation. A curation pipeline should be the minimal set of transformations that each demonstrably earned their place, and the loop, run honestly, is what enforces that minimalism.

## Proxy vs downstream eval: when a cheap proxy lies

The whole ladder rests on an assumption we have so far taken on faith: that the cheap proxy signals correlate with the downstream eval well enough to be useful. Sometimes they do, beautifully. Sometimes they do not, catastrophically, and the dangerous part is that you cannot tell which case you are in *from the proxy alone*. The only way to know whether a proxy is trustworthy is to check it against the gold standard on enough cases to estimate the correlation — and to keep checking, because a proxy that was trustworthy can stop being trustworthy when you start optimizing against it.

![Two scatter panels of proxy score versus downstream eval: the left panel shows a trustworthy proxy with points tightly along a rising diagonal and a correlation near 0.92, the right panel shows a treacherous proxy with points scattered flat at a correlation near 0.05 where the proxy improves but the eval does not](/imgs/blogs/measuring-data-quality-5.webp)

The two panels are the two worlds you can be in. On the left, the proxy and the eval move together: a data recipe that scores higher on the proxy reliably trains a better model. In that world the proxy is a genuine instrument, and you can use it to pre-screen hundreds of recipes cheaply, reserving ablations for the finalists. On the right, the proxy and the eval are decoupled: recipes that score better on the proxy are no better, or worse, downstream. In that world the proxy is not just useless, it is *actively misleading* — every decision you make to improve the proxy is a coin flip on the eval, and if you are optimizing hard, it is worse than a coin flip.

The number that distinguishes these worlds is the rank correlation between the proxy and the eval across a set of recipes you have measured both ways. Pearson tells you about linear agreement; Spearman, the rank correlation, is usually what you want, because for curation decisions you care whether the proxy *ranks* recipes the way the eval does, not whether it predicts the exact score.

```python
from scipy.stats import spearmanr, pearsonr

def proxy_trustworthiness(proxy_scores, eval_scores):
    """How well does a cheap proxy rank recipes the way the gold-standard
    eval does? Compute both correlations across recipes measured both ways."""
    rho, p_rho = spearmanr(proxy_scores, eval_scores)   # rank agreement
    r, p_r = pearsonr(proxy_scores, eval_scores)        # linear agreement
    verdict = "trust for pre-screening" if rho > 0.7 else \
              "weak — confirm with ablation" if rho > 0.4 else \
              "do NOT trust — proxy is decoupled"
    return {"spearman": rho, "pearson": r, "verdict": verdict}

# Recipes measured on a cheap proxy (e.g. avg classifier score) AND a real ablation:
proxy = [0.41, 0.55, 0.60, 0.67, 0.72, 0.80, 0.88]   # proxy metric per recipe
evald = [48.1, 49.0, 49.4, 50.2, 50.9, 51.6, 52.5]   # downstream avg per recipe
print(proxy_trustworthiness(proxy, evald))
# {'spearman': 1.0, 'pearson': 0.99, 'verdict': 'trust for pre-screening'}
```

Here is the subtle and important part: **a proxy's trustworthiness is not a fixed property — it degrades as you optimize against it.** A perplexity filter might have a Spearman of 0.8 against your eval when you are choosing between mild settings, and a Spearman of 0.1 once you push it to an extreme, because the extreme settings exploit exactly the gap between "low perplexity" and "good data." This is why the right panel is labeled treacherous rather than merely useless: the proxy looks fine in the regime where you validated it and turns on you in the regime where you are actually operating. The defense is to re-validate the proxy-to-eval correlation periodically, especially after any change that pushes a knob toward an extreme, and never to trust a correlation measured in one regime when you have moved to another. This is the bridge to data selection at scale, where proxy reliability under aggressive pruning becomes the central concern, covered in [data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning).

## Case study: how FineWeb ablated its way to a better corpus

The clearest published demonstration of this methodology is FineWeb, the 15-trillion-token web corpus released by Hugging Face in 2024. What makes FineWeb a case study rather than a dataset is that its authors did not curate by intuition and validate at the end; they *ablated every single processing decision on a small model* and let the deltas decide. Their proxy was a roughly 1.8B-parameter model trained on tens of billions of tokens, evaluated on a hand-picked suite of early-signal benchmarks chosen for being above-random, monotonic, low-variance, and correctly ranking known-good against known-bad reference datasets. Every filter, every dedup strategy, every threshold was a knob, and each one earned its place by moving that suite or was discarded.

The payoff of that discipline was a set of findings that intuition got wrong. The most cited one concerns deduplication. The intuitive move is aggressive *global* deduplication — find near-duplicates across the entire corpus, across all Common Crawl snapshots, and remove them, on the theory that duplicates are wasted tokens and memorization risk. FineWeb's ablations found that global deduplication across all snapshots performed *worse* than deduplicating each snapshot independently. The mechanism, once you see it, is sensible: global dedup preferentially removes content that recurs across snapshots, and content that recurs across years of crawls is disproportionately the *good*, stable, frequently-cited material; aggressively removing it left the corpus enriched for transient low-quality pages. No amount of staring at dedup rates would have revealed this. Only the ablation did, because only the model could feel the difference.

> The FineWeb result that matters most is not any specific threshold. It is the demonstration that a counterintuitive data decision — local beats global dedup — was invisible to every cheap metric and obvious to a small ablation. That is the entire argument for the gold standard in one finding.

The same methodology, applied with a different positive set, produced FineWeb-Edu. Here the team used an LLM to score documents for educational value, distilled those scores into a classifier (the Rung 3 / Rung 4 combination from earlier), and filtered the corpus by educational quality. The ablations showed large gains specifically on knowledge-heavy benchmarks like MMLU and ARC, exactly where a more educational corpus should help, and the gains were large enough to survive transfer to full-scale training. The reason FineWeb-Edu worked where a generic "Wikipedia versus web" classifier would have given muddier results is that the *contrast set was chosen to target the dimension that the evals reward* — and the team knew it targeted that dimension because they ablated it.

It is worth placing FineWeb next to a few other named results that lean on the same measurement backbone, because the pattern repeats. **DataComp-LM (DCLM)**, also 2024, turned this entire process into a public benchmark: a fixed training and evaluation harness so that competing data-filtering recipes could be compared on an even footing — ablation as a shared, standardized service, which is the only honest way to rank curation methods against each other. Its baseline recipe leaned on a fastText classifier and demonstrated the multiplicative compute savings that the [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) post quantifies. **RefinedWeb** (the corpus behind Falcon, 2023) used heavy filtering and deduplication of web-only data and ablated its way to showing that properly curated web text could match corpora that included curated sources like books and Wikipedia — a result that, again, no offline metric would have predicted and that an ablation made undeniable. And the foundational **"Deduplicating Training Data Makes Language Models Better"** (2021) measured the effect of exact-substring and MinHash deduplication directly on model behavior — less memorization, better held-out perplexity — rather than asserting that dedup must help because duplicates are obviously bad. In every case, the corpus is the artifact and the ablation is the reason to believe in it.

## Troubleshooting: symptom, root cause, fix

The methodology above is simple to state and easy to get subtly wrong. These are the failure modes I see most often, in the order I usually have to diagnose them, each as a symptom you observe, the root cause underneath, and the fix.

### Symptom: your ablation results are noisy and won't replicate

You run an ablation, see a +0.5 improvement, run it again next week with a fresh seed, and now it is −0.2. The "result" refuses to sit still.

**Root cause: you are reading noise as signal.** Three things produce this, and they compound. *Seed variance:* a single training run is one draw from a distribution, and on small-model evals that distribution is wide (±0.3 to ±0.7 points is normal). *Eval saturation or starvation:* an eval your proxy has saturated, or one where it scores near random, contributes pure noise to your suite average. *Too-small eval sets:* a benchmark with 300 examples has a binomial standard error around ±2.5 points at 50% accuracy — a single such eval can swamp a real effect.

**Fix:** quantify the noise floor before you read any delta — run two or three seeds per configuration and treat the seed spread as the minimum detectable effect. Average over a *suite* of evals rather than reading any single one, because independent noise shrinks under averaging. Drop saturated and near-random evals from the suite. Prefer evals with thousands of examples, or aggregate small evals into a composite before reading. And size your effect against the band: if you need to detect a +0.3 effect and your noise floor is ±0.5, you do not have a measurement problem, you have a *power* problem — you need more seeds, more tokens, or a bigger proxy, not a hotter take on the number you have.

### Symptom: the proxy says better, the full model says worse

Your 1.6B ablation clearly favors recipe B. You apply it to the real 8B run and the 8B model is no better, or worse.

**Root cause: the proxy-to-eval relationship does not transfer to the target scale.** Data-quality effects are *mostly* scale-transferable, but not always. A filter that helps a small, compute-starved model by concentrating the highest-density text can hurt a large model that had the capacity to learn from the diversity that filter removed — this is the compute-dependence of the optimal filter aggressiveness made painful. The proxy was measuring a real effect; the effect just had the opposite sign at scale.

**Fix:** validate the proxy against at least one larger run before you trust it for decisions at the target scale, especially for knobs (like filter aggressiveness) that the scaling laws predict are compute-dependent. Run the proxy at *two* small scales (say 0.4B and 1.6B) and check that the delta is stable or trending in a consistent direction across scale — a delta that flips sign between your two proxy scales is a loud warning that it will not transfer. When a knob is known to be compute-dependent, do not pick a single setting from one proxy scale; measure the trend and extrapolate, the way [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) prescribes.

### Symptom: optimizing the proxy keeps making the model worse

You build a tight loop that optimizes a cheap metric — drive down average perplexity, or push up average classifier score — and the metric keeps improving while your downstream evals stall and then decline. The harder you optimize the proxy, the worse the model.

**Root cause: Goodhart's law.** "When a measure becomes a target, it ceases to be a good measure." Every cheap proxy correlates with quality only over the range where you validated it; push past that range and you start selecting for the *difference* between the proxy and real quality. Optimize perplexity hard and you select for bland, reference-like text and strip the diversity the model needs. Optimize a Wikipedia-versus-web classifier hard and you delete code, dialogue, and forums. The proxy goes up because you are, by construction, maximizing it; the eval goes down because you are maximizing the wrong thing.

<figure class="blog-anim">
<svg viewBox="0 0 700 430" role="img" aria-label="Goodhart divergence: across optimization rounds the proxy metric keeps improving while the downstream eval rises, peaks, and then falls" style="width:100%;height:auto;max-width:760px">
<style>
.a2-axis{stroke:var(--text-secondary,#6b7280);stroke-width:2;fill:none}
.a2-proxy{stroke:var(--accent,#6366f1);stroke-width:3;fill:none}
.a2-eval{stroke:var(--text-primary,#1f2937);stroke-width:3;fill:none;stroke-dasharray:2 0}
.a2-peak{stroke:var(--text-secondary,#6b7280);stroke-width:1.5;stroke-dasharray:5 5}
.a2-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.a2-sub{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.a2-pdot{fill:var(--accent,#6366f1)}
.a2-edot{fill:var(--text-primary,#1f2937)}
@keyframes a2-pmove{0%{transform:translate(0px,0px)}20%{transform:translate(120px,-55px)}40%{transform:translate(240px,-115px)}60%{transform:translate(340px,-165px)}80%{transform:translate(440px,-205px)}100%{transform:translate(560px,-240px)}}
@keyframes a2-emove{0%{transform:translate(0px,0px)}20%{transform:translate(120px,-100px)}40%{transform:translate(240px,-170px)}60%{transform:translate(340px,-150px)}80%{transform:translate(440px,-75px)}100%{transform:translate(560px,20px)}}
.a2-pdot{animation:a2-pmove 11s linear infinite}
.a2-edot{animation:a2-emove 11s linear infinite}
@media (prefers-reduced-motion:reduce){.a2-pdot,.a2-edot{animation:none}}
</style>
<defs><marker id="a2-ah" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="var(--text-secondary,#6b7280)"/></marker></defs>
<line class="a2-axis" x1="70" y1="365" x2="665" y2="365" marker-end="url(#a2-ah)"/>
<line class="a2-axis" x1="70" y1="365" x2="70" y2="55" marker-end="url(#a2-ah)"/>
<text class="a2-sub" x="560" y="392">optimization rounds</text>
<text class="a2-sub" x="14" y="60">score</text>
<line class="a2-peak" x1="320" y1="140" x2="320" y2="365"/>
<text class="a2-sub" x="232" y="135">eval peaks, then falls</text>
<polyline class="a2-proxy" points="80,330 200,275 320,215 420,165 520,125 640,90"/>
<polyline class="a2-eval" points="80,310 200,210 320,140 420,160 520,235 640,330"/>
<rect x="86" y="64" width="16" height="10" fill="var(--accent,#6366f1)"/>
<text class="a2-lbl" x="110" y="74">proxy metric (keeps improving)</text>
<rect x="86" y="86" width="16" height="10" fill="var(--text-primary,#1f2937)"/>
<text class="a2-lbl" x="110" y="96">downstream eval (what we care about)</text>
<text class="a2-lbl" x="430" y="118">proxy up, eval down</text>
<text class="a2-lbl" x="430" y="138">= Goodhart</text>
<circle class="a2-pdot" cx="80" cy="330" r="9"/>
<circle class="a2-edot" cx="80" cy="310" r="9"/>
</svg>
<figcaption>Optimize the proxy hard enough and it decouples: the proxy keeps climbing while the downstream eval peaks and then declines. Stop at the eval peak, not the proxy peak.</figcaption>
</figure>

The animation is the shape of the trap. Early on the proxy and the eval rise together, so optimizing the proxy looks like free progress. Past the peak they diverge: the proxy keeps climbing because you keep maximizing it, and the eval turns down because you are now selecting for the proxy-quality gap. The disaster is that if you are only watching the proxy — which is the whole reason you used a cheap proxy — you never see the turn.

**Fix:** never optimize a proxy without periodically checking the real eval, and stop at the *eval* peak, not the proxy peak. Concretely: hold out an honest ablation as the referee, re-run it every few laps of proxy-optimization, and the moment the eval stops tracking the proxy, freeze the knob. Watch the proxy-to-eval correlation (the scatter from the previous section) and treat any drop in that correlation as the signature of Goodharting in progress. And bound the proxy's authority structurally — use the proxy to *pre-screen* candidates, but make the keep/revert decision on the ablation, so the cheap metric can never overrule the gold standard. The cure for Goodhart is not a better proxy; it is keeping the real measurement in the loop so the proxy can never run away unsupervised. This same dynamic — a clean training signal corrupted by over-optimization — shows up in label space too, where it manifests as silently learning from corrupted targets, dissected in [garbage in: finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise).

### Symptom: every filter "works" in the demo and nothing compounds

Each individual filter you add shows a small positive delta in isolation, but stacking all of them produces a corpus no better than two filters ago.

**Root cause: confounded, non-additive ablations.** You ablated each filter against the *raw* baseline rather than against the current champion, so you measured each filter's effect in a context that no longer exists once the other filters are applied. Filters interact: two filters that each remove the same junk are not additive, and a filter that helped on raw data can be redundant or harmful on already-cleaned data.

**Fix:** ablate each new knob against the *current best recipe*, not the original baseline, so each lap measures marginal value in the real context. Re-run the loop from the previous figure with the champion updated after every kept change. And periodically run a "leave-one-out" check on your accumulated filters — remove each one from the final recipe and re-ablate — to catch filters that earned their place against raw data but have since become dead weight. A curation pipeline should be pruned the way you prune features in a model: aggressively, against the current state, with a measurement for every cut.

## When to reach for an ablation, and when a cheap proxy is enough

The ladder exists because not every decision deserves the same measurement effort. Spending a six-hour proxy run to decide a heuristic threshold that affects 0.1% of documents is as much an engineering error as eyeballing a filter that will reshape your entire corpus. Calibrate the rung to the stakes.

**Reach for a full proxy ablation when:**

- The decision reshapes a large fraction of the corpus — filter aggressiveness, the dedup strategy, the mixing ratio between major data sources. These are the calls that move the downstream model by points, and they are worth the GPU-hours to get right.
- The cheap proxies disagree with each other, or disagree with your intuition. Disagreement is exactly when you cannot afford to guess.
- The knob is known to be compute-dependent (filter strength, epoch count, synthetic-data fraction), so a single proxy scale is not enough and you need the trend across scales.
- You are about to commit a change to a full-scale run whose cost dwarfs the ablation. If the real run costs a hundred times the ablation, the ablation is rounding error and you should always run it.
- The proxy-to-eval correlation for this knob is unvalidated or has been pushed toward an extreme, so you cannot trust the cheap signal here.

**Skip the ablation and trust a cheap rung when:**

- It is the first-pass cleanup that everyone runs — basic length, symbol-ratio, and encoding filters. These have a large, well-established effect and re-litigating them per project is waste.
- The change is tiny in corpus impact and the cheap proxy strongly agrees with your prior. A threshold that touches a handful of documents is not worth a training run.
- You have already validated that a specific proxy has a high, stable rank correlation with your eval in this regime, and you are operating inside that validated regime. A trustworthy proxy is what lets you pre-screen dozens of candidates cheaply — that is its job.
- You are exploring breadth, not committing depth. Use the cheap rungs to generate and rank a long list of candidates, then spend ablations only on the short list.

The meta-rule underneath both lists: **let the cheap rungs propose and the expensive rung decide.** Heuristics, perplexity, classifiers, and judges are fast enough to explore the space; the ablation is honest enough to make the call. A team that ablates everything moves too slowly to ship; a team that ablates nothing ships corpora it cannot defend. The whole skill of measuring data quality is knowing, for each decision in front of you, exactly how high up the ladder you need to climb — and then climbing no higher.

## Further reading

- [Data quality is a scaling axis: filtering, dedup, and the quality-quantity tradeoff](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) — why the optimal filter aggressiveness is compute-dependent, and the data-quality scaling laws behind it.
- [Look at your data before you train](/blog/machine-learning/debugging-training/look-at-your-data-before-you-train) — the manual-inspection discipline that should sit alongside every automated metric in this post.
- [Garbage in: finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise) — the same "measure before you trust" mindset, applied to corrupted targets rather than corrupted inputs.
- [Classifier and perplexity quality filtering](/blog/machine-learning/training-data/classifier-and-perplexity-quality-filtering) — the implementation deep-dive on Rungs 2 and 3 of the ladder.
- [Data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning) — selecting and pruning at scale, where proxy reliability under aggressive filtering becomes the central problem.
- Penedo et al., *The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale* (2024) — the case study, with its ablation-everything methodology and early-signal benchmark selection.
- Li et al., *DataComp-LM: In search of the next generation of training sets for language models* (2024) — ablation as a shared, standardized benchmark for comparing data recipes.
