---
title: "Language Identification and Heuristic Quality Filters: The First Sieve"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "The cheapest, first-run pass over a pretraining corpus is language ID plus a stack of shape-based heuristics. A principal-engineer tour of how fastText langid actually works, the Gopher/C4/RefinedWeb rule stacks, a worked survival-rate scenario, the collateral damage on code and low-resource languages, and a troubleshooting guide for when your non-English data quietly vanishes."
tags: ["training-data", "data-curation", "language-identification", "fasttext", "heuristic-filtering", "gopher", "refinedweb", "c4", "llm", "pretraining", "data-quality", "low-resource-languages"]
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 35
---

Once the bytes are on disk and the boilerplate is stripped, you have a mountain of extracted text and a decision to make about the order in which you clean it. Everyone gets the *list* of cleaning stages roughly right — language ID, heuristic filters, deduplication, a learned quality classifier, a perplexity pass, mixture weighting. Almost nobody thinks hard about the *order*, and the order is where most of the money is. Run your expensive stages first and you will spend a fortune fingerprinting billions of documents you were about to throw away anyway. Run them last, behind a couple of cheap sieves, and you shrink the corpus by half or more before a single MinHash signature is computed.

This post is about the two cheapest sieves, the ones that run first on absolutely everything: **language identification** and **heuristic quality filters**. They are unglamorous. They are a few hundred lines of code and one 126 MB model. They will never be the subject of a keynote. And they routinely decide whether your multilingual eval collapses, whether your code benchmarks tank, and whether the corpus you *thought* was 40% non-English quietly comes out 99% English on the other side. This is the back half of the story that started with [text extraction and boilerplate removal](/blog/machine-learning/training-data/text-extraction-and-boilerplate-removal): extraction decides what text exists, the first sieve decides what text survives.

![A six-stage pipeline from extracted text through language ID and heuristic filters to dedup, classifier, and mixture, with survival percentages falling at each stage](/imgs/blogs/language-identification-and-heuristic-quality-filters-1.webp)

The diagram above is the mental model for the whole post. Read it left to right, top row then bottom: a billion extracted documents enter, language ID and the heuristic stack (the two blue boxes) run first because they are the cheapest per-document checks in the entire pipeline, and by the time the survivors reach deduplication the corpus is already down to roughly a third of what came in. Every stage after the first sieve is more expensive per document, so every document the sieve removes is compounding savings. The percentages are illustrative — they move with your source and your thresholds — but the *shape* is universal: the cheap filters go first and do the bulk removal.

## 1. Why the first sieve runs first

The senior rule of thumb is blunt: **order your filters by cost-per-document, cheapest first, and only spend expensive compute on documents that already survived the cheap checks.** This is not a data-quality principle, it is an operations principle, and it happens to also be good for quality because the cheap filters remove the most obviously broken documents.

Put concrete numbers on "cost." Here is the per-document cost of each stage on a single CPU core, order of magnitude:

| Stage | What it does per document | Rough cost / doc | Runs on |
| --- | --- | --- | --- |
| Language ID (fastText) | hash char n-grams, one matrix-vector product | ~0.1–1 ms | everything |
| Heuristic filters | count words, symbols, lines; a dozen comparisons | ~0.1–1 ms | everything |
| Exact / fuzzy dedup | MinHash signature + LSH bucket, or suffix-array lookup | ~1–10 ms + global index | langid + heuristic survivors |
| Quality classifier | tokenize + linear/small-net inference on features or embeddings | ~1–10 ms | dedup survivors |
| Perplexity filter | full forward pass through a KenLM or small LM | ~10–100 ms | classifier survivors |

The spread is two to three orders of magnitude. A perplexity pass over the *raw* extracted corpus is not just slow, it is wasteful: most of what you would score is spam, navigation dumps, and wrong-language text that a 0.3 ms check would have killed. So the first sieve is where you pay the least and remove the most. It is also the only stage that runs, unconditionally, on 100% of the input — which means every bug in it is a bug that touches your entire corpus.

There is a second, quieter reason langid goes first: **almost every downstream stage assumes a known language.** Your quality classifier was trained on English features. Your stop-word heuristic uses English stop words. Your perplexity model is an English (or per-language) LM. Your tokenizer statistics are computed per language. If you do not partition by language up front, every one of those later stages silently misbehaves on the documents it was never designed for. Language ID is not just a filter; it is the *routing* decision that makes the rest of the pipeline coherent.

> The first sieve is the only code in your pipeline that runs on every byte you own. Treat its bugs as corpus-wide outages, because that is exactly what they are.

## 2. Language identification

### How n-gram language ID actually works

The instinct when you hear "language identification" is to picture a small neural network reading the text. For the tool everyone actually uses — fastText's `lid.176` model — that picture is wrong, and the truth is much cheaper and much more revealing about the failure modes.

![A dataflow graph: raw text fans out into character 2-, 3-, and 4-5-grams that hash into a shared bucket table, mean-pool into a 16-dim embedding, feed a linear softmax over 176 languages, and branch to keep or route based on a threshold](/imgs/blogs/language-identification-and-heuristic-quality-filters-2.webp)

fastText language ID is a **bag of character n-grams scored by a single linear layer.** Walk the graph above left to right. The document is chopped into overlapping character n-grams — typically lengths 2 through 5, so "the cat" yields `th`, `he`, `he `, `the`, and so on. Each n-gram is hashed into one row of a shared embedding table with a couple of million buckets (hashing keeps the vocabulary bounded, at the cost of occasional collisions). The n-gram vectors are averaged into a single low-dimensional document vector — 16 dimensions in `lid.176`, which is astonishingly small. That one vector goes through a linear layer and a softmax over the 176 language classes, producing a probability for each language. You take the top language and its probability, and a threshold decides whether you trust it.

That architecture explains every strength and weakness you will hit:

- **It is fast because it is shallow.** One hashed lookup per n-gram, one average, one matrix-vector product. A single core does hundreds to thousands of documents per second. There is no attention, no recurrence, no GPU.
- **It works because scripts and short character sequences are extremely language-discriminative.** The trigram `ção` screams Portuguese; `sch` leans German; `ый` is Russian. You do not need to understand a sentence to know its language, you need to recognize its texture.
- **It is calibrated only loosely.** The softmax probability is a useful *ranking* signal and a decent confidence proxy, but it is not a true posterior. Treat `0.92` as "quite sure," not as "92% of such documents are this language."
- **It falls apart on short text and mixed text**, for the exact reason it is fast: with only a handful of n-grams to average, one shared 16-dim vector cannot represent two languages at once, and a five-word string barely constrains the average at all.

The math is a one-liner. Let $v_g$ be the embedding of n-gram $g$, and let a document $d$ contain n-grams $G_d$. The document vector is the mean, and the score for language $\ell$ is a linear readout:

$$ \bar{v}_d = \frac{1}{|G_d|} \sum_{g \in G_d} v_g, \qquad p(\ell \mid d) = \operatorname{softmax}\!\big(W \bar{v}_d + b\big)_\ell $$

The averaging in the first equation is the whole ballgame. It is what makes the model fast, and it is what destroys it on code-switched documents: a paragraph that is 70% English and 30% Hindi produces an *average* vector that points somewhere between the two, and the softmax confidently returns whichever language won the tug-of-war, at a mediocre probability, with the minority language erased.

### fastText vs CLD3 vs langdetect

There are three tools you will meet in the wild. They make different tradeoffs, and picking the wrong one for your text shape is a common, quiet mistake.

| Tool | Method | Languages | Throughput | Short text | Practical notes |
| --- | --- | --- | --- | --- | --- |
| **fastText `lid.176`** | hashed char n-grams + linear softmax | 176 | very high (100s–1000s/s/core) | weak below ~20 chars | de facto standard for pretraining; 126 MB; single dependency; returns a probability |
| **CLD3 (`gcld3`)** | small neural net (n-gram embeddings + FFN) | ~107 | high | better on short strings | Chrome's detector; good for UI-length text; slightly awkward Python bindings |
| **langdetect** | Naive Bayes over char n-grams | ~55 | low (pure Python) | poor | port of Nakatani's `language-detection`; **nondeterministic unless you seed it**; fine for scripts, not for TB-scale |

For a pretraining corpus the answer is almost always fastText: it is the fastest, covers the most languages, and is the one every published pipeline (CCNet, RefinedWeb, FineWeb) standardized on, which matters because it means your language distribution is comparable to theirs. Reach for CLD3 when your unit of text is short — sentences, titles, queries — where fastText's short-text weakness bites. Reach for `langdetect` essentially never at scale; it is a fine 20-line-script tool and a performance disaster on a billion documents. If you do use it, `DetectorFactory.seed = 0` or your results are not reproducible.

Here is the fastText usage everyone copies, with the two footguns called out:

```python
import fasttext

# lid.176.bin is the official 176-language model (~126 MB).
# https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
model = fasttext.load_model("lid.176.bin")

def detect_language(text: str) -> tuple[str, float]:
    # Footgun 1: fastText treats '\n' as a document separator. If you pass a
    # multi-line doc, predict() only sees the first line. Flatten first.
    cleaned = text.replace("\n", " ").strip()
    # Footgun 2: predict() on an empty string raises. Guard it.
    if not cleaned:
        return ("unknown", 0.0)
    labels, probs = model.predict(cleaned, k=1)
    lang = labels[0].removeprefix("__label__")
    return lang, float(probs[0])

print(detect_language("The quick brown fox jumps over the lazy dog."))
# ('en', 0.98)
print(detect_language("Le renard brun rapide saute par-dessus le chien."))
# ('fr', 0.99)
print(detect_language("pip install torch"))     # short + code-ish
# ('en', 0.41)   <-- low confidence; this is the interesting regime
```

That last line is the whole reason the next few sections exist. A short, code-flavored string gets a real language label at a confidence of `0.41`. What you do with that number — where you put the cut — is one of the highest-leverage decisions in the entire cleaning pipeline.

### The confidence distribution and where to cut

Run fastText over a real extracted corpus and histogram the top-language probability, and you get a strongly right-skewed distribution: a tall pile near `1.0` (clean, unambiguous prose in one language) and a short tail sloping down to `0.0` (short strings, mixed languages, markup, code, and genuine garbage). The keep-threshold is a vertical line you draw across that histogram.

![A histogram of fastText confidence scores, right-skewed with most mass near 1.0, a dashed cut line at 0.6 separating a red rejected left tail from a green kept right region](/imgs/blogs/language-identification-and-heuristic-quality-filters-3.webp)

The figure shows the shape and the decision. Everything left of the cut is rejected; everything right is kept. Two things make this picture worth internalizing. First, because the distribution is right-skewed, a *reasonable* cut throws away only a small fraction of documents — the left tail is thin. Second, that thin tail is not uniform garbage: it is disproportionately short documents, code, markup, and — critically — every low-resource language your model is least sure about. The cut that looks harmless on the aggregate histogram can be catastrophic for a specific language, because a whole language can live entirely in that left tail.

The practical procedure is to *look at the histogram for each language you care about* and set the cut from the data, not from a default you copied. A single global number is where teams get burned, which is the subject of the next section.

### Per-language thresholds and the long tail

Here is the mistake, stated plainly: you download `lid.176`, you set `keep if lang == "en" and p >= 0.65` because that is what RefinedWeb used, you run it on your multilingual corpus, and you are surprised when the corpus comes out almost entirely English and the handful of Swahili and Telugu documents you had are gone. The threshold was fine. Applying *the same* threshold to every language was the bug.

![A five-row matrix of language tiers (English, Vietnamese, Swahili, code/markup, code-switched) against typical confidence, a good threshold, and the dominant failure mode for each](/imgs/blogs/language-identification-and-heuristic-quality-filters-4.webp)

The matrix lays out the tiers. High-resource languages like English sit near `0.95+` because the model has seen enormous amounts of them and their character texture is unambiguous — a high cut like `0.65` costs you almost nothing. Mid-resource languages sit lower; a language whose diacritics are frequently stripped by upstream extraction (so the very n-grams that identify it are missing) will produce systematically depressed confidences, and a `0.65` cut deletes a large fraction of perfectly good documents. Low-resource languages are worse still: the model is genuinely uncertain, its confidences cluster in the `0.3–0.6` band, and — the real trap — its errors are not random. A low-resource language often gets *misclassified as a nearby high-resource one*, or its documents scrape by at low confidence, so a global cut both deletes the true positives and lets in contaminants.

The fix is a per-language threshold table, informed by the per-language histogram:

```python
# Per-language keep thresholds. High-resource languages can afford a high cut;
# low-resource languages need a low cut or you delete the whole language.
LANG_THRESHOLDS = {
    "en": 0.65, "de": 0.60, "fr": 0.60, "es": 0.60,   # high-resource
    "vi": 0.45, "tr": 0.45, "id": 0.45,               # mid-resource
    "sw": 0.30, "te": 0.30, "yo": 0.25,               # low-resource
}
DEFAULT_THRESHOLD = 0.50

def keep_language(text: str, allowed: set[str]) -> bool:
    lang, p = detect_language(text)
    if lang not in allowed:
        return False
    return p >= LANG_THRESHOLDS.get(lang, DEFAULT_THRESHOLD)
```

The uncomfortable part: lowering the threshold for a low-resource language *also* lowers precision, so you keep more contamination. There is no threshold that is simultaneously high-recall and high-precision for a language the model barely knows. The tradeoff is a frontier you slide along, not a knob with a right answer:

<figure class="blog-anim">
<svg viewBox="0 0 720 460" role="img" aria-label="Precision rises and recall falls as the language-ID confidence threshold slides from low to high; the two curves cross near the middle" style="width:100%;height:auto;max-width:820px">
<title>As the langid confidence threshold slides right, precision rises and recall falls; no single cut maximizes both.</title>
<style>
.pr-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.pr-grid{stroke:var(--border,#d1d5db);stroke-width:1;stroke-dasharray:3 5;opacity:.5}
.pr-prec{fill:none;stroke:var(--accent,#6366f1);stroke-width:3}
.pr-rec{fill:none;stroke:var(--danger,#ef4444);stroke-width:3}
.pr-dotp{fill:var(--accent,#6366f1)}
.pr-dotr{fill:var(--danger,#ef4444)}
.pr-mark{stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:5 4}
.pr-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.pr-tick{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.pr-plab{font:700 15px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
.pr-rlab{font:700 15px ui-sans-serif,system-ui;fill:var(--danger,#ef4444)}
@keyframes pr-slide{0%{transform:translateX(0)}25%{transform:translateX(112px)}50%{transform:translateX(224px)}75%{transform:translateX(336px)}100%{transform:translateX(448px)}}
@keyframes pr-ridep{0%{transform:translate(0,0)}25%{transform:translate(112px,-42px)}50%{transform:translate(224px,-80px)}75%{transform:translate(336px,-112px)}100%{transform:translate(448px,-134px)}}
@keyframes pr-rider{0%{transform:translate(0,0)}25%{transform:translate(112px,20px)}50%{transform:translate(224px,58px)}75%{transform:translate(336px,122px)}100%{transform:translate(448px,218px)}}
.pr-mv{animation:pr-slide 9s ease-in-out infinite alternate}
.pr-pd{animation:pr-ridep 9s ease-in-out infinite alternate}
.pr-rd{animation:pr-rider 9s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.pr-mv,.pr-pd,.pr-rd{animation:none}.pr-mv{transform:translateX(224px)}.pr-pd{transform:translate(224px,-80px)}.pr-rd{transform:translate(224px,58px)}}
</style>
<line class="pr-axis" x1="80" y1="60" x2="80" y2="380"/>
<line class="pr-axis" x1="80" y1="380" x2="660" y2="380"/>
<line class="pr-grid" x1="80" y1="124" x2="660" y2="124"/>
<line class="pr-grid" x1="80" y1="220" x2="660" y2="220"/>
<polyline class="pr-prec" points="136,204 248,162 360,124 472,92 584,70"/>
<polyline class="pr-rec" points="136,66 248,86 360,124 472,188 584,284"/>
<line class="pr-mark pr-mv" x1="136" y1="52" x2="136" y2="380"/>
<circle class="pr-dotp pr-pd" cx="136" cy="204" r="6"/>
<circle class="pr-dotr pr-rd" cx="136" cy="66" r="6"/>
<text class="pr-plab" x="592" y="66">precision</text>
<text class="pr-rlab" x="592" y="300">recall</text>
<text class="pr-tick" x="136" y="400">0.1</text>
<text class="pr-tick" x="360" y="400">0.5</text>
<text class="pr-tick" x="584" y="400">0.9</text>
<text class="pr-tick" x="60" y="128">1.0</text>
<text class="pr-tick" x="60" y="224">0.5</text>
<text class="pr-tick" x="60" y="384">0.0</text>
<text class="pr-lbl" x="250" y="432">langid confidence threshold</text>
</svg>
<figcaption>Sliding the keep-threshold trades recall for precision: a low cut keeps almost everything (high recall, low precision), a high cut keeps only confident documents (high precision, low recall), and the curves cross where neither is maximized — which is why a language the model barely knows has no clean cut at all.</figcaption>
</figure>

This is why the 2022 "Quality at a Glance" audit (a case study below) found that many low-resource sections of popular web corpora were majority garbage: the maintainers used one global, English-appropriate configuration, and the long tail paid for it. If low-resource coverage matters to you, langid confidence is not enough — you need a second, language-specific check downstream, which is exactly the job of the [classifier and perplexity-based filtering](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering) that runs after this sieve.

### Code-switching, short text, and markup

Three text shapes break n-gram langid, and all three are common in web data:

- **Short text.** A tweet-length string has too few n-grams for the average to stabilize. Below roughly 20 characters, treat any label as a coin flip. The fix is a minimum-length gate *before* langid — if a document is shorter than a threshold, either route it out or defer the language decision to a longer context.
- **Code-switching.** A document that mixes languages produces one blended vector and one confident-but-wrong label, erasing the minority language. If code-switched content matters (it does for many real languages and for chat/forum data), detect at the *segment* level — split on paragraphs or sentences and label each — rather than assigning one language to the whole document.
- **Markup and code.** HTML fragments, JSON, and source code are full of English-looking keywords (`function`, `return`, `class`) and get labeled English at low-to-medium confidence. If you keep them as "low-quality English," they poison your English distribution; if you drop them, you lose your code data. The right move is to *route by content type first* — send code and markup to a code-specific pipeline — rather than letting a language model adjudicate a language question that does not apply.

The throughline is that langid answers exactly one question — "what natural language is this?" — and quietly returns nonsense when the input is not a single natural language. Everything that is short, mixed, or non-linguistic needs to be handled *before* it reaches the classifier, not by trusting a low-confidence label.

## 3. The heuristic quality-filter stack

Language ID answers "what language." The heuristic stack answers a different question: "does this *look like* the kind of text we want, from its shape alone?" These are deliberately dumb rules over surface statistics — word counts, character ratios, line structure — and their power comes from being cheap enough to run on every survivor of langid. They are also, precisely because they are dumb, the largest source of collateral damage in the pipeline.

The cascade runs as a sequence of independent sieves, each removing a slice of what survived the previous one. Watch the survival rate fall stage by stage:

<figure class="blog-anim">
<svg viewBox="0 0 760 560" role="img" aria-label="A batch of documents passes through six stacked sieves; each stage cuts the surviving fraction, ending near 35 percent" style="width:100%;height:auto;max-width:820px">
<title>The heuristic cascade drops documents stage by stage; the blue survivor bar shrinks and the red dropped region grows at each sieve.</title>
<style>
.hc-track{fill:var(--danger-bg,#fde2e2);stroke:var(--border,#d1d5db);stroke-width:1.5}
.hc-bar{fill:var(--accent,#6366f1);transform-box:fill-box;transform-origin:left center}
.hc-lbl{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.hc-pct{font:700 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.hc-cap{font:600 14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.hc-ind{fill:var(--accent,#6366f1)}
.hc-sw-k{fill:var(--accent,#6366f1)}
.hc-sw-d{fill:var(--danger-bg,#fde2e2);stroke:var(--border,#d1d5db);stroke-width:1}
@keyframes hc0{0%,10%{transform:scaleX(1)}16%,90%{transform:scaleX(.80)}100%{transform:scaleX(1)}}
@keyframes hc1{0%,22%{transform:scaleX(1)}28%,90%{transform:scaleX(.61)}100%{transform:scaleX(1)}}
@keyframes hc2{0%,34%{transform:scaleX(1)}40%,90%{transform:scaleX(.52)}100%{transform:scaleX(1)}}
@keyframes hc3{0%,46%{transform:scaleX(1)}52%,90%{transform:scaleX(.47)}100%{transform:scaleX(1)}}
@keyframes hc4{0%,58%{transform:scaleX(1)}64%,90%{transform:scaleX(.41)}100%{transform:scaleX(1)}}
@keyframes hc5{0%,70%{transform:scaleX(1)}76%,90%{transform:scaleX(.35)}100%{transform:scaleX(1)}}
@keyframes hcind{0%,12%{transform:translateY(0)}13%,24%{transform:translateY(80px)}25%,36%{transform:translateY(160px)}37%,48%{transform:translateY(240px)}49%,60%{transform:translateY(320px)}61%,90%{transform:translateY(400px)}100%{transform:translateY(0)}}
.b0{animation:hc0 11s ease-in-out infinite}
.b1{animation:hc1 11s ease-in-out infinite}
.b2{animation:hc2 11s ease-in-out infinite}
.b3{animation:hc3 11s ease-in-out infinite}
.b4{animation:hc4 11s ease-in-out infinite}
.b5{animation:hc5 11s ease-in-out infinite}
.hc-ind{animation:hcind 11s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.b0,.b1,.b2,.b3,.b4,.b5,.hc-ind{animation:none}.b0{transform:scaleX(.80)}.b1{transform:scaleX(.61)}.b2{transform:scaleX(.52)}.b3{transform:scaleX(.47)}.b4{transform:scaleX(.41)}.b5{transform:scaleX(.35)}.hc-ind{transform:translateY(400px)}}
</style>
<text class="hc-cap" x="20" y="22">batch of 1,000 extracted documents enters the top</text>
<rect class="hc-sw-k" x="470" y="10" width="16" height="14" rx="3"/>
<text class="hc-cap" x="492" y="22">kept</text>
<rect class="hc-sw-d" x="560" y="10" width="16" height="14" rx="3"/>
<text class="hc-cap" x="582" y="22">dropped</text>
<path class="hc-ind" d="M228,56 L246,64 L228,72 Z"/>
<text class="hc-lbl" x="20" y="70">language ID</text>
<rect class="hc-track" x="250" y="40" width="360" height="48" rx="6"/>
<rect class="hc-bar b0" x="250" y="40" width="360" height="48" rx="6"/>
<text class="hc-pct" x="628" y="70">80%</text>
<text class="hc-lbl" x="20" y="150">symbol ratio</text>
<rect class="hc-track" x="250" y="120" width="360" height="48" rx="6"/>
<rect class="hc-bar b1" x="250" y="120" width="360" height="48" rx="6"/>
<text class="hc-pct" x="628" y="150">61%</text>
<text class="hc-lbl" x="20" y="230">mean word length</text>
<rect class="hc-track" x="250" y="200" width="360" height="48" rx="6"/>
<rect class="hc-bar b2" x="250" y="200" width="360" height="48" rx="6"/>
<text class="hc-pct" x="628" y="230">52%</text>
<text class="hc-lbl" x="20" y="310">bullet / ellipsis</text>
<rect class="hc-track" x="250" y="280" width="360" height="48" rx="6"/>
<rect class="hc-bar b3" x="250" y="280" width="360" height="48" rx="6"/>
<text class="hc-pct" x="628" y="310">47%</text>
<text class="hc-lbl" x="20" y="390">stop-word check</text>
<rect class="hc-track" x="250" y="360" width="360" height="48" rx="6"/>
<rect class="hc-bar b4" x="250" y="360" width="360" height="48" rx="6"/>
<text class="hc-pct" x="628" y="390">41%</text>
<text class="hc-lbl" x="20" y="470">dup-line fraction</text>
<rect class="hc-track" x="250" y="440" width="360" height="48" rx="6"/>
<rect class="hc-bar b5" x="250" y="440" width="360" height="48" rx="6"/>
<text class="hc-pct" x="628" y="470">35%</text>
<text class="hc-cap" x="20" y="520">each sieve is cumulative: the blue survivor is the fraction of the original batch still alive</text>
</svg>
<figcaption>The heuristic cascade applied stage by stage: language ID keeps ~80%, and each successive shape-based rule trims more, leaving ~35% of the original batch after six sieves.</figcaption>
</figure>

### The Gopher / MassiveText rules

The canonical heuristic set comes from DeepMind's Gopher paper (Rae et al., 2021) and its MassiveText corpus. These are document-shape statistics, and each one targets a specific failure mode of web text. Here is the set, with what each rule kills and what it wrongly takes with it:

- **Word count in `[50, 100000]`.** Kills near-empty stubs and pathological mega-documents (usually concatenated dumps). *Collateral:* legitimately short pages — a good definition, a haiku, a focused Q&A answer — die at the low end.
- **Mean word length in `[3, 10]` characters.** Kills two shapes at once: text that is mostly single characters (spaced-out gibberish, tables rendered as columns) at the low end, and text with no spaces (URLs, concatenated tokens, some agglutinative-language content) at the high end. *Collateral:* languages with genuinely long average words, and heavily hyphenated technical prose.
- **Symbol-to-word ratio below `0.10`** for `#` and `...`. Kills documents that are mostly hashes (spam, hashtag walls) or mostly ellipses (truncated listing pages, "read more..." rails). *Collateral:* documents with legitimately heavy `#` — Markdown with many headings, some code, C preprocessor directives.
- **At least 80% of words contain an alphabetic character.** Kills number dumps, ASCII tables, log files, price lists. *Collateral:* data-heavy tables, statistical content, sports results — sometimes exactly what you want.
- **At least two common stop words present** (`the, be, to, of, and, that, have, with`). A shockingly effective English fluency check: real prose contains function words; keyword spam and machine-generated lists do not. *Collateral:* this is an **English rule wearing a language-agnostic costume** — apply it to non-English text and you delete the entire language, because German or Vietnamese prose contains none of these tokens.
- **Fewer than 90% of lines start with a bullet.** Kills navigation menus and link farms rendered as bullet lists. *Collateral:* genuine list-heavy content — recipes, step-by-step instructions, changelogs.
- **Fewer than 30% of lines end with an ellipsis.** Kills teaser/listing pages ("Top 10 ways to ... Read more ...").
- **Repetition filters:** duplicate-line fraction, duplicate-paragraph fraction, and top-n-gram character fractions all below thresholds. Kills the single most common web pathology — the same phrase or line repeated dozens of times (SEO stuffing, broken templates, chat logs). *Collateral:* poetry with refrains, legal boilerplate that is legitimately repetitive, song lyrics.

Notice the pattern: every rule is a sharp knife aimed at a real pathology, and every rule has an edge that cuts good documents. The rules are individually defensible and collectively a blunt instrument. Here is a faithful, runnable implementation of the core set:

```python
import re
from dataclasses import dataclass

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
BULLETS = ("•", "-", "*", "‣", "●", "▪")

@dataclass
class Verdict:
    keep: bool
    reason: str = "ok"

def gopher_filter(text: str) -> Verdict:
    words = text.split()
    n = len(words)
    if n < 50 or n > 100_000:
        return Verdict(False, "word_count")

    mean_len = sum(len(w) for w in words) / n
    if mean_len < 3 or mean_len > 10:
        return Verdict(False, "mean_word_length")

    # Symbol-to-word ratio: '#' and '...' are the classic Gopher symbols.
    if (text.count("#") / n) > 0.10 or (text.count("...") / n) > 0.10:
        return Verdict(False, "symbol_to_word")

    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_words / n < 0.80:
        return Verdict(False, "alpha_fraction")

    # WARNING: this is an English fluency check. Do NOT apply it to other
    # languages without swapping in that language's stop-word list.
    present = STOP_WORDS.intersection(w.lower() for w in words)
    if len(present) < 2:
        return Verdict(False, "stop_words")

    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines:
        bullet = sum(1 for ln in lines if ln.lstrip().startswith(BULLETS))
        if bullet / len(lines) > 0.90:
            return Verdict(False, "bullet_ratio")
        ellipsis = sum(1 for ln in lines if ln.rstrip().endswith("..."))
        if ellipsis / len(lines) > 0.30:
            return Verdict(False, "ellipsis_ratio")
        dup_line_frac = 1 - len(set(lines)) / len(lines)
        if dup_line_frac > 0.30:
            return Verdict(False, "duplicate_lines")

    return Verdict(True)
```

### The C4 rules

C4 (Raffel et al., 2020, the T5 corpus) attacks the same problem from a different angle: instead of document-shape statistics, it uses **line-level surface cues**. The philosophy is "clean web text ends in punctuation and does not contain code or placeholder junk."

- **Keep only lines ending in terminal punctuation** (`.`, `!`, `?`, or a closing quote). This is the signature C4 rule. It works because prose sentences end in punctuation and boilerplate (menu items, button labels, table cells) does not. *Collateral:* it is brutal to anything not written as full sentences — code, tables, lists, math, headings all get shredded line by line.
- **Drop pages with fewer than 5 sentences; drop lines with fewer than 3 words.** Removes thin, fragmentary pages.
- **Drop any page containing a word from a blocklist** (the "List of Dirty, Naughty, Obscene, and Otherwise Bad Words"). *Collateral:* famously over-broad — it deletes medical, sex-education, and LGBTQ+ content wholesale, a well-documented harm of the C4 blocklist.
- **Drop any line with "lorem ipsum."** Kills template placeholder text.
- **Drop any page containing a curly brace `{`.** A proxy for "this page has code or JavaScript." *Collateral:* deletes essentially all pages that discuss programming — a catastrophic rule if you want a code-capable model.
- **Drop any line containing "javascript."** Targets "please enable JavaScript" notices. *Collateral:* deletes legitimate discussion of the JavaScript language.
- **Three-sentence-span dedup:** remove all but one occurrence of any duplicated three-sentence span.

The C4 rules are a masterclass in how a rule that is *right on average* can be *disastrous for a subpopulation*. The `{` rule alone is why models trained on vanilla C4 were notoriously weak at code: the corpus had been specifically, if unintentionally, purged of it.

### RefinedWeb's additions

RefinedWeb (Penedo et al., 2023, the corpus behind Falcon) is the modern synthesis. Its thesis was provocative: **properly filtered and deduplicated web data alone can match or beat curated corpora** — no Wikipedia-and-books special-casing required. To get there it combined the best of both stacks and added its own:

- **fastText `lid.176` for language**, keeping English at a `0.65` confidence threshold. This is the source of the number everyone copies.
- **A tuned subset of the Gopher quality and repetition filters** — it kept the document-shape statistics that generalize and dropped or re-tuned the ones that were too aggressive.
- **Line-wise filtering:** a corrections pass that removes individual bad lines (residual navigation, social-media counters, short boilerplate lines) rather than dropping whole documents — a scalpel where Gopher and C4 use a hatchet.
- **Aggressive deduplication:** exact substring dedup (suffix array) *plus* fuzzy dedup (MinHash), which is a separate and heavier stage covered in [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale).
- **URL filtering upstream:** a blocklist and a soft-scoring of URLs to drop adult and spam domains before extraction even runs.

The measured effect is the headline: from raw CommonCrawl, the full RefinedWeb pipeline kept only about **11% of documents**, and a model trained on the survivors matched models trained on curated data. The lesson is not "filter more," it is "filter *in the right order* with rules tuned by ablation."

![A seven-row matrix comparing Gopher, C4, and RefinedWeb across rule families: shape statistics, line-level cues, and dedup, showing which stack applies each rule](/imgs/blogs/language-identification-and-heuristic-quality-filters-5.webp)

The comparison matrix makes the division of labor concrete. Gopher owns the document-shape column (apply, apply, apply down the shape rules); C4 skips those and owns the line-level surface column (terminal punctuation, `lorem`/`{`/`javascript`); and RefinedWeb keeps most of Gopher's shape rules, adopts C4's most useful line-level ideas, and upgrades dedup from document-level to line-plus-URL. The three stacks are not competitors — the production move is to compose them, which is exactly what modern corpora like FineWeb do.

### Collateral damage, made concrete

The abstract warning "these rules have false positives" does not land until you watch a specific good document die. Here is the canonical one:

![A before-after figure: an arXiv theorem page seen by the rules as symbol/word 0.34, bullet fraction 0.55, verdict DROP, versus its reality as theorem-and-proof prose, equations, and numbered steps that should be exempted](/imgs/blogs/language-identification-and-heuristic-quality-filters-6.webp)

An arXiv page with a theorem and its proof is exactly the high-value technical text a strong model needs. To the shape-based rules it reads as spam: the equations blow up the symbol-to-word ratio past `0.10`, the numbered proof steps blow past the bullet-line fraction, and mean word length is skewed by variable names and operators. Every individual rule fires correctly on its own statistic; the *composition* deletes a great document. The same story repeats for source code (curly braces, low stop-word density, high symbol ratio), data tables (low alphabetic fraction), and non-English prose (no English stop words). The fix is not to weaken the rules for everyone — that lets the real spam back in — but to **route by content type first** and apply shape rules only to the population they were designed for: general web prose. Send code to a code pipeline, math to a math-aware pipeline, and only then let the Gopher stack loose on what remains.

## 4. Worked scenario: running the cascade on a sample

Numbers make this real. Take a sample of 1,000 English-target documents freshly extracted from CommonCrawl and run the first sieve end to end, tallying survival after each rule. The driver is small:

```python
from collections import Counter

def cascade_report(docs: list[str], threshold: float = 0.65):
    n0 = len(docs)
    kept, dropped = [], Counter()
    for d in docs:
        lang, p = detect_language(d)
        if lang != "en" or p < threshold:
            dropped["langid"] += 1
            continue
        v = gopher_filter(d)
        if not v.keep:
            dropped[v.reason] += 1
            continue
        kept.append(d)

    print(f"in={n0}  kept={len(kept)} ({len(kept)/n0:.0%})")
    running = n0
    for reason in ["langid", "symbol_to_word", "mean_word_length",
                   "bullet_ratio", "ellipsis_ratio", "stop_words",
                   "duplicate_lines"]:
        running -= dropped.get(reason, 0)
        print(f"  after {reason:18s} survive {running:4d} "
              f"({running/n0:.0%})   [-{dropped.get(reason,0)}]")
    return kept
```

A representative run produces the survival curve below — the same curve the animated cascade above shows:

| After rule | Survivors | Survival | Killed this stage | What it removed |
| --- | --- | --- | --- | --- |
| (input) | 1000 | 100% | — | — |
| Language ID (`en`, `p >= 0.65`) | 800 | 80% | 200 | wrong-language, short, mixed, markup |
| Symbol-to-word ratio | 610 | 61% | 190 | hashtag walls, ellipsis listing pages, some code |
| Mean word length | 520 | 52% | 90 | spaced-out gibberish, URL dumps |
| Bullet / ellipsis ratio | 470 | 47% | 50 | nav menus, teaser rails |
| Stop-word presence | 410 | 41% | 60 | keyword spam, machine lists |
| Duplicate-line fraction | 350 | 35% | 60 | SEO stuffing, broken templates |

Roughly **two-thirds of the extracted sample is gone** after a stage that costs well under a millisecond per document, and every removed document is one you did not pay to fingerprint, classify, or score. That is the operational win.

Now the honest part — the false positives. Sample the *killed* pile and you will find good documents in it. Three real examples from a run like this:

1. **A dense statistics page** — a table of census figures with short surrounding prose — killed by `alpha_fraction` because most of its "words" were numbers. It was exactly the kind of factual, quantitative content you want.
2. **A programming tutorial** killed by `symbol_to_word`: the inline code, operators, and `#`-comments pushed the symbol ratio past `0.10`. The prose around the code was excellent.
3. **A recipe** killed by `bullet_ratio`: it was a legitimate numbered ingredient-and-step list, over 90% of lines starting with a bullet-like marker. Perfectly good instructional text, indistinguishable *by shape* from a link farm.

None of these three are bugs in any single rule. They are the *cost of using shape as a proxy for quality*. The only real defenses are content-type routing (so code and data never see the prose rules), threshold tuning by ablation (so you buy back recall where it is cheap), and accepting that some collateral damage is the price of a cheap, high-throughput first pass — you recover diversity later by *sourcing* the killed content types deliberately rather than by weakening the sieve.

## Case studies from production

### 1. Gopher / MassiveText — the rules everyone inherited

DeepMind's Gopher (Rae et al., 2021) is where the heuristic vocabulary was standardized. Its MassiveText corpus started from a large multi-source crawl and applied the quality-filter set enumerated above plus a family of repetition filters. The documented effect was twofold: the quality filters removed low-fluency and machine-generated pages, and the repetition filters removed the single most damaging web pathology — documents with heavily duplicated lines, paragraphs, or n-grams, which the paper showed degrade downstream performance out of proportion to their frequency. The lasting contribution was not any one threshold but the *framing*: quality as a set of cheap, interpretable, document-shape statistics you can compute in a single pass, each with a knob you can ablate. Every stack since — C4-derived, RefinedWeb, FineWeb, Dolma — is a variation on this template.

### 2. RefinedWeb / Falcon — filtering hard enough to skip curation

RefinedWeb (Penedo et al., 2023) tested a strong claim: that web data, filtered and deduplicated well enough, needs no curated supplements. Starting from CommonCrawl it applied URL filtering, `trafilatura` extraction, fastText langid at `0.65`, a tuned Gopher-style heuristic stack, line-wise corrections, and heavy dedup — surviving to roughly 11% of input. Falcon models trained on the result were competitive with models trained on curated mixes. The operational lessons that generalize: (a) langid and heuristics belong first and remove the bulk cheaply; (b) rules must be *tuned by ablation*, not copied — RefinedWeb explicitly dropped Gopher rules that hurt; (c) line-wise filtering (scalpel) beats whole-document dropping (hatchet) for boilerplate that is mixed into otherwise-good pages.

### 3. CCNet — langid at CommonCrawl scale

CCNet (Wenzek et al., 2020) is the reference design for the language-ID half of the sieve at scale. It ran fastText `lid.176` over CommonCrawl to partition documents by language, then applied a KenLM perplexity filter *per language* to bucket documents into head/middle/tail quality tiers. CCNet is why fastText became the default: it demonstrated that a single cheap linear classifier could produce reliable, comparable per-language partitions across the entire crawl, and it fed a generation of multilingual models (including the data behind XLM-R). The design insight worth stealing is the ordering — langid first to *route*, then a per-language quality model — which is exactly the "partition then filter" pattern that keeps every downstream English-specific rule from misfiring on non-English text.

### 4. C4 / T5 — when a blunt rule shapes a whole model

C4 (Raffel et al., 2020) is the cautionary case. Its line-level rules were reasonable heuristics, but two of them had outsized, unintended consequences. The curly-brace rule (`{` → drop page) silently removed nearly all programming content, which is a large part of why vanilla-C4 models were weak at code and why later corpora had to deliberately re-source it. The bad-words blocklist removed medical, sexual-health, and LGBTQ+ content far beyond its intent, a harm documented in the 2021 "Documenting the English Colossal Clean Crawled Corpus" audit. The lesson is not that C4 was careless — it was a landmark corpus — but that **a rule right on average can be catastrophic for a subpopulation**, and the only way to catch it is to audit the *removed* pile by category, not just eyeball the survivors.

### 5. "Quality at a Glance" — the low-resource langid catastrophe

Kreutzer et al. (2022) audited the actual sentences in popular multilingual web corpora (mC4, OSCAR, CCAligned, ParaCrawl, WikiMatrix) across 200-plus languages, with native-speaker raters. The findings were grim for the long tail: many low-resource language subsets were *majority* misclassified, non-linguistic, or wrong-language content — some had a large fraction of sentences that were not the labeled language at all, and a few were effectively unusable. The root cause was exactly the failure mode from Section 2: n-gram langid is genuinely uncertain on low-resource languages, its errors are systematic (low-resource content gets mislabeled as related high-resource languages, and vice versa), and a single global configuration cannot serve both the head and the tail. This audit is the strongest empirical argument for per-language thresholds, native-speaker spot-checks, and treating any low-resource subset produced by langid alone as suspect until verified.

### 6. FineWeb — re-deriving the stack with ablations

FineWeb (Penedo et al., 2024) revisited the whole first sieve with disciplined ablations: for each candidate filter, train a small model on data with and without it and keep only the rules that measurably help. The result largely re-confirmed the Gopher/C4/RefinedWeb heuristics but with evidence for each threshold rather than inheritance, plus a handful of new custom filters found by inspecting what survived. The meta-lesson is the one to carry into your own pipeline: **treat every heuristic as a hypothesis and test it with a small-model ablation.** Copied thresholds are a starting point, not an answer; the right cut for your source, your extractor, and your target languages is an empirical question you can afford to answer at small scale. This ablation discipline is the same one argued for in [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality).

## Troubleshooting

The first sieve fails quietly. Its bugs do not throw exceptions; they change the *composition* of a corpus, and you find out weeks later when an eval regresses. Here is the field guide, symptom to fix.

### Symptom: a whole low-resource language (or code) is nearly gone

**Detection.** Produce a per-language survival table: for each language, count documents in the input and after each sieve stage. A language that drops to near-zero after langid or after the stop-word rule is the tell. For code, watch HumanEval/MBPP; for a language, watch its downstream eval — but the survival table catches it *before* training.

**Root cause.** Almost always one of two things: a global confidence threshold that is too high for a low-resource language, or an English-specific heuristic (the stop-word rule, terminal-punctuation, curly-brace) applied to text it was never meant for.

**Fix.** Per-language thresholds from the per-language histogram (Section 2). Route code and markup out *before* the prose heuristics run. Swap the English stop-word list for the target language's, or disable that rule for non-English partitions. Re-run the survival table and confirm the language recovers.

### Symptom: langid returns confident-but-wrong labels

**Detection.** Sample documents where the assigned language disagrees with a second detector (run CLD3 as an oracle on a sample and measure agreement), or sample low-confidence documents and eyeball them. Bucket the disagreements by document length — you will almost always see the errors concentrated in short and mixed documents.

**Root cause.** Short text (too few n-grams to stabilize the average), code-switching (blended vector, minority language erased), or markup/code (English-looking keywords) — the three shapes from Section 2.

**Fix.** Add a minimum-length gate before langid (defer or drop very short docs). Detect at segment level for known code-switched sources. Strip markup and route code out before langid sees it. Never treat a label below your per-language threshold as reliable — that is what the threshold is *for*.

### Symptom: the threshold is impossible to set — junk if low, good data gone if high

**Detection.** Plot the confidence histogram for the language in question (the figure in Section 2). If it is cleanly bimodal, a good cut exists between the modes. If it is a smear with no valley, no single threshold will cleanly separate good from bad.

**Root cause.** For high-resource languages the distribution is bimodal and the cut is easy. For low-resource languages it is often a smear, because the model's uncertainty and the genuine junk overlap in the same confidence band — there is no clean separator.

**Fix.** For bimodal cases, set the cut in the valley and move on. For smeared cases, accept that langid confidence is insufficient and add a *second* signal downstream — a per-language perplexity or classifier filter (see [classifier and perplexity-based quality filtering](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering)) — rather than torturing one threshold to do two jobs.

### Symptom: "my multilingual corpus came out 99% English"

**Detection.** A language histogram of the *output*. If the input was 40% non-English and the output is 99% English, the sieve ate your other languages. Compare the input and output language distributions side by side — the discrepancy is the bug.

**Root cause.** The classic version is an accidental `lang == "en"` filter left in from an English-only experiment, or a global English threshold applied to everything, or the English stop-word rule silently deleting every non-English document. All three funnel the corpus toward English.

**Fix.** Make the allowed-language set explicit and assert on it. Use per-language thresholds. Gate every language-specific heuristic on the detected language so the English rules only touch English. Add a regression test: run the sieve on a fixed multilingual fixture and assert the output distribution matches expectations within a tolerance — this is the single highest-value test you can write for a data pipeline, and it is the same class of silent corruption tracked in [garbage in: finding label noise](/blog/machine-learning/debugging-training/garbage-in-finding-label-noise).

### Symptom: throughput collapses on the langid stage

**Detection.** The langid stage dominates wall-clock even though it is "cheap." Profile it — you will usually find the model being reloaded per document, or per-document Python overhead swamping the actual inference.

**Root cause.** Loading `lid.176.bin` inside the per-document function instead of once per worker; or calling `predict` one document at a time with heavy Python glue; or not flattening newlines, so fastText silently scores only the first line and you re-run it.

**Fix.** Load the model once per process (module-level or a worker initializer). Batch where the binding allows. Flatten newlines before `predict`. These are the difference between hundreds and thousands of documents per second per core.

## When to reach for the first sieve — and when not to

**Reach for language ID and heuristic filters when:**

- You are building a **pretraining corpus from web-scale, mixed-quality sources** (CommonCrawl, a broad crawl, scraped forums). This is exactly what they are for.
- You need to **partition by language** before any per-language stage — quality classifier, perplexity, tokenizer stats. Langid is the routing decision that makes everything downstream coherent.
- You want the **cheapest possible bulk removal** ahead of expensive stages. Nothing else removes so much for so little compute.
- You need a **reproducible, comparable** language distribution. Standardizing on fastText `lid.176` makes your corpus comparable to CCNet, RefinedWeb, and FineWeb.

**Be cautious, or skip, when:**

- Your source is **already clean and single-language** (a vetted books corpus, an internal document store). Running blunt web heuristics on curated text mostly generates false positives — you will delete good documents to remove junk that is not there.
- You care about **code, math, tables, or data-heavy content.** The prose heuristics will shred it. Route those content types out *before* the sieve and give them their own filters.
- You are working with **low-resource languages.** Langid alone is not trustworthy in the long tail; budget for per-language thresholds, native-speaker audits, and a downstream second signal, or you will ship the "Quality at a Glance" failure.
- Your documents are **short by nature** (titles, queries, tweets). fastText's short-text weakness dominates; use CLD3 or aggregate context first.

The deeper point is that the first sieve is a *filter for the common case*, tuned to general web prose. It is fast and it is right on average, and both of those virtues become liabilities the moment your data is not the common case. Know what you are routing through it, audit the pile it throws away, and treat every threshold as a hypothesis you can test — the same discipline that turns a copied config into a corpus worth training on. What survives this sieve goes on to [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) and then the learned quality filters; what it removes, you either re-source deliberately or accept as the price of a cheap, honest first pass.

## Further reading

- **Rae et al., 2021** — *Scaling Language Models: Methods, Analysis & Insights from Training Gopher.* Appendix A documents the MassiveText quality and repetition filters.
- **Raffel et al., 2020** — *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5/C4).* The line-level cleaning rules.
- **Penedo et al., 2023** — *The RefinedWeb Dataset for Falcon LLM.* The modern langid-plus-heuristics-plus-dedup synthesis.
- **Wenzek et al., 2020** — *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data.* fastText langid plus per-language perplexity at scale.
- **Kreutzer et al., 2022** — *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets.* The low-resource langid audit.
- **Penedo et al., 2024** — *The FineWeb Datasets.* Re-deriving the heuristic stack with ablations.
- Sibling posts: [text extraction and boilerplate removal](/blog/machine-learning/training-data/text-extraction-and-boilerplate-removal), [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale), [classifier and perplexity-based quality filtering](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering), and [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality).
