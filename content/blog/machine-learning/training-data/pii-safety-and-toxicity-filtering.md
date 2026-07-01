---
title: "PII, Safety, and Toxicity Filtering: Cleaning Data Without Lobotomizing It"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "How to strip personal data, toxicity, and illegal content out of a training corpus without deleting the dialects, identities, and hard topics your model actually needs to learn — with runnable detectors, a worked precision/recall scenario, and the LAION and C4 case studies."
tags: ["training-data", "pii-detection", "data-privacy", "content-safety", "toxicity-classification", "gdpr", "presidio", "csam", "data-filtering", "responsible-ai"]
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 30
---

There are two ways to get a filtering stage badly wrong, and they look like opposites. The first is to under-filter: you ship a corpus full of real people's home addresses, credit card numbers, hate speech, and — in the worst documented cases — child sexual abuse material, and then act surprised when the model recites a stranger's phone number or a regulator sends a letter. The second is to over-filter: you run a blunt "bad words" list over the crawl, feel responsible, and quietly delete most of the corpus's African-American English, most of its LGBTQ+ discussion, and most of its sex education content, because a keyword filter cannot tell a slur used in hate from the same token reclaimed in identity or quoted in a medical text. One failure gets you sued; the other gets you a model that is fluent, sanitized, and subtly bigoted by omission.

This post is about walking the ridge between those two cliffs. It is the last stop in the universal cleaning pipeline of this series — after you have [sourced and collected the raw data](/blog/machine-learning/training-data/sourcing-and-collecting-training-data) and stripped boilerplate, before you get to dedup and tokenization. The job here is narrow and load-bearing: remove personal data, remove content that is toxic or illegal, and do both without lobotomizing the corpus. We will build the detectors with real libraries, run a worked precision/recall scenario with concrete false positives and false negatives, and end at the one decision that matters more than any regex — how hard to turn the safety dial, and what you lose at each end.

![Why filter: the three compounding liabilities of shipping unfiltered training data](/imgs/blogs/pii-safety-and-toxicity-filtering-1.webp)

The diagram above is the mental model for the whole post: unfiltered data is not neutral raw material waiting to be cleaned "later." It is three separate, compounding bills — a legal one, a safety one, and a privacy-extraction one — and skipping the filter stage does not cancel those bills, it defers them to a worse time and a higher amount. The rest of this article is a tour of that triad and the machinery that pays it down.

## Why filtering is not optional

Start with the assumption most teams hold implicitly, then look at what is actually true.

| Assumption | Naive view | Reality |
|---|---|---|
| "The web is public, so using it is fine." | If a page was reachable, the data is fair game to train on. | Public visibility is not consent. GDPR treats an EU resident's name, email, or IP as personal data regardless of where it was posted; "publicly available" is not a lawful basis on its own. |
| "PII in the training set stays in the training set." | The model learns statistics, not records. | Large models memorize verbatim. Carlini et al. extracted real names, phone numbers, and addresses from GPT-2 by prompting — the training data leaks back out. |
| "A bad-words list makes the corpus safe." | Drop documents containing slurs and you are done. | Blunt blocklists over-remove reclaimed identity language and dialect while missing coded and obfuscated toxicity. Safety went down and bias went up. |
| "Image data is just pixels, legally." | Scraped images carry no special duty. | If any of them is CSAM, you have a criminal-law problem and a mandatory-reporting duty, not a data-quality problem. |

Three distinct liabilities fall out of that table, and they compound because they share an input.

**Legal exposure.** The General Data Protection Regulation caps administrative fines at 20 million euros or 4 percent of global annual turnover, whichever is higher, for the most serious violations. The California Consumer Privacy Act, as amended by the CPRA, allows civil penalties up to \$2,500 per violation and \$7,500 per intentional violation, and grants consumers a private right of action for certain breaches — which means class actions, not just regulator letters. When your corpus contains millions of EU or California residents' personal data with no lawful basis and no way to honor a deletion request, "per violation" is a frightening unit.

**Safety harm.** A model trained on unfiltered toxicity will generate toxicity, and a model trained on NSFW material will generate NSFW material on innocuous prompts. This is not hypothetical; it is the default. The safety filter is what decides how hard your downstream alignment work has to fight the base model's priors.

**Privacy extraction.** This is the one teams underestimate. Even if no human ever reads the training set, the model becomes a lossy, queryable copy of it. If a person's PII survives into the weights, a training-data extraction attack can pull it back out. Filtering is not just hygiene for the corpus you hold; it is the only durable defense against the corpus you accidentally publish inside the model.

> The filter stage is not where you make the data nice. It is where you decide which lawsuits, which extraction attacks, and which failure modes you are choosing to keep.

## PII detection: regex, NER, and the precision/recall tension

The senior rule of thumb: **structured PII is a regex problem, unstructured PII is a model problem, and you need both because each catches what the other misses.**

An email address, a US Social Security number, a credit card, an AWS access key — these have shape. A regex nails them with high precision and near-zero cost. A person's name, a street address, an employer, a rare medical condition mentioned in passing — these have no shape; they are defined by role in a sentence, not by a character pattern. Only a named-entity recognition (NER) model, trained to read context, finds them. Layer the cheap high-precision detector on top of the expensive high-recall one and you get a cascade.

![The PII detection cascade: regex, then NER, then a context classifier, then human audit — recall rises with each layer and so does cost](/imgs/blogs/pii-safety-and-toxicity-filtering-2.webp)

Read the cascade top to bottom. Each layer you add catches PII the layer above missed, and each layer costs more — regex is microseconds per document, NER is milliseconds, a transformer context classifier is tens of milliseconds and a GPU, and human audit is dollars per thousand documents but is the only source of ground truth. You do not run all four on every document at petabyte scale; you run the cheap layers on everything and escalate a sampled or high-risk subset to the expensive ones.

The production-grade way to do this without hand-rolling every regex is Microsoft Presidio, which bundles a large library of recognizers (regex plus checksum validation plus context words) with a spaCy NER backend, and pairs an analyzer with an anonymizer.

```python
# pip install presidio-analyzer presidio-anonymizer
# python -m spacy download en_core_web_lg
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()          # spaCy NER + pattern recognizers + context
anonymizer = AnonymizerEngine()

text = (
    "Reach Jane Doe at jane.doe@acme.io or +1-415-555-0147. "
    "Card 4111 1111 1111 1111, SSN 078-05-1120. "
    "Order SKU-5501-08 shipped to 120 Market St, San Francisco."
)

# Detect. Each result carries an entity type and a model/pattern confidence score.
results = analyzer.analyze(text=text, language="en")
for r in sorted(results, key=lambda x: x.start):
    print(f"{r.entity_type:15} score={r.score:.2f}  '{text[r.start:r.end]}'")

# Redact to typed placeholders so the sentence structure survives.
anonymized = anonymizer.anonymize(
    text=text,
    analyzer_results=results,
    operators={
        "DEFAULT":         OperatorConfig("replace", {"new_value": "[REDACTED]"}),
        "EMAIL_ADDRESS":   OperatorConfig("replace", {"new_value": "[EMAIL]"}),
        "PHONE_NUMBER":    OperatorConfig("replace", {"new_value": "[PHONE]"}),
        "CREDIT_CARD":     OperatorConfig("replace", {"new_value": "[CARD]"}),
        "US_SSN":          OperatorConfig("replace", {"new_value": "[SSN]"}),
        "PERSON":          OperatorConfig("replace", {"new_value": "[NAME]"}),
        "LOCATION":        OperatorConfig("replace", {"new_value": "[LOCATION]"}),
    },
)
print(anonymized.text)
```

If you would rather not take the Presidio dependency, the same idea in raw form is a set of validated regexes plus a spaCy pass:

```python
import re, spacy
nlp = spacy.load("en_core_web_lg")

PATTERNS = {
    "EMAIL":  re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "SSN":    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD":   re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "PHONE":  re.compile(r"\b(?:\+?1[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),
    "AWSKEY": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
}

def detect(text):
    spans = []
    for label, pat in PATTERNS.items():
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), label))
    for ent in nlp(text).ents:               # names, orgs, GPE (places)
        if ent.label_ in {"PERSON", "GPE", "ORG"}:
            spans.append((ent.start_char, ent.end_char, ent.label_))
    return spans
```

Secrets are a special, high-value class of structured PII, and they deserve their own dedicated scanner rather than a generic PII pass, because the cost of missing one is an active credential in your corpus. Tools like `detect-secrets` and `gitleaks` combine curated patterns for dozens of provider key formats with a high-entropy string detector, which catches keys whose format you did not enumerate.

```python
# pip install detect-secrets
from detect_secrets.core.scan import scan_line
from detect_secrets.settings import default_settings

corpus_line = "export STRIPE_KEY=sk_live_EXAMPLE_not_a_real_key_do_not_scan  # prod"
with default_settings():
    for secret in scan_line(corpus_line):
        print(secret.type, "->", secret.secret_value[:6] + "...")
# HighEntropyString / Secret Keyword -> sk_liv...
```

Treat every secrets hit as a **remove**, never a redact: a live key in training data is an incident whether or not the model ever emits it, and it should be rotated out-of-band as well as stripped from the corpus. High-entropy detection buys recall on unknown formats at the cost of precision — base64 blobs, UUIDs, and hashes trip it — so pair it with an allowlist of known-benign high-entropy shapes.

Notice the tension already lurking in the regex blocks. The credit card and phone regexes are greedy: `SKU-5501-08` will not match the card pattern, but a 9-digit product code, an ISBN, or a formatted order number frequently *will* trip a loosened phone or SSN pattern. And the SSN regex will happily match `078-05-1120` whether it is a real Social Security number or a decommissioned example. Tighten the pattern and you miss real PII; loosen it and you shred benign identifiers. That tension is the whole game, and the cleanest way to see it is a confusion matrix on a labeled sample.

## The confusion matrix, made concrete

Every detector, regex or model, lands each candidate span in one of four buckets against the truth.

![The PII detector confusion matrix: true positives and true negatives are the wins; false negatives leak PII, false positives destroy signal](/imgs/blogs/pii-safety-and-toxicity-filtering-3.webp)

The two error cells are not symmetric in consequence, which is the single most important thing to internalize about this stage:

- A **false negative** — real PII the detector passes — leaks a person's data into the weights. This is the cell that gets you the extraction attack and the regulator letter.
- A **false positive** — benign text the detector flags — destroys signal. A product code redacted to `[SSN]`, a company name redacted to `[NAME]`, a city stripped from every travel blog. Do enough of it and you have degraded the corpus while feeling safe.

We measure the tradeoff with precision and recall. With $TP$, $FP$, $FN$ the counts of true positives, false positives, and false negatives:

$$P = \frac{TP}{TP + FP}, \qquad R = \frac{TP}{TP + FN}, \qquad F_1 = \frac{2PR}{P + R}$$

Precision $P$ answers "when the detector flags something, how often is it really PII?" Recall $R$ answers "of all the real PII, how much did we catch?" You cannot maximize both on ambiguous data; you choose an operating point.

### Worked scenario: measuring the PII pass

Take a realistic 200-document sample with the PII hand-labeled, giving a gold set of 20 true PII spans. Run three configurations and count the cells.

```python
# gold = set of (doc_id, start, end) spans that are truly PII (|gold| = 20)
# pred = set produced by a configuration
def score(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    P  = tp / (tp + fp) if tp + fp else 0.0
    R  = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    return dict(TP=tp, FP=fp, FN=fn, precision=round(P, 2),
               recall=round(R, 2), f1=round(F1, 2))
```

The three configurations produce, on this sample:

| Configuration | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Regex only (tight) | 13 | 1 | 7 | 0.93 | 0.65 | 0.76 |
| Regex + spaCy NER | 18 | 7 | 2 | 0.72 | 0.90 | 0.80 |
| + context classifier | 19 | 3 | 1 | 0.86 | 0.95 | 0.90 |

Now look at the actual mistakes, because the aggregate numbers hide the story:

- **False positives (regex + NER, 7 of them):** `SKU-5501-08` matched the loosened SSN/phone pattern (the classic product-code-as-SSN error). The order number `4155550147` matched the phone pattern. spaCy tagged the software project `Apache Kafka` and the model name `Llama` as `PERSON`/`ORG`, and flagged the city name inside `New York Times` as a `GPE` location worth redacting. Each of these, if acted on, deletes a useful token.
- **False negatives (regex only, 7 of them):** the obfuscated email `jane [dot] doe [at] acme [dot] io`, the spelled-out phone "four one five, five five five...", a name that only regex saw (regex has no concept of names), and an IBAN in a format the card regex did not cover.

The regex-only column is the high-precision, low-recall operating point; adding NER buys 25 points of recall at the cost of 21 points of precision. The context classifier claws precision back by re-checking each flagged span in context — is `Jordan` a person or a country, is `SKU-5501-08` a number-shaped identifier that appears in a product catalog? — and is why serious pipelines run the third layer on the subset the first two disagree about, not on everything.

## Redact, remove, or keep

Detection is only half the stage. Once you have a span, you have three options, and picking the wrong one is its own failure mode.

![Redact, remove, or keep: the sensitivity tier of a detected span decides its fate, not the mere fact of a match](/imgs/blogs/pii-safety-and-toxicity-filtering-4.webp)

The tree encodes the policy: **sensitivity tier, not the fact of a match, decides the action.**

- **Remove** the whole document (or the span, if the document is otherwise valuable) for the highest-sensitivity classes: live credit cards, Social Security numbers, private keys, API secrets, medical records. There is no upside to keeping a real secret in the corpus at any placeholder fidelity, and a leaked AWS key in training data is an incident regardless of what the model does with it.
- **Redact to a typed placeholder** for medium-sensitivity PII you want the model to learn the *shape* of without the *value*. Replacing `jane.doe@acme.io` with `[EMAIL]` and `+1-415-555-0147` with `[PHONE]` teaches the model that emails and phone numbers exist and how they sit in a sentence, without teaching it Jane's. Typed placeholders beat a generic `[REDACTED]` because they preserve grammatical role and downstream parseability.
- **Keep** low-sensitivity or public information (a city, a public company's name, a public official acting in office) and keep false positives — but keep them *with a monitor*, tracking the flag rate so you notice when a rule starts over-firing.

The anonymization tradeoff is real and worth stating plainly. Every redaction is a small hole punched in the language. Redact too aggressively and you get corpora where "call me at [PHONE]" and "born in [DATE] in [LOCATION]" appear so often the model learns to emit the placeholders themselves — a documented failure where models trained on over-redacted data literally generate `[NAME]` and `[EMAIL]` in their output. Consistent typed placeholders, a redaction budget per document, and keeping the low-sensitivity tier are the levers that keep redaction from becoming lobotomy. Anonymization is never perfect either: with enough quasi-identifiers — ZIP code plus birthdate plus gender re-identifies a large fraction of the US population — a "de-identified" record can be re-identified. Redaction reduces risk; it does not zero it.

## Toxicity and NSFW classification

PII is about *whose* data it is. Toxicity is about *what the content says*. The tooling splits into three tiers of sophistication, and the cheapest one is where most of the damage gets done.

**Blocklists** — a list of banned words, applied by substring match. Cheap, transparent, and blunt to the point of being dangerous (more on that below).

**Classifiers** — a model that scores a text for toxicity. The two workhorses:

- **Perspective API** (Jigsaw/Google) returns per-attribute scores in [0, 1] for `TOXICITY`, `SEVERE_TOXICITY`, `IDENTITY_ATTACK`, `INSULT`, `THREAT`, and more. It is a hosted service, rate-limited, and the de facto reference for research comparisons.
- **fastText** and small transformer classifiers (for example the `detoxify` package, which wraps a `toxic-bert` model) run locally at corpus scale, which Perspective's rate limits make impractical for petabytes.

```python
# pip install detoxify
from detoxify import Detoxify
model = Detoxify("original")          # multilingual + unbiased variants also exist

docs = [
    "Thanks so much for the detailed review, this helped a lot.",
    "People like you should not be allowed to speak, you are worthless.",
    "As a gay man I found this clinic's HIV guidance genuinely useful.",
]
scores = model.predict(docs)          # dict of attribute -> list[float]
for text, tox in zip(docs, scores["toxicity"]):
    print(f"{tox:.3f}  {text[:60]}")
```

The third document is the trap. A well-tuned classifier scores it low, but a blocklist containing identity terms and a naive classifier biased by those terms will both flag it. That single example is the difference between safety filtering and censorship.

The three tiers are not interchangeable; pick by scale and by how much bias you can tolerate.

| Tool | Signal | Corpus-scale cost | Bias profile | Reach for it when |
|---|---|---|---|---|
| Bad-words blocklist | substring match | negligible | severe — over-removes dialect/identity | never as the primary filter; only for a narrow, unambiguous term set |
| fastText / `detoxify` | local model score | low (CPU/GPU, runs on petabytes) | moderate — inherits training-set bias | the corpus-scale first-pass toxicity filter |
| Perspective API | hosted model scores per attribute | high (rate-limited, per-call) | moderate, well-studied | sampled audits, research comparisons, a high-risk subset |
| LLM-as-judge | prompted classification | very high | tunable via prompt, but pricey | edge cases the cheap classifiers disagree on |

The pattern mirrors the PII cascade: a cheap local classifier scores everything, and you escalate only the ambiguous or high-risk slice to the expensive judges.

### The threshold is the dial

A classifier gives you a score, not a decision. You supply the threshold, and the threshold *is* the safety-vs-coverage tradeoff made numeric. Watch it move.

<figure class="blog-anim">
<svg viewBox="0 0 720 300" role="img" aria-label="A toxicity-score axis with a benign distribution on the left and a toxic distribution on the right; a threshold line sweeps left and right, and the shaded removed region to its right grows and shrinks, trading recall against over-removal of benign content" style="width:100%;height:auto;max-width:820px">
<title>Precision/recall frontier as the toxicity threshold slides</title>
<style>
.a1-axis{stroke:var(--text-secondary,#6b7280);stroke-width:2}
.a1-benign{fill:var(--border,#d1d5db);opacity:.55}
.a1-toxic{fill:#f0a5a5;opacity:.6}
.a1-line{stroke:var(--accent,#6366f1);stroke-width:3}
.a1-rem{fill:var(--accent,#6366f1);opacity:.14}
.a1-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.a1-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.a1-thr{font:600 13px ui-sans-serif,system-ui;fill:var(--accent,#6366f1);text-anchor:middle}
@keyframes a1-slide{0%{transform:translateX(300px)}100%{transform:translateX(520px)}}
.a1-sweep{animation:a1-slide 9s ease-in-out infinite alternate;transform:translateX(410px)}
@media (prefers-reduced-motion:reduce){.a1-sweep{animation:none}}
</style>
<clipPath id="a1clip"><rect x="80" y="58" width="560" height="166"/></clipPath>
<path class="a1-benign" d="M80,220 Q260,60 440,220 Z"/>
<path class="a1-toxic" d="M280,220 Q500,60 640,220 Z"/>
<g clip-path="url(#a1clip)"><rect class="a1-rem a1-sweep" x="0" y="58" width="640" height="164"/></g>
<line class="a1-axis" x1="80" y1="220" x2="650" y2="220"/>
<line class="a1-line a1-sweep" x1="0" y1="52" x2="0" y2="224"/>
<text class="a1-thr a1-sweep" x="0" y="44">threshold</text>
<text class="a1-lbl" x="235" y="165">benign</text>
<text class="a1-lbl" x="470" y="165">toxic / PII</text>
<text class="a1-sub" x="150" y="245">keep (coverage)</text>
<text class="a1-sub" x="560" y="245">removed (flagged)</text>
<text class="a1-sub" x="365" y="285">toxicity score: low &#8594; high</text>
</svg>
<figcaption>Slide the threshold left and you catch more toxic content but the shaded removed region eats into the benign hump (over-filtering); slide it right and toxicity escapes into the kept set. There is no setting that is both high-recall and high-precision on overlapping distributions.</figcaption>
</figure>

### Worked scenario: the threshold sweep

Score a 1,000,000-document sample with a toxicity classifier and sweep the removal threshold. Keep-rate is the fraction of the corpus surviving; toxic-recall is the fraction of truly toxic documents removed. The numbers below are representative of what a `toxic-bert`-class model produces on a general web sample.

| Threshold | Keep-rate | Toxic-recall | Benign removed | AAE removal vs baseline |
|---|---|---|---|---|
| 0.90 | 99.3% | 41% | 0.2% | 1.2x |
| 0.70 | 97.8% | 66% | 0.9% | 1.6x |
| 0.50 | 94.1% | 84% | 3.1% | 2.3x |
| 0.30 | 86.4% | 93% | 9.8% | 3.7x |

Read the last two columns together. Dropping the threshold from 0.70 to 0.30 raises toxic-recall from 66 to 93 percent — a real safety gain — but the benign-removed rate goes from under 1 percent to nearly 10 percent, and the collateral damage is *not evenly distributed*: documents in African-American English get removed at 3.7 times the baseline rate at the aggressive setting. That last column is the fairness caveat that a single "keep-rate" number hides, and it is the bridge to the most important cautionary tale in this whole stage.

```python
import numpy as np
def sweep(scores, is_toxic, thresholds):
    for t in thresholds:
        removed = scores >= t
        keep_rate    = 1 - removed.mean()
        toxic_recall = removed[is_toxic].mean()
        benign_rm    = removed[~is_toxic].mean()
        print(f"t={t:.2f}  keep={keep_rate:.3f}  "
              f"toxic_recall={toxic_recall:.2f}  benign_removed={benign_rm:.3f}")

sweep(scores, is_toxic, [0.9, 0.7, 0.5, 0.3])
```

## The blocklist bias problem

The senior rule here is uncomfortable and worth putting in bold: **a bad-words blocklist is not a safety tool, it is a demographic filter that happens to remove some slurs.**

![What a blunt blocklist actually removes: benign dialect and identity text on the left is deleted, while real harassment slips through](/imgs/blogs/pii-safety-and-toxicity-filtering-5.webp)

This is not a rhetorical claim; it is measured. When researchers documented the C4 corpus — the Colossal Clean Crawled Corpus behind T5 and many others — they analyzed the effect of the "List of Dirty, Naughty, Obscene, and Otherwise Bad Words" that was used to clean it. The blocklist disproportionately removed text associated with minority identities and dialects. Documents written in African-American English and Hispanic-aligned English were filtered at substantially higher rates than white-aligned English, and content mentioning LGBTQ+ identities was removed because terms that are slurs in one context are neutral or reclaimed identity terms in another. The blocklist could not tell the difference, because a substring match has no context.

The figure makes the double failure concrete: the benign columns — dialect, identity discussion, sex-ed and health information — get removed (the red cells), while actual harassment that avoids the exact listed strings, or spells them creatively, slips through (the amber cell). You paid coverage and fairness and got very little safety in return. A corpus filtered this way produces a model that is worse at understanding and generating minority dialects, worse at discussing health and sexuality, and *not meaningfully safer* — the worst possible trade.

The fix is not "no filtering." It is to replace substring blocklists with context-aware classifiers, to evaluate removal rates *disaggregated by dialect and identity group* rather than trusting an aggregate keep-rate, and to reserve hard string-match blocking for the narrow set of terms that are unambiguous regardless of context. Measure the bias, or you are shipping it.

## CSAM: the one filter with legal teeth

Everything above is a quality-and-risk tradeoff you get to tune. CSAM — child sexual abuse material — is not. It is a category apart, governed by criminal law, and it does not have a "coverage" side to trade against. If it is in your data, it has to come out, and in many jurisdictions the moment you know it is there you have a reporting duty.

![CSAM hash-matching and the reporting duty: a hash match triggers a legal obligation, not a quality score](/imgs/blogs/pii-safety-and-toxicity-filtering-6.webp)

The mechanism is deliberately not a classifier. You do not want a model *guessing* about this category, and you especially do not want to build a training set of positive examples to train such a classifier. Instead you match against hashes of *known* material maintained by clearinghouses:

- **Perceptual hashing** — Microsoft PhotoDNA, Facebook's PDQ (images) and TMK+PDQF (video) — computes a robust fingerprint that survives resizing, recompression, and minor edits, unlike a cryptographic hash that changes on a single flipped bit.
- The fingerprint is matched against databases maintained by the **National Center for Missing & Exploited Children (NCMEC)**, the **Internet Watch Foundation (IWF)**, the **Canadian Centre for Child Protection**, and similar bodies. Access is gated; you apply as an organization.
- A **match** triggers quarantine, exclusion from the corpus, and — for US-based providers — a mandatory report to NCMEC's CyberTipline under 18 U.S.C. 2258A. This is a legal obligation, not a data-cleaning step, and mishandling the material itself can be a crime.

The reason this matters beyond compliance is that image-text scrapers pull from the open web at a scale where some CSAM is statistically near-certain, and the discovery of it in a widely-used dataset — covered in the case studies below — is exactly what happens when this filter is skipped. Hash-matching is table stakes for any image or video corpus. Pair it with the provenance and watermarking practices covered in [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) so you can trace where a flagged asset entered your pipeline.

## The safety-vs-coverage dial

Zoom out. Every decision in this post is one knob with a failure mode at each end, and the engineering skill is choosing where to sit — explicitly, with numbers, per risk tier — rather than defaulting to an end.

![The safety-vs-coverage dial: both extremes are failure modes; the operating target sits in the tuned middle](/imgs/blogs/pii-safety-and-toxicity-filtering-7.webp)

The dial reads left to right. **Under-filtered** keeps nearly everything — maximum coverage, and maximum exposure to PII leaks, toxic generation, fines, and CSAM liability. **Over-filtered** drops so much benign dialect and identity content that coverage collapses and the model inherits demographic bias by omission. **Balanced** is not a single magic number; it is a set of per-tier decisions: remove hard for the highest-sensitivity classes, redact for the medium tier, keep-and-monitor for the low tier, use context classifiers over blocklists, and measure collateral removal disaggregated by group. The target is "acceptable risk with broad coverage preserved," and you only know you are there because you measured both sides, not because the filter felt strict.

A useful discipline: set the dial differently per risk tier rather than globally. CSAM is all the way to "remove, no exceptions." Live secrets and financial PII sit near it. Toxicity sits in a tuned middle with the threshold chosen from a disaggregated sweep. Public, low-sensitivity information sits toward "keep." One global threshold across all of these is how you end up simultaneously under-filtering the dangerous categories and over-filtering the harmless ones.

## Case studies from production

### 1. LAION-5B and the Re-LAION re-release

In December 2023, the Stanford Internet Observatory published an analysis of LAION-5B, the 5.85-billion image-text-pair dataset that trained a generation of open image models including early Stable Diffusion. Using hash-matching against known-CSAM databases (PhotoDNA and the resources of NCMEC and the Canadian Centre for Child Protection), the researchers identified over a thousand validated instances of CSAM among the URLs the dataset pointed to. The dataset was not a store of images — it was a list of links — but that distinction offered no legal or ethical cover: models had already been trained on the referenced content. LAION took the dataset offline. In August 2024 they released **Re-LAION-5B**, a cleaned version with the flagged content removed using lists supplied by IWF, the Canadian Centre for Child Protection, and partner organizations, and republished it as the safe baseline. The lesson is blunt: a scrape at web scale *will* contain this material, hash-matching is not optional, and "we only stored URLs" is not a defense. It also shows the cost asymmetry — the filter that was skipped at build time forced a full teardown and rebuild, plus every downstream model's provenance called into question.

### 2. The C4 blocklist study

Dodge et al. (2021), in "Documenting the English Colossal Clean Crawled Corpus," did the unglamorous work of auditing what C4's cleaning actually removed. Their finding on the bad-words blocklist is the canonical warning for this stage: the list disproportionately filtered documents associated with minority dialects and identities. Text in African-American English was removed at meaningfully higher rates than white-aligned English, and material discussing LGBTQ+ identities was swept out because identity terms overlapped with the blocklist. The safety benefit was marginal — real toxicity that avoided the exact strings survived — while the fairness cost was severe and invisible in any aggregate quality metric. This study is why "we used a blocklist" should make a reviewer wince, and why disaggregated removal-rate evaluation is now the expected practice.

### 3. Extracting training data from GPT-2

Carlini et al. (2021), "Extracting Training Data from Large Language Models," demonstrated the privacy-extraction liability directly. By generating large volumes of text from GPT-2 and ranking samples by a membership signal, they recovered verbatim sequences from the training data — including a real person's name, phone number, email, and physical address that appeared only a handful of times in the corpus. The model had memorized rare sequences and would emit them under the right prompt. The takeaway for the filter stage: PII that survives into training does not merely sit inert "in the weights"; it is recoverable. Deduplication reduces memorization (repeated sequences are memorized more strongly), but the durable fix is not letting the PII in. This is why the privacy branch of the opening triad is a real bill, not a theoretical one.

### 4. The Enron corpus that keeps shipping PII

The Enron email dataset — roughly half a million real emails from 150 employees, released during the federal investigation — is one of the most-used corpora in NLP for tasks from classification to summarization. It is also full of real names, personal email addresses, phone numbers, and private conversations of people who never consented to any of it and mostly were never charged with anything. It gets pulled into training mixes precisely because it is conveniently available and "already public." It is the perfect example of "publicly available is not a lawful basis": the data being on a court record does not make retraining a commercial model on Jane-from-accounting's private correspondence lawful or ethical. If you use it, run the PII pass over it like any other source; do not grandfather it in because it is famous.

### 5. The product-code massacre

A war story of the opposite failure. A team running a first PII pass on a large e-commerce and support corpus set their SSN and phone patterns loose to maximize recall — better safe than sorry. The support corpus was dense with order numbers, SKUs, tracking IDs, and case numbers, many of which are nine to eleven digits with dashes. The loose patterns matched them by the hundreds of thousands, and the pipeline redacted every one to `[SSN]` or `[PHONE]`. The model trained on the result learned that support tickets are made of placeholders, and its generated "support replies" were peppered with `[PHONE]` and `[SSN]` tokens. The root cause was a precision problem misdiagnosed as a recall problem; the fix was context validation (a real US SSN passes area/group/serial sanity checks; an order number does not) and a per-document redaction budget that flags any document losing more than a set fraction of its tokens for review rather than silent redaction.

### 6. Over-filtered health content

A safety pass on a general corpus used an off-the-shelf toxicity classifier at an aggressive threshold, chosen because a strict setting "seemed responsible." Post-hoc analysis of what got removed showed that sexual-health, harm-reduction, and mental-health content — including suicide-prevention resources phrased frankly — was removed at several times the baseline rate, because frank clinical language about bodies, drugs, and self-harm scores high on toxicity models trained largely on social-media abuse. The downstream model became evasive and unhelpful on exactly the high-stakes topics where accurate information matters most. The fix combined a lower, disaggregated threshold with a carve-out: content matching a curated health/clinical domain and register was scored by a separate, calibrated classifier rather than the general one, restoring the material without reopening the toxicity gate.

## Troubleshooting

A field guide, in the symptom-to-cause-to-fix format, for the failures this stage actually produces.

### Symptom: the model generates placeholder tokens like `[NAME]` or `[PHONE]`

- **Detection:** grep the model's outputs (or a validation-set generation sample) for your placeholder strings; count documents in the training set whose redaction ratio (redacted spans / total tokens) exceeds, say, 5 percent.
- **Cause:** over-redaction. Placeholders became frequent enough in training that the model learned to emit them as ordinary vocabulary.
- **Fix:** stop redacting the low-sensitivity tier (keep cities, public orgs); impose a per-document redaction budget and route over-budget documents to removal or review instead of blanket redaction; and consider realistic *synthetic* replacements (a fake but well-formed name/phone) instead of a constant placeholder, so the statistical signal of "a name goes here" survives without a repeated literal token.

### Symptom: precision is fine in tests but the corpus is being shredded in production

- **Detection:** sample flagged spans from the real corpus (not the test set) and hand-label a few hundred; compute precision on *that*. Watch for a specific entity type dominating the flags.
- **Cause:** distribution shift. The test set did not contain the order numbers, SKUs, ISBNs, or scientific identifiers that live in the real data and trip loosened numeric patterns.
- **Fix:** add checksum/format validation (Luhn for credit cards, area/group/serial validity for SSNs, IBAN check digits); require a context word near numeric matches ("SSN", "social security"); and always calibrate thresholds on a sample of the *actual* corpus, never on a clean benchmark.

### Symptom: removal rates look reasonable in aggregate but a group is being erased

- **Detection:** disaggregate removal rate by dialect and identity signals (a dialect classifier, or identity-term presence as a proxy) and compare each group to the baseline, exactly as in the threshold-sweep table above.
- **Cause:** blocklist or a biased classifier over-removing minority-dialect and identity content — the C4 failure.
- **Fix:** replace substring blocklists with context-aware classifiers; add the disaggregated removal rate as a *gate* in your pipeline that fails the build if any group exceeds a multiple of baseline; and for identity/health language, use a separately calibrated classifier or a curated carve-out.

### Symptom: obfuscated PII or coded toxicity sails straight through

- **Detection:** build a small adversarial set — emails written as "name [at] domain [dot] com", phones spelled out, leetspeak and homoglyph slurs, spaced-out slurs — and measure recall on it specifically.
- **Cause:** detectors keyed to canonical surface forms; evasion is trivial against pure pattern or substring matching.
- **Fix:** normalize before detection (collapse whitespace, map homoglyphs to ASCII, expand common obfuscation tokens like "[at]"/"[dot]"); add a context classifier that reads meaning rather than surface form for the toxicity side; and accept that this is an arms race — periodically refresh the adversarial set from what you find slipping through.

### Symptom: a downstream extraction probe recovers PII from the trained model

- **Detection:** run a training-data extraction attack against your own model as a red-team step — prompt with prefixes likely to precede memorized PII and check outputs against known corpus entries; measure the memorization/canary exposure with inserted canaries.
- **Cause:** PII false negatives at the filter stage plus insufficient deduplication; rare-but-present sequences get memorized.
- **Fix:** the durable fix is upstream — raise PII recall on the highest-sensitivity classes even at some precision cost, and deduplicate aggressively (repeated sequences memorize hardest). Downstream mitigations — differential privacy in training, output filters — are backstops, not substitutes for keeping the data out.

### Symptom: image scrape flagged by a partner or auditor for illegal content

- **Detection:** you should have detected it first — run perceptual hash-matching (PhotoDNA/PDQ) against clearinghouse databases at ingestion, before any training.
- **Cause:** the CSAM hash-matching step was skipped or never set up; web-scale image scrapes contain this material by default.
- **Fix:** apply for access to NCMEC/IWF/C3P hash sets, integrate hash-matching as a mandatory ingestion gate, quarantine and file the required report on any match, and treat the discovery as the legal incident it is — not a data-quality ticket. There is no threshold to tune here.

## When to reach for aggressive filtering — and when not to

Reach for aggressive, high-recall filtering when:

- The corpus feeds a model that will be **deployed to the public** or to vulnerable users, where a single memorized SSN or a toxic completion is a headline.
- You are in a **regulated domain** (health, finance, children's products) where the legal floor is high and the tolerance for leaked PII is effectively zero.
- The data source is **high-risk by construction** — raw web scrape, image-text pairs, forum dumps — where the base rate of PII, toxicity, and illegal content is high.
- You can **measure the collateral damage** and have decided the coverage loss is acceptable for this model's purpose. Aggressive is a choice you earn by measuring, not a default.

Skip or soften the filtering when:

- The corpus is **already curated and licensed** — textbooks, a vetted internal knowledge base, a dataset with a signed contract and provenance — where re-running blunt filters mostly adds false positives.
- The model is a **narrow internal tool** on data whose subjects have consented and which never leaves your walls (though CSAM hash-matching still applies to any image data, always).
- The topic space **requires** the content a naive filter would remove — a clinical model needs frank health language, a moderation model needs to see toxicity to learn it, a dialect or sociolinguistics model needs exactly the text a blocklist deletes. Here, filtering out the "unsafe"-scoring content destroys the task.
- You have not yet **measured removal disaggregated by group.** Do not ship an aggressive setting on faith; a strict filter you have not audited is more likely to be biased than safe.

The through-line of this whole stage is that filtering is a decision, not a default, and the two ways to get it wrong are equal and opposite. Under-filter and you keep the lawsuits, the toxicity, and the extraction attacks. Over-filter and you keep a fluent model with holes where whole communities' language used to be. The teams that get it right do not pick a side; they set the dial per risk tier, measure both what they removed and who it belonged to, and treat CSAM as the one place with no dial at all. The next post in this series takes the argument up a level to [the legal, ethical, and future landscape of training data](/blog/machine-learning/training-data/legal-ethics-and-the-future-of-training-data), where these filtering choices meet copyright, consent, and regulation head-on.

## Further reading

- [Sourcing and collecting training data](/blog/machine-learning/training-data/sourcing-and-collecting-training-data) — the stage before this one; where the PII and toxicity you are now filtering came from.
- [Measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) — how to quantify what filtering did to the corpus, including the coverage side of the dial.
- [Safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) — the image-side companion to CSAM hash-matching and content provenance.
- Dodge et al., "Documenting the English Colossal Clean Crawled Corpus" (2021) — the C4 blocklist audit.
- Carlini et al., "Extracting Training Data from Large Language Models" (2021) — the memorization-and-extraction result.
- Thiel, "Identifying and Eliminating CSAM in Generative ML Training Data and Models" (Stanford Internet Observatory, 2023) — the LAION-5B analysis behind the Re-LAION re-release.
