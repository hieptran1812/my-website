---
title: "The law, ethics, and future of training data: a defensibility playbook"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "The capstone of a 25-post series on training data: copyright and the lawsuits, opt-out and licensing, privacy and the right-to-be-forgotten, bias, data cards, lineage, and where the tokens come from next — framed as one question, can you defend your corpus?"
tags:
  - training-data
  - ai-copyright
  - data-governance
  - data-provenance
  - datasheets-for-datasets
  - model-collapse
  - privacy
  - data-licensing
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 31
---

The first post in this series opened with a bet: give two teams the same compute and the same architecture, tell me only how much care went into their data, and I will predict the winner. Twenty-four posts later — after collection, extraction, cleaning, deduplication, decontamination, filtering, selection, mixing, synthetic generation, and every per-modality recipe — I want to close with the part nobody puts on the architecture slide and everybody eventually gets a subpoena about. Not "is the data good?" The question that decides whether your model has a future is: **can you defend the data?**

This is the last mile that has nothing to do with perplexity. A frontier corpus in 2026 is a legal artifact as much as a technical one. It has a copyright status, a consent story, a bias profile, a paper trail, and a lineage — or it does not, and the absence is a liability you have already shipped into the weights. The engineers who win the next few years are the ones who treat "where did this token come from, and am I allowed to have learned from it?" as a first-class pipeline stage, not a thing the legal team discovers during discovery.

The diagram above — well, below — is the mental model for this entire post. A defensible pipeline answers four questions, and each one maps to a section of this article: **provenance** (can you prove where every token came from?), **consent** (can you honor a takedown or a deletion request?), **representation** (did you measure who is in the corpus?), and **reproducibility** (which exact data trained this checkpoint?). Answer all four and you have a corpus you can defend in discovery, to a regulator, and to yourself. Miss one and it becomes the thing a plaintiff's lawyer, a data-protection authority, or a journalist finds first.

![The four questions a defensible data pipeline must answer](/imgs/blogs/legal-ethics-and-the-future-of-training-data-1.webp)

This is post #25, the capstone, and it bookends [why data decides the model](/blog/machine-learning/training-data/why-data-decides-the-model). The opener argued that the corpus is the dominant *technical* lever. The closer argues that the corpus is the dominant *legal and ethical* lever too — and that the two are converging. The same properties that make data defensible in court (provenance, documentation, consent, measured composition) are the ones that make it defensible as engineering. Governance is not a tax on good data work. Increasingly, it *is* good data work.

## Why the last mile is legal, not technical

Most of us were trained to think the hard part of data is the pipeline: the dedup at scale, the quality classifier, the domain weights. That part is genuinely hard, and the previous twenty-four posts are about it. But the part that ends companies is different, and the mismatch between how engineers think about it and how it actually plays out is worth stating plainly.

| Assumption | The naive engineering view | The reality in 2026 |
| --- | --- | --- |
| What "cleared" data means | It passed the quality and safety filters | It has a license, a provenance record, and a consent basis |
| Who owns copyright risk | Legal will handle it later | The data team ships it into the weights; legal inherits it |
| What "we scraped the public web" buys | Free, abundant tokens | Free tokens plus unbounded, undocumented liability |
| Whether you can delete a user's data | Of course, it's just a delete | Not once it is trained into the weights |
| What a bias audit is | A fairness paper we might write | A composition measurement you must run before you ship |
| The moat | The weights and the recipe | The *documented, licensed, reproducible* data pipeline |

None of this means the engineering stopped mattering. It means a second axis appeared next to it, and on that axis most teams are years behind. The good news: the fixes are concrete, they are mostly documentation and discipline, and they are cheaper than a lawsuit. Let us walk the four questions, then build the artifacts that answer them.

> Your model is a lossy compression of its training set. In a courtroom, in a regulator's office, and in an incident review, you will be asked to decompress it — to explain what went in and prove you were allowed to use it. Build the pipeline so that answer exists.

## 1. Copyright and the lawsuits

**Senior rule of thumb: the training act and the source copy are two different legal questions, and you can win one while losing the other badly.**

The central legal fight of this era is whether ingesting copyrighted works to train a model is fair use (in the US) or falls under a text-and-data-mining exception (in the EU and UK). The suits that will shape the answer are already decided or in flight, and they rhyme. Every one turns on the same axis: was the ingestion a *transformative* use that the law permits, or an *unlicensed copy* that it does not — and, crucially, were the underlying copies even lawful in the first place?

![The copyright landscape: four landmark training-data suits](/imgs/blogs/legal-ethics-and-the-future-of-training-data-2.webp)

**The New York Times v. OpenAI and Microsoft.** The Times alleges that GPT-model outputs can reproduce its articles nearly verbatim, and that this substitutes for the original. OpenAI's defense is the classic one: training is transformative, and facts and news are not themselves copyrightable. The case is still live, and the most instructive part is procedural — the fight over whether OpenAI must retain and produce output logs, and the plaintiffs' focus on *memorization* as the smoking gun. The lesson for engineers is blunt: verbatim regurgitation is the exposure. A model that emits training text on demand hands the plaintiff their strongest exhibit. Deduplication and memorization-mitigation, covered in [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale), are not just quality tools; they are legal risk controls.

**Getty Images v. Stability AI.** Getty alleged that more than twelve million of its images and captions were scraped to train Stable Diffusion, and pointed to Getty watermarks resurfacing in generated images as evidence. The UK High Court ruling in November 2025 was a landmark — and a partial anticlimax. The court found that the model weights are *not* a "copy" of the training images under UK copyright law (a trained parameter set stores statistics, not stored pictures), and because Getty conceded the training happened outside the UK, the court never ruled on whether training-on-copyright is itself infringement. Getty prevailed only on a narrow trademark point where its watermarks appeared in outputs. The lesson cuts two ways: "the weights are not a copy" is a real and powerful defense, but *watermarks are provenance*, and provenance in the output is what let Getty prove copying happened at all.

**Bartz v. Anthropic.** This is the one that put a number on the table. Judge William Alsup ruled in June 2025 that training on lawfully acquired books is transformative fair use — a genuine green light for the *training act*. But he split the question: Anthropic had also downloaded more than seven million books from shadow libraries (LibGen and the Pirate Library Mirror), and Alsup held that acquiring and storing pirated copies was *not* fair use, regardless of what they were later used for. Anthropic settled the piracy claims in 2025 for roughly 1.5 billion USD — the largest copyright settlement in US history — covering around 500,000 works at approximately 3,000 USD each, with preliminary approval granted in September 2025. The lesson is the single most important sentence in this post: **the legality of how you acquired the data can matter more than the legality of what you did with it.** Buy the book, scan the book, and train — defensible. Grab the book from a pirate mirror and train — a nine-figure liability, even if the training itself would have been fine.

**Kadrey v. Meta.** In the same month, Judge Vince Chhabria ruled for Meta on the training question, finding its use of books to train Llama "highly transformative." But he went out of his way to say the ruling was narrow: Meta won because the plaintiffs *failed to build a record* on market harm, not because AI training is categorically lawful. He explicitly flagged that a better-argued case about market dilution — a flood of machine-generated books competing with the originals — might come out the other way. The lesson: fair use is being decided one evidentiary record at a time, and "no proven market harm" is a defense that erodes as the market fills with synthetic substitutes.

### Where this leaves you

Put the four together and a working doctrine emerges, at least in the US: the *training act* is trending toward fair use when the inputs are lawfully obtained and the outputs are transformative and do not regurgitate — but *unlawful acquisition* (piracy, breach of a paywall or terms of service) and *verbatim output* are the two live wires. In the EU, the frame is different: the [DSM Directive](https://eur-lex.europa.eu/eli/dir/2019/790/oj) grants a text-and-data-mining exception, but rightsholders can *reserve* their works from it (opt out) in a machine-readable way, and the AI Act layers transparency duties on top. Either way, the engineering takeaways are the same three: know your source's legality, dedup and mitigate memorization, and keep the receipts.

## 2. Opt-out and licensing: how data legitimately enters a model

**Senior rule of thumb: "we can technically access it" and "we are licensed to train on it" are different claims, and only the second one survives contact with a court.**

There are four routes data takes into a model, and they trade legal defensibility against cost and freshness. The whole sourcing decision — the subject of [sourcing and collecting training data](/blog/machine-learning/training-data/sourcing-and-collecting-training-data) — is really a portfolio choice across these four.

![Four routes data takes into a model, and what each costs](/imgs/blogs/legal-ethics-and-the-future-of-training-data-3.webp)

**Open web crawl, with opt-out honored.** The historical default: crawl the public web, filter it, train. The emerging etiquette that makes this more defensible is machine-readable opt-out. `robots.txt` has grown AI-specific user agents (GPTBot, Google-Extended, ClaudeBot, CCBot, and others), and a proposed `ai.txt` / `llms.txt` convention plus the IETF work on a standard AI-preferences signal aim to give sites a clean "do not train" switch. In the EU, honoring a rightsholder's TDM reservation is not just etiquette — it is the condition for the mining exception to apply.

```ini
# robots.txt — machine-readable opt-out for AI training crawlers
User-agent: GPTBot
Disallow: /

User-agent: Google-Extended
Disallow: /

User-agent: ClaudeBot
Disallow: /premium/

User-agent: CCBot
Disallow: /
```

If you are the one crawling, honoring these is cheap insurance: a documented policy that you respect opt-out signals is exactly the kind of good-faith evidence that helps in a dispute, and ignoring a machine-readable reservation is exactly the kind of bad-faith fact that hurts. Log which signals you saw and what you did, per domain.

**Licensed deals.** The clearest path, and the one the frontier is converging on. Reddit licensed its data (the Google agreement was reported at around 60 million USD per year); the Associated Press, Axel Springer, the Financial Times, Shutterstock, and numerous publishers have signed content-licensing deals with model builders. Here is the part engineers underrate: **a license buys you two things — fresh, high-quality data and, if the contract is written well, indemnity — but it does not buy you silence about provenance.** You still have to record *which* licensed corpus, under *which* terms, with *what* field-of-use and expiry. A license that lapses, or that covered training but not a new product surface, is a landmine if you cannot tell which tokens came from it.

**Synthetic generation.** Generate data with a model, train on it. On the IP axis this is the safest route — you (arguably) own the outputs, subject to the terms of whatever model produced them. It is cheap, scalable, and controllable, and [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation) is now central to post-training. Its risk is not legal but epistemic: model collapse, which we return to in the future section. And note the terms-of-use trap: training a competing model on the *outputs* of a commercial API often violates that API's terms, which is a contract problem even when it is not a copyright problem.

**User and interaction data.** Product logs, chats, feedback, and behavior. This is the strongest moat — proprietary, fresh, impossible for a competitor to replicate — and the most privacy-loaded. It requires a consent basis and terms of service that actually permit training, clear retention limits, and, in regulated jurisdictions, honoring deletion. Done right it is the most defensible *and* most valuable data you have. Done wrong it is a data-protection enforcement action.

## 3. Privacy, consent, and regulation

**Senior rule of thumb: a database can forget; a trained model cannot — so decide what a person's data is allowed to become before you train, not after they ask you to delete it.**

Privacy law was written for databases, where personal data sits in a row you can find and delete. Model weights break that model in a way that is not a technicality. Consider a right-to-be-forgotten request under the GDPR, or a deletion request under CCPA/CPRA. In a database, the fix is a `DELETE`. In a model, there is no row.

![Right-to-be-forgotten: a database row versus a weight](/imgs/blogs/legal-ethics-and-the-future-of-training-data-4.webp)

The left side of that figure is the world regulators designed for: indexed, addressable, deletable. The right side is the world you actually shipped. Once personal data is trained into the weights, it is smeared across billions of parameters with no index. You cannot locate one person's contribution, and "deleting" it honestly means retraining the run — a multi-million-dollar, weeks-long operation you will not do for a single request. Machine unlearning is an active research area, but no production technique reliably removes a specific person's data from a large model without unacceptable quality loss. So the honest engineering posture is: **treat "will this personal data end up in the weights?" as an irreversible decision made at ingestion time.**

That reframes the whole privacy problem as a filtering-and-consent problem *before* training, which is exactly what [PII safety and toxicity filtering](/blog/machine-learning/training-data/pii-safety-and-toxicity-filtering) is for. The defensible pattern is: detect and redact PII in the pipeline; train only on data with a lawful basis (consent, or a legitimate-interest analysis you actually wrote down); keep the raw, deletable copy in a database where a deletion request *can* be honored; and design so that a deletion request updates the source of truth and is excluded from the *next* training run, since it cannot be pulled from the current one.

### The regulatory map

- **GDPR (EU)** — personal data needs a lawful basis; data-subject rights (access, deletion, objection) apply; fines reach up to 20 million EUR or 4% of global annual turnover, whichever is higher. Several European authorities have already probed AI training on personal data.
- **CCPA/CPRA (California)** — disclosure and deletion rights, opt-out of "sale/sharing," and sensitive-data limits.
- **AI Act (EU)** — transparency and documentation obligations for general-purpose models, including summaries of training-data sources.
- **Biometric and voice law** — this is where the [speech posts](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety) come home. A person's voice is biometric-adjacent personal data. Illinois's BIPA has produced enormous settlements over biometric data used without written consent, and voice cloning has made "we recorded a consented dataset" and "we scraped voices off the internet" wildly different legal positions. If you train text-to-speech or voice-conversion models, consented, documented voice data is not a nice-to-have; it is the difference between a product and a class action. The provenance and consent record for *voice* is as load-bearing as any copyright record.

The through-line: personal data, and especially biometric data like voice and face, is the category where "we cannot delete it from the weights" collides hardest with a legal right to have it deleted. The only real defense is upstream — consent and filtering before the token ever reaches the optimizer.

## 4. Bias and representation

**Senior rule of thumb: a model's defaults are its corpus's majority, and you will not discover the skew unless you measure composition before you train and behavior after.**

Bias in a model is not usually a bug someone introduced. It is the corpus's composition, faithfully learned. If the pretraining mix is ninety percent English and heavily weighted toward a handful of Western, online, English-speaking demographics, the model's *default* assumptions — about names, dialects, holidays, legal systems, medical norms — inherit that skew, silently, and only surface when a user outside the majority hits them.

![Corpus composition becomes model behavior](/imgs/blogs/legal-ethics-and-the-future-of-training-data-5.webp)

The figure is the whole causal chain and the intervention points. Composition becomes behavior in training; you cannot un-see it after the fact, but you *can* measure and mitigate. This is the same negotiation between quality, diversity, and quantity from the opener — bias is what happens when you optimize the first and third and neglect the second. The moves:

- **Measure composition before training.** You cannot manage what you never counted. Produce per-slice statistics: language distribution, geographic and source distribution, domain mix, and, where lawful and appropriate, coarse demographic proxies. This connects directly to [data mixing, domain weighting, and curriculum](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum) — your domain weights *are* your bias policy, whether or not you named it that.
- **Measure behavior after training.** Per-slice evaluation, not a single aggregate number. A model at ninety percent average accuracy that is at sixty percent for a language spoken by hundreds of millions is not a ninety-percent model to those users. Standard bias and fairness benchmarks help, but the highest-signal test is a per-language, per-region eval you build for *your* users.
- **Mitigate deliberately.** Upsample and augment under-represented slices, curate targeted data for the tail, and reweight the mix. This trades some average-case perplexity for tail coverage — a real cost, made on purpose, recorded in the data card.

The failure mode to internalize: bias you never measured is bias you shipped. The absence of a composition report is not the absence of a composition problem; it is the absence of *knowledge* of one.

## 5. Documentation: the data card is the deliverable

**Senior rule of thumb: if the answer to "what is in this dataset and are we allowed to use it?" lives only in someone's head, you do not have a dataset — you have a rumor with tensors attached.**

The single highest-leverage governance artifact is documentation, and the field has converged on a format. [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et al.), [Data Statements for NLP](https://aclanthology.org/Q18-1041/) (Bender and Friedman), and model cards / data cards are all variants of the same idea: a structured document that travels with the corpus and answers the questions a lawyer, an auditor, and a future engineer will ask. Anatomically, a good data card has eight fields.

![Anatomy of a data card: eight fields you must fill in](/imgs/blogs/legal-ethics-and-the-future-of-training-data-6.webp)

The shaded fields — licensing, known biases, PII handling — are the ones a lawyer or auditor reads first, and the ones most teams leave blank. Do not. Here is a concrete, fillable template you can drop into a repository and require as a merge gate for any dataset that enters the pipeline.

```yaml
# data_card.yaml — datasheet for a training corpus
name: webmix-en-2026q2
version: 1.4.0
owner: data-platform@example.com
created: "2026-06-15"

provenance:
  sources:
    - id: commoncrawl-2026-18
      kind: web-crawl
      obtained: "public crawl, opt-out signals honored 2026-05"
      license: mixed / unlicensed          # the honest answer
      opt_out_respected: [robots.txt, Google-Extended, ai.txt]
    - id: licensed-news-ap
      kind: licensed
      obtained: "AP content license #2026-114"
      license: commercial-training          # field-of-use: pretraining only
      expires: "2028-01-01"
      indemnity: true
    - id: synth-math-v3
      kind: synthetic
      obtained: "generated in-house; generator: internal-70b"
      license: owned

composition:
  total_tokens: 2_240_000_000_000
  languages: { en: 0.86, es: 0.05, zh: 0.03, other: 0.06 }
  domains:   { web: 0.61, code: 0.14, books_licensed: 0.10, math_synth: 0.15 }
  geo_note: "web skews North America / Western Europe; see bias section"

collection:
  method: "distributed crawl + licensed batch ingest + synthetic gen"
  crawler_useragent: "ExampleBot/1.2 (+https://example.com/bot)"
  opt_out_policy: "robots.txt + ai.txt checked per fetch; refusals logged"

filtering:
  stages: [text_extraction, dedup_minhash, quality_classifier,
           decontamination, pii_redaction, toxicity_filter]
  decontamination_against: [mmlu, gsm8k, humaneval, internal_evals]
  # each stage links to the post that specifies it

licensing:
  overall_status: "mixed: licensed + unlicensed-web + owned-synthetic"
  unlicensed_web_fraction: 0.61
  redistribution: prohibited
  review: "legal sign-off 2026-06-14, ticket LEGAL-2211"

known_biases:
  measured: true
  report: "reports/webmix-en-2026q2-composition.html"
  summary: "English + Western skew; low-resource langs under 1% each"
  mitigations: ["upsample es/zh 1.5x", "synthetic tail augmentation"]

pii_handling:
  detection: "regex + NER (names, emails, phones, IDs)"
  action: redact
  retention: "raw deletable copy in warehouse; deletion requests -> next run"
  biometric: "no voice/face data in this corpus"

maintenance:
  changelog: "CHANGELOG.md"
  update_cadence: quarterly
  contact: data-platform@example.com
```

Two things make this real rather than theater. First, it is a **merge gate**: no source enters the pipeline without its rows in the card, enforced in CI, the same way you would not merge code without tests. Second, it links each filtering stage to the post that specifies it, so the card is not a summary but an index into the actual pipeline.

### Provenance at the content layer: C2PA

Documentation answers "what is in the corpus." A complementary standard answers "what is this individual asset, and where did it come from" — [C2PA content credentials](https://c2pa.org/), the cryptographically-signed provenance manifests attached to images, audio, and video. This ties directly to [safety, watermarking, and provenance for image generation](/blog/machine-learning/image-generation/safety-watermarking-and-provenance) and to the [voice-safety](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety) work: if your inputs carry content credentials, you can filter on them (exclude assets marked "AI-generated" to fight collapse, or "do not train"), and if your *outputs* carry them, you help the next crawler downstream do the same.

```bash
# Inspect the provenance manifest on an asset before ingesting it
c2patool asset.jpg --detailed

# In the pipeline: skip assets whose credentials flag them
#   - as AI-generated (avoid synthetic feedback loops), or
#   - with a "do not train" assertion.
c2patool asset.jpg | jq -e '.manifests[].assertions[]
  | select(.label=="c2pa.training-mining")
  | .data.entries."c2pa.ai_training".use == "notAllowed"' \
  && echo "SKIP: training reserved" || echo "OK to ingest"
```

### Worked scenario: scoring candidate sources with a license-risk matrix

You have a fixed token budget and four candidate sources for a new pretraining run. Do not argue about them in a meeting — score them. Rate each on legal risk, defensibility (could you explain it to a court with a straight face), cost, and freshness, then decide.

| Candidate source | Legal risk | Defensibility | Cost | Freshness | Verdict |
| --- | --- | --- | --- | --- | --- |
| Open web crawl, opt-out **ignored** | High | Low | Low | High | Avoid — the "free" tokens carry unbounded liability |
| Open web crawl, opt-out **honored + logged** | Medium | Medium | Low | High | Use as bulk base; document the policy |
| Licensed corpus (AP, Reddit, publisher) | Low | High | High | High | Buy the high-signal core; keep the contract terms |
| Synthetic (self-generated) | Low (IP) / Medium (collapse) | Medium | Low | n/a | Use for targeted skills; cap the fraction, anchor to real data |
| User / interaction data (with consent) | Low–Medium (privacy) | High | Low | High | The moat — invest here; get consent and retention right |
| Shadow library (Books3 / LibGen) | Extreme | None | Low | Medium | **Never.** This is the 1.5-billion-USD line |

The decision that falls out is almost always the same shape: a documented, opt-out-honoring web base for scale, a licensed core for high-signal domains, synthetic for targeted skills with a hard cap, and user data as the compounding moat — with the pirated-shadow-library row struck through in red. That last row is not a judgment call. Bartz v. Anthropic priced it.

And a short license-risk checklist to run before *any* source enters the pipeline:

```
# license-risk checklist — every source runs this gauntlet before ingestion
[ ] Do we know exactly where this came from (URL / vendor / generator)?
[ ] Was it lawfully acquired (not pirated, not behind a bypassed paywall/ToS)?
[ ] Do we have a license or lawful basis for TRAINING specifically?
[ ] Does that basis cover our product surface, not just research?
[ ] Have we honored machine-readable opt-out (robots/ai.txt/TDM)?
[ ] Is PII detected and handled; is there a deletable source of truth?
[ ] For voice/face/biometric data: do we have explicit consent?
[ ] Is all of the above written in the data card BEFORE ingestion?
```

If any box is unchecked, the source does not enter the run. That is the whole governance program in eight lines.

## 6. Data versioning and lineage: which data trained which checkpoint

**Senior rule of thumb: if you cannot answer "exactly what data produced this checkpoint" months later, you cannot reproduce it, defend it, or safely fix it — you can only apologize for it.**

Everything above collapses into one operational requirement: lineage. For any shipped checkpoint you must be able to trace back through the mix, the filters, and the sources to the exact tokens that made it. Not approximately. Exactly enough to answer a subpoena, reproduce a result, or excise a source you later discover was poisoned or unlicensed.

![Provenance and lineage: which data trained this checkpoint](/imgs/blogs/legal-ethics-and-the-future-of-training-data-7.webp)

The pattern is a **lineage manifest** produced alongside every checkpoint: content hashes of each source snapshot, the versioned filtering config, the domain-mix weights, the data-card version, and the training config (seed included). Tools like [DVC](https://dvc.org/), Git-LFS, and dataset-hashing conventions exist for exactly this; the discipline matters more than the tool.

```yaml
# lineage.yaml — emitted with checkpoint v1.4, hash abc123...
checkpoint: model-v1.4
checkpoint_sha256: abc123...
trained_on:
  data_card: webmix-en-2026q2 @ 1.4.0
  sources:
    - id: commoncrawl-2026-18
      snapshot_sha256: 9f2a...
    - id: licensed-news-ap
      snapshot_sha256: 1c77...
      license_ref: "AP #2026-114 (expires 2028-01-01)"
    - id: synth-math-v3
      snapshot_sha256: 4e0b...
  mix_weights: { web: 0.61, code: 0.14, books_licensed: 0.10, math_synth: 0.15 }
  filter_config_sha256: 77de...
train_config: { seed: 1337, tokens: 2.24e12, ckpt_step: 480000 }
```

With this in place, the four questions become answerable in minutes instead of never. "Was this checkpoint trained on the AP corpus after the license expired?" — check the snapshot date against the license ref. "A source was found to contain material we must remove — which checkpoints are affected?" — grep the lineage manifests for its hash. "Reproduce the run" — you have the seed, the config, and the source hashes. Reproducibility is not an academic nicety here; it is the mechanism that makes every other governance promise enforceable. This is the operational spine that connects [decontamination and benchmark leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage), [data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning), and the mix decisions into an auditable whole.

## 7. The future of training data

**Senior rule of thumb: fresh human text is a depleting resource, and the strategies that replace it — synthetic, licensed, interaction — each have a catch you must design around.**

Here is the structural shift the next few years are about. The stock of high-quality, freely-crawlable human text is finite, and frontier runs are consuming it faster than humans produce it. Estimates of the "peak data" moment vary, but the direction is not in doubt: the era of scaling by scraping more of the open web is ending. What replaces it is a different mix, and the mix inverts.

<figure class="blog-anim">
<svg viewBox="0 0 640 380" role="img" aria-label="The training-data mix shifts from mostly fresh human web text today to a future dominated by synthetic, licensed, and interaction data, capped by a human-text ceiling" style="width:100%;height:auto;max-width:820px">
<style>
.fx-axis{stroke:var(--border,#d1d5db);stroke-width:2}
.fx-ceil{stroke:var(--text-secondary,#6b7280);stroke-width:2;stroke-dasharray:8 6}
.fx-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.fx-sm{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.fx-human{fill:var(--surface,#e9ecef);stroke:var(--border,#d1d5db);stroke-width:1.5}
.fx-syn{fill:var(--accent,#6366f1)}
.fx-lic{fill:var(--text-secondary,#94a3b8)}
.fx-int{fill:var(--border,#cbd5e1)}
.fx-warn{font:700 13px ui-sans-serif,system-ui;fill:#b45309;text-anchor:middle}
@keyframes fx-a{0%,42%{opacity:1}55%,95%{opacity:0}100%{opacity:1}}
@keyframes fx-b{0%,42%{opacity:0}55%,95%{opacity:1}100%{opacity:0}}
.fx-A{animation:fx-a 12s ease-in-out infinite}
.fx-B{animation:fx-b 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.fx-A{animation:none;opacity:1}.fx-B{animation:none;opacity:0}}
</style>
<line class="fx-axis" x1="80" y1="300" x2="600" y2="300"/>
<text class="fx-sm" x="40" y="180" transform="rotate(-90 40 180)">share of training tokens</text>
<line class="fx-ceil" x1="80" y1="120" x2="600" y2="120"/>
<text class="fx-sm" x="500" y="112">human-text ceiling</text>
<g class="fx-A">
<rect class="fx-human" x="250" y="130" width="140" height="170"/>
<rect class="fx-lic" x="250" y="108" width="140" height="22"/>
<rect class="fx-syn" x="250" y="92" width="140" height="16"/>
<rect class="fx-int" x="250" y="78" width="140" height="14"/>
<text class="fx-lbl" x="320" y="220">fresh human</text>
<text class="fx-lbl" x="320" y="240">web text</text>
<text class="fx-lbl" x="320" y="340">today (~2024)</text>
</g>
<g class="fx-B">
<rect class="fx-human" x="250" y="215" width="140" height="85"/>
<rect class="fx-lic" x="250" y="150" width="140" height="65"/>
<rect class="fx-syn" x="250" y="60" width="140" height="90"/>
<text class="fx-lbl" x="320" y="264">human</text>
<text class="fx-lbl" x="320" y="188">licensed</text>
<text class="fx-lbl" x="320" y="112">synthetic</text>
<text class="fx-warn" x="320" y="46">model-collapse risk</text>
<text class="fx-lbl" x="320" y="340">future (~2030+)</text>
</g>
<figcaption>The training-data mix inverts: fresh human web text plateaus at a ceiling while synthetic, licensed, and interaction data grow to dominate, and unchecked synthetic feedback invites model collapse.</figcaption>
</figure>

Watch the animation: the tall human-text column plateaus at the ceiling while synthetic, licensed, and interaction data grow to fill the gap. Each replacement has a catch.

**Synthetic dominance, and the model-collapse caveat.** Synthetic data is already the workhorse of post-training and is climbing in pretraining. The danger, when models are trained on the outputs of models trained on the outputs of models, is [model collapse](/blog/machine-learning/training-data/synthetic-data-generation): the tails of the distribution thin out, rare knowledge and minority modes vanish, and successive generations regress toward a blander mean. The published research (the "curse of recursion" line of work) shows collapse is real when synthetic data *replaces* real data recursively — but that it is largely avoidable when synthetic data *accumulates alongside* a fixed anchor of real human data rather than substituting for it. The practical rule: synthetic is an amplifier, not a substitute. Cap its fraction, always keep a real-data anchor, and use provenance signals (C2PA "AI-generated" flags) to avoid unknowingly recycling your own outputs off the web.

**Licensed and proprietary data as the new moat.** As open crawl saturates and its legal risk becomes clearer, the differentiated data moves behind contracts and into products. Expect data marketplaces, more publisher and platform licensing deals, and rising value for *interaction data* — the traces of real users solving real tasks, which no competitor can scrape. The strategic consequence for a data team: the skill of the next decade is less "crawl and clean more web" and more "acquire, license, and generate defensible data, then document and version it." The moat is not the weights. It is the pipeline that can prove where the weights came from.

## Governance troubleshooting: symptom, cause, fix

These are the four ways a data-governance program actually fails in production. Each is a symptom you will recognize, a root cause, and the fix.

### Symptom: un-provenanced data you cannot defend in discovery
**Cause.** The corpus was assembled by scripts that grabbed sources without recording where each came from or under what right; the knowledge lived in a departed engineer's head. When the subpoena or the audit arrives, you can describe the pipeline but not the provenance of any specific token.
**Fix.** Make the data card a merge gate: no source enters without provenance, license, and acquisition rows. Backfill for existing corpora by hashing snapshots and reconstructing sources from crawl logs. If a source truly cannot be provenanced, treat it as unlicensed-web at best — and quarantine it out of any run you need to defend. Provenance is cheap to add at ingestion and nearly impossible to reconstruct after the fact.

### Symptom: a deletion request you cannot honor because the data is in the weights
**Cause.** Personal (or biometric) data was trained into the model with no upstream consent gate and no deletable source of truth. Now a data subject exercises a right you have no mechanism to satisfy, because there is no row to delete and retraining is infeasible.
**Fix.** Move the decision upstream permanently: detect and redact PII before training; train only on data with a lawful basis; keep the raw copy in a deletable warehouse that *is* the source of truth; and wire deletion requests to update that store and exclude the data from the *next* run. You cannot pull it from the current weights, so make sure the current weights should not have contained it in the first place. For voice and face data, require explicit consent — no exceptions.

### Symptom: a bias you shipped because you never measured composition
**Cause.** The mix was chosen for quality and quantity; diversity was never quantified; no per-slice eval existed. The skew surfaced only when users outside the majority reported systematically worse results — publicly.
**Fix.** Add composition measurement as a pre-training gate (language, geo, domain, source distributions in the data card) and per-slice evaluation as a pre-ship gate. Make the domain weights an explicit, reviewed bias policy. Then mitigate on purpose — upsample and augment the tail — and record the tradeoff. You cannot promise an unbiased model, but you can promise you measured, which is the difference between a defensible product and a headline.

### Symptom: "we cannot reproduce which data trained this checkpoint"
**Cause.** No lineage manifest. The mix weights lived in a notebook, the source snapshots were overwritten, the filter config drifted. When a source turns out to be poisoned, leaked into a benchmark, or unlicensed, you cannot tell which checkpoints inherited it — so you cannot scope the blast radius or fix it surgically.
**Fix.** Emit a lineage manifest with every checkpoint: source snapshot hashes, filter-config hash, mix weights, data-card version, and training seed. Version data like code with DVC or equivalent. The payoff is direct: any future "which checkpoints touched source X" question becomes a grep, and any "reproduce this run" request becomes possible. Reproducibility is the mechanism that makes every other governance promise real.

## Case studies from the frontier

### 1. The New York Times v. OpenAI — memorization is the exhibit
The most engineering-relevant fact of this case is not the copyright theory; it is the discovery fight over output logs and the plaintiffs' reliance on near-verbatim reproduction of Times articles. The lesson generalizes past this one suit: a model that regurgitates training text is a model that manufactures evidence against itself. Aggressive deduplication, memorization audits (probe the model for verbatim spans of known training documents before shipping), and log-retention policies you actually decided on — rather than discovered under subpoena — are the controls. Treat memorization as a shippable defect with a legal blast radius, because it is one.

### 2. Getty v. Stability — provenance in the output cuts both ways
Getty largely lost the UK copyright claim because the court held model weights are not a stored copy and the training happened outside the jurisdiction. But Getty got *into* court at all because its watermarks resurfaced in generated images — provenance in the output proved copying occurred. The double lesson: "the weights are not a copy" is a genuine, powerful defense, and it is worth understanding precisely; but visible provenance markers (watermarks, and now C2PA) are exactly how downstream copying gets detected. If you generate, strip nothing you are not entitled to strip, and expect your inputs' provenance markers to survive into your outputs unless you handle them deliberately.

### 3. Bartz v. Anthropic — how you got it matters more than what you did with it
Alsup's split ruling is the cleanest statement of the doctrine: training on lawfully acquired books was fair use, but downloading seven-million-plus books from pirate libraries was not, and the roughly 1.5-billion-USD settlement priced the difference. For a data team this is the single most actionable case in the set. It means the acquisition step — not the training step — is where the largest, most avoidable liability lives. Buy it, license it, or generate it; do not pirate it, even if the eventual training use would have been defensible. No model quality gain justifies a nine-figure acquisition-side liability.

### 4. Re-LAION-5B — you must be able to remediate a dataset
In late 2023, the [LAION-5B](/blog/machine-learning/training-data/image-text-pairs-at-scale) image-text dataset was pulled after the Stanford Internet Observatory found child-sexual-abuse-material URLs among its billions of scraped links. The response that matters here is what came next: in 2024, Re-LAION-5B was re-released with the offending content removed using takedown lists from child-safety organizations. The lesson is not "web scraping is dangerous" (though it is); it is that **a dataset must be remediable.** Because LAION is a list of URLs with content hashes, offending entries could be identified and removed, and downstream users could re-derive a clean version. A corpus you cannot audit and cannot surgically fix is a corpus you cannot govern. Build for remediation: content-addressable sources, lineage, and the ability to exclude-and-rebuild.

### 5. The Reddit and publisher licensing wave — what a license actually buys
When Reddit licensed its data (the Google deal reported around 60 million USD per year) and publishers from the AP to Axel Springer signed with model builders, the market answered the "just scrape it" question with a price. What these deals buy is instructive: fresh, high-signal, structured data *and*, when negotiated well, indemnification and clear field-of-use — but not an escape from documentation. The teams that benefit are the ones who record which licensed corpus, under which terms, with which expiry, in the data card and the lineage manifest. A license you cannot map to specific tokens in a specific checkpoint is a license you cannot rely on when it matters. The deal is the beginning of provenance, not a substitute for it.

## How defensible does your data need to be?

Not every project needs a frontier-lab governance stack. Calibrate to the stakes.

**Invest heavily in governance when:**
- You are training a model you will ship to users, sell, or expose via an API.
- Your data includes personal data, biometric/voice data, or user-generated content.
- You operate in or serve the EU, California, or other rights-heavy jurisdictions.
- The model or its outputs could plausibly attract a lawsuit or an audit — which, at any real scale, is all of them.
- Reproducibility matters for safety, compliance, or debugging (again: all of them).

**You can go lighter when:**
- The corpus is fully self-generated synthetic or fully public-domain / permissively-licensed, and documented as such.
- It is a throwaway research experiment that will never ship and never touch personal data.
- You are fine-tuning on a small, licensed, fully-documented internal dataset.

Even in the light case, write the data card. It costs an hour and it is the artifact you will most wish you had.

## The whole lifecycle, in one line

Twenty-five posts, and they reduce to a single arc. You [decide data is the lever](/blog/machine-learning/training-data/why-data-decides-the-model); you [source it](/blog/machine-learning/training-data/sourcing-and-collecting-training-data), [extract](/blog/machine-learning/training-data/text-extraction-and-boilerplate-removal) and [clean](/blog/machine-learning/training-data/language-identification-and-heuristic-quality-filters) it, [dedup](/blog/machine-learning/training-data/deduplication-at-scale) and [decontaminate](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage) it, [filter](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering), [select](/blog/machine-learning/training-data/data-selection-and-pruning), and [mix](/blog/machine-learning/training-data/data-mixing-domain-weighting-and-curriculum) it; you [generate synthetic data](/blog/machine-learning/training-data/synthetic-data-generation) and build [instruction](/blog/machine-learning/training-data/instruction-tuning-data) and [preference](/blog/machine-learning/training-data/preference-data-for-alignment) sets; you adapt it per modality for [images](/blog/machine-learning/training-data/data-for-text-to-image-diffusion), documents, and speech; you [measure quality](/blog/machine-learning/training-data/measuring-data-quality) throughout. And then — this post — you make it *defensible*: provenanced, consented, measured, documented, versioned.

The opener said the model is a lossy compression of its training set. The closer adds the corollary that governs everything above: **you are responsible for what you compressed, and one day you will have to prove it.** The teams that will still be shipping in five years are not the ones with the cleverest architecture. They never were. They are the ones who can answer the four questions — provenance, consent, representation, reproducibility — for every checkpoint they ever released. Build the pipeline so those answers exist. That is the whole job, and it is the same job the whole series has been describing, all the way down.

## Further reading

- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) — Gebru et al.; the canonical documentation format.
- [Data Statements for NLP](https://aclanthology.org/Q18-1041/) — Bender and Friedman; the NLP-specific complement.
- [C2PA / Content Credentials](https://c2pa.org/) — the content-provenance standard for images, audio, and video.
- [The Curse of Recursion / model collapse](https://arxiv.org/abs/2305.17493) — Shumailov et al.; the foundational model-collapse paper.
- [EU DSM Directive, Articles 3–4](https://eur-lex.europa.eu/eli/dir/2019/790/oj) — the text-and-data-mining exception and opt-out.
- Sibling posts: [why data decides the model](/blog/machine-learning/training-data/why-data-decides-the-model), [PII safety and toxicity filtering](/blog/machine-learning/training-data/pii-safety-and-toxicity-filtering), [synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation), [safety, watermarking, and provenance](/blog/machine-learning/image-generation/safety-watermarking-and-provenance), and [audio deepfakes, watermarking, and voice safety](/blog/machine-learning/audio-generation/audio-deepfakes-watermarking-and-voice-safety).
