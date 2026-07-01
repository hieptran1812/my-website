---
title: "Instruction-Tuning Data: Why a Thousand Good Examples Beat a Million Noisy Ones"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "SFT is where a base model learns to behave, and the data that teaches it is smaller, stranger, and more fragile than pretraining data. This is the principal-engineer's guide to sourcing, curating, deduplicating, decontaminating, and balancing instruction-tuning data — with the LIMA result, the Alpaca/FLAN/Tulu recipes, runnable curation code, and the failure modes that quietly wreck a fine-tune."
tags: ["training-data", "instruction-tuning", "sft", "lima", "alpaca", "flan", "tulu", "chat-template", "data-decontamination", "self-instruct", "llm-fine-tuning", "data-quality"]
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 36
---

There is a moment in every fine-tuning project where someone says "let's just throw more data at it," and that moment is almost always a mistake. Instruction tuning — supervised fine-tuning, SFT, whatever your team calls it — is the stage where a raw base model, fluent but feral, learns to answer a question instead of continuing it, to stop when it is done, to refuse the unsafe thing, to format code in a block. It is the difference between a model that *knows things* and a model you can *talk to*. And it turns out that this transformation is taught by a shockingly small amount of data: not the trillions of tokens of pretraining, but often a few thousand to a few hundred thousand carefully chosen examples.

The counterintuitive part — the part that trips up nearly every team the first time — is that at this stage, *more data usually makes the model worse*. A million machine-generated instruction pairs will teach your model to be verbose, sycophantic, and weirdly repetitive. A thousand hand-curated ones will teach it to be sharp. This post is about why that is true, and about the unglamorous engineering that separates a good SFT set from a bad one: where the data comes from, how to format it so the model learns behavior and not boilerplate, how to strip out the duplicates and the leaked eval questions, and how to balance the mix so one skill does not eat all the others.

![Pretraining data versus SFT data: two completely different shapes](/imgs/blogs/instruction-tuning-data-1.webp)

The diagram above is the mental model, and it is worth sitting with before we go further. Pretraining data is *raw text* — a web page, a book, a code file — and the objective is next-token prediction over every token, at a scale of trillions. SFT data is *instruction-response pairs* — one task, one gold answer — and the objective is still next-token prediction, but with the loss computed *only on the response*, at a scale of thousands to low millions. The shapes could not be more different. Pretraining teaches the model the distribution of language and the facts embedded in it; SFT teaches the model which slice of that distribution to produce when a human asks for something. Everything in this post follows from that distinction.

## Why instruction-tuning data is different

Let me start with the mismatch between how people *think* about SFT data and how it actually behaves, because the assumptions you bring from pretraining will actively hurt you here.

| Assumption from pretraining | Naive view of SFT | The reality |
| --- | --- | --- |
| More tokens is strictly better | More instruction pairs is strictly better | Past a few thousand clean pairs, quality and diversity dominate; extra noisy data *regresses* the model |
| Duplicates are mildly wasteful | Duplicates just cost a little compute | A duplicated instruction gets over-weighted and the model memorizes its exact phrasing, hurting generalization |
| Web scale hides individual bad docs | A few bad answers wash out | A single systematically-wrong response pattern (wrong format, sycophancy, refusal-avoidance) is *learned* because there is so little data |
| Contamination inflates eval a little | Eval leakage is a pretraining problem | Instruction sets leak evals too — often worse, because they are curated from the same public tasks your benchmarks use |
| Format is cosmetic | The model reads through formatting | The model over-fits the exact chat template; off-format inputs at inference cause refusals and garbled output |

The through-line is *leverage*. Because SFT data is small, every example carries enormous weight. In pretraining, one bad document is one part in a hundred billion. In SFT, one systematically-bad *pattern* across a few hundred examples is a behavior your users will see every single day. The model is not averaging over a galaxy of text anymore; it is imitating a small, sharply-defined set of demonstrations. That is why curation, not collection, is the whole game.

There is a second, subtler difference: the *objective is behavioral, not distributional*. Pretraining asks "what token comes next in the wild?" SFT asks "given this instruction, what is the *right* response?" — and "right" is a value judgment baked into your data by whoever wrote the answers. If your annotators write long-winded answers, your model will be long-winded. If your teacher model hedges, your model will hedge. You are not learning facts here; you are learning *taste*. And taste does not scale by volume — it scales by curation.

> The base model already knows almost everything it will ever know. SFT does not teach it facts; it teaches it manners. And manners are learned from a few good role models, not from a mob.

This is also the point in the pipeline where the earlier stages pay off or betray you. If your pretraining corpus was well-built — see [why data decides the model](/blog/machine-learning/training-data/why-data-decides-the-model) — the base model has the latent capability, and SFT just has to surface it. If not, no amount of instruction data will conjure a skill that was never in the pretraining mix. SFT is a *steering* stage, not a *teaching* stage, and steering a car that has no engine gets you nowhere.

## Where SFT data comes from

There are four broad families of instruction-tuning data, and most real datasets are a blend of them. Knowing the provenance of each pair in your set is not bureaucratic hygiene — it directly predicts the failure modes you will hit, because each source fails in a characteristic way.

![The four families of SFT data compared across origin, diversity, quality, scale, and cost](/imgs/blogs/instruction-tuning-data-2.webp)

The matrix above compares them on the axes that actually matter when you are assembling a set. Read it as a menu of tradeoffs, not a ranking — the right blend depends on what you are optimizing for and what you can afford. Let me walk each family.

### Human-written data (Dolly, OpenAssistant)

The gold standard, and the most expensive. Real people write instructions and real people (often different people) write the responses. **Databricks Dolly** (15,000 examples) was written by Databricks employees across seven task categories — brainstorming, classification, closed QA, generation, information extraction, open QA, summarization — specifically so the dataset could be released under a permissive license, unencumbered by the "do not train competitors" terms that hang over model-distilled data. **OpenAssistant (OASST1)** went further: a crowd-sourced, multi-turn conversation tree with more than 160,000 messages in dozens of languages, where volunteers wrote prompts, wrote replies, and *ranked* replies — giving you both SFT data and preference data from the same collection effort.

The strength of human data is diversity and authenticity: real people ask weird, specific, off-distribution things that a model-generated pipeline never would. The weakness is *unevenness*. Crowd workers vary wildly in skill; some answers are brilliant, some are lazy, some are subtly wrong. Human data almost always needs a quality-filtering pass on top, which is where the classifier-and-perplexity machinery from [quality filtering](/blog/machine-learning/training-data/classifier-and-perplexity-based-quality-filtering) reappears, aimed now at responses instead of documents.

**The senior rule of thumb: human data sets the ceiling on quality but not the floor. Filter it as hard as you would filter the web.**

### Self-Instruct and Alpaca: distilling a strong teacher

This is the technique that democratized instruction tuning, and also the one that causes the most trouble. **Self-Instruct** (Wang et al., 2022) starts from a small seed set of human-written tasks — the original used 175 — and bootstraps a much larger set by prompting a strong model to generate new instructions in the same style, then generate responses to them, then filter. **Alpaca** (Stanford, 2023) applied exactly this recipe: 175 seeds, `text-davinci-003` as the teacher, and out came 52,000 instruction-response pairs that were used to fine-tune LLaMA-7B into something that behaved, at a glance, like a chat model — for a few hundred dollars of API calls.

![The Self-Instruct / Alpaca distillation pipeline, from 175 seeds to 52k pairs](/imgs/blogs/instruction-tuning-data-3.webp)

The pipeline above shows the loop. A few-shot prompt is sampled from the seed pool; the teacher generates a new instruction; the teacher generates a response; a filter removes near-duplicates, over-long or over-short generations, and instructions that are not really tasks (the original Self-Instruct filtered instructions with ROUGE-L overlap above 0.7 against any existing instruction). The surviving pairs are added back to the pool, and the loop continues. The magic is the *bootstrap*: 175 human examples become 52,000 through the teacher's generative capacity.

The limitations are equally important, and you should internalize them before you build a distillation pipeline of your own:

- **The student can never exceed the teacher.** Every capability, every bias, every stylistic tic of the teacher model is copied into the student. If the teacher hedges, over-apologizes, or writes ten-paragraph answers to yes/no questions, so will your student. This is why models distilled from a verbose teacher are themselves verbose.
- **Errors are laundered into looking authoritative.** The teacher's confident-but-wrong answers become training targets, and the student learns to be confidently wrong *in the same places*. There is no ground-truth check in the loop.
- **Diversity collapses toward the teacher's defaults.** Machine-generated instructions cluster around the kinds of tasks the teacher likes to produce. You get a hundred variations of "write a poem about X" and very few genuinely hard, weird, or adversarial prompts.
- **Legal encumbrance.** Data generated by a commercial model usually carries terms prohibiting its use to train competing models. Alpaca itself was released for research only for exactly this reason.

Self-Instruct is not bad — it is *cheap*, and cheap is a real advantage. But treat distilled data as a *volume filler that needs aggressive quality control*, not as a quality source. The whole point of the LIMA result, which we will get to, is that a small pile of this stuff loses to a smaller pile of the good stuff.

### FLAN: academic tasks templated into instructions

The **FLAN** collection (and its descendants FLAN-T5, FLAN 2022) takes a completely different route. Instead of generating instructions, it takes hundreds of *existing academic NLP datasets* — sentiment classification, natural language inference, question answering, summarization, translation — each with real gold labels, and wraps every example in one of many natural-language instruction templates. A sentiment example like `(text="I loved it", label=positive)` becomes a dozen different instructions: "Is the following review positive or negative? ...", "Classify the sentiment: ...", "Would you say this person enjoyed the movie? ...", and so on.

The result is enormous — the FLAN 2022 collection is on the order of 15 million examples across 1,800-plus tasks — and its responses are *grounded in real labels*, not a teacher's guesses. That is FLAN's great strength: the answers are correct because they came from datasets built by researchers who cared about correctness. Its weakness is the flip side: templated academic data is *stilted*. The instructions all sound like a benchmark, the answers are often single words or short spans, and a model trained purely on FLAN is excellent at NLP tasks and stiff as a board in open conversation. FLAN teaches *task-following*; it does not teach *chat*.

**The senior rule of thumb: FLAN-style data is the cheapest way to buy broad task coverage with correct labels, but it must be mixed with conversational data or your model will answer everything like a multiple-choice exam.**

### Tulu: the curated mixture

**Tulu** (Allen Institute for AI) is not a single-source dataset — it is a *recipe for mixing sources*, and it is the most useful mental model for how modern instruction tuning is actually done. The Tulu papers systematically studied which combinations of open datasets — FLAN, a chain-of-thought subset, Dolly, OpenAssistant, code data, GPT-distilled data, hard-reasoning data — produce the best all-around model, and the headline finding was blunt: *no single dataset wins on everything, and the best model comes from a deliberate blend.* Tulu mixes human data for authenticity, FLAN for task breadth, code and math for reasoning, and a controlled amount of distilled data for conversational polish, then dedups and decontaminates the whole thing.

The Tulu philosophy is the one to adopt: **you are not choosing a source, you are designing a mixture.** The rest of this post is largely about how to design that mixture well — which duplicates to remove, which eval questions to keep out, and how to balance the categories so the blend actually delivers on all its skills at once.

## The LIMA result: quality and diversity beat quantity

If you remember one empirical result from this entire post, make it this one. **LIMA** ("Less Is More for Alignment," Zhou et al., 2023) fine-tuned a 65B-parameter LLaMA on just **1,000** carefully curated instruction-response pairs — no RLHF, no preference optimization, just supervised fine-tuning on a thousand examples — and it produced a model that was competitive with, and in human preference tests sometimes preferred over, models trained on *tens of thousands* of examples plus reinforcement learning from human feedback.

![LIMA: a thousand curated examples outperform fifty thousand noisy ones](/imgs/blogs/instruction-tuning-data-4.webp)

The figure above is the whole argument. On the left, the "big and noisy" path: fifty thousand machine-generated pairs, whose training signal is verbose, templated, and shot through with the teacher's errors, producing a *lower* helpfulness win-rate. On the right, the "small and curated" path: a thousand hand-selected, diverse, expertly-written examples with a clean and format-consistent signal, producing a *higher* win-rate. Same base model, dramatically different data, and the small set wins.

The LIMA authors framed this as the **Superficial Alignment Hypothesis**: a model's knowledge and capabilities are learned almost entirely during pretraining, and alignment (instruction tuning) is a *lightweight* process that teaches the model *which subdistribution of its existing knowledge to use* when interacting with a user. Under this hypothesis, you do not need many examples to teach a format and a style — you need *enough* examples, and they need to be *good* and *diverse*. Once the model has seen a thousand clean demonstrations spanning the range of things you want it to do, it has learned the format. Adding fifty thousand more mediocre ones just drags the average response toward mediocre.

Two design principles fall out of this that should govern how you build every SFT set:

1. **Diversity matters more than count.** A thousand examples covering a thousand *different* kinds of task teaches far more than a thousand near-duplicate paraphrases of the same task. This is why dedup (below) is not just about saving compute — near-duplicates actively reduce the effective diversity of a small set.
2. **Quality is a hard floor, not an average.** In a set this small, a handful of examples with wrong formatting or sycophantic answers will teach the model to reproduce them. You cannot outvote a bad pattern with good ones when the whole set is only a few thousand pairs.

The practical upshot: **spend your effort on curation, not collection.** It is almost always better to hand-review a thousand examples until they are excellent than to auto-generate a hundred thousand and hope the average is fine. LIMA is the empirical license to be ruthless about what you throw away.

A caveat, because this is a principal-engineer post and not a hype piece: LIMA's thousand examples were *extraordinarily* well chosen, drawn from high-quality forums (Stack Exchange, wikiHow) and hand-written by the authors, with deliberate attention to diversity and format consistency. "A thousand examples" is not a magic number — it is a demonstration that curation quality can substitute for volume. If your thousand examples are mediocre, you get a mediocre model. The result is *less is more when the less is excellent*, not *less is more, full stop*.

## Format and template hygiene

Before we talk about curation mechanics, we have to talk about *format*, because format is where the most avoidable SFT disasters happen. The model does not just learn the content of your responses — it learns the exact byte-level structure that wraps them, and if you are careless, it learns the wrapper *instead of* the task.

![Format bias: an over-fit model handles in-template inputs but breaks on off-format ones](/imgs/blogs/instruction-tuning-data-6.webp)

The graph above shows the failure mode. A model trained on exactly one chat template learns to associate the *presence of that template* with "produce an assistant response." Feed it an input that matches the template precisely and it parses correctly and answers well. Feed it an input with different delimiters, or no system message, or an extra newline, and it can fall off a cliff: leaking template tokens into the output, mis-parsing where the user turn ends, or refusing entirely because the input "looks wrong." This is *format bias*, and it is one of the top causes of "the model works in my eval harness but breaks in production."

There are four disciplines that prevent it.

### Use a real chat template, applied consistently

Every modern tokenizer ships a chat template — a Jinja specification of how a list of role-tagged messages is serialized into a single string of special tokens. Use it. Do not hand-concatenate strings.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

messages = [
    {"role": "system", "content": "You are a terse, correct coding assistant."},
    {"role": "user", "content": "Reverse a singly linked list in Python."},
    {"role": "assistant", "content": "def reverse(head):\n    prev = None\n    while head:\n        head.next, prev, head = prev, head, head.next\n    return prev"},
]

# What the model sees at INFERENCE (prompt only, ready to generate):
prompt = tok.apply_chat_template(
    messages[:-1], tokenize=False, add_generation_prompt=True
)
# What the model sees during TRAINING (the full conversation):
full = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

The single most common formatting bug is a mismatch between the string you train on and the string `apply_chat_template` produces at inference — a missing `add_generation_prompt`, a stray BOS token added twice, a system prompt present in training but absent at serving. The model learns the *training* format, and any deviation at inference degrades it. If you take one habit from this section: **render your training strings with the same `apply_chat_template` call your inference stack uses, and diff them.** For a deeper catalogue of these bugs, see [chat template and formatting bugs](/blog/machine-learning/debugging-training/chat-template-and-formatting-bugs).

### Compute loss only on the response

This is non-negotiable and easy to get wrong. During SFT you want the model to learn *to produce responses*, not to *predict the user's questions*. If you compute the loss over the entire sequence — instruction included — you waste capacity teaching the model to generate instructions, and you dilute the response signal. The objective you actually want is

$$\mathcal{L} = -\sum_{t \,\in\, \text{response}} \log p_\theta(y_t \mid y_{<t}, x)$$

where $x$ is the rendered prompt and the sum runs only over response tokens. In practice you implement this by building a label tensor that masks the prompt span with the ignore-index (`-100` in PyTorch cross-entropy):

```python
import torch

def build_labels(tok, messages, ignore_index=-100):
    full = tok.apply_chat_template(messages, add_generation_prompt=False)
    prefix = tok.apply_chat_template(messages[:-1], add_generation_prompt=True)
    labels = list(full)
    for i in range(len(prefix)):        # mask the entire prompt span
        labels[i] = ignore_index
    return torch.tensor(full), torch.tensor(labels)

input_ids, labels = build_labels(tok, messages)
# loss = model(input_ids=input_ids, labels=labels).loss  -> only response counts
```

Most training frameworks (TRL's `SFTTrainer`, Axolotl) can do this for you with a `train_on_responses_only` or completion-only collator, but you must *verify it is on*. A model trained with loss on the whole sequence is a classic "why is it so bad at following instructions" bug — it spent half its gradient learning to autocomplete prompts.

### Keep formatting consistent, and vary the system prompt deliberately

Two competing pressures. On one hand, *within* a training set, the response format should be consistent: if half your code answers use fenced blocks and half do not, the model learns an incoherent policy. On the other hand, if *every* example shares the identical system prompt and identical structure, you get the format bias from the figure above — the model welds itself to that exact wrapper. The resolution is to be consistent about *what a good answer looks like* while varying the *scaffolding*: mix in examples with and without a system prompt, with different (but valid) system prompts, and with slightly different phrasings of the same request, so the model learns the *task* is invariant to the wrapper.

### Pack sequences without cross-contaminating attention

SFT examples vary wildly in length, and naive padding wastes enormous compute. The standard fix is sequence packing — concatenating multiple short examples into one long sequence — but done naively it lets tokens from example A attend to tokens from example B, teaching the model spurious cross-example dependencies. The fix is a block-diagonal attention mask (or FlashAttention's variable-length API) so each packed example only attends to itself. This is subtle enough to deserve its own treatment; see [sequence packing for LLM fine-tuning](/blog/machine-learning/training-techniques/sequence-packing-llm-fine-tuning) for the mechanics and the gotchas.

## Deduplication and decontamination for SFT

Now the two curation steps that quietly determine whether your eval numbers mean anything: removing duplicates, and removing leaked benchmark questions.

**Deduplication** matters more for SFT than for pretraining, and for a non-obvious reason. In a giant pretraining corpus, a duplicated document is a rounding error. In a small SFT set, a duplicated instruction is *over-weighting a specific behavior* — the model sees "summarize this article" phrased fifty near-identical ways and learns to produce that one style with high confidence, at the expense of the diversity that LIMA showed is the whole point. Near-duplicates are especially insidious in machine-generated data, where a distillation loop happily produces a thousand paraphrases of the same underlying task. The dedup machinery is the same MinHash-plus-LSH and semantic clustering described in [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale); here you point it at the *instruction* (and optionally the instruction-plus-input) rather than the full document, because two examples with the same instruction and different responses are usually the redundancy you want to collapse.

**Decontamination** is the one people skip and then regret. Instruction datasets leak benchmarks *badly*, often worse than pretraining corpora, because they are frequently curated from the same public sources your evals come from. FLAN is built from academic datasets; if one of those datasets overlaps with your eval, you have trained on the test set. Self-Instruct pipelines happily regenerate famous benchmark questions because the teacher has them memorized. The result is an SFT model that scores wonderfully on MMLU or GSM8K and disappoints in production, because the benchmark measured memorization, not skill. The detection technique is n-gram overlap against your eval sets (plus canary strings where available), exactly as in [decontamination and benchmark leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage) — but you must remember to run it on your *instruction data*, not just your pretraining data. Most teams decontaminate the corpus and forget the fine-tuning set entirely.

> If you did not decontaminate your instruction data against your eval sets, your eval scores are marketing, not measurement. This is the single most common way teams lie to themselves about a fine-tune.

The order matters, too. Dedup first (it shrinks the set and removes the trivial cases), then decontaminate (n-gram matching is expensive; run it on the smaller set), then balance and quality-curate. We will see the exact counts in the worked scenario.

## Mixing skills: balancing capabilities

Here is a failure that surprises people: you assemble a big, diverse, deduplicated, decontaminated SFT set, fine-tune, and the model gets *worse* at some things it used to do. You dig in and discover that 55% of your examples are code, because code data is abundant and easy to collect, and the model has quietly become a code model that has forgotten how to write a friendly email. This is **capability regression from an unbalanced mix**, and it is entirely a data-composition problem.

<figure class="blog-anim">
<svg viewBox="0 0 660 330" role="img" aria-label="Category mix rebalancing: a code-heavy SFT set where code is 55 percent of examples, then a balanced set capped near 20 percent per category" style="width:100%;height:auto;max-width:820px">
<title>Rebalancing an SFT category mix from code-dominated to balanced</title>
<style>
.a6-axis{stroke:var(--border,#d1d5db);stroke-width:2}
.a6-name{font:600 15px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.a6-cap{font:700 17px ui-sans-serif,system-ui;text-anchor:middle}
.a6-bar{fill:var(--border,#d1d5db)}
.a6-hot{fill:#e5484d}
.a6-good{fill:var(--accent,#6366f1)}
.a6-capA{fill:#e5484d}
.a6-capB{fill:var(--accent,#6366f1)}
@keyframes a6-fadeA{0%,42%{opacity:1}55%,95%{opacity:0}100%{opacity:1}}
@keyframes a6-fadeB{0%,42%{opacity:0}55%,95%{opacity:1}100%{opacity:0}}
.a6-A{animation:a6-fadeA 9s ease-in-out infinite}
.a6-B{animation:a6-fadeB 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.a6-A{animation:none;opacity:1}.a6-B{animation:none;opacity:0}}
</style>
<line class="a6-axis" x1="40" y1="250" x2="620" y2="250"/>
<text class="a6-name" x="80"  y="278">code</text>
<text class="a6-name" x="190" y="278">math</text>
<text class="a6-name" x="300" y="278">chat</text>
<text class="a6-name" x="410" y="278">extract</text>
<text class="a6-name" x="520" y="278">summ</text>
<g class="a6-A">
<rect class="a6-hot" x="48"  y="70"  width="64" height="180"/>
<rect class="a6-bar" x="158" y="206" width="64" height="44"/>
<rect class="a6-bar" x="268" y="186" width="64" height="64"/>
<rect class="a6-bar" x="378" y="220" width="64" height="30"/>
<rect class="a6-bar" x="488" y="210" width="64" height="40"/>
<text class="a6-cap a6-capA" x="330" y="40">before: code is 55% of the mix</text>
</g>
<g class="a6-B">
<rect class="a6-good" x="48"  y="154" width="64" height="96"/>
<rect class="a6-good" x="158" y="162" width="64" height="88"/>
<rect class="a6-good" x="268" y="158" width="64" height="92"/>
<rect class="a6-good" x="378" y="166" width="64" height="84"/>
<rect class="a6-good" x="488" y="160" width="64" height="90"/>
<text class="a6-cap a6-capB" x="330" y="40">after: capped near 20% per category</text>
</g>
</svg>
<figcaption>The same five skill categories, before and after balancing: a code-dominated mix flattens to a roughly even distribution so no single skill drowns out the rest.</figcaption>
</figure>

The animation above shows the intervention. On the left, the raw mix — code towering over everything because it was the cheapest to collect. On the right, the same categories capped near a target share so the gradient the model sees is spread across all the skills you care about. The mechanism is simple: gradient descent optimizes the *average* loss, so whichever category has the most examples gets the most gradient and dominates the learned policy. If you want a model that is equally good at five things, the data has to *represent* those five things in proportion to how much you value them — not in proportion to how easy they were to scrape.

Balancing is not the same as making everything equal. It is about setting a *target distribution* that reflects your product priorities, then sampling toward it. A coding assistant should be code-heavy on purpose; a general assistant should not be code-heavy by accident. The distinction is *intentional* composition versus *incidental* composition. The tool for the job is a per-category cap (or a weighted sampler), and we will write it in the next section.

Two refinements that separate a decent mix from a good one. First, balance at the level of *skill*, not just surface category: "reasoning" examples and "chat" examples can both be tagged "general" and still exercise entirely different capabilities. Tag by what the example *teaches*. Second, keep a small held-out slice of every category for evaluation, so you can *see* the regression when it happens instead of discovering it from a user complaint. If your held-out email-writing score drops after adding a pile of code data, the mix is off.

## Worked scenario: assembling an SFT set

Let me make this concrete with numbers. Suppose you are building a general-purpose assistant and you have collected a raw pool of 180,000 candidate instruction-response pairs from a blend of sources: some human-written (Dolly, OASST), some FLAN-templated, some distilled from a strong teacher, some scraped from public sharing sites. Here is the funnel from raw pool to shippable set, with the count at each stage.

![The SFT curation funnel: from a 180k raw pool to a 12k shipped set](/imgs/blogs/instruction-tuning-data-5.webp)

The timeline above tracks the survivors through four stages. Let me narrate each with the reasoning behind the cut.

**Stage 0 — Raw pool: 180,000 pairs.** This is everything you collected, warts and all. It includes exact duplicates (the same Stack Exchange answer scraped twice), near-duplicates (the distillation loop's fifty paraphrases of "write a haiku"), leaked eval questions, and a wildly skewed category distribution. It is not shippable, and training on it directly would give you a verbose, code-biased model that scores suspiciously high on your benchmarks.

**Stage 1 — Deduplicate: 180,000 → 142,000.** Run exact-match dedup first (hash the normalized instruction-plus-input), which catches the scraped-twice cases cheaply. Then MinHash-plus-LSH near-dedup at a Jaccard threshold around 0.8 on the instruction text, which collapses the paraphrase clusters. We lose 38,000 pairs — about 21% — and almost all of that loss is redundancy that was hurting diversity. The MinHash collision probability for a candidate pair with Jaccard similarity $s$, using $b$ bands of $r$ rows, is $1 - (1 - s^{r})^{b}$, which lets you tune the bands to catch the 0.8-and-above pairs while ignoring merely-similar ones.

**Stage 2 — Decontaminate: 142,000 → 138,000.** Build the set of 13-gram shingles from every question in your eval suite — say MMLU and GSM8K — and drop any training example whose instruction shares a 13-gram with an eval question. We lose only 4,000 pairs here, which *sounds* small, but those 4,000 were the ones that would have inflated your eval scores by several points and lied to you about the model's real ability. The absolute count is small; the epistemic value is enormous.

**Stage 3 — Balance categories: 138,000 → 40,000.** The 138k set is 55% code, 20% chat, and a long tail of everything else. You decide the target mix for a general assistant is roughly 20% each across code, reasoning/math, chat, extraction/QA, and summarization/writing. To hit that with the least-represented category as the binding constraint, you cap each category and end up around 40,000 pairs. This is the biggest cut, and it is the one people flinch at — "we're throwing away 98,000 good examples!" — but they were not good *for the mix*; they were making the model lopsided.

**Stage 4 — Quality curation: 40,000 → 12,000.** The LIMA step. Score every remaining pair for quality — with a reward model, an LLM-as-judge rubric, or heuristics like response length sanity, format consistency, and refusal-appropriateness — and keep the top slice, subject to preserving the category balance from Stage 3. You ship 12,000 pairs. This is where you would rather have a strong preference-labeling process, which is the subject of the next post, [preference data for alignment](/blog/machine-learning/training-data/preference-data-for-alignment).

The headline: **180,000 candidates became 12,000 shipped pairs — a 93% cut — and the 12,000-pair model beats the 180,000-pair model on every axis you care about.** That is the LIMA lesson operationalized. The funnel is not throwing away value; it is *concentrating* it.

## The curation code

Here is the curation pipeline from the worked scenario, in runnable form. It uses `datasketch` for MinHash-LSH, plain n-gram sets for decontamination, and a per-category cap for balancing. This is deliberately close to what you would actually run — the only thing elided is the quality-scoring model in the final step, which is a call out to a reward model or judge.

```python
from datasketch import MinHash, MinHashLSH
from collections import defaultdict
import re, random

TOKEN = re.compile(r"\w+")

def tokens(text):
    return TOKEN.findall(text.lower())

def ngrams(text, n=13):
    toks = tokens(text)
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}

def minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for t in tokens(text):
        m.update(t.encode("utf-8"))
    return m

# --- Stage 1: near-duplicate removal on the instruction (+ optional input) ---
def dedup(examples, threshold=0.8, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []
    for i, ex in enumerate(examples):
        key = (ex["instruction"] + " " + ex.get("input", "")).strip()
        m = minhash(key, num_perm)
        if lsh.query(m):                 # a near-duplicate is already kept
            continue
        lsh.insert(str(i), m)
        kept.append(ex)
    return kept

# --- Stage 2: decontaminate against eval questions via shared n-grams ---
def build_banned_ngrams(eval_questions, n=13):
    banned = set()
    for q in eval_questions:
        banned |= ngrams(q, n)
    return banned

def decontaminate(examples, banned, n=13):
    clean = []
    for ex in examples:
        probe = ex["instruction"] + " " + ex.get("input", "")
        if ngrams(probe, n) & banned:    # overlaps an eval question -> drop
            continue
        clean.append(ex)
    return clean

# --- Stage 3: cap each category at a target share of the final budget ---
def balance(examples, target_share, budget):
    by_cat = defaultdict(list)
    for ex in examples:
        by_cat[ex["category"]].append(ex)
    out = []
    for cat, items in by_cat.items():
        cap = int(target_share.get(cat, 0.05) * budget)
        random.shuffle(items)
        out.extend(items[:cap])
    random.shuffle(out)
    return out

# --- Stage 4: quality curation (LIMA step); score_fn is your judge/reward model ---
def quality_curate(examples, score_fn, keep_frac=0.3):
    scored = sorted(examples, key=score_fn, reverse=True)
    return scored[: int(len(scored) * keep_frac)]

# --- run the funnel ---
pool = load_raw_pool()                                   # 180_000
deduped = dedup(pool, threshold=0.8)                     # ~142_000
banned = build_banned_ngrams(load_eval_questions())      # MMLU + GSM8K shingles
clean = decontaminate(deduped, banned)                   # ~138_000
target = {"code": 0.20, "math": 0.20, "chat": 0.20,
          "extraction": 0.20, "summarization": 0.20}
balanced = balance(clean, target, budget=40_000)         # ~40_000
final = quality_curate(balanced, score_fn=my_judge, keep_frac=0.3)  # ~12_000
save_sft_set(final)
```

A few implementation notes that matter in practice. The 13-gram choice for decontamination is deliberate — long enough that legitimate short instructions ("Summarize this:") do not collide, short enough to catch a leaked question even if a few words were changed. The `keep_frac=0.3` in quality curation is a knob: tighten it if your judge is trustworthy, loosen it if you are worried about throwing away good examples the judge under-rates. And note the *order* — dedup before decontaminate before balance before quality — which minimizes expensive operations on data you are going to drop anyway.

For the chat-template formatting that turns these curated pairs into training sequences, the `apply_chat_template` and loss-masking code from the format-hygiene section is exactly what you feed the deduplicated, decontaminated, balanced set into. Curation produces the pairs; template hygiene turns them into gradients on the right tokens.

## Case studies from production

### 1. LIMA — the thousand-example proof

The canonical case, worth restating as an engineering lesson rather than a headline. LIMA's team fine-tuned LLaMA-65B on 1,000 examples: 750 hand-selected from community forums (Stack Exchange, wikiHow, Reddit's writing communities) chosen for quality and diversity, plus 250 written by the authors themselves with deliberate stylistic consistency. No RLHF. The result held its own against RLHF-tuned models in human evaluation. The engineering lesson is not "use a thousand examples" — it is that *the authors spent their effort on the examples, not the algorithm.* They treated each of the thousand as a hand-crafted artifact. The failure mode LIMA warns against is the reflex to reach for more data or a fancier training objective when the real lever is sitting in the quality of a small set you could review by hand in a week.

### 2. Alpaca — cheap distillation and its ceiling

Stanford Alpaca fine-tuned LLaMA-7B on 52,000 pairs generated by `text-davinci-003` for roughly \$500 of API cost, and it *felt* like a chat model — which is exactly why it became a cautionary tale. Alpaca inherited every limitation of its teacher: it hallucinated confidently, it was verbose, and its "knowledge" was bounded by what the teacher would say. Teams that cloned the recipe and shipped it discovered that a model which demos well can still be systematically unreliable, because the distillation copied the teacher's confident errors verbatim. The lesson: distillation is a fantastic *bootstrap* and a dangerous *destination*. Use it to get to a working baseline cheaply, then replace the machine-generated bulk with curated, verified data — do not ship the raw distillate.

### 3. Tulu and FLAN — the mixture beats the monolith

The Allen Institute's Tulu work is the best public study of *how to mix*. Their systematic sweep found that a model trained only on FLAN was excellent at benchmark-style tasks and poor at open conversation; a model trained only on distilled GPT data was conversational but weak at reasoning; and the strongest all-around model came from a *deliberate blend* of FLAN (task breadth, correct labels), chain-of-thought and code (reasoning), and a measured amount of human and distilled conversational data (chat polish). FLAN's own lesson reinforces this: templating hundreds of academic datasets into instructions bought enormous task coverage with real gold labels, but FLAN-only models are stiff. The combined lesson across both: **no single source is complete, and the mixture is a design decision you should make on purpose, measured against a per-category eval.**

### 4. Dolly — the licensing-clean human set

Databricks Dolly (`databricks-dolly-15k`) is instructive for a different reason: it was built to be *legally unencumbered*. Because it was written entirely by employees rather than distilled from a commercial model, it could be released for any use, including commercial fine-tuning, without the "do not train competitors" cloud. It is small (15k) and its quality is uneven — employee-written answers vary — but it demonstrated that a modest, license-clean, human-written set is enough to give a base model real instruction-following behavior. The lesson for teams with commercial constraints: provenance is a first-class attribute of SFT data, and "we distilled it from a frontier model" can be a legal landmine, not just a quality caveat.

### 5. OpenAssistant — SFT and preference data from one effort

OASST1's design is worth copying: by having volunteers write prompts, write replies, *and rank* replies in a conversation tree, the project harvested supervised demonstrations and preference comparisons from the same annotation effort. The lesson is operational — if you are paying humans to produce SFT data, design the task so it also yields the ranking signal you will need for the preference-tuning stage. Do not run two separate, expensive annotation campaigns when one well-designed one gives you both. This directly sets up [preference data for alignment](/blog/machine-learning/training-data/preference-data-for-alignment).

### 6. The Vicuna / ShareGPT shortcut and its evaluation trap

Vicuna fine-tuned on roughly 70,000 conversations scraped from ShareGPT (users sharing their ChatGPT transcripts) and reported near-parity with the teacher on an *LLM-as-judge* evaluation. Two lessons collided here. The positive one: real user conversations are a rich, diverse source of instructions that no synthetic pipeline reproduces — actual humans ask messy, multi-turn, context-dependent things. The cautionary one: the evaluation was itself run by a strong LLM judge that *favored the verbose, confident style of the teacher the data was distilled from*, inflating the apparent quality. The lesson: when your data is distilled from a model and your eval is *judged* by a similar model, you have a closed loop that rewards mimicry over correctness. Break the loop with human eval or a judge from a different model family.

### 7. The category-collapse regression

A pattern I have watched play out on more than one team: an assistant is doing fine, someone adds a large, freshly-scraped code-instruction dataset to "improve coding," and the next model is a better coder and a *worse* everything-else. Support-email drafting gets terse and robotic; summaries start reading like commit messages. The root cause is always the same — the new data was large enough to tip the category mix, and gradient descent followed the majority. The fix is always the same too: cap the new category at its target share, rebalance, and retrain. The detection that saves you is a *per-category held-out eval* run on every candidate model, so the regression shows up as a number before it shows up as a complaint.

## Troubleshooting

The compressed field guide. Each row is a symptom you will actually observe, the detection that pins down the cause, and the fix. Print it and tape it next to your fine-tuning runs.

| Symptom | Likely cause | Detection | Fix |
| --- | --- | --- | --- |
| Model refuses or garbles off-format inputs; leaks template tokens | Format bias — over-fit to one chat template | Probe with inputs that vary delimiters / drop the system prompt; watch for special tokens in output | Vary system prompts and scaffolding in training; render with the real `apply_chat_template`; add format-robustness examples |
| Suspiciously high MMLU/GSM8K, weak in production | Eval contamination in the instruction set | n-gram (13-gram) overlap of instructions against eval questions; canary-string search | Decontaminate the *SFT set* (not just pretraining); re-run evals; distrust the old numbers |
| Model got worse at a skill after adding data | Capability regression from unbalanced mix | Per-category held-out eval before/after; inspect category histogram | Cap the dominant category at target share; rebalance and retrain |
| Verbose, hedging, sycophantic answers | Weak-teacher noise from distillation | Compare response-length distribution and hedging-phrase rate vs. a human reference set | Replace machine-generated bulk with curated/verified data; quality-curate harder (lower `keep_frac`) |
| Confident, consistent factual errors | Laundered teacher errors baked in | Spot-check responses against ground truth on a factual slice | Add ground-truth-labeled data (FLAN-style); drop unverifiable distilled answers on factual tasks |
| Model learns to autocomplete prompts, poor instruction-following | Loss computed on prompt tokens, not response-only | Inspect the label tensor; check the collator | Turn on completion-only / response-only masking; verify the prompt span is `-100` |
| Low diversity, repetitive phrasings | Near-duplicate instructions over-weighting a pattern | MinHash-LSH cluster-size histogram on instructions | Near-dedup at Jaccard ~0.8 on the instruction; keep one exemplar per cluster |
| Great on English, breaks on other languages | Monolingual SFT set over a multilingual base | Per-language eval slice; language-ID histogram of the set | Add multilingual instruction data (OASST is multilingual); balance by language |

Let me expand the four the spec calls out, because the one-liner does not do them justice.

**Format bias.** The symptom is a model that is brilliant in your eval harness — which always uses the exact training template — and brittle the moment a real request arrives with a slightly different shape. The reason it hides is that your eval reproduces your training format perfectly, so the bias is invisible until production. Detection is a deliberate *format-perturbation probe*: take a handful of prompts, render them with a different-but-valid delimiter set, drop the system message, add an extra newline, and see whether quality collapses or template tokens leak into the output. The fix is diversity of scaffolding at training time — vary the system prompt, include some examples with none, and if you serve multiple front-ends, include their formats in training. Do *not* try to fix this in the eval; the eval is telling the truth about a narrow slice.

**Eval gaming via contaminated instructions.** The symptom is a benchmark score that outruns the model's real ability. It is not usually deliberate cheating — it is a curation pipeline that pulled instructions from the same public tasks the benchmark uses, or a distillation teacher that memorized the benchmark and regenerated it. Detection is n-gram overlap between your instruction set and every eval you report, plus canary strings if your evals embed them. The fix is to decontaminate the instruction set explicitly and re-run the evals, and — culturally — to treat any un-decontaminated score as unpublishable. The uncomfortable corollary is that some of your historical numbers were inflated; better to find that yourself than to have a customer find it.

**Capability regression from a narrow mix.** The symptom is that a change intended to improve one skill silently degraded another. The cause is that the SFT loss optimizes the average, so the biggest category wins the gradient tug-of-war. Detection *must* be proactive: a per-category held-out set, evaluated on every candidate model, so a drop in email-writing quality shows up as a metric the moment you add a code dump. The fix is intentional composition — set a target distribution, cap categories to it, and rebalance — plus the discipline to *never* add a large single-source dataset without checking its effect on the whole eval panel.

**Response-quality noise from a weak teacher.** The symptom is a model that is fluent but subtly bad: verbose, hedging, over-apologetic, confidently wrong in the exact places the teacher was. The cause is distillation from a model whose flaws became your training targets, with no ground-truth check to catch them. Detection is comparative — measure your set's response-length distribution, hedging-phrase frequency, and factual accuracy on a labeled slice against a human-written reference, and watch for the tell-tale teacher signature. The fix is to stop treating distilled data as a quality source: use it for bootstrapping volume, then quality-curate aggressively (a lower `keep_frac`), replace factual-task answers with ground-truth-labeled data, and verify the responses you keep.

## When to reach for curation-first SFT, and when not to

Reach for the curation-heavy approach in this post when:

- **You are building a general-purpose or user-facing assistant** where behavior, tone, and format consistency matter as much as raw capability.
- **You have a strong base model** and you are steering it, not teaching it — the Superficial Alignment Hypothesis is on your side, and a small curated set will do.
- **You report benchmark numbers anyone will act on.** If a score influences a launch decision, decontamination is mandatory, not optional.
- **You have humans who can review examples.** A week of hand-reviewing a thousand pairs beats a month of generating a hundred thousand.
- **Your data blends multiple sources.** The moment you have more than one source, you have a mixture-design problem, and Tulu's lesson applies.

Skip or downweight it when:

- **You are teaching a genuinely new, narrow skill** the base model has never seen (a proprietary query language, a specialized format). Here you may legitimately need *more* examples, because you are closer to teaching than steering — though you should still dedup and decontaminate.
- **You are doing pure research on training dynamics** where you *want* a large, uncurated set as a controlled variable.
- **You are in the earliest prototyping stage** and just need *any* instruction-following behavior to test the rest of the stack — a raw Alpaca-style distillate is a fine throwaway baseline. Just do not ship it.
- **The downstream stage will fix it.** If you are going straight into heavy RLHF or preference optimization, some SFT noise gets corrected later — but "later" is more expensive than "now," so curate anyway when you can.

The meta-lesson, and the one to carry out of here: **instruction tuning is a curation discipline, not a collection discipline.** The base model already has the knowledge. Your job is to show it a small, clean, diverse, well-balanced set of demonstrations of the behavior you want, with the format handled carefully and the eval leakage stripped out. A thousand good examples really do beat a million noisy ones — not as a slogan, but as a measured, reproducible result you can operationalize with the funnel and the code above.

## Further reading

- [Synthetic data generation](/blog/machine-learning/training-data/synthetic-data-generation) — how to *make* instruction data when you cannot collect it, and how to keep synthetic data from collapsing toward the generator's defaults (the upstream of the Self-Instruct problem).
- [Preference data for alignment](/blog/machine-learning/training-data/preference-data-for-alignment) — the next stage after SFT: pairwise comparisons, reward models, and DPO-style data.
- [Decontamination and benchmark leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage) — the full n-gram-overlap and canary-string machinery, pointed at instruction data here.
- [Deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) — MinHash, LSH banding math, and semantic dedup, reused above for near-duplicate instructions.
- [Sequence packing for LLM fine-tuning](/blog/machine-learning/training-techniques/sequence-packing-llm-fine-tuning) — how to pack variable-length SFT examples without cross-contaminating attention.
- [Chat template and formatting bugs](/blog/machine-learning/debugging-training/chat-template-and-formatting-bugs) — the catalogue of `apply_chat_template` mismatches behind most format-bias incidents.
- LIMA: Zhou et al., "Less Is More for Alignment" (2023). Self-Instruct: Wang et al. (2022). FLAN: Wei et al. (2021) and the FLAN 2022 collection. Tulu: Wang et al., "How Far Can Camels Go?" (2023).
