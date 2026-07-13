---
title: "LLMs for Recommendation: The LLM4Rec Landscape"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A field guide to using large language models in a recommender, the five LLM4Rec roles, where world knowledge beats collaborative signal, and where a two-tower still wins on cost and latency."
tags:
  [
    "recommendation-systems",
    "recsys",
    "llm4rec",
    "large-language-models",
    "cold-start",
    "zero-shot",
    "machine-learning",
    "retrieval",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/llms-for-recommendation-llm4rec-1.png"
---

A product manager drops a launch into your queue on a Friday afternoon: ten thousand new SKUs, all of them with rich titles and descriptions, none of them with a single click. Your recommender, the one you have spent two quarters tuning, is an ID-based two-tower model. It learned a 256-dimensional vector for every item that has interaction history. For these ten thousand new SKUs it has nothing — their embeddings are still random initialization noise — so they are invisible to retrieval and will stay invisible until enough users stumble onto them to bootstrap a signal. That bootstrap can take weeks, and weeks is exactly what the merchandising team does not have.

Meanwhile, a different reality is sitting in your terminal. A large language model has, in its pretraining, read more about board games, cookware, and indie films than any one of your users ever will. It does not know that *user 88301* clicked item *4471 then item 9982*, but it does know that someone who liked "a cooperative deck-building game about ecological collapse" might enjoy "a worker-placement game about terraforming Mars." It can reason about that from text alone, with zero interaction data, on the first day a product exists. That is a capability your collaborative-filtering stack structurally cannot have, because collaborative filtering only knows the IDs people co-clicked, and a brand-new ID has been co-clicked with nothing.

This is the promise and the trap of using LLMs for recommendation, the topic of this post. The promise is real: world knowledge, reasoning over text, cold-start from descriptions, natural-language interfaces, and zero/few-shot transfer to brand-new domains — none of which an ID-based model gives you. The trap is equally real: LLMs are slow, expensive, do not natively know your catalog or your users' dense behavior, carry their own position and popularity biases, and will cheerfully recommend an item that does not exist in your inventory. The whole game of LLM4Rec is figuring out *which role* the LLM should play so you capture the promise without paying the trap. The figure below is the map we will follow — the five roles an LLM can take in a recommender.

![Taxonomy tree showing five roles an LLM can play in a recommender split into frozen prompted use and weight-touching tuned use](/imgs/blogs/llms-for-recommendation-llm4rec-1.png)

By the end of this post you will be able to: place any LLM4Rec idea into a five-role taxonomy; write a runnable zero/few-shot recommender that grounds its output to your catalog; build the hybrid where an LLM embeds item text and a fast model does the serving; reason quantitatively about the token-and-latency cost of LLM ranking versus a two-tower's sub-millisecond retrieval; and decide, honestly, when an LLM earns its place in your stack and when it does not. We will keep the series' spine in view throughout: the **retrieval → ranking → re-ranking funnel** fed by the **serve → log → train → serve feedback loop**, read off the **offline ↔ online reality gap**. The LLM does not replace that funnel. It slots into specific stages of it.

A quick orientation on terms, since this post sits at the seam of two fields. *Collaborative filtering* (CF) means predicting what you will like from the behavior of users who behaved like you — it learns from co-occurrence in interaction logs, not from item content. An *ID-based* model represents each user and item by a learned vector keyed on an integer ID. *Cold start* is the problem of recommending an item or to a user with little or no interaction history. *NDCG@K* (normalized discounted cumulative gain at K) is a ranking metric that rewards putting relevant items near the top of a K-length list; *HitRate@K* (or Recall@K) is the simpler fraction of times the held-out item appears in the top K. We will lean on these throughout, defining anything new as it shows up.

## 1. Why an LLM at all: what world knowledge buys you

Start with the honest question a staff engineer will ask in design review: *we already have a two-tower retriever and a DCN ranker that hit our targets — why would we add a model that is a thousand times slower and a thousand times more expensive per inference?* If you cannot answer that crisply, you should not add the LLM. So here is the crisp answer, in four capabilities that ID-based CF structurally lacks.

**World knowledge.** An LLM has read the internet. It knows that *The Lord of the Rings* and *The Hobbit* are by the same author, that a "mirrorless full-frame camera" and a "DSLR" are substitutes, that someone shopping for hiking boots in October is probably planning a fall trip. Your CF model knows none of this a priori; it can only *learn* the camera/DSLR substitution if enough users actually co-purchased them, and it learns nothing about an item nobody has touched. The LLM brings a prior over the entire world of products and content, for free, before a single click is logged.

**Reasoning and explainability.** A CF model produces a score; it cannot tell you *why*. An LLM can produce a score *and* a sentence of justification — "recommended because you watched three documentaries about deep-sea ecosystems." That justification is not just a UI nicety; it is a debugging and trust surface, and in regulated domains it can be a compliance requirement. The reasoning also lets the model handle compositional requests ("something like X but cheaper and for a beginner") that no fixed-feature ranker can parse.

**Cold start through text.** This is the headline. An item is a piece of text — a title, a description, a set of attributes. A user, in the cold-start case, is a short profile or a single stated intent. An LLM maps both into the same space its pretraining built, so a brand-new item is *immediately* recommendable from its description, and a brand-new user is recommendable from a one-line stated preference. No interaction history required.

**Zero/few-shot transfer.** Stand up a recommender in a brand-new vertical — say you just launched a marketplace for vintage synthesizers — and you have no model, no embeddings, no logs. A frozen LLM gives you a *day-zero* baseline by prompting alone. It will not be as good as a model trained on six months of your logs, but six months from now is not the question; the question is what you ship next week.

**Natural-language interface.** There is a fifth capability that does not show up in offline metrics but reshapes the product: the user can *talk* to the recommender. "Show me something like my last watch but lighter" is a query no keyword search or fixed filter UI can express, and no ID-based model can consume — it is compositional, comparative, and references a moving target (the user's own history). An LLM parses it natively. This unlocks *preference elicitation*: instead of inferring taste only from passive clicks, the system can *ask* — "are you in the mood for something familiar or something new?" — and incorporate the stated answer immediately, without waiting for it to show up in behavioral logs. For users whose taste is hard to infer from sparse history (the cold-start user again, from a different angle), being able to state a preference in words is a strictly more direct signal than a few ambiguous clicks. The interface is not a gimmick; for some product surfaces it is the whole reason to bring in an LLM.

Hold that list next to the costs, which are just as structural. A modern flagship like Claude Opus 4.8 prices around \$5 per million input tokens and \$25 per million output tokens; the cheaper Claude Haiku 4.5 is roughly \$1/\$5 per million. A single recommendation prompt that includes a user's history and a few dozen candidates is easily several thousand tokens. Multiply by traffic and the bill is not a rounding error. Worse, the latency is hundreds of milliseconds to a couple of seconds per call, against a two-tower's sub-millisecond dot-product retrieval. And the LLM does not natively know your catalog — it will name items that do not exist — nor your users' dense behavioral patterns, which are exactly where CF shines. The matrix below lays the trade cleanly side by side.

![Comparison matrix of LLM as recommender versus ID based collaborative filtering across world knowledge cold start dense behavior cost and latency](/imgs/blogs/llms-for-recommendation-llm4rec-2.png)

The pattern in that matrix is the whole post in miniature. The two paradigms are not competitors so much as complements with mirror-image strengths: the LLM owns the top two rows (knowledge, cold start), CF owns the bottom three (dense behavior, cost, latency). Every good LLM4Rec design is an attempt to *use the LLM only where it wins* and let the cheap fast model carry everything else. Keep that lens as we walk the taxonomy.

## 2. A taxonomy of LLM4Rec: five roles

The literature on LLMs-for-recommendation exploded after 2022, and the surveys (Wu et al., "A Survey on Large Language Models for Recommendation," 2023; Lin et al., "How Can Recommender Systems Benefit from Large Language Models," 2023) converge on a useful split. The cleanest organizing question is: **is the LLM used as-is, or are its weights updated?** Everything else hangs off that.

**Role 1 — LLM as the recommender (zero/few-shot prompting).** You feed the LLM the user's history and ask it directly for recommendations. No training, no catalog embeddings — just a prompt. "Here are the last 20 movies this user watched; from this candidate list of 20, rank the ones they will watch next." The LLM is the whole recommendation engine. This is the fastest thing to stand up and the easiest to reason about, and it is where most teams start. It is also where the failure modes are loudest, which is why we will spend a whole section on it.

**Role 2 — LLM as feature/representation generator.** Here the LLM never serves online. Offline, you use it to turn each item's text into an embedding (or to summarize/augment item and user descriptions), and those vectors become *features* inside a traditional model. The LLM is a fancy text encoder bolted onto a CF/ranking pipeline. This is, in practice, the most *deployed* role, because it captures the LLM's content understanding while keeping serving fast and cheap.

**Role 3 — LLM as a ranker/re-ranker.** The funnel already has a candidate generator (retrieval). You hand the LLM the small candidate set — tens, maybe a couple hundred items — and ask it to score or reorder them. Because re-ranking only touches a short list, the cost is bounded, and the LLM's reasoning can fix subtle ordering mistakes a cheap ranker makes. This is the role that respects the funnel most naturally: cheap models do the heavy filtering, the LLM polishes the top.

**Role 4 — LLM tuned into a recommender.** You actually update weights — full fine-tuning, instruction-tuning, or parameter-efficient methods like LoRA — so the LLM internalizes your catalog and behavior patterns. P5 (Geng et al., 2022) is the canonical early example: cast *every* recommendation task as text-to-text and train one model on all of them. Generative recommendation (TIGER, P5, and successors) lives here too. This role is deep enough that the **next post in the series**, [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice), is dedicated to it; we will only sketch it here.

**Role 5 — conversational/agentic recommenders.** The LLM holds a multi-turn dialogue, asks clarifying questions, calls tools (your retrieval API, a price filter, a stock checker), and threads the results into a recommendation. This is the role that exploits the *natural-language interface* the most, and it is the subject of the sibling post on [generative and conversational recommendation](/blog/machine-learning/recommendation-systems/generative-and-conversational-recommendation).

That is the map in figure 1. Notice the top-level split: roles 1–3 leave the LLM *frozen* (you only prompt it), roles 4–5 touch weights or build an agentic loop around it. The frozen roles are cheaper to operate and faster to ship; the tuned roles can be far stronger but demand training infrastructure, labeled data, and ongoing maintenance. Most production stacks today live in roles 2 and 3 — feature generation and re-ranking — precisely because they bolt the LLM's strengths onto an existing funnel without betting the whole system on a slow, expensive, ungrounded model. We will spend the bulk of this post on roles 1, 2, and 3, with a science section that explains *why* the failure modes happen.

## 3. Zero/few-shot prompting as a recommender (role 1)

Let us make role 1 concrete, because it is both the most seductive and the most misunderstood. The task: given a user's interaction history and a candidate set, ask the LLM to rank the candidates by predicted relevance. The first design decision is the *prompt format*, and it matters more than people expect.

A workable template for a movie recommender looks like this. We give the model a clear instruction, the history as a numbered list of titles, the candidate set as a second numbered list, and an explicit output contract so the response is parseable.

```markdown
You are a movie recommendation engine. A user has watched these
movies, most recent last:
1. Arrival
2. Blade Runner 2049
3. Dune: Part One
4. Interstellar
...

From the candidate list below, rank ALL candidates from most to
least likely the user watches next. Output ONLY a JSON array of the
candidate numbers in ranked order, e.g. [3, 1, 5, ...].

Candidates:
1. Dune: Part Two
2. The Notebook
3. Foundation (series)
4. Tenet
5. La La Land
...
```

Three things in that template are doing real work. **"Most recent last"** gives the model the recency signal it would otherwise have to guess — we will see in the science section that LLMs are notoriously bad at perceiving order on their own. **"Rank ALL candidates"** plus a numbered candidate list constrains the output to a permutation of *known* IDs, which sidesteps a lot of the hallucination problem (more on grounding later). **The JSON contract** makes parsing deterministic instead of regex-on-prose roulette.

Here is a runnable implementation against an API-style chat call. It builds the prompt, calls the model, parses the ranking, and — crucially — grounds the result back to real catalog IDs so a bad parse can never leak a phantom item into the response.

```python
import json, re
import anthropic  # pip install anthropic

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

def build_prompt(history_titles, candidate_titles):
    hist = "\n".join(f"{i+1}. {t}" for i, t in enumerate(history_titles))
    cand = "\n".join(f"{i+1}. {t}" for i, t in enumerate(candidate_titles))
    return (
        "You are a movie recommendation engine. A user watched these "
        "movies, most recent last:\n" + hist +
        "\n\nFrom the candidate list below, rank ALL candidates from "
        "most to least likely the user watches next. Output ONLY a JSON "
        "array of candidate numbers, e.g. [3, 1, 5].\n\nCandidates:\n" + cand
    )

def llm_rank(history_titles, candidate_ids, candidate_titles):
    prompt = build_prompt(history_titles, candidate_titles)
    resp = client.messages.create(
        model="claude-haiku-4-5",       # cheap tier for a ranker
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    # robust parse: grab the first [...] block
    m = re.search(r"\[[\d,\s]+\]", text)
    order = json.loads(m.group(0)) if m else []
    # GROUND: map 1-based positions back to real catalog IDs,
    # drop anything out of range, append any candidate the model
    # forgot so we never return fewer items than asked.
    ranked, seen = [], set()
    for pos in order:
        idx = pos - 1
        if 0 <= idx < len(candidate_ids) and idx not in seen:
            ranked.append(candidate_ids[idx]); seen.add(idx)
    for idx, cid in enumerate(candidate_ids):
        if idx not in seen:
            ranked.append(cid)
    return ranked
```

Notice that we ask the model to output *positions*, not titles. This is a deliberate grounding trick from Hou et al. (2023): if the model returns "Dune: Part Two" we have to fuzzy-match that string back to a catalog ID, which is error-prone; if it returns the position `1` we have a clean integer index into a list we control. The model can still scramble or drop positions, so the grounding loop validates every index and back-fills anything missing. The model literally cannot return an item that is not in the candidate set, because we never let it emit a free-form title.

**Few-shot** is the same machinery with a twist: before the real query you prepend one or two solved examples — a history, a candidate set, and the correct ranking — so the model learns the *format and the kind of reasoning* you want from demonstration. In practice few-shot reliably beats zero-shot for ranking because it pins down the output format and nudges the model toward recency-aware, less-popularity-biased rankings. The cost is more input tokens per call.

### Where role 1 shines

- **Cold start.** No history? The candidate list plus a one-line stated preference ("I want something cerebral and slow-paced") is enough for a sensible ranking. CF gives you popularity; the LLM gives you reasoning.
- **Brand-new catalog/domain.** Day-zero baseline with zero training. The vintage-synth marketplace ships a recommender on launch day.
- **Explainability.** Ask for a one-sentence reason per item and you get a free justification surface.
- **Compositional intent.** "Like the last one but a comedy and under two hours" — the LLM parses it; a fixed ranker cannot.

### Where role 1 fails (be honest)

This is the part teams skip and then get paged for. A frozen LLM as the *whole* recommender underperforms a tuned two-tower on the metrics that pay the bills, for reasons that are not bugs but properties of how these models work:

- **Popularity bias.** The model recommends what it saw a lot of in pretraining, which skews toward globally popular items regardless of this user's taste. Hou et al. measured exactly this.
- **Position bias.** The order you list candidates in leaks into the ranking. List the right answer last and the model is *less* likely to pick it. This is a known LLM serial-position effect, not a recommendation-specific one.
- **Poor at large candidate sets.** Ask it to rank 500 candidates and quality collapses — the list overflows useful context, attention smears, and the model anchors on the first and last few items. Role 1 is really only viable as a *re-ranker over tens of candidates* (which is why role 3 exists).
- **Hallucination.** Without the position-output trick above, the model invents plausible titles that are not in your catalog. We will dedicate a section to grounding.
- **Cost and latency.** Already covered, but it bears repeating: this is the dominant practical objection.

The honest summary: as a *standalone* recommender on a warm catalog with dense behavioral signal, a frozen LLM loses to a well-tuned [two-tower retriever](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval) on NDCG and crushes it on cost. Its real value is in the regimes where CF is structurally blind — cold start, new domains, compositional intent — and as a re-ranker over a short, pre-filtered list.

## 4. The science: prompting as conditional generation, and why the biases exist

Let us make the *why* rigorous, because "the LLM has popularity bias" is a folk claim until you can derive it.

### The recommendation prompt is a conditional generative model

A causal language model defines a distribution over token sequences and factorizes it autoregressively. When we prompt it with a user's history and an instruction and read off an item, we are sampling from

$$
P(\text{item} \mid \text{history}, \text{instruction}) \;=\; \prod_{t=1}^{T} P\!\left(w_t \mid w_{\lt t},\, \text{history},\, \text{instruction}\right),
$$

where $w_1, \dots, w_T$ are the tokens that spell the item (its title, its position number, or its semantic ID). The recommendation is whichever item maximizes — or the model samples from — this conditional probability. That single equation explains a lot. The model is not optimizing your ranking metric; it is optimizing next-token likelihood under its pretraining distribution, *conditioned* on your prompt. Everything good (world knowledge) and everything bad (popularity, position bias) flows from that distribution being a pretraining artifact, not a behavior model fit to your logs.

### Why popularity bias is structural

Decompose the item probability with Bayes' rule. For an item with text $x$,

$$
P(x \mid \text{context}) \;\propto\; P(\text{context} \mid x)\, P(x).
$$

The prior $P(x)$ is the model's marginal belief about how likely the *string* $x$ is to appear — and popular items appear far more often in pretraining text (more reviews, more articles, more mentions). So $P(x)$ is systematically larger for popular items, which biases the posterior toward them *independent of the user's context*. This is not a tuning failure; it is the prior leaking through. A two-tower trained on your logs has no such global prior — its scores come from a dot product of vectors fit to your interaction data, so a niche item that this user's neighbors loved can score above a blockbuster. That is precisely the dense-behavior advantage CF holds.

### Why position bias exists

Causal attention plus learned positional encodings give transformers a serial-position profile: tokens early and late in the context get disproportionate attention weight, the middle is "lost." When candidates are listed in the prompt, an item's *list position* therefore influences its selection probability — a confound that has nothing to do with relevance. Hou et al. (2023) found that simply shuffling the candidate order changed the LLM's top pick, and that *bootstrapping* (ranking several shuffles and aggregating) measurably reduced the variance. The fix is an ensemble over permutations, which costs more calls — another reason role 1 is expensive.

### Why order-perception is weak

CF models that are sequential — [SASRec/BERT4Rec](/blog/machine-learning/recommendation-systems/self-attention-for-sequences-sasrec-bert4rec) — are *trained* to predict the next item from an ordered history, so order is in their objective. A frozen LLM was trained to predict next tokens in web text; "this user watched A then B then C" is just three titles in a row to it, and it has no learned notion that the *most recent* item should dominate. You have to tell it ("most recent last," or weight recent items in the prompt), and even then it under-uses recency. This is the empirical finding behind "LLMs struggle to perceive the order of historical interactions" in Hou et al.

### Why few-shot helps, in the same framework

The conditional-generation view also explains *why* a couple of in-context examples reliably lift ranking quality. A demonstration does two things to the conditional $P(\text{item} \mid \text{history, instruction, demos})$. First, it sharpens the *format* posterior — after seeing one example where the answer is a JSON array of positions, the probability mass on "produce a JSON array of positions" collapses onto exactly that, killing the parse-failure tail. Second, and more subtly, a demonstration where the *recent* item drives the correct answer nudges the model toward recency-weighting, partially correcting the order-perception weakness without any fine-tuning. You are, in effect, doing a tiny gradient-free adaptation of the conditional through the context window. The cost is more input tokens, and there are diminishing returns past two or three demonstrations because the format is already pinned and the extra examples mostly add tokens. The empirical sweet spot most teams find is one to three demonstrations — enough to fix format and recency, not so many that the prompt bloats.

### Popularity bias as an exposure problem you can correct

It is worth connecting the LLM's popularity bias to the same machinery the rest of the series uses for biased logs, because the cure is analogous. In a logged recommender, popular items are over-represented because they were shown more — a *missing-not-at-random* (MNAR) problem you correct with inverse-propensity weighting, dividing each example's contribution by its exposure probability. The LLM's popularity bias is the same shape but the "exposure" is *pretraining frequency*: popular item strings appeared more in the training corpus, so the prior $P(x)$ over-weights them. You cannot reweight pretraining after the fact, but you can fight the symptom at inference: (a) instruct the model explicitly that popularity is not relevance ("rank by fit to this user's taste, not by general fame"); (b) calibrate the output by *subtracting* a popularity baseline — if you have item popularity counts, demote items the model ranks high that are also globally popular, since some of that ranking is prior leakage not user fit; (c) bootstrap over candidate-order shuffles to wash out the position component. None of these is a clean fix the way a tuned model would be, but each measurably reduces the bias, and together they are the difference between an LLM that just re-surfaces blockbusters and one that respects the user's actual taste.

### The cost model, made quantitative

This is the part that decides architectures. A two-tower retrieval is, at serve time, a single user-embedding lookup plus an approximate-nearest-neighbor search over the item index — sub-millisecond, and effectively free per query (a few floating-point ops). Contrast the LLM. If you prompt with $H$ tokens of history, $C$ candidates at roughly $c$ tokens of text each, plus a fixed instruction of $I$ tokens, the input is

$$
N_{\text{in}} \;\approx\; I + H + C \cdot c,
$$

and the cost per request, at input price $p_{\text{in}}$ and output price $p_{\text{out}}$ per token with $N_{\text{out}}$ output tokens, is

$$
\text{cost} \;=\; N_{\text{in}}\, p_{\text{in}} + N_{\text{out}}\, p_{\text{out}}.
$$

Latency scales with $N_{\text{in}} + N_{\text{out}}$ through the model's prefill and decode, landing in the hundreds-of-milliseconds-to-seconds range. The two-tower's cost is dominated by the ANN index, which is microseconds-to-milliseconds and amortized across all queries. We will put real numbers on this in a worked example. The takeaway is the *shape*: LLM cost grows linearly in the number of candidates you ask it to consider, while ANN cost is roughly constant in catalog size (logarithmic, with a good index). That is why you never let an LLM see your whole catalog — you let cheap retrieval cut it to tens of items first.

## 5. The item representation problem: how do you name an item to an LLM?

Before any of this works at scale you have to answer a deceptively simple question: *how do you refer to an item when you talk to the LLM?* There are three answers, and they define three sub-fields.

**Title / text.** Refer to the item by its human-readable description — "Dune: Part Two (2024)." This is what makes cold start and reasoning possible: the model understands the *content*. The downsides are that titles can be ambiguous (multiple movies share a name), long titles burn tokens, and the model can hallucinate a title that does not exist in your catalog. Most role-1 and role-3 systems use text, with a grounding step.

**Numeric ID.** Refer to the item by its integer ID — "item 4471." This is compact and unambiguous, but to a *frozen* LLM it is meaningless noise; the model has no idea what item 4471 is. Numeric IDs only work for a *tuned* model (role 4) that has learned, during fine-tuning, what each ID's behavior pattern is — and even then, a raw integer carries no structure (item 4471 and 4472 are unrelated), so the model has to memorize each one, which is sample-inefficient.

**Semantic ID.** This is the clever middle path and the bridge to generative retrieval. Instead of a flat integer, you assign each item a short sequence of *codewords* derived from its content embedding — e.g. item → `(c_12, c_4, c_88)` — using a residual-quantizing autoencoder (RQ-VAE). Items with similar content get *overlapping* codeword prefixes, so the ID space has structure the model can generalize over: learn that prefix `(c_12, c_4, ...)` means "cerebral sci-fi" and you handle new items in that cluster for free. This is the representation behind TIGER (Rajput et al., 2023) and a thread we pick up in [autoencoders and the road to generative retrieval](/blog/machine-learning/recommendation-systems/autoencoders-and-the-road-to-generative-retrieval). The model *generates* the codewords of the recommended item token by token, which makes the whole recommender a sequence-to-sequence problem — covered in depth in the generative-recommendation post.

The choice cascades into everything. Text gives you cold start but needs grounding; numeric IDs need a tuned model and waste capacity; semantic IDs need an extra quantizer but give you a structured, generative-friendly item vocabulary. For the frozen roles in this post (1–3), text is the practical default, and grounding is mandatory.

There is a token-budget angle to this choice that bites at scale and is easy to miss. Referring to items by full title text is *expensive in context*: a candidate list of 50 items at 16 tokens each is 800 tokens of just *naming* the candidates, before any reasoning. Numeric IDs are compact (one or two tokens each) but meaningless to a frozen model. Semantic IDs sit in between — a handful of codeword tokens per item — and they carry *structure*, so the model can reason over the codeword prefixes rather than re-reading full descriptions. For a tuned generative model this is a real efficiency win: the item vocabulary is small and structured, decoding an item is a few tokens, and the trie-constrained decode is cheap. For the frozen roles, you are stuck with text (the model has no learned ID vocabulary), so the practical move is to *trim* the text — short titles plus one or two discriminative attributes, not full marketing blurbs — to keep the per-candidate token cost down. The representation choice is therefore not just an accuracy decision; it is a direct lever on the cost model from section 4, because tokens-per-item multiplies straight into the per-request bill. A common production compromise: use trimmed text for the frozen re-ranker (role 3) where cold start and reasoning matter, and migrate to semantic IDs if and when you move to a tuned generative system (role 4) where token efficiency and grounding-by-construction pay off.

## 6. Catalog grounding: the LLM does not know your inventory

A frozen LLM, asked to recommend a movie, will happily output "Blade Runner 2099" — a sequel that does not exist. It is doing exactly what it was trained to do (produce a plausible continuation), but "plausible" is not "in my catalog." If you serve that, the user clicks and lands on a 404. Grounding is the discipline of forcing every recommendation to be a real, in-stock catalog item, and there are three levels of it.

![Dataflow graph of the catalog grounding problem where a free LLM may hallucinate an item and constrained decoding or retrieval matching forces it back to a valid catalog ID](/imgs/blogs/llms-for-recommendation-llm4rec-7.png)

**Level 1 — output positions, not titles (the cheapest fix).** This is the trick from the code in section 3. You give the model a numbered candidate list and ask it to output the *numbers* in ranked order. The output space is now a permutation of indices into a list you control, so by construction every result maps to a real item. This works only when you already have a candidate set (i.e. role 3, re-ranking), but that is the common case in a funnel.

**Level 2 — fuzzy match the generated title to the catalog (RAG-style rescue).** When the model *does* emit free text (no candidate list given), embed the generated title and run an ANN search over your catalog's item-title embeddings, snapping the hallucination to its nearest real neighbor. "Blade Runner 2099" snaps to "Blade Runner 2049." This is retrieval-augmented grounding: the LLM proposes, a vector index disposes. It costs one embedding and one ANN lookup per generated item, which is cheap.

**Level 3 — constrained decoding (the airtight fix).** Build a trie (prefix tree) of the token sequences of every valid catalog item — titles or, better, semantic IDs. At each decode step, *mask the logits* so the model can only emit tokens that continue a valid path in the trie. The model literally cannot generate an out-of-catalog item; every completed sequence is a real ID. This is how generative retrieval systems guarantee validity, and libraries like `transformers` support it through `prefix_allowed_tokens_fn` in `generate`. The cost is the trie and a per-step mask; the payoff is zero hallucination by construction.

```python
# Level 3 sketch: constrained decoding over a catalog trie.
# catalog_trie maps a tuple of already-decoded token ids -> set of
# allowed next token ids (built once from your catalog item strings).
def prefix_allowed_tokens_fn(batch_id, input_ids):
    decoded = tuple(input_ids.tolist()[prompt_len:])  # only generated part
    allowed = catalog_trie.get(decoded)
    # if the path is complete or unknown, fall back to EOS only
    return list(allowed) if allowed else [tokenizer.eos_token_id]

out = model.generate(
    **inputs,
    max_new_tokens=16,
    num_beams=4,
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
)
```

In practice most teams use level 1 for re-ranking (it is free and airtight when you have candidates) and reserve level 3 for tuned generative systems. Level 2 is the pragmatic catch-all when you genuinely want the model to generate open-vocabulary suggestions, as in conversational rec. Whatever you pick, *grounding is not optional* — shipping a recommender that can return phantom items is how you lose user trust in one bad demo.

## 7. The hybrid: LLM as a feature generator (role 2), the one that actually deploys

If section 3 is the seductive role and section 6 is its tax, this is the role that quietly powers most real systems. The idea: the LLM never serves a live request. Offline, you run it once per item to turn the item's text into a dense embedding, cache those vectors, and feed them as *features* into the same fast traditional ranker you already have. The LLM's content understanding flows into your model; the LLM's cost and latency stay out of the serving path entirely.

![Stack diagram of the hybrid where an LLM encodes item text offline into vectors that become frozen features inside a fast traditional ranker](/imgs/blogs/llms-for-recommendation-llm4rec-5.png)

Here is the offline embedding step with `sentence-transformers`, which gives you a high-quality text encoder without an API bill. (For maximum quality you can swap in an LLM-based embedding endpoint; the interface is the same — text in, vector out.)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("all-mpnet-base-v2")  # 768-d, runs on CPU/GPU

def item_text(row):
    # concatenate the fields that describe the item's content
    return f"{row['title']}. {row['genres']}. {row['description']}"

texts = [item_text(r) for r in catalog]          # one string per item
item_emb = encoder.encode(
    texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True
)                                                 # shape: (n_items, 768)
np.save("item_text_emb.npy", item_emb.astype("float32"))
```

Those 768-dimensional vectors are now a *feature* you concatenate with your existing ID embeddings, count features, and context features inside a ranker. Critically, the text embedding is defined for *every* item — including brand-new ones — so the moment a SKU is created you can compute its content vector and the ranker can score it. That is the cold-start fix, achieved without ever calling an LLM at request time. Below, a minimal PyTorch ranking head that fuses the frozen text feature with a learned ID embedding.

```python
import torch, torch.nn as nn

class HybridRanker(nn.Module):
    def __init__(self, n_items, id_dim=64, text_dim=768, hidden=256):
        super().__init__()
        self.id_emb = nn.Embedding(n_items, id_dim)
        # project the frozen LLM/text feature down, then fuse
        self.text_proj = nn.Linear(text_dim, id_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(id_dim * 2 + 8, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, item_id, text_feat, ctx):     # ctx: (B, 8) context feats
        e_id = self.id_emb(item_id)                  # (B, id_dim)
        e_tx = self.text_proj(text_feat)             # (B, id_dim) from frozen vec
        x = torch.cat([e_id, e_tx, ctx], dim=-1)
        return self.mlp(x).squeeze(-1)               # CTR logit
```

The `text_feat` tensor is read straight from the cached `.npy` — the LLM's work is done offline and amortized across every request for the rest of the item's life. For a new item with no interaction history the `id_emb` is uninformative (random init), but `text_proj(text_feat)` carries real content signal, so the ranker can still place the item sensibly. As the item accumulates clicks, `id_emb` learns and takes over — a graceful handoff from content to collaborative signal. This is the practical synthesis of the [content-based and hybrid recommenders](/blog/machine-learning/recommendation-systems/content-based-and-hybrid-recommenders) idea, now with an LLM-grade text encoder.

Why is this the role that deploys? Because it pays the LLM cost *once per item, offline*, and gets sub-millisecond serving forever after. A catalog of one million items embedded once is a single batch job; re-embed only when an item's text changes. There is no per-request LLM call, no latency hit, no per-query token bill. You get the LLM's content understanding at CF's serving economics. The matrix in section 11 will show this hybrid winning both warm NDCG *and* cold-start hit-rate at a cost that rounds to zero — the best Pareto point in the whole post.

### The cold-start contrast, drawn out

The clearest way to see what the text feature buys you is to follow one brand-new item through both pipelines side by side. The figure below contrasts the two: the ID-only recommender on the left, the text-native pipeline on the right.

![Before and after contrast of an ID only recommender that fails cold start versus a text native recommender that handles a new item from its description](/imgs/blogs/llms-for-recommendation-llm4rec-4.png)

On the left, the new item has zero clicks. In an ID-only model its embedding is whatever the random initializer produced — a vector with no relationship to anything the model learned, because the only way an ID embedding acquires meaning is gradient signal from interactions, and there have been none. So the dot product between any user and this item is noise; the item is never retrieved, never shown, and therefore never accumulates the interactions it would need to escape the cold state. That is the self-reinforcing trap of cold start in a pure-ID system: no exposure without signal, no signal without exposure. The item can sit there for weeks.

On the right, the same item is a piece of text the moment it is created. The encoder reads its title and description, maps it to the same 768-dimensional space every other item lives in, and places it *next to its content neighbors* — the cerebral-sci-fi cluster, say. Now a user whose history sits in that cluster has a real, non-noise affinity to the new item from minute one, so it surfaces, gets shown, and starts collecting the very interactions that will later train its ID embedding. The text feature breaks the trap by giving the item a meaningful position before any behavioral signal exists. This is not a marginal improvement; it is the difference between "recommendable on day one" and "invisible until the cold-start problem solves itself," which it rarely does without help.

Two practical notes keep this honest. First, the text feature is only as good as the text — sparse, templated, or SEO-spam descriptions produce weak embeddings, so the cold-start lift is bounded by your catalog's content quality. Second, the handoff from content to collaborative signal needs to be *smooth*: if the ID embedding suddenly dominates the moment a few clicks arrive, you get a jarring shift in what the item is recommended next to. The clean fix is to let both features flow into the ranker together (as the `HybridRanker` does) and let the model learn the blend, rather than hard-switching between a "cold" and a "warm" code path.

## 8. LLM as a re-ranker (role 3): respecting the funnel

Role 1 fails on large candidate sets; role 3 is the disciplined version that fixes that by only ever showing the LLM a short, pre-filtered list. The funnel does the heavy lifting — cheap retrieval cuts a million items to a few hundred, a cheap ranker cuts those to twenty — and *only then* does the LLM see the twenty and reorder them. This is the architecturally honest place for an LLM in a high-traffic system, because the expensive model touches the fewest items.

![Dataflow graph of a zero-shot LLM recommendation flow where history and candidates merge into a prompt the LLM ranks and the parsed output is grounded to catalog IDs](/imgs/blogs/llms-for-recommendation-llm4rec-3.png)

The figure traces the flow: the user's history and the retrieved candidate set both feed the prompt; the LLM produces a ranking; you parse it; you ground it to real IDs; you serve the top-K. It is exactly the `llm_rank` code from section 3, now positioned at the *end* of the funnel where the candidate count is small and the latency budget per request can absorb one LLM call (or you do it asynchronously and cache).

The economic logic is the whole point. Re-ranking twenty candidates is a few hundred input tokens, not the tens of thousands a whole-catalog scan would need. The cost per request is bounded and small, and the LLM's reasoning genuinely improves the top of the list — it can catch that a user who watched three slow-burn dramas will not want the action sequel your CTR model ranked first because action sequels have high global CTR. That is the LLM correcting a *popularity bias in your own ranker* using *world knowledge about content*, which is exactly the complementary strength we want.

But be disciplined about *when* you spend the call. Re-ranking every request synchronously will blow your latency SLA. The usual patterns: (a) re-rank only for *cold* users/items where CF is weakest and the LLM's marginal value is highest; (b) re-rank asynchronously and cache the result for a session; (c) re-rank only the top-N slate that actually gets shown, not the full candidate set. Each pattern trades coverage for cost. The stress test: what happens at 10,000 requests per second? You cannot synchronously LLM-rerank all of them — the bill and the latency forbid it. So you reserve the LLM for the slice where it pays, and the cheap ranker carries the rest. The funnel is not just an efficiency trick; it is the mechanism that makes an LLM affordable at all.

There is one more operational subtlety that decides whether role 3 survives contact with production: *caching and batching*. An LLM re-rank for a given (user-session, candidate-set) pair is deterministic enough to cache for the life of the session, so the second and subsequent page-loads in a session reuse the first re-rank for free — which often turns "one LLM call per request" into "one LLM call per session," a 5-to-20× reduction in calls depending on your session depth. Batching helps on the other axis: if you re-rank for many users in a short window, you can pack several users' prompts into one batched API call and amortize the fixed overhead, and the provider's batch tier is typically half price. The combination — cache per session, batch across users, restrict to the cold slice — is what makes the per-request cost from section 10 shrink from a deal-breaker to a line item. The mistake teams make is treating an LLM re-rank like a stateless feature computed fresh every request; treat it instead like an expensive, cacheable artifact you produce sparingly and reuse aggressively, and the economics flip from impossible to merely careful.

A final design check before you ship role 3: make sure the LLM's re-rank is *additive*, not *destructive*. The cheap ranker already encodes a lot of fitted signal; the LLM should refine the top of its list using reasoning the cheap ranker lacks, not throw away the cheap ranker's ordering wholesale. The robust pattern is to blend — take a weighted combination of the cheap ranker's score and the LLM's rank position, rather than replacing one with the other — so that when the LLM's reasoning is off (it will be, on the warm cases where CF was already right), the fitted signal still anchors the result. An LLM re-rank that *overrides* a good CF ranking on warm traffic is a common way to make offline numbers look interesting and online engagement quietly drop.

## 9. P5 and tuned recommenders (role 4): the unification idea

We will keep this short because the [next post](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice) is dedicated to it, but the conceptual move in P5 (Geng et al., "Recommendation as Language Processing: A Unified Pretrain, Personalized Prompt & Predict Paradigm," RecSys 2022) is too important to skip.

P5's insight: *every* recommendation task is text-to-text if you frame it right. Rating prediction, sequential recommendation, explanation generation, review summarization, direct recommendation — historically each needed its own architecture and loss. P5 casts them all as the same thing: a prompt in, a text answer out, trained on a shared encoder-decoder (a T5 backbone). A rating-prediction example becomes the prompt "What star rating will user_23 give item_7?" with target "4." A sequential example becomes "user_23 bought item_4, item_9, item_2, what next?" with target "item_88." One model, one objective, all tasks. The benefit is transfer — knowledge learned on rating prediction helps sequential recommendation — and zero/few-shot generalization to *new* task templates the model never saw, because it learned the general skill of "follow a personalized prompt."

The catch is that P5 uses *numeric* item IDs as tokens, so it must be *trained* on your catalog to learn what each ID means; a frozen model cannot do this. That is the boundary between role 1 (frozen, text-based, cold-start strong) and role 4 (tuned, ID/semantic-ID-based, dense-behavior strong). The successors — TIGER's semantic IDs, instruction-tuned recommenders, LoRA-adapted chat models — all live in role 4 and trade frozen flexibility for fitted accuracy. The whole tuning toolchain (LoRA, QLoRA, data formatting, evaluation) is the next post; here, just hold the unification idea: *recommendation is language processing when items are tokens*.

It is worth being precise about what tuning *fixes* and what it *introduces*, because that trade is the whole reason role 4 is a separate decision and not an obvious upgrade. Tuning fixes the two structural weaknesses of the frozen LLM. It fixes dense-behavior accuracy, because the model now learns from your interaction logs what actually co-occurs, instead of guessing from world knowledge — the popularity prior gets overwritten by fitted preferences. And it fixes catalog grounding for free when you use semantic IDs, because the model's output vocabulary *is* your catalog (a generated codeword sequence is a real item by construction, no separate grounding pass needed). What tuning introduces is the entire machinery of training: labeled data, a finetune pipeline, compute, evaluation harnesses, and the ongoing maintenance burden of retraining as the catalog and behavior drift. It also forfeits the frozen model's best property — instant cold start on a brand-new domain with zero training — because a tuned model only knows the catalog it was trained on. A genuinely new item is still cold to a tuned ID-based model until it appears in a retraining cycle, unless that model *also* consumes item text (which the best ones do). So role 4 is not strictly better than the frozen roles; it is a different point on the same trade-off surface, stronger on warm accuracy and weaker on day-zero flexibility, and you pick it when you have the data and the infrastructure to pay for the accuracy. The blend many production systems land on is role 2 (LLM text embeddings as features) *plus* a lightly tuned ranker — content for cold, fitted weights for warm, no giant generative model in the hot path. The next post makes the tuning concrete; this post's job is to place it on the map.

## 10. Worked examples: a real prompt, and the cost arithmetic

#### Worked example: one user, one prompt, a parsed ranking

Take a real-ish user. History (most recent last): *Arrival, Blade Runner 2049, Dune: Part One, Interstellar*. Retrieval handed us five candidates, with their catalog IDs:

| Pos | Candidate | Catalog ID |
| --- | --- | --- |
| 1 | Dune: Part Two | 8821 |
| 2 | The Notebook | 3310 |
| 3 | Foundation (series) | 9145 |
| 4 | Tenet | 7702 |
| 5 | La La Land | 2256 |

We send the section-3 prompt to Claude Haiku 4.5. The model returns:

```json
[1, 4, 3, 5, 2]
```

Parsing and grounding through `llm_rank`: position 1 → ID 8821 (Dune: Part Two), position 4 → 7702 (Tenet), position 3 → 9145 (Foundation), position 5 → 2256 (La La Land), position 2 → 3310 (The Notebook). The ranked catalog IDs are `[8821, 7702, 9145, 2256, 3310]`. The reasoning the model would give if asked: the user's history is hard sci-fi with a cerebral, Villeneuve/Nolan bent, so the *Dune* sequel and the Nolan film and the sci-fi series rank top; the two romances rank bottom. That ordering is *correct* and *content-aware* in a way a popularity prior would not be — *The Notebook* almost certainly has higher global popularity than *Foundation*, yet the LLM ranks it last because it read the user's taste from the titles. This is the role-1 win in miniature: reasoning over content, no training, no logs. And because we output positions and grounded them, there is zero chance of returning a phantom film.

Now stress it. Suppose retrieval had handed us 50 candidates instead of 5, with *The Notebook* listed first. Position bias would pull *The Notebook* up; popularity bias would pull it up further. The fix is the bootstrapping ensemble from Hou et al.: rank three shuffled orderings of the 50 and aggregate by mean rank. That triples the call cost — which leads directly to the next example.

#### Worked example: the token, latency, and dollar cost of LLM ranking 500 candidates versus a two-tower

Let us price ranking 500 candidates with a chat LLM, per request, and compare to a two-tower. Assume each candidate is described in roughly 16 tokens of text (title + a couple of attributes), the user history is 300 tokens, and the instruction is 200 tokens. Input tokens:

$$
N_{\text{in}} \approx I + H + C \cdot c = 200 + 300 + 500 \times 16 = 8{,}500 \text{ tokens.}
$$

Output: a ranked list of 500 numbers is about 1,500 tokens. Price it on Claude Haiku 4.5 at \$1 per million input and \$5 per million output:

$$
\text{cost} = 8{,}500 \times \frac{1}{10^6} + 1{,}500 \times \frac{5}{10^6} = \$0.0085 + \$0.0075 \approx \$0.016 \text{ per request.}
$$

On the flagship Claude Opus 4.8 at \$5/\$25 it is about \$0.08 per request. Now multiply by traffic. At a modest 100 requests per second that is roughly \$1,600 per hour on Haiku, or about \$38,000 per day — to rank, on a single tier of the funnel. Latency: an 8,500-token prefill plus a 1,500-token decode lands around 1.5 to 2.5 seconds p99, which already violates a typical 100 ms recommendation SLA.

Against this, the two-tower. Retrieval over the same 500 candidates (or the whole million-item catalog) is one user-embedding forward pass plus an ANN search. The ANN lookup over a million items with an HNSW index is on the order of tens of microseconds to a couple hundred microseconds; the dot products are a single matmul. Per-query compute cost rounds to a fraction of a hundredth of a cent of GPU/CPU time. So:

| Metric | LLM ranks 500 (Haiku) | Two-tower ANN |
| --- | --- | --- |
| Tokens / request | ~10,000 | 0 |
| Latency p99 | ~2,000 ms | < 1 ms |
| Cost / request | ~\$0.016 | ~\$0.000001 |
| Cost / day at 100 rps | ~\$38,000 | ~\$3 |

That four-orders-of-magnitude cost-and-latency gap is *the* reason you never let an LLM see 500 candidates. You let cheap retrieval cut the field to twenty first, then maybe spend one bounded LLM call on the short list — and even then only for the user/item slices where the LLM's reasoning pays. The figure makes the contrast visceral.

![Before and after contrast of an LLM ranking five hundred candidates at high token cost and seconds of latency versus a two-tower retrieving in sub millisecond time](/imgs/blogs/llms-for-recommendation-llm4rec-6.png)

The cost reality is not a footnote; it is the constraint that shapes every sane LLM4Rec architecture. The LLM goes where the candidate count is tiny and its reasoning is decisive (re-ranking cold cases), or it goes offline entirely (feature generation). It does not go in the hot path scanning the catalog.

## 11. Results: an honest before→after on a named dataset

Now the measurement angle, done the way the series demands: a named dataset, a temporal split (no leakage), the metrics that matter, and a comparison that lets the LLM lose where it should. We use the MovieLens-20M style setup that recurs across this series — leave-one-out evaluation where the most recent interaction per user is held out, NDCG@10 and HitRate@10 computed over a candidate set, plus a separately constructed *cold-start* slice of items with fewer than five training interactions.

The honest design points first, because the headline numbers are only meaningful if the protocol is sound:

- **Temporal split, not random.** Hold out each user's *last* interaction; train on everything before it. A random split leaks future behavior into training and inflates every model, but it inflates the *LLM* most because the LLM can latch onto popularity that a random split smears across time. (See [offline vs online, the two worlds](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys) for why this matters.)
- **Full metrics, not sampled.** Rank against the full candidate set or a large fixed one; the KDD'20 result (Krichene & Rendle, "On Sampled Metrics for Item Recommendation") showed sampled metrics can *reorder* methods. We report over a 100-item candidate set held constant across methods so the comparison is fair.
- **Cold-start measured separately.** Average NDCG hides the cold-start story because cold items are rare. We carve out a slice of low-interaction items and report HitRate@10 on it alone.

With that protocol, representative results (illustrative, in the range the literature and our own runs land in — exact numbers depend on the encoder and prompt, so treat them as a defensible Pareto picture, not a benchmark claim):

![Results matrix comparing zero-shot LLM few-shot LLM LLM embeddings plus CF and pure two-tower CF across NDCG cold-start hit-rate latency and cost](/imgs/blogs/llms-for-recommendation-llm4rec-8.png)

| Approach | NDCG@10 (warm) | Cold-start HR@10 | Latency p99 | Cost / 1k req |
| --- | --- | --- | --- | --- |
| Zero-shot LLM (role 1) | 0.31 | 0.42 | ~1,900 ms | ~\$30 |
| Few-shot LLM (role 1) | 0.38 | 0.45 | ~2,100 ms | ~\$45 |
| LLM-embeddings + CF (role 2 hybrid) | 0.46 | 0.40 | ~8 ms | ~\$0.02 |
| Pure CF two-tower | 0.47 | 0.11 | ~6 ms | ~\$0.01 |

Read this table carefully, because it is the post's thesis in numbers. **On warm NDCG, pure CF wins** (0.47), with the hybrid a hair behind (0.46) and the frozen LLM well behind (0.31–0.38) — the LLM's popularity and order-perception weaknesses cost it exactly where dense behavioral signal is available. **On cold start, the LLMs win big** (0.42–0.45 versus CF's 0.11) — the text-native models recommend new items from descriptions while CF is blind to them. **On cost and latency, CF and the hybrid win by four orders of magnitude** — the hybrid pays the LLM offline and serves in single-digit milliseconds, while the frozen-LLM-in-the-loop approaches cost tens of dollars per thousand requests and take seconds.

The Pareto winner is unambiguous: **the role-2 hybrid.** It captures 98% of pure CF's warm accuracy *and* nearly four times its cold-start hit-rate, at essentially CF's serving cost. That is not a coincidence — it is the direct consequence of using the LLM only where it wins (content understanding for cold items, computed offline) and CF for everything else (warm ranking, served cheaply). The frozen-LLM-as-recommender approaches are a *day-zero baseline* and a *cold-case re-ranker*, not a steady-state serving strategy.

#### Worked example: the cold-start lift, in interaction terms

Make the 0.40 vs 0.11 cold-start gap concrete. Suppose you launch 1,000 new items and each gets shown in 10,000 recommendation slates on day one. HitRate@10 of 0.11 (pure CF) means roughly 11% of those slates put a relevant new item in the top 10 — but with random ID embeddings, "relevant" is mostly chance; in practice cold items barely surface at all. HitRate@10 of 0.40 (hybrid) means 40% of slates surface a relevant new item — a 3.6× lift in new-item exposure on launch day. If even a fraction of that extra exposure converts, the merchandising team's Friday launch is no longer dead on arrival. That is the business case for role 2 in one number, and it is why the cold-start fix is the most reliably valuable thing an LLM does for a recommender. (For the broader cold-start toolkit beyond LLMs, see [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem).)

## 12. Case studies: what the literature and shipped systems actually found

**P5 (Geng et al., RecSys 2022).** The unification result. By casting rating, sequential, explanation, review, and direct recommendation as one text-to-text problem on a T5 backbone, P5 matched or beat task-specific baselines on several tasks *and* showed zero/few-shot generalization to unseen prompt templates. The lesson that stuck: a single instruction-following model can absorb the whole task zoo, and the transfer between tasks is a real benefit, not a curiosity. The limitation that stuck: it needs training on your catalog because items are numeric ID tokens, so it is a role-4 system, not a frozen one.

**LLMs are zero-shot rankers (Hou et al., ECIR 2024, arXiv 2023).** The most-cited honest assessment of role 1. They formalized recommendation as conditional ranking — sequential history as the condition, retrieved items as candidates — and measured a frozen LLM ranking them. The findings are the backbone of this post's "where it fails" section: LLMs have *promising* zero-shot ranking ability, but they (a) struggle to perceive interaction order and (b) are biased toward popular items. Crucially, both can be *mitigated* with prompt design (telling the model about recency) and *bootstrapping* (aggregating over shuffled candidate orders to wash out position bias), at which point a zero-shot LLM can challenge conventional models *on ranking a candidate set*. Note the careful scope: ranking a *candidate set*, not scanning a catalog — exactly the role-3 framing.

**LLM embeddings for cold start (the deployed pattern).** Across industry write-ups and the content-based recommendation literature, the consistent finding is that using a strong text encoder (sentence-transformers, or an LLM embedding endpoint) to featurize items closes most of the cold-start gap while leaving serving untouched — the role-2 hybrid. The numbers vary by domain, but the direction is reliable: large cold-start lift, negligible warm-accuracy loss, zero serving-cost increase. This is the role that quietly ships.

**Conversational recommendation.** Systems that wrap an LLM in a dialogue loop — ask clarifying questions, call a retrieval tool, refine — exploit the natural-language interface that is uniquely the LLM's. The research (and a growing set of products) shows these excel at *preference elicitation* for users who cannot articulate a query as keywords, and at *compositional* requests. The honest caveat is latency and the fact that the LLM still needs grounding and tool access to your real catalog; the dialogue is the front-end, the funnel is still the back-end. The [generative and conversational recommendation](/blog/machine-learning/recommendation-systems/generative-and-conversational-recommendation) post goes deep here.

The architecture that makes role 5 actually work is worth naming because it ties the whole post together. The LLM does *not* hold your catalog in its weights; it holds the *conversation* and the *reasoning*, and it reaches your catalog through tools. A typical loop: the user says "I want a sci-fi film but nothing too bleak"; the LLM turns that into a structured query, *calls your retrieval API* (the same two-tower or ANN index from the rest of your stack) to fetch real candidates, then re-ranks and explains them — which is role 3 (re-ranking) and role 5 (dialogue) composed. The catalog grounding is automatic because the candidates came *from* your catalog via the tool call, not from the LLM's imagination. This is the same retrieval-augmentation pattern that grounds any LLM application: let the model reason, let a retrieval system supply the facts. The dialogue front-end exploits the natural-language interface — the thing only an LLM can offer — while the retrieval back-end keeps serving fast, grounded, and cheap. The failure mode to avoid is letting the LLM *generate* recommendations directly in the conversation without a tool call, which reopens every hallucination and popularity-bias problem from role 1. In conversational rec, the rule is unchanged: the LLM proposes the *query and the reasoning*; the retrieval index disposes the *items*.

The through-line across all four: the LLM's value is highest exactly where collaborative signal is absent (cold start, new domain, stated-but-unlogged intent) and its cost is most justified exactly where it touches the fewest items (re-ranking a short list, embedding offline). No case study finds a frozen LLM beating a tuned CF model as a high-throughput, warm-catalog, full-funnel recommender — and you should be suspicious of any claim that does, because the cost arithmetic forbids it.

## 13. When to reach for an LLM in your stack (and when not to)

Decisive recommendations, because "it depends" helps no one in a design review.

**Reach for an LLM as a feature generator (role 2) almost always, if cold start hurts you.** This is the highest-ROI move in the whole landscape: offline text embeddings, cached, fed to your existing ranker. It fixes cold start, adds content understanding, and costs nothing at serve time. If you ship one thing from this post, ship this. The only reason not to is if your catalog is tiny and stable with no cold-start problem at all.

**Reach for an LLM as a re-ranker (role 3) for the cold/ambiguous slice.** When CF is weakest — new users, new items, sparse sessions — spend one bounded LLM call to reorder a short candidate list, and cache it. Do *not* re-rank every request synchronously; you cannot afford it and you do not need it. The warm, dense-behavior majority is served fine by the cheap ranker.

**Reach for a tuned LLM (role 4) when you have the data, the infra, and a transfer story.** If you have labeled multi-task data, a training pipeline, and a reason to want one model across rating/sequential/explanation tasks, fine-tuning (P5-style, or LoRA on a chat model) can be worth it. It is a real commitment — see the next post — and overkill if a two-tower plus a hybrid ranker already hits your targets.

**Reach for a conversational LLM (role 5) when the interface is the product.** Preference elicitation, compositional queries, a chat-native surface — these are genuine LLM-only capabilities. If your users would benefit from talking to the recommender, build it; just remember the funnel still lives behind the dialogue.

**Do NOT reach for a frozen LLM as your standalone high-throughput recommender.** On a warm catalog with dense behavioral signal, it loses on accuracy *and* on cost to a tuned two-tower. The numbers in section 11 are not close. Use it as a day-zero baseline, a cold-case re-ranker, or an offline encoder — not as the engine that serves your peak traffic.

**Do NOT skip grounding, ever.** A recommender that can return items not in your catalog is a trust bug waiting to ship. Output positions, or fuzzy-match to the catalog, or constrain decoding — pick one, but do it.

**Do NOT trust an offline LLM win without an online test.** This is the series' recurring lesson and it bites hardest with LLMs, whose biases (popularity, position) interact badly with the offline/online gap. An LLM that looks great on a sampled offline metric can flatten online because it just re-surfaces popular items. Measure on a temporal split with full metrics, then A/B test.

## 14. Key takeaways

- **The LLM and CF have mirror-image strengths.** LLMs own world knowledge and cold start; ID-based CF owns dense behavior, cost, and latency. Every good design uses the LLM only where it wins.
- **There are five LLM4Rec roles**, splitting on frozen-versus-tuned: as recommender, as feature generator, as re-ranker (frozen); tuned recommender, conversational (weight-touching/agentic).
- **The prompt is conditional generation** $P(\text{item} \mid \text{history, instruction})$, optimizing pretraining likelihood, not your ranking metric — which is *why* popularity bias (a large prior $P(x)$ for popular strings) and position bias (transformer serial-position effects) are structural, not bugs.
- **Cost scales linearly in candidates; ANN cost is roughly constant.** Never let an LLM see your whole catalog. Ranking 500 candidates is ~\$0.016 and ~2 s per request; a two-tower is ~\$0.000001 and sub-millisecond — four orders of magnitude.
- **Grounding is mandatory.** Output positions (free, airtight with a candidate set), fuzzy-match to the catalog (RAG rescue), or constrain decoding over a trie (zero hallucination by construction).
- **The hybrid (role 2) is the Pareto winner.** Offline LLM text embeddings as features capture ~98% of warm CF accuracy *and* multiply cold-start hit-rate, at CF's serving cost. Ship this first.
- **P5's idea is that recommendation is language processing** when items are tokens — one text-to-text model across all tasks, with transfer and few-shot generalization, at the cost of needing training on your catalog.
- **Measure honestly.** Temporal split, full (not sampled) metrics, a separate cold-start slice, then an online A/B — because LLM biases interact badly with the offline/online gap.
- **Compose roles; do not pick one.** The strongest real stacks run several roles at once: LLM text embeddings as features for cold start (role 2), an LLM re-rank on the cold or ambiguous slice (role 3), and a conversational front-end where the interface earns it (role 5) — all behind the same fast retrieval funnel. The question is never "LLM or CF"; it is "which stage of my funnel does an LLM improve enough to justify its cost," and the answer is usually a specific, bounded slice, not the whole system.

## 15. Further reading

- **Geng, Liu, Fu, Ge, Zhang. "Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5)." RecSys 2022.** The text-to-text unification and the foundation of generative recommendation.
- **Hou, Zhang, Wang, et al. "Large Language Models are Zero-Shot Rankers for Recommender Systems." ECIR 2024 (arXiv 2305.08845).** The honest measurement of frozen-LLM ranking — promising but order-weak and popularity-biased, with prompt/bootstrap mitigations. Code: RUCAIBox/LLMRank.
- **Rajput, Mehta, Singh, et al. "Recommender Systems with Generative Retrieval (TIGER)." NeurIPS 2023.** Semantic IDs via RQ-VAE and generating the recommended item token by token.
- **Wu, Zheng, Qiu, et al. "A Survey on Large Language Models for Recommendation." 2023.** The taxonomy this post follows; a good map of the literature.
- **Krichene, Rendle. "On Sampled Metrics for Item Recommendation." KDD 2020.** Why sampled offline metrics can reorder methods — read before you trust any offline LLM4Rec number.
- **Within this series:** [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) (the funnel map), [the two-tower model for retrieval](/blog/machine-learning/recommendation-systems/the-two-tower-model-for-retrieval), [the cold-start problem](/blog/machine-learning/recommendation-systems/the-cold-start-problem), [finetuning LLMs for recommendation in practice](/blog/machine-learning/recommendation-systems/finetuning-llms-for-recommendation-in-practice), [generative and conversational recommendation](/blog/machine-learning/recommendation-systems/generative-and-conversational-recommendation), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
- **Out-links:** for the LLM machinery itself, see the [large language model](/blog/machine-learning/large-language-model/how-to-build-effective-rag-system) track (RAG, embeddings, decoding, finetuning).
