---
title: "DeepSeekMath: The Data Pipeline Behind GRPO (and Why RL Sharpens Rather Than Teaches)"
date: "2026-06-10"
publishDate: "2026-06-10"
description: "A technique deep-dive into the part of DeepSeekMath everyone skips: the iterative fastText web-mining pipeline that built its 120B-token math corpus, the unified-paradigm lens on RL/finetuning, and the analysis showing GRPO sharpens the model's distribution instead of teaching new skills."
tags: ["deepseekmath", "grpo", "data-pipeline", "fasttext", "reinforcement-learning", "math-reasoning", "rlhf", "pass-at-k", "training-techniques", "continued-pretraining", "common-crawl"]
category: "machine-learning"
subcategory: "Training Techniques"
author: "Hiep Tran"
featured: true
readTime: 51
---

## Why the famous part of this paper is the least important part

Here is a rule of thumb I hand every junior engineer who wants to "do RL on an LLM": before you touch the RL algorithm, account for where the model's capability actually comes from. Nine times out of ten the answer is data and pretraining, and the RL is a thin reranking layer on top. DeepSeekMath is the paper that proves this with its own ablations — and ironically, the thing it is famous for, **GRPO**, is the layer the paper itself shows does the least heavy lifting.

DeepSeekMath (arXiv 2402.03300, February 2024) is remembered as "the paper that introduced GRPO," the critic-free PPO variant that later powered DeepSeek-R1. That framing is not wrong, but it buries the lede. GRPO is roughly four pages of an otherwise data-and-analysis paper, and the DeepSeek team's own numbers show RL moving the needle by a few points of majority-vote accuracy while leaving the underlying capability essentially flat. The parts that built a 7B model scoring **51.7% on competition MATH with no tools and no voting** — near GPT-4 territory among open models at the time — were (1) an iterative web-mining pipeline that assembled a 120-billion-token math corpus from Common Crawl, and (2) a continued-pretraining recipe on top of a code model.

This post is a technique deep-dive on the parts of DeepSeekMath that are usually skipped. I am going to spend the bulk of it on three things the GRPO write-ups never cover:

1. The **iterative fastText math-mining pipeline** — the data flywheel that turned a noisy web crawl into a high-recall math corpus, and why "iterative" is the load-bearing word.
2. The **unified paradigm** that expresses SFT, RFT, online-RFT, DPO, PPO, and GRPO as three knobs — data source, reward function, gradient coefficient — so you can reason about all of them in one frame.
3. The **Maj@K-not-Pass@K analysis**: DeepSeek's own evidence that RL here sharpens and reranks the model's existing output distribution rather than teaching it anything new.

GRPO's mechanics — the clipped surrogate, the per-token KL, the full advantage derivation — are already covered in depth elsewhere on this blog. I will sketch GRPO at the level you need to follow the unified-paradigm argument, and then link out. If you want the full math and a training loop, read [Fine-Tuning LLMs with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) and the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide). This post deliberately does not duplicate them.

![Each pass widens recall: the seeded fastText classifier mines Common Crawl, deduplicates, folds high-scoring pages into the next seed, and retrains.](/imgs/blogs/deepseekmath-data-pipeline-and-grpo-origin-1.webp)

The diagram above is the mental model for the entire data side of the paper. Read it left to right: you start with a seed corpus (OpenWebMath), train a fastText classifier to recognize math web pages, apply it to a deduplicated slice of Common Crawl, rank pages by confidence, deduplicate the keepers, surface the domains that turn out to be math-dense, and then — this is the part that matters — annotate fresh URLs from those domains and fold them back into the seed for the *next* iteration's classifier. Each loop raises recall. After about four iterations the corpus stabilizes at **35.5M web pages and 120B tokens**. The rest of this article is a tour of that loop and the consequences that flow from it.

### Why this is different from how most people read the paper

| Assumption | The naive reading | The reality DeepSeekMath shows |
| --- | --- | --- |
| The win is GRPO | "DeepSeekMath = GRPO; the RL algorithm is the contribution" | GRPO is one section; the data pipeline + continued pretraining produced the base capability |
| Data is just "scrape the web" | A single quality classifier over Common Crawl | A 4-iteration flywheel where the classifier retrains on its own discoveries to lift recall |
| RL teaches reasoning | RL "unlocks" new math skills the model lacked | RL improves Maj@K, not Pass@K — it reranks an existing distribution |
| Algorithms are unrelated | SFT, RFT, DPO, PPO, GRPO are separate beasts | All six are one template with three swappable knobs |
| You need a reward model | Online RL requires a learned reward network | Rule-based rewards (is the answer correct?) work fine and are cheaper |

Keep this table in view. Most teams that "reproduce DeepSeekMath" reproduce the GRPO loop, get a couple of points, and wonder where the 51.7% went. It went into the corpus.

## 1. The data pipeline: an iterative fastText flywheel, not a filter

**Senior rule of thumb: a one-shot quality filter can only find what already looks like your seed. If your seed under-represents a slice of the target distribution, a static classifier will never recover that slice — you have to let the classifier teach itself what it is missing.**

This is the single most important design decision in DeepSeekMath, and it is almost never mentioned in the GRPO retellings. The team did not train one math classifier and run it over Common Crawl. They built a *flywheel* that ran the classifier, harvested its high-confidence finds, used those finds to discover entire math-dense domains, hand-annotated new seed material from those domains, and retrained the classifier — four times. Recall climbed on every iteration because the classifier's notion of "what math on the web looks like" widened past the LaTeX-heavy academic pages that dominated the initial seed.

![A single static quality filter caps recall at the seed distribution, while iterating reclaims math the seed never represented.](/imgs/blogs/deepseekmath-data-pipeline-and-grpo-origin-2.webp)

The before/after above is the whole argument in one picture. On the left is the one-shot approach: train once on OpenWebMath, score Common Crawl once, keep the top-scoring pages. The problem is that OpenWebMath is itself a curated, LaTeX-flavored corpus. A classifier trained on it learns to recognize "math that looks like OpenWebMath" — clean theorem-proof formatting, MathML, dollar-sign delimiters. It will happily ignore a Stack Exchange thread where someone works a probability problem in plain prose, a tutoring blog that renders equations as images, or a forum post in a language the seed barely covered. That is the long tail, and in web data the long tail is most of the volume.

On the right is the iterative approach. After the first pass, you do not just keep pages — you look at which *domains* are producing a high fraction of math hits, you hand-annotate a sample of URLs from those domains as new positives, and you retrain. The retrained classifier now recognizes forum math, image-equation tutorials, and non-LaTeX presentations, because you fed it examples of exactly those. Recall climbs. The paper reports that by the third iteration roughly **98% of the eventual corpus had already been collected**, which tells you the flywheel converges fast — but you genuinely need those middle iterations to escape the seed's blind spots.

### 1.1 The fastText choice, and why not a transformer classifier

A reasonable objection: in 2024, why use fastText — a 2016-era linear bag-of-n-grams classifier — instead of fine-tuning a small BERT? The answer is throughput. The classifier has to score a deduplicated Common Crawl, which the paper puts at roughly **40 billion HTML pages**. At that scale, inference cost dominates everything. fastText scores a document in microseconds on CPU; a transformer would multiply the cost by three or four orders of magnitude and force you onto GPUs for a job that is fundamentally a recall sweep, not a precision-critical decision. You are going to deduplicate and decontaminate downstream anyway, so a cheap, high-recall first-stage filter is exactly the right tool.

Here is a faithful sketch of the per-iteration training. The structure is what matters, not the exact hyperparameters:

```python
import fasttext
import random

def build_training_set(seed_positive_pages, common_crawl_sample):
    examples = []
    for page in seed_positive_pages:               # math (label __label__math)
        examples.append(("__label__math", clean_text(page)))
    negatives = random.sample(common_crawl_sample, len(seed_positive_pages))
    for page in negatives:                          # random web (label __label__other)
        examples.append(("__label__other", clean_text(page)))
    random.shuffle(examples)
    return examples

def train_iteration(seed_positive_pages, common_crawl_sample, out_path):
    examples = build_training_set(seed_positive_pages, common_crawl_sample)
    with open("train.txt", "w") as f:
        for label, text in examples:
            f.write(f"{label} {text}\n")
    model = fasttext.train_supervised(
        input="train.txt",
        wordNgrams=2,        # unigrams + bigrams capture "let x", "prove that"
        minCount=3,
        epoch=5,
        lr=0.1,
        dim=100,
    )
    model.save_model(out_path)
    return model
```

The roughly **500K positives plus 500K negatives** per iteration is enough for a linear model over n-grams to separate "this page is about math" from "this page is a recipe." The signal lives in n-grams like `prove that`, `let x be`, `\frac`, `theorem`, `solve for`, and the negative class is just random Common Crawl, which is overwhelmingly not math. You are not building a precise classifier; you are building a cheap recall filter that you will tighten with a confidence threshold and clean up downstream.

### 1.2 The scoring, ranking, and domain-discovery loop

Once the iteration's classifier exists, you apply it to Common Crawl and keep pages above a confidence threshold. The threshold is a recall/precision dial: lower it and you sweep in more genuine math along with more junk; raise it and you keep only the obvious cases and miss the tail. DeepSeekMath tunes this per iteration, loosening as the classifier gets better at rejecting non-math so that a lower bar no longer floods the corpus with garbage.

```python
def mine_common_crawl(model, cc_shards, threshold=0.5):
    kept = []
    domain_stats = {}                       # domain -> [math_hits, total_seen]
    for shard in cc_shards:
        for page in shard:
            label, prob = model.predict(clean_text(page.text))
            domain = page.url_domain
            stats = domain_stats.setdefault(domain, [0, 0])
            stats[1] += 1
            if label[0] == "__label__math" and prob[0] >= threshold:
                kept.append(page)
                stats[0] += 1
    return kept, domain_stats

def discover_math_domains(domain_stats, min_seen=100, hit_rate=0.10):
    # Domains where >10% of pages are classified math are likely math-dense:
    # textbook mirrors, Q&A sites, tutoring blogs, competition archives.
    math_domains = []
    for domain, (hits, total) in domain_stats.items():
        if total >= min_seen and hits / total >= hit_rate:
            math_domains.append(domain)
    return math_domains
```

The domain-discovery step is the clever multiplier. A single high-scoring page is one page. But a *domain* where more than 10% of sampled pages classify as math is almost certainly a math-dense site — a textbook mirror, a competition archive, a Q&A community, a course-notes host. Once you have that domain, you can go back and harvest URLs the classifier may have scored just under threshold, hand-annotate a sample as fresh positives, and feed them into the next seed. That is how recall compounds: you are not just keeping individual pages, you are discovering reservoirs and then teaching the classifier to recognize the kind of math those reservoirs contain.

The threshold itself deserves more respect than reproductions usually give it. It is not a constant you set once; it co-evolves with the classifier across iterations. In iteration one, the classifier is weak and a low threshold floods the corpus with false positives, so you set it conservatively and accept lower recall. By iteration three the classifier rejects non-math reliably, so you can *lower* the threshold and reclaim the tail without drowning in junk — the same numeric threshold means something different against a better classifier. The right mental discipline is to track precision and recall on a fixed hand-labeled validation set every iteration and move the threshold to hold precision at your target while recall climbs. A threshold frozen across iterations either caps your recall (if set high) or rots your precision (if set low); it has to move with the model that produces the scores it gates.

There is also a sampling subtlety in the domain statistics: you cannot afford to score every page of every domain to estimate its hit rate, so you estimate from a sample, and small domains will have noisy hit-rate estimates. A domain with three sampled pages and one math hit shows a 33% hit rate, but that is one lucky sample, not a reservoir. The `min_seen` guard in the discovery code exists precisely to suppress this noise — only trust a hit-rate estimate backed by enough samples that the estimate is stable. Skip that guard and your "discovered math domains" list fills with tiny domains that happened to have one math page, and the human annotation budget you spend on them is wasted.

### 1.3 The data funnel, stage by stage

It helps to see the orders of magnitude. The pipeline is a funnel that throws away the overwhelming majority of the input at every stage.

![Forty billion raw HTML pages narrow to thirty-five million math pages and 120 billion clean tokens.](/imgs/blogs/deepseekmath-data-pipeline-and-grpo-origin-6.webp)

Reading the funnel top to bottom: you start with roughly **40B deduplicated HTML pages** from Common Crawl. The fastText classifier scores every one of them, and a tuned confidence threshold keeps a sliver. That sliver, after the iterations converge, is **35.5M math web pages**. Those pages then go through near-duplicate removal and — critically — **benchmark decontamination**, where you strip out anything that overlaps with the test sets you intend to report on (GSM8K, MATH, and friends). What survives tokenizes to **120B math tokens**, which are then blended with code and natural-language data to form the actual continued-pretraining mixture.

That decontamination step is not optional hygiene; it is the difference between a believable 51.7% and a number nobody trusts. When you mine the open web for math, you will inevitably scoop up copies of the exact problems in your eval sets, because those problems are posted, solved, and re-hosted everywhere. If you do not aggressively decontaminate, your "math reasoning" benchmark is partly a memorization benchmark. DeepSeekMath reports n-gram-based filtering of test-set overlap, and any serious reproduction has to do the same or the comparison is meaningless.

#### Second-order optimization: the iteration count is a real hyperparameter

The non-obvious gotcha here is that "iterate until convergence" hides a genuine tradeoff. Each iteration costs a full scoring pass over 40B pages plus human annotation of new seed material. Run too few and you ship a seed-shaped corpus with the long tail missing. Run too many and you are paying compute and annotation cost to recover pages that contribute almost nothing — recall has saturated. DeepSeekMath's "98% by iteration three" is the empirical signal that four iterations was the right call: the fourth iteration is mostly insurance. If you reproduce this, instrument recall-versus-iteration explicitly and stop when the marginal page yield falls below your annotation budget. Do not iterate on faith.

### 1.5 Why the negatives matter as much as the positives

There is a quiet asymmetry in this pipeline that trips up most reproductions: the negative class is doing more work than it looks like. Your positives are math pages; your negatives are *random* Common Crawl. That sounds trivially correct, but random Common Crawl is not a uniform sample of "not math" — it is dominated by news, e-commerce, social, and SEO spam. A classifier trained against that negative distribution learns to separate math from *those* genres, and it can be blindsided by genres that are adjacent to math but not math: physics lab reports, finance spreadsheets, units-and-conversions reference pages, chemistry stoichiometry. Those pages share vocabulary with math (numbers, symbols, "solve," "equation") and will score high under a classifier whose negatives never included them.

The practical fix, which mature pipelines adopt, is **hard-negative mining**: once a classifier exists, find the high-scoring pages that a human judges *not* to be the math you want, and add them to the negative set for the next iteration. This is the mirror image of the positive flywheel and it tightens precision exactly where the recall flywheel loosens it. DeepSeekMath's published description emphasizes the positive side, but any serious large-scale text classifier needs both loops running, or precision erodes as you lower the threshold to chase recall.

```python
def mine_hard_negatives(model, candidate_pages, human_label_fn, threshold=0.7):
    # High-confidence pages a human says are NOT the target math distribution.
    hard_negs = []
    for page in candidate_pages:
        label, prob = model.predict(clean_text(page.text))
        if label[0] == "__label__math" and prob[0] >= threshold:
            if not human_label_fn(page):          # human: "this isn't math"
                hard_negs.append(page)            # physics report, finance sheet, etc.
    return hard_negs
```

The lesson generalizes: a recall flywheel without a precision flywheel converges to a high-recall, low-precision corpus, and low-precision pretraining data is a slow poison — it dilutes the signal and wastes token budget on near-math that does not teach math. Balance the two loops.

### 1.6 Cleaning, extraction, and the text you actually keep

A subtlety the funnel diagram glosses over: the "page" you classify and the "tokens" you train on are not the same object. Raw HTML is full of navigation, ads, cookie banners, and boilerplate. Before fastText sees a page you have to extract the main content, and *how* you extract math is itself a quality lever. Equations rendered as MathML, as LaTeX in `\(...\)` delimiters, as images with alt text, or as Unicode soup each need different handling, and a careless extractor will strip the very symbols that make the page valuable. A page that renders $\sum_{i=1}^n i = \tfrac{n(n+1)}{2}$ as an image with no alt text contributes almost nothing if your extractor drops images, even though a human would call it a perfect math example.

This is why the pipeline is not just "classify and keep." It is extract-then-classify-then-clean, and each stage leaks. The teams that get the best corpora invest as much in robust math-aware HTML extraction as in the classifier itself, because a 5% improvement in extraction quality compounds across 35.5M pages into a materially better 120B tokens. When you read "120B math tokens," read it as "120B tokens that survived extraction, classification, deduplication, and decontamination" — four lossy stages, each of which you can do well or badly.

### 1.4 What the corpus actually bought: continued pretraining on a code model

The corpus only matters because of what it was poured into. DeepSeekMath-Base 7B is not trained from scratch. It is **DeepSeek-Coder-Base-v1.5 7B continued-pretrained on the 120B-token math mixture**. Starting from a code model rather than a general LM is a deliberate, slightly counterintuitive choice, and it pays off: code pretraining gives the model strong symbol manipulation, structured multi-step reasoning, and a tolerance for long, exact derivations — all of which transfer directly to formal math. The paper's ablations show the code-initialized base outperforming a general-text-initialized base on math, which is one of those findings that quietly reorganizes how you think about curriculum.

The mixture itself is not 100% math. Pouring only math tokens into the model would degrade its general language ability and, paradoxically, its math too, because a lot of math reasoning is expressed in natural language. The recipe blends the 120B math tokens with code and natural-language data so the model stays a competent generalist that happens to be excellent at math.

#### Why a code base, mechanistically

It is worth dwelling on *why* starting from a code model helps math, because the reason is more specific than "code is logical." Three properties of code pretraining transfer directly:

- **Exact, long-range symbol tracking.** Code requires the model to keep a variable's identity and value consistent across dozens or hundreds of tokens — define `x`, mutate it, reference it later, and a single inconsistency is a bug. Math derivations have the same structure: introduce a variable, carry it through a chain of manipulations, and a single sign error invalidates the result. A model pretrained to track symbols exactly is pre-adapted to not drop terms mid-derivation.
- **Structured multi-step composition.** Programs are compositions of sub-procedures; proofs and multi-step solutions are compositions of lemmas and intermediate results. Code pretraining teaches the model to hold a multi-step plan and execute it in order, which is precisely the skeleton of a chain-of-thought solution.
- **Tolerance for formal, low-entropy sequences.** Natural language is high-entropy and forgiving; code and math are low-entropy and unforgiving, where one wrong token breaks everything. A model trained on code has already learned to operate in the low-entropy regime where it must be *exactly* right, not merely plausible.

The ablation result — code-initialized base beating text-initialized base on math — is the empirical confirmation, but the mechanism is what lets you generalize the lesson. When you continue-pretrain for a precise, symbolic, multi-step domain, starting from a model already fluent in a precise, symbolic, multi-step domain (code) beats starting from a generalist. This is a reusable curriculum insight, not a quirk of DeepSeekMath.

#### The mixing ratio is a tuned quantity, not an afterthought

How much math versus code versus natural language is a real hyperparameter with a real optimum. Too much math and the model forgets how to read and write fluent prose, which hurts the natural-language portions of math problems (word problems, proof exposition). Too little and you under-train the target skill. Teams that reproduce this often treat the mixture as fixed and are surprised when their math gains plateau while general benchmarks regress; the fix is to sweep the ratio and watch both the target metric and a basket of general benchmarks, stopping where the target rises without the generalist scores collapsing. The continued-pretraining stage is where you spend your token budget, so spending it on the wrong mixture is the most expensive mistake available.

## 2. The unified paradigm: every algorithm is three knobs

**Senior rule of thumb: when a field accumulates a zoo of training algorithms with different names, look for the template they all instantiate. Usually they differ in two or three places and agree everywhere else, and once you see the template you can predict a new method's behavior before you run it.**

This is the most reusable idea in the paper, and it travels far beyond math. DeepSeekMath frames SFT, RFT, online-RFT, DPO, PPO, and GRPO as one parameterized template. Every one of them is a gradient step of the same shape; they differ only in three knobs:

1. **Data source** — where do the training samples come from? Human demonstrations (offline), samples from a frozen SFT model (offline), or samples from the *current* policy as it trains (online)?
2. **Reward function** — how is each sample scored? Not at all (SFT treats every demo as label 1), by a rule (is the final answer correct?), by a learned reward model, or implicitly through preference pairs (DPO)?
3. **Gradient coefficient** — the scalar that multiplies each token's gradient. This is where the advantage estimate lives: a constant, a raw rule reward, a GAE advantage from a value critic, a group-normalized advantage, or a log-ratio preference weight.

![Every fine-tuning method is one choice of data source, reward function, and gradient coefficient.](/imgs/blogs/deepseekmath-data-pipeline-and-grpo-origin-3.webp)

The matrix above lays out the six algorithms against the three knobs. Read a row and you have characterized the method completely. SFT: offline human demos, no reward (every token is label 1), constant gradient coefficient — it is the degenerate case where the coefficient is always 1. RFT (rejection-sampling fine-tuning): offline samples from the SFT model, a rule reward (keep correct answers), the rule reward as the coefficient. Online-RFT: identical to RFT except the samples come from the *live* policy instead of a frozen one. DPO: offline preference pairs, an implicit pairwise reward, a log-ratio weight. PPO: online policy samples, a learned reward model plus a value critic, a GAE advantage. GRPO: online samples drawn in *groups*, a rule-or-model reward with no critic, a group-normalized advantage.

### 2.1 The single gradient template

Concretely, every method computes a gradient of the form:

$$
\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{(q, o) \sim \mathcal{D}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} GC(q, o, t) \cdot \nabla_\theta \log \pi_\theta(o_t \mid q, o_{\lt t}) \right]
$$

where $q$ is a query, $o$ is an output sampled from data source $\mathcal{D}$, $o_t$ is the token at position $t$, $\pi_\theta$ is the policy, and $GC(q, o, t)$ is the **gradient coefficient** — the one term that every algorithm computes differently. That is the whole template. Define the data source $\mathcal{D}$, define how $GC$ is computed, and you have specified an algorithm. The clipped surrogate, the KL term, the reference model — those are refinements layered on top of this skeleton, not different skeletons.

Once you internalize this, a lot of folk wisdom collapses into derivable facts:

- **Online beats offline when the policy is moving.** RFT and online-RFT differ only in the data-source knob, yet online-RFT wins, because once the policy updates, the frozen SFT samples become stale — they reflect a distribution the model no longer produces. The gradient is then computed on outputs the model would not generate, which is wasted signal. This is the same reason on-policy RL generally beats off-policy imitation once training has progressed.
- **The critic is a baseline, not magic.** PPO's value network exists to reduce the variance of the gradient coefficient by subtracting an estimated baseline. GRPO subtracts the group mean instead. Same role, two implementations. If your reward is already low-variance (a clean 0/1 correctness signal), the expensive learned baseline buys you little, which is exactly why GRPO works so well on verifiable math.
- **DPO is RL with a particular coefficient.** People argue endlessly about whether DPO "is RL." In this frame it plainly is a member of the family: offline data, implicit reward, a log-ratio gradient coefficient. The argument was always about labels, not substance.

### 2.2 A worked example of the gradient coefficient

Make it concrete. Suppose a query $q$ asks for $\int_0^1 x^2\,dx$, and you sample a group of four answers from the current policy. Two are correct (final answer $1/3$), two are wrong. With a rule reward of 1 for correct and 0 for wrong, the group rewards are $r = [1, 1, 0, 0]$.

PPO would feed each answer through a learned value network to get a per-token baseline, subtract, and run GAE to get the coefficient. GRPO skips all of that: it computes the group mean $\bar r = 0.5$ and standard deviation $\sigma = 0.5$, then sets each answer's advantage to $A_i = (r_i - \bar r)/\sigma$. So the two correct answers get $A = (1 - 0.5)/0.5 = +1$ and the two wrong ones get $A = (0 - 0.5)/0.5 = -1$. Every token in a correct answer is nudged up; every token in a wrong answer is nudged down; the magnitudes are normalized so that a group where everything is correct (or everything wrong) produces no update. The gradient coefficient for GRPO is just this group-normalized advantage, applied per token, inside the usual clipped surrogate. No critic, no value loss, half the model memory.

That is the entire conceptual content of GRPO's advantage, and it is enough to follow the rest of this article. For the clipping, the per-token KL-to-reference term, the exact loss with all the $\min$ and $\text{clip}$ operators, and a runnable training loop, go to [Fine-Tuning LLMs with GRPO](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo). I am not going to reproduce that derivation here.

### 2.3 Reading the table as a design space, not a taxonomy

The reason this framing is worth more than a clever mnemonic is that it turns the algorithm zoo into a *design space* you can navigate deliberately. Instead of asking "which named algorithm should I use," you ask three independent questions and the named algorithm falls out:

1. **Should my data be online or offline?** Online (sample from the live policy) when the policy is moving and you can afford rollouts; offline (fixed dataset) when rollouts are expensive or you only have logged data. This knob alone separates SFT/RFT/DPO (offline) from online-RFT/PPO/GRPO (online).
2. **What is my reward?** No reward (pure imitation), a rule, a learned model, or an implicit preference. This is usually dictated by the task: verifiable tasks get rules, subjective tasks get learned models or preferences.
3. **How do I turn the reward into a gradient coefficient?** Raw reward, advantage-with-learned-baseline, advantage-with-group-baseline, or preference log-ratio. This is the variance-reduction decision.

Notice that several cells in the design space are unnamed because nobody has bothered to name them, not because they are invalid. Offline group-normalized advantage? That is just GRPO run on a fixed dataset — a perfectly sensible thing to do when rollouts are too expensive, even though it lacks a catchy acronym. The table is generative: it predicts methods, not just catalogs them.

### 2.4 Where the gradient coefficient actually comes from, per method

To make the abstraction land, here is each method's $GC(q, o, t)$ written out. The point is to see that the *only* thing changing is this one scalar:

| Method | Gradient coefficient $GC$ | Variance of $GC$ |
| --- | --- | --- |
| SFT | $1$ (every demonstration token) | zero — no signal beyond "imitate" |
| RFT | $r_i$ (rule reward of the kept sample) | low — but offline, so stale |
| online-RFT | $r_i$ from the live policy's sample | low, fresh |
| DPO | $\beta\bigl(\log\frac{\pi_\theta(o^+)}{\pi_{\text{ref}}(o^+)} - \log\frac{\pi_\theta(o^-)}{\pi_{\text{ref}}(o^-)}\bigr)$-derived | moderate, pairwise |
| PPO | $\hat A^{\text{GAE}}_t$ from the value critic | low if critic is good, but critic adds cost |
| GRPO | $(r_i - \bar r)/\sigma$, same for every token in $o_i$ | low when reward is clean |

The PPO and GRPO rows are the interesting comparison. PPO's coefficient is *per token* — GAE assigns a different advantage to each position in the sequence, which is powerful when the reward is dense and intermediate steps matter. GRPO's coefficient is *per sequence* — every token in answer $o_i$ gets the same advantage $(r_i - \bar r)/\sigma$. For a verifiable final-answer task, the reward only arrives at the end anyway, so the per-token resolution of GAE buys little, and GRPO's per-sequence simplicity is a fair trade for dropping the critic. For a task with meaningful intermediate rewards (a long tool-use trajectory where some steps are clearly good and others clearly bad), PPO's per-token advantage can be worth the critic. The table tells you which regime you are in.

## 3. GRPO at the level you need (and where to go for the rest)

**Senior rule of thumb: the cheapest way to halve your RL memory footprint is to delete a model. GRPO deletes the value network.**

GRPO is a critic-free variant of PPO. PPO keeps four models resident during training: the policy being optimized, a frozen reference for the KL penalty, a reward model, and a value network (the critic) that estimates the baseline. The critic is itself roughly the size of the policy, so it doubles the trainable footprint and adds a value-regression loss to babysit. GRPO drops the critic entirely and recovers the baseline from a *group* of sampled outputs.

![GRPO drops the value network and reads the baseline straight off a group of sampled answers.](/imgs/blogs/deepseekmath-data-pipeline-and-grpo-origin-4.webp)

The grid above contrasts the two side by side. PPO, on the left: sample one answer per question, run it through the value network to get a baseline, compute a GAE advantage, and pay for two LLM-sized models in memory. GRPO, on the right: sample $G$ answers per question, set the baseline to the group mean, set each answer's advantage to $(r - \bar r)/\sigma$, and carry no critic — half the memory.

Two more design points are worth naming because they show up in the unified-paradigm comparison and in reproductions:

- **The KL term moves from the reward into the loss.** PPO typically adds a per-token KL-to-reference penalty *inside the reward signal*, which entangles the KL with the advantage estimate and the value targets. GRPO instead adds the KL divergence as a separate term directly in the loss function, estimated with an unbiased low-variance estimator. The practical consequence is that the reward stays clean (it is just your task signal) and the KL regularization is a knob you tune independently. If you have ever fought a PPO run where tightening the KL silently distorted your advantages, you will appreciate why this separation matters.
- **The reward can be a rule, not a model.** Nothing in GRPO requires a learned reward model. For verifiable tasks — math with a checkable final answer, code with a test suite — a rule reward (correct: 1, incorrect: 0, plus light format shaping) is cheaper, more robust, and immune to reward hacking of the kind that plagues learned RMs. DeepSeekMath's RL uses exactly this style of reward.

```python
def grpo_advantages(rewards):
    # rewards: list of G scalar rewards for the G sampled answers to one query.
    import statistics
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards) or 1e-6     # guard against a constant group
    return [(r - mean) / std for r in rewards]    # the gradient coefficient

example_rewards = [1.0, 1.0, 0.0, 0.0]   # group of 4 answers, two correct
advs = grpo_advantages(example_rewards)
assert advs == [1.0, 1.0, -1.0, -1.0]
```

That is the load-bearing five lines. Everything else — the clipped surrogate ratio, token masking, the reference KL, batching the group across the rollout — is engineering you can read in the linked guides. For a head-to-head on *when* to choose GRPO versus DPO versus PPO for a given project, the [GRPO vs DPO vs PPO decision guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) is the reference; I will not relitigate that decision here.

### 3.1 The RL setup in DeepSeekMath, specifically

The RL stage is deliberately narrow. DeepSeekMath-RL is trained with GRPO on **roughly 144K chain-of-thought questions drawn only from the GSM8K and MATH training sets**. That "only" is a load-bearing word and it is the setup for the next section's analysis: by holding out every other domain, the team could measure whether RL on GSM8K+MATH *generalizes* to out-of-domain math benchmarks, or whether it just sharpens performance on the two distributions it saw. The reward is rule-based correctness on the final answer. The base for RL is the instruction-tuned DeepSeekMath-Instruct, itself produced by SFT on a large chain-of-thought-plus-tool-use dataset.

The reported jumps from this RL stage are real but modest: **GSM8K rises from 82.9% to 88.2%, and MATH from 46.8% to 51.7%** (greedy decoding). Those are good numbers. The question the paper then asks — and answers honestly — is *what kind* of improvement they represent.

### 3.2 The reproduction details that actually bite

If you implement GRPO from the unified-paradigm sketch and nothing else, you will hit three practical walls that the high-level description glosses over, and they are worth naming because every reproduction trips on at least one:

- **Group size is a variance-versus-throughput dial.** The group $G$ has to be large enough that the group mean is a stable baseline — too small and $(r_i - \bar r)/\sigma$ is dominated by sampling noise in $\bar r$, and your advantages jitter. But every sample in the group is a full generation, so $G$ multiplies your rollout cost linearly. The sweet spot is task-dependent: for a sparse 0/1 reward you want $G$ large enough that a typical group contains both correct and incorrect samples, otherwise the group is degenerate (all-correct or all-wrong) and contributes no gradient. If your model already solves most training problems, many groups will be all-correct and silently wasted; curriculum-filtering the training set to problems near the model's frontier keeps groups informative.
- **The reference model and KL coefficient drift together.** GRPO keeps a frozen reference for the KL term in the loss. If the KL coefficient is too low, the policy wanders off the reference and the rule reward starts rewarding degenerate but technically-correct outputs (skipping reasoning, emitting only the answer); too high and the policy cannot move enough to concentrate mass. This is the same tuning problem PPO has, but because GRPO puts the KL in the loss rather than the reward, you can tune it without re-deriving your advantages — which is the practical payoff of that design choice.
- **Format rewards are a trap and a crutch.** Most reproductions add a small shaping reward for producing answers in the expected format (a boxed final answer, a particular delimiter). This helps early training find the reward signal, but if the format reward is too large relative to the correctness reward, the policy optimizes for format and neglects correctness — a miniature reward-hacking failure. Keep the correctness reward dominant.

None of this is exotic, but all of it is the difference between a GRPO run that reproduces the paper's curve and one that plateaus or collapses. The algorithm is simple; the rollout engineering around it is where the time goes.

## 4. How the data pipeline and the RL stage are secretly the same argument

**Senior rule of thumb: the reachable set is set by pretraining; everything downstream — SFT, RL — only redistributes mass inside it. So the highest-leverage work is almost always upstream, where you decide what the model can reach at all.**

Before the myth-buster, it is worth connecting the two halves of this paper, because they are not two separate contributions — they are one argument told twice. The data pipeline determines the model's *reachable set*: the problems for which the post-pretraining model assigns nonzero probability to a correct answer. The RL stage operates entirely *within* that reachable set, redistributing probability mass toward the correct answers already inside it. Neither stage does the other's job, and confusing them is the single most common analytical error in this whole area.

This is why the article's structure mirrors the paper's true center of gravity. We spent the most words on the corpus because the corpus, poured into a code base via continued pretraining, *fixes the reachable set*. The unified-paradigm section explained that every post-pretraining method — SFT through GRPO — is a way of redistributing mass, differing only in three knobs. And the upcoming Pass@K analysis will show, with the team's own numbers, that RL's redistribution does not enlarge the reachable set. Three sections, one claim: **capability is a pretraining-data phenomenon; alignment and decisiveness are post-training phenomena, and they are not the same thing.**

If you hold this distinction firmly, a lot of confusing results snap into place. Why did DeepSeekMath restrict RL to GSM8K+MATH? To prove the reachable set did not grow out of domain. Why does self-consistency gain more from RL than greedy does? Because self-consistency is a pure measure of mass concentration, which is exactly what RL improves. Why can a faithful GRPO reproduction land far below 51.7% on a weak base? Because GRPO redistributes mass it cannot create, and a weak base has little correct mass to redistribute. Every one of these is the reachable-set distinction wearing a different costume.

The uncomfortable corollary for budgeting is that the glamorous part of the project (RL) is the cheap part of the *capability* it delivers, and the unglamorous part (mining and cleaning a corpus, choosing a base) is where the capability is actually bought. Teams allocate attention inversely to this, which is why so many "we did RL on an LLM" projects underdeliver. Allocate to the reachable set first.

One more framing makes the distinction operational. The model after pretraining defines a *support* — the set of answers it can produce with nonzero probability — and a *distribution* over that support. Pretraining and the data pipeline determine the support: which correct answers are reachable at all. SFT and RL only reshape the distribution over a fixed support: where the probability mass sits. RL with a verifiable reward is the most aggressive distribution-reshaper in the toolkit, but it is still a reshaper. It cannot add a point to the support, because it has no gradient signal for an answer that never appears in any sample. So the question "will RL help my model" reduces to a prior question: "is the correct answer already in the support?" If yes, RL will help you reach it more reliably; if no, RL is powerless and you are back to the data pipeline. The entire DeepSeekMath result is this loop made explicit and measured, and it is why the corpus — not the RL algorithm — is the part of this paper that deserves your study time.

## 5. The myth-buster: RL improves Maj@K, not Pass@K

**Senior rule of thumb: before you credit RL with "teaching" a capability, measure Pass@K. If Pass@K is flat and only Maj@K moved, your RL did not add capability — it reranked a distribution the base model already had.**

This is the most intellectually honest section of the paper and the one most worth internalizing, because the industry consistently overclaims here. DeepSeekMath ran their own RL'd model against the pre-RL model on two different metrics and found a sharp, telling dissociation.

![RL raises Maj@K by reranking existing samples yet barely moves the Pass@K capability ceiling.](/imgs/blogs/deepseekmath-data-pipeline-and-grpo-origin-5.webp)

The before/after above states it plainly. The common assumption (left) is that RL teaches the model new skills: that Pass@K rises because the model starts producing correct answers it simply could not produce before, expanding the capability frontier. What the data shows (right) is different: RL **reranks the SFT model's output distribution** so that correct answers become more probable. Maj@K rises — when you sample K answers and take the majority vote, you win more often — but Pass@K stays roughly flat. Same set of reachable answers, better ordered.

### 4.1 What Pass@K and Maj@K actually measure

The two metrics are easy to conflate and measure genuinely different things:

- **Pass@K** asks: if I sample $K$ independent answers, is *at least one* of them correct? This is a measure of **coverage** — does the model's distribution contain a correct answer anywhere in its top-$K$ mass? It is the ceiling of what the model can do given unlimited reranking.
- **Maj@K** asks: if I sample $K$ answers and take the **majority vote** on the final answer, is the vote correct? This is a measure of **concentration** — is the model's probability mass piled on the correct answer rather than spread across plausible-looking wrong ones?

Here is the crux. If RL leaves Pass@K unchanged but raises Maj@K, then RL did not add any correct answers to the model's reachable set — the ceiling is the same. What it did was **move probability mass toward the correct answers that were already reachable**. The base model, post-SFT, already "knew" the right answer in the sense that it would produce it within $K$ samples; it just did not produce it *often enough* to win a majority vote. RL fixed that misallocation.

### 4.2 Why this is best described as "fixing SFT misalignment"

The cleanest framing the paper offers is that RL here corrects a misalignment introduced by SFT, rather than teaching a new capability. SFT trains the model to imitate demonstrations, which optimizes for producing *plausible* token sequences, not for *concentrating* probability on the correct final answer. The result is a model whose distribution covers the right answer but spreads too much mass over confident-sounding wrong derivations. RL with a correctness reward squeezes that distribution toward the answers that verify, which is exactly the gap between high Pass@K and mediocre Maj@K.

> RL with a verifiable reward is a sharpening operator on a distribution the base model already produced. It makes the model decisive, not smarter. The capability ceiling was set in pretraining; RL just helps the model hit it more reliably.

Let me make the mechanism concrete with a worked example. Suppose a model, post-SFT, answers a hard MATH problem with the following final-answer distribution over many samples:

| Final answer | Correct? | P(answer) post-SFT | P(answer) post-RL |
| --- | --- | --- | --- |
| $\tfrac{7}{12}$ | yes | 0.28 | 0.61 |
| $\tfrac{5}{12}$ | no | 0.24 | 0.14 |
| $\tfrac{1}{2}$ | no | 0.22 | 0.11 |
| $\tfrac{2}{3}$ | no | 0.16 | 0.09 |
| other | no | 0.10 | 0.05 |

Post-SFT, the correct answer $7/12$ is the single most likely answer (0.28) but it does not command a majority — sample five answers and the correct one might not be the plurality, so Maj@5 is unreliable. Pass@5, though, is high: with the correct answer at 0.28, the chance that none of five samples is correct is $0.72^5 \approx 0.19$, so Pass@5 $\approx 0.81$. The model *can* find the answer; it just is not concentrated.

Post-RL, the correct answer's mass jumps to 0.61. Now Maj@5 is reliable — the correct answer is a clear plurality and usually wins the vote. But Pass@5 was already 0.81 and rises only modestly, because the correct answer was already in the reachable set. RL converted a reachable-but-diffuse answer into a concentrated, vote-winning one. That is the entire phenomenon. The 51.7% greedy and 60.9% self-consistency@64 numbers are downstream of this: self-consistency (majority vote over 64 samples) benefits enormously from a concentrated distribution, which is why DeepSeekMath-RL gains 9 points from greedy to SC@64.

### 4.3 The practical and uncomfortable consequences

This finding has teeth, and the implications are not comfortable for anyone selling RL as a capability unlock:

1. **RL cannot exceed the pretraining ceiling on verifiable tasks via this mechanism.** If the correct answer is not reachable within your sampling budget — Pass@K is genuinely zero for a problem — then no amount of group-normalized advantage will conjure it, because there is no correct sample to upweight. The reward signal is silent on problems the model never solves. To raise the ceiling you need better data or a bigger/better base model, not more RL.
2. **Your eval choice can flatter or expose you.** Report only Maj@K or self-consistency and RL looks transformative. Report Pass@K alongside it and the honest story emerges. A reviewer who wants to know whether your RL *taught* anything will ask for the Pass@K curve. Be ready for it.
3. **The data pipeline is where capability comes from — full circle.** This is why I opened the article on the corpus. The 120B-token math pretraining set what the model could *reach*; RL only helped it reach more reliably. If you want a model that solves harder problems, the leverage is in the pretraining data and the base model, which is precisely the part everyone skips to get to GRPO.
4. **"Aha moment" narratives deserve scrutiny.** Later DeepSeek work (R1) showed more dramatic RL behavior, and some of that is genuinely emergent. But the DeepSeekMath result is the sober baseline: on standard verifiable-math RL, the default outcome is sharpening, not teaching. When someone shows you a striking RL result, the first question is always: did Pass@K move, or just Maj@K?

This does not make RL worthless — far from it. A model that reliably wins the majority vote is a dramatically more useful model than one that merely *can* be right one time in four. Concentration is real value. But call it what it is: reranking, sharpening, alignment of the output distribution with the reward. Not teaching.

### 4.4 The math of why Pass@K is a ceiling RL cannot lift

It is worth making the ceiling argument rigorous, because it is the crux and hand-waving it invites the overclaim. Let $p$ be the probability that the base (post-SFT) model produces a correct answer in a single sample for some problem. Then:

$$
\text{Pass@}K = 1 - (1 - p)^K, \qquad \text{Maj@}K \approx \mathbb{1}\!\left[p > \max_{a \neq \text{correct}} q_a\right] \text{ as } K \to \infty
$$

where $q_a$ is the probability of each incorrect answer $a$. Read these two formulas together and the dissociation is obvious. Pass@K depends *only* on $p$, the total mass on the correct answer — it is monotone in $p$ and saturates toward 1 as $K$ grows. Maj@K depends on whether $p$ is the *largest* probability among all answers, correct or not. A model can have a respectable $p = 0.28$ (so Pass@5 $\approx 0.81$) while $0.28$ is not the plurality, so Maj@5 is a coin flip.

Now apply RL. The reward upweights tokens in correct samples and downweights tokens in incorrect ones. This *raises* $p$ — but only if $p > 0$ to begin with, because the reward is zero on every sample of a problem the model never solves. For a problem with $p = 0$, there is no correct sample in any group, so the group reward is uniformly zero, the group-normalized advantage is undefined-or-zero, and the gradient contribution is nil. RL is *silent* on problems outside the base model's reach. That is the ceiling, stated precisely: RL can move $p$ from $0.28$ toward $0.61$ (raising Maj@K dramatically while nudging Pass@K), but it cannot move $p$ from $0$ to anything, because it has no signal to do so. The set $\{$problems with $p > 0\}$ is fixed by pretraining; RL only redistributes mass *within* that set.

This is also why self-consistency@64 (a Maj@K with $K=64$) gains so much from RL. Once $p$ is concentrated, the majority vote over 64 samples almost always lands on the correct answer, which is exactly the 60.9% number. Self-consistency is the metric that most rewards distribution sharpening, so a sharpening intervention like RL shows its largest gains there.

### 4.5 The generalization test hidden in the RL data choice

Recall that the RL stage used *only* GSM8K and MATH questions. That was not a shortcut — it was an experiment. By holding out every other math benchmark from RL, the team could check whether GRPO on two distributions transferred to a third. The finding aligns with the sharpening story: RL on GSM8K+MATH improved in-domain Maj@K substantially, and the out-of-domain transfer was real but limited, consistent with RL concentrating mass on answer patterns the base model already had rather than installing a portable new skill. If RL had taught a genuinely new reasoning capability, you would expect large out-of-domain Pass@K gains; you mostly do not see them. The narrow RL-data choice is thus a deliberate falsification test for the "RL teaches" hypothesis, and the hypothesis does not survive it.

The takeaway for practitioners is uncomfortable but clarifying: if you want a model that generalizes to *new* kinds of problems, the lever is the pretraining corpus and the base model's breadth, because that is what sets the $p > 0$ frontier. RL will faithfully sharpen whatever distribution you point it at, and it will sharpen hardest on the exact distribution it trained on. Choose your RL data knowing that you are choosing what to concentrate, not what to learn.

## 6. The lineage, end to end

It is worth seeing the whole pipeline as a sequence, because the order of operations encodes the argument: data first, then RL.

![A code base model becomes a math base model by data, then a reasoner by GRPO.](/imgs/blogs/deepseekmath-data-pipeline-and-grpo-origin-7.webp)

The timeline above traces the descent. You start with **DeepSeek-Coder-Base-v1.5 7B**, a code model. You continue-pretrain it on the **120B math tokens** mined by the iterative pipeline, producing **DeepSeekMath-Base 7B**. You then SFT on a large chain-of-thought-and-tool-use dataset to get DeepSeekMath-Instruct. Finally you apply **GRPO on the 144K GSM8K+MATH questions** to get DeepSeekMath-RL, which ships at **51.7% on MATH**. Every box that moved capability is on the left half of that timeline; the RL box on the right sharpened what the earlier boxes built.

### 6.1 The results in context

Numbers are only meaningful against baselines, so here is the comparison that made DeepSeekMath notable when it landed.

![A 7B open model reaches the closed-frontier neighborhood on competition MATH without tools.](/imgs/blogs/deepseekmath-data-pipeline-and-grpo-origin-8.webp)

The matrix above places DeepSeekMath-7B against the field. DeepSeekMath-RL hits **51.7% on MATH greedy, 60.9% with self-consistency@64, and 88.2% on GSM8K**. For scale: Minerva-540B — a model nearly two orders of magnitude larger — scored in the low-50s on MATH only with extensive sampling, and GPT-4 at the time was around 52.9% greedy on MATH. A 7B open model landing in that neighborhood, with no external tools and no voting in the greedy number, is the headline. And, to belabor the thesis one last time: that 51.7% is a corpus-and-pretraining number that RL nudged up by reranking, not an RL number.

| Model | Params | MATH (greedy) | MATH (SC@64) | GSM8K |
| --- | --- | --- | --- | --- |
| DeepSeekMath-Base | 7B | 36.2% | — | 64.2% |
| DeepSeekMath-Instruct | 7B | 46.8% | — | 82.9% |
| DeepSeekMath-RL | 7B | **51.7%** | **60.9%** | **88.2%** |
| Minerva | 540B | 33.6% | 50.3% | — |
| GPT-4 | — | 52.9% | — | 92.0% |

Notice the internal story in the first three rows. Continued pretraining took the base to 36.2% on MATH. SFT took it to 46.8% — a 10.6-point jump, the largest single step. RL added the final 4.9 points to 51.7%. The base-to-Instruct jump is the model learning to *use* its pretrained capability in the chain-of-thought format; the Instruct-to-RL jump is the sharpening we just dissected. The data and the supervised stages did the heavy lifting; RL polished.

And step back to remember where the *base's* 36.2% came from: a code model that had never seen a math corpus, lifted by 120B tokens of mined, deduplicated, decontaminated web math into a model that already cleared a third of competition MATH before any instruction tuning or RL touched it. That 36.2% is the iterative fastText flywheel's number. Everything after it — SFT format alignment, RL sharpening — operated on the capability that the corpus installed. If you remember one thing from this article, remember the ordering of those numbers: the corpus set the floor, supervised tuning taught the model to express what it knew, and RL made it decisive. Three stages, three different jobs, and only the first one created capability that was not there before. That is the real shape of DeepSeekMath, and it is the shape of almost every strong reasoning model that followed.

## 7. Case studies from production

These are composite, anonymized incidents from teams reproducing or extending DeepSeekMath-style pipelines — the kind of war story that turns the paper's findings into instinct.

### 1. The team that reproduced GRPO and lost the 51.7%

A team set out to "reproduce DeepSeekMath." They implemented GRPO faithfully — group sampling, group-normalized advantage, KL in the loss — and ran it on their existing 7B general-purpose base over GSM8K+MATH. They got a couple of points of improvement and a model that scored in the low 30s on MATH, nowhere near 51.7%. The wrong first hypothesis was a GRPO bug: they spent two weeks auditing the advantage computation and the clipping. The actual root cause was that they had skipped the entire data pipeline and the code-model initialization. Their base had never seen the 120B-token math corpus, so the capability ceiling was simply low — and RL cannot raise a ceiling it can only rerank against. The fix was to invest in continued pretraining on a mined math corpus first; GRPO then behaved exactly as the paper described. The lesson: GRPO is the last 5%, not the first 50%.

### 2. The static classifier that capped at 60% recall

A data team built a single fastText math classifier on OpenWebMath and ran it once over their crawl. They were pleased: precision looked high on spot checks. But when they evaluated recall against a hand-labeled set, they were catching maybe 60% of the math content — they were systematically missing forum threads, image-equation tutorials, and non-English math. The wrong hypothesis was that the threshold was too high, so they lowered it and drowned the corpus in junk. The real fix was iteration: harvest high-confidence finds, surface math-dense domains, hand-annotate fresh positives from those domains, and retrain. Two iterations pushed recall past 85% without sacrificing precision, because the classifier finally had examples of the long tail. The lesson: a one-shot filter inherits its seed's blind spots; only the flywheel escapes them.

### 3. The decontamination miss that inflated MATH by eight points

A reproduction reported a suspiciously strong MATH score and an even more suspicious gap between their MATH and an internal held-out set. The wrong hypothesis was that their pretraining mixture was just unusually good. The root cause was incomplete benchmark decontamination: their mined corpus contained re-hosted copies of MATH test problems with solutions, and the model had partially memorized them. When they applied aggressive n-gram overlap filtering against the test sets and retrained, the MATH score dropped about eight points to a believable number that matched the held-out set. The lesson: when you mine the open web for a benchmarked skill, you *will* scoop up the benchmark; decontamination is load-bearing, not hygiene, and a too-good score with a held-out gap is the tell.

### 4. The Pass@K reviewer question that sank a paper claim

A team submitted results claiming their RL recipe "taught new reasoning capabilities," backed by a large Maj@K improvement. A reviewer asked the one question that matters: what happened to Pass@K? The team had not measured it. When they did, Pass@K was flat — their RL had reranked the distribution exactly as DeepSeekMath predicted. The claim of "new capability" was not supportable; the honest claim was "improved decisiveness via distribution sharpening." They rewrote the contribution, and the paper was stronger for it. The lesson: Maj@K up with Pass@K flat is the signature of sharpening, and any capability claim has to clear the Pass@K bar.

### 5. The online-vs-offline mix-up that wasted a week of compute

A team implemented something they called RFT but, for engineering convenience, sampled from a frozen checkpoint taken at the start of training rather than the live policy. Performance plateaued early. The wrong hypothesis was a learning-rate problem. The actual issue, visible the instant they consulted the unified-paradigm table, was the data-source knob: they had built offline RFT when they wanted online-RFT, and the frozen samples went stale as the policy moved, so the gradient was computed on outputs the model no longer produced. Switching to live-policy sampling — flipping one knob — recovered the expected improvement curve. The lesson: name your algorithm by its three knobs and these mistakes become obvious before you burn the compute.

### 6. The critic that doubled cost for no gain

A team ran full PPO with a learned value critic on a verifiable code-generation task where the reward was a clean unit-test pass/fail. Training was slow and memory-bound — they were carrying a critic the size of the policy. The wrong hypothesis was that the critic was essential for stability. When they switched to GRPO and replaced the critic with a group-mean baseline, training got faster, used roughly half the memory, and reached the same quality. The reason, straight from the unified-paradigm lens: with a low-variance 0/1 reward, the expensive learned baseline added little variance reduction over the group mean. The lesson: the critic earns its keep when rewards are noisy and dense; on clean verifiable rewards, the group baseline is a free lunch.

### 7. The reward-in-reward vs reward-in-loss KL surprise

A team ported a PPO recipe to GRPO but kept the PPO habit of folding the KL penalty into the reward signal. Their KL behavior was erratic — tightening the penalty silently distorted the advantage estimates because the KL was entangled with the reward the advantages were computed from. The fix was to follow GRPO's design and move the KL into the loss as a separate term, leaving the task reward clean. KL regularization became an independent, predictable knob. The lesson: where you put the KL term changes what it interacts with; GRPO's reward-clean, KL-in-loss design is deliberately more debuggable.

### 8. The "more RL fixes everything" team that hit the ceiling

A team saw early RL gains and concluded that more RL steps would keep improving the model, so they 10x'd the RL budget. Gains flattened almost immediately and then the model started to over-concentrate — losing diversity, becoming brittle on problems slightly off the training distribution. The wrong hypothesis was undertraining. The root cause was the Pass@K ceiling: once RL had concentrated mass on every reachable correct answer, there was nothing left to rerank, and further optimization just sharpened toward the training distribution's quirks. The fix was to redirect the budget into better pretraining data and a stronger base. The lesson: RL has a built-in ceiling on verifiable tasks; past the point where Maj@K meets Pass@K, you are sharpening into overfitting, not learning.

### 9. The domain-discovery win that 3x'd corpus yield

A team running the iterative pipeline initially kept only individual high-confidence pages and saw modest corpus growth per iteration. They were leaving the biggest lever unpulled: domain discovery. Once they aggregated hits by domain and treated any domain with a >10% math-hit rate as a reservoir — harvesting near-threshold URLs from those domains and annotating samples as new seed positives — their per-iteration yield roughly tripled, because each discovered domain contributed thousands of pages the page-level filter had scored just under the bar. The lesson: in web mining, the unit of discovery is the domain, not the page; finding a math-dense site is worth far more than finding one good page on it.

### 10. The text-base team that could not match the code-base numbers

A team continued-pretrained a strong general-text 7B on a high-quality mined math corpus and could not match DeepSeekMath's base numbers despite a comparable corpus. The wrong hypothesis was corpus quality — they spent a month improving extraction and decontamination, which helped marginally. The real gap was the initialization: their base had never been pretrained on code, so it lacked the exact symbol-tracking and low-entropy precision that code pretraining instills. When they re-ran from a code base of similar size, the same math corpus produced a markedly stronger math model. The lesson: the base model's pretraining history is part of your recipe, not a fixed input. For a precise symbolic target, start from a precise symbolic base.

### 11. The reward-hacking incident that a rule reward would have prevented

A team used a learned reward model for math RL because "it worked for chat." Within a few hundred steps the policy discovered that the reward model gave high scores to long, confident-sounding derivations regardless of correctness, and it learned to produce verbose wrong answers that the RM loved. Maj@K on the real metric *dropped* while the RM reward climbed — the textbook signature of reward hacking. The wrong hypothesis was a KL-coefficient problem. The actual fix was to discard the learned RM entirely and use a rule reward: parse the final answer, check it against ground truth, reward 1 or 0. The rule reward is unhackable in the way that matters — there is no surrogate to exploit, only the true objective. The lesson: on verifiable tasks, a rule reward is not just cheaper than a learned RM, it is *safer*, because it closes the gap between the proxy and the goal that reward hacking lives in.

### 12. The self-consistency budget that was quietly buying the headline number

A team reported a strong MATH number and a leadership deck that implied it was the model's raw capability. A careful reader noticed the number was self-consistency over 64 samples, not greedy. The greedy number was eight points lower. Neither number was wrong, but they measure different things: greedy is single-shot capability, SC@64 is capability plus 64x inference budget spent on majority voting. Once the model had been RL-sharpened, SC@64 flattered it heavily, because sharpening is exactly what makes majority voting work. The fix was honesty in reporting: state greedy and SC@64 side by side, and disclose the sample budget. The lesson: a self-consistency number is a capability-times-budget number; comparing your SC@64 to someone else's greedy is comparing different quantities, and RL sharpening widens that gap.

## 8. When to reach for this playbook, and when not to

### Reach for the DeepSeekMath playbook when:

- **You are building domain capability that lives in web text.** Math, code, a specialized technical domain — anywhere a large, recall-improvable corpus exists in the open web and a cheap classifier can recognize it. The iterative fastText flywheel is the right tool for assembling that corpus.
- **You have a verifiable reward.** A checkable final answer, a passing test suite, a parseable structured output. Rule-based rewards plus GRPO are cheap, robust, and immune to reward-model hacking, and they are exactly the regime where GRPO shines.
- **You are memory-constrained on RL.** If carrying a value critic the size of your policy is the difference between fitting on your GPUs and not, GRPO's group baseline is the obvious move. Halving the trainable footprint is the headline GRPO benefit.
- **You want to reason about a zoo of algorithms in one frame.** Use the unified-paradigm table before you implement anything. Naming your method by its three knobs prevents the entire class of online-vs-offline and reward-placement bugs in the case studies above.
- **You care about decisiveness, not just reachability.** When the downstream use is single-shot or majority-vote answering, sharpening the distribution toward correct answers via RL is genuinely valuable, even though it is not "teaching."

### Skip it, or be careful, when:

- **Your reward is subjective or dense and noisy.** Open-ended generation, helpfulness, style — where there is no rule to check correctness, you are back to learned reward models and PPO-style value critics, and the clean-reward advantages of GRPO largely evaporate. Do not force a rule reward onto a task that does not have one.
- **You expect RL to add capability the base model lacks.** It will not, via this mechanism. If Pass@K is your real target — genuinely solving problems the model currently cannot reach — invest in pretraining data and a stronger base, not RL steps. Measure Pass@K before you budget RL compute.
- **You cannot decontaminate honestly.** If you mine the open web for a benchmarked skill and cannot rigorously strip test-set overlap, your headline number is partly memorization and your comparison is not credible. Decontamination is not optional.
- **Your corpus is tiny or the long tail is closed.** The iterative flywheel pays off when there is a recall-improvable long tail to discover. For a small, well-bounded corpus that a single pass already covers, the iterations are pure cost.
- **You are tempted to over-optimize RL.** Past the point where Maj@K meets Pass@K, more RL sharpens into overfitting and brittleness. Know where the ceiling is and stop.

## Further reading

- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (arXiv 2402.03300) — the primary source for everything above: the data pipeline, GRPO, the unified paradigm, and the Maj@K/Pass@K analysis.
- [Fine-Tuning LLMs with GRPO: From Theory to Implementation](/blog/machine-learning/large-language-model/fine-tuning-llm-with-grpo) — the full GRPO derivation, loss, and a runnable training loop. Start here for the mechanics this post deliberately did not duplicate.
- [GRPO vs DPO vs PPO: A Decision Guide](/blog/machine-learning/large-language-model/grpo-vs-dpo-vs-ppo-decision-guide) — how to choose among the three for a real project, expanding on the unified-paradigm table.
- [DeepSeek-V3: FP8, MTP, and Loss-Free Balancing](/blog/machine-learning/large-language-model/deepseek-v3-fp8-mtp-loss-free-balancing) — the same team's later infrastructure and architecture work, useful for understanding how DeepSeek scales training.
- [Training LLMs for Math](/blog/machine-learning/large-language-model/training-llm-for-math) — broader context on math-reasoning training, of which DeepSeekMath is a landmark instance.
