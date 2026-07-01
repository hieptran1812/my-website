---
title: "Data Mixing, Domain Weighting, and Curriculum: The Recipe, Not Just the Ingredients"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "Once your corpus is clean, the next lever is proportion. A principal-engineer's guide to domain weighting, DoReMi's learned mixtures, multilingual temperature sampling, ordering, and the high-quality annealing phase that makes the last tokens punch above their weight."
tags: ["training-data", "data-mixing", "domain-weighting", "doremi", "temperature-sampling", "curriculum-learning", "annealing", "mid-training", "pretraining-data", "group-dro", "multilingual", "data-curation"]
category: "machine-learning"
subcategory: "Training Data"
author: "Hiep Tran"
featured: true
readTime: 35
---

Two teams start with the exact same pile of cleaned, deduplicated, decontaminated tokens. Same web crawl, same code repos, same books, same Wikipedia dump, same math corpus. One team ships a model that tops the math and code leaderboards at 8B parameters. The other ships a model that is fine at everything and remarkable at nothing, from the identical raw material. The difference is not the ingredients. It is the recipe: how many tokens of each source the model actually sees, in what proportion, in what order, and what it reads *last*.

This is the part of the data pipeline that quietly decides your model's personality, and it is the part that gets the least rigor. Everyone can tell you their deduplication rate. Almost nobody can tell you *why* their code fraction is 12 percent instead of 8, or whether upsampling Wikipedia three times actually helped or just felt safe. The mixture is treated as a hand-me-down constant — copied from the last run, which copied it from a paper, which picked it partly by intuition. That is a mistake, because the mixture is a hyperparameter with a larger effect on downstream capability than most architecture choices you would agonize over.

The diagram below is the mental model for the whole post: a pretraining corpus is not a bucket, it is a *portfolio*. Each domain has a deliberately chosen token share, and that share is almost never the natural distribution of the raw data. The web is 90-plus percent of what you crawled; here it is 67 percent of what the model sees. Wikipedia is a rounding error of the raw internet; here it is upsampled to three epochs. Every one of those numbers is a decision, and this article is about how to make those decisions with evidence instead of vibes.

![Stacked view of a pretraining corpus as six weighted domains, from 67 percent filtered web down to 3 percent thrice-epoched Wikipedia](/imgs/blogs/data-mixing-domain-weighting-and-curriculum-1.webp)

If you have not read the earlier posts in this series, this one sits directly downstream of [data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning) — selection decides *which documents survive*; mixing decides *how often the survivors are sampled*. It also leans hard on [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality), because every mixing decision below is only as trustworthy as the ablation loop that scores it.

## Why the mixture is different from what you assume

The intuition most engineers carry into their first mixing decision is that data is additive: if math data helps and code data helps, then more of each helps more, and the optimal recipe is just "as much good stuff as possible." Almost every part of that intuition is wrong in a way that costs you accuracy.

| Assumption | Naive view | Reality |
| --- | --- | --- |
| More of a helpful domain is better | Monotonic: keep adding | Diminishing, then *negative*, once the domain is epoched past its unique content |
| Per-domain gains add up | Optimal mix = union of best domains | Domains compete for fixed capacity; a domain that helps alone can hurt in the mix |
| The natural distribution is a fine default | Sample proportional to token counts | Web dominates by volume, not by value-per-token; you must reweight |
| Order does not matter for a big enough model | Shuffle everything, done | Bulk order is robust, but the *last* few percent of tokens matter disproportionately |
| One global mixture is enough | Pick weights once, train | The best setups shift the mixture mid-flight — mid-training and an annealing cooldown |
| Multilingual balance is just "add more languages" | Sample languages proportionally | Proportional sampling starves low-resource languages; you need temperature |

Every row in that table is a section of this post. The throughline is that mixing is an *optimization problem over proportions*, not a shopping list, and the objective — downstream capability at a fixed token budget — is non-linear, non-additive, and stage-dependent.

## 1. Domain proportions as a first-class hyperparameter

**The senior rule: treat every domain's token share as a tunable knob with a measured effect, not a constant you inherited.**

Start with the vocabulary, because the field is sloppy about it. Let there be domains indexed by $d$, each contributing $N_d$ raw tokens. The natural distribution samples domain $d$ with probability proportional to $N_d$. A *mixture* replaces that with a chosen weight vector $\alpha = (\alpha_1, \dots, \alpha_D)$ where each $\alpha_d \ge 0$ and $\sum_d \alpha_d = 1$. During training, each batch draws its documents from domain $d$ with probability $\alpha_d$. That single vector is the object every technique in this post is trying to set.

The weight $\alpha_d$ and the raw count $N_d$ together determine the *effective epoch count* for domain $d$. If your total training budget is $T$ tokens, domain $d$ contributes $\alpha_d \cdot T$ tokens to the run, drawn from a pool of $N_d$ unique tokens, so it is seen roughly ${(\alpha_d \cdot T)/N_d}$ times. This is the number that actually matters and the one people forget to compute. A "3 percent Wikipedia" weight sounds gentle until you notice Wikipedia has only about 30 billion unique tokens, so at a 3-trillion-token budget that is ${90/30 = 3}$ epochs — you are showing the model every Wikipedia article three times. Meanwhile "67 percent web" against a 15-trillion-token filtered crawl is a fraction of one epoch. Same run, wildly different epoch counts, and the epoch count is what governs memorization risk (covered in the troubleshooting section).

Here is the standard modern web-heavy recipe from the figure above, written out with the numbers that matter:

| Domain | Weight | Tokens (3T budget) | Unique pool | Effective epochs |
| --- | --- | --- | --- | --- |
| Web (filtered CommonCrawl) | 67% | 2.0T | 15T | 0.13 |
| Code (The Stack v2) | 10% | 300B | 900B | 0.33 |
| Books + academic | 8% | 240B | 400B | 0.60 |
| Multilingual web | 8% | 240B | 3T | 0.08 |
| Math (proof-pile, web-math) | 4% | 120B | 80B | 1.50 |
| Wikipedia | 3% | 90B | 30B | 3.00 |

Two things jump out. First, the small high-value domains (math, Wikipedia) are the ones being epoched hardest — that is deliberate, they are dense with the reasoning and factual patterns you want, but it is also exactly where overfitting starts. Second, the web, despite being two-thirds of the mixture, is barely a tenth of an epoch: there is so much of it that dilution is free. The whole art is deciding how far to push the top rows without tipping them into memorization.

### Second-order effect: mixture weights interact with tokenizer and sequence packing

A subtlety that bites people who copy a mixture across projects: $\alpha_d$ is a probability over *documents* or *tokens*, and which one you chose changes the actual composition. If you sample proportionally to documents but code files are three times longer than web pages on average, your token-level code fraction silently inflates. Always define and log the mixture at the token level, and re-derive it whenever you change the tokenizer, because a tokenizer that splits code more aggressively will change every domain's token count and therefore every effective epoch. The mixture is not portable across tokenizers, and treating it as portable is a common source of "we copied their recipe and it didn't work."

## 2. Manual mixtures: The Pile, Llama, and upsampling by feel

**The senior rule: a hand-tuned mixture is a hypothesis, and it is only as good as the ablations you ran to justify each weight — most people run none.**

The first generation of open mixtures was entirely manual, and studying them is worth it because the reasoning is explicit. The Pile (2020) was 22 sub-datasets with per-source weights chosen by the authors, and crucially with *deliberate upsampling multipliers*. Wikipedia was set to about 3 epochs, Books3 and the OpenWebText2 sources were weighted up relative to their raw size, and the sprawling but noisy Common Crawl derivative (Pile-CC) was held to a fraction of what its raw volume would have justified. The Pile paper is refreshingly honest that these were judgment calls informed by perplexity checks, not a solved optimization — they picked weights that made a small model's validation loss look good across their component sets and shipped it.

Llama-1's mixture followed the same philosophy with different numbers: roughly 67 percent CommonCrawl, 15 percent C4 (a *second*, differently-filtered web source, because the authors found the two web crawls were complementary rather than redundant), around 4.5 percent each for GitHub and Wikipedia and books, and 2.5 percent for arXiv and StackExchange. The Wikipedia and books portions were trained for roughly two epochs while everything else stayed near one — the same "upsample the good small stuff" instinct as The Pile, expressed as an epoch multiplier rather than a weight.

The pattern across every manual mixture is identical: **downweight the abundant-but-noisy web below its natural share, and upsample the scarce-but-dense sources above theirs.** That instinct is correct in direction. The problem is that "how much" was answered by feel, prior runs, and a few small-scale perplexity checks, and perplexity on a held-out slice of a domain is a notoriously weak proxy for whether that domain's weight is actually optimal for downstream tasks. You can lower validation perplexity on your Wikipedia slice by cranking Wikipedia's weight while quietly degrading the model's code ability, and a perplexity dashboard will show you nothing but green.

The manual approach has three real failure modes, and they motivate everything in the next section: it does not scale to many domains (nobody hand-tunes 100 language weights), it gives no guarantee about the *worst* domain (you can accidentally starve one), and it is not reproducible (the next engineer inherits your numbers with no idea which were load-bearing).

## 3. Learned domain weights: DoReMi and Group DRO

**The senior rule: if you cannot justify a domain weight with a measurement, let a cheap proxy model measure it for you.**

DoReMi (Domain Reweighting with Minimax Optimization, Xie et al., 2023) is the cleanest answer to "stop guessing the weights." The insight is that you do not need to run your expensive 8B model to find good domain weights — you can find them on a tiny proxy and *transfer* the resulting weight vector to the big run. The diagram below is the whole method.

![DoReMi flow: a baseline mix trains a reference and a proxy model, per-domain excess loss feeds Group DRO reweighting to produce transferable domain weights](/imgs/blogs/data-mixing-domain-weighting-and-curriculum-2.webp)

The procedure has three stages. First, train a small *reference* model (the paper uses 280M parameters) on some baseline mixture — uniform, or a heuristic guess. Second, train a small *proxy* model of the same size, but instead of a fixed mixture, optimize its domain weights online with Group Distributionally Robust Optimization (Group DRO). Third, take the average of the proxy's domain weights over training and use that vector to train the large model.

The objective is what makes this principled. Group DRO does not minimize average loss; it minimizes the *worst-case* loss over domains, measured as **excess loss** — the proxy's loss on domain $d$ minus the reference's loss on domain $d$. Excess loss matters because raw loss is not comparable across domains: some domains are intrinsically high-entropy (web) and some are low (structured code), so a domain with high absolute loss is not necessarily one the model is failing to learn — it might just be harder. Subtracting the reference baseline normalizes for that, leaving a quantity that means "how much room is left to improve on this domain." Group DRO then continually shifts weight toward whichever domain currently has the highest excess loss, and the training dynamics look like an exponentiated-gradient update on the weight vector:

$$
\alpha_d^{(t+1)} \propto \alpha_d^{(t)} \cdot \exp\!\big(\eta \cdot \ell^{\text{excess}}_d\big)
$$

where $\ell^{\text{excess}}_d$ is the current per-domain excess loss and $\eta$ is a step size. Domains the proxy is still struggling to fit get their weight pushed up; domains it has essentially mastered relative to the reference get pushed down. Here is that update as runnable code, which is genuinely all the DRO bookkeeping amounts to:

```python
import numpy as np

def doremi_update(alpha, proxy_loss, ref_loss, eta=1.0, smoothing=1e-3):
    """One Group DRO step on the domain-weight vector.

    alpha       : (D,) current domain weights, sums to 1
    proxy_loss  : (D,) proxy model's per-domain loss this step
    ref_loss    : (D,) reference model's per-domain loss (fixed)
    """
    excess = np.maximum(proxy_loss - ref_loss, 0.0)   # clip: no credit for beating ref
    alpha = alpha * np.exp(eta * excess)              # exponentiated-gradient step
    alpha = alpha / alpha.sum()                        # renormalize to the simplex
    # uniform smoothing keeps any domain from collapsing to zero weight
    D = len(alpha)
    alpha = (1 - smoothing) * alpha + smoothing * (np.ones(D) / D)
    return alpha / alpha.sum()

# domains: [web, code, books, math, wiki]
alpha   = np.array([0.20, 0.20, 0.20, 0.20, 0.20])   # start uniform
ref     = np.array([2.10, 1.05, 2.30, 1.80, 2.05])   # reference per-domain loss
for step in range(2000):
    # in a real run proxy_loss comes from a forward pass on a batch sampled ~ alpha
    proxy = ref + np.array([0.35, 0.02, 0.18, 0.40, 0.05]) * np.exp(-step / 800)
    alpha = doremi_update(alpha, proxy, ref, eta=0.5)
print({d: round(w, 3) for d, w in
       zip(["web","code","books","math","wiki"], alpha)})
```

The DoReMi paper's headline result on The Pile is that these proxy-tuned weights, transferred to a much larger model, improved downstream accuracy and reached the baseline's perplexity roughly 2.6 times faster in training — with the domain weights found by a 280M proxy applied to an 8B model. The weights it discovered were not intuitive: it *downweighted* some domains that human curators had upsampled, because those domains were already easy for the model relative to the reference and were burning budget that harder domains could use better.

There are caveats worth stating plainly, because DoReMi is not magic. The result is sensitive to the choice of reference mixture — a bad reference gives bad excess-loss signals. It optimizes for *balanced* domain loss, which is not identical to optimizing your specific downstream eval suite; if you care disproportionately about code, minimizing worst-case loss across all domains may leave code gains on the table. And transfer across a large scale gap is empirical, not guaranteed — the weights that are optimal at 280M are *usually* close to optimal at 8B, but "usually" is doing work in that sentence. Treat DoReMi as a strong prior on the weights, then confirm with a few full-scale ablations, not as an oracle.

### Manual versus learned, side by side

The contrast is worth making explicit, because the choice is not purely technical — it is about reproducibility and risk.

![Before-after comparison: manual hand-tuned mixing versus learned DoReMi weights, contrasting guesswork against a reproducible robustness guarantee](/imgs/blogs/data-mixing-domain-weighting-and-curriculum-3.webp)

Manual mixing is a one-time guess with no guarantee about the worst domain and no record of which weights were justified. Learned mixing gives you a reproducible objective, a worst-case-domain guarantee, and — the part that pays for the extra proxy run — a real downstream lift at the same token budget. If you are training anything you will iterate on, the proxy run is cheap insurance. If you are doing a single throwaway experiment, a copied manual mixture is fine.

## 4. Temperature sampling for multilingual balance

**The senior rule: when one category dwarfs the others by volume, sample by a tempered distribution, not the raw one, or the tail never gets learned.**

Multilingual data is the sharpest case of the "natural distribution is a bad default" problem. If you crawl the web and sample languages proportionally to their token counts, English (and a handful of other high-resource languages) will be 80-plus percent of every batch, and languages like Swahili or Telugu will appear so rarely the model never builds competence in them. But you also cannot just go uniform — treating Swahili and English as equally weighted when English has a thousand times more data wastes enormous capacity re-reading a tiny Swahili corpus while ignoring most of English.

The mC4 / mT5 solution is **temperature sampling**. Given the natural per-language probabilities $p_\ell$ (proportional to each language's token count), sample instead from a tempered distribution:

$$
q_\ell = \frac{p_\ell^{\,\alpha}}{\sum_{\ell'} p_{\ell'}^{\,\alpha}}
$$

where the exponent $\alpha \in (0, 1]$ is the temperature knob (mT5 used $\alpha = 0.3$; some papers parameterize it as a temperature $T$ with $\alpha = 1/T$, so an exponent of 0.3 is a temperature of about 3.3 — same idea, watch which convention a paper uses). When $\alpha = 1$ you recover the natural distribution. As $\alpha$ drops toward zero the distribution flattens toward uniform, pulling mass off the head and onto the tail. The figure below shows exactly what that does to a four-language toy distribution.

![Matrix of sampling probabilities by exponent alpha and language, showing English falling from 60 to 35 percent while Swahili rises from 5 to 17 percent as alpha drops](/imgs/blogs/data-mixing-domain-weighting-and-curriculum-4.webp)

Read the columns top to bottom. At $\alpha = 1$, English takes 60 percent of samples and Swahili gets 5. Drop to $\alpha = 0.3$ and English falls to 35 percent while Swahili climbs to 17 — more than triple its natural share. The mass came off the head and went to the tail, which is precisely the point: low-resource languages get enough repetitions to actually learn, high-resource languages give up some of their (heavily diminished-returns) extra epochs. Here is the computation, which is short enough that there is no excuse for guessing:

```python
import numpy as np

def temperature_sample_weights(token_counts, alpha=0.3):
    """Tempered sampling weights over categories (languages, domains).

    token_counts : dict name -> raw token count
    alpha        : exponent in (0, 1]; lower = flatter = more upsampling of the tail
    returns      : dict name -> sampling probability, summing to 1
    """
    names  = list(token_counts)
    counts = np.array([token_counts[n] for n in names], dtype=float)
    p      = counts / counts.sum()          # natural distribution
    q      = p ** alpha                      # temper
    q      = q / q.sum()                     # renormalize
    return dict(zip(names, q))

counts = {"English": 600e9, "German": 250e9, "Hindi": 100e9, "Swahili": 50e9}
for a in (1.0, 0.7, 0.3):
    w = temperature_sample_weights(counts, alpha=a)
    print(f"alpha={a}: " + ", ".join(f"{k} {v:.0%}" for k, v in w.items()))
```

The same trick works for *any* categorical imbalance, not just languages — you can temper domain weights, or source weights within a domain (e.g. balancing across GitHub languages inside your code domain). The knob to internalize: **lower $\alpha$ helps the tail and costs the head; the right value depends on how much you care about tail competence versus head fluency.** A model meant to be strong in 100 languages wants a low $\alpha$ (0.2-0.3); a primarily-English model that just needs to not be helpless in other languages wants something closer to 0.7. There is no universal best value, which is exactly why it is a knob and not a constant.

### Second-order effect: temperature interacts with epoch count on the tail

The catch nobody mentions: aggressively upsampling a low-resource language with a small $\alpha$ means you are epoching that tiny corpus many times. Swahili at 17 percent of a 3-trillion-token run is 510 billion tokens drawn from maybe 50 billion unique — ten epochs. You have solved the "never seen it" problem and created a "memorized it" problem. Temperature sampling and the diminishing-returns-of-epoching problem from the troubleshooting section are the same problem viewed from two angles, and the fix is the same: cap the effective epochs, and if a language needs more than a few epochs to reach target competence, the honest answer is that you need more *unique* data for it, not more repetitions.

## 5. Curriculum and ordering: does the sequence matter?

**The senior rule: for the bulk of training, order is surprisingly robust — do not over-engineer a curriculum — but the boundaries of training (warmup and especially the end) are where order earns its keep.**

There is a persistent hope that language models, like students, should learn easy things before hard things — short before long, simple before complex, high-frequency before rare. This is "curriculum learning," and the honest empirical summary is: for large-scale pretraining, elaborate difficulty-ordered curricula mostly *do not help*, and sometimes hurt, once the model and dataset are large. A well-shuffled uniform mixture is a strong baseline that is very hard to beat with a clever ordering, because SGD over hundreds of billions of tokens washes out most sequence effects — the model sees so much of everything that the order it saw things in early on gets overwritten.

That is the robust, boring, correct default: **shuffle thoroughly and trust it.** If you take one thing from this section, it is that you should not spend a month building a difficulty scorer to order your corpus, because the expected payoff is near zero and you risk introducing subtle distributional artifacts (all the "easy" data at the start biases early representations in ways that can be sticky).

But "order mostly does not matter" has two important exceptions, and they are the reason this section is not just one sentence. The first is *recency*: whatever the model sees in the final phase of training, when the learning rate is low and updates are precise, sticks disproportionately. The second is *contiguity artifacts*: if your shuffle is imperfect and a domain clumps — say all your code lands in one contiguous stretch because of how shards were ordered — you get a transient during which the model over-specializes and then forgets. Both exceptions point at the same high-value region: the *end* of training. That is where the mixture stops being uniform and starts being a curriculum-of-one-step, and it is the single most important scheduling decision in modern pretraining.

## 6. Annealing and mid-training: why the last tokens punch above their weight

**The senior rule: reserve your highest-quality data for the learning-rate cooldown at the end of pretraining — those tokens shape the final model far more than their count suggests.**

Modern pretraining is not one uniform pass. The best open recipes — Llama-3, OLMo, MiniCPM, and the FineWeb ablations — all end with a distinct **annealing** phase (also called the cooldown, or decay phase): the last 10-ish percent of the token budget, during which the learning rate decays from its stable value down toward zero, and *simultaneously the data mixture is upgraded* to a curated blend of the highest-quality sources plus, increasingly, synthetic data. The timeline below is the shape every large run now has.

![Timeline of a pretraining run: warmup, a long stable web-heavy phase, mid-training reweight, then the annealing cooldown on a high-quality mix before post-training](/imgs/blogs/data-mixing-domain-weighting-and-curriculum-5.webp)

Why does the end matter so much? Because of how the learning rate schedule interacts with what the model retains. During the high-LR stable phase, the model is making large, noisy updates — it is learning general structure but also constantly overwriting itself. During the cooldown, updates are small and precise: the model is settling into a minimum, and whatever data it settles on is what it settles *toward*. The loss curve visibly bends downward during annealing — there is a characteristic sharp drop when the LR decays — and the FineWeb team used exactly this to build a fast data-quality evaluator: take a mostly-trained model, anneal it on candidate data versus a reference, and compare the downstream lift. Data that produces a bigger annealing drop is better data. The last tokens are a magnifying glass.

The mechanism is worth animating, because it is two things happening at once — the learning rate falling *and* the mixture shifting — and the coupling is the whole point.

<figure class="blog-anim">
<svg viewBox="0 0 760 380" role="img" aria-label="A training run: the learning rate stays high through the bulk web-heavy phase, then decays to zero in the final annealing window while the data mix flips from web-heavy blue to high-quality green" style="width:100%;height:auto;max-width:820px">
<style>
.an-axis{stroke:var(--border,#d1d5db);stroke-width:1.5}
.an-lr{fill:none;stroke:var(--accent,#6366f1);stroke-width:3}
.an-zone{fill:#2f9e44;opacity:.12}
.an-head{stroke:var(--text-primary,#1f2937);stroke-width:2;stroke-dasharray:4 4}
.an-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.an-sub{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.an-web{fill:var(--accent,#6366f1)}
.an-hq{fill:#2f9e44}
.an-neu{fill:var(--border,#d1d5db)}
.an-cap{font:600 14px ui-sans-serif,system-ui;text-anchor:middle}
.an-capA{fill:var(--accent,#6366f1)}
.an-capB{fill:#2f9e44}
@keyframes an-sweep{0%{transform:translateX(0)}100%{transform:translateX(640px)}}
@keyframes an-fadeA{0%,78%{opacity:1}92%,100%{opacity:0}}
@keyframes an-fadeB{0%,78%{opacity:0}92%,100%{opacity:1}}
.an-play{animation:an-sweep 10s linear infinite}
.an-mixA{animation:an-fadeA 10s linear infinite}
.an-mixB{animation:an-fadeB 10s linear infinite}
@media (prefers-reduced-motion:reduce){.an-play{animation:none;transform:translateX(560px)}.an-mixA{animation:none;opacity:0}.an-mixB{animation:none;opacity:1}}
</style>
<text class="an-lbl" x="60" y="40">learning rate over the run</text>
<rect class="an-zone" x="590" y="60" width="110" height="200"/>
<text class="an-sub" x="595" y="80">annealing</text>
<path class="an-lr" d="M60,180 L110,100 L590,100 C650,100 665,250 700,250"/>
<line class="an-axis" x1="60" y1="260" x2="700" y2="260"/>
<text class="an-sub" x="60" y="278">0%</text>
<text class="an-sub" x="360" y="278">training progress</text>
<text class="an-sub" x="675" y="278">100%</text>
<line class="an-head an-play" x1="60" y1="60" x2="60" y2="260"/>
<g class="an-mixA">
<rect class="an-web" x="60"  y="300" width="448" height="34" rx="4"/>
<rect class="an-neu" x="508" y="300" width="77"  height="34" rx="4"/>
<rect class="an-neu" x="585" y="300" width="115" height="34" rx="4"/>
<text class="an-cap an-capA" x="380" y="360">bulk web-heavy mix</text>
</g>
<g class="an-mixB">
<rect class="an-hq"  x="60"  y="300" width="288" height="34" rx="4"/>
<rect class="an-hq"  x="348" y="300" width="128" height="34" rx="4" opacity="0.7"/>
<rect class="an-neu" x="476" y="300" width="128" height="34" rx="4"/>
<rect class="an-web" x="604" y="300" width="96"  height="34" rx="4"/>
<text class="an-cap an-capB" x="380" y="360">high-quality + synthetic cooldown</text>
</g>
</svg>
<figcaption>The last tokens punch above their weight: as the learning rate decays through the annealing window, the data mix shifts from web-heavy (blue) to a curated high-quality and synthetic blend (green).</figcaption>
</figure>

**Mid-training** is the closely related idea one step earlier: before the final cooldown, many recipes shift the mixture at around 75-90 percent of the budget to upweight the domains they most want the model to be good at — more math, more code, more instruction-like data — while the LR is still moderately high enough to actually learn new capabilities rather than just polish. Llama-3's documented recipe adjusts the data mix in stages and reserves a high-quality mixture for the final phase; the effect is a model that has broad coverage from the stable phase and sharpened, deliberately-chosen strengths from the end.

The practical recipe for the annealing set: it should be your genuinely best data — the top decile of your quality classifier's scores, hand-curated instruction and reasoning data, high-quality math and code, and carefully-filtered synthetic data generated to fill known gaps. It should be *diverse* (annealing on a narrow set overfits to that set), and it must be *rigorously decontaminated*, for reasons the troubleshooting section makes vivid. Budget it at roughly the last 10-20 percent of tokens; the LR decay shape (linear or cosine to near-zero) matters less than the mixture upgrade.

## A worked scenario: computing a defensible mixture from ablation deltas

Enough principle. Here is the calculation a staff engineer actually runs when someone asks "so what weights should we use?" Suppose you have five domains and, from the ablation loop in [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality), you have measured each domain's **marginal per-token value**: the change in your downstream eval score when you add a fixed slice of that domain to a baseline mix, normalized per billion tokens. You also know each domain's unique-token pool. The measurements come back like this:

| Domain | Marginal value (pts / 100B tok) | Unique pool | Interference flag |
| --- | --- | --- | --- |
| Web | +0.4 | 15T | none |
| Code | +1.1 | 900B | mild with math |
| Books | +0.6 | 400B | none |
| Math | +1.8 | 80B | strong with code |
| Wikipedia | +0.9 | 30B | none |

The naive move is to sort by marginal value and pour in the top domains: math is worth the most per token, so make it huge. This is wrong for two reasons the table already tells you. Math has only 80 billion unique tokens, so a huge weight means many epochs (memorization). And math has a *strong interference flag* with code — measured by running math and code together and seeing the combined gain fall short of the sum of their solo gains.

The defensible procedure is a constrained allocation, not a sort:

1. **Set a per-domain epoch cap.** Say no domain may exceed 4 epochs (past that, our epoching ablation showed negative returns). At a 2-trillion-token budget, that caps math at ${4 \times 80\text{B} = 320\text{B}}$ tokens (16 percent) and Wikipedia at ${4 \times 30\text{B} = 120\text{B}}$ tokens (6 percent), regardless of how good their marginal value looks.
2. **Start from marginal value, then apply the caps.** Math wants a big weight on value alone, but the epoch cap binds first — it lands at 16 percent, not 40.
3. **Apply the interference discount.** Because math and code interfere strongly, do not give both their standalone-optimal weight. Run the pair together at a few weight ratios and pick the ratio that maximizes *combined* downstream score. Suppose that ablation says a 12 percent code / 12 percent math split beats 16/16 (the extra math and code were cannibalizing each other's capacity). Take the 12/12.
4. **Fill the remainder with the diluting workhorse.** Web has effectively unlimited unique tokens and never interferes, so it absorbs whatever weight the capped domains do not use — it is the ballast.

Working it through for a 2-trillion-token run: Wikipedia 6 percent (epoch-capped), Math 12 percent (interference-capped, below its epoch cap), Code 12 percent (interference-capped), Books 10 percent (0.5 epoch — comfortable), Web 60 percent (ballast). That sums to 100. Notice the final mixture is *not* sorted by marginal value — math and Wikipedia, the two highest-value-per-token domains, are held down by caps, and web, the lowest-value domain, is the largest, because its value comes from cheap dilution and it is the only domain you can pour in without hitting a wall. This is the exact shape DoReMi tends to discover automatically, which is a good sign the reasoning is sound. If you have the budget, run DoReMi to cross-check these hand-derived weights; if the proxy strongly disagrees, one of your ablation measurements is noisier than you thought.

## A mixture-weight sampler you can actually run

The sampler that turns a weight vector into a token stream is simple, and getting it right matters — a subtly biased sampler silently changes your mixture. The core is a weighted interleave over per-domain shard iterators:

```python
import random
from pathlib import Path

def domain_shard_stream(shard_dir, seed=0):
    """Yield documents from one domain's shards, forever, in shuffled order."""
    rng = random.Random(seed)
    shards = sorted(Path(shard_dir).glob("*.jsonl"))
    while True:
        rng.shuffle(shards)
        for shard in shards:
            with open(shard) as f:
                lines = f.readlines()
            rng.shuffle(lines)
            yield from lines

def mixed_stream(domains, weights, seed=0):
    """Weighted interleave of domain streams.

    domains : dict name -> shard_dir
    weights : dict name -> alpha_d (need not be normalized)
    """
    names   = list(domains)
    streams = {n: domain_shard_stream(domains[n], seed + i)
               for i, n in enumerate(names)}
    total   = sum(weights[n] for n in names)
    probs   = [weights[n] / total for n in names]
    rng     = random.Random(seed)
    while True:
        pick = rng.choices(names, weights=probs, k=1)[0]
        yield next(streams[pick])

# usage
domains = {"web": "data/web", "code": "data/code", "math": "data/math",
           "books": "data/books", "wiki": "data/wiki"}
weights = {"web": 0.60, "code": 0.12, "math": 0.12, "books": 0.10, "wiki": 0.06}
stream  = mixed_stream(domains, weights)
batch   = [next(stream) for _ in range(1024)]
```

Two production notes this toy version omits but you must not. First, in a distributed run, each data-parallel worker must sample from the *same* mixture but with *different* documents — seed the per-worker RNG with the global rank so workers do not draw identical batches, but keep the mixture weights identical so the global batch composition matches `weights`. Second, verify the realized composition: after a few thousand batches, count the actual fraction of tokens from each domain and assert it matches `weights` within a percent. The number of times a shard-ordering bug or a length-vs-document mismatch has quietly turned a "12 percent code" mixture into 20 percent code is not small, and you will only catch it by measuring the output of the sampler, not by trusting its input.

To implement the annealing schedule, you swap the weight vector partway through: run `mixed_stream(domains, stable_weights)` for the first 85 percent of steps, then switch to `mixed_stream(hq_domains, anneal_weights)` for the cooldown, where `hq_domains` points at your curated high-quality shards and `anneal_weights` upweights them heavily.

## Case studies

### 1. DoReMi beats the hand-tuned Pile weights

The original DoReMi experiment is the cleanest evidence that learned beats hand-tuned. The authors took The Pile — whose 22-domain weights had been carefully hand-chosen by the EleutherAI team with perplexity checks — and asked whether a 280M proxy could find better weights. It could. The Group DRO proxy reallocated weight in ways that surprised the curators: it downweighted several domains the humans had left at high weight (because the reference model already fit them well, so they had low excess loss and little left to teach), and it held up domains with persistent excess loss. Transferring the discovered weights to an 8B model trained on The Pile improved average downstream accuracy across the evaluation suite and, strikingly, reached the baseline model's final validation perplexity in roughly 40 percent of the training steps. The lesson that generalizes: human curators systematically over-weight domains that *feel* important (Wikipedia, books) and under-weight the boring-but-still-teaching web, because excess loss is not something intuition tracks well.

### 2. Llama-3's staged mixture and high-quality finish

Llama-3's data recipe is a production-scale demonstration of mid-training plus annealing. The team trained on a large multi-trillion-token corpus with a mixture that was itself tuned via scaling-law experiments on smaller models — they ran many small runs with different mixtures, fit the downstream performance as a function of domain proportions, and extrapolated the optimal mix to the full scale (a scaling-laws approach to mixing that connects directly to [Chinchilla-style compute-optimal reasoning](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling)). Crucially, they did not hold that mixture fixed: the documented recipe adjusts the data mix during training and uses a higher-quality data mixture in the final stage of pretraining, exactly the annealing pattern. The reported effect is that the final-stage high-quality data measurably lifted benchmark performance beyond what the stable-phase mixture alone produced — the last tokens, again, doing outsized work.

### 3. FineWeb turns annealing into a measurement tool

The FineWeb and FineWeb-Edu project made the annealing insight *operational*. To decide whether a filtering change improved their web dataset, they needed a fast, reliable signal — full pretraining runs are too slow to iterate on. Their solution was an "annealing ablation": take a model trained partway on a reference mix, then anneal it (decay the LR to zero) on a fixed budget of the *candidate* data versus the *reference* data, and compare downstream scores. Because the annealing phase magnifies the effect of the data it sees, this gave a strong, cheap signal about data quality that correlated with full-run outcomes. FineWeb-Edu — the education-filtered subset that became a standard high-quality source — was validated substantially through this annealing methodology. It is a lovely example of a training technique (annealing) becoming a measurement technique, and it is why annealing shows up in this series' [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality) post as well.

### 4. mC4 / mT5 and the temperature that made 101 languages viable

mT5 trained a single model on 101 languages from the mC4 corpus, and the only reason the low-resource languages learned anything is temperature sampling at $\alpha = 0.3$. At the natural distribution, the smallest languages would have received on the order of one-thousandth the samples of English and been effectively absent from training. Tempering to 0.3 pulled enough mass onto the tail to give every language real representation, at the cost of English fluency that the authors judged acceptable for a multilingual model. The follow-on lesson the community learned the hard way: 0.3 is aggressive enough that the smallest languages get epoched many times, so mT5's low-resource performance is partly memorization of a small corpus rather than deep competence — which is why later multilingual efforts paired temperature sampling with active collection of more unique low-resource data rather than leaning on temperature alone. Temperature buys you presence, not depth.

### 5. The Pile's honest upsampling multipliers

The Pile deserves credit for being explicit where later datasets were vague. Its documentation lists, per component, both the raw size and the *epochs* — the upsampling multiplier applied. Wikipedia at roughly 3 epochs, Books3 upweighted, PubMed and arXiv given specific multipliers, and Pile-CC (the web component) deliberately held below its raw-proportional share. This transparency is why The Pile remains a teaching artifact: you can read the mixture as a set of explicit hypotheses. The weakness, visible in hindsight and confirmed by DoReMi, is that several of those multipliers were too high — the human instinct to upsample "high-quality" sources like Wikipedia and books ran ahead of what the excess-loss evidence would have supported. The Pile is the "before" picture; DoReMi is the "after."

## Troubleshooting: when the mixture goes wrong

Three failure modes account for most mixing disasters. Each has a distinct symptom, a non-obvious cause, and a specific fix.

### Symptom: a domain that helps in isolation hurts in the mixture

You ablate a domain on its own — train baseline plus a slug of math data — and it improves your math eval by two points. Delighted, you add it to the full mixture at a healthy weight. Downstream math *drops* relative to the baseline mixture. The domain that was a clear win alone is a net loss in the blend.

![Before-after: a math corpus that adds 2.1 points in isolation nets minus 0.4 points once it competes with code in the full mixture](/imgs/blogs/data-mixing-domain-weighting-and-curriculum-6.webp)

**Cause:** interference. Model capacity at a fixed parameter count and token budget is finite, and domains compete for it. Math and code overlap in the representations they demand (symbolic manipulation, structured syntax, precise multi-step reasoning) and also compete for the *same* budget, so adding math at a high weight steals capacity and tokens that code was using — and if code's marginal value was high, the net is negative even though math in isolation looked great. Per-domain gains measured against a baseline are not additive; the cross-terms are real and often large.

**Detection:** never trust solo ablations for domains you suspect overlap. Run the *pairwise* ablation — the two domains together at several weight ratios — and compare the combined downstream score to the sum of the solo gains. If combined is materially below the sum, you have interference. A cheaper early warning: track per-domain eval scores *during* a mixture run; if adding a domain makes an unrelated domain's eval sag, capacity is being reallocated.

**Fix:** treat interfering domains as a joint allocation, as in the worked scenario — find the ratio that maximizes their *combined* contribution rather than giving each its standalone-optimal weight, and accept that the joint optimum gives each less than it "deserves" alone. If both domains are critical and interference is severe, the deeper fix is more capacity (a bigger model) or more unique data so neither domain has to be epoched into the other's budget.

### Symptom: an upsampled domain's eval climbs, then its generalization collapses

You upsample a small high-value domain — Wikipedia, or a niche language — and early in training its held-out eval improves nicely. You keep the high weight. Later, that eval plateaus and then *degrades*, and worse, the model starts reproducing verbatim strings from that domain and its performance on adjacent held-out data (that it should generalize to) gets worse.

![Matrix of marginal and cumulative gain by epoch multiplier, peaking around four epochs then turning negative into overfitting and memorization](/imgs/blogs/data-mixing-domain-weighting-and-curriculum-7.webp)

**Cause:** epoching past the domain's unique content. A small corpus has limited unique information; once the model has seen it a few times, additional epochs teach nothing new and instead drive memorization of the specific token sequences. The figure shows the shape: marginal gain per epoch decays (1.8, 0.9, 0.3 points for the first few doublings) and then turns negative (−0.2, −1.1) as memorization sets in; cumulative gain peaks around four epochs and falls after. The exact peak depends on the domain and model size, but the shape is universal.

**Detection:** compute effective epochs for every domain (the $(\alpha_d \cdot T)/N_d$ number from section 1) and flag any above 3-4. Watch for the classic memorization tell: rising verbatim-reproduction rate on that domain's training strings, and a growing gap between training-slice loss (falling) and held-out-slice loss (flat or rising). This connects to [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale) — near-duplicates inflate a domain's *apparent* unique size, so you epoch it more than you think.

**Fix:** cap effective epochs (3-4 is a common ceiling), which for a fixed budget means capping the domain's weight. If the domain genuinely needs more representation to hit target competence, the answer is more *unique* data, not more repetitions — collect or generate additional unique content for that domain rather than turning up the weight. Synthetic data can help here specifically because it adds unique tokens instead of re-showing the same ones.

### Symptom: benchmark scores jump suspiciously during the annealing phase

Your annealing phase produces a beautiful bump on a target benchmark — bigger than the stable phase ever gave. It looks like the high-quality cooldown working exactly as designed. Then the model does poorly on a fresh, held-out variant of that same benchmark, or a competitor's private eval, and the gap is embarrassing.

**Cause:** anneal-set contamination. The annealing set is small, hand-curated, and heavily weighted, which makes it the *most dangerous* place in the entire pipeline for eval leakage. A single contaminated document — a benchmark question that slipped into your "high-quality curated" set, or synthetic data generated by a model that had itself memorized the benchmark — gets epoched many times at the exact moment the model is most receptive (low LR, precise updates). The result is memorized answers masquerading as learned capability. Because the annealing set is curated by hand and often includes instruction-like data close in form to benchmarks, contamination is *more* likely here than in the bulk web, not less.

**Detection:** run your strictest decontamination — n-gram overlap and, ideally, embedding-based near-duplicate detection — against the annealing set specifically, treating it with far more suspicion than the bulk corpus. Compare performance on public benchmarks (which could be contaminated) against private or freshly-constructed held-out variants; a large public-versus-private gap that *appears or widens after annealing* is the signature. This is covered end to end in [decontamination and benchmark leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage), and the annealing set is the single most important input to decontaminate.

**Fix:** decontaminate the annealing set to a higher standard than everything else, quarantine any synthetic data whose generator's training set you cannot vouch for, and keep a truly-held-out eval that never touches any curation loop so you always have an honest number. If a benchmark jump appears only during annealing and only on public evals, assume contamination until proven otherwise.

## When to invest in mixing, and when to just copy a recipe

**Invest real effort in the mixture when:**

- You are training a model you will iterate on — the proxy runs and ablations amortize across every future run.
- You have many domains or languages (more than a handful of hand-tunable weights), where DoReMi or temperature sampling replaces guesswork that does not scale.
- You have specific capability targets (strong code, strong math, strong multilingual) that pull against each other — interference is real and worth measuring.
- You can afford an annealing phase, which is nearly free relative to the full run and reliably lifts final quality.
- Your domains span very different unique-pool sizes, so effective-epoch math genuinely constrains the weights.

**Just copy a known-good recipe when:**

- You are running a one-off experiment or a reproduction where the mixture is not the variable you are studying.
- Your data is dominated by one domain (mostly web, small everything-else) so the mixture has little room to matter.
- You lack the budget for even a small proxy run — in which case a published web-heavy mixture with a modest annealing phase is a strong default that is hard to beat blindly.
- Your token budget is small relative to your unique data (well under one epoch of everything), where epoch caps and interference are not yet binding.

The meta-point: mixing is where a clean corpus becomes a *capable* model, and it is the highest-leverage lever left after selection and cleaning are done. The natural distribution of your data is almost never the right one; the manual mixtures of the last generation were right in direction and wrong in magnitude; and the learned, staged, annealed mixtures of the current generation win because they replace intuition with measurement at every step — proxy-tuned weights, tempered language sampling, capped epochs, and a high-quality finish. Get the recipe right and the same ingredients that made a mediocre model make a great one.

For the compute-optimal side of how much data to use at all, pair this with [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) and [data-constrained scaling laws](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) — the latter is directly relevant to the epoching-limits discussion, because it quantifies how many times you can profitably repeat data before returns vanish.

## Further reading

- Xie et al., "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining" (2023) — the proxy-and-transfer method and Group DRO objective.
- Gao et al., "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" (2020) — the explicit per-source weights and epoch multipliers.
- Xue et al., "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer" (2021) — temperature sampling at scale across 101 languages.
- Penedo et al., "The FineWeb Datasets" (2024) — the annealing ablation as a data-quality measurement tool.
- Sibling posts: [data selection and pruning](/blog/machine-learning/training-data/data-selection-and-pruning), [measuring data quality](/blog/machine-learning/training-data/measuring-data-quality), [deduplication at scale](/blog/machine-learning/training-data/deduplication-at-scale), and [decontamination and benchmark leakage](/blog/machine-learning/training-data/decontamination-and-benchmark-leakage).
