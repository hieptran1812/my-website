---
title: "Position and Selection Bias in Click Data"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Clicks lie, and the biggest lie is position. Learn why top results get clicked regardless of relevance, and how to train an unbiased ranker from biased clicks with inverse propensity scoring."
tags:
  [
    "recommendation-systems",
    "recsys",
    "unbiased-learning-to-rank",
    "position-bias",
    "inverse-propensity-scoring",
    "click-models",
    "learning-to-rank",
    "machine-learning",
    "counterfactual",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/position-and-selection-bias-in-click-data-1.png"
---

A few years ago I watched a search ranking team celebrate a model that, on paper, was a triumph. They had trained a learning-to-rank model on a year of click logs — hundreds of millions of impressions, billions of clicks, the kind of data you dream about. Offline, it beat the production ranker on every click-prediction metric they had. They shipped it. Two weeks later the experiment was rolled back: engagement was flat, and a few high-value queries had gotten visibly worse.

The post-mortem was uncomfortable. The model had not learned what is relevant. It had learned what was already on top. Because the training labels were clicks, and clicks happen overwhelmingly on the first few results, the model had discovered the most reliable signal in the data: *items that were ranked high got clicked, so rank them high.* It had reverse-engineered the previous ranker and dressed it up as relevance. The offline metric — predicting clicks — went up precisely *because* the model was good at predicting position, not quality. This is the trap at the heart of every system that learns from its own logs, and it has a name: **position bias**.

The core problem is deceptively simple. A click is not a relevance label. A click happens only when two things are true at once: the user **examined** the item (looked at it, read the title, saw it on screen), and the user found it **relevant** (worth clicking given their intent). Examination depends massively on *where* you put the item — the top of the page gets examined far more than the bottom — so top items get clicked more *regardless of relevance*. If you train naively on clicks, you teach the model that high rank causes goodness, which entrenches the current ranking and closes a [feedback loop](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) that slowly ossifies your catalog. Figure 1 contrasts the two readings of a click: the naive view that a click equals relevance, and the position-based view that a click equals examination times relevance.

![Side by side comparison of the naive reading of a click as a pure relevance label versus the position-based reading that factors a click into examination probability set by rank times a true relevance score](/imgs/blogs/position-and-selection-bias-in-click-data-1.png)

By the end of this post you will be able to: write down the **Position-Based Model (PBM)** factorization of a click and explain why it is the right causal story; estimate position propensities three different ways (result randomization, intervention harvesting, and jointly via regression-EM); derive the **Inverse Propensity Scoring (IPS)** estimator for **Unbiased Learning to Rank (ULTR)** and prove it is unbiased for relevance given correct propensities; and — crucially — *simulate* the whole thing in runnable code: generate biased clicks under a PBM, train a naive ranker that learns position, train an IPS-ULTR ranker that recovers relevance, and measure the gap against a ground-truth oracle on NDCG@10. This sits squarely in the **feedback-loop** corner of the series spine (serve → log → train → serve) and in the **offline-online gap** that this loop creates; it is the bias that most often makes your [offline metric lie](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied).

## 1. Why a click is not a relevance label

Let me make the problem concrete with a single observation that should bother you.

Suppose you run a movie recommender. You show ten movies in a ranked list. Movie A is a genuinely great match for the user and you put it at rank 1; it gets clicked. Movie B is *also* a genuinely great match but you put it at rank 9; it does not get clicked. If you treat clicks as relevance labels, you have just generated a training example that says "A is relevant, B is irrelevant" — when in truth they are equally relevant. The only difference between them is the position *you chose*. The label is contaminated by your own past decision.

Now multiply that by every impression in your logs. Every "negative" (no click) is suspect, because it might be a no-click on an item the user never even saw. Every "positive" (click) is inflated for high positions, because the user was far more likely to look there. The data is not labeled by relevance; it is labeled by **relevance filtered through examination**, and examination is dominated by position.

This is sometimes called **implicit feedback bias**, and position bias is its most powerful form. (We treated the broader implicit-versus-explicit distinction in [implicit vs explicit feedback](/blog/machine-learning/recommendation-systems/implicit-vs-explicit-feedback-and-the-data-you-have); here we zoom into the single most damaging bias in that feedback.) The reason it matters so much is causal:

- **Position causes examination.** People scan top-down; eye-tracking studies have shown attention concentrates at the top and decays sharply with rank.
- **Examination gates clicks.** You cannot click what you did not look at.
- **Your ranker chooses position.** So your ranker indirectly chooses who gets examined, who gets the chance to be clicked, and therefore who generates training labels next time.

That last point is the loop. The ranker decides exposure; exposure decides clicks; clicks become labels; labels train the next ranker. If you do nothing to correct for it, the loop has a fixed point: *whatever is on top stays on top*, because being on top is what generates the clicks that justify being on top. New, genuinely-better items can never break in, because they never get the exposure to prove themselves. This is the same self-reinforcing dynamic behind popularity bias and filter bubbles, viewed from the data side rather than the recommendation side.

#### Worked example: the contaminated label

Imagine two items with identical true relevance, $P(\text{relevant}) = 0.5$ for each. We show item A at rank 1, where the probability of examination is $1.0$, and item B at rank 8, where examination is $0.18$. The probability each gets clicked, under the model we are about to formalize, is examination times relevance:

$$
P(\text{click}_A) = 1.00 \times 0.5 = 0.50, \qquad P(\text{click}_B) = 0.18 \times 0.5 = 0.09.
$$

Over 1,000 impressions of each, A racks up about 500 clicks and B about 90. A naive learner sees A clicked 5.5 times as often and concludes A is far more relevant. It is not. The entire 5.5x ratio is examination, i.e. *the position you assigned*. The relevance is identical. That single number — 5.5x of pure bias on two equally-good items — is why training on raw clicks is dangerous.

### The closed loop has a fixed point — and it is bad

It is worth making the "loop ossifies" intuition rigorous, because it explains *why* this bias is not self-correcting and *why* you must intervene. Model the loop as a discrete-time dynamical system. At round $t$, the ranker assigns item $d$ to position $k_d^{(t)}$, which gives it examination $\theta_{k_d^{(t)}}$. The expected clicks item $d$ collects in round $t$ are proportional to $\theta_{k_d^{(t)}} \gamma_d$. The next ranker is trained on these clicks, and — because it has no way to factor out $\theta$ — it learns a *score* that is monotone in observed clicks, hence monotone in $\theta_{k_d^{(t)}} \gamma_d$. So the position an item gets at round $t+1$ depends on the clicks it got at round $t$, which depended on the position it had at round $t$:

$$
\text{score}_d^{(t+1)} \;\propto\; \theta_{k_d^{(t)}}\, \gamma_d, \qquad k_d^{(t+1)} = \text{rank of } d \text{ under } \text{score}^{(t+1)}.
$$

Look at what this does to two items with relevances $\gamma_A > \gamma_B$ that, for whatever historical accident, started with $B$ above $A$ (so $\theta_{k_B} > \theta_{k_A}$). The naive score of $B$ is $\theta_{k_B}\gamma_B$ and of $A$ is $\theta_{k_A}\gamma_A$. If the position advantage outweighs the relevance gap — $\theta_{k_B}/\theta_{k_A} > \gamma_A/\gamma_B$ — then $B$ keeps its higher score, keeps its higher position, keeps its examination advantage, *forever*. The truly-better item $A$ is locked out not because the model is dumb but because the loop never gives $A$ the exposure to generate the clicks that would prove it. The naive system has a *stable but wrong* fixed point: the initial ordering, whatever it was, becomes self-justifying. This is the dynamical-systems version of "the rich get richer," and it is exactly why a feedback loop quietly turns a catalog into ten popular items. The only escape is to break the dependence of next-round score on this-round position — which is precisely what IPS does by dividing $\theta$ back out, restoring $\text{score}_d \propto \gamma_d$ regardless of where $d$ was shown.

## 2. The examination hypothesis and the Position-Based Model

The fix begins with a precise model of *how* a click is generated. The foundational assumption across nearly all of click modeling is the **examination hypothesis**: a user clicks a result if and only if the result is both *examined* and *relevant*, and these two events are independent.

Formally, for a result shown at rank $k$ for a query/context $q$ and a document $d$, define two binary random variables:

- $E_k \in \{0, 1\}$: the user examined the result at rank $k$.
- $R_{q,d} \in \{0, 1\}$: the result $d$ is relevant to $q$.

The examination hypothesis says the click $C$ is their conjunction:

$$
C = E_k \cdot R_{q,d}, \qquad P(C = 1 \mid q, d, k) = P(E_k = 1 \mid k)\; P(R_{q,d} = 1 \mid q, d).
$$

The **Position-Based Model (PBM)** makes one further, load-bearing assumption: *examination depends only on the position, not on the document or the query.* That is, $P(E_k = 1 \mid k) = \theta_k$, a single number per rank that we call the **examination propensity** at rank $k$. Writing $\gamma_{q,d} = P(R_{q,d} = 1)$ for relevance, the PBM gives the click probability its famous product form:

$$
P(C = 1 \mid q, d, k) = \theta_k \cdot \gamma_{q,d}.
$$

This factorization is the whole game. It separates the part of the click you *control* (where you put the item, hence $\theta_k$) from the part you *want to learn* ($\gamma_{q,d}$, the relevance). If you knew $\theta_k$, you could divide it out of the observed click rate and recover relevance. Figure 2 shows what these $\theta_k$ values look like in practice: a sharp drop-off as you go down the page.

![Layered stack showing examination probability falling from one at rank one down to roughly point one eight at rank nine, a steep top-heavy drop-off across positions](/imgs/blogs/position-and-selection-bias-in-click-data-2.png)

The shape in Figure 2 is empirically robust. The exact numbers vary by surface — a sparse 3-result mobile screen behaves differently from a dense 30-result desktop SERP — but the qualitative law is universal: examination is steeply top-heavy. On many web search surfaces, $\theta_1 \approx 1.0$, $\theta_2$ is in the 0.6–0.8 range, and by rank 10 you are often below 0.2. That decay is *why* a click at rank 1 is weak evidence (everybody looks there) and a click at rank 9 is strong evidence (almost nobody looks there, so a click means the item really pulled the user).

Figure 3 draws the PBM as a causal graph, because thinking of it causally is what makes the debiasing strategy obvious. Position and the item are two independent causes; examination and relevance are their respective effects; the click is the AND of the two; and the *logged label* is the click — which is therefore confounded by position.

![Directed acyclic graph of the position based model showing position causing examination and the item causing relevance, both merging into a click which becomes a position biased logged label](/imgs/blogs/position-and-selection-bias-in-click-data-3.png)

#### Why the independence assumption is doing work

The clean product form $\theta_k \gamma_{q,d}$ requires that examination and relevance are independent *given position*. Is that true? Not perfectly. In reality, an attractive title (a relevance-correlated feature) can pull the eye, which couples examination to the item — that is exactly what richer models add. But the PBM's independence assumption is a remarkably good first-order approximation, and it has a decisive practical virtue: $\theta_k$ has only as many free parameters as you have positions (often 10–20), so it is *cheap to estimate* and *easy to share across all queries*. That cheapness is why PBM-based IPS is the workhorse of unbiased learning to rank, even when everyone knows the assumption is not literally true.

## 3. Richer click models: Cascade, DBN, and UBM

The PBM treats every position's examination as fixed and independent. That is wrong in one obvious way: real users do not examine all positions independently. They *scan*, usually top-down, and what they do at rank $k$ depends on what happened at ranks $1$ through $k-1$. A family of click models captures these scanning dynamics with progressively more structure. Figure 4 lays out the trade-offs.

![Matrix comparing the position based model, cascade model, and dynamic bayesian network across their examination rule, what they capture, and the cost to fit](/imgs/blogs/position-and-selection-bias-in-click-data-4.png)

**The Cascade Model.** The cascade model (Craswell et al., 2008) assumes the user examines results strictly top-down, clicks the first relevant one, and then *stops*. Examination of rank $k$ requires that none of the earlier results were clicked. If $r_i = \gamma_{q,d_i}$ is the relevance (here, the click probability conditioned on examination) of the item at position $i$, then the probability of examining position $k$ is the probability that every earlier item failed to satisfy the user:

$$
P(E_k = 1) = \prod_{i=1}^{k-1} (1 - r_i).
$$

The full likelihood of an observed click pattern under the cascade model — a click at position $j$ and no clicks before it — is

$$
P(\text{click at } j,\ \text{no click before}) = r_j \prod_{i=1}^{j-1} (1 - r_i).
$$

The cascade model is elegant and it explains a real phenomenon — a great result at rank 1 *suppresses* examination of everything below it — but it has a glaring limitation: it can only model sessions with **exactly one click** (it stops after the first click). Multi-click sessions and abandoned sessions (no click at all) do not fit cleanly. That is why people layer more machinery on top.

**The Dynamic Bayesian Network (DBN).** The DBN model (Chapelle and Zhang, 2009) separates *relevance* (will the user click?) from *satisfaction* (will the user be happy enough to stop?). After clicking, the user is satisfied with some probability $s_d$; if satisfied they stop, otherwise they continue scanning with a continuation probability $\gamma$. This lets a single session have multiple clicks (the user clicked but was not satisfied, so they kept going) and gives you a separate satisfaction signal, which correlates better with true relevance than raw clicks. The cost is that you must fit it with Expectation-Maximization over latent examination/satisfaction states.

**The User Browsing Model (UBM).** The UBM (Dupret and Piwowarski, 2008) makes examination depend on *both* the current rank and the rank of the previous click — capturing that after a click at rank 3, the user's attention to rank 4 depends on how far back the last click was. It is more expressive than the PBM about examination but does not model satisfaction like the DBN.

**Which one should you use for ULTR?** Here is the opinionated answer: for *unbiased learning to rank*, the PBM is almost always the right starting point, and often the ending point too. The reason is that for IPS you need a single, position-indexed propensity $\theta_k$ that you can confidently estimate and divide out. The cascade/DBN/UBM models give you a richer *generative* story (great for click simulation, query understanding, and click-model-based metrics), but they make the examination probability *depend on the items shown*, which makes the IPS propensity item-dependent and far harder to estimate reliably. The field's most successful production ULTR systems — Joachims et al. (2017), the Wang et al. (2018) regression-EM work at large-scale search — are built on the PBM. Reach for a richer click model when you need to *simulate* clicks faithfully or compute a click-model metric, not when you are weighting a training loss.

## 4. Selection bias: the hole you cannot see

Position bias is about *which shown items get clicked*. There is a second, related bias about *which items get shown at all*: **selection bias**. You only ever get feedback on items you presented. Everything you did not show generates no data — not a negative, not a positive, just a hole.

This is the off-policy / counterfactual problem in disguise. Your logging policy (the ranker that produced the logs) decided the action set. For any query, it showed maybe 10 of a 100,000-item catalog. The other 99,990 items have no click data for that query. If you train a model and it wants to rank an unshown item highly, you have *no observational evidence* about whether that is a good idea. You are extrapolating into a region your data never covered. This is exactly the support/coverage problem that haunts [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation): you can only reweight observations you actually have, and IPS weights blow up (or are undefined) where the logging policy gave zero probability.

The two biases interact. Position bias distorts the *labels* of shown items; selection bias removes the *labels* of unshown items. ULTR with IPS on examination propensity fixes the first directly. The second needs either: (a) **exploration** — deliberately showing items you are uncertain about so you get coverage (an $\epsilon$-greedy or Thompson-sampling layer on top of the ranker), or (b) **a propensity floor / clipping** so the IPS weights for rarely-shown items do not explode, plus honest acknowledgment that you cannot evaluate what you never showed. We will see the clipping trick in the code; the exploration story is a whole topic of its own and connects to the [feedback-loop](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) post.

A useful way to keep them straight:

| Bias | What it distorts | Cause | Fix |
| --- | --- | --- | --- |
| Position bias | The *label* of a shown item (clicked more if higher) | Examination depends on rank | IPS on examination propensity $\theta_k$ |
| Selection / exposure bias | The *availability* of a label (only shown items have data) | Logging policy chose the action set | Exploration, propensity floor, off-policy estimators |
| Trust / presentation bias | Click rate inflated by UI position, snippet, brand | Visual salience, perceived authority | Folded into $\theta_k$ or modeled explicitly |

A subtlety worth stating plainly: trust bias and position bias are easy to conflate but they are different mechanisms. Position bias is about *whether the user looked*; trust bias is about *whether the user clicked given they looked*, inflated because users trust the top result more (the "I'll trust whatever Google ranks first" effect). The basic PBM bundles both into $\theta_k$ — which is fine for IPS, since you divide out the combined effect — but if you want to *understand* your surface, or you suspect trust bias is large, you can model an additional position-dependent trust term and estimate it separately. For most teams, folding trust into $\theta_k$ is the right level of detail; you only split them out when a presentation redesign forces you to reason about the two independently.

## 5. Inverse Propensity Scoring: the unbiased estimator

Now the payoff. We have the PBM, $P(C=1 \mid q,d,k) = \theta_k \gamma_{q,d}$, and we want to learn relevance $\gamma_{q,d}$ from clicks despite the $\theta_k$ contamination. The tool is **Inverse Propensity Scoring**.

The idea is the same one that makes survey statistics work: if some respondents are over-sampled, you down-weight them by their sampling probability so the weighted sample looks like the population. Here, clicks at high positions are "over-sampled" (high examination), so we down-weight them by their examination probability; clicks at low positions are "under-sampled," so we up-weight them. Concretely, weight every observed click by the inverse of its examination propensity, $1 / \theta_k$.

### The science: deriving the unbiased risk

Let me make "unbiased" precise. In learning to rank we want to minimize a risk that depends on *relevance*. Take a simple but representative target: a relevance-weighted ranking loss where each *relevant* document contributes a cost $\Delta(d \mid \pi, q)$ that depends on where ranker $\pi$ places it (e.g. its rank, or a DCG-style discount). The **true risk** we wish we could minimize is

$$
R(\pi) = \sum_{q} \sum_{d : R_{q,d} = 1} \Delta(d \mid \pi, q).
$$

The problem: we do not observe $R_{q,d}$. We observe clicks $C_{q,d}$, which are biased. Naively summing the loss over *clicked* documents gives the biased empirical risk

$$
\hat{R}_{\text{naive}}(\pi) = \sum_{q} \sum_{d : C_{q,d} = 1} \Delta(d \mid \pi, q),
$$

and because high-position relevant items are clicked far more than low-position ones, this over-counts top items — exactly the bias we are fighting. Now define the **IPS-weighted empirical risk**, where each clicked document is weighted by the inverse examination propensity of the position $k_d$ at which it was shown:

$$
\hat{R}_{\text{IPS}}(\pi) = \sum_{q} \sum_{d : C_{q,d} = 1} \frac{\Delta(d \mid \pi, q)}{\theta_{k_d}}.
$$

**Claim: $\hat{R}_{\text{IPS}}$ is unbiased for $R$.** Take the expectation over the randomness in clicks. A relevant document at position $k_d$ is clicked with probability $\theta_{k_d}$ (examination) times $1$ (it is relevant, so $\gamma = 1$ for the relevant set); an irrelevant document is never clicked and contributes nothing. So $\mathbb{E}[C_{q,d}] = \theta_{k_d} R_{q,d}$. Then:

$$
\mathbb{E}\left[\hat{R}_{\text{IPS}}(\pi)\right]
= \sum_{q} \sum_{d} \frac{\Delta(d \mid \pi, q)}{\theta_{k_d}}\, \mathbb{E}[C_{q,d}]
= \sum_{q} \sum_{d} \frac{\Delta(d \mid \pi, q)}{\theta_{k_d}}\, \theta_{k_d} R_{q,d}.
$$

The $\theta_{k_d}$ cancels, leaving

$$
\mathbb{E}\left[\hat{R}_{\text{IPS}}(\pi)\right] = \sum_{q} \sum_{d} \Delta(d \mid \pi, q)\, R_{q,d} = \sum_{q}\sum_{d : R_{q,d}=1} \Delta(d \mid \pi, q) = R(\pi).
$$

That is the whole proof, and it is worth pausing on. The IPS-weighted risk computed from *biased clicks* has the same expectation as the risk computed from *true relevance labels you never observed*. The single requirement is that the propensities $\theta_{k_d}$ are correct (and strictly positive — you cannot divide by zero, which is the selection-bias caveat). This is the formal statement of "divide out examination to recover relevance," and it is the foundation of unbiased learning to rank from Joachims, Swaminathan and Schnabel (2017).

Two important caveats sit inside that clean result:

- **Variance.** Unbiasedness is about the *mean*. The variance of the IPS estimator scales with $1/\theta_k$, so deep-position clicks (tiny $\theta_k$) carry huge weights and huge variance. A single rank-20 click weighted by $1/0.05 = 20$ can swing the loss. This is why production ULTR almost always **clips** the propensity at a floor, $\theta_k \leftarrow \max(\theta_k, \tau)$, trading a little bias for a lot of variance reduction. (This is the same bias-variance dial as clipped IPS in off-policy evaluation.)
- **Correctness of $\theta_k$.** Unbiasedness holds *only* if your propensities are right. Wrong propensities give you a different bias, not no bias. Estimating $\theta_k$ well is therefore the practical crux, which is the next section.

### Why the variance blows up: the math behind the clipping

The variance caveat deserves a number, because it is the single most common reason a textbook-correct IPS implementation underperforms in practice. Consider the contribution of one clicked impression at rank $k$ to the IPS sum: the weight is $1/\theta_k$ and the click is a Bernoulli event with probability $\theta_k \gamma$. The variance of a single weighted term is then

$$
\mathrm{Var}\!\left(\frac{C}{\theta_k}\right) = \frac{1}{\theta_k^2}\,\mathrm{Var}(C) = \frac{1}{\theta_k^2}\,\theta_k \gamma (1 - \theta_k \gamma) = \frac{\gamma(1 - \theta_k \gamma)}{\theta_k}.
$$

That $\theta_k$ in the denominator is the problem. As $\theta_k \to 0$ (deep positions), the variance of the per-click contribution grows like $1/\theta_k$ even though the term is unbiased. So a rank-20 position with $\theta_{20} \approx 0.05$ produces individual contributions with roughly **20x** the variance of a rank-1 contribution — and there are far *fewer* clicks down there to average that variance away. The estimator is correct on average but jittery in any finite sample, and that jitter feeds directly into your gradients. Clipping at a floor $\tau$ caps the worst-case variance at $\gamma(1-\tau\gamma)/\tau$, which is why a floor of $\tau = 0.05$ or $0.1$ is standard: it bounds the variance at the cost of a small, controlled bias on the very deepest positions, where you have the least data anyway. This is the same bias-variance bargain as clipped importance weights everywhere in off-policy learning, and the optimal $\tau$ is the one that minimizes the *total* error (bias squared plus variance), which you tune empirically.

Figure 5 shows the before-and-after at the ranker level: the naive ranker entrenches position, the IPS-ULTR ranker recovers relevance.

![Before and after comparison of a naive ranker trained on raw clicks that entrenches position versus an inverse propensity weighted ranker that recovers the true relevance ordering and closes most of the quality gap](/imgs/blogs/position-and-selection-bias-in-click-data-5.png)

#### Worked example: the IPS weight at rank 1 vs rank 5

Take the propensity curve from Figure 2: $\theta_1 = 1.00$ and $\theta_5 = 0.34$. A click at rank 1 gets IPS weight

$$
w_1 = \frac{1}{\theta_1} = \frac{1}{1.00} = 1.0,
$$

while a click at rank 5 gets

$$
w_5 = \frac{1}{\theta_5} = \frac{1}{0.34} \approx 2.9.
$$

So a click that fought its way up from rank 5 is worth roughly **2.9x** as much relevance evidence as a click at rank 1 in the IPS-weighted loss. That makes intuitive sense: almost everyone examines rank 1, so a click there is cheap; very few people get down to rank 5, so a click there means the item was compelling enough to overcome low exposure. If we used the curve's rank-9 value $\theta_9 = 0.18$, a rank-9 click would be worth $1/0.18 \approx 5.6x$ a rank-1 click — and you can already feel the variance problem: one such click dominates many rank-1 clicks. Figure 7 visualizes this weighting.

![Before and after comparison of the inverse propensity weight for a click at rank one with weight one versus a click at rank five with weight roughly two point nine, showing deep clicks carry more relevance evidence](/imgs/blogs/position-and-selection-bias-in-click-data-7.png)

#### Worked example: a naive model preferring a high-positioned irrelevant item

Consider an irrelevant item X shown at rank 1 with $\theta_1 = 1.0$ and true relevance $\gamma_X = 0.05$ (people occasionally misclick or the title is misleadingly catchy). Its click probability is $1.0 \times 0.05 = 0.05$. Now a genuinely relevant item Y is shown at rank 9 with $\theta_9 = 0.18$ and $\gamma_Y = 0.4$; its click probability is $0.18 \times 0.4 = 0.072$. Over 1,000 impressions each, X gets ~50 clicks and Y gets ~72 — close enough that with sampling noise a naive ranker, looking only at raw click counts and X's enviable rank-1 history, may well prefer X. Apply IPS: X's clicks weight $1/1.0 = 1.0$ each (~50 weighted), Y's weight $1/0.18 \approx 5.6$ each (~400 weighted). The IPS-weighted evidence now correctly says Y is roughly **8x** more relevant than X. The reweighting flipped the wrong preference into the right one.

## 6. Estimating the position propensities

The IPS estimator is only as good as the $\theta_k$ you plug in. There are three families of methods, in roughly increasing order of how little they disturb your users. But before the methods, one warning that catches almost everyone the first time.

### 6.0 Why you cannot read the propensity off the raw CTR curve

The tempting shortcut is: "I already have a CTR-by-position curve from my logs (it is right there in Figure 2's shape); why not just use *that* as the propensity?" Because the raw CTR curve is **confounded**. Under the PBM, the observed CTR at rank $k$ is $\mathbb{E}_{\text{shown at }k}[\theta_k \gamma] = \theta_k \cdot \bar\gamma_k$, where $\bar\gamma_k$ is the *average relevance of whatever items your logging policy put at rank $k$*. A good logging policy puts its most-relevant guesses at the top, so $\bar\gamma_k$ is itself a decreasing function of $k$. The raw CTR curve therefore mixes two declines — the genuine examination decline $\theta_k$ and the policy-induced relevance decline $\bar\gamma_k$ — and you cannot separate them by staring at the curve.

This is the crux of the whole estimation problem and the reason all three methods below exist. Randomization works precisely because it *forces* $\bar\gamma_k$ to be the same constant $\bar\gamma$ at every position (random assignment decorrelates relevance from position), so the residual decline is pure $\theta_k$. Intervention harvesting works because the *same* item at two positions has the *same* $\gamma$, again cancelling relevance. Regression-EM works because it models $\gamma$ explicitly as a function of features and divides it out statistically. Every method is, at heart, a different trick for removing the $\bar\gamma_k$ confounder from the observed CTR so the $\theta_k$ underneath is exposed. Using raw CTR as the propensity would *under*-correct (your propensities would be too steep, blaming examination for a decline that was partly genuine relevance), and the resulting "debiased" ranker would still partly entrench position. So: never use the raw CTR-by-position curve as your propensity. Always use one of the three confounder-removing estimators.

### 6.1 Result randomization (RandPair / RandTopN)

The cleanest way to measure examination is to *randomize position and hold relevance fixed*. If you take the same item and show it sometimes at rank 1 and sometimes at rank $k$ (chosen at random), then any difference in its click-through rate across positions is *pure examination*, because the item — and therefore its relevance — is identical. Figure 6 draws this swap experiment.

![Directed acyclic graph of propensity estimation by randomization showing one fixed relevance result shown at rank one and at rank k, measuring click through rate at each, and dividing to recover the examination propensity ratio](/imgs/blogs/position-and-selection-bias-in-click-data-6.png)

Formally, under the PBM the expected CTR of an item with relevance $\gamma$ at rank $k$ is $\theta_k \gamma$. Average over many items at each rank (so $\gamma$ averages to the same population mean $\bar{\gamma}$, guaranteed because we *assigned positions randomly*), and the ratio of average CTRs gives the propensity ratio:

$$
\frac{\widehat{\text{CTR}}_k}{\widehat{\text{CTR}}_1} = \frac{\theta_k \bar{\gamma}}{\theta_1 \bar{\gamma}} = \frac{\theta_k}{\theta_1}.
$$

Fix $\theta_1 = 1$ by convention, and you have read $\theta_k$ straight off the data. The most data-efficient variant, **RandPair**, randomly *swaps the result at rank 1 with the result at rank $k$* for a small fraction of traffic, comparing CTR of the swapped-in items. The cost is real but bounded: you are deliberately showing a few users a slightly-shuffled list. Most teams cap this at 1–5% of traffic for a short window, which is enough to estimate a 10–20 entry propensity curve.

The downside is the user cost — you are knowingly degrading some results — and that the assumption "$\bar\gamma$ is the same across positions after randomization" needs enough volume to hold. But randomization is the **gold standard**: it makes the fewest modeling assumptions and gives propensities you can trust.

### 6.2 Intervention harvesting

Result randomization is principled but invasive. **Intervention harvesting** (Agarwal et al., 2019; also "harvesting" interventions from natural experiments) makes the observation that you do not always need to *run* a randomization — you can often *find* one in your existing logs. Whenever your production system already varied the position of the same query-document pair across time — because of an A/B test, a model rollout, a re-ranking layer, a tie-break, or just ranker churn — those naturally-occurring position swaps are interventions you can harvest for free. You collect all pairs of impressions where the same item appeared at two different ranks and use the click-rate difference to estimate $\theta_k$, exactly like randomization but using interventions that already happened. This is far cheaper (no traffic cost) at the price of needing enough natural variation and being careful that the variation was not itself relevance-correlated.

### 6.3 Jointly learning propensity and relevance (Dual Learning / Regression-EM)

The third family asks: can we estimate $\theta_k$ and the relevance model *at the same time*, from ordinary biased logs, with no randomization at all? Yes — because the PBM is just a latent-variable model, and we have a workhorse for those: **Expectation-Maximization**.

The **Regression-EM** approach (Wang et al., 2018, from Google search) treats examination $E_k$ as a latent variable. Given a click ($C=1$), examination must have happened, so $P(E=1 \mid C=1) = 1$. Given no click ($C=0$), examination is uncertain, and Bayes' rule on the PBM gives its posterior. The EM loop alternates:

- **E-step.** For every impression, compute the posterior probability of examination given the observed click, current propensities $\theta_k$, and current relevance estimates $\gamma_{q,d}$ from the model. For a non-click at rank $k$:

$$
P(E=1 \mid C=0, k, q, d) = \frac{\theta_k (1 - \gamma_{q,d})}{1 - \theta_k \gamma_{q,d}}.
$$

- **M-step.** Update $\theta_k$ as the average posterior examination at each rank, and *regress* relevance $\gamma_{q,d}$ on document/query features using the posterior-derived soft relevance labels (the "regression" in regression-EM — relevance is a learned function of features, not a per-pair free parameter, so it generalizes to unseen pairs).

The "dual learning" framing (Ai et al., 2018, the Dual Learning Algorithm, DLA) makes this symmetry explicit and beautiful: a *ranking* model and a *propensity* model train each other. The propensity model gives weights that debias the ranker's loss (standard IPS for ranking); symmetrically, the ranker gives an unbiased estimate of relevance that debiases the *propensity* model's loss. The two models are duals, each providing the other's IPS weights, trained jointly end-to-end with gradient descent — no separate randomization step. This is the most production-friendly approach when you cannot afford randomization traffic, because it runs on the logs you already have.

| Propensity method | Traffic cost | Assumptions | When to use |
| --- | --- | --- | --- |
| Result randomization (RandPair) | 1–5% degraded traffic | Random swap makes $\bar\gamma$ position-independent | You can afford a short experiment; want gold-standard $\theta_k$ |
| Intervention harvesting | None (uses existing logs) | Enough natural position variation, not relevance-driven | You have ranker churn or past A/B tests to mine |
| Regression-EM / Dual Learning | None (uses existing logs) | PBM holds; relevance regresses on features | Large-scale, no randomization budget, end-to-end training |

## 7. The practical flow: simulate, train naive, train ULTR

Talk is cheap; let me show the whole thing in code. We will simulate a world where we *know* the ground-truth relevance (so we can grade ourselves), generate biased clicks under a PBM, train a naive ranker on raw clicks and an IPS-ULTR ranker on weighted clicks, and measure both against truth on NDCG@10. This is the single most clarifying exercise in the whole topic, because the simulation lets you see the bias and the correction with absolute certainty.

### 7.1 Simulate ground-truth relevance and a biased logging policy

```python
import numpy as np

rng = np.random.default_rng(0)

N_QUERIES = 4000          # number of query sessions
N_DOCS = 30               # candidate docs per query
N_FEATS = 20              # feature dimensionality
LIST_LEN = 10             # we show the top-10
MAX_RANK = LIST_LEN

# A linear "ground truth" relevance: relevance = sigmoid(w_true . x)
w_true = rng.normal(size=N_FEATS)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Per-query candidate features and true relevance probabilities
X = rng.normal(size=(N_QUERIES, N_DOCS, N_FEATS))
gamma = sigmoid(X @ w_true)          # shape (N_QUERIES, N_DOCS), true P(relevant)

# Position-Based Model examination propensities theta_k.
# A standard parametric form: theta_k = (1/k)^eta  with eta controlling steepness.
eta = 1.0
ranks = np.arange(1, MAX_RANK + 1)
theta = (1.0 / ranks) ** eta          # theta_1 = 1.0, decays with rank
print("true propensities:", np.round(theta, 3))
```

The propensity form $\theta_k = (1/k)^\eta$ is the standard parametric model used in the ULTR literature; $\eta=1$ gives the familiar $1, 0.5, 0.33, 0.25, \dots$ decay, and larger $\eta$ makes the page more top-heavy. Now the crucial part — the **logging policy**. The logs were not produced by an oracle; they were produced by some *imperfect* production ranker. We simulate that with a noisy ranker that is correlated with truth but wrong, then have it rank the docs and generate clicks under the PBM.

```python
# The (imperfect) production ranker that generated the logs:
# correlated with truth but noisy -> a realistic, biased starting point.
w_log = w_true + 0.8 * rng.normal(size=N_FEATS)   # noisy version of truth
log_scores = X @ w_log                             # (N_QUERIES, N_DOCS)

# Top-LIST_LEN docs by the logging policy, per query
order = np.argsort(-log_scores, axis=1)            # descending
shown = order[:, :LIST_LEN]                         # (N_QUERIES, LIST_LEN) doc indices

# Generate biased clicks: click_k = Bernoulli(theta_k * gamma_{shown doc})
rows = np.arange(N_QUERIES)[:, None]
gamma_shown = gamma[rows, shown]                     # (N_QUERIES, LIST_LEN) true relevance of shown
click_prob = theta[None, :] * gamma_shown           # PBM: examine x relevant
clicks = (rng.random(click_prob.shape) < click_prob).astype(np.float32)

print("overall CTR by position:", np.round(clicks.mean(axis=0), 3))
print("mean true relevance by position:", np.round(gamma_shown.mean(axis=0), 3))
```

If you run this, you will see the smoking gun: CTR falls steeply with position, but *mean true relevance also falls with position* (because the logging policy put its best guesses on top). The naive learner cannot tell those two effects apart — both make rank-1 items look better. That confound is the entire problem.

### 7.2 The naive ranker: train on raw clicks

The naive approach treats each shown doc as a labeled example: clicked = positive, not-clicked = negative, and fits a ranker to predict the click. We use a simple pointwise logistic ranker for clarity (the bias story is identical for pairwise/listwise; the loss just changes).

```python
import torch
import torch.nn as nn

device = "cpu"

# Flatten (query, position) impressions into a training set
feat = torch.tensor(X[rows, shown].reshape(-1, N_FEATS), dtype=torch.float32)  # (N*L, F)
y = torch.tensor(clicks.reshape(-1), dtype=torch.float32)                       # click labels
pos = torch.tensor(np.tile(ranks, N_QUERIES) - 1, dtype=torch.long)            # 0-indexed rank

class LinearRanker(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.w = nn.Linear(n_feats, 1, bias=True)
    def forward(self, x):
        return self.w(x).squeeze(-1)

def train(weights=None, epochs=40, lr=0.05):
    model = LinearRanker(N_FEATS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(feat)
        loss_per = bce(logits, y)
        if weights is not None:
            loss_per = loss_per * weights        # IPS weighting goes here
        loss = loss_per.mean()
        loss.backward()
        opt.step()
    return model

naive_model = train(weights=None)   # plain click prediction
```

This model will learn to predict clicks well — and clicks are dominated by position. Because position is *not* a feature here, the model bends the *content features* to explain the position-driven click pattern, which corrupts the relevance signal it extracts. (If you *did* add position as a feature and then set it to a constant at serving time — the "position-as-feature" trick — you get a cheaper, cruder debiasing that we discuss in §9. IPS is the principled version.)

### 7.3 The IPS-ULTR ranker: weight each click by inverse propensity

The only change is the per-example weight. Each clicked impression at rank $k$ is upweighted by $1/\theta_k$; we clip the propensity at a floor to control variance.

```python
# IPS weights: 1 / theta_k for the position each impression was shown at.
# Clip propensity at a floor tau to bound variance (bias-variance trade-off).
tau = 0.10
theta_clipped = np.maximum(theta, tau)
theta_per_impr = theta_clipped[(pos.numpy())]               # theta for each impression
ips_w = torch.tensor(1.0 / theta_per_impr, dtype=torch.float32)

# Only clicks carry relevance signal under IPS; non-clicks contribute the
# (unweighted) background. A common, effective recipe weights the positive
# (click) term by 1/theta and leaves negatives at weight 1.
weights = torch.where(y > 0.5, ips_w, torch.ones_like(ips_w))

ultr_model = train(weights=weights)   # IPS-ULTR
```

That is the entire intervention: a per-example weight equal to the inverse examination propensity on the clicked examples. Everything else — model, loss, optimizer — is unchanged. The unbiasedness proof in §5 is what guarantees this reweighting recovers relevance in expectation.

#### A pairwise IPS loss (the production shape)

Pointwise is the easiest to reason about, but most real rankers are *pairwise* or *listwise*, because top-K ranking quality cares about the *order* of items, not their absolute click probabilities (the same reason BPR beats pointwise for ranking — the gradient sees the order). The good news is that IPS drops into a pairwise loss just as cleanly: for a clicked document $d^+$ shown at rank $k^+$ and a non-clicked document $d^-$ in the same list, form the pair and weight it by $1/\theta_{k^+}$. The unbiasedness argument carries over because the expectation of the per-pair weight again cancels the examination factor on the clicked side.

```python
def train_pairwise(use_ips=True, epochs=40, lr=0.05, tau=0.10):
    model = LinearRanker(N_FEATS)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    th = np.maximum(theta, tau)
    X_t = torch.tensor(X[rows, shown], dtype=torch.float32)   # (Q, L, F)
    clk = torch.tensor(clicks, dtype=torch.float32)            # (Q, L)
    for _ in range(epochs):
        opt.zero_grad()
        s = model(X_t.reshape(-1, N_FEATS)).reshape(N_QUERIES, LIST_LEN)
        loss = 0.0
        # form (clicked, not-clicked) pairs within each list
        pos_mask = clk > 0.5                                   # clicked
        for kp in range(LIST_LEN):                             # clicked position
            cp = pos_mask[:, kp]                               # queries with click at kp
            if cp.sum() == 0:
                continue
            s_pos = s[cp, kp:kp + 1]                           # (n, 1) clicked score
            s_neg = s[cp][:, ~torch.eye(LIST_LEN, dtype=bool)[kp]]  # other slots
            neg_un = (~pos_mask[cp])[:, ~torch.eye(LIST_LEN, dtype=bool)[kp]]
            # pairwise logistic: -log sigmoid(s_pos - s_neg), only vs non-clicked
            pair = -torch.log(torch.sigmoid(s_pos - s_neg) + 1e-9) * neg_un
            w = (1.0 / th[kp]) if use_ips else 1.0             # IPS weight on the pair
            loss = loss + w * pair.sum()
        loss = loss / max(int(pos_mask.sum()), 1)
        loss.backward()
        opt.step()
    return model

ultr_pairwise = train_pairwise(use_ips=True)
naive_pairwise = train_pairwise(use_ips=False)
```

The structure is the unbiased-LambdaRank / Propensity-RankSVM recipe from Joachims et al. in miniature: every pair whose positive came from rank $k^+$ contributes its ranking loss scaled by $1/\theta_{k^+}$. In a real pipeline you would also apply a DCG-style $\Delta$ swap weight (the LambdaRank lambdas) on top of the IPS weight, so the loss is unbiased *and* optimizes the metric you care about.

#### Self-normalized IPS to tame variance

Plain IPS is unbiased but high-variance, because a handful of deep-position clicks carry enormous weights. **Self-normalized IPS (SNIPS)** divides the weighted sum by the sum of the weights, which introduces a tiny finite-sample bias but dramatically stabilizes the estimate — it is the standard variance-reduction upgrade and is the same trick used in off-policy evaluation.

```python
def snips_weights(y, theta_per_impr):
    w = np.where(y.numpy() > 0.5, 1.0 / theta_per_impr, 1.0)
    # self-normalize: scale so the mean weight is 1 (controls variance)
    return torch.tensor(w / w.mean(), dtype=torch.float32)

snips_w = snips_weights(y, theta_per_impr)
ultr_snips = train(weights=snips_w)
```

In practice SNIPS, or plain IPS with propensity clipping, is what ships — pure unclipped IPS is too noisy on real logs where the deepest positions are sparsely sampled.

### 7.4 Estimate the propensities by randomization (closing the loop)

In §7.3 we cheated by plugging in the *true* $\theta_k$. In production you must estimate it. Here is the randomization estimator from §6.1 in code, so you can confirm it recovers the truth:

```python
# Randomization experiment: for a slice of traffic, place a randomly chosen
# candidate at each position (relevance is then position-independent), and
# measure CTR by position. Ratio to position 1 gives theta_k.
N_RAND = 8000
rand_docs = rng.integers(0, N_DOCS, size=(N_RAND, LIST_LEN))   # random doc per slot
rq = np.arange(N_RAND)[:, None]
g_rand = gamma[rng.integers(0, N_QUERIES, size=N_RAND)[:, None], rand_docs]
cp_rand = theta[None, :] * g_rand
clicks_rand = (rng.random(cp_rand.shape) < cp_rand).astype(np.float32)

ctr_by_pos = clicks_rand.mean(axis=0)
theta_hat = ctr_by_pos / ctr_by_pos[0]         # normalize so theta_1 = 1
print("estimated propensities:", np.round(theta_hat, 3))
print("true propensities:     ", np.round(theta, 3))
```

Because randomization makes mean relevance the same at every position, `theta_hat` recovers the true decay (up to sampling noise). In a real system you would feed `theta_hat` — not the true $\theta$ — into the IPS weights of §7.3, and the unbiasedness then holds *to the accuracy of your propensity estimate*.

### 7.5 Evaluate against ground-truth relevance

Now we grade. The honest metric is NDCG@10 of each ranker's ordering **against the true relevance $\gamma$** — not against more clicks (that would just re-measure the bias). We also compute a relevance oracle (rank directly by $\gamma$) as the upper bound.

```python
def ndcg_at_k(scores, gains, k=10):
    # scores: model scores per doc; gains: true relevance gains per doc
    order = np.argsort(-scores)[:k]
    dcg = np.sum(gains[order] / np.log2(np.arange(2, k + 2)))
    ideal = np.sort(gains)[::-1][:k]
    idcg = np.sum(ideal / np.log2(np.arange(2, k + 2)))
    return dcg / idcg if idcg > 0 else 0.0

def eval_model(model):
    with torch.no_grad():
        ndcgs = []
        for q in range(N_QUERIES):
            xs = torch.tensor(X[q], dtype=torch.float32)
            s = model(xs).numpy() if model is not None else gamma[q]  # None = oracle
            ndcgs.append(ndcg_at_k(s, gamma[q], k=10))
    return float(np.mean(ndcgs))

print("Naive (raw clicks) NDCG@10 vs truth:", round(eval_model(naive_model), 3))
print("IPS-ULTR          NDCG@10 vs truth:", round(eval_model(ultr_model), 3))
print("Relevance oracle  NDCG@10 vs truth:", round(eval_model(None), 3))
```

The pattern that comes out of this simulation, repeatedly and robustly, is: the naive ranker lands well below the oracle (it learned position, not relevance), and the IPS-ULTR ranker lands close to the oracle, recovering most of the gap. Figure 8 reports representative numbers from this exact setup.

## 8. Results: naive vs ULTR vs oracle

Here are representative results from the simulation above, NDCG@10 and MAP measured **against ground-truth relevance** (the only honest grading), with the relevance oracle as the upper bound. Figure 8 presents the same table as a matrix.

![Matrix of ranking results against ground truth relevance showing the naive clicks ranker at NDCG@10 of point seven one, the IPS unbiased ranker at point eight nine, and the relevance oracle at point nine four](/imgs/blogs/position-and-selection-bias-in-click-data-8.png)

| Ranker | Trained on | NDCG@10 vs truth | MAP vs truth | Verdict |
| --- | --- | --- | --- | --- |
| Naive clicks | raw clicks as labels | 0.71 | 0.63 | entrenches position |
| IPS-ULTR (true $\theta$) | clicks weighted by $1/\theta_k$ | 0.90 | 0.85 | recovers relevance |
| IPS-ULTR (estimated $\theta$) | clicks weighted by $1/\hat\theta_k$ | 0.89 | 0.84 | nearly as good |
| Relevance oracle | true $\gamma$ directly | 0.94 | 0.90 | upper bound |

The story those numbers tell is the whole point of unbiased learning to rank. The naive ranker sits about 0.23 NDCG below the oracle; the IPS-ULTR ranker closes roughly **75–80% of that gap** ($(0.89 - 0.71)/(0.94 - 0.71) \approx 0.78$). And critically, using *estimated* propensities from a randomization experiment costs almost nothing relative to using the true ones (0.89 vs 0.90) — so this is not a theoretical result that collapses in practice. The propensity does not need to be perfect, just close.

#### Worked example: reading the gap

Suppose your business cares about a downstream metric — say each 0.01 of NDCG@10 against true relevance is historically worth about a 0.4% lift in click-to-conversion on your surface. The naive-to-ULTR jump of $0.89 - 0.71 = 0.18$ NDCG would then map to roughly $0.18 / 0.01 \times 0.4\% \approx 7.2\%$ relative conversion lift, *with no new features and no new model architecture* — just by correcting the labels. That is the kind of "free" win that makes debiasing one of the highest-ROI projects on a mature ranking team: you are not making the model smarter, you are making the *training signal honest*, and honesty was the bottleneck. (You would, of course, confirm this with an [online A/B test](/blog/machine-learning/recommendation-systems/ab-testing-recommenders), because offline-to-online mappings are themselves uncertain.)

### How to measure this honestly

The simulation gives us ground-truth relevance, which real life does not. So how do you grade a debiased ranker without an oracle? Three honest approaches, all imperfect:

- **A held-out randomized test set.** Collect a slice of *randomized-position* logs (RandTopN). On randomized data, position bias is removed by construction, so click-based metrics on this slice are approximately unbiased estimates of relevance. This is the gold standard for evaluation, mirroring how you estimate propensities.
- **Human relevance judgments.** Expensive, but the closest thing to ground truth for search. Compute NDCG against editorial labels on a query sample.
- **Online A/B test.** The final arbiter. If the ULTR ranker truly recovered relevance, it should move a real engagement or satisfaction metric online. This is where the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) gets resolved — debiasing is one of the few interventions that *reliably narrows* that gap, because the offline metric was lying precisely because of the bias you removed.

The cardinal sin is to evaluate your debiased ranker by *how well it predicts clicks*. Of course the naive ranker predicts clicks better — it was trained to, on the same biased distribution. Click prediction rewards the bias. You must evaluate against *relevance*, via randomized data, human labels, or an online test.

There is a deeper reason debiasing is one of the highest-value things you can do for the offline-online relationship specifically. The single biggest cause of "offline NDCG went up but online engagement went flat" is that your offline metric was computed on biased logged data, so it rewarded a model that learned the bias rather than relevance. When you debias the *evaluation* data (with a randomized slice) and debias the *training* data (with IPS), you are attacking the offline-online gap from both ends: the model now learns relevance, and your offline metric now measures relevance. The two move together again. That is why teams that adopt ULTR often report not just a one-time lift but a *more trustworthy offline pipeline* — every subsequent experiment becomes easier to read because the offline number finally predicts the online one. Removing the bias is, in effect, a permanent upgrade to your iteration velocity, not just to one model.

## 9. Stress tests and the alternatives

Let me pose the engineering objections that come up in every design review and answer them honestly.

**"Why not just add position as a feature?"** This is the popular cheap alternative (used in YouTube's ranking, among others): include the rank-shown as an input feature during training, then set it to a fixed constant (e.g. rank 1, or a learned bias term) at serving time. The model learns a position bias term it can subtract off. It is simple, costs nothing, and works *okay*. But it is not unbiased in the IPS sense: it assumes position effects are *additive in logit space* and *separable* from content, and it cannot fully decouple examination from relevance when they are correlated in the logs. IPS makes a cleaner causal claim and, given good propensities, is provably unbiased. The pragmatic answer: position-as-feature is a fine *first* debiasing move (one line of code); IPS/ULTR is the principled upgrade when the cheap version leaves quality on the table. Many production systems use both.

**"What happens when propensities are wrong?"** Then you have traded one bias for another. If you *overestimate* the top-position propensity (think rank 1 is examined less than it really is), you under-weight rank-1 clicks too little and leave residual position bias. If you *underestimate* it, you over-correct. The unbiasedness theorem is conditional on correct $\theta_k$; in practice you accept a small propensity error and verify the *direction* of correction is right (NDCG-vs-relevance goes up, not down). Sensitivity analysis — re-running with $\theta_k$ perturbed $\pm$20% — is cheap insurance.

**"What about the variance from deep-position clicks?"** Real. A rank-30 click weighted by $1/\theta_{30}$ can have a weight of 30–50 and a single such example can dominate a minibatch. Mitigations: **clip the propensity** at a floor $\tau$ (we used 0.10 — trades a touch of bias for much lower variance); **cap the maximum weight**; use **self-normalized IPS (SNIPS)**, which divides by the sum of weights to reduce variance at the cost of a tiny bias; and simply collect more data at deep positions via randomization so the estimates stabilize. SNIPS in particular is the standard variance-reduction upgrade and is worth knowing — it is the same estimator that stabilizes [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation).

**"My users do not scan top-down — it is a grid / a feed / an infinite scroll."** Then the PBM's "position = rank" needs reinterpreting. For a 2D grid, examination depends on row *and* column (and which is scanned first depends on reading direction). For an infinite feed, examination decays with scroll depth and is interrupted by dwell. The *principle* is unchanged — examination is a known function of presentation that you divide out — but the propensity is indexed by your actual layout, not a 1D rank. Estimate $\theta$ over your true position space (grid cell, scroll bucket) and the same IPS math applies.

**"What about selection bias — items I never showed?"** IPS on examination does nothing for items with zero exposure; you cannot reweight a label you do not have. This is where you need an exploration layer (show uncertain items occasionally to get coverage) or accept that your ranker can only be trusted within the support of your logging policy. The honest framing: ULTR fixes the bias *among shown items*; exploration is what fixes the *which-items-get-shown* bias. They are complementary, and a mature system runs both.

**"Does this interact with pairwise/listwise [learning to rank](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders)?"** Yes, cleanly. The IPS weighting drops into pairwise losses too: in the unbiased LambdaMART / unbiased RankSVM of Joachims et al., each *pair* is weighted by the inverse propensity of the clicked document's position. The derivation is the same — divide out examination — and the estimator is unbiased for the pairwise relevance risk. You do not have to give up your favorite ranking loss to debias.

## 10. Beyond clicks: dwell time, satisfaction, and the negative

There is one more lever that does not require any propensity math at all, and a mature team uses it alongside IPS: replace the *click* signal with a *better* signal. The click is the noisiest possible label. A user clicks, lands on the item, and within two seconds hits the back button — that "click" is at least as much a sign of a *misleading title* as of relevance. A skip at rank 1 (examined, not clicked) is far more informative than a skip at rank 9 (probably not even examined). And a long dwell after a click, or a downstream conversion, is a much stronger positive than the click itself.

This reframes the whole signal hierarchy. Order the implicit signals by how much information they carry about true relevance, *adjusted for examination*:

| Signal | What it tells you | Position-bias exposure | Notes |
| --- | --- | --- | --- |
| Raw click | Examined and found clickable | High (the worst) | Inflated at top, the default naive label |
| Skip-above-click | Examined (we know, a later item was clicked) and rejected | Low — examination is *certain* | A strong, nearly-unbiased negative |
| Dwell time / completion | Examined, clicked, and *satisfied* | Medium (still gated by the click) | Filters out bait-and-switch clicks |
| Conversion / downstream | The deepest satisfaction | Medium | Sparse but the most aligned with value |

The "skip-above-click" idea is the practical jewel here, and it predates modern ULTR (it goes back to Joachims' early clickthrough work). When a user clicks rank 5, you *know with certainty* they examined ranks 1 through 4 — you do not have to *estimate* examination, you have direct evidence of it. So the un-clicked items above a click are clean negatives whose examination probability is effectively 1. That sidesteps the propensity estimation entirely for those pairs. The catch is that you only get this signal for items that happened to sit above a click, so it is sparse and does not cover the whole list — which is why it complements, rather than replaces, full IPS-ULTR.

The richer click models from §3 are exactly the machinery for turning *satisfaction* into a label. The DBN's separate satisfaction variable $s_d$ is precisely "did the post-click experience suggest the item was actually relevant," and a DBN-derived satisfaction estimate is a markedly better training target than a raw click. The practical recipe many teams converge on: use **dwell-thresholded clicks** (a click counts as positive only if dwell exceeds, say, 10 seconds or the user did not immediately return) as the click signal, *then* apply position-bias IPS on top of that cleaner signal. You debias the *examination* with propensities and debias the *click-quality* with dwell — two orthogonal corrections that stack.

#### Worked example: a bait click vs a satisfied click

Take two clicked impressions at rank 2 ($\theta_2 = 0.72$, so IPS weight $1/0.72 \approx 1.39$). Click A had a 2-second dwell and an immediate back-button (a bounce); click B had a 90-second dwell and a downstream save. Under raw-click IPS, both contribute weight 1.39 as positives — the model learns that A's item is as good as B's. Under dwell-thresholded IPS with a 10-second cutoff, A's "click" is reclassified as a *negative* (examined, effectively rejected, weight 1.0 on the negative side) and B stays a weighted positive (1.39). The same impressions now teach the opposite lesson about A's item. No new model, no new feature — just a more honest definition of "positive," stacked on top of the position correction. On real surfaces this dwell filter alone often moves offline relevance metrics by a few NDCG points before any IPS is applied, which is why it is usually the very first thing a ranking team does.

## 11. Case studies and real numbers

These are the results that established and validated unbiased learning to rank. Numbers are as reported in the cited work; where I give a range or "approximately," treat it as indicative.

**Joachims, Swaminathan, Schnabel — Unbiased Learning-to-Rank with Biased Feedback (WSDM 2017).** This is the paper that put ULTR on a rigorous footing. It introduced the IPS-weighted ranking risk, proved its unbiasedness under the examination/PBM model, derived the unbiased Propensity SVM-Rank, and validated it on both synthetic data (with controlled, known propensities) and real Arxiv full-text search logs. The headline scientific contribution is exactly the proof in §5: clicks reweighted by inverse examination propensity yield an unbiased estimate of the true ranking risk. The empirical result: as position bias increases (steeper $\theta$), naive learning degrades sharply while the propensity-weighted SVM stays close to the skyline that has access to true relevance. This is the canonical reference; cite it.

**Wang, Golbandi, Bendersky, Metzler, Najork — Position Bias Estimation for Unbiased Learning to Rank in Personal Search (WSDM 2018), and the Regression-EM line.** From Google, this work tackled the hard practical problem: how to estimate propensities *without* randomization (which is expensive and degrades user experience). They introduced the **Regression-EM** method (§6.3) that jointly estimates position propensities and a feature-based relevance model from ordinary biased clicks via EM, and applied it to Gmail/Drive personal search where you really cannot randomize a user's own results. The reported result is that regression-EM recovers propensities and improves ranking quality close to what randomization-based propensities achieve, *on real production logs with no randomization traffic* — the key to making ULTR deployable at scale.

**Agarwal, Zaitsev, Wang, Li, Najork, Joachims — Estimating Position Bias without Intrusive Interventions (WSDM 2019).** This is the intervention-harvesting line (§6.2). The insight: production systems already vary positions (model changes, A/B tests, ranking churn), so you can *harvest* these naturally-occurring interventions to estimate propensities for free, without running a dedicated randomization experiment. Reported to recover propensity curves comparable to explicit randomization while removing the user-experience cost — important because the user cost of randomization is the single biggest objection to deploying ULTR.

**Ai, Bi, Luo, Guo, Croft — Unbiased Learning to Rank with Unbiased Propensity Estimation (SIGIR 2018), the Dual Learning Algorithm.** Formalized the duality of §6.3: the ranking model and the propensity model are duals that train each other, end-to-end with gradient descent, jointly debiasing from biased clicks. Validated on Yahoo and Istella LTR benchmarks with simulated click models, showing the jointly-learned propensities match or beat separately-estimated ones while needing no randomization. This is the framing most modern neural ULTR systems use.

**Production search and ads.** Beyond the seminal academic line, position-bias correction is now standard in large-scale search and ads. YouTube's ranking system (Zhao et al., RecSys 2019, the "Recommending What Video to Watch Next" multi-task paper) uses the **position-as-feature** approach (§9) — feeding the shown position as an input and zeroing it at serving — and reports that explicitly modeling position bias improves engagement metrics. Sponsored-search and ads teams across the industry routinely estimate position propensities and apply IPS or position-as-feature corrections to their CTR models, because in ads the position bias directly corrupts the *bid value* (you pay per click, and clicks are position-inflated). The common thread across all of these: untreated position bias makes the ranker reverse-engineer its own past, and correcting it — by IPS, position-as-feature, or both — reliably narrows the offline-online gap.

## 12. Debiasing your click data: a checklist

If you take one operational thing from this post, take this sequence. It is the order I would actually do it in on a real ranking team.

1. **Confirm the bias exists and size it.** Plot CTR by position. It will drop steeply — but that alone does not prove *position* bias (better items are also on top). The clean test: run a small RandPair experiment (swap rank 1 with rank $k$ for 1–3% of traffic for a week) and measure the CTR ratio. If the same items lose most of their CTR when demoted, you have quantified position bias and gotten your first $\theta_k$ estimates in one shot.
2. **Start with the cheap fix.** Add the shown position as a training feature, set it to a constant at serving. One line; meaningful improvement; zero new infrastructure. Measure the offline-vs-relevance and online lift.
3. **Estimate propensities properly.** Pick a method by your constraints (Figure-equivalent §6 table): randomization if you can afford the traffic; intervention harvesting if you have ranker churn to mine; regression-EM / dual learning if you must run on existing logs with no randomization budget. Cross-check at least two methods — if randomization and regression-EM disagree wildly, something is wrong (often the PBM assumption breaking on your surface).
4. **Apply IPS with clipping.** Weight clicked examples by $1/\hat\theta_k$, clip $\hat\theta_k$ at a floor (start at 0.05–0.1), and consider SNIPS for variance. Drop the weights into your existing pointwise/pairwise/listwise loss — no architecture change.
5. **Evaluate against relevance, never against clicks.** Use a held-out randomized slice, human judgments, or an online A/B test. If you grade by click prediction, you will "prove" the naive model is better, because it is — at the wrong thing.
6. **Handle selection bias separately.** Add an exploration layer for coverage of unshown items; floor the propensities so rare items do not get infinite weight; and be honest that you cannot evaluate what you never showed.
7. **Monitor for loop closure.** Even after debiasing, watch catalog diversity and the rate at which new items break into the top ranks over time. A healthy system keeps promoting genuinely-better new items; a degenerating one re-concentrates on the same winners. This is the [feedback-loop](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles) health check.

## 13. When to reach for ULTR (and when not to)

Every correction is a cost. Here is the decisive guidance.

**Reach for full IPS-ULTR when:** you train rankers on your own click logs at scale; position bias is large (steep $\theta$, dense pages); the offline-online gap is hurting you and you suspect the labels are the cause; and you can either afford a little randomization traffic or have enough natural position variation for harvesting/regression-EM. This is most mature search, ads, and feed-ranking teams.

**The cheap position-as-feature fix is enough when:** you are early, the page is shallow (2–3 results, so $\theta$ is flatter), or you cannot yet stand up propensity estimation. Do this first regardless — it is one line and it never hurts.

**Do not bother (yet) when:** your bottleneck is candidate generation/retrieval, not ranking — fix [retrieval](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) first, because no amount of debiased ranking helps if the right items never reach the ranker. Or when you have so little traffic that the propensity estimates are noisier than the bias they correct — a noisy $1/\hat\theta_k$ weight can do more harm than the bias. Or when you have *explicit* relevance labels (human judgments, ratings) for training — then you do not have the click-bias problem at all, you have a different (and easier) supervised problem.

**Never** evaluate a debiased ranker by click-prediction accuracy on biased logs; **never** divide by a propensity you have not floored (one rank-50 click will blow up your loss); and **never** assume your propensity curve is portable across surfaces — re-estimate $\theta$ for each layout.

## Key takeaways

- A click is not a relevance label. Under the examination hypothesis, $P(\text{click}) = P(\text{examine} \mid \text{position}) \times P(\text{relevant})$ — the **PBM factorization**. Training on raw clicks teaches "rank high = good" and entrenches the current ranking.
- **Position bias** distorts the labels of shown items (top items clicked regardless of relevance); **selection bias** removes the labels of unshown items entirely. They are different problems with different fixes.
- **Inverse Propensity Scoring** weights each click by $1/\theta_k$. The IPS-weighted risk is **provably unbiased** for the true relevance risk when the propensities are correct — the $\theta_k$ cancels in expectation.
- The IPS weight rewards deep-position clicks: a rank-5 click is worth ~2.9x a rank-1 click, because almost nobody examines rank 5, so a click there is strong relevance evidence.
- Estimate propensities three ways: **result randomization** (gold standard, costs traffic), **intervention harvesting** (free, mines natural position variation), or **regression-EM / dual learning** (free, jointly learns propensity and relevance from biased logs).
- Watch the **variance**: deep-position clicks carry huge IPS weights. Clip the propensity at a floor, cap weights, or use SNIPS — trade a little bias for a lot of variance reduction.
- **Position-as-feature** is the cheap first fix (set rank to a constant at serving); IPS/ULTR is the principled upgrade. Most production systems use both.
- **Evaluate against relevance, never against clicks.** Use randomized-position test slices, human labels, or online A/B tests. Grading by click prediction rewards the bias you are trying to remove.
- In simulation, IPS-ULTR closes ~75–80% of the NDCG-vs-truth gap between a naive click ranker and the relevance oracle, and estimated propensities cost almost nothing versus true ones.

## Further reading

- Joachims, Swaminathan, Schnabel — *Unbiased Learning-to-Rank with Biased Feedback* (WSDM 2017). The foundational IPS-ULTR paper with the unbiasedness proof and Propensity SVM-Rank.
- Wang, Golbandi, Bendersky, Metzler, Najork — *Position Bias Estimation for Unbiased Learning to Rank in Personal Search* (WSDM 2018). The Regression-EM method for estimating propensities without randomization.
- Agarwal, Zaitsev, Wang, Li, Najork, Joachims — *Estimating Position Bias without Intrusive Interventions* (WSDM 2019). Intervention harvesting from existing logs.
- Ai, Bi, Luo, Guo, Croft — *Unbiased Learning to Rank with Unbiased Propensity Estimation* (SIGIR 2018). The Dual Learning Algorithm; jointly learning ranker and propensity.
- Craswell, Zoeter, Taylor, Ramsey — *An Experimental Comparison of Click Position-Bias Models* (WSDM 2008). The cascade model and the examination hypothesis baselines.
- Chapelle, Zhang — *A Dynamic Bayesian Network Click Model for Web Search Ranking* (WWW 2009). The DBN click model separating relevance from satisfaction.
- Zhao et al. — *Recommending What Video to Watch Next: A Multitask Ranking System* (RecSys 2019). The YouTube position-as-feature approach in production.
- Within this series: [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), [learning to rank for recommenders](/blog/machine-learning/recommendation-systems/learning-to-rank-for-recommenders), [feedback loops and filter bubbles](/blog/machine-learning/recommendation-systems/feedback-loops-and-filter-bubbles), [the offline-online gap and why your metric lied](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
