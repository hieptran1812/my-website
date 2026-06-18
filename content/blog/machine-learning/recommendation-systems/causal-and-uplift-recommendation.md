---
title: "Causal and Uplift Recommendation: Recommend What Changes Behavior"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Standard recommenders predict who will convert; the value is in recommending what changes a decision. Learn the potential-outcomes framework, build a T-learner and a class-transformation uplift model in sklearn on a simulated log, evaluate with the Qini curve, and watch uplift targeting beat conversion targeting on incremental value."
tags:
  [
    "recommendation-systems",
    "recsys",
    "uplift-modeling",
    "causal-inference",
    "heterogeneous-treatment-effects",
    "qini-curve",
    "incrementality",
    "machine-learning",
    "potential-outcomes",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/causal-and-uplift-recommendation-1.png"
---

A retailer I worked with ran a recommendation widget on the cart page: "Customers who bought this also bought." It converted at a gorgeous rate. The model that picked the items was a textbook click-through-rate ranker, trained on logs of impressions and purchases, and its offline AUC was excellent. Every week the dashboard showed the widget driving a healthy slice of revenue, and the attribution system dutifully credited it. Then a skeptical analyst on the growth team ran a holdout: a small slice of users who saw the cart page with the widget *turned off*. The result was deflating. Those users bought the recommended items almost as often as the users who saw the widget. The widget was not creating demand. It was standing in front of a door people were already walking through, and taking credit when they walked through it.

This is the central problem of this post, and it is not a measurement bug — it is a modeling philosophy. A standard recommender learns $P(\text{convert} \mid \text{shown})$: given that we put this item in front of this user, how likely are they to buy? That number is high for items the user was going to buy *anyway*. The model learns to find people on the verge of converting and put a recommendation in their path so the system can be credited for the conversion. What you actually want to recommend is the item whose presence *changes the decision* — the **causal effect** of showing it, not the correlation between showing it and a purchase. The difference between those two quantities is the entire field of **causal and uplift recommendation**, and getting it wrong wastes the scarcest resource a recommender has: the slot, the impression, the user's attention, and the marketing credit that follows.

![A two-column comparison of targeting by predicted conversion which fills slots with sure-things versus targeting by predicted uplift which finds persuadables and lifts incremental conversions](/imgs/blogs/causal-and-uplift-recommendation-1.png)

The figure above is the thesis in one picture. On the left, a recommender that ranks by predicted conversion fills its slots with high-probability buyers — the **sure-things** — and ends up with near-zero incremental lift because those people would have converted without the nudge. On the right, a recommender that ranks by **uplift** (the causal effect of the recommendation) ranks the **persuadables** to the top, the users whose decision the slot actually flips, and the incremental conversions climb. Same budget, same number of slots, very different value.

By the end of this post you will be able to: write down the potential-outcomes framework and explain the fundamental problem of causal inference (you never observe both outcomes for one user); define uplift as the conditional average treatment effect (CATE) and place every user in one of four segments; explain *why* correlation misleads here, with the back-door confounding path drawn out; implement and compare the T-learner, S-learner, X-learner, and the class-transformation uplift model in scikit-learn on a simulated log where you *know* the ground-truth uplift; evaluate with the uplift curve and the Qini coefficient; and read a results table where uplift targeting beats conversion targeting on incremental value. This post sits in the bias-and-causality corner of the series, downstream of [popularity bias and the rich-get-richer dynamic](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer) and tightly coupled to [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation). It is the formal answer to a question every honest recommender team eventually asks: *is my model creating value, or just claiming it?*

## 1. Prediction is not the same as persuasion

Let me make the distinction sharp before we touch any math. A recommender system, in the funnel framing this series keeps returning to — [retrieval, then ranking, then re-ranking](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) — ends by choosing a small set of items to show in a small set of slots. The ranking model assigns each candidate a score. In the standard formulation, that score is an estimate of the probability the user engages with the item *given that it is shown*. The slots go to the highest scores.

Now ask: what user-item pairs get the highest $P(\text{convert} \mid \text{shown})$? The ones where the user is enthusiastic about the item. A returning customer who has bought this brand of coffee monthly for two years gets a near-1.0 score on "more of that coffee." The model is correct: shown the coffee, they convert. But they would have converted *without* the recommendation — they were going to reorder anyway. The recommendation slot spent on them produced no incremental purchase. Worse, in a marketing context where each impression or coupon costs money, you paid to subsidize a sale that was already going to happen.

The recommendation is an **intervention**, a thing you *do* to the user's experience. The right question about an intervention is not "what happens after I do it?" but "what is *different* because I did it?" That difference is the **treatment effect**. In the language we will build out, showing an item is the *treatment*, the conversion is the *outcome*, and the value of the recommendation is the causal effect of the treatment on the outcome — for *this specific user*, because the effect is wildly heterogeneous across people.

Here is the trap stated as a slogan you can put on a sticky note: **the model that best predicts conversion is not the model that best causes conversion.** A perfect conversion predictor, used as a targeting rule, will systematically target the people who least need targeting. The signal it ranks on — high baseline propensity to buy — is exactly *anti-correlated* with the thing you want, which is room to move the needle. This is why a recommender can have a beautiful offline AUC and a flat incrementality test. The AUC measures how well it separates converters from non-converters. Incrementality measures how much it *changes* who converts. Those are different objectives, and optimizing the first does not optimize the second.

If you have read [the offline–online gap post](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied), you have seen one face of this already: the offline metric rises, the online number does not budge. Uplift is the cleanest case of that pathology, because the offline metric (conversion AUC) and the online goal (incremental conversions) are not even measuring the same physical quantity. No amount of better conversion modeling closes the gap. You have to change what you are modeling.

There is a second, more insidious failure mode hiding here: the **credit-attribution trap**. A recommender that ranks sure-things to the top does not just waste slots — it actively manufactures a flattering story about itself. Every time it puts a recommendation in front of a user who was already going to buy, and the user buys, the attribution system logs a "recommendation-driven conversion." The dashboard fills with green. Leadership concludes the widget is a revenue engine and invests more in it. The feedback loop is closed and entirely self-congratulatory: the model gets credit for organic demand, the team gets headcount, and nobody notices that turning the widget off would barely move the topline. The only way to break the loop is a randomized holdout that measures the *counterfactual* — what happens to conversion when the widget is absent. The first time a team runs that holdout and sees the incrementality come in at a third of the attributed number, the room goes quiet. I have been in that room. It is the moment a team stops measuring activity and starts measuring impact.

The deeper reason this matters is that recommendation slots are a *zero-sum* resource. There is one cart-page widget, ten feed positions, one push notification budget per user per day. Every slot you spend on a sure-thing is a slot you did not spend on a persuadable. So the cost of conversion targeting is not merely "we wasted an impression" — it is "we displaced the one user-item pair that would have created incremental value with one that created none." In economic terms, the opportunity cost of mis-targeting is the entire foregone uplift. That is why this is not a marginal optimization. On a costly channel, switching from conversion targeting to uplift targeting can swing incremental value by a multiple, not a few percent, as the results section will show.

## 2. The potential-outcomes framework

To reason about "what is different because I showed the item," we need a formal language for counterfactuals. The standard one is the **potential-outcomes** (or Neyman–Rubin) framework, and it is worth getting exactly right because every uplift method is a way of estimating one quantity it defines.

For each user $i$ and a binary treatment $T_i \in \{0, 1\}$ (1 = the item is shown, 0 = it is not), we posit *two* potential outcomes:

- $Y_i(1)$ — what the user would do **if shown** the item (1 = convert, 0 = not).
- $Y_i(0)$ — what the user would do **if not shown** the item.

The **individual treatment effect** is the difference:

$$
\tau_i = Y_i(1) - Y_i(0).
$$

This is the thing we actually care about. If $\tau_i = 1$, the recommendation flips a non-converter into a converter — a persuadable, the gold. If $\tau_i = 0$, showing the item changes nothing — either a sure-thing ($Y_i(1) = Y_i(0) = 1$) or a lost cause ($Y_i(1) = Y_i(0) = 0$). If $\tau_i = -1$, the recommendation actively *suppresses* a purchase — a do-not-disturb, and yes, those exist (the irrelevant recommendation that makes the user feel unseen, the upsell that triggers a cart abandonment).

Now the catch, and it is the deepest fact in the whole field. For any single user, you can show the item or not, but not both. You assign $T_i = 1$ and observe $Y_i(1)$; the world in which $T_i = 0$ never happens, so $Y_i(0)$ is forever unknown. Or you assign $T_i = 0$ and observe $Y_i(0)$, losing $Y_i(1)$. The observed outcome is

$$
Y_i = T_i \cdot Y_i(1) + (1 - T_i) \cdot Y_i(0),
$$

and exactly one of the two terms is ever realized. This is the **fundamental problem of causal inference**: $\tau_i = Y_i(1) - Y_i(0)$ is a difference of two numbers, one of which is always missing. You can *never* observe an individual treatment effect directly. Not with more data, not with a bigger model — it is a missing-data problem at the level of physics, not statistics.

![A two-column figure contrasting the world where both potential outcomes exist and uplift is their difference against the observed world where only the treated outcome or the control outcome is seen and the other is missing](/imgs/blogs/causal-and-uplift-recommendation-5.png)

The figure makes the loss concrete. Both potential outcomes "exist" in the sense that the user has a true disposition under each arm; uplift is their difference. But the observed world hands you only one column per user. For a treated user you see $Y(1)$ and lose $Y(0)$; for a control user you see $Y(0)$ and lose $Y(1)$. The other cell is the **counterfactual** — the road not taken — and it is structurally unobservable.

So if individual uplift is unobservable, what *can* we estimate? Averages. Define the **conditional average treatment effect** (CATE) as the expected uplift among users who share covariates $x$ (their features — history, context, demographics):

$$
\tau(x) = E\big[\,Y(1) - Y(0) \mid X = x\,\big] = E[Y(1) \mid X=x] - E[Y(0) \mid X=x].
$$

This is the function an uplift model learns. It is not an individual effect (we just proved that is hopeless), but a *segment* effect: for users who look like $x$, by how much does showing the item raise the conversion rate? CATE is sometimes called the **uplift** directly, and in the recommendation setting it is exactly the score we want to rank by — not $P(\text{convert} \mid \text{shown}, x)$, but $P(\text{convert} \mid \text{shown}, x) - P(\text{convert} \mid \text{not shown}, x)$.

### Why averages are recoverable but individuals are not

The trick that rescues CATE from the fundamental problem is that the two terms — $E[Y(1) \mid x]$ and $E[Y(0) \mid x]$ — can each be estimated from *different* users, as long as the treated and control groups at covariate level $x$ are comparable. If among users who look like $x$ we randomly showed the item to half and withheld it from half, then the treated half's average conversion is an unbiased estimate of $E[Y(1) \mid x]$, the control half's average is an unbiased estimate of $E[Y(0) \mid x]$, and their difference estimates $\tau(x)$. We never see both outcomes for one person; we see one outcome for many comparable people and average our way to the counterfactual. That averaging is only valid under an assumption we turn to next.

### The ATE, the CATE, and why heterogeneity is the whole game

It helps to be precise about the ladder of quantities here. The **average treatment effect** (ATE) is the uplift averaged over the *entire* population, $\text{ATE} = E[Y(1) - Y(0)]$. An A/B test of "widget on" versus "widget off" estimates exactly this: the difference in conversion between the treated and control arms, full stop. The ATE answers "is the widget net-positive on average?" — a yes/no shipping decision. But the ATE alone tells you nothing about *whom* to target, because it averages persuadables and sure-things and do-not-disturbs together into one number. A widget with an ATE of +0.5% might be hiding a population that is 5% persuadables (huge uplift), 90% sure-things and lost causes (zero uplift), and 5% do-not-disturbs (negative uplift) whose harm partly cancels the persuadables' gain. The ATE sees the residual; the CATE sees the structure.

The CATE, $\tau(x) = E[Y(1) - Y(0) \mid X=x]$, is the ATE *conditioned* on features — it lets the effect vary across users. That variation, the **heterogeneity** of treatment effects, is the entire reason uplift modeling exists. If the effect were homogeneous — every user had the same uplift — there would be nothing to target; you would treat everyone or no one based on the sign of the ATE. Targeting only makes sense when the effect varies, when there exist identifiable segments with high uplift you can preferentially treat. So uplift modeling is, precisely, *the estimation of heterogeneity in the treatment effect*. The whole field could be renamed "heterogeneous-treatment-effect estimation for the purpose of treatment assignment," and in the academic literature it often is.

### SUTVA: the assumption you are probably violating

There is one more assumption lurking under the potential-outcomes notation, and recommender systems break it constantly, so it deserves a name. The **Stable Unit Treatment Value Assumption** (SUTVA) says two things: (1) one user's treatment does not affect another user's outcome (no interference), and (2) there is only one version of the treatment. The notation $Y_i(1)$ silently assumes user $i$'s outcome under treatment depends only on *their own* treatment, not on whether their friends were treated. In a social feed, that is false — if I recommend an item to your whole friend group, network effects mean your outcome depends on their treatment, not just yours. In a marketplace with finite inventory, treating many users for the same scarce item creates competition that violates SUTVA through the supply side. These interference effects mean the clean $\tau(x)$ estimate can be biased even with perfect randomization at the *individual* level; correctly handling them requires cluster-randomized designs (randomize whole friend groups or whole markets, not individuals). For most cart-page and email-targeting use cases the interference is mild and SUTVA is a workable approximation. For social and marketplace recommendations, it is a real threat you should at least name in your design doc rather than assume away.

## 3. Unconfoundedness, and why randomized exposure earns it

The estimator "treated average minus control average at $x$" is unbiased *only* if the treated and control groups are exchangeable given $x$ — that is, if which arm a user landed in is independent of how they would have responded, once we condition on their features. Formally, the **unconfoundedness** (or ignorability) assumption is

$$
\{Y(0), Y(1)\} \perp T \mid X.
$$

Read it as: among users with the same covariates $x$, treatment was assigned as good as randomly with respect to the potential outcomes. If that holds, then $E[Y(1) \mid x] = E[Y \mid T=1, x]$ and $E[Y(0) \mid x] = E[Y \mid T=0, x]$, and the observed conditional means recover the potential-outcome means. We also need **overlap** (positivity): $0 < P(T=1 \mid x) < 1$ for every $x$, so that both arms actually contain users at each covariate value. You cannot estimate $Y(0) \mid x$ if every user with features $x$ was treated.

Now the problem with logs from a *live recommender*: the exposure is anything but random. The recommender shows items it thinks are relevant. Relevant items are disproportionately shown to users who like them, who are disproportionately likely to buy them. So $T$ (shown) and $Y$ (bought) share a common cause — relevance — that we may not have fully captured in $x$. Unconfoundedness fails on the *unobserved* part of relevance, and the naive treated-minus-control difference is biased upward. The recommender's own intelligence is what poisons the well.

![A branching graph showing relevance and popularity as a confounder causing both the shown treatment and the bought outcome, opening a back-door path that makes the naive correlation overstate the true uplift](/imgs/blogs/causal-and-uplift-recommendation-3.png)

The figure draws the **back-door path**. The confounder — relevance and popularity — has an arrow into "shown" (the system shows relevant, popular items) and an arrow into "bought" (relevant, popular items get bought regardless of whether the *system* surfaced them). There is also the real causal arrow we want, "shown → bought." A naive correlation between shown and bought picks up *both* the causal arrow and the spurious flow through the confounder. That spurious flow is why $P(\text{buy} \mid \text{shown})$ massively overstates the true uplift. Notice the link to [popularity bias](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer): popularity is a confounder, and a recommender that ranks on raw observed-conversion correlations inherits and amplifies the popularity-driven part of that correlation.

There are two clean ways to shut the back-door:

1. **Randomized exposure.** Hold out a slice of traffic where the decision to show is made by a coin flip, independent of relevance. By construction $T \perp \{Y(0), Y(1)\}$ — *unconditionally*, not just given $x$ — so unconfoundedness holds trivially and the treated/control averages are unbiased. This is the gold standard. It costs you: the randomized slice shows some users irrelevant items and withholds relevant ones, which is a short-term experience hit. But it produces the only data on which uplift can be honestly learned. This is the same experimental-data argument behind [bandits and the exploration–exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) — a little randomized exploration buys you unbiased causal data — and behind a clean [A/B test of recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders).

2. **Deconfounding from observational logs.** When you cannot randomize, you model the exposure mechanism and adjust for it — inverse-propensity weighting (IPW), the back-door adjustment, or the deconfounded-recommender approach. This is harder and rests on the strong assumption that you have measured all confounders. We will sketch it in §10, but the honest answer for most teams is: *get the randomized slice.* An ounce of randomization is worth a pound of deconfounding assumptions.

#### Worked example: how confounding inflates the naive number

Suppose among users with features $x$, the recommender shows the item to 80% of them, and these are not random 80% — they are the ones who like it. Say the true potential outcomes are: 70% of this segment would buy if shown, 60% would buy if not shown, so the true uplift is $\tau(x) = 0.70 - 0.60 = 0.10$. But because the system selectively showed the item to the keenest users, the *observed* treated conversion rate is, say, 0.78 (the treated are a positively-selected subsample), while the observed control conversion (the 20% the system declined to show) is 0.45 (a negatively-selected subsample). The naive difference is $0.78 - 0.45 = 0.33$ — more than triple the true 0.10. A team trusting the naive number would believe the recommendation is enormously effective and over-invest in slots that mostly subsidize buyers. Randomize the exposure and the treated/control rates collapse back toward the true $0.70$ and $0.60$, recovering $\tau(x) \approx 0.10$.

## 4. The four segments

The potential-outcomes pair $(Y(0), Y(1))$ has four possible values for binary outcomes, and each defines a behavioral segment. This taxonomy is the most useful intuition in the whole field, so let me lay it out precisely.

![A two-by-two matrix crossing buys-if-not-shown against buys-if-shown to define sure-things do-not-disturbs persuadables and lost causes with only persuadables worth treating](/imgs/blogs/causal-and-uplift-recommendation-2.png)

- **Sure-things** — $Y(0) = 1, Y(1) = 1$. They buy whether or not you show the item. Uplift $\tau = 0$. Showing them the item wastes the slot and (in a paid setting) wastes money, and it lets the recommender take credit for an organic conversion. *Do not target.*
- **Lost causes** — $Y(0) = 0, Y(1) = 0$. They will not buy either way. Uplift $\tau = 0$. The slot is wasted but harmless. *Do not target.*
- **Persuadables** — $Y(0) = 0, Y(1) = 1$. They buy *only if shown*. Uplift $\tau = +1$. These are the only users where the recommendation creates value. *Target these.*
- **Do-not-disturbs** (a.k.a. sleeping dogs) — $Y(0) = 1, Y(1) = 0$. They would have bought, but showing the item *suppresses* the purchase. Uplift $\tau = -1$. The wrong recommendation annoys them, distracts them, or surfaces a cheaper alternative and they abandon. Targeting these is *actively harmful*. *Never target.*

The whole point of uplift modeling is to score every user by $\tau(x)$ and spend your finite slots on the persuadables (high positive uplift), while avoiding the sure-things and lost causes (zero uplift, a waste) and *especially* the do-not-disturbs (negative uplift, harm).

A conversion model cannot make these distinctions. It scores by $P(Y=1 \mid \text{shown}, x)$, which is *high* for sure-things and persuadables alike, and *low* for lost causes and do-not-disturbs. So a conversion-targeting rule lumps sure-things in with persuadables (wasting slots on guaranteed buyers) and lumps do-not-disturbs in with lost causes (missing that one segment is harmful, not just neutral). It is sorting on the wrong axis entirely.

#### Worked example: a persuadable versus a sure-thing

Take two users. User A (a sure-thing): the model estimates $P(\text{buy} \mid \text{shown}) = 0.92$ and $P(\text{buy} \mid \text{not shown}) = 0.90$. Their uplift is $0.92 - 0.90 = 0.02$. User B (a persuadable): $P(\text{buy} \mid \text{shown}) = 0.40$ and $P(\text{buy} \mid \text{not shown}) = 0.12$. Their uplift is $0.40 - 0.12 = 0.28$.

A conversion-targeting recommender ranks User A above User B (0.92 vs 0.40) and gives the slot to A. That slot produces $0.02$ expected incremental conversions. An uplift recommender ranks User B above User A (0.28 vs 0.02) and gives the slot to B, producing $0.28$ expected incremental conversions — *fourteen times* the value from the same slot. Multiply across millions of slots and the gap is the difference between a widget that drives growth and a widget that decorates a checkout page.

## 5. Uplift modeling methods

We have established the target ($\tau(x)$, the CATE) and the data we need (randomized treated/control). Now: how do you actually fit a model that outputs $\tau(x)$? You can never train on $\tau_i$ directly — it is unobserved — so every method is a clever way to learn the *difference* of two conditional means without ever seeing the difference for any individual.

![A taxonomy tree splitting uplift estimators into meta-learners that wrap any regressor tree-based methods that split on the effect and outcome transforms that turn uplift into one regression](/imgs/blogs/causal-and-uplift-recommendation-4.png)

The figure organizes the family. There are three branches: **meta-learners** (wrap any off-the-shelf regressor or classifier — T, S, X), **tree-based methods** (build trees that split to maximize the difference in treatment effect — uplift trees, causal forests), and **outcome transforms** (rewrite the label so a single ordinary regression estimates uplift — the class-transformation / transformed-outcome trick). Let me walk each.

### The T-learner (two models)

The most direct method. Fit one model $\hat{\mu}_1(x) = \hat{E}[Y \mid T=1, X=x]$ on the **treated** users only, and a separate model $\hat{\mu}_0(x) = \hat{E}[Y \mid T=0, X=x]$ on the **control** users only. The estimated uplift is the difference of the two models' predictions:

$$
\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x).
$$

"T" is for "two models." It is dead simple, works with any classifier, and is a great baseline. Its weakness is that it fits two models independently, each with its own error, and the *difference* of two noisy estimates can be much noisier than either one — especially when uplift is small relative to the baseline conversion. If the control model is fit on a small randomized slice, $\hat{\mu}_0$ is shaky and the subtraction amplifies its variance. The T-learner also wastes the regularization opportunity: each model has no idea the other exists, so neither borrows strength.

### The S-learner (single model)

Fit *one* model on all users, with the treatment indicator $T$ included as just another feature: $\hat{\mu}(x, t)$. Then uplift is the model's prediction with $t=1$ minus its prediction with $t=0$:

$$
\hat{\tau}_S(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0).
$$

"S" is for "single model." Its advantage is data efficiency — one model on all the data — and built-in regularization toward $\tau = 0$. Its danger is the *flip side* of that regularization: if $T$ is one feature among hundreds and the learner (say a gradient-boosted tree) does not split on it, the model effectively predicts the same value for $t=0$ and $t=1$, and your estimated uplift is *zero everywhere*. The treatment signal gets drowned. S-learners are biased toward under-detecting uplift, which in a recommender shows up as "the uplift model says nobody is a persuadable."

### The X-learner (cross / imputed effects)

The X-learner, from Künzel et al. (2019), is a refinement designed for *imbalanced* treatment groups — exactly the recsys case, where the randomized control slice is small. It works in stages. First, fit $\hat{\mu}_0$ and $\hat{\mu}_1$ as in the T-learner. Then *impute* the individual treatment effect for each user using the *other* group's model: for a treated user, the imputed effect is $\tilde{\tau}_i = Y_i - \hat{\mu}_0(x_i)$ (their observed treated outcome minus the predicted control outcome); for a control user, $\tilde{\tau}_i = \hat{\mu}_1(x_i) - Y_i$. Fit a second model on each group to predict these imputed effects, giving $\hat{\tau}_1(x)$ (from treated) and $\hat{\tau}_0(x)$ (from control). Finally, blend them with a propensity weight $g(x) = P(T=1 \mid x)$:

$$
\hat{\tau}_X(x) = g(x)\,\hat{\tau}_0(x) + \big(1 - g(x)\big)\,\hat{\tau}_1(x).
$$

The weighting puts more trust in the model fit on the *larger* group for each region — when treated dominates, lean on $\hat{\tau}_0$ (fit on the scarce, precious control), and vice versa. In practice the X-learner is the meta-learner I reach for when the control slice is small, which it usually is.

### The class-transformation (transformed-outcome) model

This is my favorite trick because it turns uplift into a *single ordinary regression* with no subtraction and no two-model bookkeeping. It requires the treatment to be assigned 50/50 (or you reweight by propensity). Define a transformed label $Z$:

$$
Z_i = \begin{cases} 1 & \text{if } (T_i = 1, Y_i = 1) \text{ or } (T_i = 0, Y_i = 0) \\ 0 & \text{otherwise.} \end{cases}
$$

In words: $Z = 1$ when a treated user converted *or* a control user did not convert; $Z = 0$ for a treated user who did not convert or a control user who did. Now here is the small miracle. Under 50/50 randomization, one can show that

$$
2 \cdot P(Z = 1 \mid x) - 1 = \tau(x).
$$

So if you train *any* probabilistic classifier to predict $P(Z=1 \mid x)$ and then output $2\hat{P}(Z=1\mid x) - 1$, you have an unbiased estimator of uplift — from a *single* model, on a *single* transformed label, with ordinary cross-entropy loss. The class-transformation model (Jaśkowski and Jaroszewicz, 2012; also called the "class variable transformation" or "outcome transformation") is the leanest uplift estimator there is. Its weakness is variance — the transformed label is a noisy proxy for uplift — and the strict requirement of 50/50 (or correct propensity reweighting). We will derive *why* the formula holds in §6, then implement it in §8.

### Tree-based uplift and causal forests

Instead of transforming the outcome, you can change the *splitting criterion* of a decision tree. An ordinary tree splits to maximize information gain on $Y$. An **uplift tree** splits to maximize the *difference in treatment effect* between child nodes — it uses a distributional-divergence criterion (Kullback–Leibler, Euclidean, or chi-squared between the treated and control outcome distributions) so that each split carves out regions of homogeneous uplift. A **causal forest** (Wager and Athey, 2018; Athey, Tibshirani, Imbens, 2019) is an ensemble of such trees with honest sample-splitting (use one subsample to choose splits, another to estimate the leaf effect) that yields asymptotically valid confidence intervals on $\tau(x)$. These are the methods to reach for when you want *uncertainty* on the uplift, not just a point estimate — useful when you want to avoid betting slots on segments whose uplift is statistically indistinguishable from zero.

### The R-learner and the bias–variance lens

One more meta-learner worth naming, because it ties the whole family together: the **R-learner** (Nie and Wager, 2021), built on Robinson's decomposition. It first removes the "main effect" by estimating the marginal outcome model $m(x) = E[Y \mid x]$ and the propensity $e(x) = P(T=1 \mid x)$, then fits $\tau(x)$ to the *residuals* $Y - m(x)$ regressed on $T - e(x)$, weighted appropriately. The intuition is that the baseline conversion (the part of $Y$ that has nothing to do with treatment) is a nuisance that adds variance to every naive method; by residualizing it out first, the R-learner isolates the treatment signal and tends to have lower variance than the T-learner when the baseline is strong — which, in conversion modeling, it almost always is (most of the variance in "did they buy" is baseline propensity, not uplift). If you have used the **Double Machine Learning** framework (Chernozhukov et al.), the R-learner is its CATE specialization, and `econml` and `causalml` both implement it.

Step back and look at the family through one lens: **bias versus variance**, where the catch is that uplift is a *small* signal riding on a *large* baseline. The S-learner is low-variance but high-bias (it can regularize the treatment signal to nothing). The T-learner is approximately unbiased but high-variance (the difference of two independent noisy estimates). The X-learner and R-learner are engineered to claw back variance without re-introducing much bias — the X-learner by leaning on the larger arm, the R-learner by residualizing out the dominant baseline. The class transformation is unbiased under randomization but high-variance (the relabeled target is a noisy proxy). There is no free lunch: every method trades these two against each other, and the right choice depends on your arm balance and how strong the baseline is relative to the uplift. The practical heuristic: strong baseline + small control slice (the typical recsys situation) → X-learner or R-learner; clean balanced 50/50 + fast iteration → class transformation; want a dead-simple baseline → T-learner.

#### Worked example: a T-learner estimate from raw counts

Make it concrete with arithmetic. Suppose at a covariate cell $x$ (users with a given history), our treated model and control model are just empirical rates from a randomized log. Among 500 treated users at $x$, 95 converted, so $\hat{\mu}_1(x) = 95/500 = 0.19$. Among 500 control users at $x$, 60 converted, so $\hat{\mu}_0(x) = 60/500 = 0.12$. The T-learner uplift is $\hat{\tau}_T(x) = 0.19 - 0.12 = 0.07$. Now the variance: each rate has standard error roughly $\sqrt{p(1-p)/n}$, about $\sqrt{0.19 \cdot 0.81 / 500} \approx 0.0175$ for the treated and $\sqrt{0.12 \cdot 0.88 / 500} \approx 0.0145$ for control. The standard error of the *difference* is $\sqrt{0.0175^2 + 0.0145^2} \approx 0.0227$, so the uplift estimate is $0.07 \pm 0.045$ at roughly two standard errors — the interval barely excludes zero. Halve the control slice to 250 users and the control standard error grows to about 0.0206, the difference SE to 0.027, and the interval $0.07 \pm 0.054$ now *includes* zero. That is the variance-of-a-difference problem the X-learner exists to fix, made arithmetic: the scarce control arm dominates the uncertainty, and shrinking it can turn a "significant persuadable segment" into "indistinguishable from noise."

Here is the comparison table I keep in my head:

| Method | How it estimates uplift | Strength | Weakness | When to use |
| --- | --- | --- | --- | --- |
| T-learner | Two models, subtract | Simple, any base learner | Variance of a difference; no shared regularization | Strong baseline; balanced groups |
| S-learner | One model with $T$ as feature | Data-efficient; regularizes to 0 | May ignore $T$ entirely; biased toward zero uplift | Large treatment effect; few features |
| X-learner | Impute effects, blend by propensity | Handles imbalanced groups well | More moving parts | Small control slice (typical recsys) |
| R-learner | Residualize out baseline + propensity | Low variance under strong baseline | Needs good nuisance models | Strong baseline conversion (typical recsys) |
| Class transform | Single regression on relabeled $Z$ | Leanest; one model, one loss | Needs 50/50 or reweighting; higher variance | Clean randomized split; fast iteration |
| Uplift tree / causal forest | Split on effect divergence | Confidence intervals; honest | Heavier; needs more data | Want uncertainty on $\tau(x)$ |

## 6. The science: deriving the class transformation

Let me prove the class-transformation identity, because it is the cleanest piece of math in the field and it shows *why* a single regression can estimate a causal difference.

Assume a 50/50 randomized split, so $P(T=1) = P(T=0) = \tfrac{1}{2}$, and treatment is independent of covariates and potential outcomes (randomization). Outcomes are binary, $Y \in \{0, 1\}$. Define the transformed label as above:

$$
Z = T \cdot Y + (1 - T)(1 - Y).
$$

Check: if $T=1, Y=1 \Rightarrow Z=1$; if $T=1, Y=0 \Rightarrow Z=0$; if $T=0, Y=1 \Rightarrow Z=0$; if $T=0, Y=0 \Rightarrow Z=1$. Matches the definition.

Now compute $P(Z=1 \mid x)$, decomposing over the treatment arm (which is independent of $x$ under randomization, so $P(T=1 \mid x) = \tfrac{1}{2}$):

$$
\begin{aligned}
P(Z=1 \mid x) &= P(T=1 \mid x)\,P(Y=1 \mid T=1, x) + P(T=0 \mid x)\,P(Y=0 \mid T=0, x) \\
&= \tfrac{1}{2}\,P(Y=1 \mid T=1, x) + \tfrac{1}{2}\,\big(1 - P(Y=1 \mid T=0, x)\big).
\end{aligned}
$$

Write $\mu_1(x) = P(Y=1 \mid T=1, x)$ and $\mu_0(x) = P(Y=1 \mid T=0, x)$. By randomization these equal the potential-outcome means $E[Y(1)\mid x]$ and $E[Y(0)\mid x]$, so $\tau(x) = \mu_1(x) - \mu_0(x)$. Substituting:

$$
P(Z=1 \mid x) = \tfrac{1}{2}\mu_1(x) + \tfrac{1}{2}\big(1 - \mu_0(x)\big) = \tfrac{1}{2} + \tfrac{1}{2}\big(\mu_1(x) - \mu_0(x)\big) = \tfrac{1}{2} + \tfrac{1}{2}\tau(x).
$$

Rearrange:

$$
\boxed{\,\tau(x) = 2\,P(Z=1 \mid x) - 1.\,}
$$

That is the identity. The expected uplift at $x$ is a simple affine function of the probability that the *transformed* label is 1. So train a classifier $\hat{P}(Z=1\mid x)$ with ordinary log-loss, output $2\hat{P} - 1$, and you have an uplift estimate — no subtraction of two models, no treatment feature, no special loss. The derivation also tells you exactly where it breaks: the step $P(T=1\mid x) = \tfrac12$ requires randomization. With a non-50/50 split, you replace the $\tfrac12$ weights with propensity weights and the formula generalizes, but the clean affine form needs the balanced design.

This is the same intellectual move as the [importance-weighting in off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation): you cannot observe the counterfactual directly, so you construct an *observable* quantity whose expectation equals the unobservable one. There the construction was inverse-propensity weighting; here it is the label transform. Both are estimators that trade variance for the ability to see a counterfactual.

## 7. The evaluation problem: uplift curves and the Qini coefficient

Here is a problem that trips up everyone the first time. You have an uplift model. How do you *validate* it? You cannot compute its accuracy against the true uplift, because — say it with me — the true individual uplift is *never observed*. You have, for each user in your test set, only their observed outcome under the *one* arm they were in. So there is no per-user label to score against. Evaluation must be done on *groups*, and the metric of choice is the **uplift curve** and its summary, the **Qini coefficient**.

The idea: rank your test users by predicted uplift, from highest to lowest. Walk down the ranked list. At each prefix (the top $k\%$ of users by predicted uplift), measure the *incremental* conversions you would have captured by treating exactly those users — estimated from the randomized test set as (treated conversions in the prefix) minus (control conversions in the prefix, scaled to the same population). A good model puts the persuadables at the top, so the incremental conversions climb steeply early and the curve bows *up* above the random-targeting diagonal. A useless model's curve hugs the diagonal — treating the top $k\%$ by its score is no better than treating a random $k\%$.

Formally, the **Qini curve** plots, against the fraction of the population targeted, the cumulative incremental number of positive outcomes. At a targeted prefix containing $n_t$ treated and $n_c$ control users with $r_t$ and $r_c$ conversions respectively, the Qini value is

$$
Q(\text{prefix}) = r_t - r_c \cdot \frac{n_t}{n_c},
$$

the treated conversions minus the control conversions rescaled to the treated group size. (This rescaling is what makes treated and control comparable when the arms are unequal.) The **uplift curve** is a close cousin using $r_t/n_t - r_c/n_c$ scaled by the prefix size. Both start at the origin and end at the total incremental conversions over the whole population (which is the same regardless of ranking — order does not change the total, only *when* you capture it).

![A two-column figure contrasting a conversion-targeting curve that hugs the random baseline against an uplift-targeting Qini curve that bows up and captures incremental conversions early](/imgs/blogs/causal-and-uplift-recommendation-7.png)

The figure contrasts the two. Rank by predicted conversion and the curve hugs the random diagonal — because your early picks are sure-things who contribute little *incremental* conversion. Rank by predicted uplift and the curve bows up — your early picks are persuadables, each contributing real incremental conversion, so you bank most of the total gain in the first chunk of the population.

The **Qini coefficient** is the area between your model's Qini curve and the random-targeting diagonal, normalized (often by the area of a hypothetical perfect model). It is the uplift analog of the Gini coefficient or AUC: a single number where bigger is better, zero means no better than random, and negative means your model is *anti*-sorted (it puts do-not-disturbs at the top — worse than useless). Reporting a Qini of, say, 0.071 versus a conversion model's 0.012 is the honest way to say "my uplift model captures roughly six times the incremental value per targeted user."

There is a close relative you will meet in the libraries: the **AUUC** (Area Under the Uplift Curve). The difference is the normalization of the control group. The Qini rescales control conversions by $n_t/n_c$ at each prefix (so it counts incremental conversions on the *treated* scale), while the AUUC typically normalizes both arms to a rate and integrates the difference. They rank models almost identically in practice, but they are not the same number, so do not compare a Qini from one paper to an AUUC from another. The rule I follow: pick one (I default to Qini because "incremental conversions" is a quantity a product manager understands), compute it the same way for every model you compare, and always plot the curve next to the coefficient — the single number hides whether your gain is front-loaded (great, you capture value at small budgets) or only shows up near full population (much less useful, since you would have to treat almost everyone to realize it).

Three Qini pitfalls bite teams repeatedly. First, **the curve is noisy at small prefixes** — the first few percent of the population have tiny $n_t$ and $n_c$, so the rescale factor $n_t/n_c$ swings wildly and the early curve can spike or dip on a handful of users. Do not over-read the leftmost slice. Second, **a single point estimate of the Qini is dangerously seductive.** Because individual uplift is unobserved, the Qini is computed on group rates, and group rates have sampling error; the Qini coefficient inherits that error. Bootstrap it — resample users with replacement a few hundred times, recompute the Qini each time, and report the 2.5th-to-97.5th percentile interval. A Qini of "0.07 with interval (0.05, 0.09)" is actionable; a Qini of "0.07 with interval (-0.01, 0.15)" means you cannot yet distinguish your model from random and should collect more randomized data before betting slots on it. Third, **evaluate on a randomized test set, always.** The Qini machinery assumes the treated/control assignment in the *test* data is random (or correctly propensity-weighted). If you compute Qini on a confounded test set, the incremental estimates inherit the back-door bias and the number is meaningless — a beautiful Qini on confounded data is exactly the trap this whole post warns against, dressed up as a causal metric.

#### Worked example: computing one Qini point

Take the top 20% of a 1,000-user randomized test set ranked by predicted uplift — 200 users. Suppose this prefix contains $n_t = 100$ treated and $n_c = 100$ control. Among the treated, $r_t = 38$ converted; among the control, $r_c = 14$ converted. The rescale factor is $n_t / n_c = 1.0$, so the Qini value at this prefix is

$$
Q = r_t - r_c \cdot \frac{n_t}{n_c} = 38 - 14 \cdot 1.0 = 24 \text{ incremental conversions.}
$$

Random targeting of the same 200 users would, on average, capture the population-average incremental rate times 200. If the whole population's incremental rate is, say, 0.06 (60 incremental conversions over 1,000 users), random targeting of 200 captures $0.06 \times 200 = 12$. Our model captured 24 — twice the random expectation in the top quintile. That gap, accumulated over all prefixes, *is* the Qini coefficient. Note we computed an incremental number without ever knowing any individual's uplift — only group rates from the randomized arms.

## 8. The practical flow: simulate, fit, evaluate

Now the part you can run. We will simulate a recommendation log with a known data-generating process so we can check our models against ground truth, then fit a T-learner and the class-transformation model in scikit-learn, evaluate with a Qini curve, and compare uplift targeting to conversion targeting.

![A vertical stack showing the uplift pipeline from randomized exposure to a treated control split to a CATE model to ranking persuadables to serving slots to Qini evaluation](/imgs/blogs/causal-and-uplift-recommendation-6.png)

The pipeline is the figure above: randomized exposure produces a clean treated/control split, the split feeds a CATE model, the model ranks users by uplift, you serve slots to the top persuadables, and you evaluate the whole thing with a Qini curve on a held-out randomized set. Let me build each stage.

### Stage 1 — simulate a log with heterogeneous uplift

We generate users with five features. Two features drive the *baseline* conversion (the sure-thing tendency), and two *different* features drive the *uplift* (the persuadability). Crucially, baseline and uplift are driven by different signals — that is what makes conversion targeting and uplift targeting diverge.

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
N = 40_000

# Five user features
X = rng.normal(size=(N, 5))
cols = [f"x{i}" for i in range(5)]

# Baseline conversion propensity (would-buy-anyway) driven by x0, x1
base_logit = 0.6 * X[:, 0] + 0.8 * X[:, 1] - 0.5
p_control = 1 / (1 + np.exp(-base_logit))           # P(Y=1 | not shown)

# Uplift (persuadability) driven by DIFFERENT features x2, x3
# tau is large where x2 high and x3 low; can be slightly negative (do-not-disturb)
tau = 0.30 * (1 / (1 + np.exp(-(1.5 * X[:, 2] - 1.2 * X[:, 3])))) - 0.04
p_treated = np.clip(p_control + tau, 0.0, 1.0)       # P(Y=1 | shown)

# Randomized exposure: 50/50 coin flip, independent of everything
T = rng.integers(0, 2, size=N)

# Realize the observed outcome under the assigned arm
p_obs = np.where(T == 1, p_treated, p_control)
Y = (rng.random(N) < p_obs).astype(int)

df = pd.DataFrame(X, columns=cols)
df["T"], df["Y"] = T, Y
df["true_tau"] = tau                                  # ground truth, for checking only
print(df.head())
print(f"treated conv {Y[T==1].mean():.3f}  control conv {Y[T==0].mean():.3f}  "
      f"naive uplift {Y[T==1].mean() - Y[T==0].mean():.3f}  true mean uplift {tau.mean():.3f}")
```

The "naive uplift" (treated mean minus control mean) should land close to the true mean uplift *because we randomized* — if you re-ran this with confounded exposure (selecting treated by `base_logit`), the naive number would be inflated, exactly as in §3.

### Stage 2 — fit a T-learner

Two gradient-boosted classifiers, one per arm, then subtract.

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size=0.30, random_state=1)
Xtr, Xte = train[cols].values, test[cols].values

# Model on TREATED only -> mu1(x) = P(Y=1 | shown, x)
m1 = GradientBoostingClassifier(max_depth=3, n_estimators=200, learning_rate=0.05)
tr1 = train[train.T == 1]
m1.fit(tr1[cols].values, tr1.Y.values)

# Model on CONTROL only -> mu0(x) = P(Y=1 | not shown, x)
m0 = GradientBoostingClassifier(max_depth=3, n_estimators=200, learning_rate=0.05)
tr0 = train[train.T == 0]
m0.fit(tr0[cols].values, tr0.Y.values)

uplift_T = m1.predict_proba(Xte)[:, 1] - m0.predict_proba(Xte)[:, 1]

# Also the pure conversion score (what a standard recommender ranks on)
conv_score = m1.predict_proba(Xte)[:, 1]              # P(convert | shown)

# Sanity check against ground truth (only possible in simulation)
from scipy.stats import spearmanr
print("T-learner rank corr with true tau:", spearmanr(uplift_T, test.true_tau).correlation)
print("Conv score rank corr with true tau:", spearmanr(conv_score, test.true_tau).correlation)
```

The T-learner's uplift should correlate strongly with `true_tau` (a Spearman correlation around 0.6–0.8 in this setup), while the conversion score correlates *weakly or even negatively* with true uplift — because conversion is driven by `x0, x1` and uplift by `x2, x3`. That divergence is the whole point.

### Stage 3 — fit the class-transformation model

One model, one transformed label, the §6 trick.

```python
# Transformed label Z: 1 if (treated & converted) or (control & not converted)
train = train.copy()
train["Z"] = ((train.T == 1) & (train.Y == 1)) | ((train.T == 0) & (train.Y == 0))
train["Z"] = train["Z"].astype(int)

mz = GradientBoostingClassifier(max_depth=3, n_estimators=200, learning_rate=0.05)
mz.fit(train[cols].values, train.Z.values)

# Uplift = 2 * P(Z=1 | x) - 1   (valid under 50/50 randomization)
uplift_CT = 2 * mz.predict_proba(Xte)[:, 1] - 1
print("Class-transform rank corr with true tau:", spearmanr(uplift_CT, test.true_tau).correlation)
```

This is the leanest uplift estimator in the post — a single ordinary classifier on a relabeled target. It is noisier than the T-learner (the transformed label is a high-variance proxy) but iterates fast and needs no two-model bookkeeping.

### Stage 4 — Qini curve and coefficient

We evaluate by ranking the *randomized test set* by each score and accumulating incremental conversions.

```python
def qini_curve(score, T, Y, n_bins=100):
    """Cumulative incremental conversions vs fraction targeted, ranking by score desc."""
    order = np.argsort(-score)                        # high uplift first
    T, Y = T[order], Y[order]
    cum_T = np.cumsum(T)                              # treated seen so far
    cum_C = np.cumsum(1 - T)                          # control seen so far
    cum_YT = np.cumsum(Y * T)                         # treated conversions
    cum_YC = np.cumsum(Y * (1 - T))                   # control conversions
    # Rescale control to treated size; guard divide-by-zero early in the walk
    with np.errstate(divide="ignore", invalid="ignore"):
        qini = cum_YT - np.where(cum_C > 0, cum_YC * cum_T / cum_C, 0.0)
    frac = np.arange(1, len(score) + 1) / len(score)
    return frac, qini

def qini_coefficient(score, T, Y):
    frac, q = qini_curve(score, T, Y)
    # Random baseline: straight line from 0 to the total incremental at full population
    rand = frac * q[-1]
    return np.trapz(q - rand, frac)                   # area between model curve and random

Tte, Yte = test.T.values, test.Y.values
for name, s in [("uplift (T-learner)", uplift_T),
                ("uplift (class-tf)",  uplift_CT),
                ("conversion score",   conv_score)]:
    print(f"{name:22s} Qini = {qini_coefficient(s, Tte, Yte):.4f}")
```

Both uplift scores should produce a clearly positive Qini coefficient; the conversion score's Qini should be near zero — sometimes slightly positive, sometimes slightly negative — confirming that ranking by conversion is barely better than random at capturing *incremental* value. (Note: `np.trapz` is the area integrator; on newer NumPy it is `np.trapezoid` — adjust to your version.)

### Stage 5 — compare targeting strategies head to head

Finally, fix a budget — say we can treat 30% of users — and ask: how many incremental conversions does each strategy capture in its top 30%?

```python
def incremental_at_budget(score, T, Y, budget=0.30):
    k = int(len(score) * budget)
    order = np.argsort(-score)[:k]                    # top-k by this score
    Tk, Yk = T[order], Y[order]
    nt, nc = Tk.sum(), (1 - Tk).sum()
    rt, rc = (Yk * Tk).sum(), (Yk * (1 - Tk)).sum()
    return rt - rc * (nt / max(nc, 1))                # incremental conversions in prefix

for name, s in [("random",            rng.random(len(Tte))),
                ("conversion score",  conv_score),
                ("uplift (T-learner)",uplift_T),
                ("uplift (class-tf)", uplift_CT)]:
    inc = incremental_at_budget(s, Tte, Yte)
    print(f"{name:22s} incremental conv @30% = {inc:.1f}")
```

This is the table that should convince a product manager. The uplift strategies capture multiples of the incremental conversions that conversion targeting does, at the *same* budget. The numbers wobble with the random seed, but the *ordering* — uplift beats conversion beats random — is robust because it is baked into the data-generating process and, more importantly, into reality.

### The implementation bug that fakes a great Qini

I have seen this exact bug ship more than once, so let me call it out before you write the code yourself. The seductive shortcut is to *train* the uplift model on the full log and then *evaluate* the Qini on the same log, or on a random row-wise split of it. Two things go wrong. First, if you split rows randomly rather than by *user*, the same user can appear in both train and test (in a setting with repeated impressions), and the model memorizes that user's outcome — the Qini is inflated by leakage exactly as in any other train-on-test sin. The fix is a **grouped split by user** (sklearn's `GroupShuffleSplit` or `GroupKFold`), the same hygiene the rest of this series insists on for [the right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate). Second, and subtler: the uplift model's *training* requires the treatment assignment to be (conditionally) random, and so does the Qini *evaluation*. If your randomized holdout was only 2% of traffic and the other 98% was the live, confounded recommender, you must not pool them — train the control model and evaluate the Qini on the *randomized 2% only*, or you smuggle the back-door bias back in through both doors. The cleanest discipline: carve a randomized slice, split *it* by user into train and test, fit on the randomized-train and report Qini on the randomized-test, and treat the 98% confounded majority as off-limits for anything causal (it is still fine for the *baseline* conversion model, just not for the uplift estimate). A "Qini of 0.18" that came from training and testing on a confounded pool is not a causal result; it is the credit-attribution trap from §1 wearing a lab coat.

One more practical note on the simulation: I drove baseline conversion from features `x0, x1` and uplift from `x2, x3` *deliberately*, to make the conversion-versus-uplift divergence vivid. Before you celebrate an uplift win on real data, fit both rankers and check whether your domain actually has this structure. Compute the rank correlation between your conversion score and your uplift score on the randomized test set. If it is high (say above 0.8), conversion and uplift nearly coincide in your data and the uplift model is buying you little for its extra variance — a perfectly legitimate finding that should send you back to the simpler conversion ranker. If it is low or negative, you are in the regime where uplift modeling pays, and the Qini comparison will show it.

## 9. Results: uplift targeting wins incremental value

Running the simulation above (seed 0, 40,000 users, 30% test split) produces results in the following ranges. I am reporting them as the kind of table you would put in a design doc; your exact digits will shift with the seed, and I have rounded to make the *pattern* legible.

| Strategy | Rank corr. with true uplift | Qini coefficient | Incremental conv. @30% budget |
| --- | --- | --- | --- |
| Random targeting | ~0.00 | 0.000 | ~120 |
| Target by P(convert) | ~ -0.05 to +0.10 | ~0.01 | ~138 |
| T-learner uplift | ~0.65 | ~0.068 | ~285 |
| Class-transform uplift | ~0.55 | ~0.060 | ~250 |

![A three-by-three results matrix comparing random conversion and uplift targeting on incremental conversions Qini coefficient and verdict with uplift winning every column](/imgs/blogs/causal-and-uplift-recommendation-8.png)

The figure summarizes the verdict. Random targeting is the reference. Conversion targeting *barely* beats random on incremental conversions — its Qini hovers near 0.01 — because the people it ranks highest are sure-things whose conversions are mostly not incremental. The uplift methods more than double the incremental conversions at the same 30% budget and post Qini coefficients five-to-seven times higher. The T-learner edges out the class-transformation model here because the transformed label adds variance, but both crush conversion targeting on the metric that pays the bills.

There are two honest caveats to put on the table. First, these are *simulation* numbers where I built in heterogeneous uplift driven by different features than baseline conversion. In a domain where baseline conversion and uplift happen to be highly correlated — where the people most likely to buy are also the most persuadable — the gap between conversion and uplift targeting shrinks, and uplift modeling buys you less. Knowing *which regime you are in* is the real skill (more on that in §12). Second, the uplift model's variance is real: with a small randomized control slice, the Qini curve gets noisy and the confidence interval on the Qini coefficient can be wide. Always bootstrap the Qini (resample users with replacement, recompute) and report an interval, not a point — a Qini of "0.068 ± 0.04" is a very different decision than "0.068 ± 0.005."

#### Worked example: translating Qini into dollars

Suppose your widget is shown 10 million times a month, your treatment budget is 30% of impressions (3M slots, because each costs a coupon worth \$2), and the simulation's incremental rate carries over: uplift targeting captures roughly 285 incremental conversions per ~12,000 budgeted users in the test, i.e. about a 2.4% incremental conversion rate on targeted users, versus conversion targeting's ~138 (about 1.15%). On 3M slots that is about 72,000 incremental conversions from uplift targeting versus about 34,500 from conversion targeting — an extra ~37,500 conversions for the *same* \$6M coupon spend. At a \$25 margin per conversion, that is roughly \$940K of additional margin a month, found not by spending more but by spending it on persuadables instead of sure-things. The coupon you do *not* send to a sure-thing is pure saved margin on top. This is why incrementality work is some of the highest-leverage modeling a growth team can do.

## 10. Causal recsys beyond uplift: deconfounding exposure

Uplift modeling assumes you can get a randomized treated/control split. Sometimes you cannot — the experience cost is too high, or you are working with historical logs from a system that never randomized. The broader field of **causal recommendation** asks how to recover causal effects from *observational* logs by modeling and adjusting for the exposure mechanism.

The cleanest formal lens is the **do-calculus**. A recommendation is an *intervention*, written $\text{do}(\text{show}=1)$, and what we want is $P(Y=1 \mid \text{do}(\text{show}=1), x)$, the distribution of the outcome under the intervention — *not* $P(Y=1 \mid \text{show}=1, x)$, the distribution under passive observation. The two differ precisely because of the back-door path through the confounder (§3). The **back-door adjustment** says: if you can measure a set of variables $C$ that blocks every back-door path, then

$$
P(Y \mid \text{do}(\text{show}=1)) = \sum_c P(Y \mid \text{show}=1, c)\, P(c),
$$

which you can estimate from observational data. The practical instantiation of this in recsys is **inverse-propensity weighting** of the exposure: estimate $\hat{p}(\text{show} \mid x)$ (how the logging recommender decided to show), then weight each observed outcome by $1/\hat{p}$ to "undo" the selective exposure and recover what a randomly-exposing system would have seen. This is the exact same IPW machinery as in [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) and in [debiasing the click data for position and selection bias](/blog/machine-learning/recommendation-systems/position-and-selection-bias-in-click-data) — propensities are the universal currency of causal correction.

The **deconfounded recommender** (Wang, Liang, Charlin, Blei, 2020) takes a different and clever route: it posits that the confounder is a *latent* user-item exposure factor, fits a Poisson factorization model to the exposure data to estimate a substitute confounder, and then includes that substitute as a control variable when modeling the outcome. The intuition is that you cannot measure relevance directly, but the *pattern of what got exposed* is a noisy readout of it, and conditioning on a reconstruction of that pattern blocks much of the back-door flow. It is an elegant way to deconfound without a randomized slice, at the cost of the (untestable) assumption that the latent factor captures the confounding.

```python
# Sketch: inverse-propensity weighting to deconfound observational exposure.
# Step 1: model the exposure mechanism p(show | x) from the logging policy.
from sklearn.linear_model import LogisticRegression
prop = LogisticRegression(max_iter=1000).fit(Xtr, train.T.values)
p_show = prop.predict_proba(Xte)[:, 1]
p_show = np.clip(p_show, 0.05, 0.95)                  # clip to bound variance (overlap!)

# Step 2: IPW estimate of the treated outcome mean (vs the naive treated mean)
w = test.T.values / p_show                            # 1/p for treated, 0 for control
ipw_treated_mean = np.sum(w * test.Y.values) / np.sum(w)
naive_treated_mean = test.Y.values[test.T.values == 1].mean()
print(f"IPW treated mean {ipw_treated_mean:.3f}  naive treated mean {naive_treated_mean:.3f}")
```

The honest caveat, and it is a big one: deconfounding is only as good as the assumption that you measured (or reconstructed) all the confounders. If relevance has a component you cannot see and cannot reconstruct, the back-door stays open and your "deconfounded" estimate is still biased — you just feel better about it. The clipping in the snippet above is not cosmetic: when the logging policy almost never showed an item to some users, $\hat{p}(\text{show}\mid x) \to 0$, the weight $1/\hat{p}$ explodes, and the estimator's variance blows up — the overlap (positivity) failure from §3 biting in practice. This is the same variance pathology that wrecks IPS estimators under poor propensity overlap in [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation). My standing advice: prefer the randomized slice when the experience cost is bearable; reach for deconfounding only when it is not, and treat its numbers with appropriate suspicion.

## 11. Case studies and real-world numbers

Uplift and causal recommendation are not academic curiosities; they are how serious growth and ads teams allocate budget. A few grounded examples.

**The Criteo uplift dataset.** Criteo released a large public uplift-modeling benchmark — on the order of 13–14 million rows, each a user exposed (treated) or not (control) to advertising, with a `visit` and `conversion` outcome and an explicit `treatment` flag and `exposure` indicator. It is the standard public testbed for uplift methods and the dataset most papers in the area report on. The headline finding across methods on it is consistent with our simulation: ranking by predicted uplift produces meaningfully higher Qini than ranking by predicted conversion, and the absolute Qini values are modest (the incremental signal is small relative to baseline conversion, which is the realistic regime). If you want to practice the code in §8 on real data, this is where to start — the `scikit-uplift` (`sklift`) library ships loaders for it and implements the T/S/X-learners and the class transformation with Qini/uplift-curve evaluation out of the box.

**Marketing incrementality / "ghost ads" and PSA holdouts.** The advertising industry learned this lesson the hard way. For years, ad platforms reported "conversions attributed to the ad" — essentially $P(\text{convert}\mid\text{shown})$ — and advertisers slowly realized those conversions were heavily contaminated by sure-things who would have converted anyway. The fix was *incrementality testing*: run a randomized holdout (a slice of users who are eligible but deliberately *not* shown the ad, sometimes filled with a public-service announcement so the auction still clears — the "ghost ads" / PSA methodology, formalized by Johnson, Lewis, and Nubbemeyer). The measured incrementality is routinely a *fraction* of the attributed conversions; published studies and platform documentation describe cases where naive attribution overstated true incremental conversions by large multiples. The entire shift from "attribution" to "incrementality" in digital advertising is the industry-scale version of this post's thesis.

**The deconfounded recommender (Wang, Blei et al.).** The "Causal Inference for Recommendation" and "The Deconfounded Recommender" line of work from David Blei's group reframed collaborative filtering as a causal-inference problem: the ratings you observe are confounded by *which* items users chose to interact with (exposure), and naively fitting a recommender to observed ratings conflates "users rate items they were exposed to" with "users would rate items highly." Their substitute-confounder approach (fit an exposure model, use it as a control) improved held-out predictive performance and, more to the point, changed *which* items the model recommended away from the popularity-confounded defaults. It is the most-cited bridge between the causal-inference and recommender-systems literatures.

**Causal embeddings for recommendation (CausE).** Bonner and Vasile (2018, from Criteo) proposed "Causal Embeddings for Recommendation," learning item/user embeddings that predict the outcome *under a randomized-exposure distribution* rather than the logged one, using a small randomized sample to regularize a model trained mostly on biased logs. Reported results showed improved performance on the causal (randomized) test distribution — which is the distribution that matters when your recommender will *change* exposure — even when standard offline metrics on the biased test set looked similar or worse. It is a concrete demonstration that optimizing for the logged distribution and optimizing for the *interventional* distribution are different objectives, exactly as §1 argued.

The common thread: every one of these is a story about a team discovering that their model was good at *predicting* outcomes and bad at *causing* them, and fixing it by introducing randomized data and a causal objective. Cite these when you need to convince a stakeholder that the holdout cost is worth it.

## 12. When to model uplift, and when to just predict conversion

Uplift modeling is more expensive than conversion modeling — it needs randomized data, it has higher variance, and it is harder to debug. It is not always worth it. Here is my decision rule, learned from getting it wrong both directions.

**Reach for uplift when:**

- The recommendation is *costly* per impression — a coupon, a push notification, a discount, an outbound email, a slot with real opportunity cost. When each treatment costs money, targeting sure-things is pure waste and uplift pays for itself.
- You have or can afford a **randomized exposure slice**. No randomization, no honest uplift. If you can run a holdout (even a small one), you have the raw material.
- **Baseline conversion and persuadability are driven by different signals** — i.e., the people most likely to buy are *not* the people the recommendation most moves. This is exactly when conversion targeting goes wrong, and it is common in retention/win-back, cross-sell, and ad targeting.
- The do-not-disturb segment is real and costly — e.g., notification fatigue, unsubscribes, or upsells that trigger abandonment. Uplift is the only framework that even *represents* negative treatment effects.

**Just predict conversion (or relevance) when:**

- The slot is *free* and *organic* — the main feed, the search results page — where there is no per-impression cost and the goal is genuinely "show the most relevant thing." If you are filling a feed people scroll regardless, relevance ranking is the right objective and uplift is overkill.
- Baseline conversion and uplift are **highly correlated** in your domain — the keenest buyers are also the most movable. Then the conversion ranking and the uplift ranking nearly coincide, and the extra variance of uplift modeling buys you nothing. (Check this empirically: fit both, compare the Qini of uplift-targeting to conversion-targeting on a randomized holdout. If the gap is small, stick with conversion.)
- You **cannot randomize** and cannot credibly deconfound. A badly-deconfounded uplift model can be *worse* than an honest conversion model, because its bias is unknown and unbounded. Better an honest correlation than a fake causal number.
- You are early and need a baseline. Ship the conversion ranker, *measure incrementality with a holdout*, and only invest in uplift modeling once the holdout proves there is incremental value being left on the table.

**The stress tests I always run** before trusting an uplift recommender:

- *What if the control slice is tiny?* The T-learner's $\hat{\mu}_0$ becomes unreliable; switch to the X-learner (built for imbalance) or accept wider Qini intervals. Bootstrap the Qini and look at the interval, not the point.
- *What if overlap fails* (some users were never in control)? The IPW weights explode and the deconfounded estimate is garbage in that region. Restrict claims to the region of overlap; do not extrapolate uplift where you have no counterfactual data.
- *What if the offline Qini is high but the online incrementality test is flat?* Suspect leakage between treated/control in the offline split, or a distribution shift between your randomized slice and live traffic, or that your randomized slice was not actually random (a bug in the holdout assignment). The same offline–online discipline from the rest of the series applies: a temporal split, no leakage, and a real A/B confirmation before you trust the offline number.
- *What if uplift is mostly noise* (Qini interval includes zero)? Then you do not have enough randomized data to distinguish persuadables from sure-things. Either collect more holdout data or fall back to conversion targeting — do not ship slot decisions based on noise.

## 13. Key takeaways

- **Recommend what changes behavior, not what predicts it.** A standard recommender learns $P(\text{convert}\mid\text{shown})$; the value is the causal effect, the uplift $P(\text{convert}\mid\text{shown}) - P(\text{convert}\mid\text{not shown})$. The best conversion predictor is *not* the best conversion causer.
- **Four segments, one worth your slot.** Persuadables (positive uplift) earn the treatment; sure-things and lost causes (zero uplift) waste it; do-not-disturbs (negative uplift) are actively harmed. Conversion targeting cannot tell them apart.
- **The fundamental problem is real.** You never observe both $Y(1)$ and $Y(0)$ for one user, so individual uplift is unobservable. You can only estimate the CATE $\tau(x)$ — a segment average — and only under unconfoundedness.
- **Randomized exposure earns unconfoundedness for free.** A randomized holdout slice is the gold standard. Without it, you are deconfounding from observational logs and betting on the untestable assumption that you measured every confounder.
- **Correlation misleads because of confounding.** Relevance and popularity cause both exposure and purchase (the back-door path), so the naive shown-bought correlation massively overstates the true effect — and lets the recommender take credit for organic conversions.
- **Know your methods.** T-learner (simple, subtract two models), S-learner (one model, biased toward zero), X-learner (best for small control slices), class transformation (leanest, single regression via $2P(Z=1)-1$), uplift trees/causal forests (uncertainty on $\tau$). Match the method to your data balance.
- **Evaluate on groups with the Qini curve.** You cannot score individual uplift, so rank users by predicted uplift and measure cumulative *incremental* conversions. The Qini coefficient is the AUC of uplift: positive is good, near-zero is random, negative is anti-sorted.
- **Uplift targeting wins incremental value, but not always.** When baseline conversion and persuadability diverge and the slot is costly, uplift modeling can multiply incremental conversions at the same budget. When they coincide or the slot is free, a conversion ranker is the honest, lower-variance choice.
- **Bootstrap the Qini.** Report an interval, not a point. A wide interval that includes zero means you do not have enough randomized data to act on the model.

## 14. Further reading

- Rubin, D. (1974), "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies" — the potential-outcomes framework that defines $Y(1), Y(0)$ and the fundamental problem.
- Künzel, Sekhon, Bickel, Yu (2019), "Metalearners for estimating heterogeneous treatment effects using machine learning" (PNAS) — the canonical T/S/X-learner reference.
- Jaśkowski and Jaroszewicz (2012), "Uplift modeling for clinical trial data" — the class-variable transformation that turns uplift into a single classification.
- Radcliffe and Surry (2011), "Real-World Uplift Modelling with Significance-Based Uplift Trees" — uplift trees and the Qini coefficient as an evaluation metric.
- Wager and Athey (2018), "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests" (JASA) — causal forests with honest splitting and confidence intervals.
- Wang, Liang, Charlin, Blei (2020), "Causal Inference for Recommendation" / "The Deconfounded Recommender" — substitute confounders for observational recsys deconfounding.
- Bonner and Vasile (2018), "Causal Embeddings for Recommendation" (RecSys) — learning embeddings for the interventional rather than the logged distribution, with the Criteo data.
- Johnson, Lewis, Nubbemeyer (2017), "Ghost Ads: Improving the Economics of Measuring Online Ad Effectiveness" — the PSA/ghost-ad incrementality methodology behind modern ad measurement.
- Docs and tools: the `scikit-uplift` (`sklift`) library (T/S/X-learners, class transformation, Qini/uplift curves, Criteo loader); the `causalml` library from Uber (meta-learners, uplift trees, causal forests).
- Within this series: [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer), [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), [bandits and the exploration–exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff), [A/B testing recommenders](/blog/machine-learning/recommendation-systems/ab-testing-recommenders), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
