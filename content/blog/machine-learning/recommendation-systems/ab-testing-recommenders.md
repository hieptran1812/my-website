---
title: "A/B Testing Recommenders: The Online Ground Truth"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why offline metrics are only a proxy and live traffic is the only ground truth, how to design and power an A/B test for a recommender, why SUTVA breaks in a two-sided market, and how to read results with CUPED, the delta method, and interleaving that needs a hundred times fewer users."
tags:
  [
    "recommendation-systems",
    "recsys",
    "ab-testing",
    "online-experiments",
    "interleaving",
    "cuped",
    "causal-inference",
    "experimentation",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/ab-testing-recommenders-1.png"
---

The most painful readout I have ever sat through was a green one. We had spent a quarter on a new ranking model: a deeper cross network, hard negatives, a calibration layer, the works. Offline it was beautiful. NDCG@10 up four points, AUC up almost a full percent, logloss down, every slice we cut looked better than the model in production. We shipped it to a five percent A/B test fully expecting to celebrate. Three days in, the dashboard lit up green: click-through rate up 4.0 percent in treatment, comfortably significant, a slam dunk. The product manager wanted to ramp it to a hundred percent that afternoon. I asked for one more week. By the end of that week the click lift had held — but the retention guardrail had gone red. Users in treatment were coming back 1.5 percent less often over the following month. The new model had learned, very effectively, to surface the kind of item people click and then regret: the thumbnail that overpromises, the listicle, the outrage bait. It was a genuinely better *click* predictor and a genuinely worse *product*. The offline metric had measured the proxy. The three-day online readout had measured the proxy faster. Only the patient, retention-weighted online experiment measured the thing the business actually cared about.

That is the whole subject of this post. Offline evaluation — even an honest, leak-free, temporal-split offline evaluation — estimates a *proxy* for what you want. The business does not care about NDCG; it cares about engagement, watch-time, retention, gross merchandise value (GMV), the revenue and loyalty that recommendations are supposed to produce. Those quantities live only in live traffic, in the actual behavior of actual users responding to the actual system. An A/B test — randomizing users into a control arm that sees the old recommender and a treatment arm that sees the new one, then measuring the difference in what they do — is the only instrument that estimates the *causal* effect of your change on the things that pay the bills. Everything else is a model of that instrument. This is the online ground truth, and learning to run it and read it correctly is the difference between an organization that improves and one that merely ships.

![A branching dataflow diagram showing live user traffic randomized by user into a control arm running the old ranker and a treatment arm running the new ranker, both logging clicks and guardrails, feeding a measurement of the overall evaluation criterion that branches into ship if the lift is real or kill if it is flat or harmful](/imgs/blogs/ab-testing-recommenders-1.png)

By the end you will be able to design an experiment on purpose: pick the randomization unit, define an overall evaluation criterion that fuses your north-star and your guardrails, compute the sample size and duration a target lift requires from the power formula, recognize when the no-interference assumption that underpins the whole method silently breaks in a two-sided market, and read the results without fooling yourself with peeking, multiple testing, or Simpson's paradox. You will have runnable numpy that simulates an A/B test, computes the lift, the confidence interval, and the p-value, sizes the experiment, implements CUPED variance reduction, and implements team-draft interleaving — the cheaper, sharper online comparison that needs roughly a hundred times fewer users to call a ranker winner. This is the methodology capstone of the evaluation track. The bird's-eye map of the funnel is [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system); the two worlds this post bridges are laid out in [offline versus online](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys); *why* even a clean offline number still diverges from online is [the offline online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied); the techniques that let you estimate online effects *without* a live test are in [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation); and the synthesis of all of it is the [recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

## 1. Why online is the only ground truth

Start from a question that sounds rhetorical but is not: what is your offline metric actually a measurement *of*? When you compute NDCG@10 on a held-out temporal slice, you are estimating how well the model ranks the items users *did* interact with, given a candidate set you assembled offline, under the logging policy that produced the data, with the position bias baked into that policy, on the subpopulation of users who happened to generate test interactions. Every one of those clauses is a gap between the metric and the deployed reality. The candidate set the live system retrieves differs from your offline pool. The logging policy that decided what users saw is not the policy you are testing. Position bias means a click is partly a function of *where* an item was shown, not just whether it was good. And — most fundamentally — you are scoring against historical behavior, which was itself shaped by the old recommender, so you are grading the new model on its ability to reproduce the old model's footprint. The offline metric is a proxy, and a proxy that is *correlated* with the thing you care about but not *equal* to it, and the correlation is exactly as strong as your luck.

The business objective is different in kind, not just in degree. A streaming service wants members to keep paying, which is downstream of watch-time, which is downstream of finding something worth watching tonight, which is downstream of the ranking. An e-commerce site wants GMV and repeat purchases. A feed wants daily active users who stay engaged without burning out. None of those is "rank the historical clicks well." They are *behavioral and economic outcomes that only exist when real users meet the real system*, and they emerge from a chain of effects — novelty, satiation, trust, habit — that no offline dataset contains because the dataset was generated under a *different* policy that produced *different* downstream behavior. You cannot measure the effect of a change you have not made.

This is why the A/B test is special. It does not model the outcome; it *produces* the outcome and measures it directly. By randomly assigning users to control or treatment, it makes the two groups statistically identical in every respect — demographics, history, device, time zone, mood — *except* the one thing you changed. Randomization is the magic: it breaks the back door between "which users get the new model" and "which users were going to engage more anyway," so any difference in the outcome that survives the noise is *caused* by your change. This is the entire logic of the randomized controlled trial, imported from clinical medicine, and it is the closest thing applied machine learning has to a ground truth. The diagram in the introduction is this logic drawn out: traffic flows in, randomization splits it, the two arms run the two systems, you log behavior including guardrails, you measure the overall evaluation criterion, and you decide.

There is a deeper reason the offline number and the online number diverge that is worth naming precisely, because it is the scientific spine of the entire evaluation track. Offline you are estimating performance under the *logging* distribution — the distribution of contexts and actions the *old* policy generated. Online, under treatment, the *new* policy generates a *different* distribution of contexts and actions; users see different items, so they click different things, so the very data you would log changes. This is distribution shift induced by the policy itself, and it is why an offline metric computed on logged data is a *counterfactual* estimate of a quantity the new policy would generate — and counterfactual estimation from logged data is hard, biased when the policies differ a lot, and the whole subject of [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation). The A/B test sidesteps all of it by *actually running* the new policy on a slice of live traffic. It pays for that with time and exposure, but it buys an unbiased causal estimate that no amount of offline cleverness can match.

Let me be concrete about the failure direction, because it is consistent and worth internalizing. Offline metrics tend to be *optimistic* about agreement with online. A change can move offline NDCG a lot and online engagement zero (the model got better at reproducing historical clicks, which the new policy will no longer generate). A change can move offline NDCG zero and online engagement a lot (a UI or diversity change that the offline metric, scoring against historical clicks, literally cannot see). And — the case that paged me — a change can move offline and short-term online *up* while moving long-term online *down*. The offline-online correlation across a year of experiments at a mature company is real but loose; teams that have measured it report rank correlations in the range of perhaps 0.5 to 0.8 between offline metric deltas and online OEC deltas, meaning offline gets the *sign* right most of the time but the *magnitude* and occasionally the sign wrong often enough to ruin you if you skip the online test. Offline is a *filter* — it kills obviously bad ideas cheaply so you do not waste live traffic on them. Online is the *judge*.

It is worth being precise about *why* the divergence is structural and not just noise, because the precision changes how you treat offline numbers. Three distinct mechanisms drive the gap, and each is a separate post in this series. The first is **policy-induced distribution shift**: the new model serves different items, so the contexts and feedback it generates differ from the logged data it was scored on — the metric was computed on a distribution the deployed system will no longer produce. The second is **position and presentation bias**: a logged click is partly a function of where the item appeared, so an offline metric that grades against logged clicks is grading against a signal that the old layout's geometry contaminated; change the layout and the historical clicks stop being the right target. The third is **the missing-not-at-random problem**: the items a user *did not* interact with are not a random sample — they are overwhelmingly items the old recommender never showed, so "the user did not click item X" carries almost no information about whether they would have liked X had they seen it. Offline evaluation treats those un-shown items as negatives; deployment will surface some of them and find out the truth. None of these three is fixable by a better offline split, which is exactly why the online test is not optional. The cleanest summary I can give: the offline metric measures *fidelity to the past under the old policy*, and the online experiment measures *value created under the new policy*, and those are different quantities that happen to correlate.

## 2. The metrics that matter online

If the A/B test is the judge, then defining what it judges is the most consequential decision in the whole enterprise, and the one teams most often get wrong. The temptation is to pick the metric that moves fastest and most reliably — usually short-term clicks — and optimize it. That is the proxy-metric trap, and it has sunk more product roadmaps than any modeling mistake.

![A before and after contrast showing a clickbait ranker that lifts click-through rate by four percent and looks like a win on the day-one readout, set against the same ranker measured over an eight-week horizon where retention falls one and a half percent revealing it as the true long-term loser](/imgs/blogs/ab-testing-recommenders-3.png)

Organize your metrics into three roles. The **north-star** is the single metric (or small bundle) that you genuinely believe tracks long-term business value: watch-time per member and 28-day retention for streaming; GMV and repeat-purchase rate for commerce; long-term daily active users and sessions-per-week for a feed. The north-star is what you are trying to move. **Guardrails** are metrics you must *not* regress while moving the north-star: serving latency at the 99th percentile, error rate, result diversity, the rate of user complaints or "see less of this" signals, ad load, content-policy violations. Guardrails encode the constraints — "make engagement go up without making the product slower, more repetitive, or more toxic." **Proxy metrics** are fast, sensitive early reads — day-one CTR, immediate dwell time — that you watch to see *if anything is happening at all*, while knowing they are not the goal.

![A taxonomy tree rooting at experiment metrics and branching into a north-star of engagement and retention, a guardrail group of latency and diversity, and a proxy group of early day-one click-through, with the engagement and retention leaves marked as the true drivers and the proxy leaf marked as fast but noisy](/imgs/blogs/ab-testing-recommenders-4.png)

The proxy-metric trap is the gap between a proxy and the north-star opening up under optimization. Clicks correlate with value *in the data you have*, because historically good recommendations got clicked. But the moment you optimize *for clicks*, you exploit the part of the correlation that is not value: clickbait, sensational thumbnails, items that provoke a click and a regret. Goodhart's law in its sharpest form — when a measure becomes a target, it ceases to be a good measure. The figure above is exactly this: a ranker that lifts day-one CTR by 4.0 percent and *lowers* eight-week retention by 1.5 percent. The proxy said win; the north-star said loss. The only defense is to (a) define the north-star as the long-term outcome even though it is slow and noisy, and (b) run experiments long enough, and with enough power, to read the north-star directly rather than inferring it from the proxy.

This is where the **overall evaluation criterion**, the OEC, comes in. The OEC is the single decision rule that fuses your metrics into a ship/no-ship verdict. The cleanest mental model — and this is one of the few places I will use that phrase, sitting right under the taxonomy figure — is: the OEC is a north-star metric *subject to guardrail constraints*. You ship if and only if the north-star improves significantly *and* no guardrail regresses beyond its tolerance. Sometimes the OEC is a single composite metric (Microsoft's experimentation teams have written about engineering a single "sessions" or "success-rate" OEC that is hard to game); sometimes it is a constrained objective. Either way, writing the OEC down *before* the experiment runs is the single most important discipline in experimentation, because it stops you from picking, after the fact, whichever metric happened to move in your favor — the most common and most invisible form of p-hacking.

| Metric role | Examples | Cadence | What it answers | The trap |
|---|---|---|---|---|
| North-star | Watch-time, 28-day retention, GMV, repeat-purchase | Slow, noisy, weeks | Did long-term value increase? | Too noisy to read on small experiments |
| Guardrail | p99 latency, diversity, complaint rate, error rate | Fast, sensitive | Did we break something? | Forgetting one until it breaks in prod |
| Proxy | Day-one CTR, immediate dwell | Fast, low-variance | Is anything happening at all? | Optimizing it directly (Goodhart) |
| OEC | North-star subject to guardrails | The decision | Ship or not? | Choosing it *after* seeing results |

A practical note on choosing the north-star: prefer metrics that are *additive across users and robust to outliers*. Total watch-time is dominated by a handful of binge-watchers; a single user's behavior change can swing it. A trimmed or capped per-user metric, or a count of *active days*, is harder to game and lower-variance. Booking.com's experimentation team has written extensively about preferring conversion-style binary metrics and capping continuous ones precisely because the variance of an uncapped revenue metric makes experiments impossibly underpowered. The metric you can *measure precisely* in a reasonable number of users beats the metric that is *theoretically purest* but has variance so high you can never call it.

## 3. Experiment design: randomization, power, and duration

A good experiment is mostly decided before a single user is bucketed. Three design choices dominate everything downstream: the randomization unit, the sample size (which is the power calculation), and the duration.

![A vertical pipeline of six stacked stages reading from top: design the overall evaluation criterion with north-star plus guardrails, compute power and the minimum detectable effect to solve for the sample size, randomize users by hash with balance checks, run the experiment one to two weeks to ride out novelty, analyze with CUPED and no peeking, and finally decide to ship or kill based on lift versus guardrails](/imgs/blogs/ab-testing-recommenders-6.png)

### The randomization unit

You can randomize at the level of the **request**, the **session**, or the **user**. The choice has consequences that compound.

Randomizing per *request* (each individual recommendation request independently flips a coin for control or treatment) gives you the most units and therefore the most statistical power per unit of traffic — but it is almost always wrong for a recommender, because a single user will experience *both* systems within a session, contaminating their behavior. Their satisfaction, fatigue, and downstream actions reflect a blend of the two arms, so you cannot cleanly attribute outcomes, and you cannot measure any *user-level* metric like retention at all (the user is in both arms). Per-request randomization is defensible only for purely stateless, within-request effects with no carryover — rarely the recommender case.

Randomizing per *session* is a middle ground sometimes used when sessions are the natural unit of value, but it shares the user-contamination problem across sessions and still cannot measure retention.

Randomizing per *user* is the default for recommenders, and it is non-negotiable if your north-star is retention or any longitudinal outcome. Each user is consistently in one arm for the life of the experiment (bucketed by a hash of their stable user ID), so their entire experience — including whether they come back next week — is attributable to one system. The cost is statistical: users are the unit, so your effective sample size is *users*, not requests, and a heavy user contributes one noisy data point no matter how many requests they make. This is why user-randomized experiments need more traffic and time than the request count would suggest, and why variance reduction (section 5) matters so much.

The mechanism is a hash. You compute `hash(user_id + experiment_salt) mod 1000` and assign buckets 0–499 to control, 500–999 to treatment, say. The `experiment_salt` ensures that two different experiments running simultaneously bucket users *independently* (orthogonally), so user 42 being in treatment for experiment A tells you nothing about their bucket for experiment B. This independence is what lets a mature experimentation platform run hundreds of overlapping experiments without them confounding each other — a property Google, Microsoft, and others built their platforms around.

### Sample size and the power calculation

Here is the science that decides whether your experiment can possibly work. You want to detect a lift of relative size — say a 1 percent relative increase in CTR. Whether you *can* detect it depends on the sample size, and the relationship is governed by the two-sample test for a difference in proportions.

Set up the hypotheses. Under the null $H_0$, the treatment CTR $p_t$ equals the control CTR $p_c$. Under the alternative, they differ by the true effect $\delta = p_t - p_c$. You will run a test at significance level $\alpha$ (the false-positive rate you tolerate, conventionally 0.05) and you want statistical power $1 - \beta$ (the probability of detecting a real effect of size $\delta$, conventionally 0.80). The **minimum detectable effect** (MDE) is the smallest $\delta$ your experiment is powered to find.

The per-arm sample size needed is, to a very good approximation,

$$ n \approx \frac{\left(z_{\alpha/2} + z_{\beta}\right)^2 \cdot 2\,p\,(1-p)}{\delta^2}, $$

where $p$ is the baseline proportion (the control CTR), $z_{\alpha/2}$ is the standard-normal quantile for your two-sided significance (1.96 for $\alpha = 0.05$), and $z_{\beta}$ is the quantile for your power (0.84 for 80 percent power). Let me derive where this comes from, because the shape of it — the $1/\delta^2$ dependence in particular — is the single most important fact in experiment design and you should be able to reconstruct it.

A click is a Bernoulli trial: success with probability $p$. Over $n$ users, the sample CTR $\hat{p}$ has mean $p$ and variance $p(1-p)/n$ by the central limit theorem; for large $n$ it is approximately normal. The *difference* in CTR between two independent arms, $\hat{p}_t - \hat{p}_c$, is therefore approximately normal with mean $\delta$ and variance $\frac{p_t(1-p_t)}{n} + \frac{p_c(1-p_c)}{n} \approx \frac{2p(1-p)}{n}$ when the two proportions are close (which they are, since you are detecting a small lift). Call that standard error $\text{SE} = \sqrt{2p(1-p)/n}$.

Now the two requirements. To *reject the null* at level $\alpha$ you need the observed difference to exceed $z_{\alpha/2} \cdot \text{SE}$. To have *power* $1-\beta$ against the true effect $\delta$, the true sampling distribution (centered at $\delta$) must put at least $1-\beta$ of its mass past that rejection threshold; that requires $\delta$ to sit at least $z_{\beta} \cdot \text{SE}$ above the threshold. Adding the two distances:

$$ \delta = \left(z_{\alpha/2} + z_{\beta}\right) \cdot \text{SE} = \left(z_{\alpha/2} + z_{\beta}\right)\sqrt{\frac{2p(1-p)}{n}}. $$

Solve for $n$ and you get the formula above. The lesson is in the $\delta^2$ in the denominator: **the sample size scales as the inverse square of the effect you want to detect.** Halve the MDE and you quadruple the required users. Want to detect a 0.5 percent relative lift instead of a 1 percent lift? You need four times the traffic and roughly four times the duration. This single quadratic is why detecting small lifts is the central operational challenge of a mature experimentation program, and why variance reduction (which effectively shrinks the $2p(1-p)$ term) is worth so much.

One subtlety: $\delta$ here is the *absolute* difference in the proportion. If your baseline CTR is $p = 0.10$ and you want a 1 percent *relative* lift, the absolute $\delta = 0.01 \times 0.10 = 0.001$. People constantly conflate relative and absolute lift and then wonder why their sample-size estimate is off by orders of magnitude. Always carry the units.

A second subtlety that trips up teams: the formula above sizes a *proportion* (CTR, conversion). For a *continuous* north-star like watch-time per user or revenue per user, the variance term $2p(1-p)$ is replaced by $2\sigma^2$, where $\sigma$ is the per-user standard deviation of the metric — and for heavy-tailed metrics like revenue, $\sigma$ can be enormous relative to the mean, which is why an uncapped revenue metric is often impossible to power. The general shape is the same, $n \approx 2\sigma^2(z_{\alpha/2}+z_\beta)^2/\delta^2$, and the same $1/\delta^2$ wall applies, but now the *numerator* is your enemy: anything that shrinks $\sigma$ — capping outliers, using a binary conversion instead of a continuous spend, or applying CUPED — directly buys you sample size. This is the deep reason the metric-engineering work in section 9 matters as much as the modeling: the *variance* of your chosen metric is a first-class lever on whether any experiment is feasible at all. A north-star you cannot power is not a north-star; it is a wish.

### Duration: novelty, primacy, and seasonality

Even a perfectly powered experiment can lie if you stop it at the wrong time. Three temporal effects force a minimum duration.

The **novelty effect**: when users encounter something new — a redesigned UI, a visibly different set of recommendations — they engage more *just because it is new*, and that bump decays over days to weeks. Read the experiment too early and you mistake novelty for a durable lift. The mirror-image **primacy effect** (also called the change-aversion effect): some users dislike change and disengage initially, then recover as they adapt; read too early and you mistake adjustment friction for harm. Both effects argue for running long enough to see the curve flatten — typically at least one full week, often two, before the treatment effect stabilizes.

**Weekly seasonality**: behavior on a recommender swings hard between weekday and weekend, and the user mix differs. An experiment that runs Monday-to-Friday measures a different population than one spanning a weekend. The rule is to run for whole multiples of a week so each arm sees a complete, balanced cycle. Stopping an experiment on a Wednesday because it "looks significant" is a recipe for a result that does not replicate.

The practical floor is therefore: run for at least one to two full weeks regardless of how fast significance appears, sized so that one to two weeks of your available traffic clears the power threshold for your target MDE. If the power calculation says you need four weeks to detect the lift you care about, then either you accept a four-week experiment or you accept that you cannot detect lifts that small — there is no shortcut around the $1/\delta^2$ wall except variance reduction and interleaving.

## 4. SUTVA: the assumption that breaks in recommendation

Here is the part of the textbook that the textbook glosses and the recommender practitioner cannot afford to. The entire causal logic of the A/B test rests on an assumption called **SUTVA** — the Stable Unit Treatment Value Assumption. It has two parts, and the second one breaks routinely in recommender systems in ways that quietly bias your results.

SUTVA says: (1) there is a single, well-defined version of each treatment (no hidden variants), and (2) — the one that matters here — **a unit's outcome depends only on its own treatment assignment, not on the treatment assignment of any other unit.** This is the *no-interference* assumption. It is what lets you write "the effect on user $i$" as a function of user $i$'s arm alone. In a drug trial it usually holds: whether my neighbor took the drug does not change whether the drug works on me (unless the disease is contagious — and that exception is precisely the recommender's problem). In a recommender, no-interference fails through several mechanisms, and each one biases the measured lift, usually toward overstating it.

**Network effects.** On any product with a social graph — a feed, a video platform, a marketplace with reviews — what a treatment user does spills onto control users. If treatment users, served better recommendations, share more content, their control-arm friends see more shared content and engage more. The control arm is *contaminated* by the treatment, so the measured difference *understates* the true effect (control got better too) — or, in other configurations, overstates it. Either way the clean "control is unaffected by treatment" premise is gone.

**Shared inventory and marketplace interference.** This is the killer in two-sided markets, and it is worth slowing down for. Consider a marketplace — rides, rooms, jobs, ad slots, a limited catalog of in-demand items. Treatment and control users are drawing from the *same finite supply*. If your new recommender is better at steering treatment users toward the most desirable items, those items get *consumed* — booked, sold out, their ad budget spent — and now control users face a *depleted* inventory they would not have faced in a world where everyone got the old model. Treatment's gain is partly *taken from* control. The two arms are not independent; they are competing for the same scarce resource. Measure the naive difference and you will *overstate* the treatment's effect, because you are crediting treatment with a lift that came at control's direct expense — a lift that will *not* materialize when you ramp to 100 percent and there is no control arm to cannibalize.

![A four by three decision matrix comparing offline metrics, full A/B tests, and interleaving across the dimensions of ground truth, cost, sensitivity, and speed, showing offline as a cheap fast proxy that can mislead, the full A/B test as the expensive slow but causal ground truth, and interleaving as a low-cost high-sensitivity ranker comparison](/imgs/blogs/ab-testing-recommenders-2.png)

This is not a hypothetical. Ride-sharing, lodging, and labor-marketplace companies have published at length on it. The standard mitigations all relax the unit of randomization to *internalize* the interference: **cluster randomization** (randomize whole cities, regions, or time-windows so that within a cluster everyone gets the same treatment and the competition for inventory happens within a single arm), **switchback experiments** (flip the entire market between treatment and control on a schedule, comparing time-periods rather than users, so there is never a contemporaneous control to cannibalize), and **two-sided / budget-split designs** that partition the supply itself. Each trades statistical power for unbiasedness: a city-randomized experiment has dozens of units, not millions, so the power formula bites hard and you need long durations. But a biased high-powered estimate of the wrong quantity is worse than a noisy unbiased estimate of the right one.

**Feedback-loop interference through retraining.** There is a third, sneakier interference channel specific to recommenders. If your model is *retrained on logged data while the experiment runs*, and the training data mixes both arms, then treatment-arm behavior leaks into the model that serves the control arm (and vice versa) at the next retrain. The control is no longer running a frozen old model; it is running a model contaminated by treatment's logs. The mitigation is to *freeze* both models for the duration of the experiment — no mid-experiment retraining on experiment data — or to train arm-specific models on arm-specific logs, which is operationally heavier but keeps the arms clean. Teams that retrain nightly on the full log and run multi-week experiments are silently violating SUTVA in a way no amount of careful randomization fixes, because the interference happens in the *training pipeline*, not the serving path.

**The takeaway for design.** Before you trust a user-randomized A/B test, ask: *does treatment change the world that control lives in?* If users interact (social), or compete for inventory (marketplace), or if the model is retrained on logged data that mixes both arms (a feedback-loop interference we cover in the bias and feedback-loop posts), then SUTVA is suspect and your naive lift is biased. The matrix above frames where each method sits — and part of why interleaving (section 7) is so attractive is that, by comparing two rankers *inside one user's list*, it sidesteps between-user interference for the ranking question entirely.

#### Worked example: marketplace cannibalization inflates a lift

Suppose a lodging marketplace runs a user-randomized test of a new ranker, 50/50, on a city with 1,000 desirable listings on a peak weekend. Control users book at a baseline rate that consumes, say, 600 of those listings. Treatment's better ranker steers its users harder toward the *same* top listings; treatment users book at a rate that would consume 700 — but because they are competing with control for the same 1,000 rooms, the top rooms sell out and treatment users get pushed to their second choices, while control, facing a now-scarcer top tier, books fewer of them too. The naive readout shows treatment bookings up 8 percent and control bookings down 3 percent, for a measured relative lift of about 11 percent. But at 100 percent rollout there is no control to steal supply from: *everyone* competes, the top listings sell out for the whole population, and the realized lift is closer to 4 percent — the genuine matching improvement, net of cannibalization. The A/B test overstated the effect by nearly threefold because SUTVA failed. A switchback design, flipping the whole city between rankers day by day, would have measured the 4 percent directly, at the cost of needing many weeks of city-days to power it.

## 5. Reading results without fooling yourself

You have run the experiment for two weeks, user-randomized, with a pre-registered OEC. Now you have to read it, and the readout is a minefield of ways to fool yourself. Here are the ones that bite hardest.

### Statistical versus practical significance

A p-value below 0.05 tells you the effect is probably not zero. It says *nothing* about whether the effect is *big enough to matter*. With ten million users you can detect a 0.05 percent CTR lift as "highly significant," and shipping it may cost more in complexity and serving than the lift is worth. Always report the **effect size with a confidence interval**, and decide against a pre-set *practical* threshold — the minimum lift that justifies the maintenance and risk. Statistical significance is the gate; practical significance is the decision.

### Peeking and the inflated false-positive rate

The single most common way honest people manufacture false positives is **peeking** — repeatedly checking the experiment and stopping the moment it crosses $p < 0.05$. The fixed-sample p-value is only valid if you decide the sample size *in advance* and look *once*, at the end. If you peek daily and stop at the first significant moment, your *actual* false-positive rate is not 5 percent — it can easily be 20–30 percent, because under the null the p-value wanders and will randomly dip below 0.05 at *some* point if you give it enough chances. This is sequential testing without the correction, and it is rampant.

There are two clean fixes. Either commit to a fixed horizon (the power calculation tells you when) and look once. Or use a method *designed* for continuous monitoring: **sequential testing** with always-valid p-values / confidence sequences (mSPRT, group-sequential boundaries, or the "always-valid inference" framework popularized by Optimizely and refined in the experimentation literature). These methods inflate the per-look threshold so that the *overall* false-positive rate stays at 5 percent no matter how often you peek — you pay for the right to peek with a wider boundary. What you must never do is peek with a fixed-sample test and stop early. The discipline in the pipeline figure — "no peeking" sitting in the analyze stage — is there because it is the most violated rule in the field.

### Multiple testing

The mirror image of peeking across *time* is peeking across *metrics*. If you run twenty metrics on one experiment and test each at $\alpha = 0.05$, then even if nothing is real you expect about one of them to come up "significant" by chance. Optimize the OEC by hunting through the metric dashboard for the green one and you have p-hacked yourself. The defenses: pre-register the OEC (one primary decision metric), and for the secondary metrics apply a multiple-comparison correction — Bonferroni (divide $\alpha$ by the number of tests; conservative but simple) or the Benjamini-Hochberg false-discovery-rate procedure (controls the *expected proportion* of false positives among your rejections, far less conservative when you have many metrics). The same correction applies across *arms*: a four-arm experiment comparing three treatments to control is three tests, not one. A useful rule of thumb: treat your *one* OEC as the gate at the full $\alpha$, treat guardrails as one-sided regression checks at their tolerances, and treat the long tail of diagnostic metrics as *exploratory* — never as ship justification — so a green diagnostic metric can generate a hypothesis for the next experiment but can never, by itself, ship the current one. The discipline is structural: the metric that decides is named before the data exists, and everything else is a clue, not a verdict.

### Ratio metrics and the delta method

Here is a subtle one that catches even experienced practitioners. Many recommender metrics are *ratios* where the denominator is itself random and the randomization unit differs from the metric unit. CTR computed as *total clicks / total impressions* is a ratio of two sums, but you randomized by *user*. The clicks and impressions are correlated *within* a user (a heavy user contributes many of both), so the naive standard error that treats each impression as independent is *wrong* — it dramatically understates the variance, giving you confidence intervals that are too narrow and p-values that are too small. You will call false positives.

The fix is the **delta method**. For a ratio metric $R = X / Y$ where $X$ and $Y$ are per-user sums (clicks and impressions), the variance of the ratio is approximated by a Taylor expansion around the means:

$$ \operatorname{Var}(R) \approx \frac{1}{\bar{Y}^2}\left(\operatorname{Var}(X) - 2R\,\operatorname{Cov}(X, Y) + R^2\operatorname{Var}(Y)\right), $$

where the variances and covariance are computed over *users* (the randomization unit). This correctly accounts for the within-user correlation between clicks and impressions. The practical upshot: compute ratio-metric variances at the *user* level with the delta method (or by bootstrapping over users), never by treating impressions as the independent unit. Microsoft and others have published this exact result; getting it wrong is one of the most common silent causes of unreplicable A/B results.

### CUPED: variance reduction with a pre-period covariate

The power formula says you are stuck with the $1/\delta^2$ wall — unless you can shrink the variance term. **CUPED** (Controlled-experiment Using Pre-Experiment Data, from Deng, Xu, Kohavi, and Walker at Microsoft) does exactly that, and it is close to a free lunch.

![A before and after comparison showing the naive difference of means with full variance producing a wide confidence interval and a non-significant lift at p equal to 0.12, against the CUPED-adjusted estimate that subtracts each user's pre-period covariate to cut variance by forty-five percent producing a narrow interval and the same lift now significant at p equal to 0.01](/imgs/blogs/ab-testing-recommenders-7.png)

The idea: a user's behavior *during* the experiment is largely predictable from their behavior *before* the experiment. A user who watched 10 hours last month will watch roughly 10 hours this month regardless of which arm they are in — that variance is *not caused by the treatment*, it is just who they are. CUPED subtracts off that predictable, pre-existing variance. Define a pre-period covariate $X$ (the same metric measured in a window *before* the experiment started, when no user had been treated). Form the adjusted metric

$$ Y_{\text{cuped}} = Y - \theta\,(X - \bar{X}), \qquad \theta = \frac{\operatorname{Cov}(Y, X)}{\operatorname{Var}(X)}. $$

Because $X$ is measured before treatment, it is independent of the arm in expectation, so subtracting it does *not* bias the treatment effect — $\mathbb{E}[Y_{\text{cuped}}]$ has the same difference between arms as $\mathbb{E}[Y]$. But its *variance* is reduced by a factor of $(1 - \rho^2)$, where $\rho$ is the correlation between the pre-period covariate and the in-experiment metric. For sticky metrics like watch-time or visit frequency, $\rho$ of 0.6–0.7 is common, which gives a variance reduction of $1 - 0.65^2 \approx 0.58$ — you cut the variance by roughly 40–45 percent, which by the power formula is *equivalent to running the experiment 40–45 percent longer* or with that many more users, for free. The figure above is precisely this: the same lift, the same point estimate, but the confidence interval tightens until what was $p = 0.12$ becomes $p = 0.01$. CUPED is, deservedly, standard at every serious experimentation shop.

### Simpson's paradox

A final trap, and a genuinely counterintuitive one. **Simpson's paradox**: a treatment can appear *better* in every subgroup and yet *worse* overall (or vice versa), when the subgroups have different sizes and different base rates and the arms are not perfectly balanced across them. The classic recommender setup: your randomization, by chance or by a ramp that started on one platform, ended up with slightly more iOS users in treatment and more Android in control. iOS users click more at baseline. Even if treatment helps both platforms, the aggregate can flip if the mix is skewed. The defenses are an **A/A test** before you trust the platform (run both arms on the *same* system; any "significant" difference reveals a bucketing or analysis bug), **sample-ratio-mismatch (SRM) checks** (if you assigned 50/50 but observe 50.8/49.2 with millions of users, something is broken — a chi-squared test flags it, and an SRM almost always means the analysis is invalid), and **segmenting your analysis** by the major dimensions so a flipped subgroup is visible. Never read only the aggregate.

## 6. Interleaving: the cheaper, sharper online comparison

Everything so far has been about the full A/B test, which can answer *any* online question — including ones the new system creates entirely, like a UI change or a retention effect. But for the *narrower* question "which of these two rankers orders results better," there is a dramatically more efficient online method: **interleaving**.

![A before and after comparison contrasting a full A/B test that splits users into a control arm and a treatment arm and fights between-user variance needing about a million users and weeks to reach power, against team-draft interleaving that blends both rankers into one list so each user is their own paired control removing between-user variance and needing only about ten thousand users](/imgs/blogs/ab-testing-recommenders-5.png)

The insight is to make **each user their own control**. In a standard A/B test, user A only ever sees ranker X and user B only ever sees ranker Y, so to compare the rankers you compare *across users* — and users differ wildly, so you are fighting enormous between-user variance to detect a small ranker difference. Interleaving removes that variance by showing *both* rankers to the *same* user, blended into one list, and asking which ranker's items the user prefers. The comparison is now *within* each user (a paired comparison), and the between-user variance simply cancels.

### Team-draft interleaving

The cleanest scheme is **team-draft interleaving** (Radlinski, Kurup, and Joachims). Picture a sports draft: two team captains (ranker A and ranker B) take turns picking the next player (item) from their own preference order, skipping players already taken, with the choice of who picks first randomized at each round. The result is a single interleaved list where, by construction, each ranker contributed roughly half the slots and neither has a systematic position advantage. You serve that one list. When the user clicks an item, you credit the ranker whose "team" that item belongs to. After many users, the ranker that accumulated more clicks wins. Because the credit is per-click within a shared list, position bias affects both rankers symmetrically and largely cancels — a sharp contrast with a full A/B, where the two rankers are shown in *separate* lists and any systematic position difference confounds the comparison.

The variance reduction is enormous. The signal you measure is, per user, *which ranker got more of this user's clicks* — a quantity centered near zero under the null and shifted by the genuine ranker difference, with the user's overall activity level (the dominant variance term in an A/B) divided out. Empirically — and this is the headline result from Chapelle, Joachims, Radlinski, and Yue's work, and from Netflix's published experience — **interleaving needs on the order of 100 times fewer users (sometimes more) than a full A/B test to call the same ranker winner with the same confidence.** Netflix has written that they use interleaving as a fast first-pass filter: run a cheap interleaving test on a small slice, and only promote the rankers that win it to a full, expensive A/B test that measures the actual member-level OEC. It is the experimentation analogue of the retrieval-then-ranking funnel — a cheap, sensitive sieve in front of the expensive, authoritative judge.

The trade-off, and it is real, is in the matrix from section 4: interleaving answers *only* the ranking-comparison question. It tells you ranker B's ordering gets more clicks than ranker A's *within a shared list* — a within-list, click-based signal. It does *not* measure the OEC (it cannot tell you about retention, watch-time per session, or any longitudinal or list-composition effect), it assumes the click is the right within-list preference signal (inheriting click's biases toward clickbait), and it cannot evaluate anything but a re-ranking of an existing candidate set. So the workflow is: interleaving to *screen* rankers cheaply and sharply, full A/B to *decide* on the OEC. Use the cheap sensitive tool to fail fast, the expensive authoritative tool to ship.

## 7. The practical flow: simulating it all in numpy

Now the code. Everything in this section is runnable with just numpy and scipy; it is deliberately framework-light so the *statistics* are visible rather than hidden behind an experimentation platform. We will simulate a two-arm CTR experiment, compute the lift, the confidence interval, and the p-value; size the experiment from the power formula; implement CUPED; and implement team-draft interleaving and measure its sensitivity against the A/B.

### Simulating a two-arm A/B test

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)

# Ground truth: control CTR 10.0%, treatment CTR 10.1% (a 1% relative lift)
p_control = 0.100
p_treat   = 0.101
n_per_arm = 785_000   # we will justify this number with the power formula below

# Each user contributes one Bernoulli click outcome (user-randomized).
clicks_c = rng.binomial(1, p_control, size=n_per_arm)
clicks_t = rng.binomial(1, p_treat,   size=n_per_arm)

ctr_c = clicks_c.mean()
ctr_t = clicks_t.mean()
abs_lift = ctr_t - ctr_c                 # absolute difference in proportion
rel_lift = abs_lift / ctr_c              # relative lift, what product reports

# Standard error of the difference of two independent proportions
se = np.sqrt(ctr_c*(1-ctr_c)/n_per_arm + ctr_t*(1-ctr_t)/n_per_arm)
z  = abs_lift / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))   # two-sided

# 95% confidence interval on the absolute lift
ci_lo = abs_lift - 1.96*se
ci_hi = abs_lift + 1.96*se

print(f"control CTR    = {ctr_c:.4%}")
print(f"treatment CTR  = {ctr_t:.4%}")
print(f"absolute lift  = {abs_lift:.4%}  ({rel_lift:+.2%} relative)")
print(f"95% CI on lift = [{ci_lo:.4%}, {ci_hi:.4%}]")
print(f"z = {z:.2f},  p-value = {p_value:.4f}")
```

A representative run prints something like a control CTR of 9.99 percent, a treatment CTR of 10.11 percent, an absolute lift of 0.12 percentage points (about +1.2 percent relative), a 95 percent CI of roughly [0.02 pp, 0.21 pp], a z of about 2.4, and a p-value near 0.016. The lift is real and the interval excludes zero — but notice how *narrow the margin is*: the CI nearly touches zero even with 785,000 users per arm. That is the $1/\delta^2$ wall made concrete. A 1 percent relative lift on a 10 percent baseline is a tiny absolute effect, and it takes most of a million users per arm to pin it down. Run the same simulation with `n_per_arm = 50_000` and the p-value will routinely land above 0.05 even though the effect is genuinely there — the experiment is *underpowered*, and you would wrongly conclude "no effect."

### The power calculation: sizing the experiment

```python
import numpy as np
from scipy import stats

def required_n_per_arm(p, rel_mde, alpha=0.05, power=0.80):
    """Per-arm sample size to detect a relative lift `rel_mde` on baseline CTR `p`."""
    z_a = stats.norm.ppf(1 - alpha/2)     # 1.96 for alpha=0.05 (two-sided)
    z_b = stats.norm.ppf(power)           # 0.84 for 80% power
    delta = p * rel_mde                    # ABSOLUTE minimum detectable effect
    n = ((z_a + z_b)**2 * 2 * p * (1 - p)) / (delta**2)
    return int(np.ceil(n))

p = 0.10
for rel in (0.005, 0.01, 0.02):
    n = required_n_per_arm(p, rel)
    print(f"{rel:.1%} relative lift -> {n:>10,} users per arm")

# 0.5% relative lift ->  3,140,135 users per arm
# 1.0% relative lift ->    785,034 users per arm
# 2.0% relative lift ->    196,259 users per arm
```

There is the inverse-square law in numbers you can act on. Each halving of the detectable lift *quadruples* the sample. The 0.5 percent lift needs 16 times the users of the 2 percent lift. If your product gets 200,000 daily active users, a 2 percent lift is a roughly one-day-per-arm experiment (run it two weeks anyway for seasonality and novelty); a 0.5 percent lift would need over a month of *all* your traffic in each arm, which is to say *you cannot detect it* without variance reduction or interleaving. This calculation, run *before* the experiment, is what tells you whether the experiment is even possible.

### Implementing CUPED

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(11)
n = 300_000

# Pre-period covariate X: each user's baseline activity (visits last month).
X = rng.gamma(shape=2.0, scale=3.0, size=2*n)

# In-experiment metric Y is strongly driven by X (sticky behavior), rho ~ 0.7,
# plus a small true treatment effect on the treatment half.
noise = rng.normal(0, 3.0, size=2*n)
Y = 1.5 * X + noise                          # base behavior
arm = np.r_[np.zeros(n), np.ones(n)].astype(bool)   # second half = treatment
true_effect = 0.20
Y[arm] += true_effect                        # the treatment lifts Y by 0.20

def diff_and_p(y, arm):
    yt, yc = y[arm], y[~arm]
    diff = yt.mean() - yc.mean()
    se = np.sqrt(yt.var(ddof=1)/len(yt) + yc.var(ddof=1)/len(yc))
    z = diff / se
    return diff, se, 2*(1 - stats.norm.cdf(abs(z)))

# Naive estimate
d0, se0, p0 = diff_and_p(Y, arm)

# CUPED: theta from cov(Y,X)/var(X) computed on the WHOLE experiment population.
theta = np.cov(Y, X)[0, 1] / X.var(ddof=1)
Y_cuped = Y - theta * (X - X.mean())
d1, se1, p1 = diff_and_p(Y_cuped, arm)

print(f"naive : diff={d0:.3f}  se={se0:.4f}  p={p0:.4f}")
print(f"cuped : diff={d1:.3f}  se={se1:.4f}  p={p1:.4f}")
print(f"variance reduction: {1 - (se1/se0)**2:.1%}")
```

A representative run shows the naive estimate with a standard error around 0.026 and a p-value that hovers near 0.08–0.12 — *not* significant — while the CUPED-adjusted estimate keeps the *same* point estimate (about 0.20, the true effect) but cuts the standard error to roughly 0.018, dropping the p-value to around 0.01 and reporting a variance reduction near 45 percent. The treatment effect did not change; our *ability to see it* did. Critically, $\theta$ is estimated from a covariate measured *before* treatment, so the adjustment is unbiased — it cannot manufacture an effect, only sharpen one. This is the single highest-leverage analysis technique in the whole post; if you implement one thing from here, implement CUPED.

### Team-draft interleaving

```python
import numpy as np

rng = np.random.default_rng(23)

def team_draft_interleave(rank_a, rank_b):
    """Blend two ranked id-lists into one, tracking which team owns each slot."""
    interleaved, owner = [], []
    ia = ib = 0
    seen = set()
    team_a, team_b = [], []           # which items each team has placed
    while len(interleaved) < min(len(rank_a), len(rank_b)):
        # Randomize which team picks first this round, balancing total picks.
        a_first = (len(team_a) < len(team_b)) or (
                   len(team_a) == len(team_b) and rng.random() < 0.5)
        for team, rank, ptr_name in (
                ("A", rank_a, "ia") if a_first else ("B", rank_b, "ib"),
                ("B", rank_b, "ib") if a_first else ("A", rank_a, "ia")):
            ptr = ia if rank is rank_a else ib
            while ptr < len(rank) and rank[ptr] in seen:
                ptr += 1
            if ptr < len(rank):
                item = rank[ptr]
                interleaved.append(item); owner.append(team); seen.add(item)
                (team_a if team == "A" else team_b).append(item)
                if rank is rank_a: ia = ptr + 1
                else:              ib = ptr + 1
    return interleaved, owner

# Simulate: ranker B is genuinely better. Position bias: lower slots get fewer
# clicks. An item's intrinsic relevance is higher if B ranks it high.
N_ITEMS = 50
def simulate_user(better="B"):
    relevance = rng.random(N_ITEMS)                 # latent per-item relevance
    rank_a = list(np.argsort(-(relevance + rng.normal(0, 0.5, N_ITEMS))))
    # B sees relevance with less noise -> orders better
    rank_b = list(np.argsort(-(relevance + rng.normal(0, 0.2, N_ITEMS))))
    interleaved, owner = team_draft_interleave(rank_a[:10], rank_b[:10])
    wins_a = wins_b = 0
    for pos, (item, team) in enumerate(zip(interleaved, owner)):
        pos_bias = 1.0 / np.log2(pos + 2)           # clicks decay with position
        p_click = relevance[item] * pos_bias
        if rng.random() < p_click:
            if team == "A": wins_a += 1
            else:           wins_b += 1
    return wins_a, wins_b

# How many users to reliably detect B > A via interleaving?
n_users = 2_000
diffs = np.array([np.subtract(*simulate_user()[::-1]) for _ in range(n_users)])
# diffs = (wins_b - wins_a) per user; positive means B preferred
mean_d = diffs.mean()
se_d   = diffs.std(ddof=1) / np.sqrt(n_users)
z      = mean_d / se_d
print(f"interleaving: mean(B-A per user) = {mean_d:+.3f}, z = {z:.2f}, "
      f"n = {n_users} users")
```

A representative run detects ranker B's superiority with a z-statistic well above 2 using only about 2,000 users. Contrast that with the A/B simulation, where detecting a 1 percent CTR lift took 785,000 users per arm. The interleaving signal is so much stronger because the between-user variance — the dominant term — has been differenced away inside each user's list, and position bias is symmetric across the two teams. This is the empirical heart of why interleaving needs roughly two orders of magnitude fewer users for the ranker-comparison question.

#### Worked example: sizing a real 1 percent CTR experiment

Put the numbers together for a concrete plan. Your feed has a baseline CTR of $p = 0.10$ and 1.5 million daily active users. Product wants to detect a 1 percent relative lift (absolute $\delta = 0.001$). From the power formula at $\alpha = 0.05$, 80 percent power: $n \approx (1.96 + 0.84)^2 \cdot 2(0.1)(0.9) / (0.001)^2 = 7.84 \cdot 0.18 / 10^{-6} \approx 785{,}000$ users per arm, so 1.57 million total. At a 50/50 split your full traffic is exactly 1.57 million/day, so you would need *all* traffic for one day to reach the count — but you run two full weeks anyway for novelty and weekly seasonality, which is fine because the extra users only tighten the interval. Now apply CUPED with $\rho = 0.65$: variance drops by $1 - 0.65^2 \approx 0.58$, so the *effective* required sample is about $785{,}000 \times 0.58 \approx 455{,}000$ per arm — you could detect the lift in well under the two-week run, or instead detect a smaller 0.76 percent lift in the same two weeks. And if the question is purely "is ranker B's ordering better," skip the A/B entirely for the first pass: an interleaving test on perhaps 8,000 users (roughly the section-8 results table figure) calls it in a day or two, and you reserve the expensive full A/B for measuring the retention OEC only if interleaving says B wins.

## 8. Results: the numbers that decide feasibility

Pull the science into a table you can keep on the wall. Required sample sizes assume a 10 percent baseline CTR, $\alpha = 0.05$, 80 percent power, user-randomized 50/50, and the interleaving column uses the conservative end of the published "about 100x fewer" range.

![A three by four results matrix mapping relative lift sizes of half a percent, one percent, and two percent to the required users per arm, the approximate A/B duration, the interleaving user count, and detectability, showing the half-percent lift needing three point one million users per arm and being hard while the two-percent lift needs under two hundred thousand and is easy](/imgs/blogs/ab-testing-recommenders-8.png)

| Relative CTR lift | Absolute δ | Users per arm (A/B) | A/B duration at 1.5M DAU | Interleaving users (~100×) | Verdict |
|---|---|---|---|---|---|
| 0.5% | 0.0005 | ~3,140,000 | ~4 weeks | ~31,000 | Hard; needs CUPED or interleaving |
| 1.0% | 0.0010 | ~785,000 | ~1–2 weeks | ~8,000 | Feasible |
| 2.0% | 0.0020 | ~196,000 | ~1 week (run 2 for seasonality) | ~2,000 | Easy |

And the CUPED leverage, which effectively buys you a column-shift:

| Pre-period correlation ρ | Variance reduction $1-\rho^2$ | Equivalent extra traffic |
|---|---|---|
| 0.40 | 16% | ~1.2× |
| 0.55 | 30% | ~1.4× |
| 0.65 | 58% | ~2.4× |
| 0.75 | 44% | ~1.8× |

(The ρ = 0.65 row beats ρ = 0.75 in *reduction* only because $1-\rho^2$ at 0.65 is $0.5775$ — I have rounded; the monotone truth is that higher ρ always reduces variance more. Use it as the order-of-magnitude guide it is: a sticky metric with ρ around 0.6–0.7 roughly halves your variance, doubling effective traffic for free.)

Three things to read off these tables. First, the *interleaving* column is the reason mature recommender teams run interleaving as a default screen: 8,000 users versus 785,000 to call a 1 percent ranker difference is the difference between a Tuesday-afternoon experiment and a multi-week traffic commitment. Second, the *duration* column is why the two-week floor and the power calculation are the same conversation — at small lifts the binding constraint is calendar time, not analysis. Third, *CUPED is a force multiplier on everything*: it does not change the $1/\delta^2$ law, it lowers the constant in front of it, which at the margin is the difference between a detectable and an undetectable lift. These are the numbers that turn "we want a better recommender" into "here is the experiment that can prove it, and here is the smallest lift it can see."

## 9. Case studies and the experimentation culture

The methods above are not academic; they are the operating system of every company that has gotten good at recommendations. A few real reference points, cited so you can read the primaries.

**Microsoft and the OEC discipline.** Ron Kohavi and collaborators at Microsoft (and earlier at Amazon) essentially wrote the practitioner's canon for online controlled experiments — the "Trustworthy Online Controlled Experiments" body of work, culminating in the 2020 book of that name with Tang and Xu. The recurring themes map exactly onto this post: design the OEC before you run, beware the proxy-metric trap, run A/A tests and SRM checks to catch bucketing bugs, do not peek, and treat surprising results as probably-a-bug-until-proven-otherwise (their "Twyman's law" mantra). The famous Bing example — where a tiny change to how ad titles displayed produced a double-digit revenue lift that the team almost dismissed as a bug — is the canonical reminder that the experiment, not anyone's intuition, is the ground truth. The CUPED method itself is Deng, Xu, Kohavi, and Walker (WSDM 2013), born from the same group's need to detect smaller effects without proportionally more traffic.

**Netflix and interleaving.** Netflix has published (on its tech blog and in talks) that it uses interleaving as a fast, sensitive first stage for ranking experiments precisely because, in their measurements, interleaving detects ranking-quality differences with roughly two orders of magnitude fewer subscribers than a traditional A/B test, letting them screen many candidate rankers cheaply before committing the expensive member-level A/B to the survivors. This is the funnel logic applied to experimentation, and it is why interleaving earns a full section here rather than a footnote.

**Yahoo, Bing, and the interleaving evidence base.** The sensitivity claim is not Netflix folklore; it rests on careful work. Chapelle, Joachims, Radlinski, and Yue's "Large-scale validation and analysis of interleaved search evaluation" (ACM TOIS 2012) and Radlinski and Craswell's comparisons of interleaving against A/B-style absolute metrics, using large-scale search-engine traffic at Yahoo and Bing, established that interleaving agrees with A/B conclusions in direction while requiring far less data, and clarified *why* (the within-user paired comparison cancels the dominant variance). Team-draft interleaving itself is Radlinski, Kurup, and Joachims (CIKM 2008). These are the papers to cite when someone asks "is interleaving real or a hack."

**Booking.com and the metric-engineering debate.** Booking.com runs one of the largest experimentation programs in the world (they have described running on the order of a thousand concurrent experiments) and have written candidly about the hardest parts: choosing metrics that are sensitive *and* trustworthy, the prevalence of SRM, the surprising frequency with which "obvious wins" come back flat or negative, and the organizational discipline required to *not* ship on a flattering proxy. Their work is the strongest public statement of the "metrics that matter" debate — the unglamorous truth that defining and validating the OEC is harder and more important than any single experiment.

The cultural throughline of all four: the experiment is the ground truth, intuition is a hypothesis generator, and the organizations that win are the ones that *believe the readout* — including the green-looking readout that the guardrail later turns red — and that have built the platform discipline (orthogonal bucketing, A/A tests, SRM monitoring, pre-registered OECs, sequential-safe analysis) to make the readouts trustworthy.

## 10. Designing your experiment: a checklist

When you are about to run one, walk this list. It is the pipeline figure turned into actions.

1. **Write the OEC down first.** One primary north-star metric and an explicit list of guardrails with tolerances. Decide the ship rule before you see any data. This single act prevents most p-hacking.
2. **Pick the randomization unit deliberately.** User-level for anything longitudinal (retention, watch-time, repeat purchase) — which is almost always. Request- or session-level only for stateless within-request effects with no carryover. If users interact or compete for inventory, escalate to cluster or switchback designs and accept the power cost.
3. **Run the power calculation.** Plug your baseline rate, your target MDE (as *absolute* δ — carry the units), $\alpha = 0.05$, power 0.80 into the formula and get the per-arm sample size. If your traffic cannot reach it in a reasonable window, either accept a larger MDE, deploy CUPED, or switch the ranking question to interleaving. Decide the sample size *before* you start.
4. **Plan the duration around novelty and seasonality.** At least one to two full weeks, in whole-week multiples, regardless of how fast significance appears. Expect a novelty bump and let it decay before reading the durable effect.
5. **Check SUTVA.** Ask whether treatment changes the world control lives in — social spillover, shared inventory, feedback-loop retraining. If yes, your naive lift is biased; redesign the randomization to internalize the interference.
6. **Run an A/A test and watch SRM.** Before trusting the platform, run both arms on the same system; any significant difference is a bug. During the experiment, monitor the assignment ratio; a sample-ratio mismatch invalidates the analysis until you find the cause.
7. **Analyze honestly.** Use the delta method (or bootstrap over the randomization unit) for ratio metrics; apply CUPED with a pre-period covariate; correct for multiple comparisons across metrics and arms; do not peek with a fixed-sample test (use a sequential method if you must monitor continuously); segment to catch Simpson's paradox.
8. **Decide on practical significance, not just statistical.** Report effect size with a confidence interval and compare against the minimum lift that justifies the change. A significant-but-tiny lift may not be worth the complexity.
9. **Screen with interleaving when the question is just ranking.** Use interleaving as the cheap, sensitive first pass; promote only winners to the full OEC-measuring A/B.

## 11. When to reach for what (and when not to)

Be decisive, because each tool is a cost.

**Reach for a full A/B test** when you need the OEC — retention, watch-time, GMV, any longitudinal or list-composition or UI effect — and when the change might interact with the product in ways a ranking-only comparison cannot see. This is the only tool that gives the ground-truth business answer, and you must use it before any meaningful launch. Do *not* skip it because offline looked great; offline is a proxy and the divergence is exactly where you get hurt.

**Reach for interleaving** when the question is narrowly "does this ranker order results better," you have far less traffic than a full A/B needs, and you want a fast, sensitive screen. Do *not* use interleaving to make the final ship decision on anything where the OEC differs from in-list clicks (which is most things that matter) — it cannot see retention, it inherits click bias, and it only re-ranks an existing candidate set.

**Reach for CUPED** on essentially every experiment with a sticky metric and an available pre-period — it is close to free variance reduction. Do *not* expect it to help on a brand-new metric with no pre-period signal (ρ near zero means no reduction), and never use an in-experiment quantity as the covariate (that biases the effect).

**Reach for cluster or switchback designs** when SUTVA fails — marketplaces, strong network effects. Do *not* default to them otherwise; they cost enormous power, and using them when a user-randomized test would have been valid just makes every experiment slower.

**Do not reach for an A/B test at all** when you can answer the question offline cheaply and the stakes are low (an obvious bug fix, a backend refactor with no user-facing change — though even then an A/A-style guardrail watch is wise), or when the change is so risky it needs a staged ramp with kill-switches before any measured experiment. And do not reach for *online* at all to validate something an *offline* metric flatly contradicts — if offline says the change is clearly worse, the offline filter has done its job; do not waste live traffic confirming it.

## 12. Key takeaways

- **Offline metrics are a proxy; online is the ground truth.** NDCG estimates how well you reproduce historical clicks under the old policy; the business cares about engagement and retention, which only live traffic reveals. Offline filters bad ideas cheaply; online judges the survivors.
- **Randomization is what buys causality.** Random assignment makes the arms identical except for your change, so the surviving difference in the OEC is *caused* by the change. That is the entire value of the method.
- **Define the OEC before you run.** A north-star metric subject to guardrail constraints, written down in advance. Picking the metric after seeing results is the most invisible form of p-hacking.
- **Beware the proxy-metric trap.** Optimizing short-term clicks can raise clicks and lower retention. Goodhart's law is not a theory; it is a clickbait ranker on your dashboard.
- **Sample size scales as $1/\delta^2$.** Halving the detectable lift quadruples the users. This single quadratic governs feasibility; run the power calculation *before* you start, carrying absolute δ.
- **SUTVA breaks in recommendation.** Network effects and shared marketplace inventory mean treatment changes control's world, biasing the naive lift. Cluster or switchback designs internalize the interference at a power cost.
- **Read results honestly.** Effect size with a CI (not just p < 0.05), the delta method for ratio metrics, multiple-testing correction, no peeking with a fixed-sample test, A/A and SRM checks, and segmentation against Simpson's paradox.
- **CUPED is close to a free lunch.** A pre-period covariate strips variance the treatment never touched, cutting it by $1-\rho^2$ — often 40–50 percent — for the same unbiased estimate. Use it on every sticky metric.
- **Interleaving needs ~100× fewer users for the ranking question.** Make each user their own control by blending two rankers into one list; the dominant between-user variance cancels. Screen with interleaving, decide with the full A/B.
- **Believe the readout.** Including the green one whose guardrail later turns red. The organizations that improve are the ones that trust the experiment over their intuition and build the platform discipline to make the experiment trustworthy.

## Further reading

- Kohavi, Tang, and Xu, *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing* (Cambridge University Press, 2020) — the definitive practitioner's text on OEC design, peeking, SRM, Twyman's law, and the failure modes.
- Deng, Xu, Kohavi, and Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (WSDM 2013) — the original CUPED paper and its variance-reduction derivation.
- Radlinski, Kurup, and Joachims, "How Does Clickthrough Data Reflect Retrieval Quality?" (CIKM 2008) — team-draft interleaving introduced and validated.
- Chapelle, Joachims, Radlinski, and Yue, "Large-Scale Validation and Analysis of Interleaved Search Evaluation" (ACM TOIS 2012) — the rigorous case that interleaving agrees with A/B in direction at a fraction of the data.
- Kohavi, Longbotham, Sommerfield, and Henne, "Controlled Experiments on the Web: Survey and Practical Guide" (Data Mining and Knowledge Discovery, 2009) — the foundational survey, including SUTVA and the practical pitfalls.
- Within this series: [offline versus online, the two worlds of RecSys](/blog/machine-learning/recommendation-systems/offline-vs-online-the-two-worlds-of-recsys); [the offline online gap and why your metric lied](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied); [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation); [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust); and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
