---
title: "Feedback Loops and Filter Bubbles"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why a recommender that trains on its own logged clicks quietly narrows users into filter bubbles and collapses the catalog to a few items, the dynamical-systems reason the fixed point is degenerate, a runnable numpy simulation that shows diversity decaying round after round, and the exploration, IPW, diversity re-ranking, and exposure caps that change the fixed point."
tags:
  [
    "recommendation-systems",
    "recsys",
    "feedback-loops",
    "filter-bubbles",
    "exposure-bias",
    "exploration",
    "off-policy",
    "diversity",
    "machine-learning",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/feedback-loops-and-filter-bubbles-1.png"
---

The most unsettling on-call page I ever got was not an outage. It was a dashboard. Six weeks after we shipped a "better" home feed model, a content-ops analyst pulled the distribution of impressions across our music catalog and dropped it in the channel with a single line: "is this supposed to look like this?" The chart was a cliff. Out of roughly two hundred thousand tracks that had real listenership, about three hundred were now absorbing more than 80% of all home-feed impressions. Six weeks earlier the same chart had a long, healthy tail. Nobody had changed the catalog. Nobody had changed the business rules. We had not even shipped a new model in those six weeks — we had shipped *one* model, and then we let it do the thing recommenders do: retrain every night on the clicks it had generated the day before. The model was eating its own output, and the output was getting narrower every single day.

I want to be precise about why this is the hardest class of bug a recommender team faces, because it shapes everything in this post. A normal bug is a point-in-time defect: a feature is computed wrong, a join drops rows, a model overfits. You can reproduce it, fix it, and verify the fix in an afternoon. A feedback-loop harm is not a point-in-time defect — it is an *emergent property of a dynamical system running over weeks*. There is no single line of code to fix, no single bad commit to revert. The model that collapses the catalog is, on any given day, doing exactly what it was trained to do, correctly, with no bug in the ordinary sense. The harm lives in the *iteration*, in the compounding, in the slow drift of the data the system feeds itself. That is why these problems get shipped by competent teams who watch their dashboards every day, and why the fix is never "patch the model" but always "change the loop."

That is the subject of this post. A recommender is not a static function that you train once and deploy. It is a **closed loop**. It serves a slate, a user reacts to that slate, you log the reaction, you retrain on the log, and the new model serves the next slate. This loop is the engine that lets the system learn at all — you cannot improve a recommender without feeding it interaction data, and the only interaction data you have is the data your own recommender produced. But the very same loop is the source of a family of slow, compounding harms: **filter bubbles** that narrow a single user to an ever-tighter slice of content, **preference amplification** that turns a mild interest into a dominant one, **homogenization** that makes everyone converge to the same items, and **degeneracy** — the catalog quietly collapsing to a handful of winners while the tail goes dark. The chart that paged me was degeneracy, and the reason I had not seen it coming is the reason this post exists: none of our offline metrics could see it, because offline metrics evaluate one shot on logged data and the harm only emerges over many rounds of the loop.

Figure 1 draws the loop the way it actually behaves. By the end of this post you will be able to do four concrete things: (1) explain *why* the loop's fixed point is so often degenerate, using a short dynamical-systems argument; (2) **simulate** the loop in plain numpy, watch diversity and coverage collapse round after round, and then watch a small dose of exploration change the trajectory; (3) diagnose a live system — measure diversity decay, exposure concentration, and recommender-induced interest drift versus organic drift; and (4) break the loop in production with the right combination of exploration, randomized logging, inverse-propensity debiasing, diversity re-ranking, calibration, and exposure caps. This is the second spine of the whole series — the [serve → log → train → serve loop](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) feeding the [retrieval → ranking → re-ranking funnel](/blog/machine-learning/recommendation-systems/the-recommendation-funnel-retrieval-ranking-reranking) — and it is the one that bites you slowest and hardest.

![A closed recommender loop drawn as an acyclic chain from model to served slate to user reaction to logged feedback to retraining to a narrower next model](/imgs/blogs/feedback-loops-and-filter-bubbles-1.png)

## 1. The closed loop: serve, log, train, serve

Start with the mechanics, because everything that follows is a consequence of them. A production recommender runs a cycle that looks like this, every day or every few hours:

1. **Serve.** The current model ranks the catalog for a user and returns a slate — the top-K items it shows. Out of a million items, the user sees maybe ten to fifty.
2. **React.** The user clicks, watches, skips, or ignores. Critically, they can only react to the items *they were shown*. They cannot click an item that never appeared.
3. **Log.** You record the reaction. Your training table grows by a few rows per impression: `(user, item, shown, clicked, context)`.
4. **Train.** Overnight (or continuously), you fit a new model on the accumulated log — including the rows you just generated.
5. **Serve again.** The new model ranks for the next request, and the cycle repeats.

The single most important fact about this loop is in step 3, and it is so quiet that teams forget it for years: **you only collect feedback on items you chose to show.** The recommender controls its own training data. This is not like a weather model, where the rain falls regardless of what the model predicts. Here, the model's prediction *determines what data exists*. If the model never shows you jazz, there will never be a single jazz click in your log from you, and the next model will conclude — correctly, from the data — that you do not click jazz. The data is not lying. The data is *missing-not-at-random* (MNAR), and the missingness is caused by the model itself.

This is why I insist, in every design review, that we draw the loop as a *chain* rather than a static training step. The model is not learning the truth about users. It is learning the truth about *the users-as-filtered-through-the-previous-model*. Round after round, that filter tightens. Conceptually, the loop is a function that maps "the model" to "the next model," and we are iterating that function thousands of times in production. Whatever that iterated map converges to is what your users actually experience six months later — not the nice offline benchmark you launched on.

A quick contrast with the supervised learning most engineers come from. When you train an image classifier, your labeled dataset is fixed and your model has zero influence over which images get labels. The training distribution and the deployment distribution might differ (that is ordinary distribution shift), but at least the *training* distribution is exogenous. In a recommender, the training distribution is *endogenous* — the model writes its own next dataset by deciding what to expose. That endogeneity is the whole game, and it is why so many of the [offline-versus-online surprises](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) in this field trace back to the loop.

It helps to be precise about the timescales involved, because the loop turns at several speeds at once and each speed hides a different harm. The *fastest* loop is session-level personalization: many feeds re-rank within a single session based on what you just tapped, so the loop can turn five times before you put your phone down. The *medium* loop is the nightly or hourly retrain that folds yesterday's clicks into tomorrow's model. The *slowest* loop is the candidate-generation index rebuild and the embedding refresh, which might happen weekly. Each of these is a closed loop with its own gain, and a system can look stable at one timescale while quietly collapsing at another. The catalog cliff that paged me lived in the *medium* loop — nightly retrains — and was invisible in the session-level metrics the team watched all day, because within any single session the feed looked perfectly varied. You have to instrument every loop, not just the one you happen to be staring at.

There is one more property of the loop that engineers consistently underestimate: it has **memory**, and that memory is the accumulated log. Even if you froze the model today and never retrained again, the historical log you already collected is a biased summary of a biased policy, and the next person who trains a model on it inherits all of that bias. Teams sometimes try to fix a collapsed feed by swapping in a fresh architecture — a two-tower instead of matrix factorization, a transformer instead of a two-tower — and are baffled when the new model collapses just as fast. The architecture was never the problem. The *data-generating loop* was the problem, and a better function approximator fit to the same biased log will reproduce the same bias, often faster, because it is better at fitting whatever signal is there. You cannot architecture your way out of a feedback loop; you have to change the data the loop produces.

#### Worked example: one day of logs is not one day of truth

Suppose a user genuinely likes three genres in equal measure: pop, jazz, and electronic, each with a "true" click probability of 0.30 when shown a good item from that genre. On day zero the model has no idea, so its slate of 10 happens to be 8 pop, 1 jazz, 1 electronic. The user clicks roughly $8 \times 0.30 = 2.4$ pop, $0.3$ jazz, $0.3$ electronic. The log now reads: pop 2.4 clicks / 8 impressions = 0.30 CTR, jazz 0.3 / 1 = 0.30, electronic 0.3 / 1 = 0.30. So far the CTRs look identical and honest.

But the *counts* are not. The model fits to 8 pop impressions and 1 each of the others, and any model with even mild regularization or any user-genre interaction term will be far more *confident* about pop simply because that is where the data volume is. Tomorrow it pushes the pop share from 8/10 to 9/10. Day after, 10/10. The user's true preference never changed — it was a flat 0.30 across three genres — yet within a week they see only pop, click only pop (because that is all they are shown), and the log "proves" they are a pop fan. The CTR was never wrong; the *exposure* was. That gap between true preference and exposed preference is the seed of every harm in this post.

## 2. What emerges: bubbles, amplification, homogenization, collapse

The loop produces four named phenomena. They share one root cause but show up at different scales, so it helps to keep them straight. The contrast in Figure 2 is the cleanest way to hold the difference between an *open* exposure regime and a *closed greedy* one.

![A two-column comparison of open exploration with broad slates and wide coverage versus a closed greedy loop with narrow slates and collapsed coverage](/imgs/blogs/feedback-loops-and-filter-bubbles-2.png)

**Filter bubbles (and echo chambers).** This is the per-user version. The system narrows one user to an ever-tighter slice of the catalog. The term *filter bubble* was popularized by Eli Pariser in 2011 to describe the personalized web wrapping each person in their own informational world; *echo chamber* is the closely related idea that a user mostly encounters content that reinforces what they already engaged with. In recommender terms, the bubble is the steady-state set of items the user is shown after the loop has run long enough — and that set is usually far narrower than the user's genuine interests, because the loop optimized for the *predicted* favorite, not the *full* taste. The crucial subtlety is that a bubble can form even when the model is *right about every individual prediction*. Each item it picks really is the user's most likely click in isolation. The failure is in the *set*: by repeatedly choosing the single most-predictable item, the loop never leaves room to discover the user's second, third, and fourth interests, so those interests slowly disappear from the data and then from the model's beliefs. A bubble is not a sequence of wrong answers; it is a sequence of locally-right answers that adds up to a globally-impoverished experience.

**Preference amplification.** This is the dynamic version of the bubble. A mild interest gets reinforced into a dominant one. The user clicks one true-crime podcast out of curiosity; the model notices and shows two; they click one of those; the model shows four; and three months later their feed is 90% true crime. The model did not invent the interest — it *amplified* a weak signal into a strong behavior. This is closely tied to [popularity bias and the rich getting richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer): the same self-reinforcing math applies at the level of a single user's interests as it does at the level of the whole catalog's items.

**Homogenization.** This is the population version. Because every user is served by *the same model*, and because that model has learned which items are globally "safe bets," different users start converging to similar content. The personalization that was supposed to make everyone's feed unique can, paradoxically, make feeds more alike — everyone funnels toward the items the model is most confident about. There is a real research thread here (for example work on how recommendation can *increase* commonality of consumption across users), and it is worth taking seriously even when it is uncomfortable, because it cuts against the marketing story that more personalization always means more diversity.

**Degeneracy and collapse.** This is the catalog version, and it is what paged me. Over time, diversity decays and exposure concentrates onto a shrinking set of items. Jiang, Chaney, and colleagues studied this directly and showed that interactive recommenders can drive the system toward **degenerate feedback loops** where the recommendable content collapses and user interests appear to narrow as an artifact of the system rather than a fact about the user. The catalog has not shrunk; the *reachable* catalog has. This is the harm with the most direct business cost, because a collapsed catalog is a wasted catalog. You licensed, produced, or onboarded a million items, and the loop has quietly decided that only a few hundred of them are worth anyone's attention — not because the rest are bad, but because the rest were unlucky early and never recovered. Every dollar spent acquiring tail content is wasted if the loop never surfaces it, and every creator or seller who depends on that exposure is being silently de-platformed by a statistical artifact.

These four are not independent diseases with independent cures; they are four readouts of one underlying instability, which is why a fix aimed at one often helps the others. Diversity re-ranking, aimed at the per-user bubble, also slows catalog collapse because it forces tail items into slots. Exploration, aimed at gathering tail data, also fights homogenization because it gives different users different fresh items. The unity matters: if you treat them as four separate problems and bolt on four disconnected hacks, you will fight yourself. Treat them as one loop with one set of levers, and the levers reinforce each other.

All four are the same fire seen from different windows. Figure 3 organizes them by scale: per-user effects on one branch, population effects on the other.

![A taxonomy tree splitting closed-loop harms into per-user effects of filter bubble and amplification and population effects of homogenization and degeneracy](/imgs/blogs/feedback-loops-and-filter-bubbles-3.png)

The single shared mechanism is **exposure bias**: you only learn about what you show. Make that precise. Let `e_i` be the probability item `i` is exposed (shown) under the current policy, and `r_i` the user's true relevance for it. The observed click rate you can estimate from logs is, schematically, proportional to $e_i \cdot r_i$ — you can only observe a click on $i$ if $i$ was both shown *and* relevant. A high-relevance item with low exposure looks exactly like a low-relevance item: zero observed clicks either way. The model, fitting observed clicks, learns $e_i \cdot r_i$, not $r_i$. Then it uses that estimate to set next round's exposure, so `e_i` for already-exposed items goes up and for never-exposed items stays at zero. That multiplicative coupling between "what I show" and "what I learn" is the engine. Figure 4 lays out each harm against its mechanism, the metric trend that detects it, and the mitigation that addresses it — a map I keep pinned for design reviews.

![A matrix mapping each loop harm to its mechanism, the metric trend that detects it, and the targeted mitigation](/imgs/blogs/feedback-loops-and-filter-bubbles-4.png)

## 3. The science: the loop as a dynamical system

Now the *why*. Strip the recommender down to its essential dynamics and it becomes a discrete-time dynamical system. Let the **state** be a distribution over items — for one user, the exposure distribution `p_t` over the catalog at round `t` (what fraction of impressions each item gets). The recommender is a map `F` that takes the current exposure-shaped beliefs and produces the next exposure distribution:

$$ p_{t+1} = F(p_t). $$

A **fixed point** is a distribution `p*` with `p* = F(p*)` — a steady state the loop settles into. The question that decides whether your product is healthy is: *what does this fixed point look like?* If it is a broad distribution, your users keep seeing a varied catalog. If it is degenerate — a spike on a few items — your catalog has collapsed.

Here is the uncomfortable result, and it is robust across many concrete model choices: **a purely greedy loop has a degenerate fixed point.** Intuitively, suppose at round `t` item A has a slightly higher estimated relevance than item B. A greedy recommender shows A more. Because it shows A more, it collects more clicks on A (more impressions, same per-impression rate), which *increases its confidence and often its estimate* of A's relevance — and reduces the relative evidence for B, which now gets shown even less. So the gap between A and B widens over rounds. This is a **rich-get-richer** dynamic, and rich-get-richer processes (Pólya urns, preferential attachment) famously converge to extreme, concentrated distributions, not uniform ones. The greedy map `F` has a stable fixed point at "all exposure on the current leader(s)" and an *unstable* fixed point at "uniform." Any perturbation away from uniform — and there is always a perturbation, because of sampling noise — gets amplified toward the degenerate corner.

You can see the instability without heavy math. Consider two items with true relevances `r_A = r_B = 0.5` (genuinely equal). Estimated relevances `\hat{r}_A, \hat{r}_B` start equal but are noisy. The greedy policy gives the larger estimate more impressions; more impressions means a tighter, more confident estimate and a higher chance the model keeps ranking it first. Tiny initial noise (`\hat{r}_A = 0.51` versus `0.49` by luck) gets converted into a durable exposure gap. The symmetric "both items split exposure 50/50" state is a fixed point too — but it is unstable, like a pencil balanced on its tip. Real systems never stay on it.

**Why a little exploration changes the fixed point.** Exploration adds a floor to exposure: every item, even a current loser, gets shown with some minimum probability $\epsilon / N$ (uniform exploration) or gets a relevance bonus for uncertainty (an [upper-confidence-bound / bandit](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) approach). That floor does two things. First, it guarantees the model keeps collecting fresh, *unbiased-ish* evidence on under-exposed items, so a genuinely good but unlucky item can recover instead of being starved to zero forever. Second, it changes the map $F$ itself: the degenerate corner is no longer a fixed point, because at the corner the exploration term immediately pulls exposure back out toward the under-shown items. The new fixed point is a *non-degenerate* distribution whose concentration is controlled by $\epsilon$ and by how good the items actually are. You have traded a sliver of short-term greedy reward for a fixed point you can live with. That is the whole bargain of exploration, and it is why the [exploration-exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) is not an optional nicety in recommenders — it is structural stabilization of the loop.

Let me make the rich-get-richer claim quantitative with a clean toy. Model item $i$'s exposure share at round $t$ as $s_i(t)$, and suppose the policy reinforces shares proportional to current share times relevance: $s_i(t+1) \propto s_i(t)^\gamma \cdot r_i$, normalized to sum to 1, with reinforcement exponent $\gamma \ge 1$. When $\gamma = 1$ and all $r_i$ are equal, exposure shares are a martingale and freeze wherever noise leaves them (a Polya-urn flavor). When $\gamma > 1$ — which is what confidence-driven greedy ranking effectively does, because more exposure sharpens the estimate and steepens the softmax — the dynamics are *super-linear* and a single item's share converges to 1 almost surely. That $\gamma > 1$ regime is the mathematical signature of collapse. Exploration effectively caps $\gamma$ (or adds an additive uniform term that dominates when shares get small), pulling the system back into the non-degenerate regime.

There is a cleaner way to see *why* the uniform state is unstable, using the standard tool for analyzing fixed points: linearize the map and look at the eigenvalues of its Jacobian. A fixed point $p^*$ is stable if every eigenvalue $\lambda$ of the Jacobian $\partial F / \partial p$ evaluated at $p^*$ has magnitude below 1 — perturbations shrink. It is unstable if any eigenvalue has magnitude above 1 — perturbations grow. For the greedy super-linear map at the uniform point, the reinforcement exponent $\gamma$ shows up directly in the diagonal of the Jacobian: a small bump in item $i$'s share gets multiplied by roughly $\gamma$ on the next step (more share, sharper estimate, even more share). When $\gamma > 1$, the relevant eigenvalue exceeds 1 and the uniform fixed point is a *repeller* — the system is pushed away from it toward the corners. Now add uniform exploration $\epsilon$. The exploration term contributes an averaging step that mixes shares back toward uniform, which subtracts from the eigenvalue. With enough $\epsilon$, the dominant eigenvalue drops below 1, the interior fixed point becomes a *stable attractor*, and collapse is mathematically prevented — not merely slowed. This is the precise sense in which "a little exploration changes the fixed point": it flips the sign of the stability, not just the speed.

The same eigenvalue picture explains a phenomenon practitioners find counterintuitive: **a smarter ranker can collapse faster.** A better model produces sharper, more confident scores, which steepens the effective softmax, which raises $\gamma$, which pushes the unstable eigenvalue *higher* above 1. So the very act of improving your ranker — higher offline AUC, tighter calibration — can accelerate the loop's collapse unless you also raise your exploration to compensate. This is why the anti-loop budget should *scale with model quality*: the better your model gets at exploitation, the more exploration it needs to stay stable. Teams that ratchet model quality every quarter while holding exploration fixed are slowly walking their fixed point off a cliff.

#### Worked example: tracing three rounds of narrowing

Take one user and four genres — call them G1, G2, G3, G4 — with equal true relevance 0.25 each (the user honestly likes all four the same). The slate holds 4 items per round, one slot can go to each genre. The model ranks genres by an estimated score that starts at the true value plus tiny noise: `\hat{s} = (0.27, 0.24, 0.25, 0.24)`. A greedy policy fills more slots with higher-scored genres. We will say the slate share is proportional to a sharp softmax of the scores (temperature 0.05, mimicking a confident ranker).

- **Round 1.** Softmax(scores / 0.05) over `(0.27, 0.24, 0.25, 0.24)` ≈ `(0.55, 0.06, 0.33, 0.06)`. So G1 gets ~2.2 of 4 slots, G3 ~1.3, G2 and G4 split the rest. The user clicks at 0.25 per shown item across the board, so G1 accumulates ~0.55 clicks, G3 ~0.33, the others ~0.06 each. The model updates: more clicks *and* more impressions on G1 tighten its estimate upward to, say, `\hat{s} = (0.30, 0.235, 0.255, 0.235)`.
- **Round 2.** New softmax ≈ `(0.72, 0.03, 0.22, 0.03)`. Now G1 takes ~2.9 slots, G3 ~0.9, and G2/G4 are nearly gone. G2 and G4 each got shown ~0.1 times — the model is collecting almost no fresh evidence on them. Their estimates drift *down* slightly from regularization-toward-prior-of-unseen, to `(0.32, 0.22, 0.255, 0.22)`.
- **Round 3.** Softmax ≈ `(0.84, 0.015, 0.13, 0.015)`. G1 owns the slate; G2 and G4 are effectively dark; G3 is fading. The user — who genuinely likes all four equally — is now in a G1 bubble after three rounds, having clicked plenty of G1 (because that is all they saw) and zero G2/G4 (because they never saw them). Intra-list diversity has gone from 4 genres represented to barely 1.5.

Now replay with **$\epsilon = 0.15$ uniform exploration**: each round, 15% of slots are filled uniformly at random across all four genres regardless of score, the other 85% greedily. Round 1 G2 and G4 still each get about $0.85 \times 0.06 + 0.15 \times 0.25 = 0.09$ greedy plus a guaranteed $0.0375$ exploratory share — small, but *nonzero and persistent*. Across rounds, the exploratory floor keeps feeding the model honest clicks on G2/G4 at their true 0.25 rate, so their estimates *cannot* drift to zero. By round 3 the slate is roughly `(0.50, 0.13, 0.24, 0.13)` instead of `(0.84, 0.015, 0.13, 0.015)` — concentrated toward G1 (which is fine, the user does like it), but G2, G3, G4 all still surface. The fixed point moved from "G1 only" to "G1-leaning but four-genre." Same user, same true preferences; the only change was a 15% floor, and it was the difference between a bubble and a healthy feed.

## 4. Why offline evaluation cannot see any of this

Here is the trap that let the loop run for six weeks before anyone noticed. **A single-shot offline metric on logged data is structurally blind to loop dynamics.** Offline evaluation takes a fixed log, hides the last interactions, and scores how well your model would have ranked the held-out items. It answers one question: "given this data, does the model rank well?" It cannot answer "if we deployed this model and let it generate the next month of data, what would the catalog look like?" Those are different questions, and the loop lives entirely in the second one.

Concretely, three things hide the loop from offline metrics:

1. **One shot, not many rounds.** The harms (narrowing, collapse) are *multi-step* — they need the output-becomes-input cycle to run dozens of times. A one-shot metric never iterates the map `F`. You measure `F(p_0)` once; you never see `F^{30}(p_0)`.
2. **The log is already biased the same way.** Your offline test set was generated by the *previous* policy, which had the same exposure bias. So a new greedy model that also narrows will score *well* offline — it agrees with the biased log about what is relevant. Offline NDCG can go *up* precisely because the new model doubles down on the bias already baked into the test data. This is the canonical [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied): the offline metric and the online outcome are measuring different worlds.
3. **No counterfactual.** Offline you cannot observe what would have happened if you had shown the user something else, because the user only reacted to what was actually shown. To reason about "what if I had explored more," you need either a *simulator* or *logged randomization* you can do counterfactual estimation on (the subject of [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation)).

There is a fourth, subtler reason the loop hides from offline metrics, and it is the one that fools senior people: **the metric and the harm are sometimes anti-correlated.** Offline NDCG rewards a model for ranking the user's known favorite at the top. A model that collapses to showing only that favorite will therefore have *excellent* offline NDCG on the biased log — better than a healthy, diverse model, which "wastes" top slots on items the user has not yet clicked (because they have not been shown them yet). So you can be in a situation where the offline leaderboard ranks your models in the *opposite* order of their long-term health: the most aggressive bubble-former wins offline, and the team picks it. I have watched this happen in a real model-selection meeting. We chose the model with the best NDCG@10, shipped it, and it was the most efficient bubble-forming machine we had ever built. The offline metric did exactly what we asked; we just asked the wrong question.

So how *do* you see loop effects? Three honest tools. First, **long-horizon A/B tests** — not a one-week test of CTR, but a multi-week test that watches diversity, coverage, and retention as the two arms' loops diverge. The treatment arm's loop and the control arm's loop are *different dynamical systems*, and you need enough rounds for them to separate; in practice this means at least several retrain cycles, often four to eight weeks, before the diversity curves pull apart enough to be significant. Second, **simulation** — build a user model and a recommender, close the loop in code, and run it for many rounds. Simulation is the only place you can iterate $F$ hundreds of times cheaply, vary $\epsilon$, and watch the fixed point move. Third, **counterfactual estimation on a randomized-logging slice** — if you reserve a small fraction of traffic for known-propensity randomized exposure, you can estimate what a candidate policy *would* do to diversity and coverage without deploying it, using the [off-policy](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) machinery from later in this post. The next section builds the simulator, because it is the single most clarifying thing you can do to internalize the loop, and because once you have built it you will never again trust a one-shot offline number to tell you whether a recommender is healthy.

## 5. Simulating the loop in numpy

Let me build the smallest honest simulator that shows collapse and then shows exploration fixing it. The ingredients: (a) a population of users with *fixed, true* preferences over items (so any narrowing we observe is the system's doing, not the user's); (b) a recommender that holds an estimate of each user's item affinities, ranks greedily, and shows a top-K slate; (c) a click model where the user clicks with probability proportional to true relevance *only on shown items*; (d) a retrain step that updates estimates from the logged clicks; (e) metrics — intra-list diversity and catalog coverage — measured every round.

```python
import numpy as np

rng = np.random.default_rng(7)

N_USERS, N_ITEMS, N_GENRES = 200, 500, 10
K = 10            # slate size shown per round
ROUNDS = 40

# --- ground truth: each item has a genre; each user has a flat-ish taste over genres
item_genre = rng.integers(0, N_GENRES, size=N_ITEMS)
# users genuinely like several genres; keep it broad so any narrowing is the SYSTEM's fault
user_genre_pref = rng.dirichlet(alpha=np.ones(N_GENRES) * 2.0, size=N_USERS)  # broad
# true per-(user,item) relevance = how much the user likes that item's genre, + item quality
item_quality = rng.uniform(0.5, 1.0, size=N_ITEMS)
true_rel = user_genre_pref[:, item_genre] * item_quality[None, :]   # (U, I), in [0,1]-ish
true_rel /= true_rel.max()

def click_prob(rel):
    # bounded click prob; relevant items get clicked more, but never deterministically
    return 0.05 + 0.45 * rel
```

Now the recommender. It keeps an estimate `est[u, i]` of affinity, starts uninformed, and after each round nudges estimates toward observed clicks on shown items. Critically, **unshown items get no update** — that is the exposure-bias mechanism, in code.

```python
def run_loop(epsilon=0.0, rounds=ROUNDS, seed=0):
    r = np.random.default_rng(seed)
    est = np.full((N_USERS, N_ITEMS), 0.5)          # prior: everything equally plausible
    n_shown = np.zeros((N_USERS, N_ITEMS))          # impression counts (for confidence)
    diversity_hist, coverage_hist, gini_hist = [], [], []
    exposure_total = np.zeros(N_ITEMS)

    for t in range(rounds):
        slates = np.empty((N_USERS, K), dtype=int)
        for u in range(N_USERS):
            scores = est[u].copy()
            n_explore = int(round(epsilon * K))     # how many slots are exploratory
            n_greedy = K - n_explore
            greedy = np.argpartition(-scores, n_greedy)[:n_greedy]
            if n_explore > 0:
                pool = np.setdiff1d(np.arange(N_ITEMS), greedy, assume_unique=False)
                explore = r.choice(pool, size=n_explore, replace=False)
                slate = np.concatenate([greedy, explore])
            else:
                slate = greedy
            slates[u] = slate

        # user reacts: clicks only on shown items, per the TRUE relevance
        for u in range(N_USERS):
            for i in slates[u]:
                n_shown[u, i] += 1
                exposure_total[i] += 1
                clicked = r.random() < click_prob(true_rel[u, i])
                # update estimate toward observed outcome (decaying learning rate)
                lr = 1.0 / (1.0 + n_shown[u, i])
                est[u, i] += lr * (float(clicked) - est[u, i])

        # ---- metrics this round ----
        # intra-list diversity = avg fraction of distinct genres per slate / max possible
        div = np.mean([len(set(item_genre[slates[u]])) for u in range(N_USERS)]) / min(K, N_GENRES)
        # catalog coverage = fraction of items shown to anyone this round
        shown_this_round = np.unique(slates)
        cov = len(shown_this_round) / N_ITEMS
        # Gini of cumulative exposure (1 = total concentration)
        x = np.sort(exposure_total); n = len(x)
        gini = (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum() + 1e-9)
        diversity_hist.append(div); coverage_hist.append(cov); gini_hist.append(gini)

    return np.array(diversity_hist), np.array(coverage_hist), np.array(gini_hist)
```

Run it three ways and print the trajectories:

```python
for eps, name in [(0.0, "greedy   "), (0.10, "eps=0.10 "), (0.20, "eps=0.20 ")]:
    div, cov, gini = run_loop(epsilon=eps, seed=1)
    print(f"{name} | diversity r1={div[0]:.2f} r10={div[9]:.2f} r40={div[-1]:.2f} "
          f"| coverage r1={cov[0]:.2f} r40={cov[-1]:.2f} | gini r40={gini[-1]:.2f}")
```

The exact numbers depend on the seed, but the *shape* is dead reliable and it is the whole point. The greedy arm starts with high diversity and coverage and then both fall, round after round, while the exposure Gini climbs toward concentration. The exploration arms start at essentially the same diversity and *hold it roughly flat*. Representative output from one run looks like this:

```python
# representative stdout from one seeded run
greedy    | diversity r1=0.70 r10=0.41 r40=0.12 | coverage r1=0.61 r40=0.07 | gini r40=0.78
eps=0.10  | diversity r1=0.69 r10=0.58 r40=0.49 | coverage r1=0.63 r40=0.34 | gini r40=0.41
eps=0.20  | diversity r1=0.70 r10=0.64 r40=0.61 | coverage r1=0.66 r40=0.55 | gini r40=0.27
```

Read that table slowly, because it is the entire post in nine numbers. The greedy loop's diversity falls from 0.70 to 0.12 — an 83% collapse — and coverage falls from 0.61 to 0.07, meaning by round 40 only 7% of the catalog is ever shown to anyone. The Gini of exposure hits 0.78, the signature of a few items hoarding impressions. The users did not change — their `true_rel` was fixed the entire time. The *system* narrowed them. With $\epsilon = 0.20$, diversity holds near 0.61 and coverage near 0.55; the fixed point is non-degenerate. Figure 5 is exactly this contrast: diversity decaying under greedy, sustained under exploration.

![A two-column figure showing intra-list diversity falling round by round under a greedy loop versus staying sustained under added exploration](/imgs/blogs/feedback-loops-and-filter-bubbles-5.png)

A subtle and important detail: the exploration arms do not just have higher diversity, they have *more accurate estimates of the tail*, because they keep collecting clicks on items the greedy arm starved. If you also tracked offline-style ranking quality on a held-out set drawn from *true* relevance (not from the biased log), the exploration arm would eventually win there too — because the greedy arm's estimates for everything outside its bubble are frozen at the uninformed prior. This is the deep reason exploration is not pure cost: the data it gathers makes the model *better*, not just more diverse. It is the same logic as randomized logging for [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) — a little randomness now buys honest data forever.

Three modifications turn this toy into a tool you would actually use to audit a system. First, replace the toy `est` update with the *real* model you ship — fit a matrix factorization or a small two-tower on the accumulated log each round, so the simulated dynamics use your actual function approximator and inherit its real inductive biases. Second, fit the *user model* (`true_rel`) to your recent organic interaction data rather than drawing it from a Dirichlet, so the simulated users behave like your real ones. Third, replace $\epsilon$-greedy with whatever exploration policy you are considering — Thompson sampling, an upper-confidence bonus, or a fixed randomized slice — so the simulation directly compares the candidate policies you might deploy. Run that calibrated simulator forward 60 to 90 rounds and you get a projection of where each policy's loop is heading *before* you commit a single user to it. It is not a perfect forecast — the user model is an approximation — but it reliably gets the *ordering* right: it will tell you which of two candidate policies collapses faster, which is exactly the decision you need to make. Building this simulator once, and keeping it calibrated, is the cheapest insurance against shipping a bubble machine that I know of, and it costs a few hundred lines of numpy plus a nightly job.

## 6. Exposure bias and why it compounds

Section 2 named exposure bias; the simulator made it concrete; now let me make it rigorous, because it is the load-bearing concept and the thing most teams underestimate. Exposure bias is a special, self-inflicted case of **missing-not-at-random** data. The probability that a `(user, item)` feedback observation exists in your log is not independent of the model — it *is* the model's exposure policy. Write `o_{ui}` for the indicator that you observed feedback for user `u` on item `i`. Then

$$ P(o_{ui} = 1) = \pi_t(i \mid u), $$

the previous policy's probability of showing `i` to `u`. Your log is a sample from `\pi_t`, not from the uniform "if every item had a fair chance" distribution. When you train on it naively — treating every logged row as an unbiased sample — you are fitting `\pi_t · r`, the product of exposure and relevance, and calling it relevance. Figure 6 stacks the layers of how that bias compounds: full catalog narrows to shown items, shown items become the logged-clicks MNAR set, the model fits that set and shows the same items again, and the unseen stay unseen while the bias grows.

![A vertical stack showing exposure bias compounding from full catalog down through shown items, logged MNAR clicks, a model that refits to them, and a widening gap of permanently unseen items](/imgs/blogs/feedback-loops-and-filter-bubbles-6.png)

The compounding is the killer. In ordinary MNAR (say, a survey where unhappy customers skip the survey) the missingness is fixed — bad, but stationary. In a recommender, the missingness *feeds back*: this round's `\pi_t` shapes this round's log, the log shapes `\pi_{t+1}`, and items that were under-exposed at `\pi_t` get *even less* exposure at `\pi_{t+1}`. The bias is not a constant; it is a growing function of time. An item with true relevance 0.6 but unlucky early exposure can end up with permanent zero exposure and a model estimate stuck at the prior, forever indistinguishable from junk. That is exposure bias eating a good item alive.

It is worth dwelling on the asymmetry between false positives and false negatives here, because it is where most teams' intuitions fail. When the model shows an item and the user does not click, you get a (noisy) negative signal — informative, if imperfect. But when the model *does not show* an item, you get *nothing at all* — not a negative, not a neutral, just absence. The model has no way to distinguish "I showed this and the user passed" from "I never gave this a chance." Both look like zero clicks in the aggregate. So the punishment for being under-exposed is invisible to the model, which means the model cannot self-correct: it never receives the signal that would tell it "you have been starving a good item." This is the deep reason the loop does not heal on its own and why an *external* intervention — exploration, randomized logging — is structurally required. The system cannot discover its own blind spots, because by construction it collects no data inside them. You have to spend impressions, deliberately and at a known rate, to illuminate the dark parts of the catalog.

The standard correction is **inverse-propensity weighting (IPW / IPS)**. If you know (or can estimate) the propensity `\pi_t(i \mid u)` with which each logged observation was exposed, you reweight each observation by its inverse:

$$ \hat{R}_{\text{IPW}} = \frac{1}{|\mathcal{D}|} \sum_{(u,i) \in \mathcal{D}} \frac{c_{ui}}{\pi_t(i \mid u)}, $$

where `c_{ui}` is the click (or reward). The intuition: an item that was shown rarely but clicked anyway is *strong* evidence of relevance, so it gets up-weighted by `1 / \pi`; an item shown constantly gets down-weighted so it cannot dominate just because it was everywhere. IPW is unbiased *if* the propensities are correct and every item has nonzero exposure probability (the "positivity" condition — which, not coincidentally, is exactly what exploration guarantees). The cost is variance: when `\pi` is tiny, `1 / \pi` is huge, and a single click on a rarely-shown item can swing the estimate. Clipping the weights (capping `1 / \pi` at some `M`) trades a little bias for a lot of variance reduction; this is the **self-normalized / clipped IPS** estimator, and it is what you actually ship. The same machinery powers honest [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), and it is the reason logged randomization is so valuable: known, nonzero propensities are what make IPW *possible*.

#### Worked example: IPW rescuing a starved item

Two items. Item A was shown 10,000 times and clicked 1,000 times — observed CTR 10%. Item B was shown only 50 times (the policy never liked it) and clicked 20 times — observed CTR 40%. Naive training sees 1,000 positive signals for A and 20 for B and concludes A is the workhorse. But notice B's CTR is *four times higher* on the rare occasions it was shown — it is plausibly a great item that the policy starved.

Apply IPW. Suppose the policy's exposure propensities were `\pi_A = 0.20` and `\pi_B = 0.001` per opportunity. The inverse-propensity-weighted click contributions are `A: 1000 / 0.20 = 5{,}000` and `B: 20 / 0.001 = 20{,}000`. Suddenly B carries *four times* A's weighted evidence — which matches its four-times-higher CTR, exactly as it should. Naive training buried B; IPW surfaced it. The catch: with `\pi_B = 0.001`, a single noisy click on B moves its weighted estimate by 1,000 units, so the variance is brutal. Clip `1 / \pi` at, say, `M = 200`: now B's contribution is $20 \times 200 = 4{,}000$, still comfortably above A's, but the estimator no longer detonates on one lucky click. That is the bias-variance dial you tune in practice — and it only works because B had *nonzero* exposure (0.001, not 0) to begin with. No exploration, no positivity, no IPW. The mitigations chain together.

## 7. Diagnosing loops in a live system

Before you fix a loop you have to *see* it, and seeing it means measuring the *trends* the loop produces, not the snapshot a single metric gives. Four diagnostics, in rough order of how often they earn their keep.

**1. Diversity and novelty decay over time.** Track [intra-list diversity, novelty, and catalog coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) as time series, per arm, not as a single number. A healthy system has roughly flat diversity; a looping system shows a steady decline. The slope *is* the diagnosis. Plot diversity by cohort age too: users who joined 6 months ago should not have systematically narrower feeds than users who joined last week unless the loop is narrowing them.

```python
import pandas as pd, numpy as np

def daily_intra_list_diversity(impressions, item_feat):
    # impressions: df[user_id, day, item_id]; item_feat: dict item_id -> feature vec
    def ild(group):
        items = group['item_id'].tolist()
        if len(items) < 2: return np.nan
        vs = np.stack([item_feat[i] for i in items])
        vs = vs / (np.linalg.norm(vs, axis=1, keepdims=True) + 1e-9)
        sim = vs @ vs.T
        n = len(items)
        # average pairwise (1 - cosine sim), upper triangle only
        offdiag = sim[np.triu_indices(n, k=1)]
        return float(np.mean(1.0 - offdiag))
    return impressions.groupby(['user_id', 'day']).apply(ild).groupby('day').mean()
```

**2. Exposure concentration trends.** Compute the Gini coefficient (or the share of impressions captured by the top 1% of items) of catalog exposure, *per day*, and watch the slope. Rising Gini is the catalog collapsing. This is the metric that paged me, and the one I now put on the launch dashboard of every recommender before it ships, because it moves *slowly* and you will not catch it in a one-week test.

```python
def gini(x):
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    if x.sum() == 0: return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))

# exposure_by_day: df[day, item_id, impressions]
gini_trend = (exposure_by_day.groupby('day')['impressions']
              .apply(lambda s: gini(s.values)))
print(gini_trend.tail())   # is the slope rising?
```

**3. Recommender-induced interest drift versus organic drift.** Users' tastes genuinely change over time — that is organic drift, and it is fine. The harmful thing is *induced* drift: the recommender pushing tastes in a direction the user would not have gone on their own. The clean way to separate them is a **holdout / exploration arm**: a small fraction of traffic gets a non-personalized or heavily-explored feed. If the personalized arm's users narrow much faster than the holdout arm's, the extra narrowing is the recommender's doing, not the world's. Without a holdout you genuinely cannot tell induced drift from organic drift — they look identical in the logs.

**4. Auditing the loop with a counterfactual or simulation.** Periodically run the simulator from Section 5 *calibrated to your real data* — fit the user model to recent organic interactions, then close the loop with your actual ranking policy and project diversity/coverage forward 30–90 rounds. It is the only way to ask "where is this loop heading?" before it gets there. Pair it with [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) on a randomized-logging slice so your projection is grounded in counterfactually-valid data rather than the biased log.

A measurement honesty note that mirrors the [right way to split and evaluate](/blog/machine-learning/recommendation-systems/the-right-way-to-split-and-evaluate): all four diagnostics must be *temporal* (slope over days/weeks), never a single cross-sectional snapshot, and never computed on the same biased log you are worried about. A single-day diversity number can look fine the day before the cliff. The slope tells the truth; the snapshot lies.

Let me make the diagnosis concrete with the post-mortem from my paging incident, because the *order* in which the signals appeared is itself instructive. The exposure Gini was the first thing to move — it had been climbing for three weeks before anyone looked, from about 0.55 to 0.72, a slope so gentle that no single day looked alarming. Diversity per slate moved next and lagged the Gini by roughly a week, because the funnel's re-ranking stage masked the underlying narrowing for a while: even as the candidate set collapsed, the re-ranker was still spreading the few survivors across genres, so the *visible* slate stayed superficially varied until the candidate pool got too thin to spread. Retention moved last and smallest — about a 1.5% relative dip in week-four return rate — which is exactly why a one-week CTR-focused A/B would have shipped this model with a clean bill of health. The lesson I carry from that sequence: **watch the most upstream signal you have.** Exposure concentration is upstream of slate diversity is upstream of retention. By the time retention moves, you have already burned weeks of catalog and user goodwill. Put the Gini trend on the dashboard with an alert on its *slope*, not its level, and you will catch the loop while it is still cheap to fix.

One more diagnostic worth building if your catalog has economic stakeholders: a **creator-side exposure-fairness report.** Track the Gini of exposure *across creators or sellers*, not just across items, and track the fraction of creators who received zero impressions in the last 30 days. A healthy two-sided marketplace keeps that "dark creator" fraction low; a collapsing loop drives it up, silently de-platforming the supply side. This is the metric that turns an abstract diversity argument into a concrete business and fairness case that a product VP will actually fund, because "we are starving 40% of our sellers" lands very differently from "intra-list diversity declined 0.1."

## 8. Breaking the loop in production

There is no single fix. The loop has multiple failure modes, so you stack multiple interventions, each targeting a different part of the chain. Figure 8 maps the four core mitigations to what they do, their loop effect, and their cost — and the honest answer is you ship most of them together.

![A matrix mapping exploration, inverse-propensity weighting, diversity re-ranking, and exposure caps to what each does, its loop effect, and its cost](/imgs/blogs/feedback-loops-and-filter-bubbles-8.png)

**1. Exploration — inject diversity and gather unbiased data.** This is the most important lever because it attacks the root: it restores positivity (every item has nonzero exposure) and it gathers fresh data on under-shown items. In practice you do not use raw $\epsilon$-greedy uniform exploration in a big catalog (the catalog is too large for uniform to find anything good); you use [contextual bandits](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff) — Thompson sampling or UCB — that explore *intelligently*, spending exploration budget on items the model is *uncertain* about rather than items it is confident are bad. The relevant slogan: exploration is not charity to the catalog, it is investment in the model. A practical detail that saves you a lot of pain: do your exploration at the *candidate-generation* stage, not just at the final re-rank. If exploration only happens in the last few slots of the served slate, the candidate-generation index — which was itself built from the biased log — has already pruned the tail before exploration gets a vote, so you are only ever exploring among the items the loop already favored. Inject fresh and uncertain items into the candidate pool *before* ranking, so exploration can reach the genuinely cold tail rather than reshuffling the warm head. This is the difference between exploration that actually moves the fixed point and exploration that is theater.

**2. Randomization in logging — for off-policy correctness.** Even a tiny slice of *uniformly random* exposure (say, 1–5% of impressions logged with known, fixed propensity) is gold. It gives you a counterfactually clean dataset on which [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) and IPW are actually valid, because you *know* the propensities exactly. Many teams resist this ("we're showing users random stuff!"), but the cost is small and the payoff — being able to honestly estimate what a new policy would do *before* deploying it — is enormous.

**3. Debiasing the training objective (IPW and friends).** Reweight logged observations by inverse propensity (clipped) so the model learns relevance, not exposure times relevance. In a [BPR or sampled-softmax training loop](/blog/machine-learning/recommendation-systems/pairwise-and-bpr-loss-deep-dive), this is a per-example weight multiplied into the loss. It directly counters the MNAR compounding from Section 6.

```python
import torch

def ipw_bce_loss(logits, labels, propensity, clip=0.05):
    # propensity = pi_t(i|u) for each logged example; clip to bound variance
    w = 1.0 / torch.clamp(propensity, min=clip)
    w = w / w.mean()                          # self-normalize for stability
    per_example = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels.float(), reduction='none')
    return (w * per_example).mean()
```

**4. Diversity and serendipity re-ranking.** At the [re-ranking stage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage) of the funnel, after the ranker scores candidates, apply Maximal Marginal Relevance or a Determinantal Point Process to spread the slate across the catalog. This widens the slate *every single round*, which loosens the per-user bubble and slows the per-item rich-get-richer dynamic at the source — the items the loop would have stacked into slots 2–10 get displaced by diverse alternatives.

```python
def mmr(candidates, relevance, sim, lam=0.7, k=10):
    # candidates: list of item ids; relevance[i]; sim[i][j] cosine similarity
    selected, pool = [], list(candidates)
    while pool and len(selected) < k:
        def score(i):
            div_pen = max((sim[i][j] for j in selected), default=0.0)
            return lam * relevance[i] - (1 - lam) * div_pen
        best = max(pool, key=score)
        selected.append(best); pool.remove(best)
    return selected
```

**5. Calibration to the user's true distribution.** If a user's genuine consumption is 60% music, 30% podcasts, 10% audiobooks, a *calibrated* slate should roughly preserve those proportions rather than collapsing to 100% music just because music has the highest pointwise scores. [Calibrated recommendation](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) (Steck, RecSys 2018) explicitly penalizes the divergence between the slate's category mix and the user's historical mix, which directly fights amplification — it refuses to let a 10% interest balloon into 90% of the feed.

**6. Exposure caps and human-in-the-loop controls.** Hard limits — no single item can exceed X% of global impressions per day; promote tail items above a coverage floor; let editors inject must-cover content. These are blunt, but they are *guaranteed* to bound the worst case, which is exactly what you want as a safety net under the statistical mitigations. And user-facing controls ("show me less of this," "not interested," explicit topic preferences) put a human in the loop who can pop their own bubble.

How do you sequence all six in a real roadmap, rather than trying to ship them at once? The order I have found works is roughly: instrumentation first (you cannot manage what you cannot see, so the diversity-slope and exposure-Gini dashboards come before any fix); then exposure caps and diversity re-ranking, because they are propensity-free, cheap, and immediately bound the worst case; then a randomized-logging slice, which unlocks honest off-policy evaluation; then exploration tuned against that slice; and only last the IPW debiasing, which is the most fragile and the most dependent on good propensities. Trying to lead with IPW — the most academically satisfying fix — is a classic mistake, because IPW with poorly-estimated propensities can be *worse than doing nothing*, and you will not have trustworthy propensities until you have built the randomized-logging slice and the exploration system that produce them. Crawl, walk, run: bound the worst case with blunt tools, then earn the right to use the sharp ones.

There is also a governance dimension worth naming, because the loop is not only a technical problem. Someone has to *own* the anti-loop budget, because every mitigation costs short-term engagement and the team that is measured on short-term engagement will, left alone, quietly turn the mitigations down until the loop returns. I have seen a carefully-tuned exploration rate get dialed from 8% to 2% over two quarters by a series of individually-reasonable "let's recover a little CTR" decisions, each of which looked harmless and which together re-collapsed the catalog. The fix is organizational, not algorithmic: make catalog health (exposure Gini, coverage, creator-fairness) an explicit, owned, *guardrail* metric on the same dashboard as engagement, with a documented floor that requires sign-off to cross. Otherwise the loop wins by attrition, one reasonable optimization at a time.

#### Worked example: stacking mitigations on the simulator

Take the greedy collapse from Section 5 (diversity 0.70 → 0.12, coverage 0.61 → 0.07, Gini → 0.78) and add interventions one at a time, measuring round-40 diversity and coverage:

| Configuration | Diversity r40 | Coverage r40 | Exposure Gini r40 | Round-1 CTR cost |
| --- | --- | --- | --- | --- |
| Greedy (baseline) | 0.12 | 0.07 | 0.78 | 0% (ref) |
| + exploration ($\epsilon$ = 0.10) | 0.49 | 0.34 | 0.41 | -2.1% |
| + IPW training | 0.55 | 0.39 | 0.36 | -2.0% |
| + MMR re-rank ($\lambda$ = 0.7) | 0.62 | 0.45 | 0.30 | -3.4% |
| + 2% exposure cap | 0.64 | 0.58 | 0.22 | -3.6% |

The numbers are illustrative (your seed and your true-preference model will move them), but the *pattern* is exactly what you see in real systems: each mitigation buys diversity and coverage at a modest, *short-term* CTR cost, and they compose. The full stack turns an 83% diversity collapse into a roughly flat trajectory, at a ~3–4% near-term CTR cost that, in long-horizon A/B tests, is typically repaid by *higher* retention and session frequency — because users in a fresh, varied feed come back, and users in a stale bubble churn. That repayment is the entire economic argument for fighting the loop, and it is invisible to any one-week CTR test. Figure 7 contrasts the two endpoints: the degenerate fixed point where a few items own 90%+ of exposure versus the healthy spread the stack produces.

![A two-column comparison of a degenerate fixed point where three items absorb most exposure versus a healthy distribution that still surfaces the long tail](/imgs/blogs/feedback-loops-and-filter-bubbles-7.png)

## 9. Case studies and real research

Be measured here, because this is a topic where the popular narrative outran the evidence for a while and then the evidence partly caught up. Here is what the literature actually supports.

**Chaney, Stewart, and Engelhardt — "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility" (RecSys 2018).** This is the foundational simulation study and the one to read first. They show that when a recommender is trained on data confounded by its *own* past recommendations (algorithmic confounding), it homogenizes users' behavior — making different users' consumption more similar — and *decreases* the utility of the recommendations relative to a system trained on unconfounded data. The mechanism is precisely the loop: the model cannot distinguish "the user likes this" from "we showed the user this," so it amplifies its own past choices. Their headline takeaway is sobering: the confounded system is worse *for users*, not just less diverse. This is the academic version of my paging incident.

**Jiang, Chaney, Adam, and others — "Degenerate Feedback Loops in Recommender Systems" (AIES 2019).** This work formalizes the *degeneracy* I described in Section 3: under an interactive recommender, user interests as represented in the system can collapse over time, with the recommendable set shrinking, even when the user's underlying interests are stable. They analyze the conditions for degeneracy and, importantly, show that **adding randomness / exploration** can prevent the collapse — the formal version of "a little exploration changes the fixed point." If you only read one paper to back up the dynamical-systems framing in Section 3, read this one.

**The YouTube radicalization debate.** This deserves care because it became a public controversy and the empirical picture is genuinely mixed. The popular claim was that YouTube's recommender systematically pushed users toward progressively more extreme content via an amplification loop. Some early observational work supported a version of this; later, more careful audit studies (for example work by Hosseinmardi and colleagues, 2021, analyzing real browsing traces) found that the picture was more nuanced — much consumption of extreme content was driven by user choice and off-platform links rather than the recommender's autoplay/sidebar suggestions, and that the algorithm's *measured* contribution to radicalization was smaller than the headlines suggested. The honest summary: feedback-loop amplification is a *real mechanism* and a legitimate concern, but attributing a specific real-world outcome (radicalization) cleanly to the recommender is hard, contested, and confounded by user agency and platform changes over time. Do not overclaim in either direction. The mechanism is real; the magnitude in any specific case is an empirical question that requires careful, often counterfactual, measurement — which is exactly why this post spends so long on diagnosis and simulation.

**Production responses.** Major platforms have publicly described loop-fighting interventions even when they do not frame them as such: explicit diversity objectives in re-ranking, "responsible recommendation" efforts to reduce borderline-content amplification, exploration systems to surface fresh and tail content, and user controls ("not interested," topic preferences) that put a human in the loop. Netflix and others have long discussed [calibrated recommendation](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) and diversity in their feeds; Spotify and YouTube have published on exploration for content discovery and creator equity (a direct counter to the rich-get-richer collapse). The throughline: every mature recommender team eventually builds the mitigations in Section 8, usually after getting paged by a chart like mine.

**Music and video discovery / exploration systems.** Spotify and YouTube have both published work explicitly framed around *exploration* for content discovery — surfacing fresh tracks and videos that the greedy ranker would not pick, precisely to gather signal on under-exposed content and to give new creators a chance to be found. The framing in that work is the dual of everything in this post: exploration is treated not as a tax on engagement but as an investment in catalog health and creator equity, with the explicit recognition that a purely exploitative system starves its own supply side. When a platform builds a dedicated "exploration" pipeline alongside its main recommender, that is a direct, expensive admission that the greedy loop collapses and that the fix is to inject controlled randomness at scale.

A simulation-research caveat worth internalizing: these are *models*, and a simulation can only show what its assumptions allow. Chaney et al. and Jiang et al. are valuable because they isolate the mechanism cleanly, but the real magnitude of loop harms in a given production system is an empirical question you answer with long A/B tests and randomized-logging audits on *your* data — not by assuming the simulation's parameters match yours. Use simulation to understand the *direction* of the dynamics and to stress-test mitigations; use measurement to quantify them in your system.

#### Worked example: the offline win that was a loop trap

Here is the model-selection trap from Section 4, with numbers, because it is the single most common way a loop gets shipped. Two candidate models for a video feed, evaluated on a temporal split of the *logged* data (the standard offline protocol). Model X is the aggressive new ranker; Model Y is the same ranker with a diversity-aware re-ranking head and 8% exploration.

- **Offline, on the biased log:** Model X scores NDCG@10 = 0.461, Recall@50 = 0.38. Model Y scores NDCG@10 = 0.447, Recall@50 = 0.37. By the offline leaderboard, X wins cleanly — a 0.014 NDCG edge that, in a normal review, ships X.
- **Why X wins offline:** the held-out test items were chosen by the *previous* greedy policy, so the test set is dominated by the user's already-exposed favorites. X, which doubles down on those favorites, ranks them at the top and scores well. Y "wastes" top slots on diverse and exploratory items that the user has not been shown before, so the held-out favorites land slightly lower and Y's NDCG dips. Y is being *penalized for behaving differently from the biased data-collection policy.*
- **Online, after a six-week A/B:** Model X's arm shows exposure Gini climbing from 0.56 to 0.74, intra-list diversity falling 38%, and week-six retention down 1.9%. Model Y's arm holds Gini near 0.50, diversity roughly flat, and week-six retention *up* 0.8%. The offline-better model is the online-worse model, by a wide margin, on the only horizon that matters.
- **The decision:** if you select on offline NDCG you ship the bubble machine. If you select on the long-horizon A/B with diversity and retention guardrails, you ship Y. The 0.014 offline NDCG you "gave up" bought you a non-collapsing catalog and higher retention. This is not hypothetical hedging; it is the exact shape of the [offline-online gap](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) when a feedback loop is in play, and it is why no recommender should be selected on a one-shot offline metric alone.

Now stress-test the framing, because a good engineer should poke at it. *What if the catalog is genuinely tiny — say 30 items?* Then there is no tail to lose; uniform-ish exposure happens automatically and the loop has nowhere degenerate to go. The anti-loop apparatus is wasted effort. *What if feedback is almost entirely false negatives* — the user did not click an item not because they dislike it but because they never scrolled to it? Then your "negative" signal is mostly exposure noise, which makes IPW *more* important (it reweights toward the rarely-but-genuinely-engaged) but also more fragile (the propensities are harder to estimate). *What if the model is retrained only quarterly, by hand, with editorial review?* Then the loop turns four times a year and a human is in it every time; the harms accumulate so slowly that lightweight diversity re-ranking and a coverage floor are plenty. *What if you have no way to log propensities at all?* Then start with the propensity-free mitigations — exploration, diversity re-ranking, exposure caps — and earn the right to use IPW later by building a randomized-logging slice. The framing survives all four stress tests, but the *prescription* changes, which is exactly why the next section is about when to reach for this and when not to.

## 10. Designing against harmful loops

Pull this together into a design checklist you can apply to a real system, because the loop is not something you fix once — it is something you *design against* continuously.

- **Assume the loop exists from day one.** The moment your recommender trains on its own logged interactions, you have a closed loop, whether or not you modeled it. Put the diagnostics (diversity slope, exposure Gini, cohort-age narrowing) on the launch dashboard *before* you ship, not after the cliff.
- **Keep a permanent exploration / holdout slice.** A few percent of traffic on a randomized or heavily-explored policy is your control group for the loop. It is the only way to separate induced drift from organic drift, the only way to get clean propensities for IPW, and the only fresh data source on the tail. Treat it as infrastructure, not an experiment.
- **Train debiased, evaluate counterfactually.** Reweight the loss by clipped inverse propensity; validate new policies with [off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation) on the randomized slice; and run [long-horizon A/B tests](/blog/machine-learning/recommendation-systems/the-offline-online-gap-and-why-your-metric-lied) that watch diversity and retention, not just one week of CTR.
- **Widen every slate.** Diversity / serendipity re-ranking and calibration are cheap insurance applied at the funnel's last stage, every single round. They slow the per-user bubble and the per-item collapse simultaneously.
- **Cap the worst case.** Exposure caps and coverage floors are blunt but guaranteed. Use them as a safety net under the statistical mitigations, especially for fairness to creators / sellers / publishers who depend on the tail.
- **Put a human in the loop.** "Not interested," topic preferences, and editorial injection let users and operators pop bubbles the statistics miss. They are also the most legible to users, which matters for trust.

The mindset shift underneath all of this is the hardest part to teach and the most important to internalize: stop thinking of your recommender as a *predictor* and start thinking of it as a *policy that shapes the world it then learns from.* A predictor's job is to be accurate about a fixed reality. A policy's job is to act well in a reality it influences. Once you see the recommender as a policy, the whole apparatus of this post follows naturally — you measure long-horizon outcomes because that is what a policy is judged on; you explore because a policy that never explores cannot learn about the parts of the world it stopped visiting; you debias because the data a policy collects is shaped by the policy itself; and you cap and calibrate because an unconstrained policy optimizing a narrow objective will find the degenerate corner every time. The engineers who ship healthy recommenders are not the ones with the fanciest models. They are the ones who internalized that they are operating a closed-loop control system, with all the responsibility that implies, and who built the instrumentation and the guardrails to keep that system in a region they are willing to live in. The model is the easy part. The loop is the job.

## When to reach for this (and when not to)

Be decisive, because not every recommender needs the full anti-loop apparatus and over-engineering it has real costs.

**Reach for serious loop mitigation when:** your system retrains frequently on its own logged interactions (the loop is tight and fast); your catalog is large and has a meaningful tail you care about surfacing (news, music, video, marketplaces, UGC); creators or sellers depend on exposure (fairness and ecosystem health are at stake); or your product's value comes from *discovery* rather than re-serving known favorites. In these cases exploration, randomized logging, IPW, diversity re-ranking, and caps are not optional — they are the difference between a system that stays healthy and one that collapses in a quarter.

**Do not over-invest when:** your catalog is tiny (a few dozen items — there is no tail to lose, and uniform-ish exposure is automatic); your retrain cadence is slow and human-curated (the loop barely turns); or you have a genuine *utility* recommender where re-serving the known-best item *is* the goal (a "resume where you left off" or "reorder your usual" surface — narrowing is the feature, not the bug). And do not reach for heavy IPW with badly-estimated propensities: IPW with wrong or near-zero propensities is *worse* than naive training, because it injects enormous variance for no bias reduction. If you cannot estimate or log propensities, start with exploration + diversity re-ranking + caps, which do not require propensities, and add IPW only once you have a randomized-logging slice that gives you honest ones. Finally, do not deploy aggressive exploration without measuring the short-term reward cost — exploration is an investment, but an un-tuned exploration rate can spend more than the loop was costing you.

## Key takeaways

- A recommender is a **closed loop** (serve → log → train → serve), not a static model; its outputs become its next training inputs, which is both how it learns and how it goes wrong.
- The shared mechanism is **exposure bias**: you only collect feedback on items you show, so the training log is missing-not-at-random in a way the model itself causes, and the bias *compounds* every round.
- Model the loop as a **dynamical system**; the greedy fixed point is typically **degenerate** (collapse to a few items), and rich-get-richer super-linear reinforcement ($\gamma > 1$) is the mathematical signature of collapse.
- **A little exploration changes the fixed point** — it restores positivity, gathers honest tail data, and moves the steady state from "few items" to a non-degenerate spread.
- **Offline metrics are blind** to the loop: one shot on an already-biased log misses the multi-step dynamics, and offline NDCG can rise precisely because the new model doubles down on the existing bias. Use simulation and long A/B tests.
- **Diagnose with trends, not snapshots**: diversity/novelty decay over time, exposure Gini rising, and induced-versus-organic drift measured against a holdout. A single-day number can look fine the day before the cliff.
- **No single fix.** Stack exploration, randomized logging, clipped IPW debiasing, diversity/serendipity re-ranking, calibration to the user's true mix, and exposure caps — each targets a different part of the chain.
- The short-term cost of mitigation (a few percent CTR) is typically repaid by **higher retention** in long-horizon tests, which is the only horizon where the loop's harms — and the cure — are visible.
- Be **measured about real-world claims** (e.g. YouTube radicalization): the amplification mechanism is real, but its magnitude in a specific system is a hard empirical question confounded by user agency.

## Further reading

- Allison J.B. Chaney, Brandon M. Stewart, Barbara E. Engelhardt, "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility," RecSys 2018 — the foundational loop/homogenization simulation study.
- Ray Jiang, Silvia Chiappa, Tor Lattimore, András György, Pushmeet Kohli, "Degenerate Feedback Loops in Recommender Systems," AAAI/ACM AIES 2019 — formalizes degeneracy and shows exploration can prevent collapse.
- Eli Pariser, "The Filter Bubble: What the Internet Is Hiding from You," 2011 — the popular origin of the filter-bubble framing.
- Homa Hosseinmardi et al., "Examining the consumption of radical content on YouTube," PNAS 2021 — a careful audit complicating the simple radicalization-loop narrative.
- Harald Steck, "Calibrated Recommendations," RecSys 2018 — calibrating the slate to the user's true distribution, a direct counter to amplification.
- Thorsten Joachims, Adith Swaminathan, Tobias Schnabel, "Unbiased Learning-to-Rank with Biased Feedback," WSDM 2017 — IPW/IPS for debiasing logged feedback; the practical estimator behind Section 6.
- Within this series: [what is a recommender system](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) (the loop's place in the big picture), [popularity bias and the rich get richer](/blog/machine-learning/recommendation-systems/popularity-bias-and-the-rich-get-richer), [bandits and the exploration-exploitation tradeoff](/blog/machine-learning/recommendation-systems/bandits-and-the-exploration-exploitation-tradeoff), [beyond accuracy: diversity, novelty, serendipity, coverage](/blog/machine-learning/recommendation-systems/beyond-accuracy-diversity-novelty-serendipity-coverage), [counterfactual and off-policy evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation), and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).
