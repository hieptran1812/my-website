---
title: "The Quant Curriculum Map: What to Learn, in What Order"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The master index of the quant knowledge portfolio: a sequenced, role-aware curriculum for traders, researchers, and developers that tells you exactly what to learn first, why, and which deep-dive teaches it — because you cannot learn everything, so you sequence by leverage."
tags: ["quant-careers", "quant-finance", "careers", "curriculum", "study-plan", "probability", "mental-math", "programming", "markets", "self-study", "interview-prep", "learning-roadmap"]
category: "trading"
subcategory: "Quant Careers"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — You cannot learn everything a quant could conceivably know, so the only sane strategy is to sequence the curriculum by leverage: learn the highest expected-value things first, weighted to your target role.
>
> - The whole portfolio sits in five pillars — probability and statistics, mental math, programming, markets literacy, and the math foundation — plus a research layer on top for QR.
> - The pillars are not independent: arithmetic and probability sit at the base and unlock almost everything else, so they go first regardless of role.
> - Returns to study are steeply diminishing — the first ~40 hours on a gating topic buy most of the lift; the 200th hour on a non-gating one buys almost nothing.
> - The single number to remember: order beats volume. A focused 6-month plan that front-loads the gates beats a scattered 18-month one that reads 50 books.

A student emails me a photo of their desk. On it: a stack of fourteen books. *Options, Futures, and Other Derivatives.* The two-volume *Probability and Statistics* set. A measure-theory text. A 900-page C++ reference. *Stochastic Calculus for Finance*, both volumes. A stats-for-machine-learning brick. Three competitive-programming books. The message reads: "I want to break into quant. I've collected the reading list everyone posts online. I have maybe six months before applications. Where do I even start? I'm paralyzed."

This is the most common failure mode I see, and it has nothing to do with talent. The student is smart enough to read every one of those books. What they are missing is not capacity — it is *sequence*. They have a pile of topics and no ordering function, so they freeze, or worse, they start with whatever sounds most impressive (usually stochastic calculus, which for most of them is the single worst place to begin). Six months later they can recite the Black-Scholes derivation and they still cannot multiply 47 × 23 in their head fast enough to clear an Optiver screen.

The fix is a map. Not a longer reading list — a shorter, ordered one, with an explicit rule for what to learn first. That rule is the spine of this entire series: *the job is a probabilistic edge, and so is preparing for it.* Every hour of study is a bet with an expected payoff. Some hours buy a lot of interview-pass probability; some buy almost none. The skill of building a curriculum is the skill of ranking those bets and spending your scarce hours on the top of the list. This post is that ranking — the master index of the knowledge track, with every node routed to a deep-dive so you never have to relearn the technique from scratch.

![A tree diagram of the five quant knowledge pillars — probability and statistics, mental math, programming, markets literacy, and the math foundation — each branching into its concrete sub-topics.](/imgs/blogs/the-quant-curriculum-map-what-to-learn-in-what-order-1.png)

If you read one post in the knowledge track, read this one. The six sibling posts each go deep on one pillar; this is the hub that tells you which to open, when, and why — and routes you out to the [math-for-quants](/blog/trading/math-for-quants/probability-spaces-random-variables-math-for-quants) and quant-interview series when you need the underlying technique rather than the career framing.

## Foundations: the five knowledge pillars and how they fit

Before we can sequence anything, we need the pieces. Everything a quant interview tests and the job demands sits in five knowledge pillars, plus one layer that sits on top for researchers. Let me define each from zero, because the whole point of this series is that you might be brilliant and still new to how this industry carves up its knowledge.

**Pillar 1 — Probability and statistics.** This is the lingua franca of the entire field. Probability is the math of uncertainty: how to compute the chance of an event, how to update a belief when new information arrives (that's Bayes' rule), and how to compute an *expected value* — the probability-weighted average of all outcomes, which is the single most important quantity in trading. Statistics is the inverse problem: given data, what can you infer about the process that generated it? Estimators (the sample mean, the regression coefficient), distributions (normal, binomial, Poisson), and hypothesis tests (is this signal real or noise?) all live here. If a quant has one superpower, it is fluency in expected value under uncertainty. The sibling deep-dive is [the probability and statistics you must own](/blog/trading/quant-careers/the-probability-and-statistics-you-must-own); the underlying math lives in the [math-for-quants probability track](/blog/trading/math-for-quants/probability-spaces-random-variables-math-for-quants).

**Pillar 2 — Mental math and estimation.** Market makers (firms that quote a buy price and a sell price on a security and profit from the difference, called *the spread*) need to do arithmetic fast and accurately under pressure, because prices move in milliseconds. So the screening interview for Jane Street, Optiver, SIG, IMC, Akuna, and Citadel Securities is often a timed arithmetic sprint: roughly 60 to 80 questions in 8 minutes, no calculator, no scratch paper, with a pass bar around 70 to 85 percent correct. Estimation is the cousin skill: Fermi problems ("how many piano tuners are in Chicago?") that test whether you can decompose an unknown quantity into knowable pieces and stay *calibrated* — meaning your confidence matches your accuracy. The sibling post is [mental math and estimation as a trainable skill](/blog/trading/quant-careers/mental-math-and-estimation-as-a-trainable-skill).

**Pillar 3 — Programming and data structures.** Quant developers (QD) and many researchers (QR) write production code. The bar splits by role. Research leans on Python for data wrangling, backtesting, and modeling. Low-latency firms — Jump, HRT, Citadel Securities — lean hard on C++ because they are fighting for microseconds, which means you need to understand the memory model, undefined behavior, templates, lock-free structures, and how the CPU cache works. On top of the language sits the *DSA bar*: data structures and algorithms, the competitive-programming-adjacent material that HackerRank and Codility screens test. The sibling is [programming for quants: Python, C++, and the DSA bar](/blog/trading/quant-careers/programming-for-quants-python-cpp-and-the-dsa-bar).

**Pillar 4 — Markets literacy.** You can be a probability genius and still get cut in a trading game because you do not know what a *limit order book* is, why the *bid-ask spread* exists, what *fair value* means, or how an option's price relates to the underlying. This pillar is the minimum vocabulary of how markets actually work — order books, microstructure, the difference between a market order and a limit order, what a derivative is, and what "no-arbitrage" means. It is small but non-optional, and it is the pillar self-taught candidates most often skip. The sibling is [the markets knowledge every quant needs](/blog/trading/quant-careers/the-markets-knowledge-every-quant-needs).

**Pillar 5 — The math foundation.** Linear algebra (vectors, matrices, eigen-decomposition, principal component analysis), multivariable calculus (gradients, optimization), and — for some research roles — stochastic calculus (Brownian motion, Itô's lemma, the math behind continuous-time pricing). The critical honesty here, which we will return to: most of stochastic calculus is *interview theater* for a trader and a genuine daily tool only for a narrow set of derivatives-research seats. Learning it first is the single most common sequencing mistake. The sibling is [the math foundation: linear algebra, calculus, stochastic calc](/blog/trading/quant-careers/the-math-foundation-linear-algebra-calculus-stochastic-calc).

**The research layer (on top, for QR).** Statistics and machine learning applied to finding a trading signal — what researchers call *alpha*, the part of a return not explained by the broad market. This is feature engineering, model fitting, backtesting without fooling yourself, and the discipline of killing your own ideas. It is built *on* the probability and linear-algebra pillars, not parallel to them. The sibling is [statistics and ML for alpha research: the researcher's toolkit](/blog/trading/quant-careers/statistics-and-ml-for-alpha-research-the-researchers-toolkit), and the project that proves it is covered in [building signal projects, competitions, and papers](/blog/trading/quant-careers/building-signal-projects-competitions-and-papers-that-prove-it).

Five pillars. A dozen-ish sub-topics. The tree in Figure 1 lays the whole map out at once. The mistake the paralyzed student made was treating these as a flat checklist to grind through top to bottom. They are not flat, and they are not independent — which is the first thing the sequencing rule has to respect.

### What "leverage" means here, concretely

Throughout this post I will rank topics by *leverage*, and I owe you a precise definition before I lean on it. Leverage is the expected interview-pass probability (or job-readiness) you gain per hour of study. It is a ratio: lift divided by hours. A topic has high leverage when a small number of hours moves your odds a lot — like the first 40 hours of mental-math drilling for a market-making screen, which can take you from "auto-reject" to "pass." A topic has low leverage when even many hours move your odds little — like the 100th hour of measure theory for a trading seat that never tests it.

Leverage is not the same as importance-in-the-abstract or even difficulty. Stochastic calculus is hard and intellectually deep, and for the right seat it is genuinely important. But for a candidate targeting a trading role with six months of runway, its *leverage* is near zero, because no round in that interview will reward it and the hours could have gone to a gating topic. The entire curriculum problem is: estimate leverage for each topic given your target role, then spend hours in descending order of leverage until you run out of hours. Everything below is an application of that one rule.

Three properties of leverage are worth stating because they drive every decision below. First, leverage is **personal** — it depends on your current gap, so the same topic has high leverage for someone who scores a 2 and near-zero leverage for someone who already scores a 5. Second, leverage is **role-conditional** — the same topic's ceiling changes by role, so mental math is a top-leverage gate for a trader and a maintenance item for a researcher. Third, leverage is **time-varying** — as you close a gap, the topic's leverage falls and something else rises to the top of your ranking, which is exactly why the plan is a loop and not a one-time list. Hold those three in mind and the rest of the post is just bookkeeping.

### What this post is, and what it is not

A quick framing so you use this correctly. This is the *index* of the knowledge portfolio — it tells you which pillar to study, in what order, and why, and it routes you to the post that teaches each one. It is deliberately *not* the place where you learn probability or write your first C++ template; those live in the sibling deep-dives and the math and interview series. Trying to learn the actual material from a hub post is its own failure mode — you would get a shallow tour of everything and mastery of nothing, which is the opposite of what the leverage rule prescribes. Read this once to build your sequence, then leave it and go deep on one pillar at a time. Return only when you need to re-rank.

## The dependency map: what unlocks what

The pillars sit on top of each other. You cannot meaningfully learn statistics before you can compute a basic probability, and you cannot do alpha research before you can do statistics. Skipping a prerequisite does not save time — it defers a cost, and you pay it later, under interview pressure, when relearning is most expensive.

![A layered dependency graph showing arithmetic at the base feeding into probability and linear algebra, which unlock estimation, statistics, stochastic calculus, and the alpha-research layer.](/imgs/blogs/the-quant-curriculum-map-what-to-learn-in-what-order-2.png)

Figure 2 makes the prerequisite structure explicit. Read it bottom-up. **Arithmetic speed** is the base layer — it is the substrate under both mental-math rounds and the back-of-envelope work you do when sizing a quote in a trading game, and it makes probability calculations fast enough to do live. From there, **probability** is the great unlocker: it grounds estimation (a Fermi estimate is just an expected-value computation with rough inputs), it unlocks statistics (estimators and tests are probability applied to data), and it leads into stochastic calculus (which is, at its heart, probability in continuous time). **Linear algebra** sits alongside probability as a second foundation, supported by arithmetic, and it feeds the **ML / alpha-research layer** together with statistics.

The practical consequences of this graph are sharp:

- **Arithmetic and probability go first, for everyone.** They are upstream of nearly every other node. An hour spent here pays off in multiple downstream topics, which is the definition of high leverage. A candidate who front-loads these is investing in the root of the tree; a candidate who starts at a leaf (stochastic calculus, say) is watering a branch whose roots are not yet in the soil.
- **Statistics and ML are downstream — do not start there.** I regularly meet researchers-to-be who jump straight into gradient boosting and neural nets for alpha and cannot cleanly state Bayes' rule or compute the variance of a sum. Their models overfit because they never internalized the probability that would have warned them. Build the base first.
- **Stochastic calculus is a downstream, role-gated branch.** It depends on probability, and for most roles it is optional. Its position in the graph — late, and off to the side — is the visual argument against learning it first.

This is why a flat reading list is actively harmful: it hides the dependency structure. The map's job is to surface it, so you spend your earliest, freshest hours on the nodes that unlock the most.

There is a subtler point in the graph that catches even careful planners: a prerequisite edge does not mean you must *master* the upstream node before touching the downstream one — it means you need *enough* of the upstream node to make the downstream work pay off. You do not need a measure-theoretic command of probability before doing a single statistics problem; you need fluency with expected value, conditional probability, and variance, which is a far smaller and faster target. The danger of reading the dependency graph too literally is that you spend a year "finishing" probability before you allow yourself to start statistics. The right reading is: learn the upstream node to *working competence*, start the downstream node, and let the downstream work tell you which upstream gaps to patch. Depth on the prerequisite follows demand from the dependent topic — it does not precede it. This is the dependency-graph version of the theory-before-practice myth we will hit later, and it is why even a correct prerequisite order can be applied wrongly.

One more consequence is organizational rather than mathematical: because arithmetic and probability are shared upstream of every role, they are the topics where the sibling deep-dives and the math-for-quants series overlap the most, and where you most want to route out rather than re-derive. The probability you need for a trading game, the probability under your statistics, and the probability under stochastic calculus are *the same probability* — learned once, at the base, it pays into all three branches. That shared base is the strongest argument for the front-loading rule: nowhere else does a single block of hours feed so many downstream nodes.

#### Worked example: sequencing by leverage — mental math first vs measure theory first

Meet **Maya**, a math undergraduate aiming at a trading (QT) seat. She has 6 months and can commit about 12 hours a week — call it roughly 300 hours total. She is deciding what to do with her *first* 40 hours: drill mental math, or study measure theory (the rigorous foundation under probability, and the kind of impressive-sounding topic the paralyzed student gravitated to).

Let me put rough, *illustrative* numbers on it — these are not measured, but they capture the shape of the real tradeoff, and they match the curves in Figure 6 below. Suppose the firms Maya targets gate the first-round screen on a timed mental-math test. Cold, she would fail that screen with probability around 80 percent — it is the single biggest filter in her funnel. The first 40 hours of structured mental-math drilling are the steep part of a diminishing-returns curve: they plausibly lift her pass-probability on that screen from roughly 20 percent to roughly 65 percent. Call that a **+45 percentage-point** swing on the gating round.

Now measure theory. It is beautiful and it is upstream of probability *in a formal sense* — but no round in Maya's target interview tests it. The Lebesgue integral will not come up in a market-making game. So 40 hours of measure theory lifts her interview-pass probability by, generously, **+1 to +2 percentage points** — the trace amount that comes from marginally sharper probabilistic intuition. Maybe less.

The expected-value comparison is not close. Same 40 hours; one bet returns roughly +45 points on the round that matters most, the other returns +1 to +2. The leverage ratio differs by more than an order of magnitude. Maya drills mental math first — not because measure theory is unimportant in some cosmic sense, but because at *her* point in *her* funnel, the marginal hour buys 20× to 40× more pass-probability in arithmetic.

*The first hours of prep are the highest-leverage hours you will ever spend; squandering them on an impressive-but-untested topic is the most expensive mistake in the whole plan.*

## The role-weighted sequence: QT, QR, and QD paths

The dependency graph tells you what is upstream of what. It does not tell you *how much each topic matters for your specific role* — and that weighting changes the sequence substantially. A trader and a developer share the same base layer but diverge sharply above it. So the second input to your curriculum, after the dependency order, is the role weight.

![A matrix with topics as rows and the three roles — QT trader, QR researcher, QD developer — as columns, color-coded by how critical each topic is for each role.](/imgs/blogs/the-quant-curriculum-map-what-to-learn-in-what-order-3.png)

Figure 3 is the weighting matrix. Green cells are critical-and-gating (a topic the role's interview will hard-filter on); amber is useful-but-not-gating; gray is minor. Read down each column to get the role's priority stack.

### The QT (trader) path

A quant trader makes markets or takes positions; the job rewards fast, calibrated decisions under uncertainty. The interview reflects that.

1. **Mental math (critical, gating).** The timed arithmetic screen is the first filter at most market makers — 60 to 80 questions in 8 minutes. Fail it and nothing else you know matters. This is the highest-leverage topic for a QT and it goes first.
2. **Probability and EV (critical).** Trading games and brainteaser rounds are pure expected-value reasoning — quoting a fair market, computing pot-odds-style break-evens, updating on new information. This is the core skill the games actually test.
3. **Markets literacy (critical for desk fluency).** You need to walk into a trading game already knowing what a bid-ask spread is and what fair value means, or you burn precious round time learning the rules instead of playing well.
4. **Python (useful).** Helpful for any take-home or modeling, but rarely the gating filter for a pure trading seat.
5. **Statistics / ML, C++, stochastic calc (minor to useful).** A trader uses statistical intuition, but deep ML and low-latency C++ are other roles' core, and stochastic calc is largely theater here.

The QT sequence is therefore: **arithmetic → probability/EV → markets → light Python**, with everything else deprioritized. Notice how cleanly this falls out of combining the dependency graph (arithmetic and probability first) with the role weights (mental math and games are the QT gates).

### The QR (researcher) path

A quant researcher hunts for signal in data. The job rewards statistical rigor and the discipline not to fool yourself. The interview adds a research case — a signal or backtest take-home — on top of the math.

1. **Probability and EV (critical, foundational).** Still first, because everything downstream depends on it, and QR rounds probe it hard.
2. **Statistics / ML (critical — this is the job).** Estimators, hypothesis testing, regression, regularization, cross-validation, and the overfitting traps. This is the QR's core, and the deep technique lives in [statistics and ML for alpha research](/blog/trading/quant-careers/statistics-and-ml-for-alpha-research-the-researchers-toolkit).
3. **Python (critical, daily tool).** The researcher's hands. Pandas, NumPy, a backtesting harness, clean experiment code.
4. **Linear algebra (critical — feeds ML).** PCA, covariance matrices, regression in matrix form. It is a prerequisite for understanding the models, not optional polish.
5. **Mental math, markets, C++ (useful, not gating).** A QR benefits from arithmetic fluency and markets context, but is rarely cut on a timed sprint the way a trader is.
6. **Stochastic calc (role-gated).** Genuinely useful for derivatives-research seats at a Two Sigma or D.E. Shaw; ignorable for many statistical-arbitrage seats. Learn it only if your target sub-field uses it.

The QR sequence: **probability → statistics/ML + linear algebra (in parallel) → Python depth → research-case practice**, with mental math as maintenance rather than a sprint focus.

### The QD (developer) path

A quant developer builds the systems — the trading infrastructure, the backtesting platform, the low-latency execution path. The job rewards engineering excellence; at HFT firms it rewards C++ and systems depth above all.

1. **C++ and DSA (critical, the latency bar).** For Jump, HRT, and Citadel Securities, this is *the* gate: data structures and algorithms, then C++ depth (memory model, undefined behavior, templates, lock-free, cache behavior), then systems design. The deep technique is in [the coding interview](/blog/trading/quantitative-finance/quant-interview-process-strategy-how-to-prepare)-adjacent posts on C++ and DSA in the quant-interview series.
2. **Probability and EV (useful).** A developer reasons about systems probabilistically and faces some math rounds, but it is rarely the deciding filter.
3. **Python (useful).** Tooling, scripting, glue, and research-platform work.
4. **Markets literacy (useful).** You build the systems that touch markets, so you need to understand order types and the data you are moving.
5. **Mental math, statistics, stochastic calc (minor).** Present but not central; do not spend your scarce hours here.

The QD sequence: **DSA → C++ depth → systems design → supporting probability and markets**. It is the most distinct of the three, because its critical pillar (programming) is a different pillar entirely from the QT's (mental math) and the QR's (statistics).

The deep meaning of the matrix is this: *the same map, weighted three different ways, produces three different sequences.* There is no universal "learn this first" — there is only "learn this first *for your role*." Pick the column before you build the plan.

#### Worked example: a 6-month QT study plan with weekly hours

Let me turn Maya's QT priorities into an actual schedule. She has 6 months and 12 hours a week — about 312 hours total. Here is how I would allocate them, in descending leverage order, respecting the diminishing-returns curve (front-load the gates, do not over-invest past the flat part).

| Phase | Weeks | Hours/week | Focus | Why this allocation |
|---|---|---|---|---|
| 1. Gates | 1-8 | 12 | ~7h mental-math drills, ~5h probability/EV | The screen gate plus the core game skill — the two highest-leverage topics |
| 2. Depth | 9-16 | 12 | ~4h mental-math maintenance, ~4h markets literacy, ~4h trading-game practice | Lock in the gate, add desk fluency, start playing games |
| 3. Polish | 17-24 | 12 | ~6h mock interviews + games, ~3h probability review, ~3h light Python | Convert knowledge to performance under pressure |

Total: roughly 96 hours on mental math (the gate), 80 on probability/EV, 32 on markets, 50 on game practice and mocks, the rest on review and light Python. Notice what is *absent*: no measure theory, no stochastic calculus, no deep C++, no machine learning. Those are not on a QT's critical path, and at 312 total hours Maya cannot afford to dilute the gates. The plan is aggressive precisely because it is narrow.

Now the EV framing. Cold, Maya's odds of clearing a single firm's full loop might be around 3 to 5 percent (these processes are extremely selective). The mental-math phase alone plausibly multiplies her screen-pass odds by 3×; the games and probability work lift her conditional pass-rate on later rounds; the mocks reduce the variance of choking on the day. Stack those multipliers and her per-firm odds might climb toward 10 to 15 percent. Apply that across the [recruiting funnel](/blog/trading/quantitative-finance/quant-interview-process-strategy-how-to-prepare) — apply to enough firms — and a single offer becomes likely rather than a long shot. That is the whole game: order the hours to maximize the per-round pass-probability, then run the funnel.

*A 6-month plan is not 26 weeks of equal effort across every topic — it is a front-loaded campaign that spends its best hours on the two or three gates that decide the role.*

#### Worked example: the funnel math — why the curriculum is an EV multiplier, not a guarantee

It is worth making the connection between the curriculum and the offer explicit, because students routinely either overrate or underrate what a study plan can do. A plan does not *guarantee* an offer — these processes are too selective for that. What it does is lift your per-round pass-probability, and those lifts *compound* across the funnel.

Walk it through for Maya. Suppose she applies to 10 firms, and at each firm the loop is roughly: a mental-math screen, then a probability/games round, then a final superday. Cold — no targeted prep — call her per-round pass-probabilities 0.25, 0.30, and 0.40. Her probability of clearing one full loop is 0.25 × 0.30 × 0.40 ≈ **3 percent**. Across 10 firms, her expected number of offers is 10 × 0.03 = **0.3** — she more likely than not ends the season with nothing.

Now run the same arithmetic after her front-loaded plan. The mental-math phase lifts the screen pass-rate from 0.25 to, say, 0.70; the probability and games work lifts the second round from 0.30 to 0.55; the mocks lift the superday from 0.40 to 0.55 by cutting choke-variance. Per-loop probability becomes 0.70 × 0.55 × 0.55 ≈ **21 percent**. Across 10 firms, expected offers = 10 × 0.21 = **2.1**. The plan did not make any single firm a sure thing — 21 percent is still a coin-flip-and-then-some per firm — but it turned an expected 0.3 offers into 2.1, a **7×** swing, by lifting each round and letting the funnel multiply the lifts.

Notice the leverage logic inside the funnel: the *screen* lift (0.25 → 0.70) is the biggest single contributor, because it is the first multiplicative term and it was the lowest cold. That is the funnel-level reason the curriculum front-loads the gate — fixing the weakest, earliest multiplier moves the product the most. A later round that is already at 0.55 has less room to move the product than a screen stuck at 0.25.

*A study plan is an expected-value multiplier on a probabilistic funnel, not a guarantee on any one firm — and the highest-leverage hours are the ones that fix your lowest, earliest multiplier.*

## The phased timeline: months 0-3, 3-6, 6-12

The role weights give you *what* to prioritize. The calendar gives you *when*. Whatever your role, the prep splits naturally into three phases keyed to the diminishing-returns curve and the recruiting calendar.

![A four-phase timeline showing months 0-3 as the gates, 3-6 as depth, 6-12 as polish, and month 12 onward as apply-and-iterate.](/imgs/blogs/the-quant-curriculum-map-what-to-learn-in-what-order-4.png)

Figure 4 lays out the generic 12-month arc; compress it to 6 if that is your runway by overlapping the phases.

**Months 0-3 — the gates.** Spend the freshest, highest-leverage hours on the topics that hard-filter your role: mental math and probability for a QT; probability, statistics, and Python for a QR; DSA and C++ for a QD. The goal of this phase is to get *past the first filter*, because no later strength rescues you from an auto-reject. This is the steep part of every learning curve, where hours buy the most.

**Months 3-6 — the depth.** Now build beyond the gate into genuine competence: markets literacy and trading-game fluency for the trader; statistics depth, linear algebra, and a real backtesting workflow for the researcher; systems design and C++ depth for the developer. You are moving along the diminishing-returns curve — still climbing, but each hour buys less than in phase 1, which is correct, because you have already banked the gate.

**Months 6-12 — the polish.** Convert knowledge into performance. Mock interviews, timed game practice, the research-case take-home, and one portfolio project that proves your skill (covered in [building signal projects](/blog/trading/quant-careers/building-signal-projects-competitions-and-papers-that-prove-it)). This phase is about reducing variance — not learning new material so much as making sure you do not choke on material you know. It also aligns with when applications open, so your prep peaks exactly as the funnel does.

**Month 12+ — apply and iterate.** Each rejection is data. Bombed the mental-math screen? Your gate phase was under-invested; go back and drill. Cleared the screen but lost the trading game? Your EV reasoning or your calibration needs work. Treat the funnel as a feedback loop that tells you which pillar to reinforce next. The candidates who improve fastest are the ones who diagnose each loss to a specific pillar instead of concluding "I'm just not good enough."

#### Worked example: a 6-month QR study plan

Meet **Wei**, a CS PhD aiming at a research (QR) seat at a systematic fund. Same 6 months, but Wei can commit more — about 15 hours a week, roughly 390 hours — because the PhD schedule is flexible. Wei's priorities differ from Maya's: statistics and ML are the gate, not mental math.

| Phase | Weeks | Hours/week | Focus | Why this allocation |
|---|---|---|---|---|
| 1. Gates | 1-8 | 15 | ~6h probability, ~5h statistics, ~4h Python/pandas | The QR foundation — probability under the stats, Python as the daily tool |
| 2. Depth | 9-16 | 15 | ~5h ML for alpha, ~4h linear algebra, ~3h backtesting discipline, ~3h markets | Build the researcher's toolkit and learn to not fool yourself |
| 3. Polish | 17-24 | 15 | ~7h research-case practice, ~4h one signal project, ~4h mocks + brainteaser review | Produce the take-home-quality artifact the interview actually grades |

Total: roughly 48 hours probability, 40 statistics, 60 ML/alpha, 32 linear algebra, 56 Python and backtesting, 70 on the research case and project, the rest on review. The biggest single bucket is the research case and project — because for a QR, the take-home *is* the interview, and the artifact that demonstrates clean out-of-sample discipline is worth more than any extra hour of theory. Note what Wei *de-prioritizes*: mental-math sprints (useful, not gating for QR) get only maintenance attention, and stochastic calculus appears only if the target fund's sub-field uses it.

The EV math here runs through a different bottleneck than Maya's. Wei is unlikely to be cut on a timed arithmetic screen, but very likely to be cut on a research case that shows an overfit signal — a backtest with a Sharpe of 3.0 in-sample that collapses to 0.4 out-of-sample. The single highest-leverage thing Wei can learn is the discipline of *killing your own idea*: purged cross-validation, deflated Sharpe ratios, honest out-of-sample testing. That is why the plan front-loads probability and statistics (so the overfitting traps are visible) and back-loads a real project (so there is a clean artifact to discuss). The [research-case sibling post](/blog/trading/quant-careers/the-research-case-and-take-home-how-to-ace-it) and the quant-research deep-dives are where that technique is taught in detail.

*A researcher's plan optimizes for one thing above all — producing an artifact that survives scrutiny — because the QR interview rewards intellectual honesty far more than it rewards model complexity.*

### How the sibling deep-dives plug in

This post is deliberately a hub, not a textbook. Each pillar has a dedicated sibling post that goes three levels deeper, and each of those routes further out to the [math-for-quants](/blog/trading/math-for-quants/probability-spaces-random-variables-math-for-quants) series (for the rigorous math) and the quant-interview series (for the problem technique). The division of labor is intentional: this series owns the *career and sequencing* layer; the math series owns the *derivations*; the quant-interview series owns the *drills and problems*. You should never re-derive a result that the math series already proves, and you should never re-grind a problem type the interview series already covers — you route to them and keep your own hours for what is not written elsewhere.

![A pipeline showing the build-your-own-plan loop: pick a role, audit gaps, rank by leverage, schedule fixed hours, measure with mocks, then re-audit and loop back.](/imgs/blogs/the-quant-curriculum-map-what-to-learn-in-what-order-5.png)

## How to build your own learning plan

You now have the map (Figure 1), the dependency order (Figure 2), and the role weights (Figure 3). Here is how to turn them into a personal plan you will actually follow. Figure 5 shows the loop; let me walk each step.

**Step 1 — Pick a role.** QT, QR, or QD. This is the single highest-leverage decision, because it sets which column of the matrix you optimize. If you are genuinely undecided, the four-paths framing in the role-archetype posts can help; default to the role whose gating skill you are already closest to, since that minimizes the hours to your first offer. Do not try to optimize for all three — a plan that hedges across roles is a plan that masters none of their gates.

**Step 2 — Audit your gaps honestly.** Score yourself 1 to 5 on each of the five pillars (plus the research layer if QR). Be brutal: "I took a probability class" is not a 5; a 5 means you can solve a hard conditional-probability brainteaser cold, under time pressure. The most common self-audit error is overrating the pillar you enjoy and underrating the one you avoid — and you avoid it precisely because it is your weakness. If you cannot tell, take a timed diagnostic: a mental-math sprint, a set of probability brainteasers, a HackerRank medium. Your score is where the timer stops you.

**Step 3 — Rank by leverage.** For each pillar, leverage = (how gating it is for your role) × (how far you are from the bar) ÷ (hours to close the gap). A pillar that is critical for your role *and* where you score 2 is your top priority. A pillar where you already score 5, or that is minor for your role, drops to the bottom. This is the step that produces the *order* — and it is where most self-study plans go wrong, because they rank by interest or by what is trendy instead of by leverage.

**Step 4 — Schedule fixed weekly hours.** Put the hours in your calendar as recurring blocks, and protect them. Spaced, consistent practice beats cramming for the same reason it does in any skill: the timed-arithmetic and brainteaser skills are *motor* skills as much as knowledge, and motor skills consolidate with sleep and repetition. Two hours a day, six days a week, for six months is far more effective than the same total hours crammed into the last two months.

**Step 5 — Measure.** Every few weeks, re-take a timed diagnostic in your top-priority pillars. Mocks, timed games, HackerRank under a clock. The measurement matters because your *felt* progress is a terrible estimator of your *actual* progress — you feel ready long before you are, and the diagnostic is the reality check.

**Step 6 — Re-audit and loop.** Feed the measurement back into Step 2. As you close a gap, that pillar's leverage drops and another rises to the top of the ranking; your plan should shift accordingly. The plan is not a static document you write once — it is a control loop you re-run every few weeks, steering your hours toward whatever your current weakest gating pillar is.

The reason this works is that it directly implements the leverage rule under a real constraint (your finite hours). It refuses to let you spend the 80th hour on a pillar you have already mastered while a gating pillar sits at a 2. That refusal is the entire value of having a plan instead of a reading list.

#### Worked example: Maya and Wei each build a phased plan

Let me run both characters through the loop end-to-end so the steps are concrete.

**Maya (QT).** *Step 1:* role = QT. *Step 2:* she audits — mental math 2 (she is slow and uncalibrated), probability 4 (strong math major), markets 1 (never studied them), Python 3, C++ 1, stats 3, stochastic calc 2. *Step 3:* leverage ranking. Mental math is critical-and-gating *and* she scores 2 — top priority. Markets is critical-for-desk-fluency *and* she scores 1 — second. Probability is critical but she is already a 4, so it needs only sharpening, not building — third. C++ and stochastic calc are minor for QT, so despite low scores they drop to the bottom (low leverage: the gap is large but the topic is not gating). *Step 4:* she schedules 12 hours a week as detailed in the QT plan above. *Step 5:* week-4 diagnostic — her mental-math sprint score went from 35 percent to 58 percent; still below the ~75 percent bar, so she holds the allocation. *Step 6:* by week 12 she hits 78 percent on mental math, its leverage drops, and markets-plus-games rises to the top — exactly the phase-2 shift the plan predicted.

**Wei (QR).** *Step 1:* role = QR. *Step 2:* probability 4, statistics 4, ML 3, Python 5 (it is a CS PhD), linear algebra 4, mental math 2, markets 1, research-case discipline 2 (knows the models, has never run a leakage-free backtest). *Step 3:* leverage ranking. Research-case discipline is critical-for-QR *and* he scores 2 — top priority, even though his raw stats knowledge is strong, because the *application* (not fooling yourself) is the gate. ML for alpha is critical and he is a 3 — second. Markets at 1 is useful-not-gating, so it ranks below those despite the low score. Python at 5 needs no hours at all — it drops off entirely, freeing time for the gates. *Step 4:* he schedules 15 hours a week per the QR plan. *Step 5:* his week-8 diagnostic is a self-graded mock research case — and he catches himself peeking at out-of-sample data while tuning, which is exactly the leak that would have sunk him. *Step 6:* that failure raises the leverage of backtesting discipline even higher, and he reallocates phase-2 hours toward purged cross-validation.

Same loop, two completely different plans — because the inputs (role, gaps, leverage) are different. That is the point: the *process* is universal, the *plan* is personal. Anyone who hands you a one-size-fits-all "quant reading list" has skipped the only step that matters.

*The plan that gets you hired is not the most ambitious one — it is the one whose order matches your role's gates and your own gaps, re-steered every few weeks by honest measurement.*

## Common misconceptions

The curriculum is where the most expensive myths live, because believing them costs you months. Here are the four I see wreck the most plans.

**Myth 1: "I need to learn everything."** No one knows everything in this field, and trying to is how you end up like the paralyzed student with fourteen books and no progress. The senior quants I know are *deep* in a few pillars and merely *literate* in the rest. A trader does not need to derive Itô's lemma; a developer does not need to estimate a deflated Sharpe; a researcher does not need to write lock-free C++. The whole discipline of a curriculum is *choosing what to skip*. Breadth is the enemy of a six-month plan. The correct mindset is not "what could I learn?" but "what is the minimum set that clears my role's gates?" — and then learning that set deeply enough to perform under pressure.

**Myth 2: "Start with stochastic calculus."** This is the single most common sequencing error, and it is seductive because stochastic calculus *sounds* like the most quant-y topic and the books are intimidating in a way that feels important. But look back at Figure 2: stochastic calculus is a late, role-gated branch. For a trader it is interview theater — no market-making game will ever ask you to apply Itô's lemma. For most researchers outside derivatives-pricing seats it is rarely the daily tool. Starting here means spending your freshest, highest-leverage hours on a topic with near-zero leverage for your role, while the gating topics (mental math, probability, DSA) sit untouched. If you take one thing from this post: *do not start with stochastic calculus unless your specific target seat prices derivatives.*

**Myth 3: "Theory before practice."** The instinct is to "really understand" a topic before practicing problems — read the whole probability textbook before doing any brainteasers, master C++ semantics before writing any competitive-programming solutions. This inverts the right order. The interview tests *applied* skill under time pressure, and applied skill is built by *doing*, not by reading. You should learn just enough theory to start practicing, then practice, then return to theory only to patch the specific gaps your practice exposes. A mental-math sprint is a motor skill; you do not improve it by reading about arithmetic. The same is true of trading games, brainteasers, and coding screens. Practice is not the reward after theory — it is the primary training mechanism, and theory is its support.

**Myth 4: "One resource is enough."** People ask "what is the one book / one course that will get me a quant job?" There is no such resource, for a structural reason: the skills are different in kind. A book teaches probability theory; it does not build your timed-arithmetic motor skill, give you reps in a trading game, or grade your research-case for leakage. The curriculum is a *portfolio* of resources — a drill app for mental math, a problem set for probability, a platform for coding, a mock-interview partner for games, a dataset for a project. Anyone selling "the one resource" is selling you a flat list when you need an ordered portfolio. Use the resource map (Figure 7) instead: route each pillar to the deep-dive that teaches *that pillar's* skill in its own right way.

There is a fifth myth worth a sentence, because it underlies the other four: **"more hours always helps."** They do not, past the flat part of the curve. The 100th hour on your strongest pillar is worth a fraction of the 1st hour on your weakest gating pillar. The whole leverage framework exists to stop you from pouring hours into a topic that has stopped paying off while a gate sits open.

And a sixth, which is quieter but does the most long-run damage: **"prep ends when I get the offer."** The curriculum framework does not retire at the offer letter — it just re-weights. The pillars that mattered to get hired (mental-math sprints, brainteasers) fade in importance once you are on the desk, and the ones that were merely useful for the interview (markets microstructure, real risk management, the specifics of your firm's systems) become the gates for *promotion*. The senior path is its own curriculum with its own leverage ranking, and the candidates who treat learning as a phase that ends at the offer plateau early. The leverage loop in Figure 5 is a tool for a career, not just an interview season — you run it again at every level, with the role weights shifted toward whatever the next rung actually rewards.

## How it plays out

Let me ground all of this in the real, observable dynamics of how curricula succeed and fail — because the framework is only worth anything if it matches what actually happens to candidates.

![A data chart of illustrative interview-pass lift versus study hours for four topics, all showing diminishing returns with sharply different ceilings by topic.](/imgs/blogs/the-quant-curriculum-map-what-to-learn-in-what-order-6.png)

Figure 6 is the empirical heart of the whole post, and I have flagged it *illustrative* deliberately — these are not measured numbers, but the *shape* is real and well-supported by how candidates actually progress. Three features matter:

**First, every curve is concave — returns diminish fast.** The first ~40 hours on any topic (the shaded zone) buy most of the total lift; the curve flattens hard after that. This is the single most important fact for sequencing. It means you should *spread* your early hours across your top few gating pillars rather than maxing out one — four pillars at 40 hours each beats one pillar at 160 hours, because you are always on the steep part of each curve. It also means there is a point past which more study of a topic is nearly wasted; recognizing that point and moving on is a skill in itself.

**Second, the ceilings differ wildly by topic — and that gap encodes the role weight.** Mental math (green) tops out around a 45-point lift for a QT because it is the gating screen; stochastic calculus (red, dashed) tops out around 8 points because no QT round rewards it. Same hours, radically different maximum payoff. The ceiling is the role weight made visible: a topic's ceiling is how much pass-probability mastering it can *ever* buy you for your role. You sequence by which curve has the highest remaining slope at your current position — which early on means the high-ceiling gates.

**Third, the cheap lift is gone fast — so the order of your first 40-hour blocks dominates.** Because each curve front-loads its payoff, the question "which topic do I spend my first 40 hours on?" repeated four or five times *is* the curriculum. Get that ordering right and you bank the cheap, high-leverage lift from every gating pillar before diminishing returns set in. Get it wrong — 160 hours of measure theory before a single mental-math drill — and you arrive at applications having spent your best hours on the flat part of a low-ceiling curve.

Now the broader reality, with the honest caveats this series insists on. These programs are *extremely* selective — top quant intern and new-grad seats are widely reported at low-single-digit-percent acceptance or below, with thousands of applicants per seat. A perfect curriculum does not make you a sure thing; it raises your per-round pass-probability and lets you run the funnel enough times that an offer becomes likely. The backgrounds that get hired — math, CS, physics, statistics, EE, operations research, often with a competition signal like Putnam, ICPC, or IMC Prosperity — got there by mastering the gates, not by reading the most books. And a PhD is *not* required for trading or most developer roles; it is common but not mandatory for research-scientist seats at the likes of Two Sigma and D.E. Shaw. Your curriculum should be calibrated to your *role*, not to a prestige bar you have invented in your head.

![A grid mapping each of the five pillars to its sibling career deep-dive in this series and to the deeper math-for-quants or quant-interview post that teaches the underlying technique.](/imgs/blogs/the-quant-curriculum-map-what-to-learn-in-what-order-7.png)

Figure 7 is the resource map and the reason this post is a hub. Each pillar routes left to the sibling career deep-dive in this series (the *what and why*) and right to the deeper math-for-quants or quant-interview post (the *how*). The path through the whole knowledge track is: start here, pick your role, build your plan with the leverage loop, then open the sibling for whichever pillar is currently at the top of your ranking — and from there route out to the technique posts when you need the derivation or the drills. You never relearn from scratch; you always know which door to open next.

#### Worked example: the cost of getting the order wrong

One last number, because the cost of bad sequencing is concrete. Take two candidates, both QT-bound, both with exactly 160 hours before applications open.

**Candidate A (right order):** 40 hours mental math, 40 probability/EV, 40 markets + games, 40 mocks. From Figure 6's curves, each 40-hour block lands on the steep part of a high-ceiling curve, so A banks roughly: +40 points (mental math, near its 45 ceiling), +35 (probability), +13 (markets), plus sharply reduced choke-variance from mocks. A walks into the screen with a pass-probability around 75 to 80 percent and strong game performance behind it.

**Candidate B (wrong order):** 160 hours of stochastic calculus and measure theory, because they "wanted to really understand the math first." From Figure 6, stochastic calc tops out near +8 points for a QT and B never touched the gates. B walks into the timed mental-math screen cold — pass-probability around 20 percent — and is auto-rejected before any human sees the impressive math they learned.

Same 160 hours. Same raw talent. Candidate A's expected number of offers across a normal application list is *multiples* of Candidate B's, entirely because of *order*. Nothing about effort or ability separates them — only the sequence. That gap is the entire thesis of this post, in one comparison.

*The most expensive mistake in quant prep is not studying too little — it is studying the right topics in the wrong order, and discovering it only when the auto-reject email arrives.*

## When this matters / Further reading

This post matters most at the very beginning — when you are staring at that stack of books and need a sequence, not a longer list. Come back to it whenever your plan stalls: re-run the leverage loop, re-rank your pillars against your current diagnostics, and re-point your hours at the top gating gap. It also matters at every transition — switching target roles, recovering from a rejection, or deciding whether a shiny new topic is worth your hours (run it through the leverage test: gating for your role? how far from the bar? worth it?). The framework is the same at the start of prep and three rejections in; only the inputs change.

The one idea to carry out of here: *you cannot learn everything, so sequence by leverage.* Pick your role, audit your gaps, rank by expected pass-probability per hour, front-load the gates, and re-steer every few weeks. Order beats volume, every time.

From here, open the pillar that sits at the top of your ranking:

- **Probability and statistics** — [the probability and statistics you must own](/blog/trading/quant-careers/the-probability-and-statistics-you-must-own), routing to the [math-for-quants probability foundations](/blog/trading/math-for-quants/probability-spaces-random-variables-math-for-quants).
- **Mental math** — [mental math and estimation as a trainable skill](/blog/trading/quant-careers/mental-math-and-estimation-as-a-trainable-skill).
- **Programming** — [programming for quants: Python, C++, and the DSA bar](/blog/trading/quant-careers/programming-for-quants-python-cpp-and-the-dsa-bar).
- **Markets literacy** — [the markets knowledge every quant needs](/blog/trading/quant-careers/the-markets-knowledge-every-quant-needs).
- **The math foundation** — [the math foundation: linear algebra, calculus, stochastic calc](/blog/trading/quant-careers/the-math-foundation-linear-algebra-calculus-stochastic-calc).
- **The research layer and the project** — [statistics and ML for alpha research](/blog/trading/quant-careers/statistics-and-ml-for-alpha-research-the-researchers-toolkit) and [building signal projects, competitions, and papers](/blog/trading/quant-careers/building-signal-projects-competitions-and-papers-that-prove-it).

And when you are ready to turn the curriculum into an application campaign, route out to the [quant interview process and preparation strategy](/blog/trading/quantitative-finance/quant-interview-process-strategy-how-to-prepare), which connects the knowledge plan to the funnel that converts it into an offer.

*Comp, acceptance-rate, and firm figures cited throughout are reported ranges (levels.fyi, Glassdoor, firm pages, and the 2026 quant-pay surveys) as of June 2026, and the learning-curve numbers in Figure 6 and the worked examples are illustrative — the shapes are real, the exact values are stylized to make the leverage argument concrete.*
