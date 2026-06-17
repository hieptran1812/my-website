---
title: "Building Signal: Projects, Competitions, and Papers That Prove It"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Knowledge is invisible until you prove it. This is the portfolio guide: what to build and do so a recruiter and an interviewer can see you can do the job, weighted by role, with the projects, competitions, and papers that actually carry signal and how to present them."
tags:
  [
    "quant-careers",
    "quant-finance",
    "careers",
    "projects",
    "portfolio",
    "competitions",
    "kaggle",
    "backtest",
    "github",
    "signal",
    "quant-trader",
    "quant-researcher",
  ]
category: "trading"
subcategory: "Quant Careers"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A recruiter cannot see what you know, only what you have shipped, so your portfolio is the proof; build the one project the role actually does, present it as a question-method-result-limitation story, and let depth beat a long list.
>
> - One deep, well-presented flagship project beats ten tutorial clones; the role you target decides which flagship to build (a backtest/alpha repo for QR, a market-making or order-book sim for QT/QD, a clean systems project for SWE).
> - Competitions carry real signal but each signals a *different* thing to a *different* audience: Kaggle reads for ML research, IMC Prosperity for trading EV, ICPC/Codeforces for QD/SWE, olympiads as a raw-ability tailwind for QR/QT.
> - Papers help mostly for research-scientist roles; for almost everyone else a public repo with an honest result is higher signal per hour than chasing a publication.
> - The single number to remember: a credible backtest is defined by what is in the *middle* — a real out-of-sample split and honest cost accounting — not by the headline Sharpe.

Two candidates apply to the same quant-research seat in the same week. Same school, same major, same GPA, both took the stochastic-calculus elective, both list Python and C++. On paper they are interchangeable. One of them gets a first-round screen; the other gets a templated rejection. The difference is a single line near the top of one resume: a link to a public repository titled `equity-reversal-backtest`, with a one-sentence README that reads *"Does a 5-day reversal predict S&P 500 returns, 2010–2024? Walk-forward, net of 5 bps costs: Sharpe 0.9 out-of-sample versus 1.6 in-sample — edge halves after costs and decays post-2020."*

The recruiter cannot read minds. She has roughly ten seconds per resume and several hundred resumes for one seat. She has no instrument that measures how much linear algebra lives in a candidate's head, how cleanly they think about overfitting, or whether they would panic the first time a signal decays in production. She has only the artifacts in front of her. One candidate handed her an artifact that *looks like the job* — a falsifiable question, an out-of-sample test, costs applied, and the intellectual honesty to state where the edge breaks. The other handed her a list of adjectives. She is not being unfair when she screens out the adjectives. She is doing the only thing the information lets her do.

This is the uncomfortable truth that the rest of this post is built on: in quant hiring, knowledge is invisible until you prove it, and a project is the cheapest, highest-bandwidth proof you can manufacture. The previous posts in this track mapped *what to learn* — [the curriculum, in order](/blog/trading/quant-careers/the-quant-curriculum-map-what-to-learn-in-what-order). This one closes the track by answering the other half of the question: what to *build and do* so that learning becomes legible to someone who will never see inside your head. Figure 1 is the map we will spend the post filling in — which project type sends a strong signal for which role.

![Matrix showing four project types as rows and four roles QT, QR, QD, SWE as columns, with each cell rated strong, some, or weak signal for that role](/imgs/blogs/building-signal-projects-competitions-and-papers-that-prove-it-1.png)

## Foundations: why signal beats claims

Before we talk about *what* to build, we need to be precise about the thing we are trying to produce. The word that runs through this whole series is **edge** — a small, repeatable advantage that pays off over many tries. Getting hired is an edge problem too: you are trying to nudge the probability that a recruiter, a screener, and then an interviewer all say yes. A project is one of the few levers you fully control that moves that probability. So let us define the terms from zero, because the rest of the post leans on them.

**Signal versus claim.** A *claim* is something you assert about yourself: "I know machine learning," "I am proficient in C++," "I am a strong problem solver." A *signal* is evidence a third party can verify without trusting you: a repository they can clone and run, a competition leaderboard with your name on it, a result a reviewer can reproduce. The entire screening machine of a quant firm exists to convert noisy claims into trusted signals, because claims are free to make and therefore carry almost no information. Every applicant claims to be a strong problem solver. Far fewer can point at a thing that proves it. The asymmetry is the whole game.

**Why this matters more in quant than almost anywhere.** Most industries hire on a blend of pedigree, interview performance, and references. Quant firms do too, but they are unusually *evidence-hungry* because the job itself is about distinguishing real edge from noise. A firm that pays a new graduate a [first-year package in the 450k–650k USD range](/blog/trading/quant-careers/do-you-need-a-phd-the-backgrounds-that-get-hired) (base 250k–375k plus a sign-on of 50k–200k, on-target, as reported across levels.fyi and firm pages in 2025–2026) is making a large bet on an unproven person. They de-risk that bet by demanding proof. A candidate who shows up with proof is, quite literally, lowering the firm's risk on the hire — and they will pay for that.

**A project is a proxy for on-the-job ability.** This is the cleanest way to think about why a good project works. The firm cannot watch you do the job before they hire you. So they look for the closest available substitute: did you do something *shaped like* the job, on your own time, and do it well? A quant researcher's job is to ask a market question, build a signal, test it honestly out-of-sample, account for costs, and either kill it or write it up. A backtest repository is a miniature of exactly that loop. A quant trader's job is to quote two-sided prices and manage inventory under uncertainty; a market-making simulator is a miniature of that. A quant developer's job is to build correct, fast systems; a low-latency order-book component is a miniature of that. The project's power comes from its *isomorphism* to the job. A project that looks like the job is a high-signal proxy. A project that looks like a tutorial is not, no matter how much code it contains.

**The role weights everything.** Notice that the same project does not signal equally for every role. Figure 1 makes this concrete: a backtest repo is *strong* signal for a researcher and only *some* signal for a trader, because the trader's day is about live quoting and risk, not offline alpha mining. An order-book latency component is strong for a developer and near-zero for a researcher. This is the single most common mistake we will return to: people build the project they find interesting, or the one their friend built, rather than the project that mirrors the role they are applying to. Before you write a line of code, you should be able to name the role and therefore the project. The four roles — trader (QT), researcher (QR), developer (QD), and engineer/SWE — are spelled out in [the four-paths post](/blog/trading/quant-careers/the-four-paths-trader-researcher-developer-engineer); your portfolio is downstream of that choice.

**Two people we will follow.** To keep this concrete, meet our recurring pair. **Maya** is a math undergraduate aiming at a trading seat. She is fast at mental math, likes games, and wants to be on a desk quoting prices. **Wei** is a CS PhD student aiming at a research-scientist role. He is comfortable with statistics, ML, and writing things up carefully. Throughout the post, Maya will build the project a trader should build and Wei will build the project a researcher should build, and we will watch each of them make the same key decisions: scope tight, ship, present honestly, iterate. They are not the same person doing different things by accident — their portfolios *diverge by role on purpose*, which is the whole lesson.

With the vocabulary in place, the rest of the post is the practical playbook: the right project per role, the competitions that carry real signal, when a paper is worth it, how to present the work so a busy reader gets it in ten seconds, the anti-patterns that quietly sink most portfolios, and a sequenced plan to build all of it without burning out.

## The right project, by role

The first and most important decision is not *how good* your project is but *whether it is the right project*. A flawless implementation of the wrong thing signals little. Let us go role by role.

### Quant research: a backtest / alpha-research repository

For a research seat, the canonical flagship is a backtest of a trading idea — a small piece of [alpha research](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) you took from question to honest conclusion. This is the highest-isomorphism project in the entire field, because researching, testing, and *killing* signals is the literal job. The repository should let a reviewer trace the full arc: a falsifiable hypothesis, clean dated data, a signal constructed from that data, an out-of-sample test that you did not peek at while building, costs subtracted, and a write-up that states the result and where it breaks.

What makes it credible is not the headline number. A repository that reports Sharpe 3.0 is *less* believable than one that reports Sharpe 0.7, because the experienced reader knows that anyone reporting a 3 either overfit, leaked future data, or forgot costs. The credible signal is the *discipline*: that you split the data properly, that you applied realistic costs, and that you were willing to write "this edge halves after costs and decays after 2020." A researcher's most valuable trait is the ability to distrust their own result, and the write-up is where you demonstrate it. If you want the deep mechanics of doing this right — purged cross-validation, deflated Sharpe, the whole anti-overfitting toolkit — that lives in the [overfitting and purged-CV post](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research) and we link out rather than re-derive it here. Your job in the project is to *show you applied it*.

There is a deeper reason the modest, honest backtest out-signals the spectacular one, and it is worth dwelling on because it reframes how you should build. The experienced reviewer is not evaluating your *idea* — most retail-discoverable signals are weak or dead, and they know it. They are evaluating your *process for handling a result you want to be true*. The single most dangerous moment in any research project is the instant the backtest prints a beautiful number, because that is the instant your incentive flips from finding truth to protecting the number. Every shortcut — peeking at the test set "just to check," dropping the two bad years, lowering the cost estimate until the strategy survives — is a small lie you tell yourself first. A reviewer who has lived through this knows it intimately and reads your repository for the fingerprints of someone who resisted those temptations: a test set touched exactly once, a cost figure that errs high, a limitations section that names the years where it does not work. The Sharpe is almost beside the point. What you are really shipping is a demonstration that you can be trusted alone with data and a tempting result, which is precisely the thing a research firm cannot test in a 45-minute interview and desperately wants to know.

A practical corollary: pick a question whose *honest answer is allowed to be "no, mostly not."* A backtest that concludes "this well-known anomaly has largely decayed and what survives is eaten by costs" is a stronger artifact than one that claims to have found free money, because the former is almost certainly true and the latter almost certainly is not. Choosing a falsifiable question and then reporting that it mostly falsified is the most senior move an early-career researcher can make on paper, and it costs you nothing — the firm is not hiring you because you found alpha in your bedroom; they are hiring you because you will not fool yourself with their capital.

#### Worked example: what a credible backtest project must contain

Wei wants one flagship for his research applications. He resists the urge to build a deep-learning price predictor (too easy to overfit, too hard to believe) and instead picks a small, legible question: *does short-horizon mean reversion in large-cap US equities survive transaction costs?* Here is the spec he holds himself to, item by item, with the number that makes each item credible.

- **Data.** Daily closes for the S&P 500 constituents, 2010–2024, with a survivorship-bias-free universe (he includes companies that were later delisted, because excluding them inflates returns). Source and date range stated in the README. *Cost of skipping this: a survivorship-biased universe can add 1–2% annualized of pure fiction.*
- **Signal.** A simple, explainable rule: rank stocks by their trailing 5-day return, go long the bottom decile and short the top decile, rebalance weekly. No 40-parameter model. The simplicity *is* the credibility — a reviewer can hold the whole idea in their head.
- **Out-of-sample split.** He builds and tunes only on 2010–2018, then runs 2019–2024 *once*, untouched, as the true test. In-sample Sharpe comes out 1.6; out-of-sample comes out 0.9. The gap between those two numbers is the single most informative thing in the project, because it quantifies how much of the in-sample result was real.
- **Costs.** He applies 5 bps per side of round-trip cost (a defensible estimate for liquid large-caps). With weekly rebalancing of a long-short decile book, costs eat roughly half the gross edge, dragging a gross Sharpe near 1.8 down to the net 0.9 he reports.
- **A written result.** Three sentences at the top of the README: the question, the net-of-cost out-of-sample Sharpe of 0.9, and the honest limitation that the edge concentrates pre-2020 and is marginal after costs.

Wei spends about three weekends on this. The code is maybe 300 lines. It will out-signal a 3,000-line "deep RL trading agent" with no out-of-sample discipline every single time, because every line of it reads like the job.

*A credible backtest is not the one with the biggest number; it is the one whose author clearly tried to prove themselves wrong and reported what survived.*

### Quant trading and quant dev: a market-making or order-book simulator

For a trading seat, the job is quoting two-sided markets and managing inventory under uncertainty — so the flagship that mirrors it is a [market-making simulator](/blog/trading/quantitative-finance/market-making-simulator-quant-research). You build a tiny market: a stream of orders arrives, you post a bid and an ask, you get filled, you accumulate inventory, and you have to manage the risk that inventory creates when the price moves against you. The signal it sends is *exactly* the trading instinct an interviewer probes in the [market-making game rounds](/blog/trading/quant-careers/the-research-case-and-take-home-how-to-ace-it): can you reason about expected value per trade, widen your spread when you are uncertain, and skew your quotes to offload inventory before it hurts?

For a quant-dev or low-latency seat, the same simulator can be re-pointed toward systems: build an [order-book simulator](/blog/trading/quantitative-finance/order-book-simulator-quant-research) with a real matching engine — price-time priority, order add/cancel/modify, a clean data structure for the book — and then build *one* low-latency component well. Not a whole HFT stack; one component that demonstrates you understand where the microseconds go. A lock-free single-producer single-consumer queue, a cache-friendly order-book representation, a benchmark that shows you measured before and after an optimization. The firms that hire hardest on this — Jump, HRT, Citadel Securities — care less about breadth and more about whether you can reason about the memory model and prove a speedup with a number.

#### Worked example: Maya scopes a market-making sim; Wei scopes an alpha repo

Maya and Wei sit down on the same Saturday to plan one flagship each. Watch how the *same planning discipline* produces two completely different projects because their target roles differ.

**Maya (QT, market-making sim).** Her question: *given a simple order-flow model, what spread and inventory skew maximize expected P&L while keeping inventory bounded?* Her tight scope: a single simulated asset, Poisson order arrivals, a mid-price that random-walks, and an agent that posts a symmetric spread she can widen and a skew that pushes quotes to flatten inventory. Her measurable result: she runs the agent across 10,000 simulated sessions and reports that a spread of 4 ticks with an inventory-proportional skew yields an average P&L of +120 ticks per session at a 95th-percentile inventory of ±40 lots, versus a naive zero-skew agent that earns +60 ticks but blows out to ±150 lots. The skew *halved her tail inventory while doubling P&L* — that is a sentence a trading interviewer will want to dig into, which is exactly what you want a project to provoke.

**Wei (QR, alpha repo).** His question and spec are the reversal backtest from the previous worked example. His measurable result is the 0.9 out-of-sample net Sharpe and the honest decay note.

Both projects took roughly three weekends. Both are about 300 lines. Both produce *one number and one honest caveat* that an interviewer can interrogate. Neither tried to be a product. The difference is entirely in *what the project does*, and that difference maps one-to-one to the role each is targeting.

*The planning discipline is identical across roles; the project is not — Maya builds the thing a trader does, Wei builds the thing a researcher does, and that fit is most of the signal.*

### Software engineering: a clean DSA / systems project

If you are targeting a pure software-engineering role at a quant firm (these exist in large numbers — exchange connectivity, data infrastructure, risk systems, research platforms), the flagship is a clean systems or data-structures-and-algorithms project that shows engineering judgment. A small but real distributed system: a key-value store with a write-ahead log and a recovery path, a job scheduler with backpressure, a streaming aggregator that handles out-of-order events. The signal is correctness, tested edge cases, a sensible architecture, and a README that explains the *tradeoffs* you chose — not a feature list. SWE interviewers at these firms read code the way they read production code: they look for whether you thought about failure, concurrency, and the path that breaks.

The number that makes a systems project credible is different from a researcher's Sharpe, but it plays the same role. For a key-value store it might be "recovers a 10-million-key dataset from the write-ahead log in 1.2 seconds after a simulated crash, with zero lost acknowledged writes across 1,000 fault-injection runs." For a streaming aggregator it might be "sustains 200,000 events per second with correct results under 30 seconds of clock skew." Notice the shape: a *correctness property* held under a *measured load*, with the failure case explicitly tested. A systems project with no benchmark and no fault injection is the engineering equivalent of a backtest with no out-of-sample split — it shows you can write code that works on the happy path, which every applicant can claim, and says nothing about the part that actually matters in production. The reviewer wants to see that you went looking for the place your system breaks and reported what you found, because the job is keeping systems alive while markets are open and a bug costs real money in real time. A README section titled "Failure modes I tested and what happened" is, for a SWE candidate, the exact analogue of the researcher's limitations section: the highest-trust part of the whole repository.

A note on language and depth for the low-latency dev track specifically. The firms with the hardest systems bar — Jump, HRT, Citadel Securities — care about C++ depth that goes well past "I can write a class": the memory model, undefined behavior, cache behavior, lock-free structures, and where the microseconds physically go. A small project that demonstrates *one* such concern with a before-and-after measurement ("replacing the `std::map` order book with a flat array of price levels cut median update latency from 850 ns to 110 ns, profiled with `perf`") signals more than a sprawling matching engine that was never profiled. Depth on one measured optimization beats breadth across an unmeasured stack, the same rule that governs the whole post, applied to nanoseconds.

The common thread across all four roles is the same: **build the thing the role actually does, then prove it does it with a number.** A researcher proves an out-of-sample Sharpe; a trader proves a P&L-versus-inventory tradeoff; a developer proves a latency reduction; an engineer proves a correctness property under load. The number is what turns a project from a claim into a signal.

## Competitions that carry real signal

Competitions are the other major signal source, and they are widely misunderstood. A competition result is attractive because it is *externally verified* — you cannot fake a Kaggle medal or an ICPC ranking, so it carries trust that a self-reported project cannot. But competitions are not interchangeable. Each one proves a specific thing to a specific audience, and a medal in the wrong arena buys you very little for your target role. Figure 3 lays out the map.

![Grid mapping five competition types to what each one signals and which role it reads strongest for, from Kaggle and IMC Prosperity to olympiads and ICPC](/imgs/blogs/building-signal-projects-competitions-and-papers-that-prove-it-3.png)

**Kaggle.** A top finish or a competition medal signals machine-learning modeling skill, feature engineering, validation discipline, and the willingness to grind a leaderboard. This reads strongest for QR and ML-research roles. The caveat that everyone misses: Kaggle rewards a *very specific* skill — squeezing the last fraction of accuracy out of a fixed dataset with a known metric — that is only a slice of real research. A Kaggle grandmaster who has never thought about out-of-sample decay or transaction costs is impressive but incompletely calibrated for markets. The signal is real; it is just narrower than the badge suggests. A single strong, well-written competition entry that you can *explain* — your validation strategy, why your model generalized — beats a wall of medals you can only point at.

**IMC Prosperity.** This is the market-relevant competition that is closest to a trading job: a multi-round trading-and-optimization game run by IMC Trading where teams trade simulated assets, manage positions, and reason about a simulated market. A strong placement signals trading EV thinking, the ability to model a market, and teamwork under time pressure. It reads strongest for QT seats — and because it is run by a market-making firm, a good result is also a direct line into that firm's pipeline.

**Firm challenges.** Optiver, Jane Street, and others run their own coding and trading challenges, sometimes as standalone events, sometimes as the front door of their recruiting funnel. These have a dual signal: they prove the relevant skill (speed and EV for Optiver-style challenges, problem-solving for Jane Street puzzles) *and* they put you on the firm's radar directly. The EV here is unusually good because the effort doubles as an application — winning or even placing well can generate a recruiter outreach you did not have to chase.

**Math olympiads (IMO, Putnam) and informatics olympiads (IOI).** These signal a raw problem-solving *ceiling* — the kind of compressed, high-pressure reasoning that quant firms prize. An IMO medal or a strong Putnam rank is a powerful tailwind for QR and QT applications because it is a famously hard, externally-verified ability test. The honest caveat: these are mostly accessible if you competed in school; an adult cannot easily go win the IMO. If you have one, foreground it. If you do not, do not despair — they are a tailwind, not a gate.

**ICPC and Codeforces.** Competitive programming — a strong ICPC regional or world-finals result, or a high Codeforces rating — signals algorithmic depth and coding speed under pressure. This reads strongest for QD and SWE seats and is a genuine tailwind for the coding rounds at HRT, Jump, and Citadel Securities, whose [coding bar](/blog/trading/quant-careers/the-quant-resume-that-passes-the-screen) is competitive-programming-adjacent. A Codeforces rating is a clean, continuous, verifiable number — one of the most efficient signals you can build if algorithms are your strength.

A subtlety that decides whether a competition is worth your hours: the signal lives in the *result*, but the *learning* lives in the process, and only one of those two things ages well. A Codeforces rating you earned by genuinely getting faster at algorithms keeps paying off, because the underlying skill shows up again in the live coding rounds; a rating you inflated by grinding pattern-matched problems decays the moment an interviewer asks you to reason aloud about an unfamiliar one. The same is true of Kaggle: the candidates who convert a medal into an offer are the ones who can articulate *why* their validation scheme prevented leakage and *why* their model generalized — the medal opened the door, but their understanding closed the deal. So treat a competition as a forcing function for real skill, not as a badge to farm. The honest test is simple: if you could not re-derive your result in front of an interviewer, the competition taught you less than its leaderboard claims, and the signal will not survive contact with the room.

One more practical point on choosing *which* competition. Because each one signals a different thing to a different audience (Figure 3), the worst outcome is to spread yourself across all of them and place strongly in none. A team that finishes top-decile in IMC Prosperity has a sharp, role-matched signal for trading; the same hours spread across a half-finished Kaggle entry, a casual ICPC attempt, and a Prosperity flameout produce three weak signals that sum to nothing. Pick the competition whose signal matches the role you decided on in step one of your portfolio plan, commit to placing well in that one, and let the others go. Depth beats breadth in competitions exactly as it does in projects, and for the same reason: a screener is looking for one strong, verifiable proof, not a participation list.

#### Worked example: the expected value of entering a competition

Should Maya spend a month on IMC Prosperity? Let us treat the decision the way the whole series treats everything — as expected value under uncertainty. We will compare *effort*, *signal*, and *realistic payoff*, and we will be honest that the payoff is a probability, not a certainty.

Maya is choosing between two uses of one month (~80 hours): (A) grind toward a strong IMC Prosperity finish, or (B) deepen her market-making simulator into a second, related project.

- **Option A — IMC Prosperity.** Realistically, a first-time team has maybe a 15% chance of a top-decile finish that is genuinely resume-worthy, a 50% chance of a respectable-but-unremarkable finish, and a 35% chance of an early flameout. The *signal* on a top-decile finish is high *and* role-matched (trading EV, the exact thing QT screens for) *and* it routes directly into IMC's pipeline. Call the value of the strong outcome a meaningful callback lift; the value of the middling outcome is small (a line on the resume but not a differentiator); the flameout is near zero. Weighted: \$0.15 \times \text{high} + 0.50 \times \text{small} + 0.35 \times 0\$. The EV is decent but front-loaded onto the 15% tail.
- **Option B — deepen the sim.** Near-certain to produce *something* shippable (call it a 90% chance of a solid second project), but the marginal signal of a second project on top of an already-strong first one is smaller than the signal of the first one was — diminishing returns, which we will quantify in the misconceptions section.

The honest conclusion is not "always do the competition." It is: a competition is a *high-variance, externally-verified* signal, and you should size your bet on it the way you would size any positive-EV bet with a fat tail — worth it if you can afford the time and the role-match is high, skippable if the variance would crowd out a near-certain flagship. For Maya, whose target is trading and whose first project is already strong, IMC Prosperity is a good marginal bet *because the signal is role-matched and the firm pipeline is a bonus*. For Wei, whose target is research, that same month is better spent on a second, deeper research artifact, because Prosperity's trading signal is off-target for him.

*A competition is a positive-EV lottery ticket whose value depends almost entirely on whether its signal matches your target role; buy the ticket when the role-match is high and you can absorb the variance.*

## Papers and research: when they are worth it

The most over-romanticized item on this list is the published paper. Let us be precise about when it actually moves the needle, because the conventional wisdom is wrong for most candidates.

A peer-reviewed publication is a strong, externally-verified signal of one specific thing: that you can do original research, see it through review, and write at a professional standard. For **research-scientist roles** at the most research-heavy systematic shops — [Two Sigma and D. E. Shaw](/blog/trading/quant-careers/do-you-need-a-phd-the-backgrounds-that-get-hired) being the canonical examples — a track record of real publications (especially in ML, statistics, optimization, or a quantitative science) is genuinely valuable and sometimes close to expected. These firms hire many PhDs precisely because the work resembles academic research, and a paper is the native currency of that world.

For almost everyone else — every trading seat, every dev seat, most research seats at prop and market-making firms — a paper is *low signal per hour*. The reason is brutal arithmetic: a publishable paper is often a year or more of work with an uncertain outcome, whereas a credible backtest repository is three weekends with a near-certain shippable result. If the firm wants to see that you can do the *job-shaped* research loop, a clean public repo demonstrates it faster and more directly than a paper on an unrelated topic. A paper on graph neural networks does not prove you can build a trading signal; a trading-signal backtest does.

The honest framing: **do not chase a paper for the sake of a line on your resume.** If you are already in a PhD and publishing is the natural output of your program, by all means foreground your best paper and connect it to the role. But if you are an undergraduate or a career-switcher wondering whether you "need" a publication to get hired, the answer is almost always no — and the time you would spend chasing one is far better spent on a deep, well-presented project. The [PhD question post](/blog/trading/quant-careers/do-you-need-a-phd-the-backgrounds-that-get-hired) covers this in full; the short version for portfolios is that the paper is a *research-scientist* signal, not a universal one.

There is a middle path worth naming: a *technical write-up that is not a peer-reviewed paper* — a long, careful blog post or a detailed project report that reads like a research note. This captures much of the "I can do and communicate research" signal at a fraction of the cost. Wei's reversal backtest, written up as a clear research note with the methodology and the honest limitations, is doing the work of a paper for the purpose of a job application, without the year of review.

## How to present the work

A common and painful failure: a candidate builds a genuinely good project and then presents it so poorly that no one ever sees the quality. Presentation is not polish for its own sake — it is the *channel* through which your signal reaches the reader, and a noisy channel destroys signal. There are three layers to get right: the README, the resume bullet, and the demo.

### The README: question, method, result, limitations, reproduce

A reviewer skims a README top-down for about ten seconds before deciding whether to read more. The structure in Figure 6 is built around that reality: the question and the result must land first, with the supporting detail below for the reader who keeps going.

![Five-layer stack of a project README from question and result at the top down through method, limitations, and reproduce steps at the bottom](/imgs/blogs/building-signal-projects-competitions-and-papers-that-prove-it-6.png)

- **Question (one sentence).** The single thing the repo answers. "Does 5-day reversal predict S&P 500 returns?" A reviewer who reads only this line already knows what kind of mind built the repo.
- **Result, stated first.** Most engineers bury the result at the bottom. Invert it: put the headline number and one plot at the top, where a skim catches it. "Sharpe 0.9 out-of-sample, net of 5 bps costs."
- **Method.** The reproducible recipe: data range and source, the signal rule, the out-of-sample split, how costs were applied. Enough that a reviewer believes the result and could re-derive it.
- **Limitations.** Where the edge breaks, what you did *not* test, why you would not trade it live. This section, counterintuitively, is the highest-trust part of the whole README — it proves you are the kind of person who interrogates their own work. Its absence is a red flag to an experienced reader.
- **Reproduce.** One command runs the whole thing end-to-end, dependencies are pinned, the figures regenerate from data. A repo that does not run is worse than no repo, because it converts a positive signal into a negative one.

### The quantified resume bullet

The README lives in the repo; the bullet lives on the resume and is the thing that earns the *click* to the repo. The transformation in Figure 2 is the entire skill: replace topic-words with an action, a tool, and a number.

![Before-and-after comparison of a weak project bullet that names a topic versus a strong bullet that states a question, a method, and a quantified result](/imgs/blogs/building-signal-projects-competitions-and-papers-that-prove-it-2.png)

#### Worked example: a weak project README and bullet, rewritten strong

Take a single real-feeling project and watch the before-and-after, because the gap between them is the difference between getting filtered and getting read.

**Before (weak).** README: *"A machine learning project for stock trading. Uses Python, pandas, and scikit-learn to predict stock prices. Got good results in backtesting."* Resume bullet: *"Built a stock-trading bot using machine learning; achieved good results in backtesting."*

What a screener sees: three claims, zero verifiable facts. "Good results" is the phrase that auto-rejects, because it is the phrase someone uses when they have not measured anything or when the result does not survive scrutiny. There is no question, no out-of-sample, no costs, no number. It reads like a tutorial the candidate followed.

**After (strong).** README opens with: *"Does 5-day reversal predict S&P 500 returns, 2010–2024? Walk-forward, net of 5 bps costs: Sharpe 0.9 out-of-sample versus 1.6 in-sample. Limitation: edge halves after costs and concentrates pre-2020."* Resume bullet: *"Backtested a 5-day equity-reversal signal on the S&P 500 (2010–2024) with walk-forward validation and 5 bps costs; reported Sharpe 0.9 out-of-sample (vs 1.6 in-sample) and documented the post-2020 decay."*

What a screener now sees: a falsifiable question, a real validation method, two numbers whose *gap* shows statistical maturity, costs, and an honest caveat — every word a verifiable fact. The bullet is longer, but it earns its length; the screener can decide in ten seconds that this person thinks like a researcher, and the repo link rewards the click. The project did not change. Its *legibility* did, and legibility is what the channel transmits.

*The weak version is a list of adjectives; the strong version is a list of facts, and a screener can only act on facts.*

### The demo

For some projects — a simulator with a visualization, a systems component with a benchmark dashboard — a short demo (a GIF in the README, a one-page result plot, a 90-second screen recording) compresses the signal further. It is not always necessary, but when the project has a visual or a measurable runtime story, a demo lets the reviewer *see* the result without running anything. Maya's market-making sim, for instance, gains a lot from a single plot of P&L-versus-inventory across the naive and skewed agents — one image carries the whole result.

### The anti-patterns

There are three ways portfolios quietly fail, and they are common enough to name explicitly.

- **Tutorial clones.** A project that follows a well-known tutorial step-by-step — the canonical "predict stock prices with an LSTM" notebook that thousands of people have submitted — signals almost nothing, because it proves you can follow instructions, not that you can do the job. Reviewers recognize these on sight. If your project's structure matches a popular tutorial, it is not signal; it is noise wearing a costume.
- **No results.** A project with code but no measured outcome — no Sharpe, no latency number, no correctness property, no plot — leaves the reviewer with nothing to evaluate. "I built X" without "and here is what X achieved" is a half-finished signal. The number is not optional.
- **Overclaiming.** A README that reports a Sharpe of 4 with no costs and no out-of-sample, or a bullet that says "production-grade trading system," sets off every alarm an experienced reviewer has. Overclaiming converts a project from positive signal to *negative* signal, because now the reviewer wonders what else you would overstate. Calibrated honesty — reporting a modest, real number — is more impressive than an incredible one, in this field specifically, because calibration is the job.

## How to build your portfolio

Knowing what carries signal is half the battle; the other half is sequencing the work so you actually ship without burning a year. The plan in Figure 7 is deliberately a loop, not a checklist: pick the role, build one flagship, ship it, present it, then iterate from real feedback.

![Six-stage sequenced pipeline for building a portfolio from picking a role through one flagship, tight scoping, shipping, presenting, and iterating with a feedback loop](/imgs/blogs/building-signal-projects-competitions-and-papers-that-prove-it-7.png)

**Step 1: Pick the role first.** Everything downstream depends on this. Trader, researcher, developer, or engineer — the choice determines which project in Figure 1 you build. Do not build a generic project and hope it fits everywhere; it will fit nowhere strongly. If you are genuinely unsure between two roles, pick the one whose project you would enjoy building, because enthusiasm is what gets a project finished.

**Step 2: One flagship, not five.** Choose a single project that *is* the thing the role does. One deep project beats a long list, for a reason we will quantify in a moment. The anatomy of the strongest flagship — a backtest — is in Figure 4; the same discipline (a sharp question, honest testing, a real result) applies to a simulator or a systems component.

![Pipeline showing the six stages of a credible backtest project from a falsifiable question through data, signal, an out-of-sample test, costs, and an honest write-up](/imgs/blogs/building-signal-projects-competitions-and-papers-that-prove-it-4.png)

**Step 3: Scope it tight.** The most common reason projects never ship is that they are scoped too big. Maya's sim is *one* asset, *one* order-flow model. Wei's backtest is *one* signal on *one* universe. A project you can finish in two or three weekends and present cleanly beats an ambitious project that sits 60% done forever. A shipped 0.9-Sharpe backtest beats an unshipped "comprehensive multi-strategy platform" every time.

**Step 4: Ship it.** Public repository, README with the result up top, runs with one command. "Shipped" means a stranger can find it, understand it, and run it. An impressive project on your laptop is worth nothing to a recruiter who cannot see it.

**Step 5: Present it.** Turn the project into one quantified resume bullet (Figure 2) and, where it helps, a demo. This is the step people skip and it is where most signal is lost — a good project with a bad bullet never gets the click.

**Step 6: Iterate.** Once it is out, you get feedback — from interviews, from mentors, from people who actually clone it. Use that feedback to deepen the flagship (add the cost analysis you skipped, the edge case you missed) or, only once the first is genuinely strong, start a second related project. The loop matters more than any single pass.

#### Worked example: the funnel math of one strong project versus a list

Let us make the "depth beats breadth" claim numeric, because it is the thesis of the whole post and it deserves a number rather than an assertion. Figure 5 is the illustrative model behind this.

![Bar chart showing illustrative callback rates rising from a resume with no project, to five tutorial clones, to one strong flagship, to a flagship plus a competition result](/imgs/blogs/building-signal-projects-competitions-and-papers-that-prove-it-5.png)

Suppose — and these are *illustrative* relative numbers, a teaching model of direction, not measured per-firm data, since no public dataset of quant callback rates exists — that a strong-but-unsignalled candidate (good school, good GPA, only coursework) gets a callback on roughly 15% of applications. This baseline is the same one used across the series, in [the resume post](/blog/trading/quant-careers/the-quant-resume-that-passes-the-screen) and the PhD post. Now add evidence:

- **Five tutorial clones, no results: ~19%.** Five projects that look like tutorials barely move the baseline — from 15% to maybe 19% — because the screener discounts them almost entirely. Five times near-zero is still near-zero. The candidate spent a month of weekends and bought four percentage points.
- **One strong flagship: ~42%.** A single deep, role-matched, honestly-presented project roughly *2.5x* the baseline, from 15% to ~42%, because it converts the candidate from "claims competence" to "demonstrated the job." This single project did more than the five clones by a wide margin, on comparable total effort, because it is the *right shape*.
- **Flagship plus a competition result: ~58%.** Adding an externally-verified, role-matched competition result on top of the flagship lifts it further, to ~58%, because now there are *two independent signals* pointing the same way.

Run the funnel math on Maya. With a 15% baseline and 20 applications, her expected callbacks are \$20 \times 0.15 = 3\$. With one strong flagship lifting her to ~42%, the same 20 applications yield \$20 \times 0.42 = 8.4\$ expected callbacks — she nearly *triples* her interview pipeline from a single well-built project. The five-clone path would have left her at \$20 \times 0.19 = 3.8\$, barely better than nothing, for the same calendar time. The lesson is not subtle: the marginal return on the *first* deep project is enormous, and the marginal return on additional shallow projects is nearly flat.

*The first deep project is the highest-return move in the whole portfolio; the curve flattens hard after it, so depth on one beats breadth across five.*

## Common misconceptions

The portfolio is one of the most myth-ridden parts of breaking into quant. Five corrections do most of the work.

**Myth: more projects are always better.** This is the misconception the funnel math directly refutes. The first deep, role-matched project roughly triples your callback rate; the second adds less; the fifth shallow one adds almost nothing and can even hurt, because a long list of tutorial-grade repos signals that you *cannot tell which work is good*. Curation is itself a signal. Two or three deep projects, well-presented, beat ten shallow ones. The recruiter is not counting repos; she is looking for the one that proves the job.

**Myth: a Kaggle medal guarantees an offer.** A medal is a strong, verified signal of one specific skill — and it is genuinely valuable — but it is not a golden ticket. It signals modeling and validation ability, which reads strongly for QR and ML roles and weakly for trading or low-latency dev. And it does not exempt you from the interview loop: you will still face the [round-by-round process](/blog/trading/quant-careers/the-research-case-and-take-home-how-to-ace-it), the mental math, the probability, the live problem-solving. The medal gets you the screen and a benefit of the doubt; it does not get you the offer. Treat it as a powerful application-stage signal, not as a substitute for being good in the room.

**Myth: you need a published paper.** For the overwhelming majority of seats — every trading and dev role, most research roles at prop and market-making firms — you do not. A paper is a research-scientist signal, valuable at the most academic systematic funds and largely irrelevant elsewhere. A clean public backtest demonstrates the job-shaped research loop in three weekends; a paper takes a year with uncertain payoff. Do not let "I have no publications" stop you from applying, and do not chase a paper as a resume line when a project would prove more, faster.

**Myth: forking and renaming an existing project counts.** It does not, and worse, it can blow up an interview. A reviewer who clones your repo can read its git history and its structure; a thin wrapper over someone else's work is transparent. And if an interviewer asks you to walk through a design decision in "your" project and you cannot explain *why* it was made — because you did not make it — the bottom falls out of your credibility instantly. The project must be yours in the sense that you can defend every choice in it. Inspiration from others is fine and normal; passing off others' work as your own is the fastest way to fail a final round.

**Myth: the code is the hard part.** The code is necessary but it is rarely what separates a strong portfolio from a weak one. The hard part is the *thinking around the code* — choosing a falsifiable question, splitting the data honestly, applying realistic costs, and writing the limitation you would rather hide. Two people can write the same 300 lines; the one whose README states the out-of-sample gap and the cost drag has the stronger portfolio, because that is the part that mirrors the actual job. Spend your marginal hour on the question and the write-up, not on a fourth model.

## How it plays out in the real world

In practice, the portfolio interacts with the rest of the funnel in a few predictable ways, and seeing them concretely is the best argument for building one.

**At the screen, the project earns the read.** The top firms — Jane Street, Optiver, SIG, IMC, Jump, HRT, Citadel, Two Sigma, D. E. Shaw — receive thousands of applications per seat, with intern and new-grad acceptance commonly described as low-single-digit percent or below (widely reported, approximate). The screener's job is to *cut*, fast. A resume with a single line linking to a credible, role-matched repo is dramatically more likely to survive that cut than an identical resume without one, because it gives the screener a reason to keep reading. This is the entire mechanism behind the two-identical-resumes story we opened with, and it is the most reliable real-world effect of a portfolio.

**In the interview, the project becomes the conversation.** Quant interviews routinely include a segment where the interviewer asks you to walk through a project on your resume. This is a gift if the project is real and yours: you get to drive a conversation on your home turf, demonstrate depth, and show how you think about exactly the tradeoffs the job involves. Wei gets to talk about why he split his data the way he did and why he reports the out-of-sample number; Maya gets to explain why her inventory skew halved her tail risk. It is a disaster if the project is a fork you cannot defend. The project you put on your resume is a promise that you can discuss it in depth — keep that promise or do not make it.

**Internships convert, and projects get you the internship.** The single highest-leverage fact about quant recruiting is that the [internship is the real interview](/blog/trading/quant-careers/do-you-need-a-phd-the-backgrounds-that-get-hired): strong programs convert most interns to return offers, and most full-time seats come from intern conversion. Internship pay is already enormous — Jane Street interns at roughly an annualized 300k USD, Citadel and Citadel Securities interns at 4,300–5,800 USD per week plus a 15k–25k sign-on and corporate housing for 2026, per the firm and survey disclosures cited across this series. Since the internship is the door to the full-time seat, getting the *internship* screen is where a project pays off most, and intern applications are exactly where candidates have the least work experience to show — which makes a project the dominant available signal. If you build one flagship in your life, build it before internship recruiting.

**The signal compounds with the rest of the application.** A project does not stand alone. It works alongside the [resume that ranks your signals](/blog/trading/quant-careers/the-quant-resume-that-passes-the-screen), the [research case or take-home](/blog/trading/quant-careers/the-research-case-and-take-home-how-to-ace-it) where you demonstrate the same skills live, and the technical foundations from the [curriculum map](/blog/trading/quant-careers/the-quant-curriculum-map-what-to-learn-in-what-order). The strongest candidates present a *consistent* picture: a researcher whose resume, project, take-home, and interview answers all point at the same disciplined out-of-sample mindset. The project is the load-bearing piece of that consistency, because it is the one a recruiter can verify before they ever talk to you.

A realistic arc, then, looks like this. Maya, six months out from trading recruiting, builds one market-making simulator over three weekends, ships it with a P&L-versus-inventory plot, writes the one quantified bullet, and enters IMC Prosperity as a marginal positive-EV bet. Wei, in his PhD, takes his strongest existing research instinct and writes one reversal backtest as a public research note, foregrounds his best paper, and skips the trading competitions that are off-target for him. Neither built ten things. Each built the *right* thing, presented it so a ten-second skim catches the result, and can defend every line of it in a room. That is what a portfolio is for: it makes invisible knowledge legible to someone who has to bet a 500k-USD package on you having it.

## When this matters / Further reading

This matters most in the months before you apply — especially before *internship* recruiting, where you have the least work experience and the most to gain from a single strong, role-matched project. It matters less once you are several years into a desk and your track record speaks for itself, but the habit it builds — framing work as a question, a method, a result, and an honest limitation — is exactly the habit that makes a good senior quant, so the discipline outlasts the job hunt.

If you take one thing from this post, take the thesis: **a small number of deep, well-presented projects beats a long list, and the right project is the one the role actually does.** Pick the role, build the one flagship, scope it tight, ship it, present the result first, and be honest about where it breaks. Knowledge is invisible until you prove it — so prove it with the thing that looks like the job.

**Read next, within this series:**

- [The Quant Curriculum Map: What to Learn in What Order](/blog/trading/quant-careers/the-quant-curriculum-map-what-to-learn-in-what-order) — the learning side of the same coin; build from what you have learned.
- [The Quant Resume That Passes the Screen](/blog/trading/quant-careers/the-quant-resume-that-passes-the-screen) — how to rank and write the bullets your project earns.
- [The Research Case and Take-Home: How to Ace It](/blog/trading/quant-careers/the-research-case-and-take-home-how-to-ace-it) — the live version of the same skills your backtest demonstrates.
- [Do You Need a PhD? The Backgrounds That Get Hired](/blog/trading/quant-careers/do-you-need-a-phd-the-backgrounds-that-get-hired) — where papers and pedigree actually matter, and where they do not.

**Build the projects themselves (technical how-to):**

- [Building an Alpha Signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) — the QR flagship, end to end.
- [Market-Making Simulator](/blog/trading/quantitative-finance/market-making-simulator-quant-research) — the QT/QD flagship.
- [Order-Book Simulator](/blog/trading/quantitative-finance/order-book-simulator-quant-research) — the low-latency QD flagship.
- [Mini Backtest Coding Challenge](/blog/trading/quantitative-finance/mini-backtest-coding-challenge-quant-interviews) — a tight, interview-grade version of the backtest.

**External references** (cite generically; comp and acceptance figures are reported, approximate, and survivorship-biased): levels.fyi salary pages for Jane Street and Citadel, the "Young & Calculated" 2026 quant-pay and internship surveys, and firm career pages (janestreet.com, hudsonrivertrading.com) for internship-conversion and challenge details.
