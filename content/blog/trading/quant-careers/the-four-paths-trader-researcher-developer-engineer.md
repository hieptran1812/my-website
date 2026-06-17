---
title: "The Four Paths: Trader, Researcher, Developer, Engineer"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "A decision-grade comparison of the four core quant career paths -- Trader, Researcher, Developer, Engineer -- so you can choose your lane: the core skill, the day, the comp ceiling and its variance, the autonomy, the failure mode, the 10-year arc, the personality that fits, and the prep that gets you in."
tags:
  [
    "quant-careers",
    "quant-finance",
    "careers",
    "quant-trader",
    "quant-researcher",
    "quant-developer",
    "software-engineer",
    "career-choice",
    "compensation",
    "expected-value",
    "variance",
    "career-path",
  ]
category: "trading"
subcategory: "Quant Careers"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Four very different careers hide behind the word "quant," and the right question is not "which pays most" but "which payoff shape fits the person you are."
>
> - The four core paths are the **Quant Trader** (decides the bet, owns the P&L), the **Quant Researcher** (finds the edge, owns the signal), the **Quant Developer** (builds the trade path, owns the latency), and the **Software / Infra Engineer** (holds the whole machine up, owns reliability and scale). A firm needs all four because each one's output is the next one's input.
> - They differ less in prestige than in **payoff shape**. The trader has the highest ceiling and the widest variance — and the brutal "P&L-or-out" downside. The researcher's path is intellectual with a long, slow feedback loop. The developer builds the machine, with very strong and stable comp. The engineer owns scale and reliability, with the tightest band and the **most transferable** skill set.
> - Choosing is an expected-value-under-variance decision, not a pick of the biggest headline number. A trader's reported \$3.5M survivor median at year 8 sits on top of a band whose floor is a poor seat at ~\$475k and an exit; an engineer's ~\$850k is far more *certain*. Same EV math you'd run on a trade — run it on your career.
> - The one fact to remember: **you can switch paths later, but you cannot un-spend the years.** Choose the lane whose *daily work* and *failure mode* you can live with for a decade, because the comp follows the work, not the other way around.

Picture one person — call her Maya, a math undergrad who just cleared two trading-game rounds and a probability screen — standing at a literal fork in a hallway at a top firm's open house. Down the left corridor, a row of traders stare at depth-of-book ladders, P&L numbers twitching in real time, someone laughing too loud after a good fill. Down the right, a quieter room: researchers in headphones, six monitors of notebooks and cross-validation curves, a whiteboard covered in someone's failed signal. Around a corner, developers argue about a cache miss that cost four microseconds. And in a windowless room nobody at the open house gets shown, infra engineers watch dashboards that, if they go red, take all three of the other rooms down with them.

The recruiter who walked Maya in said something that stuck: *"You could do any of these. The interviews overlap more than you'd think. The question isn't whether you'd get in — it's which one you'd still want to be doing at thirty-five."* That is the question this post is built to answer.

The companion post in this series, [what is a quant, really](/blog/trading/quant-careers/what-is-a-quant-really-the-taxonomy-of-roles), **defined** these roles — who sits where, what deliverable each owns, how the titles shift across firm types. This post does the next thing: it helps you **choose**. For each of the four paths we'll lay out, side by side, the core skill, the day-to-day, the comp ceiling and — crucially — its *shape*, the autonomy, the failure mode that washes people out, the ten-year trajectory, the personality that thrives, and the prep that actually gets you in. Figure 1 is the whole comparison on one page; the rest of the post is the honest detail behind each cell.

![Comparison grid with four columns -- Quant Trader, Quant Researcher, Quant Developer, Software and Infra Engineer -- and rows for core skill, the day, comp shape, autonomy, washes out when, and who fits](/imgs/blogs/the-four-paths-trader-researcher-developer-engineer-1.png)

This is post five in the series, and it sits between three siblings worth opening in another tab. [The taxonomy of roles](/blog/trading/quant-careers/what-is-a-quant-really-the-taxonomy-of-roles) defined the cast; [the firm archetypes](/blog/trading/quant-careers/the-firm-archetypes-prop-vs-hft-vs-pod-shop-vs-systematic-fund) explains how the *same path* looks different at a prop shop versus a pod versus a systematic fund; and [do you need a PhD](/blog/trading/quant-careers/do-you-need-a-phd-the-backgrounds-that-get-hired) maps which background gets you into which lane. Here we stay on the choice itself.

## Foundations: the four levers a quant firm needs

Before you can choose a path, you have to see why a firm needs all four — because the paths are defined by their place in one machine, not by job titles in isolation. If you already work in markets, skim this. If you are brilliant but new to finance, read it slowly, because every choice later in the post is just a different answer to *"which lever do you want to be?"*

Start with the only thing a quant firm fundamentally does: it **turns information into trades that have positive expected value, repeated enough times that the law of large numbers makes the edge show up as profit.** Everything else is plumbing around that sentence. Let's define the plumbing.

**Expected value (EV)** is how every decision in this industry is scored. If a bet wins \$1 with probability 0.55 and loses \$1 with probability 0.45, its EV is `0.55 × (+\$1) + 0.45 × (−\$1) = +\$0.10` per dollar risked. A firm does not need to be right often; it needs to be right *on average*, by enough, after costs, across a huge number of repetitions. This is the spine of the whole series: *the job is a probabilistic edge — and so is getting it, and so is choosing it.* You'll see the same EV-versus-variance math we apply to trades applied to your own career before the post is done.

**Edge** is any reason you expect positive EV. A market maker's edge is the **spread** — the gap between the price buyers will pay (the **bid**) and the price sellers will accept (the **ask**) — plus skill at managing the inventory that piles up. A systematic fund's edge is a **signal** (also called **alpha**): a number, computed from data, that forecasts which assets will out- or under-perform. Edge is always probabilistic and always *decays* — once enough people find the same signal, it stops paying.

**P&L** ("profit and loss") is the running tally of money made and lost. **Realized** P&L is locked in; **unrealized** (or "mark-to-market") P&L is the paper value of open positions. A trader's career is, quite literally, the integral of a P&L curve over time.

**A book** is the full set of positions one person or desk owns. "Owning a book" means owning its P&L — and being on the hook when it goes wrong. **Latency** is the delay between an event and your response, measured in microseconds (millionths of a second) on the fastest desks. **Slippage** is the gap between the price you expected and the price you actually got. **Deferred comp** is the chunk of a bonus paid out over later years rather than immediately — a retention tool that quietly raises the cost of quitting.

Now the machine itself. Raw **data** comes in — ticks, order books, fundamentals, news, alternative data. A **researcher** turns that data into a signal: a validated forecast that survives out-of-sample. A **developer** turns the signal into a fast, correct **trade path**: the code that takes a market-data packet, computes a decision, and gets an order onto the exchange in microseconds. A **trader** sizes the resulting risk, oversees the live book, and is judged on the P&L. And an **engineer** builds and holds up the platform underneath all three — the data pipelines, the deployment system, the storage, the monitoring — so that the other three can do their jobs without the whole thing falling over at 9:31 a.m. Figure 3 (later in the post) draws this as a loop, not a line: the P&L at the end is *attributed* back to the signal and the trade that produced it, and that attribution feeds the next round of research.

The crucial property is that **none of the four can ship value alone.** A signal with no execution path is a paper; a trade path with no signal is a fast way to lose money; a trader with no research is a gambler; and all three are dead the moment the platform underneath them goes down. The firm needs every lever, and it pays each one for owning a different kind of risk. That is why these are four real careers and not four flavors of one — and it is why the choice between them matters so much. You are not picking a salary. You are picking which risk you want to own for a decade.

One more piece of vocabulary that shapes the choice: **deferred comp** and **garden leave**. A large fraction of a senior person's bonus is often paid out over the following two-to-three years rather than all at once, and many seats carry a **non-compete** that bars you from joining a rival for a period — often three to twelve months of "garden leave," paid but idle — after you quit. Both are retention tools, and both raise the *cost of switching*: the deferred comp you'd forfeit and the months you'd sit out are a real, computable price on leaving. This matters to path choice because it bites the front-office paths hardest — a trader or researcher carrying a book or a signal portfolio is the person a firm most wants to lock in — while the engineer's transferable skills and weaker lock-ins keep their exit cheaper. The freedom to leave is itself part of a path's payoff shape.

A note on the firm-shape caveat before we dive in: the *same path name* means materially different things at a **market maker** (Optiver, Jane Street, IMC, SIG), a **pod shop** (Citadel, Millennium), and a **systematic fund** (Two Sigma, D. E. Shaw). A "trader" at a market maker quotes a vol surface all day; a "trader" at a pod shop is a portfolio manager who hires a team. A "researcher" at Two Sigma or D. E. Shaw is often a PhD research scientist on months-long horizons, while a "researcher" at a fast market maker works on far shorter signals next to the traders. We flag those splits as we go, but the firm-by-firm detail lives in [the firm archetypes post](/blog/trading/quant-careers/the-firm-archetypes-prop-vs-hft-vs-pod-shop-vs-systematic-fund).

## Path 1 — The Quant Trader: own the bet, own the P&L

![Layered graph showing data feeding the researcher, the researcher's signal feeding the developer, the developer's trade path feeding the trader, the engineer's platform feeding the developer, and the trader's realized risk feeding a P&L attribution stage](/imgs/blogs/the-four-paths-trader-researcher-developer-engineer-3.png)

**The core skill.** A trader's skill is *making good decisions under uncertainty, fast, repeatedly, without flinching.* Notice what's *not* on that list: it is not the hardest math (a researcher's is harder), and it is not the most code (a developer writes more). The trader's craft is calibration — knowing the difference between a 55% bet and a 60% bet and sizing each correctly — plus the emotional regulation to keep doing it when the last three decisions lost money. At a market maker, the concrete form of this is quoting two-sided markets and managing inventory; at a pod shop, it is sizing a portfolio of positions against a risk budget. Figure 3 above shows where the trader sits: at the end of the pipeline, holding the realized risk, with the P&L flowing out of their seat.

**The day.** A trader lives on a clock of seconds to minutes. The morning is pre-open: checking overnight news, positioning, risk limits. The session is a wall of attention — quotes adjusting, fills coming in, hedges firing, the P&L number moving the whole time. There is genuine adrenaline and genuine grind; the same five-second loop, thousands of times, while staying sharp on the one in a thousand that matters. The day ends with the book flat or hedged and a number that is either green or red, and which everyone — including you — can see.

**The comp shape — and this is the whole point.** The trader has the **highest ceiling and the widest variance** of any path. As reported on levels.fyi and Glassdoor for 2025–2026, a strong junior trader at a top firm clears \$450k–\$650k first-year total comp; a strong mid-level trader 2–3 years in can hit \$1.5M in a good seat (base ~\$200k + bonus ~\$1.3M); and a top trader 5–7 years in is an outlier at \$8M–\$12M. But every one of those numbers is *conditioned on a live, profitable book.* The base is flat (~\$200k–\$300k); the bonus is the lever, and the bonus is a direct function of your P&L contribution. It does **not** repeat automatically. Figure 2 shows the survivor median curve; Figure 4 shows the band, and the band is the truth the median hides.

![Line chart of median total comp in thousands of USD by years of experience for the four paths, with the trader curve rising steepest to about 3.5 million dollars at year 8 and the engineer curve the steadiest at about 850 thousand](/imgs/blogs/the-four-paths-trader-researcher-developer-engineer-2.png)

**The autonomy.** Among the highest of the four. A trader who has earned a book makes real-money decisions all day with their own discretion inside risk limits. At a pod shop the autonomy is near-total — a PM runs an independent book — and so is the accountability.

**The failure mode — "P&L-or-out."** This is the defining downside, and it is real. A trader who loses money, or who simply fails to make enough relative to the capital and risk they're consuming, *loses the seat.* There is no slow PIP, no two-year glide path. A bad stretch can end a career in months. The industry's reputation for being "up-or-out" is most literally true here. The skills are also the least transferable: a great trader's edge is partly the specific markets, products, and firm infrastructure they've internalized, which does not port cleanly to a different desk, let alone a different industry.

**The ten-year arc.** Survive the first two years and the curve steepens fast (Figure 2): you go from quoting one product under tight limits to owning a book with a real risk budget, mentoring juniors, and being one of the people whose P&L the desk depends on. Figure 5 draws this junior-to-senior jump for all four paths. The arc has the highest peak and the highest chance of *not getting there.*

**Who fits.** Calm under loss. Competitive but not tilted by it. Fast mental arithmetic. Comfortable being measured in public, daily, by a single number. Energized rather than drained by markets and speed. Crucially: **high tolerance for variance** — you have to genuinely prefer a wide payoff distribution with a high ceiling to a narrow one with a guaranteed floor, because that is the literal shape of the job.

**How to prep.** The interview is built to test exactly the job: timed **mental-math screens** (60–80 questions in 8 minutes, no calculator, pass bar ~70–85%), **probability and expected-value** puzzles, and **trading / market-making games** that test calibration, updating, and grace under pressure. Susquehanna famously uses poker as training; Jane Street's games are pure EV-under-uncertainty. The right preparation is decision-theory practice, not memorization — start with [decision-making under uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews) and drill the speed.

#### Worked example: a trader's P&L-or-out math — what "carrying your seat" means

Suppose a firm allocates Maya a book and the *cost* of her seat — her base salary, her share of the data feeds, the technology, the risk capital's expected return — is roughly \$1.2M a year. That is the bar her book must clear before she has contributed a dollar of net profit. Say her strategy has a genuine edge: an expected Sharpe of 1.0 on \$50M of allocated risk capital, which at, say, 8% target volatility is `\$50M × 8% × 1.0 = \$4.0M` of *expected* annual P&L.

On average she clears the bar comfortably: `\$4.0M expected − \$1.2M cost = \$2.8M` of net contribution, of which she might take home 10–20%, so a bonus in the \$300k–\$560k range on top of base. Good year. But "Sharpe 1.0" means the *standard deviation* of her annual P&L is also about \$4.0M (one unit of return per unit of risk). So in any given year her P&L is roughly `\$4.0M ± \$4.0M`. A one-standard-deviation bad year lands near \$0 — below her seat cost. A two-standard-deviation bad year is *negative \$4M*, and now she is not just unpaid, she has lost the firm money. Do that for two of her first three years and the seat is gone, regardless of the edge being real, because the firm cannot tell a true edge having a bad run from no edge at all until many years of data accumulate — and it will not wait.

*The trader's comp is a call option on a noisy P&L: enormous upside, but the same volatility that creates the upside is what ends careers, and the firm exercises its "fire" option long before the law of large numbers proves you were right.*

## Path 2 — The Quant Researcher: own the edge

**The core skill.** A researcher's skill is *finding a real, robust statistical edge in noisy data and proving — to a skeptic, especially yourself — that it will survive out-of-sample.* This is the most math- and statistics-heavy path. The hard part is not running a model; the hard part is *not fooling yourself.* Financial data has a tiny signal-to-noise ratio, the data-mining temptation is enormous, and an overfit backtest looks exactly like a discovery until it loses money live. The researcher's defining craft is the discipline that separates the two: out-of-sample testing, purged cross-validation, and the willingness to kill a beautiful idea. The mechanics live in [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research).

**The day.** A researcher lives on a clock of days to months. The day is a Jupyter notebook, a feature pipeline, a backtest that won't finish for an hour, a results meeting, a literature dive. It is quiet, deep, and slow-feedback. The dirty secret of the job is that *most ideas fail* — a researcher who tests fifty hypotheses might find two that survive, and the emotional reality is long stretches of negative results punctuated by rare wins. This is the opposite of the trader's instant scoreboard, and it suits a very different temperament.

**The comp shape.** High, but it **realizes more slowly** than the trader's. A researcher's bonus is tied to the P&L of signals they produced — but a signal often has to run live for months before anyone is sure it works, so the payoff lags the work by a year or more. Reported 2025 figures (levels.fyi): Jane Street QR median ~\$250k with a range up to ~\$565k; Citadel QR median ~\$325k, reaching ~\$642k at senior levels and a top reported ~\$721k. A pod-shop QR around 4 years in might see ~\$575k (base \$175k + bonus \$400k). The ceiling is high — a researcher who owns a signal that prints \$50M a year is paid like it — but it arrives later and with somewhat less variance than the trader's lottery (Figure 4). At Two Sigma and D. E. Shaw, the most research-heavy shops, this path shades into a "research scientist" role where a PhD is common.

**The autonomy.** High on *ideas* — a good researcher has wide latitude on what to investigate — but lower on *deployment*, because a signal has to clear a review and a risk process before it touches real money. You own the hypothesis; you share ownership of whether it ships.

**The failure mode.** "No signal ships." A researcher who spends a year and produces nothing that survives out-of-sample is in trouble — not as instantly as a losing trader, but the slow-burn version is just as real. The subtler failure is *overfitting yourself into a false discovery*, shipping a signal that looked great in backtest and decays the moment it meets the market; that burns credibility, which is the researcher's true capital. The discipline to avoid it is its own skill — see [overfitting, purged CV, and the deflated Sharpe](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) for why the honest researcher kills more ideas than they ship.

**The ten-year arc.** From testing ideas other people framed, to setting the research agenda, to owning a portfolio of signals and the juniors who help build them (Figure 5). The senior researcher's value compounds: a library of validated signals and a nose for which ideas are worth a week. Comp grows strongly though it tops out below the very top traders.

**Who fits.** Deeply curious. Patient with slow loops and comfortable with mostly-failure. Intellectually honest to the point of being hard on your own ideas. Strong in probability, statistics, and the specific machine-learning-for-finance toolkit. If the idea of spending three weeks on a signal that turns out not to work sounds *interesting* rather than *demoralizing*, this is your path.

**How to prep.** The interview is a **research case** — a signal or backtest take-home, or a live case — that tests framing, avoiding overfitting, out-of-sample discipline, and the write-up, plus probability and statistics depth. The single most-tested skill is being able to **kill your own idea** out loud. Prep with [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) and the probability foundations in [the math-for-quants series](/blog/trading/math-for-quants/probability-spaces-random-variables-math-for-quants).

#### Worked example: the researcher's slow-realizing comp versus the trader's fast lottery

Take Wei, a CS PhD aiming at research, and compare him with Maya the trader over their first three years, holding *expected* total comp roughly equal so we can see the *shape* difference.

Maya the trader: year 1 ~\$500k, and because her bonus tracks a noisy P&L, year 2 could be \$1.3M (a great seat and a great year) or \$300k (the bonus that "didn't repeat"). Her three-year path might read \$500k, \$1.3M, \$300k — a total of \$2.1M, but lumpy and unpredictable, and the \$300k year is the one that puts her seat at risk.

Wei the researcher: year 1 ~\$360k. His big year-1 signal is promising but won't be *confirmed* live until midway through year 2, so his year-1 bonus is modest. Year 2, with the signal now proven, his comp jumps to ~\$575k. Year 3, owning two signals, ~\$750k. His path reads \$360k, \$575k, \$750k — a total of \$1.685M, *less* than Maya's \$2.1M, but **monotonically increasing and far more predictable.** Wei never has a \$300k scare year; his downside is "the signal takes longer to prove," not "I lose the seat."

*The researcher trades a lower, later peak for a smoother climb: the work pays off on a lag, but the lag also smooths the variance, so the same EV comes with a tamer distribution — which is exactly why a patient person should prefer it even at a lower headline number.*

## Path 3 — The Quant Developer: build the machine

**The core skill.** A developer's skill is *writing correct, fast, low-latency code that turns a signal into trades in the real world.* On the fastest desks this means C++ at a level most software engineers never touch: the memory model, undefined behavior, templates, lock-free data structures, cache lines, kernel-bypass networking. The trade path is where a researcher's beautiful signal either makes money or dies to slippage and latency, so the developer's microseconds are *directly* worth money. This is craft work of a high order — precision under hard real-time constraints — and it is in brutal demand. The depth bar is real; [C++ for low-latency quant interviews](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews) is the map of it.

**The day.** A developer's day is building and hardening the trade path: profiling a hot loop, fixing a race condition, shaving microseconds, reviewing a colleague's lock-free queue, debugging why an order didn't fire. The clock is microseconds-to-milliseconds in production but days-to-weeks in project terms — more predictable than a trader's, more concrete than a researcher's. There is real satisfaction in it: you can *see* the machine you built run, and a four-microsecond win is a number you can point to.

**The comp shape — strong and stable.** This is the path's headline advantage. Developer base salaries are similar to traders' and researchers' (~\$250k–\$375k at top firms), and total comp is excellent — new-grad ~\$320k on-target, rising to ~\$480k at 2 years and ~\$700k at 5 years on the survivor curve (Figure 2). What's different is the **shape**: the band is far tighter than the trader's (Figure 4). A developer's comp does not swing with a single book's P&L; it compounds steadily with seniority and impact. You give up the trader's \$8M outlier ceiling, but you also give up the \$300k scare year and the P&L-or-out cliff. For a person who values compounding certainty over a lottery ticket, this is often the *better risk-adjusted deal* — see the worked example below.

**The autonomy.** Medium. A developer works against specs and within an architecture; there is real ownership of *how* something is built, but *what* gets built is shaped by the desk's needs. Senior developers gain architecture-level autonomy across the firm's trade path.

**The failure mode.** "Causes an outage." A developer whose code takes down the trade path at the wrong moment — a bug that mis-prices, double-fires, or freezes during a volatile open — can cost the firm a fortune and a great deal of trust. The failure is sharp but bounded; one outage does not usually end a career the way one losing quarter ends a trader's, and the discipline of testing and code review exists precisely to make failures recoverable. The slower failure is stagnation: a developer who stops growing past mid-level plateaus on comp.

**The ten-year arc.** From shipping specced components under close review, to owning the trade-path architecture across the firm, to being one of the people the whole desk's speed and correctness depends on (Figure 5). The arc is steady and the destination is senior-staff-level comp that compounds reliably — \$1M+ for survivors at year 8 — without the trader's variance.

**Who fits.** A craftsman. Precise, detail-obsessed, satisfied by building things that work. Strong systems and CS fundamentals, deep in at least one of C++/low-latency or distributed systems. Prefers a problem with a *right answer* to a problem that's only ever probabilistic. Values a strong, predictable income and the pleasure of building over the adrenaline of the bet.

**How to prep.** The interview leans hardest on **algorithms / data structures** (competitive-programming-adjacent), then for low-latency firms a **C++ depth round** (memory model, UB, templates, lock-free, cache behavior) and a **systems-design round** (kernel bypass, real-time constraints). HRT, Jump, and Citadel Securities push hardest here. Prep with [the coding interview for quants](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews) and build something real and fast.

#### Worked example: stable comp compounding versus the trader's lottery

Compare a developer's steady path with a trader's lottery over five years, in expected-value *and* certainty terms.

The developer (call him Dev) is on the survivor curve from Figure 2: \$320k, \$400k, \$480k, \$580k, \$700k — a five-year total of **\$2.48M**, and the standard deviation around each year is small, maybe ±\$60k, because the comp tracks level and impact, not a noisy P&L. His five-year outcome is essentially *certain* to land within ~10% of \$2.48M.

The trader (Maya) has a higher *expected* path but enormous spread. Model each year as a mixture: 60% chance of a "standard" outcome, 25% chance of a "strong" outcome, 15% chance of a "poor" outcome that also risks the seat. Her expected five-year total might be ~\$3.5M — clearly higher than Dev's \$2.48M in expectation. But the *distribution* is the story: roughly a 1-in-3 chance she has at least one poor year that costs her the seat and resets her to ~\$0 of bonus and a job search, and a small chance (maybe 5%) she's the star clearing \$2M+ in a single year. Her five-year outcome ranges from "washed out at \$900k total and starting over" to "\$8M and a legend."

Now apply a little risk aversion — which any honest person should, because you cannot diversify your single career the way a firm diversifies its many traders. If your utility is roughly logarithmic in wealth (the standard model for someone who can't afford ruin), Dev's certain \$2.48M can deliver *higher utility* than Maya's higher-EV-but-high-variance \$3.5M, because the chance of the bad branch — washing out — is weighted heavily by anyone who needs the income.

*The developer is not "settling": for a person who can't run their career as a diversified portfolio, a tighter band at slightly lower EV can be the higher-utility bet — the trader's extra expected dollars are bought with variance you personally have to bear undiversified.*

## Path 4 — The Software / Infra Engineer: own the scale

**The core skill.** An engineer's skill is *building and operating systems that are reliable and scale* — the data pipelines, deployment infrastructure, storage, monitoring, and tooling that every other path runs on top of. This is the most *transferable* path: the distributed-systems and reliability skills a quant infra engineer builds are the same ones that pay top-of-market at any large tech company, which gives this path a unique property — an outside option that the other three lack. Where the developer owns the *trade path*, the engineer owns the *platform underneath it*. In Figure 3, the engineer is the lever feeding the developer the platform; if that lever fails, the whole loop stops.

**The day.** An engineer's day is platform work: a data pipeline that needs to scale, a deployment system that needs to be safe, a storage layer that's becoming a bottleneck, a monitoring gap that nearly caused an incident, an on-call rotation. The clock is the calendar — sprints, projects, reliability targets — the most "normal software job" rhythm of the four. The work is less glamorous internally (you're a layer below the people making the bets) and the discipline is reliability engineering: making sure the thing never goes down, especially at the open.

**The comp shape — solid and tightest.** This path has the **narrowest band** of the four (Figure 4) and the most predictable curve (Figure 2): ~\$280k new-grad on-target, ~\$420k at 2 years, ~\$600k at 5 years, ~\$850k at 8 years on the survivor curve. The ceiling is the lowest of the four because the engineer is one step removed from the P&L — your comp tracks impact and seniority, not a trading book. But "lowest ceiling" here still means top-decile tech comp, and the path's hidden value is **optionality**: an infra engineer at a quant firm can walk into a senior role at any major tech company, which both raises their floor (a strong outside offer) and de-risks the whole career.

**The autonomy.** Medium. An engineer owns platform decisions and architecture within the firm's needs, with senior engineers setting the reliability bar for many desks. Less day-to-day "place the bet" autonomy than a trader, more long-horizon "shape the platform" autonomy.

**The failure mode.** "Can't scale the system" — or, more acutely, an outage that takes the platform down and stops everyone from trading. The failure is the most *visible* (when infra is down, everyone knows) but also the most *recoverable*: reliability engineering is built around blameless post-mortems and the assumption that incidents happen. One incident rarely ends a career; chronic inability to make systems scale or stay up does. The quieter risk is being undervalued internally relative to the front-office paths — an engineer has to advocate for their impact in a building where the traders make the headline numbers.

**The ten-year arc.** From owning one service under guidance to owning the platform and the reliability bar for many desks (Figure 5), with a clear senior-staff-to-principal ladder and the unique twist that the skills compound *outside* the firm too. The arc is the steadiest of the four and the most portable.

**Who fits.** Systems-minded. Energized by reliability, scale, and elegant infrastructure rather than by the bet itself. Comfortable being a foundational layer below the glory. Values a strong, stable income *and* the freedom of a deeply transferable skill set. Lowest need for variance tolerance of the four — and that's a feature, not a flaw, if it matches who you are.

**How to prep.** Strong CS fundamentals, **systems design**, distributed systems, and reliability engineering, plus the standard algorithms/coding screens. The bar overlaps heavily with senior software engineering at top tech companies, which is exactly why the skills transfer. Build and operate something at scale; the [coding and systems prep](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews) overlaps with the developer's path.

## Junior vs senior: what scope actually changes

It is tempting to think of the four paths as four destinations, but each is really a ten-year *trajectory*, and the trajectory is the same shape on every path: you start owning one small piece, and seniority widens the surface you're responsible for until you own a whole bet, a whole portfolio, a whole architecture, or a whole platform. Figure 5 lays the junior and senior versions of all four side by side so you can see the scope expansion as the constant it is.

![Before-and-after comparison with a junior column and a senior column, showing how each of the four paths expands its scope from one task to a whole book, signal portfolio, architecture, or platform](/imgs/blogs/the-four-paths-trader-researcher-developer-engineer-5.png)

Read the figure as four parallel stories of the *same promotion*. The **junior trader** trades one product under tight risk limits and is told, in effect, "don't lose money while you learn the desk"; the **senior trader** owns a book and a risk budget, sizes positions for the whole desk's exposure, and mentors the juniors who are where she was. The **junior researcher** tests ideas other people framed, one signal at a time, under a senior's review; the **senior researcher** *sets the research agenda* — decides which questions are worth a team's quarter — and owns a portfolio of validated signals. The **junior developer** ships specced components under close code review; the **senior developer** owns the trade-path architecture across the whole firm and decides how the machine is built. The **junior engineer** owns one service and is on-call under guidance; the **senior engineer** owns the platform and sets the reliability bar that many desks depend on.

Two things follow from seeing it this way. First, the *core skill never changes* — a senior trader is still making calibrated decisions under uncertainty, just on a bigger book; a senior engineer is still doing reliability engineering, just for more systems. You are not learning a new craft at year five; you are pointing the same craft at a larger surface. That's why choosing the path you genuinely enjoy at the junior level matters so much: the senior version is *more* of the same daily work, not different work. Second, the *failure mode scales with the scope*. A junior trader's bad day costs a small limit; a senior trader's bad call moves the whole desk's P&L. A junior engineer's outage hits one service; a senior engineer's design flaw can take down the platform for everyone. Seniority is not just more comp — it is more *consequence*, which is exactly why the comp rises. If the larger version of the consequence (a desk-sized P&L swing, a firm-wide outage, a research agenda that produced nothing for a quarter) sounds energizing rather than terrifying, you're on the right path.

## The honest tradeoffs: ceiling vs variance vs stability

Now put the four paths on one set of axes. The headline numbers (Figure 2) tell you the *medians*; the bands (Figure 4) tell you the *truth*. Read both together.

![Bar chart of compensation spread at five years showing floor median and ceiling for each path, with the trader band spanning roughly 475 thousand to 4 million dollars and the engineer band the tightest from 400 to 900 thousand](/imgs/blogs/the-four-paths-trader-researcher-developer-engineer-4.png)

Here is the core tradeoff in one paragraph. **Ceiling** runs trader > researcher > developer > engineer. **Variance** runs the same way: the trader's band is roughly eight times wider than the engineer's at the five-year mark. **Stability and transferability** run the *opposite* way: engineer > developer > researcher > trader. The trader buys the highest ceiling with the widest variance and the least transferable skills and the harshest failure mode. The engineer buys the tightest, most portable, most recoverable career with the lowest ceiling. The developer and researcher sit in between, with the developer leaning toward the engineer's stability and the researcher toward the trader's intellectual ownership and slower-realizing upside.

The mistake almost everyone makes is to compare *medians* — "trader makes \$2M at five years, engineer makes \$600k, obviously be a trader." That comparison is wrong twice over. First, it ignores the **floor**: the trader's \$2M median sits on a distribution whose bottom tail is "washed out and starting over at \$900k total," while the engineer's \$600k is nearly *certain*. Second, it ignores **survivorship**: the trader median is computed over *people still in seat*, which is exactly the population that didn't wash out. The honest comparison is the *distribution*, not the point — which is why Figure 4 is the most important figure in this post, and why the next worked example does the EV-and-spread math explicitly.

#### Worked example: comp ceiling vs variance — expected value AND spread over five years

Let's put numbers on all four paths over five years, reporting both the expected total *and* a rough spread (standard deviation), so you can see what you're actually choosing. These use the survivor curves (Figure 2) for the median and the bands (Figure 4) for the spread; treat them as illustrative, reported-range models, not guarantees.

- **Trader:** expected five-year total ≈ **\$3.5M**, but with a spread of roughly **±\$2.5M** and a meaningful probability (~30%) of a wash-out branch that truncates the total near \$0.9M and ends in a job search. Coefficient of variation (spread ÷ mean) ≈ **0.7** — the highest by far.
- **Researcher:** expected ≈ **\$2.6M**, spread ≈ **±\$1.2M**, wash-out probability lower (~15%) and slower. CV ≈ **0.45**.
- **Developer:** expected ≈ **\$2.5M**, spread ≈ **±\$0.5M**, wash-out rare. CV ≈ **0.2**.
- **Engineer:** expected ≈ **\$2.1M**, spread ≈ **±\$0.35M**, wash-out rare and recoverable (outside option). CV ≈ **0.17** — the lowest, and the only path with a strong external floor.

So the *expected values* are closer than the headlines suggest — \$3.5M vs \$2.6M vs \$2.5M vs \$2.1M — a spread of about 1.7x from top to bottom. But the *coefficients of variation* differ by about 4x. The trader is not "making 70% more for the same risk"; the trader is making ~40–65% more in expectation while carrying ~4x the relative variance and a 30% chance of the bad branch.

*Choosing a path is not picking the highest mean; it is picking the point on the mean-variance frontier that matches your own risk tolerance and your ability to bear an undiversified bet — exactly the calculation a portfolio manager runs, applied to the one position you can never hedge, which is your own career.*

## How to choose your path: a decision framework

Numbers can't choose for you, because the right choice depends on *who you are*. So here is a four-question framework that routes a person to a path. Figure 6 draws it as a flow; the questions are ordered by how decisively they split the field.

![Pipeline of eight stages: four decision questions about feedback speed, markets versus models, build versus decide, and variance tolerance, flowing into the four path outcomes for trader, researcher, developer, and engineer](/imgs/blogs/the-four-paths-trader-researcher-developer-engineer-6.png)

**Question 1 — Do you want feedback in seconds, or in months?** This is the single most decisive split, because it's about temperament you can't easily change. If a real-time scoreboard energizes you and slow loops make you restless, you lean *trader* (and to a lesser extent *developer*, who sees code run). If you'd rather work deeply for weeks toward a result that takes months to confirm, you lean *researcher*. The trader's instant P&L and the researcher's quarter-long signal are opposite emotional worlds; most people know within a day of honest reflection which one they want.

**Question 2 — Are you energized by markets, or by models and data?** If you find order books, flows, and the live texture of markets genuinely interesting, lean *trader*. If you find the data and the modeling interesting *regardless of the market it's about*, lean *researcher* or *engineer*. This separates people who love finance from people who love the technical problem that happens to be in finance — and both are completely valid, but they're happy in different seats.

**Question 3 — Do you want to DECIDE the bet, or BUILD the machine?** This splits the front office (trader, researcher — who decide *what* the firm bets on) from the build side (developer, engineer — who build *how* it bets). If the satisfaction is in being responsible for the call, lean front-office. If the satisfaction is in building a thing that works and watching it run, lean build-side. There is no prestige ranking here despite what the comp ceilings imply — a senior developer or engineer is as respected and as well-paid-in-real-terms as most researchers.

**Question 4 — Do you crave variance, or stability you control?** This is the tiebreaker and the most honest question. Within the front office, the trader is the high-variance option and the researcher the lower-variance one. Within the build side, both are stable, but the developer leans slightly higher-ceiling and the engineer leans more transferable and tightest-band. If you genuinely prefer a high-ceiling lottery and can bear the wash-out tail, the trader's seat is built for you. If you need a floor you can count on, the engineer's is. Most people, when they answer this honestly rather than aspirationally, are surprised which way they lean.

The framework's whole point is that there's no globally "best" path — there's a best path *for a given person*, and the questions are designed to surface the person rather than the prestige. Figure 7 makes the same point from the trait side: it grids common traits against which path each one is *decisive* for, so you can read down your own column.

![Matrix of six traits -- calm under loss, fast mental math, patience for slow loops, code craftsmanship, systems thinking, and tolerance for variance -- scored against the four paths, showing each path rewards a distinct profile](/imgs/blogs/the-four-paths-trader-researcher-developer-engineer-7.png)

Read Figure 7 down a column, not across a row. *Calm under loss* is **decisive** for a trader and merely *useful* for everyone else. *Patience for slow loops* is decisive for a researcher and a *low need* for a trader. *Code craftsmanship* and *systems thinking* are decisive for the build-side paths. And *tolerance for variance* is literally **required** for a trader, *medium* for a researcher, and *low to lowest* for the build side. The figure's lesson: a trait that makes you a great trader can be irrelevant to being a great engineer, so "am I smart enough" is the wrong question — "which profile am I" is the right one.

#### Worked example: Maya and Wei each choose, with the EV-and-variance reasoning

Now let's watch our two recurring characters actually choose, using the framework and the numbers.

**Maya** (math undergrad, aced two trading games) runs the four questions. Q1: she *loves* fast feedback — the trading game's instant scoreboard was the best part. Q2: markets genuinely interest her. Q3: she wants to decide the bet, not build the machine. Q4: she's 22, has no dependents, a strong safety net, and a high appetite for variance. Every arrow points to *trader*. Now the EV check: she's choosing the \$3.5M expected / ±\$2.5M / 30%-washout branch. For her situation — young, no obligations, genuinely energized by the daily scoreboard, able to bear a wash-out and restart — the high-variance bet is *correctly* hers: her ability to absorb the bad branch is high, so the risk-adjusted value of the trader's distribution is, for *her*, the best of the four. She targets market-maker internships and drills mental math and trading games.

**Wei** (CS PhD, aiming at research) runs the same questions. Q1: slow loops suit him — he's spent five years on a dissertation and *likes* deep, deferred-payoff work. Q2: he's energized by the modeling problem more than the market. Q3: he wants to decide what to investigate (front-office, research side) but the idea of a live P&L scoreboard makes him anxious. Q4: he has student debt and prefers a smoother climb. The arrows point to *researcher*. His EV check: \$2.6M expected / ±\$1.2M / 15%-slow-washout. For him — patient, debt-averse, energized by models — the researcher's smoother, slightly-lower-mean distribution is *higher utility* than the trader's, even though Maya, looking at the same two distributions, correctly prefers the trader's. **Same two payoff shapes, opposite correct choices, because the chooser's risk tolerance and temperament differ.** Wei targets systematic funds, builds a portfolio of clean backtests, and practices killing his own ideas out loud.

*The framework doesn't tell you which distribution is "better" in the abstract — it tells you which distribution is better for the specific person doing the choosing, which is the only sense in which "better" means anything for a career you can't diversify.*

## How it plays out in the real world

Strip away the framework and look at four real-shaped arcs, built from the reported ranges (levels.fyi, Glassdoor, and the 2026 "Young & Calculated" survey, as of 2025–2026) and flagged as illustrative.

The **trader arc** at a top market maker: intern at ~\$300k annualized, convert to a new-grad seat at \$450k–\$650k first-year total, and then the fork. The survivor who finds a profitable niche is at \$1.5M by year 2–3 (Jane Street QT 75th percentile ~\$407k base-plus, with bonus stacking on top) and a star is an \$8M–\$12M outlier by year 5–7. The non-survivor is gone in eighteen months and tells a very different story at the reunion — and you almost never hear that story, which is the survivorship bias in one sentence.

The **researcher arc** at a systematic fund (Two Sigma, D. E. Shaw): often entered with a PhD, new-grad QR total ~\$250k–\$360k, climbing to a reported \$575k around year 4 at a pod shop and ~\$642k–\$721k at senior levels at Citadel per levels.fyi. The arc is smoother and slower; the payoff is a portfolio of signals you own and the agenda-setting that comes with it.

The **developer arc** at an HFT shop (HRT, Jump, Citadel Securities): a hard C++/systems filter on the way in, base in the \$250k–\$375k band, total climbing steadily — ~\$320k new-grad to ~\$700k by year 5 — with a senior-staff ladder and no P&L cliff. Five Rings and Jane Street lead H1B base disclosures at ~\$300k; Citadel Securities ~\$257k, which tells you the *base* (the certain part) is genuinely strong on this path.

The **engineer arc** is the developer arc with a wider exit door: the same strong, stable comp, one notch lower ceiling, and a skill set that ports directly to senior roles at any major tech company — which is why this is the path you'd pick if "I want to never be trapped" ranks above "I want the biggest possible number."

## Common misconceptions

This industry is hyped and gate-kept by misinformation, so let's correct the four myths that lead people to choose the wrong path.

**Myth 1 — "The trader is always the richest, so be a trader."** False on two counts. First, the trader has the highest *ceiling and median*, but also by far the widest *variance* and a ~30% wash-out tail (Figure 4); the *expected utility* for a risk-averse person can easily be lower than the developer's tighter band. Second, the trader median is a **survivorship illusion**: it's computed over people still in seat, i.e. exactly the ones who didn't wash out. The honest statement is "the trader has the highest ceiling, the widest variance, and the harshest downside" — which is a payoff *shape*, not a ranking. Pick it because the shape fits you, not because the headline number is biggest.

**Myth 2 — "The researcher is 'the smart one.'"** False, and it does real damage by pushing technically brilliant people who'd be happier elsewhere into research. All four paths require top-tier ability; they require *different* abilities. A great trader's calibration-under-pressure is a rare cognitive gift; a great developer's ability to write correct lock-free code is a rare gift; a great engineer's systems intuition is a rare gift. The researcher's gift is statistical honesty and the patience to fail repeatedly. "Smartest" is a category error — there is no single axis, and choosing research *because* it sounds like the smart-person path is choosing for prestige, which Figure 7 shows is exactly the wrong basis.

**Myth 3 — "Dev / engineering is a fallback for people who couldn't trade."** Emphatically false. The developer and engineer paths have *strong, stable, top-decile comp* (Figure 2), high demand, the best risk-adjusted shape (Figure 4), and — for the engineer — the most transferable skills of any path. Firms like HRT explicitly say "engineering excellence drives everything," and the C++ and systems bar is in many ways the *hardest technical filter* in the building. People choose these paths because the *work* — building precise, fast, reliable machines — is what they love, and they are paid extremely well for it. Treating them as a consolation prize is how you end up a miserable trader instead of a thriving developer.

**Myth 4 — "You pick once and you're locked in forever."** False, and it's the most paralyzing myth. The paths overlap enough that switching is common: developers move into research, researchers move toward trading, traders who burn out on variance move to research or strategy, engineers move into developer roles. The interviews overlap (everyone does coding; front-office adds games and probability; build-side adds systems design), the firms are the same, and the underlying EV-under-uncertainty mindset transfers across all four. What you *can't* do is un-spend the years — so the goal isn't to find the one perfect path on day one, it's to pick the lane whose *daily work and failure mode* you can live with now, knowing you can adjust later. The cost of switching is real (you reset some seniority and relationships) but it is a cost, not a wall.

## When this matters / Further reading

This choice matters most at three moments: when you're targeting internships (the internship *is* the interview, and market-maker, research, and dev tracks recruit differently, so you have to aim before you apply); when you're choosing between offers across paths (now you're picking a payoff *shape*, and Figure 4 is the lens); and at the two-to-three-year mark, when the curves diverge and you learn whether the lane you picked actually fits — the moment when an honest switch, while it costs you some seniority, is far cheaper than a decade in the wrong seat.

The throughline is the series spine: *the job is a probabilistic edge, and so is the career.* You evaluated four paths the way you'd evaluate four trades — not by their best-case headline, but by their full distribution of outcomes weighted by your own ability to bear the downside. The trader is a high-ceiling, high-variance call option with a wash-out tail; the researcher is a high, slower-realizing, lower-variance climb; the developer is a strong, stable, compounding craft; and the engineer is the tightest, most transferable, most recoverable of all. None dominates the others. The right one is the one whose daily work energizes you and whose failure mode you can survive — because, as the whole post has insisted, the comp follows the work, and the work is what you actually do every day for ten years.

**Read next, in this series:**

- [What is a quant, really? The taxonomy of roles](/blog/trading/quant-careers/what-is-a-quant-really-the-taxonomy-of-roles) — the post that *defined* these roles, who sits where, and what deliverable each one owns. Read it first if any role term here was new.
- [The firm archetypes: prop vs HFT vs pod shop vs systematic fund](/blog/trading/quant-careers/the-firm-archetypes-prop-vs-hft-vs-pod-shop-vs-systematic-fund) — the *same path* looks materially different at a market maker, a pod, and a systematic fund; this is how to layer firm choice on top of path choice.
- [Do you need a PhD? The backgrounds that get hired](/blog/trading/quant-careers/do-you-need-a-phd-the-backgrounds-that-get-hired) — which background maps to which path, and where a PhD is required, helpful, or irrelevant.

**Prep, by path:**

- Trader → [decision-making under uncertainty](/blog/trading/quantitative-finance/decision-making-under-uncertainty-quant-interviews) for the EV-and-calibration core of the trading-game rounds.
- Researcher → [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) for the research-case craft, and [the probability foundations](/blog/trading/math-for-quants/probability-spaces-random-variables-math-for-quants) underneath it.
- Developer / Engineer → [C++ for low-latency quant interviews](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews) for the depth and systems-design bar.

**Sources for the comp and firm facts** cited above: levels.fyi (Jane Street and Citadel salary pages), Glassdoor, H1B base-salary disclosures, efinancialcareers, and the "Young & Calculated" 2026 quant-pay and internship surveys, all as of 2025–2026. Every figure here is a *reported range*, presented with its conditionality — bonuses don't repeat automatically, a strong year is not the median, and the people who quote the biggest numbers are the ones who survived the filter. Treat them as illustrative of *shape*, not as a promise of *level*.
