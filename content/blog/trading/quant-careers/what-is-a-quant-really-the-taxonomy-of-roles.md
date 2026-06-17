---
title: "What Is a Quant, Really? The Taxonomy of Roles"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Quant is an umbrella over four very different jobs on one trading floor. This disambiguates the real roles a top firm hires -- Quant Trader, Quant Researcher, Quant Developer, Software Engineer, plus Desk Strat, Risk Quant, and Quant Analyst -- what each does hour-to-hour, the math-vs-coding-vs-markets mix, who they sit next to, the deliverable they own, and how the role differs at a market maker, a pod shop, and a systematic fund."
tags:
  [
    "quant-careers",
    "quant-finance",
    "careers",
    "quant-trader",
    "quant-researcher",
    "quant-developer",
    "software-engineer",
    "trading-floor",
    "job-roles",
    "market-maker",
    "pod-shop",
    "systematic-fund",
  ]
category: "trading"
subcategory: "Quant Careers"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — "Quant" is not a job. It is an umbrella over at least four jobs that sit on the same floor, talk to each other constantly, and get confused for one another by everyone outside the building.
>
> - The four core roles are the **Quant Trader (QT)** who owns a live book of risk, the **Quant Researcher (QR)** who owns a validated signal, the **Quant Developer (QD)** who owns the system that runs both, and the **Software Engineer (SWE)** who owns the infrastructure under all of it. Three more — **Desk Strat**, **Risk Quant**, and **Quant Analyst** — sit alongside them.
> - The roles differ less in prestige than in a **measurable mix**: how much math, how much coding, how much markets, on what clock, and with how much autonomy. A QT lives on a clock of seconds; a systematic QR lives on a clock of months; a low-latency QD lives on a clock of **microseconds**.
> - The *same title* means different things at a **market maker** (Optiver, Jane Street), a **pod shop** (Citadel, Point72), and a **systematic fund** (Two Sigma, D. E. Shaw). A "QT" at a market maker quotes a vol surface all day; a "QT" at a pod shop is a portfolio manager who *hires* a pod.
> - The one fact to remember: **the deliverable defines the role.** A trader is paid for realized P&L today; a researcher is paid for an out-of-sample signal that survives next quarter. Pick the deliverable you want to own, and the role names itself.

A candidate I'll call Maya — a math undergrad who has aced two trading-game rounds and a probability screen — sat across from a recruiter and said, with total sincerity, "I want to be a quant." The recruiter smiled the way you smile at a question you have answered a thousand times and replied: "Great. Which one?"

Maya didn't have an answer, because nobody had ever told her there was more than one. The word "quant" had arrived in her life as a single shimmering thing — high pay, hard math, secretive firms — and she had aimed at it the way you aim at a city without yet knowing which street you'll live on. The recruiter's question was not a trick. It was the most important question in her job search, and she was about to learn that the same word on the same floor referred to four people who do almost nothing alike.

Walk onto the floor of a top firm and you'll see them sitting within arm's reach of each other. One person is staring at a depth-of-book ladder, fingers on a keyboard, adjusting a quote every few seconds; her P&L updates in real time and she will be judged on it tonight. Two seats over, someone is in a Jupyter notebook running a cross-validation that won't finish for an hour; he is hunting for a signal he hopes will still work in three months. Behind them, a third person is reading a flame graph, trying to shave four microseconds off the path between a market-data packet arriving and an order leaving — four microseconds that are worth real money. And down the row, a fourth person is on a video call about the deployment pipeline that lets all three of them ship code to production without taking the whole desk down. All four are "quants." All four would describe their job completely differently.

This post is the map Maya needed. We are going to disambiguate the roles a top firm actually hires for, name what each one *does* hour to hour, and locate the deliverable each one owns. By the end you should be able to point at one box on a chart and say "that's the job I want" — which is worth far more in a job search than wanting to be a "quant" in the abstract. Figure 1 is that chart: the taxonomy, with "Quant" at the top and the real roles as its leaves.

![Taxonomy tree with Quant at the top branching into front-office, build, and strat-risk groups, then into Quant Trader, Quant Researcher, Quant Developer, Software Engineer, Desk Strat, Risk Quant, and Quant Analyst](/imgs/blogs/what-is-a-quant-really-the-taxonomy-of-roles-1.png)

This is the second post in the series, and it pairs tightly with two siblings: [the four paths — trader, researcher, developer, engineer](/blog/trading/quant-careers/the-four-paths-trader-researcher-developer-engineer) goes deeper on choosing between the big four, and [how quant firms actually make money](/blog/trading/quant-careers/how-quant-firms-actually-make-money) explains the business that creates these jobs in the first place. Here we stay on the taxonomy: who is who, and why it matters which one you become.

## Foundations: the trading floor and the research org, from zero

Before we can split the roles apart we need a shared vocabulary, because the roles are *defined* by their relationship to a handful of concepts. If you already work in markets, skim this; if you are brilliant but new to finance, read it slowly, because every role below is just a different answer to "which of these things do you own?"

**A market** is a place where buyers and sellers meet to exchange an asset — a stock, an option, a futures contract, a government bond, a cryptocurrency. At any instant there is a highest price a buyer is willing to pay (the **bid**) and a lowest price a seller is willing to accept (the **ask** or **offer**). The gap between them is the **spread**. If Apple is "bid 200.00, ask 200.02," the spread is two cents.

**A market maker** is a firm that quotes *both* a bid and an ask continuously and stands ready to trade either side. It earns the spread: it buys at 200.00 from whoever wants to sell, sells at 200.02 to whoever wants to buy, and pockets the two cents — many, many times a day. The risk is that the price moves against the inventory it accumulates before it can offload it. Firms like **Optiver** (founded 1986, Amsterdam), **Jane Street** (founded 2000, New York), **Susquehanna / SIG** (1987), and **IMC** (1989) are, at their core, market makers. So is **Citadel Securities**, which — confusingly — is a different company from the Citadel hedge fund.

**Edge** is any reason you expect to make money on a trade on average. The market maker's edge is the spread plus its skill at managing inventory. A signal-driven fund's edge is a forecast: a belief that this stock will outperform that one over the next week. Edge is always probabilistic — you are not right every time; you are right *on average*, by enough, after costs.

**Expected value (EV)** is how you quantify edge. If a trade wins \$1 with probability 0.55 and loses \$1 with probability 0.45, its EV is `0.55 × (+\$1) + 0.45 × (−\$1) = +\$0.10`. Everything in this industry — the trades, the research bets, even the job-search strategy — is an EV calculation under uncertainty. That is the spine of this whole series: *the job is a probabilistic edge, and so is getting it.*

**P&L** ("profit and loss") is the running tally of money made and lost. **Realized** P&L is locked in; **unrealized** ("mark-to-market") P&L is the paper value of open positions. A trader's day is, quite literally, watching a P&L number move.

**A book** is the full set of positions one person or desk is responsible for. "My book" is "everything I'm long and short and the risk that comes with it." Owning a book means owning its P&L — and being on the hook when it goes wrong.

**A signal** (or **alpha**) is a quantitative forecast of future returns: a number, computed from data, that says "this asset is likely to go up (or down) relative to others." A good signal is a tiny statistical edge applied across thousands of names. **Alpha** is the part of returns you generate through skill, above what the market gives you for free. Building one is the QR's core craft, covered end-to-end in [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research).

**Execution** is the act of turning a desired position into actual trades in the market without giving away too much edge to slippage and fees. "I want to be long 100,000 shares" is a decision; *getting* there at a good average price, without moving the market against yourself, is execution — a craft of its own.

**A desk** is a team organized around a strategy or product — "the ETF desk," "the equity options desk," "the rates desk." **A pod** is the pod-shop version: a small team led by a **portfolio manager (PM)** who runs an independent book with its own risk limits and its own P&L, inside a larger fund like **Citadel** (founded 1990 by Ken Griffin) or Point72. **A systematic fund** like **Two Sigma** (2001, ~70 billion USD AUM) or **D. E. Shaw** (1988, ~65 billion USD AUM) runs centralized models across the whole firm rather than independent human-run pods.

Two more terms will keep recurring. **Latency** is the time delay between an event and a response — between a price update arriving and your order leaving — measured in microseconds (millionths of a second) on the fastest desks. **Slippage** is the difference between the price you expected and the price you actually got: if you decide to buy at \$100.00 but your buying pressure pushes the average fill to \$100.04, you've slipped four cents, and across millions of shares that is real money lost. Latency is mostly the QD's problem; slippage is mostly the QT's and the execution system's problem; and both quietly eat the QR's beautiful backtested edge once it meets the real world — which is exactly why a signal that looks great on paper can be untradeable in practice.

Put those together and you have the **front-to-back pipeline** that all of these roles live inside, shown in Figure 3: raw **data** comes in, **research** turns it into a **signal**, an **execution system** turns the signal into orders, a **trader** sizes and oversees the resulting risk, and the trades produce **P&L**, which is attributed back and fed into the next round of research.

![Pipeline of seven stages from raw data through research signal execution system trader and P&L to a feedback stage that loops back to research](/imgs/blogs/what-is-a-quant-really-the-taxonomy-of-roles-3.png)

The crucial property of this pipeline is that it is a **loop**, not a line. The P&L at the end is attributed — broken down to figure out which signal, which trade, and which decision made or lost money — and that attribution feeds straight back into the next round of research. A signal that decays gets retired; a trade that worked gets more capital; a system bottleneck that cost slippage gets prioritized for the next sprint. Every role is a different station on this loop, and every role's output is some other role's input. That interdependence is why these four people sit within arm's reach of each other, and it is why "they're all just quants" both is and isn't true. Now we can name them, one station at a time.

## The Quant Trader (QT): owns the live book

The Quant Trader is the person whose P&L you can watch in real time. If the firm is a market maker, the QT is responsible for keeping good two-sided quotes in the market and managing the inventory that piles up as a result. If the firm is a pod shop, the QT *is* the PM — the person who decides what the book holds, sizes the positions, and is on the hook for every dollar.

**What they do, hour to hour.** A market-making QT spends the day inside the live market. They watch the order book, watch their inventory drift long or short, and continuously adjust their quotes — widening the spread when volatility spikes, skewing it to lean against an unwanted position, pulling quotes entirely around a news event. They are not typing much math; they are making fast, calibrated decisions under pressure, many times a minute, all day. A pod QT spends less time on the millisecond clock and more on the position level: which stocks to hold, how big, against which hedges, and when to cut a loser. Both are, fundamentally, **risk managers with a P&L target.**

**Inputs and outputs.** The QT *consumes* signals (from QRs), tools (from QDs), and live market data; they *produce* a managed book and the realized P&L it throws off. The deliverable is the book.

**The math/coding/markets mix.** Medium math, medium coding, *high* markets. A QT needs enough math to reason about EV, variance, correlation, and option greeks in their head — fast — but they are not deriving new models. They need enough coding to prototype an idea, query their positions, and read a backtest, but they are not building production systems. What they need most is **markets intuition**: a feel for how prices move, how liquidity dries up, how to update a belief when the tape surprises them. This is exactly what the trading-game interview rounds at Jane Street and SIG test — see [market-making games in quant interviews](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews) for the technique, and note that SIG literally trains traders with **poker** because poker is EV-under-uncertainty with a clock.

**Who they sit next to.** Other traders, the QRs feeding them signals, and the QDs who own the systems they trade through. On a pod they sit *with* the pod.

**A day-fragment.** It's 9:25 a.m., five minutes before a Fed announcement. The market-making QT widens her quotes across the board — the spread she'll accept just went up, because the next thirty seconds carry far more uncertainty than usual, and a quote that was fairly priced an hour ago is now a gift to anyone who knows something she doesn't. At 9:30 the number prints, prices jolt, her inventory swings short in a heartbeat, and she leans her quotes to buy back what she's short before the move runs further. By 9:35 the dust is settling, she re-tightens her spreads, and her P&L for the morning is up — not because she predicted the Fed, but because she managed her risk *around* an event she couldn't predict. That is the job: not forecasting, but pricing uncertainty and managing inventory through it.

#### Worked example: a QT sizing a quote and its EV

Maya, now a junior market-making QT on an equity-options desk, is quoting a particular option. Her model says fair value is \$2.00. She decides to quote **\$1.98 bid / \$2.02 ask** — a two-cent edge on each side, a four-cent spread.

A customer order arrives to *buy* 100 contracts (each contract is 100 shares, so 10,000 underlying units). Maya sells 100 contracts at her ask of \$2.02. Her immediate edge versus fair value is `\$2.02 − \$2.00 = \$0.02` per share × 10,000 = **\$200** of theoretical edge captured.

But she is now *short* 100 contracts of an option whose fair value is \$2.00, and she has to hold that risk until she can hedge or offload it. Suppose the bid-side fill probability and the chance the underlying moves against her give the position a holding cost — call it an expected adverse move of \$0.015 per share before she can flatten. That is `\$0.015 × 10,000 = \$150` of expected give-back.

Her expected P&L on the trade is therefore `\$200 − \$150 = \$50`. Positive — but thin. Now suppose volatility is rising and her real adverse-move estimate is \$0.025 per share, or \$250. Then her EV is `\$200 − \$250 = −\$50`: the trade is a *loser* at this spread. The correct response is not to refuse to trade; it is to **widen the quote** — say to \$1.96 / \$2.04 — so the captured edge rises to \$400 and the EV goes back positive even under the higher adverse-move estimate. The skill is doing this arithmetic in milliseconds, repeatedly, with calibrated estimates, while the market is moving.

*The QT's job, stripped to its core, is to keep quoting at spreads where expected edge beats expected adverse selection — and to recompute that the instant the market tells her the estimate was wrong.*

## The Quant Researcher (QR): owns the signal

The Quant Researcher is the person staring at a notebook that won't finish for an hour. Their job is to find a real, durable, tradeable forecast of future returns — a signal — and to prove it is real before anyone bets money on it. Where the QT lives on a clock of seconds, the QR lives on a clock of weeks: an idea, a careful test, a write-up, a kill-or-promote decision.

**What they do, hour to hour.** Form a hypothesis ("post-earnings drift is stronger in mid-caps"), pull and clean the data, engineer features, fit a model, and — the part that separates good QRs from dangerous ones — try as hard as possible to *disprove their own idea* before trusting it. They run out-of-sample tests, purge look-ahead bias, and ask whether the apparent edge would survive transaction costs. Most ideas die here, and a good QR kills their own ideas cheerfully, because the alternative is losing real money in production. The discipline of killing ideas honestly is its own skill, covered in the research-writeup material the series links out to.

**Inputs and outputs.** The QR *consumes* data and research infrastructure; they *produce* a validated signal plus the evidence that it is real and an estimate of how much it can be traded. The deliverable is the signal — and the trust that it will work out of sample.

**The math/coding/markets mix.** *High* math, *high* coding, medium markets. A QR needs real statistics and machine learning — cross-validation done correctly, an understanding of overfitting, the ability to reason about whether a Sharpe ratio is real or a mirage — and enough engineering to manipulate large datasets fluently in Python. Markets intuition matters (a signal that makes no economic sense is usually overfit) but the QR is one step removed from the live tape. At a systematic fund like Two Sigma or D. E. Shaw, this is the *core* role, and a PhD is common — though, as we'll see in the myths section, not required.

**Who they sit next to.** Other researchers, the QDs who turn validated signals into production code, and — at a pod — the PM/QT they feed signals to in a tight loop.

**A day-fragment.** The QR's morning backtest finished overnight and the result looks *too* good — a Sharpe of 2.4, which his instinct says is suspicious for a signal this simple. Most of the day is spent trying to break it: he checks for look-ahead bias (is he accidentally using tomorrow's data to predict tomorrow?), re-runs it with realistic costs, splits the sample into a period he never looked at, and stress-tests it across market regimes. By late afternoon he finds it — a subtle timing bug where the signal used a closing price that wouldn't actually be known until after the trade. Corrected, the Sharpe falls to 0.7. A worse researcher would have shipped the 2.4 and lost money in production; the good one is *relieved* to have caught it, because the most expensive signals are the ones that look brilliant right up until real money hits them. The job is as much disciplined skepticism as it is cleverness.

#### Worked example: a QR judging whether a signal is any good

Wei — our recurring CS-PhD aiming at research — has built a candidate equity signal and needs to decide whether it is worth promoting. He measures the **information coefficient (IC)**: the rank correlation between his signal today and realized returns tomorrow, across the universe of stocks each day. His signal posts an average daily IC of **0.04**.

To a newcomer, 0.04 looks like failure — it is almost zero correlation. To a quant, it is *normal and potentially valuable*, because the signal is applied across thousands of names every day, and a tiny edge repeated at huge breadth compounds. The **fundamental law of active management** captures this: information ratio is approximately `IC × √(breadth)`. If Wei's signal has IC ≈ 0.04 and he can take roughly 200 independent bets a day across ~250 trading days (breadth ≈ 50,000), his information ratio is on the order of `0.04 × √50,000 ≈ 0.04 × 224 ≈ 9` in this idealized form — wildly optimistic, because real bets are correlated, but it explains *why* a 0.04 IC is exciting rather than embarrassing.

Wei then computes the backtested **Sharpe ratio** — annualized return divided by annualized volatility — and gets **1.8** gross. But he is disciplined: he re-runs the test with realistic transaction costs and finds the signal turns over its entire book every two days. After costs, the net Sharpe drops to **0.9**, and the strategy only makes money if execution is cheap. That changes everything: the signal is real but its *tradeability* depends on the QDs and the execution stack. A "good" signal is never just a high backtest number; it is a number that survives out-of-sample testing, costs, and the deflation you apply for the dozens of variants you tried. The full battery — IC, IR, Sharpe, drawdown, turnover, decay — is exactly the toolkit in [evaluating alpha signals: IC, Sharpe, turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research).

*The QR's job, stripped to its core, is to separate signals that are real from signals that merely look real on the data you happened to test — and a 0.04 IC that survives that scrutiny is worth more than a 0.20 IC that doesn't.*

## The Quant Developer (QD): owns the system

The Quant Developer is the person reading a flame graph, hunting microseconds. They build and own the software that turns the QR's signal and the QT's decisions into actual orders in the market — fast, correctly, and without falling over. At a high-frequency firm the QD's craft is *latency*: the time between a market-data packet arriving and an order leaving the building. At a systematic fund the QD's craft is *throughput and correctness*: a research platform and backtest engine that thousands of strategies can run on.

**What they do, hour to hour.** Write and optimize production trading code (often C++ for the latency-critical path), profile and eliminate bottlenecks, design the data structures that hold an order book in memory, reason about cache lines and lock-free queues and kernel-bypass networking, and — crucially — make sure the system never loses money through a bug. A single mispriced quote shipped to production can cost more in a minute than the QD's salary for a year, so correctness is not optional. They also build the tools the QRs and QTs use: backtesters, simulators, position dashboards.

**Inputs and outputs.** The QD *consumes* signals and trading logic plus performance requirements; they *produce* the live trading system, the execution path, and the research tooling. The deliverable is the working system.

**The math/coding/markets mix.** Medium math, *high* coding, medium markets. The QD needs deep computer-science fundamentals — algorithms, data structures, the memory hierarchy, concurrency — and, at low-latency firms, genuine C++ mastery: the memory model, undefined behavior, templates, cache effects. The interview reflects this: firms like **Hudson River Trading** (2002, "engineering excellence drives everything") and **Jump Trading** (1999, deep systems bar) put candidates through a C++ depth round and a systems-design round on kernel bypass and real-time constraints. HRT explicitly lets you pick C++ or Python for parts of the loop. The technique is laid out in [C++ for low-latency quant interviews](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews).

**Who they sit next to.** Other developers, the SWE/infra engineers below them, and the QTs and QRs whose systems they build. On a low-latency desk, the QD often sits *on the desk* with the traders.

**A day-fragment.** The QD opens a profiler trace from the previous trading session and finds a recurring spike: every few hundred microseconds, the order path stalls for an extra two microseconds. The hunt is forensic — is it a memory allocation on the hot path (a cardinal sin in low-latency code), a cache miss, a lock contention, a garbage-collection pause in a component that should never garbage-collect? She traces it to a logging call that, under load, occasionally blocks on disk. The fix is to move logging off the critical path entirely, into a separate thread with a lock-free queue. She writes it, tests it in the simulator the SWE team maintains, and ships it behind a flag so it can be rolled back instantly if it misbehaves in production. The whole change is maybe forty lines of C++, and it is worth more than its weight in gold because on this desk, two microseconds on the hot path is the difference between winning and losing a race that happens tens of thousands of times a day.

#### Worked example: a QD reasoning about a latency budget in microseconds and dollars

A QD on an HFT options desk is asked: "Is it worth spending two weeks to shave **4 microseconds** off our order path?" The current path — from a market-data packet hitting the network card to an order leaving — is **18 µs**. The proposal would bring it to **14 µs**.

Here is the dollars-and-microseconds reasoning. In this market, the desk competes to react to a price update before other market makers. Suppose that on the events that matter — a sudden move where being first to update your quote means you either capture a good fill or avoid getting picked off — each such event is worth, in expectation, about **\$5** of edge to whoever is fastest, and there are roughly **40,000** such events per day. Being 4 µs faster doesn't win *every* race, but historical data suggests it flips the desk from winning ~48% of these races to ~54% — a **6-percentage-point** improvement.

The daily value of those 4 microseconds is `40,000 events × \$5 × 6% = \$12,000` per day, or roughly **\$3 million per year** if it holds (it won't hold perfectly — competitors also speed up — but even half of that is enormous). Against that, two weeks of one QD's time costs the firm perhaps \$30,000–\$50,000 fully loaded. The EV of the project is overwhelmingly positive, and "is 4 µs worth it?" answers itself. This is why HFT firms employ FPGA engineers and pay for kernel-bypass NICs: at the front of the queue, microseconds *are* money.

*The QD's job, stripped to its core, is to turn engineering into edge — and on a latency-critical desk, the exchange rate between microseconds and dollars is steep enough that a few microseconds justifies a small team.*

## The Software Engineer (SWE / infra): owns the foundation

The Software Engineer is the person on the deployment-pipeline call. The distinction between QD and SWE is real but fuzzy, and firms draw the line differently. The cleanest way to see it: the **QD** owns code that is *close to the trade* — the execution path, the pricing engine, the backtester. The **SWE/infra** engineer owns the code that *everything else runs on* — the data pipelines, the compute cluster, the deployment system, the monitoring, the internal developer platform, the storage layer that holds petabytes of market data.

**What they do, hour to hour.** Build and operate the platform: distributed data ingestion, the systems that store and serve historical data to researchers, CI/CD pipelines that let hundreds of people ship code safely, observability so that when something breaks at 2 a.m. someone knows *where*, and the cloud or on-prem compute fabric that backtests run on. At a systematic fund, "AI/ML plus distributed computing" is the stated culture of the place, and the infra team is what makes research at scale possible. This work looks the most like Big-Tech software engineering of any role here — which is exactly why strong SWEs from Google or Meta are recruited into quant firms, often at a comp bump.

**Inputs and outputs.** The SWE *consumes* requirements from researchers, traders, and developers plus reliability targets; they *produce* the platform, the data infrastructure, and the tooling. The deliverable is a foundation that doesn't fall over.

**The math/coding/markets mix.** *Low* math, *high* coding, *low* markets — and that is completely fine. You can have a long, well-paid, intellectually serious career in a quant firm without ever forming an opinion about a stock. The SWE is judged on systems: scalability, reliability, latency, and developer velocity. Markets knowledge helps you collaborate, but it is not the deliverable. This is the role that most directly refutes the "you must be a math genius" myth.

**Who they sit next to.** Other engineers, the QDs (the QD/SWE boundary is a daily conversation), and platform stakeholders across the firm.

**A day-fragment.** A SWE on the data platform starts the day with an alert: an overnight data-ingestion job failed and a chunk of the previous day's market data is missing, which means a dozen researchers' backtests would silently run on incomplete data if nobody catches it. The morning is triage — find the bad records, backfill them, and add a validation check so the same gap can't pass undetected next time. The afternoon is a design review for a new feature store that will let researchers share engineered features instead of each re-computing them, which could save thousands of CPU-hours a week. None of this involves a market view, and all of it is load-bearing: a quant fund is, in large part, a software company that happens to trade, and the SWE is the reason the software holds. The reason strong Big-Tech engineers get recruited (often with a comp bump) is that this is genuinely hard distributed-systems work with unusually high stakes — a platform outage during market hours can halt trading firm-wide.

The math/coding/markets mix across all the roles is summarized in Figure 2 — read down a column to see which roles are math-heavy versus code-heavy, and read across a row to see how a single role blends the three.

![Matrix grading five roles QT QR QD SWE and Strat across five dimensions math coding markets time-horizon and autonomy with colored cells](/imgs/blogs/what-is-a-quant-really-the-taxonomy-of-roles-2.png)

## The control and product roles: Desk Strat, Risk Quant, Quant Analyst

The four roles above are the headline acts, but a top firm hires three more quant-flavored roles that you should be able to recognize — partly so you can target them, partly so you understand the floor you'll work on.

**Desk Strat** (short for "strategist," a term that originated at investment banks like Goldman Sachs) sits *with* a trading desk and builds the models, tools, and pricing logic the desk needs *right now*. A strat is a hybrid: part QR (builds models), part QD (writes the tools), embedded with a specific desk and serving its immediate needs. Where a pure QR might spend a month on one signal, a strat might ship three small pricing improvements in a week because the desk needs them. The clock is hours to weeks, the autonomy is lower (you serve the desk's agenda), and the math/markets mix is high because you must understand the product you are pricing — exotic options, structured products, a particular futures market. Strats are common at banks and at firms with complex products; the role is the connective tissue between research and trading.

**Risk Quant** owns the *model* of how much the firm could lose. They build and validate the risk models — value-at-risk, stress scenarios, exposure limits, margin models — that keep the firm from blowing up. This is high-math, medium-coding, and it sits slightly apart from the P&L-chasing front office: the risk quant's job is to be the adult in the room, to say "this position is too big for the tail risk it carries." It is intellectually deep (a lot of probability and extreme-value statistics) and absolutely critical, even if it gets less glory than the trading seat. At a multi-strat pod shop, the central risk function is what lets the firm give PMs autonomy without the whole place being one correlated bet.

**Quant Analyst** is the most overloaded title of all, and you should read it carefully in any specific posting. At some firms it is an entry-level research or trading-support role. At banks it often means model validation, pricing-library work, or regulatory-capital modeling. At buy-side shops it can mean a data or reporting role. The lesson is not the precise definition — it is that **the title alone tells you little**, and you must read the actual responsibilities in the job description. (This is good practice for *every* title in this taxonomy, but it bites hardest here.)

These three share a property: they own a **model or a tool**, not a live book and not the core production system. They are the strat/risk/control branch of the tree in Figure 1, and they are excellent careers — often with better hours and lower variance than the front office, which for many people is a feature, not a bug.

## The deliverable defines the role: QT versus QR, side by side

If you take one idea from this post, take this: **the deliverable defines the role, and the clock defines the deliverable.** The clearest way to see it is to put the two most-confused roles — QT and QR — directly side by side, which Figure 5 does.

![Two-column comparison matrix showing what a Quant Trader owns versus what a Quant Researcher owns across deliverable clock scored-on bad-day and failure-mode rows](/imgs/blogs/what-is-a-quant-really-the-taxonomy-of-roles-5.png)

A **QT owns a live book of risk on a clock of seconds to a day.** They are scored on realized P&L *today*. Their bad day is a position blowing through a stop; their failure mode is freezing or tilting under pressure. A **QR owns a validated signal on a clock of weeks to months.** They are scored on out-of-sample IC and Sharpe. Their bad day is a signal that looked great in the backtest dying in live trading; their failure mode is fooling themselves with an overfit result.

Notice that neither owns the other's deliverable. A brilliant researcher who cannot stomach watching a live P&L swing will be miserable as a trader. A sharp trader who finds careful out-of-sample validation tedious will be a dangerous researcher. They collaborate — the QR's signal feeds the QT's book — but they are paid for different things and they fail in different ways. When you choose a role, you are choosing which clock you want to live on and which deliverable you want your name attached to.

This is also why the *interview* differs by role. The QT interview is a [market-making game](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews) and a mental-math screen, because the firm is testing your EV-under-pressure and your calibration. The QR interview is a research case — a take-home signal or a live modeling problem — because the firm is testing whether you can find a real edge and, just as importantly, kill a fake one. The QD interview is C++ depth and systems design. They are different tests because they are different jobs.

## How the roles shift by firm type

Here is the subtlety that trips up even people inside the industry: the *same title* means different things at different kinds of firm. A "Quant Trader" at Optiver and a "Quant Trader" at Citadel have job descriptions that barely overlap. The three archetypes — covered in depth in [the firm archetypes: prop vs HFT vs pod shop vs systematic fund](/blog/trading/quant-careers/the-firm-archetypes-prop-vs-hft-vs-pod-shop-vs-systematic-fund) — bend every role. Figure 6 lays out how QT, QR, and QD each change across a market maker, a pod shop, and a systematic fund.

![Three by four grid showing how Quant Trader Quant Researcher and Quant Developer roles change across a market maker a pod shop and a systematic fund](/imgs/blogs/what-is-a-quant-really-the-taxonomy-of-roles-6.png)

**At a market maker** (Optiver, Jane Street, SIG, IMC, Citadel Securities), the firm's edge is the spread plus speed and inventory management. The **QT** quotes a vol surface or a set of instruments and manages inventory all day — a fast, market-facing job. The **QR** prices instruments and calibrates the models the quotes are built on. The **QD** owns the microsecond order gateway and the low-latency stack. The whole organization is oriented around *being on the other side of a flow at a good price*, so the QT seat is the central, prestigious one and the clock is short.

**At a pod shop** (Citadel, Point72, Millennium, Balyasny), the firm is a collection of independent **pods**, each a PM running a book. Here the **QT is a portfolio manager** — they own a book, they often *hire* the QRs and QDs in their pod, and they are accountable for the pod's P&L with sharp consequences (a pod that breaches its drawdown limit can be shut down). The **QR** in a pod feeds the PM signals in a tight, fast feedback loop — closer to the trade than a systematic-fund QR. The **QD** builds the research and trading platform *for the pod*. The organizing unit is the autonomous, high-accountability pod, and the QT/PM sits at the top of it.

**At a systematic fund** (Two Sigma, D. E. Shaw, WorldQuant, Renaissance), the firm runs centralized models across everything, so the **QR is the core role** — the long-horizon researcher hunting durable alpha is what the firm *is*. A pure discretionary **QT is rarer**; trading is largely automated, and the "trader" role becomes overseeing the execution of the model and handling exceptions. The **QD** builds the distributed backtest cluster and the production model-running system. WorldQuant takes this to its logical extreme with its **alpha-factory** model: a distributed network of researchers all producing formulaic alphas on a shared platform. The organizing unit is the centralized model, and the researcher is king.

So when a posting says "Quant Trader," your first question should be "at what kind of firm?" — because that single fact tells you whether you'd be quoting options at microsecond speed, running an independent book as a PM, or babysitting an automated execution system. The title is a starting point, not an answer.

## Which role fits which person

Roles are not better or worse; they fit different people. Here is how to think about the match, and then we'll map our two recurring characters onto the chart.

You probably fit **Quant Trader** if you are energized rather than drained by real-time decision-making, you like markets and want a fast feedback loop on your judgment, you are calibrated and competitive without being reckless, and you can stay cold when money is moving against you. The trading-game rounds aren't a hazing ritual; they are a genuine preview of the job, and people who love them tend to love the seat.

You probably fit **Quant Researcher** if you find a hard statistical problem more satisfying than a fast decision, you have the discipline to attack your own ideas, you enjoy the long arc of an investigation more than the adrenaline of the tape, and you can sit with uncertainty for weeks. If reading [overfitting and purged cross-validation](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research) sounds like fun rather than homework, this might be you.

You probably fit **Quant Developer** if you love systems and performance, you get a kick out of making something fast and correct, you'd rather build the machine than place the bet, and the idea of shaving microseconds (or scaling a backtester to a thousand machines) lights you up. You can be deeply technical and well-paid here without ever forming a market view.

You probably fit **Software Engineer / infra** if you love distributed systems, reliability, and platforms, and you want the intellectual seriousness and comp of a quant firm without the market-facing parts. This is the role for the strong Big-Tech engineer who wants harder problems and better pay.

And you might fit **Strat / Risk / Analyst** if you like being the connective tissue (strat), the disciplined model of disaster (risk), or you're entering through a more product- or validation-shaped door (analyst) — often with better work-life balance than the front office.

#### Worked example: mapping Maya and Wei to the role that fits each

Recall **Maya**: a math undergrad who lit up in the trading-game rounds, loves fast decisions, is competitive, has solid-but-not-research-grade coding, and described the adrenaline of the market-making sim as "the most fun interview I've ever done." Map her against Figure 2's dimensions. Math: medium-high but not her edge. Coding: medium. Markets: *high* — she's hungry for the tape. Clock: she wants the *fast* one. Autonomy: she wants to own a P&L. That profile points squarely at **Quant Trader**, ideally at a **market maker** where the trading-game skills she already loves are the daily job. The EV of her targeting QT roles — given her demonstrated fit — is far higher than spraying applications across every "quant" posting. She should aim her prep at mental-math speed and market-making games, not at C++ systems design.

Now recall **Wei**: a CS-PhD who built a signal, measured its IC at 0.04, and *enjoyed* the discipline of killing his own first three ideas before trusting the fourth. He finds a hard cross-validation problem more satisfying than a fast trade, and he's comfortable with a result taking weeks. Math: high. Coding: high. Markets: medium. Clock: he's happy on the *slow* one. That profile points at **Quant Researcher**, and given his PhD and his patience for long-horizon work, a **systematic fund** like Two Sigma or D. E. Shaw — where the QR is the core role and a PhD is common — is the highest-EV target. He should aim his prep at research cases and signal-evaluation, and link out to [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) for the craft. If he instead chased the QT seat because it sounded higher-status, he'd be optimizing for the wrong fit — and probably interviewing worse, because the trading games reward a temperament he doesn't have.

*Same word, "quant," two completely different right answers — and the moment each of them can point at one box on the chart, their job search gets dramatically more efficient.*

## Common misconceptions

The fog around "quant" is thick with myths. Here are the ones that cost people the most.

**Myth 1: "QR is higher status than QT, so aim for research."** There is a persistent belief that the researcher is the "real" quant and the trader is somehow lesser. This is backwards in many places. At a market maker or a pod shop, the **QT/PM owns the P&L** — and at the senior end, the trader's comp can dwarf the researcher's, precisely because the trader carries the risk and the accountability. The two roles are differently scored, not ranked. Status varies by firm and by seat, and chasing the one that *sounds* prestigious instead of the one that *fits* is how people end up in a job they're bad at. Comp by role at the new-grad level is shown in Figure 4, and the gaps are smaller than the mythology suggests.

![Bar chart of reported new-grad total compensation ranges by role with Quant Trader and Quant Researcher leading Quant Developer and Software Engineer](/imgs/blogs/what-is-a-quant-really-the-taxonomy-of-roles-4.png)

**Myth 2: "Developers and engineers are 'just support.'"** This one is both wrong and expensive — wrong because the QD often *is* the edge (re-read the microsecond worked example: a small dev team can be worth millions a year), and expensive because believing it leads strong engineers to skip quant firms or to undervalue their own seat. At HRT and Jump, "engineering excellence drives everything" is not a slogan; it is the business model. The dev and infra roles are first-class, well-paid, and central. The pay reflects it: as of 2025–2026, new-grad QD and SWE total comp at top firms runs in the low-to-mid hundreds of thousands of USD, and the senior engineering ladder is long.

**Myth 3: "You must code in C++."** C++ is the language of the *low-latency path* — the QD seat at an HFT firm. But vast swaths of this industry run on **Python** (research, signals, tooling), **OCaml** (Jane Street's functional-programming bet), and ordinary backend stacks (the SWE/infra world). A QR who is fluent in Python and statistics and a SWE who is excellent at distributed systems may never write performance-critical C++. C++ depth matters intensely for *one* track and barely at all for others. Match the language to the role, not to the rumor.

**Myth 4: "Quant means math genius and nothing else."** Math matters, but the taxonomy makes clear that it is *one* of three axes, and it is the dominant axis for only some roles. The QT's edge is markets intuition and calibrated decision-making. The QD's and SWE's edge is engineering. Even the QR's edge is as much *research discipline* — the honesty to kill your own ideas — as raw mathematical horsepower. A "math genius" who can't code, can't reason under pressure, and can't tell a real signal from an overfit one will struggle in every one of these seats. The roles reward a *blend*, and different blends. The backgrounds that get hired span math, CS, physics, statistics, EE, and operations research — and a PhD, while common in research, is **not required** for trading or most dev roles.

**Myth 5 (bonus): "There's one path in, so prepare one way."** Because the roles are different jobs with different deliverables, they have different interviews and different prep. Preparing for a QR research case will not get you through a QD systems-design round, and grinding C++ will not make you faster at a market-making game. Pick the role first; then prep for *that* role. The EV of targeted preparation is far higher than generic "quant prep."

## How it plays out in the real world

Let's ground all of this in real firms and real numbers, with the conditionality the honest version requires.

The four roles map cleanly onto the named firms. **Jane Street** hires QTs, QRs, ML researchers, and FPGA/systems engineers — and famously runs its whole stack in OCaml, interviews with puzzles and markets games, and converts most of its strong interns. **Optiver** and **SIG** are market makers whose front door is a brutal mental-math screen (Optiver's is roughly 80 questions in 8 minutes, no calculator) for the trading track. **HRT** and **Jump** are engineering-first HFT shops where the QD/SWE bar is the dominant filter — C++ depth, systems design, kernel bypass. **Citadel** (the pod-shop hedge fund) and **Citadel Securities** (the market maker) are *different companies with different hiring*, a distinction that confuses an astonishing number of applicants. **Two Sigma**, **D. E. Shaw**, and **WorldQuant** are systematic shops where the QR is the protagonist.

On comp — and read this with its conditionality, because the headline numbers are survivorship-biased. As of 2025–2026, **new-grad total comp** (base + sign-on + on-target bonus) at the top tier runs roughly **\$450k–\$650k on-target** for front-office roles, with **base** around **\$250k–\$375k** and the rest in sign-on and bonus. Jane Street quotes an "annualized equivalent of \$300k" base across its QT/QR/ML/FPGA roles. On levels.fyi (2025), Jane Street QR comp clusters around a **\$250k** median with a range from roughly \$307k at L1 up toward \$565k; Citadel QR runs higher, with a median near **\$325k**, L1 around \$336k, and senior levels reaching \$642k and beyond. QT comp at Jane Street shows a base near \$300k with the 75th percentile around \$407k and the 90th near \$512k. QD and SWE/infra new grads start a notch lower on bonus though similar on base — the low-to-mid hundreds of thousands — with a long, lucrative senior engineering ladder. Figure 4 shows these new-grad bands by role.

Three caveats you must internalize. First, **base is flat-ish; the bonus is the lever**, and the bonus ties to P&L contribution and **does not repeat automatically**. A trader who earns a \$1.3M bonus in a strong year (giving ~\$1.5M total at 2.5 years of experience in a great seat) can earn a fraction of that the next year if the seat or the P&L changes. The headline number describes a *good year for a survivor*, not a guaranteed annuity. Second, the people quoting "\$600k by year five" are the ones who *survived* an up-or-out filter; many wash out, switch firms, or land well below the headline — that is the [how quant firms actually make money](/blog/trading/quant-careers/how-quant-firms-actually-make-money) business model doing its job. Third, the **internship is the real interview**: programs like Jane Street's (interns at roughly an annualized \$300k) and Citadel's (\$4,300–\$5,800/week for 2026 plus sign-on and housing) convert most strong interns to full-time, so the highest-EV move is to win the internship, in the right role, first.

Finally, the **time-horizon spread** across roles — which is really a spread across *clocks* — is worth seeing on a log scale, because it spans about thirteen orders of magnitude. Figure 7 plots it: a QD on an HFT gateway reasons in **microseconds**, a market-making QT in **seconds to minutes**, a pod QT or QR in **hours to weeks**, and a systematic-fund QR in **months to years**. When people say these are "all quant jobs," they are technically right and practically misleading: a job measured in microseconds and a job measured in years are not the same job.

![Horizontal log-scale chart of time-horizon by role from microseconds for an HFT developer through seconds for a market-making trader to months and years for a systematic researcher](/imgs/blogs/what-is-a-quant-really-the-taxonomy-of-roles-7.png)

## When this matters / Further reading

This matters the moment you start applying. A candidate who tells a recruiter "I want to be a quant" has handed over the steering wheel; a candidate who says "I want a Quant Trader seat at a market maker, because I love the markets-games temperament and the fast clock" has told the recruiter exactly where to put them — and has, not coincidentally, prepared for the right interview. The taxonomy is not trivia. It is the difference between a scattershot search with a low offer rate and a targeted one with a high one, which is just the EV-under-uncertainty spine of this series applied to your own career.

The practical takeaways:

- **Identify your deliverable.** Do you want to own a live book (QT), a validated signal (QR), a trading system (QD), a platform (SWE), or a model/tool (strat/risk/analyst)? The deliverable is the role.
- **Pick your clock.** Microseconds, seconds, weeks, or years — the time-horizon in Figure 7 is a real, daily fact of the job, and it should match your temperament.
- **Read past the title.** "Quant Trader" and "Quant Analyst" mean different things at different firms; always read the responsibilities, and always ask "at what kind of firm?"
- **Prep for the role, not for "quant."** The QT, QR, and QD interviews are different tests. Targeted prep has far higher EV than generic prep.

Where to go next. Within this series: [the four paths — trader, researcher, developer, engineer](/blog/trading/quant-careers/the-four-paths-trader-researcher-developer-engineer) goes deeper on choosing among the big four; [how quant firms actually make money](/blog/trading/quant-careers/how-quant-firms-actually-make-money) explains the business that creates and pays for these roles; and [the firm archetypes: prop vs HFT vs pod shop vs systematic fund](/blog/trading/quant-careers/the-firm-archetypes-prop-vs-hft-vs-pod-shop-vs-systematic-fund) maps how every role bends by firm type. For the craft each role hires for, link out to [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) and [evaluating alpha signals: IC, Sharpe, turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research) for the QR track, and [C++ for low-latency quant interviews](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews) for the QD track. For comp and firm facts, the reported ranges above come from levels.fyi (Jane Street and Citadel pages), Glassdoor, H1B disclosures, and the "Young & Calculated" 2026 quant-pay survey — useful starting points, but always read them as conditional, survivorship-biased ranges rather than promises.

Maya's recruiter asked "which one?" and that question, which felt like an obstacle, was actually the gift: it forced her to choose a deliverable and a clock, and the choice made everything after it sharper. Figure out which box on the tree is yours. Then go get *that* job.
