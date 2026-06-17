---
title: "From Senior to PM, or Starting Your Own"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "The top of the quant ladder is where you stop being paid a salary and start being paid a slice of the money you make. This is the honest map of the two routes there: taking a portfolio-manager seat inside a pod shop versus going independent and running your own fund, what each actually requires, the economics, and the survivorship math nobody puts in the recruiting deck."
tags: ["quant-careers", "quant-finance", "careers", "portfolio-manager", "hedge-fund", "pod-shop", "fund-launch", "compensation", "seeding", "entrepreneurship", "risk", "track-record"]
category: "trading"
subcategory: "Quant Careers"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — At the top of the quant ladder, the question stops being "what's my next promotion?" and becomes "do I want to *run real money*?" There are two routes: take a **portfolio-manager (PM) seat** at a pod shop or multi-strat — run a book, often a small team, and earn a **payout percentage of your P&L** — or **go independent**, seeding or spinning out your own fund and collecting a **2-and-20** fee stream you own outright. The thesis: this is where the genuinely asymmetric upside in the whole career lives, but it demands a *portable, attributable track record* and a tolerance for being the one who carries the entire risk.
>
> - **A PM payout scales with the book; an employee bonus does not.** A PM earning ~10-20% of a book's net P&L takes home a number that grows linearly with how much they make. An employee bonus on the same contribution is smaller, more discretionary, and caps out. That gap is the whole reason to want the seat.
> - **A fund is a business, not a strategy.** A 2% management fee has to pay salaries, data, prime broker, tech, legal, and compliance. Below a break-even AUM — roughly **75 million USD** in our illustrative model — the fixed costs eat the fee and the founder relies entirely on the 20% performance fee, which only pays in a good year.
> - **The currency that buys both seats is an attributable track record.** Not "I worked on the strategy that made money" — *this book, this P&L, this Sharpe, audited, and portable enough to take with you*. Without it, no firm hands you a capital line and no seeder writes a check.
> - **The one number to remember:** most small fund launches close quietly within a few years. The headline outcomes are real, but they are the *survivors*. Independence is the most asymmetric bet in the career — uncapped upside, a real and common downside, and you carry all of it.

Wei has been a principal researcher for two years, and on a Thursday afternoon his head of research closes the door and makes him an offer that he will think about every night for the next month. The desk wants to give him his own book. Not a research mandate that feeds someone else's P&L — his own capital line, his own risk limit, his own name on the daily attribution report, and a payout: a *percentage* of whatever that book makes, net of its costs. He would hire one or two analysts under him. He would, for the first time, be a portfolio manager. The number his boss writes on the whiteboard for a good year is larger than anything Wei has earned, and it is also, his boss is careful to say, "not a salary — it's a function of what the book does."

Wei walks home doing math he has never had to do before. As a principal researcher his comp is large and, relatively speaking, *stable* — a strong base, a bonus that tracks his research impact, a number that does not crater if one quarter goes sideways. The PM seat is the opposite shape: bigger on the upside, but his pay is now *his P&L times a percentage*, and his P&L can be negative. And there is a third option he has barely let himself consider, the one his old colleague Maya took eighteen months ago when she left to start her own fund with a seeder's capital. She is either about to get very rich or about to spend three years learning that running a business is a different job than having an edge. Wei does not yet know which.

This post is the map for that decision — the one almost nobody at the start of a quant career thinks about, because it sits at the *top* of the ladder, past senior, past the [IC-versus-management fork](/blog/trading/quant-careers/the-ic-vs-management-fork-staff-principal-pm-or-lead). It is where the career stops being about being paid well for being excellent and becomes about *owning the consequence of being excellent*. There are two doors, and here is the whole picture before we open either one.

![A tree diagram showing a senior quant branching into two routes to running money, a portfolio-manager seat with a capital line and a payout and a team, and going independent through a seed or a spin-out into a 2-and-20 fund](/imgs/blogs/from-senior-to-pm-or-starting-your-own-1.png)

Notice what the two routes share at the trunk: *a portable track record and your name on the P&L*. Everything downstream — the capital line, the seeder, the LPs, the payout, the fee — is something that record buys you. The left branch, the PM seat, rents you most of the rest: the firm hands you capital, infrastructure, and a way to hire. The right branch, going independent, makes you supply all of it. The rest of this post earns that picture: it defines every term from zero, walks each route, prices out the economics in dollars, and tells you the part the recruiting deck leaves out — the survivorship reality of fund launches and what it actually means to be the one who carries the whole risk.

## Foundations: what running money means

Before we can compare the routes, we have to define the words, because this is the part of the career where the vocabulary changes entirely. Up to senior, you were paid for your *work*. From here, you are paid for *outcomes you own* — and the language is the language of ownership: books, payouts, AUM, fees, capital, infrastructure. Let us build each from zero.

**Running money.** "Running money" means being the person whose *decisions* determine whether a pool of capital makes or loses money, and being *paid as a function of that result*. A junior trader executes within someone's limits; a researcher's signal feeds someone's book. The person running money is the one the P&L is *attributed to*. When the firm asks "did this capital make money this year?" the answer has a name attached, and that name is the one running the money. Everything in this post is about becoming that name.

**Portfolio manager (PM).** A PM is the person who runs a *book* — a defined pool of capital with a risk budget — and owns its P&L. At a multi-strategy hedge fund, the PM is the central unit of the whole organization: the firm is, in effect, a collection of PMs each running their own book, with the firm providing the capital, the platform, and the risk oversight. The PM decides what to trade, how big to be, when to add and when to cut. Their pay is tied directly and explicitly to their book's result.

**A book.** A book is the set of positions a PM is responsible for, plus the capital and risk limit behind it. "I run a 200-million-dollar book" means the firm has allocated that much capital to your strategies and given you a risk budget to deploy it within. The book is the unit of accounting: its P&L is computed daily, its risk is monitored in real time, and your payout is a function of its net result over the year.

**Payout percentage.** This is the heart of PM economics. Instead of a base salary plus a discretionary bonus, a PM is paid a *percentage of the net P&L their book generates*, after the firm subtracts the costs allocated to that book (financing, data, the salaries of the PM's team, a share of platform costs). Reported payout rates at pod shops cluster in roughly the **10% to 20%** range of net P&L, with the exact number depending on the firm, the PM's leverage, and how much infrastructure the firm provides. The crucial property: this is *linear in the P&L*. Make twice as much, take home roughly twice as much. That linearity is what makes the PM seat fundamentally different from an employee bonus, and we will price the difference in dollars shortly.

**AUM (assets under management).** AUM is the total amount of capital being managed — by a PM (their book size), by a desk, or by a whole fund. For an independent fund, AUM is the single most important number in the business, because the management fee is a *percentage of AUM* and the fixed costs of running the fund are roughly constant. A 500-million-dollar fund and a 50-million-dollar fund can run nearly identical infrastructure; the larger one just has ten times the fee revenue to pay for it. Scale, in this business, is destiny.

**2-and-20.** This is the classic hedge-fund fee structure, and it is two fees, not one. The **management fee** is typically **2% of AUM per year**, charged regardless of performance — it is meant to cover the cost of running the business. The **performance fee** (or "carry," "incentive fee") is typically **20% of the profits** the fund generates, charged only when the fund makes money, and usually only above a "high-water mark" (you do not get paid twice for recovering losses you already charged on). So a fund with 200 million AUM that returns 10% in a year collects 2% × 200M = 4 million in management fee, plus 20% × (10% × 200M) = 20% × 20M = 4 million in performance fee. "2-and-20" is shorthand; in practice fees have compressed, and many funds charge less, but the *structure* — a flat fee on assets plus a slice of profits — is universal.

**Seeding (and the anchor investor).** A *seeder* is an investor who provides the initial capital and often the working-capital backing to launch a new fund, in exchange for a cut — frequently a share of the fund's *revenue* (a slice of the management and performance fees), and sometimes an equity stake in the management company itself. An *anchor investor* is similar: a large allocator who commits a big first check (the "anchor" of your initial AUM) in exchange for favorable terms — lower fees, capacity rights, transparency. Seeding solves the chicken-and-egg problem of launching: you cannot raise from cautious LPs without a track record at *your own firm*, and you cannot build that record without capital. The seeder breaks the loop, and charges for it.

**Spinning out.** A *spin-out* is when a team (or an individual) leaves an established firm to launch their own fund, usually built around a strategy and a track record they developed at the prior firm. Spin-outs are how a large fraction of new quant funds are born — and they are fraught, because the prior firm typically claims the IP and the track record, and binds the departing person with a **non-compete** and **garden leave** (a paid period where you cannot work, designed to let your edge and your information go stale). We will return to these as real costs.

**The infrastructure stack.** A trading strategy is the *top* of a tall stack of machinery, and the stack is the same whether you run a book inside a firm or a fund on your own — the difference is who pays for and operates it. The layers, bottom to top: **data** (market and alternative data feeds, licenses, history, cleaning), **research** (the backtest engine, compute, the signal library), **execution** (order routing, broker and exchange connectivity), **risk** (real-time limits and monitoring, the kill-switch), **operations** (trade reconciliation, prime broker, fund administration, accounting, investor reporting), and **compliance and legal** (registration, the fund's legal structure, audits, the regulator). Inside a firm, you rent the bottom five from the platform. On your own, you build, buy, or outsource every single one before a trade goes out.

**LPs and the GP.** A fund is legally structured as a partnership. The **general partner (GP)** is the manager — you, the founder, the entity that runs the fund and collects the fees. The **limited partners (LPs)** are the investors — pensions, endowments, funds-of-funds, family offices, wealthy individuals — who put in capital, take the investment returns, and have limited liability (they can lose their investment but are not on the hook for the fund's other obligations). When people talk about "raising money," they mean raising LP commitments.

With the vocabulary built, we can compare the two routes honestly. Start with the one most quants will actually be offered: the PM seat.

## The PM route: a book, a payout, a team, and a leash

The PM seat is the most common way a senior quant graduates into running money, because it is the *product* a multi-strategy pod shop sells. Firms like [Citadel](/blog/trading/quant-careers/owning-pnl-and-owning-research-the-accountability-ladder), Millennium, Point72, Balyasny, and ExodusPoint are structured as collections of semi-autonomous PMs ("pods"), each running a book, with the firm supplying capital, leverage, infrastructure, and a hard risk overlay. When you become a PM at one of these, you are not getting a promotion in the normal sense — you are being handed the keys to a unit of the business and told that your pay now equals your performance.

### What the PM seat actually is

A PM at a pod shop gets four things and gives up one. They get: a **capital line** (the firm allocates capital to their book), **leverage** (the firm's balance sheet lets them run a book several times larger than the raw capital), the **infrastructure stack** (data, execution, risk, ops, compliance — all provided by the platform), and the ability to **build a team** (hire analysts and junior PMs, paid by the firm, working on your book). What they give up is *autonomy over risk*: the firm's central risk team sets hard limits, and if the book breaches a drawdown threshold — often a stop-loss in the range of 5% to 10% of allocated capital — the firm can and will *cut the book down or shut it entirely*. This is the leash, and it is short.

The pod-shop model is, in one sentence, *the firm rents you everything except the edge, takes the platform risk, and pays you a slice of what your edge produces — but flattens you fast if you start losing*. It is a remarkably clean arrangement for someone with a real, attributable edge and no desire to run a business. You do the thing you are best at — find and trade an edge — and someone else runs the prime broker relationship, files with the regulator, and reconciles the trades.

### The payout

The PM payout is the number that draws people to the seat, so let us be precise about its shape. A PM is paid a **percentage of the book's net P&L** — net meaning *after the firm deducts the costs allocated to the book*: financing and leverage costs, data and platform fees, and the salaries and bonuses of the PM's own team. Reported payout rates cluster around **10% to 20%** of net P&L. The exact figure depends on the firm, the PM's tenure and leverage, and how much cost the firm loads onto the book. There is usually little or no guaranteed base relative to the payout — the seat is *eat what you kill*, with a modest draw against the year's payout.

The defining property is **linearity**. An employee's bonus is a discretionary number that a manager assigns, loosely correlated with contribution, smoothed across years, and capped by precedent and budget. A PM's payout is a *formula*: net P&L times a percentage. There is no committee deciding you "deserve" less than the formula in a great year (though there is plenty deciding the *percentage* when you negotiate the seat). This is why PMs with strong books out-earn nearly everyone else in the industry — and why a PM with a flat or negative year can take home close to nothing.

#### Worked example: the PM payout versus an employee bonus on the same P&L

Wei is trying to value the PM seat against staying a principal researcher, and the cleanest way is to compare what he takes home from the *same* P&L contribution under each arrangement.

Suppose Wei's strategies generate **50 million USD** of net P&L in a year — a strong but not absurd outcome for a mid-sized book.

As a **PM** with a 15% payout (the middle of the reported 10-20% range): his take-home is 15% × 50M = **7.5 million USD**. If the book instead makes 100 million, his payout is 15% × 100M = **15 million**. If it makes 25 million, he gets 15% × 25M = **3.75 million**. If it makes 10 million, 1.5 million. The payout is a straight line through the origin: double the P&L, double the pay.

As a **principal researcher** whose work contributed that same 50 million, his bonus is a *discretionary* number. It tracks his contribution, but it is smoothed, capped, and decided by a committee balancing the whole desk's budget. On a 50-million contribution a strong principal might be paid a bonus that, even in a great seat, tops out around **2 to 3 million** — and crucially, it *saturates*: if his work had contributed 100 million instead of 50, his bonus would not double to 6 million; it would creep up to maybe 3 million, because the firm reserves the linear upside for the PM who *owns* the book.

![A grouped bar chart comparing a PM payout at fifteen percent of net book P&L against an employee bonus on the same P&L across book sizes of ten, twenty-five, fifty, and one hundred million, showing the PM payout scaling linearly while the bonus flattens out](/imgs/blogs/from-senior-to-pm-or-starting-your-own-3.png)

The chart makes the asymmetry visceral. At a 10-million book the two numbers are close — 1.5 million payout versus a roughly 0.8-million bonus, not life-changing either way. But the lines *diverge*: at 100 million the PM takes 15 million while the employee tops out near 3 million. The PM payout is a slice of the upside; the bonus is a thank-you note with a ceiling.

*The reason to want the PM seat is not that the average year pays more — it is that the great year pays you in full instead of paying you a smoothed, capped fraction.*

### The team

Most PMs at scale do not trade alone — they run a small team, typically a handful of researchers, analysts, and execution specialists, hired and paid by the firm but working on the PM's book. This is the management dimension of the seat, and it is exactly the hybrid the [IC-versus-management fork post](/blog/trading/quant-careers/the-ic-vs-management-fork-staff-principal-pm-or-lead) describes: a PM is an owner of risk *and* a manager of people. The team's salaries are a cost charged against the book before the payout is computed, which means **building a team is a leveraged bet on your own P&L** — you pay for the analysts out of your net, and you only come out ahead if their work lifts the book's P&L by more than they cost. A PM who hires three analysts at, say, a fully-loaded 500k each is spending 1.5 million of pre-payout P&L; that team has to generate well over 1.5 million of incremental net P&L just to break even on the payout that the PM personally would have kept.

This changes the job. A principal researcher protects long blocks of deep work. A PM splits the day between *running the book* (the IC act) and *running the team* (the management act), and the better the book does, the more the team and the management grow. Some quants discover they love this. Others discover that the management half is a tax on the part they actually wanted, and that the seat they took to "run money" is half-spent in 1:1s and hiring loops.

### The leash

The part of the PM seat that recruiting decks underweight is the **risk leash**, and it is the single most important thing to understand before taking the seat. At a pod shop, the firm's central risk function sets hard limits on the book — gross and net exposure, concentration, factor exposures — and, critically, a **drawdown stop**: if the book loses more than a threshold (commonly somewhere in the 5-10% range of allocated capital, sometimes tighter) from its high-water mark, the firm *de-risks the book* — cuts the capital, forces positions down, and in the limit *shuts the pod and lets the PM go*.

This is rational from the firm's side — they are running dozens of pods on shared leverage and cannot let one blow a hole in the whole vehicle — but it has a brutal consequence for the PM: **the firm controls your survival, and a single bad stretch can end the seat even if your edge is real**. A strategy that would have recovered handsomely if held through a drawdown does not get the chance, because the leash cuts you at the bottom. This is why PM tenure at pod shops is often *short* — the up-or-out is fast and unsentimental — and why the [risk-discipline](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up) that keeps you *inside* the limits is not optional craft but existential. A PM who cannot size to stay off the stop does not get to find out whether their edge was real. The payout is linear in the P&L; the leash is what determines whether you survive long enough to collect it.

## Going independent: seed versus spin-out

The other door is the one Maya walked through: leave the firm, raise capital, and run your own fund. This is the route with the genuinely uncapped upside, and it is also the route that turns you from a person with an edge into a person running a *business* — a different, harder, and much riskier job. There are two main on-ramps.

### The spin-out

A **spin-out** is the most common origin story for a quant fund: a PM or a senior team leaves an established firm and launches their own vehicle around the strategy and track record they built there. The appeal is obvious — you stop earning a payout *slice* and start owning the *whole* fee stream. But the spin-out has three hard frictions that the inside-the-firm path does not:

**The IP and track-record problem.** The strategy you ran belongs to the firm, not to you. You generally cannot take the code, the signals, or the data. What you *can* sometimes take — and this is negotiated and litigated constantly — is an *attributable record of your performance*, a statement that "this person ran this book and produced this Sharpe over these years," which is the currency you need to raise money. Firms guard this fiercely.

**The non-compete and garden leave.** Most senior quant contracts include a non-compete (you cannot trade a competing strategy for a period) and garden leave (a paid period, often 3 to 12 months, where you are still employed but cannot work — designed to let your edge and your live market information go stale before you can use it). Garden leave is a real cost: it is a year of your prime earning life spent waiting, and your strategy's edge may decay while you sit out, especially if it was high-turnover. We will price this.

**The cold start.** You leave with a track record but *no infrastructure and no AUM*. Everything the firm provided — data, execution, risk, ops, compliance — you now build from zero, and you do it while you have no fee revenue yet.

### The seed

A **seeding** deal solves the cold start by bringing in an investor who provides the launch capital (and often working-capital backing for the business) in exchange for a cut of the economics — typically a meaningful share of the fund's *revenue* (a slice of both the management and performance fees) for a period of years, and sometimes an equity stake in the management company. Seeders are specialists (some firms exist purely to seed emerging managers) or large allocators making a bet on a manager early.

The trade is clear: the seeder takes a large chunk of your upside in exchange for making the launch *possible* and *survivable*. A common structure might give the seeder, say, 20-30% of the fund's revenue for the first several years, in exchange for an anchor commitment large enough to clear the break-even AUM and cover the launch costs. You give away a slice of the economics, but you get to exist — and an anchor from a credible seeder is itself a signal that helps you raise the *rest* of your AUM from other LPs.

### The capital and track-record threshold

The thing that gates *both* on-ramps is the same one that gates the PM seat: a **track record**. But for independence the bar is higher, because you are asking an outside investor — who has never managed you, never seen your daily attribution, and has many other managers to choose from — to trust you with their capital. They want to see, at minimum: a *multi-year, attributable, audited* record; a Sharpe that survives scrutiny; a strategy with clear *capacity* (it can absorb real AUM without the edge collapsing); and a coherent story for why the edge persists. The discipline behind that — honest out-of-sample testing, [evaluating a signal by IC, Sharpe, and turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research), the willingness to kill ideas that do not hold up — is exactly what the research series builds, and it is what an allocator's due-diligence team will probe relentlessly before wiring a dollar.

#### Worked example: the track-record and capital threshold to get seeded

Maya is trying to figure out whether she is *seedable* before she risks leaving her seat. Put yourself in the seeder's shoes and run the numbers from their side.

A seeder is considering anchoring Maya's fund with **50 million USD**. They will take **25% of the fund's revenue** for five years in exchange. For this to be worth their while, Maya's fund has to (a) survive, (b) attract more capital, and (c) perform. The seeder's due diligence demands:

- A **track record** of at least 3-4 years of attributable P&L. Maya has 4 years as a PM with an audited record: a Sharpe of ~1.8 on a 150-million book, net of costs. *This clears the bar* — a sub-1.0 Sharpe or a record under two years would likely not.
- **Capacity**: the seeder asks whether her edge survives at scale. Maya's strategy is mid-frequency and she estimates it absorbs up to ~300 million before the edge degrades. *This matters* — a strategy that only works on 20 million is not a fund, it is a hobby.
- A **break-even plan**: with a 50-million anchor and 2-and-20, the fund's first-year management fee is 2% × 50M = 1 million, plus a performance fee only if she makes money. Against ~3 million of running costs (next section), *the anchor alone does not clear break-even* — Maya has to raise another ~25-50 million in the first year to stop bleeding, and the seeder's anchor is the credibility that lets her do it.

The seeder runs their own expected value: if Maya raises to 200 million within two years and returns 10% annually, the fund's revenue is roughly 4M management + 4M performance = ~8 million a year, of which the seeder's 25% is ~2 million a year — a strong return on their backing. If Maya fails to raise past the anchor and closes in year two, the seeder loses their economics but their *capital* is largely intact (it was invested, not spent). The seeder is making a portfolio bet across many managers, knowing most will not scale.

*To get seeded you do not need to be a sure thing — you need an attributable multi-year record, a strategy with real capacity, and a credible path past break-even, because the seeder is pricing the probability that you become a real business, not the certainty.*

The requirements split cleanly across the two routes, and the picture below lays them side by side.

![A matrix comparing the PM seat against going independent across track record, capital, the infrastructure stack, the team, and who carries the risk, showing that a PM rents most of it from the firm while an independent must supply everything](/imgs/blogs/from-senior-to-pm-or-starting-your-own-2.png)

Read down the *capital* and *infra* rows and the real difference jumps out: the PM seat *rents* you capital, infrastructure, and the ability to hire, in exchange for a payout slice and a short leash. Independence makes you *supply* all of it, in exchange for owning the whole stream and carrying the whole risk. The track-record row is the only one where the two routes agree — both demand it, and independence demands it be *portable*. The bottom row is the one people underestimate: a PM carries *market* risk on their book; an independent carries market risk *plus* the entire *business* risk of a company that can fail even when the strategy works.

## The infrastructure and cost stack: a fund is a business

The single biggest misconception about going independent is that the hard part is the strategy. The strategy is the part you already have — it is why anyone would seed you. The hard part is that **a fund is a business**, and a business has a cost structure that exists whether or not you trade well. Here is the stack you have to stand up, and what each layer costs.

![A six-layer stack showing the infrastructure required to run money, from data at the bottom through research, execution, risk, operations, and compliance and legal at the top, with the lower layers marked as cost and friction and the top layer as a non-negotiable regulatory requirement](/imgs/blogs/from-senior-to-pm-or-starting-your-own-5.png)

Walk it from the bottom:

- **Data.** Market data feeds, history, and any alternative data your strategy needs — licensed, cleaned, and stored. This ranges from tens of thousands of dollars a year for basic market data to *millions* for premium alternative datasets. Inside a firm this is a shared, sunk cost; on your own it is a line item you negotiate and pay.
- **Research.** The backtest engine, the compute to run it, the signal library — the machinery that turns ideas into strategies, described in [the research workflow](/blog/trading/quant-careers/the-research-workflow-in-production-from-idea-to-live-signal). You either rebuild it (months of engineering you are not getting paid for yet) or rent a platform.
- **Execution.** Order routing, broker and exchange connectivity, and the systems that turn a target portfolio into fills without leaking the alpha to the market through bad execution. For a high-frequency strategy this is a deep technical lift; for a mid-frequency one you can lean more on a broker's algos.
- **Risk.** Real-time exposure and limit monitoring, and the **kill-switch** that flattens you before a bug or a bad day turns into ruin. Inside a pod shop the firm runs this *for* you (it is the leash); on your own *you* are the risk team, which is a conflict of interest you have to manage with hard, pre-committed rules — the [risk discipline](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up) that keeps you alive.
- **Operations.** Trade reconciliation, the prime broker relationship, fund administration, accounting, treasury, and investor reporting. This is unglamorous and constant, and most small funds *outsource* it to a fund administrator — another fee.
- **Compliance and legal.** Registration (with the SEC and/or other regulators depending on size and jurisdiction), the fund's legal structure (the GP/LP entities, the offering documents), audits, KYC/AML, and ongoing regulatory filings. This is *non-negotiable* — you cannot legally run other people's money without it — and it is a meaningful recurring cost in legal and compliance fees.

Add it up and a small systematic shop's fixed cost runs in the rough range of **2 to 4 million dollars a year**, before you pay yourself a dime, and largely *independent of AUM*. That last property is the entire economic story of the fund business, and it is why scale is destiny.

#### Worked example: the 2-and-20 minus running costs, and what actually reaches the founder

Maya wants to know, concretely, how much of the famous "2-and-20" actually lands in her pocket at different fund sizes. Assume a 10% gross return year (so the performance fee is meaningful) and a running cost of **3 million a year** (mid-range for a small systematic shop).

The fee revenue at AUM *A* (in millions) is: management fee 2% × *A*, plus performance fee 20% × (10% × *A*) = 2% × *A*. So gross fees are **4% of AUM** in a 10%-return year. The founder's take is gross fees minus the 3-million running cost.

- At **50 million AUM**: gross fees = 4% × 50M = 2 million. Minus 3 million of cost = **−1 million**. *Maya is paying out of pocket to keep the fund alive.*
- At **75 million AUM**: gross fees = 4% × 75M = 3 million. Minus cost = **0**. This is **break-even** — the fee revenue exactly covers the cost of running the business, and the founder earns nothing.
- At **100 million AUM**: gross fees = 4 million. Minus cost = **1 million** to the founder.
- At **250 million AUM**: gross fees = 10 million. Minus cost = **7 million**.
- At **500 million AUM**: gross fees = 20 million. Minus cost = **17 million**.

![A line chart of fund economics showing gross fees from a 2-and-20 structure rising with AUM, a flat running cost line of about three million per year, and the founder take crossing from negative to positive at a break-even AUM of about seventy-five million](/imgs/blogs/from-senior-to-pm-or-starting-your-own-4.png)

The chart shows the brutal nonlinearity. Below ~75 million AUM the founder is *underwater* — the 2% management fee cannot cover the fixed costs, and the founder is funding the business out of their own pocket or relying entirely on the performance fee, which only exists in a good year. The red zone on the left is where most launches live and die. Above break-even the line climbs steeply, because every additional dollar of AUM adds fee revenue against a nearly fixed cost base. At 500 million, the founder keeps 17 million of a 20-million gross — the marginal AUM is almost pure profit.

This is why **the first job of a new fund is not to trade well — it is to raise to scale before the lean years bankrupt the business**, and why a seeder's anchor that clears or approaches break-even is worth giving up 25% of revenue for.

*The 2-and-20 sounds like a fortune, but on a small fund the 2% does not even cover the lights; the fund business is a scale business, and below break-even AUM you are paying for the privilege of running money.*

## The economics and risk-reward, side by side

Now we can put the two routes' economics next to each other and Wei's decision into focus. Both routes pay you as a function of P&L, but the *shape* of the bet is completely different.

The **PM seat** is a *levered, capped-downside, short-fuse* bet. Upside: a 10-20% payout on a book the firm capitalizes and levers, with no business risk and no fundraising — in a great year a PM can take home eight figures. Downside: a flat or losing year pays close to nothing, and the risk leash can end the seat in a single bad stretch. You keep your *time* (someone else runs the business) but you do not keep the *whole stream* (the firm takes the rest of the P&L), and you do not control your own survival (the firm's risk team does).

**Independence** is an *uncapped-upside, real-downside, slow-burn* bet. Upside: you own the *entire* fee stream — every dollar above the costs is yours, and a fund that scales to billions makes its founder generationally wealthy. Downside: you fund the lean years yourself, a single drawdown can trigger LP redemptions that shrink your AUM below break-even, you carry the business risk on top of the market risk, and you have given up the safe, large PM comp you could have had instead (the opportunity cost is enormous). And the base rate is unforgiving — most small launches do not scale and close within a few years.

![A before-after comparison of the risk-reward of going independent, with the downside on the left listing self-funded lean years, redemption risk, the high closure rate of small launches, carrying every risk, and the foregone PM comp, and the upside on the right listing owning the whole fee stream, compounding, the survivors reaching scale, full control, and uncapped pay](/imgs/blogs/from-senior-to-pm-or-starting-your-own-7.png)

The before-after picture is the honest version of the pitch. The right column — owning the whole stream, compounding, full control, uncapped pay — is real and is why people do it. The left column — self-funding the lean years, redemption risk, the high closure rate, carrying every risk, the foregone safe comp — is equally real and is the part that does not make it into the founder's interview after they have made it. *Both columns are true at once.* Independence is not a better deal than the PM seat; it is a *more asymmetric* deal — a bigger maximum, a worse and more probable failure mode, and you carry all of it.

#### Worked example: Wei weighs the PM seat against staying a principal researcher

Wei finally sits down and does the expected-value math, the same EV-under-uncertainty thinking the whole series runs on — and the same logic the [Kelly criterion post](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) applies to position sizing, applied here to a career bet.

**Option A: stay a principal researcher.** Comp is roughly stable: call it **1.8 million** in a typical year, with low variance — a bad year is maybe 1.2 million, a great year 2.5 million. Expected value ≈ **1.8 million/year**, with small risk. His edge does not depend on a leash; his seat is secure.

**Option B: take the PM seat**, 15% payout on a book he expects to run at ~50 million of net P&L in a good year. He estimates the distribution of his book's annual P&L:

- 40% chance of a *good* year (~50M P&L → 15% × 50M = 7.5M payout)
- 30% chance of an *okay* year (~20M P&L → 3M payout)
- 20% chance of a *flat* year (~5M P&L → 0.75M payout)
- 10% chance of a *bad* year (negative P&L → ~0 payout, and a real chance the leash ends the seat)

Expected payout = 0.40 × 7.5M + 0.30 × 3M + 0.20 × 0.75M + 0.10 × 0 = 3.0M + 0.9M + 0.15M + 0 = **4.05 million/year** in expectation.

On pure EV, the PM seat wins handily: **~4.05 million versus ~1.8 million**. But Wei has to weight three things the raw EV hides. First, **variance**: the PM number is a wide distribution with a real zero in it, while the researcher number is tight — and a bad PM year is not just low pay, it is a 10% chance the *seat ends*. Second, **survival**: if the leash cuts him in a bad first year, he loses the seat *and* the option on all the good years behind it — the EV above assumes he survives to keep playing, which the [risk-of-ruin logic](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up) says he should not assume. Third, **what he actually wants to do all day** — the PM seat is half management, and Wei is not sure he wants the team half.

His decision rule is the one the senior's-edge logic recommends: *the EV favors the seat by more than 2×, the downside is survivable (he has savings and a strong outside option as a principal researcher), and the bet is repeatable if he sizes his risk to stay off the leash* — so he takes the seat, but negotiates hard for a higher payout percentage and a slightly looser drawdown limit, because those two terms are what convert the favorable EV into survivable EV.

*A career bet is sized like any other: take it when the EV is favorable, the downside is survivable, and you can keep playing — and negotiate the terms that protect your survival, not just the ones that raise your upside.*

## The survivorship reality of fund launches

Everything above prices the *economics*. This section prices the *base rate*, because it is the number the recruiting deck and the founder's victory-lap interview both omit, and it is the most important input to the independence decision.

Fund launches are subject to severe **survivorship bias**. The funds you read about are the ones that scaled — the ones a famous PM spun out and grew to ten billion. You do not read about the far larger number that launched, never crossed break-even AUM, ran for two or three years on the founder's savings and a shrinking anchor, and then closed quietly, returned the LPs' capital, and disappeared. Industry data on fund formation and closure consistently shows that **a large fraction of new fund launches close within their first few years** — the launch-and-liquidation churn is high, and the median new fund never reaches the scale at which the economics work. The headline survivors are exactly that: survivors.

The mechanism of failure is rarely "the strategy stopped working." It is more often *business* failure layered on top of *market* friction:

- **The lean-years trap.** The fund launches below break-even AUM, burns the founder's capital and the anchor for a year or two, and never raises enough to become self-funding. The strategy might be fine; the *business* runs out of runway.
- **The redemption spiral.** A single bad drawdown year — even a recoverable one — triggers LP redemptions. Redemptions shrink AUM, which cuts fee revenue below the fixed-cost line, which forces cost-cutting that hurts performance, which triggers more redemptions. A pod shop's leash cuts you fast; an LP base's loss of confidence cuts you slowly but just as fatally.
- **The capacity wall.** A strategy with a real edge at 50 million has *no edge* at 500 million because the trades move the market against the fund. A founder who raises past their strategy's capacity watches their Sharpe collapse precisely as they finally reach the AUM that pays — a cruel inversion.
- **The single-person-business risk.** The founder is the edge, the salesperson, the risk manager, and the CEO. Burnout, a key-person departure, an operational error, or a compliance failure can end the fund independent of performance.

![A horizontal timeline of the spin-out path running from building an attributable record over years one to five, through the decision to leave and the non-compete and garden leave, through finding a seeder and building the stack, to a lean small-AUM launch, and finally to crossing break-even AUM and scaling over the following years](/imgs/blogs/from-senior-to-pm-or-starting-your-own-6.png)

The timeline is the optimistic path — the one that works — and even it is *years* long: a multi-year record before you can leave, a garden-leave gap, a pre-launch build, a lean launch, and only *then*, two to four years after launch, the scale that makes the economics work. The survivorship reality is that most launches do not make it to the green node on the right. None of this means do not do it. It means *price the base rate honestly* — go in knowing that the modal outcome is a quiet close, that the asymmetric upside is real but improbable, and that the foregone PM comp is the true cost of the lottery ticket.

## How to know you're ready

Set the romance aside and use a checklist. You are ready to run money — by either route — when you can answer *yes*, honestly, to most of these:

- **You have a real, attributable, multi-year track record.** Not "I contributed to a strategy that made money" — *this book, this P&L, this Sharpe, over these years, and someone other than me would attest to it.* If your record is two years or its attribution is fuzzy, you are not there yet. Build the record first; it is the only currency that buys either seat.
- **Your edge is yours, not the platform's.** Be brutally honest about how much of your performance came from the firm's data, execution, leverage, and risk overlay versus your own decisions. An edge that only exists *inside* the firm's machinery does not travel — and a PM seat at a *new* firm or an independent launch will expose that fast.
- **Your strategy has capacity.** It must absorb real AUM (for independence) or a real capital line (for a PM seat) without the edge collapsing. A 20-million strategy is a great side project and a terrible fund.
- **You can size to survive the leash (PM) or the redemption cliff (independent).** The [risk discipline](/blog/trading/quant-careers/risk-discipline-and-not-blowing-up) that keeps you off the drawdown stop is not optional. If you cannot run your strategy *small enough to survive a bad stretch*, you will not last long enough to collect the good years.
- **You can stomach pay that equals performance.** From here your comp is a function of P&L, with a real zero in the distribution. If a year of near-zero pay would break you financially or psychologically, the safer principal-IC seat is the better choice, and there is no shame in it.
- **(For independence only) You actually want to run a business.** Fundraising, LP relations, hiring, compliance, ops, the prime-broker call — this is the *majority* of an independent founder's job, and almost none of it is the research you fell in love with. If that list fills you with dread rather than energy, take the PM seat and let someone else run the business.

If you cannot yet say yes to the first three, the answer is not "give up" — it is "go build the track record," which is what the entire arc from [your first 90 days](/blog/trading/quant-careers/your-first-90-days-ramping-without-blowing-up) through [owning P&L and owning research](/blog/trading/quant-careers/owning-pnl-and-owning-research-the-accountability-ladder) is for.

## Common misconceptions

**"A good track record is enough."** It is necessary, not sufficient. A great Sharpe on a small, high-capacity-doubtful strategy will not raise a fund, because allocators price *capacity* and *persistence*, not just the headline number — and a record that is not *attributable and portable* (it lived inside the firm's platform and IP) cannot leave with you at all. The record is the ticket to the conversation; the *business case* is what closes the raise.

**"You keep all the profit."** Only the independent founder keeps the *fee stream*, and even they share it — with the seeder (a revenue cut for years), with their team (salaries before they pay themselves), and with the cost stack (the 2-4 million a year of running costs). A *PM* keeps only their *payout percentage* — 10-20% of net P&L — the firm keeps the rest as the price of the capital, leverage, and infrastructure it provides. Nobody keeps all the profit. The question is always *which slice* and *at what risk*.

**"Independence equals freedom."** This is the most expensive misconception in the section. An independent founder trades the firm's leash for a *harsher* set of masters: the LPs (who can redeem and end you), the regulator (who can shut you down), the cost structure (which runs whether or not you make money), and the fundraising treadmill (which never stops). A PM has a short leash but spends the day *trading*; an independent founder has no leash but spends the day *running a company* and answering to investors. Freedom from the firm is not freedom — it is a different, often heavier, set of obligations.

**"Bigger AUM is always better."** Only up to your strategy's *capacity*. Raising past the AUM your edge can absorb *destroys* the edge — the trades move the market, slippage eats the alpha, and the Sharpe collapses precisely as the fee revenue peaks. The discipline of *turning capital away* — closing the fund to new money at the capacity limit — is one of the hardest and most important things a founder does, and the funds that survive longest are often the ones that refused to grow past their edge. More AUM is more *fee revenue* and, past capacity, *less return*; those two facts fight, and the second one is what keeps LPs.

**"The PM seat is just a senior trader with a bigger title."** No — it is a *different economic and managerial role*. A senior trader is paid a (large, smoothed, capped) bonus on their contribution and manages mostly themselves; a PM is paid a *linear payout* on a book they own, runs a *team*, carries a *risk leash* that can end the seat, and is the unit of the firm's whole P&L. The title change is small; the change in how you are paid and what can end you is total.

## How it plays out in the real world

The named structures behind this post are concrete and reportable. **Multi-strategy pod shops** — Citadel, Millennium, Point72, Balyasny, ExodusPoint — are *built* on the PM-with-a-book model: the firm is a portfolio of pods, each a PM running a book on the firm's capital and platform, paid a payout, and governed by a hard risk overlay with fast de-risking on drawdowns. This is why pod-shop PM seats are simultaneously the highest-paying employed seats in the industry and among the *shortest-tenured* — the up-or-out is real, and the leash is short. The [pod-shop archetype post](/blog/trading/quant-careers/the-firm-archetypes-prop-vs-hft-vs-pod-shop-vs-systematic-fund) walks the structure in detail.

The **independent launch** path is equally real and equally documented: the history of the industry is full of funds spun out by PMs who left a platform — D. E. Shaw and Two Sigma alumni who started their own systematic shops, traders who left the big pods to run their own vehicles. Seeding firms exist specifically to back emerging managers, and the standard deal — capital plus working-capital backing in exchange for a multi-year revenue share — is a well-worn template. The fee structure (2-and-20, compressed in practice), the GP/LP partnership form, the high-water mark, the running-cost stack, and the survivorship churn of launches are all standard features of the hedge-fund business as widely reported across industry data, fund-formation surveys, and the trade press.

The honest summary of the real world is the one Wei arrives at and Maya is living: **the PM seat is the higher-EV, lower-variance, less-free route to running money — you keep your time and your survival is the firm's call; independence is the uncapped, higher-variance, all-the-risk route — you keep the whole stream and carry the whole business.** Both are real top-of-ladder destinations. Neither is the "win." The win is choosing the one whose *shape of bet* matches your edge, your risk tolerance, and the day you actually want — and going in with the base rates priced honestly rather than the survivor's story. The asymmetric upside is real. So is the part nobody posts about.

## When this matters / Further reading

This decision matters the moment you have something rare: a *real, attributable track record*. Before that, the path is to build it — the entire arc from ramping without blowing up through owning your own P&L. After that, the fork in this post is the highest-stakes choice in the career, and the worst way to make it is to drift into the seat someone offers you without pricing the EV, the variance, and the survival odds yourself.

Read these next:

- [The IC-versus-management fork: staff, principal, PM, or lead](/blog/trading/quant-careers/the-ic-vs-management-fork-staff-principal-pm-or-lead) — the level *below* this one, where you first choose between going deep as an IC and moving toward the PM-style hybrid; this post is what the PM prong leads to.
- [Owning P&L and owning research: the accountability ladder](/blog/trading/quant-careers/owning-pnl-and-owning-research-the-accountability-ladder) — how attribution and accountability build the track record that buys both routes here.
- [The firm archetypes: prop vs HFT vs pod shop vs systematic fund](/blog/trading/quant-careers/the-firm-archetypes-prop-vs-hft-vs-pod-shop-vs-systematic-fund) — the structures behind the PM seat (the pod shop) and the independent launch (the systematic fund).
- [Evaluating alpha signals: IC, Sharpe, turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research) — how an allocator's due-diligence team measures the track record you are selling, and what makes a Sharpe credible.
- [The Kelly criterion and sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) — the sizing logic behind Wei's career bet: take a favorable EV only at a size that keeps you in the game.

On the real numbers: comp, payout, and fund-economics figures here are illustrative midpoints drawn from the series data appendix (levels.fyi, efinancialcareers, Glassdoor, and the 2026 "Young & Calculated" quant-pay survey), the standard hedge-fund fee structure, and widely reported fund-formation and closure data, as of 2026. Treat every dollar figure as a round, conditional estimate — payouts, fees, and survival rates vary enormously by firm, strategy, and year, and the headline outcomes are, as always in this industry, the survivors.
