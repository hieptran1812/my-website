---
title: "The Liquidity Cycle and Asset Prices: The Tide That Lifts and Sinks Everything"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Asset prices are partly a function of how much money is sloshing around chasing them. This is the deep-dive on the liquidity cycle — central-bank balance sheets, net liquidity, and financial conditions — and why a rising liquidity tide lifts almost every asset while a draining one sinks them, running as a second clock next to the business cycle."
tags: ["asset-allocation", "cross-asset", "liquidity", "net-liquidity", "quantitative-easing", "quantitative-tightening", "financial-conditions", "central-banks", "real-yields", "global-liquidity", "risk-on-risk-off", "portfolio-construction"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Asset prices are not set only by earnings and growth. They are also set by how much money is sloshing around the financial system looking for a home. That pool of money — *liquidity* — has its own cycle, driven by central-bank balance sheets, the cost of money, and financial conditions. When the tide is coming in (QE, rate cuts, easy conditions) almost every asset rises together; when it goes out (QT, rate hikes, tight conditions) almost everything falls together.
>
> - **Liquidity is a second clock running next to the business cycle — and sometimes the more powerful one.** The 2020-21 "everything bull" and the 2022 "everything bust" were liquidity events, not earnings events.
> - **The trader's proxy is net liquidity = Fed assets − reverse repo (RRP) − the Treasury account (TGA).** It tracked the S&P 500 closely from 2021 to 2023: it drained as RRP and TGA soaked up reserves in 2022, and the market drained with it.
> - **More liquidity lowers discount rates, lifts risk appetite, and floods cash toward assets — so multiples expand, credit spreads compress, and correlations go up.** Draining liquidity runs every step in reverse.
> - **The one fact to remember:** the Fed created roughly **\$4.7 trillion** of new reserves between March 2020 and April 2022; over that span the S&P 500 roughly doubled and Bitcoin rose about eightfold. When the tide went out in 2022, the same highest-beta assets fell hardest — Bitcoin **−64%**.

In 2021, almost nothing you could buy went down. A boring index fund of US stocks returned **+28.7%**. A house in most of the country jumped double digits. Gold held its ground. Bitcoin, having already tripled in 2020, climbed another **60%**. Even a meme stock or a profitless software company with no earnings could quadruple. It felt, to a lot of people, like a permanent new normal where every chart went up and to the right and the only mistake was sitting in cash.

Then, in 2022, it all reversed at once — and again, almost nothing worked. The same US stock index fell **−18.1%**. Bonds, the supposedly safe ballast, had their worst year in over a century at **−13.0%**. Bitcoin gave back **−64%**. Long-duration tech stocks fell **30%** or more. The assets that had risen the most fell the hardest, and they all fell *together*, which is exactly what is not supposed to happen if each asset is priced on its own separate story.

What changed between those two years was not the economy — the US economy was actually *growing* through most of 2022. What changed was the tide. In 2020-21 the world's central banks were flooding the financial system with money; in 2022 they began draining it back out. This is the story of the **liquidity cycle**: the rise and fall of how much money is sloshing around chasing assets, and why that tide lifts and sinks almost everything at once. The figure below is the mental model we will build for the whole post.

![Rising liquidity lifts every asset at once and draining liquidity sinks them](/imgs/blogs/the-liquidity-cycle-and-asset-prices-1.png)

The diagram above is the thesis in one picture. When the liquidity tide comes in — the green side — stocks, gold, and especially the highest-beta assets like Bitcoin float up together. When it goes out — the red side — the same assets sink together, and the ones that floated highest sink fastest. The mechanism in the middle is what we will spend most of the post on: more money in the system lowers discount rates, lifts risk appetite, and leaves cash hunting for a home, so multiples expand and spreads compress. The fast read at the bottom — the dollar and real yields — is the dashboard a working allocator actually watches. By the end you will be able to look at a single chart of net liquidity and know whether the wind is at the back of risk assets or in their face.

This is the deep-dive on liquidity in the *Cross-Asset Playbook* series. In the companion post on [the business cycle and the investment clock](/blog/trading/cross-asset/the-business-cycle-and-the-investment-clock) we treated growth and inflation as the clock that decides which asset to own. Liquidity is the *second* clock. Sometimes the two agree; sometimes liquidity overrules the business cycle entirely. An allocator who watches only one of them is reading half the dashboard. We build heavily on the macro posts about [what liquidity means](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders) and [the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide); here the focus is narrower and more practical — what the liquidity cycle *does to cross-asset prices*, and how to position for it.

## Foundations: what "liquidity" actually means here

The word "liquidity" is overloaded — it means three different things in finance, and people switch between them without warning. Before we can talk about a liquidity *cycle*, we have to be precise about which liquidity we mean, because the cross-asset tide is driven by one specific kind.

### The three meanings of "liquidity"

The first meaning is **market liquidity**: how easily you can buy or sell a *specific* asset without moving its price. A blue-chip stock you can sell instantly at the quoted price is "liquid"; a vacation cabin that takes nine months to sell is "illiquid." That is a property of an individual asset, and it is not what this post is about.

The second meaning is **funding liquidity**: how easily a bank, a hedge fund, or a company can borrow the cash it needs to fund its positions and operations. When funding liquidity dries up — when nobody will lend overnight — even healthy institutions can be forced to sell assets to raise cash. That is closer to our topic, and it is covered in depth in the macro post on [what liquidity means for funding](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders).

The third meaning — the one that drives the cross-asset tide — is **system liquidity**, sometimes called *monetary liquidity* or just "liquidity" by traders. This is the total pool of money and money-like assets sloshing around the financial system, set mostly by the central bank. It is the answer to a simple question: *how much cash is out there looking for a home?* When that pool grows, more money is competing to buy a roughly fixed set of assets, and prices rise. When it shrinks, money is pulled out of the competition, and prices fall. That is the liquidity we mean for the rest of this post.

Let us define the building blocks of system liquidity from zero.

### The central-bank balance sheet

A central bank — the US Federal Reserve, the European Central Bank, the Bank of Japan — has a *balance sheet* just like any other institution: a list of what it owns (assets) and what it owes (liabilities). What makes a central bank special is that it can create money, in the form of *bank reserves*, simply by typing the figure into existence. It uses that power to buy assets.

A **bank reserve** is electronic money that commercial banks hold in their account *at the central bank*. It is the most basic, most liquid money there is — the cash that banks use to settle with each other. When the Fed buys a \$1,000,000 Treasury bond from a bank, it does not hand over a suitcase of paper; it simply credits the bank's reserve account with \$1,000,000 of newly created reserves. The Fed's *assets* go up by \$1,000,000 (it now owns the bond), and its *liabilities* go up by \$1,000,000 (it now owes the bank that reserve balance). New money has entered the system.

So the size of the Fed's balance sheet — total assets, the line the data world calls **WALCL** — is a direct readout of how much base money the Fed has pumped into the financial system. When that line is rising, the Fed is creating reserves; when it is falling, it is destroying them. The figure below shows that line from 2019 to 2025, and you can read the entire liquidity cycle of the era off a single curve.

![Fed balance sheet total assets rose from 3.8 to 8.96 trillion then drained to 6.66 trillion](/imgs/blogs/the-liquidity-cycle-and-asset-prices-2.png)

The shape tells the whole story. The Fed's assets sat near **\$3.80 trillion** in September 2019. They had crept to **\$4.31 trillion** by March 2020 — and then, as the pandemic hit, they *exploded*: to **\$7.09 trillion** by June 2020, **\$7.36 trillion** by year-end, **\$8.76 trillion** by the end of 2021, and a peak of **\$8.96 trillion** in April 2022. That is the green QE ramp. Then the curve turns: **\$8.55 trillion** by the end of 2022, **\$7.68 trillion** by the end of 2023, **\$6.87 trillion** by the end of 2024, **\$6.66 trillion** by mid-2025. That is the red QT drain. The single most important macro chart of the decade is, at its core, the size of one institution's balance sheet — because that line is the master valve on system liquidity.

### Bank reserves: the liquidity that actually reaches markets

There is a subtlety worth nailing down, because it is where a lot of confusion lives. The Fed's *total assets* are the headline, but the part of the balance sheet that matters most for markets is the **reserves** side — the electronic money banks hold at the Fed. Reserves are the raw fuel of system liquidity, and they do not move one-for-one with the balance sheet, because the RRP and TGA (which we meet properly in the next section) can absorb them.

The curated reserve data tells the story cleanly. Bank reserves sat at just **\$1.47 trillion** in September 2019. QE then more than doubled them to **\$3.16 trillion** by the end of 2020 and **\$4.19 trillion** by the end of 2021 — the peak of the flood. Then they *fell*, to **\$3.03 trillion** by the end of 2022, even though the Fed's balance sheet had barely shrunk by then. Where did the reserves go? They were absorbed by the parking lots — the RRP filling up — rather than destroyed by QT. By the end of 2023 reserves had partly recovered to **\$3.49 trillion** as the RRP drained back out, and stood near **\$3.27 trillion** at the end of 2024.

The takeaway is that **reserves, not headline assets, are the cleanest measure of how much base money is actually in the banking system** — and the gap between the two (the parking lots) is exactly what net liquidity, our next topic, is built to capture. When reserves are abundant, funding is easy and risk assets have a tailwind; when reserves grow scarce, funding stress can erupt suddenly, as it did in the September 2019 repo spike, when reserves had been allowed to fall too far.

### QE and QT: opening and closing the valve

The two levers that move the balance sheet have names.

**Quantitative easing (QE)** is the central bank *buying* assets — usually government bonds and mortgage securities — with newly created reserves. The point is to push money into the system: to lower long-term interest rates, to flood banks with reserves, and to nudge investors out of safe bonds and into riskier assets. QE is the tide coming in. (The macro deep-dive on [quantitative easing](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders) covers the plumbing; here we care about its effect on prices.)

**Quantitative tightening (QT)** is the reverse: the central bank *shrinking* its balance sheet, usually by letting bonds mature without replacing them. As a bond it holds matures, the Treasury pays it back, reserves are extinguished, and the balance sheet falls. QT pulls money back out of the system. It is the tide going out.

The key insight is that QE and QT change the *quantity* of money in the system, which is a different lever from the *price* of money. The price of money is the interest rate, set by the central bank's policy rate. In 2020-21 the Fed pulled *both* easy levers at once: it cut the policy rate to near zero (cheap money) *and* ran QE (lots of money). In 2022 it slammed both the other way: it hiked rates from near zero to over 5% (expensive money) *and* ran QT (less money). When both levers move together, the liquidity tide is at its most powerful — which is exactly why 2020-21 and 2022 were such extreme, everything-moves-together years.

### Financial conditions: the full dashboard

Liquidity does not reach asset prices through the balance sheet alone. It reaches them through **financial conditions** — a broad measure of how easy or hard it is to get and use money across the whole economy. Financial conditions bundle together several dials:

- **Interest rates** — the policy rate and the level of bond yields. Lower rates = easier conditions.
- **Credit spreads** — the extra yield a risky borrower pays over a safe one. Tight spreads = easy conditions, because risky borrowers can fund cheaply.
- **The dollar** — a strong dollar tightens conditions for the whole world. A weak dollar loosens them.
- **Equity prices** — rising stocks make households and firms feel richer and more willing to spend and borrow, which loosens conditions in a self-reinforcing loop.

When financial conditions are *easy* — low rates, tight spreads, a soft dollar, rising stocks — money is cheap and abundant, and asset prices have a tailwind. When they are *tight* — high rates, wide spreads, a strong dollar, falling stocks — money is dear and scarce, and asset prices have a headwind. Liquidity and financial conditions are two views of the same underlying thing: *how much money is available, and at what cost, to chase assets.*

#### Worked example: how one bond purchase becomes new system liquidity

Let us trace a single QE purchase to see how a balance-sheet entry turns into money chasing assets. Suppose the Fed buys a \$10,000 Treasury bond from a pension fund.

Step one: the Fed creates \$10,000 of new reserves and credits the pension fund's bank with them. The Fed now owns the bond; the bank's reserve account is \$10,000 fatter. No money was taken from anyone — it was created.

Step two: the pension fund no longer has the bond, but it has \$10,000 of fresh cash. It did not want cash; it wanted a yielding asset. With the safe bond gone and safe yields pushed down by the Fed's buying, the fund reaches for the next-best yield — a corporate bond, or a dividend stock. It buys \$10,000 of corporate bonds.

Step three: the company that sold those corporate bonds now has \$10,000 of cash, which *it* puts to work — and so on. The single \$10,000 of created reserves does not sit still; it ripples outward, with each holder reaching a little further out the risk curve because safe yields are too low. Multiply this by the **\$4.7 trillion** the Fed actually created between 2020 and 2022, and you have a tidal wave of cash, all of it nudged toward riskier assets. The intuition: **QE does not just add money — it pushes that money out the risk curve, which is why risk assets rise the most.**

## Net liquidity: the trader's proxy

The Fed's balance sheet is the master valve, but it is not the whole story of how much liquidity actually reaches the markets. Two other government accounts can *absorb* reserves even while the Fed's balance sheet is unchanged — and accounting for them gives traders a sharper proxy called **net liquidity**.

### The formula, in plain English

The most popular trader's gauge of system liquidity is:

$$
\text{Net liquidity} = \text{Fed assets} - \text{Reverse repo (RRP)} - \text{Treasury account (TGA)}
$$

where each term is a dollar amount on or near the Fed's balance sheet. The idea is that Fed assets *create* reserves, while the RRP and the TGA *park* reserves where they can no longer chase assets. Subtracting the two parking lots from the total gives the reserves that are actually free to slosh around the financial system. Let us define the two subtractions from zero.

**The reverse repo facility (RRP)** is a parking lot the Fed runs for money-market funds and others. A money-market fund with cash it cannot otherwise invest safely can lend it to the Fed overnight in exchange for a bond as collateral, earning a set rate. When \$1 of cash goes into the RRP, it leaves the open financial system and sits inertly at the Fed until the next morning. A *rising* RRP balance means money is being parked rather than chasing assets — a drain on effective liquidity, even though the Fed's total balance sheet has not changed.

**The Treasury General Account (TGA)** is the US government's checking account at the Fed. When the Treasury raises money — by issuing bonds or collecting taxes — the cash lands in the TGA and is, for the moment, out of the private financial system. A *rising* TGA balance drains reserves; a *falling* TGA (the government spending the money back out) returns reserves to the system. So the TGA is a second parking lot that can soak up or release liquidity independently of what the Fed is doing.

The point of net liquidity is to capture all three forces at once: the Fed adding reserves (Fed assets), the money-market funds parking them (RRP), and the Treasury sequestering them (TGA). It is a rough, much-debated proxy — but a useful one, because it tracked the stock market remarkably closely through 2021-2023.

### Why it tracked the S&P so closely

The figure below shows the net-liquidity proxy computed from the curated balance-sheet data, and the shape is the point: it peaked around the start of 2022 and *drained* through that year, exactly when the S&P 500 was falling **−18.1%**.

![Net liquidity computed from Fed assets minus RRP minus TGA drained through 2022](/imgs/blogs/the-liquidity-cycle-and-asset-prices-3.png)

Walk the chart. At the end of 2021, net liquidity was high — the Fed's balance sheet was near its peak and the parking lots were not yet full. Through 2022, two things drained it. First, the Fed began QT, slowly shrinking the top line. Second, and faster, the RRP *exploded*: as the Fed hiked rates, money-market funds had a flood of cash and nowhere safe to put it, so they parked it at the Fed. The RRP went from **\$0.79 trillion** in mid-2021 to **\$1.60 trillion** at the end of 2021, **\$2.30 trillion** by mid-2022, and a peak of **\$2.55 trillion** at the end of 2022. That is \$2.55 trillion of cash pulled out of the asset-chasing pool and frozen at the Fed. Net liquidity fell, and the S&P fell with it.

Then the process reversed. Through 2023 and 2024 the RRP *emptied out* — from \$2.55 trillion at the end of 2022 down to **\$1.02 trillion** at the end of 2023, **\$0.16 trillion** at the end of 2024, and a near-empty **\$0.01 trillion** by late 2025. As that parked cash came back into the system, it offset the Fed's QT, net liquidity stabilized and rose, and risk assets recovered into 2023-2024. The drain and the refill of the RRP were a hidden liquidity cycle running *inside* the QT period — which is why the simple story "QT = stocks down" did not hold in 2023.

For the deeper plumbing of the RRP and the TGA, the macro post on [global liquidity, the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide) is the companion; here the lesson is the cross-asset one: *watch net liquidity, not just the headline balance sheet.*

![Reverse repo and the Treasury account stacked area soaked up reserves in 2022](/imgs/blogs/the-liquidity-cycle-and-asset-prices-4.png)

The stacked area above isolates the two reserve drains — the RRP in amber, the TGA in lavender. Together they peaked near **\$2.9 trillion** of reserves siphoned out of the system around the end of 2022. That is the hidden hand that turned the Fed's gradual QT into a sharp, fast liquidity squeeze in 2022, and it is invisible if you watch only the balance-sheet headline.

#### Worked example: computing net liquidity for end-2022

Let us compute the net-liquidity proxy with the curated numbers so the formula stops being abstract. Take the readings near the end of 2022:

- Fed assets (WALCL): **\$8.55 trillion**
- Reverse repo (RRP): **\$2.55 trillion**
- Treasury account (TGA): **\$0.35 trillion**

Net liquidity = \$8.55T − \$2.55T − \$0.35T = **\$5.65 trillion**.

Now compare to the end of 2021:

- Fed assets: **\$8.76 trillion**
- RRP: **\$1.60 trillion**
- TGA: **\$0.46 trillion**

Net liquidity = \$8.76T − \$1.60T − \$0.46T = **\$6.70 trillion**.

So between end-2021 and end-2022, net liquidity fell from \$6.70T to \$5.65T — a drop of about **\$1.05 trillion**, or roughly **16%**. Notice that the Fed's *headline* balance sheet barely moved over that span (from \$8.76T to \$8.55T, down just \$0.21T) — almost the entire liquidity drain came from the RRP filling up, not from QT. Over the same window the S&P 500 fell **−18.1%**. The intuition: **the parking lots, not the headline balance sheet, did most of the 2022 draining — and the market tracked the net number, not the headline.**

## The mechanism: how liquidity moves prices

We have the gauges. Now the heart of the post: *why* does more liquidity push almost every asset up, and less push almost everything down? The answer is three channels working at once. The figure below lays them out.

![Liquidity reaches prices through lower discount rates higher risk appetite and more cash chasing assets](/imgs/blogs/the-liquidity-cycle-and-asset-prices-5.png)

### Channel one: the discount rate falls

Every financial asset is, at bottom, a claim on future cash — a stock's future earnings, a bond's future coupons, a property's future rent. To value it today, you *discount* those future dollars back to the present using an interest rate. The higher the rate, the less a future dollar is worth today; the lower the rate, the more it is worth.

The valuation of any asset can be written, in its simplest form, as:

$$
\text{Value} = \frac{\text{future cash flows}}{(1 + r)^t}
$$

where $r$ is the discount rate and $t$ is how many years away the cash flow is. The single variable that matters most here is $r$. When liquidity floods in and the Fed pushes rates and bond yields down, $r$ falls — and because it sits in the *denominator*, a lower $r$ makes *every* asset's present value rise at once. This is the most universal channel: it does not care whether you own stocks, bonds, gold, or real estate. Lower the discount rate and you raise the value of every future-cash-flow claim in the world simultaneously. It is why a liquidity wave lifts *everything*, not just one asset class. (The variable $r$, in real terms, is the subject of its own deep-dive: [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything).)

The assets that respond most are the **long-duration** ones — the ones whose cash flows are furthest in the future, because they are discounted over the most years. A profitless growth stock whose payoff is 20 years away, or a 30-year bond, moves far more for a given change in $r$ than a value stock paying a fat dividend today. This is exactly why long-duration tech fell **30%** or more in 2022 when $r$ rose, while a dividend-heavy utility barely budged.

### Channel two: risk appetite rises

The second channel is psychological and structural at once. When the central bank pushes safe yields toward zero, it makes holding cash and safe bonds *painful* — you earn nothing. Investors who need a return are forced to **reach for yield**: to take more risk than they otherwise would, because the safe option pays too little to live on. A pension fund that needs 7% a year cannot get it from a 0.5% Treasury, so it buys corporate bonds, then stocks, then private equity, climbing the risk ladder rung by rung.

This is deliberate. It is the *portfolio-balance channel* of QE — the central bank pushes investors out of the assets it is buying and into riskier ones, precisely to raise risk appetite and loosen conditions. The result is that in a high-liquidity regime, the *price of risk* falls: credit spreads compress (risky borrowers pay almost as little as safe ones), volatility drops, and the riskiest assets get bid up the most. In 2021, profitless companies, meme stocks, and speculative crypto all soared — not because their fundamentals improved, but because abundant liquidity had crushed the reward for safety and sent money scrambling for any return it could find.

### Channel three: cash has to go somewhere

The third channel is the most mechanical. When the Fed creates \$4.7 trillion of new reserves, that money does not vanish — it sits in the system as a claim someone holds. The total quantity of cash and near-cash in the world has *gone up*, while the quantity of assets to buy has stayed roughly fixed. More money chasing the same assets means, almost by arithmetic, higher asset prices. There are simply more buyers than sellers at the old price, so the price rises until balance returns.

This is why a liquidity wave shows up first and most violently in the assets with the smallest "float" — the smallest pool of available supply relative to the money chasing them. Crypto is the extreme case: a relatively tiny market cap, so when a wall of liquidity turns toward it, the price moves enormously. When the wall recedes, it collapses just as hard.

There is one important caveat to the cash-chasing channel, and it is what keeps liquidity from being a magic money button. New reserves only lift asset prices if they actually *circulate* — if the people holding the cash choose to spend it on assets rather than sit on it. Economists call the rate at which money changes hands its **velocity**. In a confident, risk-hungry market, new liquidity has high velocity: it ripples quickly out the risk curve, and prices respond fast. In a frightened market — a banking panic, a deflationary scare — velocity can collapse, and even huge reserve injections sit idle as banks and investors hoard cash rather than deploy it. This is why QE was far more powerful for asset prices in 2020-21 (when confidence was high and velocity strong) than it was in, say, 2009-2012 (when the economy was scarred and much of the new money sat as excess reserves rather than chasing risk). Liquidity sets the *potential* for higher asset prices; risk appetite and velocity decide how much of that potential is realized. The practical upshot for an allocator is that a liquidity flood arriving into an *already confident* market — easing on top of greed — is the most explosive combination, while the same flood into a frightened market works more slowly.

#### Worked example: the cash-chasing channel and a fixed pool of assets

Let us make the third channel concrete with a tiny, closed market. Take a market with just two assets: a stock worth \$1,000,000 of total value and a Treasury bond worth \$1,000,000, with \$200,000 of cash on the sidelines. Total wealth: \$2,200,000.

Now the central bank runs QE: it buys the entire \$1,000,000 bond from its holder with newly created cash. The bond is gone from private hands, and there is now \$1,200,000 of cash chasing a single remaining asset — the \$1,000,000 stock. Nobody wants to hold all that cash earning nothing, so they bid for the stock. With \$1,200,000 of cash hunting one \$1,000,000 stock, the price is bid up: the stock might re-rate to \$1,400,000 or more before holders are willing to sit on the remaining cash. The *quantity* of assets did not change — there is still one stock — but its *price* rose because the pool of money chasing it grew and the pool of alternative assets shrank.

That is the cash-chasing channel in miniature: take a fixed pool of assets, flood the system with cash, remove a safe asset from the menu, and the remaining assets must re-price upward to absorb the money. The intuition: **QE shrinks the supply of safe assets while expanding the supply of cash, so the price of the remaining risk assets is mechanically bid up — even with zero change in their fundamentals.**

### The three channels together: correlations go up

Here is the cross-asset punchline. Each of the three channels pushes *many* assets in the *same* direction at the *same* time. Lower discount rates lift all future-cash-flow assets. Higher risk appetite lifts all risky assets. More cash bids up all assets with limited supply. So when liquidity is rising, the normal diversification between asset classes *breaks down* — stocks, bonds, gold, credit, and crypto all rise together, because they are all responding to the same single driver. And when liquidity drains, they all *fall* together, for the same reason. This is why a liquidity-driven market has unusually *high correlation across assets*: the tide is moving every boat at once, so the boats stop moving independently — the diversification you thought you owned quietly disappears just when you need it.

#### Worked example: a 1% drop in the discount rate, across two assets

Let us put a number on the discount-rate channel, because it is the most universal. Take two assets that each pay \$100 in the future, and lower the discount rate from 5% to 4% — a one-percentage-point easing.

**Asset A — a short-duration claim**, paying its \$100 in 2 years. At 5%: value = \$100 ÷ (1.05)² = \$100 ÷ 1.1025 = **\$90.70**. At 4%: value = \$100 ÷ (1.04)² = \$100 ÷ 1.0816 = **\$92.46**. The price rose about **+1.9%**.

**Asset B — a long-duration claim**, paying its \$100 in 20 years. At 5%: value = \$100 ÷ (1.05)²⁰ = \$100 ÷ 2.653 = **\$37.69**. At 4%: value = \$100 ÷ (1.04)²⁰ = \$100 ÷ 2.191 = **\$45.64**. The price rose about **+21.1%**.

The *same* 1% drop in $r$ lifted the long-duration asset more than ten times as much as the short-duration one. The intuition: **a liquidity wave that lowers discount rates lifts every asset, but it lifts the long-duration, far-future-payoff assets — growth stocks, long bonds, crypto — by far the most, which is exactly why they lead in both directions.**

## Two clocks: liquidity vs the business cycle

We now have two separate engines of asset prices. The first is the **business cycle** — growth and inflation, the subject of [the investment clock](/blog/trading/cross-asset/the-business-cycle-and-the-investment-clock) — which decides whether earnings are rising or falling and therefore which assets *deserve* to do well. The second is the **liquidity cycle** — how much money is sloshing around. They are two clocks, and the art of reading markets is knowing which one is in charge at any given moment.

### When the two clocks agree

Sometimes both clocks point the same way, and the move is enormous. The clearest case is **2020**. The business cycle was catastrophic — the sharpest recession since the Depression, with output collapsing and unemployment spiking. On the business cycle alone, you would expect stocks to crash and stay down. But the liquidity clock was running the opposite way and *far harder*: the Fed flooded the system with \$3 trillion of reserves in a matter of months. The liquidity flood overwhelmed the recession, and the S&P 500, having crashed 34% in March, finished 2020 up **+18.4%** — one of the strangest divergences between the economy and the market in history. The recession was real; the liquidity tide was bigger.

The reverse agreement happened in **2022**: *both* clocks tightened at once. Inflation forced the Fed to hike rates and start QT (liquidity draining), while growth was slowing toward a feared recession (business cycle weakening). With both clocks against risk, there was nowhere to hide — which is why 2022 was one of the rare years that stocks *and* bonds fell together, the classic 60/40 portfolio having its worst year since 1937. When both clocks tighten, diversification fails, because the thing hurting one asset is the same thing hurting the other.

### When liquidity overrules the business cycle

The more interesting — and more dangerous — case is when the two clocks *disagree*, and liquidity wins. The defining example is the long bull market of **2013-2021**. Through most of that decade, economic growth was *mediocre*: the post-2008 recovery was the weakest in modern history, with sluggish GDP, low productivity, and tepid wage growth. On the business cycle alone, it should have been a forgettable era for stocks. Instead it was one of the great bull markets, with US equities compounding double digits year after year. Why? Because the liquidity clock was wide open: years of near-zero rates and repeated rounds of QE kept the tide high, and that tide lifted valuations even as earnings grew slowly. The bull market was, to a significant degree, a *re-rating* — multiples expanding because money was cheap and abundant, not because the economy was booming.

This is the most important practical lesson of the whole post: **liquidity can override the business cycle for years at a time.** An allocator who, in 2015 or 2017, said "growth is mediocre, so I'll de-risk" was right about the economy and wrong about the market — because they were watching the wrong clock. Conversely, an allocator who, in early 2022, said "the economy is still growing, so risk is fine" was right about the economy and *catastrophically* wrong about the market, because the liquidity clock had flipped to draining. The economy and the market are not the same thing, and the liquidity cycle is a big part of the gap between them.

#### Worked example: the 2020 divergence in numbers

Let us make the two-clocks idea concrete with the 2020 numbers. Take two analysts in April 2020.

The first watches only the business cycle. She sees GDP collapsing at a 30%+ annualized rate, 20 million jobs lost in a month, and corporate earnings cratering. Her model says: sell stocks, this is a depression. Had you followed her, you would have sat in cash while the S&P returned **+18.4%** for the calendar year.

The second watches the liquidity clock. He sees the Fed's balance sheet leaping from **\$4.31 trillion** in March to **\$7.09 trillion** by June — \$2.78 trillion of new reserves in three months. His model says: the tide is coming in faster than the economy is sinking, so risk assets will rise *despite* the recession. He bought. A \$10,000 stock position bought near the March 2020 low would have been worth roughly **\$17,000** by year-end as the market roughly recovered and pushed higher.

Both analysts read the data correctly. Only one read the *right clock*. The intuition: **in a liquidity-driven episode, the size and speed of the money flood matters more for prices than the state of the economy — so you must watch both clocks and know which is dominant.**

## Global liquidity: it is not just the Fed

So far we have talked as if the Fed were the whole story. It is the biggest single source of liquidity, but it is not the only one — and a serious allocator watches *global* liquidity, because money flows across borders and the world's central banks add up.

### The other central banks

The European Central Bank, the Bank of Japan, the People's Bank of China, and the Bank of England all run their own balance sheets, and their collective expansion or contraction is the *global* liquidity tide. The Bank of Japan in particular has run enormous, near-permanent QE for years, and the yen it creates flows out into global markets through the *carry trade* — borrowing cheaply in yen to buy higher-yielding assets elsewhere. When a major central bank that had been easing suddenly tightens, the global tide can turn even if the Fed has not moved. Global liquidity is the *sum* of these flows, and it has its own dedicated deep-dive in the macro series: [global liquidity, the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide).

### China and the credit impulse

China deserves special mention because it runs the world's second-largest liquidity engine, and it works differently from the Fed's. China eases not mainly through QE but through *credit* — directing its banks to lend more. The change in the rate of new credit creation is called the **credit impulse**, and because China is the marginal buyer of so many global commodities, the Chinese credit impulse has historically *led* global industrial activity and commodity prices by several months. When China's credit impulse turns up, copper, oil, emerging-market stocks, and the whole real-asset complex tend to follow. It is a second liquidity dial, partly independent of the Fed, that especially matters for the cyclical, commodity-linked corner of a portfolio.

### The dollar as the global liquidity valve

The single most important transmitter of global liquidity is the **US dollar**, and it works as the master valve of global financial conditions. Because the world borrows, prices, and saves in dollars, the dollar's level *is* global financial conditions. A *falling* dollar loosens conditions everywhere — it makes the roughly \$13 trillion of dollar debt owed outside the US lighter, frees up capacity for emerging markets to borrow and grow, and is effectively a global liquidity injection. A *rising* dollar does the opposite: it is a global liquidity drain, a margin call on the whole world's dollar borrowers. This is why the dollar is the fastest cross-asset read on the liquidity tide: when it is falling, the global tide is usually rising, and vice versa.

#### Worked example: a falling dollar as a global liquidity injection

Let us size the dollar channel with round numbers. Suppose a Brazilian company has borrowed **\$1,000,000** in dollars, and the dollar then *weakens* 10% against the Brazilian real.

Before: the \$1,000,000 debt cost, say, 5,000,000 reais to repay (at 5 reais per dollar). After the 10% dollar drop (now ~4.5 reais per dollar): the same \$1,000,000 debt costs only **4,500,000 reais** — the borrower just got **500,000 reais** richer with no change in the dollar amount owed. Across the **\$13 trillion** of dollar debt owed outside the US, a broad 10% dollar decline frees up enormous repayment capacity, and that freed-up money flows into spending, investment, and risk assets worldwide.

The reverse is brutal: a *rising* dollar makes that same debt heavier, forces borrowers to cut and sell, and drains global liquidity. The intuition: **the dollar is a global liquidity valve — a weak dollar is an injection that lifts risk assets everywhere, a strong dollar is a drain that sinks them, which is why a sharp dollar move is the fastest warning the tide is turning.**

## How it shows up in real markets

Theory is cheap. Let us walk through the named episodes where the liquidity cycle, not the business cycle, was the dominant force — each with real dates and numbers.

### The 2020-21 everything bull

This is the cleanest liquidity episode in modern history. From March 2020 to April 2022, the Fed created roughly **\$4.7 trillion** of new reserves (assets from \$4.31T to \$8.96T), cut rates to near zero, and ran the easiest financial conditions on record. Almost every asset class rose, and the highest-beta ones rose the most. The figure below shows the asset-price echo.

![High-beta assets rise most and fall hardest as the liquidity tide comes in and out](/imgs/blogs/the-liquidity-cycle-and-asset-prices-6.png)

Read the bars. In 2020, US bonds returned about **+7.5%** (rates falling), gold **+25.1%**, the S&P **+18.4%**, and Bitcoin a staggering **+303%**. In 2021, the S&P returned **+28.7%** and Bitcoin another **+60%**, while bonds turned slightly negative at **−1.5%** as yields began to creep up. The ordering is the tell: the highest-beta, longest-duration, smallest-float asset — Bitcoin — rose the most in the flood, exactly as the discount-rate and cash-chasing channels predict. Speculative tech, SPACs, and meme stocks behaved the same way. This was not an earnings boom; corporate earnings in 2020 *fell*. It was a liquidity boom — a re-rating of every asset because money was cheap and abundant and had to go somewhere.

### The 2022 everything bust

In 2022 the tide went out, and the same chart runs in reverse. The Fed hiked from near zero to over 4% and began QT, and the RRP soaked up \$2.55 trillion of reserves. The S&P fell **−18.1%**, bonds fell **−13.0%** (their worst year in a century), and Bitcoin fell **−64%** — falling hardest precisely because it had risen hardest. Long-duration tech fell 30%+ as the discount-rate channel worked in reverse. The crucial point, made earlier, is that the US economy was still *growing* through most of 2022; this was a liquidity-and-rates bear market, not a recession bear market. An allocator watching only the economy would have been blindsided; one watching the liquidity clock saw it coming.

### The 2023 "QT but rally" puzzle

2023 confused everyone who used the simple rule "QT = stocks down." The Fed was *still* running QT all year, yet the S&P returned **+26.3%**. The resolution is net liquidity. Even as the Fed shrank its balance sheet, the RRP *drained* — from \$2.55 trillion at the end of 2022 to \$1.02 trillion at the end of 2023 — releasing more than \$1.5 trillion of parked cash back into the system. That refill more than offset QT, so *net* liquidity actually held up or rose even while the headline balance sheet fell. The lesson is exactly the one from the net-liquidity section: the headline balance sheet can mislead; you have to net out the parking lots.

### The 2013 "taper tantrum"

In May 2013, the Fed merely *hinted* that it would slow (not stop, just slow) its QE purchases. The reaction was violent: the 10-year Treasury yield jumped about 100 basis points (1 percentage point) in months, emerging-market currencies and bonds sold off hard, and risk assets wobbled — all on a mere *signal* that the liquidity tide might recede a little. The taper tantrum is the proof that markets price the *direction* of liquidity, not just its level. The tide did not even go out; the Fed only suggested it might stop coming in quite so fast, and that was enough to shake the most liquidity-sensitive assets.

### The 2018 QT selloff

In 2018 the Fed ran QT for the first time, shrinking its balance sheet while also hiking rates. Liquidity drained gradually all year, and by the fourth quarter risk assets cracked: the S&P fell nearly 20% peak-to-trough into late December, and credit spreads widened sharply — despite a US economy that was, on the surface, booming with strong GDP and record-low unemployment. The Fed reversed course in early 2019, and markets recovered. 2018 was a dress rehearsal for 2022: a liquidity-drain bear market in the teeth of a strong economy.

### The 2008-09 birth of QE

QE was not always the standard tool. It was invented, at scale, in the depths of the 2008 financial crisis, when conventional rate cuts had already taken the policy rate to zero and the economy was still collapsing. The Fed began buying mortgage securities and Treasuries, and its balance sheet — which had sat near \$0.9 trillion for years — ballooned past \$2 trillion almost overnight, eventually reaching \$4.5 trillion across successive rounds (QE1, QE2, QE3) by 2014. The stock market bottomed in March 2009, within weeks of the first big liquidity push, and began the long bull market that ran for over a decade. The crucial nuance, foreshadowing the velocity point above, is that this liquidity was far *less* potent per dollar than the 2020 version: confidence was shattered, banks hoarded the new reserves, and much of the money sat idle rather than chasing risk. The tide came in, but slowly, into a frightened market — which is why the 2009-2014 recovery in asset prices was a long grind rather than the vertical melt-up of 2020-21.

### The Bank of Japan and the global carry tide

The clearest reminder that liquidity is *global* is Japan. The Bank of Japan has run the most aggressive, longest-lasting QE program in the developed world, holding its policy rate at or below zero for years and buying not just bonds but, unusually, stocks (ETFs) directly. The yen it creates does not stay home — it funds the global *carry trade*, in which investors borrow at near-zero in yen and buy higher-yielding assets around the world. For years this was a steady source of global liquidity, a tailwind beneath risk assets that had nothing to do with the Fed. The flip side appeared in August 2024, when the BOJ finally hiked and the yen surged: the carry trade unwound violently, forcing global investors to sell assets to repay yen loans, and stock markets worldwide dropped sharply in a matter of days. It was a liquidity-drain episode sourced not from Washington but from Tokyo — proof that an allocator who watches only the Fed is watching only part of the tide.

## Common misconceptions

**"The stock market reflects the economy."** Only loosely, and sometimes not at all. In 2020, the economy had its worst quarter in modern history while the stock market finished the year up double digits — because the liquidity tide overwhelmed the recession. The market reflects *future cash flows discounted by a rate that liquidity controls*, not current GDP. The two can diverge for years; 2013-2021 was mediocre growth and a huge bull market, driven by liquidity.

**"QE directly buys stocks, that's why they go up."** No — the Fed (in the US) buys government bonds and mortgage securities, not equities. Stocks rise *indirectly*, through the three channels: a lower discount rate, forced reaching for yield, and cash rippling out the risk curve. The distinction matters because it means the effect is broad — it lifts every risk asset, not just the ones the Fed touches.

**"As long as the Fed's balance sheet is shrinking, stocks must fall."** 2023 disproved this cleanly: QT ran all year and the S&P rose 26%. The headline balance sheet is not the same as net liquidity. When the RRP and TGA *release* parked reserves faster than QT removes them, net liquidity can rise even as the balance sheet falls. Watch the net number.

**"Liquidity only matters for crypto and speculative junk."** Liquidity matters most *visibly* for high-beta assets, but it moves everything, because the discount-rate channel touches every future-cash-flow claim. In 2022, the supposedly safe 60/40 portfolio of stocks and bonds had its worst year since 1937 — a liquidity-and-rates event that hit the most conservative allocation in the book, not just the speculative fringe.

**"You can't trade liquidity because the data is lagged and noisy."** The balance sheet, RRP, and TGA are published weekly or daily by the Fed and Treasury — among the timeliest macro data that exist. And you do not even need them in real time: the *fast reads* — the dollar and real yields — move continuously and tell you the direction of the tide before the balance-sheet data confirms it.

## When to own it: the liquidity playbook

This is the payoff. The liquidity cycle is not an asset you buy; it is a *lens* that tells you how aggressively to own everything else. The figure below is the playbook matrix.

![The liquidity-cycle playbook matrix of what to lean into when liquidity expands or drains](/imgs/blogs/the-liquidity-cycle-and-asset-prices-7.png)

### The core rule

**Track the *direction* of net liquidity and financial conditions, and lean with the tide.** When liquidity is expanding — the balance sheet rising, the RRP and TGA draining, conditions easing — lean *risk-on* across assets, and tilt toward the highest-beta, longest-duration names (growth stocks, long bonds, credit, crypto), because they capture the most upside from a falling discount rate and rising risk appetite. When liquidity is draining — the balance sheet shrinking, the parking lots filling, conditions tightening — *de-risk*, even if the economy looks fine, and tilt toward short-duration, cash-like, and defensive holdings. The single most important word is *direction*: markets price the change in liquidity, not its level, which is why the taper tantrum and the 2018 QT selloff happened on the *turn*, not at any particular absolute level.

### The fast read: the dollar and real yields

You do not need to compute net liquidity every morning. The two fastest, most continuous reads on the liquidity tide are:

- **The dollar (DXY).** A *falling* dollar means loosening global conditions and a rising tide — green for risk. A *rising* dollar means tightening conditions and a draining tide — a warning across the whole risk complex.
- **Real yields.** The *real* yield — the interest rate after subtracting expected inflation — is the cleanest single read on financial conditions. *Falling* real yields mean easing conditions and a rising tide; *rising* real yields mean tightening and a draining tide. (See [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything).)

When the dollar and real yields are both *falling*, the tide is almost certainly coming in — lean risk-on. When both are *rising*, the tide is going out — raise hedges and cut the highest-beta exposure first. These two move every day and front-run the weekly balance-sheet data.

### Sizing, pairing, and what invalidates the case

**Sizing.** The liquidity tilt is a *modifier* on your strategic allocation, not a replacement for it. In an expanding-liquidity regime, you might tilt 5-15% more toward risk and duration than your neutral weight; in a draining regime, the same amount toward cash and defense. The bigger and faster the liquidity move (2020, 2022), the bigger the justified tilt; a gentle drift warrants only a gentle lean.

**Pairing.** The highest-beta liquidity proxy — crypto, covered in [crypto and digital assets, the new high-beta class](/blog/trading/cross-asset/crypto-digital-assets-the-new-high-beta-class) — is the purest expression of a rising tide, but also the first thing to cut when it turns. Pair an aggressive liquidity tilt with a fast, liquid hedge (cash, short-duration Treasuries, or the dollar itself), so you can de-risk quickly when the dollar and real yields turn.

**What invalidates the case.** The liquidity tilt is *wrong* when the business cycle clock dominates and disagrees — for example, when a genuine credit or recession event is unfolding *despite* easy liquidity, the falling-earnings clock can overwhelm the rising-money clock. It is also wrong at turning points you misread: if you are leaning risk-on because liquidity *was* expanding, but the dollar and real yields have quietly turned up, the tide has changed and your thesis is broken. The discipline is simple — when the fast reads (the dollar, real yields) flip against your position, respect them, because they lead the slow data.

#### Worked example: sizing a liquidity tilt across the cycle

Let us turn the playbook into numbers with a simple \$100,000 portfolio whose neutral, all-weather weight is 60% stocks / 40% bonds. The liquidity tilt adjusts those weights based on which way the tide is running.

In late 2020, the fast reads were screaming *tide in*: the dollar was falling toward its 2021 lows and real yields were deeply negative, while the Fed's balance sheet was racing higher. An allocator leaning with the tide might tilt 10 points more toward risk and duration — say 70% stocks / 25% bonds / 5% in the highest-beta sleeve (crypto or growth). On the curated 2021 returns, the risk-tilted book captured roughly 0.70 × 28.7% (stocks) plus a slice of crypto's +60%, beating the neutral 60/40, which earned about 0.60 × 28.7% + 0.40 × (−1.5%) ≈ **+16.6%**.

Then the fast reads flipped in early 2022: the dollar surged and real yields turned sharply positive — *tide out*. The disciplined move is to cut the tilt and de-risk, dropping to perhaps 45% stocks / 30% short-duration bonds / 25% cash, and selling the high-beta sleeve *first*. On the 2022 returns, the de-risked book lost far less than the neutral 60/40, which fell about 0.60 × (−18.1%) + 0.40 × (−13.0%) ≈ **−16.1%** — its worst year since 1937. An allocator who had instead held the *2021* risk-tilt into 2022 would have been crushed, with the high-beta sleeve down 64%.

The whole exercise hinges on respecting the turn: the tilt that made you money in 2021 is the tilt that ruins you in 2022, and the only thing that told you to switch — before the balance-sheet data confirmed it — was the dollar and real yields turning up. The intuition: **the liquidity tilt is a steering wheel, not a set-and-forget allocation; the fast reads tell you when to turn it, and the cost of ignoring the turn is the difference between +17% and −16%.**

The deepest lesson of the liquidity cycle is humility about what actually drives markets. It is tempting to believe that asset prices reflect the careful judgment of millions of investors weighing fundamentals. Much of the time they do. But in the big moves — the 2020-21 melt-up, the 2022 collapse — the dominant force was not fundamentals at all. It was the tide: how much money the world's central banks were pumping in or pulling out. Learn to watch that tide, as a second clock running next to the business cycle, and you will understand the years when the economy and the market seem to live in different worlds — because, on the liquidity clock, they do.

## Further reading & cross-links

- [The business cycle and the investment clock](/blog/trading/cross-asset/the-business-cycle-and-the-investment-clock) — the *other* clock, growth and inflation, and which asset each phase favors. Read alongside this post: liquidity is the second clock running next to the first.
- [Real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) — the cleanest fast read on financial conditions and the discount-rate channel in one number.
- [Crypto and digital assets, the new high-beta class](/blog/trading/cross-asset/crypto-digital-assets-the-new-high-beta-class) — the purest, highest-beta expression of the liquidity tide, rising most and falling hardest.
- [Global liquidity, the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide) — the macro deep-dive on the world's combined central-bank balance sheets, the RRP and TGA plumbing, and the credit impulse.
- [What liquidity means for funding and global traders](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders) — the three meanings of liquidity and the funding-market plumbing this post builds on.

*This is educational, not investment advice. Liquidity-driven moves are powerful but hard to time, and every asset that rises with the tide can fall just as far when it turns — size positions accordingly.*
