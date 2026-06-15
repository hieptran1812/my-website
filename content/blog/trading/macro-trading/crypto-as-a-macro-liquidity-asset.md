---
title: "Crypto as a Macro Asset: Bitcoin, Liquidity, and Real Rates"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Whatever you think of its long-run thesis, Bitcoin trades today as a macro asset — a high-beta proxy for global liquidity and risk appetite, acutely sensitive to real interest rates and the dollar. This is a beginner-friendly deep dive into the liquidity beta, why a zero-cashflow asset behaves like pure duration, the dollar link, the halving-vs-macro debate, and how to trade it off net liquidity, real yields, and DXY."
tags: ["macro", "monetary-policy", "bitcoin", "crypto", "global-liquidity", "real-yields", "net-liquidity", "dollar", "risk-on-risk-off", "duration", "halving", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Whatever you think of its long-run thesis, Bitcoin trades *today* as a macro asset: a high-beta proxy for global liquidity and risk appetite, acutely sensitive to real interest rates and the dollar. Trading it without the macro lens is trading blind.
>
> - Bitcoin tracks the **Fed's balance sheet** more reliably than it tracks any on-chain metric. The 2020-21 liquidity flood took it from under \$5,000 to nearly \$69,000; the 2022 real-rate shock crushed it; the 2024 liquidity-and-ETF wave took it past \$100,000.
> - Because it has **no cash flow**, Bitcoin is a pure-duration asset: its entire value is a distant payoff that discounts at the longest, most rate-sensitive horizon there is. When the **10Y TIPS real yield** rose from about **-1% to +1.7%** in 2022, Bitcoin fell **-65%** (from **\$46,306** to **\$16,548**).
> - It is **not** uncorrelated and it is **not** a crisis hedge in the moment of panic. In risk-driven regimes it behaves as a higher-beta version of the **Nasdaq** — it moves roughly **2-3x** the global-liquidity swing.
> - The one playbook to remember: read **net liquidity** (Fed assets minus the ON RRP minus the TGA), **real yields**, and **the dollar (DXY)**. Rising liquidity + falling real yields + soft dollar = tailwind; the opposite = headwind. Then **size for the beta**, because that is the part that ruins people.

In the autumn of 2024, a strange thing happened to two charts that, on paper, have nothing to do with each other. One was the Federal Reserve's balance sheet plus the cash sloshing around its plumbing — the most boring, institutional, suit-and-tie number in all of finance. The other was the price of Bitcoin, an asset born from a cypherpunk manifesto, championed by people who wanted *nothing to do* with central banks. And yet, lay the two charts on top of each other and they move like dance partners. When the Fed's effective liquidity expanded, Bitcoin rose. When it drained, Bitcoin fell. Not loosely — tightly, with a lag of weeks, across years.

This is awkward for almost everyone. It is awkward for the Bitcoin believer, who would prefer the price reflect adoption, scarcity, and the slow triumph of sound money over fiat debasement. And it is awkward for the traditional macro investor, who would prefer to dismiss crypto as a casino with no analytical handle. But the data does not care what either camp prefers. Over the last cycle, the single best predictor of Bitcoin's price was not the number of active addresses, not the hash rate, not the halving countdown, not even the adoption headlines. It was **global liquidity** — the worldwide tide of money the central banks create — and Bitcoin's exquisite sensitivity to the **real interest rate** and the **dollar**.

That is the claim of this post, and we are going to build it from the ground up, assuming you know nothing about either crypto or macro. The thesis is not that Bitcoin's long-run store-of-value case is right or wrong — that is a separate, slower debate. The thesis is narrower and, for a trader, far more useful: *right now, on a one-to-three-year horizon, Bitcoin behaves as a macro asset.* It is the highest-beta expression of one underlying force — the global cost and quantity of money. Read that force, and you read Bitcoin. Ignore it, and the price will look like random noise punctuated by inexplicable booms and busts. We will go all the way to a concrete playbook: which three dials to watch, what they tell you, and how to size a position so the beta works for you instead of against you.

![Bitcoin trades as a high-beta proxy for global liquidity, real rates, the dollar, and risk appetite](/imgs/blogs/crypto-as-a-macro-liquidity-asset-1.png)

## Foundations: why crypto became a macro asset

Before we can talk about how Bitcoin trades, we have to separate two completely different questions that people constantly mash together. The mashing is the source of most bad crypto arguments.

### Two questions, kept apart

**Question one is the long-run thesis.** Will Bitcoin, over decades, become a widely held store of value — a kind of digital gold, a hedge against the debasement of government currencies, a neutral reserve asset outside any state's control? That is a question about adoption curves, network effects, regulation, and whether a fixed-supply asset can anchor a meaningful share of the world's savings. It is a genuine and interesting debate, and it is explored from the origin story in [Bitcoin and the cypherpunk vision](/blog/trading/finance/bitcoin-and-the-cypherpunk-vision). It plays out over ten or twenty years.

**Question two is how it trades today.** On a horizon a trader actually cares about — weeks to a few years — what moves the price? What makes it go up 300% in a year and then fall 65% the next? This is not a question about long-run adoption. It is a question about *flows and discount rates* — the same forces that move every other risk asset. And the answer, empirically, is that crypto trades as a macro asset.

Here is the crucial point: **both can be true at once.** Bitcoin can be slowly winning a long-run adoption battle *and* trade, day to day and year to year, as a leveraged bet on global liquidity. The long-run thesis sets the slow trend; the macro tide sets the violent cycle around that trend. A trader operates in the cycle. This post is about the cycle. We park the long-run debate and ask only: *what is the price doing, and why, on the horizon you trade?*

### What "a macro asset" means

A **macro asset** is one whose price is driven primarily by economy-wide variables — the quantity of money, the level of interest rates, the strength of the dollar, the appetite for risk — rather than by anything specific to itself. A single company's stock is a *micro* asset: it rises and falls partly on its own earnings, products, and management. The whole stock *index* is more of a macro asset: it rises and falls mostly on rates, liquidity, and growth, because the idiosyncratic stories of five hundred companies wash out and leave the common macro factor behind.

Bitcoin is unusual because it has **no idiosyncratic fundamental at all** in the financial sense. A company has earnings. A bond has coupons. A currency has an interest rate and a trade balance behind it. Bitcoin has none of these. There is no cash flow, no profit, no dividend, no central bank setting its rate. So when you strip away the one thing — micro fundamentals — that gives most assets a private story, what is left? Only the macro factors. With nothing to anchor it to its own economics, Bitcoin floats almost entirely on the macro tide. That is precisely *why* it became such a pure, high-beta macro asset: it has nothing else to be.

### The liquidity beta, in one idea

The central concept of this whole post is **liquidity beta**. Let us build it carefully.

"Liquidity" here means **macro liquidity** — the total quantity of money-like balances the central banks and banking system have pushed into the financial system. It is the ocean of money in which all assets float. (There are other meanings of "liquidity"; we use this one throughout, and the full anatomy is in [global liquidity, the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide) and [what liquidity means](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders).)

"Beta" is a measure of *how much* an asset moves when its driver moves. A beta of 1 means it moves one-for-one. A beta of 2 means it moves twice as much. A beta of 0.5 means it moves half as much. When we say Bitcoin has a high **liquidity beta**, we mean: when the global money tide rises or falls by some amount, Bitcoin moves by a *multiple* of that amount, in the same direction.

The rough empirical numbers, observed across the last cycle, are striking. When global liquidity swings, the Nasdaq (the tech-heavy US stock index) moves roughly 1.5 times as much. Bitcoin moves roughly **2-3 times** as much. Bitcoin is, in this sense, the highest-octane expression of the liquidity trade available in liquid markets. It is what you buy when you want maximum exposure to "more money is being created" — and what crushes you when the money drains.

### The three macro dials that set the tide

Macro liquidity is not one number you can read off a single screen. For a crypto trader, it resolves into three dials, and almost everything in this post comes back to them:

- **The quantity of money / net liquidity.** How much cash is in the system and chasing assets. The cleanest proxy is **net liquidity**: the Fed's balance sheet minus two cash "sinks" that absorb money — the overnight reverse repo facility (ON RRP) and the Treasury's checking account (the TGA). The full mechanic is in [the central-bank balance sheet, net liquidity, reserves, RRP, and TGA](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga). Rising net liquidity = more cash = tailwind for risk.
- **The real interest rate.** The cost of money *after* stripping out inflation. This is the discount rate the whole market uses to value future payoffs, and Bitcoin — as we will see — is the most discount-rate-sensitive asset there is. Real yields up = headwind; real yields down = tailwind.
- **The dollar.** Because the dollar is the world's funding currency, its strength is a measure of how tight global funding is. A strong dollar drains effective global liquidity even if the Fed has not changed anything. Strong dollar = headwind; soft dollar = tailwind.

Hold those three dials in your head. By the end, your entire crypto-macro playbook will be: *read the three dials, set your bias, size for the beta.* Everything in between is explaining why each dial works.

## The liquidity beta: Bitcoin as high-beta global liquidity

Let us start with the headline fact and make it concrete, because it is the fact that converts skeptics: Bitcoin tracks the Fed's balance sheet better than it tracks anything intrinsic to itself.

### Why the balance sheet is the cleanest liquidity proxy

Recall how a central bank creates money. When the Fed wants to inject money, it *buys assets* — usually government bonds — and pays for them by creating new bank reserves, which is electronic money it conjures into existence. The bonds land on the asset side of its balance sheet; the new reserves land on the liability side. The balance sheet grows on both sides at once, and the system now holds reserves that did not exist a moment before. (The full mechanic of money creation is in [quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money) and [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).)

So the *size of the Fed's balance sheet* is, to a first approximation, a direct measure of how much money it has pumped into the system. It is public, updated weekly, and unambiguous. When it expands, the Fed is flooding; when it shrinks (a process called quantitative tightening, or QT), the Fed is draining. This makes it the single cleanest macro-liquidity dial available, which is why it became the chart everyone overlays on Bitcoin.

### The chart that converts skeptics

The Fed's balance sheet went from about \$4.3 trillion at the start of 2020 to a peak of **\$8.96 trillion** by April 2022 — it more than doubled in two years. Then, under QT, it drained back toward **\$6.66 trillion** by mid-2025. Now lay Bitcoin's year-end prices alongside it.

![Bitcoin year-end close on a log axis versus the Fed balance sheet, 2018 to 2025](/imgs/blogs/crypto-as-a-macro-liquidity-asset-2.png)

The shape tells the story. As the balance sheet ramped through 2020-21, Bitcoin ramped — from \$28,990 at the end of 2020 to \$46,306 at the end of 2021 (and to an intraday peak near \$69,000 in November 2021). As the balance sheet rolled over and QT began draining it through 2022, Bitcoin collapsed. As liquidity conditions turned supportive again into 2024, Bitcoin ran to \$93,429 by year-end. Three regime turns in the liquidity dial; three regime turns in the price. The correlation is not perfect — nothing in markets is — but it is far tighter than any on-chain metric over the same window.

#### Worked example: Bitcoin's liquidity beta in the 2020-21 flood

Let us put a number on the beta during the flood. We will compare the move in liquidity to the move in price.

- **The liquidity move.** The Fed's balance sheet went from roughly \$4.3 trillion (early 2020) to \$8.96 trillion (April 2022) — call it a peak-to-trough expansion that roughly **doubled** the Fed's footprint, a +108% expansion at its widest. Even measured year-end to year-end, from \$7.36 trillion (end 2020) to \$8.76 trillion (end 2021), that is a +19% expansion in just the Fed's slice of global liquidity, and the other major central banks were expanding in parallel.
- **The Bitcoin move.** Over the same flood, Bitcoin's year-end close went from **\$28,990** (2020) to **\$46,306** (2021) — a +60% year. But that understates it: the intra-cycle move ran from under \$5,000 in March 2020 to nearly \$69,000 in November 2021, a move of well over **+1,200%** from the liquidity-crisis low to the liquidity-flood high.
- **The beta.** Even on the conservative year-end measure, a +19% expansion in the Fed's balance sheet sat alongside a +60% rise in Bitcoin — a ratio above **3x**. On the broader flood measure, the ratio is far higher still. Whichever window you choose, Bitcoin amplified the liquidity swing by a large multiple.

So a \$1 move in the "liquidity factor" showed up as several dollars of move in Bitcoin — that is what a high liquidity beta means, and it is why crypto is the loudest instrument in the liquidity orchestra.

### Why crypto sits at the far end of the risk spectrum

There is a clean reason crypto's beta is the highest of any liquid asset class. When new money floods the system, it does not sit still. Investors who suddenly hold more cash than they want push it into assets — and they push it out along the **risk spectrum**, from safest to riskiest. First into Treasuries, then investment-grade bonds, then large-cap stocks, then small-caps and high-yield, then speculative growth, and at the very far end — the riskiest, most speculative, most "I will buy it because it is going up" assets — crypto.

Money flows down that spectrum like water finding the lowest valley last. When the tide is rising, the marginal new dollar eventually reaches the far end and crypto gets a disproportionate share, because there is less of it and the buyers are most aggressive there. When the tide falls, the far end empties *first and hardest*, because the riskiest assets are the first thing leveraged players sell to raise cash. This asymmetry — last to fill, first to drain — is exactly what produces a high beta. The full anatomy of how money rotates along this spectrum is in [risk-on, risk-off, how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates).

### Stablecoins: crypto's own internal liquidity gauge

There is a second liquidity signal that lives *inside* crypto itself, and it is one of the most useful tells a macro trader can watch: the total supply of **stablecoins** — dollar-pegged tokens like USDT and USDC that function as crypto's cash. A stablecoin is a claim on a dollar, held on-chain, that traders use as dry powder: to buy crypto, they first convert dollars into stablecoins, then deploy them. So the *aggregate supply* of stablecoins is, in effect, the amount of cash sitting on the sidelines inside the crypto system, waiting to be spent.

That makes stablecoin supply a real-time, market-priced gauge of crypto-specific liquidity, and it behaves exactly as the macro lens predicts. When global liquidity is expanding and risk appetite is rising, money flows into the crypto system, stablecoin supply grows, and that growing pile of on-chain cash is the fuel for the next leg higher. When liquidity drains and risk appetite falls, stablecoins are redeemed back into dollars and the aggregate supply *shrinks* — the dry powder is leaving the casino. Through the 2022 bear market, total stablecoin supply contracted for the first time in its history, a quiet confirmation that the liquidity tide had turned, lined up precisely with the price collapse. Through the 2023-24 recovery, it expanded again ahead of the price.

For a trader, this gives a crypto-native cross-check on the macro signal. The macro dials — net liquidity, real yields, the dollar — tell you what the *global* tide is doing; stablecoin supply tells you whether that tide is actually flowing *into the crypto system* yet. When the two agree — global liquidity rising *and* stablecoin supply growing — the signal is strongest. When they diverge — global liquidity rising but stablecoins still shrinking — the macro tailwind has not yet reached crypto, and you wait for the confirmation. It is the on-chain verification of the off-chain thesis.

## Real-rate sensitivity: a zero-cashflow asset is pure duration

The liquidity beta explains the *quantity* dial. Now we turn to the *price* dial — the real interest rate — and it is here that Bitcoin's behavior is most mathematically precise, and most surprising to newcomers. To understand it we need one idea from bond math: **duration**.

### What duration is, built from zero

Every asset that pays off in the future is worth, today, the *present value* of those future payoffs. A dollar you will receive in ten years is worth less than a dollar today, because you could invest today's dollar and have more than a dollar in ten years. To convert a future dollar into today's value, you **discount** it — divide it by a growth factor that depends on the interest rate and how far away the payoff is.

The further away the payoff, the harder the discounting bites, and — crucially — the more *sensitive* today's value is to a change in the interest rate. **Duration** is the name for that sensitivity: it measures how much an asset's price moves when the discount rate moves. A short-duration asset (payoffs arriving soon) barely flinches when rates change. A long-duration asset (payoffs arriving far in the future) swings violently. (Duration is unpacked further in [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).)

Here is the load-bearing insight. A bond pays *coupons* — cash that arrives soon, every year, cushioning the blow when rates rise, because a big chunk of the bond's value is near-term cash that barely discounts. An asset that pays *nothing* until some distant date has no such cushion. Its entire value sits in the far future, so its entire value discounts at the longest, most rate-sensitive horizon there is. A zero-coupon asset has the **longest possible duration** for its maturity.

### Bitcoin: the longest-duration asset there is

Now look at Bitcoin through this lens. It pays **no cash flow, ever** — no coupon, no dividend, no yield. There is no near-term cash to cushion a rate rise. Whatever value the market assigns to Bitcoin is entirely a bet on a *distant* payoff: that it will be worth much more, far in the future. That means its entire value behaves like a single payment in the far future — which is to say, it behaves like the **longest-duration asset in the market**.

![A bond's coupons cushion a rate rise while Bitcoin pays nothing and discounts at the longest duration](/imgs/blogs/crypto-as-a-macro-liquidity-asset-3.png)

This is why Bitcoin is *acutely* sensitive to the real interest rate. When the discount rate rises, every future payoff is worth less today, and the further out the payoff, the more it shrinks. Bitcoin's payoff is the furthest out of all (effectively "the future"), so a rise in the real rate hits it harder than it hits a coupon-paying bond, a dividend-paying stock, or even a high-growth tech stock whose earnings are at least *somewhere* on the horizon. Bitcoin's earnings are nowhere; its whole story is the far future; its duration is maximal.

### Why *real* rates, not nominal

We keep saying *real* rate, not *nominal* rate, and the distinction is essential. The **nominal** interest rate is the headline number — say, a 4% Treasury yield. The **real** rate is the nominal rate *minus expected inflation* — it is the true, inflation-adjusted cost of money. If a bond yields 4% but inflation is expected to run at 3%, the real return is only about 1%. The cleanest market measure of the real rate is the yield on **Treasury Inflation-Protected Securities (TIPS)**, which strip out inflation and quote a pure real yield directly. The full case for why this is the master signal is in [real vs nominal, inflation, real yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).

The reason real rates matter for a zero-cashflow asset, and not nominal rates, is subtle but important. If inflation is high and nominal rates are high *only because of inflation*, then a fixed-supply asset like Bitcoin should — in theory — keep pace, because its scarcity is supposed to protect against debasement. What actually hurts a no-yield asset is a rise in the *real* rate: when the cost of money rises in inflation-adjusted terms, the opportunity cost of holding a thing that yields nothing goes up. You could instead hold a TIPS bond earning a positive *real* return, guaranteed. Every percentage point of real yield is a percentage point of return Bitcoin must beat just to break even against the safe alternative. When real yields go deeply negative — as they did in 2021, around -1% — holding a no-yield asset costs you nothing relative to bonds, and speculative assets fly. When real yields surge positive — as they did in 2022, to +1.7% — the safe alternative suddenly pays you real money, and the no-yield asset gets repriced brutally.

#### Worked example: the 2022 real-rate crash, -65%

This is the cleanest demonstration of duration in the post, so let us be precise with the numbers.

- **The real-rate move.** The 10-year TIPS real yield went from about **-1.04%** at the end of 2021 to **+1.74%** by October 2022 — a swing of roughly **+2.8 percentage points**, from deeply negative to solidly positive. This was one of the fastest real-rate tightenings in modern history, driven by the Fed hiking its policy rate from near zero to over 4% to fight 9% inflation.
- **The Bitcoin move.** Over the same window, Bitcoin's year-end close fell from **\$46,306** (end 2021) to **\$16,548** (end 2022). That is a drop of **\$29,758**, or about **-64%** — call it -65%.
- **Reading it as duration.** A +2.8 percentage-point rise in the real discount rate, applied to the longest-duration asset there is, produced a roughly two-thirds loss of value. No coupon cushioned it because there is no coupon. No earnings backstopped it because there are no earnings. The entire payoff is in the far future, and the far future just got repriced. Note that the Fed's balance sheet had only just begun draining in 2022 — the *quantity* dial barely moved — yet Bitcoin fell 65%. The crash was overwhelmingly a *real-rate* event, not a liquidity-quantity event. The duration lens explains it where a money-supply lens alone would not.

![Bitcoin price falling as the 10-year real yield rises through 2022](/imgs/blogs/crypto-as-a-macro-liquidity-asset-4.png)

So the single sharpest risk to a crypto position is not a hack or a regulatory headline — it is a rise in the real interest rate, because a zero-cashflow asset is pure, maximal duration.

## The dollar and the risk-on correlation

We have two of the three dials: the quantity of money (liquidity beta) and the price of money (real-rate duration). The third dial is **the dollar**, and it interlocks with a fourth idea — Bitcoin's correlation with risk assets like the Nasdaq. Together they complete the picture of crypto as a macro instrument.

### Why a strong dollar drains crypto

The dollar is the world's funding currency. A vast amount of global borrowing, trade, and debt is denominated in dollars, including by entities that earn their revenue in other currencies. When the dollar *strengthens*, every one of those dollar debts becomes heavier to service in local-currency terms, global funding conditions tighten, and effective liquidity drains out of the system — even if the Fed has not changed a thing. A rising dollar is, in effect, a global tightening. (Why the dollar holds this central role is the subject of [the dollar system, why USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy).)

The dollar's strength is tracked by the **DXY**, the dollar index, which measures the dollar against a basket of major currencies. When DXY surges, it is a reliable signal that global funding is tightening and risk assets are about to feel it. The DXY ripped to a multi-decade high near 114.8 in September 2022 — the same window in which Bitcoin was collapsing. The strong dollar and the real-rate shock were two faces of the same Fed tightening, and Bitcoin, as the highest-beta risk asset, absorbed the full force of both. Conversely, when the dollar softens, global funding loosens, effective liquidity expands, and crypto catches a tailwind. So the rule is simple and load-bearing: **strong dollar = headwind for crypto; soft dollar = tailwind.**

### The risk-on correlation with the Nasdaq

Here is the fact that most surprises people who think of Bitcoin as an independent, uncorrelated asset: over the last cycle, Bitcoin has been **strongly correlated with the Nasdaq** — and the correlation *rises* exactly when it matters most, in risk-driven regimes.

Why? Because the Nasdaq is full of long-duration growth stocks — companies whose value is mostly in far-future earnings, which makes them the most rate-sensitive corner of the equity market. Bitcoin is the *even longer-duration* version of the same trade: a payoff entirely in the future, with no current earnings at all. When real rates rise and risk appetite falls, both get sold, and Bitcoin — being further out the risk and duration spectrum — gets sold harder. When real rates fall and risk appetite returns, both rally, and Bitcoin rallies harder. They are not independent stories; they are the same risk-on/risk-off trade expressed at two different betas.

> **A note on the data.** The curated dataset behind this post's charts does not include a Nasdaq series, so the BTC-Nasdaq relationship here is described and cited rather than charted. The empirical record is well documented: through 2021-2023, the rolling correlation between Bitcoin and the Nasdaq frequently ran above 0.5 and spiked higher during stress, and Bitcoin's volatility ran several times that of the index. The chart below is conceptual — it shows the *structure* of the relationship (both move together; Bitcoin moves more), not a fitted time series.

![Bitcoin and the Nasdaq move together in risk regimes with Bitcoin amplifying the move](/imgs/blogs/crypto-as-a-macro-liquidity-asset-7.png)

#### Worked example: sizing the beta — \$100,000 in crypto vs \$100,000 in equity

This is where the high beta stops being an abstraction and starts being your account balance. Suppose you hold a \$100,000 crypto allocation and a friend holds a \$100,000 equity (stock-index) allocation. They are the same number of dollars. They are nowhere near the same amount of *risk*.

- **The equity allocation.** Suppose a moderate risk-off move takes the broad stock index down 10%. The \$100,000 equity position loses about **\$10,000**, leaving \$90,000. Uncomfortable, but survivable.
- **The crypto allocation, by liquidity beta.** Crypto runs at roughly **2-3x** the beta of equities to the same macro shock. So the same risk-off move that takes stocks down 10% plausibly takes crypto down **20-30%**. The \$100,000 crypto position loses **\$20,000 to \$30,000**, leaving \$70,000 to \$80,000.
- **The deeper point.** In a real macro shock — the 2022 type — crypto did not fall 20-30%; it fell **-65%**. A \$100,000 position became roughly **\$35,000**. The same dollar amount, in crypto, carried the risk of an equity position three or four times larger. So if you want crypto to carry the *same risk* as a \$100,000 stock allocation, you hold something closer to **\$30,000-\$50,000** of crypto, not \$100,000.

The takeaway that keeps you solvent: in crypto, the question is never *whether* to be exposed but *how much*, and "how much" must be measured in risk (beta times volatility), not in dollars.

### When the crypto-Nasdaq correlation breaks, and why it returns

Beginners often seize on moments when Bitcoin *decouples* from the Nasdaq as proof it is "not really a macro asset." It is worth understanding why those decouplings happen and why they are temporary. The crypto-equity correlation is not a law of physics; it is a *regime* feature. It runs highest precisely when macro is the dominant driver — during liquidity floods and drains, real-rate shocks, and risk-on/risk-off swings, which has been most of the time over the past five years. In those regimes Bitcoin and the Nasdaq are both just high-beta expressions of the same liquidity-and-real-rate tide, so they move together, often with a correlation above 0.6.

The correlation breaks when a *crypto-specific* shock temporarily overwhelms the macro signal: an exchange collapse like FTX, a regulatory crackdown, a major hack, or a halving narrative the crowd fixates on. In those windows, idiosyncratic crypto news drives the price more than macro does, and the correlation to equities falls or even briefly inverts. But these are the exceptions that prove the rule: once the crypto-specific shock is digested, the price snaps back to trading on the macro dials, because over any horizon longer than a few weeks the dominant force on a zero-cashflow, maximally-speculative asset is the cost and quantity of money. The trader's posture is to treat a macro-driven move as the base case, recognize a crypto-specific decoupling for what it is — temporary and idiosyncratic — and expect the macro correlation to reassert once the shock clears.

## The halving vs macro debate

No discussion of crypto is complete without the **halving** — the most cited, most mythologized event in the Bitcoin calendar — and the question of whether it, rather than macro, drives the price. A macro trader needs a clear-eyed answer.

### What the halving is

Bitcoin's supply schedule is fixed in code. New coins are created as a reward to "miners" who process transactions, and roughly every four years that reward is cut in half — the **halving**. This steadily reduces the rate of new supply until, eventually, the total caps out at 21 million coins, forever. Halvings have occurred in 2012, 2016, 2020, and 2024. Each one cuts the flow of new coins entering the market in half.

The halving narrative says: supply issuance drops, and if demand holds steady, less new supply means higher prices — so Bitcoin should rally in the 12-18 months after each halving. And indeed, large bull runs *have* followed each halving. The narrative looks confirmed by the chart. This is the strongest argument that crypto marches to its own internal clock, independent of macro.

### Why the macro lens explains it better

Here is the problem with the halving-drives-price story, and it is a problem of *magnitude and timing*. Consider the actual supply math. Around a halving, the reduction in new daily issuance is tiny relative to the total stock of coins already in existence and trading. New issuance is a fraction of a percent of the circulating supply and a *minuscule* fraction of daily trading volume. A halving cuts something that was already a rounding error in half — it makes a tiny number tinier. It is very hard to see how that small a supply change, fully known and scheduled years in advance, could move a trillion-dollar asset by hundreds of percent. Efficient markets should have priced a fully predictable, calendar-known supply cut long before it arrived.

Now consider what *else* was happening at each "halving bull run." The 2016 halving was followed by a 2017 run — during a period of easy global liquidity and the post-crisis low-rate regime. The 2020 halving was followed by the 2021 run — during the largest liquidity flood in history, with real rates at -1%. The 2024 halving was followed by a run to \$100,000 — during a liquidity-and-ETF-driven loosening with real rates beginning to fall. In every case, the "halving bull run" coincided with a **macro liquidity expansion**. The halving cycle and the macro liquidity cycle have, by coincidence of timing, run roughly in sync — and the macro force is enormous where the halving force is tiny.

![The crypto-macro driver map showing net liquidity, real rates, dollar, risk appetite, and the halving as a slow background](/imgs/blogs/crypto-as-a-macro-liquidity-asset-5.png)

The honest synthesis is this: the halving is a real, slow, structural *supply background* that gently supports the long-run scarcity thesis. But it is not the *trigger* of the cycle. The trigger of each violent boom and bust is the macro tide — liquidity, real rates, the dollar. When the next halving lands in a *tightening* macro regime instead of a loosening one, the halving narrative will be tested, and the macro lens predicts the halving will lose. Put bluntly: if you have to choose one lens, choose macro. It explains the magnitude; the halving cannot.

#### Worked example: the 2024 run past \$100,000

The 2024 cycle is the best test case for separating the two stories, because both a halving *and* a major macro catalyst arrived in the same year.

- **The halving.** Bitcoin's fourth halving occurred in April 2024, cutting new issuance again. On the pure halving narrative, this was the trigger.
- **The macro and structural catalysts.** Simultaneously, US spot Bitcoin ETFs launched in early 2024, opening a vast new channel of institutional and retail demand that could buy Bitcoin through an ordinary brokerage account — a genuine, large, *demand-side* shock. And the macro regime was turning supportive: inflation was cooling toward target, the Fed had stopped hiking and markets were pricing cuts, and real yields were rolling over from their 2023 peak. Net liquidity conditions were improving as the ON RRP cash pile drained back into markets.
- **The result.** Bitcoin's year-end close ran to **\$93,429** in 2024 (from \$42,265 at the end of 2023), briefly crossing **\$100,000** in December — a +121% year.
- **Attribution.** A tiny scheduled supply cut, or a multi-hundred-billion-dollar new demand channel plus a turning macro liquidity regime? The magnitude points overwhelmingly at the macro-and-ETF wave. The halving was the background; the liquidity-and-demand surge was the engine.

So even in the year the two stories overlapped most cleanly, the macro lens carries the weight and the halving is the slow undertone.

## The structural shift: ETFs and institutional ownership

One development deserves its own treatment because it has changed *how strongly* crypto trades on macro: the arrival of spot Bitcoin ETFs in early 2024 and the institutionalization that followed. Before 2024, buying Bitcoin meant opening a crypto-exchange account — a hurdle that kept most institutional and traditional-finance money out. The spot ETFs demolished that hurdle. A pension fund, a financial advisor, or a retiree could suddenly buy Bitcoin in a normal brokerage account with a ticker, the same way they buy any stock or commodity fund. Tens of billions of dollars flowed in within the first year.

The macro consequence is subtle but important. As Bitcoin's ownership shifted from crypto-native holders toward traditional asset allocators, its behavior became *more* tightly coupled to macro, not less — because those new holders manage it inside a macro framework. They buy it as a risk asset and a liquidity play, they trim it when real yields rise and the dollar strengthens, and they rebalance it alongside their stocks and bonds. The marginal buyer of Bitcoin is now far more likely to be someone running a macro book than a cypherpunk holding through anything. That is why the post-2024 era has seen Bitcoin trade *even more* like a high-beta macro asset: the ETF wave that carried it past \$100,000 was itself a liquidity-and-flows story, and the holders it brought in reinforce the macro sensitivity rather than dilute it.

For a trader, the practical upshot is that the macro framework in this post has become *more* reliable over time, not less. As crypto institutionalizes, the idiosyncratic, retail-driven, narrative-led moves that occasionally broke the macro link are being diluted by a growing base of holders who treat Bitcoin as exactly what this post argues it is — a high-beta proxy for global liquidity and real rates. The deeper history of how Bitcoin got here is in [Bitcoin and the cypherpunk vision](/blog/trading/finance/bitcoin-and-the-cypherpunk-vision); the point for macro trading is that the asset's macro beta is structurally rising, not fading.

## Common misconceptions

The macro view of crypto cuts directly against three beliefs that are widespread, intuitive, and wrong. Each is corrected with a number.

### Misconception 1: "Crypto is uncorrelated — it's a great diversifier and a hedge"

The pitch you hear is that Bitcoin is uncorrelated with stocks, so adding it to a portfolio improves diversification, and that it is a hedge against financial turmoil. The data says the opposite *when it matters*. Through 2021-2023, Bitcoin's rolling correlation with the Nasdaq frequently exceeded 0.5 and rose *higher* during stress — exactly when you would want a diversifier to zig while stocks zag, Bitcoin zigged harder in the same direction. Correlations are not constant: crypto can look uncorrelated during quiet, range-bound periods, which lulls people into the diversification story. But correlations go to 1 in a crisis, and crypto's go *above* 1 in beta terms. The number to remember: in the 2022 risk-off, the Nasdaq fell about 33% and Bitcoin fell **-65%** — same direction, twice the magnitude. That is not a diversifier; that is a leveraged version of the same trade.

### Misconception 2: "The halving drives the price independent of macro"

We covered the mechanism above; here is the number that settles it. The supply reduction at a halving is a fraction of a percent of circulating supply and a vanishing fraction of daily volume — a rounding error, cut in half, fully known years in advance. Meanwhile, the macro liquidity swings around each halving were measured in *trillions* of dollars of central-bank balance-sheet expansion and *percentage points* of real-rate movement. The 2022 crash happened with no halving anywhere near it — pure macro. The "halving bull runs" all coincided with liquidity floods. When you put the tiny, scheduled supply change next to the enormous, surprise macro moves, the attribution is not close. The halving is a slow background, not the trigger.

### Misconception 3: "It's digital gold — it protects you in a crisis"

The long-run aspiration is that Bitcoin becomes a crisis hedge like gold. The *in-the-moment* reality, so far, is the reverse. In an acute liquidity crisis, leveraged players sell whatever they can to raise cash, and the most liquid, highest-beta, most-speculative holdings go first. In March 2020, as the COVID panic hit, Bitcoin did not act like gold — it fell roughly 50% in a matter of days, *alongside* stocks, as everyone scrambled for dollars. Gold also wobbled but recovered far faster and is the genuine flight-to-safety asset in panic. Bitcoin behaved like the *riskiest* asset in the room, because in a funding crisis that is exactly what it is. The store-of-value thesis may mature over decades, but in the acute moment of a crisis, crypto is risk-on, not a hedge. Confusing the long-run aspiration with today's behavior is how people get caught holding the riskiest asset precisely when they thought they held the safest.

## How it shows up in real markets

Theory is only worth anything if it reads the tape. Let us walk the last cycle as three regimes, each defined by the macro dials, and watch Bitcoin do exactly what the framework predicts.

![Bitcoin year-end path from 2018 to 2025 with the 2021, 2022, and 2024 regime turns annotated](/imgs/blogs/crypto-as-a-macro-liquidity-asset-6.png)

### 2020-21: the liquidity flood and the everything rally

In the spring of 2020, the world's central banks opened the spigots all at once to fight the COVID shock. The Fed's balance sheet more than doubled toward \$8.96 trillion; policy rates went to zero; the 10-year real yield fell to about -1%. All three macro dials were screaming "tailwind": quantity flooding, real rates deeply negative, and an eventually-softening dollar. The far end of the risk spectrum lit up. Bitcoin went from under \$5,000 in March 2020 to nearly \$69,000 in November 2021, closing 2020 at \$28,990 and 2021 at \$46,306. It was the loudest instrument in the "everything rally," moving a multiple of the liquidity swing — exactly the high liquidity beta the framework predicts. A trader watching the three dials saw a tailwind so strong that the only question was sizing.

### 2022: the real-rate shock and the -65% crash

Then inflation arrived — CPI peaked at 9.06% in June 2022, a 40-year high. The Fed slammed the brakes, hiking from near zero to over 4% by year-end, and the 10-year real yield ripped from -1.04% to +1.74%. The dollar surged to a multi-decade high near 114.8. Now all three dials flipped to "headwind": real rates up sharply, dollar surging, and the Fed beginning to drain its balance sheet. The longest-duration asset in the market got repriced first and hardest. Bitcoin fell from \$46,306 to \$16,548 — a **-65%** crash — even though the Fed's balance sheet had barely begun to shrink. This is the cleanest possible proof that the *real-rate dial*, not just the quantity dial, drives crypto: the money supply was still near its peak, yet the price was cut by two-thirds, because the discount rate on a zero-cashflow asset had risen 2.8 points. A trader who was watching real yields saw this coming as a single move.

### 2024: the liquidity-and-ETF wave past \$100,000

By 2024, the regime turned again. Inflation had cooled toward target, the Fed had stopped hiking and begun cutting, and real yields rolled over from their 2023 peak near 2.5%. Net liquidity improved as the ON RRP cash pile drained back into markets, cushioning the effect of ongoing QT. And a structural demand shock arrived: US spot Bitcoin ETFs launched, opening a huge new buying channel. The dials turned back toward "tailwind" — and the halving in April added a slow supply background on top. Bitcoin ran from \$42,265 (end 2023) to \$93,429 (end 2024), briefly crossing \$100,000 in December. The macro tide turned, the highest-beta asset responded most, and the ETF channel amplified the demand — precisely the framework's prediction.

### The pattern across all three

Step back and the three regimes rhyme. Each turning point in Bitcoin's price was a turning point in the macro dials — liquidity, real rates, the dollar — and *not* a turning point in any on-chain metric or halving date that arrived out of sync with macro. The 2021 top, the 2022 bottom, and the 2024 run were all macro events. A trader analyzing individual crypto fundamentals saw a hundred unrelated stories; a trader watching three dials saw three moves. The whole edge is in being the second trader.

### The asymmetry that defines the asset

Pull the three episodes together and a single property defines Bitcoin as a macro asset: an extreme **asymmetry of beta** to the liquidity cycle. In the up-phase it does not merely rise with liquidity — it rises *multiples* faster, because it sits at the far end of the risk spectrum where the last marginal dollar arrives and where the buyers are most aggressive. In the down-phase it does not merely fall with liquidity — it falls *harder and first*, because it is the most speculative, most leveraged, most easily-sold thing in the book the moment funding tightens. The 2020-21 flood multiplied it more than tenfold; the 2022 drain cut it by roughly three-quarters; the 2024 wave multiplied it severalfold again. No major liquid asset has a wider amplitude to the same underlying tide. For a trader, that asymmetry is the entire opportunity *and* the entire danger: get the liquidity direction right and crypto pays more than any other expression of the view; get it wrong, or get the sizing wrong, and the same beta that made you rich in the flood wipes you out in the drain. Everything in the playbook below exists to harness the up-asymmetry while surviving the down-asymmetry — and the survival half is the one that actually decides whether you are still trading after a full cycle.

## How to trade it / the playbook

Everything above resolves into one disciplined playbook. The crypto-macro trader does two things: reads the three dials to set a directional bias, and sizes the position for the beta so the volatility does not destroy the account. Conviction and timing come last; *liquidity regime and size* come first.

![The crypto-macro playbook mapping net liquidity, real yields, and the dollar to position and size](/imgs/blogs/crypto-as-a-macro-liquidity-asset-8.png)

### Dial one: net liquidity (the quantity of money)

Watch **net liquidity** — the Fed's balance sheet minus the ON RRP minus the TGA — as a slow trend, not a daily wiggle. Read it as a rough 13-week direction. *Rising* net liquidity is a tailwind for the highest-beta asset on Earth; *falling* net liquidity is a headwind. The components: a shrinking balance sheet (QT) drains; a draining ON RRP *adds* cash back to markets (this is why 2023-24 felt loose despite QT); a rising TGA (Treasury rebuilding its cash) drains, while a falling TGA adds. The full mechanic, including how to compute it, is in [the central-bank balance sheet, net liquidity, reserves, RRP, and TGA](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga). Globally, watch the four major central banks' combined balance sheets in dollars, per [global liquidity](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide). **Signal: net liquidity trending up = lean long; trending down = cut size.**

### Dial two: real yields (the price of money)

Watch the **10-year TIPS real yield** as the single most dangerous dial for a zero-cashflow asset. Falling or deeply negative real yields are a powerful tailwind — they lower the opportunity cost of holding something that yields nothing and lower the discount rate on the far-future payoff. Rising real yields are the sharpest headwind there is, because of the maximal-duration effect; a 2.8-point rise in real yields cut Bitcoin by two-thirds in 2022 with the money supply barely moving. The full case is in [real vs nominal, inflation, real yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal). **Signal: real yields falling = strong tailwind; real yields rising toward +2% = strong headwind, respect it above everything else.**

### Dial three: the dollar (the funding condition)

Watch **DXY** as the funding-tightness gauge. A surging dollar tightens global funding and drains effective liquidity even if the Fed is still; a soft or falling dollar loosens it. The dollar's 2022 surge to ~114.8 ran alongside Bitcoin's collapse; dollar softness has accompanied every crypto rally. The full picture is in [the dollar system](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy). **Signal: DXY soft or falling = supportive; DXY surging = global tightening, cut risk.**

### Combining the dials into a bias

The three dials usually point the same way, because they are three faces of one regime — the global cost and quantity of money. The strongest *long* setup is rising net liquidity, falling real yields, and a soft dollar all at once: that is the 2020-21 and 2024-type tailwind, and it is when crypto's high beta works *for* you. The strongest *flat-or-short* setup is the mirror: draining net liquidity, rising real yields, and a surging dollar — the 2022-type headwind, when the beta works *against* you, hard. When the dials conflict, weight real yields most, because duration is crypto's deepest exposure. Notice what is *not* on this list: the halving calendar, on-chain metrics, and price-chart patterns. They are background, not triggers. The macro regime is the trigger.

### Sizing for the beta — the part that actually matters

Here is the discipline that separates survivors from casualties. Because crypto runs at roughly **2-3x** the liquidity beta of equities *and* several times their volatility, a position sized in *dollars* like a stock position carries *multiples* of the risk. The fix is to size by **risk budget**, not by dollar amount. If you would hold \$100,000 of a stock index for a given risk level, hold something closer to \$30,000-\$50,000 of crypto to carry the *same* risk — because in a real macro shock, crypto can fall 65% where stocks fall 33%. A crypto sleeve should be small precisely because each dollar is so loud. The right discipline is a tiny, volatility-budgeted sleeve that you can hold through a -65% drawdown without being forced to sell at the bottom; the trap is sizing by conviction in dollars and getting liquidated when the real-rate dial turns. You will be right about direction far more often than you survive being wrong about size.

### The invalidation

Every macro view needs a kill switch. The crypto-macro long thesis is invalidated when the dials turn decisively to headwind: net liquidity rolling over into a sustained drain, the 10-year real yield breaking higher and trending up, and the dollar surging. When those three line up, the highest-beta asset is the *last* thing you want to be long and over-sized in — that is the 2022 configuration, and it cost the over-sized everything. Conversely, do not fight a clear three-dial tailwind by shorting crypto on a halving-skepticism thesis; the macro tide overwhelms the supply story every time. The view flips with the regime, not with the price chart. Watch the dials; let them set the bias; size for the beta; and respect the real-rate kill switch above all. That is trading crypto with the macro lens — which, given how it actually trades today, is the only way to trade it without trading blind.

## Further reading & cross-links

- [Global liquidity: the world's money tide](/blog/trading/macro-trading/global-liquidity-the-worlds-money-tide) — the combined central-bank balance sheets plus the eurodollar system, and why every risk asset (Bitcoin included) rides the same tide.
- [Real vs nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — why the real interest rate, not the nominal one, is the discount rate that prices a zero-cashflow asset.
- [The central-bank balance sheet: net liquidity, reserves, RRP, and TGA](/blog/trading/macro-trading/central-bank-balance-sheet-net-liquidity-reserves-rrp-tga) — how to compute the net-liquidity dial that sets the crypto tailwind or headwind.
- [Bitcoin and the cypherpunk vision](/blog/trading/finance/bitcoin-and-the-cypherpunk-vision) — the long-run store-of-value thesis we deliberately parked, so you can hold both the slow trend and the violent macro cycle in view at once.
- [The dollar system: why USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — why a strong dollar tightens global funding and drains the highest-beta asset.
- [Risk-on, risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the risk spectrum down which money flows, with crypto at the far, highest-beta end.
- [Reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — the duration intuition that explains why a no-cashflow asset is the most rate-sensitive thing in the market.
