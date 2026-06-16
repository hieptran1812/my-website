---
title: "Liquidity and the Margin Cycle: The Fuel Behind Vietnamese Sector Waves"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How market liquidity and broker margin lending amplify sector rotation on VN-Index, why leverage makes the waves violent, and how to read the liquidity regime as a trader."
tags: ["vietnam-stocks", "margin", "liquidity", "sector-rotation", "vn-index", "leverage", "deleveraging", "brokers", "risk-management", "market-structure"]
category: "trading"
subcategory: "Vietnam Stocks"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — On VN-Index, liquidity *is* the story: a retail-dominated market running on broker margin lending means every sector wave is amplified by leverage, so cheap money lifts the high-beta sectors on the way up and forced deleveraging crushes the same sectors on the way down.
>
> - **Margin (ký quỹ) is borrowed money** from your broker to buy more stock than your cash allows. It magnifies both gains and losses, and the loan is fixed while the stock price floats, so every price move lands entirely on *your* equity.
> - **The margin cycle is reflexive**: rising prices mark up the collateral, which unlocks more borrowing room, which funds more buying, which lifts prices again — and the same loop runs violently in reverse when prices fall and margin calls force selling.
> - **Read liquidity as a regime gauge**: track average daily trading value (ADTV), total margin balance, and the market-wide margin-to-equity ratio. When all three hit records together, you are late in the cycle — size down.
> - **The one number to remember**: total broker margin balance went from ~52 trillion dong at end-2019 to a ~195 trillion peak in 2021, collapsed ~37% in the 2022 crash, then made fresh records past ~370 trillion dong (~\$14.6bn) by Q3 2025.

## A "căng margin" morning

It is the morning of April 25, 2022. VN-Index has been bleeding for three weeks, and over the previous fortnight a wave of high-flying property and speculative midcaps has cracked. This morning the index opens down, slides through the prior day's low, and then something mechanical takes over. Sell orders pile in at the floor price — the maximum daily down-limit of -7% on HOSE — on dozens of tickers at once. They are not discretionary sells. They are forced sells: brokers liquidating client accounts whose equity has fallen below the maintenance threshold on their margin loans. By the close, VN-Index has dropped nearly 5% in a single session, and the most "liquid" leaders — the names everyone owned because you could always get out of them — are precisely the ones nailed to the floor with no buyers underneath.

Vietnamese traders have a phrase for the state these accounts are in: *căng margin* — literally "margin is stretched," the white-knuckle condition where leverage is maxed out and a small adverse move triggers a chain reaction. What looked like a healthy, liquid bull market three months earlier — record turnover, record account openings, record margin balances at the brokers — had become a deleveraging machine. The very fuel that powered the 2021 melt-up (cheap broker credit pyramided onto rising collateral) became the accelerant of the 2022 collapse. Same sectors. Same stocks. Opposite sign.

This post is about that fuel. To understand why Vietnamese sector waves are so violent — why brokers, steel, property, and speculative midcaps can double in a quarter and halve in the next — you have to understand two things that are more important on VN-Index than on almost any developed market: **liquidity** (how much money is actually trading) and **margin** (how much of that money is borrowed). VN-Index is roughly 85-90% retail by volume, and a large share of that retail activity runs on borrowed money. That single structural fact is the engine behind the rotation. We will build up both concepts from zero, walk the margin machine step by step, show why leverage amplifies rotation, give you the three or four numbers that read the regime, and end with a concrete playbook for trading the liquidity cycle instead of being run over by it.

![Margin flywheel loop with up cycle in green and down cascade in red](/imgs/blogs/liquidity-and-the-margin-cycle-vietnam-1.png)

The figure above is the whole post in one picture. On the green path, a rising price marks up the collateral, which unlocks more margin room, which funds more buying, which lifts the price again — a self-reinforcing flywheel. On the red path, a falling price breaches the equity ratio, triggers a margin call, forces selling, and drives the price lower still — the same flywheel spinning backward. Everything between here and the playbook is an elaboration of these two loops and how to tell which one you are living in.

## Foundations: what liquidity and margin actually are

Before any mechanism, two definitions. Get these right and the rest follows.

### Liquidity: how easily you can turn stock into cash without moving the price

**Liquidity** is the ease with which you can buy or sell a position quickly, in size, without pushing the price against yourself. A liquid stock has a deep order book — lots of resting buy and sell orders stacked near the current price — so a big order gets filled near the last trade. An illiquid stock has a thin book, so the same order "walks" up or down through empty price levels and you get a terrible average price.

The market-wide proxy for liquidity is **trading value** — the total dong amount that changes hands. On VN-Index the headline number is **ADTV, average daily trading value**: add up the value of every trade on HOSE in a day (price times volume, summed across all tickers), then average over a period. ADTV is quoted in trillion dong per day. When ADTV is high, money is sloshing around, orders fill easily, and prices can run; when ADTV collapses, the market goes quiet, spreads widen, and even small sell orders can knock a stock down several percent because nobody is bidding.

A useful intuition before the numbers: **liquidity is the water level in a swimming pool.** When the pool is full (high ADTV), you can dive in anywhere and the water absorbs you. When the pool is half-drained (low ADTV), the same dive cracks your skull on the concrete — the same sell order that barely registered last month now sends a stock limit-down because there is no depth to absorb it. The water level moves the *whole pool* at once, which is why liquidity is a market-wide regime variable, not a stock-by-stock detail.

#### Worked example: reading depth versus turnover

Suppose a midcap trades 50 billion dong of value on an average day, and the best bid stack (the resting buy orders within ~2% of the last price) totals 5 billion dong. You want to sell a 2 billion dong position (~\$79,000 at \$1 = 25,400 VND). On a normal day, your 2 billion fits inside the 5 billion of nearby bids — you sell near the last price, paying maybe 0.3% in slippage. Now take a risk-off day: turnover on the stock falls to 12 billion dong and the nearby bid stack thins to 1.2 billion. Your same 2 billion dong order now exceeds the entire nearby bid, so you walk the book down — your average fill might be 4-5% below the last print, and the act of selling itself prints those lower prices for everyone watching. The lesson: liquidity is not a property of your order, it is a property of the book on the day you trade, and it evaporates exactly when you most need it.

### Margin: borrowing from your broker to buy more stock

**Margin** — *ký quỹ* in Vietnamese, literally "deposit/collateral" — is buying stock with a mix of your own cash and money borrowed from your broker, using the stock you buy (plus any stock already in the account) as collateral for the loan. If you put up 100 million dong of your own cash and the broker lends you another 100 million, you can hold 200 million dong of stock. You have doubled your exposure. That is leverage.

Three terms define a margin account:

- **The loan (the borrowed amount).** Fixed in dong. It does not shrink when the stock falls; you owe the broker the same amount plus interest regardless of what the market does. This fixity is the whole source of the danger — we will return to it.
- **The equity (your skin in the game).** Equity = current market value of the stock minus the loan. It floats with the price. When the stock rises, all of the gain accrues to your equity (the loan is fixed). When the stock falls, all of the loss comes out of your equity *first*. This asymmetry is leverage in one sentence.
- **The maintenance ratio (the floor).** The broker requires your equity to stay above some minimum fraction of the position's value — commonly around 30% in Vietnam, though it varies by broker and by stock. **Equity ratio = equity ÷ market value of the position.** If a falling price pushes this ratio below the floor, the broker issues a **margin call**: post more cash within a short window (often by the next session), or the broker force-sells your stock to bring the ratio back into line.

A homely analogy: margin is a **mortgage on a house you flip.** You put 20% down, the bank lends 80%, and you own a house worth far more than your cash. If the house appreciates 10%, your equity (the 20%) jumps ~50% — leverage working for you. If it drops 10%, your equity gets *halved*, and if it drops far enough the bank can demand you top up or it forecloses. The stock-margin version just runs on a daily mark-to-market and a much faster trigger: there is no slow foreclosure process, the force-sell happens in a single session.

The reason all of this matters for *sectors* — not just individual gamblers — is scale. When a large fraction of an entire market is running on margin, the maintenance floors of millions of accounts are bunched together. A market-wide price drop trips all of them at once, and the forced selling that results is itself a market-wide event that drives prices lower, tripping more floors. The micro-mechanism (one account's margin call) becomes a macro-mechanism (a deleveraging cascade across the whole index). That is the bridge from the individual to the wave.

### The cost of leverage: interest, not just risk

There is a second, slower cost to margin that beginners overlook: **you pay interest on the loan every day you hold it.** Vietnamese brokers charge annual margin rates that have historically run in the low-to-mid teens — call it roughly 12-14% a year, varying by broker and by promotional period. That sounds modest until you annualize it against a short holding period. A 13% annual rate is about 0.036% *per calendar day*; hold a leveraged position for two months and you have paid roughly 2.1% of the *borrowed* amount in interest, win or lose. This is the *carry cost* of leverage, and it has two consequences for the cycle. First, it means margin is only worth using when you expect the stock to move *up faster than the interest bleed* — it is a tool for momentum, not for patient holding, which is part of why leveraged money concentrates in fast-moving high-beta sectors rather than slow defensives. Second, the interest is itself a revenue line for the brokers; a market running a 370 trillion dong margin book at ~13% is generating on the order of 48 trillion dong a year in margin interest for the securities industry, which is exactly why broker earnings are so geared to the size of the margin balance.

### T+ settlement: the timing wrinkle that interacts with margin

One Vietnam-specific detail sharpens the leverage dynamic: **settlement timing**, the famous *T+* rule. When you buy a stock, the shares do not arrive in your account instantly — they settle a couple of business days later (the market moved to a faster cycle over time, but a settlement lag remains). Practically, this means a stock you buy today is not freely sellable until it settles, and money has to clear before it can be redeployed. For a leveraged trader this lag matters in a falling market: if your account gets a margin call but the shares you would sell to meet it are still settling, you can be forced to sell *other*, already-settled holdings — often your most liquid, highest-quality names — to raise cash fast. The settlement lag thus *concentrates* forced selling in the liquid leaders even more, because those are the settled, sellable assets a stressed account can actually liquidate today. It is a small piece of plumbing, but it is one more reason the cascade hits the leaders hardest.

#### Worked example: the interest bleed on a flat trade

You borrow **100 million dong** on margin at a **13% annual rate** (~\$3,940 of borrowed money at \$1 = 25,400 VND) and hold for **45 calendar days**, during which the stock goes *nowhere* — flat. The interest accrues at 13% × (45 ÷ 365) ≈ **1.60%** of the borrowed 100 million, or about **1.6 million dong**, with no offsetting gain. On your own 100 million of equity, that flat-but-leveraged trade has cost you **−1.6%** purely in carry. Now imagine the stock fell 5% over those 45 days: the position lost 5% × 200 million = 10 million dong *plus* the 1.6 million interest, so your equity dropped from 100 million to ~88.4 million — a **−11.6% hit** on your own money from a 5% stock decline. The intuition: leverage charges rent whether or not you are right, so a leveraged book that merely treads water in a choppy, range-bound market quietly bleeds — which is one reason leveraged investors are itchy to sell into the first real weakness, accelerating the unwind.

## The Vietnamese margin machine

Vietnam's market is unusually leverage-driven, and the plumbing is specific. Let us walk it.

### Who lends, and against what

In Vietnam, margin loans come from **securities companies (brokers)**, not from banks directly. The big lenders are the large brokerages — names like SSI, VND (VNDirect), HCM (HSC), VCI (Vietcap), MBS, SHS, VIX, and a handful of others; the market's margin lending is concentrated in roughly the **top 10-13 firms**. When you open a margin account, the broker extends you credit and holds your shares as collateral. The broker, in turn, funds that lending from its own equity capital plus short-term borrowing (interbank, bonds, bank credit lines). So a broker's margin book is a leveraged carry: it borrows at one rate, lends to clients at a higher rate, and pockets the spread.

Not every stock can be bought on margin. Each broker (within regulatory bounds) publishes a **margin list** of eligible tickers and an allowed loan ratio per ticker. Blue-chip large-caps with deep liquidity get generous ratios; thin, volatile small-caps get low ratios or are excluded entirely. Newly listed stocks, stocks under warning, and stocks with poor financials are typically barred. This matters for rotation: when a sector is "marginable" at high ratios, leverage flows into it freely; when a regulator or a broker cuts a stock's margin ratio (often after a sharp run), it can puncture the move on its own.

### The regulated limits

Two limits cap how much leverage the system can build:

- **The per-account loan-to-asset cap.** Regulation limits how much a client can borrow relative to the account's total assets — historically the initial margin ratio meant you could borrow up to roughly 1:1 against eligible collateral (i.e., put up ~50% and borrow ~50%), giving up to ~2x gross exposure. Brokers can be more conservative.
- **The broker's own margin-to-equity cap.** A securities firm's *total* outstanding margin loans are capped at a multiple of its own equity (the well-known limit is **200% of owners' equity** — margin loans cannot exceed twice the firm's capital). This is the systemic ceiling. It is also why, in big bull markets, you see a wave of **capital raises by brokers**: they hit the 2x cap, can lend no more, and must issue new shares to expand the margin book. A flurry of broker rights issues is itself a late-cycle liquidity tell.

#### Worked example: how margin doubles both gains and losses

Take a clean case. You have **100 million dong** of your own cash (~\$3,940 at \$1 = 25,400 VND). You buy on margin at 1:1, so the broker lends another 100 million and you hold **200 million dong** of stock. Now two scenarios:

- **The stock rises 10%.** Your 200 million position becomes 220 million. The loan is still 100 million, so your equity is 220 − 100 = **120 million**. You started with 100 million of equity; you now have 120 million. That is a **+20% return on your money** from a +10% move in the stock. Leverage doubled the gain.
- **The stock falls 10%.** Your 200 million position becomes 180 million. The loan is still 100 million, so your equity is 180 − 100 = **80 million** — a **−20% return on your money** from a −10% move. Leverage doubled the loss.

The intuition: because the loan is fixed, the *entire* swing in the position value lands on your equity, so a 2x leveraged position turns every 1% move in the stock into a 2% move in your account.

![Margin call before and after a 20 percent price fall showing equity ratio breach](/imgs/blogs/liquidity-and-the-margin-cycle-vietnam-4.png)

### A sharper edge: pledged shares and insider leverage

There is a particularly dangerous form of leverage hiding inside the Vietnamese market: **founders and major shareholders pledging their own stock for loans.** A company's chairman or controlling family will often pledge a large block of their shares as collateral to borrow — sometimes through a broker margin account, sometimes through a bank — to fund other ventures or to buy yet more shares. This is leverage on the *control block*, not just on a retail trader's small position, and it is far more destabilizing. When the stock falls and the pledged block faces a call, the forced sale is enormous relative to the float, and because the seller is the insider, it sends a devastating signal to the market. In the 2022 crash, several high-profile property and conglomerate names cratered partly because pledged insider blocks were force-sold, which both crushed the price and shattered confidence. The tell to watch is disclosed pledge ratios — when a large fraction of a controlling stake is pledged, the stock carries hidden cascade risk that does not show up in the company's own balance sheet. This is leverage at the most concentrated, least visible layer of the market, and it is one reason individual high-beta names can fall far harder than the sector average in a deleveraging event.

### Why brokers' own earnings ride the cycle

Here is a second-order effect that makes the cycle even sharper: **the brokers are themselves a high-beta sector.** A securities firm makes money three ways — brokerage commissions (a cut of every trade), margin interest (the spread on the loan book), and proprietary trading/investment gains. All three balloon in a bull market: turnover is high (more commissions), the margin book is maxed out (more interest), and the firm's own equity book is rising (prop gains). All three collapse in a bear: turnover dries up, the margin book shrinks as clients deleverage, and prop positions take losses. So broker stocks are a *leveraged play on liquidity itself.* They are usually the first sector to run when a bull starts (rising turnover lifts their earnings outlook before the rest of the market reprices) and among the first to crack when liquidity rolls over. We will see this pattern repeatedly in the case studies. For the mechanics of how a securities firm makes money, the companion post [inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) walks the revenue lines in detail.

## Why leverage amplifies rotation

We have the pieces. Now assemble them into the central claim: **leverage is why Vietnamese sector waves are so violent**, and why the *same* sectors lead both up and down.

### The same money buys more on the way up

Start with a market where everyone holds cash. A bullish catalyst arrives — say the State Bank cuts policy rates, making the dong cheap (the mechanics of that are covered in [Vietnam monetary policy: the State Bank, the dong, and the credit ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling)). Prices rise. In an all-cash market, the buying power is capped at the cash on hand. In a margin market, something extra happens: **as prices rise, the collateral value of stock already held rises, which unlocks additional borrowing room, which funds additional buying.** A 100 million dong portfolio that was fully invested can now, after a 30% gain, support more margin — the investor pyramids, buying more on the way up. Across millions of accounts, this turns a normal rally into a melt-up, and it concentrates in whatever sectors are most marginable and most liked: brokers, steel, property, speculative midcaps. These are the **high-beta** sectors — the ones that move more than the index in both directions.

### Forced selling on the way down

Now the reverse. A bearish catalyst hits — a rate hike, a bond-market scare, a geopolitical shock. Prices fall. In an all-cash market, falling prices simply mean paper losses; holders can sit. In a margin market, falling prices eat into equity, and once equity ratios breach the maintenance floor, **the selling is no longer voluntary.** The broker force-sells to protect its loan. And brokers force-sell what they *can* sell quickly — the liquid large-cap leaders — because dumping an illiquid small-cap into a thin book just nails it limit-down with no fill. So the forced selling concentrates in the very names that led on the way up. Those names gap lower, which marks down collateral across other accounts, which trips *more* floors, which forces *more* selling. The down-move feeds itself.

The asymmetry is the key insight: **on the way up, leverage is a choice that amplifies gains; on the way down, deleveraging is a compulsion that amplifies losses.** Nobody is forced to buy more as prices rise — but everyone is forced to sell as prices fall through the floors. That asymmetry is why VN-Index crashes are faster and more violent than its rallies, and why the high-beta sectors that lead the rally lead the crash.

### Why these specific sectors move most

Why brokers, steel, property, and speculative midcaps, and not, say, utilities or consumer staples? Two reasons stack on top of each other: *fundamental* beta and *flow* beta.

**Fundamental beta** is how much a sector's actual earnings swing with the cycle. Brokers, as we saw, earn from turnover, margin interest, and prop trading — every line is geared to liquidity, so their earnings can triple in a bull and vanish in a bear. Steel earnings ride the gap between volatile global commodity prices and fixed-ish costs, so margins balloon when prices rise and collapse when they fall — operationally leveraged. Property developers run on debt and recognize profit lumpily on project handovers, so their earnings and their balance sheets are both rate-sensitive and credit-sensitive; cheap money inflates them, tight money or a bond scare deflates them. These sectors *deserve* big price swings because their fundamentals genuinely swing.

**Flow beta** is the leverage amplification on top. Because these are the liked, liquid, high-momentum names, they are exactly what marginable money pours into on the way up and exactly what brokers force-sell on the way down. The flow magnifies the already-large fundamental swing. A defensive sector like utilities has low fundamental beta (regulated, steady cash flows) *and* low flow beta (margin money does not chase it because it does not move fast enough to beat the interest carry), so it stays quiet through the cycle. The high-beta sectors get hit by both barrels at once — which is why a 40% index decline can mean a 70% decline in the broker and property names. When you overweight or underweight "the high-beta sleeve" in the playbook below, this is the cluster you mean.

#### Worked example: the margin call that liquidates your stock

Return to the 200 million dong position (100 million cash + 100 million loan), and set the maintenance ratio at **30%**. The position falls. We need to find the price drop that triggers the call.

The equity ratio is (market value − loan) ÷ market value = (V − 100) ÷ V, where V is the current position value. Set this equal to 30%:

- (V − 100) ÷ V = 0.30  →  V − 100 = 0.30V  →  0.70V = 100  →  **V ≈ 142.9 million dong.**

So when the position falls from 200 million to about 143 million — a **drop of roughly 28.5%** — the equity ratio hits the 30% floor and the margin call fires. At that point your equity is 142.9 − 100 = 42.9 million, down from your original 100 million. To restore the ratio to a safe level, the broker force-sells stock to repay part of the loan. If the broker sells enough to bring the loan down so the ratio returns to, say, 40%, it must liquidate a large slice of the position into a falling market — locking in the loss at the worst possible price. The intuition: with the loan fixed, a ~28% price drop wipes out more than half your equity *and* hands control of your selling to the broker, who sells into the very weakness that caused the call.

(The before/after figure above used a 20% drop against a 30% floor as a starting illustration: a 20% fall takes the 200 million position to 160 million, equity to 60 million, and the ratio to 37.5% — close to but not yet through the floor; push the drop a little further to ~28% and the floor breaks. The exact trigger depends on the broker's specific maintenance ratio and how much existing equity cushion the account carries.)

## Reading the liquidity regime

If liquidity is the fuel, you want a fuel gauge. Four readings, taken together, tell you whether the tank is filling or draining.

### ADTV — the turnover trend

The first and simplest gauge is **average daily trading value.** Rising ADTV means money is flowing in and risk appetite is building; falling ADTV means money is leaving and the market is going to sleep. The level matters too: a market doing 25 trillion dong a day is a different animal from one doing 7 trillion. ADTV tends to *lead* — turnover expands before the index makes its high (new money arrives first) and contracts before the index makes its low (money leaves first). Watch the trend and the absolute level together.

![HOSE average daily trading value area chart 2019 to 2025](/imgs/blogs/liquidity-and-the-margin-cycle-vietnam-2.png)

The chart above is the cleanest single picture of Vietnamese liquidity. ADTV on HOSE ran around 4.7 trillion dong/day in 2019, roughly 7.4 trillion in 2020, then exploded to ~21.6 trillion in 2021 as a wall of new retail money arrived — about **4.6x the 2019 level** in two years. It dipped through the 2022 deleveraging, recovered through 2023-2024, and pushed to fresh highs around 25 trillion in 2025. The shape of that curve is the shape of the bull-bear cycle.

### Total margin balance — the leverage tide

The second gauge is the **total margin balance** outstanding across all brokers — the sum of every margin loan in the system. This is the most direct measure of how much borrowed money is in the market. Rising margin balance means investors are leaning into leverage (risk-on, but also building fragility); a sharp fall means forced or voluntary deleveraging (risk-off). The brokers disclose their margin books quarterly, and aggregators like FiinTrade compile the system total.

![Total broker margin balance line chart 2019 to 2025 quarters](/imgs/blogs/liquidity-and-the-margin-cycle-vietnam-3.png)

The trajectory tells the story of the cycle in one line. Total margin balance climbed from ~52 trillion dong at end-2019 to a **~195 trillion peak in 2021** as the bull blew off, then **collapsed ~37% to ~122 trillion by end-2022** as the crash forced deleveraging, then rebuilt to ~180 trillion (2023) and ~245 trillion (2024), and made **fresh records past 280, 304, and ~370 trillion dong** through the first three quarters of 2025. At ~370 trillion dong, the system's margin book is around **\$14.6bn** (at \$1 = 25,400 VND) — an all-time high, and a number worth keeping in front of you.

### Margin-to-equity — stretched or slack

The third gauge normalizes margin by the *equity* it is leaning on. A raw margin balance of 370 trillion dong is alarming in a small market and unremarkable in a huge one; what matters is margin relative to the equity base supporting it. The **market-wide margin-to-equity ratio** (total margin loans ÷ aggregate investor equity, or a broker-level version, margin ÷ owners' equity) tells you whether the leverage is *stretched* or *slack*. When margin debt approaches or exceeds the equity behind it, the system is fragile: a small price drop knocks out a large fraction of the equity cushion, and the floors trip en masse.

![Illustrative market margin-to-equity horizontal bar chart at four dates](/imgs/blogs/liquidity-and-the-margin-cycle-vietnam-7.png)

The illustrative chart contrasts a few regimes: at the **end-2021 blow-off top**, margin-to-equity was extreme (around 120% on this illustrative measure — margin debt exceeding the equity base, deeply stretched); after the 2022-2023 shock and deleveraging it fell to a slack ~69% (post-shock, lots of dry powder); it rebuilt to ~85% through 2024 and pushed toward ~95% by Q3 2025 — stretched again, though not yet at the 2021 extreme. The reference line at 100% marks where market margin debt equals market equity; above it, the system has very little cushion. (These specific percentages are illustrative — the *direction and regime* are the point, not the decimal.)

### New account openings — the marginal buyer

The fourth gauge is **new brokerage account openings**, reported monthly. Because VN-Index is so retail-driven, the flow of new accounts is a direct read on the marginal buyer. A surge in new accounts (the 2021 bull saw monthly account openings hit records) means fresh, often inexperienced and often leveraged money is arriving — bullish for momentum, but a classic late-cycle euphoria signal when it spikes. A collapse in new openings means the retail bid is exhausted.

**Putting the four together:** the regime is *risk-on and early* when ADTV is rising from a low base, margin balance is recovering off a trough, margin-to-equity is slack, and account openings are picking up. The regime is *risk-on but late and fragile* when all four are at or near records simultaneously — high ADTV, record margin balance, stretched margin-to-equity, euphoric account openings. That late-stage configuration is the single most reliable "size down" signal the Vietnamese market gives you. The companion macro post on the [business cycle and investment clock for Vietnam](/blog/trading/vietnam-stocks/business-cycle-investment-clock-vietnam) maps these liquidity readings onto where the broader economy sits in its cycle.

#### Worked example: reading the regime and flagging "size down"

Suppose you are watching the tape in a strong bull. Over six months, HOSE ADTV climbs from **7 trillion dong/day to 22 trillion dong/day** — that is roughly **\$0.28bn to \$0.87bn** of daily turnover (at \$1 = 25,400 VND), a ~3x jump. Simultaneously, the total margin balance prints a fresh all-time record, broker after broker announces a rights issue to expand its lending capacity (they have hit the 2x equity cap), and new account openings spike to a monthly high. None of these is individually a sell signal — they are all "bullish" on the surface. But *together*, at records, they describe a tank that is nearly full: there is little incremental leverage left to add, the marginal buyer is the most stretched, and the system has minimal cushion. The disciplined read is **late-stage — cut position size and tighten stops**, not "the trend is strong, add more." The intuition: liquidity indicators are most dangerous precisely when they are most bullish, because a full tank can only drain.

## The reflexive loop

We keep using the word *reflexive*. It deserves its own section, because it is the deepest reason the margin cycle is so powerful — and it is the idea George Soros built a career on.

Ordinary supply-and-demand thinking treats price as an *output*: fundamentals determine value, and price gravitates toward value. **Reflexivity** says that in a leveraged market, price is also an *input*: the price level itself changes the fundamentals that supposedly determine it. In the margin cycle, the channel is collateral. A higher stock price means more valuable collateral, which means more borrowing capacity, which means more buying, which means a higher price. Price feeds the very buying power that drives price. The loop has no natural anchor — it can run far above any sensible valuation on the way up, and far below it on the way down, because the feedback overwhelms the fundamentals on both sides.

#### Worked example: reflexivity unlocks fragile buying power

You hold a stock worth **100 million dong** that the broker margins at a 50% loan ratio, so it supports **50 million dong** of borrowing. The stock rises **30%** to 130 million. Now the same position supports 50% of 130 million = **65 million dong** of borrowing — an extra **15 million dong** of buying power unlocked purely by the price rise, with no new cash deposited (~\$590 of fresh firepower conjured from a paper gain). You use it to buy more, helping push the price up further, unlocking still more room. This is the green flywheel in action. But notice *why it is fragile*: that extra 15 million is borrowed against an unrealized gain. If the stock gives back the 30%, the collateral shrinks back to 100 million, the supportable loan falls back to 50 million, and you are now *over*-borrowed by 15 million — exactly the condition that triggers a forced sale. The intuition: reflexive buying power is real on the way up and vanishes on the way down, which is what converts a smooth rally into a violent unwind.

![Deleveraging cascade pipeline from index drop to forced selling to more margin calls](/imgs/blogs/liquidity-and-the-margin-cycle-vietnam-6.png)

The cascade figure traces the red loop step by step: a sharp index drop breaches equity ratios, which fires margin calls, which forces selling of the liquid large-cap leaders (because those are what brokers can actually sell into a falling market), which gaps those leaders lower, which trips more accounts' floors, which fires more calls. Each turn of the loop is mechanical, not emotional — and that is exactly why it does not stop at "fair value." It stops when the forced selling is exhausted, which is usually well below any level a fundamental investor would have predicted. The mirror image of this cascade on the way up is what builds the bubble; the reflexive loop is symmetric in structure but asymmetric in *speed*, because forced selling is faster and more concentrated than voluntary buying.

![Margin balance and VN-Index dual axis chart 2020 to 2024 rising and falling together](/imgs/blogs/liquidity-and-the-margin-cycle-vietnam-5.png)

If reflexivity is real, margin and price should rise and fall *together* — and they do. The dual-axis chart overlays total margin balance (left axis) on the VN-Index year-end level (right axis) from 2020 to 2024. They peak together in 2021 (margin ~195 trillion, index ~1,498), collapse together in 2022 (margin ~122 trillion, index ~1,007), and rebuild together through 2023-2024. The two lines are not independent — margin is both a *cause* of the index move (more leverage funds more buying) and a *consequence* of it (a higher index unlocks more margin). That two-way causation is reflexivity made visible.

The practical payoff of grasping reflexivity is that it tells you *what kind* of mistake the market is making. A non-reflexive thinker looks at a stretched, expensive market and says "this is overvalued, it must mean-revert soon" — and gets run over, because the reflexive loop can push price far above value for a long time as rising collateral keeps funding more buying. The reflexive thinker instead asks "is the feedback loop still being *fed*?" — is cheap money still arriving, is margin still expanding, are new accounts still opening? As long as the loop is fed, the overvaluation can extend; the warning is not "it is expensive" but "the fuel that feeds the loop is running out" (rates turning up, margin hitting the system cap, account openings rolling over). Symmetrically, in a crash the reflexive thinker does not catch the falling knife just because the market looks cheap; they wait for the forced selling — the *fuel of the downward loop* — to be exhausted. Reflexivity reframes the whole game from "price versus value" to "is the feedback loop being fed or starved," which is precisely the question the liquidity dashboard answers.

## Common misconceptions

A few beliefs about liquidity and margin are widespread in Vietnamese retail circles, and each one gets investors hurt. Here they are, corrected with numbers.

**Misconception 1: "Margin is just for gamblers — serious investors don't use it."** Margin is a tool, not a vice; the danger is not its existence but its *size relative to the market* and its concentration at the wrong time. The problem on VN-Index is not that some traders use 2x leverage — it is that a record fraction of the *whole market* is leveraged simultaneously near a top, so the forced-selling floors are all bunched together. A disciplined investor running modest leverage in an early-cycle low-margin environment faces very different risk from one running the same leverage when the system-wide margin balance is at an all-time high. The number that matters is not "do you use margin" but "what is the market's margin-to-equity, and where are we in the cycle." Same leverage, opposite risk, depending on the regime.

**Misconception 2: "High liquidity means the market is safe — I can always get out."** This is exactly backwards at turning points. High liquidity *on the way up* is comforting, but liquidity is a fair-weather friend: it is highest when everyone wants to buy and evaporates precisely when everyone wants to sell. In the April 2022 cascade, the most "liquid" leaders were the ones pinned limit-down with **no bid** — you could not get out at any price near the last print. Liquidity is not a constant property of a stock; it is a state of the order book that collapses in a forced-selling event. The time you most want to sell is the time the depth is gone. Plan your exit when liquidity is abundant, not when you finally need it.

**Misconception 3: "Rising margin balance is always bullish — it means people are confident."** A *rising* margin balance off a low base is indeed an early-cycle positive: leverage is rebuilding, risk appetite is returning. But a margin balance at an **all-time record**, with margin-to-equity stretched and brokers raising capital to lend more, is a late-cycle *warning*, not a green light. The same rising line means opposite things depending on level and context. In 2021, "margin keeps making new highs" was cited as bullish right up to the top; the record balance was the fuel that made the 2022 collapse so violent. The level and the rate of change matter more than the direction alone.

**Misconception 4: "If I don't use margin, the margin cycle doesn't affect me."** Even an all-cash investor is fully exposed to *other people's* leverage. The forced selling in a deleveraging cascade marks down the price of the stocks *you* hold, regardless of how you financed them. The 2022 crash punished cash and margin investors alike on the way down — the difference is that the cash investor could choose to hold through it, while the margin investor was force-sold at the bottom. You cannot opt out of the cascade's price impact; you can only choose whether you are a forced seller or a patient holder when it hits.

**Misconception 5: "Brokers are a defensive, steady-fee business."** On VN-Index, broker stocks are among the *highest-beta* sectors, not defensive ones. Their earnings (commissions + margin interest + prop gains) are a triple-levered bet on liquidity itself, so they soar in bulls and crater in bears. Treating a securities firm like a stable financial utility is a category error; it is the purest play on the very liquidity cycle this post is about.

## How it shows up on VN-Index

Theory is cheap. Here is the margin cycle in the actual record of VN-Index, with dates and numbers.

### The 2021 liquidity boom

After the COVID crash of March 2020, the State Bank cut rates aggressively and the dong got cheap. Bank deposit rates fell to multi-decade lows, and a generation of Vietnamese savers — locked at home, watching deposit yields collapse — discovered the stock market. The result was a retail tidal wave. Monthly new brokerage accounts hit records; HOSE ADTV climbed from ~7.4 trillion dong/day in 2020 to **~21.6 trillion in 2021** (with peak sessions far higher); and the **total margin balance rocketed from ~90 trillion dong at end-2020 to a ~195 trillion peak in 2021** — more than doubling in a year. The high-beta sectors led: broker stocks (the direct play on turnover and margin), steel (HPG and the smaller mills, riding a global commodity boom), property and speculative midcaps. VN-Index climbed from ~1,103 at end-2020 to ~1,498 at end-2021, with an intraday peak above 1,500 in early 2022. It was a textbook reflexive melt-up: cheap money → rising prices → rising collateral → more margin → more buying → rising prices. The fuel tank was full to the brim.

### The 2022 forced-deleveraging crash

Then the loop reversed. Global central banks began hiking in 2022; the State Bank followed, raising rates and tightening dong liquidity. On top of that, a domestic **corporate bond crisis** (*trái phiếu doanh nghiệp*) erupted in late 2022 — high-profile defaults and a regulatory crackdown froze a funding channel that property developers and other issuers had relied on, and the scare bled straight into the equity market. As prices fell, the reflexive loop spun backward. Margin calls fired; brokers force-sold the liquid leaders; the leaders gapped down; more accounts breached their floors; more selling. VN-Index fell from ~1,500 in early 2022 to a **low around 874 in November 2022** — a drop of more than 40% — and the **total margin balance collapsed ~37% from the ~195 trillion peak to ~122 trillion by year-end** as the system deleveraged. The high-beta sectors that led the bull led the bust: brokers, property, and speculative midcaps fell hardest, many losing 60-80% from their highs. The same names. The same leverage. The opposite sign. Anyone who had read "record margin balance" as purely bullish in late 2021 learned in 2022 what that record balance was actually measuring: the size of the eventual forced-selling wave.

The *texture* of that decline is worth dwelling on, because it is the cascade figure made real. The selling did not come in a smooth slide; it came in violent clusters of floor-price sessions where dozens of high-beta tickers were nailed limit-down at -7% simultaneously with no bid underneath. Those clustered floor days are the fingerprint of forced selling: discretionary sellers spread their exits, but margin-call selling is synchronized — it fires whenever the index breaks a level that trips a critical mass of accounts at once. Between the clusters there were sharp relief rallies (every leveraged unwind has them, as shorts cover and bottom-fishers step in), and each relief rally sucked in buyers who thought the worst was over, only to be force-sold in the next leg. The deleveraging took *months* to fully work through, because the margin balance had to fall from 195 to 122 trillion dong — roughly 73 trillion dong of borrowed money (~\$2.9bn at \$1 = 25,400 VND) had to be liquidated, and there is no way to sell that much leveraged stock into a falling market quickly without crushing the price. That is the difference between a normal correction and a leverage unwind: a correction ends when valuations look cheap; a leverage unwind ends only when the borrowed money is actually gone.

### The 2024-2025 new records

The system rebuilt. Through 2023 and 2024, falling rates and recovering confidence pulled the margin balance back up — to ~180 trillion dong (end-2023) and ~245 trillion (end-2024) — and ADTV recovered toward 21 trillion dong/day. Then 2025 took it to fresh extremes: the total margin balance pushed past **280 trillion (Q1), 304 trillion (Q2), and ~370 trillion dong (Q3)** — all-time records, around **\$14.6bn** at the system level — while ADTV ran near 25 trillion dong/day. By Q3 2025 the margin-to-equity gauge had climbed back toward the stretched end of its range (illustratively ~95%, approaching but not yet at the 2021 extreme). The lesson of the cycle is not that records guarantee a crash tomorrow — a stretched market can stay stretched and even grind higher for a while — but that a market making fresh margin records is, by construction, carrying maximum fragility. The fuel that powers the next leg up is the same fuel that will accelerate the next deleveraging. Reading where you are in this cycle is the single most valuable liquidity skill on VN-Index.

These liquidity dynamics interact with the foreign flows discussed in [foreign flows, ETFs, and the index effect on Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam) and with the sector leadership patterns in [sector rotation explained: leaders and laggards](/blog/trading/vietnam-stocks/sector-rotation-explained-leaders-and-laggards) — margin and foreign money are the two big external taps, and the high-beta sectors sit at the intersection of both.

## The playbook: trading the liquidity regime

Everything above is diagnosis. Here is what to *do* with it. The core idea is simple: **treat liquidity as a regime variable that sets your aggressiveness, and let the high-beta sectors be the lever you push or pull.** When the regime is early and fuel is filling the tank, you can lean into the high-beta names. When the regime is late and the tank is full, you size down before the forced sellers arrive.

### The signals to watch (your liquidity dashboard)

Keep four numbers in front of you, updated as they print:

- **ADTV trend and level.** Rising off a low base = early, lean in. Plateauing at a record while the index stalls = distribution, get cautious. Falling = money leaving, defensive.
- **Total margin balance.** Recovering off a trough = constructive. At an all-time record, especially with brokers announcing rights issues to lend more = late-cycle, raise cash.
- **Margin-to-equity (stretched vs slack).** Slack (well below the prior cycle's high) = the system has cushion, drops get bought. Stretched (near or above the prior peak) = little cushion, drops cascade.
- **New account openings.** Picking up off a low = fresh buyers arriving. Spiking to euphoric records = late-cycle warning.

When these line up bullishly *and* at low-to-mid levels, the regime is risk-on — overweight the high-beta cyclicals (brokers first, then steel, property, midcaps). When they line up at *records simultaneously*, the regime is late — that is your cue to trim the high-beta names back to neutral or underweight, regardless of how strong the trend feels.

### Position sizing — the heart of it

The single most important application is **sizing**. Your position size in high-beta sectors should be *inversely* related to how stretched the margin regime is. Concretely:

- **Early cycle (margin slack, ADTV rising off a low):** you can run a full-size, even modestly leveraged, allocation to brokers and cyclicals. The reflexive loop is working *for* you, and drops get bought.
- **Mid cycle (margin rebuilding, ADTV healthy):** full size, but no incremental leverage. Let the trend run; keep some dry powder.
- **Late cycle (margin at records, margin-to-equity stretched, account openings euphoric):** cut high-beta exposure to half or less, eliminate your own margin entirely, and raise cash. You are not calling the top — you are refusing to be the marginal forced seller when the cascade comes. The asymmetry is the point: the upside left in a stretched market is small and the downside is a 40% cascade, so the risk/reward of staying full-size is terrible even if the exact timing is unknowable.

#### Worked example: sizing down before the cascade

Say your normal high-beta sleeve (brokers + steel + property midcaps) is **30% of a 1 billion dong portfolio**, i.e., 300 million dong (~\$11,800 at \$1 = 25,400 VND). The liquidity dashboard flashes late-cycle: ADTV at a record, total margin balance at an all-time high, brokers raising capital, account openings euphoric. You cut the high-beta sleeve to **half — 15%, or 150 million dong** — and move the freed 150 million to cash. Now suppose the cascade hits and the high-beta sleeve falls **40%**. On the full 300 million, that is a **120 million dong loss (−12% of the whole portfolio)**; on the trimmed 150 million, it is a **60 million dong loss (−6%)**. By sizing down on the *signal* rather than the *event*, you halved the drawdown and — critically — you are holding cash to buy the leaders back at the bottom when margin-to-equity has reset to slack. The intuition: in a leveraged market you cannot reliably time the top, but you *can* reliably reduce how much you have at risk when the fuel gauge reads "full," and that alone changes your cycle math.

### The warning signs of an imminent cascade

Beyond the slow-moving dashboard, watch for the fast tells that a deleveraging cascade is starting:

- **Liquid leaders breaking down on rising volume** while the index is still near highs — distribution by stretched holders.
- **Floor-price (limit-down) clusters** appearing across multiple high-beta names in the same session — the signature of forced selling, not discretionary selling.
- **Brokers cutting margin ratios** on stocks that have run hard — they are pulling fuel, which can puncture the move on its own.
- **A funding shock** in the background — a rate hike, a bond-market scare, a liquidity squeeze — that gives the over-leveraged a reason to be force-sold. The 2022 cascade had the corporate-bond crisis as its trigger.

When these appear together, the cascade has likely begun; the time to have sized down was *before* them, on the dashboard signal.

### The other side of the trade: buying the deleveraged bottom

Sizing down before a cascade is only half the edge. The other half is **buying the high-beta leaders back after the forced selling has exhausted itself** — and the same liquidity gauges that warned you to step aside also tell you when to step back in. After a deleveraging crash, the total margin balance has collapsed (the 2022 example: ~37% off the peak), margin-to-equity has reset from stretched to slack, and ADTV has bottomed. That combination — *low* margin balance, *slack* margin-to-equity, *depressed* turnover starting to tick up — is the mirror image of the late-cycle warning, and it is the most favorable setup the Vietnamese market offers: the forced sellers are gone, the float has passed from leveraged weak hands to unleveraged strong hands, and the reflexive flywheel is poised to start spinning *up* again. The high-beta sectors that fell hardest (brokers, property, steel) are typically the ones that bounce hardest off such a bottom, because the same leverage that crushed them now amplifies the recovery. The discipline is symmetric: you size *down* into records and size *up* into a deleveraged, slack regime — and the cash you raised by sizing down is exactly what funds the bottom-buying. This is why reading the liquidity cycle is not a one-directional "avoid risk" rule but a two-sided framework for *when to be aggressive and when to be patient.*

Note one subtlety: do not confuse the *first* sharp drop with the bottom. A cascade can have multiple waves as successive cohorts of accounts breach their floors, and the margin balance keeps falling for weeks or months as the deleveraging works through the system. The signal to buy is not "the index fell a lot" — it is "the margin balance has stopped falling and the margin-to-equity gauge has reset to slack." Wait for the leverage to actually come out, not just for the price to drop.

### The invalidation — when this view is wrong

A disciplined view names what would prove it wrong. The liquidity-regime playbook is invalidated, and you should *not* be defensively positioned, when: **ADTV and margin balance are rising from genuinely low levels** (early cycle — lean in, do not fight it); **margin-to-equity is slack** (the system has cushion, so drops get bought rather than cascading); and **rates are falling with a supportive State Bank** (the macro backdrop is filling the tank, not draining it). In that configuration, being underweight the high-beta sectors is the error — you would be fighting a reflexive loop that is working in your favor. The skill is not perma-bearishness about margin; it is reading *which way the flywheel is spinning* and positioning with it, while keeping enough discipline to size down when the gauges hit records. Get that one judgment right and you turn the most dangerous feature of the Vietnamese market — its leverage — into the thing that tells you when to be aggressive and when to step aside.

## Further reading & cross-links

- [Sector rotation explained: leaders and laggards on VN-Index](/blog/trading/vietnam-stocks/sector-rotation-explained-leaders-and-laggards) — how leadership passes from one sector to the next, which the margin cycle amplifies.
- [The business cycle and the investment clock for Vietnam](/blog/trading/vietnam-stocks/business-cycle-investment-clock-vietnam) — where the economy sits maps onto which way the liquidity flywheel spins.
- [Foreign flows, ETFs, and the index effect on Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam) — the other big external liquidity tap that interacts with domestic margin.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — why broker earnings (commissions + margin interest + prop) make securities firms a leveraged play on liquidity itself.
- [Vietnam monetary policy: the State Bank, the dong, and the credit ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling) — the rate and liquidity dials that fill or drain the tank.
