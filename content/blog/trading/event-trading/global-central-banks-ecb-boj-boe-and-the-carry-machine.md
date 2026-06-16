---
title: "Global Central Banks: The ECB, BoJ, BoE, and the Carry Machine"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The Fed is the biggest, but the European Central Bank, Bank of England, and especially the Bank of Japan move global markets too — and the gap between their policy rates powers the FX carry trade that, when it unwinds, can crash everything."
tags: ["event-trading", "macro", "central-banks", "ecb", "boj", "carry-trade", "yen", "dollar", "vn-index", "bitcoin", "rate-differential", "global-markets"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The Federal Reserve is the loudest central bank, but it is not the only one that moves your portfolio; the *gaps* between the world's policy rates power a global borrow-cheap-buy-high machine, and when that machine seizes, it crashes every market at once.
>
> - Four banks set the tempo: the **Fed** (US dollar), the **ECB** (euro), the **Bank of England** (pound), and the **Bank of Japan** (yen) — the last of which kept rates near or below zero for almost three decades and quietly became the funding source for half the world's leverage.
> - The **carry trade** is the transmission belt: borrow a near-zero-rate currency (the yen), buy a higher-yielding asset (US Treasury bills at 5%, Mexican bonds, US tech stocks, crypto), and pocket the **rate differential** — until a central-bank move flips the math and everyone unwinds through the same door.
> - On **August 5, 2024**, a tiny BoJ hike plus a soft US jobs print detonated exactly that machine: the **Nikkei fell −12.4%** (worst day since 1987), the **S&P −3.0%**, **Bitcoin −15%**, and the yen ripped from 162 to 142 in days.
> - The one number to remember: a **~3.5-percentage-point** rate gap between the ECB (4.00%) and the BoJ (~0%) in 2023 — that spread is the fuel, and the unwind is the fire.

At 8:30 a.m. in Tokyo on Monday, August 5, 2024, the screens went the wrong shade of red. The Nikkei 225 — Japan's benchmark stock index — was falling so fast that the exchange tripped its circuit breakers. By the close it was down **12.4%**, the single worst day for Japanese stocks since the Black Monday crash of October 1987. This was not a war, not a bankruptcy, not a pandemic. Nothing had *blown up* in the way crises usually announce themselves. Two small, dull-sounding events had happened the week before: the Bank of Japan had nudged its policy interest rate up by a quarter of a point (to about 0.25%), and the US had reported a softer-than-expected jobs number on Friday, August 2.

Two boring policy events. And yet the damage did not stay in Tokyo. By the time New York opened, the S&P 500 was down **3.0%** and the Nasdaq −3.43%; Bitcoin, which trades around the clock, had cratered roughly **15%**; the volatility index (the VIX, Wall Street's "fear gauge") spiked intraday to **65.73**, a level normally seen only in genuine financial panics. The yen — the Japanese currency — surged from about 162 per dollar to under 142 in a matter of days. A Vietnamese investor watching the VN-Index in Ho Chi Minh City felt it too, as foreign money fled risk everywhere at once.

How does a 0.25% rate tweak in Japan vaporize trillions of dollars of value across Tokyo, New York, and crypto in a single session? The answer is the subject of this entire post, and it has a name: the **carry trade**, the giant, invisible machine that runs on the *differences* between the world's central-bank interest rates. The Fed gets all the headlines, but the machine is powered by the ECB, the Bank of England, and above all the Bank of Japan. Once you can see the machine — how it is built, what feeds it, and what makes it seize — that confusing red Monday turns into a single, readable story.

![The carry machine borrow cheap yen buy high yield earn the spread then a rate move forces a global unwind](/imgs/blogs/global-central-banks-ecb-boj-boe-and-the-carry-machine-1.png)

## Foundations: the global central-bank map

Before we can trade any of this, we need a map. Who are the major central banks, what lever does each one pull, and how does a decision in Frankfurt or Tokyo end up in your account in New York or Hanoi? We build every term from zero — no finance background required.

### What a central bank does, in one sentence

A **central bank** is the institution that controls the price and quantity of a country's money. Its main lever is the **policy interest rate** — the rate at which commercial banks borrow from, or park money at, the central bank overnight. Move that one rate and you move the cost of money for the entire economy: mortgages, business loans, savings accounts, and — crucially for traders — the yield on that country's government bonds and the attractiveness of its currency. (For the full mechanism of how a rate decision propagates, see [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates); this post is about the *cross-border* effects.)

There are roughly a dozen central banks that matter to global markets, but four set the tempo:

- The **Federal Reserve (the Fed)** — the United States. Its rate is the **federal funds rate**, the gravitational center of all global finance because the dollar is the world's reserve currency.
- The **European Central Bank (ECB)** — the 20 countries that use the euro. Its key lever is the **deposit facility rate**, the rate banks earn for parking cash at the ECB overnight.
- The **Bank of England (BoE)** — the United Kingdom. Its lever is **Bank Rate**.
- The **Bank of Japan (BoJ)** — Japan. For most of the last 30 years its policy rate sat at, or below, zero. That is the single most important fact in this entire post.

### Defining the terms you will hear all post

A handful of pieces of jargon do all the work. Here they are, from scratch.

- **Deposit rate (ECB).** The interest the ECB pays banks to leave money with it overnight. It is the *floor* under all euro-area money-market rates, which is why traders watch it rather than the ECB's other rates.
- **Bank Rate (BoE).** The UK equivalent — the rate the Bank of England pays on commercial banks' reserves, and the anchor for sterling money markets.
- **NIRP — Negative Interest Rate Policy.** When a central bank sets its policy rate *below zero*, charging banks to hold money rather than paying them. It sounds absurd, but Japan and the euro area both did it for years to fight the *opposite* problem of the 2020s — too-low inflation, even deflation (falling prices). The BoJ's rate was −0.10% from 2016 until March 2024.
- **YCC — Yield Curve Control.** A BoJ tool where the central bank doesn't just set the overnight rate but pins a *longer-term* bond yield (Japan's 10-year) at a target by buying unlimited amounts of bonds. It is monetary policy with the volume turned to eleven.
- **Rate differential.** The *gap* between two countries' policy rates — say ECB 4.00% minus BoJ 0.10% ≈ 3.9 percentage points. This gap is the engine of everything that follows.
- **Funding currency.** A currency you *borrow* because it is cheap (low interest rate). The yen has been the world's favorite funding currency for two decades; the Swiss franc and, at times, the euro have played the role too.
- **The carry trade.** Borrow a cheap funding currency, convert it, and buy a higher-yielding asset. You earn the rate differential — the "carry" — for as long as the position stays calm. (We dedicate a whole section to it below, and the deep mechanics live in [Carry trade unwinds: 1998, 2008, 2024 — when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).)
- **An unwind.** The reverse of the carry trade, done in a panic: sell the high-yield asset, buy back the funding currency to repay the loan, all at once, alongside thousands of others doing the same. An unwind is fast, self-reinforcing, and the reason a tiny rate tweak can crash everything.

### The decision calendars

Each bank meets on a schedule, and the schedule *is* the event calendar:

- **Fed:** eight scheduled meetings a year (the FOMC), each a two-day affair ending with a statement, a press conference, and quarterly projections.
- **ECB:** eight Governing Council monetary-policy meetings a year, statement at 14:15 Frankfurt time, press conference 45 minutes later.
- **BoE:** eight Monetary Policy Committee meetings a year, with a published *vote split* (e.g. 7–2 to hold) that markets read closely.
- **BoJ:** eight meetings a year, but with famously vague, slow-moving communication — which is exactly why its rare *changes* (like the 2024 hikes) detonate so violently. A surprise from the world's sleepiest central bank is the most dangerous kind.

Mark these dates. A non-US central-bank meeting is a scheduled event you can position around, exactly like a US CPI or jobs report — the subject of the rest of this series.

### Why a rate decision becomes a currency move

The chain from "the ECB raised rates" to "EUR/USD jumped" is worth spelling out, because it is the same chain for every central bank and it explains everything downstream. Money is mobile and rational at the margin: a global investor with cash to park will move it to wherever it earns the best risk-adjusted return. If the ECB pays more to hold euros than the Fed pays to hold dollars, some capital rotates from dollars into euros. More demand for euros, less demand for dollars, and the price of the euro in dollars — EUR/USD — rises.

But it is rarely the *level* of rates that moves a currency on the day; it is the *change in expectations*. Markets are forward-looking. If everyone already expects the ECB to hike, that hike is **priced in** — the euro has already risen in anticipation, and the decision itself does nothing. What moves the currency on the announcement is the **surprise**: a hike when a hold was expected, a hawkish tone when a dovish one was priced, or a vote split more divided than the consensus. This is the single most important idea in the whole series: price already contains the consensus, and only the surprise moves anything on release. A central-bank decision that exactly matches expectations can be a non-event even if the rate changed.

That is why traders obsess over the *guidance* — the words the central bank uses about the future — at least as much as the rate decision itself. A central bank can hold rates and still crush its currency by signaling cuts ahead, or hike and still lift its currency by promising more. The number is the headline; the guidance is the trade.

### The yield channel: bonds move too

A rate decision does not only move the currency; it moves that country's **government bonds**. A bond's price moves opposite to its yield, and the yield is anchored by the expected path of the policy rate. When a central bank surprises hawkish, short-dated government bond yields jump (the market prices a higher path for the policy rate), and bond *prices* fall. When it surprises dovish, yields drop and prices rise. So a single decision moves three things at once — the policy rate, the currency, and the bond market — and each of those then radiates outward to stocks, commodities, and foreign markets. Keep this triad in mind: **rate, currency, bonds** are the three faces of one decision.

## The ECB: the eurozone's decision and the euro

The **European Central Bank** sets monetary policy for the 20 countries that share the euro — about 350 million people and the world's second-largest reserve-currency bloc. That makes the ECB the second-most-watched central bank on earth, and its decisions ripple into US and global markets through the **EUR/USD** exchange rate (the most-traded currency pair in the world) and through European government-bond yields.

### What the ECB watches and what it moves

The ECB's official mandate is **price stability**, defined as 2% inflation over the medium term. Its key lever is the **deposit facility rate**. When eurozone inflation exploded after 2021, the ECB went on the most aggressive tightening campaign in its history: from **−0.50%** (yes, negative) in mid-2022 up to a **4.00% peak in September 2023** — a 4.5-percentage-point move in about 14 months. It then began cutting as inflation cooled, reaching 2.00% by mid-2025.

The chart below plots that ECB path against the Bank of Japan's, and the picture tells the whole story of why a global carry machine exists. While the ECB was hiking to 4%, the BoJ stayed pinned near zero. That **gap** — roughly 3.5 to 4 percentage points at its widest — is the rate differential that pays the carry trade.

![ECB deposit rate versus Bank of Japan policy rate from 2019 to 2025 showing the widening gap](/imgs/blogs/global-central-banks-ecb-boj-boe-and-the-carry-machine-2.png)

### How an ECB decision reaches US and Vietnamese assets

A surprisingly hawkish ECB (raising rates, or signaling more hikes) makes euro deposits pay more, so global capital rotates *into* the euro and *out of* the dollar — EUR/USD rises, and the **dollar index (DXY)** falls. Because so many global assets are priced against the dollar, a weaker dollar is generally *supportive* for risk assets, emerging-market currencies, gold, and crypto. The reverse — a dovish ECB or a hawkish Fed — strengthens the dollar and tightens global financial conditions. This is the dollar channel, and it is the main way Frankfurt's decisions reach Hanoi: through the dollar and through global risk appetite. The full mechanism is in [Trading the dollar: DXY, carry, and the dollar smile](/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile).

#### Worked example: an ECB surprise on a \$50,000 EUR/USD position

Say you are long EUR/USD — you own euros against dollars — with a \$50,000 notional position, expecting the ECB to sound hawkish. The ECB delivers a surprise hike and signals more.

- EUR/USD jumps +1.2% on the decision. Your move: \$50,000 × 1.2% = **+\$600**.
- The dollar index (DXY) falls about −0.9% in sympathy, which lifts the gold you also hold: a \$10,000 gold position × +0.9% = **+\$90**.
- But your separately-held US tech stock, priced in the now-weaker dollar with a slightly higher discount rate from the rate-differential shift, barely moves: a \$20,000 position × −0.1% = **−\$20**.
- Net across the three legs: +\$600 + \$90 − \$20 = **+\$670** on the ECB surprise.

The lesson: a non-US central bank moves your portfolio mainly through the *currency it controls* and the dollar on the other side of that pair — the equity leg is a second-order echo.

### What makes the ECB harder than the Fed

The ECB has a structural complication the Fed does not: it sets one monetary policy for twenty very different economies. When German inflation and Italian inflation diverge, or when the bond yields of a heavily-indebted member (Italy, Greece) spike relative to a safe one (Germany), the ECB faces a **fragmentation** problem — the gap between member-state borrowing costs threatens to fracture the currency union. The ECB built a tool for exactly this (the Transmission Protection Instrument, which lets it buy a stressed country's bonds) precisely because a normal central bank does not have to worry about one of its "states" going bankrupt and leaving the currency.

For a trader, this means the ECB has a second thing to watch beyond inflation: the **spread** between Italian and German 10-year bond yields (the "BTP-Bund spread"). When that spread blows out, the ECB may turn dovish to calm it even if inflation argues for hikes — and that dovish turn weakens the euro. So a euro position is partly a bet on European political and fiscal stability, not just on inflation. This is why EUR/USD can fall on bad Italian political news even when the ECB hasn't met.

The ECB also communicates differently from the Fed. There is no "dot plot" of individual policymakers' rate projections; instead the market parses the tone of the President's press conference and the ECB's staff inflation projections, updated quarterly. A hawkish revision to the staff forecast can move the euro as much as a rate change.

#### Worked example: a dovish ECB hold that still moves markets

Suppose the ECB *holds* its deposit rate at 3.75% — exactly as expected — but the President's press conference flags rising recession risk and hints at cuts ahead. The rate did not change, yet the guidance is dovish versus what was priced.

- EUR/USD falls −0.8% on the dovish guidance: a \$40,000 long-EUR position loses \$40,000 × −0.8% = **−\$320**.
- The dollar index rises about +0.6% as money rotates back to dollars, which weighs on your gold: a \$15,000 gold position × −0.6% = **−\$90**.
- German 2-year bond yields drop ~10 basis points as cuts get priced; a \$25,000 long-bund position gains roughly \$25,000 × (10bp × ~2-year duration) ≈ \$25,000 × 0.20% = **+\$50**.
- Net across the three: −\$320 − \$90 + \$50 = **−\$360** — and the *rate never changed*.

The lesson: with central banks, the guidance is often the whole trade — a hold can move more money than a hike if the words surprise.

## The Bank of Japan: three decades of zero, and the pivot that broke the carry trade

If the Fed is the most powerful central bank, the **Bank of Japan** is the most *consequential* for the plumbing of global leverage — and almost nobody outside finance understands why. The reason is a single, strange fact: Japan kept interest rates at, or below, zero for almost thirty years.

### Why Japan ran rates at zero for a generation

After Japan's colossal stock-and-property bubble burst in 1990, the country fell into a decades-long fight not against inflation but against its opposite: **deflation**, where prices *fall* year after year, consumers delay spending (why buy today if it's cheaper tomorrow?), and the economy stagnates. To fight that, the BoJ cut rates to zero in the late 1990s, kept them there, and in 2016 went *negative* (NIRP, −0.10%) while also pinning the 10-year government-bond yield near zero (YCC). For a generation, borrowing in yen was nearly free.

Cheap, abundant, near-zero-cost money has to go somewhere. It went *out* — into the rest of the world, as the yen carry trade. Japanese pension funds, banks, insurers, and global hedge funds all borrowed yen and bought higher-yielding assets abroad. This is why the BoJ "funds half the world's carry": it is the cheapest large pool of money on the planet, and capital flows downhill from cheap to expensive.

Japan is also, by a wide margin, the world's **largest net creditor nation** — Japanese households and institutions hold trillions of dollars of foreign assets, much of it US Treasuries and US equities. Japan's Government Pension Investment Fund (GPIF), the largest pension fund on earth, holds a huge slug of foreign stocks and bonds. The Japanese life insurers ("lifers") are among the biggest single buyers of US corporate and government bonds. All of this is, in economic terms, a giant carry trade: long high-yielding foreign assets, implicitly short the cheap yen that funds the home economy. So when people say the BoJ funds the world, it is not a metaphor — Japanese savings genuinely sit underneath an enormous fraction of global asset prices.

### Why the deflation trap kept rates at zero so long

It is worth understanding *why* Japan couldn't just raise rates back to normal. In a deflationary economy, even a 0% interest rate is *too high* in real terms — because if prices are falling 1% a year, then cash that earns 0% is actually *gaining* 1% of purchasing power for doing nothing. That makes hoarding cash attractive and borrowing-to-invest unattractive, which strangles growth. The BoJ spent thirty years trying to convince Japanese households and firms that prices would *rise*, so they would spend and invest today. It tried zero rates, then negative rates, then yield curve control, then massive asset purchases — an entire encyclopedia of "unconventional" monetary policy that the rest of the world later copied after 2008. Only when global inflation finally washed into Japan in 2022–2023 did the BoJ get the rising prices it had wanted for decades — and only then could it begin to normalize. The 2024 hikes were not the BoJ panicking about inflation like the Fed in 2022; they were the BoJ *finally escaping* a thirty-year trap.

### The 2024 exit, and why it mattered most

For decades the carry trade was a calm, profitable, almost boring strategy precisely *because* everyone trusted the BoJ to keep rates near zero forever. The yen would gently weaken, the foreign asset would gently appreciate, and you collected the spread. By mid-2024 the yen had slid to **161.9 per dollar** (July 2024) — its weakest since 1986 — and the carry trade was the most crowded it had ever been.

Then the BoJ blinked. In **March 2024** it ended eight years of negative rates (lifting the policy rate to +0.10%) — the symbolic end of the NIRP era. In **July 2024** it hiked again, to about 0.25%, and signaled more. Tiny moves in absolute terms. But they were the first cracks in the one assumption the entire carry trade rested on: *that yen funding would stay free forever.* The yen-carry mechanism below shows why those cracks propagated so violently — one cheap funding source had fanned out into a thousand crowded risk positions, and the BoJ hike yanked them all back through the same door at the same time.

![The yen carry mechanism cheap yen funding fans out into leveraged positions then snaps back on a BoJ hike](/imgs/blogs/global-central-banks-ecb-boj-boe-and-the-carry-machine-3.png)

When a funding cost that everyone assumed was zero starts rising, two things happen at once. First, the *carry* (the spread you earn) shrinks, so the trade is less worth doing. Second, and far worse, the *funding currency strengthens* — the yen rises — which means the loan you have to repay just got bigger in your own currency. Both legs turn against you simultaneously. That double-whammy is what makes a carry unwind so much more violent than an ordinary selloff.

## The Bank of England and the others, briefly

The **Bank of England** deserves a mention because the pound is a major currency and the UK is a global financial center, but for our purposes it behaves like a smaller, more volatile version of the ECB. The BoE sets **Bank Rate**, publishes a **vote split** that markets parse for hints (a 5–4 vote to hold is read very differently from a 9–0), and moves the pound (GBP/USD, nicknamed "cable") on surprises. It also gave the world a recent lesson in central-bank-meets-market chaos: the September 2022 UK "mini-budget" crisis, when a botched fiscal plan sent UK government-bond yields spiking and forced the BoE into emergency bond-buying to stop pension funds from collapsing — a reminder that even a developed-market central bank can be a source of a global volatility shock.

The others on the watch list:

- The **Swiss National Bank (SNB)** runs the Swiss franc, the *other* classic funding currency, and is famous for occasionally intervening directly in FX markets.
- The **People's Bank of China (PBoC)** sets the yuan's reference rate and controls the world's second-largest economy; its easing or tightening drives commodity demand and emerging-market sentiment, which matters enormously to Vietnam as a neighbor and supply-chain partner.
- The **State Bank of Vietnam (SBV)** is Vietnam's own central bank — covered in depth in the Vietnam track of this series and in [Vietnam's monetary policy: the State Bank, the dong, and the credit ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling). The SBV's job is partly *reactive*: when the Fed hikes hard and the dollar surges, the SBV must often raise its own rates to defend the dong, as it did in autumn 2022 (lifting the refinance rate from 4.0% to 6.0%).

The September 2022 UK episode is worth dwelling on, because it shows a *developed-market* central bank becoming the epicenter of a global volatility shock — the kind of thing that's supposed to only happen in emerging markets. A new UK government announced large unfunded tax cuts; the bond market revolted, and UK 30-year gilt yields spiked so fast that pension funds running leveraged "liability-driven investment" strategies faced collapse as their collateral evaporated. The Bank of England had to step in with emergency, unlimited bond-buying — easing into the chaos even as it was *supposed* to be tightening to fight inflation. The pound briefly hit an all-time low against the dollar (near 1.03). The lesson for a global trader: the source of the next shock is not always the obvious one, and a fiscal mistake can force a central bank into a market-moving U-turn overnight.

The unifying point: every one of these banks moves global markets through the same two channels — the **currency it controls** and the **global risk appetite** its surprises create. You don't need to model all of them; you need to know which one is the source of the next shock.

#### Worked example: a hawkish BoE surprise on a \$30,000 sterling position

You are long GBP/USD (cable) with a \$30,000 notional, and the Bank of England surprises with a hawkish hold — a 5–4 vote that nearly hiked, plus tough inflation language.

- GBP/USD rises +1.0% on the hawkish surprise: \$30,000 × 1.0% = **+\$300**.
- UK 2-year gilt yields jump ~15 basis points as the market prices a higher path; a \$20,000 short-gilt position gains roughly \$20,000 × (15bp × ~2-year duration) ≈ \$20,000 × 0.30% = **+\$60**.
- The stronger pound nudges the dollar index down −0.4%, lifting a \$10,000 gold holding by \$10,000 × +0.4% = **+\$40**.
- Net across the three legs: +\$300 + \$60 + \$40 = **+\$400** on a hawkish surprise from a "smaller" central bank.

The lesson: the BoE is not the Fed, but a sterling position reacts to its surprises through the exact same rate-currency-bonds triad — the mechanism is universal, only the size differs.

## The carry machine: borrow cheap, buy high-yield, and the unwind

Now we assemble the machine in full. The carry trade is the single most important cross-border transmission mechanism in markets, and once you see its mechanics, the August 2024 crash becomes obvious in hindsight.

### The trade, step by step

1. **Borrow the funding currency.** You borrow yen at, say, 0.1% per year. Yen is cheap because the BoJ keeps it cheap.
2. **Convert and buy a higher-yielding asset.** You sell the yen for dollars and buy a US Treasury bill yielding 5%. Or you go further out the risk curve — Mexican bonds at 10%, US tech stocks, Bitcoin.
3. **Earn the spread.** As long as nothing moves, you collect the difference: 5% − 0.1% = **4.9%** per year, for doing essentially nothing. Add leverage (borrowing several times your capital) and that 4.9% becomes 15% or 25%.
4. **The risk you're short.** You are implicitly betting the yen *won't* strengthen and the asset *won't* crash. You are, in options language, **short volatility** — collecting small steady premiums while quietly exposed to a rare, violent loss.

#### Worked example: a yen-carry trade that earns the spread — until it doesn't

You put up \$100,000 of your own capital and borrow the yen-equivalent of \$100,000 at 0.1%, then buy US Treasury bills yielding 5%.

- Annual carry earned on the borrowed \$100,000: \$100,000 × (5% − 0.1%) = **+\$4,900 per year**.
- That is a steady, almost boring return — roughly +\$408 every month while the yen behaves.
- Now the unwind hits: in the first week of August 2024 the position drops about 15% as the asset falls and the yen surges. On \$100,000 that is \$100,000 × (−15%) = **−\$15,000** in a single week.
- One bad week erased **more than three years** of carry income (\$15,000 ÷ \$4,900 ≈ 3.1 years).

The intuition: carry is picking up nickels in front of a steamroller — years of small gains, then one move that takes it all back and more.

### The two legs of the unwind

A carry position has two ways to lose, and in an unwind *both* fire at once:

- **The FX leg.** The yen you borrowed strengthens, so repaying the loan costs more. If you borrowed when USD/JPY was 162 and the rate snaps to 142, the yen has appreciated ~12% against the dollar — your dollar asset now buys back fewer yen than you owe.
- **The asset leg.** The high-yield asset you bought (stocks, EM bonds, crypto) falls in the same risk-off panic, because *everyone else* running the same trade is selling the same assets to raise cash and repay their own yen loans.

The chart below shows the FX leg in isolation — the long, gentle slide of the yen from 2019 (when it was around 108 per dollar) to its 2024 extreme, and then the violent August snap-back.

![USD JPY exchange rate 2019 to 2024 with the July 2024 peak and the August 2024 reversal](/imgs/blogs/global-central-banks-ecb-boj-boe-and-the-carry-machine-4.png)

#### Worked example: the FX leg on an unhedged \$50,000 yen-funded position

You funded a \$50,000 position by borrowing yen when USD/JPY was 162. You left the currency exposure *unhedged* (you did not lock in the exchange rate). The yen then strengthens to 142.

- At 162 yen/dollar, your \$50,000 was a loan of 162 × 50,000 = **¥8,100,000**.
- The yen strengthens to 142: to repay ¥8,100,000 now costs 8,100,000 ÷ 142 = **\$57,042**.
- The currency move alone cost you \$57,042 − \$50,000 = **−\$7,042**, about −14% — before the asset itself moved a cent.
- If the asset *also* fell, say −10%, that is another \$50,000 × −10% = **−\$5,000**, for a combined ≈ **−\$12,042** loss.

The lesson: in a carry unwind the currency leg can hurt as much as the asset leg, and an unhedged carry trade is really two leveraged bets stacked on top of each other.

### Why the unwind self-reinforces

The deadly feature is the feedback loop. Yen strengthens → carry traders lose on the FX leg → they sell their foreign assets to cut risk → those assets fall → other traders' positions hit stop-losses and margin calls → they too sell assets and buy back yen → yen strengthens more → repeat. Each step makes the next worse. That is why August 5, 2024 was a *cascade*, not a dip: a self-amplifying chain that ran until the selling exhausted itself (and the BoJ verbally backed off further hikes).

### Leverage is the multiplier on everything

The reason a carry unwind is so destructive is **leverage** — borrowing to amplify the position. A carry trader rarely buys \$100,000 of assets with \$100,000 of capital; they post \$100,000 as margin and control \$300,000 or \$500,000 of assets, borrowing the rest in cheap yen. Leverage multiplies the carry income on the way up — and multiplies the losses, and triggers forced selling, on the way down.

#### Worked example: how leverage turns a 5% spread into a 15% gain and a wipeout

You post \$100,000 of capital and use 3× leverage, controlling \$300,000 of US T-bills funded with borrowed yen.

- Carry on \$300,000 at a 4.9% spread: \$300,000 × 4.9% = **+\$14,700 per year** — a 14.7% return on your \$100,000 capital, from a 4.9% spread. That is the seduction of leverage.
- Now the unwind: the position drops 15% on \$300,000 = \$300,000 × −15% = **−\$45,000** in a week.
- That \$45,000 loss is **−45% of your \$100,000 capital** — and at 3× leverage, a loss approaching your margin triggers a forced liquidation that crystallizes it.
- The same unleveraged trade would have lost 15% of \$100,000 = −\$15,000; leverage tripled both the gain *and* the catastrophe.

The lesson: leverage is why a small price move becomes a margin call, and why the most leveraged players are forced to sell first and hardest — turning a wobble into a cascade.

### Who actually runs the carry trade

It helps to know the players, because *who* is positioned tells you how violent an unwind will be. The carry trade is run by a wide cast: global macro hedge funds (explicitly, with leverage), Japanese retail investors (the famous "Mrs. Watanabe" who buys foreign-currency deposits and high-yield funds), Japanese institutions buying foreign bonds, and — crucially — *implicit* carry that doesn't look like a carry trade at all. A US tech-stock momentum fund partly funded by cheap borrowing, an emerging-market bond fund, a leveraged crypto position: all of them depend on cheap, abundant global liquidity, and all of them unwind in the same direction when that liquidity is yanked. This is why the August 2024 unwind hit assets that had no obvious connection to Japan — they were *implicit* carry, riding the same liquidity tide.

## How non-US decisions transmit to US and Vietnamese assets

Here is the question that matters most for a reader in New York or Ho Chi Minh City: *I don't trade the yen or euro directly — why should I care what the BoJ or ECB does?* Because their decisions reach your assets through two channels you cannot avoid: **the dollar** and **global liquidity**.

### The dollar channel

Every non-US central-bank move changes the rate differential against the US, which moves that currency against the dollar, which moves the **dollar index (DXY)**. And the dollar is the price of money for the whole world. A stronger dollar:

- Tightens global financial conditions (dollar debt gets more expensive everywhere).
- Pulls capital *home* to the US, away from emerging and frontier markets.
- Pressures commodities (priced in dollars) and crypto (a pure liquidity asset — see [Crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset)).

So even a decision made in Tokyo or Frankfurt reaches a US tech stock or a Vietnamese bank stock by first moving the dollar.

There is a useful mental shortcut here: the dollar is the *inverse* of global risk appetite. When markets are calm and money is chasing yield, capital flows *out* of dollars into higher-returning assets everywhere, and the dollar drifts weaker — good for emerging markets, commodities, and crypto. When markets panic, capital rushes *back* into dollars as the ultimate safe haven, the dollar surges, and everything risky falls. So a non-US central-bank shock that sparks risk-off automatically strengthens the dollar, which then tightens conditions for every dollar-borrower on earth — a Brazilian company with dollar debt, a Vietnamese importer, a leveraged crypto fund. The dollar is the transmission cable, and it carries the shock whether or not your asset has anything to do with the bank that caused it. This counter-cyclical behavior — the "dollar smile" — is detailed in the dollar deep-dive linked above.

### The global-liquidity channel

The carry trade *is* global liquidity. When yen funding is cheap and abundant, that money flows into risk assets worldwide, inflating everything from US equities to Vietnamese real estate. When the carry unwinds, that liquidity is *withdrawn* all at once — a global margin call. This is why the Aug-2024 unwind hit Vietnam even though the SBV did nothing: foreign investors raising cash everywhere sold their VN holdings too.

The transmission to Vietnam follows a clear chain, shown below. Vietnam is not priced off the BoJ directly — it is priced off the *risk-off and the dollar* that a BoJ shock creates.

![How a non-US central bank shock reaches the VN-Index through risk-off the dollar and foreign outflows](/imgs/blogs/global-central-banks-ecb-boj-boe-and-the-carry-machine-6.png)

The VN-Index reacts with a *lag* — usually a few sessions — because foreign flows take time to register on the Ho Chi Minh Stock Exchange (HOSE), and because Vietnamese retail investors often buy the first dip before the foreign-selling wave fully arrives. In 2024, foreign investors net-sold roughly **−90 trillion VND** of Vietnamese stocks, the heaviest outflow on record, partly driven by exactly this global de-risking and a strong dollar (USD/VND rose toward 25,485 by year-end). The mechanics of these flows live in [Foreign flows, ETFs, and the index effect in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam).

There is a second, more direct way a non-US central bank reaches Vietnam: through the **State Bank of Vietnam's** own forced response. The SBV runs a tightly-managed currency, keeping USD/VND within a controlled band. When the Fed hikes hard or a global risk-off sends the dollar surging, the dong comes under depreciation pressure, and the SBV must defend it — by selling some of its dollar reserves, by tightening domestic liquidity, or by raising its own policy rates. That is exactly what happened in autumn 2022: as the Fed tightened aggressively and the dollar ripped to multi-decade highs, the SBV was forced to lift its refinance rate from 4.0% all the way to 6.0% in two moves to stem dong weakness and reserve loss. Higher domestic rates and tighter liquidity then hit Vietnamese stocks directly — the VN-Index fell from its January 2022 peak near 1,528 to a trough of **911 on November 15, 2022**, a roughly −40% drawdown. So a foreign central bank can reach Vietnam *twice*: first through the dollar and foreign flows, and second by forcing the SBV's hand. Both channels point the same way in a global tightening.

When the global tide turned in 2023 — the Fed slowed, the dollar softened — the SBV reversed course, cutting its refinance rate back to 4.5% across three moves between April and June 2023, and the VN-Index recovered toward 1,130 by year-end and 1,267 by the end of 2024. The point: Vietnamese monetary policy is partly a *reaction* to the global rate environment that the big central banks set, which is why a VN-focused investor still has to watch the Fed, the ECB, and the BoJ.

#### Worked example: a foreign-outflow day on a \$8,000 VN-stock position

You hold \$8,000 of Vietnamese large-cap stocks (banks and real estate, the most foreign-owned). A global carry unwind triggers a risk-off wave, and foreign investors dump VN equities.

- The VN-Index falls −3.2% over the two sessions after the shock as foreign selling lands: \$8,000 × −3.2% = **−\$256**.
- The dong weakens 0.6% against the dollar in the same window, so your USD-measured loss widens: an extra \$8,000 × −0.6% = **−\$48**.
- Combined USD hit: −\$256 − \$48 = **−\$304**, about −3.8% — caused by a central-bank decision made 3,000 km away in Tokyo.
- Had you cut to \$4,000 ahead of a known BoJ meeting, the same −3.8% would have cost just **−\$152**.

The lesson: a Vietnamese portfolio is not insulated from global central banks; it is downstream of them, with a short lag that you can sometimes position ahead of.

## How it reacted: real episodes

Theory is cheap. Here are two dated episodes where the carry machine showed its teeth, with real numbers.

### August 5, 2024: the carry unwind in detail

The setup was a textbook crowded carry. The yen had slid to **161.9 per dollar** in July 2024, the weakest since 1986. Speculators were record-net-short the yen (everyone was on the same side of the boat). The rate differential against the US was enormous (US ~5.5%, Japan ~0.25%). Then two things broke the calm in four days:

1. **July 31, 2024:** the BoJ hiked to ~0.25% and signaled more — the funding cost was rising.
2. **August 2, 2024:** the US jobs report came in soft (+114k vs ~175k expected, unemployment up to 4.3%), which made the market expect *Fed cuts* — narrowing the US-Japan differential from the other side.

Both blades of the scissors closed on the carry trade at once. Over the weekend, leveraged players scrambled to unwind. Monday, August 5, was the result. The chart below shows how a Tokyo unwind hit *every* market — not just Japan.

![August 2024 cascade across markets Nikkei S and P Nasdaq and Bitcoin all falling](/imgs/blogs/global-central-banks-ecb-boj-boe-and-the-carry-machine-5.png)

The Nikkei fell **−12.4%**, its worst day since 1987. The S&P 500 fell **−3.0%**, the Nasdaq 100 **−3.43%**, and Bitcoin — the most liquidity-sensitive asset of all — **−15%**. The VIX spiked intraday to **65.73** from a calm 23.4 two sessions earlier. And then, just as fast, it reversed: the Nikkei rallied **+10.23%** the very next day (August 6) once the BoJ's deputy governor publicly walked back the hawkish tone, the yen stopped surging, and the forced selling exhausted itself. The whole episode lasted about three sessions — the signature of a positioning-driven unwind, not a fundamental collapse. (For why a crowded position primes exactly this kind of violent reversal, see [Positioning and the pain trade](/blog/trading/event-trading/positioning-and-the-pain-trade).)

#### Worked example: the August 2024 unwind across three positions

You went into the first week of August 2024 holding three carry-flavored positions, each \$20,000.

- A **Japan equity** position (you owned the Nikkei) fell −12.4% on August 5: \$20,000 × −12.4% = **−\$2,480** in one day.
- A **US tech** position (Nasdaq) fell −3.43%: \$20,000 × −3.43% = **−\$686**.
- A **Bitcoin** position fell −15%: \$20,000 × −15% = **−\$3,000**.
- Total one-day damage across the \$60,000: −\$2,480 − \$686 − \$3,000 = **−\$6,166**, about −10.3% of the book.
- But if you held through August 6, the Nikkei's +10.23% bounce recovered \$20,000 × +10.23% = **+\$2,046** of the Japan leg.

The lesson: a positioning unwind is brutal *and* fast — the same crowding that makes the drop violent makes the snap-back violent too, which is why selling into the panic is often the worst trade.

### 2022: the yen's slide past 150 and the intervention watch

The August 2024 crash was the *unwind*; the years before it were the *build-up*. As the Fed hiked aggressively in 2022 while the BoJ stayed at −0.10%, the US-Japan rate differential blew out, and the yen slid relentlessly: from about 115 per dollar at the start of 2022 to **131** by year-end, touching **151.9** in October 2022. A weakening yen was the carry trade *working* — the funding currency was depreciating, exactly what a carry trader wants. But it also got so extreme that Japan's Ministry of Finance intervened directly in the FX market (selling dollars, buying yen) to slow the slide. Every leg of that 2022–2024 yen weakness was simultaneously building the crowded position that would detonate in August 2024.

#### Worked example: carrying the yen-funded trade through 2022, in dollars

You ran a \$30,000 yen-funded US T-bill carry through 2022, unhedged, starting at USD/JPY 115.

- The rate spread you earned: roughly \$30,000 × (4.0% − 0.0%) for the year ≈ **+\$1,200** in carry (US rates rose through the year; use ~4% average).
- The FX tailwind: the yen fell from 115 to 131, a ~12% depreciation. Your dollar asset now repays a cheaper yen loan — a currency *gain* of about \$30,000 × 12% ≈ **+\$3,600**.
- Total 2022 gain ≈ +\$1,200 + \$3,600 = **+\$4,800**, about +16% — most of it from the *currency*, not the interest spread.
- That outsized FX gain is exactly what lured more capital into the crowded trade that blew up in 2024.

The lesson: when a carry trade is "working," most of the profit often comes from the funding currency falling — which is also the precise thing that will reverse, violently, in the unwind.

### October 1998: the original yen-carry unwind

August 2024 was not the first time the yen carry trade detonated — it was a rerun. In the autumn of 1998, the collapse of the giant hedge fund Long-Term Capital Management (LTCM) and the Russian debt default forced a frantic global de-risking. A huge yen carry trade had built up through the 1990s (borrow yen at near-zero, buy higher-yielding assets worldwide), and as funds scrambled to cut risk and repay yen loans, the yen exploded higher: USD/JPY fell from about 136 to 112 in a matter of days in early October 1998 — a roughly 18% move in the funding currency in under a week, one of the most violent currency moves in modern history. The mechanics were identical to 2024: a crowded carry, a risk shock, a self-reinforcing scramble to buy back the funding currency. The full cross-crisis pattern — 1998, 2008, and 2024 — is the subject of [Carry trade unwinds: 1998, 2008, 2024 — when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks). The takeaway: the carry machine is not a 2024 quirk; it is a recurring feature of a world built on rate differentials, and it breaks the same way every time.

## Common misconceptions

A few myths cause real money to be lost. Each is corrected with a number.

**"Only the Fed matters; I can ignore the other central banks."** The single worst US equity day of 2024 was not caused by the Fed — the S&P fell −3.0% on August 5 because of a *Bank of Japan* hike. The BoJ funds an estimated large share of the world's carry leverage; ignoring it cost passive US investors a sharp drawdown they never saw coming. You cannot ignore the bank that funds half the world's risk.

**"A carry trade is safe — it's just earning interest."** Carry earns a small, steady spread (our example: +\$4,900/year on \$100,000) and then loses it all in a week (−\$15,000 in August 2024). It is *short volatility*: it looks safe right up until it isn't. The return profile is the opposite of safe — many small wins, then one catastrophic loss.

**"A stronger yen is good for Japanese stocks."** The opposite, in the short run. Many large Japanese companies are exporters who earn revenue abroad; a stronger yen shrinks those foreign earnings when converted home. More importantly, a surging yen *is the unwind* — which is why the Nikkei fell −12.4% on the day the yen ripped higher. Currency and equity moved together, down.

**"Vietnam is insulated because the SBV didn't change rates."** The SBV did nothing in early August 2024, yet the VN-Index still felt the global risk-off as foreign investors sold to raise cash worldwide. Vietnam is downstream of the dollar and global liquidity, both of which a non-US central bank can move. Foreign investors net-sold a record ~90 trillion VND in 2024.

**"The crash means the carry trade is dead."** It bounced almost immediately — the Nikkei rallied +10.23% the next day. Carry trades don't die in an unwind; they get smaller, then rebuild as soon as the rate differential reopens. The machine resets; it does not switch off.

**"The BoJ is small, so its moves are small."** The BoJ's policy *rate change* in 2024 was tiny — from −0.10% to +0.25%, a fraction of a single Fed hike. But the *positioning* behind the yen was enormous, and it is positioning, not the size of the rate move, that determines the violence of the reaction. A 0.25% move detonated trillions of dollars because the trade resting on top of it was record-crowded. Always weigh the surprise against the *positioning*, not against the size of the policy step — a small surprise into a crowded trade beats a large surprise into a balanced one.

## The playbook: how to trade global central banks and the carry

You will not out-forecast the BoJ. What you *can* do is read the conditions that make a central-bank meeting dangerous, and position around them. The figure below is the checklist: before any non-US central-bank decision, check the three gauges.

![The trader global central bank read watch the rate differential the funding currency and crowded carry](/imgs/blogs/global-central-banks-ecb-boj-boe-and-the-carry-machine-7.png)

**The three gauges, before any major non-US central-bank meeting:**

1. **The rate differential.** How wide is the gap between this bank and the funding currency (or the Fed)? A wide gap (ECB 4% vs BoJ 0%) means a big carry incentive and a big position to unwind.
2. **The funding currency.** Is the yen (or franc) weak and trending? A very weak, very crowded funding currency is dry tinder — the more extreme the slide, the more violent the eventual snap-back.
3. **Crowded carry.** How one-sided is positioning? Record net-short yen in the CFTC futures data (the "Commitments of Traders" report — see [Following the flows: positioning, COT, and dealer hedging](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging)) means everyone is trying to exit the same door at once.

When all three are lit — wide spread, weak crowded funding currency, one-sided positioning — you have the August 2024 setup, and a hawkish surprise can detonate the carry trade.

**The if-then map for a BoJ (or ECB/BoE) decision:**

- **Consensus = no change, dovish hold.** Base case: the carry stays on, the funding currency drifts weaker, risk assets grind up. Position: stay risk-on but keep position sizes modest if the three gauges are lit.
- **Hawkish surprise (hike, or hawkish signal).** The dangerous scenario. The funding currency surges, the carry unwinds, *everything risk* sells off — Japanese and US equities, EM, crypto, and (with a lag) the VN-Index. Position: cut risk and/or hedge *before* the meeting if the gauges are lit; you cannot reliably react fast enough after.
- **Dovish surprise (cut, or dovish signal).** The funding currency weakens further, the carry gets *more* attractive, risk assets rally. Position: a tailwind for risk-on, but watch that it doesn't push the crowd to an even more dangerous extreme.

**Sizing and risk around the event:**

- **Size down into a lit-gauge meeting.** If all three gauges are flashing, halve your risk-asset exposure ahead of the decision. As the worked example showed, cutting an \$8,000 VN position to \$4,000 turned a −\$304 hit into −\$152.
- **Hedge the FX leg explicitly.** The currency leg of a carry trade can hurt as much as the asset leg (−\$7,042 on a \$50,000 position in our example). If you run carry, hedge the funding currency or accept that you hold two leveraged bets, not one.
- **Don't sell into the panic.** The unwind is fast and the bounce is fast (Nikkei −12.4% then +10.23%). The time to act is *before* the meeting; once the cascade is running, you are usually selling the low.
- **The invalidation.** Your "the carry is safe" thesis is invalidated the moment the funding currency starts strengthening *into* the meeting — that is the market front-running the unwind. Respect it.

**How to actually express the view.** You rarely need to trade the yen directly to play this. A few practical expressions, from most to least direct:

- **Reduce risk-asset exposure into a lit-gauge meeting.** The simplest and often best trade is to *do less* — trim equities, crypto, and EM ahead of a dangerous BoJ or ECB decision, and redeploy after the dust settles. Cutting exposure costs nothing if the meeting is benign and saves you the cascade if it isn't.
- **Hold or add dollars / dollar cash.** Because a carry unwind strengthens the dollar, holding dollar cash (or a dollar-index proxy) is a built-in hedge against the risk-off it creates — it rises precisely when your risk assets fall.
- **Buy downside protection.** Options on equity indices (puts) or volatility get cheap when everyone is complacent into a "no-change-expected" meeting. A small defined-risk put position can pay off enormously in a cascade, with a known maximum loss equal to the premium.
- **For the brave: fade the unwind, carefully.** The bounce after a positioning unwind is violent (Nikkei +10.23% the day after −12.4%). Buying *into* the panic — once the central bank verbally backs off and the forced selling exhausts — has historically been profitable, but it is a knife-catch and demands strict risk limits.

A non-US central-bank meeting is exactly like a US CPI or jobs release: a scheduled event with a knowable consensus, a surprise that moves the sign, and a reaction you can position around — the core idea of this whole series, developed in [The reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently).

## Further reading & cross-links

- [Carry trade unwinds: 1998, 2008, 2024 — when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — the full anatomy of how carry trades blow up, across three crises.
- [Trading the dollar: DXY, carry, and the dollar smile](/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile) — the dollar channel that transmits every non-US central-bank move to global assets.
- [What moves exchange rates: rates, flows, and carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) — why rate differentials drive currencies in the first place.
- [Positioning and the pain trade](/blog/trading/event-trading/positioning-and-the-pain-trade) — why crowded positioning makes an unwind violent.
- [The reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) — how to read whether a surprise is good or bad news, regime by regime.
