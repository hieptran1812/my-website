---
title: "Case Study — 2022: The Year Stocks and Bonds Fell Together"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The capstone case study of the series: how a 40-year-high inflation shock in 2022 flipped the stock-bond correlation positive, broke the 60/40 in its worst year since 1937, and proved that bonds only hedge a stock crash when the shock is about growth, not inflation."
tags: ["asset-allocation", "cross-asset", "inflation", "stock-bond-correlation", "sixty-forty", "real-yields", "commodities", "bonds", "regime", "2022", "fed", "diversification"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — In 2022 an **inflation shock** — the worst in 40 years — did something the modern playbook said couldn't happen: it made **stocks and bonds fall together**. The classic 60/40 portfolio had its worst year since 1937. The only winners were **commodities, energy, cash, and the dollar**. The lesson is the whole series in one sentence: bonds hedge a stock crash *only when the shock is about growth*. When the shock is inflation, the Fed hikes instead of cuts, and bonds fall *with* stocks.
>
> - **The setup:** post-COVID stimulus and broken supply chains pushed inflation up; the Ukraine war in February 2022 poured oil on the fire. CPI hit **9.06%** in June 2022, a 40-year high, and a late Fed then hiked at the fastest pace in decades.
> - **The scoreboard:** S&P 500 **−18.1%**, US bonds (the Agg) **−13.0%** (worst year in the index's history), the 60/40 **−16.0%** (worst since 1937), REITs −24.9%, high yield −11.2%, Bitcoin −64%. Commodities **+16.1%**, energy equities **~+65%**, cash **+1.5%**.
> - **The mechanism:** the 10-year real yield went from about **−1% to +1.7%** in a year, and a rising real yield re-prices every long-duration asset — bonds, tech stocks, gold, crypto — *down at the same time*.
> - **The one fact to remember:** the stock-bond correlation flipped from negative (the 2010s) to about **+0.55** in 2022. Diversification by bonds alone failed because the regime changed.

In the spring of 2022, a saver who had done everything by the book opened their statement and felt the floor tilt. They owned the most respected, most boring, most "you can't go wrong with this" portfolio in finance: 60% stocks, 40% bonds. The whole point of the bonds was insurance. For forty years, whenever stocks had a bad spell, the bonds went *up* and softened the blow. That was the deal. That was why you paid the opportunity cost of holding bonds at all.

In 2022, the deal broke. Stocks fell hard — the S&P 500 ended the year down about 18%. And the bonds, the insurance, the part that was supposed to cushion the fall, fell *too*, by about 13% — the worst year in the entire history of the US bond index. There was no cushion. The two things in the portfolio that were supposed to zig and zag against each other both went straight down. For the first time in most investors' careers, the 60/40 offered no shelter at all. It had its worst year since 1937.

The figure below is the scoreboard for the whole year, and it is this post's thesis in one picture: nearly everything fell — stocks, bonds, the 60/40 that blends them, real estate, junk bonds, crypto — while a small set of assets, the ones that win when inflation is the enemy, went up. Commodities, energy producers, cash, and the dollar were the year's only refuges. Everything that follows is the story of *why* that happened, told in order with the real dates and numbers, and then the single most important regime lesson of this entire series.

![Horizontal bar chart of 2022 total returns showing stocks bonds the sixty forty REITs high yield and Bitcoin down in red and commodities energy and cash up in green](/imgs/blogs/case-study-2022-stocks-and-bonds-both-fell-1.png)

By the end, you will be able to look at a world where inflation is high and rising and the central bank is hiking, and say, with real historical grounding: "this is an inflation shock, not a growth shock — my bonds are not going to save me here, I want commodities, energy, cash, and short duration, and I should not lean on the 60/40 alone." That single read is the most expensive lesson of the post-COVID era, and 2022 taught it to a whole generation of investors at once.

## Foundations: the setup — how the world got to a 40-year inflation high

Before we can trade the lesson, we have to understand the machine that produced 2022. A 9% inflation print and the fastest rate-hike cycle in decades did not fall from the sky; they were the product of a specific chain of stimulus, supply shocks, and a central bank that moved too late and then too fast. Let us build the setup from zero, assuming you have never thought about why prices rise or what a central bank actually does.

### First, the words you need: inflation, the Fed, nominal versus real, and duration

Five terms run through this entire post, so let us pin them down immediately.

**Inflation** is the rate at which the general price level rises. We read it off the *Consumer Price Index*, or CPI — the cost of a typical basket of goods and services that households buy. "Inflation is 9%" means that basket costs about 9% more than it did a year ago. A healthy, normal world runs near 2%; 2022 ran more than four times that.

The **Federal Reserve**, or "the Fed," is the United States' central bank. Its main lever is the *policy interest rate* — the rate at which banks lend to each other overnight, which it sets in a target range. When the Fed *raises* that rate, borrowing gets more expensive across the whole economy, demand cools, and inflation eventually slows. When it *cuts* the rate, borrowing gets cheaper, demand heats up, and the economy is stimulated. The Fed's job, in one line, is to keep inflation near 2% and employment healthy. In 2022 it had badly missed the first half of that mandate, and it spent the year racing to catch up. (We cover the mechanics in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).)

A **nominal** number is the raw figure on the screen — the dollar amount before adjusting for anything. A **real** number adjusts for inflation; it tells you what your money can actually *buy*. If your portfolio rises 5% but prices rose 9%, your *nominal* return is +5% but your *real* return is roughly −4%: you got poorer in purchasing-power terms. This distinction is the spine of the whole series, and we go deep on it in [real vs nominal: inflation, real yields, and the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).

**Duration** is the single most important word for understanding 2022, so we will define it carefully and return to it. Duration is a measure of how sensitive an asset's price is to a change in interest rates. An asset whose value comes mostly from cash flows *far in the future* — a 30-year bond, a tech company whose profits are years away, gold (which pays nothing, ever) — has *long* duration: when interest rates rise, its price falls a lot. An asset whose value comes from cash *soon* — a 1-year bond, a cheap company spitting out dividends today, a money-market fund — has *short* duration and barely flinches. In 2022, interest rates rose violently, and *everything with long duration fell together*. That sentence is most of the story.

### Why prices took off: stimulus meets broken supply chains

The inflation of 2022 was years in the making. When COVID-19 hit in early 2020, governments and central banks responded with the largest peacetime stimulus in history. The US government sent direct payments to households, expanded unemployment benefits, and ran enormous deficits; the Fed cut its policy rate to near zero and bought trillions of dollars of bonds (a policy called *quantitative easing*, which we explain in [quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money)). The result was a flood of money and a population, fresh out of lockdown, eager to spend it.

That demand hit an economy that could not keep up. COVID had snarled the world's supply chains: factories had shut, ports were backed up, shipping containers were in the wrong places, and there were not enough workers, trucks, or chips to make and move everything people wanted to buy. You had *too much money chasing too few goods* — which is the textbook definition of inflation. Used-car prices spiked because new cars were stuck waiting for semiconductors. Shipping a container from Asia to the US went from about \$2,000 to over \$15,000 at the peak. Prices climbed all through 2021.

For most of 2021, the Fed called this inflation **"transitory"** — a word that became infamous. The theory was that once supply chains healed and the stimulus faded, inflation would melt back to 2% on its own, so there was no need to raise rates and risk choking the recovery. It was a reasonable bet. It was also wrong, and the wrongness was about to get much worse.

There is a structural reason the Fed was so reluctant to move, and it matters for understanding why the eventual hikes were so violent. For the entire decade before COVID, the problem had been *too little* inflation, not too much — the Fed had spent years trying and failing to push inflation *up* to its 2% target. So when prices started rising in 2021, the institutional instinct was to welcome it, not fight it; the muscle memory was all about avoiding a premature tightening that might kill the recovery, as had arguably happened after the 2008 crisis. The Fed was, in effect, fighting the last war. By the time it became undeniable that this inflation was not transitory, the central bank was months behind, prices had a head of steam, and the only way to catch up was to hike harder and faster than it otherwise would have. The lateness of the start is *why* the 2022 cycle was the fastest in decades — and the speed of the hikes is what gave markets no time to adjust.

### The detonator: the Ukraine war and the commodity spike

On **February 24, 2022**, Russia invaded Ukraine. Russia is one of the world's largest exporters of oil, natural gas, wheat, and fertilizer; Ukraine is one of the largest exporters of grain and sunflower oil. The war and the sanctions that followed sent the prices of those commodities vertical. Crude oil, which had started 2022 around \$75 a barrel, spiked above \$120. European natural gas prices multiplied several times over. Wheat hit records. Food and energy — the two things every household and every business cannot avoid buying — got dramatically more expensive, almost overnight.

This is what economists call a **supply shock** — a sudden, externally imposed jump in the cost of producing goods. A supply shock is uniquely nasty because it pushes inflation *up* (everything costs more) at the same time as it pushes growth *down* (the economy can afford to do less). It is exactly the kind of shock that the 1970s ran on, and it is the reason 2022 rhymes with stagflation even though it never fully became it. We trace this commodity-and-energy mechanism in [energy, oil and gas: the inflation engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine).

So the inflation that the Fed had bet would fade *accelerated* instead. The "transitory" call collapsed. By the spring of 2022, inflation was not a supply-chain hiccup that would resolve itself; it was a 40-year high, and the central bank had to slam on the brakes — late.

### The Fed turns: from "transitory" to the fastest hikes in decades

Having been late, the Fed then moved fast. In **March 2022** it raised its policy rate for the first time, by 0.25%, lifting the target range off its near-zero floor. Then it accelerated: 0.50% in May, an enormous 0.75% in June, another 0.75% in July, 0.75% again in September, 0.75% in November, and 0.50% in December. By the end of 2022 the target range's upper bound had gone from **0.25% to 4.50%** — more than four full percentage points in nine months. To put that in context, the previous tightening cycle (2015–2018) took *three years* to raise rates by about 2.25%. In 2022 the Fed did roughly double that, in a quarter of the time. It was the fastest pace of hikes in four decades.

The mental model for the whole setup is the figure later in this post that shows the three forces moving at once: inflation surging, the Fed hiking, and — the part that did the damage to your portfolio — the *real yield* flipping from deeply negative to firmly positive. Hold that word "real yield" for a moment; it is the engine, and we will spend a whole section on it. First, the chronology.

So the setup, in one sentence: an unprecedented wave of stimulus met a supply-constrained world, a war spiked commodities on top of it, a central bank that had bet on "transitory" was forced into the fastest hiking cycle in decades — and that collision is what broke the 60/40.

## The chronology: how 2022 actually unfolded

A case study earns its name by telling the story in order, with dates. 2022 was not a single crash; it was a year-long grind in which the same force — rising rates driven by inflation — methodically re-priced one asset class after another, and then all of them together. Let us walk it.

### Late 2021: the warning signs the market shrugged off

The trouble was visible before 2022 even began. CPI inflation, which had been running near 1.4% at the start of 2021, climbed relentlessly through the year: 5.0% by May, 5.4% in the summer, and **6.8% by November 2021**. Bonds had already started to notice — the US Aggregate bond index actually ended 2021 down about 1.5%, an unusual losing year for bonds that, in hindsight, was the first tremor. But stocks were euphoric: the S&P 500 finished 2021 up about 28.7%, near all-time highs, led by enormous, expensive technology companies. Long-duration everything — long bonds, growth stocks, gold, and especially crypto — was priced for a world of permanently near-zero interest rates. That world was about to end.

### January–February: the long-duration unwind begins

The repricing started at the top of the duration ladder — with the most rate-sensitive assets — even before the Fed's first hike. The reasoning markets ran was simple: if the Fed is about to raise rates aggressively, then the *discount rate* used to value distant future profits is about to go up, which mechanically lowers the present value of any asset whose payoff is far away. So the assets that fell first and hardest were the longest-duration ones: speculative, profitless tech stocks, the high-growth darlings of 2020–2021, and crypto. The Nasdaq, heavy with long-duration growth, entered a correction in January. Bitcoin, which had peaked near \$69,000 in November 2021, was already sliding. The market was front-running the Fed.

### June: the 9% print and the capitulation

The defining moment arrived in June 2022. The CPI report for June showed inflation running at **9.06% year-over-year — a 40-year high.** It was higher than anyone in markets had expected, and it killed the last hope that inflation was about to roll over on its own. The Fed responded with a 0.75% hike, its largest single move since 1994. Stocks capitulated: the S&P 500 fell into a *bear market* (a drop of 20% or more from the peak), and by mid-June it was down more than 23% from its January high. Bonds were falling right alongside, because the same surging-rate force that crushed stocks crushed bonds too. There was no rotation from one to the other — both were going down for the *same* reason.

#### Worked example: why a long bond loses 13% when yields rise

Let us make the bond loss concrete, because "bonds fell 13%" stays vague until you see the arithmetic. The reason a bond's *price* falls when interest rates *rise* is the most counterintuitive fact in finance for beginners, so we will build it from one bond.

Say you own a bond that pays a fixed \$3 a year (a 3% coupon on a \$100 face value) and matures in 10 years. You bought it when prevailing 10-year interest rates were 3%, so it was worth its \$100 face value — a fair price for \$3 a year.

Now suppose, over 2022, prevailing 10-year yields jump from about 1.5% at the start of the year to about 3.9% by the end — a rise of roughly 2.4 percentage points. New bonds are now being issued paying around \$3.90 a year. Nobody will pay you \$100 for your old bond paying only \$3 when they can buy a fresh one paying \$3.90. So the price of your old bond has to *fall* until its yield matches the new world.

How far does it fall? The rough rule is: price change ≈ −(duration) × (change in yield). A 10-year bond has a duration of roughly 8.5 years. So a 2.4-percentage-point rise in yield costs you about 8.5 × 2.4% ≈ **−20%** on a single long bond. The broad bond *index* (the Agg) has a shorter average duration, around 6 years, and a mix of maturities, so its loss was smaller — about **−13%**. But the mechanism is identical: rates up, bond prices down, and the longer the duration, the bigger the hit. The lesson: a bond is a stream of fixed future dollars, and when the interest rate used to value those dollars rises, the whole stream is worth less today.

### September: the dollar peaks and global pressure builds

By September 2022, the US was hiking faster than almost anyone else, which made dollar-denominated assets more attractive and pulled money into the dollar from all over the world. The US Dollar Index (DXY), which measures the dollar against a basket of major currencies, surged to an intraday peak around **114.8 on September 28, 2022** — its highest level in two decades. A soaring dollar is its own form of global tightening: it makes commodities (priced in dollars) more expensive for everyone else, and it strains any country or company that borrowed in dollars and now has to repay with a weaker home currency. The strong dollar added to the pain almost everywhere — except for a US saver holding cash, who was earning a rising yield on the world's most-wanted currency. We unpack this in [the dollar: cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity).

### Year-end: the scoreboard nobody had a hedge for

By December 2022, the Fed had taken its policy rate to a 4.50% upper bound, inflation had finally started to roll over (CPI eased to about 6.45% by December), and the year's damage was locked in. The figure below shows the three forces that drove the whole year in one chart: inflation climbing to its 9.06% peak, the Fed's policy rate stair-stepping from 0.25% to 4.50%, and — the line that did the damage — the 10-year real yield surging from about −1% to about +1.7%. When you understand why that green line matters more than the other two, you understand 2022.

![Line chart of 2021 to 2022 showing CPI inflation rising to nine percent the Fed funds rate stepping up to four point five percent and the ten year real yield rising from minus one percent to plus one point seven percent](/imgs/blogs/case-study-2022-stocks-and-bonds-both-fell-2.png)

The story of the year, then, is a single force expressed three ways: inflation forced the Fed to hike, the hikes drove real yields up, and rising real yields re-priced every long-duration asset down at the same time. Now let us see exactly what fell, and what didn't.

## The cross-asset scoreboard: what fell, and the few things that didn't

Here is the full 2022 ledger, the numbers behind the cover figure. These are calendar-year total returns (price change plus any income), so they are the real damage a buy-and-hold investor experienced.

| Asset | 2022 return | What it tells you |
|---|---|---|
| US stocks (S&P 500) | **−18.1%** | Higher discount rate cut the value of future profits |
| US bonds (Agg) | **−13.0%** | Worst year in the index's history — rates up, prices down |
| 60/40 portfolio | **−16.0%** | Worst since 1937 — no offset between the two sleeves |
| US REITs (Nareit) | −24.9% | Property is leveraged and long-duration; rates crushed it |
| High yield (US HY) | −11.2% | Both rate damage and spread widening |
| Bitcoin | −64% | The highest-beta, longest-duration casualty of all |
| Gold | −0.3% | Roughly flat — held value but did not shine |
| Commodities (BCOM) | **+16.1%** | The clear winner — it *was* the inflation |
| Energy equities (S&P) | **~+65%** | The standout — sold what the world was short of |
| Cash (3-mo T-bill) | **+1.5%** | The quiet hero — no drawdown, rising yield |

Three things in that table deserve emphasis, because they are the heart of the case study.

First, **almost everything was red, and the red things were red for the same reason.** Stocks, bonds, REITs, high yield, and crypto did not fall because of five separate stories. They fell because of one story — rising real interest rates — hitting five assets of different durations. The longest-duration assets (crypto, growth stocks, REITs, long bonds) fell most; the shortest-duration assets (cash, short bonds) fell least or rose. This is why the year felt so airless to diversified investors: the "diversification" they owned was mostly different flavors of the same long-duration bet.

Second, **the 60/40 had nowhere to hide.** Normally, the 40% in bonds is the airbag — in 2008 it returned +5.2% while stocks fell 37%, and in 2020 it returned +7.5% while stocks crashed and recovered. The airbag is what makes a 60/40 lose much less than pure stocks in a bad year. In 2022 the airbag didn't deploy: stocks fell 18.1% *and* bonds fell 13.0%, so the blend fell 16.0% — barely better than stocks alone. We unpack the engine of this portfolio in [the stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).

Third, **the winners all shared one trait: they win when inflation is the problem.** Commodities literally *are* the prices that were rising. Energy producers sold the oil and gas the world was suddenly short of, at much higher prices, and printed record profits. Cash, paradoxically, became valuable precisely because the Fed was hiking — the yield on a Treasury bill went from near 0% to over 4% during the year, so for the first time in a long while, doing nothing actually paid. The connecting logic is the regime, which we cover in [late-cycle overheat: when real assets win](/blog/trading/cross-asset/late-cycle-overheat-when-real-assets-win).

#### Worked example: the 60/40 versus pure stocks versus the inflation-aware mix

Let us see the diversification failure in dollars. Take three investors who each start 2022 with \$100,000.

Investor A holds **pure stocks**. Down 18.1%, they end the year with \$100,000 × (1 − 0.181) = **\$81,900** — a loss of \$18,100.

Investor B holds the **classic 60/40**: \$60,000 in stocks and \$40,000 in bonds. The stock sleeve loses 18.1% on \$60,000 = −\$10,860. The bond sleeve, which was supposed to *cushion* that, instead loses 13.0% on \$40,000 = −\$5,200. Total loss: \$10,860 + \$5,200 = **−\$16,060**, leaving \$83,940. The 60/40 lost \$16,060 versus pure stocks' \$18,100 — a hedge that saved only about \$2,000, instead of the \$5,000-plus cushion bonds usually provide. The bond sleeve that was supposed to *offset* the stock loss *added to it* this year.

Investor C holds an **inflation-aware mix**: \$40,000 stocks, \$20,000 short-duration bonds, \$20,000 commodities and energy, \$10,000 gold, \$10,000 cash. Run the sleeves: stocks −18.1% on \$40k = −\$7,240; short bonds about −5% on \$20k = −\$1,000; commodities/energy about +25% on \$20k = +\$5,000; gold −0.3% on \$10k = −\$30; cash +1.5% on \$10k = +\$150. Total: −\$7,240 − \$1,000 + \$5,000 − \$30 + \$150 = **−\$3,120**, leaving \$96,880. The inflation-aware portfolio lost only 3.1% — a swing of roughly **\$13,000** versus the 60/40 — entirely because it held the assets that win when inflation is the shock. The lesson: in 2022, the difference between a brutal year and a mild one was not skill or timing; it was simply *owning the right regime's assets.*

The figure below shows that worked example as a picture — the 60/40 on the left, where both sleeves are red and the year ends down \$16,060, and the inflation-aware mix on the right, where the commodity and energy sleeve turns green and offsets most of the equity loss, ending down just \$3,120.

![Two panel bar chart comparing a one hundred thousand dollar sixty forty portfolio losing sixteen thousand dollars with both sleeves red against an inflation aware mix losing three thousand dollars with a green commodity and energy sleeve](/imgs/blogs/case-study-2022-stocks-and-bonds-both-fell-6.png)

## Why bonds failed as a hedge: the difference between an inflation shock and a growth shock

This is the conceptual heart of the post, and it is the lesson that ties the whole series together. To understand 2022 — and to never be blindsided by it again — you have to understand *why* bonds usually hedge stocks, and the precise condition under which that hedge stops working.

### The usual case: a growth shock, where bonds are the airbag

Most stock-market crashes of the last 40 years were **growth shocks** — events where the economy suddenly looks like it is about to weaken or fall into recession. The 2008 financial crisis was a growth shock; the 2020 COVID crash was a growth shock; the dot-com bust was, in part, a growth shock. In a growth shock, here is the chain of events for your portfolio:

The economy looks like it is weakening, so corporate profits are about to fall, so *stocks fall*. But the same weakening economy means inflation is about to drop (a weak economy has slack demand), so the Fed *cuts* interest rates to stimulate. And when the Fed cuts rates, bond prices *rise* (the mirror image of the worked example above: rates down, prices up). So in a growth shock, stocks fall and bonds rise *at the same time, for the same root reason* — a weakening economy. The bonds are a genuine airbag. This is why, for the entire 2010s, the stock-bond correlation was *negative*: bad news for stocks was good news for bonds, because bad news meant rate cuts.

### The 2022 case: an inflation shock, where the airbag becomes a second airbag-shaped rock

Now flip the shock. In 2022 the problem was not that the economy was weakening; the problem was that *inflation was too high*. That changes everything about how the Fed responds. Faced with inflation, the Fed does not cut — it *hikes*, aggressively, to cool demand and bring prices down. And when the Fed hikes, bond prices *fall*. So the chain becomes:

Inflation is too high, so the Fed *hikes*, so the discount rate on all future cash flows rises, so *stocks fall* (their distant profits are worth less today) *and bonds fall* (their fixed coupons are worth less when new bonds pay more). Both fall, at the same time, for the same root reason — rising rates driven by inflation. The bonds are no longer an airbag; they are a second thing falling on you. The correlation flips *positive*: bad news for stocks (rising rates) is *also* bad news for bonds.

The figure below is the mechanism in one flow: an inflation shock forces the Fed to hike, hiking drives real yields up, rising real yields re-price both stocks and bonds down, the correlation flips positive, and the bond hedge fails. Trace it once and you have the entire causal chain that broke the 60/40.

![Flow diagram showing an inflation shock leading to the Fed hiking real yields rising stocks and bonds both falling the correlation flipping positive and the bond hedge failing](/imgs/blogs/case-study-2022-stocks-and-bonds-both-fell-4.png)

### The single variable that does the work: the real yield

If you want one number that explains 2022, it is the **real yield** — the interest rate *after* subtracting expected inflation. The cleanest way to read it is the yield on a 10-year *Treasury Inflation-Protected Security* (TIPS), a government bond whose payments rise with inflation, so its quoted yield is already inflation-adjusted. We give this variable a whole post of its own in [real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything), because it is, quite literally, the discount rate used to value every long-duration asset on earth.

In December 2021, the 10-year real yield was about **−1.04%** — deeply negative. A negative real yield means money was effectively free in inflation-adjusted terms, which is rocket fuel for any asset whose value comes from the distant future: long bonds, growth stocks, gold, crypto, real estate. When real money is free, you discount far-off cash flows very gently, so things with far-off cash flows are worth a lot. That is the world that inflated everything through 2021.

By October 2022, the 10-year real yield had surged to about **+1.74%** — a swing of nearly **2.8 percentage points** in less than a year. Suddenly money was expensive again in real terms, so the market discounted those distant cash flows much more harshly, and *every long-duration asset re-priced down together*. Bonds, tech stocks, gold, and Bitcoin all fell — not because of separate stories, but because they all share the same sensitivity to that one rising real yield. The real-yield surge is *the* engine of 2022. Everything else is downstream.

#### Worked example: the same future dollar, discounted at −1% versus +2% real

Let us make the discount-rate mechanism concrete, because it is the reason long-duration assets fell together. The value today of a dollar you will receive in the future is that dollar divided by (1 + the rate), compounded for each year you wait. Use the *real* rate, and you are valuing real purchasing power.

Take a dollar of profit you expect to receive in **10 years** — the kind of distant cash flow a growth stock is mostly made of.

At a real discount rate of **−1%** (the late-2021 world), its present value is \$1 ÷ (1 − 0.01)^10 = \$1 ÷ 0.904 ≈ **\$1.11**. A future dollar is worth *more* than a dollar today, because real money is free. That is why expensive, profitless growth stocks could trade at sky-high valuations.

Now move the real rate to **+2%** (the late-2022 world). The same future dollar is now worth \$1 ÷ (1.02)^10 = \$1 ÷ 1.219 ≈ **\$0.82**. The exact same future dollar lost about **26%** of its present value, purely because the discount rate moved — no change in the business, no change in the profit, just the rate. Now apply that to an asset whose value is *entirely* far-future cash flows, and you get a 30%, 50%, or (for Bitcoin) 64% drawdown. The lesson in one line: when the real yield jumps, every asset priced off distant cash flows falls together, and the further out the cash flow, the harder it falls.

### The proof: the correlation flip

The single cleanest piece of evidence for the regime change is the stock-bond correlation itself. *Correlation* is a number from −1 to +1 that measures how two assets move relative to each other: +1 means they move in lockstep, −1 means they move exactly opposite, and 0 means no relationship. For the 60/40 to work as a diversifier, you *want* stocks and bonds to be negatively correlated — when one zigs, the other zags.

The figure below shows that correlation over the decades. For most of the 2000s and 2010s, it was solidly *negative* — around −0.3 to −0.4 — which is exactly why the 60/40 was such a smooth ride: bonds reliably offset stocks. Then in 2022 it snapped to about **+0.55**, its most positive in a generation. Notice the longer history too: before 2000, in the high-inflation decades, the correlation was *also* positive. The negative-correlation era of 2000–2021 was the exception, not the rule — it was the product of a specific, low-inflation, growth-shock-dominated world. When inflation came back as the master variable, the correlation went back to where it sits in inflationary regimes: positive.

![Line chart of the stock bond correlation over the decades showing positive values before 2000 negative values through the 2000s and 2010s and a sharp flip to plus zero point five five in 2022](/imgs/blogs/case-study-2022-stocks-and-bonds-both-fell-3.png)

This is the fact to carry out of the whole series: **the stock-bond correlation is not a constant; it is a function of the regime.** It is negative when growth shocks dominate (the Fed cuts into trouble) and positive when inflation shocks dominate (the Fed hikes into trouble). The 60/40 is a bet that the correlation is negative. In 2022, that bet was simply wrong, because the regime had changed.

## The echo of the 1970s: the same stagflationary logic

Anyone who lived through the 1970s — or read the case study on it — felt a chill in 2022. The two episodes share the same skeleton, and understanding the resemblance (and the limits of it) is the surest way to recognize an inflation regime the next time one appears.

### What 2022 shared with the 1970s

The 1970s were the defining stagflation decade: a money system cut loose from gold, run hot by policy, then hit by oil shocks, producing high-and-rising inflation with weak growth. In that world, *paper assets lost and real assets won.* Stocks were flat in nominal terms but down roughly two-thirds in real terms; long bonds were destroyed by rising yields; gold ran roughly 24x; oil and commodities soared. The 60/40 was exactly the wrong thing to own. We tell that full story in [the 1970s: when stagflation killed the 60/40 and real assets won](/blog/trading/cross-asset/case-study-1970s-stagflation-commodities-win).

2022 rhymed with all of it in miniature: a supply shock (the Ukraine war instead of OPEC), inflation at a multi-decade high, a Fed forced to hike hard, paper assets (stocks and bonds) falling together, and real assets (commodities, energy) leading. The *mechanism* — inflation as the master variable, the Fed hiking, real yields rising, real assets winning and paper losing — was identical. 2022 was a one-year stagflation scare, and the cross-asset playbook that worked was the same one that worked in the 1970s.

### What made 2022 different from 2008 and 2020

The contrast that matters most is not with the 1970s but with the *other* recent crises, because it is the contrast that explains why so many investors were caught off guard. In **2008** (the global financial crisis) and **2020** (the COVID crash), the shock was about *growth and credit*, not inflation — so the Fed *cut* rates to near zero, bonds *rallied* (long Treasuries returned +25.9% in 2008), gold rose, and the 60/40's bond sleeve did exactly its job. A whole generation of investors learned, from 2008 and 2020, that "in a crisis, bonds go up." That lesson was true — for *those* crises. It was a lesson about growth shocks, and they mistook it for a universal law.

The figure below puts the two regimes side by side. In an inflation shock (1970s, 2022), the Fed hikes, real yields rise, bonds fall *with* stocks, the correlation is positive, and commodities/energy/cash win. In a growth shock (2008, 2020), the Fed cuts, real yields fall, bonds *rally* to offset stocks, the correlation is negative, and Treasuries/gold/cash win. Same 60/40 portfolio, opposite behavior — because the Fed does opposite things in the two worlds.

![Comparison matrix of an inflation shock versus a growth shock showing the Fed hiking versus cutting bonds falling versus rallying a positive versus negative correlation and different winners in each regime](/imgs/blogs/case-study-2022-stocks-and-bonds-both-fell-5.png)

#### Worked example: the bond sleeve in 2008 versus 2022 on a \$100,000 portfolio

Let us quantify the difference the regime makes to the *same* 60/40. Both years had a roughly 18–37% stock drop, but the bond sleeve behaved oppositely.

In **2008** (a growth shock): stocks fell about 37%, so the \$60,000 stock sleeve lost about −\$22,200. But the \$40,000 bond sleeve *rose* about 5.2%, gaining +\$2,080. Net 60/40: −\$22,200 + \$2,080 ≈ **−\$20,120**, a −20% year. The bonds didn't fully save you, but they offset more than \$2,000 of the loss and, crucially, *gained while stocks crashed* — they were a true diversifier.

In **2022** (an inflation shock): stocks fell 18.1%, so the \$60,000 stock sleeve lost −\$10,860. The \$40,000 bond sleeve, instead of offsetting, *also fell* 13.0%, losing −\$5,200. Net 60/40: −\$10,860 − \$5,200 = **−\$16,060**, a −16% year. The stock loss was *smaller* than 2008, but the portfolio felt worse to many investors because the part that was supposed to protect them turned on them.

The lesson: the bond sleeve gained in 2008 and lost in 2022 — on the *same* portfolio, in *both* crises — purely because of which shock it was. Know the shock, and you know whether your hedge will show up.

## Common misconceptions

2022 detonated several beliefs that beginners (and plenty of professionals) hold as gospel. Here are the ones worth correcting with numbers.

### "Bonds always cushion a stock crash"

This is the big one, and 2022 is its tombstone. The belief comes from honest experience: in 2000, 2008, and 2020, bonds did cushion the crash, because those were growth shocks and the Fed cut rates. But "bonds always cushion stocks" is really "bonds cushion stocks *in growth shocks*." In an inflation shock, the Fed hikes instead of cuts, and bonds fall *with* stocks — exactly as they did in 2022 (Agg −13.0%) and throughout the 1970s. The bond hedge is conditional on the regime, not unconditional. The whole error is forgetting that the bond's behavior depends on what the *Fed* does, and the Fed does opposite things in the two regimes.

### "The 60/40 is dead"

After 2022, this headline was everywhere, and it is wrong — but wrong in an instructive way. The 60/40 is not *dead*; it is *regime-dependent*. It is an excellent portfolio when inflation is low and stable and shocks are about growth (most of 1982–2021), and a poor one when inflation is the dominant force (the 1970s, 2022). The honest framing is not "the 60/40 is dead" but "the 60/40 needs an inflation hedge to survive an inflation regime." Add a sleeve of commodities, energy, and short-duration assets, and the portfolio becomes robust across regimes again — which is the entire idea behind [all-weather and risk parity: owning every regime](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime). Notice, too, that the 60/40 recovered strongly in 2023 and 2024 once inflation faded. A portfolio that has one terrible year in a regime it wasn't built for is not dead; it is just not all-weather.

### "Gold is the inflation hedge, so it should have soared in 2022"

Gold's 2022 result surprises people: with inflation at a 40-year high, gold returned about **−0.3%** — flat. Wasn't gold supposed to be *the* inflation hedge? The subtlety is that gold competes with cash and bonds, and what hurts gold is *rising real yields*, not inflation itself. Gold pays no interest, so when real yields surge (as they did, from −1% to +1.7%), the opportunity cost of holding a zero-yield asset jumps, and that pressure roughly cancels gold's inflation tailwind. Gold did its real job — it *held its value* while almost everything else fell, which over a −16% year for the 60/40 is genuinely useful — but it did not soar, because the same rising real yield that crushed bonds also capped gold. We dig into this in [gold: money, insurance, or just a rock?](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock).

### "Crypto is a hedge against fiat debasement and money-printing"

This was the loudest crypto thesis of 2020–2021: Bitcoin is "digital gold," an inflation hedge, an escape from central-bank money-printing. Then in 2022 — the highest-inflation year in 40 years — Bitcoin fell about **64%**, more than any major asset. Far from hedging inflation, crypto behaved like the *longest-duration, highest-beta risk asset on the board*: it had soared most when real yields were deeply negative, and it fell most when they surged. The lesson is that in 2022, crypto was not anti-fiat insurance; it was a leveraged bet on free money, and the money stopped being free.

### "If stocks are falling, just move to cash and wait"

Half-right, and 2022 is the rare year it actually worked — which makes it the exception that proves the rule. Cash (T-bills) returned about +1.5% in 2022 and, crucially, had *no drawdown* while everything else bled, so moving to cash genuinely protected you. But this works only in a *rising-rate* environment, where cash yields are climbing and there is no rally to miss. In a normal growth shock, the Fed is cutting rates *toward* zero, cash yields are collapsing, and stocks often bottom and rebound violently while you sit in cash earning nothing — so "wait in cash" usually means missing the recovery. Cash was a hero in 2022 specifically because it is a short-duration asset in a regime that punished duration. We cover this fully in [cash and money markets: the underrated asset](/blog/trading/cross-asset/cash-money-markets-the-underrated-asset).

## How it showed up across assets: the leaders and the casualties

Let us walk the scoreboard asset by asset, because each one teaches a piece of the regime lesson.

### Energy equities: the standout

The single best place to be in 2022 was energy producers. The S&P 500 energy sector returned roughly **+65%** while the broad index fell 18%. The logic was direct: the Ukraine war and years of underinvestment had left the world short of oil and gas exactly when demand recovered, so prices spiked — and energy companies, which sell oil and gas, saw their revenues and profits explode. They had been deeply unloved and cheap going into 2022 (the market had spent a decade preferring growth and tech), which made the move even larger. Energy was the purest expression of the year's logic: *own what the world is suddenly short of.* We trace this linkage in [oil, inflation, and equities](/blog/trading/cross-asset/oil-inflation-and-equities-the-energy-linkage).

### Commodities: the asset that *was* the inflation

Broad commodities (the Bloomberg Commodity Index) returned about **+16.1%** in 2022 — the only major non-energy-equity asset class in the green. This is almost tautological: inflation *is* rising prices for goods, and commodities *are* those goods — oil, gas, wheat, metals. When the CPI is at a 40-year high, the things in the CPI basket are, by definition, going up. Commodities are the one asset class with a structurally *positive* relationship to inflation surprises, which is exactly why they are the core inflation hedge a 60/40 lacks. The catch is that commodities are volatile and pay no income, so they are a hedge to *size* deliberately, not a core holding.

### Stocks: a discount-rate crash, not an earnings crash

The S&P 500's −18.1% is worth dissecting, because it was an unusual kind of bear market. In a normal recession, stocks fall because *earnings* fall. In 2022, earnings actually held up reasonably well — companies were mostly fine. Stocks fell almost entirely because of the *discount rate*: the rising real yield made every future dollar of profit worth less today, so valuations compressed even as profits held. This is why the most expensive, longest-duration stocks (profitless tech, growth darlings) fell 30%, 50%, or more, while cheap, short-duration value and dividend stocks held up far better. It was a re-rating, not a recession. Value beat growth by one of the widest margins in decades.

### Bonds: the worst year in the index's history

The US Aggregate bond index's −13.0% was, in a single word, *historic* — the worst calendar year since the index began. Bonds had not had a year remotely this bad in the modern era, and it happened for the simplest reason: they started the year with tiny yields (around 1.5% on the 10-year), which left enormous room for prices to fall as yields rocketed to nearly 4%. A bond paying almost nothing has very little income to offset a price decline, so the price decline *is* the return. The flip side, often missed in the gloom: by the end of 2022, bonds yielded around 4–5% again, which meant they were finally offering real compensation and were far better positioned to hedge the *next* growth shock. The pain of 2022 reset bonds back to being useful. We cover this in [government bonds: the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration).

### Cash: the quiet hero

Cash returned about +1.5% and, more importantly, never fell. In a year when the 60/40 lost 16%, a flat +1.5% with no drawdown was a top-decile outcome. Cash won because it is the shortest-duration asset there is — its value is unaffected by rising rates, and its *yield* actually rises as the Fed hikes. For the first time since 2007, holding cash paid you a real return. The 2022 lesson about cash is that it is not "the absence of an investment" — in a rising-rate, duration-punishing regime, it is one of the best assets you can own.

### Bitcoin and REITs: the long-duration casualties

At the other extreme, Bitcoin (−64%) and REITs (−24.9%) were the worst major casualties, and for the *same* reason: both are extremely long-duration. Bitcoin's entire value is a far-future story with no current cash flows, so it is maximally sensitive to the discount rate. REITs (real-estate investment trusts) own property, which is leveraged (bought with lots of debt) and valued off long-dated rents — both of which get hammered when rates rise. They fell hardest not because their underlying stories collapsed, but because they sit at the very top of the duration ladder, and 2022 was a year that punished duration above all else.

## The lessons: the inflation-regime playbook

This is the payoff, and it is the single most important regime lesson in the whole series. Everything above was the story; here is what to *do* with it.

### Lesson 1: know which shock you are in before you trust the hedge

The master skill is diagnosis. Before you rely on any hedge, ask one question: *is this an inflation shock or a growth shock?* The tell is what the central bank is doing. If inflation is high and rising and the Fed is *hiking*, you are in an inflation regime — bonds will *not* hedge your stocks, and you need real assets and short duration. If the economy is weakening and the Fed is *cutting*, you are in a growth regime — bonds *will* hedge, and duration is your friend. The same portfolio behaves oppositely in the two worlds, so the diagnosis comes first and the hedge second. The cleanest single signal is the real yield and the direction of Fed policy, which we build into a live read in [reading the regime in real time: the dashboard](/blog/trading/cross-asset/reading-the-regime-in-real-time-the-dashboard).

### Lesson 2: in an inflation regime, own the winners — and shorten duration on everything

When you have diagnosed an inflation shock, the rotation is mechanical. Add the assets that win when the Fed is hiking: **commodities and energy** (they are the inflation), **cash and short-duration bills** (they reprice up with rates and never fall), and **short-duration, pricing-power equities** (value, energy, dividend payers — not profitless long-duration growth). And shorten the duration of everything you own, because duration is precisely what gets punished. You do not need to predict the exact path of inflation; you just need to own the regime's assets and stop holding the regime's victims.

The figure below is that playbook as a matrix: on the left, what to add or overweight (commodities and energy, cash and short bills, short-duration and value equities); on the right, what to trim or rethink (long bonds, long-duration growth, and over-reliance on the 60/40 alone). The numbers in the cells are 2022's actual results, so the playbook is not theory — it is what worked.

![Matrix of the inflation regime playbook with assets to add or overweight on the left including commodities energy cash and short duration and assets to trim or rethink on the right including long bonds long duration growth and the sixty forty alone](/imgs/blogs/case-study-2022-stocks-and-bonds-both-fell-7.png)

### Lesson 3: do not declare the 60/40 dead — give it an inflation hedge

The right response to 2022 is not to abandon the 60/40; it is to *upgrade* it. The 60/40's weakness is a single, specific blind spot: it has no asset that wins when inflation is the shock. Patch that blind spot with a modest sleeve — say 10–20% split across commodities, energy, and short-duration/inflation-linked bonds — and the portfolio becomes robust across both regimes. In 2022 that patch would have turned a −16% year into something far milder, as the worked example showed. The goal is a portfolio that does not *need* you to forecast the regime correctly — it holds something for each one. That is the all-weather idea, and 2022 was its single best advertisement in 40 years.

#### Worked example: a 10% inflation sleeve carved out of a 60/40

Let us size the patch precisely, so "add a sleeve" becomes a number. Start with a standard \$100,000 60/40 (\$60,000 stocks, \$40,000 bonds) and carve out a 10% inflation sleeve by trimming \$5,000 from each side: you now hold \$55,000 stocks, \$35,000 bonds, and \$10,000 split between commodities/energy and short-duration bills (say \$7,000 commodities/energy, \$3,000 cash-like).

Run the 2022 sleeves: stocks −18.1% on \$55,000 = −\$9,955; bonds −13.0% on \$35,000 = −\$4,550; commodities/energy about +25% on \$7,000 = +\$1,750; cash +1.5% on \$3,000 = +\$45. Total: −\$9,955 − \$4,550 + \$1,750 + \$45 = **−\$12,710**, a −12.7% year. Compare that to the unpatched 60/40's −\$16,060 (−16.1%). A modest 10% tilt — not a dramatic, bet-the-house move — recovered roughly **\$3,350** of the loss, turning a −16% year into a −13% one. The lesson: you do not need a large or heroic inflation sleeve to matter; even a 10% slice of the right regime's assets meaningfully softens the blow, because the sleeve is *up* exactly when the rest is down.

### Lesson 4: know what would tell you you're wrong

A regime call is only as good as its exit. The inflation playbook is right *while* inflation is high and the Fed is hiking; it becomes wrong when that reverses. The signals to watch for the regime ending: inflation rolling over decisively (CPI falling back toward 2–3%), the Fed pausing and then *cutting*, and the real yield peaking and starting to fall. When those flip, the regime is rotating back toward growth-shock dynamics — bonds become useful hedges again, duration becomes your friend, and the commodity/energy leadership fades. Indeed, that is roughly what happened in 2023–2024: inflation eased, the hiking stopped, and the 60/40 promptly had two strong years. The point is to hold the inflation playbook with conviction *and* with a clear, pre-committed list of what would make you put it down.

### A note on sizing, humility, and what this is not

Two honest caveats. First, this is educational, not advice — none of it is a recommendation to buy or sell any specific asset; it is a framework for understanding why a diversified portfolio behaved the way it did. Second, regimes are easier to label in hindsight than in the moment. In early 2022, plenty of smart people still believed inflation was transitory; the regime call was *not* obvious in real time. So size your tilts with humility: an inflation hedge is a sleeve you add to a diversified core, not a bet-the-portfolio conviction. The deepest lesson of 2022 is not "always hold commodities" — it is "your diversification is only as good as your understanding of *which shock* it was built to survive." Bonds are a magnificent hedge against the wrong thing, and a magnificent hedge against the right thing is the difference between a −16% year and a −3% one.

## Further reading & cross-links

2022 is the capstone because it brings the whole series' machinery to bear on one year. To go deeper on the pieces:

- [The stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) — why the 60/40 works, and the regime-dependence of the correlation that 2022 exposed.
- [Real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) — the −1% to +1.7% swing that was the engine of 2022, and why it re-prices all long-duration assets together.
- [The 1970s: when stagflation killed the 60/40 and real assets won](/blog/trading/cross-asset/case-study-1970s-stagflation-commodities-win) — the original inflation-regime case study that 2022 rhymed with.
- [Late-cycle overheat: when real assets win](/blog/trading/cross-asset/late-cycle-overheat-when-real-assets-win) — the regime that 2022 lived in, and why commodities and energy lead it.
- [All-weather and risk parity: owning every regime](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) — how to build the inflation hedge into the portfolio so the next 2022 is a footnote, not a catastrophe.

The one sentence to carry out of the entire series: **diversification is not a fixed property of a portfolio — it depends on the regime, and the regime depends on whether the shock is about growth or about inflation.** Know which shock you are in, and you will know whether your hedge is going to show up.
