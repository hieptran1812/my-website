---
title: "How Policy Sets the Bond Market: The Yield Curve"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How the policy rate anchors the front end of the yield curve, how expectations and term premium build the long end, what each curve shape says, and why the 2s10s inversion became policy's most famous recession siren."
tags: ["monetary-policy", "yield-curve", "bond-market", "term-premium", "treasury-yields", "fed-funds", "2s10s", "inversion", "qe-qt", "duration", "central-banks", "asset-valuation"]
category: "trading"
subcategory: "Policy & Markets"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The yield curve is not one price; it is a whole structure that policy builds. The central bank's policy rate *anchors the front end* — a 2-year Treasury yield is essentially the market's average forecast of where the policy rate will sit over the next two years. The *long end* adds a second ingredient, the **term premium**, the extra yield you demand for locking your money up for a decade. Quantitative easing and tightening bend the long end directly. The *shape* of the curve — normal, flat, inverted, steep — is the bond market's verdict on growth and the path of policy.
>
> - A long yield = the average short rate the market expects over the bond's life **+ a term premium**. The Fed sets the first ingredient directly and influences the second.
> - The **2s10s spread** (the 10-year yield minus the 2-year yield) is the single most-watched recession signal in finance: when the Fed hikes the front end above where the market thinks rates will settle, the curve inverts and warns of a downturn.
> - In 2022-23 the Fed hiked the front end to 5.50% while the 10-year lagged near 4%, inverting the 2s10s to about **−108 basis points** in July 2023 — the deepest inversion since 1981, the Volcker era. Then the 2025 cuts began re-steepening it even as tariffs added term premium to the long end.
> - The one number to remember: a 10-year bond with a duration of about 8.5 loses roughly **8.5%** of its price for every one-percentage-point rise in its yield. The curve is where policy turns into that price.

In the eighteen months from March 2022 to the summer of 2023, the Federal Reserve raised its policy rate from a floor near zero to a peak of 5.25-5.50% — the fastest tightening cycle since Paul Volcker broke the back of inflation in the early 1980s. The front end of the Treasury market did exactly what the textbook says it should: the 2-year yield, which tracks where investors think the policy rate is headed, climbed in lockstep, peaking above 5%. But the 10-year yield, the benchmark long rate that sets mortgages and corporate borrowing costs around the world, refused to keep up. It lagged, stalling near 4% even as the front end blew past it.

The result was an *inversion*: a curve that sloped downward, where you got paid **more** to lend money for two years than for ten. By July 2023 the gap between the 10-year and 2-year yields — the famous "2s10s" spread — reached about −108 basis points, more than a full percentage point inverted. That was the deepest inversion since 1981. To anyone who watches the bond market, an inversion of that magnitude is not an abstraction; it is a siren. An inverted curve has preceded every U.S. recession of the past half-century. The Fed had, in effect, built a recession warning into the price of government debt.

Then the story turned again. As the Fed began cutting in late 2024 and through 2025 — three cuts to a 3.50-3.75% range by December 2025 — the front end fell, and the curve began to re-steepen, climbing back to a positive 50-plus basis points. But the long end stayed sticky, propped up by a new force: the 2025 tariff shock, which raised the *term premium* by injecting inflation uncertainty and a heavier supply of Treasuries into the long end of the curve. The same curve that had inverted as a recession warning was now steepening for two different reasons at once — cuts at the front, term premium at the back. Reading which was which became the whole game.

![Pipeline showing the policy rate flowing through the front end, term premium, the long end, QE and QT, the curve shape, and asset repricing](/imgs/blogs/how-policy-sets-the-bond-market-the-yield-curve-1.png)

The chain above is this entire post in one line. A central bank pulls a lever — the policy rate. That lever anchors the *front end* of the curve. The *long end* is built from two ingredients: the market's expectation of average future short rates, plus a *term premium* for bearing duration risk. Balance-sheet policy — QE and QT — bends the long end directly. The result is a *curve shape*, and that shape reprices every bond, signals the path of growth, and flashes the recession warning the whole market watches. Let us build it from the ground up.

## Foundations: what a yield curve actually is

Strip away the jargon and a government bond is a simple promise: lend the Treasury a sum of money today, and it will pay you a fixed stream of interest and then return your principal on a set date. The **yield** of that bond is the single annual interest rate that makes the price you pay today equal to the present value of all those future payments. It is the return you lock in if you buy the bond and hold it to maturity. A higher price means a lower yield, and a lower price means a higher yield — the two move in opposite directions, the seesaw at the heart of all fixed income (we cross-link the full mechanics below).

Now here is the key fact that makes a *curve*: the Treasury does not borrow at one maturity. It issues bills that mature in weeks or months, notes that mature in 2, 5, 7, and 10 years, and bonds that mature in 20 and 30 years. Each of those has its own yield. If you plot those yields on a chart — maturity on the horizontal axis, yield on the vertical axis — and connect the dots, you get the **yield curve**: a snapshot of the interest rate the government pays to borrow money for every length of time, from overnight to thirty years.

The curve is not a static thing. It moves every day as the market reprices the future. And crucially, the two ends of the curve are driven by *different forces*. This is the single most important idea in the whole subject:

- The **front end** (bills and the 2-year note) is anchored almost entirely by the *policy rate* and where the market thinks it is going. The Fed controls the overnight rate directly, and short-dated yields are essentially a forecast of the average overnight rate over the bond's short life. When the Fed hikes, the front end follows almost mechanically.
- The **long end** (the 10-year and 30-year) is driven by the market's *long-run expectations* for the average policy rate, plus a **term premium** — extra compensation for the risk of locking your money up for a decade or more. The Fed influences the long end, but it does not control it the way it controls the front.

This split is why the curve has a *shape* at all. If every maturity were priced off the same number, the curve would be flat. It is the gap between "where rates are now" (the front end) and "where rates will average over the long run, plus a premium" (the long end) that gives the curve its slope. And that slope is information. It is the bond market telling you what it expects the economy and policy to do.

### The policy rate: the anchor at the front

The starting point of the whole structure is the **policy rate** — in the United States, the federal funds rate, the rate at which banks lend reserves to each other overnight. The Fed sets a target *range* for this rate (for example, 5.25-5.50% at the 2023 peak) and uses its tools to keep the effective rate inside that band. This is the one number the central bank moves directly, and it is the foundation everything else is built on.

Why does the overnight rate anchor the front end of the curve? Think about a simple arbitrage. Suppose the Fed has the overnight rate at 5% and is expected to hold it there for the next two years. You could buy a 2-year Treasury note, or you could roll over an overnight deposit at 5% every single night for two years. If the 2-year note yielded much more than 5%, everyone would buy it and sell the overnight deposit, pushing the note's price up and its yield down until the two roughly matched. If it yielded much less, the reverse. The result: **the 2-year yield is approximately the average overnight rate the market expects over the next two years.** The front end is a forecast of policy.

This is why the front end moves so tightly with the Fed. When the Fed hikes — or, more precisely, when the market revises its forecast of where the Fed is going — the 2-year yield reprices within minutes. It is the purest expression of monetary policy in the entire market. (For the statistical version of this relationship — how tightly the front end actually tracks the expected fed funds path — see the macro-correlations post linked at the end.)

![Step chart of the fed funds upper bound against the 10-year Treasury yield from 2019 to 2026 showing the front end leading and the long end lagging](/imgs/blogs/how-policy-sets-the-bond-market-the-yield-curve-2.png)

The chart above makes the split visible. The blue step line is the fed funds upper bound — the lever, moving in discrete meeting-day jumps. The orange line is the 10-year yield — the long end, a smoother market price. Notice three things. First, in the 2022-23 hiking cycle, the blue line shoots up to 5.50% while the orange line climbs more reluctantly and tops out near 4%, leaving the front end *above* the long end — that is the inversion. Second, the 10-year often moves *ahead* of the Fed: in late 2024 and 2025 it was already pricing the cutting cycle. Third, when the Fed finally began cutting in 2025, the front end dropped while the long end stayed sticky — the re-steepening. The front end follows policy almost mechanically; the long end has a mind of its own.

#### Worked example: pricing a 2-year note off an expected hike path

Suppose the fed funds rate is at 4.50% today, and the market expects the Fed to hike to 4.75% in three months, to 5.00% three months after that, and then hold at 5.00% for the rest of the two years. What should the 2-year note yield?

The 2-year yield is roughly the *average* expected overnight rate over the 24 months. Break it into the expected rate in each quarter:

```
Quarter 1 (months 0-3):    4.50%
Quarter 2 (months 3-6):    4.75%
Quarter 3 (months 6-9):    5.00%
Quarters 4-8 (months 9-24): 5.00%  (held)
```

Now take the time-weighted average. Three months at 4.50%, three months at 4.75%, and the remaining eighteen months at 5.00%:

```
Average = (3 x 4.50% + 3 x 4.75% + 18 x 5.00%) / 24
        = (13.50 + 14.25 + 90.00) / 24
        = 117.75 / 24
        = 4.91%
```

So the 2-year note should yield about **4.91%**, even though the Fed's rate is only 4.50% today. The note is pricing in the hikes that have not happened yet. If the Fed then *surprises* by signaling it will hold at 4.50% instead, the expected average drops to 4.50% and the 2-year yield falls by about 41 basis points — its price rises — the instant the market reprices the path. The front end is a forecast, so it moves on *changes to the forecast*, not just on the hikes themselves. That is why a 2-year yield can fall on the day of a hike, if the hike is smaller or more dovish than the market feared.

### Who actually buys the curve

There is a second force on the curve that the expectations story alone misses, and it becomes the star of the modern story: **supply and demand for the bonds themselves.** The Treasury does not issue a fixed amount of debt; it issues whatever the government's deficit requires, across the maturity spectrum, every single week. When the deficit is small, supply is modest and the curve trades mostly on rate expectations. When the deficit is large — as it has been in the 2020s, running 6% of GDP or more — the Treasury floods the market with new bonds, and *someone* has to buy them. That someone demands a price, and the price of absorbing a flood of long-dated supply is a higher term premium and a higher long yield.

Who are the buyers? Roughly three groups, and each has its own behavior. **Foreign official buyers** — central banks like the People's Bank of China and the Bank of Japan, plus sovereign wealth funds — historically soaked up enormous amounts of long Treasuries as the world's reserve asset; their steady, price-insensitive demand suppressed the term premium for decades. **The Fed itself** is a buyer during QE and a *non*-buyer (a net seller, effectively) during QT, which is exactly the lever we discussed. And **private investors** — pension funds, insurers, banks, and traders — are the marginal, price-sensitive buyers who set the term premium day to day. When foreign demand wanes and the Fed steps back, the price-sensitive private buyers become the marginal holders, and they demand more yield to take down the supply. This is why the term premium rose through 2023-25: heavy issuance met a smaller pool of price-insensitive buyers, so the private market had to clear the supply at a higher yield. The curve is not just a forecast of policy; it is a market that has to *clear*, and clearing a flood of supply lifts the long end.

## How the long end is built: expectations plus term premium

The front end is almost pure policy-rate expectations. The long end is more interesting, because it has a second ingredient. The 10-year yield is built from two parts:

1. **The expectations component** — the market's forecast of the *average* overnight policy rate over the entire ten years. By the same arbitrage logic as the 2-year, a 10-year yield should roughly equal the average short rate you expect to earn by rolling over short-term debt for a decade. If you think the Fed will average 3.5% over the next ten years, that 3.5% is the backbone of the 10-year yield.

2. **The term premium** — the *extra* yield investors demand on top of the expectations component, as compensation for the risks of locking up their money for ten years instead of rolling short. Those risks are real: inflation could surprise to the upside and erode the value of your fixed coupons; the supply of bonds could swell and depress their price; and a long bond's price swings far more than a short one when yields move (its *duration* is higher), so you are taking more mark-to-market risk. The term premium is the market's price for all of that.

In equation form, the cleanest way to think about it is:

```
10Y yield = (average expected short rate over 10 years) + (term premium)
```

This decomposition is the whole reason the long end can diverge from the front. The Fed sets the policy rate, which dominates the *first* ingredient. But the Fed does not directly set the term premium — that is a market-determined quantity, driven by inflation uncertainty, the supply of government debt, foreign demand for Treasuries, and the Fed's own balance-sheet operations. When the term premium rises, the long end rises even if expectations for the policy rate do not change at all. This is precisely what happened in 2025: the Fed was *cutting* (lowering the expectations component), yet the 10-year stayed sticky because tariffs and large deficits were pushing the *term premium* up.

![Stacked bar figure showing a 2-year and a 10-year yield each split into an expected average short rate band and a term premium band](/imgs/blogs/how-policy-sets-the-bond-market-the-yield-curve-6.png)

The figure above makes the construction concrete. The 2-year yield (left) is almost entirely the blue "expected average short rate" band — its term premium is tiny, because two years is not long to lock up money. The 10-year yield (right) has a much taller amber term-premium band stacked on top of its expectations band. The lavender box on the right is the key policy hook: **QE shrinks the term-premium band, and QT lifts it.** When the central bank buys long-dated Treasuries, it removes duration risk from the market's hands and compresses the term premium; when it lets bonds run off (QT), it pushes that duration risk back onto private investors and the term premium rises. This is how balance-sheet policy bends the long end without touching the policy rate at all.

### The term premium is where the action is

For most of the 2010s, the term premium was unusually low — at times even *negative*, meaning investors accepted less yield on a 10-year bond than the expected average short rate alone would justify. Why would anyone do that? Three reasons: relentless central-bank QE that soaked up the supply of long bonds; enormous foreign demand for Treasuries as the world's safe asset; and a deep belief, after years of low inflation, that rates would stay low forever. All three suppressed the term premium and flattened the curve.

The 2020s reversed much of this. Inflation came roaring back in 2021-22, making the future path of rates genuinely uncertain — a higher term premium. The Fed switched from QE to QT in 2022, pushing duration risk back to private hands. And the U.S. government began running very large deficits, flooding the market with new Treasury supply that someone had to absorb. By 2025, the 2025 tariff shock added a fresh layer of inflation uncertainty. Every one of these forces lifts the term premium, and a higher term premium props up the long end of the curve — which is why the 10-year stayed stubbornly near 4-4.5% even as the Fed cut the policy rate.

#### Worked example: how \$500B of QT lifts the long end via term premium

Quantitative tightening (QT) is the Fed letting its bond holdings mature without reinvesting the proceeds. When a \$10 billion Treasury note in the Fed's portfolio matures, the Treasury must repay it — and to fund that repayment, the Treasury issues a *new* bond to the public. So the net effect of QT is that the *public* has to absorb bonds the Fed used to hold. More supply for private investors to digest means a higher term premium.

How big is the effect? A rough rule of thumb from the research literature is that roughly every **\$1 trillion** of QT lifts the 10-year term premium by about **5 to 25 basis points**, with a central estimate near 10 basis points (the range is wide because the effect depends on market conditions). Take the midpoint and scale it:

```
QT amount absorbed by the market:    $500 billion
Term-premium sensitivity (midpoint): ~10 bp per $1,000 billion
Term-premium lift = $500B / $1,000B x 10 bp
                  = 0.5 x 10 bp
                  = 5 bp
```

So \$500 billion of QT lifts the 10-year yield by roughly **5 basis points** through the term-premium channel alone — *without the Fed touching the policy rate*. Now run it on a 10-year bond with a duration of about 8.5. A 5 basis point yield rise translates to a price hit of about `8.5 x 0.05% = 0.43%`. That sounds small, but on the roughly \$28 trillion of marketable Treasury debt, even a few basis points of term-premium lift moves hundreds of billions of dollars of market value, and it tightens financial conditions for every borrower whose rate keys off the 10-year. The lesson: the Fed has *two* dials on the long end — the policy rate through expectations, and the balance sheet through the term premium. (For the full mechanics of how QE and QT move markets, see the macro-trading cross-links below.)

### Real yields: the long end with inflation stripped out

There is one more cut that separates *real* policy stance from the noise of inflation, and it is the cleanest read on how restrictive the long end actually is: the **real yield.** The yields we have discussed so far are *nominal* — they include compensation for expected inflation. But a lender ultimately cares about purchasing power, not headline percentages. If a 10-year bond yields 5% and inflation is expected to average 3% over the decade, the lender's *real* return — the growth in actual buying power — is only about 2%. That 2% is the real yield, and it is what truly determines whether money is cheap or expensive in the long run.

You can read the real yield directly off the market, because the Treasury issues inflation-protected securities (TIPS) whose principal and coupons rise with inflation. The yield on a 10-year TIPS *is* a 10-year real yield — the return you lock in over and above realized inflation. Subtract the TIPS real yield from the nominal 10-year yield and you get the market's expected inflation over ten years (the "breakeven inflation rate"). So the nominal long yield decomposes one more level: **nominal 10Y = expected real rate + expected inflation + the real-and-inflation term premia.** The Fed's policy rate works on the *real* side: when the Fed hikes the nominal policy rate faster than inflation, real yields rise, and rising real yields are what actually tighten the economy.

![Line chart of the 10-year TIPS real yield from 2020 to 2026 showing the swing from minus one percent in 2021 to plus two point five percent in 2023](/imgs/blogs/how-policy-sets-the-bond-market-the-yield-curve-5.png)

The chart above shows how violently the real yield swung in the recent cycle, and it is the single best picture of the long end's true policy stance. In 2021, with the Fed at zero and QE in full force, the 10-year real yield fell to about **−1.1%** — meaning a lender in 10-year TIPS was guaranteed to *lose* about 1.1% of purchasing power a year. Money was free, even *cheaper* than free in real terms; that negative real yield is exactly why risk assets, growth stocks, and gold ran so hard in 2021. Then, as the Fed hiked aggressively and switched to QT, the real yield rocketed to about **+2.5% by October 2023** — the most restrictive long-end real stance in over a decade. A swing from −1.1% to +2.5% is a 3.6-percentage-point tightening in the *real* cost of long money, and it is what repriced everything from housing to high-growth equities. When you want to know whether the long end is genuinely easy or genuinely tight, ignore the nominal headline and watch the real yield. (The macro-correlations post on real yields, linked below, shows why this is the cleanest macro correlation in the book.)

## Duration: why a long bond's price is so sensitive to its yield

Before we read the shapes of the curve, we need one more building block, because it explains why the long end matters so much: **duration**. Duration is the single most important number in fixed income, and we treat it only briefly here because the fixed-income series builds the full math — but you need the intuition.

Duration measures how much a bond's *price* changes when its *yield* changes. Concretely, a bond with a duration of 8.5 will lose approximately 8.5% of its price if its yield rises by one percentage point, and gain approximately 8.5% if its yield falls by one point. Duration rises with maturity (a 10-year bond has more duration than a 2-year) and falls with the coupon rate (a bond paying big coupons gets its money back faster, so it is less sensitive). For a plain 10-year Treasury, duration is typically around 8 to 9.

This is *why the long end is where the price risk lives*. A 2-year note with a duration near 2 barely moves when yields shift; a 10-year note with a duration near 8.5 moves four times as much; a 30-year bond with a duration near 18 swings violently. When the curve moves, the long end is where fortunes are made and lost — and where the term premium has to compensate you for that volatility.

#### Worked example: a 10-year bond's price hit from a +1pp yield move

Suppose you own a 10-year Treasury note trading at par (a price of \$100 per \$100 of face value), with a 4.0% coupon and a duration of 8.5. The Fed's QT and a tariff-driven term-premium rise push the 10-year yield up by one full percentage point, from 4.0% to 5.0%. What happens to your bond?

The first-order price change is simply minus the duration times the yield change:

```
Price change  ~  -Duration x (change in yield)
              =  -8.5 x (+1.0%)
              =  -8.5%
```

Your \$100 bond falls to about **\$91.50** — a loss of \$8.50 per \$100 of face value, from a yield move of a single percentage point. Nothing about the bond changed: it still pays a 4.0% coupon and still returns \$100 at maturity. The only thing that changed is the rate at which the market discounts those payments. Convexity (the curvature that duration ignores) softens the blow a little — the true loss is closer to 8.1% than 8.5% — but the headline is brutal and correct: **a one-point rise in the 10-year yield is an 8-to-9 percent capital loss on the bond.** This is exactly why 2022 was the worst year for bonds in modern history: the 10-year yield rose from 1.5% to over 4%, and long-duration bond portfolios fell 15-20%. The curve is where policy turns into that price. (For how to measure and trade this precisely, see the modified-duration and DV01 cross-link below.)

## The shapes of the curve and what each one says

Now we can read the curve. Its *shape* — the relationship between short and long yields — is the bond market's compressed forecast of growth and policy. There are four classic shapes, and each tells a different story.

![Hand-drawn chart of four idealized yield-curve shapes: a normal upward curve, a flat curve, an inverted downward curve, and a steep curve](/imgs/blogs/how-policy-sets-the-bond-market-the-yield-curve-3.png)

**Normal (upward-sloping).** The most common shape: short yields are low, long yields are higher, the curve slopes gently up. This is the healthy default. It says the economy is growing, the policy rate is at or below its long-run level, and investors demand a positive term premium for lending long. A normal curve is what you see in the middle of an expansion. Lending long pays more than lending short, as compensation for the extra duration risk.

**Flat.** Short and long yields are roughly equal; the curve is nearly horizontal. This usually appears late in a tightening cycle, when the Fed has raised the front end up toward the level of the long end. A flat curve is a transition signal — the cycle is maturing, and the market is unsure whether the next move is more hikes or the first cut. It is the calm before the curve either re-steepens (if the Fed pauses and growth holds) or inverts (if the Fed keeps hiking into a slowdown).

**Inverted (downward-sloping).** Short yields are *higher* than long yields; the curve slopes down. This is the recession siren. It happens when the Fed hikes the policy rate above where the market thinks rates will settle over the long run — so the front end is jacked up by current tight policy, while the long end is pulled *down* by the market's expectation that the Fed will have to *cut* rates in the future to rescue a slowing economy. An inverted curve is the market saying: "Policy is too tight; a recession is coming; the Fed will be cutting before long." We will dig into exactly why this is such a reliable signal below.

**Steep.** Long yields are much higher than short yields; the curve slopes up sharply. This typically appears at the *start* of a recovery, after the Fed has cut the policy rate to the floor (pinning the front end low) while the market prices a return to growth and inflation in the future (pulling the long end up). A steep curve is a reflation signal — the economy is healing, and the term premium is rebuilding. Early 2021 was a textbook steep curve: the Fed at zero, the long end climbing on reopening optimism.

The crucial discipline is to **read the slope, not the level.** A 5% 10-year yield can be part of a steep curve (if the 2-year is at 3%) or an inverted one (if the 2-year is at 5.5%). The *level* tells you how restrictive policy is overall; the *slope* tells you what the market expects to happen next. Both matter, but the slope is the forecast.

![Snapshot of the US Treasury yield curve on four dates showing a flat low curve in 2020, a normal upward curve in 2021, a deep inversion in 2023, and a re-steepened curve in 2025](/imgs/blogs/how-policy-sets-the-bond-market-the-yield-curve-4.png)

The chart above shows these shapes in *real* Treasury data on four dates. In July 2020 (gray), the whole curve is crushed low and nearly flat — the Fed at zero plus QE flattened everything. In June 2021 (green), a clean normal upward curve as the recovery took hold. In July 2023 (red), the deep inversion — the 3-month bill at 5.45% towering over the 10-year at 3.96%, the curve sloping down across the belly. And by December 2025 (blue), the curve has re-steepened: the front end has come down with the Fed's cuts, while the long end has lifted on term premium, restoring a positive slope. One market, four shapes, four different messages about growth and policy.

## The 2s10s inversion: policy's most famous recession signal

Of all the slopes you can measure on the curve, one has become iconic: the **2s10s spread**, the 10-year yield minus the 2-year yield. When this number is positive, the curve is normally sloped; when it goes negative, the curve is *inverted* in the most-watched part of its range. The 2s10s inversion has preceded every U.S. recession since the 1960s, usually by 6 to 18 months. It is, with the possible exception of the policy rate itself, the most closely followed number in macro.

Why does it work? The logic flows directly from how the two ends are built. The 2-year yield reflects *current and near-term* policy — when the Fed is in a hiking cycle to fight inflation, the 2-year is high. The 10-year yield reflects the *long-run average* policy rate plus term premium — and the long-run average is pulled down whenever the market believes today's tight policy will eventually break the economy and force the Fed to cut. So the curve inverts precisely when the market's view is: "policy is tight *now*, but it cannot stay this tight; the Fed is squeezing the economy hard enough that it will have to reverse, probably into a recession." An inversion is the bond market pricing the Fed's own future easing.

![Area chart of the 2s10s spread in basis points from 2021 to 2026 showing the inversion that bottomed at minus 108 basis points in July 2023 and then re-steepened](/imgs/blogs/how-policy-sets-the-bond-market-the-yield-curve-7.png)

The chart above tracks the 2s10s spread through the most recent cycle. It was comfortably positive in 2021 (a normal curve, Fed at zero). It crossed below zero in July 2022 as the Fed hiked aggressively into a slowing economy. It deepened relentlessly, bottoming at about **−108 basis points in July 2023** — the deepest inversion since 1981, the height of the Volcker squeeze. It then climbed back toward zero through 2024 and turned positive in late 2024, re-steepening through the 2025 cutting cycle to a positive 50-plus basis points. The red region is the inverted, recession-warning zone; the green region is the normal, upward-sloping zone. Forty years apart, an inversion this deep had only one precedent — and the precedent was the worst inflation fight in modern memory.

#### Worked example: the inversion — why a 5.5% 2-year above a 4.0% 10-year is the market pricing cuts

Let us make the inversion arithmetic concrete, because this is the heart of the signal. In mid-2023, suppose the 2-year yields 5.5% and the 10-year yields 4.0% — a 2s10s spread of −150 basis points (close to the real trough). What is the market actually saying?

The 2-year at 5.5% says: *the average policy rate over the next two years is expected to be about 5.5%.* The Fed is tight and will stay tight for a while. Fine.

The 10-year at 4.0% says: *the average policy rate over the next ten years, plus a term premium, is about 4.0%.* Assume a modest term premium of, say, 0.5%. Then the *expectations component* of the 10-year is about `4.0% − 0.5% = 3.5%`. So the market expects the policy rate to **average 3.5% over the next decade.**

Now do the subtraction. If the rate averages 5.5% over the first two years but 3.5% over the full ten years, then the rate in *years 3 through 10* must average well below 3.5% to drag the ten-year average down that far. Solve it:

```
10-year average = 3.5%
First 2 years average = 5.5%
Let X = average rate over years 3-10 (8 years)

(2 x 5.5% + 8 x X) / 10 = 3.5%
11.0% + 8X = 35.0%
8X = 24.0%
X = 3.0%
```

The market is pricing the policy rate to **fall from 5.5% to an average of 3.0%** over years 3-10 — a series of cuts totaling roughly 250 basis points. The only reason the Fed cuts that much is that the economy weakens enough to need rescuing. **That is what an inverted curve is: the bond market pricing in the cuts that follow a recession.** The inversion is not a mystical omen; it is the arithmetic of the market saying current policy is too tight to last.

A caveat worth keeping: the signal has a real track record but it is *not* infallible, and its lead time is long and variable. The 2022-23 inversion was the deepest in four decades, yet the recession it warned of did not arrive on the old schedule — the economy proved more resilient than the curve expected, partly because of the enormous fiscal support still in the system and partly because the term premium, not just a recession forecast, was doing some of the work. An inversion is a high-quality warning, not a guarantee or a timing tool. (The macro-correlations and fixed-income posts linked below quantify the hit rate and the lead times.)

## Common misconceptions

**"The Fed sets the 10-year yield."** No. The Fed sets the *overnight* policy rate directly, which anchors the front end. The 10-year is a market price built from the *expected average* policy rate over ten years plus a term premium. The Fed *influences* the long end — through guidance about the rate path, and directly through QE/QT on the term premium — but it does not set it. In 2025 the Fed cut the policy rate three times while the 10-year barely moved, because the term premium rose to offset the lower expected path. If the Fed set the 10-year, that could not happen.

**"An inverted curve means a recession is imminent."** The lead time is long and variable — historically 6 to 24 months, and the 2022-23 inversion stretched the lag further than usual. An inversion is a reliable *direction* signal, not a *timing* signal. Treating "the curve inverted last week" as "sell everything today" has burned plenty of traders; the curve often inverts a year or more before the downturn, and stocks have frequently risen during the inverted window.

**"A higher 10-year yield is always bad for bonds."** A higher yield *today* means a capital loss on bonds you already own (the price fell to lift the yield). But it also means a *higher future return* on bonds you buy now — you are locking in a richer coupon. The seesaw cuts both ways. After the brutal 2022 repricing, the higher yields of 2023-25 made Treasuries genuinely attractive again for the first time in a decade. "Bonds had a terrible year" and "bonds are now a good buy" can both be true, because one is about the price drop and the other is about the new yield.

**"Term premium is just a fudge factor."** It is a real, measurable quantity that policy moves on purpose. QE compresses it; QT lifts it; big deficits and inflation uncertainty raise it; foreign safe-asset demand lowers it. The Fed spent the 2010s deliberately suppressing the term premium through QE to ease financial conditions when the policy rate was already at zero. The term premium is not a residual — it is a policy target in its own right.

**"The curve is the same everywhere."** Each currency has its own curve, set by its own central bank. In the 2010s Japan's curve was pinned near zero across its whole length by yield-curve control; Europe's had negative yields out to ten years. The U.S. curve is the world's benchmark, but "the yield curve" is always a *specific country's* curve, built by *that* central bank's policy.

**"A steepening curve is always good news."** It depends entirely on which end is moving. A "bull steepener" — where the curve steepens because the *front end falls* (the Fed is cutting) — is usually risk-friendly, the classic start-of-recovery signal. A "bear steepener" — where the curve steepens because the *long end rises* (term premium climbing on inflation or fiscal worry) — is the opposite: it tightens conditions for every long-term borrower and often signals stress about inflation or debt sustainability. The 2025 steepening had elements of both. The word "steepening" tells you nothing on its own; you have to ask whether the front end fell or the long end rose. The same is true of flattening: a "bull flattener" (long end falling on growth fear) and a "bear flattener" (front end rising on hikes) carry opposite messages. Always name which end moved.

## Case studies: four curves policy built

The clearest way to see policy build the curve is to watch it happen at four real moments — each one a different shape, each one made by a specific policy. Keep the decomposition in mind as you read each: every curve is the *front end* (the policy rate and its expected path) plus the *long end* (the expected average rate over the long run, plus a term premium that QE/QT and supply move). In each case below, ask which of those pieces the policy was pushing on, and the shape will explain itself.

### The 2022-23 inversion: the deepest since Volcker

This is the headline case. Inflation hit a 40-year high of about 9% in mid-2022, and the Fed responded with the fastest tightening cycle since the early 1980s — 525 basis points of hikes in sixteen months, lifting the policy rate from a floor near zero to 5.25-5.50% by July 2023. The front end of the curve tracked the hikes almost perfectly: the 2-year yield climbed from under 1% in early 2022 to over 5% by late 2023.

The long end refused to follow. The 10-year yield rose too, but it stalled near 4%, well below the 2-year. Why? Because the market did not believe a 5.5% policy rate could last. Investors expected the aggressive hikes to slow the economy and force the Fed to cut — so the long-run *expected average* rate, which drives the 10-year, stayed far below the current rate. The gap inverted the curve. By July 2023 the 2s10s reached about −108 basis points, the deepest inversion since 1981.

The arithmetic was exactly the worked example above: a 5.5% front end over a 4.0% long end, with the long end implying the policy rate would average well under 3.5% over the decade — a forecast of substantial future cuts, which only happen when the economy weakens. The curve had encoded a recession warning. The recession was slower to arrive than history suggested, but the *mechanism* was textbook: tight policy at the front, an expectation of cuts at the back, an inverted curve in between.

### The 2020 collapse: zero plus QE flattens everything

Now run the policy in reverse. When COVID hit in March 2020, the Fed slashed the policy rate to zero in a matter of days and launched unlimited QE — committing to buy Treasuries and mortgage bonds in whatever quantity it took to stabilize markets. The effect on the curve was dramatic. The front end was pinned at the floor by the zero policy rate. And the long end was crushed *down* by two forces at once: the market's expectation that rates would stay near zero for years, and the Fed's massive QE purchases, which absorbed the supply of long bonds and compressed the term premium toward zero — even negative.

The result was the gray curve in our snapshot chart: the entire structure flattened down near the floor, with the 10-year falling to about **0.5%** at its low in early 2020 and the 30-year barely above 1%. This is QE's signature on the curve. The policy rate flattens the front end at zero; QE flattens the long end by removing duration risk from the market. When the Fed wants to ease financial conditions but has no room left to cut the policy rate, it reaches for the long end through the balance sheet — and the whole curve sinks. (See the liquidity-channel and QE/QT cross-links below for how this everything-bid works.)

It is worth pausing on *why* a 0.5% 10-year yield is such an extraordinary number. At a 0.5% yield, a 10-year Treasury locks in a nominal return of half a percent per year for a decade — and with inflation expected to run around 2%, that is a guaranteed *real* loss of roughly 1.5% a year. Investors knowingly accepted that loss, which tells you two things. First, in a panic, the safety and liquidity of a U.S. Treasury are worth paying for; the bond is not bought for its return but for its certainty. Second, the Fed's QE was so dominant a buyer that it overwhelmed the price-sensitive private market entirely — the central bank, not the fundamentals, was setting the long yield. That is the most direct demonstration in the whole post of policy *building* the curve: in March 2020 the Fed reached into the long end and pinned it down by sheer balance-sheet force, and the entire structure of interest rates for the U.S. economy was, for a moment, an administered price rather than a market one.

### Volcker's ~16% 10-year in 1981: when policy broke the back of inflation

Go back four decades for the extreme case. By the late 1970s, U.S. inflation had spiraled to nearly 15%, and the bond market had lost faith that the Fed would ever control it. Paul Volcker, appointed Fed chair in 1979, took the policy rate to a peak above **19%** — a level that sounds impossible today — to crush inflation by force. The 10-year Treasury yield peaked at about **15.84% in September 1981**, the highest in the modern history of the U.S. bond market.

That curve was deeply inverted, too — short rates above 19% towered over the 15.84% long bond — because the market expected (correctly) that once inflation broke, Volcker would cut hard. And he did. The 1981 inversion was the last time the 2s10s reached the depths it would revisit in 2023, which is exactly why the 2023 inversion was described as "the deepest since 1981." But the more important legacy is what came *after*. Once Volcker's policy convinced the market that inflation was beaten, the term premium and inflation expectations collapsed, and the 10-year yield began a 40-year decline — from 15.84% in 1981 all the way to 0.5% in 2020. The entire generational bull market in bonds was launched by a single, brutally credible policy.

![Line chart of the US 10-year Treasury yield from 1981 to 2025 showing the decline from the Volcker peak to the 2020 low and the 2022 reset](/imgs/blogs/how-policy-sets-the-bond-market-the-yield-curve-8.png)

The chart above is the long arc: the Volcker peak at 15.84% in 1981, the 40-year decline as credible policy and globalization suppressed inflation and the term premium, the 0.5% COVID low in 2020, and the sharp reset from 2022. The entire downward sweep is the bond market slowly rebuilding the trust that Volcker's policy first earned. Read it as one long policy story: the level of the curve is the accumulated verdict of forty years of central banking.

### The 2025 curve: cuts at the front, tariff term premium at the back

The most recent case is the most subtle, because *two policies pull the curve in opposite directions at once*. Through 2025, the Fed cut the policy rate three times — to a 3.50-3.75% range by December — as inflation cooled enough to allow easing. Lower policy rate, lower front end: the 2-year fell, and the curve re-steepened from its inverted lows back to a positive slope.

But the long end did *not* fall the way a simple cutting cycle would predict. It stayed sticky near 4-4.5%. The reason was the 2025 tariff shock. The "Liberation Day" tariffs of April 2025 — a 10% universal tariff plus higher "reciprocal" rates — pushed the average U.S. effective tariff rate from about 2.4% to the mid-teens, the highest since the 1930s. Tariffs are a tax on imports that raises consumer prices, so they injected fresh *inflation uncertainty* into the long-run outlook. And the large deficits of the period flooded the market with Treasury supply. Both forces lifted the **term premium** — and a higher term premium props up the long end even as the Fed cuts the front.

So the 2025 curve steepened for a *mixed* reason: the front end fell because of Fed cuts (the expectations component dropping), while the long end held up because of tariff-and-supply-driven term premium. This is the most important lesson of the whole post in one snapshot: **the same curve shape can mean different things depending on which end is moving and why.** A steepening driven by front-end cuts is a normal late-cycle re-steepening; a steepening driven by long-end term premium is a warning about inflation and fiscal sustainability. In 2025 it was both at once, and reading the difference was the entire job. (For how tariffs work as a market force, see the tariffs cross-link below.)

### What the four curves have in common

Lay the four cases side by side and a single pattern emerges. In every one, the *front end* did exactly what the policy rate told it to do — pinned at zero in 2020, jacked to 5.5% in 2023, above 19% under Volcker, easing in 2025. The front end is obedient. It is the *long end* that carried all the information, and in each case the long end was telling you something the policy rate alone could not. In 2020 it told you the market believed in years of zero rates and trusted QE to hold the term premium down. In 2023 it told you the market expected the hikes to break the economy and force deep cuts. Under Volcker it told you, eventually, that the inflation fight would be won and rates would fall for forty years. In 2025 it told you that inflation uncertainty and fiscal supply had put a floor under long yields the Fed's cuts could not push through.

That is the deepest takeaway of the whole subject: **the front end is policy, but the long end is the market's verdict on policy.** When the two diverge — when the curve inverts, or steepens for the "wrong" reason — that divergence is the most valuable signal the bond market produces, because it is the gap between what the central bank is doing and what the market believes about where it will end up. A trader who can decompose any curve into "what is the policy rate doing to the front" and "what is the market saying about the long run and the term premium at the back" has the entire macro picture in two numbers. Every case study above is just that decomposition applied to a different year.

## What it means for asset values: the playbook

The curve is not an academic chart; it is the master price that reprices most of the financial world. Here is how to use it.

**Bonds reprice directly off their point on the curve.** A bond's price is set by the yield at its maturity. When the 10-year yield rises a point, every 10-year bond loses about 8.5% (its duration). When the curve steepens because the front end fell, short-dated bonds rally hardest; when it steepens because the long end rose, long bonds get hit. *Match your duration to your view of which end of the curve is moving.* If you think the Fed will cut, the front end falls and short-to-intermediate bonds win. If you fear rising term premium, the long end is the danger zone.

**The 10-year is the price of money for the whole economy.** Mortgage rates, corporate bond yields, and the discount rate on every long-dated cash flow key off the 10-year. When the long end rises on term premium, it tightens conditions for *every* borrower whose rate references it — even if the Fed is cutting. This is why the 2025 episode mattered: the Fed was easing, but the sticky long end kept mortgage and corporate borrowing costs high. (See the transmission-of-rates cross-link for how the 10-year reaches your mortgage.)

**The curve shape is a growth and policy forecast.** A deepening inversion is the market pricing a recession and future cuts — defensive territory: favor duration, quality, and the assets that do well when growth slows. A steepening from the front end is an easing cycle — risk-friendly. A steepening from the long end is an inflation-and-fiscal warning — watch real assets and the term premium. *Read which end is moving before you read the shape.*

**Equities reprice through the discount rate.** A rising long end lifts the discount rate on far-future earnings, hammering long-duration growth stocks hardest — the same mechanism the discount-rate channel post explores in depth. The curve is the input to that channel; the shape and level of the curve set the rate at which every stock's future profits get discounted to today.

**The signals to watch, and what invalidates the read.** Watch the 2s10s spread for the recession warning, the 10-year term premium estimate (the New York Fed publishes one) to separate "expectations" from "premium," and the gap between where the Fed says it is going (the dot plot) and where the curve is priced. The read is invalidated when the curve moves for a *technical* reason — a flight-to-safety bid, a foreign central bank rebalancing, a Treasury supply surprise — rather than a genuine change in the growth-and-policy outlook. Always ask: *is this the expectations component moving, or the term premium?* The answer changes what the curve is telling you.

The single discipline that ties it all together: the yield curve is policy made visible. The Fed sets the anchor at the front; expectations and the term premium build the long end; QE and QT bend it; and the resulting shape is the bond market's continuously-updated forecast of growth and policy. Learn to read which force is moving which end, and the curve stops being a confusing tangle of lines and becomes the clearest signal in all of macro.

## Further reading and cross-links

- [The discount-rate channel: how rates reprice cash flows](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) — how the yields on this curve flow into the discount rate that prices every asset.
- [The expectations channel: forward guidance and credibility](/blog/trading/policy-and-markets/the-expectations-channel-forward-guidance-and-credibility) — how the Fed shapes the *expectations component* of the long end with words, not just hikes.
- [How policy moves credit spreads and the Fed put](/blog/trading/policy-and-markets/how-policy-moves-credit-spreads-and-the-fed-put) — what sits on top of the Treasury curve: the extra yield on risky debt.
- [Yield-curve inversion: the recession signal that mostly works](/blog/trading/fixed-income/yield-curve-inversion-the-recession-signal-that-mostly-works) — the fixed-income deep dive on the inversion's hit rate, lead times, and failures.
- [Why the yield curve usually slopes up: term premium and expectations](/blog/trading/fixed-income/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations) — the full term-premium math we treat briefly here.
- [Duration: the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) and [Modified duration and DV01: measuring and trading rate risk](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk) — the price-sensitivity math behind the worked examples.
- [From the ten-year yield to your mortgage: the transmission of rates](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates) — how the long end reaches real-economy borrowing costs.
- [The yield curve as a growth signal and its asset correlation](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation) — the statistical relationship between the curve and asset returns.
- [QE vs QT: how balance-sheet policy moves markets](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets) and [The central-bank toolkit: rates, QE, QT, forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) — how the balance sheet bends the term premium.
- [Tariffs and trade policy as a market force](/blog/trading/policy-and-markets/tariffs-and-trade-policy-as-a-market-force) — how the 2025 tariff shock fed the term premium at the back of the curve.
