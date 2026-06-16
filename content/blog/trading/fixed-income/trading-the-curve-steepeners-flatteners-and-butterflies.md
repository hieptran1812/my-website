---
title: "Trading the curve: steepeners, flatteners, and butterflies"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into how professionals bet on the shape of the yield curve rather than its level — the three factors of level, slope, and curvature, the four steepener/flattener regimes, and how to build a DV01-neutral curve trade."
tags: ["fixed-income", "bonds", "yield-curve", "steepener", "flattener", "butterfly", "curve-trading", "dv01", "duration", "us-treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — most bond traders do not bet on whether rates go up or down; they bet on how the *shape* of the yield curve changes, and this post shows you how to build those trades from scratch.
> - A whole yield curve moves in just **three independent ways**: a **level** shift (the curve slides up or down in parallel), a **slope** change (the short end moves relative to the long end), and a **curvature** change (the middle, or *belly*, moves relative to the ends, or *wings*).
> - Slope trades come in four flavors, named by *which end moves* and *which way*: **bull steepener**, **bear steepener**, **bull flattener**, and **bear flattener** — and each one is a fingerprint of where the economy sits in the cycle.
> - A **steepener** is a two-leg trade: you buy a short-maturity bond and short a long-maturity bond (a *flattener* is the reverse). You size the legs so their **DV01** — the dollars they each gain or lose per basis point — is equal, which makes the trade flat to a parallel move and sensitive only to the slope.
> - Our running example is a **2s10s steepener** sized at **\$5,000 per basis point** a leg: if the curve steepens 25 basis points, it pays about **\$125,000**, and it nets **\$0** on any parallel move up or down.
> - A **butterfly** trades curvature: you go long the two wings and short twice the belly (or the reverse), betting the middle of the curve cheapens or richens relative to the ends.

There is a question every new bond trader eventually gets asked, and it sounds like a riddle: *"Do you have a view on rates, or a view on the curve?"* The first time you hear it, the two sound like the same thing. Aren't rates the curve? But they are not the same bet at all, and the difference is the whole subject of this post. Having a view on *rates* means you think yields are going up or down — you are betting on the **level** of the curve. Having a view on the *curve* means you think its **shape** is going to change — the gap between short rates and long rates will widen or narrow, or the middle will move relative to the ends — and you want to profit from that shape change *whether or not* the overall level of rates moves at all.

This is one of the most important mental shifts in fixed income, and it is the thing that separates someone who has read about bonds from someone who trades them. The level of rates is famously hard to forecast; the Federal Reserve, with armies of economists, regularly gets it wrong. But the *shape* of the curve follows the business cycle with a rhythm that is far more reliable, and — crucially — you can build a trade that isolates the shape change and cancels out the level. That is what a **curve trade** is: a position engineered so that if rates all move together by the same amount, you make and lose nothing, but if the curve changes shape, you get paid.

![Three small panels showing the three independent ways a yield curve moves: a level panel where the whole curve slides up in parallel, a slope panel where the long end lifts to steepen the curve, and a curvature panel where the belly rises relative to the wings](/imgs/blogs/trading-the-curve-steepeners-flatteners-and-butterflies-1.png)

The diagram above is the mental model for the entire post. A yield curve — the line that plots yield against maturity — looks like it can wiggle in a thousand ways, but in practice almost every move is a blend of just three: a parallel **level** shift, a **slope** change that tilts the short end against the long end, and a **curvature** change that bends the **belly** (the middle maturities) relative to the **wings** (the short and long ends). Master those three factors and you can name any curve move, build a trade for each one, and — most importantly — size the trade so it bets on exactly the factor you care about and nothing else. (Everything here is educational, not investment advice; the goal is to understand the mechanics, not to recommend a trade.)

## Foundations: the words you need before we build a single trade

Let's assemble the vocabulary from zero. If you have read the earlier posts in this series, much of this is a refresher; if not, do not skip it, because every trade later in the post is built out of these pieces.

A **bond** is a tradable loan. You, the buyer, are the lender; the **issuer** — for us, almost always the **US Treasury**, the borrowing arm of the US government — is the borrower. The bond promises a stream of fixed payments: periodic **coupons** (the interest) and the **face value** (also called **par**, conventionally \$1,000 or \$100 per bond) returned at **maturity**, the date the loan ends. A **2-year note** matures in two years; a **10-year note** in ten.

The **yield** of a bond — more precisely its **yield to maturity (YTM)** — is the single interest rate that makes all the bond's future payments, discounted back to today, add up to its current price. It is the return you earn if you buy now and hold to maturity. The first thing to internalize about bonds is the seesaw: **price and yield move in opposite directions.** When yields rise, existing bond prices fall, and vice versa. (We built that relationship from scratch in [the price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds).)

A **basis point** (abbreviated *bp*, pronounced "bip") is **one hundredth of one percent**: 0.01%. Bond yields move in tiny increments, so the whole market quotes in basis points. "The 10-year sold off 8 bps" means its yield rose 0.08%. A move of 100 bps is a full percentage point.

Now the central object of this post.

### What the yield curve actually is

The **yield curve** is a single chart: maturity on the horizontal axis (from a few months out to 30 years), yield on the vertical axis, with a point for each maturity's yield. Connect the dots and you get a curve. At any moment there is one Treasury yield curve, and it is arguably the most-watched line in all of finance, because it tells you the price of borrowing money for *every* length of time at once. We covered how to read it in [the yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance), and why it usually slopes upward in [term premium and expectations](/blog/trading/fixed-income/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations).

A few standard pieces of jargon for parts of the curve:

- The **front end** (or **short end**): the shortest maturities, from 3-month bills out to about the 2-year or 3-year note. This end is anchored to where the market thinks the Fed will set overnight rates.
- The **long end**: the longest maturities, the 10-year, 20-year, and 30-year bonds. This end reflects long-run expectations for growth, inflation, and the extra yield investors demand for locking money up for decades.
- The **belly**: the middle of the curve, roughly the 5-year and 7-year. It is the part that gets "left out" when you compare the front end and the long end — and it is exactly what a *butterfly* trade is about.
- The **wings**: the two ends on either side of the belly. In a butterfly, the belly is the body and the short and long maturities are the wings.

### Slope: the single number that names the curve's shape

Traders compress the whole shape of the curve into a handful of **slope** measures — the yield difference between two maturities:

- The **2s10s** (said "twos-tens") is the most quoted: the **10-year yield minus the 2-year yield**. If the 10-year yields 4.50% and the 2-year yields 4.00%, the 2s10s is +50 bps, and the curve is *upward-sloping* (or **steep**) over that range.
- The **3m10y** ("three-month–ten-year") is the **10-year yield minus the 3-month bill yield** — the Fed's favorite recession gauge.
- A **flat** curve has a slope near zero; an **inverted** curve has a *negative* slope, where short yields are *higher* than long yields — an unusual and ominous state we covered in [yield curve inversion](/blog/trading/fixed-income/yield-curve-inversion-the-recession-signal-that-mostly-works).

When a trader says "I'm long the 2s10s steepener," they are betting the 2s10s number is going to get *bigger* (more positive). That is a bet on *slope*, not level. The whole game of this post is turning that one-line view into a real, sized, two-bond position.

### DV01: the dollars-per-basis-point that makes curve trades possible

The last building block — and the one that does all the heavy lifting — is **DV01**, the *dollar value of an "01"* (a one-basis-point move). It is the **number of dollars you gain or lose if your bond's yield moves one basis point.** We built DV01 from scratch in [modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk), and it is the single most important tool here, so let's restate the intuition.

A long-maturity bond is far more sensitive to yield changes than a short one, because its payments are locked in for longer. A 10-year note might lose roughly **\$850** in value for every \$1,000,000 of face value when its yield rises one basis point — its DV01 is about \$850 per million. A 2-year note, with much less time on the clock, might have a DV01 of only about **\$190 per million**. *Same one-basis-point move, wildly different dollar impact.* That difference is the reason you cannot build a curve trade by simply buying \$1,000,000 of one bond and shorting \$1,000,000 of another — the legs would not be balanced, and you would accidentally be betting on the level. The whole craft of curve trading is sizing the two legs so their DV01s match. Hold that thought; it is the key that unlocks everything below.

#### Worked example: reading the shape from four numbers

Before we trade anything, let's make sure you can read a curve's shape off raw yields. Suppose the Treasury curve today is: 3-month bill 5.30%, 2-year 4.80%, 5-year 4.40%, 10-year 4.30%. Compute the three things a curve trader looks at first.

- **Slope (2s10s):** 10-year minus 2-year = 4.30% − 4.80% = **−50 bps.** Negative — the curve is *inverted* between 2 and 10 years.
- **Slope (3m10y):** 10-year minus 3-month = 4.30% − 5.30% = **−100 bps.** Even more inverted at the very front — a deep recession-warning signal.
- **Curvature (2-5-10 fly spread):** `2 × (5y) − (2y) − (10y)` = 2 × 4.40 − 4.80 − 4.30 = 8.80 − 9.10 = **−30 bps.** A negative fly spread means the belly is *rich* (its yield is below where a straight line between the wings would put it) — the curve sags in the middle.

So in one glance: the curve is inverted (short yields above long), the inversion is deepest at the very front (the Fed is holding rates high), and the belly is rich relative to the wings. *Three small subtractions turn a list of yields into a complete description of the curve's shape — level set aside, what remains is slope and curvature, and those are what you trade.*

## The three factors: level, slope, curvature

Here is the deep idea that organizes all curve trading. Decades of data show that essentially every movement of the entire yield curve — all the daily wiggles across every maturity — can be explained by just **three independent factors**. Statisticians find them with a technique called *principal component analysis*, but you do not need the math to grasp them, because they have plain-English names:

1. **Level (the first factor).** The whole curve slides up or down roughly in parallel. Every maturity's yield rises by about the same amount, or falls by about the same amount. This is by far the biggest factor — it explains the majority of all curve variance. When "rates go up," this is mostly what is happening.
2. **Slope (the second factor).** The short end and the long end move in *opposite* directions, or by *different* amounts. The curve tilts. The 2s10s spread widens (steepens) or narrows (flattens). This is the second-biggest factor.
3. **Curvature (the third factor).** The belly moves relative to the wings. The middle of the curve sags or bulges while the two ends stay put. This is the smallest of the three, but it is real, tradeable, and the basis of the butterfly.

The reason this decomposition matters so much for trading is that **the three factors are (approximately) independent.** A pure slope change can happen with no change in level; a pure curvature change can happen with no change in level or slope. So if you can build a position that responds to *only one* factor — say, slope — and is *blind* to the other two, you have isolated a clean bet. That is precisely what a DV01-neutral curve trade does. Level is the noisy, hard-to-forecast factor; by neutralizing it, you get to express a view on the more predictable slope and curvature factors without the level drowning out your signal.

#### Worked example: decomposing a real curve move

Suppose on Monday the curve is: 2-year at 4.00%, 5-year at 4.20%, 10-year at 4.50%. By Friday it is: 2-year at 4.20%, 5-year at 4.30%, 10-year at 4.50%.

Let's decompose the move factor by factor. The 2-year rose +20 bps, the 5-year rose +10 bps, the 10-year was unchanged. Is this level, slope, or curvature?

- **Level:** if it were a pure level shift, all three would move by the same amount. They did not — so there is little or no level component here.
- **Slope:** the front end (2-year, +20 bps) rose while the long end (10-year, 0 bps) held. The 2s10s spread went from 4.50% − 4.00% = +50 bps to 4.50% − 4.20% = +30 bps. The curve **flattened by 20 bps**. That is a big slope move.
- **Curvature:** did the belly move differently from the average of the wings? The wings (2-year and 10-year) moved on average (+20 + 0) / 2 = +10 bps. The belly (5-year) also moved +10 bps. So the belly moved exactly in line with the wings — **no curvature change.**

The clean read: this week's move was *mostly slope* — a bear flattener (front end up, long end flat) — with no curvature. *Once you can name a curve move by its three factors, you can pick the trade that profits from exactly the factor you predicted.*

## The four slope regimes: bull and bear, steepener and flattener

A slope change has a direction (steepening or flattening) and a *cause* (which end is doing the moving). Combining the two gives **four named regimes**, and learning them is like learning the four seasons of the bond market — each one tells you something specific about the economy.

The naming convention trips up beginners, so let's be precise. "Bull" and "bear" refer to the *bond market*, and remember the seesaw: **bond bull = yields falling = prices rising**; **bond bear = yields rising = prices falling**. "Steepener" and "flattener" refer to whether the 2s10s slope is *widening* or *narrowing*. Cross those two and you get:

| Regime | What's happening | Driven by | Typical cycle phase |
|---|---|---|---|
| **Bull steepener** | Yields falling, slope widening | The **front end falls fastest** (Fed easing) | Early recovery; Fed cutting into a slowdown |
| **Bear steepener** | Yields rising, slope widening | The **long end rises fastest** | Reflation; growth/inflation or heavy bond supply at the long end |
| **Bull flattener** | Yields falling, slope narrowing | The **long end falls fastest** | Late cycle; flight to safety; a growth scare |
| **Bear flattener** | Yields rising, slope narrowing | The **front end rises fastest** (Fed hiking) | Late expansion; the Fed tightening |

![A two by two matrix of the four slope regimes — bull steepener, bear steepener, bull flattener, bear flattener — each showing which end of the curve moves, the cycle phase it belongs to, and what it signals about the economy](/imgs/blogs/trading-the-curve-steepeners-flatteners-and-butterflies-2.png)

The figure above lays out all four. The trick to remembering them is to ask two questions in order: *(1) Are yields generally rising or falling?* That gives you bear or bull. *(2) Is the gap between long and short widening or narrowing?* That gives you steepener or flattener. The deeper insight is **which end is driving the move**, because that is what connects the curve to the economy:

- A **bull steepener** is driven by the *front end collapsing*. Short yields are pinned to expected Fed policy, so when the front end drops fast, the market is screaming "rate cuts are coming." This is the classic early-recovery shape: the Fed is easing aggressively, short yields fall, long yields fall less (because long-run growth and inflation are starting to look better), and the curve gets dramatically steeper.
- A **bear flattener** is the mirror image, driven by the *front end rising*. The Fed is hiking, short yields climb toward the new policy rate, while long yields rise less (the market believes the hikes will eventually slow the economy and inflation). The curve flattens. This is the signature of a late-cycle tightening campaign — and if the Fed keeps hiking, a bear flattener eventually pushes the curve into *inversion*.
- A **bear steepener** is driven by the *long end* selling off. Long yields jump — maybe on a hot inflation print, a growth surprise, or a wave of new bond issuance the market has to absorb — while the front end, anchored by the Fed, barely moves. This is a "term premium" or "supply" story, and it can be violent.
- A **bull flattener** is driven by the *long end rallying* (long yields falling fast) while the front end holds. This is the flight-to-safety shape: investors pile into long bonds for protection during a growth scare, dragging long yields down toward the front end.

#### Worked example: classifying a week in the bond market

The 2-year goes from 4.80% to 4.55% (down 25 bps). The 10-year goes from 4.20% to 4.10% (down 10 bps). What regime is this?

Step 1 — bull or bear? Both yields *fell*, so prices rose: this is a **bull** move. Step 2 — steepener or flattener? The 2s10s slope went from 4.20% − 4.80% = −60 bps (inverted) to 4.10% − 4.55% = −45 bps. The slope rose from −60 to −45 — it *widened* (became less negative). So the curve **steepened**. Step 3 — which end drove it? The 2-year fell 25 bps versus the 10-year's 10 bps, so the *front end drove it*.

Verdict: a **bull steepener**, led by the front end. In plain English, the market is now pricing in faster or deeper Fed rate cuts — the front end is racing lower in anticipation. *A bull steepener led by a collapsing front end is the bond market's loudest signal that easing is on the way.*

## Building a steepener: the two-leg trade

Now we make it real. A **steepener** is a bet that the 2s10s slope will get *bigger* (more positive, or less negative). Mechanically, you want to profit if the 10-year yield rises relative to the 2-year, or the 2-year falls relative to the 10-year — i.e., if the gap widens whichever way it happens.

There are two legs:

1. **Long the 2-year** (the front leg). You *buy* the 2-year note. Remember the seesaw: a long bond position gains when its yield *falls*. So this leg profits if the 2-year yield drops.
2. **Short the 10-year** (the back leg). You *short* the 10-year note — borrow it and sell it, planning to buy it back later. A short position gains when the yield *rises* (price falls). So this leg profits if the 10-year yield climbs.

Put them together: you make money if the 2-year falls *or* the 10-year rises — both of which widen the 2s10s. You also make money if both move but the 10-year rises more than the 2-year, and so on. The position's P&L tracks the *change in the slope*, not the change in either yield alone.

A **flattener** is just the reverse: short the 2-year, long the 10-year, betting the slope *narrows*.

![A before and after diagram of a 2s10s steepener: at entry the long 2-year leg and short 10-year leg are DV01-matched at five thousand dollars per basis point each and net to zero P&L; after a 25 basis point steepening the long leg makes fifty thousand dollars, the short leg makes seventy-five thousand, for a total of one hundred twenty-five thousand](/imgs/blogs/trading-the-curve-steepeners-flatteners-and-butterflies-3.png)

The figure above is our running steepener, sized so each leg has a DV01 of **\$5,000 per basis point** (we will derive that sizing in a moment). At entry, the legs cancel: the long 2-year loses on a rate rise, the short 10-year gains on a rate rise, and if rates all move together the two cancel exactly — the position is *level-neutral*. But it is alive to slope. Watch what happens when the curve steepens.

#### Worked example: the steepener that pays \$125,000 on a 25bp steepening

We put on the steepener with each leg at **\$5,000 per basis point** of DV01. The curve then steepens by 25 bps, and let's say it happens this way: the 2-year yield falls 10 bps and the 10-year yield rises 15 bps. (The slope went from, say, +50 to +75 bps — a 25 bp steepening.)

Compute each leg's P&L using the rule *P&L = DV01 × (number of basis points the yield moved), with the sign set by whether the position gains or loses on that move:*

- **Long 2-year leg:** the 2-year yield *fell* 10 bps. A long position gains when yields fall, so this leg makes a profit: \$5,000/bp × 10 bp = **+\$50,000**.
- **Short 10-year leg:** the 10-year yield *rose* 15 bps. A short position gains when yields rise (the bond's price fell, and you sold it high), so: \$5,000/bp × 15 bp = **+\$75,000**.
- **Total:** \$50,000 + \$75,000 = **+\$125,000.**

Notice the elegant result: the total equals \$5,000/bp × 25 bp = \$125,000 — the **leg DV01 times the total change in slope.** It did not matter that the 2-year fell while the 10-year rose; what mattered was that the *gap* between them widened by 25 bps. *A DV01-matched steepener pays its leg DV01 times the change in slope, no matter how that change splits between the two ends.*

This is the whole point of the trade. You did not need to predict whether yields would go up or down. You only needed to be right that the curve would steepen.

## Making it DV01-neutral: sizing so you bet on shape, not level

Here is the step that beginners get wrong and professionals obsess over. If you simply bought \$1,000,000 of the 2-year and shorted \$1,000,000 of the 10-year, your trade would **not** be a clean slope bet — it would be dominated by the level. Why? Because the 10-year's DV01 (≈ \$850 per million) dwarfs the 2-year's (≈ \$190 per million). On a parallel move where both yields rise, say, 10 bps, the short 10-year leg would gain \$8,500 while the long 2-year leg would lose only \$1,900 — a net \$6,600 gain that has *nothing to do with the slope* and everything to do with the level falling. You would think you had a curve trade, but you would really have a short-duration bet in disguise.

The fix is to size the legs so their **DV01s are equal.** Then, on any parallel move, one leg's gain exactly offsets the other's loss, and the only thing left is the slope. To make each leg \$5,000 per basis point:

- **2-year leg:** \$5,000/bp ÷ \$190/bp per \$1M = about **\$26.3 million** of face value. (You need a lot of the low-DV01 2-year to build up to \$5,000/bp.)
- **10-year leg:** \$5,000/bp ÷ \$850/bp per \$1M = about **\$5.9 million** of face value. (The high-DV01 10-year gets you there with far less face.)

![A three by three sizing table for a DV01-neutral steepener showing that the 2-year leg has low DV01 per million and needs twenty six point three million of face, while the 10-year leg has high DV01 per million and needs only five point nine million, so the two legs cancel to zero per basis point on a parallel move](/imgs/blogs/trading-the-curve-steepeners-flatteners-and-butterflies-6.png)

The table above is the sizing in full. The striking feature is the **face-value mismatch**: you trade about \$26.3M of the 2-year against only \$5.9M of the 10-year — a ratio of roughly 4.5 to 1 by face. To a newcomer that looks lopsided, but it is exactly right, because *the trade balances dollars of risk, not dollars of face.* The 2-year is so insensitive that you need a big slug of it to match the punch of a small slug of the 10-year. This is the single most important sizing principle in all of curve trading: **match DV01, never face value.**

The number you divide one DV01 by the other to get is called the **hedge ratio** — here \$850 ÷ \$190 ≈ 4.47 — and it tells you how much *more* of the front leg you need per unit of the back leg. (This is the same DV01-matching logic we used to *hedge* a single position in the [modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk) post; a curve trade is just a hedge you *want* to leave imperfect along the slope dimension.)

#### Worked example: proving the trade is level-neutral

Let's verify the sizing did its job by hitting it with a pure parallel move. Suppose *both* the 2-year and the 10-year yields rise exactly 30 bps (a clean level shift, no slope change).

- **Long 2-year leg:** yield rose 30 bps, and a long position *loses* when yields rise: \$5,000/bp × 30 bp = **−\$150,000.**
- **Short 10-year leg:** yield rose 30 bps, and a short position *gains* when yields rise: \$5,000/bp × 30 bp = **+\$150,000.**
- **Total:** −\$150,000 + \$150,000 = **\$0.**

Exactly zero. The parallel move washed out completely. Now do the same with a parallel *fall* of 30 bps and you get +\$150,000 on the long leg and −\$150,000 on the short leg — again \$0. *DV01-matching is what makes the trade blind to the level: the legs are equal and opposite on any parallel move, so only the slope can move your P&L.*

This is why, on a desk, the first question about any curve trade is "what's your DV01 per leg?" — never "how much face did you buy?" Face value is almost meaningless; DV01 is the position.

### The full P&L map: every scenario in one table

The cleanest way to *see* what a DV01-neutral steepener does is to subject it to a battery of scenarios at once. The table below takes our \$5,000-per-leg steepener and runs it through six different curve moves — two pure parallel shifts, two slope moves, and two "single-end" moves — and shows the P&L of each leg and the total.

![A six row P&L table for the DV01-neutral steepener showing that both parallel scenarios net to zero, the steepening scenarios pay one hundred twenty-five thousand dollars, the flattening scenarios lose one hundred twenty-five thousand, and the single-end moves pay or lose depending on whether the front end falls or rises](/imgs/blogs/trading-the-curve-steepeners-flatteners-and-butterflies-7.png)

Read the rightmost column top to bottom and the design of the trade jumps out. The two **parallel** scenarios (everything up 25 bps, everything down 25 bps) both net to exactly **\$0** — the legs cancel, the level is invisible. Every scenario where the curve **steepens** pays **+\$125,000**, and every scenario where it **flattens** loses **−\$125,000**, *regardless of how the move splits between the two ends.* The "steepen 25bp" row (2-year −10, 10-year +15) and the "bull steepen" row (2-year −25, 10-year flat) both pay the same \$125,000, because in each case the slope widened by the same 25 bps. The trade has been engineered to care about one number — the change in slope — and nothing else.

This table is also the best way to internalize the asymmetry of curve risk. A steepener is a *directional bet on the slope*: it has unlimited upside if the curve keeps steepening and symmetric downside if it flattens. It is "hedged" only against the level; along the slope dimension it is fully exposed, by design. That is the trade-off at the heart of every curve position — you give up the level to get a clean shot at the shape, but the shape can move against you just as easily as for you.

#### Worked example: reading the trade's P&L off the slope alone

Here is a shortcut the table makes obvious. Once a steepener is DV01-matched, you do not need to track the two legs separately at all — you can compute the whole P&L from the *change in the slope* and the *leg DV01*:

$$\text{P\&L} \approx \text{(leg DV01)} \times \Delta(\text{slope in bps})$$

Suppose over a month the 2s10s slope moves from +40 bps to +65 bps — a +25 bp steepening — and your leg DV01 is \$5,000/bp. Then P&L ≈ \$5,000/bp × 25 bp = **+\$125,000**, full stop. You did not need the individual moves of the 2-year and 10-year; the slope change carried all the information. If instead the slope had fallen from +40 to +30 (a 10 bp flattening), the same formula gives \$5,000/bp × (−10 bp) = **−\$50,000.** *A DV01-matched curve trade collapses two yields into one number — the slope — so its entire P&L is just the leg DV01 times how far that one number moved.*

## The influence thread: why the slope tracks the cycle and the Fed

Why bother forecasting the slope at all? Because — unlike the level — **the slope of the curve follows the business cycle with a rhythm you can actually read.** This is the deep "why it matters" of curve trading, and it ties straight back to the spine of this whole series: bonds are the price of money, and the curve's shape is a real-time readout of where the economy is and what the Fed is doing about it.

The mechanism runs through the two ends:

- **The front end is the Fed's leash.** The 2-year yield is, roughly, the market's average forecast of where the Fed funds rate will be over the next two years. When the Fed is *hiking*, the front end gets dragged up. When the market expects *cuts*, the front end falls fast.
- **The long end is the economy's verdict.** The 10-year yield reflects long-run growth, long-run inflation, and the *term premium* — the extra yield investors demand for the risk of locking up money for a decade. It moves on the big-picture outlook, not the next FOMC meeting.

![A time series chart showing the 2s10s slope falling from steep early in the cycle toward inversion late in the cycle as the Fed funds rate rises, then re-steepening once the Fed begins cutting, with an inversion zone marked below the zero line](/imgs/blogs/trading-the-curve-steepeners-flatteners-and-butterflies-4.png)

The figure above is the master pattern of the post. Trace one cycle:

1. **Early cycle (steep curve).** The Fed has cut rates to the floor to fight the last recession. The front end is pinned low. The long end starts to rise as growth recovers and inflation expectations firm up. Result: the curve is **steep** — a big positive 2s10s.
2. **Mid cycle (flattening).** Growth is solid, inflation is rising, and the Fed begins *hiking*. Each hike drags the front end up; the long end rises more slowly because the market trusts the Fed to control inflation eventually. The curve **flattens** — a bear flattener, the dominant regime of a tightening campaign.
3. **Late cycle (inversion).** The Fed hikes aggressively to cool an overheating economy. The front end pushes *above* the long end. The 2s10s goes **negative** — the curve **inverts**. Historically this has been the bond market's most reliable recession warning, because it says short-term money is more expensive than long-term money, a sign the Fed has tightened "too far."
4. **Turn (re-steepening).** A slowdown or recession arrives, the Fed pivots to *cutting*, and the front end collapses. The curve **re-steepens** violently — often a bull steepener as the market races to price in cuts. And the cycle begins again.

This pattern is why curve traders spend so much energy on Fed-watching: the front end is essentially a bet on the next several FOMC decisions, so a curve trade is, at bottom, a structured bet on monetary policy. The [macro-trading series covers the policy side in depth](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession), and [how monetary policy moves bonds](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity) connects the dots from the Fed to your P&L. The point for us is that the slope is *forecastable from the cycle*, which is exactly what makes a level-neutral slope trade worth putting on.

#### Worked example: reading the cycle to pick the trade

It is late in an expansion. The Fed has hiked seven times and signals more to come; inflation is still hot. The 2s10s is +20 bps and shrinking. Which curve trade fits this read, and why?

The cycle script says: continued Fed hikes → front end keeps rising → bear flattener → eventual inversion. So the natural trade is a **flattener**: short the 2-year, long the 10-year, betting the 2s10s falls (toward zero and below). If you size it at \$5,000/bp per leg and the 2s10s falls from +20 to −30 bps (a 50 bp flattening), the trade pays \$5,000/bp × 50 bp = **+\$250,000** — and it does so even though *all* yields might be rising, because the flattener is level-neutral. *In a late-cycle tightening, the cycle itself hands you the flattener; you are just monetizing the Fed's own script.*

(The opposite read — a Fed about to pivot to cuts — argues for a *steepener*, anticipating the re-steepening at step 4. The hardest and most lucrative call in curve trading is catching that turn from flattener to steepener.)

## Butterflies: trading the curvature, not the slope

So far we have traded the level (and neutralized it) and the slope. The third factor — **curvature** — has its own trade, and it is the most elegant of the three: the **butterfly**.

A butterfly is a *three-leg* trade involving the belly and the two wings. The classic structure for betting the belly will **cheapen** (its yield will rise relative to the wings):

- **Long the short wing** (e.g., the 2-year): +1 unit of DV01.
- **Short the belly** (e.g., the 5-year): −2 units of DV01 — *twice* the wing DV01.
- **Long the long wing** (e.g., the 10-year): +1 unit of DV01.

The "2-5-10 fly" is the canonical example. Because you are short *two* units of belly DV01 and long *one* unit of each wing, the trade is **slope-neutral and level-neutral** — it cancels out both the first and second factors — and responds only to *curvature*: whether the belly moves relative to the average of the wings. The reverse butterfly (short the wings, long the belly) bets the belly will **richen** (its yield falls relative to the wings).

![A yield curve showing a 2-5-10 butterfly trade: a smooth fair curve and a dashed curve where the 5-year belly has cheapened so its yield is too high, with three trade legs below the chart — long the 2-year wing with plus one unit of DV01, short the 5-year belly with minus two units of DV01, and long the 10-year wing with plus one unit of DV01](/imgs/blogs/trading-the-curve-steepeners-flatteners-and-butterflies-5.png)

The figure above shows the idea. The solid line is the "fair" curve, smoothly connecting the maturities. The dashed line is what happens if the belly cheapens: the 5-year yield rises above where a smooth curve would put it, creating a little bump. A butterfly that is short the belly profits from exactly that bump. The metric traders watch is the **butterfly spread**: twice the belly yield minus the sum of the wing yields, or `2 × (5y yield) − (2y yield) − (10y yield)`. When that number rises, the belly is cheapening relative to the wings, and the short-belly fly pays.

Why would the belly move relative to the wings? Often it is *relative value* — the belly gets temporarily mispriced because of supply (a heavy 5-year Treasury auction) or because investors crowd into a particular maturity. Butterflies are the bread and butter of relative-value desks precisely because curvature mean-reverts more reliably than level or slope: a belly that gets too cheap tends to richen back, and vice versa.

#### Worked example: a 2-5-10 butterfly that pays on a cheapening belly

You put on a short-belly 2-5-10 fly. Each *wing* leg has a DV01 of \$2,500/bp (long), and the *belly* leg has a DV01 of \$5,000/bp (short) — so DV01 is balanced: +2,500 + 2,500 wings versus −5,000 belly. Now suppose:

- The 2-year yield rises 5 bps, the 10-year yield rises 5 bps (the wings move together — a small level shift), and the 5-year belly yield rises 15 bps (the belly cheapens hard).

Leg by leg:

- **Long 2-year wing:** yield rose 5 bps, long position loses: \$2,500/bp × 5 = **−\$12,500.**
- **Long 10-year wing:** yield rose 5 bps, long position loses: \$2,500/bp × 5 = **−\$12,500.**
- **Short 5-year belly:** yield rose 15 bps, short position gains: \$5,000/bp × 15 = **+\$75,000.**
- **Total:** −\$12,500 − \$12,500 + \$75,000 = **+\$50,000.**

Now check it is curvature, not level: the butterfly spread went up by `2 × 15 − 5 − 5 = +20 bps`, and the belly DV01 (\$5,000/bp) times *half* that spread change captures the move. The wings' 5 bp parallel rise washed out (you lost on both wings but the belly's offsetting position absorbed it). *A butterfly isolates curvature: it pays when the belly moves relative to the wings and ignores a parallel shift in the wings themselves.*

### Why curvature mean-reverts, and what makes a fly "rich" or "cheap"

The reason butterflies are a perennial favorite of relative-value desks is that curvature is the most *mean-reverting* of the three factors. The level wanders for years; the slope follows the multi-year business cycle; but the belly's deviation from a smooth curve tends to snap back over days or weeks. The intuition is supply-and-demand: when a fresh 5-year Treasury auction floods the belly with new bonds, the belly cheapens (its yield ticks up) until enough buyers step in to absorb the supply, and then it richens back toward fair. Nothing fundamental has changed about the economy — it is a temporary digestion problem — so the spread reverts.

A trader judges whether a fly is "rich" or "cheap" by comparing the current butterfly spread to its own recent history. If `2 × (5y) − (2y) − (10y)` is sitting two standard deviations above its three-month average, the belly is unusually *cheap*, and the relative-value play is to *buy* the belly (a long-belly fly), betting it richens back to the mean. If the spread is unusually low, the belly is *rich* and you fade it with a short-belly fly. This is statistical, not directional: the trader is not forecasting the economy at all, only betting that an unusual dislocation reverts to normal. The risk, as always, is that "unusual" becomes "the new normal" — a structural shift (say, a change in Treasury issuance patterns) that re-rates the belly permanently and turns a mean-reversion bet into a slow bleed.

### Choosing the trade from your view

Pulling the three factors together, the decision of *which* trade to put on falls out of *which factor* you have a view on. The table below maps a view to its trade.

| Your view | The trade | Legs | Neutral to | Bets on |
|---|---|---|---|---|
| "The curve will steepen" | **Steepener** | Long 2y, short 10y (DV01-matched) | Level | Slope ↑ |
| "The curve will flatten" | **Flattener** | Short 2y, long 10y (DV01-matched) | Level | Slope ↓ |
| "The belly is too cheap" | **Long-belly butterfly** | Short wings, long 2× belly | Level + slope | Curvature ↓ |
| "The belly is too rich" | **Short-belly butterfly** | Long wings, short 2× belly | Level + slope | Curvature ↑ |
| "Rates will fall, all of them" | (Not a curve trade) | Just go long duration | Nothing | Level ↓ |

The bottom row is the contrast that makes the whole idea click: if your only view is on the *level* — rates are going down — you do not need a curve trade at all; you just buy bonds and take the duration. The curve trades exist precisely for the cases where you have a view on *shape* and want to strip the level out so it cannot drown your signal. Choosing the trade is therefore a two-step question: *which factor am I confident about, and which factors do I want to neutralize so they don't get in the way?*

## How the legs actually get traded: bonds, futures, and swaps

In the worked examples we bought and shorted the cash bonds directly, which is the clearest way to see the mechanics. In practice, professionals usually express curve trades through instruments that make the legs cheaper and easier to manage:

- **Treasury futures.** Each Treasury futures contract (the 2-year, 5-year, 10-year, and "ultra" contracts) has a known DV01, so you can build a DV01-neutral steepener by trading a calculated *number of contracts* in each leg rather than millions in cash bonds. Futures require only **margin** (a small good-faith deposit) instead of paying full price, so the trade is highly **capital-efficient** — and that efficiency is also where the leverage and the risk live.
- **Interest-rate swaps.** A swap lets you take pure exposure to a maturity's rate without owning a bond at all, and swap curves trade alongside the Treasury curve. Curve trades in swap form are enormous in size.
- **The repo market.** To *short* a cash Treasury you must borrow it, which you do in the **repo** (repurchase) market — you put up cash, receive the bond, sell it, and pay a small borrowing fee. Some bonds, especially the newest "on-the-run" issues, can go *special* (expensive to borrow), which raises the cost of the short leg. This funding cost is the hidden friction in any cash curve trade.

The instrument choice does not change the logic one bit — you still match DV01 and bet on the factor — but it changes the *cost*, the *leverage*, and the *carry* (whether you earn or pay to hold the position over time). The [quantitative-finance series goes deep on the analytics](/blog/trading/quantitative-finance/fixed-income-analytics) behind sizing these instruments precisely.

#### Worked example: carry and roll-down on a steepener

A curve trade does not just live or die on whether the slope moves — it also earns or pays **carry** (the net interest you collect or owe just for holding the position) and **roll-down** (the gain from a bond "aging" down a sloped curve). Suppose your steepener is long the 2-year (yielding 4.0%) and short the 10-year (yielding 4.5%), financed in repo.

The short 10-year leg means you are *paying* the 10-year's higher yield (you owe the coupons to whoever you borrowed the bond from), and the long 2-year leg *earns* the lower 2-year yield. So on an upward-sloping curve, a steepener has **negative carry** — you bleed a little every day you hold it, because you are short the high-yielding long bond and long the low-yielding short bond. If the slope is +50 bps and the trade is sized at \$5,000/bp, the rough annual carry drag is on the order of 50 bps of yield differential applied across the position — meaningful enough that the steepening has to happen *fast enough* to beat the bleed.

*A curve trade's edge is the slope move minus the carry; a steepener on a steep curve must steepen enough, soon enough, to overcome its negative carry.* This is why timing matters so much: a "right" curve view that takes two years to play out can still lose money to carry.

### Where curve trades actually go wrong

A DV01-neutral curve trade looks safe — the level is hedged, the legs offset — but that safety is narrow and conditional. Four things turn a tidy curve position into a real loss, and a desk thinks about all of them before sizing:

- **The slope goes the wrong way.** This is the obvious one and the whole point: a steepener that flattens loses leg-DV01 times the flattening. There is no level hedge against an adverse *slope* move, because the slope is exactly what you chose to bet on. The "hedged" feeling is an illusion if it makes you forget you are fully exposed along the one dimension that matters.
- **The hedge ratio drifts.** DV01s are not constant. As yields change, each bond's DV01 changes too (this is *convexity*, the subject of [a whole earlier post](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story)), and the longer-maturity leg's DV01 shifts more. A steepener that was perfectly DV01-matched at inception can drift to slightly *un*-matched after a big rate move, quietly re-introducing a small level bet you did not intend. Desks *re-hedge* periodically to keep the legs balanced.
- **Carry bleeds you while you wait.** As the worked example showed, an upward-sloping curve makes a steepener cost money to hold. A view that is right but slow can still be a losing trade once the daily carry is netted against the eventual slope move.
- **The funding leg gets expensive or vanishes.** To short the long bond you must keep borrowing it in repo. If that bond goes "special" (scarce and costly to borrow), your funding cost jumps; in a stress event, the ability to borrow it at all can dry up, forcing you to unwind at the worst possible moment.

The deepest danger is leverage interacting with crowding. Because curve trades carry little DV01 per dollar of capital, they are almost always run with leverage — and the same well-known relative-value trades attract many of the same players. When a shock forces a wave of them to unwind at once, the "mean-reverting" spread can gap *against* everyone simultaneously, and a position that looked hedged becomes a fire sale. That dynamic, not a wrong forecast, is what destroyed Long-Term Capital Management in 1998 and has scarred curve desks ever since. *Level-neutral is not loss-neutral: a curve trade is a focused bet, and the focus cuts both ways.*

## Common misconceptions

**"A curve trade is just a bet that rates will go up (or down)."** No — that is a *level* bet, and a properly sized curve trade is built specifically to be *neutral* to the level. The whole craft is DV01-matching the legs so a parallel move in rates nets to zero. A steepener can make money in a rising-rate world, a falling-rate world, or a flat one — all that matters is the slope.

**"To balance the trade, match the face value of the two legs."** This is the most common and most expensive beginner error. You match **DV01**, not face value. Because a 10-year has roughly 4–5 times the DV01 of a 2-year, a DV01-neutral steepener holds *far more* 2-year face than 10-year face (in our example, \$26.3M versus \$5.9M). Matching face value would leave you accidentally short duration — a level bet wearing a curve costume.

**"A steepener always means yields are falling."** No. *Bull* steepener means yields falling (front end leads); *bear* steepener means yields rising (long end leads). "Steepener" only tells you the slope is widening, not which direction yields are going. You have to specify bull or bear to know the full story.

**"An inverted curve guarantees a recession, so just trade off the inversion."** Inversion has preceded most US recessions, but the lead time is long and variable (sometimes a year or more), the curve can stay inverted for a long time, and there have been false signals. As a *trade*, "the curve will re-steepen eventually" can be right in direction and still ruinous on timing and carry. We unpacked the signal's real track record in [yield curve inversion](/blog/trading/fixed-income/yield-curve-inversion-the-recession-signal-that-mostly-works).

**"Curve trades are low-risk because they're hedged."** They are *level-hedged*, not *risk-free*. A steepener that flattens loses money just as cleanly as it would have made it. And because curve trades are usually run with leverage (small DV01 per dollar of capital, expressed through futures or repo), a modest adverse slope move can produce a large loss relative to the capital posted. "Relative value" is not "no risk."

**"The belly and the wings move together, so curvature isn't worth trading."** Most of the time the belly does track the wings — but not always, and the deviations are tradeable precisely because they tend to mean-revert. Supply (a heavy auction in one maturity), positioning, and Fed expectations all push the belly around relative to the ends. Butterflies exist because that third factor, small as it is, is real and recurrent.

## How it shows up in real markets

**The 2004–2006 bear flattener (the "conundrum").** As the Fed under Alan Greenspan raised the funds rate from 1% to 5.25% across 2004–2006, the front end of the curve marched up with it — a textbook bear flattener. The puzzle, which Greenspan famously called a "conundrum," was that the long end *barely moved*: 10-year yields stayed near 4–4.5% even as short rates more than quadrupled. Heavy foreign demand for Treasuries (especially from Asian central banks) was pinning the long end down. The curve flattened and then inverted in 2006 — and a recession followed in 2007–2008. A flattener trade put on early in that hiking cycle would have paid handsomely; the conundrum was, in trading terms, the long end refusing to rise while the front end did all the work.

**The 2019 inversion and 2020 re-steepening.** In 2019, after a 2015–2018 hiking cycle, the 2s10s briefly went *negative* — the curve inverted — and the financial press lit up with recession talk. Then COVID hit in early 2020, the Fed slashed rates to zero, and the front end collapsed: the curve re-steepened violently in a classic bull steepener as the market priced in years of zero rates. A trader who held a steepener into the 2020 turn — betting on exactly that front-end collapse — would have caught one of the cleanest curve moves of the era. It is a textbook illustration of step 4 of the cycle script: the pivot from inversion to re-steepening.

**The 2022 bear flattener and deep inversion.** As the Fed hiked from near-zero to over 5% in 2022–2023 to fight the worst inflation in forty years, the curve delivered an enormous bear flattener: the 2-year yield rocketed from under 1% to over 4% while the 10-year rose far less. The 2s10s inverted to around **−1 percentage point** by mid-2023 — the deepest inversion since the early 1980s. Flattener trades were the dominant winning curve position through 2022. This episode (and the bond rout that accompanied it) sits behind the [2022 case where stocks and bonds both fell](/blog/trading/cross-asset/case-study-2022-stocks-and-bonds-both-fell).

**The 2023–2024 "bull steepener" debate.** With the curve deeply inverted, a popular trade became the steepener — betting that when the Fed finally started cutting, the front end would drop and the curve would normalize (re-steepen) from its extreme inversion. The trade was correct in *thesis* but punishing in *timing and carry*: the curve stayed inverted far longer than many expected, and the negative carry on a steepener bled traders who put it on too early. It is the living example of the misconception above — right direction, wrong (and very expensive) timing.

**The 1994 bond massacre.** When the Fed unexpectedly hiked through 1994, the entire curve sold off violently — a level shock more than a curve trade — but it reshaped how desks thought about *duration* and *DV01* risk, the very tools that make curve trades possible. The episode (covered alongside other [bond crises](/blog/trading/fixed-income/the-great-bond-crises-1994-2008-2020-2022-and-2023)) is a reminder that even a "level-neutral" curve book can be threatened if the level move is large enough to break the assumptions behind the hedge.

**Relative-value butterfly desks.** Less visible to the public, large relative-value funds and bank trading desks run butterfly books continuously, exploiting small, mean-reverting mispricings in the belly of the curve — often around Treasury auction cycles, when a wave of new supply in one maturity temporarily cheapens it. These trades are low-margin and high-volume, leaning on leverage to turn a few basis points of curvature mean-reversion into a real return. The blow-up risk is the same one that felled Long-Term Capital Management in 1998: when many leveraged players hold the same relative-value position and are forced to unwind at once, the "mean-reverting" spread can blow out instead, turning a hedged-looking book into a cascade of losses.

## When this matters to you, and further reading

You may never put on a 2s10s steepener — but the curve's shape touches your life whether you trade it or not. The slope of the curve is the bond market's collective forecast of the economy and the Fed, distilled into a single number you can check any day. A steepening front end is the market telling you rate cuts (and cheaper mortgages, car loans, and business credit) may be coming; a deepening inversion is the market warning that the Fed has tightened hard enough to risk a recession. Reading the curve is, in a real sense, reading the price of money across time — the spine of [this whole series](/blog/trading/fixed-income/why-bonds-rule-the-world-fixed-income-introduction).

To go deeper from here: revisit [the yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) for how to read the shape, [modified duration and DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk) for the sizing math that makes curve trades possible, and [yield curve inversion](/blog/trading/fixed-income/yield-curve-inversion-the-recession-signal-that-mostly-works) for what the slope says about recessions. For the policy lens — what actually moves each end of the curve — see [reading the yield curve in the macro series](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) and [how monetary policy moves bonds](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity). And for the heavy analytics behind sizing these trades precisely, the [fixed-income analytics deep dive](/blog/trading/quantitative-finance/fixed-income-analytics) is the next step up. The next post in this series turns to [what actually moves the yield curve](/blog/trading/fixed-income/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply) — the forces that drive the level, slope, and curvature you now know how to trade.
