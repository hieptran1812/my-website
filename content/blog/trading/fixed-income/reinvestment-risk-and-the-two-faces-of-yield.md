---
title: "Reinvestment risk and the two faces of yield"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Why a rate change helps and hurts a bond investor at the same time — price risk on one side, reinvestment risk on the other — and why the holding period where they cancel is exactly the bond's duration."
tags: ["fixed-income", "bonds", "reinvestment-risk", "price-risk", "duration", "immunization", "yield-to-maturity", "horizon-yield", "treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **The one idea:** every change in interest rates does two opposite things to a bond investor at once — it moves the *price* of the bond (which hurts you when you sell early) and it changes the rate you can earn *reinvesting your coupons* (which helps you when rates rise). These two forces pull in opposite directions, and the exact holding period where they cancel out is the bond's **duration**.
> - When rates rise, your bond's **price falls** (price risk), but your **future coupons get reinvested at the new, higher rate** (reinvestment risk, which now works *for* you). When rates fall, it is the reverse.
> - The **quoted yield to maturity (YTM)** quietly assumes you reinvest every coupon at the YTM itself — an assumption that almost never holds in real life.
> - Your **actual** return — the *realized* or *horizon* yield — depends on the path rates take and how long you hold, not just the number on the screen the day you buy.
> - Hold the bond for exactly its **duration**, and the price loss and the reinvestment gain offset almost perfectly. This is the engine behind **immunization**, the subject of the next post.
> - A **zero-coupon bond has no reinvestment risk at all** — there are no coupons to reinvest — so if you hold it to maturity its return is locked in the day you buy it.

You buy a bond yielding 4%, and you think you know your return: 4%. The number is right there in the quote. But here is a fact that trips up almost everyone the first time they meet it: *that 4% is not what you will actually earn* unless a very specific thing happens — unless every coupon the bond pays you over its life gets reinvested at exactly 4% too. If rates drift up or down even a little, your real, end-of-day return will be different from the quoted yield. Sometimes higher. Sometimes lower. The quote is a promise with fine print.

This is the part of bond investing that lives on the *other side* of the relationship most people already know. Everyone learns early that when rates rise, bond prices fall — that is **price risk**, and we covered its mechanics in [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much). But rising rates do something else at the same time, something that quietly *helps* you: every coupon you collect from now on can be reinvested at the new, higher rate. That is **reinvestment risk**, and it is the hidden twin of price risk. The two are not independent annoyances. They are the same rate move seen from two sides, and they fight each other.

![A seesaw showing that when interest rates rise, a bond's price falls now which hurts the seller, while the coupons get reinvested at higher rates later which helps the holder, with the two effects pulling in opposite directions](/imgs/blogs/reinvestment-risk-and-the-two-faces-of-yield-1.png)

The diagram above is the mental model to carry through this whole post: a single rate change presses down on one side (price falls, you lose if you sell) and lifts the other (coupons compound faster, you gain if you wait). Whether the net effect on your wealth is positive or negative depends entirely on *when* you need the money. Sell tomorrow and price risk dominates. Hold long enough and reinvestment makes you whole. Somewhere in between is a sweet spot where the two cancel — and that sweet spot turns out to be one of the most important numbers in all of fixed income. We will build the whole thing from a single \$1,000 bond.

If you want the heavy mathematical machinery behind duration and reinvestment, the sibling post [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics) does the calculus. Here we stay concrete: real coupons, real dollars, one bond, and a rate that jumps overnight so we can watch both forces fire at once.

## Foundations: the words you need before the two faces make sense

Before we can be precise about how price risk and reinvestment risk offset, we have to nail down a handful of terms. None of them are hard. Each is a name for something you can already picture.

A **bond** is a tradable loan. You hand the issuer money today; they promise you a fixed schedule of payments — a **coupon** (fixed interest) each period and the **face value** (also called **par** or **principal**, the amount printed on the bond, our standard \$1,000) back at the end. The day the loan ends is the **maturity**.

The **coupon rate** is the fixed interest as a percent of face value. A 4% coupon on a \$1,000 bond pays \$40 a year, forever, no matter what rates do later. That word *fixed* is the whole reason both of our risks exist: because the coupon can't change, everything else — price, reinvested earnings — has to do the adjusting.

The **market interest rate** (or **required yield**, or just "rates") is the return investors currently demand to lend for that length and risk. It floats. When people say "rates went up," this is the number that moved.

**Yield to maturity (YTM)** is the single discount rate that makes the present value of all the bond's future cash flows equal to the price you pay. Loosely, it is "the internal rate of return if you hold to maturity." It is the number quoted on your screen. The thing nobody tells beginners: the YTM is only the return you *actually realize* under one strict assumption, which we will dismantle in a moment.

**Reinvesting** means taking a coupon when it arrives and putting it back to work — buying more bonds, rolling it into a money-market fund, whatever earns interest. A \$40 coupon received in year 1 doesn't just sit in a drawer; if you reinvest it at 5%, it grows to \$42 by year 2, \$44.10 by year 3, and so on. The rate you can reinvest *at* is set by the market when each coupon arrives — which is exactly why a future rate change matters to a bond you already own.

**Price risk** (also called **market risk** or **interest-rate risk**) is the risk that the bond's market price moves against you before you sell. Rates up, price down. If you have to sell early, you eat that loss.

**Reinvestment risk** is the risk that the rate at which you reinvest your coupons turns out to be lower (or higher) than you'd planned. Rates *down* hurt the reinvester; rates *up* help them. Notice the sign is the *opposite* of price risk — that opposition is the entire point of this post.

**Realized yield** (also **horizon yield** or **holding-period yield**) is the return you *actually* earn over your specific holding period, given the price you paid, the coupons you collected, the rates you reinvested them at, and the price you sold for. This is the honest number. The quoted YTM is a special case of it that only comes true if rates never move.

**Duration** is — for now — the holding period at which price risk and reinvestment risk exactly offset. (There is a deeper definition involving the weighted average time to receive the bond's cash flows; we'll connect the two later. For a coupon bond, duration is a few years shorter than maturity.) Duration is the hinge of this entire post.

With those terms — bond, coupon, face value, market rate, YTM, reinvesting, price risk, reinvestment risk, realized yield, duration — you have everything you need. Now let's make the two faces exact.

## The fine print in every yield quote: the reinvestment assumption

Here is the claim that surprises everyone: **the quoted yield to maturity silently assumes you reinvest every coupon at the YTM.** It is baked into the math. When a bond is quoted at 4% YTM, the only way you actually walk away with a 4% compound return is if every \$40 coupon you receive over the bond's life is itself reinvested at 4% until maturity.

Why? Because YTM is defined as the single rate $y$ that solves

$$
P = \sum_{t=1}^{N} \frac{C}{(1+y)^t} + \frac{F}{(1+y)^N}
$$

where $P$ is the price you pay, $C$ is the coupon, $F$ is the face value, $N$ is the number of periods, and $y$ is the yield to maturity. Discounting at $y$ is mathematically identical to *compounding forward at $y$*. So when this equation says your money grows at $y$, it is assuming each coupon, once received, keeps growing at $y$ — not at whatever rate the market actually offers when the coupon lands. The YTM is a round-trip rate: discount in, compound out, same rate both ways.

That assumption is almost never true. Rates move constantly. The coupon you receive in year 3 gets reinvested at year-3's rate, which is some number nobody knew on the day you bought. So the YTM is best read not as "the return you will get" but as "the return you would get *if* the world stood still." It is a benchmark, not a guarantee.

#### Worked example: the same bond, two reinvestment worlds

Let's make this concrete with our running example. You buy the **Northwind 4s of 2031** — a \$1,000 par, 4% annual-coupon, 5-year bond — at par (\$1,000), so its quoted YTM is exactly 4%. You plan to hold all five years to maturity. The quote says 4%. What do you actually earn?

It depends entirely on what you do with the \$40 coupons. Walk both worlds:

- **World A — reinvest at 4% (the YTM assumption).** Each \$40 coupon compounds at 4% from the day it arrives until year 5. The year-1 coupon grows for 4 years: \$40 × 1.04⁴ = \$46.79. The year-2 coupon grows for 3 years: \$40 × 1.04³ = \$44.99. Year 3: \$40 × 1.04² = \$43.26. Year 4: \$40 × 1.04 = \$41.60. Year 5: \$40 (no time to grow) = \$40.00. Add them: the coupons-plus-growth total **\$216.65**. Add the \$1,000 principal back: you end with **\$1,216.65**. Your compound return: (1,216.65 / 1,000)^(1/5) − 1 = exactly **4.00%**. The quote came true — because you reinvested at the YTM.

- **World B — reinvest at 2% (rates fell).** Same coupons, but now each one only earns 2%. Year-1 coupon: \$40 × 1.02⁴ = \$43.30. Year 2: \$40 × 1.02³ = \$42.45. Year 3: \$40 × 1.02² = \$41.62. Year 4: \$40 × 1.02 = \$40.80. Year 5: \$40.00. Total coupons-plus-growth: **\$208.16**. Plus \$1,000 principal = **\$1,208.16**. Your compound return: (1,208.16 / 1,000)^(1/5) − 1 = **3.85%**. You bought a "4%" bond and earned 3.85% — the missing 15 basis points is reinvestment risk biting.

The gap is real money: \$8.49 over five years on a \$1,000 bond, purely from where you parked the coupons. *Your realized return is not the quoted yield — it is the quoted yield only if you reinvest at the quoted yield, and a different number otherwise.*

### How big is the reinvestment assumption, really?

It's tempting to wave this away — "15 basis points, who cares." But the size of the reinvestment effect grows with three things, and on real bonds it can be enormous, not marginal.

First, it grows with **coupon size**. A high-coupon bond throws off more cash early, and more cash to reinvest means more exposure to whatever rate the market offers. A 10% bond has roughly twice the reinvestment exposure of a 5% bond of the same maturity, because twice as many dollars are hitting your account and looking for a home. The 4% on our Northwind 4s is modest; on a junk bond paying 9%, reinvestment risk is a first-order driver of total return.

Second, it grows with **maturity**. Over 5 years, a coupon reinvested at the wrong rate has only a few years to compound the error. Over 30 years, that same coupon's growth path diverges enormously: a coupon received in year 2 of a 30-year bond gets reinvested for 28 years, and the difference between compounding it at 3% versus 6% over 28 years is a factor of more than two. For long bonds, *most of your total dollar return can come from reinvested coupons rather than from the coupons themselves* — a fact that astonishes people the first time they compute it.

#### Worked example: on a 30-year bond, reinvested coupons dwarf the original coupons

Buy a 30-year \$1,000 bond with a 6% coupon (\$60 a year), held to maturity, and reinvest every coupon at 6%. The thirty \$60 coupons, by themselves, total \$1,800. But each one compounds. The year-1 coupon grows for 29 years at 6% (\$60 × 1.06²⁹ = \$326); the year-2 coupon for 28 years (\$308); and so on down to the final \$60. Sum the grown-up coupons and you get roughly **\$4,743** — meaning about **\$2,943 of "interest on interest"** sits on top of the \$1,800 of raw coupons. Add the \$1,000 principal and you end with about \$5,743. Of your total \$4,743 in earnings beyond principal, **62%** came from reinvested coupons, not from the coupons themselves. Now imagine reinvesting at 3% instead of 6%: the interest-on-interest collapses, and your realized 30-year yield drops from 6% toward 4.5%. *On long bonds, the reinvestment rate isn't a footnote — it's the majority of your return, which is why the quoted YTM is least trustworthy precisely for the longest bonds.*

Third, the effect grows with **how far rates move**. A 25-basis-point wiggle barely registers; a multi-point regime change, like 2022's jump from near-zero to 4%+, rewrites the entire reinvestment picture. The reinvestment assumption is small when rates are stable and large when they aren't — and rates are rarely stable for long.

## The two faces: price risk and reinvestment risk are the same rate move, seen from opposite sides

Now the heart of it. Suppose you buy our Northwind 4s at par, and the *very next day*, the market rate jumps from 4% to 6%. Two things happen simultaneously, and they point in opposite directions.

**Face one — price risk (the hit, felt now).** Your bond pays \$40 a year, but the market now wants 6%. Nobody will pay you \$1,000 for a 4% bond when fresh 6% bonds exist. To sell, you must drop your price until your bond delivers 6% to the buyer. Discounting the same five payments at 6% gives roughly **\$915.75** — a loss of about \$84 if you sold today. That is price risk: rates up, price down, and you eat it if you sell early.

**Face two — reinvestment risk, working *for* you (the help, felt later).** But you don't have to sell. If you hold, every coupon from now on gets reinvested not at 4% but at the new 6%. Your \$40 coupons compound faster. Over the remaining life of the bond, that extra reinvestment income piles up. The longer you hold, the more it accumulates — and at some point it fully recovers the price hit you took on day one.

These two faces are not separate risks you can diversify away. They are the *same* rate move (4% → 6%) producing two opposite effects on your wealth. Price risk says "rates up is bad." Reinvestment risk says "rates up is good." Which one wins depends on one variable: **how long you hold.**

![A before and after comparison showing that a rate rise from four to six percent immediately drops the bond price below par, the loss leg, while over the remaining years the coupons reinvested at the higher rate accumulate extra income, the gain leg](/imgs/blogs/reinvestment-risk-and-the-two-faces-of-yield-2.png)

The figure splits the single rate rise into its two consequences: on the left, the immediate mark-to-market loss (price falls from \$1,000 to about \$916); on the right, the slow accumulation of extra reinvestment income as each coupon now earns 6% instead of 4%. The loss is a lump, felt instantly. The gain is a trickle, felt over years. Early in your holding period the lump dominates and you're underwater; late in your holding period the trickle has caught up and surpassed it. The crossing point — where trickle exactly equals lump — is the number we're chasing.

#### Worked example: rates jump to 6% the day after you buy — trace both faces over five years

Let's trace the full life of the Northwind 4s after the overnight jump to 6%, holding all five years to maturity, and compare it to the "rates never moved" baseline.

*The price hit (felt only if you sell early):* the day after the jump, the bond is worth about **\$915.75**, an \$84.25 unrealized loss. But if you hold to maturity, the bond pays back its full \$1,000 face value regardless — so the *price* loss disappears at maturity. What you keep is the reinvestment gain.

*The reinvestment gain (felt if you hold):* now your \$40 coupons compound at 6%, not 4%. Year-1 coupon grows 4 years: \$40 × 1.06⁴ = \$50.50. Year 2: \$40 × 1.06³ = \$47.64. Year 3: \$40 × 1.06² = \$44.94. Year 4: \$40 × 1.06 = \$42.40. Year 5: \$40.00. Total coupons-plus-growth: **\$235.48**. Add \$1,000 principal = **\$1,235.48**. Your realized 5-year return: (1,235.48 / 1,000)^(1/5) − 1 = **4.32%**.

Compare: under the "rates stay at 4%" world you earned exactly 4.00% (World A above). After the jump to 6%, holding to maturity, you earned **4.32%** — *more* than the quote promised. The rate rise that would have cost you \$84 if you'd panicked and sold on day two actually *helped* you, because you held long enough for the reinvestment boost to more than repay the price loss. *Over a long enough horizon, a rate rise is good news for a bond holder, not bad — the opposite of the day-one headline.*

### Why the two faces must point in opposite directions

It is worth pausing on *why* this opposition is guaranteed, not just convenient. The price of a bond is the present value of its future cash flows; a higher market rate discounts those future dollars more harshly, so price falls. That is a statement about *value today*. But the reinvestment of coupons is a statement about *growth into the future*; a higher market rate compounds your reinvested dollars faster, so your future wealth rises. Discounting and compounding are the same operation run in opposite directions — one looks back from the future to today, the other looks forward from today to the future — so any single rate change *must* move price and reinvestment value in opposite directions. The opposition isn't an empirical regularity that might break; it's baked into the arithmetic of the time value of money.

This is also why you can never escape both at once. A portfolio with no price risk (all cash, reinvested daily) has maximum reinvestment risk — every dollar is constantly re-priced at the new rate. A portfolio with no reinvestment risk (a single zero held to maturity) has maximum price risk if you're forced to sell early. You're always somewhere on a spectrum that trades one for the other, and the only free variable that lets you balance them is your holding period.

### The mark-to-market trap

A subtle and costly mistake hides in the day-one price drop. When rates jumped to 6%, your bond's *market value* fell to about \$916. If you check your brokerage statement, it shows an \$84 loss in red. That number is real in the sense that it's what you'd get if you sold *right now* — but it is *not* a loss you've actually suffered if your plan was to hold. The accounting term is **mark-to-market**: your position is re-valued at current prices every day, whether or not you intend to transact at those prices.

The trap is treating the mark-to-market loss as the whole story and selling in a panic — thereby converting a temporary, paper loss into a permanent, realized one, *and* throwing away the reinvestment boost that would have repaid it. The investor who understands the two faces sees the red number, shrugs, and keeps collecting coupons at the now-higher rate. The investor who doesn't sells at the bottom. Same bond, same rate move, opposite outcomes — the only difference is understanding which face you're looking at.

## Realized yield versus the quoted YTM: what you actually earn

We now have two numbers for the "same" 4% bond: the **quoted YTM** (4%, fixed at purchase) and the **realized yield** (what you actually earned, which came out at 3.85%, 4.00%, or 4.32% depending on the reinvestment rate). The quoted YTM is a single dot on the screen. The realized yield is a curve — it bends with the rate you reinvest at.

The relationship is clean and worth memorizing: **if you reinvest below the YTM, you earn less than the YTM; reinvest above it, you earn more; reinvest exactly at it, you earn the YTM.** The quoted yield is the pivot point, the one reinvestment rate at which the promise comes exactly true.

![An XY chart with reinvestment rate on the horizontal axis and realized five year yield on the vertical axis, showing a rising line that passes through the point where reinvesting at four percent yields exactly the quoted four percent, below it for lower reinvestment rates and above it for higher ones](/imgs/blogs/reinvestment-risk-and-the-two-faces-of-yield-4.png)

The chart plots realized 5-year yield (vertical) against the rate you reinvest your coupons at (horizontal), for our Northwind 4s held to maturity. The line slopes up: higher reinvestment rates mean higher realized return. It crosses the quoted-YTM line exactly where the reinvestment rate equals 4% — that is the one point where "what you earn" equals "what you were quoted." Everywhere to the left (you reinvested at less than 4%) you fall short; everywhere to the right (you reinvested above 4%) you beat the quote. The quote is not a lie, but it is a *conditional* truth, and the condition is the entire x-axis of this picture.

#### Worked example: realized yield across low, median, and high reinvestment rates

Hold the Northwind 4s to maturity (5 years), vary only the reinvestment rate, and read off the realized return. Each row reinvests every coupon at the stated rate:

| Reinvestment rate | Coupons + growth | Total at year 5 | Realized 5-yr yield |
|---|---|---|---|
| 2% (rates fell) | \$208.16 | \$1,208.16 | 3.85% |
| 4% (= quoted YTM) | \$216.65 | \$1,216.65 | 4.00% |
| 6% (rates rose) | \$235.48 | \$1,235.48 | 4.32% |

The realized yield ranges from 3.85% to 4.32% — a spread of 47 basis points — *on the exact same bond*, driven by nothing but where the coupons were reinvested. Note the asymmetry: the 6% case beats the quote by 32 bps while the 2% case lags by only 15 bps, because compounding at a higher rate over multiple years has a convex, accelerating effect. *The yield you're quoted is the yield you'd get in a frozen world; the yield you actually earn is a moving target that the rate path drags around.*

### Why sophisticated investors quote a "horizon yield" instead

Because the screen's YTM is a conditional truth, professional fixed-income investors rarely take it at face value for a real decision. Instead they compute a **horizon yield** (or **total-return yield**): they pick an explicit holding period, make an explicit assumption about the reinvestment rate, assume a sale price at the horizon, and solve for the return that ties it all together. It's the same realized-yield calculation we just did, dressed in formal clothes and made the centerpiece of the analysis rather than an afterthought.

The horizon-yield framing forces the two faces into the open. To compute it you *must* state (a) how long you'll hold — which sets your price-risk exposure at the sale date — and (b) what rate you'll reinvest at — which sets your reinvestment income. The single YTM number hides both of these behind one figure; the horizon yield exposes them. A bond desk comparing two bonds will often find that the one with the *lower* quoted YTM has the *higher* horizon yield for their actual holding period, because its cash-flow timing fits their horizon better. The quote ranks them one way; the honest analysis ranks them the other.

This is also where the path of rates, not just the level, starts to matter. Our table above assumed a single reinvestment rate for all five years. In reality rates wander, and each coupon is reinvested at *that year's* rate, then those proceeds re-roll at later years' rates. The realized yield is a function of the entire *path*, which is why it can never be known in advance — only estimated under a scenario. The best you can do is choose a horizon that makes the answer insensitive to the path, which is exactly what the next section is about.

## The influence picture: realized return depends on how long you hold and which way rates moved

Here is where the two faces resolve into a single, beautiful pattern. We have established that a rate *rise* hurts your price now but helps your reinvestment later. So the realized return from a rate rise *depends on your holding period*: sell early and you're down; hold long and you're up. The same logic runs in reverse for a rate *fall*: you get an instant price gain (you could sell for a profit) but your coupons now reinvest at a lower rate, dragging your long-run return down.

Plot realized return against holding period for both scenarios and something remarkable appears: **the two lines cross at a single holding period — and that holding period is the bond's duration.**

![An XY chart with holding period in years on the horizontal axis and total realized return on the vertical axis, showing two lines that cross at the duration mark near four point five years, the rate rise line starting low then climbing above the rate fall line which starts high then declining](/imgs/blogs/reinvestment-risk-and-the-two-faces-of-yield-3.png)

The chart is the influence picture of this entire post. On the horizontal axis is your holding period in years; on the vertical, your total realized return. The blue line is the "rates rose to 6%" scenario; the gray line is the "rates fell to 2%" scenario. Trace the blue line: at a short horizon you're below the original 4% (price loss dominates), but it climbs as the reinvestment boost accumulates, and by maturity you're well above 4%. The gray line does the mirror image: a short-horizon price *gain* puts you above 4% early, but the lower reinvestment rate drags you down over time. The two lines cross at one point — and that point, near 4.5 years for our Northwind 4s, is the bond's **duration**. At exactly that horizon, your realized return is the *same* whether rates rose or fell. The rate move stopped mattering. That is not a coincidence; it is the definition of duration in disguise, and it is the foundation of the next post on immunization.

#### Worked example: the same bond at three holding periods after a rate rise

Take the Northwind 4s, suppose rates jump to 6% on day one, and ask: what's my realized return if I hold for 2 years, for 4.5 years (the duration), or for the full 5 years? At each horizon you collect coupons (reinvested at 6%) plus the sale price of the bond at that point (a bond with fewer years left, priced at 6%).

- **Hold 2 years, then sell.** You collect 2 coupons, reinvested at 6%: \$40 × 1.06 + \$40 = \$82.40. You sell a now-3-year bond at 6%: it's worth about \$946.54. Total: \$1,028.94. Realized 2-year return: (1,028.94/1,000)^(1/2) − 1 = **1.43% a year** — well *below* the 4% you expected, because the price loss hasn't been repaid yet.
- **Hold ~4.5 years (the duration).** Coupons reinvested at 6% plus the sale price of the stub bond come out to a realized annualized return of almost exactly **4.00%** — the original yield, untouched by the rate move. The price loss and the reinvestment gain have netted to zero.
- **Hold the full 5 years.** As computed earlier, **4.32%** — now the reinvestment gain has *overshot* the price loss, and you're ahead of the quote.

Read the three numbers in order — 1.43%, 4.00%, 4.32% — and you can see the price-risk drag fading and the reinvestment-risk boost taking over as the horizon lengthens. *Duration is the horizon where the fade and the boost exactly trade places.*

### What the crossing point really means

Stare at the crossing point a moment longer, because it carries the whole punchline. To the *left* of the duration, the two lines fan apart with the rate-fall scenario on top: if you must sell early, you'd rather rates had fallen (price gain) than risen (price loss). To the *right* of the duration, the lines have swapped — the rate-rise scenario is on top: if you hold long, you'd rather rates had risen (reinvestment windfall) than fallen (reinvestment drag). At the duration itself, the lines touch: the rate move you experienced makes *no difference* to your realized return. You've earned your original 4% either way.

This is a genuinely surprising claim, so let it land. We started by saying interest-rate risk is the central danger of bond investing. The crossing point says: that danger has a hole in it. If you hold for exactly the duration, a *one-time* rate change — up or down — leaves your realized return essentially unchanged. You haven't hedged with derivatives or paid for insurance; you've simply chosen a holding period that lets the bond's two opposing sensitivities cancel themselves. That self-canceling horizon is the single most useful fact in the practical management of bond portfolios, and the entire discipline of **immunization** is built on engineering it deliberately.

The caveat that keeps it honest: the cancellation is exact only for a *small, one-time, parallel* shift in rates that happens *right after you buy*. For large moves, repeated moves, or moves that arrive partway through your holding period, the offset is approximate, and the residual gap is governed by **convexity** — the curvature we met in [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much). But "approximately immune to rate moves if you hold for the duration" is still a remarkably powerful tool, and it's astonishing how much of institutional fixed income runs on it.

## Why the offset happens at duration: the cash-flow timeline view

Why *that* horizon and not some other? The cleanest way to see it is to look at when the bond actually pays you. A coupon bond hands you money at many dates: small coupons early, then the big lump of principal at maturity. Duration is the *weighted average* of those payment dates — weighted by how much present value each payment carries. For our 5-year 4% bond, most of the value sits in the final \$1,040 payment, so the weighted-average payment date lands around year 4.5, not year 5.

Now connect that to the two faces. Each coupon you receive *before* your horizon gets reinvested — so coupons are exposed to **reinvestment risk** (their fate depends on future rates). The bond's *remaining value* that you'll sell *at* your horizon is exposed to **price risk** (its sale price depends on future rates). The catch: reinvestment risk and price risk respond to rates in opposite directions. If your horizon is too short, too much value is still locked in the bond's price (price risk dominates). If your horizon is too long, too much value has already been paid out and reinvested (reinvestment risk dominates). At exactly the duration, the average dollar arrives right at your horizon — and the two exposures balance.

![A cash flow timeline of a five year four percent bond showing four small coupons of forty dollars in years one through four and a large final payment of one thousand forty dollars at year five, with each coupon reinvested and compounding forward at the new rate toward the holding horizon](/imgs/blogs/reinvestment-risk-and-the-two-faces-of-yield-5.png)

The timeline shows the bond's actual cash flows: a \$40 coupon at the end of each of years 1 through 4, then \$1,040 (final coupon plus \$1,000 principal) at year 5. The green arrows above the line are inflows you receive; the curved arrows show each early coupon being carried forward and compounded at the reinvestment rate toward your horizon. Because the big payment sits at the end, the *center of mass* of the cash flows — the duration — lands around year 4.5. That center of mass is precisely the holding period at which the reinvestment growth of the early coupons exactly offsets the price sensitivity of the late principal. The geometry of when you get paid *is* the duration, and the duration *is* the offset horizon.

#### Worked example: where the center of mass of a 5-year 4% bond actually sits

Duration is a present-value-weighted average of payment times. For the Northwind 4s priced at par (4% YTM), discount each payment, multiply by its year, sum, and divide by the price:

| Year (t) | Payment | PV at 4% | t × PV |
|---|---|---|---|
| 1 | \$40 | \$38.46 | \$38.46 |
| 2 | \$40 | \$36.98 | \$73.96 |
| 3 | \$40 | \$35.56 | \$106.68 |
| 4 | \$40 | \$34.19 | \$136.76 |
| 5 | \$1,040 | \$854.80 | \$4,274.00 |
| **Sum** | | **\$1,000.00** | **\$4,629.86** |

Duration = \$4,629.86 / \$1,000.00 = **4.63 years** (this is Macaulay duration). So our 5-year bond has its center of mass at 4.63 years — and that, not 5 years, is the horizon at which price risk and reinvestment risk cancel. *A coupon bond's offset horizon is always shorter than its maturity, because the coupons pull the average payment date forward.*

### The two definitions of duration are the same thing

We've now met duration twice, defined two different ways, and it's worth showing they're the same. Earlier we said duration is "the holding period at which price risk and reinvestment risk cancel." Just now we computed it as "the present-value-weighted average time to receive the cash flows." These sound unrelated, but they're identical, and seeing why ties the whole post together.

Think of the bond as a portfolio of individual cash flows — five separate payments, each like a tiny zero-coupon bond. Each payment has its own little balance of the two faces. A payment due *before* your horizon spends its remaining life being *reinvested* (reinvestment risk). A payment due *after* your horizon is still *in the bond*, to be sold at your horizon (price risk). When your horizon equals the weighted-average payment date, the cash flows arriving early — and reinvested — exactly balance the cash flows arriving late — and re-priced. The mathematical reason is that price sensitivity falls off linearly with how much *sooner* a payment comes due, and reinvestment sensitivity rises linearly with how much *later* it can compound; the two linear effects cross at the weighted average. So "where the two risks cancel" and "the average payment date" land on the same number by construction.

This is why duration, a number that *looks* like it's only about price sensitivity (how much price moves per 1% rate change), is *also* the immunizing horizon. One number wears two hats because the two faces are two sides of one coin.

#### Worked example: how the offset horizon shifts with the coupon

The duration — and therefore the offset horizon — depends on the coupon, because higher coupons pull the center of mass forward. Compare three 5-year, \$1,000 bonds at a 4% market rate:

- **A 0% coupon (a zero):** all the value is in the single year-5 payment, so duration = **5.00 years**, equal to maturity.
- **A 4% coupon (our Northwind 4s):** modest coupons pull the average in slightly, giving duration = **4.63 years**.
- **A 10% coupon:** big \$100 coupons arrive early and carry real present value, pulling the center of mass down to roughly **4.31 years**.

Same 5-year maturity, three different offset horizons — 5.00, 4.63, 4.31 years — purely because of how front-loaded the cash flows are. *The fatter the coupon, the sooner the average dollar arrives, the shorter the horizon at which the two faces cancel.*

## The zero-coupon bond: no coupons, so no reinvestment risk

Now flip the whole problem on its head. What if a bond pays *no coupons at all*? A **zero-coupon bond** (a "zero") makes a single payment: it's sold today at a discount and pays back face value at maturity, with nothing in between. A 5-year zero might cost \$822 today and pay \$1,000 in five years; the difference *is* your interest.

Here is the elegant consequence: **a zero-coupon bond has no reinvestment risk, because there are no coupons to reinvest.** All of your return is locked in a single future payment. If you hold it to maturity, your realized return is *exactly* the yield you bought it at, no matter what rates do in between. There is no coupon arriving in year 3 that has to be reinvested at year-3's unknown rate. There is just one cash flow, at the end.

This also makes the duration story click into place. A zero's duration *equals its maturity* — all the value is in that one final payment, so the center of mass is literally the maturity date. And sure enough, a zero held to its maturity (= its duration) has its price risk and reinvestment risk both fully resolved: reinvestment risk is zero by construction, and price risk vanishes because you hold to the day it pays face value. The offset-at-duration rule holds in its purest form when the bond *is* a single payment at its duration.

![A before and after comparison showing on the left a coupon bond with several coupon payments each carrying reinvestment risk, and on the right a zero coupon bond with a single payment at maturity and no coupons to reinvest, so no reinvestment risk](/imgs/blogs/reinvestment-risk-and-the-two-faces-of-yield-6.png)

The figure contrasts the two. On the left, the coupon bond: a stream of small coupons, each tagged with reinvestment risk because its future earning rate is unknown. On the right, the zero: a single payment at maturity, no coupons, nothing to reinvest, so the reinvestment-risk tags are gone. The coupon bond gives you cash sooner — useful if you need income — but every dollar that arrives early carries the risk of being reinvested at a worse rate. The zero gives you nothing until the end, but in exchange it removes reinvestment risk entirely and locks your return. *If you have a known future liability — tuition in 10 years, say — a zero maturing on that date is the only bond whose return you can guarantee today.*

The trade-off is worth stating plainly, because the zero is not a free lunch. By concentrating everything in a single distant payment, the zero has the *longest possible duration* for its maturity — its duration equals its maturity exactly, while a coupon bond of the same maturity has a shorter one. Long duration means high *price* sensitivity: if you're forced to sell a long zero before maturity, its price will have swung violently with rates in the meantime. So the zero doesn't abolish interest-rate risk; it *concentrates* it into pure price risk and removes the reinvestment component. Whether that's good or bad depends entirely on whether your horizon matches the maturity. Match them, and the zero is the safest bond in existence — guaranteed return, no reinvestment uncertainty, no early-sale needed. Mismatch them, and it's one of the most volatile instruments you can hold. The zero is the cleanest illustration of the post's whole thesis: there is no bond with *zero* interest-rate risk, only bonds where you've chosen how to split the risk between the two faces, and a horizon that determines whether that split helps or hurts.

#### Worked example: a 5-year zero earns its yield no matter what rates do

You buy a 5-year zero for **\$821.93**, which prices to a 4% yield (because \$821.93 × 1.04⁵ = \$1,000). You hold to maturity. Now let rates do whatever they want in between — crash to 1%, spike to 8%, anything. What do you earn?

You earn exactly 4%. Always. Your year-5 payment is a contractual \$1,000; you paid \$821.93; the return is (\$1,000 / \$821.93)^(1/5) − 1 = **4.00%**, regardless of the rate path. There are no coupons to reinvest at a different rate, and you never sell early, so price moves in the meantime are irrelevant to your final outcome. Compare this to the coupon bond, whose realized return swung from 3.85% to 4.32% over the same rate scenarios. *The zero trades away all reinvestment risk for the simplicity of a single locked-in cash flow — that's why pension funds and insurers love them for matching long-dated promises.*

## The worked realized-return table: putting both faces in one grid

To pull everything together, let's lay out the realized return on the Northwind 4s across two dimensions at once: the rate scenario (did rates fall, stay, or rise) and the holding period (did you sell early, hold to duration, or hold to maturity). This single grid is the whole post in one picture.

![A matrix grid with holding period across the top as two years and four point six years and five years, and rate scenario down the side as rates fell and unchanged and rates rose, showing the realized return in each cell with the duration column showing nearly identical returns across all rate scenarios](/imgs/blogs/reinvestment-risk-and-the-two-faces-of-yield-7.png)

The matrix puts holding period across the columns (2 years / 4.6 years = duration / 5 years) and the rate scenario down the rows (rates fell to 2% / unchanged at 4% / rose to 6%). Read across any row and you see how holding longer changes your fate. Read down any column and you see how the rate move changes it. But the magic is the middle column — the duration column: the realized returns there are nearly identical (all close to 4.00%) *regardless of which way rates moved*. That is immunization in a single column. Outside that column, the rate scenario matters a lot; inside it, it barely matters at all. The two faces of yield cancel, by construction, at the duration.

#### Worked example: reading the grid — why the duration column is flat

Take the three cells in the duration column (hold ~4.6 years):

- **Rates fell to 2%:** you took an instant price *gain* (could sell high), but reinvested coupons at only 2%. Over 4.6 years these roughly net to a realized **~4.0%**.
- **Rates unchanged at 4%:** nothing moved; you simply earn the **4.0%** you were quoted.
- **Rates rose to 6%:** you took an instant price *loss*, but reinvested coupons at 6%. Over 4.6 years these roughly net to a realized **~4.0%**.

Now contrast the same three scenarios in the 2-year column: roughly 6.3% (rates fell, you sell the bond at a fat price before the low reinvestment rate can hurt), 4.0% (unchanged), and 1.4% (rates rose, you eat the price loss before reinvestment can repay it). The 2-year row spans nearly 5 percentage points; the duration row spans almost nothing. *Match your holding period to the bond's duration and you've neutralized interest-rate risk — that's the entire trick, and it falls straight out of the two faces canceling.*

## Common misconceptions

**"My bond yields 4%, so I'll earn 4%."** Only if you reinvest every coupon at 4% and hold to maturity. The quoted YTM bakes in a reinvestment-at-YTM assumption that almost never holds. Your *realized* return depends on the rates you actually reinvest at — it can land anywhere from below to above the quote, as our 3.85%-to-4.32% spread showed for a single bond.

**"Rising rates are always bad for a bond investor."** Only if you sell before the reinvestment boost catches up. Rising rates hurt your *price* today but *help* your reinvestment income for the rest of the bond's life. Hold past the duration and a rate rise leaves you *better* off than if rates had never moved — our Northwind 4s earned 4.32% after a rate jump, versus 4.00% if rates had stayed put.

**"Reinvestment risk and price risk are two separate things I should worry about independently."** They are the *same* rate move seen from opposite sides, and they fight each other. You can't eliminate both; you can only choose a holding period that balances them. At the duration they cancel; away from it, one dominates — but you never face the full force of both at once in the same direction.

**"A longer bond always has more interest-rate risk."** It has more *price* risk and *less* reinvestment risk per dollar — but what actually matters for a holder is the *net*, which is governed by duration relative to your horizon, not by maturity alone. A 30-year bond held for 30 years and a 5-year bond rolled six times can have very different risk profiles even though both "cover" 30 years.

**"Zero-coupon bonds are the riskiest bonds because they pay nothing until the end."** For *price* risk, yes — a zero has the maximum duration for its maturity, so its price swings hardest when rates move. But for *reinvestment* risk it is the *safest* bond there is: zero, by construction. If your goal is to lock in a guaranteed return to a known future date, a zero is the *least* risky instrument, not the most. The "risk" label depends entirely on which face you're looking at.

**"YTM and realized yield are basically the same number."** They coincide only in a frozen world. In any real market where rates move, the realized yield is a different, path-dependent number. Sophisticated investors quote a *horizon yield* under an explicit reinvestment assumption precisely because they know the screen's YTM is a benchmark, not a forecast.

## How it shows up in real markets

**The 2022 rate shock and the "hold to maturity" debate.** When the Federal Reserve raised its policy rate from near 0% to over 4% during 2022, bond *prices* fell hard — the longest Treasuries lost 30%+ of their market value, the worst year for bonds in modern history. Headlines screamed about losses. But buy-and-hold investors who held individual bonds to maturity faced a very different reality: their bonds still paid full face value at the end, and every coupon and maturing bond they held could now be reinvested at 4%+ instead of near zero. For a long-horizon investor, 2022's rate spike was, on a hold-to-maturity basis, a *gift* — exactly the two-faces dynamic, with the price-loss headline masking a reinvestment-income windfall. The episode is a textbook case of why mark-to-market losses and realized losses diverge; see [why bonds rule the world](/blog/trading/fixed-income/why-bonds-rule-the-world-fixed-income-introduction) for the broader stakes.

**Pension funds and liability-driven investing (LDI).** Defined-benefit pension funds owe specific dollar amounts at specific future dates — a stream of liabilities that behaves exactly like a bond. To make sure they can pay regardless of rate moves, they *immunize*: they buy bond portfolios whose **duration matches the duration of their liabilities**, so that price risk and reinvestment risk cancel at the horizon when the payments come due. This is the two-faces offset deployed at the scale of hundreds of billions of dollars. When UK LDI funds blew up in September 2022, the proximate trigger was a leverage/collateral spiral — but the underlying discipline they practice is precisely the duration-matching this post derives.

**STRIPS: the market manufactures pure zeros.** The U.S. Treasury lets dealers "strip" a coupon bond into its individual cash flows and sell each one as a separate zero-coupon security, called a STRIP. Why does demand exist? Because zeros have *no reinvestment risk*: an insurer or pension that owes a fixed sum in 2045 can buy a 2045 STRIP and lock in its return today, immune to every rate move between now and then. The STRIPS market — hundreds of billions outstanding — exists largely to let institutions buy the one instrument that removes reinvestment risk entirely.

**Bond ladders for retirees.** A common income strategy is to build a *ladder* — bonds maturing every year for, say, ten years. The hidden logic is reinvestment-risk management: as each rung matures, the principal is reinvested at *current* rates. If rates have risen, the reinvestment happens at the higher rate (good); if they've fallen, only one rung reinvests at a time, smoothing the hit. A ladder is, in effect, a way to dollar-cost-average your reinvestment-rate exposure rather than betting it all on one future rate. It is reinvestment risk, managed by diversification across time.

**Callable bonds: reinvestment risk weaponized against you.** When a bond is *callable*, the issuer can repay it early — and they do exactly when it hurts you most: when rates have *fallen*. They hand you your principal back precisely when the only thing you can do with it is reinvest at the new, lower rate. The call feature is, in essence, the issuer forcing the worst-case reinvestment scenario on you, which is why callable bonds must offer a higher yield to compensate. We unpack the mechanics in [the many yields: current yield, YTM, and yield to call](/blog/trading/fixed-income/the-many-yields-current-yield-ytm-and-yield-to-call).

**Mortgages and the homeowner's free option.** A 30-year mortgage is, to the lender, a long bond — and homeowners can prepay it (refinance) whenever rates fall, which is a call option handed to the borrower for free. When rates drop, millions of homeowners refinance at once, repaying their old high-rate loans; the investors who held those mortgages (through mortgage-backed securities) get their principal back exactly when they can only reinvest it at the new, lower rate. This is reinvestment risk at the scale of trillions of dollars, and it has its own name: **prepayment risk**. It's why mortgage-backed securities trade at higher yields than Treasuries of similar duration — investors demand to be paid for the reinvestment risk the homeowner's prepayment option dumps on them. The 2003 and 2020–2021 refinancing waves are textbook cases: rate cuts triggered prepayment surges that handed MBS investors back their cash at the worst possible moment to redeploy it.

**Why the long bond's quote lies the most.** Recall that reinvestment's share of total return grows with maturity. This is why the quoted yield on a 30-year Treasury is, in a deep sense, the *least* reliable forecast of realized return in the whole bond market — more than half of your eventual dollars depend on reinvestment rates that won't be known for decades. Long-bond investors who lived through the 1980s saw this from the happy side: they'd locked in double-digit coupons and got to reinvest them, for years, at still-high rates, earning realized returns that beat even their generous quotes. Investors who bought 30-year bonds at the 2020 lows of ~1.3% are living the unhappy mirror image — though, per the two faces, the rate rise since then means the coupons they *do* collect now reinvest at far better rates than the quote assumed, partly cushioning the price carnage if they hold.

## When this matters to you and further reading

The moment this stops being abstract is the moment you have a *horizon* — a date you'll need the money. If your horizon is short and you might have to sell, price risk is your enemy and you want short-duration bonds. If your horizon is long and fixed (retirement, a future tuition bill), you can use the two-faces offset to your advantage: match your bond's duration to your horizon and you've largely neutralized interest-rate risk, turning a "4% bond" into something much closer to a genuine 4% promise. And if you want an *ironclad* lock with no reinvestment risk at all, a zero-coupon bond maturing on your target date is the purest instrument finance offers.

The natural next step is **immunization** — the formal strategy of duration-matching that this post's offset makes possible; it's the subject of the next post in this series. To go deeper on the surrounding ideas: [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much) builds the price-risk side in full; [price and yield: the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds) grounds the whole price–yield relationship; and for the policy backdrop that drives the rate moves doing all this work, [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) takes the macro view. The two faces of yield are, in the end, just a reminder that a bond is a promise made in dollars and time — and that the value of a promise depends on when you plan to collect.

*This is educational, not individualized financial advice. Bonds can lose money; match any strategy to your own horizon and risk tolerance.*
