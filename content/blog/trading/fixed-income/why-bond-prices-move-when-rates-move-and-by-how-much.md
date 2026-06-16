---
title: "Why bond prices move when rates move, and by how much"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the exact mechanism that links interest rates to bond prices: how big the move is, why longer and lower-coupon bonds move more, and what clean price, dirty price, accrued interest, and day-count conventions mean for the cash you actually pay."
tags: ["fixed-income", "bonds", "interest-rates", "bond-pricing", "accrued-interest", "day-count", "clean-price", "dirty-price", "duration", "us-treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — a bond's price moves opposite to interest rates, and the *size* of that move is set by how far away its cash flows are — so a 1% change in rates is almost never a 1% change in price.
> - When market yields rise, the price of an existing bond *falls*, and vice versa — the seesaw at the heart of fixed income.
> - The magnitude is the real story: a +1% rate move costs our 5-year bond about **−4.4%**, and a 30-year bond about **−15.4%**.
> - **Longer maturity** and **lower coupon** both *amplify* the move, because they push the bond's money further into the future where discounting bites hardest.
> - The price you see quoted (the **clean price**) is not the cash you pay; you also owe the seller **accrued interest**, and the sum is the **dirty (invoice) price**.
> - **Day-count conventions** (30/360, actual/actual, actual/360) decide exactly how many days of interest you owe — and they give different dollar answers on the same trade.

You have probably heard the headline a hundred times: "The Fed raised rates, and bonds sold off." It sounds like a law of nature, but it raises an obvious question that almost nobody answers. *By how much?* If rates go up by 1%, does a bond lose 1%? Lose 10%? Lose nothing? The honest answer — the one that separates someone who *knows* bonds from someone who has only heard about them — is that it depends entirely on the bond, and the difference is enormous. The exact same 1% rate move that barely scratches a short Treasury bill can vaporize 15% of a long Treasury bond's value in an afternoon.

This post is about that "by how much." We will start from the seesaw — *price up means yield down* — that you may already have met in [the price–yield introduction](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds), and we will turn it from a direction into a *magnitude*. We will see exactly why distance-in-time is the amplifier, why a tiny coupon makes a bond more dangerous, not safer, and why the price your screen quotes is a polite fiction that hides part of the bill.

Why does this deserve a whole post? Because the gap between "rates went up" and "here is what my bond is worth now" is where almost all of the real money in fixed income is made and lost. A retiree who thinks her bond fund is "safe" because it holds government debt, a bank treasurer who parks deposits in long Treasuries, a hedge fund leveraging the 30-year — all of them live or die by the *size* of the price move, not its direction. Getting the direction right is easy and nearly useless; getting the magnitude right is hard and worth everything. By the end you will be able to look at any bond and estimate, in your head, roughly how much its price will move for a given change in rates — and you will understand the settlement mechanics well enough to know what you will actually pay when you buy it.

![A before and after comparison of one bond showing that a one percent rise in market yields takes the price from one thousand dollars down to nine hundred fifty six dollars, a drop of more than four percent](/imgs/blogs/why-bond-prices-move-when-rates-move-and-by-how-much-1.png)

The diagram above is the mental model for the whole post: one ordinary bond, one ordinary 1% move in rates, and a price change four times larger than the rate change. Hold that picture in your head — a 1% input producing a 4% output — and everything that follows is just an explanation of where that multiplier comes from, and how to predict it before it happens. (Everything here is educational, not investment advice; the point is to understand the mechanism, not to tell you what to buy.)

## Foundations: the words you need before we do any math

Let's build the vocabulary from zero. If you have read the earlier posts in this series, treat this as a quick refresher; if this is your first one, do not skip it, because every later sentence leans on these definitions.

A **bond** is a tradable loan. When you buy one, you are the lender. The borrower — the **issuer** — promises to pay you a fixed stream of cash: a periodic **coupon** (the interest), and then the **face value** (also called **par**, almost always \$1,000 per bond) back at the end. The **maturity** is the date that final payment lands. So a "5-year \$1,000 par bond with a 4% coupon" is a contract to pay you 4% of \$1,000 every year — \$40 — for five years, plus your \$1,000 back at the end.

A small but crucial wrinkle: most bonds pay coupons **semiannually**, twice a year. Our 4% bond therefore pays **\$20 every six months**, not \$40 once a year. That "twice a year" habit will matter a lot when we count days, so lock it in now.

The **coupon rate** is fixed forever — it is printed on the bond and never changes. What *does* change every second the bond trades is its **price** and, mirror-image to the price, its **yield**.

- The **price** is what someone will pay you for the bond today. It is quoted as a percentage of par: a price of "95.624" means 95.624% of \$1,000, or \$956.24.
- The **yield** — more precisely the **yield to maturity (YTM)** — is the single interest rate that makes the bond's future cash flows, discounted back to today, add up to its current price. Think of it as the bond's *true* return if you buy it now and hold it to the end. A **basis point**, which you will see constantly, is one hundredth of a percent: 0.01%. "Yields rose 40 basis points" means rates rose 0.40%.

Here is the relationship that drives everything. The coupon is fixed. So if buyers in the market suddenly demand a *higher* return — a higher yield — the only way an old bond can deliver that higher return is for its **price to fall**, so the buyer pays less today for the same fixed future cash. And if buyers will accept a *lower* yield, they bid the old bond's price *up*. Price and yield sit on opposite ends of a seesaw. That inverse link is the *direction*. The rest of this post is about the *distance*.

One last term: **present value**. A dollar promised a year from now is worth less than a dollar in your hand today, because today's dollar can earn interest in the meantime. *Discounting* is the act of shrinking a future dollar back to its today-value. The further away the dollar, and the higher the interest rate you discount at, the smaller its present value. If that idea is new, the companion post [discounting cash flows: how a bond is priced](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced) builds it brick by brick; here we will use it as a tool.

### The one formula that does all the work

Everything in this post is downstream of a single equation — the bond pricing formula — so let's write it once, in plain symbols, and define every piece. The price of a bond is the present value of all its future cash flows:

$$P = \sum_{t=1}^{n} \frac{C}{(1 + y)^{t}} + \frac{F}{(1 + y)^{n}}$$

- $P$ is the price (the present value we are solving for).
- $C$ is the coupon paid each period (for our bond, \$20 per six-month period).
- $F$ is the face value, returned at maturity (\$1,000).
- $y$ is the yield *per period* — for a semiannual bond, half the annual yield, so a 5% annual yield means $y = 0.025$.
- $n$ is the number of periods until maturity (a 5-year semiannual bond has $n = 10$).
- $t$ is the period index, running from the first coupon ($t=1$) to maturity ($t=n$).

Read it left to right: take each coupon $C$, divide it by $(1+y)$ raised to the power of how many periods away it is — that is the discounting — and add up all of them, then add the present value of the face value $F$. The whole formula is just "shrink every future dollar back to today, then sum."

The single most important thing to notice is where $y$ lives: in the *denominator*, raised to a *power*. When $y$ goes up, every denominator gets bigger, so every term gets smaller, so $P$ falls. That is the seesaw, written in algebra. And because the exponent $t$ is largest for the most distant cash flows, a change in $y$ hits the far-future terms hardest — which, as we are about to see, is the entire reason the price move is bigger than the rate move. You will never have to compute this by hand (a spreadsheet or calculator does it instantly), but if you understand *why* $y$ sitting in a powered denominator makes distant cash flows so rate-sensitive, you understand bonds.

## From direction to magnitude: pricing the same bond at two rates

The cleanest way to feel the magnitude is to price the *same* bond twice — once at the old rate, once at the new rate — and subtract. That is exactly what the cover figure showed, and now we will do the arithmetic ourselves.

Our bond pays \$20 every six months for 5 years (that's 10 payments), plus \$1,000 at the end. To find its price at a given yield, we discount each of those 11 cash flows back to today and add them up. The discount rate per six-month period is half the annual yield. We do not need to grind every term by hand — the pattern is what matters — but seeing one full calculation makes the abstraction concrete.

#### Worked example: pricing our bond at 4% and again at 5%

You own the **5-year \$1,000 par bond with a 4% coupon** (\$20 semiannual). Today the market yield for bonds like this is exactly 4%, the same as the coupon. When the yield equals the coupon, the bond prices at exactly par:

> Price at 4% yield = \$20 discounted at 2% per period for 10 periods + \$1,000 discounted back 10 periods = **\$1,000.00.**

Now suppose the Fed hikes and the market yield for this bond jumps to 5%. Nothing about the bond changed — it still pays \$20, still returns \$1,000 — but buyers now demand a 5% return, so they discount harder (2.5% per period):

> Price at 5% yield = \$20 discounted at 2.5% per period for 10 periods + \$1,000 discounted back 10 periods = **\$956.24.**

The price fell from \$1,000 to \$956.24. That is a loss of **\$43.76**, or **−4.38%**. A 1% rise in the *rate* produced a 4.38% fall in the *price*.

*The lesson in one line: the rate moved 1%, but because every one of the bond's eleven future payments got repriced at once, the price moved more than four times as far.*

Why four times and not one? Because the rate change does not hit one cash flow — it hits *all* of them, and the more distant ones get hit hardest. The \$20 coupon due in six months barely cares whether it is discounted at 2% or 2.5%. But the \$1,000 due in five years is discounted ten times over, and each of those ten compoundings now happens at a stiffer rate. Stack those effects across every payment and you get a price change several times larger than the rate change. That multiplier has a name — **duration** — and it is the subject of [an entire later post](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income); for now, just hold the intuition that *the price move is the rate move, magnified by how far away the bond's money sits*.

### The move is not symmetric, and that is a gift

Look closely and you will notice something subtle. We saw +1% in yield cost the bond −4.38%. But what does −1% in yield *give* you? Re-price at 3% and the bond rises to \$1,046.11 — a gain of **+4.61%**. The downside (−4.38%) is *smaller* than the upside (+4.61%) for the same size rate move. The price–yield relationship is not a straight line; it is a gentle curve that bows toward the origin. That curvature works in the bondholder's favor: prices rise a little more when rates fall than they fall when rates rise. The property has a name, **convexity**, and it gets [its own post](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story). File it away; it is one of the quietly beautiful facts of fixed income.

#### Worked example: a one-basis-point move and the trader's unit

Traders rarely talk in whole percents. They talk in **basis points** (0.01%) and in dollars per bond. So let's shrink the move. If our \$1,000 bond's yield rises by a single basis point — from 4.00% to 4.01% — its price falls from \$1,000.00 to about **\$999.55**. That is a loss of roughly **\$0.45 per bond**, or 45 cents.

That number, the dollar price change for a one-basis-point move, is so useful it has a name: **DV01** ("dollar value of an 01"), sometimes called PV01. For our bond, DV01 ≈ **\$0.45**. If you held a \$10 million face position (10,000 bonds), a one-basis-point move would swing your value by about \$4,500; a 40-basis-point day would be roughly \$180,000. We give DV01 [its own treatment later](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk), but introduce it here because it is the most honest way to express "by how much": not as a vague percentage, but as a concrete dollar wobble per tick of rates.

*The lesson: scale the rate move down to one basis point and the price move becomes a clean, almost-linear dollar figure — the DV01 — which is how desks actually budget their risk.*

### Where the loss actually lives

It is worth slowing down to see *which* cash flows lose value when rates rise, because the answer is the key that unlocks both amplifiers in this post. Let's decompose our 5-year bond's −\$43.76 price drop, payment by payment, and watch where the damage concentrates.

#### Worked example: dissecting the −\$43.76 loss

Take the same 5-year \$1,000 bond, and list the present value of each of its eleven cash flows at a 4% yield and again at a 5% yield. The right-hand column is how much each flow's present value *changed* when the rate rose:

| Period | Cash flow | PV at 4% | PV at 5% | Change |
|---|---|---|---|---|
| 1 (6 mo) | \$20 | \$19.61 | \$19.51 | −\$0.10 |
| 2 (1 yr) | \$20 | \$19.22 | \$19.04 | −\$0.19 |
| 3 | \$20 | \$18.85 | \$18.57 | −\$0.27 |
| 5 (2.5 yr) | \$20 | \$18.11 | \$17.68 | −\$0.44 |
| 9 (4.5 yr) | \$20 | \$16.74 | \$16.01 | −\$0.72 |
| 10 (5 yr) | \$20 + \$1,000 | \$836.76 | \$796.82 | **−\$39.93** |
| **Total** | | **\$1,000.00** | **\$956.24** | **−\$43.76** |

Look at the bottom row. Of the total −\$43.76 loss, **−\$39.93 — over 90% of it — came from the final payment**, the coupon-plus-principal due in five years. The near-term coupons barely moved: the first one lost a dime. This is the whole story of interest-rate risk in one table. A bond's value is dominated by its most distant cash flow (here, the \$1,000 principal alone is **82%** of the price), and that distant flow is the one a rate change punishes hardest, because it gets discounted by $(1+y)$ raised to the highest power.

*The lesson: a rate move does its damage almost entirely to a bond's most distant cash flows, so anything that pushes a bond's money further out — longer maturity, smaller coupon — must amplify the price move.*

That single sentence is the bridge to the next two sections. Both amplifiers — maturity and coupon — work through exactly this channel: they change *how far away* the bond's money sits, and distance is what rate moves punish.

### A rate rise is not pure bad news

Before we leave the magnitude question, one honest caveat that beginners almost never hear. We have framed a rate rise as a loss, and for the *price* of a bond you already own, it is. But a rate rise has a *second* effect that pulls in the opposite direction: it lets you reinvest your coupons at the new, higher rate. The \$20 coupons our bond throws off can now be put to work at 5% instead of 4%. Over a long enough holding period, that extra reinvestment income partly — and eventually fully — offsets the price loss.

This is the **two faces of yield**: a rate rise hurts you through **price risk** (the bond you hold is worth less today) but helps you through **reinvestment risk turning into reinvestment *reward*** (your future coupons earn more). The two effects move in opposite directions, and there is a magic holding period — roughly equal to the bond's duration — at which they exactly cancel, so a buy-and-hold investor is left unharmed by the rate move. This is the deep idea behind *immunization*, the technique pensions and insurers use to lock in a return regardless of what rates do.

#### Worked example: the coupon you reinvest at the higher rate

You own our 5-year bond and rates jump from 4% to 5% the day after you buy it. Your *price* drops by \$43.76 immediately — that stings. But now every \$20 coupon you collect over the next five years gets reinvested at 5% instead of 4%. A rough way to see the offset: ten coupons of \$20, each earning an extra ~1% for the years they sit reinvested, claw back a meaningful chunk of that \$43.76 over the bond's life. If you *had* to sell tomorrow, the offset does you no good — you eat the full price loss. But if your horizon matches the bond's duration of ~4.5 years, the higher reinvestment income roughly cancels the price hit, and you end up about where you started.

*The lesson: a rate rise is an immediate price loss but a slow reinvestment gain, and at a holding period near the bond's duration the two cancel — which is why "rates rose, I lost money" is only true if you are forced to sell.*

The full machinery of price risk versus reinvestment risk, and the horizon where they net to zero, gets [its own post in the duration track](/blog/trading/fixed-income/reinvestment-risk-and-the-two-faces-of-yield). For now, just hold the nuance: the magnitude we have been computing is the *price* effect, which is what matters most for anyone who might sell, mark to market, or post collateral — but it is not the *whole* story for a true buy-and-hold investor.

## The amplifier, part one: maturity

We have established that the price move is bigger than the rate move. Now the central question of this post: *what controls the multiplier?* The first and biggest lever is **maturity** — how far in the future the bond's money sits.

The intuition is pure discounting. A rate change is a change in how harshly you shrink future dollars. A bond whose big payment (the \$1,000 principal) arrives in two years feels a rate change applied across only two years of compounding. A bond whose \$1,000 arrives in thirty years feels that same rate change applied across thirty years of compounding — and compounding is exponential, so thirty years of slightly-harsher discounting is a *much* bigger haircut than two years of it. Distance in time is leverage on rate moves.

![A bar chart showing the percent price drop for a one percent rate rise across maturities, with the two year bar shortest at minus one point nine percent and the thirty year bar longest at minus fifteen point four percent](/imgs/blogs/why-bond-prices-move-when-rates-move-and-by-how-much-2.png)

The chart above is the single most important picture in the post — it is the correlation between the rate move and the price move, sorted by maturity. Every bar is the *same* 4% coupon bond getting the *same* +1% rate shock; the only thing that changes is the maturity. Read it left to right and the amplifier reveals itself: a 2-year bond loses just **−1.9%**, a 5-year **−4.4%**, a 10-year **−7.8%**, a 20-year **−12.6%**, and a 30-year **−15.4%**. Same rate move, wildly different damage.

#### Worked example: the same shock across five maturities

Imagine five bonds, all 4% coupon, all priced at par when the market yield is 4%. Rates rise to 5% across the board. Here is the dollar and percent damage on each \$1,000 bond:

| Maturity | Price at 4% | Price at 5% | Dollar change | Percent change |
|---|---|---|---|---|
| 2-year | \$1,000.00 | \$981.19 | −\$18.81 | **−1.88%** |
| 5-year | \$1,000.00 | \$956.24 | −\$43.76 | **−4.38%** |
| 10-year | \$1,000.00 | \$922.05 | −\$77.95 | **−7.79%** |
| 20-year | \$1,000.00 | \$874.49 | −\$125.51 | **−12.55%** |
| 30-year | \$1,000.00 | \$845.46 | −\$154.54 | **−15.45%** |

Notice the move grows with maturity, but *not* in a straight line — going from 2 to 5 years roughly doubles the move, but going from 20 to 30 years adds far less per extra year. The amplifier rises with maturity but with diminishing returns, because the most distant cash flows are already so heavily discounted that pushing them further barely changes their (already tiny) present value.

*The lesson: maturity is the master dial on interest-rate risk — stretch a bond's money further into the future and you multiply how much its price reacts to every move in rates.*

This is the whole reason the financial press treats "the long bond" as a barometer of fear. A pension fund or insurer that owns 30-year Treasuries is making a leveraged bet on rates whether they think of it that way or not. When you read that "long-duration assets got crushed," this chart is the mechanism underneath the sentence. The macro lens on why the long end and short end of the curve respond to different forces lives in [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable); here we are just measuring the sensitivity itself.

## The amplifier, part two: the coupon

Maturity is the obvious lever. The second one surprises almost everyone the first time they meet it: for two bonds of the *same maturity*, the one with the **lower coupon** moves *more* for the same rate change. A small coupon makes a bond *more* rate-sensitive, not less.

Why? Because the coupon controls *when* you get your money back, on average. A bond with fat coupons hands you a lot of cash early — every coupon is a little chunk of your investment returned and no longer exposed to future rate moves. A bond with skinny coupons makes you wait: most of your money is locked up in the distant principal payment. And we just learned that distant money is exactly the money that gets repriced hardest when rates move. So the low-coupon bond, with its center of gravity pushed further out, has a longer effective wait — and a bigger price swing.

![A bar chart comparing two ten year bonds hit by the same one percent rate rise, with the two percent coupon bond falling eight point four percent and the eight percent coupon bond falling only seven percent](/imgs/blogs/why-bond-prices-move-when-rates-move-and-by-how-much-7.png)

The chart contrasts two bonds that mature on the *same* day, ten years out, hit by the *same* rise in yields from 4% to 5%. The 2% coupon bond drops **−8.4%**; the 8% coupon bond drops only **−7.0%**. Identical maturity, identical rate move — the low coupon takes the bigger hit, because it makes you wait longer for your cash.

#### Worked example: high coupon versus low coupon, same maturity

You are choosing between two 10-year \$1,000 bonds, both currently yielding 4%:

- **Bond L (low coupon, 2%)** pays \$10 semiannually. Priced at a 4% yield, it costs about \$836 (it sells at a discount because its coupon is below the market rate).
- **Bond H (high coupon, 8%)** pays \$40 semiannually. Priced at a 4% yield, it costs about \$1,327 (a premium, because its coupon beats the market).

Now yields rise to 5%. Re-price both:

- Bond L falls from ~\$836 to ~\$766 — a change of **−8.4%**.
- Bond H falls from ~\$1,327 to ~\$1,234 — a change of **−7.0%**.

The low-coupon bond lost a bigger *fraction* of its value. The reason is the timing of cash: Bond H has returned a big slice of your money in coupons by year five, so less of its value depends on the distant principal; Bond L keeps you waiting, so more of its value is exposed to the rate move.

*The lesson: a low coupon is not "safer" — it concentrates your money in the distant future, which is precisely where rate changes hurt most, so low-coupon bonds swing harder.*

This is why **zero-coupon bonds** — bonds with *no* coupon at all, which pay only the lump sum at maturity — are the most rate-sensitive instruments of their maturity in existence. A 30-year zero is essentially a pure bet on 30-year rates with maximum leverage, because 100% of its value sits in a single payment three decades away. The heavy mathematics of how coupon and maturity combine into one sensitivity number is the domain of [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics); the takeaway for now is that *two dials — maturity and coupon — together set the size of every price move*.

### The two dials combine into one number

Here is the elegant part. Maturity and coupon do not act as two separate, hard-to-track levers. They combine into a *single* number that captures a bond's rate sensitivity: its **duration**, measured in years. Loosely, duration is the *weighted-average time you wait to get your money back*, where each cash flow is weighted by its present value. A long-maturity, low-coupon bond has a long duration; a short-maturity, high-coupon bond has a short one. Our 5-year 4% bond has a duration of about **4.5 years** — slightly less than its 5-year maturity, because the coupons pull the average wait forward.

And duration is not just a description — it is a *prediction*. To a first approximation, the percentage price change for a small rate move is just:

$$\frac{\Delta P}{P} \approx -D \times \Delta y$$

where $D$ is the (modified) duration and $\Delta y$ is the change in yield. For our bond, $D \approx 4.5$, so a +1% (0.01) rate move predicts a price change of about −4.5% — strikingly close to the −4.38% we actually computed. The small gap between the −4.5% the formula predicts and the −4.38% reality is convexity, the curvature we met earlier, and it grows with the size of the move.

#### Worked example: predicting a price move from duration alone

Suppose someone tells you only two facts: a bond has a modified duration of 7 years, and rates just rose by 0.25% (25 basis points). Without knowing the coupon, the maturity, or anything else, you can estimate the price move:

> Price change ≈ −7 × 0.25% = **−1.75%.**

On a \$100,000 position that is about −\$1,750. That is the entire power of duration: it compresses everything in this post — the discounting, the maturity amplifier, the coupon amplifier — into one number you can multiply by the rate move to get the answer. Duration is so central that it earns [its own post as "the most important number in fixed income,"](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) and the dollar version, DV01, gets [a post of its own too](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk).

*The lesson: maturity and coupon collapse into a single number, duration, which you multiply by the rate move to predict the price move — the whole first half of this post in one multiplication.*

## The cash you actually pay: clean price, dirty price, and accrued interest

So far we have talked about "the price" as if a single number changes hands. It does not. The price you see quoted on a screen — the **clean price** — is deliberately *not* the cash you pay. The actual cash, the **dirty price** (also called the **invoice price**), includes an extra piece: **accrued interest**. Understanding this is not pedantry; it is the difference between the number you expect to wire and the number that actually leaves your account.

Here is the problem accrued interest solves. Coupons are paid on fixed dates — say, every January 15 and July 15. But bonds trade *every* day, including the days *between* coupons. Suppose you buy a bond on, say, day 121 of a 184-day coupon period. The seller held the bond for 121 of those days. They *earned* interest for those 121 days — but they will not be the holder of record on the coupon date, so they will not receive a penny of that coupon. The buyer (you) will collect the *entire* coupon a couple of months later, even though you only held the bond for the last 63 days of the period. That is unfair to the seller. **Accrued interest** is the fix: at settlement, the buyer reimburses the seller for the interest the seller earned but will not be paid. You pay it now; you get it back inside the full coupon later.

![A timeline-style chart showing accrued interest rising in a straight line from zero dollars at the last coupon to twenty dollars at the next coupon, with a marker at day one hundred twenty one showing thirteen dollars and fifteen cents](/imgs/blogs/why-bond-prices-move-when-rates-move-and-by-how-much-4.png)

The chart shows accrued interest filling up like a meter. At the moment a coupon is paid, accrued resets to \$0. Then it climbs in a straight line as days pass, reaching the full coupon amount (\$20 for our semiannual bond) just before the next payment. The slope is the daily interest you are racking up. The marker shows our example: 121 days into a 184-day period, accrued interest has climbed to **\$13.15**.

So the dirty price — the cash that actually settles — is:

$$\text{Dirty price} = \text{Clean price} + \text{Accrued interest}$$

where the clean price is the quoted, market-driven number (the one that moves with rates, the star of the first half of this post), and accrued interest is a mechanical, predictable add-on that depends only on the calendar.

#### Worked example: what you actually wire for our bond

You buy our \$1,000 par bond. The screen quotes a clean price of **95.624**, which means 95.624% of par, or **\$956.24** (the same price we computed at a 5% yield — let's say that is where the bond trades today). You settle on **day 121 of a 184-day** coupon period.

1. **Clean price:** 95.624% × \$1,000 = **\$956.24** — the quoted number.
2. **Accrued interest:** the semiannual coupon is \$20. You owe the seller the fraction of it they earned: \$20 × (121 ÷ 184) = **\$13.15**.
3. **Dirty (invoice) price:** \$956.24 + \$13.15 = **\$969.39** — the cash you actually wire per bond.

A couple of months later, on the coupon date, you collect the *full* \$40 annual coupon (or, on this semiannual schedule, the \$20 for the current period). Of that \$20, **\$13.15** simply repays you for the accrued interest you fronted at settlement, and the remaining **\$6.85** is the interest you genuinely earned for the days you held it. (Across the year, you net the \$40 you were entitled to.) Nothing was lost; the accrued interest was a timing adjustment, not a fee.

*The lesson: the quoted clean price is the negotiable, rate-driven number, but the cash that leaves your account is the dirty price — clean plus the calendar-driven accrued interest you owe the seller.*

![A pipeline showing four steps from the quoted clean price of nine hundred fifty six dollars to adding thirteen dollars of accrued interest to a dirty invoice price of nine hundred sixty nine dollars to collecting the next coupon](/imgs/blogs/why-bond-prices-move-when-rates-move-and-by-how-much-6.png)

The pipeline above walks the settlement end to end: start with the quoted clean price, add the accrued interest the calendar says you owe, sum to the dirty price you actually pay, and then watch the next coupon repay the accrued slice. Read it once and the clean/dirty distinction stops being a vocabulary quiz and becomes a cash-flow you can trace.

### The sawtooth: why the invoice jumps on coupon day

Because accrued interest climbs steadily and then resets to zero whenever a coupon is paid, the dirty price traces a distinctive shape over time: a slow climb, a sudden drop, a slow climb, a sudden drop. A sawtooth. Meanwhile the clean price — the number on the screen — does *not* jump on coupon day; it drifts only as the market's view of rates changes. Beginners who watch the dirty price and expect it to behave like a stock are baffled by the coupon-day cliff. There is nothing mysterious about it: the bond just paid out cash, so the next buyer no longer owes for that coupon, and the invoice resets.

![A chart showing the dirty price climbing then dropping by twenty dollars on each coupon date in a sawtooth pattern while the clean price stays a roughly flat dashed line](/imgs/blogs/why-bond-prices-move-when-rates-move-and-by-how-much-3.png)

The figure makes the two prices visible side by side. The solid line — the dirty price — ramps up by \$20 over each coupon period and then drops by \$20 the instant the coupon is paid. The dashed line — the clean price — stays roughly level, drifting only with rates. The vertical distance between them at any moment *is* the accrued interest. This is why bonds are quoted clean: stripping out the predictable, calendar-driven sawtooth lets traders compare bonds on the part that actually reflects market conditions.

#### Worked example: reading the sawtooth on coupon day

Suppose, with no change in rates, our bond's clean price holds steady at \$956.24 for a whole coupon period. Watch the dirty price:

- **Day 0 (just after a coupon):** accrued = \$0, dirty = \$956.24.
- **Day 92 (halfway):** accrued = \$20 × (92 ÷ 184) = \$10.00, dirty = \$966.24.
- **Day 183 (just before the next coupon):** accrued ≈ \$20 × (183 ÷ 184) = \$19.89, dirty ≈ \$976.13.
- **Day 184 (coupon paid):** the \$20 coupon is paid out; accrued resets to \$0; dirty drops straight back to \$956.24.

The dirty price climbed nearly \$20 and then fell \$20 in a single day — yet the *quoted* clean price never moved, and you never gained or lost a cent from the sawtooth. The "drop" is just the bond returning the cash it had been accumulating.

*The lesson: the dirty price's coupon-day cliff is an accounting artifact of accrued interest resetting — the clean price is the number that actually carries the market's verdict on the bond.*

## Counting the days: the conventions that decide your invoice

We have been blithely writing "121 days out of 184." But who decides it is 184? And why not 182, or 180? This is where bonds get gloriously, maddeningly specific. The number of days in the period and the number of days you have held are both set by a **day-count convention** — a rulebook for counting calendar time. Different markets use different rulebooks, and they give *different dollar answers on the same trade*. If you have ever wondered why two brokers quote slightly different accrued interest on the identical bond, this is why.

There are three you must know:

- **30/360** ("thirty over three-sixty"): pretend every month has exactly 30 days and every year has exactly 360. It is the banker's idealized calendar — clean, simple, slightly fictional. A coupon period is treated as 180 days. Used for most US **corporate** and **municipal** bonds, and many agency bonds.
- **Actual/actual** ("act/act"): count the *real* calendar days you held, over the *real* number of days in the actual coupon period. No fiction. This is the **US Treasury** standard — notes, bonds, and TIPS all use it. It is the most precise and the most honest.
- **Actual/360** ("act/360"): count real elapsed days, but divide by a 360-day year. Because a real year has 365 days but you divide by 360, this convention quietly pays slightly *more* interest than a strict calendar would — about 365/360 ≈ 1.4% more per year. It is the **money-market** standard: Treasury bills, commercial paper, and most floating-rate notes.

![A comparison matrix of three day-count conventions showing how each counts the period, where it is used, and the different accrued interest each produces on the same trade](/imgs/blogs/why-bond-prices-move-when-rates-move-and-by-how-much-5.png)

The matrix lays the three side by side. Notice the punchline in the right-hand column: on the *same* trade, the three conventions produce **\$13.33, \$13.15, and \$13.44** of accrued interest. The differences are small in dollars but they are real, they compound across a large portfolio, and — critically — they are *not* a matter of opinion. Each is the correct answer *under its own rulebook*; the only error is applying the wrong rulebook to a given bond.

#### Worked example: the same trade under three rulebooks

You buy our \$1,000 par, 4% coupon bond (\$20 per semiannual period) and you are 121 actual calendar days into the period since the last coupon. How much accrued interest do you owe under each convention?

- **30/360:** the convention says a coupon period is 180 days, and (by its month-counting rules) you have held it for 120 of them. Accrued = \$20 × (120 ÷ 180) = **\$13.33**.
- **Actual/actual:** the real period is 184 days and you have held 121 of them. Accrued = \$20 × (121 ÷ 184) = **\$13.15**.
- **Actual/360:** count the 121 real days, but apply the money-market year. Accrued = \$1,000 × 4% × (121 ÷ 360) = **\$13.44**.

Same bond, same day, three different invoices. The gaps are pennies on a single \$1,000 bond — but on a \$50 million position they are real money, and on the wrong side of a settlement dispute they are a phone call you do not want to make.

*The lesson: day-count is not a detail — it is the rulebook that turns "how long have I held this" into a precise dollar figure, and using the wrong one mis-states the cash on every trade.*

#### Worked example: why act/360 quietly pays more

Here is the subtle one. A money-market instrument quoting a 4% rate on act/360 does *not* actually pay you 4% over a real year. Because you accrue real days but divide by a 360-day year, a full 365-day year earns you 4% × (365 ÷ 360) = **4.056%** in true annual terms.

Put concretely: park \$1,000,000 in a 4% act/360 instrument for one real year and you collect \$1,000,000 × 4% × (365 ÷ 360) = **\$40,556**, not \$40,000. The extra \$556 is not generosity — it is baked into the convention. This is why you cannot directly compare a money-market rate (act/360) to a bond yield (act/act or 30/360) without converting; the day-count is doing silent work on the number.

*The lesson: act/360 inflates the effective rate by roughly 1.4% of itself, so a "4%" money-market rate is really closer to 4.06% — always check the day-count before comparing yields.*

#### Worked example: counting the days under 30/360

The 30/360 convention has its own little arithmetic that trips people up, so let's count one period by its rules. Say a corporate bond last paid a coupon on **January 15** and you settle on **May 15**. Under 30/360, you count the days as if every month were exactly 30 days:

> Days from Jan 15 to May 15 = (number of whole months) × 30 + (difference in day-of-month). From January to May is 4 months, and the day-of-month is the same (15 to 15), so the count is 4 × 30 + 0 = **120 days**.

The full coupon period under 30/360 is treated as **180 days** (a half-year). So the accrued interest is \$20 × (120 ÷ 180) = **\$13.33** — exactly the 30/360 figure in the matrix above.

The convention also has quirky edge rules for month-ends: if the start date is the 31st it is bumped to the 30th, and if the end date is the 31st (and the start was the 30th or 31st) it too is bumped to the 30th. These tiny rules exist so that 30/360 never produces a "31-day month," keeping every month a clean 30. They rarely change the answer by more than a day's interest, but on a settlement desk processing thousands of trades, "rarely" is not "never," and the rulebook has to be exact.

*The lesson: 30/360 trades real-calendar precision for arithmetic simplicity — every month is 30 days, every year is 360 — which is why its day counts can differ from the actual calendar even when the dates look identical.*

## Putting the two halves together: a full trade, start to finish

Let's stitch the rate-driven half and the calendar-driven half into one transaction, because in real life they happen at the same time.

#### Worked example: a complete settlement after a rate move

In January, our \$1,000 par, 4% coupon bond trades at par — clean price \$1,000.00 — because the market yield is 4%. You consider buying but wait.

By spring, the Fed has hiked and the market yield for this bond has risen to 5%. Two things have now happened to your eventual invoice:

1. **The clean price fell.** At a 5% yield, the bond's clean price is now **\$956.24** — a drop of \$43.76, or −4.38%, exactly the rate-driven move we computed at the start. This is the part of the price that *moved with rates*.
2. **Accrued interest is on the meter.** You settle 121 days into the current 184-day coupon period (a Treasury, so actual/actual). Accrued = \$20 × (121 ÷ 184) = **\$13.15**.

Your dirty (invoice) price is therefore \$956.24 + \$13.15 = **\$969.39** per bond. If you buy 100 bonds (\$100,000 face), you wire **\$96,939**.

Notice how the two effects live in different worlds. The −4.38% is *market risk* — it reflects the rate change and would have moved against you (or for you) regardless of the calendar. The +\$13.15 is *plumbing* — it is fully predictable from the date and the day-count, and you get it back in the next coupon. A beginner who lumps them together sees a confusing single number; a practitioner sees a rate bet plus a calendar adjustment, and tracks each separately.

*The lesson: every bond purchase is two things at once — a rate-sensitive clean price that carries your market risk, and a mechanical accrued-interest add-on that is pure timing — and keeping them mentally separate is half of understanding fixed income.*

## Common misconceptions

**"A 1% rise in rates means a 1% loss on my bond."** This is the single most common and most expensive misunderstanding in fixed income. The rate move and the price move are *not* the same size. For our 5-year bond the multiplier was about 4.4×; for a 30-year bond it is closer to 15×. The whole concept of *duration* exists precisely to measure this multiplier. Treating a 1% rate move as a 1% price move will understate your risk by a factor of four to fifteen.

**"Low-coupon and zero-coupon bonds are the safe, conservative choice."** The opposite is true for *interest-rate* risk. A low coupon pushes your money into the distant future, which is exactly where rate changes bite hardest, making the bond *more* price-volatile, not less. A 30-year zero-coupon Treasury is one of the most rate-sensitive instruments in the market. (Low-coupon bonds can be lower in *credit* risk if they're Treasuries, but that is a different axis entirely — covered in the [credit-risk track](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back).)

**"The price on my screen is what I'll pay."** No — the screen shows the *clean* price. The cash that settles is the *dirty* price, which adds accrued interest. On a bond bought mid-period, the invoice can be \$10–\$20 per bond higher than the quote. You are not being overcharged; you are reimbursing the seller for interest they earned, and you get it back in the next coupon.

**"Accrued interest is a fee or a cost."** It is neither. It is a timing transfer. You pay the seller for the days they held the bond, and the very next coupon pays you back the same amount. Over a full holding period it nets to exactly the interest you were entitled to. The only thing it costs you is a little cash up front.

**"Day-count is trivial bookkeeping that doesn't matter."** On one \$1,000 bond, the difference between conventions is pennies. On an institutional portfolio it is real money, and applying the *wrong* convention to a bond — say, pricing a corporate (30/360) as if it were a Treasury (act/act) — produces a flatly incorrect invoice. Worse, comparing a money-market rate (act/360) to a bond yield (act/act) without converting silently overstates the money-market rate by about 1.4% of itself. The conventions are small but they are not optional.

**"Prices fall as much as they rise for the same rate move."** Almost, but not quite. The price–yield curve bends, so a bond gains slightly *more* when rates fall than it loses when rates rise by the same amount — the convexity gift we saw (−4.38% for +1% versus +4.61% for −1%). For small moves the asymmetry is tiny; for large moves and long bonds it becomes a meaningful edge in the bondholder's favor.

## How it shows up in real markets

**The 2022 bond rout — the textbook punished in public.** In 2022 the Federal Reserve raised its policy rate from near 0% to over 4% in a single year, the fastest hiking cycle in four decades. Watch the mechanism from this post play out at scale: the rate moved a few percent, but *prices* moved far more, and the longest bonds moved most of all. The Bloomberg US Aggregate bond index — supposedly the "safe" part of a portfolio — fell roughly 13% on the year, its worst calendar return in modern history. Long-dated Treasury indices fell on the order of 25–30%. The reason is exactly the maturity amplifier: index durations of six, fifteen, even twenty-plus years turned a several-percent rate move into a double-digit price collapse. Anyone who believed "bonds are safe, a 1% rate move is a 1% loss" learned the multiplier the hard way. The cross-asset version of this episode — where stocks *and* bonds fell together — is dissected in [the 2022 case study](/blog/trading/cross-asset/case-study-2022-stocks-and-bonds-both-fell).

**The 1994 "bond massacre" — when the mechanism first taught Wall Street a lesson.** Long before 2022, the Federal Reserve under Alan Greenspan surprised markets in 1994 by roughly doubling the policy rate from 3% to 6% over the year. The bond market, positioned for calm, was hammered: the move wiped out an estimated \$1.5 trillion of global bond value, and it exposed everyone who had quietly stacked on duration or leverage. Orange County, California — a municipality, of all things — had borrowed heavily to buy interest-rate-sensitive securities, effectively a leveraged bet that rates would stay low. When rates jumped, the maturity-and-leverage amplifier turned a few-percent rate move into catastrophic losses, and the county filed the largest municipal bankruptcy in US history at the time. The 1994 episode is the original proof that the abstract mechanism in this post is anything but abstract: misjudge the *magnitude* of a rate move and the multiplier can end careers and bankrupt governments.

**Silicon Valley Bank, 2023 — duration risk hiding on a balance sheet.** SVB had taken billions of dollars of deposits and parked them in long-dated, low-coupon Treasuries and mortgage bonds bought when rates were near zero. On paper these were the *safest possible* credits — US government guaranteed, no default risk. But they carried enormous *interest-rate* risk: long maturity plus low coupon is the maximum-amplifier combination from this very post. When the Fed hiked in 2022–23, the market value of those bonds fell so far that the unrealized losses exceeded the bank's equity. When depositors got nervous and pulled cash, SVB had to sell those bonds at the depressed prices, crystallizing the loss, and the bank collapsed in days. The lesson is brutal and exact: "no credit risk" did not mean "no risk," because duration was the risk all along. The full anatomy is in [the SVB and Credit Suisse case study](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).

**The 30-year Treasury and the long bond's reputation.** Market commentators obsess over the 30-year Treasury yield not because many ordinary investors own it, but because it is the most rate-sensitive liquid instrument the government issues — the bar at the far right of our maturity chart. A move of even 20–30 basis points in the 30-year yield translates into a several-percent swing in its price, which is why hedge funds and pension liability managers use it as both a barometer and a leverage tool. When you hear that "the long end sold off hard," the maturity amplifier is the entire content of the sentence.

**Treasury bills and the act/360 quirk in money-market funds.** When short-term rates rose to around 5% in 2023, money-market funds — which hold T-bills and commercial paper quoted on act/360 — were advertising yields that looked simple but hid the day-count subtlety. A "5%" act/360 instrument actually delivers about 5.07% in true annual terms over a 365-day year. Across the trillions of dollars in money-market funds, that 1.4%-of-itself uplift is a genuine, if quiet, edge — and a trap for anyone naively comparing a money-fund quote to a bond's actual/actual yield without converting first.

**The 2020 dash-for-cash and accrued-interest plumbing under stress.** In March 2020, as the pandemic hit, even the Treasury market briefly seized up. In normal times, the clean/dirty distinction and day-count conventions are invisible plumbing; in a liquidity crisis, settlement mechanics suddenly matter, because dealers must price not just the clean market value but the exact accrued interest on millions of trades clearing at once. The episode is a reminder that the "boring" settlement machinery in this post is what lets the world's largest market function at all — and when it strains, central banks intervene to keep the plumbing flowing.

## When this matters to you, and where to go next

You feel this mechanism whether or not you own a single bond. The 10-year Treasury yield — set by exactly the price–yield seesaw in this post — is the reference rate that prices your **mortgage**, your **car loan**, and the discount rate used to value the **stocks** in your retirement account. When that yield moves, the price of nearly everything moves with it, by an amount governed by the same "how far away is the money" logic we just made precise. The journey from the 10-year yield to your mortgage rate gets [its own post](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) in the macro track.

If you take one thing from this post, make it the multiplier. "Rates moved 1%" is only the input. The output — the price move — is that 1% magnified by maturity and lowered-coupon, and it can be four, ten, or fifteen times larger. That multiplier has a name, **duration**, and it is the single most important number in fixed income — which is why the [next track](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) is devoted entirely to measuring it, turning it into dollars with [DV01](/blog/trading/fixed-income/modified-duration-and-dv01-measuring-and-trading-rate-risk), and correcting it for the curvature we called [convexity](/blog/trading/fixed-income/convexity-why-duration-is-not-the-whole-story). For the full derivation behind every formula here, the quantitative companion is [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics). Once you can predict the size of a price move before it happens, you have stopped merely reading bond headlines and started understanding them.
