---
title: "Duration: the most important number in fixed income"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "What duration actually measures, why it is the single number that tells you how much a bond will move when rates move, and how to compute it from scratch on a $1,000 bond."
tags: ["fixed-income", "bonds", "duration", "macaulay-duration", "modified-duration", "interest-rate-risk", "convexity", "yield-to-maturity", "treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **The one idea:** duration is the single number that tells you how much a bond's price will move when interest rates move. It is the "center of gravity" of a bond's cash flows — and, almost magically, the same number is also the percentage price change for a 1% change in yield.
> - **Macaulay duration** is the weighted-average time, in years, until you get your money back — each cash flow weighted by how much of the bond's value it represents.
> - **Modified duration** turns that into a price-sensitivity rule: a bond with a modified duration of 5 falls about 5% if yields rise 1%, and rises about 5% if yields fall 1%.
> - Duration is essentially the **slope of the price–yield curve**: it is the bond's sensitivity to rates, expressed as a clean percentage.
> - A **zero-coupon bond's duration equals its maturity** (one payment, all the weight at the end); a **coupon bond's duration is always less** than its maturity, because some cash arrives early.
> - Higher coupon, lower maturity, and higher yields all **shorten** duration. Duration is *the* summary statistic of interest-rate risk — the one number a bond manager checks first.

Suppose you are handed two bonds and told that interest rates are about to rise by one percentage point. One bond will barely flinch. The other will lose a fifth of its value. They might have the same issuer, the same credit rating, even similar yields. So what is the one number that separates them — the number that tells you, *before* anything happens, how much each bond will move?

That number is **duration**. It is the most important single statistic in all of fixed income, and it is the number that bond managers, central bankers, and risk officers check first, before anything else about a bond. If you understand duration, you understand interest-rate risk. If you don't, every bond is a black box that moves for reasons you can't predict.

Here is the surprising part. Duration starts life as something completely intuitive — *the average time you wait to get your money back* — and it ends up, through a small piece of arithmetic, as something completely practical: *the percentage your bond's price moves for each 1% move in rates*. Those two ideas sound unrelated. The whole point of this post is to show you that they are the same number, and to make both of them obvious.

![A balance beam showing a bond's cash flows placed along a timeline, with the fulcrum sitting at the duration point where the beam balances](/imgs/blogs/duration-the-most-important-number-in-fixed-income-1.png)

The diagram above is the mental model to carry through the whole post: a bond's cash flows are weights placed along a timeline, and duration is the *balance point* — the spot where the beam would tip neither forward nor back. A 5-year bond that pays coupons along the way has its balance point a little before year 5, because some of the weight (the coupons) sits earlier on the beam. A 5-year bond with no coupons has all its weight at the very end, so its balance point *is* year 5. That single picture — where the weight sits — is duration, and everything else is a way of making it precise.

We will build it from zero on one running example: a **\$1,000 par, 4% coupon, 5-year bond** (its duration turns out to be about 4.63 years) compared against a **5-year zero-coupon bond** (duration exactly 5). By the end you will be able to compute duration by hand, read it off a price–yield curve, and use it to predict — to within a fraction of a percent — what a rate move does to a bond, a portfolio, or a bond fund you might own. If you want the macro view of why "the price of money" matters for the whole economy, the sibling post [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) takes that angle; here we stay inside the bond.

## Foundations: the words you need before duration makes sense

Duration sits on top of a few ideas. None are hard, but we have to be precise about each before the magic works. If you have read the earlier posts in this series, this is review; if not, here is everything you need.

A **bond** is a tradable loan. You hand the issuer money today; in return they promise a fixed stream of payments and the return of your principal at the end. The **par value** (or **face value**, or **principal**) is the amount repaid at maturity — our standard is **\$1,000**. The **coupon** is the fixed annual interest payment, quoted as a percentage of par: a **4% coupon** on a \$1,000 bond pays **\$40 a year**. The **maturity** is when the principal comes back — five years, in our running example.

The **yield** (specifically the **yield to maturity**, or **YTM**) is the single discount rate that makes the present value of all those future payments equal to the bond's price today. It is the bond's internal rate of return if you buy it and hold it to the end. When we say "rates rise 1%", we mean the bond's yield rises by one percentage point — from 4% to 5%, say. (One **basis point**, written *bp* or *bps*, is one hundredth of a percent — 0.01% — so 1% is 100 bps. Bond people quote yield moves in basis points the way other people quote them in percent.)

The **present value** of a future dollar is what that dollar is worth today, once you discount it for the time you have to wait. A dollar arriving in one year, discounted at 4%, is worth `1 / 1.04 = $0.9615` today. A dollar arriving in five years is worth `1 / 1.04⁵ = $0.8219`. The further out the cash, the less it is worth now — and, crucially, the *more sensitive* its present value is to the discount rate. That last sentence is the seed of the entire idea of duration. Hold onto it.

Finally, the **price** of a bond is simply the sum of the present values of every cash flow it pays. For our 4% bond at a 4% yield:

$$
P = \frac{40}{1.04} + \frac{40}{1.04^2} + \frac{40}{1.04^3} + \frac{40}{1.04^4} + \frac{1040}{1.04^5} = \$1{,}000
$$

Here $P$ is the price, the numerators are the cash flows (four \$40 coupons, then a final \$40 coupon plus the \$1,000 principal), and $1.04$ is one plus the 4% yield. When the coupon equals the yield, the price equals par — \$1,000. We built the seesaw of price and yield in the sibling post [price and yield: the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds); duration is the number that tells you *how steep* that seesaw is. With those terms in hand, we can finally ask the real question: how much does this price move when the yield moves?

## The problem duration solves: "by how much?"

Everyone learns the direction first: rates up, bond prices down. That is the easy half, and it is covered in [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much). The hard half — the half that actually matters for managing money — is *by how much*. A 1% rate move is not a 1% price move. On a short bond it might be a fraction of a percent; on a 30-year Treasury it can be 18-20%. The magnitude depends on the bond.

You could, of course, answer "by how much" the brute-force way: re-price the bond at the new yield and compare. That works, but it is slow, it has to be redone for every bond and every rate scenario, and it gives you a number for *one* move rather than a feel for the bond's *sensitivity*. What we want is a single statistic, computed once, that says: "this bond moves about X% for every 1% move in rates." That statistic is duration. It is the bond's price sensitivity, distilled to one number — and once you have it, you can predict price moves in your head.

To see why one number is worth so much, look at how price actually responds across a range of yields. Here is our \$1,000, 4% coupon, 5-year bond re-priced at several yields, alongside the move predicted by its duration:

| Yield | Actual price | Change from par | Duration estimate |
|---|---|---|---|
| 2% | \$1,094.27 | +9.43% | +8.90% |
| 3% | \$1,045.80 | +4.58% | +4.45% |
| 4% | \$1,000.00 | 0% | 0% |
| 5% | \$956.71 | −4.33% | −4.45% |
| 6% | \$915.75 | −8.42% | −8.90% |

Two things jump out. First, the duration estimate (a straight line through the 4% point with slope 4.45) tracks the actual price beautifully for small moves and drifts only a little even at ±2%. Second, the *actual* prices are always a touch higher than the duration line predicts — up moves are bigger than the estimate, down moves are smaller. That asymmetry is convexity, the curvature we'll return to. But the headline is that one number, 4.45, reproduced the entire column of price moves with a single multiplication. That is the leverage duration gives you: compute it once, and you can answer "by how much?" for any rate scenario without re-pricing anything.

Two flavors of that statistic exist, and they are two views of the same thing:

- **Macaulay duration** answers it in the language of *time*: the weighted-average number of years you wait to receive the bond's cash, with each year weighted by how much of the bond's value arrives then. Named after Frederick Macaulay, who defined it in 1938.
- **Modified duration** answers it in the language of *price*: the percentage the price changes for a 1% change in yield. It is Macaulay duration divided by `(1 + yield)`, and it is the number you actually use to manage risk.

They are tied together by one line of calculus we will get to. First, let's make the *time* version concrete, because that is the one your intuition can grab.

## Macaulay duration: the center of gravity of the cash flows

Picture the bond's cash flows as physical weights sitting on a seesaw, one at each payment date along a timeline. At year 1 sits a small weight (the present value of the first \$40 coupon). At year 2, a slightly smaller one. And at year 5 sits a huge weight — the present value of the final \$40 coupon *plus* the \$1,000 principal, which dwarfs everything else.

Now ask: where is the **balance point** of that seesaw — the single spot where, if you put a fulcrum under it, the beam would balance? That balance point is the **Macaulay duration**. It is, literally, the center of gravity of the cash flows along the time axis. The far weight at year 5 pulls the balance point out toward 5; the small early coupons pull it back toward 0. Where it settles — about 4.63 years for our bond — is the *average time you wait to get your money*, where "average" weights each date by how much value lands there.

Mathematically, Macaulay duration $D_{\text{mac}}$ is:

$$
D_{\text{mac}} = \sum_{t} t \cdot \frac{PV_t}{P}
$$

where $t$ is the time (in years) of each cash flow, $PV_t$ is the present value of the cash flow at time $t$, $P$ is the bond's total price (the sum of all the $PV_t$), and the fraction $PV_t / P$ is the **weight** — the share of the bond's value that arrives at time $t$. The weights add up to 1, exactly like the weights in any weighted average. You multiply each payment date by its weight and add them up. That's it.

![A horizontal bar chart of the present-value weight of each cash flow, tiny bars for the early coupons and one dominant bar at year five, summing to the duration](/imgs/blogs/duration-the-most-important-number-in-fixed-income-5.png)

The chart above shows where the weight actually sits. The first four coupons together carry only about 15% of the bond's value; the final payment carries 85%. That is *why* the balance point lands at 4.63 and not, say, 3 — the enormous final weight drags the average far out toward maturity. The early coupons barely move it.

It is worth pausing on *why* the weights are what they are, because the intuition pays off everywhere. The final payment is huge for two reasons: it bundles the last coupon with the entire \$1,000 principal, and even after discounting five years it is still by far the largest single cash flow. The early coupons are small in two ways at once — they are small in face amount (\$40 versus \$1,040) and they are discounted the least, so they contribute little to the *total* value and therefore earn little *weight*. This is the deep reason a coupon bond's duration sits so close to its maturity: the principal repayment, sitting at the very end, dominates the weighting and anchors the balance point near the final date. Cut the coupon to zero and the anchor becomes the *only* weight, and duration snaps exactly to maturity. Crank the coupon way up and you spread meaningful weight across the early years, prying the balance point back toward today. Everything about how duration behaves follows from this one tug-of-war between the dominant final principal and the early coupons.

There is also a subtle point hiding in the weights that explains *all* of duration's behavior with yield. Far-off cash flows are discounted by a higher power of `(1 + y)`, so when the yield changes, their present value changes proportionally *more* than near-term cash. A dollar at year 1 loses about 1% of its value if the yield rises 1%; a dollar at year 10 loses about 10%. Duration is, in effect, the *average maturity of the value* precisely because the sensitivity of each cash flow to yield is proportional to how far away it is. The "average time" interpretation and the "price sensitivity" interpretation are the same fact wearing two outfits — and the bridge between them is just dividing by `(1 + y)`, which we do next.

#### Worked example: Macaulay duration of the 4% bond by hand

Let's compute it cash flow by cash flow, with the bond yielding 4% (so it is priced at par, \$1,000). For each year we find the present value of that year's cash, divide by the \$1,000 price to get the weight, and multiply by the year.

- **Year 1:** cash = \$40. PV = `40 / 1.04 = $38.46`. Weight = `38.46 / 1000 = 0.0385`. Contribution = `1 × 0.0385 = 0.0385`.
- **Year 2:** cash = \$40. PV = `40 / 1.04² = $36.98`. Weight = `0.0370`. Contribution = `2 × 0.0370 = 0.0740`.
- **Year 3:** cash = \$40. PV = `40 / 1.04³ = $35.56`. Weight = `0.0356`. Contribution = `3 × 0.0356 = 0.1067`.
- **Year 4:** cash = \$40. PV = `40 / 1.04⁴ = $34.19`. Weight = `0.0342`. Contribution = `4 × 0.0342 = 0.1368`.
- **Year 5:** cash = \$1,040 (last coupon + principal). PV = `1040 / 1.04⁵ = $854.80`. Weight = `0.8548`. Contribution = `5 × 0.8548 = 4.2740`.

Add the contributions: `0.0385 + 0.0740 + 0.1067 + 0.1368 + 4.2740 = 4.63` years. *The 4% bond's Macaulay duration is 4.63 years — you "effectively" wait 4.63 years to get your money, even though the bond runs 5, because a sliver of it comes back early as coupons.*

![A table laying out each year, its cash flow, present value, value weight, and the year times the weight, with the final column summing to the bond's duration of about four point six three years](/imgs/blogs/duration-the-most-important-number-in-fixed-income-7.png)

The table above is the entire calculation in one view: each row is a year, the columns walk from the cash flow to its present value to its weight to its contribution, and the final column sums to the duration. It is worth memorizing the *shape* of this table rather than the numbers — every Macaulay duration anywhere is this same five-column march, whether the bond runs 2 years or 30.

#### Worked example: the zero-coupon bond's duration equals its maturity

Now take a **5-year zero-coupon bond**: it pays no coupons, just \$1,000 at year 5. There is exactly one cash flow, so there is exactly one weight, and that weight is 1.0 (all of the value arrives at year 5). The Macaulay duration is:

$$
D_{\text{mac}} = 5 \times 1.0 = 5 \text{ years}
$$

*A zero's duration always equals its maturity, because there is no early cash to pull the balance point back — all the weight sits at the end.* This is the cleanest fact in the whole subject, and it is the anchor for everything else: a coupon bond is a zero with some weight shaved off the end and sprinkled earlier, so its duration is *always less than its maturity*. Our 4% bond (duration 4.63) sits below its 5-year maturity for exactly that reason. The zero (duration 5) sits right at it.

### Computing duration in practice

You will rarely compute duration by hand more than once — every spreadsheet and bond library does it for you — but seeing the whole calculation in a few lines of code removes any remaining mystery. Here is the entire thing for our running bond, runnable as-is:

```python
import numpy as np

par, coupon_rate, ytm, years = 1000.0, 0.04, 0.04, 5
times = np.arange(1, years + 1)                       # years 1..5
cashflows = np.full(years, coupon_rate * par)         # $40 coupons
cashflows[-1] += par                                  # add $1,000 principal at year 5

pv = cashflows / (1 + ytm) ** times                   # present value of each flow
price = pv.sum()                                       # bond price
weights = pv / price                                   # share of value at each date

macaulay = (times * weights).sum()                     # weighted-average time
modified = macaulay / (1 + ytm)                        # price sensitivity

print(f"price     = ${price:,.2f}")                    # 1,000.00
print(f"macaulay  = {macaulay:.4f} years")             # 4.6299
print(f"modified  = {modified:.4f}")                   # 4.4518
print(f"+1% move  = {-modified * 0.01 * 100:.2f}%")    # -4.45%
```

That is the complete machine: cash flows, present values, weights, a weighted average, and a division by `(1 + y)`. Every duration number in this post — and every duration on every bond fund fact sheet — comes out of those eight lines. The only refinements the real world adds are semiannual periods, day-count conventions, and, for bonds with options, re-pricing up and down instead of using the closed-form weights. The skeleton never changes.

## Modified duration: turning years into a price-change rule

The center-of-gravity story is satisfying, but a balance point measured in *years* doesn't directly tell you what happens to your *money* when rates move. The bridge between the two is a small piece of calculus, and it produces the number you will actually use every day: **modified duration**.

Recall that the price is a sum of cash flows discounted by $(1+y)^t$. If you take the derivative of price with respect to yield — that is, ask "how fast does price change as yield changes?" — the algebra works out to:

$$
\frac{1}{P}\frac{dP}{dy} = -\frac{D_{\text{mac}}}{1+y} = -D_{\text{mod}}
$$

In words: the **percentage** change in price for a small change in yield equals minus the Macaulay duration divided by `(1 + yield)`. We give that quantity its own name, **modified duration**, $D_{\text{mod}} = D_{\text{mac}} / (1+y)$. The minus sign is the seesaw: yields up, price down. The size — the modified duration — is how steep the seesaw is.

This gives the rule that makes duration the most-used number in the field:

$$
\frac{\Delta P}{P} \approx -D_{\text{mod}} \times \Delta y
$$

A bond with modified duration 5 loses about 5% if yields rise 1% (`Δy = +0.01`), and gains about 5% if yields fall 1%. That is the whole game. One number, one multiplication, and you have the price move.

Why does dividing by `(1 + y)` convert "years" into "percent per percent"? Because Macaulay duration weighted each cash flow by `t`, but the *sensitivity* of a discounted cash flow `CF / (1+y)^t` to the yield is `t / (1+y)` times its value — the extra `(1+y)` in the denominator comes from differentiating the discount factor. So the price-sensitivity version of the weighted average is the time version scaled down by `(1 + y)`. At a 4% yield that factor is only 1.04, so modified duration is just a few percent below Macaulay duration; at very high yields the gap grows. The two numbers are close cousins, and people are often loose about which one they mean — but when you are computing an actual price move, it is always modified duration you want.

**A real-world wrinkle: semiannual coupons.** U.S. Treasuries and most corporate bonds pay coupons *twice a year*, not once, and yields are quoted on a semiannual basis. The duration machinery is identical, but you run it in half-year periods: a 5-year bond has ten periods, each cash flow is half the annual coupon, and the discount rate per period is half the annual yield. The Macaulay duration then comes out in *half-year* units, which you divide by 2 to express in years, and modified duration divides by `(1 + y/2)` instead of `(1 + y)`. The numbers shift slightly — semiannual compounding makes durations a touch shorter — but every idea in this post carries over unchanged. We use annual compounding throughout for clarity; just know that a real Treasury's published duration was computed semiannually.

#### Worked example: predicting the 4% bond's price move

Our 4% bond has Macaulay duration 4.63 and yields 4%, so its modified duration is:

$$
D_{\text{mod}} = \frac{4.63}{1.04} = 4.45
$$

Now suppose yields rise 1%, from 4% to 5% (`Δy = +0.01`). The duration rule predicts:

$$
\frac{\Delta P}{P} \approx -4.45 \times 0.01 = -0.0445 = -4.45\%
$$

So the price should fall about 4.45%, from \$1,000 to roughly \$955.50. Let's check by actually re-pricing the bond at 5%: discount the same cash flows by 1.05 instead of 1.04, and you get **\$956.71** — a fall of **4.33%**. The duration estimate (−4.45%) is close but slightly overstates the loss. *Duration predicts the move within a fraction of a percent for a 1% shift; the small gap (4.45% predicted vs 4.33% actual) is **convexity**, which we'll meet shortly — and it always works in your favor.*

#### Worked example: the zero moves more than the coupon bond

Take the 5-year zero (Macaulay duration 5, modified duration `5 / 1.04 = 4.81`) and apply the same +1% rate move:

$$
\frac{\Delta P}{P} \approx -4.81 \times 0.01 = -4.81\%
$$

Re-pricing exactly: the zero falls from \$821.93 to \$783.53, a drop of **4.67%**. Compare the two bonds under the *identical* rate shock: the coupon bond loses 4.33%, the zero loses 4.67%. *Same maturity, same rate move, but the zero loses more — because its money all sits at the far end of the beam, giving it the longer duration and the harder fall.* This is the practical payoff of the center-of-gravity idea: where the weight sits decides how much you lose.

## Duration is the slope of the price–yield curve

Here is the geometric way to see all of this at once, and it is the picture that makes duration click for good. Plot a bond's price (vertical axis) against its yield (horizontal axis). The relationship is a downward-sloping, gently curved line — high price at low yields, low price at high yields. (We drew this curve in [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much).)

**Duration is the slope of that curve at the bond's current yield.** A steep slope means a small change in yield produces a big change in price — high duration. A gentle slope means the price barely budges — low duration. When we said "modified duration is the percentage price change for a 1% yield move", we were describing the slope of the price–yield line, normalized by the price so it comes out as a percentage.

![A price–yield curve for a bond with a straight tangent line touching it at the current yield, showing that duration is the slope of the curve and that the straight line underestimates the true convex price](/imgs/blogs/duration-the-most-important-number-in-fixed-income-2.png)

The figure shows the curved price–yield line and a *straight* line just touching it at the current 4% yield. That straight line is the **duration tangent** — duration's linear approximation of the bond. For small rate moves, the tangent hugs the curve and duration is an excellent predictor. For large moves, the curve pulls away from the straight line: the actual price (on the curve) is always *above* the tangent line, whether rates rise or fall. That gap between the curve and its tangent is **convexity**, and it is why the duration estimate slightly overstated the loss in our worked example. The bond's true price falls a little less than the straight-line estimate when rates rise, and rises a little more when rates fall. Convexity is a small bonus to the bondholder, and it grows with maturity. (Convexity has its own post in this series; here, just notice that duration is the *slope* and convexity is the *curvature*.)

#### Worked example: reading sensitivity off the slope

Imagine two bonds plotted on the same chart at the same 4% yield. Bond A is our 4% 5-year (modified duration 4.45). Bond B is a 4% 30-year (modified duration roughly 17). At the 4% point, Bond B's price–yield line is nearly four times as steep as Bond A's. Push yields up 1% and Bond A slides down its gentle slope by about 4.4%; Bond B plunges down its steep slope by about 17%. *Two dots on the same chart, but the steepness of the line through each dot is the whole story — and that steepness is duration.* The longer bond's curve is both steeper (more duration) and more bowed (more convexity), which is why long bonds are the wild ones when rates move.

### Where duration runs out: the convexity correction

Duration is a *straight line*, but the price–yield relationship is a *curve*. For small rate moves the difference is invisible; for large ones it matters, and the correction term has a name — **convexity**. If duration is the slope of the price–yield curve, convexity is its curvature — how fast the slope itself changes as yields move. The full second-order estimate of a price move is:

$$
\frac{\Delta P}{P} \approx -D_{\text{mod}} \times \Delta y + \tfrac{1}{2} \times C \times (\Delta y)^2
$$

where $C$ is the convexity and $(\Delta y)^2$ is the yield change squared. The key feature is that this correction term is **always positive** for an ordinary bond, because the price–yield curve always bows upward (it is convex). That means: when rates *rise*, convexity adds back some price, so your loss is smaller than duration alone predicted; when rates *fall*, convexity adds even more price, so your gain is bigger than duration predicted. Either way, the curvature works *for* the bondholder. That is exactly the asymmetry we saw in the price table — up moves a bit bigger, down moves a bit smaller than the straight-line estimate.

#### Worked example: how much convexity rescues you

Our 4% bond loses 4.45% by the duration estimate when rates rise 1%, but only 4.33% in reality — a 0.12% rescue from convexity. Now scale the shock up to a brutal +3% (rates from 4% to 7%). Duration alone screams `−4.45 × 3 = −13.36%`. The actual reprice? The bond falls to \$876.99, a loss of just **12.30%** — convexity has clawed back more than a full percentage point. And on the upside, a −3% move (rates to 1%) that duration estimates at +13.36% actually delivers **+14.56%**. *Duration is the right answer for small moves and a conservative, slightly-too-harsh answer for big ones; the bigger the move, the more the curvature matters, and it always tilts in the bondholder's favor.* This is why long, convex bonds are prized in volatile markets — they fall less and rise more than their duration suggests. (Convexity gets its own full post in this series; here, the point is just that duration is the first-order term and convexity is the correction.)

## Why a zero's duration equals its maturity, and a coupon bond's is less

We've now seen the punchline twice — once as a balance point and once as a price move. Let's make the rule itself airtight, because it is the fact that lets you *estimate duration without computing anything*.

A **zero-coupon bond** has one cash flow, at maturity. All the weight is there. The balance point can only be at maturity. So **duration = maturity**, always, for a zero.

A **coupon bond** is a zero plus a series of earlier, smaller payments (the coupons). Those early payments put a little weight before maturity, which drags the balance point *backward* from the maturity date. So a coupon bond's duration is **always strictly less than its maturity**. The bigger the coupons, the more weight sits early, and the shorter the duration. This gives a fast mental model: a coupon bond's duration is its maturity, minus a haircut that grows with the coupon.

![Two curves of duration plotted against maturity, one for a zero-coupon bond rising along the 45-degree line and one for a coupon bond rising below it, with the gap widening as maturity grows](/imgs/blogs/duration-the-most-important-number-in-fixed-income-3.png)

The chart plots Macaulay duration against maturity for both kinds of bond. The zero traces the **45-degree line**: its duration equals its maturity at every point, by definition. The coupon bond's line sits *below* it everywhere, and the gap widens as maturity grows — a 1-year coupon bond's duration is barely below 1, but a 30-year 4% coupon bond's Macaulay duration (about 18 years, a modified duration near 17) is far below its 30-year maturity, because three decades of coupons stack up a lot of early weight. *For a zero, duration and maturity are the same word; for a coupon bond, duration is the honest measure of how long your money is really tied up.*

Notice something else in the coupon-bond curve: it *flattens* as maturity stretches out. The duration of a perpetual bond — one that pays coupons forever and never returns principal — does not run off to infinity; it converges to `(1 + y) / y`, which at a 4% yield is just 26 years. The reason is that cash promised 80 or 90 years from now, discounted at 4%, is worth almost nothing today, so it carries almost no weight and cannot pull the balance point any further out. This is why a 30-year bond's duration (≈17) is nowhere near 30, and why a 100-year bond's duration is only modestly longer than a 30-year's, not three times longer. Discounting puts a ceiling on how long duration can get. The implication for risk is practical: extending a portfolio from 10-year to 30-year bonds adds a lot of duration, but extending from 30-year to 100-year adds much less than the maturity jump suggests — diminishing returns on rate risk as you reach for ever-longer paper.

#### Worked example: same maturity, different durations

Three 5-year bonds, all yielding 4%, differing only in coupon:

- **5-year zero** (0% coupon): duration = **5.00** years.
- **5-year, 4% coupon** (our running bond): duration = **4.63** years.
- **5-year, 8% coupon**: duration = **4.37** years.

All mature in exactly 5 years. But the 8% bond hands you twice the annual cash of the 4% bond, so more of its value arrives early, so its balance point sits closest to today — and it will move the *least* when rates change (`Dmod ≈ 4.20`, versus 4.45 for the 4% bond and 4.81 for the zero). *Maturity tells you the last date; duration tells you the true exposure — and two bonds with the same maturity can have meaningfully different rate risk.*

## The coupon effect: higher coupons shorten duration

The last example deserves its own picture, because the coupon effect is the lever bond investors pull most often to control rate risk without changing maturity. Hold maturity fixed at 5 years and slide the coupon up: 0%, 4%, 8%, 12%. As the coupon rises, more cash arrives early, the balance point moves toward today, and duration falls.

![A descending chart of Macaulay duration against coupon rate for a fixed five-year maturity, starting at five years for a zero and falling as the coupon rises](/imgs/blogs/duration-the-most-important-number-in-fixed-income-6.png)

The numbers, all for a 5-year bond at a 4% yield: a **0% coupon** gives duration **5.00**; **4%** gives **4.63**; **8%** gives **4.37**; **12%** gives **4.18**. The curve descends and flattens — the first coupons you add shorten duration the most, and each additional point of coupon shortens it a little less. This is why, during a rate-rise scare, investors rotate toward **higher-coupon bonds**: same maturity, less rate risk, because the cash comes home sooner. It is also why **deeply discounted, low-coupon bonds** are the most dangerous things to hold into a rate spike — they have the longest duration for their maturity.

There is a second lever hiding in the same picture: the *level of yields* itself. When yields are high, the early coupons are discounted harder, so relatively more of the bond's value is concentrated near the front — and duration shrinks. When yields are low (as they were across the 2010s), the opposite happens: discounting is gentle, far cash retains more of its value, and durations *lengthen* across the whole bond market. This is a quietly important macro fact. A decade of near-zero rates didn't just lower yields; it stretched the duration of every bond outstanding, loading the entire financial system with more rate risk per dollar of bonds held. When rates finally rose in 2022, that accumulated duration is precisely what turned an ordinary rate cycle into the worst bond drawdown in modern history. Duration is not a fixed property of a bond; it breathes with the level of yields, and the whole market's duration breathes together.

#### Worked example: using coupon to cut rate risk

You hold \$100,000 of the 4% 5-year bond (modified duration 4.45). You're worried rates will jump 1%, which would cost you about `4.45% × $100,000 = $4,450`. You can't shorten the maturity without selling, but you can swap into a 5-year, 8% coupon bond (modified duration ≈ 4.20). Now the same 1% jump costs about `4.20% × $100,000 = $4,200` — you've shaved \$250 of risk off a \$100,000 position purely by choosing a higher coupon, with no change in maturity or credit. *The coupon is a dial on rate risk: turn it up to soften the blow from rising rates, turn it down (or buy zeros) to amplify the gain from falling ones.*

## Duration is the summary statistic of rate risk

Step back and see what duration buys you. A bond is, in full, a list of cash flows and dates — messy, hard to compare, different for every security. Duration crushes all of that into **one number** that captures the thing you most need to know: how much the bond moves when rates move. That is why it is *the* summary statistic of interest-rate risk.

Its real power is **additivity**. The duration of a portfolio is just the value-weighted average of the durations of its bonds. A \$1 million portfolio that is half in a duration-3 bond and half in a duration-9 bond has a portfolio duration of `0.5 × 3 + 0.5 × 9 = 6`. So a bond manager doesn't track a thousand securities individually; she tracks one number — portfolio duration — and knows instantly that a 1% rate move costs her about 6% of the portfolio. The whole discipline of **immunization** (matching a portfolio's duration to the time horizon of a liability, like a pension payout) rests on this single statistic.

This is also how the dollar-risk number known as **DV01** (the "dollar value of a basis point" — the dollar change in price for a 1 bp yield move) is built. DV01 is just `modified duration × price × 0.0001`. For our \$1,000 bond: `4.45 × 1000 × 0.0001 = $0.445` per basis point. Traders quote risk in DV01s and hedge so the DV01s offset; it is duration, wearing a dollar suit.

There is a useful distinction lurking here between *percentage* risk and *dollar* risk. Modified duration is a percentage sensitivity, so a \$1,000 bond and a \$1,000,000 position with the same modified duration have the same *percentage* exposure but vastly different *dollar* exposure. **Dollar duration** (sometimes "money duration") closes that gap: it is `modified duration × price`, the dollar change in value for a 1.0 (i.e., 100%) change in yield, and DV01 is simply dollar duration scaled to a single basis point. When a trader needs to hedge, percentages are not enough — she has to neutralize *dollars*, and that is what DV01 is for.

#### Worked example: hedging one bond with another

You own \$10 million face of the 4% 5-year bond and you want to hedge its rate risk by shorting 30-year Treasuries (modified duration ≈ 17, priced near par). How many 30-years do you short? You match DV01s. Your position's DV01 is `4.45 × $10,000,000 × 0.0001 = $4,450` per basis point. Each \$1 of 30-year face has a DV01 of `17 × 1 × 0.0001 = $0.0017`. To offset \$4,450 per basis point you need `$4,450 / $0.0017 ≈ $2.6 million` face of 30-years. *Because the 30-year has nearly four times the duration, you hedge \$10 million of the short bond with only about \$2.6 million of the long one — duration tells you not just the risk, but exactly how much of one bond cancels another.* This DV01-matching is the everyday arithmetic of every bond trading desk.

### Duration is additive — that's its superpower for portfolios

The single property that makes duration indispensable at scale is that it **adds up**. The duration of a portfolio is the value-weighted average of the durations of its holdings — full stop. There is no cross-term, no interaction to worry about, because to first order each bond's price move is independent and proportional to its own duration. A manager running a \$2 billion bond fund does not stare at ten thousand securities; she computes one number, the portfolio's average duration, and reads her entire rate exposure off it. Want to cut the fund's rate risk by 20% ahead of a Fed meeting? Sell enough long bonds (or buy enough short ones, or short Treasury futures) to drop the average duration by 20%. The whole machinery of professional fixed-income risk management reduces to *steering one number*.

#### Worked example: barbell vs bullet, same duration

A "bullet" portfolio puts all \$1 million into a single duration-7 bond — duration 7. A "barbell" splits the same \$1 million half into a duration-2 bond and half into a duration-12 bond: portfolio duration `0.5 × 2 + 0.5 × 12 = 7`. *Identical duration, so identical first-order rate risk* — both lose about 7% if rates rise 1% in a parallel shift. But they are not identical bonds: the barbell has more convexity (the long leg curves a lot) and behaves differently if the yield curve *twists* rather than shifting in parallel. *Duration says they're the same; the second-order details say they're not — which is exactly why duration is the necessary first number and never the last one.*

![A grouped bar chart comparing a long-duration bond fund and a short-duration bond fund across five yield-change scenarios, with green gain bars when yields fall and red loss bars when yields rise](/imgs/blogs/duration-the-most-important-number-in-fixed-income-4.png)

The figure above is duration's most practical face: how a bond *fund's* return tracks the 10-year Treasury yield across a range of rate moves. A long-term Treasury fund with a duration of about 6.5 moves roughly `−6.5 × Δy`: when the 10-year yield drops 1%, the fund gains about 6.5%; when it rises 1%, the fund loses about 6.5%. A short-term fund with a duration near 2 barely moves under the same yield swing — its bars are a fraction of the long fund's in every scenario. If you own a bond fund, the single most important number on its fact sheet is its **average effective duration** — it is the multiplier between the headline yield move you read in the news and the gain or loss in your account. Note "effective" duration here: for bonds with options (callable bonds, mortgages), the cash flows themselves change when rates move, so we measure duration by re-pricing the bond up and down a small amount and reading the slope numerically. For a plain Treasury or corporate bond, effective and modified duration are essentially the same number.

#### Worked example: a rate move through a fund you own

You hold \$50,000 in a long-term Treasury bond fund with an effective duration of 6.5. The Federal Reserve surprises the market and the 10-year yield falls 0.5% (50 bps), from 4.3% to 3.8%. Your fund's price should rise about:

$$
\frac{\Delta P}{P} \approx -6.5 \times (-0.005) = +0.0325 = +3.25\%
$$

That's a gain of about `3.25% × $50,000 = $1,625` from a single half-point move in one yield. Reverse it — the 10-year rises 0.5% — and you'd be down about \$1,625. *The yield number on the evening news is not abstract: multiply the change by your fund's duration and you have, to a good approximation, the change in your own balance.*

## Common misconceptions

**"Duration is just the bond's maturity."** No — they are equal *only* for a zero-coupon bond. For every coupon bond, duration is shorter than maturity, because the coupons return some of your money before the final date and pull the balance point back. A 30-year Treasury with a 4% coupon has a duration around 17, not 30. Confusing the two will roughly double your estimate of a long bond's rate risk.

**"Duration is measured in years, so it's about time, not price."** Macaulay duration *is* in years, and it genuinely measures average time-to-cash. But modified duration — the same number divided by `(1 + yield)` — is a pure price-sensitivity: a percentage move per 1% rate move. The unit "years" is a historical artifact; the *use* of duration is almost always about price. Both are the same underlying fact seen from two sides.

**"A higher yield means more rate risk."** The opposite, holding other things equal. Modified duration is Macaulay duration divided by `(1 + yield)`, so a *higher* yield actually *shrinks* modified duration a little. More importantly, higher coupons (which often come with higher yields) front-load cash and shorten duration. The bonds with the most rate risk are long-maturity, *low*-coupon bonds — the ones whose money is parked far in the future.

**"Duration tells you everything about how a bond moves."** It tells you the first-order move — the slope. For small rate changes it is excellent. For large ones, the price–yield curve bends away from duration's straight-line estimate, and you need **convexity** (the curvature) to capture the rest. Duration alone slightly overstates losses when rates rise and understates gains when they fall; convexity corrects both, always in the bondholder's favor. Duration is necessary, not sufficient.

**"Duration is fixed for a bond."** It drifts. As a bond ages toward maturity, its duration falls (less time left, less weight far out). As its yield changes, modified duration changes. And for bonds with embedded options — callable bonds, mortgage-backed securities — duration can move sharply, even turn *negative* in extreme cases, because the cash flows themselves reschedule when rates move. That is why practitioners track *effective* duration, re-measured as conditions change, not a number stamped on at issue.

**"Two bonds with the same duration carry the same risk."** They carry the same *first-order rate* risk, which is what duration measures — but not the same *total* risk. They can differ in convexity (how they behave in big moves), in credit risk (the chance the issuer doesn't pay, which duration ignores entirely), and in liquidity. Duration is the rate-risk summary, not a complete risk picture. A Treasury and a junk bond with identical durations are not identically risky.

**"Rising rates are always bad for a bond holder."** Not over a long enough horizon. Duration captures the *immediate* price hit, but a higher yield also means every coupon you receive can be reinvested at a better rate, and the eventual maturity payment is unchanged. There is, in fact, a beautiful result hiding in duration: if you hold a bond for *exactly its Macaulay duration*, the price loss from a one-time rate rise is almost perfectly offset by the gain from reinvesting coupons at the new, higher rate. That is the original reason Macaulay defined the number in 1938, and it is the foundation of *immunization*: match your holding period to the bond's duration and you lock in your yield regardless of which way rates jump next. So duration is not only a measure of risk — it is also the horizon at which that risk cancels itself out. Rising rates sting today and heal over time, and duration is the number that tells you exactly how long the healing takes.

## How it shows up in real markets

**Silicon Valley Bank, 2023 — duration risk that sank a bank.** SVB had taken in a flood of deposits and parked tens of billions in long-dated Treasuries and mortgage-backed securities — high duration, bought when yields were near 1-2%. When the Fed raised rates roughly 5 percentage points across 2022-2023, those long-duration holdings fell enormously in price (a portfolio with duration near 6 loses about 6% per 1% rate rise, and these ran longer). The unrealized losses were so large that when depositors pulled their money and SVB had to sell, the losses became real and the bank failed within days. It is the cleanest modern lesson that duration is not academic: a bank that mismatched the duration of its assets against the short duration of its deposits was destroyed by a rate move it could have measured in advance. The mechanism is exactly the duration rule in this post, scaled to a balance sheet. (The episode is dissected in [SVB and Credit Suisse: the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**The 2022 bond bear market — when "safe" bonds fell 20%+.** In 2022 the U.S. 10-year Treasury yield rose from about 1.5% to about 4.3% — roughly a 2.8% jump. Long-duration funds got crushed: the longest Treasury index funds, with durations near 17, fell on the order of `−17 × 2.8% ≈ −30%`+ at the worst, and the broad U.S. bond market had its worst year on record. Millions of conservative savers in "safe" total-bond-market funds (duration around 6) watched them fall double digits. The single number that predicted the carnage in advance was each fund's duration — and almost nobody outside the bond world had ever looked at it.

**Long-duration government bonds as a recession hedge.** The flip side: when the economy weakens and the Fed cuts rates, long-duration bonds *soar*, precisely because of the same steep slope. In the 2008 crisis and again in March 2020, as yields collapsed, long-Treasury funds (duration near 17) posted double-digit gains while stocks fell. Investors who hold long bonds *for* their high duration are buying a deliberate bet that rates will fall — the same property that made them dangerous in 2022 made them a haven in 2020. Duration is direction-neutral; it just tells you the size of the swing. (This is the engine behind the [stock–bond correlation and the 60/40 portfolio](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).)

**Pension funds and liability-driven investing.** A pension fund owes payments decades into the future — a liability with a very long duration. To avoid being whipsawed by rates, funds practice **immunization**: they buy bonds whose *asset* duration matches the *liability* duration, so that when rates move, the value of the assets and the value of the obligations move together and cancel. The entire multi-trillion-dollar discipline of liability-driven investing is built on the single statistic in this post. When it goes wrong it goes wrong spectacularly — the UK "gilt crisis" of September 2022 was a duration-and-leverage mismatch in pension hedges that forced the Bank of England to intervene.

**Mortgage-backed securities and negative convexity.** When you hold a pool of home mortgages as a bond, the homeowners can refinance — they prepay when rates fall and hold on when rates rise. That makes the cash flows reschedule against you: as rates fall, prepayments shorten the bond's duration just when you'd want it long; as rates rise, the duration *extends* just when you'd want it short. This "negative convexity" means a mortgage bond's *effective* duration swings around with the level of rates, and measuring it requires re-pricing the bond up and down — exactly the effective-duration method described above. It is why mortgage portfolios need constant duration hedging, and why a naive modified-duration number badly misstates their risk.

**Austria's 100-year bond — duration in its most extreme form.** In 2017 and again in 2020, Austria issued *century bonds* — government debt maturing in 100 years — at very low coupons. Because nearly all the value of a 100-year, low-coupon bond sits in the far-distant principal, its duration is enormous: well over 40 years. That made it a pure, almost frightening, bet on the level of rates. When yields fell in 2019-2020, the 2117 bond's price roughly doubled; when yields surged in 2022, it lost the majority of its value, falling on the order of 60-70% from its peak. Same bond, same issuer, no default risk to speak of — the entire ride was duration. The century bond is the cleanest demonstration that duration, not credit, is the dominant risk in high-quality long bonds, and that "safe" sovereign debt can be staggeringly volatile if its duration is long enough.

**Convexity as a deliberate trade.** Sophisticated investors don't just measure convexity — they buy and sell it. A "duration-neutral, long-convexity" position holds bonds (often a barbell of very short and very long maturities) arranged so the portfolio's duration is zero but its convexity is positive. Such a position is roughly flat to small rate moves but *profits from large moves in either direction*, because the curvature pays off whichever way rates jump. It is the bond market's version of being long volatility. When 2020's pandemic shock sent yields gyrating, long-convexity positioning paid off handsomely; in calm, range-bound markets it slowly bleeds. The point for our purposes: once you understand that duration is the slope and convexity is the curvature, you can see that markets price and trade *both* — duration is the first number, but it is not the only one professionals manage.

## When this matters to you, and where to go next

If you own any bonds — directly, in a fund, in a target-date retirement account — duration is the number that turns the abstract phrase "interest rates moved" into a concrete dollar amount in your account. Find the average effective duration on your bond fund's fact sheet, multiply it by the rate move you read about, and you have a good estimate of your gain or loss before you even log in. For a saver choosing between a short-term and a long-term bond fund, that single number *is* the choice: more duration means more reward when rates fall and more pain when they rise.

To go deeper, the natural next steps in this series are **convexity** (the curvature that duration misses, and why it always helps the bondholder) and the **key-rate durations** that decompose a bond's sensitivity to different parts of the yield curve. For the formal derivations and the analytics behind DV01, immunization, and effective duration, see [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics) and [bond pricing](/blog/trading/quantitative-finance/bond-pricing). For the macro angle — why the rates that drive all of this move in the first place — see [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) and [the central bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance). Master duration, and you have mastered the single most important number in fixed income — the one that tells you, before the market moves, exactly how much you stand to win or lose.

*This is educational material about how bonds behave, not investment advice; duration measures risk, it does not eliminate it, and every bond that can gain on falling rates can lose just as much on rising ones.*
