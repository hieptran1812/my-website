---
title: "How Monetary Policy Moves Bonds: Duration, Convexity, and the Curve"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A beginner-friendly deep dive into how a central bank's rate decisions reprice the entire bond market, with duration as the exact number that tells you how much a bond falls when yields rise, convexity as the curvature correction, and the yield curve as the map of where policy hits."
tags: ["macro", "monetary-policy", "bonds", "duration", "convexity", "fixed-income", "treasury-yields", "yield-curve", "interest-rates", "bond-math", "fed", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bond is a stream of fixed future payments, so its price moves *opposite* to yields, and the size of that move is not a vibe — it is an exact number called duration, with convexity as the second-order correction.
>
> - **Duration is the master number.** A bond's price change is approximately *minus duration times the yield change*. A 10-year Treasury with a duration of about 8.5 loses roughly **8.5%** of its value when yields rise **1.00%**. Multiply duration by the size of the rate move and you have the price hit to the basis point.
> - **Convexity is the curvature.** The real price-yield relationship is a curve, not a straight line, and it always bows in the holder's favor: duration *overstates* the loss on a yield rise and *understates* the gain on a yield fall. The longer the bond, the more this matters.
> - **The curve is the map.** Monetary policy pins the front end (the 2-year tracks the policy rate almost mechanically) while the long end (the 10-year, 30-year) tracks the *expected average* of future short rates plus a term premium. A policy move reprices the whole curve, but not evenly.
> - The one number to remember: when the Fed hiked from near zero to 5.5% in 2022, long Treasuries (20-year-plus) fell about **31%** — the worst year for the "safest" asset in modern history, and every basis point of it was predictable from duration.

In 2022 the safest asset on earth had its worst year in living memory. Long-dated US Treasury bonds — government IOUs with zero default risk, the thing institutions buy precisely *because* they cannot lose money to a missed payment — fell about **31%** over the calendar year. A retiree who had done everything "right," who had moved out of stocks and into long government bonds for safety, watched nearly a third of that money evaporate. The 2020 vintage 30-year Treasury, issued at a yield near 1.5%, traded down to roughly half its face value at the lows. Nothing defaulted. No coupon was missed. The bonds were exactly as safe as advertised. And they still cratered.

What happened was not a credit event. It was a *rate* event, and it was completely mechanical. The Federal Reserve, fighting the worst inflation in forty years (the consumer price index peaked at **9.06%** in June 2022), raised its policy rate from a floor of 0.25% to 4.50% over nine months and then to 5.50% the following year — the fastest tightening cycle in four decades. When the policy rate goes up, the yield demanded on every bond goes up. And when the yield on a bond goes up, its price — by the iron logic of present value — goes *down*. The only question is *how much*, and the answer to that question is the entire subject of this post. It has a name, two names actually: **duration** and **convexity**.

Here is the promise of this article, and it is a strong one: by the end, you will be able to price the effect of a monetary-policy move on any bond, to the basis point, with arithmetic you can do on a napkin. Bonds are the most direct expression of a view on interest rates, and unlike almost everything else in markets, the math underneath them is *exact*. A stock's reaction to a rate hike is a guess. A bond's reaction is a calculation. We will build that calculation from absolute zero — what a bond even is, why its price moves opposite to its yield, what duration measures, why there are two kinds of duration, what convexity corrects, and how a central bank's decision ripples down the whole yield curve — and then we will turn it into a trading playbook.

![Price-yield seesaw with yield and price on opposite ends and duration as the lever](/imgs/blogs/how-monetary-policy-moves-bonds-duration-convexity-1.png)

The figure above is the mental model for the whole post. Picture a seesaw. On one end sits the bond's yield; on the other end sits its price. They are rigidly connected: push the yield end up and the price end *must* go down, every time, by the geometry of the lever. Duration is the **length of the lever arm** — a short-duration bond is a stubby seesaw that barely tips when yields move, while a long-duration bond is a long lever that swings violently. The Fed's job, when it changes policy, is to push down on the yield end. Everything that follows is just working out, precisely, how far the price end swings.

## Foundations: what a bond is and why price moves opposite to yield

Before any trading, four ideas have to be rock-solid: what a **bond** is, why its **price moves inverse to its yield**, what **duration** measures, and what **convexity** corrects. Everything in the rest of the post is a consequence of these four. If you already know that a bond is a stream of fixed cash flows discounted at a yield, skim to the duration section — but read the inverse-relationship part carefully, because the *reason* price moves opposite to yield is the foundation that makes duration intuitive rather than memorized.

### A bond is a stream of fixed future payments

Start with the simplest possible object. A bond is a loan you make, usually to a government or a company, structured as a fixed schedule of payments. When the US Treasury issues a 10-year note with a 4% coupon and a \$1,000 face value, it is promising you exactly this: pay me \$1,000 today, and I will pay you \$40 every year for ten years (the coupon, which is 4% of face), and then return your \$1,000 at the end (the principal, or face value, repaid at maturity). That is the entire contract. The payments are *fixed* — \$40 a year, every year, no matter what happens to interest rates, inflation, or the economy. This fixedness is the whole reason bonds behave the way they do.

So a bond is nothing more than a known stream of future dollars: a series of small inflows (the coupons) followed by one big inflow (the principal). The defining question of all bond math is: *what is that future stream worth today?* Because the payments are fixed in dollar terms but you are buying them with today's dollars, the answer depends entirely on one thing — the interest rate you could earn elsewhere. That rate is the bond's **yield**.

### Why price moves opposite to yield: the present-value seesaw

Here is the single most important idea, and it is the reason the seesaw exists. The price of a bond is the **present value** of its future payments — what that stream of fixed dollars is worth today, after discounting each payment back to the present at the prevailing interest rate. And present value moves *opposite* to the discount rate. Always. Mechanically. Let me make this concrete rather than abstract.

Suppose you own a bond that pays \$40 a year. When you bought it, the going rate on similar bonds was 4%, so \$40 a year was a perfectly competitive payout and the bond was worth its \$1,000 face value. Now suppose the Fed hikes and newly issued bonds of the same maturity pay 5% — a brand-new bond pays \$50 a year on the same \$1,000. Your old bond still pays only \$40. Nobody will pay you \$1,000 for a bond paying \$40 when they could buy a fresh one paying \$50 for the same money. So the *price* of your old bond has to fall until its \$40 annual payout represents a competitive 5% return on the lower purchase price. The price falls so the yield rises to match the market. That is the inverse relationship, and it is not a coincidence or a convention — it is arithmetic. **A bond's fixed payments become more or less attractive only by its price moving, because the payments themselves cannot move.**

The same logic runs in reverse. If the Fed cuts and new bonds pay only 3%, your old bond paying \$40 suddenly looks generous, and buyers bid its price *up* above \$1,000 until the \$40 payout represents a 3% yield on the higher price. Yields down, price up. This is the seesaw in the cover figure: yield on one end, price on the other, rigidly opposed.

There is a precise formula underneath this, and we will use it, but the formula is just the bookkeeping of the intuition above. The price of a bond is the sum of every future payment, each divided by a growing power of one-plus-the-yield. We will write the exact code for it shortly. The crucial takeaway is structural: **because every payment is divided by a power of the yield, raising the yield shrinks the present value of every single payment at once, and the bond's price drops.**

### The yield curve: every maturity has its own yield

The government does not issue just one bond — it issues a whole menu of maturities: 1-month and 3-month bills, 2-year and 5-year notes, 10-year notes, and 20-year and 30-year bonds. Each has its own yield, set continuously by the world's deepest market. Plot every maturity's yield on one chart — maturity on the horizontal axis, yield on the vertical — and you get the **yield curve**, the term structure of interest rates. (For a full treatment of the curve's *shape* and what its slope forecasts, see [reading the yield curve: slope, inversion, and recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).)

For this post the curve matters in two ways. First, **different maturities have different durations**, so the same policy move hits different parts of the curve with wildly different force. Second, **policy does not move the whole curve evenly**: the front end is pinned almost mechanically to the policy rate, while the long end floats on expectations. We will build both of those ideas in full. For now, hold onto the picture of a single line — the price of money across every horizon at once — that the central bank can push around.

### Basis points and the units of the bond market

Before going further, two units that bond traders breathe. First, yields are quoted and moved in **basis points** (bps), where one basis point is one-hundredth of a percentage point. A yield rising from 4.00% to 4.25% rose 25 basis points; the 2022 hiking cycle of 5.25 percentage points was "525 basis points." When you hear that "the 10-year sold off 8 bps today," it means its yield rose 0.08% and its price fell by roughly its duration times 0.08% — for an 8.5-duration bond, about 0.68%. Living in basis points keeps you precise: a "small" 10-bps move on a long bond is a real chunk of price, and a "big" 100-bps move is a duration-sized event.

Second, professionals often measure rate risk not in percent but in dollars, with a number called **DV01** — the *dollar value of a 1-basis-point* move. DV01 is just duration translated into the currency of the position: it is the dollars you gain or lose if the yield moves one basis point. For a \$1,000,000 position with a duration of 8.5, a 1-bp move is 8.5 × 0.0001 × \$1,000,000 = \$850, so the DV01 is \$850. DV01 is the unit you use to *match* the two legs of a steepener or flattener — you size each leg so their DV01s are equal, which strips out the level of rates and leaves only the slope. Duration and DV01 are the same idea in two costumes: duration is the percentage lever, DV01 is the same lever measured in dollars per basis point.

### Duration and convexity, in one sentence each

Two more terms to define before we go deep, so the map is clear:

- **Duration** is the bond's price sensitivity to yields — specifically, the *approximate percentage* the price falls for a 1-percentage-point rise in yield. A duration of 8.5 means roughly an 8.5% price drop for a +1% yield move. It is the slope of the price-yield line.
- **Convexity** is the *curvature* of that line. The price-yield relationship is not actually straight; it bows. Convexity measures how much, and it always works in the bondholder's favor — duration alone overstates losses and understates gains. It is the second derivative, the correction to the straight-line estimate.

Those are the two numbers that, together, let you price any policy move's effect on any bond. The rest of this post is making each one precise and then trading on it.

## Duration as the master number

Duration is the single most useful number in fixed income, and once it clicks it changes how you see the entire bond market. The core relationship is so simple it fits on one line:

**percentage price change ≈ −(modified duration) × (change in yield)**

That is it. That is the master equation. If a bond has a modified duration of 8.5 and its yield rises by 1.00 percentage point (which bond traders call 100 basis points), its price falls by approximately 8.5 × 1.00 = 8.5%. If the yield rises by only 0.25% (25 basis points), the price falls by about 8.5 × 0.25 = 2.1%. Multiply the duration by the size of the rate move, flip the sign, and you have the price impact. The whole reason this works is the seesaw: duration is the lever length, the yield move is how hard you push, and the product is how far the price end swings.

### Where duration comes from: a weighted average of payment timing

The word "duration" was chosen deliberately — it originally measured *time*. The first and most intuitive definition, **Macaulay duration**, is the weighted-average number of years you wait to receive the bond's cash flows, where each payment's weight is its share of the bond's present value. A zero-coupon bond that pays everything at year 10 has a Macaulay duration of exactly 10 years: all the money arrives at one moment, so the average wait is the full maturity. A coupon bond that pays you something every year along the way has a *shorter* duration than its maturity, because some of your money comes back early — those early coupons pull the average wait below the final maturity date.

This is the first place beginners go wrong, so let me state it sharply: **duration is not the maturity.** A 10-year Treasury does not have a duration of 10; it has a duration closer to 8.5, because its coupons return some cash before year 10. A 30-year bond does not have a duration of 30; it is closer to 17 to 19, depending on its coupon. The higher the coupon, the more cash comes back early, and the *shorter* the duration. A zero-coupon bond is the only case where duration equals maturity, because there is exactly one payment at the very end.

Why does the timing of cash flows control price sensitivity? Because money far in the future is discounted by more powers of the yield, so its present value is *more* sensitive to a change in that yield. A dollar arriving in 30 years, divided by (1 + yield) thirty times, swings hard when the yield ticks up. A dollar arriving next year barely moves. The longer your money is locked away in the future, the more a change in the discount rate reprices it — and duration is precisely the weighted-average measure of how far in the future your money sits.

### Modified vs Macaulay duration

There are two durations and you need to keep them straight, though they are nearly the same number. **Macaulay duration** is the weighted-average time-to-cash-flow, measured in years — the original "how long until I get my money back on average" idea. **Modified duration** is the one traders actually use for price moves: it is the Macaulay duration divided by (1 + yield-per-period), and it directly gives the *percentage price change per 1% yield change*. The relationship is:

**modified duration = Macaulay duration ÷ (1 + yield)**

For a bond yielding 4%, a Macaulay duration of 8.84 years becomes a modified duration of 8.84 ÷ 1.04 = 8.5. The two numbers are close — they differ only by the small (1 + yield) factor — but they answer different questions. Macaulay answers "how many years, on average, until I'm repaid?" Modified answers "what percent does my price move per 1% yield change?" When someone says "this bond has a duration of 8.5," they almost always mean modified duration, because that is the number that translates a rate move into a price move. Throughout this post, when I write "duration" without a qualifier, I mean modified duration.

### Computing the price hit: the code

Here is the actual calculation, in Python, so you can see there is no magic — just present value and a tiny bit of calculus. This function prices a bond from its cash flows and then estimates its modified duration by repricing it at a slightly higher and slightly lower yield (a numerical derivative). Read it once; the whole post is encoded in these few lines.

```
def bond_price(face, coupon_rate, ytm, years, freq=1):
    coupon = face * coupon_rate / freq          # cash per coupon period
    n = years * freq                            # total number of periods
    y = ytm / freq                              # yield per period
    pv = 0.0
    for t in range(1, n + 1):
        pv += coupon / (1 + y) ** t             # present value of each coupon
    pv += face / (1 + y) ** n                   # present value of principal
    return pv

def modified_duration(face, coupon_rate, ytm, years, freq=1):
    h = 0.0001                                  # 1 basis point bump
    p_up = bond_price(face, coupon_rate, ytm + h, years, freq)
    p_dn = bond_price(face, coupon_rate, ytm - h, years, freq)
    p_0 = bond_price(face, coupon_rate, ytm, years, freq)
    return -(p_up - p_dn) / (2 * h * p_0)       # percent price move per 1.0 in yield

p = bond_price(1000, 0.04, 0.04, 10)            # 1000.0 (priced at par)
d = modified_duration(1000, 0.04, 0.04, 10)     # ~8.1 (annual compounding)
print(p, d)                                      # 1000.0  8.11
    # a 10-year, 4%-coupon par Treasury sits around 8.1 to 8.5
    # depending on coupon and yield; this post rounds to ~8.5
    # as the conventional 10-year duration figure.
```

Run it and a 10-year, 4%-coupon Treasury priced at par returns a modified duration of about 8.1, which rounds to the conventional ~8.5 the market quotes for a 10-year (a slightly lower coupon or yield lengthens it). That single number is the lever length for that bond. Bump the yield 1% and multiply: an 8.5% price loss. The `modified_duration` function is doing nothing but measuring the slope of the price-yield line at the current yield — repricing the bond a hair above and a hair below the current yield and seeing how fast the price changes. That slope, expressed as a percentage of price, *is* duration.

#### Worked example: the \$1,000,000 position and the +1% yield move

You manage a \$1,000,000 position in a 10-year Treasury with a modified duration of 8.5. The Fed surprises the market with a hawkish stance and the 10-year yield rises by 1.00 percentage point — 100 basis points — over the following weeks. What happens to your position?

Apply the master equation directly. The percentage price change is approximately −(duration) × (yield change) = −8.5 × 1.00% = **−8.5%**. On a \$1,000,000 position, that is a loss of 0.085 × \$1,000,000 = **\$85,000**. Your million-dollar "safe" government-bond holding just lost \$85,000, and nothing went wrong — no default, no missed coupon, no credit downgrade. The bond is still exactly as creditworthy as before. The yield rose, so the price fell, and duration told you the size of the fall before it happened. Had you held a 2-year note instead (duration about 1.9), the same 1% yield move would have cost you only 1.9 × \$10,000 = \$19,000. **The intuition to keep: the loss on a bond from a rate move is not a vibe — it is duration times the move times your position, and you can compute it before the trade.**

![Bar chart of percent price change for 2, 5, 10, and 30 year bonds on a 1 percent yield rise](/imgs/blogs/how-monetary-policy-moves-bonds-duration-convexity-2.png)

The bar chart above makes the duration-as-lever point visceral. The *same* policy move — a uniform +1.00% rise in yields across the curve — costs a 2-year bond about 1.9%, a 5-year about 4.5%, a 10-year about 8.5%, and a 30-year about 19%. One rate move, four wildly different outcomes, and the only thing that differs is duration. This is why, when you have a view on rates, the *first* decision is not which bond to buy but *how much duration to take* — because duration is the dial that sets how hard your view bites. A trader who is right on the direction of rates but wrong on duration can still lose, or leave most of the move on the table.

### What makes duration large or small: the three drivers

Three things determine a bond's duration, and knowing them lets you estimate the lever length without a calculator. First and most important is **maturity**: longer bonds have longer durations, because more of your money sits far in the future where the discounting bites hardest. This is the dominant driver — a 30-year bond will always have far more duration than a 2-year, full stop. Second is the **coupon**: higher coupons *shorten* duration, because more cash comes back early, pulling the weighted-average wait forward. A zero-coupon 10-year (all money at the end) has the maximum duration of 10; a high-coupon 10-year has a shorter duration because its fat coupons return cash along the way. Third is the **yield level** itself: higher yields slightly *shorten* duration, because at higher discount rates the distant payments are worth proportionally less, so they carry less weight in the average. This third effect is why duration is not a fixed constant — a bond's duration shifts a little as its yield moves, which is exactly the convexity we will get to.

Put these together and you can sanity-check any duration claim. Someone tells you a long Treasury fund has a duration of 25? Impossible for a 30-year bond with a normal coupon — the coupon shortens it to 17 to 19. Someone says a 2-year corporate has a duration of 5? Wrong — maturity caps it near 1.9. The three drivers — maturity up, coupon down, yield down, all *lengthen* duration — let you carry a rough duration in your head for any bond, which is all you need to estimate its price hit from a policy move.

A subtle but important consequence: in a *low-rate* world, durations are *longer* across the board, because low coupons and low yields both lengthen the lever. That is precisely why the 2020–2021 bond market was so dangerous. With yields near record lows, the typical long bond had its longest duration in decades — the most price sensitivity to a rate rise exactly at the moment when rates had nowhere to go but up. The setup for the 2022 crash was baked into the duration math before a single hike happened: ultra-low yields had stretched the lever to its maximum length, so when the seesaw tipped, it tipped violently.

### Duration is additive: the portfolio number

One more property makes duration indispensable: it is **additive across a portfolio**, weighted by dollar value. If half your money is in a 2-year bond (duration 1.9) and half in a 30-year bond (duration 19), your portfolio duration is roughly the dollar-weighted average: 0.5 × 1.9 + 0.5 × 19 = 10.45. That single number tells you that a 1% rise in yields across the curve would cost your blended portfolio about 10.45%. Fund managers live by this number — "portfolio duration" is the one dial that summarizes a bond book's entire exposure to a parallel shift in rates. When a manager says they are "shortening duration ahead of the Fed," they mean they are selling long bonds and buying short ones to lower that single number, so the book bleeds less if yields rise.

## Convexity: the second-order correction

Duration is a straight-line approximation, and straight lines are wrong about curves. The real relationship between a bond's price and its yield is not a line — it is a *curve*, and it bows. Convexity is the name for that bow, and understanding it is what separates someone who can roughly estimate a bond's move from someone who can price it accurately for *large* rate moves.

### Why the price-yield line is actually curved

Recall the pricing formula: each payment is divided by a power of (1 + yield). That division is not linear in the yield — it is a hyperbola. Dividing by (1.05) versus (1.04) is a different proportional change than dividing by (1.10) versus (1.09). So as the yield rises across a wide range, the price does not fall along a straight line; it falls along a curve that flattens out. The curve is **convex** — it bows toward the origin, always sitting *above* the straight tangent line that duration draws.

This curvature has a beautiful consequence for the bondholder, and it is the single most important thing to understand about convexity: **the curve always lies above the straight-line estimate, so duration overstates your loss when yields rise and understates your gain when yields fall.** When the yield goes up, the real price falls *less* than duration predicts (good — your loss is smaller). When the yield goes down, the real price rises *more* than duration predicts (also good — your gain is bigger). Convexity is asymmetric in your favor on both sides. You want it.

![Two-panel before and after showing duration straight line versus the convex true price curve](/imgs/blogs/how-monetary-policy-moves-bonds-duration-convexity-4.png)

The figure above contrasts the two estimates side by side. On the left, duration alone treats the price-yield relationship as a straight line: a +1% yield move and a −1% yield move are perfect mirror images, the loss exactly equal to the gain. On the right, the true convex curve breaks that symmetry. For our 10-year bond, duration says +1% costs you 8.5% and −1% earns you 8.5%. The convex truth is that +1% costs a bit *less* than 8.5% and −1% earns a bit *more* than 8.5%. The straight line is a decent guess for small moves and a meaningfully wrong guess for large ones — and in 2022, the moves were large.

![Computed price-yield curve for a 10-year bond across yields showing the convex shape](/imgs/blogs/how-monetary-policy-moves-bonds-duration-convexity-5.png)

This chart is the same idea drawn from the actual pricing math. The solid blue line is the true price of a 10-year, 4%-coupon bond computed at every yield from 0.5% to 8.5%; the dashed red line is the duration approximation — the straight tangent at today's 4% yield. Notice that the blue curve sits *above* the red line everywhere except at the single point where they touch (today's yield). The green shaded gap between them *is* convexity. It is small near the tangent point (so duration is a fine estimate for small moves) and grows as you move far from today's yield (so duration's error matters for big moves). For a 10-year bond the gap is modest; for a 30-year bond, as we will see, it is large enough to change how you trade.

### The full second-order formula

The complete estimate adds the convexity term to the duration term:

**percentage price change ≈ −(duration × Δy) + (½ × convexity × Δy²)**

The first term is the straight-line duration estimate. The second term is the curvature correction, and notice it has Δy *squared* — which is always positive regardless of whether yields rose or fell. That squared term is why convexity helps you in both directions: it adds a positive amount whether the yield went up or down, shrinking losses and boosting gains. The bigger the move (the bigger Δy), the more that squared term matters, because squaring a large number grows fast. For a 0.25% move, Δy² is tiny and convexity is negligible. For a 2% move, Δy² is sixteen times larger and convexity becomes a real number you cannot ignore.

#### Worked example: why a 30-year's loss on +1% is less than twice a 15-year's

Here is the convexity effect made concrete, and it is the kind of thing that trips up people who only know duration. A 15-year bond has a duration of roughly 11; a 30-year bond has a duration of roughly 19. Duration *alone* would suggest that since the 30-year has not-quite-double the duration of the 15-year, its loss on a +1% move should be a bit under double. But the relationship is even less than duration-proportional once you account for convexity, and the gap widens as moves get large.

Take a +1% yield move. The 15-year, duration 11, loses about 11% from the duration term, but convexity claws back a fraction, so the real loss is closer to **10.5%**. The 30-year, duration 19, loses about 19% from the duration term — but the 30-year has *much* higher convexity (curvature grows with maturity), so it claws back more, and its real loss is closer to **17.5%** rather than the full 19%. So the 30-year's actual loss of about 17.5% is less than 2× the 15-year's 10.5% (which would be 21%). On a \$500,000 position in each, the 15-year loses about \$52,500 and the 30-year about \$87,500 — the long bond hurts far more, but convexity means it hurts a touch less than a naive doubling of duration would say. **The intuition: convexity is the long bond's small consolation prize — the longer and more convex the bond, the more the curvature softens the blow on the way down and amplifies the bounce on the way up.**

### Why convexity is worth paying for

Because convexity helps you in both directions, it is a desirable property, and in efficient markets desirable properties cost money. Two bonds with the same duration but different convexity will trade at slightly different yields — the more convex one yields a touch less, because you are paying for the asymmetry. This matters most in volatile-rate environments: if you expect *big* rate moves but are unsure of the direction, high convexity is pure value, because you win more on the up-move than you lose on the down-move of the same size. Mortgage-backed securities famously have *negative* convexity (the curve bows the wrong way, because homeowners refinance when rates fall, cutting off your upside), which is exactly why they yield more than Treasuries of similar duration — investors demand compensation for holding the wrong-way curvature. For a plain Treasury, though, convexity is a free gift baked into the math, and it is always in your favor.

### Negative convexity: when the curve bows the wrong way

Most bonds have positive convexity (the curve bows in your favor), but one huge asset class does not: **mortgage-backed securities** (MBS). A mortgage bond is a claim on a pool of homeowners' mortgage payments, and homeowners have an option you do not control — they can refinance. When yields *fall*, homeowners rush to refinance into cheaper loans, paying off the old mortgage early, so your high-yielding bond gets handed back to you as cash exactly when you would have wanted to keep it. Your upside is capped: the price barely rises on a yield fall because the bond keeps getting prepaid. When yields *rise*, the opposite happens — nobody refinances, the mortgages stay outstanding longer, your money is locked in a now-below-market bond, and the duration *extends* right when you would have wanted it short. This is **negative convexity**: the curve bows *against* you, amplifying losses and capping gains, the mirror image of a Treasury's friendly curvature.

Negative convexity is why mortgage bonds yield more than Treasuries of similar duration — you are paid extra to hold the wrong-way curvature and the prepayment uncertainty. It also makes the mortgage market a hidden amplifier of rate moves: when yields rise sharply, MBS durations *extend*, forcing mortgage investors to sell other bonds to rebalance their duration, which pushes yields up further. This "convexity hedging" feedback loop has contributed to several violent Treasury selloffs. For a trader, the takeaway is that not all convexity is a gift — in the mortgage market it is a cost, and a bond's curvature can be working for you or against you depending on what optionality is embedded in its cash flows.

## How policy moves the curve: front end versus long end

Now we connect the bond math to the central bank. A monetary-policy decision does not move all bonds equally, because it hits different parts of the yield curve through different channels. Understanding *which* part of the curve a given policy move controls is what lets you position correctly. (For the full transmission story — how a single rate decision propagates into every asset — see [monetary-policy transmission: how rate changes reach markets](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets).)

### The front end is the policy rate

The short end of the curve — the 3-month bill, the 1-year, the 2-year note — is pinned almost mechanically to the Federal Reserve's policy rate. The Fed directly sets the overnight federal funds rate, and short Treasuries trade as a near-arithmetic reflection of where that rate is and where the market expects it to be over the next year or two. The 2-year yield, in particular, is essentially the market's forecast of the *average* fed funds rate over the next two years. When the Fed hikes or signals more hikes, the 2-year jumps almost in lockstep. (For exactly how the Fed pins that overnight rate, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).)

This is why the front end is called the **policy end** of the curve. It is the part the central bank controls most directly, and because short bonds have low duration, even large yield moves there produce modest price moves. The 2-year going from 0.7% to 4.4% in 2022 — a brutal 3.7% rise — cost a 2-year holder only about 7% (1.9 duration × 3.7%), painful but survivable. The front end *moves* the most in yield terms during a hiking cycle, but it *loses* the least in price terms, precisely because its duration is short.

### The long end is expectations plus term premium

The long end — the 10-year, the 30-year — is a different animal. The Fed does not set it directly. Instead, the 10-year yield is roughly the market's expectation of the *average* short rate over the next ten years, plus a cushion called the **term premium** — extra yield investors demand for locking their money up for a decade and bearing the risk that rates, inflation, or the government's finances surprise them. So the long end moves on *expectations*: if the market believes the Fed's hikes will succeed in crushing inflation and the Fed will eventually have to cut, long yields may rise far *less* than short yields, or even fall, while the Fed is still hiking. That is exactly what produces an inverted curve.

This asymmetry — front end pinned to policy, long end floating on expectations — is why a policy move does not shift the curve in parallel. In a typical hiking cycle, the front end rockets up while the long end lags, *flattening* and eventually *inverting* the curve. The long end has the *most* duration (the 30-year's lever is the longest), so even its smaller yield moves can produce large price losses. The danger zone for a bondholder is the worst of both: a long-duration bond whose yield rises a lot. In 2022 the 10-year rose from 1.5% to over 4% — about a 2.5% move on an 8.5-duration bond, roughly a 21% loss — and the 30-year did even worse.

![Line chart of 2-year and 10-year Treasury yields from 2020 to 2026 showing the curve repricing](/imgs/blogs/how-monetary-policy-moves-bonds-duration-convexity-3.png)

The chart above shows the repricing in real data. The amber line is the 2-year yield (the policy-driven front end); the blue line is the 10-year (expectations plus term premium). Watch what the Fed's 2022 hiking cycle did: the front end *exploded* from about 0.13% in late 2020 to over 5% by late 2023, while the long end rose far less — from about 0.9% to a peak near 4.9%. The front end led, the long end followed at a fraction of the pace, and by mid-2022 the 2-year had risen *above* the 10-year (the curve inverted). The whole curve repriced upward, but the *front* did most of the moving in yield terms while the *long* end did most of the damage in price terms because of its longer duration.

#### Worked example: the curve shift and a \$1,000,000 long-bond holding

Suppose at the end of 2020 you held \$1,000,000 of a long-dated Treasury position with a duration of about 17 (a typical long-bond fund), bought when the 10-year yielded about 0.9%. Over 2021–2023 the Fed hiked, and the relevant long yield rose from roughly 1.5% to roughly 4.0% — a move of about 2.5 percentage points (250 basis points). What did that do to your holding?

Apply duration: the percentage price change is approximately −(17) × (2.5%) = **−42.5%** from the duration term. Convexity claws back a few points on a move this large — call the real loss around **−38%** after the curvature correction. On a \$1,000,000 position, that is a loss in the neighborhood of \$380,000 to \$425,000. Your "risk-free" long-bond holding lost something like 40% of its value, with zero default risk, purely because the yield it was priced at went from 1.5% to 4.0%. The data series for the front and long end ([data.UST2Y and data.UST10Y in the chart above](/imgs/blogs/how-monetary-policy-moves-bonds-duration-convexity-3.png)) show exactly this repricing. **The intuition: a long-duration bond bought at a low yield is a coiled spring — when the policy rate normalizes off the floor, the long end reprices and a holder can lose a third or more of "safe" money.**

## Common misconceptions

Three myths about bonds cause more losses than almost anything else in macro, and each one is corrected by a single number.

### "Bonds are safe, so they can't lose much"

This is the costliest myth, and 2022 demolished it. "Safe" in bond-land means *free of default risk* — the issuer will not miss a payment. It says nothing about *price risk*. A 30-year Treasury cannot default (the US government prints the currency it owes), but it can and did fall about 31% in a single year when yields rose. Safety from credit risk is not safety from interest-rate risk, and for long-duration bonds the interest-rate risk dwarfs anything an investment-grade corporate's credit risk would add. The retiree who moved into long Treasuries "for safety" in 2021 took on enormous duration risk without knowing it. **The number that corrects the myth: a default-free 30-year bond fell roughly 31% in 2022 — bigger than the S&P 500's 18% decline that same year.**

### "Duration is just the maturity"

We hit this earlier but it bears repeating because it is so common. A 10-year bond does not have a duration of 10; it has a duration closer to 8.5, because its coupons return cash before maturity and shorten the weighted-average wait. A 30-year does not have a duration of 30; it is closer to 17 to 19. The only bond whose duration equals its maturity is a zero-coupon bond, which makes a single payment at the end. Conflating maturity with duration leads you to overestimate the price sensitivity of high-coupon bonds and to misjudge the lever length entirely. **The number: a 10-year, 4%-coupon Treasury has a modified duration of about 8.5, not 10 — a 15% error if you use maturity instead.**

### "The Fed controls long yields"

The Fed sets the overnight rate, which pins the *front* end of the curve. It does *not* directly set the 10-year or 30-year yield. The long end is the market's expectation of average future short rates plus a term premium, and the market can — and routinely does — move the long end *against* the Fed's direction. In 2024, when the Fed began *cutting* the policy rate, the 10-year yield actually *rose*, because the market revised up its expectations for growth, inflation, and bond supply. A trader who assumes a Fed cut automatically lowers long yields will be repeatedly blindsided. The long end answers to expectations and the term premium, and supply matters too — large deficits mean more bonds to absorb, which pushes long yields up regardless of the policy rate. (See [deficits, debt, and bond supply: why issuance moves yields](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields).) **The number: in 2024 the Fed cut the policy rate by 1.00 percentage point while the 10-year yield rose roughly 0.7% — the long end moving opposite to policy.**

## How it shows up in real markets

The 2022 bond market is the cleanest natural experiment in fixed-income history, and it is worth walking through what duration and convexity predicted versus what happened.

### The 2022 curve shift and the long-bond crash

Coming into 2022, the curve sat near historic lows: the 2-year yielded about 0.7%, the 10-year about 1.5%, the 30-year about 1.9%. Inflation had surged to a 40-year high — the consumer price index hit **9.06%** in June 2022 — and the Fed, having been "patient" through 2021, pivoted to the fastest tightening in four decades, raising the policy rate from 0.25% to 4.50% by December 2022 and onward to 5.50% by mid-2023. (For how a hiking cycle gets *priced* before it happens, see [the terminal rate and rate-cut cycles: pricing the path](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path).)

The curve repriced violently and unevenly, exactly as the front-end-versus-long-end framework predicts. The 2-year, pinned to policy, rocketed from 0.7% to 4.4% — a 3.7% move, but on a 1.9-duration bond, "only" about a 7% price loss. The 10-year rose from 1.5% to over 4% — a 2.5% move on an 8.5-duration bond, roughly 21%. And the 30-year, with its 17-to-19 duration, took the full force: even its smaller yield move produced the biggest price collapse on the curve. Long Treasury index funds (the 20-year-plus bucket) fell about **31%** for the year — a number unprecedented in modern bond-market history and worse than the stock market's decline.

![Horizontal bar chart of 2022 returns for long Treasuries, investment grade, and high yield bonds](/imgs/blogs/how-monetary-policy-moves-bonds-duration-convexity-6.png)

The bar chart above shows the 2022 bond carnage by bucket, and the ordering proves the whole thesis. Long Treasuries (20-year-plus, duration roughly 17 to 18) fell about 31%. Investment-grade corporate bonds (duration about 8) fell about 15%. High-yield bonds (duration about 4, plus a credit cushion from their high coupons) fell about 11%. Notice the perverse ranking: the *default-free* long Treasuries lost the *most*, while the *riskier* high-yield bonds lost the *least*. Credit quality had nothing to do with it. **Duration** drove the entire outcome. The bonds with the longest levers fell the hardest, exactly as the duration math says they must. A trader who ranked these bonds by credit safety got the loss ordering exactly backwards; a trader who ranked them by duration got it exactly right.

#### Worked example: the 2022 crash on a \$500,000 long-Treasury allocation

You had allocated \$500,000 to a long-Treasury fund (the 20-year-plus bucket) at the start of 2022, reasoning that with a recession risk on the horizon, government bonds were the safe haven. By year-end, long Treasuries had returned about **−31.2%** (the figure from [data.ASSET_RETURNS_2022](/imgs/blogs/how-monetary-policy-moves-bonds-duration-convexity-6.png)). Your \$500,000 became 0.688 × \$500,000 = **\$344,000**, a loss of \$156,000 — nearly a third of the allocation gone, in the asset you bought *for safety*.

Could you have seen it coming with duration? Yes, to the basis point. The fund's duration was about 17.5. The relevant long yield rose roughly 1.8 percentage points over 2022. Duration predicts −(17.5) × (1.8%) = −31.5%, almost exactly the realized −31.2% (convexity narrowed the gap slightly). The crash was not a surprise to anyone who multiplied duration by the rate move. **The intuition: the 2022 long-bond crash was the master equation playing out in public — duration times the yield move equals the loss, and a \$500,000 "safe" allocation became \$344,000 with no default in sight.**

### When duration risk becomes a bank run: SVB, March 2023

The 2022 duration repricing did not just hurt bond funds — it broke a bank. Silicon Valley Bank had taken its flood of deposits and bought long-dated Treasuries and mortgage bonds at the low yields of 2020–2021, parking them in a "held-to-maturity" book where the paper losses were not marked on the income statement. But the losses were real. As yields rose roughly 2.5 to 4 percentage points across 2022, the duration math chewed through SVB's bond portfolio: a multi-year-duration book repriced down by tens of percent, opening an unrealized loss large enough to exceed the bank's entire equity. The bonds had not defaulted — they were Treasuries and agency mortgages, default-free. They had simply *fallen in price* by exactly what duration said they would when yields rose that much.

The bank was solvent on paper only because the losses sat in the held-to-maturity bucket. The moment depositors realized that selling those bonds to meet withdrawals would crystallize the losses and wipe out the capital, the run was on, and in March 2023 SVB collapsed in 48 hours — the second-largest bank failure in US history at the time. The lesson is brutal and exactly the theme of this post: **duration risk is real risk even when credit risk is zero.** A portfolio of the safest bonds in the world, bought at low yields and held through a hiking cycle, can lose enough value to destroy an institution, and "held to maturity" accounting hides the loss right up until someone forces a sale. Every basis point of SVB's bond losses was computable from duration in 2021 — the regulators and the bank simply chose not to mark it.

### The mirror image: when cuts come

Duration cuts both ways, which is the entire reason long bonds are a *trade* and not just a hazard. When the cycle turns and the Fed cuts — or when a recession scare collapses the long end — that same 17-duration long bond *gains* 17% for every 1% yields fall, plus a convexity bonus that makes the gain even larger than the symmetric duration estimate. The investor who buys long duration at the *top* of a hiking cycle, just as yields peak, is positioned for exactly the violent rally that the 2022 holder suffered in reverse. The long bond is a high-octane expression of a rate view: it punishes you brutally if you are early and wrong, and it pays you spectacularly if you are right that yields have peaked. Volcker's 1980 rate shock and the subsequent multi-decade bond bull is the canonical example of long duration paying off enormously once the peak was in. (See [Paul Volcker's 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation).)

## How to trade it: the playbook

Everything above turns into one disciplined process: form a rate view, then choose duration to match it. Duration is the dial that converts a directional view into a sized position, and the curve view tells you whether to trade the level or the slope.

![Matrix of bond playbook positions for falling yields, rising yields, flattening, and steepening](/imgs/blogs/how-monetary-policy-moves-bonds-duration-convexity-7.png)

The matrix above is the whole playbook on one page. Read it as: *your view* (left) picks *the position* (middle-left), the *math* (middle-right) sizes the bet, and the *invalidation* (right) tells you when you are wrong. Let me walk each row.

### Match duration to your rate view

The core move is to set your portfolio's duration to express your conviction on the *direction* of yields:

- **If you think yields will fall** (the Fed will cut, growth is slowing, inflation is rolling over), **go long duration** — buy the 10-year or 30-year. The longer the duration, the more you make per basis point of decline. A correctly-timed long-duration bet just before a cutting cycle is one of the cleanest trades in macro: an 8.5-duration 10-year gains about 8.5% for every 1% yields fall, and a 17-duration long bond nearly doubles that. The invalidation is **sticky inflation** that forces the Fed higher-for-longer — if inflation re-accelerates, the cuts you priced in evaporate and your long-duration bet bleeds.
- **If you think yields will rise** (the Fed will hike, inflation is hot, or bond supply is surging from large deficits), **go short duration** — hold cash, bills, or short-dated notes, or actively short long bonds. Short duration is the defensive crouch: a 2-year barely moves while a 30-year gets crushed, so shortening duration is how a bond manager survives a hiking cycle. The invalidation is a **growth scare** that pulls yields back down even as the Fed talks tough — at which point your short-duration defensive book underperforms the long bonds you avoided.

The discipline is to size the position by duration, not by face value. A \$1,000,000 position in a 2-year and a \$1,000,000 position in a 30-year are *not* the same bet — the 30-year carries ten times the rate risk. Always translate your view into a *duration* target, then build the position to hit it.

### Steepeners and flatteners: trading the slope

The second dimension is the *shape* of the curve, traded with two-legged positions that profit from the *difference* between two yields rather than the level of rates:

- **A steepener** profits when the curve steepens — when long yields rise relative to short yields, or short yields fall relative to long. You put it on by going long the front end (e.g., the 2-year) and short the long end (e.g., the 10-year), DV01-matched so the *level* of rates washes out and only the *slope* matters. Steepeners are the classic *early-cycle* trade: when the Fed starts cutting, the front end falls fast (it tracks policy) while the long end holds up on growth optimism, so the curve steepens and the trade pays. The invalidation is a **term-premium spike** that sells off the long end so hard the curve steepens for the *wrong* reason, against your structure.
- **A flattener** is the mirror: long the long end, short the front end, profiting when short yields rise faster than long yields. Flatteners are the classic *hiking-cycle* trade: as the Fed hikes, the front end rockets up while the long end lags (the market prices eventual cuts), so the curve flattens. The 2022 flattening was one of the cleanest flattener trades in decades. The invalidation is the **Fed pausing early** — if the front end stops rising before you expected, the flattening stalls.

The beauty of steepeners and flatteners is that they are *relative* trades: by matching the dollar-duration (DV01) of the two legs, you strip out the overall direction of rates and bet purely on the slope. This isolates a cleaner view — "I think the curve will flatten" is often a higher-conviction call than "I think rates will rise," because the slope is driven by the predictable front-versus-long mechanics of a policy cycle. (For the slope as a recession signal and how to read it, see [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession); for the balance-sheet side of policy that also moves the long end, see [QE vs QT: how balance-sheet policy moves markets](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets).)

### The one-page discipline

Put it all together and the process is mechanical:

1. **Form a rate view.** Where is the policy rate going, and what is the market already pricing? Your edge is the *difference* between your view and the priced path, not the view itself. (See [the terminal rate and pricing the path](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path).)
2. **Pick a duration to match the conviction.** Bullish on bonds (yields falling) → long duration, sized so a 1% move produces the P&L you want. Bearish → short duration. Compute the dollar P&L *before* you trade: duration × expected yield move × position size.
3. **Decide level versus slope.** A pure directional view on the *level* of rates → outright long or short duration. A view on the *shape* → a DV01-matched steepener or flattener.
4. **Account for convexity on big moves.** For small moves, duration is enough. For the kind of large repricing a policy regime change produces (the 2022 type), remember convexity is working in your favor on the long end — your loss is a bit smaller and your gain a bit bigger than the straight-line estimate.
5. **Write the invalidation.** What would tell you the view is wrong? Sticky inflation kills a long-duration bet; an early Fed pause kills a flattener. Define it before you enter, because the long bond's leverage means being wrong is expensive.

The deepest point is the one we started with: bonds are the most *direct* and most *exact* expression of a rate view in all of macro. A stock's reaction to the Fed is a guess wrapped in earnings and sentiment. A bond's reaction is a calculation — duration times the move, with a convexity correction for the big swings. When the Fed hiked in 2022 and long Treasuries fell about 31%, every basis point of that loss was sitting in the duration number the whole time, waiting to be computed. Learn duration and convexity and you stop being surprised by the bond market. You start pricing it.

## Further reading & cross-links

- [Reading the yield curve: slope, inversion, and the recession signal](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — the companion to this post: what the *shape* of the curve forecasts, and how to read the 2s10s.
- [Interest rates: the price of money and the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — why one rate sits underneath every asset price, the discount rate that duration discounts at.
- [The terminal rate and rate-cut cycles: pricing the path](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path) — how the market prices a hiking or cutting path before it happens, which is what moves the curve.
- [Deficits, debt, and bond supply: why issuance moves yields](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields) — the supply side of the long end, the other force besides expectations that the Fed does not control.
- [Monetary-policy transmission: how rate changes reach markets](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets) — the full chain from a policy decision to every asset's reaction.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — exactly how the central bank pins the overnight rate that anchors the front end.
- [Paul Volcker's 1980 rate shock](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation) — the historical bookend: the rate shock that gave way to the greatest bond bull market in history, the mirror image of 2022.
