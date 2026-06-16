---
title: "Convexity: why duration is not the whole story"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into convexity: why duration draws a straight line through a curved relationship, how the convexity correction fixes the error on big rate moves, why positive convexity is a free good and negative convexity (callables, MBS) is a trap, and where it all shows up in real markets."
tags: ["fixed-income", "bonds", "convexity", "duration", "interest-rates", "bond-pricing", "mbs", "callable-bonds", "barbell", "us-treasuries"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — duration draws a *straight line* through a relationship that is actually *curved*, and convexity is the name for that curve; it is the second-order term that duration leaves out.
> - On a small rate move, duration alone is plenty. On a big one it lies — and the bigger the move, the bigger the lie.
> - For a normal bond the curve always sits *above* the straight line, so duration **overstates losses** and **understates gains** — that one-sided error is **positive convexity**, and it is a free good.
> - The full estimate is two terms: a *duration term* (`−ModDur × Δy`) plus a *convexity correction* (`+½ × Convexity × Δy²`) that grows with the **square** of the move.
> - On a ±2% shock, duration alone misjudges our 30-year bond by **7 to 10 percentage points**; adding convexity closes almost the entire gap.
> - **Long bonds, zero-coupon bonds, and barbells** carry the most convexity; **callable bonds and mortgage bonds (MBS)** have *negative* convexity — the curve bends the wrong way, and that is a trap, not a gift.

You have probably already met duration — the number that tells you how much a bond's price moves when rates move. If a bond has a duration of 5, then a 1% rise in rates costs you about 5% of your money. It is the single most useful number in fixed income, and if you only ever learn one risk measure, learn that one. But there is a quiet problem hiding inside it, and the day you discover that problem is the day you stop being a tourist in the bond market and start being a resident.

The problem is this: duration assumes the relationship between rates and prices is a *straight line*. It is not. It is a *curve*. For small movements the straight line and the curve are so close that nobody cares about the difference. But interest rates do not always move in small, polite steps. In a crisis — March 2020, the 2022 hiking cycle, the gilt blow-up in the UK — rates lurch by one, two, three percent in weeks, and on a move that big the straight line and the curve part company by a startling amount. Duration, used naively, will tell a 30-year bondholder they have lost a third of their money when the truth is closer to a quarter; it will tell them they have gained a third when the truth is nearly a half. That gap — the difference between the line and the curve — is **convexity**, and this whole post is about it.

![The true price to yield relationship drawn as a curve sitting above a straight duration tangent line, with the vertical distance between them labeled as the convexity gap and the point where they touch marked as the tangent point](/imgs/blogs/convexity-why-duration-is-not-the-whole-story-1.png)

The diagram above is the mental model for everything that follows. The straight orange line is what duration believes: a constant slope, the same price change per unit of rate change no matter how far you travel. The blue curve is the truth: a bowed shape that pulls *away* from the line in both directions. They kiss at exactly one point — today's price, today's yield — and that is why duration feels accurate when rates barely budge. Everywhere else, the curve floats above the line, and the green double-arrows mark the gap. Convexity is the size of that gap. Hold the picture in your head; the rest is just an explanation of why the curve bends, how to measure the bend, and why the direction it bends decides whether you have a gift or a trap. (Everything here is educational, not investment advice — the goal is to understand the mechanism, not to tell you what to buy.)

## Foundations: the building blocks you need first

Let's assemble the vocabulary from zero. If you have read [the post on why bond prices move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much), this is a refresher; if not, do not skip it, because every later sentence leans on these four ideas.

**A bond is a stream of fixed cash flows.** When you buy a bond you are lending money. The borrower — the **issuer** — promises a fixed schedule: a periodic **coupon** (the interest) and then the **face value** (also called **par**, almost always \$1,000 per bond) returned at **maturity** (the final date). Our running example all post will be a plain **5-year \$1,000 par note with a 4% coupon** — it pays \$40 a year for five years, then \$1,000 back — and its big cousin, a **30-year \$1,000 par bond, also 4% coupon**. To keep the arithmetic readable we will treat coupons as annual and use annual compounding; real Treasuries pay semiannually, which shifts the third decimal place but changes none of the intuition.

**Price and yield move on a seesaw.** The coupon rate is printed on the bond and never changes. What changes every second is the **price** and, mirror-image to it, the **yield** — the single annual return that makes the bond's future cash flows worth exactly its current price. When market yields rise, the price of an existing bond *falls* (its fixed coupons are now stingy compared to new bonds), and when yields fall, its price *rises*. That inverse link is the [price–yield seesaw](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds).

**Duration is the slope of that seesaw.** Duration answers "by how much?" There are two flavors worth naming. **Macaulay duration** is the weighted-average time until you get your money back, measured in years. **Modified duration** (the one we use for price moves, written `ModDur`) is Macaulay duration adjusted slightly, and it has a beautifully concrete meaning: it is the *percentage price change for a 1% change in yield*. Our 5-year bond has a modified duration of about **4.45**; our 30-year bond, about **17.29**. So duration says: nudge yields up 1% and the 5-year loses ~4.45%, the 30-year ~17.29%.

**A basis point** is one hundredth of a percent — 0.01%. Rates and spreads are quoted in *basis points* ("bps") because the moves are small: a 25 bps hike is a quarter of one percent. When we say "a 2% rate move," that is 200 basis points — a large move, the kind that only happens in cycles and crises, and exactly the kind where convexity starts to matter.

With those four in hand, here is the one sentence that motivates the entire post: **modified duration is the slope of the price–yield curve at one single point, and a slope is a straight line, but the curve is not straight.** Duration is the tangent. Convexity is everything the tangent misses.

## Why the relationship is a curve, not a line

Why is the price–yield relationship curved at all? The cleanest way to see it is to remember *how* a bond is priced: every future cash flow is divided by `(1 + y)` raised to the power of how many years away it is. That is [discounting](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced), and the key feature is that dividing by a power is not a linear operation — it bends.

Think about a single \$1,000 payment due 30 years out. At a 4% yield it is worth \$1,000 / 1.04³⁰ = about \$308. Now cut the yield in half, to 2%: it is worth \$1,000 / 1.02³⁰ = about \$552 — it nearly *doubled*. But double the yield instead, to 8%: it is worth \$1,000 / 1.08³⁰ = about \$99 — it fell by only ~\$209, less than it rose. The same-sized move in yield, up versus down, does **not** produce the same-sized move in price. The downside is bounded (price can only fall toward zero), but the upside is not (price can keep climbing as the discount rate shrinks toward zero). That asymmetry is the curve. A straight line, by definition, treats up-moves and down-moves as perfect mirror images; the real bond does not, and the gap between the symmetric line and the asymmetric truth is convexity.

#### Worked example: the same yield move is not symmetric

*Setup.* Take our 30-year bond, priced at par: \$1,000 at a 4% yield. Modified duration is 17.29, so duration's straight-line rule says every 1% move in yield is worth ~17.29% of price, in either direction.

*Step 1 — what duration predicts.* Yields drop 1%? Duration says price rises 17.29%, to about \$1,173. Yields rise 1%? Duration says price falls 17.29%, to about \$827. Symmetric: +\$173 up, −\$173 down.

*Step 2 — what actually happens.* Reprice the bond properly. At a 3% yield it is worth about \$1,196 — a gain of \$196, *more* than the \$173 duration promised. At a 5% yield it is worth about \$846 — a loss of \$154, *less* than the \$173 duration threatened. The real gain beat the estimate; the real loss undershot it.

*Step 3 — read the asymmetry.* You gained \$196 but only lost \$154 on the same-sized move in opposite directions. That \$42 of "extra good news" did not come from luck. It is the curve. *Positive convexity means the curve always treats you better than the straight line — bigger gains than promised, smaller losses than threatened.*

### A picture you already know: the speed-and-acceleration analogy

If the math feels abstract, borrow an analogy from physics, because the relationship is exactly the same. Imagine a bond's price as a car's position, and yield as time ticking forward. **Duration is the car's speed** — how fast position changes right now. **Convexity is the car's acceleration** — how fast the speed itself is changing. If you knew only the car's current speed and tried to predict where it will be in ten seconds, you would be roughly right for a one-second prediction and badly wrong for a ten-second one, because the car is accelerating the whole time and your speed-only guess never accounts for it. That is duration without convexity: a speed reading used to predict a long journey.

The reason this matters is that bonds are *always accelerating*. As yields fall, a bond's price does not just rise — it rises *faster and faster*, because its duration is itself increasing as yields drop. As yields rise, the price falls but at a *decelerating* rate, because duration shrinks. Duration captures the instantaneous speed; convexity captures the fact that the speed is constantly changing. A risk model that uses duration alone is a navigator who knows the boat's current speed but has forgotten that the engine is throttling up. For a short hop that is fine. For a long voyage through a storm, it puts you on the rocks.

This analogy also explains, in one stroke, why convexity is *always positive for a normal bond*. Acceleration toward higher prices when rates fall, and decelerating losses when rates rise, are the *same physical fact* seen from two sides — the curve bends one consistent way. It takes an embedded option, working against you, to make a bond *decelerate its gains* and *accelerate its losses*, which is the negative-convexity case we will meet later. For a plain vanilla bond, the engine only ever throttles in your favor.

### Foundations recap: the three numbers that describe any bond's rate risk

Before we compute, lock in the hierarchy, because almost every confusion about convexity comes from blurring these three levels into one. Think of them as the zeroth, first, and second facts about a bond's price as yields change.

- **The price itself** (the zeroth-order fact): what the bond is worth right now, at today's yield. One number, one point on the curve.
- **Duration** (the first-order fact): the *slope* of the curve at that point — how fast price changes for a small change in yield. Measured so that modified duration reads directly as "percent price change per 1% yield change."
- **Convexity** (the second-order fact): the *curvature* at that point — how fast the slope itself changes as you move along the curve. It is what turns the straight tangent into a bow.

Every later sentence is just these three in conversation. Price tells you where you stand; duration tells you which way and how steeply the ground falls away; convexity tells you that the ground is curved, so the steepness you measured will not hold as you walk. You need all three to navigate, and the further you intend to walk — the bigger the rate move — the more the third one matters.

## The formula: duration plus a convexity correction

So how do we turn "the curve floats above the line" into a number we can actually compute? With a two-term formula. Duration gives the first, straight-line piece; convexity gives the curved correction you add back. Written out, the percentage price change for a yield change of `Δy` is:

$$
\frac{\Delta P}{P} \;\approx\; \underbrace{-\,\text{ModDur}\times \Delta y}_{\text{duration term}} \;+\; \underbrace{\tfrac{1}{2}\times \text{Convexity}\times (\Delta y)^2}_{\text{convexity correction}}
$$

Let's name every symbol. `ΔP / P` is the percentage change in price we are solving for. `ModDur` is modified duration — the slope. `Δy` is the change in yield, written as a decimal (a 2% move is `Δy = 0.02`). `Convexity` is the curvature number, the bond's second-order sensitivity (the math is the same discounting, differentiated twice; we will treat it as a given number per bond). The first term is the duration estimate you already know. The second term is the new piece.

![A four step pipeline showing a rate move feeding into a duration term that is minus modified duration times the yield change, then a convexity term that is one half times convexity times the yield change squared, then the two summing into the total price change](/imgs/blogs/convexity-why-duration-is-not-the-whole-story-2.png)

Two features of that second term do all the work, and the pipeline above traces them. First, the convexity term has `(Δy)²` in it — the *square* of the rate move. Squaring a small number makes it tiny (0.01² = 0.0001) and squaring a big number keeps it meaningful (0.03² = 0.0009, nine times larger). That is precisely why convexity is irrelevant for small moves and decisive for large ones: the correction grows with the square of the shock while the duration term grows only linearly. Second — and this is the gift — for a normal bond the convexity number is *positive*, and `(Δy)²` is always positive (a square cannot be negative), so the entire convexity correction is **positive whether rates rise or fall**. It *adds* to your gains when rates drop and *subtracts from your losses* when rates rise. The straight line is wrong in the same friendly direction both ways.

#### Worked example: the 5-year bond on a 2% move (where convexity is small)

*Setup.* Our 5-year 4% bond, priced at par (\$1,000) at a 4% yield. Modified duration 4.45, convexity 25. Shock: a 2% move, so `Δy = 0.02`.

*Step 1 — duration term.* `−4.45 × 0.02 = −0.0890`, i.e. −8.90%. For a rate *cut* of 2% it is +8.90%.

*Step 2 — convexity correction.* `½ × 25 × 0.02² = ½ × 25 × 0.0004 = 0.0050`, i.e. +0.50%. Positive either way.

*Step 3 — combine and check.* Rates +2%: −8.90% + 0.50% = **−8.40%**. The true reprice is −8.42%. Rates −2%: +8.90% + 0.50% = **+9.40%**; the true reprice is +9.43%. Duration alone was off by about half a percentage point in each direction — small enough that most people would never bother. *For a short bond, convexity is a rounding error; you can almost ignore it.*

#### Worked example: the 30-year bond on the same 2% move (where convexity is large)

*Setup.* Now the 30-year 4% bond, also at par (\$1,000) at 4%. Modified duration 17.29, convexity **420** — sixteen times the 5-year's convexity. Same shock, `Δy = 0.02`.

*Step 1 — duration term.* `−17.29 × 0.02 = −0.3458`, i.e. −34.58%. A rate cut of 2% gives +34.58%.

*Step 2 — convexity correction.* `½ × 420 × 0.02² = ½ × 420 × 0.0004 = 0.0840`, i.e. +8.40%. That is not a rounding error — it is bigger than many bonds' entire annual return.

*Step 3 — combine.* Rates +2%: −34.58% + 8.40% = **−26.18%**. Rates −2%: +34.58% + 8.40% = **+42.98%**.

*Step 4 — compare to the truth.* The true reprice at 6% is −27.5%; at 2% it is +44.8%. Duration *alone* said −34.6% and +34.6%. So on the way up, duration overstated the loss by 7 percentage points; on the way down it understated the gain by more than 10. Adding the convexity term cut the error to about a point. *On a long bond and a big move, ignoring convexity is not a rounding error — it is a different answer.*

### What the convexity number actually means

Duration has a satisfyingly concrete unit: years, and "percent per percent." Convexity's number — 25 for the 5-year, 420 for the 30-year, 860 for the 30-year zero — looks more mysterious. What *is* a convexity of 420? Here is the honest, useful answer: the raw number has units of "years squared," and on its own it is not meant to be read directly. It only becomes a price effect after you multiply it by `½` and by `(Δy)²` in the formula. The number's job is to be a *multiplier on the squared rate move*, and the right way to build intuition for it is to do exactly that and see what falls out.

Take the convexity term, `½ × Convexity × (Δy)²`, and feed it a 1% move (`Δy = 0.01`). For the 5-year (convexity 25): `½ × 25 × 0.0001 = 0.00125`, about +0.13%. For the 30-year (convexity 420): `½ × 420 × 0.0001 = 0.021`, about +2.1%. For the 30-year zero (convexity 860): `½ × 860 × 0.0001 = 0.043`, about +4.3%. So the convexity number, once you run it through the formula at a standard 1% move, translates into a "convexity bonus" of a tenth of a percent, two percent, or four percent respectively — and that bonus is what you add to your gains and subtract from your losses. That is the practical reading: *convexity is how many extra percentage points of one-sided help you get, scaled by the square of how far rates travel.*

Two scaling rules follow immediately, and they are worth memorizing because they let you reason about convexity without ever opening a spreadsheet. First, **double the rate move and the convexity bonus quadruples** — because it depends on `(Δy)²`. A 2% move gives four times the convexity help of a 1% move; a 3% move gives nine times. Second, **the convexity number itself scales roughly with maturity squared**, so going from a 5-year to a 30-year (six times the maturity) multiplies convexity by roughly thirty-six — close to the sixteen-to-seventeen-fold jump we actually see once coupons and discounting are accounted for. Both rules are the same `squared` relationship showing up in two places: convexity squares with maturity, and its effect squares with the size of the move. When both are large at once — a long bond in a big move — the two squarings compound, and that is the regime where duration alone becomes genuinely dangerous.

#### Worked example: turning a raw convexity number into a dollar cushion

*Setup.* You hold \$50,000 face of the 30-year bond, currently worth \$50,000 (at par). Convexity 420. You want to know, in dollars, how much cushion convexity gives you if rates rise 1.5% (a 150 bps shock).

*Step 1 — the convexity term as a percentage.* `½ × 420 × 0.015² = ½ × 420 × 0.000225 = 0.04725`, about +4.7%.

*Step 2 — turn it into dollars.* 4.7% of your \$50,000 position is about +\$2,360. That is the size of the cushion: the amount by which convexity makes your loss *smaller* than the straight-line duration estimate.

*Step 3 — sanity check against duration.* Duration alone (`−17.29 × 0.015`) projects a −25.9% loss, about −\$12,960. The true loss, once you add the +\$2,360 convexity cushion, is closer to −\$10,600 (a −21.2% move). *The "years squared" number nobody can interpret on its own just became \$2,360 of real money you would have miscounted if you stopped at duration.*

## The influence: how convexity changes the rate-to-P&L relationship

Here is the part that matters for the wider world, and it is the heart of why duration "is not the whole story." Convexity does not just tweak a number; it *changes the shape of the relationship* between how far rates move and how much money you make or lose. Duration says that relationship is a straight line — double the rate move and you exactly double the P&L, in a clean mirror. Convexity says no: the relationship is bent, and the bend matters more and more as the move gets bigger.

![A chart with the size of the rate shock on the horizontal axis from minus three percent to plus three percent and the resulting price change on the vertical axis, showing the true repricing as a curve and the duration only estimate as a straight line, the two touching at zero and fanning apart at the extremes](/imgs/blogs/convexity-why-duration-is-not-the-whole-story-4.png)

The chart above plots both stories for the 30-year bond. The horizontal axis is the *size of the rate shock*, from a 3% cut on the left to a 3% rise on the right. The vertical axis is the resulting price change. The orange straight line is the duration-only estimate; the blue curve is the true repricing. Look at what happens as you walk outward from the center. At the middle — no rate change — the two lines touch: duration is perfect when nothing moves. Take a small step in either direction and they stay almost glued together: for a 0.25% wiggle the error is invisible. But keep walking. By a 2% move the lines have visibly separated; by 3% they have fanned wide apart. And notice the *direction* of the error is always the same: the blue curve is *above* the orange line on both sides. On the downside (rates falling, left) that means your true gain is bigger than duration promised. On the upside (rates rising, right) it means your true loss is smaller than duration threatened. The green arrows mark the gap at the extremes — about +10 percentage points of extra gain on the −3% side, about +7 points of avoided loss on the +3% side.

That is the influence in one picture: **convexity is a one-sided friend whose help grows with the size of the move.** It does nothing for you in calm markets and a great deal for you in violent ones. This is exactly backwards from most risks, which hurt more the wilder things get; positive convexity *helps* more the wilder things get. It is why long-duration bonds, despite their terrifying duration numbers, can be a surprisingly good thing to own going into a crisis: if rates collapse in a flight to safety, the convex curve hands you more upside than the duration math ever suggested.

#### Worked example: a violent crisis move, with and without convexity

*Setup.* Imagine a flight-to-safety panic: the 30-year Treasury yield collapses by 3% (300 bps) in a few weeks, from 4% to 1%. You hold our 30-year bond, bought at \$1,000. ModDur 17.29, convexity 420, `Δy = −0.03`.

*Step 1 — duration says.* `−17.29 × (−0.03) = +0.5187`, i.e. +51.9%, a price of about \$1,519.

*Step 2 — add convexity.* `½ × 420 × (−0.03)² = ½ × 420 × 0.0009 = 0.189`, i.e. +18.9%. Total: +51.9% + 18.9% = **+70.8%**, about \$1,708.

*Step 3 — the true reprice.* Mark the bond at a 1% yield and it is worth about \$1,774, a gain of +77.4%. (The two-term formula slightly undershoots the true value on a move this enormous because convexity itself keeps growing, but it captures the lion's share.)

*Step 4 — the lesson.* Duration alone would have told you to expect a +52% gain. The bond actually delivered +77%. Convexity was worth an extra **25 percentage points** in the one scenario where you most wanted the help. *In a crisis, the convexity you were ignoring shows up as the gift you did not know you owned.*

## Positive versus negative convexity: the gift and the trap

Everything so far assumed the curve bends the *good* way — bowing up and away from the line, helping you in both directions. That is **positive convexity**, and ordinary bonds (Treasuries, plain corporate bonds, anything with a fixed schedule the issuer cannot change) all have it. But some bonds bend the *other* way, and for those bonds convexity flips from a gift into a trap.

![Two price to yield curves on the same axes, a normal bond curving upward with positive convexity and a callable bond or mortgage backed security that flattens into a price ceiling at low yields, illustrating negative convexity, with a dotted line marking the ceiling where the bond is called away](/imgs/blogs/convexity-why-duration-is-not-the-whole-story-3.png)

The chart above puts both kinds side by side. The green curve is a normal bond — the familiar bow, climbing ever faster as yields fall toward the left. The red curve is a **callable bond** or a **mortgage-backed security (MBS)**, and notice how it *flattens* at low yields instead of soaring. It runs into a ceiling. That ceiling is the whole story of **negative convexity**.

Why does it happen? Because of an embedded option that belongs to *the other side*. A **callable bond** gives the *issuer* the right to buy the bond back early at a set price — to "call" it. A **mortgage** gives the *homeowner* the right to refinance — to pay off the old loan early and take out a cheaper one. In both cases, the right to repay early is held by the borrower, and the borrower will exercise it at exactly the worst moment for you: when rates fall. When rates drop, a normal bond's price would soar; but the issuer of a callable bond simply calls it back at par (say \$1,000-ish) and reissues cheaper, and the homeowner refinances, handing your MBS its principal back early. Either way, you do not get the soaring price — your upside is *capped near the call price*. The bond can still fall freely when rates rise (no one is going to refinance a cheap mortgage when rates are high), so you keep the full downside and lose the upside. The curve bends down where it should bend up.

#### Worked example: a callable bond at a falling yield

*Setup.* Northwind Corp issues a 30-year bond, 4% coupon, callable at \$1,020 any time after year five. You buy it at par, \$1,000. Rates then fall sharply, to 2%.

*Step 1 — what a non-callable Northwind bond would do.* Reprice a plain 30-year 4% bond at a 2% yield and it is worth about \$1,448 — a 44.8% gain. That is the upside a normal bond hands you.

*Step 2 — what the callable bond actually does.* Northwind looks at the market, sees it can reissue 30-year debt at 2% instead of 4%, and calls your bond at \$1,020. You get \$1,020, not \$1,448. Your "gain" is +2%, not +44.8%.

*Step 3 — the asymmetry.* Now imagine rates had risen to 6% instead. Northwind would *not* call (why retire cheap 4% debt?), and your bond would fall to about \$725 — the full −27.5% loss, just like a normal bond. So you eat the entire downside but your upside is chopped off near \$1,020. *Negative convexity means heads the issuer wins, tails you lose — you are short the option, and the option is exercised against you exactly when you would have profited most.*

This is why investors *demand a higher yield* to hold callable bonds and MBS: they are being paid a premium to write that option. The extra yield is the price of the negative convexity. Whether it is worth it depends entirely on how violently rates move — which is a way of saying that with negative-convexity bonds, you are quietly selling volatility, and you find out the price the next time rates lurch.

### Effective duration and effective convexity: measuring a bond that can change shape

There is a subtle but important problem lurking in the callable example. The ordinary duration and convexity numbers we have been using are computed from a *fixed* schedule of cash flows — they assume the bond pays exactly what it promised, on exactly the dates it promised. That assumption is fine for a Treasury, which cannot change its mind. It is *false* for a callable bond or an MBS, whose cash flows actually change when rates change: the bond gets called, the mortgage gets refinanced, the schedule reshuffles. You cannot measure the rate risk of a shape-shifting bond with a formula that assumes the shape is fixed.

The fix is to measure duration and convexity *empirically* rather than from a formula — by actually repricing the bond (with a proper model of the embedded option) at a slightly higher yield and a slightly lower yield, and reading the slope and curvature off those repriced values. These are called **effective duration** and **effective convexity**. The recipe is simple in spirit: bump the yield up by a small amount, reprice; bump it down by the same amount, reprice; the difference between the two repriced values (relative to the move) is the effective duration, and the way that difference itself changes is the effective convexity. For a plain bond, effective duration equals ordinary modified duration — nothing new. For an option-embedded bond, they diverge sharply, and the effective convexity comes out *negative*, which is the formula's way of confirming that the curve bends the wrong way.

#### Worked example: effective convexity reveals the hidden trap

*Setup.* You are handed two bonds quoted at the same price (\$1,000) and the same modified duration on paper (say 7). One is a plain 8-year Treasury; the other is a callable corporate. You want to know which one is dangerous.

*Step 1 — bump rates down 1%, reprice both.* The Treasury rises to about \$1,073 (a clean +7.3%, slightly more than 7% because of its positive convexity). The callable rises to only about \$1,015 — the call ceiling caps it.

*Step 2 — bump rates up 1%, reprice both.* The Treasury falls to about \$933 (a −6.7% loss, slightly less than 7% — positive convexity cushioning again). The callable falls to about \$928 — almost the full −7.2%, because no one calls a bond when rates rise.

*Step 3 — read the convexity.* The Treasury gained more than it lost (+7.3% vs −6.7%): positive effective convexity. The callable gained *less* than it lost (+1.5% vs −7.2%): negative effective convexity. Same headline duration, opposite curvature.

*Step 4 — the lesson.* On paper they looked identical. Repriced at the extremes, one is a friend and the other is a trap. *Effective duration and effective convexity are how you catch a negative-convexity bond that an ordinary-formula risk report would have mislabeled as harmless.*

## What gives a bond more convexity: maturity, coupon, and dispersion

If convexity is so important on big moves, the natural next question is: which bonds have a lot of it, and which have a little? Three levers control it, and all three trace back to the same root cause — *how spread out in time a bond's cash flows are.*

![A chart with maturity in years on the horizontal axis and convexity on the vertical axis, showing a zero coupon bond curve climbing steeply and far above a lower eight percent coupon bond curve, both rising faster than a straight line as maturity lengthens](/imgs/blogs/convexity-why-duration-is-not-the-whole-story-5.png)

The chart above shows the two strongest levers at once. **Maturity** is on the horizontal axis, and both curves climb — convexity rises as maturity lengthens, and it rises *faster than linearly* (the curves bend upward themselves). A 5-year bond has convexity around 25; a 30-year, around 420 — not six times more for six times the maturity, but roughly *seventeen* times more. Convexity is roughly proportional to maturity *squared*, which is why long bonds are convexity powerhouses. The second lever is **coupon**: the blue curve is a zero-coupon bond and the orange dashed curve is an 8% coupon bond of the same maturities, and the zero sits far above. Lower coupon means more convexity. A **zero-coupon bond** — one that pays no coupons at all, just a single lump at maturity — has the most convexity of any bond of its maturity, because *all* of its cash is concentrated at the most distant, most curvature-sensitive point. Pile coupons in along the way and you pull the bond's center of gravity earlier, which shrinks both duration *and* convexity.

The third lever does not fit on that chart because it is about *spread*, not a single bond: for a given duration, the more *dispersed* a portfolio's cash flows are across time, the more convexity it has. This is the principle behind the **barbell**.

#### Worked example: which of three bonds is most convex

*Setup.* Three bonds, all at a 4% yield: (A) a 5-year 4% coupon bond, (B) a 30-year 4% coupon bond, (C) a 30-year zero-coupon bond.

*Step 1 — rank by maturity.* The two 30-year bonds will dwarf the 5-year. The 5-year's convexity is ~25; both 30-years are in the hundreds.

*Step 2 — break the tie with coupon.* Between the two 30-years, the zero wins. The 30-year 4% bond has convexity ~420; the 30-year zero has convexity ~860, roughly double, because none of its money arrives early. Its price is only about \$308 today (one payment, 30 years out, heavily discounted), but its *sensitivity* to rates is the most extreme of the three.

*Step 3 — the ranking.* Most convex to least: 30-year zero (≈860) ≫ 30-year 4% (≈420) ≫ 5-year 4% (≈25). *To stack convexity, push your money as far into the future and as concentrated at the end as you can — long and low-coupon is the recipe.*

## Harvesting convexity for free: the barbell versus the bullet

Here is where convexity stops being a passive property and becomes something you can deliberately *buy* — and, remarkably, sometimes for free. The trick exploits the fact that duration is a simple weighted average (it mixes linearly) but convexity, being a squared effect, rewards *spreading cash flows apart*.

Compare two portfolios with the *same duration*. A **bullet** puts all your money in a single intermediate bond — say one 10-year bond. A **barbell** splits the money between a very short bond and a very long one — heavy on each end, nothing in the middle, like the weights on a barbell. You can choose the barbell's mix so its *duration exactly matches the bullet's*. But because convexity grows with maturity squared, the long leg of the barbell contributes a hugely disproportionate amount of convexity, far more than the short leg gives up. Same duration, more convexity.

![A before and after comparison showing a bullet portfolio of one ten year bond on the left with a modified duration of eight point one and convexity of eighty one, versus a barbell of sixty percent two year and forty percent thirty year bonds on the right with the same modified duration of eight point one but convexity of one hundred seventy three, more than double](/imgs/blogs/convexity-why-duration-is-not-the-whole-story-7.png)

The comparison above makes it concrete. On the left, the bullet: 100% in a single 10-year bond. Its modified duration is 8.1 and its convexity is 81. On the right, the barbell: 60% in a 2-year bond and 40% in a 30-year bond. Its modified duration is *also* 8.1 — identical rate sensitivity, by construction — but its convexity is **173**, more than double the bullet's. The two portfolios will respond almost identically to small rate moves (same duration), but on a big move the barbell's extra convexity hands it bigger gains and smaller losses. You got more of the one-sided friend for the same duration risk.

#### Worked example: barbell versus bullet on a big swing

*Setup.* Two \$10,000 portfolios at a 4% yield. Bullet: \$10,000 in a 10-year 4% bond (ModDur 8.1, convexity 81). Barbell: \$6,000 in a 2-year 4% bond and \$4,000 in a 30-year 4% bond (blended ModDur 8.1, convexity 173). Shock: rates fall 2%, `Δy = −0.02`.

*Step 1 — duration term (identical for both).* `−8.1 × (−0.02) = +0.162`, i.e. +16.2%, about +\$1,620 for each portfolio. Same duration, same first-order move.

*Step 2 — convexity correction (where they differ).* Bullet: `½ × 81 × 0.02² = 0.0162`, +1.62%, about +\$162. Barbell: `½ × 173 × 0.02² = 0.0346`, +3.46%, about +\$346.

*Step 3 — total.* Bullet gains ~+17.8% (+\$1,782); barbell gains ~+19.7% (+\$1,966). The barbell made about \$184 more on a \$10,000 position from nothing but its shape.

*Step 4 — and on the way down?* If rates had instead risen 2%, the same convexity terms are still *positive*, so the barbell loses *less*: its larger convexity cushions the fall by an extra ~1.8 percentage points. *Same duration, but the barbell is better in both directions on a big move — that is the convexity edge, and in a fair market you pay for it with a slightly lower yield.*

That last clause is the catch, and it is worth saying plainly: convexity is rarely *truly* free. In a well-functioning market the barbell will usually yield a touch less than the bullet — the market charges you for the extra convexity through a slightly lower running income. Convexity is most valuable when rates are volatile and least valuable when they are calm, so the barbell-versus-bullet choice is really a bet on volatility: take the barbell when you expect big moves, take the bullet (and pocket the extra yield) when you expect quiet.

## Convexity as a tradable position: the cost of carry and the link to options

For most investors convexity is a property they hold passively. For professional rates traders it is a position they put on and take off deliberately, and seeing it through their eyes sharpens the whole concept. The key reframing is this: **convexity is the bond market's version of an option's gamma.** An option's value, like a bond's price, is a curved function of the thing it depends on, and the trader who is "long gamma" makes money on big moves in either direction and pays for the privilege through time decay. Long convexity is the same trade in rates: you profit from large yield moves in either direction, and you pay for it through a lower yield — the bond-market equivalent of time decay. Long convexity is, almost literally, being long an option on interest-rate volatility.

That framing dissolves a lot of confusion. Why does the barbell yield less than the bullet? For the same reason an option costs a premium: you are buying optionality, and optionality is never free. Why is negative convexity *paid* a higher yield? Because the MBS or callable holder is *short* the option — they collect a premium (the extra yield) in exchange for handing the other side the right to reshape their cash flows. The entire fixed-income world can be read as a marketplace where convexity is bought and sold, with yield as the price tag: positive convexity costs yield, negative convexity earns it, and the fair exchange rate between them is set by how volatile everyone expects rates to be.

This is also where the **cost of carry** becomes vivid. If you load up on convexity by holding long-duration, low-coupon bonds and rates simply sit still for two years, you have paid the premium — the lower yield you accepted — and collected nothing for it, because convexity only pays off on *moves*. The long-convexity trader, like the long-gamma options trader, is in a quiet race against the clock: every calm day, the carry bleeds a little; every violent day, the convexity gushes. Whether the trade wins depends entirely on whether the realized volatility of rates exceeds the volatility that was priced in when you paid for the convexity. That is why rates desks talk about convexity and implied volatility in the same breath — they are two views of the same wager.

#### Worked example: did the convexity pay for its carry?

*Setup.* You buy our 30-year bond (convexity 420) instead of an equal-duration bullet, accepting a yield that is 0.20% (20 bps) lower as the price of the extra convexity. You hold for one year. Position: \$100,000.

*Step 1 — the carry cost.* Giving up 0.20% of yield on \$100,000 for a year costs you about \$200 in forgone income. That is your "premium."

*Step 2 — the calm scenario.* Rates barely move all year — say a 0.25% wiggle. The convexity bonus is `½ × 420 × 0.0025² ≈ 0.00013`, about +0.013%, or +\$13. You paid \$200 and collected \$13. The convexity trade *lost* about \$187 — the carry ate it.

*Step 3 — the volatile scenario.* Instead, rates swing 2% during the year. The convexity bonus is `½ × 420 × 0.02² = 0.0084`, about +\$840. You paid \$200 and collected \$840 of one-sided help. The convexity trade *won* by about \$640.

*Step 4 — the takeaway.* Same bond, same premium, opposite outcomes, decided entirely by how much rates actually moved. *Long convexity is a bet on volatility wearing a bond's clothing; you win it in storms and lose it in calm, exactly like being long an option.*

## Reading the full table: how the pieces fit on the 30-year bond

Let's collect the 30-year bond's behavior into one place, because seeing the whole accounting at once is what makes convexity click. The table below decomposes a ±2% move into its parts: what duration alone says, what the convexity term adds, what the two together predict, and what the bond is actually worth when you reprice it properly.

![A table for the thirty year bond on a two percent rate move with rows for rates rising and falling and columns for duration only, convexity add, duration plus convexity, and true reprice, showing duration alone overstates the loss by seven points and understates the gain by ten while adding convexity closes almost the entire gap](/imgs/blogs/convexity-why-duration-is-not-the-whole-story-6.png)

Read the table above row by row. On the top row, rates rise 2%: duration alone screams −34.6%, but the convexity term adds back +8.4%, landing at −26.2%, within a hair of the true −27.5%. On the bottom row, rates fall 2%: duration says +34.6%, convexity adds the same +8.4% (it is positive both ways, remember), and the duration-plus-convexity estimate of +43.0% sits close to the true +44.8%. The single most important thing in the table is the middle "convexity add" column: it is **+8.4% in both rows**. That is the visual proof that positive convexity is one-sided — it shrinks the loss and swells the gain by the same amount, every time. Duration is the symmetric guess; convexity is the asymmetric correction that always tilts in your favor.

#### Worked example: estimating a price in your head with both terms

*Setup.* Someone tells you the 30-year yield just jumped 1.5% (150 bps), from 4% to 5.5%. You hold our 30-year bond at \$1,000 and want a fast mental estimate. ModDur 17.29 (call it ~17), convexity 420 (call it ~400), `Δy = 0.015`.

*Step 1 — duration term, roughly.* `−17 × 0.015 = −0.255`, about −25.5%.

*Step 2 — convexity correction, roughly.* `½ × 400 × 0.015² = ½ × 400 × 0.000225 = 0.045`, about +4.5%.

*Step 3 — combine.* −25.5% + 4.5% ≈ **−21%**, a price near \$790. The true reprice is about \$782, a −21.8% move. Your two-term mental math landed within a percentage point.

*Step 4 — the takeaway.* The duration term alone (−25.5%) would have had you bracing for a much worse loss than reality. *Carrying the convexity term in your head turns a scary, wrong estimate into a calm, accurate one — that is the practical payoff of not stopping at duration.*

## Common misconceptions

**"Duration is wrong, so I should ignore it."** No — duration is the *first and biggest* term, right far more often than it is wrong. For small moves it is essentially exact, and even for large moves it captures most of the answer. Convexity is a *correction*, not a replacement. The mistake is not using duration; the mistake is stopping there when the move is big or the bond is long.

**"Convexity is some exotic thing only quants care about."** Convexity is just the curvature of a relationship every bondholder already lives with. Anyone who owns a long bond fund owns convexity whether they have heard the word or not. In a sharp rate rally it quietly makes them more money than the headline duration suggested; in a callable bond or a mortgage fund, it quietly *costs* them. Not knowing the word does not make you immune to the effect.

**"More convexity is always better, so I should chase it."** Positive convexity is a genuine good, but in a fair market it is *priced* — you usually pay for it with a lower yield. Buying convexity is a bet that rates will be volatile enough for the curvature to earn back the yield you gave up. In a long, calm stretch, the high-convexity barbell can *underperform* the plain bullet, because you paid for an insurance that never paid out.

**"A bond with negative convexity is just a bad bond to avoid."** Negative convexity is a *feature with a price*, not a defect to flee. Callable bonds and MBS pay you extra yield precisely because you are absorbing their negative convexity — you are being compensated for selling the issuer or homeowner an option. For an investor who genuinely believes rates will stay range-bound, harvesting that extra yield can be perfectly rational. The danger is owning it *without knowing*, so that the capped upside ambushes you in the next rate rally.

**"Convexity and duration are alternatives — pick one risk number."** They measure different things and you need both. Duration is the slope; convexity is the curvature. A bond can have low duration and high convexity (a barbell), or high duration and modest convexity (a bullet). Quoting one without the other is like describing a car by its speed but not whether it is accelerating.

**"Since convexity helps in both directions, it has no downside for a normal bond."** Correct for the *price math* — positive convexity genuinely helps both ways. The downside is the opportunity cost: the yield you forgo to hold it. There is no free lunch in an efficient market, only a lunch whose price is a slightly lower coupon. The convexity is real; so is the bill.

**"The convexity formula is exact, so I can trust it on any move."** The two-term formula is an *approximation* — a Taylor expansion truncated after the second term. It is excellent for moves up to a percent or two and visibly undershoots on enormous moves (you saw it miss the +77% crisis reprice, landing at +71%), because the true curve has even higher-order curvature the formula throws away. For everyday risk it is more than enough; for stress-testing a 4% or 5% shock, reprice the bond properly rather than trusting any closed-form shortcut. Duration is the first approximation, convexity the second, and the true reprice is the only thing that is actually correct.

## How it shows up in real markets

**The 2022 hiking cycle and the long-bond bloodbath.** When the Federal Reserve raised rates from near zero to over 5% across 2022, long-duration Treasuries had one of their worst years in history — the longest-maturity Treasury index fell on the order of 30%+. Duration explained most of the carnage, but convexity quietly *softened* it: because the move was a *rise*, the positive-convexity correction subtracted from the loss. Holders of 30-year bonds lost a great deal, but slightly *less* than a naive duration-times-yield-change calculation would have projected. Convexity does not save you from a rate rise; it just makes the loss a little smaller than the straight line threatens. (See the broader episode in [the 2022 case study where stocks and bonds both fell](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).)

**March 2020 and the convexity gift in a flight to safety.** In the COVID panic, Treasury yields collapsed in days as the world scrambled for safe assets. Long Treasuries soared — and here convexity worked the friendly way, *adding* to gains beyond what duration alone predicted. A 30-year position rallied harder than its duration math suggested, exactly the asymmetry the influence chart promised: the bigger and faster the rate fall, the more the curvature paid. This is the core reason long government bonds are prized as a [crisis hedge and "risk-free anchor"](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) — their convexity makes them most generous precisely when everything else is bleeding.

**The MBS market and the convexity-hedging feedback loop.** The US mortgage market is the world's great laboratory of *negative* convexity, and it has real macro consequences. When rates fall, millions of homeowners refinance, shortening the duration of mortgage portfolios held by banks, insurers, and agencies; to stay hedged, those holders must *buy* Treasuries, which pushes yields down further — a self-reinforcing rally. When rates rise, refinancing stops, mortgage durations *extend*, and holders must *sell* Treasuries to rebalance, pushing yields up further. This "convexity hedging" by MBS holders has, on several occasions (notably the 2003 and 2013 episodes), measurably amplified Treasury moves. Negative convexity is not just an investor's problem; it can become the whole market's accelerant.

**The UK gilt crisis of 2022 and leveraged duration.** When UK long-gilt yields spiked after the September 2022 "mini-budget," pension funds running leveraged liability-driven strategies faced collateral calls as their long-dated holdings cratered. The losses were a duration story first — these were very long instruments — but the speed and size of the move is exactly the regime where the straight-line duration assumption breaks down and the full curved repricing matters. The episode is a reminder that on a violent move, you must reprice properly, not lean on a linear approximation calibrated for calm. (For the policy backdrop, see [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).)

**Callable corporate and municipal bonds in a rally.** Whenever rates fall meaningfully, issuers of callable corporate and municipal bonds exercise their calls in waves, retiring expensive debt and reissuing cheaply — and bondholders who expected a long stream of high coupons get their principal back early at the call price, with the upside chopped off. This is negative convexity meeting the real world: the higher yield those bonds offered was the compensation, and the call is the moment the bill comes due. Investors who modeled them on duration alone, ignoring the embedded option, consistently overestimate their upside in a rate rally.

**Pension and insurance demand for long-bond convexity.** Long-dated liabilities — a pension's promise to pay retirees decades from now, an insurer's annuity book — behave like very long, very convex bonds: their present value swings enormously when long rates move. To hedge that, these institutions are structural buyers of the longest, most convex assets they can find, especially long Treasuries and strips (zero-coupon Treasuries, the most convex instruments in the market). This persistent demand for convexity at the long end is one reason very long yields are often *lower* than a naive model would predict — the convexity is so prized by liability hedgers that they bid the price up and accept the lower yield. Convexity is not just a risk measure here; it is a thing with its own supply, demand, and price, embedded in the shape of the yield curve itself. (See how this duration-matching anchors a portfolio in [government bonds as the risk-free anchor and duration](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration).)

**The long-bond rally that kept surprising bears.** In several rate-cutting cycles, traders who shorted long bonds expecting modest moves were repeatedly stopped out by rallies larger than their duration models suggested. The mechanism was convexity: as yields fell, the bonds' duration *extended*, so each further drop in yield produced a bigger price gain than the last — a self-accelerating squeeze on anyone short. This is the trader's-eye view of the influence chart: being short a positively convex asset means your losses *accelerate* against you in exactly the move you bet against, while a long holder's gains accelerate in their favor. Convexity is why "the long bond always rallies more than you think" is a piece of desk folklore with real math behind it.

## When this matters to you, and further reading

If you own a bond fund, convexity is already in your account — silently helping in a long-duration government fund during a rate rally, silently hurting in a mortgage or callable-heavy fund. You do not need to compute it, but you should know which way your funds bend: a long-Treasury fund is your convex friend in a crisis, and a high-yield-via-callables or mortgage fund is quietly short the option. If you ever build a bond portfolio deliberately, the barbell-versus-bullet choice is the cleanest lever you have for dialing convexity up or down at a chosen duration — and the right setting depends entirely on how volatile you expect rates to be.

The deeper you go in fixed income, the more convexity stops being a footnote to duration and becomes a thing traders actively buy and sell. To follow that thread, the natural next steps are the heavier-math treatments of [bond pricing](/blog/trading/quantitative-finance/bond-pricing) and [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics), where duration and convexity are derived properly from the discounting math. For the macro lens — why the rate moves that make convexity matter happen in the first place — start with [interest rates as the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable). And to see where this bond sits among everything else you could own, [government bonds as the risk-free anchor](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) ties the convex long bond back to the rest of the portfolio. Duration tells you the slope; convexity tells you the curve; together they tell you, finally, the whole story.
