---
title: "Spot rates, the zero curve, and bootstrapping"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Why discounting every cash flow at one yield is an approximation, and how to build the true per-maturity discount curve by bootstrapping spot rates from real Treasury prices."
tags: ["fixed-income", "bonds", "spot-rates", "zero-curve", "bootstrapping", "discounting", "yield-curve", "treasuries", "term-structure"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **The one-sentence thesis:** a single yield-to-maturity is a convenient blended average, but the honest way to price a bond is to discount each cash flow at its own *spot rate* — the true cost of money for that exact maturity — and you can recover those spot rates from a handful of observed Treasury prices by a step-by-step procedure called bootstrapping.
> - A bond is not one loan; it is a bundle of separate IOUs, one for each payment date, and each deserves its own discount rate.
> - The **spot (zero) rate** for maturity *t* is the annual return on a single dollar paid only at *t* — a pure, coupon-free rate.
> - **Bootstrapping** peels the curve one maturity at a time: the 1-year rate falls out directly, then you use it to solve the 2-year rate, then both to solve the 3-year rate.
> - A coupon bond is exactly a *portfolio of zero-coupon bonds*; value the pieces, add them up, and you have priced the whole.
> - The **par yield**, the **spot rate**, and the **forward rate** are three different summaries of the same curve — and on an upward-sloping curve the spot sits above the par and the forward sits above the spot.
> - Get the curve wrong and you misprice every bond, every swap, and every mortgage that hangs off it; this is the plumbing under the entire fixed-income world.

Here is a question that sounds simple and isn't. You hold a 3-year US Treasury note that pays you \$5 in one year, \$5 in two years, and \$105 in three years. What is it worth today?

Most introductions to bonds answer with one number — the *yield-to-maturity* — and tell you to discount every cash flow at that single rate. That works, in the sense that it gives you a price. But it quietly assumes something false: that a dollar arriving in one year and a dollar arriving in three years should be discounted at *the same* rate. They almost never should. Money has a different price at every horizon. Borrowing for one year is cheaper than borrowing for ten; that is the whole reason the yield curve has a shape at all. A single yield-to-maturity crushes that shape into one blended average and then pretends the average is the truth.

![A side-by-side comparison showing on the left a single yield discounting every bond cash flow at one rate, and on the right a spot curve giving each year its own true discount rate](/imgs/blogs/spot-rates-the-zero-curve-and-bootstrapping-1.png)

The diagram above is the mental model for this entire post. On the left is the world of one rate: every cash flow, near or far, gets discounted at the same 4.50% yield-to-maturity. On the right is the world this post will build: a *spot curve*, where the one-year cash flow is discounted at the one-year rate, the two-year cash flow at the two-year rate, and so on — each at its own true, coupon-free price of money. The left is an approximation. The right is how a trading desk, a pension fund, or a central bank actually values a stream of future dollars. By the end you will be able to take three real Treasury prices and hand-build the right-hand curve yourself.

This post sits in the pricing track of the series. It assumes you already know how to discount a single cash flow and how a bond's price moves opposite to its yield — if those feel shaky, read [discounting cash flows: how a bond is priced](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced) and [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much) first. Here we go one level deeper: not *a* discount rate, but the whole *curve* of them, and where it comes from.

## Foundations: the words we need before we build anything

Let me define every term from zero, because the rest of the post leans on each one. A practitioner can skim this section; if you are new, do not skip it.

**Present value (PV).** A dollar tomorrow is worth less than a dollar today, because today's dollar can earn interest. If the relevant annual interest rate is *r*, then a dollar arriving in one year is worth $\frac{1}{1+r}$ today. The act of converting a future dollar into a today-dollar is *discounting*, and the today-value is the *present value*. Everything in bond pricing is an exercise in present value.

**Discount factor.** Rather than carry the rate around, we often carry the multiplier directly. The *discount factor* for maturity *t* is the present value of one dollar paid at *t*:

$$
DF(t) = \frac{1}{(1+z_t)^{t}}
$$

where $z_t$ is the spot rate for maturity *t* (defined just below). A discount factor is always between 0 and 1; the further out the dollar, the smaller the factor. If $DF(3) = 0.8626$, then one dollar paid in three years is worth about 86.26 cents today.

**Coupon.** Most bonds pay periodic interest called the *coupon*. A \$1,000-par bond with a 5% annual coupon pays \$50 a year. (Real US Treasuries pay *semi-annually* — \$25 every six months on that bond — but to keep the arithmetic clean and visible, this post uses **annual coupons** and a **\$100 face value** throughout. The logic is identical; only the period count changes, and I will flag the semi-annual wrinkle where it matters.)

**Face value (par).** The lump sum the issuer repays at the end — the *face*, *par*, or *principal*. Our examples use \$100 par.

**Yield-to-maturity (YTM).** The single rate that, used to discount *all* of a bond's cash flows, reproduces its market price. It is an internal rate of return. It is genuinely useful as a one-number summary — but as we will see, it is an average that depends on the bond's coupon, not a clean property of the maturity.

**Zero-coupon bond.** A bond with *no* coupons. You buy it below face today and receive face at maturity; the entire return is the gap between purchase price and face. A 3-year zero that pays \$100 at maturity and costs \$86.26 today earns you \$13.74 over three years. Because it has exactly one cash flow, a zero is the cleanest possible instrument — there is nothing to blend.

**Spot rate (a.k.a. zero rate).** Here is the star of the show. The *spot rate* for maturity *t*, written $z_t$, is the annual yield on a zero-coupon bond maturing at *t*. It is the true, undiluted price of money for that one horizon — the rate at which a single dollar paid at *t* gets discounted. "Spot" because it is the rate for money locked up *starting now*. "Zero" because it is the rate implied by zero-coupon instruments. The two words mean the same thing.

**The zero curve (spot curve, term structure).** Plot the spot rate against maturity and you get the *zero curve* or *spot curve* — the term structure of interest rates in its purest form. This is subtly different from the more familiar *par yield curve* you see on the news (more on that distinction later). The zero curve is the set of discount rates that prices everything.

**Basis point (bp).** One hundredth of a percent — 0.01%. A move from 4.50% to 4.51% is one basis point. Bond people quote everything in basis points because the moves that matter are small.

With those in hand, here is the single most important reframing in this post.

### A coupon bond is a portfolio of zero-coupon bonds

Look again at our 3-year note: \$5 in year 1, \$5 in year 2, \$105 in year 3 (the last payment is the final \$5 coupon plus the \$100 face). There is no law that says these three cash flows must travel together. You could, in principle, sell the right to the year-1 \$5 to one investor, the year-2 \$5 to another, and the year-3 \$105 to a third. Each of those slivers is a zero-coupon bond: a single payment on a single date.

This is not a thought experiment. The US Treasury runs exactly this machine. It is called **STRIPS** — Separate Trading of Registered Interest and Principal of Securities — and it lets dealers split a Treasury note into its individual coupon and principal payments, each of which then trades as its own zero-coupon bond. The coupon bond and the bundle of strips are economically the same thing.

That equivalence is the key that unlocks everything:

$$
P_{\text{coupon bond}} = \sum_{t} C_t \cdot DF(t) = \sum_t \frac{C_t}{(1+z_t)^t}
$$

The price of a coupon bond is just the sum of its cash flows, each discounted at *its own maturity's* spot rate. Not one rate — a different rate for each piece. If you accept that a bond is a portfolio of zeros, you have already accepted that it needs a curve of rates, not a single number.

## Why one yield-to-maturity is a simplification, not the truth

If the spot-rate view is the honest one, why does anyone use a single YTM? Because it is convenient and, for a single bond, harmless. The YTM is *defined* to reproduce the bond's price — it is whatever rate makes the discounted cash flows add up to what the bond costs. So for the one bond you are looking at, the YTM is exact by construction.

The trouble starts the moment you compare two bonds, or try to price a *new* bond from existing ones. The YTM is a weighted average of the spot rates, and the weights depend on the bond's own cash flows. Change the coupon and you change the weights and therefore the YTM — even if the maturity is identical and the underlying spot curve has not moved at all.

#### Worked example: two 3-year bonds, same maturity, different yields

Suppose the true spot curve is the one we will bootstrap below: $z_1 = 4.00\%$, $z_2 = 4.51\%$, $z_3 = 5.05\%$. Price two different 3-year bonds off it.

**Bond One — a 3-year, \$100-par note with a 5% coupon.** Cash flows \$5, \$5, \$105.

$$
P = \frac{5}{1.0400} + \frac{5}{1.0451^2} + \frac{105}{1.0505^3} = 4.81 + 4.58 + 90.57 = \$99.96
$$

Solving for the single rate that reproduces \$99.96 gives a YTM of **5.02%**.

**Bond Two — a 3-year, \$100-par note with an 8% coupon.** Cash flows \$8, \$8, \$108.

$$
P = \frac{8}{1.0400} + \frac{8}{1.0451^2} + \frac{108}{1.0505^3} = 7.69 + 7.32 + 93.16 = \$108.18
$$

Solving for its single YTM gives **5.00%**.

Two bonds, both maturing in exactly three years, priced off the *identical* spot curve — and their yields-to-maturity differ by two basis points (5.02% vs 5.00%). The higher-coupon bond has a slightly lower YTM because more of its value comes from the earlier, lower-rate years. *The YTM is not a property of "three years"; it is a property of a particular bond's cash-flow shape laid over the curve.*

That is the whole problem with one rate. The spot curve, by contrast, is the same for everyone: $z_3 = 5.05\%$ is *the* three-year rate, full stop, regardless of what coupon happens to be attached to it. To price consistently — to make sure two bonds with the same cash flows always get the same value, and to price a brand-new issue off the market — you need the spot curve, not a forest of bond-specific YTMs.

## The zero curve: what we are trying to build

Before we build it, let us look at the finished object so the construction has a destination.

![A line chart of the zero curve showing spot rates rising from about four percent at one year to just over five percent at three years, with dashed horizontal gridlines and labeled axes for spot rate and maturity in years](/imgs/blogs/spot-rates-the-zero-curve-and-bootstrapping-2.png)

That is the zero curve we are going to recover: $z_1 = 4.00\%$, $z_2 = 4.51\%$, $z_3 = 5.05\%$. Each dot is a single number — the rate that discounts one dollar paid at that exact maturity. Read it left to right and it tells a story everyone in markets watches obsessively: short money is cheap, long money is dearer, and the gap between them is the slope of the curve. Here the curve slopes *up*, which is its usual resting shape; when it flattens or inverts, the bond market is sending a signal worth a whole separate post — see [reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).

The defining feature of the zero curve, the thing that makes it different from the curve you see on TV, is that **every point is coupon-free**. A 3-year zero rate of 5.05% is contaminated by nothing — no earlier coupons, no reinvestment assumptions, no blending. It is the pure, isolated cost of three-year money. That purity is exactly why it is the right input for discounting: when you discount a year-3 cash flow, you want the year-3 *zero* rate, not some average that has year-1 and year-2 rates mixed into it.

The catch — and the reason bootstrapping exists — is that pure zero-coupon bonds are not always quoted at every maturity you need. STRIPS exist, but their prices can be thin and distorted; the deep, liquid, trustworthy prices live in the *coupon-bearing* Treasury market. So we face a puzzle: the market hands us the prices of coupon bonds, but we want the curve of zero rates hiding inside them. Bootstrapping is how we extract one from the other.

## Bootstrapping: peeling the curve one maturity at a time

The word "bootstrap" comes from "pulling yourself up by your bootstraps" — building something from the ground using only what you already have. That is exactly the spirit. We will recover the spot curve one rung at a time, and each rung uses the rungs already built.

![A pipeline showing four stages: solve the one-year spot rate directly from a one-year Treasury, then strip the known coupon to solve the two-year rate, then strip two coupons to solve the three-year rate, ending in a complete zero curve that prices any bond](/imgs/blogs/spot-rates-the-zero-curve-and-bootstrapping-3.png)

The logic of the pipeline above is the whole technique in one picture:

1. **The 1-year rate falls out for free.** A 1-year bond has only one future cash flow, so its price already *is* a discounted single payment. Solve for the rate directly — no curve needed.
2. **The 2-year rate uses the 1-year rate.** A 2-year bond has two cash flows. We already know how to discount the first one (use $z_1$), so we *strip it out* — subtract its present value from the bond's price. What remains must be the present value of the final cash flow, and now there is only one unknown rate to solve for: $z_2$.
3. **The 3-year rate uses both.** A 3-year bond has three cash flows. Discount the first with $z_1$, the second with $z_2$, strip both out, and the remainder pins down $z_3$.

Each step turns a multi-unknown problem into a one-unknown problem by leaning on the rates already discovered. That is the entire method. Let us run it on real numbers.

### The setup: three Treasuries

We will bootstrap a 3-year curve from three US Treasury notes. To keep every digit visible we use \$100 face values and annual coupons. These are *illustrative* prices chosen to give clean spot rates, but they are exactly what real bootstrapping looks like — the on-the-run Treasury complex gives you one liquid bond at roughly each maturity, and you walk up the ladder.

| Bond | Maturity | Annual coupon | Price | Cash flows (\$) |
|---|---|---|---|---|
| A | 1 year | 4.0% | \$100.00 | \$104 at t=1 |
| B | 2 years | 4.5% | \$100.00 | \$4.50 at t=1, \$104.50 at t=2 |
| C | 3 years | 5.0% | \$99.96 | \$5 at t=1, \$5 at t=2, \$105 at t=3 |

Two of these bonds (A and B) are priced at exactly par (\$100), which is no accident — a bond trades at par when its coupon equals its yield, and these are roughly the par-yield Treasuries at those maturities. Bond C trades a hair below par. Now we strip.

#### Worked example: the 1-year spot rate

Bond A pays a single cash flow of \$104 (its \$4 coupon plus \$100 face) in one year and costs \$100.00 today. There is only one cash flow, so its price is just that cash flow discounted once:

$$
100.00 = \frac{104}{1 + z_1}
$$

Solve for $z_1$:

$$
1 + z_1 = \frac{104}{100.00} = 1.0400 \quad\Rightarrow\quad z_1 = 4.00\%
$$

The 1-year spot rate is **4.00%**, and the 1-year discount factor is $DF(1) = 1 / 1.04 = 0.9615$. *With a single-cash-flow bond, the spot rate is not bootstrapped at all — it simply is the bond's yield, because there are no earlier coupons to contaminate it.*

#### Worked example: the 2-year spot rate

Bond B pays \$4.50 in one year and \$104.50 in two years, and costs \$100.00. Its price must equal the sum of its two discounted cash flows:

$$
100.00 = \frac{4.50}{1 + z_1} + \frac{104.50}{(1 + z_2)^2}
$$

We already know $z_1 = 4.00\%$, so we can compute the first term exactly — this is the "strip out the known coupon" step:

$$
\frac{4.50}{1.0400} = \$4.327
$$

Subtract it from the price. Whatever is left over must be the present value of the final \$104.50 cash flow:

$$
\frac{104.50}{(1 + z_2)^2} = 100.00 - 4.327 = \$95.673
$$

Now there is exactly one unknown. Solve for $z_2$:

$$
(1 + z_2)^2 = \frac{104.50}{95.673} = 1.09226 \quad\Rightarrow\quad 1 + z_2 = \sqrt{1.09226} = 1.0451
$$

The 2-year spot rate is **4.51%**, and $DF(2) = 1 / 1.0451^2 = 0.9155$. *Notice the move: we used the rate we already had to remove the early cash flow, which left a one-cash-flow problem we could solve directly — that is the bootstrap.*

#### Worked example: the 3-year spot rate

Bond C pays \$5, \$5, and \$105, and costs \$99.96. Its price equation has three terms:

$$
99.96 = \frac{5}{1 + z_1} + \frac{5}{(1 + z_2)^2} + \frac{105}{(1 + z_3)^3}
$$

We know $z_1$ and $z_2$, so we compute and strip the first two terms:

$$
\frac{5}{1.0400} = \$4.808 \qquad \frac{5}{1.0451^2} = \$4.578
$$

Together they are \$9.385. Subtract from the price to isolate the final cash flow:

$$
\frac{105}{(1 + z_3)^3} = 99.96 - 9.385 = \$90.575
$$

One unknown again. Solve:

$$
(1 + z_3)^3 = \frac{105}{90.575} = 1.15926 \quad\Rightarrow\quad 1 + z_3 = 1.15926^{1/3} = 1.0505
$$

The 3-year spot rate is **5.05%**, and $DF(3) = 1 / 1.0505^3 = 0.8626$. *Each rung climbed only one new unknown at a time, because every earlier coupon was discounted at a rate we had already nailed down.*

We now have the full curve: $z_1 = 4.00\%$, $z_2 = 4.51\%$, $z_3 = 5.05\%$ — exactly the curve in the chart above, recovered from nothing but three observed prices.

### The whole bootstrap on one page

![A table laying out the bootstrap for the one, two, and three year bonds, showing each bond's price and coupons, the present value of earlier coupons stripped out, the spot rate solved at each step, and the resulting discount factor](/imgs/blogs/spot-rates-the-zero-curve-and-bootstrapping-6.png)

The table above is the same three worked examples compressed into the form a desk actually keeps it in. Read each row left to right: the bond's price and cash flows go in on the left; you strip the present value of the earlier coupons (using the spot rates already solved); you solve for this maturity's spot rate; and you record the discount factor. Read top to bottom and you watch the curve being built rung by rung. Every number in that table appeared in the worked examples — nothing is invented for the picture.

A few things worth noticing in the table that the prose can blur over. First, the "strip earlier PV" column is empty for the 1-year bond, grows to one term for the 2-year, and two terms for the 3-year — the work compounds as you climb, but each step still leaves only one unknown. Second, the discount factors fall monotonically (0.9615 → 0.9155 → 0.8626) even though we are not assuming that; it falls out of the rising rates, and it is what guarantees that a dollar further away is always worth less today. If a bootstrapped discount factor ever came out *higher* than a nearer one, you would know immediately that one of your input prices was stale or wrong.

## Seeing the bond as a strip of zeros

We claimed early on that a coupon bond is a portfolio of zero-coupon bonds. The bootstrap makes that literal, and it is worth drawing.

![A diagram showing a single three-year coupon note at the top splitting into three separate zero-coupon bonds below: a five dollar zero due in one year, a five dollar zero due in two years, and a one hundred five dollar zero due in three years, each with its own spot rate and present value](/imgs/blogs/spot-rates-the-zero-curve-and-bootstrapping-7.png)

The figure decomposes Bond C into its three zeros. Each sliver is priced at *its own* maturity's spot rate:

- The \$5 due in 1 year, discounted at $z_1 = 4.00\%$, is worth \$4.81.
- The \$5 due in 2 years, discounted at $z_2 = 4.51\%$, is worth \$4.58.
- The \$105 due in 3 years, discounted at $z_3 = 5.05\%$, is worth \$90.57.

Add them: \$4.81 + \$4.58 + \$90.57 = **\$99.96** — exactly the bond's price. The whole equals the sum of its parts because the parts are real, separately-tradable claims. This is the deepest reason the spot curve is the right tool: it is the only set of rates that prices *each separable piece* correctly at the same time. A single YTM cannot do that — it would value all three slivers at the same 5.02%, overpaying for the near pieces and underpaying for the far one, with the errors happening to cancel only for *this specific* bundle.

#### Worked example: pricing a brand-new bond off the curve

Here is where the curve earns its keep. Suppose a fictional issuer, **Northwind Corp**, has nothing to do with our three Treasuries but wants to know what a risk-free 3-year, \$100-par note with a 6% coupon *should* cost. We do not have a 6% Treasury to read a price off. But we have the curve, and the curve prices anything:

$$
P = \frac{6}{1.0400} + \frac{6}{1.0451^2} + \frac{106}{1.0505^3}
$$

$$
P = 5.77 + 5.49 + 91.44 = \$102.70
$$

(The \$106 in year 3 is the \$6 coupon plus \$100 face.) Without ever observing a 6% bond, we priced one to the penny, using only the zero curve bootstrapped from three other bonds. *That is the payoff of building the curve: it lets you price every bond consistently, including ones that have never traded, which is precisely what you need to value a new issue, mark a portfolio, or check whether a quoted bond is cheap or rich.*

## Discounting the right way vs the convenient way

We have asserted that a flat YTM is an approximation. Let us quantify the error, because "approximation" can mean anything from "rounding noise" to "you just lost real money."

![A side-by-side comparison pricing the same three-year five percent bond two ways: on the left discounting all cash flows at a flat four point five percent yield gives one hundred one dollars and thirty-seven cents, on the right discounting each cash flow at its own spot rate gives ninety-nine dollars and ninety-six cents](/imgs/blogs/spot-rates-the-zero-curve-and-bootstrapping-5.png)

The figure prices Bond C — our 3-year 5% note — two ways. On the right is the correct method: each cash flow at its own spot rate, summing to the true price of **\$99.96**. On the left is the convenient method: discount *everything* at a single flat 4.50% rate.

#### Worked example: the cost of using one flat rate

Discount Bond C's cash flows at a flat 4.50%:

$$
\frac{5}{1.045} + \frac{5}{1.045^2} + \frac{105}{1.045^3} = 4.78 + 4.58 + 92.01 = \$101.37
$$

The flat-rate method says the bond is worth **\$101.37**. The spot-curve method says **\$99.96**. The flat rate *overvalues the bond by \$1.41 per \$100 of face* — about 1.4% of its value.

Why does it overvalue? Look at where the error lives. The first two cash flows are tiny and barely move. The damage is almost entirely in the year-3 term: the flat rate discounts the big \$105 payment at 4.50%, but its true spot rate is 5.05% — fully 55 basis points higher. Discounting a large, far-off cash flow at too low a rate inflates its present value, and on an upward-sloping curve a flat YTM that lands somewhere in the middle of the curve will *always* be too low for the longest, largest cash flow. *On an upward-sloping curve, a single mid-curve discount rate systematically overvalues long bonds, because it under-discounts the back-loaded principal payment where most of the value lives.*

A dollar and forty-one cents on a hundred sounds small. Scale it. A bond desk running \$5 billion of 3-year paper that mispriced its discounting by 1.4% would be carrying roughly \$70 million of phantom value on its book. On a 30-year bond, where the curve has far more room to slope and the principal is discounted across decades, the gap between flat-rate and spot-rate pricing can run into many points, not fractions of one. This is not academic hair-splitting; it is the difference between a mark you can trade on and one that blows up at the next audit.

## Par yields, spot rates, and forward rates: three views of one curve

Now we can clear up a confusion that trips up almost everyone. There are at least three different "yield curves," and they are not the same line. They are three ways of summarizing the *same* underlying set of discount factors.

**The par yield curve** is the one on the news. The par yield at maturity *t* is the coupon rate that would make a brand-new *t*-year bond trade at exactly par (\$100). It is a coupon-bond yield, so it blends all the spot rates up to *t*. The Treasury publishes this daily as the "par yield curve," and it is what most people mean when they say "the yield curve."

**The spot (zero) curve** is what we just built: one pure, coupon-free rate per maturity.

**The forward curve** answers a different question: not "what is the rate from now to year *t*," but "what is the rate from year *t–1* to year *t*" — the rate for a single future year, implied by today's curve. Forwards are the marginal rates; spots are the average of the forwards up to that point. Forward rates get their own full treatment in [forward rates: what the market expects rates to be](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be), but they belong in this picture because all three come from the same discount factors.

The relationship between the three is the most important structural fact about the term structure, so here it is on one chart.

![A line chart showing three curves rising with maturity: the par yield curve in green sitting lowest, the spot or zero curve in blue sitting just above it, and the forward curve in amber rising most steeply and sitting highest, with a legend identifying each](/imgs/blogs/spot-rates-the-zero-curve-and-bootstrapping-4.png)

When the curve slopes *up*, as in the figure (an illustrative steeper curve, used so the three lines separate clearly), they stack in a fixed order: **par yield below, spot rate above it, forward rate highest of all.** The intuition is mechanical, not mysterious.

- The **forward** is highest because it is the *marginal* rate — the cost of money for one specific future year. On a rising curve, each new year added is more expensive than the average so far, so the forward for that year sits above the running average.
- The **spot** is the average of all the forwards up to that maturity, so it lags the forward — it has cheaper early years mixed in.
- The **par** sits just *below* the spot because a par bond pays coupons all along the way, and those early coupons are discounted at the lower early spot rates, dragging the bond's overall yield down slightly relative to the pure back-loaded zero.

Flip the curve — make it slope *down* (inverted) — and the order flips with it: par above spot above forward. The ordering is not a coincidence; it is arithmetic, and once you see it you can never un-see which curve someone is quoting.

#### Worked example: the par yield and the forward implied by our curve

Take our bootstrapped curve ($z_1 = 4.00\%$, $z_2 = 4.51\%$, $z_3 = 5.05\%$; discount factors 0.9615, 0.9155, 0.8626).

**The 3-year par yield.** The par coupon *c* is the one that makes the bond worth exactly \$100:

$$
c \cdot (DF_1 + DF_2 + DF_3) + 100 \cdot DF_3 = 100
$$

$$
c = \frac{100 \,(1 - DF_3)}{DF_1 + DF_2 + DF_3} = \frac{100 \,(1 - 0.8626)}{0.9615 + 0.9155 + 0.8626} = \frac{13.74}{2.7396} = 5.02\%
$$

So the 3-year par yield is **5.02%**, sitting just *below* the 3-year spot rate of 5.05% — exactly the ordering the figure promised.

**The 2-to-3-year forward rate.** This is the rate for the single year from *t=2* to *t=3*, implied by the curve:

$$
1 + f_{2,3} = \frac{DF_2}{DF_3} = \frac{0.9155}{0.8626} = 1.0613 \quad\Rightarrow\quad f_{2,3} = 6.13\%
$$

The implied rate for that one future year is **6.13%** — well above the 5.05% three-year spot, because the spot is an average that still carries the cheaper years 1 and 2. *The par yield, the spot rate, and the forward are three different averages of the same forwards: the forward is the marginal year, the spot is the average to date, and the par is that average dragged down by coupon timing.*

## What the curve's shape does to your bootstrap

So far our curve sloped politely upward. Real curves do not always cooperate, and the bootstrap is indifferent to shape — it solves whatever rates the prices imply, even when those rates do something counterintuitive. This is a strength, not a weakness: bootstrapping is a *measurement*, not a model, so it will faithfully report a flat, humped, or inverted curve if that is what the market is pricing. But it pays to see what those shapes do to the spot rates, because the relationship between coupon yields and spot rates flips with the slope.

An **upward-sloping** (normal) curve, like ours, makes spot rates rise faster than coupon yields. Because a coupon bond's early payments are discounted at the lower early spot rates, its overall yield is pulled below the pure zero rate at the same maturity. That is why, on our curve, the 3-year par yield (5.02%) sat just under the 3-year spot (5.05%). Stretch the curve steeper and the gap widens; the longest spot rates can sit well above the par yields you read on a screen.

A **flat** curve collapses all the distinctions. If every spot rate equals, say, 4.50%, then the par yield, the spot rate, and the forward rate are all 4.50% at every maturity, and — this is the one case where it is true — discounting at a single flat YTM gives exactly the right answer. A flat curve is the only world in which the "one rate for everything" shortcut is not an approximation at all. It is also rare and usually fleeting, a way-station as the curve transitions between normal and inverted.

An **inverted** curve — short rates above long rates — flips the ordering. Now spot rates *fall* with maturity, the par yield sits *above* the spot rate, and the forward rate sits *below* both. Inversions are famous because they have preceded most US recessions; the mechanism and the track record get a full post in [reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession). For our purposes the point is narrower: the bootstrap handles an inversion without changing a single step.

#### Worked example: bootstrapping a point on an inverted curve

Suppose short rates have spiked. Keep our 1-year bond (\$100.00, 4% coupon, so $z_1 = 4.00\%$ as before), but now imagine a *2-year* bond whose price the market has bid up: a 2-year, \$100-par bond with a 5% coupon trading at **\$101.50** (above par, because its coupon beats the lower long rate). Cash flows: \$5 at t=1, \$105 at t=2.

$$
101.50 = \frac{5}{1.0400} + \frac{105}{(1 + z_2)^2}
$$

Strip the first coupon: $5 / 1.0400 = \$4.808$. The remainder is $101.50 - 4.808 = \$96.692$, so:

$$
(1 + z_2)^2 = \frac{105}{96.692} = 1.08593 \quad\Rightarrow\quad z_2 = 4.21\%
$$

The 2-year spot rate is **4.21%** — still just above the 1-year rate of 4.00%, so this particular price gives a barely-normal curve. But push the 2-year bond's price higher still, to \$103.00 (investors piling into the longer bond), and the same arithmetic gives $z_2 \approx 3.41\%$, now genuinely *below* the 1-year rate — a textbook inversion, bootstrapped from nothing but the two prices. *The procedure never cares which way the curve slopes; it reports the rates the prices imply, and an inverted result is a real signal, not a numerical error.*

## When bootstrapping breaks (and how to tell)

Because the bootstrap leans each rung on the rungs below it, an error early in the curve propagates upward — and there are a handful of ways the whole thing can go quietly wrong. A practitioner's instinct is to *sanity-check the output*, because the math will happily return nonsense if you feed it nonsense.

**Stale or off-market input prices.** The bootstrap is only as good as the prices going in. If one of your benchmark bonds is quoted from a trade that happened hours ago while the market has since moved, its implied spot rate is wrong — and so is every spot rate solved *above* it, because each later rung stripped a coupon at that bad rate. This is why desks bootstrap from the most liquid, most recently-traded *on-the-run* Treasuries and treat thin off-the-run quotes with suspicion.

**Negative forward rates.** A red flag that something is broken: if your bootstrapped discount factors ever *rise* with maturity ($DF(3) > DF(2)$), the implied forward rate for that period is negative — the market would be paying you to hold money longer, which (outside genuine negative-rate regimes like Europe and Japan in the 2010s) usually means a bad input price rather than a real economic signal. A monotonic, smoothly-declining set of discount factors is the bootstrap's own self-check.

**Missing maturities and interpolation risk.** You almost never have a clean bond at every rung. The gaps get filled by interpolation, and *how* you interpolate is a modeling choice with real consequences: interpolate linearly on the spot rates and you can accidentally manufacture jagged, oscillating forward rates between your anchor points. Better schemes interpolate on the log of the discount factors or fit a smooth spline precisely to keep the implied forwards well-behaved. The bootstrap pins the curve at the bonds you observe; everything between them is an assumption, and two desks using different assumptions will disagree.

**Coupon effects and tax distortions.** Two bonds at the same maturity but very different coupons can imply slightly different curves because of taxes, liquidity, and demand for specific cash-flow shapes (a pension that wants the back-loaded principal will bid up low-coupon bonds). Careful curve-builders either use a consistent set of bonds or fit the curve to *all* available bonds at once with a smoothing model, rather than naively bootstrapping from an arbitrary three.

#### Worked example: how one stale price poisons the curve above it

Suppose Bond B's true price is \$100.00 (giving $z_2 = 4.51\%$), but your screen shows a stale \$100.40. Re-run the 2-year step: strip the \$4.327 first coupon, leaving $100.40 - 4.327 = \$96.073$ for the final cash flow, so $(1+z_2)^2 = 104.50 / 96.073 = 1.08771$ and $z_2 = 4.29\%$ — a 22-basis-point error from a 40-cent price slip. Now climb to the 3-year bond: you would strip its 2-year coupon at the *wrong* 4.29% rate, contaminating $z_3$ as well. *A single stale input does not stay local — it leaks into every spot rate solved above it, which is exactly why the integrity of the input prices matters more than the elegance of the algorithm.*

## The curve as the foundation under everything riskier

Up to here the curve has been built entirely from US Treasuries, which the market treats as effectively free of default risk. That is deliberate: the Treasury spot curve is the *risk-free* term structure, the base layer of all valuation. Everything riskier is priced as this curve *plus a spread* — and that is the influence thread of this whole series, made concrete.

Take our fictional **Northwind Corp** again. A 3-year Northwind bond is not risk-free; the company could default. So the market does not discount Northwind's cash flows at the Treasury spot rates — it discounts them at the Treasury spot rates *plus a credit spread* that compensates for the chance of not being paid back. If Northwind's 3-year credit spread is 150 basis points, you bootstrap and use a Northwind discount rate of roughly $5.05\% + 1.50\% = 6.55\%$ for its year-3 cash flow, and correspondingly higher rates for the nearer years.

#### Worked example: the same bond, risk-free vs risky

Price a 3-year, \$100-par, 6% bond two ways. Risk-free (our Treasury curve), we already found it is worth \$102.70. Now price the *Northwind* version, adding a flat 150-basis-point spread to each spot rate:

$$
P = \frac{6}{1.0550} + \frac{6}{1.0601^2} + \frac{106}{1.0655^3} = 5.69 + 5.34 + 87.63 = \$98.65
$$

The credit risk knocks **\$4.05** off the price (\$102.70 → \$98.65) — that is the dollar value the market puts on Northwind's chance of default over three years. *The risk-free spot curve is the floor; every credit, every mortgage, every emerging-market bond is that floor plus a spread, which is why getting the Treasury curve right is the precondition for pricing anything else.*

That spread layer is the subject of the credit track — [credit spreads: pricing the probability of default](/blog/trading/fixed-income/credit-spreads-pricing-the-probability-of-default) — and the allocation view in [corporate credit: investment grade, high yield, spreads](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads). The mechanism is always the same: the Treasury zero curve discounts the safe cash flows, and a spread on top accounts for everything that makes a cash flow less than certain.

## Compounding, day-counts, and the rest of the real-world fine print

The bootstrap we ran is correct in spirit and exact in our annual-coupon world. Real markets add a layer of conventions that change the *digits* but not the *method*. It is worth knowing them so you are not blindsided.

**Semi-annual coupons.** US Treasuries pay twice a year, so a 3-year note has six cash flows, not three, and you bootstrap at six-month rungs (0.5y, 1.0y, 1.5y, …). The spot rates are usually quoted on a *semi-annual bond-equivalent* basis, meaning $DF(t) = 1 / (1 + z_t/2)^{2t}$. The procedure is identical — solve the nearest rung first, strip it, climb — there are just more rungs.

**Interpolation between rungs.** You rarely get a clean liquid bond at every single maturity. Between observed points, desks interpolate — linearly on the spot rates, or (better) on the *log discount factors*, or with a smooth spline — to fill in maturities like 2.7 years where no benchmark bond trades. Different banks use different interpolation schemes, which is one reason two desks can disagree by a basis point on the same curve.

**Day-count conventions.** "One year" is not always 365 days. Treasuries use actual/actual; money-market instruments often use actual/360. The fraction of a year between payments — and therefore the exponent in the discount factor — depends on the convention. This is the same accrued-interest machinery covered in [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much); it shifts the numbers a touch without touching the logic.

**Continuous compounding.** Quants frequently work in continuously-compounded rates, where $DF(t) = e^{-z_t \cdot t}$. It makes the algebra cleaner (forwards become simple differences of spot×time) and is the convention behind most of the heavy curve-modeling math — see [yield curve modeling](/blog/trading/quantitative-finance/yield-curve-modeling) and [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics) for that machinery. It is the same curve, re-expressed.

#### Worked example: how much the semi-annual wrinkle moves a number

Take our 1-year rate. We found $z_1 = 4.00\%$ on an annual basis. Quoted as a semi-annual bond-equivalent yield, the *same* 1-year discount factor (0.9615) implies a rate $y$ solving $1/(1 + y/2)^2 = 0.9615$, which gives $y = 3.96\%$ — four basis points lower than the annual figure. The discount factor, the actual economic content, is unchanged at 0.9615; only the *quoting convention* shifts the headline rate. *Always check which compounding basis a rate is quoted on before you compare two curves — a few basis points of "difference" can be pure convention.*

## Common misconceptions

**"A bond's yield-to-maturity is the rate the market charges for that maturity."** No. The YTM is a bond-specific blended average whose value depends on the bond's coupon, as our two-3-year-bonds example showed (5.02% vs 5.00% off the *same* curve). The rate the market charges for a maturity is the *spot* rate, which is coupon-free. The YTM is a summary of one bond; the spot rate is a property of the term structure.

**"The yield curve on the news is the spot curve."** Almost always no. The widely-quoted Treasury "yield curve" is the *par yield* curve — coupon-bond yields, not zero rates. On an upward-sloping curve the par curve sits *below* the spot curve, by a few basis points at short maturities and more at long ones. If you discount cash flows using par yields as if they were spot rates, you will misprice — subtly at the short end, materially at the long end.

**"Bootstrapping requires fancy math or a solver."** For the early rungs it is grade-school algebra: the 1-year rate is a division, the 2-year a square root, the 3-year a cube root. A spreadsheet handles a full semi-annual 30-year curve with nothing more exotic than `=RATE()` or a one-line goal-seek per rung. The complexity in real curve-building is not the bootstrap; it is the *interpolation* and *smoothing* between observed bonds, and deciding which bonds to trust.

**"Each spot rate is set independently by supply and demand."** Not quite. The spot rates are *jointly* implied by coupon-bond prices through the bootstrap — they are extracted, not separately quoted (outside the STRIPS market). Move one coupon bond's price and you change one or more spot rates downstream of it. The curve is a system, which is why a single mispriced or stale input bond can poison several spot rates above it.

**"A flat discount rate is fine because the errors are small."** They are small for a short bond on a gently-sloping curve and large for a long bond on a steep one — and they are *systematic*, not random. As our example showed, a flat mid-curve rate consistently *overvalues* long bonds on an upward curve by under-discounting the back-loaded principal. Systematic mispricing does not wash out across a portfolio; it accumulates in one direction.

**"Spot rates and forward rates are the same thing if the curve is smooth."** They coincide only where the curve is *flat*. The moment it slopes, the forward (the marginal one-year rate) diverges from the spot (the average rate), as our 6.13% forward vs 5.05% spot showed. Confusing the two is one of the most common and costly errors in fixed income — it leads people to think the market "expects" rates the curve never implied.

## How it shows up in real markets

**The Treasury STRIPS market makes zeros real.** Since 1985 the US Treasury has let dealers separate a note or bond into its individual coupon and principal payments, each trading as a standalone zero-coupon security. A 30-year bond becomes 61 separate strips (60 semi-annual coupons plus one principal). The strips market lets investors who want a single, certain payment on a single date — say, a pension that owes a lump sum in 2040 — buy exactly that, instead of a coupon bond whose interim payments they would have to reinvest at unknown future rates. The existence of STRIPS is the market's own admission that a coupon bond *is* a portfolio of zeros; the bootstrap just reads the zero curve back out of the coupon-bond prices.

**Every swap desk lives or dies on its discount curve.** When a bank prices an interest-rate swap — a contract to exchange fixed for floating payments, the single largest derivatives market on earth at hundreds of trillions in notional — it discounts every future leg using a bootstrapped curve. Before 2008, desks discounted with a curve bootstrapped from LIBOR-based instruments. After the crisis exposed the credit risk hiding in LIBOR, the market moved to *OIS discounting* — building the discount curve from overnight-indexed-swap rates that better reflect near-risk-free funding — and then to the SOFR curve after LIBOR's 2023 retirement. Each transition was, at bottom, a fight over *which prices to bootstrap the discount curve from*. Trillions of dollars of contract values reprice when the answer changes.

**The 2008 OIS-LIBOR blowup was a curve story.** For years, traders had treated the LIBOR curve and the OIS curve as nearly identical — the spread between them was a few basis points and ignored. In the autumn of 2008 that spread exploded past 350 basis points as banks stopped trusting each other's credit. Anyone still bootstrapping a single curve and using it both to *project* floating payments and to *discount* them was suddenly carrying enormous errors. The crisis permanently split the world into multiple curves — one to project, one to discount — and made "which curve" a front-page risk-management question. The mechanism is exactly the one in this post: get the discount curve wrong and every cash flow downstream is mispriced.

**Pension and insurance valuation is pure spot-rate discounting.** A life insurer or defined-benefit pension owes a schedule of payments stretching out 30, 40, 50 years. Regulators in many jurisdictions require them to value those liabilities by discounting each future payment at the appropriate *spot* rate from a prescribed curve — not at a single blended rate, precisely because a single rate would misstate the value of the long-dated obligations. When the spot curve shifts, the present value of those liabilities can swing by tens of percent, which is why these institutions hedge with such intensity. This is the immunization and liability-matching machinery covered in the duration track; it all runs on the curve we just built.

**Central banks read the curve to read the economy.** When the Federal Reserve, the ECB, or the Bank of England study "what the market expects," they are not reading par yields off a screen — they bootstrap the spot curve and then extract the *forward* curve from it, because forwards are the market's implied path of future short rates. The famous recession-signaling "yield curve inversion" is, underneath, a statement about the shape of the spot and forward curves. For how policymakers use and move that curve, see [the central bank toolkit: rates, QE, QT, and forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) and the macro view in [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

**The Treasury publishes its own fitted curve every day.** The "Daily Treasury Par Yield Curve Rates" that markets quote endlessly are not raw bond yields — the US Treasury *fits* a smooth curve to the prices of recently-auctioned securities using a quasi-cubic spline, then reads off the par yields at standard maturities (1mo, 3mo, … 20yr, 30yr). That fitting step is the institutional, smoothed cousin of the bootstrap we did by hand: take observed coupon-bond prices, extract a consistent term structure, publish it. Every analyst who pulls those numbers into a model is standing on a curve someone built from prices exactly the way this post describes — which is why understanding the construction, not just the output, lets you spot when a quoted curve is doing something the underlying bonds do not support.

**The 2020 dash-for-cash deranged the curve for a week.** In March 2020, as the pandemic hit, even the world's safest asset briefly broke. Investors scrambling for cash sold *everything*, including Treasuries, and the normally pristine relationship between bond prices and the implied curve fractured — off-the-run bonds traded at wildly different implied yields than on-the-run bonds of the same maturity, and anyone bootstrapping naively from all of them would have produced a jagged, arbitrage-ridden curve. The Federal Reserve stepped in with hundreds of billions in Treasury purchases specifically to restore orderly pricing. The episode is a vivid reminder that the bootstrap assumes the input prices are clean and mutually consistent; when liquidity vanishes, that assumption fails and the curve you extract is only as trustworthy as the market that produced it.

**Your mortgage hangs off this curve.** A 30-year fixed mortgage is priced off long-term rates that trace back to the bootstrapped Treasury curve — the risk-free spot rates plus spreads for credit and prepayment. When the long end of the spot curve rises 50 basis points, mortgage rates follow within days, and the monthly payment on a new \$400,000 loan jumps by a couple hundred dollars. The chain from a Treasury auction to your kitchen-table budget runs straight through the curve this post built — the risk-free spots set the floor, and the mortgage spread on top accounts for credit and the option you hold to prepay. The full transmission is traced in the influence track of this series.

## When this matters to you and where to go next

If you ever value anything that pays cash on a schedule — a bond, a pension, an annuity, a lease, a structured note, even a stream of business cash flows in a discounted-cash-flow model — you are implicitly choosing a discount curve. The lesson of this post is that the honest choice is not one rate but a curve of them, and that the curve is *recoverable* from prices you can observe. You do not have to take a single yield on faith; you can build the term structure yourself and price every cash flow at its own true rate.

From here, three natural next steps. To turn the spot curve into the market's forecast of future rates, read [forward rates: what the market expects rates to be](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be) — the forward curve we previewed gets built properly there. To understand how the whole curve moves and what its shape signals, the yield-curve track of this series starts at [the yield curve explained](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance). And for the heavy mathematics — short-rate models, smoothing splines, no-arbitrage curve construction — step across to [yield curve modeling](/blog/trading/quantitative-finance/yield-curve-modeling) and [bond pricing](/blog/trading/quantitative-finance/bond-pricing). The curve you bootstrapped by hand here is the object all of those refine.

*This is educational material, not investment advice. The prices, rates, and bonds in the worked examples are illustrative and chosen for clean arithmetic; real-market curves require live data and the conventions described above.*
