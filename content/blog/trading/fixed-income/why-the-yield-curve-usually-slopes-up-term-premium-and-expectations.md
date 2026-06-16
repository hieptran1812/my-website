---
title: "Why the yield curve usually slopes up: term premium and expectations"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-scratch explanation of why long-term interest rates are usually higher than short-term ones — the expectations hypothesis, the term premium, and how a long yield breaks into a forecast plus a fee for risk."
tags: ["fixed-income", "bonds", "yield-curve", "term-premium", "expectations-hypothesis", "interest-rates", "liquidity-preference", "duration"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — the upward slope of a normal yield curve is two things added together: the market's forecast of where short-term rates are going, plus an extra fee investors charge for the risk of locking money up for a long time.
> - The **expectations hypothesis** says a long yield is just the *average* of the short rates the market expects over the bond's life — so a rising path of expected short rates, all by itself, bends the curve upward.
> - But that can't be the whole story, because long bonds are *riskier* than rolling short bills, and risk-averse investors demand to be paid for that. That extra pay is the **term premium**.
> - The long yield therefore **decomposes** cleanly: `long yield ≈ average expected short rate + term premium`. The first piece is a forecast; the second is compensation for risk.
> - The curve *normally slopes up* because the term premium is *usually positive* and large enough to tip the balance — even when the market expects rates to hold steady, you get paid extra to go long.
> - To **flatten or invert** the curve, the expectations piece has to fall *faster* than the premium can hold it up — which is exactly what happens when the market expects big rate cuts ahead.
> - The term premium is **not a constant**: inflation uncertainty, the supply of bonds, and the world's hunger for safe assets all push it around, and that variation is real money for anyone holding duration.

Pull up a screen of US Treasury yields on almost any ordinary day and you will notice something so familiar that most people never stop to ask why it is true: the longer you lend the government your money, the more it pays you. A three-month Treasury bill might yield 4.2%, a two-year note 4.0%, a ten-year note 4.3%, a thirty-year bond 4.5%. The line connecting those dots — the *yield curve* — usually slopes gently upward from left to right. It is so reliably up-sloping that we call an up-sloping curve "normal" and a down-sloping one "inverted," as if the second were a kind of illness.

But *why*? Why should a longer loan pay more? The instinctive answer — "because you wait longer, so you deserve more" — is half-right and half a trap. It's a trap because the yield is an *annual* rate: it already says how much you earn *per year*, regardless of how many years you wait. Waiting ten years instead of one does not, by itself, justify a higher rate *per year*. So the real question is sharper than it first appears: why should the *annual* price of money be higher for a ten-year loan than for a one-year loan? There are two distinct reasons, and the whole job of this post is to separate them cleanly and show you how they add up.

![A before and after comparison showing a single ten-year yield number on the left and on the right that same number split into an expected short rate path of three point eight percent plus a term premium of zero point five percent](/imgs/blogs/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations-1.png)

The diagram above is the mental model for the entire post, and everything else is an unpacking of it. On the left is what most people assume a long yield is: one number, read as a pure forecast. On the right is what it really is: a *sum*. Part of the ten-year yield is the market's best guess at the average level of short-term rates over the next ten years — call it the **expectations** piece. The other part is an extra slice of yield the market tacks on to compensate lenders for the genuine risks of going long — the **term premium**. A 4.30% ten-year yield might be 3.80% of expectation and 0.50% of premium. The curve slopes up partly because the market expects rates to drift higher, and partly — often mostly — because that premium is positive.

This is post #17 in *The Bond Market, From the Ground Up*, and it is the theory chapter of the yield-curve track. We have already built the machinery this rests on: present value and [discounting](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced), the [spot (zero) rates](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping) that give one clean rate per maturity, and the [forward rates](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be) hiding inside the curve. Forwards told us *what the curve implies* about future rates. This post asks the deeper question those forwards raise but cannot answer on their own: how much of the curve's shape is genuine forecast, and how much is a risk premium dressed up as a forecast? Get this distinction right and the yield curve stops being a wall of numbers and becomes a readable story about expectations and fear.

## Foundations: the vocabulary you need before we build the curve

Almost every confusion about the yield curve is really a confusion about *which rate* someone is talking about. Before we can decompose anything, we need a shared, precise vocabulary, built from zero. If you have read the earlier posts in this series, this will be review you can skim; if you are new, do not skip it.

**An interest rate is the price of money over time.** When you lend \$100 for a year at 5%, you get \$105 back: the \$5 is the *rent* on your money. A higher rate means money rented out for a year is more expensive. Everything in fixed income is ultimately about this one price and how it varies with *how long* you rent the money out and *who* you rent it to.

**A bond is a loan you can trade.** When the US government wants to borrow, it sells a bond: you hand over money today, and the government promises to pay you fixed amounts (coupons) along the way and your principal back at the end (maturity). The *yield* of a bond is the single annual return you earn if you buy it at today's price and hold it to maturity — it is the bond's price expressed as a rate. We unpacked this fully in [the anatomy of a bond](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer); here, just hold onto "yield = the annual rate the bond pays you at today's price."

**A spot rate (or zero rate) is the single annual rate that applies from today out to one specific maturity.** When someone says "the two-year rate is 4%," they almost always mean: lock your money up for two years, do nothing, and it grows at an effective 4% per year. Crucially, a spot rate is an *average* over its whole period — the two-year rate blends everything the market expects to happen across both years into one number. The set of all spot rates, plotted against maturity, *is* the yield curve. We built spot rates from scratch in [the bootstrapping post](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping); here we take them as given.

**A short rate is the rate for the shortest period the market trades** — overnight, or a few months — *at a single point in time*. Today's short rate is just today's overnight or three-month rate; you can read it off a screen. The *future* short rate is whatever that rate will actually be on some future date, which nobody knows today. This distinction — today's *known* short rate versus tomorrow's *unknown* short rate — is the hinge the whole post turns on.

**A forward rate is the rate for a future period, implied today by the existing curve.** If the one-year spot rate is 3% and the two-year spot rate is 4%, the curve implicitly contains a rate for the *second* year alone — the one-year-one-year forward — and a little arithmetic (no-arbitrage) pins it at about 5.01%. We derived this in full in [the forward rates post](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be). The key fact we carry forward: a forward rate is *not* the same thing as the expected future short rate. The gap between them is — you guessed it — the term premium, and this post is where we finally pin that gap down.

A few smaller terms, defined inline so nothing trips you later. A *basis point* (bp) is one hundredth of a percent — 0.01% — so 50 bps is half a percentage point; rate moves are quoted in basis points because the differences that matter are small. *Duration* is, loosely, how sensitive a bond's price is to a change in rates: a long bond has high duration, meaning its price swings a lot when rates move (we covered this in depth in [the duration post](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income)). *Risk-averse* means, between two bets with the same average payoff, you prefer the one with less uncertainty — the assumption that, more than any other, explains why the term premium exists at all. *No-arbitrage* means there is no way to make a guaranteed, riskless profit out of nothing; it is the iron law that keeps the curve internally consistent.

### Our running example: a Treasury benchmark and Northwind Corp

To keep everything concrete, we will lean on two issuers throughout, the same pair the rest of this series uses. The first is the **US Treasury** — the risk-free benchmark, where the curve is cleanest because there is no default risk muddying the rates. When we say "the one-year rate is 3%," picture a one-year Treasury bill. The second is a fictional investment-grade company, **Northwind Corp**, whose bonds yield a bit more than Treasuries to compensate for the small chance it does not pay you back. Most of this post lives on the Treasury curve, because the expectations-versus-premium logic is exact there. Near the end we will note how Northwind's curve carries a *second* term premium on top — a credit term premium — but the core ideas are identical.

### What the curve actually looks like, and the three "normal" shapes

Before theory, a picture of the thing itself. The yield curve has three textbook shapes:

- **Normal (upward-sloping):** long yields above short yields. The everyday state. The 2y might be 4.0% and the 10y 4.5%.
- **Flat:** long and short yields roughly equal across maturities. Often a transition state, and a warning sign.
- **Inverted (downward-sloping):** short yields *above* long yields. Rare, uncomfortable, and historically a strong recession signal — the [classic inversion-recession link](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) the macro track covers in detail.

The puzzle this post solves is why "normal" is normal — why the up-slope is the default and the other shapes are the exceptions that need a story. The answer is the term premium. Let's build it.

## Part one: the expectations hypothesis — a long rate as an average of short rates

Start by pretending, for a few pages, that investors do not care about risk at all — that they are perfectly indifferent between a sure thing and a coin flip with the same average payoff. This is a lie, but it is a *useful* lie, because it isolates the first of our two forces. Under this assumption, the yield curve is governed entirely by what's called the **expectations hypothesis** (sometimes the "pure" expectations hypothesis to flag exactly this no-risk assumption).

The expectations hypothesis says this: **a long-term interest rate is simply the average of the short-term rates the market expects to prevail over the bond's life.** That's it. The two-year rate is the average of this year's one-year rate and next year's expected one-year rate. The ten-year rate is the average of the ten one-year rates the market expects over the next decade. The curve's shape just *is* the shape of the expected short-rate path, smeared into an average.

Why would that be true? Because of the same no-arbitrage logic that gave us forward rates. If you have a two-year horizon, you can either lock in the two-year rate, or invest one year and roll into next year's one-year rate. If everyone were risk-neutral, those two strategies would have to offer the *same expected* return — otherwise everyone would pile into whichever looked better, moving prices until the gap closed. Setting the locked-in two-year growth equal to the *expected* roll-over growth is exactly what makes the long rate an average of expected short rates.

![An XY chart with maturity in years on the horizontal axis and yield in percent on the vertical axis, showing a dashed rising line for the expected short rate path and a solid line below it for the yield curve which is the running average of that path](/imgs/blogs/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations-2.png)

The chart above makes the mechanism visible. The dashed line is the market's *expected path* of future one-year rates: it starts at 3% and climbs toward 4%-plus as the years go on — the picture of a market that thinks rates are headed up. The solid line is the yield curve that results: each point is the *running average* of the expected one-year rates up to that maturity. Notice two things. First, the curve slopes up — purely because the path it is averaging slopes up. Second, the curve sits *below* the path: an average always lags the rising series it is averaging, because the early, lower numbers drag the average down. That lag is not a flaw; it is the signature of averaging, and it is why the *short end* of the curve hugs today's short rate while the *long end* drifts toward where rates are expected to settle.

This already explains an enormous amount. A normal, up-sloping curve, under pure expectations, *means* the market expects short rates to rise. A flat curve means it expects them to stay put. An inverted curve means it expects them to *fall*. The curve becomes a forecast you can read directly. Let's nail the arithmetic with a worked example.

#### Worked example: building a two-year rate from expected one-year rates

You are deciding between two ways to be invested, risklessly, for two years, with \$1,000. Today's one-year Treasury rate is 3.00%. The market expects next year's one-year rate to be 5.00%.

**Path A — roll one-year bills.** Year one, your \$1,000 grows at 3%: \$1,000 × 1.03 = \$1,030. Year two, you reinvest the \$1,030 at the *expected* 5%: \$1,030 × 1.05 = \$1,081.50.

**Path B — lock in the two-year rate.** Whatever two-year rate $r_2$ the market quotes, you get \$1,000 × (1 + $r_2$)² at the end.

Under pure expectations, these must give the same *expected* ending wealth, so \$1,000 × (1 + $r_2$)² = \$1,081.50. Solving: (1 + $r_2$)² = 1.0815, so 1 + $r_2$ = √1.0815 = 1.03995, giving $r_2$ ≈ **4.00%**.

Look at what happened. The two-year rate (4.00%) landed almost exactly halfway between the first-year rate (3%) and the expected second-year rate (5%) — it's the *average* of the two, just as the hypothesis promised. The two-year rate is higher than the one-year rate *only because* the market expects the second year's rate to be higher.

*The intuition: under pure expectations, a longer rate is just a longer average — the curve slopes up exactly to the extent the market expects short rates to climb.*

#### Worked example: extending the average out to ten years

Now stretch the same logic to a ten-year yield. Suppose the market's expected path of one-year rates over the next decade is, year by year: 3.00, 3.40, 3.70, 3.90, 4.00, 4.10, 4.10, 4.10, 4.15, 4.20 percent. (A path that rises quickly at first, then flattens out around 4.1% — a very typical "rates normalize upward then settle" shape.)

The pure-expectations ten-year yield is the average of those ten numbers. Add them: 3.00 + 3.40 + 3.70 + 3.90 + 4.00 + 4.10 + 4.10 + 4.10 + 4.15 + 4.20 = 38.65. Divide by 10: **3.865%**, which we'll round to **3.80%** once we account for the fact that the true math compounds geometrically rather than arithmetically (the geometric average of a rising path is slightly below the simple average, a small correction we'll keep in mind).

So under pure expectations alone, this market produces a ten-year yield of about 3.8% — and a clearly up-sloping curve, since the one-year rate is 3.0% and the ten-year is 3.8%. The entire up-slope, in this risk-free fantasy, is the expected rise in short rates.

*The intuition: the ten-year yield is a ten-year weather forecast for the price of money — it tells you the average short rate the market expects, no more and no less, if risk didn't matter.*

### Why the pure expectations hypothesis is beautiful — and wrong

The expectations hypothesis is elegant, and it captures something real: expectations genuinely *are* a big part of the curve's shape. But as a complete theory it is decisively, measurably wrong, and it is worth being precise about *how* it fails, because the failure is exactly the term premium walking in the door.

Here is the problem. If the pure expectations hypothesis were true, then *on average* it should make no difference whether you hold long bonds or roll short bills — the expected return should be identical. But it is not. Over the long run of market history, holding longer-maturity bonds has earned a *higher average return* than rolling short bills. The extra return is small per year but persistent and real. If long and short were truly expected to pay the same, that excess would not exist. Its existence is the fingerprint of a risk premium: investors have been *paid* to hold the riskier long bonds.

There's a second tell. The expectations hypothesis implies that forward rates are unbiased forecasts of future short rates. But when researchers compare the forward rates the curve implied to the short rates that *actually* showed up later, forwards turn out to be *biased high* — they consistently predicted higher future short rates than materialized. That systematic over-prediction is not the market being dumb. It is the term premium: forwards sit *above* expected short rates by exactly the premium investors demand. The curve was never lying about its forecast; it was quoting forecast-plus-fee, and naive readers mistook the whole thing for forecast.

So the expectations hypothesis is the right *first half* of the story. We now add the second half.

## Part two: the term premium — getting paid to take duration risk

Drop the fantasy. Investors *are* risk-averse: faced with two strategies that have the same expected return but different risk, they prefer the safer one, and they will only take the riskier one if it pays *more*. Now ask the obvious question: is a long bond riskier than rolling short bills?

Yes — in three concrete, separable ways.

**Interest-rate (price) risk.** A long bond's price swings violently when rates move, because of its high duration. If you buy a ten-year bond and rates jump 1%, your bond's market price drops roughly 8–9% overnight — a real loss if you have to sell. Roll short bills instead and a rate jump barely dents you; your bill matures in months and you reinvest at the new, higher rate. We quantified exactly this in [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much). The long bond exposes you to far more mark-to-market pain.

**Inflation risk.** A long bond pays you fixed dollars far into the future. If inflation surprises higher over those years, those future dollars buy less than you bargained for — your *real* return gets eaten. The longer the bond, the more years of inflation surprise you're exposed to. A three-month bill barely cares about long-run inflation; a thirty-year bond is a thirty-year bet that inflation behaves.

**Liquidity and commitment risk.** Tying your money up for ten years means ten years of not having it for something else — an emergency, a better opportunity, a change of plans. Even though you *can* sell a Treasury bond any day, you can only sell it at *whatever price the market gives you that day*, which may be a loss. Short bills give you your cash back, at par, soon and reliably.

Stack those three up and the conclusion is forced: a long bond is genuinely riskier than rolling short bills, so a risk-averse investor will *not* hold it for the same expected return. They demand extra. That extra expected return — the yield on the long bond *above and beyond* the average expected short rate — is the **term premium**. It is the price of duration, the fee the market pays you to bear the risks of going long.

![A before and after comparison contrasting a world where investors do not care about risk and the curve is flat with a world where risk averse investors demand half a percent more to hold a ten year bond so the curve slopes up](/imgs/blogs/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations-5.png)

The figure above is the whole liquidity-preference idea in one picture. On the left, in the risk-neutral fantasy, a 4%-expected long bond and a 4%-expected roll of bills are equally attractive, so investors are indifferent and the curve is flat. On the right, in the real world, the long bond carries extra duration and inflation risk, so investors will only hold it if it offers, say, 4.5% — a half-percent term premium — to compensate. That premium lifts the long yield above the short, and the curve slopes up *even when the market expects rates to stay flat*. This is the deep reason "normal" is up-sloping: there is a structural, almost-always-positive premium baked into long yields, on top of whatever the expectations piece is doing.

The formal name for this is **liquidity preference theory** (John Hicks's term) or, in a richer version, **preferred-habitat theory** (Modigliani and Sutch). Liquidity preference says investors *generically* prefer shorter, more liquid holdings and must be bribed with a premium to go long — so the premium rises with maturity and the curve has a built-in upward tilt. Preferred-habitat refines this: different investors have natural *home* maturities (a pension fund wants thirty-year bonds to match its thirty-year liabilities; a money-market fund wants bills), and the premium at any maturity depends on the *imbalance* of supply and demand in that maturity's neighborhood. If everyone wants thirty-year bonds and few are issued, the thirty-year premium can even go *negative*. We'll see that happen in real markets later. For now, the takeaway is that the premium is positive *on average* but not guaranteed positive everywhere, and it varies.

#### Worked example: the term premium as the wedge in a ten-year yield

Return to our running market. The pure-expectations ten-year yield, from the average of the expected short-rate path, was **3.80%**. Now suppose investors demand a **0.50%** (50 bp) term premium to hold the ten-year bond rather than roll bills for a decade.

The actual ten-year yield the market quotes is the sum:

$$y_{10} = \underbrace{3.80\%}_{\text{expected short rates}} + \underbrace{0.50\%}_{\text{term premium}} = 4.30\%$$

Now compare across the curve. The one-year yield is essentially all expectation (a one-year bond has tiny duration risk, so its premium is near zero): about 3.00%. The ten-year is 4.30%. The curve's total up-slope from 1y to 10y is 1.30%. Of that, 0.80% (= 3.80% − 3.00%) is the *expectations* piece — the market thinks short rates are heading up — and 0.50% is the *term premium* piece. The slope is forecast and fee, mixed together in one line on the screen.

*The intuition: the quoted long yield is forecast plus fee — and if you read the whole slope as forecast, you will badly overestimate how much the market expects rates to rise.*

![An XY chart with maturity in years on the horizontal axis and yield in percent on the vertical axis, showing a dashed lower line for the pure expectations curve and a solid higher line for the actual curve with the widening vertical gap between them labeled as the term premium](/imgs/blogs/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations-3.png)

The chart above is the single most important picture in the post, so sit with it. The dashed line is the pure-expectations curve — the curve you'd get from the forecast alone, the same solid line from our earlier figure. The solid line is the *actual* curve the market quotes. The vertical gap between them, at each maturity, is the term premium. Three features matter. First, the gap is *positive* — the real curve sits above the expectations curve — which is why curves normally slope up more than pure expectations would justify. Second, the gap *widens with maturity* — a thirty-year bond carries far more duration and inflation risk than a two-year, so it commands a fatter premium. Third, the gap is the *whole reason* you can't read a curve as a pure forecast: the steepness you see is forecast *plus* a maturity-increasing fee, and untangling them is the central skill of yield-curve analysis.

## The decomposition, stated cleanly

We can now write the central equation of this entire post. For a bond of maturity $n$:

$$y_n = \underbrace{\bar{r}_n}_{\text{expected average short rate}} + \underbrace{TP_n}_{\text{term premium}}$$

where $y_n$ is the $n$-year yield, $\bar{r}_n$ is the average of the short rates the market expects over the next $n$ years, and $TP_n$ is the term premium for maturity $n$. Every long yield in the world is, to a very good approximation, these two numbers added together. The first is a *forecast* — it changes when the market changes its mind about the economy and the central bank. The second is a *risk premium* — it changes when risk, supply, and the appetite for safe assets change. They have different drivers, they move for different reasons, and the entire art of reading the bond market is figuring out, when a yield moves, *which* piece moved.

This decomposition is not just a teaching device. Central banks and researchers estimate it for real, using statistical term-structure models. The two most famous estimates of the US ten-year term premium are the **ACM model** (Adrian, Crump, and Moench, from the New York Fed) and the **Kim-Wright model** (from the Federal Reserve Board). Both publish a time series of the estimated term premium going back decades. Their estimates are *model-dependent* — you cannot observe the term premium directly, only infer it — and they disagree at the margin, which is itself an honest reminder that the split between "forecast" and "premium" is an estimate, not a measurement. But both tell the same broad story: the term premium was large and positive in the high-inflation 1980s, fell steadily for thirty years, and spent much of the 2010s and early 2020s near zero or even *negative* — a remarkable state where investors were willing to hold long bonds for *less* than the expected average short rate, because they valued the bonds' safety and hedging properties so highly.

#### Worked example: decomposing a real-ish ten-year move

Suppose the ten-year Treasury yield rises from 4.00% to 4.40% over a month — a 40 bp move, the kind that makes headlines. A naive reader says "the market now expects much higher rates." But the decomposition forces a better question: which piece moved?

Case 1 — *expectations moved.* Strong jobs and inflation data make the market expect the central bank to hold rates higher for longer. The expected-average-short-rate piece rises from 3.60% to 4.00% (+40 bp); the term premium is unchanged at 0.40%. Yield goes 4.00% → 4.40%. This is a *forecast* repricing — the economy looks hotter.

Case 2 — *the premium moved.* A surge in Treasury issuance to fund a widening deficit floods the market with long bonds, and buyers demand a discount. The expected-short-rate piece is unchanged at 3.60%; the term premium rises from 0.40% to 0.80% (+40 bp). Same 4.00% → 4.40% yield, same headline — but a completely different *meaning*. Here the economy hasn't necessarily changed; investors are just charging more to absorb the supply.

The two cases have identical headlines and opposite implications. In Case 1, the bond market is forecasting a stronger economy and tighter policy. In Case 2, it is reacting to supply and risk with no view change on the economy at all. Anyone trading, hedging, or forecasting off the curve *must* know which one it is — and the only way to know is to estimate the decomposition.

*The intuition: the same yield move can be a forecast or a fee depending on which piece moved, so "rates rose" is never a complete sentence — always ask which half.*

## Why the curve normally slopes up — and what it takes to flatten or invert

Now we can answer the title question precisely. The curve normally slopes up because of the *combination* of the two forces, and in particular because the term premium provides a positive baseline tilt that the expectations piece has to actively fight to overcome.

Decompose the slope itself. The slope from the short end to the long end is:

$$\text{slope} = (\bar{r}_{\text{long}} - r_{\text{short}}) + (TP_{\text{long}} - TP_{\text{short}})$$

The first bracket is the *expectations slope*: positive if the market expects short rates to rise, negative if it expects cuts. The second bracket is the *premium slope*: almost always positive, because the term premium grows with maturity (long bonds are riskier than short ones). So:

- **Normal (up-sloping):** the usual state. The premium slope is positive, and either the market expects rates roughly flat-to-up, or it expects modest cuts that the premium slope more than offsets. Both forces push up, or the premium wins.
- **Flat:** the two forces roughly cancel. Typically the market expects *cuts* (negative expectations slope) that exactly offset the positive premium slope. A flat curve is the market saying "we expect rate cuts ahead, but you're still paid a premium for duration, and the two net out."
- **Inverted (down-sloping):** the rare, dramatic state. The expectations slope must be *strongly negative* — the market expects *big* rate cuts — by *more* than the positive premium slope can hold up. An inverted curve is the market forecasting a sharp future fall in short rates, usually because it expects a recession to force the central bank to cut hard. The premium is *still positive*; it's just being overwhelmed by a steeply falling forecast.

This reframes the famous inversion-recession signal beautifully. An inverted curve is not magic; it is the expectations piece screaming "rates are coming down a lot, soon" loudly enough to drag the whole curve below today's short rate *despite* the premium. And rates come down a lot, soon, mainly when the economy is headed for trouble. So inversion is a recession signal *because* it encodes a forecast of aggressive future cuts. The macro track works through [the slope-inversion-recession link](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) and [how the central bank actually sets the short rate](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) the curve is forecasting; here, the point is that the term premium is the reason inversion is *rare* — the premium tilt means the forecast has to be dramatically negative to flip the whole curve.

#### Worked example: how much expected cutting it takes to invert the curve

Suppose the term premium adds a steady +0.50% tilt from the 2-year to the 10-year point (premium slope = +0.50%). Today's 2-year yield is 4.50%. For the 10-year yield to fall *below* the 2-year — an inversion — the expectations slope has to overcome the +0.50% premium tilt and then some.

Concretely, the 10-year yield is `(expected average short rate over 10y) + TP_10`. For the 10-year to sit at, say, 4.20% (30 bp *below* the 2-year's 4.50%, a textbook inversion), and with TP_10 around 0.50%, the expected average short rate over ten years must be about 4.20% − 0.50% = **3.70%**. But the 2-year already implies short rates near 4.5% now. So the market must expect short rates to average a full 0.80% *below* today's level over the next decade — i.e., it's pricing substantial rate cuts, front-loaded into the next couple of years.

That's the quantitative content of an inversion: not "rates will be a hair lower" but "the central bank will cut meaningfully, and soon, probably because the economy weakens." The premium tilt is precisely why a *small* expected cut leaves the curve still up-sloping, and only a *large* one inverts it.

*The intuition: because the term premium props the long end up, inverting the curve takes a forecast of serious rate cuts — which is why inversion has been such a reliable recession alarm.*

## What moves the term premium — the second variable hiding in every yield

If the term premium were a fixed constant, this post would be simpler and the bond market would be calmer. It is not. The premium *breathes*, sometimes dramatically, and those breaths are some of the biggest yield moves in history that have nothing to do with the rate forecast. Understanding what pushes it around is the difference between a beginner who reads every yield move as a forecast change and a practitioner who knows when the market is just repricing risk.

![A matrix figure with four rows for inflation uncertainty bond supply safe asset demand and growth risk and two columns showing what pushes the term premium up versus down for each driver](/imgs/blogs/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations-6.png)

The matrix above lays out the four main drivers and which direction each pushes the premium. Walk through them.

**Inflation uncertainty.** This is the single biggest driver of the term premium over history. Remember, a fixed-coupon long bond is a bet that inflation behaves. When inflation is wild and unpredictable — as in the 1970s and early 1980s — that bet is terrifying, and investors demand a fat premium to make it. When inflation is low and *credibly anchored* by a trusted central bank — as in the 2000s and 2010s — the bet is easy, and the premium shrinks toward zero. The thirty-year decline in the US term premium from the 1980s to the 2010s is, in large part, the story of inflation being tamed and inflation *uncertainty* collapsing. It is not that the market forecast lower rates the whole time; it is that the *fee* for inflation risk evaporated.

**Bond supply.** The government does not choose how much you should be paid to hold its debt — the market does, and the market charges more when there's more debt to absorb. When deficits widen and the Treasury issues a flood of long bonds, or when the central bank runs *quantitative tightening* (QT) by shrinking its bond holdings and letting more bonds land in private hands, the supply of duration the market must hold goes up, and the premium rises to entice buyers. Conversely, *quantitative easing* (QE) — the central bank *buying* long bonds and taking them out of the market — shrinks the supply of duration the private sector holds and *compresses* the premium. A huge fraction of QE's effect on yields runs through exactly this channel: not by changing the rate forecast, but by squeezing the term premium. The macro track covers [how deficits and issuance move yields](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields) in depth.

**Safe-asset demand.** Treasuries are the world's premier safe asset — the thing everyone runs *to* in a crisis. When fear spikes, a wall of money floods into Treasuries (a "flight to quality"), bidding their prices up and yields *down*, compressing the term premium and sometimes driving it negative. The same happens structurally when foreign central banks and reserve managers park trillions in Treasuries as reserves: that persistent, price-insensitive demand holds the premium down for years. The flip side: if that demand *retreats* — foreign buyers step back, a reserve manager sells — the premium springs back up. Much of the debate about whether US yields will stay "structurally low" is really a debate about whether global safe-asset demand for Treasuries will persist.

**Growth and volatility risk (the hedge value of bonds).** Here is a subtle one. Bonds are often a *hedge* for a stock portfolio: when the economy tanks, stocks fall but bonds usually rally (rates get cut), so bonds cushion the blow. An asset that pays off exactly when you're in pain is *valuable as insurance*, and investors will accept a *lower* yield — even a negative term premium — for it. This is why, in the low-inflation 2010s, the term premium could go negative: bonds were such good stock-hedges that people paid up (accepted low yields) for the insurance. But this hedge relationship is not permanent. In 2022, when inflation drove rates up, stocks *and* bonds fell together — bonds stopped hedging — and that loss of hedge value was part of why the term premium rose. We dig into this in [the stock-bond correlation post](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).

#### Worked example: how a supply shock moves a yield without touching the forecast

The Treasury announces it will sharply increase issuance of 10- and 30-year bonds next quarter to fund a wider deficit. The economic outlook hasn't changed — the market's forecast for short rates is exactly the same. What happens to the 10-year yield?

The expected-average-short-rate piece, $\bar{r}_{10}$, is unchanged at, say, 3.80%. But the market now has to absorb, say, an extra \$200 billion of long-duration bonds. To get investors to hold that extra duration, the term premium has to rise — say from 0.40% to 0.60% (+20 bp). The 10-year yield goes from 4.20% to **4.40%**.

A reporter writes "bond market signals expectations of higher rates." Wrong. The forecast didn't move at all. The market is simply charging 20 bp more *fee* to hold the extra supply. If you were hedging a mortgage pipeline or pricing a corporate bond off that 10-year, treating this as a forecast change would lead you badly astray.

*The intuition: supply moves the fee, not the forecast — and the screen looks identical either way, which is why the decomposition is worth real effort.*

## The influence: why the term premium is real money, not an accounting fiction

It would be fair, at this point, to be suspicious. The term premium is *estimated*, not observed. It's the gap between a quoted yield and an unobservable forecast. Isn't it just a fudge factor — the name we give to "the part of the yield we can't explain"? This is the most important objection in the whole subject, and the answer is no, for a concrete, testable reason: **a higher term premium today predicts higher realized excess returns on long bonds tomorrow.** If the premium were a meaningless residual, it would have no predictive power. It does.

![A bar chart with the starting term premium on the horizontal axis going from negative to high and the realized excess return of long bonds over short bills on the vertical axis showing taller bars and an upward trend line for higher starting premiums](/imgs/blogs/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations-4.png)

The chart above is the influence figure, and it is the empirical payoff of the whole theory. Take the history of the bond market. Sort every period into buckets by how *fat* the term premium was at the start. Then measure, for each bucket, what the long bond *actually earned* over and above rolling short bills in the period that followed. The pattern is clear and is one of the more robust findings in finance: when the starting premium was *negative*, long bonds barely beat bills (or lost to them); when the starting premium was *fat*, long bonds earned a large excess return. The premium is *compensation you can collect*, not an accounting ghost — when the market is paying you a lot to hold duration, holding duration has, on average, paid off.

This is the heart of the famous **Cochrane-Piazzesi** result and a long line of related research: a single factor extracted from the shape of the forward curve (closely related to the term premium) predicts bond excess returns with surprising power, and the relationship is *not* explained by changing rate forecasts. It's the premium doing real work. The practical upshot for anyone who holds bonds: the *level* of the term premium is a rough gauge of whether you're being well-paid or poorly-paid to take duration risk right now. A negative term premium is the market telling you it's *expensive* to own the insurance that long bonds provide; a fat positive premium is the market paying you generously to lend long.

#### Worked example: collecting the premium versus the forecast

You hold a 10-year Treasury for one year, then sell it. Your return has two sources. First, the *carry and roll*: you earn the 10-year's yield, and as the bond ages into a 9-year bond it may roll down the (up-sloping) curve to a lower yield, lifting its price (we covered this "rolling down the curve" return in [the forward rates post](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be)). Second, *price changes from yield moves* over the year.

Suppose you buy at a 10-year yield of 4.30% (3.80% expectation + 0.50% premium). Over the year, the rate forecast turns out *exactly right* — short rates evolve just as expected — so the expectations piece is unchanged. And the term premium is also unchanged. Then your one-year return is essentially the yield you earned plus the roll-down: you *collected the premium*. You earned more than rolling bills (which paid roughly the expected short rate) precisely by the term-premium amount, plus roll. That extra is the premium showing up as cash in your account — the realized excess return the influence chart is built from.

Now flip it. Suppose over the year the term premium *jumps* from 0.50% to 1.00% (a 50 bp premium widening, no forecast change). The 10-year yield rises 50 bp, and your bond's price *falls* about 4.5% (its duration times the yield move). That capital loss swamps the year's coupon, and you *lose* money — even though the rate forecast was dead-on. You were paid a premium to bear duration risk, and that risk *materialized*: the premium itself moved against you.

*The intuition: the term premium is both the reward for holding duration and the source of its risk — you get paid the premium for bearing the chance that the premium itself lurches.*

## Putting the pieces together: the full decomposition, worked end to end

Let's assemble everything into one clean, fully-worked construction of a 10-year yield, so you can see the whole machine run.

![A matrix figure laying out a year by year construction of a ten year yield from the expected one year rate path through the running average to the pure expectations yield of three point eight percent plus a fifty basis point term premium giving four point three percent](/imgs/blogs/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations-7.png)

The matrix above is the recipe card. Read it left to right, top to bottom.

**Row 1 — the expected short-rate path.** The market expects one-year rates of 3.00, 3.40, 3.70, 3.90 (years 1–4) and 4.00, 4.10, 4.10, 4.10, 4.15, 4.20 (years 5–10). This is the *forecast*, the input the expectations hypothesis averages.

**Row 2 — the running average.** After four years, the average of the path so far is about 3.50%. By year ten, the back years pull the average up to about 3.80%. That 3.80% is the *pure-expectations 10-year yield* — the yield you'd quote if risk didn't matter.

**Row 3 — add the term premium.** Investors demand 0.50% extra to hold ten years of duration and inflation risk. That's the *fee*.

**Row 4 — the quoted yield.** 3.80% expectation + 0.50% premium = **4.30%**. That's the number on the screen. Forecast plus fee, assembled from scratch.

#### Worked example: re-pricing the whole curve when the forecast and the premium both move

One last example, to show the machine handling a realistic, messy shock. Start from our 10-year at 4.30% (3.80% + 0.50%). Now two things happen at once: (1) a soft inflation report makes the market expect *lower* short rates — the expected average drops from 3.80% to 3.50% (−30 bp); and (2) the same report calms inflation fears, *also* compressing the term premium from 0.50% to 0.35% (−15 bp).

New 10-year yield: 3.50% + 0.35% = **3.85%**. The yield fell 45 bp. Of that drop, 30 bp was a *forecast* change (the market now expects lower rates) and 15 bp was a *premium* change (less inflation fear, smaller fee). A headline says "10-year yield plunges 45 bp on soft inflation data" — true, but it bundles two distinct stories. The forecast piece tells you the market sees a softer economy; the premium piece tells you it sees less inflation *risk*. They happened to move the same direction here, reinforcing each other, but they need not — and an analyst who can't split them can't tell whether the move is mostly about growth or mostly about risk.

*The intuition: real yield moves are usually a blend of forecast and fee moving together — reading the curve well means estimating the split, not just the total.*

## How the term premium shows up for credit and other curves

A quick but important extension before the closers. Everything above was built on the risk-free Treasury curve. What about Northwind Corp, our investment-grade issuer?

Northwind's yield curve carries *everything* the Treasury curve does — an expectations piece and a Treasury term premium — *plus* a second layer: a **credit spread** that itself has an expectations piece (expected default losses) and a **credit risk premium** (extra pay for bearing default *uncertainty*). And just as the Treasury term premium grows with maturity, the *credit* term premium typically grows with maturity too: a 10-year Northwind bond is exposed to ten years of "could this company deteriorate?" risk, versus two years for a 2-year bond. So a corporate curve is normally even steeper than the Treasury curve, because it stacks a rising credit-risk premium on top of a rising rate-term premium. We unpack credit spreads fully in [the corporate-credit post](/blog/trading/cross-asset/corporate-credit-investment-grade-high-yield-spreads); the structural lesson is that the "forecast plus fee" decomposition is *recursive* — every layer of risk you add to a bond adds its own expectations piece and its own risk premium.

## Common misconceptions

**"An upward-sloping curve means the market expects rates to rise."** This is the single most common error, and the whole post is its correction. An up-sloping curve is *forecast plus premium*. Because the term premium is usually positive, the curve can slope up even when the market expects rates to stay *flat* or fall *modestly*. Reading the entire slope as a rate forecast systematically overestimates how much the market expects rates to rise — by exactly the term premium.

**"Forward rates are the market's prediction of future rates."** Forwards are *forecast plus premium*, just like long yields — they sit *above* expected future short rates by the term premium. They are biased-high forecasts. Treating a forward as a clean prediction means you'll consistently expect rates to be higher than the market actually does. Forwards are breakevens, not crystal balls.

**"The term premium is just the part of the yield we can't explain — a fudge factor."** No: a higher term premium *predicts higher realized excess returns* on long bonds, which a meaningless residual could not do. It's real, collectible compensation for real risk. It's *unobservable* (we estimate it, we don't measure it), which is a genuine limitation — but unobservable is not the same as fictional.

**"The term premium is always positive."** Usually, but not always. In the 2010s, US term-premium estimates went *negative* for stretches: investors accepted *less* than the expected average short rate to hold long bonds, because the bonds' safety and stock-hedging value were worth paying for. Preferred-habitat effects (a glut of demand for a particular maturity) can also drive a local premium negative. "Normally positive" is the right intuition; "always positive" is wrong.

**"If the curve is flat, the market has no view."** A flat curve is not a shrug. With a positive term premium baked in, a flat curve usually means the market expects *rate cuts* — the negative expectations slope is exactly offsetting the positive premium slope. A flat curve is the market expecting easing, not the market expecting nothing.

**"A steeper curve is always good for the economy / a bullish signal."** Steepening can mean opposite things depending on *which piece* steepened. A "bull steepener" (short yields falling fast as the market prices cuts) often precedes or accompanies a slowdown. A "bear steepener" (long yields rising on a fatter term premium, e.g. from supply or inflation fears) can signal stress about debt and inflation. Same shape change, different stories — you have to decompose it.

## How it shows up in real markets

**The 1970s–80s inflation premium.** In the high-inflation era, the US term premium was enormous — estimates put the 10-year premium at several percentage points. Investors had been burned by inflation eating their fixed coupons and demanded a fat fee to lend long. When Paul Volcker's Fed crushed inflation in the early 1980s (covered in the macro track's [Volcker case](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable)), the *premium* — not just the forecast — collapsed over the following decades as inflation credibility was rebuilt. The thirty-year bond bull market that followed was, in large part, a thirty-year compression of the inflation term premium. Anyone who read falling long yields purely as "the market expects lower rates" missed that half the move was a shrinking fee for inflation risk.

**QE and the 2010s negative premium.** When the Fed launched quantitative easing after 2008, it bought trillions of dollars of long Treasuries and mortgage bonds, pulling duration out of private hands. The explicit theory was the "portfolio balance channel": by shrinking the supply of duration the market had to hold, QE would *compress the term premium* and lower long yields without necessarily changing the rate forecast. Fed researchers' own estimates credited QE with pushing the 10-year term premium down by roughly 100 bp at its peak effect. By the mid-2010s, ACM and Kim-Wright estimates of the 10-year premium had gone *negative* — a striking state where, on paper, investors were lending the government for ten years at *less* than the expected average overnight rate, because Treasuries were such prized safe, stock-hedging assets in a low-inflation world.

**The 2013 "taper tantrum."** In May 2013, when Fed Chair Ben Bernanke hinted that QE bond-buying might slow ("taper"), the 10-year yield jumped about 100 bp over a few months. Critically, the rate-*forecast* piece barely moved — the market didn't suddenly expect dramatically higher short rates. What spiked was the *term premium*: the prospect of less Fed buying (and thus more duration for private investors to absorb) snapped the premium back up. The taper tantrum is the cleanest real-world demonstration that the premium is a distinct, volatile variable: a yield can lurch 100 bp on a *supply/premium* shock with the forecast nearly still.

**2022: the premium and the forecast move together.** In 2022, surging inflation drove a brutal bond selloff. Both pieces moved up at once: the *forecast* rose (the market priced aggressive Fed hikes) *and* the *term premium* rose (inflation uncertainty came roaring back, bonds stopped hedging stocks as both fell together, and heavy issuance loomed). The 10-year went from ~1.5% to over 4%. This episode is a reminder that the two pieces are not independent — a single shock (an inflation scare) can move *both* the rate forecast and the inflation risk premium in the same direction, amplifying the yield move. It also broke the comfortable [stock-bond hedge of the 60/40 portfolio](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine), because part of what made bonds good hedges — a low, stable term premium with negative stock correlation — reversed.

**The deficit-and-supply debate, 2023 onward.** From 2023, with large US deficits and the Fed running QT (shrinking its balance sheet, the opposite of QE), a live market debate raged over how much of rising long yields was *forecast* (a resilient economy keeping the Fed higher-for-longer) versus *term premium* (a flood of Treasury issuance forcing buyers to demand a fatter fee for duration). Term-premium estimates rose off their negative-2010s lows back toward positive territory. This is the decomposition playing out in real time as the central question for bond investors: every basis point of yield is contested between "the economy is strong" and "there's too much debt to absorb cheaply," and the answer determines whether the move reverses or sticks. The macro track's [deficits and bond supply post](/blog/trading/macro-trading/deficits-debt-bond-supply-why-issuance-moves-yields) and the discussion of [bond vigilantes](/blog/trading/macro-trading/sovereign-debt-and-the-bond-vigilantes) are essentially about a rising supply-driven term premium.

## When this matters to you, and where to go next

The expectations-versus-premium split is not an academic nicety — it touches your life through every long-term rate you ever face. Your 30-year mortgage rate is priced off the long end of the curve, so it carries a term premium: part of what you pay your lender is the fee the *bond* market charges for duration risk, passed through to you. When you read that "the 10-year yield rose and mortgage rates jumped," now you know to ask whether that was the market forecasting a stronger economy (forecast piece) or just charging more to hold duration (premium piece) — because the two have very different implications for whether the rate will come back down. The same split governs the discount rate that prices [every stock and long-lived asset](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything): when the term premium rises, the rate used to value future cash flows rises, and growth stocks (whose value is far in the future) fall hardest.

For the next step in this series, two directions branch out. To see *how the curve is read for the economy* — the inversion signal, the slope as a recession gauge — head to the macro track's [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession). To go *deeper into the math* — how term-structure models actually estimate the premium, how affine models work — the quant track's [yield-curve modeling post](/blog/trading/quantitative-finance/yield-curve-modeling) and [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics) take it from here. Within this series, the natural companions are [forward rates](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be), which gave us the no-arbitrage engine this post built on, and [duration](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income), which quantifies the very risk the term premium pays you to bear.

The one sentence to carry away: a long yield is a forecast plus a fee, the fee is usually positive, and learning to tell the two apart is the difference between reading the bond market and just looking at it. *(This is educational, not investment advice.)*
