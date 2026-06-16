---
title: "Forward rates: what the market expects rates to be"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How the yield curve secretly contains the market's forecast of future interest rates, why the math is just no-arbitrage, and what forwards can and cannot tell you."
tags: ["fixed-income", "bonds", "forward-rates", "yield-curve", "spot-rates", "no-arbitrage", "term-premium", "interest-rates"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **The big idea:** the shape of today's yield curve secretly contains the market's forecast of where short-term interest rates are going — and you can extract that forecast with nothing but arithmetic.
> - A **forward rate** is a rate for a future period that you can lock in *today*, implied by the rates that already exist on the curve.
> - It comes from **no-arbitrage**: investing for two years must earn the same as investing one year and rolling into a one-year rate set a year from now. The rate that makes those two paths tie is the forward.
> - Because of that, the upward slope of the curve *is* the market saying "we expect short rates to rise" — forwards are the curve's embedded forecast.
> - Forwards are also a **breakeven**: they are the future rate at which locking in and rolling over come out exactly even, so they tell you what has to happen for one strategy to beat the other.
> - But a forward is *not* a clean prediction — it sits a little above the true expected path by the **term premium**, the extra yield investors demand for tying up money. So forwards are a biased forecast, and the bias is real money.
> - "Rolling down the curve" — a bond aging into a lower yield on an upward-sloping curve — is a return source that falls straight out of the same math.

Here is a fact that sounds like a magic trick the first time you hear it. I can tell you, today, the interest rate the market expects to prevail one year from now — not by reading tea leaves, not by surveying economists, but by doing a single division on two numbers you can pull off any government-bond screen. The yield curve already contains that number. It has been hiding in plain sight the whole time.

That hidden number is the **forward rate**, and it is one of the most useful and most misunderstood objects in all of finance. Useful, because it converts the entire term structure of interest rates — the wall of yields for every maturity from overnight to thirty years — into a story about the *future*: where the market thinks the price of money is headed. Misunderstood, because people routinely take that story literally, as if the curve were a crystal ball, when it is closer to a crystal ball with a thumb on the scale.

![Two timelines showing two ways to be invested for two years, one locking in the two-year rate directly and the other investing one year and rolling into a future one-year rate, both ending at the same dollar amount](/imgs/blogs/forward-rates-what-the-market-expects-rates-to-be-1.png)

The diagram above is the mental model, and the whole post is really just an unpacking of it. You have \$1,000 and a two-year horizon. You can lock the money up for two years at the two-year rate. Or you can put it in a one-year investment, get your money back in a year, and then reinvest — *roll* — into whatever one-year rate exists at that point. Those two paths cover the exact same two years with the exact same (risk-free) money. If one reliably beat the other, you could borrow on the cheap path and lend on the rich one and pocket the difference for free. Markets do not leave free money lying around for long. So the two paths must tie — and the future one-year rate that makes them tie is the forward rate. Everything else in this post is consequences of that one sentence.

This is post #10 in *The Bond Market, From the Ground Up*. It is the capstone of the pricing track: we have built up present value, discounting, and the [spot (zero) rates that bootstrap the curve](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping). Spot rates told us the single rate that applies from *today* out to each maturity. Forward rates are the next, sharper question: given those spot rates, what does the market think the rate will be for a *future* period — say, the year that runs from one year out to two years out? Answering that turns a static snapshot of yields into a dynamic forecast, and it is the bridge to the whole yield-curve track that follows.

## Foundations: the three rates you need to keep straight

Before we can talk about forwards we need to nail down some vocabulary, because almost every confusion about forward rates is really a confusion about *which rate someone is quoting*. There are three, and they answer three different questions.

**A spot rate** (also called a *zero rate*) is the single annual interest rate that applies from today out to one specific maturity, for money you lock up the whole time. When someone says "the two-year rate is 4%," they almost always mean the two-year spot rate: invest today, do nothing, and your money grows at an effective 4% per year for two years. The spot rate is an *average* over the period — it blends together everything the market expects to happen between now and that maturity into one number. We built spot rates from scratch in the [spot-rate and bootstrapping post](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping); here we take them as given inputs.

**A forward rate** is the interest rate for a *future* period, agreed *today*. The cleanest example is the **one-year-one-year forward**, written *1y1y* and read "one-year, one-year": the rate for the one-year period that begins one year from now and ends two years from now. You are not investing for that year yet — you are extracting the rate the curve *implies* for it. There are forwards for every future window: the 2y1y (the one-year rate starting two years out), the 1y2y (a two-year rate starting one year out), the 3m3m (the three-month rate starting three months out), and so on. The forward is always a rate for a slice of time that has not started yet.

**A short rate** is the interest rate for the shortest period the market trades — overnight, or a few months — at a *single point in time*. Today's short rate is just today's overnight or three-month rate. The *future* short rate is what that rate will actually be on some future date, which nobody knows today. The deep punchline of this whole post is that **forward rates are the market's best public guess at future short rates** — with a caveat we will spend real time on.

A few smaller terms, defined inline so nothing trips you up later. A *basis point* (bp) is one hundredth of a percent — 0.01% — so 25 bps is a quarter of a percentage point; rate moves are quoted in basis points because the differences that matter are small. *No-arbitrage* means there is no way to make a guaranteed profit with zero risk and zero net investment; it is the single most powerful assumption in fixed-income pricing, because it pins down what prices and rates *must* be relative to each other. *Compounding* means earning interest on your interest: \$1,000 at 4% for two years is not \$1,000 + \$40 + \$40 = \$1,080, it is \$1,000 × 1.04 × 1.04 = \$1,081.60, because in the second year you earn 4% on the \$1,040 you had, not just on the original \$1,000.

#### Worked example: spot rate as a blended average

Suppose the one-year spot rate is 3.0% and the two-year spot rate is 4.0%. What does the 4% actually represent? It is the *single* rate that, applied for two years, grows your money the right amount: \$1,000 × (1.04)² = \$1,081.60. But why is the two-year rate higher than the one-year rate? The market is telling you it expects the rate for the *second* year to be higher than 3%, and the two-year spot is the average of the first-year rate (3%) and that expected second-year rate, blended over two years. The whole job of a forward rate is to *un-blend* that average and recover the second-year piece.

*The intuition: a spot rate is an average over the whole period, and a forward rate is the marginal piece you get when you strip the known early part out of a longer average.*

### Why "one rate" was always a lie

There is one more piece of groundwork, and it explains why forwards exist at all. When you first learn about bonds, you imagine a single "the interest rate." But there is no such thing — there is a *whole curve* of rates, one for every maturity, and the [spot-rate post](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping) showed why pretending otherwise is a mistake. Money locked up for ten years does not earn the same rate as money locked up overnight. The set of all those rates, plotted against maturity, is the *term structure of interest rates* — the yield curve.

Once you accept that there is a different rate for every horizon, an obvious question follows: these rates are not independent. The two-year rate cannot be just anything given the one-year rate, because — as we are about to prove — the two-year investment *contains* a one-year investment inside it. The relationships *between* the points on the curve are exactly what forwards capture. A forward rate is the answer to "given everything the curve already says about one-year and two-year money, what is it implicitly saying about the year *in between*?" Forwards are the connective tissue of the curve: they are what makes the curve internally consistent rather than a random scatter of numbers.

This is why forwards are not an optional advanced topic — they are baked into the very existence of a curve. The moment you have more than one maturity quoted, you have forwards, whether anyone computes them or not. They are sitting there in the ratios between the spot rates, the way the average speed of a journey already determines your speed on each leg if you know the distances.

### Our running example: a Treasury benchmark and Northwind Corp

To keep the numbers concrete, we will lean on two issuers throughout. The first is the **US Treasury** — the risk-free benchmark, where the curve is cleanest and the no-arbitrage logic is exact (no default risk to muddy the rates). When we say "the one-year rate is 3%," picture a one-year Treasury bill. The second is a fictional company, **Northwind Corp**, an investment-grade firm whose bonds yield a bit more than Treasuries to compensate for the small chance it does not pay you back — its *credit spread*, the extra yield over the risk-free rate. Forwards work the same way for Northwind as for Treasuries; the only difference is that Northwind's forwards embed a forward *credit spread* on top of the forward *risk-free rate*, a refinement we will touch on at the end. For the core mechanics, the Treasury curve is all we need, and it is where every forward calculation should start.

## The no-arbitrage engine: why two paths must tie

Now we can derive the forward properly. The argument is so clean it feels like cheating, and it is worth slowing down to feel why it works, because the same logic underpins almost everything in fixed income.

![A five-step pipeline deriving the forward rate from no-arbitrage, setting the two-year growth factor equal to the one-year factor times the unknown forward factor and solving for the forward](/imgs/blogs/forward-rates-what-the-market-expects-rates-to-be-2.png)

You have \$1,000 and you want to be invested, risklessly, for exactly two years. The market gives you two ways to do it.

**Path A — lock it in.** Buy a two-year zero-coupon government bond yielding the two-year spot rate, 4.0%. You do nothing for two years. At the end you have:

$$1000 \times (1.04)^2 = \$1{,}081.60$$

**Path B — roll it over.** Buy a one-year zero yielding the one-year spot rate, 3.0%. In one year you collect \$1,000 × 1.03 = \$1,030. Then you take that \$1,030 and reinvest it for the second year at whatever the one-year rate happens to be at that point. Call that unknown future rate $f$. At the end of year two you have:

$$1000 \times 1.03 \times (1 + f)$$

Here is the crucial observation. Both paths are *risk-free* (government bonds, no default risk) and both cover the *exact same two years* with the *exact same \$1,000*. If Path A reliably ended with more money than Path B, every rational investor would choose A, nobody would buy the one-year bond to roll, and the prices would move until the gap closed. If B reliably won, the reverse. The only stable, no-arbitrage outcome is that the two paths end with the *same* amount of money:

$$\underbrace{(1.04)^2}_{\text{lock in 2y}} = \underbrace{(1.03) \times (1 + f)}_{\text{roll 1y then 1y}}$$

The left side is fixed by the market. The only unknown is $f$, the future one-year rate that makes the equation balance. Solve it:

$$1 + f = \frac{(1.04)^2}{1.03} = \frac{1.0816}{1.0300} = 1.0501 \quad\Rightarrow\quad f = 5.01\%$$

That \$5.01% is the **1y1y forward rate**. It is not a rate anyone has promised to pay you. It is the rate that is *already implied* by the one-year and two-year spot rates sitting on the screen, because no-arbitrage forces it to be there. Notice it is *higher* than both spot rates (3% and 4%). That is not an accident — it is the market saying, through the arithmetic, "we think the one-year rate a year from now will be about 5%."

#### Worked example: deriving the 1y1y forward step by step

Let us do the whole computation slowly, the way you would on paper, because this is the calculation the entire post hangs on.

You are given two numbers off the curve: the one-year spot rate $s_1 = 3.0\%$ and the two-year spot rate $s_2 = 4.0\%$.

Step 1 — turn each spot rate into a *growth factor*, the number \$1 multiplies by over the period:
- One year at 3.0%: growth factor = $(1.03)^1 = 1.0300$.
- Two years at 4.0%: growth factor = $(1.04)^2 = 1.0816$.

Step 2 — write the no-arbitrage identity. The two-year growth must equal the first-year growth times the second-year (forward) growth:

$$(1 + s_2)^2 = (1 + s_1) \times (1 + f_{1,1})$$

$$1.0816 = 1.0300 \times (1 + f_{1,1})$$

Step 3 — solve for the forward growth factor by dividing:

$$1 + f_{1,1} = \frac{1.0816}{1.0300} = 1.0501$$

Step 4 — read off the rate: $f_{1,1} = 0.0501 = 5.01\%$.

Check it against the original two paths. Path A: \$1,000 × 1.0816 = \$1,081.60. Path B: \$1,000 × 1.03 × 1.0501 = \$1,030 × 1.0501 = \$1,081.60. Identical, to the penny. That is no-arbitrage doing its job.

*The intuition: the two-year spot already bakes in two one-year legs; divide out the first leg you know and what is left is the forward, the market's price for the second leg.*

![A three-by-three grid showing the one-year spot, two-year spot, their growth factors, and how dividing the two-year factor by the one-year factor isolates the 1y1y forward rate of 5.01 percent](/imgs/blogs/forward-rates-what-the-market-expects-rates-to-be-7.png)

The grid above is the same calculation laid out as a table you can reuse for any pair of maturities. The general formula, for the forward rate covering the period from year $n$ to year $n+m$, is just the same division generalised:

$$(1 + f_{n,m})^m = \frac{(1 + s_{n+m})^{n+m}}{(1 + s_n)^{n}}$$

where $s_n$ is the spot rate to year $n$, $s_{n+m}$ is the spot rate to year $n+m$, and $f_{n,m}$ is the forward rate for the $m$-year period starting at year $n$. Don't let the subscripts scare you — every forward you will ever compute is "the long growth factor divided by the short growth factor, then turned back into an annual rate." That is the entire machine.

#### Worked example: a forward further out the curve

Let us extend the curve and compute the 2y1y forward — the one-year rate starting *two* years from now. Suppose the spot curve reads: $s_2 = 4.0\%$ and $s_3 = 4.5\%$.

Growth factors: two years at 4.0% is $(1.04)^2 = 1.0816$; three years at 4.5% is $(1.045)^3 = 1.1412$.

The forward growth for the third year is the three-year factor divided by the two-year factor:

$$1 + f_{2,1} = \frac{1.1412}{1.0816} = 1.0551 \quad\Rightarrow\quad f_{2,1} = 5.51\%$$

So the implied one-year rate two years out is 5.51% — higher again than the 5.01% implied for one year out. The forwards keep climbing because the spot curve keeps rising. Each successive forward is the *marginal* rate the market needs to justify extending the curve one more year.

*The intuition: forwards are the marginal rates that the spot curve is the running average of — every time the average ticks up, the marginal rate that pulled it up is higher still.*

## Forwards relative to spots: the influence the curve carries

Here is where forwards stop being an arithmetic curiosity and start being a *picture of the market's mind*. Put the spot curve and the implied forward curve on the same axes and a structural relationship jumps out.

![A chart with maturity in years on the horizontal axis and yield in percent on the vertical axis, showing the spot curve in blue rising from 3 to 5 percent and the implied forward curve in green sitting above it](/imgs/blogs/forward-rates-what-the-market-expects-rates-to-be-3.png)

Look at the two curves. The blue spot curve rises gently from 3.0% (one year) to 5.0% (five years). The green forward curve sits *above* it the whole way, rising faster and topping out higher. This is not a coincidence of my chosen numbers — it is a mathematical law:

> **When the spot curve slopes upward, the forward curve lies above it. When the spot curve slopes downward, the forward curve lies below it. When the spot curve is flat, the two coincide.**

The reason is exactly the average-versus-marginal relationship from the foundations section. The spot rate to year $n$ is the *average* of all the one-year forwards from now to year $n$. If that average is *rising* as you go further out, then each new forward you add must be *above* the running average — that is the only way an average can increase. It is the same arithmetic as a student's grade-point average: if your GPA is going up, your most recent semester's grade must be above your GPA. The forward is the "most recent semester"; the spot is the "GPA."

This is why the influence runs the way it does: the slope of the spot curve *is* the position of the forward curve. A steep upward spot curve implies forwards far above the current short rate, which is the market shouting "we expect rates to rise a lot." A flat curve implies forwards roughly equal to today's short rate — "we expect rates to hold." An inverted curve, sloping down, implies forwards *below* today's short rate — "we expect rates to fall," which is the bond market's classic recession warning, the subject of the [yield-curve inversion post](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).

#### Worked example: building the whole forward curve from spots

Take a fuller spot curve and crank out the one-year forwards for each year. Spots: $s_1 = 3.00\%$, $s_2 = 4.00\%$, $s_3 = 4.50\%$, $s_4 = 4.80\%$, $s_5 = 5.00\%$.

The one-year forward for each successive year is "this year's growth factor divided by last year's":

| Period | Calculation | Forward rate |
|---|---|---|
| 0y → 1y (= today's 1y spot) | $(1.03)^1 / 1$ | **3.00%** |
| 1y → 2y (1y1y) | $(1.04)^2 / (1.03)^1 = 1.0816 / 1.0300$ | **5.01%** |
| 2y → 3y (2y1y) | $(1.045)^3 / (1.04)^2 = 1.1412 / 1.0816$ | **5.51%** |
| 3y → 4y (3y1y) | $(1.048)^4 / (1.045)^3 = 1.2068 / 1.1412$ | **5.75%** |
| 4y → 5y (4y1y) | $(1.05)^5 / (1.048)^4 = 1.2763 / 1.2068$ | **5.76%** |

Notice two things. First, every forward (except the trivial first-year one, which equals the one-year spot) is *above* the spot curve at the same point — the green curve above the blue. Second, the forwards rise faster early and then flatten, mirroring the gentle rounding-off of the spot curve. The forwards are doing all the work of pulling the average up; once they stop rising much (5.75% to 5.76%), the spot curve flattens too.

*The intuition: the forward curve is the spot curve's "marginal cost" twin — the spot curve is just the running average of all the forwards underneath it.*

## Forwards as a forecast — and the term-premium asterisk

We have established the mechanical fact: the curve implies a path of future short rates. Now the interpretive question. **Is that implied path actually what the market expects?** This is where careful people earn their keep, because the honest answer is "mostly, but not exactly, and the gap matters."

![A chart with years ahead on the horizontal axis and the short rate in percent on the vertical axis, showing the forward path in green rising from 3 toward 5 percent and a lower dashed expected path in blue, with the gap labeled as the term premium](/imgs/blogs/forward-rates-what-the-market-expects-rates-to-be-4.png)

Under the purest theory — the **expectations hypothesis** — the forward rate *equals* the expected future short rate, full stop. The logic: if investors only cared about expected return, they would be indifferent between locking in long and rolling over short, so the rates would line up exactly, and the forward would be a clean, unbiased forecast. Under that view, the green forward path in the chart above *is* the market's forecast.

But investors do not only care about expected return. They care about risk, and tying your money up for two years instead of rolling it one year at a time exposes you to the danger that rates spike and you are stuck earning the old rate while the world earns more. To accept that risk, investors demand a little extra yield. That extra yield is the **term premium** — the compensation for committing to a longer maturity. We unpack it fully in the [term-premium and expectations post](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) and the [heavy-math curve-modeling post](/blog/trading/quantitative-finance/yield-curve-modeling); here the one thing to internalise is its *direction*.

Because the term premium is baked into longer spot rates, it inflates the forwards computed from them. So:

$$\text{forward rate} = \underbrace{\text{expected future short rate}}_{\text{the genuine forecast}} + \underbrace{\text{term premium}}_{\text{a positive bias, usually}}$$

The forward path (green, in the chart) sits *above* the true expected path (blue, dashed) by the term premium. Forwards are a *biased* forecast — biased high, in normal times — because they include payment for risk that has nothing to do with where rates are actually going. A naive reader of the curve who takes forwards as gospel will systematically over-predict rate increases. This is not a small effect: estimates of the term premium on the 10-year Treasury have ranged from around +2% in the early 1980s to *negative* values in the 2010s and 2020s, when heavy central-bank bond buying squashed it below zero.

#### Worked example: stripping the term premium out of a forward

Suppose the 1y1y forward you computed is 5.01%, and a term-premium model (or a survey of economists, or the Fed's own published estimates) suggests the one-year term premium one year out is about 0.40% — 40 basis points. What does the market *actually* expect the one-year rate to be in a year?

$$\text{expected future short rate} = \text{forward} - \text{term premium} = 5.01\% - 0.40\% = 4.61\%$$

So the curve *implies* 5.01%, but once you remove the risk compensation, the market's genuine central expectation for the one-year rate a year out is closer to 4.61%. If you were betting on where rates go — say, positioning a portfolio for a Fed pause — using the raw 5.01% forward would have you bracing for more hikes than the market truly expects. The 40-bp wedge is the difference between reading the curve literally and reading it correctly.

*The intuition: forwards are the market's forecast wearing a risk-premium coat; to see the real forecast you have to take the coat off, and the coat is rarely zero.*

This is the single most important caveat in the post, so let me state it as bluntly as possible: **a forward rate is what the market will charge or pay today for a future period, not a promise about what rates will be.** It is a *breakeven* and a *risk-adjusted forecast*, not a prophecy. People who lost money "fading the forwards" — betting against the curve's implied path — usually did so by forgetting that the curve was never claiming to predict; it was quoting a price that already included a premium.

## Forwards as the breakeven rate

There is a second, equally useful way to read a forward that has nothing to do with forecasting: the forward is a **breakeven**. It is the future rate at which two strategies tie. This framing is what makes forwards actionable for anyone deciding between locking in and rolling over.

![A before-and-after comparison showing that if next year's one-year rate prints below 5.01 percent then locking in wins, and if it prints above 5.01 percent then rolling over wins, with both tying exactly at the forward](/imgs/blogs/forward-rates-what-the-market-expects-rates-to-be-5.png)

Go back to the two paths. You have \$1,000 for two years. Lock in at the two-year spot (4%) and you are guaranteed \$1,081.60. Roll over — one year at 3%, then a year at whatever rate exists then — and your outcome depends on that unknown future rate. The forward (5.01%) is precisely the future rate at which rolling exactly matches locking in. So:

- **If next year's one-year rate turns out *above* 5.01%**, rolling beats locking. You did better not committing.
- **If it turns out *below* 5.01%**, locking beats rolling. You should have committed.
- **If it lands *exactly* at 5.01%**, it is a dead heat — both end at \$1,081.60.

That reframes the whole decision. You are not asking "will rates rise?" — a vague question. You are asking the sharp question: "will next year's one-year rate be above or below 5.01%?" The forward gives you the exact hurdle. This is enormously practical. A treasurer deciding whether to lock in two-year funding or roll one-year debt twice is, whether they know it or not, betting on whether the future short rate will beat the forward.

#### Worked example: the breakeven decision with real dollars

You run the cash for a small business and have \$1,000,000 to invest for two years. Two-year deposits pay 4.0%; one-year deposits pay 3.0%. Lock in for two years and you are certain to have:

$$\$1{,}000{,}000 \times (1.04)^2 = \$1{,}081{,}600$$

Or roll: one year at 3% gets you \$1,030,000, then reinvest for the second year at next year's one-year rate. The forward says the breakeven is 5.01%. Test two scenarios:

- **Rates rise to 6%** next year. Rolling gives \$1,030,000 × 1.06 = **\$1,091,800** — you beat the lock-in by \$10,200.
- **Rates fall to 4%** next year. Rolling gives \$1,030,000 × 1.04 = **\$1,071,200** — you fall short of the lock-in by \$10,400.

The forward, 5.01%, is the exact dividing line between those outcomes. Above it, rolling wins; below it, locking wins. Your real decision is not "rates up or down" but "will next year's one-year rate clear 5.01%?" And remember the term-premium asterisk: because the lock-in includes term premium, the *expected* outcome slightly favours locking in even though the breakeven is 5.01% — you are being paid a little to commit.

*The intuition: a forward turns the fuzzy question "where are rates going?" into the sharp, testable question "will the future rate beat this exact breakeven?"*

## Locking in a forward: turning the implied rate into a real contract

So far the forward has been an *implied* number — a rate the curve contains but that nobody has promised you. But here is something that surprises people: you can actually *lock in* a forward rate today, securing the rate for a future period without waiting for that period to arrive. You do not need a bank or a derivative to do it; you can build the lock out of ordinary bonds. Understanding this construction is what finally makes the forward feel concrete rather than abstract, because it shows the forward is not a forecast at all — it is a *tradeable price*.

The trick is to combine a long position and a short position so that the cash flows in the near term cancel and you are left with exactly one future period's exposure. Suppose you want to lock in, today, the one-year rate for the period one year out — the 1y1y. You do two things at once: you *buy* a two-year zero-coupon bond and you *short* (borrow and sell) a one-year zero-coupon bond, sized so the two cost the same today. The one-year short means you owe \$1,000 in a year; the two-year long means you receive a larger amount in two years. Net it out, and your position is: pay nothing now, owe \$1,000 in one year, receive a fixed amount in two years. That is *exactly* a one-year investment that starts in a year — at a rate you have nailed down today. And the rate you nailed down is, by no-arbitrage, the 1y1y forward. The market lets you trade the forward because the forward is just a recipe of bonds you already can trade.

This is also precisely what a *forward-rate agreement* and the fixed leg of an *interest-rate swap* do under the hood — they package this bond combination into a single contract so you do not have to assemble it yourself. The economics are identical to the home-made version: you are locking the forward.

#### Worked example: locking in the 1y1y forward with two bonds

Let us build the lock with real numbers, using our running curve: one-year spot 3.0%, two-year spot 4.0%, so the 1y1y forward is 5.01%.

Today's bond prices (price of a zero that pays \$1,000 at maturity):
- One-year zero: $\$1{,}000 / 1.03 = \$970.87$.
- Two-year zero: $\$1{,}000 / (1.04)^2 = \$1{,}000 / 1.0816 = \$924.56$.

Now construct the lock so it costs nothing today. Buy one two-year zero for \$924.56. To pay for it, short one-year zeros — sell \$924.56 worth, which is $\$924.56 / \$970.87 = 0.9523$ of a one-year zero, obligating you to repay $0.9523 \times \$1{,}000 = \$952.30$ in one year.

Trace the cash flows:
- **Today:** +\$924.56 (from the short) − \$924.56 (buy the two-year) = **\$0**. Costs nothing.
- **Year 1:** you must repay the one-year short: **−\$952.30**.
- **Year 2:** the two-year zero matures: **+\$1,000**.

So you have manufactured a position that pays out \$1,000 in two years in exchange for putting in \$952.30 in one year. The implied rate over that one future year is:

$$\frac{\$1{,}000}{\$952.30} - 1 = 1.0501 - 1 = 5.01\%$$

Exactly the 1y1y forward — locked in today, with zero money down. You did not predict anything; you *priced* the future year off the bonds that already exist.

*The intuition: you can build a forward out of an ordinary long and short bond, which proves the forward is a tradeable price set by no-arbitrage, not a wishful forecast.*

This construction is also the deepest reason the no-arbitrage argument from the start of the post *has* to hold. If the two-year spot drifted out of line with the one-year spot and the "fair" forward, an arbitrageur would assemble exactly this long-short package, lock in a rate above what they could borrow at, and harvest a riskless profit until prices snapped back. The forward is not pinned down by anyone's opinion about the future — it is pinned down by the threat of this trade. That is why we can extract a "forecast" from prices at all: the forecast is enforced by money, not by consensus.

## Rolling down the curve: a return source hiding in the slope

The same machinery explains one of the most reliable, least flashy sources of return in fixed income: **rolling down the curve**, also called *roll-down* or *carry-and-roll*. It falls straight out of the fact that, on an upward-sloping curve, a bond's yield *drops* as it ages — and falling yields mean rising prices.

![A timeline showing a five-year bond bought at a four percent yield aging into a four-year bond at a lower three point seven percent yield after one year, earning the coupon plus a price gain from rolling down the curve](/imgs/blogs/forward-rates-what-the-market-expects-rates-to-be-6.png)

Here is the mechanism. Buy a five-year bond yielding 4.0%. A year passes. It is now a *four-year* bond. But the yield curve has not moved — it is still upward-sloping, and on that curve the four-year point yields, say, 3.7%, lower than the five-year point's 4.0%. Your bond's yield has *fallen* from 4.0% to 3.7% not because rates dropped, but simply because the bond *slid down the curve* to a shorter, lower-yielding maturity. And since [bond prices move opposite to yields](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds), that fall in yield is a *gain* in price. You earn the coupon *plus* that price gain — a total return above the bond's yield, harvested purely from the slope of the curve.

#### Worked example: the roll-down return

You buy a five-year bond at par, \$1,000, with a 4% coupon, yielding 4.0%. Hold it one year. You collect the coupon:

$$\text{coupon} = 4\% \times \$1{,}000 = \$40$$

Now it is a four-year bond. The curve is unchanged and prices a four-year bond at a 3.7% yield. Because the yield is now 0.3% below the bond's 4% coupon, the bond is worth slightly *more* than par. Roughly, the price gain is the yield drop times the bond's remaining duration (its price sensitivity to yield, about 3.7 years for a four-year bond). So:

$$\text{roll gain} \approx 0.3\% \times 3.7 \times \$1{,}000 \approx \$11$$

Total one-year return:

$$\frac{\$40 \text{ coupon} + \$11 \text{ roll gain}}{\$1{,}000} = \frac{\$51}{\$1{,}000} \approx 5.1\%$$

You earned roughly 5.1% over the year, versus the 4.0% yield you bought at — an extra ~1.1% harvested *just from the bond aging down a sloped curve*, with no change in interest rates at all.

*The intuition: on an upward-sloping curve, time itself pushes a bond's yield down and its price up, so you get paid to do nothing but hold and let the bond grow older.*

But — and this is the connection that ties the post together — roll-down is *not* free money, and the reason is forwards. The forward rate is precisely the future yield at which roll-down earns you *nothing extra*. If next year the four-year point on the curve rises to exactly the forward rate implied today, the price loss from that yield rise will exactly cancel the roll-down gain, and you will earn only the bond's yield. So roll-down is a bet that the curve will *not* rise to the levels the forwards imply. You are, once again, fading the forwards — collecting the term premium if the implied rate increases do not materialise. When forwards are right, roll-down delivers nothing; when forwards overstate rate increases (the usual case, thanks to term premium), roll-down quietly pays.

#### Worked example: when roll-down evaporates

Same five-year bond bought at a 4.0% yield. The 1y forward for the four-year point — the four-year yield the curve implies for one year from now — is, say, 4.3% (forwards above spots on a rising curve). Suppose the curve actually rises so that in a year the four-year point yields exactly that forward, 4.3%, instead of staying at 3.7%.

Now your bond's yield went *up* from 4.0% to 4.3% — a 0.3% rise. The price *loss* is roughly:

$$0.3\% \times 3.7 \times \$1{,}000 \approx -\$11$$

Total return: \$40 coupon − \$11 price loss = \$29, or about 2.9% — *below* the 4.0% yield you bought at, and far below the 5.1% roll-down case. The roll-down gain vanished because the curve moved exactly to the forwards.

*The intuition: roll-down pays you only to the extent the future does not match the forwards; if the curve realises its own implied path, the slope-harvesting strategy earns nothing extra.*

## Forwards in continuous and instantaneous form (a brief deeper layer)

Everything above used annual compounding because it is the most intuitive. Practitioners and the [heavy-math curve-modeling literature](/blog/trading/quantitative-finance/yield-curve-modeling) more often use *continuous compounding*, which makes the algebra cleaner and the forwards additive. You do not need this to use forwards, but it is worth a paragraph so the notation does not surprise you elsewhere.

With continuously compounded spot rates, the no-arbitrage identity stops being a product and becomes a sum. The continuously compounded forward rate $f$ for the period between times $T_1$ and $T_2$ is:

$$f = \frac{s_2 T_2 - s_1 T_1}{T_2 - T_1}$$

where $s_1$ and $s_2$ are the continuously compounded spot rates to $T_1$ and $T_2$. The numerator is "total accumulated rate-times-time out to $T_2$, minus the same out to $T_1$" — the rate-time content of just the forward window — and dividing by the window length annualises it. It is the same average-versus-marginal idea, now expressed as a difference instead of a ratio.

Push the window to an instant ($T_2 \to T_1$) and you get the **instantaneous forward rate** — the forward rate for an infinitesimally short period at a specific future date. The instantaneous forward curve is the true "marginal rate" curve: each point is the rate the market implies for an instant at that future moment. The whole spot curve is just the *average* of the instantaneous forwards out to each maturity, which is the calculus version of the GPA analogy. Short-rate and curve models in [quantitative finance](/blog/trading/quantitative-finance/yield-curve-modeling) are often written directly in terms of instantaneous forwards because they are the cleanest primitive.

#### Worked example: the continuous-compounding forward

Suppose the continuously compounded one-year spot is 2.96% and the two-year spot is 3.92% (these are the continuous-compounding equivalents of the 3% and 4% annual rates we used — continuous rates are always a touch lower for the same growth). The continuous 1y1y forward is:

$$f = \frac{0.0392 \times 2 - 0.0296 \times 1}{2 - 1} = \frac{0.0784 - 0.0296}{1} = 0.0488 = 4.88\%$$

Converting that continuous 4.88% back to an annually compounded rate gives $e^{0.0488} - 1 = 5.00\%$ — essentially the same 5.01% we got with annual compounding, off only by rounding. The two conventions agree on the economics; they just dress it differently.

*The intuition: continuous compounding turns the forward from a ratio into a clean difference of rate-times-time, which is why curve modelers prefer it, but the answer is the same forward.*

## Forward credit spreads: the same idea applied to Northwind Corp

Everything so far used the risk-free Treasury curve, where no-arbitrage is exact because there is no chance of not being repaid. But the forward machine works on *any* yield curve — including a corporate one — with one honest refinement: a corporate forward embeds not just a forward risk-free rate but a forward *credit spread*, the future compensation the market demands for the chance the company defaults.

Recall Northwind Corp, our fictional investment-grade issuer. Suppose its bonds yield a flat 0.80% — 80 basis points — over Treasuries at every maturity today. Then Northwind's one-year yield is 3.80% (3.00% Treasury + 0.80% spread) and its two-year yield is 4.80% (4.00% + 0.80%). Run the forward calculation on Northwind's *own* curve and you get the company's implied 1y1y forward yield. But there is a subtlety the Treasury case hid: the forward *spread* — the credit spread the market implies for that future year — may not equal today's spread. If the curve of credit spreads is itself upward-sloping (longer Northwind bonds carry wider spreads, which is normal because more can go wrong over more time), then the *forward* spread is wider than the spot spread, exactly as forward rates exceed spot rates on a rising curve. The same average-versus-marginal logic applies twice: once to the risk-free rate, once to the credit spread.

#### Worked example: Northwind's forward yield and forward spread

Today: Northwind one-year yield 3.80%, two-year yield 4.85% (note the two-year spread is slightly wider, 0.85% vs 0.80% — an upward-sloping spread curve).

Northwind's 1y1y forward *yield* uses the same division as before:

$$1 + f^{\text{Northwind}} = \frac{(1.0485)^2}{1.0380} = \frac{1.0994}{1.0380} = 1.0592 \quad\Rightarrow\quad f^{\text{Northwind}} = 5.92\%$$

The Treasury 1y1y forward was 5.01%. So Northwind's *forward credit spread* — the spread the market implies for the second year — is:

$$5.92\% - 5.01\% = 0.91\%$$

That is wider than today's 0.80% spot spread. The market is implying that one year from now, the one-year credit spread on Northwind will be about 91 bps, not 80 — it expects Northwind's borrowing premium to widen a little, perhaps because the firm's debt matures or the cycle ages. A credit investor reads this exactly like a rate trader reads the rate forwards: it is the breakeven spread, the level at which buying Northwind one-year paper now versus a year from now comes out even.

*The intuition: forwards are not just a Treasury tool — every yield curve, including a risky corporate one, implies forward rates and forward spreads by the very same division.*

This is the bridge to the credit track of the series, where the [chance you don't get paid back](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping) becomes the central object. For now, the lesson is that the forward machine is universal: point it at any curve and it un-blends that curve's averages into the marginal rates hiding inside.

## Common misconceptions

**"The forward rate is the market's prediction of future rates."** This is the big one, and it is *almost* right, which is what makes it dangerous. The forward is the expected future short rate *plus a term premium* — compensation for the risk of locking money up. In normal times the term premium is positive, so forwards sit *above* the genuine expected path and systematically over-predict rate increases. Take the curve literally and you will brace for more hikes than the market actually expects. The forward is a *risk-adjusted breakeven*, not a clean forecast.

**"If forwards predict rates will rise, I should avoid long bonds."** No — because the rise is *already in the price*. The whole point of no-arbitrage is that the forwards are baked into today's yields. If you sell long bonds because forwards imply higher rates, you are betting rates will rise *even more* than the forwards imply. If rates merely do what the forwards already say, every maturity earns the same expected return and you gain nothing by avoiding the long end. You only profit by having a view *different from* the forwards.

**"A higher forward rate means I'll earn more."** A forward is not a return you can simply pocket. You can lock in a forward (via forward-rate agreements or by combining bonds), but doing so just gives you exactly that rate over that future window — no edge. The forward is a price, not a gift. The only way it becomes "more" is if the realised rate comes in *below* the forward (so your locked-in rate beats the market) — which is, again, a bet against the curve.

**"Rolling down the curve is free money."** It feels free — you earn more than the yield just by holding — but it is compensation for a real risk: that the curve rises to the forwards and the price loss eats the roll-down gain. Roll-down pays precisely *because* it does not always pay; it is the term premium showing up as realised return when forwards over-predict, and it stings when forwards under-predict.

**"The forward curve and the yield curve are the same thing."** They are tightly linked but different. The yield (spot) curve plots the average rate to each maturity; the forward curve plots the marginal rate for each future period. On an upward-sloping curve the forward curve sits *above* the spot curve; on a downward-sloping (inverted) curve it sits *below*. Reading one off as if it were the other will get the magnitude — and sometimes the direction of your conclusion — wrong.

**"Forwards only matter for bond traders."** Forwards quietly set prices across the economy. The rate on a forward-starting loan, the fixed leg of an interest-rate swap, the breakeven on a mortgage-refinance decision, the cost a company locks in for future borrowing — all are forward rates. Anyone choosing between a fixed and floating rate, or between locking in funding now versus later, is implicitly trading against the forwards whether they know the word or not.

## How it shows up in real markets

**Pricing forward-starting loans and rate locks.** When a corporate treasurer asks a bank to commit *today* to a loan rate for borrowing that starts in six months, the bank does not guess — it reads the rate straight off the forward curve and adds a margin. A *forward-rate agreement* (FRA) is a contract whose entire payoff is the difference between a future realised short rate and a forward rate agreed today; trillions of dollars of these change hands. The same machine prices the fixed leg of an interest-rate swap, which is just a strip of forwards bundled together. Every one of these instruments is the no-arbitrage forward of this post, wearing a contract.

**The 2004–2006 "conundrum."** As the Federal Reserve raised its policy rate from 1% to 5.25% over 2004–2006, the forward curve implied that long-term rates should rise too. They barely budged — the 10-year Treasury yield stayed near 4.5% even as the Fed hiked. Then-Chairman Alan Greenspan called it a "conundrum." In forward language, the long-end forwards were *not* moving up the way the short-end hikes implied, which meant the term premium was collapsing — foreign central banks and reserve managers were buying Treasuries so heavily that the compensation for holding long duration evaporated. Forwards revealed, in real time, that something was pinning the long end down. (See the [central-bank toolkit post](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) for how policy and demand interact.)

**Quantitative easing and the squashed term premium.** From 2008 through the late 2010s, the Fed and other central banks bought trillions in long bonds, deliberately to push down long-term yields. A direct effect was to crush the term premium — at times to *negative* values, meaning investors were paying for the privilege of holding long bonds. With a negative term premium, forwards sat *below* the true expected rate path, flipping the usual bias. Anyone mechanically reading forwards as forecasts during this era under-predicted, not over-predicted, future short rates. It is the cleanest real-world proof that the term premium is not a textbook footnote but a moving, sign-flipping number you must account for.

**The 2022–2023 inversion and the recession debate.** When the Fed hiked aggressively to fight inflation in 2022–2023, the yield curve inverted — short rates above long rates. In forward terms, the forwards for periods two and three years out fell *below* the current short rate: the market was implying that the Fed would have to *cut* rates, the classic recession signal explored in the [yield-curve inversion post](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession). Traders watched the forward path obsessively to read the market's implied timing of the first cut — a textbook case of forwards being used exactly as this post describes: the curve's embedded forecast of the central bank's next move, read off by anyone willing to do the arithmetic.

**Mortgage and refinancing decisions.** A household choosing between a 5-year fixed mortgage and a 1-year rate they will refinance four times is making the lock-versus-roll decision from this post, at kitchen-table scale. The 5-year fixed rate embeds the forwards for years two through five plus a term premium; rolling the 1-year exposes them to whatever the short rate does. The forward rate is the breakeven that decides which is cheaper — and the term premium is why the fixed option usually looks "expensive" up front yet often wins, because lenders charge for the rate risk the borrower offloads onto them.

**Carry-and-roll strategies at the big bond funds.** Roll-down is a workhorse of how large fixed-income managers — the kind profiled in the [PIMCO and the bond market piece](/blog/trading/finance/pimco-and-the-bond-market) — generate return in calm markets. On a steep, stable curve they hold intermediate bonds and harvest the roll-down as bonds age into lower yields. The strategy is, in disguise, a systematic bet that realised rates will come in below the forwards — that the term premium will be earned rather than given back. It works for years and then, in a rate spike like 2022, gives much of it back at once, exactly as the "when roll-down evaporates" example warned.

**Reading the "implied number of cuts" in market commentary.** When you see a headline like "the market is now pricing in three rate cuts by year-end," that number is not a survey — it is read straight off the forward curve. Analysts take the forwards for the policy rate over the coming meetings, subtract today's rate, and divide by the typical 0.25% move to get an implied count of hikes or cuts. Through 2024 these implied counts swung wildly meeting to meeting — markets priced in six cuts at the start of the year, then repriced toward one or two as inflation proved sticky. Every one of those headline swings was the forward curve being recomputed as new data arrived. If you understand forwards, you can do this arithmetic yourself instead of taking the headline's word for it — and you will notice that the raw count is slightly inflated by term premium, so the market's *genuine* expectation is usually a touch more dovish than "the curve is pricing in N cuts" suggests.

**Pension and insurance liability matching.** A pension fund knows it must pay benefits decades into the future, and an insurer knows it owes claims on a long schedule. To make sure they will have the money, they discount those future obligations using the forward curve and buy bonds whose forwards line up with when the cash is needed. The forward rate for "the year that starts 20 years from now" is exactly what tells them how much a dollar owed then is worth today, and what rate they can lock in to fund it. This is the most consequential, least glamorous use of forwards on earth — it is how trillions in retirement and insurance promises are kept solvent. When forwards move, the present value of those liabilities moves with them, which is why a long-end rate shift can blow a hole in (or repair) a pension's funding status overnight, as the UK's 2022 gilt crisis showed when liability-driven investment funds were caught by a sudden surge in long-dated forwards.

## When this matters to you, and where to go next

You now own a genuinely powerful lens: the yield curve is not a static list of rates, it is an *embedded forecast*, and forward rates are how you extract it. The next time you see a steep curve, you can read it as "the market expects rates to rise"; an inverted one as "the market expects cuts"; and in both cases you can do the division yourself to put a number on the implied future short rate — then mentally subtract a term premium to recover the genuine expectation underneath. That single skill changes how you read every rate headline.

It also reframes any lock-versus-roll decision you will ever face — a CD ladder, a mortgage, a business's funding plan — as a clean bet against a known breakeven, rather than a vague guess about "where rates are going." And it explains a quiet return source, roll-down, along with exactly the risk that makes it not free.

To go deeper: the [term-premium and expectations-hypothesis material](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) unpacks the asterisk we kept flagging; the [yield-curve-modeling post](/blog/trading/quantitative-finance/yield-curve-modeling) and the [bond-pricing fundamentals](/blog/trading/quantitative-finance/bond-pricing) give the continuous-time math; the [central-bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) shows who actually moves the short rates that forwards try to predict; and the [government-bonds-as-anchor allocation view](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration) connects all of this to how a portfolio uses duration. Within this series, this post closes the pricing track that began with [discounting cash flows](/blog/trading/fixed-income/discounting-cash-flows-how-a-bond-is-priced) and [spot rates and bootstrapping](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping) — and it opens the door to the yield-curve track, where the curve stops being a pricing tool and becomes the most-watched chart in finance.

*This article is educational, not investment advice. Forward rates are forecasts with a known bias, not promises; anyone acting on the curve's implied path is taking real risk, and the term premium means the curve is rarely telling you exactly what it seems to say.*
