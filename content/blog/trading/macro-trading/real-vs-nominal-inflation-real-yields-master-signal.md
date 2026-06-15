---
title: "Real vs Nominal: Inflation, Real Yields, and the Number That Moves Everything"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Markets price in real terms. The 10-year real yield — nominal yield minus expected inflation — is the master dial behind gold, growth stocks, and crypto. Learn to read it."
tags: ["macro", "monetary-policy", "real-yields", "inflation", "tips", "breakeven-inflation", "discount-rate", "fisher-equation", "bonds", "valuation"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Markets ultimately price assets in *real* (inflation-adjusted) terms, and the 10-year real yield — the nominal yield minus the inflation the market expects — is the single most important macro number for asset valuation.
>
> - **Real yield = nominal yield − breakeven inflation.** This is the Fisher decomposition: a nominal yield is split into the true reward for waiting (the real yield) plus compensation for losing purchasing power (expected inflation). The real yield is what is left after inflation.
> - **The real yield is the discount rate behind every asset.** Raise it, and the present value of future cash flows shrinks — which is exactly why growth stocks, gold, crypto, and long bonds all fall when real yields rise. Lower it (especially below zero), and the same assets rip higher.
> - **2021 → 2023 is the case study.** The 10-year real yield went from about **−1.04% (Dec 2021)** to **+2.48% (Oct 2023)**. That single swing repriced everything long-duration. The "everything rally" and the brutal drawdown that followed were the *same* story told in two directions.
> - **The one number to remember:** watch the **10-year real yield (FRED ticker DFII10)**. It is the master dial. Watch it and you watch the cost of capital for the entire market.

In the autumn of 2022, almost everything that had made money for a decade stopped working at once. Long-dated technology stocks that had compounded relentlessly fell 50, 60, 70 percent from their highs. Bitcoin had already cratered. Gold, supposedly the inflation hedge, was *down* on the year even as inflation printed at a 40-year high. Thirty-year Treasury bonds — the "safest" asset on Earth — were having one of the worst years in their recorded history. To a lot of people it looked like chaos, a thousand unrelated things breaking.

It was not chaos. It was one number moving.

That number is the **real yield** — the interest rate you earn *after* subtracting expected inflation. For most of 2020 and 2021 the U.S. 10-year real yield was deeply negative: lending money to the government for a decade guaranteed you a loss of purchasing power. Negative real yields are the financial equivalent of free money, and when money is free, every asset whose value lives far out in the future gets bid to the moon. Then, in 2022, the real yield swung violently positive — from roughly −1% to nearly +2.5% in under two years. The true cost of capital came roaring back. And every long-duration asset, all at once, had to be repriced for a world where waiting actually costs something again.

This post is about that number. By the end you will be able to look at one chart — the 10-year real yield — and understand more about why markets are moving than most people get from a week of financial news. We will build it from absolute zero: what "real" versus "nominal" even means, how inflation is measured, what TIPS and breakevens are, the Fisher equation that ties it all together, and then the payoff — how to actually *read and trade* the master dial.

![Nominal yield bar split into real yield and expected inflation with the identity beside it](/imgs/blogs/real-vs-nominal-inflation-real-yields-master-signal-1.png)

## Foundations: real vs nominal, CPI, TIPS, and breakevens

Before we can talk about real yields, we have to be ruthless about one distinction that almost everyone blurs: the difference between a **nominal** number and a **real** number.

### Nominal vs real: the only distinction that matters here

A **nominal** quantity is measured in raw dollars, at face value, with no adjustment. If your salary is \$100,000, that is a nominal figure. If your bond pays 4%, that 4% is a nominal yield. If your house "went up 50%" over a decade, that is a nominal gain. Nominal means "as named" — the number on the price tag, the screen, the paycheck.

A **real** quantity is the same thing adjusted for inflation — that is, measured in *purchasing power* rather than in dollars. The question a real number answers is: *how much actual stuff can I buy with this?* Your salary might be \$100,000, but if prices have doubled since you signed the offer, your *real* salary is half what it was. Your bond might pay 4%, but if prices are rising 4% a year, your *real* return is zero — you end the year with more dollars that each buy proportionally less, and you are exactly where you started.

Here is the intuition that makes "real" click. Imagine you put \$100 in a jar today. A year from now you still have \$100 — that is nominal, and it never changes. But what can that \$100 *buy*? If a basket of groceries that cost \$100 today costs \$109 a year from now, your jar of cash has quietly lost about 8% of its power to buy groceries even though the dollar figure never moved. The nominal value sat still; the real value fell. **Inflation is a tax on anyone holding nominal dollars, collected silently, without a vote.**

This is the whole game. Humans *think* in nominal terms — we feel richer when our account balance is bigger — but the economy *runs* on real terms. You cannot eat dollars. You eat what dollars buy. Markets, in the long run, are pricing the second thing, not the first. That is why the real yield ends up being the variable that matters.

### What inflation is, and how it is measured (CPI vs PCE)

**Inflation** is simply the rate at which the general price level rises — the speed at which each dollar loses purchasing power. If inflation is 3% per year, then on average the things you buy cost 3% more this year than last, and your nominal dollar buys about 3% less.

But "the price level" is an abstraction. To turn it into a number you actually have to pick a *basket* of goods and services — rent, groceries, gasoline, healthcare, a haircut, a streaming subscription — track its total cost over time, and report the percentage change. That is what a price index does. The two indices a trader needs to know are:

- **CPI (Consumer Price Index).** Published monthly by the Bureau of Labor Statistics. It tracks a fixed-ish basket meant to represent what a typical urban consumer buys. CPI is the headline number — when the news says "inflation came in at 3.2%," they almost always mean CPI year-over-year. It is the most-watched print, and markets trade violently on the monthly CPI release.
- **PCE (Personal Consumption Expenditures price index).** Published by the Bureau of Economic Analysis. It covers a broader, shifting basket and accounts for the fact that people substitute (when beef gets expensive, you buy more chicken). **PCE is the Federal Reserve's preferred gauge** — when the Fed says it targets "2% inflation," it means 2% on core PCE. PCE typically runs a few tenths *below* CPI because of that substitution effect.

The practical takeaway: **CPI is what the market reacts to in the moment; PCE is what the Fed actually steers toward.** Both tell the same broad story. We will use CPI in the charts because it is the headline series and the one tied directly to the inflation-protected bonds we are about to meet.

A crucial nuance: there is a difference between the inflation that *already happened* (realized CPI, a backward-looking fact) and the inflation the market *expects to happen* (a forward-looking guess). The real yield depends on the second one. You do not get to subtract last year's inflation from today's bond yield; you have to subtract the inflation expected over the *life* of the bond. How do we get a market estimate of that? That is where TIPS come in.

![Line chart of US CPI year over year from 2020 to 2026 peaking in 2022](/imgs/blogs/real-vs-nominal-inflation-real-yields-master-signal-3.png)

The chart above is the inflation that drove the whole story. CPI ran near or below the Fed's 2% target through 2020, then accelerated through 2021, and peaked at **9.06% in June 2022** — a 40-year high. That spike is what forced the Fed to slam rates higher, and it is what dragged real yields out of negative territory. Notice, too, the 2025–26 re-acceleration on the right side of the chart: inflation is not a one-time event that gets "solved." It is a regime that can return, which is precisely why traders watch *expected* inflation as a live variable, not a settled fact.

There is one more measurement subtlety worth carrying, because it explains a recurring market argument. Both CPI and PCE are *averages over a basket*, and the basket has stubborn, slow-moving components (housing and services) and fast, jumpy ones (energy and food). When a headline inflation print spikes, traders immediately split it into **headline** (everything) and **core** (stripping out food and energy, which are volatile and not driven by monetary policy). Core is the cleaner read on the underlying trend, and it is what the Fed leans on. The reason this matters for real yields: a headline spike driven by an oil shock can be transitory and need not lift real yields much, whereas a *core* spike — sticky services inflation — is what forces a sustained policy response and a real-yield repricing. When you decompose a CPI surprise, you are really asking "will this move force the discount rate higher?" — and the answer lives in core, not in the gasoline line item.

### Why "real" is what actually matters — for purchasing power and for valuation

It is worth slowing down on *why* the real number, not the nominal one, is the thing markets ultimately price. There are two separate reasons, and a trader needs both.

The first is **purchasing power**, which we have already met. You consume goods and services, not dollars. If your wealth doubles in dollars but prices triple, you are poorer in every way that touches your life — fewer groceries, less rent covered, a smaller retirement. Every saver, every pension fund, every endowment has a real liability: future consumption, denominated in stuff. So the return that matters to them is the *real* return, the one measured in stuff. A nominal return that merely keeps pace with inflation has, for these purposes, earned nothing. This is not an accounting nicety; it is the actual economic fact that the people allocating trillions of dollars are trying to manage.

The second reason is **valuation**, and it is the one that turns "real" from a personal-finance footnote into the master signal of markets. When an investor decides what an asset is worth, they compare its expected *real* return to the *real* return available risk-free. If a stock is expected to compound at 6% real, and TIPS offer 2% real risk-free, the stock offers a 4% real risk premium — and that premium is what justifies its price. Now raise the risk-free real rate to 4%: the stock's premium collapses to 2%, and to restore an adequate premium its price has to *fall* until its expected real return rises again. **The entire market is, at all times, pricing the real risk premium over the real risk-free rate.** That is why the real yield is not just one variable — it is the denominator against which everything else is measured. Move it and you move the reference point for every valuation in the system simultaneously.

This also dissolves a common confusion. People ask "is the market expensive?" as if it were an absolute question. It is not. The market's "expensiveness" — its earnings yield, its multiples — only means something *relative to the real yield*. A 20x multiple is cheap when real yields are −1% and expensive when they are +3%, because the alternative (risk-free real return) is what the multiple competes against. Valuation is always relative, and the thing it is relative *to* is the real yield. Hold that and the rest of this post is downhill.

### TIPS: bonds that adjust for inflation

A normal Treasury bond is a nominal instrument. You lend the government \$1,000, it pays you a fixed coupon and returns your \$1,000 at maturity — in *nominal* dollars. If inflation eats half your purchasing power along the way, that is your problem; the government pays back exactly what it promised on the face.

**TIPS — Treasury Inflation-Protected Securities** — fix that. A TIPS bond's principal is adjusted upward in line with CPI. If inflation is 5% over a year, the bond's principal grows by roughly 5%, and the coupon (a fixed *percentage*) is paid on that grown principal. The result: a TIPS holder is made whole for inflation automatically. **The yield quoted on a TIPS bond is therefore a *real* yield** — it is the return you earn *over and above* inflation, by construction.

This is the magic that makes real yields observable. You do not have to guess at the real yield; the market quotes it directly. The 10-year TIPS yield (FRED ticker **DFII10**) is the market's real yield on a 10-year horizon, in real time. When that number is −1%, the market is literally telling you it will accept losing 1% of purchasing power per year for the safety of holding government debt. When it is +2%, the market is demanding 2% of real return on top of whatever inflation does.

### Breakeven inflation: the market's inflation forecast, for free

Now put the two bonds side by side. You can buy a **nominal** 10-year Treasury yielding, say, 4.05%. Or you can buy a **real** (TIPS) 10-year yielding 1.74%. The difference between them is the inflation rate at which you would be *indifferent* between the two — the inflation rate that makes them break even. That is why it is called **breakeven inflation**:

```
breakeven inflation = nominal yield − real (TIPS) yield
```

If realized inflation over the next decade comes in *higher* than the breakeven, the TIPS wins (you got paid for inflation the nominal bond didn't compensate). If it comes in *lower*, the nominal bond wins. So the breakeven is the market's collective, money-on-the-line forecast of average inflation over that horizon. It is one of the cleanest inflation expectations measures that exists, and you get it for nothing by subtracting two yields.

Hold these three quantities in your head — nominal yield, real yield, breakeven inflation — because they are bound together by a single identity, and that identity is the heart of this entire post.

![Two line series for nominal and real 10-year Treasury yields with the gap labeled breakeven](/imgs/blogs/real-vs-nominal-inflation-real-yields-master-signal-2.png)

The chart above plots both yields on one axis from 2020 to 2026. The blue line is the nominal 10-year (what the bond pays in dollars). The green line is the real 10-year (the TIPS yield, what you earn after inflation). **The vertical gap between them is breakeven inflation** — the market's inflation forecast. Look at 2021: the real yield is around −1%, far below zero, while the nominal yield sits near 1.5%. The market was accepting a guaranteed real loss to own bonds. Then watch the green line climb relentlessly through 2022 and 2023, crossing zero in the spring of 2022 and reaching nearly +2.5% by late 2023. That climb in the green line — the real yield going from deeply negative to sharply positive — is the master signal we are here to learn.

## The Fisher decomposition: taking a nominal yield apart

The relationship between nominal yields, real yields, and inflation has a name. It comes from the economist Irving Fisher, and it is called the **Fisher equation**. In its simple, everyday-arithmetic form it says:

```
nominal yield ≈ real yield + expected inflation
```

Rearrange it and you get the identity that matters most for trading:

```
real yield ≈ nominal yield − expected inflation
```

This is the same thing as the breakeven relationship from the last section — *expected inflation* and *breakeven inflation* are the same forward-looking quantity. The Fisher equation is just the statement that a nominal interest rate has two jobs bundled into it. Part of the rate compensates you for the inflation you expect to lose. The rest — what is left over — is the *real* reward for lending your money out and waiting. Strip off the inflation compensation, and the real yield is the remainder.

Think of it like a paycheck with taxes withheld. The gross figure (nominal yield) looks impressive, but a chunk of it is automatically claimed by inflation before it ever benefits you. The take-home pay (real yield) is what you can actually spend on more purchasing power. A 6% nominal yield in a 5% inflation world is a 1% real yield — barely above break-even — while a 3% nominal yield in a 0% inflation world is a *3%* real yield, three times richer in what actually matters. **The nominal number can lie to you. The real number cannot.** This is why a trader who looks only at nominal yields is reading the gross figure and ignoring the withholding.

There is also an exact version of the Fisher equation, because percentages compound rather than simply add:

```
(1 + nominal) = (1 + real) × (1 + inflation)
```

Solving for the real return:

```
real return = (1 + nominal) / (1 + inflation) − 1
```

For small numbers the simple subtraction is close enough, and traders use it all day — but for big inflation numbers the gap matters, and we will work the exact version below so you can see it.

There is a subtle third term that desks know about and beginners can safely tuck away for later: the **inflation risk premium**. Breakeven inflation is not a pure forecast; it also embeds a little extra compensation that nominal-bond holders demand for *bearing* inflation uncertainty. So the true decomposition is closer to `nominal yield = real yield + expected inflation + inflation risk premium`. In practice the premium is small and slow-moving (often a few tenths of a percent), so reading breakeven as "the market's inflation forecast" is a fine working approximation. But it is why economists distinguish "breakeven inflation" (what you read off the bond math) from "expected inflation" (the underlying forecast) — the gap between them *is* that premium. For trading purposes, the level and direction of the breakeven are what you act on; the premium matters only when you are trying to be surgically precise about the underlying forecast.

It also helps to name the two halves of the nominal yield by what they *reward*. The **real yield** is the *price of time* — the pure compensation for deferring consumption, independent of inflation. The **breakeven** is the *price of inflation insurance* — what the market charges to bear the erosion of purchasing power. A nominal bond bundles both into one number, which is exactly why it is such a treacherous thing to trade naively: a single move in the nominal yield could be the price of time changing, or the price of inflation insurance changing, and those two have nearly opposite implications for your gold, your growth stocks, and your dollar exposure. Unbundling them is the entire skill.

#### Worked example: your real return on a 5% bond

You buy a bond paying **5%** nominal. Over the year inflation runs at **4%**. What did you actually earn in purchasing power?

The quick Fisher answer is subtraction: `5% − 4% = ~1%` real return. Good enough for a back-of-envelope read, and most desks would just say "you made about a point in real terms."

The exact answer uses the compounding form. Start with \$100. After the 5% coupon you have \$105 nominal. But each dollar now buys 4% less, so deflate by inflation:

```
real value = $105 / (1 + 0.04) = $105 / 1.04 = $100.96
```

Your real return is **0.96%**, not a clean 1%. The exact figure is slightly *below* the simple subtraction because inflation compounds against your full \$105, not just your original \$100. The arithmetic shortcut `nominal − inflation` slightly overstates your real return, and the error grows as inflation gets larger.

**Intuition:** you nominally earned \$5, but inflation quietly clawed back about \$4 of purchasing power, leaving you with under a dollar of *actual* gain — which is the only gain that buys you anything.

## Why the real yield is the discount rate behind every asset price

Here is where the whole thing becomes a master signal rather than a piece of bond-market trivia. To see it, we need one more foundational idea: how anything with future cash flows is priced.

### Present value, in one paragraph

A dollar promised to you ten years from now is worth *less* than a dollar in your hand today, because the dollar in your hand can be invested and grow. To find what a future dollar is worth *today* — its **present value** — you "discount" it by an interest rate:

```
present value = future cash flow / (1 + r)^t
```

where `r` is the discount rate and `t` is the number of years out. The bigger the `r`, the more you divide by, and the smaller the present value. The farther out the cash flow (bigger `t`), the harder that division bites. **This single formula prices everything** — a bond, a stock, a startup, a rental property, a gold bar (with its zero cash flow), a token. An asset is just a claim on future cash flows (or future resale value), and its price is the sum of those flows discounted back to today.

### And the discount rate *is* the real yield

So what number do you plug in for `r`? The honest answer for the *real* value of an asset is: the **real yield**, plus a risk premium for how uncertain the cash flows are. The risk-free real yield is the floor — the minimum real return any investor can earn with zero risk by buying TIPS. Every risky asset has to clear that bar plus extra. So when the real yield rises, the discount rate on *every* asset rises with it, and present values fall across the board. When the real yield falls (especially below zero), the discount rate collapses, and present values inflate everywhere at once.

This is why the real yield is the **master dial**. It is not one input among many; it is the *base rate of the entire valuation system*. Turn it up and you compress the present value of every future dollar in the economy. Turn it down and you expand them. The Fed, by setting policy rates and steering inflation expectations, is ultimately turning this dial — and the market reads the dial directly off the TIPS curve.

![Four-step flow from rising real yield to falling present value to long-duration assets repricing down](/imgs/blogs/real-vs-nominal-inflation-real-yields-master-signal-4.png)

The figure traces the channel: a rising real yield raises the discount rate, which shrinks the present value of distant cash flows, which forces long-duration assets to reprice down. The key word is **duration** — how far out an asset's value lives. A 1-year cash flow barely cares about the discount rate; a 30-year cash flow cares enormously, because that big `t` in the exponent magnifies every change in `r`. Growth stocks (most of their value is in profits a decade-plus away), gold (no cash flow at all, pure store-of-value, infinite duration in a sense), crypto (the same, but more so), and long bonds are all *long-duration* assets. That is precisely why they all move together against the real yield.

Let us make "duration" concrete, because it is the single most useful concept for translating a real-yield move into a price move. Duration is, roughly, the *percentage change in an asset's price for a one-percentage-point change in the discount rate*. A 2-year bond has a duration near 2, so a 1% rise in yields knocks about 2% off its price. A 30-year bond has a duration near 20, so the same 1% rise knocks about 20% off — ten times the damage from the identical rate move. Equities have an *implied* duration too: a profitable, cash-rich value stock might behave like a 5–10 year instrument, while a hyper-growth name whose profits are all projected for the 2030s and beyond behaves like a 30-, 40-, even 50-year instrument. Crypto and zero-cash-flow gold are at the far end — their "duration" is enormous because *none* of their value is in the near term.

This is why the real-yield move of 2022 did not hit every asset equally. It was not that tech is "riskier" in some vague sense; it is that tech's cash flows sit farther out in time, so its duration is longer, so the same rise in the real discount rate compressed its present value far more. The math is mechanical and merciless: longer duration means bigger sensitivity to `r`, and `r` is the real yield. When you hear a strategist say "long-duration assets got hit," they are describing this exponent — the `t` in `(1 + r)^t` — doing its work. A trader's job is to know, for each thing in the book, roughly *how long* its duration is, because that number tells you how hard a given real-yield move will hit it.

### The real-yield curve, not just one point

So far we have spoken of "the" real yield as if it were a single number. In fact there is a whole **term structure** of real yields — a 2-year real yield, a 5-year, a 10-year, a 30-year — and they do not always move together. The 10-year (DFII10) is the workhorse because it is the standard reference duration for equity and gold valuation, but the *shape* of the real curve carries information too. When short real yields rise faster than long real yields (the real curve flattening or inverting), the market is pricing aggressive near-term tightening with the expectation that it gets reversed later — a recession-watch signal. When long real yields lead the move higher, the market is repricing the *whole* future cost of capital, which is the more punishing scenario for long-duration assets because it lifts the discount rate at every horizon.

For most trading purposes the 10-year real yield is the right single number to track — it sits at the duration that matters for the assets most people hold, and it is the cleanest, most liquid point on the TIPS curve. But know that it is one point on a curve, and that the curve's *shape* is a second-order signal about whether the market thinks the real-rate regime is durable or about to turn.

#### Worked example: backing out breakeven inflation from the data

Let us read breakeven inflation straight off the two yields, at a matched date. In **October 2022**, the nominal 10-year Treasury yielded about **4.05%** and the 10-year TIPS (real) yield was about **1.74%**. The breakeven is just the difference:

```
breakeven inflation = nominal − real
                    = 4.05% − 1.74%
                    = 2.31%
```

So in October 2022 — *with realized CPI still running above 8%* — the market's 10-year forward inflation expectation was only about **2.3%**. Read that again: even at the height of the worst inflation in 40 years, the bond market was betting that inflation would average barely above the Fed's 2% target over the coming decade. The market believed the Fed would win.

**Intuition:** breakeven inflation is the market putting money on a forecast. The fact that it stayed near 2.3% during a 9% CPI print is itself a tradeable signal — it told you the spike was seen as temporary, which is why long-term real yields, not breakevens, did the heavy lifting in repricing assets.

#### Worked example: a long-duration DCF when real yields go from −1% to +2%

Now the punchline calculation. Take a single cash flow — **\$100 to be received in 10 years** — and price it under two real-yield regimes drawn straight from the data: the **−1%** real yield of 2021 and the **+2%** real yield of 2024.

At a **−1%** real discount rate (2021):

```
PV = $100 / (1 + (−0.01))^10
   = $100 / (0.99)^10
   = $100 / 0.9044
   = $110.57
```

At a **+2%** real discount rate (2024):

```
PV = $100 / (1 + 0.02)^10
   = $100 / (1.02)^10
   = $100 / 1.2190
   = $82.03
```

The present value fell from **\$110.57** to **\$82.03** — a drop of about **26%** — purely because the real discount rate rose by three percentage points. And that is for a cash flow only 10 years out. Push the cash flow to 30 years and the same rate move slices the present value by roughly *60%*.

**Intuition:** nothing about the \$100 changed. The business did not get worse; the gold bar is the same bar. The *only* thing that moved was the real yield — the master dial — and it knocked a quarter to two-thirds off the value of long-dated claims. That single mechanism is why growth stocks and crypto fell so hard in 2022 even though their underlying stories were unchanged.

![Bar chart of present value of $100 due in 10 years falling as the real discount rate rises](/imgs/blogs/real-vs-nominal-inflation-real-yields-master-signal-6.png)

The chart makes the sensitivity visceral. The present value of that \$100 sits at **\$111** when the real discount rate is −1% and collapses to **\$61** by the time real rates reach +5%. Each bar is just `$100 / (1 + r)^10` evaluated at a different real rate — pure math, no forecast. The leftmost (negative-rate) bar is *above* \$100: when real yields are negative, a future dollar is worth *more* than a present one, which is the financial absurdity that powered the 2021 mania. The middle and right bars show the disciplined world we live in now, where waiting genuinely costs something and valuations have to respect it.

## Gold and growth stocks as real-yield trades

Two assets make the real-yield mechanism crystal clear, because their entire investment case can be rewritten as a bet on real yields.

### Gold: a bond with no coupon

Gold pays you nothing. No dividend, no coupon, no rent. Its only return is price appreciation, and its appeal is as a store of value that cannot be printed. That makes gold one of the longest-duration assets in existence — its value is entirely about the indefinite future.

So why does gold care about real yields? Because the **opportunity cost** of holding gold *is* the real yield. If you can earn +2% real on a risk-free TIPS bond, then holding a lump of metal that yields nothing means giving up 2% of guaranteed purchasing power every year. Gold has to "compete" against that real return. When real yields are deeply negative — when the alternative is a *guaranteed loss* of purchasing power in bonds — gold's zero yield suddenly looks fantastic, and gold rips. When real yields turn sharply positive, the carry cost of holding gold soars, and gold struggles. This is why gold rose through the negative-real-yield era of 2019–2021 and then went sideways-to-down in 2022 even as inflation screamed: **gold is not a hedge against inflation; it is a hedge against negative real yields.** The two often coincide, but it is the real yield that does the work.

### Growth stocks: cash flows that live in the future

A "growth" stock is one whose value comes mostly from profits expected far in the future rather than today. By construction, the bulk of its present value sits at large `t` in the discounting formula — which makes it acutely sensitive to the discount rate, exactly like a long-dated bond. A "value" stock, by contrast, throws off cash *now*; less of its value is exposed to the far future, so it has shorter duration and cares less about real yields.

This is the entire mechanical reason growth crushed value from 2010 to 2021 (real yields grinding ever lower, inflating the value of distant cash flows) and then violently underperformed in 2022 (real yields spiking, deflating those same cash flows). The "growth vs value" rotation that strategists agonize over is, to a first approximation, a single chart inverted: the real yield. When the real yield falls, long-duration growth wins. When it rises, short-duration value wins. Crypto is the same trade with the duration dial turned to maximum — no cash flows at all, pure future story — which is why it is the most violently real-yield-sensitive asset of all.

### Crypto: the longest-duration trade on the board

It is worth stating plainly because so much crypto commentary gets it backwards. Bitcoin is often pitched as "digital gold, an inflation hedge." But just like gold, what it actually trades on is **real yields**, not inflation. Bitcoin has no cash flows, no coupon, no dividend — its entire value is a story about the indefinite future, which makes its effective duration close to infinite. That is the longest-duration position you can take, so it is the most sensitive thing on the board to the real discount rate. The clean proof is 2022: inflation was running at a 40-year high — supposedly the perfect environment for an "inflation hedge" — and Bitcoin fell roughly 65% on the year. Why? Because real yields were exploding upward, and the longest-duration asset takes the biggest hit when the real discount rate rises. Crypto did not fail as an inflation hedge; it succeeded perfectly as a *negative-real-yield asset* whose tailwind became a headwind. When you see crypto rip in an easy-money year and crater in a tightening year, you are watching the real-yield mechanism at maximum amplitude.

### The dollar: the other side of the real-yield trade

The U.S. dollar is the one major asset that tends to move *with* real yields rather than against them, and understanding why completes the playbook. Currencies are priced largely on relative real returns: capital flows toward the place that pays the highest *real* return for the risk. When U.S. real yields rise — when dollar-denominated risk-free assets suddenly pay a healthy real return — global capital rotates *into* dollars to capture that return, and the dollar strengthens. When U.S. real yields fall, the dollar's real carry advantage shrinks and capital rotates out, softening the dollar. This is why the dollar is the natural *hedge* against a rising-real-yield regime: in 2022, as real yields spiked and gold, growth, crypto, and long bonds all fell, the dollar index surged to multi-decade highs. The same dial that crushed long-duration assets lifted the dollar. If you want a single position that profits when real yields rise, long dollars is often the cleanest expression — it is the mirror image of the long-duration assets that get hurt.

## Common misconceptions

A handful of intuitive-sounding beliefs about money will actively cost you if you trade on them. Each is corrected with a number.

### Myth 1: "Nominal returns are what matter — I made money, the number went up."

The most seductive error in finance. If your portfolio rose 8% in a year when inflation ran 9%, you did not make money — you *lost* about 1% of purchasing power. You can buy less stuff than you could a year ago, despite a bigger account balance. From mid-2021 to mid-2022, CPI peaked at **9.06%**: anyone holding cash or low-yielding assets *felt* stable (the nominal number didn't fall) while quietly losing nearly a tenth of their real wealth. **Always translate nominal into real before you congratulate yourself.** The market does; so should you.

### Myth 2: "Inflation is always bad for stocks."

It depends entirely on *why* and on what real yields do. Moderate, demand-driven inflation that comes with growth can be fine or even good for equities — companies raise prices, nominal earnings grow, and if real yields stay low the discount rate stays friendly. What kills stocks is not inflation per se but the *policy response to it*: the Fed hiking aggressively and driving **real** yields sharply positive. In 2022, stocks fell not because inflation was 9% but because the real 10-year yield went from −1% to +1.7% — a ~270 basis point swing in the discount rate. Inflation was the trigger; the real-yield spike was the bullet.

### Myth 3: "Cash is safe."

Cash is *nominally* safe — its dollar value cannot fall — but in real terms it is one of the riskiest things you can hold during inflation, because it is guaranteed to lose purchasing power at the inflation rate. Cash earning 0% while inflation runs 9% loses 9% of its real value in a year, every year, with certainty. We will put a precise dollar figure on this below. "Safe" is a statement about nominal volatility, not about preserving what your money can buy.

### Myth 4: "Real yields and inflation are the same thing — high inflation means high real yields."

They are nearly *opposite* in the short run. When inflation surprises higher and the Fed is behind, real yields can be deeply *negative* (nominal rates haven't caught up). Real yields rise only when nominal rates climb *faster* than inflation expectations — which usually means the Fed is fighting inflation hard. So a spike in inflation can come with *falling* real yields (early in the spike) or *rising* real yields (once the Fed responds). Never assume; subtract.

#### Worked example: the real cost of "safe" cash, 2021–2022

You hold **\$100,000** in cash, feeling prudent, from mid-2021 to mid-2022. The dollar figure never drops — your statement still reads \$100,000. But CPI over that window peaked at **9.06%** year-over-year. In real terms:

```
real value = $100,000 / (1 + 0.0906)
           = $100,000 / 1.0906
           = $91,693
```

Your \$100,000 now buys what **\$91,693** bought a year earlier. You lost roughly **\$8,300 of purchasing power** — about 8.3% — while doing nothing "risky" at all. Had you instead bought a 1-year TIPS, the inflation adjustment would have largely offset that loss.

**Intuition:** "safe" is a nominal word. In a 9% inflation year, sitting in cash is a guaranteed −8% real trade — one of the worst risk-adjusted positions on the board, dressed up as caution.

## How it shows up in real markets

Now stitch the mechanism onto the actual tape. The real yield does not just explain things in theory; it draws a clean line through the last several years of market history.

### 2020–2021: negative real yields and the everything-rally

Coming out of the pandemic crash, the Fed cut policy rates to zero and bought trillions in bonds, while inflation expectations stayed anchored. The result: the 10-year real yield went deeply negative, sitting around **−1.0% for most of 2021** (it was −1.06% in January 2021 and −1.04% in December 2021). Lending to the U.S. government for a decade guaranteed a real loss. When the risk-free real return is *negative*, the discount rate on everything collapses, and the present value of every long-dated cash flow inflates. So *everything* rallied: stocks (especially long-duration tech), crypto, real estate, SPACs, unprofitable growth, NFTs, you name it. The longer the duration and the flimsier the cash flow, the bigger the gain — exactly what the discounting math predicts when `r` goes below zero. This was not irrational exuberance in a vacuum; it was rational repricing to a negative discount rate, layered with some genuine mania on top.

### 2022–2023: positive real yields and the drawdown

Then inflation refused to be transitory, CPI rocketed to 9.06%, and the Fed embarked on the fastest hiking cycle in 40 years — lifting the funds-rate ceiling from 0.25% in early 2022 to 5.50% by mid-2023. Crucially, the Fed hiked *faster* than inflation expectations rose, so **real** yields exploded upward. The 10-year real yield crossed zero in spring 2022 and reached **+2.48% by October 2023**. The discount rate on the entire market had swung by about 3.5 percentage points in under two years. Everything that had soared on negative real yields now had to be repriced for sharply positive ones. Long-duration tech fell hardest, crypto cratered, the most speculative growth lost 80–90%, and even "safe" long bonds had a historic drawdown. Same dial, opposite direction.

![Area chart of the 10-year real yield swinging from negative to positive with zero as the regime divide](/imgs/blogs/real-vs-nominal-inflation-real-yields-master-signal-5.png)

The figure isolates the move that did the damage. The red region is the negative-real-yield regime of 2020–2021 — the easy-money world where waiting was free and long-duration assets ran. The green region is the positive-real-yield regime that arrived in 2022 — the disciplined world where capital has a real cost again. **The zero line is the regime divide.** Crossing it from below was the single most important macro event of the cycle, and it was visible in real time on this one series. If you had been watching DFII10 and nothing else, you would have seen the regime change coming as the green line marched toward zero in early 2022 — well before the worst of the drawdown.

### Gold vs real yields: the cleanest relationship in macro

Over multi-year horizons, gold and the real yield trace out one of the most reliable inverse relationships in all of markets. Gold climbed through the falling-real-yield decade and peaked alongside the deepest negative real yields. When real yields spiked in 2022, gold's tailwind became a headwind and it went sideways-to-down despite 9% inflation — the single best proof that gold tracks *real yields*, not inflation. (Gold's later strength has coincided with markets pricing real-yield cuts and rising sovereign-risk demand — again, a real-rate-and-safe-haven story, not a simple inflation story.) If you only ever overlaid two lines in macro, make them gold and the inverted 10-year real yield.

### When the relationship "breaks" (and why it usually didn't)

Every few months someone declares that "gold has decoupled from real yields" or "growth stocks stopped caring about rates." Sometimes that is real and informative; usually it is a sign the observer is using the wrong inputs. Three things commonly cause an *apparent* break:

First, **using realized inflation instead of expected inflation.** If you compute a "real yield" by subtracting *last month's* CPI from today's nominal yield, you will get a number that bounces around meaninglessly and correlates with nothing. The real yield that matters is the *forward-looking* one the TIPS market quotes (DFII10), built on expected inflation over the bond's life. Always use the market's real yield, not a homemade one.

Second, **a second driver swamping the real-yield channel.** Gold is mostly a real-yield trade, but it also carries a safe-haven and a central-bank-demand component. In a geopolitical crisis or a sovereign-debt scare, those components can dominate and gold can rise even as real yields tick up. The relationship hasn't "broken"; a second, larger force is temporarily in control. The discipline is to ask *which* driver is dominant right now, not to abandon the framework.

Third, **lags and positioning.** The repricing is not instantaneous. Markets can lag a real-yield move for weeks when positioning is offside, then snap to it violently. A few weeks of "decoupling" often resolves into a sharp catch-up. The signal is right; the timing is noisy. That is true of every macro relationship and is not a reason to discard the best one you have.

The takeaway: the real-yield framework is not a mechanical law that holds tick-for-tick. It is the dominant gravitational force, and like gravity it can be temporarily overwhelmed by a thrust in another direction. Knowing it is the *baseline* — and knowing what the second-order forces are — is exactly what separates a trader who understands the tape from one who is surprised by it.

### 2023–2026: higher-for-longer and the live re-acceleration

The story did not end with the 2022–23 spike. Real yields stayed *positive and elevated* — the "higher for longer" regime — even as the Fed began trimming the policy rate from its 5.50% peak. A positive real yield in the 1.5%–2.5% range is a fundamentally different world from the negative-real-yield decade that preceded it: capital has a real cost again, the most speculative long-duration trades no longer get a free ride, and "the everything rally" cannot simply resume. Then, as the CPI chart showed, inflation began *re-accelerating* in 2025–26, climbing back above 4%. That re-acceleration is a live test of the framework: if it forces breakevens higher and the Fed back onto a tightening footing, real yields could press higher still, and the long-duration assets that have re-inflated would face the same headwind that hit them in 2022. The point is not to predict the outcome here but to show that the *same dial* is still the thing to watch. Every cycle since has been, at its core, a story about which side of zero the real yield sits on and which direction it is heading.

## How to trade it / The playbook

Everything above collapses into a small set of things you actually do. This is the payoff.

### 1. Watch the 10-year real yield as your master dial

Put **DFII10 (the 10-year TIPS real yield)** on your screen and treat it as the most important macro number you track. Its *level* tells you the regime (negative = easy money, long-duration assets favored; positive and rising = tight, long-duration assets pressured). Its *direction* tells you the wind. The single most important threshold is **zero**: crossing from negative to positive, as in 2022, is a regime change that reprices everything long-duration. You do not need to forecast it perfectly — you need to know which side of zero you are on and which way it is moving.

A practical reading frame: bucket the real yield into rough zones. **Below −0.5%** is the deeply-easy regime where long-duration assets run and "buy the dip" works almost mechanically. **−0.5% to +1%** is a transition zone — friendly, but the free-money tailwind is fading. **+1% to +2.5%** is the disciplined regime we have lived in since 2022, where valuation discipline returns and the most speculative trades stop getting a free pass. **Above +2.5%** is genuinely restrictive — the level that historically strains long-duration assets and eventually the broader economy. You will not trade off these buckets mechanically, but they orient you instantly: a glance at the real yield tells you which of these worlds you are standing in.

And watch the *change*, not just the level. A 25-basis-point move in the 10-year real yield over a few weeks is a meaningful repricing of the cost of capital; a 50-basis-point move is a regime tremor. Because long-duration assets have durations of 20, 30, even 50, a 50-bp real-yield move implies *double-digit* percentage moves in the most sensitive assets — which is exactly the kind of move that feels like "the market broke" to anyone not watching the dial.

### 2. Watch breakevens to know *why* the real yield is moving

Always decompose. A nominal yield move can come from the real yield *or* from breakeven inflation, and the two mean opposite things for your book:

- **Nominal up because real up, breakeven flat** → the Fed is winning / growth is firming; discount rate rising; *bad* for gold, growth, crypto, long bonds; *good* for the dollar.
- **Nominal up because breakeven up, real flat** → an inflation *scare*; this is the one case where gold can rally *with* rising nominal yields, because the real yield isn't the thing moving. Don't blindly short gold on a rising nominal yield until you've checked the breakeven.

Pull up the 10-year breakeven (nominal minus DFII10) right next to the real yield. The two together tell you the whole story; either one alone can mislead you.

### 3. Map your book's real-yield sensitivity (its "real-yield beta")

Go position by position and ask: *how long-duration is this?* The more an asset's value depends on far-future cash flows or pure store-of-value demand, the more it falls when real yields rise. A rough ranking, longest-duration (most real-yield-sensitive) first: **crypto → unprofitable growth / long-dated tech → gold → long-duration bonds → broad equities → value / high-cash-flow stocks → short-duration bonds → cash.** If your book is stacked at the top of that list, you are running a giant *short* position on real yields whether you meant to or not. Know that before the dial moves, not after.

This is the single most useful exercise in the post, so do it literally. Write down each position and tag it with a rough duration and therefore a rough real-yield beta. A portfolio of long-dated tech, a gold allocation, and a crypto sleeve is, in aggregate, an enormous bet that real yields *fall* — even if you never thought of it that way, even if you "diversified" across asset classes. The diversification is an illusion, because all three are the *same* real-yield trade wearing different clothes. The 2022 drawdown was so brutal for so many "balanced" portfolios precisely because the supposed diversifiers — long bonds, gold, growth, crypto — were all long-duration and all reprised by the same rising real yield at the same time. Real diversification means deliberately holding something with the *opposite* real-yield sign: short-duration assets, the dollar, or cash, which gain (in relative terms) when real yields rise. If every line in your book wins together when real yields fall, you do not have a portfolio; you have one position in a trench coat.

### A worked read: a hot CPI print

Tie it together with how you would actually react on a data day. Suppose CPI comes in hot — well above expectations. The naive read is "inflation up, buy gold, it's an inflation hedge." The real-yield read is more careful: a hot CPI raises the odds the Fed hikes or stays tight, which pushes *nominal* yields up — but you must check whether the move is in the **real** yield (the Fed is expected to get more restrictive in real terms, *bad* for gold and growth) or in the **breakeven** (the market just marked up its inflation forecast, real yields flat, which can *support* gold). In the 2022 episode, hot prints drove real yields up and gold *down*, the opposite of the naive trade. Your checklist on the print: (1) did nominal yields move? (2) of that move, how much was real (DFII10) vs breakeven? (3) which of your positions has the most real-yield beta? The answer to those three tells you how your book should be leaning before the headline-chasers have finished reading the press release.

![Two-column comparison of how rising and falling real yields affect gold growth crypto dollar and bonds](/imgs/blogs/real-vs-nominal-inflation-real-yields-master-signal-7.png)

The playbook figure lays out both regimes side by side. **Real yields rising (tighter):** headwind for gold, growth stocks, crypto, and long bonds; tailwind for the dollar. **Real yields falling (easier):** the mirror image — gold, growth, and crypto get their tailwind back; the dollar softens. The middle column is the signal stack (real yield, breakeven, your book's beta) and the things that *invalidate* the simple read — chiefly an inflation scare (breakeven up, real down) that lifts gold while the dollar is mixed, or a growth shock that pulls real yields *and* risk assets down together. The discipline is to always ask **why** the real yield moved before you act on the direction.

### 4. The position and the invalidation

The cleanest expressions of a real-yield view: when you expect real yields to *rise*, lean toward the dollar and short-duration assets and away from gold, long-duration growth, and crypto; when you expect them to *fall*, do the reverse. The trade is **invalidated** when the *driver* of the yield move flips — for example, you're positioned for "rising real yields hurt gold," but the next leg up in nominal yields is actually a breakeven (inflation-scare) move with real yields flat or falling, in which case gold can rally and your thesis is wrong for the right reason. The rule that survives every regime: **never trade the nominal yield. Decompose it into real plus breakeven, and trade the part that actually moves the discount rate — the real yield.**

Two more invalidation cases deserve a flag, because they are the ones that catch people out. The first is the **growth-shock case**: in a sharp recession scare, real yields fall (the market prices Fed cuts) *and* risk assets fall (earnings expectations collapse), so the usual "falling real yields are good for stocks" relationship inverts for a while — the cash-flow numerator is dropping faster than the discount-rate denominator is helping. The second is the **safe-haven case** for gold: in a genuine crisis, gold can rally on flight-to-safety demand even as real yields rise, because its safe-haven component temporarily dominates its real-yield component. In both cases the framework hasn't failed; a second, larger force is in control, and the move you'd expect from real yields alone is being overridden. The professional habit is to always name the *dominant* driver before you size a position — real yield, breakeven, growth, or safe-haven — rather than assuming the real-yield channel is the only thing operating. Most of the time it is the biggest force; knowing when it isn't is the edge.

Finally, sizing and patience. The real-yield signal is a *directional gravity*, not a market-timing tool. It will tell you the wind, not the day. Real-yield moves can run further and faster than feels reasonable (the 2022 move was historic), and they can pause for weeks before assets catch up. So express the view with sizing and time horizon that can survive the noise: scale in, respect that the catch-up can be violent, and treat the real yield as the slow tide that decides where the water level ends up, not the individual waves. Get the tide right — which side of zero, which direction, driven by what — and you are positioned with the deepest force in markets at your back instead of in your face.

That is the master dial. One number — the 10-year real yield — sets the true cost of capital, the discount rate behind every asset, and the wind behind gold, growth, and crypto. Learn to watch it, learn to decompose it, and you are reading the deepest current in markets while everyone else stares at the nominal number on the screen.

## Further reading & cross-links

- [Interest Rates: The Price of Money and the Master Variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — the broader frame for why rates sit at the center of every asset price; the real yield is the inflation-adjusted heart of it.
- [How the Fed Sets Interest Rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the mechanism by which the central bank turns the policy dial that ultimately moves real yields.
- [Paul Volcker and the 1980 Rate Shock: Killing Inflation](/blog/trading/finance/paul-volcker-1980-rate-shock-killing-inflation) — the historical case study of driving real yields violently positive to break an inflation regime.
- [What Money Really Is: Base Money, Broad Money, and What Traders Watch](/blog/trading/macro-trading/what-money-really-is-base-money-broad-money-traders) — the money-supply backdrop that, alongside real yields, conditions the inflation half of the Fisher equation.
