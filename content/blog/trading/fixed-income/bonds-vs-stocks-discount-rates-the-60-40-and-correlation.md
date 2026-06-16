---
title: "Bonds vs stocks: discount rates, the 60/40, and correlation"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into how bonds and stocks are joined at the hip through the interest rate: why a higher discount rate lowers every stock's value (and crushes long-duration growth names hardest), why the 60/40 portfolio leans on bonds as ballast, and why the stock-bond correlation flips sign by regime — usually negative, but brutally positive in 2022 when both fell together."
tags: ["fixed-income", "bonds", "stocks", "discount-rate", "60-40-portfolio", "stock-bond-correlation", "diversification", "duration", "asset-allocation"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — bonds and stocks look like opposites, but they are priced by the same machine: a stream of future cash discounted back to today at a rate that starts with the bond market. When that rate rises, every cash flow far in the future is worth less now — which is why a jump in yields can sink a tech stock harder than the bank's bond.
> - A stock is a **very long-duration cash-flow stream** (dividends and buybacks stretching decades out), so its price is *extremely* sensitive to the discount rate. Raise the rate and you slash the present value of cash flows that arrive in year 20 — exactly where growth stocks keep most of their value.
> - The **60/40 portfolio** — 60% stocks, 40% bonds — is the classic balanced mix. Bonds are the *ballast*: in a normal recession the Fed cuts rates, bond prices rise, and that gain cushions the equity crash. That is the whole reason the 40 exists.
> - The **stock-bond correlation** is not a constant. It is usually *negative* when the shock is about growth (a recession scare — bonds rally as stocks fall, the hedge works). It turns *positive* when the shock is about inflation and rates (both repriced down together).
> - **2022** is the cautionary tale: inflation surged, the Fed hiked hard, and a 60/40 portfolio lost roughly **−16%** because *both* legs fell at once — the worst year for the strategy in modern history.
> - The diversification benefit is real but **conditional**: it depends on the sign of the correlation, which depends on what is driving markets. Diversification is insurance that lapses exactly when the risk is inflation.
> - Running example: a **\$100,000 60/40 portfolio** — we trace its return in a normal year, in a normal recession where bonds save the day, and in 2022 where they did not.

Why does a single number announced by a committee of central bankers — the interest rate — move the price of a technology stock that pays no dividend, a 30-year government bond, a house, and a gold bar, all at once and often in the same direction? And why, in most years, do bonds quietly *rise* when stocks fall — making them the thing that stops a balanced portfolio from cratering — and yet in 2022 did exactly the opposite, falling *with* stocks and leaving investors with nowhere to hide?

The answer to both questions is one idea, and it is the idea this whole post is built around: **every financial asset is a claim on future cash, and the price of that asset is its future cash flows discounted back to today at a rate that starts in the bond market.** A bond's cash flows are fixed and spelled out in the contract. A stock's cash flows are uncertain and stretch out forever. But the *machine* that turns "future money" into "what it's worth now" is the same machine — and the dial on that machine is the interest rate. Turn the dial and you reprice everything. That shared dial is the thread that joins bonds and stocks at the hip, and it is why a fixed-income concept — the discount rate — turns out to be the master variable for the stock market too.

![A side by side comparison of a stock's value before and after the discount rate rises, showing that a higher rate lowers the present value of all future cash flows and hits the far future cash flows of a long duration growth stock hardest](/imgs/blogs/bonds-vs-stocks-discount-rates-the-60-40-and-correlation-1.png)

The diagram above is the mental model for everything that follows. On the left, a stock is valued by discounting its future cash flows at a low rate; the bars are tall because money in the future is still worth a lot when rates are low. On the right, the rate has risen, and every bar shrinks — but the *far* bars, the cash flows ten and twenty years out, shrink the most. That is the entire story of why rising rates hurt long-dated growth stocks more than steady value stocks, why bonds and stocks are connected, and why the 60/40 portfolio works — until it doesn't. (Everything here is educational, not investment advice; the goal is to understand the mechanism, not to tell you what to buy or hold.)

## Foundations: the building blocks you need first

Let's assemble the vocabulary from zero. A few of these terms you may have met in a bond context; here we stretch them to cover stocks too, because the punchline is that the same toolkit prices both.

**A bond is a promise to pay fixed cash on fixed dates.** When you buy a bond you are lending money. The borrower (the *issuer*) promises to pay you periodic interest — *coupons* — and to return the original loan amount — the *face value* or *par*, almost always **\$1,000** — at a set *maturity* date. A US Treasury is a bond issued by the US government, treated as effectively certain to pay, so its yield is the *risk-free rate* — the baseline price of money. The full anatomy of par, coupon, and maturity lives in [the anatomy of a bond](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer).

**A stock is a residual claim on a company's future.** When you buy a stock you buy a tiny slice of ownership. Unlike a bondholder, you are promised *nothing* — no fixed coupon, no return of principal, no maturity date. What you own is a claim on whatever cash the company throws off forever: the *dividends* it chooses to pay, and the *buybacks* (when the company uses cash to repurchase its own shares, raising the value of each remaining share). The stream is uncertain and, crucially, it has **no end date**. A stock is a perpetual, variable cash-flow stream; a bond is a finite, fixed one. That single difference drives most of what follows.

**The discount rate — the dial that prices everything.** A dollar you receive in the future is worth less than a dollar today, because today's dollar can be invested to grow. The *discount rate* is the rate at which we shrink future dollars back to today's value. The value of *any* asset is the sum of all its future cash flows, each shrunk by the discount rate for how far away it is. The discount rate is not invented from nothing — it is built up from the *risk-free rate* (the Treasury yield) plus a *risk premium* (extra return demanded for uncertainty). When the bond market moves the risk-free rate, the floor under every discount rate moves with it. This is the master variable; its macro side is told in [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).

**Present value — the arithmetic of the dial.** *Present value* (PV) is what a future cash flow is worth today after discounting. The formula for a single cash flow $C$ arriving in $t$ years at discount rate $r$ is:

$$\text{PV} = \frac{C}{(1+r)^t}$$

Here $C$ is the future cash, $r$ is the annual discount rate (as a decimal), and $t$ is the number of years until it arrives. The whole value of a stock or bond is just this formula summed over every cash flow it will ever pay. Notice the $t$ in the exponent: the further out a cash flow is, the harder the denominator grows, and the more a change in $r$ matters. That exponent is the secret to everything.

**Duration — how sensitive a price is to the rate.** *Duration* measures how much an asset's price moves when its discount rate changes, and it is roughly the *weighted-average time* until you get your cash. A bond that pays you back soon has *low* duration and barely flinches when rates move; a bond that pays you back far in the future has *high* duration and swings hard. The full mechanics are in [duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income). The key leap in this post: **a stock behaves like a bond with enormous duration**, because so much of its value sits in cash flows decades away.

**Correlation — do two things move together?** *Correlation* is a number between −1 and +1 that says whether two assets tend to move in the same direction. **+1** means they move in perfect lockstep; **−1** means perfectly opposite (one up, one down); **0** means no relationship. It is the single most important number in building a balanced portfolio, because two assets that move *opposite* each other can cancel out each other's bad days. The stock-bond correlation — and the shocking fact that its *sign* is not fixed — is the heart of this post.

**A portfolio, and the 60/40.** A *portfolio* is just the collection of assets you hold. The classic balanced portfolio is the **60/40**: 60% of your money in stocks (for growth) and 40% in bonds (for stability and income). It has been the default recommendation for ordinary investors and pension funds for decades, and its entire logic rests on the bond leg behaving differently from the stock leg when things go wrong.

With those seven ideas in hand, here is the sentence that motivates the whole post: **a stock and a bond are both bundles of future cash priced by the same discount rate, so they are cousins, not strangers — and whether they help or hurt each other in a portfolio depends entirely on whether that shared discount rate is moving for good reasons (growth) or bad ones (inflation).**

## A stock is a long-duration bond in disguise

Start with the cleanest possible version. Forget about growth, risk, and uncertainty for a moment, and price the two assets side by side with the same formula.

A **5-year \$1,000 Treasury note with a 4% coupon** pays you \$40 a year for five years, then \$1,000 back at the end. Its value is the present value of those six cash flows. The bulk of the money — the \$1,000 principal — arrives in year 5. So the *weighted-average* time until you get paid is a bit under 5 years. Its duration is roughly 4.5 years. If the discount rate rises by 1%, the price falls by about 4.5%. Annoying, but bounded. (This is the seesaw covered in [price and yield](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds).)

Now price a stock. Imagine a steady company — call it **Evergreen Utilities** — that you expect to pay roughly \$3 per share in dividends-plus-buybacks every year, growing slowly, *forever*. There is no maturity, no return of principal. The value is the present value of an infinite stream of cash flows. Where is the "weight" of that stream? Not in the next five years — those near-term dividends are a small fraction of the total. Most of the value sits in years 10, 20, 30, and beyond, summed across the infinite tail. The weighted-average time until you get your cash is *enormous* — often **15 to 30 years or more**. A stock is, in duration terms, a bond from the distant future.

That is the whole insight, and it is worth saying plainly: **the reason interest rates move stocks is that a stock is a long-duration cash-flow stream, and long-duration things are violently sensitive to the discount rate.** The bond market doesn't move stocks by some mysterious sentiment channel; it moves them through the arithmetic of the present-value formula, the same arithmetic that prices the bond.

#### Worked example: the same rate hike, two very different hits

Let's make the duration difference concrete with real arithmetic. Suppose the discount rate rises by **1 percentage point** — say from 4% to 5%.

The 5-year Treasury note has a duration of about 4.5 years. The price impact is approximately duration × rate change:

$$\Delta P \approx -D \times \Delta r = -4.5 \times 0.01 = -0.045$$

So its price falls about **−4.5%**. A \$1,000 note becomes worth about \$955. You lost \$45 of price, but you still collect every \$40 coupon and your \$1,000 back in five years. The damage is real but small and self-healing.

Now Evergreen Utilities, with an effective duration of, say, **18 years** (typical for a steady dividend payer). The same 1% rate rise:

$$\Delta P \approx -18 \times 0.01 = -0.18$$

Its price falls about **−18%**. A \$100 stock drops to about \$82 — for the *exact same* move in the discount rate that cost the bond only 4.5%. The stock didn't fall four times as much because of fear or news; it fell four times as much because its cash flows sit four times further out in time.

*A stock and a bond feel the same rate hike, but the stock feels it four times harder because its money lives four times further in the future.*

### The single formula that makes it rigorous

There is a clean equation that turns this intuition into algebra, and it is worth seeing because it shows *exactly* why a stock's price is so leveraged to the rate. For a company paying a dividend $D$ next year, growing it at rate $g$ forever, discounted at rate $r$, the value of the stock simplifies to the **Gordon growth model**:

$$V = \frac{D}{r - g}$$

Here $V$ is the stock's value today, $D$ is next year's cash flow, $g$ is the perpetual growth rate of that cash flow, and $r$ is the discount rate. Stare at the denominator: $r - g$. When $r$ is only a little above $g$, that denominator is *tiny*, and dividing by a tiny number gives a *huge* value — and tiny numbers are violently sensitive to small changes. If $r = 7\%$ and $g = 5\%$, the denominator is 2%; nudge $r$ up to 8% and the denominator becomes 3% — a 50% jump in the denominator, which slashes the value by a third. The closer a company's growth rate is to the discount rate (the hallmark of a high-growth stock), the smaller the denominator, and the more a rate move detonates the price. This is the same long-duration sensitivity from before, now visible as a single fraction: growth stocks are the ones whose denominators are smallest, which is why they get repriced the hardest.

#### Worked example: the Gordon model under a rate rise

Take **Evergreen Utilities** with next-year cash flow $D = \$3$, growth $g = 4\%$, discount rate $r = 7\%$:

$$V = \frac{\$3}{0.07 - 0.04} = \frac{\$3}{0.03} = \$100$$

Now raise $r$ to 8% (a 1-point rate rise), $g$ unchanged:

$$V = \frac{\$3}{0.08 - 0.04} = \frac{\$3}{0.04} = \$75$$

The value fell from \$100 to \$75 — a **−25%** drop — from a single 1-point rate move, because the denominator jumped from 3% to 4%. Now repeat for **Hyperion Software**, a growth name with $g = 6\%$ and the same $r = 7\%$ ($D = \$3$): start at \$3 / 0.01 = \$300, then \$3 / 0.02 = \$150 — a **−50%** collapse from the identical rate move, because its denominator started at a razor-thin 1%.

*The closer a stock's growth rate sits to the discount rate, the thinner the denominator that prices it, and the more a small rate move blows the price apart — which is precisely the definition of a growth stock.*

## Why growth stocks get hit hardest

Here is where it gets sharper, and where the 2022 carnage in technology stocks finally makes sense. Not all stocks have the same duration. The duration of a stock depends on *when* its cash flows arrive — and that varies enormously between a **value** stock and a **growth** stock.

A **value stock** is a mature company — a bank, a utility, a consumer-staples giant — that already earns a lot of cash *now* and pays much of it out as dividends *now*. A big chunk of its value sits in near-term cash flows. Its duration is high (it's still a stock), but not extreme.

A **growth stock** is a company whose value is almost entirely in the *future* — a young software or biotech firm that earns little or nothing today but is expected to earn vast sums a decade or two out. Almost *none* of its value is in the next few years; it is *all* in the far tail. Its effective duration is gigantic — 30, 40, sometimes 50+ years. And recall the present-value formula: the further out a cash flow, the more a change in $r$ destroys its present value, because the exponent $t$ is bigger.

So when the discount rate jumps, the growth stock — whose value is concentrated exactly where the discounting bites hardest — gets *crushed*, while the value stock, with more cash up front, takes a softer blow. This is not a metaphor; it is the exponent in the denominator doing its work.

#### Worked example: a 1% rate rise hits growth far harder than value

Picture two companies, each "worth" \$100 a share before rates move.

**Value Co.** is a mature firm. Roughly half its value comes from cash flows in the next 5 years, half from beyond. Effective duration: about **12 years**. A 1% rate rise:

$$\Delta P \approx -12 \times 0.01 = -0.12 \quad\Rightarrow\quad \$100 \to \$88$$

It falls **−12%**, to about \$88.

**Hyperion Software** is a high-growth firm earning almost nothing today; nearly all its value is in cash flows 15+ years out. Effective duration: about **35 years**. The same 1% rate rise:

$$\Delta P \approx -35 \times 0.01 = -0.35 \quad\Rightarrow\quad \$100 \to \$65$$

It falls **−35%**, to about \$65 — for the identical move in rates that cost Value Co. only 12%. Hyperion didn't have worse news; it just had *later* cash flows, and later cash flows are exactly what a higher discount rate punishes.

*Two stocks, one rate move, wildly different damage — because the discount rate punishes the future, and a growth stock is almost all future.*

This is precisely what happened in 2022. The Nasdaq-heavy basket of long-duration tech and growth names fell roughly **−33%** on the year, while the broad value-tilted Dow fell only single digits, *for the same underlying cause*: the discount rate, dragged up by the bond market, repriced the far-future cash flows that growth stocks are made of. The full macro mechanism — why the Fed moves the risk-free rate at all — is in [the central bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) and [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

## The 60/40 portfolio and why bonds are the ballast

Now we can build the portfolio. The **60/40** — 60% stocks, 40% bonds — has been the default balanced portfolio for half a century, and its logic is beautifully simple: stocks provide the *growth*, bonds provide the *stability*, and — the crucial part — bonds tend to do *well* in exactly the moments when stocks do *badly*. Bonds are the *ballast*: the heavy weight low in the hull that keeps the ship from capsizing when the equity sea gets rough.

Why would bonds rise when stocks fall? Because of *what usually causes stocks to fall*. The classic equity crash is a **recession scare**: growth weakens, earnings fall, fear rises, and stocks drop. But a weakening economy is exactly when the Federal Reserve *cuts* interest rates to stimulate borrowing and spending. Falling rates mean **rising bond prices** (the seesaw again — see [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much)). So in a normal recession the same event — a growth scare — pushes stocks *down* and bonds *up*. The bond leg of your portfolio rallies precisely when the stock leg is bleeding, cushioning the blow.

![A before and after comparison showing a balanced sixty forty portfolio in a normal recession where stocks fall but bonds rise, with the bond gain partly offsetting the equity loss so the total drawdown is much smaller than holding stocks alone](/imgs/blogs/bonds-vs-stocks-discount-rates-the-60-40-and-correlation-3.png)

The figure shows the mechanism. On the left is what a recession does to your stock-only money: a steep drawdown, the full force of the crash. On the right is the 60/40: the stock sleeve still falls, but the bond sleeve *rises*, and the two partly cancel. The portfolio still loses money — bonds rarely rise as much as stocks fall — but it loses far less, and the smoother ride is what lets investors stay invested instead of panic-selling at the bottom. The allocation lens on this engine, including how much bond to hold, is developed in [the stock-bond correlation, the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).

#### Worked example: a \$100,000 60/40 in a normal recession

Start with our running portfolio: **\$100,000**, split \$60,000 in a stock index fund and \$40,000 in an intermediate Treasury bond fund (duration about 6 years).

A recession hits. Stocks fall **−25%**. The Fed cuts rates by about 2 percentage points to fight the slowdown, and with a 6-year duration the bond fund *gains* about +12% (6 × 2%).

- **Stock sleeve:** \$60,000 × (1 − 0.25) = \$45,000. A loss of **−\$15,000**.
- **Bond sleeve:** \$40,000 × (1 + 0.12) = \$44,800. A gain of **+\$4,800**.
- **Total:** \$45,000 + \$44,800 = **\$89,800**. A loss of **−\$10,200, or −10.2%**.

Compare that to a stock-only investor, who would be down the full **−\$25,000, or −25%**. The bonds didn't make the recession profitable, but they turned a brutal −25% into a survivable −10.2% — they absorbed roughly **\$4,800** of the pain and softened the rest.

*The 40 in 60/40 earns its keep in a recession: bonds rally as the Fed cuts, and that gain is the ballast that keeps you from capsizing.*

This is why pensions, endowments, target-date retirement funds, and ordinary savers all gravitated to some version of the 60/40. It is not the highest-returning portfolio — pure stocks beat it over very long horizons — but it has historically delivered most of the stock market's return with far smaller drawdowns, because the bond ballast worked. The reason it worked is the next, and most important, idea in this post: the *correlation* between the two legs.

## The stock-bond correlation and why its sign flips

Here is the single most important — and most misunderstood — fact in asset allocation: **the correlation between stocks and bonds is not a constant. It changes sign depending on what is driving markets.**

For most investors who came of age between roughly 2000 and 2021, the stock-bond correlation was *negative*. Bonds zigged when stocks zagged. This felt like a law of nature, baked into the very definition of "balanced portfolio." It was not a law of nature. It was a feature of a particular *regime* — a long stretch in which the dominant shock to markets was about *growth*, and inflation was low and stable. Change the regime, and the sign of the correlation flips.

![A line chart of the rolling stock and bond correlation over time from the year two thousand to twenty twenty four, sitting below the zero line for most of the period meaning bonds hedged stocks, then spiking above zero in twenty twenty two when both fell together](/imgs/blogs/bonds-vs-stocks-discount-rates-the-60-40-and-correlation-2.png)

This is the centerpiece figure, and it is the picture every allocator should have burned into memory. (It is illustrative — the exact path is stylized to show the shape, not a precise data series; the as-of regime read is through 2024.) The horizontal line is zero correlation. For most of the 2000s and 2010s the line sits *below* zero — bonds and stocks moved opposite, the hedge worked, the 60/40 hummed. Then look at the right end: in 2022 the line *crosses above* zero into positive territory. Bonds and stocks suddenly moved *together*, both falling, and the entire premise of the balanced portfolio broke for a year. To understand why the sign flips, you have to ask: *what kind of shock is hitting markets?*

### Growth shocks: the correlation is negative (the hedge works)

A **growth shock** is news about the *real economy* — a recession scare, a banking wobble, a geopolitical fright that threatens demand. When growth is the worry, two things happen at once:

1. **Stocks fall.** Lower expected earnings, higher fear, equities drop.
2. **Bonds rise.** A weak economy means the Fed will *cut* rates (and means low future inflation), so bond yields fall and bond prices rise. Investors also flee *to* the safety of Treasuries — the "flight to quality" — bidding their prices up further.

Stocks down, bonds up: **negative correlation.** The bond leg hedges the stock leg. This is the regime that prevailed for two decades, because in a low-inflation world almost every market scare was fundamentally a *growth* scare. Government bonds as the risk-free anchor that rallies in a flight to quality are covered in [government bonds, the risk-free anchor](/blog/trading/cross-asset/government-bonds-the-risk-free-anchor-and-duration).

### Inflation/rate shocks: the correlation is positive (the hedge fails)

An **inflation shock** is different in kind. When the worry is that inflation is too *high*, two things happen at once — and both are bad for both assets:

1. **Bonds fall.** High inflation erodes the fixed coupons a bond pays, and forces the Fed to *raise* rates to fight it. Rising rates mean *falling* bond prices. Bonds get hammered.
2. **Stocks fall too.** The very same rising discount rate that crushes bonds also reprices stocks downward (remember: a stock is a long-duration asset, exquisitely sensitive to the rate). Higher rates lower the present value of all those future cash flows.

Now *both* assets fall, *driven by the same cause* — the rising discount rate. That is **positive correlation**, and it is the nightmare for a 60/40 investor, because the ballast becomes a second anchor dragging the ship down. Diversification doesn't just stop helping; the two legs *amplify* each other.

![A two by two matrix showing that a growth shock makes the stock bond correlation negative so bonds hedge stocks, while an inflation or rate shock makes the correlation positive so bonds and stocks fall together](/imgs/blogs/bonds-vs-stocks-discount-rates-the-60-40-and-correlation-4.png)

The matrix above is the decision rule. Read down the left column — a *growth* shock — and you see stocks down, bonds up, correlation negative, hedge working. Read the right column — an *inflation/rate* shock — and you see stocks down, bonds down, correlation positive, hedge failing. The sign of the stock-bond correlation is, at root, a question about *which column you are in* — and that depends on whether the dominant fear in the market is recession or inflation.

#### Worked example: the same portfolio, two kinds of shock

Take our \$100,000 60/40 again (\$60k stocks, \$40k bonds, 6-year bond duration) and run it through both shocks.

**Growth shock (recession scare):** stocks fall −20%, the Fed cuts ~1.5%, so bonds gain ~+9% (6 × 1.5%).
- Stocks: \$60,000 × 0.80 = \$48,000 (−\$12,000)
- Bonds: \$40,000 × 1.09 = \$43,600 (+\$3,600)
- **Total: \$91,600, a loss of −8.4%.** The bonds cushioned \$3,600 of the loss.

**Inflation shock (rates surge):** inflation spikes, the Fed hikes ~3%, so bonds *lose* ~−18% (6 × 3%), and the same rate surge drags stocks down −20%.
- Stocks: \$60,000 × 0.80 = \$48,000 (−\$12,000)
- Bonds: \$40,000 × 0.82 = \$32,800 (−\$7,200)
- **Total: \$80,800, a loss of −19.2%.** The bonds *added* \$7,200 to the loss.

Same portfolio, same −20% equity fall, but the bond leg swung from cushioning \$3,600 to *piling on* \$7,200 — a \$10,800 difference — purely because the shock changed from growth to inflation.

*The bond leg is a hedge in a growth shock and a second loss in an inflation shock; the regime decides which.*

## 2022: the year both legs fell together

2022 was the real-world stress test of everything above, and it failed the 60/40 the way the theory said it would. After a decade of near-zero rates, inflation surged to a 40-year high (US CPI peaked around **9% in June 2022**). The Federal Reserve responded with the fastest hiking cycle in modern history, lifting its policy rate from near zero to over 4% in a single year. That is a textbook **inflation/rate shock** — and the discount rate dial got cranked hard.

The result: the discount rate rose, and it repriced *everything* downward at once.

- The broad **US stock market** (S&P 500) fell about **−18%** on the year.
- Long-duration **tech and growth** (Nasdaq) fell about **−33%**.
- The broad **US bond market** (Bloomberg US Aggregate) fell about **−13%** — its worst year on record by a wide margin.
- A simple **60/40 portfolio** lost roughly **−16%** — also one of its worst years ever.

![A before and after comparison of two thousand twenty two showing the discount rate surging upward in the middle, with stocks falling on the left and bonds falling on the right at the same time, both legs of the portfolio down together](/imgs/blogs/bonds-vs-stocks-discount-rates-the-60-40-and-correlation-5.png)

The figure captures why 2022 was so disorienting. In a normal bad year, one of these two panels would be green — the bond panel would be rising, offsetting the falling stock panel. In 2022, *both* panels are red. The rising discount rate in the middle is the common cause: it crushed bond prices directly (rising yields, falling prices) and crushed stock prices through the same present-value channel. There was no ballast because the thing that sank the ship was the thing the ballast is made of. The bank failures that followed in 2023 were downstream of this same rate surge — see [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs), where a bond portfolio's losses from 2022's rate spike helped topple a bank.

#### Worked example: the 2022 60/40 in dollars

Take our \$100,000 60/40 and apply 2022's actual rough numbers.

- **Stock sleeve:** \$60,000 × (1 − 0.18) = \$49,200. A loss of **−\$10,800**.
- **Bond sleeve:** \$40,000 × (1 − 0.13) = \$34,800. A loss of **−\$5,200**.
- **Total:** \$49,200 + \$34,800 = **\$84,000**. A loss of **−\$16,000, or −16%**.

Now contrast with the *normal recession* version from earlier, where the same portfolio lost only −10.2% even though stocks fell *more* (−25% vs −18%). The recession was a worse year for stocks but a *better* year for the portfolio, because the bonds gained \$4,800 instead of losing \$5,200 — a \$10,000 swing in the bond leg, driven entirely by whether the Fed was cutting or hiking.

*2022 cost the 60/40 more than a deeper recession would have, not because stocks fell further, but because for once the bonds fell with them.*

### Three years, one portfolio, three outcomes

It helps to lay the three scenarios side by side, because the *same* \$100,000 60/40 portfolio behaves completely differently depending only on the *kind* of year it walks into. The numbers below come straight from the worked examples above.

| Scenario | Stock return | Bond return | Portfolio return | Bond leg did what? |
|---|---|---|---|---|
| Normal good year | +12% | +3% | **+8.4%** | small income, no drama |
| Normal recession (growth shock) | −25% | +12% | **−10.2%** | rallied, cushioned the crash |
| 2022 (inflation/rate shock) | −18% | −13% | **−16.0%** | fell *with* stocks, no cushion |

Read the rightmost column top to bottom and you have the entire thesis of this post. The bond leg is a quiet income source in a good year, a powerful shock absorber in a growth-driven recession, and — once in a generation — a *second source of loss* when the shock is inflation. The portfolio's holdings never changed across these three rows; only the regime did. That is what it means to say the stock-bond correlation is the master switch: it determines which of these three rows you live in.

![A worked comparison table of a one hundred thousand dollar sixty forty portfolio across a normal good year a normal recession and the year two thousand twenty two, showing the stock leg the bond leg and the total return, with the bond leg green when it rises and red when it falls](/imgs/blogs/bonds-vs-stocks-discount-rates-the-60-40-and-correlation-7.png)

The figure puts those same three scenarios on one canvas so the role of the bond leg jumps out by color. In the good year and the recession the bond cell is green — it added value. In the 2022 column it is red — it subtracted. Notice that the recession row, despite a far worse *stock* return than 2022 (−25% vs −18%), produced a *better* portfolio return (−10.2% vs −16%), purely because the bond cell flipped from green to red. The lesson an allocator takes from this table is uncomfortable but precise: your worst year is not necessarily the one with the worst stock market — it is the one where your hedge stops hedging.

#### Worked example: the income the bond leg pays even in calm years

The table's top row deserves its own arithmetic, because the bond leg isn't only there for crashes — in an ordinary year it quietly pays you. Suppose stocks return a steady +12% and the bond fund yields about 4.5% with no rate change, so it returns roughly its coupon, +4.5%, before we round to +3% net of a small drag. On our \$100,000:

- **Stock sleeve:** \$60,000 × 1.12 = \$67,200 (+\$7,200).
- **Bond sleeve:** \$40,000 × 1.045 = \$41,800 (+\$1,800).
- **Total:** \$109,000, a gain of **+9.0%** — of which the bonds contributed \$1,800 of steady, low-volatility income while you waited for a crash that didn't come.

*The bond leg earns its place in two ways — a rally when stocks crash, and a steady coupon when they don't — so even in the boring years you are paid to hold the insurance.*

## When diversification helps — the math of the correlation

Step back to the general principle, because the 2022 story is a special case of a deeper truth: **the benefit of holding two assets together depends on their correlation, and the lower (more negative) the correlation, the bigger the benefit.**

When two assets are *negatively* correlated, their bad days don't line up — when one is down the other is often up, so the combined portfolio's ups and downs are *smaller* than either asset alone. This is the only "free lunch" in finance: you can reduce risk without giving up much return, simply by combining things that don't move together. When two assets are *positively* correlated, their bad days *do* line up, and combining them barely reduces risk at all — you just own two versions of the same bet.

The math is captured in the portfolio volatility formula for two assets. For a portfolio with weight $w$ in asset 1 and $(1-w)$ in asset 2:

$$\sigma_p^2 = w^2\sigma_1^2 + (1-w)^2\sigma_2^2 + 2w(1-w)\rho\,\sigma_1\sigma_2$$

Here $\sigma_p$ is the portfolio's volatility (its risk), $\sigma_1$ and $\sigma_2$ are the two assets' volatilities, $w$ is the weight in asset 1, and $\rho$ (rho) is the correlation between them. The whole drama lives in that last term, $2w(1-w)\rho\sigma_1\sigma_2$: when $\rho$ is *negative*, that term is *negative*, and it *subtracts* from total risk — the diversification benefit. When $\rho$ is *positive*, it *adds* to risk. The sign of $\rho$ literally flips diversification from a benefit into a penalty.

![A line chart showing portfolio risk on the vertical axis against the stock bond correlation on the horizontal axis, with risk low when correlation is negative and rising steeply as correlation turns positive, illustrating that diversification shrinks risk only when the two assets move oppositely](/imgs/blogs/bonds-vs-stocks-discount-rates-the-60-40-and-correlation-6.png)

The chart shows the diversification benefit directly. On the horizontal axis is the stock-bond correlation, running from −1 (perfectly opposite) on the left to +1 (perfect lockstep) on the right. On the vertical axis is the resulting risk of the 60/40 portfolio. When correlation is deeply negative (left side), portfolio risk is *low* — the legs cancel. As correlation climbs toward and past zero (right side), risk rises steeply — the cancellation disappears and you are left owning two correlated bets. The 60/40 sat comfortably on the left side of this chart for two decades; in 2022 it slid to the right, and its risk jumped accordingly. The deeper allocation math is in [the stock-bond correlation, the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).

#### Worked example: the same 60/40, two correlations

Suppose stocks have volatility $\sigma_1 = 18\%$ and bonds $\sigma_2 = 6\%$, with a 60/40 split ($w = 0.6$). We compute portfolio volatility under two correlation regimes.

**Negative regime ($\rho = -0.3$, the old normal):**
$$\sigma_p^2 = 0.6^2(0.18)^2 + 0.4^2(0.06)^2 + 2(0.6)(0.4)(-0.3)(0.18)(0.06)$$
$$= 0.011664 + 0.000576 - 0.0009331 = 0.011307$$
$$\sigma_p = \sqrt{0.011307} \approx 10.6\%$$

**Positive regime ($\rho = +0.5$, the 2022 world):**
$$\sigma_p^2 = 0.011664 + 0.000576 + 2(0.6)(0.4)(0.5)(0.18)(0.06)$$
$$= 0.011664 + 0.000576 + 0.0015552 = 0.013795$$
$$\sigma_p = \sqrt{0.013795} \approx 11.7\%$$

The portfolio's risk rose from about **10.6%** to **11.7%** — roughly a **10% increase in volatility** — for no change in holdings, purely because the correlation flipped from −0.3 to +0.5. And that understates the felt pain: the *returns* themselves also turned negative together, so the real-world drawdown was far worse than the volatility number alone suggests.

*Diversification is not a property of the assets you hold; it is a property of how they move together — and when that changes, your risk changes even though your portfolio didn't.*

## What actually drives the correlation regime

If the sign of the stock-bond correlation is the master switch, what flips it? The cleanest answer in the academic and practitioner literature is **the level and volatility of inflation.**

In a **low, stable inflation** world, the central bank has room to cut rates aggressively whenever growth stumbles. Every market scare is therefore a growth scare, the Fed-cut-and-bonds-rally reflex kicks in, and the correlation stays *negative*. This described 2000–2021 almost perfectly: inflation parked near 2%, so bonds were free to be a pure growth hedge.

In a **high, volatile inflation** world, the central bank is *constrained* — it can't cut to rescue stocks because it's busy fighting inflation, and the dominant shock becomes inflation itself, which hurts both assets. The correlation turns *positive*. This described the 1970s and early 1980s — when stocks and bonds were positively correlated for years — and it described 2022. The 1970s stagflation episode is exactly why older investors were never surprised that 2022 happened; they had seen the positive-correlation regime before.

This is why the regime question is really an *inflation* question. As long as inflation stays low and well-behaved, the 60/40 hedge works and bonds are ballast. The day inflation becomes the market's central fear, the hedge inverts and bonds become a co-conspirator. The macro plumbing of how inflation forces the Fed's hand is in [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession), and the variable that ties real (inflation-adjusted) rates to every asset price is unpacked in [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything).

There is a subtler channel beneath the inflation story, and it is worth naming because it explains *why* high inflation does the damage rather than just *that* it does. When inflation is low and predictable, a bond is a near-perfect growth hedge: its yield is mostly a clean forecast of future short rates, so a recession scare drags the yield down and the price up. When inflation is high and *volatile*, a new ingredient swells in the yield — the **inflation risk premium**, the extra return investors demand for the uncertainty that inflation will surprise them and erode their fixed coupons. That premium moves *with* inflation fear, and inflation fear is itself a thing that hurts stocks (it pulls forward rate hikes and squeezes margins). So a single underlying force — rising inflation uncertainty — now pushes bond yields up (prices down) *and* stock valuations down at the same time. The bond stops being a clean bet on growth and becomes a bet on inflation, which is exactly the bet stocks are *also* losing. That shared inflation exposure is the deep reason the correlation turns positive: both assets quietly load onto the same risk factor, and when that factor blows up, they sink together. It is the same mechanism that makes long bonds and growth stocks the two assets that suffer most in an inflation scare — both are long-duration claims on a future that just got more expensive to wait for.

#### Worked example: why TIPS would have helped in 2022

There is a fixed-income asset built precisely for the inflation regime: a **TIPS** (Treasury Inflation-Protected Security), whose principal rises with inflation (see [TIPS and inflation-linked bonds](/blog/trading/fixed-income/tips-and-inflation-linked-bonds-protecting-purchasing-power)). Consider an investor who in early 2022 had split their \$40,000 bond sleeve into \$20,000 nominal Treasuries and \$20,000 short-duration TIPS.

- **Nominal sleeve:** \$20,000 × (1 − 0.13) = \$17,400 (−\$2,600), hit by the rate surge.
- **Short TIPS sleeve:** roughly flat to slightly negative, say \$20,000 × (1 − 0.03) = \$19,400 (−\$600), because the inflation adjustment to principal partly offset the rate hit and the short duration limited the rate damage.
- **Bond sleeve total:** \$36,800, a loss of −\$3,200 — versus −\$5,200 for the all-nominal sleeve.

The TIPS didn't make 2022 a good year, but they cut the bond leg's loss by about \$2,000 because they carried *inflation* protection rather than pure *growth*-regime exposure.

*The bond leg's job is to hedge whatever shock is coming; nominal bonds hedge growth shocks, inflation-linked bonds hedge inflation shocks, and a robust 60/40 in an uncertain world might want some of each.*

## How investors are responding — beyond the simple 60/40

The 2022 break did not kill the 60/40; balanced portfolios still hold most of the world's retirement money and have recovered. But it did puncture the assumption that bonds are an *unconditional* hedge, and it pushed thoughtful allocators toward a more honest framing.

First, **bonds hedge growth risk, not inflation risk.** A portfolio that wants protection in *both* regimes needs more than nominal bonds — it might add inflation-linked bonds (TIPS), commodities, or real assets that do well when inflation is the problem. The all-nominal 60/40 is implicitly a bet that the next crisis will be a growth crisis, not an inflation one.

Second, **the higher rate level after 2022 made bonds attractive again on their own merits.** When the 10-year Treasury yielded 1.5% in 2020, the bond leg offered almost no income and almost no room to rally further. After yields rose to 4–5%, bonds once again paid a real income and had room to fall (in yield) and rally (in price) if a recession arrives. Paradoxically, the year that broke the 60/40 also restocked the ammunition that makes it work — bonds with a 4.5% starting yield can deliver real ballast in the next growth scare in a way 1.5% bonds never could.

Third, **the correlation is being watched as a live regime indicator, not assumed.** Sophisticated allocators now track the rolling stock-bond correlation explicitly and ask, every quarter, "are we in a growth-shock world or an inflation-shock world?" — because the answer determines whether their bonds are insurance or ballast that has turned to lead.

## Common misconceptions

**"Bonds and stocks always move in opposite directions."** No — this was true of one long regime (2000–2021), not of finance in general. The correlation's *sign* depends on whether the dominant shock is about growth (negative correlation, bonds hedge) or inflation (positive correlation, both fall). For long stretches of the 1970s and again in 2022, stocks and bonds fell together. Treating the negative correlation as a law is exactly the mistake that made 2022 so painful for so many.

**"Bonds are safe, so they can't lose much money."** Bonds carry little *credit* risk (a Treasury will pay you back), but plenty of *interest-rate* risk. A long-duration bond can fall 20–30% in price when rates surge — in 2022 the longest Treasuries fell more than 30%, a stock-sized loss from a "safe" asset. "Safe" means *you get your money back at maturity*, not *the price never drops*. The price risk is the whole subject of [duration](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income).

**"Interest rates only matter for bonds; stocks are about earnings."** Earnings matter, but rates set the *discount rate* that turns those earnings into a price, and a stock is a long-duration claim, so it is *more* rate-sensitive than most bonds, not less. The 2022 tech crash was overwhelmingly a discount-rate story, not an earnings story — the earnings forecasts barely changed; the rate at which they were discounted exploded.

**"Growth stocks are risky because the companies are risky."** Part of the risk is business risk, yes — but a huge part is *duration* risk. A growth stock's value lives far in the future, which makes it extraordinarily sensitive to the discount rate regardless of how good the business is. A flawless company whose cash flows are all 20 years out will still get repriced violently by a rate move. The risk is partly *when* the cash arrives, not just *whether* it arrives.

**"If the 60/40 broke in 2022, diversification is dead."** Diversification didn't fail; an *assumption* about it failed — the assumption that the stock-bond correlation is permanently negative. Diversification is still the only free lunch, but it is *conditional* on correlation, and a robust version diversifies across *regimes* (adding inflation hedges), not just across the two assets that happen to hedge each other in a growth-shock world.

**"The discount rate is just the Fed's policy rate."** The policy rate is the short-term anchor, but the discount rate that prices a 20-year cash flow is built from *long-term* yields plus a risk premium — and those long yields reflect the market's expectations for growth, inflation, and term premium far beyond the Fed's next meeting. The Fed sets the floor; the bond market sets the rest of the curve, which is where most of a stock's discounting actually happens.

## How it shows up in real markets

**The 2022 everything-selloff.** The cleanest modern example of the whole thesis. Inflation hit 9%, the Fed hiked from ~0% to over 4% in a year, the discount rate surged, and *every* long-duration asset repriced down together: the S&P 500 about −18%, the Nasdaq about −33%, the US Aggregate bond index about −13%, the classic 60/40 about −16%. There was nowhere to hide in nominal stocks and bonds because the common driver — the discount rate — moved against both. It was the first time in over four decades that both major asset classes had a sharply negative year simultaneously, and it taught a generation of investors that the negative stock-bond correlation is a regime, not a constant.

**The 1970s stagflation.** The previous positive-correlation regime. Through much of the 1970s and into the early 1980s, high and volatile inflation meant stocks and bonds repeatedly fell together; the 60/40 offered little protection because the dominant shock was inflation, not growth. Investors who lived through it were unsurprised by 2022 — they had seen what happens to a bond hedge when the central bank is fighting inflation instead of rescuing growth. It is the historical proof that the negative correlation of 2000–2021 was the exception, not the rule.

**The 2008 financial crisis — the hedge working perfectly.** The mirror image of 2022. In 2008 the shock was a *growth* and credit collapse, not inflation. Stocks crashed about −37% (S&P 500), and Treasuries *soared* as the Fed slashed rates to zero and investors fled to safety — long Treasuries returned strongly positive on the year. A 60/40 fell far less than stocks alone, and the bond ballast did exactly its job. 2008 and 2022 are the two bookends: the same portfolio, opposite outcomes, decided entirely by whether the shock was growth (2008) or inflation (2022).

**The March 2020 COVID crash.** A compressed, vivid example of the growth-shock hedge. As the pandemic hit and stocks fell roughly −34% in five weeks, the Fed cut to zero and Treasuries rallied hard — the 10-year yield collapsed toward 0.5%, sending bond prices up and cushioning balanced portfolios. It was a textbook flight-to-quality, negative-correlation event, and it happened in real time over a few weeks, making the mechanism unusually easy to see.

**The 2013 "taper tantrum."** A reminder that even within the negative-correlation era, a *rate* shock can briefly turn the correlation positive. When the Fed hinted in mid-2013 that it would slow its bond purchases, long yields jumped, and both bonds *and* rate-sensitive equity sectors (utilities, REITs) sold off together. It was a small-scale preview of 2022: when the shock comes through the *rate* channel rather than the *growth* channel, the hedge inverts even in an otherwise benign era.

**Long-duration tech vs value in 2022–2023.** The intra-equity version of the duration story. As rates surged in 2022, long-duration growth names (the Nasdaq) fell about −33% while shorter-duration value sectors held up far better — and as rates stabilized and fell in late 2023, the same growth names roared back hardest. The rotation between growth and value over those two years tracked the direction of the 10-year yield with uncanny fidelity, because the difference between the two styles is, at root, a difference in *duration*.

**Risk parity and the leverage trap of 2022.** A cautionary footnote to the whole thesis. *Risk-parity* funds take the 60/40 idea further: they hold a lot of bonds — often *more* than stocks in dollar terms — and use leverage (borrowing) to make the low-volatility bond leg pull its weight, betting on the negative correlation to keep total risk down. The strategy thrived for over a decade while the correlation was negative. Then 2022 arrived: bonds and stocks fell together, the correlation flipped positive, and the leveraged bond exposure that was supposed to *diversify* instead *amplified* the loss. Several of the largest risk-parity funds had double-digit negative years, worse than a plain 60/40, precisely because they had bet bigger on the assumption that broke. It is the sharpest real-world lesson that the negative stock-bond correlation is a regime to be respected, not a constant to be leveraged.

## When this matters to you and further reading

If you hold a target-date retirement fund, a balanced fund, or any classic mix of stocks and bonds, you own a bet on the stock-bond correlation whether you realize it or not. Understanding that the bond ballast is *conditional* — that it hedges growth shocks but can amplify inflation shocks — is the difference between being blindsided by a year like 2022 and understanding it as the regime doing exactly what regimes do. The single most useful habit is to ask, before assuming your bonds will save you in the next downturn: *is the thing I'm worried about a recession, or inflation?* The answer tells you which way your two assets will move together.

To go deeper, follow the discount-rate thread outward. The macro source of the rate itself is in [interest rates, the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and [the central bank toolkit](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance). The allocation lens — how much bond to hold, and how to diversify across regimes — is in [the stock-bond correlation, the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) and [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything). The fixed-income mechanics that make bonds rate-sensitive are in [duration](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) and [why bond prices move when rates move](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much). And the inflation-hedging cousin that would have helped in 2022 is in [TIPS and inflation-linked bonds](/blog/trading/fixed-income/tips-and-inflation-linked-bonds-protecting-purchasing-power). Hold them all in your head at once and you see the whole machine: one discount rate, two assets, and a correlation whose sign decides whether they protect you or sink you together.
