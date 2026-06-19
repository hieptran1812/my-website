---
title: "The Stock-Bond Correlation Regime: The Engine Inside 60/40"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The single most important number in portfolio construction is the correlation between stocks and bonds. When it is negative, bonds hedge stocks and 60/40 works; when it flips positive, both fall together and diversification vanishes. The driver of the sign is the inflation regime."
tags: ["macro", "correlation", "stock-bond-correlation", "sixty-forty", "diversification", "inflation", "regime", "portfolio-construction", "discount-rate", "risk-parity", "2022", "bonds"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The correlation between stocks and bonds is the master assumption hidden inside every diversified portfolio. When it is **negative** (bonds rise when stocks fall), 60/40 works and bonds are a genuine hedge; when it flips **positive** (both fall together, as in 2022), diversification evaporates exactly when you need it. The sign is set by the **inflation regime**.
>
> - For two decades (about 2000-2020) the stock-bond correlation was reliably **negative** (around −0.40 to −0.55) — the "great diversifying era" that made 60/40 look like a free lunch. In 2022 it spiked to **+0.60** and both legs fell at once.
> - The driver is inflation: when inflation is below 2% the correlation averages about **−0.45** (bonds hedge); above 4% it averages about **+0.50** (both fall). Growth shocks make stocks and bonds move oppositely; inflation/rate shocks reprice both with the same discount-rate move.
> - The same assets get riskier when the correlation rises. A 60/40 portfolio's volatility climbs from about **8.3% at corr −0.40 to 11.2% at corr +0.60** — a 36% jump in risk with zero change in the holdings.
> - **The one fact to remember:** there is no fixed "stock-bond correlation." There is a disinflation regime where bonds hedge and an inflation regime where they do not — and the entire promise of 60/40 depends on which one you are standing in.

In 2021, a financial advisor could have shown you a chart that looked like the closest thing in finance to a law of nature. For more than two decades, every time stocks had a bad month, US government bonds had a good one. The S&P 500 would drop 5%, and long-dated Treasuries would rise 2-3% to cushion the blow. This relationship was so dependable that an entire industry was built on it: the classic **60/40 portfolio** — 60% stocks, 40% bonds — was sold as the responsible default for anyone who wanted to grow money without lying awake at night. The bonds were not there to make you rich. They were there to be the airbag that deployed when stocks crashed.

Then came 2022. US inflation tore to a 40-year high. The S&P 500 fell about **18%** on the year. The airbag was supposed to deploy. Instead, long-dated Treasuries fell about **31%** — their worst year in modern history. The "safe" half of the portfolio did not cushion the fall; it *added* to it. A plain 60/40 portfolio lost about **16%**, one of its worst years ever, and it lost that money in the most disorienting way possible: both halves went down together. The diversification that decades of charts had promised simply was not there. The thing investors had paid for — the negative correlation — had flipped its sign.

This post is about that sign. It is the engine inside 60/40, the master assumption behind nearly every diversified portfolio on earth, and almost nobody who owns one could tell you what it is or why it changes. We are going to build it from zero: why bonds *can* hedge stocks in the first place, what 60/40 quietly assumes, how the correlation enters the math of portfolio risk, why the sign flips between negative and positive, and how to tell which regime you are in right now. By the end, 2022 will not look like a freak accident. It will look like exactly what the regime predicted.

![Growth shock with bonds hedging stocks versus inflation shock with both falling](/imgs/blogs/the-stock-bond-correlation-regime-1.png)

This is the portfolio-construction chapter of a series about one deceptively simple question: how does macro data move asset prices? We are not re-deriving *why* the Fed hikes when inflation runs hot — the macro-trading series owns that mechanism, and we will cite it. We are not replaying the ten-minute market reaction to a single CPI print — the event-trading series owns that. This post is about the thing in between: the **measurable statistical relationship** between two asset classes, why that relationship has a sign, why the sign flips, and what it does to your money when it does. We build everything from scratch, so if "correlation," "duration," or "discount rate" are fuzzy terms, you are exactly the reader this is written for.

## Foundations: correlation, bonds, and what 60/40 assumes

Before we can talk about why the stock-bond correlation flips, we need four plain-English ideas nailed down: what a correlation actually measures, why a bond's price moves, why a bond can hedge a stock, and what the 60/40 portfolio quietly assumes about all of this. If you have read earlier posts in this series, treat the correlation part as a refresher; if this is your first one, this section gives you everything you need.

### What a correlation measures

A **correlation** is a single number, between −1 and +1, that summarizes how two things tend to move together. Statisticians write it as *r* (or the Greek letter ρ, "rho"). The intuition:

- **r = +1** means the two move in perfect lockstep: when one goes up, the other always goes up by a proportional amount. Think of a thermometer in Celsius and the same thermometer in Fahrenheit — perfectly positively correlated.
- **r = −1** means they move in perfect opposition: when one goes up, the other always goes down. A see-saw is the picture: one end up means the other end down.
- **r = 0** means no reliable relationship: knowing one tells you nothing about the other.

Most real relationships live in between. A correlation of +0.6 means "they usually move the same direction, but not always and not by a fixed amount." A correlation of −0.4 means "they usually move opposite directions, with meaningful exceptions."

The single most important thing to understand about correlation for this entire series is that **it is not a constant**. People talk about "the" stock-bond correlation as though it were a fixed property of the universe, like the speed of light. It is not. It is a *statistic measured over a window of time*, and it changes — sometimes slowly, sometimes violently — as the economic regime changes. The whole sister post [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) is devoted to this point, and the stock-bond correlation is its single best example. A correlation you measure over 2010-2020 can have the *opposite sign* of the one you measure over 2022. Treating it as fixed is the single most expensive mistake in portfolio construction.

### Why a bond's price moves

A **bond** is a loan you make to a government or a company. You hand over, say, \$1,000 today; in return you get a fixed stream of interest payments (called coupons) and your \$1,000 back at the end (at "maturity"). The key word is *fixed*: a bond promises a specific dollar schedule that does not change once issued.

So why does a bond's *price* move around if its payments are fixed? Because the world's interest rate changes, and that changes what a fixed stream is worth. Here is the intuition. Suppose you own a bond paying \$30 a year (a 3% coupon on \$1,000). Now suppose newly issued bonds start paying \$50 a year (5%), because the central bank raised rates. Nobody wants your stingy \$30 bond at \$1,000 anymore — they can get \$50 elsewhere. So the price of your bond *falls* until its yield matches the new 5%. **When interest rates rise, existing bond prices fall. When rates fall, existing bond prices rise.** This inverse relationship is the most important fact in fixed income, and we lean on the [duration and convexity](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity) mechanism rather than re-deriving it here.

The sensitivity of a bond's price to a change in rates is called its **duration**, measured in years. A bond with a duration of 17 years (typical for a long Treasury) loses about 17% of its value if rates rise 1 percentage point, and gains about 17% if rates fall 1 point. This is why **long-dated** bonds are the ones that swing the most — and why long Treasuries fell 31% in 2022 when long-term yields jumped. Duration is leverage on the interest rate: the longer the bond, the bigger the price move per unit of rate change.

### Why a bond can hedge a stock

Now we can see the magic that makes 60/40 work. Consider a classic **growth scare** — the economy looks like it is heading into recession. What happens?

- **Stocks fall.** Stocks are claims on corporate earnings, and a recession means earnings drop. So stock prices decline.
- **The central bank cuts rates** to fight the slowdown — cheaper money to stimulate borrowing and spending. Investors also flee *to* the safety of government bonds, pushing their prices up directly.
- **Bond prices rise**, because (remember) when rates fall, bond prices rise — and they rise *more* the longer the duration.

So in a growth scare, stocks fall and bonds rise. They move in *opposite* directions. That is a **negative correlation**, and it is exactly what you want from a hedge: the asset that does well when your main holding does badly. The bond is an airbag inflated by the same force — falling rates — that deflates the stock. This is not luck; it is mechanism. In a world where the dominant risk is *growth*, bonds are a structural hedge for stocks.

### A stock is a very long-duration bond in disguise

There is one more idea that makes the whole flip click into place, and almost nobody teaches it: **a stock is, mathematically, a kind of bond with an extremely long duration.** A bond pays a stream of cash (coupons) and you discount that stream to get its price. A stock is *also* a claim on a stream of cash — future dividends and earnings, stretching out indefinitely — and you discount *that* stream to get its price. The only real difference is that a bond's cash flows are fixed and finite, while a stock's are growing and effectively infinite.

That difference is exactly what makes a stock behave like a *very long* bond. Because a stock's cash flows extend so far into the future, a large share of its value comes from earnings 10, 20, 30 years out. And the further out a cash flow sits, the more its present value is crushed by a rise in the discount rate — that is the definition of duration. A fast-growing technology company, whose profits are mostly expected decades from now, has the longest "equity duration" of all, which is precisely why the Nasdaq 100 (−32.5% in 2022) fell harder than the broad S&P (−18.1%) when rates jumped. It was the longest-duration asset in the room.

Once you see stocks as long-duration claims, the positive correlation in 2022 stops being a mystery and becomes almost obvious. A long Treasury and a long-duration stock are *both* highly sensitive to the discount rate. When that rate rips higher, the long bond gets clobbered and so does the long-duration stock — for the *same reason*. They are cousins on the duration spectrum, and a big rate move hits the whole family. The negative correlation of 2000-2020 only existed because, in that low-inflation world, the thing moving markets was *earnings* (a stock-only force) rather than *rates* (a shared force). Change what is driving markets, and you change whether stocks and bonds are cousins or opposites.

### What 60/40 assumes

The 60/40 portfolio — 60% stocks for growth, 40% bonds for safety — is the most common single recipe in all of investing. Pension funds, target-date retirement funds, and millions of individual investors hold some version of it. And its entire promise rests on **one assumption that is almost never stated out loud: that the stock-bond correlation is negative.**

If bonds reliably rise when stocks fall, then 40% of your portfolio acts as a shock absorber. Your bad years are less bad. Your ride is smoother. You can hold more risk than you otherwise could, because the bonds catch you. The diversification is real, and it is what mathematicians sometimes call the closest thing to a free lunch in finance — a topic the sister post [correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) and [the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) develop in full.

But if the correlation turns *positive* — if bonds start falling at the same time stocks fall — then your "safe" 40% is no longer a hedge. It is just a second, lower-octane way to lose money. The free lunch is canceled. And the cruel part is *when* it gets canceled: positive correlation tends to show up in exactly the kind of high-inflation, rising-rate environment where you most need protection. The diversification fails precisely when you need it most. Hold that thought — it is the whole story.

## The signature line: two decades of hedge, then the 2022 flip

Now let us look at the actual history. The chart below shows the rolling correlation between US stock returns and long-dated Treasury returns from 1990 to 2025. (A "rolling" correlation is one computed over a moving window — here roughly the trailing two years — so we can watch it change over time.)

![Stock bond rolling correlation from 1990 to 2025 with regime bands](/imgs/blogs/the-stock-bond-correlation-regime-2.png)

Read it left to right and a clear story emerges, with three distinct chapters:

**1990-1998: positive.** In the early 1990s the correlation was positive, around +0.3 to +0.45. This was the tail end of an era where inflation was still a live concern, and (as we will see) inflation makes the correlation positive. Stocks and bonds tended to move together. 60/40 did not diversify the way the textbooks would later claim.

**2000-2020: negative — the great diversifying era.** Around 1998-2000 the correlation crossed below zero and *stayed* there for two decades, often sitting around −0.40 to −0.55. This is the period that taught a whole generation of investors that bonds hedge stocks. Every equity wobble — the dot-com crash, the 2008 financial crisis, the 2018 selloff, the 2020 COVID crash — was met by rising bond prices. The negative correlation was so persistent that people began to treat it as permanent. It was the empirical foundation of "risk parity," of target-date funds, of the entire 60/40 industrial complex.

**2022: the flip to +0.60.** Then inflation arrived, and the correlation did not just weaken — it *inverted*, spiking to about +0.60, its highest reading in decades. Stocks and bonds fell together, hard. The two-decade assumption that bonds hedge stocks was, for one brutal year, exactly backwards.

**2023-2025: the unstable middle.** Since the peak, the correlation has eased — to roughly +0.45, then +0.30, then +0.25 — but crucially it has *stayed positive*. We are not back in the comfortable negative regime. We are in a higher-inflation world where the hedge works less well, and the correlation could break either way depending on what the next shock is.

It is worth sitting with how *strange* the 2000-2020 negative regime is in the sweep of history. If you extend the chart back further, into the 1970s and 1980s, the stock-bond correlation was *positive* — often strongly so. That was the last great inflation, and stocks and bonds fell together repeatedly as the Fed under Paul Volcker drove rates to nearly 20% to break it. The two-decade negative correlation that everyone now treats as normal was, in the long view, the *anomaly* — a special gift of the low-inflation "Great Moderation." Investors who built their entire mental model in 2000-2020 were calibrating to the single most diversification-friendly regime in modern financial history and assuming it would last forever. 2022 was not a new phenomenon; it was the old phenomenon returning after a twenty-year vacation. The honest framing is that the negative correlation is *conditional on low inflation*, and low inflation is not guaranteed.

### How the correlation is actually measured

A quick but important practical note, because the number you read depends entirely on *how* you measure it. The stock-bond correlation is computed on *returns*, not price levels — you take the periodic returns of stocks (say, monthly S&P total returns) and the periodic returns of bonds (monthly long-Treasury total returns) over a chosen window, and run the Pearson correlation on those two return series. Three choices quietly determine the answer, and getting them wrong is how people end up arguing past each other:

- **The window length.** A 90-day window is jumpy and reacts fast; a five-year window is smooth and slow. Our chart uses roughly two years, a common compromise. A short window would have shown the flip to positive *faster* in 2022 but would also whipsaw on every passing scare; a long window lags badly, still half-reflecting 2022 well into 2024. There is no single "correct" window — it is a deliberate trade-off between responsiveness and stability, which the [rolling-correlation window](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) post unpacks in full.
- **The bond chosen.** "Bonds" is not one thing. Long Treasuries (20+ year) have huge duration and swing the most, so they show the most dramatic correlation flip. The aggregate bond index, full of shorter-dated and corporate bonds, is tamer. The stock-bond correlation you cite should always specify *which* bond — the long Treasury is the cleanest expression of the pure duration-vs-equity relationship.
- **The frequency.** Daily, weekly, and monthly returns can give meaningfully different correlations because of how news gets digested over different horizons. Monthly is the standard for the strategic, regime-level question we care about here.

None of this changes the *story* — every reasonable specification shows the negative-then-positive flip — but it explains why two analysts can quote slightly different numbers for "the" stock-bond correlation. They are measuring real but distinct things. The sign and the regime are robust; the second decimal place is not.

#### Worked example: how much "diversification" the sign change destroyed

Consider two identical-looking 60/40 portfolios, each holding the same stocks and the same bonds, differing only in the regime they live in. In the negative-correlation regime, the bonds genuinely offset the stocks: in a year where stocks fall 10%, the bonds might rise 4%, so the blended portfolio falls only about \$3,600 on a \$100,000 stake (0.6 × −10% + 0.4 × +4% = −4.4% → −\$4,400, before the diversification smoothing we will compute below). In the positive-correlation regime, that same 10% stock drop comes with bonds *also* falling, say 6%: now the portfolio loses 0.6 × −10% + 0.4 × −6% = −8.4%, or about **\$8,400** — nearly double the loss, on the very same holdings. **The intuition: the sign of one number doubled the drawdown without anyone changing a single position.**

## The driver: it is the inflation regime

So the correlation flips. Why? What turns the negative-correlation hedge regime into the positive-correlation no-hedge regime? The answer is the single most important idea in this post: **the sign of the stock-bond correlation is set by the inflation regime.**

The chart below conditions the stock-bond correlation on the level of inflation. Instead of plotting it over time, it sorts every period into an inflation bucket and shows the average correlation in each.

![Stock bond correlation by inflation regime bars](/imgs/blogs/the-stock-bond-correlation-regime-3.png)

The pattern is stark and monotonic:

- **Inflation below 2%:** correlation about **−0.45**. Bonds hedge stocks. This is the disinflationary, growth-dominated world of 2000-2020.
- **Inflation 2-3%:** correlation about **−0.30**. Still a hedge, slightly weaker.
- **Inflation 3-4%:** correlation about **+0.05**. The hedge is essentially gone — neutral.
- **Inflation above 4%:** correlation about **+0.50**. Bonds and stocks fall together. This is 2022.

The flip point sits somewhere around 3-4% inflation. Below it, bonds are an airbag; above it, they are a second engine of loss. This is the same flip threshold that the sister post [inflation and stocks, the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips) finds for the inflation-equity relationship — and that is not a coincidence. The same mechanism drives both, as we are about to see.

Why does inflation control the sign? Because it controls *which kind of shock dominates the market's attention*. And different shocks move stocks and bonds in different relative directions.

## The mechanism of the flip: it is all the discount rate

Here is the engine room. To value any asset, you take its future cash flows and "discount" them back to today using an interest rate — the **discount rate**. A higher discount rate makes future money worth less today, so it lowers the asset's price. A lower discount rate does the reverse. Both stocks and bonds are priced off this same discount rate, anchored on government bond yields. That shared anchor is the key to the whole flip.

There are two fundamentally different kinds of macro shock, and they hit the discount rate in opposite ways.

![Discount rate mechanism that sets the stock bond correlation sign](/imgs/blogs/the-stock-bond-correlation-regime-4.png)

### Growth shocks → negative correlation

A **growth shock** is news that the economy is weakening — a recession scare, a demand collapse, a banking wobble. It does two things at once:

1. **It hurts stocks through the *numerator*.** The expected future earnings of companies fall, because a weak economy means weak profits. The cash flows shrink, so stock prices fall.
2. **It helps bonds through the *discount rate*.** A weak economy means the central bank will *cut* rates to stimulate, and investors flee to the safety of government bonds. Yields fall. And falling yields mean rising bond prices.

So a growth shock pushes stocks down (earnings) and bonds up (lower rates). They diverge. **Negative correlation.** The bond is doing its airbag job, inflated by the same rate cut that signals the economy is in trouble. In a world where growth is the thing markets fear most, bonds are a beautiful hedge. This was the world of 2000-2020, when inflation was dead and every scare was a *growth* scare.

### Inflation/rate shocks → positive correlation

An **inflation shock** is different. It is news that prices are rising too fast — a CPI print that comes in hot, an energy spike, a wage-price spiral. It hits *both* assets through the *same channel*: the discount rate.

1. **It hurts bonds directly.** Higher inflation means the central bank will *hike* rates to fight it, and it means the fixed dollars a bond pays are worth less in real terms. Yields rise across the curve. Rising yields mean falling bond prices — and the longer the duration, the bigger the fall.
2. **It hurts stocks through the *same rising discount rate*.** When the discount rate jumps, the present value of corporate earnings — especially earnings far in the future, like a fast-growing tech company's — collapses. This is called **P/E compression** (the price-to-earnings multiple shrinks). Stocks fall not because earnings dropped, but because the *rate used to value those earnings* rose.

So an inflation shock pushes *both* stocks and bonds down, through *one* shared mechanism: a rising discount rate. They move together. **Positive correlation.** The bond is not a hedge anymore, because the thing killing the bond (rising rates) is the very same thing killing the stock. This is the world of 2022.

That is the entire flip in one sentence: **when the dominant shock is growth, stocks and bonds are priced by different things and diverge; when the dominant shock is inflation/rates, they are priced by the same thing and move together.** And inflation determines which kind of shock dominates — at low inflation the Fed has room to cut into any weakness (growth shocks rule), while at high inflation the Fed is forced to keep tightening even into weakness (rate shocks rule). The inflation regime *is* the correlation regime.

### Why the central bank's reaction function is the hidden switch

It is worth being precise about *why* inflation controls which shock dominates, because the real switch is the central bank's behavior. Think about what the Fed does when the economy weakens. In a **low-inflation world**, a weak economy is an unambiguous problem and the Fed responds by cutting rates — there is no inflation tying its hands. Falling rates lift bond prices, so a growth scare automatically produces the bond rally that hedges stocks. The Fed is, in effect, your portfolio's co-pilot: every time stocks get scary, the Fed eases, and your bonds win. The negative correlation is partly *manufactured* by a Fed that is free to ride to the rescue.

Now picture a **high-inflation world**. The economy weakens, but inflation is still 6%. The Fed cannot ride to the rescue — if it cuts rates into high inflation, it risks letting inflation spiral. So it keeps rates high, or even keeps hiking, *even as the economy and stocks struggle*. Now the bond does not rally when stocks fall, because the Fed is not cutting. Worse, if the weakness comes *with* an inflation surprise, the Fed hikes harder and bonds fall alongside stocks. The co-pilot has been grounded by inflation. This is why high inflation is so toxic for the 60/40 investor: it does not just hurt returns, it *disables the hedging mechanism* by taking away the Fed's freedom to cut. The correlation is positive in high inflation precisely because the central bank's reaction function has flipped from "cut into weakness" to "tighten despite weakness." The mechanism behind that policy choice is the subject of the [duration and convexity](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity) post; here, the point is that the Fed's hands being tied is what flips the sign.

#### Worked example: decomposing the 2022 stock and bond moves

Take a long Treasury with a duration of about 17 years. In 2022, the 10-year Treasury yield rose from roughly 1.5% to roughly 4.0% — a jump of about 2.5 percentage points. Duration math says the price change is about −17 × 2.5% = **−42.5%** before coupons and curve effects partly offset it, landing the actual long-Treasury index near **−31%**. Now the same 2.5-point rate rise hits stocks: a rough rule of thumb is that the S&P 500's earnings multiple compresses by 1-1.5 points per percentage-point rise in the discount rate, and the S&P's P/E fell from about 24 to about 19 over 2022 — a ~20% multiple compression that, combined with flat-to-soft earnings, produced the **−18%** price return. **The intuition: one number — the 2.5-point rise in the discount rate — drove both the bond's 31% loss and the stock's 18% loss; that shared driver is why the correlation went positive.**

## The portfolio consequences: why the same assets get riskier

This is not an academic curiosity. The sign of the stock-bond correlation changes how *risky your portfolio is*, even if you never touch a single holding. To see why, we need one piece of portfolio math — the formula for how two assets combine into a portfolio's risk.

### How correlation enters portfolio variance

Risk in finance is usually measured by **volatility** — the standard deviation of returns, which is roughly how much the value bounces around, expressed as an annual percentage. A portfolio with 10% volatility typically lands within about ±10% of its expected return in a given year, and within ±20% in the worse one-in-twenty years. Higher volatility means a wilder ride: bigger gains in good years, but also bigger losses in bad ones, and a greater chance of a drawdown deep enough to make you sell at the bottom. Volatility is not the *only* kind of risk — a bond can have low volatility and still wipe you out via default — but for a diversified stock-and-Treasury portfolio it is the dominant one, and it is the risk that diversification is designed to reduce.

The square of volatility is called **variance**, and variance is the quantity that combines cleanly across assets. You cannot simply add volatilities together, but you *can* add variances together with a correlation-weighted cross term, which is why the portfolio risk formula is written in variance and then square-rooted back into volatility at the end. For a two-asset portfolio (weight *w_s* in stocks, *w_b* in bonds, with stock volatility *σ_s*, bond volatility *σ_b*, and correlation *ρ*), the portfolio variance is:

```
variance = (w_s × sigma_s)^2 + (w_b × sigma_b)^2 + 2 × w_s × w_b × sigma_s × sigma_b × rho
```

Stare at the last term. The first two terms are always positive — they are just the standalone risks of each leg. But the third term, the **cross term**, carries the sign of *ρ*, the correlation. When *ρ* is negative, that whole term is *subtracted* — it *reduces* the portfolio's variance below the sum of its parts. That subtraction *is* the diversification benefit. When *ρ* is positive, the term is *added* — it makes the portfolio riskier. The correlation is literally the knob that turns diversification on or off.

### The variance-vs-correlation curve

The chart below makes this concrete. It holds everything fixed — 60% stocks at 15% volatility, 40% bonds at 8% volatility — and varies only the correlation, plotting the resulting portfolio volatility.

![Sixty forty portfolio volatility versus stock bond correlation](/imgs/blogs/the-stock-bond-correlation-regime-6.png)

The line slopes up the whole way. At a correlation of −0.40 (the old diversifying regime), the 60/40 portfolio's volatility is about **8.3%**. At a correlation of +0.60 (2022), it is about **11.2%**. That is a 36% increase in risk — on *exactly the same portfolio*. You did not buy riskier stocks or longer bonds. The only thing that changed was a number you cannot directly observe and most people never think about. The regime did all the work.

#### Worked example: 60/40 risk on a \$100,000 portfolio, two regimes

Put \$60,000 in stocks (volatility 15%) and \$40,000 in bonds (volatility 8%). Plug into the variance formula.

In the **negative-correlation regime** (ρ = −0.40):
```
variance = (0.60 x 0.15)^2 + (0.40 x 0.08)^2 + 2 x 0.60 x 0.40 x 0.15 x 0.08 x (-0.40)
         = 0.00810 + 0.00102 - 0.00230 = 0.00682
volatility = sqrt(0.00682) = 8.3%
```
On \$100,000, a one-standard-deviation year is about ±\$8,300.

In the **positive-correlation regime** (ρ = +0.60):
```
variance = 0.00810 + 0.00102 + 0.00346 = 0.01258
volatility = sqrt(0.01258) = 11.2%
```
On \$100,000, a one-standard-deviation year is about ±\$11,200.

The variance rose 84% and the volatility rose 36% — and the cross term swung from *subtracting* \$2,300 of risk to *adding* \$3,460 of it. **The intuition: flipping the correlation sign turned the bonds from a risk-reducer into a risk-amplifier, costing you about \$2,900 of extra annual risk on a \$100,000 portfolio without changing a single holding.**

#### Worked example: why "add more bonds" does not buy safety in the wrong regime

A common reflex after a scary year is to "de-risk" by holding more bonds — say shifting from 60/40 to 40/60. Does that help? It depends entirely on the correlation. Keep the same volatilities (stocks 15%, bonds 8%) and compute the 40/60 portfolio's volatility in both regimes.

In the **negative regime** (ρ = −0.40): variance = (0.40 × 0.15)² + (0.60 × 0.08)² + 2 × 0.40 × 0.60 × 0.15 × 0.08 × (−0.40) = 0.0036 + 0.002304 − 0.002304 = 0.0036, so volatility ≈ **6.0%**. Adding bonds genuinely cut risk versus the 8.3% of the 60/40 — the extra bonds are uncorrelated ballast.

In the **positive regime** (ρ = +0.60): variance = 0.0036 + 0.002304 + 0.003456 = 0.00936, so volatility ≈ **9.7%**. The "conservative" 40/60 portfolio is now *more* volatile than the 60/40 was in the good regime (8.3%) — because the extra bonds you added are *correlated* with the stocks. On a \$100,000 stake, you thought you were buying safety and instead you bought \$9,700 of annual risk where you expected \$6,000. **The intuition: bonds only buy you safety when they are uncorrelated with stocks; in the positive-correlation regime, "more bonds" is just "more of the same loss," and de-risking by adding duration backfires.**

## How it shows up in real markets

Numbers on a curve are one thing; what it felt like to live through the regimes is another. Here are the three dated cases that define the modern stock-bond correlation.

### 2000-2020: the great diversifying era

For two decades, the negative correlation was the single most reliable relationship in markets, and it paid off in every crisis:

- **2008 financial crisis:** the S&P 500 fell about 37% on the year while long Treasuries returned roughly +26%. A 60/40 portfolio lost far less than stocks alone — the bonds did exactly their job.
- **2020 COVID crash:** in the March panic the S&P fell over 30% in weeks; Treasuries rallied hard as the Fed slashed rates to zero. The airbag deployed again.

This era is *why* 60/40 became gospel. Inflation was dead — core inflation spent most of those years *below* the Fed's 2% target — so every shock was a growth shock, and growth shocks make bonds hedge stocks. An entire generation of investors, advisors, and risk models learned the negative correlation as a fact of nature. The problem is that it was a fact of a *regime*, not of nature.

The diversification was not just real — it was *generous*. Because the bonds reliably rallied in every equity crisis, a 60/40 investor could ride out the 2008 collapse and the 2020 crash with drawdowns roughly half as deep as a pure stock investor, and still capture most of the recovery. That smoother ride is worth real money: it keeps investors from panic-selling at the bottom, which is where most retail losses actually come from. The negative correlation did not only lower the *statistical* variance on a spreadsheet; it lowered the *behavioral* risk of a human being abandoning the plan at the worst possible moment. When the correlation flipped in 2022, that behavioral cushion vanished too — investors watched both halves of their "balanced" portfolio fall together, lost faith in the strategy, and a wave of them de-risked near the lows, locking in losses just before markets recovered. The sign of one number reshaped not just the math but the psychology.

### 2022: the flip

In 2022 the regime changed, and the chart below shows the carnage. It is the single most important picture in this post for one reason: it shows the two halves of 60/40 falling *together*.

![2022 asset returns showing stocks and bonds both fell together](/imgs/blogs/the-stock-bond-correlation-regime-5.png)

The S&P 500 fell **18.1%**. The Nasdaq 100, full of long-duration tech, fell **32.5%**. And long Treasuries — the supposed safe haven — fell **31.2%**, almost as much as the riskiest equities. Investment-grade corporate bonds fell **15.4%**; even high-yield bonds fell **11.2%**. The plain 60/40 portfolio lost **16.1%**, one of its worst years on record. The only things that rose were commodities (+16.1%, the inflation winner), the US dollar (+8.2%), and gold (roughly flat at −0.3%). 

Everything that was supposed to diversify a stock portfolio — the bonds — went down *with* the stocks, because the shock was inflation, and inflation makes them move together. This is the canonical case study of the sister post [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant): the relationship people trusted most chose the worst possible moment to invert.

#### Worked example: the hedge that wasn't

Imagine you held a \$100,000 portfolio split \$60,000 stocks / \$40,000 long Treasuries going into 2022, *expecting* the bonds to be your airbag. Your reasonable prior, based on 2000-2020, was that in a bad stock year the bonds would *rise* — say +5% in a growth scare — partly offsetting the stock loss. Under that prior, your bond leg would have *gained* about \$2,000.

What actually happened: stocks fell 18.1%, so the stock leg lost \$10,860. And the bonds, instead of gaining \$2,000, fell 31.2% — a loss of \$12,480. The bond leg you bought *for protection* lost *more* than the stock leg it was supposed to protect. The total portfolio lost about \$23,340, or 23.3%. **The intuition: the gap between the \$2,000 your hedge "should" have made and the \$12,480 it actually lost — about \$14,480 on a \$100,000 portfolio — is the precise dollar cost of being wrong about the correlation sign.**

### 2023-2025: the unstable middle

Since 2022 the correlation has eased back toward +0.25 to +0.30 — positive, but no longer extreme. This is an uncomfortable in-between regime. Inflation has come down from its highs but not all the way to the cozy sub-2% world; it has hovered in the 3% zone, right around the flip point. That means the stock-bond correlation is *unstable*: it could slide back negative if inflation keeps falling and the next shock is a growth scare, or it could spike positive again on a fresh inflation surprise. The honest answer for this period is that bonds are a *partial* hedge — better than 2022, worse than 2010 — and you cannot assume the airbag is fully reinflated.

What makes this period genuinely hard is that the *daily* behavior keeps switching. On some days in 2023-2024, a weak jobs report sent both stocks up and yields down — the old negative-correlation, growth-shock reflex, where the market cheers a weaker economy because it means rate cuts. On other days, a hot CPI print sent both stocks and bonds down together — the positive-correlation, inflation-shock reflex. The market itself could not decide which regime it was in, because inflation was sitting right at the threshold where the sign is genuinely ambiguous. For an allocator, the practical takeaway is to *size your reliance on the bond hedge to the inflation reading*: at 3% and falling, lean toward trusting it; at 3% and rising, treat it as switched off until proven otherwise. The correlation near the flip point is not a number you can pin down — it is a coin that the next inflation print will flip.

There is also a subtle measurement trap here that the sister post on [rolling correlation windows](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) develops in depth. A two-year rolling correlation in 2024 still *contains* the 2022 data, so it will read more positive than the current relationship actually is. By the time a long window confirms a regime change, the regime may have already changed again. This is exactly why we said to watch inflation — a leading driver — rather than the trailing correlation itself.

## Common misconceptions

Even sophisticated investors carry wrong mental models about the stock-bond correlation. Here are the five that cost the most, each corrected with a number.

**Myth 1: "Bonds always diversify stocks."** No — bonds diversify stocks only when the correlation is negative, which is only when inflation is low. Over 2000-2020 the correlation averaged about −0.45 (great hedge); in 2022 it was +0.60 (anti-hedge). "Bonds diversify stocks" is a statement about a regime, not a law. The 1970s, another high-inflation decade, also saw stocks and bonds fall together — 2022 was a rerun, not a freak.

**Myth 2: "2022 was a once-in-a-century black swan."** No — it was the *predictable* behavior of the high-inflation regime. The correlation-by-inflation chart shows that above 4% inflation, the correlation has historically been around +0.50. Given that inflation hit 9%, a positive stock-bond correlation was the *base case*, not a tail event. Calling it a black swan is just admitting you were using a model calibrated to the wrong regime.

**Myth 3: "Just rebalance into bonds when they fall — buy low."** This sounds wise and failed badly in 2022. Rebalancing assumes mean reversion: that the cheap asset will bounce. But in a rising-rate regime, bonds kept falling all year as yields kept climbing, so each "buy the dip" purchase bought a further loss. Rebalancing works when the correlation is negative and shocks are temporary; it is a trap when a structural rate repricing is underway. (We work the rebalancing math below.)

**Myth 4: "A higher bond allocation means a safer portfolio."** Only when the correlation is negative. When it is positive, adding bonds adds *correlated* risk. In 2022, a "conservative" 40/60 portfolio (more bonds) was not meaningfully safer than 60/40, because the extra bonds fell too. Safety comes from *uncorrelated* assets, not just from *less-volatile* ones.

**Myth 5: "The correlation is too abstract to matter to a normal investor."** It is the most concrete thing in your portfolio. We computed it: the same 60/40 portfolio is 36% riskier at +0.60 correlation than at −0.40. If you hold a target-date fund, a pension, or any balanced portfolio, this single number is silently setting your risk every single day. The pension that assumed a −0.4 correlation in its risk model and ran into a +0.6 reality in 2022 was carrying nearly double the risk it thought it had, and it found out the hard way.

**Myth 6: "Long bonds are safer than stocks because bonds are 'safe assets.'"** Safety has two meanings, and conflating them is dangerous. Government bonds carry almost no *credit* risk — the US Treasury will pay you back. But they carry enormous *duration* risk — sensitivity to interest rates. A 17-year-duration Treasury can fall 31% in a year, as 2022 proved, which is more than the S&P's 18% drop. "Safe from default" is not the same as "safe from loss." A long bond is a high-volatility bet on interest rates wearing the costume of a safe asset, and in a rate-shock regime the costume comes off.

## How to read it and use it

You cannot trade the stock-bond correlation directly, but you can read which regime you are in and adjust how much you trust your diversification. Here is the practical playbook.

### The signal: watch inflation, not the correlation itself

The correlation is a *lagging* readout — by the time a rolling two-year correlation has clearly flipped, the regime has already changed. The *leading* signal is inflation. Specifically:

- **Is core inflation below ~3% and falling?** You are likely in (or heading toward) the negative-correlation regime. Bonds are a real hedge. 60/40 works roughly as advertised.
- **Is inflation above ~3-4% or rising?** You are in the positive-correlation regime, or at risk of it. Bonds are a weak-to-negative hedge. Do not count on the airbag.

The flip point around 3-4% inflation is the threshold to watch. It is the same threshold the sister posts [inflation and stocks](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips) and the [business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) identify, because they are all manifestations of the same discount-rate mechanism.

### The regime check: what kind of shock is the market pricing?

Ask yourself: *is the market currently afraid of recession, or of inflation?* You can read this off the daily tape. On a day when bad economic news (weak jobs, soft retail sales) makes *both* stocks and bonds *rise* (stocks because "the Fed will cut," bonds because yields fall), you are in a growth-shock-dominated, negative-correlation world. On a day when a hot inflation print makes *both* stocks and bonds *fall* together, you are in an inflation-shock-dominated, positive-correlation world. The market tells you which regime it is in by how it reacts to news.

There is a simple three-question checklist you can run in under a minute. First, **where is core inflation?** Below 3% and falling tilts the odds toward the bond hedge working; above 3-4% or rising tilts toward it failing. Second, **what is the Fed's stance?** If the Fed has room to cut and is signaling cuts, growth shocks will dominate and bonds will hedge; if the Fed is pinned by inflation and signaling "higher for longer," rate shocks will dominate and bonds will not. Third, **how did the market react to the last surprise?** If the last hot inflation print sent stocks and bonds down together, the positive-correlation regime is live, full stop. When all three line up — low inflation, easing Fed, growth-driven reactions — you can lean on 60/40 with confidence. When they line up the other way, treat your bond allocation as duration risk rather than as a hedge, and source your diversification elsewhere.

A useful habit is to *stress-test your portfolio under the opposite correlation* before you need to. Take your current holdings and ask: what would a year look like if stocks fell 18% *and* bonds fell 31%, the way they did in 2022? If that scenario would do unacceptable damage, you are over-reliant on the negative correlation, and you should add regime-diverse hedges *now*, while it is cheap and calm, rather than after the flip. The whole point of understanding the regime is to act before the tape forces you to.

### What diversifies when bonds don't

The deepest practical lesson: if bonds stop hedging in the inflation regime, you need *other* diversifiers. The matrix below maps what actually worked in each regime.

![What diversifies stocks by regime growth shock versus inflation shock](/imgs/blogs/the-stock-bond-correlation-regime-7.png)

- **Commodities** are the natural inflation hedge — they *are* the inflation. When energy and food prices spike, that *is* the inflation that is hurting stocks and bonds, so the same force that sinks your portfolio's two main legs lifts your commodity exposure. In 2022 the broad commodity index (BCOM) rose about 16% while stocks and bonds fell double digits. The catch is that commodities are a *regime-specific* tool, not an all-weather one: in a growth shock, demand for raw materials collapses and commodities fall *with* stocks. So commodities hedge the inflation regime and add risk in the growth regime — the mirror image of bonds. Holding both means you always own *something* that is hedging, which is the core insight of risk parity.
- **Cash and short-dated Treasury bills** have almost no duration, so their prices barely move when rates rise — a one-year bill loses well under 1% even on a big rate jump, versus the 31% a long Treasury can lose. In an inflation regime, cash is a genuine safe harbor: boring, but it does not crash. And there is a second gift. As yields rose in 2022-2023, T-bills started paying 4-5% for the first time in over a decade, finally rewarding the patient with real income. The cost of holding cash is that in a growth shock, when the Fed slashes rates, your cash yield evaporates and you miss the bond rally — so cash, too, is regime-specific, strongest exactly where bonds are weakest.
- **Trend-following and managed futures** strategies are the cleanest example of an asset that *thrives* on the very thing that breaks 60/40. These strategies systematically go *short* falling markets and *long* rising ones, across stocks, bonds, currencies, and commodities. In 2022, the positive stock-bond correlation handed them a gift: when stocks and bonds fall *together* in a sustained trend, a strategy that is short both rides one big, clean, one-directional move. Many trend funds had their best year in a decade in 2022 — they were *long* the same inflation/rate shock that was crushing everyone else's diversified portfolio. The price of trend is that it tends to lose money in choppy, range-bound, mean-reverting markets, so it is a hedge against *sustained* regime shifts, not against quick V-shaped scares.

The unifying lesson is that *no single asset hedges every regime*. Bonds hedge growth shocks; commodities and cash and trend hedge inflation shocks. The mistake of the 60/40 era was to let one hedge — bonds — carry the entire diversification load, on the unstated assumption that all shocks would be growth shocks. A portfolio that genuinely survives every regime owns a *basket* of hedges, each calibrated to a different kind of shock, so that whatever the next crisis is, *some* part of the portfolio is the airbag. That is the entire philosophy of the all-weather and risk-parity approaches, and it is the constructive answer to the question this post poses.

This is the logic behind **risk parity and all-weather** portfolios, which deliberately spread across assets that hedge *different* regimes rather than betting everything on the stock-bond hedge. The sister post [all-weather and risk parity](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) develops this in full; the one-line version is: do not let your entire diversification depend on a single correlation that you know can flip.

#### Worked example: the rebalancing trap in 2022

Start 2022 with the \$100,000 portfolio at \$60,000 stocks / \$40,000 long Treasuries. A disciplined rebalancer rebalances back to 60/40 whenever the weights drift. By year-end, stocks fell 18.1% to \$49,140 and long Treasuries fell 31.2% to \$27,520, a total of \$76,660. Because bonds fell *more* than stocks, the stock weight drifted *up* to 64.1% and the bond weight *down* to 35.9%. To rebalance back to 60/40, the rule says: sell \$3,144 of stocks and buy \$3,144 more bonds — buy *more* of the asset that just lost you 31%. And because yields kept rising into 2023, that extra bond purchase kept losing money. **The intuition: rebalancing is a bet that the loser will bounce, and in a structural rate-repricing regime it instead doubles you down into the falling asset — the rebalancing P&L was negative precisely because the correlation was positive and the trend was persistent.**

### What invalidates the read

Be honest about the limits. The inflation-regime model says positive correlation in high inflation — but the *timing* is loose, and the correlation can stay positive for a while even as inflation falls (we are living that now, in the 3% zone). The model also assumes the central bank reacts to inflation by hiking; if a central bank chose to tolerate high inflation rather than fight it, the mechanism would weaken. And the relationship is about *US* stocks and *US* Treasuries — the master cross-asset correlations, like the dollar's grip on everything, are mapped in [the macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) and govern how this transmits abroad. The single cleanest invalidation: if you see stocks and bonds *both rising* on bad growth news, the negative-correlation regime is back, regardless of what your two-year rolling window still says.

## The one number that runs your portfolio

Step back and look at what we have built. The stock-bond correlation is not a footnote. It is the master assumption inside 60/40, inside target-date funds, inside pensions, inside risk parity — inside nearly every diversified portfolio on the planet. Its sign decides whether your bonds are an airbag or a second anchor. And that sign is not a constant: it was positive in the inflationary early 1990s, negative through the great moderation of 2000-2020, and positive again in the 2022 inflation shock.

The driver is the inflation regime, working through one shared channel: the discount rate. In a low-inflation world, shocks are growth shocks, and growth shocks push stocks and bonds in opposite directions — the bond hedges. In a high-inflation world, shocks are rate shocks, and a rate shock pushes stocks and bonds the *same* direction through the same discount-rate move — the bond stops hedging. The flip point sits around 3-4% inflation, the same threshold that governs the inflation-equity relationship, because it is the same mechanism.

The practical discipline is humbling: do not trust your diversification blindly. Read the inflation regime. Watch whether the market reacts to bad news by buying bonds or selling them. And when you are in the high-inflation, positive-correlation regime, reach for the diversifiers that *do* work there — commodities, cash, trend — instead of clinging to a stock-bond hedge that the regime has switched off. The investor who survives the next 2022 is not the one with the most bonds. It is the one who knew the correlation was a regime, not a law.

If you remember nothing else, remember this: every diversified portfolio is, at its heart, a bet on a single correlation, and that correlation is a function of inflation. The 60/40 you or your pension fund holds is quietly long the assumption that bonds will rise when stocks fall — an assumption that is true in a low-inflation world and false in a high-inflation one. You do not have to predict the future to manage this. You only have to keep asking, regime by regime, the one question this whole post is built around: *if stocks fall tomorrow, will my bonds be the airbag or the second car in the crash?* The answer changes with the inflation regime, and your job as an investor is to keep checking which answer is true right now — and to never let yourself be surprised, as so many were in 2022, when the airbag turns out to have been disconnected all along.

## Further reading and cross-links

Within this series:
- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the master idea that every correlation, including this one, changes sign across regimes; 2022 is its headline case.
- [Inflation and stocks, the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips) — the same flip threshold for the inflation-equity link; the U-shape behind the discount-rate story.
- [The business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) — how asset correlations rotate through the four phases of the cycle.
- [The macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — the full map of which macro driver moves which asset, including yields' grip on everything.

In the cross-asset series (portfolio construction):
- [Stock-bond correlation, the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) — the allocator's-lens companion to this post.
- [All-weather and risk parity, owning every regime](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) — building a portfolio that does not depend on a single correlation.
- [Correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) — why a negative correlation is the closest thing to free money in finance.

For the underlying mechanism:
- [How monetary policy moves bonds: duration and convexity](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity) — why a rate move of 2.5 points can take 31% off a long Treasury.
