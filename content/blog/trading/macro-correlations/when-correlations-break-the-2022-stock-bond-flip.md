---
title: "When Correlations Break: The 2022 Stock-Bond Flip"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "2022 is the definitive case study of a correlation breaking: the negative stock-bond correlation that made 60/40 work for two decades flipped sharply positive, both legs fell together, and the standard balanced portfolio had its worst year in a century. Here is the timeline, the mechanism, the damage, and the lesson."
tags: ["macro", "correlation", "stock-bond-correlation", "sixty-forty", "diversification", "inflation", "regime", "2022", "discount-rate", "rate-shock", "portfolio-construction", "commodities"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — In 2022 the stock-bond correlation, negative and "reliable" for two decades, flipped sharply positive, and the standard 60/40 portfolio had its worst year in roughly a century because both legs fell together. The trigger was a regime change from disinflation to an inflation and rate shock: one rising discount rate crushed equity multiples and bond prices at the same time, collapsing two assets into a single rate bet.
>
> - The rolling stock-bond correlation ran around **−0.40 to −0.55** through the 2000-2020 "diversifying era," then spiked to about **+0.60** in 2022 — the cleanest sign-flip in modern markets.
> - The damage: the S&P 500 fell about **18.1%**, long Treasuries fell about **31.2%** (their worst year ever), and a plain 60/40 portfolio lost about **16.1%** — both halves down together, the exact failure 60/40 is supposed to prevent.
> - The flip was the regime, not a freak: stock-bond correlation runs about **−0.45 when inflation is below 2%** and about **+0.50 when inflation is above 4%**. With CPI at a 9.1% peak and the fastest hiking cycle in 40 years, 2022 sat squarely in the "both fall" bucket.
> - **The one fact to remember:** diversification that depends on the Fed *cutting* during a stock selloff fails when the Fed is *hiking* into an inflation shock. Bonds hedge a growth scare; nothing in a classic 60/40 hedges an inflation scare.

In January 2022, an investor with a balanced 60/40 portfolio held what the entire financial-advice industry called the responsible default: 60% in stocks for growth, 40% in bonds for safety. The pitch was simple and, on the evidence of two decades, almost unimpeachable. When stocks fell, bonds rose and cushioned the blow. The bonds were not there to make you rich; they were the airbag that deployed in a crash. Charts going back to the late 1990s showed it working over and over: a bad month for the S&P 500 was a good month for Treasuries, like clockwork.

By October of that year, the same investor had watched the airbag not just fail to deploy but actively make the crash worse. The S&P 500 was down about 18% on the year. Long-dated Treasuries — the "safe" half — were down about 31%, their worst year in modern history. A plain 60/40 portfolio had lost about 16%, one of its worst calendar years on record, and it had lost that money in the single most disorienting way possible: both halves fell at the same time. The diversification that decades of charts had promised was simply not there.

This post is the definitive case study of that break. It is the canonical example for this entire series, because it is the moment a correlation that everyone treated as a constant revealed itself as a regime. We are going to dissect the flip from every angle: a recap of why the stock-bond correlation is usually negative, exactly what changed in 2022, the mechanism that made one rising rate sink both legs, the timeline month by month, the damage to a real portfolio, what *would* have diversified, and the forward lesson for anyone who owns a balanced portfolio. By the end, 2022 will not look like an accident. It will look like exactly what the regime predicted — and a preview of every future inflation shock.

![Diversifying regime with bonds hedging stocks versus the 2022 flip with both legs red](/imgs/blogs/when-correlations-break-the-2022-stock-bond-flip-1.png)

This is the "when it breaks" chapter of a series about one deceptively simple question: how does macro data move asset prices? We are not re-deriving *why* the Fed hikes when inflation runs hot — the [macro-trading series](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) owns that mechanism, and we cite it. We are not replaying the ten-minute market reaction to a single CPI print — the event-trading series owns that. This post is about the thing in between: the **measurable statistical relationship** between stocks and bonds, why it has a sign, why that sign flipped in 2022, and what it did to your money. We build everything from scratch, so if "correlation," "duration," or "discount rate" are fuzzy terms, you are exactly the reader this is written for.

## Foundations: correlation, the discount rate, and what 60/40 assumes

Before we can dissect the break, we need four plain-English ideas nailed down: what a correlation actually measures, why a bond's price moves at all, why a bond can hedge a stock in the first place, and what the 60/40 portfolio quietly assumes about all of it. If you have read [the stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) post, treat this as a fast refresher; if this is your first one in the series, this section gives you everything you need.

### What a correlation measures

Start with an everyday picture before any math. Suppose you are watching two people on a see-saw at a playground. When one goes up, the other goes down — always, perfectly, because they are rigidly connected. Now suppose instead you are watching two strangers walking through a park: their paths have nothing to do with each other, and where one goes tells you nothing about the other. Finally, picture two friends walking together, mostly side by side but occasionally drifting apart and coming back. Those three pictures — rigid opposition, total independence, and loose togetherness — are the three things a correlation measures, and finance lives mostly in the third.

A **correlation** is a single number, between −1 and +1, that summarizes how two things tend to move together. Statisticians write it as *r* (or the Greek letter ρ, "rho"). The intuition is short:

- **r = +1** means perfect lockstep: when one goes up, the other always goes up by a proportional amount. A thermometer in Celsius and the same thermometer in Fahrenheit are perfectly positively correlated.
- **r = −1** means perfect opposition: when one goes up, the other always goes down. A see-saw is the picture — one end up means the other end down.
- **r = 0** means no reliable relationship: knowing one tells you nothing about the other.

Most real relationships live in between. A correlation of +0.6 means "they usually move the same direction, but not always and not by a fixed amount." A correlation of −0.4 means "they usually move opposite directions, with meaningful exceptions." There is one more idea worth carrying forward: correlation measures only *direction and tightness*, not *magnitude*. A separate number, the **beta**, tells you how big one move is per unit of the other — how many percent stocks fall per percentage point of rate rise, for instance. Correlation and beta usually move together, but the post focuses on the correlation because it is the *sign* that flipped in 2022, and the sign is what decides whether your hedge helps or hurts. The series post [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) separates these formally.

The single most important thing to understand for this entire series is that **a correlation is not a constant**. People talk about "the" stock-bond correlation as if it were a fixed property of the universe, like the speed of light. It is not. It is a *statistic measured over a window of time*, and it changes — sometimes slowly, sometimes violently — as the economic regime changes. The sister post [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) is devoted to exactly this idea, and 2022 is its single best example: a correlation measured over 2010-2020 had the *opposite sign* of the one measured over 2022. Treating it as fixed is the single most expensive mistake in portfolio construction.

### Why a bond's price moves

A **bond** is a loan you make to a government or a company. You hand over, say, \$1,000 today; in return you receive a fixed stream of interest payments (called coupons) and your \$1,000 back at the end (at "maturity"). The key word is *fixed*: a bond promises a specific dollar schedule that does not change once it is issued.

So why does a bond's *price* move around if its payments are fixed? Because the world's interest rate changes, and that changes what a fixed stream is worth. Here is the intuition. Suppose you own a bond paying \$30 a year (a 3% coupon on \$1,000). Now suppose newly issued bonds start paying \$50 a year (5%) because the central bank raised rates. Nobody wants your stingy \$30 bond at \$1,000 anymore — they can get \$50 elsewhere. So the price of your bond *falls* until its effective yield matches the new 5%. **When interest rates rise, existing bond prices fall; when rates fall, existing bond prices rise.** This inverse relationship is the most important fact in fixed income.

The sensitivity of a bond's price to a change in rates is called its **duration**, measured in years. A bond with a duration of 17 years (typical for a long Treasury) loses about 17% of its value if rates rise 1 percentage point, and gains about 17% if rates fall 1 point. This is why **long-dated** bonds swing the most — and it is the entire explanation for why long Treasuries fell about 31% in 2022 when long-term yields jumped. Duration is leverage on the interest rate: the longer the bond, the bigger the price move per unit of rate change. We lean on the full [duration and convexity](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) mechanism rather than re-deriving it here.

### The discount rate: the variable that prices everything

Here is the concept that ties the whole post together. The value of any financial asset is the present value of the cash it will throw off in the future. A dollar arriving in ten years is worth less than a dollar today, and the rate you use to shrink future dollars down to today's value is called the **discount rate**. The discount rate is, at its core, the interest rate on safe money plus a premium for risk.

The crucial point: **the same discount rate prices both stocks and bonds.** A bond's cash flows are its coupons and principal; a stock's cash flows are its future earnings and dividends, stretching far into the future. When the discount rate rises, *every* future cash flow is worth less today — so both bond prices and stock prices fall. When it falls, both rise. The reason stocks and bonds do not move in perfect lockstep all the time is that two *other* things also move stock and bond cash flows: the level of corporate earnings (which depends on growth) and the safety premium investors demand. Hold those steady, change only the discount rate, and stocks and bonds become the same trade. That sentence is the seed of 2022.

### Why a bond can usually hedge a stock

Now we can see the magic that made 60/40 work for twenty years. Consider a classic **growth scare** — the economy looks like it is sliding into recession. What happens?

- **Stocks fall.** Stocks are claims on corporate earnings, and a recession means earnings drop. Stock prices decline.
- **The Fed cuts rates.** Faced with a weakening economy, the central bank lowers interest rates to stimulate borrowing and spending.
- **Bonds rise.** Lower rates mean the discount rate falls, and because bond prices move inversely to rates, existing bonds gain value. Investors also flee to the safety of Treasuries, bidding their prices up further.

So in a growth scare, stocks fall *and* bonds rise. They move in opposite directions — a **negative correlation**. The 40% in bonds gains while the 60% in stocks loses, and the portfolio's total swing is smaller than either piece alone. That is the entire promise of 60/40, and it is real — *in a growth-driven world.*

### What 60/40 quietly assumes

A **60/40 portfolio** holds 60% in stocks and 40% in bonds. Its appeal rests on a single hidden assumption: that the stock-bond correlation is, and will stay, *negative*. Diversification is the only free lunch in finance — combining two assets that do not move together gives you a smoother ride than either alone — and 60/40 is the most popular way ordinary investors buy that lunch. The [cross-asset post on the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) works through the portfolio math in detail.

But the negative correlation is not a law of nature. It is a feature of a *growth-shock-dominated* world, where the biggest threat to markets is recession and the Fed's reflex is to cut. The 2000-2020 era was exactly such a world: low, stable inflation meant every market scare was a growth scare, the Fed always cut, and bonds always rallied to the rescue. Investors came to believe the hedge was permanent. It was conditional all along — and 2022 was the condition flipping.

## The flip: what 2022 actually did to the correlation

Let us look at the number itself. The chart below shows the rolling stock-bond correlation from 1990 to 2025 — the correlation of S&P 500 total returns against long Treasury returns, measured over a roughly 24-month window. The green band is where the correlation is negative (bonds hedge stocks, 60/40 works); the red band is where it is positive (both fall together, 60/40 breaks).

![Rolling stock-bond correlation from 1990 to 2025 spiking to positive 0.60 in 2022](/imgs/blogs/when-correlations-break-the-2022-stock-bond-flip-2.png)

Read it left to right. In the 1990s the correlation was modestly positive — a holdover from the higher-inflation 1970s and 80s. Around 1998-2000 it turned negative and *stayed* negative for two decades, settling near −0.45 to −0.55. This is the "great diversifying era," the period that taught a whole generation of investors that bonds reliably hedge stocks. Then look at 2022: the line rips straight up to about +0.60. Not a wobble, not noise — a clean, violent sign-flip to the most positive reading in over thirty years. It eased to about +0.45 in 2023 and +0.30 by 2024-25, but it did not return to the comfortable negative territory of the diversifying era.

That single spike is the subject of this post. It is the cleanest example in modern markets of a correlation that everyone treated as a fixed parameter revealing itself as a regime-dependent variable. And critically, it did its damage at exactly the worst time — when stocks were already falling, the bonds that were supposed to cushion the fall fell with them.

#### Worked example: how much "diversification" the flip erased

Diversification benefit shows up in a portfolio's volatility — how much its value swings. The variance of a 60/40 portfolio (with weights 0.6 stocks, 0.4 bonds) is:

```
var = (0.6 * vol_s)^2 + (0.4 * vol_b)^2
      + 2 * 0.6 * 0.4 * corr * vol_s * vol_b
```

Take stock volatility of 16% and bond volatility of 12% (roughly long-Treasury levels). In the diversifying era, corr = −0.40:

- First term: (0.6 × 16)^2 = 9.6^2 = 92.2
- Second term: (0.4 × 12)^2 = 4.8^2 = 23.0
- Cross term: 2 × 0.6 × 0.4 × (−0.40) × 16 × 12 = −35.4
- Variance = 92.2 + 23.0 − 35.4 = 79.8 → volatility = √79.8 ≈ **8.9%**

Now flip the correlation to +0.60, as in 2022, holding everything else fixed:

- Cross term: 2 × 0.6 × 0.4 × (+0.60) × 16 × 12 = +53.1
- Variance = 92.2 + 23.0 + 53.1 = 168.3 → volatility = √168.3 ≈ **13.0%**

The same holdings — not one share bought or sold — went from about 8.9% volatility to about 13.0%, a **46% jump in risk**, purely because the correlation flipped sign. On a \$1,000,000 portfolio, a one-standard-deviation bad year went from roughly a \$89,000 swing to roughly a \$130,000 swing. **The flip did not just lose money; it silently made the entire portfolio far riskier than its owner believed, with no change in what was held.**

## The damage: the worst year for 60/40 in a century

Now the carnage, asset by asset. The chart below shows 2022 total returns across the major asset classes. The two legs of a 60/40 portfolio — stocks and long Treasuries — are both deep in the red, and they fell *together*.

![2022 total returns by asset showing stocks and bonds both falling while commodities and dollar rose](/imgs/blogs/when-correlations-break-the-2022-stock-bond-flip-3.png)

The numbers are stark. The S&P 500 returned about −18.1%. The Nasdaq 100, packed with long-duration tech whose value sits far in the future, returned about −32.5% — the discount-rate hit lands hardest on the longest-dated cash flows, whether those cash flows belong to a 20-year bond or a high-growth software company. Long Treasuries (20-year-plus) returned about −31.2%, their worst calendar year in the history of the index. Even investment-grade corporate bonds fell about 15.4% and high-yield bonds about 11.2%. The plain 60/40 portfolio lost about 16.1%.

To put that 16% in perspective: a balanced 60/40 portfolio of US stocks and bonds has had only a handful of years worse than 2022 in the last hundred-plus years, and most of those were in the Great Depression. What made 2022 uniquely painful was not the *size* of the loss but its *character*. In a normal bad year — 2008, say — stocks crater but bonds rally hard, so 60/40 loses far less than stocks alone. In 2008, the S&P fell about 37% while long Treasuries *gained* roughly 26%, so a 60/40 portfolio lost only about 13% — far less than the stock leg, exactly as the hedge promised. In 2022, the bonds fell *more* than the stocks. The hedge did not just fail to help; it was the bigger source of the loss. That contrast — bonds up 26% in 2008 versus down 31% in 2022, against similar-sized equity drawdowns — is the single cleanest illustration of what the correlation sign does to a portfolio. Same two assets, same balanced split, opposite outcomes, and the only thing that changed was whether the dominant shock was a growth scare or an inflation scare.

#### Worked example: the 60/40 drawdown on a \$100,000 portfolio

Start the year with \$100,000 in a classic 60/40 split: \$60,000 in an S&P 500 fund and \$40,000 in a long-Treasury fund. Apply the 2022 returns.

- Stock leg: \$60,000 × (1 − 0.181) = \$60,000 × 0.819 = **\$49,140** (a \$10,860 loss).
- Bond leg: \$40,000 × (1 − 0.312) = \$40,000 × 0.688 = **\$27,520** (a \$12,480 loss).
- Total: \$49,140 + \$27,520 = **\$76,660** — a loss of \$23,340, or about **−23.3%** for this long-duration version.

Notice the bond leg lost *more dollars* (\$12,480) than the stock leg (\$10,860), even though it was the smaller and "safer" allocation. The standard 60/40 benchmark, which uses an intermediate-duration aggregate bond index rather than long Treasuries, lost about 16.1% — so the same \$100,000 became roughly **\$83,900**. **Either way, the half of the portfolio bought specifically to soften a stock crash instead deepened it — the precise opposite of its job description.**

### The rebalancing trap

There is a second, subtler way the flip hurt disciplined investors, and it is worth its own treatment because it punishes exactly the behavior textbooks praise. **Rebalancing** is the practice of periodically selling whatever has gone up and buying whatever has gone down to restore your target weights. In a normal year it is a quiet source of extra return: you systematically sell high and buy low. When stocks fall and bonds rise, rebalancing means selling some of the appreciated bonds and buying the cheap stocks — and historically, that has paid off as both assets revert toward their long-run paths.

In 2022, rebalancing was a trap. Because both legs fell together, there was no appreciated asset to harvest. Worse, an investor who rebalanced *into* the falling assets — selling whichever had fallen less to buy whichever had fallen more — kept pouring money into a sinking market for the entire year. The bonds did not bounce back the way they always had; they kept falling as the Fed kept hiking. The rebalancing discipline that had added value for two decades, by buying the dip in whichever asset had cheapened, instead bought dip after dip in a market with no bottom until October. The mechanical rule assumed mean-reversion, and a regime change is precisely the situation where mean-reversion fails: the old mean is gone.

#### Worked example: the rebalancing trap in numbers

Take an investor who rebalances to 60/40 at mid-year. They started January with \$100,000 (\$60,000 stocks, \$40,000 long bonds). By June, with the first-half rout, say stocks were down 20% and long bonds down 22%:

- Stocks: \$60,000 × 0.80 = \$48,000
- Bonds: \$40,000 × 0.78 = \$31,200
- Total: \$79,200; current weights ≈ 60.6% stocks, 39.4% bonds.

The portfolio is nearly on target, so a strict rebalancer makes only a tiny trade — but a *contrarian* rebalancer who tops up the worse performer would buy more long bonds, the asset that had fallen most. Now apply the second-half path, where long bonds fell *further* (another ~12%) while stocks roughly flattened. Every extra dollar moved into bonds at mid-year lost another 12%. Suppose they shifted \$5,000 from stocks into bonds at the June low:

- That \$5,000 in bonds became \$5,000 × 0.88 = \$4,400 by year-end — a \$600 loss on the rebalance alone.
- Had they left it in flat stocks, it would still be \$5,000.

The rebalance *cost* them \$600 on a \$5,000 trade, about 12%, because it added to a position that kept falling. **In a regime flip, rebalancing toward the "cheap" asset is not buying low — it is averaging down into a trend, and the discipline that wins in a mean-reverting world bleeds you in a regime-change year.**

## The mechanism: one discount rate, two assets, the same loss

Why did this happen? The answer is the discount-rate concept from the Foundations section, applied to a specific shock. The chart below walks the causal chain from the inflation shock to the correlation flip.

![Flow showing one rising discount rate compressing equity multiples and bond prices into one rate bet](/imgs/blogs/when-correlations-break-the-2022-stock-bond-flip-6.png)

Trace it node by node. An **inflation shock** hit: US CPI tore from under 2% to a peak of about 9.1% by mid-2022, the highest in roughly forty years. To fight it, the **Fed hiked hard** — from a target of 0.25% in March 2022 to 4.50% by December, the fastest hiking cycle in four decades. That sent the **discount rate up** across the curve: the 10-year Treasury yield rose from about 1.5% at the start of the year to over 4% by autumn.

Now the single discount rate forks into both assets. On the equity side, a higher discount rate means future earnings are worth less today, so **price-to-earnings multiples compress** — investors pay fewer dollars per dollar of earnings. Stocks fell, with the longest-duration names (tech, the Nasdaq) hit hardest. On the bond side, a higher yield means **duration bites**: a long Treasury with ~17-year duration loses roughly 17% per percentage point of yield increase, and yields rose more than two points. Bonds fell, hard.

The two falling assets converge on the bottom node: the **correlation flips** from about −0.40 to +0.60. This is the heart of the matter. In a growth shock, the discount rate moves *down* (Fed cuts), which lifts bonds while stocks fall — opposite directions, negative correlation. In an inflation shock, the discount rate moves *up* (Fed hikes), which sinks both — same direction, positive correlation. **The sign of the stock-bond correlation is, to a first approximation, the sign of what is happening to the discount rate, and that is set by whether the dominant shock is growth or inflation.**

This is why 2022 was not a freak. The diversification 60/40 sells depends on the Fed *cutting* into a stock selloff. When the Fed is *hiking* into an inflation shock, that whole machine runs in reverse. The bonds are no longer a hedge against the stocks; they are a second, more leveraged bet on the same rising rate. Two assets collapse into one trade. The [cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) shows how the same rate move propagates across every market, not just these two.

#### Worked example: decomposing the discount-rate hit on stocks

A rough but instructive model: a stock's fair P/E multiple is inversely related to the discount rate. Suppose the market's discount rate (a safe yield plus an equity risk premium) was about 7% at the start of 2022, supporting a P/E around 21 (since 1 / 0.07 ≈ 14, plus a growth adjustment that lifts it). The 10-year yield rose by roughly 2.5 percentage points over the year.

Push the discount rate up by that 2.5 points to about 9.5%. Holding earnings flat, the fair multiple falls roughly in proportion to the inverse of the rate: 7 / 9.5 ≈ 0.74, so the multiple compresses by about 26%. A 26% multiple compression on flat earnings is a 26% price fall from rates alone. In reality earnings held up somewhat, cushioning the S&P to about −18%, while the Nasdaq's longer-duration earnings took closer to the full hit at about −32.5%. **The same arithmetic that says a long bond falls when its yield rises says a long-duration stock falls when its discount rate rises — stocks and bonds were running the identical math, which is exactly why they fell together.**

The duration idea deserves one more pass, because it explains the cruelest detail of 2022: the *safest-looking* asset fell hardest. A long Treasury and a high-growth tech stock are, mathematically, the two most duration-heavy things you can own. The bond's cash flows stretch out 20-plus years; the tech stock's earnings are mostly expected far in the future. Both have most of their present value sitting in distant cash flows, and distant cash flows are the most sensitive to the discount rate, because the discounting compounds over more years. So when the discount rate jumped, the 20-year Treasury (−31%) and the Nasdaq (−32.5%) — the two assets that look least alike on the surface — moved almost identically, while the shorter-duration value stocks and the energy sector held up far better. The lesson hidden inside the wreckage is that **duration, not asset class, is the real risk factor in a rate shock.** A "diversified" portfolio of long bonds and growth stocks is, in duration terms, doubling down on the same bet.

This is also why the correlation flip is *asymmetric* in its danger. In a growth shock, the discount rate falls and bonds rally, but stocks fall on the earnings hit — the two effects partly cancel, and the correlation is negative but not violently so. In an inflation shock, the discount rate rises and there is no offsetting force: higher rates hurt bond prices directly *and* compress equity multiples, with the earnings backdrop often neutral or worse (margins squeezed by rising input costs). Both legs feel the full, undiluted force of the same rising rate, so the positive correlation in an inflation regime tends to be *stronger* than the negative correlation in a growth regime. The data bears this out: the +0.50 average above 4% inflation is a larger magnitude than the −0.30 to −0.45 in the low-inflation buckets. The inflation regime does not just remove the hedge; it builds a stronger, more dangerous link in its place.

## The trigger: a 9.1% CPI and the fastest hiking cycle in 40 years

Step back to the macro picture that set everything in motion. The flip needed a specific trigger: a shock big enough to force the Fed to hike fast and far. That shock was inflation. The chart below overlays CPI inflation (left axis, red) with the Fed funds target rate (right axis, blue) through the 2020-2024 window.

![CPI inflation peaking at 9.1 percent with the Fed funds rate stepping from 0.25 to 4.50 percent](/imgs/blogs/when-correlations-break-the-2022-stock-bond-flip-4.png)

The story is in the two lines. CPI was running near or below the Fed's 2% target through 2020. Then it took off — past 5% in mid-2021, past 7% by January 2022, to a peak of about 9.1% in June 2022, a level not seen since the early 1980s. For a while the Fed called it "transitory" and kept rates pinned at the zero bound. When it became clear the inflation was not going away, the Fed pivoted to the most aggressive tightening in modern history: the funds rate stepped from 0.25% in March 2022 to 4.50% by December, including an unprecedented run of four consecutive 0.75-point hikes.

That is the input to the whole chain. A 9.1% inflation print is the kind of number that forces a central bank to choose price stability over everything else, including the stock market and the bond market. The Fed's reaction function in 2022 was the inverse of a growth scare: instead of cutting to support asset prices, it hiked to crush demand, *knowing* it would hurt asset prices.

It is worth naming the policy error that amplified the shock, because it is part of why the flip was so violent. Through most of 2021, the Fed insisted the inflation was "transitory" — a temporary side effect of supply-chain snarls and reopening demand that would fade on its own. So it kept the funds rate at zero and kept buying bonds (quantitative easing) even as CPI climbed past 5%, then 6%, then 7%. When the "transitory" call was finally abandoned in late 2021 and early 2022, the Fed found itself far behind the curve, with inflation near 8% and a policy rate still at zero. The catch-up was therefore not a measured tightening but a sprint — four straight 0.75-point hikes, a pace unheard of in the modern era. A more gradual tightening, begun earlier, might have repriced the discount rate slowly enough to spread the pain across years; instead the entire repricing was compressed into nine months, which is why both stocks and bonds fell so far so fast. The lateness of the start made the steepness of the path, and the steepness of the path is what made the correlation flip so abrupt. For the full mechanism of how the Fed's path drives the front end of the curve, see [the fed funds path and front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation); for the macro narrative of the cycle itself, see the macro-trading post on [2021-2023 inflation and the fastest hiking cycle](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders).

The deeper lesson is about which shock dominates. In a low-inflation world, the Fed has the freedom to cut whenever markets wobble — the so-called "Fed put" — because there is no inflation constraint. In a high-inflation world, that freedom vanishes. The Fed *cannot* ride to the rescue, because cutting would re-ignite the inflation it is trying to kill. So the negative stock-bond correlation, which depends on the Fed cutting into weakness, is structurally unavailable. The inflation regime does not merely flip the correlation; it removes the mechanism that would have made it negative.

## Why above 4% inflation, the correlation flips

This is not a one-off observation about 2022. It is a documented regularity: the sign of the stock-bond correlation depends on the *level* of inflation. The chart below shows the average stock-bond correlation conditioned on the inflation regime.

![Stock-bond correlation by inflation regime, negative below 2 percent and positive above 4 percent](/imgs/blogs/when-correlations-break-the-2022-stock-bond-flip-5.png)

The pattern is monotonic and striking. When inflation is below 2%, the correlation averages about **−0.45**: bonds hedge stocks, 60/40 works. In the 2-3% range it eases to about −0.30, still negative. In the 3-4% range it is roughly flat at about +0.05 — the hedge is gone but not yet reversed. And above 4% it averages about **+0.50**: both fall together, 60/40 breaks. The 2022 flip lived in that right-most bucket, because CPI spent the year well north of 4%.

The economic logic ties back to the discount rate. At low, stable inflation, the dominant risk to markets is growth, and growth shocks move stocks and bonds oppositely (Fed cuts, bonds rally as stocks fall). At high inflation, the dominant risk is inflation and rates, and inflation shocks move stocks and bonds the same way (Fed hikes, both fall). The crossover sits somewhere in the 3-4% range, where neither force clearly dominates. The threshold is not magic — it is the level above which the Fed's hand is forced and the "Fed put" disappears. This regime dependence is the central thesis of the related post [inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips), and it generalizes the [four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants) framework to the specific case of bonds.

#### Worked example: which regime are you standing in?

Suppose you are deciding how much faith to put in your bond hedge today, and core inflation is running at 3.5%. Where does that put you? Read off the chart: the 3-4% bucket has a correlation near +0.05 — essentially zero. That means bonds neither reliably hedge stocks nor reliably amplify them; the diversification benefit you are paying for is roughly *nil*.

Quantify the cost. With corr = +0.05 instead of the −0.40 you assumed, redo the 60/40 variance from earlier (stock vol 16%, bond vol 12%):

- Cross term at corr +0.05: 2 × 0.6 × 0.4 × 0.05 × 16 × 12 = +4.4
- Variance = 92.2 + 23.0 + 4.4 = 119.6 → volatility = √119.6 ≈ **10.9%**

Versus 8.9% at the assumed −0.40, your real risk is about **22% higher** than the brochure implied, and a single bad inflation surprise could push the correlation into outright positive territory. **The number that matters for your hedge is not last decade's average correlation; it is the inflation regime you are standing in right now — and a 3.5% reading means the hedge is already gone.**

## How it shows up in real markets: the 2022 timeline

Abstractions become concrete in a timeline. Here is how the flip unfolded month by month, because the *grind* of 2022 — the absence of a single clean crash followed by a recovery — was part of what made it so demoralizing.

**January-March: the repricing begins.** Coming into 2022 the 10-year yield was about 1.5% and the Fed had not yet hiked. As inflation prints kept surprising to the upside, the bond market began pricing in aggressive tightening. The Fed delivered its first hike (0.25 points) in March. Both stocks and bonds drifted down together — the flip was already underway, though few named it yet. The S&P was down about 5% for the quarter; long bonds were down more.

**April-June: the bond rout and the CPI shock.** This was the worst stretch. Long-dated Treasuries fell relentlessly as yields marched higher. Then on June 10, the May CPI print came in at 8.6% (later the June print would peak at 9.1%), shattering hopes that inflation had crested. The Fed responded on June 15 with a 0.75-point hike, its largest since 1994. By the end of June, the S&P was down about 20% from its January high — a bear market — and long Treasuries were down a comparable amount. Both legs of 60/40 were in free fall simultaneously.

**July-October: the grind and the bottom.** A summer rally raised hopes, then died as the Fed kept hiking 0.75 points at each meeting. The 10-year yield punched above 4% in October, its highest since 2008. Stocks bottomed in mid-October near a 25% drawdown from the peak; long bonds bottomed around the same time after a roughly 30-plus-percent fall. The misery was synchronized.

**November-December and into 2023: the partial normalization.** As inflation prints finally began to roll over (CPI fell to 6.5% by December), the bond rout eased and stocks staged a recovery. The correlation stayed positive — both assets now rallied *together* on every soft inflation print, just as they had fallen together on every hot one. Through 2023 the correlation drifted from about +0.60 toward +0.45 but never went back to the negative diversifying-era levels. The regime had changed, and it stayed changed.

The defining feature of the timeline is the *synchronization*. There was never a month where stocks fell and bonds saved the day. For an entire year, the two assets a balanced investor owned moved as one. The [correlation during crises](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails) post documents the broader pattern of diversification failing precisely when it is needed; 2022 is the inflation-driven version of that failure.

It is worth dwelling on how *psychologically* corrosive a synchronized grind is, compared to a sharp crash. In 2008 or March 2020, stocks fell fast and far, but bonds rallied violently at the same time, so a balanced investor watched half their portfolio go up even as the other half fell — painful, but the airbag was visibly working, and the recovery came within months. In 2022 there was no airbag and no fast bottom: just twelve months of both halves drifting lower, every relief rally fading, every hot CPI print resetting the decline. There was no single day to point to as "the crash" and no moment where the diversification did its job. That slow, relentless, synchronized character is the signature of an inflation-driven drawdown, and it is exactly what a 60/40 investor is least prepared for emotionally, because every prior bad year had featured the bond rally that 2022 never delivered.

### The 1970s precedent: this had happened before

The strongest evidence that 2022 was a regime, not an accident, is that the same pattern appeared the last time inflation was the dominant macro force: the 1970s and early 1980s. In that era, US inflation ran in the high single digits and into the teens, the Fed under Paul Volcker eventually hiked the funds rate above 19%, and the stock-bond correlation was *positive* — around +0.35 on the rolling measure. Stocks and bonds repeatedly fell together as inflation spiked and rates rose, exactly as in 2022. Bonds were not a hedge in the 1970s any more than they were in 2022, for precisely the same reason: when inflation is the shock, the rising discount rate is the common enemy of both.

What happened next is the part most investors forgot. From the early 1980s, as Volcker broke the back of inflation and a long disinflation began, the correlation drifted down and eventually turned negative around 1998-2000, ushering in the diversifying era. The negative stock-bond correlation that a generation came to treat as permanent was, in the long sweep of history, the *exception* — a feature of a uniquely benign 25-year disinflation. The positive correlation of 2022 was a return to the older, more common pattern. The [structural shifts](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays) post traces this longer arc and the danger of backtests that only span the diversifying window.

## What would have diversified: the 2022 scorecard

If stocks and bonds both fell, did *anything* protect a portfolio? Yes — but mostly things a conventional 60/40 investor did not own in any size. The matrix below scores the major holdings on whether they actually diversified in 2022.

![Scorecard of what fell and what held in 2022 across stocks bonds commodities cash and trend](/imgs/blogs/when-correlations-break-the-2022-stock-bond-flip-7.png)

Read the third column — "did it diversify?" The classic 60/40 components all score "no." Stocks fell 18.1%; long Treasuries fell 31.2%; the 60/40 blend lost 16.1% with both legs red. The diversifiers were elsewhere:

- **Commodities** returned about +16.1% (the Bloomberg Commodity Index). This is the natural inflation hedge: when the shock *is* inflation, the prices of physical things — energy, metals, agriculture — go up by definition. Energy stocks did even better, with the S&P energy sector returning about +65% in 2022. Commodities were the single best diversifier precisely because they have positive inflation beta, the one thing stocks and bonds both lack.
- **Cash** (3-month T-bills) returned about +1.5% and, crucially, did not fall. As the Fed hiked, the yield on cash climbed through the year, and for the first time in a long time holding cash paid you to wait. Dry powder was a winning position.
- **Trend / managed futures** strategies returned roughly +20-30% in 2022. These systematic strategies were short bonds and long commodities and the dollar — they rode the regime rather than fighting it, and 2022 was one of their best years on record.
- **The US dollar** (DXY) rose about +8.2% as the Fed out-hiked other central banks, making dollar cash a relative winner too.

Two assets deserve a closer look because they confused a lot of investors. **Gold** was nearly flat in 2022 (about −0.3%), which disappointed people who think of gold as the classic inflation hedge. The truth is more subtle: gold does not track inflation directly; it tracks *real yields* — the inflation-adjusted interest rate. In 2022 real yields rose sharply (from about −1% to nearly +1.8% on the 10-year TIPS), which is normally a strong headwind for gold. Gold held flat anyway, cushioned by central-bank buying and safe-haven demand, which was actually an impressive result given the real-yield surge. The lesson is that gold is a hedge against *falling real yields*, not against inflation per se, and 2022 was a rising-real-yield year. The [inflation and gold](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story) post unpacks this distinction.

**TIPS** (Treasury Inflation-Protected Securities) also surprised people. You might expect inflation-linked bonds to thrive in a 9% inflation year, but TIPS fell about 12% in 2022. The reason: TIPS are still *bonds*, with duration, and the real-yield component rose enough to overwhelm the inflation-compensation gain. They fell less than nominal Treasuries (the inflation accrual helped), but they still fell, because duration risk dominated. The takeaway is that even the "inflation bond" is not immune when the shock is a *rate* shock — the discount-rate channel reaches almost everything with fixed cash flows.

The broad lesson is not "sell all your stocks and bonds." It is that a portfolio whose only two assets are stocks and bonds has *one* effective bet — the discount rate — even though it looks like two. True diversification against an inflation shock requires an asset with the opposite inflation exposure: real assets, commodities, energy, or cash that re-prices upward as rates rise. The [all-weather and risk-parity](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) approach of owning every macro regime exists precisely to survive years like 2022, when the two assets everyone owns collapse into one.

#### Worked example: what a 10% commodity sleeve would have added

Take the long-duration version of the portfolio from earlier that lost 23.3%, and carve out a 10% commodity sleeve, funded by trimming the stock and bond legs proportionally. New weights: 54% stocks, 36% long bonds, 10% commodities.

- Stock leg: \$54,000 × 0.819 = \$44,226
- Bond leg: \$36,000 × 0.688 = \$24,768
- Commodity leg: \$10,000 × 1.161 = \$11,610
- Total: \$44,226 + \$24,768 + \$11,610 = **\$80,604**

Versus the \$76,660 from the no-commodity version, the 10% sleeve added about **\$3,944**, lifting the year's return from about −23.3% to about −19.4% — nearly four percentage points of protection from a single sleeve. **A small allocation to the one asset with positive inflation beta meaningfully cushioned the worst stock-bond year in a century — that is what real diversification against an inflation shock looks like, and it is exactly what plain 60/40 lacks.**

## The variance under the flip: why the same portfolio became a different animal

It is worth making one idea fully explicit, because it is the deepest and least intuitive consequence of the flip: **the correlation does not just change your returns; it changes your risk, and it does so invisibly.** Two portfolios with identical holdings can carry wildly different risk depending only on the correlation between their components. The investor who held 60/40 in January 2022 owned, on paper, exactly the same fund shares as the investor who held it in January 2012. But because the correlation had flipped from about −0.40 to a path that would reach +0.60, the 2022 investor was carrying a far riskier portfolio — and almost none of them knew it, because nothing in their account statement had changed.

This matters because risk-budgeting models, which professional allocators use to size positions, mostly run on historical correlations. A model fed the 2000-2020 correlation of −0.40 would conclude that 60/40 is a moderate-risk portfolio and might even *add leverage* to hit a volatility target — the logic behind some risk-parity strategies, which lever up bonds precisely because their negative correlation with stocks made the combination look safe. When the correlation flipped positive, those leveraged structures took the flip and multiplied it: levered bonds in a rising-rate, positive-correlation regime is the single worst place to be, and many risk-parity funds had a brutal 2022 for exactly this reason. The model's assumption — a stable negative correlation — was the load-bearing wall, and it gave way.

#### Worked example: the full variance bridge from −0.40 to +0.60

Let us put precise numbers on the risk transformation, building on the earlier calculation. Use stock volatility 16%, bond volatility 12%, weights 0.6 and 0.4, and a \$500,000 portfolio. The portfolio variance has three pieces: the stock term (fixed at 92.2), the bond term (fixed at 23.0), and the cross term, which is the only thing the correlation touches.

- At corr = −0.40: cross term = −35.4 → variance 79.8 → vol **8.9%** → a one-sigma year swings about \$44,600.
- At corr = 0.00: cross term = 0 → variance 115.2 → vol **10.7%** → a one-sigma year swings about \$53,700.
- At corr = +0.60: cross term = +53.1 → variance 168.3 → vol **13.0%** → a one-sigma year swings about \$64,900.

The cross term swung by 88.5 points of variance (from −35.4 to +53.1) — more than the entire stock term and bond term combined could not change. On the \$500,000 portfolio, the *expected* size of a bad year grew from roughly \$44,600 to roughly \$64,900, a **45% increase in risk**, with not one share traded. **The single most important number in your portfolio is the one that never appears on your statement — the correlation between your assets — and in 2022 it quietly turned a moderate-risk portfolio into an aggressive one overnight.**

## Common misconceptions

The 2022 flip generated a lot of confused commentary. Here are the myths that need correcting, each with a number.

**Myth 1: "Bonds are always safe / always hedge stocks."** No. Bonds are safe against *default* (a Treasury will pay you back) but they are not safe against *rising rates*, and they are only a hedge against stocks when the dominant shock is growth, not inflation. In 2022, long Treasuries fell about 31% — a bigger loss than the S&P 500's 18%. "Safe" meant safe from default, not safe from loss.

**Myth 2: "2022 was a black-swan accident nobody could have predicted."** No. The stock-bond correlation has been positive in every high-inflation regime in recorded history — the 1970s and 80s ran around +0.35. The data showed correlation averaging about +0.50 whenever inflation ran above 4%. With CPI at 9.1%, a positive correlation was the *base case*, not a tail event. The surprise was not that the correlation flipped; it was that so many investors had forgotten it could.

**Myth 3: "The Fed will always cut to rescue the market (the Fed put)."** Only when inflation is low. The Fed put is a feature of a disinflation regime, where the central bank has room to ease. In 2022, with inflation at a 40-year high, the Fed hiked into a 25% equity drawdown — the opposite of a rescue. The put has a strike, and that strike is the inflation rate.

**Myth 4: "60/40 is dead."** No — but it is *conditional*. 60/40 is a bet that the dominant shock will be growth, not inflation. In the disinflation regimes that have dominated most of the post-1990 period, that bet pays. In inflation regimes, it does not. The fix is not to abandon 60/40 but to recognize what it assumes and to add an inflation-hedging sleeve so the portfolio is not a single disguised bet on the discount rate. Through 2023-24, as inflation cooled, 60/40 recovered strongly — the regime, not the strategy, was the variable.

**Myth 5: "A long enough window gives you the 'true' correlation."** No — averaging across regimes gives you a meaningless blend. The full-1990-2025 average stock-bond correlation is mildly negative, but that single number hides a +0.35 era, a −0.45 era, and a +0.60 spike. Using the long-run average to size your hedge would have left you fully exposed in 2022. The right window is the *current regime*, not all of history. The [rolling correlation post](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) explains why the window choice is itself a decision.

## How to read it and use it

So how do you actually use all this? The payoff is a small, practical playbook for anyone who owns a balanced portfolio or thinks about asset allocation.

**1. Check the inflation regime first, always.** Before you trust your bond hedge, ask one question: is core inflation below 3%, between 3-4%, or above 4%? Below 3%, your bond hedge is real (correlation negative). Above 4%, it is gone or reversed (correlation positive). In the 3-4% gray zone, treat the hedge as roughly absent. This single check would have warned you off relying on bonds in early 2022, when inflation was already above 7% and rising.

**2. Read the correlation as a regime indicator, not a constant.** Track the rolling stock-bond correlation itself. A negative, stable reading confirms you are in a diversifying regime. A reading rising toward zero or turning positive is a warning that the regime is shifting and your portfolio is becoming a single rate bet. The number is a thermometer for which world you are in.

**3. Hold something with positive inflation beta.** The only structural fix for the flip is an asset that *rises* when inflation is the shock: commodities, energy, real assets, TIPS (inflation-linked bonds), or cash that re-prices upward as rates climb. Even a 10% sleeve, as the worked example showed, adds meaningful protection in an inflation year without much cost in normal years.

**4. Recognize what invalidates the negative-correlation assumption.** The signal to abandon "bonds will hedge me" is not a stock selloff — it is *the cause* of the selloff. If stocks are falling because of growth fears, bonds will help. If stocks are falling because rates are rising on inflation, bonds will not help; they are part of the problem. Diagnose the *driver*, not just the direction.

**5. Don't fight the regime; allocate to it.** The strategies that won in 2022 — trend-following, commodity tilts, holding cash — were the ones that recognized the regime had changed and positioned for it, rather than waiting for the old correlation to reassert itself. Regime-aware allocation means adjusting the *mix* as the inflation regime moves, not holding a static 60/40 and hoping.

To make this concrete, here is a simple monitoring routine you could run with three free data series. First, the **inflation level and trend**: pull core CPI or core PCE year-over-year; below 3% and falling is a green light for the bond hedge, above 4% or rising is a red light. Second, the **rolling stock-bond correlation**: compute the trailing 6-to-12-month correlation of a broad stock index against a long-Treasury index; watch for it crossing from negative toward zero, which is the early warning that the hedge is weakening before it fully reverses. Third, the **Fed's direction**: a Fed that is cutting or on hold with low inflation supports the negative correlation; a Fed forced to hike into inflation removes it. When all three line up bearish — high and rising inflation, a correlation drifting positive, a hiking Fed — you are in or entering a 2022-type regime, and the time to add an inflation-hedging sleeve is *before* the correlation finishes flipping, not after the damage is done. None of this requires forecasting; it requires reading the regime you are already standing in.

The forward lesson is the largest one in this series. There is no fixed "stock-bond correlation," no permanent diversification, no portfolio that is safe in every regime. There is a disinflation regime where bonds hedge stocks and 60/40 works, and an inflation regime where they do not. 2022 was the moment markets remembered the difference, in the most expensive way possible.

It would be a mistake to read this post as an argument against 60/40, or against bonds, or in favor of timing the market. It is none of those. Bonds are still the best hedge against the most common kind of shock — a growth scare and recession — and in a low-inflation world that hedge is genuine and valuable. The argument is narrower and more durable: know what your hedge is a hedge *against*, and check whether the current regime is the one your hedge is built for. A 60/40 portfolio is a magnificent instrument for surviving growth shocks and a poor one for surviving inflation shocks, and the inflation level is the dial that tells you which kind of shock the world is set up to deliver. Add a sleeve that pays off in the regime your core portfolio is weak in, watch the inflation level and the rolling correlation, and you will not be surprised the way 2022 surprised almost everyone.

The investors who survive the *next* inflation shock will be the ones who learned, from this one, that the most dangerous assumption in finance is that a correlation is a constant. It is a regime — and the regime can change in a single year.

## Further reading and cross-links

Within this series — the correlation structure itself:

- [The stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) — the full portfolio-construction treatment of the correlation 2022 broke.
- [Correlation by regime: the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants) — the growth × inflation map that places 2022 in the stagflation/inflation quadrant.
- [Inflation and stocks: the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips) — why the equity-inflation relationship reverses above ~4%.
- [Structural shifts: why today's correlations aren't yesterday's](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays) — the longer arc of regime change behind the 2022 flip.
- [Correlation during crises: when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails) — the broader pattern of correlations converging when you need them apart.
- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the series' founding idea, of which 2022 is the canonical proof.

Mechanism and portfolio context (other complete series):

- [Cross-asset: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) — the portfolio math of the stock-bond correlation.
- [Macro-trading: the business cycle and the fastest hiking cycle](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) — the macro narrative of the 2021-2023 inflation shock.
- [Macro-trading: how policy moves every asset](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) — how a single rate move propagates across all markets.
