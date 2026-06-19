---
title: "Bond Yields: The Master Correlation With Every Asset"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why the 10-year Treasury yield is the discount rate behind every other price, how a rise in it pressures stocks, gold, crypto, real estate and emerging markets while lifting the dollar, and why long-duration assets fall the most."
tags: ["macro", "correlation", "bond-yields", "interest-rates", "duration", "real-yields", "discount-rate", "cross-asset", "10-year-treasury", "regime"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The 10-year Treasury yield is the discount rate that prices every future cash flow, so when it rises, almost every other asset falls together (S&P about −0.45, Nasdaq −0.60, the long bond −0.95, gold −0.55, Bitcoin −0.50, emerging markets −0.50) while the dollar rises (+0.45) — and the longer an asset's cash flows sit in the future, the harder the yield hits it.
>
> - **One variable, many shadows.** Almost every cross-asset correlation in this series ultimately routes through the 10Y, because a yield is a discount rate and a discount rate touches the present value of everything.
> - **Duration is the dial.** A 1-percentage-point rise in the discount rate cuts the present value of a cash flow 30 years out by about 25%, but a 5-year cash flow by only about 5% — which is why growth stocks and crypto fall more than value and cash.
> - **The sign is stable, the strength is not.** Yields and risk assets move inversely across most regimes, but the magnitude swings: 2022's yield surge crushed stocks *and* bonds at once (stock-bond correlation jumped to +0.6), and gold's clean −0.8 link to real yields snapped as central banks bought.
> - **The one number to remember:** a rise in the 10Y carries roughly a **−0.95** correlation with the long Treasury bond itself and **−0.5 to −0.6** with every risk asset — it is the closest thing markets have to a single master switch.

In October 2023 the US 10-year Treasury yield touched 5% for the first time since 2007. Nothing about the economy had broken that week. There was no bank failure, no war headline, no recession print. Yet the Nasdaq 100 fell roughly 3% in days, gold sagged, the dollar pressed multi-month highs, and emerging-market equities and high-yield bonds wobbled in sympathy. A single number on a single government bond had reached out and tugged on the price of almost everything else in the world.

That is the puzzle this post unwinds. Why should the yield on a boring government IOU be able to move a tech stock, a gold bar, a Bitcoin, an apartment building in Ho Chi Minh City, and the Brazilian real all at once? The answer is the most important sentence in macro: **a bond yield is a discount rate, and a discount rate prices every future cash flow there is.** Once you see that, the entire cross-asset map reorganizes around one variable. The 10Y is not just *another* asset that correlates with the others — it is the gravitational center they all orbit. Master this one relationship and roughly half of the other correlations in this series stop being separate facts to memorize and become consequences of a single mechanism.

It is worth saying plainly how unusual this is. Most correlations in markets are pairwise and roughly symmetric — oil and energy stocks, the euro and the dollar, two tech names in the same sector. The 10Y is different in kind: it is not one node in the web, it is the *substrate* the web is drawn on. Move it, and every node shifts. That asymmetry is why professional macro traders watch the 10Y the way a sailor watches the wind — not as one more instrument to trade, but as the condition that determines how every other instrument behaves. The rest of this post earns that claim from first principles and then shows you exactly how the relationship behaves, asset by asset, regime by regime.

![Diagram of the 10-year Treasury yield at the center with arrows to stocks bonds gold Bitcoin emerging markets and the dollar](/imgs/blogs/bond-yields-the-master-correlation-with-every-asset-1.png)

This is the rates "hub" of the series. Every other post — the cleaner real-yield driver, the front-end fed-funds path, the stock-bond regime, the full correlation matrix — connects back here. We build it from absolute zero: what a yield even is, why bond prices move opposite to yields, why distant cash flows get hit hardest, and then we read the empirical correlation of the 10Y with each asset, watch it strengthen and break across real episodes, and end with how a trader actually uses it.

## Foundations: what a bond yield is, and why it discounts everything

Start with the bond itself, because everything else is built on it. A bond is a loan you make to a borrower — here, the US government. You hand over money today; in return the government promises a fixed schedule of payments: small coupon payments along the way and the original principal back at maturity. A 10-year Treasury is just a promise to pay you a fixed stream of dollars over the next ten years.

The **price** of the bond is what you pay today for that fixed stream of future dollars. The **yield** is the single interest rate that makes those future dollars worth exactly today's price. It is the return you earn if you buy the bond and hold it to maturity. Crucially, the future payments are *fixed* — the government promised a set number of dollars. So if the price you pay changes, the *only* thing that can change to keep the math consistent is the yield. That gives us the first iron law of bonds.

### Why yields and bond prices move inversely

Suppose a bond promises to pay you \$1,000 in one year and nothing else. If you pay \$952 for it today, your yield is about 5% (because 952 × 1.05 ≈ 1,000). Now suppose fear or inflation pushes market interest rates up, and buyers will only pay \$926 for that same \$1,000 promise. Your yield just rose to about 8% (926 × 1.08 ≈ 1,000) — but notice that the *price fell* from \$952 to \$926 to make that happen. The payment never changed; only the price moved, and the yield moved the opposite way.

That is the inverse relationship, and it is not a market quirk — it is arithmetic. **The fixed future payment is divided by a bigger number when rates rise, so its present value shrinks.** A bond's price *is* the present value of its fixed payments, so a higher yield mechanically means a lower price. When you hear "yields rose today," you should automatically translate it to "bond prices fell today." They are two descriptions of the same event.

Why would the market suddenly only pay \$926 for a promise it valued at \$952 yesterday? Two forces. The first is the **alternative.** If newly issued bonds now pay 8%, no one will pay full price for your old bond yielding 5% — its price must fall until its effective yield matches the new market rate. The second is **inflation.** If buyers expect inflation to erode the purchasing power of those future dollars, they demand a higher yield to compensate, which again means a lower price today. Both forces push the same way, and both are exactly the forces that swing across macro regimes — which is the deep reason the 10Y is so volatile and so connected to everything else. The yield is the market's running verdict on the value of future dollars, and that verdict changes with every data print about growth, inflation, and policy.

#### Worked example: turning a yield move into a bond price loss

You hold \$10,000 face value of a 10-year Treasury. Its "modified duration" — the sensitivity of its price to yields, which we define properly in a moment — is about 8.5 years. The 10Y yield jumps from 4.0% to 5.0%, a move of 1 percentage point (100 basis points). The price change is approximately −duration × yield change = −8.5 × 1.0% = **−8.5%**. Your \$10,000 position loses about **\$850** of market value, even though the bond will still pay every promised dollar if you hold it to maturity. The intuition: the bond's *future* dollars did not change, but the *price today* of those future dollars fell because money now earns more elsewhere.

### The 10Y as the risk-free discount rate

Here is the leap from "a bond's price" to "the price of everything." The US government can print the dollars it owes, so a Treasury is treated as the closest thing to a default-free, risk-free promise of future dollars. That makes its yield the **risk-free rate** — the baseline return you can earn for simply waiting, with no risk taken.

Now ask: what is *any* asset worth? A stock is a claim on a company's future profits. An apartment is a claim on future rent. Gold is a claim on... future resale value. Bitcoin is a bet on future adoption. **Every asset is a claim on cash flows or value that arrive in the future.** And to value future money in today's terms, you must *discount* it — divide it down — by some rate that captures the time value of money. The natural starting point for that discount rate is the risk-free rate, because that is what you give up by tying your money into the asset instead of parking it in Treasuries.

So the 10Y yield is not just the yield on one bond. It is the **discount rate sitting underneath the valuation of every asset that pays off in the future.** When it rises, the denominator under everyone's future cash flows gets bigger, and the present value of those cash flows shrinks. That single mechanism is why one bond yield can tug on stocks, real estate, gold, crypto, and emerging markets simultaneously. They are not correlated with each other by coincidence; they share a common input, and that input is the 10Y.

This is also the cleanest answer to a question beginners always ask: *why do stocks and crypto and gold so often crash together, when they are supposedly different things?* The textbook promise of diversification is that unrelated assets zig and zag independently, so a basket is smoother than any one holding. But if every asset is a claim on future cash flows and every claim is discounted by the *same* rate, then they are not unrelated at all — they share a hidden common factor. On a quiet day, asset-specific stories dominate and the diversification looks real. On a day when the common factor moves hard — a yield spike — the shared exposure surfaces, and "diversified" assets fall in unison. Diversification fails when you need it most precisely because the thing you were diversifying against, the discount rate, was underneath all of them the whole time. That is the cross-asset version of the master correlation, and it is why this post is the hub.

### Duration: the sensitivity dial

If the discount rate is the *force*, **duration is how hard each asset feels it.** Duration measures how far in the future an asset's cash flows sit, weighted by size. A money-market fund has near-zero duration — you get your cash back tomorrow. A 10-year bond has a duration around 8–9 years. A fast-growing tech company whose profits are mostly expected a decade out has a very *long* effective duration. Gold and Bitcoin, which pay no cash flow at all and are valued entirely on a distant terminal price, behave like the longest-duration assets of all.

The rule that ties it together: **the longer the duration, the more a given change in the discount rate moves the price.** A 1-point rise in yields barely scratches a 1-year cash flow but takes a deep bite out of a 30-year one. This single fact explains the *ranking* of the correlations we are about to see — why the Nasdaq's negative beta to yields is bigger than the S&P's, and why crypto's is bigger still. The discount rate is the same for everyone; duration decides who gets hurt most.

### What "correlation" actually means here

Because this is the rates hub of a series about correlations, we should pin down the word precisely, since we lean on it in every chart below. A **correlation** is a single number, between −1 and +1, that summarizes how two things tend to move together. A correlation of +1 means they move in perfect lockstep in the same direction; −1 means perfect lockstep in opposite directions; 0 means no linear relationship at all. When we say "the 10Y yield and the Nasdaq have a correlation of −0.60," we mean: across the sample, days when yields rose were *usually* days when the Nasdaq fell, and the relationship was fairly strong but not perfect.

Two warnings come baked in. First, **correlation is not causation** — two series can move together because one drives the other, because a third thing drives both, or by pure coincidence over a short window. We argue causation here only because we have a *mechanism*: the discount-rate channel. The number alone would not justify the claim. Second, **a correlation is a regime average, not a constant.** The −0.60 is the typical relationship over a long sample that blends many different market environments; in any given month it can be much stronger, much weaker, or even flip sign. The entire series is built on that second warning, and this post will demonstrate it twice — with the stock-bond flip and the gold break.

There is a closely related cousin you should know: **beta.** Where correlation tells you the *direction and tightness* of a relationship on a unitless −1-to-+1 scale, beta tells you the *magnitude* in real units — "for every 10bp the 10Y rises, this asset falls X percent." A high correlation can come with a small beta (tightly linked but barely moving) and vice versa. In the worked examples we slip between the two: the bars are correlations (sign and strength), the dollar P&L comes from betas (size of move). For the formal statistics — Pearson versus Spearman, how a rolling window changes the number, why the window length matters — the series has dedicated posts; here we use correlation as a working tool and keep the mechanism front and center.

## The mechanism: why a higher discount rate compresses distant cash flows

Let us make the duration logic concrete, because it is the engine of the whole post. The present value of a single dollar received `n` years from now, discounted at rate `r`, is:

```
PV = 1 / (1 + r)^n
```

The `n` in the exponent is the key. Compounding the discount over more years makes the denominator grow *geometrically*, so distant cash flows are punished far more by a rate increase than near ones. Watch what happens when `r` goes from 4% to 5% for cash flows at different horizons.

![Bar chart of present value of one dollar at one to thirty years before and after a one point yield rise](/imgs/blogs/bond-yields-the-master-correlation-with-every-asset-5.png)

#### Worked example: why a 1pp yield move hits a 30-year asset more than a 5-year one

A dollar received in **5 years** is worth 1 / 1.04^5 = \$0.822 at a 4% discount rate. Raise the rate to 5% and it is worth 1 / 1.05^5 = \$0.784 — a loss of about **4.7%**. Now take a dollar received in **30 years**: at 4% it is worth 1 / 1.04^30 = \$0.308; at 5% it is worth 1 / 1.05^30 = \$0.231 — a loss of nearly **25%**. The *same* 1-point rate rise destroyed five times more value from the distant cash flow. The takeaway: when yields rise, the market does not punish all assets equally — it punishes the ones whose payoff is furthest away the hardest, and that is precisely what "long duration" means.

This is the cleanest way to understand why a growth-stock index falls more than a value index when the 10Y jumps. A "value" company — a bank, an oil major, a utility — earns most of its cash now and over the next few years; its valuation is short-duration, so a higher discount rate barely moves it. A "growth" company — an unprofitable software firm whose investors are paying for profits expected in 2035 — has nearly all its value in distant cash flows. Raise the discount rate and you take a 30-year-style bite out of it. The Nasdaq is stuffed with long-duration growth; the S&P is more balanced; so the Nasdaq's negative correlation with yields is *structurally* larger.

### Convexity: the curvature that makes big moves worse

Duration is a *linear* approximation — it assumes the price falls in proportion to the yield rise. In reality the relationship curves, and that curvature is called **convexity.** For a normal bond, convexity is your friend: prices fall a little *less* than duration predicts when yields rise a lot, and rise a little *more* when yields fall. But for the present-value math of a long-duration asset, the geometry of `1 / (1 + r)^n` means the *percentage* damage from each additional point of yield is roughly stable while the dollar amounts compound — and crucially, the assets that fall most are precisely those whose value is concentrated furthest out, where the exponent bites hardest. The practical upshot for the cross-asset correlation: **large yield moves do not just hurt long-duration assets proportionally more — they hurt them in a way that is hard to hedge linearly,** because the sensitivity itself shifts as yields move. A growth-stock book that looks 20% rate-sensitive at 4% yields can behave more sensitively as yields climb, which is part of why the 2022 drawdowns in the longest-duration assets overshot simple duration estimates.

You do not need the full convexity formula to trade the correlation. The working takeaway is that the −0.60 Nasdaq correlation and the −0.50 crypto correlation are *averages* that get more negative in the tails — the bigger and faster the yield move, the more the long-duration assets underperform their linear estimate. The correlation is not just strong; it is strongest exactly when you least want it to be, in a violent yield spike. For the formal treatment of duration and convexity in the bond itself, the macro-trading post on [how monetary policy moves bonds](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity) is the reference.

#### Worked example: P/E compression on a \$10,000 growth-stock position

You own \$10,000 of a high-growth stock trading at a forward P/E of 40, meaning the market pays \$40 for every \$1 of next year's earnings — a rich multiple that only makes sense if you discount a long runway of growing profits. A simplified way to see the rate sensitivity: a long-duration equity's fair multiple moves roughly inversely with the discount rate. If the 10Y real discount rate rises by 1 point and that compresses the justified multiple from 40 to 34 (a 15% multiple cut, in line with what long-duration tech endured in 2022), your \$10,000 position falls to about **\$8,500** — a **\$1,500 loss** with no change in the company's actual earnings. Same earnings, higher discount rate, lower price. A value stock at a P/E of 12 in the same scenario might see its multiple slip only to about 11.5, a loss closer to **\$400** on a \$10,000 position. The intuition: the higher multiple was always a bet on distant cash flows, and distant cash flows are exactly what a higher discount rate devalues.

## The centerpiece: the 10Y-yield row of the correlation map

Now we can read the empirical numbers. Across this series, every macro driver gets a *row* in a master correlation matrix — its footprint across all assets. The 10Y yield's row is the one that pulls the hardest. These are researched approximations of the documented sign and relative strength of each correlation, not a single tick-exact sample, but the *pattern* is robust and well established in cross-asset research.

![Horizontal bar chart of the correlation of a rise in the 10-year yield with each asset return](/imgs/blogs/bond-yields-the-master-correlation-with-every-asset-2.png)

Read the bars from the bottom up and the duration logic falls right out:

- **US 10Y bond: −0.95.** The most negative of all, and the most mechanical — the bond *is* the yield. A rising yield is, by arithmetic, a falling bond price. This is the purest expression of the inverse law.
- **Nasdaq: −0.60.** Long-duration growth equity. More rate-sensitive than the broad market because more of its value sits in distant cash flows.
- **Gold: −0.55.** Gold pays no coupon and no dividend; its only "yield" is zero. When Treasuries pay more, the opportunity cost of holding a non-yielding metal rises, so gold loses ground. (We will see this link is cleaner against *real* yields specifically.)
- **Bitcoin: −0.50** and **EM equity: −0.50.** Crypto trades as the longest-duration risk asset around; emerging markets carry both a long-duration growth profile and dollar-funding sensitivity. Both feel the yield.
- **S&P 500: −0.45.** The broad market — a blend of short and long duration, so its negative beta is real but milder than the Nasdaq's.
- **US dollar: +0.45.** The lone positive. A higher US yield makes dollar assets more attractive relative to the rest of the world, pulling capital *in* and lifting the currency. Higher yields are a headwind for risk assets but a *tailwind* for the dollar.

The shape of this row is the whole thesis: **one driver, one sign for risk assets (down), one exception (the dollar, up), and a magnitude ordered by duration.** Almost everything else in this series is a refinement of this picture.

#### Worked example: turning the row into a single-day move

Suppose the 10Y rises 20 basis points (0.20 percentage point) on a hot inflation print — a typical hawkish day in the 2022–23 regime. The data-surprise betas in this series put the Nasdaq's same-session reaction near −1.0% per 0.1pp of *core-CPI* surprise, and a 20bp yield move on such a day maps to roughly a 1.5–2% Nasdaq decline, a ~0.9% S&P decline, gold off ~0.8%, Bitcoin off ~1.5%, and the dollar up ~0.5%. On a \$10,000 Nasdaq position that is roughly a **\$175 loss** in a session; on a \$10,000 S&P position, about **\$90**. Same shock, different duration, different damage — the ranking is exactly the bar chart. (For the intraday mechanics of *which* releases move yields and how fast, see the event-trading series; this post is about the *standing* correlation, not the release-day reaction.)

### The dollar channel: the one asset that rises with yields

The lone positive bar deserves its own explanation, because the dollar is the second-most-important spoke after the bond itself. A currency's value, simplified, reflects the relative return on holding assets denominated in it. When US 10Y yields rise relative to German Bunds or Japanese JGBs, dollar-denominated assets offer a better yield for the same risk, so global capital rotates *into* dollars to capture it. More demand for dollars means a stronger dollar — hence the +0.45 correlation between a 10Y rise and the dollar index.

This matters because the strong dollar then becomes its *own* transmission channel, hitting a second layer of assets. A stronger dollar pressures commodities (which are priced in dollars, so they get more expensive for the rest of the world and demand softens), gold (a non-dollar reserve asset), and especially **emerging markets**, many of which borrow in dollars. When the dollar rises, an emerging-market government or company that owes dollars but earns local currency sees its debt burden grow in real terms — a direct financial tightening. So a rising 10Y hits EM *twice*: once through the long-duration discount-rate channel, and again through the dollar-funding channel. That double exposure is why EM equity carries a −0.50 correlation to the 10Y despite being geographically far from the US Treasury market. The dollar's role as cross-asset gravity is the subject of the cross-asset series; here the point is simply that the 10Y's reach extends *through* the dollar to a whole second tier of assets.

#### Worked example: the EM double-hit

An emerging-market equity ETF you hold \$10,000 of has two exposures to a rising 10Y. Channel one: as a long-duration risk asset, it carries roughly the same discount-rate sensitivity as global growth equity, so a 50bp 10Y rise drags it down perhaps 3% on the rate channel alone — a **\$300** loss. Channel two: the same 50bp rise lifts the dollar ~1%, and a stronger dollar historically pressures EM equity by an additional ~1.5% through the funding-and-translation channel — another **\$150** loss. Total: about **\$450** on the \$10,000 position from a single 50bp yield move, roughly double what a US-only long-duration asset would lose. The intuition: the 10Y does not just discount EM cash flows; it also tightens EM financial conditions through the dollar, so the correlation is mechanically larger than duration alone would suggest.

### The real-estate channel: the slow-moving spoke

Real estate is the spoke that moves slowest but feels the 10Y most directly of all, because property is bought almost entirely with borrowed money tied to long-term rates. A home or a commercial building is the ultimate long-duration asset — a claim on decades of future rent or use — and its purchase is financed by a mortgage whose rate tracks the 10Y closely. When the 10Y rises, mortgage rates rise, the monthly payment a given price implies rises, and affordability falls. The discount-rate channel and the financing channel point the same way: higher yields, lower property values.

The reason real estate does not show up with a clean daily correlation like the others is *lag*. Property prices are sticky — sellers resist marking down, transactions are slow, appraisals lag — so the 10Y's effect shows up over quarters, not days. But it is there, and it is large. The 2022 yield surge that doubled US mortgage rates from ~3% to ~7% froze the housing market and re-priced commercial real estate sharply over the following year. The correlation is real; it just operates on a slower clock, which is itself a useful lesson — *the same driver can have very different lead-lag profiles across its spokes.*

## How the 10Y itself has moved: the path that re-priced everything

A correlation only matters because the driver actually moves. And the 10Y has moved violently. From the COVID-era floor near 0.6% in 2020 — when the discount rate was effectively zero and *every* asset rallied because future cash flows were divided by almost nothing — to nearly 5% in late 2023, the discount rate underneath the entire global asset map roughly *octupled*. That is the single most important macro chart of the decade.

![Line chart of the 10-year US Treasury yield from 2020 to 2026](/imgs/blogs/bond-yields-the-master-correlation-with-every-asset-3.png)

Three regimes are visible, and each one tells you about the correlation:

- **2020, yields near zero — the everything-rally.** With the discount rate at the floor, the present value of distant cash flows was enormous. Long-duration assets feasted: unprofitable tech, SPACs, crypto, and growth-at-any-price all soared. This is the duration logic in *reverse* — a near-zero discount rate inflates exactly the assets that a high one crushes. The same `1 / (1 + r)^n` formula that punishes distant cash flows when `r` is high *rewards* them lavishly when `r` is near zero: a dollar received in 30 years is worth \$0.31 at a 4% rate but \$0.74 at a 1% rate, more than double. That is the arithmetic behind the entire 2020–21 mania in long-dated, speculative assets — the discount rate was so low that the most distant, most speculative bets became the most valuable, exactly inverting the 2022 ranking.
- **2022, the yield surge — stocks and bonds fell together.** As the Fed hiked and inflation raged, the 10Y rocketed from ~1.5% to over 4%. The discount rate rose for everyone at once, so the S&P fell ~19%, the Nasdaq ~33%, *and* long bonds had their worst year in modern history. Diversification failed precisely because a single common driver — rising yields — was hammering stocks and bonds simultaneously.
- **2023 August–October, the 5% scare.** A second yield spike, this time driven by the long end (term premium and fiscal-supply worries), wobbled equities again. The Nasdaq's ~3% drop in days as the 10Y kissed 5% is the episode we opened with.

The lesson of the path is that **the 10Y is not a slow-moving backdrop; it is an active driver that can swing 200–300 basis points in a year, and when it does, it drags the whole asset map with it.**

## Decompose the yield: nominal = real + breakeven

The 10Y nominal yield you see quoted is actually two things glued together, and pulling them apart is essential to understanding the correlation cleanly.

```
nominal yield = real yield + breakeven inflation
4.5%          ~ 2.0%      + 2.3%   (illustrative, 2024-25 levels)
```

The **real yield** is the true, inflation-adjusted cost of money — what you earn *after* inflation eats its share. You can observe it directly from inflation-protected Treasuries (TIPS). The **breakeven** is the difference between the nominal and real yield, and it is the market's expectation of average inflation over the next 10 years.

![Diagram decomposing the nominal ten-year yield into a real yield and a breakeven inflation component](/imgs/blogs/bond-yields-the-master-correlation-with-every-asset-4.png)

Why does the split matter for correlation? Because **the real-yield component is the cleaner driver of risk assets.** When the nominal yield rises because *real* yields rose — money genuinely got more expensive, often because the Fed is tightening into a strong economy — that is the part that compresses valuations and pressures gold, growth stocks, and crypto. When the nominal yield rises only because *breakeven inflation* rose, the signal is murkier: some assets (real assets, inflation hedges) may even benefit. This is the single most important refinement of this post, and the next post in this series — [real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation) — is built entirely around it. The breakeven side has its own post too: [PCE, breakevens, and the forward inflation correlation](/blog/trading/macro-correlations/pce-breakevens-and-the-forward-inflation-correlation).

#### Worked example: the same nominal move, two very different meanings

Two days, two identical 15bp rises in the nominal 10Y, opposite asset reactions. On **day one**, TIPS data shows real yields rose 15bp while breakevens were flat — the move was *all real*. Gold falls 0.7%, the Nasdaq falls 1%, the dollar firms; a \$10,000 gold position loses about **\$70** and a \$10,000 Nasdaq position about **\$100**. On **day two**, the same 15bp nominal rise is *all breakeven* — real yields flat, inflation expectations up. Gold *rises* 0.5% as an inflation hedge, the dollar is roughly flat, and growth stocks barely flinch; that same \$10,000 gold position now *gains* about **\$50**. Identical headline yield move, opposite gold reaction — because the correlation that actually binds gold is to the *real* component, not the nominal one. This is why "yields up = gold down" is only a half-truth.

## Gold and real yields: the cleanest version of the link, and its break

Gold is the perfect case study for the real-yield refinement, and it also teaches the most important caveat in this entire series: **a strong correlation can break.** For most of the post-2007 era, gold tracked the 10Y *real* yield almost perfectly inversely. Gold pays no income, so when real Treasuries offered a juicy positive real return, holding metal had a real opportunity cost and gold fell; when real yields went negative (you *lost* purchasing power holding Treasuries), gold soared. The fit was tight.

![Scatter plot of gold price versus the ten-year real yield showing a clean negative line that breaks after 2022](/imgs/blogs/bond-yields-the-master-correlation-with-every-asset-6.png)

Why real yields specifically, and not nominal? Because gold's whole appeal is as a store of value that holds its purchasing power. The competing asset is an inflation-protected Treasury, which *also* preserves purchasing power but additionally pays a real yield. When that real yield is high, the TIPS strictly dominates gold — same inflation protection, plus a positive real return — so money leaves gold. When the real yield is negative, the TIPS *loses* purchasing power and gold, which at least holds steady, looks attractive by comparison, so money floods in. The driver is the *real* opportunity cost, which is exactly the real yield. Nominal yields can rise purely because expected inflation rose, and that does not hurt gold at all — it may help it — which is why the nominal correlation is messier than the real one.

The blue dots (2007–2021) trace a clean downward line: the correlation between the real yield and the gold price over that window is about **−0.96**. That is one of the strongest, most reliable macro correlations that has ever existed. You could practically price gold off the TIPS yield.

And then look at the red diamonds (2022–2025). Real yields surged from negative all the way to roughly +2%, which by the old relationship should have crushed gold toward \$1,200. Instead gold *climbed* to record highs above \$2,600. The correlation over that window flipped to about **+0.8** — the opposite sign. What broke it? A new, larger buyer entered: central banks, particularly outside the West, bought gold aggressively as a reserve asset less exposed to dollar politics. A structural demand shock overwhelmed the discount-rate channel.

#### Worked example: betting the broken correlation

A trader in early 2023 sees real yields at 1.7% and reasons from the 2007–2021 fit — corr −0.96 — that gold "should" be near \$1,250. They short a \$10,000 notional gold position expecting it to fall toward that level. Over the next two years real yields rose *further*, to ~2.0%, which by the old model should have pushed gold even lower. Instead gold rose to ~\$2,650 — a roughly **36% move against the short**, a **\$3,600 loss** on the \$10,000 notional. The correlation that had held for fourteen years with a −0.96 fit did not just weaken; it inverted. The intuition every macro trader must internalize: a correlation is a *regime average*, not a law of physics, and the cleaner and more famous it is, the more painful its eventual break.

This is exactly the kind of decoupling the dedicated gold post explores: [inflation and gold, the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story).

## Crypto as the longest-duration risk asset

If the discount-rate framework is right, then Bitcoin — an asset with no cash flow, valued entirely on a distant terminal adoption story — should behave like the *longest-duration* risk asset of all, and it should be most correlated with yields and with growth equity precisely when the rates story dominates the market. The data backs this up.

![Line chart of the rolling correlation between Bitcoin and the Nasdaq from 2019 to 2025](/imgs/blogs/bond-yields-the-master-correlation-with-every-asset-7.png)

Before 2020, Bitcoin's correlation with the Nasdaq was near zero — it traded on its own crypto-native narrative, indifferent to macro. Then in the 2022 rate shock the correlation spiked to about **0.65**. As the discount rate ripped higher, crypto fell in lockstep with long-duration tech, because both are bets on distant cash flows and both got re-priced by the same rising denominator. Bitcoin lost ~65% in 2022, deeper than the Nasdaq's ~33%, consistent with an even longer effective duration. As the rate shock faded into 2024–25, the correlation drifted back toward 0.2–0.3 and crypto re-coupled to its own halving and adoption dynamics.

The lesson generalizes: **the 10Y's grip on an asset is strongest when rates are the dominant market story and weakest when an asset-specific narrative takes over.** Crypto is correlated to yields *through the long-duration risk channel*, not because anyone is discounting Bitcoin coupons — there are none — but because in a rate-driven regime, all the longest-duration bets trade as one big risk position.

This time-varying correlation is itself a tradeable signal. When you see crypto's correlation with the Nasdaq climbing toward 0.6, that is the market telling you a macro/rates regime has taken over and crypto has become a leveraged bet on the discount rate rather than on its own adoption story. When the correlation falls back toward 0.2, the rates channel has loosened and crypto is back to trading on halvings, flows, and protocol news. A trader who tracks the correlation itself — not just the price — knows *which game is being played* at any moment, which is half the battle. The same logic applies to any long-duration asset: the moment its correlation to the 10Y spikes, you are no longer trading the asset, you are trading the discount rate, and you should size and hedge accordingly.

#### Worked example: the duration ranking in a single drawdown

In 2022's rate surge, line up the drawdowns by duration. Short-duration cash: roughly flat. The S&P (mixed duration): about −19%. The Nasdaq (long-duration growth): about −33%. Bitcoin (longest duration): about −65%. A \$10,000 allocation to each would have ended the year worth roughly \$10,000 in cash, **\$8,100** in the S&P, **\$6,700** in the Nasdaq, and **\$3,500** in Bitcoin. The single driver — a rising discount rate — explains the entire ranking, and the ranking *is* the duration ladder. That is the master correlation doing its work across the full risk spectrum at once.

## Lead, lag, and the direction of causality

A correlation has a *timing* as well as a sign and a strength, and the 10Y's timing is subtle because it can be both cause and effect. Usually the yield leads: a yield move out of the bond market re-prices the discount rate, and risk assets follow within the same session or over a few days, fastest for the most liquid and longest-duration assets (futures and crypto react in minutes, equities in hours, real estate in quarters). That ordering — yields first, spokes after — is why traders watch the 10Y as a real-time barometer of the macro mood.

But the arrow can run the other way too. In a sharp risk-off event — a banking scare, a geopolitical shock — money *flees into* Treasuries for safety, which *pushes yields down*. Here falling stocks are causing falling yields, not the reverse. The correlation reads the same on the chart (stocks down, yields down → they moved together inversely to a yield *rise*), but the causality is flipped: it is a flight-to-quality, not a discount-rate move. Telling the two apart is the single most useful skill in trading this correlation, and the tell is *what else is happening*: in a discount-rate move, the dollar rises and gold falls with the yield; in a flight-to-quality, the dollar may rise but gold *also* rises and the move is driven by fear, not inflation. The mechanism behind the move tells you whether to fade it or respect it.

## How to measure it yourself

You do not have to take the −0.60 on faith — the whole point of a correlation series is that these numbers are *measurable*. The recipe is simple enough to run in a spreadsheet or a few lines of pandas. Take daily (or weekly) percent changes of the 10Y yield and of each asset over a chosen window, then compute the correlation of the two change-series. Three choices determine what number you get, and getting them wrong is how people end up with misleading correlations.

First, **use changes, not levels.** Two assets can both be trending up over a decade and show a high level-correlation that means nothing; the relationship that matters for trading is whether they move together *day to day*, which is a correlation of *changes*. Second, **pick the window deliberately.** A full-sample correlation blends every regime into one number that may describe none of them — the gold-real-yield correlation is −0.96 in 2007–2021 and +0.8 in 2022–2025, so the full-sample number near zero is *worse than useless.* A rolling window (say 90 days or 24 months) shows you the correlation *changing*, which is the real story. Third, **align the timing** — if one series leads the other, an un-lagged correlation understates the true relationship, so check a few lead-lag shifts.

#### Worked example: a conditioned correlation beats a single number

A trader computes the full-sample correlation of the 10Y yield with gold and gets roughly −0.2 — weak, almost useless. Frustrated, they *condition on the real-yield regime* instead: over the 2007–2021 window when central-bank buying was modest, the correlation of gold with the real yield is −0.96; over 2022–2025 it is +0.8. The single blended number hid two strong, opposite relationships. Acting on the −0.2 would have meant treating gold as roughly independent of rates — and missing both the clean pre-2022 trade and the dangerous post-2022 reversal. The lesson worth real money: **the right correlation is a conditional one. Always ask "in which regime?" before you trust the number,** because the master correlation, like every correlation in this series, is a regime average that can flip.

## Common misconceptions

**"Higher yields are always bad for stocks."** No — the *level* and the *reason* both matter. Yields rising from 1% to 3% in a healthy economy can coincide with rising stock prices, because the growth that pushes yields up also lifts earnings. The correlation turns sharply negative when yields rise *fast*, rise for *inflation/hawkish* reasons, or rise from an already-high level so the discount-rate damage dominates the earnings boost. The standing −0.45 correlation is a regime average across both kinds of episodes; in a "good growth" regime the same-day correlation can be near zero or even positive. (The inflation-regime dependence is the subject of [inflation and stocks, the correlation that flips](/blog/trading/macro-correlations/inflation-and-stocks-the-correlation-that-flips).)

**"It's the nominal yield that drives gold and growth stocks."** Mostly it's the *real* yield. As the gold scatter showed, gold's clean −0.96 link was to the real yield, not the nominal one — and the two can diverge whenever inflation expectations move. If you watch only the nominal 10Y you will be confused on every day that the move is all breakevens. The real yield is the cleaner master signal.

**"Bonds always diversify stocks, so 60/40 is safe."** Only when the stock-bond correlation is negative — which it was for the two decades from roughly 2000 to 2021, but emphatically *was not* in 2022, when both fell together and the correlation jumped to about +0.6. When inflation is the dominant macro risk, the same rising-yield shock hits stocks and bonds at once, and your "diversifier" becomes a second source of loss. The correlation is conditional on the inflation regime. (The full story: [the stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime).)

**"Crypto is an inflation hedge / a hedge against the system."** In the regime that matters most — a rate shock — Bitcoin traded as the highest-beta long-duration risk asset, falling *harder* than the Nasdaq, not hedging anything. Its correlation to the Nasdaq hit 0.65 in 2022. It can decouple in calm regimes, but when the 10Y is driving, crypto is risk-on, not a hedge.

**"A correlation this strong is reliable."** The opposite warning applies. Gold's −0.96 link to real yields was about as strong as macro correlations get, and it still inverted to +0.8 when a structural buyer arrived. The strength of a correlation tells you how cleanly it has held *in-sample*; it tells you nothing about whether the regime that produced it will persist. The cleanest, most-traded correlations are the ones whose breaks hurt the most, precisely because everyone is leaning on them in the same direction when they snap.

**"The 10Y always leads the other assets."** Usually, but not in a panic. In a flight-to-quality — a banking crisis, a war scare — frightened money rushes *into* Treasuries, which pushes yields *down*, and here the causality runs from risk assets to yields, not the other way. The chart looks the same (stocks and yields fell together), but trading it as a discount-rate move would be a mistake: the right read is fear-driven, and the tell is that gold and the dollar both rise. Always ask whether a yield move is a *discount-rate* move (inflation/policy, gold falls) or a *flight-to-quality* move (fear, gold rises) before you act on the correlation.

## How it shows up in real markets

**2020 — the everything-rally.** The Fed cut to zero and the 10Y collapsed to ~0.6%. With the discount rate at the floor, present values of distant cash flows ballooned, and the longest-duration assets led: unprofitable tech, IPOs, SPACs, and crypto all soared while the real economy was still in lockdown. The lesson: a falling discount rate inflates exactly the assets a rising one crushes. This was the master correlation running in reverse, lifting everything at once.

**2022 — the synchronized crash.** Inflation hit a 40-year high, the Fed hiked from 0.25% to 4.5%, and the 10Y surged past 4%. The discount rate rose for every asset simultaneously: the S&P fell ~19%, the Nasdaq ~33%, Bitcoin ~65%, gold churned, and — critically — long Treasuries had a historic loss too. The stock-bond correlation flipped from about −0.45 to +0.6, so the classic 60/40 portfolio had one of its worst years ever. There was nowhere to hide in duration, because the thing being re-priced *was* duration. This is the canonical demonstration that the 10Y is the common driver.

**2023, August–October — the 5% term-premium spike.** This time the surge came from the *long end*: fiscal-supply worries and a rising term premium pushed the 10Y to ~5% even as the Fed was nearly done hiking. Equities, which had rallied all summer, wobbled — the Nasdaq shed ~3% in days, growth led the decline, and the dollar firmed. A pure long-end yield move, with no change in the economic outlook, re-priced the most rate-sensitive assets first. Duration was the sorting variable, exactly as the framework predicts.

**2024–25 — gold breaks the link.** Real yields stayed high (~2%) yet gold ran to records above \$2,600 on central-bank buying. The single most reliable macro correlation of the prior fifteen years inverted. The episode is the standing reminder that the discount-rate channel is powerful but not the *only* channel — a large enough structural demand shock can overwhelm it.

### The stock-bond correlation: the master correlation's most important consequence

The most consequential downstream effect of the 10Y being a common driver is what it does to the relationship *between* stocks and bonds — the engine of every balanced portfolio on earth. For two decades, roughly 2000 to 2021, stocks and bonds were negatively correlated: when stocks fell, it was usually because growth was disappointing, the Fed cut, yields fell, and bond prices *rose* — so bonds cushioned equity losses. That negative correlation is the entire premise of the 60/40 portfolio, and it was *not* an accident; it held because the dominant macro risk in that era was *growth*, and growth shocks push stocks and bonds in opposite directions.

Then the dominant risk changed. When *inflation* becomes the thing the market fears, the logic inverts. An inflation shock makes the Fed hike, which pushes yields up — hurting bonds *and*, through the discount-rate channel, hurting stocks. Now both fall together. The stock-bond correlation, which had averaged about −0.45, jumped to roughly **+0.6** in 2022. The "diversifier" became a second engine of loss in the exact year you needed it most. The number is conditional on the inflation regime: data in this series shows the stock-bond correlation runs around −0.45 when inflation is below 2%, near zero in the 3–4% band, and around +0.5 when inflation tops 4%. The whole flip is governed by which risk the 10Y is responding to — growth or inflation — which is, once again, the master correlation deciding the fate of every other relationship downstream of it.

#### Worked example: when your hedge stops hedging

A classic 60/40 portfolio holds \$60,000 in the S&P and \$40,000 in long Treasuries. In a normal (negative-correlation) regime, a 10% equity drop of −\$6,000 is partly offset because the same growth scare pulls yields down and the bond sleeve gains, say, +4% or +\$1,600 — net loss about **\$4,400**, the bonds did their job. In the 2022 inflation regime the correlation flipped to +0.6: the same −10% equity loss of −\$6,000 came *alongside* a rising-yield, falling-bond move, so the bond sleeve *also* fell, say −10% or −\$4,000 — net loss about **\$10,000**, more than double, with the hedge adding to the damage instead of absorbing it. The intuition: the bond was never an independent hedge; it was a bet that the 10Y would fall when stocks did, and that bet only pays in a growth-driven, not an inflation-driven, selloff. This is why the dedicated [stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) post calls it the most important regime switch in macro.

## How to read it and use it

Here is the playbook the whole post builds toward.

**The signal.** Treat the 10Y yield — and especially its *real* component — as the master macro variable on your screen. When it is moving fast, expect risk assets to move inversely and the dollar to move with it, with magnitude ordered by duration: bonds and growth and crypto most, value and cash least. A practical morning routine for anyone trading macro-sensitive assets: glance at the 10Y nominal yield, the 10Y TIPS real yield, and the dollar index *before* you look at your individual positions, because those three will explain most of what your book does that day. If the 10Y is quiet, asset-specific stories will dominate and the correlations will look loose; if the 10Y is moving, the spokes will move with it and idiosyncratic analysis takes a back seat to the rate read.

**The regime check before you trade the correlation.** Ask three questions every time. (1) *Why* are yields moving — real yields (cleaner, more negative for risk) or breakevens (murkier)? Pull up the TIPS yield, not just the nominal. (2) *What regime* is the stock-bond correlation in — if inflation is the dominant risk, bonds will not diversify, so do not lean on them to hedge equity. (3) *Is an asset-specific narrative overriding the rates channel* — gold under central-bank buying, crypto in a halving-driven bull, a single stock on earnings — in which case the standing correlation may not bind right now.

**What invalidates it.** The correlation is conditional, so it is "invalidated" not by a single day but by a regime shift: when the *reason* for the yield move changes (growth-driven vs inflation-driven), when a structural buyer enters (gold 2022), or when an asset's own story takes the wheel (crypto in calm regimes). A trader who treats the −0.5 betas as constants will get run over the day the regime turns; a trader who re-checks the *why* behind every yield move stays on the right side of the master correlation.

**The position sizing payoff.** Because the 10Y sits underneath everything, it is the single best hedge ratio input you have. If your book is long a basket of long-duration risk (growth equity, crypto, EM), you are implicitly *short* the 10Y yield — a rate spike is your worst day. Sizing a small Treasury or rate-hedge position against that exposure is the cleanest way to neutralize the one risk that hits your whole book at once. That is the practical reason the hub matters: hedge the center, and you have partially hedged every spoke.

#### Worked example: hedging the hub

A \$100,000 portfolio is 60% long-duration growth equity and 20% crypto — call it \$80,000 of assets with an effective duration sensitivity that, empirically, loses about 1.2% for every 10bp rise in the 10Y. A 50bp yield spike would cost roughly 6% of that \$80,000, or about **\$4,800**. To offset it, you hold a rate hedge that *gains* when yields rise — say a short-duration-Treasury or rate-futures position sized to make about \$4,000 on a 50bp move. The net loss shrinks from \$4,800 to about **\$800**. You did not predict the yield move; you simply recognized that the master correlation routes your entire risk book through one variable and hedged that variable directly. That is the whole point of identifying the hub.

The deepest takeaway is conceptual, not tactical. Once you see the 10Y as the discount rate underneath every price, the cross-asset world stops being a list of separate markets to learn one at a time and becomes one system with a single dial at its center. Stocks, bonds, gold, crypto, real estate, and emerging markets are not seven independent things that happen to move together; they are seven different *durations* of the same underlying bet — that future cash flows are worth something today — and the 10Y is the price of that "today versus future" trade-off. Turn the dial, and they all respond, in an order set by how far out their cash flows sit. Every other post in this series adds a refinement to that picture, but they all sit on top of this one mechanism. If you remember one thing, remember that the boring government bond yield is the gravity that every other asset price falls toward.

## Further reading and cross-links

This post is the rates hub of the series. The natural next steps:

- [Real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation) — the refinement that the *real* yield, not the nominal, is the cleaner driver.
- [The fed-funds path and the front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation) — how the short end (2Y, policy) feeds the long end.
- [The stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) — why bonds stop diversifying when inflation dominates.
- [The macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — the full grid this post's row sits inside.
- [Inflation and gold, the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story) — the gold decoupling in depth.

For the *mechanism* behind why rates are the master variable and how policy moves bonds, lean on the macro-trading series rather than re-deriving it:

- [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable)
- [How monetary policy moves bonds: duration and convexity](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity)
- [How policy moves every asset: the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map)

And for the allocator's view of why one variable prices the whole map:

- [Real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything)

For the bond market itself from the ground up — pricing, duration, convexity, the yield curve — the fixed-income series is the deep reference, and the [covariance and correlation tooling](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants) behind measuring any of these correlations lives in the math-for-quants series.
