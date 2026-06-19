---
title: "The VIX, Risk-On/Risk-Off, and the Correlation Spike"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why your diversified portfolio becomes one big bet in a crisis: how the VIX signals the switch from risk-on to risk-off, and why average correlations spike toward one exactly when you need diversification most."
tags: ["macro", "correlation", "vix", "risk-on-risk-off", "diversification", "volatility", "crisis", "deleveraging", "portfolio-risk", "regime"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Correlation is conditional: across a book of risk assets it sits near +0.25 in calm markets but spikes toward +0.80 in a crisis, so your "diversified" portfolio collapses into one big bet exactly when you need protection. The VIX is the fear gauge that signals the switch from risk-on to risk-off.
>
> - **Risk-on vs risk-off (RORO)** is the single dominant factor in a panic: in calm markets assets follow their own stories (earnings, rates, growth); in a crisis everything trades on one question — *risk, yes or no?*
> - **The cruel irony of correlation**: average pairwise correlation across major risk assets rises from ~0.25 (calm) to ~0.45 (normal selloff) to ~0.80 (crisis/deleveraging). Diversification works least when you need it most.
> - **The VIX is the switch.** Below ~20 you are in the diversifying regime; above ~30 you are in the everything-falls-together regime. The VIX hit 80.9 in 2008, 82.7 in March 2020, and 65.7 in the August 2024 yen-carry unwind.
> - **The mechanism is deleveraging**: a vol shock triggers margin calls and VaR cuts, and forced sellers dump *what is liquid* (everything), which is what physically drives correlations to one.
> - **The one number to remember**: a 10-asset book at 20% vol each carries ~11.4% portfolio volatility at 0.25 correlation but ~18.1% at 0.80 — your one-sigma risk on a \$100,000 book jumps from \$11,402 to \$18,111 without you changing a single position.

In late February 2020, a portfolio manager I know described his book the way a textbook would want him to: US large-cap equities, a slug of long Treasuries, some gold, a basket of investment-grade credit, a sliver of emerging-market stocks, and a small crypto position he kept "for the asymmetry." On paper it was beautifully diversified. The trailing twelve-month correlation across those sleeves averaged about 0.25 — low enough that the risk model said the whole thing should breathe gently, with the bonds and gold cushioning whatever the stocks did.

Three weeks later, between February 19 and March 23, the S&P 500 fell 33.9% peak-to-trough. That part he expected; equities crash. What he did not expect was that *everything else fell with them.* Long Treasuries had a few violent down days as leveraged players sold their most liquid asset to raise cash. Gold dropped roughly 12% in the worst week. Investment-grade credit gapped wider. Bitcoin lost half its value in two days. The VIX — the market's fear gauge — went from a sleepy 14 to an all-time intraday peak of 82.7. His "diversified" book had quietly turned into one enormous bet on a single thing: *risk-off.* The diversification he had paid for, in the form of lower long-run returns from holding bonds and gold, evaporated in the exact week he was counting on it.

This is not bad luck and it is not unique to 2020. It is the most important, least intuitive fact about correlation: it is a regime, not a constant, and the regime that matters — the crisis — is the one where correlations all rush toward +1. This post builds that idea from zero. We will define the VIX, define risk-on/risk-off, show with real numbers how average correlations spike, derive the portfolio-variance math that turns a spike into a dollar loss, explain the deleveraging mechanism that physically causes it, and finish with how to read the VIX as a regime switch you can actually use.

![Risk-on assets fan out and diversify, risk-off assets collapse onto one factor](/imgs/blogs/the-vix-risk-on-risk-off-and-the-correlation-spike-1.png)

## Foundations: the VIX, RORO, and conditional correlation

Three ideas underpin everything that follows. Let us define each from the ground up, with no finance background assumed.

### What is correlation, really?

Correlation measures how two things move *together*. The standard measure is the **Pearson correlation coefficient**, written r (or the Greek letter ρ, "rho"), which always sits between −1 and +1. A correlation of +1 means the two assets move in perfect lockstep: when one is up 2%, the other is up some fixed multiple of 2%, every time. A correlation of −1 means they move in perfect opposition. A correlation of 0 means knowing one tells you nothing about the other.

For a portfolio, correlation is not a curiosity — it is the whole game. The entire benefit of diversification comes from holding assets that are *less than perfectly correlated.* If you own two assets that always move together (r = +1), owning both is no safer than owning a double dose of one. If you own two assets that move independently (r = 0), their wiggles partly cancel and the combined ride is smoother. The series' companion post [What correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) builds the statistic itself; here we care about one property above all: **correlation is conditional.** The number you compute depends entirely on *which days* you include. Calm days and crisis days are drawn from different worlds, and averaging them together produces a single full-sample number that describes neither.

### The asymmetry: correlation is worse on the way down

There is one more property of conditional correlation that turns an inconvenience into a genuine trap: the spike is **asymmetric.** Correlations do not just rise in *any* large move — they rise far more in large *down* moves than in large *up* moves. This is the "downside correlation" or "asymmetric correlation" effect, and it is one of the best-documented anomalies in markets.

The reason is the mechanism we will build in detail later: the deleveraging spiral fires on the way down (margin calls, forced selling of everything liquid) but has no symmetric counterpart on the way up. When markets rip higher in a melt-up, individual stories reassert — the AI stock outruns the utility, the oil major outruns the bank — and correlations stay moderate. But when markets crash, the single risk-off factor swamps every story and correlations converge. So the very tool you rely on, diversification, is *most* impaired in exactly the scenario — a crash — where you most need it, and *least* impaired in the melt-up where you would happily take more correlation in exchange for the gains. The market gives you decorrelation when you want correlation and correlation when you want decorrelation. This asymmetry is precisely why a symmetric, full-sample correlation number is so dangerous: it averages the benign upside correlation with the malignant downside correlation and reports a comforting middle that exists on no actual day.

### What is the VIX?

The **VIX** is the Cboe Volatility Index. In one sentence: it is the market's estimate, expressed as an annualized percentage, of how much the S&P 500 is likely to move over the next 30 days, backed out from the prices of S&P 500 options. It is not a price you can buy directly; it is a calculation. When traders pay up for options — puts to protect against a fall, calls to bet on a bounce — option prices rise, and the VIX rises with them. That is why it is nicknamed the "fear gauge": when investors are nervous, they bid up insurance, and the VIX climbs.

The mechanics of how the VIX is computed from a strip of option prices belong to the options world; the deep version lives in the options-volatility series at [The VIX and vol products: VIX, VXX, UVXY, and the cost of the roll](/blog/trading/options-volatility/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll). For our purposes you need three facts:

1. **The VIX is in "vol points," which are annualized percent.** A VIX of 16 means options are pricing roughly a 16% standard-deviation move in the S&P over the *next year.*
2. **Its long-run average is about 19.5.** Most of the time it lives in the teens to low twenties.
3. **It spikes — violently — in a crisis,** typically to 3–4× its average, and those spikes are short-lived. Fear is mean-reverting; complacency is the default state.

To turn the VIX into something tangible, divide by the square root of the number of periods in a year. The expected one-month move is roughly VIX ÷ √12, and the expected one-day move is roughly VIX ÷ √252. We will use this below.

### What is risk-on / risk-off (RORO)?

**Risk-on / risk-off** describes the two moods markets oscillate between. In a **risk-on** mood, investors are comfortable, they reach for return, and they buy "risky" assets — stocks, high-yield bonds, emerging markets, crypto, commodities — while selling "safe" ones like Treasuries, the US dollar, and (sometimes) gold. In a **risk-off** mood, the reverse: they dump everything risky and flee to safety. The *why* and *how* of this rotation — which assets are the havens, how money physically moves — is covered in the mechanism series at [Risk-on/risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates). What matters *here* is the statistical consequence.

When the market is risk-on, each asset is free to follow its own fundamentals. A tech stock rises on good earnings; an oil producer rises on a supply shock; a bond rises when inflation cools. These are separate stories, so the assets are weakly correlated. But when the market flips risk-off, a single question swamps every fundamental: *do I want risk or not?* Suddenly the tech stock, the oil producer, the bond, the EM index, and the crypto coin are all answering the *same* question, in the same direction, on the same day. That shared answer is one dominant factor, and a single dominant factor is exactly what high correlation means.

That is the entire thesis in one paragraph: **risk-on lets assets decouple and diversification works; risk-off collapses them onto one factor and diversification fails.** The VIX is the gauge that tells you which regime you are in. Everything else in this post is putting numbers on it.

### Why "one dominant factor" *means* high correlation

It is worth slowing down on the link between "everyone is answering the same question" and "correlations are high," because it is the engine of the whole post and it is more than a metaphor — it is a piece of statistics.

Imagine each asset's daily return splits into two parts: a piece driven by a *common* factor (call it F — the market-wide risk-on/risk-off mood) and a piece that is *idiosyncratic* (call it ε — the asset's own news, its earnings, its supply shock). Write asset i's return as r_i = b_i × F + ε_i, where b_i is how sensitive asset i is to the common factor and the ε_i terms are independent of each other and of F. This is the simplest version of a "factor model," and it captures exactly the intuition above.

Now ask: what is the correlation between two assets, i and j, under this model? The only thing they *share* is the common factor F. Their idiosyncratic pieces ε_i and ε_j are independent, so they contribute nothing to the co-movement. The correlation between i and j therefore depends entirely on (a) how big the common factor's swings are relative to the idiosyncratic swings, and (b) how much each asset loads on the factor. When the common factor is *quiet* — calm markets, where each asset's day is dominated by its own news — the idiosyncratic pieces dominate, the shared part is small, and pairwise correlations are low. When the common factor is *loud* — a crisis, where the risk-off mood swamps every individual story — the shared part dominates, the idiosyncratic pieces become rounding errors, and pairwise correlations rush toward one.

This is the precise mechanism behind the spike. Nothing about the *assets* changed between calm and crisis. What changed is the *variance of the common factor* relative to the idiosyncratic variances. In calm markets, F is a gentle breeze and each asset sails on its own; in a crisis, F is a hurricane and every asset is just blown in the same direction. The VIX is, in effect, a real-time reading of how loud F has become. A VIX of 14 says the common factor is quiet — your assets are mostly trading on their own stories. A VIX of 80 says the common factor is deafening — there is only one story, and every asset is in it.

#### Worked example: how a louder factor lifts correlation

Put rough numbers on the factor model to see the mechanism bite. Suppose two assets each load equally on the common factor (b = 1) and each has its own idiosyncratic volatility of 15% per year. In calm markets the common factor's volatility is, say, 8%. Then each asset's total variance is the factor part (8%² = 64) plus the idiosyncratic part (15%² = 225), and the *shared* variance is just the 64 from the factor. The correlation is the shared piece over the total: 64 ÷ (64 + 225) = **0.22** — right in the calm-market range. Now a crisis hits and the common factor's volatility quadruples to 32% while the idiosyncratic 15% is unchanged. The shared variance is now 32%² = 1,024, and the correlation becomes 1,024 ÷ (1,024 + 225) = **0.82.** The assets did not change; the factor got 4× louder, and the correlation leapt from 0.22 to 0.82. The intuition: correlation is the *share of variance that is common,* so anything that amplifies the common factor — a panic, the VIX quadrupling — mechanically drags every pairwise correlation toward one.

## The correlation spike: from 0.25 to 0.80

Here is the centerpiece. If you take a basket of major risk assets — global equities, credit, commodities, EM, crypto — and compute the *average pairwise correlation* among them (the average r across every pair), you do not get a single number. You get a number that depends on the regime, and the dependence is dramatic.

![Average pairwise correlation rises from 0.25 in calm markets to 0.80 in a crisis](/imgs/blogs/the-vix-risk-on-risk-off-and-the-correlation-spike-2.png)

In a calm market, the average pairwise correlation across these assets runs around **0.25** — low enough that diversification does real work. In an ordinary selloff (a 5–10% equity dip, the kind that happens once or twice a year), it rises to around **0.45.** And in a genuine crisis — a deleveraging event, a liquidity scramble — it spikes toward **0.80,** close enough to one that the distinction between "a portfolio of ten assets" and "ten units of the same asset" nearly disappears.

Read that progression again, because it is the cruelest feature in all of finance. Diversification is the one free lunch the textbooks promise you — and it is served in full on the calm days when you do not need it, and snatched away in the crisis when you do. The cross-asset companion post [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) catalogs episode after episode of exactly this; the broader principle that this regime-dependence is the *defining* property of correlation is the spine of [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant).

Why "toward 0.80" and not all the way to 1.0? Because even in a panic, assets retain a flicker of their own identity — Treasuries do eventually catch a bid as the flight-to-quality reasserts, gold sometimes decouples once the forced selling is done. But 0.80 is more than enough to destroy a portfolio's risk budget, as the next section shows in dollars.

#### Worked example: reading the spike as a diversification count

A useful way to feel a correlation number is to convert it into an "effective number of independent bets." For an equal-weight book of N assets with average pairwise correlation ρ, the number of *truly independent* positions you effectively hold is roughly N ÷ (1 + (N−1)ρ). Take a 10-asset book. At ρ = 0.25, that is 10 ÷ (1 + 9 × 0.25) = 10 ÷ 3.25 ≈ **3.1 effective bets** — you thought you owned ten things, you really owned about three. At ρ = 0.80, it is 10 ÷ (1 + 9 × 0.80) = 10 ÷ 8.2 ≈ **1.2 effective bets** — you own, for all practical purposes, *one* thing. The intuition: as correlation climbs, your ten carefully chosen positions silently merge into a single trade you never meant to put on.

## The diversification illusion: what the spike costs in dollars

A correlation number is abstract. A portfolio's volatility in dollars is not. Let us connect them with the one equation every investor should know.

For an equal-weight portfolio of N assets, each with the same individual volatility σ (sigma, the standard deviation of returns), and with the same average pairwise correlation ρ between every pair, the portfolio's volatility collapses to a clean closed form:

```
sigma_p = sigma * sqrt( 1/N + (1 - 1/N) * rho )
```

Stare at the term in the square root. It has two pieces. The 1/N piece is the part that *diversifies away* — it shrinks as you add more assets. The (1 − 1/N) × ρ piece is the part you *cannot* diversify away, because it scales with the correlation ρ. As ρ rises, the second term swells and swallows the portfolio's volatility. In the limit ρ → 1, the whole bracket goes to 1 and sigma_p → σ: a thousand assets behave exactly like one. That single term is why the crisis is so dangerous — it is not that the assets get more volatile (though they do); it is that the *correlation* term in this formula explodes.

![Portfolio volatility rises steeply as average correlation increases](/imgs/blogs/the-vix-risk-on-risk-off-and-the-correlation-spike-5.png)

Plot that formula across the full range of ρ, holding N and σ fixed, and you get a curve that climbs relentlessly: portfolio volatility is *lowest* when correlation is low and rises toward the single-asset volatility as correlation approaches one. The three regime dots — calm, normal selloff, crisis — sit on that curve at exactly the levels from the correlation-spike chart above. The crisis dot is not a different curve; it is the same portfolio, sliding up the same line, because the market moved its correlation, not its holdings.

#### Worked example: a \$100,000 book at 0.25 vs 0.80 correlation

Take a concrete book: \$100,000 spread equally across N = 10 assets, each with a 20% annual volatility (σ = 0.20). Run the formula at both regimes.

In the calm regime (ρ = 0.25): sigma_p = 0.20 × √(0.10 + 0.90 × 0.25) = 0.20 × √0.325 = **11.40%.** A one-standard-deviation year is therefore ±\$11,402 on the \$100,000 book. That is the risk you signed up for, the risk your risk model reported, the risk you sized your positions around.

In the crisis regime (ρ = 0.80): sigma_p = 0.20 × √(0.10 + 0.90 × 0.80) = 0.20 × √0.82 = **18.11%,** or ±\$18,111. You did not buy a single new position. You did not lever up. The *market* changed your correlation from 0.25 to 0.80, and in doing so it inflated your portfolio's risk by 59% — your one-sigma exposure jumped from \$11,402 to \$18,111, an extra **\$6,709** of risk you never chose. In *variance* terms (volatility squared, the quantity that actually adds up), the crisis book carries 2.52× the variance of the calm book. The takeaway: a crisis does not just make your assets fall — it secretly more-than-doubles the variance of the very portfolio you built to be safe.

#### Worked example: the diversification you *thought* you had

Run the same book one more time at the diversification ideal — ρ = 0, perfectly independent assets. sigma_p = 0.20 × √0.10 = **6.32%,** or ±\$6,325 on \$100,000. That \$6,325 is the dream the brochure sold: ten independent 20%-vol assets blending down to a 6.3%-vol portfolio. Reality at ρ = 0.25 already nearly doubles it to \$11,402. The crisis at ρ = 0.80 nearly triples the dream to \$18,111 — which is 91% of the way to just owning a single one of the assets (\$20,000 of one-sigma risk at σ = 20%). In one sentence: by the time the VIX is screaming, you are carrying almost the full risk of a single undiversified bet while still paying the long-run return drag of holding "diversifiers" that quit on you. This is why the cross-asset note [Correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) calls the free lunch "conditional" — you eat it on calm days and skip it on the day it matters.

## The VIX as a regime switch

The correlation spike is the disease; the VIX is the thermometer. The reason traders watch the VIX so obsessively is that it is a remarkably good real-time read on *which correlation regime you are in,* available continuously, before the correlations themselves have finished spiking.

![The VIX year-end level with calm, transition, and crisis bands](/imgs/blogs/the-vix-risk-on-risk-off-and-the-correlation-spike-7.png)

Roughly speaking there are three bands. **Below ~20** is the calm, risk-on regime: assets follow their own stories, average correlations sit near 0.25, and diversification does its job. **Between ~20 and ~30** is a transition zone — nerves are fraying, correlations are creeping up, and the market is one bad headline away from flipping. **Above ~30** is the risk-off regime: fear dominates, the single risk-on/risk-off factor takes over, and average correlations march toward 0.80. The exact thresholds are fuzzy and drift over time, but the *existence* of a regime switch around the high-teens-to-thirty range is one of the most robust patterns in markets.

You can see the switch directly in the year-end VIX levels: most years close in the teens (2017 closed at 11.0, 2019 at 13.8, 2023 at 12.5), but the crisis years carry the scars (2018 closed at 25.4 after February's Volmageddon; 2020 at 22.8 after the COVID crash; 2022 at 21.7 amid the rate shock). Overlay the intra-year *spikes* and the picture is stark: those panic days punched to 37.3 (Feb 2018), 82.7 (March 2020), and 65.7 (August 2024) — far above any year-end close, because the spikes are violent and brief.

#### Worked example: the VIX-implied expected move in dollars

The VIX number itself is an expected move, so you can price the regime directly. Recall the one-month expected one-sigma move ≈ VIX ÷ √12. On a calm day with the VIX at its 19.5 average, that is 19.5 ÷ 3.46 = **5.6%** — over the next month, the S&P has a roughly two-in-three chance of staying within ±5.6%. On a \$100,000 S&P position, that is a ±\$5,633 one-sigma swing. Now take March 2020, VIX at 82.7: the implied one-month move is 82.7 ÷ 3.46 = **23.9%,** or ±\$23,873 on the same \$100,000 — *more than four times* the calm-day risk, priced into the options market in real time. The VIX is not predicting the *direction* of the move; it is telling you the *size,* and in a crisis the size alone is enough to force the deleveraging that drives correlations to one. The intuition: when the VIX quadruples, the option market is telling you the daily moves are about to quadruple too, and a portfolio sized for 1% days cannot survive 5% days.

### The VIX term structure as a stress tell

There is a second, subtler signal in the VIX complex: the **term structure** — the relationship between the VIX (30-day implied vol) and VIX futures expiring further out. In normal, calm markets the curve is in **contango**: near-term vol is *cheaper* than far-term vol, because the market expects today's calm to give way to some future uncertainty. The curve slopes up.

In a crisis the curve inverts into **backwardation**: near-term vol becomes *more expensive* than far-term, because the panic is *now* and the market expects it to fade. A flip from contango to backwardation is one of the cleanest "the regime just switched" signals available — it says the fear is immediate, not theoretical. The full mechanics of the curve, and how to trade it, live in [The term structure of volatility: contango, backwardation, and the VIX curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve). For the correlation story, the term-structure inversion is simply an early, market-priced confirmation that you have crossed from the 0.25 world into the 0.80 world.

## The deleveraging mechanism: why correlations are *forced* to one

So far we have described *that* correlations spike. The deeper question is *why* — and the answer is not psychology, it is plumbing. The correlation spike is manufactured by a specific, mechanical chain: the deleveraging spiral.

![A volatility shock triggers margin calls and forced selling that drives correlations to one](/imgs/blogs/the-vix-risk-on-risk-off-and-the-correlation-spike-6.png)

It runs like this. A shock hits — a bad inflation print, a default, a geopolitical rupture. Volatility jumps and the **VIX spikes** from, say, 20 to 50. Now two things happen at once, and both point the same direction. First, **margin desks raise haircuts**: a broker who lent you money to hold positions sees volatility explode and demands more collateral, *right now.* Second, **risk models breach their limits**: nearly every large fund runs a Value-at-Risk (VaR) budget that scales with volatility, so when volatility doubles, the model mechanically orders the fund to cut exposure to stay inside its risk limit. Both forces command the same action: **sell.**

Here is the crucial, non-obvious step. When you are forced to raise cash fast, you do not sell what has fallen the most — those positions are already marked down and may be illiquid. You sell **what is liquid:** your large-cap stocks, your Treasuries, your gold, your most-traded credit. You sell your *best* assets, not your worst, because liquidity is what you need. And every leveraged player in the market is doing the *same thing at the same time,* because they are all running similar VaR models and facing similar margin calls. The market briefly has **one seller, one trade:** sell everything liquid to raise cash.

That single dominant seller is, mechanically, a single dominant factor — which is exactly the definition of high correlation. The assets are not falling together because their fundamentals suddenly aligned; they are falling together because the *same forced seller* is hitting all of them simultaneously. And then the spiral feeds itself: the selling causes more losses, the losses trigger more margin calls and more VaR breaches, which force more selling. Correlations do not drift to 0.80 — they are *dragged* there by the plumbing of leverage. This is the through-line of the dedicated crisis post [Correlation during crises: when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails), which walks the 1998, 2008, and 2020 episodes in detail.

This is also why **liquidity** is the hidden variable underneath the VIX. When central banks flood the system with liquidity, the deleveraging spiral is easier to arrest and correlations relax faster; when liquidity drains, the spiral runs hotter. The everything-falls-together-then-everything-rallies-together dynamic is really the liquidity tide moving every boat at once — the subject of [Global liquidity and the everything correlation](/blog/trading/macro-correlations/global-liquidity-and-the-everything-correlation).

### The VaR feedback loop, made concrete

The most under-appreciated link in the spiral is the **Value-at-Risk feedback loop,** because it is automatic, system-wide, and pro-cyclical by design. Value-at-Risk is the dominant risk-budgeting tool at banks and many funds: a fund sets a dollar VaR limit — "we will not risk losing more than X on a normal bad day" — and then sizes positions so the *estimated* loss stays under that limit. The estimate is built from recent volatility and correlation. And here is the trap: VaR scales with volatility, so when volatility doubles, the *same positions* suddenly report double the VaR, blowing through the limit even though nothing about the holdings changed. The model's only remedy is to cut exposure — to sell — until the reported VaR is back under the cap.

Now picture every large fund running a similar model. A vol shock doubles everyone's reported VaR at once. Everyone is ordered to sell at once. The selling pushes prices down and volatility *up,* which raises VaR *again,* which forces *more* selling. The risk system built to *contain* losses becomes the *amplifier* of them — and because every fund is selling the same liquid assets simultaneously, the selling shows up as a correlation spike. This is why deleveraging is mechanical rather than emotional: even a perfectly unemotional, rule-following risk manager is *required* by the model to sell into the panic.

#### Worked example: the VaR cut that forces a sale

Take a fund with a \$10,000,000 book and a one-day VaR limit of \$200,000 (it will not risk more than 2% on a normal bad day). In calm markets its model estimates daily portfolio volatility at 1.0%, so its one-sigma daily risk is \$100,000 and its (roughly two-sigma) VaR is about \$200,000 — right at the limit, fully invested. Now a shock doubles estimated daily volatility to 2.0%. The *same* \$10,000,000 book now reports a VaR of about \$400,000 — twice the limit. To get back under \$200,000, the model orders the fund to halve its risk, which means selling roughly \$5,000,000 of positions into a falling market. Multiply that forced \$5,000,000 sale across hundreds of similar funds and you have the deleveraging wave that drives correlations to one. The intuition: a volatility doubling does not just *predict* selling — it *commands* it, because the risk model mechanically halves the position the moment vol doubles.

## Measuring the spike: how to actually see it in the data

If the correlation spike is real, you should be able to *measure* it, not just assert it. There are three honest ways to look at conditional correlation, and each one tells the same story in a different language.

**Rolling-window correlation.** The simplest tool: instead of computing one correlation over your whole history, compute it over a moving window — say the last 60 trading days — and slide the window forward day by day. You get a *time series* of correlation that rises and falls with the regime. In every crisis, this rolling line spikes upward, often from the 0.2–0.3 range into the 0.7–0.8 range, within a few weeks. The catch is the window length: a short window (20 days) is jumpy and reacts fast but is noisy; a long window (250 days) is smooth but lags the regime change by months, so it can still be reporting "calm-market correlation" deep into a crisis. There is no perfect window, which is the whole point of [Rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters): the number you report is a choice, and that choice can hide or reveal the spike.

**Exceedance (tail) correlation.** A sharper tool, built specifically for the question we care about. Instead of one correlation over all days, compute the correlation *conditional on big moves* — for example, only on the days when the market fell more than 2%. This is called exceedance or tail correlation, and it isolates exactly the crisis days that a full-sample number drowns out. The empirical finding, again and again, is stark: tail correlation is far higher than full-sample correlation. Two assets with a benign 0.3 average correlation routinely show 0.7+ correlation *on the down days that matter.* This is the statistical fingerprint of the asymmetry — assets diversify on ordinary days and converge on the bad ones — and it is why a single full-sample r is not just incomplete but actively misleading.

**Correlation conditioned on the VIX level.** The most decision-useful framing for this post: bucket your history by the VIX level and compute the average pairwise correlation in each bucket. The result is essentially the three-bar chart at the top of this post — low VIX maps to ~0.25, mid VIX to ~0.45, high VIX to ~0.80. Conditioning on the VIX turns the fear gauge into a direct read on your *effective* diversification, which is what makes it a usable real-time signal rather than a post-mortem statistic.

#### Worked example: how much your rolling window can lie

Suppose true correlation is 0.25 for 200 days, then jumps to 0.80 and stays there. A 250-day rolling window computes correlation over the *last* 250 days — so on the very first crisis day, 200 of those 250 days are still calm-market days. The window's reported correlation is a blend, roughly (200 × 0.25 + 50 × 0.80) ÷ 250 ≈ **0.36** — barely moved, even though true correlation is already 0.80. A trader relying on that long window would size a \$100,000 book as if portfolio vol were ~12% (\$12,000-ish of one-sigma risk) when it is actually ~18% (\$18,111). The long window does not just lag — it *understates the live crisis risk by half a turn of leverage* for weeks. The intuition: the smoother your correlation estimate, the longer it lies to you in exactly the regime where the truth is most expensive.

## Common misconceptions

A handful of plausible-sounding beliefs about diversification and the VIX are dangerously wrong. Each one has cost real money.

**Myth 1: "I'm diversified, so a crash can't really hurt me."** This is the deadliest one. Diversification is measured by your *calm-market* correlations, but it is *spent* at your *crisis* correlations. The portfolio that looks like ±\$11,402 of one-sigma risk at ρ = 0.25 is actually a ±\$18,111 portfolio at ρ = 0.80. Your diversification does not protect you in the crash; it quietly disappears at the moment you reach for it. The honest framing: you are diversified against ordinary wiggles, and undiversified against panics.

**Myth 2: "Bonds always hedge stocks, so 60/40 is safe."** The stock-bond correlation is *itself* a regime that flips. It was reliably negative (bonds hedged stocks) for two decades, but it flipped strongly positive in the 2022 inflation shock, when both fell together. The full story is in [The stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime). Even when stocks and bonds are negatively correlated on *average,* there are crisis days inside a deleveraging where forced sellers dump Treasuries *too* — the March 2020 Treasury sell-off being the textbook case. A hedge that works 95% of the time can fail on exactly the 5% of days you bought it for.

**Myth 3: "A low VIX means the market is safe."** A low VIX means the market is *calm,* which is not the same thing. The lowest-VIX environments are often the most fragile, because complacency lets leverage build up — and that leverage is the fuel for the next deleveraging spiral. The 2018 Volmageddon detonated *from* a VIX in the low teens. A low VIX tells you the deleveraging hasn't started yet; it tells you nothing about how much leverage is waiting to unwind.

**Myth 4: "Gold is a crisis hedge, so it will go up in the crash."** Sometimes — eventually — but not on the worst deleveraging days. Gold is liquid, which makes it a *prime* asset to sell in a margin scramble. In the worst week of March 2020 gold fell roughly 12%, *with* equities, before recovering later. In a true risk-off panic, the relevant question for an asset is not "is it a long-run hedge?" but "is it liquid enough to be force-sold for cash?" — and the most liquid hedges get hit first. (Gold's real driver in normal times is real yields, not crisis fear; see [Inflation and gold: the real yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story).)

**Myth 5: "The full-sample correlation is the correlation."** A single number computed over ten years blends calm days and crisis days into a meaningless average. Two assets can show a benign 0.3 full-sample correlation and a terrifying 0.85 crisis correlation. The full-sample figure understates exactly the risk you most need to size for. Correlation must always be reported *conditional on the regime,* a point hammered home in [Rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters).

## How it shows up in real markets

The pattern is not theoretical. Here are three episodes where the VIX flagged the switch and the correlation spike did the damage, plus the long history of VIX crisis peaks.

![VIX peaks in past crises versus the long-run average](/imgs/blogs/the-vix-risk-on-risk-off-and-the-correlation-spike-3.png)

The crisis-peak chart sets the scene: against a long-run average of 19.5, the VIX peaked at 45.7 in the 1998 LTCM/Russia crisis, 80.9 in the 2008 global financial crisis, 82.7 in the 2020 COVID crash, and 65.7 in the August 2024 yen-carry unwind. Every one of those spikes coincided with the average pairwise correlation across risk assets jumping toward 0.80. The VIX and the correlation spike are two readings of the same underlying event: a forced, system-wide deleveraging.

### LTCM, 1998: the original correlation-to-one disaster

Long-Term Capital Management was a hedge fund run by Nobel laureates whose entire strategy was built on the assumption that the correlations they measured in calm markets would hold. They put on dozens of supposedly *unrelated* trades — convergence bets across different countries, instruments, and asset classes — and levered them heavily, reasoning that with so many independent positions, diversification would keep total risk small. The calm-market correlations said the trades were independent; the risk model said the leverage was safe.

Then Russia defaulted in August 1998, the VIX spiked to 45.7, and the deleveraging began. LTCM discovered, all at once, that its "independent" trades were not independent at all in a crisis — they were all, secretly, the *same* trade: a bet that liquidity and risk appetite would hold. When liquidity vanished, every position moved against them *together,* their carefully diversified book behaving like one giant leveraged bet on risk-on. The fund lost about \$4.6 billion in a few months and had to be rescued by a consortium of banks to prevent its forced liquidation from spiraling through the system. LTCM is the canonical lesson of this entire post: the correlations you measure in the calm are not the correlations you face in the crisis, and the more leverage you stack on the calm-market number, the more violently the crisis correction arrives.

### COVID, March 2020: everything fell together

The cleanest modern example. The VIX rose from ~14 in mid-February to its all-time intraday peak of 82.7 on March 16, 2020. As the deleveraging ran, the single risk-off factor took over completely: the S&P 500 fell 33.9% peak-to-trough, but so did almost everything else. Gold dropped ~12% in its worst week; investment-grade and high-yield credit gapped wider together; emerging-market equities and Bitcoin (which lost ~50% in two days) traded as pure high-beta risk. A textbook-diversified book behaved like a single leveraged equity bet, exactly as the ρ = 0.80 formula predicts.

The most instructive piece was the Treasury market — the supposed ultimate safe haven. For a few days in mid-March, long Treasuries *fell with stocks,* the cardinal sin a hedge is not supposed to commit. The reason was pure deleveraging: a large, heavily levered "Treasury basis trade" run by hedge funds blew up, and to meet margin calls those funds had to sell their most liquid asset — Treasuries — into the panic. The flight-to-quality bid that normally lifts Treasuries in a crash was, briefly, overwhelmed by forced sellers dumping them. Bonds did eventually catch their safe-haven bid as the dust settled, but the lesson is permanent: even the safest hedge can fall *with* the book on the worst deleveraging days, because on those days the question is not "is it safe?" but "is it liquid enough to be force-sold?" The selloff stopped only when the Fed flooded the system with liquidity on March 23 — buying Treasuries directly, backstopping credit, and re-liquefying the plumbing — at which point the *correlation worked in reverse,* and everything rallied together for two years.

### The 2018 Volmageddon: a crisis born of calm

On February 5, 2018, the VIX more than doubled in a single day, closing around 37 after a 2017 spent mostly below 12. There was no economic catastrophe behind it. Instead, a popular trade — selling volatility through products like the XIV note, which profited as long as the VIX stayed low and the term structure stayed in contango — had grown enormous during the long calm. When the VIX ticked up, those short-vol products were *mechanically forced to buy* volatility to cover, which spiked the VIX further, which forced more buying. The XIV note lost about 96% of its value in a day and was liquidated. The lesson for our story: the lowest-volatility regimes breed the leverage that fuels the next spike. A low VIX is not the absence of risk; it is the accumulation of it. (How short-vol and the curve interact is in [The VIX and vol products](/blog/trading/options-volatility/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll).)

### August 2024: the yen-carry unwind

On August 5, 2024, the VIX spiked intraday to 65.7 — its third-highest reading on record, behind only 2008 and 2020 — even though there was no recession or banking failure. The trigger was a *leverage* unwind: for years, traders had borrowed in cheap yen to buy higher-yielding assets worldwide (the "carry trade"). When the Bank of Japan hiked and the yen surged, that trade went underwater, and the forced unwind hit everything at once. Japanese equities crashed, US tech sold off, crypto dropped, and the correlation across risk assets spiked toward one — for a few days — before liquidity and calmer heads reversed it. This episode is the purest illustration of the mechanism: *no fundamental news, just forced deleveraging,* and yet the correlation spike was as real as in any recession.

### The damage, in one chart

![S&P 500 peak-to-trough drawdowns across the big risk-off regimes](/imgs/blogs/the-vix-risk-on-risk-off-and-the-correlation-spike-4.png)

Stack the equity damage from the big regimes and you see why the correlation spike matters so much: the S&P fell 56.8% in the 2008 GFC, 33.9% in the 2020 COVID crash, and 25.4% in the 2022 rate shock. The point is not just that stocks fell — it is that during the worst of each of these, *the rest of a diversified book fell with them,* because correlations had spiked. The drawdown chart is the equity leg of a loss that, in a high-correlation regime, the whole portfolio shared. A 60/40 portfolio lost about 16% in 2022 precisely because the stock leg and the bond leg fell together that year — diversification that would have cushioned the equity loss in a normal selloff did not show up.

#### Worked example: the \$ drawdown when correlations spike

Tie it together on the \$100,000 book. Suppose your risk model, calibrated on calm-market correlations (ρ = 0.25, portfolio vol 11.4%), tells you a "really bad" two-sigma year is about −22.8%, or −\$22,804. You size your positions and your stomach for that. Now the crisis hits, ρ jumps to 0.80, and your *actual* portfolio vol is 18.1%. The same two-sigma move is now −36.2%, or **−\$36,222.** The loss you prepared for was \$22,804; the loss the regime can deliver is \$36,222 — an extra **\$13,418** of drawdown, roughly 13% of the entire book, generated purely by the correlation spike and not by any change in your holdings. The intuition: a risk budget built on calm-market correlations understates your true crisis drawdown by more than half a turn of leverage, which is why "I sized for a bad year" is not the same as "I sized for a *crisis.*"

## What actually survives a correlation spike

If almost everything correlates to one in a crisis, is diversification simply a lie? No — but the things that genuinely diversify a crisis are not the things most portfolios own, and they share an uncomfortable trait: they *cost you* in the calm. That cost is not a flaw; it is the price of a payoff that shows up precisely when correlations spike. Three categories survive the spike, and understanding *why* sharpens the whole framework.

**Long volatility itself.** The one asset that is *defined* to rise when the VIX rises is volatility. Owning long-dated put options on the index, or a structured long-vol position, is the only "asset" whose payoff is mechanically linked to the correlation spike, because the spike and the vol spike are the same event. The catch is brutal: long vol *bleeds* every calm day through time decay — you pay a premium that erodes day after day, year after year, while the crisis you are insuring against does not come. This is the mirror image of the 2018 Volmageddon short-vol blow-up: the short-vol sellers collected steady premium for years and then lost everything in a day; the long-vol buyer pays steadily for years and then collects everything in a day. There is no free crisis hedge — only insurance whose premium you pay in advance.

**Trend-following / managed futures.** Systematic trend strategies tend to be *short* the market by the time a crisis is in full swing, because the crash itself is a down-trend they have already begun to follow. Their crisis payoff is not guaranteed — a sudden, gap-down crash (a "V-shaped" shock with no preceding trend) can catch them flat — but over many crises they have provided genuine, positive, *crisis-period* returns that are uncorrelated with a long book. They, too, pay a cost: in choppy, trendless calm markets they grind out small losses, and investors abandon them right before they pay off.

**Cash and the very front of the curve.** The most unglamorous diversifier is the one that works most reliably: cash. Short-dated Treasury bills and cash do not get force-sold (they *are* the cash everyone is scrambling for), so their correlation to the risk book in a crisis is essentially zero. The price is obvious — cash earns little and drags your long-run return — but in the worst week of a deleveraging, the only position whose value you can be certain of is the one denominated in the very thing the forced sellers are desperate to raise.

Notice the pattern: every genuine crisis diversifier is *negatively carried* — it costs you a little, continuously, in the calm. That is not a coincidence. If an asset paid you positively in the calm *and* protected you in the crisis, everyone would own it, its price would be bid up, and that free protection would vanish. The market does not give away crisis insurance. The honest version of "diversification" is therefore: spread risk across assets to smooth the *ordinary* ride (the ρ = 0.25 world), and *separately* pay for genuine tail protection to survive the *crisis* (the ρ = 0.80 world). Conflating the two — believing your ordinary diversification *is* your crisis protection — is the mistake at the heart of every "I was diversified and still got crushed" story.

## How to read it and use it

You cannot stop correlations from spiking. But you can stop being *surprised* by it, and you can build a process around the regime switch. Here is the playbook.

**1. Treat the VIX as a regime indicator, not a price.** You are not trying to forecast the VIX; you are trying to read *which regime you are in.* The simple version: VIX below ~20 → assume the calm regime, where your diversification is real. VIX above ~30 → assume the crisis regime, where your effective number of bets has collapsed toward one. The 20–30 zone is "pay attention; the switch is live." Pair the level with the **term structure**: a flip from contango to backwardation is your confirmation that the switch has flipped.

**2. Stress-test your portfolio at crisis correlations, not calm ones.** This is the single most valuable habit. Before you trust your risk model, re-run it with the average pairwise correlation forced to 0.80 and see what your portfolio volatility and drawdown become. On the \$100,000 book that means planning for \$18,111 of one-sigma risk and a \$36,222 two-sigma drawdown — not the \$11,402 and \$22,804 the calm model reports. If the crisis number is more than you can survive, you are over-sized *now,* in the calm, when you can still do something about it.

**3. Know which of your "diversifiers" are actually liquidity, not hedges.** In a deleveraging, the most liquid assets get sold first, so liquid "hedges" (gold, long Treasuries, large-cap defensives) can fall *with* the book on the worst days. The hedges that hold up in a crisis tend to be the ones that *cost you* in the calm — long-dated puts, trend-following overlays, deeply out-of-the-money tail protection — precisely because everyone wants them only when it is too late. There is no free crisis hedge; there is only insurance you pay for in advance.

**4. Watch credit spreads as the corroborating canary.** The VIX is the equity-option read on fear; credit spreads are the bond-market read on the same thing, and they tighten toward each other in a crisis. When the VIX and high-yield spreads are *both* spiking, the deleveraging is real and broad; when the VIX spikes but spreads stay calm, the stress may be contained to equities. The credit side of this signal is the subject of [Credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary).

**5. Size positions to a "crisis correlation budget," not a "calm correlation budget."** Concretely: decide how much you can lose in a *crisis,* then back out how big your book can be given crisis-regime correlations — and hold that size in the calm. Most blow-ups happen because a fund sizes to its calm-market risk, looks well within its limit, and then watches the same positions breach the limit by a wide margin the instant correlations spike. Sizing to the crisis number means you carry *less* in the calm than your risk model says you could — which feels like leaving return on the table right up until the day it saves the fund.

#### Worked example: sizing to survive the spike

Say your hard limit is "never risk more than a 30% drawdown" on a \$100,000 book, defined as a two-sigma year. If you size using calm-market correlations (portfolio vol 11.4%), a two-sigma year is 22.8%, comfortably inside 30%, so the calm model says you could even add risk. But size using the crisis correlation (portfolio vol 18.1%) and a two-sigma year is 36.2% — already *past* your 30% limit. To respect the limit at crisis correlations you must shrink the book until 2 × portfolio vol ≤ 30%, i.e. portfolio vol ≤ 15%, which means cutting your at-risk capital by roughly a fifth versus what the calm model permits. That ~\$20,000 of capital you *don't* deploy is the price of surviving the spike — invisible and annoying in the calm, decisive in the crisis. The intuition: the right position size is the one that respects your drawdown limit at ρ = 0.80, not the one that looks merely "within limits" at ρ = 0.25.

**6. Remember that the correlation works in reverse, too.** The same plumbing that drags correlations to one on the way down drags them to one on the way *up* once liquidity returns — the post-crisis "everything rally." After March 2020, the Fed's liquidity flood sent every risk asset up together for two years. So the regime switch is not only a warning to de-risk; it is also, eventually, a signal that the all-correlated rebound is coming. Reading the liquidity tide that drives both directions is the everything-correlation story.

**What invalidates the signal?** Two things. First, a VIX spike with *no* deleveraging — a brief, isolated equity wobble where margin desks and VaR models do not actually force selling — can fade without a correlation spike; not every VIX pop is a crisis. Second, structural shifts in *who* holds the leverage (e.g., post-2008 bank deleveraging moving risk to less-levered hands) can change how violently a given VIX level translates into a correlation spike. The thresholds are heuristics, not laws. The discipline is to *check the regime,* size for the crisis correlation, and never confuse a calm-market diversification number for protection you actually have.

The deepest lesson of the VIX and the correlation spike is a humbling one. The diversification that every textbook promises is real — on the days you do not need it. On the day you do, the market collapses your careful, decorrelated book onto a single factor, and the VIX is the gauge that tells you it is happening. You cannot outlaw the spike. But you can size your risk for the 0.80 world instead of the 0.25 world, watch the fear gauge for the switch, and pay in advance for the few hedges that actually hold when everything else goes to one.

## Further reading & cross-links

Within this series:

- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the foundational idea that the number you compute depends entirely on the regime.
- [Correlation during crises: when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails) — the 1998 / 2008 / 2020 deleveraging walkthrough.
- [Credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary) — the bond-market read on the same fear the VIX measures.
- [The stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) — why even the classic hedge is itself a regime that flips.
- [Global liquidity and the everything correlation](/blog/trading/macro-correlations/global-liquidity-and-the-everything-correlation) — the liquidity tide that moves every asset together in both directions.
- [The macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — the full map of which driver moves which asset.

Mechanism and cross-asset context:

- [Risk-on/risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the *why* and *how* of the rotation between risky and safe assets.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — the cross-asset catalog of the spike.
- [Correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) — why the free lunch is conditional on the regime.

The VIX itself, in depth:

- [The VIX and vol products: VIX, VXX, UVXY, and the cost of the roll](/blog/trading/options-volatility/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll) — how the fear gauge is built and traded.
- [The term structure of volatility: contango, backwardation, and the VIX curve](/blog/trading/options-volatility/the-term-structure-of-volatility-contango-backwardation-and-the-vix-curve) — the stress tell in the shape of the curve.
