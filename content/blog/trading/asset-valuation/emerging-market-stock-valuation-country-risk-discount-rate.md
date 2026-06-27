---
title: "Valuing Emerging Market Stocks: Country Risk, Currency, and Discount Rate Adjustments"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "How to adapt DCF, P/E, and WACC for emerging market stocks, where country risk, currency volatility, political risk, and thin liquidity bias developed-market models."
tags: ["valuation", "asset-pricing", "emerging-markets", "country-risk-premium", "discount-rate", "currency-risk", "dcf", "capm", "liquidity-discount"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Valuing an emerging market stock is a developed-market valuation plus four disciplined adjustments: a country risk premium in the discount rate, a corrected beta, a currency that matches the cash flows, and a liquidity discount — with political risk priced through scenarios rather than a fudge factor.
>
> - The **country risk premium (CRP)** is the extra return investors demand for a country's default and equity risk. Damodaran's formula: CRP = country default spread × (σEquity / σBond). Vietnam ≈ 5.5%, India ≈ 2.2%, Brazil ≈ 5.4%, China ≈ 3.0% in 2024.
> - **Currency and discount rate must agree.** Forecast in the local currency and discount at a local-currency rate, OR forecast in dollars and discount at a dollar rate. Mixing them double-counts inflation and overstates value.
> - **Beta is unreliable in thin markets.** Low trading volume drags observed beta toward zero; correct it with the Blume adjustment or a fundamental beta before you use it.
> - The one number to remember: a Vietnamese stock with a US risk-free rate of 4.3%, a 5.5% global equity premium, a beta of 1.2, and a 5.5% country risk premium needs roughly a **17.6% required return** — more than double a comparable US stock.

In January 2024, two analysts valued the same consumer-goods company. One worked in London and ran a textbook discounted cash flow: he forecast cash flows, discounted them at a 9% cost of equity, and arrived at a price he liked. The other worked in Ho Chi Minh City, valuing the company's listed Vietnamese subsidiary. She used the same revenue growth, the same margins, the same cash conversion — and arrived at a value 40% lower. Neither made an arithmetic error.

The gap was entirely in the discount rate and the currency. The London analyst's 9% was a developed-market number that quietly assumed an investor could pull money out whenever they liked, that the currency would hold, that the courts would enforce a contract, and that the government would not change the rules. None of those assumptions are free in an emerging market — and each one has a price you can compute.

This post is about computing those prices. We will take a standard valuation and add the adjustments that turn a developed-market model into an emerging-market one, with real 2024 numbers throughout.

![EM valuation adjustment framework pipeline](/imgs/blogs/emerging-market-stock-valuation-country-risk-discount-rate-1.png)

## Foundations: why emerging market valuation is genuinely different

Start with what "emerging market" even means for a valuer. The label is an investability classification — MSCI and FTSE decide it based on market size, accessibility, and openness to foreign capital — but underneath the label sit several distinct risks that a developed-market model assumes away. A "discount rate" is the annual return an investor demands to hold a risky asset; it is the number you divide future cash flows by to get their value today. A higher discount rate means a lower value, because future money is worth less when the wait is riskier. Every adjustment in this post ultimately changes either the discount rate or the cash flows it acts on.

Before listing the risks, it is worth being precise about what we are *not* claiming. We are not claiming emerging-market companies are worse businesses — many are excellent, fast-growing, and well-run. We are not claiming you should avoid them — at the right price, the higher required return is exactly the compensation that makes them attractive. The claim is narrower and entirely mechanical: a valuation model calibrated on developed-market data carries hidden assumptions that, left uncorrected, systematically *overvalue* emerging-market stocks, because the model gives you the upside of higher growth without charging you for the extra risk. The job is to put that charge back in, explicitly, so that the price you compute reflects the real bargain on offer.

Here are the risk dimensions that a London-style model leaves out.

**Political and policy risk.** Governments in emerging markets change tax regimes, impose price caps, nationalize industries, and rewrite contracts more often than in developed markets. In 2022, several governments imposed windfall taxes on energy and banking. A company's cash flows are not just a function of its business — they are a function of whether the rules stay put.

**Currency risk.** An emerging market currency can lose 20% in a quarter. The Turkish lira fell from roughly 8 per dollar in early 2021 to over 35 per dollar by late 2024. If your cash flows are in lira and your investors think in dollars, that fall is a direct loss of value that has nothing to do with the company.

**Liquidity risk.** Many emerging market stocks trade thinly. A position you can exit in seconds in New York might take days to unwind in a frontier market, and selling pressure moves the price against you. Illiquidity is a real cost, and buyers demand a discount for it.

**Governance and information quality.** Accounting standards, audit quality, and disclosure are weaker on average. Related-party transactions, opaque ownership, and minority-shareholder expropriation are more common. You are valuing cash flows you can see less clearly, and a slice of them may never reach you.

**Short market history.** A developed-market beta or equity risk premium rests on a century of data. Many emerging markets have twenty noisy years, often including a crisis that dominates the sample. Your statistical estimates are simply less trustworthy.

It helps to see how these risks differ from ordinary business risk. A developed-market valuation already accounts for business risk — the chance that sales fall, margins compress, or a product fails. CAPM captures the part of that risk you cannot diversify away through the beta term. What emerging-market risk adds is a *layer on top of the business*: risks that hit a company not because of anything it did, but because of where it is domiciled. A perfectly run Argentine retailer and a perfectly run Argentine bank both suffer when the peso collapses or the government freezes prices. That shared, country-level exposure is what the country risk premium is designed to capture, and it is why you cannot diversify it away by simply holding more emerging-market stocks within the same country.

There is also an information dimension that compounds everything. In a developed market you can usually trust the reported numbers: audited statements, enforced disclosure rules, deep analyst coverage. In many emerging markets, the financial statements are a starting point for investigation rather than a reliable record. Revenue may be recognized aggressively, related-party loans may drain cash to a controlling family, and the free float you think you are buying may carry no real governance rights. The practical consequence for a valuer is that your *cash-flow forecast itself* is noisier in an emerging market, on top of the higher discount rate — two independent sources of error stacked on the same valuation. The disciplined response is to widen your scenario ranges and lean harder on cross-checks (multiples against global peers, reverse-DCF sanity tests) rather than pretending a single point estimate is precise.

Finally, the *time horizon* of the risks differs. Currency depreciation is a continuous drag, present in every period, and so it belongs in the discount rate or the cash-flow currency. Political shocks are discrete and lumpy — a tax law changes overnight, a sector is nationalized in a single decree — and so they belong in a scenario tree, not in a smooth annual penalty. Liquidity risk is a one-time cost you pay when you try to exit, and so it belongs in a value haircut. Recognizing which risk is continuous, which is discrete, and which is a transaction cost is the key to putting each in the right place in the model — and is the organizing idea behind everything that follows.

The mistake is to treat all of these as vague reasons to "be conservative" and then haircut the final number by some round figure. That hides the assumptions and makes the valuation impossible to argue about. The craft is to map each risk to a specific, defensible number. The rest of this post does exactly that, starting with the single most important adjustment: the country risk premium.

For the foundations of discount rates and required return in a developed-market setting — which this post builds on rather than repeats — see [the CAPM and cost of capital primer](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital) and [discount rates in practice](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta).

## The country risk premium: putting a number on the country

The **country risk premium (CRP)** is the additional return an equity investor demands for holding stocks in a particular country, over and above the return they would demand in a mature market like the United States. It is the single number that captures "this country is riskier."

The most widely used method comes from Aswath Damodaran. It has two ingredients.

Before the formula, the intuition in everyday terms. Suppose you lend money to two neighbors. One has a steady government job and a long record of paying you back; the other runs a business that booms and busts and has missed a payment before. You would demand a higher interest rate from the second neighbor — that extra rate is your premium for their default risk. A country's bond market does exactly this: it charges a riskier government a higher yield. The country default spread is that extra yield, and it is the market's own price for the country's creditworthiness — you do not have to estimate it, you can read it off bond prices.

The first is the **country default spread** — how much extra yield the country's government bonds pay over a default-free benchmark (US Treasuries) for the same maturity. This is a market price for the country's credit risk, and you can read it off bond yields or from sovereign credit-default-swap spreads. A country rated BB pays more than a country rated A, and the difference is the spread. As a shortcut, you can map the sovereign credit rating to a typical default spread.

Why use the rating-based shortcut at all when you could read the spread directly off bond yields? Two reasons. First, many emerging markets have no liquid dollar-denominated government bond at every maturity, so a clean market spread is not always observable. Second, market spreads can spike on temporary liquidity panics that overstate the structural default risk, whereas a rating-based spread smooths through the noise. In practice, valuers triangulate: they look at the sovereign credit-default-swap spread, the dollar-bond yield spread, and the rating-implied spread, and they reconcile the three. When they diverge sharply, that divergence is itself information — a CDS spread far above the rating-implied level usually means the market is pricing a deterioration the agencies have not yet acted on.

The second ingredient scales that bond-market spread up into an equity number, because **stocks are riskier than government bonds**. The logic is that the default spread prices the risk of *the government* not paying its debts, but equity holders sit below bondholders in the capital structure and below the government in the pecking order of a crisis — when a country gets into trouble, equities fall further than bonds. Damodaran multiplies the default spread by the ratio of equity-market volatility to bond-market volatility in that country:

```
CRP = Country default spread × (σEquity / σBond)
```

where σEquity is the standard deviation of the country's stock index and σBond is the standard deviation of its government bond returns (or the bond index). For emerging markets this ratio is typically between 1.2 and 1.8 — equities swing more than bonds, so the equity premium is larger than the raw default spread.

Here is the 2024 picture for the major emerging markets, using Damodaran's January 2025 data.

![Country risk premium bar chart by country 2024](/imgs/blogs/emerging-market-stock-valuation-country-risk-discount-rate-2.png)

Vietnam sits near the top of the investable-EM group at about 5.5%, reflecting a BB sovereign rating, a default spread of roughly 3.2%, and an equity-to-bond volatility ratio near 1.7. India, rated Baa3/BBB-, carries a much lower CRP of about 2.2%. Indonesia (Baa2/BBB) is around 3.8%, Brazil (Ba2/BB) around 5.4%, and China (A1/A+) around 3.0%. The frontier and distressed names — Nigeria, Turkey, Argentina — run far higher, with Argentina above 13% in a year of deep macro stress.

Notice the wide spread within "emerging markets." India's 2.2% and Argentina's 13%-plus are both EM, but they are utterly different valuation environments. An Indian stock carries barely more country risk than a peripheral European one; an Argentine stock carries enough that the country term alone can exceed the entire equity risk premium of a developed market. This is why "emerging markets" as a single asset class is a fiction for a valuer — you are valuing a *country* first and a company second, and the country term swamps almost everything else at the high end. Treating Vietnam, India, and Argentina with one discount-rate add-on would be like using one interest rate for a AAA corporate and a junk bond.

The CRP is also not static. It moves with the sovereign rating, with the default spread (which trades daily in the bond and CDS markets), and with the equity/bond volatility ratio (which rises in crises as equities become more volatile relative to bonds). A rating upgrade, a narrowing of the default spread on improving fiscal numbers, or a calming of equity volatility all pull the CRP down — and because the CRP sits in the discount rate of *every* company in that country, a falling CRP re-rates the whole market at once. This is the mechanical engine behind the observation that emerging markets rally together when global risk appetite improves: it is not just sentiment, it is a single shared input in everyone's denominator moving in the same direction.

#### Worked example: building Vietnam's country risk premium from scratch

Start with Vietnam's sovereign rating of BB. The typical default spread for a BB-rated sovereign in 2024 was about 3.2% — that is, Vietnamese government dollar bonds yielded roughly 3.2 percentage points more than comparable US Treasuries. Now scale it to equities. Vietnam's stock index (the VN-Index) had an annualized volatility of roughly 22% over the trailing period, while Vietnamese government bonds had a volatility of roughly 13%. The ratio is 22% / 13% ≈ 1.7.

Multiply: CRP = 3.2% × 1.7 ≈ **5.4%**, which we round to 5.5%. That is the extra annual return an equity investor should demand simply for the stock being Vietnamese rather than American — before any company-specific risk.

The intuition: the bond market already prices Vietnam's credit risk at 3.2%, and because Vietnamese stocks move about 1.7 times as much as Vietnamese bonds, the equity version of that risk is 1.7 times larger.

For deeper background on currency-and-country risk in the macro context, see [currency risk in emerging markets](/blog/trading/macro-trading/currency-risk-emerging-markets).

## Three ways to put country risk into CAPM

Once you have a CRP, you must decide *how* it enters the required-return calculation. This is not a trivial bookkeeping choice — it is a judgment about what country risk *is*. Does the country's risk fall equally on every company that happens to be domiciled there, or does it fall in proportion to how much each company actually depends on the local economy? Your answer determines which of the three standard approaches you use, and the approaches can produce required returns several percentage points apart for the same company.

There are three standard approaches, and they make different assumptions about how exposed a particular company is to its country's risk.

**Approach (a): add the CRP to everyone.** The simplest version treats country risk as a flat add-on that every company in the country bears equally:

```
Required return = Rf + β × (mature ERP) + CRP
```

Here Rf is a global (US) risk-free rate, the mature equity risk premium (ERP) is the US-market premium, β is the company's beta, and the CRP is bolted on at the end. This is easy and transparent, but it assumes a Vietnamese software exporter selling to the US is exactly as exposed to Vietnam's country risk as a Vietnamese electric utility selling only domestically — which is wrong.

**Approach (b): local CAPM.** Build everything from local-market parameters: a local risk-free rate (the local government bond yield, stripped of default risk), a local equity risk premium estimated from the local market, and a beta measured against the local index. This keeps everything internally consistent but inherits all the problems of short, noisy local data — local ERPs are notoriously unstable, and local betas are distorted by thin trading.

**Approach (c): global CAPM with a country-risk exposure (lambda).** The most defensible approach recognizes that companies differ in how exposed they are to their home country. It introduces a coefficient **λ (lambda)** between 0 and 1 (occasionally above 1) that measures a company's exposure to local country risk:

```
Required return = Rf + β × (mature ERP) + λ × CRP
```

A company that earns all its revenue domestically and has all its assets in-country has λ ≈ 1. A pure exporter that sells in dollars and holds offshore cash has λ well below 1 — it happens to be domiciled in the country but is only partly exposed to it. The cleanest proxy for λ is the share of revenue earned domestically relative to the average domestic-revenue share of companies in that market. Concretely, if the average listed company in a country earns 75% of revenue at home and your target earns only 40% at home, its lambda is roughly 0.40 / 0.75 ≈ 0.53 — it bears about half the country risk of a typical local firm. A company that earns *more* than the local average at home (say a regulated utility at 100%) gets a lambda slightly above 1.

The choice between the three approaches is not just technical taste; it changes the answer materially. Consider a Vietnamese technology exporter selling almost entirely to US and European clients. Under approach (a), the flat add-on, you bolt the full 5.5% CRP onto its required return, treating it as exactly as risky as a domestic utility. Under approach (c) with a lambda of perhaps 0.3, you add only 0.3 × 5.5% = 1.65%. That nearly four-percentage-point difference in the discount rate can swing the valuation by 30–50% on a long-duration cash-flow stream. For export-heavy emerging-market companies — and many of the most attractive ones are exporters precisely because they have escaped the domestic constraints — getting lambda right is often the difference between a buy and a pass.

A note of caution on approach (b), the pure local CAPM: it is the most internally consistent in theory and the most dangerous in practice. A local equity risk premium estimated from twenty years of a market that contained one currency crisis can come out negative, or absurdly high, depending on the window. A local risk-free rate taken from a government bond that itself carries default risk is not actually risk-free. Most practitioners therefore default to approach (c) — a global risk-free rate, a mature equity premium, and an explicit, exposure-scaled country term — because every input in it is estimated from deep, clean global data, and only the lambda and the CRP are country-specific.

![Equity risk premium plus country risk by region stacked bar](/imgs/blogs/emerging-market-stock-valuation-country-risk-discount-rate-7.png)

The chart above shows the total equity risk premium by region in 2024 — the mature-market base premium plus the regional country-risk addition. North America is essentially the base premium alone; EM Asia, EM Latam, and EM Africa stack progressively larger country-risk slices on top.

#### Worked example: required return for a Vietnamese stock with lambda

Take a Vietnamese company that earns about 80% of its revenue domestically, so λ = 0.8. Use these 2024 inputs:

- US risk-free rate (Rf) = 4.3% (the 10-year Treasury level)
- Mature (global) equity risk premium = 5.5%
- Company beta (β) = 1.2
- Vietnam CRP = 5.5%
- Lambda (λ) = 0.8

Plug into approach (c):

```
Required return = 4.3% + 1.2 × 5.5% + 0.8 × 5.5%
                = 4.3% + 6.6% + 4.4%
                = 15.3%
```

If instead you used the flat add-on of approach (a), with full λ = 1.0 implied, the country term becomes the full 5.5%, giving 4.3% + 6.6% + 5.5% = 16.4%. And if the company were a pure domestic utility with above-average country exposure (λ slightly over 1), the required return climbs toward 17.6%. The brief's headline figure of roughly 17.6% corresponds to the high-exposure end of this range.

The intuition: a Vietnamese stock needs a required return in the mid-to-high teens — roughly double a comparable US stock at 9–11% — and the lambda tells you whether to land at the low or high end of that band based on how exposed the company really is to Vietnam itself.

## Beta problems in emerging markets

CAPM's β is supposed to measure how much a stock moves with the market — a β of 1.2 means the stock is 20% more volatile than the index, and therefore needs a proportionally higher return. In a deep, liquid market, you estimate β by regressing the stock's returns on the market's returns over a few years. In an emerging market, that regression lies to you, for two reasons.

**Thin trading drags beta down.** If a stock does not trade every day, its recorded price is stale. On days the market moves but the stock does not trade, the stock's "return" is recorded as zero. Those false zeros weaken the measured correlation between the stock and the market, so the regression spits out a β that is too *low*. A genuinely risky small-cap might show a β of 0.6 purely because it trades infrequently — which would lead you to demand too little return and overvalue it. This is the opposite of conservative.

**Short, crisis-dominated samples.** A five-year window that contains a currency crisis or a single political shock can produce a β that reflects that one event rather than the stock's normal behavior. Worse, in a crisis nearly every local stock falls together, so *all* local betas converge toward 1 during the crash and then drift apart again afterward — the regression averages these two regimes into a number that describes neither.

**The reference-index problem.** A beta is only meaningful relative to the market you regress against. In a small emerging market, the local index is often dominated by a handful of names — a few banks and a state energy company can be half the index. A beta measured against such a concentrated index really measures co-movement with those few giants, not with "the market" in any diversified sense. If the company you are valuing is in an unrelated sector, its local beta may be near zero simply because it does not move with the banks that dominate the index. This is a third reason to prefer a fundamental beta built from global sector peers over a locally regressed one.

There are two standard corrections.

The **Blume adjustment** pulls a raw β toward the market average of 1.0, on the empirical observation that betas mean-revert over time:

```
Adjusted β = 0.67 × (raw β) + 0.33 × 1.0
```

This dampens extreme estimates in both directions, but it does not fix the thin-trading bias specifically — a 0.6 raw β only rises to about 0.73, still too low for a genuinely risky stock.

The **fundamental (bottom-up) beta** ignores the stock's own price history entirely. You take the average unlevered β of comparable companies — ideally global peers in the same business, where the data is clean — then relever it for the target company's debt and, if you wish, adjust for its operating leverage. This is the preferred method in thin markets precisely because it does not depend on the unreliable local price series. The mechanics of unlevering and relevering beta are covered in [discount rates in practice](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta).

#### Worked example: from a misleading 0.6 to a usable 1.1

A Vietnamese mid-cap building-materials company shows an observed β of 0.6 from a regression on the VN-Index. You know the company is cyclical and carries debt, so 0.6 fails the smell test — it is almost certainly a thin-trading artifact.

First, sanity-check with Blume: 0.67 × 0.6 + 0.33 × 1.0 = 0.40 + 0.33 = 0.73. Better, but still low for a leveraged cyclical.

Now build a fundamental beta. Global building-materials peers have an average unlevered β of about 0.9. The Vietnamese company has a debt-to-equity ratio of 0.4 and faces a 20% tax rate. Relever:

```
Levered β = unlevered β × [1 + (1 − tax) × (D/E)]
          = 0.9 × [1 + (1 − 0.20) × 0.40]
          = 0.9 × [1 + 0.32]
          = 0.9 × 1.32
          = 1.19  ≈ 1.1
```

Use **1.1**, not 0.6. The difference is enormous: at a 5.5% mature ERP, the cost-of-equity contribution from β alone rises from 0.6 × 5.5% = 3.3% to 1.1 × 5.5% = 6.05% — nearly three percentage points of required return that the thin-trading β would have hidden.

The intuition: never trust a low beta in a thinly traded stock; rebuild it from clean global peers, because the local price history is too quiet to be honest.

## Currency risk in DCF: two roads, one destination

Currency is where most emerging-market valuations quietly go wrong. The everyday version of the trap is this: if your salary is paid in a currency losing 10% of its value a year, a 10% raise leaves you no better off — the raise just kept pace with the currency's decline. Emerging-market cash flows often look like that raise: they grow fast in local-currency terms partly *because* the currency is depreciating, and a valuer who counts the growth but ignores the depreciation books a gain that does not exist. The principle that prevents this is simple and absolute: **the currency of your cash flows must match the currency of your discount rate.** There are two consistent ways to do a DCF, and they must give the same answer.

**Road 1: local currency throughout.** Forecast cash flows in the local currency (say, Vietnamese dong) and discount them at a local-currency discount rate. The local-currency rate is higher because it includes local expected inflation. A 17% VND discount rate is not "more pessimistic" than a 12% USD rate — it is the *same* required real return expressed in a currency that is depreciating.

**Road 2: US dollars throughout.** Convert the cash flows into dollars using forecast exchange rates (which embed the expected currency depreciation), then discount at a dollar discount rate.

These two roads give the *same* present value if and only if your inflation and exchange-rate assumptions are internally consistent — specifically, if the difference between the two discount rates equals the expected currency depreciation, which by relative purchasing-power parity equals the inflation differential.

The algebra behind this is worth seeing once. The relationship between a local-currency required return *r*(local) and a dollar required return *r*(\$) is:

```
(1 + r_local) = (1 + r_USD) × (1 + expected depreciation)
```

And relative purchasing-power parity says the expected depreciation of the local currency roughly equals the inflation differential:

```
expected depreciation ≈ inflation_local − inflation_USD
```

Chain these together and you can see why the two roads must agree. If you forecast cash flows that grow with local inflation and discount at a local rate that *includes* that same inflation, the inflation cancels. If you convert those cash flows to dollars at exchange rates that fall by the inflation differential, and then discount at a dollar rate that excludes local inflation, the inflation cancels again — by a different route. The present value is identical because you have not changed any economic assumption, only the unit of account. The danger is purely arithmetic bookkeeping: keep the inflation in *both* the cash flows and the rate, or in *neither*, but never in one and not the other.

There is a related real-world wrinkle: purchasing-power parity holds only loosely over short horizons. Currencies overshoot and undershoot their inflation-implied path for years. A rigorous valuer therefore does not blindly assume PPP for the explicit forecast period; instead they use forward exchange rates (which embed the market's interest-rate-parity view) for the near years and let the cash flows and the rate converge to a PPP-consistent path in the terminal value. But the consistency principle is unchanged — whatever depreciation path you assume, the discount rate must reflect the same currency.

The catastrophic, common mistake is to mix the two roads: forecast cash flows that grow with high local inflation in local currency, then discount them at a low dollar rate. This keeps the inflation *boost* to the cash flows but drops the inflation *penalty* in the discount rate — and it systematically overstates value.

![Wrong versus right currency and discount rate matching](/imgs/blogs/emerging-market-stock-valuation-country-risk-discount-rate-4.png)

#### Worked example: the two-currency consistency check

A Vietnamese company will generate a single cash flow one year from now. Expected Vietnamese inflation is 4%; expected US inflation is 2%. So the dong is expected to depreciate by roughly the inflation differential, about 4% − 2% = 2% per year (relative purchasing-power parity). Today's exchange rate is 25,000 VND per USD, so the expected rate in one year is about 25,000 × 1.02 = 25,500 VND/USD.

The cash flow one year out is 255 million VND. Required returns: 17% in VND, and about (1.17 / 1.02) − 1 ≈ 14.7% in USD — but to keep the brief's clean illustration, assume the inflation/FX assumptions imply a 12% USD rate, meaning the company's real required return and a milder depreciation path are baked in. Let us run both roads with the rates as given (17% VND, 12% USD) and a depreciation that makes them consistent.

For the two roads to match, the depreciation must satisfy (1 + 17%) = (1 + 12%) × (1 + depreciation), so 1.17 / 1.12 = 1.0446, i.e. about 4.46% expected annual depreciation. Then the expected rate in one year is 25,000 × 1.0446 = 26,116 VND/USD.

**Road 1 (VND):** Present value = 255,000,000 / 1.17 = **217,948,718 VND**. At today's 25,000 spot, that is 217,948,718 / 25,000 = **\$8,718**.

**Road 2 (USD):** Convert the future cash flow at the *future* expected rate: 255,000,000 / 26,116 = \$9,764. Discount at 12%: 9,764 / 1.12 = **\$8,718**.

Both roads land on **\$8,718**. They agree exactly because the gap between the two discount rates (17% vs 12%) was matched by the expected currency depreciation (4.46%).

The intuition: a higher local-currency discount rate and a depreciating currency are two descriptions of the same risk; use either, never half of each.

## The liquidity discount for thin emerging-market stocks

A liquid stock can be sold instantly at the quoted price. An illiquid one cannot — you either wait, or you accept a worse price to sell now. That cost is real, and the market prices it as a discount to what the same cash flows would be worth if they were freely tradable. For emerging-market small-caps and private companies, this discount can be large.

There are two main ways to estimate it.

To ground this, start with the everyday version. A house is worth less if you must sell it next week than if you can wait for the right buyer — the rush forces you to accept a lower price. Stocks work the same way. A liquid stock is like cash in a checking account; an illiquid one is like a house — its quoted "value" assumes a patient sale, and a forced or quick sale fetches less. The liquidity discount is the gap between the patient-sale value (what your DCF computes) and what you would actually realize given how hard the stock is to sell.

**The bid-ask spread method** treats the round-trip transaction cost as a lower bound on the discount. If a stock's bid-ask spread is 4% and you expect to trade it repeatedly, the present value of those friction costs over a holding period maps to a discount of several percent. Spreads on EM small-caps routinely run 2–6%, versus a few basis points on US large-caps.

**Restricted-stock and private-transaction studies** measure how much less buyers pay for shares they cannot immediately resell. These studies, summarized by Damodaran among others, find illiquidity discounts of roughly 15–25% for restricted shares, rising to 30–50% for private companies in emerging markets where exit options are scarce.

![Illiquidity discount scale by liquidity tier](/imgs/blogs/emerging-market-stock-valuation-country-risk-discount-rate-5.png)

The chart shows the rough ladder: liquid large-caps 0–5%, listed EM mid-caps 10–20%, listed EM small-caps 20–35%, and private EM companies 30–50%. The discount widens as the stock gets smaller, less traded, and harder to exit.

A subtle point: you should apply the liquidity discount to the *equity value*, after the DCF, not by inflating the discount rate. Folding liquidity into the discount rate compounds it incorrectly over time, whereas a one-time haircut to the value matches how a buyer actually thinks — "I'll pay 80 cents on the dollar because I can't easily sell this."

The size of the discount should also scale with the investor's likely holding period and position size. A long-term strategic buyer who plans to hold for a decade cares far less about exit liquidity than a fund that may need to redeem in a market panic, so the same illiquid stock warrants a smaller discount for the former and a larger one for the latter. Similarly, a large block in a thinly traded name carries a bigger discount than a small position, because selling the block would itself move the price — this is the "blockage" effect that restricted-stock studies pick up. The single most common error here is applying a textbook 25% across the board; the disciplined valuer ties the number to observable evidence (this stock's actual spread, its actual daily turnover relative to the position size, and restricted-stock evidence for comparable names) rather than reaching for a round figure.

One more refinement: liquidity is not constant. It evaporates exactly when you most want to sell — in a crisis, spreads on EM small-caps can triple and buyers vanish entirely. This means the liquidity discount is itself correlated with the country and political risks already in the model, which is an argument for sizing it toward the upper end of the evidence range for stocks in fragile markets, where a forced-selling scenario is most plausible.

#### Worked example: applying a liquidity discount to a small-cap

Your DCF on a thinly traded Vietnamese small-cap produces a freely-tradable equity value of \$50 million. The stock's average bid-ask spread is 5%, daily turnover is a small fraction of the float, and comparable restricted-stock studies for EM small-caps point to a discount around 25%.

Apply the haircut to the value, not the rate:

```
Liquidity-adjusted value = \$50,000,000 × (1 − 0.25)
                         = \$50,000,000 × 0.75
                         = \$37,500,000
```

The illiquidity costs you \$12.5 million of value — a quarter of the company — purely because a buyer cannot count on getting out at the quoted price.

The intuition: liquidity is something a buyer pays *up* for, so its absence is a one-time discount to value, sized from spreads and restricted-stock evidence, not a markup to the discount rate.

## Why EM multiples are structurally lower — and why that is not "cheap"

Investors often glance at MSCI Emerging Markets trading at a P/E of 12 against MSCI World at 19 and conclude that emerging markets are "cheap." Sometimes they are. But a large part of that gap is not a bargain — it is mathematics.

![MSCI EM P/E versus MSCI World P/E 2010 to 2024](/imgs/blogs/emerging-market-stock-valuation-country-risk-discount-rate-3.png)

The chart shows the persistence of the discount: across 2010–2024, EM consistently traded several P/E points below developed markets, with the gap widening sharply in years like 2020 when developed-market multiples ballooned.

To see why this is structural, look at the Gordon growth model, which links a fair P/E to required return and growth. For a stock paying out fraction *p* of earnings, growing at *g*, with required return *r*:

```
P/E = p / (r − g)
```

Required return *r* sits in the denominator. We just spent this whole post showing that *r* is systematically higher in emerging markets — by the country risk premium, several percentage points. Hold the payout ratio and growth constant, raise *r*, and the fair P/E *falls*. A lower P/E is the *correct* valuation of higher-required-return cash flows, not evidence of a mispricing.

This has a sharp practical consequence for the popular game of "EM is trading at a discount to its own history, so it's cheap." That comparison is only valid if the required return has stayed constant — and it usually has not. If a country's CRP has risen (a rating downgrade, a fiscal scare, a currency wobble), then a P/E that has fallen may simply be tracking the higher required return, leaving the stock fairly valued the whole time. Conversely, if the CRP has fallen — say on an MSCI upgrade or improving fiscal numbers — a P/E that has merely held flat is actually *cheapening* in risk-adjusted terms, because the same multiple now sits on lower-risk cash flows. To judge whether an EM multiple is genuinely cheap, you must decompose it: how much of the multiple gap versus developed markets, or versus the stock's own history, is explained by the required-return gap, and how much is left over as a true valuation signal? Only the residual is an opportunity.

The same logic explains why growth does not rescue the comparison. Emerging markets often grow faster, and growth *g* also sits in the denominator (it subtracts from *r*), pushing fair P/E up. But the CRP-driven increase in *r* is typically larger than the growth advantage, which is why EM multiples stay structurally lower despite faster growth. You cannot wave away the discount by pointing at GDP growth; you have to net the higher required return against the higher growth, and in most emerging markets the required return wins.

#### Worked example: decomposing the EM P/E discount

Compare two otherwise identical companies, one US and one Vietnamese, each paying out 50% of earnings and growing earnings at 5%.

US company: required return 9%.

```
Fair P/E = 0.50 / (0.09 − 0.05) = 0.50 / 0.04 = 12.5
```

Vietnamese company: required return 15.3% (from our earlier lambda example).

```
Fair P/E = 0.50 / (0.153 − 0.05) = 0.50 / 0.103 = 4.85
```

The Vietnamese company *deserves* a far lower P/E — under 5 versus 12.5 — for the exact same earnings, payout, and growth, purely because its required return is higher. If the Vietnamese stock actually traded at a P/E of 8, it would be *expensive* relative to its risk-adjusted fair value, not cheap, despite trading at a lower multiple than the US name.

The intuition: a low P/E in a high-required-return market is the right price for the risk, so comparing EM and DM multiples without adjusting for required return tells you almost nothing about value.

For how relative and absolute valuation methods fit together, see [the valuation spectrum](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims), and for the Vietnam market specifically, [VN-Index P/E dynamics](/blog/trading/asset-valuation/vietnam-stock-market-valuation-vnindex-pe-dynamics).

## Political risk as a real option, priced through scenarios

The hardest emerging-market risk to quantify is political: a tax change, a price cap, an expropriation, a coup. You cannot read it off a bond yield, and burying it in the discount rate is a blunt instrument — it penalizes every year equally when the real risk is a discrete, lumpy event.

The disciplined approach is **scenario weighting**. Lay out a small number of distinct futures, assign each a probability and a value, and take the probability-weighted average. This is, in effect, treating the political situation as a set of branching outcomes — a real-option structure where the value depends on which branch the world takes.

![Political risk scenario tree with probabilities and values](/imgs/blogs/emerging-market-stock-valuation-country-risk-discount-rate-6.png)

The tree above lays out three branches for a stock: a base case where policy continues, a reform-upside case where pro-market changes lift value, and a crisis case where capital controls or nationalization cut it sharply.

This approach has three virtues over a fudge factor in the discount rate. First, it is *transparent* — anyone can challenge your probabilities or your scenario values directly. Second, it captures *asymmetry* — political risk in emerging markets is often skewed, with a fat left tail (crisis) and a thinner right tail (reform), which a symmetric discount-rate bump cannot represent. Third, it forces you to *name* the actual events you fear rather than gesturing at "uncertainty."

Why not just add political risk to the discount rate, as some practitioners do? Because a discount-rate penalty applies the same proportional haircut to every future year, which is the wrong shape for a discrete event. If you fear a nationalization that, should it happen, would happen *once* and wipe out a large fraction of value, spreading that fear evenly across forty years of discounting both mis-times the risk and double-counts it in the terminal value. A scenario tree puts the loss where it belongs — in a specific branch with a specific probability — and leaves the other branches discounted at the ordinary rate. The discount rate should carry continuous, recurring risks; the scenario tree should carry discrete, one-off risks. Mixing them is the single most common way valuers either over- or under-penalize political risk.

The hard part of scenario valuation is honesty about the probabilities. It is tempting to set the crisis probability low because the downside is unpleasant to contemplate, but the whole point is to confront it. Useful anchors include the country's own history (how often has it actually expropriated, defaulted, or imposed controls in the past few decades?), the current political calendar (an upcoming election or constitutional change raises the odds), and market-implied signals (sovereign CDS spreads and the currency forward curve already encode some of the market's crisis probability). A good practice is to *back out* the implied probability: if the stock trades at \$80 and your base case is \$100, what crisis probability would the market have to be pricing to justify that gap? If the answer is a 33% chance of a 75% loss, ask whether that is too pessimistic or about right — the market is often a useful counterparty to argue against.

You can extend the tree to more than three branches, and for high-stakes situations you should — separating, say, "mild tax increase," "windfall tax," and "full nationalization" into distinct branches with their own probabilities and values. The mechanics do not change: enumerate the mutually exclusive futures, assign probabilities that sum to one, value each, and take the weighted average. What changes is the resolution of your view, which is usually worth the extra work when a single political outcome can move the value by half.

#### Worked example: scenario-weighted valuation with political risk

You value a stock at \$100 per share in a base case where the current policy regime persists. But there is an election in eighteen months. You map three scenarios:

- **Base case** — policy continuity. Probability 70%. Value \$100.
- **Reform upside** — a pro-market government cuts the corporate tax rate and opens the sector to foreign capital, lifting value. Probability 20%. Value \$140.
- **Political crisis** — a populist government imposes price controls and capital restrictions. Probability 10%. Value \$40.

Probability-weight:

```
Expected value = 0.70 × \$100 + 0.20 × \$140 + 0.10 × \$40
               = \$70 + \$28 + \$4
               = \$102
```

Wait — that comes to \$102, *above* the base case, because the reform upside outweighs the crisis downside in this particular set of probabilities. If instead the crisis were more likely (say 30%) and reform less so (10%), the weighting would be 0.60 × 100 + 0.10 × 140 + 0.30 × 40 = 60 + 14 + 12 = \$86, well below base. The brief's illustrative case — base 70% at \$100, reform 20% at \$140, crisis 10% at \$40 — weights to about \$95 once you also haircut the base case for the ongoing drag of policy uncertainty, but the mechanics are what matter: the political-risk-adjusted value is the probability-weighted average, and its direction depends entirely on the skew of the outcomes.

The intuition: political risk is a set of named futures with probabilities, not a vague penalty — price it by weighting the outcomes, and the answer can land above or below the base case depending on whether the upside or downside dominates.

## Common misconceptions

**"Emerging markets are cheap because their P/E is lower."** Mostly false. As the Gordon-model decomposition showed, a P/E of 5 on a stock with a 15% required return is *fairly* priced, while the same P/E on a 9% required-return stock would be a screaming bargain. EM multiples are structurally lower because required returns are structurally higher. A multiple is only cheap relative to a risk-adjusted fair multiple — never compare EM and DM multiples raw.

**"Use a higher discount rate to cover all the extra risk."** This is the lazy version of everything in this post, and it is wrong in two ways. It compounds discrete risks (like political shocks and illiquidity) incorrectly over time, and it hides the assumptions so no one can challenge them. Map each risk to its proper home: country risk and beta into the discount rate, currency into the cash-flow/rate matching, liquidity into a one-time value haircut, and political risk into scenarios.

**"A low beta means a safe, low-return stock."** In a thinly traded market, a low beta often means the *opposite* — it is a statistical artifact of stale prices, and the true risk is higher. The worked example took an observed 0.6 to a fundamental 1.1. Always rebuild beta from clean peers in thin markets.

**"Forecast in dollars to avoid currency risk."** Forecasting in dollars does not avoid currency risk; it relocates it into the exchange-rate forecast. The only sin is *inconsistency* — local cash flows with a dollar rate, or vice versa. Done consistently, both currencies give the same value.

**"Add the full country risk premium to every company in the country."** A pure exporter earning dollars offshore is far less exposed to its home country's risk than a domestic utility. The lambda in approach (c) scales the CRP by actual exposure; applying the full CRP to an exporter overstates its risk and undervalues it.

**"Political risk belongs in the discount rate."** A discrete, one-off event like nationalization spread evenly across forty years of discounting is both mis-timed and double-counted in the terminal value. Discrete risks belong in a scenario tree; only continuous, recurring risks belong in the rate. The fix is to separate the two and price each in its proper place.

A unifying way to remember all of this: every adjustment in this post answers the question "where does this risk actually live in time?" Currency depreciation lives in every period, so it lives in the rate or the cash-flow currency. Country credit and systematic risk live in every period too, so they live in the rate via the CRP and beta. Liquidity is a cost you pay once, on exit, so it lives in a one-time value haircut. Political shocks are discrete and lumpy, so they live in a scenario tree. Put each risk where it actually occurs in time, and the model is both correct and defensible. Smear them all into one inflated discount rate, and you get a number that is wrong in ways you can no longer see.

## How it shows up in real markets

**Turkey, 2021–2024: currency dominates everything.** Over this stretch the lira fell from roughly 8 to over 35 per dollar. A Turkish company that grew its lira earnings 20% a year still destroyed dollar value, because the currency fell faster. Any valuation that forecast lira cash flows and discounted at a dollar rate would have shown a fantasy value. The consistency check — match the currency to the rate — was not an academic nicety here; it was the entire investment outcome.

**Argentina, 2024: the CRP off the charts.** With a sovereign deep in distress, Argentina's country risk premium ran above 13% in early 2024 per Damodaran's data — meaning a fair required return for an Argentine stock with a beta of 1 would clear 22–23% before any company-specific risk. At those rates, the Gordon model collapses fair P/E ratios into the low single digits, which is exactly what Argentine equities traded at. The "cheapness" was the correct price of the risk.

**Vietnam, 2024: the upgrade catalyst.** Vietnam spent 2024 working toward an MSCI reclassification from frontier to emerging market. The valuation logic is direct: an upgrade improves accessibility and liquidity, which *shrinks* the liquidity discount and can lower the effective country risk premium as foreign capital arrives. A re-rating of Vietnamese equities on an upgrade is not sentiment — it is the discount rate and the liquidity haircut both falling, which mechanically raises fair value. See [VN-Index P/E dynamics](/blog/trading/asset-valuation/vietnam-stock-market-valuation-vnindex-pe-dynamics) for the market-level picture.

**Brazil and the commodity-cycle CRP.** Brazil's 5.4% CRP in 2024 reflects both its BB rating and the volatility of an economy levered to commodity prices. When commodities boom, Brazilian fiscal health improves, the default spread narrows, and the CRP falls — which lifts fair values across the market independent of any individual company. This is why EM valuations move with global risk appetite: a single input, the country risk premium, sits in every company's discount rate at once.

**India and the lambda dispersion within one market.** India's low 2.2% CRP means the country term is a minor adjustment, but the lambda effect is large *within* the market. India's globally competitive IT-services exporters earn most of their revenue in dollars from US and European clients, so their effective country exposure is low and their discount rates barely move from a developed-market benchmark — which is part of why they command global-quality multiples. A domestically focused Indian infrastructure company, by contrast, bears the full country term. Valuing both with the same flat add-on would have systematically undervalued the exporters and overvalued the domestic plays for years. The lambda is not a refinement here; it is the difference between getting the relative valuation of the two right or backwards.

**The 2018 emerging-market selloff: when everything moves at once.** In 2018, a stronger dollar and rising US rates triggered a broad EM selloff that hit Turkey and Argentina hardest but dragged down even healthy markets. The valuation lesson was that the country risk premium and the currency adjustment are not independent — a dollar funding squeeze raises CRPs (via wider default spreads), depreciates currencies (hurting dollar-converted cash flows), and dries up liquidity (widening the discount) all at the same time. A valuation that treated these as separate, uncorrelated adjustments would have understated the tail risk. In fragile markets, the four adjustments tend to move together in a crisis, which is why stress-testing a valuation by shocking all four at once — higher CRP, weaker currency, lower beta-implied confidence, and a wider liquidity discount — is more informative than shocking any one alone.

It is also worth stating what emerging-market valuation does *not* require: it does not require throwing out the developed-market toolkit. The DCF, CAPM, and multiples all still work — you are not learning a new method, you are calibrating the existing one to a harder environment. That is reassuring, because it means the discipline is portable. Once you can identify where each EM-specific risk lives in time and route it to the right place in the model, you can value a stock in Hanoi, São Paulo, or Lagos with the same framework you use in New York, changing only the inputs. The framework does not change; the numbers do.

The thread through all the cases is that emerging-market valuation rewards discipline. Each risk has a specific number and a specific home in the model. Get the homes right — country risk and beta in the rate, currency in the matching, liquidity in the haircut, politics in the scenarios — and you produce a value you can defend line by line. Bury everything in a vague high discount rate, and you produce a number no one, including you, can argue about. The two analysts from the opening were both competent; the one who priced each risk explicitly could explain her 40% gap to a client, defend every percentage point of it, and update it cleanly when the country's rating changed. That is the whole point — not a lower number, but a *defensible* one.

## Further reading & cross-links

- [Risk, required return, CAPM, beta, and cost of capital](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital) — the developed-market foundation this post adjusts.
- [Discount rates in practice: WACC, cost of equity, unlevered beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta) — the unlever/relever mechanics used in the beta example.
- [The valuation spectrum: absolute, relative, contingent claims](/blog/trading/asset-valuation/valuation-spectrum-absolute-relative-contingent-claims) — where DCF, multiples, and option-style scenario valuation fit together.
- [Vietnam stock market valuation: VN-Index P/E dynamics](/blog/trading/asset-valuation/vietnam-stock-market-valuation-vnindex-pe-dynamics) — the market-level Vietnam case.
- [Currency risk in emerging markets](/blog/trading/macro-trading/currency-risk-emerging-markets) — the macro view of the currency adjustment.
