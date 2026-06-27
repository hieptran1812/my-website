---
title: "Vietnam Stock Market Valuation: Why VN-Index P/E Behaves Differently"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "A from-scratch guide to why Vietnam's market trades at a structural P/E discount, and how to apply global valuation methods — CAPM with a country risk premium, P/B for banks, foreign-room premiums — to a frontier market."
tags: ["valuation", "asset-pricing", "vietnam", "vn-index", "country-risk-premium", "price-to-book", "emerging-markets", "foreign-ownership", "capm", "equity-valuation"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Vietnam's stock market trades at a structurally lower P/E (~12-15x) than developed markets (~18-25x) not because its companies are worse, but because its market *structure* stacks several discount drivers on top of fair value.
>
> - A retail-dominated, margin-leveraged market (~85% of volume) produces violent boom-bust cycles that compress multiples fast — the VN-Index P/E went from 19.2x in 2021 to 9.8x in 2022.
> - Foreign ownership caps (49% on most stocks), Frontier/EM index status, thin liquidity, and VND currency risk all keep the global buyer pool small, so the price stays cheap relative to earnings.
> - You cannot copy a developed-market discount rate. Add a **country risk premium** (CRP) of ~5-7% to your required return, and the "low" P/E suddenly looks fair.
> - Vietnamese banks — ~30% of the index — are best valued on **price-to-book**, not P/E. The one number to remember: at ROE 20% and required return 15%, a bank's justified P/B is `(0.20 − g) / (0.15 − g)`, which lands near 1.7x for a fast-growing bank.

In late 2022 a foreign fund manager looking at Vietnam saw something that looked too good to be true. Vietcombank — the country's largest and most profitable bank, growing earnings at double digits, with return on equity above 20% — was trading at a trailing P/E under 10. In the United States, a bank growing that fast with that profitability would command a premium multiple. The arithmetic seemed to scream "buy."

Six months earlier, that same fund manager had watched the VN-Index fall 33% in a matter of months. Stocks that had traded at 19x earnings in early 2021 were changing hands at single-digit multiples. Margin accounts blew up. Retail investors who had piled in during the 2020-2021 boom were forced to sell at any price. The "cheap" market got cheaper, fast, and the question that should have come before the buy order was: *cheap compared to what?*

The two scenes are not in tension; they are the same market viewed through two different mistakes. The first mistake is to look at a single-digit P/E and conclude "bargain" without asking what discount rate the market is applying. The second mistake is to watch a 33% crash and conclude "this market is uninvestable" without recognizing that the crash was a leverage event, not an earnings event, and that the underlying businesses kept compounding through it. Both mistakes come from importing a developed-market intuition wholesale into a market whose plumbing is fundamentally different.

That is the entire puzzle of Vietnamese equity valuation. The numbers look mispriced through a developed-market lens, but the lens itself is wrong. A P/E of 13 in Ho Chi Minh City does not mean the same thing as a P/E of 13 in New York, because the discount rate behind it, the buyer pool, the leverage in the system, and the currency the cash flows are denominated in are all different. This post builds the correct lens from zero — what the VN-Index is made of, why its multiple sits where it does, and exactly how to value a Vietnamese stock with the same global tools you would use anywhere, adjusted for the things that actually differ.

![VN market structure discount drivers stack](/imgs/blogs/vietnam-stock-market-valuation-vnindex-pe-dynamics-1.png)

## Foundations: what the VN-Index is and how it is priced

Before any valuation, you need to know what you are valuing. Let me define every term from the ground up.

**The VN-Index** is the benchmark stock index for the Ho Chi Minh City Stock Exchange (HSX or HOSE), the larger of Vietnam's two main exchanges (the other is Hanoi, HNX). It is a capitalization-weighted index, meaning each company's influence on the index is proportional to its total market value — the same construction as the S&P 500. When people say "the VN-Index is at 1,200," they mean the index level; when they say "the VN-Index trades at 13x," they mean the aggregate price-to-earnings ratio of its constituents.

**Price-to-earnings (P/E)** is the single most-quoted valuation number in the world. It is simply the price you pay for one unit of annual profit:

`P/E = Price per share ÷ Earnings per share`

If a stock costs \$100 and earns \$5 per share, its P/E is 20 — you are paying \$20 for every \$1 of annual profit. Equivalently, the **earnings yield** (the inverse, `E/P`) is 5%: the company earns 5% of your purchase price every year. A high P/E means you are paying a lot for current earnings, usually because you expect them to grow; a low P/E means the opposite. The same logic at the index level: the VN-Index P/E is the total market value of all constituents divided by their total earnings.

Now the crucial fact that drives everything else. The VN-Index is **banking-heavy**. Financials — overwhelmingly commercial banks — make up roughly 30% of the index by weight. Real estate developers and industrials are the next-largest blocks. Add a smattering of consumer staples, steel, and a small but growing technology slice, and you have the whole index. This composition matters enormously, because banks and real estate developers are valued *differently* from the asset-light tech and consumer companies that dominate the S&P 500. A market that is one-third banks should not be expected to carry a tech-market multiple.

The second structural fact is **who trades it**. The Vietnamese market is *retail-dominated*: individual investors, not institutions, account for roughly 85% of daily trading volume. Compare this to the United States, where institutions — pension funds, mutual funds, index funds — dominate. Retail-dominated markets behave differently. They are more momentum-driven, more prone to herding, and far more sensitive to leverage, because individuals trade with borrowed money (margin) in a way that institutions, constrained by mandates, do not.

The third fact is **settlement and leverage mechanics**. Vietnam settles trades on a **T+2** basis — you receive your shares two business days after the trade. Combined with a vibrant **margin trading** culture (brokers lend investors money to buy more stock than their cash allows), this creates a system where, when prices fall, margin calls force selling, which pushes prices lower, which triggers more margin calls. This reflexive loop is the engine of Vietnam's boom-bust cycles, and it is the single biggest reason the P/E can move so violently.

Put those three facts together — a bank-heavy index, a retail-dominated tape, and a leverage-amplified settlement system — and you already have most of the explanation for why the VN-Index P/E behaves differently. The rest of this post fills in the discount-rate machinery and shows you how to value a stock inside this structure.

Before moving on, it is worth pinning down two more foundational pieces, because they recur in every valuation that follows: the difference between trailing and forward multiples, and the role of the discount rate.

**Trailing versus forward P/E.** When you read "the VN-Index trades at 13.8x," you need to ask: 13.8 times *which* earnings? **Trailing P/E** uses the last twelve months of actual reported earnings — backward-looking, factual, but stale, because it reflects a business as it was, not as it will be. **Forward P/E** uses analysts' estimates of next year's earnings — forward-looking and more relevant for a fast-growing economy, but only as good as the estimates. In a high-growth market like Vietnam, where corporate earnings can grow 15-20% in a single year, the gap between trailing and forward multiples is large. A stock at a trailing 15x with 20% expected earnings growth is at a forward `15 ÷ 1.20 = 12.5x` — meaningfully cheaper on the number that matters. Throughout this post the figures quoted are trailing unless noted, but when you value an individual Vietnamese stock you should usually work forward, because the growth is the whole point.

**The discount rate is the hidden variable behind every multiple.** This is the single idea to carry through the entire post. A P/E is not a free-floating number; it is the visible shadow of an invisible discount rate. The higher the rate at which a market discounts future cash flows, the less those cash flows are worth today, and the lower the price you will pay per dollar of current earnings — a lower P/E. Two markets with identical companies and identical growth will trade at different P/Es if, and only if, they discount the future at different rates. Almost everything that makes Vietnam's multiple "different" reduces to one fact: Vietnam's discount rate is structurally higher than a developed market's. Hold that thought, because it is the thread that ties together the foreign-room cap, the currency risk, the illiquidity, and the country risk premium into a single coherent story.

#### Worked example: reading a P/E as an earnings yield

Suppose the VN-Index trades at a P/E of 13.8 (its end-2024 level). Flip it to an earnings yield: `1 ÷ 13.8 = 7.25%`. Now compare that to the Vietnamese 10-year government bond yield, which was around 2.7% in local terms at end-2024 — wait, that is the local nominal yield in VND. The earnings yield of 7.25% sits well above the local risk-free rate, implying an equity risk premium of roughly `7.25% − 2.7% ≈ 4.5%` in VND terms. Now do the same for the S&P 500 at a P/E of 22.5: earnings yield `1 ÷ 22.5 = 4.4%`, against a US 10-year of ~4.57%, implying a *negative* spread of about −0.2%. **The takeaway: on an earnings-yield-minus-bond basis, Vietnam actually offered more compensation for equity risk than the US did at end-2024 — the low P/E is partly a high-discount-rate phenomenon, not pure pessimism.**

## Why the P/E discount persists

A low multiple that *stays* low for years is not a mispricing — it is a structural discount. The market is collectively demanding a higher earnings yield (lower price per dollar of earnings) because it perceives more risk or fewer buyers. Let me walk through each driver, because each one is a separate lever pressing the multiple down.

![VN-Index P/E time series 2015 to 2024](/imgs/blogs/vietnam-stock-market-valuation-vnindex-pe-dynamics-2.png)

**MSCI Frontier / Emerging-Market status.** Global index providers like MSCI classify markets into Developed, Emerging, and Frontier tiers. Vietnam is currently classified as a **Frontier market** with an ongoing campaign for promotion to Emerging. This classification is not a cosmetic label — it determines how many global passive dollars are *forced* to buy the market. Trillions of dollars track Emerging-Market indices; a tiny fraction track Frontier indices. When a stock is in the S&P 500, every S&P 500 index fund must hold it, creating a permanent, price-insensitive bid. Vietnamese stocks have almost none of that automatic demand. No auto-bid means the price has to be low enough to attract discretionary buyers, who are pickier and demand a discount.

**Foreign ownership limits (foreign room).** Vietnamese law caps foreign ownership at **49%** of most listed companies (and historically 30% for banks, though this has been adjusted upward for some). The pool of shares available to foreign investors is called **foreign room**. When foreigners have bought up to the cap, the room is "full," and no additional foreign capital can enter at the market price. This is a hard ceiling on demand from the deepest, most valuation-driven pool of capital in the world. A demand ceiling, by definition, caps the price.

**Thin liquidity.** Daily trading value on the HSX, while growing, is a fraction of developed-market exchanges. Thin liquidity means large investors cannot enter or exit positions without moving the price against themselves. Illiquidity is a risk, and investors demand to be paid for bearing it — in the form of a lower entry price, i.e., a lower P/E.

**Information asymmetry.** Disclosure standards, audit quality, and the availability of English-language financials are improving but still lag developed markets. When you cannot fully trust or fully understand the numbers, you pay less for them. This is a direct haircut to the multiple.

**VND currency risk.** A foreign investor earns returns in Vietnamese dong but spends in dollars or euros. The dong has historically depreciated against the dollar in a managed, gradual fashion — typically 1-3% per year, with occasional sharper adjustments. That expected depreciation is a drag on foreign returns, so foreigners demand a higher local-currency return to compensate, which again means a lower price.

Stack these five drivers and you have a market structurally priced to deliver a higher return than a developed market — which is the same thing as saying it trades at a lower multiple. None of them are about the companies being bad. They are about the market being small, gated, leveraged, and denominated in a softening currency.

It helps to see how these drivers reinforce one another rather than simply adding up. Frontier status keeps passive money out, which keeps liquidity thin; thin liquidity makes large positions risky, which keeps institutions away, which keeps the market retail-dominated; retail dominance amplifies the boom-bust cycle, which raises perceived risk, which justifies the foreign-room caps that gate the deepest pool of buyers. The system is self-reinforcing, which is why the discount is so persistent — it would take a structural change, like the MSCI Emerging-Market promotion discussed later, to break the loop rather than a single good earnings season.

A useful way to sanity-check the discount is to compare it to the local cost of capital directly. Vietnamese deposit rates have historically run 5-7% in nominal VND terms, and the State Bank of Vietnam's policy rate has cycled between roughly 4% and 6.5% over the past several years. When a "safe" bank deposit pays 6% in dong, an equity investor needs a meaningfully higher expected return to bear stock-market risk — say 13-16% in dong. That high local hurdle rate is the domestic mirror of the foreign investor's CRP: both groups, for different reasons, demand a high required return, and a high required return *is* a low multiple. The discount is not one group's pessimism; it is the equilibrium of a market where both local and foreign capital have high hurdle rates.

There is also a governance and ownership-concentration dimension worth naming. Many Vietnamese listed companies have a dominant founding family or a controlling state stake, which means the free float — the shares actually available to trade — is smaller than the headline market cap suggests. A small free float concentrates volatility (the same buying or selling pressure moves a smaller pool of shares more violently) and raises the risk that minority shareholders' interests diverge from the controlling owner's. Both effects are, once again, reasons to demand a higher return and pay a lower multiple. The pattern is relentless: every structural feature of this market points the same direction, toward a higher discount rate and a lower P/E.

#### Worked example: decomposing the discount

Take the end-2024 gap: VN-Index at 13.8x versus the S&P 500 at 22.5x. The earnings-yield gap is `7.25% − 4.44% = 2.81%`. Where does that ~2.8% of extra demanded return come from? A rough attribution: country/political risk ~1.0%, illiquidity ~0.6%, expected VND depreciation ~0.8%, and information/governance discount ~0.4%. These do not need to be precise — the point is that **the entire P/E gap can be reconstructed as a sum of identifiable, structural risk premiums, each of which a foreign buyer is rationally demanding compensation for.** The multiple is not "wrong"; it is the market clearing at a higher required return.

## The boom-bust cycle: how margin compresses multiples fast

The structural discount explains the *level* of the VN-Index P/E. The *volatility* of that P/E — the way it can halve in a year — comes from leverage and retail behavior.

![VN P/E cycle phases timeline](/imgs/blogs/vietnam-stock-market-valuation-vnindex-pe-dynamics-6.png)

Here is the cycle, phase by phase. In an **undervalued** phase, fear is high, margin balances are low, and the P/E sits near the bottom of its range — call it 10x. As confidence returns, a **momentum** phase begins: prices rise, gains attract new retail money, and investors lever up on margin to amplify returns. The P/E drifts up toward its average. Then comes **euphoria**: everyone is in, margin balances hit record highs, and the P/E overshoots to 19x as buyers pay any price. The system is now maximally fragile, because a large share of the float is held on borrowed money.

Then something — a rate hike, a policy shock, a fraud revelation, a global risk-off — triggers the **margin call** phase. Prices fall enough that brokers demand more collateral. Leveraged investors who cannot post collateral are sold out automatically. That forced selling pushes prices lower, triggering the next tranche of margin calls. This is a reflexive cascade, and it is why the **crash** phase is so fast: the P/E can fall from 19x to 10x in months, not years. Finally, the **recovery** phase arrives, but the recovery is often driven as much by *earnings growth resetting the denominator* as by price rising. When E grows and P is flat, P/E falls back to "cheap" even without a crash.

The historical record is brutal and instructive. In 2007, the VN-Index peaked near 1,170 amid a speculative frenzy, then collapsed to around 235 by early 2009 — an ~80% drawdown, one of the worst of any major market in the global financial crisis. In 2021, fueled by lockdown-era retail account openings and ultra-low rates, the index peaked above 1,500 with a P/E of 19.2x. In 2022, a combination of rising rates, a corporate-bond market scandal, and arrests of prominent business figures triggered a −33% crash, dragging the P/E to 9.8x. Same companies, same earnings power, less than half the multiple — that is leverage and sentiment, not fundamentals.

Why is the cycle so much more violent in Vietnam than in a developed market? The answer is the *combination* of high retail participation and high leverage operating on a thin float. In the United States, when prices fall, the marginal seller is often a rules-based institution rebalancing slowly, and the marginal buyer includes price-insensitive index funds and corporate buyback programs that step in mechanically. In Vietnam, when prices fall, the marginal seller is a leveraged individual facing a margin call, and there is no large price-insensitive buyer to absorb the supply — passive frontier money is too small to matter. Forced selling meets a thin bid, so the price gaps down hard. The same mechanics work in reverse on the way up: leveraged retail buying meets a thin offer, and prices gap up. The result is a market that overshoots fair value in both directions, which is precisely why anchoring to a *fair* multiple — rather than chasing the trend — is so valuable for a disciplined investor.

A second amplifier is the role of the corporate-bond market, which the 2022 episode exposed. Many real estate developers had funded themselves through opaque corporate-bond issuance, often sold to retail investors. When confidence in those bonds collapsed, developers faced a funding crunch, their equity values fell, and the contagion spread to the banks that held exposure and to the broader market through sentiment. This is a reminder that in a less-developed financial system, the channels of contagion are less ring-fenced — a problem in one corner (developer bonds) can transmit to the whole equity market faster than in a deeper, more compartmentalized market. For the valuation analyst, the practical implication is that the discount rate should carry a premium for exactly this kind of systemic fragility, and that the *correlation* between sectors rises sharply in a crisis, so diversification within the VN-Index protects you less than you would hope when it matters most.

#### Worked example: how a margin call cascade compresses the multiple

Suppose a stock trades at 50,000 VND with earnings per share of 4,000 VND, a P/E of 12.5x. An investor buys with 50% margin: 50,000 VND of stock funded by 25,000 VND equity and 25,000 VND broker loan. The broker requires the loan to stay below 70% of position value (a maintenance margin). Now the price falls 20% to 40,000 VND. Position value is 40,000; the 25,000 loan is now `25,000 ÷ 40,000 = 62.5%` of value — still safe. But the price falls another 15% to 34,000 VND: the loan is now `25,000 ÷ 34,000 = 73.5%`, breaching the 70% limit. The broker force-sells. That selling, multiplied across thousands of similarly leveraged accounts, pushes the price down further — say to 31,000 VND, a P/E of `31,000 ÷ 4,000 = 7.75x`. **Earnings never changed; the multiple fell from 12.5x to 7.75x purely because leveraged holders were forced to become sellers at the same moment.**

## Sector P/E differentials: why one market has many multiples

The "VN-Index P/E" is an average that hides enormous dispersion. Because the index is a mix of fundamentally different business models, the sector multiples diverge sharply, and understanding why is essential to valuing any individual stock.

![Sector P/E differentials across the VN-Index 2024](/imgs/blogs/vietnam-stock-market-valuation-vnindex-pe-dynamics-5.png)

**Banks (~9x).** Vietnamese banks trade at the lowest multiples in the index, typically 8-10x earnings. This is not a sign they are bad — many are the most profitable companies in the country, with ROEs above 20%. Banks trade at low P/Es everywhere in the world, for structural reasons: their earnings are leveraged (a bank is essentially a leveraged bond portfolio), cyclical (loan losses spike in downturns), and opaque (you cannot fully see asset quality from outside). The market applies a low multiple to compensate. Because banks are 30% of the index, they drag the *index* P/E down mechanically — the VN-Index looks cheap partly because it is one-third low-multiple banks.

**Industrials (~11-12x).** Steel, construction, logistics, and manufacturing names trade modestly below the index average. These are cyclical, capital-intensive businesses with thin margins, so the market pays a middling multiple.

**Real estate developers (~15-20x).** During land-price booms, real estate developers can command high multiples because the market extrapolates booming pre-sale revenue. But these are among the most volatile and leverage-dependent businesses in the country, and their multiples collapse hardest in busts. The 2022 corporate-bond crisis hit this sector especially hard.

**Consumer (~18x).** Branded consumer staples — the dominant dairy company, brewers, retailers — trade at the kind of premium multiples you see globally, because their earnings are stable, defensive, and growing with a young, urbanizing population.

**Technology (~22x).** The small tech slice, led by the country's flagship IT-services and software firm, trades at developed-market-like multiples because it has structural growth, export revenue (partially dollar-denominated, which dampens VND risk), and is the kind of company global investors are happy to own at the foreign-room cap.

The lesson for valuation is that you must value each company on the method appropriate to its business model and against its *sector* multiple, not the index average. Pricing a bank at the consumer multiple, or a developer at the bank multiple, will be wildly wrong.

There is a second, subtler lesson hiding in the sector dispersion: the index multiple drifts over time partly because the *composition* drifts, not only because individual sector multiples move. As Vietnam's economy matures, the technology and consumer slices grow faster than the bank and steel slices. If the high-multiple sectors gain index weight, the blended index P/E rises even if no single sector re-rates. This is exactly what happened in the US over the past two decades, where the rise of high-multiple technology pulled the S&P 500's average multiple up structurally. A forward-looking Vietnam analyst should therefore expect some upward drift in the index multiple over the coming decade purely from mix shift — a reason not to anchor too rigidly to the historical ~14x average when thinking many years out.

It is also worth being precise about why real estate developers carry such a wide multiple range. In a land-price boom, a developer recognizes revenue on pre-sales of projects that are years from completion, so reported earnings can surge and the market extrapolates that surge into a high multiple. But the same accounting that flatters earnings on the way up punishes them on the way down: when sales stall, the recognized revenue dries up while the debt taken on to acquire land remains, and earnings — and the multiple — collapse together. This is why a developer's P/E is among the least reliable single-number valuations in the market, and why analysts often switch to net-asset-value (NAV) approaches — valuing the land bank and projects directly — for developers, just as they switch to P/B for banks. The general principle is that the more an industry's reported earnings diverge from its underlying economics, the less you should trust its P/E and the more you should reach for a balance-sheet-based method.

#### Worked example: why the index multiple is a weighted blend

Build a toy three-sector index: 30% banks at 9x, 30% real estate at 17x, 40% everything-else at 14x. Naively averaging the multiples is wrong because P/E is a ratio; the correct way is to weight *earnings yields* (the inverses) by market value, then invert. Earnings yields: banks `1/9 = 11.1%`, real estate `1/17 = 5.9%`, other `1/14 = 7.1%`. Weighted earnings yield: `0.30×11.1% + 0.30×5.9% + 0.40×7.1% = 3.33% + 1.77% + 2.84% = 7.94%`. Invert: `1 ÷ 0.0794 = 12.6x`. **The blended index multiple (~12.6x) sits below a naive average of the three (13.3x) precisely because the heavy bank weight pulls the harmonic blend down — the index looks cheaper than its components on a simple-average basis.**

## Applying global valuation methods in a Vietnamese context

Now the heart of the matter. The valuation *methods* — discounted cash flow, the capital asset pricing model, relative multiples — are universal. What changes in Vietnam are the *inputs*, especially the discount rate. Get the inputs right and the same machinery works perfectly.

### CAPM with a country risk premium

The **Capital Asset Pricing Model (CAPM)** says the return an investor should require on a stock equals the risk-free rate plus a premium for bearing market risk, scaled by the stock's sensitivity to the market (its **beta**):

`Required return = Risk-free rate + Beta × Equity risk premium`

If you tried to value a Vietnamese stock by plugging in a US risk-free rate and a US equity risk premium, you would get a required return far too low, and therefore a fair-value price far too high — you would conclude every Vietnamese stock is a screaming buy, and you would be repeatedly wrong. The fix is the **country risk premium (CRP)**: an extra slug of required return added to compensate for the country-specific risks — default risk, political risk, currency risk — that you do not bear in a developed market. The professor Aswath Damodaran publishes widely-used CRP estimates; for Vietnam, the figure has historically sat in the **5-7%** range.

The CRP-adjusted CAPM becomes:

`Required return = Risk-free rate + Beta × (Mature-market ERP) + CRP`

(There are variants — some scale the CRP by a lambda that measures the company's exposure to the local economy — but adding it as a flat term is the standard, defensible approach.)

![Country risk premium before and after on required return](/imgs/blogs/vietnam-stock-market-valuation-vnindex-pe-dynamics-4.png)

Once you raise the required return by the CRP, the "justified" P/E falls, and the market's low multiple suddenly looks correct. This is the single most important insight in Vietnamese valuation: **the low P/E is not the market saying the companies are bad; it is the market discounting fair cash flows at a fairly high rate.** A foreign analyst who forgets the CRP will perpetually think Vietnam is cheap and perpetually be confused when it stays cheap.

#### Worked example: a CRP-adjusted required return

Value a mid-cap Vietnamese industrial. Use a US 10-year Treasury risk-free rate of 4.5% as the base (working in a USD-equivalent framework that already prices the dollar risk-free rate), a mature-market equity risk premium of 5.0% (Damodaran's implied US ERP was ~4.6% in early 2025; round to 5.0%), a stock beta of 1.1, and a Vietnam CRP of 6.0%. Then:

`Required return = 4.5% + 1.1 × 5.0% + 6.0% = 4.5% + 5.5% + 6.0% = 16.0%`

Compare that to what you would have gotten ignoring the CRP: `4.5% + 1.1 × 5.0% = 10.0%`. **The CRP nearly doubled the cost of equity from 10% to 16%, and a 16% discount rate justifies a far lower multiple than a 10% rate — which is exactly why the same company trades at, say, 11x in Hanoi and would trade at 18x if it were listed in New York.**

### The thin-trading beta problem

CAPM needs a beta, and here Vietnam has a subtle trap. **Beta** measures how much a stock moves relative to the market. It is estimated by regressing the stock's returns on the index's returns. But in a thinly-traded market, a stock may not trade every day, or may trade with stale prices. When a stock's price does not update, its measured returns are artificially smooth, and a smooth return series produces a *low* measured beta — sometimes well below 1, or even near zero — even for a genuinely risky stock.

This is dangerous. A naively low beta plugged into CAPM gives a low required return and an inflated fair value. The thinly-traded small-cap that *looks* low-risk on its beta is often the riskiest thing in the market. The practitioner fixes for this include: using a longer estimation window, using the sector's beta rather than the individual stock's, adjusting for non-trading (the Dimson or Scholes-Williams corrections that add lagged market returns to the regression), or simply applying a floor — never trust a frontier-market small-cap beta below, say, 0.8. **In Vietnam, low observed beta is more often a measurement artifact than a statement of true risk.**

A related approach many practitioners prefer for frontier markets is the **bottom-up beta**. Instead of regressing the individual stock's noisy, thinly-traded returns against the index, you take the average beta of comparable companies in the same industry from a deeper, more liquid market (often a basket of global or regional peers), un-lever it to strip out their capital structures, then re-lever it for the Vietnamese company's own debt level. This gives a beta grounded in the *business risk* of the industry rather than the *trading noise* of one illiquid ticker. For a Vietnamese steel maker, you might start from the global steel-industry beta, un-lever and re-lever for the local company's leverage, and arrive at a far more trustworthy number than the local regression would give. The bottom-up method is more work, but in a market where price data is noisy, grounding the beta in fundamentals is exactly the right instinct.

Note also what beta does *not* capture: the country risk premium handles the systematic country-level risks, but idiosyncratic governance risk — a controlling shareholder acting against minorities, weak disclosure, related-party transactions — is not in beta at all. For the riskiest names, analysts sometimes add a small company-specific premium on top of the CAPM output, or simply apply a haircut to their fair-value estimate. The discipline is to be explicit about every premium you are charging and why, rather than burying risk in a single fudged number.

### ERP adjustments

The equity risk premium itself deserves a Vietnam-specific thought. The US ERP can be estimated from a century of data. Vietnam's market is young — the HSX opened in 2000 — and its short history is dominated by two giant crashes, so a purely historical Vietnamese ERP is statistically meaningless. The standard approach is therefore the *building-block* one used above: take a reliable mature-market ERP, then add the CRP on top. You are not trying to measure Vietnam's ERP directly; you are constructing it as "the world's equity risk plus Vietnam's country risk."

#### Worked example: from required return to a justified P/E

The link between a discount rate and a multiple runs through the constant-growth (Gordon) model. For a company paying out a fraction of earnings as dividends, the justified P/E is approximately `Payout ratio × (1 + g) ÷ (r − g)`, where `r` is the required return and `g` is the long-run earnings growth. Take a Vietnamese industrial with a 40% payout ratio, long-run growth of 6%, and the two discount rates from before. With the CRP (r = 16%): `P/E = 0.40 × 1.06 ÷ (0.16 − 0.06) = 0.424 ÷ 0.10 = 4.2x`. Without the CRP (r = 10%): `P/E = 0.40 × 1.06 ÷ (0.10 − 0.06) = 0.424 ÷ 0.04 = 10.6x`. **The same business is "worth" 10.6x at a developed-market discount rate but only 4.2x once Vietnam's country risk is priced in — the discount rate, not the earnings, drives almost the entire valuation difference.**

## The foreign investor's perspective

A foreign buyer faces three things a local does not: a currency translation, a hard ownership ceiling, and the risk of getting money back out. Each has a direct price consequence.

### FX hedging cost and the currency drag

A foreign investor's total return is the local stock return plus the change in the VND/USD exchange rate. If a stock returns 15% in dong but the dong falls 3% against the dollar, the dollar return is roughly 12%. Hedging that currency exposure has a cost — driven by the interest-rate differential between Vietnam and the US — and that cost eats into returns. Whether they hedge or not, foreigners require a higher *local* return to clear the same hurdle in dollars, which translates into demanding a lower entry price (lower P/E).

### The foreign-room premium

This is the most Vietnam-specific valuation quirk of all. When a stock's foreign room is full — foreigners already own up to the 49% cap — a foreign investor who wants in cannot buy at the on-screen market price. They must buy from another foreigner, in a negotiated off-market block trade, and because demand exceeds the fixed supply of "foreign-eligible" shares, those blocks change hands at a **premium** to the local market price. Premiums of **10-30%** are common for the most-coveted, room-locked names.

![Foreign room premium by stock 2024](/imgs/blogs/vietnam-stock-market-valuation-vnindex-pe-dynamics-7.png)

The foreign-room premium effectively creates *two prices* for the same share: the local price (what a Vietnamese investor pays) and the foreign price (local price plus premium). For valuation, this matters in two directions. If you are a foreign buyer, your *effective* cost basis — and therefore your *effective* P/E — is higher than the screen, so a stock at a 13x local P/E with a 20% room premium actually costs you `13 × 1.20 = 15.6x`. And if you already *own* a room-locked stock, that premium is a real, realizable asset — you can sell your shares to another foreigner at the premium price.

There is a subtle valuation question about *who* the foreign-room premium accrues to. When you buy a room-locked stock at a premium, you have paid for a scarce right — the right to own foreign-restricted shares. As long as the room stays full, you can sell that right to the next foreigner. But if Vietnam relaxes the foreign-ownership caps, or is promoted to Emerging-Market status and the supply of foreign-eligible shares effectively expands, the scarcity evaporates and the premium can compress toward zero. So the foreign-room premium is itself a *position* with its own risk: you are long the persistence of the cap. A foreign investor must decide whether the premium they pay today will still be there when they exit, which is a genuinely different question from whether the underlying company is a good business.

This also means the foreign-room premium is information. A stock trading at a large and persistent room premium is telling you that the world's valuation-driven capital wants more of it than the rules allow — a kind of revealed preference for quality. The names that command the biggest premiums (the flagship bank, the dominant consumer-staples company, the leading tech exporter) are exactly the ones global investors would overweight if they could. Reading the premium as a quality signal, while pricing it correctly into your effective cost basis, is a uniquely Vietnamese piece of valuation craft.

### Repatriation risk

Finally, getting capital out. Vietnam permits foreign investors to repatriate proceeds, but the process runs through the banking system and is subject to documentation and, in stress scenarios, the risk of delay or capital controls. This tail risk — small in normal times, real in a crisis — is one more reason foreigners demand a discount. You pay less for an asset you might not be able to sell and repatriate freely.

The repatriation channel is also where currency risk and liquidity risk compound. In a crisis, the dong tends to weaken at exactly the moment foreign investors want to exit, and the queue to convert VND to dollars lengthens. So the worst-case scenario stacks three losses: the stock falls, the currency falls, and the exit is slow. Each of these is individually modest in normal times, but they are *correlated in the tail* — they all go wrong together precisely when it matters most. This correlation is the deepest reason the structural discount exists and persists: a rational foreign investor prices not just the average outcome but the ugly joint outcome, and demands compensation for it in the form of a lower entry multiple. Understanding that the discount is partly insurance against a correlated-crisis tail, rather than simple pessimism about growth, is what separates a sophisticated read of the Vietnamese market from a naive one.

#### Worked example: the foreign-room premium calculation

Vietcombank trades at 90,000 VND per share on the local market, but its foreign room is full. A foreign fund wants to buy and finds a block available from a departing foreign holder at a 15% premium. The foreign purchase price is `90,000 × 1.15 = 103,500 VND`. If VCB's earnings per share are, say, 6,000 VND, the local P/E is `90,000 ÷ 6,000 = 15.0x`, but the foreign buyer's *effective* P/E is `103,500 ÷ 6,000 = 17.25x`. **The room premium quietly turned a 15x stock into a 17.25x stock for the foreign buyer — and any foreign analyst who values VCB at the screen price is understating what they will actually pay by 15%.**

## Valuing a Vietnamese bank: use P/B, not P/E

Banks are 30% of the index, so you cannot value the Vietnamese market without being able to value a bank. And the right tool is **not** P/E. It is **price-to-book (P/B)**.

**Book value** is the accounting net worth of a company: total assets minus total liabilities, also called shareholders' equity. **Price-to-book** is the market price divided by book value per share — how many times net worth you are paying.

Why P/B for banks? Three reasons. First, a bank's earnings are volatile and easily distorted by loan-loss provisions — in a good year provisions are low and earnings look great; in a bad year a single large write-off can wipe out earnings, making P/E meaningless (a near-zero E sends P/E to infinity). Book value is far more stable. Second, a bank's balance sheet *is* its business — it is a portfolio of loans and deposits — so book value is a meaningful measure of the productive asset base in a way it is not for, say, a software company. Third, there is a clean, theoretically grounded link between P/B, profitability, and the discount rate that lets you compute a justified P/B directly.

That link comes from the Gordon model applied to book value. If a bank earns a return on equity (ROE) of `r_e`, grows its book at rate `g`, and investors require a return `r`, then the justified price-to-book is:

`P/B = (ROE − g) ÷ (r − g)`

This formula is the workhorse of bank valuation worldwide, and it makes the logic crisp: a bank should trade above book (P/B > 1) exactly when its ROE exceeds the return investors require. If ROE = r, the bank is just earning its cost of capital and should trade at exactly book (P/B = 1). If ROE > r, it is creating value and deserves a premium. The faster it grows that excess return (`g`), the bigger the premium.

Now you can see why Vietnamese banks, despite low P/Es, are not necessarily cheap on P/B: a bank with a 20% ROE and a 15% required return justifies a P/B well above 1, and if it is already trading there, it is fairly valued — its low *P/E* is just the natural consequence of banks earning leveraged returns that the market caps at a low earnings multiple.

There is a clean bridge between the P/B and P/E views that is worth making explicit, because it dissolves any apparent contradiction. By definition, `P/E = (P/B) ÷ ROE` — price-to-book divided by return on equity. So a bank trading at 1.7x book with a 20% ROE has a P/E of `1.7 ÷ 0.20 = 8.5x`. The "low" 8.5x P/E and the "rich" 1.7x P/B are the *same valuation* expressed two ways; they are perfectly consistent. The reason banks look cheap on P/E but full on P/B is simply that they earn very high returns on equity, and high ROE mathematically turns a healthy P/B into a low-looking P/E. Once you internalize this identity, the bank "discount" stops being mysterious: it is arithmetic.

The biggest judgment call in Vietnamese bank valuation is not the formula but the *quality of book value itself*. The `(ROE − g)/(r − g)` model assumes reported book value is real. But a bank's book value can be overstated if non-performing loans (NPLs) have not been fully recognized — if bad loans are still carried at face value rather than written down. Vietnamese banks have, at various points, carried restructured or problem loans in ways that flatter reported book value and ROE. The careful analyst therefore stress-tests the book: estimate the true NPL ratio, haircut book value for under-provisioned bad loans, and recompute. A bank trading at a reported 1.5x book might be at 1.9x book on a conservatively-adjusted basis — meaningfully more expensive. The formula is universal; the integrity of its inputs is where Vietnam-specific diligence earns its keep.

A final practitioner note on banks: ROE is not a constant, and the model is acutely sensitive to the *sustainable* level you assume. A bank may post a 22% ROE in a credit boom and 12% in a bust. Plugging the boom-year ROE into the formula will badly overvalue the bank. The right input is a mid-cycle, sustainable ROE — what the bank earns on average across good years and bad. Normalizing ROE across the cycle is the single most important discipline in bank valuation, in Vietnam or anywhere.

#### Worked example: P/B valuation of a Vietnamese bank

Value a Vietcombank-like bank. Assume a sustainable ROE of 20%, a long-run book-value growth rate of 8% (a fast-growing bank in a fast-growing economy), and a CRP-adjusted required return of 15% (risk-free 4.5% + beta 0.9 × ERP 5.0% + CRP 6.0% ≈ 15.0%). Then:

`Justified P/B = (0.20 − 0.08) ÷ (0.15 − 0.08) = 0.12 ÷ 0.07 = 1.71x`

If the bank's book value per share is 50,000 VND, the justified price is `1.71 × 50,000 = 85,700 VND`. If it trades at 90,000 VND, it is roughly fairly valued — even slightly rich — *despite* a single-digit-looking P/E. Now stress the inputs: if growth slows to 5%, `P/B = (0.20 − 0.05) ÷ (0.15 − 0.05) = 0.15 ÷ 0.10 = 1.50x`, a justified price of 75,000 VND — the same bank is 17% cheaper just from a slower growth assumption. **A bank's P/B is exquisitely sensitive to the spread between ROE, growth, and the discount rate; the low P/E that lures foreign buyers can sit alongside a P/B that says the stock is already full-priced.**

## Common misconceptions

Most errors in valuing Vietnam come from a single root: applying a developed-market reflex to a frontier-market structure. Each of the misconceptions below is a specific instance of that root error, and each is corrected by returning to the same principle — the multiple is the shadow of a discount rate, and Vietnam's discount rate is structurally high.

**"Vietnam's low P/E means it is cheap."** No — the low P/E is largely a high-discount-rate phenomenon. At a CRP-adjusted required return of 16%, a 13x P/E can be exactly fair value. Cheapness is a statement about price *relative to a correct discount rate*, and the correct rate for Vietnam is 500-700 basis points above a developed market's. A 13x P/E in Vietnam is not the same bargain as a 13x P/E in the US, any more than 100,000 VND is the same as \$100.

**"The VN-Index P/E and the S&P 500 P/E are comparable numbers."** They are not, for two compounding reasons: the discount rates differ (CRP), and the sector mixes differ. The VN-Index is one-third low-multiple banks; the S&P 500 is one-third high-multiple technology. Even at identical discount rates, the index multiples would diverge purely from composition. Compare sector-to-sector, never index-to-index.

**"A low beta means a Vietnamese stock is low-risk."** Often the opposite. In a thinly-traded market, a low measured beta is usually a statistical artifact of stale prices, not a true measure of risk. The illiquid small-cap with a beta of 0.5 may be the most dangerous holding you have. Use sector betas or a beta floor.

**"Foreign and local investors pay the same price."** Not when the foreign room is full. The room premium creates two prices for the same share, and a foreign buyer's effective P/E can be 10-30% higher than the screen. Always value from your *own* effective cost basis.

**"Banks with low P/Es are bargains."** Banks should be valued on P/B against ROE, not on P/E. A bank with a 9x P/E and a 1.7x P/B that matches its `(ROE − g)/(r − g)` justified level is fairly valued, full stop. The low P/E is a structural feature of leveraged-earnings businesses, not a discount. And before trusting any of it, haircut the reported book value for under-provisioned bad loans — a bank that looks fair on reported numbers can be expensive on a conservatively adjusted balance sheet, which is precisely where careful Vietnam-specific diligence pays off.

## How it shows up in real markets

**The 2021-2022 cycle.** This is the cleanest modern illustration of every force in this post. Through 2020-2021, lockdown-era account openings flooded the retail market, margin balances hit records, and the VN-Index P/E climbed from ~15.5x (2020) to 19.2x (2021) as euphoria peaked above index 1,500. Then 2022 brought rising global rates, a domestic corporate-bond scandal centered on real estate developers, and high-profile arrests. The reflexive margin-call cascade did its work: the index fell roughly 33%, and the P/E crashed to 9.8x. Crucially, aggregate corporate earnings did not fall by anything like that amount — the multiple collapse was a leverage-and-sentiment event, not an earnings event. By end-2023 and into 2024, with earnings growing and prices recovering modestly, the P/E settled back to the 12.5-13.8x range, near its long-run average.

**The valuation lesson from that cycle:** an investor using a constant developed-market discount rate would have seen "cheap" at 19x in 2021 (relative to the S&P's 28x) and bought the top. An investor anchored to Vietnam's structural discount and the leverage in the system would have recognized 19x as *expensive for Vietnam* — well above its ~14x average — and the maxed-out margin balances as the warning sign. The right reference is the market's own history and its CRP-adjusted fair multiple, not a cross-border comparison.

#### Worked example: was the VN-Index cheap or expensive, 2020-2024?

Anchor to a fair multiple of ~14x (the decade average, consistent with a CRP-adjusted required return). Walk the years: 2020 at 15.5x was slightly rich; 2021 at 19.2x was ~37% above fair — clearly expensive, and the maxed margin balances confirmed fragility; 2022 at 9.8x was ~30% below fair — genuinely cheap, the moment the structural discount overshot into panic; 2023 at 12.5x was modestly cheap; 2024 at 13.8x was about fair. Now cross-check against earnings: aggregate VN-Index earnings grew through most of this window, so the 2022 multiple collapse was *not* matched by an earnings collapse — the denominator held while the numerator panicked. **Measuring against the market's own ~14x fair multiple, rather than against the S&P 500, correctly flags 2021 as the time to trim and 2022 as the time to buy — the exact opposite of what a naive cross-border P/E comparison would have told you.**

**The benchmark comparison, done correctly.** At end-2024, the VN-Index at 13.8x sat below MSCI Emerging Markets (12.4x is actually *below* VN — Vietnam trades at a slight premium to the broad EM index, reflecting its growth) and well below the S&P 500 (22.5x) and far above MSCI Frontier (10.8x). Reading this correctly: Vietnam is priced like what it is — a high-growth frontier market on the cusp of EM promotion, cheaper than developed markets because of its structural discounts, but already carrying a small premium to the frontier average because investors are pricing in the growth and the potential MSCI upgrade.

![VN vs S&P 500 vs MSCI EM and Frontier P/E 2024](/imgs/blogs/vietnam-stock-market-valuation-vnindex-pe-dynamics-3.png)

It is worth dwelling on why Vietnam trades at a premium to the broad MSCI Frontier index but a discount to MSCI Emerging Markets. Within the frontier universe, Vietnam is the standout: a large, fast-growing economy with a deep manufacturing base, a young demographic profile, and a credible path to EM promotion. Investors pay up for that relative quality, which is why Vietnam sits above the frontier average. But it has not yet crossed into the EM tier, so it still lacks the forced passive bid and still carries the full frontier-market structural discounts. The result is a market caught between two classifications — too good for the frontier multiple, not yet eligible for the EM one. This in-between status is itself a source of the valuation puzzle, and it is why the single most-watched catalyst for a re-rating is the MSCI classification decision.

That last point — the MSCI upgrade — is the live catalyst. If Vietnam is promoted from Frontier to Emerging status, a wave of passive EM money would be forced to buy the market, adding exactly the price-insensitive auto-bid the market currently lacks. That single change would compress the structural discount and re-rate the multiple higher. It is the clearest example of how a *structural* driver of the P/E, not an earnings driver, could move the whole market — and why understanding the structure, not just the earnings, is the heart of valuing Vietnam.

## Putting it together

One more discipline ties the whole framework together: always state your valuation in terms of an *implied return*, not just a fair-value price. Because the discount rate is the hidden driver of every multiple, the cleanest way to express a Vietnam valuation is "at today's price, this stock offers an expected return of X% in dong, or X% minus expected depreciation in dollars." Framing it this way forces you to be explicit about the CRP, the currency drag, and the growth assumption, and it makes cross-market comparison honest: a 16% expected dong return on a Vietnamese industrial and a 9% expected dollar return on a US peer can be compared directly once you net out expected depreciation and the differing risk. The P/E is a shorthand; the implied return is the substance. Train yourself to translate every multiple back into the return it implies, and the Vietnamese market stops looking anomalous and starts looking like what it is — a coherent, high-required-return market.

Vietnamese equity valuation is not a different discipline; it is the same discipline with the country-specific inputs filled in honestly. The methods — P/E read as an earnings yield, CAPM for the discount rate, the Gordon model linking rate to multiple, P/B for banks — are universal. What you must change is the discount rate (add a 5-7% country risk premium), the beta (distrust thinly-traded low betas), the method per sector (P/B for the bank-heavy core), and your effective cost basis (add the foreign-room premium if you are buying from abroad). Do that, and the "anomalously cheap" market resolves into a perfectly rational one: a small, gated, leveraged, fast-growing market clearing at a high required return. The low P/E was never the mystery. The discount rate behind it was the answer all along.

## Further reading & cross-links

- [Risk and required return: CAPM, beta, and the cost of capital](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital) — the foundation for the CRP-adjusted discount rate used throughout this post.
- [The price-to-earnings ratio: valuing stocks on earnings](/blog/trading/asset-valuation/price-to-earnings-ratio-pe-valuation-stocks) — the core multiple, read as an earnings yield.
- [The price-to-book ratio: valuing equity on net worth](/blog/trading/asset-valuation/price-to-book-ratio-pb-valuation-equity) — the right tool for the bank-heavy VN-Index core.
- [WACC: the weighted average cost of capital](/blog/trading/equity-research/wacc-weighted-average-cost-capital) — how the cost of equity feeds a full discount rate.
- [Interest rates, bonds, and stocks: how they move together](/blog/trading/macro-trading/interest-rates-bonds-stocks-relationship) — the rate environment that drives the discount rate and the margin cycle.
