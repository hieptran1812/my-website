---
title: "What Is Value? The Philosophy and Frameworks of Asset Pricing"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A beginner-to-practitioner guide to the three philosophical frameworks of asset value — intrinsic, relative, and market — and how each maps to a real finance method."
tags: ["valuation", "asset-pricing", "intrinsic-value", "market-value", "fundamental-analysis", "price-vs-value", "investment-philosophy", "dcf", "comps", "efficient-markets"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Price is what you pay; value is what you get — and the gap between them is where every investing insight lives.
>
> - There are three fundamentally different answers to "what is this worth?": what it earns (intrinsic value), what similar things sell for (relative value), and what a buyer will pay right now (market/liquidation value).
> - Each framework maps to a real finance method: DCF for intrinsic, comparable multiples for relative, and book/liquidation analysis for market value.
> - The frameworks often disagree — that disagreement is not a bug, it is the signal that tells you which is most trustworthy in a given situation.
> - At VN-Index P/E of 11x in 2022, you were paying roughly half what you would pay for US earnings at the same time — whether that was cheap depends entirely on which framework you use and what you believe about future growth.

Two investors are staring at the same stock. The price on their screens says \$100. One of them — call her Ana — says "this is cheap, I'm buying." The other — call him Ben — says "this is expensive, I'm selling." They are both looking at the same number. They are not disagreeing about the price. They are disagreeing about the *value*.

This distinction, between price and value, is the most important idea in all of finance. Every bubble, every crash, every great investment, every catastrophic loss traces back to a moment when price and value diverged — and someone noticed, or didn't. Warren Buffett has called this the most fundamental insight in investing. Benjamin Graham, his teacher, made it the spine of a 700-page book. And yet most people who follow markets have never thought carefully about what "value" actually means.

The answer, it turns out, is not one thing. There are three distinct answers to "what is this worth?" — three philosophical frameworks, each internally consistent, each capturing something real, and each capable of leading you to a wildly different number for the same asset. Understanding all three, and knowing when each is most reliable, is the foundation of everything else in valuation. The diagram below maps the terrain.

![Three frameworks of asset value — intrinsic, relative, and market — mapped to their finance methods](/imgs/blogs/what-is-value-philosophy-frameworks-asset-pricing-1.png)

## Foundations: Price, Value, and Why They Differ

Before we can talk about frameworks, we need to build the vocabulary from zero. Every term introduced in this section will be used for the rest of the post — and for the rest of this series.

### What is a price?

A *price* is the number at which a transaction actually occurs. You walk into a market, you find a willing seller, and you agree on a number. That is the price. It is objective in the narrowest sense: it is what actually happened. It is also the only number that everyone involved in a market can see in real time.

Prices are set by the intersection of supply and demand at a moment in time. They incorporate everything that every buyer and seller knows, believes, fears, and desires at that instant. They can be moved by emotions, rumors, liquidity needs, margin calls, index rebalancing, tax-loss harvesting, and a hundred other factors that have nothing to do with the underlying asset's economics.

### What is value?

*Value* is an estimate of what something is *worth* — an assessment that exists independently of the current price. Value is inherently subjective: it depends on who is doing the estimating, what information they have, what discount rate they use, what growth they assume, and which framework they apply.

The key word is *estimate*. Value is never a fact; it is always a judgment. Ana estimates \$100 stock is worth \$150, so she calls it cheap. Ben estimates it is worth \$70, so he calls it expensive. They disagree not because one of them is bad at math, but because they are starting from different assumptions — different growth forecasts, different discount rates, different frameworks for what "worth" even means.

### The margin of safety

If value is an estimate, then the gap between your estimated value and the current price is what Graham called the *margin of safety* — the cushion that protects you if your estimate is wrong. If you believe something is worth \$120 and you can buy it at \$80, your margin of safety is \$40, or 33%. If your estimate is wrong by \$30 in the wrong direction, you still break even. If you buy at \$119, hoping to sell at \$120, one small estimation error destroys your return.

![Price vs intrinsic value — the margin of safety is the gap between market price and estimated worth](/imgs/blogs/what-is-value-philosophy-frameworks-asset-pricing-2.png)

The margin-of-safety concept is Framework 1 — intrinsic value — in its purest form. But it already implies something important: before you can know whether your margin of safety is \$40 or zero, you need a *method* for estimating value. That method is what the three frameworks provide.

### Why prices and values differ

If everyone used the same framework with the same inputs, prices would always equal value. They don't, for several reasons:

1. **Different information.** One investor has done deep research on a company's supply chain; another is relying on last quarter's headline earnings. They are not valuing the same thing.
2. **Different time horizons.** A long-term investor pricing a business on its five-year cash flows will arrive at a very different number than a trader pricing it on next quarter's earnings.
3. **Different frameworks.** An intrinsic-value investor might see a \$150 value while a relative-value investor compares it to peers trading at 15× earnings and calls it fairly priced at \$100.
4. **Non-economic forces.** A fund that owns a stock needs to raise cash to meet redemptions. It will sell at any price. That sale has nothing to do with value.

These differences create the opportunity that valuation tries to exploit: if you can estimate value more accurately than the market, you can buy things that are genuinely cheaper than they appear, or avoid things that look cheap but aren't.

---

## Framework 1 — Intrinsic Value: What the Asset Actually Earns

The first and oldest framework for value is intrinsic value: the idea that an asset is worth the sum of all the future cash flows it will generate for its owner, discounted back to the present at a rate that reflects the riskiness of those flows.

This framework has a long intellectual history. The core idea was formalized by John Burr Williams in *The Theory of Investment Value* (1938) and has been the dominant academic and practitioner framework ever since. The Discounted Cash Flow model (DCF) is its most common implementation; the Dividend Discount Model (DDM) is its oldest.

### The intuition before the formula

Imagine you are considering buying a vending machine. The machine will generate \$1,000 in profit this year, and you expect that to stay flat forever. How much should you pay for it?

You would not pay any amount — at some price, the investment becomes unattractive. If you could earn 5% per year safely in a government bond, you would need the vending machine to return at least 5% to be worth owning. So the most you would pay is \$1,000 / 0.05 = \$20,000. That is the *present value* of a perpetuity at a 5% discount rate.

Now suppose you expect profits to grow 2% per year. The formula becomes \$1,000 / (0.05 − 0.02) = \$33,333. Growth added \$13,333 in value. That is the Gordon Growth Model — the simplest version of the intrinsic value framework.

### The formal definition

For an asset that generates cash flows $CF_t$ over $N$ years and then a terminal value $TV$:

$$V_0 = \sum_{t=1}^{N} \frac{CF_t}{(1+r)^t} + \frac{TV}{(1+r)^N}$$

Where:
- $V_0$ = intrinsic value today
- $CF_t$ = expected cash flow in year $t$
- $r$ = discount rate (the required return, reflecting risk)
- $TV$ = terminal value (usually a perpetuity: $CF_{N+1} / (r - g)$, where $g$ is the long-run growth rate)

The discount rate $r$ is not arbitrary — it is the return you *require* to take on the risk of this particular asset. A riskier asset requires a higher return, which means the same cash flow is worth less when discounted at a higher rate. This is why two companies with identical cash flows but different risk profiles will have different intrinsic values. For a deep dive into discount rates and how they are computed, see [WACC and the cost of capital](/blog/trading/equity-research/wacc-weighted-average-cost-capital).

### What the intrinsic value framework requires

To apply this framework, you need:

1. A forecast of future cash flows (earnings, free cash flow, dividends)
2. A discount rate that reflects the asset's risk
3. A terminal value assumption (how the business performs after your explicit forecast period)

All three are estimates. All three are uncertain. That uncertainty is the framework's greatest weakness and also its greatest strength: it forces you to make your assumptions explicit, so you can be wrong in a traceable, correctable way.

#### Worked example: The pizza shop

Let us value a small pizza restaurant in Ho Chi Minh City. The owner wants to sell. You are considering buying it.

The restaurant generates \$60,000 in profit per year (after the owner's salary). You expect profits to grow at 3% per year because the neighborhood is developing. You require a 12% return on this investment — it is risky, illiquid, and you are betting on a single location. How much is the business worth?

Using the Gordon Growth Model:

$$V = \frac{\$60,000}{0.12 - 0.03} = \frac{\$60,000}{0.09} = \$\mathbf{667,000}$$

So the intrinsic value of this pizza shop is approximately \$667,000. If the owner is asking \$500,000, you have a \$167,000 margin of safety — you are buying at roughly 75 cents on the dollar. If they are asking \$800,000, you are paying above intrinsic value and should either negotiate, find a reason your estimates are too pessimistic, or walk away.

The one-sentence intuition: intrinsic value converts an uncertain future stream of cash into a single present number by asking "what would I need to earn to justify this risk?"

---

## Framework 2 — Relative Value: What Comparable Things Sell For

The second framework takes a completely different starting point: instead of estimating cash flows and discount rates from scratch, you look at what *similar* assets are currently trading for and price the asset by comparison.

This is the *comps* approach — comparable company analysis, comparable transaction analysis, or simply "trading multiples." It is the most widely used method in practice because it is fast, it requires no complex forecasting, and it directly reflects current market sentiment.

### The intuition

When you buy a house, the first thing a real estate agent does is pull up "comparable sales" — houses of similar size, condition, and location that sold recently. If three similar houses sold for \$500,000 last month, you have a reasonable prior that your house is worth roughly \$500,000. You do not need to forecast the rental income, compute a discount rate, and project a terminal value. The market has spoken.

The same logic applies to stocks. If five similar technology companies are trading at an average of 20× earnings, and your company earns \$5 per share, the relative-value estimate is \$100 per share. Simple, fast, grounded in current market reality.

### The multiples menu

Different assets use different multiples, each chosen to normalize for something that makes raw revenue or profit comparison misleading:

| Multiple | Formula | Best for |
|---|---|---|
| P/E (Price-to-Earnings) | Price / Earnings per share | Mature, profitable companies |
| EV/EBITDA | Enterprise Value / EBITDA | Comparing across capital structures |
| P/S (Price-to-Sales) | Price / Revenue per share | High-growth companies with no earnings |
| P/B (Price-to-Book) | Price / Book Value per share | Banks, asset-heavy companies |
| EV/Revenue | Enterprise Value / Revenue | SaaS, subscription businesses |
| Dividend Yield | Annual Dividend / Price | Income-focused investing, utilities |

A *multiple* is just a ratio that makes the current price comparable across companies of different sizes. If Apple earns \$6 per share and trades at 30× earnings (price = \$180), and Microsoft earns \$10 per share and also trades at 30× (price = \$300), the multiples tell you they are valued equivalently by the market — each dollar of earnings costs \$30 to buy.

### What relative value tells you — and does not

Relative value is powerful because it is anchored to reality. It tells you "relative to what the market is currently paying for similar things, this asset looks cheap or expensive." That is genuinely useful information.

What it does *not* tell you is whether the market is right. In March 2000, every internet company looked cheap relative to other internet companies at 200× revenue. In October 2008, every bank looked cheap relative to other banks. Relative value is a measure of comparison, not of absolute worth. When the entire sector is mispriced, comps will not save you.

#### Worked example: The pizza shop, relative approach

Back to the pizza restaurant. You do some research and find that small food-service businesses in Vietnam typically sell at 2.5× to 3.5× annual revenue, with the best ones reaching 4×. Your pizza shop has \$200,000 in annual revenue (note: revenue, not profit — multiples use revenue here, not the \$60,000 profit).

Relative value range: \$200,000 × 2.5 = \$500,000 to \$200,000 × 4 = \$800,000.

Your median estimate is \$200,000 × 3 = \$600,000.

That is in the same neighborhood as the intrinsic value (\$667,000). That consistency is reassuring — it suggests both frameworks are pointing at roughly the same number, which reduces your risk of a large estimation error. When intrinsic and relative value agree, you have a more robust price estimate.

The one-sentence intuition: relative value says "the market has already priced thousands of similar deals — let those transactions do your work."

---

## Framework 3 — Market Value and Liquidation Value: What Someone Will Pay Right Now

The third framework is the most concrete and the most limiting: what would this asset fetch if you sold it today? This is sometimes called *market value* (the current price in an active market) or *liquidation value* (what you would get if you had to sell quickly, potentially below fair value).

### The three flavors

There are actually three related but distinct concepts here:

**Book value** is the accounting value — the original cost of assets minus accumulated depreciation, as reported on the balance sheet. It has no necessary relationship to market prices or cash-generating ability.

**Market value** (for listed assets) is simply the current price. For a stock trading at \$100 with 100 million shares outstanding, the market capitalization — the total market value of the equity — is \$10 billion.

**Liquidation value** is the amount you would receive if you broke the company apart, sold every asset individually, and paid off all the debts. For most going-concern businesses, liquidation value is well below intrinsic value, because the whole is worth more than the sum of its parts (the *franchise value* or *goodwill* disappears in a liquidation).

For asset-heavy businesses — real estate, mining companies, shipping companies — liquidation value is an important floor. You rarely want to pay more than what you could recover by selling off the assets.

### When this framework dominates

The liquidation/market framework matters most in three situations:

1. **Distressed investing.** When a company is insolvent, the question is not "what will it earn?" but "what will the bankruptcy process recover?" Here, asset appraisals, real estate valuations, and machinery liquidation estimates replace DCF models.

2. **Real estate.** The "replacement cost" approach — how much would it cost to rebuild this property from scratch? — is one of three standard real estate valuation methods and is a variant of liquidation value logic.

3. **Deep value investing.** When a company's stock trades *below* book value (P/B < 1), the market is implying you can buy the assets for less than their accounting cost. Sometimes this is a warning sign (the assets are worth less than their book values); sometimes it is a genuine opportunity.

#### Worked example: The pizza shop, liquidation approach

What if the pizza restaurant fails? What would you recover?

- Commercial kitchen equipment: \$40,000
- Furniture and fixtures: \$8,000
- Lease security deposit: \$5,000 (if recoverable)
- Inventory: \$2,000
- Total gross: \$55,000
- Less: outstanding debts: \$15,000
- Liquidation value: **\$40,000**

Compare: intrinsic value \$667,000, relative value \$600,000, liquidation value \$40,000. This shows how extreme the downside can be if the business stops generating cash. The going-concern value (what the business is worth as a running operation) is more than 15× the liquidation value. You are paying for the future, and if the future does not arrive, you lose most of your investment. This is the risk that the discount rate is supposed to compensate you for.

The one-sentence intuition: liquidation value is the floor — the number that tells you the worst-case outcome if everything stops working.

---

## The Valuation Spectrum: From Pure Cash Flow to Pure Sentiment

The three frameworks do not exist in isolation. They form a spectrum from "pure fundamental analysis" on the left to "pure market sentiment" on the right, with each major valuation method occupying a place on that spectrum.

![Valuation spectrum from pure cash flow to pure market sentiment, with DCF through momentum labeled](/imgs/blogs/what-is-value-philosophy-frameworks-asset-pricing-4.png)

At the far left sits the Discounted Cash Flow model — the most rigorous intrinsic-value method. Every dollar of value must be justified by a forecasted cash flow. The model cares nothing about what the market currently thinks; it cares only about what you think the business will earn.

Moving right, the Dividend Discount Model is similar but restricts cash flows to dividends — what actually gets paid to shareholders. Then come *comparable multiples* (EV/EBITDA, P/E), which are anchored in current market prices. Technical analysis is further right — it makes no claims about intrinsic value at all, relying entirely on price patterns and trading volume. Momentum strategies are at the extreme right: buy what is going up because it tends to keep going up.

None of these is universally correct. The right method depends on the asset, the time horizon, the quality of available data, and the market environment. A seasoned analyst uses the spectrum deliberately: intrinsic-value methods for long-duration decisions, relative-value methods for near-term positioning, and technical analysis for timing.

---

## How the Three Frameworks Conflict — and When Each Is Right

In the pizza shop example, intrinsic (\$667K), relative (\$600K), and liquidation (\$40K) all told different stories. For public equities, the conflicts can be just as stark. Here is how to decide which framework to trust.

### When intrinsic value is most reliable

Intrinsic value methods are most reliable when:

- The business has long, stable operating history with predictable cash flows (utilities, consumer staples, toll roads)
- You have confidence in long-run earnings power (even if near-term is lumpy)
- You are taking a long time horizon (5+ years)
- The entire sector is being re-rated by the market (relative value fails when peers are all mispriced)

The weakness is sensitivity to assumptions. A 1 percentage point change in the discount rate can move a DCF value by 20-30%. The terminal value — which usually represents 60-80% of total DCF value — is particularly sensitive. Small changes in long-run growth assumptions produce large value swings.

### When relative value is most reliable

Relative value methods are most reliable when:

- You need to price something quickly and current market conditions are your benchmark
- The company is similar to many peers (no unique characteristics that make it non-comparable)
- You are an investment banker pricing an IPO or M&A deal (you need to know what the market will actually pay, today)
- Transaction speed matters (you cannot run a full DCF in 48 hours)

The weakness is that relative value is market-relative, not absolute. If the whole market is overvalued, comps say everything is fairly priced.

### When market/liquidation value is most reliable

This framework is most reliable when:

- The company is in financial distress or liquidation
- You are analyzing asset-heavy businesses (REITs, mining, shipping, banks)
- You want to establish a floor on downside
- The business is not a going concern

### Using all three together: the triangulation principle

Professional analysts do not pick one framework — they triangulate. If intrinsic value says \$80, relative value says \$85, and book value floor is \$40, you have a consistent picture: the stock is worth roughly \$80-85 with a floor at \$40. If intrinsic says \$150 and relative says \$80, something is wrong with either the comparables or your DCF assumptions, and you need to investigate why before trading.

The goal is not to find *the* right value — it is to build a range of plausible values and understand the story behind each one.

---

## The Efficient Market Debate: Do Prices Already Reflect Value?

The three frameworks assume that prices and values can diverge — that there are opportunities to find assets trading below intrinsic value. This assumption is contested by *Efficient Market Hypothesis* (EMH), one of the most important and debated ideas in finance.

![Efficient market hypothesis levels — weak, semi-strong, and strong — and what information each form assumes is priced in](/imgs/blogs/what-is-value-philosophy-frameworks-asset-pricing-6.png)

### The three forms of efficiency

EMH comes in three increasingly strong versions, first described by Eugene Fama in 1970:

**Weak form:** Prices reflect all *past price information*. Technical analysis cannot consistently earn excess returns because all historical patterns are already priced in. Academic evidence broadly supports weak-form efficiency in major markets.

**Semi-strong form:** Prices reflect all *publicly available information* — earnings reports, news, analyst forecasts, economic data. Fundamental analysis (reading 10-Ks and building DCF models on public data) cannot consistently beat the market if this is true. Most academic evidence supports semi-strong efficiency in large-cap US markets, at least approximately.

**Strong form:** Prices reflect *all* information, including private (insider) information. This version is clearly false — insider trading prosecutions show that private information does move prices before it becomes public.

### What this means for valuation

If markets are even semi-strong efficient, why do intrinsic-value analysts bother? There are several honest answers:

1. **Not all markets are equally efficient.** Small-cap stocks, emerging market equities (including Vietnam), private companies, and illiquid instruments are far less efficiently priced than large-cap US equities. Valuation analysis creates more alpha in less efficient markets.

2. **Efficiency is a tendency, not a guarantee.** Even in efficient markets, prices overshoot and undershoot. The 2022 VN-Index at 11× P/E or the March 2020 S&P 500 crash created temporary mispricings that resolved over subsequent months.

3. **Valuation imposes discipline on your thinking.** Even if you cannot consistently find "cheap" stocks, understanding intrinsic value tells you the cost of being wrong and the range of plausible outcomes. That is risk management, not market-beating.

4. **The relevant question is relative efficiency.** The market as a whole may be roughly efficient, but your specific analysis might be better-informed than the marginal investor for a specific company. Buffett's edge was not that markets were inefficient generally — it was that he understood certain kinds of businesses (consumer franchises, insurance floats) better than most participants.

The practical implication: treat market prices with respect — they aggregate a lot of information — but do not treat them as infallible. The best investors combine the discipline of valuation (what *should* this be worth?) with respect for prices (what is the market *telling me* right now?).

---

## Common Misconceptions

### Misconception 1: A low price means it is cheap

This is perhaps the single most common error in investing. A stock at \$5 is not "cheap" and a stock at \$5,000 is not "expensive." What matters is the price *relative to value*. A \$5 stock of a company with no earnings, no assets, and no prospects may be wildly overpriced. A \$5,000 stock of a company generating \$200 per share in earnings at a stable 6% growth rate has a P/E of 25× — not cheap, but arguably fair.

Buffett addressed this directly in his 2004 shareholder letter: "Price is what you pay; value is what you get." The entire framework of this post rests on that separation.

### Misconception 2: Paying a high P/E multiple always means overpaying

A high P/E is not automatically a sign of overvaluation — it may simply reflect the market's expectation of high future growth. In 2013, Amazon traded at over 1,000× trailing earnings. Applying a "market average" P/E of 15-18× would have told you to sell. In fact, Amazon's intrinsic value at that time was dramatically higher than its already-high price, because the market's P/E framework was comparing a capital-intensive growth machine to mature retailers.

The lesson: a P/E ratio is only meaningful in context. You need to know the growth rate, the margin trajectory, and the capital efficiency of the business to judge whether a given multiple is too high or too low.

### Misconception 3: Intrinsic value is a precise number

It is not. It is a range with substantial uncertainty at both ends. A professional DCF for a mature business might produce a range of \$80-\$130 depending on assumptions. An honest analyst does not say "this stock is worth exactly \$110.43." They say "the range of intrinsic values under our range of assumptions is roughly \$90-\$130, with the central case around \$110. The current price of \$85 represents a meaningful discount to the low end of that range."

Treating valuation as a precise calculation is a classic mistake that produces false confidence. The value of a DCF is not the number it spits out; it is the discipline of forcing you to articulate your assumptions and test their sensitivity.

### Misconception 4: The market price is always wrong

The mirror-image error: if "price = value" is too trusting, "price is always wrong" is too arrogant. Most of the time, for most liquid assets, the market price is a very good approximation of value. The edge cases — when it diverges meaningfully — are rare and often require special conditions (a market panic, an orphaned security with no analyst coverage, a company going through a misunderstood transition).

The correct posture: prices are a prior. You start with the price as a reasonable estimate of value, and then you ask: "what would have to be true for this price to be wrong, and do I have evidence that those things are true?"

### Misconception 5: All three frameworks should agree

They often do not, and that is expected. The frameworks answer different questions. Intrinsic value answers "what is the present value of future cash flows?" Relative value answers "what are comparable assets selling for today?" Liquidation value answers "what would you get in a forced sale?" A going concern with high future growth will have intrinsic > relative > liquidation, sometimes by an order of magnitude. That is not a contradiction; it is a description of three different aspects of the same asset.

---

## How It Shows Up in Real Markets

### The VN-Index at 11x P/E (2022 trough)

In late 2022, the VN-Index — Vietnam's main stock market index — hit a trailing P/E of approximately 11.2×. At the same time, the S&P 500 was trading at roughly 18× (down from its peak of 38× in 2020, itself a COVID-era distortion). The long-run average S&P 500 P/E is around 16-17×.

![VN-Index P/E vs S&P 500 P/E 2015 to 2024 dual-line comparison](/imgs/blogs/what-is-value-philosophy-frameworks-asset-pricing-8.png)

From a relative-value perspective, you were buying Vietnamese corporate earnings at roughly half the price of US corporate earnings. That is a striking discount. A relative-value analyst would ask: "Is the discount justified by lower growth, higher risk, or worse corporate governance? Or is it an opportunity?"

The intrinsic-value framing adds nuance. If Vietnamese companies were growing earnings at 15-20% annually (which many were in the 2015-2022 period), a P/E of 11× implies a much higher earnings yield than 11× alone suggests. An earnings yield of 1/11 = 9.1% on a growing business is fundamentally cheap compared to a 10-year government bond at 4-5%.

The market-value lens adds the floor: in a financial crisis, what would these companies fetch? If the answer is "a lot less than 11×," the floor is not protecting you much.

The resolution: the 2022 VN-Index trough turned out to be a genuine buying opportunity. By 2023-2024, the index recovered significantly, producing roughly 30-35% returns from the trough in local currency. The relative-value signal was real; the timing was uncertain.

### Buffett's Coca-Cola purchase (1988)

In 1988 and 1989, Warren Buffett's Berkshire Hathaway purchased approximately \$1.3 billion worth of Coca-Cola stock. At the time, Coca-Cola was trading at roughly 15× earnings — not obviously cheap by relative value standards (the market was around 12-14×).

Buffett's thesis was intrinsic-value based. He estimated that Coca-Cola's moat — its brand, distribution network, and global growth potential — would allow it to compound earnings at roughly 15% per year for the foreseeable future. At 15× earnings with 15% growth, the intrinsic value was dramatically higher than the price, because growth multiples dramatically understate the value of a compounding franchise.

By 1997, Berkshire's \$1.3 billion investment was worth roughly \$13 billion — a 10× return in nine years. The insight was not that Coca-Cola was cheap by comps. It was that the intrinsic value, properly estimated, was far above what any multiple-based approach would show.

### The S&P 500 P/E expansion 2010-2020

From 2010 to 2020, S&P 500 corporate earnings roughly doubled. But prices roughly quadrupled. How is that possible? P/E expansion: the market was willing to pay more for each dollar of earnings in 2020 than it was in 2010.

![S&P 500 trailing P/E ratio 2010 to 2024 showing expansion and contraction](/imgs/blogs/what-is-value-philosophy-frameworks-asset-pricing-3.png)

In 2010, the trailing P/E was 16.3×. By 2020, it peaked at 38.3× — partly a COVID distortion (earnings collapsed while the market anticipated recovery), but even stripping out 2020, the P/E had reached 24-28× by 2019. This was driven by:

1. Falling interest rates (lower discount rates → higher intrinsic values)
2. Increasing concentration in mega-cap technology stocks with very high growth expectations
3. Passive investing flows that bought regardless of valuation

The lesson: earnings growth explains roughly half of equity returns over time. The other half comes from multiple expansion (or is destroyed by multiple contraction). Understanding valuation frameworks helps you separate the two.

### A house: three valuations, three numbers

Consider a house in Da Nang, Vietnam. The owner wants to sell. You commission three estimates:

**Income approach (intrinsic value):** The house rents for \$18,000 per year. After expenses, net operating income is \$14,000. Comparable capitalization rates (*cap rates* — the inverse of a P/E ratio for real estate) are 5% in this market. Intrinsic value = \$14,000 / 0.05 = **\$280,000**.

**Comparable sales (relative value):** Five similar houses sold in the past six months at an average of \$240,000-\$310,000, with a median around \$270,000. Relative value = **\$270,000**.

**Replacement cost (liquidation/cost approach):** Building a comparable new house would cost \$150,000 in materials and labor. The land is worth \$100,000. Replacement cost = **\$250,000**.

Three methods, three numbers: \$250,000, \$270,000, \$280,000. Here they are close — which is reassuring. If the owner is asking \$260,000, all three frameworks suggest it is fairly priced to modestly undervalued. If they are asking \$400,000, the gap demands explanation: is there something special about this house that the comps do not capture? A view, a renovation, a unique location?

When the three methods agree within 10-15%, the price is probably fair. When they diverge by 50%+, you need to understand why — that divergence often contains the most important information of all.

---

## The Risk-Return Reality: What the Numbers Show

Before closing, let us look at the empirical record on risk-return across asset classes — because ultimately, every valuation framework is a tool for asking "am I being adequately compensated for the risk I am taking?"

![Risk-return scatter of major asset classes 2000 to 2024, average return vs standard deviation](/imgs/blogs/what-is-value-philosophy-frameworks-asset-pricing-5.png)

Over the 2000-2024 period, the risk-return relationship across major asset classes has been broadly positive but far from smooth:

- **US Stocks (S&P 500):** Average annual return ~10.7%, standard deviation ~15.2%. The highest return in this set, with moderate volatility.
- **REITs:** ~9.1% return, ~19.3% volatility. High return but more volatile than stocks.
- **Gold:** ~8.9% return, ~15.8% volatility. Comparable return to stocks but with a very different risk profile — gold performs well in crises when stocks fall.
- **Emerging Markets:** ~7.1% return, ~22.4% volatility. More volatility, less return than US stocks over this period — the "emerging market premium" was largely absent.
- **International Stocks:** ~5.4% return, ~17.1% volatility. Significant underperformance relative to US equities.
- **US Bonds:** ~3.8% return, ~5.9% volatility. Low return, low risk — the anchor asset.
- **Cash:** ~1.8% return, ~0.6% volatility. Almost no risk, almost no real return.

What does this tell us about valuation? The equity risk premium — the extra return of stocks over cash, roughly 8.9 percentage points over this period — is the market's collective answer to the question "how much must I be paid to accept the uncertainty of corporate earnings?" When you build a DCF model and choose a discount rate, you are implicitly making a claim about this premium. If the market's implied risk premium is currently very low (stocks are priced as if they will definitely earn high returns), either the market is right and this time is different, or stocks are overpriced.

This is why intrinsic-value analysis and relative-value analysis must be supplemented with a view on the macro environment — interest rates, growth expectations, and risk appetite. The connection runs directly through the discount rate. For more on this relationship, see [Interest rates, bonds, and stocks](/blog/trading/macro-trading/interest-rates-bonds-stocks-relationship).

---

## A Margin of Safety in Practice

Let us bring the concept of margin of safety back to close the circle.

![Margin of safety vertical comparison — market price below intrinsic value, with the safety buffer labeled](/imgs/blogs/what-is-value-philosophy-frameworks-asset-pricing-7.png)

The margin of safety framework says: estimate intrinsic value, then only buy when the price is significantly below that estimate. The "significant" discount is not arbitrary — it is calibrated to your estimation uncertainty.

If you are very confident in your DCF (a stable utility with 20 years of history), a 10-15% discount might be sufficient. If you are less confident (a high-growth technology company where the terminal value dominates), you might want a 30-40% discount before you feel the position is adequately protected.

Here is a practical decision framework:

| Estimation confidence | Price/Value ratio where you buy | Margin of safety |
|---|---|---|
| Very high (stable utility) | 0.85-0.90× | 10-15% |
| High (established franchise) | 0.75-0.80× | 20-25% |
| Medium (cyclical business) | 0.65-0.70× | 30-35% |
| Low (high-growth, high-uncertainty) | 0.55-0.60× | 40-45% |

This is not a formula — it is a judgment framework. The key insight is that the margin of safety is not a fixed percentage. It is a function of how confident you are in your estimate of value. The more uncertain the estimate, the wider the required cushion.

## The Mathematics of Value: Going Deeper on Each Framework

Now that we have built intuition across all three frameworks and seen them in action across real markets, it is worth going one level deeper into the math. The goal is not to derive formulas for their own sake — it is to understand *why* the formulas are built the way they are, what each variable controls, and what happens when you change the assumptions.

### Intrinsic value: the sensitivity problem

The DCF formula has a well-known problem: small changes in the discount rate or terminal growth rate produce enormous changes in value. Let us make this concrete.

Suppose a company generates \$100 in free cash flow today, growing at 5% per year for 5 years, and then in perpetuity at 3% thereafter. We compute value at different combinations of discount rate and terminal growth rate (the numbers below come directly from the `DCF_SENSITIVITY` table in the series data, using a baseline FCF of 100 per year over 5 years):

| WACC \ Terminal growth | 2% | 3% | 4% |
|---|---|---|---|
| 8% | \$1,312 | \$1,487 | \$1,731 |
| 10% | \$1,082 | \$1,197 | \$1,348 |
| 12% | \$912 | \$996 | \$1,103 |

Two observations stand out immediately. First, a move from 8% to 12% WACC — a 4-percentage-point change in discount rate — cuts the value by roughly 30% (from ~\$1,500 to ~\$1,000 at 3% terminal growth). This is why rising interest rates are so destructive to equity valuations: they raise the discount rate, and higher discount rates reduce present value even if earnings are unchanged.

Second, a move from 2% to 4% terminal growth rate at a fixed 8% WACC adds nearly 32% to value (\$1,312 → \$1,731). Terminal growth assumptions are the single most impactful input in a long-duration DCF model, because the terminal value typically represents 60-80% of total intrinsic value.

The practical implication: always present a DCF as a range, not a point estimate. Present the base case (your central scenario), a bull case (+1% growth, -1% WACC), and a bear case (-1% growth, +1% WACC). The spread of that range tells you how much certainty you actually have.

#### Worked example: Sensitivity for a Vietnamese bank

Vietcombank (VCB) earned approximately 22,500 billion VND in net profit in 2023, and the consensus expects roughly 8-10% earnings growth over the next five years. Investors typically require a 12-14% return on Vietnamese bank equity (reflecting currency risk, regulatory risk, and Vietnam's emerging-market status).

At the midpoints — 9% growth, 13% required return — a simplified perpetuity-with-growth gives:

$$V = \frac{22{,}500 \times 1.09}{0.13 - 0.09} = \frac{24{,}525}{0.04} = 613{,}125 \text{ billion VND}$$

At the time of this writing, VCB's market cap was roughly 400,000-450,000 billion VND. That implies a discount to intrinsic value under these assumptions — but notice how sensitive the answer is. If the required return is 14% instead of 13%:

$$V = \frac{24{,}525}{0.14 - 0.09} = \frac{24{,}525}{0.05} = 490{,}500 \text{ billion VND}$$

A single percentage point in required return moved the value down by roughly 20%. And if you use a 12% required return instead:

$$V = \frac{24{,}525}{0.12 - 0.09} = \frac{24{,}525}{0.03} = 817{,}500 \text{ billion VND}$$

The full range — \$490,500 to \$817,500 billion VND — is enormous. The intrinsic value of this bank, under reasonable assumption ranges, is somewhere between "at market" and "50% above market." That is not a flaw in the method; it is an honest acknowledgment of the estimation uncertainty in fast-growing emerging market businesses.

The one-sentence intuition: in high-growth, high-uncertainty markets like Vietnam, intrinsic-value ranges are wide — and that width is itself valuable information about the risk of the investment.

### Relative value: the comparables selection problem

The key challenge in comps analysis is not computation — any spreadsheet can divide price by earnings. The challenge is selecting truly comparable companies. Bad comparables produce bad valuations.

The classic comparability checklist:

1. **Industry and business model.** A software company should not be compared to a manufacturing company even if both earn \$5 per share. Their growth rates, capital requirements, and margin profiles are completely different.

2. **Size.** A \$100 billion market cap company typically commands a premium multiple over a \$1 billion company in the same industry — larger companies have more predictable earnings, better access to capital, and lower failure risk. Comparing them directly misstates relative value.

3. **Growth rate.** The PEG ratio (*P/E-to-growth* ratio) corrects for this: a P/E of 30× on a company growing at 30% per year is PEG = 1.0, while a P/E of 30× on a company growing at 5% per year is PEG = 6.0. The same multiple means something very different for companies growing at different rates.

4. **Geography and currency.** A Vietnamese company's P/E cannot be directly compared to a US company's P/E without adjusting for Vietnam's higher required returns (due to currency risk and market risk premium), different accounting standards, and different tax regimes.

5. **Capital structure.** Two companies with the same operating earnings but different debt levels will have different P/E ratios because interest expense is deducted before earnings. Use EV/EBITDA — which is enterprise-value-neutral to capital structure — when comparing companies with different leverage.

#### Worked example: The PEG ratio in practice

In mid-2024, VPBank (a leading Vietnamese private bank) was trading at approximately 10× forward earnings. Two large US regional banks were trading at 11-12×. A naive comparison says they are similarly valued.

But VPBank was growing earnings at roughly 15-20% per year, while the US regional banks were growing at 3-5%. Adjusting:

- VPBank PEG = 10 / 17.5 (midpoint) = **0.57×**
- US regional bank PEG = 11.5 / 4 (midpoint) = **2.9×**

On a growth-adjusted basis, VPBank was trading at about one-fifth the price of the US regional banks. Whether that discount is justified by risk (currency, regulatory, credit risk in Vietnam's property-linked loan book) or represents a genuine opportunity is a question the PEG ratio cannot answer on its own — but it makes the comparison honest in a way that raw P/E does not.

The one-sentence intuition: a multiple only tells you what the market is paying; dividing by growth tells you what the market is paying *per unit of future value*.

### Liquidation value: the going-concern premium

One of the most important concepts in valuation is the *going-concern premium* — the difference between a company's intrinsic value as a running business and its liquidation value if the business were shut down and assets sold off.

For most profitable businesses, this gap is enormous. A McDonald's franchise restaurant might have liquidation value of \$200,000 (kitchen equipment, furniture, lease deposit) but intrinsic value of \$1.5 million as a running operation. The \$1.3 million difference is the going-concern premium — the value of the customer relationships, the brand association, the trained staff, the operational know-how, and the future earnings stream.

Going-concern premium is high when:
- The business has strong brand or customer loyalty
- Earnings are high relative to asset base (high return on invested capital)
- Growth prospects are strong
- The business would be difficult for a competitor to replicate

Going-concern premium is low (or negative) when:
- The business is losing money
- Assets could be sold for more as a liquidation than the business earns
- The business operates in a structurally declining industry

A P/B ratio (price-to-book) above 1 means the market is pricing in a going-concern premium. A P/B below 1 means the market believes the assets are worth more dead than alive — a classic *deep value* signal, though often also a warning that something fundamental is wrong with the business.

---

## Applying the Frameworks: A Decision Tree

How does a professional analyst actually choose which framework to use? Here is a practical decision tree:

**Step 1: Can you forecast cash flows?**
- If yes (the business has 5+ years of history, stable margins, predictable revenue), proceed to DCF.
- If no (startup, highly cyclical, loss-making), skip to Step 2.

**Step 2: Are there good comparables?**
- If yes (public companies of similar size, business model, geography exist), use comps.
- If no (highly unique business, no direct peers), you must go back to first principles or use asset-based approaches.

**Step 3: What is the asset structure?**
- If the company is asset-heavy (real estate, mining, shipping, bank), liquidation value sets an important floor. Start there.
- If the company is a pure service or software business with few tangible assets, liquidation value is close to zero — ignore this framework.

**Step 4: What is the context?**
- Long-term investment decision (buy and hold 5+ years): weight intrinsic value more heavily.
- Near-term investment, IPO, or M&A transaction (where you will exit in 1-3 years): weight relative value more heavily.
- Distressed situation (company may not survive): weight liquidation value most heavily.

In practice, most professionals:
1. Run a DCF for the central intrinsic-value estimate
2. Check against comps to make sure the DCF is not wildly out of line with the market
3. Compute liquidation value as the floor
4. Present all three in a "football field" chart that shows the range of values under different assumptions

The football field is one of the most useful diagrams in investment banking — it shows, on a single chart, where the current price sits relative to all three frameworks' estimated value ranges.

---

## The Role of Time Horizon in Choosing a Framework

One underappreciated dimension is that the right framework often depends on how long you plan to hold the asset. This is not just a practical consideration — it reflects something deep about what "value" means over different time scales.

**Over a 1-week horizon,** the dominant driver of price is market sentiment, technical factors, and short-term news flow. Intrinsic value is irrelevant — even if you are correct that a stock is worth \$150, the market can keep it at \$80 for weeks or months. Technical analysis and flow-based approaches dominate at this timescale.

**Over a 1-year horizon,** earnings releases, guidance updates, and sector re-ratings start to matter. Relative value (P/E vs. peers, P/E vs. history) becomes meaningful. A stock trading at a big discount to peers often re-rates toward the mean over 6-18 months — the "mean reversion" that value investors exploit.

**Over a 5-year+ horizon,** the compounding of earnings becomes the dominant factor. A business that grows earnings at 15% per year doubles its earnings roughly every five years. That compounding eventually overwhelms any starting valuation difference. Intrinsic value (DCF, dividends, long-run earnings power) is the most relevant framework here.

John Maynard Keynes captured this time-horizon dimension in his famous observation: "In the long run, we are all dead." He was pointing out that markets can remain irrational longer than investors can remain solvent — meaning that even a correct intrinsic-value estimate is not enough if you cannot survive the time it takes for the market to agree with you.

The practical implication: match your framework to your time horizon. If you are managing a portfolio with quarterly performance reviews, relative value is probably more actionable than a 10-year DCF. If you are managing an endowment or a long-duration pension fund, intrinsic value is the right anchor.

#### Worked example: The same stock, three time horizons

Consider Hoa Phat Group (HPG), Vietnam's largest steel producer. Suppose we are analyzing it in mid-2023 when the stock was trading at roughly 24,000 VND per share.

**1-week view (technical/sentiment):** HPG had been trading in a range of 22,000-26,000 VND for six weeks, with support at 22,000. A short-term trader might have used this range-trading signal to buy near 22,000 and sell near 26,000 — with no reference to intrinsic value whatsoever.

**1-year view (relative value):** HPG was trading at roughly 8-9× forward earnings, compared to global steel peers at 6-8× (reflecting Vietnam's higher growth premium) and regional steel companies at 7-9×. Comps suggested HPG was roughly fairly valued on a 12-month view, with some upside if steel margins recovered.

**5-year view (intrinsic value):** HPG had announced major capacity expansion plans (Dung Quat Phase 2, targeting ~14 million tons of steel capacity by 2025-2026). If these expansions came in on budget and at full utilization, the company's earnings power in 2027-2028 would be roughly 2.5-3× the 2023 base. A DCF on that earnings trajectory, using a 13% required return and 3% terminal growth, suggested intrinsic value of 35,000-40,000 VND per share — a 45-65% premium to the 2023 price.

The three analyses gave three different "answers" — but they were answering three different questions. The short-term trader was right to ignore the DCF. The long-term investor was right to base the position on intrinsic value. Neither was wrong; they were operating on different time horizons with different frameworks.

The one-sentence intuition: time horizon determines which kind of value is actually realized — technical signals decay in days, multiples converge in months, cash flows compound over years.

---

## Value in the Context of Macro and Interest Rates

No discussion of valuation frameworks would be complete without connecting them to the macro environment, because the key variable in every intrinsic-value model — the discount rate — is directly tied to interest rates set by central banks.

The relationship works like this. The discount rate in a DCF model is typically built from the *risk-free rate* (usually the 10-year government bond yield) plus an *equity risk premium* (the extra return required for taking equity risk). If the Fed raises the risk-free rate from 1% to 5%, every DCF model using a 1% risk-free rate base becomes instantly wrong — the discount rate rises, and intrinsic values fall, even if the company's earnings have not changed at all.

This is exactly what happened in 2022. The US Federal Reserve raised the federal funds rate from near-zero in January 2022 to over 4% by December 2022. The S&P 500 P/E compressed from roughly 28× at the start of the year to roughly 18× by year-end. Corporate earnings actually grew in 2022 — yet the stock market fell 18%. The entire loss came from multiple compression driven by rising discount rates, not from deteriorating business fundamentals.

The VN-Index experienced an even more severe version of the same dynamic. As global risk appetite collapsed and foreign investors withdrew capital from emerging markets, required returns on Vietnamese equities rose sharply. The VN-Index P/E fell from roughly 17-18× in early 2022 to 11.2× by year-end — a 35% compression in the multiple, driving the index down roughly 33% despite Vietnam's GDP continuing to grow at 8%.

For more on this mechanism — how central bank policy flows through to asset prices via the discount rate — see [How the Fed Sets Interest Rates](/blog/trading/finance/how-the-fed-sets-interest-rates) and the broader treatment of macro-asset interactions in [Interest Rates, Bonds, and Stocks](/blog/trading/macro-trading/interest-rates-bonds-stocks-relationship).

The key takeaway: valuation frameworks do not operate in a macro vacuum. The discount rate is the bridge between the micro (a specific company's cash flows) and the macro (global interest rates, risk appetite, inflation expectations). An intrinsic-value analyst who ignores macro is like an architect who ignores soil conditions — the structure might be perfectly designed, but it can still sink.

---

## Building the Habit: Valuation as a Mental Practice

The three frameworks are not just tools for professional analysts. They are a mental habit — a way of approaching prices that becomes automatic with practice. Here is how to build it.

**Every time you see a price, ask: "Relative to what?"** A stock at \$50 is not cheap or expensive — it is a number. What makes it cheap or expensive is the value that justifies it. Start with: "What would this be worth if it were a private business? What are similar things selling for? What would you get if it were liquidated?"

**Practice on everyday transactions.** When you buy a coffee for \$5, you are implicitly doing a relative-value analysis: is this expensive compared to alternatives? When you decide whether to rent or buy a house, you are doing an intrinsic-value vs. market-value analysis. These everyday judgments use the same frameworks as professional valuation — just with shorter time horizons and less formal math.

**Read earnings releases with a framework in mind.** When a company reports earnings, the market's first instinct is to focus on whether earnings beat or missed estimates. But the valuation-framework reader asks: "What does this change about intrinsic value? Is the multiple still appropriate given what I now know about growth and risk? Are the comps still relevant?"

**Track your estimates against outcomes.** The most powerful way to improve at valuation is to write down your estimated intrinsic value *before* you see what happens, then compare your estimate to the actual outcome 12-24 months later. This calibration exercise is what separates analysts who improve from those who stay at the same level indefinitely.

The goal is not to be right every time — valuation estimates are, by definition, uncertain. The goal is to be *calibrated*: to have a sense of when your estimates are likely to be close and when they are likely to be wide, and to size your conviction accordingly.

---

## Further Reading and Cross-Links

The three frameworks introduced here are each the subject of entire sub-disciplines of finance. As you proceed through this series, each will be developed into its full methodological depth:

**On intrinsic value and DCF:** The complete mechanics of building a multi-period DCF, choosing terminal growth rates, and stress-testing assumptions are covered in the equity-research series. Start with [Discounted Cash Flow: A Complete Guide](/blog/trading/equity-research/discounted-cash-flow-dcf-complete-guide). The discount rate — the hinge of the entire intrinsic-value framework — is covered in the WACC article in the same series.

**On discount rates and interest rates:** The intrinsic-value framework is deeply connected to the macro environment because the discount rate moves with interest rates. The relationship between interest rates, bonds, and stock valuations is explored in [Interest Rates, Bonds, and Stocks](/blog/trading/macro-trading/interest-rates-bonds-stocks-relationship). For a deep dive into how central bank policy sets the floor for discount rates, see [How the Fed Sets Interest Rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

**On relative value and market multiples:** The next posts in this series build the full toolkit of equity multiples — P/E, EV/EBITDA, P/S, P/B — from first principles, with sector-by-sector application and worked examples from both US and Vietnamese markets.

**On technical analysis (market-value end of the spectrum):** For context on what the pure price-based methods do and do not tell you, see [What Is Technical Analysis](/blog/trading/technical-analysis/what-is-technical-analysis). Understanding where technical analysis sits on the valuation spectrum clarifies both its strengths and its limitations.

The rest of this series uses the frameworks built here as the foundation for every method. Every DCF model we build, every multiple we compute, every option we price answers, implicitly or explicitly, the question we started with: what does it mean for this price to be right?

*This post is educational and describes general valuation concepts. It is not financial advice and should not be relied upon for investment decisions in any specific security.*
