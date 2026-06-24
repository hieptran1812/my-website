---
title: "Risk and Required Return: CAPM, Beta, and the Cost of Capital"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Why riskier assets must offer higher returns. Build CAPM from first principles — beta, equity risk premium, country risk — and see how it sets the discount rate in every valuation model."
tags: ["capm", "beta", "risk-premium", "required-return", "cost-of-equity", "systematic-risk", "equity-risk-premium", "valuation"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Every asset's price is just the present value of its future cash flows, and the discount rate you use to compute that present value is your *required return* — the minimum return that compensates you for the risk you're bearing.
>
> - Risk means the distribution of outcomes is wide; investors demand a premium to accept that width
> - Only *systematic* risk (market-wide) earns a return premium — company-specific risk can be diversified away for free
> - CAPM quantifies that premium: E(R) = Rf + β × ERP, where β measures how much a stock amplifies market swings
> - The equity risk premium (ERP) is approximately 4.6% above the risk-free rate in the US as of early 2025
> - Investing in Vietnam adds a country risk premium of roughly 2%, which is why Vietnamese stocks historically trade at lower P/E multiples than US stocks

---

Open two investment options on your screen. The first is a 10-year US Treasury bond yielding 4.5% per year. Your money is backed by the full faith and credit of the United States government — the most creditworthy borrower in the world. The second is a bond issued by a Vietnamese mid-cap steel company, yielding 9%. Same duration. Same basic structure. A coupon every six months, your principal back at the end.

Why the difference? The Vietnamese company is not the US government. It could face raw material shortages, currency fluctuations, regulatory changes, or a construction cycle downturn that threatens its ability to pay. The 4.5 percentage point spread is not charity — it is the market's price tag for that extra risk. Every single day, across every single asset, markets are solving a version of this equation: *what return do I need to accept this level of risk?*

This post answers that question from first principles. We will build up from the intuition of what risk actually means, to the mechanics of diversification, to beta, to the Capital Asset Pricing Model, to the equity risk premium, and finally to how you adjust for country risk when you're investing in emerging markets like Vietnam. By the end, you will know exactly how to compute the required return for any stock — and why that number sits at the heart of every serious valuation.

![Security Market Line showing required return vs beta](/imgs/blogs/risk-required-return-capm-beta-cost-capital-1.png)

The diagram above is the mental model for this entire post. That upward-sloping line is the *Security Market Line* — it shows that higher systematic risk (measured by beta on the x-axis) demands higher expected return (on the y-axis). The y-intercept is the risk-free rate. The slope is the equity risk premium. Every stock sits on or near this line; stocks below it are overpriced relative to their risk, and stocks above it offer more return than their risk warrants.

---

## Foundations: What Risk Actually Means

Before any formula, we need to get precise about what "risk" means in finance. In everyday language, risk means something bad might happen. In finance, risk means the *range of possible outcomes is wide*. Both the bad and the good matter.

When you invest \$1,000 in a US Treasury for one year, you know with near-certainty that you'll have \$1,045 at the end. The distribution of outcomes is a spike at one number. When you invest \$1,000 in a high-growth tech stock, you might end up with \$600 or \$2,000 or anything in between. The distribution is wide.

Statisticians measure that width with *standard deviation* — the average distance between an observed outcome and the mean outcome. A stock with a 30% annual standard deviation means its annual returns typically fall within 30 percentage points above or below its average. If the average is 10%, you might get –20% in a bad year and +40% in a good year. A stock with a 5% standard deviation is far tamer; most years will be close to average.

*Variance* is simply the square of standard deviation, and it shows up in portfolio math. *Downside deviation* focuses only on the negative side — some investors care more about avoiding losses than about the full width of the distribution. For most CAPM-based valuation work, standard deviation and beta are the key measures.

Here's the foundational claim of modern finance: investors are *risk-averse*. Given two investments with the same expected return, they will always prefer the one with lower standard deviation. This is not a cynical assumption — it reflects the math of diminishing marginal utility. Losing \$10,000 hurts more than gaining \$10,000 helps. So investors require *compensation* to take on risk. That compensation is a higher expected return.

### Three Faces of Risk

Finance uses several different risk measures depending on what question you're asking.

**Standard deviation** captures the total width of the return distribution — both the upside and the downside. It's the most widely used measure in academic finance. But it treats a +40% surprise exactly the same as a −40% surprise, which feels wrong to most investors.

**Downside deviation** (also called *semi-deviation*) looks only at returns below a threshold — often zero or a minimum acceptable return. It treats losses differently from gains, which better reflects how most people actually experience risk. The *Sortino Ratio* uses downside deviation instead of standard deviation to measure risk-adjusted returns.

**Value at Risk (VaR)** answers a specific question: "What is the maximum loss I should expect 95% (or 99%) of the time?" A one-day VaR of \$1 million at 95% confidence means that on a typical bad day, losses should not exceed \$1 million — but 5% of days could be worse. VaR is widely used in banking regulation and risk management.

**Expected Shortfall (CVaR)** goes further: "When things do go beyond the VaR threshold, how bad does it get on average?" CVaR is a better measure of tail risk than VaR and is increasingly used by regulators.

For the purposes of CAPM and cost of equity estimation, we primarily use **standard deviation** and **beta** (a measure of co-movement with the market). But understanding the full toolkit helps you interpret risk disclosures and know when CAPM's assumptions matter.

### Diversification: Getting Risk for Free

Suppose you own one stock — say, a Vietnamese consumer goods company. Its returns have a standard deviation of 30% per year. Your entire portfolio swings with that company's fortunes: a product recall sends you down 25%, a blockbuster quarter sends you up 40%.

Now suppose you own 30 stocks across different sectors and geographies. The bad quarter at the consumer goods company is offset by a good quarter at the steel maker. The product recall at one firm barely moves your portfolio because it represents only 3% of your holdings. Your portfolio's standard deviation might fall to 15% — even though the average stock in the portfolio still has 30% individual volatility.

That reduction is *free*. You did not give up expected return. You just canceled noise. This is the power of diversification, and it is one of the few genuine free lunches in finance.

The mathematics work because stock returns are not perfectly correlated. When one company is having a terrible quarter, another is having its best quarter. When the airline industry suffers from a fuel spike, the energy sector benefits. When a technology company's product flops, its competitor gains market share. These offsetting moves partially cancel at the portfolio level, reducing total swing without reducing average return.

#### Worked Example:

Imagine you hold a single stock with a standard deviation of 30%. Your neighbor holds a portfolio of 20 stocks, each with 30% standard deviation, but their returns are not perfectly correlated — when one zigs, another zags. In the idealized case of zero correlation, the portfolio standard deviation is approximately:

$$\sigma_p \approx \sigma_{stock} \times \frac{1}{\sqrt{N}}$$

For N = 20: \$\sigma_p \approx 30\% \times \frac{1}{\sqrt{20}} \approx 30\% \times 0.224 \approx 6.7\%\$

That estimate assumes zero correlation between stocks, which is unrealistic. In practice, stocks are correlated with each other through macroeconomic factors — recessions hurt most businesses, rate hikes compress most valuations, global crises ripple through every sector. So the real portfolio standard deviation bottoms out not at zero but at roughly 15–16% for a broad US equity portfolio — the standard deviation of the market itself. You cannot diversify away *all* risk. You can only diversify away the company-specific piece.

This bottoming out happens because all stocks share *one common factor*: their sensitivity to the macroeconomic environment. That shared factor is what CAPM calls systematic risk. No matter how many stocks you add, you cannot escape the risk that the entire economy falls into recession, that the central bank raises rates dramatically, or that a pandemic shuts down global activity. Those events hit every stock, in every sector, in every geography (to varying degrees).

The intuition here is that the residual — the risk you cannot escape no matter how many stocks you own — is the risk that earns a return premium. If you could eliminate it by diversifying, the market would never pay you to bear it. The market only prices what it cannot eliminate.

### The Portfolio Variance Formula

For completeness, the exact formula for portfolio variance with N stocks and pairwise correlations is:

$$\sigma_p^2 = \sum_{i=1}^{N} w_i^2 \sigma_i^2 + \sum_{i=1}^{N} \sum_{j \neq i} w_i w_j \rho_{ij} \sigma_i \sigma_j$$

where \$w_i\$ is the portfolio weight in stock i, \$\sigma_i\$ is its standard deviation, and \$\rho_{ij}\$ is the correlation coefficient between stocks i and j.

As N grows large with equal weights (\$w_i = 1/N\$), the first term (individual variances) shrinks toward zero because \$w_i^2 = 1/N^2\$ and there are N such terms. The second term (cross-product correlations) converges to the *average covariance* between any two stocks. That average covariance is exactly the systematic variance — the variance explained by the common market factor. No amount of diversification can eliminate it.

Practically speaking, most of the diversification benefit is captured with 20–30 stocks across different sectors. Going from 1 to 10 stocks cuts total risk dramatically; going from 50 to 500 stocks adds minimal additional diversification benefit.

![Systematic vs unsystematic risk comparison](/imgs/blogs/risk-required-return-capm-beta-cost-capital-2.png)

---

## Systematic vs Unsystematic Risk: The Key Distinction

The most important concept in this entire post is the distinction between two fundamentally different kinds of risk.

*Unsystematic risk* (also called *idiosyncratic risk* or *company-specific risk*) is the part of a stock's volatility that is unique to that company. A pharmaceutical company fails its Phase III drug trial — that's unsystematic risk. A retailer's CEO resigns unexpectedly — unsystematic. A steel manufacturer gets hit by a dock workers' strike — unsystematic. These events are bad for that company, but they don't systematically affect all companies at the same time.

*Systematic risk* (also called *market risk*) is the part of a stock's volatility that comes from broad economic forces affecting all companies simultaneously. The Federal Reserve raises interest rates 300 basis points in a single year — that hurts nearly every stock because it raises discount rates. A global pandemic shuts down economic activity — that hits every business simultaneously. A sovereign debt crisis spreads panic through financial markets — all assets fall together.

The crucial insight: because unsystematic risk can be diversified away, the market does not pay you to bear it. A rational investor holds a diversified portfolio and has already eliminated idiosyncratic risk at zero cost. The market only compensates you for the risk you *cannot* avoid — systematic risk.

This is why CAPM ignores total standard deviation and focuses on *beta* — a measure of how much of a stock's volatility is systematic.

![Asset class risk-return scatter](/imgs/blogs/risk-required-return-capm-beta-cost-capital-3.png)

The scatter plot above shows the risk-return trade-off across major asset classes from 2000 to 2024. Cash at the bottom-left barely moves and barely pays. Emerging market stocks in the upper-right are the most volatile but have delivered the highest returns over long horizons. The rough upward slope confirms the core claim: more risk (standard deviation), more return — but the relationship is imperfect, and the type of risk matters enormously.

---

## Beta: Measuring Systematic Risk

*Beta* (β) quantifies how much a stock's returns move in lockstep with the market. Technically it is the slope of the regression line when you plot the stock's excess returns (above the risk-free rate) against the market's excess returns:

$$\beta = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}$$

where \$R_i\$ is the stock's return, \$R_m\$ is the market's return, Cov is covariance, and Var is variance. In plain English: beta tells you how many percentage points the stock moves, on average, for every one percentage point move in the market.

- **β = 1.0**: The stock moves in line with the market. If the S&P 500 rises 10%, this stock tends to rise 10%. If the market falls 15%, this stock tends to fall 15%.
- **β > 1.0**: The stock *amplifies* market moves. A stock with β = 2.0 tends to rise 20% when the market rises 10%, and fall 20% when the market falls 10%.
- **β < 1.0**: The stock *dampens* market moves. A stock with β = 0.5 typically moves only half as much as the market.
- **β < 0**: Very rare — the stock tends to move *opposite* the market. Gold sometimes has a negative beta relative to equity indices.

Beta is estimated empirically from historical return data, typically using 3–5 years of monthly returns. It is not constant — a company's beta changes as its business changes, its leverage changes, or market conditions shift.

![Stock betas comparison horizontal bar chart](/imgs/blogs/risk-required-return-capm-beta-cost-capital-4.png)

Look at the range in the bar chart. TSLA (β ≈ 2.14) is almost twice as sensitive to market moves as the average stock. In a bull market, that's exciting; in a bear market, it's painful. JNJ (β ≈ 0.54) barely reacts to market swings — people still buy Band-Aids and prescription drugs when the economy slows. VCB.HM (Vietcombank, β ≈ 0.82) trades somewhat below market sensitivity, while HPG.HM (Hoa Phat Group, β ≈ 1.15) is slightly above market sensitivity, consistent with its steel business being cyclical.

### What Beta Does and Doesn't Capture

Beta captures only the *linear* relationship between a stock and the market. It ignores:
- Non-linear exposures (the stock does fine in mild downturns but collapses in deep crises)
- Liquidity risk (how hard it is to sell the stock at a fair price)
- Tail risk (the probability of an extreme, catastrophic loss)
- Currency risk for cross-listed or foreign-earnings companies

For all its limitations, beta is the single most widely used measure of systematic risk, and it serves as the backbone of CAPM.

### Computing Beta from Raw Data

To ground this in practice, here is how beta is computed. Collect monthly stock returns and monthly market (S&P 500) returns over 5 years. Subtract the risk-free rate (T-bill rate) from each to get *excess returns*. Plot the stock's excess return (y-axis) against the market's excess return (x-axis). Run an ordinary least-squares regression. The slope of the best-fit line is beta.

In Python, the calculation looks like:

```python
import numpy as np

  # monthly_stock_excess: array of stock excess returns
  # monthly_mkt_excess: array of market excess returns
beta = np.cov(monthly_stock_excess, monthly_mkt_excess)[0][1] / np.var(monthly_mkt_excess)
```

The R-squared of that regression tells you what fraction of the stock's return variance is explained by market movements. A high R-squared (say, 0.7) means the stock is tightly linked to the market — systematic risk dominates. A low R-squared (say, 0.2) means the stock dances to its own drum — most of its volatility is idiosyncratic.

### Levered vs Unlevered Beta

A company's observed (or *levered*) beta reflects both its business risk and its financial structure. A company that borrows heavily amplifies its equity's beta — in good times, leverage magnifies returns for equity holders; in bad times, it magnifies losses.

To compare the intrinsic business risk across companies with different capital structures, analysts use *unlevered beta* (also called *asset beta*):

$$\beta_{unlevered} = \frac{\beta_{levered}}{1 + (1 - T) \times D/E}$$

where T is the corporate tax rate, D is the market value of debt, and E is the market value of equity.

This is especially useful when estimating beta for private companies. Find the average unlevered beta of comparable public peers, then re-lever it using the target company's own D/E ratio. This *bottom-up beta* approach avoids the statistical noise of estimating beta from limited private company history.

---

## The Capital Asset Pricing Model (CAPM)

CAPM is a one-liner that changed finance forever. Developed independently by William Sharpe, John Lintner, and Jan Mossin in the 1960s — building on Harry Markowitz's portfolio theory from 1952 — it formalizes a simple idea: in a world where all investors hold efficiently diversified portfolios and care only about expected return and variance, the *only* source of risk that earns compensation is systematic risk. Every stock's required return must be proportional to how much systematic risk it contributes to a well-diversified portfolio.

The genius of CAPM is reducing an enormously complex question (what return does this stock deserve?) to a single equation with three measurable inputs:

$$E(R_i) = R_f + \beta_i \times (R_m - R_f)$$

Or in words: *the required return on an asset equals the risk-free rate plus beta times the equity risk premium*.

The three inputs are: the risk-free rate (what you can earn with zero risk), beta (how much systematic risk the stock carries), and the equity risk premium (the market's price per unit of systematic risk). We will examine each in detail.

Let's unpack each piece.

**\$R_f\$ — the risk-free rate** is the return on an investment with zero default risk and zero volatility. In practice, analysts use the yield on short-term US Treasury bills or, more commonly for long-term valuations, the 10-year US Treasury yield. As of December 2024, the 10-year US Treasury yields approximately 4.57%.

**\$\beta_i\$ — beta** is the measure of systematic risk we just defined. A higher beta means more market risk, which means a higher required return.

**\$(R_m - R_f)\$ — the equity risk premium (ERP)** is the excess return that investors demand to hold the market portfolio instead of a risk-free bond. It is the *price of one unit of systematic risk*. Multiply it by beta to get the total risk premium required for a specific stock.

CAPM is elegant because it says something profound: in an efficient market, only *one* kind of risk gets priced — systematic risk, measured by beta. All other risk either gets diversified away or is not compensation-worthy because rational investors can eliminate it for free.

### The Security Market Line

The Security Market Line (SML) is the graphical representation of CAPM. On the x-axis: beta. On the y-axis: expected return. The SML is a straight line that starts at the risk-free rate when beta = 0 and rises with a slope equal to the equity risk premium.

Every correctly priced asset sits exactly on the SML. Stocks *above* the line offer more return than their beta warrants — they are *undervalued* (in CAPM terms, their *alpha* is positive). Stocks *below* the line offer less return than their beta warrants — they are *overvalued* (negative alpha). In a perfectly efficient market, no stocks would persistently sit off the line, because arbitrageurs would buy the underpriced ones and sell the overpriced ones until equilibrium was restored.

In practice, stocks sit above and below the SML for extended periods. That persistence is either evidence of market inefficiency, evidence that CAPM's model is incomplete (missing factors beyond beta), or evidence of measurement error in beta. This is the heart of active management: finding stocks that genuinely sit above the SML (offering excess returns per unit of systematic risk) rather than stocks that merely appear to do so because of a biased beta estimate.

The SML is not to be confused with the *Capital Market Line* (CML), which plots expected return against *total* standard deviation (not beta) for portfolios that combine the market portfolio with the risk-free asset. The CML applies to the set of efficient portfolios; the SML applies to individual stocks including inefficient ones.

![CAPM formula component breakdown](/imgs/blogs/risk-required-return-capm-beta-cost-capital-5.png)

#### Worked Example:

**Apple (AAPL):** Using current (Dec 2024) data:
- Risk-free rate: \$R_f\$ = 4.5% (approximating the 10-yr Treasury)
- Beta: β = 1.21 (5-year monthly, Yahoo Finance)
- Equity Risk Premium: ERP = 4.6% (Damodaran implied, Jan 2025)

$$E(R_{AAPL}) = 4.5\% + 1.21 \times 4.6\% = 4.5\% + 5.57\% = 10.07\%$$

Interpretation: an investor in AAPL needs to expect at least a 10.07% annual return to be compensated for the risk they're taking. If the stock is priced such that it's expected to return only 8% from current levels, CAPM says it's overvalued — you're not getting paid enough for the risk.

#### Worked Example:

**High-risk startup (β ≈ 2.0):** Early-stage startups are extremely volatile — their fortunes swing dramatically with the economic cycle, access to capital, and consumer sentiment. A startup with an estimated beta of 2.0:

$$E(R_{startup}) = 4.5\% + 2.0 \times 4.6\% = 4.5\% + 9.2\% = 13.7\%$$

This is the minimum acceptable annual return for a venture investor. If a startup investor can't project a path to 13–14% annualized returns (let alone the 20–30%+ that VCs typically target), the risk-adjusted case simply doesn't pencil out. CAPM here provides the *floor* — the absolute minimum above which the investment makes economic sense.

### CAPM in a DCF Model

CAPM's most important real-world application is as an input to discounted cash flow (DCF) valuation. In a DCF model, you project a company's future free cash flows and then discount them back to today at a discount rate — the *weighted average cost of capital* (WACC). The WACC is a blend of the cost of equity and the cost of debt, weighted by how much of each the company uses.

The *cost of equity* — what investors require to hold the stock — is almost always computed using CAPM. So when an analyst plugs a 10.07% cost of equity into Apple's DCF model, they are saying: "Apple's shareholders require at least 10.07% per year, and that's the rate I'll use to discount their future cash flows."

A higher beta → higher cost of equity → higher WACC → lower present value of future cash flows → lower valuation. This is the mechanical chain that connects risk to price. See [WACC: The Weighted Average Cost of Capital](/blog/trading/equity-research/wacc-weighted-average-cost-capital) for a deeper treatment of how the cost of equity is combined with the cost of debt.

---

## The Equity Risk Premium: The Price of Market Risk

The equity risk premium (ERP) is arguably the most debated single number in all of finance. It is the expected excess return of stocks over the risk-free rate — the reward investors demand for choosing equities instead of Treasuries.

There are two fundamentally different ways to estimate it, and they can disagree by several percentage points — with large consequences for any DCF model.

### Historical ERP

The simplest approach: look at history. How much have stocks returned above Treasury bills, on average, over a long period? Using US data from 1928 to 2024 (source: Damodaran Online):

- **Arithmetic mean**: about 8.4% per year — the average annual excess return
- **Geometric mean**: about 6.2% per year — the compounded annualized excess return

The gap between arithmetic and geometric is real: arithmetic is the right measure for a one-period bet; geometric reflects what a long-term buy-and-hold investor actually earned. Most practitioners lean on the geometric mean for long-horizon valuations.

The problem with historical ERP is that it's backward-looking. The conditions of 1928–2024 — US global dominance, falling long-term rates for much of the period, extraordinary technological progress — may not repeat. A 8.4% historical ERP might be a badly biased estimate of what investors should expect going forward.

### Implied ERP

A forward-looking alternative: infer the ERP from current market prices. If you take the S&P 500's current price, its expected dividends and buybacks, and an assumed long-term growth rate, you can back out the discount rate the market is *implicitly* using. Subtract the current risk-free rate, and you get the implied ERP.

As of January 2025, Damodaran's implied ERP for the US market is approximately **4.6%** — the number we use throughout this post. This is meaningfully lower than the historical arithmetic average (8.4%), which makes sense: valuations are higher today, so the expected *future* excess return is lower.

![ERP estimation methods comparison bar chart](/imgs/blogs/risk-required-return-capm-beta-cost-capital-6.png)

The bar chart illustrates how wildly different methods can diverge. If you use the arithmetic historical average (8.36%), you'll compute a cost of equity roughly 3.8 percentage points higher than if you use the implied ERP (4.6%). That's not a rounding error — it translates to a DCF valuation difference of 40–60% on a typical growth company. The choice of ERP is consequential.

Most professional analysts today use an implied or survey-based ERP in the range of 4–5.5%, with Damodaran's implied ERP as a common anchor.

### Why the ERP Varies Over Time

The implied ERP is not constant. It expands when markets are fearful and contracts when markets are complacent. During the 2008–2009 financial crisis, the implied ERP for US equities rose above 7% as investors demanded extraordinary compensation for holding risky assets. During the post-COVID euphoria of 2020–2021, it compressed toward 4% or below as investors chased returns. In January 2025, it sits at approximately 4.6%.

This time-variation in the ERP is economically meaningful. When the ERP is high, the market is telling you that risk tolerance is low — expected future returns are higher to compensate. When the ERP is compressed, the market is saying risk tolerance is high — people are willing to accept lower expected returns for the privilege of holding equities. Contrarian investors pay close attention to the implied ERP: a very compressed ERP is one signal that valuations are stretched; a very elevated ERP may signal an opportunity.

#### Worked Example:

At the end of 2021, with the S&P 500 near all-time highs and the 10-year Treasury at roughly 1.5%, the implied ERP had compressed to around 4.1%. Apply CAPM for the market portfolio (β = 1.0):

$$E(R_{market}) = 1.5\% + 1.0 \times 4.1\% = 5.6\%$$

That was the expected annual return to holding the S&P 500 from those price levels — not very inspiring given the historical average of ~10%. Alternatively, the *earnings yield* of the S&P 500 (the inverse of P/E, or 1/28 ≈ 3.6%) was below the risk-free rate of 1.5% by only a modest margin, suggesting equities were priced for perfection.

Fast-forward to October 2022: the S&P 500 had fallen ~25% and the 10-year was at 4.0%. The implied ERP had expanded to around 5.5%. Apply CAPM:

$$E(R_{market}) = 4.0\% + 1.0 \times 5.5\% = 9.5\%$$

The expected return had nearly doubled from the 2021 level — not because anything about the businesses changed, but because prices fell and the discount rate normalized. This is the mechanical core of equity market cycles: when investors lose risk appetite, they demand higher returns, prices must fall to provide those returns, and the expected forward return rises.

---

## Country Risk Premium: Adjusting for Emerging Markets

CAPM was developed with US capital markets in mind. Apply it globally and you immediately confront an uncomfortable fact: not all countries have the same risk. A US Treasury is backed by the most liquid, most creditworthy government in the world. A 10-year bond issued by the Vietnamese government, or a stock listed on the Ho Chi Minh Stock Exchange, carries additional layers of risk:

- **Currency risk**: the Vietnamese dong can depreciate against the dollar, eroding returns for foreign investors
- **Political risk**: policy changes, regulatory shifts, expropriation risk
- **Liquidity risk**: the VN-Index is smaller and less liquid than the S&P 500; exits during a crisis are harder
- **Institutional risk**: less transparent accounting, different governance standards, weaker creditor protections

These risks are systematic within Vietnam — they affect all Vietnamese assets simultaneously. So they do not diversify away within a Vietnam-only portfolio.

The *country risk premium* (CRP) is the additional return investors demand to hold emerging-market assets instead of developed-market equivalents. Damodaran estimates Vietnam's CRP at approximately **2.0%** as of 2025, based on the sovereign credit default swap spread and a scaling factor for the equity market's additional volatility.

The adjusted CAPM for an emerging market becomes:

$$E(R) = R_f + \beta \times ERP_{US} + CRP_{Vietnam}$$

For a Vietnamese stock with β = 1.0:

$$E(R) = 4.5\% + 1.0 \times 4.6\% + 2.0\% = 11.1\%$$

Compare that to the US market's 9.1% (4.5% + 4.6%). The Vietnamese investor demands ~200 basis points more per year.

### How the Country Risk Premium Is Estimated

Damodaran's method for the CRP has two components:

1. **Default spread**: the yield spread between the country's government bonds and a matched-maturity US Treasury. Vietnam's sovereign bonds trade at roughly 1.0–1.2% above Treasuries in normal conditions.

2. **Equity volatility adjustment**: the default spread reflects bond market risk. Equity markets are more volatile than bond markets, so the CRP for equities is scaled up:

$$CRP = \text{Default Spread} \times \frac{\sigma_{equity}}{\sigma_{bond}}$$

For Vietnam, if the equity market's standard deviation is about 1.5–1.7× the sovereign bond spread volatility, the CRP lands around 1.5–2.2%. Damodaran uses approximately 2.0% for Vietnam in his January 2025 update.

This number matters enormously in practice. Many international investors and foreign-owned companies operating in Vietnam use this CRP to set their hurdle rates for Vietnamese investments. A project that returns 10% might be acceptable in the US (clearing a 9.1% hurdle) but fall short of the 11.1% hurdle rate in Vietnam. This is a key reason why foreign direct investment into emerging markets is not unlimited even when growth rates are high — the required return rises with perceived risk, raising the bar for viable projects.

### The Maturity Premium for Vietnam

Some analysts add one more adjustment: a *liquidity premium* or *market development premium* for the fact that the VN-Index has thinner trading, more volatile sentiment-driven swings, and weaker institutional infrastructure than the NYSE or NASDAQ. For a foreign investor who cannot easily short stocks or hedge currency, this additional premium might be 0.5–1.0%, pushing the total required return to 11.5–12% for average-risk Vietnamese equities.

The practical implication for portfolio construction: investors allocating to Vietnamese equities should expect to sit with volatility for extended periods, and they should ensure the expected cash flows or capital gains justify the higher required return. A Vietnamese stock promising 9% annualized returns might look fine versus a savings account — but against a risk-adjusted benchmark of 11–12%, it is actually destroying value for a rational international investor.

![Country risk premium layered stack Vietnam required return](/imgs/blogs/risk-required-return-capm-beta-cost-capital-7.png)

#### Worked Example:

**Why Vietnamese stocks trade at lower P/E multiples than US stocks.** At end-2024, the VN-Index traded at roughly a 13.9× trailing P/E, while the S&P 500 traded at about 27.6×. Many investors attribute this to "VN stocks being cheap." But CAPM provides the structural explanation.

A simple *earnings yield* model says P/E ≈ 1 / (required return − growth rate). Assume a long-term earnings growth rate of 7% for Vietnam (a growing economy) and 5% for the US (a mature economy):

- US: P/E ≈ 1 / (9.1% − 5%) = 1 / 4.1% ≈ 24× (close to the actual S&P 500 valuation)
- Vietnam: P/E ≈ 1 / (11.1% − 7%) = 1 / 4.1% ≈ 24× ... wait, the math is symmetric here

The key insight: if Vietnam's higher growth exactly offsets the higher required return, the P/E ratio should be similar. The fact that the VN-Index trades at a lower P/E (13.9× vs 27.6×) suggests either the market assigns even higher required returns to Vietnamese equities than our 11.1% estimate (possibly because of even higher perceived risk, less liquid market premium, or foreign ownership limits), or that the market applies a lower growth assumption than 7%. Probably both. The CRP is a floor estimate, not a ceiling.

The deeper takeaway for a Vietnamese investor: when you see a VN-Index P/E of 14× and a US P/E of 28×, you are not automatically looking at a "cheap" market. You are looking at a market where the discount rate is structurally higher because the risk premium is structurally higher. Whether that premium is *too high* or *too low* relative to actual risk is the active management question.

---

## Estimating ERP from the Data: What History Says

Let's look at the S&P 500's actual annual returns versus T-bill rates from 2010 to 2024 to calibrate our intuition about what the equity premium actually looks like year-by-year.

![S&P 500 annual returns vs excess return over T-bills 2010 to 2024](/imgs/blogs/risk-required-return-capm-beta-cost-capital-8.png)

The chart is instructive in several ways. First, the *average* excess return over the period (roughly 14.5% given the strong bull market) is significantly higher than the implied ERP of 4.6% — but that reflects exceptional capital gains from multiple expansion, not necessarily a permanently higher risk premium. Second, notice the enormous year-to-year swing: 2013 delivered an excess return of ~32%, while 2022 delivered a loss of ~22%. No rational investor expects 4.6% per year like clockwork. The ERP is a *long-run average expectation*, not a guarantee.

Third, notice that even in negative years for stocks (2018, 2022), T-bills still returned something positive. That's the whole point of the risk-free rate — it is a floor that doesn't move with equity market swings. The investor in equities accepted additional volatility in exchange for the expectation of higher long-term returns.

---

## Limitations of CAPM and What Came After

CAPM is a model, and all models are wrong; some are useful. Its assumption set is aggressive:

- Investors hold well-diversified portfolios (ignoring home bias, concentration by choice, liquidity constraints)
- Markets are frictionless with no taxes, no transaction costs
- All investors have the same time horizon and information
- Beta is stable over time
- Investors care only about mean and variance of returns (not skewness, not kurtosis)

In practice, these assumptions fail constantly. But the model's value is not in its literal truth — it is in its structure. CAPM gives you a systematic, defensible way to think about what return compensates for what risk.

### Fama-French Three-Factor Model

In 1992, Eugene Fama and Kenneth French published research showing that beta alone does not explain all variation in stock returns. Two additional factors mattered:

1. **Size (SMB — Small Minus Big)**: Small-cap stocks tend to outperform large-cap stocks over time, even after controlling for beta. This might reflect illiquidity risk or the extra risk of small-company failure.
2. **Value (HML — High Minus Low)**: Stocks with high book-to-market ratios (*value stocks*) tend to outperform growth stocks over time. This might reflect financial distress risk — cheap stocks are cheap because they face fundamental challenges.

The Fama-French three-factor model:

$$E(R_i) = R_f + \beta_i \times ERP + s_i \times SMB + h_i \times HML$$

This model explains more of the cross-sectional variation in stock returns than CAPM alone. Later extensions added momentum (the tendency of recent winners to keep winning), profitability, and investment factors.

For most valuation work, practitioners still use CAPM for cost of equity estimation. Fama-French is more common in academic research and factor-investing contexts. See [Expected Value and Probability Distributions](/blog/trading/math-for-quants/expected-value-probability-distributions) for the statistical underpinnings of multi-factor models.

### Why Practitioners Keep Using CAPM

Despite its known flaws, CAPM survives in practice because:
- It requires only one firm-specific input (beta), which can be estimated from public data
- The alternatives (Fama-French, APT) are harder to implement and more data-hungry
- Valuation is not a physics problem seeking exact truth — a reasonable, defensible estimate beats a complex model with uncertain inputs
- DCF models are so sensitive to other assumptions (terminal growth rate, near-term FCF estimates) that the precision of the cost of equity formula often doesn't determine the outcome

### The Arbitrage Pricing Theory (APT)

A conceptually appealing alternative is the *Arbitrage Pricing Theory*, developed by Stephen Ross in 1976. APT says that asset returns are driven by multiple systematic factors — not just the broad market, but also inflation surprises, industrial production changes, unexpected yield curve shifts, and others. The required return is:

$$E(R_i) = R_f + \sum_k \beta_{ik} \times \lambda_k$$

where \$\beta_{ik}\$ is the sensitivity to factor k and \$\lambda_k\$ is the risk premium for factor k.

APT is more general than CAPM but less prescriptive — it doesn't tell you which factors to use or how to estimate the risk premia. In practice, the Fama-French model is a more workable implementation of the multi-factor idea, because it uses observable portfolio returns (the size and value factors) rather than abstract macroeconomic variables.

For most practitioners, CAPM remains the default. But awareness of Fama-French and APT prevents over-reliance on a single beta as the complete description of systematic risk.

---

## Common Misconceptions

### Misconception 1: "High beta means a bad investment"

High beta means high *systematic* risk. If you are compensated for that risk with higher expected returns, high beta can be a perfectly sensible choice for an investor with a long time horizon and risk tolerance. A portfolio of high-beta growth stocks might be exactly right for a 25-year-old saving for retirement in 40 years. The misconception is conflating beta with badness rather than with risk-return calibration.

### Misconception 2: "Low beta means safe"

Defensive stocks (utilities, consumer staples) have low beta because their cash flows are stable regardless of the economic cycle. But low beta does not mean "cannot lose money." A utility stock could be dramatically overvalued, have regulatory risk, carry heavy debt, or face a disruption that the market hasn't priced in. Low beta means low *market-correlated* risk — not zero risk. Johnson & Johnson (β ≈ 0.54) still fell 25% peak-to-trough during the 2022 rate-driven selloff.

### Misconception 3: "Beta is stable and objective"

Beta is estimated from historical data, which means it reflects the company's past risk profile. A company that was a stable industrial conglomerate and then pivoted to speculative AI acquisitions will have an outdated beta for months. Similarly, different data providers report different beta estimates depending on their lookback period (3-year vs 5-year) and return frequency (daily vs weekly vs monthly). Beta is an input, not a fact.

### Misconception 4: "The equity risk premium is the return you should expect from stocks"

The ERP is the *expected excess return over the risk-free rate*. To get the total expected stock market return, you add the risk-free rate: 4.5% + 4.6% = 9.1%. But even that is an *expectation*, not a guarantee. In any given decade, stocks can significantly underperform (the US stock market returned close to zero in real terms from 2000 to 2010). The ERP is a long-run expected value with enormous short-run variance around it.

### Misconception 5: "A higher discount rate always makes an investment worse"

A higher discount rate lowers the present value of future cash flows. But higher discount rates are used for *riskier* investments — which is appropriate. If you are analyzing a safe bond at a 4.5% discount rate and a risky growth company at a 12% discount rate, you're not unfairly penalizing the growth company. You're correctly reflecting that its cash flows are uncertain, and the market demands compensation for that uncertainty. The fair comparison is risk-adjusted: which investment offers the best return *per unit of risk*?

### Misconception 6: "CAPM is dead because researchers keep finding anomalies"

Every few years, a new paper documents a pattern that seems to beat the market on a risk-adjusted basis — momentum, low volatility, quality, and so on. Critics declare CAPM dead. But these findings actually support the spirit of CAPM rather than destroying it: they show that beta is an *incomplete* measure of systematic risk, not that systematic risk is irrelevant. The response has been to extend CAPM (Fama-French three-factor, Carhart four-factor, Fama-French five-factor) rather than replace its core logic.

CAPM may mismeasure the *quantity* of systematic risk, but its central claim — that investors demand compensation for bearing undiversifiable risk — remains the bedrock of modern asset pricing. No serious alternative to CAPM says "you don't need to be paid for risk." Every serious model agrees on the principle; they just differ on the dimensionality of risk.

### Misconception 7: "The country risk premium is fixed and known precisely"

Country risk premiums are estimates based on sovereign credit spreads, equity market volatility, and judgment. They can change dramatically in a short time. Vietnam's CRP widened significantly in 2022 as global risk-off sentiment struck emerging markets. A CRP that was 1.5% in January 2022 might have been priced at 3% by November 2022 by the market. When using a country risk premium in a DCF, the key question is not "what is Vietnam's CRP?" but "what CRP should I use for this investment's time horizon, and what happens to my valuation if it widens by 1%?" Scenario analysis around the CRP — just like around the ERP — is essential for any serious emerging-market valuation.

---

## How It Shows Up in Real Markets

### The 2022 Rate Shock and Why Growth Stocks Fell Harder

In 2022, the Federal Reserve raised the federal funds rate from near-zero to 4.25–4.50%. The 10-year Treasury yield jumped from 1.5% to 3.9%. This was a massive shock to the risk-free rate component of every CAPM equation.

For a stock with a 10-year DCF horizon, most of the value comes from terminal-year cash flows discounted over a decade. A 2.4 percentage point increase in the discount rate compounds powerfully over time: a cash flow received in year 10 that was discounted at 1.5% is suddenly discounted at 3.9%. Its present value falls by roughly 22%. For high-growth stocks whose cash flows are heavily weighted toward the distant future, this effect was catastrophic. The ARK Innovation ETF (concentrated in high-β, long-duration growth stocks) fell nearly 75% from its peak.

Meanwhile, value stocks and dividend payers — companies whose cash flows arrive sooner and whose betas are lower — held up significantly better. CAPM correctly predicted the relative performance: a stock with β = 2.0 should decline roughly twice as much as the market index for a given increase in the discount rate.

### VN-Index P/E Multiple Compression in 2022

Vietnam experienced a version of the same phenomenon in 2022. Rising global risk-free rates increased the required return for all assets globally. For Vietnamese equities, which already carry a country risk premium, the effect compounded: the VN-Index fell from around 1,530 in April 2022 to below 900 in November 2022 — a decline of over 40%. The P/E multiple contracted from about 18× to 11×.

This was a classic "discount rate expansion" event. Corporate earnings in Vietnam did not collapse 40%. Profits for most VN-listed companies continued to grow modestly. What changed was the denominator in every valuation: as rates rose globally and the perceived risk premium for emerging markets widened, the required return for Vietnamese stocks rose, and present values compressed.

### NVIDIA in 2024: When High Beta Rewards Investors

NVIDIA (NVDA) carried a beta of approximately 1.68 at year-end 2024. Using CAPM:

$$E(R_{NVDA}) = 4.5\% + 1.68 \times 4.6\% = 4.5\% + 7.73\% = 12.23\%$$

Investors required roughly 12% annual return to hold NVDA. Over calendar year 2024, NVIDIA returned approximately 170%. Investors were massively rewarded — not merely compensated for the systematic risk, but dramatically over-compensated.

Does this mean CAPM failed? Not exactly. CAPM predicts long-run average returns, not any single year's outcome. NVIDIA's extraordinary 2024 performance reflected a fundamental business surprise (AI chip demand exceeded virtually every estimate) that was unsystematic relative to the overall market at the time it developed. CAPM tells you the *required return*; the *actual return* in any year can deviate enormously. The whole point of a risk premium is that sometimes the risk materializes (the stock falls 40%) and sometimes it doesn't (the stock rises 170%). On average, across many years and many stocks, the required return and the realized return converge. This is the probabilistic nature of all risk-return relationships in finance.

### Using CAPM to Flag Overvaluation

In late 2021, speculative tech stocks were trading at 50–100× forward earnings. Apply CAPM with β = 2.0, Rf = 1.5% (the then-yield), and ERP = 4.5%:

$$E(R) = 1.5\% + 2.0 \times 4.5\% = 10.5\%$$

For a company with no near-term profits, growing at 30% per year and trading at 80× forward earnings, the only way to justify an 80× multiple is if those 30% growth rates persist for an extremely long time and you're willing to discount at near zero. At a 10.5% required return, the math doesn't work unless growth is truly extraordinary and sustained for 15–20 years. Most of these companies fell 60–90% in 2022 as the discount rate normalized. CAPM, used correctly, was telling you the math was strained.

### Using CAPM to Value Defensive Stocks

Johnson & Johnson (JNJ) is a classic defensive stock — pharmaceuticals, medical devices, consumer health products. Its beta is approximately 0.54, meaning it moves only about half as much as the market. Applying CAPM:

$$E(R_{JNJ}) = 4.5\% + 0.54 \times 4.6\% = 4.5\% + 2.48\% = 6.98\%$$

An investor requires only ~7% annual return to hold JNJ — dramatically less than AAPL's 10.07%. This is not because JNJ's business is worse. It is because JNJ's cash flows are far less sensitive to economic cycles. Hospitals buy surgical instruments in recessions just as in booms. Consumers buy Tylenol when they're sick regardless of stock market levels. The lower beta means lower systematic risk, which means a lower required return.

Here's the important implication for valuation: discounting JNJ's future free cash flows at 7% produces a higher present value than discounting them at 10%, even for identical cash flows. The lower cost of equity (from lower beta) directly inflates the valuation multiple at which JNJ should trade. This is why stable, predictable businesses with low beta command premium valuations — it's not irrational. It's rational risk pricing.

### Applying CAPM to a Vietnamese Steel Company

Hoa Phat Group (HPG.HM) is Vietnam's largest steel producer. With β ≈ 1.15, let's compute the required return for a Vietnamese investor:

- Rf = 4.5% (US 10-yr as the global benchmark; some analysts use Vietnam government bonds at ~4.5–5%)
- β = 1.15
- ERP = 4.6%
- CRP (Vietnam) = 2.0%

$$E(R_{HPG}) = 4.5\% + 1.15 \times 4.6\% + 2.0\% = 4.5\% + 5.29\% + 2.0\% = 11.79\%$$

If a DCF for HPG projects free cash flows growing at 6% perpetually, the terminal value is:

$$TV = \frac{FCF_{t+1}}{E(R) - g} = \frac{FCF}{11.79\% - 6\%} = \frac{FCF}{5.79\%} \approx 17.3 \times FCF$$

Compare to a similar steel company in the US: assuming ERP = 4.6%, Rf = 4.5%, β = 1.0, no CRP:

$$E(R_{US Steel}) = 4.5\% + 1.0 \times 4.6\% = 9.1\%$$

$$TV_{US} = \frac{FCF}{9.1\% - 4\%} = \frac{FCF}{5.1\%} \approx 19.6 \times FCF$$

The US steel company trades at a higher multiple of FCF, not because it is a better business, but because investors require a lower return for bearing less country risk. This is CAPM doing real work: quantifying the risk premium that justifies observed market multiples.

---

## Sensitivity: How Much Does the Cost of Equity Move the Valuation?

A natural question for practitioners: how sensitive is a DCF valuation to changes in the cost of equity? The answer is: extremely sensitive for long-duration cash flows, less so for near-term ones.

Consider a company generating \$100 million of free cash flow per year, growing at 3% per year in perpetuity. The terminal value formula is:

$$TV = \frac{FCF \times (1+g)}{K_e - g}$$

At \$K_e\$ = 9% (US market average), g = 3%:
$$TV = \frac{\$100M \times 1.03}{9\% - 3\%} = \frac{\$103M}{6\%} = \$1,717M$$

Now raise the cost of equity by 200 basis points to 11% (a Vietnam-level required return):
$$TV = \frac{\$103M}{11\% - 3\%} = \frac{\$103M}{8\%} = \$1,288M$$

A 2 percentage point change in the cost of equity shrinks the terminal value by **25%**. For a typical growth company where 60–80% of the total DCF value sits in the terminal value, this is enormous. It's why international investors in Vietnamese stocks who use a 11% discount rate see materially lower fair values than local investors using a 9% rate.

This sensitivity also explains why CAPM inputs deserve careful thought. When you choose an ERP of 4.6% vs 8.4% (the arithmetic historical average), you are making a choice that could change your valuation by 30–50% on a long-duration growth company. The "right" ERP is not a mechanical choice — it requires judgment about whether current market conditions are a reliable indicator of future risk premia.

## The CAPM Toolkit in Practice

When practitioners use CAPM in a real valuation, the steps are clear and sequential. The inputs are observable, the math is transparent, and the output — the cost of equity — feeds directly into every DCF model. Here is the step-by-step process:

1. **Identify the risk-free rate**: typically the 10-year government bond yield of the country whose currency the cash flows are denominated in. For US-dollar-denominated valuations, use the 10-year US Treasury.

2. **Estimate beta**: use regression of monthly excess returns over 3–5 years. For private companies or new firms without history, use the "bottom-up beta" approach: find the average beta of comparable public companies, unlever it to remove financial leverage, then re-lever it using the target company's capital structure.

3. **Choose an ERP**: most analysts use a current implied ERP (Damodaran's number is widely cited) or a survey-based estimate. Be consistent: if you use the implied ERP, use the risk-free rate from the same date.

4. **Add a country risk premium** for non-US investments. Damodaran maintains a country-by-country CRP table updated annually.

5. **Compute the cost of equity**: \$K_e = R_f + \beta \times ERP + CRP\$

6. **Combine with cost of debt** to get WACC. See [WACC: The Weighted Average Cost of Capital](/blog/trading/equity-research/wacc-weighted-average-cost-capital) for the full construction.

The cost of equity you compute here is the discount rate that goes into every DCF you run. It is not a number you set once and forget — it should be reviewed any time the risk-free rate moves materially (a 100 bps change in Treasury yields changes your cost of equity by 100 bps), any time the company's capital structure changes (re-levering raises beta), or any time comparable company betas shift (industry conditions change what counts as "normal" risk for this business). Good valuation practice treats the cost of equity as a living estimate, not a static plug. It connects directly to the [time value of money framework](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model) — a higher required return means future cash flows are discounted more aggressively, producing a lower valuation today.

---

## Further Reading and Cross-Links

This post is part of the **Asset Valuation** series. If you came here having already read the series introduction, you have the philosophical foundation. Now that you have the required return, the next natural step is applying it.

- [What Is Value? Philosophy, Frameworks, and Asset Pricing](/blog/trading/asset-valuation/what-is-value-philosophy-frameworks-asset-pricing) — the series foundation, explaining why intrinsic value exists and how markets price assets
- [Time Value of Money: The Engine Behind Every Valuation Model](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model) — the mechanics of discounting that make the required return matter
- [WACC: The Weighted Average Cost of Capital](/blog/trading/equity-research/wacc-weighted-average-cost-capital) — how the cost of equity (from CAPM) is combined with the cost of debt to build the full discount rate
- [Expected Value and Probability Distributions](/blog/trading/math-for-quants/expected-value-probability-distributions) — the mathematical underpinnings of expected returns and risk measurement

For practitioners working in Vietnamese markets, Damodaran's annual update to country risk premiums (available at his NYU website) is indispensable for keeping your CAPM inputs current. Vietnam's CRP changes year to year as its sovereign credit rating evolves and its market integration with global capital deepens.

---

*This post is educational and illustrative. It uses data sourced from public providers (Damodaran Online, Yahoo Finance, Macrotrends, JP Morgan Guide to the Markets) as cited. All worked examples are for illustration purposes and do not constitute investment advice. Required returns in real valuations are judgment calls that depend on factors not fully captured by any single model. The goal of CAPM is not a precise answer — it is a disciplined, transparent, and repeatable framework for thinking about the relationship between risk and expected return.*
