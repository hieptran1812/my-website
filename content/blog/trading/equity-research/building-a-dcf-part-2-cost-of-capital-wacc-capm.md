---
title: "Building a DCF, Part 2: The Discount Rate — WACC, CAPM, and Cost of Debt"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A first-principles guide to the discount rate in a DCF — why we discount at the cost of capital, how to build the cost of equity with CAPM, the after-tax cost of debt and the tax shield, and how the two blend into WACC, the single number that can swing a valuation by a third."
tags: ["equity-research", "corporate-finance", "wacc", "capm", "cost-of-capital", "cost-of-equity", "cost-of-debt", "beta", "equity-risk-premium", "discount-rate", "valuation", "dcf"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The discount rate is where most DCFs quietly go wrong, because a one-point change in it can swing the valuation by roughly a third — and unlike the cash flows, the rate is built from a handful of inputs you can reason about precisely.
>
> - **We discount at the cost of capital** because that is the return the people who funded the business — shareholders and lenders — *require* for the risk they take. A dollar earned far in the future is worth less today, and the cost of capital is the rate at which it shrinks.
> - **Cost of equity comes from CAPM:** Ke = risk-free rate + beta × equity risk premium. The risk-free rate is a long government bond yield; beta is how much the stock moves with the market; the equity risk premium is the extra return stocks must offer over bonds, historically about 4–6%.
> - **Cost of debt is the firm's borrowing yield, after tax:** Kd × (1 − tax rate). Interest is tax-deductible, so the government effectively pays part of the bill — debt is the cheapest capital, which is exactly why too much of it is dangerous.
> - **WACC blends the two by market-value weights:** WACC = (E/V) × Ke + (D/V) × Kd × (1 − t). It is the hurdle every project and every business must clear, and the rate at which we discount the whole firm.
> - **The classic mistakes** — book-value weights, a stale single-stock beta, a mismatched risk-free rate and risk premium, forgetting the tax shield, one WACC for divisions of wildly different risk — each push the rate the wrong way, and because value is so sensitive to it, each one quietly distorts the answer.

In [Part 1 of this series](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) we built the numerator of a discounted cash flow model: a forecast of the cash a business will throw off in the years ahead. That work is hard, judgement-laden, and important. But there is a quieter, more dangerous number sitting in the denominator of every DCF — the **discount rate** — and it is where a startling amount of valuation goes wrong. Not because the math is hard. The math is a single line. It goes wrong because the rate is assembled from inputs that look authoritative, get copied from a textbook or a data terminal without much thought, and then silently move the answer by tens of percent.

Here is the uncomfortable fact that should make you care about this post: in a typical DCF, changing the discount rate by **one percentage point** — from, say, 8% to 9% — can change the computed value of the business by **roughly 30%**. A one-point move in a number that most people set with a shrug. By contrast, you can argue for hours about whether next year's revenue grows 6% or 7% and barely move the valuation at all. The discount rate is the highest-leverage assumption in the whole model, and it is the one most often set on autopilot. This post is about taking it off autopilot.

![Two cost-of-capital legs, the high cost of equity and the lower after-tax cost of debt, merging through their market-value weights into a single blended WACC of 8.4 percent](/imgs/blogs/building-a-dcf-part-2-cost-of-capital-wacc-capm-1.png)

The figure above is the destination of this entire post. The discount rate we want is the **weighted average cost of capital, or WACC** — the blended cost of all the money a business runs on. Money comes from two kinds of people: **shareholders**, who own equity and demand a high return because they bear the most risk; and **lenders**, who hold debt and accept a lower return because they get paid first. WACC is simply the average of those two costs, weighted by how much of each the company uses. By the end of this post you will be able to build every piece of that number from scratch — the cost of equity through CAPM, the after-tax cost of debt, and the weights that blend them — and you will know exactly which of the standard mistakes to avoid. We will compute the whole thing for a company called **Northwind Industries**, the same fictional industrial-pump maker we have used throughout this series, so the numbers compound rather than reset.

## Foundations: required return, opportunity cost, and the time value of money

Before we touch CAPM or WACC, we need to be precise about *why* a discount rate exists at all. The whole edifice rests on three ideas, and if you hold them firmly the rest of the post is bookkeeping.

**First: a dollar in the future is worth less than a dollar today.** This is the time value of money, and it is not a trick of inflation — it is true even with zero inflation. If someone offers you a guaranteed \$100 a year from now, you would not pay \$100 for it today, because you could take that \$100 now, invest it safely, and have more than \$100 in a year. The future \$100 is therefore worth *less* than \$100 today. *How much* less depends on the return you could earn in the meantime — and that return is the discount rate. To find the value today (the **present value**) of a future cash flow, we divide it by one plus the discount rate, once for each year we wait:

$$\text{PV} = \frac{\text{Cash flow in year } n}{(1 + r)^n}$$

If the rate is 8%, a \$100 cash flow due in one year is worth \$100 ÷ 1.08 = \$92.59 today; due in ten years it is worth \$100 ÷ 1.08¹⁰ = \$46.32; due in thirty years, just \$9.94. The further out the cash, the more the discount rate bites — and notice it bites *exponentially*, not linearly, because the rate compounds: it is raised to a higher power for each additional year of waiting. A \$100 payment three decades away is worth less than a tenth of its face value at an 8% rate. This exponential decay is the mathematical heart of everything that follows: it is why the rate matters so enormously for businesses whose value lies in cash flows many years away, and why a one-point change in the rate, compounded across decades, moves the answer so violently.

**Second: the right discount rate is an opportunity cost.** When you put money into a business, you give up the chance to put it somewhere else of similar risk. The return you forgo by choosing *this* investment over the next-best alternative of equal risk is the opportunity cost of the capital — and that is precisely the return this investment must clear to be worth making. A safe government bond and a speculative biotech do not share a discount rate, because the alternatives you give up are not the same. **Risk sets the rate.** The more uncertain the cash flows, the higher the return investors demand to bear that uncertainty, and the higher the discount rate.

**Third: the rate is a *required* return, set by the capital providers, not by the company.** A business does not get to decide what return its investors deserve any more than a borrower decides their own interest rate. The lenders and shareholders, collectively, demand a return commensurate with the risk they are taking, and the company must earn at least that much or it is destroying value. This is the crucial reframing: the discount rate is not a number the analyst invents — it is the *market's required return* on capital of this risk, which the analyst's job is to estimate. The cost of capital and the required return are the same number seen from two sides: it is a *cost* to the company that must pay it and a *required return* to the investors who demand it.

With those three ideas, the structure of the rest of the post is fixed. The capital comes from two sources — equity and debt — each with its own required return. We estimate each, then blend them. Let us define the remaining vocabulary precisely.

**Equity** is ownership. Shareholders buy a piece of the company and are entitled to whatever is left after everyone else — suppliers, employees, lenders, the tax authority — has been paid. Because they are last in line and their return is uncapped but uncertain, equity is the *riskiest* capital, and shareholders demand the *highest* return for it. That required return is the **cost of equity**, written Ke.

**Debt** is a loan. Lenders give the company money and are owed fixed interest and repayment on a schedule, ahead of shareholders. Because they get paid first and can seize collateral or force bankruptcy if unpaid, debt is *safer* than equity, so lenders accept a *lower* return. That return is the **cost of debt**, written Kd.

**The risk-free rate**, written rf, is the return on an investment with no default risk and known cash flows — in practice, the yield on a long-dated government bond from a stable issuer (a 10-year U.S. Treasury, say). It is the floor under every other rate: the return you can get for taking essentially no risk. Every risky asset must offer *more* than this, and "how much more" is the central question of asset pricing.

**Beta**, written β, measures how much an asset's price moves *with the overall market*. A beta of 1.0 means the stock tends to move one-for-one with the market; 1.5 means it amplifies market moves by half again; 0.5 means it moves only half as much. Beta is the standard measure of the kind of risk that *cannot be diversified away* — the risk you are still exposed to even after holding hundreds of stocks — and CAPM says it is the only risk investors get paid to bear.

**The equity risk premium**, or ERP, is the *extra* return investors demand for holding stocks in general rather than safe government bonds. It is the reward for bearing market risk, and historically it has been on the order of 4–6% per year. The ERP is the single most argued-over number in finance, and we will spend real time on it.

**WACC**, the weighted average cost of capital, is the blend of Ke and Kd, weighted by the proportion of each in the company's capital. It is the discount rate for the *whole firm* — the rate that values the cash flows available to *all* capital providers together. It is also the company's **hurdle rate**: the minimum return any investment must earn to be worth funding, a connection explored in depth in [the cost of capital and the hurdle rate](/blog/trading/equity-research/cost-of-capital-and-the-hurdle-rate).

One more foundational distinction, because it determines *which* discount rate you use. A DCF can be built on **free cash flow to the firm (FCFF)** — the cash available to *everyone*, before any payments to lenders — or on **free cash flow to equity (FCFE)** — the cash left for *shareholders alone*, after lenders are paid. The two require different discount rates: FCFF is discounted at **WACC** (the blended cost of all capital, because the cash belongs to all providers), while FCFE is discounted at the **cost of equity alone** (because that cash belongs only to shareholders). This must be kept consistent — discounting firm cash flows at the cost of equity, or equity cash flows at WACC, is a classic and value-distorting error. The distinction is developed fully in [free cash flow: FCFF vs FCFE](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe); for this post, our headline rate is WACC, the rate for the most common DCF, which discounts FCFF.

## The cost of equity: building Ke with CAPM

Equity is the hard one. With debt, you can often just look up what the company pays to borrow. But shareholders do not have a stated interest rate — nobody sends them a coupon labelled "your required return is 10%." Their required return is *implicit*, buried in the price they are willing to pay for the stock, and we have to infer it. The standard tool for that inference is the **Capital Asset Pricing Model**, or CAPM (pronounced "cap-em").

CAPM is one elegant idea: **the only risk investors get compensated for is the risk they cannot diversify away.** If you hold one stock, you bear two kinds of risk — risk specific to that company (a factory fire, a botched product launch) and risk shared by the whole market (a recession, a rate shock). The company-specific risk you can eliminate for free, simply by holding many stocks, because the factory fire at one is offset by the lucky break at another. Since you can erase that risk at no cost, the market will not pay you to bear it. What you *cannot* erase is the market-wide risk — when the whole economy sinks, diversification does not save you. *That* is the risk investors must be paid to hold, and beta is its measure. CAPM turns this into a formula:

$$K_e = r_f + \beta \times \text{ERP}$$

In words: the return a stock must offer equals the risk-free rate (what you get for no risk) plus the stock's beta times the equity risk premium (the reward for the market risk you are forced to bear). It is a floor plus a risk-scaled premium. Let us look at each input, then build it up.

![The cost of equity built as a vertical stack, starting from a four percent risk-free floor, adding six percent for beta times the equity risk premium, reaching a ten percent required return](/imgs/blogs/building-a-dcf-part-2-cost-of-capital-wacc-capm-2.png)

The figure shows CAPM as exactly what it is — a *build-up*. You start at the risk-free floor (4%), the return for taking no risk at all. On top of that you stack the **risk premium for this specific stock**, which is the market's risk premium (the ERP, here 5%) scaled by how much market risk this stock carries (its beta, here 1.2). That gives 1.2 × 5% = 6%. Add the two and you have the cost of equity: 4% + 6% = 10%. Every cost of equity you will ever compute is this same two-layer stack; the only thing that changes is the height of each layer.

### The risk-free rate: which government yield, and why long

The risk-free rate is the foundation, so getting it right matters. In principle it is the return on a truly riskless asset; in practice we use the **yield on a long-dated government bond** from a stable, own-currency issuer — for U.S. companies, the 10-year (or sometimes 20- or 30-year) Treasury yield.

Two choices trip people up. **First, which maturity?** The instinct is to grab a short rate like the 3-month T-bill because it is "more truly riskless." That is a mistake. A DCF values cash flows stretching decades into the future, and the discount rate should match that horizon. A 10-year (or longer) government yield reflects the market's expectations over a long period and is the right horizon match for a long-lived stream of cash. Using a 3-month rate imports today's short-term monetary conditions into a 30-year valuation, which is a horizon mismatch. **Second, which country's bond?** The risk-free rate should be in the same currency as the cash flows, and from an issuer that genuinely will not default in that currency. A U.S. Treasury yield is the standard for dollar cash flows; for a company in an economy whose own government carries real default risk, the local government yield is *not* risk-free, and you must build the rate up differently (more on country risk shortly).

For Northwind, a U.S. industrial company, we use the 10-year Treasury yield, which we will take as **4.0%**. That is our floor.

### Beta: market sensitivity, and the levering problem

Beta is the most slippery input in CAPM, because it is both *estimated with error* and *contaminated by the company's debt*. Let us take those in turn.

A stock's beta is typically *estimated* by running a statistical regression of the stock's returns against the market's returns over some past window — say, five years of monthly data. The slope of that regression is the **raw historical beta**: how much the stock has moved, on average, for each 1% move in the market. The problem is that this estimate is noisy. Two years of different data can hand you a beta of 1.1 or 1.4 for the same company. It also looks backward, while the discount rate is about the future. For these reasons, practitioners rarely trust a single stock's raw beta. They smooth it (a common adjustment nudges the raw beta toward 1.0, on the empirical observation that betas drift toward the market over time), and — more importantly — they often estimate beta from a *group of comparable companies* rather than one noisy stock. That averaging is more robust, but it introduces a new problem: different companies carry different amounts of debt, and **debt changes beta.**

Here is the intuition. Beta measures the risk borne by *equity holders*. A company with a lot of debt has riskier equity than an otherwise-identical company with none, because debt adds fixed interest payments that must be met in good times and bad — leverage amplifies the ups and downs that reach the shareholders. So two companies in the same business, with the same *underlying* operating risk, will show *different* equity betas purely because they carry different amounts of debt. The one with more debt will show a higher beta. That extra beta is *financial* risk, not business risk, and if you borrow a comparable company's beta without adjusting for its leverage, you import its capital structure into your estimate.

The fix is a two-step dance: **unlever** each comparable's observed beta to strip out *its* debt, leaving the pure business risk (the **asset beta**, also called the unlevered beta), then **relever** that asset beta to add back *your* company's debt. The standard formula (the "Hamada" relationship) is:

$$\beta_{\text{unlevered}} = \frac{\beta_{\text{levered}}}{1 + (1 - t)\frac{D}{E}}$$

and relevering inverts it:

$$\beta_{\text{levered}} = \beta_{\text{unlevered}} \times \left[1 + (1 - t)\frac{D}{E}\right]$$

where D/E is debt-to-equity and t is the tax rate (the tax term appears because the tax-deductibility of interest dampens how much leverage amplifies risk). You do not need to memorize the algebra; you need to understand the *motion* — pull the comparable's leverage out, push your own leverage in.

![Unlevering a peer beta of 1.5 down to an asset beta of 0.84, then relevering it at Northwind's lighter debt to arrive at an equity beta of 1.13 to use in CAPM](/imgs/blogs/building-a-dcf-part-2-cost-of-capital-wacc-capm-4.png)

The figure walks the full round trip. On the left, a comparable company shows an observed equity beta of 1.50, but it is heavily levered (debt-to-equity of 1.0). Strip that leverage out and the underlying business risk — the asset beta — is only 0.84. On the right, we carry that 0.84 of pure business risk over to Northwind, which carries less debt (debt-to-equity of about 0.43), and relever it to get an equity beta of 1.13. *That* is the beta that belongs in Northwind's CAPM, because it reflects Northwind's actual business and Northwind's actual financing. The contrast on the figure is the whole lesson: the same business risk (0.84) produces a different equity beta depending on how much debt sits on top of it.

#### Worked example: unlevering and relevering a comparable's beta

Northwind has no long, clean trading history of its own, so we estimate its beta from a close comparable — call it **Stalwart Pumps** — a peer in the same industry. Stalwart's raw equity beta, from a regression, is **1.50**. But Stalwart is financed aggressively: it carries debt-to-equity (D/E) of **1.0** (equal parts debt and equity). With a tax rate of 21%, we unlever Stalwart's beta to find the pure business risk:

$$\beta_{\text{asset}} = \frac{1.50}{1 + (1 - 0.21)(1.0)} = \frac{1.50}{1 + 0.79} = \frac{1.50}{1.79} = 0.84$$

So the underlying pump business, with no debt at all, has a beta of about **0.84** — it is slightly less volatile than the market once you remove the leverage. Now we relever to Northwind's own capital structure. Northwind carries \$300M of debt against an *equity market value* of about \$700M (we will justify that market value shortly), so its D/E is 300 ÷ 700 = **0.43**:

$$\beta_{\text{Northwind}} = 0.84 \times \left[1 + (1 - 0.21)(0.43)\right] = 0.84 \times \left[1 + 0.34\right] = 0.84 \times 1.34 = 1.13$$

Northwind's relevered equity beta is about **1.13**. Notice it is below Stalwart's 1.50 — because Northwind uses less debt, its equity is less risky, even though the two run the same kind of business.

*The beta you should use is never just "the comparable's beta"; it is the comparable's business risk, refinanced with your company's balance sheet.*

A practical note: the example uses a clean 1.13 from one comparable, but the brief of beta estimation in practice is to gather *several* comparables, unlever each, average the asset betas (to wash out single-stock noise), then relever the average. The principle is identical; there are just more rows. For the rest of this post, to keep the headline numbers round and to match the canonical CAPM example, we will use a beta of **1.2** for Northwind — close to the 1.13 we computed, and a reminder that beta is an *estimate*, comfortably a range rather than a decimal-precise fact.

### The equity risk premium: the most argued-over number in finance

The equity risk premium is the extra annual return investors demand for holding stocks instead of risk-free bonds. It is also genuinely uncertain — reasonable experts put it anywhere from about 4% to 6%, and that range alone is enough to move a valuation materially. Where does it come from? There are two main approaches.

**The historical approach** looks at how much stocks have actually out-returned government bonds over long periods — a century or more of data. In the U.S., realized equity returns have beaten long-term government bonds by roughly 4–6% per year on average, depending on the exact period and whether you use an arithmetic or geometric mean. The appeal is that it is grounded in real, observed data. The drawback is that the past may not repeat — and there is a subtle survivorship problem, because the markets with the best long histories (like the U.S.) are partly chosen *because* they did well, which biases the historical premium upward.

**The implied (forward-looking) approach** flips the logic. Instead of looking backward, it asks: *given today's stock prices and analysts' forecasts of future cash flows, what return is the market pricing in?* You take the current price of the whole market, the expected dividends and buybacks, and solve for the discount rate that makes them consistent — that rate, minus the risk-free rate, is the *implied* ERP. Its appeal is that it reflects today's valuations rather than ancient history; its drawback is that it depends on the cash-flow forecasts you feed in. In recent years the implied U.S. ERP has tended to land in the 4–5.5% range.

In practice, many analysts settle on a number around **5%** as a reasonable central estimate, often cross-checking the historical and implied figures. We will use **5.0%** for Northwind, while being honest that this is the single softest input in the whole calculation.

#### Worked example: computing Northwind's cost of equity with CAPM

We now have all three inputs. Risk-free rate rf = **4.0%** (the 10-year Treasury). Beta β = **1.2** (Northwind's relevered equity beta). Equity risk premium ERP = **5.0%**. Plug into CAPM:

$$K_e = r_f + \beta \times \text{ERP} = 4.0\% + 1.2 \times 5.0\% = 4.0\% + 6.0\% = 10.0\%$$

Northwind's shareholders require a **10.0%** annual return. Read it as a sentence: they want 4% just for parting with their money (the risk-free floor), plus another 6% to compensate for the fact that Northwind's stock is 1.2 times as sensitive to market swings as the average stock (1.2 × the 5% market reward). If Northwind cannot earn at least 10% on the equity capital invested in it, its shareholders would be better off in a diversified market index, and the company is destroying value for them.

*The cost of equity is not a fee the company pays out in cash — it is the return shareholders silently demand, and a business that fails to deliver it is quietly making its owners poorer than the alternatives.*

### The Security Market Line: CAPM as a picture

CAPM is more illuminating drawn than written. Put **beta on the horizontal axis** and **required return on the vertical axis**, and the model becomes a single straight line — the **Security Market Line (SML)**. Every asset in the world, according to CAPM, must sit *on* this line: its required return is determined entirely by where its beta falls.

![The Security Market Line plotting required return against beta, a straight line from the four percent risk-free intercept rising five percent for each unit of beta, with Northwind at beta 1.2 landing at ten percent](/imgs/blogs/building-a-dcf-part-2-cost-of-capital-wacc-capm-3.png)

The line is fully described by two numbers. Its **intercept** — where it crosses the vertical axis at beta = 0 — is the risk-free rate, 4%: an asset with no market sensitivity earns the riskless return and nothing more. Its **slope** is the equity risk premium, 5% per unit of beta: each additional unit of market risk earns another 5% of required return. Walk along the line and you can read off any asset's cost of equity. At beta = 1.0 sits the *market itself*, requiring rf + 1 × ERP = 4% + 5% = 9%. At beta = 1.2 sits Northwind, requiring 4% + 1.2 × 5% = 10% — a touch above the market, because it carries a touch more market risk. Defensive, low-beta assets sit low and to the left (low risk, low required return); aggressive, high-beta assets sit high and to the right.

The SML also reframes what "mispricing" means in CAPM's world. A stock that offers a *higher* return than its beta warrants would plot *above* the line — a bargain, in CAPM terms, because you are paid more than the risk demands. One offering less plots below the line — overpriced. In equilibrium, CAPM says, competition drags every asset onto the line. Whether real markets actually behave this way is exactly what the critiques in the next section dispute — but as a disciplined way to translate risk into a required return, the line is hard to beat.

### Size, country, and the limits of CAPM

CAPM is a clean model, and clean models leave things out. Three adjustments and critiques are worth knowing.

**The size premium.** Empirically, smaller companies have tended to deliver higher returns than their betas alone predict — they are riskier in ways beta does not fully capture (less liquidity, more fragility, thinner access to capital). Many practitioners add a **size premium** of a percent or two to the cost of equity for small companies, on top of the CAPM number. It is a patch on CAPM's known under-pricing of small-cap risk, and how big a patch is itself debated.

**Country risk.** For a company operating in an emerging market, the local government bond is not truly risk-free, and the equity carries political, currency, and expropriation risks that a U.S. or German firm does not. The standard fix is to add a **country risk premium** — often derived from the spread between the country's government bonds and a benchmark like U.S. Treasuries, sometimes scaled up because equities are riskier than bonds. A DCF of a Brazilian or Indonesian company that uses a bare CAPM with a U.S. risk-free rate will badly understate the required return.

**The deeper critiques.** CAPM has been challenged for decades. Beta is unstable and hard to estimate; the model's central prediction (that beta alone explains returns) holds weakly in the data; and other factors — company size, value-vs-growth, profitability, momentum — explain returns that CAPM cannot. Multi-factor models (most famously the Fama–French models) add these factors and fit historical returns better. There is also the **build-up method**, common for small private companies, which skips beta entirely and literally *adds up* premiums: risk-free rate + general equity premium + size premium + industry premium + a company-specific premium. It is cruder and more subjective than CAPM, but for a tiny private business with no usable beta, it can be more honest. Despite all this, CAPM remains the workhorse of the cost of equity, because it is simple, transparent, and forces you to reason explicitly about the two things that matter most — the floor and the risk premium. Use it, but hold its output as a *range*, not a precise number.

## The cost of debt: cheaper than it looks, because of tax

Debt is the easier leg, and there is one beautiful idea that makes it cheaper than its sticker price: **interest is tax-deductible.**

The starting point is the company's **pre-tax cost of debt** — the rate it actually pays to borrow. There are two ways to find it. The clean way, if the company has bonds trading in the market, is the **yield to maturity (YTM)** on those bonds — the return a buyer would earn holding the bond to maturity at today's price. The YTM, not the bond's original coupon, is what matters, because it reflects the *current* market's required return on the company's debt. (The coupon is a historical artifact of when the bond was issued; if the company's risk has changed, the YTM will have moved while the coupon stayed fixed.) The other way, when there are no traded bonds, is to build it up: take the **risk-free rate** and add a **credit spread** appropriate to the company's credit quality. A company rated BBB might borrow at the risk-free rate plus a spread of, say, 2%; a riskier company pays a wider spread. Either way you arrive at the pre-tax cost of debt — what lenders charge.

Now the magic. When a company pays interest, that interest is a deductible expense — it reduces the company's taxable income, and therefore its tax bill. So every dollar of interest does not actually cost the company a full dollar; it costs a dollar *minus* the tax the company would otherwise have paid on that dollar. If the tax rate is 21%, then each dollar of interest saves 21 cents of tax, so the *true* cost is only 79 cents. This is the **interest tax shield**, and it is why the relevant cost of debt for WACC is the **after-tax** cost:

$$K_d^{\text{after-tax}} = K_d^{\text{pre-tax}} \times (1 - t)$$

![Interest of eighteen million on three hundred million of debt at a six percent pre-tax rate, reduced by a twenty-one percent tax shield to a real after-tax cost of debt of 4.74 percent](/imgs/blogs/building-a-dcf-part-2-cost-of-capital-wacc-capm-5.png)

The figure makes the tax shield concrete. On the left is the naive view: the company borrows at 6%, pays interest, and "debt costs 6%." On the right is the truth: because that interest is deductible, the tax authority effectively refunds the tax rate times the interest. The 6% pre-tax rate becomes a 4.74% after-tax rate — debt is meaningfully cheaper than its headline yield. This tax shield is the single biggest reason debt is the cheapest form of capital, and (as we will see) the reason companies that earn well above their cost of capital can create value by using *some* debt — while too much of it courts disaster, a tension explored in [leverage and coverage](/blog/trading/equity-research/leverage-and-coverage-debt-that-compounds-vs-kills).

#### Worked example: Northwind's after-tax cost of debt

Northwind carries \$300M of debt, on which it pays a pre-tax interest rate of **6.0%** — that is its YTM, the market's current required return on its bonds. Its tax rate is **21%**. First, the annual interest in dollars:

$$\text{Interest} = \$300\text{M} \times 6.0\% = \$18.0\text{M}$$

That \$18.0M is deductible, so it shields income from tax. The tax saved is:

$$\text{Tax shield} = 21\% \times \$18.0\text{M} = \$3.78\text{M}$$

So although Northwind sends \$18.0M to bondholders, its *net* cost after the tax saving is \$18.0M − \$3.78M = **\$14.22M**. As a rate, that is:

$$K_d^{\text{after-tax}} = 6.0\% \times (1 - 0.21) = 6.0\% \times 0.79 = 4.74\%$$

Northwind's real cost of debt is **4.74%**, not 6%. The 1.26-point gap is the government quietly subsidizing the borrowing through the deductibility of interest.

*Debt's true cost is its yield net of the tax it saves — which is why a profitable, tax-paying company finds debt cheaper than an unprofitable one that has no taxable income for the interest to shield.*

That last point is a subtlety worth holding: the tax shield is only worth something if the company actually pays tax. A loss-making firm with no taxable income gets no benefit from interest deductibility, so its after-tax cost of debt is *not* reduced — for such a firm, you would not apply the (1 − t) discount, or you would apply it only to the extent it expects to be a taxpayer. The shield is real, but it is contingent on profits.

## The weights: market value, not book value

We now have the two costs — Ke = 10.0% and after-tax Kd = 4.74%. To blend them into WACC, we need the *weights*: what fraction of the company's total capital is equity, and what fraction is debt. This is where a common, value-distorting error lives, so let us be careful.

The weights must be at **market value**, not **book value**. The distinction is fundamental. The **book value** of equity is an accounting figure on the balance sheet — roughly the historical money shareholders put in plus accumulated retained earnings. It often bears little relationship to what the equity is actually worth. The **market value** of equity is what the stock market says the equity is worth right now: the share price times the number of shares (the **market capitalization**). For a successful company, market value can be many times book value; for a distressed one, it can be far below. (For a refresher on the two, see [the balance-sheet post](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth).)

Why must we use market value? Because WACC is meant to capture the return investors *require on the money they have at risk today* — and the money at risk is the *market* value of their stake, not what it cost them years ago. A shareholder who owns stock worth \$700M today requires a return on \$700M, regardless of whether the company's books show \$400M of equity. Using book weights pretends investors only care about historical cost, which is simply false. For debt, market and book value are usually close (especially for shorter-term debt), so book debt is often an acceptable approximation — but for equity, the gap can be enormous, and using book equity can throw the weights, and therefore WACC, badly off.

A second, forward-looking refinement: strictly, WACC should use the company's **target** capital structure — the mix of debt and equity it intends to maintain over the long run — not necessarily today's snapshot, which might be temporarily distorted (a company mid-way through paying down a big acquisition loan, say). For a stable company, today's market-value mix is a fine proxy for the target. For one whose financing is in flux, you reason about where it is heading. The weights should reflect the *sustained* capital structure that will finance the cash flows you are discounting.

With Northwind's equity worth \$700M at market and its debt \$300M, the total capital (often called **enterprise value** in this context, V = E + D) is \$1,000M. The equity weight is E/V = 700 ÷ 1,000 = **70%**; the debt weight is D/V = 300 ÷ 1,000 = **30%**. We are ready to blend.

## WACC: blending the two costs

The weighted average cost of capital is exactly what its name says — the average of the two costs of capital, weighted by how much of each the company uses:

$$\text{WACC} = \frac{E}{V} K_e + \frac{D}{V} K_d (1 - t)$$

The first term is the equity weight times the cost of equity; the second is the debt weight times the *after-tax* cost of debt. Add them, and you have the single rate at which the whole firm's cash flows are discounted.

#### Worked example: Northwind's WACC

We have every piece. Equity weight E/V = **70%**; cost of equity Ke = **10.0%**. Debt weight D/V = **30%**; after-tax cost of debt = **4.74%**. Blend them:

$$\text{WACC} = (0.70 \times 10.0\%) + (0.30 \times 4.74\%)$$

$$\text{WACC} = 7.00\% + 1.42\% = 8.42\%$$

Northwind's WACC is about **8.4%**. That is the rate at which we discount Northwind's free cash flow to the firm to value the entire business. Look at how the two pieces contribute: equity, despite being only 70% of the capital, supplies 7.00 of the 8.42 points, because it is both the larger weight *and* the more expensive capital; debt, at 30% of the capital and a cheap after-tax 4.74%, adds just 1.42 points. This is the general shape — equity dominates most companies' WACC, both because most firms are majority equity-funded and because equity is the costlier capital.

*WACC is a weighted average in the most literal sense: it sits between the cost of debt and the cost of equity, pulled toward whichever the company uses more of.*

Notice, too, what would happen if Northwind used *more* debt. Debt is cheaper than equity, so shifting the mix toward debt would, mechanically, pull WACC down — up to a point. That is the seductive logic of leverage, and it is only half true. As a company piles on debt, two things push back: lenders demand higher rates (the credit spread widens) as default risk rises, *and* the equity becomes riskier (beta rises, via the relevering we did earlier), so the cost of equity climbs too. Beyond a moderate level of debt, those effects overwhelm the tax-shield savings and WACC starts rising again. There is, in theory, an optimal capital structure that minimizes WACC — enough debt to capture the tax shield, not so much that distress costs and rising rates swamp it. This is why "just add more debt to lower the discount rate" is a trap, and why the *target* structure in WACC should be a sustainable one.

#### Worked example: why more debt does not keep lowering WACC

It is tempting to think that because debt (4.74% after-tax) is so much cheaper than equity (10%), Northwind should just keep substituting debt for equity to drive WACC down. Let us test the naive version first, holding the two costs *fixed* and only changing the weights. At Northwind's actual 70/30 mix, WACC is 8.42%, as we computed. Push the mix to 50/50, still holding Ke at 10% and after-tax Kd at 4.74%:

$$\text{WACC}_{\text{naive}} = (0.50 \times 10.0\%) + (0.50 \times 4.74\%) = 5.00\% + 2.37\% = 7.37\%$$

That looks like a free lunch — a full point of WACC saved just by reshuffling. But the costs do *not* hold fixed. At 50/50, debt-to-equity rises from 0.43 to 1.0, so Northwind's relevered beta jumps. Using our relevering formula with the 0.84 asset beta: β = 0.84 × [1 + 0.79 × 1.0] = 0.84 × 1.79 = **1.50**, lifting the cost of equity to Ke = 4% + 1.50 × 5% = **11.5%**. And the heavier debt load widens Northwind's credit spread, lifting the pre-tax cost of debt from 6% to, say, 7.5%, an after-tax 5.93%. Re-blend with the *correct* costs:

$$\text{WACC}_{\text{real}} = (0.50 \times 11.5\%) + (0.50 \times 5.93\%) = 5.75\% + 2.96\% = 8.71\%$$

So the *real* WACC at 50/50 is **8.71%** — *higher* than the 8.42% at 70/30, not lower. The naive calculation said leverage would save a point; the honest calculation, which lets the costs respond to the added risk, says it would *raise* WACC. The two opposing forces — cheaper average cost from more debt, versus higher costs on both legs from more risk — are exactly the tension that creates an optimal structure somewhere in the middle.

*Leverage only lowers WACC while the tax-shield savings outrun the rising cost of risk; past that point, more debt makes the discount rate worse, not better — which is precisely why you cannot read WACC off the weights alone.*

## Why WACC and ROIC together define value creation

WACC is not just a discounting input; paired with returns on capital, it is the engine of value creation itself. Recall from [the returns-on-capital post](/blog/trading/equity-research/returns-on-capital-roic-roe-roa) that **ROIC** (return on invested capital) measures the after-tax operating profit a business earns on each dollar of capital invested in it. WACC measures the *cost* of that capital. The relationship between them is the single most important comparison in business analysis:

- **If ROIC > WACC**, the business earns more on its capital than the capital costs. Every dollar invested creates value, and — crucially — *growth multiplies that value*. A company earning 18% on capital that costs 8% should reinvest every dollar it can.
- **If ROIC = WACC**, the business earns exactly its cost of capital. It is running to stand still: growth neither creates nor destroys value, it just gets bigger.
- **If ROIC < WACC**, the business earns *less* than its capital costs. Now the logic inverts horribly: **growth destroys value.** Every additional dollar invested earns less than it costs, so the more the company grows, the more value it incinerates. A shrinking, low-return business can be worth *more* than a growing one.

This is why WACC is the line in the sand. Northwind earns an ROIC of 18% (from the returns-on-capital post) against a WACC of 8.4% — a **spread of about 9.6 points**. That positive spread is the source of its value, and the reason its growth is worth paying for. Get the WACC wrong — say it as 11% instead of 8.4% — and you would shrink that spread, understate the value of growth, and badly misjudge the business. The discount rate is not a technicality bolted onto the valuation; it is half of the value-creation equation.

## The sensitivity of value to WACC

We have arrived at the warning in this post's title. Because value is the present value of cash flows stretching far into the future, and because the discount rate compounds — it appears raised to higher and higher powers for more distant years — value is *extraordinarily* sensitive to the rate. Small changes in WACC produce large changes in value.

![A steeply falling convex curve of enterprise value against WACC, where raising the rate from eight to nine percent drops the value from eight hundred thirty-three million to seven hundred fourteen million, a fourteen percent swing](/imgs/blogs/building-a-dcf-part-2-cost-of-capital-wacc-capm-6.png)

The figure plots Northwind's enterprise value against the WACC used to discount it, holding the cash flows fixed. Two features stand out. **First, the curve slopes steeply downward** — every increase in the rate cuts the value, and the cuts are large. **Second, the curve is convex (it steepens at low rates)** — at low discount rates, value explodes upward, because distant cash flows are barely discounted and dominate the total. This convexity is why valuations of long-duration assets (high-growth companies whose cash is years away) are *especially* sensitive to the rate. The mark on the figure shows the headline: moving WACC from 8% to 9% drops the value from \$833M to \$714M — a **14% haircut from a single percentage point.** And the swing is larger still at lower rates: from 6% to 7%, value falls from \$1,250M to \$1,000M, a 20% drop.

#### Worked example: value swinging when WACC moves from 8% to 9%

Let us make the sensitivity concrete with a simple, transparent model. Suppose Northwind throws off free cash flow that next year is \$50M and grows forever at 2% per year. The value of a growing perpetuity is FCF ÷ (WACC − g), where g is the growth rate. At a WACC of **8%**:

$$\text{Value} = \frac{\$50\text{M}}{0.08 - 0.02} = \frac{\$50\text{M}}{0.06} = \$833\text{M}$$

Now nudge WACC up by one point, to **9%**, changing nothing else:

$$\text{Value} = \frac{\$50\text{M}}{0.09 - 0.02} = \frac{\$50\text{M}}{0.07} = \$714\text{M}$$

The value fell from \$833M to \$714M — a drop of \$119M, or about **14%**, from a one-point change in a single input. And the effect is amplified by the growth rate sitting in the denominator: because we are dividing by (WACC − g), and g is close to WACC, a small change in WACC is a large *proportional* change in that small difference. Push g up toward WACC (a higher-growth company) and the sensitivity becomes violent — which is exactly why DCFs of high-growth companies are so contentious. The cash-flow forecast gets the attention; the discount rate does the damage.

*A DCF is only as trustworthy as its discount rate, and a model that reports a single precise valuation while hiding the WACC it depends on is hiding the most important number in the room.*

The practical response to this sensitivity is not despair but **discipline**: always present a DCF as a *range* across a band of plausible WACCs (and growth rates), not a single point. A sensitivity table — value at WACC of 7.5%, 8.0%, 8.5%, 9.0%, 9.5% — tells the reader far more honest a story than one number to four significant figures. The valuation is a cloud, and the WACC is the axis along which the cloud is most stretched.

## Common misconceptions

**"Book-value weights are fine — they're right there on the balance sheet."** No. WACC must use *market-value* weights, because it captures the return investors require on the capital they have at risk *today*, which is its market value, not its historical accounting cost. For a company whose stock trades well above book, book weights overweight debt (because they understate equity), pulling WACC artificially down. The balance sheet is the wrong place to read the weights; the stock market is the right one.

![A grid of six common cost-of-capital mistakes, each a way the discount rate goes wrong, paired with its fix, from book-value weights to using one WACC for divisions of different risk](/imgs/blogs/building-a-dcf-part-2-cost-of-capital-wacc-capm-7.png)

The figure collects the six recurring errors and their fixes. Beyond book weights, the most common are these.

**"Beta is just whatever the data terminal shows."** A single stock's raw five-year beta is noisy and backward-looking, and it bakes in that company's specific leverage. Professionals estimate beta from a peer group, unlever each peer to strip out its debt, average the asset betas, and relever to the target's own structure. A discount rate built on one unadjusted regression beta is built on sand.

**"Use a short, truly-riskless rate for the risk-free rate."** The risk-free rate should match the long horizon of the cash flows — a 10-year (or longer) government yield, not a 3-month bill. Using a short rate imports today's monetary conditions into a multi-decade valuation. And the risk-free rate and the ERP must be *consistent*: if you use a long government yield as rf, your ERP should be the premium over *that* long yield, not over short bills. Mixing a short rf with a long-horizon ERP double-counts (or omits) part of the term premium.

**"Forget the tax shield — debt costs what the company pays."** The cost of debt in WACC is the *after-tax* cost, Kd × (1 − t), because interest is tax-deductible. Discounting at the pre-tax cost overstates WACC and understates value. (The exception: a company that pays no tax gets no shield, so the (1 − t) reduction does not apply.)

**"One WACC fits the whole company."** A conglomerate with a stable utility division and a volatile software division should *not* discount both at one blended WACC. The utility's safe cash flows deserve a low rate; the software unit's risky cash flows deserve a high one. Using a single corporate WACC over-values the risky division (discounting its risky cash at too low a rate) and under-values the safe one. Divisions of different risk need different discount rates — ideally a *divisional WACC* built from the risk of that business, often using comparable pure-play companies in each division's industry.

**"Add a couple of points to WACC to be conservative."** Padding the discount rate "to be safe" is double-counting risk and a recipe for self-deception. Risk belongs in the cash-flow forecast (model the downside scenarios explicitly, probability-weight the outcomes) *or* in the discount rate (via beta and the risk premium) — not in both. Arbitrarily inflating WACC by 2% "for conservatism" on top of a risk-adjusted beta penalizes the same risk twice, and it hides a real judgement (how bad could the cash flows be?) behind a vague fudge factor. Put the conservatism where it can be examined: in the numbers, not in a mystery markup on the rate.

## How it shows up in real markets

**The duration trap in growth-stock valuations.** When central banks raised interest rates sharply in 2022, the hardest-hit stocks were not the ones with weak businesses today — they were the high-growth technology companies whose value lay in cash flows many years out. The mechanism is exactly the convexity we saw: long-duration cash flows are the most sensitive to the discount rate, and as the risk-free rate (the floor under every WACC) jumped, those distant cash flows got discounted far more heavily, and the valuations collapsed. A company expected to be hugely profitable in 2032 is, in present-value terms, mostly a bet on the discount rate — and when the rate moved, the bet repriced violently. The 2022 growth-stock drawdown was, in large part, a lesson in WACC sensitivity delivered by the bond market.

**Why utilities and tech trade at such different multiples.** A regulated water utility and a fast-growing software firm can have similar accounting profits and yet trade at wildly different price-to-earnings multiples. Part of the answer is growth, but part is the discount rate. The utility's cash flows are stable, contractual, and low-beta — investors require a low return, so a low WACC, which (for the same cash flows) supports a higher value. The software firm's cash flows are uncertain and high-beta — a high required return, a high WACC, and so a lower value per dollar of *current* cash (offset, in its case, by expected growth). Differences in the cost of capital, rooted in differences in business risk, are silently embedded in every valuation multiple you see in the market.

**Warren Buffett's hurdle and the discount-rate debate.** Buffett has long been skeptical of mechanically applying CAPM and beta, famously arguing that *beta measures volatility, not risk*, and that a stock that has fallen is often *less* risky, not more, even though its recent volatility (and thus beta) may have risen. His approach leans on a conservative required return and a margin of safety rather than a precise CAPM-derived WACC. The debate is instructive: it does not deny that a discount rate is needed — it questions whether beta is the right way to set it. The intrinsic-value, owner-minded approach is developed in [Buffett, Berkshire, and value investing](/blog/trading/finance/warren-buffett-berkshire-value-investing). The honest reading is that CAPM gives you a disciplined starting estimate, but judgement about the *actual* riskiness of a specific business — its competitive durability, its cyclicality, its balance-sheet fragility — should temper the mechanical output.

**The danger of a "house" discount rate.** Many companies set a single corporate hurdle rate and apply it to every project, regardless of risk. This is the "one WACC for all divisions" mistake institutionalized — and it has a predictable, value-destroying consequence. Safe projects (which deserve a low rate) look unattractive against the high house rate and get rejected; risky projects (which deserve a high rate) clear the same house rate too easily and get funded. Over time, a company using one rate for everything systematically rejects its safe value-creating projects and accepts its risky value-destroying ones, drifting toward a riskier, worse portfolio of investments. The fix — risk-adjusted, project-level or divisional hurdle rates — is conceptually simple and organizationally hard, which is why the mistake is so common.

## When this matters and further reading

The discount rate is the part of a DCF most worth getting right and most often gotten wrong. It rewards effort precisely because it is *small and bounded* — a handful of inputs you can reason about — while having outsized leverage on the answer. Spend your skepticism here: interrogate the risk-free rate's horizon, the beta's leverage adjustment, the ERP's source, the weights' market basis, the tax shield's applicability, and whether one rate really fits the whole business. Then present the valuation as a range across plausible rates, never a single false-precision point.

This post sits in the middle of the DCF arc. It builds on [Part 1: forecasting the cash flows](/blog/trading/equity-research/building-a-dcf-part-1-forecasting), which produces the numerator we discount, and on [free cash flow: FCFF vs FCFE](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe), which determines *which* cash flows we are discounting and therefore *which* rate — WACC for firm cash flows, cost of equity for equity cash flows — applies. The WACC we built here is the line against which [returns on capital: ROIC, ROE, ROA](/blog/trading/equity-research/returns-on-capital-roic-roe-roa) are judged, the comparison that defines value creation. And it feeds forward into [terminal value: the part that dominates](/blog/trading/equity-research/terminal-value-the-part-that-dominates), where the same discount-rate sensitivity reappears in even more concentrated form — because the terminal value, by construction, sits furthest in the future and is therefore most exposed to the rate. The discount-rate discipline you build here is the discipline that makes the rest of the DCF trustworthy. For the broader role of the hurdle rate in corporate decisions, see [the cost of capital and the hurdle rate](/blog/trading/equity-research/cost-of-capital-and-the-hurdle-rate).
