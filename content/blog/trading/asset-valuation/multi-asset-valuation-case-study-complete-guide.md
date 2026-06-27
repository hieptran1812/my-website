---
title: "From First Principles to Fair Value: A Complete Multi-Asset Valuation Case Study"
date: "2026-06-27"
publishDate: "2026-06-27"
description: "One analyst values five different assets in one sitting — a tech giant, a bank, an option, a Treasury bond, and Bitcoin — and shows how every method is the same act of discounting risky future cash flows."
tags: ["valuation", "asset-pricing", "dcf", "options", "bonds", "crypto", "discount-rate", "case-study", "terminal-value", "capstone"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Valuing five wildly different assets in one sitting reveals that every method is the same act: estimate future cash flows, then discount them at a rate that pays you for the risk.
>
> - Apple gets a discounted-cash-flow model cross-checked by P/E and EV/EBITDA; a bank gets price-to-book and a dividend model; an option gets Black-Scholes; a bond gets yield and duration; Bitcoin gets scenario weighting.
> - The discount rate is the spine that connects all five — the Treasury yield feeds Apple's WACC, the bank's required return, the option's risk-free rate, and the rate you use to discount a crypto bet.
> - Risk is measured differently for each (beta, volatility, country-risk premium, liquidity), but it always answers one question: how much extra return do I demand for the uncertainty?
> - The single number to remember: 59% to 95% of a long-horizon model's value lives in the terminal value — so the long-run growth assumption, the most uncertain input, is also the most important.

There is a quiet skill that separates a real valuation craftsperson from someone who merely knows the formulas: the ability to move between asset classes without mentally switching toolkits, because they have internalized that there is only one toolkit. This post is an attempt to install that skill by demonstration rather than assertion. We will do the actual arithmetic on five very different assets, with real numbers, and the punchline will emerge from the work itself.

A working analyst rarely values one thing in isolation. On a normal Tuesday they might glance at a tech stock before the open, check whether a bank looks cheap on book value, price a hedge with a put option, watch the 10-year Treasury yield because it moves everything else, and field a question about whether Bitcoin is expensive. Five assets, five asking prices, five completely different-looking spreadsheets.

The temptation is to treat these as five unrelated crafts — as if pricing an option had nothing to do with pricing a bank. That instinct is wrong, and unlearning it is the single biggest jump in valuation skill. Underneath the different formulas sits one engine: you forecast cash a thing will produce, you decide how risky that cash is, and you discount it back to today at a rate that compensates for the risk. The bond makes this obvious because its cash flows are contractual. Bitcoin hides it because its cash flows are zero and you have to value the *belief* about future cash flows instead. Everything in between is a variation on the same theme.

Why value five assets in one sitting instead of studying them one at a time? Because the connective tissue only becomes visible when the methods sit side by side. Study a DCF alone and the terminal value looks like a technical footnote; line it up against a bank's dividend model and a Bitcoin scenario and you see that *all three* are dominated by a long-run assumption, and that the dominance is the real lesson, not the arithmetic. Study an option in isolation and the risk-free rate inside it looks like a convention; line it up against the Treasury that *sets* that rate and you see the wiring. Five isolated lessons teach five methods. Five assets valued together teach the one principle the methods share — and that principle, not the methods, is what survives into every new asset you will ever have to price.

This is the capstone of the series. Rather than introduce a new method, it puts the whole toolkit to work on one long case study and shows the connective tissue. We will value Apple, Vietcombank, an S&P 500 put, a U.S. 10-year Treasury, and Bitcoin — with real 2024-era numbers — and at every step point back to the one idea doing the heavy lifting.

![Five assets and their valuation methods on one discount-rate spine](/imgs/blogs/multi-asset-valuation-case-study-complete-guide-1.png)

## Foundations: the three things every valuation shares

Before any asset, fix the vocabulary. Every valuation — without exception — is built from three ingredients. If you can name them for an asset, you can value it; if you cannot, you are guessing.

**Cash flows.** The money the asset is expected to put in your pocket over time: dividends, coupons, free cash flow, a payoff at expiry, a sale price. This is the *numerator*. A stock's cash flow is the free cash flow the business throws off; a bond's is its coupons and the return of principal; an option's is whatever it is worth when it expires; Bitcoin's is — uncomfortably — nothing contractual at all, which is exactly why it is the hardest to value.

**A discount rate.** Money in the future is worth less than money today, because you could have invested today's money and because the future is uncertain. The discount rate converts a future dollar into a today-dollar. It is the *denominator*. This idea — the engine of every model — is laid out in [the time value of money](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model). The higher the rate, the more future cash is shrunk, and the lower the value today.

**Risk.** The reason discount rates differ across assets. A government bond's coupons are nearly certain, so you discount them at a low rate. A young company's cash flows are a guess, so you demand a higher rate to compensate. Risk is the *bridge* between cash flows and the discount rate: it is what turns a generic interest rate into the specific rate this asset deserves. How risk maps to a required return is the job of [the CAPM and beta](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital).

The single most useful sentence in valuation: **value is future cash flows discounted at a rate that compensates for their risk.** Memorize it. Every formula below is that sentence wearing a different costume. The Gordon growth formula for a perpetuity, the Black-Scholes equation for an option, the present-value sum for a bond, the price-to-book ratio for a bank — each is an algebraic rearrangement of that one sentence, specialized to the shape of a particular asset's cash flows. When a new asset class appears that you have never seen, you do not need a new theory; you need to answer the three questions and pick the rearrangement that fits the cash-flow pattern. That is why a single principle can carry you across stocks, bonds, options, and crypto without ever leaving the same idea.

One more building block: the **risk-free rate**, the return on lending to a government that will not default, usually proxied by the yield on a Treasury bond. It is the floor under every other discount rate. When the 10-year Treasury yields 4.3%, no rational investor accepts less than 4.3% for anything riskier — so 4.3% becomes the base on top of which every risk premium is stacked. That is why the bond is not just one of the five assets; it is the keystone that holds up the other four. We will return to this repeatedly.

It helps to see how the three ingredients change shape across asset classes before we touch any of them individually. For a bond, the cash flows are written into the contract, the risk is mostly the chance rates move against you, and the discount rate is simply the market yield — there is almost nothing to estimate, which is why bonds are the reference case. For a stock, the cash flows must be *forecast* from the business, the risk is how much those forecasts could be wrong, and the discount rate is a built-up cost of capital. For an option, the "cash flow" is a conditional payoff that depends on where the underlying ends up, the risk is the volatility of that underlying, and the discount rate is the risk-free rate applied inside a probability-weighted formula. For Bitcoin, the cash flow is zero, the risk is existential, and the discount rate is overwhelmed by an uncertainty premium so large that the model degenerates into a weighted guess. Same three ingredients, increasing difficulty — and the increasing difficulty is *exactly* the increasing uncertainty in the cash flows.

This ordering — bond, bank, option, stock, crypto, ranked by how knowable the cash flows are — is the spine of the case study. We will not march in that exact order, but keep the ranking in mind: the more you have to *guess* about the future to value something, the wider your honest answer will be, and the smaller your position should be. The discipline is always to estimate and bound, never to invent.

With the vocabulary fixed, start with the asset most people think they already understand.

## Asset 1 — Apple: a mature business valued three ways

Apple in its 2024 fiscal year did roughly \$391B in revenue and about \$94B in net income, on roughly 15.4B diluted shares, for earnings per share near \$6.08. At a share price around \$182 that is a price-to-earnings ratio of about 30x. The job is not to recite those numbers; it is to decide whether \$182 is a fair price, and the discipline is to attack it from more than one direction.

**The first lens: the P/E multiple.** A price-to-earnings ratio of 30x says you are paying \$30 for every \$1 of current annual earnings. On its own that number is meaningless — it is only cheap or expensive relative to how fast those earnings grow and how certain they are. The full mechanics of reading a P/E live in [the P/E ratio guide](/blog/trading/asset-valuation/price-to-earnings-ratio-pe-valuation-stocks); the short version is that a 30x multiple on a company growing high-single-digits is a *premium*, justified only if the growth and the durability are real. Apple earns it through enormous returns on capital and a buyback program that steadily shrinks the share count, which lifts EPS even when net income is flat.

**The second lens: EV/EBITDA.** P/E is distorted by capital structure and by buybacks, so professionals cross-check with enterprise value over EBITDA. Enterprise value — roughly market capitalization plus debt minus cash — was about \$2.8T, and EBITDA (earnings before interest, taxes, depreciation, and amortization) about \$125B, for an EV/EBITDA near 22x. This lens ignores how the company is financed and asks what the *whole business* costs relative to its operating cash generation. The two multiples agreeing — both saying "premium but not absurd" — is more convincing than either alone.

Why bother with two multiples plus a DCF? Because each has a blind spot the others cover. P/E breaks for companies with lots of debt or distorted earnings; EV/EBITDA fixes the capital-structure problem but ignores the real cash drains of capital spending and taxes; the DCF captures everything but depends on long-range forecasts that no one can verify. Triangulation is the discipline: when a multiple-based read and a cash-flow-based read land in the same neighborhood, your confidence is earned, and when they diverge sharply, the divergence itself is the finding — it points to exactly which assumption is doing the work. A \$133 DCF against a 30x P/E and a 22x EV/EBITDA tells a coherent story: the market is paying a growth premium the conservative DCF does not grant, and now you can argue about that premium specifically rather than about the price in the abstract.

**The third lens: a discounted-cash-flow model.** Multiples compare Apple to other companies; a DCF values Apple on its own merits by projecting its cash and discounting it. Apple's free cash flow runs near \$100B a year. Assume it grows about 5% annually for ten years, then settles into a 3% perpetual growth rate, and discount everything at a weighted-average cost of capital (WACC) of about 9%. The mechanics of building this model are in [the DCF walkthrough](/blog/trading/asset-valuation/discounted-cash-flow-dcf-equity-valuation-step-by-step), and the recipe for the 9% rate itself is in [discount rates in practice](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta).

![Apple DCF present value of forecast cash flows versus terminal value](/imgs/blogs/multi-asset-valuation-case-study-complete-guide-2.png)

#### Worked example: Apple's DCF, terminal value, and per-share fair value

Project free cash flow from a \$100B base, growing 5% a year:

```
Year 1 FCF  = 100 x 1.05        = 105.0
Year 10 FCF = 100 x 1.05^10     = 162.9   (US$ billions)
```

Discount each year at 9% and sum the ten present values:

```
Sum of PV(FCF, years 1-10) at 9% WACC = $819B
```

Now the terminal value — what the business is worth at the end of year 10, growing forever at 3%, using the Gordon growth formula:

```
TV(year 10) = FCF_10 x (1 + g) / (WACC - g)
            = 162.9 x 1.03 / (0.09 - 0.03)
            = 167.8 / 0.06
            = $2,796B
```

That \$2,796B is a year-10 number; discount it back ten years at 9%:

```
PV(terminal value) = 2,796 / 1.09^10 = $1,181B
```

Add the two pieces for enterprise value, add roughly \$50B of net cash, and divide by 15.4B shares:

```
Enterprise value = 819 + 1,181            = $2,000B
Equity value     = 2,000 + 50 net cash    = $2,050B
Per-share value  = 2,050,000 / 15,400     ≈ $133
```

A \$133 fair value against a \$182 market price says Apple, on these conservative assumptions, is priced for *more* than 5% cash-flow growth — the market is paying for either faster growth or a lower discount rate than we assumed. That is not a sell signal; it is the model telling you precisely which assumption you would have to believe to justify the price.

This is where a DCF earns its keep — not as a single number, but as a sensitivity machine. Reverse the question: what growth rate justifies \$182? Hold the 9% WACC fixed and solve for the cash-flow growth that lifts the per-share value to \$182, and you find the market is implying something closer to 7-8% perpetual-ish growth rather than our 5%. Now the judgment is concrete and falsifiable: do you believe Apple can compound free cash flow at 7-8% for a decade, given its size? If yes, \$182 is fair; if no, it is rich. A good analyst does not deliver "\$133, sell"; they deliver "the price requires 7-8% growth — here is why I do or do not believe that." The number is the start of the argument, not the end of it.

The discount rate itself deserves a sentence of scrutiny, because it is the input people wave at most casually. Apple's 9% WACC is built up, not assumed: start from the risk-free rate (the 4.3% Treasury yield), add an equity risk premium — the extra return stocks pay over bonds, historically 4-5% — scaled by Apple's beta of roughly 1.2, which gives a cost of equity around 9-10%. Apple carries little net debt, so the weighted-average barely moves off the cost of equity, landing near 9%. Every one of those pieces traces back to the Treasury yield, which is the thread we keep pulling. Shift the risk-free rate up a point and the WACC rises toward 10%, and — as the sensitivity below shows — that one point can knock 15-20% off the per-share value, because it both shrinks every discounted cash flow *and* shrinks the terminal value that dominates the total.

Notice the most important fact in that example: of the \$2,000B enterprise value, \$1,181B — about 59% — is the terminal value. More than half of Apple's worth is a guess about the world after year 10. Hold that thought; it returns as the thread that ties all five assets together.

A word on buybacks, because they confuse beginners. When Apple buys back shares, total enterprise value does not change — the business is the same. But the same value is now split across fewer shares, so value *per share* rises. That is why Apple can grow EPS faster than net income, and why a per-share DCF must use a shrinking share count to be honest.

## Asset 2 — Vietcombank: why a bank refuses the stock playbook

Try to value a bank with a DCF and you will tie yourself in knots, because for a bank, debt is not financing — it is raw material. A bank's deposits are a liability on its balance sheet, yet they are the very thing it lends out to make money. "Free cash flow to the firm" is almost meaningless when the firm's product *is* moving cash around. So banks get valued on the balance sheet itself, through price-to-book.

Vietcombank (VCB) is Vietnam's largest bank by market value. Take a return on equity (ROE) of about 22%, a book value per share around 55,000 VND, a required return on equity of 15.4%, and a sustainable growth rate of 8%. The required return is high because this is an emerging market: on top of the global risk-free rate sits a country-risk premium for Vietnam, the logic of which is spelled out in [emerging-market valuation](/blog/trading/asset-valuation/emerging-market-stock-valuation-country-risk-discount-rate). The full bank case study is in [valuing Vietcombank](/blog/trading/asset-valuation/valuing-vietcombank-vcb-pb-ddm-case-study); here we use it to make one point.

The justified price-to-book ratio for a bank follows from a clean formula: a bank should trade above book if and only if it earns more on its equity than shareholders demand. Concretely, justified P/B equals (ROE − g) divided by (required return − g). The mechanics of reading a P/B sit in [the P/B ratio guide](/blog/trading/asset-valuation/price-to-book-ratio-pb-valuation-equity).

#### Worked example: VCB's justified price-to-book and price target

```
Justified P/B = (ROE - g) / (r - g)
              = (0.22 - 0.08) / (0.154 - 0.08)
              = 0.14 / 0.074
              = 1.89x
```

Multiply by book value per share:

```
Price target = 1.89 x 55,000 VND ≈ 104,000 VND  (about $4.16)
```

The reason VCB deserves to trade at nearly twice book value is captured in a single comparison: it earns 22% on equity while shareholders only demand 15.4% — that 6.6-point gap, compounding on a growing book, is exactly what a premium to book value pays for.

There is a second route to the same answer that makes the cash-flow connection explicit: the dividend-discount model (DDM). A bank's shareholders ultimately receive dividends, and a stable bank can pay out the portion of earnings it does not need to retain to fund growth. With a 22% ROE and 8% sustainable growth, the bank must retain about 8/22 — roughly 36% — of earnings to fund that growth, leaving a 64% payout ratio. Run those dividends through the Gordon growth formula (next dividend divided by required return minus growth) and you arrive at a value close to the 1.89x book the P/B route produced. The two methods agree because they are the same statement: a bank is worth more than its book when it earns more on that book than shareholders demand. P/B is just the DDM compressed into a single ratio for banks.

Stand back and see the symmetry with Apple. Apple uses P/E because its earnings are the product; VCB uses P/B because its equity capital is the product. Different ratios, identical logic — both ask whether the price you pay is justified by the returns the asset generates relative to the return you demand. The required return in both cases is the discount rate wearing the costume of "cost of equity." [What value even means](/blog/trading/asset-valuation/what-is-value-philosophy-frameworks-asset-pricing) is the same across both; only the plumbing differs.

The country-risk premium deserves its own beat, because it is the single biggest reason an emerging-market bank trades differently from a developed-market one. VCB's 15.4% required return is not 15.4% because the bank is reckless — it is well-run — but because Vietnamese assets carry sovereign, currency, and liquidity risks that a U.S. investor does not face when buying a U.S. bank. The premium is layered on top of the global risk-free rate: a U.S.-investor's required return might start from the 4.3% Treasury, add an equity risk premium, then add several points of country-risk premium for Vietnam. That stacked rate is why the *same* fundamentals — a 22% ROE — justify a lower price-to-book in Vietnam than they would in a market with a lower required return. Lower the discount rate by improving country risk (a credit-rating upgrade, a more stable currency) and the justified P/B rises mechanically, with no change in the bank's actual operations. The asset got more valuable because the *denominator* shrank, which is the cleanest demonstration that the discount rate, not just the cash flows, drives value.

## Asset 3 — An S&P 500 put option: valuing a contingent claim

The first two assets pay you for owning a business. An option is different in kind: it is a *contingent claim*, a contract that pays only if a condition is met. A put option on the S&P 500 ETF (SPY) gives you the right to sell SPY at a fixed strike price. It is worth something only if SPY falls below that strike — which makes its cash flow conditional, and so its valuation needs a different machine: the Black-Scholes-Merton (BSM) model.

Price the put on these inputs: SPY trading at 500, a strike of 475 (5% out of the money), a risk-free rate of 5.3%, implied volatility of 18%, and six months to expiry (T = 0.5 years). The full derivation lives in [the Black-Scholes guide](/blog/trading/asset-valuation/black-scholes-model-greeks-options-valuation); the engine, though, is still discounting. BSM computes the probability-weighted payoff of the option and discounts it back at the risk-free rate. The two terms d1 and d2 are just the standardized distances of the current price from the strike, accounting for drift and volatility.

An option's value splits cleanly into two pieces, and naming them makes everything else legible. **Intrinsic value** is what the option would be worth if it expired right now — for this put, with SPY at 500 above the 475 strike, intrinsic value is zero, because you would not exercise the right to sell at 475 when you can sell at 500 in the market. **Time value** is everything above intrinsic: the worth of the remaining six months during which SPY could still fall below 475. The entire \$10.13 price of this put is time value, the premium you pay for optionality you do not yet have but might. As expiry approaches, time value decays toward zero — the option converges to its intrinsic value — which is the source of the famous "theta bleed" that option buyers fight and option sellers harvest. The chart below shows this split directly: the dashed payoff line is intrinsic value, and the gap between it and the smooth BSM curve is time value.

![SPY put option payoff at expiry versus current Black-Scholes value](/imgs/blogs/multi-asset-valuation-case-study-complete-guide-4.png)

#### Worked example: Black-Scholes value and the Greeks of the SPY put

```
d1 = [ ln(S/K) + (r + sigma^2 / 2) x T ] / (sigma x sqrt(T))
   = [ ln(500/475) + (0.053 + 0.18^2 / 2) x 0.5 ] / (0.18 x sqrt(0.5))
   = 0.6748
d2 = d1 - sigma x sqrt(T) = 0.6748 - 0.1273 = 0.5476

Put value = K x e^(-rT) x N(-d2) - S x N(-d1)
          = 475 x e^(-0.053 x 0.5) x N(-0.5476) - 500 x N(-0.6748)
          ≈ $10.13 per share
```

The put costs about \$10.13 per share even though it is out of the money and would pay nothing if SPY expired today — that \$10.13 is pure time value, the price of six months of insurance against a fall below 475.

Now read the Greeks as risk measures, not as Greek letters. **Delta** for this put is about −0.25: the put loses roughly \$0.25 of value for every \$1 SPY rises, and gains \$0.25 for every \$1 it falls. **Gamma** tells you how fast that delta itself changes as SPY moves — it is the curvature, and it is largest near the strike. **Vega** measures sensitivity to volatility: because an option's value rises with uncertainty, a jump in implied volatility lifts the put even if SPY does not move. The Greeks are simply the partial derivatives of the same value formula, each isolating one source of risk. They are how an option trader decomposes "what could change my P&L" into named, hedgeable pieces.

There is a subtlety worth pausing on, because it is where Black-Scholes departs from the stock and bond models in a way that confuses newcomers. BSM does not discount the option's expected payoff at a *risk-adjusted* rate the way a DCF discounts a stock at WACC. It discounts at the *risk-free* rate, and it gets away with this because of a deep result: an option can be replicated by a continuously rebalanced portfolio of the underlying and cash, so its value cannot depend on anyone's risk preferences — only on the cost of building that replicating portfolio. The probabilities inside N(d1) and N(d2) are therefore not real-world probabilities; they are risk-neutral probabilities, a mathematical device that lets the risk-free rate do the discounting. You do not need to derive this to use the model, but you should know *why* the risk-free rate, not WACC, sits inside it — otherwise the formula looks like it is ignoring risk, when in fact it has handled risk by replication rather than by a premium.

The practical reading of the Greeks is what makes an option a *managed* position rather than a bet. A trader holding this put knows that a delta of −0.25 means the position behaves like being short 25 shares of SPY for small moves — so to be market-neutral they would buy 25 SPY-equivalent shares against each put. Gamma tells them that hedge will drift as SPY moves, requiring re-hedging; the closer SPY gets to the 475 strike, the faster delta changes and the more often they must trade. Vega warns that even a frozen SPY can move their P&L if implied volatility shifts — a put bought at 18% volatility and marked at 22% gains value with the underlying unchanged. Each Greek isolates one axis of risk so it can be hedged or sized independently. That decomposition is the entire reason options can be used as precision instruments rather than blunt directional wagers.

The connection back to the bond is direct: that 5.3% risk-free rate inside the BSM formula is the *same* Treasury-anchored rate that discounted Apple's cash flows. Change the risk-free rate and the option reprices — the put's value falls slightly as rates rise, because the present value of the strike you would receive on exercise shrinks. It is a small effect for a six-month put, but it is the same chain reaction we will draw at the end, reaching even into a contract whose payoff has nothing obvious to do with interest rates.

## Asset 4 — A U.S. 10-year Treasury: the cleanest valuation of all

The bond is the easiest asset to value and the most important, because its discount rate *is* the price. A 10-year Treasury with a 4.3% coupon and a \$1,000 face value pays \$43 a year (\$21.50 every six months) for ten years and returns the \$1,000 at maturity. Its value is just those cash flows discounted at the market yield. When the yield equals the coupon — 4.3% — the bond is worth exactly its \$1,000 face value, "at par." That is the cleanest possible illustration of the core sentence: the value is the contractual cash flows discounted at a rate set by the market. The full treatment is in [bond valuation, yield, duration, and convexity](/blog/trading/asset-valuation/bond-valuation-yield-duration-convexity).

The term *yield to maturity* (YTM) is worth nailing down because it is the bond's whole story in one number. The YTM is the single discount rate that makes the present value of all the bond's future cash flows equal to its current market price. It is, in other words, the internal rate of return you earn if you buy the bond today and hold it to maturity, reinvesting coupons at that same rate. This is the bond-market mirror of the equity DCF: where a stock analyst assumes a discount rate and solves for value, a bond is *quoted* by its value (the price) and you solve for the discount rate (the yield). The two are the same equation read in opposite directions. When commentators say "the 10-year yield rose," they are saying bond prices fell — the discount rate went up, so the present value of those fixed coupons went down. Every other asset in this article inherits that rate as its starting point, which is why a single number on a bond screen moves the entire market.

The interesting part is what happens when the yield moves. Bond price and yield move in opposite directions — when newly issued bonds offer 5.3%, your 4.3% bond is less attractive, so its price must fall until its *yield to maturity* matches the market's 5.3%.

![Ten-year Treasury price falling as yield rises along a convex curve](/imgs/blogs/multi-asset-valuation-case-study-complete-guide-3.png)

#### Worked example: the 10-year Treasury repriced when yields rise 1%

First the quick, linear estimate using duration. The modified duration of this bond is about 8.06 years, which means a 1-percentage-point rise in yield should cut the price by roughly 8.06%:

```
Duration estimate of price change
= -modified duration x change in yield x price
= -8.06 x 0.01 x $1,000
≈ -$81
Estimated new price ≈ $919
```

Now the exact calculation, discounting all twenty semiannual cash flows at the new 5.3%:

```
Exact new price at 5.3% yield = $923.15
Actual price change = -$76.85  (-7.7%)
```

The duration estimate (−\$81) overshoots the true fall (−\$77) — and that gap is **convexity**: because the price-yield curve bends, the linear duration line always predicts a slightly bigger loss than actually occurs when yields rise, and the curvature works in the bondholder's favor.

Duration deserves a plain definition because it is the bridge between a bond and every other asset. Duration is the *interest-rate sensitivity* of a price — roughly, how many percent the price moves for a one-point move in yield. A 10-year coupon bond has a duration near 8 years (less than 10 because the coupons return cash before maturity); a 30-year zero-coupon bond has a duration of 30, which is why it is the most rate-sensitive thing in the bond market. The same concept reaches into equities: a high-growth stock whose value is mostly terminal value is "long duration" in exactly this sense — its cash flows arrive far in the future, so its price is highly sensitive to the discount rate. That is why growth stocks and long bonds sell off together when rates rise; they share a duration, even though one is a contract and the other is a guess about a business.

Convexity, the curvature we just measured, is not a footnote — it is an asset in its own right. Because the price-yield curve bends toward the bondholder, a bond *gains* more when yields fall than it loses when yields rise by the same amount. A long-duration, high-convexity bond is therefore a slightly asymmetric bet in the holder's favor, which is part of why investors will accept a marginally lower yield for more convexity. The same asymmetry, in option language, is gamma — and seeing that the bondholder's convexity and the option-holder's gamma are the same mathematical curvature is one of those connections that collapses two apparently separate topics into one.

Here is why this asset is the keystone. The 4.3% yield is not just *this* bond's discount rate — it is the risk-free rate that feeds into every other valuation in this article. It is the base of Apple's 9% WACC, the floor under VCB's 15.4% required return, the 5.3% inside the option's BSM formula (a different point on the same yield curve), and the rate against which Bitcoin's no-yield gamble is measured. The relationship between this single rate and every risk asset is the subject of [how interest rates link bonds and stocks](/blog/trading/macro-trading/interest-rates-bonds-stocks-relationship), and the rate itself is set by the process in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

## Asset 5 — Bitcoin: valuing a thing with no cash flows

Bitcoin breaks the engine, and that is what makes it instructive. It pays no dividend, no coupon, no free cash flow. There is no numerator to discount. So the honest answer is that Bitcoin cannot be valued the way the first four assets can — and pretending otherwise, with false precision, is the most common mistake in crypto analysis. The mistake is seductive because the tools *look* like they should transfer: people build elaborate models with discount rates and growth curves, and the elaborateness disguises the fact that there is no cash flow at the bottom of any of it. A model with no cash-flow anchor is not a valuation; it is a dressed-up opinion about adoption. Recognizing that is not a defeat — it is the correct diagnosis, and it changes the right tool from a spreadsheet to a probability-weighted scenario tree. What you can do is build *bounded scenarios* and weight them, which is itself a valuation method — just one that wears its uncertainty on its sleeve. The fuller toolkit is in [crypto valuation](/blog/trading/asset-valuation/crypto-valuation-bitcoin-ethereum-token-pricing).

The proxies people use are revealing precisely because of how they fail. The **NVT ratio** (network value to transactions) divides Bitcoin's market value by the dollar value of on-chain transactions — a P/E-style multiple where "transactions" stand in for "earnings." A high NVT says the network is valued richly relative to its actual economic throughput, the way a high P/E says a stock is valued richly relative to its earnings. The problem is that there is no anchor for what NVT *should* be. A stock's P/E can be tied to a growth rate and a discount rate; Bitcoin's NVT floats free, because there is no cash-flow theory underneath it. It is a *relative* gauge — useful for spotting when Bitcoin is expensive versus its own history — but it cannot output an absolute fair value.

#### Worked example: reading Bitcoin's NVT ratio as a relative gauge

```
Network value (market cap)        = $60,000 x ~19.7M coins ≈ $1,180B
Daily on-chain transaction value  ≈ $8B  (illustrative)
Annualized transaction value      = 8 x 365            = $2,920B
NVT ratio = network value / annual transactions
          = 1,180 / 2,920                                ≈ 0.40 (or ~147 on daily basis)
```

An NVT well above its own multi-year average flags the network as richly valued relative to the economic activity flowing across it — but because nothing pins the "right" level, this number sizes a *lean*, never a fair value, which is the honest limit of every cash-flow-free metric.

**Metcalfe's law** argues a network's value scales with the square of its users, giving a rough adoption-based anchor. Neither metric pins down a price; both bracket a range. So the practical method is to define a small set of futures, assign probabilities, and take the weighted average.

![Bitcoin scenario fan from 2024 to 2030 with probability-weighted path](/imgs/blogs/multi-asset-valuation-case-study-complete-guide-5.png)

#### Worked example: Bitcoin's probability-weighted fair value

From a spot price near \$60,000, define three scenarios for the end of the horizon and weight them by how likely each looks:

```
Bear case:  $20,000  x  25% probability  =  $5,000
Base case:  $70,000  x  50% probability  = $35,000
Bull case: $150,000  x  25% probability  = $37,500
                                  --------
Probability-weighted value        = $77,500
```

The weighted fair value of \$77,500 sits above the \$60,000 spot, but the honest takeaway is the *spread*, not the point estimate — a fair value that swings from \$20,000 to \$150,000 depending on which future you believe is telling you the asset is barely valuable at all in any precise sense, and that the position should be sized accordingly.

That last sentence is the whole lesson of Bitcoin. The width of the fair-value range is not a flaw in the analysis; it is information. The hardest asset to value is also the one with the widest reasonable range — and a rational investor responds to a wide range with a *small* position, because conviction should scale with how narrowly you can bound the answer. Position sizing, properly understood, is just valuation uncertainty expressed in dollars.

Contrast the ranges to feel the point. The Treasury's fair value is a single number to the penny — its cash flows are contractual and its only uncertainty is the discount rate, which the market quotes for you. Apple's fair value spans maybe \$120 to \$220 depending on the growth assumption — a roughly two-to-one range driven by the terminal value. VCB's spans perhaps 80,000 to 130,000 VND as the ROE-versus-cost-of-equity gap flexes. The option's value is tight given its inputs but those inputs (especially volatility) can shift fast. And Bitcoin spans seven-to-one. If you sized every position equally you would be taking wildly different amounts of *valuation risk* in each. The professional move is the opposite: let the position size shrink as the fair-value range widens, so that the dollars at risk reflect the confidence in the estimate. The Treasury can be a large, calm holding; Bitcoin should be a small one, not because it cannot go up, but because you cannot say with any precision what it is worth. That asymmetry between conviction and sizing is the bridge from valuation to portfolio construction.

## The connecting threads

Five assets, five spreadsheets, one engine. Now make the connections explicit.

**Thread one: the discount rate is the spine.** Trace it through every asset. The Treasury yield of 4.3% is the literal price of the bond. That same risk-free rate, plus an equity risk premium scaled by Apple's beta, builds Apple's 9% WACC. Add a Vietnam country-risk premium and you get VCB's 15.4% required return. A point on the same Treasury curve — 5.3% — is the risk-free rate inside the option's Black-Scholes formula. And the rate you would discount a Bitcoin payoff at starts from that same risk-free floor before piling on an enormous uncertainty premium. One rate, five disguises. That is why a bond desk and an equity desk are not really in separate businesses.

**Thread two: risk is measured differently but means the same thing.** Apple's risk shows up as beta — how much it swings with the market. The option's risk is volatility — the standard deviation of returns. VCB carries a country-risk premium for Vietnam. Bitcoin carries a liquidity-and-adoption premium so large it dominates everything else. The bond carries almost none, which is why it sets the floor. These look like four unrelated quantities, but each answers the identical question: *how much extra return do I demand for bearing this uncertainty?* Risk is never valued for its own sake; it is valued only as the reason a discount rate is higher than the risk-free rate.

It is worth lining up the risk premia as a stack to see the architecture. Start at the bottom with the 4.3% risk-free rate — the price of patience with no default risk. On a U.S. equity, add an equity risk premium of 4-5% for bearing business and market risk, scaled by beta, landing Apple near 9%. For VCB, take that equity logic and add several more points of country-risk premium for sovereign and currency exposure, reaching 15.4%. For the option, the risk does not enter as a premium on the discount rate at all — it enters through volatility inside the formula, which is the option-specific way of pricing the same underlying uncertainty. For Bitcoin, the premium for adoption-and-survival risk is so vast that it swamps the base rate entirely, which is why no single discount rate can tame it and the scenario method takes over. Read top to bottom, the stack tells you a story: each asset is the risk-free rate plus a premium that compensates for a *specific, nameable* uncertainty, and the size of that premium is the market's price for that uncertainty. Mispricing, when it happens, is almost always a mispriced premium — the market demanding too little extra return for a risk that later shows up.

**Thread three: the assets are not as independent as they look.** Because they all hang from the same discount-rate spine, a single shock — a rise in rates — moves all of them at once. This is why "diversification" across stocks, bonds, options, and crypto offers less protection than it appears: the correlation hiding inside them is the shared discount rate.

![Before and after a one-percent rate rise rippling through all five assets](/imgs/blogs/multi-asset-valuation-case-study-complete-guide-6.png)

The chain is mechanical. Rates rise 1%. The bond falls to \$923 directly. Apple's WACC rises from 9% toward 10%, which shrinks the present value of its cash flows and compresses its P/E. VCB's required return climbs above 15.4%, which lowers its justified price-to-book. The option's carry gets more expensive and risk appetite drops. And Bitcoin, the asset furthest out on the risk curve, tends to sell off hardest because a higher risk-free rate raises the bar that a no-yield asset must clear. Five assets, one cause.

## What changes each asset's value most

If you can only stress-test one input per asset, choose the one the value hinges on:

- **The bond** is most sensitive to *rates*. Its entire value is a rate calculation, and duration tells you exactly how much: 8.06 years of price change per point of yield.
- **Apple** is most sensitive to *earnings growth*. Shift the 5% growth assumption and the whole DCF moves, because growth compounds into both the explicit forecast and the terminal value.
- **VCB** is most sensitive to the *gap between ROE and cost of equity*. The justified P/B formula has that spread in the numerator; shrink it and the premium to book collapses.
- **The option** is most sensitive to *volatility*. Vega is the dominant Greek for a six-month option; a few points of implied volatility move the price more than a small move in SPY.
- **Bitcoin** is most sensitive to *the adoption scenario probabilities*. Nudge the bull-case weight and the whole fair value lurches, because there is no cash-flow anchor to hold it steady.

Knowing the dominant variable is what separates a model from a number. A model you can stress; a number you can only believe or disbelieve. The practical workflow follows directly: once you have a base-case value, flex the dominant variable across a plausible range and read off the resulting spread of values — that spread *is* your honest fair-value range, and the position you take should be sized to it. For the bond, flex the yield and you get a tight, confident range. For Bitcoin, flex the scenario weights and the range explodes. The discipline is to spend your research time where it moves the answer: another week studying Apple's next quarter barely shifts a model dominated by terminal growth, while an hour spent pressure-testing the long-run growth assumption changes everything. Effort should flow to the assumption with the most leverage on the value, not to the one that is easiest to look up.

## The common thread: terminal value dominates

Return to the fact buried in Apple's DCF: 59% of the value lived in the terminal value, the piece representing everything after year 10. This is not an Apple quirk — it is the rule for any long-horizon model.

![Terminal value as a share of total model value across four asset models](/imgs/blogs/multi-asset-valuation-case-study-complete-guide-7.png)

Look across models. In Apple's DCF the terminal value is about 59% of the total on conservative assumptions, and well over 70% on more generous growth. In VCB's dividend-discount model, the perpetual-growth tail is roughly 90% of the value. A mature utility's DCF runs near 85%. A high-growth software company can be 95% terminal value, because almost none of its worth is in the cash it generates *now*. Even Bitcoin's scenario value is entirely a long-run bet — 100% "terminal," in a sense, since none of its value comes from intervening cash flows.

The implication is sobering and clarifying at once. The input that matters most — the long-run growth rate, the perpetual ROE, the adoption odds — is also the input you can least defend. Most of every valuation rests on the part of the future you can see least clearly. This is why two honest analysts can value the same asset and disagree by a factor of two: they are not arguing about next year's earnings, which are roughly knowable; they are arguing about the terminal assumption, which is roughly *un*knowable. The discipline is not to eliminate that uncertainty — you cannot — but to name it, bound it, and size your conviction to how wide the bound is. That is the same lesson Bitcoin taught, generalized to every asset.

There is a practical defense against terminal-value tyranny, and good analysts use all of it. First, sanity-check the terminal value as an *exit multiple*: divide it by the final-year earnings or cash flow and ask whether that implied multiple is reasonable for a mature business — a terminal value that implies a 40x exit P/E on a slow-growing giant is a red flag, no matter how innocent the perpetual growth rate looked. Second, keep the perpetual growth rate below the long-run growth rate of the economy; a company cannot grow faster than the world forever, so a terminal growth rate above roughly 3-4% is quietly assuming the company eventually becomes the entire economy. Third, run the model at a range of terminal assumptions and report the *range*, not a point — exactly as we did for Bitcoin. The through-line of this whole article is that the right output of a valuation is rarely a single number; it is a range, plus an honest statement of which assumption the range hangs on. The bond is the only asset here that escapes this, and it escapes only because its cash flows are contractual rather than forecast.

## Common misconceptions

**"Different assets need fundamentally different valuation theories."** No. They need different *plumbing* for the same theory. Apple's DCF, VCB's P/B, the option's BSM, the bond's YTM, and Bitcoin's scenarios are five implementations of "discount risky future cash flows." The bond proves it cleanly because its cash flows are contractual; the others are the same act with more estimation.

**"A lower P/E or P/B always means cheaper."** No. A 30x P/E on Apple can be cheaper than a 10x P/E on a declining business, because the multiple is only meaningful relative to growth and risk. VCB's 1.89x book is justified precisely because its 22% ROE clears its 15.4% cost of equity — a bank earning *below* its cost of equity deserves to trade *below* book even though that looks "cheap."

**"The risk-free rate is just the bond market's problem."** No. It is the floor under every valuation in this article. When it moved 1%, all five assets repriced. An equity investor who ignores the 10-year yield is ignoring the denominator of their own model.

**"A precise fair value means a good valuation."** Often the opposite. Bitcoin's range of \$20,000 to \$150,000 is more honest than a single confident \$77,500. False precision hides the very uncertainty that should drive position sizing.

**"Buybacks create value out of nothing."** No. They redistribute the same enterprise value across fewer shares. Per-share value rises, total value does not — and a buyback above fair value actively destroys per-share value, the same way overpaying for anything does.

**"An option ignores risk because it discounts at the risk-free rate."** No. The option handles risk through replication, not through a risk premium on the discount rate. Volatility — the option's measure of risk — enters through the d1 and d2 terms, which is why a higher implied volatility raises the put's value even though the discount rate never changed. The risk is in the formula; it is just not in the discount rate.

**"Bitcoin's wide fair-value range means the analysis failed."** The reverse. A wide, honest range is the correct output for an asset with no cash flows; the failure would be a single confident number that hides the uncertainty. The range is what tells you to hold a small position, which is the most useful thing the valuation produces.

## How it shows up in real markets

The October 2022 to October 2023 rate-rise episode is this whole article playing out live. As the 10-year Treasury yield climbed from roughly 1.5% in early 2022 toward 5% in late 2023, every asset on the discount-rate spine moved in the direction the chain predicts. Long-duration Treasury prices fell hard — exactly the duration arithmetic from Asset 4. High-multiple growth stocks, whose value is overwhelmingly terminal value (the most rate-sensitive piece), de-rated sharply, while cash-rich mature names like Apple held up better because less of their value sat in the distant future. Bank stocks, valued on book, wrestled with the offsetting effects of higher lending margins and mark-to-market losses on their bond holdings. Option premiums rose with volatility as the VIX spiked. And Bitcoin, the asset furthest out the risk curve, fell roughly 65% from its 2021 peak as the risk-free hurdle rose. One macro variable, five asset classes, the same spine.

Run the chain forward as a single trade idea to feel how connected these assets are. An analyst in late 2021 who correctly forecast that the Fed would raise rates sharply had, in that one view, a thesis on all five assets at once. Short long-duration Treasuries directly, because their price falls mechanically with yield. Underweight the highest-multiple growth stocks, because their terminal-value-heavy valuations are the most rate-sensitive equities. Be cautious on emerging-market banks like VCB, whose required return rises on two fronts as both the global rate and the country premium widen. Expect option premiums to richen as volatility spikes into the repricing. And expect Bitcoin, the longest-duration risk asset of all, to fall hardest. That is not five separate forecasts; it is one macro view propagating through one shared discount rate. The investor who sees the spine trades the whole board from a single insight, while the investor who treats each asset as its own island is surprised five times by the same event.

The Vietnam angle sharpens the country-risk point. Through 2022-2023, as the State Bank of Vietnam moved rates and the dong came under pressure, the discount rate appropriate for VCB rose on two fronts at once — the global risk-free rate climbing *and* the country-risk premium widening. An analyst valuing VCB in dollars had to layer both effects, which is exactly why emerging-market valuations swing more than developed-market ones for the same underlying business. The mechanics are in [emerging-market valuation](/blog/trading/asset-valuation/emerging-market-stock-valuation-country-risk-discount-rate), and the broader role of central-bank balance sheets in setting that floor is in [quantitative easing explained](/blog/trading/finance/quantitative-easing-explained-printing-money).

Step back to the whole board one last time. The bond was a pure rate calculation with a fair value to the penny. Apple was a forecast of cash flows, cross-checked three ways, hinging on a growth rate. The bank was the same equity logic rerouted through book value because its capital is its product. The option was a contingent claim valued by replication, its risk living in volatility rather than a discount-rate premium. And Bitcoin was a confession that some assets have no cash flows at all, so the best you can do is bound the futures and weight them. Five methods that look nothing alike on the page are, underneath, one act repeated: estimate the cash, judge the risk, pick the rate, and respect the range. The methods diverge only because the cash flows get harder to know as you move from a Treasury to a token — and the difficulty of the cash flows, not any difference in theory, is what makes the spreadsheets look different.

The practical close: when you next see five different assets, do not reach for five different mental models. Find the cash flows, find the risk, find the rate — and ask which assumption the value hangs on. That single habit, applied across every asset class, is what the whole series was building toward. The craftsperson's edge is not knowing more formulas than the next analyst; it is seeing that the formulas were always the same formula, and spending your judgment where it actually matters — on the one or two assumptions that the value truly depends on.

## Further reading & cross-links

- [What is value? Philosophy and frameworks for asset pricing](/blog/trading/asset-valuation/what-is-value-philosophy-frameworks-asset-pricing)
- [The time value of money: the engine inside every valuation model](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model)
- [Risk and required return: CAPM, beta, and the cost of capital](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital)
- [Discount rates in practice: WACC, cost of equity, and unlevered beta](/blog/trading/asset-valuation/discount-rates-practice-wacc-cost-equity-unlevered-beta)
- [Discounted cash flow (DCF) equity valuation, step by step](/blog/trading/asset-valuation/discounted-cash-flow-dcf-equity-valuation-step-by-step)
- [The price-to-earnings (P/E) ratio for valuing stocks](/blog/trading/asset-valuation/price-to-earnings-ratio-pe-valuation-stocks)
- [The price-to-book (P/B) ratio for valuing equity](/blog/trading/asset-valuation/price-to-book-ratio-pb-valuation-equity)
- [The Black-Scholes model and the Greeks for options valuation](/blog/trading/asset-valuation/black-scholes-model-greeks-options-valuation)
- [Bond valuation: yield, duration, and convexity](/blog/trading/asset-valuation/bond-valuation-yield-duration-convexity)
- [Crypto valuation: pricing Bitcoin, Ethereum, and tokens](/blog/trading/asset-valuation/crypto-valuation-bitcoin-ethereum-token-pricing)
- [Valuing Vietcombank (VCB): a P/B and DDM case study](/blog/trading/asset-valuation/valuing-vietcombank-vcb-pb-ddm-case-study)
- [Emerging-market stock valuation: country risk and the discount rate](/blog/trading/asset-valuation/emerging-market-stock-valuation-country-risk-discount-rate)
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates)
- [Quantitative easing explained: printing money](/blog/trading/finance/quantitative-easing-explained-printing-money)
- [Interest rates, bonds, and stocks: the relationship](/blog/trading/macro-trading/interest-rates-bonds-stocks-relationship)
