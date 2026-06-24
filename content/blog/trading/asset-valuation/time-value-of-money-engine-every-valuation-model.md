---
title: "Time Value of Money: The Engine of Every Valuation Model"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "Why a dollar today is worth more than a dollar tomorrow — and how that single insight powers every DCF, bond price, mortgage, and option model ever built."
tags: ["time-value-of-money", "present-value", "future-value", "discounting", "compounding", "tvm", "valuation", "dcf", "annuity", "npv"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Time Value of Money is the single idea that every valuation model is built on: a dollar today is worth more than a dollar in the future because it can be invested, and that difference is precisely measurable.
>
> - **Future Value** compounds a present amount forward: FV = PV × (1 + r)^n
> - **Present Value** discounts a future cash flow backward: PV = CF / (1 + r)^n
> - **Annuities** apply PV to a stream of equal payments — the formula that prices bonds, mortgages, and leases
> - **NPV** is the sum of all discounted cash flows and the universal "accept or reject" rule for any investment
> - The discount rate you choose — your required return — controls everything: a 1% change in r moves a long-dated asset's price by more than you'd expect

---

## The lottery problem: which would you choose?

Imagine you win the lottery. The prize is \$1,000,000. You have two options:

**Option A:** Receive the full \$1,000,000 today, in a single lump sum.

**Option B:** Receive \$50,000 per year for 20 years — which adds up to \$1,000,000 in total payments.

Most people, when asked this question quickly, guess that the two options are roughly equal — after all, both pay out \$1 million. A few clever people say Option A must be better "because of inflation." But very few people can say *by exactly how much* Option A is better, or why.

Here is the precise answer: at a 6% discount rate — a reasonable assumption for what you could earn investing in a diversified portfolio — Option A is worth \$1,000,000 today, and Option B is worth only \$573,496 today. Option A is worth **74% more**.

That arithmetic gap — \$426,504 of value that disappears simply because of *when* you receive the money — is the Time Value of Money (TVM) in action. It is not a minor adjustment or a rounding assumption. It is the foundational truth on which every valuation model in finance is built. The DCF model, bond pricing, mortgage math, dividend discount models, option pricing — every single one traces back to the same core insight: *a dollar you receive sooner is worth more than a dollar you receive later, and the difference is exactly calculable.*

This post builds TVM from the ground up. We start with the three reasons money has time value, then build the mechanics of compounding and discounting one formula at a time, then show how those mechanics extend to annuities, perpetuities, and finally to Net Present Value — the universal investment decision rule. Along the way, we'll price a Vietnamese government bond, compute a mortgage payment, and see why inflation data from the State Bank of Vietnam makes TVM feel very concrete to anyone saving in Vietnamese đồng.

![TVM timeline: $100 compounds to $133 over 3 years at 10%](/imgs/blogs/time-value-of-money-engine-every-valuation-model-1.png)

The diagram above is the mental model for the whole post. A dollar invested today at 10% becomes \$1.10 in year one, \$1.21 in year two, and \$1.33 in year three. Flip the arrows and you have discounting: knowing that \$1.33 arrives in year three, what is it worth today? Exactly \$1.00. The entire post is just this diagram made rigorous.

---

## Foundations: why money has time value

Before we write a single formula, we need to understand *why* money has time value. There are three distinct reasons, and each maps onto a real cost you pay — or a real opportunity you give up — when you receive money later instead of now.

### Reason 1: opportunity cost

If I give you \$100 today, you can invest it. At 5% annual interest, it becomes \$105 in one year. If you agree to wait a year for your \$100, you forego that \$5 — that is the *opportunity cost* of waiting. It is not inflation, it is not risk, it is simply the earning power that a dollar in your hands has right now.

This is why even in a world with zero inflation and perfectly safe investments, money would still have time value. A savings account paying 0.5% means your \$100 today is worth more than \$100 in a year — by exactly 50 cents — purely because of the earning opportunity.

### Reason 2: inflation

Prices generally rise over time. A bag of groceries that costs \$50 today will likely cost \$52 or \$53 in a year. If you hold \$50 in cash for a year, its *real purchasing power* erodes — you can buy slightly less with it. So even if there were no investment return available at all, you'd prefer \$50 today over \$50 in a year because \$50 today buys more real stuff.

Inflation and opportunity cost are related but separate. Your investment return (say, 5%) compensates you for *both* inflation (say, 2.5%) *and* the real component of the return (the remaining 2.5%). This distinction matters in practice — when you're valuing a company that operates in a high-inflation environment like Vietnam, you need to think carefully about whether your cash flows and your discount rate are expressed in *nominal* terms (including inflation) or *real* terms (stripping it out). More on this when we get to the Vietnam section.

### Reason 3: risk

Money promised in the future is less certain than money in your hand today. The counterparty who owes you \$100 in five years might default. The business whose cash flows you're projecting might miss its targets. Even a government can — and historically has — failed to honor its obligations. Risk *requires compensation*: if you are going to take on uncertainty, you demand a higher return to justify accepting it.

This third reason is the one that makes discount rates vary so widely across different investments. A 30-day U.S. Treasury bill, backed by the full faith and credit of the U.S. government, commands a discount rate near the "risk-free rate" — perhaps 4.5% today. A venture-capital investment in a pre-revenue startup might require a 35% discount rate to justify the enormous risk of loss. The formula is the same; the rate is vastly different.

### The three reasons unified: the discount rate

In practice, the discount rate `r` that we plug into our present value formulas bundles all three reasons together:

```
r = risk-free rate + inflation premium + risk premium
```

Or more cleanly: `r` is the return you could earn on an alternative investment of similar risk. This is what economists call the *opportunity cost of capital*. If you can earn 8% per year by investing in a diversified stock index, then any investment you evaluate should be discounted at (at least) 8% — because that is what you give up to tie your money up in this alternative.

This is the key insight that makes TVM not just an accounting concept but a decision tool: every time you compute the present value of a cash flow, you are implicitly comparing it to your best alternative use of that money.

---

## Future Value: the power of compounding

Now let's build the mechanics. Start with the simplest question: if you invest \$1,000 today at 8% per year, how much do you have in three years?

Year 1: \$1,000 × 1.08 = **\$1,080**

Year 2: \$1,080 × 1.08 = **\$1,166.40**

Year 3: \$1,166.40 × 1.08 = **\$1,259.71**

Notice what happened. In year 1, the interest was \$80 (8% of \$1,000). In year 2, the interest was \$86.40 — not \$80 — because you earned 8% on the larger balance of \$1,080. In year 3, the interest was \$93.31. This is **compounding**: you earn returns *on* your returns, not just on your original principal.

The pattern is clear. We multiplied by 1.08 three times, which is the same as multiplying by (1.08)³. This gives us the **Future Value formula**:

$$FV = PV \times (1 + r)^n$$

Where:
- `FV` = Future Value (what you end up with)
- `PV` = Present Value (what you start with)
- `r` = the interest rate per period (as a decimal: 8% = 0.08)
- `n` = number of periods

This formula is the engine that runs all of finance. Everything else is either a variation of it or its inverse.

### Compounding vs. simple interest

It's worth pausing to understand why compounding is so dramatically different from *simple interest*, where you earn interest only on your original principal and never on accumulated interest.

With simple interest at 8%, your \$1,000 grows by \$80 every single year. After 30 years: \$1,000 + (30 × \$80) = **\$3,400**.

With compound interest at 8%, your interest each year is a little larger than the year before — because the balance it's applied to keeps growing. After 30 years: \$1,000 × (1.08)^30 = **\$10,063**.

That is a difference of \$6,663 — nearly 3× more wealth — simply from the compounding mechanism, not from any difference in the stated rate.

![Compounding vs simple interest: $1,000 at 8% over 30 years](/imgs/blogs/time-value-of-money-engine-every-valuation-model-2.png)

The figure above makes this vivid. The simple interest line grows linearly — a straight diagonal. The compound interest line curves exponentially upward — slowly at first, then accelerating dramatically in the later years. This is the famous "hockey stick" shape that people associate with wealth accumulation, and it comes entirely from the mathematical property of (1+r)^n: exponential growth.

The implication is profound: time matters enormously. An investor who starts at age 25 with the same savings rate as one who starts at age 35 ends up with a dramatically larger nest egg at retirement — not because they worked harder, but because their money had more years to compound.

### The Rule of 72

A useful shortcut: to estimate how many years it takes to double your money at a given rate, divide 72 by the rate. At 8%, money doubles in 72/8 = 9 years. At 6%, it takes 72/6 = 12 years. At 12%, just 72/12 = 6 years.

This is an approximation (the exact answer uses logarithms), but it is accurate enough for mental arithmetic and is used constantly by professional investors to do quick "sanity checks" on projections.

### Compounding frequency: more often is more powerful

So far we've assumed annual compounding — interest is calculated and added to the balance once per year. In reality, most financial instruments compound more frequently: savings accounts may compound monthly, credit cards daily, continuous-compounding models exist in derivatives pricing.

How does compounding frequency affect the outcome? Suppose you deposit \$1,000 at a stated annual rate of 12%. Let's compare different compounding frequencies over one year:

| Compounding | Effective Rate | \$1,000 Grows To |
|---|---|---|
| Annual (1×/yr) | 12.000% | \$1,120.00 |
| Semi-annual (2×/yr) | 12.360% | \$1,123.60 |
| Quarterly (4×/yr) | 12.551% | \$1,125.51 |
| Monthly (12×/yr) | 12.683% | \$1,126.83 |
| Daily (365×/yr) | 12.747% | \$1,127.47 |
| Continuous | 12.750% | \$1,127.50 |

With annual compounding, the 12% stated rate *is* the effective rate. With monthly compounding, the stated 12% translates to an effective annual rate (EAR) of 12.683% — because each month's interest is added to the balance and earns its own interest for the remaining months.

The general formula: **EAR = (1 + r/m)^m − 1**, where `r` is the stated annual rate and `m` is the number of compounding periods per year.

For continuous compounding — the mathematical limit as m → ∞ — the formula becomes: **FV = PV × e^(r×t)**, where `e` ≈ 2.71828 is Euler's number. This formula is used extensively in derivatives pricing (the Black-Scholes model for options uses continuous compounding throughout).

**Why this matters in practice:** When comparing financial products, always convert to the *effective annual rate* to make a true apples-to-apples comparison. A mortgage advertised at 7% monthly compounding has a higher true cost than one at 7% semi-annual compounding. In Vietnam, bank deposit advertisements frequently quote annual rates, but the compounding frequency varies; always read the fine print to understand the true return.

### Solving for r and n: other uses of the FV formula

The future value formula FV = PV × (1+r)^n can be rearranged to solve for any of its four components:

**Solving for r** (the return on an investment):

If you paid \$600 for an investment that returns \$1,000 in 8 years, what is your annual return?

```
r = (FV/PV)^(1/n) − 1 = (1000/600)^(1/8) − 1 = (1.667)^0.125 − 1 = 1.0659 − 1 = 6.59%
```

Your annualized return is **6.59%**. This calculation is used constantly in practice — for computing the historical annual return of a stock, the yield-to-maturity of a zero-coupon bond, the IRR of a real estate investment.

**Solving for n** (how long until you reach a target):

If you invest \$10,000 at 7% per year, how many years until you have \$25,000?

```
n = ln(FV/PV) / ln(1+r) = ln(25000/10000) / ln(1.07) = ln(2.5) / ln(1.07) = 0.916 / 0.0677 = 13.5 years
```

These inverse calculations — solving for r or n rather than FV or PV — are just as important as the forward calculation. They answer the questions investors actually ask: "How much is this investment really earning me?" and "How long until I reach my savings goal?"

### Growth at different rates

The impact of the discount rate compounds (literally) over time. The difference between an 8% and a 12% annual return looks modest in year one (\$80 vs. \$120), but balloons enormously over decades.

![Compound growth at 4%, 8%, 12% over 30 years](/imgs/blogs/time-value-of-money-engine-every-valuation-model-3.png)

At 4%, \$1,000 becomes \$3,243 in 30 years. At 8%, it becomes \$10,063. At 12%, it becomes \$29,960. The 4% difference between the 8% and 12% scenarios produces nearly \$20,000 more wealth — almost a 3× gap — not from any magic, but simply because of exponential growth over a long enough time horizon. This is why small differences in return assumptions matter enormously when valuing long-lived assets.

#### Worked example: the lottery revisited

Let's now compute the lottery problem precisely. You win \$1M. Option A is \$1M today. Option B is \$50,000 per year for 20 years. Assume you can earn 6% per year on investments.

The key question: what is the *present value* of Option B?

This is an annuity (a stream of equal payments), and we'll derive the exact formula shortly. For now, let's just think about the first few payments:

- Payment 1, one year from now: \$50,000 / (1.06)^1 = \$47,170
- Payment 2, two years from now: \$50,000 / (1.06)^2 = \$44,500
- Payment 10, ten years from now: \$50,000 / (1.06)^10 = \$27,920
- Payment 20, twenty years from now: \$50,000 / (1.06)^20 = \$15,590

Each payment is worth less and less in today's dollars because you have to wait longer to receive it. Sum all 20 discounted payments and you get \$573,496.

**Option A is worth \$1,000,000 and Option B is worth \$573,496. Option A is 74% more valuable.** The \$426,504 difference is the time value: money delayed is money destroyed — partially, measurably, precisely.

---

## Present Value: discounting back to today

Now we flip the direction. Instead of asking "what does \$100 today become in the future?", we ask: "what is a promised future cash flow worth *right now*?"

This operation is called **discounting**, and it is the arithmetic inverse of compounding. If FV = PV × (1+r)^n, then:

$$PV = \frac{CF}{(1+r)^n}$$

Where `CF` is the future cash flow expected at time `n`. This is the most important formula in all of finance. Every DCF model, every bond price, every mortgage payment, every option price — all of them reduce to computing this fraction many times and summing the results.

![PV formula decomposition: every component labeled](/imgs/blogs/time-value-of-money-engine-every-valuation-model-4.png)

Let's break down each component using the diagram above:

- **CF** (Cash Flow): the amount of money you expect to receive at time `n`. This could be a coupon payment on a bond, a dividend from a stock, or the terminal value of a business in year 10 of a DCF.
- **r** (Discount Rate): your required annual return — the opportunity cost of capital adjusted for risk. This is the "price of waiting" you demand.
- **n** (Periods): how many years (or months, or quarters) into the future the cash flow arrives.
- **PV** (Present Value): what that future cash flow is worth *right now*, in today's money.

The denominator (1+r)^n is called the **discount factor**. At 10% for 5 years, the discount factor is (1.10)^5 = 1.611 — meaning you divide the future cash flow by 1.611 to get its present value.

#### Worked example: a future payment

Suppose you're promised a \$10,000 payment five years from now. You can earn 8% per year on your money. What is that promise worth today?

PV = \$10,000 / (1.08)^5 = \$10,000 / 1.469 = **\$6,806**

Intuition: if you invested \$6,806 today at 8% per year for 5 years, you'd have exactly \$10,000 at the end. So the promise of \$10,000 in five years, to an investor who can earn 8%, is exactly equivalent to having \$6,806 in hand today. The two are interchangeable — *at an 8% discount rate*.

If your required return were higher — say, 12% — the present value drops further: PV = \$10,000 / (1.12)^5 = \$5,674. The same future payment is worth *less* to an investor who demands a higher return. This is why higher-risk assets trade at lower prices.

### The discount rate controls everything

The single most important sensitivity in any valuation model is the discount rate. Change it by 1 percentage point and long-duration assets (those whose cash flows arrive far in the future) can move 10–30% in price.

![How discount rate affects PV: $1,000 in 10 years at various rates](/imgs/blogs/time-value-of-money-engine-every-valuation-model-6.png)

The bar chart above shows the present value of a single \$1,000 payment received in 10 years, at five different discount rates:

| Discount Rate | PV of \$1,000 in 10 Years |
|---|---|
| 3% | \$744 |
| 5% | \$614 |
| 8% | \$463 |
| 10% | \$386 |
| 15% | \$247 |

The difference between a 3% and a 15% discount rate causes the same \$1,000 cash flow to be valued at \$744 vs. \$247 — a 3× gap. This is why arguments about "what discount rate to use" are not academic — they determine whether an asset looks cheap or expensive by hundreds of percent.

---

## Annuities: valuing a stream of equal cash flows

Most real-world financial instruments don't produce a single lump-sum payment. They produce a *series* of payments over time: bond coupon payments every six months, monthly mortgage payments, annual dividend payments, lease payments. When those payments are equal in size and occur at regular intervals, the stream is called an **annuity**.

We could compute the present value of an annuity by discounting each payment individually and summing them — that's exactly what the annuity timeline below shows. But mathematicians long ago figured out a shortcut that turns those N individual calculations into a single formula.

![Annuity timeline: 5 payments of $1,000 each discounted back to today](/imgs/blogs/time-value-of-money-engine-every-valuation-model-5.png)

The five green boxes in the diagram represent five equal payments of \$1,000 at the end of years 1 through 5. Each arrow is a discounting operation back to time zero. The year-1 payment loses the least value (divided by 1.10^1 = 1.10); the year-5 payment loses the most (divided by 1.10^5 = 1.611). Sum all five present values and you get \$3,791.

The **Present Value of an Annuity formula** collapses this sum:

$$PV_{\text{annuity}} = C \times \frac{1 - (1 + r)^{-n}}{r}$$

Where:
- `C` = the payment amount per period (equal for all periods)
- `r` = discount rate per period
- `n` = number of periods

The bracketed fraction is called the *annuity factor* (or *present value interest factor of an annuity*). For C = \$1,000, r = 10%, n = 5:

Annuity factor = [1 − (1.10)^{-5}] / 0.10 = [1 − 0.6209] / 0.10 = 0.3791 / 0.10 = 3.791

PV = \$1,000 × 3.791 = **\$3,791**

### Annuity due vs. ordinary annuity

There's a subtle timing distinction worth knowing. An **ordinary annuity** (also called "annuity in arrears") has payments at the *end* of each period — this is the standard we've been using. An **annuity due** has payments at the *beginning* of each period — like rent, which is typically paid at the start of the month.

For an annuity due, all payments arrive one period earlier, so each one is discounted one fewer time. The present value of an annuity due is simply the ordinary annuity PV multiplied by (1+r):

PV_due = PV_ordinary × (1 + r)

For our example: PV_due = \$3,791 × 1.10 = \$4,170. Receiving the same five \$1,000 payments one period earlier is worth \$379 more.

#### Worked example: mortgage payment calculation

Let's compute the monthly mortgage payment on a \$300,000 30-year fixed-rate mortgage at 7% per year. Mortgages are ordinary annuities with monthly payments — so we need to adjust the inputs.

- Monthly rate `r` = 7% / 12 = 0.5833% = 0.005833
- Number of periods `n` = 30 × 12 = 360 months
- Present Value (loan balance) = \$300,000

The annuity formula rearranges to solve for the payment `C`:

$$C = PV \times \frac{r}{1 - (1 + r)^{-n}}$$

Plugging in:

```
C = $300,000 × [0.005833 / (1 − (1.005833)^−360)]
  = $300,000 × [0.005833 / (1 − 0.1239)]
  = $300,000 × [0.005833 / 0.8761]
  = $300,000 × 0.006653
  = $1,996
```

The monthly payment is **\$1,996**. Over 360 months, you pay 360 × \$1,996 = \$718,560 in total — more than twice the original loan. The \$418,560 excess is entirely interest: the price of borrowing \$300,000 for 30 years. This is time value of money made visible in your bank statement every month.

---

## Perpetuities: cash flows that last forever

What if the cash flows never stop? A **perpetuity** is an annuity with an infinite life. This sounds abstract, but it's practically useful: preferred stock often promises a fixed dividend indefinitely, and the "terminal value" in most DCF models is calculated using a perpetuity formula.

Take the annuity formula as n → ∞: (1+r)^{-n} → 0, so the 1 − (1+r)^{-n} term becomes just 1. The formula collapses to:

$$PV_{\text{perpetuity}} = \frac{C}{r}$$

A perpetuity paying \$100 per year at a 10% discount rate is worth \$100 / 0.10 = **\$1,000** today. Double-check: if you invest \$1,000 at 10% and spend the interest each year, you can pay \$100 per year forever without touching the principal. The formula is exact.

### The Gordon Growth Model (a growing perpetuity)

Most real cash flows grow over time. A company's dividends tend to grow with the economy; a building's rent tends to rise with inflation. A **growing perpetuity** discounts cash flows that grow at a constant rate `g` forever:

$$PV_{\text{growing perpetuity}} = \frac{C_1}{r - g}$$

Where `C_1` is next year's cash flow, `r` is the discount rate, and `g` is the constant growth rate. This formula requires r > g (otherwise the present value would be infinite or negative, which makes no sense).

#### Worked example: Gordon Growth Model

A company pays an annual dividend of \$4 per share and you expect that dividend to grow at 3% per year indefinitely. Your required return on this investment is 9%.

$$P = \frac{\$4}{0.09 - 0.03} = \frac{\$4}{0.06} = \$66.67$$

The fair value of this share is **\$66.67** — and that is literally the entire *Dividend Discount Model* in one step. The entire DCF framework for dividend-paying stocks traces back to this perpetuity formula. If the share trades at \$50, it looks cheap (dividend yield exceeds required return); if it trades at \$90, it looks expensive.

Notice how sensitive the price is to `g`. If growth expectations rise from 3% to 4%, the price becomes \$4 / (0.09 − 0.04) = \$4 / 0.05 = \$80 — a 20% increase in price from a 1% increase in growth expectations. This is why stocks are so sensitive to small changes in long-run growth forecasts.

---

## Net Present Value: the universal decision rule

We now have all the tools needed for the most important concept in corporate finance: **Net Present Value (NPV)**.

NPV is the sum of the present values of *all* cash flows associated with an investment — inflows (positive) and outflows (negative). It answers the question: "If I make this investment, how much value am I creating or destroying, measured in today's dollars?"

$$NPV = \sum_{t=0}^{n} \frac{CF_t}{(1+r)^t}$$

Where `CF_t` is the cash flow at time `t` (negative for the initial outlay, positive for future inflows).

The **NPV decision rule** is:
- **NPV > 0**: Accept. The investment creates value. It earns more than your required return `r`.
- **NPV < 0**: Reject. The investment destroys value. It earns less than your required return `r`.
- **NPV = 0**: Indifferent. The investment earns exactly `r` — your required return, no more, no less.

![NPV decision pipeline: from cash flows to accept or reject](/imgs/blogs/time-value-of-money-engine-every-valuation-model-7.png)

#### Worked example: an investment decision

You're evaluating a project that requires an upfront investment of \$1,000 and generates \$300 in year 1, \$400 in year 2, and \$500 in year 3. Your required return is 10%.

Step 1 — List cash flows: −\$1,000 (now), +\$300 (yr 1), +\$400 (yr 2), +\$500 (yr 3).

Step 2 — Discount each:

- t=0: −\$1,000 / (1.10)^0 = −\$1,000.00
- t=1: +\$300 / (1.10)^1 = +\$272.73
- t=2: +\$400 / (1.10)^2 = +\$330.58
- t=3: +\$500 / (1.10)^3 = +\$375.66

Step 3 — Sum: −\$1,000 + \$272.73 + \$330.58 + \$375.66 = **−\$20.97**

NPV is slightly negative. **Reject this project.** Even though total undiscounted cash inflows (\$1,200) exceed the outlay (\$1,000), once we account for the time value of money and your 10% required return, the project destroys value. You'd be better off investing that \$1,000 elsewhere at 10%.

If you lower the discount rate to 8%, the same project has NPV = +\$30 and you accept it. The critical rate where NPV exactly equals zero is called the **Internal Rate of Return (IRR)** — here approximately 9.7% — and it is the project's "true" return, measured in TVM-adjusted terms.

### The Internal Rate of Return (IRR): NPV's cousin

NPV requires you to choose a discount rate upfront. The **Internal Rate of Return (IRR)** answers a related question differently: "What discount rate makes this investment's NPV exactly equal to zero?" In other words, what is the investment's *own inherent return*, expressed as a percentage?

For the project above (−\$1,000 initial outlay, \$300/\$400/\$500 inflows), we showed NPV = −\$20.97 at 10%. If we lower the discount rate slightly to ~9.7%, NPV would be exactly zero. That 9.7% is the IRR.

The IRR decision rule: accept if IRR > your required return (cost of capital); reject if IRR < required return. This gives the same answer as NPV in most cases, but IRR has pitfalls — notably, projects with multiple sign changes (cash outflows in the middle of a project) can have multiple IRRs, which are mathematically meaningless. NPV is the theoretically superior tool; IRR is the practically popular one because it produces an intuitive percentage that non-finance audiences can grasp.

**Worked example: bond yield-to-maturity as an IRR**

When you buy the Vietnamese government bond we priced earlier at \$959, you receive five years of \$60 coupons plus \$1,000 at maturity. What is your actual return if you hold to maturity?

The answer is the IRR of these cash flows: −\$959 today, +\$60 in years 1–4, +\$1,060 in year 5. Solving numerically gives an IRR of exactly **7%** — confirming that buying a bond priced at a discount to face value earns you more than the coupon rate. This "IRR of a bond" is what the market calls the **yield to maturity (YTM)** — it is the single discount rate that equates the bond's price to the present value of its cash flows. It is IRR applied to fixed income.

### NPV vs. "just adding up the cash flows"

A common error is to compare investments by their total *undiscounted* cash flows. Consider two projects, each requiring a \$500 outlay:

| Project | Year 1 | Year 2 | Year 3 | Total |
|---|---|---|---|---|
| A | \$600 | 0 | 0 | \$600 |
| B | 0 | 0 | \$650 | \$650 |

Project B has higher total cash flows. But at 10%:

- NPV_A = −\$500 + \$600/1.10 = −\$500 + \$545 = **+\$45**
- NPV_B = −\$500 + \$650/1.331 = −\$500 + \$488 = **−\$12**

Project A is better. Receiving \$600 in one year beats receiving \$650 in three years, despite the higher dollar amount, because money today is worth more than money later. This is TVM in decision-making form.

---

## How discount rate choice changes everything

We touched on this earlier, but it deserves deeper treatment because it's where so many valuations go wrong — or where sophisticated analysts look to stress-test a thesis.

The present value formula inverts the compounding exponent: PV = CF / (1+r)^n. Because of the exponential, small changes in `r` have large effects on `n`-year-out cash flows, and the effect amplifies as `n` grows.

Let's quantify this. Consider a \$100 cash flow at different time horizons, discounted at 8% vs. 10%:

| Years Out | PV at 8% | PV at 10% | Difference |
|---|---|---|---|
| 1 year | \$92.59 | \$90.91 | −\$1.68 (1.8%) |
| 5 years | \$68.06 | \$62.09 | −\$5.97 (8.8%) |
| 10 years | \$46.32 | \$38.55 | −\$7.77 (16.8%) |
| 20 years | \$21.45 | \$14.86 | −\$6.59 (30.7%) |
| 30 years | \$9.94 | \$5.73 | −\$4.21 (42.3%) |

A 2-percentage-point increase in the discount rate cuts the PV of a 30-year cash flow by 42%. This is why long-duration assets — growth stocks, long-dated bonds, infrastructure projects with 40-year cash flows — are so sensitive to changes in interest rates or risk premiums. When central banks raise rates, they're effectively raising the discount rate for the entire economy, which mechanically reduces the present value of all future cash flows.

### The risk-free rate as the foundation

In practice, discount rates are usually built up from the risk-free rate (typically the yield on 10-year government bonds) plus a risk premium that compensates for the specific risks of the investment:

```
r = risk-free rate + equity risk premium + size/industry premium + specific risk
```

For U.S. stocks, a common estimate is: 4.5% (risk-free) + 4.5% (equity risk premium) = 9% baseline. Riskier companies add further premiums; safer companies (utilities, consumer staples) might use a lower premium.

The choice of equity risk premium is among the most contested topics in finance. Estimates range from 3% to 7% depending on methodology and time period studied. A 2-percentage-point change in the ERP would change the fair value of the S&P 500 index by roughly 15–25% — which is why it matters whether you're using a 4% or 6% assumption.

### Nominal vs. real discount rates: a crucial distinction

The discount rate you use in TVM can be expressed in two ways:

- **Nominal rate**: includes inflation. A nominal rate of 9% at 3% inflation means you're earning 6% in real terms.
- **Real rate**: inflation-adjusted. A real rate of 6% means your purchasing power grows by 6% per year.

The **Fisher equation** links them: (1 + nominal rate) = (1 + real rate) × (1 + inflation rate). For small rates, this approximates to: real rate ≈ nominal rate − inflation.

The rule is: **your cash flow forecasts and your discount rate must both be in the same terms.** If you forecast cash flows in *nominal* dollars (growing with inflation as well as real growth), discount them at the *nominal* rate. If you project *real* cash flows (constant purchasing power, no inflation in the numbers), discount at the *real* rate. Either approach gives the same answer — but mixing them produces a number that is wrong by exactly the inflation assumption.

In emerging-market contexts like Vietnam, where inflation has historically been volatile, this distinction is especially important. A Vietnamese company projecting revenues in nominal VND — growing partly from real business growth and partly from inflation — should be discounted at a nominal VND-denominated rate. Applying a USD-based discount rate to VND cash flows without adjusting for the inflation and currency differential is a common and costly error.

### Duration: the TVM sensitivity measure

Investors and analysts need a single number that captures how sensitive a security's price is to changes in the discount rate. That number is called **duration** (specifically, Macaulay duration for bonds).

Duration is the *weighted average time* until you receive your cash flows, where each cash flow's weight is its present value as a fraction of the total price. A bond that pays all its cash flows tomorrow has duration near zero (rate changes barely affect its price). A 30-year zero-coupon bond — which pays nothing until maturity — has duration of exactly 30 years and enormous rate sensitivity.

The practical rule: for a 1-percentage-point increase in the discount rate, a bond's price falls by approximately `duration × 1%`. A 10-year duration bond loses about 10% of its value for every 1% rise in rates. This is why long-duration bonds are so much riskier when rates are rising — they experience the biggest price impact.

Growth stocks are often described as "long-duration" assets. Their earnings are expected far in the future — years 5, 10, 20 and beyond. Like a long-term bond, their prices are extremely sensitive to the discount rate assumption. This is mechanically why growth stocks sold off sharply in 2022 when the U.S. Federal Reserve raised rates aggressively: higher rates meant higher discount rates, which slashed the present values of those distant future earnings.

---

## Common misconceptions

### Misconception 1: "Compound interest applies only to savings accounts"

Compounding is universal. Any time you earn returns on previous returns — dividends reinvested in more shares, profits reinvested into a business, a hedge fund's gains used to generate next year's profits — compounding is occurring. The formula (1+r)^n applies to any process where returns are proportional to the base. Even *losses* compound: a 50% loss requires a 100% gain to recover, not a 50% gain, because the base has shrunk.

### Misconception 2: "If the total cash flows add up to more than the investment, it's a good deal"

This ignores time value. As we showed in the NPV section, a project generating \$650 in three years can have lower NPV than one generating \$600 in one year. The test is not "do undiscounted inflows exceed outflows" but "is NPV positive." Many bad investments pass the former test and fail the latter.

### Misconception 3: "Inflation is already priced in, so I don't need to worry about it"

This is half-right and half-wrong. If you use *nominal* cash flows (flows that include the expected price increases from inflation), you must use a *nominal* discount rate (one that also includes an inflation premium). If you use *real* cash flows (constant-dollar flows with inflation stripped out), you must use a *real* discount rate. The error happens when people mix levels: projecting cash flows in real terms but discounting at a nominal rate, or vice versa. The rule is simple: **match the inflation assumption in your cash flows to the inflation assumption in your discount rate.**

### Misconception 4: "A higher discount rate always means a lower valuation"

For most standard cash flows, yes. But some financial instruments — like floating-rate bonds, where coupons rise when rates rise — are partially protected from rising discount rates. And some complex derivative structures have nonlinear sensitivities. For a standard DCF or bond price, higher r = lower price. But don't overgeneralize.

### Misconception 5: "TVM math requires a spreadsheet"

The three key formulas — FV = PV × (1+r)^n, PV = CF / (1+r)^n, and the annuity formula — can all be estimated mentally using the Rule of 72 and compound-factor tables. For a working professional, being able to estimate that "\$1M in 20 years at 7% is about \$250K today" (answer: \$258K — within 3%) is more valuable than perfect calculator dependence.

### Misconception 6: "Higher growth always means higher stock prices"

The Gordon Growth Model shows why this is incomplete. P = C/(r − g). If growth g rises but required return r rises by the *same amount* (because the growth is riskier), the price doesn't change. What matters is the *gap* (r − g), not the level of g alone. A company growing at 15% per year but requiring 20% returns to compensate for its risk has a lower price than a stable company growing at 5% with a 7% required return: 1/(0.20−0.15) = 20x earnings vs. 1/(0.07−0.05) = 50x earnings. The market doesn't pay for growth; it pays for growth *in excess of* the return required to bear the risk of achieving it.

---

## How TVM shows up in real markets

### Mortgage math: the cost of time

Every mortgage is an annuity — a series of equal monthly payments that repay both principal and interest over a fixed term. The lender discounts those future payments at the mortgage rate to arrive at the loan amount they're willing to extend.

The monthly payment formula we used earlier tells borrowers exactly what each 1% increase in mortgage rate costs them. On a \$300,000, 30-year mortgage:

- At 5%: monthly payment = **\$1,610** / total paid = \$579,767
- At 7%: monthly payment = **\$1,996** / total paid = \$718,560
- At 9%: monthly payment = **\$2,413** / total paid = \$868,742

The difference between a 5% and 9% rate: \$288,975 more in total interest payments — almost the entire original loan amount again. This is why rising interest rates devastate affordability in real estate markets and why central bank rate decisions have such outsized effects on housing.

### Vietnamese government bond pricing

Let's price a specific bond using TVM. Suppose you're looking at a Vietnamese 5-year government bond with:
- Face value (principal): \$1,000
- Annual coupon rate: 6% (so you receive \$60 per year)
- Current market yield: 7% (the going rate for similar bonds in the market)

The bond pays five annual coupons of \$60 plus the \$1,000 face value at maturity. Its fair price is the PV of all those cash flows, discounted at the market yield of 7%:

| Year | Cash Flow | Discount Factor | Present Value |
|---|---|---|---|
| 1 | \$60 | 1/(1.07)^1 = 0.9346 | \$56.07 |
| 2 | \$60 | 1/(1.07)^2 = 0.8734 | \$52.41 |
| 3 | \$60 | 1/(1.07)^3 = 0.8163 | \$48.98 |
| 4 | \$60 | 1/(1.07)^4 = 0.7629 | \$45.77 |
| 5 | \$1,060 | 1/(1.07)^5 = 0.7130 | \$755.78 |
| **Total** | | | **\$959.01** |

The bond trades at **\$959** — a *discount* to its \$1,000 face value — because the coupon rate (6%) is lower than the market yield (7%). This "bond prices fall when yields rise" relationship is one of the most fundamental in fixed-income markets, and it is a direct consequence of PV mechanics.

You can also verify this with the annuity formula: PV of coupons = \$60 × [1 − (1.07)^{-5}] / 0.07 = \$60 × 4.100 = \$246.00. PV of principal = \$1,000 / (1.07)^5 = \$712.99. Total = \$246.00 + \$712.99 = **\$958.99** (matches, within rounding).

### Vietnam inflation context

Time value of money is not abstract when you live in a country with meaningful inflation. In Vietnam, the annual CPI inflation rate has ranged from 0.6% (2015) to a recent spike near 4.7% (2016), and bank deposit rates at major institutions like Vietcombank and BIDV have historically ranged from 5% to 8%, with a notable peak above 8% in 2022 during the global inflationary wave.

![Vietnam CPI inflation vs bank deposit rate 2015-2024](/imgs/blogs/time-value-of-money-engine-every-valuation-model-8.png)

The green line in the chart above is the **real deposit rate** — what you actually earn above inflation. Notice that the real rate has generally been positive (meaning you do grow your purchasing power by saving in Vietnamese banks), but it compressed significantly in 2022 when inflation rose sharply while deposit rates didn't keep pace immediately.

This chart illustrates a critical TVM principle: the **nominal interest rate** (what the bank advertises) and the **real interest rate** (what you actually earn in purchasing-power terms) are different. The Fisher equation formalizes this:

```
real rate ≈ nominal rate − inflation rate
```

In 2022, with inflation at ~3.2% and deposit rates at ~8%, the real return was approximately 4.8%. In 2021, with inflation at just 1.8% and deposit rates at ~5.5%, the real return was similarly ~3.7%. For a Vietnamese investor evaluating whether to put money in a savings account versus buying corporate bonds versus investing in stocks, the real discount rate — not the nominal one — is what determines whether you're creating or destroying purchasing power.

For Vietnamese corporate bonds, which carry higher risk than government bonds, appropriate discount rates are typically 10–13% in nominal terms — implying real returns of 6–9% when inflation is near 3–4%. Using TVM, we can compute whether those bonds are fairly priced, expensive, or cheap based on the credit quality of the issuer.

### Lease and installment pricing: TVM in everyday transactions

TVM is embedded in virtually every financial contract you sign, whether or not it is labeled as such.

**Car financing:** Suppose a car dealer offers you a \$30,000 car with "0% financing for 36 months." The monthly payment is \$30,000 / 36 = \$833. Sounds free! But if the dealer's cost of capital is 6% per year (0.5% per month), the present value of those 36 payments to the dealer is:

PV = \$833 × [1 − (1.005)^{-36}] / 0.005 = \$833 × 32.87 = **\$27,381**

The dealer is only receiving \$27,381 in present-value terms, even though you pay \$30,000 in total. So either the dealer has priced the car higher than they otherwise would (\$30,000 vs. a cash price of, say, \$27,000), or the manufacturer is subsidizing the interest cost to move inventory. "0% financing" is never truly free — the time value has to come from somewhere.

**Rental vs. ownership decisions:** At any given moment, you can think of a property's value as the present value of all the future rents it could generate. If a property generates \$2,000/month net rent indefinitely and the appropriate capitalization rate (a real estate discount rate) is 5%, its value is approximately \$2,000 × 12 / 0.05 = **\$480,000**. This "income approach" to real estate valuation is a direct application of the perpetuity formula.

**Retirement savings:** The TVM annuity formula in reverse is the most powerful personal finance tool available. If you want to have \$1,000,000 at retirement in 30 years, earning 7% annually, how much must you save per year?

We need to solve for C in the future value of an annuity formula: FV = C × [(1+r)^n − 1] / r

\$1,000,000 = C × [(1.07)^30 − 1] / 0.07 = C × [7.612 − 1] / 0.07 = C × 94.46

C = \$1,000,000 / 94.46 = **\$10,586 per year**, or about \$882 per month. If you wait 10 years and start at age 35 instead of 25, and only have 20 years to compound:

\$1,000,000 = C × [(1.07)^20 − 1] / 0.07 = C × 40.99

C = \$1,000,000 / 40.99 = **\$24,393 per year** — more than double the savings rate, simply because of 10 fewer years of compounding. Starting early is worth \$13,807 per year in required savings — an amount that compounds further into an enormous lifetime cost of procrastination.

### DCF valuation in practice

When a professional analyst builds a DCF model for a company — say, a Vietnamese real estate developer — they are doing nothing more (and nothing less) than:

1. Forecasting the company's free cash flows for 5–10 years
2. Estimating a terminal value using the growing perpetuity formula
3. Discounting all those cash flows back to today using a discount rate (WACC for the whole firm, or equity cost of capital for just the equity portion)
4. Summing all the PVs to get the "intrinsic value" of the firm
5. Comparing that intrinsic value to the market price to decide whether to buy, sell, or hold

Step 3 — choosing the discount rate — is where most of the subjectivity (and most of the disagreement among analysts) lives. Two analysts looking at the same company can arrive at valuations that differ by 50% or more, simply because one uses a 10% discount rate and the other uses a 13% rate. Everything else — the cash flow forecasts, the growth assumptions, the terminal multiple — can be identical.

This is why Warren Buffett has said that he uses the U.S. 30-year Treasury yield as his discount rate (adjusted for confidence in the forecast). It sounds overly simple, but it reflects a deep insight: if you cannot reliably project cash flows 10 or 20 years out, adding "risk premiums" on top of a noisy forecast just adds noise on noise.

### The corporate bond market

In Vietnam's growing corporate bond market, TVM pricing is applied (or misapplied) constantly. Before the 2022 corporate bond market tightening — which saw a number of high-profile defaults by real estate companies — many retail investors were buying five-year corporate bonds at coupons of 9–12% without adequately accounting for the credit risk embedded in those rates.

The TVM framework would ask: if a risk-free government bond yields 6% and a corporate bond yields 10%, the implied extra yield (the *credit spread*) is 400 basis points. That spread is the market's estimate of the default risk. For an investor to be fairly compensated, the expected loss from default, multiplied by its probability, must be at least equal to the 4% premium. If default probability is 2% per year and recovery in default is 40 cents on the dollar, the expected annual loss is 2% × 60% = 1.2% per year — less than the 4% spread, suggesting the bond looked attractive on a risk-adjusted basis. But if default probability was actually 5% per year (as some issuers turned out to warrant), the expected annual loss was 5% × 60% = 3%, and the 4% spread barely compensated for the risk.

This is TVM and probability working together: the discount rate is not just an opportunity cost, it is also a compensation for the *risk of not receiving the promised cash flows at all*.

### Calculating with TVM in Python

For practitioners, reproducing these calculations in code is both a learning tool and a practical necessity. Here is a minimal Python implementation of the core TVM functions:

```python
def future_value(pv, r, n):
    """FV = PV * (1 + r)^n"""
    return pv * (1 + r) ** n

def present_value(cf, r, n):
    """PV = CF / (1 + r)^n"""
    return cf / (1 + r) ** n

def annuity_pv(c, r, n):
    """PV of ordinary annuity: C * [1 - (1+r)^-n] / r"""
    return c * (1 - (1 + r) ** (-n)) / r

def annuity_payment(pv, r, n):
    """Solve for C: pv = C * [1 - (1+r)^-n] / r"""
    return pv * r / (1 - (1 + r) ** (-n))

def perpetuity_pv(c, r):
    """PV of perpetuity: C / r"""
    return c / r

def growing_perpetuity_pv(c1, r, g):
    """Gordon Growth Model: C_1 / (r - g)"""
    if r <= g:
        raise ValueError("r must exceed g for a finite present value")
    return c1 / (r - g)

def npv(cashflows, r):
    """NPV = sum of CF_t / (1+r)^t for t=0,1,...,n"""
    return sum(cf / (1 + r) ** t for t, cf in enumerate(cashflows))

coupons = [60, 60, 60, 60, 1060]
price = npv([-959] + coupons[:-1] + [1060], 0.07)
print(f"Bond NPV at 7%: ${price:.2f}")  # should be near 0

monthly_payment = annuity_payment(300000, 0.07/12, 360)
print(f"Monthly payment: ${monthly_payment:.0f}")  # ~$1,996

lottery_pv = annuity_pv(50000, 0.06, 20)
print(f"Lottery annuity PV: ${lottery_pv:,.0f}")  # ~$573,496
```

Running these confirms every example in this post, and extending them to more complex scenarios — irregular cash flows, varying growth rates, mid-year discounting conventions — is straightforward. The math never changes; only the inputs do.

---

## Further reading and cross-links

Time Value of Money is the foundation. The rest of the Asset Valuation series builds directly on it:

- **[What Is Value? Philosophy, Frameworks, and Asset Pricing](/blog/trading/asset-valuation/what-is-value-philosophy-frameworks-asset-pricing)** — the first post in this series covers why assets have value at all and the different schools of thought (intrinsic value, relative value, market-clearing price)
- **[Discounted Cash Flow: The Complete Guide](/blog/trading/equity-research/discounted-cash-flow-dcf-complete-guide)** — takes the TVM framework from this post and applies it to a full company valuation, with terminal value, WACC, and sensitivity tables
- **[Bond Pricing, Yield, and Duration: Complete Guide](/blog/trading/fixed-income/bond-pricing-yield-duration-complete-guide)** — dives deep into how TVM applies to fixed-income instruments, including duration (the sensitivity of price to yield) and convexity
- **[The Yield Curve Explained](/blog/trading/quantitative-finance/yield-curve-explained)** — the yield curve is a market-derived set of discount rates for different time horizons; understanding TVM is prerequisite to understanding why the yield curve matters

For deeper quantitative treatment:

- **Expected Value and Probability Distributions** (math-for-quants series) — when cash flows are uncertain, you need to discount *expected* cash flows, which requires understanding probability
- **Interest Rates, Bonds, and Stocks** (macro-trading series) — how central bank policy transmits into discount rates and asset prices across the whole economy

### The one rule that governs all of valuation

Every valuation model you'll ever use — no matter how complex, no matter how many columns the spreadsheet has — is just this: list the expected cash flows, choose a discount rate that reflects the risk of receiving them, discount each back to today, and sum. The complexity in real models comes from *estimating* those cash flows and *justifying* that discount rate, not from the mathematics of the operation itself.

TVM is the arithmetic engine. Learning it thoroughly — not just how to apply the formulas, but why the discount rate works the way it does and why timing matters so much — is the single highest-leverage thing a student of finance can do. Once it clicks, the entire landscape of financial instruments starts to look like variations on the same theme.

The lottery problem at the top of this post has an exact answer. Now you know how to compute it.

---

## Putting it all together: a TVM reference map

Here is a compact reference table of every formula introduced in this post, with a one-line description and a worked number:

| Concept | Formula | Example |
|---|---|---|
| **Future Value** | FV = PV × (1+r)^n | \$1,000 × (1.08)^10 = \$2,159 |
| **Present Value** | PV = CF / (1+r)^n | \$2,159 / (1.08)^10 = \$1,000 |
| **Effective Annual Rate** | EAR = (1 + r/m)^m − 1 | 12% monthly = 12.68% EAR |
| **Solving for r** | r = (FV/PV)^(1/n) − 1 | (1,000/600)^(1/8) − 1 = 6.59% |
| **Solving for n** | n = ln(FV/PV) / ln(1+r) | ln(2.5)/ln(1.07) = 13.5 years |
| **Annuity PV** | C × [1−(1+r)^−n] / r | \$1,000 × 3.791 = \$3,791 |
| **Annuity payment** | PV × r / [1−(1+r)^−n] | \$300,000 mortgage → \$1,996/mo |
| **Perpetuity PV** | C / r | \$100 / 0.10 = \$1,000 |
| **Gordon Growth Model** | C₁ / (r − g) | \$4 / (0.09 − 0.03) = \$66.67 |
| **NPV** | Σ CF_t / (1+r)^t | −\$1,000 + PV(inflows) = ? |

Every valuation tool in professional finance — the DCF model, bond pricing, mortgage amortization, option pricing models, real estate cap rates, dividend discount models — reduces to one or more rows of this table. Master these nine formulas and you have the mathematical foundation for the entire discipline.

The journey from here leads naturally to two questions that this post deliberately deferred: (1) how do you forecast the cash flows `CF` that go into these formulas? and (2) how exactly do you derive the discount rate `r` for different types of investments? Both are rich topics with their own post-length treatments — see the [DCF guide](/blog/trading/equity-research/discounted-cash-flow-dcf-complete-guide) for the former and [WACC](/blog/trading/equity-research/wacc-weighted-average-cost-capital) for the latter. But the TVM mechanics you've built here are the prerequisite for both — and for every other quantitative tool in the valuation toolkit.
