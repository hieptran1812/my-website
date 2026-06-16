---
title: "Discounting cash flows: how a bond is actually priced"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A from-scratch walkthrough of how a bond's price is just the present value of every coupon and the principal, why the discount rate is the master dial, and how to price a real bond by hand."
tags: ["fixed-income", "bonds", "present-value", "discounting", "time-value-of-money", "bond-pricing", "discount-rate", "zero-coupon"]
category: "trading"
subcategory: "Fixed Income"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **A bond's price is nothing more than the present value of every dollar it will ever pay you — and present value is just future money shrunk back to today by a discount rate.**
> - A dollar in your hand today is worth more than a dollar a year from now, because today's dollar can earn interest. That gap is the *time value of money*, and the *discount factor* measures exactly how much a future dollar shrinks.
> - To price a bond, you discount each coupon and the final principal back to today, then add them up. That single sum is the price. There is no other magic.
> - The *discount rate* is the master dial. Turn it up and every future dollar shrinks more, so the price falls; turn it down and the price rises. This is the entire reason bond prices move when interest rates move.
> - A bond priced at exactly its discount rate (coupon rate = discount rate) trades at *par* — its face value. A higher discount rate prices it at a *discount*; a lower one, at a *premium*.
> - The zero-coupon bond — one payment, far in the future, discounted once — is the atom of fixed income. Every coupon bond is just a bundle of zeros.

Why does a \$1,000 bond sometimes sell for \$1,000, sometimes for \$960, and sometimes for \$1,040 — even though the piece of paper promises the exact same payments either way? Why does a small move in interest rates, the kind you barely notice on your savings account, knock a few percent off the value of a bond you already own?

The answer is one idea, and once you have it, the entire trillion-dollar machinery of the bond market becomes legible. The idea is *discounting*: a future dollar is worth less than a present dollar, and a bond's price is simply the sum of all its future dollars, each one shrunk back to what it's worth today. That's it. A bond is a promise to pay you a stream of money on a schedule, and its price is the present value of that stream.

![A timeline showing one hundred dollars today next to one hundred dollars in three years, with the future amount shrinking down to its smaller present value](/imgs/blogs/discounting-cash-flows-how-a-bond-is-priced-1.png)

The diagram above is the mental model for this entire post: \$100 promised three years from now is *not* worth \$100 today. If money can earn, say, 5% a year, then a future \$100 is worth only about \$86 right now — because \$86 invested at 5% grows to \$100 in three years. The future amount has to be *discounted* back to the present, and the bigger the interest rate or the longer the wait, the more it shrinks. Pricing a bond is just doing this shrinking trick to every payment the bond makes, and adding the results.

This is the foundation of the entire fixed-income discipline, and it sits at the center of how the whole financial world is priced. The reason mortgages, stocks, pension liabilities, and government budgets all move when "rates" move is that they are all, underneath, streams of future cash flows being discounted — and the discount rate is set in the bond market. Bonds are the price of money, and discounting is how that price gets attached to everything else. (For the macro view of why that single rate matters so much, see [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).)

In this post we build the whole thing from zero. We start with the time value of money and the discount factor, define present value carefully, then price a real bond by hand — a 3-year \$1,000 note with a 5% coupon — first at a 5% discount rate (where it lands exactly at par) and then at 6% (where it falls to a discount). We'll see why the discount rate is the master dial, why later cash flows contribute less, and why the humble zero-coupon bond is the building block everything else is made from. By the end, the bond pricing formula won't be a formula you memorize — it'll be a sentence you understand.

## Foundations: the time value of money, discount factors, and present value

Before we can price anything, we need four ideas, each built on the last: interest, the time value of money, the discount factor, and present value. Take them slowly — every later section leans on these.

### Interest: money has a rental price

When you lend money, you expect to be paid back *more* than you handed over. The extra is *interest* — the rent on money. If you lend \$1,000 for a year at a 5% annual interest rate, at the end of the year you get back your \$1,000 plus \$50 of interest, for \$1,050 total.

The arithmetic is just multiplication. *Growing* money forward in time means multiplying by `(1 + r)`, where `r` is the interest rate written as a decimal (5% = 0.05):

$$ \text{Future value} = \text{Present value} \times (1 + r) $$

- **Present value** — the amount you have today.
- **`r`** — the interest rate per period, as a decimal.
- **Future value** — what that amount grows to after one period.

So \$1,000 today at 5% becomes \$1,000 × 1.05 = \$1,050 in a year. Wait two years and it compounds — the second year's interest is earned on the whole \$1,050, not just the original \$1,000:

$$ \text{Future value} = \text{Present value} \times (1 + r)^n $$

where `n` is the number of periods. \$1,000 at 5% for two years is \$1,000 × 1.05² = \$1,000 × 1.1025 = \$1,102.50. For three years, \$1,000 × 1.05³ = \$1,157.63. The exponent is doing the compounding — interest earning interest.

#### Worked example: how compounding frequency changes the rate

Here's a wrinkle that matters for real bonds. "5% a year" is ambiguous until you say *how often* it compounds. If it compounds once a year, \$1,000 grows to \$1,050.00. But most US bonds pay interest *semiannually* — twice a year — so the convention is to compound half the rate twice. Then \$1,000 grows to \$1,000 × (1 + 0.05/2)² = \$1,000 × 1.025² = \$1,050.625. The same headline "5%" delivers \$1,050.63 instead of \$1,050.00, because the first half-year's interest itself earns interest in the second half.

The gap looks tiny over one year (62 cents), but it compounds over a bond's life and across the whole market. It's why a bond's quoted yield always comes with a compounding convention attached — and why we'll keep this post on clean *annual* compounding so the arithmetic stays transparent, then handle the semiannual reality in the next post. *The same nominal rate can mean slightly different things depending on how often it compounds, so a careful pricer always nails down the convention before touching the numbers.*

### The time value of money: today beats tomorrow

Here is the idea that runs the whole show. **A dollar today is worth more than a dollar tomorrow**, because the dollar today can be put to work and earn interest in the meantime. This is the *time value of money*. It is not a quirk of psychology or impatience (though those exist too) — it is a hard, arithmetic fact as long as money can earn a positive return.

Flip the growth equation around. If \$1,000 *grows into* \$1,050 in a year at 5%, then \$1,050 a year from now is *worth* exactly \$1,000 today. The future amount, brought back to the present, is smaller. Bringing future money back to today is called *discounting*, and it's the reverse of compounding: instead of multiplying by `(1 + r)`, you *divide* by it.

$$ \text{Present value} = \frac{\text{Future value}}{(1 + r)^n} $$

That single equation is the engine of bond pricing. Every payment a bond makes is a "future value" sitting at some date `n` years out; to know what it's worth today, you divide by `(1 + r)^n`. The whole rest of this post is applying this one division, over and over, and adding up the answers.

#### Worked example: shrinking \$100 across three years

You're promised \$100, payable three years from now, and money can earn 5% a year. What is that promise worth today?

Divide the future \$100 by `(1.05)³`:

- `(1.05)³` = 1.05 × 1.05 × 1.05 = 1.157625
- Present value = \$100 ÷ 1.157625 = **\$86.38**

Check it the other way: \$86.38 invested today at 5% grows to \$86.38 × 1.157625 ≈ \$100.00 in three years. The two are the same amount of money, just observed at different times.

Notice how the shrinkage depends on both knobs. At a 5% rate over 3 years, \$100 shrinks to \$86.38. Push the rate to 8% and it shrinks to \$100 ÷ 1.08³ = \$79.38. Stretch the wait to 10 years at 5% and it shrinks to \$100 ÷ 1.05¹⁰ = \$61.39. *The further out the money and the higher the rate, the less a future dollar is worth today — and that single sentence is why long bonds fall hardest when rates rise.*

### The discount factor: the price of one future dollar

It's worth isolating the shrinking machine into its own object, because it's the cleanest way to think about pricing. The *discount factor* is the present value of exactly **one** dollar received `n` periods from now, at rate `r`:

$$ DF(n) = \frac{1}{(1 + r)^n} $$

The discount factor is always a number between 0 and 1 (for positive rates): one future dollar is always worth less than one present dollar. At 5%:

| Years out `n` | Discount factor `1/(1.05)^n` | One future dollar is worth today |
|---|---|---|
| 0 | 1.0000 | \$1.0000 |
| 1 | 0.9524 | \$0.9524 |
| 2 | 0.9070 | \$0.9070 |
| 3 | 0.8638 | \$0.8638 |
| 5 | 0.7835 | \$0.7835 |
| 10 | 0.6139 | \$0.6139 |
| 30 | 0.2314 | \$0.2314 |

Read the bottom row: at 5%, a dollar promised 30 years out is worth just 23 cents today. The discount factor is the single most useful object in fixed income, because **once you know the discount factor for each future date, pricing any stream of cash flows is just multiply-and-add.** You take each payment, multiply by its discount factor, and sum. We'll lean on this table for the rest of the post.

There's a clean way to read the discount factor that demystifies it: `DF(n)` is the price *today* of a contract that pays exactly \$1 in `n` years. If someone offered to sell you a slip of paper guaranteeing \$1 in three years, and money earns 5%, the fair price for that slip is 86.38 cents — anything more and you'd do better just investing the cash yourself; anything less and you'd snap it up. So the discount-factor table is really a *price list for future dollars*, one row per delivery date. A bond is then just a shopping basket of future dollars (some in year 1, some in year 2, a big pile in the final year), and its price is the basket totaled up at those per-date prices. Holding that picture — future dollars have posted prices, and a bond is a basket of them — makes everything that follows feel less like algebra and more like arithmetic at a checkout counter.

![A curve showing how the present value of one future dollar falls as the number of years increases, drawn for a five percent and an eight percent discount rate](/imgs/blogs/discounting-cash-flows-how-a-bond-is-priced-2.png)

The figure above is the discount factor curve — how that single future dollar shrinks as you push it further into the future. Two things to notice. First, it *decays*: each extra year of waiting takes another bite, but the bites get smaller (the curve flattens). Second, the higher line is the gentler 5% rate and the lower line is the steeper 8% rate — **a higher discount rate pulls the whole curve down**, shrinking every future dollar harder. Hold that picture; it is the reason the price-versus-rate relationship we build later slopes downward.

### Present value of a stream: just add up the pieces

A bond doesn't make one payment — it makes many. So we need the present value not of a single future dollar but of a *whole schedule* of them. The good news: present value is additive. The value today of getting \$50 next year *and* \$50 the year after *and* \$1,050 the year after that is just the sum of the three present values, computed one at a time.

$$ PV = \frac{C_1}{(1+r)^1} + \frac{C_2}{(1+r)^2} + \cdots + \frac{C_n}{(1+r)^n} $$

- **`C_t`** — the cash flow paid at time `t` (a coupon, or a coupon plus principal).
- **`r`** — the discount rate per period.
- **`n`** — the final period, when the bond matures.

That sum *is* the price of the bond. Everything else in this post is unpacking that one equation, term by term, with real numbers.

Two properties of present value are worth naming because we'll use them constantly. First, **present value is linear**: the value of a combined stream equals the sum of the values of its parts, and doubling every cash flow doubles the present value. That's why we can chop a bond into pieces, value each, and add — and why a coupon bond turns out to be a bundle of zero-coupon bonds. Second, **present value is monotonic in the rate**: hold the cash flows fixed and raise the discount rate, and the present value of *every* positive future cash flow falls, so the total falls. There are no exceptions for a plain bond (every cash flow is positive), which is exactly why the price-versus-rate curve never turns around — it slopes down everywhere. Keep both facts in mind; together they explain most of how bonds behave.

### A bond, defined from zero

Quickly, so no term sneaks past undefined. A *bond* is a tradable loan. You (the buyer/lender) hand the issuer money today; in return the issuer promises a fixed schedule of payments. The standard contract has three numbers:

- **Face value** (or *par value*, or *principal*) — the amount repaid at the end, almost always \$1,000 per bond by convention. This is the "loan amount" that comes back at maturity.
- **Coupon rate** — the annual interest rate the bond pays on its face value. A 5% coupon on a \$1,000 face means \$50 of interest per year, paid as *coupons*. (The word comes from old paper bonds with detachable coupons you clipped and mailed in to collect interest.)
- **Maturity** — the date the loan ends, when the final coupon and the full face value are paid back.

For clarity we'll use *annual* coupons throughout this post — one payment a year — so the arithmetic stays clean. Real Treasury notes and most corporate bonds pay *semiannually* (twice a year), which we handle in the next post on price-and-rate mechanics. The logic is identical; you just discount twice as many, half-sized payments. (The full contract — indentures, day-counts, call features — is dissected in [the anatomy of a bond](/blog/trading/fixed-income/anatomy-of-a-bond-par-coupon-maturity-issuer).)

So our running example bond promises this: a 3-year, \$1,000 par note with a 5% annual coupon pays you \$50 at the end of year 1, \$50 at the end of year 2, and \$50 + \$1,000 = \$1,050 at the end of year 3. That's the whole contract. Three cash flows. Now we price it.

## Pricing a bond is discounting its cash flows and adding them up

Here is the central claim of the entire post, stated as plainly as possible:

> **The price of a bond is the present value of every cash flow it will ever pay, discounted at the rate the market currently demands.**

Nothing more. Let's do it with the running example. First we need a discount rate. The discount rate is the *return the market currently requires* to lend to this issuer for this term — we'll call it the *yield* and unpack where it comes from shortly. For now, set the discount rate equal to the coupon rate, 5%, and watch what happens.

#### Worked example: pricing the 3-year 5% bond at a 5% discount rate

The bond pays \$50, \$50, and \$1,050 at years 1, 2, 3. Discount each at 5% using our discount factors from the table above, then sum.

| Year `t` | Cash flow `C_t` | Discount factor `1/(1.05)^t` | Present value `C_t × DF` |
|---|---|---|---|
| 1 | \$50 | 0.95238 | \$47.62 |
| 2 | \$50 | 0.90703 | \$45.35 |
| 3 | \$1,050 | 0.86384 | \$907.03 |
| **Price** | | | **\$1,000.00** |

Add the present-value column: \$47.62 + \$45.35 + \$907.03 = **\$1,000.00**. The bond is worth exactly its \$1,000 face value. When the discount rate equals the coupon rate, the bond prices at *par*. *This is the anchor of the whole subject: a bond at par is one whose promised interest rate exactly matches what the market demands, so there's no premium or discount to pay.*

![A timeline of the three-year bond showing each cash flow discounted back to today and summed to the price of one thousand dollars](/imgs/blogs/discounting-cash-flows-how-a-bond-is-priced-3.png)

The figure above is the whole operation in one picture: three future cash flows up the timeline (\$50, \$50, \$1,050 in green, the money you receive), each pulled back to today by its discount factor, and the three present values stacked up at `t = 0` to make the price (in blue, the bond — the thing being valued). Read it left-to-right as time, and read the diagonal pull-backs as discounting. The price on the left is *literally* the sum of those three shrunken payments. There is no separate "price formula" hiding behind this; the picture is the formula.

Why does it land exactly on \$1,000? Intuition: if you buy this bond for \$1,000 and the market rate is 5%, you're earning 5% on your money (the \$50 annual coupon is 5% of \$1,000), which is exactly the market rate. You're getting paid the going rate, no more, no less — so the fair price is exactly the face value. There's nothing to discount away and nothing to add. Par is the equilibrium.

### Now turn the dial: pricing at a 6% discount rate

Suppose interest rates in the wider economy rise. Newly issued 3-year bonds of the same quality now pay a 6% coupon. Our old bond still only pays \$50 a year — its contract is fixed. Nobody will pay \$1,000 for a bond paying \$50 when they could buy a fresh one paying \$60 for the same \$1,000. So our bond's *price* must fall until its return matches the new 6% the market demands. We find the new price by discounting the *same* cash flows at the *new, higher* rate.

#### Worked example: re-pricing the same bond at a 6% discount rate

Same \$50, \$50, \$1,050 cash flows. New discount factors at 6%: `1/1.06 = 0.94340`, `1/1.06² = 0.89000`, `1/1.06³ = 0.83962`.

| Year `t` | Cash flow `C_t` | Discount factor `1/(1.06)^t` | Present value `C_t × DF` |
|---|---|---|---|
| 1 | \$50 | 0.94340 | \$47.17 |
| 2 | \$50 | 0.89000 | \$44.50 |
| 3 | \$1,050 | 0.83962 | \$881.60 |
| **Price** | | | **\$973.27** |

The sum is \$47.17 + \$44.50 + \$881.60 = **\$973.27**. The same bond, same promised payments, is now worth about \$973 instead of \$1,000 — it trades at a *discount* to par. The price fell by \$26.73, or about 2.7%, because we discounted every cash flow at a higher rate, shrinking each one a little harder. *A bond paying less than the market demands must sell below face value, and exactly how far below is just the present-value arithmetic at the higher rate.*

This is the seesaw at the heart of fixed income, and now you can see *why* it exists: it isn't a rule someone imposed, it's a direct consequence of dividing by `(1 + r)`. Raise `r`, every denominator gets bigger, every present value gets smaller, the sum (the price) falls. Lower `r`, prices rise. (The full mechanics of the seesaw — why a 1% rate move is *not* a 1% price move, and how to measure the sensitivity — get their own treatment in [price and yield, the seesaw at the heart of bonds](/blog/trading/fixed-income/price-and-yield-the-seesaw-at-the-heart-of-bonds).)

#### Worked example: pricing at a 4% discount rate — the premium case

For completeness, run the dial the other way. Rates *fall* to 4%, so a bond paying \$50 a year now pays *more* than the market demands. Discount factors at 4%: `0.96154`, `0.92456`, `0.88900`.

| Year `t` | Cash flow `C_t` | Discount factor `1/(1.04)^t` | Present value `C_t × DF` |
|---|---|---|---|
| 1 | \$50 | 0.96154 | \$48.08 |
| 2 | \$50 | 0.92456 | \$46.23 |
| 3 | \$1,050 | 0.88900 | \$933.45 |
| **Price** | | | **\$1,027.75** |

Now the bond is worth **\$1,027.75** — a *premium* of about \$27.75 over par. It pays more interest than newly issued bonds, so buyers will pay extra for it. *The same three cash flows priced at three different rates give three different prices — par at 5%, a discount at 6%, a premium at 4% — and the only thing that changed was the discount rate.*

Three prices for one set of promises. The cash flows never moved. The discount rate did all the work. Let that sink in: **the bond's price is a function of the discount rate, and almost nothing else in the short run.**

## The discount rate is the master dial

We've now priced the same bond at 4%, 5%, and 6% and gotten \$1,027.75, \$1,000.00, and \$973.27. If we kept turning the dial and plotted price against rate, we'd trace a curve — and that curve is the single most important relationship in the bond market.

![A downward-sloping convex curve plotting the bond price against the discount rate, marking par at five percent and the discount and premium prices](/imgs/blogs/discounting-cash-flows-how-a-bond-is-priced-4.png)

This is the influence figure — the proof that price is a *function* of the rate. The horizontal axis is the discount rate; the vertical axis is the resulting price of our 3-year 5% bond. The curve slopes **down**: higher rate, lower price, always. It crosses par (\$1,000) exactly where the rate hits the 5% coupon — the point we computed by hand. To its left (lower rates) the bond trades at a premium; to its right (higher rates), at a discount. And the curve is **convex** — it bows toward the origin, falling fast at first and then flattening, which is a deep property we'll only name here (it's called *convexity*) and develop fully in a later post. For now the headline is simple: **the discount rate is the dial, and the price is what the dial reads.**

#### Worked example: how far the price moves for each 1% of rate

Let's quantify the dial's sensitivity with our own numbers. Going from 5% to 6% (a 1-percentage-point, or 100-*basis-point*, rise — a *basis point* is one hundredth of a percent, 0.01%), the price fell from \$1,000.00 to \$973.27, a drop of \$26.73 or **2.67%**. Going from 5% to 4% (a 100 bp fall), the price rose from \$1,000.00 to \$1,027.75, a gain of \$27.75 or **2.78%**.

Two things jump out. First, a 1% change in rates moved the price by roughly 2.7% — *not* 1%. The price sensitivity is bigger than the rate change, and how much bigger depends on the bond's maturity and coupon (that multiplier is called *duration*, the subject of [duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income)). Second, the down-move (\$26.73) and the up-move (\$27.75) aren't quite equal — the gain is a touch bigger than the loss. That asymmetry *is* convexity, the curve's bow, and it's a small gift to the bondholder. *The discount rate doesn't move the price one-for-one; it moves it by a leveraged, slightly asymmetric amount, and pinning down that amount is what the rest of interest-rate risk is about.*

### Where does the discount rate come from?

We've been turning the rate like it's ours to set, but in a real market the discount rate is *given to you* by the market, and it has structure. It's worth pausing on what the discount rate actually *is*, because beginners often picture it as some arbitrary number an analyst chooses. It isn't. The discount rate is the *opportunity cost* of the money — the return you could earn on a comparable alternative. If you can earn 5% lending to equally risky borrowers, then you should discount this bond's cash flows at 5%, because 5% is what your money is worth elsewhere. That's why the market, not the analyst, sets it: the discount rate is just the going price of capital for that risk and that horizon, and it's quoted continuously in the prices of every other bond. Stripped down, the rate you should discount a bond's cash flows at is built from a few layers:

- **The real risk-free rate** — the return on money with no default risk and no inflation, the pure rent on capital.
- **Expected inflation** — you need compensation for the fact that future dollars buy less. Real rate + expected inflation ≈ the *nominal* risk-free rate, which for the world is the US Treasury yield of matching maturity. (The real-versus-nominal split is its own master signal — see [real vs nominal yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).)
- **A credit spread** — extra yield demanded if the issuer might *default* (fail to pay). A US Treasury has ~none; a company has some. The riskier the issuer, the higher the spread, the higher the discount rate, the lower the price.
- **A term/liquidity premium** — extra for tying your money up longer or in a harder-to-sell bond.

For a US Treasury, the discount rate is essentially just the risk-free yield for that maturity. For a corporate bond like our fictional issuer **Northwind Corp**, you add a credit spread on top.

#### Worked example: the same cash flows, two issuers, two prices

Imagine two 3-year \$1,000 bonds, each with a 5% coupon. One is a US Treasury; the other is issued by Northwind Corp, a solid but not bulletproof company. The Treasury discounts at the risk-free 5%. Northwind, because it might default, discounts at 5% + a 1.5% credit spread = 6.5%.

- **Treasury** at 5%: prices at par, **\$1,000.00** (we computed this).
- **Northwind** at 6.5%: discount factors `0.93897`, `0.88166`, `0.82785`; PVs of \$46.95 + \$44.08 + \$869.24 = **\$960.27**.

Identical promised payments, but Northwind's bond is worth about \$40 less *purely because the market demands a higher discount rate for the risk of not getting paid back.* The credit spread is a discount-rate add-on, and it works through the exact same present-value arithmetic. *Risk doesn't change what a bond promises — it changes the rate you discount those promises at, and that's the whole mechanism behind credit pricing.* (Credit risk and spreads get built from scratch in [credit risk: the chance you don't get paid back](/blog/trading/fixed-income/credit-risk-the-chance-you-dont-get-paid-back).)

### A shortcut for the intuition: the perpetuity

There's a special case that makes the discount rate's grip on price almost shockingly clear. A *perpetuity* is a bond that pays a fixed coupon forever and never repays principal — an infinite stream of coupons. It sounds exotic, but these existed: the British government's *consols*, first issued in the 1750s, paid a fixed coupon with no maturity for over 250 years. The price of a perpetuity collapses to a one-line formula:

$$ P = \frac{C}{r} $$

- **`C`** — the fixed annual coupon.
- **`r`** — the discount rate.

A perpetuity paying \$50 a year, discounted at 5%, is worth \$50 ÷ 0.05 = \$1,000. Now turn the dial. At 4%, it's worth \$50 ÷ 0.04 = \$1,250. At 6%, it's worth \$50 ÷ 0.06 = \$833.33. At 2.5%, it doubles to \$2,000. With no principal and no maturity to anchor it, the perpetuity's price is *purely* a function of the discount rate — and you can see the inverse relationship laid bare: halve the rate and the price doubles. A real coupon bond is a "tamer" version of this, because its principal repayment at a fixed date pins down part of its value, but the same inverse force is at work. *The perpetuity strips bond pricing down to a single division — price equals coupon over rate — and shows the price-rate seesaw in its purest, most violent form.*

### A shortcut for the coupons: the annuity factor

Pricing a 30-year bond by hand would mean 30 separate discount-and-add steps. There's a closed-form shortcut, because the stream of equal coupons is an *annuity* — a fixed payment for a fixed number of periods. The present value of \$1 per period for `n` periods at rate `r` is the *annuity factor*:

$$ A(n, r) = \frac{1 - (1+r)^{-n}}{r} $$

So a bond's price splits cleanly into two pieces: the coupons (an annuity) plus the principal (a single discounted lump sum):

$$ P = C \times A(n, r) + \frac{F}{(1+r)^n} $$

For our 3-year 5% bond at 5%: the annuity factor is `(1 − 1.05⁻³) / 0.05 = (1 − 0.863838) / 0.05 = 2.72325`. So the coupons are worth \$50 × 2.72325 = \$136.16, and the principal is worth \$1,000 × 0.86384 = \$863.84. Add them: \$136.16 + \$863.84 = **\$1,000.00**, the same par price we got the long way. *The annuity factor is just the discount-and-add for the coupons done once in closed form; it turns a 30-line table into two multiplications.*

## Why later cash flows contribute less

Look back at the par-pricing table. The year-1 coupon of \$50 contributed \$47.62 to the price; the identical year-2 coupon of \$50 contributed only \$45.35; and the \$50 *coupon* portion of the year-3 payment contributed even less. The same \$50, promised later, is worth less today — because it's discounted by a bigger power of `(1 + r)`. Distance in time is itself a discount.

![Bars showing the present value contribution of each year's cash flow, with the small early coupons and the dominant final principal payment](/imgs/blogs/discounting-cash-flows-how-a-bond-is-priced-5.png)

The figure above breaks the \$1,000 price into where it comes from. Two tiny green bars are the year-1 and year-2 coupons (\$47.62 and \$45.35 — money you receive). The towering blue bar is the year-3 payment of \$1,050, worth \$907.03 today, because it carries the principal back. Two lessons. First, **for a normal coupon bond, the principal repayment dominates the price** — over 90% of this bond's value is that final \$1,050. The coupons are the seasoning; the return of principal is the meal. Second, the coupon bars *shrink* as you move out in time, which is the time value of money made visual: later money is worth less.

#### Worked example: the same coupon, worth less each year

Trace one \$50 coupon across the years at 5%, just to feel the decay:

- Year 1: \$50 ÷ 1.05 = \$47.62
- Year 2: \$50 ÷ 1.05² = \$45.35
- Year 3: \$50 ÷ 1.05³ = \$43.19
- Year 10: \$50 ÷ 1.05¹⁰ = \$30.70
- Year 30: \$50 ÷ 1.05³⁰ = \$11.57

The identical promised \$50 is worth \$47.62 if paid next year but only \$11.57 if you have to wait 30 years for it. *This is why long-dated bonds are so sensitive to rates: a huge share of their value sits in distant cash flows, and distant cash flows are exactly the ones a higher discount rate shrinks the most.* A 30-year bond's price reacts violently to rate moves precisely because its big principal payment is 30 powers of `(1 + r)` away.

This decay also explains a subtlety that confuses many beginners. People assume a bond's coupon is "where the return comes from." But for a long bond, most of the *value* is the principal, and most of the *price movement* comes from re-discounting that distant principal when rates change. The coupon is almost a rounding error in the price dynamics of a 30-year bond. Keep this in your pocket; it pays off when we get to duration.

#### Worked example: a 30-year bond, where the value really sits

Take a 30-year \$1,000 bond with a 5% annual coupon (\$50/year), priced at a 5% discount rate so it sits at par, \$1,000. Where does that \$1,000 of value come from? Split it the way we just learned, into coupons and principal:

- **The principal**, a single \$1,000 paid 30 years out, is worth \$1,000 ÷ 1.05³⁰ = \$1,000 × 0.2314 = **\$231.38** today.
- **The 30 coupons**, an annuity of \$50 for 30 years at 5%, are worth \$50 × [(1 − 1.05⁻³⁰) / 0.05] = \$50 × 15.3725 = **\$768.62** today.

Add them: \$231.38 + \$768.62 = \$1,000.00. So for a long bond the *coupons* dominate the value (about 77%), unlike the short bond where the principal dominated. But here's the twist that matters for risk: that \$231 of principal value, sitting 30 powers of `(1 + r)` away, is the most rate-sensitive piece of the whole bond. Re-discount it at 6% and it drops to \$1,000 ÷ 1.06³⁰ = \$174.11 — a 25% fall in that one chunk — while the year-1 coupon barely moves. *The cash flows farthest in the future carry the least value but the most risk, which is the seed of everything duration measures.*

## The zero-coupon bond: the atom of fixed income

We can now meet the simplest bond of all, and the most important one for understanding pricing: the *zero-coupon bond*. A zero pays **no coupons at all** — just a single payment of face value at maturity. You buy it for less than \$1,000 today and collect \$1,000 at the end. The entire return is the gap between the discounted purchase price and the \$1,000 you eventually get. There is exactly one cash flow, discounted exactly once.

![A timeline of a zero-coupon bond showing a single discounted purchase today and one face-value payment at maturity with no coupons in between](/imgs/blogs/discounting-cash-flows-how-a-bond-is-priced-6.png)

The figure above is the cleanest cash-flow diagram in finance: one outflow today (you pay the discounted price, in red), nothing in between, one inflow at maturity (you collect \$1,000 of face, in green). No coupons clutter the middle. The zero is pricing distilled to its essence — a single discount factor applied to a single payment.

#### Worked example: pricing a 3-year zero-coupon bond

A 3-year zero-coupon bond pays \$1,000 at maturity and nothing before. At a 5% discount rate, its price is just one application of the discount factor:

$$ \text{Price} = \frac{\$1{,}000}{(1.05)^3} = \$1{,}000 \times 0.86384 = \$863.84 $$

You pay \$863.84 today and receive \$1,000 in three years. Your return is built entirely from the \$136.16 of price *accretion* — the price climbing toward face value as maturity approaches. There are no coupons to reinvest, which makes a zero's return perfectly predictable if you hold it to maturity. *A zero-coupon bond is just one discount factor wearing a price tag; learn to price the zero and you can price anything.*

### Every coupon bond is a bundle of zeros

Here's the punchline that ties the whole post together, and it's genuinely beautiful. **A coupon bond is nothing but a portfolio of zero-coupon bonds.** Our 3-year 5% bond — \$50, \$50, \$1,050 — is exactly the same as owning:

- a 1-year zero paying \$50,
- a 2-year zero paying \$50, and
- a 3-year zero paying \$1,050.

Price each little zero separately and add them up, and you get the bond's price. This is the "additive present value" idea wearing its formal clothes.

#### Worked example: rebuilding the bond price from three zeros

Price each cash flow as its own zero at 5%, then sum:

| Zero | Pays | Maturity | Price `pay × DF` |
|---|---|---|---|
| 1-year zero | \$50 | 1y | \$50 × 0.95238 = \$47.62 |
| 2-year zero | \$50 | 2y | \$50 × 0.90703 = \$45.35 |
| 3-year zero | \$1,050 | 3y | \$1,050 × 0.86384 = \$907.03 |
| **Total** | | | **\$1,000.00** |

It's the *exact same arithmetic* as the par-pricing table earlier — because it's the same thing seen from a different angle. The bond is the bundle; the price is the sum of the parts. *This is why zeros are the atom of the market: any bond, any stream of fixed payments, is a basket of zeros, and the price of the basket is the sum of the prices of its atoms.*

This decomposition isn't just a teaching trick. It's how the market actually works. The US Treasury runs a program called STRIPS that literally separates a coupon bond into its individual cash flows and lets each trade as its own zero. And it foreshadows the next two posts in this track: if each cash flow could in principle be discounted at its *own* rate (a 1-year rate for the year-1 coupon, a different 3-year rate for the principal), then using one single rate for all of them is an approximation. Untangling that — the *spot* or *zero* curve — is exactly where we go next, in [spot rates, the zero curve, and bootstrapping](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping).

## The pricing formula, assembled

Let's put the whole machine in one place, now that every piece is intuitive. The price of any bond is:

$$ P = \sum_{t=1}^{n} \frac{C_t}{(1+r)^t} = \frac{C}{(1+r)^1} + \frac{C}{(1+r)^2} + \cdots + \frac{C + F}{(1+r)^n} $$

- **`P`** — the price (present value of all cash flows).
- **`C`** — the coupon paid each period (coupon rate × face value).
- **`F`** — the face value, repaid at maturity (added to the last coupon).
- **`r`** — the discount rate per period (the yield).
- **`n`** — the number of periods until maturity.

![A pipeline showing the four stages of bond pricing from cash flows to discount factors to present values to the summed price](/imgs/blogs/discounting-cash-flows-how-a-bond-is-priced-7.png)

The figure above is the formula as an assembly line, the way to remember it without memorizing symbols. **Stage 1:** list the cash flows the bond promises (\$50, \$50, \$1,050). **Stage 2:** compute a discount factor for each date from the rate (`1/(1.05)^t`). **Stage 3:** multiply each cash flow by its discount factor to get its present value (\$47.62, \$45.35, \$907.03). **Stage 4:** sum the present values — that sum *is* the price (\$1,000). Cash flows in, price out, with the discount factor as the conversion step. Every bond ever priced runs down this same pipeline.

If you can read this pipeline, you can price a bond, and you understand the deepest fact about the bond market: **price and yield are two ways of saying the same thing.** Give me the cash flows and a yield, I compute the price. Give me the cash flows and a price, I can solve (iteratively) for the yield that makes the formula balance — that's the *yield to maturity*, the single rate that, plugged in as `r`, reproduces the market price. The whole quoting convention of the bond market — quoting a bond by its yield rather than its dollar price — is just running this pipeline backward. (For the formal machinery of solving for yield, and the heavier math of analytics, see [bond pricing](/blog/trading/quantitative-finance/bond-pricing) and [fixed-income analytics](/blog/trading/quantitative-finance/fixed-income-analytics).)

#### Worked example: pricing a 5-year Northwind note from scratch

One full from-scratch run to cement it. Northwind Corp issues a 5-year \$1,000 note with a 4% annual coupon (\$40/year), and the market demands a 5% yield to lend to Northwind for 5 years. Price it.

Cash flows: \$40 at years 1–4, and \$40 + \$1,000 = \$1,040 at year 5. Discount each at 5%:

| Year | Cash flow | DF at 5% | Present value |
|---|---|---|---|
| 1 | \$40 | 0.95238 | \$38.10 |
| 2 | \$40 | 0.90703 | \$36.28 |
| 3 | \$40 | 0.86384 | \$34.55 |
| 4 | \$40 | 0.82270 | \$32.91 |
| 5 | \$1,040 | 0.78353 | \$814.87 |
| **Price** | | | **\$956.71** |

The note prices at **\$956.71**, a discount to its \$1,000 face — because its 4% coupon is below the 5% the market demands, so it must sell cheap enough that a buyer earns the full 5%. The \$43.29 discount is exactly the present-value shortfall from underpaying \$10 of coupon (4% vs 5% on \$1,000) each year for five years. *Run any bond down the four-stage pipeline and the price falls out — the formula is long, but it's just one division repeated and summed.*

## Common misconceptions

**"A bond's price is its face value."** No — face value is what you get back *at maturity*. The price is what the bond trades for *today*, and it equals the present value of all remaining cash flows. Price equals face value only in the special case where the discount rate happens to equal the coupon rate (par). The rest of the time the bond trades at a premium or a discount, and a 30-year bond can trade at 60 cents on the dollar or \$1.40 on the dollar depending on rates.

**"If I hold to maturity, the price doesn't matter."** The price you *paid* matters enormously — it determines your return. Two people who buy the same bond at \$1,000 and \$960 earn very different yields even holding to the same maturity. What's true is that the *daily price swings* in between don't change your outcome if you hold to maturity and the issuer doesn't default; you'll still get the contracted cash flows. But the entry price is locked into your return forever.

**"Higher coupon always means a better bond."** A higher coupon means bigger payments, but the *price adjusts* so that two equally risky bonds offer the same yield. A 6% bond will simply cost more (trade at a premium) than a 4% bond of the same maturity and risk. You don't get extra return from a high coupon — you pay for it up front. What the coupon *does* change is the *shape* of your cash flows and your sensitivity to rates (lower-coupon bonds have more of their value in the distant principal, so they're more rate-sensitive).

**"The discount rate is the coupon rate."** These are different animals that only coincide at par. The *coupon rate* is fixed in the bond's contract and never changes. The *discount rate* (the yield) is whatever the market currently demands, and it moves every day. Pricing is the act of discounting the fixed coupons at the moving market rate. Confusing the two is the single most common beginner error.

**"You discount with one rate because that's the right rate."** Using a single rate `r` for every cash flow is a convenient simplification (it's what "yield to maturity" does), but the *honest* picture discounts each cash flow at its own maturity-specific *spot rate*. The single yield is a kind of weighted average of those spot rates. We use one rate here to build intuition; the multi-rate truth is the subject of the spot-curve post.

**"Longer bonds are riskier because they default more."** Maturity risk and default risk are different. A 30-year Treasury has essentially zero default risk but enormous *price* risk — its distant cash flows are hammered when rates rise. A 2-year junk bond has little price sensitivity but real default risk. Discounting separates these cleanly: maturity shows up in the *powers* of `(1 + r)`; default risk shows up in the *level* of `r` (via the credit spread).

**"A zero-coupon bond pays no interest, so it earns nothing."** A zero pays no *coupons*, but it absolutely earns interest — the interest is baked into the price. You buy it below face value and it accretes up to \$1,000 at maturity, and that accretion *is* your interest, compounding silently inside the price. Our 3-year zero bought at \$863.84 and redeemed at \$1,000 earned exactly 5% a year; the IRS even taxes that built-in growth annually as "phantom income" even though you receive no cash until maturity. The absence of a coupon doesn't mean the absence of a return — it means the return arrives all at once, at the end.

**"If a bond's price is below par, the issuer is in trouble."** Not necessarily. A bond trades below par whenever its coupon is lower than the current market rate — which happens to perfectly healthy issuers all the time, simply because rates rose after the bond was issued. A pristine US Treasury issued in 2020 with a 1% coupon trades far below par today, and it's the safest credit on earth. A *discount* reflects the rate environment, not the issuer's health; a *credit-driven* price drop (a widening spread) is the one that signals trouble, and you tell them apart by whether the bond's yield moved with the whole market or jumped on its own.

## How it shows up in real markets

**The 2022 bond rout — discounting in real time.** When the US Federal Reserve raised its policy rate from near 0% to over 4% across 2022, the discount rate on every bond in the world jumped. Run the pricing formula with a bigger `r` and prices fall — and they fell hard. The Bloomberg US Aggregate Bond Index lost roughly 13% in 2022, its worst year on record, and long-dated Treasuries fell far more (the 20-year-plus Treasury ETF lost around 30%). Nothing about the bonds' promised cash flows changed; the only thing that moved was the discount rate, and the present-value math did the rest. This is the price-versus-rate curve from Figure 4, lived. (The cross-asset view of that year — when stocks fell too — is in [the 2022 case study](/blog/trading/cross-asset/case-study-2022-stocks-and-bonds-both-fell).)

**Silicon Valley Bank, 2023 — discounting can sink a bank.** SVB had bought a large book of long-dated Treasuries and mortgage bonds in 2020–2021 when rates were near zero. When rates spiked in 2022, the present value of those bonds fell sharply — exactly the re-discounting mechanism in this post, applied to distant cash flows that shrink the most. On paper the bonds were still "money-good" (no default risk), but their *market value* had dropped by billions because the discount rate had risen. When depositors pulled their money and SVB had to sell those bonds before maturity, the paper losses became real losses, and the bank failed in days. Discounting isn't an abstraction; it's what determined whether SVB was solvent. (The episode is dissected in [SVB and Credit Suisse, the 2023 bank runs](/blog/trading/finance/svb-credit-suisse-2023-bank-runs).)

**Treasury STRIPS — the market sells the atoms.** The US Treasury's STRIPS program ("Separate Trading of Registered Interest and Principal of Securities") does literally what this post describes: it strips a coupon bond into its individual cash flows and lets each trade as its own zero-coupon bond. A 30-year Treasury becomes 60 semiannual coupon-strips plus one principal-strip, each priced as a single discounted payment. Pension funds and insurers love long-dated principal STRIPS because a single far-future zero is the purest way to match a single far-future liability — one discount factor, one payment, no reinvestment to worry about. The "bundle of zeros" idea is a real, multi-trillion-dollar market.

**US Savings Bonds and T-bills — zeros you already own.** A US Treasury bill is a zero-coupon bond: you buy a 1-year bill for, say, \$952 and receive \$1,000 at maturity, with the \$48 gap as your entire return. Older Series EE savings bonds worked the same way — bought at half face value and accreting to face over time. Every time you've seen a "buy at \$X, get \$1,000 later" instrument, you've met the discount factor in the wild. The price was just \$1,000 times the discount factor for that horizon.

**Zero rates and the discount factor going haywire — 2020.** In March 2020, as the pandemic hit, short-term Treasury yields briefly went *negative* — investors paid more than \$1,000 for a bill that would return \$1,000, accepting a small guaranteed loss for the safety of holding Treasuries in a panic. Plug a negative `r` into the discount factor and it climbs *above* 1: a future dollar becomes worth *more* than a present dollar. It's the same formula, just with the dial turned below zero — a vivid reminder that the discount factor is a mechanical consequence of the rate, not a law that rates must be positive.

**Your mortgage is a bond, priced the same way.** When you take a 30-year fixed mortgage, you become the *issuer* of a bond and the bank is the buyer of it. Your monthly payments are the coupon-and-principal stream; the bank "prices" your loan by discounting those payments at a rate built on the bond market (specifically the yield on mortgage-backed securities, which tracks the 10-year Treasury plus a spread). This is why mortgage rates rise the day after Treasury yields rise: the discount rate on the cash flows you promise just went up, so the price the bank will pay for your loan went down — and it passes that through as a higher rate to you. The present-value math you did on a 3-year note is the same math that sets the biggest number on your mortgage statement. *Discounting isn't a Wall Street abstraction; it's the rule that decides what your house payment is.* The full chain is traced in [from the ten-year yield to your mortgage](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates).

## When this matters to you, and where to go next

This single idea — price equals the present value of future cash flows, discounted at the rate the market demands — is the foundation under almost everything in finance. It prices your mortgage (the bank discounts your future payments), your pension (the fund discounts its future obligations), and even stocks (a share is, in theory, the present value of future dividends, discounted at a rate built on the bond market's). The bond market sets the discount rate, and the discount rate sets every price. That's the influence thread of this whole series: bonds are the price of money, and discounting is how that price reaches everything else. The chain from the 10-year Treasury yield all the way to the rate on your home loan is traced in [from the ten-year yield to your mortgage](/blog/trading/fixed-income/from-the-ten-year-yield-to-your-mortgage-the-transmission-of-rates).

From here, the next three posts build directly on this pricing engine. [Why bond prices move when rates move — and by how much](/blog/trading/fixed-income/why-bond-prices-move-when-rates-move-and-by-how-much) quantifies the seesaw and adds the real-world wrinkles of clean versus dirty price and accrued interest. [Spot rates, the zero curve, and bootstrapping](/blog/trading/fixed-income/spot-rates-the-zero-curve-and-bootstrapping) replaces our single discount rate with a whole curve of maturity-specific rates — the honest way to discount. And [duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income) turns the price-versus-rate curve into a single sensitivity number you can trade and hedge with.

This is educational material, not investment advice. But the next time you read that "bond prices fell as yields rose," you'll know exactly what happened: someone turned the discount-rate dial, and the present-value arithmetic — the same arithmetic you just did by hand on a 3-year \$1,000 note — did the rest.
