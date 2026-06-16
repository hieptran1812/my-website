---
title: "The Time Value of Money: Discounting, the Engine Under Every Valuation"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A dollar today is worth more than a dollar next year, and exactly how much more is the one idea that powers every valuation; this post builds discounting, compounding, perpetuities, and the discount rate from zero so DCF, bond pricing, and multiples all become obvious."
tags: ["equity-research", "corporate-finance", "time-value-of-money", "discounting", "present-value", "discount-rate", "gordon-growth", "perpetuity", "compounding", "valuation"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A dollar today is worth more than a dollar next year, and discounting is the arithmetic that tells you *exactly* how much more; master this one idea and every valuation method in finance becomes a variation on it.
>
> - **Compounding** runs money forward: \$1 invested at a rate `r` grows to `(1+r)^n` after `n` years. **Discounting** runs it backward: a dollar arriving in year `n` is worth `1/(1+r)^n` today. They are the same machine driven in opposite directions.
> - The **present value** of any future cash flow is `PV = FV / (1+r)^n`. The value of a business is just the present value of *all* the cash it will ever hand its owners, added up.
> - The **discount rate** `r` is your **required return** — the return you could earn elsewhere for the same risk. It bundles a risk-free rate plus a risk premium, and it is the single most powerful lever in any valuation.
> - A **perpetuity** (a cash flow forever) is worth `CF / r`; a **growing perpetuity** is worth `CF / (r − g)`. That tiny `(r − g)` denominator is the workhorse of terminal value — and a loaded gun, because as `g` creeps toward `r` the value explodes toward infinity.
> - A **higher discount rate crushes far-off cash flows far more than near ones**, which is exactly why rising interest rates hammer long-duration growth stocks hardest. This is the intuition under every "rates up, growth down" headline.

Here is a question that sounds like a riddle but is the most important question in all of finance: *would you rather have \$1,000 today or \$1,000 a year from now?* Almost everyone answers "today," and almost everyone is right — but most people cannot say precisely *why*, or *how much* the difference is worth. That gap, between knowing a dollar today beats a dollar tomorrow and being able to put an exact number on the difference, is the entire subject of this post. Close that gap and you have the engine under every valuation method ever invented.

Because here is the thing the textbooks bury under formulas: a company is worth the cash it will hand its owners over its whole life. But that cash arrives over decades — a little next year, a little the year after, a stream stretching out toward the horizon. You cannot just add those dollars up, because a dollar arriving in 2046 is simply not worth as much *to you, today* as a dollar in your pocket right now. To turn a stream of future cash into one honest number — what the business is worth *today* — you need a way to translate every future dollar back into today's money. That translation is called **discounting**, and the rate you translate at is the **discount rate**. Get those two ideas, and discounted cash flow valuation, bond pricing, the Gordon growth model, and even the price-to-earnings multiple all stop being separate tricks and reveal themselves as one idea wearing different clothes.

![A timeline showing compounding pushing a dollar forward in time and discounting pulling a future dollar back to today as the two directions of one machine](/imgs/blogs/time-value-of-money-discounting-for-investors-1.png)

The figure above is the whole post in one picture. On the right, **compounding** takes a dollar you have today and pushes it forward through time — at 8 percent a year, \$100 today becomes \$108 next year, \$116.64 the year after, snowballing into the future. On the left, **discounting** is the exact reverse arrow: it takes a dollar you *will* receive in the future and pulls it back to today, shrinking it as it travels because you had to wait for it. These are not two different ideas. They are one machine — the multiplication by `(1+r)` — run forward (compounding) or backward (discounting). Everything in this post is built from that single back-and-forth.

We are going to start from absolute zero. We will not assume you know what a present value is, what a discount rate means, or why a perpetuity is worth what it is worth. We will define every symbol the first time it appears, ground every formula in a concrete dollar example using a recurring make-believe company — **Northwind Industries**, the same firm that runs through this whole series — and build up, step by patient step, until the scariest-looking equation in valuation (the terminal-value formula) feels obvious. By the end you will understand not just *how* to discount but *why* it works, why the discount rate matters so much, and why a small change in one number can double or halve what a business appears to be worth. That last point is not academic: it is the reason the same company can look cheap to one investor and expensive to another, and the reason a quarter-point move in interest rates can wipe billions off the market in an afternoon.

## Foundations: why money has a time value, from first principles

Let us be ruthless about starting at the bottom. Before any formula, we need to answer the plain-English question: *why* is a dollar today worth more than a dollar later? There are three reasons, and they stack on top of each other. Understanding them is more important than memorizing the equations, because the equations are just these three reasons made precise.

### Reason one: opportunity cost (the big one)

The deepest reason a dollar today beats a dollar tomorrow has nothing to do with inflation or risk. It is **opportunity**. A dollar in your hand right now can be *put to work*. You can deposit it and earn interest. You can lend it. You can buy a slice of a good business and earn its profits. By the time next year arrives, that dollar has become something more than a dollar — it has earned a return. So if I offer you \$1,000 next year instead of \$1,000 now, I am not offering you the same thing late; I am offering you *less*, because I have robbed you of a year of earning power. The dollar today is worth more by exactly the amount it could have earned in the meantime. This is **opportunity cost**, and it is the engine of the entire idea. Even in a world with zero inflation and zero risk, money would still have a time value, purely because capital can be put to work and grow.

### Reason two: risk and uncertainty

The second reason is **risk**. A dollar promised to you next year might not arrive. The person promising it could go broke, change their mind, or vanish. The business that owes it could have a terrible year. A bird in the hand is worth two in the bush precisely because the bush-birds might fly off. The further into the future a promised payment lies, and the shakier the promiser, the more uncertain it is, and the less you should value it today. We will see that risk enters the math through the discount rate: riskier future cash gets discounted at a higher rate, which shrinks it more.

### Reason three: inflation

The third reason — usually the smallest of the three, though it dominated the 1970s and roared back in the early 2020s — is **inflation**. Over time, prices generally rise, so a dollar buys less next year than it does today. If a coffee costs \$5 now and \$5.15 next year, then \$5 next year no longer buys you a coffee. Inflation erodes the *purchasing power* of money, so even a perfectly certain future dollar is worth less in real terms than a present one. We will return to inflation when we separate **nominal** rates (which include expected inflation) from **real** rates (which strip it out).

Put the three together and the conclusion is airtight: a future dollar is worth less than a present dollar because of what you could have done with the money (opportunity), the chance you never get it (risk), and the erosion of what it buys (inflation). The genius of the time-value-of-money framework is that it bundles all three into a single number, the **discount rate**, and gives us a clean way to convert any future dollar into its honest worth today.

### The vocabulary we will use throughout

Let us pin down the words before we use them, because the whole subject collapses into confusion if these float.

- **Present value (PV)** — what a future amount of money is worth *today*. This is the number we are almost always solving for, because "what is this worth today" is the question an investor actually cares about.
- **Future value (FV)** — what a present amount of money will grow to at some point in the future, after earning a return.
- **The rate, `r`** — the annual rate of return, written as a decimal (8 percent is `r = 0.08`). When we push money forward, it is the rate we earn; when we pull money back, it is the rate we discount at. We will call it the **discount rate** or the **required return** — they are the same thing seen from two angles.
- **`n`** — the number of periods (usually years) between today and when the cash arrives.
- **Cash flow (CF)** — a payment of money at a point in time. A bond's coupon, a company's dividend, the profit a business hands its owners — these are all cash flows.
- **A stream** — a sequence of cash flows arriving at different times (year 1, year 2, year 3, …). Valuing a *stream* is the whole game, because every real investment is a stream, not a single payment.

With that vocabulary, we can build the two formulas that everything else stands on.

## Compounding: how a dollar grows forward in time

Start with the easy direction — pushing money *forward*. Suppose you put \$100 into an account that pays 8 percent a year. After one year, you have your original \$100 plus 8 percent of it, which is \$8, for a total of \$108. We can write that as:

`FV after 1 year = 100 × (1 + 0.08) = 108`

The `(1 + 0.08)` is the **growth factor** — multiply by it to advance one year. Now leave the money for a second year. The key insight, and the reason compounding is called the eighth wonder of the world, is that in year two you earn 8 percent not on your original \$100 but on the whole \$108 — you earn interest on your interest. So:

`FV after 2 years = 108 × (1 + 0.08) = 116.64`

You earned \$8 the first year and \$8.64 the second, because that second year's interest was calculated on a bigger base. Repeat the multiplication for `n` years and you get the master formula for compounding:

`FV = PV × (1 + r)^n`

Read it in plain English: *take what you have today (PV), and multiply it by the growth factor `(1+r)` once for every year it grows.* The exponent `n` is just "how many times do I apply the growth factor." That is all compounding is — repeated multiplication by `(1 + r)`.

#### Worked example: \$100 compounding at 8% for 10 years

Let us run the full machine. **Northwind Industries** has \$100 of spare cash and can earn 8 percent a year on it. What is it worth in 10 years?

`FV = 100 × (1.08)^10`

We need `(1.08)^10`. Multiplying it out year by year: 1.08, 1.1664, 1.2597, 1.3605, 1.4693, 1.5869, 1.7138, 1.8509, 1.9990, and finally **2.1589** after the tenth year. So:

`FV = 100 × 2.1589 = $215.89`

Your \$100 has more than doubled. Now look closely at where that came from. If Northwind had earned **simple** interest — 8 percent of the *original* \$100 every year, never reinvesting — it would have earned \$8 per year for 10 years, or \$80, ending at \$180. But compounding earned \$115.89, a full \$35.89 more. That extra \$35.89 is *interest earned on interest* — the snowball. And the longer the horizon, the more the snowball dominates: over 30 years at 8 percent, \$100 becomes \$1,006, of which only \$240 is simple interest and a staggering \$666 is interest-on-interest. *Compounding means your money's growth itself starts growing, which is why time in the market beats almost everything else over the long run.*

### The Rule of 72: a mental-math shortcut

Before we flip the machine around, here is a trick worth carrying for life. To estimate how many years it takes money to **double** at a given rate, divide 72 by the rate (as a whole number). At 8 percent, 72 ÷ 8 = 9 years to double. (Our exact calculation above hit 2.0 right around year 9 — `(1.08)^9 = 1.999` — so the rule nailed it.) At 6 percent it is 12 years; at 12 percent, 6 years; at 4 percent, 18 years. It is not exact, but it is close enough to do in your head, and it builds an instinct for how violently small differences in rate compound over time. A portfolio earning 9 percent doubles every 8 years; one earning 6 percent doubles every 12. Over 48 years, the first doubles six times (×64) and the second only four times (×16) — the same starting dollar ends up four times richer purely from a three-point edge in rate. That is the tyranny and the gift of compounding, and it is exactly why the discount rate, which we turn to next, is such a powerful lever.

## Discounting: pulling a future dollar back to today

Now we flip the machine. Compounding answered "if I have money today, what will it be worth later?" Discounting answers the question an investor actually cares about: *"if I will receive money later, what is it worth to me today?"*

The algebra is trivial — we just solve the compounding formula for `PV` instead of `FV`. Starting from `FV = PV × (1+r)^n`, divide both sides by `(1+r)^n`:

`PV = FV / (1 + r)^n`

That is the single most important formula in valuation, and you already understand it, because it is just compounding run backward. Where compounding *multiplies* by the growth factor once per year, discounting *divides* by it once per year. The thing you divide by, `(1+r)^n`, has a name: the **discount factor's** reciprocal. More usefully, define the **discount factor**:

`DF = 1 / (1 + r)^n`

The discount factor is a number between 0 and 1 that tells you "one dollar received in year `n` is worth *this many* cents today." At 10 percent, a dollar in 5 years has a discount factor of `1 / (1.10)^5 = 1 / 1.6105 = 0.6209` — so it is worth about 62 cents today. Multiply any future cash flow by its discount factor and you get its present value. That is the whole act of discounting: scale each future dollar down by the factor that reflects how long you wait and how much you give up by waiting.

#### Worked example: the PV of \$1,000 received in 5 years at 10%

Northwind is owed **\$1,000** by a customer who will pay in exactly **5 years**. Money of similar risk earns 10 percent a year, so we discount at `r = 0.10`. What is that promise worth today?

`PV = 1000 / (1.10)^5`

We need `(1.10)^5`: 1.10, 1.21, 1.331, 1.4641, and **1.61051** at year five. So:

`PV = 1000 / 1.61051 = $620.92`

The promise of \$1,000 in five years is worth only about **\$621 today**. Why? Because \$621 invested today at 10 percent grows back to exactly \$1,000 in five years — check it: `621 × 1.61051 = 1,000`. The two statements are identical: "\$1,000 in five years is worth \$621 today" and "\$621 today grows to \$1,000 in five years" are the same sentence read in opposite directions. The \$379 difference between the \$1,000 face amount and the \$621 present value is the price of waiting — the return you'd otherwise have earned over those five years. *Discounting puts an exact dollar number on impatience: it tells you precisely what a delay costs you.*

### The discount factor table: an investor's most-used tool

Because the discount factor depends only on the rate and the number of years, you can build a little table of them and reuse it. Here is one at 10 percent, the rate from our example:

| Year `n` | Discount factor `1/(1.10)^n` | One dollar in year `n` is worth… |
|---|---|---|
| 1 | 0.9091 | 90.9¢ |
| 2 | 0.8264 | 82.6¢ |
| 3 | 0.7513 | 75.1¢ |
| 5 | 0.6209 | 62.1¢ |
| 10 | 0.3855 | 38.6¢ |
| 20 | 0.1486 | 14.9¢ |
| 30 | 0.0573 | 5.7¢ |

Stare at the bottom of that table for a moment, because it contains the single most consequential fact in this whole post. At 10 percent, a dollar you will receive **30 years from now is worth less than 6 cents today.** The far future, discounted, almost vanishes. A promise of \$1 million in 2056 is worth about \$57,000 right now. This is not a quirk of the math; it is the math telling you something true and humbling about how little the distant future is worth when capital can be productively employed in the meantime. Hold onto this — when we get to why rising rates crush growth stocks, this collapsing tail is the entire story.

![Decay curves showing how the present value of one dollar shrinks toward zero over time and how a higher discount rate makes it shrink far faster](/imgs/blogs/time-value-of-money-discounting-for-investors-2.png)

The figure above plots the discount factor — the present value of a single future dollar — against the year it arrives, for three different discount rates. Every curve starts at \$1.00 today (a dollar now is worth a dollar, no discounting needed) and decays toward zero as the cash arrives further out. The crucial feature is how *differently* the curves fall. At a gentle 4 percent (top curve), a dollar 30 years out is still worth about 31 cents. At 10 percent (middle), it is worth under 6 cents. At 16 percent (bottom), it is worth barely a penny. The same future dollar can be worth thirty cents or one cent depending entirely on the rate you discount at — which is your first glimpse of just how much power lives in that single number `r`.

## The discount rate: what it really is and what it bundles

We have been using `r` as if it were handed to us. It is not — choosing it is the hardest and most important judgment in valuation, so let us understand exactly what it represents.

The discount rate is your **required return**: the annual return you *demand* to part with your money and bear the risk of this particular investment. Equivalently, it is your **opportunity cost of capital** — the return you could earn on the best available alternative of similar risk. If you could earn 10 percent on something equally risky, you have no reason to accept less than 10 percent here, so you discount at 10 percent. The discount rate is the hurdle the investment must clear to be worth making.

That single number bundles together the three reasons money has a time value. We can write it as a sum:

`discount rate = risk-free rate + risk premium`

The **risk-free rate** is the return you can earn with essentially no risk of not being paid — in practice, the yield on a short-term government bond (a U.S. Treasury bill, say). It compensates you for pure time and for expected inflation: even with zero risk, you demand *some* return to wait, and you demand extra to offset the erosion of inflation. As of writing it might be around 4 to 5 percent.

The **risk premium** is the *extra* return you demand on top of the risk-free rate to compensate for the chance that this particular investment disappoints — that the cash flows are smaller than forecast, or never come at all. A rock-solid blue-chip company earns a small premium; a speculative startup earns a large one. The riskier and less certain the cash flows, the bigger the premium, the higher the discount rate, and — crucially — the *lower* the present value. Risk, in this framework, literally makes a business worth less today, by raising the rate at which we shrink its future cash.

![A vertical stacked bar showing the discount rate built up from a risk-free base plus an equity risk premium plus a company-specific premium](/imgs/blogs/time-value-of-money-discounting-for-investors-6.png)

The figure above shows the discount rate as a stack you build from the ground up. The foundation is the **risk-free rate** — pure compensation for time and inflation, the floor that even a perfectly safe investment must clear. On top of it sits the **equity risk premium** — the extra return investors demand for bearing the general risk of owning stocks rather than government bonds, historically around 4 to 6 percent. On top of *that* sits any **company-specific premium** for risks unique to this firm — a shaky balance sheet, a volatile industry, a single-product business. Add the layers and you get the rate you discount with. The taller the stack, the harder the future cash gets shrunk, and the less the business is worth today. This is why two analysts can look at the same company and reach wildly different values: they are stacking different premiums. (The disciplined way to build this stack for a real company — the weighted average cost of capital, or WACC, and the capital asset pricing model, CAPM — is the subject of [building a DCF, part 2: cost of capital](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm); here we just need to know the rate exists and what it bundles.)

## Valuing a stream of cash flows

A single future payment is a warm-up. Every real investment — a bond, a rental property, an entire company — throws off a *stream* of cash flows arriving in different years. The beautiful thing is that valuing a stream requires no new idea at all. Because present values are just dollars-as-of-today, they are directly comparable and **additive**. So the recipe is simply:

1. Forecast the cash flow in each future year.
2. Discount each one back to today with `PV = CF / (1+r)^n`, using that year's `n`.
3. Add up all the present values.

In symbols, the present value of a stream of cash flows `CF₁, CF₂, …, CF_N` is:

`PV = CF₁/(1+r)¹ + CF₂/(1+r)² + … + CF_N/(1+r)^N`

That sum has a name when the cash flows are a company's free cash flows: it is a **discounted cash flow (DCF) valuation**. The entire DCF machine, which looks intimidating in a spreadsheet, is nothing more than this: forecast the cash, discount each year, add it up. (The forecasting half — projecting those cash flows in the first place — is its own craft, covered in [building a DCF, part 1: forecasting](/blog/trading/equity-research/building-a-dcf-part-1-forecasting).)

#### Worked example: PV of a 5-year \$200/yr cash stream at 10%

Northwind signs a contract that will pay it **\$200 at the end of each year for the next five years**. With money discounted at 10 percent, what is the contract worth today? We discount each year's \$200 by its own discount factor (which we already tabulated above) and sum:

| Year | Cash flow | Discount factor at 10% | Present value |
|---|---|---|---|
| 1 | \$200 | 0.9091 | \$181.82 |
| 2 | \$200 | 0.8264 | \$165.29 |
| 3 | \$200 | 0.7513 | \$150.26 |
| 4 | \$200 | 0.6830 | \$136.60 |
| 5 | \$200 | 0.6209 | \$124.18 |
| **Total** | **\$1,000** | | **\$758.16** |

The contract promises \$1,000 of total cash (five payments of \$200), but it is worth only **\$758 today**, because the later payments are worth progressively less. Notice the pattern in the present-value column: year 1's \$200 is worth \$182, but year 5's identical \$200 is worth only \$124 — the same payment, shrunk 32 percent simply for arriving four years later. The total, \$758, is the most you should rationally pay today for this stream if 10 percent is your required return. Pay less and you beat your hurdle; pay more and you fall short of it. *A stream of cash is worth the sum of its discounted parts, and the later parts always count for less.*

![A vertical bar chart where each year's cash flow shrinks to its present value and the discounted bars stack up to the total value of the stream](/imgs/blogs/time-value-of-money-discounting-for-investors-3.png)

The figure above is the worked example made visual. For each of the five years, the tall faint bar is the raw \$200 cash flow, and the shorter solid bar in front of it is that cash flow's present value after discounting. You can see the present-value bars shrinking year by year even though the cash flows are identical — the discount factor eats more of each later payment. Stack those five present-value bars and they sum to the \$758 total value of the stream. This is the literal anatomy of a DCF: a company's value is just a taller version of this picture, with one discounted bar for every future year of cash, all summing to one number — what the business is worth today.

### The annuity: a shortcut for level streams

The contract above paid the same \$200 every year. A stream of *equal* payments for a fixed number of years has a name — an **annuity** — and it is common enough (think mortgage payments, car loans, pension payouts, a bond's coupons) that mathematicians found a shortcut so you don't have to discount each year separately. The present value of an annuity paying `CF` per year for `n` years at rate `r` is:

`PV = CF × [1 − (1+r)^(−n)] / r`

Let us sanity-check it against our worked example: `CF = 200`, `r = 0.10`, `n = 5`. The bracket is `[1 − (1.10)^(−5)] / 0.10 = [1 − 0.6209] / 0.10 = 0.3791 / 0.10 = 3.791`. So `PV = 200 × 3.791 = $758.16` — exactly the total we got by discounting each year by hand. The annuity formula is not a new idea; it is a closed-form shortcut for "discount each equal payment and add them up." That bracketed term, `3.791`, is called the **annuity factor**: it says "five years of \$1/year at 10 percent is worth \$3.79 today." Multiply by the payment size and you are done. Handy, but never mysterious — it is just the cash-flow sum in a smaller package.

## The perpetuity: a cash flow that never ends

Now we make a leap that feels like it should be impossible and turns out to be the most useful trick in valuation. What if a cash flow goes on **forever**? A stream of \$1 every year, with no end date, stretching to infinity. Surely an infinite number of payments is worth an infinite amount?

It is not — and the reason is the collapsing tail we saw earlier. Each year's dollar is worth less than the last (discounted harder), and the discount factors shrink toward zero fast enough that the *infinite sum converges to a finite number*. Adding up infinitely many ever-smaller pieces lands on a clean, simple answer. A perpetual stream of `CF` per year, discounted at rate `r`, is worth:

`PV = CF / r`

That is it. To value a cash flow that lasts forever, just divide it by the discount rate. The intuition is gorgeous once you see it: if you have a pile of money `P` earning rate `r`, it throws off `P × r` of income every year forever without touching the principal. So to *generate* an income of `CF` forever, you need a pile of `P = CF / r`. The perpetuity value is just "how much principal would I need, at this rate, to spin off this income forever." A \$50-per-year perpetuity at 5 percent is worth \$1,000, because \$1,000 at 5 percent yields exactly \$50 a year, endlessly.

Why does anyone care about a fairy-tale "forever" cash flow? Because **a mature company is, to a first approximation, a perpetuity.** A stable business throwing off roughly constant profit year after year, with no fixed end date, is exactly a perpetual stream of cash. The perpetuity formula is the skeleton of how we value the "and it keeps going" part of every business — the part that, as we will see, usually dominates the whole valuation.

#### Worked example: a level perpetuity — Northwind's mature cash flow

Suppose **Northwind Industries** has matured into a stable, boring, cash-cow business that hands its owners **\$50 of cash every year**, and you expect that to continue indefinitely with no growth. Investors of similar risk require a 9 percent return, so `r = 0.09`. What is this perpetual stream worth?

`PV = CF / r = 50 / 0.09 = $555.56`

The endless \$50-a-year stream is worth about **\$556 today.** Check the intuition: \$556 invested at 9 percent yields `556 × 0.09 = $50.04` a year — just about exactly the perpetuity payment, forever, without ever touching the \$556 principal. So owning the perpetuity and owning \$556 of capital at 9 percent are economically the same thing. *A perpetuity's value is simply the principal that would generate its payment forever at the prevailing rate.* This single formula, with one twist we add next, becomes the most important calculation in valuing the long-term future of any company.

## The growing perpetuity: the workhorse of terminal value

Real companies rarely stand still — their cash flows tend to *grow* over time, with inflation and with the business itself. So we need the perpetuity formula's grown-up sibling: the **growing perpetuity**, a cash flow that starts at `CF` next year and grows at a constant rate `g` forever. Its value is:

`PV = CF / (r − g)`

This is the **Gordon growth model**, named after Myron Gordon, and it is the single most-used formula in equity valuation. Look at what changed from the plain perpetuity: the denominator went from `r` to `(r − g)`. The growth rate `g` is *subtracted* from the discount rate. The intuition: growth makes each future payment bigger than the last, which fights against the discounting that makes each one smaller. Subtracting `g` from `r` captures that tug-of-war — it is the *net* rate at which the stream effectively shrinks once you account for both the discounting headwind and the growth tailwind. A faster-growing stream shrinks more slowly in present-value terms, so it is worth more, which is exactly what dividing by a smaller `(r − g)` does.

One critical rule before we use it: **`g` must be less than `r`.** A cash flow cannot grow faster than the discount rate forever — if it did, the payments would eventually grow faster than discounting could shrink them, the infinite sum would *not* converge, and the value would be infinite (nonsense). Economically, no company can grow faster than the whole economy forever, so the long-run `g` is capped at something like the economy's nominal growth rate (a few percent). When you see a terminal-growth assumption of 2 or 3 percent in a DCF, this is why: it must be modest, and it must be below the discount rate.

#### Worked example: a growing perpetuity, CF = \$50, r = 9%, g = 3%

Now let us give Northwind a pulse. Instead of a flat \$50 forever, suppose its cash flow is **\$50 next year and grows at 3 percent every year after that** — \$51.50, then \$53.05, and so on, forever. Investors still require 9 percent. What is this growing stream worth?

`PV = CF / (r − g) = 50 / (0.09 − 0.03) = 50 / 0.06 = $833.33`

The growing stream is worth **\$833**, compared with just **\$556** for the flat \$50 perpetuity we valued a moment ago. A mere 3 percent of annual growth lifted the value by 50 percent — from \$556 to \$833. That is the leverage of growth in a perpetuity: because the growth compounds forever, even a modest `g` adds enormous value. Notice the denominator did all the work: it fell from `0.09` to `0.06`, and dividing by a number two-thirds as large makes the result one-and-a-half times as big. *Growth in a perpetuity is powerful precisely because it never stops — and it shows up entirely in that shrinking `(r − g)` denominator.*

![A before and after figure contrasting a flat perpetuity worth cash flow over r with a growing perpetuity worth cash flow over r minus g and the larger value that results](/imgs/blogs/time-value-of-money-discounting-for-investors-4.png)

The figure above sets the two formulas side by side. On the left is the plain perpetuity — a flat \$50 stream, valued at `CF / r = 50 / 0.09 = $556`. On the right is the growing perpetuity — the same \$50 starting point now growing 3 percent a year, valued at `CF / (r − g) = 50 / 0.06 = $833`. The only difference is the `− g` in the denominator, but it lifts the value by half. This is the workhorse of **terminal value**: when an analyst projects a company's cash flows out five or ten years and then needs to capture *everything after that* — the long tail stretching to infinity — they collapse it into a single growing perpetuity with this exact formula. That terminal value typically accounts for the *majority* of a DCF's total, which is why [terminal value is the part that dominates](/blog/trading/equity-research/terminal-value-the-part-that-dominates) and why the assumptions you feed into `(r − g)` matter more than almost anything else in the model.

## The `(r − g)` danger: when small assumptions become enormous numbers

Here is where the Gordon growth model turns from a friend into a loaded weapon, and where most amateur valuations go quietly insane. The value depends on the *difference* `(r − g)`, and that difference can be small. When you divide by a small number, tiny changes in the inputs produce huge swings in the output. As `g` creeps toward `r`, the denominator shrinks toward zero, and the value rockets toward infinity. The formula is extraordinarily **sensitive** right where analysts most want to use it.

Watch what happens as we push Northwind's growth rate up while holding `r` at 9 percent and `CF` at \$50:

| Growth `g` | Denominator `(r − g)` | Value `50 / (r − g)` |
|---|---|---|
| 0% | 0.09 | \$556 |
| 2% | 0.07 | \$714 |
| 3% | 0.06 | \$833 |
| 4% | 0.05 | \$1,000 |
| 5% | 0.04 | \$1,250 |
| 6% | 0.03 | \$1,667 |
| 7% | 0.02 | \$2,500 |
| 8% | 0.01 | \$5,000 |
| 8.9% | 0.001 | \$50,000 |

The same \$50 cash flow is worth \$556 at zero growth and **\$50,000** if you assume 8.9 percent growth. The value has gone up a hundredfold from a growth assumption that moved by less than nine percentage points. As `g` approaches `r`, the value approaches infinity — the formula literally blows up. This is not a bug; it is the honest consequence of assuming something grows almost as fast as you discount it, forever. But it means the growing-perpetuity formula is dangerous in careless hands: an analyst who wants a company to look cheap can nudge the terminal growth rate up half a point and conjure billions of dollars of "value" out of a denominator that was already small. The discipline is to keep `g` genuinely modest — anchored to long-run economic growth, always comfortably below `r` — and to *always* test how the answer moves when you wiggle the assumptions.

#### Worked example: sensitivity — the same perpetuity at g = 3% vs g = 5%

Let us make the danger concrete with two reasonable-sounding assumptions. Northwind's cash flow is \$50, `r = 9%`. An optimistic analyst argues growth should be 5 percent, not the 3 percent we used. How much does that two-point change in `g` move the value?

At `g = 3%`: `PV = 50 / (0.09 − 0.03) = 50 / 0.06 = $833`

At `g = 5%`: `PV = 50 / (0.09 − 0.05) = 50 / 0.04 = $1,250`

A two-percentage-point change in a single assumption — from 3 percent to 5 percent — raised the valuation by **50 percent**, from \$833 to \$1,250. Nothing about the business changed; only one number in a spreadsheet moved by two points. Now imagine this is a real company with a \$100 billion terminal value: that same innocent-looking tweak just added \$50 billion. *The `(r − g)` denominator is the most sensitive number in valuation — which is exactly why honest analysts present a whole table of values across a range of `g` and `r`, never a single false-precision figure.* When you read an equity research report and see a sensitivity grid of share prices across different growth and discount-rate assumptions, this is the danger it exists to expose.

![A curve showing the value of a growing perpetuity rising gently and then exploding toward infinity as the growth rate approaches the discount rate](/imgs/blogs/time-value-of-money-discounting-for-investors-5.png)

The figure above plots the value of our \$50 perpetuity against the assumed growth rate `g`, holding the discount rate at 9 percent. For low `g` the curve rises gently — going from 0 to 2 percent growth adds a manageable amount. But as `g` climbs past 5, 6, 7 percent and the denominator `(r − g)` shrinks toward zero, the curve bends sharply upward and then rockets toward the sky. The vertical dashed line at `g = r = 9%` is the asymptote where the value becomes infinite and the formula breaks. The whole picture is a warning: valuation built on a growing perpetuity is standing on the steep part of a cliff, and a small slip in the growth assumption sends the value tumbling — or soaring — far more than the slip itself would suggest.

## Discrete versus continuous compounding

A small technical point that trips people up: so far we have compounded **once a year** — interest is added at the end of each year. But interest can be added more often. A bank might compound monthly (twelve times a year), daily, or even continuously. The more frequently you compound, the more interest-on-interest you earn within the year, so the effective growth is slightly higher.

If you compound `m` times per year at annual rate `r` for `n` years, the formula becomes:

`FV = PV × (1 + r/m)^(m×n)`

Each period earns a smaller slice (`r/m`) but there are more periods (`m × n`). Push `m` to infinity — compounding every instant — and the formula converges to the elegant **continuous compounding** form using Euler's number `e ≈ 2.71828`:

`FV = PV × e^(r×n)` and correspondingly `PV = FV × e^(−r×n)`

Don't let `e` intimidate you — it is just the limit of "compound as often as possible." The practical difference is usually small. \$100 at 8 percent for one year is \$108.00 compounded annually, \$108.30 compounded monthly, and \$108.33 compounded continuously — a difference of a few cents. Continuous compounding matters in derivatives pricing and academic finance (it makes the calculus clean), but for valuing a business, annual or quarterly discounting is the norm and the difference rarely changes a decision. Know that it exists, know it makes growth *slightly* faster the more often you compound, and move on. The big ideas — present value, discount rate, perpetuities — are identical regardless of compounding frequency.

## Nominal versus real rates: stripping out inflation

We flagged inflation as one of the three reasons money has a time value. Here is how it enters the math precisely. A **nominal** rate is the rate you actually see quoted — it includes expected inflation. A **real** rate strips inflation out, measuring growth in *purchasing power* rather than in dollars. The approximate relationship is just subtraction:

`real rate ≈ nominal rate − inflation rate`

If your investment earns a nominal 8 percent and inflation runs at 3 percent, your *real* return — the growth in what your money can actually buy — is only about 5 percent. The dollars grew 8 percent, but each dollar buys 3 percent less, so your purchasing power grew about 5 percent. (The exact relationship is `(1 + nominal) = (1 + real) × (1 + inflation)`, but the subtraction is close enough for intuition.)

The golden rule that keeps you out of trouble: **be consistent.** Discount nominal cash flows (cash flows that include the effect of inflation, which is what companies actually report) at nominal discount rates. Discount real cash flows (inflation-adjusted) at real discount rates. Mixing them — discounting nominal cash flows at a real rate, or vice versa — is one of the most common valuation errors, and it systematically biases the answer. In practice, almost all corporate DCFs are done in nominal terms: the cash-flow forecasts already bake in expected price increases, so they are discounted at a nominal rate that also includes inflation. As long as inflation sits on *both* sides — in the cash flows and in the rate — it largely cancels, and you are left valuing the real economics of the business. *Match the units: nominal with nominal, real with real, never crossed.*

## Why a higher discount rate crushes long-dated cash flows most

We now arrive at the payoff — the single most important practical consequence of everything above, and the intuition that explains the most-discussed market dynamic of the last few years: *why do rising interest rates hammer high-growth stocks so much harder than stable, cash-rich ones?*

The answer is hidden in the discount factor `1/(1+r)^n` and the way it depends on `n`. When the discount rate rises, *every* future cash flow's present value falls — but the **distant** cash flows fall far more than the near ones, because they are divided by `(1+r)` many more times. Each extra year of waiting multiplies the damage. A higher `r` does not shave value evenly off the future; it takes a small bite out of next year's cash and a savage one out of the cash twenty years away.

Make it concrete. Compare \$100 due in 1 year and \$100 due in 20 years, and see what a rate rise from 5 percent to 8 percent does to each:

| Cash flow | PV at 5% | PV at 8% | Change |
|---|---|---|---|
| \$100 in **1 year** | \$95.24 | \$92.59 | −2.8% |
| \$100 in **20 years** | \$37.69 | \$21.45 | **−43.1%** |

The rate rise barely touched the near cash flow — the 1-year \$100 lost less than 3 percent of its value. But it gutted the far cash flow — the 20-year \$100 lost **43 percent**. Same three-point rate move, fifteen times the damage, purely because the distant cash flow gets discounted over twenty compounding years instead of one. This differential sensitivity has a name borrowed from bond math: **duration.** A cash flow (or a company) whose value sits mostly in the far future has *long duration* and is violently sensitive to rates; one whose value is in near-term cash has *short duration* and barely flinches.

Now connect it to stocks. A mature, profitable company — a utility, a consumer-staples giant — has most of its value in cash flows arriving soon. It is *short-duration*. A high-growth company that loses money today and promises enormous profits a decade or two out has nearly all of its value in distant cash flows. It is *long-duration*. So when interest rates rise — when `r` goes up across the whole market — the long-duration growth stock gets clobbered while the short-duration value stock shrugs it off. It is the exact same arithmetic as the table above, scaled up to entire companies. This is why "the Fed is hiking, growth stocks are getting crushed" is not a vibe or a narrative; it is the discount-factor table doing its inevitable work. The further out your cash, the more a higher rate hurts you.

![A before and after figure contrasting a short duration cash flow that barely loses value when rates rise with a long duration cash flow that loses much more](/imgs/blogs/time-value-of-money-discounting-for-investors-7.png)

The figure above contrasts the two cases directly. On the left, a **short-duration** cash flow — \$100 arriving in a year — barely shrinks when the discount rate rises from 5 percent to 8 percent, slipping from \$95 to \$93. On the right, a **long-duration** cash flow — \$100 arriving in twenty years — collapses from \$38 to \$21 under the very same rate rise, losing more than two-fifths of its value. Same companies, same cash, same rate move; wildly different damage, entirely because of *when* the cash arrives. This is the engine under every "rates up, growth down" headline, and once you see it as the discount factor doing arithmetic, those headlines stop being mysterious and start being predictable.

## Common misconceptions

Even people who can recite the formulas often carry a few stubborn misunderstandings. Let us clear the most damaging ones.

**"An infinite stream of cash flows must be worth an infinite amount."** No. As long as the discount rate exceeds the growth rate (`r > g`), the present values of the later payments shrink fast enough that the infinite sum converges to a finite number — `CF/r` for a flat perpetuity, `CF/(r−g)` for a growing one. Infinity in time does not mean infinity in value, because discounting tames the tail. The only way a perpetuity is worth infinity is if you (wrongly) assume `g ≥ r`, which says the cash grows at least as fast as you discount it, forever — an economic impossibility.

**"The discount rate is just the inflation rate, or just the interest rate on a loan."** No. The discount rate is your *total required return*, which bundles the risk-free rate (itself containing time preference and inflation) *plus* a risk premium for the specific investment. Inflation is one ingredient, not the whole thing. Using a low rate because "inflation is only 3 percent" while ignoring risk will badly overvalue a risky business. The riskier the cash flows, the higher the rate, the lower the value.

**"A higher growth rate is always good for valuation, so I should assume aggressive growth."** Growth is good, but in the `(r − g)` denominator it is *dangerously* good — small increases produce large value swings, and assuming `g` near `r` produces absurd, near-infinite values. Aggressive growth assumptions are the single most common way amateur valuations talk themselves into overpaying. Discipline means keeping terminal `g` modest (anchored to long-run economic growth, a few percent at most) and *always* checking sensitivity.

**"Present value and future value are different, complicated topics."** They are the same machine. `FV = PV × (1+r)^n` and `PV = FV / (1+r)^n` are one equation solved for different unknowns. Compounding pushes money forward; discounting pulls it back. If you understand one, you understand both — there is no second concept to learn.

**"Discounting and inflation are the same thing — I discount because of inflation."** Inflation is only the smallest of the three reasons. The dominant reason is opportunity cost: money today can be put to work and earn a return. Even in a zero-inflation world, you would still discount future cash, because capital is productive and waiting costs you the return you'd otherwise earn. People who think discounting is "just an inflation adjustment" use rates far too low and overvalue everything.

**"A more precise valuation is a more accurate one — I should compute the value to the penny."** The opposite is true. Because the output is so sensitive to the discount rate and growth assumptions (the `(r − g)` problem), a single precise-looking number is *false* precision. A value of "\$847.33 per share" implies a confidence the inputs cannot support. The honest output of a discounting model is a *range* across reasonable assumptions, and the most useful artifact is a sensitivity table, not a point estimate.

## How it shows up in real markets

The arithmetic of discounting is not a classroom exercise; it is the gravity that the whole market moves under. A few places it shows up vividly, using real, well-known episodes (round and illustrative figures, not claimed-exact).

**The 2022 growth-stock crash.** When the U.S. Federal Reserve raised interest rates sharply through 2022 to fight inflation, the broad market fell — but the damage was wildly uneven. Profitless, high-growth technology and software companies, whose entire value lived in cash flows projected a decade or more into the future, were *long-duration* assets, and many fell 50 to 80 percent. Meanwhile short-duration value names — energy producers, banks, consumer staples whose cash arrives now — held up far better and some rose. There was no change in the underlying businesses fast enough to explain that gap. What changed was `r`. A higher discount rate took a small bite out of near cash and a savage bite out of far cash, exactly as our 1-year-versus-20-year table predicted. The "duration of equities" became dinner-table conversation that year, and it is nothing but the discount factor `1/(1+r)^n` doing its work on companies instead of bonds.

**Bond prices and the inverse rule.** A bond is the purest discounting machine that exists — it is literally a fixed stream of coupon cash flows plus a principal repayment, and its price is just the present value of that stream. This is why bond prices move *opposite* to interest rates: when market rates `r` rise, you discount the bond's fixed coupons at a higher rate, so its present value (price) falls; when rates drop, the price rises. And long-maturity bonds (long duration) swing far more than short ones for the same rate move — the same `n`-in-the-exponent effect. Everything we built for stocks was first worked out on bonds; equity duration is the bond idea imported.

**Terminal value dominating real DCFs.** In a typical professional DCF, an analyst forecasts explicit cash flows for five or ten years and then uses the growing-perpetuity formula to capture everything beyond — the terminal value. In practice, that single Gordon-growth term often accounts for **60 to 80 percent of the entire estimated value of the company.** That is a startling fact with a sobering implication: most of what a DCF says a company is worth comes from one formula, `CF/(r−g)`, fed by two of the most uncertain numbers in the whole model — the long-run growth rate and the discount rate. It is why two reputable banks can publish target prices 40 percent apart on the same stock, in perfect good faith: they nudged `r` and `g`, and the sensitive denominator did the rest. When you read a research report, the assumptions behind the terminal value deserve more scrutiny than any other line.

**Why low rates inflated everything in the 2010s.** For most of the 2010s, interest rates sat near historic lows. A low risk-free rate means a low discount rate across the board, which means *every* future dollar is worth more today — and the distant dollars, the long-duration ones, are worth dramatically more (run the discount-factor table at 2 percent versus 6 percent). That is a large part of why valuations stretched so high in that decade and why speculative, far-future-cash-flow assets — unprofitable growth companies, certain crypto assets, anything justified by a distant payoff — soared. When rates normalized, the same arithmetic reversed and those assets fell hardest. The level of interest rates is, quite literally, the price of time, and when the price of time changes, the value of everything that pays off in the future changes with it.

**The honest investor's defense: margin of safety.** Because discounting is so sensitive to assumptions, the great value investors refuse to trust any single computed value. Warren Buffett has said he'd rather be approximately right than precisely wrong, and insists on a **margin of safety** — buying only well below his best estimate of intrinsic value, so that even if his discount rate or growth assumption is off, he still comes out fine (see [Warren Buffett and Berkshire's value investing](/blog/trading/finance/warren-buffett-berkshire-value-investing)). The margin of safety is, at root, a humility tax paid *because* the `(r − g)` denominator is so treacherous. You demand a discount to your estimate precisely because you know the estimate could be wrong by a lot.

## When this matters and further reading

The time value of money is the foundation everything else in valuation is built on, so it shows up the moment you try to put a number on any investment. You need it to value a business by its cash flows, to price a bond, to compare an investment that pays off soon against one that pays off later, to decide whether a stock's price is reasonable given the cash it will produce, and to understand why the market reprices violently when interest rates move. If you internalize just one thing from this post, make it this: *a dollar's worth depends on when you get it, and discounting is the exact arithmetic of that "when."* Once that is reflex, the rest of valuation is detail.

From here, the natural next steps in this series build directly on the engine you now have:

- [The two pillars of valuation: intrinsic vs. relative](/blog/trading/equity-research/two-pillars-of-valuation-intrinsic-vs-relative) — how discounting (intrinsic value) sits alongside multiples (relative value), and when to lean on each.
- [Building a DCF, part 1: forecasting](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) — the other half of a DCF: how to project the cash flows that you then discount with the machine from this post.
- [Building a DCF, part 2: cost of capital (WACC and CAPM)](/blog/trading/equity-research/building-a-dcf-part-2-cost-of-capital-wacc-capm) — the disciplined way to choose the discount rate `r` for a real company, instead of pulling it from the air.
- [Terminal value: the part that dominates](/blog/trading/equity-research/terminal-value-the-part-that-dominates) — a deep dive on the growing-perpetuity term that, as we saw, usually accounts for most of a valuation, and how to keep its assumptions honest.

For the deeper mathematics — convergence of infinite series, the formal treatment of discount factors, and the probability behind risk premiums — the [math-for-quants foundations](/blog/trading/math-for-quants) go further than we needed to here. But you do not need any of that to value a business well. You need the one idea this post was built on: future cash, pulled back to today, shrunk by the rate that captures what waiting and risk cost you. That is discounting, and it is the engine under every valuation you will ever do.
