---
title: "The ROIC–WACC Spread: The Engine of Intrinsic Value"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "One relationship explains more about long-run shareholder returns than any other — the spread between the return a company earns on its capital and the cost of that capital, multiplied by how much it can reinvest and how long the spread survives. This is the engine under every DCF and the deepest test of business quality."
tags: ["equity-research", "corporate-finance", "roic", "wacc", "intrinsic-value", "value-creation", "reinvestment", "moats", "compounding", "valuation"]
category: "trading"
subcategory: "Equity Research"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — One relationship explains more about long-run shareholder returns than any other: the *spread* between the return a company earns on invested capital (ROIC) and the cost of that capital (WACC), multiplied by how much the company can reinvest, and sustained over how long.
>
> - **A positive spread (ROIC > WACC) means growth creates value.** Every dollar reinvested earns more than it costs, so the more the company grows, the more wealth it builds for owners.
> - **A zero spread (ROIC = WACC) makes growth worthless.** The company runs faster but stands still in value terms — it earns back exactly what its capital costs and not a cent more.
> - **A negative spread (ROIC < WACC) means growth *destroys* value.** This is the counterintuitive core: a fast-growing low-return business is *shrinking* its intrinsic value as it expands, because every new dollar invested earns less than the dollar costs.
> - **Intrinsic value has three drivers, not one:** the size of the spread, the *reinvestment rate* (how much capital can be redeployed at that spread, equal to growth ÷ ROIC), and the **durability** of the spread — how many years a moat keeps competition from dragging ROIC back down to WACC.
> - **High-ROIC compounders with long runways are the holy grail** because they multiply all three drivers together; they are also the rarest businesses on earth, which is exactly why the market pays so much for them.

There is a single relationship in finance that, once you truly understand it, reorganizes everything else you know about investing around it. It is not a clever ratio or a proprietary screen. It is a piece of arithmetic so simple it fits on the back of an envelope, and yet it is the reason some companies are worth ten times the capital invested in them while others — equally large, equally busy, equally "profitable" by the accountant's reckoning — are worth less than the cash that was poured into them. The relationship is the **spread between the return a business earns on its invested capital and the cost of that capital**, and the discipline of equity research, stripped to its skeleton, is the practice of estimating that spread, judging how long it will last, and figuring out how much capital the business can pour into it.

Everything else in this series — the [income statement](/blog/trading/equity-research/income-statement-line-by-line-revenue-to-net-income), the [balance sheet](/blog/trading/equity-research/balance-sheet-what-a-company-owns-owes-and-is-worth), [margins](/blog/trading/equity-research/profitability-margins-gross-operating-net), [returns on capital](/blog/trading/equity-research/returns-on-capital-roic-roe-roa), the [cost of capital](/blog/trading/equity-research/cost-of-capital-and-the-hurdle-rate), the [discounted cash flow model](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) — is, in the end, an apparatus for measuring this one thing and turning it into a number. The discounted cash flow is not really a model of cash flows; it is a model of the spread, wearing the costume of cash flows. The multiples are not really shortcuts for value; they are shorthand for the spread and its durability. If you grasp the spread, the rest of equity research becomes the working-out of a single idea rather than a grab-bag of unrelated techniques.

This post is about that idea. We will build it from zero, the way the whole series builds everything from zero: define the pieces, assemble them into the value-creation identity, and then push on the identity until it yields its most counterintuitive and most useful consequences — including the one that trips up nearly every beginner and a surprising number of professionals: *that growth, the thing investors prize above all, is not always good, is sometimes worthless, and is sometimes actively destructive.*

![A vertical tree breaking intrinsic value into three drivers — the spread, the reinvestment rate, and the durability — each branching into a value-creating and a value-destroying case](/imgs/blogs/roic-wacc-spread-the-engine-of-intrinsic-value-1.png)

The figure above is the map for the entire post. Intrinsic value created above and beyond the capital already invested in a business is not one thing but a *product* of three: how wide the spread is, how much capital the company can reinvest at that spread, and how many years competition lets the spread survive. Multiply a wide spread by a large reinvestment runway by a long durability and you get a compounding machine worth many times its capital. Take away any one of the three — let the spread be zero, or the reinvestment runway short, or the durability brief — and the magic evaporates. We will spend the rest of this post on those three drivers, one at a time, and then on what happens when they combine.

## Foundations: capital, return, cost, and the spread

Before we can talk about the spread, we need four ideas defined precisely. Three of them — invested capital, ROIC, and WACC — have full posts of their own, so here we will define them tightly enough to do the arithmetic and point you to the deep treatments. The fourth idea, the spread itself, is new, and it is the spine of everything that follows.

**Invested capital** is the pool of money tied up in the operating business. It is the sum of what owners have put in (equity) and what lenders have put in (interest-bearing debt), minus any cash the company is holding that isn't actually deployed in operations (excess cash). The logic is that we want to measure the return on the capital that is *working* — funding factories, inventory, receivables, product development — not on a pile of idle cash sitting in a money-market account. If you put \$1,000M into a business, that \$1,000M is the base on which you expect to earn a return. The full mechanics, including the asset-side definition, live in the [returns-on-capital post](/blog/trading/equity-research/returns-on-capital-roic-roe-roa).

**ROIC — return on invested capital** — is the percentage that pool earns each year. Specifically it is NOPAT (net operating profit after tax — the company's operating profit, taxed as if it carried no debt) divided by invested capital:

$$\text{ROIC} = \frac{\text{NOPAT}}{\text{Invested capital}}$$

ROIC measures the *business*, scrubbed of how it happens to be financed. A company earning \$150M of NOPAT on \$1,000M of invested capital has a 15% ROIC: it earns fifteen cents of after-tax operating profit per year for every dollar tied up in it. ROIC is the single best one-number summary of business quality, because it answers the question that actually matters: *when this company invests a dollar, how many cents does it get back, year after year?*

**WACC — the weighted average cost of capital** — is what that capital *costs*. Capital is never free. Lenders charge interest; shareholders demand a return for the risk they bear. Blend the cost of debt and the cost of equity, weighted by how much of each the company uses, and you get WACC — the minimum return the business must earn just to keep its capital providers whole. If a company's WACC is 9%, then nine cents of every dollar of capital is owed, each year, to the people who supplied that dollar, before the company has created a single cent of value for anyone. The derivation — CAPM, the cost of debt, the weights — is the subject of the [cost-of-capital post](/blog/trading/equity-research/cost-of-capital-and-the-hurdle-rate); for our purposes, WACC is the hurdle, the price of the money the business runs on.

**The spread** is the whole point. It is simply ROIC minus WACC:

$$\text{Spread} = \text{ROIC} - \text{WACC}$$

A company with a 15% ROIC and a 9% WACC has a spread of six percentage points. Those six points are the *excess return* — the profit the business earns over and above what its capital costs. They are, quite literally, value being created out of thin air each year: the business takes money that costs 9% and turns it into 15%, pocketing the difference for its owners. A company with a 9% ROIC and a 9% WACC has a zero spread: it earns exactly what its capital costs, creating no value, merely treading water. And a company with a 6% ROIC and a 9% WACC has a *negative* spread of minus three points: it is destroying three cents of value per dollar of capital every year, even if its income statement shows a healthy-looking profit.

That last sentence is worth re-reading, because it contains the idea that separates people who understand value creation from people who merely read financial statements. **A company can report an accounting profit every single year and still be destroying value the entire time**, if the profit it earns on its capital is less than the cost of that capital. The income statement, which knows nothing about the cost of equity, will never tell you this. The spread is the only thing that will.

#### Worked example: the spread is value created per year

Let us anchor this in dollars immediately, using a fictional company we will return to throughout the post: **Northwind Industries**, a maker of industrial pumps. Northwind has \$1,000M of invested capital, earns a 15% ROIC, and faces a 9% WACC.

The value Northwind creates each year is the spread times the capital it is earned on:

$$\text{Value created per year} = (\text{ROIC} - \text{WACC}) \times \text{Invested capital}$$

$$= (15\% - 9\%) \times \$1{,}000\text{M} = 6\% \times \$1{,}000\text{M} = \$60\text{M}$$

Northwind earns \$150M of NOPAT on its \$1,000M of capital. Of that \$150M, exactly \$90M (9% of \$1,000M) is the "rent" owed to the capital providers — it is what the capital costs. The remaining \$60M is pure economic profit: value created above and beyond the cost of the money. That \$60M, repeated and grown and discounted, is what makes Northwind worth more than its \$1,000M of capital.

*The spread times capital is the only profit that actually belongs to the owners as value — everything up to the cost of capital was already owed to someone.*

## The value-creation identity: why the spread is the engine

Now we assemble the foundations into the relationship that gives this post its title. Strip a discounted cash flow model down to its essentials and what you find underneath is a single statement, often called the **value-creation identity** or the *economic-profit* view of value:

> A business is worth its invested capital *plus the present value of all the economic profit it will ever earn* — and economic profit each year is the spread times the capital deployed.

In symbols, the value of the operating business is:

$$\text{Value} = \text{Invested capital} + \text{PV of future } (\text{ROIC} - \text{WACC}) \times \text{capital}$$

Read that slowly. The first term, invested capital, is the money already in the business — its "cost basis," what it would be worth if it merely earned its cost of capital forever and created nothing extra. The second term is the *premium*: the present value of every future year's spread-times-capital. If the spread is positive, the premium is positive and the business is worth *more* than its capital. If the spread is zero, the premium is zero and the business is worth *exactly* its capital. If the spread is negative, the premium is negative and the business is worth *less* than the capital sunk into it — which is the formal statement of a value trap.

This is the same number a [discounted cash flow](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) produces; it is just decomposed differently. A standard DCF discounts free cash flows. The economic-profit form discounts the spread. Done correctly, the two give the identical answer — but the economic-profit form is enormously more *illuminating*, because it shows you exactly where value comes from. It comes from the spread. Nowhere else. Growth, margins, scale, market share — all of them matter only insofar as they widen the spread, extend its durability, or expand the capital base on which a positive spread is earned. None of them creates value on its own.

This is why the spread is "the engine." Every other lever an investor or a manager can pull — cut costs, raise prices, expand abroad, make an acquisition, buy back stock — feeds into value *only through its effect on the spread, the reinvestment, or the durability*. If a strategy widens the spread, it creates value. If it doesn't touch the spread, it is, at best, rearranging deck chairs. The spread is the carburetor; everything else is the fuel line.

It is worth pausing on *why* the economic-profit form and the ordinary free-cash-flow DCF must give the same answer, because the equivalence is not a coincidence and seeing it cements the whole idea. A free-cash-flow DCF discounts NOPAT minus reinvestment. Now split NOPAT into two pieces: the part that merely covers the cost of capital (WACC times invested capital) and the part that exceeds it (the spread times invested capital). The first piece, discounted forever, is worth exactly the invested capital itself — a stream that earns precisely its discount rate is worth its principal, no more. The second piece, discounted, is the economic-profit premium. Add them and you get invested capital plus the premium, which is the value-creation identity. The two models are the same model viewed from two angles: one looks at the cash, the other at the *excess* return embedded in the cash. The economic-profit angle is better for *thinking*, because it isolates the only part of the cash flow that actually represents value creation — the spread — and lets the rest, the part that was always owed to capital providers, fall away. When a DCF and a multiple disagree, or when a valuation feels fragile, returning to the economic-profit form almost always reveals that the disagreement is really a disagreement about the spread or its durability, hiding inside the cash-flow arithmetic.

## Growth: the multiplier that can run in reverse

Here is where the idea earns its keep, because here it overturns the single most common intuition in all of investing: that growth is good.

Growth is *not* inherently good. Growth is a **multiplier on the spread**. It scales up whatever the spread already is. If the spread is positive, growth multiplies a good thing and creates more value. If the spread is zero, growth multiplies zero and creates nothing. And if the spread is negative, growth multiplies a *bad* thing and destroys value faster the more the company grows. Growth, in other words, is leverage on the spread — and like all leverage, it amplifies in both directions.

![Three growth curves over time on one chart — a green line earning above WACC rising, a flat blue line earning exactly WACC, and a red line earning below WACC falling](/imgs/blogs/roic-wacc-spread-the-engine-of-intrinsic-value-2.png)

The chart above makes the point in one picture. Three companies all grow their revenue and capital at the same 10% a year. The only difference among them is the spread. The green company earns a +6-point spread (ROIC 15% against WACC 9%); as it grows, its intrinsic value compounds upward, because each new dollar of capital it deploys earns six points more than it costs. The blue company earns exactly its cost of capital — a zero spread; as it grows, its value stays flat, tracking its invested capital exactly, because every new dollar earns precisely what it costs and contributes nothing extra. And the red company earns a −3-point spread (ROIC 6% against WACC 9%); as it grows, its value *erodes*, because every new dollar it invests to grow loses three cents relative to the cost of the money. Same growth, three completely different fates — and the spread is the only thing that distinguishes them.

This is the deepest and most useful counterintuition in fundamental analysis, so let us prove it with arithmetic rather than asking you to take it on faith.

#### Worked example: growth that destroys value

Return to Northwind, but imagine a struggling division of it — call it **Northwind Components** — that earns a ROIC of only 6% against the same 9% WACC. Its spread is negative three points. Management, proud of its growth, decides to invest an extra \$100M to expand the division by 10%.

What does that \$100M earn? At a 6% ROIC, it earns \$6M of NOPAT per year. What does it *cost*? At a 9% WACC, the \$100M of capital costs \$9M per year. So the division now earns \$6M on capital that costs \$9M:

$$\text{Net value created} = \$6\text{M (earned)} - \$9\text{M (cost of capital)} = -\$3\text{M per year}$$

The expansion *destroys* \$3M of value every year, forever (or until the spread changes). The income statement will show the division's profit going *up* — it now earns \$6M more in NOPAT than before, and revenue is up 10% — and an unsophisticated analyst, or an empire-building CEO, will call this a success. But intrinsic value has fallen by the present value of −\$3M a year. The faster Northwind Components grows, the poorer Northwind's owners become.

*A growing business that earns below its cost of capital is a machine for converting shareholder wealth into bigger revenue numbers; the growth is the symptom, not the cure.*

![A before-and-after comparison of two companies each investing 100 million dollars to grow 10 percent, one earning 15 percent and creating value, the other earning 6 percent and destroying it](/imgs/blogs/roic-wacc-spread-the-engine-of-intrinsic-value-3.png)

The figure contrasts the two cases on identical \$100M of new investment. On the left, "Goodco" earns 15% on the new capital — \$15M of NOPAT against a \$9M cost — and nets +\$6M of value created per year. On the right, "Badco" earns 6% — \$6M against the same \$9M cost — and nets −\$3M, value destroyed. The investment is identical. The growth rate is identical. The capital deployed is identical. Only the spread differs, and the spread flips the sign of the outcome from creation to destruction. This is what it means to say growth is value-neutral on its own: it is a magnifying glass held over the spread.

#### Worked example: growth that creates nothing

Now the in-between case, the one that surprises people most because the company *looks* fine. Imagine a third version of Northwind that earns exactly its cost of capital: ROIC 9%, WACC 9%, spread zero. Management invests \$100M to grow 10%.

The new capital earns \$9M of NOPAT (9% of \$100M). It costs \$9M (9% of \$100M). Net value created:

$$\$9\text{M (earned)} - \$9\text{M (cost)} = \$0$$

The growth creates *nothing*. Not a negative — a clean, exact zero. Revenue rises 10%, NOPAT rises by \$9M, the company is unambiguously bigger, the press release writes itself — and intrinsic value does not move by one cent. The company has run faster only to stay in the same place. Every dollar it reinvested simply replaced one unit of "earning the cost of capital" with another unit of "earning the cost of capital."

*When a company earns exactly its cost of capital, growth is pure water: it changes the size of the glass but not the value of what's inside.*

This zero-spread case is the hinge of the whole framework. Above it, growth helps; below it, growth hurts; at it, growth is irrelevant. And it explains a phenomenon that mystifies people who haven't internalized the spread: why two companies growing at the same rate can trade at wildly different valuations, and why the market sometimes *punishes* a company for announcing an ambitious, capital-hungry growth plan. If the market believes the new capital will earn below the cost of capital, more growth is *bad news*, and the stock falls on the announcement. The market, for all its faults, often understands the spread better than the company's own management.

## The second driver: the reinvestment rate

So far we have treated growth as a free choice — management simply "decides" to grow 10%. In reality, growth must be *funded*. To grow, a company must reinvest capital: build capacity, fund working capital, develop products, make acquisitions. And the amount it must reinvest to achieve a given growth rate is governed by — what else — its ROIC. This is the second driver of intrinsic value, and it ties growth and returns together in a single, beautiful relationship.

The relationship is this: **the reinvestment rate equals the growth rate divided by ROIC.**

$$\text{Reinvestment rate} = \frac{g}{\text{ROIC}}$$

The reinvestment rate is the fraction of NOPAT a company must plough back into the business (rather than pay out to owners) to sustain a growth rate of *g*. The logic is mechanical: if you earn a ROIC of, say, 20%, then to grow your earnings 10% next year you need to add enough capital that, earning 20%, produces 10% more profit. The capital you must add is 10% ÷ 20% = 50% of this year's NOPAT. A high ROIC makes each point of growth *cheap to buy*, because each reinvested dollar throws off a lot of incremental profit. A low ROIC makes growth *expensive*, because each reinvested dollar throws off little.

![A before-and-after comparison showing a high-ROIC firm funding 9% growth by reinvesting only 45% of profit while a low-ROIC firm must reinvest 150% and cannot self-fund](/imgs/blogs/roic-wacc-spread-the-engine-of-intrinsic-value-4.png)

The figure shows the two extremes. On the left, a high-ROIC firm earning 20% wants to grow 9% a year; its reinvestment rate is 9% ÷ 20% = 45%, so it ploughs back 45% of NOPAT and keeps the other 55% as free cash flow it can pay out to owners. It funds its growth comfortably from its own profits and still showers cash on shareholders. On the right, a low-ROIC firm earning 6% wants the same 9% growth; its reinvestment rate is 9% ÷ 6% = 150%. It must reinvest *one and a half times* its entire NOPAT to grow that fast — which is impossible from internal cash flow alone. It must raise outside capital, dilute its owners, or pile on debt, just to fund growth that (because its spread is negative) is destroying value anyway. The low-ROIC firm is on a treadmill that speeds up the harder it runs.

This is why ROIC and growth cannot be discussed in isolation, and why a "growth stock" with a low ROIC is one of the most dangerous animals in the market. Its growth is real, its revenue genuinely climbs — but the growth is consuming far more capital than it returns, and the only way to keep the music playing is to keep raising money. The connection between this idea and free cash flow is direct: a company's free cash flow is its NOPAT times one minus the reinvestment rate. A high-ROIC company with modest growth keeps most of its NOPAT as free cash. A low-ROIC company chasing growth keeps little or none — and may consume cash it doesn't have. The [free-cash-flow post](/blog/trading/equity-research/free-cash-flow-fcff-vs-fcfe) walks through that mechanic in detail.

#### Worked example: the reinvestment math for Northwind

Take our healthy Northwind: ROIC 20% (we'll use the higher figure here to illustrate a genuine compounder), and suppose it can grow 12% a year. Its reinvestment rate is:

$$\text{Reinvestment rate} = \frac{g}{\text{ROIC}} = \frac{12\%}{20\%} = 60\%$$

Northwind reinvests 60% of its NOPAT to fund 12% growth and pays out (or accumulates) the remaining 40% as free cash flow. If its NOPAT this year is \$200M, it reinvests \$120M and has \$80M of free cash flow available to owners. Crucially, because its spread is enormous (20% ROIC minus, say, a 9% WACC = +11 points), every one of those reinvested dollars is creating value. Northwind is the dream: it grows fast, the growth creates value, *and* it still generates surplus cash. That combination — high return, ample reinvestment, surplus cash — is what a genuine compounder looks like in the financials.

Contrast a company that wanted the same 12% growth at a 6% ROIC: its reinvestment rate would be 12% ÷ 6% = 200%. It would need to reinvest *twice* its entire NOPAT — an obvious impossibility without continuous external financing. The arithmetic itself tells you the growth plan is unfinanceable and, worse, value-destroying.

*Reinvestment rate is the bridge between the income statement and growth: it tells you how much profit a company must give up today to be bigger tomorrow, and a high ROIC makes that toll cheap.*

## The third driver: durability, and the fade

We now have two of the three drivers: the spread and the reinvestment rate. But there is a problem lurking in the value-creation identity. It assumes the spread *persists*. And in a competitive economy, it usually doesn't.

A wide positive spread is the most attractive thing in business — a company earning 20% on capital that costs 9% is minting money. And precisely because it is so attractive, it draws competition like blood draws sharks. Rivals enter the market, copy the product, undercut the price, hire away the talent, and bid up the cost of the inputs. Each of these competitive forces nibbles at the spread. Prices fall, or costs rise, or both, and the ROIC drifts downward. Capitalism, working as designed, drives extraordinary returns back toward the ordinary. The technical name for this process is the **fade**, and the number of years a company can hold its spread before the fade closes it is called the **competitive advantage period**, or CAP.

![A single ROIC curve fading downward over ten years from 15% toward a flat 9% WACC line, with the shrinking gap between them labeled as the value the moat creates](/imgs/blogs/roic-wacc-spread-the-engine-of-intrinsic-value-5.png)

The figure shows the fade in its purest form. A company begins with a 15% ROIC and a 9% WACC — a +6-point spread. Year by year, competition erodes the ROIC: 14%, then 12%, then 11%, drifting downward until, by year ten, it reaches 9% and the spread has closed to zero. From that point on, the company earns exactly its cost of capital and creates no further value. The total value the business will ever create is the area in that shrinking gap between the ROIC curve and the flat WACC line — and notice what that means: *the value is finite, and it is set by how slowly the gap closes.* A company whose spread fades in three years creates a fraction of the value of an otherwise-identical company whose spread fades in twenty. The durability of the spread — the width of the competitive advantage period — is therefore the third great driver of intrinsic value, and it is the most important single thing a [moat](/blog/trading/equity-research/economic-moats-durable-competitive-advantage) does: a moat does not (mainly) widen the spread; it *lengthens the time the spread survives*. A moat is durability made concrete.

This reframes what a "great business" is. A great business is not merely one with a high ROIC today; high ROICs are everywhere and most of them are about to fade. A great business is one with a high ROIC *that competition cannot easily erode* — protected by a brand customers won't abandon, a network that gets stronger as it grows, switching costs that lock customers in, a cost advantage rivals can't match, or a regulatory barrier they can't cross. Those protections are what stretch the competitive advantage period from three years to twenty, and in doing so they multiply the value the business creates many times over. The whole discipline of competitive strategy, from an investor's seat, reduces to one question: *how long can this spread last?*

The fade is not a theoretical tidiness invented to make models converge; it is one of the most robust empirical regularities in all of corporate finance. When researchers sort thousands of companies by ROIC and follow each cohort forward over a decade, the pattern is unmistakable: the highest-return companies see their ROICs drift downward over time, and the lowest-return companies see theirs drift *upward*, both pulled toward a middling, economy-wide average. Extraordinary returns regress toward the mean because capitalism is, in essence, a machine for competing away excess profit. This is exactly why durability is so precious and so rare: the *default* outcome for a wide spread is to close, and a business that resists that gravity for a decade or two is doing something genuinely unusual. It also explains a common modeling discipline — assuming that any company's ROIC fades toward its cost of capital over an explicit forecast horizon unless there is a concrete, nameable reason it won't. The burden of proof sits with durability, not against it: you should assume the spread fades and then ask the moat to earn back the years, rather than assume permanence and let the model quietly bake in an advantage no business in history has actually sustained.

#### Worked example: what the fade does to value

Let us put numbers on durability using Northwind once more. Suppose Northwind has \$1,000M of invested capital and a starting spread of +6 points (ROIC 15%, WACC 9%), creating \$60M of economic profit in year one. We will compare two scenarios for the fade.

**Scenario A — fast fade (3-year CAP):** the spread shrinks roughly evenly to zero over three years: +6 points, then +4, then +2, then zero thereafter. The economic profit stream is roughly \$60M, \$40M, \$20M, then \$0. The sum of that stream (ignoring discounting for the intuition) is about \$120M of total value created.

**Scenario B — slow fade (10-year CAP):** the spread shrinks to zero over ten years: +6, +5.4, +4.8, ... declining gently to zero in year ten. The economic-profit stream runs roughly \$60M, \$54M, \$48M, ... summing to roughly \$330M of total value created — *nearly three times* Scenario A's, from the identical starting spread and identical capital.

The only thing that changed between A and B was *how long the spread lasted*. Same business quality at the start, same capital, same growth — and a business worth almost three times as much, purely because its moat holds the spread for a decade instead of three years. This is why durability sits as a co-equal driver alongside the spread itself, and why seasoned investors will pay up enormously for a *durable* advantage and barely at all for a fleeting one.

*Two businesses can earn the identical return today and be worth wildly different amounts; the difference is not how high they fly but how long they can stay up there before competition pulls them down.*

## Putting the three drivers together: value per dollar of capital

We now have all three drivers — spread, reinvestment, durability — and can ask the question that ties the whole post together: *how much is a business worth relative to the capital invested in it?* This ratio, value-to-invested-capital, is the cleanest way to see all three drivers acting at once, and it maps directly onto the price-to-book ratio you see quoted on every stock.

![A matrix with spread on the rows and durability on the columns, showing value-to-capital multiples rising from 1.0x at zero spread to over 3.5x for a wide, durable spread, and falling below 1.0x for negative spreads](/imgs/blogs/roic-wacc-spread-the-engine-of-intrinsic-value-6.png)

The matrix lays out value-to-capital as a function of the spread (down the rows) and its durability (across the columns). Read it as the master summary of everything above:

- **Along the top row (zero spread), every cell reads 1.0×**, no matter the durability. A business that earns exactly its cost of capital is worth exactly its capital, and no amount of "durability" changes that — there is nothing to make durable. Growth here is the pure water of the earlier example.
- **Moving down to positive spreads, the multiple rises** — and it rises *faster* the more durable the spread is. A modest +3-point spread that fades in three years is worth only about 1.1× capital; the same +3-point spread sustained for twenty years is worth perhaps 1.8×. A wide +6-point spread that lasts twenty years pushes past 3.5× — the holy-grail compounder, a business worth three and a half times the capital sunk into it.
- **The bottom row, negative spread, sinks below 1.0×** — and sinks *further* the longer the negative spread persists, because a durable bad business destroys value for longer. A −3-point spread that lasts a decade is a 0.6× business: a long-lived value trap, worth less than half the capital poured into it, with management cheerfully reinvesting all the while.

This single matrix is, in a real sense, the entire post. Intrinsic value relative to capital is a surface that rises with the spread and with its durability, peaks at the high-spread, long-durability corner, and falls through 1.0× into value-destruction territory as the spread goes negative. Every business sits somewhere on this surface, and the job of equity research is to figure out which cell — and to notice when the market has priced a business as if it were in a different cell than it really is.

#### Worked example: from the spread to a justified price-to-book

The matrix connects directly to a multiple you can look up on any stock: price-to-book. Recall from the [multiples post](/blog/trading/equity-research/multiples-101-pe-ev-ebitda-pb-ps-peg) that book value is, roughly, the equity capital invested in a business, and price-to-book compares market value to that invested capital. The spread is what *justifies* a price-to-book above 1.0×.

Take a simplified, durable business: ROIC 15%, cost of equity 9%, growing 4% a year forever, with the spread assumed to persist. A standard result (a rearrangement of the dividend-growth model in terms of returns) gives a justified price-to-book of:

$$\frac{P}{B} = \frac{\text{ROE} - g}{\text{cost of equity} - g} = \frac{15\% - 4\%}{9\% - 4\%} = \frac{11\%}{5\%} = 2.2\times$$

A business earning 15% on equity that costs 9%, growing 4%, is worth about 2.2 times its book value — the market *should* pay a premium, because the spread is real and (we are assuming) durable. Now run the same formula for a business earning only its cost of equity, ROE 9%:

$$\frac{P}{B} = \frac{9\% - 4\%}{9\% - 4\%} = \frac{5\%}{5\%} = 1.0\times$$

Exactly book value — no premium, because there is no spread. And a business earning *below* its cost of equity, ROE 6%:

$$\frac{P}{B} = \frac{6\% - 4\%}{9\% - 4\%} = \frac{2\%}{5\%} = 0.4\times$$

Forty cents on the dollar of book — a deep discount, because the spread is negative and growth makes it worse. The justified price-to-book is just the spread (and its durability and growth) translated into a multiple. When you see a stock trading at 4× book, the market is telling you it believes in a wide, durable spread. When you see one at 0.5× book, the market is pricing value destruction. Your job is to decide whether the market's implied spread is right.

*A price-to-book multiple is not a separate fact about a stock; it is the spread, its durability, and its growth, compressed into one number — which is why a "cheap" low-multiple stock is often cheap for the excellent reason that it destroys value.*

## The compounder: when all three drivers point the same way

The rarest and most valuable thing in all of investing is a business in which the three drivers reinforce one another: a **wide spread**, a **long runway** to reinvest at that spread, and the **durability** to keep the spread alive for decades. Such a business is a compounding machine, and understanding why requires seeing the three drivers multiply rather than add.

![A single steeply rising convex curve showing invested capital compounding from one billion dollars to roughly nine point six billion dollars over twenty years at a 12 percent reinvestment-driven growth rate, with the curve bending upward in the later years](/imgs/blogs/roic-wacc-spread-the-engine-of-intrinsic-value-7.png)

The figure traces what happens when a high-ROIC business reinvests a large share of its profits at that high return, year after year. Start with \$1,000M of invested capital earning a 20% ROIC. Reinvest 60% of NOPAT (recall: 60% reinvestment at 20% ROIC funds 12% growth). The capital base compounds at 12% a year: \$1,000M becomes \$1,120M, then \$1,254M, then \$1,405M, and so on. After twenty years it has grown to roughly \$9,600M — nearly *ten times* the starting capital — and the whole time, every one of those dollars has been earning an 11-point spread over its 9% cost. The curve is *convex*: it bends upward, because compounding adds the most in the later years, when the base is largest. The first decade roughly triples the capital; the second decade triples it again.

This is the holy grail because it stacks all three drivers on top of one another, multiplicatively. The wide spread means each dollar creates a lot of value. The long reinvestment runway means the company can keep deploying *more and more* dollars into that wide spread, rather than running out of attractive places to put money. And the durability means the spread doesn't fade while all this compounding happens. A business that has all three is not merely valuable — it is valuable in a way that grows super-linearly with time, which is why the great long-term fortunes in equity markets have so often come from holding a single high-ROIC, long-runway compounder for decades and simply letting the arithmetic work.

It is also why such businesses are so rare and so expensive. The market knows what a durable compounder is worth, and it bids the price up accordingly. The hard part of investing in compounders is almost never *recognizing* one in hindsight; it is *paying a price today* that still leaves room for return, and — harder still — being *right* about the durability, because the entire thesis rests on a spread persisting for a decade or two, which is exactly the thing competition is most determined to destroy. The reinvestment runway is the other fragile assumption: a company can have a wonderful spread but nowhere to reinvest, in which case it should return cash to owners rather than force growth into low-return projects — a discipline many managements lack, and a frequent way that a one-time compounder quietly becomes a value-destroyer when its runway runs out.

#### Worked example: the compounding of a 20%-ROIC business

Let us make the compounder concrete with Northwind one final time. Northwind earns a 20% ROIC, reinvests 60% of NOPAT (growing 12% a year), faces a 9% WACC, and — critically — holds its spread for a long time thanks to a genuine moat. Start with \$1,000M of invested capital and \$200M of NOPAT.

- **Year 1:** NOPAT \$200M, reinvest \$120M (60%), pay out \$80M. Capital grows to \$1,120M. Economic profit (spread × capital) = 11% × \$1,000M = \$110M.
- **Year 5:** capital has compounded at 12% to roughly \$1,760M; NOPAT is about \$350M; economic profit is about 11% × \$1,760M ≈ \$194M — *and rising every year* because the capital base on which the spread is earned keeps growing.
- **Year 10:** capital is roughly \$3,100M, NOPAT about \$620M, economic profit about \$340M a year.
- **Year 20:** capital is roughly \$9,600M, NOPAT about \$1,920M, economic profit roughly \$1,050M a year.

The economic profit didn't just persist — it *grew tenfold*, because the spread was earned on an ever-larger base. That is the engine running in its most powerful configuration: a wide spread, reinvested at scale, sustained by durability, compounding for two decades. The terminal value such a business commands is enormous, which is exactly why the [terminal-value post](/blog/trading/equity-research/terminal-value-the-part-that-dominates) warns that the assumption of a persistent spread in the terminal period is the single most consequential — and most dangerous — input in any valuation.

*A compounder's power is that the spread is earned on a base that the spread itself keeps enlarging; get the durability right and time does the rest, get it wrong and the whole tower comes down.*

## How to estimate the spread and its durability for a real company

The framework is only useful if you can apply it to an actual company, so here is the practical procedure, drawing on the tools built across this series.

**Step one: compute ROIC, honestly.** Take operating profit (EBIT), tax it at the company's effective rate to get NOPAT, and divide by invested capital (equity + interest-bearing debt − excess cash). Use several years, not one, because a single year can be flattered by a good cycle or one-time items — the [quality-of-earnings](/blog/trading/equity-research/quality-of-earnings-accruals-one-offs-red-flags) lens matters here. For intangible-heavy businesses (software, brands, pharma), the reported invested capital understates the real capital because accounting expenses R&D and marketing rather than capitalizing them; serious analysts capitalize a portion to get an honest denominator, or the ROIC will look impossibly, meaninglessly high.

**Step two: estimate WACC.** Build the cost of equity from CAPM, take the after-tax cost of debt from the company's borrowing rates, and weight by the capital structure. The [cost-of-capital post](/blog/trading/equity-research/cost-of-capital-and-the-hurdle-rate) covers the mechanics. Don't over-engineer this — a WACC estimate good to within a point or so is fine, because the spread is usually large relative to the uncertainty in WACC, and over-precision here is false precision.

**Step three: compute the spread, and ask whether it's real.** ROIC minus WACC gives you the spread today. But a single year's spread can be a fluke — a temporary shortage, a one-off price spike, a competitor's stumble. Ask whether the spread reflects something *structural* about the business (a real cost advantage, a real network effect) or something *cyclical* that will mean-revert. A structural spread is worth pricing; a cyclical one is a trap.

**Step four — the hard part: judge the durability.** This is where finance ends and business judgment begins, and there is no formula for it. Look at the *history* of the ROIC — has it been stable for a decade, or is it newly elevated? Look at the *source* of the advantage — is it a brand, a network, switching costs, a cost edge, a regulatory moat, and is that source getting stronger or weaker? Look at the *competition* — who is attacking, with what, and how successfully? Look at *capital intensity and reinvestment opportunity* — can the company keep deploying capital at the high return, or is its runway nearly used up? The output of step four isn't a number so much as a judgment about where on the fade curve the company sits and how steep its descent will be. The whole apparatus of competitive-strategy analysis exists to inform this one judgment.

**Step five: synthesize.** Quality is the spread. Runway is the reinvestment opportunity. Moat is the durability. Multiply the three in your head: a wide spread × a long runway × a durable moat is a compounder you should be willing to pay up for; a narrow or fading spread, a short runway, or a weak moat knocks the value down sharply. The market has its own implied view of all three baked into the price (you can extract it with a [reverse DCF](/blog/trading/equity-research/reverse-dcf-and-sensitivity-analysis)); your edge, if you have one, comes from being right about the spread or its durability when the market is wrong.

## Common misconceptions

**"Growth is always good."** The single most expensive error in investing. Growth is a multiplier on the spread; it is good only when the spread is positive, irrelevant when the spread is zero, and *destructive* when the spread is negative. A fast-growing company earning below its cost of capital is destroying value faster the more it grows, no matter how impressive its revenue chart looks. Always ask what return the growth is earning before you celebrate it.

**"A profitable company creates value."** Accounting profit and value creation are different things. A company can report a profit every year and still destroy value, if that profit represents a return below the cost of capital. The income statement charges for the cost of debt (interest) but never charges for the cost of equity, so it systematically overstates value creation. Only the spread — which charges for *all* the capital — tells you whether value is actually being created. This blind spot is also where accounting games hide; the [Enron](/blog/trading/finance/enron-2001-accounting-fraud) and [Wirecard](/blog/trading/finance/wirecard-the-german-fintech-fraud) cases are, at bottom, stories of reported returns that were never real.

**"High ROIC means a great long-term investment."** Not by itself. A high ROIC today is worth little if it fades fast; what matters is the high ROIC *times its durability*. Most high returns are temporary, competed away within a few years. The great businesses are the rare ones whose moats keep the ROIC high for a decade or more. Confusing a high *current* return with a *durable* one is how investors overpay for businesses about to revert to the mean.

**"A low price-to-book stock is cheap."** Often it is cheap for an excellent reason: the business destroys value, so it *should* trade below book. A price-to-book of 0.5× is the market correctly pricing a negative spread. The mistake is to treat a low multiple as automatically a bargain; the multiple is the spread and its durability in disguise, and a low one usually signals a bad spread, not a market error. Cheapness is only opportunity when the market's implied spread is wrong.

**"WACC is just a technical detail for the DCF."** WACC is the hurdle that determines whether anything the company does creates value at all. It is not a plug for a spreadsheet; it is the dividing line between value creation and value destruction. A two-point error in WACC can flip a business from a value-creator to a value-destroyer in your analysis, which is why understanding *what* WACC represents matters far more than computing it to three decimal places.

**"Reinvesting all profits back into the business is always the responsible thing to do."** Only if the business can reinvest at a return above its cost of capital. A company with a positive spread but no remaining runway — nowhere to deploy more capital at the high return — should *return* cash to owners through dividends or buybacks, not force it into low-return projects to manufacture growth. Retaining cash that then earns below the cost of capital is one of the most common quiet value-destroyers, and it is precisely how a former compounder decays once its runway is exhausted.

## How it shows up in real markets

The spread is not an academic abstraction; it is the thing that, over years and decades, separates the great compounding stories from the cautionary tales. A few patterns recur, and once you see them through the lens of the spread, they stop being mysterious.

**The high-ROIC compounders the market adores.** Businesses like a dominant payments network, a branded-consumer-staples giant, or a entrenched enterprise-software platform have historically earned ROICs far above their cost of capital — often 20%, 30%, or more — and, crucially, *sustained* those returns for many years behind moats of network effects, brand, and switching costs. These are the businesses that trade at high multiples of book and earnings, and the high multiple is not (usually) market irrationality; it is the market correctly pricing a wide, durable spread reinvested over a long runway. Warren Buffett's entire philosophy, laid out in the [Berkshire study](/blog/trading/finance/warren-buffett-berkshire-value-investing), is essentially the search for businesses with a high return on capital protected by a durable moat — the spread and its durability, in plain English, decades before "ROIC" became a fashionable acronym. (These figures are illustrative of the pattern, not precise current data for any one company.)

**The value-destroying empire-builders.** The other recurring story is the company — often in a capital-intensive, commoditized industry like airlines, much of heavy manufacturing, or certain stretches of telecom and utilities — that grows revenue impressively for years while earning a ROIC stubbornly below its cost of capital. Management is lionized for the growth; the revenue chart goes up and to the right; and intrinsic value quietly erodes the whole time, because every reinvested dollar earns less than it costs. These businesses are frequently the ones that, after a decade of "growth," have created no shareholder value at all and sometimes destroyed a great deal. The tell is always the same: rising revenue and earnings sitting on top of a ROIC that never clears the hurdle. Growth was the symptom investors cheered; the negative spread was the disease they ignored.

**The fade, in slow motion.** Watch a former high-flyer over a long enough period and you will often see the fade play out in real time: a company that earned spectacular returns when its product was novel and its moat intact, gradually competed down toward its cost of capital as rivals catch up, patents expire, or technology shifts. The stock that was a darling at a 4× book multiple drifts down to a market multiple not because anything dramatic happened, but because the market is steadily revising downward its estimate of how long the spread will last. The most expensive misjudgments in growth investing are almost always durability misjudgments — paying for twenty years of spread and getting five.

**The market as a spread-estimating machine.** Finally, the cleanest way to see the spread in action is to treat the stock price itself as the market's estimate of the spread and its durability, and reverse-engineer it. A [reverse DCF](/blog/trading/equity-research/reverse-dcf-and-sensitivity-analysis) extracts the spread and competitive-advantage period the current price implies. When you do this across a market, you find that high-multiple stocks embed assumptions of wide, durable spreads — sometimes plausibly, sometimes not — and low-multiple stocks embed assumptions of narrow, fading, or negative spreads. Your job as an analyst is never to compute the spread in the abstract; it is to compare *your* honest estimate of the spread and its durability against the one the price already contains, and to act only where you have a genuine, defensible difference of view.

## When this matters, and further reading

The ROIC–WACC spread is the idea that turns the rest of equity research from a collection of techniques into a single, coherent discipline. Every statement you analyze, every ratio you compute, every model you build is, in the end, an instrument for estimating three things: how wide the spread is, how much capital can be reinvested at it, and how long it will last. Master those three and you understand not just whether a business is profitable, but whether it is *creating value* — and that distinction is the whole game.

Use the spread whenever you are tempted to celebrate growth: stop and ask what return that growth is earning, because growth below the cost of capital is wealth destruction wearing a growth costume. Use it whenever a stock looks cheap or expensive on a multiple: the multiple is the spread and its durability in disguise, so the real question is whether the market's implied spread matches your own. And use it whenever you are sizing up a "great business": a high return today is necessary but nowhere near sufficient — durability is co-equal, and judging it is where the real work lies.

To go deeper, the natural next steps are the building blocks this post synthesized and the strategy ideas it points toward:

- [Returns on capital: ROIC, ROE, ROA](/blog/trading/equity-research/returns-on-capital-roic-roe-roa) — the numerator of the spread, built from the statements.
- [The cost of capital and the hurdle rate](/blog/trading/equity-research/cost-of-capital-and-the-hurdle-rate) — where WACC, the other half of the spread, comes from.
- [Building a DCF, part 1: forecasting](/blog/trading/equity-research/building-a-dcf-part-1-forecasting) — how the spread becomes the cash flows a valuation discounts.
- [Terminal value: the part that dominates](/blog/trading/equity-research/terminal-value-the-part-that-dominates) — why the assumption of a persistent terminal spread is the most consequential input in any model.
- [Economic moats: durable competitive advantage](/blog/trading/equity-research/economic-moats-durable-competitive-advantage) — the deep dive on durability, the third driver, and the part of the engine that is hardest to judge and most valuable to get right.
