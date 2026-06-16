---
title: "Interest Rates and Mortgages: The Master Variable That Moves Property"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into why interest rates are the single biggest short-run driver of house prices: how a mortgage turns a monthly payment into a purchase price, how a rate change moves buying power, and why the 2020-21 lows inflated housing while the 2022-23 spike froze it."
tags: ["real-estate", "property", "interest-rates", "mortgage", "affordability", "buying-power", "real-rates", "monetary-policy", "vietnam", "housing"]
category: "trading"
subcategory: "Real Estate"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Interest rates are the single biggest short-run driver of property prices, because almost everyone buys with a mortgage and a mortgage converts a monthly *payment* budget into a purchase *price*.
>
> - A mortgage is a machine that turns a fixed monthly payment into a loan amount, and the loan amount plus your down payment is the price you can bid. The exchange rate of that machine is the interest rate.
> - When rates fall, the same payment supports a bigger loan, so buyers bid prices up; when rates rise, buying power collapses and prices stall or fall. The mechanism is **affordability**, not vague "cheap money".
> - Housing is more rate-sensitive than almost any other asset because it is bought with **heavy leverage** and it is a very **long-duration** asset — a small change in the discount rate moves its value a lot.
> - The one number to remember: when the US 30-year mortgage rate went from **3% to 7%** in 2022, the *same* income bought a house roughly **37% smaller**. That is why housing boomed in 2020-21 and froze in 2022-23.

In January 2021, a software engineer in Austin we'll call **Dana** locked a 30-year mortgage at **2.65%** — the lowest rate ever recorded in the United States. On a budget of \$2,000 a month for principal and interest, the bank would lend her about \$474,000. Add her down payment and she could bid roughly \$593,000 for a house. She felt rich. Houses that had seemed out of reach were suddenly, almost magically, affordable. So were they for every other buyer in her city, all at once, and prices took off.

Twenty-one months later, in October 2022, the *same* mortgage cost **7.08%**. Dana's salary hadn't changed. Her down payment hadn't changed. But now the bank would only lend her about \$301,000 on that same \$2,000 budget — and the house she could bid for had shrunk to about \$376,000. Nothing about the houses had changed. The neighborhoods, the square footage, the schools were all identical. What changed was a single number on a screen at the central bank, and that number quietly erased a third of every buyer's purchasing power in under two years. Sellers who listed in 2022 watched their phones go silent.

Ten thousand kilometres away, a Ho Chi Minh City marketing manager we'll call **Minh** lives the same equation in a different currency and a crueller form. His bank offered a "preferential" rate of about **8%** for the first two years on a ₫5.6 billion (≈ \$216,000) loan — and then a reset to **13%** that he had to read the fine print to find. The rate is the master variable in his story too; it just arrives on a timer.

This post is about the one force that moves property faster and harder than any other in the short run: **the interest rate, transmitted through the mortgage**. The diagram above is the mental model for the whole article — the *same* monthly payment buys a bigger house at a low rate and a smaller house at a high rate, because a mortgage is a converter between payments and prices, and the interest rate sets the exchange rate. We'll build that converter from zero, show exactly how a rate change moves buying power, explain why housing is so much more rate-sensitive than other assets, separate the rate you see (nominal) from the rate that matters (real), trace the lag between rates moving and prices moving, and dissect Vietnam's teaser-then-reset trap. Minh and Dana will run all the way through, in ₫ and \$ side by side.

![Two side by side panels showing the same monthly payment of 1,200 US dollars buying a 285,000 dollar loan and a 356,000 dollar house at a 3 percent rate, versus a 180,000 dollar loan and a 225,000 dollar house at a 7 percent rate](/imgs/blogs/interest-rates-and-mortgages-the-master-variable-1.png)

## Foundations: how a mortgage turns a payment into a price

Before we can talk about rates moving prices, we have to be precise about the machine in the middle. Almost nobody buys a house by writing one cheque. They buy a *monthly payment*. The interest rate is what converts that payment into a purchase price — and once you see the conversion clearly, the whole rest of the post follows mechanically.

### What a mortgage actually is

A *mortgage* is a loan secured by the property: the bank gives you a lump sum today, you pay it back in fixed monthly instalments over a long term (typically 20-30 years), and if you stop paying, the bank can take the house. Each monthly payment does two jobs at once. Part of it is *interest* — the rent you pay the bank for the money you've borrowed. The rest is *principal* — the chunk that actually reduces what you owe. This gradual repayment of principal is called *amortization* (from the Latin for "to kill off" — you slowly kill the debt).

Two features matter for everything that follows. First, the monthly payment is *fixed* in a fixed-rate mortgage — it's the same number in year 1 and year 25. Second, in the early years almost all of each payment is interest, because the balance you owe is still huge. That second fact is why mortgages are so sensitive to the interest rate: in the early years, you are mostly paying for the *rate*, not paying *down* the loan.

### Why the early payments are almost all interest

This deserves a moment, because it's the most counter-intuitive feature of a mortgage and it's load-bearing for the whole rate story. Each month, the interest portion is the *current* balance times the monthly rate. At the start of a 30-year loan the balance is at its maximum, so the interest charge is at its maximum, and only the small leftover goes to principal. As the balance shrinks, the interest charge shrinks and the principal portion grows — so the *same* fixed payment slowly tilts from mostly-interest to mostly-principal over the life of the loan. The line that tracks this tilt is the *amortization schedule*.

The practical upshot: in the first several years of a high-rate loan, a homeowner is renting money from the bank and barely owns more of the house than the day they signed. That is why the rate matters so much more than people expect — for a long stretch of the loan, the payment *is* the interest. Lower the rate, and a far bigger share of that fixed payment goes to actually buying the house; raise it, and the payment vanishes into the bank's pocket.

#### Worked example: where the first month's payment goes

Take Dana's \$300,000 loan over 30 years. At a **3%** rate, the monthly interest charge in month 1 is $300{,}000 \times (0.03/12) = \$750$. Her total payment is about \$1,265, so roughly \$515 of the first payment actually pays down the loan. At a **7%** rate, month-1 interest is $300{,}000 \times (0.07/12) = \$1{,}750$. Her payment is now about \$1,996, so only \$246 reduces the balance — *less than half* the principal she'd build at 3%, on a payment that's \$731 a month higher.

*At a high rate, you pay more and own less — the early years of a costly mortgage are almost pure rent to the bank.*

### The payment formula (the one equation in this post)

Here is the single equation that drives property prices in the short run. For a loan of amount $L$ at a monthly interest rate $i$ over $n$ months, the fixed monthly payment $M$ is:

$$M = L \cdot \frac{i\,(1+i)^n}{(1+i)^n - 1}$$

Don't be intimidated — we will almost never use it forwards. $L$ is the loan, $i$ is the *monthly* rate (the annual rate divided by 12: a 6% annual rate is $i = 0.06/12 = 0.005$ per month), and $n$ is the number of months ($30 \times 12 = 360$ for a 30-year loan). The formula just says: spread the loan plus all its interest over $n$ equal payments.

What we actually care about is the formula *backwards* — given a payment you can afford, how big a loan does it support? Solving for $L$:

$$L = M \cdot \frac{1 - (1+i)^{-n}}{i}$$

Here $M$ is the monthly payment you can budget, and $L$ is the loan that payment buys. This is the **payment-to-price conversion**, and it is the heart of the entire post. A buyer doesn't decide "I'll spend \$300,000." A buyer decides "I can afford \$2,000 a month," and the rate decides how much house that becomes.

#### Worked example: a fixed budget buys a much bigger loan at a low rate

Let's run Dana's numbers. She budgets $M = \$2{,}000$ a month for principal and interest on a 30-year loan ($n = 360$).

At a **3%** rate, the monthly rate is $i = 0.03/12 = 0.0025$. Plugging in:

$$L = 2000 \cdot \frac{1 - (1.0025)^{-360}}{0.0025} \approx \$474{,}000$$

At a **7%** rate, $i = 0.07/12 = 0.005833$:

$$L = 2000 \cdot \frac{1 - (1.005833)^{-360}}{0.005833} \approx \$301{,}000$$

Same \$2,000 a month. Same person, same job, same savings. At 3% it buys a \$474,000 loan; at 7% it buys only \$301,000 — a drop of about **37%**. The rate didn't change Dana's income by a cent. It changed what her income is *worth* as buying power.

*The mortgage is a converter, and the interest rate is the exchange rate between your monthly payment and the price you can pay.*

### Buying power, price, and the down payment

The loan is most of the story, but not all of it. The **price** a buyer can bid is the loan plus the **down payment** — the cash they put in up front. If Dana puts 20% down, then the loan is 80% of the price, so:

$$\text{Price} = \frac{\text{Loan}}{1 - \text{down-payment fraction}} = \frac{\text{Loan}}{0.80}$$

At 3%, her \$474,000 loan supports a price of $474{,}000 / 0.80 \approx \$593{,}000$. At 7%, her \$301,000 loan supports $301{,}000 / 0.80 \approx \$376{,}000$. We call the maximum price a buyer can support their **buying power** (or "purchasing power" / "affordability"). It is the single most important quantity in short-run housing, and it is set almost entirely by two things: the payment they can budget, and the interest rate.

Notice that the down payment barely moves with rates — Dana's savings are her savings. It's the *loan* that swings, and because the loan is the large majority of the price (80% in this example), the swing in the loan drives almost the entire swing in buying power. This is the first hint of why **leverage** makes housing so rate-sensitive; we'll return to it.

### The policy rate vs the mortgage rate

People say "the central bank raised rates" as if there's one rate. There are many, and the distinction matters. The **policy rate** is the very short-term rate the central bank sets directly (the Fed funds rate in the US, the refinancing/OMO rates the State Bank of Vietnam steers). The **mortgage rate** is what a bank charges *you* for a 30-year home loan. The mortgage rate is built up from the policy rate plus the cost of long-term funding plus the lender's margin and risk premium. The two move together over time but not lockstep — the mortgage rate can rise even when the policy rate is flat (if long-term bond yields rise), and it doesn't fall instantly when the central bank cuts.

For housing, the mortgage rate is the one that lands on the buyer. But the policy rate is usually the *trigger*, because it sets the tone for the whole interest-rate structure. When we say "rates moved property," the causal chain is policy rate $\to$ bond yields and funding costs $\to$ mortgage rate $\to$ buyer budgets $\to$ prices. We'll trace that chain explicitly in the section on transmission lag.

### Fixed vs floating

A **fixed-rate** mortgage locks the rate (and so the payment) for the life of the loan, or for a long initial period. A **floating** (or "adjustable" / "variable") mortgage has a rate that resets periodically — usually tied to a reference rate plus a margin — so the payment changes over time. The US market is dominated by long fixed-rate loans (the famous 30-year fixed). Vietnam's market is effectively floating: a short *teaser* fixed period, then a reset to a floating rate. This single structural difference is why a rate shock plays out completely differently in the two countries, as we'll see.

### Nominal vs real interest rates

One more foundation, and it's the one most people get wrong. The **nominal** rate is the headline number — the 7% on the loan document. The **real** rate is the nominal rate *minus inflation* — what you pay after accounting for the fact that money (and wages, and house prices) are themselves losing value:

$$\text{real rate} \approx \text{nominal rate} - \text{inflation rate}$$

A *basis point*, by the way, is one hundredth of a percent (0.01%); a rate moving "from 6.00% to 6.50%" is a move of 50 basis points. We'll use the term a few times. The real-rate idea will get its own section, because it explains why an 8% mortgage can be cheap in one country and crushing in another — but plant the seed now: the rate that truly governs the cost of borrowing is the real one.

With the machine defined, we can now do the thing this whole post is about: turn the rate dial and watch the price move.

## How a rate change moves buying power: the core mechanism

The single most important idea in short-run real estate is this: **a buyer's maximum price moves inversely with the interest rate, and it moves a lot.** Let's make it visual and exact.

![A downward sloping curve showing the maximum 30-year loan a fixed 1,200 dollar monthly payment supports as the mortgage rate rises, with the 3 percent point marked at about 285,000 dollars and the 7 percent point at about 180,000 dollars](/imgs/blogs/interest-rates-and-mortgages-the-master-variable-2.png)

The curve above is the payment-to-price conversion drawn out. On the horizontal axis is the mortgage rate; on the vertical axis is the maximum loan a *fixed* payment (here \$1,200 a month) can support. The curve falls steeply as rates rise. At 3% the payment buys a \$285,000 loan; at 7% it buys only \$180,000. Same payment, every month. The shaded band between 3% and 7% is the buying power that the 2022 rate shock simply deleted.

Three things about this curve are worth dwelling on, because they're the mechanism.

### The curve is steep, and steepest where rates are low

Buying power doesn't fall in a gentle straight line — it falls fast, and it falls fastest when rates are already low. Going from 2% to 3% costs more buying power than going from 9% to 10%. Why? Because at low rates, a 1-percentage-point change is a *large proportional* change in the cost of money (3% is 50% more than 2%), and because the long 30-year term magnifies small rate differences into big payment differences. This is why the moves from a 2.65% record low were so explosive in both directions: housing was sitting on the steepest part of the curve.

### It's the payment that's fixed, not the price

The reason rates move prices is that buyers anchor on what they can *pay each month*, not on a sticker price. A household that can comfortably handle \$2,000 a month will stretch to a bigger house when rates fall (because that \$2,000 now buys more loan) and is forced into a smaller one when rates rise. Multiply that across every buyer in a market and the *clearing price* — the price at which buyers and sellers actually transact — moves with the rate. Sellers can ask whatever they like; buyers can only bid what their payment supports.

### Demand is what moves, and it moves all at once

Here's the subtle part. A rate change doesn't gently nudge a few buyers. It hits *every* mortgage buyer in the market simultaneously, because they're all borrowing in the same rate environment. When the rate falls, every buyer's budget expands at once, so they all bid more aggressively against each other for the same fixed stock of houses — and prices jump. When the rate rises, every buyer's budget shrinks at once, bids collapse, and either prices fall or transactions simply stop (we'll see why "stop" is more common than "fall" when we discuss the freeze). Because the supply of houses can't change quickly, the entire shock lands on price and volume.

### Why a rate cut runs the whole chain in reverse — and loops

It helps to trace a *cut* step by step, because the chain is what people miss when they say "low rates pump prices" without saying *how*. A cut lowers the mortgage rate, which lowers the monthly cost of any given loan, which means the same budget now supports a bigger loan, which lets buyers bid a higher price, which lifts the clearing price as those higher bids compete. And then it *loops*: rising prices pull in still more buyers (fear of missing out, plus owners trading up on their new equity), whose bids push prices higher again, until either supply finally responds or affordability runs out at the new, higher price. The diagram below lays out that ripple.

![A pipeline of seven steps showing a central bank rate cut leading to cheaper mortgages, a lower monthly payment, a bigger supportable loan, a higher buyer bid, a higher clearing price, and then more demand looping back to push bids higher still](/imgs/blogs/interest-rates-and-mortgages-the-master-variable-7.png)

The loop is why low-rate booms tend to *overshoot* rather than settle at a tidy new equilibrium: the price rise itself becomes a reason to buy, feeding back into more demand. A rate hike runs the identical chain in reverse — except, as we'll see, the downside path tends to freeze transactions rather than crash prices, because the loop that powered the boom doesn't run as cleanly in reverse.

#### Worked example: a US buyer's max price drops about 30% when rates go 3% to 7%

Let's make the headline number concrete with Dana, putting 20% down both times.

At **3%**: loan \$474,000, so max price $= 474{,}000 / 0.80 = \$592{,}500 \approx \$593{,}000$.

At **7%**: loan \$301,000, so max price $= 301{,}000 / 0.80 = \$376{,}250 \approx \$376{,}000$.

The drop in her max price is $(593{,}000 - 376{,}000) / 593{,}000 \approx 37\%$. (The *loan* drops 37% and so does the price, because her down-payment percentage is the same; if she instead kept her down payment fixed in dollars rather than as a percentage, the price drop would be a bit smaller, around 30% — which is the number you'll often hear quoted.) Either way, a 4-percentage-point rate move took roughly a third off what a fixed income can buy.

*Rates don't change the house; they change the number of houses a given salary can reach — and in 2022 that number fell by a third in under two years.*

![Two side by side panels showing Dana with a 2,000 dollar monthly budget supporting a 474,000 dollar loan and a 593,000 dollar maximum price at a 3 percent rate in early 2021, versus a 301,000 dollar loan and a 376,000 dollar maximum price at a 7 percent rate in late 2022, about 37 percent less](/imgs/blogs/interest-rates-and-mortgages-the-master-variable-6.png)

The figure above is the 2022 squeeze drawn out. The two panels share Dana's unchanged \$2,000 budget and her unchanged down payment; only the rate moves, from 3% to 7%. That single change drops the loan her budget supports from \$474,000 to \$301,000 and her maximum price from \$593,000 to \$376,000 — about 37% less house, bought by the same person with the same income. This is the picture sellers across America woke up to in late 2022: their pool of qualified buyers could suddenly only bid two-thirds of last year's price.

#### Worked example: Minh's ₫50 million payment buys far more loan at 8% than at 13%

Now Vietnam, where the same mechanism runs in dong and at much higher rates. Suppose Minh can budget **₫50 million a month** (≈ \$1,930) for principal and interest on a 20-year loan ($n = 240$). (Vietnamese mortgages are often shorter than US ones, which makes them even more payment-heavy.)

At the **8%** teaser rate, $i = 0.08/12 = 0.006667$:

$$L = 50{,}000{,}000 \cdot \frac{1 - (1.006667)^{-240}}{0.006667} \approx ₫5.98 \text{ billion} \ (\approx \$231{,}000)$$

At the **13%** reset rate, $i = 0.13/12 = 0.010833$:

$$L = 50{,}000{,}000 \cdot \frac{1 - (1.010833)^{-240}}{0.010833} \approx ₫4.27 \text{ billion} \ (\approx \$165{,}000)$$

The *same* ₫50 million a month supports a ₫5.98 billion loan at 8% but only ₫4.27 billion at 13% — about **29% less**. (If we use a 30-year term, the gap is even wider: ₫6.81bn at 8% versus ₫4.52bn at 13%, a 34% drop.) Minh's payment hasn't moved. The rate has eaten a third of his buying power.

*In dong or in dollars, the law is the same: the higher the rate, the smaller the house a fixed payment buys — and Vietnam's rates are high to begin with.*

### What sets the payment budget in the first place: the income cap

We've been treating the monthly payment as a free choice, but in practice the bank caps it. Lenders limit your payment to a fraction of your income — a *debt-to-income* (DTI) or *debt-service* ratio, typically that total debt payments not exceed roughly 35-45% of gross income. This cap is what ultimately ties buying power to *income*, and it's why the rate matters so directly: for a fixed income, the bank fixes the maximum *payment*, and the rate then fixes the maximum *loan* that payment can carry. The buyer isn't choosing to stretch; the rate is doing the stretching (or the squeezing) for them, within an income-set ceiling.

This is also why a rate rise can lock a buyer *out entirely* rather than merely down-sizing them. If a household needs a certain house to live near work and schools, and the rate rise pushes the required payment above the bank's DTI cap on their income, the loan is simply refused — they don't get a smaller house, they get *no* mortgage. A rate move doesn't just shift the price every buyer can pay; at the margin it removes whole cohorts of buyers from the market, which is why demand can fall faster than a smooth curve suggests.

#### Worked example: how the rate decides whether Minh qualifies at all

Suppose Minh earns ₫120 million a month (≈ \$4,630) and his bank caps housing debt service at 40% of income — so his maximum payment is ₫48 million a month. He wants the ₫5.6 billion loan over 20 years.

At the **8% teaser**, that loan costs about ₫46.8 million a month — *just under* his ₫48 million cap. He qualifies, barely.

At the **13% reset rate**, the *same* loan costs about ₫65.6 million a month — far *above* his ₫48 million cap. On those terms the bank would never have approved him. He only got the loan because the teaser rate let him squeak under the ceiling; the reset pushes his payment to a level his income could never have qualified for in the first place.

*The teaser doesn't just lower the early payment — it lets a buyer qualify for a loan whose true, reset cost their income can't actually support. That gap is the trap.*

## Why housing is so rate-sensitive: leverage and duration

Lots of assets are affected by interest rates. Housing is affected *more* than almost any of them, and for two compounding reasons: it's bought with heavy **leverage**, and it's an extremely **long-duration** asset. Understanding these two is what separates a real grasp of housing from the folk wisdom of "low rates good, high rates bad."

### Leverage amplifies the rate's effect on the buyer's equity

Most people buy a house with 10-30% of their own money and 70-90% borrowed. That borrowed share is **leverage**, and it has a dramatic consequence: a small percentage change in the house's price becomes a large percentage change in the buyer's *equity* (their own money in the deal). We cover this in depth in our piece on [leverage and the mortgage](/blog/trading/real-estate/leverage-and-the-mortgage-how-debt-amplifies-property), but here's the part that matters for rates.

Because buyers are leveraged, they're priced off the *loan* they can carry, and the loan is the part of the price most sensitive to the rate. When Dana's down payment is 20% and her loan is 80%, an interest-rate move that swings the *loan* by 37% swings her whole *price* by 37% too — because the loan is the large majority of the price. If buyers paid cash (no leverage), a rate change would affect them only through the modest opportunity cost of their money, and housing would be far less rate-sensitive. It's precisely *because* housing is bought on credit that the credit rate is its master variable. The broader feedback between credit availability and prices is the subject of our [credit cycle and property prices](/blog/trading/real-estate/the-credit-cycle-and-property-prices) deep dive.

### Duration: housing is a very long-lived stream of value

The second reason is more abstract but just as powerful. In finance, *duration* measures how sensitive an asset's value is to a change in the interest rate used to discount its future cash flows — and long-lived assets have high duration. (For the full intuition on why long-duration assets swing hardest with rates, see [price and yield, the seesaw at the heart of bonds](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) and the bond literature it links to.)

A house is about the longest-duration real asset an ordinary person owns. Think of what a house "pays" you: decades of housing services (the rent you'd otherwise pay), stretching out 30, 50, 100 years into the future. The *value* of a house is, in effect, the present value of all that future housing — and the lower the interest rate, the more those distant years are worth today. When rates fall, the far-future value of a house gets discounted less harshly, so its present value rises. When rates rise, those distant years get discounted away and the value falls.

This is the same mechanism that makes a 30-year bond move more than a 2-year bond when rates change: the longer the stream of future value, the more a change in the discount rate matters. A house is a 30-to-100-year "bond" that you can live in. That long duration is *intrinsic* to housing — it's why even cash buyers, who use no leverage, should rationally pay more for a house when rates are low (their alternative — earning interest on the cash — is worth less).

So leverage and duration stack: leverage makes the buyer's *equity* hyper-sensitive to price, and duration makes the price itself sensitive to the rate. Put them together and you get an asset class whose value can swing 30%+ on a few percentage points of interest-rate change. No wonder rates are the master variable.

### Why rates matter more for housing than for most assets

It's worth saying explicitly why housing is *unusually* rate-sensitive even among financial assets. A typical stock is bought with cash (no leverage for most retail investors) and its value rests partly on growing earnings that can outrun rates; gold pays no cash flow and trades on different forces entirely. Housing is the rare asset that combines *both* amplifiers at once and at extreme settings: it is bought with more leverage than almost anything an individual ever owns (often 5-10x their cash via the mortgage), *and* it is one of the longest-duration assets there is (decades of housing services). Add a third factor — housing is a *necessity* with a deep, near-universal buyer base, so a change in financing terms moves a huge fraction of the population's bid simultaneously — and you have the most rate-sensitive large asset class on Earth. The total stock of global real estate is worth roughly \$393 trillion, more than global equities and bonds *combined*, and a meaningful slice of it reprices every time the rate dial turns. That is why central bankers watch housing so nervously: it is the biggest, most leveraged, most rate-sensitive thing they move when they touch rates.

#### Worked example: why even an all-cash buyer cares about the rate

Suppose a cash buyer is choosing between a house and parking the money in a safe bond. A house that delivers roughly ₫240 million a year in housing value (the rent she'd otherwise pay on a ₫6 billion HCMC flat, say a 4% gross yield).

When safe bonds yield **3%**, that ₫240 million stream is "worth" about $240 / 0.03 = ₫8$ billion as a perpetual stream (a rough capitalisation). When bonds yield **8%**, the same ₫240 million is worth only $240 / 0.08 = ₫3$ billion. The house *services* are identical; the rate at which we capitalise them collapsed the value. A rational cash buyer would pay far more for that house when rates are low — because her alternative (lending the money out) pays so little.

*Even with zero borrowing, low rates raise what a house is worth, because they lower the return on every alternative use of the money — that's duration, not leverage.*

## Nominal vs real rates: what really matters

Now we pay off the seed planted in Foundations. The rate on the loan document is the **nominal** rate. The rate that actually governs the *burden* of the debt is the **real** rate — nominal minus inflation. Confusing the two is the single most common mistake people make about rates and housing.

![A four row matrix comparing a low real rate scenario and a high real rate scenario across nominal rate, inflation, the resulting real rate, and the cost of carrying the loan](/imgs/blogs/interest-rates-and-mortgages-the-master-variable-4.png)

The matrix above makes the point with one comparison. Take an 8% nominal mortgage in two different worlds. In a world with 6% inflation and fast-rising wages, the *real* rate is only about +2% — and because your salary (and the house's price) are inflating, the fixed loan balance erodes quickly in real terms. The 8% feels survivable, even cheap. In a world with 1% inflation and flat wages, the same 8% nominal rate is a +7% real rate — a genuinely heavy, lasting burden, because nothing is inflating away the debt and your income isn't catching up. **Same sticker rate, opposite reality.**

### Why real rates are the honest measure

Two forces hide inside a high nominal rate. First, inflation silently shrinks the real value of your fixed monthly payment over time: a ₫50 million payment that feels enormous today is trivial in 20 years if wages have tripled. Second, inflation lifts nominal house prices and nominal wages, so the loan you took out shrinks as a fraction of everything else. Both forces *help the borrower* and both are invisible if you look only at the nominal rate. The real rate nets them out.

This is also why historically high nominal mortgage rates didn't always crush housing. US mortgage rates were *18%* in 1981 — but inflation was running 10-13%, so the real rate was only mid-single digits, and crucially, wages were inflating fast enough to outrun the fixed payment within a few years. The 7% of 2023, against ~3% inflation, was a *higher real rate* than the 18% of 1981 — which is part of why it bit so hard. The number on the loan is not the burden; the number minus inflation is.

For the full treatment of why real yields, not nominal ones, are the master signal across all asset markets, see our companion piece [real vs nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).

#### Worked example: the real-rate view of Minh's 8% versus 13% reset

Back to Minh in HCMC. His teaser rate is 8% nominal. Vietnamese inflation has run roughly 3-4% in recent years; let's use 4%, and assume his wages and HCMC nominal prices rise broadly in line with growth-plus-inflation, call it 8% a year.

At the **8% teaser**, the real rate is about $8\% - 4\% = +4\%$. Heavy, but with nominal HCMC prices rising ~8% a year, his fixed ₫5.6 billion loan shrinks fast relative to the (rising) value of the flat and his (rising) salary.

At the **13% reset**, the real rate jumps to about $13\% - 4\% = +9\%$. Now the loan is genuinely expensive in real terms — he's paying 9% above inflation to carry the debt — *and* the higher payment hits before his wages can grow into it. The reset is brutal not just because 13 is bigger than 8, but because it pushes the *real* cost from "manageable" to "punishing" in one step.

*The headline rate tells you the payment; the real rate tells you the burden — and Minh's reset doubles the real burden, not just the sticker.*

## The transmission lag: rates move before prices

If rates are the master variable, why don't house prices move the instant the central bank acts? Because the rate change has to travel through a chain of human decisions before it lands in a published price — and each link takes time. This *transmission lag* is why rates are a *leading* indicator of prices: by the time the price index confirms a turn, the rate that caused it moved many months earlier.

![A five step timeline showing a policy rate move at month zero, mortgage rates following within weeks, buyer budgets resetting within months, transactions shifting over several months, and the price index printing six to eighteen months later](/imgs/blogs/interest-rates-and-mortgages-the-master-variable-5.png)

Walk the chain in the timeline above:

1. **The policy rate moves (month 0).** The central bank cuts or hikes overnight. This is instant and public.
2. **Mortgage rates follow (weeks 1-6).** Lenders reprice new loans as funding costs and long-term yields adjust. This is fast but not instant — and mortgage rates can move *ahead* of the policy rate if bond markets anticipate it.
3. **Buyer budgets reset (months 1-3).** Buyers who were pre-approved at the old rate get re-quoted; new buyers run new numbers. The maximum loan their payment supports changes — but people don't re-shop their budget the same day the rate moves.
4. **Transactions shift (months 2-6).** Deals already in progress close at old terms; new deals get bid, negotiated, and signed at the new affordability level. Real-estate transactions are slow — searching, bidding, financing, and closing take months.
5. **The price index prints (months 6-18).** Published indices (Case-Shiller in the US, Ministry of Construction averages in Vietnam) are built from *completed* sales, often with a reporting lag of months. So the rate change you saw on the news shows up in the headline price number a year or more later.

The whole chain means a rate move today is a price move *coming*, not a price move *now*. This is why experienced market-watchers treat mortgage rates and mortgage *applications* as leading indicators and the price index as a lagging confirmation — the topic of our [leading indicators](/blog/trading/real-estate/the-credit-cycle-and-property-prices) discussion in the credit-cycle piece.

The lag also explains a recurring confusion. When rates spiked in 2022, prices in many markets didn't immediately *crash* — they froze. Volume collapsed long before prices did, because the first thing a rate shock kills is the *transaction*, not the *price*. Sellers anchored on yesterday's prices and simply refused to cut; buyers couldn't afford yesterday's prices and simply stopped bidding. The result was a stand-off: few sales, sticky asking prices, and only a gradual grind lower where forced sellers appeared. The lag is why "rates up = prices down" is true *eventually* but rarely *immediately*.

### The lag is also why the policy mistake is invisible until it's too late

There's a second, more dangerous consequence of the transmission lag. Because prices respond to rates with a delay of a year or more, a central bank that holds rates too low for too long won't see the overheating in the price index until the boom is already large. By the time the headline price number flashes red, the affordability mechanism has been pumping for a year — and the central bank now has to *raise* rates into an already-inflated market, which then takes another year to cool. The lag turns rate policy into steering a ship with a long delay between the wheel and the rudder: by the time you see the boat turning, you've already over-corrected. This is one reason housing cycles are so prone to overshoot in both directions, and why so many booms end in a freeze or a bust rather than a soft landing. The rate is the master variable, but it's a master variable with a slow, treacherous response time.

#### Worked example: counting the months from a cut to a price bump

Suppose the State Bank of Vietnam cuts its policy rate in March (as it did, four times, across 2023). Trace it:

- **March:** policy rate cut announced.
- **April-May:** banks trim new mortgage rates and advertise fresh teaser deals.
- **May-July:** buyers who were sidelined run new numbers; a ₫50 million payment now reaches a bigger loan, so a few re-enter the market.
- **July-November:** deals get agreed and signed at slightly firmer prices; primary developers quietly stop discounting.
- **Late year into next:** the average-price series ticks up as those completed sales feed the index.

A cut in March shows up as a visible price bump perhaps 6-12 months later. *If you wait for the price index to confirm the turn, you are reading news that the rate already told you three quarters ago.*

## Vietnam's wrinkle: the teaser-then-reset trap

Everything so far applies worldwide. But the *structure* of the mortgage changes how the rate shock is delivered — and Vietnam's structure turns the rate into a time bomb on a timer. This is the part of the story a US-centric explanation misses entirely.

In the US, the dominant product is the 30-year *fixed*. Once Dana locks 2.65%, she keeps 2.65% for 30 years no matter what rates do. This has a famous side effect — the *lock-in effect* — that we'll cover under misconceptions. In Vietnam, almost no one gets a 30-year fixed. Instead, banks offer a **preferential "teaser" rate** — often around 8%, and in some 2024 promotions as low as 5.3-7.2% — fixed for the first **12 to 24 months**. After the teaser expires, the loan *resets* to a **floating rate**, which has recently meant **12-14%** (one state-owned bank's post-preferential ceiling was cited at 13.9% in 2026). The floating rate is typically defined as a reference rate plus a fixed margin, so it also drifts with policy after the reset.

The teaser is a marketing device: it makes the early payment look affordable and gets the buyer to sign. But the buyer is exposed to the *full* rate the moment the teaser ends — and to whatever rates do after that. The rate is still the master variable; it just arrives on a delay the buyer often underestimates.

#### Worked example: Minh's teaser-to-reset payment shock

Minh borrows ₫5.6 billion (≈ \$216,000) over 20 years. (We'll use the standard payment formula forwards now.)

During the **8% teaser** ($i = 0.006667$, $n = 240$):

$$M = 5{,}600{,}000{,}000 \cdot \frac{0.006667 \,(1.006667)^{240}}{(1.006667)^{240} - 1} \approx ₫46.8 \text{ million / month}$$

After the **13% reset** ($i = 0.010833$, on the remaining balance — but for clarity let's reprice the full term at 13%):

$$M = 5{,}600{,}000{,}000 \cdot \frac{0.010833 \,(1.010833)^{240}}{(1.010833)^{240} - 1} \approx ₫65.6 \text{ million / month}$$

His payment jumps from about **₫46.8 million to ₫65.6 million** — a **40% increase**, roughly ₫18.8 million (≈ \$725) more every month, overnight, on a salary that did not rise 40%. (In practice the reset reprices the *remaining* balance over the *remaining* term, which pushes the payment even higher because less principal has been paid off in only two years.) This is *payment shock*, and it is the Vietnamese household's version of the 2022 rate squeeze — except it's guaranteed by the loan structure, not triggered by the central bank.

*A US buyer who locks a low fixed rate is insulated from rate hikes; a Vietnamese buyer on a teaser is signed up for the full rate in advance, on a fuse measured in months.*

### Why the teaser structure makes Vietnamese prices extra rate-sensitive

Because so many Vietnamese borrowers are on floating or about-to-reset loans, a rise in rates hits *existing* owners' payments, not just *new* buyers' budgets. When rates rose and the bond market froze in 2022-23 (after the Tân Hoàng Minh and Vạn Thịnh Phát arrests), the squeeze hit two ways at once: new buyers' affordability collapsed *and* existing teaser borrowers faced resets they couldn't cover. The result was forced selling pressure on top of frozen demand — the same rate shock, amplified by the loan structure. We trace this episode in detail in [Vietnam property cycles 2007, 2013, 2022](/blog/trading/real-estate/the-credit-cycle-and-property-prices).

## Common misconceptions

Rates and housing generate more confident-but-wrong folk wisdom than almost any topic in finance. Here are the five that cost people the most.

### "Low rates always help buyers"

This feels obviously true and is half wrong. Low rates *expand* every buyer's budget — but they expand *every* buyer's budget at once, so buyers bid the saving straight into higher prices. The household ends up with the *same* monthly payment for a house that now costs more. Low rates make houses easier to *finance* but more *expensive to own*, and they transfer wealth to people who already own (whose homes rise in value) at the expense of people trying to buy in. The 2020-21 boom is the proof: rates hit record lows and first-time buyers found it *harder*, not easier, to get on the ladder, because prices outran the rate saving. Cheap money helps the *first* buyers through the door and hurts everyone who arrives after prices have adjusted.

### "Rates don't matter if you pay cash"

We did the math on this one. Even with zero borrowing, the rate sets the return on every *alternative* use of your cash. When safe bonds yield 1%, a house yielding 4% in housing services looks compelling and a cash buyer will rationally pay up for it; when bonds yield 8%, that same house looks expensive next to the bond and the cash buyer holds out for a lower price. This is the *duration* channel, and it means rates move prices even in a market of pure cash buyers — just more slowly and less violently than in a leveraged one. Cash insulates you from the *payment*, not from the *valuation*.

### "High rates always crash prices"

Not immediately, and sometimes not at all in nominal terms. The first casualty of a rate spike is *volume*, not price: transactions freeze as sellers refuse to cut and buyers can't pay. Nominal prices are *sticky downward* — people would rather not sell than sell at a loss — so a rate shock often produces a long, quiet freeze with flat-to-slightly-down prices and collapsed turnover, not a crash. A real crash needs *forced sellers* (job losses, resets borrowers can't cover, leveraged developers facing maturities). Where those appear, prices fall hard; where they don't, the market just goes to sleep until rates come back down. The 2022-23 US experience was a freeze, not a 2008-style crash, precisely because most owners had locked low fixed rates and weren't forced to sell.

### "The Fed sets mortgage rates"

The central bank sets the *policy* rate — a very short-term rate. The 30-year mortgage rate is driven mostly by *long-term* bond yields (which reflect the market's expectations of future inflation and rates over decades) plus a lender margin. The two are correlated but can diverge: long-term mortgage rates sometimes *fall* when the central bank *hikes* (if the market expects the hikes to crush future inflation), and *rise* when it cuts. The mortgage rate is a market price for long-term money, not a number the central bank dials directly.

### "Buy now before rates rise" (and its mirror, "wait for rates to fall")

Both halves of this are traps because they ignore that the *price* moves opposite to the rate. If you rush to buy *before* rates rise, you are buying at the top of the price range that low rates created — you lock a cheap rate onto an expensive house. If you wait to buy *until* rates fall, every other waiting buyer pounces at the same moment and bids the price up, so the lower rate you waited for is partly or wholly eaten by the higher price you now pay. The rate and the price are two ends of the same seesaw, set by the affordability mechanism: you rarely get a cheap rate *and* a cheap price at the same time, because the cheap rate is what made the price expensive. The honest framing isn't "rates up so buy now" or "rates down so wait" — it's "what total monthly payment can I sustainably afford, on the *reset* rate if I'm on a teaser, through a full cycle?" That payment, not the headline rate, is the thing to anchor on.

### "A lower rate is always worth waiting for"

The lock-in effect shows why this is subtler than it sounds. Millions of US owners are sitting on 3% mortgages they locked in 2020-21. They *can't afford to move*, because trading a 3% loan for a 7% one would balloon their payment — so they stay put, which strangles the supply of homes for sale, which keeps *prices* high even as high rates "should" be pushing them down. Low rates you locked in the past become a golden handcuff that distorts the whole market. And for a buyer waiting for rates to fall: when they do, every other waiting buyer pounces too, and prices jump — so the lower rate you waited for can be more than eaten by the higher price you now pay. "Marry the house, date the rate" (refinance later) is the industry's answer, but it's a bet, not a guarantee.

## How it shows up in real markets

The mechanism isn't theoretical. It is the dominant explanation for the biggest housing moves of the last several years, in country after country.

### The 2020-21 global boom: rates at the floor

When the pandemic hit in 2020, central banks slashed rates to emergency lows and bought bonds to push long-term yields down further. The US 30-year mortgage rate fell to a record **2.65% in January 2021**. The effect on buying power was enormous: a fixed payment suddenly reached a far bigger loan, every buyer's budget ballooned at once, and they bid the saving straight into prices. The US Case-Shiller National index, which had recovered to about 230 pre-pandemic, surged to about **308 by June 2022** — a roughly one-third rise in barely two years, the fastest nominal appreciation on record. The same script ran in Canada, Australia, the UK, New Zealand, and across much of Europe and Asia: synchronized rate cuts produced a synchronized housing boom. It wasn't mysterious "cheap money sloshing around" — it was the affordability mechanism, with the rate on the steepest part of the curve.

### The 2022-23 freeze: the fastest mortgage-rate spike in 40 years

Then inflation forced central banks to reverse, hard. The US 30-year mortgage rate rocketed from 2.65% to **7.08% by October 2022** and peaked near **7.79% in October 2023** — more than a *tripling* in under two years, the sharpest jump since the early 1980s.

![A line chart of the US 30 year fixed mortgage rate from 2020 to 2026 showing the record low of 2.65 percent in January 2021, the spike past 7 percent in October 2022, the 7.79 percent peak in October 2023, and a decline to 6.52 percent by June 2026](/imgs/blogs/interest-rates-and-mortgages-the-master-variable-3.png)

The chart above tells the whole story of the master variable in one line. And the housing market's response was exactly what the affordability mechanism predicts: buying power collapsed (the same income now bought ~30-37% less house), *transactions* froze (US existing-home sales fell to multi-decade lows), but nominal *prices* held up surprisingly well — the Case-Shiller index barely dipped before grinding higher again to ~322 by 2024 and ~330 by early 2026. Why didn't prices crash despite the worst affordability in a generation? Two reasons from this post: the lag (the shock hits volume first), and the lock-in effect (owners with 3% loans refused to sell, choking supply). High rates froze the market without crashing it — a freeze, not a 2008. By June 2026 the rate had eased back to **6.52%**, and the market began, slowly, to thaw — exactly the transmission lag running in reverse.

### Vietnam 2022-23: the rate shock meets the bond freeze

Vietnam's version was sharper because of the loan structure and a simultaneous credit shock. Through 2021, cheap credit and a presale boom had pushed HCMC and Hanoi prices up fast. Then in 2022 the corporate-bond market — the lifeblood of Vietnamese developers — froze after the **Tân Hoàng Minh** (April 2022) and **Vạn Thịnh Phát / SCB** (October 2022) arrests, and mortgage rates climbed as the SBV tightened. Teaser loans began resetting into a 12-14% world, and developers couldn't roll their bonds. Transactions seized; primary projects stalled mid-construction. The SBV responded by cutting policy rates *four times* across 2023 and issuing Decree 08 to let developers restructure their bonds — a textbook attempt to reverse the rate shock and re-open the affordability channel. The recovery was slow and uneven, exactly as the transmission lag predicts. The rate was the master variable on the way down (2022 squeeze) and the master tool on the way back up (2023 cuts).

### The 1981 contrast: high nominal, lower real

It's worth holding 1981 next to 2023 to drive home the real-rate point. In 1981 the US 30-year mortgage rate hit **~18%** — far above 2023's 7.79%. Yet 1981 housing, while weak, did not see anything like the affordability collapse of 2022-23 in *real* terms, because inflation was running 10-13% and wages were inflating fast: the *real* rate was lower, and a few years of high inflation shrank those fixed payments dramatically relative to rising incomes. 2023's 7% against ~3% inflation was a *higher real burden* than 1981's 18% against ~12%. The nominal number screams; the real number is what actually squeezes. This is why a country with structurally higher inflation (and faster nominal wage growth), like Vietnam, can sustain mortgage rates that would look catastrophic in a low-inflation economy.

### Why the boom and the freeze were global at once

One striking feature of both episodes is how *synchronized* they were across countries. The US, Canada, Australia, New Zealand, the UK, Sweden, South Korea, and much of the eurozone all boomed together in 2020-21 and froze together in 2022-23. That synchronization is the affordability mechanism revealing itself on a global scale: most major central banks cut to near-zero at the same time in 2020 and hiked aggressively at the same time in 2022, so the rate dial turned the same way everywhere at once, and every leveraged housing market repriced in step. It wasn't a coincidence of local stories; it was one global rate cycle running through one global affordability machine. The markets that fell hardest in 2022-23 (Sweden, New Zealand, Canada) were precisely those with the most floating-rate mortgages — where the rate shock hit *existing* owners' payments immediately, not just new buyers' budgets, exactly as in Vietnam. The markets that held up best (the US) were those with long fixed-rate loans and the lock-in effect. Same shock, different loan structures, different outcomes — the single most useful lesson in this whole post.

### Singapore and Hong Kong: the same mechanism, different valves

In rate-sensitive Asian financial hubs, governments don't just watch the mechanism — they intervene in it. When ultra-low global rates threatened to inflate property to dangerous levels, Singapore and Hong Kong layered on *macroprudential* tools: higher stamp duties for additional properties, tighter loan-to-value limits, and Total Debt Servicing Ratio caps that explicitly limit how much of the affordability boost from low rates a buyer can use. These are admissions of exactly the thesis of this post — that low rates pump prices through buying power — and deliberate attempts to throttle the conversion. They don't repeal the master variable; they put a regulator on it.

## When this matters / Further reading

The interest rate touches your life through housing whether you own, rent, or are trying to buy. If you're **buying**, the rate environment is the difference between a fixed payment reaching a generous house or a cramped one — and if you're in a teaser-then-reset market like Vietnam, the rate you'll *actually* pay is the reset rate, not the headline teaser, so budget for it before you sign. If you **own**, the rate that you locked is now an asset or a liability: a low fixed rate is a golden handcuff worth keeping; a floating loan is a risk to hedge or pay down. If you're trying to **time** anything, remember the lag — the rate moves quarters before the price index does, so the mortgage rate and mortgage applications tell you where prices are heading long before the headline number confirms it. And whatever the headline rate, do the real-rate arithmetic: subtract inflation, because that's the number that says whether the debt is truly cheap.

The deepest takeaway is the one in the title. Of all the forces that move property — demographics, supply, zoning, taxes, sentiment — interest rates are the one that moves it *fastest and hardest in the short run*, because they act directly on the affordability machine that nearly every buyer runs on. Master the payment-to-price conversion and you understand most of what makes housing boom and freeze.

To go deeper on the adjacent mechanisms:

- [Leverage and the mortgage: how debt amplifies property](/blog/trading/real-estate/leverage-and-the-mortgage-how-debt-amplifies-property) — how the borrowed share turns a price move into a much bigger equity move, both ways.
- [The credit cycle and property prices](/blog/trading/real-estate/the-credit-cycle-and-property-prices) — how the *availability* of credit (not just its price) expands and contracts to drive the property cycle.
- [Income, affordability, and the price-to-income ratio](/blog/trading/real-estate/income-affordability-and-the-price-to-income-ratio) — the other half of affordability: why HCMC and Hanoi rank among the world's least affordable cities even before you add the rate.
- [Real vs nominal: inflation, real yields, the master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — the full case for why the real rate, not the nominal one, is what drives every asset market.

*This article is educational, not financial advice. Rates, prices, and policy are as of June 2026 and go stale quickly; check current figures before acting on any of them.*
