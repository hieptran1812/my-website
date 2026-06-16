---
title: "How Monetary Policy Moves Real Estate and Credit: Mortgages, Cap Rates, and Spreads"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A beginner-friendly deep dive into the three precise channels through which central-bank policy reaches property and credit — the mortgage rate that tracks the 10-year, the cap rate that values commercial buildings, and the credit spread that prices default risk — with the worked dollar math and the trader's playbook for each."
tags: ["macro", "monetary-policy", "real-estate", "mortgage-rates", "cap-rates", "credit-spreads", "high-yield", "commercial-real-estate", "interest-rates", "discount-rate", "default-risk", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Policy reaches property and credit through three precise channels, and once you can name each one you can read the whole real-estate-and-credit complex from a handful of prints: the mortgage rate (which tracks the 10-year yield), the cap rate (a discount rate on rents), and the credit spread (the extra yield over Treasuries for default risk).
>
> - **Residential real estate runs on the mortgage rate.** The 30-year fixed tracks the 10-year Treasury yield, and when policy pushed that yield up the mortgage rate went from a record-low **2.65%** to **7.79%** — which roughly **doubled** the monthly payment on the same loan and froze the US housing market.
> - **Commercial real estate is valued off cap rates,** which are nothing more than a discount rate applied to a building's rents. When the discount rate rose from about 4.5% to 5.9%, the same income stream was suddenly worth far less — a property throwing off \$100,000 of net operating income went from \$2,222,222 to \$1,694,915, a **24% loss in value with zero change in the rent**.
> - **Corporate credit prices off spreads,** the extra yield a borrower pays over Treasuries to compensate for default risk. Tighter policy widens spreads — high-yield spreads blew out to **5.69%** in 2022 — which raises every company's real borrowing cost at exactly the moment refinancing matters most.
> - The one idea to remember: **a higher policy rate is a higher discount rate, and almost everything you can own is a stream of future cash flows being discounted.** Raise the discount rate and the present value of those cash flows — a house's affordability, a building's price, a bond's spread cushion — falls.

In January 2021, the average 30-year fixed mortgage rate in the United States hit **2.65%** — the lowest it had ever been in the history of the Freddie Mac survey, which goes back to 1971. A family buying a \$500,000 house with \$100,000 down was financing \$400,000 at a rate that, after inflation, was nearly free money. Refinancing was a national pastime. Homebuilders could not pour foundations fast enough. The largest asset on most American household balance sheets — the home — was being bought with the cheapest long-term credit in living memory.

Less than three years later, in October 2023, that same 30-year fixed mortgage rate touched **7.79%**, the highest since the year 2000. The monthly principal-and-interest payment on that identical \$400,000 loan went from roughly \$1,612 to roughly \$2,877 — it nearly **doubled** — without the house getting any bigger, any nicer, or any closer to a good school. And then the market did the only thing it could: it froze. Existing-home sales collapsed to levels not seen in nearly thirty years. Nobody with a 3% mortgage wanted to sell and re-buy at 7%. Inventory dried up, transactions cratered, and an entire industry of agents, lenders, and movers went quiet — not because the economy crashed, but because one number, the price of long-term credit, had moved.

That is the entire subject of this post, told through three channels. The mortgage rate is the wire that runs from policy to housing. A second wire — the **cap rate** — runs from policy to commercial buildings: offices, apartments, warehouses, shopping centers, all of which are valued the way a bond is valued, by discounting their rents. And a third wire — the **credit spread** — runs from policy to the entire universe of corporate borrowing, pricing the risk that a company simply fails to pay you back. Each wire carries the same current (a change in the discount rate), but it lights up a different market. We will build every one of these from absolute zero — what a mortgage is, why its rate tracks the 10-year, what a cap rate even means, why it is secretly a discount rate, what a spread is and why it widens — and then hand you a concrete playbook for reading and trading all three.

![Tighter policy splits into the mortgage, cap-rate, and credit-spread channels reaching housing, commercial real estate, and credit](/imgs/blogs/how-monetary-policy-moves-real-estate-and-credit-1.png)

## Foundations: mortgages and the 10-year, cap rates, and credit spreads

Before we can trade any of this, we need five ideas built from nothing: what a mortgage actually is and how its monthly payment is computed, why the mortgage rate tracks the **10-year Treasury yield** rather than the Fed's overnight rate, what a **cap rate** is and why it is really a discount rate, what a **credit spread** is and why it compensates for default risk, and how policy moves each one. Everything else in this post is a consequence of these five. If you have read [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable), this is the same master variable showing up in three new places.

### What a mortgage is — and why the payment is what it is

A **mortgage** is a loan to buy real estate, secured by the property itself: if you stop paying, the lender can take the house. The defining feature of the standard American mortgage is that it is a **30-year, fixed-rate, fully amortizing** loan. Let us unpack each word, because each one is load-bearing.

- **30-year** means you have three decades to pay it back. That long term is exactly why the *level* of long-term interest rates matters so much: a small change in the rate, compounded over thirty years, moves the payment a lot.
- **Fixed-rate** means the interest rate is locked at origination and never changes for the life of the loan. This is unusual globally — in most countries mortgages reset every few years — and it is the source of a uniquely American phenomenon we will spend real time on later: the **lock-in effect**.
- **Fully amortizing** means each monthly payment is a blend of interest (the cost of borrowing) and principal (paying down the balance), structured so that the loan reaches exactly zero at the end of the term. Early on, almost all of the payment is interest; late in the loan, almost all of it is principal.

The monthly payment on such a loan is not guesswork — it comes from one formula, the level-payment amortization formula. If you borrow a principal \$P at a monthly interest rate \$r (the annual rate divided by 12) over \$n monthly payments (30 years is 360 payments), the fixed monthly payment \$M is:

\$M = P \times \dfrac{r(1+r)^n}{(1+r)^n - 1}

This single equation is the hinge of the entire residential-real-estate channel. The borrower does not experience the interest *rate* directly — they experience the monthly *payment*, which is what their budget actually has to cover. And as we will see in a moment, that payment is brutally sensitive to the rate.

### Why the mortgage rate tracks the 10-year Treasury, not the Fed

Here is the single most common beginner confusion in all of real estate: people assume the Federal Reserve "sets" mortgage rates. It does not. The Fed sets one overnight rate, the **federal funds rate** — the rate banks charge each other to borrow reserves for a single night. A 30-year mortgage is the polar opposite of an overnight loan; it is the longest-duration consumer credit that exists. So why would its price be set by the shortest rate?

It is not. The 30-year mortgage rate tracks the **10-year US Treasury yield** — the rate the US government pays to borrow for ten years. Two reasons:

1. **Maturity matching.** Although a mortgage is nominally a 30-year loan, the average mortgage is paid off (through a sale or a refinance) in well under ten years. So the *effective* life of a mortgage is close to a decade, which makes the 10-year Treasury the natural benchmark — the closest risk-free rate of the same approximate duration.
2. **It is the benchmark for all long-term US dollar borrowing.** The 10-year Treasury yield is the risk-free anchor for the whole long end of the curve. Everything riskier than the US government — a homeowner, a corporation, a city — borrows at the 10-year plus a spread for its extra risk. The mortgage rate is, basically, the 10-year yield plus a roughly 1.5-to-3 percentage-point spread that covers the lender's risk and the cost of bundling and servicing the loan.

So the causal chain for housing is: **Fed policy and inflation expectations move the 10-year Treasury yield → the mortgage rate moves with it (10-year plus a spread) → the monthly payment on a given loan moves → affordability and housing demand move.** The Fed influences the 10-year (by setting the path of short rates and shaping inflation expectations), but it does not set it, which is why mortgage rates sometimes *rise* even as the Fed *cuts* — if the bond market decides long-term inflation risk is higher, the 10-year climbs regardless of the overnight rate. For the full mechanics of how a policy change propagates out to the long end and to consumer credit, see [the monetary policy transmission mechanism](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets), and for why the slope of the curve matters, [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).

There is a subtlety worth pinning down, because it explains why the mortgage rate sometimes moves *more* than the 10-year. The spread between the two — call it the **primary mortgage spread** — is not constant. It is normally about 1.7 percentage points, but it widened to nearly 3 percentage points in 2022-23. Why? The spread compensates the lender for two risks beyond the government's credit: **prepayment risk** (when rates fall, borrowers refinance and hand the loan back early, so the lender loses the high-coupon asset just when it is most valuable) and **liquidity/servicing costs**. When rate *volatility* is high — as it was in 2022, when nobody knew how high the Fed would go — prepayment risk is harder to price, so lenders demand a fatter spread. The result is a vicious double-whammy for borrowers: the 10-year rose *and* the spread over it widened, so the mortgage rate climbed even faster than the Treasury yield alone. When you watch mortgage rates, watch the spread over the 10-year, not just the absolute rate — a widening spread is a sign of stress in the mortgage-financing machine itself, not just in rates.

#### Worked example: decomposing the mortgage rate into the 10-year plus a spread

Take two snapshots and split the mortgage rate into its two parts — the risk-free 10-year base and the spread the lender charges on top.

- **January 2021:** the 10-year Treasury yielded about 1.1% and the 30-year mortgage rate was **2.65%**. The primary mortgage spread was therefore about 2.65% − 1.1% = **1.55 percentage points** — a normal, healthy spread in a calm, low-volatility market.
- **October 2023:** the 10-year had climbed to about 4.9% and the mortgage rate hit its peak of **7.79%**. The spread had widened to about 7.79% − 4.9% = **2.89 percentage points** — nearly double the 2021 spread, because rate volatility had spiked and prepayment risk was hard to price.
- **The decomposition:** of the 5.14-percentage-point rise in the mortgage rate (2.65% to 7.79%), about 3.8 points came from the higher 10-year and about 1.3 points came from the *wider spread*. Roughly a quarter of the mortgage shock was the financing machine charging more, not the Treasury yield rising.

The intuition: the mortgage rate is the 10-year plus a spread, and in a tightening cycle both terms rise — so mortgage borrowers get hit harder than the headline 10-year move suggests.

### What a cap rate is — and why it is a discount rate

Now to commercial real estate, which is valued completely differently from a house. A house is priced by comparison: what did similar houses nearby sell for? A commercial building — an apartment complex, an office tower, a warehouse — is priced like an investment, by the income it produces. The key number is the **capitalization rate**, universally shortened to **cap rate**.

The definition is disarmingly simple. A property's **net operating income (NOI)** is the annual rent it collects minus the annual operating costs to run it (property taxes, insurance, maintenance, management) — but *before* any mortgage payments or income taxes. The cap rate is:

\$\text{Cap rate} = \dfrac{\text{Net operating income}}{\text{Property price}}

In words: the cap rate is the **annual yield** the property throws off if you bought it for cash. A building that produces \$100,000 of NOI per year and sells for \$2,000,000 has a cap rate of \$100,000 / \$2,000,000 = 5%. It is the unlevered return on the property — the same idea as a bond's yield, just for a building instead of a bond.

Now rearrange that same equation to solve for price instead of yield:

\$\text{Property price} = \dfrac{\text{Net operating income}}{\text{Cap rate}}

Look at what this says. The value of a commercial building is its income **divided by** the cap rate. And that structure — a stream of income divided by a rate to get a present value — is exactly the structure of a discounted-cash-flow valuation. The cap rate *is* the discount rate the market is applying to the building's rents. A low cap rate means the market is willing to pay a high price for each dollar of income (a high valuation, like a low bond yield meaning a high bond price). A high cap rate means the market demands more income per dollar paid (a low valuation). The cap rate and the property's price move in **opposite directions**, for exactly the same reason a bond's yield and price move in opposite directions: they are two sides of the same discounting equation.

![Same net operating income discounted at a low versus a high cap rate produces a high versus a low property value](/imgs/blogs/how-monetary-policy-moves-real-estate-and-credit-4.png)

This is the crux of the commercial channel, so sit with it. When policy tightens and the risk-free 10-year yield rises, every investor's required return rises with it — why accept a 4.5% cap rate on a risky building when a riskless Treasury now pays 4.5%? Investors demand a higher cap rate, which mechanically means a lower price, even if the building's rent did not change by a single dollar. The discount rate rose, so the present value of the rents fell. We will put hard numbers on this shortly.

There is one more layer worth making explicit, because it is the bridge between policy and the cap rate. A cap rate can be decomposed the same way a mortgage rate can: it is the **risk-free 10-year yield plus a property risk premium** (and, for a growing income stream, minus expected rent growth). In rough form, cap rate ≈ 10-year yield + property risk premium − expected NOI growth. The property risk premium compensates the buyer for everything riskier about a building than a Treasury: tenants can default, leases roll over, buildings need capital, and real estate is illiquid. In the cheap-money era of 2021, the 10-year was under 1% and the risk premium was compressed by yield-starved capital chasing any return, so cap rates fell to record lows. As policy tightened and the 10-year ran toward 5%, the risk-free base of every cap rate rose almost mechanically — and that is the direct line from a Fed decision to the price of an office building. The cap rate is not set in a real-estate vacuum; it is anchored to the same 10-year Treasury that anchors the mortgage rate, which is why both channels fire together when policy moves.

### What a credit spread is — and why it widens

The third channel is corporate credit. When a company borrows by issuing a **bond**, it promises to pay periodic interest (coupons) and return your principal at maturity. But unlike the US government, a company can go bankrupt and fail to pay. To compensate you for that **default risk**, a corporate bond must offer a higher yield than a Treasury bond of the same maturity. That extra yield is the **credit spread**.

\$\text{Corporate bond yield} = \text{Treasury yield (same maturity)} + \text{Credit spread}

The spread is the price of default risk, quoted in percentage points or basis points (one basis point = 0.01%). The market sorts corporate borrowers into two broad buckets by credit quality, which is the single most important distinction in credit:

- **Investment grade (IG)** — strong, financially sound companies (think a blue-chip industrial or a major bank), rated BBB− or higher. Their spreads are thin, typically around 1% over Treasuries, because default is unlikely.
- **High yield (HY)**, also called "junk" — weaker, more leveraged companies (think a heavily indebted retailer or a speculative startup), rated BB+ or lower. Their spreads are much wider, often 3% to 6% or more, because default is a real possibility.

The technical measure professionals actually watch is the **option-adjusted spread (OAS)** — the spread over Treasuries after adjusting for any embedded options in the bond, like the issuer's right to call it early. When you hear "high-yield OAS is at 350," it means high-yield bonds yield 3.50 percentage points more than equivalent Treasuries. The OAS is the market's real-time, dollar-weighted vote on how scared it is about corporate defaults.

Why does a spread exist at all, in dollar terms? A useful approximation: the spread roughly equals the **probability of default × the loss given default** (the fraction you fail to recover if the borrower goes under). If a class of bonds has a 4% annual chance of defaulting and you would recover only 40% of your money in a default (a 60% loss), the spread you should demand is about 4% × 60% = 2.4 percentage points, plus a premium for bearing the risk and for illiquidity. This decomposition is why spreads are so sensitive to the economic outlook: both the default probability *and* the expected recovery worsen when a recession looms, so the required spread rises on two fronts at once.

Why does tighter policy widen spreads, then? Two linked reasons. First, **higher rates raise the probability of default directly**: a company that borrowed cheaply now faces a much higher cost to refinance its debt, and some companies that were viable at 3% are not viable at 8%. A leveraged firm with debt coming due is suddenly looking at a refinancing cost that may exceed its operating cash flow — the textbook path to default. Second, **tighter policy slows the economy**, which shrinks corporate revenues and profits and lowers recovery values (assets are worth less in a downturn), which makes both the default probability and the loss-given-default worse across the board. So as policy tightens, the market demands a wider spread to hold corporate credit — which raises the actual borrowing cost for every company, deepening the slowdown. The effect is concentrated at the bottom: an investment-grade firm with a fortress balance sheet barely notices, while a junk-rated firm living on cheap refinancing is in real trouble. That is why the high-yield line moves so much more than the investment-grade line. For how this connects to the broader plumbing of credit creation and funding markets, see [shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market).

### How policy moves each one — the unifying idea

Step back and notice that all three channels are the *same mechanism wearing three costumes*. A higher policy rate, working through the 10-year Treasury and through expectations, raises the **discount rate** the whole market applies to future cash flows:

- For a **house**, the higher discount rate shows up as a higher mortgage rate, which raises the monthly payment and prices out marginal buyers — demand falls.
- For a **commercial building**, the higher discount rate shows up as a higher cap rate, which directly lowers the price the same rents are worth — value falls.
- For a **corporate bond**, the higher discount rate shows up as both a higher risk-free base *and* a wider credit spread, which raises the borrower's cost and lowers the bond's price — credit tightens.

One lever, one underlying mechanism — the discount rate — three markets. Now we walk down each wire in turn, with the dollar math.

## The mortgage channel: the rate goes up, the payment goes up, demand collapses

The residential channel is the most visible to ordinary people because almost everyone touches it. Let us trace it precisely, from the rate to the freeze.

### The path of the rate

The chart below shows the 30-year fixed mortgage rate from its record low through its peak. Notice the shape: a near-vertical climb through 2022 and into 2023, from **2.65%** to **7.79%** — the steepest, fastest mortgage-rate increase in modern history. This was not a gentle drift; it was the bond market repricing the entire path of Fed policy as inflation ran to a 40-year high, dragging the 10-year Treasury up and the mortgage rate with it.

![Line chart of the 30-year fixed mortgage rate rising from a record low of 2.65 percent to a peak of 7.79 percent](/imgs/blogs/how-monetary-policy-moves-real-estate-and-credit-2.png)

The rate roughly **tripled** in under three years. To feel why that froze the market, we have to convert the rate into the thing buyers actually face: the monthly payment.

### From rate to payment — the brutal nonlinearity

A buyer does not shop for a rate; they shop for a payment they can afford. The amortization formula turns the rate into that payment, and the relationship is sharply nonlinear — each percentage point of rate adds *more* to the payment than the last, because of how compounding works over 360 months. Here is the core calculation as a short, runnable function, so you can see exactly where the numbers come from:

```
def monthly_payment(principal, annual_rate_pct, years=30):
    r = annual_rate_pct / 100 / 12      // monthly rate
    n = years * 12                       // total number of payments
    if r == 0:
        return principal / n
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)

loan = 400_000
for rate in (2.65, 4.0, 5.52, 6.42, 6.85, 7.79):
    pay = monthly_payment(loan, rate)
    print(f"{rate:>5}%  ->  ${pay:,.0f} per month")
```

Run it on a \$400,000 loan and you get the bar chart below: \$1,612 a month at 2.65%, climbing to \$2,877 a month at 7.79%. Same house, same loan size, same buyer — the only thing that changed is the price of credit, and the payment went up by more than \$1,200 a month, which is \$15,000 a year of after-tax income that has to come from somewhere.

![Bar chart of the monthly payment on a 400,000 dollar loan rising with the mortgage rate from 1,612 to 2,877 dollars](/imgs/blogs/how-monetary-policy-moves-real-estate-and-credit-3.png)

#### Worked example: the mortgage payment shock

Take a buyer financing exactly \$400,000 over 30 years.

- **At the record-low rate of 2.65%:** the monthly rate is 0.0265 / 12 = 0.0022083, over 360 payments. Plug into the formula and the principal-and-interest payment is **\$1,612 per month**. Over the life of the loan, total payments are about \$580,000 — meaning roughly \$180,000 of interest on the \$400,000 borrowed.
- **At the peak rate of 7.79%:** the monthly rate is 0.0779 / 12 = 0.0064917. The same \$400,000 over the same 360 months now costs **\$2,877 per month** — an increase of \$1,265 a month, or about **78%** more. Over the life of the loan, total payments balloon to about \$1,036,000, of which roughly **\$636,000 is interest** — more than 1.5 times the amount borrowed.
- **What it means for who can buy:** lenders typically cap the housing payment at about 28% of gross income. A \$1,612 payment is affordable on roughly \$69,000 of income; the \$2,877 payment requires roughly \$123,000. The same house just demanded a buyer earning nearly twice as much.

The intuition: the buyer never sees the rate, only the payment, and a tripling of the rate nearly doubled the payment — which is what actually priced people out.

### The lock-in effect: why prices did not crash

Here is where housing surprises people. With payments doubling and affordability cratering, you would expect home *prices* to crash. They did not. Nationally, prices dipped only modestly and then resumed climbing. The reason is the **lock-in effect**, and it is a direct consequence of that uniquely American fixed-rate mortgage.

Consider a homeowner who locked a 30-year fixed mortgage at 3% in 2021. Their payment is fixed for thirty years; the rate increase does not touch them. But now they want to move — a bigger house, a new city, a job. To move, they must sell their current home (ending the 3% loan) and finance a new one at the prevailing 7% rate. Even buying an *identically priced* house, their monthly payment would jump enormously. So they do the rational thing: **they do not move.** They stay put, keeping their cheap loan.

Multiply that decision across millions of households and you get the freeze. The lock-in effect chokes off the **supply** of existing homes for sale at the same time high rates choke off **demand**. With both supply and demand collapsing together, transaction *volume* craters — but prices, which are set at the margin by the few transactions that do happen, stay sticky. The market did not crash; it seized.

This is also why housing is the *slowest* of the three channels to transmit, and why a trader cannot expect a clean, fast response. The mortgage shock works through the real economy with a long lag: most homeowners have a fixed rate, so they feel nothing immediately; only new buyers and movers are exposed, and they simply step back rather than transact. The drag shows up gradually — in collapsing transaction volume, in homebuilders slowing starts, in the army of agents and lenders and movers and furniture sellers who depend on the churn of home sales seeing their incomes dry up. That is the second-order effect of the freeze: an entire ecosystem of housing-related activity goes quiet even though home *prices* on paper look fine. The headline "home prices are stable" hides a real economic contraction in housing *activity*. For a macro trader, the lesson is that the housing channel is a slow burn, not a spark — the position is in the rate and in the rate-sensitive equities, not in a bet that house prices will crash next quarter.

There is a third-order effect worth naming: the **wealth effect**. A home is the largest store of wealth for most households, and when home values feel frozen or falling, owners feel less wealthy and spend less — which is itself a drag on the economy that the Fed *wants* when it is fighting inflation. The housing channel is therefore doing real work on aggregate demand even when prices are sticky, by freezing the wealth-effect spending and the activity ecosystem. The freeze is the transmission, not a failure of it.

#### Worked example: the lock-in trap

A homeowner has a \$300,000 mortgage balance at a 3% fixed rate. Their principal-and-interest payment is **\$1,265 per month** (run the formula: \$300,000 at 3% over 30 years). They want to move to a different house of the same \$300,000 price, financing the same \$300,000.

- **The new loan at 7%** costs **\$1,996 per month** — \$731 more every month, or about \$8,800 more a year, for the *exact same-priced house*.
- The only thing they would be buying with that extra \$731 a month is the privilege of moving. For most households, that is not worth it, so they stay — and their house never comes to market.
- The aggregate effect: the homeowner is "locked in" to their home by their own cheap mortgage. The lower their old rate relative to today's, the tighter the lock.

The intuition: a fixed-rate mortgage is an asset to the borrower when rates rise, and walking away from a 3% loan to take a 7% one is a cost most people refuse to pay — which is why supply froze and prices held.

## The cap-rate channel: commercial property revalued by the discount rate

Now to commercial real estate (CRE), where the lock-in cushion does not exist. A commercial building has no sentimental owner with a cheap fixed mortgage refusing to sell; it is an investment asset, marked to a required yield, and when that required yield rises the value falls — directly, mechanically, immediately.

### Cap rates compressed, then repriced

The chart below shows the all-property average US cap rate. Watch the path: cap rates **compressed** to a record-low 4.6% in 2021, when policy was at the zero bound and capital was hunting for any yield it could find, bidding building prices up. Then, as policy tightened and the risk-free 10-year yield climbed toward 5%, cap rates **repriced upward** — to 5.5% in 2023 and 5.9% in 2024. Each tick higher in the cap rate is a tick lower in the value of every building's income.

![Line chart of US commercial cap rates falling to 4.6 percent in 2021 then rising to 5.9 percent by 2024](/imgs/blogs/how-monetary-policy-moves-real-estate-and-credit-5.png)

The move looks small — 4.6% to 5.9% is only 1.3 percentage points. But because price is income *divided by* the cap rate, a small change in the denominator is a large change in the value. That is the whole danger of CRE in a tightening cycle: a modest rise in the discount rate produces an outsized drop in price.

#### Worked example: the cap-rate repricing

Take a stabilized commercial property — say a suburban office or apartment building — that produces exactly \$100,000 of net operating income per year. The rent does not change in this example; only the market's required yield does.

- **At a 4.5% cap rate (cheap-money era):** value = \$100,000 / 0.045 = **\$2,222,222**. The market is paying about 22 times the annual income for the building.
- **At a 5.9% cap rate (after tightening):** value = \$100,000 / 0.059 = **\$1,694,915**. The market now pays about 17 times the income.
- **The loss:** \$2,222,222 − \$1,694,915 = **\$527,307**, a drop of about **24%** — and the building's rent never changed by a single dollar. The discount rate rose, so the present value of the identical rents fell.

The intuition: commercial property is a bond made of bricks, and just as a bond's price falls when its yield rises, a building's price falls when its cap rate rises — no change in the rent required.

### Why CRE has no lock-in cushion — and the refinancing wall

Two features make CRE far more dangerous than housing in a tightening cycle. First, there is **no lock-in effect**: commercial buildings are owned by funds, REITs, and partnerships that mark their portfolios to market and must report values to investors. There is no emotional refusal to sell at a loss when a pension fund needs liquidity. Second, and more dangerously, commercial mortgages are usually **short-term and floating or balloon** — a typical CRE loan is 5-to-10 years, after which the entire balance comes due and must be **refinanced**.

That refinancing requirement is the time bomb. A building bought in 2021 at a 4.5% cap rate with a cheap 3.5% loan now has to refinance in, say, 2026 — but the building is worth less (the cap rate rose to 5.9%), *and* the new loan costs far more (rates roughly doubled), *and* the lender will only lend against the lower value. The owner faces a "**refinancing wall**": they may have to inject fresh equity just to refinance, and if they cannot, the building defaults. This is precisely the slow-motion stress that hit US office buildings from 2023 onward — not because rents collapsed, but because the discount rate rose and the loans came due into a higher-rate, lower-value world.

#### Worked example: the CRE refinancing wall

An investor bought a building in 2021 for \$2,222,222 (a 4.5% cap rate on \$100,000 of NOI), financing it with a 65% loan-to-value mortgage: a \$1,444,444 loan, with \$777,778 of their own equity.

- **In 2026 the loan comes due** and must be refinanced. But the cap rate has risen to 5.9%, so the building is now worth only \$1,694,915.
- **The lender's 65% loan-to-value limit** now caps the new loan at 0.65 × \$1,694,915 = **\$1,101,695**.
- **But the old loan balance is about \$1,444,444** (interest-only, so little was paid down). To refinance, the owner must come up with \$1,444,444 − \$1,101,695 = **\$342,749 of fresh cash** — on top of the higher new interest rate. If they do not have it, the building goes back to the lender.

The intuition: in CRE, a rising discount rate does not just lower the paper value — it triggers a real cash crisis when the short-term loan rolls over against a lower appraisal.

### Not all buildings are equal: cap-rate dispersion by type

A single "all-property average" cap rate hides enormous dispersion, and the dispersion is itself a signal. Different property types carry different risk premiums, so they sit at different cap rates and reprice by different amounts when the discount rate rises. Roughly:

- **Multifamily apartments and industrial/logistics** carry the *lowest* cap rates (the highest valuations per dollar of income), because their income is seen as the most durable — people always need housing, and e-commerce keeps warehouses full. These traded in the 4-5% range at the peak.
- **Offices** carry *higher* cap rates because their income is riskier, and after 2020 that risk exploded as remote work hollowed out demand. Office cap rates blew out the most, sometimes to 8-10% or higher for troubled assets, on top of falling NOI.
- **Retail** sits in between, with wide variation between thriving grocery-anchored centers and dying malls.

The point for a trader is that the cap-rate channel does not hit CRE uniformly — it concentrates in the riskiest property types, where the discount-rate rise compounds with a deterioration in the income itself. When you short or underweight CRE in a tightening cycle, the cleanest expression is the *most discount-rate-sensitive and income-impaired* segment (offices) rather than the most durable (apartments and industrial). The all-property average is a blunt instrument; the dispersion underneath it is where the real risk and the real opportunity live.

## The credit-spread channel: tighter policy widens spreads, raises borrowing costs

The third wire runs to the entire corporate-borrowing universe. Here the discount rate shows up not just as a higher risk-free base, but as a **widening spread** — the market demanding more compensation for the risk that companies fail to pay.

### Spreads widen as policy tightens

The chart below shows two spreads through the 2022 tightening: investment-grade OAS (the blue line, thin and well-behaved) and high-yield OAS (the red line, wide and volatile). Watch the high-yield line: as the Fed hiked aggressively through 2022, HY spreads blew out from about 3% to a peak of **5.69%** in mid-2022. Investment-grade spreads widened too, but far less — from under 1% to about 1.6% — because strong companies are not at meaningful risk of default even when money tightens.

![Dual line chart of investment-grade and high-yield credit spreads with high-yield peaking at 5.69 percent in 2022](/imgs/blogs/how-monetary-policy-moves-real-estate-and-credit-6.png)

The shape tells the story. When policy tightens and the economy is expected to slow, the *weakest* borrowers are repriced the hardest — the high-yield line moves a lot, the investment-grade line moves a little. The gap between the two lines is itself a fear gauge: when HY spreads widen far more than IG spreads, the market is pricing rising default risk concentrated in the most leveraged companies. When both compress together, the market is sanguine. For how this fits the broader credit cycle, see [the monetary policy transmission mechanism](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets).

#### Worked example: the spread blowout

You own \$1,000,000 face value of a high-yield bond portfolio, with an average duration of about 4 years. In early 2022, before the tightening, the high-yield spread was about 3.0%. By mid-2022 it had widened to its peak of **5.69%** — a widening of 2.69 percentage points.

- **The yield you require rose by 2.69%** (from the spread widening) on top of any rise in the underlying Treasury yield. Bond prices fall when required yields rise, by approximately the duration times the yield change.
- **The price hit from the spread widening alone** is roughly −(duration × spread change) = −(4 × 2.69%) ≈ **−10.8%**. On your \$1,000,000 holding, that is a paper loss of about **\$108,000** — and that is *before* adding the loss from the rise in the risk-free Treasury yield over the same period, which pushed the total high-yield drawdown to roughly −11% for 2022.
- **For the borrowers themselves**, the same widening means a company refinancing \$1,000,000 of debt now pays an extra 2.69% in interest — about **\$26,900 more per year** — purely from the spread, at exactly the moment its business is slowing.

The intuition: a credit spread is the market's price of default risk, and when tighter policy raises that risk, both bondholders (through price losses) and borrowers (through higher refinancing costs) pay for it at once.

#### Worked example: the same loan, investment grade versus high yield

Two companies each need to refinance \$10,000,000 of debt in mid-2022, when the 10-year Treasury yielded about 3%. The only difference is their credit quality and therefore their spread.

- **The investment-grade firm** borrows at the 10-year plus its IG spread of about 1.6%: a total yield of about 3% + 1.6% = **4.6%**, costing about **\$460,000 a year** in interest on the \$10,000,000.
- **The high-yield firm** borrows at the 10-year plus the peak HY spread of 5.69%: a total yield of about 3% + 5.69% = **8.69%**, costing about **\$869,000 a year** — nearly double the IG firm's cost for the identical \$10,000,000 of debt.
- **The spread is the whole difference:** \$869,000 − \$460,000 = **\$409,000 a year**, paid purely because the market judges the second firm more likely to default. When policy tightens and that HY spread widens further, the gap grows — and the weak firm, already paying the most, is squeezed hardest exactly when its business is slowing.

The intuition: the credit spread is a tax on weakness that rises in a downturn, which is why tighter policy hits the most leveraged borrowers first and hardest.

### The doom loop: why spreads matter to everyone, not just bond traders

The credit-spread channel has a feedback property that makes it more dangerous than it looks. Wider spreads raise borrowing costs, which weakens companies, which raises default risk, which widens spreads further — a doom loop that can feed on itself in a stress event. And because corporate credit funds the real economy (companies borrow to invest, hire, and roll over existing debt), a spread blowout is not a sideshow for bond desks; it is a tightening of financial conditions for the entire economy. When high-yield spreads spike, equity markets almost always fall with them, because the same default risk that hurts bondholders threatens the equity beneath those bonds. Spreads are one of the cleanest real-time reads on financial stress that exists — which is exactly why they belong in the playbook.

### Why credit leads equities

There is a structural reason credit spreads tend to lead equities at turning points, and it is worth understanding because it is the basis for one of the most reliable cross-asset signals there is. A company's capital structure is a layer cake: bondholders sit *senior* to shareholders, meaning if the firm hits trouble, the bondholders get paid first and the shareholders get whatever is left, which in a default is often nothing. So bondholders are, in a sense, the more conservative and better-informed analysts of a company's survival — their entire return is about *not* getting wiped out, so they price deterioration early. Equity holders, by contrast, own the upside lottery ticket and are prone to hope. When the people whose job is to worry about getting paid back start demanding a wider spread, it is an early warning that the equity below them is in danger.

Empirically, high-yield spreads often begin widening *before* the equity market rolls over, and — just as importantly — often begin *tightening* before equities bottom, because credit prices the all-clear before the headlines turn. This is why a stock trader who ignores spreads is flying with one instrument missing: the bond market is voting on the same companies with more money and more conservatism, and that vote leads. Keep the high-yield OAS on the same screen as the index you trade, and treat a fast divergence — credit deteriorating while equities hold up — as a warning that the slower, more hopeful market has not caught up yet.

## Common misconceptions

Three myths cause more bad real-estate-and-credit trades than any others. Each one is corrected by a number you have already seen.

### "Housing always falls when rates rise"

It feels obvious: rates double, affordability craters, so prices must crash. But in 2022-23 the mortgage rate tripled to 7.79% and national home prices barely dipped before resuming their climb. The reason is the **lock-in effect**: the fixed-rate mortgage made millions of owners unwilling to sell their cheap loans, which collapsed *supply* at the same time high rates collapsed *demand*. With both sides frozen, transaction *volume* crashed — existing-home sales fell to roughly 30-year lows — but *prices*, set at the margin, stayed sticky. The correct mental model is: rising rates first freeze housing *volume*; prices only fall later, and only if forced selling (job losses, distress) overwhelms the lock-in. Volume is the leading indicator, not price.

### "A cap rate is just a fixed property characteristic"

Beginners often treat the cap rate as a fixed attribute of a building, like its square footage. It is not — it is a **market-determined discount rate** that moves with the risk-free 10-year yield and with risk appetite. The same building had a 4.6% cap rate in 2021 and a 5.9% cap rate in 2024 with no change to the building at all. Treating the cap rate as fixed is the single most expensive error in CRE: it makes you assume a building's value is stable when in fact it is a leveraged bet on where the discount rate goes. A 1.3-percentage-point rise in the cap rate cut the example building's value by 24% — the cap rate is the most important variable in the valuation, and it is anything but fixed.

### "Spreads only matter to bond traders"

If you only trade stocks, it is tempting to ignore credit spreads as a fixed-income curiosity. This is dangerous. The high-yield spread is one of the best leading indicators of equity stress that exists, because it prices the default risk sitting *underneath* the equity. In 2022, the high-yield OAS peaked near 5.69% in the same window that the S&P 500 was deep in a bear market and the Nasdaq fell over 30%. When HY spreads blow out, it is a signal that financial conditions are tightening for the whole economy — and equities are about to feel it, if they have not already. Every cross-asset trader should have a high-yield-spread chart open, regardless of what they trade.

### "Cap rates move one-for-one with interest rates"

A more sophisticated error, common among real-estate analysts, is to assume that a 1-point rise in the 10-year mechanically lifts cap rates by 1 point. It does not, and the gap between the two is itself informative. Recall that the cap rate is the 10-year plus a property risk premium minus expected rent growth. When the 10-year rose by nearly 4 points from 2021 to 2024, the all-property cap rate rose by only about 1.3 points — far less than one-for-one. The cushion came from a *compression of the property risk premium* (investors stayed bullish on real estate and accepted thinner compensation) and from expectations of higher future rent growth in an inflationary environment. The danger is that this cushion can vanish: if sentiment sours and the risk premium normalizes back up *while* the 10-year is also high, cap rates can lurch higher in a delayed, non-linear catch-up, repricing CRE downward in a second leg. So the right model is not "cap rates follow the 10-year one-for-one" but "cap rates follow the 10-year *plus a risk premium that can compress and then snap back*" — and watching the cap-rate-minus-10-year spread tells you how much cushion is left before that snap-back.

## How it shows up in real markets

Three real episodes from the recent cycle show all three channels firing in real time, with the numbers attached.

### The 2022-23 mortgage shock

This is the cleanest example in the dataset. The 30-year fixed mortgage rate ran from 2.65% (January 2021) to 7.79% (October 2023) as the Fed hiked from a 0.25% ceiling to 5.50% and the 10-year Treasury climbed from under 1% to nearly 5%. The monthly payment on a \$400,000 loan went from roughly \$1,612 to roughly \$2,877. The result was not a price crash but a **freeze**: existing-home sales fell to their lowest level in nearly three decades, mortgage-refinance volume essentially evaporated, and the lock-in effect kept inventory off the market. Housing demonstrated the textbook pattern — *volume* collapsed first, *prices* held, exactly as the lock-in effect predicts.

### The CRE repricing

While houses froze, commercial real estate **repriced**. All-property cap rates rose from a record-low 4.6% in 2021 to 5.9% by 2024, mechanically lowering the value of the same income streams by roughly 20-25%. The pain concentrated in **offices**, where the cap-rate repricing combined with a structural collapse in demand (remote work cut occupancy) and a refinancing wall (cheap pandemic-era loans coming due into a high-rate world). Some trophy office towers traded at 50-70% discounts to their 2019 values — a combination of higher cap rates *and* lower NOI. The 2022 sector data captures the equity-market echo: real-estate stocks fell **−26.2%** in 2022, among the worst-performing S&P 500 sectors, as the market priced the discount-rate hit to property.

### The 2022 spread widening

The credit channel fired simultaneously. High-yield OAS widened from about 3% to a peak of **5.69%** in 2022 as the Fed tightened and recession fears mounted, while investment-grade OAS widened more modestly to about 1.6%. This was the credit-market expression of the same rate shock that hit stocks and bonds: 2022 was the year both stocks *and* bonds fell together — long Treasuries lost over 30%, investment-grade corporates lost about 15%, and high-yield bonds lost about 11% — a synchronized rate-shock selloff with no place to hide except cash and commodities. The spread widening was the warning siren; by the time it peaked, the damage to risk assets was already done.

### Reading the three episodes as one event

The three episodes above are not three separate stories — they are one event seen through three windows. A single force, a rising discount rate driven by the fastest Fed tightening in forty years, hit all three channels in the same window of 2022-23. Housing *froze* (volume collapsed, prices stuck via lock-in). Commercial real estate *repriced* (cap rates rose, values fell 20-25%, offices worst of all). Corporate credit *widened* (high-yield OAS to 5.69%, borrowing costs up, the weakest firms squeezed). The reason they moved together is that they are all the same valuation equation — future cash flows divided by a discount rate — and the discount rate is the variable the Fed controls. A trader who understood that they were one event, not three, had the cleanest possible thesis: position for the discount rate, and every channel pays off in the same direction. The mistake was to treat housing, CRE, and credit as separate asset classes with separate drivers; the edge was to see the single lever behind all three.

## How to trade it / The playbook

Everything above resolves into a single, compact playbook: three channels, each with one clean signal to watch, what it means, and the position it supports. This is the payoff — concrete signals you can read off public data.

![Playbook matrix mapping the mortgage, cap-rate, and credit-spread channels to a signal, a meaning, and a position](/imgs/blogs/how-monetary-policy-moves-real-estate-and-credit-7.png)

### Channel 1 — Watch the mortgage rate (and the lock-in gap)

**The signal.** Track the 30-year fixed mortgage rate *against* the 10-year Treasury yield — the spread between them (normally 1.5-3 percentage points) tells you whether mortgage stress is coming from rates or from lender risk aversion. Just as important, track the **lock-in gap**: the difference between the prevailing mortgage rate and the average rate on outstanding mortgages. When that gap is wide (prevailing 7% vs. existing 3.5%), supply is frozen and housing *volume* will stay depressed regardless of price.

**The position.** When the lock-in gap is wide and rates are peaking, do *not* short homebuilders on an affordability thesis — frozen supply props up prices and builders of new homes capture the demand that cannot find existing inventory. Instead, the cleaner expression is on the *rate* itself: long duration (the 10-year) when you think policy is about to ease, which simultaneously relieves the mortgage rate and benefits rate-sensitive housing and REIT equities. When rates begin to *fall*, the lock-in gap narrows, frozen sellers return, and volume — and homebuilder earnings — recover.

**The invalidation.** The lock-in thesis breaks if **forced selling** appears — a spike in unemployment or distressed sales that overwhelms the unwillingness to sell. Watch the unemployment rate and mortgage-delinquency data; if both turn up sharply, the volume-freeze gives way to a genuine price decline, and the "prices stay sticky" view is invalidated.

### Channel 2 — Watch the cap-rate spread

**The signal.** Track the **cap-rate-minus-10-year spread** — the cushion between the yield a building pays and the risk-free rate. When that spread is thin (a 5% cap rate against a 4.5% 10-year offers only 0.5% of compensation for taking property risk), CRE is dangerously overpriced and vulnerable to any further rise in rates. A healthy spread is 2-3 percentage points; a spread near zero is a flashing red light.

**The position.** When the cap-rate spread is thin and rates are still rising, the trade is to be **short or underweight CRE-heavy exposures** — REITs concentrated in office, and especially the **regional banks** that hold concentrated CRE loan books and face the refinancing wall on their balance sheets. When the cap-rate spread is wide (cap rates have repriced up to offer real compensation) and rates are peaking, that is the *re-entry* signal: the discount-rate damage is done and the income yield is finally attractive.

**The invalidation.** The short-CRE thesis is invalidated when the 10-year yield **decisively falls**, because a falling discount rate re-inflates property values directly — the same mechanic in reverse. If the Fed pivots to cutting and the 10-year drops a full point, cap rates compress and CRE values recover, so cover the short.

### Channel 3 — Watch high-yield OAS

**The signal.** Keep a **high-yield OAS** chart open at all times, regardless of what you trade. The level matters (below ~3% is complacent, above ~5% is stressed, above ~8% is crisis), but the *direction and speed* matter more: a fast widening of HY spreads is one of the earliest and cleanest signals of tightening financial conditions and rising recession risk. Watch the **HY-minus-IG gap** too — when junk spreads widen far faster than investment-grade, default fear is concentrating in the weakest borrowers.

**The position.** When HY OAS is widening through 5% with momentum, **reduce risk across the board** — trim equities (especially the leveraged and unprofitable names whose bonds are in the high-yield index), favor quality and cash, and if you trade credit directly, the spread blowout is a tradable short on high-yield credit. When HY OAS has spiked and begins to *compress* (often as the Fed signals a pivot), that is among the most reliable risk-on signals there is — credit leads the all-clear.

**The invalidation.** The risk-off thesis is invalidated when HY spreads **roll over and tighten** even as headlines stay grim — credit is forward-looking and tends to bottom (in spread terms, peak) before the economic news does. If spreads are compressing while the doom-scrolling continues, the worst is being priced out, and the defensive posture should be unwound.

### The one-line synthesis

All three channels are the same trade in three costumes: **a bet on the direction of the discount rate.** When you expect policy and the 10-year to *rise*, lean against property and credit — frozen housing volume, repricing CRE, widening spreads. When you expect the discount rate to *fall*, lean the other way — recovering volume, re-inflating buildings, compressing spreads. Watch the mortgage-rate-versus-10-year gap, the cap-rate-versus-10-year spread, and the high-yield OAS, and you are reading the entire real-estate-and-credit complex off three numbers.

## Further reading & cross-links

- [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — the foundation under all three channels: why a rate is the price of money and the discount rate on everything.
- [The monetary policy transmission mechanism](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets) — how one overnight rate reaches the 10-year, the credit channel, and the asset-price channel, at different speeds.
- [Reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — the 10-year that anchors mortgage rates and cap rates lives on the curve; here is how to read its shape.
- [Shadow banking and the repo market](/blog/trading/finance/shadow-banking-and-the-repo-market) — the plumbing of credit creation that sits beneath corporate spreads and the funding of real estate.
